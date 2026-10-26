use crate::plugin::{PluginDiagnostic, PluginManager, RegisteredTool};
use crate::provider::{InferenceProvider, ProviderFactory};
#[cfg(test)]
use crate::types::Role;
use crate::types::{
    ChatRequest, ChatResponse, Message, ProviderConfig, ProviderKind, SamplingParams,
    ToolDefinition, ToolResult, Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    fn registered_tools(&self) -> Result<(Vec<RegisteredTool>, Vec<PluginDiagnostic>)>;
    async fn execute(&self, tool: &RegisteredTool, arguments: Value) -> Result<ToolResult>;
}

#[async_trait]
impl ToolExecutor for PluginManager {
    fn registered_tools(&self) -> Result<(Vec<RegisteredTool>, Vec<PluginDiagnostic>)> {
        PluginManager::registered_tools(self)
    }

    async fn execute(&self, tool: &RegisteredTool, arguments: Value) -> Result<ToolResult> {
        self.invoke_registered(tool, arguments).await
    }
}

#[derive(Debug, Clone)]
pub struct HarnessOptions {
    pub enable_tools: bool,
    pub max_steps: usize,
    pub sampling: SamplingParams,
}

impl Default for HarnessOptions {
    fn default() -> Self {
        Self {
            enable_tools: false,
            max_steps: 8,
            sampling: SamplingParams::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolExecution {
    pub plugin_id: String,
    pub tool_name: String,
    pub call_id: String,
    pub result: ToolResult,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HarnessStep {
    pub index: usize,
    pub response: ChatResponse,
    pub tool_executions: Vec<ToolExecution>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HarnessReport {
    pub provider: ProviderKind,
    pub model: String,
    pub answer: Message,
    pub messages: Vec<Message>,
    pub steps: Vec<HarnessStep>,
    pub usage: Usage,
    pub diagnostics: Vec<String>,
}

pub struct Harness {
    config: ProviderConfig,
    provider: Arc<dyn InferenceProvider>,
    executor: Arc<dyn ToolExecutor>,
}

impl Harness {
    pub fn new(config: ProviderConfig, plugins: PluginManager) -> Result<Self> {
        let provider = ProviderFactory::build(config.clone())?;
        Self::from_components(config, provider, Arc::new(plugins))
    }

    pub fn from_components(
        config: ProviderConfig,
        provider: Arc<dyn InferenceProvider>,
        executor: Arc<dyn ToolExecutor>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            provider,
            executor,
        })
    }

    /// Runs model requests until the assistant stops calling tools or the
    /// explicit step budget is exhausted. Provider failures abort the turn;
    /// tool failures are durable tool results the model can inspect.
    pub async fn run(
        &self,
        mut messages: Vec<Message>,
        options: HarnessOptions,
    ) -> Result<HarnessReport> {
        if messages.is_empty() {
            return Err(CondenserError::Config(
                "a harness turn needs at least one message".into(),
            ));
        }
        if !(1..=32).contains(&options.max_steps) {
            return Err(CondenserError::Config(
                "max_steps must be between 1 and 32".into(),
            ));
        }
        options.sampling.validate()?;

        let (tools, diagnostics) = if options.enable_tools {
            self.executor.registered_tools()?
        } else {
            (Vec::new(), Vec::new())
        };
        let definitions = tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.qualified_name.clone(),
                description: format!(
                    "{} (from {})",
                    tool.definition.description, tool.plugin_name
                ),
                input_schema: tool.definition.input_schema.clone(),
            })
            .collect::<Vec<_>>();
        let mut diagnostics = diagnostics
            .into_iter()
            .map(|diagnostic| format!("{}: {}", diagnostic.path.display(), diagnostic.message))
            .collect::<Vec<_>>();
        let mut steps = Vec::new();
        let mut usage = Usage::default();

        for index in 0..options.max_steps {
            let response = self
                .provider
                .complete(&ChatRequest {
                    model: self.config.model.clone(),
                    messages: messages.clone(),
                    sampling: options.sampling.clone(),
                    tools: definitions.clone(),
                })
                .await?;
            usage.add_assign(&response.usage);
            messages.push(response.message.clone());
            let mut step = HarnessStep {
                index: index + 1,
                tool_executions: Vec::new(),
                response: response.clone(),
            };

            if response.message.tool_calls.is_empty() {
                return Ok(HarnessReport {
                    provider: self.config.kind,
                    model: response.model,
                    answer: response.message,
                    messages,
                    steps: {
                        steps.push(step);
                        steps
                    },
                    usage,
                    diagnostics,
                });
            }

            for call in &response.message.tool_calls {
                let started = Instant::now();
                let registered = tools.iter().find(|tool| tool.qualified_name == call.name);
                let (plugin_id, result) = match registered {
                    Some(tool) => {
                        let arguments = match call.arguments() {
                            Ok(arguments) => arguments,
                            Err(error) => {
                                let result =
                                    ToolResult::failure(format!("invalid tool arguments: {error}"));
                                messages.push(Message::tool_result(
                                    call,
                                    result.content.clone(),
                                    true,
                                ));
                                step.tool_executions.push(ToolExecution {
                                    plugin_id: tool.plugin_id.clone(),
                                    tool_name: call.name.clone(),
                                    call_id: call.id.clone(),
                                    result,
                                    duration_ms: started.elapsed().as_secs_f64() * 1000.0,
                                });
                                continue;
                            }
                        };
                        let result = match self.executor.execute(tool, arguments).await {
                            Ok(result) => result,
                            Err(error) => ToolResult::failure(error.to_string()),
                        };
                        (tool.plugin_id.clone(), result)
                    }
                    None => {
                        diagnostics.push(format!(
                            "model requested unregistered tool `{}`; returned an error tool result",
                            call.name
                        ));
                        (
                            "unregistered".into(),
                            ToolResult::failure(format!(
                                "tool `{}` is not registered in this Condenser profile",
                                call.name
                            )),
                        )
                    }
                };
                messages.push(Message::tool_result(
                    call,
                    result.content.clone(),
                    result.is_error,
                ));
                step.tool_executions.push(ToolExecution {
                    plugin_id,
                    tool_name: call.name.clone(),
                    call_id: call.id.clone(),
                    result,
                    duration_ms: started.elapsed().as_secs_f64() * 1000.0,
                });
            }
            steps.push(step);
        }

        Err(CondenserError::StepLimit(options.max_steps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{PluginManifest, RegisteredTool};
    use crate::types::ToolCall;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    struct MockProvider {
        responses: Mutex<VecDeque<ChatResponse>>,
        requests: Mutex<Vec<ChatRequest>>,
    }

    #[async_trait]
    impl InferenceProvider for MockProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Kadmium
        }

        async fn list_models(&self) -> Result<Vec<crate::types::ModelInfo>> {
            Ok(Vec::new())
        }

        async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.requests.lock().unwrap().push(request.clone());
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| CondenserError::provider("mock", "no response queued"))
        }
    }

    struct MockExecutor {
        tools: Vec<RegisteredTool>,
    }

    #[async_trait]
    impl ToolExecutor for MockExecutor {
        fn registered_tools(&self) -> Result<(Vec<RegisteredTool>, Vec<PluginDiagnostic>)> {
            Ok((self.tools.clone(), Vec::new()))
        }

        async fn execute(&self, tool: &RegisteredTool, arguments: Value) -> Result<ToolResult> {
            Ok(ToolResult::success(format!(
                "{}:{}",
                tool.plugin_id,
                arguments["text"].as_str().unwrap_or_default()
            )))
        }
    }

    fn registered_tool() -> RegisteredTool {
        let manifest = PluginManifest {
            protocol_version: 1,
            id: "example".into(),
            name: "Example".into(),
            version: "0.1.0".into(),
            description: "Test".into(),
            command: "unused".into(),
            args: Vec::new(),
            tools: Vec::new(),
        };
        RegisteredTool::new(
            crate::plugin::Plugin {
                manifest,
                root: std::env::current_dir().unwrap(),
            },
            ToolDefinition {
                name: "echo".into(),
                description: "Echo".into(),
                input_schema: serde_json::json!({"type": "object"}),
            },
        )
    }

    fn response(message: Message) -> ChatResponse {
        ChatResponse {
            model: "mock".into(),
            message,
            finish_reason: Some("stop".into()),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                total_tokens: 2,
            },
        }
    }

    #[tokio::test]
    async fn tool_turn_feeds_results_back_to_the_model() {
        let call = ToolCall {
            id: "call-1".into(),
            name: "example__echo".into(),
            arguments_json: "{\"text\":\"hello\"}".into(),
        };
        let mut first = Message::assistant("");
        first.tool_calls = vec![call];
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([
                response(first),
                response(Message::assistant("done")),
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let executor = Arc::new(MockExecutor {
            tools: vec![registered_tool()],
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider.clone(),
            executor,
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("run the tool")],
                HarnessOptions {
                    enable_tools: true,
                    max_steps: 4,
                    sampling: SamplingParams::default(),
                },
            )
            .await
            .unwrap();
        assert_eq!(report.answer.content, "done");
        assert_eq!(report.messages.len(), 4);
        assert_eq!(report.messages[2].role, Role::Tool);
        assert_eq!(report.messages[2].content, "example:hello");
        assert_eq!(report.usage.total_tokens, 4);
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests[0].tools.len(), 1);
        assert_eq!(requests[1].messages.len(), 3);
    }

    #[tokio::test]
    async fn tools_are_opt_in() {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([response(Message::assistant("done"))])),
            requests: Mutex::new(Vec::new()),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider.clone(),
            Arc::new(MockExecutor { tools: Vec::new() }),
        )
        .unwrap();
        harness
            .run(vec![Message::user("hello")], HarnessOptions::default())
            .await
            .unwrap();
        assert!(provider.requests.lock().unwrap()[0].tools.is_empty());
    }
}
