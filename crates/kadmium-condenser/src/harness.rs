use crate::dsv41_mcl4::{detect_tool_doom_loop, segment_reasoning_and_answer};
use crate::plugin::{PluginDiagnostic, PluginManager, RegisteredTool};
use crate::provider::{InferenceProvider, ProviderFactory, synthetic_tool_call_id};
#[cfg(test)]
use crate::types::Role;
use crate::types::{
    ChatRequest, ChatResponse, Message, ProviderConfig, ProviderKind, SamplingParams, ToolCall,
    ToolDefinition, ToolResult, Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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

pub const MAX_HARNESS_STEPS: usize = 256;
pub const MAX_CALLS_PER_STEP: usize = 16;

#[derive(Debug, Clone)]
pub struct HarnessOptions {
    pub enable_tools: bool,
    /// Exact qualified tool names. None preserves all-tools behavior; Some([]) denies all.
    pub allowed_tools: Option<Vec<String>>,
    pub max_steps: usize,
    pub max_calls_per_step: usize,
    pub sampling: SamplingParams,
}

impl Default for HarnessOptions {
    fn default() -> Self {
        Self {
            enable_tools: false,
            allowed_tools: None,
            max_steps: 8,
            max_calls_per_step: MAX_CALLS_PER_STEP,
            sampling: SamplingParams::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Completed,
    StepLimit,
    Cancelled,
    BudgetExceeded,
    Failed,
}

#[derive(Debug, Clone)]
pub struct RunControl {
    pub cancellation: CancellationToken,
    pub deadline: Option<tokio::time::Instant>,
    pub max_output_tokens: Option<u64>,
    pub max_tool_calls: Option<usize>,
}

impl Default for RunControl {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            deadline: None,
            max_output_tokens: None,
            max_tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum HarnessEvent {
    ModelStarted {
        step: usize,
    },
    ModelToken {
        step: usize,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_delta: Option<String>,
        generated_tokens: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        tok_per_s: Option<f64>,
    },
    ModelCompleted {
        step: usize,
        response: ChatResponse,
    },
    ToolStarted {
        step: usize,
        call_id: String,
        tool_name: String,
    },
    ToolCompleted {
        step: usize,
        call_id: String,
        tool_name: String,
        result: ToolResult,
    },
    Finished {
        report: HarnessReport,
    },
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
    pub status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Default)]
struct ToolRegistry {
    qualified: HashMap<String, usize>,
    unqualified: HashMap<String, Option<usize>>,
    validators: Vec<Option<Arc<jsonschema::Validator>>>,
}

impl ToolRegistry {
    fn new(tools: &[RegisteredTool]) -> Self {
        let mut registry = Self {
            qualified: HashMap::with_capacity(tools.len()),
            unqualified: HashMap::with_capacity(tools.len()),
            validators: Vec::with_capacity(tools.len()),
        };
        for (index, tool) in tools.iter().enumerate() {
            registry
                .qualified
                .insert(tool.qualified_name.clone(), index);
            registry
                .unqualified
                .entry(tool.unqualified_name().to_string())
                .and_modify(|match_index| *match_index = None)
                .or_insert(Some(index));
            registry.validators.push(
                jsonschema::validator_for(&tool.definition.input_schema)
                    .ok()
                    .map(Arc::new),
            );
        }
        registry
    }

    fn get_with_validator<'a>(
        &'a self,
        tools: &'a [RegisteredTool],
        name: &str,
    ) -> Option<(&'a RegisteredTool, Option<&'a jsonschema::Validator>)> {
        let index = self
            .qualified
            .get(name)
            .copied()
            .or_else(|| self.unqualified.get(name).copied().flatten())?;
        let tool = tools.get(index)?;
        let validator = self.validators.get(index).and_then(|v| v.as_deref());
        Some((tool, validator))
    }
}

#[inline]
fn needs_reasoning_or_dsml_scan(content: &str, reasoning: Option<&str>) -> bool {
    content.contains("#T$#")
        || content.contains("#A$#")
        || content.starts_with("#U$#")
        || content.contains("<think>")
        || content.contains("DSML")
        || reasoning.is_some_and(|r| r.contains("DSML"))
}

enum WaitOutcome<T> {
    Completed(T),
    Stopped(RunStatus, String),
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

    pub fn new_with_client(
        config: ProviderConfig,
        plugins: PluginManager,
        client: reqwest::Client,
    ) -> Result<Self> {
        let provider = ProviderFactory::build_with_client(config.clone(), client)?;
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

    /// Runs model requests until the assistant stops calling tools or a run
    /// limit is reached. Tool failures remain durable model-visible results.
    pub async fn run(
        &self,
        messages: Vec<Message>,
        options: HarnessOptions,
    ) -> Result<HarnessReport> {
        self.run_with_control(messages, options, RunControl::default(), None)
            .await
    }

    /// Runs with cooperative cancellation, a whole-run deadline, aggregate
    /// budgets, and best-effort progress events. Tools remain serial.
    pub async fn run_with_control(
        &self,
        mut messages: Vec<Message>,
        options: HarnessOptions,
        mut control: RunControl,
        events: Option<mpsc::Sender<HarnessEvent>>,
    ) -> Result<HarnessReport> {
        if messages.is_empty() {
            return Err(CondenserError::Config(
                "a harness turn needs at least one message".into(),
            ));
        }
        if !(1..=MAX_HARNESS_STEPS).contains(&options.max_steps) {
            return Err(CondenserError::Config(format!(
                "max_steps must be between 1 and {MAX_HARNESS_STEPS}"
            )));
        }
        if !(1..=MAX_CALLS_PER_STEP).contains(&options.max_calls_per_step) {
            return Err(CondenserError::Config(format!(
                "max_calls_per_step must be between 1 and {MAX_CALLS_PER_STEP}"
            )));
        }
        options.sampling.validate()?;

        if options.max_steps > 32 {
            control.max_tool_calls.get_or_insert(64);
            control.deadline.get_or_insert_with(|| {
                tokio::time::Instant::now() + std::time::Duration::from_secs(1800)
            });
        }

        let (mut tools, diagnostics) = if options.enable_tools {
            self.executor.registered_tools()?
        } else {
            (Vec::new(), Vec::new())
        };
        if options.enable_tools
            && let Some(allowed) = &options.allowed_tools
        {
            for name in allowed {
                if !tools.iter().any(|tool| &tool.qualified_name == name) {
                    return Err(CondenserError::Config(format!(
                        "allowed tool `{name}` is not registered; use an exact plugin__tool name"
                    )));
                }
            }
            tools.retain(|tool| allowed.contains(&tool.qualified_name));
        }
        let registry = ToolRegistry::new(&tools);
        let mut definitions = tools
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
        definitions.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        let mut diagnostics = diagnostics
            .into_iter()
            .map(|diagnostic| format!("{}: {}", diagnostic.path.display(), diagnostic.message))
            .collect::<Vec<_>>();
        let mut steps = Vec::new();
        let mut usage = Usage::default();
        let mut model = self.config.model.clone();
        let mut answer = Message::assistant("");
        let mut tool_call_count = 0usize;
        let mut prior_tool_signatures: Vec<Vec<(String, String)>> = Vec::new();

        for index in 0..options.max_steps {
            if let Some((status, error)) = stopped(&control) {
                return self.finish(
                    model,
                    answer,
                    messages,
                    steps,
                    usage,
                    diagnostics,
                    status,
                    Some(error),
                    &events,
                );
            }
            if let Some(error) = output_budget_error(&control, usage.output_tokens) {
                return self.finish(
                    model,
                    answer,
                    messages,
                    steps,
                    usage,
                    diagnostics,
                    RunStatus::BudgetExceeded,
                    Some(error),
                    &events,
                );
            }

            emit(&events, HarnessEvent::ModelStarted { step: index + 1 });
            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: messages.clone(),
                sampling: options.sampling.clone(),
                tools: definitions.clone(),
            };
            let (token_tx, token_forwarder) = if let Some(ref event_sender) = events {
                let (tx, mut rx) = mpsc::unbounded_channel::<crate::provider::StreamedTokenDelta>();
                let sender_clone = event_sender.clone();
                let step_num = index + 1;
                let handle = tokio::spawn(async move {
                    while let Some(tok) = rx.recv().await {
                        let _ = sender_clone.try_send(HarnessEvent::ModelToken {
                            step: step_num,
                            delta: tok.delta,
                            reasoning_delta: tok.reasoning_delta,
                            generated_tokens: tok.generated_tokens,
                            tok_per_s: tok.tok_per_s,
                        });
                    }
                });
                (Some(tx), Some(handle))
            } else {
                (None, None)
            };
            let provider_wait = tokio::time::timeout(
                std::time::Duration::from_secs(self.config.timeout_secs),
                self.provider.complete_stream(&request, token_tx),
            );
            let wait_res = controlled_wait(&control, provider_wait).await;
            if let Some(handle) = token_forwarder {
                let _ = handle.await;
            }
            let mut response = match wait_res {
                WaitOutcome::Completed(Ok(Ok(response))) => response,
                WaitOutcome::Completed(Ok(Err(error))) => {
                    return self.finish(
                        model,
                        answer,
                        messages,
                        steps,
                        usage,
                        diagnostics,
                        RunStatus::Failed,
                        Some(error.to_string()),
                        &events,
                    );
                }
                WaitOutcome::Completed(Err(_)) => {
                    let error = CondenserError::provider(
                        self.config.kind.label(),
                        format!(
                            "request timed out after {} seconds",
                            self.config.timeout_secs
                        ),
                    );
                    return self.finish(
                        model,
                        answer,
                        messages,
                        steps,
                        usage,
                        diagnostics,
                        RunStatus::Failed,
                        Some(error.to_string()),
                        &events,
                    );
                }
                WaitOutcome::Stopped(status, error) => {
                    return self.finish(
                        model,
                        answer,
                        messages,
                        steps,
                        usage,
                        diagnostics,
                        status,
                        Some(error),
                        &events,
                    );
                }
            };
            let structured_call_count = response.message.tool_calls.len();
            if needs_reasoning_or_dsml_scan(
                &response.message.content,
                response.message.reasoning.as_deref(),
            ) {
                let segmented = segment_reasoning_and_answer(
                    &response.message.content,
                    response.message.reasoning.as_deref(),
                    None,
                );
                if segmented.thought.is_some() || !segmented.dsml_tool_calls.is_empty() {
                    response.message.content = segmented.answer;
                    response.message.reasoning = segmented.thought;
                    response
                        .message
                        .tool_calls
                        .extend(segmented.dsml_tool_calls);
                }
            }
            normalize_tool_calls(&mut response.message.tool_calls, structured_call_count);
            emit(
                &events,
                HarnessEvent::ModelCompleted {
                    step: index + 1,
                    response: response.clone(),
                },
            );

            usage.add_assign(&response.usage);
            model = response.model.clone();
            answer = response.message.clone();
            messages.push(response.message.clone());
            let calls = response.message.tool_calls.clone();
            let mut step = HarnessStep {
                index: index + 1,
                tool_executions: Vec::new(),
                response: response.clone(),
            };

            if let Some(error) = output_budget_error(&control, usage.output_tokens) {
                record_blocked_calls(
                    &calls,
                    "output-token-budget",
                    &format!("{error}; no tool in this batch was started"),
                    index + 1,
                    &mut messages,
                    &mut step,
                    &events,
                );
                steps.push(step);
                return self.finish(
                    model,
                    answer,
                    messages,
                    steps,
                    usage,
                    diagnostics,
                    RunStatus::BudgetExceeded,
                    Some(error),
                    &events,
                );
            }
            if let Some((status, error)) = stopped(&control) {
                record_blocked_calls(
                    &calls,
                    "run-control",
                    &format!("{error}; no tool in this batch was started"),
                    index + 1,
                    &mut messages,
                    &mut step,
                    &events,
                );
                steps.push(step);
                return self.finish(
                    model,
                    answer,
                    messages,
                    steps,
                    usage,
                    diagnostics,
                    status,
                    Some(error),
                    &events,
                );
            }
            if calls.is_empty() {
                steps.push(step);
                return self.finish(
                    model,
                    answer,
                    messages,
                    steps,
                    usage,
                    diagnostics,
                    RunStatus::Completed,
                    None,
                    &events,
                );
            }
            if calls.len() > options.max_calls_per_step {
                let error = format!(
                    "tool-call count {} exceeds per-step limit {}",
                    calls.len(),
                    options.max_calls_per_step
                );
                record_blocked_calls(
                    &calls,
                    "tool-call-budget",
                    &format!("{error}; no tool in this batch was started"),
                    index + 1,
                    &mut messages,
                    &mut step,
                    &events,
                );
                steps.push(step);
                return self.finish(
                    model,
                    answer,
                    messages,
                    steps,
                    usage,
                    diagnostics,
                    RunStatus::BudgetExceeded,
                    Some(error),
                    &events,
                );
            }

            let doom_warning = detect_tool_doom_loop(&prior_tool_signatures, &calls, 3);
            if let Some(ref warning) = doom_warning {
                diagnostics.push(warning.clone());
            }
            prior_tool_signatures.push(
                calls
                    .iter()
                    .map(|call| (call.name.clone(), call.arguments_json.clone()))
                    .collect(),
            );

            for (call_index, call) in calls.iter().enumerate() {
                if let Some((status, error)) = stopped(&control) {
                    record_blocked_calls(
                        &calls[call_index..],
                        "run-control",
                        &format!("{error}; tool was not started"),
                        index + 1,
                        &mut messages,
                        &mut step,
                        &events,
                    );
                    steps.push(step);
                    return self.finish(
                        model,
                        answer,
                        messages,
                        steps,
                        usage,
                        diagnostics,
                        status,
                        Some(error),
                        &events,
                    );
                }
                if control
                    .max_tool_calls
                    .is_some_and(|limit| tool_call_count >= limit)
                {
                    let error = format!("aggregate tool-call budget {} exhausted", tool_call_count);
                    record_blocked_calls(
                        &calls[call_index..],
                        "tool-call-budget",
                        &format!("{error}; tool was not started"),
                        index + 1,
                        &mut messages,
                        &mut step,
                        &events,
                    );
                    steps.push(step);
                    return self.finish(
                        model,
                        answer,
                        messages,
                        steps,
                        usage,
                        diagnostics,
                        RunStatus::BudgetExceeded,
                        Some(error),
                        &events,
                    );
                }

                let started = Instant::now();
                if let Some(ref warning) = doom_warning {
                    let result = ToolResult::failure(format!(
                        "Condenser blocked a repeated tool batch. {warning}; synthesize your final answer or choose different arguments."
                    ));
                    record_tool_result(
                        "doom-loop-guard",
                        call,
                        result,
                        started,
                        index + 1,
                        &mut messages,
                        &mut step,
                        &events,
                    );
                    continue;
                }

                let Some((tool, validator)) = registry.get_with_validator(&tools, &call.name)
                else {
                    diagnostics.push(format!(
                        "model requested unregistered or ambiguous tool `{}`; returned an error tool result",
                        call.name
                    ));
                    record_tool_result(
                        "unregistered",
                        call,
                        ToolResult::failure(format!(
                            "tool `{}` is not registered or is ambiguous in this Condenser profile",
                            call.name
                        )),
                        started,
                        index + 1,
                        &mut messages,
                        &mut step,
                        &events,
                    );
                    continue;
                };
                let arguments = match call.arguments() {
                    Ok(arguments) => arguments,
                    Err(error) => {
                        record_tool_result(
                            &tool.plugin_id,
                            call,
                            ToolResult::failure(format!("invalid tool arguments: {error}")),
                            started,
                            index + 1,
                            &mut messages,
                            &mut step,
                            &events,
                        );
                        continue;
                    }
                };
                let schema_valid = validator.map(|v| v.is_valid(&arguments)).unwrap_or(false);
                if !schema_valid {
                    record_tool_result(
                        &tool.plugin_id,
                        call,
                        ToolResult::failure("tool arguments do not match the declared JSON Schema"),
                        started,
                        index + 1,
                        &mut messages,
                        &mut step,
                        &events,
                    );
                    continue;
                }

                emit(
                    &events,
                    HarnessEvent::ToolStarted {
                        step: index + 1,
                        call_id: call.id.clone(),
                        tool_name: call.name.clone(),
                    },
                );
                tool_call_count += 1;
                let result = match controlled_wait(&control, self.executor.execute(tool, arguments))
                    .await
                {
                    WaitOutcome::Completed(Ok(result)) => result,
                    WaitOutcome::Completed(Err(error)) => ToolResult::failure(error.to_string()),
                    WaitOutcome::Stopped(status, error) => {
                        record_tool_result(
                            &tool.plugin_id,
                            call,
                            ToolResult::failure(format!(
                                "{error}; tool execution was interrupted and its side effects are unknown"
                            )),
                            started,
                            index + 1,
                            &mut messages,
                            &mut step,
                            &events,
                        );
                        record_blocked_calls(
                            &calls[call_index + 1..],
                            "run-control",
                            &format!("{error}; tool was not started"),
                            index + 1,
                            &mut messages,
                            &mut step,
                            &events,
                        );
                        steps.push(step);
                        return self.finish(
                            model,
                            answer,
                            messages,
                            steps,
                            usage,
                            diagnostics,
                            status,
                            Some(error),
                            &events,
                        );
                    }
                };
                record_tool_result(
                    &tool.plugin_id,
                    call,
                    result,
                    started,
                    index + 1,
                    &mut messages,
                    &mut step,
                    &events,
                );
            }
            steps.push(step);
            if index + 1 == options.max_steps {
                return self.finish(
                    model,
                    answer,
                    messages,
                    steps,
                    usage,
                    diagnostics,
                    RunStatus::StepLimit,
                    None,
                    &events,
                );
            }
        }

        unreachable!("max_steps is validated as nonzero")
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        model: String,
        answer: Message,
        messages: Vec<Message>,
        steps: Vec<HarnessStep>,
        usage: Usage,
        diagnostics: Vec<String>,
        status: RunStatus,
        error: Option<String>,
        events: &Option<mpsc::Sender<HarnessEvent>>,
    ) -> Result<HarnessReport> {
        let report = HarnessReport {
            provider: self.config.kind,
            model,
            answer,
            messages,
            steps,
            usage,
            diagnostics,
            status,
            error,
        };
        emit(
            events,
            HarnessEvent::Finished {
                report: report.clone(),
            },
        );
        Ok(report)
    }
}

fn emit(events: &Option<mpsc::Sender<HarnessEvent>>, event: HarnessEvent) {
    if let Some(sender) = events {
        let _ = sender.try_send(event);
    }
}

fn stopped(control: &RunControl) -> Option<(RunStatus, String)> {
    if control.cancellation.is_cancelled() {
        Some((RunStatus::Cancelled, "run cancelled".into()))
    } else if control
        .deadline
        .is_some_and(|deadline| tokio::time::Instant::now() >= deadline)
    {
        Some((RunStatus::Cancelled, "run deadline exceeded".into()))
    } else {
        None
    }
}

async fn controlled_wait<F: Future>(control: &RunControl, future: F) -> WaitOutcome<F::Output> {
    let deadline = control.deadline;
    tokio::select! {
        biased;
        _ = control.cancellation.cancelled() => {
            WaitOutcome::Stopped(RunStatus::Cancelled, "run cancelled".into())
        }
        _ = async {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        } => {
            WaitOutcome::Stopped(RunStatus::Cancelled, "run deadline exceeded".into())
        }
        output = future => WaitOutcome::Completed(output),
    }
}

fn output_budget_error(control: &RunControl, output_tokens: u64) -> Option<String> {
    control
        .max_output_tokens
        .filter(|limit| output_tokens >= *limit)
        .map(|limit| format!("output token budget {limit} exhausted at {output_tokens} tokens"))
}

fn normalize_tool_calls(calls: &mut Vec<ToolCall>, structured_call_count: usize) {
    let structured_signatures = calls[..structured_call_count]
        .iter()
        .map(tool_call_signature)
        .collect::<HashSet<_>>();
    if calls.len() > structured_call_count {
        let mut dsml_calls = calls.split_off(structured_call_count);
        dsml_calls.retain(|call| !structured_signatures.contains(&tool_call_signature(call)));
        calls.extend(dsml_calls);
    }

    let mut seen = HashSet::new();
    for (index, call) in calls.iter_mut().enumerate() {
        if call.id.trim().is_empty() || !seen.insert(call.id.clone()) {
            let mut replacement = synthetic_tool_call_id(index);
            while seen.contains(&replacement) {
                replacement = synthetic_tool_call_id(index);
            }
            call.id = replacement;
            seen.insert(call.id.clone());
        }
    }
}

fn tool_call_signature(call: &ToolCall) -> (String, String) {
    let arguments = serde_json::from_str::<Value>(&call.arguments_json)
        .map(canonical_json)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| call.arguments_json.trim().to_string());
    (call.name.clone(), arguments)
}

fn canonical_json(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_json).collect()),
        Value::Object(values) => {
            let sorted = values
                .into_iter()
                .map(|(key, value)| (key, canonical_json(value)))
                .collect::<std::collections::BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect())
        }
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn record_tool_result(
    plugin_id: &str,
    call: &ToolCall,
    result: ToolResult,
    started: Instant,
    step_index: usize,
    messages: &mut Vec<Message>,
    step: &mut HarnessStep,
    events: &Option<mpsc::Sender<HarnessEvent>>,
) {
    messages.push(Message::tool_result(
        call,
        result.content.clone(),
        result.is_error,
    ));
    emit(
        events,
        HarnessEvent::ToolCompleted {
            step: step_index,
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            result: result.clone(),
        },
    );
    step.tool_executions.push(ToolExecution {
        plugin_id: plugin_id.into(),
        tool_name: call.name.clone(),
        call_id: call.id.clone(),
        result,
        duration_ms: started.elapsed().as_secs_f64() * 1000.0,
    });
}

fn record_blocked_calls(
    calls: &[ToolCall],
    plugin_id: &str,
    error: &str,
    step_index: usize,
    messages: &mut Vec<Message>,
    step: &mut HarnessStep,
    events: &Option<mpsc::Sender<HarnessEvent>>,
) {
    for call in calls {
        record_tool_result(
            plugin_id,
            call,
            ToolResult::failure(error),
            Instant::now(),
            step_index,
            messages,
            step,
            events,
        );
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

    struct CountingExecutor {
        tool: RegisteredTool,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl ToolExecutor for CountingExecutor {
        fn registered_tools(&self) -> Result<(Vec<RegisteredTool>, Vec<PluginDiagnostic>)> {
            Ok((vec![self.tool.clone()], Vec::new()))
        }

        async fn execute(&self, _tool: &RegisteredTool, _arguments: Value) -> Result<ToolResult> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ToolResult::success("executed"))
        }
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
            telemetry: None,
        }
    }

    #[tokio::test]
    async fn tool_allowlist_filters_catalog_and_blocks_excluded_calls() {
        let excluded = registered_tool();
        let mut allowed = excluded.clone();
        allowed.qualified_name = "example__read".into();
        allowed.definition.name = "read".into();
        let mut first = Message::assistant("");
        first.tool_calls = vec![
            ToolCall {
                id: "blocked".into(),
                name: "example__echo".into(),
                arguments_json: "{}".into(),
            },
            ToolCall {
                id: "allowed".into(),
                name: "example__read".into(),
                arguments_json: "{\"text\":\"ok\"}".into(),
            },
        ];
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([
                response(first),
                response(Message::assistant("done")),
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider.clone(),
            Arc::new(MockExecutor {
                tools: vec![excluded, allowed],
            }),
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("read only")],
                HarnessOptions {
                    enable_tools: true,
                    allowed_tools: Some(vec!["example__read".into()]),
                    ..HarnessOptions::default()
                },
            )
            .await
            .unwrap();
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests[0].tools.len(), 1);
        assert_eq!(requests[0].tools[0].name, "example__read");
        assert!(report.messages[2].is_error);
        assert_eq!(report.messages[2].tool_call_id.as_deref(), Some("blocked"));
        assert!(!report.messages[3].is_error);
        assert_eq!(report.messages[3].content, "example:ok");
    }

    #[tokio::test]
    async fn empty_allowlist_and_tools_off_never_execute_a_plugin() {
        for (enabled, allowed) in [(true, vec![]), (false, vec!["example__echo".into()])] {
            let mut first = Message::assistant("");
            first.tool_calls = vec![ToolCall {
                id: "call".into(),
                name: "echo".into(),
                arguments_json: "{}".into(),
            }];
            let provider = Arc::new(MockProvider {
                responses: Mutex::new(VecDeque::from([
                    response(first),
                    response(Message::assistant("done")),
                ])),
                requests: Mutex::new(Vec::new()),
            });
            let executor = Arc::new(CountingExecutor {
                tool: registered_tool(),
                calls: std::sync::atomic::AtomicUsize::new(0),
            });
            let harness = Harness::from_components(
                ProviderConfig::new(ProviderKind::Kadmium),
                provider.clone(),
                executor.clone(),
            )
            .unwrap();
            let report = harness
                .run(
                    vec![Message::user("try a call")],
                    HarnessOptions {
                        enable_tools: enabled,
                        allowed_tools: Some(allowed),
                        ..HarnessOptions::default()
                    },
                )
                .await
                .unwrap();
            assert!(provider.requests.lock().unwrap()[0].tools.is_empty());
            assert_eq!(executor.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
            assert!(report.messages[2].is_error);
        }
    }

    #[tokio::test]
    async fn unknown_or_unqualified_allowlist_names_fail_before_inference() {
        for name in ["echo", "missing__tool"] {
            let provider = Arc::new(MockProvider {
                responses: Mutex::new(VecDeque::new()),
                requests: Mutex::new(Vec::new()),
            });
            let harness = Harness::from_components(
                ProviderConfig::new(ProviderKind::Kadmium),
                provider.clone(),
                Arc::new(MockExecutor {
                    tools: vec![registered_tool()],
                }),
            )
            .unwrap();
            let result = harness
                .run(
                    vec![Message::user("hello")],
                    HarnessOptions {
                        enable_tools: true,
                        allowed_tools: Some(vec![name.into()]),
                        ..HarnessOptions::default()
                    },
                )
                .await;
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("exact plugin__tool name")
            );
            assert!(provider.requests.lock().unwrap().is_empty());
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
                    allowed_tools: None,
                    max_steps: 4,
                    max_calls_per_step: 16,
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

    #[tokio::test]
    async fn dsml_and_uta_reasoning_traces_execute_tools_and_preserve_history() {
        let dsml_step1 = Message::assistant(
            "#U$# Echo test\n#T$# Let us invoke echo with 6 * 7 = 42.\n#A$# Invoking echo.\n\
             <｜DSML｜ calls>\n\
             <｜DSML｜ invoke name=\"example__echo\">\n\
             <｜DSML｜ parameter name=\"text\" string=\"true\">42</｜DSML｜ parameter>\n\
             </｜DSML｜ invoke>\n\
             </｜DSML｜ calls>",
        );
        let uta_step2 = Message::assistant(
            "#T$# Tool returned example:42. Verified 6 * 7 = 42.\n#A$# Echo returned example:42.",
        );
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([response(dsml_step1), response(uta_step2)])),
            requests: Mutex::new(Vec::new()),
        });
        let executor = Arc::new(MockExecutor {
            tools: vec![registered_tool()],
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Mcl4Coe),
            provider.clone(),
            executor,
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("Echo test")],
                HarnessOptions {
                    enable_tools: true,
                    allowed_tools: None,
                    max_steps: 4,
                    max_calls_per_step: 16,
                    sampling: SamplingParams::default(),
                },
            )
            .await
            .unwrap();

        // Transcript preserves reasoning on both steps
        assert_eq!(
            report.messages[1].reasoning.as_deref(),
            Some("Let us invoke echo with 6 * 7 = 42.")
        );
        assert_eq!(report.messages[1].content, "Invoking echo.");
        assert_eq!(report.messages[2].content, "example:42");
        assert_eq!(
            report.answer.reasoning.as_deref(),
            Some("Tool returned example:42. Verified 6 * 7 = 42.")
        );
        assert_eq!(report.answer.content, "Echo returned example:42.");

        // The harness preserves canonical history; the real MCL4 provider owns
        // outbound compaction instead of compacting a second time here.
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests[1].messages[1].content, "Invoking echo.");
        assert_eq!(
            requests[1].messages[1].reasoning.as_deref(),
            Some("Let us invoke echo with 6 * 7 = 42.")
        );
    }

    #[tokio::test]
    async fn doom_loop_guard_blocks_the_repeated_side_effect() {
        let repeated_call = || {
            let mut message = Message::assistant("trying again");
            message.tool_calls = vec![ToolCall {
                id: "call-1".into(),
                name: "example__echo".into(),
                arguments_json: "{\"text\":\"same\"}".into(),
            }];
            response(message)
        };
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([
                repeated_call(),
                repeated_call(),
                repeated_call(),
                response(Message::assistant("recovered")),
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let executor = Arc::new(CountingExecutor {
            tool: registered_tool(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider,
            executor.clone(),
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("run the tool")],
                HarnessOptions {
                    enable_tools: true,
                    allowed_tools: None,
                    max_steps: 4,
                    max_calls_per_step: 16,
                    sampling: SamplingParams::default(),
                },
            )
            .await
            .unwrap();

        assert_eq!(report.answer.content, "recovered");
        assert_eq!(executor.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        let blocked = &report.steps[2].tool_executions[0];
        assert_eq!(blocked.plugin_id, "doom-loop-guard");
        assert!(blocked.result.is_error);
    }

    struct PendingProvider;

    #[async_trait]
    impl InferenceProvider for PendingProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Kadmium
        }

        async fn list_models(&self) -> Result<Vec<crate::types::ModelInfo>> {
            Ok(Vec::new())
        }

        async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            std::future::pending().await
        }
    }

    fn tool_response() -> ChatResponse {
        let mut message = Message::assistant("calling");
        message.tool_calls = vec![ToolCall {
            id: "call-1".into(),
            name: "example__echo".into(),
            arguments_json: "{\"text\":\"hello\"}".into(),
        }];
        response(message)
    }

    fn pending_harness() -> Harness {
        Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            Arc::new(PendingProvider),
            Arc::new(MockExecutor { tools: Vec::new() }),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn step_limit_returns_the_complete_partial_report() {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([tool_response()])),
            requests: Mutex::new(Vec::new()),
        });
        let executor = Arc::new(CountingExecutor {
            tool: registered_tool(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider,
            executor,
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("run")],
                HarnessOptions {
                    enable_tools: true,
                    max_steps: 1,
                    ..HarnessOptions::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(report.status, RunStatus::StepLimit);
        assert_eq!(report.steps.len(), 1);
        assert_eq!(report.steps[0].tool_executions.len(), 1);
        assert_eq!(report.messages.len(), 3);
        assert!(report.error.is_none());
    }

    #[tokio::test]
    async fn cancellation_interrupts_a_pending_provider() {
        let control = RunControl::default();
        let cancellation = control.cancellation.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancellation.cancel();
        });
        let report = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            pending_harness().run_with_control(
                vec![Message::user("wait")],
                HarnessOptions::default(),
                control,
                None,
            ),
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(report.status, RunStatus::Cancelled);
        assert!(report.steps.is_empty());
        assert_eq!(report.error.as_deref(), Some("run cancelled"));
    }

    #[tokio::test]
    async fn deadline_interrupts_a_pending_provider() {
        let control = RunControl {
            deadline: Some(tokio::time::Instant::now() + std::time::Duration::from_millis(20)),
            ..RunControl::default()
        };
        let report = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            pending_harness().run_with_control(
                vec![Message::user("wait")],
                HarnessOptions::default(),
                control,
                None,
            ),
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(report.status, RunStatus::Cancelled);
        assert_eq!(report.error.as_deref(), Some("run deadline exceeded"));
    }

    #[tokio::test]
    async fn output_budget_blocks_tools_and_the_next_provider_dispatch() {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([
                tool_response(),
                response(Message::assistant("must not be requested")),
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let executor = Arc::new(CountingExecutor {
            tool: registered_tool(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider.clone(),
            executor.clone(),
        )
        .unwrap();
        let report = harness
            .run_with_control(
                vec![Message::user("run")],
                HarnessOptions {
                    enable_tools: true,
                    ..HarnessOptions::default()
                },
                RunControl {
                    max_output_tokens: Some(1),
                    ..RunControl::default()
                },
                None,
            )
            .await
            .unwrap();

        assert_eq!(report.status, RunStatus::BudgetExceeded);
        assert_eq!(provider.requests.lock().unwrap().len(), 1);
        assert_eq!(executor.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(report.steps[0].tool_executions.len(), 1);
        assert!(report.steps[0].tool_executions[0].result.is_error);
    }

    #[tokio::test]
    async fn aggregate_tool_budget_is_checked_before_side_effects() {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([tool_response()])),
            requests: Mutex::new(Vec::new()),
        });
        let executor = Arc::new(CountingExecutor {
            tool: registered_tool(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider,
            executor.clone(),
        )
        .unwrap();
        let report = harness
            .run_with_control(
                vec![Message::user("run")],
                HarnessOptions {
                    enable_tools: true,
                    ..HarnessOptions::default()
                },
                RunControl {
                    max_tool_calls: Some(0),
                    ..RunControl::default()
                },
                None,
            )
            .await
            .unwrap();

        assert_eq!(report.status, RunStatus::BudgetExceeded);
        assert_eq!(executor.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(report.messages.last().unwrap().is_error);
        assert_eq!(
            report.messages.last().unwrap().tool_call_id.as_deref(),
            Some("call-1")
        );
    }

    #[test]
    fn tool_call_ids_are_repaired_centrally() {
        let mut calls = vec![
            ToolCall {
                id: String::new(),
                name: "one".into(),
                arguments_json: "{}".into(),
            },
            ToolCall {
                id: "duplicate".into(),
                name: "two".into(),
                arguments_json: "{}".into(),
            },
            ToolCall {
                id: "duplicate".into(),
                name: "three".into(),
                arguments_json: "{}".into(),
            },
        ];
        normalize_tool_calls(&mut calls, 3);

        assert!(calls.iter().all(|call| !call.id.trim().is_empty()));
        assert_eq!(
            calls
                .iter()
                .map(|call| call.id.clone())
                .collect::<HashSet<_>>()
                .len(),
            3
        );
        assert_eq!(calls[1].id, "duplicate");
    }

    #[test]
    fn cross_format_dedup_preserves_intentional_structured_duplicates() {
        let structured = |id: &str| ToolCall {
            id: id.into(),
            name: "example__echo".into(),
            arguments_json: "{\"b\":2,\"a\":1}".into(),
        };
        let mut calls = vec![
            structured("structured-1"),
            structured("structured-2"),
            ToolCall {
                id: "dsml-duplicate".into(),
                name: "example__echo".into(),
                arguments_json: "{\"a\":1,\"b\":2}".into(),
            },
            ToolCall {
                id: "dsml-distinct".into(),
                name: "example__echo".into(),
                arguments_json: "{\"a\":9,\"b\":2}".into(),
            },
        ];
        normalize_tool_calls(&mut calls, 2);

        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].id, "structured-1");
        assert_eq!(calls[1].id, "structured-2");
        assert_eq!(calls[2].id, "dsml-distinct");
    }

    #[tokio::test]
    async fn provider_failure_after_prior_work_returns_a_failed_report() {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([tool_response()])),
            requests: Mutex::new(Vec::new()),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider,
            Arc::new(MockExecutor {
                tools: vec![registered_tool()],
            }),
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("run")],
                HarnessOptions {
                    enable_tools: true,
                    ..HarnessOptions::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(report.status, RunStatus::Failed);
        assert_eq!(report.steps.len(), 1);
        assert!(report.error.is_some());
    }

    #[tokio::test]
    async fn schema_invalid_arguments_never_reach_tool_execution() {
        let mut invalid_call = Message::assistant("calling");
        invalid_call.tool_calls = vec![ToolCall {
            id: "call-1".into(),
            name: "example__echo".into(),
            arguments_json: "{}".into(),
        }];
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([
                response(invalid_call),
                response(Message::assistant("recovered")),
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let mut tool = registered_tool();
        tool.definition.input_schema = serde_json::json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        });
        let executor = Arc::new(CountingExecutor {
            tool,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider,
            executor.clone(),
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("run")],
                HarnessOptions {
                    enable_tools: true,
                    ..HarnessOptions::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(report.status, RunStatus::Completed);
        assert_eq!(executor.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(report.steps[0].tool_executions[0].result.is_error);
        assert!(
            report.steps[0].tool_executions[0]
                .result
                .content
                .contains("JSON Schema")
        );
    }

    #[tokio::test]
    async fn first_provider_failure_preserves_the_accepted_prompt() {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider,
            Arc::new(MockExecutor { tools: Vec::new() }),
        )
        .unwrap();
        let report = harness
            .run(
                vec![Message::user("preserve this prompt")],
                HarnessOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(report.status, RunStatus::Failed);
        assert_eq!(report.messages[0].content, "preserve this prompt");
        assert!(report.error.is_some());
    }

    #[tokio::test]
    async fn long_horizon_step_and_call_bounds_are_enforced() {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(VecDeque::from([response(Message::assistant("done"))])),
            requests: Mutex::new(Vec::new()),
        });
        let harness = Harness::from_components(
            ProviderConfig::new(ProviderKind::Kadmium),
            provider,
            Arc::new(MockExecutor { tools: Vec::new() }),
        )
        .unwrap();

        let ok_report = harness
            .run(
                vec![Message::user("long horizon")],
                HarnessOptions {
                    max_steps: MAX_HARNESS_STEPS,
                    max_calls_per_step: MAX_CALLS_PER_STEP,
                    ..HarnessOptions::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(ok_report.status, RunStatus::Completed);

        let over_steps = harness
            .run(
                vec![Message::user("too many steps")],
                HarnessOptions {
                    max_steps: MAX_HARNESS_STEPS + 1,
                    ..HarnessOptions::default()
                },
            )
            .await
            .unwrap_err();
        assert!(over_steps.to_string().contains("256"));

        let over_calls = harness
            .run(
                vec![Message::user("too many calls")],
                HarnessOptions {
                    max_calls_per_step: MAX_CALLS_PER_STEP + 1,
                    ..HarnessOptions::default()
                },
            )
            .await
            .unwrap_err();
        assert!(over_calls.to_string().contains("16"));
    }
}
