use super::{InferenceProvider, decode_json, versioned_endpoint};
use crate::types::{
    ChatRequest, ChatResponse, Message, ModelInfo, ProviderConfig, ProviderKind, Role, ToolCall,
    Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub(crate) struct AnthropicProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig, client: reqwest::Client) -> Self {
        Self { config, client }
    }

    fn authorize(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder
            .header(
                "x-api-key",
                self.config.api_key.as_deref().unwrap_or_default(),
            )
            .header("anthropic-version", ANTHROPIC_VERSION)
    }
}

#[async_trait]
impl InferenceProvider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = versioned_endpoint(self.config.base_url(), "models");
        let response = self
            .authorize(self.client.get(url))
            .send()
            .await
            .map_err(|error| CondenserError::provider("Anthropic", error.to_string()))?;
        let value: Value = decode_json(ProviderKind::Anthropic, response).await?;
        let models = value["data"]
            .as_array()
            .ok_or_else(|| CondenserError::provider("Anthropic", "missing `data` model list"))?;
        Ok(models
            .iter()
            .filter_map(|model| {
                model["id"].as_str().map(|id| ModelInfo {
                    id: id.to_string(),
                    display_name: model["display_name"].as_str().map(str::to_string),
                    context_length: None,
                })
            })
            .collect())
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.config.validate()?;
        let body = request_body(request)?;
        let url = versioned_endpoint(self.config.base_url(), "messages");
        let response = self
            .authorize(self.client.post(url).json(&body))
            .send()
            .await
            .map_err(|error| CondenserError::provider("Anthropic", error.to_string()))?;
        let value: Value = decode_json(ProviderKind::Anthropic, response).await?;
        parse_chat(value, &request.model)
    }
}

fn request_body(request: &ChatRequest) -> Result<Value> {
    request.sampling.validate()?;
    if request.sampling.seed.is_some() {
        return Err(CondenserError::UnsupportedOption {
            provider: "Anthropic".into(),
            option: "seed".into(),
        });
    }
    let system = request
        .messages
        .iter()
        .filter(|message| message.role == Role::System)
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let messages = wire_messages(&request.messages);
    if messages.is_empty() {
        return Err(CondenserError::Config(
            "Anthropic needs at least one non-system message".into(),
        ));
    }
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.sampling.max_tokens.unwrap_or(1024)
    });
    let object = body.as_object_mut().expect("request object");
    if !system.is_empty() {
        object.insert("system".into(), json!(system));
    }
    if let Some(value) = request.sampling.temperature {
        object.insert("temperature".into(), json!(value));
    }
    if let Some(value) = request.sampling.top_p {
        object.insert("top_p".into(), json!(value));
    }
    if let Some(value) = request.sampling.top_k {
        object.insert("top_k".into(), json!(value));
    }
    if !request.sampling.stop.is_empty() {
        object.insert("stop_sequences".into(), json!(request.sampling.stop));
    }
    if !request.tools.is_empty() {
        object.insert(
            "tools".into(),
            json!(
                request
                    .tools
                    .iter()
                    .map(|tool| json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    }))
                    .collect::<Vec<_>>()
            ),
        );
    }
    Ok(body)
}

fn wire_messages(messages: &[Message]) -> Vec<Value> {
    let mut output = Vec::new();
    let mut tool_results = Vec::new();
    for message in messages
        .iter()
        .filter(|message| message.role != Role::System)
    {
        if message.role == Role::Tool {
            tool_results.push(wire_message(message)["content"][0].clone());
            continue;
        }
        if !tool_results.is_empty() {
            output.push(json!({"role": "user", "content": std::mem::take(&mut tool_results)}));
        }
        output.push(wire_message(message));
    }
    if !tool_results.is_empty() {
        output.push(json!({"role": "user", "content": tool_results}));
    }
    output
}

fn wire_message(message: &Message) -> Value {
    match message.role {
        Role::Assistant if !message.tool_calls.is_empty() => {
            let mut content = Vec::new();
            if !message.content.is_empty() {
                content.push(json!({"type": "text", "text": message.content}));
            }
            content.extend(message.tool_calls.iter().map(|call| {
                let input: Value = serde_json::from_str(&call.arguments_json)
                    .unwrap_or_else(|_| json!({"raw": call.arguments_json}));
                json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": input
                })
            }));
            json!({"role": "assistant", "content": content})
        }
        Role::Tool => json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
                "is_error": message.is_error
            }]
        }),
        Role::Assistant => json!({"role": "assistant", "content": message.content}),
        Role::User | Role::System => {
            json!({"role": "user", "content": message.content})
        }
    }
}

fn parse_chat(value: Value, fallback_model: &str) -> Result<ChatResponse> {
    let blocks = value["content"]
        .as_array()
        .ok_or_else(|| CondenserError::provider("Anthropic", "response has no content blocks"))?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => text.push_str(block["text"].as_str().unwrap_or_default()),
            Some("tool_use") => {
                tool_calls.push(ToolCall {
                    id: block["id"].as_str().unwrap_or_default().to_string(),
                    name: block["name"].as_str().unwrap_or_default().to_string(),
                    arguments_json: block["input"].to_string(),
                });
            }
            _ => {}
        }
    }
    let input_tokens = value["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = value["usage"]["output_tokens"].as_u64().unwrap_or(0);
    Ok(ChatResponse {
        model: value["model"]
            .as_str()
            .unwrap_or(fallback_model)
            .to_string(),
        message: Message {
            role: Role::Assistant,
            content: text,
            tool_calls,
            tool_call_id: None,
            name: None,
            is_error: false,
        },
        finish_reason: value["stop_reason"].as_str().map(str::to_string),
        usage: Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SamplingParams;

    #[test]
    fn maps_tool_results_to_anthropic_user_blocks() {
        let call = ToolCall {
            id: "tool-1".into(),
            name: "example__echo".into(),
            arguments_json: "{}".into(),
        };
        let wire = wire_message(&Message::tool_result(&call, "done", false));
        assert_eq!(wire["role"], "user");
        assert_eq!(wire["content"][0]["type"], "tool_result");
        assert_eq!(wire["content"][0]["tool_use_id"], "tool-1");
    }

    #[test]
    fn anthropic_requires_max_tokens_but_not_a_client_default_elsewhere() {
        let request = ChatRequest {
            model: "claude-test".into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams::default(),
            tools: Vec::new(),
        };
        let body = request_body(&request).unwrap();
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn parses_text_and_tool_use_blocks() {
        let parsed = parse_chat(json!({
            "model": "claude-test",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "tool-1", "name": "example__echo", "input": {"text": "hello"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 3}
        }), "fallback").unwrap();
        assert_eq!(parsed.message.content, "Checking.");
        assert_eq!(parsed.message.tool_calls.len(), 1);
        assert_eq!(parsed.usage.total_tokens, 8);
    }

    #[test]
    fn groups_parallel_tool_results_into_one_user_message() {
        let first = ToolCall {
            id: "tool-1".into(),
            name: "example__echo".into(),
            arguments_json: "{}".into(),
        };
        let second = ToolCall {
            id: "tool-2".into(),
            name: "example__echo".into(),
            arguments_json: "{}".into(),
        };
        let messages = wire_messages(&[
            Message::user("run both"),
            Message::tool_result(&first, "one", false),
            Message::tool_result(&second, "two", false),
            Message::assistant("finished"),
        ]);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"].as_array().unwrap().len(), 2);
        assert_eq!(messages[1]["content"][0]["tool_use_id"], "tool-1");
        assert_eq!(messages[1]["content"][1]["tool_use_id"], "tool-2");
    }
}
