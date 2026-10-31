use super::{InferenceProvider, decode_json, versioned_endpoint};
use crate::dsv41_mcl4::segment_reasoning_and_answer;
use crate::types::{
    ChatRequest, ChatResponse, Message, ModelInfo, ProviderConfig, ProviderContent, ProviderKind,
    Role, ToolCall, Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub(crate) struct AnthropicProvider {
    config: ProviderConfig,
    client: reqwest::Client,
    max_output_cache: Mutex<HashMap<String, u32>>,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig, client: reqwest::Client) -> Self {
        Self {
            config,
            client,
            max_output_cache: Mutex::new(HashMap::new()),
        }
    }

    fn authorize(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder
            .header(
                "x-api-key",
                self.config.api_key.as_deref().unwrap_or_default(),
            )
            .header("anthropic-version", ANTHROPIC_VERSION)
    }

    async fn resolve_max_tokens(&self, model: &str) -> Result<u32> {
        if let Some(cached) = self
            .max_output_cache
            .lock()
            .expect("anthropic max_output_cache lock")
            .get(model)
            .copied()
        {
            return Ok(cached);
        }

        let encoded_model = encode_path_segment(model);
        let url = versioned_endpoint(self.config.base_url(), &format!("models/{encoded_model}"));
        let response = self
            .authorize(self.client.get(url))
            .send()
            .await
            .map_err(|error| CondenserError::provider("Anthropic", error.to_string()))?;
        let value: Value = decode_json(ProviderKind::Anthropic, response).await?;
        let resolved = extract_anthropic_max_output(&value).ok_or_else(|| {
            CondenserError::Config(format!(
                "Anthropic model `{model}` did not advertise `max_output_tokens` or `max_tokens`; supply `sampling.max_tokens` explicitly"
            ))
        })?;
        self.max_output_cache
            .lock()
            .expect("anthropic max_output_cache lock")
            .insert(model.to_string(), resolved);
        Ok(resolved)
    }
}

fn extract_anthropic_max_output(model: &Value) -> Option<u32> {
    super::maximum_output(model).or_else(|| {
        model
            .get("max_tokens")
            .and_then(super::parse_positive_u64)
            .and_then(|v| u32::try_from(v).ok())
    })
}

fn encode_path_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b':') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
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
        let mut cache = self
            .max_output_cache
            .lock()
            .expect("anthropic max_output_cache lock");
        Ok(models
            .iter()
            .filter_map(|model| {
                let id = model["id"].as_str()?;
                if let Some(max_out) = extract_anthropic_max_output(model) {
                    cache.insert(id.to_string(), max_out);
                }
                Some(ModelInfo {
                    id: id.to_string(),
                    display_name: model["display_name"].as_str().map(str::to_string),
                    context_length: super::advertised_context_length(model),
                })
            })
            .collect())
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.config.validate()?;
        request.sampling.validate()?;
        super::reject_context_override(request, "Anthropic")?;
        let max_tokens = match request.sampling.max_tokens {
            Some(tokens) => tokens,
            None => self.resolve_max_tokens(&request.model).await?,
        };
        let body = request_body(request, max_tokens)?;
        let url = versioned_endpoint(self.config.base_url(), "messages");
        let mut req_builder = self.authorize(self.client.post(url).json(&body));
        // Only lengthens an owner-set timeout; `0` (the default) stays unlimited.
        if self.config.timeout_secs != 0 && max_tokens > 16_384 {
            req_builder = req_builder.timeout(std::time::Duration::from_secs(
                self.config.timeout_secs.max(1800),
            ));
        }
        let response = req_builder
            .send()
            .await
            .map_err(|error| CondenserError::provider("Anthropic", error.to_string()))?;
        let value: Value = super::decode_generation_json(ProviderKind::Anthropic, response).await?;
        parse_chat(value, &request.model)
    }
}

fn request_body(request: &ChatRequest, max_tokens: u32) -> Result<Value> {
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
        "max_tokens": max_tokens
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
            let content = match &message.provider_content {
                Some(ProviderContent::Anthropic(blocks)) => json!(blocks),
                None => {
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
                    Value::Array(content)
                }
            };
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
    let provider_content = Some(ProviderContent::Anthropic(blocks.clone()));
    let mut text = String::new();
    let mut thinking_text = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => text.push_str(block["text"].as_str().unwrap_or_default()),
            Some("thinking") => {
                thinking_text.push_str(block["thinking"].as_str().unwrap_or_default());
            }
            Some("tool_use") => {
                let id = block["id"]
                    .as_str()
                    .filter(|id| !id.trim().is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| super::synthetic_tool_call_id(tool_calls.len()));
                tool_calls.push(ToolCall {
                    id,
                    name: block["name"].as_str().unwrap_or_default().to_string(),
                    arguments_json: block["input"].to_string(),
                });
            }
            _ => {}
        }
    }
    let explicit_thinking = if thinking_text.trim().is_empty() {
        None
    } else {
        Some(thinking_text.as_str())
    };
    let segmented = segment_reasoning_and_answer(&text, explicit_thinking, None);
    let mut merged_calls = segmented.dsml_tool_calls;
    merged_calls.extend(tool_calls);

    let input_tokens = value["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = value["usage"]["output_tokens"].as_u64().unwrap_or(0);
    Ok(ChatResponse {
        model: value["model"]
            .as_str()
            .unwrap_or(fallback_model)
            .to_string(),
        message: Message {
            role: Role::Assistant,
            content: segmented.answer,
            reasoning: segmented.thought,
            tool_calls: merged_calls,
            tool_call_id: None,
            name: None,
            is_error: false,
            provider_content,
        },
        finish_reason: value["stop_reason"].as_str().map(str::to_string),
        usage: Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        },
        telemetry: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SamplingParams, ToolDefinition};

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
    fn anthropic_requires_resolved_output_limit() {
        let request = ChatRequest {
            model: "claude-test".into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams::default(),
            tools: Vec::new(),
        };
        let model_meta = json!({
            "id": "claude-test",
            "max_output_tokens": 8192,
            "max_input_tokens": 200000
        });
        let max_out = extract_anthropic_max_output(&model_meta).unwrap();
        assert_eq!(max_out, 8192);
        assert_eq!(
            super::super::advertised_context_length(&model_meta),
            Some(200_000)
        );
        let body = request_body(&request, max_out).unwrap();
        assert_eq!(body["max_tokens"], 8192);
    }

    #[test]
    fn tool_declarations_use_exact_anthropic_schema() {
        let request = ChatRequest {
            model: "claude-test".into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams {
                max_tokens: Some(64000),
                ..Default::default()
            },
            tools: vec![ToolDefinition {
                name: "example__echo".into(),
                description: "Echo text".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }),
            }],
        };
        let body = request_body(&request, 4096).unwrap();
        assert_eq!(
            body["tools"],
            json!([{
                "name": "example__echo",
                "description": "Echo text",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            }])
        );
    }

    #[test]
    fn parses_text_and_tool_use_blocks() {
        let parsed = parse_chat(
            json!({
                "model": "claude-test",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "tool-1", "name": "example__echo", "input": {"text": "hello"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 3}
            }),
            "fallback",
        )
        .unwrap();
        assert_eq!(parsed.message.content, "Checking.");
        assert_eq!(parsed.message.tool_calls.len(), 1);
        assert_eq!(parsed.usage.total_tokens, 8);
    }

    #[test]
    fn preserves_signed_redacted_and_unknown_blocks_for_tool_replay() {
        let content = json!([
            {"type": "thinking", "thinking": "Check the input.", "signature": "signed-value"},
            {"type": "redacted_thinking", "data": "encrypted-value"},
            {"type": "future_block", "payload": {"arbitrary": [1, 2, 3]}},
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "tool-1", "name": "example__echo", "input": {"text": "hello"}}
        ]);
        let parsed = parse_chat(
            json!({
                "model": "claude-test",
                "content": content.clone(),
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 3}
            }),
            "fallback",
        )
        .unwrap();
        assert_eq!(parsed.message.content, "Checking.");
        assert_eq!(
            parsed.message.reasoning.as_deref(),
            Some("Check the input.")
        );
        assert_eq!(parsed.message.tool_calls.len(), 1);
        assert_eq!(
            parsed.message.provider_content,
            Some(ProviderContent::Anthropic(
                content.as_array().unwrap().clone()
            ))
        );

        let serialized = serde_json::to_string(&parsed.message).unwrap();
        let replay: Message = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            wire_message(&replay),
            json!({"role": "assistant", "content": content})
        );
    }

    #[test]
    fn synthesizes_missing_and_blank_tool_call_ids() {
        let parsed = parse_chat(
            json!({
                "model": "claude-test",
                "content": [
                    {"type": "tool_use", "name": "first", "input": {}},
                    {"type": "tool_use", "id": "  ", "name": "second", "input": {}}
                ],
                "usage": {}
            }),
            "fallback",
        )
        .unwrap();
        assert_eq!(parsed.message.tool_calls.len(), 2);
        assert!(!parsed.message.tool_calls[0].id.trim().is_empty());
        assert!(!parsed.message.tool_calls[1].id.trim().is_empty());
        assert_ne!(
            parsed.message.tool_calls[0].id,
            parsed.message.tool_calls[1].id
        );
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
    }
}
