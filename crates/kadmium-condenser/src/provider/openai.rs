use super::{InferenceProvider, decode_json, versioned_endpoint};
use crate::types::{
    ChatRequest, ChatResponse, Message, ModelInfo, ProviderConfig, ProviderKind, Role,
    SamplingParams, ToolCall, Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};

pub(crate) struct OpenAiProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl OpenAiProvider {
    pub fn new(config: ProviderConfig, client: reqwest::Client) -> Self {
        Self { config, client }
    }

    fn authorize(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.config.api_key.as_deref().filter(|key| !key.is_empty()) {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }
}

#[async_trait]
impl InferenceProvider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        self.config.kind
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = versioned_endpoint(self.config.base_url(), "models");
        let response = self
            .authorize(self.client.get(url))
            .send()
            .await
            .map_err(|error| {
                CondenserError::provider(self.config.kind.label(), error.to_string())
            })?;
        let value: Value = decode_json(self.config.kind, response).await?;
        parse_models(value)
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.config.validate()?;
        let body = request_body(request, self.config.kind)?;
        let url = versioned_endpoint(self.config.base_url(), "chat/completions");
        let response = self
            .authorize(self.client.post(url).json(&body))
            .send()
            .await
            .map_err(|error| {
                CondenserError::provider(self.config.kind.label(), error.to_string())
            })?;
        let value: Value = decode_json(self.config.kind, response).await?;
        parse_chat(value, &request.model)
    }
}

fn request_body(request: &ChatRequest, kind: ProviderKind) -> Result<Value> {
    request.sampling.validate()?;
    let sampling = &request.sampling;
    if kind == ProviderKind::Kadmium {
        if !request.tools.is_empty() {
            return Err(CondenserError::UnsupportedOption {
                provider: kind.label().into(),
                option: "tools".into(),
            });
        }
        if sampling != &SamplingParams::default() {
            return Err(CondenserError::UnsupportedOption {
                provider: kind.label().into(),
                option: "sampling controls".into(),
            });
        }
        return Ok(json!({
            "model": request.model,
            "messages": request.messages.iter().map(wire_message).collect::<Vec<_>>(),
            "stream": false
        }));
    }

    if sampling.top_k.is_some() {
        return Err(CondenserError::UnsupportedOption {
            provider: kind.label().into(),
            option: "top_k".into(),
        });
    }
    let mut body = json!({
        "model": request.model,
        "messages": request.messages.iter().map(wire_message).collect::<Vec<_>>(),
        "stream": false
    });
    let object = body.as_object_mut().expect("request object");
    insert_option(object, "temperature", sampling.temperature.map(Value::from));
    insert_option(object, "top_p", sampling.top_p.map(Value::from));
    insert_option(object, "max_tokens", sampling.max_tokens.map(Value::from));
    insert_option(object, "seed", sampling.seed.map(Value::from));
    if !sampling.stop.is_empty() {
        object.insert("stop".into(), json!(sampling.stop));
    }
    if !request.tools.is_empty() {
        object.insert(
            "tools".into(),
            json!(
                request
                    .tools
                    .iter()
                    .map(|tool| json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                        }
                    }))
                    .collect::<Vec<_>>()
            ),
        );
    }
    Ok(body)
}

fn insert_option(object: &mut serde_json::Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        object.insert(key.into(), value);
    }
}

fn wire_message(message: &Message) -> Value {
    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut wire = json!({"role": role, "content": message.content});
    if !message.tool_calls.is_empty() {
        wire["tool_calls"] = json!(
            message
                .tool_calls
                .iter()
                .map(|call| json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json}
                }))
                .collect::<Vec<_>>()
        );
    }
    if let Some(id) = &message.tool_call_id {
        wire["tool_call_id"] = json!(id);
    }
    wire
}

fn parse_models(value: Value) -> Result<Vec<ModelInfo>> {
    let models = value["data"].as_array().ok_or_else(|| {
        CondenserError::provider("OpenAI-compatible", "missing `data` model list")
    })?;
    Ok(models
        .iter()
        .filter_map(|model| {
            model["id"]
                .as_str()
                .or_else(|| model["name"].as_str())
                .or_else(|| model["model"].as_str())
        })
        .map(ModelInfo::new)
        .collect())
}

fn parse_chat(value: Value, fallback_model: &str) -> Result<ChatResponse> {
    let choice = value["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .ok_or_else(|| CondenserError::provider("OpenAI-compatible", "response has no choices"))?;
    let message = &choice["message"];
    let content = content_text(&message["content"]);
    let tool_calls = message["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(index, call)| {
                    let function = &call["function"];
                    let arguments_json = match &function["arguments"] {
                        Value::String(raw) => raw.clone(),
                        Value::Null => "{}".into(),
                        other => other.to_string(),
                    };
                    ToolCall {
                        id: call["id"]
                            .as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| super::synthetic_tool_call_id(index)),
                        name: function["name"].as_str().unwrap_or_default().to_string(),
                        arguments_json,
                    }
                })
                .filter(|call| !call.name.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let usage = Usage {
        input_tokens: value["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        output_tokens: value["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        total_tokens: value["usage"]["total_tokens"].as_u64().unwrap_or(0),
    };
    Ok(ChatResponse {
        model: value["model"]
            .as_str()
            .unwrap_or(fallback_model)
            .to_string(),
        message: Message {
            role: Role::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
            is_error: false,
        },
        finish_reason: choice["finish_reason"].as_str().map(str::to_string),
        usage,
    })
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                block["text"]
                    .as_str()
                    .or_else(|| block.as_str())
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SamplingParams, ToolDefinition};

    #[test]
    fn openai_tool_calls_round_trip_raw_arguments() {
        let parsed = parse_chat(
            json!({
                "model": "test-model",
                "choices": [{"message": {"role": "assistant", "content": null,
                    "tool_calls": [{"id": "abc", "type": "function", "function": {
                        "name": "example__echo", "arguments": "{\"text\":\"hello\"}"
                    }}]
                }, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            }),
            "fallback",
        )
        .unwrap();
        assert_eq!(
            parsed.message.tool_calls[0].arguments_json,
            "{\"text\":\"hello\"}"
        );
        assert_eq!(parsed.usage.total_tokens, 6);
    }

    #[test]
    fn kadmium_diagnostics_do_not_silently_accept_tools_or_sampling() {
        let request = ChatRequest {
            model: "diagnostic".into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams::default(),
            tools: vec![ToolDefinition {
                name: "example__echo".into(),
                description: "Echo".into(),
                input_schema: json!({"type": "object"}),
            }],
        };
        assert!(request_body(&request, ProviderKind::Kadmium).is_err());
    }

    #[test]
    fn parses_content_block_text() {
        assert_eq!(
            content_text(&json!([{"type": "text", "text": "hello"}])),
            "hello"
        );
    }
}
