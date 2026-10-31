use super::{InferenceProvider, decode_json, endpoint};
use crate::dsv41_mcl4::segment_reasoning_and_answer;
use crate::types::{
    ChatRequest, ChatResponse, Message, ModelInfo, ProviderConfig, ProviderKind, Role, ToolCall,
    Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};

pub(crate) struct OllamaProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl OllamaProvider {
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
impl InferenceProvider for OllamaProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = endpoint(self.config.base_url(), "api/tags");
        let response = self
            .authorize(self.client.get(url))
            .send()
            .await
            .map_err(|error| CondenserError::provider("Ollama", error.to_string()))?;
        let value: Value = decode_json(ProviderKind::Ollama, response).await?;
        let models = value["models"]
            .as_array()
            .ok_or_else(|| CondenserError::provider("Ollama", "missing `models` list"))?;
        Ok(models
            .iter()
            .filter_map(|model| model["model"].as_str().or_else(|| model["name"].as_str()))
            .map(ModelInfo::new)
            .collect())
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.config.validate()?;
        let shown = self
            .authorize(
                self.client
                    .post(endpoint(self.config.base_url(), "api/show"))
                    .json(&json!({"model":request.model})),
            )
            .send()
            .await
            .map_err(|e| CondenserError::provider("Ollama", e.to_string()))?;
        let shown: Value = decode_json(ProviderKind::Ollama, shown).await?;
        let maximum = native_context(&shown)?;
        let mut resolved = request.clone();
        let context = request.sampling.context_window.unwrap_or(maximum);
        if context > maximum {
            return Err(CondenserError::Config(format!(
                "Requested context {context} exceeds model maximum {maximum}"
            )));
        }
        resolved.sampling.context_window = Some(context);
        let body = request_body(&resolved)?;
        let url = endpoint(self.config.base_url(), "api/chat");
        let response = self
            .authorize(self.client.post(url).json(&body))
            .send()
            .await
            .map_err(|error| CondenserError::provider("Ollama", error.to_string()))?;
        let value: Value = decode_json(ProviderKind::Ollama, response).await?;
        parse_chat(value, &request.model)
    }
}

fn native_context(shown: &Value) -> Result<u32> {
    let info = &shown["model_info"];
    let architecture = info["general.architecture"].as_str().unwrap_or("");
    info[format!("{architecture}.context_length")].as_u64()
        .or_else(|| info["context_length"].as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0 && *value <= i32::MAX as u32)
        .ok_or_else(|| CondenserError::Config("Ollama did not report the model's native context length; refusing to use a smaller backend default".into()))
}

fn request_body(request: &ChatRequest) -> Result<Value> {
    request.sampling.validate()?;
    let sampling = &request.sampling;
    let mut options = serde_json::Map::new();
    if let Some(value) = sampling.temperature {
        options.insert("temperature".into(), json!(value));
    }
    if let Some(value) = sampling.top_p {
        options.insert("top_p".into(), json!(value));
    }
    if let Some(value) = sampling.top_k {
        options.insert("top_k".into(), json!(value));
    }
    options.insert(
        "num_predict".into(),
        sampling.max_tokens.map_or(json!(-1), |value| json!(value)),
    );
    if let Some(value) = sampling.context_window {
        options.insert("num_ctx".into(), json!(value));
    }
    if let Some(value) = sampling.seed {
        options.insert("seed".into(), json!(value));
    }
    if !sampling.stop.is_empty() {
        options.insert("stop".into(), json!(sampling.stop));
    }
    Ok(json!({
        "model": request.model,
        "messages": request.messages.iter().map(wire_message).collect::<Vec<_>>(),
        "stream": false,
        "truncate": false,
        "shift": false,
        "tools": request.tools.iter().map(|tool| json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        })).collect::<Vec<_>>(),
        "options": Value::Object(options)
    }))
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
                .map(|call| {
                    let arguments: Value = serde_json::from_str(&call.arguments_json)
                        .unwrap_or_else(|_| json!({"raw": call.arguments_json}));
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": arguments}
                    })
                })
                .collect::<Vec<_>>()
        );
    }
    if let Some(id) = &message.tool_call_id {
        wire["tool_call_id"] = json!(id);
    }
    if let Some(name) = &message.name {
        wire["tool_name"] = json!(name);
    }
    wire
}

fn parse_chat(value: Value, fallback_model: &str) -> Result<ChatResponse> {
    let message = &value["message"];
    if !message.is_object() {
        return Err(CondenserError::provider(
            "Ollama",
            "response has no message object",
        ));
    }
    let raw_content = message["content"].as_str().unwrap_or_default();
    let explicit_thinking = message["thinking"]
        .as_str()
        .or_else(|| message["reasoning_content"].as_str());
    let segmented = segment_reasoning_and_answer(raw_content, explicit_thinking, None);

    let mut tool_calls = segmented.dsml_tool_calls;
    if let Some(calls) = message["tool_calls"].as_array() {
        for (index, call) in calls.iter().enumerate() {
            let function = &call["function"];
            let name = function["name"].as_str().unwrap_or_default().to_string();
            if name.is_empty() {
                continue;
            }
            let arguments_json = match &function["arguments"] {
                Value::String(raw) => raw.clone(),
                Value::Null => "{}".into(),
                other => other.to_string(),
            };
            tool_calls.push(ToolCall {
                id: call["id"]
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| super::synthetic_tool_call_id(index)),
                name,
                arguments_json,
            });
        }
    }

    let input_tokens = value["prompt_eval_count"].as_u64().unwrap_or(0);
    let output_tokens = value["eval_count"].as_u64().unwrap_or(0);
    Ok(ChatResponse {
        model: value["model"]
            .as_str()
            .unwrap_or(fallback_model)
            .to_string(),
        message: Message {
            role: Role::Assistant,
            content: segmented.answer,
            reasoning: segmented.thought,
            tool_calls,
            tool_call_id: None,
            name: None,
            is_error: false,
            provider_content: None,
        },
        finish_reason: value["done_reason"].as_str().map(str::to_string),
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
    use crate::types::SamplingParams;

    #[test]
    fn maps_sampling_to_ollama_options() {
        let request = ChatRequest {
            model: "local".into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams {
                max_tokens: Some(32),
                seed: Some(7),
                top_k: Some(8),
                ..SamplingParams::default()
            },
            tools: Vec::new(),
        };
        let body = request_body(&request).unwrap();
        assert_eq!(body["options"]["num_predict"], 32);
        assert_eq!(body["options"]["seed"], 7);
        assert_eq!(body["options"]["top_k"], 8);
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn normalizes_ollama_tool_arguments_to_raw_json() {
        let parsed = parse_chat(
            json!({
                "model": "local",
                "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "example__echo", "arguments": {"text": "hello"}}}
                ]},
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 2
            }),
            "fallback",
        )
        .unwrap();
        assert_eq!(
            parsed.message.tool_calls[0].arguments_json,
            "{\"text\":\"hello\"}"
        );
        assert_eq!(parsed.usage.total_tokens, 5);
    }
}
