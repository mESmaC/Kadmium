use super::{InferenceProvider, decode_json, endpoint};
use crate::dsv41_mcl4::segment_reasoning_and_answer;
use crate::types::{
    ChatRequest, ChatResponse, Message, ModelInfo, ProviderConfig, ProviderKind, Role, ToolCall,
    Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;

pub(crate) struct OllamaProvider {
    config: ProviderConfig,
    client: reqwest::Client,
    native_ctx_cache: Mutex<HashMap<String, ContextWindow>>,
}

/// What `/api/show` reports about a model's context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ContextWindow {
    /// Largest window a request may ask for.
    maximum: u32,
    /// Window sent when the request does not choose one.
    default: u32,
}

impl OllamaProvider {
    pub fn new(config: ProviderConfig, client: reqwest::Client) -> Self {
        Self {
            config,
            client,
            native_ctx_cache: Mutex::new(HashMap::new()),
        }
    }

    fn authorize(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.config.api_key.as_deref().filter(|key| !key.is_empty()) {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }

    async fn fetch_native_context(&self, model: &str) -> Result<Option<ContextWindow>> {
        if let Some(cached) = self
            .native_ctx_cache
            .lock()
            .expect("ollama native_ctx_cache lock")
            .get(model)
            .copied()
        {
            return Ok(Some(cached));
        }

        let url = endpoint(self.config.base_url(), "api/show");
        let response = self
            .authorize(self.client.post(url).json(&json!({"model": model})))
            .send()
            .await
            .map_err(|error| CondenserError::provider("Ollama", error.to_string()))?;
        let shown: Value = decode_json(ProviderKind::Ollama, response).await?;
        let native = native_context(&shown);
        if let Some(ctx) = native {
            self.native_ctx_cache
                .lock()
                .expect("ollama native_ctx_cache lock")
                .insert(model.to_string(), ctx);
        }
        Ok(native)
    }
}

/// A Modelfile `num_ctx` is an explicit choice (Kadmium's loader writes the owner's
/// window or a confirmed RoPE-scaled one), so it is the default; otherwise the
/// declared context. Either may be requested up to the larger of the two.
fn native_context(shown: &Value) -> Option<ContextWindow> {
    let modelfile = shown
        .get("parameters")
        .and_then(Value::as_str)
        .and_then(|parameters| {
            parameters.lines().find_map(|line| {
                let mut parts = line.split_whitespace();
                (parts.next() == Some("num_ctx"))
                    .then(|| parts.next()?.parse::<u32>().ok())
                    .flatten()
                    .filter(|ctx| *ctx > 0)
            })
        });
    let declared = declared_context(shown);
    let default = modelfile.or(declared)?;
    Some(ContextWindow {
        maximum: declared.map_or(default, |declared| declared.max(default)),
        default,
    })
}

fn declared_context(shown: &Value) -> Option<u32> {
    if let Some(arch) = shown
        .pointer("/model_info/general.architecture")
        .and_then(|v| v.as_str())
    {
        let key = format!("/model_info/{arch}.context_length");
        if let Some(ctx) = shown
            .pointer(&key)
            .and_then(super::parse_positive_u64)
            .and_then(|v| u32::try_from(v).ok())
        {
            return Some(ctx);
        }
    }
    shown
        .pointer("/model_info/context_length")
        .or_else(|| shown.pointer("/details/context_length"))
        .or_else(|| shown.get("context_length"))
        .and_then(super::parse_positive_u64)
        .and_then(|v| u32::try_from(v).ok())
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
            .filter_map(|model| {
                let id = model["model"].as_str().or_else(|| model["name"].as_str())?;
                Some(ModelInfo {
                    id: id.to_string(),
                    display_name: Some(id.to_string()),
                    context_length: super::advertised_context_length(model),
                })
            })
            .collect())
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.config.validate()?;
        request.sampling.validate()?;
        // Ollama otherwise falls back to its small default window for pulled models
        // without a Modelfile `num_ctx`, so the full window is always sent explicitly.
        let native_ctx = self.fetch_native_context(&request.model).await?;
        if let (Some(requested_ctx), Some(window)) = (request.sampling.context_window, native_ctx)
            && requested_ctx > window.maximum
        {
            return Err(CondenserError::Config(format!(
                "requested context_window ({requested_ctx}) exceeds model `{}` native context length ({})",
                request.model, window.maximum
            )));
        }
        let mut resolved = request.clone();
        resolved.sampling.context_window = request
            .sampling
            .context_window
            .or(native_ctx.map(|window| window.default));
        let body = request_body(&resolved)?;
        let url = endpoint(self.config.base_url(), "api/chat");
        let response = self
            .authorize(self.client.post(url).json(&body))
            .send()
            .await
            .map_err(|error| CondenserError::provider("Ollama", error.to_string()))?;
        let value: Value = super::decode_generation_json(ProviderKind::Ollama, response).await?;
        parse_chat(value, &request.model)
    }
}

fn request_body(request: &ChatRequest) -> Result<Value> {
    request.sampling.validate()?;
    let sampling = &request.sampling;
    let mut options = serde_json::Map::new();
    options.insert(
        "num_predict".into(),
        sampling.max_tokens.map_or_else(|| json!(-1), |v| json!(v)),
    );
    if let Some(ctx) = sampling.context_window {
        options.insert("num_ctx".into(), json!(ctx));
    }
    if let Some(value) = sampling.temperature {
        options.insert("temperature".into(), json!(value));
    }
    if let Some(value) = sampling.top_p {
        options.insert("top_p".into(), json!(value));
    }
    if let Some(value) = sampling.top_k {
        options.insert("top_k".into(), json!(value));
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
                context_window: Some(65_536),
                seed: Some(7),
                top_k: Some(8),
                ..SamplingParams::default()
            },
            tools: Vec::new(),
        };
        let body = request_body(&request).unwrap();
        assert_eq!(body["options"]["num_predict"], 32);
        assert_eq!(body["options"]["num_ctx"], 65_536);
        assert_eq!(body["truncate"], false);
        assert_eq!(body["shift"], false);
        assert_eq!(body["options"]["seed"], 7);
        assert_eq!(body["options"]["top_k"], 8);
        assert_eq!(body["stream"], false);

        let default_req = ChatRequest {
            model: "local".into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams::default(),
            tools: Vec::new(),
        };
        let default_body = request_body(&default_req).unwrap();
        assert_eq!(default_body["options"]["num_predict"], -1);
        assert_eq!(default_body["truncate"], false);
        assert_eq!(default_body["shift"], false);

        let shown = json!({
            "model_info": {
                "general.architecture": "qwen2",
                "qwen2.context_length": 131072,
                "llama.context_length": 4096
            }
        });
        assert_eq!(
            native_context(&shown),
            Some(ContextWindow {
                maximum: 131_072,
                default: 131_072
            })
        );
        // An owner-lowered Modelfile window is kept by default but can be raised.
        let lowered = json!({
            "model_info": {"general.architecture": "llama", "llama.context_length": 8192},
            "parameters": "stop \"<|eot|>\"
        num_ctx                        4096"
        });
        assert_eq!(
            native_context(&lowered),
            Some(ContextWindow {
                maximum: 8192,
                default: 4096
            })
        );
        // A RoPE-scaled window baked by the loader exceeds the declared one.
        let scaled = json!({
            "model_info": {"general.architecture": "llama", "llama.context_length": 32768},
            "parameters": "num_ctx 131072"
        });
        assert_eq!(
            native_context(&scaled),
            Some(ContextWindow {
                maximum: 131_072,
                default: 131_072
            })
        );
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
