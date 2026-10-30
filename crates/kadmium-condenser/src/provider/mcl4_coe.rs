//! Native provider adapter for `MCL4-COE` (`http://127.0.0.1:7861`) and
//! `DeepSeek-V4.1` (`DeepSeek-V4.1-Flash-1017-NVFP4`, `Strata`, `DeepInfer`, `FreeToken`, `Colibri`).
//!
//! Speaks:
//! 1. **`MCL4-COE` Native HTTP API** (`/api/generate`, `/api/status`, `/api/checkpoints`,
//!    `/api/config`, `/api/whiteboard`, `/api/probe/decision`)
//! 2. **DeepSeek-V4.1 OpenAI-compatible Server API** (`/v1/chat/completions` with `DSML` tool
//!    translation, `reasoning_content`, `<think>...</think>`, and `reasoning_effort`)

use super::{
    InferenceProvider, StreamedTokenDelta, TokenStreamSender, endpoint, synthetic_tool_call_id,
    versioned_endpoint,
};
use crate::dsv41_mcl4::{
    Mcl4CoeKnobs, compact_messages_with_tool_pruning, discover_local_dsv41_shards_and_checkpoints,
    segment_reasoning_and_answer,
};
use crate::types::{
    ChatRequest, ChatResponse, Message, ModelInfo, ProviderConfig, ProviderKind, Role, ToolCall,
    Usage,
};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub(crate) struct Mcl4CoeProvider {
    config: ProviderConfig,
    client: Client,
}

impl Mcl4CoeProvider {
    pub fn new(config: ProviderConfig, client: Client) -> Self {
        Self { config, client }
    }

    fn authorize(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.config.api_key.as_deref().filter(|key| !key.is_empty()) {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }

    fn canonical_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "deepseek-v4.1-flash-1017-nvfp4".to_string(),
                display_name: Some(
                    "DeepSeek-V4.1-Flash-1017-NVFP4 (40L · 384E top-6 · MLA 128+512 · .mcl4c)"
                        .to_string(),
                ),
                context_length: Some(131_072),
            },
            ModelInfo {
                id: "deepseek-v4.1-flash-ist-das-2.25bpw".to_string(),
                display_name: Some(
                    "DeepSeek-V4.1-Flash IST-DAS 2.25 bpw (IQ2_XS + 111 GB Engram + DSpark)"
                        .to_string(),
                ),
                context_length: Some(131_072),
            },
            ModelInfo {
                id: "deepseek-v4.1-flash-ist-das-2.00bpw".to_string(),
                display_name: Some(
                    "DeepSeek-V4.1-Flash IST-DAS 2.00 bpw (IQ2_XXS + 111 GB Engram + DSpark)"
                        .to_string(),
                ),
                context_length: Some(131_072),
            },
            ModelInfo {
                id: "deepseek-v4.1-flash-ist-das-1.75bpw".to_string(),
                display_name: Some(
                    "DeepSeek-V4.1-Flash IST-DAS 1.75 bpw (IQ1_M + 111 GB Engram + DSpark)"
                        .to_string(),
                ),
                context_length: Some(131_072),
            },
            ModelInfo {
                id: "mcl4-coe-v7-mitotic-3b".to_string(),
                display_name: Some(
                    "MCL4-COE v7 Mitotic 3B (9-Operator COE + Dream-RL + Whiteboard)".to_string(),
                ),
                context_length: Some(32_768),
            },
            ModelInfo {
                id: "mcl4-coe-v7-balanced-768-e8".to_string(),
                display_name: Some(
                    "MCL4-COE v7 Balanced 768d E8 (top-4 · Tabu 0.35 · Recurrent Pondering)"
                        .to_string(),
                ),
                context_length: Some(16_384),
            },
        ]
    }

    fn parse_native_turn_body(&self, request: &ChatRequest, body: &Value) -> Result<ChatResponse> {
        let turn = body.get("turn").unwrap_or(body);
        let raw_trace = turn
            .get("trace")
            .or_else(|| turn.get("raw_output"))
            .and_then(|v| v.as_str());
        let explicit_thought = turn
            .get("segments")
            .and_then(|s| s.get("thought"))
            .and_then(|v| v.as_str());
        let explicit_answer = turn
            .get("segments")
            .and_then(|s| s.get("answer"))
            .and_then(|v| v.as_str());
        if raw_trace.is_none() && explicit_answer.is_none() {
            return Err(CondenserError::provider(
                self.config.kind.label(),
                "native response is missing generation output",
            ));
        }

        let segmented = segment_reasoning_and_answer(
            explicit_answer.or(raw_trace).unwrap_or_default(),
            explicit_thought,
            None,
        );

        let prompt_toks = turn
            .get("tokens")
            .and_then(|t| t.get("prompt"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let gen_toks = turn
            .get("tokens")
            .and_then(|t| t.get("generated"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let finish_reason = if !segmented.dsml_tool_calls.is_empty() {
            Some("tool_calls".to_string())
        } else {
            Some("stop".to_string())
        };

        Ok(ChatResponse {
            model: request.model.clone(),
            message: Message {
                role: Role::Assistant,
                content: segmented.answer,
                reasoning: segmented.thought,
                tool_calls: segmented.dsml_tool_calls,
                tool_call_id: None,
                name: None,
                is_error: false,
                provider_content: None,
            },
            finish_reason,
            usage: Usage {
                input_tokens: prompt_toks,
                output_tokens: gen_toks,
                total_tokens: prompt_toks + gen_toks,
            },
            telemetry: Some(slim_native_telemetry(turn)),
        })
    }

    async fn try_mcl4_native_generate(
        &self,
        request: &ChatRequest,
        knobs: &Mcl4CoeKnobs,
        compacted_messages: &[Message],
        include_knobs: bool,
        token_tx: Option<TokenStreamSender>,
    ) -> Result<Option<ChatResponse>> {
        let mut payload = native_request_body(request, knobs, compacted_messages, include_knobs);
        if token_tx.is_some() {
            payload["stream"] = json!(true);
        }
        let url = endpoint(self.config.base_url(), "api/generate");
        let mut response = self
            .authorize(self.client.post(url).json(&payload))
            .send()
            .await
            .map_err(|error| {
                CondenserError::provider(self.config.kind.label(), error.to_string())
            })?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }
        let status = response.status();
        let is_event_stream = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.to_ascii_lowercase().starts_with("text/event-stream"));

        if status.is_success() && is_event_stream {
            let mut buffer = String::new();
            let mut final_body: Option<Value> = None;
            while let Some(chunk) = response.chunk().await.map_err(|error| {
                CondenserError::provider(self.config.kind.label(), error.to_string())
            })? {
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);
                    let Some(json_str) = line.strip_prefix("data:").map(str::trim) else {
                        continue;
                    };
                    if json_str.is_empty() {
                        continue;
                    }
                    let Ok(frame) = serde_json::from_str::<Value>(json_str) else {
                        continue;
                    };
                    match frame.get("type").and_then(|v| v.as_str()) {
                        Some("delta") => {
                            if let Some(ref tx) = token_tx {
                                let delta = frame
                                    .get("delta")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string();
                                let reasoning_delta = frame
                                    .get("reasoning_delta")
                                    .and_then(|v| v.as_str())
                                    .filter(|s| !s.is_empty())
                                    .map(str::to_string);
                                let generated_tokens = frame
                                    .get("generated_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    as usize;
                                let tok_per_s = frame.get("tok_per_s").and_then(|v| v.as_f64());
                                let _ = tx.send(StreamedTokenDelta {
                                    delta,
                                    reasoning_delta,
                                    generated_tokens,
                                    tok_per_s,
                                });
                            }
                        }
                        Some("done") => {
                            final_body = Some(frame);
                        }
                        Some("error") => {
                            let err_msg = frame
                                .get("error")
                                .and_then(|v| v.as_str())
                                .unwrap_or("native stream error");
                            return Err(CondenserError::provider(
                                self.config.kind.label(),
                                err_msg.to_string(),
                            ));
                        }
                        _ => {}
                    }
                }
            }
            if let Some(body) = final_body {
                return Ok(Some(self.parse_native_turn_body(request, &body)?));
            }
            return Err(CondenserError::provider(
                self.config.kind.label(),
                "native SSE stream closed before done event",
            ));
        }

        let bytes = response.bytes().await.map_err(|error| {
            CondenserError::provider(self.config.kind.label(), error.to_string())
        })?;
        let body: Value = serde_json::from_slice(&bytes).map_err(|error| {
            CondenserError::provider(
                self.config.kind.label(),
                format!("could not decode native response: {error}"),
            )
        })?;
        if !status.is_success() {
            let err_msg = body
                .get("error")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).trim().to_string());
            if err_msg.contains("No checkpoint loaded") {
                return Ok(None);
            }
            return Err(CondenserError::provider(
                self.config.kind.label(),
                format!("HTTP {status}: {err_msg}"),
            ));
        }

        Ok(Some(self.parse_native_turn_body(request, &body)?))
    }

    async fn try_openai_compatible_generate(
        &self,
        request: &ChatRequest,
        knobs: &Mcl4CoeKnobs,
        compacted_messages: &[Message],
    ) -> Result<ChatResponse> {
        let mut messages_json = Vec::with_capacity(compacted_messages.len());
        for m in compacted_messages {
            let role_str = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            let mut item = json!({
                "role": role_str,
                "content": m.content,
            });
            if let Some(tc_id) = &m.tool_call_id {
                item["tool_call_id"] = json!(tc_id);
            }
            if !m.tool_calls.is_empty() {
                item["tool_calls"] = Value::Array(
                    m.tool_calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": c.arguments_json,
                                }
                            })
                        })
                        .collect(),
                );
            }
            messages_json.push(item);
        }

        let body = openai_request_body(request, knobs, messages_json);
        let url = versioned_endpoint(self.config.base_url(), "chat/completions");
        let response = self
            .authorize(self.client.post(url).json(&body))
            .send()
            .await
            .map_err(|error| {
                CondenserError::provider(self.config.kind.label(), error.to_string())
            })?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(|error| {
            CondenserError::provider(self.config.kind.label(), error.to_string())
        })?;
        if !status.is_success() {
            let message = String::from_utf8_lossy(&bytes);
            return Err(CondenserError::provider(
                self.config.kind.label(),
                format!("HTTP {status}: {}", message.trim()),
            ));
        }
        let payload: Value = serde_json::from_slice(&bytes).map_err(|error| {
            CondenserError::provider(
                self.config.kind.label(),
                format!("could not decode OpenAI-compatible response: {error}"),
            )
        })?;
        let choice = payload
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                CondenserError::provider(
                    self.config.kind.label(),
                    "OpenAI-compatible response has no choices",
                )
            })?;
        let msg_obj = choice.get("message").ok_or_else(|| {
            CondenserError::provider(
                self.config.kind.label(),
                "OpenAI-compatible response choice has no message",
            )
        })?;
        let raw_content = msg_obj
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let explicit_reasoning = msg_obj
            .get("reasoning_content")
            .or_else(|| msg_obj.get("thinking"))
            .or_else(|| msg_obj.get("reasoning"))
            .and_then(|v| v.as_str());

        let segmented = segment_reasoning_and_answer(raw_content, explicit_reasoning, None);
        let mut tool_calls = segmented.dsml_tool_calls;

        if let Some(api_calls) = msg_obj.get("tool_calls").and_then(|v| v.as_array()) {
            for (idx, tc) in api_calls.iter().enumerate() {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| synthetic_tool_call_id(idx));
                let func = tc.get("function").unwrap_or(tc);
                let Some(name) = func.get("name").and_then(|v| v.as_str()) else {
                    continue;
                };
                if name.is_empty() {
                    continue;
                }
                let arguments_json = match func.get("arguments") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Null) | None => "{}".to_string(),
                    Some(other) => other.to_string(),
                };
                tool_calls.push(ToolCall {
                    id,
                    name: name.to_string(),
                    arguments_json,
                });
            }
        }

        let usage_obj = payload.get("usage");
        let input_tokens = usage_obj
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage_obj
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let total_tokens = usage_obj
            .and_then(|u| u.get("total_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(input_tokens + output_tokens);

        Ok(ChatResponse {
            model: payload
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or(&request.model)
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
            finish_reason: choice
                .get("finish_reason")
                .and_then(|v| v.as_str())
                .map(String::from),
            usage: Usage {
                input_tokens,
                output_tokens,
                total_tokens,
            },
            telemetry: None,
        })
    }
}

fn is_native_mcl4_v7_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("mcl4") && model.contains("v7")
}

fn default_knobs_for_model(model: &str) -> Mcl4CoeKnobs {
    if is_native_mcl4_v7_model(model) {
        Mcl4CoeKnobs::mcl4_coe_v7_default()
    } else {
        Mcl4CoeKnobs::deepseek_v41_default()
    }
}

fn unsupported_native_sampling_option(request: &ChatRequest) -> Option<&'static str> {
    if request.sampling.top_p.is_some() {
        Some("top_p")
    } else if request.sampling.top_k.is_some() {
        Some("sampling.top_k (native top_k is reserved for router experts; use mcl4_coe.top_k)")
    } else if request.sampling.seed.is_some() {
        Some("seed")
    } else if !request.sampling.stop.is_empty() {
        Some("stop")
    } else {
        None
    }
}

fn insert_option(object: &mut serde_json::Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        object.insert(key.into(), value);
    }
}

fn native_request_body(
    request: &ChatRequest,
    knobs: &Mcl4CoeKnobs,
    messages: &[Message],
    include_knobs: bool,
) -> Value {
    let mut body = json!({
        "prompt": build_mcl4_prompt(messages, &request.tools),
        "mode": knobs.mode.as_str(),
        "top_k": knobs.top_k,
    });
    let object = body.as_object_mut().expect("native request object");
    insert_option(
        object,
        "max_new_tokens",
        request.sampling.max_tokens.map(Value::from),
    );
    insert_option(
        object,
        "temperature",
        request.sampling.temperature.map(Value::from),
    );

    // Without an explicit MCL4 configuration, preserve the server's inexpensive
    // defaults instead of enabling adaptive recurrence or pager maintenance.
    if include_knobs {
        let overrides = json!({
            "recurrence_steps": knobs.recurrence_steps,
            "adaptive_recurrent_depth": knobs.adaptive_recurrent_depth,
            "max_recurrence_steps": knobs.max_recurrence_steps,
            "recurrent_entropy_threshold": knobs.recurrent_entropy_threshold,
            "depth_ponder": knobs.depth_ponder,
            "recurrent_kv": knobs.recurrent_kv,
            "dense_subset_k": knobs.dense_subset_k,
            "combine_dense_and_sparse": knobs.combine_dense_and_sparse,
            "recurrent_coverage_penalty": knobs.recurrent_coverage_penalty,
            "release_mitotic_pairs": knobs.release_mitotic_pairs,
            "use_whiteboard": knobs.use_whiteboard,
            "use_windowless_pager": knobs.use_windowless_pager,
            "lock_thought": knobs.lock_thought,
        });
        object.extend(
            overrides
                .as_object()
                .expect("native knob object")
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        insert_option(
            object,
            "thought_prefix",
            knobs.thought_prefix.as_ref().map(|value| json!(value)),
        );
        insert_option(
            object,
            "answer_prefix",
            knobs.answer_prefix.as_ref().map(|value| json!(value)),
        );
    }
    body
}

fn openai_request_body(
    request: &ChatRequest,
    knobs: &Mcl4CoeKnobs,
    messages_json: Vec<Value>,
) -> Value {
    let mut body = json!({
        "model": request.model,
        "messages": messages_json,
        "stream": false,
        "reasoning_effort": knobs.reasoning_effort,
        "chat_template_kwargs": {
            "enable_thinking": knobs.reasoning_effort != "none"
        }
    });
    let object = body.as_object_mut().expect("OpenAI request object");
    insert_option(
        object,
        "temperature",
        request.sampling.temperature.map(Value::from),
    );
    insert_option(object, "top_p", request.sampling.top_p.map(Value::from));
    insert_option(object, "top_k", request.sampling.top_k.map(Value::from));
    insert_option(
        object,
        "max_tokens",
        request.sampling.max_tokens.map(Value::from),
    );
    insert_option(object, "seed", request.sampling.seed.map(Value::from));
    if !request.sampling.stop.is_empty() {
        object.insert("stop".into(), json!(request.sampling.stop));
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
                            "parameters": tool.input_schema,
                        }
                    }))
                    .collect::<Vec<_>>()
            ),
        );
    }
    body
}

fn slim_native_telemetry(turn: &Value) -> Value {
    let mut telemetry = serde_json::Map::new();
    for key in ["turn_id", "tokens", "latency_ms", "tok_per_s"] {
        if let Some(value) = turn.get(key) {
            telemetry.insert(key.into(), value.clone());
        }
    }
    if let Some(config) = turn.get("config_snapshot") {
        for key in ["mode", "recurrence_steps", "top_k", "temperature"] {
            if let Some(value) = config.get(key) {
                telemetry.insert(key.into(), value.clone());
            }
        }
    }
    if let Some(verification) = turn.get("verification") {
        for key in ["thought_equation_score", "harness_score"] {
            if let Some(value) = verification.get(key) {
                telemetry.insert(key.into(), value.clone());
            }
        }
    }
    Value::Object(telemetry)
}

fn build_mcl4_prompt(messages: &[Message], tools: &[crate::types::ToolDefinition]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                if !msg.content.trim().is_empty() {
                    parts.push(format!("[System] {}", msg.content.trim()));
                }
            }
            Role::User => {
                if !msg.content.trim().is_empty() {
                    parts.push(format!("#U$# {}", msg.content.trim()));
                }
            }
            Role::Assistant => {
                if !msg.content.trim().is_empty() {
                    parts.push(format!("#A$# {}", msg.content.trim()));
                }
            }
            Role::Tool => {
                let tool_label = msg.name.as_deref().unwrap_or("tool");
                parts.push(format!("#O$# [{tool_label}] {}", msg.content.trim()));
            }
        }
    }
    if !tools.is_empty() {
        let mut declarations = vec![
            "[Available DSML Tools — invoke via <｜DSML｜ calls><｜DSML｜ invoke name=\"TOOL\">...]"
                .to_string(),
        ];
        let mut tools: Vec<_> = tools.iter().collect();
        tools.sort_by(|left, right| left.name.cmp(&right.name));
        declarations.extend(tools.into_iter().map(|tool| {
            format!(
                "[Tool: {}]\nDescription: {}\nArguments JSON Schema: {}",
                tool.name, tool.description, tool.input_schema
            )
        }));
        parts.push(declarations.join("\n"));
    }
    parts.join("\n")
}

#[async_trait]
impl InferenceProvider for Mcl4CoeProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Mcl4Coe
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let mut models = Self::canonical_models();

        // Discover local MCL4DS41 plastic upper-tail RL/RSI checkpoints on X:\MCL4-COE
        let local_inv = discover_local_dsv41_shards_and_checkpoints();
        for ckpt in &local_inv.checkpoints {
            if !models.iter().any(|m| m.id == ckpt.checkpoint_id) {
                let stats_suffix = match (ckpt.star_difficulty, ckpt.pp, ckpt.val_mcqa) {
                    (Some(d), Some(pp), Some(mcqa)) => format!(
                        " (step {} · d*={:.2}★ · {:.1}pp · MCQA {:.2}%)",
                        ckpt.step, d, pp, mcqa
                    ),
                    _ => format!(" (step {})", ckpt.step),
                };
                models.push(ModelInfo {
                    id: ckpt.checkpoint_id.clone(),
                    display_name: Some(format!(
                        "DeepSeek-V4.1 MCL4DS41 Plastic Tail {}{}",
                        ckpt.checkpoint_id, stats_suffix
                    )),
                    context_length: Some(131_072),
                });
            }
        }

        // Check live MCL4-COE `/api/checkpoints` if running
        let url = endpoint(self.config.base_url(), "api/checkpoints");
        if let Ok(resp) = self.authorize(self.client.get(url)).send().await
            && resp.status().is_success()
            && let Ok(payload) = resp.json::<Value>().await
            && let Some(arr) = payload.get("checkpoints").and_then(|v| v.as_array())
        {
            for item in arr.iter().take(12) {
                if let Some(cid) = item.get("checkpoint_id").and_then(|v| v.as_str())
                    && !models.iter().any(|m| m.id == cid)
                {
                    models.push(ModelInfo {
                        id: cid.to_string(),
                        display_name: Some(format!("MCL4-COE Checkpoint: {cid}")),
                        context_length: Some(32_768),
                    });
                }
            }
        }

        Ok(models)
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.complete_stream(request, None).await
    }

    async fn complete_stream(
        &self,
        request: &ChatRequest,
        token_tx: Option<TokenStreamSender>,
    ) -> Result<ChatResponse> {
        self.config.validate()?;
        request.sampling.validate()?;

        let include_knobs = request.sampling.mcl4_coe.is_some();
        let knobs = request
            .sampling
            .mcl4_coe
            .clone()
            .unwrap_or_else(|| default_knobs_for_model(&request.model));

        // Apply `dsh-compaction-basic` + DeepSeek-V4.1 multi-turn thought stripping and old tool pruning
        let (compacted_messages, _) = compact_messages_with_tool_pruning(
            &request.messages,
            knobs.strip_prior_thoughts_in_multiturn,
            knobs.prune_old_tool_results_chars,
        );

        let unsupported_native = unsupported_native_sampling_option(request);
        if is_native_mcl4_v7_model(&request.model)
            && let Some(option) = unsupported_native
        {
            return Err(CondenserError::UnsupportedOption {
                provider: self.config.kind.label().into(),
                option: option.into(),
            });
        }

        // The native API has no fields for generic sampling top-p/top-k, seed,
        // or stop sequences. DeepSeek requests carrying them use OpenAI only.
        if unsupported_native.is_none()
            && let Some(resp) = self
                .try_mcl4_native_generate(
                    request,
                    &knobs,
                    &compacted_messages,
                    include_knobs,
                    token_tx,
                )
                .await?
        {
            return Ok(resp);
        }

        self.try_openai_compatible_generate(request, &knobs, &compacted_messages)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SamplingParams, ToolDefinition};

    fn request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams::default(),
            tools: Vec::new(),
        }
    }

    #[test]
    fn native_defaults_are_model_specific_and_omit_expensive_knobs() {
        let request = request("mcl4-coe-v7-balanced-768-e8");
        let knobs = default_knobs_for_model(&request.model);
        let body = native_request_body(&request, &knobs, &request.messages, false);

        assert_eq!(knobs.top_k, 4);
        assert_eq!(body["top_k"], 4);
        assert!(body.get("adaptive_recurrent_depth").is_none());
        assert!(body.get("use_windowless_pager").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("max_new_tokens").is_none());
        assert_eq!(default_knobs_for_model("deepseek-v4.1-flash").top_k, 6);
    }

    #[test]
    fn native_router_top_k_never_reads_generic_sampling_top_k() {
        let mut request = request("mcl4-coe-v7-mitotic-3b");
        request.sampling.top_k = Some(12);
        let mut knobs = Mcl4CoeKnobs::mcl4_coe_v7_default();
        knobs.top_k = 3;
        let body = native_request_body(&request, &knobs, &request.messages, false);

        assert_eq!(body["top_k"], 3);
        assert_eq!(
            unsupported_native_sampling_option(&request),
            Some(
                "sampling.top_k (native top_k is reserved for router experts; use mcl4_coe.top_k)"
            )
        );
    }

    #[test]
    fn openai_body_forwards_all_supported_sampling_controls() {
        let mut request = request("deepseek-v4.1-flash");
        request.sampling = SamplingParams {
            temperature: Some(0.25),
            top_p: Some(0.8),
            top_k: Some(40),
            max_tokens: Some(128),
            seed: Some(42),
            stop: vec!["DONE".into()],
            mcl4_coe: None,
        };
        let body = openai_request_body(
            &request,
            &Mcl4CoeKnobs::deepseek_v41_default(),
            vec![json!({"role": "user", "content": "hello"})],
        );

        assert!((body["temperature"].as_f64().unwrap() - 0.25).abs() < 1e-6);
        assert!((body["top_p"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["stop"], json!(["DONE"]));
    }

    #[test]
    fn native_tools_are_sorted_and_include_full_declarations() {
        let tools = vec![
            ToolDefinition {
                name: "zeta".into(),
                description: "Last tool".into(),
                input_schema: json!({"type": "object", "properties": {"z": {"type": "number"}}}),
            },
            ToolDefinition {
                name: "alpha".into(),
                description: "First tool".into(),
                input_schema: json!({"type": "object", "required": ["text"]}),
            },
        ];
        let prompt = build_mcl4_prompt(&[], &tools);
        let alpha = prompt.find("[Tool: alpha]").unwrap();
        let zeta = prompt.find("[Tool: zeta]").unwrap();

        assert!(alpha < zeta);
        assert!(prompt.contains("Description: First tool"));
        assert!(prompt.contains("\"required\":[\"text\"]"));
        assert!(prompt.contains("\"z\":{\"type\":\"number\"}"));
    }

    #[test]
    fn native_telemetry_excludes_prompt_traces_and_routing_dumps() {
        let turn = json!({
            "turn_id": 7,
            "prompt": "private prompt",
            "trace": "full trace",
            "tokens": {"prompt": 3, "generated": 2},
            "latency_ms": 12.5,
            "tok_per_s": 8.0,
            "config_snapshot": {"mode": "uta", "top_k": 4, "active_parameters": 1000},
            "verification": {"thought_equation_score": 1.0, "harness": "secret"},
            "expert_routing": {"large": true}
        });
        let telemetry = slim_native_telemetry(&turn);

        assert_eq!(telemetry["turn_id"], 7);
        assert_eq!(telemetry["tokens"]["generated"], 2);
        assert_eq!(telemetry["mode"], "uta");
        assert_eq!(telemetry["thought_equation_score"], 1.0);
        assert!(telemetry.get("prompt").is_none());
        assert!(telemetry.get("trace").is_none());
        assert!(telemetry.get("expert_routing").is_none());
        assert!(telemetry.get("harness").is_none());
    }
}
