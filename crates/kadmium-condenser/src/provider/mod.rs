use crate::types::{ChatRequest, ChatResponse, ModelInfo, ProviderConfig, ProviderKind};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;

mod anthropic;
mod mcl4_coe;
mod ollama;
mod openai;

static NEXT_SYNTHETIC_TOOL_CALL_ID: AtomicU64 = AtomicU64::new(1);

/// Providers are allowed to omit tool-call IDs. Condenser supplies a
/// process-unique replacement so repeated `call-0` values cannot associate a
/// later run's result with an earlier message in the agent timeline.
pub(crate) fn synthetic_tool_call_id(index: usize) -> String {
    let sequence = NEXT_SYNTHETIC_TOOL_CALL_ID.fetch_add(1, Ordering::Relaxed);
    format!("call-{sequence}-{index}")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedTokenDelta {
    pub delta: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_delta: Option<String>,
    pub generated_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tok_per_s: Option<f64>,
}

pub type TokenStreamSender = mpsc::UnboundedSender<StreamedTokenDelta>;

#[async_trait]
pub trait InferenceProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse>;
    async fn complete_stream(
        &self,
        request: &ChatRequest,
        _token_tx: Option<TokenStreamSender>,
    ) -> Result<ChatResponse> {
        self.complete(request).await
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderDescriptor {
    pub kind: ProviderKind,
    pub label: &'static str,
    pub default_base_url: &'static str,
    pub default_model: &'static str,
    pub default_api_key_env: &'static str,
    pub requires_api_key: bool,
    pub note: &'static str,
}

pub struct ProviderFactory;

impl ProviderFactory {
    pub fn descriptors() -> Vec<ProviderDescriptor> {
        ProviderKind::ALL
            .into_iter()
            .map(|kind| ProviderDescriptor {
                kind,
                label: kind.label(),
                default_base_url: kind.default_base_url(),
                default_model: kind.default_model(),
                default_api_key_env: kind.default_api_key_env(),
                requires_api_key: kind == ProviderKind::Anthropic,
                note: match kind {
                    ProviderKind::Kadmium => {
                        "Uses the local coordinator's current reference diagnostics, not model-scale generation."
                    }
                    ProviderKind::Mcl4Coe => {
                        "Native DeepSeek-V4.1 (NVFP4 .mcl4c / MLA 128+512 / DSML tools) and MCL4-COE (#U$#->#T$#->#A$# / 9-Operator COE) harness."
                    }
                    ProviderKind::Ollama => {
                        "Runs models already available from a local or remote Ollama daemon."
                    }
                    ProviderKind::OpenAi => {
                        "Works with hosted OpenAI-compatible APIs and local servers such as DeepInfer, Strata, FreeToken, Colibri, LM Studio, or llama.cpp."
                    }
                    ProviderKind::Anthropic => {
                        "Uses the Anthropic Messages API with your request-scoped API key."
                    }
                },
            })
            .collect()
    }

    pub fn build(config: ProviderConfig) -> Result<Arc<dyn InferenceProvider>> {
        config.validate_connection()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(30))
            .user_agent(concat!("Condenser/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| CondenserError::provider(config.kind.label(), error.to_string()))?;
        Self::build_with_client(config, client)
    }

    /// Builds a lightweight provider around a caller-owned connection pool.
    /// The request timeout remains per request so providers with different
    /// timeouts can safely share one client.
    pub fn build_with_client(
        config: ProviderConfig,
        client: reqwest::Client,
    ) -> Result<Arc<dyn InferenceProvider>> {
        config.validate_connection()?;
        match config.kind {
            ProviderKind::Kadmium | ProviderKind::OpenAi => {
                Ok(Arc::new(openai::OpenAiProvider::new(config, client)))
            }
            ProviderKind::Mcl4Coe => Ok(Arc::new(mcl4_coe::Mcl4CoeProvider::new(config, client))),
            ProviderKind::Ollama => Ok(Arc::new(ollama::OllamaProvider::new(config, client))),
            ProviderKind::Anthropic => {
                Ok(Arc::new(anthropic::AnthropicProvider::new(config, client)))
            }
        }
    }
}

pub(crate) fn endpoint(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

pub(crate) fn versioned_endpoint(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/v1") {
        endpoint(base, path)
    } else {
        endpoint(base, &format!("v1/{path}"))
    }
}

const MAX_PROVIDER_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

pub(crate) async fn decode_json<T: DeserializeOwned>(
    kind: ProviderKind,
    mut response: reqwest::Response,
) -> Result<T> {
    let status = response.status();
    if response
        .content_length()
        .is_some_and(|length| length > MAX_PROVIDER_RESPONSE_BYTES as u64)
    {
        return Err(CondenserError::provider(
            kind.label(),
            format!("provider response exceeds {MAX_PROVIDER_RESPONSE_BYTES} bytes"),
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| CondenserError::provider(kind.label(), error.to_string()))?
    {
        if bytes.len().saturating_add(chunk.len()) > MAX_PROVIDER_RESPONSE_BYTES {
            return Err(CondenserError::provider(
                kind.label(),
                format!("provider response exceeds {MAX_PROVIDER_RESPONSE_BYTES} bytes"),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        let body = String::from_utf8_lossy(&bytes);
        let message: String = body.chars().take(4_000).collect();
        return Err(CondenserError::provider(
            kind.label(),
            format!("HTTP {status}: {}", message.trim()),
        ));
    }
    serde_json::from_slice(&bytes).map_err(|error| {
        CondenserError::provider(
            kind.label(),
            format!("could not decode provider response: {error}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_accept_host_and_v1_base_urls() {
        assert_eq!(
            versioned_endpoint("http://localhost:1234", "models"),
            "http://localhost:1234/v1/models"
        );
        assert_eq!(
            versioned_endpoint("http://localhost:1234/v1/", "models"),
            "http://localhost:1234/v1/models"
        );
    }

    #[test]
    fn synthetic_tool_call_ids_are_unique_across_turns() {
        let first = synthetic_tool_call_id(0);
        let second = synthetic_tool_call_id(0);
        assert_ne!(first, second);
        assert!(first.starts_with("call-"));
        assert!(second.ends_with("-0"));
    }

    #[tokio::test]
    async fn provider_response_declared_over_limit_is_rejected_before_body_read() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (close_tx, close_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await.unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                MAX_PROVIDER_RESPONSE_BYTES + 1
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            let _ = close_rx.await;
        });
        let response = reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap();
        let error = decode_json::<serde_json::Value>(ProviderKind::OpenAi, response)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("exceeds"));
        let _ = close_tx.send(());
        server.await.unwrap();
    }
}
