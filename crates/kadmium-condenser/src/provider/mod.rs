use crate::types::{ChatRequest, ChatResponse, ModelInfo, ProviderConfig, ProviderKind};
use crate::{CondenserError, Result};
use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

mod anthropic;
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

#[async_trait]
pub trait InferenceProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse>;
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
                    ProviderKind::Ollama => {
                        "Runs models already available from a local or remote Ollama daemon."
                    }
                    ProviderKind::OpenAi => {
                        "Works with hosted OpenAI-compatible APIs and local servers such as LM Studio or llama.cpp."
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
            .user_agent(concat!("Condenser/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| CondenserError::provider(config.kind.label(), error.to_string()))?;
        match config.kind {
            ProviderKind::Kadmium | ProviderKind::OpenAi => {
                Ok(Arc::new(openai::OpenAiProvider::new(config, client)))
            }
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

pub(crate) async fn decode_json<T: DeserializeOwned>(
    kind: ProviderKind,
    response: reqwest::Response,
) -> Result<T> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| CondenserError::provider(kind.label(), error.to_string()))?;
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
}
