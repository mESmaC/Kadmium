use crate::{CondenserError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub is_error: bool,
    /// Provider-specific response content retained for lossless replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_content: Option<ProviderContent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderContent {
    Anthropic(Vec<serde_json::Value>),
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }

    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            is_error: false,
            provider_content: None,
        }
    }

    pub fn tool_result(call: &ToolCall, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(call.id.clone()),
            name: Some(call.name.clone()),
            is_error,
            provider_content: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON is retained across the provider/plugin boundary. Providers that
    /// return parsed objects serialize them once; providers with raw strings do
    /// not needlessly parse and re-emit them.
    pub arguments_json: String,
}

impl ToolCall {
    pub fn arguments(&self) -> Result<serde_json::Value> {
        if self.arguments_json.trim().is_empty() {
            return Ok(serde_json::json!({}));
        }
        Ok(serde_json::from_str(&self.arguments_json)?)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn failure(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub compact_history: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcl4_coe: Option<crate::dsv41_mcl4::Mcl4CoeKnobs>,
}

impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        if self.temperature.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err(CondenserError::Config(
                "temperature must be finite and non-negative".into(),
            ));
        }
        if self
            .top_p
            .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
        {
            return Err(CondenserError::Config(
                "top_p must be in the interval (0, 1]".into(),
            ));
        }
        if self.top_k == Some(0) {
            return Err(CondenserError::Config(
                "top_k must be at least 1 when supplied".into(),
            ));
        }
        if self.max_tokens == Some(0) {
            return Err(CondenserError::Config(
                "max_tokens must be at least 1 when supplied".into(),
            ));
        }
        if self.context_window == Some(0) {
            return Err(CondenserError::Config(
                "context_window must be at least 1 when supplied".into(),
            ));
        }
        if self.stop.iter().any(|value| value.is_empty()) {
            return Err(CondenserError::Config(
                "stop sequences cannot contain an empty string".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub sampling: SamplingParams,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl Usage {
    pub fn add_assign(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            display_name: Some(id.clone()),
            id,
            context_length: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Kadmium,
    #[serde(
        rename = "mcl4-coe",
        alias = "mcl4_coe",
        alias = "mcl4",
        alias = "deepseek-v4.1",
        alias = "dsv41"
    )]
    Mcl4Coe,
    Ollama,
    #[serde(alias = "openai-compatible")]
    OpenAi,
    Anthropic,
}

impl ProviderKind {
    pub const ALL: [Self; 5] = [
        Self::Kadmium,
        Self::Mcl4Coe,
        Self::Ollama,
        Self::OpenAi,
        Self::Anthropic,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Kadmium => "Kadmium",
            Self::Mcl4Coe => "MCL4-COE / DeepSeek-V4.1",
            Self::Ollama => "Ollama",
            Self::OpenAi => "OpenAI-compatible",
            Self::Anthropic => "Anthropic",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::Kadmium => "http://127.0.0.1:7848",
            Self::Mcl4Coe => "http://127.0.0.1:7861",
            Self::Ollama => "http://127.0.0.1:11434",
            Self::OpenAi => "https://api.openai.com",
            Self::Anthropic => "https://api.anthropic.com",
        }
    }

    pub fn default_api_key_env(self) -> &'static str {
        match self {
            Self::Kadmium => "KADMIUM_API_KEY",
            Self::Mcl4Coe => "MCL4_API_KEY",
            Self::Ollama => "OLLAMA_API_KEY",
            Self::OpenAi => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Self::Kadmium => "deepseek-v3-671b-moe-q4k",
            Self::Mcl4Coe => "deepseek-v4.1-flash-1017-nvfp4",
            Self::Ollama => "llama3.2",
            Self::OpenAi | Self::Anthropic => "",
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Kadmium => "kadmium",
            Self::Mcl4Coe => "mcl4-coe",
            Self::Ollama => "ollama",
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        })
    }
}

impl FromStr for ProviderKind {
    type Err = CondenserError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "kadmium" | "kdm" => Ok(Self::Kadmium),
            "mcl4-coe" | "mcl4_coe" | "mcl4" | "deepseek-v4.1" | "deepseek-v4" | "dsv41" => {
                Ok(Self::Mcl4Coe)
            }
            "ollama" | "local" => Ok(Self::Ollama),
            "openai" | "openai-compatible" | "compatible" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            other => Err(CondenserError::Config(format!(
                "unknown provider `{other}`; expected kadmium, mcl4-coe, ollama, openai, or anthropic"
            ))),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

fn default_timeout_secs() -> u64 {
    300
}

impl ProviderConfig {
    pub fn new(kind: ProviderKind) -> Self {
        Self {
            kind,
            model: kind.default_model().to_string(),
            base_url: Some(kind.default_base_url().to_string()),
            api_key: None,
            timeout_secs: default_timeout_secs(),
        }
    }

    pub fn base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or_else(|| self.kind.default_base_url())
    }

    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(CondenserError::Config(format!(
                "a model id is required for {}",
                self.kind.label()
            )));
        }
        self.validate_connection()
    }

    /// Validates the fields needed to contact a provider. Model catalogs use
    /// this alone; completion calls also use `validate` and require a model.
    pub fn validate_connection(&self) -> Result<()> {
        if self.timeout_secs == 0 {
            return Err(CondenserError::Config(
                "timeout_secs must be at least 1".into(),
            ));
        }
        let url = reqwest::Url::parse(self.base_url()).map_err(|error| {
            CondenserError::Config(format!("invalid provider base URL: {error}"))
        })?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(CondenserError::Config(
                "provider base URL must use http or https".into(),
            ));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(CondenserError::Config(
                "put credentials in the API-key field, not in the base URL".into(),
            ));
        }
        if self.kind == ProviderKind::Anthropic
            && self
                .api_key
                .as_deref()
                .is_none_or(|key| key.trim().is_empty())
        {
            return Err(CondenserError::Config(
                "Anthropic requires an API key (use ANTHROPIC_API_KEY or --api-key-env)".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_validation_rejects_invalid_values() {
        assert!(SamplingParams::default().validate().is_ok());
        assert!(
            SamplingParams {
                top_p: Some(0.0),
                ..SamplingParams::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingParams {
                top_k: Some(0),
                ..SamplingParams::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingParams {
                max_tokens: Some(0),
                ..SamplingParams::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn provider_names_and_defaults_are_stable() {
        assert_eq!(
            "openai-compatible".parse::<ProviderKind>().unwrap(),
            ProviderKind::OpenAi
        );
        assert_eq!(
            "kdm".parse::<ProviderKind>().unwrap(),
            ProviderKind::Kadmium
        );
        assert_eq!(
            "mcl4-coe".parse::<ProviderKind>().unwrap(),
            ProviderKind::Mcl4Coe
        );
        assert_eq!(
            "dsv41".parse::<ProviderKind>().unwrap(),
            ProviderKind::Mcl4Coe
        );
        assert_eq!(
            ProviderKind::Kadmium.default_base_url(),
            "http://127.0.0.1:7848"
        );
        assert_eq!(
            ProviderKind::Mcl4Coe.default_base_url(),
            "http://127.0.0.1:7861"
        );
        assert_eq!(
            ProviderKind::Ollama.default_base_url(),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn tool_arguments_preserve_raw_json() {
        let call = ToolCall {
            id: "call-1".into(),
            name: "example__echo".into(),
            arguments_json: "{\"text\":\"hello\"}".into(),
        };
        assert_eq!(call.arguments().unwrap()["text"], "hello");
    }

    #[test]
    fn provider_debug_output_redacts_api_keys() {
        let mut config = ProviderConfig::new(ProviderKind::OpenAi);
        config.api_key = Some("secret-value".into());
        let debug = format!("{config:?}");
        assert!(debug.contains("[redacted]"));
        assert!(!debug.contains("secret-value"));
    }
}
