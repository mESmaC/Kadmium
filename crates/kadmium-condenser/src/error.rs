use thiserror::Error;

pub type Result<T, E = CondenserError> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum CondenserError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("{provider} does not support `{option}`")]
    UnsupportedOption { provider: String, option: String },
    #[error("{provider} request failed: {message}")]
    Provider { provider: String, message: String },
    #[error("plugin error: {0}")]
    Plugin(String),
    #[error("tool `{0}` is not registered")]
    UnknownTool(String),
    #[error("the harness reached its {0}-step limit with tool calls still pending")]
    StepLimit(usize),
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

impl CondenserError {
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
        }
    }
}
