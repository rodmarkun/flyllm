use std::error::Error;
use std::fmt;
use serde_json;

/// Custom error types for LLM operations
#[derive(Debug)]
pub enum LlmError {
    /// Error from the HTTP client
    RequestError(reqwest::Error),
    /// Error from the API provider
    ApiError(String),
    /// Parsing error
    ParseError(String),
    /// Provider is disabled
    ProviderDisabled(String),
    /// Configuration error
    ConfigError(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::RequestError(err) => write!(f, "Request error: {}", err),
            LlmError::ApiError(msg) => write!(f, "API error: {}", msg),
            LlmError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LlmError::ProviderDisabled(provider) => write!(f, "Provider disabled: {}", provider),
            LlmError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl Error for LlmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            LlmError::RequestError(err) => Some(err),
            _ => None,
        }
    }
}

/// Convert reqwest errors to LlmError
impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        LlmError::RequestError(err)
    }
}

/// Convert serde_json errors to LlmError
impl From<serde_json::Error> for LlmError {
    fn from(err: serde_json::Error) -> Self {
        LlmError::ParseError(err.to_string())
    }
}

/// Result type alias for LLM operations
pub type LlmResult<T> = Result<T, LlmError>;