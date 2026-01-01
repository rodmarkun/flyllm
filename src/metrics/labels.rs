//! Label helpers for consistent metric labeling

use crate::errors::LlmError;
use crate::ProviderType;

/// Standard label keys
pub mod keys {
    /// Provider name label key
    pub const PROVIDER: &str = "provider";
    /// Model name label key
    pub const MODEL: &str = "model";
    /// Task name label key
    pub const TASK: &str = "task";
    /// Error type label key
    pub const ERROR_TYPE: &str = "error_type";
}

/// Convert ProviderType to label value string
pub fn provider_label(provider: ProviderType) -> &'static str {
    match provider {
        ProviderType::Anthropic => "anthropic",
        ProviderType::OpenAI => "openai",
        ProviderType::Mistral => "mistral",
        ProviderType::Google => "google",
        ProviderType::Ollama => "ollama",
        ProviderType::LMStudio => "lmstudio",
        ProviderType::Groq => "groq",
        ProviderType::Cohere => "cohere",
        ProviderType::TogetherAI => "togetherai",
        ProviderType::Perplexity => "perplexity",
    }
}

/// Convert LlmError to error type label string
pub fn error_type_label(error: &LlmError) -> &'static str {
    match error {
        LlmError::RequestError(_) => "request_error",
        LlmError::ApiError(_) => "api_error",
        LlmError::RateLimit(_) => "rate_limit",
        LlmError::ParseError(_) => "parse_error",
        LlmError::ProviderDisabled(_) => "provider_disabled",
        LlmError::ConfigError(_) => "config_error",
    }
}
