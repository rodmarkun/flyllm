use serde::{Serialize, Deserialize};
use std::pin::Pin;
use futures::Stream;
use crate::errors::LlmError;

/// Enum representing the different LLM providers supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ProviderType {
    Anthropic,
    OpenAI,
    Mistral,
    Google,
    Ollama,
    LMStudio,
    Groq,
    Cohere,
    TogetherAI,
    Perplexity,
}

/// Unified request structure used across all providers
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmRequest {
    pub messages: Vec<Message>,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

/// Standard message format used across providers
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Unified response structure returned by all providers
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub model: String,
    pub usage: Option<TokenUsage>,
}

/// Token usage information returned by providers
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Default for TokenUsage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0
        }
    }
}

/// Information about an LLM model
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub provider: ProviderType,
}

/// Display implementation for ProviderType
impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::Anthropic => write!(f, "Anthropic"),
            ProviderType::OpenAI => write!(f, "OpenAI"),
            ProviderType::Mistral => write!(f, "Mistral"),
            ProviderType::Google => write!(f, "Google"),
            ProviderType::Ollama => write!(f, "Ollama"),
            ProviderType::LMStudio => write!(f, "LMStudio"),
            ProviderType::Groq => write!(f, "Groq"),
            ProviderType::Cohere => write!(f, "Cohere"),
            ProviderType::TogetherAI => write!(f, "TogetherAI"),
            ProviderType::Perplexity => write!(f, "Perplexity"),
        }
    }
}

impl From<&str> for ProviderType {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "anthropic" => ProviderType::Anthropic,
            "openai" => ProviderType::OpenAI,
            "mistral" => ProviderType::Mistral,
            "google" => ProviderType::Google,
            "ollama" => ProviderType::Ollama,
            "lmstudio" => ProviderType::LMStudio,
            "groq" => ProviderType::Groq,
            "cohere" => ProviderType::Cohere,
            "togetherai" => ProviderType::TogetherAI,
            "perplexity" => ProviderType::Perplexity,
            _ => panic!("Unknown provider: {}", value),
        }
    }
}

/// A chunk of streamed content from an LLM provider
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// The text content of this chunk
    pub content: String,
    /// The model that generated this chunk (may be empty until final chunk)
    pub model: Option<String>,
    /// Whether this is the final chunk in the stream
    pub is_final: bool,
    /// Token usage information (typically only available in final chunk)
    pub usage: Option<TokenUsage>,
}

impl StreamChunk {
    /// Create a new content chunk
    pub fn content(text: impl Into<String>) -> Self {
        Self {
            content: text.into(),
            model: None,
            is_final: false,
            usage: None,
        }
    }

    /// Create a final chunk with usage information
    pub fn final_chunk(model: impl Into<String>, usage: Option<TokenUsage>) -> Self {
        Self {
            content: String::new(),
            model: Some(model.into()),
            is_final: true,
            usage,
        }
    }
}

/// Type alias for a boxed stream of chunks
pub type LlmStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>;
