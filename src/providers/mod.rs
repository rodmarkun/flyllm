/// Module for various LLM provider implementations
///
/// This module contains implementations for different LLM providers:
/// - Anthropic (Claude models)
/// - OpenAI (GPT models)
/// - Mistral AI
/// - Google (Gemini models)
/// - Ollama
///
/// Each provider implements a common interface for generating text
/// completions through their respective APIs.

pub mod anthropic;
pub mod openai;
pub mod types;
pub mod provider;
pub mod google;
pub mod mistral;
pub mod ollama;

pub use types::{ProviderType, LlmRequest, LlmResponse, Message, TokenUsage};
pub use provider::{LlmProvider, create_provider};
pub use anthropic::AnthropicProvider;
pub use openai::OpenAIProvider;