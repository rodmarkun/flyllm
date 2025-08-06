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
pub mod instances;
pub mod google;
pub mod mistral;
pub mod ollama;
pub mod model_discovery;

pub use model_discovery::ModelDiscovery;
pub use types::{ProviderType, LlmRequest, LlmResponse, Message, TokenUsage, ModelInfo};
pub use instances::{LlmInstance, create_instance};
pub use anthropic::AnthropicInstance;
pub use openai::OpenAIInstance;