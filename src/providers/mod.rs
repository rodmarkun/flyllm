/// Module for various LLM provider implementations
///
/// This module contains implementations for different LLM providers:
/// - Anthropic (Claude models)
/// - OpenAI (GPT models)
/// - Mistral AI
/// - Google (Gemini models)
/// - Ollama (local)
/// - LM Studio (local, OpenAI-compatible)
/// - Groq (ultra-fast inference)
/// - Cohere (enterprise LLMs)
/// - Together AI (open-source models)
/// - Perplexity (search-augmented)
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
pub mod lmstudio;
pub mod groq;
pub mod cohere;
pub mod togetherai;
pub mod perplexity;
pub mod model_discovery;

pub use model_discovery::ModelDiscovery;
pub use types::{ProviderType, LlmRequest, LlmResponse, Message, TokenUsage, ModelInfo};
pub use instances::{LlmInstance, create_instance};
pub use anthropic::AnthropicInstance;
pub use openai::OpenAIInstance;
pub use lmstudio::LMStudioInstance;
pub use groq::GroqInstance;
pub use cohere::CohereInstance;
pub use togetherai::TogetherAIInstance;
pub use perplexity::PerplexityInstance;