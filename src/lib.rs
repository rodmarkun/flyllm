//! FlyLLM is a Rust library that provides a load-balanced, multi-provider client for Large Language Models.
//!
//! It enables developers to seamlessly work with multiple LLM providers through a unified API
//! with request routing, load balancing, and failure handling.
//!
//! # Features
//!
//! - **Multi-provider support**: Integrate with OpenAI, Anthropic, Google, Mistral, Ollama,
//!   LM Studio, Groq, Cohere, Together AI, and Perplexity
//! - **Load balancing**: Distribute requests across multiple providers
//! - **Automatic retries**: Handle provider failures with configurable retry policies
//! - **Task routing**: Route specific tasks to the most suitable providers
//! - **Metrics tracking**: Monitor response times, error rates, and token usage
//!
//! # Example
//!
//! ```no_run
//! use flyllm::{LlmManager, ProviderType, GenerationRequest, TaskDefinition};
//!
//! async fn example() {
//!     // Create a manager using the builder pattern
//!     let manager = LlmManager::builder()
//!         .define_task(TaskDefinition::new("chat"))
//!         .add_instance(ProviderType::OpenAI, "gpt-4-turbo", "api-key")
//!         .supports("chat")
//!         .build()
//!         .await
//!         .expect("Failed to build manager");
//!
//!     // Generate a response
//!     let request = GenerationRequest {
//!         prompt: "Explain Rust in one paragraph".to_string(),
//!         task: None,
//!         params: None,
//!     };
//!
//!     let responses = manager.generate_sequentially(vec![request]).await;
//!     println!("{}", responses[0].content);
//! }
//! ```

pub mod providers;
pub mod errors;
pub mod constants;
pub mod load_balancer;
pub mod config;

#[cfg(feature = "metrics")]
pub mod metrics;

pub use providers::{
    ProviderType,
    LlmRequest,
    LlmResponse,
    LlmInstance,
    create_instance,
    AnthropicInstance,
    OpenAIInstance,
    ModelInfo,
    ModelDiscovery,
    StreamChunk,
    LlmStream
};

pub use errors::{LlmError, LlmResult};

pub use load_balancer::{LlmManager, GenerationRequest, LlmManagerResponse, TaskDefinition};

#[cfg(feature = "metrics")]
pub use metrics::describe_metrics;

/// Initialize the logging system
///
/// This should be called at the start of your application in case
/// you want to activate the library's debug and info logging.
pub fn use_logging() {
    env_logger::init();
}