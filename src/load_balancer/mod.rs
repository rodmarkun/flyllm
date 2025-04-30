/// Load balancer module for distributing requests across multiple LLM providers
///
/// This module contains components for:
/// - Managing provider instances with metrics tracking
/// - Selecting appropriate providers based on tasks and load
/// - Implementing different load balancing strategies
/// - Handling retries and fallbacks when providers fail
/// - Tracking token usage across providers

pub mod instances;
pub mod manager;
pub mod strategies;
pub mod tasks;
pub mod builder;

pub use manager::{LlmManager, GenerationRequest, LlmManagerResponse};
pub use tasks::TaskDefinition;