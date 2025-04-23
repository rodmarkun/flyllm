pub mod instances;
pub mod manager;
pub mod strategies;
pub mod tasks;

pub use manager::{LlmManager, GenerationRequest, LlmManagerResponse};
pub use tasks::TaskDefinition;