//! Configuration module for TOML-based LlmManager configuration.
//!
//! This module provides the ability to configure FlyLLM using TOML files
//! instead of the builder pattern, making it easier to manage configurations
//! declaratively.
//!
//! # Example Configuration File
//!
//! ```toml
//! [settings]
//! strategy = "lru"
//! max_retries = 3
//!
//! [[tasks]]
//! name = "summary"
//! max_tokens = 500
//!
//! [[providers]]
//! type = "openai"
//! model = "gpt-4-turbo"
//! api_key = "${OPENAI_API_KEY}"
//! tasks = ["summary"]
//! ```
//!
//! # Environment Variables
//!
//! API keys and other sensitive values can reference environment variables
//! using the `${VAR_NAME}` syntax. These are resolved at load time.

mod types;
mod loader;

pub use types::{Config, Settings, TaskConfig, ProviderConfig};
pub use loader::{load_config, parse_config};
