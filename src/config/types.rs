//! Configuration types for TOML-based configuration.
//!
//! These types map directly to the TOML configuration file structure.

use serde::Deserialize;

/// Root configuration structure.
#[derive(Debug, Deserialize, Default)]
pub struct Config {
    /// Global settings for the LlmManager.
    #[serde(default)]
    pub settings: Settings,

    /// Task definitions with their parameters.
    #[serde(default)]
    pub tasks: Vec<TaskConfig>,

    /// Provider instance configurations.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

/// Global settings for the LlmManager.
#[derive(Debug, Deserialize)]
pub struct Settings {
    /// Load balancing strategy: "lru", "lowest_latency", or "random".
    #[serde(default = "default_strategy")]
    pub strategy: String,

    /// Maximum number of retries for failed requests.
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,

    /// Optional folder path for debug logging.
    pub debug_folder: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            strategy: default_strategy(),
            max_retries: default_max_retries(),
            debug_folder: None,
        }
    }
}

fn default_strategy() -> String {
    "lru".to_string()
}

fn default_max_retries() -> usize {
    5
}

/// Task definition configuration.
#[derive(Debug, Deserialize)]
pub struct TaskConfig {
    /// Name of the task (used for routing).
    pub name: String,

    /// Maximum tokens for this task.
    pub max_tokens: Option<u32>,

    /// Temperature setting for this task.
    pub temperature: Option<f32>,
}

/// Provider instance configuration.
#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    /// Provider type: "openai", "anthropic", "mistral", etc.
    #[serde(rename = "type")]
    pub provider_type: String,

    /// Model identifier (e.g., "gpt-4-turbo", "claude-3-sonnet-20240229").
    pub model: String,

    /// API key (supports environment variable syntax: "${VAR_NAME}").
    #[serde(default)]
    pub api_key: String,

    /// List of task names this provider supports.
    #[serde(default)]
    pub tasks: Vec<String>,

    /// Whether this provider is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Custom endpoint URL (for Ollama, LM Studio, or self-hosted providers).
    pub endpoint: Option<String>,

    /// Optional name identifier (useful when having multiple instances of the same provider).
    pub name: Option<String>,
}

fn default_true() -> bool {
    true
}
