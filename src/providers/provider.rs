use crate::load_balancer::tasks::TaskDefinition;
use crate::providers::types::{LlmRequest, LlmResponse, ProviderType};
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::openai::OpenAIProvider;
use crate::providers::ollama::OllamaProvider;
use crate::errors::LlmResult;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use std::time::Duration;
use reqwest::Client;

use super::google::GoogleProvider;
use super::mistral::MistralProvider;

/// Common interface for all LLM providers
/// 
/// This trait defines the interface that all LLM providers must implement
/// to be compatible with the load balancer system.
#[async_trait]
pub trait LlmProvider {
    /// Generate a completion from the LLM provider
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse>;
    /// Get the name of this provider
    fn get_name(&self) -> &str;
    /// Get the currently configured model name
    fn get_model(&self) -> &str;
    /// Get the tasks this provider supports
    fn get_supported_tasks(&self) -> &HashMap<String, TaskDefinition>;
    /// Check if this provider is enabled
    fn is_enabled(&self) -> bool;
}

/// Base provider implementation with common functionality
///
/// Handles common properties and functionality shared across all providers:
/// - HTTP client with timeout
/// - API key storage
/// - Model selection
/// - Task support
/// - Enable/disable status
pub struct BaseProvider {
    name: String,
    client: Client,
    api_key: String,
    model: String,
    supported_tasks: HashMap<String, TaskDefinition>,
    enabled: bool,
}

impl BaseProvider {
    /// Create a new BaseProvider with specified parameters
    ///
    /// # Parameters
    /// * `name` - Provider name identifier
    /// * `api_key` - API key for authentication
    /// * `model` - Default model identifier to use
    /// * `supported_tasks` - Map of tasks this provider supports
    /// * `enabled` - Whether this provider is enabled
    pub fn new(name: String, api_key: String, model: String, supported_tasks: HashMap<String, TaskDefinition>, enabled: bool) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self { name, client, api_key, model, supported_tasks, enabled }
    }

    /// Get the HTTP client instance
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get the API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Get the current model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Check if this provider is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the provider name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the map of supported tasks
    pub fn supported_tasks(&self) -> &HashMap<String, TaskDefinition> {
        &self.supported_tasks
    }
}

/// Factory function to create a provider instance based on type
///
/// # Parameters
/// * `provider_type` - Which provider type to create
/// * `api_key` - API key for authentication
/// * `model` - Default model identifier
/// * `supported_tasks` - List of tasks this provider supports
/// * `enabled` - Whether this provider should be enabled
///
/// # Returns
/// * Arc-wrapped trait object implementing LlmProvider
pub fn create_provider(provider_type: ProviderType, api_key: String, model: String, supported_tasks: Vec<TaskDefinition>, enabled: bool, endpoint_url: Option<String>) -> Arc<dyn LlmProvider + Send + Sync> {
    let supported_tasks: HashMap<String, TaskDefinition> = supported_tasks
        .into_iter()  
        .map(|task| (task.name.clone(), task)) 
        .collect();
    match provider_type {
        ProviderType::Anthropic => Arc::new(AnthropicProvider::new(api_key, model, supported_tasks, enabled)),
        ProviderType::OpenAI => Arc::new(OpenAIProvider::new(api_key, model, supported_tasks, enabled)),
        ProviderType::Mistral => Arc::new(MistralProvider::new(api_key, model, supported_tasks, enabled)),
        ProviderType::Google => Arc::new(GoogleProvider::new(api_key, model, supported_tasks, enabled)),
        ProviderType::Ollama => Arc::new(OllamaProvider::new(api_key, model, supported_tasks, enabled, endpoint_url))
    }
}