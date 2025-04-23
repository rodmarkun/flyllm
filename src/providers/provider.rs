use crate::load_balancer::tasks::TaskDefinition;
use crate::providers::types::{LlmRequest, LlmResponse, ProviderType};
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::openai::OpenAIProvider;
use crate::errors::LlmResult;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use std::time::Duration;
use reqwest::Client;

use super::google::GoogleProvider;
use super::mistral::MistralProvider;

#[async_trait]
pub trait LlmProvider {
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse>;
    fn get_name(&self) -> &str;
    fn get_model(&self) -> &str;
    fn get_supported_tasks(&self) -> &HashMap<String, TaskDefinition>;
    fn is_enabled(&self) -> bool;
}

pub struct BaseProvider {
    name: String,
    client: Client,
    api_key: String,
    model: String,
    supported_tasks: HashMap<String, TaskDefinition>,
    enabled: bool,
}

impl BaseProvider {
    pub fn new(name: String, api_key: String, model: String, supported_tasks: HashMap<String, TaskDefinition>, enabled: bool) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self { name, client, api_key, model, supported_tasks, enabled }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn supported_tasks(&self) -> &HashMap<String, TaskDefinition> {
        &self.supported_tasks
    }
}

pub fn create_provider(provider_type: ProviderType, api_key: String, model: String, supported_tasks: Vec<TaskDefinition>, enabled: bool) -> Arc<dyn LlmProvider + Send + Sync> {
    let supported_tasks: HashMap<String, TaskDefinition> = supported_tasks
        .into_iter()  
        .map(|task| (task.name.clone(), task)) 
        .collect();
    match provider_type {
        ProviderType::Anthropic => Arc::new(AnthropicProvider::new(api_key, model, supported_tasks, enabled)),
        ProviderType::OpenAI => Arc::new(OpenAIProvider::new(api_key, model, supported_tasks, enabled)),
        ProviderType::Mistral => Arc::new(MistralProvider::new(api_key, model, supported_tasks, enabled)),
        ProviderType::Google => Arc::new(GoogleProvider::new(api_key, model, supported_tasks, enabled)),
    }
}