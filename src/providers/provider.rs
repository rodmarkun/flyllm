use crate::providers::types::{LlmRequest, LlmResponse, ProviderType};
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::openai::OpenAIProvider;
use crate::errors::LlmResult;
use std::sync::Arc;

use async_trait::async_trait;
use std::time::Duration;
use reqwest::Client;

#[async_trait]
pub trait LlmProvider {
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse>;
    fn get_name(&self) -> &str;
    fn get_model(&self) -> &str;
    fn is_enabled(&self) -> bool;
}

pub struct BaseProvider {
    client: Client,
    api_key: String,
    model: String,
    enabled: bool,
    name: String,
}

impl BaseProvider {
    pub fn new(name: String, api_key: String, model: String, enabled: bool) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, api_key, model, enabled, name }
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
}

pub fn create_provider(provider_type: ProviderType, api_key: String, model: String, enabled: bool) -> Arc<dyn LlmProvider + Send + Sync> {
    match provider_type {
        ProviderType::Anthropic => Arc::new(AnthropicProvider::new(api_key, model, enabled)),
        ProviderType::OpenAI => Arc::new(OpenAIProvider::new(api_key, model, enabled)),
    }
}