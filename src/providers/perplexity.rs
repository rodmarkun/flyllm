use std::collections::HashMap;

use crate::load_balancer::tasks::TaskDefinition;
use crate::providers::instances::{LlmInstance, BaseInstance};
use crate::providers::types::{LlmRequest, LlmResponse, TokenUsage, Message};
use crate::errors::{LlmError, LlmResult};
use crate::constants;

use async_trait::async_trait;
use reqwest::header;
use serde::{Serialize, Deserialize};

/// Provider implementation for Perplexity AI's API
///
/// Perplexity provides LLMs with built-in web search capabilities and citations.
/// API endpoint: https://api.perplexity.ai/chat/completions
/// Uses OpenAI-compatible API format with Bearer token authentication.
///
/// Available models: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro
pub struct PerplexityInstance {
    base: BaseInstance,
}

/// Request structure for Perplexity's chat completion API
#[derive(Serialize)]
struct PerplexityRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

/// Response structure from Perplexity's chat completion API
#[derive(Deserialize)]
struct PerplexityResponse {
    choices: Vec<PerplexityChoice>,
    model: String,
    usage: Option<PerplexityUsage>,
}

/// Individual choice from Perplexity's response
#[derive(Deserialize)]
struct PerplexityChoice {
    message: Message,
}

/// Token usage information from Perplexity
#[derive(Deserialize)]
struct PerplexityUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl PerplexityInstance {
    /// Creates a new Perplexity provider instance
    ///
    /// # Parameters
    /// * `api_key` - Perplexity API key (required)
    /// * `model` - Default model to use (e.g., "sonar", "sonar-pro", "sonar-reasoning")
    /// * `supported_tasks` - Map of tasks this provider supports
    /// * `enabled` - Whether this provider is enabled
    pub fn new(
        api_key: String,
        model: String,
        supported_tasks: HashMap<String, TaskDefinition>,
        enabled: bool,
    ) -> Self {
        let base = BaseInstance::new("perplexity".to_string(), api_key, model, supported_tasks, enabled);
        Self { base }
    }
}

#[async_trait]
impl LlmInstance for PerplexityInstance {
    /// Generates a completion using Perplexity's API
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("Perplexity".to_string()));
        }

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", self.base.api_key()))
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format: {}", e)))?,
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let perplexity_request = PerplexityRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: false,
        };

        let response = self
            .base
            .client()
            .post(constants::PERPLEXITY_API_ENDPOINT)
            .headers(headers)
            .json(&perplexity_request)
            .send()
            .await?;

        let response_status = response.status();

        // Check for rate limiting
        if response_status.as_u16() == 429 {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Rate limit exceeded".to_string());
            return Err(LlmError::RateLimit(format!("Perplexity rate limit: {}", error_text)));
        }

        if !response_status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| format!("Unknown error. Status: {}", response_status));
            return Err(LlmError::ApiError(format!("Perplexity API error: {}", error_text)));
        }

        let perplexity_response: PerplexityResponse = response.json().await?;

        if perplexity_response.choices.is_empty() {
            return Err(LlmError::ApiError("No response from Perplexity".to_string()));
        }

        let usage = perplexity_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: perplexity_response.choices[0].message.content.clone(),
            model: perplexity_response.model,
            usage,
        })
    }

    fn get_name(&self) -> &str {
        self.base.name()
    }

    fn get_model(&self) -> &str {
        self.base.model()
    }

    fn get_supported_tasks(&self) -> &HashMap<String, TaskDefinition> {
        self.base.supported_tasks()
    }

    fn is_enabled(&self) -> bool {
        self.base.is_enabled()
    }
}
