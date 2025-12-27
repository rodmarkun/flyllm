use std::collections::HashMap;

use crate::load_balancer::tasks::TaskDefinition;
use crate::providers::instances::{LlmInstance, BaseInstance};
use crate::providers::types::{LlmRequest, LlmResponse, TokenUsage, Message};
use crate::errors::{LlmError, LlmResult};
use crate::constants;

use async_trait::async_trait;
use reqwest::header;
use serde::{Serialize, Deserialize};

/// Provider implementation for Together AI's API
///
/// Together AI provides fast inference on 200+ open-source models.
/// API endpoint: https://api.together.xyz/v1/chat/completions
/// Uses OpenAI-compatible API format with Bearer token authentication.
pub struct TogetherAIInstance {
    base: BaseInstance,
}

/// Request structure for Together AI's chat completion API (OpenAI-compatible)
#[derive(Serialize)]
struct TogetherAIRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

/// Response structure from Together AI's chat completion API
#[derive(Deserialize)]
struct TogetherAIResponse {
    choices: Vec<TogetherAIChoice>,
    model: String,
    usage: Option<TogetherAIUsage>,
}

/// Individual choice from Together AI's response
#[derive(Deserialize)]
struct TogetherAIChoice {
    message: Message,
}

/// Token usage information from Together AI
#[derive(Deserialize)]
struct TogetherAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl TogetherAIInstance {
    /// Creates a new Together AI provider instance
    ///
    /// # Parameters
    /// * `api_key` - Together AI API key (required)
    /// * `model` - Default model to use (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    /// * `supported_tasks` - Map of tasks this provider supports
    /// * `enabled` - Whether this provider is enabled
    pub fn new(
        api_key: String,
        model: String,
        supported_tasks: HashMap<String, TaskDefinition>,
        enabled: bool,
    ) -> Self {
        let base = BaseInstance::new("togetherai".to_string(), api_key, model, supported_tasks, enabled);
        Self { base }
    }
}

#[async_trait]
impl LlmInstance for TogetherAIInstance {
    /// Generates a completion using Together AI's API
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("TogetherAI".to_string()));
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

        let together_request = TogetherAIRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: false,
        };

        let response = self
            .base
            .client()
            .post(constants::TOGETHERAI_API_ENDPOINT)
            .headers(headers)
            .json(&together_request)
            .send()
            .await?;

        let response_status = response.status();

        // Check for rate limiting
        if response_status.as_u16() == 429 {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Rate limit exceeded".to_string());
            return Err(LlmError::RateLimit(format!("Together AI rate limit: {}", error_text)));
        }

        if !response_status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| format!("Unknown error. Status: {}", response_status));
            return Err(LlmError::ApiError(format!("Together AI API error: {}", error_text)));
        }

        let together_response: TogetherAIResponse = response.json().await?;

        if together_response.choices.is_empty() {
            return Err(LlmError::ApiError("No response from Together AI".to_string()));
        }

        let usage = together_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: together_response.choices[0].message.content.clone(),
            model: together_response.model,
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
