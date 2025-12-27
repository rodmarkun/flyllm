use std::collections::HashMap;

use crate::load_balancer::tasks::TaskDefinition;
use crate::providers::instances::{LlmInstance, BaseInstance};
use crate::providers::types::{LlmRequest, LlmResponse, TokenUsage, Message};
use crate::errors::{LlmError, LlmResult};
use crate::constants;

use async_trait::async_trait;
use reqwest::header;
use serde::{Serialize, Deserialize};

/// Provider implementation for Cohere's API (v2)
///
/// Cohere provides enterprise-grade LLMs with RAG and tool use capabilities.
/// API endpoint: https://api.cohere.com/v2/chat
/// Uses Bearer token authentication.
pub struct CohereInstance {
    base: BaseInstance,
}

/// Message format for Cohere v2 API
#[derive(Serialize, Clone)]
struct CohereMessage {
    role: String,
    content: String,
}

/// Request structure for Cohere's v2 chat API
#[derive(Serialize)]
struct CohereRequest {
    model: String,
    messages: Vec<CohereMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

/// Response structure from Cohere's v2 chat API
#[derive(Deserialize, Debug)]
struct CohereResponse {
    message: CohereResponseMessage,
    #[serde(default)]
    usage: Option<CohereUsage>,
}

/// Response message from Cohere
#[derive(Deserialize, Debug)]
struct CohereResponseMessage {
    role: String,
    content: Vec<CohereContentBlock>,
}

/// Content block in Cohere response
#[derive(Deserialize, Debug)]
struct CohereContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
}

/// Token usage information from Cohere
#[derive(Deserialize, Debug)]
struct CohereUsage {
    #[serde(default)]
    billed_units: Option<CohereBilledUnits>,
    #[serde(default)]
    tokens: Option<CohereTokens>,
}

/// Billed units from Cohere (alternative token counting)
#[derive(Deserialize, Debug)]
struct CohereBilledUnits {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
}

/// Token counts from Cohere
#[derive(Deserialize, Debug)]
struct CohereTokens {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
}

impl CohereInstance {
    /// Creates a new Cohere provider instance
    ///
    /// # Parameters
    /// * `api_key` - Cohere API key (required)
    /// * `model` - Default model to use (e.g., "command-r-plus", "command-r")
    /// * `supported_tasks` - Map of tasks this provider supports
    /// * `enabled` - Whether this provider is enabled
    pub fn new(
        api_key: String,
        model: String,
        supported_tasks: HashMap<String, TaskDefinition>,
        enabled: bool,
    ) -> Self {
        let base = BaseInstance::new("cohere".to_string(), api_key, model, supported_tasks, enabled);
        Self { base }
    }

    /// Convert internal Message format to Cohere's message format
    fn convert_messages(messages: &[Message]) -> Vec<CohereMessage> {
        messages
            .iter()
            .map(|m| CohereMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect()
    }

    /// Extract text content from Cohere's response content blocks
    fn extract_content(content_blocks: &[CohereContentBlock]) -> String {
        content_blocks
            .iter()
            .filter_map(|block| {
                if block.content_type == "text" {
                    block.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join("")
    }
}

#[async_trait]
impl LlmInstance for CohereInstance {
    /// Generates a completion using Cohere's v2 API
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("Cohere".to_string()));
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
        headers.insert(
            header::ACCEPT,
            header::HeaderValue::from_static("application/json"),
        );

        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let cohere_request = CohereRequest {
            model: model.clone(),
            messages: Self::convert_messages(&request.messages),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: false,
        };

        let response = self
            .base
            .client()
            .post(constants::COHERE_API_ENDPOINT)
            .headers(headers)
            .json(&cohere_request)
            .send()
            .await?;

        let response_status = response.status();

        // Check for rate limiting
        if response_status.as_u16() == 429 {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Rate limit exceeded".to_string());
            return Err(LlmError::RateLimit(format!("Cohere rate limit: {}", error_text)));
        }

        if !response_status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| format!("Unknown error. Status: {}", response_status));
            return Err(LlmError::ApiError(format!("Cohere API error: {}", error_text)));
        }

        let response_text = response.text().await?;
        if response_text.is_empty() {
            return Err(LlmError::ApiError(
                "Received empty response body from Cohere".to_string(),
            ));
        }

        let cohere_response: CohereResponse = serde_json::from_str(&response_text)
            .map_err(|e| {
                LlmError::ApiError(format!(
                    "Failed to parse Cohere JSON response: {}. Body: {}",
                    e, response_text
                ))
            })?;

        // Extract text content from response
        let content = Self::extract_content(&cohere_response.message.content);

        // Map token usage - Cohere v2 uses different structures
        let usage = cohere_response.usage.and_then(|u| {
            // Try tokens first, then billed_units
            if let Some(tokens) = u.tokens {
                let input = tokens.input_tokens.unwrap_or(0);
                let output = tokens.output_tokens.unwrap_or(0);
                Some(TokenUsage {
                    prompt_tokens: input,
                    completion_tokens: output,
                    total_tokens: input + output,
                })
            } else if let Some(billed) = u.billed_units {
                let input = billed.input_tokens.unwrap_or(0);
                let output = billed.output_tokens.unwrap_or(0);
                Some(TokenUsage {
                    prompt_tokens: input,
                    completion_tokens: output,
                    total_tokens: input + output,
                })
            } else {
                None
            }
        });

        Ok(LlmResponse {
            content,
            model,
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
