use std::collections::HashMap;

use crate::load_balancer::tasks::TaskDefinition;
use crate::providers::instances::{LlmInstance, BaseInstance};
use crate::providers::types::{LlmRequest, LlmResponse, LlmStream, StreamChunk, TokenUsage, Message};
use crate::providers::streaming::OpenAIStreamChunk;
use crate::errors::{LlmError, LlmResult};
use crate::constants;

use async_trait::async_trait;
use reqwest::header;
use serde::{Serialize, Deserialize};
use futures::StreamExt;

/// Provider implementation for Groq's API
///
/// Groq provides ultra-fast LLM inference using their LPU hardware.
/// API endpoint: https://api.groq.com/openai/v1/chat/completions
/// Uses OpenAI-compatible API format with Bearer token authentication.
pub struct GroqInstance {
    base: BaseInstance,
}

/// Request structure for Groq's chat completion API (OpenAI-compatible)
#[derive(Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

/// Response structure from Groq's chat completion API
#[derive(Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
    model: String,
    usage: Option<GroqUsage>,
}

/// Individual choice from Groq's response
#[derive(Deserialize)]
struct GroqChoice {
    message: Message,
}

/// Token usage information from Groq
#[derive(Deserialize)]
struct GroqUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl GroqInstance {
    /// Creates a new Groq provider instance
    pub fn new(
        api_key: String,
        model: String,
        supported_tasks: HashMap<String, TaskDefinition>,
        enabled: bool,
    ) -> Self {
        let base = BaseInstance::new("groq".to_string(), api_key, model, supported_tasks, enabled);
        Self { base }
    }

    fn build_headers(&self) -> Result<header::HeaderMap, LlmError> {
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
        Ok(headers)
    }
}

#[async_trait]
impl LlmInstance for GroqInstance {
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("Groq".to_string()));
        }

        let headers = self.build_headers()?;
        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let groq_request = GroqRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: false,
        };

        let response = self.base.client()
            .post(constants::GROQ_API_ENDPOINT)
            .headers(headers)
            .json(&groq_request)
            .send()
            .await?;

        let response_status = response.status();

        if response_status.as_u16() == 429 {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Rate limit exceeded".to_string());
            return Err(LlmError::RateLimit(format!("Groq rate limit: {}", error_text)));
        }

        if !response_status.is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| format!("Unknown error. Status: {}", response_status));
            return Err(LlmError::ApiError(format!("Groq API error: {}", error_text)));
        }

        let groq_response: GroqResponse = response.json().await?;

        if groq_response.choices.is_empty() {
            return Err(LlmError::ApiError("No response from Groq".to_string()));
        }

        let usage = groq_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: groq_response.choices[0].message.content.clone(),
            model: groq_response.model,
            usage,
        })
    }

    async fn generate_stream(&self, request: &LlmRequest) -> LlmResult<LlmStream> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("Groq".to_string()));
        }

        let headers = self.build_headers()?;
        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let groq_request = GroqRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: true,
        };

        let response = self.base.client()
            .post(constants::GROQ_API_ENDPOINT)
            .headers(headers)
            .json(&groq_request)
            .send()
            .await?;

        let response_status = response.status();

        if response_status.as_u16() == 429 {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Rate limit exceeded".to_string());
            return Err(LlmError::RateLimit(format!("Groq rate limit: {}", error_text)));
        }

        if !response_status.is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| format!("Unknown error. Status: {}", response_status));
            return Err(LlmError::ApiError(format!("Groq API error: {}", error_text)));
        }

        let byte_stream = response.bytes_stream();

        let chunk_stream = byte_stream
            .map(|result| result.map_err(|e| LlmError::RequestError(e)))
            .flat_map(|result| {
                match result {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        let chunks: Vec<Result<StreamChunk, LlmError>> = text
                            .lines()
                            .filter_map(|line| {
                                let line = line.trim();
                                if line.starts_with("data: ") {
                                    let data = &line[6..];
                                    if data == "[DONE]" {
                                        return None;
                                    }
                                    match serde_json::from_str::<OpenAIStreamChunk>(data) {
                                        Ok(chunk) => chunk.to_stream_chunk().map(Ok),
                                        Err(e) => Some(Err(LlmError::ParseError(
                                            format!("Failed to parse streaming chunk: {}", e)
                                        ))),
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        futures::stream::iter(chunks)
                    }
                    Err(e) => futures::stream::iter(vec![Err(e)])
                }
            });

        Ok(Box::pin(chunk_stream))
    }

    fn supports_streaming(&self) -> bool {
        true
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
