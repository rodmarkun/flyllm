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

/// Provider implementation for OpenAI's API (GPT models)
pub struct OpenAIInstance {
    base: BaseInstance,
}

/// Request structure for OpenAI's chat completion API
/// Maps to the format expected by OpenAI's API
#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// Response structure from OpenAI's chat completion API
#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    model: String,
    usage: Option<OpenAIUsage>,
}

/// Individual choice from OpenAI's response
#[derive(Deserialize)]
struct OpenAIChoice {
    message: Message,
}

/// Token usage information from OpenAI
#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl OpenAIInstance {
    /// Creates a new OpenAI provider instance
    ///
    /// # Parameters
    /// * `api_key` - OpenAI API key
    /// * `model` - Default model to use (e.g. "gpt-4-turbo")
    /// * `supported_tasks` - Map of tasks this provider supports
    /// * `enabled` - Whether this provider is enabled
    pub fn new(api_key: String, model: String, supported_tasks: HashMap<String, TaskDefinition>, enabled: bool) -> Self {
        let base = BaseInstance::new("openai".to_string(), api_key, model, supported_tasks, enabled);
        Self { base }
    }

    /// Build request headers for OpenAI API
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
impl LlmInstance for OpenAIInstance {
    /// Generates a completion using OpenAI's API
    ///
    /// # Parameters
    /// * `request` - The LLM request containing messages and parameters
    ///
    /// # Returns
    /// * `LlmResult<LlmResponse>` - The response from the model or an error
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("OpenAI".to_string()));
        }

        let headers = self.build_headers()?;
        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let openai_request = OpenAIRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: None,
            stream_options: None,
        };

        let response = self.base.client()
            .post(constants::OPENAI_API_ENDPOINT)
            .headers(headers)
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("OpenAI API error: {}", error_text)));
        }

        let openai_response: OpenAIResponse = response.json().await?;

        if openai_response.choices.is_empty() {
            return Err(LlmError::ApiError("No response from OpenAI".to_string()));
        }

        let usage = openai_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: openai_response.choices[0].message.content.clone(),
            model: openai_response.model,
            usage,
        })
    }

    /// Generates a streaming completion using OpenAI's API
    async fn generate_stream(&self, request: &LlmRequest) -> LlmResult<LlmStream> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("OpenAI".to_string()));
        }

        let headers = self.build_headers()?;
        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let openai_request = OpenAIRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: Some(true),
            stream_options: Some(StreamOptions { include_usage: true }),
        };

        let response = self.base.client()
            .post(constants::OPENAI_API_ENDPOINT)
            .headers(headers)
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("OpenAI API error: {}", error_text)));
        }

        // Create a stream that processes the SSE response
        let byte_stream = response.bytes_stream();

        let chunk_stream = byte_stream
            .map(|result| {
                result.map_err(|e| LlmError::RequestError(e))
            })
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
                    Err(e) => {
                        futures::stream::iter(vec![Err(e)])
                    }
                }
            });

        Ok(Box::pin(chunk_stream))
    }

    /// Returns whether this provider supports native streaming
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Returns provider name
    fn get_name(&self) -> &str {
        self.base.name()
    }

    /// Returns current model name
    fn get_model(&self) -> &str {
        self.base.model()
    }

    /// Returns supported tasks for this provider
    fn get_supported_tasks(&self) -> &HashMap<String, TaskDefinition> {
        &self.base.supported_tasks()
    }

    /// Returns whether this provider is enabled
    fn is_enabled(&self) -> bool {
        self.base.is_enabled()
    }
}
