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
use url::Url;
use futures::StreamExt;

/// Provider implementation for LM Studio (OpenAI-compatible local server)
///
/// LM Studio runs a local server that exposes an OpenAI-compatible API.
/// Default endpoint: http://localhost:1234/v1/chat/completions
pub struct LMStudioInstance {
    base: BaseInstance,
    endpoint_url: String,
}

/// Request structure for LM Studio's chat completion API (OpenAI-compatible)
#[derive(Serialize)]
struct LMStudioRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

/// Response structure from LM Studio's chat completion API
#[derive(Deserialize)]
struct LMStudioResponse {
    choices: Vec<LMStudioChoice>,
    model: String,
    usage: Option<LMStudioUsage>,
}

/// Individual choice from LM Studio's response
#[derive(Deserialize)]
struct LMStudioChoice {
    message: Message,
}

/// Token usage information from LM Studio
#[derive(Deserialize)]
struct LMStudioUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl LMStudioInstance {
    /// Creates a new LM Studio provider instance
    ///
    /// # Parameters
    /// * `api_key` - Optional API key (LM Studio typically doesn't require one, but can be configured)
    /// * `model` - Default model to use (model name loaded in LM Studio)
    /// * `supported_tasks` - Map of tasks this provider supports
    /// * `enabled` - Whether this provider is enabled
    /// * `endpoint_url` - Optional custom endpoint URL. If None, uses default localhost:1234
    pub fn new(
        api_key: String,
        model: String,
        supported_tasks: HashMap<String, TaskDefinition>,
        enabled: bool,
        endpoint_url: Option<String>,
    ) -> Self {
        let base_endpoint = endpoint_url.unwrap_or_else(|| constants::LMSTUDIO_API_ENDPOINT.to_string());

        // Validate and ensure the path ends correctly
        let final_endpoint = match Url::parse(&base_endpoint) {
            Ok(mut url) => {
                if !url.path().ends_with("/v1/chat/completions") && !url.path().ends_with("/chat/completions") {
                    if url.path() == "/" || url.path().is_empty() {
                        url.set_path("/v1/chat/completions");
                    } else {
                        let current_path = url.path().trim_end_matches('/');
                        if !current_path.ends_with("/v1") {
                            url.set_path(&format!("{}/v1/chat/completions", current_path));
                        } else {
                            url.set_path(&format!("{}/chat/completions", current_path));
                        }
                    }
                }
                url.to_string()
            }
            Err(_) => {
                eprintln!(
                    "Warning: Invalid LM Studio endpoint URL '{}' provided. Falling back to default: {}",
                    base_endpoint, constants::LMSTUDIO_API_ENDPOINT
                );
                constants::LMSTUDIO_API_ENDPOINT.to_string()
            }
        };

        let base = BaseInstance::new("lmstudio".to_string(), api_key, model, supported_tasks, enabled);

        Self {
            base,
            endpoint_url: final_endpoint,
        }
    }
}

#[async_trait]
impl LlmInstance for LMStudioInstance {
    /// Generates a completion using LM Studio's OpenAI-compatible API
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("LMStudio".to_string()));
        }

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        // Add Authorization header if an API key is provided (optional for LM Studio)
        if !self.base.api_key().is_empty() {
            match header::HeaderValue::from_str(&format!("Bearer {}", self.base.api_key())) {
                Ok(val) => {
                    headers.insert(header::AUTHORIZATION, val);
                }
                Err(e) => {
                    return Err(LlmError::ConfigError(format!(
                        "Invalid API key format for LM Studio: {}",
                        e
                    )))
                }
            }
        }

        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let lmstudio_request = LMStudioRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: false,
        };

        let response = self
            .base
            .client()
            .post(&self.endpoint_url)
            .headers(headers)
            .json(&lmstudio_request)
            .send()
            .await?;

        let response_status = response.status();
        if !response_status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| format!("Unknown error. Status: {}", response_status));
            return Err(LlmError::ApiError(format!("LM Studio API error: {}", error_text)));
        }

        let response_text = response.text().await?;
        if response_text.is_empty() {
            return Err(LlmError::ApiError(
                "Received empty response body from LM Studio".to_string(),
            ));
        }

        let lmstudio_response: LMStudioResponse = serde_json::from_str(&response_text)
            .map_err(|e| {
                LlmError::ApiError(format!(
                    "Failed to parse LM Studio JSON response: {}. Body: {}",
                    e, response_text
                ))
            })?;

        if lmstudio_response.choices.is_empty() {
            return Err(LlmError::ApiError("No response from LM Studio".to_string()));
        }

        let usage = lmstudio_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: lmstudio_response.choices[0].message.content.clone(),
            model: lmstudio_response.model,
            usage,
        })
    }

    async fn generate_stream(&self, request: &LlmRequest) -> LlmResult<LlmStream> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("LMStudio".to_string()));
        }

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        // Add Authorization header if an API key is provided (optional for LM Studio)
        if !self.base.api_key().is_empty() {
            match header::HeaderValue::from_str(&format!("Bearer {}", self.base.api_key())) {
                Ok(val) => {
                    headers.insert(header::AUTHORIZATION, val);
                }
                Err(e) => {
                    return Err(LlmError::ConfigError(format!(
                        "Invalid API key format for LM Studio: {}",
                        e
                    )))
                }
            }
        }

        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());

        let lmstudio_request = LMStudioRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: true,
        };

        let response = self
            .base
            .client()
            .post(&self.endpoint_url)
            .headers(headers)
            .json(&lmstudio_request)
            .send()
            .await?;

        let response_status = response.status();
        if !response_status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| format!("Unknown error. Status: {}", response_status));
            return Err(LlmError::ApiError(format!("LM Studio API error: {}", error_text)));
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
