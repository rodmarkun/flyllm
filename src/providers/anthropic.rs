use crate::providers::provider::{LlmProvider, BaseProvider};
use crate::providers::types::{LlmRequest, LlmResponse, TokenUsage};
use crate::errors::{LlmError, LlmResult};

use async_trait::async_trait;
use reqwest::header;
use serde::{Serialize, Deserialize};

pub struct AnthropicProvider {
    base: BaseProvider,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    model: String,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
    #[serde(rename = "type")]
    content_type: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String, enabled: bool, complexity: u32) -> Self {
        let base = BaseProvider::new("anthropic".to_string(), api_key, model, enabled, complexity);
        Self { base }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("Anthropic".to_string()));
        }

        let mut headers = header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            header::HeaderValue::from_str(self.base.api_key())
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format: {}", e)))?,
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "anthropic-version",
            header::HeaderValue::from_static("2023-06-01"),
        );
        
        let model = request.model.clone().unwrap_or_else(|| self.base.model().to_string());
        
        // Extract system message and regular messages
        let mut system_content = None;
        let mut regular_messages = Vec::new();
        
        for msg in &request.messages {
            if msg.role == "system" {
                system_content = Some(msg.content.clone());
            } else {
                regular_messages.push(AnthropicMessage {
                    role: msg.role.clone(),
                    content: msg.content.clone(),
                });
            }
        }
        
        // Ensure we have at least one message
        if regular_messages.is_empty() && system_content.is_some() {
            regular_messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: format!("Using this context: {}", system_content.unwrap()),
            });
            system_content = None;
        }
        
        if regular_messages.is_empty() {
            return Err(LlmError::ApiError("Anthropic requires at least one message".to_string()));
        }
        
        let anthropic_request = AnthropicRequest {
            model,
            system: system_content,
            messages: regular_messages,
            max_tokens: request.max_tokens.unwrap_or(1024),
            temperature: request.temperature,
        };
        
        let response = self.base.client()
            .post("https://api.anthropic.com/v1/messages")
            .headers(headers)
            .json(&anthropic_request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("Anthropic API error: {}", error_text)));
        }
        
        let anthropic_response: AnthropicResponse = response.json().await?;
            
        if anthropic_response.content.is_empty() {
            return Err(LlmError::ApiError("No response from Anthropic".to_string()));
        }
        
        let usage = anthropic_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });
        
        let text = anthropic_response.content.iter()
            .filter(|c| c.content_type == "text")
            .map(|c| c.text.clone())
            .collect::<Vec<String>>()
            .join("");
        
        Ok(LlmResponse {
            content: text,
            model: anthropic_response.model,
            usage,
        })
    }
    
    fn get_name(&self) -> &str {
        self.base.name()
    }
    
    fn get_model(&self) -> &str {
        self.base.model()
    }
    
    fn is_enabled(&self) -> bool {
        self.base.is_enabled()
    }

    fn get_complexity(&self) -> u32 {
        self.base.complexity()
    }
}