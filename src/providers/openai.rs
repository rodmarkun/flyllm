use crate::providers::provider::{LlmProvider, BaseProvider};
use crate::providers::types::{LlmRequest, LlmResponse, TokenUsage, Message};
use crate::errors::{LlmError, LlmResult};

use async_trait::async_trait;
use reqwest::header;
use serde::{Serialize, Deserialize};

pub struct OpenAIProvider {
    base: BaseProvider,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    model: String,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: Message,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String, enabled: bool) -> Self {
        let base = BaseProvider::new("openai".to_string(), api_key, model, enabled);
        Self { base }
    }
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    async fn generate(&self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        if !self.base.is_enabled() {
            return Err(LlmError::ProviderDisabled("OpenAI".to_string()));
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
        
        let openai_request = OpenAIRequest {
            model,
            messages: request.messages.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };
        
        let response = self.base.client()
            .post("https://api.openai.com/v1/chat/completions")
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
    
    fn get_name(&self) -> &str {
        self.base.name()
    }
    
    fn get_model(&self) -> &str {
        self.base.model()
    }
    
    fn is_enabled(&self) -> bool {
        self.base.is_enabled()
    }
}