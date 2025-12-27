use crate::providers::types::{ModelInfo, ProviderType};
use crate::errors::{LlmError, LlmResult};
use crate::constants;
use reqwest::{Client, header};
use serde::Deserialize;
use std::time::Duration;

/// Helper module for listing available models from providers
/// without requiring a fully initialized provider instance
pub struct ModelDiscovery;

impl ModelDiscovery {
    /// Create a standardized HTTP client for model discovery
    fn create_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client")
    }

    /// List available models from Anthropic
    ///
    /// # Parameters
    /// * `api_key` - Anthropic API key
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_anthropic_models(api_key: &str) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();
        
        let mut headers = header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            header::HeaderValue::from_str(api_key)
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format for Anthropic: {}", e)))?,
        );
        headers.insert(
            "anthropic-version",
            header::HeaderValue::from_static(constants::ANTHROPIC_API_VERSION),
        );
        
        let models_endpoint = "https://api.anthropic.com/v1/models";
        
        let response = client.get(models_endpoint)
            .headers(headers)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await
                .unwrap_or_else(|_| format!("Unknown error reading error response body, status: {}", status));
            return Err(LlmError::ApiError(format!("Anthropic API error ({}): {}", status, error_text)));
        }
        
        let response_bytes = response.bytes().await?;
        
        #[derive(Deserialize, Debug)]
        struct AnthropicModelsResponse {
            data: Vec<AnthropicModelInfo>,
        }
        #[derive(Deserialize, Debug)]
        struct AnthropicModelInfo {
            id: String,
            display_name: String,
        }
        
        let anthropic_response: AnthropicModelsResponse = serde_json::from_slice(&response_bytes)
            .map_err(|e| {
                let snippet_len = std::cmp::min(response_bytes.len(), 256); 
                let response_snippet = String::from_utf8_lossy(response_bytes.get(0..snippet_len).unwrap_or_default());
                LlmError::ParseError(format!(
                    "Error decoding Anthropic models JSON: {}. Response snippet: '{}'",
                    e, 
                    response_snippet
                ))
            })?;
        
        let models = anthropic_response.data.into_iter()
            .map(|m| ModelInfo {
                name: m.id,
                provider: ProviderType::Anthropic,
            })
            .collect();
        
        Ok(models)
    }

    /// List available models from OpenAI
    ///
    /// # Parameters
    /// * `api_key` - OpenAI API key
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_openai_models(api_key: &str) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format: {}", e)))?,
        );
        
        let models_endpoint = "https://api.openai.com/v1/models";
        
        let response = client.get(models_endpoint)
            .headers(headers)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("OpenAI API error: {}", error_text)));
        }
        
        #[derive(Deserialize)]
        struct OpenAIModelsResponse {
            data: Vec<OpenAIModelInfo>,
        }
        
        #[derive(Deserialize)]
        struct OpenAIModelInfo {
            id: String,
        }
        
        let openai_response: OpenAIModelsResponse = response.json().await?;
        
        let models = openai_response.data.into_iter()
            .filter(|m| m.id.starts_with("gpt-"))
            .map(|m| ModelInfo {
                name: m.id,
                provider: ProviderType::OpenAI,
            })
            .collect();
        
        Ok(models)
    }

    /// List available models from Mistral
    ///
    /// # Parameters
    /// * `api_key` - Mistral API key
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_mistral_models(api_key: &str) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format: {}", e)))?,
        );
        
        let models_endpoint = "https://api.mistral.ai/v1/models";
        
        let response = client.get(models_endpoint)
            .headers(headers)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("Mistral API error: {}", error_text)));
        }
        
        #[derive(Deserialize)]
        struct MistralModelsResponse {
            data: Vec<MistralModelInfo>,
        }
        
        #[derive(Deserialize)]
        struct MistralModelInfo {
            id: String,
        }
        
        let mistral_response: MistralModelsResponse = response.json().await?;
        
        let models = mistral_response.data.into_iter()
            .map(|m| ModelInfo {
                name: m.id,
                provider: ProviderType::Mistral,
            })
            .collect();
        
        Ok(models)
    }

    /// List available models from Google
    ///
    /// # Parameters
    /// * `api_key` - Google API key
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_google_models(api_key: &str) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        let models_endpoint = format!(
            "{}/v1beta/models?key={}",
            constants::GOOGLE_API_ENDPOINT_PREFIX, 
            api_key
        );
        
        let response = client.get(&models_endpoint)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("Google API error: {}", error_text)));
        }
        
        #[derive(Deserialize)]
        struct GoogleModelsResponse {
            models: Vec<GoogleModelInfo>,
        }
        
        #[derive(Deserialize)]
        struct GoogleModelInfo {
            name: String,
            #[serde(default)]
            display_name: Option<String>,
        }
        
        let google_response: GoogleModelsResponse = response.json().await?;
        
        let models = google_response.models.into_iter()
            .map(|m| {
                let name = m.display_name.unwrap_or_else(|| {
                    m.name.split('/').last().unwrap_or(&m.name).to_string()
                });
                
                ModelInfo {
                    name,
                    provider: ProviderType::Google,
                }
            })
            .collect();
        
        Ok(models)
    }

    /// List available models from Ollama
    ///
    /// # Parameters
    /// * `base_url` - Optional base URL for Ollama API, defaults to localhost
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_ollama_models(base_url: Option<&str>) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        // Use provided base URL or default to localhost
        let base_url = base_url.unwrap_or("http://localhost:11434");
        let models_endpoint = format!("{}/api/tags", base_url.trim_end_matches('/'));
        
        let response = client.get(&models_endpoint)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("Ollama API error: {}", error_text)));
        }
        
        #[derive(Deserialize)]
        struct OllamaModelsResponse {
            models: Vec<OllamaModelInfo>,
        }
        
        #[derive(Deserialize)]
        struct OllamaModelInfo {
            name: String,
        }
        
        let ollama_response: OllamaModelsResponse = response.json().await?;
        
        let models = ollama_response.models.into_iter()
            .map(|m| ModelInfo {
                name: m.name,
                provider: ProviderType::Ollama,
            })
            .collect();

        Ok(models)
    }

    /// List available models from LM Studio
    ///
    /// # Parameters
    /// * `base_url` - Optional base URL for LM Studio API, defaults to localhost:1234
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_lmstudio_models(base_url: Option<&str>) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        let base_url = base_url.unwrap_or("http://localhost:1234");
        let models_endpoint = format!("{}/v1/models", base_url.trim_end_matches('/'));

        let response = client.get(&models_endpoint)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("LM Studio API error: {}", error_text)));
        }

        #[derive(Deserialize)]
        struct LMStudioModelsResponse {
            data: Vec<LMStudioModelInfo>,
        }

        #[derive(Deserialize)]
        struct LMStudioModelInfo {
            id: String,
        }

        let lmstudio_response: LMStudioModelsResponse = response.json().await?;

        let models = lmstudio_response.data.into_iter()
            .map(|m| ModelInfo {
                name: m.id,
                provider: ProviderType::LMStudio,
            })
            .collect();

        Ok(models)
    }

    /// List available models from Groq
    ///
    /// # Parameters
    /// * `api_key` - Groq API key
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_groq_models(api_key: &str) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format: {}", e)))?,
        );

        let models_endpoint = "https://api.groq.com/openai/v1/models";

        let response = client.get(models_endpoint)
            .headers(headers)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("Groq API error: {}", error_text)));
        }

        #[derive(Deserialize)]
        struct GroqModelsResponse {
            data: Vec<GroqModelInfo>,
        }

        #[derive(Deserialize)]
        struct GroqModelInfo {
            id: String,
        }

        let groq_response: GroqModelsResponse = response.json().await?;

        let models = groq_response.data.into_iter()
            .map(|m| ModelInfo {
                name: m.id,
                provider: ProviderType::Groq,
            })
            .collect();

        Ok(models)
    }

    /// List available models from Cohere
    ///
    /// # Parameters
    /// * `api_key` - Cohere API key
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_cohere_models(api_key: &str) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format: {}", e)))?,
        );

        let models_endpoint = "https://api.cohere.com/v2/models";

        let response = client.get(models_endpoint)
            .headers(headers)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("Cohere API error: {}", error_text)));
        }

        #[derive(Deserialize)]
        struct CohereModelsResponse {
            models: Vec<CohereModelInfo>,
        }

        #[derive(Deserialize)]
        struct CohereModelInfo {
            name: String,
        }

        let cohere_response: CohereModelsResponse = response.json().await?;

        let models = cohere_response.models.into_iter()
            .map(|m| ModelInfo {
                name: m.name,
                provider: ProviderType::Cohere,
            })
            .collect();

        Ok(models)
    }

    /// List available models from Together AI
    ///
    /// # Parameters
    /// * `api_key` - Together AI API key
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_togetherai_models(api_key: &str) -> LlmResult<Vec<ModelInfo>> {
        let client = Self::create_client();

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| LlmError::ConfigError(format!("Invalid API key format: {}", e)))?,
        );

        let models_endpoint = "https://api.together.xyz/v1/models";

        let response = client.get(models_endpoint)
            .headers(headers)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("Together AI API error: {}", error_text)));
        }

        #[derive(Deserialize)]
        struct TogetherAIModelsResponse {
            #[serde(default)]
            data: Option<Vec<TogetherAIModelInfo>>,
        }

        #[derive(Deserialize)]
        struct TogetherAIModelInfo {
            id: String,
        }

        let together_response: TogetherAIModelsResponse = response.json().await?;

        let models = together_response.data
            .unwrap_or_default()
            .into_iter()
            .map(|m| ModelInfo {
                name: m.id,
                provider: ProviderType::TogetherAI,
            })
            .collect();

        Ok(models)
    }

    /// List available models from Perplexity
    ///
    /// Note: Perplexity doesn't have a models endpoint, returns known models
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing known model names
    pub async fn list_perplexity_models() -> LlmResult<Vec<ModelInfo>> {
        // Perplexity doesn't expose a models API endpoint
        // Return the known available models
        let known_models = vec![
            "sonar",
            "sonar-pro",
            "sonar-reasoning",
            "sonar-reasoning-pro",
            "sonar-deep-research",
        ];

        let models = known_models.into_iter()
            .map(|name| ModelInfo {
                name: name.to_string(),
                provider: ProviderType::Perplexity,
            })
            .collect();

        Ok(models)
    }

    /// List all models from a specific provider
    ///
    /// # Parameters
    /// * `provider_type` - Type of provider to query
    /// * `api_key` - API key for authentication
    /// * `base_url` - Optional base URL (used for Ollama and LM Studio)
    ///
    /// # Returns
    /// * Vector of ModelInfo structs containing model names
    pub async fn list_models(
        provider_type: ProviderType,
        api_key: &str,
        base_url: Option<&str>
    ) -> LlmResult<Vec<ModelInfo>> {
        match provider_type {
            ProviderType::Anthropic => Self::list_anthropic_models(api_key).await,
            ProviderType::OpenAI => Self::list_openai_models(api_key).await,
            ProviderType::Mistral => Self::list_mistral_models(api_key).await,
            ProviderType::Google => Self::list_google_models(api_key).await,
            ProviderType::Ollama => Self::list_ollama_models(base_url).await,
            ProviderType::LMStudio => Self::list_lmstudio_models(base_url).await,
            ProviderType::Groq => Self::list_groq_models(api_key).await,
            ProviderType::Cohere => Self::list_cohere_models(api_key).await,
            ProviderType::TogetherAI => Self::list_togetherai_models(api_key).await,
            ProviderType::Perplexity => Self::list_perplexity_models().await,
        }
    }
}