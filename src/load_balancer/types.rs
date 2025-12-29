
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use serde_json::{json, Value};

/// User-facing request for LLM generation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GenerationRequest {
    pub prompt: String,                                     // Prompt for the LLM
    pub task: Option<String>,                               // Task to route for
    pub params: Option<HashMap<String, serde_json::Value>>, // Extra parameters
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            task: None,
            params: None,
        }
    }
}

impl GenerationRequest {
    // Standard Constructor
    pub fn new(prompt: String) -> Self {
        GenerationRequest {
            prompt,
            ..Default::default()
        }
    }

    /// Creates a builder for a GenerationRequest.
    pub fn builder(prompt: impl Into<String>) -> GenerationRequest {
        GenerationRequest::new(prompt.into())
    }

    /// Sets the target task for this request.
    pub fn task(mut self, name: impl Into<String>) -> Self {
        self.task = Some(name.into());
        self
    }

    /// Adds or overrides a parameter specifically for this request.
    pub fn param(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.params
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// Sets max tokens for this generation in specific
    pub fn max_tokens(self, tokens: u32) -> Self {
        self.param("max_tokens", json!(tokens))
    }

    /// Finalizes the GenerationRequest
    pub fn build(self) -> Self {
        self
    }
}

/// Internal request structure with additional retry information
#[derive(Clone)]
pub struct LlmManagerRequest {
    pub prompt: String,
    pub task: Option<String>,
    pub params: Option<HashMap<String, serde_json::Value>>,
    pub attempts: usize,
    pub failed_instances: Vec<usize>,
}

impl LlmManagerRequest {
    /// Convert a user-facing GenerationRequest to internal format
    pub fn from_generation_request(request: GenerationRequest) -> Self {
        Self {
            prompt: request.prompt,
            task: request.task,
            params: request.params,
            attempts: 0,
            failed_instances: Vec::new(),
        }
    }
}

/// Response structure returned to users
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmManagerResponse {
    pub content: String,
    pub success: bool,
    pub error: Option<String>,
}