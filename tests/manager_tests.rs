//! Integration tests for LlmManager.

use flyllm::{LlmManager, ProviderType, TaskDefinition, GenerationRequest};

// ============================================================================
// Builder Pattern Tests
// ============================================================================

#[tokio::test]
async fn test_builder_creates_manager() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "fake-key")
        .supports("test")
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_builder_multiple_providers() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("chat"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key1")
        .supports("chat")
        .add_instance(ProviderType::Anthropic, "claude-3", "key2")
        .supports("chat")
        .add_instance(ProviderType::Mistral, "mistral-large", "key3")
        .supports("chat")
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 3);
}

#[tokio::test]
async fn test_builder_multiple_tasks() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("summary").with_max_tokens(500))
        .define_task(TaskDefinition::new("creative").with_temperature(0.9))
        .define_task(TaskDefinition::new("code").with_max_tokens(4000))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports_many(&["summary", "creative", "code"])
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_builder_with_custom_endpoint() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("chat"))
        .add_instance(ProviderType::Ollama, "llama3", "")
        .supports("chat")
        .custom_endpoint("http://localhost:11434")
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_builder_with_disabled_provider() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("chat"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("chat")
        .enabled(false)
        .build()
        .await
        .unwrap();

    // Provider is added but disabled
    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_builder_max_retries() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("test")
        .max_retries(10)
        .build()
        .await
        .unwrap();

    assert_eq!(manager.max_retries, 10);
}

#[tokio::test]
async fn test_builder_empty_providers_warning() {
    // Should succeed but with a warning (no providers)
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 0);
}

#[tokio::test]
async fn test_builder_undefined_task_error() {
    let result = LlmManager::builder()
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("undefined_task") // Task not defined
        .build()
        .await;

    assert!(result.is_err());
    if let Err(e) = result {
        let err = e.to_string();
        assert!(err.contains("undefined_task"));
    }
}

// ============================================================================
// TOML Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_from_config_str_basic() {
    let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "test-key"
tasks = ["chat"]
"#;

    let manager = LlmManager::from_config_str(toml).await.unwrap();
    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_from_config_str_multiple_providers() {
    let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key1"
tasks = ["chat"]

[[providers]]
type = "anthropic"
model = "claude-3"
api_key = "key2"
tasks = ["chat"]
"#;

    let manager = LlmManager::from_config_str(toml).await.unwrap();
    assert_eq!(manager.get_provider_count().await, 2);
}

#[tokio::test]
async fn test_from_config_str_with_settings() {
    let toml = r#"
[settings]
strategy = "random"
max_retries = 7

[[tasks]]
name = "test"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
tasks = ["test"]
"#;

    let manager = LlmManager::from_config_str(toml).await.unwrap();
    assert_eq!(manager.max_retries, 7);
}

#[tokio::test]
async fn test_from_config_str_invalid_provider() {
    let toml = r#"
[[providers]]
type = "invalid"
model = "test"
api_key = "key"
"#;

    let result = LlmManager::from_config_str(toml).await;
    assert!(result.is_err());
}

// ============================================================================
// Token Usage Tests
// ============================================================================

#[tokio::test]
async fn test_initial_usage_zero() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("test")
        .build()
        .await
        .unwrap();

    let total = manager.get_total_usage().await;
    assert_eq!(total.prompt_tokens, 0);
    assert_eq!(total.completion_tokens, 0);
    assert_eq!(total.total_tokens, 0);
}

#[tokio::test]
async fn test_instance_usage_none_initially() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("test")
        .build()
        .await
        .unwrap();

    // Instance 0 should exist but have zero usage
    let usage = manager.get_instance_usage(0).await;
    assert!(usage.is_some());
    let usage = usage.unwrap();
    assert_eq!(usage.total_tokens, 0);
}

// ============================================================================
// GenerationRequest Tests
// ============================================================================

#[test]
fn test_generation_request_basic() {
    let request = GenerationRequest {
        prompt: "Hello".to_string(),
        task: None,
        params: None,
    };

    assert_eq!(request.prompt, "Hello");
    assert!(request.task.is_none());
    assert!(request.params.is_none());
}

#[test]
fn test_generation_request_with_task() {
    let request = GenerationRequest {
        prompt: "Summarize this".to_string(),
        task: Some("summary".to_string()),
        params: None,
    };

    assert_eq!(request.task, Some("summary".to_string()));
}

#[test]
fn test_generation_request_with_params() {
    use std::collections::HashMap;
    use serde_json::json;

    let mut params = HashMap::new();
    params.insert("max_tokens".to_string(), json!(500));
    params.insert("temperature".to_string(), json!(0.7));

    let request = GenerationRequest {
        prompt: "Test".to_string(),
        task: None,
        params: Some(params),
    };

    assert!(request.params.is_some());
    let params = request.params.unwrap();
    assert_eq!(params.get("max_tokens"), Some(&json!(500)));
}
