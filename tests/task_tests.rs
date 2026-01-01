//! Tests for TaskDefinition and task routing.

use flyllm::TaskDefinition;
use serde_json::json;

// ============================================================================
// TaskDefinition Creation Tests
// ============================================================================

#[test]
fn test_task_definition_new() {
    let task = TaskDefinition::new("summary");
    assert_eq!(task.name, "summary");
    assert!(task.parameters.is_empty());
}

#[test]
fn test_task_definition_with_max_tokens() {
    let task = TaskDefinition::new("summary")
        .with_max_tokens(500);

    assert_eq!(task.parameters.get("max_tokens"), Some(&json!(500)));
}

#[test]
fn test_task_definition_with_temperature() {
    let task = TaskDefinition::new("creative")
        .with_temperature(0.9);

    // Check the temperature is set (f32 precision may differ from f64)
    let temp = task.parameters.get("temperature").unwrap().as_f64().unwrap();
    assert!((temp - 0.9).abs() < 0.01);
}

#[test]
fn test_task_definition_with_custom_param() {
    let task = TaskDefinition::new("test")
        .with_param("custom_key", "custom_value");

    assert_eq!(task.parameters.get("custom_key"), Some(&json!("custom_value")));
}

#[test]
fn test_task_definition_chained_params() {
    let task = TaskDefinition::new("full")
        .with_max_tokens(1000)
        .with_temperature(0.7)
        .with_param("top_p", 0.95);

    assert_eq!(task.parameters.get("max_tokens"), Some(&json!(1000)));

    // Check temperature with tolerance for f32 precision
    let temp = task.parameters.get("temperature").unwrap().as_f64().unwrap();
    assert!((temp - 0.7).abs() < 0.01);

    assert_eq!(task.parameters.get("top_p"), Some(&json!(0.95)));
}

#[test]
fn test_task_definition_clone() {
    let task = TaskDefinition::new("original")
        .with_max_tokens(500);

    let cloned = task.clone();

    assert_eq!(cloned.name, "original");
    assert_eq!(cloned.parameters.get("max_tokens"), Some(&json!(500)));
}

#[test]
fn test_task_definition_multiple_tasks() {
    let summary = TaskDefinition::new("summary")
        .with_max_tokens(500)
        .with_temperature(0.3);

    let creative = TaskDefinition::new("creative")
        .with_max_tokens(2000)
        .with_temperature(0.9);

    let code = TaskDefinition::new("code")
        .with_max_tokens(4000)
        .with_temperature(0.1);

    assert_ne!(summary.name, creative.name);
    assert_ne!(summary.parameters.get("max_tokens"), creative.parameters.get("max_tokens"));

    // Check temperature with tolerance for f32 precision
    let temp = code.parameters.get("temperature").unwrap().as_f64().unwrap();
    assert!((temp - 0.1).abs() < 0.01);
}

// ============================================================================
// Task Routing Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_single_provider_single_task() {
    use flyllm::{LlmManager, ProviderType};

    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("summary"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("summary")
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_single_provider_multiple_tasks() {
    use flyllm::{LlmManager, ProviderType};

    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("summary"))
        .define_task(TaskDefinition::new("creative"))
        .define_task(TaskDefinition::new("code"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports_many(&["summary", "creative", "code"])
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_multiple_providers_same_task() {
    use flyllm::{LlmManager, ProviderType};

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
async fn test_providers_with_different_tasks() {
    use flyllm::{LlmManager, ProviderType};

    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("summary"))
        .define_task(TaskDefinition::new("code"))
        .define_task(TaskDefinition::new("creative"))
        // OpenAI for code
        .add_instance(ProviderType::OpenAI, "gpt-4", "key1")
        .supports("code")
        // Anthropic for creative
        .add_instance(ProviderType::Anthropic, "claude-3", "key2")
        .supports("creative")
        // Mistral for summary
        .add_instance(ProviderType::Mistral, "mistral-large", "key3")
        .supports("summary")
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 3);
}

#[tokio::test]
async fn test_overlapping_task_support() {
    use flyllm::{LlmManager, ProviderType};

    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("chat"))
        .define_task(TaskDefinition::new("summary"))
        // OpenAI supports both
        .add_instance(ProviderType::OpenAI, "gpt-4", "key1")
        .supports_many(&["chat", "summary"])
        // Anthropic only chat
        .add_instance(ProviderType::Anthropic, "claude-3", "key2")
        .supports("chat")
        // Mistral only summary
        .add_instance(ProviderType::Mistral, "mistral-large", "key3")
        .supports("summary")
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 3);
}
