//! Tests for load balancing strategies.

use flyllm::{LlmManager, ProviderType, TaskDefinition};
use flyllm::load_balancer::strategies::{
    LoadBalancingStrategy,
    LeastRecentlyUsedStrategy,
    LowestLatencyStrategy,
    RandomStrategy
};

// ============================================================================
// Strategy Creation Tests
// ============================================================================

#[test]
fn test_lru_strategy_creation() {
    // Just verify that creation doesn't panic
    let _strategy = LeastRecentlyUsedStrategy::new();
}

#[test]
fn test_lowest_latency_strategy_creation() {
    let _strategy = LowestLatencyStrategy::new();
}

#[test]
fn test_random_strategy_creation() {
    let _strategy = RandomStrategy::new();
}

// ============================================================================
// Builder Strategy Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_builder_with_lru_strategy() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("test")
        .strategy(Box::new(LeastRecentlyUsedStrategy::new()))
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_builder_with_lowest_latency_strategy() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("test")
        .strategy(Box::new(LowestLatencyStrategy::new()))
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_builder_with_random_strategy() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("test"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key")
        .supports("test")
        .strategy(Box::new(RandomStrategy::new()))
        .build()
        .await
        .unwrap();

    assert_eq!(manager.get_provider_count().await, 1);
}

// ============================================================================
// TOML Strategy Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_config_lru_strategy() {
    let toml = r#"
[settings]
strategy = "lru"

[[tasks]]
name = "test"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
tasks = ["test"]
"#;

    let manager = LlmManager::from_config_str(toml).await.unwrap();
    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_config_lowest_latency_strategy() {
    let toml = r#"
[settings]
strategy = "lowest_latency"

[[tasks]]
name = "test"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
tasks = ["test"]
"#;

    let manager = LlmManager::from_config_str(toml).await.unwrap();
    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_config_random_strategy() {
    let toml = r#"
[settings]
strategy = "random"

[[tasks]]
name = "test"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
tasks = ["test"]
"#;

    let manager = LlmManager::from_config_str(toml).await.unwrap();
    assert_eq!(manager.get_provider_count().await, 1);
}

#[tokio::test]
async fn test_config_default_strategy() {
    let toml = r#"
[[tasks]]
name = "test"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
tasks = ["test"]
"#;

    // Should default to LRU
    let manager = LlmManager::from_config_str(toml).await.unwrap();
    assert_eq!(manager.get_provider_count().await, 1);
}

// ============================================================================
// Multi-Provider Strategy Tests
// ============================================================================

#[tokio::test]
async fn test_multiple_providers_with_strategy() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("chat"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key1")
        .supports("chat")
        .add_instance(ProviderType::Anthropic, "claude-3", "key2")
        .supports("chat")
        .add_instance(ProviderType::Mistral, "mistral-large", "key3")
        .supports("chat")
        .strategy(Box::new(LeastRecentlyUsedStrategy::new()))
        .build()
        .await
        .unwrap();

    // All three providers should be available for load balancing
    assert_eq!(manager.get_provider_count().await, 3);
}

#[tokio::test]
async fn test_strategy_with_disabled_provider() {
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("chat"))
        .add_instance(ProviderType::OpenAI, "gpt-4", "key1")
        .supports("chat")
        .add_instance(ProviderType::Anthropic, "claude-3", "key2")
        .supports("chat")
        .enabled(false) // This one is disabled
        .strategy(Box::new(RandomStrategy::new()))
        .build()
        .await
        .unwrap();

    // Both providers are added, but one is disabled
    assert_eq!(manager.get_provider_count().await, 2);
}
