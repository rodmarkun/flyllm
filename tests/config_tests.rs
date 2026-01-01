//! Tests for TOML configuration loading and parsing.

use flyllm::config::{load_config, parse_config, Config};
use std::env;
use std::fs;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// TOML Parsing Tests
// ============================================================================

#[test]
fn test_parse_minimal_config() {
    let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "test-key"
tasks = ["chat"]
"#;

    let config = parse_config(toml).unwrap();
    assert_eq!(config.tasks.len(), 1);
    assert_eq!(config.tasks[0].name, "chat");
    assert_eq!(config.providers.len(), 1);
    assert_eq!(config.providers[0].provider_type, "openai");
    assert_eq!(config.providers[0].model, "gpt-4");
    assert_eq!(config.providers[0].api_key, "test-key");
}

#[test]
fn test_parse_full_config() {
    let toml = r#"
[settings]
strategy = "lowest_latency"
max_retries = 10
debug_folder = "./debug"

[[tasks]]
name = "summary"
max_tokens = 500
temperature = 0.3

[[tasks]]
name = "creative"
max_tokens = 2000
temperature = 0.9

[[providers]]
type = "anthropic"
model = "claude-3-sonnet"
api_key = "key1"
tasks = ["summary", "creative"]
enabled = true

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key2"
tasks = ["summary"]
enabled = false
"#;

    let config = parse_config(toml).unwrap();

    // Settings
    assert_eq!(config.settings.strategy, "lowest_latency");
    assert_eq!(config.settings.max_retries, 10);
    assert_eq!(config.settings.debug_folder, Some("./debug".to_string()));

    // Tasks
    assert_eq!(config.tasks.len(), 2);
    assert_eq!(config.tasks[0].name, "summary");
    assert_eq!(config.tasks[0].max_tokens, Some(500));
    assert_eq!(config.tasks[0].temperature, Some(0.3));

    // Providers
    assert_eq!(config.providers.len(), 2);
    assert_eq!(config.providers[0].provider_type, "anthropic");
    assert!(config.providers[0].enabled);
    assert!(!config.providers[1].enabled);
}

#[test]
fn test_default_settings() {
    let toml = r#"
[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
"#;

    let config = parse_config(toml).unwrap();
    assert_eq!(config.settings.strategy, "lru");
    assert_eq!(config.settings.max_retries, 5);
    assert!(config.settings.debug_folder.is_none());
}

#[test]
fn test_provider_defaults() {
    let toml = r#"
[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
"#;

    let config = parse_config(toml).unwrap();
    assert!(config.providers[0].enabled); // Default enabled
    assert!(config.providers[0].tasks.is_empty()); // No tasks by default
    assert!(config.providers[0].endpoint.is_none()); // No custom endpoint
    assert!(config.providers[0].name.is_none()); // No name
}

#[test]
fn test_custom_endpoint() {
    let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "ollama"
model = "llama3"
api_key = ""
endpoint = "http://localhost:11434"
tasks = ["chat"]
"#;

    let config = parse_config(toml).unwrap();
    assert_eq!(config.providers[0].endpoint, Some("http://localhost:11434".to_string()));
}

#[test]
fn test_multiple_providers_same_type() {
    let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key1"
name = "openai-primary"
tasks = ["chat"]

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key2"
name = "openai-backup"
tasks = ["chat"]
"#;

    let config = parse_config(toml).unwrap();
    assert_eq!(config.providers.len(), 2);
    assert_eq!(config.providers[0].name, Some("openai-primary".to_string()));
    assert_eq!(config.providers[1].name, Some("openai-backup".to_string()));
}

// ============================================================================
// Environment Variable Resolution Tests
// ============================================================================

#[test]
fn test_env_var_resolution() {
    env::set_var("FLYLLM_TEST_KEY", "resolved-api-key");

    let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "${FLYLLM_TEST_KEY}"
tasks = ["chat"]
"#;

    let config = parse_config(toml).unwrap();
    assert_eq!(config.providers[0].api_key, "resolved-api-key");

    env::remove_var("FLYLLM_TEST_KEY");
}

#[test]
fn test_env_var_missing() {
    // Make sure the var doesn't exist
    env::remove_var("FLYLLM_NONEXISTENT_KEY");

    let toml = r#"
[[providers]]
type = "openai"
model = "gpt-4"
api_key = "${FLYLLM_NONEXISTENT_KEY}"
"#;

    let result = parse_config(toml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("FLYLLM_NONEXISTENT_KEY"));
    assert!(err.contains("not found"));
}

#[test]
fn test_env_var_in_endpoint() {
    env::set_var("FLYLLM_TEST_ENDPOINT", "http://custom:8080");

    let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "ollama"
model = "llama3"
api_key = ""
endpoint = "${FLYLLM_TEST_ENDPOINT}"
tasks = ["chat"]
"#;

    let config = parse_config(toml).unwrap();
    assert_eq!(config.providers[0].endpoint, Some("http://custom:8080".to_string()));

    env::remove_var("FLYLLM_TEST_ENDPOINT");
}

#[test]
fn test_empty_api_key_allowed() {
    let toml = r#"
[[providers]]
type = "ollama"
model = "llama3"
api_key = ""
"#;

    let config = parse_config(toml).unwrap();
    assert_eq!(config.providers[0].api_key, "");
}

// ============================================================================
// Validation Tests
// ============================================================================

#[test]
fn test_invalid_provider_type() {
    let toml = r#"
[[providers]]
type = "invalid_provider"
model = "test"
api_key = "key"
"#;

    let result = parse_config(toml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unknown provider type"));
    assert!(err.contains("invalid_provider"));
}

#[test]
fn test_undefined_task_reference() {
    let toml = r#"
[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
tasks = ["undefined_task"]
"#;

    let result = parse_config(toml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("undefined_task"));
    assert!(err.contains("not defined"));
}

#[test]
fn test_invalid_strategy() {
    let toml = r#"
[settings]
strategy = "invalid_strategy"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
"#;

    let result = parse_config(toml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unknown strategy"));
}

#[test]
fn test_valid_strategies() {
    for strategy in &["lru", "lowest_latency", "random"] {
        let toml = format!(r#"
[settings]
strategy = "{}"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "key"
"#, strategy);

        let config = parse_config(&toml).unwrap();
        assert_eq!(config.settings.strategy, *strategy);
    }
}

#[test]
fn test_all_valid_provider_types() {
    let providers = [
        "anthropic", "openai", "mistral", "google", "ollama",
        "lmstudio", "groq", "cohere", "togetherai", "perplexity"
    ];

    for provider in providers {
        let toml = format!(r#"
[[providers]]
type = "{}"
model = "test-model"
api_key = "test-key"
"#, provider);

        let result = parse_config(&toml);
        assert!(result.is_ok(), "Provider '{}' should be valid", provider);
    }
}

// ============================================================================
// File Loading Tests
// ============================================================================

#[test]
fn test_load_from_file() {
    let toml_content = r#"
[[tasks]]
name = "test"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "file-key"
tasks = ["test"]
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(toml_content.as_bytes()).unwrap();

    let config = load_config(temp_file.path()).unwrap();
    assert_eq!(config.providers[0].api_key, "file-key");
}

#[test]
fn test_load_nonexistent_file() {
    let result = load_config("/nonexistent/path/config.toml");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Failed to read"));
}

#[test]
fn test_invalid_toml_syntax() {
    let toml = r#"
[[providers]
type = "openai"  # Missing closing bracket
"#;

    let result = parse_config(toml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Failed to parse TOML"));
}
