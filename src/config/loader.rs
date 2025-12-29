//! Configuration file loading and environment variable resolution.

use std::env;
use std::fs;
use std::path::Path;
use regex::Regex;

use crate::errors::{LlmError, LlmResult};
use super::types::{Config, ProviderConfig};

/// Load and parse a TOML configuration file.
///
/// # Arguments
/// * `path` - Path to the TOML configuration file
///
/// # Returns
/// * `LlmResult<Config>` - Parsed configuration with environment variables resolved
///
/// # Example
/// ```no_run
/// use flyllm::config::load_config;
///
/// let config = load_config("flyllm.toml").unwrap();
/// ```
pub fn load_config<P: AsRef<Path>>(path: P) -> LlmResult<Config> {
    let path = path.as_ref();

    let content = fs::read_to_string(path).map_err(|e| {
        LlmError::ConfigError(format!(
            "Failed to read config file '{}': {}",
            path.display(),
            e
        ))
    })?;

    parse_config(&content)
}

/// Parse a TOML configuration string.
///
/// # Arguments
/// * `content` - TOML configuration string
///
/// # Returns
/// * `LlmResult<Config>` - Parsed configuration with environment variables resolved
pub fn parse_config(content: &str) -> LlmResult<Config> {
    let mut config: Config = toml::from_str(content).map_err(|e| {
        LlmError::ConfigError(format!("Failed to parse TOML: {}", e))
    })?;

    resolve_env_vars(&mut config)?;
    validate_config(&config)?;

    Ok(config)
}

/// Resolve environment variable references in the configuration.
///
/// Environment variables are specified using the `${VAR_NAME}` syntax.
/// If a variable is not found, an error is returned with a helpful message.
fn resolve_env_vars(config: &mut Config) -> LlmResult<()> {
    let env_var_pattern = Regex::new(r"\$\{([^}]+)\}").unwrap();

    for (idx, provider) in config.providers.iter_mut().enumerate() {
        if let Some(resolved) = resolve_env_var_string(&provider.api_key, &env_var_pattern)? {
            provider.api_key = resolved;
        } else if env_var_pattern.is_match(&provider.api_key) {
            // Extract the variable name for error message
            if let Some(caps) = env_var_pattern.captures(&provider.api_key) {
                let var_name = caps.get(1).unwrap().as_str();
                return Err(LlmError::ConfigError(format!(
                    "Environment variable '{}' not found\n  \
                     → Referenced in providers[{}].api_key\n  \
                     → Set it with: export {}=\"your-key\"",
                    var_name, idx, var_name
                )));
            }
        }

        // Also resolve endpoint if it uses env vars
        if let Some(ref endpoint) = provider.endpoint {
            if let Some(resolved) = resolve_env_var_string(endpoint, &env_var_pattern)? {
                provider.endpoint = Some(resolved);
            }
        }
    }

    // Resolve debug_folder if it uses env vars
    if let Some(ref folder) = config.settings.debug_folder {
        if let Some(resolved) = resolve_env_var_string(folder, &env_var_pattern)? {
            config.settings.debug_folder = Some(resolved);
        }
    }

    Ok(())
}

/// Resolve environment variables in a single string.
/// Returns None if no env vars are present, Some(resolved) if all resolved successfully.
fn resolve_env_var_string(s: &str, pattern: &Regex) -> LlmResult<Option<String>> {
    if !pattern.is_match(s) {
        return Ok(None);
    }

    let mut result = s.to_string();

    for caps in pattern.captures_iter(s) {
        let full_match = caps.get(0).unwrap().as_str();
        let var_name = caps.get(1).unwrap().as_str();

        match env::var(var_name) {
            Ok(value) => {
                result = result.replace(full_match, &value);
            }
            Err(_) => {
                return Err(LlmError::ConfigError(format!(
                    "Environment variable '{}' not found\n  \
                     → Set it with: export {}=\"your-value\"",
                    var_name, var_name
                )));
            }
        }
    }

    Ok(Some(result))
}

/// Validate the configuration for consistency.
fn validate_config(config: &Config) -> LlmResult<()> {
    // Check for valid provider types
    let valid_providers = [
        "anthropic", "openai", "mistral", "google", "ollama",
        "lmstudio", "groq", "cohere", "togetherai", "perplexity"
    ];

    for (idx, provider) in config.providers.iter().enumerate() {
        let provider_type = provider.provider_type.to_lowercase();
        if !valid_providers.contains(&provider_type.as_str()) {
            return Err(LlmError::ConfigError(format!(
                "Unknown provider type '{}' in providers[{}]\n  \
                 → Valid types: {}",
                provider.provider_type,
                idx,
                valid_providers.join(", ")
            )));
        }

        // Check that referenced tasks exist
        let defined_tasks: Vec<&str> = config.tasks.iter().map(|t| t.name.as_str()).collect();
        for task in &provider.tasks {
            if !defined_tasks.contains(&task.as_str()) {
                let provider_name = get_provider_display_name(provider);
                return Err(LlmError::ConfigError(format!(
                    "Task '{}' referenced by provider '{}' is not defined\n  \
                     → Define it in [[tasks]] section or remove from provider's tasks list",
                    task, provider_name
                )));
            }
        }
    }

    // Check for valid strategy
    let valid_strategies = ["lru", "lowest_latency", "random"];
    let strategy = config.settings.strategy.to_lowercase();
    if !valid_strategies.contains(&strategy.as_str()) {
        return Err(LlmError::ConfigError(format!(
            "Unknown strategy '{}'\n  \
             → Valid strategies: {}",
            config.settings.strategy,
            valid_strategies.join(", ")
        )));
    }

    Ok(())
}

/// Get a display name for a provider configuration.
fn get_provider_display_name(provider: &ProviderConfig) -> String {
    if let Some(ref name) = provider.name {
        name.clone()
    } else {
        format!("{}/{}", provider.provider_type, provider.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.providers[0].api_key, "test-key");
    }

    #[test]
    fn test_env_var_resolution() {
        env::set_var("TEST_API_KEY", "resolved-key");

        let toml = r#"
[[tasks]]
name = "chat"

[[providers]]
type = "openai"
model = "gpt-4"
api_key = "${TEST_API_KEY}"
tasks = ["chat"]
"#;

        let config = parse_config(toml).unwrap();
        assert_eq!(config.providers[0].api_key, "resolved-key");

        env::remove_var("TEST_API_KEY");
    }

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
        assert!(err.contains("not defined"));
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
}
