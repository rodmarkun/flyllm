//! Example: Loading LlmManager from a TOML configuration file
//!
//! This example demonstrates how to use the TOML configuration feature
//! to set up FlyLLM without using the builder pattern.
//!
//! Run with: cargo run --example config_file
//!
//! Before running, create a config file or set up environment variables:
//! - Copy examples/flyllm.example.toml to flyllm.toml
//! - Set your API keys as environment variables (e.g., OPENAI_API_KEY)

use flyllm::{GenerationRequest, LlmManager};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging (optional)
    flyllm::use_logging();

    println!("=== FlyLLM TOML Configuration Example ===\n");

    // Option 1: Load from a file
    // let manager = LlmManager::from_config_file("flyllm.toml").await?;

    // Option 2: Load from an embedded string (useful for testing)
    // This example uses inline TOML so it works without a config file
    let config_toml = create_demo_config();

    println!("Loading configuration...\n");

    let manager = match LlmManager::from_config_str(&config_toml).await {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to load configuration: {}", e);
            println!("\nMake sure you have set at least one API key environment variable:");
            println!("  - OPENAI_API_KEY");
            println!("  - ANTHROPIC_API_KEY");
            println!("  - Or configure a local provider (Ollama/LM Studio)");
            return Err(e.into());
        }
    };

    let provider_count = manager.get_provider_count().await;
    println!("Loaded {} provider(s) from configuration\n", provider_count);

    if provider_count == 0 {
        println!("No providers available. Please set API key environment variables.");
        return Ok(());
    }

    // Make a simple request
    println!("--- Making a test request ---\n");

    let request = GenerationRequest {
        prompt: "What is 2 + 2? Reply with just the number.".to_string(),
        task: Some("chat".to_string()),
        params: None,
    };

    let responses = manager.generate_sequentially(vec![request]).await;

    if let Some(response) = responses.first() {
        if response.success {
            println!("Response: {}", response.content);
        } else {
            println!("Request failed: {:?}", response.error);
        }
    }

    // Print token usage
    manager.print_token_usage().await;

    Ok(())
}

/// Creates a demo configuration that conditionally includes providers
/// based on available environment variables.
fn create_demo_config() -> String {
    let mut config = String::from(
        r#"
[settings]
strategy = "lru"
max_retries = 3

[[tasks]]
name = "chat"
max_tokens = 100
temperature = 0.7

[[tasks]]
name = "summary"
max_tokens = 500
temperature = 0.3
"#,
    );

    // Conditionally add providers based on available API keys
    if env::var("OPENAI_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "openai"
model = "gpt-4o-mini"
api_key = "${OPENAI_API_KEY}"
tasks = ["chat", "summary"]
"#,
        );
        println!("  [+] OpenAI provider configured");
    }

    if env::var("ANTHROPIC_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "anthropic"
model = "claude-3-haiku-20240307"
api_key = "${ANTHROPIC_API_KEY}"
tasks = ["chat", "summary"]
"#,
        );
        println!("  [+] Anthropic provider configured");
    }

    if env::var("GROQ_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "groq"
model = "llama-3.1-8b-instant"
api_key = "${GROQ_API_KEY}"
tasks = ["chat"]
"#,
        );
        println!("  [+] Groq provider configured");
    }

    if env::var("MISTRAL_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "mistral"
model = "mistral-small-latest"
api_key = "${MISTRAL_API_KEY}"
tasks = ["chat", "summary"]
"#,
        );
        println!("  [+] Mistral provider configured");
    }

    if env::var("TOGETHER_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "togetherai"
model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
api_key = "${TOGETHER_API_KEY}"
tasks = ["chat"]
"#,
        );
        println!("  [+] Together AI provider configured");
    }

    if env::var("COHERE_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "cohere"
model = "command-r"
api_key = "${COHERE_API_KEY}"
tasks = ["chat", "summary"]
"#,
        );
        println!("  [+] Cohere provider configured");
    }

    if env::var("PERPLEXITY_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "perplexity"
model = "sonar"
api_key = "${PERPLEXITY_API_KEY}"
tasks = ["chat"]
"#,
        );
        println!("  [+] Perplexity provider configured");
    }

    if env::var("GOOGLE_API_KEY").is_ok() {
        config.push_str(
            r#"
[[providers]]
type = "google"
model = "gemini-pro"
api_key = "${GOOGLE_API_KEY}"
tasks = ["chat", "summary"]
"#,
        );
        println!("  [+] Google provider configured");
    }

    // Check for local providers (always try to add if running locally)
    // Uncomment these if you have Ollama or LM Studio running
    /*
    config.push_str(r#"
[[providers]]
type = "ollama"
model = "llama3"
api_key = ""
endpoint = "http://localhost:11434"
tasks = ["chat"]
"#);
    println!("  [+] Ollama provider configured (local)");

    config.push_str(r#"
[[providers]]
type = "lmstudio"
model = "local-model"
api_key = ""
endpoint = "http://localhost:1234"
tasks = ["chat"]
"#);
    println!("  [+] LM Studio provider configured (local)");
    */

    config
}
