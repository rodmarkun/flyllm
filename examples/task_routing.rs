use flyllm::{
    AnthropicProvider, OpenAIProvider, create_provider, ProviderType,
    load_balancer::{
        manager::LlmManager,
        tasks::TaskDefinition,
    },
    errors::LlmResult,
};
use std::collections::HashMap;
use std::env;
use serde_json::json;

// Helper function to create a task definition
fn create_task(name: &str, params: Vec<(&str, serde_json::Value)>) -> TaskDefinition {
    let mut parameters = HashMap::new();
    for (key, value) in params {
        parameters.insert(key.to_string(), value);
    }
    
    TaskDefinition {
        name: name.to_string(),
        parameters,
    }
}

#[tokio::main]
async fn main() -> LlmResult<()> {
    // Get API keys from environment
    let anthropic_api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable not set");
    
    // Create a manager
    let mut manager = LlmManager::new();
    
    // Create providers
    let anthropic = create_provider(
        ProviderType::Anthropic,
        anthropic_api_key,
        "claude-3-opus-20240229".to_string(),
        true,
        5,
    );
    
    // Add providers to manager
    manager.add_instance(anthropic);
    
    // Generate responses for different tasks
    println!("Testing summarization task...");
    let summary = manager.generate_response(
        "Summarize the following text: Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.",
        None,
        None,
    ).await?;
    
    println!("Summary result:\n{}\n", summary);
    
    println!("Testing creative writing task...");
    let story = manager.generate_response(
        "Write a short story about a robot discovering emotions.",
        None,
        Some(HashMap::from([
            ("temperature".to_string(), json!(0.9)) // Override the task default
        ])),
    ).await?;
    
    println!("Creative writing result:\n{}\n", story);
    
    println!("Testing code generation task...");
    let code = manager.generate_response(
        "Write a Python function that calculates the Fibonacci sequence up to n terms.",
        None,
        None,
    ).await?;
    
    println!("Code generation result:\n{}\n", code);
    
    // Print provider statistics
    let stats = manager.get_provider_stats();
    println!("Provider Statistics:");
    for (i, stat) in stats.iter().enumerate() {
        println!("Provider {}: Response time: {}ms, Requests: {}, Errors: {}, Error rate: {:.2}%",
            i, stat.avg_response_time_ms, stat.request_count, stat.error_count, stat.error_rate);
    }
    
    Ok(())
}