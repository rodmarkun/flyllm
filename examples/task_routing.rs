use flyllm::{
    AnthropicProvider, OpenAIProvider, create_provider, ProviderType,
    load_balancer::{
        manager::{LlmManager, LlmManagerRequest, LlmManagerResponse},
        tasks::TaskDefinition,
    },
    errors::LlmResult,
};
use std::collections::HashMap;
use std::env;
use std::time::{Duration, Instant};
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

async fn run_sequential(manager: &mut LlmManager, requests: &[LlmManagerRequest]) -> Vec<LlmManagerResponse> {
    let mut results = Vec::with_capacity(requests.len());
    
    for request in requests {
        let prompt = &request.prompt;
        let task = request.task.as_deref();
        let params = request.params.clone();
        
        match manager.generate_response(prompt, task, params).await {
            Ok(content) => results.push(LlmManagerResponse {
                content,
                success: true,
                error: None,
            }),
            Err(e) => results.push(LlmManagerResponse {
                content: String::new(),
                success: false,
                error: Some(e.to_string()),
            }),
        }
    }
    
    results
}

fn print_results(results: &[LlmManagerResponse]) {
    println!("\nSummary result:");
    if results[0].success {
        println!("{}\n", results[0].content);
    } else {
        println!("Error: {}\n", results[0].error.as_ref().unwrap_or(&"Unknown error".to_string()));
    }
   
    println!("Creative writing result:");
    if results[1].success {
        println!("{}\n", results[1].content);
    } else {
        println!("Error: {}\n", results[1].error.as_ref().unwrap_or(&"Unknown error".to_string()));
    }
   
    println!("Code generation result:");
    if results[2].success {
        println!("{}\n", results[2].content);
    } else {
        println!("Error: {}\n", results[2].error.as_ref().unwrap_or(&"Unknown error".to_string()));
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
    );
   
    // Add providers to manager
    manager.add_instance(anthropic);
    
    // Create batch of requests
    let requests = vec![
        LlmManagerRequest {
            prompt: "Summarize the following text: Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.".to_string(),
            task: None,
            params: None,
        },
        LlmManagerRequest {
            prompt: "Write a short story about a robot discovering emotions.".to_string(),
            task: None,
            params: Some(HashMap::from([
                ("temperature".to_string(), json!(0.9))
            ])),
        },
        LlmManagerRequest {
            prompt: "Write a Python function that calculates the Fibonacci sequence up to n terms.".to_string(),
            task: None,
            params: None,
        },
    ];
   
    // First run sequentially and time it
    println!("Running requests sequentially...");
    let sequential_start = Instant::now();
    let sequential_results = run_sequential(&mut manager, &requests).await;
    let sequential_duration = sequential_start.elapsed();
    println!("Sequential processing completed in {:?}", sequential_duration);
    
    // Optional: Print results from sequential run
    // print_results(&sequential_results);
    
    // Now run in parallel and time it
    println!("\nRunning requests in parallel...");
    let parallel_start = Instant::now();
    let parallel_results = manager.batch_generate(requests.clone()).await;
    let parallel_duration = parallel_start.elapsed();
    println!("Parallel processing completed in {:?}", parallel_duration);
    
    // Print results from parallel run
    print_results(&parallel_results);
    
    // Calculate and display the speedup
    let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!("\nPerformance Comparison:");
    println!("Sequential time: {:?}", sequential_duration);
    println!("Parallel time: {:?}", parallel_duration);
    println!("Speedup: {:.2}x", speedup);
   
    // Print provider statistics
    let stats = manager.get_provider_stats();
    println!("\nProvider Statistics:");
    for (i, stat) in stats.iter().enumerate() {
        println!("Provider {}: Response time: {}ms, Requests: {}, Errors: {}, Error rate: {:.2}%",
            i, stat.avg_response_time_ms, stat.request_count, stat.error_count, stat.error_rate);
    }
   
    Ok(())
}