use flyllm::{
    create_provider, ProviderType,
    load_balancer::manager::{LlmManager, LlmManagerRequest, LlmManagerResponse},
    errors::LlmResult,
};
use std::collections::HashMap;
use std::env;
use std::time::Instant;
use serde_json::json;
use log::info; 

async fn run_sequential(manager: &LlmManager, requests: &[LlmManagerRequest]) -> Vec<LlmManagerResponse> {
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
    if results.len() > 0 { 
        if results[0].success {
            println!("{}\n", results[0].content);
        } else {
            println!("Error: {}\n", results[0].error.as_ref().unwrap_or(&"Unknown error".to_string()));
        }
    } else { println!("No result for index 0\n"); }


    println!("Creative writing result:");
     if results.len() > 1 { 
        if results[1].success {
            println!("{}\n", results[1].content);
        } else {
            println!("Error: {}\n", results[1].error.as_ref().unwrap_or(&"Unknown error".to_string()));
        }
    } else { println!("No result for index 1\n"); }

    println!("Code generation result:");
     if results.len() > 2 { 
        if results[2].success {
            println!("{}\n", results[2].content);
        } else {
            println!("Error: {}\n", results[2].error.as_ref().unwrap_or(&"Unknown error".to_string()));
        }
    } else { println!("No result for index 2\n"); }
}

#[tokio::main]
async fn main() -> LlmResult<()> {
    std::env::set_var("RUST_LOG", "debug"); 
    env_logger::init();

    let anthropic_api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable not set");

    let mut manager = LlmManager::new();

    let anthropic_provider = create_provider(
        ProviderType::Anthropic,
        anthropic_api_key.clone(),
        "claude-3-sonnet-20240229".to_string(),
        true,
    );

    manager.add_instance(anthropic_provider);

    let summary_params = HashMap::from([
        ("max_tokens".to_string(), json!(500)),  
        ("temperature".to_string(), json!(0.3)), 
    ]);

    let creative_params = HashMap::from([
        ("max_tokens".to_string(), json!(1500)), 
        ("temperature".to_string(), json!(0.9)),
    ]);

    let code_params = HashMap::from([
        ("max_tokens".to_string(), json!(1000)),
        ("temperature".to_string(), json!(0.2)),
    ]);

    info!("Assigning tasks to provider 0...");
    manager.assign_task_to_provider(0, "summary", Some(summary_params));
    manager.assign_task_to_provider(0, "creative_writing", Some(creative_params));
    manager.assign_task_to_provider(0, "code_generation", Some(code_params));
    info!("Tasks assigned.");

    let requests = vec![
        LlmManagerRequest {
            prompt: "Summarize the following text: Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.".to_string(),
            task: Some("summary".to_string()), 
            params: None, 
        },
        LlmManagerRequest {
            prompt: "Write a short story about a robot discovering emotions.".to_string(),
            task: Some("creative_writing".to_string()), 
            params: None, 
        },
        LlmManagerRequest {
            prompt: "Write a Python function that calculates the Fibonacci sequence up to n terms.".to_string(),
            task: Some("code_generation".to_string()), 
            params: None, 
        },
        LlmManagerRequest {
            prompt: "Write a VERY short poem about the rain.".to_string(),
            task: Some("creative_writing".to_string()), 
            params: Some(HashMap::from([ 
                ("max_tokens".to_string(), json!(50))
            ])),
        },
    ];

    println!("Running requests sequentially...");
    let sequential_start = Instant::now();
    let sequential_results = run_sequential(&manager, &requests).await;
    let sequential_duration = sequential_start.elapsed();
    println!("Sequential processing completed in {:?}", sequential_duration);

   
    println!("\nRunning requests in parallel...");
    let parallel_start = Instant::now();
    let parallel_results = manager.batch_generate(requests.clone()).await;
    let parallel_duration = parallel_start.elapsed();
    println!("Parallel processing completed in {:?}", parallel_duration);

    print_results(&parallel_results);
    println!("Very short poem result:"); 
     if parallel_results.len() > 3 { 
        if parallel_results[3].success {
            println!("{}\n", parallel_results[3].content);
        } else {
            println!("Error: {}\n", parallel_results[3].error.as_ref().unwrap_or(&"Unknown error".to_string()));
        }
    } else { println!("No result for index 3\n"); }


    if parallel_duration.as_secs_f64() > 0.0 {
        let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
        println!("\nPerformance Comparison:");
        println!("Sequential time: {:?}", sequential_duration);
        println!("Parallel time:  {:?}", parallel_duration);
        println!("Speedup: {:.2}x", speedup);
    } else {
        println!("\nPerformance Comparison:");
        println!("Sequential time: {:?}", sequential_duration);
        println!("Parallel time:  {:?} (Division by zero prevented)", parallel_duration);
        println!("Speedup: N/A");
    }


    let stats = manager.get_provider_stats();
    println!("\nProvider Statistics:");
    let total_requests_expected = requests.len() * 2;
    println!("(Expected total requests across sequential + parallel: {})", total_requests_expected);

    for stat in stats.iter() { 
        println!(
            "Instance {}: Provider: {}, Avg Response time: {}ms, Requests: {}, Errors: {}, Error rate: {:.2}%",
            stat.id,
            stat.provider_name, 
            stat.avg_response_time.as_millis(),
            stat.request_count,
            stat.error_count,
            stat.error_rate
        );
    }

    Ok(())
}