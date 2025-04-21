use flyllm::{
    // Removed create_provider as manager handles it
    ProviderType,
    load_balancer::{
        manager::{LlmManager, LlmManagerRequest, LlmManagerResponse},
        tasks::TaskDefinition // Import TaskDefinition
    },
    errors::LlmResult,
    initialize_logging // Import initialize_logging if not done elsewhere
};
use std::collections::HashMap;
use std::env;
use std::time::Instant;
use serde_json::json;
use log::{info, debug}; 

async fn run_sequential(manager: &LlmManager, requests: &[LlmManagerRequest]) -> Vec<LlmManagerResponse> {
    let mut results = Vec::with_capacity(requests.len());
    info!("Starting sequential execution...");
    for (i, request) in requests.iter().enumerate() {
        let start_req = Instant::now();
        debug!("Sending sequential request {} ({:?})...", i, request.task);
        let prompt = &request.prompt;
        let task = request.task.as_deref();
        let params = request.params.clone();

        match manager.generate_response(prompt, task, params).await {
            Ok(content) => {
                debug!("Sequential request {} successful in {:?}", i, start_req.elapsed());
                results.push(LlmManagerResponse {
                    content,
                    success: true,
                    error: None,
                })
            },
            Err(e) => {
                 debug!("Sequential request {} failed in {:?}: {}", i, start_req.elapsed(), e);
                results.push(LlmManagerResponse {
                    content: String::new(),
                    success: false,
                    error: Some(e.to_string()),
                })
            },
        }
    }
    info!("Sequential execution finished.");
    results
}

fn print_results(results: &[LlmManagerResponse]) {
    println!("\n--- Request Results ---");

    let tasks = ["Summary", "Creative Writing", "Code Generation", "Short Poem"];

    for (i, task_name) in tasks.iter().enumerate() {
        println!("{}:", task_name);
        if results.len() > i {
            if results[i].success {
                let content_preview = results[i].content.chars().take(150).collect::<String>();
                println!("Success: {}...\n", content_preview);
            } else {
                println!("Error: {}\n", results[i].error.as_ref().unwrap_or(&"Unknown error".to_string()));
            }
        } else {
            println!("No result for index {}\n", i);
        }
    }
}

#[tokio::main]
async fn main() -> LlmResult<()> {
    initialize_logging(); 

    info!("Starting Task Routing Example");

    let anthropic_api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable not set");

    let mut manager = LlmManager::new(); 

    // -- Define Tasks --
    let summary_task = TaskDefinition {
        name: "summary".to_string(),
        parameters: HashMap::from([
            ("max_tokens".to_string(), json!(500)),
            ("temperature".to_string(), json!(0.3)),
        ]),
    };

    let creative_writing_task = TaskDefinition {
        name: "creative_writing".to_string(),
        parameters: HashMap::from([
            ("max_tokens".to_string(), json!(1500)),
            ("temperature".to_string(), json!(0.9)),
        ]),
    };

    let code_generation_task = TaskDefinition {
        name: "code_generation".to_string(),
        parameters: HashMap::from([
            ("max_tokens".to_string(), json!(1000)),
            ("temperature".to_string(), json!(0.2)),
        ]),
    };

    let short_poem_task = TaskDefinition {
        name: "short_poem".to_string(), 
        parameters: HashMap::from([
             ("max_tokens".to_string(), json!(100)), 
             ("temperature".to_string(), json!(0.8)), 
        ]),
    };


    info!("Adding Anthropic providers...");

    // Provider 0: Supports Summary and Code Generation
    manager.add_provider(
        ProviderType::Anthropic,
        anthropic_api_key.clone(),
        "claude-3-haiku-20240307".to_string(), // Use a faster/cheaper model for testing if desired
        vec![summary_task.clone(), code_generation_task.clone()], // Assign tasks
        true, // Enabled
    );
     info!("Added Provider 0 (Haiku) - Supports: Summary, Code Generation");

    // Provider 1: Supports Summary and Creative Writing
    manager.add_provider(
        ProviderType::Anthropic,
        anthropic_api_key.clone(),
        "claude-3-sonnet-20240229".to_string(),
        vec![summary_task.clone(), creative_writing_task.clone()], // Assign tasks
        true, // Enabled
    );
     info!("Added Provider 1 (Sonnet) - Supports: Summary, Creative Writing");

    // Provider 2: Supports Creative Writing and Short Poems only
    manager.add_provider(
        ProviderType::Anthropic,
        anthropic_api_key.clone(),
        "claude-3-opus-20240229".to_string(), // Different model
         vec![creative_writing_task.clone(), short_poem_task.clone()], // Assign tasks
        true, // Enabled
    );
     info!("Added Provider 2 (Opus) - Supports: Creative Writing, Short Poems");

    // --- Define requests ---
    let requests = vec![
        LlmManagerRequest {
            prompt: "Summarize the following text: Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.".to_string(),
            task: Some("summary".to_string()),
            params: None, // Will use default task params
        },
        LlmManagerRequest {
            prompt: "Write a short story about a robot discovering emotions.".to_string(),
            task: Some("creative_writing".to_string()),
            params: None, // Will use default task params
        },
        LlmManagerRequest {
            prompt: "Write a Python function that calculates the Fibonacci sequence up to n terms.".to_string(),
            task: Some("code_generation".to_string()),
            params: None, // Will use default task params
        },
        LlmManagerRequest {
            prompt: "Write a VERY short poem about the rain.".to_string(),
            task: Some("creative_writing".to_string()), // Still creative writing general task
            // Override default params for this specific request
            params: Some(HashMap::from([
                ("max_tokens".to_string(), json!(50)), // Override default
            ])),
        },
         // Example using the specific 'short_poem' task (might only go to provider 2)
         LlmManagerRequest {
             prompt: "Craft a haiku about a silent dawn.".to_string(),
             task: Some("short_poem".to_string()), // Use the specific task name
             params: None, // Use short_poem defaults (max_tokens: 100)
         },
    ];
     info!("Defined {} requests.", requests.len());


    // --- Run Sequential ---
    println!("\n=== Running requests sequentially... ===");
    let sequential_start = Instant::now();
    let sequential_results = run_sequential(&manager, &requests).await;
    let sequential_duration = sequential_start.elapsed();
    println!("Sequential processing completed in {:?}", sequential_duration);


    // --- Run Parallel ---
    println!("\n=== Running requests in parallel... ===");
    let parallel_start = Instant::now();
    // Use .clone() on requests Vec if you need to use it again later
    let parallel_results = manager.batch_generate(requests.clone()).await;
    let parallel_duration = parallel_start.elapsed();
    println!("Parallel processing completed in {:?}", parallel_duration);

    // --- Print Results ---
    print_results(&parallel_results);

    info!("Task Routing Example Finished.");
    Ok(())
}