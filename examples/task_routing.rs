use flyllm::{
    ProviderType, LlmManager, GenerationRequest, LlmManagerResponse, TaskDefinition, LlmResult,
    use_logging
};
use std::collections::HashMap;
use std::env;
use std::time::Instant;
use serde_json::json;
use log::info;

#[tokio::main]
async fn main() -> LlmResult<()> {
    use_logging();

    info!("Starting Task Routing Example");

    let anthropic_api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable not set");
    let openai_api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");
    let mistral_api_key = env::var("MISTRAL_API_KEY")
        .expect("MISTRAL_API_KEY environment variable not set");
    let google_api_key = env::var("GOOGLE_API_KEY")
        .expect("GOOGLE_API_KEY environment variable not set");

    let mut manager = LlmManager::new();

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


    info!("Adding providers...");

    manager.add_provider(
        ProviderType::Mistral,
        mistral_api_key.clone(),
        "mistral-large-latest".to_string(),
        vec![summary_task.clone(), code_generation_task.clone()],
        true,
    );
     info!("Added Provider 0 (Mistral Large) - Supports: Summary, Code Generation");

    manager.add_provider(
        ProviderType::Anthropic,
        anthropic_api_key.clone(),
        "claude-3-sonnet-20240229".to_string(),
        vec![summary_task.clone(), creative_writing_task.clone(), code_generation_task.clone()],
        true,
    );
     info!("Added Provider 1 (Sonnet) - Supports: Summary, Creative Writing, Code Generation");

    manager.add_provider(
        ProviderType::Anthropic,
        anthropic_api_key.clone(),
        "claude-3-opus-20240229".to_string(),
         vec![creative_writing_task.clone(), short_poem_task.clone()],
        true,
    );
     info!("Added Provider 2 (Opus) - Supports: Creative Writing, Short Poems");

    manager.add_provider(
        ProviderType::Google,
        google_api_key.clone(),
        "gemini-2.0-flash".to_string(),
         vec![short_poem_task.clone()],
        true,
    );
     info!("Added Provider 3 (Gemini Flash) - Supports: Short Poems");

    manager.add_provider(
        ProviderType::OpenAI,
        openai_api_key.clone(),
        "gpt-3.5-turbo".to_string(),
         vec![summary_task.clone()],
        true,
    );
     info!("Added Provider 4 (OpenAI GPT 3.5) - Supports: Summary");

    let requests = vec![
        GenerationRequest {
            prompt: "Summarize the following text: Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.".to_string(),
            task: Some("summary".to_string()),
            params: None,
        },
        GenerationRequest {
            prompt: "Write a short story about a robot discovering emotions.".to_string(),
            task: Some("creative_writing".to_string()),
            params: None,
        },
        GenerationRequest {
            prompt: "Write a Python function that calculates the Fibonacci sequence up to n terms.".to_string(),
            task: Some("code_generation".to_string()),
            params: None,
        },
        GenerationRequest {
            prompt: "Write a VERY short poem about the rain.".to_string(),
            task: Some("creative_writing".to_string()),
            params: Some(HashMap::from([
                ("max_tokens".to_string(), json!(50)),
            ])),
        },
        GenerationRequest {
            prompt: "Write a rust program to sum two input numbers via console.".to_string(),
            task: Some("code_generation".to_string()),
            params: None,
        },
         GenerationRequest {
             prompt: "Craft a haiku about a silent dawn.".to_string(),
             task: Some("short_poem".to_string()),
             params: None,
         },
    ];
     info!("Defined {} requests.", requests.len());


    println!("\n=== Running requests sequentially... ===");
    let sequential_start = Instant::now();
    let sequential_results = manager.generate_sequentially(requests.clone()).await;
    let sequential_duration = sequential_start.elapsed();
    println!("Sequential processing completed in {:?}", sequential_duration);
    print_results(&sequential_results);


    println!("\n=== Running requests in parallel... ===");
    let parallel_start = Instant::now();
    let parallel_results = manager.batch_generate(requests.clone()).await;
    let parallel_duration = parallel_start.elapsed();
    println!("Parallel processing completed in {:?}", parallel_duration);
    print_results(&parallel_results);

    info!("Task Routing Example Finished.");

    // --- Comparison ---
    println!("\n--- Comparison ---");
    println!("Sequential Duration: {:?}", sequential_duration);
    println!("Parallel Duration:   {:?}", parallel_duration);

    if parallel_duration < sequential_duration && parallel_duration.as_nanos() > 0 {
        let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
        println!("Parallel execution was roughly {:.2}x faster.", speedup);
    } else if parallel_duration >= sequential_duration {
        println!("Parallel execution was not faster (or was equal) in this run.");
    } else {
         println!("Parallel execution finished too quickly to measure speedup reliably.");
    }

    Ok(())
}

fn print_results(results: &[LlmManagerResponse]) {
    println!("\n--- Request Results ---");

    let task_names = [
        "Summary Request",
        "Creative Writing Request",
        "Code Generation Request",
        "Short Poem Request (Override)",
        "Haiku Request"
    ];

    for (i, result) in results.iter().enumerate() {
        let task_label = task_names.get(i).map_or_else(|| "Unknown Task", |&name| name);
        println!("{}:", task_label);
        if result.success {
            let content_preview = result.content.chars().take(150).collect::<String>();
            println!("Success: {}...\n", content_preview);
        } else {
            println!("Error: {}\n", result.error.as_ref().unwrap_or(&"Unknown error".to_string()));
        }
    }
}