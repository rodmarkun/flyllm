use flyllm::{
    ProviderType, LlmManager, GenerationRequest, LlmManagerResponse, TaskDefinition, LlmResult,
    use_logging,
};
use std::env;
use std::time::Instant;
use log::info;

#[tokio::main]
async fn main() -> LlmResult<()> {
    use_logging(); // Setup logging

    info!("Starting Task Routing Example (Builder Pattern)");

    // --- API Keys ---
    let anthropic_api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let mistral_api_key = env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
    let google_api_key = env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY not set");

    // --- Configure Manager using Builder ---
    let manager = LlmManager::builder()
        // Define tasks centrally
        .define_task(
            TaskDefinition::new("summary")
                .with_max_tokens(500)  // Use helper or with_param
                .with_param("temperature", 0.3) // Use generic method
        )
        .define_task(
            TaskDefinition::new("creative_writing")
                .with_max_tokens(1500)
                .with_temperature(0.9)
        )
        .define_task(
            TaskDefinition::new("code_generation")
        )
         .define_task(
            TaskDefinition::new("short_poem")
                .with_max_tokens(100)
                .with_temperature(0.8)
        )

        // Add providers and link tasks by name
        .add_provider(ProviderType::Ollama, "llama2:7b", "")
            .supports("summary") // Chain configuration for this provider
            .supports("code_generation")
            .custom_endpoint("http://localhost:11434/api/chat") // This is the default Ollama endpoint, but we can specify custom ones.
            // .enabled(true) // Optional, defaults to true

        .add_provider(ProviderType::Mistral, "mistral-large-latest", &mistral_api_key)
            .supports("summary") 
            .supports("code_generation")

        .add_provider(ProviderType::Anthropic, "claude-3-sonnet-20240229", &anthropic_api_key)
            .supports("summary")
            .supports("creative_writing")
            .supports("code_generation")

        .add_provider(ProviderType::Anthropic, "claude-3-opus-20240229", &anthropic_api_key)
             .supports_many(&["creative_writing", "short_poem"]) // Example using supports_many

        .add_provider(ProviderType::Google, "gemini-2.0-flash", &google_api_key)
             .supports("short_poem")

        .add_provider(ProviderType::OpenAI, "gpt-3.5-turbo", &openai_api_key)
            .supports("summary")
            // Example: Add a disabled provider
         // .add_provider(ProviderType::OpenAI, "gpt-4", &openai_api_key)
         //     .supports("creative_writing")
         //     .supports("code_generation")
         //     .enabled(false) // Explicitly disable

        // Finalize the manager configuration
        .build()?; 

    info!("LlmManager configured with {} providers.", manager.get_provider_stats().len());

    // --- Define Requests using Builder ---
    let requests = vec![
        GenerationRequest::builder(
            "Summarize the following text: Climate change refers to long-term shifts...",
        )
        .task("summary")
        .build(),

        GenerationRequest::builder("Write a short story about a robot discovering emotions.")
            .task("creative_writing")
            .build(),

        GenerationRequest::builder(
            "Write a Python function that calculates the Fibonacci sequence up to n terms.",
        )
        .task("code_generation")
        .build(),

        // Example overriding parameters for a specific request
        GenerationRequest::builder("Write a VERY short poem about the rain.")
            .task("creative_writing") // Target creative writing task defaults...
            .max_tokens(50) // ...but override max_tokens just for this request
            // .param("temperature", 0.95) // Could override temperature too
            .build(),

        GenerationRequest::builder("Write a rust program to sum two input numbers via console.")
             .task("code_generation")
             .build(),

        GenerationRequest::builder("Craft a haiku about a silent dawn.")
            .task("short_poem")
            .build(),
    ];
    info!("Defined {} requests using builder pattern.", requests.len());


    // --- Run Requests (Sequential and Parallel) ---
    println!("\n=== Running requests sequentially... ===");
    let sequential_start = Instant::now();
    let sequential_results = manager.generate_sequentially(requests.clone()).await;
    let sequential_duration = sequential_start.elapsed();
    println!("Sequential processing completed in {:?}", sequential_duration);
    print_results(&sequential_results); 


    println!("\n=== Running requests in parallel... ===");
    let parallel_start = Instant::now();
    let parallel_results = manager.batch_generate(requests).await; // Use original requests vec
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

    manager.print_token_usage();

    Ok(())
}

fn print_results(results: &[LlmManagerResponse]) {
     println!("\n--- Request Results ---");

     let task_labels = [
         "Summary Request",
         "Creative Writing Request",
         "Code Generation Request",
         "Short Poem Request (Override)",
         "Rust Code Request", 
         "Haiku Request"
     ];

     for (i, result) in results.iter().enumerate() {
         let task_label = task_labels.get(i).map_or_else(|| "Unknown Task", |&name| name);
         println!("{}:", task_label);
         if result.success {
             let content_preview = result.content.chars().take(150).collect::<String>();
             let ellipsis = if result.content.chars().count() > 150 { "..." } else { "" };
             println!("  Success: {}{}\n", content_preview, ellipsis);
         } else {
             println!("  Error: {}\n", result.error.as_ref().unwrap_or(&"Unknown error".to_string()));
         }
     }
 }