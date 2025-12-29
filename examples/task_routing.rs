use flyllm::{
    ProviderType, LlmManager, GenerationRequest, LlmManagerResponse, TaskDefinition, LlmResult,
    use_logging, ModelDiscovery, ModelInfo,
};
use std::env;
use std::path::PathBuf;
use std::time::Instant;
use std::collections::HashMap;
use futures::StreamExt;
use log::info;

#[tokio::main]
async fn main() -> LlmResult<()> {
    env::set_var("RUST_LOG", "debug"); // Uncomment this to see debugging messages
    use_logging(); // Setup logging

    info!("Starting Task Routing Example");

    // --- API Keys (all optional) ---
    let anthropic_api_key = env::var("ANTHROPIC_API_KEY").ok();
    let openai_api_key = env::var("OPENAI_API_KEY").ok();
    let mistral_api_key = env::var("MISTRAL_API_KEY").ok();
    let google_api_key = env::var("GOOGLE_API_KEY").ok();
    let groq_api_key = env::var("GROQ_API_KEY").ok();
    let together_api_key = env::var("TOGETHER_API_KEY").ok();
    let cohere_api_key = env::var("COHERE_API_KEY").ok();
    let perplexity_api_key = env::var("PERPLEXITY_API_KEY").ok();

    // Check that at least one provider is available
    let has_any_provider = anthropic_api_key.is_some()
        || openai_api_key.is_some()
        || mistral_api_key.is_some()
        || google_api_key.is_some()
        || groq_api_key.is_some()
        || together_api_key.is_some()
        || cohere_api_key.is_some()
        || perplexity_api_key.is_some();

    if !has_any_provider {
        println!("No API keys found. Please set at least one of:");
        println!("  ANTHROPIC_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY, GOOGLE_API_KEY,");
        println!("  GROQ_API_KEY, TOGETHER_API_KEY, COHERE_API_KEY, PERPLEXITY_API_KEY");
        return Ok(());
    }

    // Print which providers are available
    println!("\n=== Available Providers ===");
    if anthropic_api_key.is_some() { println!("  ✓ Anthropic"); }
    if openai_api_key.is_some() { println!("  ✓ OpenAI"); }
    if mistral_api_key.is_some() { println!("  ✓ Mistral"); }
    if google_api_key.is_some() { println!("  ✓ Google"); }
    if groq_api_key.is_some() { println!("  ✓ Groq"); }
    if together_api_key.is_some() { println!("  ✓ Together AI"); }
    if cohere_api_key.is_some() { println!("  ✓ Cohere"); }
    if perplexity_api_key.is_some() { println!("  ✓ Perplexity"); }

    // --- Fetch and print available models ---
    print_available_models(
        anthropic_api_key.as_deref(),
        openai_api_key.as_deref(),
        mistral_api_key.as_deref(),
        google_api_key.as_deref()
    ).await;

    // --- Configure Manager using Builder ---
    let mut builder = LlmManager::builder()
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
        // Adds a debug folder for debugging all requests made
        .debug_folder(PathBuf::from("debug_folder"));

    // Conditionally add providers based on available API keys

    // Anthropic
    if let Some(ref key) = anthropic_api_key {
        builder = builder
            .add_instance(ProviderType::Anthropic, "claude-3-sonnet-20240229", key)
            .supports("summary")
            .supports("creative_writing")
            .supports("code_generation")
            .add_instance(ProviderType::Anthropic, "claude-3-opus-20240229", key)
            .supports_many(&["creative_writing", "short_poem"]);
    }

    // OpenAI
    if let Some(ref key) = openai_api_key {
        builder = builder
            .add_instance(ProviderType::OpenAI, "gpt-3.5-turbo", key)
            .supports("summary")
            .supports("code_generation");
    }

    // Mistral
    if let Some(ref key) = mistral_api_key {
        builder = builder
            .add_instance(ProviderType::Mistral, "mistral-large-latest", key)
            .supports("summary")
            .supports("code_generation");
    }

    // Google
    if let Some(ref key) = google_api_key {
        builder = builder
            .add_instance(ProviderType::Google, "gemini-2.0-flash", key)
            .supports("short_poem")
            .supports("summary");
    }

    // Groq
    if let Some(ref key) = groq_api_key {
        builder = builder
            .add_instance(ProviderType::Groq, "llama-3.1-70b-versatile", key)
            .supports("summary")
            .supports("code_generation");
    }

    // Together AI
    if let Some(ref key) = together_api_key {
        builder = builder
            .add_instance(ProviderType::TogetherAI, "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", key)
            .supports("creative_writing")
            .supports("summary");
    }

    // Cohere
    if let Some(ref key) = cohere_api_key {
        builder = builder
            .add_instance(ProviderType::Cohere, "command-r-plus", key)
            .supports("summary");
    }

    // Perplexity
    if let Some(ref key) = perplexity_api_key {
        builder = builder
            .add_instance(ProviderType::Perplexity, "llama-3.1-sonar-large-128k-online", key)
            .supports("summary");
    }

    // Finalize the manager configuration
    let manager = builder.build().await?;

    // Get provider count asynchronously
    let provider_count = manager.get_provider_count().await;
    info!("LlmManager configured with {} providers.", provider_count);

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


    // --- Streaming Example ---
    println!("\n=== Streaming Example ===");
    println!("Generating a haiku with streaming output...\n");

    let stream_request = GenerationRequest::builder("Write a haiku about falling leaves in autumn.")
        .task("short_poem")
        .build();

    match manager.generate_stream(stream_request).await {
        Ok(mut stream) => {
            print!("Streaming response: ");
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Print each chunk as it arrives (no newline to show continuous stream)
                        print!("{}", chunk.content);
                        // Flush stdout to ensure immediate display
                        use std::io::Write;
                        std::io::stdout().flush().ok();

                        // Check if this is the final chunk
                        if chunk.is_final {
                            println!("\n");
                            if let Some(usage) = chunk.usage {
                                println!("Token usage - Prompt: {}, Completion: {}, Total: {}",
                                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens);
                            }
                        }
                    }
                    Err(e) => {
                        println!("\nStreaming error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            println!("Failed to start streaming: {}", e);
        }
    }


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

    // Print token usage asynchronously
    manager.print_token_usage().await;

    Ok(())
}

/// Fetches models from all providers and prints them in a table format
async fn print_available_models(
    anthropic_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    mistral_api_key: Option<&str>,
    google_api_key: Option<&str>
) {
    println!("\n=== AVAILABLE MODELS ===");

    // Create a map to store models by provider
    let mut models_by_provider: HashMap<ProviderType, Vec<ModelInfo>> = HashMap::new();

    // Build futures only for providers with API keys
    let mut futures: Vec<(ProviderType, _)> = Vec::new();

    if let Some(key) = anthropic_api_key {
        let key = key.to_string();
        futures.push((
            ProviderType::Anthropic,
            tokio::spawn(async move { ModelDiscovery::list_anthropic_models(&key).await })
        ));
    }

    if let Some(key) = openai_api_key {
        let key = key.to_string();
        futures.push((
            ProviderType::OpenAI,
            tokio::spawn(async move { ModelDiscovery::list_openai_models(&key).await })
        ));
    }

    if let Some(key) = mistral_api_key {
        let key = key.to_string();
        futures.push((
            ProviderType::Mistral,
            tokio::spawn(async move { ModelDiscovery::list_mistral_models(&key).await })
        ));
    }

    if let Some(key) = google_api_key {
        let key = key.to_string();
        futures.push((
            ProviderType::Google,
            tokio::spawn(async move { ModelDiscovery::list_google_models(&key).await })
        ));
    }

    // Always try local providers (no API key required)
    futures.push((
        ProviderType::Ollama,
        tokio::spawn(async { ModelDiscovery::list_ollama_models(None).await })
    ));
    futures.push((
        ProviderType::LMStudio,
        tokio::spawn(async { ModelDiscovery::list_lmstudio_models(None).await })
    ));
    // Perplexity has a static model list
    futures.push((
        ProviderType::Perplexity,
        tokio::spawn(async { ModelDiscovery::list_perplexity_models().await })
    ));

    // Process results
    for (provider, future) in futures {
        match future.await {
            Ok(Ok(models)) => {
                models_by_provider.insert(provider, models);
            },
            Ok(Err(_)) => {
                // Silently skip errors for optional providers
            },
            Err(e) => {
                println!("Task error fetching {} models: {}", provider, e);
            }
        }
    }

    // Print models in a table format
    println!("\n{:<15} {:<40}", "PROVIDER", "MODEL NAME");
    println!("{}", "=".repeat(55));

    // Print models in a consistent provider order
    let display_order = [
        ProviderType::Anthropic,
        ProviderType::OpenAI,
        ProviderType::Mistral,
        ProviderType::Google,
        ProviderType::Groq,
        ProviderType::TogetherAI,
        ProviderType::Cohere,
        ProviderType::Perplexity,
        ProviderType::Ollama,
        ProviderType::LMStudio,
    ];

    for provider in display_order.iter() {
        if let Some(models) = models_by_provider.get(provider) {
            if models.is_empty() { continue; }
            for model in models {
                println!("{:<15} {:<40}", provider.to_string(), model.name);
            }
            // Add a separator between providers
            println!("{}", "-".repeat(55));
        }
    }
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