# FlyLLM

FlyLLM is a Rust library that provides a load-balanced, multi-provider client for Large Language Models. It enables developers to seamlessly work with multiple LLM providers (OpenAI, Anthropic, Google, Mistral...) through a unified API with request routing, load balancing, and failure handling.

## Features

- **Multiple Provider Support** 🌐: Unified interface for OpenAI, Anthropic, Google, and Mistral APIs
- **Task-Based Routing** 🧭: Route requests to the most appropriate provider based on predefined tasks
- **Load Balancing** ⚖️: Automatically distribute load across multiple provider instances
- **Failure Handling** 🛡️: Retry mechanisms and automatic failover between providers
- **Parallel Processing** ⚡: Process multiple requests concurrently for improved throughput
- **Custom Parameters** 🔧: Set provider-specific parameters per task or request
- **Usage Tracking** 📊: Monitor token consumption for cost management

## Installation

Add FlyLLM to your `Cargo.toml`:

```toml
[dependencies]
flyllm = "0.1.0"
tokio = { version = "1", features = ["full"] } # For async runtime
```

## Architecture
The LLM Manager (`LLMManager`) serves as the core component for orchestrating language model interactions in your application. It manages multiple LLM instances (`LLMInstance`), each defined by a model, API key, and supported tasks (`TaskDefinition`).

When your application sends a generation request (`GenerationRequest`), the manager automatically selects an appropriate instance based on configurable strategies (Last Recently Used, Quickest Response Time, etc.) and returns the generated response by the LLM (`LLMResponse`). This design prevents rate limiting by distributing requests across multiple instances (even of the same model) with different API keys.

The manager handles failures gracefully by re-routing requests to alternative instances. It also supports parallel execution for significant performance improvements when processing multiple requests simultaneously!

You can define default parameters (temperature, max_tokens) for each task while retaining the ability to override these settings in specific requests. The system also tracks token usage across all instances:

```
--- Token Usage Statistics ---
ID    Provider        Model                          Prompt Tokens   Completion Tokens Total Tokens
-----------------------------------------------------------------------------------------------
0     mistral         mistral-small-latest           109             897             1006
1     anthropic       claude-3-sonnet-20240229       133             1914            2047
2     anthropic       claude-3-opus-20240229         51              529             580
3     google          gemini-2.0-flash               0               0               0
4     openai          gpt-3.5-turbo                  312             1003            1315
```

## Usage Examples

You can activate FlyLLM's debug messages by setting the environment variable `RUST_LOG` to the value `"debug"`.

### Quick Start

```rust
use flyllm::{
    ProviderType, LlmManager, GenerationRequest, TaskDefinition,
    use_logging
};
use std::collections::HashMap;
use serde_json::json;

#[tokio::main]
async fn main() {
    // Initialize logging
    use_logging();

    // Create an LLM manager
    let mut manager = LlmManager::new();

    // Define a task with specific parameters
    let summary_task = TaskDefinition {
        name: "summary".to_string(),
        parameters: HashMap::from([
            ("max_tokens".to_string(), json!(500)),
            ("temperature".to_string(), json!(0.3)),
        ]),
    };

    // Add a provider with its supported tasks
    manager.add_provider(
        ProviderType::OpenAI,
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
        "gpt-3.5-turbo".to_string(),
        vec![summary_task.clone()],
        true,
    );

    // Create a request
    let request = GenerationRequest {
        prompt: "Summarize the following text: Climate change refers to long-term shifts in temperatures...".to_string(),
        task: Some("summary".to_string()),
        params: None,
    };

    // Generate response 
    // The Manager will automatically choose a provider fit for the task according to the selected strategy
    let responses = manager.generate_sequentially(vec![request]).await;
    
    // Handle the response
    if let Some(response) = responses.first() {
        if response.success {
            println!("Response: {}", response.content);
        } else {
            println!("Error: {}", response.error.as_ref().unwrap_or(&"Unknown error".to_string()));
        }
    }
}
```

### Adding Multiple Providers

```rust
// Add OpenAI
manager.add_provider(
    ProviderType::OpenAI,
    openai_api_key,
    "gpt-4-turbo".to_string(),
    vec![summary_task.clone(), qa_task.clone()],
    true,
);

// Add Anthropic
manager.add_provider(
    ProviderType::Anthropic,
    anthropic_api_key,
    "claude-3-sonnet-20240229".to_string(),
    vec![creative_writing_task.clone(), code_generation_task.clone()],
    true,
);

// Add Mistral
manager.add_provider(
    ProviderType::Mistral,
    mistral_api_key,
    "mistral-large-latest".to_string(),
    vec![summary_task.clone(), translation_task.clone()],
    true,
);

// Add Google (Gemini)
manager.add_provider(
    ProviderType::Google,
    google_api_key,
    "gemini-1.5-pro".to_string(),
    vec![qa_task.clone(), creative_writing_task.clone()],
    true,
);
```

### Task-Based Routing

```rust
// Define tasks with specific parameters
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

// Create requests with different tasks
let summary_request = GenerationRequest {
    prompt: "Summarize the following article: ...".to_string(),
    task: Some("summary".to_string()),
    params: None,
};

let story_request = GenerationRequest {
    prompt: "Write a short story about a robot discovering emotions.".to_string(),
    task: Some("creative_writing".to_string()),
    params: None,
};
```

### Parallel Processing

```rust
// Create multiple requests
let requests = vec![
    GenerationRequest { /* ... */ },
    GenerationRequest { /* ... */ },
    GenerationRequest { /* ... */ },
];

// Process in parallel
let parallel_results = manager.batch_generate(requests).await;

// Process each result
for result in parallel_results {
    if result.success {
        println!("Success: {}", result.content);
    } else {
        println!("Error: {}", result.error.as_ref().unwrap_or(&"Unknown error".to_string()));
    }
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are always welcome! If you're interested in contributing to FlyLLM, please fork the repository and create a new branch for your changes. When you're done with your changes, submit a pull request to merge your changes into the main branch.

## Supporting FlyLLM

If you want to support FlyLLM, you can:
- **Star** :star: the project in Github!
- **Donate** :coin: to my [Ko-fi](https://ko-fi.com/rodmarkun) page!
- **Share** :heart: the project with your friends!

## To-Do List
Next features planned:

- [x] ~~Usage tracking~~
- [ ] Log redirection to file
- [ ] Streaming responses
- [ ] Internal documentation
- [ ] Unit & Integration tests
- [ ] Builder Pattern
- [ ] Aggregation of more strategies