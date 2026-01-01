//! Example: Using FlyLLM with Prometheus metrics
//!
//! This example demonstrates how to set up metrics collection
//! for monitoring LLM operations with Prometheus and Grafana.
//!
//! ## Quick Start
//!
//! 1. Start the monitoring stack:
//!    ```bash
//!    cd monitoring && ./start.sh up
//!    ```
//!
//! 2. Run this example:
//!    ```bash
//!    OPENAI_API_KEY=your-key cargo run --example metrics --features metrics
//!    ```
//!
//! 3. Open Grafana at http://localhost:3000 (admin/admin)
//!    The FlyLLM dashboard is pre-loaded.

use flyllm::{GenerationRequest, LlmManager, ProviderType, TaskDefinition};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::env;

#[tokio::main]
async fn main() {
    // Initialize logging
    flyllm::use_logging();

    println!("=== FlyLLM Metrics Example ===\n");

    // Set up Prometheus exporter - this exposes metrics at http://127.0.0.1:9090/metrics
    // The monitoring stack's Prometheus will scrape this endpoint
    PrometheusBuilder::new()
        .with_http_listener(([0, 0, 0, 0], 9090))
        .install()
        .expect("Failed to install Prometheus exporter");

    println!("Prometheus metrics exposed at http://127.0.0.1:9090/metrics");

    // Describe FlyLLM metrics for better Prometheus discovery
    #[cfg(feature = "metrics")]
    {
        flyllm::describe_metrics();
        println!("FlyLLM metrics described for Prometheus discovery");
    }

    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("\nWarning: OPENAI_API_KEY not set, using placeholder");
        println!("Set it to see actual metrics: OPENAI_API_KEY=sk-... cargo run --example metrics --features metrics\n");
        "sk-placeholder".to_string()
    });

    // Build the manager
    let manager = LlmManager::builder()
        .define_task(TaskDefinition::new("chat").with_max_tokens(100))
        .add_instance(ProviderType::OpenAI, "gpt-4o-mini", &api_key)
        .supports("chat")
        .build()
        .await
        .expect("Failed to build manager");

    println!("\n--- Available Metrics ---");
    println!("llm_requests_total           - Total requests by provider/model/task");
    println!("llm_request_duration_seconds - Request latency histogram");
    println!("llm_tokens_prompt_total      - Prompt tokens consumed");
    println!("llm_tokens_completion_total  - Completion tokens generated");
    println!("llm_errors_total             - Errors by type");
    println!("llm_provider_healthy         - Provider health status");
    println!("llm_retries_total            - Retry attempts");
    println!("llm_rate_limits_total        - Rate limit events");

    // Make sample requests to generate metrics
    if api_key != "sk-placeholder" {
        println!("\n--- Making sample requests ---");

        for i in 1..=3 {
            println!("Request {}...", i);
            let request = GenerationRequest {
                prompt: "Say hello in one word.".to_string(),
                task: Some("chat".to_string()),
                params: None,
            };

            match manager.generate_sequentially(vec![request]).await.pop() {
                Some(response) => {
                    println!("  Response: {}", response.content);
                }
                None => {
                    println!("  No response received");
                }
            }

            // Small delay between requests
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        println!("\nMetrics have been recorded for these requests.");
    } else {
        println!("\n(Skipping actual requests - no API key configured)");
    }

    // Print token usage (built-in tracking)
    println!("\n--- Token Usage ---");
    manager.print_token_usage().await;

    println!("\n=== Server Running ===");
    println!("Prometheus metrics: http://127.0.0.1:9090/metrics");
    println!("Grafana dashboard:  http://localhost:3000 (if monitoring stack is running)");
    println!("\nMaking periodic requests to generate metrics. Press Ctrl+C to exit...\n");

    // Keep making periodic requests so the rate graphs show data
    if api_key != "sk-placeholder" {
        let mut request_count = 4u32; // We already made 3
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;

            let request = GenerationRequest {
                prompt: "Say hi.".to_string(),
                task: Some("chat".to_string()),
                params: None,
            };

            match manager.generate_sequentially(vec![request]).await.pop() {
                Some(response) => {
                    println!("[Request #{}] Response: {}", request_count, response.content.trim());
                    request_count += 1;
                }
                None => {
                    println!("[Request #{}] No response", request_count);
                }
            }
        }
    } else {
        println!("(No API key - server running but not making requests)");
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        }
    }
}
