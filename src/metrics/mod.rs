//! Metrics module for FlyLLM
//!
//! This module provides optional metrics emission for monitoring LLM operations.
//! Enable with the `metrics` feature flag.
//!
//! # Example
//!
//! ```ignore
//! use flyllm::metrics::describe_metrics;
//! use metrics_exporter_prometheus::PrometheusBuilder;
//!
//! // User sets up their preferred exporter
//! // Note: requires `metrics-exporter-prometheus` in your dependencies
//! PrometheusBuilder::new()
//!     .with_http_listener(([127, 0, 0, 1], 9090))
//!     .install()
//!     .expect("prometheus setup");
//!
//! // Describe metrics (optional, improves Prometheus discovery)
//! describe_metrics();
//! ```

pub mod labels;
mod recorder;

#[cfg(feature = "metrics-server")]
pub mod dashboard;

pub use recorder::*;

/// Metric name constants
pub mod names {
    /// Total number of LLM requests
    pub const REQUESTS_TOTAL: &str = "llm_requests_total";
    /// Request duration in seconds
    pub const REQUEST_DURATION: &str = "llm_request_duration_seconds";
    /// Total prompt tokens consumed
    pub const TOKENS_PROMPT: &str = "llm_tokens_prompt_total";
    /// Total completion tokens generated
    pub const TOKENS_COMPLETION: &str = "llm_tokens_completion_total";
    /// Total number of errors by type
    pub const ERRORS_TOTAL: &str = "llm_errors_total";
    /// Provider health status (1=healthy, 0=unhealthy)
    pub const PROVIDER_HEALTHY: &str = "llm_provider_healthy";
    /// Total number of retry attempts
    pub const RETRIES_TOTAL: &str = "llm_retries_total";
    /// Total number of rate limit responses
    pub const RATE_LIMITS_TOTAL: &str = "llm_rate_limits_total";
}

/// Describe all metrics with their units and descriptions.
/// Call this after setting up your metrics exporter for better discovery.
pub fn describe_metrics() {
    use metrics::{describe_counter, describe_gauge, describe_histogram, Unit};

    describe_counter!(
        names::REQUESTS_TOTAL,
        Unit::Count,
        "Total number of LLM requests"
    );
    describe_histogram!(
        names::REQUEST_DURATION,
        Unit::Seconds,
        "Request duration in seconds"
    );
    describe_counter!(
        names::TOKENS_PROMPT,
        Unit::Count,
        "Total prompt tokens consumed"
    );
    describe_counter!(
        names::TOKENS_COMPLETION,
        Unit::Count,
        "Total completion tokens generated"
    );
    describe_counter!(
        names::ERRORS_TOTAL,
        Unit::Count,
        "Total number of errors by type"
    );
    describe_gauge!(
        names::PROVIDER_HEALTHY,
        Unit::Count,
        "Provider health status (1=healthy, 0=unhealthy)"
    );
    describe_counter!(
        names::RETRIES_TOTAL,
        Unit::Count,
        "Total number of retry attempts"
    );
    describe_counter!(
        names::RATE_LIMITS_TOTAL,
        Unit::Count,
        "Total number of rate limit responses"
    );
}
