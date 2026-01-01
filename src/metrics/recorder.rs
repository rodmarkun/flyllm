//! Metric recording functions

use std::time::Duration;

use crate::errors::LlmError;
use crate::providers::TokenUsage;

use super::{labels, names};

/// Record a successful LLM request
pub fn record_request_success(
    provider: &str,
    model: &str,
    task: Option<&str>,
    duration: Duration,
    usage: Option<&TokenUsage>,
) {
    let task_value = task.unwrap_or("default");

    // Increment request counter
    metrics::counter!(
        names::REQUESTS_TOTAL,
        labels::keys::PROVIDER => provider.to_string(),
        labels::keys::MODEL => model.to_string(),
        labels::keys::TASK => task_value.to_string()
    )
    .increment(1);

    // Record duration histogram
    metrics::histogram!(
        names::REQUEST_DURATION,
        labels::keys::PROVIDER => provider.to_string(),
        labels::keys::MODEL => model.to_string(),
        labels::keys::TASK => task_value.to_string()
    )
    .record(duration.as_secs_f64());

    // Record token usage if available
    if let Some(usage) = usage {
        metrics::counter!(
            names::TOKENS_PROMPT,
            labels::keys::PROVIDER => provider.to_string(),
            labels::keys::MODEL => model.to_string()
        )
        .increment(usage.prompt_tokens as u64);

        metrics::counter!(
            names::TOKENS_COMPLETION,
            labels::keys::PROVIDER => provider.to_string(),
            labels::keys::MODEL => model.to_string()
        )
        .increment(usage.completion_tokens as u64);
    }
}

/// Record a failed LLM request
pub fn record_request_failure(
    provider: &str,
    model: &str,
    task: Option<&str>,
    error: &LlmError,
    duration: Duration,
) {
    let task_value = task.unwrap_or("default");

    // Increment request counter (failures still count as requests)
    metrics::counter!(
        names::REQUESTS_TOTAL,
        labels::keys::PROVIDER => provider.to_string(),
        labels::keys::MODEL => model.to_string(),
        labels::keys::TASK => task_value.to_string()
    )
    .increment(1);

    // Record duration even for failures
    metrics::histogram!(
        names::REQUEST_DURATION,
        labels::keys::PROVIDER => provider.to_string(),
        labels::keys::MODEL => model.to_string(),
        labels::keys::TASK => task_value.to_string()
    )
    .record(duration.as_secs_f64());

    // Record error with type
    metrics::counter!(
        names::ERRORS_TOTAL,
        labels::keys::PROVIDER => provider.to_string(),
        labels::keys::MODEL => model.to_string(),
        labels::keys::ERROR_TYPE => labels::error_type_label(error).to_string()
    )
    .increment(1);

    // Track rate limits specifically
    if matches!(error, LlmError::RateLimit(_)) {
        metrics::counter!(
            names::RATE_LIMITS_TOTAL,
            labels::keys::PROVIDER => provider.to_string()
        )
        .increment(1);
    }
}

/// Record a retry attempt
pub fn record_retry(provider: &str) {
    metrics::counter!(
        names::RETRIES_TOTAL,
        labels::keys::PROVIDER => provider.to_string()
    )
    .increment(1);
}

/// Update provider health gauge
pub fn set_provider_health(provider: &str, healthy: bool) {
    metrics::gauge!(
        names::PROVIDER_HEALTHY,
        labels::keys::PROVIDER => provider.to_string()
    )
    .set(if healthy { 1.0 } else { 0.0 });
}
