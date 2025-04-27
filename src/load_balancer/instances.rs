use log::debug;
use crate::providers::LlmProvider;
use crate::{LlmResponse, LlmResult};
use std::time::{Duration, Instant};
use std::sync::Arc; 

/// Metrics for an LLM instance used for monitoring and load balancing
pub struct InstanceMetrics {
    pub id: usize,
    pub model: String,
    pub request_count: usize,
    pub error_count: usize,
    pub avg_response_time: Duration,
    pub error_rate: f64,
    pub available: bool,
    pub provider_name: String,
    pub last_used: Instant,
}

/// An LLM provider instance with associated metrics
pub struct LlmInstance {
    pub id: usize,
    pub model: String,
    pub provider: Arc<dyn LlmProvider + Send + Sync>,
    pub last_used: Instant,
    pub response_times: Vec<Duration>,
    pub request_count: usize,
    pub error_count: usize,
}

impl LlmInstance {
    /// Create a new LLM instance
    ///
    /// # Parameters
    /// * `id` - Unique identifier for this instance
    /// * `provider` - Reference to the provider implementation
    pub fn new(id: usize, provider: Arc<dyn LlmProvider + Send + Sync>) -> Self {
        Self {
            id,
            model: provider.get_model().to_string(),
            provider,
            last_used: Instant::now(),
            response_times: Vec::new(),
            request_count: 0,
            error_count: 0,
        }
    }

    /// Record the result of a request for metrics tracking
    ///
    /// # Parameters
    /// * `duration` - How long the request took
    /// * `result` - The result of the request (success or error)
    pub fn record_result(&mut self, duration: Duration, result: &LlmResult<LlmResponse>) {
        self.last_used = Instant::now();
        self.request_count += 1;

        match result {
            Ok(_) => {
                self.response_times.push(duration);
                if self.response_times.len() > 10 {
                    self.response_times.remove(0);
                }
                debug!("Instance {} ({}) successfully processed request in {:?}", self.id, self.provider.get_name(), duration);
            }
            Err(e) => {
                self.error_count += 1;
                debug!("Instance {} ({}) failed to process request: {:?}", self.id, self.provider.get_name(), e);
            }
        }
    }

    /// Calculate the average response time from recent requests
    ///
    /// # Returns
    /// * Average duration, or zero if no requests recorded
    pub fn avg_response_time(&self) -> Duration {
        if self.response_times.is_empty() {
            return Duration::from_millis(0);
        }
        let total: Duration = self.response_times.iter().sum();
        total / self.response_times.len().max(1) as u32 // Avoid division by zero
    }

    /// Calculate the error rate as a percentage
    ///
    /// # Returns
    /// * Error rate from 0.0 to 100.0, or 0.0 if no requests
    pub fn get_error_rate(&self) -> f64 {
        if self.request_count > 0 {
            (self.error_count as f64 / self.request_count as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Check if this instance supports a specific task
    ///
    /// # Parameters
    /// * `task_name` - The task to check support for
    ///
    /// # Returns
    /// * Whether this instance supports the task
    pub fn supports_task(&self, task_name: &str) -> bool {
        self.provider.get_supported_tasks().contains_key(task_name)
    }

    /// Check if this instance is currently enabled
    ///
    /// # Returns
    /// * Whether this instance is enabled
    pub fn is_enabled(&self) -> bool {
        self.provider.is_enabled()
    }

    /// Get the current metrics for this instance
    ///
    /// # Returns
    /// * Metrics structure for this instance
    pub fn get_metrics(&self) -> InstanceMetrics {
        let avg_time = self.avg_response_time();
        let error_rate = self.get_error_rate();
        
        InstanceMetrics {
            id: self.id,
            model: self.model.clone(),
            request_count: self.request_count,
            error_count: self.error_count,
            avg_response_time: avg_time,
            error_rate,
            available: self.is_enabled(),
            provider_name: self.provider.get_name().to_string(),
            last_used: self.last_used
        }
    }
}