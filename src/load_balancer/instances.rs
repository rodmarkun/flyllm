// load_balancer/instances.rs
use crate::providers::LlmProvider;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc; 

pub struct InstanceMetrics {
    pub index: usize,
    pub request_count: usize,
    pub error_count: usize,
    pub avg_response_time: Duration,
    pub error_rate: f64,
    pub available: bool,
    pub provider_name: String,
}

pub struct LlmInstance {
    pub provider: Arc<dyn LlmProvider + Send + Sync>,
    pub last_used: Instant,
    pub response_times: Vec<Duration>,
    pub request_count: usize,
    pub error_count: usize,
    pub supported_tasks: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl LlmInstance {
    pub fn new(provider: Arc<dyn LlmProvider + Send + Sync>) -> Self {
        Self {
            provider,
            last_used: Instant::now(),
            response_times: Vec::new(),
            request_count: 0,
            error_count: 0,
            supported_tasks: HashMap::new(),
        }
    }

    pub fn avg_response_time(&self) -> Duration {
        if self.response_times.is_empty() {
            return Duration::from_millis(0);
        }
        let total: Duration = self.response_times.iter().sum();
        total / self.response_times.len().max(1) as u32 // Avoid division by zero
    }

    pub fn get_error_rate(&self) -> f64 {
        if self.request_count > 0 {
            (self.error_count as f64 / self.request_count as f64) * 100.0
        } else {
            0.0
        }
    }

    pub fn supports_task(&self, task_name: &str) -> bool {
        self.supported_tasks.contains_key(task_name)
    }

    pub fn is_enabled(&self) -> bool {
        self.provider.is_enabled()
    }

    pub fn get_metrics(&self, index: usize) -> InstanceMetrics {
        let avg_time = self.avg_response_time();
        let error_rate = self.get_error_rate();
        
        InstanceMetrics {
            index,
            request_count: self.request_count,
            error_count: self.error_count,
            avg_response_time: avg_time,
            error_rate,
            available: self.is_enabled(),
            provider_name: self.provider.get_name().to_string(),
        }
    }
}