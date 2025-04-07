// load_balancer/instances.rs
use crate::providers::LlmProvider;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc; 

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

    pub fn supports_task(&self, task_name: &str) -> bool {
        self.supported_tasks.contains_key(task_name)
    }

    pub fn is_enabled(&self) -> bool {
        self.provider.is_enabled()
    }
}