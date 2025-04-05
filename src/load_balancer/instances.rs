use crate::providers::{provider, LlmProvider};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

pub struct LlmInstance {
    pub provider: Box<dyn LlmProvider + Send + Sync>,
    pub last_used: Instant,
    pub response_times: Vec<Duration>,
    pub request_count: usize,
    pub error_count: usize,
}

impl LlmInstance {
    pub fn new(provider: Box<dyn LlmProvider + Send + Sync>) -> Self {
        Self {
            provider,
            last_used: Instant::now(),
            response_times: Vec::new(),
            request_count: 0,
            error_count: 0
        }
    }

    pub fn avg_response_time(&self) -> Duration {
        if self.response_times.is_empty() {
            return Duration::from_millis(0);
        }
        
        let total = self.response_times.iter().sum::<Duration>();
        total / self.response_times.len() as u32
    }

    pub fn get_complexity(&self) -> u32 {
        self.provider.get_complexity()
    }
}