use log::debug;
use crate::providers::LlmProvider;
use crate::{LlmRequest, LlmResponse, LlmResult};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc; 

pub struct InstanceMetrics {
    pub id: usize,
    pub request_count: usize,
    pub error_count: usize,
    pub avg_response_time: Duration,
    pub error_rate: f64,
    pub available: bool,
    pub provider_name: String,
}

pub struct LlmInstance {
    pub id: usize,
    pub provider: Arc<dyn LlmProvider + Send + Sync>,
    pub last_used: Instant,
    pub response_times: Vec<Duration>,
    pub request_count: usize,
    pub error_count: usize,
    pub supported_tasks: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl LlmInstance {
    pub fn new(id: usize, provider: Arc<dyn LlmProvider + Send + Sync>) -> Self {
        Self {
            id,
            provider,
            last_used: Instant::now(),
            response_times: Vec::new(),
            request_count: 0,
            error_count: 0,
            supported_tasks: HashMap::new(),
        }
    }

    pub async fn generate(&mut self, request: &LlmRequest) -> LlmResult<LlmResponse> {
        let start_time = Instant::now();
        debug!("Instance {} ({}) processing request...", self.id, self.provider.get_name());
        match self.provider.generate(request).await {
            Ok(response) => {
                let duration = start_time.elapsed();
                self.last_used = Instant::now();
                self.request_count += 1;
                self.response_times.push(duration);
                if self.response_times.len() > 10 {
                    self.response_times.remove(0);
                }
                debug!("Instance {} ({}) successfully processed request in {:?}", self.id, self.provider.get_name(), duration);
                Ok(response)
            }
            Err(e) => {
                self.last_used = Instant::now();
                self.request_count += 1;
                self.error_count += 1;
                debug!("Instance {} ({}) failed to process request: {:?}", self.id, self.provider.get_name(), e);
                Err(e)
            }
        }
    }

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

    pub fn get_metrics(&self) -> InstanceMetrics {
        let avg_time = self.avg_response_time();
        let error_rate = self.get_error_rate();
        
        InstanceMetrics {
            id: self.id,
            request_count: self.request_count,
            error_count: self.error_count,
            avg_response_time: avg_time,
            error_rate,
            available: self.is_enabled(),
            provider_name: self.provider.get_name().to_string(),
        }
    }
}