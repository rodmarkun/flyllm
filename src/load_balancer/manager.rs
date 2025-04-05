use crate::providers::{LlmProvider, LlmRequest, Message};
use crate::errors::LlmResult;
use crate::load_balancer::instances::LlmInstance;
use crate::load_balancer::strategies;
use crate::load_balancer::strategies::LoadBalancingStrategy;
use std::time::Instant;

pub struct LlmManager {
    instances: Vec<LlmInstance>,
    strategy: Box<dyn LoadBalancingStrategy + Send + Sync>,
}

impl LlmManager {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            strategy: Box::new(strategies::RoundRobinStrategy::new())
        }
    }

    pub fn new_with_strategy(strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) -> Self {
        Self {
            instances: Vec::new(),
            strategy,
        }
    }

    pub fn add_instance(&mut self, provider: Box<dyn LlmProvider + Send + Sync>) {
        self.instances.push(LlmInstance::new(provider));
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) {
        self.strategy = strategy;
    }
    
    pub async fn generate_response(&mut self, prompt: &str) -> LlmResult<String> {
        if self.instances.is_empty() {
            return Err(crate::errors::LlmError::ConfigError("No LLM providers available".to_string()));
        }

        let provider_idx = self.strategy.select_instance(&self.instances);
        
        if provider_idx >= self.instances.len() {
            return Err(crate::errors::LlmError::ConfigError("Selected provider index out of bounds".to_string()));
        }

        let start_time = Instant::now();
        let instance = &mut self.instances[provider_idx];
        instance.last_used = Instant::now();
        instance.request_count += 1;

        let request = LlmRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            model: None,
            max_tokens: None,
            temperature: None,
        };

        match instance.provider.generate(&request).await {
            Ok(response) => {
                let duration = start_time.elapsed();
                instance.response_times.push(duration);
                
                // Keep only the last 10 response times to have a reasonable moving average
                if instance.response_times.len() > 10 {
                    instance.response_times.remove(0);
                }
                
                Ok(response.content)
            },
            Err(e) => {
                instance.error_count += 1;
                Err(e)
            }
        }
    }

    pub fn get_provider_stats(&self) -> Vec<ProviderStats> {
        self.instances
            .iter()
            .map(|p| ProviderStats {
                avg_response_time_ms: p.avg_response_time().as_millis() as u64,
                request_count: p.request_count,
                error_count: p.error_count,
                error_rate: if p.request_count > 0 {
                    (p.error_count as f64 / p.request_count as f64) * 100.0
                } else {
                    0.0
                },
            })
            .collect()
    }
}

pub struct ProviderStats {
    pub avg_response_time_ms: u64,
    pub request_count: usize,
    pub error_count: usize,
    pub error_rate: f64,
}