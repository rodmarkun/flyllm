use crate::providers::{LlmProvider, LlmRequest, Message};
use crate::errors::LlmResult;
use crate::errors::LlmError;
use crate::load_balancer::instances::LlmInstance;
use crate::load_balancer::strategies;
use crate::load_balancer::strategies::LoadBalancingStrategy;
use crate::load_balancer::tasks::TaskDefinition;
use std::time::Instant;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct LlmBatchRequest {
    pub prompt: String,
    pub task: Option<String>,
    pub params: Option<HashMap<String, serde_json::Value>>,
}

pub struct LlmManager {
    instances: Vec<LlmInstance>,
    strategy: Box<dyn LoadBalancingStrategy + Send + Sync>,
    task_definitions: HashMap<String, TaskDefinition>,
}

impl LlmManager {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            strategy: Box::new(strategies::RoundRobinStrategy::new()),
            task_definitions: HashMap::new(),
        }
    }

    pub fn new_with_strategy(strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) -> Self {
        Self {
            instances: Vec::new(),
            strategy,
            task_definitions: HashMap::new(),
        }
    }

    pub fn add_instance(&mut self, provider: Arc<dyn LlmProvider + Send + Sync>) {
        self.instances.push(LlmInstance::new(provider));
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) {
        self.strategy = strategy;
    }
    
    pub async fn generate_response(&mut self, prompt: &str, task: Option<&str>, request_params: Option<HashMap<String, serde_json::Value>>) -> LlmResult<String> {
        if self.instances.is_empty() {
            return Err(LlmError::ConfigError("No LLM providers available".to_string()));
        }
        
        // Get providers that can handle this task
        let eligible_providers: Vec<usize> = if let Some(task_name) = task {
            self.instances.iter()
                .enumerate()
                .filter(|(_, inst)| inst.supported_tasks.contains_key(task_name) && inst.provider.is_enabled())
                .map(|(idx, _)| idx)
                .collect()
        } else {
            // If no task specified, use all enabled providers
            self.instances.iter()
                .enumerate()
                .filter(|(_, inst)| inst.provider.is_enabled())
                .map(|(idx, _)| idx)
                .collect()
        };
        
        if eligible_providers.is_empty() {
            return Err(LlmError::ConfigError(format!("No providers available for task: {}", task.unwrap_or("default"))));
        }
        
        // Select from eligible providers using the strategy
        let filtered_instances: Vec<&LlmInstance> = eligible_providers.iter()
            .map(|&idx| &self.instances[idx])
            .collect();
            
        // Here we assume an enhanced strategy trait that can work with a subset
        let provider_idx_in_filtered = self.strategy.select_instance(&filtered_instances);
        let provider_idx = eligible_providers[provider_idx_in_filtered];
        
        let instance = &mut self.instances[provider_idx];
        instance.last_used = Instant::now();
        instance.request_count += 1;
        
        // Merge task-specific parameters with request parameters
        let mut parameters = HashMap::new();
        if let Some(task_name) = task {
            if let Some(task_params) = instance.supported_tasks.get(task_name) {
                parameters.extend(task_params.clone());
            }
        }
        
        // Override with request-specific parameters if provided
        if let Some(req_params) = request_params {
            parameters.extend(req_params);
        }
        
        // Extract LLM parameters from the merged map
        let max_tokens = parameters.get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
            
        let temperature = parameters.get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        
        // Create and send request
        let request = LlmRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            model: None,
            max_tokens,
            temperature,
        };
        
        // Execute request and handle response as before
        let start_time = Instant::now();
        match instance.provider.generate(&request).await {
            Ok(response) => {
                let duration = start_time.elapsed();
                instance.response_times.push(duration);
                
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

    pub fn register_task(&mut self, task: TaskDefinition) {
        self.task_definitions.insert(task.name.clone(), task);
    }

    pub fn assign_task_to_provider(&mut self, provider_idx: usize, task_name: &str, parameters: Option<HashMap<String, serde_json::Value>>) {
        if provider_idx < self.instances.len() {
            let param_map = parameters.unwrap_or_default();
            self.instances[provider_idx].supported_tasks.insert(task_name.to_string(), param_map);
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