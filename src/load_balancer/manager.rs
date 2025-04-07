use crate::providers::{LlmProvider, LlmRequest, Message};
use crate::errors::LlmResult;
use crate::errors::LlmError;
use crate::load_balancer::instances::{LlmInstance, InstanceMetrics};
use crate::load_balancer::strategies;
use crate::load_balancer::strategies::LoadBalancingStrategy;
use crate::load_balancer::tasks::TaskDefinition;
use std::time::Instant;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use futures::future::join_all;

#[derive(Clone)]
pub struct LlmManagerRequest {
    pub prompt: String,
    pub task: Option<String>,
    pub params: Option<HashMap<String, serde_json::Value>>,
}

pub struct LlmManagerResponse {
    pub content: String,
    pub success: bool,
    pub error: Option<String>,
}

pub struct LlmManager {
    instances: Arc<Mutex<Vec<LlmInstance>>>,
    strategy: Arc<Mutex<Box<dyn LoadBalancingStrategy + Send + Sync>>>,
    task_definitions: HashMap<String, TaskDefinition>,
}

impl LlmManager {
    pub fn new() -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(Box::new(strategies::RoundRobinStrategy::new()))),
            task_definitions: HashMap::new(),
        }
    }

    pub fn new_with_strategy(strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(strategy)),
            task_definitions: HashMap::new(),
        }
    }

    pub fn add_instance(&mut self, provider: Arc<dyn LlmProvider + Send + Sync>) {
        let mut instances = self.instances.lock().unwrap();
        instances.push(LlmInstance::new(provider));
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) {
        let mut current_strategy = self.strategy.lock().unwrap();
        *current_strategy = strategy;
    }
    
    pub async fn generate_response(&self, prompt: &str, task: Option<&str>, request_params: Option<HashMap<String, serde_json::Value>>) -> LlmResult<String> {
        let mut instances = self.instances.lock().unwrap();
        
        if instances.is_empty() {
            return Err(LlmError::ConfigError("No LLM providers available".to_string()));
        }
        
        // Get eligible providers and their metrics
        let eligible_metrics: Vec<InstanceMetrics> = instances.iter()
            .enumerate()
            .filter_map(|(idx, inst)| {
                let is_eligible = match task {
                    Some(task_name) => inst.supported_tasks.contains_key(task_name) && inst.provider.is_enabled(),
                    None => inst.provider.is_enabled(),
                };
                
                if is_eligible {
                    Some(inst.get_metrics(idx))
                } else {
                    None
                }
            })
            .collect();
        
        if eligible_metrics.is_empty() {
            return Err(LlmError::ConfigError(format!("No providers available for task: {}", task.unwrap_or("default"))));
        }
        
        // Lock to get access to strategy
        let mut strategy = self.strategy.lock().unwrap();
        
        // Select provider using the strategy
        let provider_idx = strategy.select_instance(&eligible_metrics);
        
        let instance = &mut instances[provider_idx];
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
        
        // We need to clone the provider to use it outside the lock
        let provider = instance.provider.clone();
        
        // Drop locks before the async call to avoid deadlocks
        drop(strategy);
        drop(instances);
        
        let start_time = Instant::now();
        match provider.generate(&request).await {
            Ok(response) => {
                let duration = start_time.elapsed();
                
                // Update instance stats after completion
                let mut instances = self.instances.lock().unwrap();
                let instance = &mut instances[provider_idx];
                instance.response_times.push(duration);
                
                if instance.response_times.len() > 10 {
                    instance.response_times.remove(0);
                }
                
                Ok(response.content)
            },
            Err(e) => {
                // Update error count
                let mut instances = self.instances.lock().unwrap();
                let instance = &mut instances[provider_idx];
                instance.error_count += 1;
                
                Err(e)
            }
        }
    }

    pub async fn batch_generate(&self, requests: Vec<LlmManagerRequest>) -> Vec<LlmManagerResponse> {
        let futures = requests.into_iter().map(|request| {
            let prompt = request.prompt;
            let task_str = request.task;
            let params = request.params;
            let manager = self;
            
            
            async move {
                let task_ref = task_str.as_deref();
                match manager.generate_response(&prompt, task_ref, params).await {
                    Ok(content) => LlmManagerResponse {
                        content,
                        success: true,
                        error: None,
                    },
                    Err(e) => LlmManagerResponse {
                        content: String::new(),
                        success: false,
                        error: Some(e.to_string()),
                    },
                }
            }
        }).collect::<Vec<_>>();
        
        join_all(futures).await
    }

    pub fn register_task(&mut self, task: TaskDefinition) {
        self.task_definitions.insert(task.name.clone(), task);
    }

    pub fn assign_task_to_provider(&mut self, provider_idx: usize, task_name: &str, parameters: Option<HashMap<String, serde_json::Value>>) {
        let mut instances = self.instances.lock().unwrap();
        if provider_idx < instances.len() {
            let param_map = parameters.unwrap_or_default();
            instances[provider_idx].supported_tasks.insert(task_name.to_string(), param_map);
        }
    }

    pub fn get_provider_stats(&self) -> Vec<ProviderStats> {
        let instances = self.instances.lock().unwrap();
        instances
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