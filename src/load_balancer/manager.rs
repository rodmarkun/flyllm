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
use log::debug;

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
    instance_counter: Mutex<usize>,
}

impl LlmManager {
    pub fn new() -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(Box::new(strategies::RoundRobinStrategy::new()))),
            task_definitions: HashMap::new(),
            instance_counter: Mutex::new(0),
        }
    }

    pub fn new_with_strategy(strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(strategy)),
            task_definitions: HashMap::new(),
            instance_counter: Mutex::new(0),
        }
    }

    pub fn add_instance(&mut self, provider: Arc<dyn LlmProvider + Send + Sync>) {
        let mut instances = self.instances.lock().unwrap();
        let mut counter = self.instance_counter.lock().unwrap();
        let id = *counter;
        *counter += 1;
        debug!("Adding instance {} ({})", id, provider.get_name());
        instances.push(LlmInstance::new(id, provider));
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) {
        let mut current_strategy = self.strategy.lock().unwrap();
        *current_strategy = strategy;
    }
    
    pub async fn generate_response(&self, prompt: &str, task: Option<&str>, request_params: Option<HashMap<String, serde_json::Value>>) -> LlmResult<String> {
        let start_time = Instant::now();

        let (selected_provider, selected_instance_id, parameters) = { // Use a block to scope the lock
            let instances = self.instances.lock().unwrap();

            if instances.is_empty() {
                return Err(LlmError::ConfigError("No LLM providers available".to_string()));
            }

            // Get eligible providers and their metrics
            let eligible_instances: Vec<(usize, &LlmInstance)> = instances.iter()
                .enumerate()
                .filter(|(_, inst)| {
                    let is_eligible = match task {
                        Some(task_name) => inst.supports_task(task_name) && inst.provider.is_enabled(),
                        None => inst.provider.is_enabled(),
                    };
                    is_eligible
                })
                .collect();


            let eligible_metrics: Vec<InstanceMetrics> = eligible_instances.iter()
                .map(|(_, inst)| inst.get_metrics())
                .collect();


            if eligible_metrics.is_empty() {
                return Err(LlmError::ConfigError(format!("No providers available for task: {}", task.unwrap_or("default"))));
            }

            let selected_index_in_eligible = {
                let mut strategy = self.strategy.lock().unwrap();
                strategy.select_instance(&eligible_metrics)
            };

            let selected_instance_id = eligible_metrics[selected_index_in_eligible].id;
            let (_original_index, instance) = instances.iter().enumerate()
                                                  .find(|(_, inst)| inst.id == selected_instance_id)
                                                  .expect("Selected instance ID not found in original list"); // Should not happen


            debug!("Selected instance {} ({}) for the request.", instance.id, instance.provider.get_name());

            // Merge task-specific parameters with request parameters
            let mut final_params = HashMap::new();
            if let Some(task_name) = task {
                if let Some(task_params) = instance.supported_tasks.get(task_name) {
                    final_params.extend(task_params.clone());
                }
            }
            if let Some(req_params) = request_params {
                final_params.extend(req_params);
            }

            // Clone the provider Arc and return necessary info
            (instance.provider.clone(), instance.id, final_params)

        }; 


        let max_tokens = parameters.get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let temperature = parameters.get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let request = LlmRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            model: None, 
            max_tokens,
            temperature,
        };

        debug!("Instance {} ({}) processing request...", selected_instance_id, selected_provider.get_name());
        let result = selected_provider.generate(&request).await;
        let duration = start_time.elapsed();

        {
            let mut instances = self.instances.lock().unwrap();
            // Find the instance again to update its metrics
            if let Some(instance) = instances.iter_mut().find(|inst| inst.id == selected_instance_id) {
                 instance.record_result(duration, &result); // Use the new helper method
            } else {
                 log::warn!("Instance {} not found for metric update after request.", selected_instance_id);
            }
        } 

        match result {
            Ok(response) => Ok(response.content),
            Err(e) => Err(e),
        }
    }

     pub async fn batch_generate(&self, requests: Vec<LlmManagerRequest>) -> Vec<LlmManagerResponse> {
        let futures = requests.into_iter().map(|request| {
            let prompt = request.prompt;
            let task_str = request.task;
            let params = request.params;
            async move {
                let task_ref = task_str.as_deref();
                match self.generate_response(&prompt, task_ref, params).await {
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
        debug!("Registering task: {:?}", task);
        self.task_definitions.insert(task.name.clone(), task);
    }

    pub fn assign_task_to_provider(&mut self, provider_idx: usize, task_name: &str, parameters: Option<HashMap<String, serde_json::Value>>) {
        debug!("Assigning task {} to provider {}", task_name, provider_idx);
        let mut instances = self.instances.lock().unwrap();
        if provider_idx < instances.len() {
            let param_map = parameters.unwrap_or_default();
            instances[provider_idx].supported_tasks.insert(task_name.to_string(), param_map);
        }
    }

    pub fn get_provider_stats(&self) -> Vec<InstanceMetrics> {
        let instances = self.instances.lock().unwrap();
        instances
            .iter()
            .map(|instance| instance.get_metrics())
            .collect()
    }
}