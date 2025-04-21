use crate::providers::{LlmProvider, LlmRequest, Message};
use crate::errors::LlmResult;
use crate::errors::LlmError;
use crate::load_balancer::instances::{LlmInstance, InstanceMetrics};
use crate::load_balancer::strategies;
use crate::load_balancer::strategies::LoadBalancingStrategy;
use crate::load_balancer::tasks::TaskDefinition;
use crate::{create_provider, ProviderType};
use std::time::Instant;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use futures::future::join_all;
use log::{debug,warn};

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
    tasks_to_instances: Arc<Mutex<HashMap<String, Vec<usize>>>>,
    instance_counter: Mutex<usize>,
}

impl LlmManager {
    pub fn new() -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(Box::new(strategies::LeastRecentlyUsedStrategy::new()))),
            tasks_to_instances: Arc::new(Mutex::new(HashMap::new())),
            instance_counter: Mutex::new(0),
        }
    }

    pub fn new_with_strategy(strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(strategy)),
            tasks_to_instances: Arc::new(Mutex::new(HashMap::new())),
            instance_counter: Mutex::new(0),
        }
    }

    pub fn add_provider(&mut self, provider_type: ProviderType, api_key: String, model: String, tasks: Vec<TaskDefinition>, enabled: bool){
        debug!("Creating provider with model {}", model);
        let provider = create_provider(provider_type, api_key, model, tasks, enabled);
        self.add_instance(provider);
    }

    pub fn add_instance(&mut self, provider: Arc<dyn LlmProvider + Send + Sync>) {
        let id = { // Scope counter lock
            let mut counter = self.instance_counter.lock().unwrap();
            let id = *counter;
            *counter += 1;
            id
        };

        let new_instance = LlmInstance::new(id, provider.clone()); // Clone provider Arc for the instance
        debug!("Adding instance {} ({})", id, provider.get_name());

        { 
            let supported_tasks = provider.get_supported_tasks();
            let mut task_map = self.tasks_to_instances.lock().unwrap();
            for task_name in supported_tasks.keys() {
                task_map.entry(task_name.clone())
                    .or_insert_with(Vec::new) 
                    .push(id); 
                 debug!("Added instance {} to task mapping for '{}'", id, task_name);
            }
        } 

        { 
            let mut instances = self.instances.lock().unwrap();
            instances.push(new_instance);
        } 
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) {
        let mut current_strategy = self.strategy.lock().unwrap();
        *current_strategy = strategy;
    }
    
    pub async fn generate_response(&self, prompt: &str, task: Option<&str>, request_params: Option<HashMap<String, serde_json::Value>>) -> LlmResult<String> {
        let start_time = Instant::now();

        let (selected_provider_arc, selected_instance_id, parameters) = { // Scope for locks

            // 1. Get candidate instance IDs based on task (if any)
            let candidate_ids: Option<Vec<usize>> = match task {
                Some(task_name) => {
                    let task_map = self.tasks_to_instances.lock().unwrap();
                    // Clone the vec of IDs if found, otherwise None
                    task_map.get(task_name).cloned()
                }
                None => None, // No specific task, consider all instances initially
            };

            // If a task was specified but no instances support it, return error early
            if task.is_some() && candidate_ids.is_none() {
                 warn!("No instances found supporting task: '{}'", task.unwrap());
                return Err(LlmError::ConfigError(format!("No providers available for task: {}", task.unwrap())));
            }

            // 2. Filter candidates by availability and collect references + metrics
            let eligible_instances_data: Vec<&LlmInstance>; // Store references
            let instances = self.instances.lock().unwrap(); // Lock main instance list

            if instances.is_empty() {
                 warn!("No LLM providers configured.");
                return Err(LlmError::ConfigError("No LLM providers available".to_string()));
            }

            match candidate_ids {
                Some(ids) => {
                    // Task specified: Filter the *main* list by ID presence *and* enabled status
                     debug!("Filtering instances for task '{}' using IDs: {:?}", task.unwrap(), ids);
                    eligible_instances_data = instances.iter()
                        .filter(|inst| ids.contains(&inst.id) && inst.is_enabled())
                        .collect();
                     debug!("Found {} eligible instances for task '{}'", eligible_instances_data.len(), task.unwrap());
                }
                None => {
                    // No task specified: Filter *all* instances by enabled status
                     debug!("No specific task. Filtering all enabled instances.");
                    eligible_instances_data = instances.iter()
                        .filter(|inst| inst.is_enabled())
                        .collect();
                     debug!("Found {} eligible instances (no task)", eligible_instances_data.len());
                }
            }

            // 3. Check if any eligible instances remain
            if eligible_instances_data.is_empty() {
                let error_msg = format!("No enabled providers available{}",
                    task.map_or_else(|| "".to_string(), |t| format!(" for task: '{}'", t))
                );
                 warn!("{}", error_msg);
                return Err(LlmError::ConfigError(error_msg));
            }

            // 4. Get metrics for eligible instances
            let eligible_metrics: Vec<InstanceMetrics> = eligible_instances_data.iter()
                .map(|inst| inst.get_metrics())
                .collect();

            // 5. Select instance using strategy
             let selected_metric_index = { // Scope strategy lock
                 let mut strategy = self.strategy.lock().unwrap();
                 strategy.select_instance(&eligible_metrics) // Select returns the ID
             };

            // Find the LlmInstance reference from the eligible list using the selected ID
             let selected_instance_id = eligible_metrics[selected_metric_index].id;
             let selected_instance_ref = eligible_instances_data.iter()
                 .find(|inst| inst.id == selected_instance_id)
                 .map(|inst_ref| *inst_ref) // Dereference &&LlmInstance to &LlmInstance
                 .expect("Selected instance ID from metrics not found in eligible list"); // Should not happen


            debug!("Selected instance {} ({}) for the request.", selected_instance_ref.id, selected_instance_ref.provider.get_name());

            // 6. Merge parameters
            let mut final_params = HashMap::new();
            if let Some(task_name) = task {
                // Get task definition directly from the provider associated with the selected instance
                if let Some(task_def) = selected_instance_ref.provider.get_supported_tasks().get(task_name) {
                    final_params.extend(task_def.parameters.clone()); // Clone only parameters
                    debug!("Applied parameters from task '{}' for instance {}", task_name, selected_instance_ref.id);
                } else {
                     warn!("Task '{}' not found in supported tasks for selected instance {} ({}), though it passed filtering. Potential inconsistency?",
                           task_name, selected_instance_ref.id, selected_instance_ref.provider.get_name());
                }
            }
            if let Some(req_params) = request_params {
                final_params.extend(req_params);
                 debug!("Applied request-specific parameters for instance {}", selected_instance_ref.id);
            }

            // 7. Clone provider Arc and return needed info
            (selected_instance_ref.provider.clone(), selected_instance_ref.id, final_params)

        }; // All locks (task_map, instances, strategy) released here


        // --- Execution and Metric Recording ---

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
            model: None, // Let provider use its configured model
            max_tokens,
            temperature,
        };

        debug!("Instance {} ({}) sending request to provider...", selected_instance_id, selected_provider_arc.get_name());
        let result = selected_provider_arc.generate(&request).await;
        let duration = start_time.elapsed();
        debug!("Instance {} ({}) received result in {:?}", selected_instance_id, selected_provider_arc.get_name(), duration);

        // Lock instances again briefly to update metrics
        {
            let mut instances_guard = self.instances.lock().unwrap(); // Renamed guard
            if let Some(instance_mut) = instances_guard.iter_mut().find(|inst| inst.id == selected_instance_id) {
                 debug!("Recording result for instance {}", selected_instance_id);
                instance_mut.record_result(duration, &result);
            } else {
                 warn!("Instance {} not found for metric update after request completion.", selected_instance_id);
            }
        } // instances_guard lock released

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

    pub fn get_provider_stats(&self) -> Vec<InstanceMetrics> {
        let instances = self.instances.lock().unwrap();
        instances
            .iter()
            .map(|instance| instance.get_metrics())
            .collect()
    }
}