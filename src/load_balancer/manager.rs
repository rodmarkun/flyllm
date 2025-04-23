use crate::providers::{LlmProvider, LlmRequest, Message};
use crate::errors::LlmResult;
use crate::errors::LlmError;
use crate::load_balancer::instances::{LlmInstance, InstanceMetrics};
use crate::load_balancer::strategies;
use crate::load_balancer::strategies::LoadBalancingStrategy;
use crate::load_balancer::tasks::TaskDefinition;
use crate::{constants, create_provider, ProviderType};
use std::time::Instant;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use futures::future::join_all;
use log::{debug, info, warn}; // Added info

#[derive(Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub task: Option<String>,
    pub params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Clone)]
struct LlmManagerRequest {
    pub prompt: String,
    pub task: Option<String>,
    pub params: Option<HashMap<String, serde_json::Value>>,
    pub attempts: usize,
    pub failed_instances: Vec<usize>
}

impl LlmManagerRequest {
    // Create from user request
    fn from_generation_request(request: GenerationRequest) -> Self {
        Self {
            prompt: request.prompt,
            task: request.task,
            params: request.params,
            attempts: 0,
            failed_instances: Vec::new(),
        }
    }
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
    max_retries: usize
}

impl LlmManager {
    pub fn new() -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(Box::new(strategies::LeastRecentlyUsedStrategy::new()))),
            tasks_to_instances: Arc::new(Mutex::new(HashMap::new())),
            instance_counter: Mutex::new(0),
            max_retries: constants::DEFAULT_MAX_TRIES,
        }
    }

    pub fn new_with_strategy(strategy: Box<dyn LoadBalancingStrategy + Send + Sync>) -> Self {
        Self {
            instances: Arc::new(Mutex::new(Vec::new())),
            strategy: Arc::new(Mutex::new(strategy)),
            tasks_to_instances: Arc::new(Mutex::new(HashMap::new())),
            instance_counter: Mutex::new(0),
            max_retries: constants::DEFAULT_MAX_TRIES,
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

    pub async fn generate_sequentially(&self, requests: Vec<GenerationRequest>) -> Vec<LlmManagerResponse> {
        let mut responses = Vec::with_capacity(requests.len());
        info!("Entering generate_sequentially with {} requests", requests.len()); // *** NEW LOG ***

        for (index, request) in requests.into_iter().enumerate() { // Use enumerate for index
            info!("Starting sequential request index: {}", index); // *** NEW LOG ***
            let internal_request = LlmManagerRequest::from_generation_request(request);

            let response_result = self.generate_response(internal_request, None).await;
            info!("Sequential request index {} completed generate_response call.", index); // *** NEW LOG ***

            let response = match response_result {
                Ok(content) => {
                    info!("Sequential request index {} succeeded.", index); // *** NEW LOG ***
                    LlmManagerResponse {
                        content,
                        success: true,
                        error: None,
                    }
                },
                Err(e) => {
                    warn!("Sequential request index {} failed: {}", index, e); // *** NEW LOG ***
                    LlmManagerResponse {
                        content: String::new(),
                        success: false,
                        error: Some(e.to_string()),
                    }
                },
            };

            debug!("Pushing response for sequential request index {}", index); // *** NEW LOG ***
            responses.push(response);
            info!("Finished processing sequential request index {}", index); // *** NEW LOG ***
        }

        info!("Exiting generate_sequentially"); // *** NEW LOG ***
        responses
    }

    pub async fn batch_generate(&self, requests: Vec<GenerationRequest>) -> Vec<LlmManagerResponse> {
        info!("Entering batch_generate with {} requests", requests.len()); // *** NEW LOG ***
        // Correct way to convert GenerationRequest to LlmManagerRequest
        let internal_requests = requests.into_iter()
            .map(|request| LlmManagerRequest::from_generation_request(request))
            .collect::<Vec<_>>();

        let futures = internal_requests.into_iter().enumerate().map(|(index, request)| { // Use enumerate for index
            async move {
                info!("Starting parallel request index: {}", index); // *** NEW LOG ***
                match self.generate_response(request, None).await {
                    Ok(content) => {
                         info!("Parallel request index {} succeeded.", index); // *** NEW LOG ***
                        LlmManagerResponse {
                            content,
                            success: true,
                            error: None,
                        }
                    },
                    Err(e) => {
                        warn!("Parallel request index {} failed: {}", index, e); // *** NEW LOG ***
                        LlmManagerResponse {
                            content: String::new(),
                            success: false,
                            error: Some(e.to_string()),
                        }
                    },
                }
            }
        }).collect::<Vec<_>>();

        let results = join_all(futures).await;
        info!("Exiting batch_generate"); // *** NEW LOG ***
        results
    }

    async fn generate_response(&self, request: LlmManagerRequest, max_attempts: Option<usize>) -> LlmResult<String> {
        let start_time = Instant::now();
        let mut attempts = request.attempts;
        let mut failed_instances = request.failed_instances.clone();
        let prompt_preview = request.prompt.chars().take(50).collect::<String>(); // For logging
        let task = request.task.as_deref();
        let request_params = request.params.clone();
        let max_retries = max_attempts.unwrap_or(self.max_retries);

        info!("generate_response called for task: {:?}, prompt: '{}...'", task, prompt_preview); // *** NEW LOG ***

        while attempts <= max_retries {
            debug!("Attempt {} of {} for request (task: {:?})", attempts + 1, max_retries + 1, task); // Adjusted attempt number for clarity

            let attempt_result = self.instance_selection(&request.prompt, task, request_params.clone(), &failed_instances).await;

            match attempt_result {
                Ok((content, instance_id)) => {
                    let duration = start_time.elapsed();
                    info!("Request successful on attempt {} with instance {} after {:?}", attempts + 1, instance_id, duration); // *** NEW LOG ***
                    debug!("generate_response returning Ok for task: {:?}", task); // *** NEW LOG ***
                    return Ok(content);
                },
                Err((error, instance_id)) => {
                    warn!("Attempt {} failed with instance {}: {}", attempts + 1, instance_id, error);
                    failed_instances.push(instance_id);
                    attempts += 1;

                    if attempts > max_retries {
                         warn!("Max retries ({}) reached for task: {:?}. Returning last error.", max_retries + 1, task); // *** NEW LOG ***
                         debug!("generate_response returning Err for task: {:?}", task); // *** NEW LOG ***
                        return Err(error);
                    }

                    debug!("Retrying with next eligible instance for task: {:?}...", task);
                }
            }
        }

        // Should not reach here due to the return in the loop, but just in case
        warn!("Exited retry loop unexpectedly for task: {:?}", task); // *** NEW LOG ***
        Err(LlmError::ConfigError("No available providers after all retry attempts".to_string()))
    }

    async fn instance_selection(&self,
                              prompt: &str,
                              task: Option<&str>,
                              request_params: Option<HashMap<String, serde_json::Value>>,
                              failed_instances: &[usize]) -> Result<(String, usize), (LlmError, usize)> {

        debug!("instance_selection: Starting selection for task: {:?}", task); // *** NEW LOG ***

        // 1. Get candidate instance IDs based on task (if any)
        let candidate_ids: Option<Vec<usize>> = match task {
            Some(task_name) => {
                let task_map = self.tasks_to_instances.lock().unwrap();
                task_map.get(task_name).cloned()
            }
            None => None, // No specific task, consider all instances initially
        };

        if task.is_some() && candidate_ids.is_none() {
            warn!("No instances found supporting task: '{}'", task.unwrap());
            // Return a dummy instance ID of 0 since we don't have a specific instance to blame
            debug!("instance_selection returning Err (no task support)"); // *** NEW LOG ***
            return Err((LlmError::ConfigError(format!("No providers available for task: {}", task.unwrap())), 0));
        }

        // 2. Filter candidates by availability and collect references + metrics
        let eligible_instances_data: Vec<&LlmInstance>;
        let instances_guard = self.instances.lock().unwrap(); // Lock acquired here
        debug!("instance_selection: Acquired instances lock (1st time)"); // *** NEW LOG ***

        if instances_guard.is_empty() {
            warn!("No LLM providers configured.");
            drop(instances_guard); // Release lock before returning
             debug!("instance_selection returning Err (no providers configured)"); // *** NEW LOG ***
            return Err((LlmError::ConfigError("No LLM providers available".to_string()), 0));
        }

        match candidate_ids {
            Some(ids) => {
                debug!("Filtering instances for task '{}' using IDs: {:?}", task.unwrap(), ids);
                eligible_instances_data = instances_guard.iter()
                    .filter(|inst| ids.contains(&inst.id) && inst.is_enabled() && !failed_instances.contains(&inst.id))
                    .collect();
                debug!("Found {} eligible instances for task '{}'", eligible_instances_data.len(), task.unwrap());
            }
            None => {
                debug!("No specific task. Filtering all enabled instances.");
                eligible_instances_data = instances_guard.iter()
                    .filter(|inst| inst.is_enabled() && !failed_instances.contains(&inst.id))
                    .collect();
                debug!("Found {} eligible instances (no task)", eligible_instances_data.len());
            }
        }

        // 3. Check if any eligible instances remain
        if eligible_instances_data.is_empty() {
            let error_msg = format!("No enabled providers available{}{}",
                task.map_or_else(|| "".to_string(), |t| format!(" for task: '{}'", t)),
                if !failed_instances.is_empty() { format!(" (excluded {} failed instances)", failed_instances.len()) } else { "".to_string() }
            );
            warn!("{}", error_msg);
            drop(instances_guard); // Release lock before returning
            debug!("instance_selection returning Err (no eligible instances)"); // *** NEW LOG ***
            // Return a dummy instance ID since we don't have a specific instance to blame
            return Err((LlmError::ConfigError(error_msg), 0));
        }

        // 4. Get metrics for eligible instances
        let eligible_metrics: Vec<InstanceMetrics> = eligible_instances_data.iter()
            .map(|inst| inst.get_metrics())
            .collect();

        // 5. Select instance using strategy
        let selected_metric_index = {
            let mut strategy = self.strategy.lock().unwrap();
            debug!("instance_selection: Acquired strategy lock"); // *** NEW LOG ***
            let index = strategy.select_instance(&eligible_metrics);
            debug!("instance_selection: Released strategy lock"); // *** NEW LOG ***
            index
        };

        let selected_instance_id = eligible_metrics[selected_metric_index].id;
        // Find the LlmInstance reference from the eligible list using the selected ID
        // Need to re-find it using the ID because eligible_instances_data's indices might not match eligible_metrics indices if filtering happened
         let selected_instance_ref = eligible_instances_data.iter()
            .find(|inst| inst.id == selected_instance_id)
            .map(|inst_ref| *inst_ref) // Dereference to get &LlmInstance
            .expect("Selected instance ID from metrics not found in eligible list - LOGIC ERROR!"); // Added expect

        debug!("Selected instance {} ({}) for the request.", selected_instance_ref.id, selected_instance_ref.provider.get_name());

        // 6. Merge parameters
        let mut final_params = HashMap::new();
        if let Some(task_name) = task {
            if let Some(task_def) = selected_instance_ref.provider.get_supported_tasks().get(task_name) {
                final_params.extend(task_def.parameters.clone());
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

        // 7. Clone provider Arc and ID for use after releasing lock
        let selected_provider_arc = selected_instance_ref.provider.clone();
        let selected_id = selected_instance_ref.id;

        // Explicitly drop the guard to release the first instances lock *before* the await call
        debug!("instance_selection: Releasing instances lock (1st time) before API call"); // *** NEW LOG ***
        drop(instances_guard);


        // Create and execute the request
        let max_tokens = final_params.get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let temperature = final_params.get("temperature")
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

        debug!("Instance {} ({}) sending request to provider...", selected_id, selected_provider_arc.get_name());
        let start_time = Instant::now();
        let result = selected_provider_arc.generate(&request).await; // <<< Network call happens here
        let duration = start_time.elapsed();
        info!("Instance {} ({}) received result in {:?}", selected_id, selected_provider_arc.get_name(), duration); // Changed to info

        // Update metrics regardless of success or failure
        {
            debug!("instance_selection: Attempting to acquire instances lock (2nd time) for metrics update"); // *** NEW LOG ***
            let mut instances_guard = self.instances.lock().unwrap();
            debug!("instance_selection: Acquired instances lock (2nd time)"); // *** NEW LOG ***
            if let Some(instance_mut) = instances_guard.iter_mut().find(|inst| inst.id == selected_id) {
                debug!("Recording result for instance {}", selected_id);
                instance_mut.record_result(duration, &result);
                debug!("Finished recording result for instance {}", selected_id); // *** NEW LOG ***
            } else {
                warn!("Instance {} not found for metric update after request completion.", selected_id);
            }
            debug!("instance_selection: Releasing instances lock (2nd time) after metrics update"); // *** NEW LOG ***
            // Lock released when instances_guard goes out of scope here
        }

        // Return either content or error with the instance ID
        match result {
            Ok(response) => {
                 debug!("instance_selection returning Ok for instance {}", selected_id); // *** NEW LOG ***
                 Ok((response.content, selected_id))
            },
            Err(e) => {
                 debug!("instance_selection returning Err for instance {}: {}", selected_id, e); // *** NEW LOG ***
                 Err((e, selected_id))
            },
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