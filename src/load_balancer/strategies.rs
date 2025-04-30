use log::debug;
use rand::Rng;

use crate::load_balancer::instances::InstanceMetrics;

/// Trait defining the interface for load balancing strategies
///
/// Implementations of this trait determine how to select which LLM instance
/// will handle a particular request based on instance metrics.
pub trait LoadBalancingStrategy {
    /// Select an instance from available candidates
    ///
    /// # Parameters
    /// * `metrics` - Array of metrics for available instances
    ///
    /// # Returns
    /// * Index into the metrics array of the selected instance
    fn select_instance(&mut self, metrics: &[InstanceMetrics]) -> usize;
}

/// Strategy that selects the instance that was used least recently
/// 
/// This strategy helps distribute load by prioritizing instances
/// that haven't been used in the longest time.
pub struct LeastRecentlyUsedStrategy;

impl LeastRecentlyUsedStrategy {
    /// Creates a new LeastRecentlyUsedStrategy
    pub fn new() -> Self {
        Self {}
    }
}

impl LoadBalancingStrategy for LeastRecentlyUsedStrategy {
    /// Select the instance with the oldest last_used timestamp
    ///
    /// # Parameters
    /// * `metrics` - Array of metrics for available instances
    ///
    /// # Returns
    /// * Index into the metrics array of the least recently used instance
    ///
    /// # Panics
    /// Panics if `metrics` is empty
    fn select_instance(&mut self, metrics: &[InstanceMetrics]) -> usize {
        if metrics.is_empty() {
            panic!("LoadBalancingStrategy::select_instance called with empty metrics slice");
        }
        
        let mut oldest_index = 0;
        let mut oldest_time = metrics[0].last_used;
        
        for (i, metric) in metrics.iter().enumerate().skip(1) {
            if metric.last_used < oldest_time {
                oldest_index = i;
                oldest_time = metric.last_used;
            }
        }
        
        debug!(
            "LeastRecentlyUsedStrategy: Selected index {} (ID: {}) from {} eligible metrics with last_used: {:?}",
            oldest_index, metrics[oldest_index].id, metrics.len(), oldest_time
        );
        
        oldest_index
    }
}


/// Strategy that selects the instance with the lowest average response time.
#[derive(Debug, Default)]
pub struct LowestLatencyStrategy;

impl LowestLatencyStrategy {
     /// Creates a new LowestLatencyStrategy
     pub fn new() -> Self {
         Self {} 
     }
}

impl LoadBalancingStrategy for LowestLatencyStrategy {
    /// Select the instance with the minimum `avg_response_time`.
    ///
    /// # Parameters
    /// * `metrics` - Array of metrics for available instances.
    ///
    /// # Returns
    /// * Index into the metrics array of the fastest instance.
    ///
    /// # Panics
    /// * Panics if `metrics` is empty.
    fn select_instance(&mut self, metrics: &[InstanceMetrics]) -> usize {
        if metrics.is_empty() {
            panic!("LowestLatencyStrategy::select_instance called with empty metrics slice");
        }

        let mut best_index = 0;
        let mut lowest_time = metrics[0].avg_response_time;

        for (i, metric) in metrics.iter().enumerate().skip(1) {
            if metric.avg_response_time < lowest_time {
                best_index = i;
                lowest_time = metric.avg_response_time;
            }
        }

        debug!(
            "LowestLatencyStrategy: Selected index {} (ID: {}) from {} eligible metrics with avg_response_time: {:?}",
            best_index, metrics[best_index].id, metrics.len(), lowest_time
        );

        best_index
    }
}

/// Strategy that selects a random instance from the available pool.
#[derive(Debug, Default)]
pub struct RandomStrategy;

impl RandomStrategy {
     /// Creates a new RandomStrategy
     pub fn new() -> Self {
         Self {}
     }
}

impl LoadBalancingStrategy for RandomStrategy {
    /// Select a random instance.
    ///
    /// # Parameters
    /// * `metrics` - Array of metrics for available instances.
    ///
    /// # Returns
    /// * Index into the metrics array of a randomly chosen instance.
    ///
    /// # Panics
    /// * Panics if `metrics` is empty.
    fn select_instance(&mut self, metrics: &[InstanceMetrics]) -> usize {
        if metrics.is_empty() {
            panic!("RandomStrategy::select_instance called with empty metrics slice");
        }

        let index = rand::rng().random_range(0..metrics.len());

        debug!(
            "RandomStrategy: Selected random index {} (ID: {}) from {} eligible metrics",
            index, metrics[index].id, metrics.len()
        );

        index
    }
}