use log::debug;

use crate::load_balancer::instances::InstanceMetrics;

pub trait LoadBalancingStrategy {
    fn select_instance(&mut self, metrics: &[InstanceMetrics]) -> usize;
}

pub struct LeastRecentlyUsedStrategy;

impl LeastRecentlyUsedStrategy {
    pub fn new() -> Self {
        Self {}
    }
}

impl LoadBalancingStrategy for LeastRecentlyUsedStrategy {
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