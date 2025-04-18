use crate::load_balancer::instances::InstanceMetrics;

pub trait LoadBalancingStrategy {
    fn select_instance(&mut self, providers: &[InstanceMetrics]) -> usize;
}

// Round Robin
pub struct RoundRobinStrategy {
    last_index: usize,
}

impl RoundRobinStrategy {
    pub fn new() -> Self {
        Self { last_index: 0 }
    }
}

impl LoadBalancingStrategy for RoundRobinStrategy {
    fn select_instance(&mut self, metrics: &[InstanceMetrics]) -> usize {
        if metrics.is_empty() {
            return 0;
        }
        
        self.last_index = (self.last_index + 1) % metrics.len();
        metrics[self.last_index].id
    }
}
