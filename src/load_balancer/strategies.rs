use crate::load_balancer::instances::LlmInstance;

pub trait LoadBalancingStrategy {
    fn select_instance<'a>(&mut self, providers: &[&'a LlmInstance]) -> usize;
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
    fn select_instance<'a>(&mut self, providers: &[&'a LlmInstance]) -> usize {
        if providers.is_empty() {
            return 0;
        }
        
        self.last_index = (self.last_index + 1) % providers.len();
        self.last_index
    }
}
