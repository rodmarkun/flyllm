use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    pub name: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

pub type ProviderTaskMap = HashMap<usize, Vec<String>>;