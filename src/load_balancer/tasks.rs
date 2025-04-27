use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Definition of a task that can be routed to specific providers
///
/// Tasks represent specialized capabilities or configurations that
/// certain providers might be better suited for. Each task can have
/// associated parameters that affect how the request is processed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    pub name: String,
    pub parameters: HashMap<String, serde_json::Value>,
}