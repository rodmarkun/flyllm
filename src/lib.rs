pub mod providers;
pub mod errors;
pub mod load_balancer;

// Re-export the core types for easier usage
pub use providers::{
    ProviderType, 
    LlmRequest, 
    LlmResponse,
    LlmProvider,
    create_provider,
    AnthropicProvider,
    OpenAIProvider,
};

pub use errors::{LlmError, LlmResult};

pub fn create_chat_request(messages: Vec<providers::Message>, max_tokens: Option<u32>) -> LlmRequest {
    LlmRequest {
        messages,
        model: None, 
        max_tokens,
        temperature: None, 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use providers::Message;

    #[test]
    fn create_request_works() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            }
        ];
        
        let request = create_chat_request(messages, Some(100));
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, "user");
        assert_eq!(request.messages[0].content, "Hello!");
        assert_eq!(request.max_tokens, Some(100));
    }
}