//! Common streaming utilities for LLM providers
//!
//! This module provides utilities for parsing Server-Sent Events (SSE) streams
//! from various LLM providers.

use crate::providers::types::{StreamChunk, TokenUsage};

/// Parse a single SSE line and extract the data field
pub fn parse_sse_line(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.starts_with("data: ") {
        Some(&line[6..])
    } else {
        None
    }
}

/// Parse SSE data from a buffer, returning complete events and remaining buffer
pub fn parse_sse_buffer(buffer: &str) -> (Vec<String>, String) {
    let mut events = Vec::new();
    let mut remaining = String::new();

    for line in buffer.split('\n') {
        if let Some(data) = parse_sse_line(line) {
            if !data.is_empty() && data != "[DONE]" {
                events.push(data.to_string());
            }
        }
    }

    // Keep any incomplete line in the buffer
    if !buffer.ends_with('\n') {
        if let Some(last_newline) = buffer.rfind('\n') {
            remaining = buffer[last_newline + 1..].to_string();
        } else {
            remaining = buffer.to_string();
        }
    }

    (events, remaining)
}

/// OpenAI streaming response chunk structure
#[derive(serde::Deserialize, Debug)]
pub struct OpenAIStreamChunk {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<u64>,
    pub model: Option<String>,
    pub choices: Vec<OpenAIStreamChoice>,
    #[serde(default)]
    pub usage: Option<OpenAIStreamUsage>,
}

#[derive(serde::Deserialize, Debug)]
pub struct OpenAIStreamChoice {
    pub index: u32,
    pub delta: OpenAIStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct OpenAIStreamDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct OpenAIStreamUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl OpenAIStreamChunk {
    /// Convert to a StreamChunk
    pub fn to_stream_chunk(&self) -> Option<StreamChunk> {
        if self.choices.is_empty() {
            return None;
        }

        let choice = &self.choices[0];
        let content = choice.delta.content.clone().unwrap_or_default();
        let is_final = choice.finish_reason.is_some();

        let usage = self.usage.as_ref().map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Some(StreamChunk {
            content,
            model: self.model.clone(),
            is_final,
            usage,
        })
    }
}

/// Anthropic streaming event structure
#[derive(serde::Deserialize, Debug)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: u32, content_block: AnthropicContentBlock },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: AnthropicDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: AnthropicMessageDelta, usage: Option<AnthropicUsage> },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: AnthropicError },
}

#[derive(serde::Deserialize, Debug)]
pub struct AnthropicMessage {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(serde::Deserialize, Debug)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct AnthropicDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct AnthropicMessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

#[derive(serde::Deserialize, Debug)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl AnthropicStreamEvent {
    /// Convert to a StreamChunk if applicable
    pub fn to_stream_chunk(&self) -> Option<StreamChunk> {
        match self {
            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                if let Some(text) = &delta.text {
                    Some(StreamChunk::content(text.clone()))
                } else {
                    None
                }
            }
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let is_final = delta.stop_reason.is_some();
                if is_final {
                    let token_usage = usage.as_ref().map(|u| TokenUsage {
                        prompt_tokens: u.input_tokens.unwrap_or(0),
                        completion_tokens: u.output_tokens.unwrap_or(0),
                        total_tokens: u.input_tokens.unwrap_or(0) + u.output_tokens.unwrap_or(0),
                    });
                    Some(StreamChunk {
                        content: String::new(),
                        model: None,
                        is_final: true,
                        usage: token_usage,
                    })
                } else {
                    None
                }
            }
            AnthropicStreamEvent::MessageStart { message } => {
                // Return a chunk with model info but no content
                Some(StreamChunk {
                    content: String::new(),
                    model: Some(message.model.clone()),
                    is_final: false,
                    usage: None,
                })
            }
            _ => None,
        }
    }
}
