/// Common constants used throughout the crate

// General
pub const DEFAULT_MAX_TOKENS: u32 = 1024;
pub const DEFAULT_MAX_TRIES: usize = 5;

// OpenAI
pub const OPENAI_API_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions"; 

// Anthropic
pub const ANTHROPIC_API_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";

// Mistral
pub const MISTRAL_API_ENDPOINT: &str = "https://api.mistral.ai/v1/chat/completions";

// Google
pub const GOOGLE_API_ENDPOINT_PREFIX: &str = "https://generativelanguage.googleapis.com";