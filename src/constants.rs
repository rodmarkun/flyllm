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

// Ollama
pub const OLLAMA_API_ENDPOINT: &str = "http://localhost:11434/api/chat";

// LM Studio (OpenAI-compatible local server)
pub const LMSTUDIO_API_ENDPOINT: &str = "http://localhost:1234/v1/chat/completions";

// Groq
pub const GROQ_API_ENDPOINT: &str = "https://api.groq.com/openai/v1/chat/completions";

// Cohere (v2 API)
pub const COHERE_API_ENDPOINT: &str = "https://api.cohere.com/v2/chat";

// Together AI
pub const TOGETHERAI_API_ENDPOINT: &str = "https://api.together.xyz/v1/chat/completions";

// Perplexity
pub const PERPLEXITY_API_ENDPOINT: &str = "https://api.perplexity.ai/chat/completions";

// Rate limiting
pub const DEFAULT_RATE_LIMIT_WAIT_SECS: u64 = 2;
pub const MAX_RATE_LIMIT_WAIT_SECS: u64 = 60;