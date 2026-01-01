//! Tests for provider types and conversions.

use flyllm::ProviderType;

// ============================================================================
// ProviderType Conversion Tests
// ============================================================================

#[test]
fn test_provider_type_from_str_lowercase() {
    assert_eq!(ProviderType::from("anthropic"), ProviderType::Anthropic);
    assert_eq!(ProviderType::from("openai"), ProviderType::OpenAI);
    assert_eq!(ProviderType::from("mistral"), ProviderType::Mistral);
    assert_eq!(ProviderType::from("google"), ProviderType::Google);
    assert_eq!(ProviderType::from("ollama"), ProviderType::Ollama);
    assert_eq!(ProviderType::from("lmstudio"), ProviderType::LMStudio);
    assert_eq!(ProviderType::from("groq"), ProviderType::Groq);
    assert_eq!(ProviderType::from("cohere"), ProviderType::Cohere);
    assert_eq!(ProviderType::from("togetherai"), ProviderType::TogetherAI);
    assert_eq!(ProviderType::from("perplexity"), ProviderType::Perplexity);
}

#[test]
fn test_provider_type_from_str_mixed_case() {
    assert_eq!(ProviderType::from("Anthropic"), ProviderType::Anthropic);
    assert_eq!(ProviderType::from("OpenAI"), ProviderType::OpenAI);
    assert_eq!(ProviderType::from("MISTRAL"), ProviderType::Mistral);
    assert_eq!(ProviderType::from("Google"), ProviderType::Google);
    assert_eq!(ProviderType::from("OLLAMA"), ProviderType::Ollama);
    assert_eq!(ProviderType::from("LmStudio"), ProviderType::LMStudio);
    assert_eq!(ProviderType::from("GROQ"), ProviderType::Groq);
    assert_eq!(ProviderType::from("Cohere"), ProviderType::Cohere);
    assert_eq!(ProviderType::from("TogetherAI"), ProviderType::TogetherAI);
    assert_eq!(ProviderType::from("PERPLEXITY"), ProviderType::Perplexity);
}

#[test]
#[should_panic(expected = "Unknown provider")]
fn test_provider_type_from_str_unknown() {
    let _ = ProviderType::from("unknown_provider");
}

#[test]
fn test_provider_type_display() {
    assert_eq!(format!("{}", ProviderType::Anthropic), "Anthropic");
    assert_eq!(format!("{}", ProviderType::OpenAI), "OpenAI");
    assert_eq!(format!("{}", ProviderType::Mistral), "Mistral");
    assert_eq!(format!("{}", ProviderType::Google), "Google");
    assert_eq!(format!("{}", ProviderType::Ollama), "Ollama");
    assert_eq!(format!("{}", ProviderType::LMStudio), "LMStudio");
    assert_eq!(format!("{}", ProviderType::Groq), "Groq");
    assert_eq!(format!("{}", ProviderType::Cohere), "Cohere");
    assert_eq!(format!("{}", ProviderType::TogetherAI), "TogetherAI");
    assert_eq!(format!("{}", ProviderType::Perplexity), "Perplexity");
}

#[test]
fn test_provider_type_equality() {
    assert_eq!(ProviderType::OpenAI, ProviderType::OpenAI);
    assert_ne!(ProviderType::OpenAI, ProviderType::Anthropic);
}

#[test]
fn test_provider_type_clone() {
    let provider = ProviderType::Anthropic;
    let cloned = provider.clone();
    assert_eq!(provider, cloned);
}

#[test]
fn test_provider_type_copy() {
    let provider = ProviderType::OpenAI;
    let copied = provider; // Copy, not move
    assert_eq!(provider, copied);
}

#[test]
fn test_provider_type_debug() {
    let debug_str = format!("{:?}", ProviderType::Anthropic);
    assert_eq!(debug_str, "Anthropic");
}

// ============================================================================
// Provider Count Test
// ============================================================================

#[test]
fn test_all_providers_exist() {
    // Ensure we have all 10 providers
    let providers = vec![
        ProviderType::Anthropic,
        ProviderType::OpenAI,
        ProviderType::Mistral,
        ProviderType::Google,
        ProviderType::Ollama,
        ProviderType::LMStudio,
        ProviderType::Groq,
        ProviderType::Cohere,
        ProviderType::TogetherAI,
        ProviderType::Perplexity,
    ];
    assert_eq!(providers.len(), 10);
}
