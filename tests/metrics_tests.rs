//! Tests for the metrics module.
//!
//! These tests verify the label generation and other metrics utilities.
//! Note: Actual metric recording tests require a metrics recorder to be installed.

#[cfg(feature = "metrics")]
mod metrics_tests {
    use flyllm::metrics::labels::{error_type_label, keys, provider_label};
    use flyllm::{LlmError, ProviderType};

    #[test]
    fn test_provider_label_openai() {
        assert_eq!(provider_label(ProviderType::OpenAI), "openai");
    }

    #[test]
    fn test_provider_label_anthropic() {
        assert_eq!(provider_label(ProviderType::Anthropic), "anthropic");
    }

    #[test]
    fn test_provider_label_mistral() {
        assert_eq!(provider_label(ProviderType::Mistral), "mistral");
    }

    #[test]
    fn test_provider_label_google() {
        assert_eq!(provider_label(ProviderType::Google), "google");
    }

    #[test]
    fn test_provider_label_ollama() {
        assert_eq!(provider_label(ProviderType::Ollama), "ollama");
    }

    #[test]
    fn test_provider_label_lmstudio() {
        assert_eq!(provider_label(ProviderType::LMStudio), "lmstudio");
    }

    #[test]
    fn test_provider_label_groq() {
        assert_eq!(provider_label(ProviderType::Groq), "groq");
    }

    #[test]
    fn test_provider_label_cohere() {
        assert_eq!(provider_label(ProviderType::Cohere), "cohere");
    }

    #[test]
    fn test_provider_label_togetherai() {
        assert_eq!(provider_label(ProviderType::TogetherAI), "togetherai");
    }

    #[test]
    fn test_provider_label_perplexity() {
        assert_eq!(provider_label(ProviderType::Perplexity), "perplexity");
    }

    #[test]
    fn test_error_type_label_api_error() {
        let error = LlmError::ApiError("test".to_string());
        assert_eq!(error_type_label(&error), "api_error");
    }

    #[test]
    fn test_error_type_label_rate_limit() {
        let error = LlmError::RateLimit("test".to_string());
        assert_eq!(error_type_label(&error), "rate_limit");
    }

    #[test]
    fn test_error_type_label_parse_error() {
        let error = LlmError::ParseError("test".to_string());
        assert_eq!(error_type_label(&error), "parse_error");
    }

    #[test]
    fn test_error_type_label_provider_disabled() {
        let error = LlmError::ProviderDisabled("test".to_string());
        assert_eq!(error_type_label(&error), "provider_disabled");
    }

    #[test]
    fn test_error_type_label_config_error() {
        let error = LlmError::ConfigError("test".to_string());
        assert_eq!(error_type_label(&error), "config_error");
    }

    #[test]
    fn test_label_keys() {
        assert_eq!(keys::PROVIDER, "provider");
        assert_eq!(keys::MODEL, "model");
        assert_eq!(keys::TASK, "task");
        assert_eq!(keys::ERROR_TYPE, "error_type");
    }

    #[test]
    fn test_describe_metrics_does_not_panic() {
        // This should not panic even without a recorder installed
        flyllm::describe_metrics();
    }
}

#[cfg(feature = "metrics-server")]
mod dashboard_tests {
    use flyllm::metrics::dashboard::DashboardServerConfig;
    use std::net::SocketAddr;

    #[test]
    fn test_dashboard_config_default() {
        let config = DashboardServerConfig::default();
        let expected: SocketAddr = ([127, 0, 0, 1], 9898).into();
        assert_eq!(config.bind_address, expected);
    }

    #[test]
    fn test_dashboard_config_with_port() {
        let config = DashboardServerConfig::with_port(8080);
        let expected: SocketAddr = ([0, 0, 0, 0], 8080).into();
        assert_eq!(config.bind_address, expected);
    }

    #[test]
    fn test_dashboard_config_new() {
        let addr: SocketAddr = ([192, 168, 1, 1], 3000).into();
        let config = DashboardServerConfig::new(addr);
        assert_eq!(config.bind_address, addr);
    }
}
