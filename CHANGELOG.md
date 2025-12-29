# Changelog

All notable changes to FlyLLM will be documented in this file.

## [0.4.0] - 2025-12-29
### Added
- **TOML Configuration**: Load LlmManager from TOML files instead of using the builder pattern
  - `LlmManager::from_config_file("flyllm.toml")` for file-based configuration
  - `LlmManager::from_config_str(toml_content)` for string-based configuration
  - Environment variable support with `${VAR_NAME}` syntax for secure API key handling
  - Support for multiple API keys per provider (e.g., two OpenAI keys for different accounts)
  - User-friendly error messages for missing env vars, invalid providers, and undefined tasks
  - Example configuration file: `examples/flyllm.example.toml`
- **Streaming Support**: Added streaming responses for all providers
  - OpenAI, Anthropic, Groq, LM Studio, Together AI, Perplexity (SSE-based)
  - Mistral, Google/Gemini, Ollama, Cohere (provider-specific implementations)
  - New `send_request_streaming()` method returning `impl Stream<Item = Result<String>>`
- **New Providers**: Added support for additional LLM providers
  - Groq (fast inference)
  - Together AI (open-source models)
  - Cohere (Command models)
  - Perplexity (search-augmented)
  - LM Studio (local models)

### Changed
- Provider instances now support streaming via the `LlmProvider` trait
- Updated `print_available_models()` to handle optional API keys gracefully

## [0.3.1] - 2025-08-25
### Added
- Upon request, the conversion from `&str` to ProviderType has been implemented 

## [0.3.0] - 2025-08-06
### Added
- Refactored the internals of FlyLLM, making it way simpler to modify and understand
- Added optional debugging to LlmManager, allowing the user to store all requests and their metadata to JSON files automatically

## [0.2.3] - 2025-06-06
### Added
- Rate limiting with wait for whenever all providers are overloaded

## [0.2.2] - 2025-05-19
### Added
- Made the library entirely asynchronous, making the library more suitable for use in async contexts

## [0.2.1] - 2025-05-12
### Added
- Capability of listing all available models from all providers

## [0.2.0] - 2025-04-30
### Added
- Ollama provider support
- Builder pattern for easier configuration
- Aggregation of more basic routing strategies
- Added optional custom endpoint configuration for any provider

## [0.1.0] - 2025-04-27
### Added
- Initial release
- Multiple Provider Support (OpenAI, Anthropic, Google, Mistral)
- Task-Based Routing
- Load Balancing
- Failure Handling
- Parallel Processing
- Custom Parameters
- Usage Tracking