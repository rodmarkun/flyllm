# Changelog

All notable changes to FlyLLM will be documented in this file.

## [0.2.2] - 2025-06-06
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