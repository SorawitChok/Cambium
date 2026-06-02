# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Cambium
- Core expansion engine with surgical model modification
- Interleaved block expansion strategy (LLaMA-Pro style)
- Width expansion strategy
- Parallel adapter expansion (bottleneck and attention)
- Staged training with progressive unfreezing
- Smart initialization strategies (identity, noise, Xavier, etc.)
- Freezing manager for fine-grained parameter control
- Memory estimation and profiling utilities
- Catastrophic forgetting detection
- Integration with Hugging Face Transformers
- Integration with TRL (Transformer Reinforcement Learning)

## [0.1.0] - 2024-04-14

### Added
- First public release
- Basic expansion strategies
- Training utilities
- Documentation and examples

[Unreleased]: https://github.com/SorawitChok/Cambium/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SorawitChok/Cambium/releases/tag/v0.1.0
