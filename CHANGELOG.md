# Changelog

All notable changes to gorilla-tsc-py will be documented here.

## [Unreleased]

### Added
- Zig-zag encoding for DoD values to improve compression on datasets with negative deltas.
- Expanded test coverage with edge cases and Hypothesis-based property tests.
- Enhanced documentation with advanced examples and performance tips.
- Input validation and error handling in encoders/decoders.
- BitStream optimizations for better performance.

### Changed
- Improved error messages for invalid inputs.

## [0.1.0] - Initial Release
- Basic Gorilla compression implementation with XOR and DoD modes.
- Support for configurable precision and checkpoints.
- Basic tests and benchmarks.