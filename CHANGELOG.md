# Changelog

All notable changes to gorilla-tsc-py will be documented here.

## [Unreleased]

### Fixed
- Encoded format changed: streams written by earlier versions do not decode (most could not round-trip anyway).
- `zigzag_decode` returned the wrong value for every negative input.
- XOR mode: encoder and decoder disagreed on the reuse window (bad initial window, decoder shifted by leading
  instead of trailing zeros, window not reset at checkpoints in the decoder), and leading-zero/length fields
  overflowed their bit widths (incl. f32 using 64-bit counts).
- Value DoD modes stored the first delta and escaped dods in 32 bits, truncating f64 deltas; the escape prefix
  was written as `1111` but read as `11`.
- `xor_dod` XORed signed deltas, breaking on any decreasing step.
- Timestamp DoD escape stored 32 bits (16 for int32), truncating large jumps; bucket ranges were off by one
  (64/256/2048 wrapped); negative timestamps decoded as unsigned.
- `PairsEncoder` reordered pairs that share a timestamp by value; it now stable-sorts by timestamp.

### Added
- Regression, format (bit-cost) and validation tests; mutmut config (`uv run mutmut run --max-children 3`).

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