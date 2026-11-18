# Changelog

All notable changes to gorilla-tsc-py will be documented here.

## [0.2.0] - Unreleased

The codec is now a Rust crate (`gorilla-tsc`); the Python package is a thin
PyO3 wrapper over it. **The encoded format changed** and is now
self-describing (format version 1); streams from 0.1.0 do not decode.

### Added
- `decimal` mode: per-block power-of-ten scaling of floats to integers, with
  exceptions stored verbatim.
- `delta` mode: first/second/zero-order differences chosen per block, divided
  by their common divisor, coded with exact per-block Rice parameters or an
  adaptive binary range coder, whichever is smaller.
- `auto` mode (the new default): the smallest applicable mode per block, from
  a per-type candidate list (floats: `xor`, `decimal`, `decimal_split`,
  `dictionary`, `bit_split`, `raw`; integers: `dod`, `delta`, `dictionary`,
  `raw`) with a 4-bit selector per block.
- `raw` mode: items verbatim. In `auto` it bounds the worst case at 64 bits
  per value plus the selector.
- `dictionary` mode: a block's distinct values in numeric order plus one index
  per item (fixed width, adaptive, adaptive over differences, or adaptive over
  move-to-front ranks).
- `decimal_split` mode: decimal scaling, then each integer split into a
  reading and a nearest remainder (`x = q * 10^j + r`), for decimal readings
  that carry a low-precision offset.
- `bit_split` mode (ALP-RD style): high bits dictionary coded with an adaptive
  range coder, low bits raw, for high-precision doubles.
- Locked profiles: `learn_profile(sample)` returns a mode string such as
  `"decimal:2:first:rice"` that fixes every per-block choice, for encoding
  roughly as small as `auto` at a multiple of its speed. Any `mode=` accepts
  one. In Rust: `Profile::learn` and `Mode::Profile`.
- `encode()` / `decode()` / `inspect()` functions. `threads=n` on `encode()`
  and on every encoder class (`0` = all cores) encodes blocks in parallel with
  identical output; streaming encoders batch 8 blocks per thread. In Rust:
  `Threads`, `Encoder::with_threads`, `encode_parallel`, `encode_pairs_with`.
- `finish()` on the streaming encoders; `ts_mode` on `PairsEncoder`.
- `DecodeError` for malformed, truncated or mismatched streams.
- Type stubs (`_native.pyi`) and `py.typed`.
- Benchmarks on the Gorilla/Chimp and Elf datasets (`scripts/fetch-bench-data.sh`,
  `cargo run --release -p gorilla-tsc --example bench [-- --elf]`).
- `experiments/`: the studies behind the mode selection and the new codecs
  (separate cargo workspaces; nothing in them is a library dependency).

### Changed
- Defaults are `mode="auto"` everywhere (previously `dod` for timestamps and
  `xor` for values).
- Error messages are produced by the Rust core and list the valid choices.
- `estimate_*` report payload size only (the header is excluded) and
  `bits_per_item` from exact bits rather than whole bytes.
- Checkpoint blocks are fully independent: no state crosses a checkpoint
  (`dod_xor` previously carried state across).

### Removed
- The `debug` logging flag.
- The pure-Python codec from the package (kept in `reference/` for benchmarking).

### Fixed (in the pure-Python codec, before the port)
- `zigzag_decode` returned the wrong value for every negative input.
- XOR mode: encoder and decoder disagreed on the reuse window, and the
  leading-zero/length fields overflowed their widths (including f32 using
  64-bit counts).
- Value DoD modes truncated f64 deltas to 32 bits; the escape prefix was
  written as `1111` but read as `11`.
- `xor_dod` XORed signed deltas, breaking on any decreasing step.
- Timestamp DoD escape truncated large jumps; bucket ranges were off by one
  (64/256/2048 wrapped); negative timestamps decoded as unsigned.
- `PairsEncoder` reordered pairs that share a timestamp; it now stable-sorts
  by timestamp.

## [0.1.0] - Initial Release
- Basic Gorilla compression implementation with XOR and DoD modes.
- Support for configurable precision and checkpoints.
- Basic tests and benchmarks.
