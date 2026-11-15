# gorilla-tsc

Lossless time series compression for integer timestamps and float values: a
Rust library (`crates/gorilla-tsc`) with a thin Python package
(`gorilla_tsc_py`) on top. It started as a Python implementation of Facebook's
Gorilla codec; the default `auto` mode now compresses the standard Gorilla/Chimp
benchmark datasets 3.8–16x smaller than Gorilla.

## Results

Bits per value (lower is better) on the datasets used by the Gorilla and Chimp
papers, compressed in independent blocks of 1000 values as in the Chimp paper.
Reproduce with `scripts/fetch-bench-data.sh` then
`cargo run --release -p gorilla-tsc --example bench`.

| dataset | Gorilla (`xor`) | Chimp128 (paper) | `auto` |
| --- | ---: | ---: | ---: |
| city_temperature (2.9M) | 58.66 | 22.92 | **7.72** |
| Stocks-Germany (5M) | 46.83 | 13.20 | **2.93** |
| SSD_HDD (8.9k) | 40.18 | 17.00 | **10.68** |
| Stocks-Germany timestamps, s (5M) | 8.03 (`dod`) | – | **3.95** |
| scrape timestamps, 10 s ± 5 ms (1M) | 8.75 (`dod`) | – | **4.92** |

Our `xor` mode reproduces the published Gorilla figures (58.72 / 46.90 / 40.25),
which cross-checks the implementation.

Speed (Rust, one core of an i7-8700K, measured while other work shared the
machine, so treat as a lower bound): `auto` encodes at
16–24 MB/s and decodes at 130–330 MB/s; the Gorilla modes encode at 320–520 MB/s.
Encoding is a per-block search for the smallest representation, so it costs
more than decoding; `encode_parallel` / `encode(..., threads=n)` splits blocks
across threads and produces identical bytes.

## How it gets there

Each block of `checkpoint_interval` items is compressed independently, with
parameters chosen for that block:

- **Decimal** (floats): most real-world floats were decimals. Each block is
  scaled by the power of ten (`-18..=18`) that turns its values into integers
  at the least total cost; values that do not round-trip bit for bit (NaN,
  `-0.0`, high-precision values) are stored verbatim as exceptions. The
  integers then go through **Delta**.
- **Delta** (integers): first differences, second differences or the values
  themselves (whichever is cheapest), divided by their common divisor (price
  ticks), then coded with either
  - **Rice** coding with the transform (zig-zag, or offset from the block
    minimum) and parameter `k` chosen exactly per block, or
  - an **adaptive binary range coder** (the LZMA design) over a binarisation
    of each residual with contexts from the previous residual's magnitude,
  whichever is smaller.
- **Auto** tries every applicable mode per block and keeps the smallest
  (a 3-bit selector per block).

The classic Gorilla modes are still available: `xor` (Gorilla values), `dod`
(Gorilla timestamps) and the stacked `xor_dod` / `dod_xor`. On the benchmark
On the benchmark value datasets the stacked modes never beat plain `xor`.

## Python

```bash
uv sync          # builds the Rust extension with maturin
uv run pytest
```

```python
import gorilla_tsc_py as g

data = g.encode([20.5, 21.0, 20.8, 21.2])                # f64, mode="auto"
g.decode(data)                                            # [20.5, 21.0, 20.8, 21.2]
g.inspect(data)  # {'version': 1, 'precision': 'f64', 'mode': 'auto', 'checkpoint_interval': 100, 'count': 4}

ts = g.encode([1_700_000_000_000 + 10_000 * i for i in range(10_000)], precision="int64", threads=4)
```

Precisions are `int32`, `int64`, `f32`, `f64`; modes are `xor`, `dod`,
`xor_dod`, `dod_xor`, `delta`, `decimal` and `auto`. Timestamps must be `int`
and values `float`: an `int` passed as a value is a `TypeError` rather than a
silent conversion. Malformed or truncated input raises
`gorilla_tsc_py.DecodeError` (a `ValueError`).

The original class API remains and returns the same dict shapes:

```python
from gorilla_tsc_py import TimestampsEncoder, TimestampsDecoder, ValuesEncoder, ValuesDecoder
from gorilla_tsc_py import PairsEncoder, PairsDecoder, estimate_ts_compression, estimate_val_compression

result = ValuesEncoder(precision="f64", mode="auto", checkpoint_interval=100).encode_all([1.0, 1.5])
ValuesDecoder(result).decode_all()      # also accepts result["encoded"]

pairs = PairsEncoder().encode_all([(1002, 1.5), (1000, 1.0)])   # stable-sorted by timestamp
PairsDecoder(pairs).decode_all()        # [(1000, 1.0), (1002, 1.5)]

estimate_val_compression([1.0] * 500)   # {'ratio': ..., 'compressed_size': ..., 'bits_per_item': ...}
```

Encoders also stream: `encode_next(x)` then `finish()`. `demo.py` compares the
modes on a synthetic sensor series and `benchmark.py` compares against the
original pure-Python codec (kept in `reference/`).

## Rust

```rust
use gorilla_tsc::{decode, encode, CheckpointInterval, Config, Mode};

let cfg = Config::new(Mode::Auto).with_interval(CheckpointInterval::new(1000).unwrap());
let bytes = encode(&[20.5f64, 21.0, 20.8], cfg);
assert_eq!(decode::<f64>(&bytes).unwrap(), [20.5, 21.0, 20.8]);
```

The codec is generic over sealed `Sample` types (`i32`, `i64`, `f32`, `f64`)
whose `Word` (`u32`/`u64`) fixes every bit width, so a field width cannot
disagree with the data it describes. Arithmetic on words wraps, so residuals
always fit a word. Decoding never panics on untrusted input: it returns a
`DecodeError` for truncation, trailing data, unknown modes, impossible field
values, or a stream of a different element type.

## Format

`'G'`, version `1`, a descriptor byte (`sample_type << 4 | mode`, top two bits
reserved), varint checkpoint interval, varint item count, then the bit-packed
blocks. The format is self-describing; decoders need only the bytes.

## Development

```bash
cargo test --workspace         # Rust unit + property tests, and the Python suite against a fresh build
cargo clippy --all-targets     # pedantic lints, casts denied
cargo mutants -p gorilla-tsc   # mutation testing; equivalent mutants are excluded in .cargo/mutants.toml
```

`crates/gorilla-tsc-py/tests/python.rs` builds the extension and runs pytest
against it, so `cargo mutants -p gorilla-tsc-py` measures the Python suite
against mutations of the bindings.
