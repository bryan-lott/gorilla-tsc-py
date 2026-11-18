# gorilla-tsc

Lossless time series compression for integer timestamps and float values: a
Rust library (`crates/gorilla-tsc`) with a thin Python package
(`gorilla_tsc_py`) on top. It started as a Python implementation of Facebook's
Gorilla codec; the default `auto` mode now compresses the standard Gorilla/Chimp
benchmark datasets 2–16x smaller than Gorilla, and the 22 Elf datasets 3.3x
smaller on average.

> [!IMPORTANT]
> **`auto` is a discovery tool, not a production setting.** It searches up to
> six codecs per block, which makes it slow to encode (about 1–8 MB/s per core).
> For real use:
>
> 1. **Test compress a representative sample** of your data (10 000 values
>    is plenty) with `learn_profile`, which runs that search and returns the
>    best fixed recipe for the data, e.g. `"decimal:2:first:rice"`.
> 2. **Store that string in your config** and pass it as `mode=` in production.
>    Encoding then skips the search: typically 90–260 MB/s, with output within
>    about 1% of `auto` on data like the sample.
> 3. **Re-learn when the data changes shape** (a new sensor, new precision).
>    Locked decimal profiles already re-search any block where more than 1% of
>    the values no longer fit, so drift costs speed, not correctness or much size.
>
> ```python
> profile = g.learn_profile(sample, precision="f64")  # once, offline
> data = g.encode(values, mode=profile)               # in production
> ```
>
> Decoding needs nothing: every stream records how it was encoded. If you
> must use `auto`, pass `threads=0` to use every core (see below).

## Results

Bits per value (lower is better) on the datasets used by the Gorilla and Chimp
papers, compressed in independent blocks of 1000 values as in the Chimp paper.
Reproduce with `scripts/fetch-bench-data.sh` then
`cargo run --release -p gorilla-tsc --example bench`.

| dataset | Gorilla (`xor`) | Chimp128 (paper) | `auto` |
| --- | ---: | ---: | ---: |
| city_temperature (2.9M) | 58.66 | 22.92 | **7.63** |
| Stocks-Germany (5M) | 46.83 | 13.20 | **2.88** |
| SSD_HDD (8.9k) | 40.18 | 17.00 | **5.69** |
| Stocks-Germany timestamps, s (5M) | 8.03 (`dod`) | – | **3.92** |
| scrape timestamps, 10 s ± 5 ms (1M) | 8.75 (`dod`) | – | **4.91** |

Our `xor` mode reproduces the published Gorilla figures (58.72 / 46.90 / 40.25),
which cross-checks the implementation.

The [Elf](https://github.com/Spatio-Temporal-Lab/elf) float datasets
(`cargo run --release -p gorilla-tsc --example bench -- --elf`), with a locked
profile learned from each dataset's first 10 000 values:

| dataset | Gorilla | `auto` | profile | profile enc MB/s | learned profile |
| --- | ---: | ---: | ---: | ---: | --- |
| Air-pressure | 45.96 | **9.02** | 9.13 | 4 | `decimal_split` |
| Air-sensor | 52.56 | **47.58** | 48.24 | 3 | `dictionary` |
| Basel-temp | 60.11 | **12.04** | 12.04 | 2 | `decimal_split` |
| Basel-wind | 63.53 | **20.23** | 20.23 | 4 | `dictionary` |
| Bird-migration | 50.26 | **12.71** | 13.02 | 4 | `decimal_split` |
| Bitcoin-price | 53.37 | **21.11** | 21.11 | 246 | `decimal:4:first:rice` |
| City-temp | 54.16 | **6.41** | 6.67 | 136 | `decimal:1:first:adaptive` |
| Dew-point-temp | 53.26 | **8.24** | 8.29 | 260 | `decimal:2:first:rice` |
| IR-bio-temp | 44.50 | **3.41** | 3.43 | 251 | `decimal:2:first:adaptive` |
| PM10-dust | 30.54 | **3.00** | 3.25 | 253 | `decimal:3:first:adaptive` |
| Stocks-DE | 45.84 | **4.71** | 4.90 | 112 | `decimal:3:first:rice` |
| Stocks-UK | 36.88 | **4.72** | 4.74 | 7 | `decimal_split` |
| Stocks-USA | 43.30 | **4.42** | 4.46 | 245 | `decimal:2:first:rice` |
| Wind-Speed | 52.93 | **5.11** | 5.13 | 234 | `decimal:2:first:rice` |
| Blockchain-tr | 47.12 | **11.54** | 11.66 | 88 | `decimal:4:first:adaptive` |
| City-lat | 65.95 | **20.27** | 20.36 | 243 | `decimal:4:first:rice` |
| City-lon | 66.00 | **21.47** | 21.53 | 244 | `decimal:4:zero:rice` |
| Food-price | 36.80 | **9.16** | 9.19 | 4 | `decimal_split` |
| POI-lat | 65.88 | **53.73** | 53.73 | 21 | `bit_split` |
| POI-lon | 66.01 | **55.23** | 55.23 | 20 | `bit_split` |
| SSD-bench | 40.18 | 5.69 | **5.68** | 23 | `dictionary` |
| EV charging | 63.71 | **9.87** | 9.87 | 5 | `decimal_split` |
| **mean** | 51.77 | **15.90** | 15.99 | | |

Speed (Rust, one core of an i7-8700K): `auto` encodes at 3–8 MB/s (it tries
up to six codecs per block, each searching its own settings) and decodes at
70–330 MB/s; the Gorilla modes encode at 390–860 MB/s. A locked profile skips
the search: Delta/Decimal recipes encode at 90–260 MB/s, and a profile only
falls back to a slower fixed codec when that is more than 1% smaller on the
sample. Blocks are independent, so encoding parallelises across them with
byte-identical output: `encode(..., threads=n)` and the encoder classes'
`threads=` (`0` = every core; default `1`), or `Encoder::with_threads` /
`encode_parallel` in Rust. Streaming encoders buffer 8 blocks per thread and
then encode the batch in parallel. `auto` on 100 000 Elf values:

| dataset | 1 thread | 2 | 4 | 6 | 12 (hyperthreads) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basel-temp | 1.2 MB/s | 2.3 | 4.3 | 4.6 | 6.9 |
| City-temp | 4.9 | 8.9 | 16.7 | 20.0 | 25.4 |
| POI-lat | 0.9 | 1.8 | 3.2 | 4.1 | 5.2 |
| Stocks-DE | 4.2 | 7.6 | 13.2 | 18.9 | 21.1 |

The default stays one thread so a library call never takes every core
of a shared server unasked.

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
- **Dictionary**: the block's distinct values in numeric order, then an index
  per item (fixed width, adaptive, adaptive over differences, or adaptive
  over move-to-front ranks). Wins on few distinct values (SSD-bench, wind
  speeds).
- **Decimal split**: decimal scaling, then each integer as a reading plus a
  nearest remainder, `x = q * 10^j + r`. Readings stored with a
  low-precision offset (Basel-temp's `1.9005287`, `1.6105287`, ...) become
  cheap `q` plus a near-constant `r`.
- **Bit split** (ALP-RD style): for high-precision doubles that never scale,
  the top bits are dictionary coded with an adaptive range coder and the
  low bits stored raw.
- **Raw**: items verbatim, which bounds the worst case.
- **Auto** tries a per-type list (floats: xor, decimal, decimal split,
  dictionary, bit split, raw; integers: dod, delta, dictionary, raw) per
  block and keeps the smallest (a 4-bit selector per block). The lists come
  from measuring which codecs ever win on the benchmark data
  (`experiments/`).

### Locked profiles

`auto` is for when you do not know your data. Once you do, learn a profile
from a sample and reuse it: every per-block choice (codec, decimal exponent,
difference order, residual coder) is fixed, so encoding skips the search.
The stream format is unchanged; a profile's stream is an ordinary stream of
its codec. A decimal profile re-searches any block where its exponent leaves
more than 1% of the values unscaled, so data unlike the sample degrades
gracefully.

The classic Gorilla modes are still available: `xor` (Gorilla values), `dod`
(Gorilla timestamps) and the stacked `xor_dod` / `dod_xor`. On the benchmark
value datasets the stacked modes never beat plain `xor`.

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

readings = [round(20 + (i % 500) / 100, 2) for i in range(100_000)]
profile = g.learn_profile(readings[:10_000])   # 'decimal:2:second:adaptive'
data = g.encode(readings, mode=profile)         # no per-block search
```

Precisions are `int32`, `int64`, `f32`, `f64`; modes are `xor`, `dod`,
`xor_dod`, `dod_xor`, `delta`, `decimal`, `raw`, `dictionary`,
`decimal_split`, `bit_split`, `auto`, or a profile string from
`learn_profile` (`delta:<order>:<coder>`, `decimal:<exponent>:<order>:<coder>`). Timestamps must be `int`
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

// Learn once, then encode without the per-block search.
let profile = gorilla_tsc::Profile::learn(&[20.5f64, 21.0, 20.8], cfg.checkpoint_interval);
let bytes = encode(&[20.6f64, 20.9], Config::new(Mode::Profile(profile)));
```

The codec is generic over sealed `Sample` types (`i32`, `i64`, `f32`, `f64`)
whose `Word` (`u32`/`u64`) fixes every bit width, so a field width cannot
disagree with the data it describes. Arithmetic on words wraps, so residuals
always fit a word. Decoding never panics on untrusted input: it returns a
`DecodeError` for truncation, trailing data, unknown modes, impossible field
values, or a stream of a different element type. Decoders reserve memory only
in proportion to the input they were given, whatever the header claims, and
`cargo fuzz` targets in `fuzz/` check both properties.

The output is exactly as long as the header's item count, and a valid stream
can legitimately be tiny for a huge count (one repeated value costs almost
nothing per item). When decoding untrusted data, check the count before
decoding: `read_header(bytes)?.0.count` in Rust, `inspect(data)["count"]` in
Python.

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
