# Gorilla Time Series Compression

A Python implementation of a variant of the Gorilla time series compression algorithm, optimized for in-memory storage with configurable precision and modes.

## Features

- **Timestamps**: Delta-of-delta encoding with checkpoints every n items.
- **Values**: Configurable XOR (default, for irregular floats) or delta-of-delta (for trends) with checkpoints.
- **Precision**: Support for int32/int64 timestamps, f32/f64 values.
- **Checkpoints**: User-configurable interval (default 100) for error recovery.
- **Convenience**: Functions to estimate compression on sample data.

## Installation

Clone and install:

```bash
git clone <repo>
cd gorilla_tsc_py
pip install -e .
```

## Demo

Run the demo script to see the library in action:

```bash
python demo.py
```

It demonstrates round-trip accuracy, error handling, benchmarks, and test summaries.

## Usage

### Encoding/Decoding Timestamps

```python
from gorilla_tsc_py import TimestampsEncoder, TimestampsDecoder

timestamps = [1000, 1001, 1002, 1003]
encoder = TimestampsEncoder(precision='int64', checkpoint_interval=100)
result = encoder.encode_all(timestamps)
# result: {'encoded': bytes, 'nb_timestamps': 4, 'precision': 'int64', 'checkpoint_interval': 100}

decoder = TimestampsDecoder(result)
decoded = decoder.decode_all()
# decoded: [1000, 1001, 1002, 1003]
```

### Encoding/Decoding Values

```python
from gorilla_tsc_py import ValuesEncoder, ValuesDecoder

values = [1.0, 1.0001, 1.0002]
encoder = ValuesEncoder(precision='f64', mode='xor', checkpoint_interval=100)
result = encoder.encode_all(values)
# result: {'encoded': bytes, 'nb_values': 3, 'precision': 'f64', 'mode': 'xor', 'checkpoint_interval': 100}

decoder = ValuesDecoder(result)
decoded = decoder.decode_all()
# decoded: [1.0, 1.0001, 1.0002]
```

### Encoding/Decoding Pairs

```python
from gorilla_tsc_py import PairsEncoder, PairsDecoder

pairs = [(1000, 1.0), (1001, 1.1), (1002, 1.2)]
encoder = PairsEncoder(ts_precision='int64', val_precision='f64', val_mode='xor', checkpoint_interval=100)
result = encoder.encode_all(pairs)

decoder = PairsDecoder(result)
decoded = decoder.decode_all()
# decoded: [(1000, 1.0), (1001, 1.1), (1002, 1.2)]
```

### Estimating Compression

```python
from gorilla_tsc_py import estimate_ts_compression, estimate_val_compression

ts_data = [1000, 1001, 1002] * 100
stats = estimate_ts_compression(ts_data, precision='int64', checkpoint_interval=100)
# {'ratio': ~10.0, 'compressed_size': 50, 'bits_per_item': 1.33}

val_data = [1.0, 1.0001, 1.0002] * 100
stats = estimate_val_compression(val_data, mode='xor', precision='f64')
# {'ratio': ~5.0, 'compressed_size': 160, 'bits_per_item': 12.8}
```

## Choosing XOR vs. DoD for Values

- **XOR**: Best for floating-point data with irregular patterns (e.g., sensor noise, varying exponents). Exploits bit-level similarities for higher compression (5-10x). Default choice.
- **DoD**: Simpler arithmetic differencing. Best for predictable trends or integer-like floats. May compress worse (2-7x) for noisy data, but useful for simplicity or when XOR overhead is unwanted.

Use `estimate_val_compression` with different modes on your data to compare.

### Real-World Use Cases

#### Compressing IoT Sensor Data (Irregular/Noisy)
For data like temperature readings with noise:
```python
values = [20.5, 21.0, 20.8, 21.2, 20.9]  # Noisy floats
stats = estimate_val_compression(values, mode='xor', precision='f64')
# Likely high ratio due to bit similarities
encoder = ValuesEncoder(mode='xor')
result = encoder.encode_all(values)
```

#### Compressing Linear Trends (e.g., Counters)
For predictable increases:
```python
values = [100.0, 101.0, 102.0, 103.0]
stats = estimate_val_compression(values, mode='dod', precision='f64')
# DoD may outperform XOR here
```

### Performance Tips
- **Choose Checkpoint Interval (n)**: Smaller n (e.g., 10) improves random access but reduces compression. Larger n (e.g., 1000) maximizes ratios but slows seeks.
- **Precision Trade-Offs**: f32 halves storage but loses precision; use f64 for accuracy-critical data.
- **Mode Selection**: Test both modes on samples. XOR for varied data, DoD for trends.
- **Large Datasets**: Use incremental encoding if memory is an issue.

### Troubleshooting
- **Low Ratios?** Data may be too irregular—try XOR for floats. Check if values are identical (high ratios expected).
- **Errors on Encoding?** Ensure inputs are correct types (ints for timestamps, floats for values).
- **Precision Loss?** DoD on floats may accumulate errors; use XOR for exactness.

### Benchmark Results (on 1000 samples, checkpoint_interval=100)

- **Timestamps (linear, delta=1)**: ~37x compression, ~1.7 bits/item.
- **Values (identical)**: XOR ~39x, DoD ~33x.
- **Values (linear, small increments)**: XOR ~1.2x, DoD ~9x (DoD better for trends).
- **Values (irregular, varying)**: XOR ~1.1x, DoD ~3.5x (XOR better for noise).
- **Values (oscillating, e.g., -0.1, +0.1)**: XOR ~1.2x, DoD ~1.8x (DoD benefits from zig-zag on negatives).

## Algorithm Notes

Based on Gorilla (Pelkonen et al., 2015), with adaptations:
- Timestamps: Delta-of-delta with variable-bit encoding.
- Values: XOR with LZ/TZ compression or delta-of-delta on packed floats.
- Checkpoints: Full values every n items to reset deltas.

Lossless, efficient for time series.