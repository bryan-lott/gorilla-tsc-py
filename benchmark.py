"""Benchmark compression ratios for Gorilla."""

import sys

sys.path.insert(0, "src")

from gorilla_tsc_py import (
    TimestampsEncoder,
    ValuesEncoder,
    estimate_ts_compression,
    estimate_val_compression,
)


def benchmark():
    # Sample data
    samples = {
        "timestamps_linear": [1000 + i for i in range(1000)],
        "values_identical": [1.0] * 1000,
        "values_linear": [1.0 + i * 0.001 for i in range(1000)],
        "values_irregular": [
            1.0 + (i % 10) * 0.1 + (i % 5) * 0.01 for i in range(1000)
        ],
        "values_oscillating": [
            1.0 + (-1) ** i * 0.1 for i in range(1000)
        ],  # For zig-zag benefits
        "timestamps_large": [10**10 + i for i in range(1000)],
    }

    results = {}

    for name, data in samples.items():
        if "timestamps" in name:
            stats = estimate_ts_compression(data)
        else:
            stats_xor = estimate_val_compression(data, mode="xor")
            stats_dod = estimate_val_compression(data, mode="dod")
            stats = {"xor": stats_xor, "dod": stats_dod}
        results[name] = stats
        print(f"{name}: {stats}")

    return results


if __name__ == "__main__":
    results = benchmark()
