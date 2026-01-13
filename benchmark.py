"""Benchmark compression ratios for Gorilla.

Usage:
    python benchmark.py  # Run default samples
    python benchmark.py --file path/to/data.json  # Load from JSON file
    python benchmark.py --data "[1.0, 2.0, 3.0]"  # Inline data as JSON list

Data Structure:
    - For values: JSON array of floats, e.g., [1.0, 2.0, 3.0]
    - For timestamps: JSON array of ints, e.g., [1000, 1001, 1002]
    - File: {"type": "values" or "timestamps", "data": [list]}

The script will test all modes and recommend the best based on highest ratio.
"""

import sys
import json
import argparse

sys.path.insert(0, "src")

from gorilla_tsc_py import (
    TimestampsEncoder,
    ValuesEncoder,
    estimate_ts_compression,
    estimate_val_compression,
)


def load_data(args):
    if args.file:
        with open(args.file, "r") as f:
            data_dict = json.load(f)
        data_type = data_dict.get("type", "values")
        data = data_dict["data"]
    elif args.data:
        data = json.loads(args.data)
        data_type = "values"  # Assume values unless specified
    else:
        return None, None  # Use default samples
    return data, data_type


def benchmark_user_data(data, data_type):
    print(f"Testing user data ({len(data)} items, type: {data_type}):")
    if data_type == "timestamps":
        stats = estimate_ts_compression(data)
        print(
            f"Timestamps: Ratio {stats['ratio']:.2f}x, bits/item {stats['bits_per_item']:.2f}"
        )
        best = "DoD (timestamps always use DoD)"
    else:
        modes = ["xor", "dod", "xor_dod", "dod_xor"]
        results = {}
        for mode in modes:
            stats = estimate_val_compression(data, mode=mode)
            results[mode] = stats["ratio"]
            print(
                f"Values ({mode}): Ratio {stats['ratio']:.2f}x, bits/item {stats['bits_per_item']:.2f}"
            )
        best_mode = max(results, key=lambda k: results[k])
        best = f"{best_mode} (ratio {results[best_mode]:.2f}x)"
    print(f"\nRecommended: {best}\n")


def benchmark_default():
    # Sample data
    samples = {
        "timestamps_linear": ([1000 + i for i in range(1000)], "timestamps"),
        "values_identical": ([1.0] * 1000, "values"),
        "values_linear": ([1.0 + i * 0.001 for i in range(1000)], "values"),
        "values_irregular": (
            [1.0 + (i % 10) * 0.1 + (i % 5) * 0.01 for i in range(1000)],
            "values",
        ),
        "values_oscillating": ([1.0 + (-1) ** i * 0.1 for i in range(1000)], "values"),
        "timestamps_large": ([10**10 + i for i in range(1000)], "timestamps"),
    }

    results = {}

    for name, (data, data_type) in samples.items():
        if data_type == "timestamps":
            stats = estimate_ts_compression(data)
            results[name] = stats
        else:
            stats_xor = estimate_val_compression(data, mode="xor")
            stats_dod = estimate_val_compression(data, mode="dod")
            stats_xor_dod = estimate_val_compression(data, mode="xor_dod")
            stats_dod_xor = estimate_val_compression(data, mode="dod_xor")
            stats = {
                "xor": stats_xor,
                "dod": stats_dod,
                "xor_dod": stats_xor_dod,
                "dod_xor": stats_dod_xor,
            }
            results[name] = stats
        print(f"{name}: {results[name]}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gorilla compression modes.")
    parser.add_argument(
        "--file",
        type=str,
        help='Path to JSON file with data: {"type": "values" or "timestamps", "data": [list]}',
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Inline JSON list of floats (for values) or ints (for timestamps)",
    )
    args = parser.parse_args()

    data, data_type = load_data(args)
    if data:
        benchmark_user_data(data, data_type)
    else:
        print("Running default benchmarks...\n")
        results = benchmark_default()
