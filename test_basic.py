"""Basic tests for gorilla compression."""

import sys

sys.path.insert(0, "src")

from gorilla_tsc_py import TimestampsEncoder, ValuesEncoder, estimate_val_compression


def test_timestamps():
    timestamps = [1000, 1001, 1002]
    encoder = TimestampsEncoder()
    result = encoder.encode_all(timestamps)
    print("TS encoded:", result)


def test_values():
    values = [1.0, 1.0, 1.0]
    encoder = ValuesEncoder()
    result = encoder.encode_all(values)
    print("Val encoded:", result)
    stats = estimate_val_compression(values)
    print("Stats:", stats)


if __name__ == "__main__":
    test_timestamps()
    test_values()
