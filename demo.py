"""Tour of gorilla_tsc_py: compress a sensor-like series with each mode.

    uv run python demo.py
"""

import math

import gorilla_tsc_py as g

# One reading per 10 s with millisecond jitter, temperatures with one decimal.
timestamps = [1_700_000_000_000 + 10_000 * i + (i * 7) % 5 for i in range(10_000)]
temps = [round(20 + 5 * math.sin(i / 300) + ((i * 37) % 7 - 3) / 10, 1) for i in range(10_000)]

print(f"{len(temps)} values; raw size {len(temps) * 8} bytes per column\n")
print(f"{'mode':10} {'temps bytes':>12} {'bits/value':>11} {'timestamps bytes':>17} {'bits/value':>11}")
for mode in ["xor", "dod", "xor_dod", "dod_xor", "delta", "decimal", "auto"]:
    v = g.encode(temps, mode=mode)
    t = g.encode(timestamps, precision="int64", mode=mode)
    assert g.decode(v) == temps and g.decode(t) == timestamps
    print(f"{mode:10} {len(v):12} {len(v) * 8 / len(temps):11.2f} {len(t):17} {len(t) * 8 / len(timestamps):11.2f}")

# Streams are self-describing.
encoded = g.encode(temps)
print("\nheader:", g.inspect(encoded))

# The class API keeps the original dict-shaped results.
pairs = list(zip(timestamps, temps))
result = g.PairsEncoder().encode_all(pairs)
assert g.PairsDecoder(result).decode_all() == pairs
print(f"pairs: {len(result['encoded_ts']) + len(result['encoded_val'])} bytes for {len(pairs)} pairs")

# Estimates look at the first 1000 items.
print("estimate:", g.estimate_val_compression(temps))
