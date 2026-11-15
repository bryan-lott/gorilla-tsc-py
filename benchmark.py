"""Compare the Rust-backed package with the original pure-Python codec.

    uv run python benchmark.py            (needs scripts/fetch-bench-data.sh)
"""

import sys
import time
from pathlib import Path

import gorilla_tsc_py as g

sys.path.insert(0, str(Path(__file__).parent / "reference"))
import gorilla_reference as ref  # noqa: E402

DATA = Path(__file__).parent / "bench-data"


def column(name: str, col: int, limit: int) -> list[float]:
    out = []
    with open(DATA / name) as f:
        for line in f:
            try:
                out.append(float(line.split(",")[col]))
            except (ValueError, IndexError):
                continue
            if len(out) >= limit:
                break
    return out


def best_of(fn, runs=3):
    best = float("inf")
    for _ in range(runs):
        t = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t)
    return best, result


def main():
    n = 100_000
    for name in ["city_temperature.csv", "Stocks-Germany-sample.txt", "SSD_HDD_benchmarks.csv"]:
        vals = column(name, 2, n)
        print(f"\n{name} ({len(vals)} values)")
        print(f"  {'codec':34} {'bits/value':>10} {'encode':>10} {'decode':>10}")
        rows = []
        for mode in ["xor", "auto"]:
            enc_t, data = best_of(lambda: g.encode(vals, mode=mode, checkpoint_interval=1000))
            dec_t, out = best_of(lambda: g.decode(data))
            assert out == vals
            rows.append((f"rust  mode={mode}", len(data) * 8 / len(vals), enc_t, dec_t))
        enc_t, data = best_of(lambda: g.encode(vals, checkpoint_interval=1000, threads=4))
        rows.append(("rust  mode=auto threads=4", len(data) * 8 / len(vals), enc_t, best_of(lambda: g.decode(data))[0]))
        small = vals[:20_000]  # the pure-Python codec is slow
        enc_t, res = best_of(lambda: ref.ValuesEncoder(mode="xor", checkpoint_interval=1000).encode_all(small), runs=1)
        dec_t, out = best_of(lambda: ref.ValuesDecoder(res).decode_all(), runs=1)
        assert out == small
        scale = len(vals) / len(small)
        rows.append(("python reference xor (scaled)", len(res["encoded"]) * 8 / len(small), enc_t * scale, dec_t * scale))
        for label, bpv, e, d in rows:
            print(f"  {label:34} {bpv:10.2f} {e * 1e3:8.1f}ms {d * 1e3:8.1f}ms")


if __name__ == "__main__":
    main()
