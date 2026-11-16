#!/usr/bin/env python3
"""Summarise experiments/results/q2.json: averages and per-dataset tables.

Averages are plain means of bits/value over the 22 Elf datasets, split into
time series (14) and non-time-series (8); Stocks-Germany (5M) is reported
separately. Speeds are medians of per-dataset MB/s (MB = 1e6 raw bytes).
"""
import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path(__file__).parent / "../results/q2.json"
r = json.loads(path.read_text())
bits, enc, dec = r["bits_per_value"], r["enc_mbs"], r["dec_mbs"]
big = "Stocks-Germany (5M)"
ts = [d["name"] for d in r["datasets"] if d["kind"] == "time series" and d["name"] != big]
nts = [d["name"] for d in r["datasets"] if d["kind"] == "non-time-series"]
names = [m["name"] for m in r["methods"]]
order = ["zstd-3", "zstd-19", "shuffle-zstd-19", "xz-9", "pcodec", "shuffle-zstd-19-b1000", "pcodec-b1000",
         "gorilla-xor", "lib-decimal", "auto", "auto-b128", "auto-b4096", "auto-b16384", "auto-b65536", "auto-whole",
         "auto-reimpl", "dict", "auto+dict", "alp-rd", "split-adaptive", "auto+alp-rd", "auto+split", "decimal-lpc",
         "auto+lpc", "auto+prior", "auto+all", "ctx-fresh", "ctx-carry16", "ctx-carry-all", "ctx-single", "ctx-prior",
         "ctx-prior-s6", "ctx-fresh-s4"]
order = [m for m in order if m in names] + [m for m in names if m not in order]


def mean(m, sets):
    v = [bits[m][s] for s in sets if s in bits.get(m, {})]
    return statistics.mean(v) if len(v) == len(sets) else float("nan")


def med(d, m):
    v = list(d.get(m, {}).values())
    return statistics.median(v) if v else float("nan")


print(f"{'method':24} {'TS avg':>7} {'NTS avg':>7} {'all22':>7} {'StkGer5M':>8} {'enc MB/s':>9} {'dec MB/s':>9} verified")
ver = {m["name"]: m["lossless_verified"] for m in r["methods"]}
for m in order:
    print(f"{m:24} {mean(m, ts):7.2f} {mean(m, nts):7.2f} {mean(m, ts + nts):7.2f} {bits[m].get(big, float('nan')):8.2f} "
          f"{med(enc, m):9.1f} {med(dec, m):9.1f} {ver[m]}")

if "--per-set" in sys.argv:
    cols = [c for c in sys.argv[sys.argv.index("--per-set") + 1].split(",")]
    print()
    print(f"{'dataset':22}" + "".join(f"{c[:13]:>14}" for c in cols))
    for s in ts + nts + [big]:
        print(f"{s[:22]:22}" + "".join(f"{bits[c].get(s, float('nan')):14.2f}" for c in cols))
