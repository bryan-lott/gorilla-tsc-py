# q2-strategies: other compression strategies, measured

Research question 2: *what other compression strategies could we test with this
style of data?* This package measures external baselines and new per-block
strategies built from the public pieces of `gorilla-tsc`, on the 22 Elf
datasets plus Stocks-Germany (5M) from `bench-common`.

Findings: `../results/q2-findings.md`. Raw numbers: `../results/q2.json`.

## Reproduce

```sh
cd experiments/q2-strategies
cargo run --release -- --jobs 1          # everything, ~7 min on an i7-8700K
python3 summarize.py                      # averages table
python3 summarize.py --per-set auto,auto+all,pcodec,xz-9   # per-dataset table
```

Options: `--sets a,b` (substring match on dataset names), `--methods a,b`
(exact names), `--jobs N` (datasets in parallel; speeds are only meaningful
with 1), `--min-time S` (timing budget per measurement, default 0.3 s),
`--out path.json`. Re-running a subset merges into the existing JSON.
Per-block candidate win counts of composite methods go to
`out/block-wins.json`; the run log is `out/run.log`.

Data must already be fetched (`scripts/fetch-bench-data.sh`).

## Method of measurement

* bits/value = total compressed bytes * 8 / n, headers included (our composite
  streams write the same header bytes the library writes).
* Every method is decoded and compared with the input via `f64::to_bits`;
  `lossless_verified` in the JSON is true only if every dataset matched.
* Speeds: MB = 1e6 bytes of raw f64 input; best of up to 5 runs within the
  time budget (one run for the 5M set); single thread. Other jobs were running
  on the machine during the run, so treat speeds as +-30 %.

## Code

| file | what |
| --- | --- |
| `src/codec.rs` | `Composite`: per-block candidate selection with a fixed-width selector, exactly like `Mode::Auto`; `auto-reimpl` (the 5 library candidates) reproduces the library's size bit for bit. Copies of the library's private decimal-scaling helpers. |
| `src/baselines.rs` | zstd, byte-shuffle, xz, pcodec, and per-block wrappers. |
| `src/dict.rs` | B2 `Dict` (sorted distinct values + coded indices) and B6 `RepeatCache` (move-to-front cache of seen values, O(log n) via a Fenwick tree). |
| `src/alprd.rs` | B3 `AlpRd` (paper-style) and `SplitAdaptive` (larger range-coded dictionary). |
| `src/lpc.rs` | B4 `DecLpc`: decimal scaling + fixed order 0-3 or least-squares LPC predictors. |
| `src/ctx.rs` | B5: re-implementation of `AdaptiveBlock` with an exposed model to carry state across blocks, use one coder section, or start from a leave-one-out prior; `DecimalPrior` candidate. |
| `src/agcd.rs` | B6 `DecimalSplit`: decimal ints split as `q * 10^j + r` ("decimal reading + float32 noise"). |
| `summarize.py` | averages / per-dataset tables from the JSON. |
