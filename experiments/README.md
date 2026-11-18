# Experiments

Research code behind `results/summary.html`. Not part of the library: each
directory is its own cargo workspace so experiment dependencies (zstd,
pcodec, ...) never reach `crates/`.

- `common/` - dataset loader shared by every experiment
- `q1-mode-selection/` - does one codec do well everywhere? sample-then-lock
- `q2-strategies/` - other compression strategies and baselines
- `results/` - raw outputs (JSON/text) and the summary page

Datasets: `scripts/fetch-bench-data.sh` (Chimp + Elf benchmark sets).
