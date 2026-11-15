#!/usr/bin/env bash
# Download the benchmark datasets used by the Gorilla/Chimp papers into bench-data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench-data
base=https://raw.githubusercontent.com/panagiotisl/chimp/main/src/test/resources
for f in city_temperature.csv SSD_HDD_benchmarks.csv Stocks-Germany-sample.txt; do
    [ -s "bench-data/$f" ] || curl -fsSL "$base/$f.gz" | gunzip | tr -d "\r" > "bench-data/$f"
done
ls -la bench-data
