#!/usr/bin/env bash
# Download the benchmark datasets used by the Gorilla/Chimp papers into bench-data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench-data
base=https://raw.githubusercontent.com/panagiotisl/chimp/main/src/test/resources
for f in city_temperature.csv SSD_HDD_benchmarks.csv Stocks-Germany-sample.txt; do
    [ -s "bench-data/$f" ] || curl -fsSL "$base/$f.gz" | gunzip | tr -d "\r" > "bench-data/$f"
done
# The 22 Elf datasets (about 100k values each), as used by the Elf and ALP papers.
mkdir -p bench-data/elf
elf=https://raw.githubusercontent.com/Spatio-Temporal-Lab/elf/dev/src/test/resources/ElfTestData
for f in Air-pressure Air-sensor Basel-temp Basel-wind Bird-migration Bitcoin-price Blockchain-tr \
         City-lat City-lon City-temp Dew-point-temp Food-price IR-bio-temp PM10-dust POI-lat POI-lon \
         SSD-bench Stocks-DE Stocks-UK Stocks-USA Wind-Speed electric_vehicle_charging; do
    [ -s "bench-data/elf/$f.csv" ] || curl -fsSL "$elf/$f.csv" | tr -d "\r" > "bench-data/elf/$f.csv"
done
ls -la bench-data bench-data/elf
