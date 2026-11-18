#!/usr/bin/env bash
# Download the benchmark datasets into bench-data/ (or $BENCH_DATA).
#
#   scripts/fetch-bench-data.sh        # everything (Chimp + Elf, ~300 MB)
#   scripts/fetch-bench-data.sh elf    # only the 22 Elf sets (~20 MB), enough for the ratio check
#
# Sources are pinned to commits and every file is checked against
# scripts/bench-data.sha256, so results cannot drift with upstream edits.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
which="${1:-all}"
dir="${BENCH_DATA:-bench-data}"
mkdir -p "$dir/elf"

chimp=https://raw.githubusercontent.com/panagiotisl/chimp/320d397157c7e0696b3c64dc1711fc17a3add3da/src/test/resources
elf=https://raw.githubusercontent.com/Spatio-Temporal-Lab/elf/64e0d6004be322d9d8eaf9931e371df202f1a0eb/src/test/resources/ElfTestData

if [ "$which" = all ]; then
    for f in city_temperature.csv SSD_HDD_benchmarks.csv Stocks-Germany-sample.txt; do
        [ -s "$dir/$f" ] || curl -fsSL --retry 3 "$chimp/$f.gz" | gunzip | tr -d "\r" > "$dir/$f"
    done
fi
# The 22 Elf datasets (about 100k values each), as used by the Elf and ALP papers.
for f in Air-pressure Air-sensor Basel-temp Basel-wind Bird-migration Bitcoin-price Blockchain-tr \
         City-lat City-lon City-temp Dew-point-temp Food-price IR-bio-temp PM10-dust POI-lat POI-lon \
         SSD-bench Stocks-DE Stocks-UK Stocks-USA Wind-Speed electric_vehicle_charging; do
    [ -s "$dir/elf/$f.csv" ] || curl -fsSL --retry 3 "$elf/$f.csv" | tr -d "\r" > "$dir/elf/$f.csv"
done

# Verify the files this run is responsible for.
sums="$root/scripts/bench-data.sha256"
if [ "$which" = elf ]; then filter='elf/'; else filter=''; fi
cd "$dir"
if command -v sha256sum >/dev/null; then check=(sha256sum -c --quiet); else check=(shasum -a 256 -c --quiet); fi
grep "  $filter" "$sums" | "${check[@]}"
echo "bench data OK in $dir ($which)"
