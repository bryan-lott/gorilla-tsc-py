# q3: extending gorilla-tsc's ideas to text and binary data

Standalone cargo package (own `[workspace]`), depends on `../../crates/gorilla-tsc`.
Findings: `../results/q3-findings.md`. Machine-readable results: `../results/q3.json`.

## What is here

| file | idea |
|---|---|
| `src/cm.rs` | (3) **own** bitwise context-mixing coder: orders 0-4 and 6, word, sparse and optional record-stride contexts, match model; two count-adaptive counters per hashed context; 2 logistic mixers; 2 APM/SSE stages; 32-bit binary arithmetic coder with 12-bit probabilities. It does not use the library's coder. |
| `src/libcoder.rs` | order-1/order-2 byte contexts coded with the **library's** `RangeEncoder`/`Prob` (11-bit, shift 5, no mixing), for reference |
| `src/numtok.rs` | exact number tokens in text: (mantissa, decimals, leading-zero pad); a stream is scaled to its largest decimal count and coded with gorilla-tsc `Auto` (interval 1000), or as varints through the back end if that is smaller. Decimal counts are predicted from the value's trailing zeros, so only the residual is stored |
| `src/columnar.rs` | (1) CSV/TSV columnar transform: per-column skeleton dictionaries and numeric slot streams |
| `src/logtpl.rs` | (2) CLP/Drain-like log transform: ISO timestamps go to epoch seconds (a single gorilla stream), plus line templates, numeric slot streams per template, and id-like string variables |
| `src/structure.rs` | (4) stride detection (per-column order-0 entropy of raw vs lag-s delta bytes, s ≤ 256), then field typing by DP over u8/i16/i32/f32/i64/f64/bytes using real gorilla-tsc sizes. Also an x86 BCJ transform |
| `src/lpc.rs` | audio: stereo mode + LPC (order <= 32, exact integer) in front of the library's `RiceBlock` / `AdaptiveBlock` |
| `src/main.rs` | method registry, benchmark harness (verifies every round trip), report writer |
| `make_data.py` | derives the binary test files from the public bench data |
| `analyze.py`, `summarize.py` | print summary tables from `../results/q3.json` / `data/results.jsonl` |

## Reproduce

```sh
cd experiments/q3-text-binary
mkdir -p data && cd data
curl -LO https://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz && mkdir -p cantrbry && tar xzf cantrbry.tar.gz -C cantrbry
curl -LO http://mattmahoney.net/dc/enwik8.zip && unzip -o enwik8.zip && head -c 10000000 enwik8 > enwik8_10M && rm enwik8 enwik8.zip
journalctl -o short-iso --no-pager -n 300000 > journal.log      # whatever the journal holds (125k lines here)
cp /var/log/pacman.log pacman.log                                 # any world-readable log
cp /usr/bin/clang-tidy elf_clang-tidy                             # any x86-64 ELF, ~15 MB
ffmpeg -v error -ss 600 -t 60 -i SOME_MUSIC.mp3 -f s16le -acodec pcm_s16le -ac 2 -ar 44100 music_60s.s16le
ffmpeg -v error -i /usr/share/sounds/ocean/stereo/alarm-clock-elapsed.oga -f s16le -acodec pcm_s16le -ac 2 -ar 48000 alarm.s16le
python3 ../make_data.py      # city_temp.f64/.f32, stocks_records.bin (20-byte records), bench.sqlite
cd ..
# needs ../../bench-data (scripts/fetch-bench-data.sh) and gzip bzip2 zstd xz brotli flac on PATH
cargo build --release
target/release/q3-text-binary bench all all 6     # all jobs, 6 at a time; appends data/results.jsonl, log on stderr
target/release/q3-text-binary report              # -> ../results/q3.json
python3 analyze.py                                # tables used in q3-findings.md
```

Other commands: `q3-text-binary bench "<m1>,<m2>" <file-substring> <threads>`,
`enc|dec <method> <in> <out>`, `stride <file>`.

Notes:
- `bench` skips (method, file) pairs already in `data/results.jsonl`. Delete rows (or the file) to re-run. The full run is about 40 min of wall time with 6 jobs and up to about 10 GB RAM (CM tables are up to 1.3 GB per job).
- Every method's output is decoded and compared byte for byte with the input. `lossless_verified` in q3.json is the AND over all files.
- Timing is single-threaded per job, with 6 jobs at a time on an i7-8700K. CLI baselines are timed as process wall time, so small files are dominated by start-up; compare speeds on files of 1 MB or more.
- The SQLite file is built from the public bench data. Copying a browser profile database was not approved, so it was skipped.
- The audio is lossy-decoded MP3/Ogg rendered to 16-bit PCM. That is fine as a lossless-coding test, but it is smoother than raw recordings.
- `data/` and `target/` are git-ignored. No data or log contents are committed; results contain only sizes and timings.
