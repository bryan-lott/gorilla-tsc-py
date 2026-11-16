# Q3: can gorilla-tsc's ideas extend to text and binary data?

Code: `experiments/q3-text-binary/`. Data: `results/q3.json` (46 files, 20 methods).
Every size below comes from a real run. **Every method was decoded and compared byte for byte on every file
it was applied to; all 20 have `lossless_verified: true`.** Timings are single-threaded per job, 6 jobs at a
time on an i7-8700K. CLI tools are timed as process wall time.

## TL;DR

- **Prose and general text:** use context mixing, not numeric modelling. Our own bitwise CM coder (lpaq-style,
  written here) beats xz -9e, brotli -q11 and zstd -19 on large text by 10-20%, but it is about 250x slower
  to decode. Number modelling adds nothing to prose. The library's adaptive binary coder used naively
  (order-2 contexts, no mixing) is *worse than gzip*. The coder is fine; the modelling is what matters.
- **Numeric text (CSV) is the showcase.** Columnar parsing, exact decimal tokens and gorilla-tsc `Auto` on the
  scaled integers give **0.65x of the best standard tool** (per-file best, summed) over 227 MB of numeric CSV.
  On Stocks-Germany (143 MB): **4.96 MB vs 9.70 MB for xz -9e (-49%)**, and still -26% against our CM on the raw text.
- **Logs:** templates plus epoch timestamps through gorilla-tsc give **-29% vs xz** on the system journal,
  and the encoder runs about 2x faster than xz -9e. On a package-manager log (many unique package/version
  templates) templating loses to xz, and plain CM wins (-31% vs xz).
- **Binary with records:** stride detection plus typed fields through gorilla-tsc gives **-48% vs xz** on
  20-byte tick records, decoding at 177 MB/s. Stride-aware CM gives -60% on kennedy.xls and -38% on SQLite.
- **Audio:** the library's order ≤ 2 predictors lose to FLAC by 23%. Adding **LPC plus stereo
  decorrelation** in front of the library's own `RiceBlock`/`AdaptiveBlock` **beats `flac -8` by 4.3% and 13%**.

## Results (bytes; lower is better)

Category totals (sum over files; "best baseline" = best standard tool per file):

| category (files, raw MB) | xz -9e | brotli -q11 | best baseline | ours-cm | best ours (per file) | ratio |
|---|---:|---:|---:|---:|---:|---:|
| text (9, 11.2) | 3,112,108 | 3,082,920 | 3,053,092 | 2,593,636 | 2,593,636 | 0.850 |
| numeric-text (25, 227.1) | 16,973,976 | 18,173,908 | 15,800,894 | 12,186,086 | 10,216,124 | 0.647 |
| log (2, 23.7) | 187,524 | 196,922 | 187,524 | 162,737 | 131,879 | 0.703 |
| binary (8, 87.0) | 13,275,324 | 14,800,185 | 12,735,473 | 11,416,686 | 10,213,913 | 0.802 |
| audio (2, 11.8) | 10,712,132 | 10,171,084 | 7,054,641 (flac) | 9,552,558 | 6,729,282 | 0.954 |

Selected files:

| file | xz -9e | best other baseline | ours | which |
|---|---:|---:|---:|---|
| alice29.txt | 48,528 | 43,202 bzip2 | 39,384 | cm (2.07 bpb) |
| enwik8, first 10 MB | 2,720,256 | 2,704,839 brotli | 2,275,041 | cm (1.82 bpb) |
| xargs.1 (4 KB) | 1,812 | 1,464 brotli | 1,566 | cm, loses |
| sum (SPARC exe) | 9,500 | 10,144 brotli | 10,553 | cm, loses |
| Stocks-Germany-sample.txt | 9,704,892 | 10,467,002 brotli | 4,964,645 | columnar+cm (columnar+xz 5,097,447) |
| city_temperature.csv | 3,906,956 | 3,004,324 bzip2 | 2,623,148 | columnar+cm (raw cm 2,782,949) |
| elf/Stocks-DE.csv | 98,224 | 94,466 bzip2 | 60,445 | columnar+cm (columnar+xz 60,656) |
| elf/City-lat.csv (not a time series) | 122,264 | 106,655 bzip2 | 94,109 | raw cm; columnar adds nothing |
| system journal (125k lines) | 101,040 | 107,700 brotli | 71,812 | log+xz |
| package-manager log | 86,484 | 89,222 brotli | 60,067 | raw cm (log+xz 85,783) |
| kennedy.xls | 51,868 | 61,498 brotli | 20,679 | cm-stride (stride 13 found) |
| ptt5 (fax bitmap) | 39,860 | 40,939 brotli | 36,233 | cm-stride (stride 216 = scan line) |
| SQLite DB | 1,892,300 | 2,244,607 brotli | 1,167,090 | cm-stride |
| ELF x86-64 executable | 2,957,212 | 2,864,980 xz --x86 | 2,708,740 | BCJ + cm |
| 20-byte tick records | 1,954,704 | 2,568,980 brotli | 1,008,141 | struct: found `[i64,f64,4×bytes]` |
| city temps as f64 array | 3,207,920 | 2,961,078 bzip2 | 2,639,940 | cm-stride (struct/gorilla f64: 2,804,105) |
| music 60 s PCM | 10,050,496 | 6,789,871 flac | 6,499,636 | lpc (gorilla struct: 8,357,181) |

Speed (MB/s, enc / dec) on large files:

| method | enwik8 10 MB | Stocks 143 MB | journal 19.6 MB | PCM 10.6 MB | records 20 MB |
|---|---|---|---|---|---|
| xz -9e | 1.6 / 108 | 0.40 / 203 | 4.4 / 780 | 3.2 / 34 | 2.0 / 169 |
| zstd -19 | 2.1 / 497 | 1.6 / 865 | 9.4 / 1386 | 6.1 / 864 | 2.7 / 700 |
| brotli -q11 | 0.49 / 285 | 0.61 / 646 | 1.9 / 987 | 0.37 / 117 | 0.63 / 553 |
| ours-cm | 0.39 / 0.38 | 0.67 / 0.66 | 0.66 / 0.65 | 0.31 / 0.32 | 0.57 / 0.55 |
| ours, specialised | | columnar+xz 0.88 / 48; columnar+cm 0.33 / 1.2 | log+xz 9.1 / 117 | struct 2.2 / 55; lpc 0.43 / 28; flac 55 / 226 | struct 3.2 / 177 |

## What each idea showed

**1. Columnar/typed CSV (`columnar.rs`, `numtok.rs`).**
- How it works: each field becomes a skeleton (literal bytes plus a placeholder per number) and exact number
  tokens (mantissa, decimal places, leading-zero pad). Each column gets a skeleton dictionary and one number
  stream per slot. Streams are scaled to the stream's maximum decimal count and coded with gorilla-tsc `Auto`.
- Two tricks mattered. **Decimal-count prediction** from the value's trailing zeros (shortest form costs ~0 bits).
  **Digit split** at 10^k into a smooth high part and a low-digit stream: float-printing artefacts like
  `7.8305287` stop wrecking deltas (Basel-temp columnar+xz: 202 KB to 134 KB).
- Each column is then coded independently, as typed or raw text, whichever is smaller (an exact choice).
- Wins are large on smooth time-series columns: Stocks -36…-49%, Air-pressure -36%, Bitcoin -27% vs the best standard tool.
- It gains **nothing** on non-sequential numbers (City-lat/lon, POI, Food-price): CM on the raw digits wins, so the per-column choice falls back to text.
- Over all 22 Elf CSVs: xz 3.36 MB, raw CM 2.69 MB, columnar+xz 2.87 MB, columnar+cm 2.62 MB.
- Columnar+xz is the practical variant: 5.10 MB on Stocks (-47% vs xz) and 48 MB/s decode.

**2. Log templates (`logtpl.rs`).**
- ISO timestamps become epoch seconds in one global gorilla stream. Lines are split on spaces. Tokens that
  look like IDs (hex runs, `0x..`, ≥ 3 letter/digit transitions) become string variables; other numeric
  tokens become per-template slot streams.
- Journal: 689 templates; 391 of 420 number streams chose gorilla, 77,631 values in 25 KB. Total 71.8 KB vs
  xz 101 KB. log+xz encodes about 2x faster than xz -9e.
- Package log: 4,325 templates for 53k lines, because package names are static text in templates. The
  cross-line redundancy ("A -> B" then "B -> C") is LZ/CM territory, and templating is +0..-1% vs xz.
- Conclusion: templating pays when templates are few and variables are numeric. Otherwise, use a strong text coder.

**3. Context mixing (`cm.rs`, own coder, not the library's).**
- Model: orders 0-4 and 6, word, two sparse contexts, a match model, and two optional stride contexts. Each
  hashed context has two counters (fast and slow adaptation). Two mixers, two APMs, 12-bit arithmetic coder.
- Canterbury (11 files, mean): **1.93 bpb**, vs brotli 2.06, xz 2.21, zstd 2.23, bzip2 2.23.
- It loses on very small files (xargs.1, cp.html: no time to learn) and on `sum`.
- Speed: 0.3-0.7 MB/s both ways. Like all CM, decode is as slow as encode.
- The library's `RangeEncoder`/`Prob` with direct order-1/order-2 contexts: 3.7-4.0 bpb mean, worse than gzip (2.53).
  Direct high-order contexts dilute, and one fixed-rate `Prob` per context with no mixing cannot fix that.

**4. Structure detection (`structure.rs`).**
- Stride detection worked on everything that has records: 4 (stereo s16), 8 (f64), 4 (f32),
  20 (i64+f64+i32 records), 13 (kennedy.xls), 216 (ptt5 scan line), 42 (SQLite).
- On the 20-byte records, field-typing DP over real gorilla sizes recovered `i64, f64` and left the volume
  as bytes. Result 1.01 MB vs xz 1.95 MB; decode is 177 MB/s.
- gorilla f64 and f32 give the *same* size on the temperature array (2,804,105 B): Decimal mode rediscovers
  the 1-decimal values. Stride-aware CM (2.64 MB) still beats it on that noisy, non-smooth series.
- ELF: x86 BCJ plus CM gives -5.5% vs `xz --x86`.

**4b. Audio (`lpc.rs`).** The `struct` path (i16 through gorilla `Auto`, order ≤ 2) is +23% vs FLAC. FLAC-style
LPC (order ≤ 32, exact integer prediction) plus 4 stereo modes, feeding the library's `RiceBlock` (256-sample
partitions) or `AdaptiveBlock`, beats FLAC. Encoding is 0.43 MB/s (exhaustive search); decode is 28 MB/s.
Caveat: both clips are decoded lossy audio, which is smoother than raw recordings.

## What a text/binary extension of the library would look like (the `Auto` idea, one level up)

```
bytes ──► sniff ──┬─ delimited text ─► columnar tokenizer ─► per-column: numeric slots ─► gorilla Auto (+digit split)
                  │                                          skeleton dict / ids / text ─► byte back end
                  ├─ line logs ──────► template miner ─────► epoch ts ─► gorilla Dod/Delta; slots ─► gorilla Auto
                  │                                          template ids, string vars ─► byte back end
                  ├─ fixed records ──► stride + field DP ──► typed columns ─► gorilla Auto (i32/i64/f32/f64)
                  │                                          opaque byte fields ─► back end with stride contexts
                  ├─ PCM ────────────► stereo mode + LPC ──► RiceBlock / AdaptiveBlock residuals
                  └─ anything else ──► (BCJ if x86) ───────► byte back end
byte back end = CM (best ratio, ~0.5 MB/s) or xz/zstd (fast decode); chosen per stream by exact size
```

## Recommendations

1. **Worth adding to the library:** an `Lpc` integer mode (order chosen per block, quantised coefficients)
   next to `Delta`. It is the one place where the existing residual coders were starved by a weak predictor,
   and it beats FLAC here. It should also help smooth sensor series.
2. **Worth shipping as a companion crate, not in the core:** the exact decimal tokenizer plus
   columnar CSV (with digit-split and the per-column typed/text choice). Large, robust wins on numeric CSV,
   with xz as the back end for fast decode.
3. Record/stride detection plus typed fields is cheap and fast to decode. It is good for sensor dumps and
   struct arrays; fall back to a general coder when fewer than half the bytes are typed.
4. **Do not** build a general-purpose text compressor into a time-series library. CM wins on ratio, but at
   about 0.5 MB/s decode. zstd/xz are the right defaults for leftover strings.
5. Next: inter-column prediction, date/time-column fusion, value dictionaries for repeated values, faster CM.

## Caveats

- CM parameters were tuned briefly on four Canterbury files (also in the test set). The SQLite file was built from public bench data; copying a browser database was not approved.
- Only two logs were tested, and the journal had 125k lines, not 300k. Results contain sizes only.
- CLI timings include process start-up, so speeds on files under 1 MB mean little.
- `ours-auto` encode time is the sum of all our methods tried on that file.
