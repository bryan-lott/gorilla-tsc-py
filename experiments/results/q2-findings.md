# Q2 findings: other compression strategies, measured

Data: 22 Elf datasets (~100k f64 each; 14 time series (TS), 8 non-time-series
(NTS)) plus Stocks-Germany (5M values). All numbers are measured by
`experiments/q2-strategies` (`cargo run --release -- --jobs 1`). Raw data is in
`q2.json`. Every method was decoded and compared bit-exactly (`f64::to_bits`)
on every dataset, and all of them passed. Sizes include headers. Averages are
plain means of bits/value over datasets. Speeds are medians over datasets, in
MB/s of raw input, on one core of an i7-8700K.

## Headline

| method | TS avg | NTS avg | Stocks-Ger 5M | enc MB/s | dec MB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| gorilla-xor (b1000) | 49.06 | 56.30 | 46.83 | 425 | 329 |
| zstd-3 / zstd-19 | 19.95 / 17.14 | 28.49 / 25.61 | 8.47 / 5.51 | 225 / 4.1 | 767 / 787 |
| byte-shuffle + zstd-19 | 33.82 | 41.77 | 17.60 | 7.5 | 599 |
| xz -9 | 13.96 | **22.59** | 4.49 | 3.4 | 136 |
| pcodec (whole set) | 14.09 | 25.04 | 3.69 | 182 | 1074 |
| pcodec, blocks of 1000 | 17.22 | 25.74 | 4.46 | 61 | 736 |
| **auto (b1000, current)** | 13.58 | 26.56 | 2.93 | 15.3 | 199 |
| **auto+all (b1000, ours)** | **11.59** | 23.12 | **2.91** | 1.0 | 136 |

With block independence kept at 1000 values, `auto+all` adds five new
per-block candidates (dict, repeat-cache, decimal-split, split-adaptive and
decimal-lpc) plus the prior-initialised decimal coder. It is 14.7 % smaller
than `auto` on TS and 13.0 % smaller on NTS. It beats `auto` by more than 1 %
on 17 of 22 sets and is never worse. It now beats pcodec and xz on TS
on average. xz is still better on NTS because of exact repeats *across*
blocks (see 5). The price is encoding: trying every candidate on every block
makes it about 15x slower. Decoding is only about 30 % slower.

## Per strategy (gain = change vs `auto`, mean bits/value TS / NTS)

1. **Block-size sweep (B1)**, library `auto`:
   | block | 128 | 1000 | 4096 | 16384 | 65536 | whole |
   | --- | --- | --- | --- | --- | --- | --- |
   | TS | 13.96 | **13.58** | 13.91 | 14.34 | 14.39 | 14.47 |
   | NTS | 27.72 | 26.56 | **26.18** | 26.20 | 26.33 | 26.36 |
   | Stocks-Ger 5M | 3.98 | 2.93 | 2.78 | **2.74** | 2.79 | 3.87 |

   Bigger blocks are *not* monotonically better. One block for the whole set
   loses per-block adaptivity (mode, exponent, Rice k). For TS, 1000 is the
   sweet spot. Going from 1000 to 128 costs +2.8 % on TS (+36 % on the 5M
   stock series). Block independence itself costs very little *for auto's own
   model*. What it does cost is cross-block repeats (item 5).

2. **Dictionary (B2)**: `auto+dict` gives 12.36 / 24.07 (-9.0 % / -9.4 %).
   Encoding drops to about 4 MB/s and decoding to 138 MB/s. Big wins:
   Basel-wind -25 %, Basel-temp -19 %, SSD-bench -46 % (10.41 to 5.64),
   POI-lat -10 %, Bird-migration -9 %. A surprise: it also helps on blocks
   where *most values are distinct*. A sorted dictionary has tiny deltas, and
   the indices cost about log2(d) bits, which is cheaper than delta coding
   noisy data. So the dict candidate should not be limited to low-cardinality
   blocks. Restricting it to d <= n/2 (first run) gave only -2.1 % / -2.4 %.

3. **ALP-RD (B3)**: the paper-style split (left <= 16 bits, dictionary <= 8)
   in `auto+alp-rd` gives 13.46 / 24.49 at 10.7 / 203 MB/s. Our
   `split-adaptive` (left <= 32 bits, range-coded dictionary <= 255) does
   better: 13.36 / 24.16 at 6.0 / 190 MB/s. Both only win on POI-lat (62.14
   to 53.73, -13.5 %), POI-lon (-16 %) and Air-sensor (-6 %). These are the
   "real double" sets where auto was falling back to about 62-66 b/v, which is
   worse than zstd-19. Keep split-adaptive; ALP-RD classic is dominated by it.

4. **Linear prediction (B4)**: `auto+lpc` gives 13.54 / 26.49 (-0.3 % /
   -0.3 %). It wins only on Wind-speed (-3.2 %), City-lat (-2.0 %) and
   Vehicle-charge (-1.1 %), and costs a 4x slower encoder. Delta orders 0/1/2
   already capture almost everything. **Not worth it.**

5. **Repeat cache (B6, ours)**: a move-to-front cache of the values already
   seen. Symbol 0 means a new value (new values are coded in order with
   Decimal/Delta); otherwise the symbol is the MTF rank + 1.
   * Within 1000-blocks (`auto+repeat`): 12.78 / 24.59. Best on SSD-bench
     (-49 %), Basel-wind (-25 %), Bird-migration (-13 %) and Blockchain (-9 %).
     It overlaps with dict, and auto+all uses both.
   * Over larger windows (idea; breaks 1000-value random access):
     b16384 gives 12.33 / 22.97, b65536 gives 12.13 / 21.97, and the whole
     set gives 12.14 / 21.35. Whole-set repeats give Basel-wind 11.21
     (xz 13.29), Basel-temp 12.23, POI-lat 43.90 (xz 42.95) and Blockchain
     9.66 (xz 11.36). **This is the main thing xz/pcodec exploit that we
     don't.** In POI-lat, 33 % of values repeat an earlier value but only 1 %
     do so within their block. These numbers use one big block for every
     candidate, which hurts the others. A two-level design would probably get
     most of both: a shared dictionary per chunk (for example 64k values)
     plus 1000-value blocks referencing it. That design was **not measured**.

6. **Decimal + float-noise split (B6, ours: `decimal-split`)**: after decimal
   scaling, write `x = q*10^j + r`, where `r` is the nearest remainder, and
   code q and r each with DeltaRice. j is chosen per block.
   * Why: Basel-temp values such as `1.9005287`, `1.6105287` and `0.8205287`
     are 2-decimal readings plus a constant offset computed in float32. Their
     deltas mod 1e5 sit within ±5 of 0.
   * `auto+decsplit` gives 12.31 / 26.36 (-9.4 % / -0.8 %). Basel-temp drops
     25.65 to 11.90 (-54 %; pcodec 15.27, xz 13.01). Also Food-price -15 %,
     Bird-migration -10 %, Air-pressure -8 % and PM10 -5 %.
   * Decode speed is 166 MB/s. Encoding is 3.3 MB/s even with a cheap proxy
     that picks 2 of the 17 j values; full search was 6x slower and 1 % better
     on Basel-temp.

7. **Cross-block context for the adaptive coder (B5)**. These are decimal-only
   streams compared with our re-implementation of library Decimal
   (`ctx-fresh`, which matches `lib-decimal` within 0.005 b/v):
   | variant | TS | NTS | 5M | random access |
   | --- | --- | --- | --- | --- |
   | fresh model per block (library) | 13.93 | 27.35 | 2.93 | yes |
   | model carried over 16 blocks | 13.74 (-1.4 %) | 26.60 (-2.7 %) | 2.83 | per 16k |
   | carried over whole set | 13.72 | 26.52 | 2.82 | no |
   | + one coder, no per-block flush | 13.70 | 26.50 | 2.80 | no |
   | fresh, faster adaptation (shift 4) | 13.82 (-0.8 %) | 26.87 (-1.8 %) | 2.91 | yes |
   | leave-one-out trained prior, shift 5 | 13.81 (-0.9 %) | 26.71 (-2.3 %) | 2.94 | yes |
   | prior, shift 6 | 13.88 | 26.97 | 3.00 | yes |

   Resetting the model costs only about 1.5-3 %. The range-coder flush
   (about 4 bytes per block) costs about 0.02 b/v. As an `auto` candidate,
   the prior gives `auto+prior` 13.54 / 26.18 (-0.3 % / -1.4 %), with decoding
   down to about 124 MB/s. A one-constant change (shift 5 to 4) gets most of
   the prior's gain with no table. Low priority.

## Recommendations for the library

Worth adding as per-block modes (the 3-bit selector has 2 free codes; a
4-bit selector costs about 0.001 b/v):
1. **decimal-split** (q/r split): largest single win on float32-derived data
   and decodes cheaply. It needs a fast j heuristic, e.g. the proxy used here
   or a sample-based choice from q1.
2. **Dictionary / repeat cache**: these two overlap, and both pay off. Dict
   (sorted dictionary + indices) is simpler. The repeat cache is better on
   SSD-bench and Bird-migration. Start with dict and allow any cardinality.
3. **split-adaptive** as the high-entropy fallback instead of Xor for "real
   doubles" (POI-*, Air-sensor): -13 to -16 % there, and it decodes fast
   (about 190 MB/s).
4. Consider a **chunk-level shared dictionary** (format change) to capture
   cross-block repeats. It would be the biggest remaining gain on NTS
   (Basel-*, POI-*, Blockchain), but it is not yet measured with 1000-value
   blocks.

Not worth it: LPC predictors (-0.3 %); ALP-RD classic (dominated by
split-adaptive); bigger blocks for auto alone; carrying model state across
blocks (-1.5 to -3 % for losing random access); trained priors (at most
-1.4 % on NTS, plus a shipped table and format coupling).

Encoder cost: trying every candidate on every block is what makes `auto+all`
run at 1 MB/s. In production, use q1-style sample-then-lock or per-candidate
cheap cost estimates.

## Caveats

* The datasets are about 100k-value subsets, so they are only roughly
  comparable with the ALP paper's full-dataset numbers. For example, ALP
  reports POI-lat 55.5 and City-Temp 10.7; we measure auto+split 53.73 and
  auto 6.70.
* The prior is trained leave-one-out. Training excludes the evaluated set,
  and for any Stocks set it excludes all Stocks sets. Per-context counts come
  from fresh-model block choices.
* Byte shuffle *hurts* zstd on these sets (33.8 vs 17.1 TS). This was
  measured and verified, not a typo.
* The per-block baselines (pcodec-b1000, shuffle-zstd-19-b1000) include a
  2-byte length per block. pcodec 0.4.9 was run at default level 8. zstd is
  1.5.7 (zstd-sys). xz is xz2 0.1.7, preset 9.
* Composite streams use a 3-bit selector (4 bits for auto+all, 11
  candidates). `auto-reimpl` equals library `auto` byte-for-byte in size on
  all 23 sets.
* Speeds are noisy (±30 %): two other experiment jobs shared the machine.
  The experiment codecs are not hardened against corrupt input.
