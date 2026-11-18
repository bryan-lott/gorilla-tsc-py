//! Bit split (ALP-RD style) for high-precision floats: each word is split
//! into its top `BITS - r` bits ("left") and its low `r` bits ("right").
//!
//! Real doubles such as coordinates in radians have nearly random low bits but
//! only a few distinct sign/exponent/high-mantissa patterns. The left parts are
//! dictionary coded with an adaptive binary range coder (a bit tree; for small
//! dictionaries the previous symbol is the context), the right parts are
//! stored raw.
//!
//! Block layout: `r` (6 bits), dictionary size `d` (8 bits), the `d` left
//! values (`BITS - r` bits each), a range-coded section with one symbol per
//! item (`0..d` = dictionary entry, `d` = escape followed by the left value as
//! direct bits), then every right part (`r` bits each).

use core::marker::PhantomData;

use crate::{BitReader, BitWriter, BlockCodec, DecodeError, Prob, RangeDecoder, RangeEncoder, Word};

/// Bit-split block codec for words of type `W`.
#[derive(Debug)]
pub struct BitSplit<W>(PhantomData<W>);

impl<W> Default for BitSplit<W> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Width of the `r` field.
const R_BITS: u32 = 6;
/// Width of the dictionary-size field.
const D_BITS: u32 = 8;
/// Dictionary sizes tried (each plus an escape symbol fits a power of two).
const D_CHOICES: [usize; 9] = [0, 1, 3, 7, 15, 31, 63, 127, 255];
/// Below this many entries the previous symbol is the coding context.
const CONTEXT_MAX: usize = 8;

/// Split widths tried: the left part is at most half the word.
fn r_range<W: Word>() -> core::ops::RangeInclusive<u32> {
    W::BITS / 2..=W::BITS - 1
}

/// The low `r` bits.
const fn mask(r: u32) -> u64 {
    if r >= 64 { u64::MAX } else { (1 << r) - 1 }
}

/// Significant bits of `n` (0 for zero).
const fn bit_len(n: usize) -> u32 {
    n.bit_width()
}

/// Distinct left parts of `sorted` (ascending words) at split `r`, most
/// frequent first (ties: smaller value). Equal lefts are adjacent in a sorted
/// block, so this needs no hashing.
fn left_frequencies(sorted: &[u64], r: u32) -> Vec<(u64, usize)> {
    let left = |w: u64| if r >= 64 { 0 } else { w >> r };
    let mut runs: Vec<(u64, usize)> = Vec::new();
    for &w in sorted {
        match runs.last_mut() {
            Some((l, c)) if *l == left(w) => *c += 1,
            _ => runs.push((left(w), 1)),
        }
    }
    runs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    runs
}

/// Estimated bits for split `r` with a dictionary of the `d` most frequent lefts:
/// symbol entropy plus escaped lefts, the dictionary and the right parts.
#[allow(clippy::cast_precision_loss)] // block sizes are far below 2^52
fn estimate(freq: &[(u64, usize)], n: usize, r: u32, d: usize, word_bits: u32) -> f64 {
    let left_bits = f64::from(word_bits - r);
    let covered: usize = freq[..d].iter().map(|f| f.1).sum();
    let escaped = n - covered;
    let total = n as f64;
    let entropy: f64 = freq[..d]
        .iter()
        .map(|f| f.1)
        .chain([escaped])
        .filter(|&c| c > 0)
        .map(|c| -(c as f64) * (c as f64 / total).log2())
        .sum();
    entropy + (escaped + d) as f64 * left_bits + total * f64::from(r)
}

/// The `(r, dictionary)` with the smallest estimate.
fn choose<W: Word>(items: &[W]) -> (u32, Vec<u64>) {
    let mut sorted: Vec<u64> = items.iter().map(|w| w.to_u64()).collect();
    sorted.sort_unstable();
    let mut best: Option<(f64, u32, Vec<u64>)> = None;
    for r in r_range::<W>() {
        let freq = left_frequencies(&sorted, r);
        for d in D_CHOICES.map(|d| d.min(freq.len())) {
            let cost = estimate(&freq, items.len(), r, d, W::BITS);
            if best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, r, freq[..d].iter().map(|f| f.0).collect()));
            }
        }
    }
    best.map_or((W::BITS - 1, Vec::new()), |(_, r, dict)| (r, dict))
}

/// Range-coder probabilities: a bit tree over `1 << depth` symbols, per context.
fn tree(depth: u32, contexts: usize) -> Vec<Prob> {
    vec![Prob::default(); contexts << depth]
}

impl<W: Word> BlockCodec<W> for BitSplit<W> {
    fn encode_block(&self, items: &[W], out: &mut BitWriter) {
        let (r, dict) = choose(items);
        let (left_bits, d) = (W::BITS - r, dict.len());
        out.write(u64::from(r), R_BITS);
        out.write(d as u64, D_BITS);
        for &l in &dict {
            out.write(l, left_bits);
        }
        let symbol_of = |left: u64| dict.iter().position(|&l| l == left).unwrap_or(d);
        let depth = bit_len(d); // d + 1 symbols fit in 2^depth
        let contexts = if d < CONTEXT_MAX { d + 1 } else { 1 };
        let mut probs = tree(depth, contexts);
        {
            let mut rc = RangeEncoder::new(out);
            let mut prev = 0usize;
            for w in items {
                let left = w.to_u64() >> r;
                let symbol = symbol_of(left);
                let base = if contexts > 1 { prev << depth } else { 0 };
                let mut node = 1usize;
                for i in (0..depth).rev() {
                    let bit = (symbol >> i) & 1 == 1;
                    rc.bit(&mut probs[base + node], bit);
                    node = (node << 1) | usize::from(bit);
                }
                if symbol == d {
                    rc.direct(left, left_bits);
                }
                prev = symbol;
            }
            rc.finish();
        }
        for w in items {
            out.write(w.to_u64() & mask(r), r);
        }
    }

    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<W>) -> Result<(), DecodeError> {
        let r = u32::try_from(input.read(R_BITS)?).unwrap_or(u32::MAX);
        if !r_range::<W>().contains(&r) {
            return Err(DecodeError::Corrupt("bit split width out of range"));
        }
        let left_bits = W::BITS - r;
        let d = usize::try_from(input.read(D_BITS)?).unwrap_or(usize::MAX);
        let mut dict = Vec::with_capacity(d);
        for _ in 0..d {
            dict.push(input.read(left_bits)?);
        }
        let depth = bit_len(d);
        let contexts = if d < CONTEXT_MAX { d + 1 } else { 1 };
        let mut probs = tree(depth, contexts);
        let mut lefts = Vec::with_capacity(n);
        {
            let mut rc = RangeDecoder::new(input)?;
            let mut prev = 0usize;
            for _ in 0..n {
                let base = if contexts > 1 { prev << depth } else { 0 };
                let mut node = 1usize;
                for _ in 0..depth {
                    let bit = rc.bit(&mut probs[base + node])?;
                    node = (node << 1) | usize::from(bit);
                }
                let symbol = node - (1 << depth);
                let left = match symbol.cmp(&d) {
                    core::cmp::Ordering::Less => dict[symbol],
                    core::cmp::Ordering::Equal => rc.direct(left_bits)?,
                    core::cmp::Ordering::Greater => return Err(DecodeError::Corrupt("bit split symbol out of range")),
                };
                lefts.push(left);
                prev = symbol;
            }
        }
        for left in lefts {
            out.push(W::from_u64_truncate((left << r) | input.read(r)?));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bits<W: Word>(v: &[W]) -> usize {
        let mut w = BitWriter::new();
        BitSplit::<W>::default().encode_block(v, &mut w);
        w.bit_len()
    }

    fn round_trip<W: Word>(v: &[W]) -> Vec<W> {
        let mut w = BitWriter::new();
        BitSplit::<W>::default().encode_block(v, &mut w);
        let bytes = w.finish();
        let mut out = Vec::new();
        BitSplit::<W>::default().decode_block(v.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        out
    }

    /// Doubles uniform in [lo, lo + span): random low bits, few high patterns.
    fn real_doubles(n: usize, lo: f64, span: f64) -> Vec<u64> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                #[allow(clippy::cast_precision_loss)]
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                (lo + span * u).to_bits()
            })
            .collect()
    }

    #[test]
    fn left_frequencies_by_count_then_value() {
        let sorted = [0b0001_0000u64, 0b0001_0001, 0b0010_0000, 0b0011_0000, 0b0011_0001, 0b0011_0010];
        assert_eq!(left_frequencies(&sorted, 4), [(3, 3), (1, 2), (2, 1)]);
        assert_eq!(left_frequencies(&sorted, 8), [(0, 6)]);
        assert_eq!(left_frequencies(&sorted, 0).len(), 6);
    }

    #[test]
    fn estimate_counts_every_part() {
        // 4 items, all left 7: zero symbol entropy, 1 dictionary entry.
        let freq = [(7u64, 4usize)];
        assert!((estimate(&freq, 4, 60, 1, 64) - (4.0 * 60.0 + 4.0)).abs() < 1e-9);
        // No dictionary: every left escaped (4 bits each) plus 4 x 60 right bits.
        assert!((estimate(&freq, 4, 60, 0, 64) - (4.0 * 4.0 + 4.0 * 60.0)).abs() < 1e-9);
    }

    #[test]
    fn split_ranges() {
        assert_eq!(r_range::<u64>(), 32..=63);
        assert_eq!(r_range::<u32>(), 16..=31);
        assert_eq!(mask(0), 0);
        assert_eq!(mask(3), 0b111);
        assert_eq!(mask(64), u64::MAX);
    }

    #[test]
    fn empty_blocks_take_the_first_split() {
        // Every (r, d = 0) costs nothing; the first candidate wins the tie.
        assert_eq!(choose::<u64>(&[]), (32, vec![]));
        assert_eq!(choose::<u32>(&[]), (16, vec![]));
    }

    #[test]
    fn eight_entry_dictionaries_use_one_context() {
        // Exactly CONTEXT_MAX skewed high patterns: the smallest dictionary coded
        // without previous-symbol contexts, at the boundary of the context table.
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let v: Vec<u64> = (0..2000u64)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let pattern = (i % 100).saturating_sub(92); // 0 for 93%, then 1..=7
                (0x3F0 << 52) | (pattern << 52) | (x >> 12)
            })
            .collect();
        let (r, dict) = choose(&v);
        assert_eq!((r, dict.len()), (52, CONTEXT_MAX));
        assert_eq!(round_trip(&v), v);
    }

    #[test]
    fn real_doubles_beat_raw_by_far() {
        let v = real_doubles(1000, 0.6, 0.1);
        assert!(bits(&v) < 1000 * 56, "{} bits", bits(&v));
        assert_eq!(round_trip(&v), v);
    }

    #[test]
    fn choice_picks_shared_high_bits() {
        // Lefts above bit 40 take two values; the low 40 bits are noise.
        let v: Vec<u64> = (0..500u64).map(|i| ((i % 2 + 5) << 40) | (i.wrapping_mul(0x9E37_79B9) & mask(40))).collect();
        let (r, dict) = choose(&v);
        assert!(r <= 40, "r = {r}");
        assert!(dict.len() <= 3);
        assert!(bits(&v) < 500 * 43, "{} bits", bits(&v));
        assert_eq!(round_trip(&v), v);
    }

    fn decode_raw(n: usize, fields: &[(u64, u32)]) -> Result<Vec<u64>, DecodeError> {
        let mut w = BitWriter::new();
        for &(v, b) in fields {
            w.write(v, b);
        }
        let bytes = w.finish();
        let mut out = Vec::new();
        BitSplit::<u64>::default().decode_block(n, &mut BitReader::new(&bytes), &mut out).map(|()| out)
    }

    #[test]
    fn split_width_out_of_range_is_corrupt() {
        assert_eq!(decode_raw(1, &[(31, 6)]), Err(DecodeError::Corrupt("bit split width out of range")));
        let mut w = BitWriter::new();
        w.write(15, 6);
        let bytes = w.finish();
        let r = BitSplit::<u32>::default().decode_block(1, &mut BitReader::new(&bytes), &mut Vec::new());
        assert_eq!(r, Err(DecodeError::Corrupt("bit split width out of range")));
    }

    #[test]
    fn symbol_out_of_range_is_corrupt() {
        // r = 63, d = 2 (lefts 0 and 1): the tree has room for symbol 3, which is invalid.
        let mut w = BitWriter::new();
        w.write(63, 6);
        w.write(2, 8);
        w.write(0, 1);
        w.write(1, 1);
        let mut probs = tree(2, 3);
        {
            let mut rc = RangeEncoder::new(&mut w);
            for (i, bit) in [true, true].into_iter().enumerate() {
                let node = if i == 0 { 1 } else { 3 };
                rc.bit(&mut probs[node], bit);
            }
            rc.finish();
        }
        w.write(0, 63);
        let bytes = w.finish();
        let r = BitSplit::<u64>::default().decode_block(1, &mut BitReader::new(&bytes), &mut Vec::new());
        assert_eq!(r, Err(DecodeError::Corrupt("bit split symbol out of range")));
    }

    #[test]
    fn truncated_is_eof() {
        let v = real_doubles(300, 1.0, 0.5);
        let mut w = BitWriter::new();
        BitSplit::<u64>::default().encode_block(&v, &mut w);
        let bytes = w.finish();
        for cut in [0, 1, bytes.len() / 2, bytes.len() - 1] {
            let r = BitSplit::<u64>::default().decode_block(v.len(), &mut BitReader::new(&bytes[..cut]), &mut Vec::new());
            assert_eq!(r, Err(DecodeError::UnexpectedEof), "cut at {cut}");
        }
    }

    proptest! {
        #[test]
        fn any_u64_round_trips(v in prop::collection::vec(any::<u64>(), 1..300)) {
            prop_assert_eq!(round_trip(&v), v);
        }

        #[test]
        fn any_u32_round_trips(v in prop::collection::vec(any::<u32>(), 1..300)) {
            prop_assert_eq!(round_trip(&v), v);
        }

        #[test]
        fn few_patterns_round_trip(v in prop::collection::vec((0u64..6, any::<u32>()), 1..400)) {
            let v: Vec<u64> = v.into_iter().map(|(hi, lo)| (hi << 50) | u64::from(lo)).collect();
            prop_assert_eq!(round_trip(&v), v);
        }

        #[test]
        fn never_much_worse_than_raw(v in prop::collection::vec(any::<u64>(), 1..300)) {
            // r field + d field + range-coder flush, then right bits plus escapes.
            prop_assert!(bits(&v) <= v.len() * 64 + 6 + 8 + 40 + v.len() * 2);
        }
    }
}
