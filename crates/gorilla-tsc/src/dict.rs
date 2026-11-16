//! Dictionary coding: a block as its distinct values plus one index per item.
//!
//! Block layout: `d - 1` (`bit_len(n)` bits, `d` distinct values); a
//! dictionary-coder bit (`0` = [`DecimalRice`], `1` = [`DeltaRice`]) and the
//! distinct values in numeric order; then, if `d > 1`, a 2-bit
//! [`IndexCoder`] and one index per item.
//!
//! Sorted dictionaries have small gaps, so this pays off even when most
//! values in a block are distinct, not only for low-cardinality data.

use core::marker::PhantomData;

use crate::decimal::DecimalRice;
use crate::delta::DeltaRice;
use crate::{AdaptiveBlock, BitReader, BitWriter, BlockCodec, DecodeError, Sample};

/// Dictionary block codec for samples of type `T`.
#[derive(Debug)]
pub struct Dictionary<T>(PhantomData<T>);

impl<T> Default for Dictionary<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// How a block's indices are written. (Measured on the Elf datasets, these
/// win all but a handful of blocks; Rice variants were dropped.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexCoder {
    /// `bit_len(d - 1)` bits each.
    Fixed,
    /// [`AdaptiveBlock`] over the indices.
    Adaptive,
    /// [`AdaptiveBlock`] over differences of consecutive indices.
    AdaptiveDiff,
    /// [`AdaptiveBlock`] over move-to-front ranks (dictionaries up to [`MTF_MAX`]).
    AdaptiveMtf,
}

/// Move-to-front costs O(d) per item; beyond this many distinct values it is
/// not tried (it only wins on small, recency-skewed dictionaries).
const MTF_MAX: usize = 256;

/// Width of the index-coder field.
const INDEX_CODER_BITS: u32 = 2;

impl IndexCoder {
    const ALL: [Self; 4] = [Self::Fixed, Self::Adaptive, Self::AdaptiveDiff, Self::AdaptiveMtf];

    const fn code(self) -> u64 {
        match self {
            Self::Fixed => 0,
            Self::Adaptive => 1,
            Self::AdaptiveDiff => 2,
            Self::AdaptiveMtf => 3,
        }
    }

    fn from_code(code: u64) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.code() == code)
    }

    fn encode(self, idx: &[u64], d: usize, out: &mut BitWriter) {
        match self {
            Self::Fixed => {
                let width = bit_len(d - 1);
                for &i in idx {
                    out.write(i, width);
                }
            }
            Self::Adaptive => AdaptiveBlock::encode(idx, out),
            Self::AdaptiveDiff => AdaptiveBlock::encode(&diffs(idx), out),
            Self::AdaptiveMtf => AdaptiveBlock::encode(&mtf_forward(idx, d), out),
        }
    }

    fn decode(self, n: usize, d: usize, input: &mut BitReader<'_>) -> Result<Vec<u64>, DecodeError> {
        let mut seq = Vec::with_capacity(n);
        match self {
            Self::Fixed => {
                let width = bit_len(d - 1);
                for _ in 0..n {
                    seq.push(input.read(width)?);
                }
                Ok(seq)
            }
            Self::Adaptive => {
                AdaptiveBlock::decode(n, input, &mut seq)?;
                Ok(seq)
            }
            Self::AdaptiveDiff => {
                AdaptiveBlock::decode(n, input, &mut seq)?;
                Ok(undiffs(&seq))
            }
            Self::AdaptiveMtf => {
                AdaptiveBlock::decode(n, input, &mut seq)?;
                mtf_inverse(&seq, d)
            }
        }
    }
}

/// Significant bits of `n` (0 for zero).
fn bit_len(n: usize) -> u32 {
    n.bit_width()
}

/// Differences of consecutive indices (the first against 0), wrapping.
fn diffs(idx: &[u64]) -> Vec<u64> {
    let mut prev = 0u64;
    idx.iter()
        .map(|&i| {
            let d = i.wrapping_sub(prev);
            prev = i;
            d
        })
        .collect()
}

/// Inverse of [`diffs`].
fn undiffs(d: &[u64]) -> Vec<u64> {
    let mut prev = 0u64;
    d.iter()
        .map(|&x| {
            prev = prev.wrapping_add(x);
            prev
        })
        .collect()
}

/// Move-to-front ranks of `idx` over the alphabet `0..d`.
fn mtf_forward(idx: &[u64], d: usize) -> Vec<u64> {
    let mut order: Vec<u64> = (0..d as u64).collect();
    idx.iter()
        .map(|&i| {
            let pos = order.iter().position(|&x| x == i).unwrap_or(0);
            order[..=pos].rotate_right(1);
            pos as u64
        })
        .collect()
}

/// Inverse of [`mtf_forward`].
fn mtf_inverse(ranks: &[u64], d: usize) -> Result<Vec<u64>, DecodeError> {
    let mut order: Vec<u64> = (0..d as u64).collect();
    ranks
        .iter()
        .map(|&r| {
            let pos = usize::try_from(r)
                .ok()
                .filter(|&p| p < d)
                .ok_or(DecodeError::Corrupt("move-to-front rank out of range"))?;
            let i = order[pos];
            order[..=pos].rotate_right(1);
            Ok(i)
        })
        .collect()
}

impl<T: Sample> BlockCodec<T::Word> for Dictionary<T> {
    fn encode_block(&self, items: &[T::Word], out: &mut BitWriter) {
        let n = items.len();
        let key = |w: T::Word| T::from_word(w).order_key();
        let mut dict = items.to_vec();
        dict.sort_unstable_by_key(|&w| key(w));
        dict.dedup();
        let d = dict.len();
        out.write(d.saturating_sub(1) as u64, bit_len(n));
        let mut decimal = BitWriter::new();
        DecimalRice::<T>::default().encode_block(&dict, &mut decimal);
        let mut delta = BitWriter::new();
        DeltaRice.encode_block(&dict, &mut delta);
        if decimal.bit_len() <= delta.bit_len() {
            out.write(0, 1);
            out.append(&decimal);
        } else {
            out.write(1, 1);
            out.append(&delta);
        }
        if d <= 1 {
            return;
        }
        let idx: Vec<u64> =
            items.iter().map(|&w| dict.binary_search_by_key(&key(w), |&x| key(x)).map_or(0, |i| i as u64)).collect();
        let best = IndexCoder::ALL
            .into_iter()
            .filter(|&c| c != IndexCoder::AdaptiveMtf || d <= MTF_MAX)
            .map(|c| {
                let mut w = BitWriter::new();
                c.encode(&idx, d, &mut w);
                (c, w)
            })
            .min_by_key(|(_, w)| w.bit_len());
        if let Some((coder, w)) = best {
            out.write(coder.code(), INDEX_CODER_BITS);
            out.append(&w);
        }
    }

    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<T::Word>) -> Result<(), DecodeError> {
        let d = usize::try_from(input.read(bit_len(n))?).unwrap_or(usize::MAX).saturating_add(1);
        if d > n {
            return Err(DecodeError::Corrupt("dictionary larger than block"));
        }
        let mut dict = Vec::with_capacity(d);
        if input.read_bit()? {
            DeltaRice.decode_block(d, input, &mut dict)?;
        } else {
            DecimalRice::<T>::default().decode_block(d, input, &mut dict)?;
        }
        if d == 1 {
            out.extend(core::iter::repeat_n(dict[0], n));
            return Ok(());
        }
        // Every 2-bit code is a coder, so this cannot fail on well-sized fields.
        let coder = IndexCoder::from_code(input.read(INDEX_CODER_BITS)?).ok_or(DecodeError::Corrupt("unknown index coder"))?;
        for i in coder.decode(n, d, input)? {
            let w = usize::try_from(i).ok().and_then(|i| dict.get(i));
            out.push(*w.ok_or(DecodeError::Corrupt("dictionary index out of range"))?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bits<T: Sample>(v: &[T]) -> usize {
        let words: Vec<T::Word> = v.iter().map(|x| x.to_word()).collect();
        let mut w = BitWriter::new();
        Dictionary::<T>::default().encode_block(&words, &mut w);
        w.bit_len()
    }

    fn round_trip<T: Sample>(v: &[T]) -> Vec<T::Word> {
        let words: Vec<T::Word> = v.iter().map(|x| x.to_word()).collect();
        let mut w = BitWriter::new();
        Dictionary::<T>::default().encode_block(&words, &mut w);
        let bytes = w.finish();
        let mut out = Vec::new();
        Dictionary::<T>::default().decode_block(words.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        out
    }

    fn words<T: Sample>(v: &[T]) -> Vec<T::Word> {
        v.iter().map(|x| x.to_word()).collect()
    }

    fn decimal_bits(v: &[f64]) -> usize {
        let mut w = BitWriter::new();
        DecimalRice::<f64>::default().encode_block(&words(v), &mut w);
        w.bit_len()
    }

    #[test]
    fn index_coder_codes_round_trip() {
        for c in IndexCoder::ALL {
            assert_eq!(IndexCoder::from_code(c.code()), Some(c));
        }
        assert_eq!(IndexCoder::ALL.map(IndexCoder::code), [0, 1, 2, 3]);
        assert_eq!(IndexCoder::from_code(4), None);
    }

    #[test]
    fn move_to_front() {
        assert_eq!(mtf_forward(&[2, 2, 0, 2, 1, 1], 3), [2, 0, 1, 1, 2, 0]);
        assert_eq!(mtf_inverse(&[2, 0, 1, 1, 2, 0], 3), Ok(vec![2, 2, 0, 2, 1, 1]));
        assert_eq!(mtf_inverse(&[3], 3), Err(DecodeError::Corrupt("move-to-front rank out of range")));
    }

    #[test]
    fn differences_invert() {
        let idx = [3u64, 1, 4, 1, 5];
        assert_eq!(diffs(&idx), [3, 0u64.wrapping_sub(2), 3, 0u64.wrapping_sub(3), 4]);
        assert_eq!(undiffs(&diffs(&idx)), idx);
    }

    #[test]
    fn every_index_coder_round_trips() {
        let idx: Vec<u64> = (0..300u64).map(|i| (i * 7 + i / 13) % 11).collect();
        for c in IndexCoder::ALL {
            let mut w = BitWriter::new();
            c.encode(&idx, 11, &mut w);
            let len = w.bit_len();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(c.decode(idx.len(), 11, &mut r), Ok(idx.clone()), "{c:?}");
            assert_eq!(r.remaining() + len, bytes.len() * 8, "{c:?} read exactly what it wrote");
        }
        // Fixed width: bit_len(10) = 4 bits each.
        let mut w = BitWriter::new();
        IndexCoder::Fixed.encode(&idx, 11, &mut w);
        assert_eq!(w.bit_len(), 300 * 4);
    }

    #[test]
    fn constant_block_is_just_the_value() {
        let v = [3.25f64; 500];
        // d - 1 = 0 in bit_len(500) = 9 bits, the coder bit, then a one-item dictionary.
        let one = {
            let mut w = BitWriter::new();
            DecimalRice::<f64>::default().encode_block(&words(&v[..1]), &mut w);
            w.bit_len()
        };
        assert_eq!(bits(&v), 9 + 1 + one);
        assert_eq!(round_trip(&v), words(&v));
    }

    #[test]
    fn few_distinct_values_beat_decimal_by_far() {
        let levels = [12.5, 17.25, 3.0, 99.75, 12.5];
        let v: Vec<f64> = (0..1000).map(|i| levels[(i * 31 + i / 7) % 5]).collect();
        assert!(bits(&v) * 2 < decimal_bits(&v), "{} vs {}", bits(&v), decimal_bits(&v));
        assert_eq!(round_trip(&v), words(&v));
    }

    #[test]
    fn picks_the_cheapest_index_coder() {
        let v: Vec<i64> = (0..400).map(|i| [5, 9, 5, 5, 2][i % 5]).collect();
        let w = words(&v);
        let mut dict = w.clone();
        dict.sort_unstable_by_key(|&x| i64::from_word(x).order_key());
        dict.dedup();
        let idx: Vec<u64> = w.iter().map(|x| dict.iter().position(|d| d == x).unwrap() as u64).collect();
        let best = IndexCoder::ALL
            .iter()
            .map(|c| {
                let mut b = BitWriter::new();
                c.encode(&idx, dict.len(), &mut b);
                b.bit_len()
            })
            .min()
            .unwrap();
        let mut head = BitWriter::new();
        DecimalRice::<i64>::default().encode_block(&dict, &mut head);
        let mut alt = BitWriter::new();
        DeltaRice.encode_block(&dict, &mut alt);
        let dict_bits = head.bit_len().min(alt.bit_len());
        assert_eq!(bits(&v), 9 + 1 + dict_bits + 2 + best);
    }

    #[test]
    fn large_dictionaries_round_trip() {
        let v: Vec<i32> = (0..600).map(|i| (i * 7919) % 300).collect();
        let w = words(&v);
        let mut out = BitWriter::new();
        Dictionary::<i32>::default().encode_block(&w, &mut out);
        let bytes = out.finish();
        let mut r = BitReader::new(&bytes);
        let d = usize::try_from(r.read(bit_len(600)).unwrap()).unwrap() + 1;
        assert_eq!(d, 300);
        assert_eq!(round_trip(&v), w);
    }

    #[test]
    fn numeric_order_dictionary() {
        // Sorted by value, not by bit pattern: negatives first, -0.0 before 0.0.
        let v = [2.0f64, -1.0, 0.0, -0.0, f64::NEG_INFINITY, f64::NAN];
        let mut keys: Vec<i64> = v.iter().map(|x| x.order_key()).collect();
        keys.sort_unstable();
        let order: Vec<u64> = keys
            .iter()
            .map(|k| v.iter().find(|x| x.order_key() == *k).unwrap().to_bits())
            .collect();
        let expected = [f64::NEG_INFINITY, -1.0, -0.0, 0.0, 2.0, f64::NAN].map(f64::to_bits);
        assert_eq!(order, expected);
        assert!((-5i32).order_key() < 3i32.order_key());
        assert!((-0.5f32).order_key() < 0.25f32.order_key());
        assert_eq!(round_trip(&v), words(&v));
    }

    fn decode_raw(n: usize, fields: &[(u64, u32)]) -> Result<Vec<u64>, DecodeError> {
        let mut w = BitWriter::new();
        for &(v, b) in fields {
            w.write(v, b);
        }
        let bytes = w.finish();
        let mut out = Vec::new();
        Dictionary::<i64>::default().decode_block(n, &mut BitReader::new(&bytes), &mut out).map(|()| out)
    }

    /// A Delta-coded one-entry block for `[7]`: zigzag(7) = 14, length 4.
    const ONE_SEVEN: [(u64, u32); 2] = [(4, 7), (14, 4)];

    #[test]
    fn dictionary_larger_than_block_is_corrupt() {
        // n = 2: d - 1 field is 2 bits; d = 4 > 2.
        assert_eq!(decode_raw(2, &[(3, 2)]), Err(DecodeError::Corrupt("dictionary larger than block")));
    }

    #[test]
    fn index_out_of_range_is_corrupt() {
        // n = 2, d = 1 is a constant block: fine.
        let mut f = vec![(0, 2), (1, 1)];
        f.extend(ONE_SEVEN);
        assert_eq!(decode_raw(2, &f), Ok(vec![7, 7]));
        // Fixed-width indices into a 2-entry dictionary can't exceed 1, but
        // adaptive ones can: d = 2, Delta dictionary [7, 8], index 5.
        let mut idx = BitWriter::new();
        AdaptiveBlock::encode(&[0u64, 5], &mut idx);
        let mut w = BitWriter::new();
        w.write(1, 2);
        w.write(1, 1);
        DeltaRice.encode_block(&[7u64, 8], &mut w);
        w.write(IndexCoder::Adaptive.code(), 2);
        w.append(&idx);
        let bytes = w.finish();
        let mut out = Vec::new();
        assert_eq!(
            Dictionary::<i64>::default().decode_block(2, &mut BitReader::new(&bytes), &mut out),
            Err(DecodeError::Corrupt("dictionary index out of range"))
        );
    }

    #[test]
    fn truncated_is_eof() {
        let v: Vec<f64> = (0..200).map(|i| f64::from(i % 9) / 4.0).collect();
        let w = words(&v);
        let mut out = BitWriter::new();
        Dictionary::<f64>::default().encode_block(&w, &mut out);
        let bytes = out.finish();
        for cut in [0, 1, bytes.len() / 2, bytes.len() - 1] {
            let mut o = Vec::new();
            assert_eq!(
                Dictionary::<f64>::default().decode_block(w.len(), &mut BitReader::new(&bytes[..cut]), &mut o),
                Err(DecodeError::UnexpectedEof),
                "cut at {cut}"
            );
        }
    }

    proptest! {
        #[test]
        fn any_f64_round_trips(v in prop::collection::vec(any::<u64>(), 1..300)) {
            let v: Vec<f64> = v.into_iter().map(f64::from_bits).collect();
            prop_assert_eq!(round_trip(&v), words(&v));
        }

        #[test]
        fn any_f32_round_trips(v in prop::collection::vec(any::<u32>(), 1..300)) {
            let v: Vec<f32> = v.into_iter().map(f32::from_bits).collect();
            prop_assert_eq!(round_trip(&v), words(&v));
        }

        #[test]
        fn low_cardinality_round_trips(v in prop::collection::vec(-4i64..4, 1..600)) {
            prop_assert_eq!(round_trip(&v), words(&v));
        }

        #[test]
        fn any_i32_round_trips(v in prop::collection::vec(any::<i32>(), 1..300)) {
            prop_assert_eq!(round_trip(&v), words(&v));
        }

        #[test]
        fn order_key_is_monotone_and_injective(a in any::<u64>(), b in any::<u64>()) {
            let (x, y) = (f64::from_bits(a), f64::from_bits(b));
            prop_assert_eq!(x.order_key() == y.order_key(), a == b);
            if !x.is_nan() && !y.is_nan() {
                prop_assert_eq!(x.order_key().cmp(&y.order_key()), x.total_cmp(&y));
            }
        }
    }
}
