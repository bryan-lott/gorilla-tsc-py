//! Decimal coding of floats: scale each block by a power of ten so that values
//! become integers, store values that do not survive the round trip as
//! exceptions, and delta-code the integers.
//!
//! Block layout: zig-zagged exponent `k` (6 bits), exception count (`L` bits),
//! then each exception as position (`P` bits) and raw word, then the integers as
//! a [`DeltaRice`] block of `i64` words. `L` = bit length of the block size and
//! `P` = bit length of the largest position. Exception slots hold the previous
//! integer, so they cost one delta of zero.

use core::marker::PhantomData;

use crate::{BitReader, BitWriter, BlockCodec, DecodeError, DeltaRice, Sample, Word};

/// Decimal block codec for samples of type `T`.
#[derive(Debug)]
pub struct DecimalRice<T>(PhantomData<T>);

impl<T> Default for DecimalRice<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// A block in scaled form.
#[derive(Debug, Clone, PartialEq)]
pub struct Scaled<W> {
    /// Decimal exponent.
    pub k: i32,
    /// One integer per item (exception slots repeat the previous integer).
    pub ints: Vec<u64>,
    /// `(position, raw word)` of items that do not scale exactly.
    pub exceptions: Vec<(usize, W)>,
}

/// Scale `items` by `10^k`.
#[must_use]
pub fn scale<T: Sample>(items: &[T::Word], k: i32) -> Scaled<T::Word> {
    let mut prev = 0i64;
    let mut ints = Vec::with_capacity(items.len());
    let mut exceptions = Vec::new();
    for (i, &w) in items.iter().enumerate() {
        match T::from_word(w).to_scaled(k) {
            Some(n) => prev = n,
            None => exceptions.push((i, w)),
        }
        ints.push(prev.cast_unsigned());
    }
    Scaled { k, ints, exceptions }
}

/// The exponents worth costing: each has fewer exceptions than every smaller one.
#[must_use]
pub fn candidate_exponents<T: Sample>(items: &[T::Word]) -> Vec<i32> {
    let mut best = usize::MAX;
    let mut out = Vec::new();
    for k in -T::MAX_SCALE..=T::MAX_SCALE {
        let ex = items.iter().filter(|&&w| T::from_word(w).to_scaled(k).is_none()).count();
        if ex < best {
            best = ex;
            out.push(k);
        }
    }
    out
}

/// Exact size of `scaled` once written, in bits.
#[must_use]
pub fn cost<W: Word>(scaled: &Scaled<W>) -> usize {
    let mut w = BitWriter::new();
    write_scaled(scaled, &mut w);
    w.bit_len()
}

/// The exponent [`DecimalRice`] uses for `items`: the cheapest candidate,
/// ties going to the smaller exponent.
#[must_use]
pub fn chosen_exponent<T: Sample>(items: &[T::Word]) -> i32 {
    candidate_exponents::<T>(items)
        .into_iter()
        .min_by_key(|&k| cost(&scale::<T>(items, k)))
        .unwrap_or(0)
}

/// Write a scaled block (everything after the codec chose `k`).
pub fn write_scaled<W: Word>(scaled: &Scaled<W>, out: &mut BitWriter) {
    let n = scaled.ints.len();
    out.write(i64::from(scaled.k).cast_unsigned().zigzag(), K_BITS);
    out.write(scaled.exceptions.len() as u64, bit_len(n));
    for &(pos, raw) in &scaled.exceptions {
        out.write(pos as u64, bit_len(n.saturating_sub(1)));
        out.write(raw.to_u64(), W::BITS);
    }
    DeltaRice.encode_block(&scaled.ints, out);
}

/// Width of the zig-zagged exponent field.
const K_BITS: u32 = 6;

fn bit_len(n: usize) -> u32 {
    n.bit_width()
}

impl<T: Sample> BlockCodec<T::Word> for DecimalRice<T> {
    fn encode_block(&self, items: &[T::Word], out: &mut BitWriter) {
        write_scaled(&scale::<T>(items, chosen_exponent::<T>(items)), out);
    }

    #[allow(clippy::cast_possible_truncation)] // field widths bound every value read
    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<T::Word>) -> Result<(), DecodeError> {
        let k = input.read(K_BITS)?.unzigzag().cast_signed();
        if k.unsigned_abs() > u64::from(T::MAX_SCALE.unsigned_abs()) {
            return Err(DecodeError::Corrupt("decimal exponent out of range"));
        }
        let k = k as i32;
        let count = input.read(bit_len(n))? as usize;
        if count > n {
            return Err(DecodeError::Corrupt("more exceptions than items"));
        }
        let mut exceptions = Vec::with_capacity(count);
        for _ in 0..count {
            let pos = input.read(bit_len(n.saturating_sub(1)))? as usize;
            if pos >= n {
                return Err(DecodeError::Corrupt("exception position out of range"));
            }
            if exceptions.last().is_some_and(|&(last, _)| pos <= last) {
                return Err(DecodeError::Corrupt("exception positions not increasing"));
            }
            exceptions.push((pos, T::Word::from_u64_truncate(input.read(T::Word::BITS)?)));
        }
        let mut ints: Vec<u64> = Vec::with_capacity(n);
        DeltaRice.decode_block(n, input, &mut ints)?;
        let mut exc = exceptions.into_iter().peekable();
        for (i, n) in ints.into_iter().enumerate() {
            match exc.next_if(|&(pos, _)| pos == i) {
                Some((_, raw)) => out.push(raw),
                None => out.push(T::from_scaled(n.cast_signed(), k).to_word()),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn words(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn bits(v: &[f64]) -> usize {
        let mut w = BitWriter::new();
        DecimalRice::<f64>::default().encode_block(&words(v), &mut w);
        w.bit_len()
    }

    fn round_trip_f64(v: &[f64]) -> Vec<u64> {
        let mut w = BitWriter::new();
        DecimalRice::<f64>::default().encode_block(&words(v), &mut w);
        let bytes = w.finish();
        let mut out = Vec::new();
        DecimalRice::<f64>::default().decode_block(v.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        out
    }

    fn round_trip_f32(v: &[f32]) -> Vec<u32> {
        let ws: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
        let mut w = BitWriter::new();
        DecimalRice::<f32>::default().encode_block(&ws, &mut w);
        let bytes = w.finish();
        let mut out = Vec::new();
        DecimalRice::<f32>::default().decode_block(v.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        out
    }

    #[test]
    fn scaling_f64() {
        assert_eq!(1.5f64.to_scaled(1), Some(15));
        assert_eq!(0.1f64.to_scaled(1), Some(1));
        assert_eq!(0.1f64.to_scaled(0), None);
        assert_eq!(1200.0f64.to_scaled(-2), Some(12));
        assert_eq!(1250.0f64.to_scaled(-2), None);
        assert_eq!((-3.25f64).to_scaled(2), Some(-325));
        assert_eq!(f64::from_scaled(-325, 2), -3.25);
        assert_eq!(f64::from_scaled(12, -2), 1200.0);
        assert_eq!(0.0f64.to_scaled(0), Some(0));
    }

    #[test]
    fn unscalable_f64() {
        for v in [-0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300, 5e-324] {
            for k in -18..=18 {
                assert_eq!(v.to_scaled(k), None, "{v} at {k}");
            }
        }
        // Beyond 2^53 integers are no longer exact.
        assert_eq!(9_007_199_254_740_993.0f64.to_scaled(0), None);
        assert_eq!(9_007_199_254_740_992.0f64.to_scaled(0), None);
        assert_eq!(4_503_599_627_370_495.0f64.to_scaled(0), Some(4_503_599_627_370_495));
    }

    #[test]
    fn scaling_f32() {
        assert_eq!(0.1f32.to_scaled(1), Some(1));
        assert_eq!(2.75f32.to_scaled(2), Some(275));
        assert_eq!(f32::from_scaled(275, 2), 2.75);
        assert_eq!((-0.0f32).to_scaled(0), None);
        assert_eq!(f32::NAN.to_scaled(1), None);
    }

    #[test]
    fn integers_only_scale_at_zero() {
        assert_eq!(5i64.to_scaled(0), Some(5));
        assert_eq!(5i64.to_scaled(1), None);
        assert_eq!(i32::MIN.to_scaled(0), Some(i64::from(i32::MIN)));
        assert_eq!(i32::from_scaled(-7, 0), -7);
        assert_eq!((i64::MAX_SCALE, i32::MAX_SCALE, f32::MAX_SCALE, f64::MAX_SCALE), (0, 0, 10, 18));
    }

    #[test]
    fn scale_fills_exceptions_with_previous_integer() {
        let s = scale::<f64>(&words(&[1.5, f64::NAN, 2.5, 0.123]), 1);
        assert_eq!(s.k, 1);
        assert_eq!(s.ints, vec![15, 15, 25, 25]);
        assert_eq!(s.exceptions, vec![(1, f64::NAN.to_bits()), (3, 0.123f64.to_bits())]);
        let lead = scale::<f64>(&words(&[f64::NAN, 2.0]), 0);
        assert_eq!(lead.ints, vec![0, 2]);
    }

    #[test]
    fn candidates_are_the_exception_front() {
        // 1 decimal everywhere except one 3-decimal value.
        let v = words(&[1.5, 2.5, 3.125, 4.5]);
        assert_eq!(candidate_exponents::<f64>(&v), vec![-18, 1, 3]);
        // multiples of 100
        let v = words(&[1200.0, 1300.0]);
        assert_eq!(candidate_exponents::<f64>(&v), vec![-18, -2]);
    }

    #[test]
    fn cost_matches_encoding() {
        let v = [1.5, 2.5, f64::NAN, 3.125, 4.5];
        for k in -2..=4 {
            let s = scale::<f64>(&words(&v), k);
            let mut w = BitWriter::new();
            write_scaled(&s, &mut w);
            assert_eq!(cost(&s), w.bit_len(), "k = {k}");
        }
    }

    #[test]
    fn chooses_cheapest_candidate() {
        // One high-precision outlier: an exception is cheaper than 15-digit integers.
        let mut v: Vec<f64> = (0..200).map(|i| f64::from(i % 50) / 10.0).collect();
        v[100] = 0.123_456_789_012_345;
        let ws = words(&v);
        let best = candidate_exponents::<f64>(&ws).into_iter().map(|k| cost(&scale::<f64>(&ws, k))).min().unwrap();
        assert_eq!(bits(&v), best);
        assert_eq!(chosen_exponent::<f64>(&ws), 1);
    }

    #[test]
    fn decimal_data_is_much_smaller_than_raw() {
        let v: Vec<f64> = (0..1000).map(|i| f64::from(200 + (i * 37) % 23) / 10.0).collect();
        assert!(bits(&v) < 1000 * 8, "{} bits", bits(&v));
    }

    #[test]
    fn special_values_round_trip() {
        let v = [f64::NAN, -0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, 5e-324, f64::MAX, 1.5];
        assert_eq!(round_trip_f64(&v), words(&v));
        let nan_payload = f64::from_bits(0x7FF8_0000_0000_0042);
        assert_eq!(round_trip_f64(&[nan_payload, 1.0]), words(&[nan_payload, 1.0]));
    }

    #[test]
    fn all_exceptions_block() {
        let v = [f64::NAN; 7];
        assert_eq!(round_trip_f64(&v), words(&v));
    }

    #[test]
    fn single_item_block() {
        assert_eq!(round_trip_f64(&[2.5]), words(&[2.5]));
    }

    fn decode_raw(n: usize, fields: &[(u64, u32)]) -> Result<(), DecodeError> {
        let mut w = BitWriter::new();
        for &(v, b) in fields {
            w.write(v, b);
        }
        w.write(0, 64); // room for anything that follows
        w.write(0, 64);
        let bytes = w.finish();
        DecimalRice::<f64>::default().decode_block(n, &mut BitReader::new(&bytes), &mut Vec::new())
    }

    #[test]
    fn exception_positions_are_validated() {
        // 3-item block: L = bit_len(3) = 2, P = bit_len(2) = 2.
        let out_of_range = [(0, 6), (1, 2), (3, 2), (0, 64)];
        assert_eq!(decode_raw(3, &out_of_range), Err(DecodeError::Corrupt("exception position out of range")));
        let repeated = [(0, 6), (2, 2), (1, 2), (0, 64), (1, 2), (0, 64)];
        assert_eq!(decode_raw(3, &repeated), Err(DecodeError::Corrupt("exception positions not increasing")));
    }

    #[test]
    fn exception_count_is_validated() {
        let mut w = BitWriter::new();
        w.write(0, 6);
        w.write(3, 2); // 3 exceptions in a 2-item block
        let bytes = w.finish();
        let r = DecimalRice::<f64>::default().decode_block(2, &mut BitReader::new(&bytes), &mut Vec::new());
        assert_eq!(r, Err(DecodeError::Corrupt("more exceptions than items")));
    }

    #[test]
    fn exponent_out_of_range_is_corrupt() {
        let mut w = BitWriter::new();
        w.write(19u64 << 1, 6); // zig-zag of 19 > MAX_SCALE
        let bytes = w.finish();
        let r = DecimalRice::<f64>::default().decode_block(1, &mut BitReader::new(&bytes), &mut Vec::new());
        assert_eq!(r, Err(DecodeError::Corrupt("decimal exponent out of range")));
    }

    proptest! {
        #[test]
        fn any_f64_round_trips(v in prop::collection::vec(any::<u64>(), 1..100)) {
            let f: Vec<f64> = v.iter().map(|&b| f64::from_bits(b)).collect();
            prop_assert_eq!(round_trip_f64(&f), v);
        }
        #[test]
        fn any_f32_round_trips(v in prop::collection::vec(any::<u32>(), 1..100)) {
            let f: Vec<f32> = v.iter().map(|&b| f32::from_bits(b)).collect();
            prop_assert_eq!(round_trip_f32(&f), v);
        }
        #[test]
        fn decimals_round_trip(v in prop::collection::vec((-1_000_000i32..1_000_000, 0i32..6), 1..200)) {
            let f: Vec<f64> = v.iter().map(|&(n, d)| f64::from(n) / 10f64.powi(d)).collect();
            prop_assert_eq!(round_trip_f64(&f), words(&f));
        }
        #[test]
        fn decimals_scale_without_exceptions(v in prop::collection::vec(-100_000i32..100_000, 1..100)) {
            let f: Vec<f64> = v.iter().map(|&n| f64::from(n) / 100.0).collect();
            let s = scale::<f64>(&words(&f), chosen_exponent::<f64>(&words(&f)));
            prop_assert!(s.exceptions.is_empty());
        }
    }
}
