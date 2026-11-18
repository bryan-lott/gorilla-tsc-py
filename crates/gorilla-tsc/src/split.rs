//! Decimal split: decimal scaling, then each integer as a reading plus a
//! remainder, `x = q * 10^j + r` with `r` the nearest remainder.
//!
//! Many float series are decimal readings plus a constant offset computed in
//! lower precision. Basel temperatures such as `1.9005287`, `1.6105287` and
//! `0.8205287` are two-decimal readings plus `0.0005287`, so once scaled their
//! integers are near multiples of 10^5 with an almost constant remainder.
//! Coding `q` and `r` separately is much cheaper than coding `x`.
//!
//! Block layout: the [`DecimalRice`](crate::DecimalRice) exponent and
//! exceptions, a 5-bit [`Split`], then one [`DeltaRice`] block of the
//! integers (no split) or of `q` then one of `r`.

use core::marker::PhantomData;

use crate::Scaled;
use crate::decimal::{chosen_scaled, read_scaled_header, unscale, write_scaled_header};
use crate::{BitReader, BitWriter, BlockCodec, DecodeError, DeltaRice, Sample, Word};

/// Decimal-split block codec for samples of type `T`.
#[derive(Debug)]
pub struct DecimalSplit<T>(PhantomData<T>);

impl<T> Default for DecimalSplit<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Where a block's integers are split. `10^j` always fits an `i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    /// The integers are coded whole.
    None,
    /// `x = q * 10^j + r`, `1 <= j <= 18`.
    At(u8),
}

/// Width of the split field.
const SPLIT_BITS: u32 = 5;
/// Largest `j` with `10^j` an `i64`.
const MAX_J: u8 = 18;
/// Split points fully encoded per block, best first by a cheap proxy.
const CANDIDATES: usize = 2;

impl Split {
    fn at(j: u8) -> Option<Self> {
        (1..=MAX_J).contains(&j).then_some(Self::At(j))
    }

    const fn code(self) -> u64 {
        match self {
            Self::None => 0,
            Self::At(j) => j as u64,
        }
    }

    fn from_code(code: u64) -> Option<Self> {
        match code {
            0 => Some(Self::None),
            j => u8::try_from(j).ok().and_then(Self::at),
        }
    }

    /// `10^j`.
    fn modulus(j: u8) -> i64 {
        10i64.pow(u32::from(j))
    }
}

/// `x` as `(q, r)` with `x = q * g + r` and `-g/2 <= r < g/2` (two's-complement words).
fn split(x: &[u64], g: i64) -> (Vec<u64>, Vec<u64>) {
    x.iter().map(|&word| split_one(word.cast_signed(), g)).map(|(q, r)| (q.cast_unsigned(), r.cast_unsigned())).unzip()
}

/// One value: `x = q * g + r` with `-g/2 <= r < g/2`, for even `g > 0`.
/// Euclidean division keeps `q * g` within `i64`, so nothing overflows.
const fn split_one(x: i64, g: i64) -> (i64, i64) {
    let (q, r) = (x.div_euclid(g), x.rem_euclid(g));
    if r >= g / 2 { (q + 1, r - g) } else { (q, r) }
}

/// Inverse of [`split`]; wraps exactly like the forward direction.
fn join(q: &[u64], r: &[u64], g: i64) -> Vec<u64> {
    let g = g.cast_unsigned();
    q.iter().zip(r).map(|(&q, &r)| q.wrapping_mul(g).wrapping_add(r)).collect()
}

/// A cheap size estimate: bit widths of zig-zagged first differences.
fn proxy(v: &[u64]) -> u64 {
    v.windows(2).map(|p| u64::from(u64::BITS - p[1].wrapping_sub(p[0]).zigzag().leading_zeros())).sum()
}

/// [`proxy`] of the two halves of splitting `x` at `g`, without building them.
fn split_proxy(x: &[u64], g: i64) -> u64 {
    let mut prev = split_one(x.first().map_or(0, |w| w.cast_signed()), g);
    let mut total = 0;
    for &w in x.iter().skip(1) {
        let cur = split_one(w.cast_signed(), g);
        total += u64::from(u64::BITS - cur.0.wrapping_sub(prev.0).cast_unsigned().zigzag().leading_zeros());
        total += u64::from(u64::BITS - cur.1.wrapping_sub(prev.1).cast_unsigned().zigzag().leading_zeros());
        prev = cur;
    }
    total
}

impl<T: Sample> BlockCodec<T::Word> for DecimalSplit<T> {
    fn encode_block(&self, items: &[T::Word], out: &mut BitWriter) {
        encode_scaled(&chosen_scaled::<T>(items), out);
    }

    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<T::Word>) -> Result<(), DecodeError> {
        decode::<T>(n, input, out)
    }
}

/// Encode a block already scaled by its decimal exponent.
pub(crate) fn encode_scaled<W: Word>(scaled: &Scaled<W>, out: &mut BitWriter) {
    write_scaled_header(scaled, out);
    let mut best = BitWriter::new();
    best.write(Split::None.code(), SPLIT_BITS);
    DeltaRice.encode_block(&scaled.ints, &mut best);
    // Splitting above the largest magnitude only moves everything into r, and
    // a split whose estimate is no better than the whole is not tried.
    let widest = scaled.ints.iter().map(|w| w.cast_signed().unsigned_abs()).max().unwrap_or(0);
    let whole = proxy(&scaled.ints);
    let mut ranked: Vec<(u64, u8)> = (1..=MAX_J)
        .take_while(|&j| Split::modulus(j).unsigned_abs() <= widest)
        .map(|j| (split_proxy(&scaled.ints, Split::modulus(j)), j))
        .filter(|&(p, _)| p < whole)
        .collect();
    ranked.sort_unstable();
    for &(_, j) in ranked.iter().take(CANDIDATES) {
        let (q, r) = split(&scaled.ints, Split::modulus(j));
        let mut w = BitWriter::new();
        w.write(u64::from(j), SPLIT_BITS);
        DeltaRice.encode_block(&q, &mut w);
        if w.bit_len() >= best.bit_len() {
            continue;
        }
        DeltaRice.encode_block(&r, &mut w);
        if w.bit_len() < best.bit_len() {
            best = w;
        }
    }
    out.append(&best);
}

fn decode<T: Sample>(n: usize, input: &mut BitReader<'_>, out: &mut Vec<T::Word>) -> Result<(), DecodeError> {
    let (k, exceptions) = read_scaled_header::<T>(n, input)?;
    let split_at =
        Split::from_code(input.read(SPLIT_BITS)?).ok_or(DecodeError::Corrupt("split exponent out of range"))?;
    let mut first: Vec<u64> = Vec::with_capacity(n);
    DeltaRice.decode_block(n, input, &mut first)?;
    let ints = match split_at {
        Split::None => first,
        Split::At(j) => {
            let mut r: Vec<u64> = Vec::with_capacity(n);
            DeltaRice.decode_block(n, input, &mut r)?;
            join(&first, &r, Split::modulus(j))
        }
    };
    unscale::<T>(ints, k, exceptions, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecimalRice;
    use proptest::prelude::*;

    fn words<T: Sample>(v: &[T]) -> Vec<T::Word> {
        v.iter().map(|x| x.to_word()).collect()
    }

    fn bits<T: Sample>(v: &[T]) -> usize {
        let mut w = BitWriter::new();
        DecimalSplit::<T>::default().encode_block(&words(v), &mut w);
        w.bit_len()
    }

    fn decimal_bits<T: Sample>(v: &[T]) -> usize {
        let mut w = BitWriter::new();
        DecimalRice::<T>::default().encode_block(&words(v), &mut w);
        w.bit_len()
    }

    fn round_trip<T: Sample>(v: &[T]) -> Vec<T::Word> {
        let mut w = BitWriter::new();
        DecimalSplit::<T>::default().encode_block(&words(v), &mut w);
        let bytes = w.finish();
        let mut out = Vec::new();
        DecimalSplit::<T>::default().decode_block(v.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        out
    }

    /// Basel-like: two-decimal readings plus a constant offset computed in
    /// float32 and printed at float32 precision (then parsed back as f64).
    fn offset_readings(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let reading = f32::from(i16::try_from((i / 3) % 300).unwrap() - 150) / 100.0;
                (reading + 0.000_528_7).to_string().parse().unwrap()
            })
            .collect()
    }

    #[test]
    fn split_codes() {
        assert_eq!(Split::None.code(), 0);
        assert_eq!(Split::at(5).map(Split::code), Some(5));
        assert_eq!(Split::at(0), None);
        assert_eq!(Split::at(19), None);
        assert_eq!(Split::at(18), Some(Split::At(18)));
        for c in 0..=18 {
            assert_eq!(Split::from_code(c).map(Split::code), Some(c));
        }
        assert_eq!(Split::from_code(19), None);
        assert_eq!(Split::from_code(31), None);
    }

    #[test]
    fn split_and_join() {
        let x: Vec<u64> = [1234i64, 1250, 1249, -1251, 0, -5, 5].map(i64::cast_unsigned).to_vec();
        let (q, r) = split(&x, 100);
        let qs: Vec<i64> = q.iter().map(|&v| v.cast_signed()).collect();
        let rs: Vec<i64> = r.iter().map(|&v| v.cast_signed()).collect();
        // nearest: 1250 -> 13*100 - 50, 1249 -> 12*100 + 49
        assert_eq!(qs, [12, 13, 12, -13, 0, 0, 0]);
        assert_eq!(rs, [34, -50, 49, 49, 0, -5, 5]);
        assert_eq!(join(&q, &r, 100), x);
    }

    #[test]
    fn split_wraps_at_the_extremes() {
        let x: Vec<u64> = [i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1].map(i64::cast_unsigned).to_vec();
        for g in [10, 1_000_000_000_000_000_000] {
            let (q, r) = split(&x, g);
            assert_eq!(join(&q, &r, g), x, "g = {g}");
        }
    }

    #[test]
    fn split_proxy_matches_splitting() {
        let x: Vec<u64> = (0..300i64).map(|i| (i * 7919 % 1000 * 10_000 + i % 7 - 3).cast_unsigned()).collect();
        for j in 1..=MAX_J {
            let g = Split::modulus(j);
            let (q, r) = split(&x, g);
            assert_eq!(split_proxy(&x, g), proxy(&q) + proxy(&r), "j = {j}");
        }
        assert_eq!(split_proxy(&[], 10), 0);
    }

    #[test]
    fn split_one_is_nearest() {
        assert_eq!(split_one(149, 100), (1, 49));
        assert_eq!(split_one(150, 100), (2, -50));
        assert_eq!(split_one(-150, 100), (-1, -50));
        assert_eq!(split_one(-151, 100), (-2, 49));
        assert_eq!(split_one(i64::MAX, 10), (i64::MAX / 10 + 1, -3));
        assert_eq!(split_one(i64::MIN, 10), (i64::MIN / 10 - 1, 2));
    }

    #[test]
    fn proxy_sums_difference_widths() {
        // diffs 3, -2, 0 -> zigzag 6, 3, 0 -> widths 3, 2, 0
        assert_eq!(proxy(&[10, 13, 11, 11]), 5);
        assert_eq!(proxy(&[7]), 0);
    }

    #[test]
    fn offset_readings_split_far_better_than_decimal() {
        let v = offset_readings(1000);
        assert!(bits(&v) * 2 < decimal_bits(&v), "{} vs {}", bits(&v), decimal_bits(&v));
        assert_eq!(round_trip(&v), words(&v));
    }

    #[test]
    fn never_much_worse_than_decimal() {
        // Plain decimal data: no split, so only the 5-bit field is extra.
        let v: Vec<f64> = (0..500).map(|i| f64::from(i % 97) / 10.0).collect();
        assert_eq!(bits(&v), decimal_bits(&v) + 5);
    }

    #[test]
    fn integers_split_too() {
        // Unordered slots on a 10 s grid plus a few ms of jitter: the slot and
        // the jitter are each cheap, their sum is not (and has no common divisor).
        let v: Vec<i64> = (0..1000).map(|i| 1_700_000_000_000 + 10_000 * ((i * 7919) % 1000) + (i * 13) % 7).collect();
        assert!(bits(&v) < decimal_bits(&v));
        assert_eq!(round_trip(&v), words(&v));
    }

    fn decode_raw(fields: &[(u64, u32)]) -> Result<(), DecodeError> {
        let mut w = BitWriter::new();
        for &(v, b) in fields {
            w.write(v, b);
        }
        let bytes = w.finish();
        DecimalSplit::<f64>::default().decode_block(1, &mut BitReader::new(&bytes), &mut Vec::new())
    }

    #[test]
    fn bad_split_is_corrupt() {
        // k = 0 (6 bits), no exceptions (1 bit for n = 1), split 19.
        assert_eq!(decode_raw(&[(0, 6), (0, 1), (19, 5)]), Err(DecodeError::Corrupt("split exponent out of range")));
    }

    #[test]
    fn truncated_is_eof() {
        let v = offset_readings(300);
        let mut w = BitWriter::new();
        DecimalSplit::<f64>::default().encode_block(&words(&v), &mut w);
        let bytes = w.finish();
        for cut in [0, 1, bytes.len() / 2, bytes.len() - 1] {
            let r = DecimalSplit::<f64>::default().decode_block(
                v.len(),
                &mut BitReader::new(&bytes[..cut]),
                &mut Vec::new(),
            );
            assert_eq!(r, Err(DecodeError::UnexpectedEof), "cut at {cut}");
        }
    }

    proptest! {
        #[test]
        fn any_f64_round_trips(v in prop::collection::vec(any::<u64>(), 1..200)) {
            let v: Vec<f64> = v.into_iter().map(f64::from_bits).collect();
            prop_assert_eq!(round_trip(&v), words(&v));
        }

        #[test]
        fn any_f32_round_trips(v in prop::collection::vec(any::<u32>(), 1..200)) {
            let v: Vec<f32> = v.into_iter().map(f32::from_bits).collect();
            prop_assert_eq!(round_trip(&v), words(&v));
        }

        #[test]
        fn any_i64_round_trips(v in prop::collection::vec(any::<i64>(), 1..200)) {
            prop_assert_eq!(round_trip(&v), words(&v));
        }

        #[test]
        fn decimals_round_trip_and_stay_close_to_decimal(v in prop::collection::vec((-100_000i32..100_000, 0i32..5), 1..200)) {
            let v: Vec<f64> = v.into_iter().map(|(m, d)| f64::from(m) / 10f64.powi(d)).collect();
            prop_assert_eq!(round_trip(&v), words(&v));
            prop_assert!(bits(&v) <= decimal_bits(&v) + 5);
        }
    }
}
