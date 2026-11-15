//! Rice coding of a block of residuals, with parameters chosen per block.
//!
//! Block layout: `transform` (1 bit), then for [`Transform::Offset`] the block
//! minimum, zig-zagged and length-prefixed (bit length in log2(BITS)+1 bits,
//! then that many bits); a scale flag (1 bit) and, if set, the common divisor
//! `g > 1` of the residuals, length-prefixed; then `k` (log2(BITS) bits); then
//! one code per value of `z / g`:
//! `q = z >> k` in unary (`q` ones and a zero) followed by the low `k` bits.
//! A quotient of at least `escape_q` is written as `escape_q` ones followed by
//! the raw word, which bounds the cost of outliers.

use crate::{BitReader, BitWriter, DecodeError, Word};

/// How residuals are mapped to the unsigned values that get Rice coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Zig-zag: small magnitudes of either sign become small values.
    ZigZag,
    /// Subtract the block minimum (signed): suits residuals clustered away from zero.
    Offset,
}

/// Parameters chosen for one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RiceParams<W> {
    /// Residual mapping.
    pub transform: Transform,
    /// Block minimum, for [`Transform::Offset`].
    pub base: W,
    /// Common divisor of the mapped residuals (1 = none).
    pub scale: u64,
    /// Rice parameter: number of low bits stored verbatim.
    pub k: u32,
}

/// Rice coder for a block of residual words.
#[derive(Debug, Default, Clone, Copy)]
pub struct RiceBlock;

/// Quotients at or above this are escaped to a raw word.
const fn escape_q<W: Word>() -> u64 {
    W::BITS as u64 / 2
}

/// Width of the offset base's length field (0..=BITS).
const fn len_bits<W: Word>() -> u32 {
    W::BITS.trailing_zeros() + 1
}

/// Significant bits of `v` (0 for zero).
const fn bit_length(v: u64) -> u32 {
    v.bit_width()
}

/// Width of the `k` field.
const fn k_bits<W: Word>() -> u32 {
    W::BITS.trailing_zeros()
}

impl<W: Word> RiceParams<W> {
    /// The unsigned value coded for residual `r` (which `scale` divides).
    fn map(self, r: W) -> u64 {
        match self.transform {
            Transform::ZigZag => {
                W::from_i128_truncate(r.to_signed_i128() / i128::from(self.scale)).zigzag().to_u64()
            }
            Transform::Offset => r.wrapping_sub(self.base).to_u64() / self.scale,
        }
    }

    /// Inverse of [`RiceParams::map`].
    fn unmap(self, z: u64) -> W {
        match self.transform {
            Transform::ZigZag => {
                let q = W::from_u64_truncate(z).unzigzag().to_signed_i128();
                W::from_i128_truncate(q.wrapping_mul(i128::from(self.scale)))
            }
            Transform::Offset => W::from_u64_truncate(z.wrapping_mul(self.scale)).wrapping_add(self.base),
        }
    }

    fn header_bits(self) -> u64 {
        let base = match self.transform {
            Transform::ZigZag => 0,
            Transform::Offset => len_bits::<W>() + bit_length(self.base.zigzag().to_u64()),
        };
        let scale = if self.scale > 1 { len_bits::<W>() + bit_length(self.scale) } else { 0 };
        2 + u64::from(k_bits::<W>()) + u64::from(base) + u64::from(scale)
    }
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Common divisor of the values `transform` maps `residuals` to before scaling.
fn common_divisor<W: Word>(residuals: &[W], transform: Transform, base: W) -> u64 {
    residuals.iter().fold(0, |g, &r| {
        let v = match transform {
            Transform::ZigZag => u64::try_from(r.to_signed_i128().unsigned_abs()).unwrap_or(0),
            Transform::Offset => r.wrapping_sub(base).to_u64(),
        };
        gcd(g, v)
    })
}

/// Common divisor of the signed magnitudes of `residuals` (0 if all are zero).
pub(crate) fn signed_divisor<W: Word>(residuals: &[W]) -> u64 {
    common_divisor(residuals, Transform::ZigZag, W::ZERO)
}

/// Write `v` as its bit length (log2(BITS)+1 bits) then its significant bits.
pub(crate) fn write_prefixed<W: Word>(out: &mut BitWriter, v: u64) {
    out.write(u64::from(bit_length(v)), len_bits::<W>());
    out.write(v, bit_length(v));
}

/// Inverse of [`write_prefixed`]; `what` names the error for an overlong length.
#[allow(clippy::cast_possible_truncation)] // the length field is at most 7 bits
pub(crate) fn read_prefixed<W: Word>(input: &mut BitReader<'_>, what: &'static str) -> Result<u64, DecodeError> {
    let len = input.read(len_bits::<W>())? as u32;
    if len > W::BITS {
        return Err(DecodeError::Corrupt(what));
    }
    input.read(len)
}

/// Bits for one mapped value `z` with parameter `k`.
fn value_cost<W: Word>(z: u64, k: u32) -> u64 {
    let q = z >> k;
    if q < escape_q::<W>() { q + 1 + u64::from(k) } else { escape_q::<W>() + u64::from(W::BITS) }
}

fn mapped_cost<W: Word>(zs: &[u64], k: u32) -> u64 {
    zs.iter().map(|&z| value_cost::<W>(z, k)).sum()
}

impl RiceBlock {
    /// Exact cost in bits of coding `residuals` with `params`, header included.
    #[must_use]
    pub fn cost<W: Word>(residuals: &[W], params: RiceParams<W>) -> u64 {
        params.header_bits() + residuals.iter().map(|&r| value_cost::<W>(params.map(r), params.k)).sum::<u64>()
    }

    /// The cheapest parameters for `residuals` (exhaustive over transform and `k`).
    /// Ties prefer zig-zag, then the smaller `k`.
    #[must_use]
    pub fn choose<W: Word>(residuals: &[W]) -> RiceParams<W> {
        let base = residuals.iter().copied().min_by_key(|w| w.to_signed_i128()).unwrap_or(W::ZERO);
        let mut best: Option<(u64, RiceParams<W>)> = None;
        for (transform, base) in [(Transform::ZigZag, W::ZERO), (Transform::Offset, base)] {
            let g = common_divisor(residuals, transform, base);
            let scales: &[u64] = if g > 1 { &[1, g] } else { &[1] };
            for &scale in scales {
                let probe = RiceParams { transform, base, scale, k: 0 };
                let zs: Vec<u64> = residuals.iter().map(|&r| probe.map(r)).collect();
                for k in 0..W::BITS {
                    let p = RiceParams { k, ..probe };
                    let c = p.header_bits() + mapped_cost::<W>(&zs, k);
                    if best.is_none_or(|(b, _)| c < b) {
                        best = Some((c, p));
                    }
                }
            }
        }
        best.map_or(RiceParams { transform: Transform::ZigZag, base: W::ZERO, scale: 1, k: 0 }, |b| b.1)
    }

    /// Append `residuals` using the cheapest parameters.
    pub fn encode<W: Word>(residuals: &[W], out: &mut BitWriter) {
        let p = Self::choose(residuals);
        out.write(u64::from(p.transform == Transform::Offset), 1);
        if p.transform == Transform::Offset {
            write_prefixed::<W>(out, p.base.zigzag().to_u64());
        }
        out.write(u64::from(p.scale > 1), 1);
        if p.scale > 1 {
            write_prefixed::<W>(out, p.scale);
        }
        out.write(u64::from(p.k), k_bits::<W>());
        let esc = escape_q::<W>();
        for &r in residuals {
            let z = p.map(r);
            let q = z >> p.k;
            if q < esc {
                // q ones then a zero: q + 1 <= 32 bits.
                out.write(((1u64 << q) - 1) << 1, u32::try_from(q).unwrap_or(0) + 1);
                out.write(z & low_mask(p.k), p.k);
            } else {
                out.write((1u64 << esc) - 1, u32::try_from(esc).unwrap_or(0));
                out.write(z, W::BITS);
            }
        }
    }

    /// Read `n` residuals, appending them to `out`.
    ///
    /// # Errors
    /// On truncated input.
    #[allow(clippy::cast_possible_truncation)] // k field is at most 6 bits
    pub fn decode<W: Word>(n: usize, input: &mut BitReader<'_>, out: &mut Vec<W>) -> Result<(), DecodeError> {
        let transform = if input.read_bit()? { Transform::Offset } else { Transform::ZigZag };
        let base = if transform == Transform::Offset {
            W::from_u64_truncate(read_prefixed::<W>(input, "Rice base longer than the word")?).unzigzag()
        } else {
            W::ZERO
        };
        let scale = if input.read_bit()? {
            let g = read_prefixed::<W>(input, "Rice scale longer than the word")?;
            if g <= 1 {
                return Err(DecodeError::Corrupt("Rice scale must exceed 1"));
            }
            g
        } else {
            1
        };
        let p = RiceParams { transform, base, scale, k: input.read(k_bits::<W>())? as u32 };
        let esc = escape_q::<W>();
        for _ in 0..n {
            let mut q = 0;
            while q < esc && input.read_bit()? {
                q += 1;
            }
            let z = if q == esc { input.read(W::BITS)? } else { (q << p.k) | input.read(p.k)? };
            out.push(p.unmap(z));
        }
        Ok(())
    }
}

const fn low_mask(k: u32) -> u64 {
    if k >= 64 { u64::MAX } else { (1u64 << k) - 1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn signed(v: &[i64]) -> Vec<u64> {
        v.iter().map(|x| x.cast_unsigned()).collect()
    }

    fn encoded_bits<W: Word>(r: &[W]) -> usize {
        let mut w = BitWriter::new();
        RiceBlock::encode(r, &mut w);
        w.bit_len()
    }

    fn round_trip<W: Word>(r: &[W]) -> Vec<W> {
        let mut w = BitWriter::new();
        RiceBlock::encode(r, &mut w);
        let bytes = w.finish();
        let mut out = Vec::new();
        RiceBlock::decode(r.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        out
    }

    fn zz(p: Transform, k: u32) -> RiceParams<u64> {
        RiceParams { transform: p, base: 0, scale: 1, k }
    }

    #[test]
    fn field_widths() {
        assert_eq!((k_bits::<u64>(), k_bits::<u32>()), (6, 5));
        assert_eq!((len_bits::<u64>(), len_bits::<u32>()), (7, 6));
        assert_eq!((escape_q::<u64>(), escape_q::<u32>()), (32, 16));
    }

    #[test]
    fn cost_formula() {
        // header: transform 1 + scale flag 1 + k 6; each value: q ones + 1 + k
        let r = signed(&[0, 1, -1, 5]); // zigzag 0, 2, 1, 10
        assert_eq!(RiceBlock::cost(&r, zz(Transform::ZigZag, 0)), 8 + (1 + 3 + 2 + 11));
        assert_eq!(RiceBlock::cost(&r, zz(Transform::ZigZag, 2)), 8 + 4 * 3 + 2); // q = 0,0,0,2
        // offset adds the base: 7-bit length + zig-zag(base) bits
        let p = RiceParams { transform: Transform::Offset, base: 0u64, scale: 1, k: 0 };
        assert_eq!(RiceBlock::cost(&[3u64], p), 8 + 7 + 4);
        let p = RiceParams { transform: Transform::Offset, base: 10_000u64, scale: 1, k: 0 };
        assert_eq!(RiceBlock::cost(&[10_003u64], p), 8 + 7 + 15 + 4); // zz(10000) = 20000: 15 bits
        let p = RiceParams { transform: Transform::Offset, base: i64::MIN.cast_unsigned(), scale: 1, k: 0 };
        assert_eq!(RiceBlock::cost(&[i64::MIN.cast_unsigned()], p), 8 + 7 + 64 + 1);
        let p32 = RiceParams { transform: Transform::Offset, base: 1u32, scale: 1, k: 0 };
        assert_eq!(RiceBlock::cost(&[1u32], p32), 7 + 6 + 2 + 1);
    }

    #[test]
    fn escape_bounds_outlier_cost() {
        let p = zz(Transform::ZigZag, 0);
        // Residuals whose zig-zag value is z: q = 31 is still unary; q = 32 escapes.
        assert_eq!(RiceBlock::cost(&[31u64.unzigzag()], p), 8 + 32);
        assert_eq!(RiceBlock::cost(&[32u64.unzigzag()], p), 8 + 32 + 64);
        assert_eq!(RiceBlock::cost(&[i64::MIN.cast_unsigned()], p), 8 + 32 + 64);
        let p32 = RiceParams { transform: Transform::ZigZag, base: 0u32, scale: 1, k: 0 };
        assert_eq!(RiceBlock::cost(&[15u32.unzigzag()], p32), 7 + 16);
        assert_eq!(RiceBlock::cost(&[16u32.unzigzag()], p32), 7 + 16 + 32);
    }

    #[test]
    fn all_zero_block_is_one_bit_per_value() {
        assert_eq!(encoded_bits(&[0u64; 100]), 8 + 100);
    }

    #[test]
    fn clustered_residuals_choose_offset() {
        let r = signed(&(0..200).map(|i| 10_000 + i % 7).collect::<Vec<_>>());
        let p = RiceBlock::choose(&r);
        assert_eq!((p.transform, p.base), (Transform::Offset, 10_000));
    }

    #[test]
    fn offset_uses_signed_minimum() {
        // A cluster far from zero plus one negative outlier: the minimum is the outlier.
        let mut v: Vec<i64> = (0..200).map(|i| 1_000_000 + i % 3).collect();
        v.push(-5);
        let r = signed(&v);
        let p = RiceBlock::choose(&r);
        assert_eq!(p.transform, Transform::Offset);
        assert_eq!(p.base, (-5i64).cast_unsigned());
    }

    #[test]
    fn cost_with_scale() {
        // multiples of 25: z/25 coded; header gains the 5-bit length-prefixed 25 (7 + 5 bits)
        let r = signed(&[25, -50, 0, 75]); // /25 -> 1, -2, 0, 3 -> zz 2, 3, 0, 6
        let p = RiceParams { transform: Transform::ZigZag, base: 0, scale: 25, k: 0 };
        assert_eq!(RiceBlock::cost(&r, p), 8 + 7 + 5 + (3 + 4 + 1 + 7));
    }

    #[test]
    fn common_divisor_is_used() {
        let r = signed(&(0..200).map(|i| 25 * ((i * 7) % 9 - 4)).collect::<Vec<_>>());
        let p = RiceBlock::choose(&r);
        assert_eq!((p.transform, p.scale), (Transform::ZigZag, 25));
        assert_eq!(round_trip(&r), r);
    }

    #[test]
    fn offset_divisor_is_taken_after_subtracting_base() {
        // 1_000_003 + 10*i: offsets from the minimum are multiples of 10.
        let r = signed(&(0..200).map(|i| 1_000_003 + 10 * (i % 13)).collect::<Vec<_>>());
        let p = RiceBlock::choose(&r);
        assert_eq!((p.transform, p.base, p.scale), (Transform::Offset, 1_000_003, 10));
        assert_eq!(round_trip(&r), r);
    }

    #[test]
    fn scale_round_trips_extremes() {
        let r = [i64::MIN.cast_unsigned(), 0, i64::MIN.cast_unsigned()];
        assert_eq!(RiceBlock::choose(&r).scale, 1 << 63);
        assert_eq!(round_trip(&r), r);
        let r32 = [i32::MIN.cast_unsigned(), 0u32];
        assert_eq!(round_trip(&r32), r32);
    }

    #[test]
    fn all_zero_residuals_have_no_scale() {
        assert_eq!(RiceBlock::choose(&[0u64; 10]).scale, 1);
    }

    #[test]
    fn scale_of_zero_or_one_is_corrupt() {
        for g in [0u64, 1] {
            let mut w = BitWriter::new();
            w.write(0, 1); // zig-zag
            w.write(1, 1); // scaled
            w.write(u64::from(g.bit_width()), 7);
            w.write(g, g.bit_width());
            w.write(0, 64);
            let bytes = w.finish();
            let r = RiceBlock::decode::<u64>(1, &mut BitReader::new(&bytes), &mut Vec::new());
            assert_eq!(r, Err(DecodeError::Corrupt("Rice scale must exceed 1")), "g = {g}");
        }
    }

    #[test]
    fn small_symmetric_residuals_choose_zigzag() {
        let r = signed(&[0, 1, -1, 2, -2, 0, 1, -1]);
        assert_eq!(RiceBlock::choose(&r).transform, Transform::ZigZag);
    }

    #[test]
    fn encoded_size_equals_cost_of_choice() {
        let r = signed(&[3, -7, 1000, 2, 2, 2, -1]);
        assert_eq!(encoded_bits(&r) as u64, RiceBlock::cost(&r, RiceBlock::choose(&r)));
    }

    #[test]
    fn extremes_round_trip() {
        let r = [0, u64::MAX, 1 << 63, (1 << 63) - 1, 1, 0];
        assert_eq!(round_trip(&r), r);
        let r32 = [0, u32::MAX, 1 << 31, 7];
        assert_eq!(round_trip(&r32), r32);
    }

    #[test]
    fn overlong_base_is_corrupt() {
        let mut w = BitWriter::new();
        w.write(1, 1); // offset
        w.write(65, 7); // length > 64
        w.write(0, 64);
        let bytes = w.finish();
        let r = RiceBlock::decode::<u64>(1, &mut BitReader::new(&bytes), &mut Vec::new());
        assert_eq!(r, Err(DecodeError::Corrupt("Rice base longer than the word")));
    }

    #[test]
    fn truncated_input_is_eof() {
        let mut w = BitWriter::new();
        RiceBlock::encode(&signed(&[5, 6, 7, 8]), &mut w);
        let bytes = w.finish();
        let mut out: Vec<u64> = Vec::new();
        let short = &bytes[..bytes.len() - 1];
        assert_eq!(RiceBlock::decode(4, &mut BitReader::new(short), &mut out), Err(DecodeError::UnexpectedEof));
    }

    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    fn brute_min<W: Word>(r: &[W]) -> u64 {
        let mut best = u64::MAX;
        let base = r.iter().copied().min_by_key(|w| w.to_signed_i128()).unwrap_or(W::ZERO);
        let g_zz = r.iter().fold(0, |g, w| gcd(g, u64::try_from(w.to_signed_i128().unsigned_abs()).unwrap()));
        let g_off = r.iter().fold(0, |g, w| gcd(g, w.wrapping_sub(base).to_u64()));
        for k in 0..W::BITS {
            for (transform, base, g) in [(Transform::ZigZag, W::ZERO, g_zz), (Transform::Offset, base, g_off)] {
                for scale in [1, g] {
                    if scale >= 1 {
                        best = best.min(RiceBlock::cost(r, RiceParams { transform, base, scale, k }));
                    }
                }
            }
        }
        best
    }

    proptest! {
        #[test]
        fn round_trip_u64(r in prop::collection::vec(any::<u64>(), 0..200)) {
            prop_assert_eq!(round_trip(&r), r);
        }
        #[test]
        fn round_trip_u32(r in prop::collection::vec(any::<u32>(), 0..200)) {
            prop_assert_eq!(round_trip(&r), r);
        }
        #[test]
        fn round_trip_small(r in prop::collection::vec(-3000i64..3000, 0..200)) {
            let r = signed(&r);
            prop_assert_eq!(round_trip(&r), r);
        }
        #[test]
        fn choice_is_optimal(r in prop::collection::vec(-100_000i64..100_000, 1..100)) {
            let r = signed(&r);
            prop_assert_eq!(RiceBlock::cost(&r, RiceBlock::choose(&r)), brute_min(&r));
        }
        #[test]
        fn choice_is_optimal_scaled(r in prop::collection::vec(-300i64..300, 1..100), g in 1i64..40) {
            let r = signed(&r.iter().map(|x| x * g).collect::<Vec<_>>());
            prop_assert_eq!(RiceBlock::cost(&r, RiceBlock::choose(&r)), brute_min(&r));
            prop_assert_eq!(round_trip(&r), r);
        }
        #[test]
        fn choice_is_optimal_u32(r in prop::collection::vec(any::<u32>(), 1..50)) {
            prop_assert_eq!(RiceBlock::cost(&r, RiceBlock::choose(&r)), brute_min(&r));
        }
    }
}
