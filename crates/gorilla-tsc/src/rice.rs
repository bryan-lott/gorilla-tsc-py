//! Rice coding of a block of residuals, with parameters chosen per block.
//!
//! Block layout: `transform` (1 bit), then for [`Transform::Offset`] the block
//! minimum (a full word), then `k` (log2(BITS) bits), then one code per value:
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

/// Width of the `k` field.
const fn k_bits<W: Word>() -> u32 {
    W::BITS.trailing_zeros()
}

impl<W: Word> RiceParams<W> {
    /// The unsigned value coded for residual `r`.
    fn map(self, r: W) -> u64 {
        match self.transform {
            Transform::ZigZag => r.zigzag().to_u64(),
            Transform::Offset => r.wrapping_sub(self.base).to_u64(),
        }
    }

    /// Inverse of [`RiceParams::map`].
    fn unmap(self, z: u64) -> W {
        let z = W::from_u64_truncate(z);
        match self.transform {
            Transform::ZigZag => z.unzigzag(),
            Transform::Offset => z.wrapping_add(self.base),
        }
    }

    fn header_bits(self) -> u64 {
        1 + u64::from(k_bits::<W>()) + if self.transform == Transform::Offset { u64::from(W::BITS) } else { 0 }
    }
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
            let probe = RiceParams { transform, base, k: 0 };
            let zs: Vec<u64> = residuals.iter().map(|&r| probe.map(r)).collect();
            for k in 0..W::BITS {
                let p = RiceParams { k, ..probe };
                let c = p.header_bits() + mapped_cost::<W>(&zs, k);
                if best.is_none_or(|(b, _)| c < b) {
                    best = Some((c, p));
                }
            }
        }
        best.map_or(RiceParams { transform: Transform::ZigZag, base: W::ZERO, k: 0 }, |b| b.1)
    }

    /// Append `residuals` using the cheapest parameters.
    pub fn encode<W: Word>(residuals: &[W], out: &mut BitWriter) {
        let p = Self::choose(residuals);
        out.write(u64::from(p.transform == Transform::Offset), 1);
        if p.transform == Transform::Offset {
            out.write(p.base.to_u64(), W::BITS);
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
        let base = if transform == Transform::Offset { W::from_u64_truncate(input.read(W::BITS)?) } else { W::ZERO };
        let p = RiceParams { transform, base, k: input.read(k_bits::<W>())? as u32 };
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
        RiceParams { transform: p, base: 0, k }
    }

    #[test]
    fn field_widths() {
        assert_eq!((k_bits::<u64>(), k_bits::<u32>()), (6, 5));
        assert_eq!((escape_q::<u64>(), escape_q::<u32>()), (32, 16));
    }

    #[test]
    fn cost_formula() {
        // header 1 + 6; each value: q ones + 1 + k
        let r = signed(&[0, 1, -1, 5]); // zigzag 0, 2, 1, 10
        assert_eq!(RiceBlock::cost(&r, zz(Transform::ZigZag, 0)), 7 + (1 + 3 + 2 + 11));
        assert_eq!(RiceBlock::cost(&r, zz(Transform::ZigZag, 2)), 7 + 4 * 3 + 2); // q = 0,0,0,2
        // offset adds a full base word
        let p = RiceParams { transform: Transform::Offset, base: 0u64, k: 0 };
        assert_eq!(RiceBlock::cost(&[3u64], p), 7 + 64 + 4);
    }

    #[test]
    fn escape_bounds_outlier_cost() {
        let p = zz(Transform::ZigZag, 0);
        // Residuals whose zig-zag value is z: q = 31 is still unary; q = 32 escapes.
        assert_eq!(RiceBlock::cost(&[31u64.unzigzag()], p), 7 + 32);
        assert_eq!(RiceBlock::cost(&[32u64.unzigzag()], p), 7 + 32 + 64);
        assert_eq!(RiceBlock::cost(&[i64::MIN.cast_unsigned()], p), 7 + 32 + 64);
        let p32 = RiceParams { transform: Transform::ZigZag, base: 0u32, k: 0 };
        assert_eq!(RiceBlock::cost(&[15u32.unzigzag()], p32), 6 + 16);
        assert_eq!(RiceBlock::cost(&[16u32.unzigzag()], p32), 6 + 16 + 32);
    }

    #[test]
    fn all_zero_block_is_one_bit_per_value() {
        assert_eq!(encoded_bits(&[0u64; 100]), 7 + 100);
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
    fn truncated_input_is_eof() {
        let mut w = BitWriter::new();
        RiceBlock::encode(&signed(&[5, 6, 7, 8]), &mut w);
        let bytes = w.finish();
        let mut out: Vec<u64> = Vec::new();
        let short = &bytes[..bytes.len() - 1];
        assert_eq!(RiceBlock::decode(4, &mut BitReader::new(short), &mut out), Err(DecodeError::UnexpectedEof));
    }

    fn brute_min<W: Word>(r: &[W]) -> u64 {
        let mut best = u64::MAX;
        let base = r.iter().copied().min_by_key(|w| w.to_signed_i128()).unwrap_or(W::ZERO);
        for k in 0..W::BITS {
            best = best.min(RiceBlock::cost(r, RiceParams { transform: Transform::ZigZag, base: W::ZERO, k }));
            best = best.min(RiceBlock::cost(r, RiceParams { transform: Transform::Offset, base, k }));
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
        fn choice_is_optimal_u32(r in prop::collection::vec(any::<u32>(), 1..50)) {
            prop_assert_eq!(RiceBlock::cost(&r, RiceBlock::choose(&r)), brute_min(&r));
        }
    }
}
