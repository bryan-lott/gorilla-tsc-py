//! Adaptive (context-modelled, range-coded) coding of a block of residuals.
//!
//! Block layout: scale flag (1 bit) and, if set, a length-prefixed common
//! divisor `g > 1`; then a range-coded section. Each residual `s = r / g` is
//! binarised as: zero flag; sign; bit length `L` of `|s|` in unary; the two
//! bits below the leading one with their own contexts; the rest as direct
//! bits. Contexts depend on the magnitude class of the previous residual.
//! Models start fresh in every block.

use crate::range::{Prob, RangeDecoder, RangeEncoder};
use crate::rice::{read_prefixed, signed_divisor, write_prefixed};
use crate::{BitReader, BitWriter, DecodeError, Word};

/// Magnitude classes of the previous residual used as context.
const CLASSES: usize = 4;
/// Mantissa bits below the leading one that get modelled contexts.
const MODELLED_MANTISSA: u32 = 2;

/// Adaptive residual coder.
#[derive(Debug, Default, Clone, Copy)]
pub struct AdaptiveBlock;

struct Model {
    zero: [Prob; CLASSES],
    sign: [Prob; CLASSES],
    len: [[Prob; 65]; CLASSES],
    mantissa: [[Prob; 3]; 65],
}

impl Model {
    fn new() -> Self {
        Self {
            zero: [Prob::default(); CLASSES],
            sign: [Prob::default(); CLASSES],
            len: [[Prob::default(); 65]; CLASSES],
            mantissa: [[Prob::default(); 3]; 65],
        }
    }
}

/// Context class of a residual magnitude.
fn class(mag: u64) -> usize {
    match mag {
        0 => 0,
        1..=3 => 1,
        4..=63 => 2,
        _ => 3,
    }
}

/// `|s|` and its bit length, for a signed residual.
fn magnitude(s: i128) -> (u64, u32) {
    let mag = u64::try_from(s.unsigned_abs()).unwrap_or(u64::MAX);
    (mag, u64::BITS - mag.leading_zeros())
}

impl AdaptiveBlock {
    /// Append `residuals`.
    pub fn encode<W: Word>(residuals: &[W], out: &mut BitWriter) {
        let scale = signed_divisor(residuals).max(1);
        out.write(u64::from(scale > 1), 1);
        if scale > 1 {
            write_prefixed::<W>(out, scale);
        }
        let mut model = Model::new();
        let mut coder = RangeEncoder::new(out);
        let mut ctx_class = 0;
        for &residual in residuals {
            let value =
                if scale == 1 { residual.to_signed_i128() } else { residual.to_signed_i128() / i128::from(scale) };
            let (mag, len) = magnitude(value);
            coder.bit(&mut model.zero[ctx_class], mag != 0);
            if mag != 0 {
                coder.bit(&mut model.sign[ctx_class], value.is_negative());
                for i in 1..len {
                    coder.bit(&mut model.len[ctx_class][i as usize], true);
                }
                if len < W::BITS {
                    coder.bit(&mut model.len[ctx_class][len as usize], false);
                }
                let below = len - 1;
                let modelled = below.min(MODELLED_MANTISSA);
                let mut ctx = 0;
                for j in 0..modelled {
                    let bit = (mag >> (below - 1 - j)) & 1 == 1;
                    coder.bit(&mut model.mantissa[len as usize][ctx], bit);
                    ctx = 1 + usize::from(bit);
                }
                let rest = below - modelled;
                coder.direct(mag & ((1u64 << rest) - 1), rest);
            }
            ctx_class = class(mag);
        }
        coder.finish();
    }

    /// Read `n` residuals, appending them to `out`.
    ///
    /// # Errors
    /// On truncated or impossible input.
    pub fn decode<W: Word>(n: usize, input: &mut BitReader<'_>, out: &mut Vec<W>) -> Result<(), DecodeError> {
        let scale = if input.read_bit()? {
            let scale = read_prefixed::<W>(input, "scale longer than the word")?;
            if scale <= 1 {
                return Err(DecodeError::Corrupt("scale must exceed 1"));
            }
            scale
        } else {
            1
        };
        let mut model = Model::new();
        let mut coder = RangeDecoder::new(input)?;
        let mut ctx_class = 0;
        for _ in 0..n {
            let mut mag = 0u64;
            let mut negative = false;
            if coder.bit(&mut model.zero[ctx_class])? {
                negative = coder.bit(&mut model.sign[ctx_class])?;
                let mut len = 1;
                while len < W::BITS && coder.bit(&mut model.len[ctx_class][len as usize])? {
                    len += 1;
                }
                let below = len - 1;
                let modelled = below.min(MODELLED_MANTISSA);
                mag = 1;
                let mut ctx = 0;
                for _ in 0..modelled {
                    let bit = coder.bit(&mut model.mantissa[len as usize][ctx])?;
                    mag = (mag << 1) | u64::from(bit);
                    ctx = 1 + usize::from(bit);
                }
                let rest = below - modelled;
                mag = (mag << rest) | coder.direct(rest)?;
            }
            let value = if negative { -i128::from(mag) } else { i128::from(mag) };
            out.push(W::from_i128_truncate(if scale == 1 { value } else { value.wrapping_mul(i128::from(scale)) }));
            ctx_class = class(mag);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RiceBlock;
    use proptest::prelude::*;

    fn signed(v: &[i64]) -> Vec<u64> {
        v.iter().map(|x| x.cast_unsigned()).collect()
    }

    fn bits<W: Word>(r: &[W]) -> usize {
        let mut w = BitWriter::new();
        AdaptiveBlock::encode(r, &mut w);
        w.bit_len()
    }

    fn round_trip<W: Word>(r: &[W]) -> Vec<W> {
        let mut w = BitWriter::new();
        AdaptiveBlock::encode(r, &mut w);
        w.write(0b11, 2);
        let bytes = w.finish();
        let mut input = BitReader::new(&bytes);
        let mut out = Vec::new();
        AdaptiveBlock::decode(r.len(), &mut input, &mut out).unwrap();
        assert_eq!(input.read(2), Ok(0b11), "decoder must stop where the encoder did");
        out
    }

    /// Deterministic two-sided geometric residuals (like real deltas).
    fn laplacian(n: usize, spread: u32) -> Vec<i64> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let mag = i64::from((s >> 40).trailing_zeros().min(30)) * i64::from(spread)
                    + i64::try_from(s % u64::from(spread)).unwrap();
                if s & 1 == 0 { mag } else { -mag }
            })
            .collect()
    }

    #[test]
    fn classes() {
        assert_eq!([0, 1, 3, 4, 63, 64, u64::MAX].map(class), [0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn mostly_zero_residuals_cost_well_under_a_bit() {
        let r: Vec<i64> = (0..1000).map(|i| if i % 50 == 7 { 3 } else { 0 }).collect();
        let b = bits(&signed(&r));
        // Rice cannot go below one bit per value (>= 1008 bits here).
        assert!(b < 1000 * 3 / 10, "{b} bits");
        assert!(b * 3 < usize::try_from(RiceBlock::cost(&signed(&r), RiceBlock::choose(&signed(&r)))).unwrap());
        assert_eq!(round_trip(&signed(&r)), signed(&r));
    }

    #[test]
    fn beats_rice_on_two_sided_geometric_residuals() {
        let r = signed(&laplacian(1000, 3));
        let rice = RiceBlock::cost(&r, RiceBlock::choose(&r));
        assert!((bits(&r) as u64) < rice, "adaptive {} vs rice {rice}", bits(&r));
        assert_eq!(round_trip(&r), r);
    }

    #[test]
    fn common_divisor_costs_only_its_header() {
        let base = signed(&laplacian(500, 5));
        let scaled: Vec<u64> = base.iter().map(|&x| x.cast_signed().wrapping_mul(25).cast_unsigned()).collect();
        // flag + 7-bit length + 5 bits for 25
        assert_eq!(bits(&scaled), bits(&base) + 7 + 5);
        assert_eq!(round_trip(&scaled), scaled);
    }

    #[test]
    fn extremes_round_trip() {
        let r = [0, u64::MAX, 1 << 63, (1 << 63) - 1, 1, 2, 3, 4, 0];
        assert_eq!(round_trip(&r), r);
        let r32 = [0, u32::MAX, 1 << 31, (1 << 31) - 1, 5];
        assert_eq!(round_trip(&r32), r32);
        let all_min = [1u64 << 63; 3];
        assert_eq!(round_trip(&all_min), all_min);
    }

    #[test]
    fn empty_block() {
        assert_eq!(round_trip::<u64>(&[]), Vec::<u64>::new());
    }

    #[test]
    fn bad_scale_is_corrupt() {
        let mut w = BitWriter::new();
        w.write(1, 1);
        w.write(1, 7);
        w.write(1, 1); // g = 1
        w.write(0, 64);
        let bytes = w.finish();
        let r = AdaptiveBlock::decode::<u64>(1, &mut BitReader::new(&bytes), &mut Vec::new());
        assert_eq!(r, Err(DecodeError::Corrupt("scale must exceed 1")));
    }

    #[test]
    fn truncated_is_eof() {
        let r = signed(&laplacian(200, 7));
        let mut w = BitWriter::new();
        AdaptiveBlock::encode(&r, &mut w);
        let bytes = w.finish();
        let res =
            AdaptiveBlock::decode::<u64>(r.len(), &mut BitReader::new(&bytes[..bytes.len() / 2]), &mut Vec::new());
        assert_eq!(res, Err(DecodeError::UnexpectedEof));
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
        fn round_trip_small(r in prop::collection::vec(-40i64..40, 0..300), g in 1i64..30) {
            let r = signed(&r.iter().map(|x| x * g).collect::<Vec<_>>());
            prop_assert_eq!(round_trip(&r), r);
        }
    }
}
