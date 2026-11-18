//! Residual coders: turn a stream of residual words into bits and back.
//!
//! A coder's state is its `Default`; a fresh coder is used for every block, so
//! no state can leak across a checkpoint.

use crate::{BitReader, BitWriter, DecodeError, Word};

/// Writes and reads one residual at a time.
pub trait ResidualCoder<W: Word>: Default {
    /// Append `residual` to `out`.
    fn encode(&mut self, out: &mut BitWriter, residual: W);
    /// Read the next residual.
    ///
    /// # Errors
    /// On truncated or impossible input.
    fn decode(&mut self, input: &mut BitReader<'_>) -> Result<W, DecodeError>;
}

/// Zig-zags the residual and stores it in the smallest of a few size buckets.
/// Suited to arithmetic residuals (delta-of-delta), which cluster around zero.
#[derive(Debug, Default, Clone, Copy)]
pub struct BucketCoder;

/// Gorilla's XOR scheme: store only the "meaningful" bits between the leading
/// and trailing zeros, reusing the previous window when the new bits fit in it.
#[derive(Debug, Default, Clone, Copy)]
pub struct XorCoder {
    window: Option<Window>,
}

/// Position of the meaningful bits of the last XOR header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Window {
    leading: u32,
    trailing: u32,
}

/// `(prefix, prefix bits, payload bits)` for each bounded bucket, smallest first.
const BUCKETS: [(u64, u32, u32); 3] = [(0b10, 2, 7), (0b110, 3, 9), (0b1110, 4, 12)];
/// Prefix of the escape bucket, whose payload is a full word.
const ESCAPE: (u64, u32) = (0b1111, 4);

impl<W: Word> ResidualCoder<W> for BucketCoder {
    fn encode(&mut self, out: &mut BitWriter, residual: W) {
        let zz = residual.zigzag().to_u64();
        if zz == 0 {
            out.write(0, 1);
            return;
        }
        for (prefix, prefix_bits, payload_bits) in BUCKETS {
            if zz >> payload_bits == 0 {
                out.write(prefix, prefix_bits);
                out.write(zz, payload_bits);
                return;
            }
        }
        out.write(ESCAPE.0, ESCAPE.1);
        out.write(zz, W::BITS);
    }

    fn decode(&mut self, input: &mut BitReader<'_>) -> Result<W, DecodeError> {
        if !input.read_bit()? {
            return Ok(W::ZERO);
        }
        let mut payload_bits = W::BITS;
        for (_, _, bits) in BUCKETS {
            if !input.read_bit()? {
                payload_bits = bits;
                break;
            }
        }
        Ok(W::from_u64_truncate(input.read(payload_bits)?).unzigzag())
    }
}

impl XorCoder {
    /// Width of the leading-zeros field; counts above its maximum are capped.
    const fn lz_bits<W: Word>() -> u32 {
        W::BITS.trailing_zeros() - 1
    }
    /// Width of the length field, which stores `len - 1` (`len` is 1..=BITS).
    const fn len_bits<W: Word>() -> u32 {
        W::BITS.trailing_zeros()
    }
}

impl<W: Word> ResidualCoder<W> for XorCoder {
    fn encode(&mut self, out: &mut BitWriter, residual: W) {
        if residual == W::ZERO {
            out.write(0, 1);
            return;
        }
        out.write(1, 1);
        let bits = residual.to_u64();
        let leading = residual.leading_zeros().min((1 << Self::lz_bits::<W>()) - 1);
        let trailing = residual.trailing_zeros();
        match self.window {
            Some(w) if leading >= w.leading && trailing >= w.trailing => {
                out.write(0, 1);
                out.write(bits >> w.trailing, W::BITS - w.leading - w.trailing);
            }
            _ => {
                let len = W::BITS - leading - trailing;
                out.write(1, 1);
                out.write(u64::from(leading), Self::lz_bits::<W>());
                out.write(u64::from(len - 1), Self::len_bits::<W>());
                out.write(bits >> trailing, len);
                self.window = Some(Window { leading, trailing });
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)] // header fields are at most 6 bits wide
    fn decode(&mut self, input: &mut BitReader<'_>) -> Result<W, DecodeError> {
        if !input.read_bit()? {
            return Ok(W::ZERO);
        }
        let window = if input.read_bit()? {
            let leading = input.read(Self::lz_bits::<W>())? as u32;
            let len = input.read(Self::len_bits::<W>())? as u32 + 1;
            let trailing =
                W::BITS.checked_sub(leading + len).ok_or(DecodeError::Corrupt("XOR window wider than the word"))?;
            let w = Window { leading, trailing };
            self.window = Some(w);
            w
        } else {
            self.window.ok_or(DecodeError::Corrupt("XOR window reused before being set"))?
        };
        let len = W::BITS - window.leading - window.trailing;
        Ok(W::from_u64_truncate(input.read(len)? << window.trailing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn cost<W: Word, C: ResidualCoder<W>>(residuals: &[W]) -> usize {
        let mut c = C::default();
        let mut w = BitWriter::new();
        for &r in residuals {
            c.encode(&mut w, r);
        }
        w.bit_len()
    }

    fn round_trip<W: Word, C: ResidualCoder<W>>(residuals: &[W]) -> Vec<W> {
        let mut c = C::default();
        let mut w = BitWriter::new();
        for &r in residuals {
            c.encode(&mut w, r);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let mut d = C::default();
        residuals.iter().map(|_| d.decode(&mut r).unwrap()).collect()
    }

    fn signed(n: i64) -> u64 {
        n.cast_unsigned()
    }

    #[test]
    fn bucket_costs_u64() {
        let cases: [(i64, usize); 14] = [
            (0, 1),
            (-1, 9),
            (1, 9),
            (-64, 9),
            (63, 9),
            (-65, 12),
            (64, 12),
            (-256, 12),
            (255, 12),
            (-257, 16),
            (256, 16),
            (-2048, 16),
            (2047, 16),
            (2048, 4 + 64),
        ];
        for (n, bits) in cases {
            assert_eq!(cost::<u64, BucketCoder>(&[signed(n)]), bits, "residual {n}");
        }
        assert_eq!(cost::<u64, BucketCoder>(&[signed(-2049)]), 4 + 64);
    }

    #[test]
    fn bucket_escape_is_word_width() {
        assert_eq!(cost::<u32, BucketCoder>(&[i32::MAX.cast_unsigned()]), 4 + 32);
        assert_eq!(cost::<u32, BucketCoder>(&[u32::MAX]), 9); // -1
    }

    #[test]
    fn bucket_round_trips_extremes() {
        let v = [0, 1, u64::MAX, i64::MIN.cast_unsigned(), i64::MAX.cast_unsigned(), 2048, 4095];
        assert_eq!(round_trip::<u64, BucketCoder>(&v), v);
        let v32 = [0u32, u32::MAX, 1 << 31, 12345];
        assert_eq!(round_trip::<u32, BucketCoder>(&v32), v32);
    }

    #[test]
    fn xor_zero_is_one_bit() {
        assert_eq!(cost::<u64, XorCoder>(&[0, 0, 0]), 3);
    }

    #[test]
    fn xor_header_field_widths() {
        // u64: 1<<10 has 53 leading zeros (capped at 31) and 10 trailing: len 64-31-10 = 23.
        assert_eq!(cost::<u64, XorCoder>(&[1 << 10]), 2 + 5 + 6 + 23);
        // u32: 1<<3 has 28 leading zeros (capped at 15) and 3 trailing: len 32-15-3 = 14.
        assert_eq!(cost::<u32, XorCoder>(&[1 << 3]), 2 + 4 + 5 + 14);
        // Full-width meaningful bits fit the length field (stored as len - 1).
        assert_eq!(cost::<u64, XorCoder>(&[u64::MAX]), 2 + 5 + 6 + 64);
        assert_eq!(cost::<u32, XorCoder>(&[u32::MAX]), 2 + 4 + 5 + 32);
    }

    #[test]
    fn xor_reuses_window_when_bits_fit() {
        // First: lz 56 -> 31, tz 4, len 29. Second & third fit inside that window.
        let r = [0b1111u64 << 4, 1 << 4, 1 << 7];
        assert_eq!(cost::<u64, XorCoder>(&r), (2 + 5 + 6 + 29) + 2 * (2 + 29));
    }

    #[test]
    fn xor_equal_bounds_count_as_fitting() {
        assert_eq!(cost::<u64, XorCoder>(&[1 << 4, 1 << 4]), (2 + 5 + 6 + 29) + (2 + 29));
    }

    #[test]
    fn xor_new_header_when_bits_escape_window() {
        // Second has fewer trailing zeros, so the window cannot be reused.
        assert_eq!(cost::<u64, XorCoder>(&[1 << 4, 1 << 3]), (2 + 5 + 6 + 29) + (2 + 5 + 6 + 30));
        // Fewer leading zeros than the window allows.
        assert_eq!(cost::<u64, XorCoder>(&[1 << 4, 1 << 40]), (2 + 5 + 6 + 29) + (2 + 5 + 6 + 1));
    }

    #[test]
    fn xor_round_trips_windows() {
        let v = [1u64 << 4, 1 << 4, 1 << 3, 1 << 63, 1, u64::MAX, 0, 0xF0, 0xF00];
        assert_eq!(round_trip::<u64, XorCoder>(&v), v);
    }

    #[test]
    fn xor_reuse_before_header_is_corrupt() {
        // '1' (nonzero) then '0' (reuse window) with no window yet.
        let bytes = [0b1000_0000];
        let mut d = XorCoder::default();
        let got: Result<u64, _> = d.decode(&mut BitReader::new(&bytes));
        assert_eq!(got, Err(DecodeError::Corrupt("XOR window reused before being set")));
    }

    #[test]
    fn xor_header_with_impossible_window_is_corrupt() {
        // u64 header: lz=31, len-1=63 -> 31 + 64 > 64.
        let mut w = BitWriter::new();
        w.write(0b11, 2);
        w.write(31, 5);
        w.write(63, 6);
        w.write(0, 64);
        let bytes = w.finish();
        let got: Result<u64, _> = XorCoder::default().decode(&mut BitReader::new(&bytes));
        assert_eq!(got, Err(DecodeError::Corrupt("XOR window wider than the word")));
    }

    proptest! {
        #[test]
        fn bucket_round_trip_u64(v in prop::collection::vec(any::<u64>(), 0..100)) {
            prop_assert_eq!(round_trip::<u64, BucketCoder>(&v), v);
        }
        #[test]
        fn bucket_round_trip_small(v in prop::collection::vec(-5000i64..5000, 0..100)) {
            let v: Vec<u64> = v.into_iter().map(signed).collect();
            prop_assert_eq!(round_trip::<u64, BucketCoder>(&v), v);
        }
        #[test]
        fn xor_round_trip_u64(v in prop::collection::vec(any::<u64>(), 0..100)) {
            prop_assert_eq!(round_trip::<u64, XorCoder>(&v), v);
        }
        #[test]
        fn xor_round_trip_u32(v in prop::collection::vec(any::<u32>(), 0..100)) {
            prop_assert_eq!(round_trip::<u32, XorCoder>(&v), v);
        }
        #[test]
        fn xor_round_trip_sparse(v in prop::collection::vec((0u32..64, 0u64..16), 0..100)) {
            let v: Vec<u64> = v.into_iter().map(|(s, b)| b << s).collect();
            prop_assert_eq!(round_trip::<u64, XorCoder>(&v), v);
        }
    }
}
