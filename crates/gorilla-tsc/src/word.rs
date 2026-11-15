//! Fixed-width unsigned machine words the codec operates on.

/// An unsigned machine word. All arithmetic is wrapping, so a difference of
/// two words always fits in a word.
pub trait Word:
    Copy + Eq + Default + core::fmt::Debug + core::ops::BitXor<Output = Self> + Send + Sync + 'static
{
    /// Width in bits.
    const BITS: u32;
    /// Zero.
    const ZERO: Self;

    /// Widen to `u64` (lossless).
    fn to_u64(self) -> u64;
    /// Narrow from `u64`, keeping the low `BITS` bits.
    fn from_u64_truncate(v: u64) -> Self;
    /// `self - rhs` modulo 2^BITS.
    #[must_use]
    fn wrapping_sub(self, rhs: Self) -> Self;
    /// `self + rhs` modulo 2^BITS.
    #[must_use]
    fn wrapping_add(self, rhs: Self) -> Self;
    /// Leading zero bits.
    fn leading_zeros(self) -> u32;
    /// Trailing zero bits.
    fn trailing_zeros(self) -> u32;
    /// Zig-zag map of the two's-complement reading of `self`: small magnitudes
    /// of either sign become small unsigned values (0, -1, 1, -2 → 0, 1, 2, 3).
    #[must_use]
    fn zigzag(self) -> Self;
    /// Inverse of [`Word::zigzag`].
    #[must_use]
    fn unzigzag(self) -> Self;
    /// The two's-complement (signed) reading of `self`, widened.
    fn to_signed_i128(self) -> i128;
}

macro_rules! impl_word {
    ($u:ty, $i:ty) => {
        impl Word for $u {
            const BITS: u32 = <$u>::BITS;
            const ZERO: Self = 0;

            fn to_u64(self) -> u64 {
                u64::from(self)
            }
            #[allow(clippy::cast_possible_truncation)] // truncation is the contract
            fn from_u64_truncate(v: u64) -> Self {
                v as $u
            }
            fn wrapping_sub(self, rhs: Self) -> Self {
                <$u>::wrapping_sub(self, rhs)
            }
            fn wrapping_add(self, rhs: Self) -> Self {
                <$u>::wrapping_add(self, rhs)
            }
            fn leading_zeros(self) -> u32 {
                <$u>::leading_zeros(self)
            }
            fn trailing_zeros(self) -> u32 {
                <$u>::trailing_zeros(self)
            }
            fn zigzag(self) -> Self {
                let s = self.cast_signed();
                ((s << 1) ^ (s >> (<$u>::BITS - 1))).cast_unsigned()
            }
            fn unzigzag(self) -> Self {
                (self >> 1) ^ (self & 1).wrapping_neg()
            }
            fn to_signed_i128(self) -> i128 {
                i128::from(self.cast_signed())
            }
        }
    };
}

impl_word!(u32, i32);
impl_word!(u64, i64);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn zigzag_known_values_u64() {
        let cases: [(i64, u64); 7] =
            [(0, 0), (-1, 1), (1, 2), (-2, 3), (2, 4), (i64::MAX, u64::MAX - 1), (i64::MIN, u64::MAX)];
        for (signed, zz) in cases {
            let w = signed.cast_unsigned();
            assert_eq!(w.zigzag(), zz, "zigzag({signed})");
            assert_eq!(zz.unzigzag(), w, "unzigzag({zz})");
        }
    }

    #[test]
    fn zigzag_known_values_u32() {
        assert_eq!((-1i32).cast_unsigned().zigzag(), 1u32);
        assert_eq!(i32::MIN.cast_unsigned().zigzag(), u32::MAX);
        assert_eq!(i32::MAX.cast_unsigned().zigzag(), u32::MAX - 1);
    }

    #[test]
    fn widening_and_truncation() {
        assert_eq!(u32::MAX.to_u64(), 0xFFFF_FFFF);
        assert_eq!(u32::from_u64_truncate(0x1_2345_6789), 0x2345_6789);
        assert_eq!(u64::from_u64_truncate(u64::MAX), u64::MAX);
    }

    #[test]
    fn wrapping_arithmetic() {
        assert_eq!(Word::wrapping_sub(0u32, 1), u32::MAX);
        assert_eq!(Word::wrapping_add(u64::MAX, 2), 1);
    }

    #[test]
    fn signed_reading() {
        assert_eq!(u64::MAX.to_signed_i128(), -1);
        assert_eq!((1u64 << 63).to_signed_i128(), i128::from(i64::MIN));
        assert_eq!(5u64.to_signed_i128(), 5);
        assert_eq!(u32::MAX.to_signed_i128(), -1);
        assert_eq!(0x7FFF_FFFFu32.to_signed_i128(), i128::from(i32::MAX));
    }

    #[test]
    fn zero_counts() {
        assert_eq!(Word::leading_zeros(1u64), 63);
        assert_eq!(Word::leading_zeros(1u32), 31);
        assert_eq!(Word::trailing_zeros(8u32), 3);
        assert_eq!(Word::trailing_zeros(0u64), 64);
    }

    proptest! {
        #[test]
        fn zigzag_round_trips_u64(w: u64) { prop_assert_eq!(w.zigzag().unzigzag(), w); }
        #[test]
        fn zigzag_round_trips_u32(w: u32) { prop_assert_eq!(w.zigzag().unzigzag(), w); }
        #[test]
        fn zigzag_is_magnitude_ordered(n in -1000i64..1000) {
            let zz = n.cast_unsigned().zigzag();
            prop_assert_eq!(zz, if n < 0 { n.unsigned_abs() * 2 - 1 } else { n.unsigned_abs() * 2 });
        }
    }
}
