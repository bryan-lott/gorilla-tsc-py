//! Delta coding with Rice-coded residuals, the difference order chosen per block.
//!
//! Block layout: first item verbatim; then, if there is more than one item, a
//! 2-bit order (`0` = first differences, `1` = second differences, `2` = the
//! values themselves) and a [`RiceBlock`] of the residuals.

use crate::scheme::{Op, Sub};
use crate::{BitReader, BitWriter, BlockCodec, DecodeError, RiceBlock, Word};

/// Delta / delta-of-delta block codec.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeltaRice;

/// Which differences are Rice coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// `x[i] - x[i-1]`.
    First,
    /// `d[i] - d[i-1]`, with `d[0]` predicted as zero.
    Second,
    /// `x[i]` itself (frame of reference via the Rice offset transform).
    Zero,
}

/// Width of the order field.
const ORDER_BITS: u32 = 2;

impl Order {
    /// All orders, in tie-break preference.
    const ALL: [Self; 3] = [Self::First, Self::Second, Self::Zero];

    const fn code(self) -> u64 {
        match self {
            Self::First => 0,
            Self::Second => 1,
            Self::Zero => 2,
        }
    }

    const fn from_code(c: u64) -> Option<Self> {
        match c {
            0 => Some(Self::First),
            1 => Some(Self::Second),
            2 => Some(Self::Zero),
            _ => None,
        }
    }
}

/// Residuals of `items[1..]` under `order`.
#[must_use]
pub fn residuals<W: Word>(items: &[W], order: Order) -> Vec<W> {
    let mut prev_d = W::ZERO;
    items
        .windows(2)
        .map(|w| {
            let d = Sub::diff(w[0], w[1]);
            match order {
                Order::Zero => w[1],
                Order::First => d,
                Order::Second => {
                    let r = Sub::diff(prev_d, d);
                    prev_d = d;
                    r
                }
            }
        })
        .collect()
}

/// Inverse of [`residuals`]: rebuild items from the first item and residuals,
/// appending all but `first` to `out`.
pub fn integrate<W: Word>(first: W, residuals: &[W], order: Order, out: &mut Vec<W>) {
    let (mut x, mut d) = (first, W::ZERO);
    for &r in residuals {
        x = match order {
            Order::Zero => r,
            Order::First => Sub::undiff(x, r),
            Order::Second => {
                d = Sub::undiff(d, r);
                Sub::undiff(x, d)
            }
        };
        out.push(x);
    }
}

impl<W: Word> BlockCodec<W> for DeltaRice {
    fn encode_block(&self, items: &[W], out: &mut BitWriter) {
        let Some((&first, rest)) = items.split_first() else { return };
        out.write(first.to_u64(), W::BITS);
        if rest.is_empty() {
            return;
        }
        let cost = |r: &[W]| RiceBlock::cost(r, RiceBlock::choose(r));
        let (order, r) = Order::ALL
            .into_iter()
            .map(|o| (o, residuals(items, o)))
            .min_by_key(|(_, r)| cost(r))
            .unwrap_or((Order::First, Vec::new()));
        out.write(order.code(), ORDER_BITS);
        RiceBlock::encode(&r, out);
    }

    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<W>) -> Result<(), DecodeError> {
        if n == 0 {
            return Ok(());
        }
        let first = W::from_u64_truncate(input.read(W::BITS)?);
        out.push(first);
        if n == 1 {
            return Ok(());
        }
        let order = Order::from_code(input.read(ORDER_BITS)?).ok_or(DecodeError::Corrupt("unknown delta order"))?;
        let mut r = Vec::with_capacity(n - 1);
        RiceBlock::decode(n - 1, input, &mut r)?;
        integrate(first, &r, order, out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bits<W: Word>(items: &[W]) -> usize {
        let mut w = BitWriter::new();
        DeltaRice.encode_block(items, &mut w);
        w.bit_len()
    }

    fn round_trip<W: Word>(items: &[W]) -> Vec<W> {
        let mut w = BitWriter::new();
        DeltaRice.encode_block(items, &mut w);
        let bytes = w.finish();
        let mut out = Vec::new();
        DeltaRice.decode_block(items.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        out
    }

    fn best(r: &[u64]) -> u64 {
        RiceBlock::cost(r, RiceBlock::choose(r))
    }

    #[test]
    fn residual_orders() {
        let x = [10u64, 13, 19, 19];
        assert_eq!(residuals(&x, Order::First), vec![3, 6, 0]);
        assert_eq!(residuals(&x, Order::Second), vec![3, 3, 0u64.wrapping_sub(6)]);
        assert_eq!(residuals(&x, Order::Zero), vec![13, 19, 19]);
        assert_eq!(residuals(&[7u64], Order::First), Vec::<u64>::new());
    }

    #[test]
    fn integrate_inverts_residuals() {
        let x = [10u64, 13, 19, 19, 0, u64::MAX];
        for order in [Order::First, Order::Second, Order::Zero] {
            let mut out = vec![x[0]];
            integrate(x[0], &residuals(&x, order), order, &mut out);
            assert_eq!(out, x);
        }
    }

    #[test]
    fn single_item_is_verbatim() {
        assert_eq!(bits(&[42u64]), 64);
        assert_eq!(bits(&[42u32]), 32);
    }

    #[test]
    fn size_is_cheaper_order_plus_overhead() {
        let cases: [&[u64]; 3] = [&[0, 5, 10, 15, 20, 25, 30], &[1, 1, 2, 1, 1, 2, 1], &[100, 90, 500, 7, 7, 7]];
        for x in cases {
            let costs = [Order::First, Order::Second, Order::Zero].map(|o| best(&residuals(x, o)));
            assert_eq!(bits(x) as u64, 64 + 2 + costs.into_iter().min().unwrap(), "{x:?}");
        }
    }

    #[test]
    fn linear_series_costs_about_one_bit_per_value() {
        let x: Vec<u64> = (0..100).map(|i| 1000 + 7 * i).collect();
        let costs = [Order::First, Order::Second, Order::Zero].map(|o| best(&residuals(&x, o)));
        assert_eq!(bits(&x) as u64, 64 + 2 + costs.into_iter().min().unwrap());
        // 99 residuals at ~1 bit each plus a small Rice header.
        assert!(bits(&x) <= 64 + 2 + 99 + 30, "{} bits", bits(&x));
    }

    #[test]
    fn jittery_series_uses_first_order() {
        let x: Vec<u64> = (0..100u64).map(|i| 10_000 * i + (i * 7919) % 11).collect();
        let one = best(&residuals(&x, Order::First));
        assert!(one < best(&residuals(&x, Order::Second)));
        assert_eq!(bits(&x) as u64, 64 + 2 + one);
    }

    #[test]
    fn unordered_values_use_order_zero() {
        // A shuffled narrow range: values are cheap, their differences are not.
        // Alternating levels: values span 60, but differences swing by +/-50.
        let x: Vec<u64> = (0..200u64).map(|i| 5000 + 50 * (i % 2) + (i * i) % 7).collect();
        let zero = best(&residuals(&x, Order::Zero));
        assert!(zero < best(&residuals(&x, Order::First)));
        assert!(zero < best(&residuals(&x, Order::Second)));
        assert_eq!(bits(&x) as u64, 64 + 2 + zero);
    }

    #[test]
    fn unknown_order_is_corrupt() {
        let mut w = BitWriter::new();
        w.write(1, 64);
        w.write(3, 2);
        w.write(0, 64);
        let bytes = w.finish();
        let r = DeltaRice.decode_block(2, &mut BitReader::new(&bytes), &mut Vec::<u64>::new());
        assert_eq!(r, Err(DecodeError::Corrupt("unknown delta order")));
    }

    #[test]
    fn truncated_is_eof() {
        let mut w = BitWriter::new();
        DeltaRice.encode_block(&[1u64, 2, 3], &mut w);
        let bytes = w.finish();
        let mut out = Vec::<u64>::new();
        assert_eq!(DeltaRice.decode_block(3, &mut BitReader::new(&bytes[..4]), &mut out), Err(DecodeError::UnexpectedEof));
    }

    proptest! {
        #[test]
        fn round_trip_u64(x in prop::collection::vec(any::<u64>(), 1..200)) {
            prop_assert_eq!(round_trip(&x), x);
        }
        #[test]
        fn round_trip_u32(x in prop::collection::vec(any::<u32>(), 1..200)) {
            prop_assert_eq!(round_trip(&x), x);
        }
        #[test]
        fn round_trip_smooth(start in any::<i64>(), steps in prop::collection::vec(-50i64..50, 0..200)) {
            let mut x = vec![start.cast_unsigned()];
            for s in steps { x.push(x[x.len() - 1].wrapping_add(s.cast_unsigned())); }
            prop_assert_eq!(round_trip(&x), x);
        }
    }
}
