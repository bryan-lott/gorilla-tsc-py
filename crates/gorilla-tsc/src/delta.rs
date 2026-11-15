//! Delta coding with Rice-coded residuals, the difference order chosen per block.
//!
//! Block layout: first item verbatim; then, if there is more than one item, an
//! order bit (`0` = first differences, `1` = second differences) and a
//! [`RiceBlock`] of the residuals.

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
        d = match order {
            Order::First => r,
            Order::Second => Sub::undiff(d, r),
        };
        x = Sub::undiff(x, d);
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
        let first_order = residuals(items, Order::First);
        let second_order = residuals(items, Order::Second);
        let cost = |r: &[W]| RiceBlock::cost(r, RiceBlock::choose(r));
        let (order, r) = if cost(&second_order) < cost(&first_order) {
            (Order::Second, second_order)
        } else {
            (Order::First, first_order)
        };
        out.write(u64::from(order == Order::Second), 1);
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
        let order = if input.read_bit()? { Order::Second } else { Order::First };
        let mut r = Vec::with_capacity(n - 1);
        RiceBlock::decode(n - 1, input, &mut r)?;
        integrate(first, &r, order, out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RiceParams;
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
        assert_eq!(residuals(&[7u64], Order::First), Vec::<u64>::new());
    }

    #[test]
    fn integrate_inverts_residuals() {
        let x = [10u64, 13, 19, 19, 0, u64::MAX];
        for order in [Order::First, Order::Second] {
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
            let one = best(&residuals(x, Order::First));
            let two = best(&residuals(x, Order::Second));
            assert_eq!(bits(x) as u64, 64 + 1 + one.min(two), "{x:?}");
        }
    }

    #[test]
    fn linear_series_uses_second_order() {
        let x: Vec<u64> = (0..100).map(|i| 1000 + 7 * i).collect();
        // second differences: 7 then zeros -> zig-zag k=0 except one value
        let r = residuals(&x, Order::Second);
        assert_eq!(RiceBlock::choose(&r), RiceParams { transform: crate::Transform::ZigZag, base: 0, k: 0 });
        assert_eq!(bits(&x) as u64, 64 + 1 + best(&r));
        assert!(best(&r) < best(&residuals(&x, Order::First)));
    }

    #[test]
    fn jittery_series_uses_first_order() {
        let x: Vec<u64> = (0..100u64).map(|i| 10_000 * i + (i * 7919) % 11).collect();
        let one = best(&residuals(&x, Order::First));
        assert!(one < best(&residuals(&x, Order::Second)));
        assert_eq!(bits(&x) as u64, 64 + 1 + one);
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
