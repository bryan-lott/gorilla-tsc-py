//! Delta coding with Rice-coded residuals, the difference order chosen per block.
//!
//! Block layout: first item zig-zagged with a length prefix; then, if there is more than one item, a
//! 2-bit order (`0` = first differences, `1` = second differences, `2` = the
//! values themselves), a coder bit (`0` = [`RiceBlock`], `1` =
//! [`AdaptiveBlock`]) and the residuals in that coder. The cheapest order and
//! coder are chosen per block.

use crate::rice::{read_prefixed, write_prefixed};
use crate::scheme::{Op, Sub};
use crate::{AdaptiveBlock, BitReader, BitWriter, BlockCodec, DecodeError, RiceBlock, Word};

/// Delta / delta-of-delta block codec.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeltaRice;

/// Which differences are coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    /// `x[i] - x[i-1]`.
    First,
    /// `d[i] - d[i-1]`, with `d[0]` predicted as zero.
    Second,
    /// `x[i]` itself (frame of reference via the Rice offset transform).
    Zero,
}

/// Residual coders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Coder {
    /// [`RiceBlock`]: exact per-block Rice parameters.
    Rice,
    /// [`AdaptiveBlock`]: adaptive binary range coding.
    Adaptive,
}

impl Coder {
    /// Both coders, in tie-break preference.
    pub const ALL: [Self; 2] = [Self::Rice, Self::Adaptive];

    /// Name used in profile strings.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rice => "rice",
            Self::Adaptive => "adaptive",
        }
    }
}

/// A Delta block's choices, fixed up front instead of searched per block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeltaChoice {
    /// Which differences are coded.
    pub order: Order,
    /// How the residuals are coded.
    pub coder: Coder,
}

impl DeltaChoice {
    /// Every combination, in tie-break preference.
    #[must_use]
    pub fn all() -> Vec<Self> {
        Order::ALL.into_iter().flat_map(|order| Coder::ALL.map(|coder| Self { order, coder })).collect()
    }
}

/// Width of the order field.
const ORDER_BITS: u32 = 2;

impl Order {
    /// All orders, in tie-break preference.
    pub const ALL: [Self; 3] = [Self::First, Self::Second, Self::Zero];

    /// Name used in profile strings.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Second => "second",
            Self::Zero => "zero",
        }
    }

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

/// The coder bit and residuals.
fn write_residuals<W: Word>(r: &[W], coder: Coder, out: &mut BitWriter) {
    out.write(u64::from(coder == Coder::Adaptive), 1);
    match coder {
        Coder::Rice => RiceBlock::encode(r, out),
        Coder::Adaptive => AdaptiveBlock::encode(r, out),
    }
}

impl DeltaRice {
    /// Encode a block with `choice` instead of searching for the cheapest one.
    /// The block format is unchanged, so it decodes like any Delta block.
    pub fn encode_with<W: Word>(items: &[W], choice: DeltaChoice, out: &mut BitWriter) {
        let Some((&first, rest)) = items.split_first() else { return };
        write_prefixed::<W>(out, first.zigzag().to_u64());
        if rest.is_empty() {
            return;
        }
        out.write(choice.order.code(), ORDER_BITS);
        write_residuals(&residuals(items, choice.order), choice.coder, out);
    }
}

impl<W: Word> BlockCodec<W> for DeltaRice {
    fn encode_block(&self, items: &[W], out: &mut BitWriter) {
        let Some((&first, rest)) = items.split_first() else { return };
        write_prefixed::<W>(out, first.zigzag().to_u64());
        if rest.is_empty() {
            return;
        }
        // (bits, order, coded residuals) for every order and coder; keep the smallest.
        let mut best: Option<(usize, Order, BitWriter)> = None;
        for order in Order::ALL {
            let r = residuals(items, order);
            for coder in Coder::ALL {
                let mut w = BitWriter::new();
                write_residuals(&r, coder, &mut w);
                if best.as_ref().is_none_or(|b| w.bit_len() < b.0) {
                    best = Some((w.bit_len(), order, w));
                }
            }
        }
        if let Some((_, order, coded)) = best {
            out.write(order.code(), ORDER_BITS);
            out.append(&coded);
        }
    }

    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<W>) -> Result<(), DecodeError> {
        if n == 0 {
            return Ok(());
        }
        let first = W::from_u64_truncate(read_prefixed::<W>(input, "first value longer than a word")?).unzigzag();
        out.push(first);
        if n == 1 {
            return Ok(());
        }
        let order = Order::from_code(input.read(ORDER_BITS)?).ok_or(DecodeError::Corrupt("unknown delta order"))?;
        let mut r = Vec::with_capacity(n - 1);
        if input.read_bit()? {
            AdaptiveBlock::decode(n - 1, input, &mut r)?;
        } else {
            RiceBlock::decode(n - 1, input, &mut r)?;
        }
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

    fn adaptive_bits(r: &[u64]) -> u64 {
        let mut w = BitWriter::new();
        AdaptiveBlock::encode(r, &mut w);
        w.bit_len() as u64
    }

    /// Cost of a block's first value: zig-zagged, 7-bit (u64) length prefix.
    fn first(v: u64) -> u64 {
        7 + u64::from(v.zigzag().bit_width())
    }

    /// Cheapest coding of residuals, including the coder bit.
    fn best(r: &[u64]) -> u64 {
        1 + RiceBlock::cost(r, RiceBlock::choose(r)).min(adaptive_bits(r))
    }

    #[test]
    fn encode_with_writes_a_normal_delta_block() {
        let x: Vec<u64> = (0..300u64).map(|i| 1000 + i * i % 37).collect();
        let mut best = (usize::MAX, BitWriter::new());
        for choice in DeltaChoice::all() {
            let mut w = BitWriter::new();
            DeltaRice::encode_with(&x, choice, &mut w);
            let len = w.bit_len();
            let bytes = w.clone().finish();
            let mut out: Vec<u64> = Vec::new();
            DeltaRice.decode_block(x.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
            assert_eq!(out, x, "{choice:?}");
            if len < best.0 {
                best = (len, w);
            }
        }
        // The search picks the cheapest of the same six.
        assert_eq!(bits(&x), best.0);
        assert_eq!(DeltaChoice::all().len(), 6);
        assert_eq!(DeltaChoice::all()[0], DeltaChoice { order: Order::First, coder: Coder::Rice });
        let mut w = BitWriter::new();
        DeltaRice::encode_with(&[5u64], DeltaChoice::all()[3], &mut w);
        assert_eq!(w.bit_len(), bits(&[5u64]));
    }

    #[test]
    fn names() {
        assert_eq!(Order::ALL.map(Order::name), ["first", "second", "zero"]);
        assert_eq!(Coder::ALL.map(Coder::name), ["rice", "adaptive"]);
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
    fn first_value_is_zigzag_length_prefixed() {
        // zigzag(42) = 84: 7 bits, after a 7-bit (u64) or 6-bit (u32) length.
        assert_eq!(bits(&[42u64]), 7 + 7);
        assert_eq!(bits(&[42u32]), 6 + 7);
        // zigzag(-3) = 5: 3 bits.
        assert_eq!(bits(&[(-3i64).cast_unsigned()]), 7 + 3);
        assert_eq!(bits(&[0u64]), 7);
        // Full-width values still fit: zigzag(i64::MIN) = u64::MAX.
        assert_eq!(bits(&[i64::MIN.cast_unsigned()]), 7 + 64);
        assert_eq!(round_trip(&[i64::MIN.cast_unsigned()]), [i64::MIN.cast_unsigned()]);
        assert_eq!(round_trip(&[i32::MIN.cast_unsigned()]), [i32::MIN.cast_unsigned()]);
    }

    #[test]
    fn overlong_first_value_length_is_corrupt() {
        let mut w = BitWriter::new();
        w.write(33, 6); // u32 lengths go up to 32
        let bytes = w.finish();
        let r = DeltaRice.decode_block(1, &mut BitReader::new(&bytes), &mut Vec::<u32>::new());
        assert_eq!(r, Err(DecodeError::Corrupt("first value longer than a word")));
    }

    #[test]
    fn size_is_cheaper_order_plus_overhead() {
        let cases: [&[u64]; 3] = [&[0, 5, 10, 15, 20, 25, 30], &[1, 1, 2, 1, 1, 2, 1], &[100, 90, 500, 7, 7, 7]];
        for x in cases {
            let costs = [Order::First, Order::Second, Order::Zero].map(|o| best(&residuals(x, o)));
            assert_eq!(bits(x) as u64, first(x[0]) + 2 + costs.into_iter().min().unwrap(), "{x:?}");
        }
    }

    #[test]
    fn linear_series_costs_about_one_bit_per_value() {
        let x: Vec<u64> = (0..100).map(|i| 1000 + 7 * i).collect();
        let costs = [Order::First, Order::Second, Order::Zero].map(|o| best(&residuals(&x, o)));
        assert_eq!(bits(&x) as u64, first(x[0]) + 2 + costs.into_iter().min().unwrap());
        // 99 residuals at ~1 bit each plus a small Rice header.
        assert!(bits(&x) as u64 <= first(x[0]) + 2 + 99 + 30, "{} bits", bits(&x));
    }

    #[test]
    fn jittery_series_uses_first_order() {
        let x: Vec<u64> = (0..100u64).map(|i| 10_000 * i + (i * 7919) % 11).collect();
        let one = best(&residuals(&x, Order::First));
        assert!(one < best(&residuals(&x, Order::Second)));
        assert_eq!(bits(&x) as u64, first(x[0]) + 2 + one);
    }

    #[test]
    fn unordered_values_use_order_zero() {
        // A shuffled narrow range: values are cheap, their differences are not.
        // Uniform noise in a narrow band: differences are triangular and wider.
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let x: Vec<u64> = (0..1000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                1_000_000 + s % 1024
            })
            .collect();
        let zero = best(&residuals(&x, Order::Zero));
        assert!(zero < best(&residuals(&x, Order::First)));
        assert!(zero < best(&residuals(&x, Order::Second)));
        assert_eq!(bits(&x) as u64, first(x[0]) + 2 + zero);
    }

    #[test]
    fn unknown_order_is_corrupt() {
        let mut w = BitWriter::new();
        w.write(1, 7); // first value: length 1, then "1" (zigzag of -1)
        w.write(1, 1);
        w.write(3, 2);
        w.write(0, 64);
        let bytes = w.finish();
        let r = DeltaRice.decode_block(2, &mut BitReader::new(&bytes), &mut Vec::<u64>::new());
        assert_eq!(r, Err(DecodeError::Corrupt("unknown delta order")));
    }

    #[test]
    fn mostly_constant_series_uses_adaptive_coder() {
        let x: Vec<u64> = (0..1000u64).map(|i| 500 + u64::from(i % 97 == 0)).collect();
        let r = residuals(&x, Order::First);
        assert!(adaptive_bits(&r) < RiceBlock::cost(&r, RiceBlock::choose(&r)));
        assert!((bits(&x) as u64) < first(x[0]) + 2 + 400, "{} bits", bits(&x));
    }

    #[test]
    fn smooth_series_picks_cheaper_coder() {
        let x: Vec<u64> = (0..500u64).map(|i| 1_000_000 + i * i / 3).collect();
        for order in [Order::First, Order::Second, Order::Zero] {
            assert!(bits(&x) as u64 <= first(x[0]) + 2 + best(&residuals(&x, order)));
        }
    }

    #[test]
    fn truncated_is_eof() {
        let mut w = BitWriter::new();
        let x: Vec<u64> = (0..100).map(|i| i * i).collect();
        DeltaRice.encode_block(&x, &mut w);
        let bytes = w.finish();
        for cut in [0, 1, bytes.len() / 2, bytes.len() - 1] {
            let mut out = Vec::<u64>::new();
            assert_eq!(
                DeltaRice.decode_block(x.len(), &mut BitReader::new(&bytes[..cut]), &mut out),
                Err(DecodeError::UnexpectedEof),
                "cut at {cut}"
            );
        }
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
