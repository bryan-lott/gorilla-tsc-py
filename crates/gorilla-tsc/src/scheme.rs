//! Prediction schemes: how each item is turned into a residual for a coder.
//!
//! Every scheme writes the first item of a block verbatim, then one residual
//! per item. Schemes are `Default`; the block framing creates a fresh one per
//! checkpoint block, so blocks decode independently by construction.

use core::marker::PhantomData;

use crate::{BitReader, BitWriter, BucketCoder, DecodeError, ResidualCoder, Word, XorCoder};

/// A reversible difference between consecutive words.
pub trait Op: Default + Copy {
    /// Residual of `cur` against `prev`.
    fn diff<W: Word>(prev: W, cur: W) -> W;
    /// Inverse of [`Op::diff`]: recover `cur` from `prev` and the residual.
    fn undiff<W: Word>(prev: W, residual: W) -> W;
}

/// Wrapping subtraction.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sub;
/// Bitwise XOR.
#[derive(Debug, Default, Clone, Copy)]
pub struct Xor;

impl Op for Sub {
    fn diff<W: Word>(prev: W, cur: W) -> W {
        cur.wrapping_sub(prev)
    }
    fn undiff<W: Word>(prev: W, residual: W) -> W {
        prev.wrapping_add(residual)
    }
}

impl Op for Xor {
    fn diff<W: Word>(prev: W, cur: W) -> W {
        prev ^ cur
    }
    fn undiff<W: Word>(prev: W, residual: W) -> W {
        prev ^ residual
    }
}

fn write_verbatim<W: Word>(out: &mut BitWriter, item: W) {
    out.write(item.to_u64(), W::BITS);
}

fn read_verbatim<W: Word>(input: &mut BitReader<'_>) -> Result<W, DecodeError> {
    Ok(W::from_u64_truncate(input.read(W::BITS)?))
}

/// Per-block encoder/decoder state for one prediction scheme.
pub trait Scheme<W: Word>: Default {
    /// Append the next item.
    fn encode(&mut self, out: &mut BitWriter, item: W);
    /// Read the next item.
    ///
    /// # Errors
    /// On truncated or impossible input.
    fn decode(&mut self, input: &mut BitReader<'_>) -> Result<W, DecodeError>;
}

/// One difference, coded directly: `residual = O(prev, cur)`.
#[derive(Debug)]
pub struct Single<W, O, C> {
    prev: Option<W>,
    coder: C,
    _op: PhantomData<O>,
}

/// Two differences: `d = O1(prev, cur)`, `residual = O2(prev_d, d)`, with
/// `prev_d` starting at zero so the first residual is just `d`.
#[derive(Debug)]
pub struct Stacked<W, O1, O2, C> {
    state: StackedState<W>,
    coder: C,
    _ops: PhantomData<(O1, O2)>,
}

#[derive(Debug, Clone, Copy)]
enum StackedState<W> {
    Empty,
    Started { prev: W, prev_d: W },
}

impl<W, O, C: Default> Default for Single<W, O, C> {
    fn default() -> Self {
        Self { prev: None, coder: C::default(), _op: PhantomData }
    }
}

impl<W, O1, O2, C: Default> Default for Stacked<W, O1, O2, C> {
    fn default() -> Self {
        Self { state: StackedState::Empty, coder: C::default(), _ops: PhantomData }
    }
}

impl<W: Word, O: Op, C: ResidualCoder<W>> Scheme<W> for Single<W, O, C> {
    fn encode(&mut self, out: &mut BitWriter, item: W) {
        match self.prev {
            None => write_verbatim(out, item),
            Some(prev) => self.coder.encode(out, O::diff(prev, item)),
        }
        self.prev = Some(item);
    }

    fn decode(&mut self, input: &mut BitReader<'_>) -> Result<W, DecodeError> {
        let item = match self.prev {
            None => read_verbatim(input)?,
            Some(prev) => O::undiff(prev, self.coder.decode(input)?),
        };
        self.prev = Some(item);
        Ok(item)
    }
}

impl<W: Word, O1: Op, O2: Op, C: ResidualCoder<W>> Scheme<W> for Stacked<W, O1, O2, C> {
    fn encode(&mut self, out: &mut BitWriter, item: W) {
        self.state = match self.state {
            StackedState::Empty => {
                write_verbatim(out, item);
                StackedState::Started { prev: item, prev_d: W::ZERO }
            }
            StackedState::Started { prev, prev_d } => {
                let d = O1::diff(prev, item);
                self.coder.encode(out, O2::diff(prev_d, d));
                StackedState::Started { prev: item, prev_d: d }
            }
        };
    }

    fn decode(&mut self, input: &mut BitReader<'_>) -> Result<W, DecodeError> {
        let (item, d) = match self.state {
            StackedState::Empty => (read_verbatim(input)?, W::ZERO),
            StackedState::Started { prev, prev_d } => {
                let d = O2::undiff(prev_d, self.coder.decode(input)?);
                (O1::undiff(prev, d), d)
            }
        };
        self.state = StackedState::Started { prev: item, prev_d: d };
        Ok(item)
    }
}

/// Gorilla value compression: XOR with the previous value.
pub type XorScheme<W> = Single<W, Xor, XorCoder>;
/// Delta-of-delta (Gorilla timestamps).
pub type DodScheme<W> = Stacked<W, Sub, Sub, BucketCoder>;
/// Delta, then XOR of consecutive deltas.
pub type XorDodScheme<W> = Stacked<W, Sub, Xor, XorCoder>;
/// XOR, then difference of consecutive XORs.
pub type DodXorScheme<W> = Stacked<W, Xor, Sub, BucketCoder>;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;

    pub(crate) fn encode_all<W: Word, S: Scheme<W>>(items: &[W]) -> (Vec<u8>, usize) {
        let mut s = S::default();
        let mut w = BitWriter::new();
        for &i in items {
            s.encode(&mut w, i);
        }
        let bits = w.bit_len();
        (w.finish(), bits)
    }

    fn round_trip<W: Word, S: Scheme<W>>(items: &[W]) -> Vec<W> {
        let (bytes, _) = encode_all::<W, S>(items);
        let mut r = BitReader::new(&bytes);
        let mut s = S::default();
        items.iter().map(|_| s.decode(&mut r).unwrap()).collect()
    }

    fn bits<W: Word, S: Scheme<W>>(items: &[W]) -> usize {
        encode_all::<W, S>(items).1
    }

    #[test]
    fn ops_invert() {
        assert_eq!(Sub::diff(10u64, 3), 3u64.wrapping_sub(10));
        assert_eq!(Sub::undiff(10u64, Sub::diff(10u64, 3)), 3);
        assert_eq!(Sub::diff(0u32, u32::MAX), u32::MAX);
        assert_eq!(Xor::diff(0b1100u64, 0b1010), 0b0110);
        assert_eq!(Xor::undiff(0b1100u64, 0b0110), 0b1010);
    }

    #[test]
    fn first_item_is_verbatim() {
        let (bytes, n) = encode_all::<u64, XorScheme<u64>>(&[0x0123_4567_89AB_CDEF]);
        assert_eq!((bytes, n), (0x0123_4567_89AB_CDEFu64.to_be_bytes().to_vec(), 64));
        let (bytes, n) = encode_all::<u32, DodScheme<u32>>(&[0xDEAD_BEEF]);
        assert_eq!((bytes, n), (0xDEAD_BEEFu32.to_be_bytes().to_vec(), 32));
    }

    #[test]
    fn constant_series_cost_one_bit_per_item() {
        let v = [7u64; 10];
        assert_eq!(bits::<u64, XorScheme<u64>>(&v), 64 + 9);
        assert_eq!(bits::<u64, DodScheme<u64>>(&v), 64 + 9);
        assert_eq!(bits::<u64, XorDodScheme<u64>>(&v), 64 + 9);
        assert_eq!(bits::<u64, DodXorScheme<u64>>(&v), 64 + 9);
    }

    #[test]
    fn linear_series_is_cheap_under_dod() {
        // Deltas of 5: first residual is 5 (9 bits), then all zero.
        let v: Vec<u64> = (0..10).map(|i| 1000 + 5 * i).collect();
        assert_eq!(bits::<u64, DodScheme<u64>>(&v), 64 + 9 + 8);
    }

    #[test]
    fn stacked_second_stage_uses_previous_first_stage() {
        // Delta 3 then 4: dod residuals 3 then 1 (9 bits each).
        assert_eq!(bits::<u64, DodScheme<u64>>(&[0, 3, 7]), 64 + 9 + 9);
        // Same delta twice: xor_dod residuals 3 then 0.
        // lz of 3 is 62, capped at 31, so its payload is 64 - 31 = 33 bits.
        assert_eq!(bits::<u64, XorDodScheme<u64>>(&[0, 3, 6]), 64 + (2 + 5 + 6 + 33) + 1);
    }

    #[test]
    fn wrapping_deltas_round_trip() {
        let v = [0u64, u64::MAX, 0, u64::MAX / 2, 1];
        assert_eq!(round_trip::<u64, DodScheme<u64>>(&v), v);
        assert_eq!(round_trip::<u64, XorDodScheme<u64>>(&v), v);
        assert_eq!(round_trip::<u64, DodXorScheme<u64>>(&v), v);
    }

    #[test]
    fn empty_stream_decodes_eof() {
        let mut s = XorScheme::<u64>::default();
        assert_eq!(s.decode(&mut BitReader::new(&[])), Err(DecodeError::UnexpectedEof));
        let mut s = DodScheme::<u64>::default();
        assert_eq!(s.decode(&mut BitReader::new(&[])), Err(DecodeError::UnexpectedEof));
    }

    macro_rules! prop_round_trips {
        ($($name:ident: $w:ty, $s:ty;)*) => { proptest! { $(
            #[test]
            fn $name(v in prop::collection::vec(any::<$w>(), 0..100)) {
                prop_assert_eq!(round_trip::<$w, $s>(&v), v);
            }
        )* } };
    }

    prop_round_trips! {
        xor_u64: u64, XorScheme<u64>;
        xor_u32: u32, XorScheme<u32>;
        dod_u64: u64, DodScheme<u64>;
        dod_u32: u32, DodScheme<u32>;
        xor_dod_u64: u64, XorDodScheme<u64>;
        xor_dod_u32: u32, XorDodScheme<u32>;
        dod_xor_u64: u64, DodXorScheme<u64>;
        dod_xor_u32: u32, DodXorScheme<u32>;
    }
}
