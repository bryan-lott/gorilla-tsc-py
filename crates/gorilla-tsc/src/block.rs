//! Block codecs: compress one checkpoint block of words at a time.
//!
//! Working on whole blocks lets a codec choose parameters for the block (a
//! Rice parameter, a decimal exponent, ...) before writing it. Blocks share no
//! state, so each decodes independently.

use core::marker::PhantomData;

use crate::{BitReader, BitWriter, DecodeError, Scheme, Word};

/// Compresses a block of words.
pub trait BlockCodec<W: Word> {
    /// Append `items` (non-empty) to `out`.
    fn encode_block(&self, items: &[W], out: &mut BitWriter);
    /// Read a block of `n` items, appending them to `out`.
    ///
    /// # Errors
    /// On truncated or impossible input.
    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<W>) -> Result<(), DecodeError>;
}

/// Runs a per-item [`Scheme`] over a block, starting from fresh state.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerItem<S>(PhantomData<S>);

impl<W: Word, S: Scheme<W>> BlockCodec<W> for PerItem<S> {
    fn encode_block(&self, items: &[W], out: &mut BitWriter) {
        let mut s = S::default();
        for &w in items {
            s.encode(out, w);
        }
    }

    fn decode_block(&self, n: usize, input: &mut BitReader<'_>, out: &mut Vec<W>) -> Result<(), DecodeError> {
        let mut s = S::default();
        for _ in 0..n {
            out.push(s.decode(input)?);
        }
        Ok(())
    }
}
