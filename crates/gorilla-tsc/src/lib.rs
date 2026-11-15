//! Gorilla-style time series compression.
//!
//! Widths, signedness and encoder state are carried in the type system: the
//! codec is generic over a [`Word`] (`u32`/`u64`), so there is no place for a
//! hard-coded bit count to disagree with the data it describes.

mod adaptive;
mod bits;
mod block;
mod coder;
mod decimal;
mod delta;
mod error;
mod pairs;
mod range;
mod rice;
mod scheme;
mod stream;
mod word;

pub use adaptive::AdaptiveBlock;
pub use bits::{BitReader, BitWriter};
pub use block::{BlockCodec, PerItem};
pub use decimal::{DecimalRice, Scaled};
pub use delta::{DeltaRice, Order};
pub use coder::{BucketCoder, ResidualCoder, XorCoder};
pub use error::DecodeError;
pub use range::{Prob, RangeDecoder, RangeEncoder};
pub use rice::{RiceBlock, RiceParams, Transform};
pub use pairs::{ESTIMATE_SAMPLE, Estimate, decode_pairs, encode_pairs, estimate};
pub use scheme::{DodScheme, DodXorScheme, Op, Scheme, Single, Stacked, Sub, Xor, XorDodScheme, XorScheme};
pub use stream::{
    CheckpointInterval, Config, ConfigError, Decoded, Encoder, Header, MAGIC, Mode, Sample, SampleType, VERSION,
    decode, decode_any, encode, encode_parallel, read_header,
};
pub use word::Word;
