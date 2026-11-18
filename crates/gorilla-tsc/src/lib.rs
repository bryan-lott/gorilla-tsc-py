//! Gorilla-style time series compression.
//!
//! Widths, signedness and encoder state are carried in the type system: the
//! codec is generic over a [`Word`] (`u32`/`u64`), so there is no place for a
//! hard-coded bit count to disagree with the data it describes.
//!
//! ```
//! use gorilla_tsc::{decode, encode, CheckpointInterval, Config, Mode};
//!
//! let cfg = Config::new(Mode::Auto).with_interval(CheckpointInterval::new(1000).unwrap());
//! let bytes = encode(&[20.5f64, 21.0, 20.8], cfg);
//! assert_eq!(decode::<f64>(&bytes).unwrap(), [20.5, 21.0, 20.8]);
//! ```

mod adaptive;
mod bits;
mod bitsplit;
mod block;
mod coder;
mod decimal;
mod delta;
mod dict;
mod error;
mod pairs;
mod profile;
mod range;
mod rice;
mod scheme;
mod split;
mod stream;
mod word;

pub use adaptive::AdaptiveBlock;
pub use bits::{BitReader, BitWriter};
pub use bitsplit::BitSplit;
pub use block::{BlockCodec, PerItem};
pub use decimal::{DecimalRice, Scaled};
pub use dict::Dictionary;
pub use split::DecimalSplit;
pub use profile::{GUARD_RATIO, Profile, RECIPE_SLACK_PERCENT};
pub use delta::{Coder, DeltaChoice, DeltaRice, Order};
pub use coder::{BucketCoder, ResidualCoder, XorCoder};
pub use error::DecodeError;
pub use range::{Prob, RangeDecoder, RangeEncoder};
pub use rice::{RiceBlock, RiceParams, Transform};
pub use pairs::{ESTIMATE_SAMPLE, Estimate, decode_pairs, encode_pairs, encode_pairs_with, estimate};
pub use scheme::{DodScheme, DodXorScheme, Op, Scheme, Single, Stacked, Sub, Xor, XorDodScheme, XorScheme};
pub use stream::{
    CheckpointInterval, Codec, Config, ConfigError, Decoded, Encoder, Header, MAGIC, Mode, Sample, SampleType, Threads, VERSION,
    decode, decode_any, encode, encode_parallel, read_header,
};
pub use word::Word;
