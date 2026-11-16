//! Self-describing encoded streams: header, checkpoint blocks, typed samples.
//!
//! Layout: `MAGIC, VERSION, descriptor, varint(checkpoint_interval), varint(count)`
//! followed by the bit-packed payload. The descriptor byte is
//! `sample_type << 4 | mode`; its top two bits are reserved and must be zero.

use core::fmt;
use core::num::NonZeroU32;
use core::str::FromStr;

use crate::block::{BlockCodec, PerItem};
use crate::decimal::DecimalRice;
use crate::delta::DeltaRice;
use crate::scheme::{DodScheme, DodXorScheme, XorDodScheme, XorScheme};
use crate::{BitReader, BitWriter, DecodeError, Word};

/// First byte of every stream.
pub const MAGIC: u8 = b'G';
/// Format version written by this crate.
pub const VERSION: u8 = 1;

mod sealed {
    pub trait Sealed {}
}

/// The four element types a stream can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
    /// 32-bit signed integers.
    I32,
    /// 64-bit signed integers.
    I64,
    /// 32-bit IEEE 754 floats.
    F32,
    /// 64-bit IEEE 754 floats.
    F64,
}

/// An element type that can be compressed. Sealed: the set is fixed by the format.
pub trait Sample: Copy + Send + Sync + fmt::Debug + sealed::Sealed + 'static {
    /// The word the codec works on.
    type Word: Word;
    /// Tag stored in the header.
    const TYPE: SampleType;
    /// Bit-exact conversion to the codec word.
    fn to_word(self) -> Self::Word;
    /// Inverse of [`Sample::to_word`].
    fn from_word(w: Self::Word) -> Self;
    /// Largest decimal exponent worth trying (0 for integers).
    const MAX_SCALE: i32;
    /// Codecs [`Mode::Auto`] tries for this type.
    const AUTO: &'static [Codec];
    /// `Some(n)` iff [`Sample::from_scaled`]`(n, k)` reproduces `self` bit for bit.
    /// For floats `n` is `self * 10^k` rounded; integers only scale at `k = 0`.
    fn to_scaled(self, k: i32) -> Option<i64>;
    /// `n / 10^k` (or `n * 10^-k`) as `Self`.
    fn from_scaled(n: i64, k: i32) -> Self;
}

/// Exact powers of ten representable in an `f64` (10^0 ..= 10^22).
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18,
    1e19, 1e20, 1e21, 1e22,
];

/// Integers of magnitude below 2^53 convert to `f64` exactly.
const EXACT_INT: f64 = 9_007_199_254_740_992.0;

fn pow10(k: i32) -> f64 {
    POW10[k.unsigned_abs() as usize]
}

/// `v * 10^k` rounded, if it is a safely convertible integer.
#[allow(clippy::cast_possible_truncation)] // |s| < 2^53 is checked first
fn scaled_int(v: f64, k: i32) -> Option<i64> {
    let s = if k >= 0 { v * pow10(k) } else { v / pow10(k) };
    (s.abs() < EXACT_INT).then(|| s.round() as i64)
}

fn scale_f64(v: f64, k: i32) -> Option<i64> {
    scaled_int(v, k).filter(|&n| unscale_f64(n, k).to_bits() == v.to_bits())
}

#[allow(clippy::cast_precision_loss)] // |n| < 2^53 whenever n came from scale_f64
fn unscale_f64(n: i64, k: i32) -> f64 {
    if k >= 0 { n as f64 / pow10(k) } else { n as f64 * pow10(k) }
}

macro_rules! impl_sample {
    ($t:ty, $w:ty, $tag:ident, $to:expr, $from:expr, $max:expr, $auto:expr, $scale:expr, $unscale:expr) => {
        impl sealed::Sealed for $t {}
        impl Sample for $t {
            type Word = $w;
            const TYPE: SampleType = SampleType::$tag;
            const MAX_SCALE: i32 = $max;
            const AUTO: &'static [Codec] = $auto;
            fn to_scaled(self, k: i32) -> Option<i64> {
                let f: fn($t, i32) -> Option<i64> = $scale;
                f(self, k)
            }
            fn from_scaled(n: i64, k: i32) -> Self {
                let f: fn(i64, i32) -> $t = $unscale;
                f(n, k)
            }
            fn to_word(self) -> $w {
                let f: fn($t) -> $w = $to;
                f(self)
            }
            fn from_word(w: $w) -> Self {
                let f: fn($w) -> $t = $from;
                f(w)
            }
        }
    };
}

/// What [`Mode::Auto`] tries, per element family. On the 25 benchmark
/// datasets (Elf, Chimp, timestamps) these give the same size as trying every
/// codec: the stacked modes never win, `Decimal` on integers is `Delta` plus a
/// header, and `Delta` on raw float bits only ever won constant runs. `Raw`
/// caps every block at its uncompressed size.
const INT_AUTO: &[Codec] = &[Codec::Dod, Codec::Delta, Codec::Raw];
const FLOAT_AUTO: &[Codec] = &[Codec::Xor, Codec::Decimal, Codec::Raw];

impl_sample!(i32, u32, I32, i32::cast_unsigned, u32::cast_signed, 0, INT_AUTO,
    |v, k| (k == 0).then_some(i64::from(v)), |n, _| i32::try_from(n).unwrap_or_default());
impl_sample!(i64, u64, I64, i64::cast_unsigned, u64::cast_signed, 0, INT_AUTO,
    |v, k| (k == 0).then_some(v), |n, _| n);
impl_sample!(f32, u32, F32, f32::to_bits, f32::from_bits, 10, FLOAT_AUTO, scale_f32, unscale_f32);
impl_sample!(f64, u64, F64, f64::to_bits, f64::from_bits, 18, FLOAT_AUTO, scale_f64, unscale_f64);

fn scale_f32(v: f32, k: i32) -> Option<i64> {
    scaled_int(f64::from(v), k).filter(|&n| unscale_f32(n, k).to_bits() == v.to_bits())
}

#[allow(clippy::cast_possible_truncation)] // checked by the round trip in scale_f32
fn unscale_f32(n: i64, k: i32) -> f32 {
    unscale_f64(n, k) as f32
}

/// A per-block compression method. Every block of a stream is coded by
/// exactly one codec; [`Mode::Auto`] picks one per block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    /// XOR with the previous item (Gorilla values).
    Xor,
    /// Delta-of-delta (Gorilla timestamps).
    Dod,
    /// Delta, then XOR of consecutive deltas.
    XorDod,
    /// XOR, then difference of consecutive XORs.
    DodXor,
    /// First or second differences (chosen per block), Rice or adaptively coded.
    Delta,
    /// Floats scaled to integers by a per-block power of ten, then delta coded.
    /// Values that do not scale exactly are stored verbatim. Integers: as `Delta`.
    Decimal,
    /// Every word verbatim: the bound no block ever needs to exceed.
    Raw,
}

/// How a stream chooses codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// The same codec for every block.
    Fixed(Codec),
    /// Per block, whichever of [`Sample::AUTO`] is smallest (a 4-bit selector per block).
    Auto,
}

impl From<Codec> for Mode {
    fn from(c: Codec) -> Self {
        Self::Fixed(c)
    }
}

/// Items per independently decodable block. Never zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointInterval(NonZeroU32);

/// Encoder settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Config {
    /// Prediction mode.
    pub mode: Mode,
    /// Block length.
    pub checkpoint_interval: CheckpointInterval,
}

/// Parsed stream header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Element type.
    pub sample_type: SampleType,
    /// Settings the stream was written with.
    pub config: Config,
    /// Number of items.
    pub count: u64,
}

/// A rejected configuration string or number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

impl SampleType {
    /// Name used in configuration strings.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
    /// Uncompressed size of one item in bytes.
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
    const ALL: [Self; 4] = [Self::I32, Self::I64, Self::F32, Self::F64];
}

/// Parse `s` as one of `all` by name, or explain what was expected.
fn parse_named<T: Copy>(s: &str, what: &str, all: &[T], name: fn(T) -> &'static str) -> Result<T, ConfigError> {
    all.iter().copied().find(|&t| name(t) == s).ok_or_else(|| {
        let valid: Vec<String> = all.iter().map(|&t| format!("'{}'", name(t))).collect();
        ConfigError(format!("{what} must be one of {} (got '{s}')", valid.join(", ")))
    })
}

impl FromStr for SampleType {
    type Err = ConfigError;
    fn from_str(s: &str) -> Result<Self, ConfigError> {
        parse_named(s, "precision", &Self::ALL, Self::name)
    }
}

impl Codec {
    /// Name used in configuration strings.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Xor => "xor",
            Self::Dod => "dod",
            Self::XorDod => "xor_dod",
            Self::DodXor => "dod_xor",
            Self::Delta => "delta",
            Self::Decimal => "decimal",
            Self::Raw => "raw",
        }
    }

    /// Every codec.
    pub const ALL: [Self; 7] = [Self::Xor, Self::Dod, Self::XorDod, Self::DodXor, Self::Delta, Self::Decimal, Self::Raw];

    /// Code in the header's mode field and in [`Mode::Auto`]'s block selector.
    const fn code(self) -> u8 {
        match self {
            Self::Xor => 0,
            Self::Dod => 1,
            Self::XorDod => 2,
            Self::DodXor => 3,
            Self::Delta => 4,
            Self::Decimal => 5,
            Self::Raw => 7,
        }
    }

    fn from_code(code: u64) -> Option<Self> {
        Self::ALL.into_iter().find(|c| u64::from(c.code()) == code)
    }
}

impl Mode {
    /// Name used in configuration strings.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fixed(c) => c.name(),
            Self::Auto => "auto",
        }
    }

    /// Header code of [`Mode::Auto`]; never a codec's code.
    const AUTO_CODE: u8 = 6;

    const fn code(self) -> u8 {
        match self {
            Self::Fixed(c) => c.code(),
            Self::Auto => Self::AUTO_CODE,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        if code == Self::AUTO_CODE { Some(Self::Auto) } else { Codec::from_code(u64::from(code)).map(Self::Fixed) }
    }

    /// Every mode: each codec, then [`Mode::Auto`].
    #[must_use]
    pub fn all() -> Vec<Self> {
        Codec::ALL.into_iter().map(Self::Fixed).chain([Self::Auto]).collect()
    }
}

impl FromStr for Codec {
    type Err = ConfigError;
    fn from_str(s: &str) -> Result<Self, ConfigError> {
        parse_named(s, "codec", &Self::ALL, Self::name)
    }
}

impl FromStr for Mode {
    type Err = ConfigError;
    fn from_str(s: &str) -> Result<Self, ConfigError> {
        parse_named(s, "mode", &Self::all(), Self::name)
    }
}

impl CheckpointInterval {
    /// Gorilla's default block length.
    pub const DEFAULT: Self = Self(NonZeroU32::new(100).unwrap());

    /// `None` for zero.
    #[must_use]
    pub const fn new(n: u32) -> Option<Self> {
        match NonZeroU32::new(n) {
            Some(n) => Some(Self(n)),
            None => None,
        }
    }

    /// The interval as a count.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl TryFrom<i64> for CheckpointInterval {
    type Error = ConfigError;
    fn try_from(n: i64) -> Result<Self, ConfigError> {
        u32::try_from(n).ok().and_then(Self::new).ok_or_else(|| {
            ConfigError(format!("checkpoint interval must be between 1 and {} (got {n})", u32::MAX))
        })
    }
}

impl Default for CheckpointInterval {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Config {
    /// Settings with the default checkpoint interval.
    #[must_use]
    pub fn new(mode: impl Into<Mode>) -> Self {
        Self { mode: mode.into(), checkpoint_interval: CheckpointInterval::DEFAULT }
    }

    /// Replace the checkpoint interval.
    #[must_use]
    pub const fn with_interval(self, checkpoint_interval: CheckpointInterval) -> Self {
        Self { checkpoint_interval, ..self }
    }
}

/// Encode one block with `codec`.
fn encode_codec<T: Sample>(codec: Codec, items: &[T::Word], out: &mut BitWriter) {
    type W<T> = <T as Sample>::Word;
    match codec {
        Codec::Xor => PerItem::<XorScheme<W<T>>>::default().encode_block(items, out),
        Codec::Dod => PerItem::<DodScheme<W<T>>>::default().encode_block(items, out),
        Codec::XorDod => PerItem::<XorDodScheme<W<T>>>::default().encode_block(items, out),
        Codec::DodXor => PerItem::<DodXorScheme<W<T>>>::default().encode_block(items, out),
        Codec::Delta => DeltaRice.encode_block(items, out),
        Codec::Decimal => DecimalRice::<T>::default().encode_block(items, out),
        Codec::Raw => {
            for &w in items {
                out.write(w.to_u64(), W::<T>::BITS);
            }
        }
    }
}

/// Encode one block as `mode` says.
fn encode_block<T: Sample>(mode: Mode, items: &[T::Word], out: &mut BitWriter) {
    match mode {
        Mode::Fixed(codec) => encode_codec::<T>(codec, items, out),
        Mode::Auto => {
            let (codec, best) = T::AUTO
                .iter()
                .map(|&c| {
                    let mut w = BitWriter::new();
                    encode_codec::<T>(c, items, &mut w);
                    (c, w)
                })
                .min_by_key(|(_, w)| w.bit_len())
                .unwrap_or((Codec::Xor, BitWriter::new()));
            out.write(u64::from(codec.code()), AUTO_SELECTOR_BITS);
            out.append(&best);
        }
    }
}

/// Width of [`Mode::Auto`]'s per-block mode selector.
const AUTO_SELECTOR_BITS: u32 = 4;

/// Decode one block of `n` items coded with `codec`.
fn decode_codec<T: Sample>(
    codec: Codec,
    n: usize,
    input: &mut BitReader<'_>,
    out: &mut Vec<T::Word>,
) -> Result<(), DecodeError> {
    type W<T> = <T as Sample>::Word;
    match codec {
        Codec::Xor => PerItem::<XorScheme<W<T>>>::default().decode_block(n, input, out),
        Codec::Dod => PerItem::<DodScheme<W<T>>>::default().decode_block(n, input, out),
        Codec::XorDod => PerItem::<XorDodScheme<W<T>>>::default().decode_block(n, input, out),
        Codec::DodXor => PerItem::<DodXorScheme<W<T>>>::default().decode_block(n, input, out),
        Codec::Delta => DeltaRice.decode_block(n, input, out),
        Codec::Decimal => DecimalRice::<T>::default().decode_block(n, input, out),
        Codec::Raw => {
            for _ in 0..n {
                out.push(W::<T>::from_u64_truncate(input.read(W::<T>::BITS)?));
            }
            Ok(())
        }
    }
}

/// Decode one block of `n` items written as `mode` says.
fn decode_block<T: Sample>(
    mode: Mode,
    n: usize,
    input: &mut BitReader<'_>,
    out: &mut Vec<T::Word>,
) -> Result<(), DecodeError> {
    let codec = match mode {
        Mode::Fixed(codec) => codec,
        Mode::Auto => Codec::from_code(input.read(AUTO_SELECTOR_BITS)?).ok_or(DecodeError::Corrupt("unknown block mode"))?,
    };
    decode_codec::<T>(codec, n, input, out)
}

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let low = u8::try_from(v & 0x7F).unwrap_or_default();
        v >>= 7;
        if v == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

fn read_varint(bytes: &mut &[u8]) -> Result<u64, DecodeError> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let (&b, rest) = bytes.split_first().ok_or(DecodeError::UnexpectedEof)?;
        *bytes = rest;
        if shift == 63 && b > 1 {
            return Err(DecodeError::InvalidHeader("varint too long"));
        }
        v |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(DecodeError::InvalidHeader("varint too long"))
}

fn take_byte(bytes: &mut &[u8]) -> Result<u8, DecodeError> {
    let (&b, rest) = bytes.split_first().ok_or(DecodeError::UnexpectedEof)?;
    *bytes = rest;
    Ok(b)
}

impl Header {
    fn descriptor(&self) -> u8 {
        let t = SampleType::ALL.iter().position(|&t| t == self.sample_type).unwrap_or_default();
        u8::try_from(t << 4).unwrap_or_default() | self.config.mode.code()
    }

    fn write(&self, out: &mut Vec<u8>) {
        out.extend([MAGIC, VERSION, self.descriptor()]);
        write_varint(out, u64::from(self.config.checkpoint_interval.get()));
        write_varint(out, self.count);
    }
}

/// Streaming encoder. Push items, then [`Encoder::finish`]. Items are
/// buffered until a block is complete, then compressed together.
pub struct Encoder<T: Sample> {
    config: Config,
    out: BitWriter,
    block: Vec<T::Word>,
    count: u64,
}

impl<T: Sample> Encoder<T> {
    /// New empty stream.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config, out: BitWriter::new(), block: Vec::new(), count: 0 }
    }

    /// Append one item.
    pub fn push(&mut self, item: T) {
        self.block.push(item.to_word());
        self.count += 1;
        if self.block.len() == self.config.checkpoint_interval.get() as usize {
            encode_block::<T>(self.config.mode, &self.block, &mut self.out);
            self.block.clear();
        }
    }

    /// Items pushed so far.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Whether nothing has been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Payload size so far, excluding the header.
    #[must_use]
    pub fn payload_bits(&self) -> usize {
        let mut pending = BitWriter::new();
        if !self.block.is_empty() {
            encode_block::<T>(self.config.mode, &self.block, &mut pending);
        }
        self.out.bit_len() + pending.bit_len()
    }

    /// Header followed by the payload.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if !self.block.is_empty() {
            encode_block::<T>(self.config.mode, &self.block, &mut self.out);
        }
        let header = Header { sample_type: T::TYPE, config: self.config, count: self.count };
        let mut bytes = Vec::with_capacity(16 + self.out.bit_len() / 8);
        header.write(&mut bytes);
        bytes.extend(self.out.finish());
        bytes
    }
}

/// Compress `items` in one call.
#[must_use]
pub fn encode<T: Sample>(items: &[T], config: Config) -> Vec<u8> {
    let mut e = Encoder::new(config);
    for &i in items {
        e.push(i);
    }
    e.finish()
}

/// Compress `items` using up to `threads` threads. Blocks are independent, so
/// the output is byte-for-byte identical to [`encode`].
#[must_use]
pub fn encode_parallel<T: Sample>(items: &[T], config: Config, threads: core::num::NonZeroUsize) -> Vec<u8> {
    let interval = config.checkpoint_interval.get() as usize;
    let blocks: Vec<&[T]> = items.chunks(interval).collect();
    let per_thread = blocks.len().div_ceil(threads.get()).max(1);
    let encode_run = |run: &[&[T]]| {
        let mut out = BitWriter::new();
        for block in run {
            let words: Vec<T::Word> = block.iter().map(|x| x.to_word()).collect();
            encode_block::<T>(config.mode, &words, &mut out);
        }
        out
    };
    let parts: Vec<BitWriter> = std::thread::scope(|s| {
        let handles: Vec<_> = blocks.chunks(per_thread).map(|run| s.spawn(move || encode_run(run))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    });
    let mut payload = BitWriter::new();
    for p in &parts {
        payload.append(p);
    }
    let header = Header { sample_type: T::TYPE, config, count: items.len() as u64 };
    let mut bytes = Vec::with_capacity(16 + payload.bit_len() / 8);
    header.write(&mut bytes);
    bytes.extend(payload.finish());
    bytes
}

/// Parse the header of `bytes`, returning it and the payload.
///
/// # Errors
/// On a malformed or unsupported header.
pub fn read_header(mut bytes: &[u8]) -> Result<(Header, &[u8]), DecodeError> {
    if take_byte(&mut bytes)? != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let version = take_byte(&mut bytes)?;
    if version != VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    let descriptor = take_byte(&mut bytes)?;
    if descriptor >> 6 != 0 {
        return Err(DecodeError::InvalidHeader("reserved descriptor bits set"));
    }
    let sample_type = SampleType::ALL[usize::from(descriptor >> 4)];
    let mode = Mode::from_code(descriptor & 0x0F).ok_or(DecodeError::InvalidHeader("unknown mode"))?;
    let checkpoint_interval = u32::try_from(read_varint(&mut bytes)?)
        .ok()
        .and_then(CheckpointInterval::new)
        .ok_or(DecodeError::InvalidHeader("checkpoint interval out of range"))?;
    let count = read_varint(&mut bytes)?;
    let config = Config { mode, checkpoint_interval };
    Ok((Header { sample_type, config, count }, bytes))
}

/// Decompress a stream of `T`.
///
/// # Errors
/// If the stream is malformed, truncated, has trailing data, or holds another type.
pub fn decode<T: Sample>(bytes: &[u8]) -> Result<Vec<T>, DecodeError> {
    let (header, payload) = read_header(bytes)?;
    if header.sample_type != T::TYPE {
        return Err(DecodeError::KindMismatch { expected: T::TYPE.name(), found: header.sample_type.name() });
    }
    let interval = u64::from(header.config.checkpoint_interval.get());
    let mut input = BitReader::new(payload);
    // Every item costs at least one bit, so a lying count cannot force a huge allocation.
    let cap = usize::try_from(header.count).unwrap_or(usize::MAX).min(payload.len() * 8);
    let mut words = Vec::with_capacity(cap);
    let mut remaining = header.count;
    while remaining > 0 {
        let n = remaining.min(interval);
        decode_block::<T>(header.config.mode, usize::try_from(n).unwrap_or(usize::MAX), &mut input, &mut words)?;
        remaining -= n;
    }
    let items = words.into_iter().map(T::from_word).collect();
    let rest = input.remaining();
    if rest >= 8 || input.read(u32::try_from(rest).unwrap_or(u32::MAX))? != 0 {
        return Err(DecodeError::Corrupt("trailing data after last item"));
    }
    Ok(items)
}

/// A decoded stream of whichever type its header names.
#[derive(Debug, Clone, PartialEq)]
pub enum Decoded {
    /// `i32` items.
    I32(Vec<i32>),
    /// `i64` items.
    I64(Vec<i64>),
    /// `f32` items.
    F32(Vec<f32>),
    /// `f64` items.
    F64(Vec<f64>),
}

/// Decompress a stream without knowing its type in advance.
///
/// # Errors
/// As [`decode`].
pub fn decode_any(bytes: &[u8]) -> Result<Decoded, DecodeError> {
    Ok(match read_header(bytes)?.0.sample_type {
        SampleType::I32 => Decoded::I32(decode(bytes)?),
        SampleType::I64 => Decoded::I64(decode(bytes)?),
        SampleType::F32 => Decoded::F32(decode(bytes)?),
        SampleType::F64 => Decoded::F64(decode(bytes)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const MODES: [Mode; 8] = [
        Mode::Fixed(Codec::Xor),
        Mode::Fixed(Codec::Dod),
        Mode::Fixed(Codec::XorDod),
        Mode::Fixed(Codec::DodXor),
        Mode::Fixed(Codec::Delta),
        Mode::Fixed(Codec::Decimal),
        Mode::Fixed(Codec::Raw),
        Mode::Auto,
    ];

    fn cfg(mode: impl Into<Mode>, interval: u32) -> Config {
        Config::new(mode).with_interval(CheckpointInterval::new(interval).unwrap())
    }

    fn payload_bits<T: Sample>(items: &[T], config: Config) -> usize {
        let mut e = Encoder::new(config);
        for &i in items {
            e.push(i);
        }
        e.payload_bits()
    }

    #[test]
    fn names_round_trip() {
        for t in [SampleType::I32, SampleType::I64, SampleType::F32, SampleType::F64] {
            assert_eq!(t.name().parse::<SampleType>(), Ok(t));
        }
        for m in MODES {
            assert_eq!(m.name().parse::<Mode>(), Ok(m));
        }
        assert_eq!(
            ["int32", "int64", "f32", "f64"],
            [SampleType::I32, SampleType::I64, SampleType::F32, SampleType::F64].map(SampleType::name)
        );
        assert_eq!(["xor", "dod", "xor_dod", "dod_xor", "delta", "decimal", "raw", "auto"], MODES.map(Mode::name));
        assert_eq!(Mode::all(), MODES);
    }

    #[test]
    fn byte_widths() {
        assert_eq!(
            [SampleType::I32, SampleType::I64, SampleType::F32, SampleType::F64].map(SampleType::byte_width),
            [4, 8, 4, 8]
        );
    }

    #[test]
    fn bad_names_are_rejected_with_the_valid_set() {
        assert_eq!(
            "int16".parse::<SampleType>(),
            Err(ConfigError("precision must be one of 'int32', 'int64', 'f32', 'f64' (got 'int16')".into()))
        );
        assert_eq!(
            "rle".parse::<Mode>(),
            Err(ConfigError("mode must be one of 'xor', 'dod', 'xor_dod', 'dod_xor', 'delta', 'decimal', 'raw', 'auto' (got 'rle')".into()))
        );
    }

    #[test]
    fn checkpoint_interval_bounds() {
        assert_eq!(CheckpointInterval::new(0), None);
        assert_eq!(CheckpointInterval::new(1).map(CheckpointInterval::get), Some(1));
        assert_eq!(CheckpointInterval::default().get(), 100);
        let err = ConfigError("checkpoint interval must be between 1 and 4294967295 (got 0)".into());
        assert_eq!(CheckpointInterval::try_from(0), Err(err));
        assert!(CheckpointInterval::try_from(-1).is_err());
        assert!(CheckpointInterval::try_from(1i64 << 32).is_err());
        assert_eq!(CheckpointInterval::try_from(i64::from(u32::MAX)).map(CheckpointInterval::get), Ok(u32::MAX));
    }

    #[test]
    fn header_layout() {
        let bytes = encode::<i64>(&[], cfg(Codec::Dod, 300));
        // magic, version, I64 << 4 | Dod, varint(300) = AC 02, varint(0)
        assert_eq!(bytes, vec![b'G', 1, 0x11, 0xAC, 0x02, 0x00]);
        let bytes = encode::<f32>(&[1.5], cfg(Codec::DodXor, 1));
        let mut expected = vec![b'G', 1, 0x23, 1, 1];
        expected.extend(1.5f32.to_bits().to_be_bytes());
        assert_eq!(bytes, expected);
    }

    #[test]
    fn header_round_trips() {
        let bytes = encode::<f64>(&[1.0, 2.0, 3.0], cfg(Codec::XorDod, 7));
        let (h, _) = read_header(&bytes).unwrap();
        assert_eq!(h, Header { sample_type: SampleType::F64, config: cfg(Codec::XorDod, 7), count: 3 });
    }

    #[test]
    fn header_errors() {
        assert_eq!(read_header(&[]), Err(DecodeError::UnexpectedEof));
        assert_eq!(read_header(b"X"), Err(DecodeError::BadMagic));
        assert_eq!(read_header(b"G"), Err(DecodeError::UnexpectedEof));
        assert_eq!(read_header(&[b'G', 2, 0, 1, 0]), Err(DecodeError::UnsupportedVersion(2)));
        assert_eq!(read_header(&[b'G', 1, 0x40, 1, 0]), Err(DecodeError::InvalidHeader("reserved descriptor bits set")));
        assert_eq!(read_header(&[b'G', 1, 0x0F, 1, 0]), Err(DecodeError::InvalidHeader("unknown mode")));
        assert_eq!(read_header(&[b'G', 1, 0, 0, 0]), Err(DecodeError::InvalidHeader("checkpoint interval out of range")));
        assert_eq!(read_header(&[b'G', 1, 0, 0x80]), Err(DecodeError::UnexpectedEof));
        let too_long = [b'G', 1, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        assert_eq!(read_header(&too_long), Err(DecodeError::InvalidHeader("varint too long")));
        // interval > u32::MAX
        assert_eq!(read_header(&[b'G', 1, 0, 0x80, 0x80, 0x80, 0x80, 0x10, 0]), Err(DecodeError::InvalidHeader("checkpoint interval out of range")));
    }

    #[test]
    fn config_error_displays_its_message() {
        assert_eq!(ConfigError("boom".into()).to_string(), "boom");
    }

    #[test]
    fn varint_tenth_byte_boundary() {
        // count = u64::MAX needs ten bytes, the last holding only bit 63.
        let max = [b'G', 1, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        assert_eq!(read_header(&max).map(|h| h.0.count), Ok(u64::MAX));
        let zero_tail = [b'G', 1, 0, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        assert_eq!(read_header(&zero_tail).map(|h| h.0.count), Ok(0));
        let overflow = [b'G', 1, 0, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02];
        assert_eq!(read_header(&overflow), Err(DecodeError::InvalidHeader("varint too long")));
    }

    #[test]
    fn varint_round_trips() {
        for v in [0, 1, 127, 128, 300, u64::from(u32::MAX), u64::MAX - 1, u64::MAX] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let mut s = buf.as_slice();
            assert_eq!(read_varint(&mut s), Ok(v));
            assert_eq!(s, &[] as &[u8]);
        }
    }

    #[test]
    fn every_descriptor_decodes() {
        // The format's mode codes; Auto keeps 6 from before Raw existed.
        let codes = [
            (0, Mode::Fixed(Codec::Xor)),
            (1, Mode::Fixed(Codec::Dod)),
            (2, Mode::Fixed(Codec::XorDod)),
            (3, Mode::Fixed(Codec::DodXor)),
            (4, Mode::Fixed(Codec::Delta)),
            (5, Mode::Fixed(Codec::Decimal)),
            (6, Mode::Auto),
            (7, Mode::Fixed(Codec::Raw)),
        ];
        assert_eq!(codes.len(), MODES.len());
        for (ti, t) in [SampleType::I32, SampleType::I64, SampleType::F32, SampleType::F64].into_iter().enumerate() {
            for (code, m) in codes {
                let d = u8::try_from(ti << 4).unwrap() | code;
                let (h, _) = read_header(&[b'G', 1, d, 5, 0]).unwrap();
                assert_eq!((h.sample_type, h.config.mode), (t, m));
                // and the encoder writes the same code
                assert_eq!(Header { sample_type: t, config: cfg(m, 5), count: 0 }.descriptor(), d);
            }
        }
    }

    #[test]
    fn type_mismatch_is_reported() {
        let bytes = encode::<i64>(&[1, 2], Config::new(Codec::Dod));
        assert_eq!(decode::<f64>(&bytes), Err(DecodeError::KindMismatch { expected: "f64", found: "int64" }));
    }

    #[test]
    fn truncated_payload_is_eof() {
        let bytes = encode::<f64>(&[1.0, 2.0, 3.0], Config::new(Codec::Xor));
        assert_eq!(decode::<f64>(&bytes[..bytes.len() - 1]), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn trailing_data_is_rejected() {
        let mut bytes = encode::<f64>(&[1.0], Config::new(Codec::Xor));
        bytes.push(0);
        assert_eq!(decode::<f64>(&bytes), Err(DecodeError::Corrupt("trailing data after last item")));
        // Non-zero padding bits are trailing data too.
        let mut bytes = encode::<i64>(&[5, 5], Config::new(Codec::Dod)); // 64 + 1 bits
        *bytes.last_mut().unwrap() |= 1;
        assert_eq!(decode::<i64>(&bytes), Err(DecodeError::Corrupt("trailing data after last item")));
    }

    #[test]
    fn huge_claimed_count_fails_without_allocating() {
        let bytes = [b'G', 1, 0x11, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F];
        assert_eq!(decode::<i64>(&bytes), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn decode_any_dispatches_on_header() {
        assert_eq!(decode_any(&encode(&[1i32, -2], Config::new(Codec::Dod))), Ok(Decoded::I32(vec![1, -2])));
        assert_eq!(decode_any(&encode(&[1i64], Config::new(Codec::Dod))), Ok(Decoded::I64(vec![1])));
        assert_eq!(decode_any(&encode(&[1.5f32], Config::new(Codec::Xor))), Ok(Decoded::F32(vec![1.5])));
        assert_eq!(decode_any(&encode(&[2.5f64], Config::new(Codec::Xor))), Ok(Decoded::F64(vec![2.5])));
    }

    #[test]
    fn encoder_len_tracks_pushes() {
        let mut e = Encoder::<i64>::new(Config::new(Codec::Dod));
        assert!(e.is_empty());
        e.push(1);
        e.push(2);
        assert_eq!(e.len(), 2);
        assert!(!e.is_empty());
    }

    #[test]
    fn checkpoint_restarts_with_verbatim_item() {
        // interval 2: [verbatim, residual][verbatim, residual]
        assert_eq!(payload_bits(&[5i64, 5, 5, 5], cfg(Codec::Dod, 2)), 2 * (64 + 1));
        assert_eq!(payload_bits(&[5i64, 5, 5, 5], cfg(Codec::Dod, 4)), 64 + 3);
    }

    #[test]
    fn decimal_mode_beats_xor_on_decimal_data() {
        let v: Vec<f64> = (0..2000).map(|i| f64::from(150 + (i * 37) % 41) / 10.0).collect();
        let xor = payload_bits(&v, Config::new(Codec::Xor));
        let dec = payload_bits(&v, Config::new(Codec::Decimal));
        assert!(dec * 3 < xor, "decimal {dec} vs xor {xor}");
        assert_eq!(decode::<f64>(&encode(&v, Config::new(Codec::Decimal))), Ok(v));
    }

    #[test]
    fn decimal_mode_on_integers_matches_delta() {
        let v: Vec<i64> = (0..500).map(|i| 1000 + 3 * i + i % 4).collect();
        let delta = payload_bits(&v, Config::new(Codec::Delta));
        // per block of 100: 6-bit exponent + 7-bit (empty) exception count
        assert_eq!(payload_bits(&v, Config::new(Codec::Decimal)), delta + 5 * (6 + 7));
        assert_eq!(decode::<i64>(&encode(&v, Config::new(Codec::Decimal))), Ok(v));
    }

    #[test]
    fn auto_picks_the_smallest_mode_per_block() {
        // Block 1: one-decimal values (decimal wins). Block 2: noisy high-precision floats.
        let mut v: Vec<f64> = (0..100).map(|i| f64::from(200 + (i * 37) % 23) / 10.0).collect();
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        v.extend((0..100).map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            1.0 + f64::from(u32::try_from(s >> 40).unwrap()) / 16_777_216.0
        }));
        let per_block: usize = v
            .chunks(100)
            .map(|b| 4 + f64::AUTO.iter().map(|&m| payload_bits(b, Config::new(m))).min().unwrap())
            .sum();
        assert_eq!(payload_bits(&v, Config::new(Mode::Auto)), per_block);
        let best_single = Codec::ALL.iter().map(|&m| payload_bits(&v, Config::new(m))).min().unwrap();
        assert!(per_block < best_single);
        assert_eq!(decode::<f64>(&encode(&v, Config::new(Mode::Auto))), Ok(v));
    }

    #[test]
    fn parallel_encoding_matches_sequential() {
        let v: Vec<f64> = (0..10_007).map(|i| f64::from(i % 311) / 10.0).collect();
        let cfg = cfg(Mode::Auto, 100);
        for t in [1, 2, 3, 8, 200] {
            let threads = core::num::NonZeroUsize::new(t).unwrap();
            assert_eq!(encode_parallel(&v, cfg, threads), encode(&v, cfg), "threads = {t}");
        }
        let one = core::num::NonZeroUsize::new(4).unwrap();
        assert_eq!(encode_parallel::<i64>(&[], cfg, one), encode::<i64>(&[], cfg));
    }

    #[test]
    fn auto_candidates_by_sample_type() {
        // Measured on 25 benchmark datasets: these lose nothing against all codecs.
        assert_eq!(i64::AUTO, [Codec::Dod, Codec::Delta, Codec::Raw]);
        assert_eq!(i32::AUTO, i64::AUTO);
        assert_eq!(f64::AUTO, [Codec::Xor, Codec::Decimal, Codec::Raw]);
        assert_eq!(f32::AUTO, f64::AUTO);
    }

    #[test]
    fn raw_is_verbatim() {
        let v = [1.5f64, f64::NAN, -0.0];
        assert_eq!(payload_bits(&v, Config::new(Codec::Raw)), 3 * 64);
        assert_eq!(payload_bits(&[7i32, -7], Config::new(Codec::Raw)), 2 * 32);
        assert_eq!(bits_of(&decode::<f64>(&encode(&v, Config::new(Codec::Raw))).unwrap()), bits_of(&v));
        // header code 7
        assert_eq!(encode::<f64>(&[], Config::new(Codec::Raw))[2], 0x37);
    }

    #[test]
    fn auto_on_integers_matches_best_integer_mode() {
        let v: Vec<i64> = (0..300).map(|i| 1_700_000_000 + 10 * i + i % 3).collect();
        let per_block: usize = v
            .chunks(100)
            .map(|b| 4 + i64::AUTO.iter().map(|&m| payload_bits(b, Config::new(m))).min().unwrap())
            .sum();
        assert_eq!(payload_bits(&v, Config::new(Mode::Auto)), per_block);
    }

    #[test]
    fn auto_rejects_unknown_block_selector() {
        // f64 auto stream, one item, 4-bit selector 15.
        let bytes = [b'G', 1, 0x36, 1, 1, 0b1111_0000, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(decode::<f64>(&bytes), Err(DecodeError::Corrupt("unknown block mode")));
        // selector 6 would be Auto itself
        let bytes = [b'G', 1, 0x36, 1, 1, 0b0110_0000, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(decode::<f64>(&bytes), Err(DecodeError::Corrupt("unknown block mode")));
    }

    #[test]
    fn nan_payloads_and_signed_zero_survive() {
        let v = [f64::from_bits(0x7FF8_0000_0000_0001), -0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY];
        for m in MODES {
            let out = decode::<f64>(&encode(&v, Config::new(m))).unwrap();
            assert_eq!(out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), v.map(f64::to_bits).to_vec());
        }
    }

    fn bits_of<T: Sample>(v: &[T]) -> Vec<u64> {
        v.iter().map(|x| x.to_word().to_u64()).collect()
    }

    macro_rules! props {
        ($($name:ident: $t:ty;)*) => { proptest! { $(
            #[test]
            fn $name(v in prop::collection::vec(any::<$t>(), 0..150), m in 0usize..MODES.len(), k in 1u32..20) {
                let c = cfg(MODES[m], k);
                let bytes = encode(&v, c);
                let out = decode::<$t>(&bytes).unwrap();
                prop_assert_eq!(bits_of(&out), bits_of(&v));
                // Streaming and bulk produce identical bytes.
                let mut e = Encoder::new(c);
                for &x in &v { e.push(x); }
                prop_assert_eq!(e.finish(), bytes);
            }
        )* } };
    }

    props! {
        round_trip_i32: i32;
        round_trip_i64: i64;
        round_trip_f32: f32;
        round_trip_f64: f64;
    }

    proptest! {
        #[test]
        fn auto_never_exceeds_raw(w in prop::collection::vec(any::<u64>(), 1..400), k in 1u32..120) {
            let v: Vec<f64> = w.iter().map(|&x| f64::from_bits(x)).collect();
            let blocks = v.len().div_ceil(k as usize);
            prop_assert!(payload_bits(&v, cfg(Mode::Auto, k)) <= v.len() * 64 + blocks * 4);
        }

        #[test]
        fn parallel_matches_sequential(v in prop::collection::vec(any::<i64>(), 0..300), k in 1u32..40, t in 1usize..6, m in 0usize..MODES.len()) {
            let c = cfg(MODES[m], k);
            prop_assert_eq!(encode_parallel(&v, c, core::num::NonZeroUsize::new(t).unwrap()), encode(&v, c));
        }

        #[test]
        fn blocks_are_independent(
            a in prop::collection::vec(any::<f64>(), 1..40),
            b in prop::collection::vec(any::<f64>(), 1..40),
            m in 0usize..MODES.len(),
        ) {
            let k = u32::try_from(a.len()).unwrap();
            let joined: Vec<f64> = a.iter().chain(&b).copied().collect();
            let c = cfg(MODES[m], k);
            let solo = |v: &[f64]| payload_bits(v, cfg(MODES[m], 1000));
            // b may span more than one block if it is longer than a.
            let b_bits: usize = b.chunks(a.len()).map(solo).sum();
            prop_assert_eq!(payload_bits(&joined, c), solo(&a) + b_bits);
        }
    }
}
