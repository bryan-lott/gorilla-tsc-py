//! Self-describing encoded streams: header, checkpoint blocks, typed samples.
//!
//! Layout: `MAGIC, VERSION, descriptor, varint(checkpoint_interval), varint(count)`
//! followed by the bit-packed payload. The descriptor byte is
//! `sample_type << 2 | mode`; its high nibble is reserved and must be zero.

use core::fmt;
use core::num::NonZeroU32;
use core::str::FromStr;

use crate::scheme::{DodScheme, DodXorScheme, Scheme, XorDodScheme, XorScheme};
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
pub trait Sample: Copy + fmt::Debug + sealed::Sealed + 'static {
    /// The word the codec works on.
    type Word: Word;
    /// Tag stored in the header.
    const TYPE: SampleType;
    /// Bit-exact conversion to the codec word.
    fn to_word(self) -> Self::Word;
    /// Inverse of [`Sample::to_word`].
    fn from_word(w: Self::Word) -> Self;
}

macro_rules! impl_sample {
    ($t:ty, $w:ty, $tag:ident, $to:expr, $from:expr) => {
        impl sealed::Sealed for $t {}
        impl Sample for $t {
            type Word = $w;
            const TYPE: SampleType = SampleType::$tag;
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

impl_sample!(i32, u32, I32, i32::cast_unsigned, u32::cast_signed);
impl_sample!(i64, u64, I64, i64::cast_unsigned, u64::cast_signed);
impl_sample!(f32, u32, F32, f32::to_bits, f32::from_bits);
impl_sample!(f64, u64, F64, f64::to_bits, f64::from_bits);

/// How consecutive items are predicted. See [`crate::scheme`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// XOR with the previous item (Gorilla values).
    Xor,
    /// Delta-of-delta (Gorilla timestamps).
    Dod,
    /// Delta, then XOR of consecutive deltas.
    XorDod,
    /// XOR, then difference of consecutive XORs.
    DodXor,
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

impl Mode {
    /// Name used in configuration strings.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Xor => "xor",
            Self::Dod => "dod",
            Self::XorDod => "xor_dod",
            Self::DodXor => "dod_xor",
        }
    }
    const ALL: [Self; 4] = [Self::Xor, Self::Dod, Self::XorDod, Self::DodXor];
}

impl FromStr for Mode {
    type Err = ConfigError;
    fn from_str(s: &str) -> Result<Self, ConfigError> {
        parse_named(s, "mode", &Self::ALL, Self::name)
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
    pub const fn new(mode: Mode) -> Self {
        Self { mode, checkpoint_interval: CheckpointInterval::DEFAULT }
    }

    /// Replace the checkpoint interval.
    #[must_use]
    pub const fn with_interval(self, checkpoint_interval: CheckpointInterval) -> Self {
        Self { checkpoint_interval, ..self }
    }
}

/// Runtime-selected scheme state for one block.
enum AnyScheme<W> {
    Xor(XorScheme<W>),
    Dod(DodScheme<W>),
    XorDod(XorDodScheme<W>),
    DodXor(DodXorScheme<W>),
}

impl<W: Word> AnyScheme<W> {
    fn new(mode: Mode) -> Self {
        match mode {
            Mode::Xor => Self::Xor(XorScheme::default()),
            Mode::Dod => Self::Dod(DodScheme::default()),
            Mode::XorDod => Self::XorDod(XorDodScheme::default()),
            Mode::DodXor => Self::DodXor(DodXorScheme::default()),
        }
    }
    fn encode(&mut self, out: &mut BitWriter, w: W) {
        match self {
            Self::Xor(s) => s.encode(out, w),
            Self::Dod(s) => s.encode(out, w),
            Self::XorDod(s) => s.encode(out, w),
            Self::DodXor(s) => s.encode(out, w),
        }
    }
    fn decode(&mut self, input: &mut BitReader<'_>) -> Result<W, DecodeError> {
        match self {
            Self::Xor(s) => s.decode(input),
            Self::Dod(s) => s.decode(input),
            Self::XorDod(s) => s.decode(input),
            Self::DodXor(s) => s.decode(input),
        }
    }
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
        let m = Mode::ALL.iter().position(|&m| m == self.config.mode).unwrap_or_default();
        u8::try_from(t << 2 | m).unwrap_or_default()
    }

    fn write(&self, out: &mut Vec<u8>) {
        out.extend([MAGIC, VERSION, self.descriptor()]);
        write_varint(out, u64::from(self.config.checkpoint_interval.get()));
        write_varint(out, self.count);
    }
}

/// Streaming encoder. Push items, then [`Encoder::finish`].
pub struct Encoder<T: Sample> {
    config: Config,
    out: BitWriter,
    scheme: AnyScheme<T::Word>,
    count: u64,
}

impl<T: Sample> Encoder<T> {
    /// New empty stream.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config, out: BitWriter::new(), scheme: AnyScheme::new(config.mode), count: 0 }
    }

    /// Append one item.
    pub fn push(&mut self, item: T) {
        if self.count.is_multiple_of(u64::from(self.config.checkpoint_interval.get())) {
            self.scheme = AnyScheme::new(self.config.mode);
        }
        self.scheme.encode(&mut self.out, item.to_word());
        self.count += 1;
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
        self.out.bit_len()
    }

    /// Header followed by the payload.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
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
    if descriptor >> 4 != 0 {
        return Err(DecodeError::InvalidHeader("reserved descriptor bits set"));
    }
    let sample_type = SampleType::ALL[usize::from(descriptor >> 2)];
    let mode = Mode::ALL[usize::from(descriptor & 0b11)];
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
    let mut items = Vec::with_capacity(cap);
    let mut scheme = AnyScheme::new(header.config.mode);
    for i in 0..header.count {
        if i.is_multiple_of(interval) {
            scheme = AnyScheme::new(header.config.mode);
        }
        items.push(T::from_word(scheme.decode(&mut input)?));
    }
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

    const MODES: [Mode; 4] = [Mode::Xor, Mode::Dod, Mode::XorDod, Mode::DodXor];

    fn cfg(mode: Mode, interval: u32) -> Config {
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
        assert_eq!(["xor", "dod", "xor_dod", "dod_xor"], MODES.map(Mode::name));
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
            Err(ConfigError("mode must be one of 'xor', 'dod', 'xor_dod', 'dod_xor' (got 'rle')".into()))
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
        let bytes = encode::<i64>(&[], cfg(Mode::Dod, 300));
        // magic, version, I64<<2 | Dod, varint(300) = AC 02, varint(0)
        assert_eq!(bytes, vec![b'G', 1, 0b0101, 0xAC, 0x02, 0x00]);
        let bytes = encode::<f32>(&[1.5], cfg(Mode::DodXor, 1));
        let mut expected = vec![b'G', 1, 0b1011, 1, 1];
        expected.extend(1.5f32.to_bits().to_be_bytes());
        assert_eq!(bytes, expected);
    }

    #[test]
    fn header_round_trips() {
        let bytes = encode::<f64>(&[1.0, 2.0, 3.0], cfg(Mode::XorDod, 7));
        let (h, _) = read_header(&bytes).unwrap();
        assert_eq!(h, Header { sample_type: SampleType::F64, config: cfg(Mode::XorDod, 7), count: 3 });
    }

    #[test]
    fn header_errors() {
        assert_eq!(read_header(&[]), Err(DecodeError::UnexpectedEof));
        assert_eq!(read_header(b"X"), Err(DecodeError::BadMagic));
        assert_eq!(read_header(b"G"), Err(DecodeError::UnexpectedEof));
        assert_eq!(read_header(&[b'G', 2, 0, 1, 0]), Err(DecodeError::UnsupportedVersion(2)));
        assert_eq!(read_header(&[b'G', 1, 0x10, 1, 0]), Err(DecodeError::InvalidHeader("reserved descriptor bits set")));
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
        for (ti, t) in [SampleType::I32, SampleType::I64, SampleType::F32, SampleType::F64].into_iter().enumerate() {
            for (mi, m) in MODES.into_iter().enumerate() {
                let d = u8::try_from(ti << 2 | mi).unwrap();
                let (h, _) = read_header(&[b'G', 1, d, 5, 0]).unwrap();
                assert_eq!((h.sample_type, h.config.mode), (t, m));
            }
        }
    }

    #[test]
    fn type_mismatch_is_reported() {
        let bytes = encode::<i64>(&[1, 2], Config::new(Mode::Dod));
        assert_eq!(decode::<f64>(&bytes), Err(DecodeError::KindMismatch { expected: "f64", found: "int64" }));
    }

    #[test]
    fn truncated_payload_is_eof() {
        let bytes = encode::<f64>(&[1.0, 2.0, 3.0], Config::new(Mode::Xor));
        assert_eq!(decode::<f64>(&bytes[..bytes.len() - 1]), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn trailing_data_is_rejected() {
        let mut bytes = encode::<f64>(&[1.0], Config::new(Mode::Xor));
        bytes.push(0);
        assert_eq!(decode::<f64>(&bytes), Err(DecodeError::Corrupt("trailing data after last item")));
        // Non-zero padding bits are trailing data too.
        let mut bytes = encode::<i64>(&[5, 5], Config::new(Mode::Dod)); // 64 + 1 bits
        *bytes.last_mut().unwrap() |= 1;
        assert_eq!(decode::<i64>(&bytes), Err(DecodeError::Corrupt("trailing data after last item")));
    }

    #[test]
    fn huge_claimed_count_fails_without_allocating() {
        let bytes = [b'G', 1, 0b0101, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F];
        assert_eq!(decode::<i64>(&bytes), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn decode_any_dispatches_on_header() {
        assert_eq!(decode_any(&encode(&[1i32, -2], Config::new(Mode::Dod))), Ok(Decoded::I32(vec![1, -2])));
        assert_eq!(decode_any(&encode(&[1i64], Config::new(Mode::Dod))), Ok(Decoded::I64(vec![1])));
        assert_eq!(decode_any(&encode(&[1.5f32], Config::new(Mode::Xor))), Ok(Decoded::F32(vec![1.5])));
        assert_eq!(decode_any(&encode(&[2.5f64], Config::new(Mode::Xor))), Ok(Decoded::F64(vec![2.5])));
    }

    #[test]
    fn encoder_len_tracks_pushes() {
        let mut e = Encoder::<i64>::new(Config::new(Mode::Dod));
        assert!(e.is_empty());
        e.push(1);
        e.push(2);
        assert_eq!(e.len(), 2);
        assert!(!e.is_empty());
    }

    #[test]
    fn checkpoint_restarts_with_verbatim_item() {
        // interval 2: [verbatim, residual][verbatim, residual]
        assert_eq!(payload_bits(&[5i64, 5, 5, 5], cfg(Mode::Dod, 2)), 2 * (64 + 1));
        assert_eq!(payload_bits(&[5i64, 5, 5, 5], cfg(Mode::Dod, 4)), 64 + 3);
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
            fn $name(v in prop::collection::vec(any::<$t>(), 0..150), m in 0usize..4, k in 1u32..20) {
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
        fn blocks_are_independent(
            a in prop::collection::vec(any::<f64>(), 1..40),
            b in prop::collection::vec(any::<f64>(), 1..40),
            m in 0usize..4,
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
