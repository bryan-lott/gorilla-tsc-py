//! Error types.

use core::fmt;

/// Why a byte stream could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// The stream ended before all promised items were read.
    UnexpectedEof,
    /// The header does not start with the expected magic bytes.
    BadMagic,
    /// The stream was written by an unsupported format version.
    UnsupportedVersion(u8),
    /// A header field holds a value no encoder writes.
    InvalidHeader(&'static str),
    /// The header describes a different kind of stream than requested.
    KindMismatch {
        /// What the caller asked to decode.
        expected: &'static str,
        /// What the header says the stream holds.
        found: &'static str,
    },
    /// A payload bit pattern that no encoder produces.
    Corrupt(&'static str),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("unexpected end of stream"),
            Self::BadMagic => f.write_str("not a gorilla-tsc stream (bad magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Self::InvalidHeader(what) => write!(f, "invalid header: {what}"),
            Self::KindMismatch { expected, found } => {
                write!(f, "expected a {expected} stream, found {found}")
            }
            Self::Corrupt(what) => write!(f, "corrupt stream: {what}"),
        }
    }
}

impl std::error::Error for DecodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages() {
        let cases = [
            (DecodeError::UnexpectedEof, "unexpected end of stream"),
            (DecodeError::BadMagic, "not a gorilla-tsc stream (bad magic)"),
            (DecodeError::UnsupportedVersion(7), "unsupported format version 7"),
            (DecodeError::InvalidHeader("x"), "invalid header: x"),
            (DecodeError::KindMismatch { expected: "f64", found: "int32" }, "expected a f64 stream, found int32"),
            (DecodeError::Corrupt("y"), "corrupt stream: y"),
        ];
        for (e, msg) in cases {
            assert_eq!(e.to_string(), msg);
        }
    }
}
