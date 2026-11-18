//! Timestamp/value pairs and compression estimates.

use crate::{Config, DecodeError, Encoder, Sample, Threads, decode, encode_parallel};

/// Compress `(timestamp, value)` pairs as two streams, stable-sorted by timestamp.
///
/// Timestamps must be `Ord`, so only integer types qualify.
#[must_use]
pub fn encode_pairs<T: Sample + Ord, V: Sample>(
    pairs: &[(T, V)],
    timestamps: Config,
    values: Config,
) -> (Vec<u8>, Vec<u8>) {
    encode_pairs_with(pairs, timestamps, values, Threads::ONE)
}

/// [`encode_pairs`] with each stream's blocks encoded across `threads`
/// threads; the bytes are identical.
#[must_use]
pub fn encode_pairs_with<T: Sample + Ord, V: Sample>(
    pairs: &[(T, V)],
    timestamps: Config,
    values: Config,
    threads: Threads,
) -> (Vec<u8>, Vec<u8>) {
    let mut sorted = pairs.to_vec();
    sorted.sort_by_key(|p| p.0); // stable
    let (ts, vals): (Vec<T>, Vec<V>) = sorted.into_iter().unzip();
    (encode_parallel(&ts, timestamps, threads), encode_parallel(&vals, values, threads))
}

/// Inverse of [`encode_pairs`].
///
/// # Errors
/// If either stream is invalid or they hold different numbers of items.
pub fn decode_pairs<T: Sample, V: Sample>(timestamps: &[u8], values: &[u8]) -> Result<Vec<(T, V)>, DecodeError> {
    let ts = decode::<T>(timestamps)?;
    let vals = decode::<V>(values)?;
    if ts.len() != vals.len() {
        return Err(DecodeError::Corrupt("pair streams differ in length"));
    }
    Ok(ts.into_iter().zip(vals).collect())
}

/// Items examined by [`estimate`].
pub const ESTIMATE_SAMPLE: usize = 1000;

/// Predicted compression of a dataset, from its first [`ESTIMATE_SAMPLE`] items.
/// Sizes count the payload only; the fixed header is excluded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Uncompressed bytes / compressed bytes (infinite for an empty sample).
    pub ratio: f64,
    /// Compressed payload bytes.
    pub compressed_size: usize,
    /// Compressed bits per item (0 for an empty sample).
    pub bits_per_item: f64,
}

/// Estimate how well `data` compresses under `config`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // sizes are far below 2^52
pub fn estimate<T: Sample>(data: &[T], config: Config) -> Estimate {
    let sample = &data[..data.len().min(ESTIMATE_SAMPLE)];
    let mut e = Encoder::new(config);
    for &x in sample {
        e.push(x);
    }
    let bits = e.payload_bits();
    let compressed_size = bits.div_ceil(8);
    let uncompressed = sample.len() * T::TYPE.byte_width();
    Estimate {
        ratio: if compressed_size == 0 { f64::INFINITY } else { uncompressed as f64 / compressed_size as f64 },
        compressed_size,
        bits_per_item: if sample.is_empty() { 0.0 } else { bits as f64 / sample.len() as f64 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode;
    use crate::Codec;
    use crate::CheckpointInterval;

    #[test]
    fn pairs_sort_stably_by_timestamp() {
        let pairs = [(5i64, 2.0f64), (1, 9.0), (5, 1.0)];
        let (ts, vals) = encode_pairs(&pairs, Config::new(Codec::Dod), Config::new(Codec::Xor));
        assert_eq!(decode::<i64>(&ts), Ok(vec![1, 5, 5]));
        assert_eq!(decode::<f64>(&vals), Ok(vec![9.0, 2.0, 1.0]));
        assert_eq!(decode_pairs(&ts, &vals), Ok(vec![(1i64, 9.0f64), (5, 2.0), (5, 1.0)]));
    }

    #[test]
    fn pairs_use_their_configs() {
        let pairs = [(1i32, 1.0f32), (2, 2.0)];
        let ts_cfg = Config::new(Codec::Xor).with_interval(CheckpointInterval::new(1).unwrap());
        let (ts, vals) = encode_pairs(&pairs, ts_cfg, Config::new(Codec::DodXor));
        assert_eq!(ts, encode(&[1i32, 2], ts_cfg));
        assert_eq!(vals, encode(&[1.0f32, 2.0], Config::new(Codec::DodXor)));
    }

    #[test]
    fn pair_streams_of_different_lengths_are_corrupt() {
        let ts = encode(&[1i64, 2], Config::new(Codec::Dod));
        let vals = encode(&[1.0f64], Config::new(Codec::Xor));
        assert_eq!(decode_pairs::<i64, f64>(&ts, &vals), Err(DecodeError::Corrupt("pair streams differ in length")));
    }

    #[test]
    fn pair_stream_errors_propagate() {
        let vals = encode(&[1.0f64], Config::new(Codec::Xor));
        assert_eq!(decode_pairs::<i64, f64>(b"x", &vals), Err(DecodeError::BadMagic));
        assert_eq!(decode_pairs::<f64, f64>(&vals, b"x"), Err(DecodeError::BadMagic));
    }

    #[test]
    fn estimate_counts_payload_only() {
        // 64 bits + '10'+7 for the first dod (1), then eight '0' bits = 81 bits = 11 bytes.
        let data: Vec<i64> = (1000..1010).collect();
        let e = estimate(&data, Config::new(Codec::Dod));
        assert_eq!(e, Estimate { ratio: 80.0 / 11.0, compressed_size: 11, bits_per_item: 8.1 });
    }

    #[test]
    fn estimate_uses_item_width() {
        let e = estimate(&[1.0f32, 1.0], Config::new(Codec::Xor));
        assert_eq!(e, Estimate { ratio: 8.0 / 5.0, compressed_size: 5, bits_per_item: 16.5 });
    }

    #[test]
    fn estimate_respects_config() {
        let one = Config::new(Codec::Dod).with_interval(CheckpointInterval::new(1).unwrap());
        assert_eq!(estimate(&[5i64, 5], one).compressed_size, 16);
    }

    #[test]
    fn estimate_samples_a_prefix() {
        let data: Vec<f64> = (0..3000).map(f64::from).collect();
        let cfg = Config::new(Codec::Xor);
        let mut e = Encoder::new(cfg);
        for &x in &data[..ESTIMATE_SAMPLE] {
            e.push(x);
        }
        assert_eq!(estimate(&data, cfg).compressed_size, e.payload_bits().div_ceil(8));
    }

    #[test]
    fn estimate_of_nothing() {
        let e = estimate::<f64>(&[], Config::new(Codec::Xor));
        assert_eq!((e.ratio, e.compressed_size, e.bits_per_item), (f64::INFINITY, 0, 0.0));
    }
}
