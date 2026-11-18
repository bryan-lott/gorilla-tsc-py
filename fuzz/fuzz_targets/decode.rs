//! Decoding arbitrary bytes must never panic: it returns a value or a `DecodeError`.
//!
//! Streams whose header claims more than `MAX_ITEMS` are only header-parsed:
//! a valid stream can be tiny for a huge count (one repeated value costs
//! almost nothing per item), so decoding them is a decompression bomb by
//! design. Callers decoding untrusted data check the count first (README);
//! this target does the same. Over-reservation for lying headers is covered
//! by `crates/gorilla-tsc/tests/bounded_alloc.rs`.
#![no_main]

use gorilla_tsc::{Decoded, decode, decode_any, decode_pairs, read_header};
use libfuzzer_sys::fuzz_target;

const MAX_ITEMS: u64 = 1 << 22;

fuzz_target!(|data: &[u8]| {
    let Ok((header, _)) = read_header(data) else { return };
    if header.count > MAX_ITEMS {
        return;
    }
    // Typed decoders also exercise the type-mismatch checks.
    let _ = decode::<i32>(data);
    let _ = decode::<i64>(data);
    let _ = decode::<f32>(data);
    let _ = decode::<f64>(data);
    if let Ok(d) = decode_any(data) {
        let n = match d {
            Decoded::I32(v) => v.len(),
            Decoded::I64(v) => v.len(),
            Decoded::F32(v) => v.len(),
            Decoded::F64(v) => v.len(),
        };
        // A successful decode honours the header's count.
        assert_eq!(n as u64, header.count);
    }
    // Pair streams: the input as the timestamps, an empty value stream.
    let _ = decode_pairs::<i64, f64>(data, &[]);
});
