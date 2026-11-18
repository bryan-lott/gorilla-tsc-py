//! Decoding arbitrary bytes must never panic: it returns a value or a `DecodeError`.
#![no_main]

use gorilla_tsc::{Decoded, decode, decode_any, decode_pairs, read_header};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = read_header(data);
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
        assert_eq!(Some(n as u64), read_header(data).ok().map(|(h, _)| h.count));
    }
    // Pair streams: split the input in two.
    let mid = data.len() / 2;
    let _ = decode_pairs::<i64, f64>(&data[..mid], &data[mid..]);
});
