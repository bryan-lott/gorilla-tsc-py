//! Decoding must not reserve memory the input cannot back: a header may
//! claim billions of items, but a block decoder only reserves what its
//! remaining input could plausibly hold (found by `fuzz/` target `decode`).

use gorilla_tsc::{Mode, SampleType, decode, decode_any};
use peak_alloc::PeakAlloc;

#[global_allocator]
static PEAK: PeakAlloc = PeakAlloc;

/// Fuzzer input: an f64 decimal stream claiming a huge block (it reserved 3.2 GB).
const HUGE_BLOCK: [u8; 56] = [
    0x47, 0x01, 0x35, 0xf7, 0xf7, 0xf7, 0x77, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0x09, 0x0a, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7,
    0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7,
    0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0xf7, 0x3a, 0x21,
];

fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(u8::try_from(v & 0x7F).unwrap() | 0x80);
        v >>= 7;
    }
    out.push(u8::try_from(v).unwrap());
}

/// A header claiming one block of `2^31` items, then `payload`.
fn lying_stream(sample_type: u8, mode_code: u8, payload: &[u8]) -> Vec<u8> {
    let mut b = vec![b'G', 1, (sample_type << 4) | mode_code];
    varint(1 << 31, &mut b);
    varint(1 << 31, &mut b);
    b.extend_from_slice(payload);
    b
}

/// Peak extra heap while `f` runs, in whole MiB.
fn peak_mb_while<F: FnOnce()>(f: F) -> usize {
    PEAK.reset_peak_usage();
    let before = PEAK.current_usage();
    f();
    PEAK.peak_usage().saturating_sub(before) >> 20
}

// One test, so no other test's allocations are counted.
#[test]
fn lying_headers_do_not_reserve_unbacked_memory() {
    let mb = peak_mb_while(|| assert!(decode_any(&HUGE_BLOCK).is_err()));
    assert!(mb < 16, "fuzzer input reserved {mb} MB");

    let mut tried = 0;
    for t in 0..4u8 {
        for code in 0..16u8 {
            if gorilla_tsc::read_header(&lying_stream(t, code, &[])).is_err() {
                continue; // not a mode
            }
            tried += 1;
            for fill in [0xAAu8, 0x55, 0xF7] {
                let s = lying_stream(t, code, &[fill; 64]);
                let mb = peak_mb_while(|| {
                    let _ = decode_any(&s);
                });
                assert!(mb < 16, "type {t} mode {code} fill {fill:#x}: reserved {mb} MB");
            }
        }
    }
    assert_eq!(tried, 4 * Mode::all().len(), "every sample type x mode");
    // Sanity: the instrument sees real allocations.
    let big: Vec<u8> = vec![1; 32 << 20];
    let mb = peak_mb_while(|| drop(std::hint::black_box(big.clone())));
    assert!(mb >= 32, "allocator instrumentation saw only {mb} MB");
    let _ = (SampleType::F64, decode::<f64>(&[]));
}
