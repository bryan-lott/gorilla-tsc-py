//! External baselines (generic byte compressors and pcodec).

/// Blosc-style byte shuffle: byte plane j holds byte j of every value.
#[must_use]
pub fn shuffle(v: &[f64]) -> Vec<u8> {
    let n = v.len();
    let mut out = vec![0u8; n * 8];
    for (i, x) in v.iter().enumerate() {
        for (j, b) in x.to_le_bytes().iter().enumerate() {
            out[j * n + i] = *b;
        }
    }
    out
}

#[must_use]
pub fn unshuffle(b: &[u8]) -> Vec<f64> {
    let n = b.len() / 8;
    (0..n)
        .map(|i| {
            let mut a = [0u8; 8];
            for (j, x) in a.iter_mut().enumerate() {
                *x = b[j * n + i];
            }
            f64::from_le_bytes(a)
        })
        .collect()
}

#[must_use]
pub fn raw(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[must_use]
pub fn unraw(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

#[must_use]
pub fn zstd_c(b: &[u8], level: i32) -> Vec<u8> {
    zstd::bulk::compress(b, level).unwrap()
}

#[must_use]
pub fn zstd_d(b: &[u8], cap: usize) -> Vec<u8> {
    zstd::bulk::decompress(b, cap).unwrap()
}

#[must_use]
pub fn xz_c(b: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut e = xz2::write::XzEncoder::new(Vec::new(), 9);
    e.write_all(b).unwrap();
    e.finish().unwrap()
}

#[must_use]
pub fn xz_d(b: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut d = xz2::read::XzDecoder::new(b);
    let mut out = Vec::new();
    d.read_to_end(&mut out).unwrap();
    out
}

#[must_use]
pub fn pco_c(v: &[f64]) -> Vec<u8> {
    pco::standalone::simpler_compress(v, pco::DEFAULT_COMPRESSION_LEVEL).unwrap()
}

#[must_use]
pub fn pco_d(b: &[u8]) -> Vec<f64> {
    pco::standalone::simple_decompress::<f64>(b).unwrap()
}

/// Independent blocks, each `u32 length + compressed bytes`.
#[must_use]
pub fn blocked(v: &[f64], block: usize, f: impl Fn(&[f64]) -> Vec<u8>) -> Vec<u8> {
    let mut out = Vec::new();
    for c in v.chunks(block) {
        let b = f(c);
        // 2-byte length suffices for blocks of 1000 f64 (<= 8 kB + overhead)
        out.extend((u16::try_from(b.len()).unwrap()).to_le_bytes());
        out.extend(b);
    }
    out
}

#[must_use]
pub fn unblocked(mut b: &[u8], f: impl Fn(&[u8]) -> Vec<f64>) -> Vec<f64> {
    let mut out = Vec::new();
    while !b.is_empty() {
        let len = usize::from(u16::from_le_bytes([b[0], b[1]]));
        out.extend(f(&b[2..2 + len]));
        b = &b[2 + len..];
    }
    out
}
