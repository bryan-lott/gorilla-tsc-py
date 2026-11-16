//! Small serialisation helpers and the generic byte back ends.

use std::io::{Read, Write};

pub const PRESET_EXTREME: u32 = 0x8000_0000;

pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub fn get_varint(inp: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut s = 0;
    loop {
        let b = inp[*pos];
        *pos += 1;
        v |= u64::from(b & 0x7f) << s;
        if b < 0x80 {
            return v;
        }
        s += 7;
    }
}

pub fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

pub fn get_bytes<'a>(inp: &'a [u8], pos: &mut usize) -> &'a [u8] {
    let n = get_varint(inp, pos) as usize;
    let s = &inp[*pos..*pos + n];
    *pos += n;
    s
}

pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

pub fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Generic back end for leftover byte streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Xz,
    Zstd,
    Cm,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Xz => "xz",
            Backend::Zstd => "zstd",
            Backend::Cm => "cm",
        }
    }

    pub fn compress(self, data: &[u8]) -> Vec<u8> {
        match self {
            Backend::Xz => xz_compress(data, 9 | crate::util::PRESET_EXTREME),
            Backend::Zstd => zstd::bulk::compress(data, 19).unwrap(),
            Backend::Cm => crate::cm::compress(data, crate::cm::CmConfig { stride: 0 }),
        }
    }

    pub fn decompress(self, data: &[u8]) -> Vec<u8> {
        match self {
            Backend::Xz => xz_decompress(data),
            Backend::Zstd => {
                let mut d = zstd::stream::Decoder::new(data).unwrap();
                let mut v = Vec::new();
                d.read_to_end(&mut v).unwrap();
                v
            }
            Backend::Cm => crate::cm::decompress(data),
        }
    }
}

pub fn xz_compress(data: &[u8], preset: u32) -> Vec<u8> {
    // raw LZMA2 stream with a minimal container: the xz2 "easy encoder" adds
    // ~60 bytes of xz framing, which matters for tiny sub-streams; that's fine,
    // it is counted honestly.
    let mut e = xz2::write::XzEncoder::new(Vec::new(), preset);
    e.write_all(data).unwrap();
    e.finish().unwrap()
}

pub fn xz_decompress(data: &[u8]) -> Vec<u8> {
    let mut d = xz2::read::XzDecoder::new(data);
    let mut v = Vec::new();
    d.read_to_end(&mut v).unwrap();
    v
}

/// Gorilla-tsc `Auto` on i64 with checkpoint interval 1000.
pub fn gorilla_i64(v: &[i64]) -> Vec<u8> {
    use gorilla_tsc::{CheckpointInterval, Config, Mode, encode};
    encode(v, Config::new(Mode::Auto).with_interval(CheckpointInterval::new(1000).unwrap()))
}

pub fn ungorilla_i64(b: &[u8]) -> Vec<i64> {
    gorilla_tsc::decode::<i64>(b).expect("gorilla decode")
}

/// A list of byte sections that is compressed by a back end as one blob.
#[derive(Default)]
pub struct Blob {
    pub sections: Vec<Vec<u8>>,
}

impl Blob {
    pub fn push(&mut self, s: Vec<u8>) -> usize {
        self.sections.push(s);
        self.sections.len() - 1
    }
    /// Lengths first, then all section bodies (keeps like data together).
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_varint(&mut out, self.sections.len() as u64);
        for s in &self.sections {
            put_varint(&mut out, s.len() as u64);
        }
        for s in &self.sections {
            out.extend_from_slice(s);
        }
        out
    }
    pub fn deserialize(b: &[u8]) -> Vec<Vec<u8>> {
        let mut pos = 0;
        let n = get_varint(b, &mut pos) as usize;
        let lens: Vec<usize> = (0..n).map(|_| get_varint(b, &mut pos) as usize).collect();
        let mut out = Vec::with_capacity(n);
        for l in lens {
            out.push(b[pos..pos + l].to_vec());
            pos += l;
        }
        out
    }
}

/// Fixed-width little-endian ids (width chosen from the max id).
pub fn pack_ids(ids: &[u32]) -> Vec<u8> {
    let max = ids.iter().copied().max().unwrap_or(0);
    let w = if max < 256 { 1 } else if max < 65536 { 2 } else if max < (1 << 24) { 3 } else { 4 };
    let mut out = vec![w as u8];
    for &id in ids {
        out.extend_from_slice(&id.to_le_bytes()[..w]);
    }
    out
}

pub fn unpack_ids(b: &[u8]) -> Vec<u32> {
    let w = b[0] as usize;
    b[1..]
        .chunks_exact(w)
        .map(|c| {
            let mut a = [0u8; 4];
            a[..w].copy_from_slice(c);
            u32::from_le_bytes(a)
        })
        .collect()
}
