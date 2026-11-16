//! Idea 4: structure detection for binary data.
//!
//! 1. Stride: for s in 1..=64, sum over the s byte-columns of the order-0
//!    entropy of min(raw, delta-at-lag-s) bytes (on a sample). Smallest s
//!    within 2% of the best wins; "structured" if it beats s=1 by >10%.
//! 2. Fields: dynamic programming over the record, each field one of
//!    u8/i16/i32/f32/i64/f64 (little endian) or "bytes"; cost = actual
//!    gorilla-tsc `Auto` size on a sample of records (bytes: xz -6 size).
//! 3. Each field column of the whole file is coded accordingly; the tail
//!    (len % stride) is stored raw. No structure -> whole file via the CM coder.
//!
//! Also: an x86 BCJ (E8/E9 call/jmp) transform for executables.

use crate::util::*;
use gorilla_tsc::{CheckpointInterval, Config, Mode, decode, encode};

fn cfg() -> Config {
    Config::new(Mode::Auto).with_interval(CheckpointInterval::new(1000).unwrap())
}

fn h0(counts: &[u32; 256], n: u32) -> f64 {
    let n = f64::from(n.max(1));
    counts.iter().filter(|&&c| c > 0).map(|&c| {
        let p = f64::from(c) / n;
        -f64::from(c) * p.log2()
    }).sum()
}

pub const MAX_STRIDE: usize = 256;
/// Largest stride for which fields are typed (else: CM with stride contexts).
pub const MAX_FIELD_STRIDE: usize = 64;

pub fn detect_stride(data: &[u8]) -> (usize, Vec<f64>) {
    let sample = &data[..data.len().min(1 << 22)];
    let mut costs = vec![f64::INFINITY; MAX_STRIDE + 1];
    for s in 1..=MAX_STRIDE {
        if sample.len() < s * 64 {
            break;
        }
        let mut total = 0.0;
        for j in 0..s {
            let mut raw = [0u32; 256];
            let mut del = [0u32; 256];
            let mut n = 0;
            let mut i = j + s;
            while i < sample.len() {
                raw[sample[i] as usize] += 1;
                del[sample[i].wrapping_sub(sample[i - s]) as usize] += 1;
                n += 1;
                i += s;
            }
            total += h0(&raw, n).min(h0(&del, n));
        }
        costs[s] = total;
    }
    let best = costs[1..].iter().copied().fold(f64::INFINITY, f64::min);
    let s = (1..=MAX_STRIDE).find(|&s| costs[s] <= best * 1.02).unwrap_or(1);
    if s > 1 && costs[s] < costs[1] * 0.9 { (s, costs) } else { (1, costs) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ft {
    U8,
    I16,
    I32,
    F32,
    I64,
    F64,
    Bytes,
}

impl Ft {
    fn width(self) -> usize {
        match self {
            Ft::U8 | Ft::Bytes => 1,
            Ft::I16 => 2,
            Ft::I32 | Ft::F32 => 4,
            Ft::I64 | Ft::F64 => 8,
        }
    }
    fn code(self) -> u8 {
        self as u8
    }
    fn from(c: u8) -> Ft {
        [Ft::U8, Ft::I16, Ft::I32, Ft::F32, Ft::I64, Ft::F64, Ft::Bytes][c as usize]
    }
    pub fn name(self) -> &'static str {
        ["u8", "i16", "i32", "f32", "i64", "f64", "bytes"][self as usize]
    }
}

fn column(data: &[u8], stride: usize, off: usize, w: usize, nrec: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(nrec * w);
    for r in 0..nrec {
        v.extend_from_slice(&data[r * stride + off..r * stride + off + w]);
    }
    v
}

fn encode_field(col: &[u8], ft: Ft) -> Vec<u8> {
    match ft {
        Ft::U8 => encode(&col.iter().map(|&b| i32::from(b)).collect::<Vec<_>>(), cfg()),
        Ft::I16 => encode(
            &col.chunks_exact(2).map(|c| i32::from(i16::from_le_bytes([c[0], c[1]]))).collect::<Vec<_>>(),
            cfg(),
        ),
        Ft::I32 => encode(&col.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), cfg()),
        Ft::F32 => encode(&col.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), cfg()),
        Ft::I64 => encode(&col.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), cfg()),
        Ft::F64 => encode(&col.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), cfg()),
        Ft::Bytes => xz_compress(col, 9 | crate::util::PRESET_EXTREME),
    }
}

fn decode_field(b: &[u8], ft: Ft) -> Vec<u8> {
    match ft {
        Ft::U8 => decode::<i32>(b).unwrap().into_iter().map(|v| v as u8).collect(),
        Ft::I16 => decode::<i32>(b).unwrap().into_iter().flat_map(|v| (v as i16).to_le_bytes()).collect(),
        Ft::I32 => decode::<i32>(b).unwrap().into_iter().flat_map(i32::to_le_bytes).collect(),
        Ft::F32 => decode::<f32>(b).unwrap().into_iter().flat_map(f32::to_le_bytes).collect(),
        Ft::I64 => decode::<i64>(b).unwrap().into_iter().flat_map(i64::to_le_bytes).collect(),
        Ft::F64 => decode::<f64>(b).unwrap().into_iter().flat_map(f64::to_le_bytes).collect(),
        Ft::Bytes => xz_decompress(b),
    }
}

/// Choose the field layout of a record by DP on sampled gorilla sizes.
pub fn choose_fields(data: &[u8], stride: usize) -> Vec<Ft> {
    let nrec = (data.len() / stride).min(50_000);
    let mut best = vec![(f64::INFINITY, Vec::new()); stride + 1];
    best[0] = (0.0, Vec::new());
    for o in 0..stride {
        if !best[o].0.is_finite() {
            continue;
        }
        for ft in [Ft::U8, Ft::Bytes, Ft::I16, Ft::I32, Ft::F32, Ft::I64, Ft::F64] {
            let w = ft.width();
            if o + w > stride {
                continue;
            }
            let col = column(data, stride, o, w, nrec);
            let c = best[o].0 + encode_field(&col, ft).len() as f64;
            if c < best[o + w].0 {
                let mut v = best[o].1.clone();
                v.push(ft);
                best[o + w] = (c, v);
            }
        }
    }
    best[stride].1.clone()
}

/// Returns (compressed, description of the detected layout).
pub fn struct_encode(data: &[u8]) -> (Vec<u8>, String) {
    let (stride, _) = detect_stride(data);
    let mut out = vec![b'S'];
    put_varint(&mut out, data.len() as u64);
    put_varint(&mut out, stride as u64);
    if stride == 1 || stride > MAX_FIELD_STRIDE {
        out.extend_from_slice(&crate::cm::compress(data, crate::cm::CmConfig { stride: stride as u32 }));
        return (out, format!("stride {stride}: no fields -> cm-stride"));
    }
    let fields = choose_fields(data, stride);
    // Mostly untyped records (opaque bytes): the CM coder with stride contexts
    // models them better than per-byte-column xz.
    let typed: usize = fields.iter().filter(|f| **f != Ft::Bytes).map(|f| f.width()).sum();
    if typed * 2 < stride {
        let mut out = vec![b'S'];
        put_varint(&mut out, data.len() as u64);
        put_varint(&mut out, 1);
        out.extend_from_slice(&crate::cm::compress(data, crate::cm::CmConfig { stride: stride as u32 }));
        return (out, format!("stride {stride}: mostly untyped bytes -> cm-stride"));
    }
    let nrec = data.len() / stride;
    out.push(fields.len() as u8);
    let mut off = 0;
    for &ft in &fields {
        out.push(ft.code());
        let col = column(data, stride, off, ft.width(), nrec);
        put_bytes(&mut out, &encode_field(&col, ft));
        off += ft.width();
    }
    out.extend_from_slice(&data[nrec * stride..]);
    let desc = format!("stride {stride}: [{}]", fields.iter().map(|f| f.name()).collect::<Vec<_>>().join(","));
    (out, desc)
}

pub fn struct_decode(inp: &[u8]) -> Vec<u8> {
    assert_eq!(inp[0], b'S');
    let mut pos = 1;
    let n = get_varint(inp, &mut pos) as usize;
    let stride = get_varint(inp, &mut pos) as usize;
    if stride == 1 || stride > MAX_FIELD_STRIDE {
        return crate::cm::decompress(&inp[pos..]);
    }
    let nf = inp[pos] as usize;
    pos += 1;
    let nrec = n / stride;
    let mut out = vec![0u8; n];
    let mut off = 0;
    for _ in 0..nf {
        let ft = Ft::from(inp[pos]);
        pos += 1;
        let col = decode_field(get_bytes(inp, &mut pos), ft);
        let w = ft.width();
        for r in 0..nrec {
            out[r * stride + off..r * stride + off + w].copy_from_slice(&col[r * w..r * w + w]);
        }
        off += w;
    }
    out[nrec * stride..].copy_from_slice(&inp[pos..]);
    out
}

/// x86 BCJ: for E8/E9 with a rel32 whose top byte is 00/FF, replace the low
/// 24 bits by (rel + position) mod 2^24. The top byte is unchanged, so the
/// decoder makes exactly the same decisions: a bijection.
pub fn bcj_x86(data: &[u8], encode: bool) -> Vec<u8> {
    let mut b = data.to_vec();
    let mut i = 0;
    while i + 5 <= b.len() {
        if (b[i] == 0xE8 || b[i] == 0xE9) && (b[i + 4] == 0 || b[i + 4] == 0xFF) {
            let v = u32::from_le_bytes([b[i + 1], b[i + 2], b[i + 3], 0]);
            let p = ((i + 5) as u32) & 0xFF_FFFF;
            let nv = if encode { v.wrapping_add(p) } else { v.wrapping_sub(p) } & 0xFF_FFFF;
            b[i + 1..i + 4].copy_from_slice(&nv.to_le_bytes()[..3]);
            i += 5;
        } else {
            i += 1;
        }
    }
    b
}
