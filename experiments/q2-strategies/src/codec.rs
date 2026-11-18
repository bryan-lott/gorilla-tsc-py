//! Per-block candidate framework ("auto with extra candidates") and shared helpers.
//!
//! A [`Composite`] splits the input into blocks of `block` values, encodes each
//! block with every candidate, keeps the smallest and writes a fixed-width
//! selector in front of it, exactly like the library's `Mode::Auto` (which uses
//! 3 selector bits). Blocks share no state, so random access is preserved.
//! The stream starts with the same header bytes the library writes
//! (magic, version, descriptor, varint(interval), varint(count)) so sizes are
//! directly comparable with `gorilla_tsc::encode`.

use gorilla_tsc::{
    BitReader, BitWriter, BlockCodec, DecimalRice, DecodeError, DodScheme, DodXorScheme, PerItem,
    Sample, XorDodScheme, XorScheme,
};

/// A block codec that may decline a block (`enc` returns false) to save time.
pub trait Cand: Send + Sync {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool;
    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError>;
}

/// Adapter for the library's public block codecs.
pub struct Lib<C>(pub C);

impl<C: BlockCodec<u64> + Send + Sync> Cand for Lib<C> {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        self.0.encode_block(items, out);
        true
    }
    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        self.0.decode_block(n, inp, out)
    }
}

/// The five candidates `Mode::Auto` tries for f64 (FLOAT_AUTO in stream.rs).
#[must_use]
pub fn auto_cands() -> Vec<Box<dyn Cand>> {
    vec![
        Box::new(Lib(PerItem::<XorScheme<u64>>::default())),
        Box::new(Lib(PerItem::<DodScheme<u64>>::default())),
        Box::new(Lib(PerItem::<XorDodScheme<u64>>::default())),
        Box::new(Lib(PerItem::<DodXorScheme<u64>>::default())),
        Box::new(Lib(DecimalRice::<f64>::default())),
    ]
}

pub struct Composite {
    pub cands: Vec<Box<dyn Cand>>,
    pub block: usize,
}

pub fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let low = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

fn read_varint(b: &mut &[u8]) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let (&x, rest) = b.split_first().expect("truncated varint");
        *b = rest;
        v |= u64::from(x & 0x7F) << shift;
        if x & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}

/// Library-compatible header: 3 bytes + varint(interval) + varint(n).
#[must_use]
pub fn header(block: usize, n: usize) -> Vec<u8> {
    let mut h = vec![b'G', 1, 0x30];
    varint(&mut h, block as u64);
    varint(&mut h, n as u64);
    h
}

/// Parse [`header`]: returns (block, n, payload).
#[must_use]
pub fn parse_header(bytes: &[u8]) -> (usize, usize, &[u8]) {
    let mut b = &bytes[3..];
    let block = read_varint(&mut b) as usize;
    let n = read_varint(&mut b) as usize;
    (block, n, b)
}

impl Composite {
    #[must_use]
    pub fn new(cands: Vec<Box<dyn Cand>>, block: usize) -> Self {
        Self { cands, block }
    }

    #[must_use]
    pub fn sel_bits(&self) -> u32 {
        (self.cands.len().max(1) - 1).bit_width()
    }

    /// Encode; also returns how many blocks each candidate won.
    #[must_use]
    pub fn encode(&self, v: &[f64]) -> (Vec<u8>, Vec<usize>) {
        let words: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
        let mut out = BitWriter::new();
        let mut wins = vec![0; self.cands.len()];
        let sb = self.sel_bits();
        let block = if self.block == 0 { words.len().max(1) } else { self.block };
        for blk in words.chunks(block) {
            let mut best: Option<(usize, BitWriter)> = None;
            for (i, c) in self.cands.iter().enumerate() {
                let mut w = BitWriter::new();
                if c.enc(blk, &mut w) && best.as_ref().is_none_or(|b| w.bit_len() < b.1.bit_len()) {
                    best = Some((i, w));
                }
            }
            let (i, w) = best.expect("no candidate accepted the block");
            wins[i] += 1;
            out.write(i as u64, sb);
            out.append(&w);
        }
        let mut bytes = header(block, v.len());
        bytes.extend(out.finish());
        (bytes, wins)
    }

    pub fn decode(&self, bytes: &[u8]) -> Result<Vec<f64>, DecodeError> {
        let (block, n, payload) = parse_header(bytes);
        let mut inp = BitReader::new(payload);
        let mut words = Vec::with_capacity(n);
        let sb = self.sel_bits();
        let mut left = n;
        while left > 0 {
            let m = left.min(block);
            let i = inp.read(sb)? as usize;
            let c = self.cands.get(i).ok_or(DecodeError::Corrupt("bad selector"))?;
            c.dec(m, &mut inp, &mut words)?;
            left -= m;
        }
        Ok(words.into_iter().map(f64::from_bits).collect())
    }
}

// ---------------------------------------------------------------------------
// Decimal scaling helpers (a copy of the library's private logic in decimal.rs,
// needed because `scale`/`exception_front` are not exported).

pub const MAX_SCALE: i32 = 18;
pub const K_BITS: u32 = 6;

fn scale_range(v: f64) -> Option<(i32, i32)> {
    let ok = |k: i32| v.to_scaled(k).is_some();
    let seed = (0..=MAX_SCALE).chain((-MAX_SCALE..0).rev()).find(|&k| ok(k))?;
    let mut lo = seed;
    while lo > -MAX_SCALE && ok(lo - 1) {
        lo -= 1;
    }
    let (mut hi, mut fail) = (seed, MAX_SCALE + 1);
    while fail - hi > 1 {
        let mid = hi + (fail - hi) / 2;
        if ok(mid) {
            hi = mid;
        } else {
            fail = mid;
        }
    }
    Some((lo, hi))
}

/// Pareto front of (exponent, exception count), increasing k, decreasing exceptions.
#[must_use]
pub fn exception_front(items: &[u64]) -> Vec<(i32, usize)> {
    let span = (2 * MAX_SCALE + 1) as usize;
    let index = |k: i32| (k + MAX_SCALE) as usize;
    let mut starts = vec![0isize; span + 1];
    for &w in items {
        if let Some((lo, hi)) = scale_range(f64::from_bits(w)) {
            starts[index(lo)] += 1;
            starts[index(hi) + 1] -= 1;
        }
    }
    let mut front = Vec::new();
    let mut best = usize::MAX;
    let mut scaling = 0isize;
    for k in -MAX_SCALE..=MAX_SCALE {
        scaling += starts[index(k)];
        let ex = items.len() - scaling as usize;
        if ex < best {
            best = ex;
            front.push((k, ex));
        }
    }
    front
}

pub struct Scaled {
    pub k: i32,
    pub ints: Vec<i64>,
    pub exceptions: Vec<(usize, u64)>,
}

/// Scale by 10^k; exceptions are filled with the previous integer.
#[must_use]
pub fn scale(items: &[u64], k: i32) -> Scaled {
    let mut prev = 0i64;
    let mut ints = Vec::with_capacity(items.len());
    let mut exceptions = Vec::new();
    for (i, &w) in items.iter().enumerate() {
        match f64::from_bits(w).to_scaled(k) {
            Some(n) => prev = n,
            None => exceptions.push((i, w)),
        }
        ints.push(prev);
    }
    Scaled { k, ints, exceptions }
}

#[must_use]
pub fn bit_len(n: usize) -> u32 {
    n.bit_width()
}

/// Bits per exception for a block of `n`.
#[must_use]
pub fn per_exception(n: usize) -> usize {
    bit_len(n.saturating_sub(1)) as usize + 64
}

/// Same layout as the library: zigzag(k) in 6 bits, count, (pos, raw) pairs.
pub fn write_scaled_header(s: &Scaled, n: usize, out: &mut BitWriter) {
    let kz = ((i64::from(s.k) << 1) ^ (i64::from(s.k) >> 63)) as u64;
    out.write(kz, K_BITS);
    out.write(s.exceptions.len() as u64, bit_len(n));
    for &(pos, raw) in &s.exceptions {
        out.write(pos as u64, bit_len(n.saturating_sub(1)));
        out.write(raw, 64);
    }
}

/// Inverse of [`write_scaled_header`]: returns (k, exceptions).
pub fn read_scaled_header(n: usize, inp: &mut BitReader<'_>) -> Result<(i32, Vec<(usize, u64)>), DecodeError> {
    let kz = inp.read(K_BITS)?;
    let k = ((kz >> 1) as i64 ^ -((kz & 1) as i64)) as i32;
    let count = inp.read(bit_len(n))? as usize;
    let mut ex = Vec::with_capacity(count);
    for _ in 0..count {
        let pos = inp.read(bit_len(n.saturating_sub(1)))? as usize;
        ex.push((pos, inp.read(64)?));
    }
    Ok((k, ex))
}

/// Rebuild float words from scaled ints and exceptions.
pub fn unscale(ints: &[i64], k: i32, exceptions: &[(usize, u64)], out: &mut Vec<u64>) {
    let mut exc = exceptions.iter().peekable();
    for (i, &n) in ints.iter().enumerate() {
        match exc.next_if(|&&(pos, _)| pos == i) {
            Some(&(_, raw)) => out.push(raw),
            None => out.push(f64::from_scaled(n, k).to_bits()),
        }
    }
}

#[must_use]
pub fn zz(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[must_use]
pub fn unzz(z: u64) -> i64 {
    (z >> 1) as i64 ^ -((z & 1) as i64)
}

/// Bits `w` takes when length-prefixed with 7 bits.
pub fn write_prefixed(out: &mut BitWriter, v: u64) {
    out.write(u64::from(v.bit_width()), 7);
    out.write(v, v.bit_width());
}

pub fn read_prefixed(inp: &mut BitReader<'_>) -> Result<u64, DecodeError> {
    let l = inp.read(7)? as u32;
    if l > 64 {
        return Err(DecodeError::Corrupt("prefixed length"));
    }
    inp.read(l)
}
