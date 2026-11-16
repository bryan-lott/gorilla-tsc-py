//! Bitwise context-mixing compressor (own coder, lpaq1-style, simplified).
//!
//! Each byte is coded MSB-first as 8 binary decisions. For every bit a set of
//! models (order-0..6 byte contexts, word contexts, sparse contexts, optional
//! record-stride contexts, and a match model) each give a probability from an
//! adaptive counter. The probabilities are mixed in the logistic domain by two
//! small online-trained neural mixers, refined by two APM/SSE stages, and the
//! result drives a 32-bit binary arithmetic coder with 12-bit probabilities.
//!
//! This generalises gorilla-tsc's adaptive binary range coder (one `Prob` per
//! context, 11-bit, shift-5 adaptation) with: count-based adaptive learning
//! rates, many contexts at once, and learned mixing.

use std::sync::OnceLock;

// ---------------------------------------------------------------- tables

struct Tables {
    stretch: Vec<i16>, // 4096 entries: p(12 bit) -> ln(p/(1-p)) * 256, clipped to +-2047
    squash: Vec<i16>,  // 4096 entries: d + 2048 -> p(12 bit)
    recip: Vec<u32>,   // counter learning rates 65536 / (n + 1.5)
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut squash = vec![0i16; 4096];
        for (i, s) in squash.iter_mut().enumerate() {
            let d = (i as f64 - 2048.0) / 256.0;
            let v = 4096.0 / (1.0 + (-d).exp());
            *s = v.round().clamp(1.0, 4095.0) as i16;
        }
        // stretch = inverse of squash
        let mut stretch = vec![0i16; 4096];
        let mut pi = 0usize;
        for x in -2047i32..=2047 {
            let v = squash[(x + 2048) as usize] as usize;
            for j in pi..=v {
                stretch[j] = x as i16;
            }
            pi = v + 1;
        }
        for j in pi..4096 {
            stretch[j] = 2047;
        }
        let recip = (0..1024).map(|n| (65536.0 / (n as f64 + 1.5)) as u32).collect();
        Tables { stretch, squash, recip }
    })
}

#[inline]
fn squash(d: i32) -> i32 {
    let d = d.clamp(-2047, 2047);
    i32::from(tables().squash[(d + 2048) as usize])
}

#[inline]
fn stretch(p: i32) -> i32 {
    i32::from(tables().stretch[p as usize])
}

// ---------------------------------------------------------------- counters

/// Adaptive probability with a hit count: high 22 bits P(1), low 10 bits count.
#[inline]
fn ctr_p(c: u32) -> i32 {
    (c >> 20) as i32 // 12-bit
}

#[inline]
fn ctr_update(c: &mut u32, y: u32, limit: u32) {
    let n = *c & 1023;
    let p = (*c >> 10) as i64; // 22-bit
    let target = (y as i64) << 22;
    let r = i64::from(tables().recip[n as usize]);
    let np = p + (((target - p) * r) >> 16);
    let np = np.clamp(1 << 6, (1 << 22) - (1 << 6)) as u32;
    let nn = if n < limit { n + 1 } else { n };
    *c = (np << 10) | nn;
}

const CTR_INIT: u32 = (1 << 31) | 0; // p = 0.5, n = 0

// ---------------------------------------------------------------- mixer

struct Mixer {
    n: usize,
    weights: Vec<i32>,
    inputs: Vec<i32>,
    nin: usize,
    sel: usize,
    pr: i32,
    lr: i32,
}

impl Mixer {
    fn new(n: usize, sets: usize, lr: i32) -> Self {
        Mixer { n, weights: vec![(1 << 16) / 4; n * sets], inputs: vec![0; n], nin: 0, sel: 0, pr: 2048, lr }
    }
    #[inline]
    fn add(&mut self, x: i32) {
        self.inputs[self.nin] = x;
        self.nin += 1;
    }
    #[inline]
    fn mix(&mut self, sel: usize) -> i32 {
        debug_assert_eq!(self.nin, self.n);
        self.sel = sel * self.n;
        let w = &self.weights[self.sel..self.sel + self.n];
        let mut dot: i64 = 0;
        for i in 0..self.n {
            dot += i64::from(self.inputs[i]) * i64::from(w[i]);
        }
        let d = (dot >> 16) as i32;
        self.pr = squash(d);
        d.clamp(-2047, 2047)
    }
    #[inline]
    fn update(&mut self, y: u32) {
        let err = (((y as i32) << 12) - self.pr) * self.lr;
        let w = &mut self.weights[self.sel..self.sel + self.n];
        for i in 0..self.n {
            w[i] += (self.inputs[i] * err) >> 14;
        }
        self.nin = 0;
    }
}

// ---------------------------------------------------------------- APM / SSE

struct Apm {
    t: Vec<u16>,
    idx: usize,
    rate: u32,
}

impl Apm {
    fn new(n: usize, rate: u32) -> Self {
        let mut t = vec![0u16; n * 33];
        for i in 0..n {
            for j in 0..33 {
                t[i * 33 + j] = (squash((j as i32 - 16) * 128) * 16) as u16;
            }
        }
        Apm { t, idx: 0, rate }
    }
    #[inline]
    fn pp(&mut self, pr: i32, cx: usize) -> i32 {
        let s = stretch(pr) + 2048; // 0..4095
        let lo = (s >> 7) as usize; // 0..31
        let w = s & 127;
        let base = cx * 33 + lo;
        let p = (i32::from(self.t[base]) * (128 - w) + i32::from(self.t[base + 1]) * w) >> 11;
        self.idx = base + ((w >> 6) as usize);
        p.clamp(1, 4095)
    }
    #[inline]
    fn update(&mut self, y: u32) {
        let g = ((y << 16) + (y << self.rate) - y - y) as i32;
        let t = &mut self.t[self.idx];
        *t = (i32::from(*t) + ((g - i32::from(*t)) >> self.rate)) as u16;
    }
}

// ---------------------------------------------------------------- predictor

/// Model configuration.
#[derive(Clone, Copy, Debug)]
pub struct CmConfig {
    /// Record stride (bytes) for column contexts; 0 = none.
    pub stride: u32,
}

const N_HASHED: usize = 10; // o2 o3 o4 o6 word1 word2 sparse1 sparse2 col1 col2

struct Predictor {
    cfg: CmConfig,
    // state
    c0: u32,  // partial byte with leading 1
    c4: u32,  // last 4 bytes
    c8: u32,  // bytes 5..8
    bpos: u32,
    pos: usize,
    buf: Vec<u8>,
    word: u32,
    word_prev: u32,
    // models
    t0: Vec<u32>,
    t1: Vec<u32>,
    th: Vec<u32>,
    hbits: u32,
    base: [usize; N_HASHED],
    idx: [usize; N_HASHED + 2],
    // match model
    mm_table: Vec<u32>,
    mm_bits: u32,
    mm_ptr: usize,
    mm_len: u32,
    mm_byte: u32,
    mm_ctr: Vec<u32>,
    mm_idx: usize,
    // mixing
    m1: Mixer,
    m2: Mixer,
    a1: Apm,
    a2: Apm,
    pr_mix: i32,
    pr: i32,
    lim: u32,
}

fn hash(a: u32, b: u32) -> u32 {
    let h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_add(0x7F4A_7C15).wrapping_mul(0x85EB_CA6B);
    h ^ (h >> 15)
}

impl Predictor {
    fn new(cfg: CmConfig, len_hint: usize) -> Self {
        let lb = (usize::BITS - len_hint.max(1).leading_zeros()) as u32;
        let hbits = (lb + 2).clamp(18, 24); // slots per hashed model
        let mm_bits = (lb).clamp(16, 22);
        let n_in = 2 + 2 * N_HASHED + 1 + 1;
        let mut p = Predictor {
            cfg,
            c0: 1,
            c4: 0,
            c8: 0,
            bpos: 0,
            pos: 0,
            buf: Vec::with_capacity(len_hint),
            word: 0,
            word_prev: 0,
            t0: vec![CTR_INIT; 256],
            t1: vec![CTR_INIT; 65536],
            th: vec![CTR_INIT; (N_HASHED << hbits) * 2],
            hbits,
            base: [0; N_HASHED],
            idx: [0; N_HASHED + 2],
            mm_table: vec![0; 1 << mm_bits],
            mm_bits,
            mm_ptr: 0,
            mm_len: 0,
            mm_byte: 0,
            mm_ctr: vec![CTR_INIT; 64 * 2],
            mm_idx: 0,
            m1: Mixer::new(n_in, 256, 8),
            m2: Mixer::new(n_in, 4 * 16 * 8, 8),
            a1: Apm::new(256, 7),
            a2: Apm::new(65536, 7),
            pr_mix: 2048,
            pr: 2048,
            lim: 12,
        };
        p.byte_update();
        p.predict();
        p
    }

    fn back(&self, k: usize) -> u32 {
        if self.pos >= k { u32::from(self.buf[self.pos - k]) } else { 0 }
    }

    /// Recompute byte-level context hashes.
    fn byte_update(&mut self) {
        let c1 = self.c4 & 0xff;
        let s = self.cfg.stride as usize;
        let mut h = [0u32; N_HASHED];
        h[0] = hash(1, self.c4 & 0xffff);
        h[1] = hash(2, self.c4 & 0xff_ffff);
        h[2] = hash(3, self.c4);
        h[3] = hash(hash(4, self.c4), self.c8 & 0xffff);
        h[4] = hash(5, self.word);
        h[5] = hash(hash(6, self.word), self.word_prev);
        h[6] = hash(7, self.c4 & 0xffff_ff00);
        h[7] = hash(8, (self.c4 & 0xff00_0000) | ((self.c8 & 0xff) << 8) | c1);
        if s > 1 {
            let ph = (self.pos % s) as u32;
            let b1 = self.back(s);
            let b2 = self.back(2 * s);
            h[8] = hash(hash(9, ph), b1 | (b2 << 8));
            h[9] = hash(hash(10, ph), b1 | (c1 << 8) | (self.back(s - 1) << 16));
        } else {
            h[8] = hash(9, self.c4 & 0xff00ff);
            h[9] = hash(10, self.c4 & 0xffff0000);
        }
        let shift = 32 - (self.hbits - 8);
        for (i, hv) in h.iter().enumerate() {
            self.base[i] = (i << self.hbits) + (((hv >> shift) as usize) << 8);
        }
        // match model
        const MINLEN: usize = 6;
        if self.mm_len > 0 && self.mm_ptr < self.pos && u32::from(self.buf[self.mm_ptr]) == c1 {
            self.mm_len = (self.mm_len + 1).min(63);
            self.mm_ptr += 1;
        } else {
            self.mm_len = 0;
        }
        if self.pos >= MINLEN {
            let mut hh: u32 = 0;
            for k in 1..=MINLEN {
                hh = hh.wrapping_mul(0x2F0F_3A75).wrapping_add(u32::from(self.buf[self.pos - k]) + 1);
            }
            let slot = (hh.wrapping_mul(0x9E37_79B1) >> (32 - self.mm_bits)) as usize;
            if self.mm_len == 0 {
                let cand = self.mm_table[slot] as usize;
                if cand > 0 {
                    let mut l = 0;
                    while l < 32 && cand > l && self.buf[cand - 1 - l] == self.buf[self.pos - 1 - l] {
                        l += 1;
                    }
                    if l >= MINLEN {
                        self.mm_len = l as u32;
                        self.mm_ptr = cand;
                    }
                }
            }
            self.mm_table[slot] = self.pos as u32;
        }
        self.mm_byte = if self.mm_len > 0 { u32::from(self.buf[self.mm_ptr]) | 0x100 } else { 0 };
    }

    fn predict(&mut self) {
        let st = |c: u32| stretch(ctr_p(c));
        let c0 = self.c0 as usize;
        self.idx[0] = c0;
        self.idx[1] = (((self.c4 & 0xff) as usize) << 8) | c0;
        self.m1.add(st(self.t0[self.idx[0]]));
        self.m2.add(st(self.t0[self.idx[0]]));
        let v = st(self.t1[self.idx[1]]);
        self.m1.add(v);
        self.m2.add(v);
        for i in 0..N_HASHED {
            let ix = (self.base[i] + c0) * 2;
            self.idx[i + 2] = ix;
            let v = st(self.th[ix]);
            self.m1.add(v);
            self.m2.add(v);
            let v = st(self.th[ix + 1]);
            self.m1.add(v);
            self.m2.add(v);
        }
        // match model input
        let mut lenb = 0usize;
        if self.mm_byte != 0 {
            let expected_prefix = (self.mm_byte | 0x100) >> (8 - self.bpos);
            let mb = self.mm_byte & 0xff;
            if expected_prefix == (self.c0 | 0) && ((mb | 0x100) >> (8 - self.bpos)) == self.c0 {
                let bit = (mb >> (7 - self.bpos)) & 1;
                self.mm_idx = (self.mm_len.min(63) as usize) * 2 + bit as usize;
                let v = st(self.mm_ctr[self.mm_idx]);
                self.m1.add(v);
                self.m2.add(v);
                lenb = if self.mm_len < 16 { 1 } else if self.mm_len < 32 { 2 } else { 3 };
            } else {
                self.mm_byte = 0;
                self.mm_idx = usize::MAX;
                self.m1.add(0);
                self.m2.add(0);
            }
        } else {
            self.mm_idx = usize::MAX;
            self.m1.add(0);
            self.m2.add(0);
        }
        self.m1.add(256);
        self.m2.add(256);
        let ph = if self.cfg.stride > 1 { (self.pos % self.cfg.stride as usize).min(15) } else { 0 };
        let d1 = self.m1.mix(c0);
        let d2 = self.m2.mix((lenb * 16 + ph) * 8 + self.bpos as usize);
        self.pr_mix = squash((d1 + d2) >> 1);
        let p1 = self.a1.pp(self.pr_mix, c0);
        let p2 = self.a2.pp(self.pr_mix, (((self.c4 & 0xff) as usize) << 8) | c0);
        self.pr = ((self.pr_mix + p1 + 2 * p2 + 2) >> 2).clamp(1, 4095);
    }

    fn update(&mut self, y: u32) {
        ctr_update(&mut self.t0[self.idx[0]], y, 60);
        ctr_update(&mut self.t1[self.idx[1]], y, 250);
        let lim = self.lim;
        for i in 0..N_HASHED {
            ctr_update(&mut self.th[self.idx[i + 2]], y, lim);
            ctr_update(&mut self.th[self.idx[i + 2] + 1], y, 1023);
        }
        if self.mm_idx != usize::MAX {
            ctr_update(&mut self.mm_ctr[self.mm_idx], y, 1023);
        }
        self.m1.update(y);
        self.m2.update(y);
        self.a1.update(y);
        self.a2.update(y);
        self.c0 = (self.c0 << 1) | y;
        self.bpos += 1;
        if self.bpos == 8 {
            let c = (self.c0 & 0xff) as u8;
            self.buf.push(c);
            self.pos += 1;
            self.c8 = (self.c8 << 8) | (self.c4 >> 24);
            self.c4 = (self.c4 << 8) | u32::from(c);
            let lc = c.to_ascii_lowercase();
            if lc.is_ascii_lowercase() || c >= 128 {
                self.word = (self.word ^ u32::from(lc)).wrapping_mul(0x0100_0193);
            } else if self.word != 0 {
                self.word_prev = self.word;
                self.word = 0;
            }
            self.c0 = 1;
            self.bpos = 0;
            self.byte_update();
        }
        self.predict();
    }

    #[inline]
    fn p(&self) -> u32 {
        self.pr as u32
    }
}

// ---------------------------------------------------------------- arithmetic coder

struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    #[inline]
    fn code(&mut self, y: u32, p: u32) {
        let xmid = self.x1 + ((u64::from(self.x2 - self.x1) * u64::from(p)) >> 12) as u32;
        if y == 1 { self.x2 = xmid } else { self.x1 = xmid + 1 }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }
    fn flush(&mut self) {
        let b = self.x1.to_be_bytes();
        self.out.extend_from_slice(&b);
    }
}

struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    inp: &'a [u8],
    i: usize,
}

impl<'a> Decoder<'a> {
    fn next(&mut self) -> u32 {
        let b = self.inp.get(self.i).copied().unwrap_or(0);
        self.i += 1;
        u32::from(b)
    }
    #[inline]
    fn code(&mut self, p: u32) -> u32 {
        let xmid = self.x1 + ((u64::from(self.x2 - self.x1) * u64::from(p)) >> 12) as u32;
        let y = u32::from(self.x <= xmid);
        if y == 1 { self.x2 = xmid } else { self.x1 = xmid + 1 }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.next();
        }
        y
    }
}

// ---------------------------------------------------------------- API

/// Compress `data`. Format: varint length, stride byte(s), coded bits.
pub fn compress(data: &[u8], cfg: CmConfig) -> Vec<u8> {
    let mut out = Vec::new();
    crate::util::put_varint(&mut out, data.len() as u64);
    crate::util::put_varint(&mut out, u64::from(cfg.stride));
    let mut pr = Predictor::new(cfg, data.len());
    let mut enc = Encoder { x1: 0, x2: u32::MAX, out };
    for &c in data {
        for i in (0..8).rev() {
            let y = u32::from((c >> i) & 1);
            enc.code(y, pr.p());
            pr.update(y);
        }
    }
    enc.flush();
    enc.out
}

pub fn decompress(inp: &[u8]) -> Vec<u8> {
    let mut pos = 0;
    let n = crate::util::get_varint(inp, &mut pos) as usize;
    let stride = crate::util::get_varint(inp, &mut pos) as u32;
    let cfg = CmConfig { stride };
    let mut pr = Predictor::new(cfg, n);
    let mut dec = Decoder { x1: 0, x2: u32::MAX, x: 0, inp, i: pos };
    for _ in 0..4 {
        dec.x = (dec.x << 8) | dec.next();
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut c = 0u32;
        for _ in 0..8 {
            let y = dec.code(pr.p());
            pr.update(y);
            c = (c << 1) | y;
        }
        out.push(c as u8);
    }
    out
}
