//! B5: cross-block context for the adaptive residual coder.
//!
//! A re-implementation of the library's `AdaptiveBlock` binarisation and LZMA
//! range coder with the probability table exposed, so that the model can
//! (a) start fresh in every block (= the library), (b) carry over between
//! blocks, (c) run as a single range-coded stream without per-block flushes,
//! or (d) start each block from a fixed prior trained on *other* datasets
//! (keeps random access).
//!
//! The residuals coded are those of a decimal-scaled block (library exponent
//! search, Delta order 0/1/2); per block the coder is Rice or adaptive,
//! whichever is smaller.

use std::sync::Arc;

use gorilla_tsc::{BitReader, BitWriter, DecodeError, RiceBlock};

use crate::codec::{
    Cand, exception_front, header, parse_header, per_exception, read_prefixed, read_scaled_header, scale, unscale,
    write_prefixed, write_scaled_header,
};

const PROB_BITS: u32 = 11;
const TOP: u32 = 1 << 24;
const CLASSES: usize = 4;
const ZERO: usize = 0;
const SIGN: usize = CLASSES;
const LEN: usize = 2 * CLASSES; // [class][65]
const MANT: usize = LEN + CLASSES * 65; // [len][3]
pub const NPROB: usize = MANT + 65 * 3;

#[derive(Clone)]
pub struct Model {
    pub p: Vec<u16>,
    pub shift: u32,
}

impl Model {
    #[must_use]
    pub fn fresh(shift: u32) -> Self {
        Self { p: vec![1 << (PROB_BITS - 1); NPROB], shift }
    }
}

/// Where coded bits go: a range coder, or a counter (for training priors).
pub trait Sink {
    fn bit(&mut self, m: &mut Model, ctx: usize, bit: bool);
    fn direct(&mut self, v: u64, n: u32);
}

pub struct Enc<'a> {
    out: &'a mut BitWriter,
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    first: bool,
}

impl<'a> Enc<'a> {
    pub fn new(out: &'a mut BitWriter) -> Self {
        Self { out, low: 0, range: u32::MAX, cache: 0, cache_size: 1, first: true }
    }
    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }
    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            let mut byte = self.cache;
            loop {
                let b = byte.wrapping_add(carry);
                if self.first {
                    self.first = false;
                } else {
                    self.out.write(u64::from(b), 8);
                }
                byte = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }
    pub fn finish(mut self) {
        for _ in 0..5 {
            self.shift_low();
        }
    }
}

impl Sink for Enc<'_> {
    fn bit(&mut self, m: &mut Model, ctx: usize, bit: bool) {
        let p = &mut m.p[ctx];
        let bound = (self.range >> PROB_BITS) * u32::from(*p);
        if bit {
            self.low += u64::from(bound);
            self.range -= bound;
            *p -= *p >> m.shift;
        } else {
            self.range = bound;
            *p += ((1 << PROB_BITS) - *p) >> m.shift;
        }
        self.normalize();
    }
    fn direct(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.range >>= 1;
            if (v >> i) & 1 == 1 {
                self.low += u64::from(self.range);
            }
            self.normalize();
        }
    }
}

/// Counts zeros/ones per context.
pub struct Counter(pub Vec<[u64; 2]>);

impl Sink for Counter {
    fn bit(&mut self, _m: &mut Model, ctx: usize, bit: bool) {
        self.0[ctx][usize::from(bit)] += 1;
    }
    fn direct(&mut self, _v: u64, _n: u32) {}
}

pub struct Dec<'r, 'a> {
    inp: &'r mut BitReader<'a>,
    code: u32,
    range: u32,
}

impl<'r, 'a> Dec<'r, 'a> {
    pub fn new(inp: &'r mut BitReader<'a>) -> Result<Self, DecodeError> {
        let mut code = 0u32;
        for _ in 0..4 {
            code = (code << 8) | inp.read(8)? as u32;
        }
        Ok(Self { inp, code, range: u32::MAX })
    }
    fn normalize(&mut self) -> Result<(), DecodeError> {
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.inp.read(8)? as u32;
        }
        Ok(())
    }
    fn bit(&mut self, m: &mut Model, ctx: usize) -> Result<bool, DecodeError> {
        let p = &mut m.p[ctx];
        let bound = (self.range >> PROB_BITS) * u32::from(*p);
        let bit = self.code >= bound;
        if bit {
            self.code -= bound;
            self.range -= bound;
            *p -= *p >> m.shift;
        } else {
            self.range = bound;
            *p += ((1 << PROB_BITS) - *p) >> m.shift;
        }
        self.normalize()?;
        Ok(bit)
    }
    fn direct(&mut self, n: u32) -> Result<u64, DecodeError> {
        let mut v = 0u64;
        for _ in 0..n {
            self.range >>= 1;
            let bit = self.code >= self.range;
            if bit {
                self.code -= self.range;
            }
            v = (v << 1) | u64::from(bit);
            self.normalize()?;
        }
        Ok(v)
    }
}

fn class(mag: u64) -> usize {
    match mag {
        0 => 0,
        1..=3 => 1,
        4..=63 => 2,
        _ => 3,
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Common divisor of the signed magnitudes (0 if all zero).
#[must_use]
pub fn divisor(r: &[u64]) -> u64 {
    let mut g = 0;
    for &x in r {
        g = gcd(g, (x as i64).unsigned_abs());
        if g == 1 {
            break;
        }
    }
    g
}

/// Code the residuals (already divided by `scale`) with the AdaptiveBlock binarisation.
pub fn code<S: Sink>(r: &[u64], scale: u64, m: &mut Model, s: &mut S) {
    let mut cc = 0;
    for &x in r {
        let v = if scale == 1 { i128::from(x as i64) } else { i128::from(x as i64) / i128::from(scale) };
        let mag = u64::try_from(v.unsigned_abs()).unwrap_or(u64::MAX);
        let len = mag.bit_width();
        s.bit(m, ZERO + cc, mag != 0);
        if mag != 0 {
            s.bit(m, SIGN + cc, v < 0);
            for i in 1..len {
                s.bit(m, LEN + cc * 65 + i as usize, true);
            }
            if len < 64 {
                s.bit(m, LEN + cc * 65 + len as usize, false);
            }
            let below = len - 1;
            let modelled = below.min(2);
            let mut ctx = 0;
            for j in 0..modelled {
                let bit = (mag >> (below - 1 - j)) & 1 == 1;
                s.bit(m, MANT + len as usize * 3 + ctx, bit);
                ctx = 1 + usize::from(bit);
            }
            let rest = below - modelled;
            s.direct(mag & ((1u64 << rest) - 1), rest);
        }
        cc = class(mag);
    }
}

pub fn decode_res(n: usize, scale: u64, m: &mut Model, d: &mut Dec<'_, '_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
    let mut cc = 0;
    for _ in 0..n {
        let mut mag = 0u64;
        let mut neg = false;
        if d.bit(m, ZERO + cc)? {
            neg = d.bit(m, SIGN + cc)?;
            let mut len = 1u32;
            while len < 64 && d.bit(m, LEN + cc * 65 + len as usize)? {
                len += 1;
            }
            let below = len - 1;
            let modelled = below.min(2);
            mag = 1;
            let mut ctx = 0;
            for _ in 0..modelled {
                let bit = d.bit(m, MANT + len as usize * 3 + ctx)?;
                mag = (mag << 1) | u64::from(bit);
                ctx = 1 + usize::from(bit);
            }
            let rest = below - modelled;
            mag = (mag << rest) | d.direct(rest)?;
        }
        let v = if neg { -i128::from(mag) } else { i128::from(mag) };
        out.push((if scale == 1 { v } else { v.wrapping_mul(i128::from(scale)) }) as u64);
        cc = class(mag);
    }
    Ok(())
}

/// Delta residuals of order 0/1/2 (library definitions).
#[must_use]
pub fn delta_res(x: &[u64], order: u8) -> Vec<u64> {
    let mut prev_d = 0u64;
    x.windows(2)
        .map(|w| {
            let d = w[1].wrapping_sub(w[0]);
            match order {
                0 => w[1],
                1 => d,
                _ => {
                    let r = d.wrapping_sub(prev_d);
                    prev_d = d;
                    r
                }
            }
        })
        .collect()
}

pub fn integrate(first: u64, r: &[u64], order: u8, out: &mut Vec<u64>) {
    let (mut x, mut d) = (first, 0u64);
    out.push(first);
    for &v in r {
        x = match order {
            0 => v,
            1 => x.wrapping_add(v),
            _ => {
                d = d.wrapping_add(v);
                x.wrapping_add(d)
            }
        };
        out.push(x);
    }
}

/// Side info and residuals for one block.
pub struct Plan {
    pub side: BitWriter, // scaled header + x0 + order
    pub res: Vec<u64>,
    pub rice_bits: usize,
}

/// Candidate plans for a block: every exception-front exponent x orders 0/1/2.
#[must_use]
pub fn plans(items: &[u64]) -> Vec<Plan> {
    let n = items.len();
    let mut v = Vec::new();
    let mut best_rice = usize::MAX;
    for (k, exc) in exception_front(items).into_iter().rev() {
        if exc * per_exception(n) > best_rice {
            continue;
        }
        let s = scale(items, k);
        let ints: Vec<u64> = s.ints.iter().map(|&a| a as u64).collect();
        for order in [1u8, 2, 0] {
            let mut side = BitWriter::new();
            write_scaled_header(&s, n, &mut side);
            side.write(ints[0], 64);
            let res = delta_res(&ints, order);
            if n > 1 {
                side.write(u64::from(order), 2);
            }
            let rice_bits = if n > 1 { RiceBlock::cost(&res, RiceBlock::choose(&res)) as usize } else { 0 };
            best_rice = best_rice.min(side.bit_len() + rice_bits);
            v.push(Plan { side, res, rice_bits });
        }
    }
    v
}

/// Adaptive section bits for `res` starting from `m` (model not modified).
#[must_use]
pub fn adaptive_bits(res: &[u64], m: &Model) -> usize {
    let mut w = BitWriter::new();
    let g = divisor(res).max(1);
    let mut mm = m.clone();
    let mut e = Enc::new(&mut w);
    code(res, g, &mut mm, &mut e);
    e.finish();
    w.bit_len() + 1 + if g > 1 { 7 + g.bit_width() as usize } else { 0 }
}

fn write_scale(out: &mut BitWriter, g: u64) {
    out.write(u64::from(g > 1), 1);
    if g > 1 {
        write_prefixed(out, g);
    }
}

fn read_scale(inp: &mut BitReader<'_>) -> Result<u64, DecodeError> {
    Ok(if inp.read_bit()? { read_prefixed(inp)? } else { 1 })
}

/// Choose plan + coder for a block given the current model.
/// Returns (plan index, use_adaptive).
fn choose(ps: &[Plan], m: &Model) -> (usize, bool) {
    let mut best = (usize::MAX, 0, false);
    for (i, p) in ps.iter().enumerate() {
        let r = p.side.bit_len() + 1 + p.rice_bits;
        if r < best.0 {
            best = (r, i, false);
        }
    }
    for (i, p) in ps.iter().enumerate() {
        if p.res.is_empty() || p.side.bit_len() + 1 >= best.0 {
            continue;
        }
        let a = p.side.bit_len() + 1 + adaptive_bits(&p.res, m);
        if a < best.0 {
            best = (a, i, true);
        }
    }
    (best.1, best.2)
}

// ---------------------------------------------------------------------------
// Stream variants (B5). Layout: header; varint(len of side stream in bits);
// side stream (per block: plan side info, coder flag, Rice payload or scale);
// range-coded stream (per block or one section).

#[derive(Clone)]
pub enum Init {
    Fresh,
    Prior(Arc<Vec<u16>>),
}

#[derive(Clone)]
pub struct CtxStream {
    pub block: usize,
    pub init: Init,
    pub shift: u32,
    /// Reset the model every `reset` blocks (1 = every block, 0 = never).
    pub reset: usize,
    /// One range-coded section for the whole stream (no per-block flush).
    pub single_coder: bool,
}

impl CtxStream {
    fn start(&self) -> Model {
        match &self.init {
            Init::Fresh => Model::fresh(self.shift),
            Init::Prior(p) => Model { p: p.as_ref().clone(), shift: self.shift },
        }
    }

    #[must_use]
    pub fn encode(&self, v: &[f64]) -> Vec<u8> {
        let words: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
        let mut side = BitWriter::new();
        let mut rcw = BitWriter::new();
        let mut model = self.start();
        let mut deferred: Vec<(Vec<u64>, u64, Model)> = Vec::new();
        for (bi, blk) in words.chunks(self.block).enumerate() {
            if self.reset > 0 && bi % self.reset == 0 {
                model = self.start();
            }
            let ps = plans(blk);
            let (i, adaptive) = choose(&ps, &model);
            let p = &ps[i];
            side.append(&p.side);
            if p.res.is_empty() {
                continue;
            }
            side.write(u64::from(adaptive), 1);
            if adaptive {
                let g = divisor(&p.res).max(1);
                write_scale(&mut side, g);
                if self.single_coder {
                    // Coded at the end in one section, from the model state at
                    // this point; advance the model exactly as coding would.
                    deferred.push((p.res.clone(), g, model.clone()));
                    let mut tmp = BitWriter::new();
                    let mut e = Enc::new(&mut tmp);
                    code(&p.res, g, &mut model, &mut e);
                    e.finish();
                } else {
                    let mut e = Enc::new(&mut rcw);
                    code(&p.res, g, &mut model, &mut e);
                    e.finish();
                }
            } else {
                RiceBlock::encode(&p.res, &mut side);
            }
        }
        if self.single_coder {
            let mut e = Enc::new(&mut rcw);
            for (res, g, mut m) in deferred {
                code(&res, g, &mut m, &mut e);
            }
            e.finish();
        }
        let mut bytes = header(self.block, v.len());
        crate::codec::varint(&mut bytes, side.bit_len() as u64);
        bytes.extend(side.finish());
        bytes.extend(rcw.finish());
        bytes
    }

    pub fn decode(&self, bytes: &[u8]) -> Result<Vec<f64>, DecodeError> {
        let (block, n, payload) = parse_header(bytes);
        let mut p = payload;
        let mut side_bits = 0u64;
        let mut shift = 0;
        loop {
            let b = p[0];
            p = &p[1..];
            side_bits |= u64::from(b & 0x7F) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        let side_bytes = (side_bits as usize).div_ceil(8);
        let mut side = BitReader::new(&p[..side_bytes]);
        let mut rcr = BitReader::new(&p[side_bytes..]);
        let mut single = if self.single_coder { Some(Dec::new(&mut rcr)?) } else { None };
        let mut rcr2 = if self.single_coder { None } else { Some(BitReader::new(&p[side_bytes..])) };
        let mut model = self.start();
        let mut out = Vec::with_capacity(n);
        let mut left = n;
        let mut bi = 0;
        while left > 0 {
            let m = left.min(block);
            if self.reset > 0 && bi % self.reset == 0 {
                model = self.start();
            }
            let (k, exc) = read_scaled_header(m, &mut side)?;
            let x0 = side.read(64)?;
            let mut ints = Vec::with_capacity(m);
            if m == 1 {
                ints.push(x0);
            } else {
                let order = side.read(2)? as u8;
                let mut r = Vec::with_capacity(m - 1);
                if side.read_bit()? {
                    let g = read_scale(&mut side)?;
                    match (&mut single, &mut rcr2) {
                        (Some(d), _) => decode_res(m - 1, g, &mut model, d, &mut r)?,
                        (None, Some(rr)) => {
                            let mut d = Dec::new(rr)?;
                            decode_res(m - 1, g, &mut model, &mut d, &mut r)?;
                        }
                        _ => unreachable!(),
                    }
                } else {
                    RiceBlock::decode(m - 1, &mut side, &mut r)?;
                }
                integrate(x0, &r, order, &mut ints);
            }
            let ints: Vec<i64> = ints.into_iter().map(|a| a as i64).collect();
            unscale(&ints, k, &exc, &mut out);
            left -= m;
            bi += 1;
        }
        Ok(out.into_iter().map(f64::from_bits).collect())
    }
}

/// Per-context bit counts over a dataset (plans chosen with a fresh model).
#[must_use]
pub fn train_counts(v: &[f64], block: usize) -> Vec<[u64; 2]> {
    let words: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
    let mut c = Counter(vec![[0; 2]; NPROB]);
    let mut m = Model::fresh(5);
    for blk in words.chunks(block) {
        let ps = plans(blk);
        let (i, _) = choose(&ps, &Model::fresh(5));
        let p = &ps[i];
        if !p.res.is_empty() {
            let g = divisor(&p.res).max(1);
            code(&p.res, g, &mut m, &mut c);
        }
    }
    c.0
}

/// Prior probabilities (of a zero bit) from counts.
#[must_use]
pub fn prior_from(counts: &[[u64; 2]]) -> Vec<u16> {
    counts
        .iter()
        .map(|c| {
            let p = (c[0] as f64 + 0.5) / ((c[0] + c[1]) as f64 + 1.0);
            (p * f64::from(1u32 << PROB_BITS)).round().clamp(31.0, 2017.0) as u16
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Block-independent candidate using a fixed prior (for "auto + prior").

pub struct DecimalPrior {
    pub prior: Arc<Vec<u16>>,
    pub shift: u32,
}

impl DecimalPrior {
    fn model(&self) -> Model {
        Model { p: self.prior.as_ref().clone(), shift: self.shift }
    }
}

impl Cand for DecimalPrior {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        let ps = plans(items);
        let m = self.model();
        let (i, adaptive) = choose(&ps, &m);
        let p = &ps[i];
        out.append(&p.side);
        if p.res.is_empty() {
            return true;
        }
        out.write(u64::from(adaptive), 1);
        if adaptive {
            let g = divisor(&p.res).max(1);
            write_scale(out, g);
            let mut mm = m;
            let mut e = Enc::new(out);
            code(&p.res, g, &mut mm, &mut e);
            e.finish();
        } else {
            RiceBlock::encode(&p.res, out);
        }
        true
    }

    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        let (k, exc) = read_scaled_header(n, inp)?;
        let x0 = inp.read(64)?;
        let mut ints = Vec::with_capacity(n);
        if n == 1 {
            ints.push(x0);
        } else {
            let order = inp.read(2)? as u8;
            let mut r = Vec::with_capacity(n - 1);
            if inp.read_bit()? {
                let g = read_scale(inp)?;
                let mut m = self.model();
                let mut d = Dec::new(inp)?;
                decode_res(n - 1, g, &mut m, &mut d, &mut r)?;
            } else {
                RiceBlock::decode(n - 1, inp, &mut r)?;
            }
            integrate(x0, &r, order, &mut ints);
        }
        let ints: Vec<i64> = ints.into_iter().map(|a| a as i64).collect();
        unscale(&ints, k, &exc, out);
        Ok(())
    }
}
