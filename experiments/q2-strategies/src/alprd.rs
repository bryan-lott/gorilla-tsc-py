//! B3: ALP-RD style "split" coding for high-precision floats.
//!
//! Each word is split into `left = w >> r` (64 - r bits) and `right = w & (2^r - 1)`.
//! Left parts are dictionary coded; right parts are stored raw.
//!
//! * [`AlpRd`] follows the ALP-RD paper: left at most 16 bits, dictionary of
//!   at most 8 entries with fixed-width (0-3 bit) codes, exceptions stored as
//!   (position, left).
//! * [`SplitAdaptive`] (ours): left up to 32 bits, dictionary of up to 255
//!   entries plus an escape symbol, coded with an adaptive binary range coder
//!   (bit tree, context = previous symbol for small alphabets).

use std::collections::HashMap;

use gorilla_tsc::{BitReader, BitWriter, DecodeError, Prob, RangeDecoder, RangeEncoder};

use crate::codec::{Cand, bit_len};

fn mask(r: u32) -> u64 {
    if r >= 64 { u64::MAX } else { (1u64 << r) - 1 }
}

/// Left parts sorted by descending frequency (ties by value, for determinism).
fn left_freq(items: &[u64], r: u32) -> Vec<(u64, usize)> {
    let mut m: HashMap<u64, usize> = HashMap::new();
    for &w in items {
        *m.entry(w >> r).or_default() += 1;
    }
    let mut v: Vec<(u64, usize)> = m.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

pub struct AlpRd;

impl Cand for AlpRd {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        let n = items.len();
        let pos_bits = bit_len(n.saturating_sub(1)) as usize;
        let mut best: Option<(usize, u32, u32, Vec<u64>)> = None;
        for r in 48..=63u32 {
            let lw = (64 - r) as usize;
            let freq = left_freq(items, r);
            for b in 0..=3u32 {
                let d = (1usize << b).min(freq.len());
                let covered: usize = freq[..d].iter().map(|x| x.1).sum();
                let exc = n - covered;
                let cost = 6 + 2 + 4 + d * lw + bit_len(n) as usize + exc * (pos_bits + lw) + n * (b as usize + r as usize);
                if best.as_ref().is_none_or(|x| cost < x.0) {
                    best = Some((cost, r, b, freq[..d].iter().map(|x| x.0).collect()));
                }
            }
        }
        let (_, r, b, dict) = best.unwrap();
        let lw = 64 - r;
        out.write(u64::from(r), 6);
        out.write(u64::from(b), 2);
        out.write(dict.len() as u64, 4);
        for &l in &dict {
            out.write(l, lw);
        }
        let code: HashMap<u64, u64> = dict.iter().enumerate().map(|(i, &l)| (l, i as u64)).collect();
        let exc: Vec<(usize, u64)> =
            items.iter().enumerate().filter(|(_, w)| !code.contains_key(&(*w >> r))).map(|(i, w)| (i, w >> r)).collect();
        out.write(exc.len() as u64, bit_len(n));
        for &(p, l) in &exc {
            out.write(p as u64, bit_len(n.saturating_sub(1)));
            out.write(l, lw);
        }
        for &w in items {
            out.write(code.get(&(w >> r)).copied().unwrap_or(0), b);
            out.write(w & mask(r), r);
        }
        true
    }

    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        let r = inp.read(6)? as u32;
        let b = inp.read(2)? as u32;
        let d = inp.read(4)? as usize;
        let lw = 64 - r;
        let mut dict = Vec::with_capacity(d);
        for _ in 0..d {
            dict.push(inp.read(lw)?);
        }
        let ne = inp.read(bit_len(n))? as usize;
        let mut exc = Vec::with_capacity(ne);
        for _ in 0..ne {
            exc.push((inp.read(bit_len(n.saturating_sub(1)))? as usize, inp.read(lw)?));
        }
        let mut e = exc.into_iter().peekable();
        for i in 0..n {
            let c = inp.read(b)? as usize;
            let right = inp.read(r)?;
            let left = match e.next_if(|x| x.0 == i) {
                Some((_, l)) => l,
                None => *dict.get(c).ok_or(DecodeError::Corrupt("alprd code"))?,
            };
            out.push((left << r) | right);
        }
        Ok(())
    }
}

pub struct SplitAdaptive;

const D_CHOICES: [usize; 9] = [0, 1, 3, 7, 15, 31, 63, 127, 255];

fn entropy_bits(counts: impl Iterator<Item = usize>, n: usize) -> f64 {
    let nf = n as f64;
    counts.filter(|&c| c > 0).map(|c| -(c as f64) * (c as f64 / nf).log2()).sum()
}

/// Bit-tree probabilities for an alphabet of `1 << depth` with `ctxs` contexts.
fn tree(depth: u32, ctxs: usize) -> Vec<Prob> {
    vec![Prob::default(); ctxs << depth]
}

impl Cand for SplitAdaptive {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        let n = items.len();
        let mut best: Option<(f64, u32, Vec<u64>)> = None;
        for r in 32..=63u32 {
            let lw = f64::from(64 - r);
            let freq = left_freq(items, r);
            for &d in &D_CHOICES {
                let d = d.min(freq.len());
                let covered: usize = freq[..d].iter().map(|x| x.1).sum();
                let esc = n - covered;
                let h = entropy_bits(freq[..d].iter().map(|x| x.1).chain([esc]), n);
                let cost = h + esc as f64 * lw + d as f64 * lw + (n as f64) * f64::from(r);
                if best.as_ref().is_none_or(|x| cost < x.0) {
                    best = Some((cost, r, freq[..d].iter().map(|x| x.0).collect()));
                }
            }
        }
        let (_, r, dict) = best.unwrap();
        let lw = 64 - r;
        out.write(u64::from(r), 6);
        out.write(dict.len() as u64, 8);
        for &l in &dict {
            out.write(l, lw);
        }
        let d = dict.len();
        let code: HashMap<u64, usize> = dict.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        let depth = bit_len(d); // alphabet d + 1 <= 2^depth
        let ctxs = if d < 8 { d + 1 } else { 1 };
        let mut probs = tree(depth, ctxs);
        {
            let mut rc = RangeEncoder::new(out);
            let mut prev = 0usize;
            for &w in items {
                let sym = code.get(&(w >> r)).copied().unwrap_or(d);
                let base = if ctxs > 1 { prev << depth } else { 0 };
                let mut node = 1usize;
                for i in (0..depth).rev() {
                    let bit = (sym >> i) & 1 == 1;
                    rc.bit(&mut probs[base + node], bit);
                    node = (node << 1) | usize::from(bit);
                }
                if sym == d {
                    rc.direct(w >> r, lw);
                }
                prev = sym;
            }
            rc.finish();
        }
        for &w in items {
            out.write(w & mask(r), r);
        }
        true
    }

    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        let r = inp.read(6)? as u32;
        let d = inp.read(8)? as usize;
        let lw = 64 - r;
        let mut dict = Vec::with_capacity(d);
        for _ in 0..d {
            dict.push(inp.read(lw)?);
        }
        let depth = bit_len(d);
        let ctxs = if d < 8 { d + 1 } else { 1 };
        let mut probs = tree(depth, ctxs);
        let mut lefts = Vec::with_capacity(n);
        {
            let mut rc = RangeDecoder::new(inp)?;
            let mut prev = 0usize;
            for _ in 0..n {
                let base = if ctxs > 1 { prev << depth } else { 0 };
                let mut node = 1usize;
                for _ in 0..depth {
                    let bit = rc.bit(&mut probs[base + node])?;
                    node = (node << 1) | usize::from(bit);
                }
                let sym = node - (1 << depth);
                let left = if sym == d {
                    rc.direct(lw)?
                } else {
                    *dict.get(sym).ok_or(DecodeError::Corrupt("split symbol"))?
                };
                lefts.push(left);
                prev = sym.min(ctxs - 1);
            }
        }
        for l in lefts {
            out.push((l << r) | inp.read(r)?);
        }
        Ok(())
    }
}
