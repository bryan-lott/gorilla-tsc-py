//! B6: "decimal plus float noise" — split the decimal-scaled integers as
//! `x = q * 10^j + r` with `r` the nearest-remainder in `[-10^j/2, 10^j/2)`.
//!
//! Motivation (measured on Basel-temp): values such as `1.9005287`,
//! `1.6105287`, `0.8205287` are two-decimal readings plus a constant offset,
//! computed in float32 and printed with float32 precision. After decimal
//! scaling the integers are near multiples of 10^5 plus an almost constant
//! remainder, so coding `q` (the reading) and `r` (offset + float noise)
//! separately, each with the library's DeltaRice (best order, Rice/adaptive),
//! is much cheaper than coding `x` directly.
//!
//! j is chosen per block: a cheap proxy ranks j = 1..=17, the best two are
//! encoded in full and compared with no split.
//!
//! Layout: scaled header; j (5 bits, 0 = no split); DeltaRice(x) if j = 0,
//! else DeltaRice(q) then DeltaRice(r).

use gorilla_tsc::{BitReader, BitWriter, BlockCodec, DecodeError, DeltaRice};

use crate::codec::{Cand, exception_front, per_exception, read_scaled_header, scale, unscale, write_scaled_header};

pub struct DecimalSplit;

fn split(x: &[i64], g: i64) -> (Vec<u64>, Vec<u64>) {
    let half = g / 2;
    x.iter()
        .map(|&v| {
            let q = (i128::from(v) + i128::from(half)).div_euclid(i128::from(g));
            let r = i128::from(v) - q * i128::from(g);
            (q as i64 as u64, r as i64 as u64)
        })
        .unzip()
}

impl Cand for DecimalSplit {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        let n = items.len();
        let mut best: Option<BitWriter> = None;
        for (k, exc) in exception_front(items).into_iter().rev() {
            if best.as_ref().is_some_and(|b| exc * per_exception(n) > b.bit_len()) {
                continue;
            }
            let s = scale(items, k);
            let mut head = BitWriter::new();
            write_scaled_header(&s, n, &mut head);
            let xs: Vec<u64> = s.ints.iter().map(|&a| a as u64).collect();
            let mut w = BitWriter::new();
            w.write(0, 5);
            DeltaRice.encode_block(&xs, &mut w);
            let mut bestj = w;
            // Cheap proxy (sum of bit widths of first differences of q and r)
            // picks the two most promising j; only those are fully encoded.
            let mut proxies: Vec<(u64, u32)> = (1..=17u32)
                .map(|j| {
                    let (q, r) = split(&s.ints, 10i64.pow(j));
                    let w = |v: &[u64]| v.windows(2).map(|p| u64::from(crate::codec::zz(p[1].wrapping_sub(p[0]) as i64).bit_width())).sum::<u64>();
                    (w(&q) + w(&r), j)
                })
                .collect();
            proxies.sort_unstable();
            for &(_, j) in proxies.iter().take(2) {
                let g = 10i64.pow(j);
                let (q, r) = split(&s.ints, g);
                let mut w = BitWriter::new();
                w.write(u64::from(j), 5);
                DeltaRice.encode_block(&q, &mut w);
                if w.bit_len() >= bestj.bit_len() {
                    continue;
                }
                DeltaRice.encode_block(&r, &mut w);
                if w.bit_len() < bestj.bit_len() {
                    bestj = w;
                }
            }
            head.append(&bestj);
            if best.as_ref().is_none_or(|b| head.bit_len() < b.bit_len()) {
                best = Some(head);
            }
        }
        out.append(&best.unwrap());
        true
    }

    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        let (k, exc) = read_scaled_header(n, inp)?;
        let j = inp.read(5)? as u32;
        let ints: Vec<i64> = if j == 0 {
            let mut x: Vec<u64> = Vec::with_capacity(n);
            DeltaRice.decode_block(n, inp, &mut x)?;
            x.into_iter().map(|a| a as i64).collect()
        } else {
            let g = 10i64.pow(j);
            let mut q: Vec<u64> = Vec::with_capacity(n);
            DeltaRice.decode_block(n, inp, &mut q)?;
            let mut r: Vec<u64> = Vec::with_capacity(n);
            DeltaRice.decode_block(n, inp, &mut r)?;
            q.iter().zip(&r).map(|(&a, &b)| (a as i64).wrapping_mul(g).wrapping_add(b as i64)).collect()
        };
        unscale(&ints, k, &exc, out);
        Ok(())
    }
}
