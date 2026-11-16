//! Audio extension: FLAC-style linear prediction in front of the library's
//! residual coders. 16-bit interleaved stereo PCM, blocks of 4096 frames.
//! Per block: stereo mode (L/R, L/S, S/R, M/S), then per channel an LPC order
//! from {0 (verbatim delta-free), 2, 4, 8, 12, 16, 24, 32} with quantised
//! coefficients (Levinson-Durbin on the block's autocorrelation, Welch window).
//! Residuals: gorilla-tsc `RiceBlock` on 256-sample partitions, or the whole
//! channel with gorilla-tsc `AdaptiveBlock` (range coded); smallest wins.
//! Exact integer prediction, so the decoder reproduces samples bit for bit.

use gorilla_tsc::{AdaptiveBlock, BitReader, BitWriter, RiceBlock};

const BLOCK: usize = 4096;
const PART: usize = 256;
const ORDERS: [usize; 8] = [0, 2, 4, 8, 12, 16, 24, 32];
const PREC: u32 = 14; // coefficient precision (bits incl. sign)

fn lpc_coefs(x: &[i32], order: usize) -> Option<(Vec<i32>, u32)> {
    let n = x.len();
    if order == 0 || n <= order * 2 {
        return None;
    }
    let w: Vec<f64> = (0..n)
        .map(|i| {
            let t = (2.0 * i as f64 - (n as f64 - 1.0)) / (n as f64 + 1.0);
            f64::from(x[i]) * (1.0 - t * t)
        })
        .collect();
    let mut r = vec![0.0; order + 1];
    for (lag, rv) in r.iter_mut().enumerate() {
        *rv = (lag..n).map(|i| w[i] * w[i - lag]).sum();
    }
    if r[0] <= 0.0 {
        return None;
    }
    r[0] *= 1.0 + 1e-9;
    // Levinson-Durbin
    let mut a = vec![0.0f64; order + 1];
    let mut err = r[0];
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc -= a[j] * r[i - j];
        }
        let k = acc / err;
        let prev = a.clone();
        a[i] = k;
        for j in 1..i {
            a[j] = prev[j] - k * prev[i - j];
        }
        err *= 1.0 - k * k;
        if err <= 0.0 {
            return None;
        }
    }
    let maxc = a[1..].iter().fold(0.0f64, |m, &c| m.max(c.abs()));
    if maxc == 0.0 {
        return None;
    }
    let lim = f64::from((1i32 << (PREC - 1)) - 1);
    let mut shift = 0u32;
    while shift < 20 && maxc * f64::from(1u32 << (shift + 1)) <= lim {
        shift += 1;
    }
    let c: Vec<i32> = a[1..].iter().map(|&v| (v * f64::from(1u32 << shift)).round().clamp(-lim, lim) as i32).collect();
    Some((c, shift))
}

fn predict(hist: &[i32], n: usize, c: &[i32], shift: u32) -> i32 {
    let mut s: i64 = 0;
    for (j, &cj) in c.iter().enumerate() {
        if n > j {
            s += i64::from(cj) * i64::from(hist[n - 1 - j]);
        }
    }
    (s >> shift) as i32
}

fn residuals(x: &[i32], c: &[i32], shift: u32) -> Vec<u32> {
    (0..x.len()).map(|n| x[n].wrapping_sub(predict(x, n, c, shift)) as u32).collect()
}

fn code_residuals(res: &[u32], w: &mut BitWriter) {
    let mut rice = BitWriter::new();
    for p in res.chunks(PART) {
        RiceBlock::encode::<u32>(p, &mut rice);
    }
    let mut ad = BitWriter::new();
    AdaptiveBlock::encode::<u32>(res, &mut ad);
    if rice.bit_len() <= ad.bit_len() {
        w.write(0, 1);
        w.append(&rice);
    } else {
        w.write(1, 1);
        w.append(&ad);
    }
}

fn decode_residuals(n: usize, r: &mut BitReader) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    if r.read(1).unwrap() == 0 {
        let mut left = n;
        while left > 0 {
            let k = left.min(PART);
            RiceBlock::decode::<u32>(k, r, &mut out).unwrap();
            left -= k;
        }
    } else {
        AdaptiveBlock::decode::<u32>(n, r, &mut out).unwrap();
    }
    out
}

fn encode_channel(x: &[i32], w: &mut BitWriter) {
    let mut best: Option<BitWriter> = None;
    for &o in &ORDERS {
        let (c, shift) = if o == 0 { (Vec::new(), 0) } else {
            match lpc_coefs(x, o) {
                Some(v) => v,
                None => continue,
            }
        };
        let mut t = BitWriter::new();
        t.write(o as u64, 6);
        t.write(u64::from(shift), 5);
        for &cj in &c {
            t.write(u64::from(cj as u32) & ((1 << PREC) - 1), PREC);
        }
        code_residuals(&residuals(x, &c, shift), &mut t);
        if best.as_ref().is_none_or(|b| t.bit_len() < b.bit_len()) {
            best = Some(t);
        }
    }
    w.append(&best.unwrap());
}

fn decode_channel(n: usize, r: &mut BitReader) -> Vec<i32> {
    let o = r.read(6).unwrap() as usize;
    let shift = r.read(5).unwrap() as u32;
    let c: Vec<i32> = (0..o)
        .map(|_| {
            let v = r.read(PREC).unwrap() as i32;
            (v << (32 - PREC)) >> (32 - PREC)
        })
        .collect();
    let res = decode_residuals(n, r);
    let mut x = Vec::with_capacity(n);
    for (i, &e) in res.iter().enumerate() {
        let p = predict(&x, i, &c, shift);
        x.push(p.wrapping_add(e as i32));
    }
    x
}

/// `data` = interleaved little-endian i16 stereo.
pub fn encode(data: &[u8]) -> Vec<u8> {
    let frames = data.len() / 4;
    let s = |i: usize| i32::from(i16::from_le_bytes([data[i], data[i + 1]]));
    let mut w = BitWriter::new();
    w.write(data.len() as u64, 40);
    for b in (0..frames).step_by(BLOCK) {
        let e = (b + BLOCK).min(frames);
        let l: Vec<i32> = (b..e).map(|f| s(4 * f)).collect();
        let r: Vec<i32> = (b..e).map(|f| s(4 * f + 2)).collect();
        let side: Vec<i32> = l.iter().zip(&r).map(|(a, b)| a - b).collect();
        let mid: Vec<i32> = l.iter().zip(&r).map(|(a, b)| (a + b) >> 1).collect();
        let modes: [(&[i32], &[i32]); 4] = [(&l, &r), (&l, &side), (&side, &r), (&mid, &side)];
        let mut best: Option<BitWriter> = None;
        for (m, (a, c)) in modes.iter().enumerate() {
            let mut t = BitWriter::new();
            t.write(m as u64, 2);
            encode_channel(a, &mut t);
            encode_channel(c, &mut t);
            if best.as_ref().is_none_or(|bw| t.bit_len() < bw.bit_len()) {
                best = Some(t);
            }
        }
        w.append(&best.unwrap());
    }
    let mut out = w.finish();
    out.extend_from_slice(&data[frames * 4..]);
    out
}

pub fn decode(inp: &[u8]) -> Vec<u8> {
    let mut r = BitReader::new(inp);
    let len = r.read(40).unwrap() as usize;
    let frames = len / 4;
    let mut out = Vec::with_capacity(len);
    for b in (0..frames).step_by(BLOCK) {
        let n = (b + BLOCK).min(frames) - b;
        let m = r.read(2).unwrap();
        let a = decode_channel(n, &mut r);
        let c = decode_channel(n, &mut r);
        for i in 0..n {
            let (l, rr) = match m {
                0 => (a[i], c[i]),
                1 => (a[i], a[i] - c[i]),
                2 => (c[i] + a[i], c[i]),
                _ => {
                    let side = c[i];
                    let mid2 = (a[i] << 1) | (side & 1);
                    ((mid2 + side) >> 1, (mid2 - side) >> 1)
                }
            };
            out.extend_from_slice(&(l as i16).to_le_bytes());
            out.extend_from_slice(&(rr as i16).to_le_bytes());
        }
    }
    // tail bytes: after the bit stream, byte aligned
    let tail = len - frames * 4;
    out.extend_from_slice(&inp[inp.len() - tail..]);
    out
}
