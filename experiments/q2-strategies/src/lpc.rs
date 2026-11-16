//! B4: linear prediction on the decimal-scaled integers.
//!
//! Like the library's `DecimalRice` (same exponent search, same exception
//! layout) but with a richer predictor set per block:
//! fixed polynomial predictors of order 0-3 (orders 0-2 are what `Delta`
//! already has), plus least-squares LPC of order 1..=P fitted per block either
//! on the first differences or on the mean-removed values. Coefficients are
//! quantised to Q12 and stored in 16 bits each. Residuals are coded with the
//! library's `RiceBlock` or `AdaptiveBlock`, whichever is smaller.
//!
//! Layout: scaled header (k, exceptions); x0 (64 bits); predictor id (3 bits);
//! predictor parameters; coder flag (1 bit); residuals.

use gorilla_tsc::{AdaptiveBlock, BitReader, BitWriter, DecodeError, RiceBlock};

use crate::codec::{
    Cand, Scaled, exception_front, per_exception, read_prefixed, read_scaled_header, scale, unscale, unzz,
    write_prefixed, write_scaled_header, zz,
};

const SHIFT: u32 = 12;
const COEF_BITS: u32 = 16;

#[derive(Clone, Debug)]
pub enum Pred {
    /// Polynomial predictor of order 0..=3 (lower order near the block start).
    Fixed(u8),
    /// LPC on first differences.
    Delta(Vec<i64>),
    /// LPC on mean-removed values.
    Mean(i64, Vec<i64>),
}

impl Pred {
    fn id(&self) -> u64 {
        match self {
            Pred::Fixed(o) => u64::from(*o),
            Pred::Delta(_) => 4,
            Pred::Mean(..) => 5,
        }
    }
}

fn lpc_pred(hist: &[i64], i: usize, coefs: &[i64], lo: usize) -> i64 {
    // hist[j] for j >= lo is valid; earlier lags count as zero.
    let mut acc: i128 = 1 << (SHIFT - 1);
    for (j, &c) in coefs.iter().enumerate() {
        let lag = j + 1;
        if i >= lag + lo {
            acc += i128::from(c) * i128::from(hist[i - lag]);
        }
    }
    (acc >> SHIFT) as i64
}

fn fixed_pred(x: &[i64], i: usize, order: u8) -> i64 {
    match (order as usize).min(i) {
        0 => 0,
        1 => x[i - 1],
        2 => x[i - 1].wrapping_mul(2).wrapping_sub(x[i - 2]),
        _ => x[i - 1].wrapping_mul(3).wrapping_sub(x[i - 2].wrapping_mul(3)).wrapping_add(x[i - 3]),
    }
}

/// Residuals for i in 1..n as wrapping words.
#[must_use]
pub fn residuals(x: &[i64], p: &Pred) -> Vec<u64> {
    let n = x.len();
    match p {
        Pred::Fixed(o) => (1..n).map(|i| x[i].wrapping_sub(fixed_pred(x, i, *o)) as u64).collect(),
        Pred::Delta(c) => {
            let mut y = vec![0i64; n];
            for i in 1..n {
                y[i] = x[i].wrapping_sub(x[i - 1]);
            }
            (1..n).map(|i| y[i].wrapping_sub(lpc_pred(&y, i, c, 1)) as u64).collect()
        }
        Pred::Mean(m, c) => {
            let z: Vec<i64> = x.iter().map(|v| v.wrapping_sub(*m)).collect();
            (1..n).map(|i| z[i].wrapping_sub(lpc_pred(&z, i, c, 0)) as u64).collect()
        }
    }
}

/// Inverse of [`residuals`].
#[must_use]
pub fn reconstruct(x0: i64, res: &[u64], p: &Pred) -> Vec<i64> {
    let n = res.len() + 1;
    let mut x = Vec::with_capacity(n);
    x.push(x0);
    match p {
        Pred::Fixed(o) => {
            for i in 1..n {
                let v = fixed_pred(&x, i, *o).wrapping_add(res[i - 1] as i64);
                x.push(v);
            }
        }
        Pred::Delta(c) => {
            let mut y = vec![0i64; n];
            for i in 1..n {
                y[i] = lpc_pred(&y, i, c, 1).wrapping_add(res[i - 1] as i64);
                x.push(x[i - 1].wrapping_add(y[i]));
            }
        }
        Pred::Mean(m, c) => {
            let mut z = vec![0i64; n];
            z[0] = x0.wrapping_sub(*m);
            for i in 1..n {
                z[i] = lpc_pred(&z, i, c, 0).wrapping_add(res[i - 1] as i64);
                x.push(z[i].wrapping_add(*m));
            }
        }
    }
    x
}

/// Least-squares fit of order `p` on `t[lo..]`, quantised to Q`SHIFT`.
fn fit(t: &[i64], lo: usize, p: usize) -> Option<Vec<i64>> {
    let n = t.len();
    if n < lo + p + 8 {
        return None;
    }
    let tf: Vec<f64> = t.iter().map(|&v| v as f64).collect();
    let mut a = vec![vec![0f64; p + 1]; p];
    for i in lo + p..n {
        for j in 0..p {
            let xj = tf[i - 1 - j];
            for k in 0..p {
                a[j][k] += xj * tf[i - 1 - k];
            }
            a[j][p] += xj * tf[i];
        }
    }
    // ridge for stability
    for j in 0..p {
        a[j][j] *= 1.0 + 1e-9;
        if a[j][j] == 0.0 {
            return None;
        }
    }
    for col in 0..p {
        let piv = (col..p).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        a.swap(col, piv);
        if a[col][col].abs() < 1e-300 {
            return None;
        }
        for row in 0..p {
            if row != col {
                let f = a[row][col] / a[col][col];
                for k in col..=p {
                    a[row][k] -= f * a[col][k];
                }
            }
        }
    }
    let lim = f64::from(1u32 << (COEF_BITS - 1)) - 1.0;
    let c: Vec<i64> = (0..p).map(|j| (a[j][p] / a[j][j] * f64::from(1u32 << SHIFT)).round().clamp(-lim, lim) as i64).collect();
    c.iter().all(|v| v.abs() <= lim as i64).then_some(c)
}

pub struct DecLpc {
    pub max_order: usize,
}

fn code_residuals(r: &[u64]) -> BitWriter {
    let p = RiceBlock::choose(r);
    let rice = RiceBlock::cost(r, p) as usize;
    let mut a = BitWriter::new();
    a.write(1, 1);
    AdaptiveBlock::encode(r, &mut a);
    if rice + 1 <= a.bit_len() {
        let mut w = BitWriter::new();
        w.write(0, 1);
        RiceBlock::encode(r, &mut w);
        w
    } else {
        a
    }
}

fn write_pred(p: &Pred, out: &mut BitWriter) {
    out.write(p.id(), 3);
    let coefs = match p {
        Pred::Fixed(_) => return,
        Pred::Delta(c) => c,
        Pred::Mean(m, c) => {
            write_prefixed(out, zz(*m));
            c
        }
    };
    out.write(coefs.len() as u64, 3);
    for &c in coefs {
        out.write((c as u64) & ((1 << COEF_BITS) - 1), COEF_BITS);
    }
}

fn read_pred(inp: &mut BitReader<'_>) -> Result<Pred, DecodeError> {
    let id = inp.read(3)?;
    if id < 4 {
        return Ok(Pred::Fixed(id as u8));
    }
    let m = if id == 5 { unzz(read_prefixed(inp)?) } else { 0 };
    let p = inp.read(3)? as usize;
    let mut c = Vec::with_capacity(p);
    for _ in 0..p {
        let raw = inp.read(COEF_BITS)?;
        c.push(((raw << (64 - COEF_BITS)) as i64) >> (64 - COEF_BITS));
    }
    Ok(if id == 4 { Pred::Delta(c) } else { Pred::Mean(m, c) })
}

impl DecLpc {
    fn predictors(&self, x: &[i64]) -> Vec<Pred> {
        let mut v: Vec<Pred> = (0..=3).map(Pred::Fixed).collect();
        let n = x.len();
        if n < 16 {
            return v;
        }
        let mut y = vec![0i64; n];
        for i in 1..n {
            y[i] = x[i].wrapping_sub(x[i - 1]);
        }
        let mean = (x.iter().map(|&a| i128::from(a)).sum::<i128>() / n as i128) as i64;
        let z: Vec<i64> = x.iter().map(|a| a.wrapping_sub(mean)).collect();
        for p in 1..=self.max_order {
            if let Some(c) = fit(&y, 1, p) {
                v.push(Pred::Delta(c));
            }
            if let Some(c) = fit(&z, 0, p) {
                v.push(Pred::Mean(mean, c));
            }
        }
        v
    }

    fn encode_scaled(&self, s: &Scaled, n: usize) -> BitWriter {
        let mut out = BitWriter::new();
        write_scaled_header(s, n, &mut out);
        out.write(s.ints[0] as u64, 64);
        if n == 1 {
            return out;
        }
        let mut best: Option<BitWriter> = None;
        for p in self.predictors(&s.ints) {
            let r = residuals(&s.ints, &p);
            let mut w = BitWriter::new();
            write_pred(&p, &mut w);
            w.append(&code_residuals(&r));
            if best.as_ref().is_none_or(|b| w.bit_len() < b.bit_len()) {
                best = Some(w);
            }
        }
        out.append(&best.unwrap());
        out
    }
}

impl Cand for DecLpc {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        let n = items.len();
        let mut best: Option<BitWriter> = None;
        for (k, exc) in exception_front(items).into_iter().rev() {
            if best.as_ref().is_some_and(|b| exc * per_exception(n) > b.bit_len()) {
                continue;
            }
            let w = self.encode_scaled(&scale(items, k), n);
            if best.as_ref().is_none_or(|b| w.bit_len() < b.bit_len()) {
                best = Some(w);
            }
        }
        out.append(&best.unwrap());
        true
    }

    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        let (k, exc) = read_scaled_header(n, inp)?;
        let x0 = inp.read(64)? as i64;
        let ints = if n == 1 {
            vec![x0]
        } else {
            let p = read_pred(inp)?;
            let mut r = Vec::with_capacity(n - 1);
            if inp.read_bit()? {
                AdaptiveBlock::decode(n - 1, inp, &mut r)?;
            } else {
                RiceBlock::decode(n - 1, inp, &mut r)?;
            }
            reconstruct(x0, &r, &p)
        };
        unscale(&ints, k, &exc, out);
        Ok(())
    }
}
