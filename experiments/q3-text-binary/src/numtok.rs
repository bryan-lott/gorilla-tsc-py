//! Exact number tokens inside text, and their numeric stream coding.
//!
//! A decimal number in text (`-12.50`, `007`, `3`) becomes
//! (mantissa, decimal places, leading-zero pad). Formatting the triple
//! reproduces the original bytes exactly. Streams of numbers are coded as
//! integers scaled to the stream's largest decimal count and handed to
//! gorilla-tsc `Auto` (or, when smaller, to the byte back end).

use crate::util::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Num {
    pub mant: i64,
    pub dec: u8,
    pub pad: u8,
}

const MAX_DIGITS: usize = 18;
const POW10: [i64; 19] = {
    let mut t = [1i64; 19];
    let mut i = 1;
    while i < 19 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

fn ndigits(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 10 {
        v /= 10;
        n += 1;
    }
    n
}

/// Parse `s`, appending the skeleton (literals, `ph` per number) and numbers.
pub fn parse(s: &[u8], ph: u8, skel: &mut Vec<u8>, nums: &mut Vec<Num>) {
    let mut i = 0;
    let n = s.len();
    while i < n {
        let c = s[i];
        let signed = c == b'-'
            && i + 1 < n
            && s[i + 1].is_ascii_digit()
            && (i == 0 || !s[i - 1].is_ascii_alphanumeric());
        if !(c.is_ascii_digit() || signed) {
            skel.push(c);
            i += 1;
            continue;
        }
        let start = if signed { i + 1 } else { i };
        let mut j = start;
        while j < n && s[j].is_ascii_digit() {
            j += 1;
        }
        let int_len = j - start;
        if int_len > MAX_DIGITS {
            skel.extend_from_slice(&s[i..j]);
            i = j;
            continue;
        }
        let mut k = j;
        let mut dec = 0usize;
        if j + 1 < n && s[j] == b'.' && s[j + 1].is_ascii_digit() {
            let mut e = j + 1;
            while e < n && s[e].is_ascii_digit() {
                e += 1;
            }
            if int_len + (e - j - 1) <= MAX_DIGITS {
                dec = e - j - 1;
                k = e;
            }
        }
        let mut a: i64 = 0;
        for &d in &s[start..j] {
            a = a * 10 + i64::from(d - b'0');
        }
        let ip = a as u64;
        for &d in s.get(j + 1..k).unwrap_or(&[]) {
            a = a * 10 + i64::from(d - b'0');
        }
        if signed && a == 0 {
            // "-0" / "-0.00": keep the '-' literal (cannot be a sign of zero)
            skel.push(b'-');
            i += 1;
            continue;
        }
        let pad = int_len - ndigits(ip);
        nums.push(Num { mant: if signed { -a } else { a }, dec: dec as u8, pad: pad as u8 });
        skel.push(ph);
        i = k;
    }
}

pub fn format(n: Num, out: &mut Vec<u8>) {
    if n.mant < 0 {
        out.push(b'-');
    }
    let a = n.mant.unsigned_abs();
    let p = POW10[n.dec as usize] as u64;
    let ip = a / p;
    let fp = a % p;
    for _ in 0..n.pad {
        out.push(b'0');
    }
    let mut tmp = itoa(ip);
    out.append(&mut tmp);
    if n.dec > 0 {
        out.push(b'.');
        let f = itoa(fp);
        for _ in f.len()..n.dec as usize {
            out.push(b'0');
        }
        out.extend_from_slice(&f);
    }
}

fn itoa(mut v: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(20);
    loop {
        b.push(b'0' + (v % 10) as u8);
        v /= 10;
        if v == 0 {
            break;
        }
    }
    b.reverse();
    b
}

/// Two-part container: gorilla/num metadata section + back-end blob sections.
#[derive(Default)]
pub struct Writer {
    pub num: Vec<u8>,
    pub blob: Blob,
    pub stats: Stats,
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub streams: usize,
    pub gorilla_streams: usize,
    pub gorilla_values: usize,
    pub gorilla_bytes: usize,
}

pub struct Reader<'a> {
    pub num: &'a [u8],
    pub npos: usize,
    pub blob: Vec<Vec<u8>>,
    pub bi: usize,
}

impl Writer {
    pub fn finish(self, backend: Backend, mut header: Vec<u8>) -> Vec<u8> {
        put_bytes(&mut header, &self.num);
        let blob = backend.compress(&self.blob.serialize());
        header.extend_from_slice(&blob);
        header
    }
    pub fn section(&mut self, s: Vec<u8>) {
        self.blob.push(s);
    }
}

impl<'a> Reader<'a> {
    /// `rest` = bytes after the caller's header.
    pub fn new(rest: &'a [u8], backend: Backend) -> Self {
        let mut pos = 0;
        let num = get_bytes(rest, &mut pos);
        let blob = Blob::deserialize(&backend.decompress(&rest[pos..]));
        Reader { num, npos: 0, blob, bi: 0 }
    }
    pub fn section(&mut self) -> Vec<u8> {
        self.bi += 1;
        std::mem::take(&mut self.blob[self.bi - 1])
    }
    pub fn byte(&mut self) -> u8 {
        self.npos += 1;
        self.num[self.npos - 1]
    }
    pub fn varint(&mut self) -> u64 {
        get_varint(self.num, &mut self.npos)
    }
    pub fn bytes(&mut self) -> &'a [u8] {
        get_bytes(self.num, &mut self.npos)
    }
}

/// Integers: gorilla-tsc Auto, or zigzag-delta varints in the blob if smaller.
pub fn put_ints(w: &mut Writer, v: &[i64]) {
    w.stats.streams += 1;
    if v.is_empty() {
        w.num.push(2);
        return;
    }
    let g = gorilla_i64(v);
    // Alternatives for the byte back end: zigzag varints of deltas (1) or of
    // the values themselves (3; better for repeated non-sequential values).
    let mut alt_d = Vec::new();
    let mut alt_v = Vec::new();
    let mut prev = 0i64;
    for &x in v {
        put_varint(&mut alt_d, zigzag(x.wrapping_sub(prev)));
        put_varint(&mut alt_v, zigzag(x));
        prev = x;
    }
    // Estimate the back-end cost of the alternatives with xz -6 (cheap proxy).
    let est = |a: &Vec<u8>| if a.len() < 64 { a.len() + 8 } else { xz_compress(a, 6).len() };
    let (ed, ev) = (est(&alt_d), est(&alt_v));
    if g.len() <= ed.min(ev) {
        w.stats.gorilla_streams += 1;
        w.stats.gorilla_values += v.len();
        w.stats.gorilla_bytes += g.len();
        w.num.push(0);
        put_bytes(&mut w.num, &g);
    } else if ed <= ev {
        w.num.push(1);
        w.section(alt_d);
    } else {
        w.num.push(3);
        w.section(alt_v);
    }
}

pub fn get_ints(r: &mut Reader, n: usize) -> Vec<i64> {
    match r.byte() {
        0 => {
            let v = ungorilla_i64(r.bytes());
            assert_eq!(v.len(), n);
            v
        }
        1 => {
            let s = r.section();
            let mut pos = 0;
            let mut prev = 0i64;
            (0..n)
                .map(|_| {
                    prev = prev.wrapping_add(unzigzag(get_varint(&s, &mut pos)));
                    prev
                })
                .collect()
        }
        3 => {
            let s = r.section();
            let mut pos = 0;
            (0..n).map(|_| unzigzag(get_varint(&s, &mut pos))).collect()
        }
        _ => Vec::new(),
    }
}

fn put_small(w: &mut Writer, v: &[u8]) {
    if v.iter().all(|&x| x == 0) {
        w.num.push(0);
    } else {
        w.num.push(1);
        w.section(v.to_vec());
    }
}

fn get_small(r: &mut Reader, n: usize) -> Vec<u8> {
    if r.byte() == 0 { vec![0; n] } else { r.section() }
}

fn trailing_zeros10(mut v: u64, cap: u8) -> u8 {
    if v == 0 {
        return cap;
    }
    let mut t = 0;
    while t < cap && v % 10 == 0 {
        v /= 10;
        t += 1;
    }
    t
}

/// Cost estimate of `put_ints` (gorilla size or xz -6 proxy of the varint alternatives).
fn ints_cost(v: &[i64]) -> usize {
    if v.is_empty() {
        return 1;
    }
    let g = gorilla_i64(v).len();
    let mut alt_d = Vec::new();
    let mut alt_v = Vec::new();
    let mut prev = 0i64;
    for &x in v {
        put_varint(&mut alt_d, zigzag(x.wrapping_sub(prev)));
        put_varint(&mut alt_v, zigzag(x));
        prev = x;
    }
    let est = |a: &Vec<u8>| if a.len() < 64 { a.len() + 8 } else { xz_compress(a, 6).len() };
    g.min(est(&alt_d)).min(est(&alt_v))
}

/// Split scaled values at `m = 10^k`: high part (smooth, delta-friendly) and
/// low digits (often a near-constant float-printing artifact).
fn split(sc: &[i64], k: u8) -> (Vec<i64>, Vec<i64>) {
    let m = POW10[k as usize];
    (sc.iter().map(|&v| v / m).collect(), sc.iter().map(|&v| v % m).collect())
}

/// A stream of number tokens.
pub fn put_nums(w: &mut Writer, nums: &[Num]) {
    let s = nums.iter().map(|n| n.dec).max().unwrap_or(0);
    let scaled: Option<Vec<i64>> =
        nums.iter().map(|n| n.mant.checked_mul(POW10[(s - n.dec) as usize])).collect();
    let pads: Vec<u8> = nums.iter().map(|n| n.pad).collect();
    match scaled {
        Some(sc) => {
            w.num.push(s);
            // decimals predicted from the value: shortest form (no trailing zeros)
            let res: Vec<u8> = nums
                .iter()
                .zip(&sc)
                .map(|(n, &v)| n.dec - (s - trailing_zeros10(v.unsigned_abs(), s)))
                .collect();
            // choose how many low decimal digits to split off (0 = none)
            let mut best = (ints_cost(&sc), 0u8);
            if sc.len() >= 256 {
                for k in 1..=s.min(17) {
                    let (hi, lo) = split(&sc, k);
                    let c = ints_cost(&hi) + ints_cost(&lo);
                    if c < best.0 {
                        best = (c, k);
                    }
                }
            }
            w.num.push(best.1);
            if best.1 == 0 {
                put_ints(w, &sc);
            } else {
                let (hi, lo) = split(&sc, best.1);
                put_ints(w, &hi);
                put_ints(w, &lo);
            }
            put_small(w, &res);
        }
        None => {
            w.num.push(255);
            let m: Vec<i64> = nums.iter().map(|n| n.mant).collect();
            put_ints(w, &m);
            put_small(w, &nums.iter().map(|n| n.dec).collect::<Vec<_>>());
        }
    }
    put_small(w, &pads);
}

pub fn get_nums(r: &mut Reader, n: usize) -> Vec<Num> {
    let s = r.byte();
    let vals = if s == 255 {
        get_ints(r, n)
    } else {
        let k = r.byte();
        if k == 0 {
            get_ints(r, n)
        } else {
            let hi = get_ints(r, n);
            let lo = get_ints(r, n);
            let m = POW10[k as usize];
            hi.iter().zip(&lo).map(|(h, l)| h * m + l).collect()
        }
    };
    let d = get_small(r, n);
    let pads = get_small(r, n);
    (0..n)
        .map(|i| {
            if s == 255 {
                Num { mant: vals[i], dec: d[i], pad: pads[i] }
            } else {
                let v = vals[i];
                let dec = d[i] + (s - trailing_zeros10(v.unsigned_abs(), s));
                Num { mant: v / POW10[(s - dec) as usize], dec, pad: pads[i] }
            }
        })
        .collect()
}

/// A byte value in 1..32 (not \t \n \r) absent from `data`, for placeholders.
pub fn free_bytes(data: &[u8], k: usize) -> Option<Vec<u8>> {
    let mut seen = [false; 256];
    for &b in data {
        seen[b as usize] = true;
    }
    let v: Vec<u8> = (1u8..32)
        .chain(128..=255)
        .filter(|&b| !seen[b as usize] && b != b'\t' && b != b'\n' && b != b'\r')
        .take(k)
        .collect();
    (v.len() == k).then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_tokens() {
        for s in ["-12.50", "007", "a-3", "x-0.0y", "1.2.3", "0", "-0", "12345678901234567890123", "3.", ".5", "-1-2"] {
            let mut sk = Vec::new();
            let mut nums = Vec::new();
            parse(s.as_bytes(), 1, &mut sk, &mut nums);
            let mut out = Vec::new();
            let mut it = nums.iter();
            for &b in &sk {
                if b == 1 { format(*it.next().unwrap(), &mut out) } else { out.push(b) }
            }
            assert_eq!(String::from_utf8(out).unwrap(), s);
        }
    }
}
