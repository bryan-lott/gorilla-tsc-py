//! Locked "production profiles": decimal scaling + differences + one coder,
//! with every choice fixed up front instead of searched per block.
//!
//! Built from the library's public block coders (`RiceBlock`, `AdaptiveBlock`),
//! with its own small block framing. Every profile is verified lossless.

use gorilla_tsc::{AdaptiveBlock, BitReader, BitWriter, Order, RiceBlock, Sample, Word};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coder {
    Rice,
    Adaptive,
}

/// How each block's decimal exponent is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KPolicy {
    /// One exponent for the whole stream.
    Fixed(i32),
    /// Per block, from its first `m` values: the smallest exponent that
    /// scales as many of them as possible (ALP-style sampling).
    Sampled(usize),
    /// A locked exponent, but a block where it leaves more than 1% of
    /// values as exceptions falls back to a sampled one (if that is better).
    Guarded(i32),
    /// Per block, with no trial encoding: each value's smallest working
    /// exponent, then the smallest exponent covering all but `permille`/1000
    /// of the block.
    Covering(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    pub k: KPolicy,
    pub order: Order,
    pub coder: Coder,
}

impl Profile {
    #[must_use]
    pub fn label(&self) -> String {
        let k = match self.k {
            KPolicy::Fixed(k) => format!("k={k}"),
            KPolicy::Sampled(m) => format!("k=sampled({m})"),
            KPolicy::Guarded(k) => format!("k={k} guarded"),
            KPolicy::Covering(p) => format!("k=covering({p}/1000)"),
        };
        let o = match self.order {
            Order::First => "1st-diff",
            Order::Second => "2nd-diff",
            Order::Zero => "no-diff",
        };
        let c = match self.coder {
            Coder::Rice => "rice",
            Coder::Adaptive => "adaptive",
        };
        format!("{k} {o} {c}")
    }
}

const K_BITS: u32 = 6;
const K_OFFSET: i32 = 31;

fn bit_len(n: usize) -> u32 {
    usize::BITS - n.leading_zeros()
}

/// Smallest exponent that scales the most of `sample`.
pub fn sample_k<T: Sample>(sample: &[T]) -> i32 {
    let mut best = (0usize, 0i32);
    for k in -T::MAX_SCALE..=T::MAX_SCALE {
        let ok = sample.iter().filter(|x| x.to_scaled(k).is_some()).count();
        if ok > best.0 {
            best = (ok, k);
        }
        if ok == sample.len() {
            break;
        }
    }
    best.1
}

/// Smallest exponent at which `v` scales, searching up from 0 first.
fn min_k<T: Sample>(v: T) -> Option<i32> {
    if v.to_scaled(0).is_some() {
        let mut k = 0;
        while k > -T::MAX_SCALE && v.to_scaled(k - 1).is_some() {
            k -= 1;
        }
        Some(k)
    } else {
        (1..=T::MAX_SCALE).chain((-T::MAX_SCALE..0).rev()).find(|&k| v.to_scaled(k).is_some())
    }
}

/// Smallest exponent at least `1 - permille/1000` of `items` can use.
pub fn covering_k<T: Sample>(items: &[T], permille: usize) -> i32 {
    let span = usize::try_from(2 * T::MAX_SCALE + 1).unwrap();
    let mut hist = vec![0usize; span];
    for &v in items {
        if let Some(k) = min_k(v) {
            hist[usize::try_from(k + T::MAX_SCALE).unwrap()] += 1;
        }
    }
    let need = items.len() - items.len() * permille / 1000;
    let mut acc = 0;
    for (i, c) in hist.iter().enumerate() {
        acc += c;
        if acc >= need {
            return i32::try_from(i).unwrap() - T::MAX_SCALE;
        }
    }
    0
}

fn residuals(ints: &[u64], order: Order) -> Vec<u64> {
    let mut prev_d = 0u64;
    ints.windows(2)
        .map(|w| {
            let d = w[1].wrapping_sub(w[0]);
            match order {
                Order::Zero => w[1],
                Order::First => d,
                Order::Second => {
                    let r = d.wrapping_sub(prev_d);
                    prev_d = d;
                    r
                }
            }
        })
        .collect()
}

fn integrate(first: u64, res: &[u64], order: Order, out: &mut Vec<u64>) {
    let (mut x, mut d) = (first, 0u64);
    out.push(x);
    for &r in res {
        x = match order {
            Order::Zero => r,
            Order::First => x.wrapping_add(r),
            Order::Second => {
                d = d.wrapping_add(r);
                x.wrapping_add(d)
            }
        };
        out.push(x);
    }
}

fn scale_block<T: Sample>(items: &[T], k: i32) -> (Vec<u64>, Vec<(usize, u64)>) {
    let mut ints = Vec::with_capacity(items.len());
    let mut exceptions = Vec::new();
    let mut prev = 0u64;
    for (i, &x) in items.iter().enumerate() {
        match x.to_scaled(k) {
            Some(n) => {
                prev = n.cast_unsigned();
                ints.push(prev);
            }
            None => {
                exceptions.push((i, x.to_word().to_u64()));
                ints.push(prev);
            }
        }
    }
    (ints, exceptions)
}

pub fn encode_block<T: Sample>(items: &[T], p: Profile, out: &mut BitWriter) {
    let (k, ints, exceptions) = match p.k {
        KPolicy::Fixed(k) => {
            let (i, e) = scale_block(items, k);
            (k, i, e)
        }
        KPolicy::Sampled(m) => {
            let k = sample_k(&items[..m.min(items.len())]);
            let (i, e) = scale_block(items, k);
            (k, i, e)
        }
        KPolicy::Covering(permille) => {
            let k = covering_k(items, permille);
            let (i, e) = scale_block(items, k);
            (k, i, e)
        }
        KPolicy::Guarded(k) => {
            let (i, e) = scale_block(items, k);
            if e.len() * 100 > items.len() {
                let k2 = sample_k(&items[..64.min(items.len())]);
                let (i2, e2) = scale_block(items, k2);
                if e2.len() < e.len() { (k2, i2, e2) } else { (k, i, e) }
            } else {
                (k, i, e)
            }
        }
    };
    out.write(u64::try_from(k + K_OFFSET).unwrap(), K_BITS);
    let n = items.len();
    out.write(exceptions.len() as u64, bit_len(n));
    for &(pos, raw) in &exceptions {
        out.write(pos as u64, bit_len(n - 1));
        out.write(raw, T::Word::BITS);
    }
    // First integer: zig-zag, 7-bit length prefix.
    let z = ints[0].zigzag();
    let len = u64::BITS - z.leading_zeros();
    out.write(u64::from(len), 7);
    if len == 64 {
        out.write(z, 64);
    } else if len > 0 {
        out.write(z, len);
    }
    let r = residuals(&ints, p.order);
    match p.coder {
        Coder::Rice => RiceBlock::encode(&r, out),
        Coder::Adaptive => AdaptiveBlock::encode(&r, out),
    }
}

#[allow(clippy::missing_panics_doc)]
pub fn decode_block<T: Sample>(n: usize, p: Profile, input: &mut BitReader<'_>, out: &mut Vec<T>) {
    let k = i32::try_from(input.read(K_BITS).unwrap()).unwrap() - K_OFFSET;
    let count = usize::try_from(input.read(bit_len(n)).unwrap()).unwrap();
    let mut exceptions = Vec::with_capacity(count);
    for _ in 0..count {
        let pos = usize::try_from(input.read(bit_len(n - 1)).unwrap()).unwrap();
        exceptions.push((pos, input.read(T::Word::BITS).unwrap()));
    }
    let len = u32::try_from(input.read(7).unwrap()).unwrap();
    let z = if len == 0 { 0 } else { input.read(len).unwrap() };
    let first = z.unzigzag();
    let mut r: Vec<u64> = Vec::with_capacity(n - 1);
    match p.coder {
        Coder::Rice => RiceBlock::decode(n - 1, input, &mut r).unwrap(),
        Coder::Adaptive => AdaptiveBlock::decode(n - 1, input, &mut r).unwrap(),
    }
    let mut ints = Vec::with_capacity(n);
    integrate(first, &r, p.order, &mut ints);
    let mut exc = exceptions.into_iter().peekable();
    for (i, v) in ints.into_iter().enumerate() {
        match exc.next_if(|&(pos, _)| pos == i) {
            Some((_, raw)) => out.push(T::from_word(T::Word::from_u64_truncate(raw))),
            None => out.push(T::from_scaled(v.cast_signed(), k)),
        }
    }
}

/// Encode `items` in blocks of `block` with profile `p`.
#[must_use]
pub fn encode<T: Sample>(items: &[T], block: usize, p: Profile) -> BitWriter {
    let mut out = BitWriter::new();
    for b in items.chunks(block) {
        encode_block(b, p, &mut out);
    }
    out
}

#[must_use]
pub fn decode<T: Sample>(bytes: &[u8], n: usize, block: usize, p: Profile) -> Vec<T> {
    let mut input = BitReader::new(bytes);
    let mut out = Vec::with_capacity(n);
    let mut left = n;
    while left > 0 {
        let m = left.min(block);
        decode_block(m, p, &mut input, &mut out);
        left -= m;
    }
    out
}

// ------------------------------------------------------------ guarded lock

fn order_code(o: Order) -> u64 {
    match o {
        Order::First => 0,
        Order::Second => 1,
        Order::Zero => 2,
    }
}

fn order_from(c: u64) -> Order {
    match c {
        0 => Order::First,
        1 => Order::Second,
        _ => Order::Zero,
    }
}

fn block_bits<T: Sample>(b: &[T], p: Profile) -> BitWriter {
    let mut w = BitWriter::new();
    encode_block(b, p, &mut w);
    w
}

/// The best of a small candidate set on this block: the current exponent and
/// one sampled from the block's head, times every order and coder.
fn relearn<T: Sample>(b: &[T], cur: Profile) -> (Profile, BitWriter) {
    let KPolicy::Fixed(k0) = cur.k else { unreachable!() };
    let k1 = sample_k(&b[..32.min(b.len())]);
    let mut best: Option<(Profile, BitWriter)> = None;
    for k in if k1 == k0 { vec![k0] } else { vec![k0, k1] } {
        for order in [Order::First, Order::Second, Order::Zero] {
            for coder in [Coder::Rice, Coder::Adaptive] {
                let p = Profile { k: KPolicy::Fixed(k), order, coder };
                let w = block_bits(b, p);
                if best.as_ref().is_none_or(|(_, bw)| w.bit_len() < bw.bit_len()) {
                    best = Some((p, w));
                }
            }
        }
    }
    best.unwrap()
}

/// Locked profile that re-learns when a block costs more than `1 + tol`
/// times the previous block. Each block carries its order and coder (3 bits).
/// Returns the stream and the number of re-learns.
pub fn encode_guarded<T: Sample>(items: &[T], block: usize, init: Profile, tol: f64) -> (BitWriter, usize) {
    let mut out = BitWriter::new();
    let mut cur = init;
    let mut prev_bits: Option<usize> = None;
    let mut relearns = 0;
    for b in items.chunks(block) {
        let mut w = block_bits(b, cur);
        #[allow(clippy::cast_precision_loss)]
        let jumped = prev_bits.is_some_and(|pb| w.bit_len() as f64 > pb as f64 * (1.0 + tol));
        if jumped {
            relearns += 1;
            let (p, w2) = relearn(b, cur);
            if w2.bit_len() < w.bit_len() {
                cur = p;
                w = w2;
            }
        }
        prev_bits = Some(w.bit_len());
        out.write(order_code(cur.order), 2);
        out.write(u64::from(cur.coder == Coder::Adaptive), 1);
        out.append(&w);
    }
    (out, relearns)
}

pub fn decode_guarded<T: Sample>(bytes: &[u8], n: usize, block: usize) -> Vec<T> {
    let mut input = BitReader::new(bytes);
    let mut out = Vec::with_capacity(n);
    let mut left = n;
    while left > 0 {
        let m = left.min(block);
        let order = order_from(input.read(2).unwrap());
        let coder = if input.read(1).unwrap() == 1 { Coder::Adaptive } else { Coder::Rice };
        // k travels inside the block; KPolicy is unused when decoding.
        decode_block(m, Profile { k: KPolicy::Fixed(0), order, coder }, &mut input, &mut out);
        left -= m;
    }
    out
}

/// Locked profile re-learned every `r` blocks from the window just encoded
/// (its first and last block), over a small candidate set: the current
/// exponent, its neighbours and ones sampled from the window, times every
/// order and coder. Same stream format as [`encode_guarded`].
pub fn encode_windowed<T: Sample>(items: &[T], block: usize, init: Profile, r: usize) -> BitWriter {
    let mut out = BitWriter::new();
    let mut cur = init;
    let blocks: Vec<&[T]> = items.chunks(block).collect();
    for (i, b) in blocks.iter().enumerate() {
        if i > 0 && i % r == 0 {
            let window = [blocks[i - r], blocks[i - 1]];
            let KPolicy::Fixed(k0) = cur.k else { unreachable!() };
            let mut ks = vec![k0, k0 - 1, k0 + 1];
            for w in window {
                ks.push(sample_k(&w[..32.min(w.len())]));
            }
            ks.retain(|&k| (-T::MAX_SCALE..=T::MAX_SCALE).contains(&k));
            ks.sort_unstable();
            ks.dedup();
            let mut best: Option<(usize, Profile)> = None;
            for &k in &ks {
                for order in [Order::First, Order::Second, Order::Zero] {
                    for coder in [Coder::Rice, Coder::Adaptive] {
                        let p = Profile { k: KPolicy::Fixed(k), order, coder };
                        let bits: usize = window.iter().map(|w| block_bits(w, p).bit_len()).sum();
                        if best.is_none_or(|(bb, _)| bits < bb) {
                            best = Some((bits, p));
                        }
                    }
                }
            }
            cur = best.unwrap().1;
        }
        out.write(order_code(cur.order), 2);
        out.write(u64::from(cur.coder == Coder::Adaptive), 1);
        encode_block(b, cur, &mut out);
    }
    out
}
