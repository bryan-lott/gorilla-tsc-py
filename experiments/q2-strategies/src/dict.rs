//! B2: dictionary coding of low-cardinality blocks.
//!
//! Block layout: `d - 1` (bit_len(n) bits); dictionary flag (1 bit: 0 =
//! DecimalRice, 1 = DeltaRice on raw words) and the dictionary sorted by value;
//! if `d > 1`, an index-coder id (3 bits) and the indices.

use gorilla_tsc::{AdaptiveBlock, BitReader, BitWriter, BlockCodec, DecimalRice, DecodeError, DeltaRice, RiceBlock};

use crate::codec::{Cand, bit_len};

pub struct Dict {
    /// Decline blocks with more than `max_frac * n` distinct values.
    pub max_frac: f64,
}

const IDX_BITS: u32 = 3;

fn mtf_forward(idx: &[u64], d: usize) -> Vec<u64> {
    let mut order: Vec<u64> = (0..d as u64).collect();
    idx.iter()
        .map(|&i| {
            let pos = order.iter().position(|&x| x == i).unwrap();
            order.remove(pos);
            order.insert(0, i);
            pos as u64
        })
        .collect()
}

fn mtf_inverse(ranks: &[u64], d: usize) -> Vec<u64> {
    let mut order: Vec<u64> = (0..d as u64).collect();
    ranks
        .iter()
        .map(|&r| {
            let i = order.remove(r as usize);
            order.insert(0, i);
            i
        })
        .collect()
}

fn diffs(idx: &[u64]) -> Vec<u64> {
    let mut prev = 0u64;
    idx.iter()
        .map(|&i| {
            let d = i.wrapping_sub(prev);
            prev = i;
            d
        })
        .collect()
}

fn undiffs(d: &[u64]) -> Vec<u64> {
    let mut prev = 0u64;
    d.iter()
        .map(|&x| {
            prev = prev.wrapping_add(x);
            prev
        })
        .collect()
}

/// B6: implicit dictionary via a move-to-front cache of values already seen in
/// the block. Each value is either "new" (symbol 0; the new values form a
/// subsequence coded with DecimalRice or DeltaRice) or a repeat (symbol =
/// MTF rank + 1). Unlike [`Dict`] there is no explicit dictionary and blocks
/// with many singletons are fine: every distinct value is coded once, in order.
///
/// Layout: number of new values (bit_len(n) bits); codec flag (1 bit) + new
/// values; symbol coder flag (1 bit: 0 = Rice, 1 = adaptive) + symbols.
pub struct RepeatCache;

/// Fenwick tree over time steps marking each distinct value's last use, so
/// move-to-front ranks cost O(log n) for any block size.
struct Fenwick(Vec<i32>);

impl Fenwick {
    fn add(&mut self, mut i: usize, d: i32) {
        i += 1;
        while i < self.0.len() {
            self.0[i] += d;
            i += i & i.wrapping_neg();
        }
    }
    /// Sum of marks at positions < i.
    fn prefix(&self, mut i: usize) -> i32 {
        let mut s = 0;
        while i > 0 {
            s += self.0[i];
            i &= i - 1;
        }
        s
    }
    /// Position of the k-th mark (k >= 1).
    fn kth(&self, mut k: i32) -> usize {
        let mut pos = 0;
        let mut step = (self.0.len() - 1).next_power_of_two();
        while step > 0 {
            if pos + step < self.0.len() && self.0[pos + step] < k {
                pos += step;
                k -= self.0[pos];
            }
            step >>= 1;
        }
        pos
    }
}

impl Cand for RepeatCache {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        let n = items.len();
        let mut last: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let mut fw = Fenwick(vec![0; n + 1]);
        let mut fresh = Vec::new();
        let mut syms = Vec::with_capacity(n);
        for (i, &w) in items.iter().enumerate() {
            if let Some(&t) = last.get(&w) {
                let rank = fw.prefix(i) - fw.prefix(t + 1);
                syms.push(rank as u64 + 1);
                fw.add(t, -1);
            } else {
                fresh.push(w);
                syms.push(0);
            }
            fw.add(i, 1);
            last.insert(w, i);
        }
        if fresh.len() == n {
            return false; // no repeats: plain Decimal is strictly better
        }
        out.write(fresh.len() as u64, bit_len(n));
        let mut a = BitWriter::new();
        DecimalRice::<f64>::default().encode_block(&fresh, &mut a);
        let mut b = BitWriter::new();
        DeltaRice.encode_block(&fresh, &mut b);
        let (flag, best) = if a.bit_len() <= b.bit_len() { (0, a) } else { (1, b) };
        out.write(flag, 1);
        out.append(&best);
        let mut r = BitWriter::new();
        RiceBlock::encode(&syms, &mut r);
        let mut ad = BitWriter::new();
        AdaptiveBlock::encode(&syms, &mut ad);
        let (flag, best) = if r.bit_len() <= ad.bit_len() { (0, r) } else { (1, ad) };
        out.write(flag, 1);
        out.append(&best);
        true
    }

    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        let m = inp.read(bit_len(n))? as usize;
        let mut fresh = Vec::with_capacity(m);
        if inp.read_bit()? {
            DeltaRice.decode_block(m, inp, &mut fresh)?;
        } else {
            DecimalRice::<f64>::default().decode_block(m, inp, &mut fresh)?;
        }
        let mut syms: Vec<u64> = Vec::with_capacity(n);
        if inp.read_bit()? {
            AdaptiveBlock::decode(n, inp, &mut syms)?;
        } else {
            RiceBlock::decode(n, inp, &mut syms)?;
        }
        let mut fw = Fenwick(vec![0; n + 1]);
        let mut at: Vec<u64> = Vec::with_capacity(n); // value used at each time step
        let mut next = fresh.into_iter();
        let mut marks = 0i32;
        for (i, s) in syms.into_iter().enumerate() {
            let w = if s == 0 {
                marks += 1;
                next.next().ok_or(DecodeError::Corrupt("too few new values"))?
            } else {
                let r = (s - 1) as i32;
                if r >= marks {
                    return Err(DecodeError::Corrupt("cache rank"));
                }
                let t = fw.kth(marks - r);
                fw.add(t, -1);
                at[t]
            };
            fw.add(i, 1);
            at.push(w);
            out.push(w);
        }
        Ok(())
    }
}

impl Cand for Dict {
    fn enc(&self, items: &[u64], out: &mut BitWriter) -> bool {
        let n = items.len();
        let mut dict: Vec<u64> = items.to_vec();
        dict.sort_unstable_by(|a, b| f64::from_bits(*a).total_cmp(&f64::from_bits(*b)));
        dict.dedup();
        let d = dict.len();
        if d as f64 > self.max_frac * n as f64 {
            return false;
        }
        out.write((d - 1) as u64, bit_len(n));
        let mut a = BitWriter::new();
        DecimalRice::<f64>::default().encode_block(&dict, &mut a);
        let mut b = BitWriter::new();
        DeltaRice.encode_block(&dict, &mut b);
        if a.bit_len() <= b.bit_len() {
            out.write(0, 1);
            out.append(&a);
        } else {
            out.write(1, 1);
            out.append(&b);
        }
        if d == 1 {
            return true;
        }
        let idx: Vec<u64> = items
            .iter()
            .map(|w| {
                dict.binary_search_by(|p| f64::from_bits(*p).total_cmp(&f64::from_bits(*w))).unwrap() as u64
            })
            .collect();
        let width = bit_len(d - 1);
        let mut options: Vec<BitWriter> = Vec::with_capacity(6);
        let mut w = BitWriter::new();
        for &i in &idx {
            w.write(i, width);
        }
        options.push(w);
        let mtf = mtf_forward(&idx, d);
        let dif = diffs(&idx);
        for (seq, adaptive) in [(&idx, false), (&idx, true), (&dif, true), (&mtf, true), (&mtf, false)] {
            let mut w = BitWriter::new();
            if adaptive {
                AdaptiveBlock::encode(seq, &mut w);
            } else {
                RiceBlock::encode(seq, &mut w);
            }
            options.push(w);
        }
        let (k, best) = options.iter().enumerate().min_by_key(|(_, w)| w.bit_len()).unwrap();
        out.write(k as u64, IDX_BITS);
        out.append(best);
        true
    }

    fn dec(&self, n: usize, inp: &mut BitReader<'_>, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        let d = inp.read(bit_len(n))? as usize + 1;
        let mut dict = Vec::with_capacity(d);
        if inp.read_bit()? {
            DeltaRice.decode_block(d, inp, &mut dict)?;
        } else {
            DecimalRice::<f64>::default().decode_block(d, inp, &mut dict)?;
        }
        if d == 1 {
            out.extend(std::iter::repeat_n(dict[0], n));
            return Ok(());
        }
        let k = inp.read(IDX_BITS)?;
        let mut seq = Vec::with_capacity(n);
        let idx = match k {
            0 => {
                let width = bit_len(d - 1);
                for _ in 0..n {
                    seq.push(inp.read(width)?);
                }
                seq
            }
            1 | 5 => {
                RiceBlock::decode(n, inp, &mut seq)?;
                if k == 5 { mtf_inverse(&seq, d) } else { seq }
            }
            2..=4 => {
                AdaptiveBlock::decode(n, inp, &mut seq)?;
                match k {
                    3 => undiffs(&seq),
                    4 => mtf_inverse(&seq, d),
                    _ => seq,
                }
            }
            _ => return Err(DecodeError::Corrupt("index coder")),
        };
        for i in idx {
            out.push(*dict.get(i as usize).ok_or(DecodeError::Corrupt("dict index"))?);
        }
        Ok(())
    }
}
