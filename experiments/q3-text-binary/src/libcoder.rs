//! Baseline-of-ours: plain order-k context model on gorilla-tsc's own
//! LZMA-style binary range coder (`Prob`: 11-bit, shift-5), no mixing.
//! One binary tree of 255 `Prob`s per order-k byte context (direct indexed).

use crate::util::{get_varint, put_varint};
use gorilla_tsc::{BitReader, BitWriter, Prob, RangeDecoder, RangeEncoder};

fn ctx(hist: u32, order: u32) -> usize {
    (hist & ((1u32 << (8 * order)) - 1)) as usize
}

pub fn compress(data: &[u8], order: u32) -> Vec<u8> {
    let mut probs = vec![Prob::default(); 256usize << (8 * order)];
    let mut w = BitWriter::new();
    {
        let mut enc = RangeEncoder::new(&mut w);
        let mut hist = 0u32;
        for &c in data {
            let base = ctx(hist, order) << 8;
            let mut node = 1usize;
            for i in (0..8).rev() {
                let bit = (c >> i) & 1 == 1;
                enc.bit(&mut probs[base + node], bit);
                node = (node << 1) | usize::from(bit);
            }
            hist = (hist << 8) | u32::from(c);
        }
        enc.finish();
    }
    let mut out = Vec::new();
    put_varint(&mut out, data.len() as u64);
    out.push(order as u8);
    out.extend_from_slice(&w.finish());
    out
}

pub fn decompress(inp: &[u8]) -> Vec<u8> {
    let mut pos = 0;
    let n = get_varint(inp, &mut pos) as usize;
    let order = u32::from(inp[pos]);
    pos += 1;
    let mut probs = vec![Prob::default(); 256usize << (8 * order)];
    let mut r = BitReader::new(&inp[pos..]);
    let mut dec = RangeDecoder::new(&mut r).expect("range decoder");
    let mut out = Vec::with_capacity(n);
    let mut hist = 0u32;
    for _ in 0..n {
        let base = ctx(hist, order) << 8;
        let mut node = 1usize;
        for _ in 0..8 {
            let bit = dec.bit(&mut probs[base + node]).expect("bit");
            node = (node << 1) | usize::from(bit);
        }
        let c = (node & 0xff) as u8;
        out.push(c);
        hist = (hist << 8) | u32::from(c);
    }
    out
}
