//! Idea 1: typed/columnar transform for delimited numeric text (CSV/TSV).
//!
//! Rows split on '\n', fields on the detected delimiter. Each field becomes a
//! skeleton (non-numeric bytes, a placeholder per number) plus number tokens.
//! Per column: skeleton dictionary + id stream (back end), and one number
//! stream per slot index, coded with gorilla-tsc `Auto` on scaled integers.
//! Columns with too many distinct skeletons are stored as raw text.

use crate::numtok::*;
use crate::util::*;
use std::collections::HashMap;

fn detect_delim(data: &[u8]) -> u8 {
    let lines: Vec<&[u8]> = data.split(|&b| b == b'\n').take(1000).filter(|l| !l.is_empty()).collect();
    let mut best = (0u8, 0usize);
    for d in [b',', b'\t', b';', b'|'] {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for l in &lines {
            *counts.entry(l.iter().filter(|&&b| b == d).count()).or_default() += 1;
        }
        let (&mode, &freq) = counts.iter().max_by_key(|&(k, v)| (*v, *k)).unwrap_or((&0, &0));
        if mode > 0 && freq * 10 >= lines.len() * 9 && mode > best.1 {
            best = (d, mode);
        }
    }
    best.0 // 0 = no delimiter (one field per line)
}

pub fn encode(data: &[u8], backend: Backend) -> Option<Vec<u8>> {
    let ph = free_bytes(data, 1)?[0];
    let delim = detect_delim(data);
    let trailing_nl = data.last() == Some(&b'\n');
    let body = if trailing_nl { &data[..data.len() - 1] } else { data };
    let rows: Vec<&[u8]> = if data.is_empty() { vec![] } else { body.split(|&b| b == b'\n').collect() };
    let fields: Vec<Vec<&[u8]>> = rows
        .iter()
        .map(|r| if delim == 0 { vec![*r] } else { r.split(|&b| b == delim).collect() })
        .collect();
    let ncols = fields.iter().map(Vec::len).max().unwrap_or(0);

    let mut out = vec![b'C', ph, delim, u8::from(trailing_nl)];
    put_varint(&mut out, rows.len() as u64);
    put_varint(&mut out, ncols as u64);
    let nf: Vec<u32> = fields.iter().map(|f| f.len() as u32).collect();
    put_bytes(&mut out, &backend.compress(&pack_ids(&nf)));

    for c in 0..ncols {
        let col: Vec<&[u8]> = fields.iter().filter_map(|f| f.get(c).copied()).collect();
        let mut dict: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut order: Vec<Vec<u8>> = Vec::new();
        let mut ids = Vec::with_capacity(col.len());
        let mut slots: Vec<Vec<Num>> = Vec::new();
        let limit = (col.len() / 4).max(1024);
        let mut text = false;
        for f in &col {
            let mut sk = Vec::new();
            let mut nums = Vec::new();
            parse(f, ph, &mut sk, &mut nums);
            let next = order.len() as u32;
            let id = *dict.entry(sk.clone()).or_insert_with(|| {
                order.push(sk);
                next
            });
            if order.len() > limit {
                text = true;
                break;
            }
            ids.push(id);
            for (k, n) in nums.into_iter().enumerate() {
                if slots.len() <= k {
                    slots.push(Vec::new());
                }
                slots[k].push(n);
            }
        }
        // Each column is coded independently, so the typed-vs-text choice is exact.
        let text_c = backend.compress(&col.join(&b'\n'));
        let typed_c = (!text).then(|| {
            let mut w = Writer::default();
            w.section(order.join(&b'\n'));
            w.section(pack_ids(&ids));
            put_varint(&mut w.num, slots.len() as u64);
            for s in &slots {
                put_nums(&mut w, s);
            }
            w.finish(backend, Vec::new())
        });
        match typed_c {
            Some(t) if t.len() < text_c.len() => {
                out.push(0);
                put_bytes(&mut out, &t);
            }
            _ => {
                out.push(1);
                put_bytes(&mut out, &text_c);
            }
        }
    }
    Some(out)
}

pub fn decode(inp: &[u8], backend: Backend) -> Vec<u8> {
    assert_eq!(inp[0], b'C');
    let (ph, delim, trailing_nl) = (inp[1], inp[2], inp[3] != 0);
    let mut pos = 4;
    let nrows = get_varint(inp, &mut pos) as usize;
    let ncols = get_varint(inp, &mut pos) as usize;
    let nf = unpack_ids(&backend.decompress(get_bytes(inp, &mut pos)));
    // cells[c] = rendered fields of column c, in row order
    let mut cells: Vec<Vec<Vec<u8>>> = Vec::with_capacity(ncols);
    for c in 0..ncols {
        let n = nf.iter().filter(|&&k| k as usize > c).count();
        let mode = inp[pos];
        pos += 1;
        let body = get_bytes(inp, &mut pos);
        if mode == 1 {
            let s = backend.decompress(body);
            cells.push(if n == 0 { vec![] } else { s.split(|&b| b == b'\n').map(<[u8]>::to_vec).collect() });
            continue;
        }
        let mut r = Reader::new(body, backend);
        let dict_raw = r.section();
        let dict: Vec<&[u8]> = dict_raw.split(|&b| b == b'\n').collect();
        let ids = unpack_ids(&r.section());
        let nslots = r.varint() as usize;
        let counts: Vec<usize> = (0..nslots)
            .map(|k| ids.iter().filter(|&&id| dict[id as usize].iter().filter(|&&b| b == ph).count() > k).count())
            .collect();
        let slots: Vec<Vec<Num>> = counts.iter().map(|&n| get_nums(&mut r, n)).collect();
        let mut cur = vec![0usize; nslots];
        let mut col = Vec::with_capacity(n);
        for &id in &ids {
            let mut out = Vec::new();
            let mut k = 0;
            for &b in dict[id as usize] {
                if b == ph {
                    format(slots[k][cur[k]], &mut out);
                    cur[k] += 1;
                    k += 1;
                } else {
                    out.push(b);
                }
            }
            col.push(out);
        }
        cells.push(col);
    }
    let mut out = Vec::new();
    let mut cur = vec![0usize; ncols];
    for (i, &k) in nf.iter().enumerate().take(nrows) {
        for c in 0..k as usize {
            if c > 0 {
                out.push(delim);
            }
            out.extend_from_slice(&cells[c][cur[c]]);
            cur[c] += 1;
        }
        if i + 1 < nrows || trailing_nl {
            out.push(b'\n');
        }
    }
    out
}
