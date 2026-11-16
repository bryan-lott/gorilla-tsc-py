//! Idea 2: CLP/Drain-like log transform.
//!
//! Each line: an optional leading ISO timestamp (`[`? `YYYY-MM-DD[T ]HH:MM:SS`)
//! becomes epoch seconds in one global stream (gorilla-tsc `Auto`). The rest is
//! split on spaces; tokens without digits stay in the template; id-like tokens
//! (hex runs, 0x.., many letter/digit transitions) become string variables;
//! other tokens with digits are parsed into number tokens. The template
//! (static text + placeholders) goes to a dictionary; numbers go to one stream
//! per (template, slot) (rare templates pooled per slot); strings are grouped by
//! (template, slot) and handed to the back end with the template ids.

use crate::numtok::*;
use crate::util::*;
use std::collections::HashMap;

const POOL_MIN: usize = 16;

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn fmt_ts(t: i64, sep: u8, out: &mut Vec<u8>) {
    let days = t.div_euclid(86400);
    let s = t.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    out.extend_from_slice(
        format!("{y:04}-{m:02}-{d:02}{}{:02}:{:02}:{:02}", sep as char, s / 3600, s / 60 % 60, s % 60).as_bytes(),
    );
}

/// Parse a 19-byte ISO datetime; returns (epoch seconds, separator).
fn parse_ts(b: &[u8]) -> Option<(i64, u8)> {
    if b.len() < 19 {
        return None;
    }
    let dig = |r: std::ops::Range<usize>| -> Option<i64> {
        let mut v = 0;
        for &c in &b[r] {
            if !c.is_ascii_digit() {
                return None;
            }
            v = v * 10 + i64::from(c - b'0');
        }
        Some(v)
    };
    if b[4] != b'-' || b[7] != b'-' || !(b[10] == b'T' || b[10] == b' ') || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let (y, mo, d, h, mi, s) = (dig(0..4)?, dig(5..7)?, dig(8..10)?, dig(11..13)?, dig(14..16)?, dig(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let t = days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s;
    let mut chk = Vec::new();
    fmt_ts(t, b[10], &mut chk);
    (chk == b[..19]).then_some((t, b[10]))
}

fn is_var_token(t: &[u8]) -> bool {
    let core = t.iter().position(u8::is_ascii_alphanumeric).map_or(&t[..0], |i| &t[i..]);
    if core.len() > 2 && core[0] == b'0' && (core[1] == b'x' || core[1] == b'X') {
        return true;
    }
    let mut run = 0;
    let (mut has_d, mut has_l) = (false, false);
    let mut transitions = 0;
    let mut prev = 0u8; // 0 other, 1 digit, 2 letter
    for &c in t {
        if c.is_ascii_hexdigit() {
            run += 1;
            if c.is_ascii_digit() { has_d = true } else { has_l = true }
            if run >= 8 && has_d && has_l {
                return true;
            }
        } else {
            run = 0;
            has_d = false;
            has_l = false;
        }
        let k = if c.is_ascii_digit() { 1 } else if c.is_ascii_alphabetic() { 2 } else { 0 };
        if k != 0 && prev != 0 && k != prev {
            transitions += 1;
        }
        prev = k;
    }
    transitions >= 3
}

pub fn encode(data: &[u8], backend: Backend) -> Option<Vec<u8>> {
    let phs = free_bytes(data, 3)?;
    let (ph, ph_var, ph_ts) = (phs[0], phs[1], phs[2]);
    let trailing_nl = data.last() == Some(&b'\n');
    let body = if trailing_nl { &data[..data.len() - 1] } else { data };
    let lines: Vec<&[u8]> = if data.is_empty() { vec![] } else { body.split(|&b| b == b'\n').collect() };

    let mut dict: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut templates: Vec<Vec<u8>> = Vec::new();
    let mut tids = Vec::with_capacity(lines.len());
    let mut ts = Vec::new();
    let mut line_nums: Vec<Vec<Num>> = Vec::with_capacity(lines.len());
    let mut line_vars: Vec<Vec<&[u8]>> = Vec::with_capacity(lines.len());
    for line in &lines {
        let mut tpl = Vec::new();
        let mut rest: &[u8] = line;
        let br = usize::from(rest.first() == Some(&b'['));
        if let Some((t, sep)) = parse_ts(&rest[br..]) {
            tpl.extend_from_slice(&rest[..br]);
            tpl.push(ph_ts);
            tpl.push(sep);
            ts.push(t);
            rest = &rest[br + 19..];
        }
        let mut nums = Vec::new();
        let mut vars = Vec::new();
        for (i, tok) in rest.split(|&b| b == b' ').enumerate() {
            if i > 0 {
                tpl.push(b' ');
            }
            if !tok.iter().any(u8::is_ascii_digit) {
                tpl.extend_from_slice(tok);
            } else if is_var_token(tok) {
                tpl.push(ph_var);
                vars.push(tok);
            } else {
                parse(tok, ph, &mut tpl, &mut nums);
            }
        }
        let next = templates.len() as u32;
        let id = *dict.entry(tpl.clone()).or_insert_with(|| {
            templates.push(tpl);
            next
        });
        tids.push(id);
        line_nums.push(nums);
        line_vars.push(vars);
    }
    let mut count = vec![0usize; templates.len()];
    for &t in &tids {
        count[t as usize] += 1;
    }

    let mut header = vec![b'L', ph, ph_var, ph_ts, u8::from(trailing_nl)];
    put_varint(&mut header, lines.len() as u64);
    let mut w = Writer::default();
    w.section(templates.join(&b'\n'));
    w.section(pack_ids(&tids));
    put_ints(&mut w, &ts);
    // numbers: per (template, slot) for common templates, pooled per slot otherwise
    let mut streams: HashMap<(u32, usize), Vec<Num>> = HashMap::new();
    let mut vars: HashMap<(u32, usize), Vec<&[u8]>> = HashMap::new();
    for (i, &t) in tids.iter().enumerate() {
        let key_t = if count[t as usize] >= POOL_MIN { t } else { u32::MAX };
        for (k, n) in line_nums[i].iter().enumerate() {
            streams.entry((key_t, k)).or_default().push(*n);
        }
        for (k, v) in line_vars[i].iter().enumerate() {
            vars.entry((t, k)).or_default().push(v);
        }
    }
    let mut keys: Vec<_> = streams.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        put_nums(&mut w, &streams[&k]);
    }
    let mut vkeys: Vec<_> = vars.keys().copied().collect();
    vkeys.sort_unstable();
    let mut vs: Vec<u8> = Vec::new();
    for k in vkeys {
        for v in &vars[&k] {
            vs.extend_from_slice(v);
            vs.push(b'\n');
        }
    }
    w.section(vs);
    eprintln!(
        "  log: {} lines, {} templates, {} number streams ({} gorilla, {} values -> {} B)",
        lines.len(),
        templates.len(),
        w.stats.streams,
        w.stats.gorilla_streams,
        w.stats.gorilla_values,
        w.stats.gorilla_bytes
    );
    Some(w.finish(backend, header))
}

pub fn decode(inp: &[u8], backend: Backend) -> Vec<u8> {
    assert_eq!(inp[0], b'L');
    let (ph, ph_var, ph_ts, trailing_nl) = (inp[1], inp[2], inp[3], inp[4] != 0);
    let mut pos = 5;
    let nlines = get_varint(inp, &mut pos) as usize;
    let mut r = Reader::new(&inp[pos..], backend);
    let traw = r.section();
    let templates: Vec<&[u8]> = traw.split(|&b| b == b'\n').collect();
    let tids = unpack_ids(&r.section());
    let nts = tids.iter().filter(|&&t| templates[t as usize].contains(&ph_ts)).count();
    let ts = get_ints(&mut r, nts);
    let mut count = vec![0usize; templates.len()];
    for &t in &tids {
        count[t as usize] += 1;
    }
    let nph: Vec<usize> = templates.iter().map(|t| t.iter().filter(|&&b| b == ph).count()).collect();
    let nvar: Vec<usize> = templates.iter().map(|t| t.iter().filter(|&&b| b == ph_var).count()).collect();
    let mut sizes: HashMap<(u32, usize), usize> = HashMap::new();
    let mut vsizes: HashMap<(u32, usize), usize> = HashMap::new();
    for &t in &tids {
        let key_t = if count[t as usize] >= POOL_MIN { t } else { u32::MAX };
        for k in 0..nph[t as usize] {
            *sizes.entry((key_t, k)).or_default() += 1;
        }
        for k in 0..nvar[t as usize] {
            *vsizes.entry((t, k)).or_default() += 1;
        }
    }
    let mut keys: Vec<_> = sizes.keys().copied().collect();
    keys.sort_unstable();
    let mut streams: HashMap<(u32, usize), (Vec<Num>, usize)> = HashMap::new();
    for k in keys {
        let v = get_nums(&mut r, sizes[&k]);
        streams.insert(k, (v, 0));
    }
    let vraw = r.section();
    let mut vkeys: Vec<_> = vsizes.keys().copied().collect();
    vkeys.sort_unstable();
    let mut vit = vraw.split(|&b| b == b'\n');
    let mut vars: HashMap<(u32, usize), (Vec<&[u8]>, usize)> = HashMap::new();
    for k in vkeys {
        let v: Vec<&[u8]> = (0..vsizes[&k]).map(|_| vit.next().unwrap()).collect();
        vars.insert(k, (v, 0));
    }
    let mut out = Vec::with_capacity(inp.len() * 8);
    let mut ti = 0;
    for (i, &t) in tids.iter().enumerate().take(nlines) {
        let key_t = if count[t as usize] >= POOL_MIN { t } else { u32::MAX };
        let tpl = templates[t as usize];
        let (mut k, mut kv) = (0, 0);
        let mut j = 0;
        while j < tpl.len() {
            let b = tpl[j];
            if b == ph {
                let s = streams.get_mut(&(key_t, k)).unwrap();
                format(s.0[s.1], &mut out);
                s.1 += 1;
                k += 1;
            } else if b == ph_var {
                let s = vars.get_mut(&(t, kv)).unwrap();
                out.extend_from_slice(s.0[s.1]);
                s.1 += 1;
                kv += 1;
            } else if b == ph_ts {
                j += 1;
                fmt_ts(ts[ti], tpl[j], &mut out);
                ti += 1;
            } else {
                out.push(b);
            }
            j += 1;
        }
        if i + 1 < nlines || trailing_nl {
            out.push(b'\n');
        }
    }
    out
}
