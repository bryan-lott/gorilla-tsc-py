//! Q2: other compression strategies and external baselines, measured.
//!
//! `cargo run --release -- [--sets a,b] [--methods a,b] [--jobs N] [--min-time S]`
//! Every method is encoded, decoded and compared bit-exactly (f64::to_bits)
//! with the input; sizes are the full compressed byte counts.

mod agcd;
mod alprd;
mod baselines;
mod codec;
mod ctx;
mod dict;
mod lpc;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bench_common::{FloatSet, float_sets};
use gorilla_tsc::{CheckpointInterval, Config, Mode, decode, encode};
use rayon::prelude::*;
use serde_json::{Value, json};

use codec::{Cand, Composite, Lib, auto_cands};

type EncFn = Box<dyn Fn(&[f64]) -> (Vec<u8>, Option<Vec<usize>>) + Send + Sync>;
type DecFn = Box<dyn Fn(&[u8], usize) -> Vec<f64> + Send + Sync>;

struct Method {
    name: String,
    description: String,
    kind: &'static str,
    enc: EncFn,
    dec: DecFn,
}

fn m(name: &str, kind: &'static str, description: &str, enc: EncFn, dec: DecFn) -> Method {
    Method { name: name.into(), description: description.into(), kind, enc, dec }
}

fn lib_method(name: &str, kind: &'static str, desc: &str, mode: Mode, block: Option<u32>) -> Method {
    m(
        name,
        kind,
        desc,
        Box::new(move |v| {
            let b = block.unwrap_or_else(|| u32::try_from(v.len().max(1)).unwrap());
            let cfg = Config::new(mode).with_interval(CheckpointInterval::new(b).unwrap());
            (encode(v, cfg), None)
        }),
        Box::new(|b, _| decode::<f64>(b).unwrap()),
    )
}

fn composite(name: &str, kind: &'static str, desc: &str, make: impl Fn() -> Vec<Box<dyn Cand>> + Send + Sync + 'static) -> Method {
    composite_b(name, kind, desc, 1000, make)
}

/// `block` 0 = one block for the whole dataset.
fn composite_b(name: &str, kind: &'static str, desc: &str, block: usize, make: impl Fn() -> Vec<Box<dyn Cand>> + Send + Sync + 'static) -> Method {
    let c = Arc::new(Composite::new(make(), block));
    let c2 = c.clone();
    m(
        name,
        kind,
        desc,
        Box::new(move |v| {
            let (b, w) = c.encode(v);
            (b, Some(w))
        }),
        Box::new(move |b, _| c2.decode(b).unwrap()),
    )
}

fn ctx_method(name: &str, desc: &str, s: ctx::CtxStream) -> Method {
    let s2 = s.clone();
    m(name, "idea", desc, Box::new(move |v| (s.encode(v), None)), Box::new(move |b, _| s2.decode(b).unwrap()))
}

fn with_auto(extra: Vec<Box<dyn Cand>>) -> Vec<Box<dyn Cand>> {
    let mut v = auto_cands();
    v.extend(extra);
    v
}

/// All methods for one dataset (`prior` is trained on the other datasets).
fn methods(prior: &Arc<Vec<u16>>) -> Vec<Method> {
    use baselines::*;
    let mut v = vec![
        m("zstd-3", "baseline", "zstd level 3 on raw little-endian f64 bytes", Box::new(|v| (zstd_c(&raw(v), 3), None)), Box::new(|b, n| unraw(&zstd_d(b, n * 8)))),
        m("zstd-19", "baseline", "zstd level 19 on raw little-endian f64 bytes", Box::new(|v| (zstd_c(&raw(v), 19), None)), Box::new(|b, n| unraw(&zstd_d(b, n * 8)))),
        m("shuffle-zstd-19", "baseline", "Blosc-style byte shuffle (8 byte planes) then zstd level 19", Box::new(|v| (zstd_c(&shuffle(v), 19), None)), Box::new(|b, n| unshuffle(&zstd_d(b, n * 8)))),
        m("xz-9", "baseline", "xz/LZMA2 preset 9 on raw little-endian f64 bytes", Box::new(|v| (xz_c(&raw(v)), None)), Box::new(|b, _| unraw(&xz_d(b)))),
        m("pcodec", "baseline", "pcodec 0.4 standalone, default compression level 8, whole dataset", Box::new(|v| (pco_c(v), None)), Box::new(|b, _| pco_d(b))),
        m("shuffle-zstd-19-b1000", "baseline", "byte shuffle + zstd-19 on independent blocks of 1000 values (2-byte length per block)", Box::new(|v| (blocked(v, 1000, |c| zstd_c(&shuffle(c), 19)), None)), Box::new(|b, _| unblocked(b, |c| unshuffle(&zstd_d(c, 8000))))),
        m("pcodec-b1000", "baseline", "pcodec default config on independent blocks of 1000 values (2-byte length per block)", Box::new(|v| (blocked(v, 1000, pco_c), None)), Box::new(|b, _| unblocked(b, pco_d))),
        lib_method("gorilla-xor", "baseline", "library Mode::Xor (classic Gorilla), blocks of 1000", Mode::Xor, Some(1000)),
        lib_method("lib-decimal", "baseline", "library Mode::Decimal only, blocks of 1000", Mode::Decimal, Some(1000)),
        lib_method("auto", "baseline", "library Mode::Auto, blocks of 1000 (current best)", Mode::Auto, Some(1000)),
    ];
    for b in [128u32, 4096, 16384, 65536] {
        v.push(lib_method(&format!("auto-b{b}"), "ours", &format!("library Mode::Auto with checkpoint interval {b}"), Mode::Auto, Some(b)));
    }
    v.push(lib_method("auto-whole", "ours", "library Mode::Auto with one block for the whole dataset", Mode::Auto, None));
    v.push(composite("auto-reimpl", "ours", "sanity check: our composite framework with the 5 library candidates of Mode::Auto (must equal auto)", auto_cands));
    v.push(composite("dict", "ours", "dictionary coding alone: sorted distinct values (DecimalRice/DeltaRice) + indices (fixed/Rice/adaptive on index, index delta or move-to-front rank)", || vec![Box::new(dict::Dict { max_frac: 1.0 })]));
    v.push(composite("auto+dict", "ours", "auto plus dictionary coding as a 6th per-block candidate (any number of distinct values)", || with_auto(vec![Box::new(dict::Dict { max_frac: 1.0 })])));
    v.push(composite("decimal-split", "ours", "decimal scaling, then x = q*10^j + r (nearest remainder), q and r each DeltaRice-coded; j chosen per block (0 = no split)", || vec![Box::new(agcd::DecimalSplit)]));
    v.push(composite("auto+decsplit", "ours", "auto plus decimal-split as a 6th candidate", || with_auto(vec![Box::new(agcd::DecimalSplit)])));
    v.push(composite("repeat-cache", "ours", "move-to-front cache of values seen in the block: symbol 0 = new value (new values coded in order by DecimalRice/DeltaRice), else MTF rank + 1; symbols Rice/adaptive", || vec![Box::new(dict::RepeatCache), Box::new(Lib(gorilla_tsc::DecimalRice::<f64>::default()))]));
    v.push(composite("auto+repeat", "ours", "auto plus repeat-cache as a 6th candidate", || with_auto(vec![Box::new(dict::RepeatCache)])));
    v.push(composite_b("auto-whole+repeat", "idea", "one block for the whole dataset: auto candidates plus repeat-cache (exposes cross-block exact repeats; no random access)", 0, || with_auto(vec![Box::new(dict::RepeatCache)])));
    for b in [16384usize, 65536] {
        v.push(composite_b(&format!("auto+repeat-b{b}"), "idea", &format!("auto candidates plus repeat-cache on blocks of {b} (repeats visible across a larger window)"), b, || with_auto(vec![Box::new(dict::RepeatCache)])));
    }
    v.push(composite("alp-rd", "ours", "ALP-RD style split alone: left <= 16 bits, <= 8-entry dictionary with fixed-width codes + exceptions, right raw", || vec![Box::new(alprd::AlpRd)]));
    v.push(composite("split-adaptive", "ours", "split alone: left <= 32 bits, <= 255-entry dictionary + escape, range-coded (bit tree), right raw", || vec![Box::new(alprd::SplitAdaptive)]));
    v.push(composite("auto+alp-rd", "ours", "auto plus ALP-RD as a 6th candidate", || with_auto(vec![Box::new(alprd::AlpRd)])));
    v.push(composite("auto+split", "ours", "auto plus split-adaptive as a 6th candidate", || with_auto(vec![Box::new(alprd::SplitAdaptive)])));
    v.push(composite("decimal-lpc", "ours", "decimal scaling + per-block predictor choice: fixed order 0-3, LS-LPC order 1-3 on deltas or mean-removed values (Q12 coefs); Rice/adaptive residuals", || vec![Box::new(lpc::DecLpc { max_order: 3 })]));
    v.push(composite("auto+lpc", "ours", "auto plus decimal-lpc as a 6th candidate", || with_auto(vec![Box::new(lpc::DecLpc { max_order: 3 })])));
    let p = prior.clone();
    v.push(composite("auto+prior", "ours", "auto plus decimal with adaptive-coder probabilities initialised from a prior trained on the other datasets (leave-one-out; still block-independent)", move || with_auto(vec![Box::new(ctx::DecimalPrior { prior: p.clone(), shift: 5 })])));
    let p = prior.clone();
    v.push(composite("auto+all", "ours", "auto plus dict, repeat-cache, decimal-split, split-adaptive, decimal-lpc and prior-decimal candidates (4-bit selector)", move || {
        with_auto(vec![
            Box::new(dict::Dict { max_frac: 1.0 }),
            Box::new(agcd::DecimalSplit),
            Box::new(dict::RepeatCache),
            Box::new(alprd::SplitAdaptive),
            Box::new(lpc::DecLpc { max_order: 3 }),
            Box::new(ctx::DecimalPrior { prior: p.clone(), shift: 5 }),
        ])
    }));
    let base = ctx::CtxStream { block: 1000, init: ctx::Init::Fresh, shift: 5, reset: 1, single_coder: false };
    let pri = ctx::Init::Prior(prior.clone());
    v.push(ctx_method("ctx-fresh", "B5 reference: decimal + delta residuals, adaptive model reset every block (library behaviour), coder flushed per block", base.clone()));
    v.push(ctx_method("ctx-carry16", "adaptive model carried across 16 blocks (reset every 16 000 values), coder flushed per block", ctx::CtxStream { reset: 16, ..base.clone() }));
    v.push(ctx_method("ctx-carry-all", "adaptive model carried across the whole dataset, coder flushed per block (no random access)", ctx::CtxStream { reset: 0, ..base.clone() }));
    v.push(ctx_method("ctx-single", "model carried and one range-coded section for the whole dataset (no per-block flush, no random access)", ctx::CtxStream { reset: 0, single_coder: true, ..base.clone() }));
    v.push(ctx_method("ctx-prior", "model reset every block to a prior trained on the other datasets (block-independent)", ctx::CtxStream { init: pri.clone(), ..base.clone() }));
    v.push(ctx_method("ctx-prior-s6", "as ctx-prior with slower adaptation (shift 6)", ctx::CtxStream { init: pri, shift: 6, ..base.clone() }));
    v.push(ctx_method("ctx-fresh-s4", "as ctx-fresh with faster adaptation (shift 4)", ctx::CtxStream { shift: 4, ..base }));
    v
}

fn time_it<T>(min_time: f64, n_bytes: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best = f64::INFINITY;
    let mut total = 0.0;
    let mut reps = 0;
    loop {
        let t = Instant::now();
        let r = f();
        let s = t.elapsed().as_secs_f64();
        best = best.min(s);
        total += s;
        reps += 1;
        if total >= min_time || reps >= 5 || n_bytes > 8_000_000 {
            return (r, n_bytes as f64 / best / 1e6);
        }
    }
}

#[derive(Default)]
struct Results {
    methods: BTreeMap<String, (String, String, bool)>,
    datasets: Vec<(String, String, usize)>,
    bits: BTreeMap<String, BTreeMap<String, f64>>,
    enc: BTreeMap<String, BTreeMap<String, f64>>,
    dec: BTreeMap<String, BTreeMap<String, f64>>,
    wins: BTreeMap<String, BTreeMap<String, Vec<usize>>>,
}

fn results_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../results")
}

fn load_existing(path: &PathBuf, wins_path: &PathBuf) -> Results {
    let mut r = Results::default();
    if let Ok(w) = std::fs::read_to_string(wins_path) {
        r.wins = serde_json::from_str(&w).unwrap_or_default();
    }
    let Ok(s) = std::fs::read_to_string(path) else { return r };
    let Ok(v) = serde_json::from_str::<Value>(&s) else { return r };
    for mm in v["methods"].as_array().into_iter().flatten() {
        r.methods.insert(
            mm["name"].as_str().unwrap().into(),
            (mm["description"].as_str().unwrap().into(), mm["kind"].as_str().unwrap().into(), mm["lossless_verified"].as_bool().unwrap()),
        );
    }
    for d in v["datasets"].as_array().into_iter().flatten() {
        r.datasets.push((d["name"].as_str().unwrap().into(), d["kind"].as_str().unwrap().into(), d["n"].as_u64().unwrap() as usize));
    }
    for (key, tgt) in [("bits_per_value", &mut r.bits), ("enc_mbs", &mut r.enc), ("dec_mbs", &mut r.dec)] {
        for (mn, per) in v[key].as_object().into_iter().flatten() {
            for (dn, x) in per.as_object().unwrap() {
                tgt.entry(mn.clone()).or_default().insert(dn.clone(), x.as_f64().unwrap());
            }
        }
    }
    r
}

fn save(r: &Results, path: &PathBuf, wins_path: &PathBuf) {
    let methods: Vec<Value> = r
        .methods
        .iter()
        .map(|(n, (d, k, ok))| json!({"name": n, "description": d, "kind": k, "lossless_verified": ok}))
        .collect();
    let datasets: Vec<Value> = r.datasets.iter().map(|(n, k, c)| json!({"name": n, "kind": k, "n": c})).collect();
    let out = json!({"methods": methods, "datasets": datasets, "bits_per_value": r.bits, "enc_mbs": r.enc, "dec_mbs": r.dec});
    std::fs::write(path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    std::fs::write(wins_path, serde_json::to_string_pretty(&r.wins).unwrap()).unwrap();
}

fn family(name: &str) -> &str {
    if name.starts_with("Stocks") { "Stocks" } else { name }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let set_filter: Option<Vec<String>> = get("--sets").map(|s| s.split(',').map(String::from).collect());
    let meth_filter: Option<Vec<String>> = get("--methods").map(|s| s.split(',').map(String::from).collect());
    let jobs: usize = get("--jobs").map_or(1, |s| s.parse().unwrap());
    let min_time: f64 = get("--min-time").map_or(0.3, |s| s.parse().unwrap());
    let out_path = get("--out").map_or_else(|| results_dir().join("q2.json"), PathBuf::from);
    let wins_path = if get("--out").is_some() {
        out_path.with_extension("wins.json")
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out/block-wins.json")
    };
    std::fs::create_dir_all(wins_path.parent().unwrap()).unwrap();
    std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();

    let t0 = Instant::now();
    let all: Vec<FloatSet> = float_sets();
    eprintln!("loaded {} sets in {:.1}s", all.len(), t0.elapsed().as_secs_f64());

    // Leave-one-out priors (trained on every *other* dataset family).
    let counts: Vec<Vec<[u64; 2]>> = all.par_iter().map(|s| ctx::train_counts(&s.values, 1000)).collect();
    let priors: Vec<Arc<Vec<u16>>> = all
        .iter()
        .map(|s| {
            let mut c = vec![[0u64; 2]; ctx::NPROB];
            for (o, oc) in all.iter().zip(&counts) {
                if family(o.name) != family(s.name) {
                    for (a, b) in c.iter_mut().zip(oc) {
                        a[0] += b[0];
                        a[1] += b[1];
                    }
                }
            }
            Arc::new(ctx::prior_from(&c))
        })
        .collect();
    eprintln!("priors trained at {:.1}s", t0.elapsed().as_secs_f64());

    let res = Mutex::new(load_existing(&out_path, &wins_path));
    let selected: Vec<usize> = (0..all.len())
        .filter(|&i| set_filter.as_ref().is_none_or(|f| f.iter().any(|x| all[i].name.contains(x.as_str()))))
        .collect();
    {
        let mut r = res.lock().unwrap();
        for s in &all {
            if !r.datasets.iter().any(|d| d.0 == s.name) {
                r.datasets.push((s.name.into(), s.kind.label().into(), s.values.len()));
            }
        }
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build().unwrap();
    pool.install(|| {
        selected.par_iter().for_each(|&i| {
            let s = &all[i];
            let n = s.values.len();
            let bytes_in = n * 8;
            for meth in methods(&priors[i]) {
                if meth_filter.as_ref().is_some_and(|f| !f.iter().any(|x| meth.name == *x)) {
                    continue;
                }
                let ((bytes, wins), enc_mbs) = time_it(min_time, bytes_in, || (meth.enc)(&s.values));
                let (back, dec_mbs) = time_it(min_time, bytes_in, || (meth.dec)(&bytes, n));
                let ok = back.len() == n && back.iter().zip(&s.values).all(|(a, b)| a.to_bits() == b.to_bits());
                let bpv = bytes.len() as f64 * 8.0 / n as f64;
                eprintln!(
                    "[{:6.0}s] {:22} {:24} {:8.3} b/v  enc {:8.1}  dec {:8.1} MB/s  {}",
                    t0.elapsed().as_secs_f64(),
                    s.name,
                    meth.name,
                    bpv,
                    enc_mbs,
                    dec_mbs,
                    if ok { "ok" } else { "MISMATCH" }
                );
                let mut r = res.lock().unwrap();
                let entry = r.methods.entry(meth.name.clone()).or_insert((meth.description.clone(), meth.kind.into(), true));
                entry.0.clone_from(&meth.description);
                entry.2 &= ok;
                r.bits.entry(meth.name.clone()).or_default().insert(s.name.into(), (bpv * 1e4).round() / 1e4);
                r.enc.entry(meth.name.clone()).or_default().insert(s.name.into(), (enc_mbs * 100.0).round() / 100.0);
                r.dec.entry(meth.name.clone()).or_default().insert(s.name.into(), (dec_mbs * 100.0).round() / 100.0);
                if let Some(w) = wins {
                    r.wins.entry(meth.name.clone()).or_default().insert(s.name.into(), w);
                }
            }
            let r = res.lock().unwrap();
            save(&r, &out_path, &wins_path);
        });
    });
    eprintln!("done in {:.0}s", t0.elapsed().as_secs_f64());
}
