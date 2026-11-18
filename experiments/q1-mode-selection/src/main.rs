//! Question 1: does one codec do well everywhere, can we sample instead of
//! trying every mode on every block, and can a "production" user lock in
//! the choice up front?
//!
//! `cargo run --release` (ratios, parallel)
//! `cargo run --release -- --timing` (speed, sequential, run on a quiet machine)
//!
//! Writes ../results/q1.json (and q1-timing.json).

mod profile;

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

use bench_common::{FloatSet, IntSet, Kind, float_sets, int_sets};
use gorilla_tsc::{CheckpointInterval, Config, Encoder, Mode, Order, Sample, decode, encode};
use profile::{Coder, KPolicy, Profile};
use serde_json::{Value, json};

const BLOCK: usize = 1000;
const CONCRETE: [Mode; 6] = [Mode::Xor, Mode::Dod, Mode::XorDod, Mode::DodXor, Mode::Delta, Mode::Decimal];
const AUTO_SELECTOR_BITS: usize = 3;

fn name(m: Mode) -> &'static str {
    match m {
        Mode::Xor => "xor",
        Mode::Dod => "dod",
        Mode::XorDod => "xor_dod",
        Mode::DodXor => "dod_xor",
        Mode::Delta => "delta",
        Mode::Decimal => "decimal",
        Mode::Auto => "auto",
    }
}

fn cfg(m: Mode) -> Config {
    Config::new(m).with_interval(CheckpointInterval::new(u32::try_from(BLOCK).unwrap()).unwrap())
}

fn bits_of<T: Sample>(items: &[T], m: Mode) -> usize {
    let mut e = Encoder::new(cfg(m));
    for &x in items {
        e.push(x);
    }
    e.payload_bits()
}

fn idx(m: Mode) -> usize {
    CONCRETE.iter().position(|&c| c == m).unwrap()
}

/// Per block, the payload bits of every concrete mode.
fn matrix<T: Sample>(v: &[T]) -> Vec<[usize; 6]> {
    v.chunks(BLOCK).map(|b| CONCRETE.map(|m| bits_of(b, m))).collect()
}

fn argmin_auto<T: Sample>(costs: impl Fn(Mode) -> usize) -> Mode {
    *T::AUTO.iter().min_by_key(|&&m| costs(m)).unwrap()
}

#[allow(clippy::cast_precision_loss)]
fn bpv(bits: usize, n: usize) -> f64 {
    bits as f64 / n as f64
}

/// Ratio analysis of one dataset (everything except speed).
fn analyse<T: Sample>(name_: &str, kind: &str, v: &[T]) -> Value {
    let n = v.len();
    let mx = matrix(v);
    let fixed = |m: Mode| mx.iter().map(|r| r[idx(m)]).sum::<usize>();
    let auto_block = |r: &[usize; 6]| T::AUTO.iter().map(|&m| r[idx(m)]).min().unwrap() + AUTO_SELECTOR_BITS;
    let auto: usize = mx.iter().map(auto_block).sum();
    // The harness must agree with the library's own auto.
    assert_eq!(auto, bits_of(v, Mode::Auto), "{name_}: matrix disagrees with library auto");

    let mut winners = serde_json::Map::new();
    for &m in T::AUTO {
        let w = mx.iter().filter(|r| argmin_auto::<T>(|c| r[idx(c)]) == m).count();
        winners.insert(name(m).into(), json!(w));
    }
    let mut fixed_bpv = serde_json::Map::new();
    for &m in T::AUTO {
        fixed_bpv.insert(name(m).into(), json!(bpv(fixed(m), n)));
    }
    let best_fixed = argmin_auto::<T>(fixed);

    // Lock a mode from a sample of the first S values, encode everything with it.
    let mut lock = serde_json::Map::new();
    for s in [100usize, 1000, 10_000] {
        let sample = &v[..s.min(n)];
        let m = argmin_auto::<T>(|c| bits_of(sample, c));
        lock.insert(format!("first {s} values"), json!({"mode": name(m), "bpv": bpv(fixed(m), n)}));
    }

    // Per block, choose by trial-encoding only the first m values of the block.
    let mut probe = serde_json::Map::new();
    for m in [16usize, 64, 128, 256] {
        let total: usize = v
            .chunks(BLOCK)
            .zip(&mx)
            .map(|(b, r)| {
                let head = &b[..m.min(b.len())];
                r[idx(argmin_auto::<T>(|c| bits_of(head, c)))] + AUTO_SELECTOR_BITS
            })
            .sum();
        probe.insert(format!("first {m} of each block"), json!(bpv(total, n)));
    }

    // Full trial on every R-th block; the next R-1 blocks reuse its mode.
    let mut reprobe = serde_json::Map::new();
    for r_ in [4usize, 16, 64] {
        let mut chosen = Mode::Xor;
        let total: usize = mx
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i % r_ == 0 {
                    chosen = argmin_auto::<T>(|c| r[idx(c)]);
                }
                r[idx(chosen)] + AUTO_SELECTOR_BITS
            })
            .sum();
        reprobe.insert(format!("every {r_} blocks"), json!(bpv(total, n)));
    }

    // Auto restricted to smaller candidate sets.
    let mut subsets = serde_json::Map::new();
    let sets: &[(&str, &[Mode])] = &[
        ("decimal+xor", &[Mode::Decimal, Mode::Xor]),
        ("decimal+xor+xor_dod", &[Mode::Decimal, Mode::Xor, Mode::XorDod]),
        ("delta+dod", &[Mode::Delta, Mode::Dod]),
    ];
    for &(label, modes) in sets {
        if modes.iter().all(|m| T::AUTO.contains(m)) {
            let total: usize = mx.iter().map(|r| modes.iter().map(|&m| r[idx(m)]).min().unwrap() + AUTO_SELECTOR_BITS).sum();
            subsets.insert(label.into(), json!(bpv(total, n)));
        }
    }

    json!({
        "name": name_, "kind": kind, "n": n,
        "auto_bpv": bpv(auto, n),
        "auto_subsets": subsets,
        "fixed_bpv": fixed_bpv,
        "best_fixed": name(best_fixed),
        "winners": winners,
        "lock_from_sample": lock,
        "probe_block_head": probe,
        "reprobe": reprobe,
    })
}

fn orders() -> [Order; 3] {
    [Order::First, Order::Second, Order::Zero]
}

/// Profiles for one dataset: learned from a sample, and fixed "universal" ones.
fn learn<T: Sample>(sample: &[T], is_float: bool) -> Profile {
    let ks: Vec<i32> = if is_float { (-T::MAX_SCALE..=T::MAX_SCALE).collect() } else { vec![0] };
    let mut best: Option<(usize, Profile)> = None;
    for &k in &ks {
        for order in orders() {
            for coder in [Coder::Rice, Coder::Adaptive] {
                let p = Profile { k: KPolicy::Fixed(k), order, coder };
                let b = profile::encode(sample, BLOCK, p).bit_len();
                if best.is_none_or(|(bb, _)| b < bb) {
                    best = Some((b, p));
                }
            }
        }
    }
    best.unwrap().1
}

fn profiles<T: Sample>(v: &[T], is_float: bool) -> Vec<(String, Profile)> {
    let learned = learn(&v[..10_000.min(v.len())], is_float);
    let mut out = vec![("learned from first 10k (locked)".to_string(), learned)];
    out.push(("learned from first 1k (locked)".to_string(), learn(&v[..1000.min(v.len())], is_float)));
    out.push(("learned from first 100k (locked)".to_string(), learn(&v[..100_000.min(v.len())], is_float)));
    if is_float {
        for (s, label) in [(1000usize, "1k"), (10_000, "10k")] {
            let p = learn(&v[..s.min(v.len())], is_float);
            let KPolicy::Fixed(k) = p.k else { unreachable!() };
            out.push((format!("learned from first {label}, exception guard"), Profile { k: KPolicy::Guarded(k), ..p }));
        }
        for permille in [0usize, 10] {
            out.push((format!("learned order/coder, covering k ({permille}/1000)"), Profile { k: KPolicy::Covering(permille), ..learned }));
        }
        out.push(("universal: covering k, 1st diff, adaptive".into(), Profile { k: KPolicy::Covering(10), order: Order::First, coder: Coder::Adaptive }));
    }
    if is_float {
        out.push(("universal: sampled k, 1st diff, adaptive".into(), Profile { k: KPolicy::Sampled(32), order: Order::First, coder: Coder::Adaptive }));
        out.push(("universal: sampled k, 1st diff, rice".into(), Profile { k: KPolicy::Sampled(32), order: Order::First, coder: Coder::Rice }));
        out.push(("learned order/coder, sampled k".into(), Profile { k: KPolicy::Sampled(32), ..learned }));
    } else {
        out.push(("universal: 1st diff, adaptive".into(), Profile { k: KPolicy::Fixed(0), order: Order::First, coder: Coder::Adaptive }));
        out.push(("universal: 2nd diff, adaptive".into(), Profile { k: KPolicy::Fixed(0), order: Order::Second, coder: Coder::Adaptive }));
    }
    out
}

fn same<T: Sample>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_word() == y.to_word())
}

fn profile_results<T: Sample>(v: &[T], is_float: bool) -> Value {
    let n = v.len();
    let mut res = Vec::new();
    for (label, p) in profiles(v, is_float) {
        let w = profile::encode(v, BLOCK, p);
        let bits = w.bit_len();
        let bytes = w.finish();
        let back: Vec<T> = profile::decode(&bytes, n, BLOCK, p);
        assert!(same(&back, v), "profile {label} is not lossless");
        res.push(json!({"label": label, "profile": p.label(), "bpv": bpv(bits, n), "lossless_verified": true}));
        if label == "learned from first 10k (locked)" {
            for tol in [0.05, 0.15, 0.30] {
                let (w, relearns) = profile::encode_guarded(v, BLOCK, p, tol);
                let bits = w.bit_len();
                let back: Vec<T> = profile::decode_guarded(&w.finish(), n, BLOCK);
                assert!(same(&back, v), "guarded profile is not lossless");
                res.push(json!({
                    "label": format!("guarded lock, re-learn if a block grows >{:.0}%", tol * 100.0),
                    "profile": p.label(), "bpv": bpv(bits, n), "relearns": relearns,
                    "blocks": n.div_ceil(BLOCK), "lossless_verified": true,
                }));
            }
            for r in [16usize, 64] {
                let w = profile::encode_windowed(v, BLOCK, p, r);
                let bits = w.bit_len();
                let back: Vec<T> = profile::decode_guarded(&w.finish(), n, BLOCK);
                assert!(same(&back, v), "windowed profile is not lossless");
                res.push(json!({
                    "label": format!("windowed lock, re-learn every {r} blocks"),
                    "profile": p.label(), "bpv": bpv(bits, n), "lossless_verified": true,
                }));
            }
        }
    }
    Value::Array(res)
}

fn run_float(s: &FloatSet) -> Value {
    let mut a = analyse(s.name, s.kind.label(), &s.values);
    a["profiles"] = profile_results(&s.values, true);
    a
}

fn run_int(s: &IntSet) -> Value {
    let mut a = analyse(s.name, "integer timestamps", &s.values);
    a["profiles"] = profile_results(&s.values, false);
    a
}

// ---------------------------------------------------------------- timing

fn best_time<F: FnMut()>(mut f: F) -> Duration {
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 3 || (start.elapsed() < Duration::from_millis(300) && runs < 50) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        runs += 1;
    }
    best
}

#[allow(clippy::cast_precision_loss)]
fn mbs(n: usize, bytes_per: usize, d: Duration) -> f64 {
    (n * bytes_per) as f64 / 1e6 / d.as_secs_f64()
}

/// The per-block probe strategy as an encoder would run it.
fn probe_encode<T: Sample>(v: &[T], m: usize) -> usize {
    v.chunks(BLOCK)
        .map(|b| {
            let head = &b[..m.min(b.len())];
            let mode = argmin_auto::<T>(|c| bits_of(head, c));
            bits_of(b, mode)
        })
        .sum()
}

fn timing<T: Sample>(name_: &str, v: &[T], is_float: bool) -> Value {
    let n = v.len();
    let w = size_of::<T>();
    let mut row = serde_json::Map::new();
    let lib_modes: Vec<Mode> = if is_float {
        vec![Mode::Auto, Mode::Decimal, Mode::Xor]
    } else {
        vec![Mode::Auto, Mode::Delta, Mode::Dod]
    };
    for m in lib_modes {
        let bytes = encode(v, cfg(m));
        let e = best_time(|| {
            black_box(encode(black_box(v), cfg(m)));
        });
        let d = best_time(|| {
            black_box(decode::<T>(black_box(&bytes)).unwrap());
        });
        row.insert(format!("library {}", name(m)), json!({"enc_mbs": mbs(n, w, e), "dec_mbs": mbs(n, w, d)}));
    }
    for m in [64usize, 128] {
        let e = best_time(|| {
            black_box(probe_encode(black_box(v), m));
        });
        row.insert(format!("probe first {m} of each block"), json!({"enc_mbs": mbs(n, w, e)}));
    }
    for (label, p) in profiles(v, is_float) {
        let bytes = profile::encode(v, BLOCK, p).finish();
        let e = best_time(|| {
            black_box(profile::encode(black_box(v), BLOCK, p));
        });
        let d = best_time(|| {
            black_box(profile::decode::<T>(black_box(&bytes), n, BLOCK, p));
        });
        row.insert(format!("profile: {label}"), json!({"enc_mbs": mbs(n, w, e), "dec_mbs": mbs(n, w, d)}));
        if label == "learned from first 10k (locked)" {
            for tol in [0.05, 0.15, 0.30] {
                let bytes = profile::encode_guarded(v, BLOCK, p, tol).0.finish();
                let e = best_time(|| {
                    black_box(profile::encode_guarded(black_box(v), BLOCK, p, tol));
                });
                let d = best_time(|| {
                    black_box(profile::decode_guarded::<T>(black_box(&bytes), n, BLOCK));
                });
                row.insert(
                    format!("guarded lock, re-learn if a block grows >{:.0}%", tol * 100.0),
                    json!({"enc_mbs": mbs(n, w, e), "dec_mbs": mbs(n, w, d)}),
                );
            }
            for r in [16usize, 64] {
                let bytes = profile::encode_windowed(v, BLOCK, p, r).finish();
                let e = best_time(|| {
                    black_box(profile::encode_windowed(black_box(v), BLOCK, p, r));
                });
                let d = best_time(|| {
                    black_box(profile::decode_guarded::<T>(black_box(&bytes), n, BLOCK));
                });
                row.insert(format!("windowed lock, re-learn every {r} blocks"), json!({"enc_mbs": mbs(n, w, e), "dec_mbs": mbs(n, w, d)}));
            }
        }
    }
    json!({"name": name_, "timing": row})
}

fn results_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../results")
}

fn main() {
    let floats = float_sets();
    let ints = int_sets();
    let dir = results_dir();
    std::fs::create_dir_all(&dir).unwrap();

    if std::env::args().any(|a| a == "--timing") {
        let mut out = Vec::new();
        for s in &floats {
            eprintln!("timing {}", s.name);
            out.push(timing(s.name, &s.values, true));
        }
        for s in &ints {
            eprintln!("timing {}", s.name);
            out.push(timing(s.name, &s.values, false));
        }
        std::fs::write(dir.join("q1-timing.json"), serde_json::to_string_pretty(&out).unwrap()).unwrap();
        return;
    }

    let mut results: Vec<Value> = std::thread::scope(|sc| {
        let hf: Vec<_> = floats.iter().map(|s| sc.spawn(move || run_float(s))).collect();
        let hi: Vec<_> = ints.iter().map(|s| sc.spawn(move || run_int(s))).collect();
        hf.into_iter().chain(hi).map(|h| h.join().unwrap()).collect()
    });
    results.sort_by_key(|r| {
        let k = r["kind"].as_str().unwrap().to_string();
        (k != Kind::TimeSeries.label(), k)
    });
    let mut report = String::new();
    for r in &results {
        let _ = writeln!(
            report,
            "{:36} auto {:6.2}  best-fixed {:8} {:6.2}  winners {}",
            r["name"].as_str().unwrap(),
            r["auto_bpv"].as_f64().unwrap(),
            r["best_fixed"].as_str().unwrap(),
            r["fixed_bpv"][r["best_fixed"].as_str().unwrap()].as_f64().unwrap(),
            r["winners"]
        );
    }
    print!("{report}");
    std::fs::write(dir.join("q1.json"), serde_json::to_string_pretty(&results).unwrap()).unwrap();
}
