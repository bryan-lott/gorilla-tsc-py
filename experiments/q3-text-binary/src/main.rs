//! q3: can the library's ideas extend to text and binary data?
//!
//!   q3 bench [method-filter] [file-filter]   run jobs, append data/results.jsonl
//!   q3 report                                 data/results.jsonl -> ../results/q3.json
//!   q3 enc <method> <in> <out> / q3 dec <method> <in> <out>   (our methods)

mod cm;
mod columnar;
mod libcoder;
mod logtpl;
mod lpc;
mod numtok;
mod structure;
mod util;

use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;
use util::Backend;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

struct FileSpec {
    name: String,
    path: PathBuf,
    category: &'static str,
    rate: u32, // audio sample rate
    elf: bool,
}

fn files() -> Vec<FileSpec> {
    let d = Path::new(ROOT).join("data");
    let bench = Path::new(ROOT).join("../../bench-data");
    let mut v = Vec::new();
    let mut add = |name: &str, p: PathBuf, cat: &'static str| {
        if p.exists() {
            v.push(FileSpec { name: name.into(), path: p, category: cat, rate: 0, elf: false });
        } else {
            eprintln!("missing {}", p.display());
        }
    };
    for f in ["alice29.txt", "asyoulik.txt", "plrabn12.txt", "lcet10.txt", "fields.c", "cp.html", "grammar.lsp", "xargs.1"] {
        add(f, d.join("cantrbry").join(f), "text");
    }
    for f in ["kennedy.xls", "sum", "ptt5"] {
        add(f, d.join("cantrbry").join(f), "binary");
    }
    add("enwik8 (first 10 MB)", d.join("enwik8_10M"), "text");
    add("city_temperature.csv", bench.join("city_temperature.csv"), "numeric-text");
    add("Stocks-Germany-sample.txt", bench.join("Stocks-Germany-sample.txt"), "numeric-text");
    add("SSD_HDD_benchmarks.csv", bench.join("SSD_HDD_benchmarks.csv"), "numeric-text");
    let mut elf: Vec<_> = fs::read_dir(bench.join("elf")).map(|r| r.filter_map(Result::ok).map(|e| e.path()).collect()).unwrap_or_default();
    elf.sort();
    for p in elf {
        let n = format!("elf/{}", p.file_name().unwrap().to_string_lossy());
        add(&n, p, "numeric-text");
    }
    add("system journal (125k lines)", d.join("journal.log"), "log");
    add("package-manager log", d.join("pacman.log"), "log");
    add("ELF x86-64 executable", d.join("elf_clang-tidy"), "binary");
    add("SQLite DB (built from bench data)", d.join("bench.sqlite"), "binary");
    add("city_temperature as f64 LE array", d.join("city_temp.f64"), "binary");
    add("city_temperature as f32 LE array", d.join("city_temp.f32"), "binary");
    add("stock ticks, 20-byte records (i64 ts, f64 price, i32 vol)", d.join("stocks_records.bin"), "binary");
    add("music, 60 s 16-bit stereo PCM (decoded from MP3)", d.join("music_60s.s16le"), "audio");
    add("system sound, 16-bit stereo PCM (decoded from Ogg)", d.join("alarm.s16le"), "audio");
    for f in &mut v {
        if f.category == "audio" {
            f.rate = if f.name.starts_with("music") { 44100 } else { 48000 };
        }
        f.elf = f.name.starts_with("ELF");
    }
    v
}

// --------------------------------------------------------------- methods

struct MethodSpec {
    name: &'static str,
    desc: &'static str,
    ours: bool,
}

const METHODS: &[MethodSpec] = &[
    MethodSpec { name: "gzip -9", desc: "gzip CLI, level 9", ours: false },
    MethodSpec { name: "bzip2 -9", desc: "bzip2 CLI, level 9", ours: false },
    MethodSpec { name: "zstd -19", desc: "zstd CLI, level 19, single thread", ours: false },
    MethodSpec { name: "zstd -22 --long", desc: "zstd CLI, --ultra -22 --long=27", ours: false },
    MethodSpec { name: "xz -9e", desc: "xz CLI, -9e, single thread", ours: false },
    MethodSpec { name: "brotli -q11", desc: "brotli CLI, -q 11 -w 24", ours: false },
    MethodSpec { name: "xz -9e --x86", desc: "xz CLI with x86 BCJ filter + LZMA2 preset 9e (executables only)", ours: false },
    MethodSpec { name: "flac -8", desc: "flac CLI --best on raw 16-bit PCM (audio only)", ours: false },
    MethodSpec { name: "ours-libcoder-o1", desc: "order-1 byte context, binary tree of gorilla-tsc Prob, gorilla-tsc RangeEncoder (no mixing)", ours: true },
    MethodSpec { name: "ours-libcoder-o2", desc: "order-2 byte context, binary tree of gorilla-tsc Prob, gorilla-tsc RangeEncoder (no mixing)", ours: true },
    MethodSpec { name: "ours-cm", desc: "own bitwise context-mixing coder: orders 0-4,6, word, sparse, match model, 2 mixers, 2 APMs, 12-bit arithmetic coder", ours: true },
    MethodSpec { name: "ours-cm-stride", desc: "ours-cm plus record-stride column contexts (stride from autocorrelation-style detection)", ours: true },
    MethodSpec { name: "ours-cm-bcj", desc: "x86 BCJ transform then ours-cm (executables only)", ours: true },
    MethodSpec { name: "ours-struct", desc: "stride detection + per-field type choice (u8/i16/i32/f32/i64/f64/bytes) by DP; fields via gorilla-tsc Auto; falls back to ours-cm when no stride", ours: true },
    MethodSpec { name: "ours-lpc", desc: "audio: per-4096-frame stereo mode + LPC (order <= 32, exact integer) in front of gorilla-tsc RiceBlock (256-sample partitions) or AdaptiveBlock (audio only)", ours: true },
    MethodSpec { name: "ours-columnar+xz", desc: "CSV columns -> skeleton dictionaries + number tokens; numbers via gorilla-tsc Auto on scaled ints (or xz if smaller); rest via xz -9e", ours: true },
    MethodSpec { name: "ours-columnar+cm", desc: "as ours-columnar+xz with ours-cm as the back end", ours: true },
    MethodSpec { name: "ours-log+xz", desc: "log templates (CLP-like): timestamps -> epoch via gorilla-tsc, numbers per template slot via gorilla-tsc, ids/vars via xz -9e", ours: true },
    MethodSpec { name: "ours-log+cm", desc: "as ours-log+xz with ours-cm as the back end", ours: true },
    MethodSpec { name: "ours-auto", desc: "per file, smallest of the ours-* methods (+1 byte selector); enc time = sum of all tried", ours: true },
];

fn applicable(m: &str, f: &FileSpec) -> bool {
    match m {
        "xz -9e --x86" | "ours-cm-bcj" => f.elf,
        "flac -8" | "ours-lpc" => f.category == "audio",
        "ours-cm-stride" | "ours-struct" => matches!(f.category, "binary" | "audio"),
        "ours-columnar+xz" | "ours-columnar+cm" => f.category == "numeric-text",
        "ours-log+xz" | "ours-log+cm" => f.category == "log",
        "ours-auto" => false,
        _ => true,
    }
}

fn ours_compress(m: &str, data: &[u8]) -> Option<(Vec<u8>, String)> {
    Some(match m {
        "ours-libcoder-o1" => (libcoder::compress(data, 1), String::new()),
        "ours-libcoder-o2" => (libcoder::compress(data, 2), String::new()),
        "ours-cm" => (cm::compress(data, cm::CmConfig { stride: 0 }), String::new()),
        "ours-cm-stride" => {
            let (s, _) = structure::detect_stride(data);
            (cm::compress(data, cm::CmConfig { stride: s as u32 }), format!("stride {s}"))
        }
        "ours-cm-bcj" => (cm::compress(&structure::bcj_x86(data, true), cm::CmConfig { stride: 0 }), String::new()),
        "ours-struct" => structure::struct_encode(data),
        "ours-lpc" => (lpc::encode(data), String::new()),
        "ours-columnar+xz" => (columnar::encode(data, Backend::Xz)?, String::new()),
        "ours-columnar+cm" => (columnar::encode(data, Backend::Cm)?, String::new()),
        "ours-log+xz" => (logtpl::encode(data, Backend::Xz)?, String::new()),
        "ours-log+cm" => (logtpl::encode(data, Backend::Cm)?, String::new()),
        _ => return None,
    })
}

fn ours_decompress(m: &str, c: &[u8]) -> Vec<u8> {
    match m {
        "ours-libcoder-o1" | "ours-libcoder-o2" => libcoder::decompress(c),
        "ours-cm" | "ours-cm-stride" => cm::decompress(c),
        "ours-cm-bcj" => structure::bcj_x86(&cm::decompress(c), false),
        "ours-struct" => structure::struct_decode(c),
        "ours-lpc" => lpc::decode(c),
        "ours-columnar+xz" => columnar::decode(c, Backend::Xz),
        "ours-columnar+cm" => columnar::decode(c, Backend::Cm),
        "ours-log+xz" => logtpl::decode(c, Backend::Xz),
        "ours-log+cm" => logtpl::decode(c, Backend::Cm),
        _ => unreachable!(),
    }
}

fn cli_args(m: &str, f: &FileSpec) -> (Vec<String>, Vec<String>) {
    let s = |v: &[&str]| v.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
    match m {
        "gzip -9" => (s(&["gzip", "-9", "-c", "-n"]), s(&["gzip", "-d", "-c"])),
        "bzip2 -9" => (s(&["bzip2", "-9", "-c"]), s(&["bzip2", "-d", "-c"])),
        "zstd -19" => (s(&["zstd", "-19", "-T1", "-c", "-q"]), s(&["zstd", "-d", "-c", "-q"])),
        "zstd -22 --long" => (s(&["zstd", "--ultra", "-22", "--long=27", "-T1", "-c", "-q"]), s(&["zstd", "-d", "--long=27", "-c", "-q"])),
        "xz -9e" => (s(&["xz", "-9e", "-T1", "-c"]), s(&["xz", "-d", "-c"])),
        "xz -9e --x86" => (s(&["xz", "--x86", "--lzma2=preset=9e", "-T1", "-c"]), s(&["xz", "-d", "-c"])),
        "brotli -q11" => (s(&["brotli", "-q", "11", "-w", "24", "-c"]), s(&["brotli", "-d", "-c"])),
        "flac -8" => {
            let rate = format!("--sample-rate={}", f.rate);
            (
                vec!["flac".into(), "--silent".into(), "--best".into(), "--force-raw-format".into(), "--endian=little".into(), "--sign=signed".into(), "--channels=2".into(), "--bps=16".into(), rate, "-c".into(), "-".into()],
                s(&["flac", "--silent", "-d", "--force-raw-format", "--endian=little", "--sign=signed", "-c", "-"]),
            )
        }
        _ => unreachable!(),
    }
}

fn run_cli(args: &[String], input: &Path, output: &Path) -> f64 {
    let t = Instant::now();
    let st = Command::new(&args[0])
        .args(&args[1..])
        .stdin(fs::File::open(input).unwrap())
        .stdout(fs::File::create(output).unwrap())
        .stderr(Stdio::inherit())
        .status()
        .expect("spawn");
    assert!(st.success(), "{args:?} failed");
    t.elapsed().as_secs_f64()
}

fn run_job(m: &str, f: &FileSpec, tmp: &Path) -> Value {
    let data = fs::read(&f.path).unwrap();
    let mb = data.len() as f64 / 1e6;
    let (size, enc, dec, ok, note);
    if METHODS.iter().find(|x| x.name == m).unwrap().ours {
        let t = Instant::now();
        let Some((c, n)) = ours_compress(m, &data) else {
            return json!({"method": m, "file": f.name, "skipped": true});
        };
        enc = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let d = ours_decompress(m, &c);
        dec = t.elapsed().as_secs_f64();
        ok = d == data;
        size = c.len();
        note = n;
    } else {
        let tag: String = format!("{}-{}", m, f.name).chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let cpath = tmp.join(format!("{tag}.c"));
        let dpath = tmp.join(format!("{tag}.d"));
        let (ea, da) = cli_args(m, f);
        enc = run_cli(&ea, &f.path, &cpath);
        dec = run_cli(&da, &cpath, &dpath);
        size = fs::metadata(&cpath).unwrap().len() as usize;
        ok = fs::read(&dpath).unwrap() == data;
        let _ = fs::remove_file(&cpath);
        let _ = fs::remove_file(&dpath);
        note = String::new();
    }
    eprintln!(
        "{:<20} {:<45} {:>11} -> {:>10} ({:.3} bpb) enc {:.2} MB/s dec {:.2} MB/s {} {}",
        m,
        f.name,
        data.len(),
        size,
        size as f64 * 8.0 / data.len() as f64,
        mb / enc,
        mb / dec,
        if ok { "OK" } else { "MISMATCH" },
        note
    );
    json!({"method": m, "file": f.name, "category": f.category, "bytes": data.len(), "compressed": size,
           "enc_s": enc, "dec_s": dec, "verified": ok, "note": note})
}

fn bench(mfilter: Option<&str>, ffilter: Option<&str>, threads: usize) {
    let tmp = Path::new(ROOT).join("data/tmp");
    fs::create_dir_all(&tmp).unwrap();
    let out_path = Path::new(ROOT).join("data/results.jsonl");
    let done: HashSet<(String, String)> = fs::read_to_string(&out_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .map(|v| (v["method"].as_str().unwrap().to_string(), v["file"].as_str().unwrap().to_string()))
        .collect();
    let fs_ = files();
    let mut jobs: Vec<(&str, &FileSpec)> = Vec::new();
    for f in &fs_ {
        if ffilter.is_some_and(|x| !f.name.contains(x)) {
            continue;
        }
        for m in METHODS {
            if mfilter.is_some_and(|x| !x.split(',').any(|p| m.name == p)) {
                continue;
            }
            if applicable(m.name, f) && !done.contains(&(m.name.to_string(), f.name.clone())) {
                jobs.push((m.name, f));
            }
        }
    }
    // longest first (rough cost model)
    jobs.sort_by_key(|(m, f)| {
        let sz = fs::metadata(&f.path).map(|x| x.len()).unwrap_or(0);
        let w = if m.contains("cm") || m.contains("brotli") || m.contains("22") { 20 } else { 1 };
        std::cmp::Reverse(sz * w)
    });
    eprintln!("{} jobs on {threads} threads", jobs.len());
    let out = Mutex::new(fs::OpenOptions::new().create(true).append(true).open(&out_path).unwrap());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        jobs.par_iter().with_max_len(1).for_each(|(m, f)| {
            let v = run_job(m, f, &tmp);
            let mut o = out.lock().unwrap();
            writeln!(o, "{v}").unwrap();
        });
    });
}

fn report() {
    let rows: Vec<Value> = fs::read_to_string(Path::new(ROOT).join("data/results.jsonl"))
        .unwrap()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .filter(|v: &Value| v.get("skipped").is_none())
        .collect();
    // latest row wins per (method, file)
    let mut by: BTreeMap<(String, String), Value> = BTreeMap::new();
    for r in rows {
        by.insert((r["method"].as_str().unwrap().into(), r["file"].as_str().unwrap().into()), r);
    }
    let fs_ = files();
    let mut comp: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    let mut encm: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    let mut decm: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    let mut verified: BTreeMap<String, bool> = BTreeMap::new();
    let mut notes: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
    for ((m, f), r) in &by {
        let mb = r["bytes"].as_f64().unwrap() / 1e6;
        comp.entry(m.clone()).or_default().insert(f.clone(), r["compressed"].clone());
        encm.entry(m.clone()).or_default().insert(f.clone(), json!((mb / r["enc_s"].as_f64().unwrap() * 1000.0).round() / 1000.0));
        decm.entry(m.clone()).or_default().insert(f.clone(), json!((mb / r["dec_s"].as_f64().unwrap() * 1000.0).round() / 1000.0));
        let ok = r["verified"].as_bool().unwrap();
        let e = verified.entry(m.clone()).or_insert(true);
        *e &= ok;
        if let Some(n) = r["note"].as_str().filter(|s| !s.is_empty()) {
            notes.entry(m.clone()).or_default().insert(f.clone(), json!(n));
        }
    }
    // ours-auto
    let ours: Vec<&str> = METHODS.iter().filter(|m| m.ours && m.name != "ours-auto").map(|m| m.name).collect();
    let mut auto_ok = true;
    for f in &fs_ {
        let cands: Vec<&Value> = ours.iter().filter_map(|m| by.get(&((*m).to_string(), f.name.clone()))).collect();
        if let Some(best) = cands.iter().filter(|r| r["verified"].as_bool().unwrap()).min_by_key(|r| r["compressed"].as_u64().unwrap()) {
            let mb = best["bytes"].as_f64().unwrap() / 1e6;
            let enc_s: f64 = cands.iter().map(|r| r["enc_s"].as_f64().unwrap()).sum();
            comp.entry("ours-auto".into()).or_default().insert(f.name.clone(), json!(best["compressed"].as_u64().unwrap() + 1));
            encm.entry("ours-auto".into()).or_default().insert(f.name.clone(), json!((mb / enc_s * 1000.0).round() / 1000.0));
            decm.entry("ours-auto".into()).or_default().insert(f.name.clone(), json!((mb / best["dec_s"].as_f64().unwrap() * 1000.0).round() / 1000.0));
            notes.entry("ours-auto".into()).or_default().insert(f.name.clone(), best["method"].clone());
        } else {
            auto_ok = false;
        }
    }
    verified.insert("ours-auto".into(), auto_ok);
    let methods: Vec<Value> = METHODS
        .iter()
        .filter(|m| comp.contains_key(m.name))
        .map(|m| json!({"name": m.name, "description": m.desc, "kind": if m.ours {"ours"} else {"baseline"},
                         "lossless_verified": verified.get(m.name).copied().unwrap_or(false)}))
        .collect();
    let filesj: Vec<Value> = fs_
        .iter()
        .map(|f| json!({"name": f.name, "category": f.category, "bytes": fs::metadata(&f.path).unwrap().len()}))
        .collect();
    let out = json!({"methods": methods, "files": filesj, "compressed_bytes": comp, "enc_mbs": encm, "dec_mbs": decm,
                     "notes": notes,
                     "timing_note": "single-threaded per job; jobs run 6 at a time on an i7-8700K; CLI tools timed as wall time of the process (includes spawn + page-cache I/O)"});
    let p = Path::new(ROOT).join("../results/q3.json");
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    eprintln!("wrote {}", p.display());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("bench") => {
            let mf = a.get(2).map(String::as_str).filter(|s| *s != "all");
            let ff = a.get(3).map(String::as_str).filter(|s| *s != "all");
            let th = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
            bench(mf, ff, th);
        }
        Some("report") => report(),
        Some("enc") => {
            let d = fs::read(&a[3]).unwrap();
            let (c, n) = ours_compress(&a[2], &d).expect("not applicable");
            fs::write(&a[4], &c).unwrap();
            eprintln!("{} -> {} {}", d.len(), c.len(), n);
        }
        Some("dec") => {
            let c = fs::read(&a[3]).unwrap();
            fs::write(&a[4], ours_decompress(&a[2], &c)).unwrap();
        }
        Some("stride") => {
            let d = fs::read(&a[2]).unwrap();
            let (s, costs) = structure::detect_stride(&d);
            eprintln!("stride {s}; cost/cost(1) for 1..16: {:?}", (1..=16).map(|i| (costs[i] / costs[1] * 100.0).round() / 100.0).collect::<Vec<_>>());
        }
        _ => eprintln!("usage: q3 bench [methods|all] [file-substring|all] [threads] | report | enc/dec <method> <in> <out> | stride <file>"),
    }
}
