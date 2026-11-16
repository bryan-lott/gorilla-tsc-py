//! Compression ratio and speed on the Gorilla/Chimp benchmark datasets.
//!
//! `scripts/fetch-bench-data.sh`, then
//! `cargo run --release -p gorilla-tsc --example bench [-- --quick]`.
//!
//! Like the Chimp paper, values are split into independent blocks of 1000
//! (`checkpoint_interval = 1000`) and bits/value counts the payload only.

use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use gorilla_tsc::{CheckpointInterval, Codec, Config, Encoder, Mode, Profile, Sample, decode, encode};

const BLOCK: u32 = 1000;
const MODES: [Mode; 7] = [
    Mode::Fixed(Codec::Xor),
    Mode::Fixed(Codec::Dod),
    Mode::Fixed(Codec::XorDod),
    Mode::Fixed(Codec::DodXor),
    Mode::Fixed(Codec::Delta),
    Mode::Fixed(Codec::Decimal),
    Mode::Auto,
];

fn column(path: &Path, col: usize) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e} (run scripts/fetch-bench-data.sh)", path.display()))
        .lines()
        .filter_map(|l| l.split(',').nth(col)?.trim().parse().ok())
        .collect()
}

/// Days since 1970-01-01 for a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn stock_timestamps(path: &Path) -> Vec<i64> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|l| {
            let mut f = l.split(',');
            let date: Vec<i64> = f.next()?.split('/').map(|x| x.parse().ok()).collect::<Option<_>>()?;
            let time: Vec<i64> = f.next()?.split(':').map(|x| x.parse().ok()).collect::<Option<_>>()?;
            let (mm, dd, yy) = (date[0], date[1], date[2]);
            Some(days_from_civil(yy, mm, dd) * 86_400 + time[0] * 3600 + time[1] * 60 + time[2])
        })
        .collect()
}

/// Regular 10 s scrape interval in ms with small deterministic jitter.
fn scrape_timestamps(n: usize) -> Vec<i64> {
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let jitter = i64::try_from(x % 11).unwrap() - 5;
            1_700_000_000_000 + i64::try_from(i).unwrap() * 10_000 + jitter
        })
        .collect()
}

struct Row {
    bits_per_value: f64,
    enc_mbs: f64,
    dec_mbs: f64,
}

fn time<F: FnMut()>(mut f: F, min: Duration) -> Duration {
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 3 || start.elapsed() < min {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        runs += 1;
    }
    best
}

#[allow(clippy::cast_precision_loss)]
fn measure<T: Sample>(data: &[T], mode: Mode, min: Duration) -> Row {
    let cfg = Config::new(mode).with_interval(CheckpointInterval::new(BLOCK).unwrap());
    let full = data.len() / BLOCK as usize * BLOCK as usize; // whole blocks, as in the paper
    let data = &data[..full];
    let mut e = Encoder::new(cfg);
    for &x in data {
        e.push(x);
    }
    let bits = e.payload_bits();
    let bytes = encode(data, cfg);
    assert_eq!(
        decode::<T>(&bytes).unwrap().iter().map(|x| x.to_word()).collect::<Vec<_>>(),
        data.iter().map(|x| x.to_word()).collect::<Vec<_>>(),
        "round trip failed"
    );
    let raw = (data.len() * T::TYPE.byte_width()) as f64 / 1e6;
    let enc = time(|| drop(black_box(encode(black_box(data), cfg))), min);
    let dec = time(|| drop(black_box(decode::<T>(black_box(&bytes)).unwrap())), min);
    Row {
        bits_per_value: bits as f64 / data.len() as f64,
        enc_mbs: raw / enc.as_secs_f64(),
        dec_mbs: raw / dec.as_secs_f64(),
    }
}

fn report<T: Sample>(name: &str, data: &[T], reference: &str, min: Duration) {
    println!("\n{name} ({} values){reference}", data.len());
    println!("  {:<8} {:>10} {:>11} {:>11}", "mode", "bits/value", "enc MB/s", "dec MB/s");
    for m in MODES {
        let r = measure(data, m, min);
        println!("  {:<8} {:>10.2} {:>11.0} {:>11.0}", m.name(), r.bits_per_value, r.enc_mbs, r.dec_mbs);
    }
}

/// One line per Elf dataset (`bench-data/elf`, see `scripts/fetch-bench-data.sh`).
#[allow(clippy::cast_precision_loss)]
fn elf(dir: &Path, min: Duration) {
    const SETS: [&str; 22] = [
        "Air-pressure", "Air-sensor", "Basel-temp", "Basel-wind", "Bird-migration", "Bitcoin-price",
        "City-temp", "Dew-point-temp", "IR-bio-temp", "PM10-dust", "Stocks-DE", "Stocks-UK", "Stocks-USA",
        "Wind-Speed", "Blockchain-tr", "City-lat", "City-lon", "Food-price", "POI-lat", "POI-lon", "SSD-bench",
        "electric_vehicle_charging",
    ];
    println!(
        "{:<26} {:>8} {:>8} {:>9} {:>9}   {:>8} {:>9}  learned from first 10k",
        "Elf dataset", "gorilla", "auto", "enc MB/s", "dec MB/s", "profile", "enc MB/s"
    );
    let (mut sum_g, mut sum_a, mut sum_p) = (0.0, 0.0, 0.0);
    for name in SETS {
        let v: Vec<f64> = std::fs::read_to_string(dir.join("elf").join(format!("{name}.csv")))
            .unwrap_or_else(|e| panic!("{name}: {e} (run scripts/fetch-bench-data.sh)"))
            .lines()
            .filter_map(|l| l.trim().parse().ok())
            .collect();
        let g = measure(&v, Mode::Fixed(Codec::Xor), Duration::ZERO);
        let a = measure(&v, Mode::Auto, min);
        let profile = Profile::learn(&v[..v.len().min(10_000)], CheckpointInterval::new(BLOCK).unwrap());
        let p = measure(&v, Mode::Profile(profile), min);
        sum_g += g.bits_per_value;
        sum_a += a.bits_per_value;
        sum_p += p.bits_per_value;
        println!(
            "{name:<26} {:>8.2} {:>8.2} {:>9.1} {:>9.0}   {:>8.2} {:>9.1}  {profile}",
            g.bits_per_value, a.bits_per_value, a.enc_mbs, a.dec_mbs, p.bits_per_value, p.enc_mbs
        );
    }
    let n = SETS.len() as f64;
    println!("{:<26} {:>8.2} {:>8.2} {:>9} {:>9}   {:>8.2}", "mean", sum_g / n, sum_a / n, "", "", sum_p / n);
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let min = Duration::from_millis(if quick { 0 } else { 300 });
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench-data");
    if std::env::args().any(|a| a == "--elf") {
        elf(&dir, min);
        return;
    }
    let limit = if quick { 200_000 } else { usize::MAX };
    let take = |mut v: Vec<f64>| {
        v.truncate(limit);
        v
    };

    report("city_temperature", &take(column(&dir.join("city_temperature.csv"), 2)), "  [paper: Gorilla 58.72, Chimp128 22.92]", min);
    report("Stocks-Germany", &take(column(&dir.join("Stocks-Germany-sample.txt"), 2)), "  [paper: Gorilla 46.90, Chimp128 13.20]", min);
    report("SSD_HDD", &column(&dir.join("SSD_HDD_benchmarks.csv"), 2), "  [paper: Gorilla 40.25, Chimp128 17.00]", min);

    let mut ts = stock_timestamps(&dir.join("Stocks-Germany-sample.txt"));
    ts.truncate(limit);
    report("Stocks-Germany timestamps (s)", &ts, "", min);
    report("scrape timestamps (10 s +/- 5 ms)", &scrape_timestamps(if quick { 200_000 } else { 1_000_000 }), "", min);
}
