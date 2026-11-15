//! Which concrete mode wins each block under auto (development aid).
use gorilla_tsc::{CheckpointInterval, Config, Encoder, Mode, Sample};
fn column(p: &std::path::Path, c: usize) -> Vec<f64> { std::fs::read_to_string(p).unwrap().lines().filter_map(|l| l.split(',').nth(c)?.trim().parse().ok()).collect() }
const MODES: [Mode; 6] = [Mode::Xor, Mode::Dod, Mode::XorDod, Mode::DodXor, Mode::Delta, Mode::Decimal];
fn wins<T: Sample>(v: &[T]) -> [usize; 6] {
    let one = Config::new(Mode::Xor).with_interval(CheckpointInterval::new(1000).unwrap());
    let mut w = [0; 6];
    for b in v.chunks_exact(1000) {
        let bits = MODES.map(|m| { let mut e = Encoder::new(Config { mode: m, ..one }); for &x in b { e.push(x); } e.payload_bits() });
        let best = (0..6).min_by_key(|&i| bits[i]).unwrap();
        w[best] += 1;
    }
    w
}
fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench-data");
    for (n, f) in [("city", "city_temperature.csv"), ("stocks", "Stocks-Germany-sample.txt"), ("ssd", "SSD_HDD_benchmarks.csv")] {
        let mut v = column(&dir.join(f), 2); v.truncate(300_000);
        println!("{n:7} wins (xor dod xor_dod dod_xor delta decimal): {:?}", wins(&v));
    }
}
