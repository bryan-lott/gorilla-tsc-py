//! Time the pieces of decimal/auto encoding on real blocks (development aid).
use std::time::Instant;
use gorilla_tsc::{AdaptiveBlock, BitWriter, BlockCodec, DecimalRice, DeltaRice, RiceBlock, Sample};

fn column(p: &std::path::Path, c: usize) -> Vec<f64> {
    std::fs::read_to_string(p).unwrap().lines().filter_map(|l| l.split(',').nth(c)?.trim().parse().ok()).collect()
}

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench-data");
    let v = column(&dir.join("city_temperature.csv"), 2);
    let blocks: Vec<Vec<u64>> = v.chunks_exact(1000).take(100).map(|b| b.iter().map(|x| x.to_bits()).collect()).collect();
    let n = (blocks.len() * 1000) as f64;
    let ints: Vec<Vec<u64>> = blocks.iter().map(|b| b.iter().map(|&w| f64::from_bits(w).to_scaled(1).unwrap_or(0).cast_unsigned()).collect()).collect();
    let deltas: Vec<Vec<u64>> = ints.iter().map(|b| b.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect()).collect();
    let t = |name: &str, f: &mut dyn FnMut()| {
        let s = Instant::now();
        f();
        println!("{name:28} {:8.1} ns/value", s.elapsed().as_nanos() as f64 / n);
    };
    t("decimal encode_block", &mut || for b in &blocks { let mut w = BitWriter::new(); DecimalRice::<f64>::default().encode_block(b, &mut w); });
    t("to_scaled x37 exponents", &mut || for b in &blocks { for k in -18..=18 { std::hint::black_box(b.iter().filter(|&&w| f64::from_bits(w).to_scaled(k).is_none()).count()); } });
    t("delta encode_block (ints)", &mut || for b in &ints { let mut w = BitWriter::new(); DeltaRice.encode_block(b, &mut w); });
    t("rice choose", &mut || for d in &deltas { std::hint::black_box(RiceBlock::choose(d)); });
    t("rice encode (incl choose)", &mut || for d in &deltas { let mut w = BitWriter::new(); RiceBlock::encode(d, &mut w); });
    t("adaptive encode", &mut || for d in &deltas { let mut w = BitWriter::new(); AdaptiveBlock::encode(d, &mut w); });
    t("bitwriter 1M x 9-bit writes", &mut || { let mut w = BitWriter::new(); for i in 0..(n as u64) { w.write(i & 511, 9); } std::hint::black_box(w.finish()); });
}
