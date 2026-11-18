//! Where does a locked profile lose to auto on one dataset? (development aid)
#[path = "../profile.rs"]
mod profile;
use gorilla_tsc::{BitWriter, Order};
use profile::{Coder, KPolicy, Profile};

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "Food-price".into());
    let set = bench_common::float_sets().into_iter().find(|s| s.name == name).unwrap();
    let mut tally = std::collections::BTreeMap::new();
    let (mut locked_total, mut oracle_total) = (0usize, 0usize);
    let locked = Profile { k: KPolicy::Fixed(3), order: Order::First, coder: Coder::Adaptive };
    for b in set.values.chunks(1000) {
        let bits = |p: Profile| { let mut w = BitWriter::new(); profile::encode_block(b, p, &mut w); w.bit_len() };
        locked_total += bits(locked);
        let mut best = (usize::MAX, String::new());
        for k in -18..=18 {
            for order in [Order::First, Order::Second, Order::Zero] {
                for coder in [Coder::Rice, Coder::Adaptive] {
                    let p = Profile { k: KPolicy::Fixed(k), order, coder };
                    let c = bits(p);
                    if c < best.0 { best = (c, format!("{:?} {:?} k={}", order, coder, if k == 3 { "3(same)".to_string() } else { k.to_string() })); }
                }
            }
        }
        oracle_total += best.0;
        *tally.entry(best.1).or_insert(0) += 1;
    }
    let n = set.values.len() as f64;
    println!("{name}: locked {:.2}  per-block oracle {:.2} bits/value", locked_total as f64 / n, oracle_total as f64 / n);
    for (k, v) in tally { println!("  {v:4} blocks  {k}"); }
}
