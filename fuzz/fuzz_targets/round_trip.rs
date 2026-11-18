//! Any items under any mode, profile, block size and thread count decode to
//! exactly the input, and threaded encoding matches single-threaded.
#![no_main]

use gorilla_tsc::{CheckpointInterval, Config, Mode, Profile, Sample, Threads, decode, encode, encode_parallel};
use libfuzzer_sys::fuzz_target;

fn check<T: Sample>(items: &[T], mode_pick: u8, interval: u16, threads: u8) {
    let interval = CheckpointInterval::new(u32::from(interval % 1024) + 1).unwrap();
    let modes = Mode::all();
    let mode = match modes.get(usize::from(mode_pick) % (modes.len() + 1)) {
        Some(&m) => m,
        // One extra slot: a profile learned from the first half of the items.
        None => Mode::Profile(Profile::learn(&items[..items.len() / 2], interval)),
    };
    let config = Config::new(mode).with_interval(interval);
    let bytes = encode(items, config);
    let back = decode::<T>(&bytes).expect("own output must decode");
    assert!(back.iter().map(|x| x.to_word()).eq(items.iter().map(|x| x.to_word())), "round trip under {mode:?}");
    let t = Threads::new(usize::from(threads % 4) + 1).unwrap();
    assert_eq!(encode_parallel(items, config, t), bytes, "threaded output differs");
}

fn words<const N: usize>(data: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
    data.chunks_exact(N).map(|c| c.try_into().unwrap())
}

fuzz_target!(|data: &[u8]| {
    let [kind, mode, i0, i1, threads, rest @ ..] = data else { return };
    let interval = u16::from_le_bytes([*i0, *i1]);
    match kind % 4 {
        0 => check(&words::<4>(rest).map(i32::from_le_bytes).collect::<Vec<_>>(), *mode, interval, *threads),
        1 => check(&words::<8>(rest).map(i64::from_le_bytes).collect::<Vec<_>>(), *mode, interval, *threads),
        2 => check(&words::<4>(rest).map(f32::from_le_bytes).collect::<Vec<_>>(), *mode, interval, *threads),
        _ => check(&words::<8>(rest).map(f64::from_le_bytes).collect::<Vec<_>>(), *mode, interval, *threads),
    }
});
