//! Locked profiles: every per-block choice fixed up front, learned once from
//! a sample of the data, for fast encoding in production.
//!
//! [`Mode::Auto`](crate::Mode::Auto) tries several codecs on every block and
//! searches each one's settings. On the benchmark data a profile learned from
//! the first 10 000 values encodes about ten times faster with a median size
//! change of 0%. A profile writes ordinary blocks of one codec (usually
//! [`Codec::Delta`] or [`Codec::Decimal`]), so its streams decode like any other.
//!
//! ```
//! use gorilla_tsc::{decode, encode, CheckpointInterval, Config, Mode, Profile};
//! let prices: Vec<f64> = (0..5000).map(|i| f64::from(10_000 + i % 37) / 100.0).collect();
//! let profile = Profile::learn(&prices[..1000], CheckpointInterval::new(100).unwrap());
//! assert_eq!(profile.to_string().parse::<Profile>(), Ok(profile));
//! let bytes = encode(&prices, Config::new(Mode::Profile(profile)));
//! assert_eq!(decode::<f64>(&bytes).unwrap(), prices);
//! ```

use core::fmt;
use core::str::FromStr;

use crate::decimal::{best_encoding, scale, write_scaled_header};
use crate::delta::{Coder, DeltaChoice, Order};
use crate::stream::{encode_codec, parse_named};
use crate::{BitWriter, CheckpointInterval, Codec, ConfigError, DeltaRice, Sample};

/// A fixed encoding recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// [`Codec::Delta`] blocks with this order and coder.
    Delta(DeltaChoice),
    /// [`Codec::Decimal`] blocks scaled by `10^exponent`, integers coded with
    /// `delta`. A block where the exponent leaves more than one item in
    /// [`GUARD_RATIO`] unscaled is searched instead, so a profile learned on
    /// unrepresentative data degrades gracefully rather than storing a whole
    /// block raw.
    Decimal {
        /// Decimal exponent.
        exponent: i32,
        /// Order and coder of the scaled integers.
        delta: DeltaChoice,
    },
    /// One codec with its usual per-block search, for data where it beats
    /// every Delta and Decimal recipe (such as [`Codec::BitSplit`] for
    /// high-precision doubles, or [`Codec::Dictionary`] for few distinct values).
    Fixed(Codec),
}

/// A Decimal profile re-searches a block with more than one exception per this many items.
pub const GUARD_RATIO: usize = 100;

/// A fixed codec replaces the best Delta/Decimal recipe only if it is more
/// than this many percent smaller: recipes encode 10-80x faster, and a
/// profile is chosen for speed (use [`Mode::Auto`](crate::Mode::Auto) for
/// the last percent).
pub const RECIPE_SLACK_PERCENT: usize = 1;

/// `(bits, profile)` of the best recipe and the best fixed codec → the profile to use.
fn pick(recipe: Option<(usize, Profile)>, fixed: Option<(usize, Profile)>) -> Option<Profile> {
    match (recipe, fixed) {
        (Some((r, recipe)), Some((f, fixed))) => {
            Some(if f * 100 < r * (100 - RECIPE_SLACK_PERCENT) { fixed } else { recipe })
        }
        (r, f) => r.or(f).map(|(_, p)| p),
    }
}

/// Largest exponent magnitude any sample type scales by (`f64`'s).
const MAX_EXPONENT: i32 = 18;

const USAGE: &str = "profile must look like 'delta:<order>:<coder>' or 'decimal:<exponent>:<order>:<coder>'";

impl Profile {
    /// The codec whose blocks this profile writes (and the stream header records).
    #[must_use]
    pub const fn codec(self) -> Codec {
        match self {
            Self::Delta(_) => Codec::Delta,
            Self::Decimal { .. } => Codec::Decimal,
            Self::Fixed(c) => c,
        }
    }

    /// The profile that encodes `sample` smallest, blocked by `interval`:
    /// Delta recipes (every [`DeltaChoice`]) for integers, Decimal recipes
    /// (each exponent [`Codec::Decimal`] picks for one of the sample's blocks,
    /// times every [`DeltaChoice`]) for floats, then each codec in [`Sample::AUTO`].
    #[must_use]
    pub fn learn<T: Sample>(sample: &[T], interval: CheckpointInterval) -> Self {
        let words: Vec<T::Word> = sample.iter().map(|x| x.to_word()).collect();
        let blocks: Vec<&[T::Word]> = words.chunks(interval.get() as usize).collect();
        let mut candidates: Vec<Self> = Vec::new();
        if T::MAX_SCALE == 0 {
            candidates.extend(DeltaChoice::all().into_iter().map(Self::Delta));
        } else {
            let mut exponents: Vec<i32> = blocks.iter().map(|b| best_encoding::<T>(b).0).collect();
            exponents.sort_unstable();
            exponents.dedup();
            if exponents.is_empty() {
                exponents.push(0);
            }
            for exponent in exponents {
                candidates.extend(DeltaChoice::all().into_iter().map(|delta| Self::Decimal { exponent, delta }));
            }
        }
        let fixed: Vec<Self> = T::AUTO.iter().map(|&c| Self::Fixed(c)).collect();
        let cost = |p: Self| -> usize {
            blocks
                .iter()
                .map(|b| {
                    let mut w = BitWriter::new();
                    p.encode_block::<T>(b, &mut w);
                    w.bit_len()
                })
                .sum()
        };
        // Ties keep the earliest candidate: smaller exponent, then DeltaChoice order.
        let best = |ps: Vec<Self>| ps.into_iter().map(|p| (cost(p), p)).min_by_key(|&(c, _)| c);
        pick(best(candidates), best(fixed)).unwrap_or(Self::Delta(DeltaChoice::all()[0]))
    }

    /// Encode one block.
    pub(crate) fn encode_block<T: Sample>(self, items: &[T::Word], out: &mut BitWriter) {
        match self {
            Self::Delta(choice) => DeltaRice::encode_with(items, choice, out),
            Self::Decimal { exponent, delta } => {
                if exponent.unsigned_abs() <= T::MAX_SCALE.unsigned_abs() {
                    let scaled = scale::<T>(items, exponent);
                    if scaled.exceptions.len() * GUARD_RATIO <= items.len() {
                        write_scaled_header(&scaled, out);
                        DeltaRice::encode_with(&scaled.ints, delta, out);
                        return;
                    }
                }
                out.append(&best_encoding::<T>(items).1);
            }
            Self::Fixed(codec) => encode_codec::<T>(codec, items, out),
        }
    }
}

impl fmt::Display for Profile {
    /// `delta:<order>:<coder>`, `decimal:<exponent>:<order>:<coder>` or a codec name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Delta(d) => write!(f, "delta:{}:{}", d.order.name(), d.coder.name()),
            Self::Decimal { exponent, delta: d } => {
                write!(f, "decimal:{exponent}:{}:{}", d.order.name(), d.coder.name())
            }
            Self::Fixed(codec) => f.write_str(codec.name()),
        }
    }
}

impl FromStr for Profile {
    type Err = ConfigError;
    fn from_str(s: &str) -> Result<Self, ConfigError> {
        let usage = || ConfigError(format!("{USAGE} (got '{s}')"));
        let choice = |order: &str, coder: &str| -> Result<DeltaChoice, ConfigError> {
            Ok(DeltaChoice {
                order: parse_named(order, "order", &Order::ALL, Order::name)?,
                coder: parse_named(coder, "coder", &Coder::ALL, Coder::name)?,
            })
        };
        if !s.contains(':') {
            return parse_named(s, "profile codec", &Codec::ALL, Codec::name).map(Self::Fixed);
        }
        let parts: Vec<&str> = s.split(':').collect();
        match parts[..] {
            ["delta", order, coder] => Ok(Self::Delta(choice(order, coder)?)),
            ["decimal", exponent, order, coder] => {
                let exponent: i32 = exponent.parse().map_err(|_| usage())?;
                if exponent.unsigned_abs() > MAX_EXPONENT.unsigned_abs() {
                    return Err(ConfigError(format!(
                        "decimal exponent must be within -{MAX_EXPONENT}..={MAX_EXPONENT} (got {exponent})"
                    )));
                }
                Ok(Self::Decimal { exponent, delta: choice(order, coder)? })
            }
            _ => Err(usage()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitReader, BlockCodec, Config, DecimalRice, Mode, decode, encode};
    use proptest::prelude::*;

    const FIRST_RICE: DeltaChoice = DeltaChoice { order: Order::First, coder: Coder::Rice };
    const SECOND_ADAPTIVE: DeltaChoice = DeltaChoice { order: Order::Second, coder: Coder::Adaptive };

    fn interval(n: u32) -> CheckpointInterval {
        CheckpointInterval::new(n).unwrap()
    }

    fn block_bits<T: Sample>(p: Profile, v: &[T]) -> usize {
        let words: Vec<T::Word> = v.iter().map(|x| x.to_word()).collect();
        let mut w = BitWriter::new();
        p.encode_block::<T>(&words, &mut w);
        w.bit_len()
    }

    #[test]
    fn codecs() {
        assert_eq!(Profile::Delta(FIRST_RICE).codec(), Codec::Delta);
        assert_eq!(Profile::Decimal { exponent: 2, delta: FIRST_RICE }.codec(), Codec::Decimal);
    }

    #[test]
    fn strings() {
        let cases = [
            (Profile::Delta(FIRST_RICE), "delta:first:rice"),
            (Profile::Delta(SECOND_ADAPTIVE), "delta:second:adaptive"),
            (Profile::Decimal { exponent: 2, delta: FIRST_RICE }, "decimal:2:first:rice"),
            (
                Profile::Decimal { exponent: -3, delta: DeltaChoice { order: Order::Zero, coder: Coder::Adaptive } },
                "decimal:-3:zero:adaptive",
            ),
        ];
        for (p, s) in cases {
            assert_eq!(p.to_string(), s);
            assert_eq!(s.parse::<Profile>(), Ok(p));
        }
    }

    #[test]
    fn bad_strings() {
        let err = |s: &str| s.parse::<Profile>().unwrap_err().0;
        let usage = "profile must look like 'delta:<order>:<coder>' or 'decimal:<exponent>:<order>:<coder>'";
        for s in ["delta:first", "delta:first:rice:x", "decimal:2:first", "rle:first:rice", "decimal:x:first:rice", ":"]
        {
            assert!(err(s).starts_with(usage), "{s:?}: {}", err(s));
        }
        assert_eq!(err("delta:third:rice"), "order must be one of 'first', 'second', 'zero' (got 'third')");
        assert_eq!(err("delta:first:huffman"), "coder must be one of 'rice', 'adaptive' (got 'huffman')");
        assert_eq!(err("decimal:99:first:rice"), "decimal exponent must be within -18..=18 (got 99)");
        assert_eq!(err("decimal:-19:first:rice"), "decimal exponent must be within -18..=18 (got -19)");
        assert_eq!("decimal:18:first:rice".parse::<Profile>().map(Profile::codec), Ok(Codec::Decimal));
        assert_eq!("decimal:-18:first:rice".parse::<Profile>().map(Profile::codec), Ok(Codec::Decimal));
    }

    #[test]
    fn delta_profile_writes_a_delta_block() {
        let v: Vec<i64> = (0..500).map(|i| 1_700_000_000 + i * 60 + i % 3).collect();
        let words: Vec<u64> = v.iter().map(|x| x.to_word()).collect();
        for choice in DeltaChoice::all() {
            let mut a = BitWriter::new();
            Profile::Delta(choice).encode_block::<i64>(&words, &mut a);
            let mut b = BitWriter::new();
            DeltaRice::encode_with(&words, choice, &mut b);
            assert_eq!(a.finish(), b.finish());
        }
    }

    #[test]
    fn decimal_profile_writes_a_decimal_block() {
        let v: Vec<f64> = (0..500).map(|i| f64::from(i % 91) / 100.0).collect();
        let words: Vec<u64> = v.iter().map(|x| x.to_word()).collect();
        let p = Profile::Decimal { exponent: 2, delta: FIRST_RICE };
        let mut w = BitWriter::new();
        p.encode_block::<f64>(&words, &mut w);
        let mut expected = BitWriter::new();
        let scaled = scale::<f64>(&words, 2);
        write_scaled_header(&scaled, &mut expected);
        DeltaRice::encode_with(&scaled.ints, FIRST_RICE, &mut expected);
        let bytes = w.finish();
        assert_eq!(bytes, expected.finish());
        let mut out = Vec::new();
        DecimalRice::<f64>::default().decode_block(v.len(), &mut BitReader::new(&bytes), &mut out).unwrap();
        assert_eq!(out, words);
    }

    #[test]
    fn guard_researches_blocks_with_too_many_exceptions() {
        // Four decimals: a 2-digit exponent leaves almost everything unscaled.
        let v: Vec<f64> = (0..1000).map(|i| f64::from(i) / 10_000.0).collect();
        let locked = Profile::Decimal { exponent: 2, delta: FIRST_RICE };
        let words: Vec<u64> = v.iter().map(|x| x.to_word()).collect();
        let (_, searched) = best_encoding::<f64>(&words);
        assert_eq!(block_bits(locked, &v), searched.bit_len());
    }

    #[test]
    fn guard_threshold_is_one_percent() {
        // 1000 two-decimal values with exactly 10 (then 11) unscalable at exponent 2.
        let make = |bad: usize| -> Vec<f64> {
            (0..1000)
                .map(|i| {
                    if i < bad {
                        f64::from(u32::try_from(i).unwrap()) + 0.123_456_789
                    } else {
                        f64::from(u32::try_from(i % 50).unwrap()) / 100.0
                    }
                })
                .collect()
        };
        // A poor recipe, so a searched block is visibly different from a locked one.
        let zero_rice = DeltaChoice { order: Order::Zero, coder: Coder::Rice };
        let locked = Profile::Decimal { exponent: 2, delta: zero_rice };
        let at_limit = make(10);
        let words: Vec<u64> = at_limit.iter().map(|x| x.to_word()).collect();
        let mut expected = BitWriter::new();
        let scaled = scale::<f64>(&words, 2);
        write_scaled_header(&scaled, &mut expected);
        DeltaRice::encode_with(&scaled.ints, zero_rice, &mut expected);
        assert_eq!(block_bits(locked, &at_limit), expected.bit_len(), "10 of 1000 stays locked");
        let over = make(11);
        let words: Vec<u64> = over.iter().map(|x| x.to_word()).collect();
        let searched = best_encoding::<f64>(&words).1.bit_len();
        assert_eq!(block_bits(locked, &over), searched, "11 of 1000 is searched");
        let mut kept = BitWriter::new();
        let scaled = scale::<f64>(&words, 2);
        write_scaled_header(&scaled, &mut kept);
        DeltaRice::encode_with(&scaled.ints, zero_rice, &mut kept);
        assert!(searched < kept.bit_len(), "the search must matter for this test");
    }

    #[test]
    fn learns_delta_for_integers() {
        // Constant step: second differences are all zero.
        let v: Vec<i64> = (0..3000).map(|i| 1_700_000_000_000 + i * 1000).collect();
        let p = Profile::learn(&v, interval(1000));
        let Profile::Delta(choice) = p else { panic!("{p}") };
        assert_eq!(choice.order, Order::Second);
        for other in DeltaChoice::all() {
            assert!(block_bits(p, &v[..1000]) <= block_bits(Profile::Delta(other), &v[..1000]));
        }
    }

    #[test]
    fn learns_the_decimal_exponent_for_floats() {
        // A two-decimal random walk: thousands of distinct values, small steps.
        let mut level = 200_000i32;
        let v: Vec<f64> = (0..3000)
            .map(|i: i32| {
                level += (i * 7919 % 13) - 6;
                f64::from(level) / 100.0
            })
            .collect();
        let p = Profile::learn(&v, interval(1000));
        assert!(matches!(p, Profile::Decimal { exponent: 2, .. }), "{p}");
    }

    #[test]
    fn learns_a_fixed_codec_when_it_beats_every_recipe() {
        // Four high-bit patterns over random low 32 bits: no decimal form,
        // no repeats, but two bits per item select the high half.
        let patterns = [0.61f64, 0.62, 0.63, 0.64].map(|p| p.to_bits() >> 32);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let v: Vec<f64> = (0..3000)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                f64::from_bits((patterns[i % 4] << 32) | (x >> 32))
            })
            .collect();
        assert_eq!(Profile::learn(&v, interval(1000)), Profile::Fixed(Codec::BitSplit));
        // A handful of distinct values with many digits each.
        let few = [1.234_567_89, 98.765_432_1, 5.555_555_5];
        let v: Vec<f64> = (0..3000).map(|i| few[(i * 7) % 3]).collect();
        assert_eq!(Profile::learn(&v, interval(1000)), Profile::Fixed(Codec::Dictionary));
    }

    #[test]
    fn fixed_codecs_must_beat_recipes_by_the_slack() {
        let recipe = Profile::Delta(FIRST_RICE);
        let fixed = Profile::Fixed(Codec::Dictionary);
        assert_eq!(pick(Some((1000, recipe)), Some((990, fixed))), Some(recipe), "exactly 1% smaller: keep the recipe");
        assert_eq!(pick(Some((1000, recipe)), Some((989, fixed))), Some(fixed));
        assert_eq!(pick(Some((1000, recipe)), Some((1000, fixed))), Some(recipe));
        assert_eq!(pick(Some((1000, recipe)), Some((2000, fixed))), Some(recipe));
        assert_eq!(pick(Some((0, recipe)), Some((0, fixed))), Some(recipe));
        assert_eq!(pick(None, Some((5, fixed))), Some(fixed));
        assert_eq!(pick(Some((5, recipe)), None), Some(recipe));
        assert_eq!(pick(None, None), None);
    }

    #[test]
    fn fixed_profiles_write_their_codec() {
        let p = Profile::Fixed(Codec::BitSplit);
        assert_eq!(p.codec(), Codec::BitSplit);
        assert_eq!(p.to_string(), "bit_split");
        assert_eq!("bit_split".parse::<Profile>(), Ok(p));
        assert_eq!(
            "auto".parse::<Profile>().unwrap_err().0,
            "profile codec must be one of 'xor', 'dod', 'xor_dod', 'dod_xor', 'delta', 'decimal', 'raw', 'dictionary', 'decimal_split', 'bit_split' (got 'auto')"
        );
        let v: Vec<f64> = (0..300).map(|i| f64::from(i).sqrt()).collect();
        let bytes = encode(&v, Config::new(Mode::Profile(p)));
        assert_eq!(bytes, encode(&v, Config::new(Mode::Fixed(Codec::BitSplit))));
    }

    #[test]
    fn learns_from_empty_and_tiny_samples() {
        assert_eq!(Profile::learn::<i64>(&[], interval(100)), Profile::Delta(FIRST_RICE));
        assert_eq!(Profile::learn::<f64>(&[], interval(100)), Profile::Decimal { exponent: 0, delta: FIRST_RICE });
        assert!(matches!(Profile::learn(&[1.5f64], interval(100)), Profile::Decimal { .. }));
    }

    #[test]
    fn profile_streams_decode_as_their_codec() {
        let v: Vec<f64> = (0..2500).map(|i| f64::from(i % 300) / 10.0).collect();
        let p = Profile::learn(&v[..1000], interval(100));
        let bytes = encode(&v, Config::new(Mode::Profile(p)));
        assert_eq!(decode::<f64>(&bytes), Ok(v.clone()));
        assert_eq!(crate::read_header(&bytes).unwrap().0.config.mode, Mode::Fixed(Codec::Decimal));
        // Close to auto on data like its sample.
        let auto = encode(&v, Config::new(Mode::Auto)).len();
        assert!(bytes.len() * 10 <= auto * 11, "{} vs auto {auto}", bytes.len());
    }

    #[test]
    fn mode_strings_accept_profiles() {
        assert_eq!("delta:first:rice".parse::<Mode>(), Ok(Mode::Profile(Profile::Delta(FIRST_RICE))));
        assert_eq!(Mode::Profile(Profile::Delta(FIRST_RICE)).name(), "delta");
        assert!("auto".parse::<Mode>().is_ok());
    }

    proptest! {
        #[test]
        fn any_profile_round_trips_any_floats(
            v in prop::collection::vec(any::<u64>(), 0..400),
            exponent in -22i32..=22,
            choice in 0usize..6,
        ) {
            let v: Vec<f64> = v.into_iter().map(f64::from_bits).collect();
            let p = Profile::Decimal { exponent, delta: DeltaChoice::all()[choice] };
            let bytes = encode(&v, Config::new(Mode::Profile(p)).with_interval(interval(100)));
            let back: Vec<u64> = decode::<f64>(&bytes).unwrap().iter().map(|x| x.to_bits()).collect();
            prop_assert_eq!(back, v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }

        #[test]
        fn any_profile_round_trips_any_ints(v in prop::collection::vec(any::<i32>(), 0..400), choice in 0usize..6) {
            let p = Profile::Delta(DeltaChoice::all()[choice]);
            let bytes = encode(&v, Config::new(Mode::Profile(p)).with_interval(interval(64)));
            prop_assert_eq!(decode::<i32>(&bytes).unwrap(), v);
        }

        #[test]
        fn learned_profile_is_the_best_on_its_sample(v in prop::collection::vec(-5000i64..5000, 1..300)) {
            let p = Profile::learn(&v, interval(100));
            let cost = |p: Profile| v.chunks(100).map(|b| block_bits(p, b)).sum::<usize>();
            for c in DeltaChoice::all() {
                prop_assert!(cost(p) <= cost(Profile::Delta(c)));
            }
        }
    }
}
