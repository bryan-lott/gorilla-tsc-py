//! Datasets shared by every experiment.
//!
//! Floats: the 22 Elf benchmark datasets (about 100k values each, the same
//! sets the Elf and ALP papers use) plus the full Chimp Stocks-Germany series.
//! Integers: two timestamp series.

use std::path::{Path, PathBuf};

/// Where `scripts/fetch-bench-data.sh` puts the data.
#[must_use]
pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench-data")
}

/// Whether consecutive values are consecutive in time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    TimeSeries,
    NonTimeSeries,
}

impl Kind {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Kind::TimeSeries => "time series",
            Kind::NonTimeSeries => "non-time-series",
        }
    }
}

pub struct FloatSet {
    pub name: &'static str,
    pub kind: Kind,
    pub values: Vec<f64>,
}

pub struct IntSet {
    pub name: &'static str,
    pub values: Vec<i64>,
}

/// (file stem, name, kind) of the Elf datasets.
pub const ELF: [(&str, &str, Kind); 22] = [
    ("Air-pressure", "Air-pressure", Kind::TimeSeries),
    ("Air-sensor", "Air-sensor", Kind::TimeSeries),
    ("Basel-temp", "Basel-temp", Kind::TimeSeries),
    ("Basel-wind", "Basel-wind", Kind::TimeSeries),
    ("Bird-migration", "Bird-migration", Kind::TimeSeries),
    ("Bitcoin-price", "Bitcoin-price", Kind::TimeSeries),
    ("City-temp", "City-temp", Kind::TimeSeries),
    ("Dew-point-temp", "Dew-point-temp", Kind::TimeSeries),
    ("IR-bio-temp", "IR-bio-temp", Kind::TimeSeries),
    ("PM10-dust", "PM10-dust", Kind::TimeSeries),
    ("Stocks-DE", "Stocks-DE", Kind::TimeSeries),
    ("Stocks-UK", "Stocks-UK", Kind::TimeSeries),
    ("Stocks-USA", "Stocks-USA", Kind::TimeSeries),
    ("Wind-Speed", "Wind-speed", Kind::TimeSeries),
    ("Blockchain-tr", "Blockchain-tr", Kind::NonTimeSeries),
    ("City-lat", "City-lat", Kind::NonTimeSeries),
    ("City-lon", "City-lon", Kind::NonTimeSeries),
    ("Food-price", "Food-price", Kind::NonTimeSeries),
    ("POI-lat", "POI-lat", Kind::NonTimeSeries),
    ("POI-lon", "POI-lon", Kind::NonTimeSeries),
    ("SSD-bench", "SSD-bench", Kind::NonTimeSeries),
    ("electric_vehicle_charging", "Vehicle-charge", Kind::NonTimeSeries),
];

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e} (run scripts/fetch-bench-data.sh)", path.display()))
}

/// One float per line (Elf format); unparsable lines are skipped.
#[must_use]
pub fn float_lines(path: &Path) -> Vec<f64> {
    read(path).lines().filter_map(|l| l.trim().parse().ok()).collect()
}

/// Column `col` of a comma-separated file.
#[must_use]
pub fn float_column(path: &Path, col: usize) -> Vec<f64> {
    read(path).lines().filter_map(|l| l.split(',').nth(col)?.trim().parse().ok()).collect()
}

/// The 22 Elf datasets.
#[must_use]
pub fn elf_sets() -> Vec<FloatSet> {
    let dir = data_dir().join("elf");
    ELF.iter()
        .map(|&(file, name, kind)| FloatSet { name, kind, values: float_lines(&dir.join(format!("{file}.csv"))) })
        .collect()
}

/// Elf datasets plus the full 5M-value Chimp Stocks-Germany series.
#[must_use]
pub fn float_sets() -> Vec<FloatSet> {
    let mut v = elf_sets();
    v.push(FloatSet {
        name: "Stocks-Germany (5M)",
        kind: Kind::TimeSeries,
        values: float_column(&data_dir().join("Stocks-Germany-sample.txt"), 2),
    });
    v
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

/// Seconds timestamps of the Chimp Stocks-Germany trades.
#[must_use]
pub fn stock_timestamps() -> Vec<i64> {
    read(&data_dir().join("Stocks-Germany-sample.txt"))
        .lines()
        .filter_map(|l| {
            let mut f = l.split(',');
            let date: Vec<i64> = f.next()?.split('/').map(|x| x.parse().ok()).collect::<Option<_>>()?;
            let time: Vec<i64> = f.next()?.split(':').map(|x| x.parse().ok()).collect::<Option<_>>()?;
            Some(days_from_civil(date[2], date[0], date[1]) * 86_400 + time[0] * 3600 + time[1] * 60 + time[2])
        })
        .collect()
}

/// Regular 10 s scrape interval in ms with +/- 5 ms deterministic jitter.
#[must_use]
pub fn scrape_timestamps(n: usize) -> Vec<i64> {
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

#[must_use]
pub fn int_sets() -> Vec<IntSet> {
    vec![
        IntSet { name: "Stock timestamps (s)", values: stock_timestamps() },
        IntSet { name: "Scrape timestamps (10 s +/- 5 ms)", values: scrape_timestamps(1_000_000) },
    ]
}
