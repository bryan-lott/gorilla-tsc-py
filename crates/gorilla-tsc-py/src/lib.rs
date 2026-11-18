//! Python bindings for `gorilla-tsc`. Deliberately thin: parsing, validation
//! and all codec logic live in the core crate; this file only converts types.

use gorilla_tsc as gt;
use gorilla_tsc::{CheckpointInterval, Config, Decoded, Encoder, Mode, SampleType};
// `Result<T, PyErr>` rather than the `PyResult<T>` alias, so cargo-mutants
// can generate type-correct replacement values (it does not see through aliases).
use pyo3::create_exception;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyList, PyTuple};

create_exception!(_native, DecodeError, PyValueError, "The input is not a valid gorilla-tsc stream.");

const TS_MSG: &str = "All timestamps must be ints";
const VAL_MSG: &str = "All values must be floats";
const PAIR_MSG: &str = "All pairs must be (int, float) tuples";

fn decode_err(e: &gt::DecodeError) -> PyErr {
    DecodeError::new_err(e.to_string())
}

/// `0` = every core, `n >= 1` = exactly `n` threads.
fn parse_threads(threads: i64) -> Result<gt::Threads, PyErr> {
    match usize::try_from(threads) {
        Ok(0) => Ok(gt::Threads::available()),
        Ok(n) => gt::Threads::new(n).ok_or_else(|| PyValueError::new_err("threads out of range")),
        Err(_) => Err(PyValueError::new_err(format!("threads must be 0 (all cores) or more (got {threads})"))),
    }
}

fn parse_config(mode: &str, checkpoint_interval: i64) -> Result<Config, PyErr> {
    let mode: Mode = mode.parse().map_err(|e: gt::ConfigError| PyValueError::new_err(e.0))?;
    let interval = CheckpointInterval::try_from(checkpoint_interval).map_err(|e| PyValueError::new_err(e.0))?;
    Ok(Config::new(mode).with_interval(interval))
}

/// Timestamp element types.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IntType {
    I32,
    I64,
}

/// Value element types.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FloatType {
    F32,
    F64,
}

impl IntType {
    fn parse(s: &str) -> Result<Self, PyErr> {
        match s.parse() {
            Ok(SampleType::I32) => Ok(Self::I32),
            Ok(SampleType::I64) => Ok(Self::I64),
            _ => Err(PyValueError::new_err(format!("timestamp precision must be 'int32' or 'int64' (got '{s}')"))),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::I32 => SampleType::I32.name(),
            Self::I64 => SampleType::I64.name(),
        }
    }
}

impl FloatType {
    fn parse(s: &str) -> Result<Self, PyErr> {
        match s.parse() {
            Ok(SampleType::F32) => Ok(Self::F32),
            Ok(SampleType::F64) => Ok(Self::F64),
            _ => Err(PyValueError::new_err(format!("value precision must be 'f32' or 'f64' (got '{s}')"))),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::F32 => SampleType::F32.name(),
            Self::F64 => SampleType::F64.name(),
        }
    }
}

/// Extract an `int` of the target width; floats are a `TypeError`, out-of-range an `OverflowError`.
fn strict_int<T: TryFrom<i64>>(item: &Bound<'_, PyAny>, msg: &'static str) -> Result<T, PyErr> {
    if !item.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(msg));
    }
    let v: i64 = item.extract()?;
    T::try_from(v).map_err(|_| PyOverflowError::new_err(format!("timestamp {v} out of range for int32")))
}

/// Extract a `float`; ints are rejected rather than silently converted.
fn strict_float(item: &Bound<'_, PyAny>, msg: &'static str) -> Result<f64, PyErr> {
    Ok(item.cast::<PyFloat>().map_err(|_| PyTypeError::new_err(msg))?.value())
}

#[allow(clippy::cast_possible_truncation)] // f32 precision is the caller's explicit choice
fn to_f32(v: f64) -> f32 {
    v as f32
}

fn ints<T: TryFrom<i64>>(items: &Bound<'_, PyAny>, msg: &'static str) -> Result<Vec<T>, PyErr> {
    items.try_iter()?.map(|i| strict_int(&i?, msg)).collect()
}

fn floats(items: &Bound<'_, PyAny>, msg: &'static str) -> Result<Vec<f64>, PyErr> {
    items.try_iter()?.map(|i| strict_float(&i?, msg)).collect()
}

/// Bytes from either raw `bytes` or an encoder result dict.
fn bytes_of(data: &Bound<'_, PyAny>, key: &str) -> Result<Vec<u8>, PyErr> {
    if let Ok(d) = data.cast::<PyDict>() {
        let v = d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("missing key '{key}'")))?;
        return v.extract();
    }
    data.extract().map_err(|_| PyTypeError::new_err("expected bytes or an encoder result dict"))
}

fn decoded_to_list(py: Python<'_>, d: Decoded) -> Result<Bound<'_, PyList>, PyErr> {
    match d {
        Decoded::I32(v) => PyList::new(py, v),
        Decoded::I64(v) => PyList::new(py, v),
        Decoded::F32(v) => PyList::new(py, v.into_iter().map(f64::from)),
        Decoded::F64(v) => PyList::new(py, v),
    }
}

fn finished() -> PyErr {
    PyRuntimeError::new_err("encoder already finished")
}

fn result_dict<'py>(
    py: Python<'py>,
    bytes: &[u8],
    count_key: &str,
    count: u64,
    precision: &str,
    config: Config,
) -> Result<Bound<'py, PyDict>, PyErr> {
    let d = PyDict::new(py);
    d.set_item("encoded", PyBytes::new(py, bytes))?;
    d.set_item(count_key, count)?;
    d.set_item("precision", precision)?;
    d.set_item("mode", config.mode.name())?;
    d.set_item("checkpoint_interval", config.checkpoint_interval.get())?;
    Ok(d)
}

// --- timestamps ---------------------------------------------------------------

enum TsEncoder {
    I32(Encoder<i32>),
    I64(Encoder<i64>),
}

/// Compresses integer timestamps (delta-of-delta by default).
#[pyclass(module = "gorilla_tsc_py")]
struct TimestampsEncoder {
    inner: Option<TsEncoder>,
    precision: IntType,
    config: Config,
}

#[pymethods]
impl TimestampsEncoder {
    #[new]
    #[pyo3(signature = (precision = "int64", checkpoint_interval = 100, mode = "auto", threads = 1))]
    fn new(precision: &str, checkpoint_interval: i64, mode: &str, threads: i64) -> Result<Self, PyErr> {
        let precision = IntType::parse(precision)?;
        let config = parse_config(mode, checkpoint_interval)?;
        let threads = parse_threads(threads)?;
        let inner = match precision {
            IntType::I32 => TsEncoder::I32(Encoder::with_threads(config, threads)),
            IntType::I64 => TsEncoder::I64(Encoder::with_threads(config, threads)),
        };
        Ok(Self { inner: Some(inner), precision, config })
    }

    /// Append one timestamp. Returns True.
    fn encode_next(&mut self, ts: &Bound<'_, PyAny>) -> Result<bool, PyErr> {
        match self.inner.as_mut().ok_or_else(finished)? {
            TsEncoder::I32(e) => e.push(strict_int(ts, "Timestamp must be int")?),
            TsEncoder::I64(e) => e.push(strict_int(ts, "Timestamp must be int")?),
        }
        Ok(true)
    }

    /// Append all `timestamps`, then finish the stream.
    fn encode_all<'py>(
        &mut self,
        py: Python<'py>,
        timestamps: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyDict>, PyErr> {
        match self.inner.as_mut().ok_or_else(finished)? {
            TsEncoder::I32(e) => {
                let v = ints::<i32>(timestamps, TS_MSG)?;
                py.detach(|| e.extend(&v));
            }
            TsEncoder::I64(e) => {
                let v = ints::<i64>(timestamps, TS_MSG)?;
                py.detach(|| e.extend(&v));
            }
        }
        self.finish(py)
    }

    /// Finish the stream and return the encoded result.
    fn finish<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr> {
        let inner = self.inner.take().ok_or_else(finished)?;
        let (count, bytes) = py.detach(|| match inner {
            TsEncoder::I32(e) => (e.len(), e.finish()),
            TsEncoder::I64(e) => (e.len(), e.finish()),
        });
        result_dict(py, &bytes, "nb_timestamps", count, self.precision.name(), self.config)
    }
}

/// Decodes a timestamp stream.
#[pyclass(module = "gorilla_tsc_py")]
struct TimestampsDecoder {
    bytes: Vec<u8>,
}

#[pymethods]
impl TimestampsDecoder {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> Result<Self, PyErr> {
        Ok(Self { bytes: bytes_of(data, "encoded")? })
    }

    fn decode_all<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyList>, PyErr> {
        decode_ints(py, &self.bytes)
    }
}

fn decode_ints<'py>(py: Python<'py>, bytes: &[u8]) -> Result<Bound<'py, PyList>, PyErr> {
    match gt::decode_any(bytes).map_err(|e| decode_err(&e))? {
        d @ (Decoded::I32(_) | Decoded::I64(_)) => decoded_to_list(py, d),
        Decoded::F32(_) | Decoded::F64(_) => {
            Err(DecodeError::new_err("expected a timestamp (integer) stream, found floats"))
        }
    }
}

fn decode_floats<'py>(py: Python<'py>, bytes: &[u8]) -> Result<Bound<'py, PyList>, PyErr> {
    match gt::decode_any(bytes).map_err(|e| decode_err(&e))? {
        d @ (Decoded::F32(_) | Decoded::F64(_)) => decoded_to_list(py, d),
        Decoded::I32(_) | Decoded::I64(_) => {
            Err(DecodeError::new_err("expected a value (float) stream, found integers"))
        }
    }
}

// --- values -------------------------------------------------------------------

enum ValEncoder {
    F32(Encoder<f32>),
    F64(Encoder<f64>),
}

/// Compresses floating point values (XOR by default).
#[pyclass(module = "gorilla_tsc_py")]
struct ValuesEncoder {
    inner: Option<ValEncoder>,
    precision: FloatType,
    config: Config,
}

#[pymethods]
impl ValuesEncoder {
    #[new]
    #[pyo3(signature = (precision = "f64", mode = "auto", checkpoint_interval = 100, threads = 1))]
    fn new(precision: &str, mode: &str, checkpoint_interval: i64, threads: i64) -> Result<Self, PyErr> {
        let precision = FloatType::parse(precision)?;
        let config = parse_config(mode, checkpoint_interval)?;
        let threads = parse_threads(threads)?;
        let inner = match precision {
            FloatType::F32 => ValEncoder::F32(Encoder::with_threads(config, threads)),
            FloatType::F64 => ValEncoder::F64(Encoder::with_threads(config, threads)),
        };
        Ok(Self { inner: Some(inner), precision, config })
    }

    /// Append one value. Returns True.
    fn encode_next(&mut self, value: &Bound<'_, PyAny>) -> Result<bool, PyErr> {
        let v = strict_float(value, "Value must be float")?;
        match self.inner.as_mut().ok_or_else(finished)? {
            ValEncoder::F32(e) => e.push(to_f32(v)),
            ValEncoder::F64(e) => e.push(v),
        }
        Ok(true)
    }

    /// Append all `values`, then finish the stream.
    fn encode_all<'py>(&mut self, py: Python<'py>, values: &Bound<'py, PyAny>) -> Result<Bound<'py, PyDict>, PyErr> {
        let v = floats(values, VAL_MSG)?;
        match self.inner.as_mut().ok_or_else(finished)? {
            ValEncoder::F32(e) => {
                let v: Vec<f32> = v.into_iter().map(to_f32).collect();
                py.detach(|| e.extend(&v));
            }
            ValEncoder::F64(e) => py.detach(|| e.extend(&v)),
        }
        self.finish(py)
    }

    /// Finish the stream and return the encoded result.
    fn finish<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr> {
        let inner = self.inner.take().ok_or_else(finished)?;
        let (count, bytes) = py.detach(|| match inner {
            ValEncoder::F32(e) => (e.len(), e.finish()),
            ValEncoder::F64(e) => (e.len(), e.finish()),
        });
        result_dict(py, &bytes, "nb_values", count, self.precision.name(), self.config)
    }
}

/// Decodes a value stream.
#[pyclass(module = "gorilla_tsc_py")]
struct ValuesDecoder {
    bytes: Vec<u8>,
}

#[pymethods]
impl ValuesDecoder {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> Result<Self, PyErr> {
        Ok(Self { bytes: bytes_of(data, "encoded")? })
    }

    fn decode_all<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyList>, PyErr> {
        decode_floats(py, &self.bytes)
    }
}

// --- pairs --------------------------------------------------------------------

/// Compresses `(timestamp, value)` pairs, stable-sorted by timestamp.
#[pyclass(module = "gorilla_tsc_py")]
struct PairsEncoder {
    ts_precision: IntType,
    val_precision: FloatType,
    ts_config: Config,
    val_config: Config,
    threads: gt::Threads,
}

fn extract_pairs(pairs: &Bound<'_, PyAny>) -> Result<Vec<(i64, f64)>, PyErr> {
    pairs
        .try_iter()?
        .map(|p| {
            let p = p?;
            let t = p.cast::<PyTuple>().map_err(|_| PyTypeError::new_err(PAIR_MSG))?;
            if t.len() != 2 {
                return Err(PyTypeError::new_err(PAIR_MSG));
            }
            Ok((strict_int(&t.get_item(0)?, PAIR_MSG)?, strict_float(&t.get_item(1)?, PAIR_MSG)?))
        })
        .collect()
}

fn narrow_ts(pairs: &[(i64, f64)]) -> Result<Vec<(i32, f64)>, PyErr> {
    pairs
        .iter()
        .map(|&(t, v)| {
            let t = i32::try_from(t)
                .map_err(|_| PyOverflowError::new_err(format!("timestamp {t} out of range for int32")))?;
            Ok((t, v))
        })
        .collect()
}

fn narrow_val<T: Copy>(pairs: &[(T, f64)]) -> Vec<(T, f32)> {
    pairs.iter().map(|&(t, v)| (t, to_f32(v))).collect()
}

#[pymethods]
impl PairsEncoder {
    #[new]
    #[pyo3(signature = (
        ts_precision = "int64", val_precision = "f64", val_mode = "auto",
        checkpoint_interval = 100, ts_mode = "auto", threads = 1,
    ))]
    fn new(
        ts_precision: &str,
        val_precision: &str,
        val_mode: &str,
        checkpoint_interval: i64,
        ts_mode: &str,
        threads: i64,
    ) -> Result<Self, PyErr> {
        Ok(Self {
            ts_precision: IntType::parse(ts_precision)?,
            val_precision: FloatType::parse(val_precision)?,
            ts_config: parse_config(ts_mode, checkpoint_interval)?,
            val_config: parse_config(val_mode, checkpoint_interval)?,
            threads: parse_threads(threads)?,
        })
    }

    fn encode_all<'py>(&self, py: Python<'py>, pairs: &Bound<'py, PyAny>) -> Result<Bound<'py, PyDict>, PyErr> {
        let pairs = extract_pairs(pairs)?;
        let (tc, vc, th) = (self.ts_config, self.val_config, self.threads);
        let (ts, vals) = match (self.ts_precision, self.val_precision) {
            (IntType::I64, FloatType::F64) => py.detach(|| gt::encode_pairs_with(&pairs, tc, vc, th)),
            (IntType::I64, FloatType::F32) => {
                let p = narrow_val(&pairs);
                py.detach(|| gt::encode_pairs_with(&p, tc, vc, th))
            }
            (IntType::I32, FloatType::F64) => {
                let p = narrow_ts(&pairs)?;
                py.detach(|| gt::encode_pairs_with(&p, tc, vc, th))
            }
            (IntType::I32, FloatType::F32) => {
                let p = narrow_val(&narrow_ts(&pairs)?);
                py.detach(|| gt::encode_pairs_with(&p, tc, vc, th))
            }
        };
        let d = PyDict::new(py);
        d.set_item("encoded_ts", PyBytes::new(py, &ts))?;
        d.set_item("encoded_val", PyBytes::new(py, &vals))?;
        d.set_item("nb_pairs", pairs.len())?;
        d.set_item("ts_precision", self.ts_precision.name())?;
        d.set_item("val_precision", self.val_precision.name())?;
        d.set_item("ts_mode", tc.mode.name())?;
        d.set_item("val_mode", vc.mode.name())?;
        d.set_item("checkpoint_interval", tc.checkpoint_interval.get())?;
        Ok(d)
    }
}

/// Decodes pairs written by `PairsEncoder`.
#[pyclass(module = "gorilla_tsc_py")]
struct PairsDecoder {
    ts: Vec<u8>,
    vals: Vec<u8>,
}

#[pymethods]
impl PairsDecoder {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> Result<Self, PyErr> {
        Ok(Self { ts: bytes_of(data, "encoded_ts")?, vals: bytes_of(data, "encoded_val")? })
    }

    fn decode_all<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyList>, PyErr> {
        let ts = decode_ints(py, &self.ts)?;
        let vals = decode_floats(py, &self.vals)?;
        if ts.len() != vals.len() {
            return Err(DecodeError::new_err("pair streams differ in length"));
        }
        let pairs: Vec<Bound<'py, PyTuple>> =
            ts.iter().zip(vals.iter()).map(|(t, v)| PyTuple::new(py, [t, v])).collect::<Result<_, PyErr>>()?;
        PyList::new(py, pairs)
    }
}

// --- functions ----------------------------------------------------------------

fn estimate_dict(py: Python<'_>, e: gt::Estimate) -> Result<Bound<'_, PyDict>, PyErr> {
    let d = PyDict::new(py);
    d.set_item("ratio", e.ratio)?;
    d.set_item("compressed_size", e.compressed_size)?;
    d.set_item("bits_per_item", e.bits_per_item)?;
    Ok(d)
}

/// Estimate compression of integer timestamps from their first 1000 items.
#[pyfunction]
#[pyo3(signature = (data, precision = "int64", checkpoint_interval = 100, mode = "auto"))]
fn estimate_ts_compression<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    precision: &str,
    checkpoint_interval: i64,
    mode: &str,
) -> Result<Bound<'py, PyDict>, PyErr> {
    let config = parse_config(mode, checkpoint_interval)?;
    let e = match IntType::parse(precision)? {
        IntType::I32 => gt::estimate(&ints::<i32>(data, TS_MSG)?, config),
        IntType::I64 => gt::estimate(&ints::<i64>(data, TS_MSG)?, config),
    };
    estimate_dict(py, e)
}

/// Estimate compression of float values from their first 1000 items.
#[pyfunction]
#[pyo3(signature = (data, mode = "auto", precision = "f64", checkpoint_interval = 100))]
fn estimate_val_compression<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    mode: &str,
    precision: &str,
    checkpoint_interval: i64,
) -> Result<Bound<'py, PyDict>, PyErr> {
    let config = parse_config(mode, checkpoint_interval)?;
    let precision = FloatType::parse(precision)?;
    let v = floats(data, VAL_MSG)?;
    let e = match precision {
        FloatType::F32 => gt::estimate(&v.into_iter().map(to_f32).collect::<Vec<_>>(), config),
        FloatType::F64 => gt::estimate(&v, config),
    };
    estimate_dict(py, e)
}

/// Compress `values` in one call, optionally across `threads` threads (the
/// output does not depend on the thread count). Returns the encoded bytes.
#[pyfunction]
#[pyo3(signature = (values, precision = "f64", mode = "auto", checkpoint_interval = 100, threads = 1))]
fn encode<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    precision: &str,
    mode: &str,
    checkpoint_interval: i64,
    threads: i64,
) -> Result<Bound<'py, PyBytes>, PyErr> {
    let config = parse_config(mode, checkpoint_interval)?;
    let threads = parse_threads(threads)?;
    let precision: SampleType = precision.parse().map_err(|e: gt::ConfigError| PyValueError::new_err(e.0))?;
    let bytes = match precision {
        SampleType::I32 => {
            let v = ints::<i32>(values, TS_MSG)?;
            py.detach(|| gt::encode_parallel(&v, config, threads))
        }
        SampleType::I64 => {
            let v = ints::<i64>(values, TS_MSG)?;
            py.detach(|| gt::encode_parallel(&v, config, threads))
        }
        SampleType::F32 => {
            let v: Vec<f32> = floats(values, VAL_MSG)?.into_iter().map(to_f32).collect();
            py.detach(|| gt::encode_parallel(&v, config, threads))
        }
        SampleType::F64 => {
            let v = floats(values, VAL_MSG)?;
            py.detach(|| gt::encode_parallel(&v, config, threads))
        }
    };
    Ok(PyBytes::new(py, &bytes))
}

/// Learn a locked profile from a sample of `values`: a mode string for
/// `encode` and the encoders that skips auto's per-block search.
#[pyfunction]
#[pyo3(signature = (values, precision = "f64", checkpoint_interval = 100))]
fn learn_profile(values: &Bound<'_, PyAny>, precision: &str, checkpoint_interval: i64) -> Result<String, PyErr> {
    let interval = parse_config("auto", checkpoint_interval)?.checkpoint_interval;
    let precision: SampleType = precision.parse().map_err(|e: gt::ConfigError| PyValueError::new_err(e.0))?;
    let profile = match precision {
        SampleType::I32 => gt::Profile::learn(&ints::<i32>(values, TS_MSG)?, interval),
        SampleType::I64 => gt::Profile::learn(&ints::<i64>(values, TS_MSG)?, interval),
        SampleType::F32 => {
            let v: Vec<f32> = floats(values, VAL_MSG)?.into_iter().map(to_f32).collect();
            gt::Profile::learn(&v, interval)
        }
        SampleType::F64 => gt::Profile::learn(&floats(values, VAL_MSG)?, interval),
    };
    Ok(profile.to_string())
}

/// Decode any stream to a list of ints or floats, as its header says.
#[pyfunction]
fn decode<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> Result<Bound<'py, PyList>, PyErr> {
    decoded_to_list(py, gt::decode_any(&bytes_of(data, "encoded")?).map_err(|e| decode_err(&e))?)
}

/// Read a stream header: `version`, `precision`, `mode`, `checkpoint_interval`, `count`.
#[pyfunction]
fn inspect<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> Result<Bound<'py, PyDict>, PyErr> {
    let (h, _) = gt::read_header(&bytes_of(data, "encoded")?).map_err(|e| decode_err(&e))?;
    let d = PyDict::new(py);
    d.set_item("version", gt::VERSION)?;
    d.set_item("precision", h.sample_type.name())?;
    d.set_item("mode", h.config.mode.name())?;
    d.set_item("checkpoint_interval", h.config.checkpoint_interval.get())?;
    d.set_item("count", h.count)?;
    Ok(d)
}

#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> Result<(), PyErr> {
    m.add("DecodeError", m.py().get_type::<DecodeError>())?;
    m.add_class::<TimestampsEncoder>()?;
    m.add_class::<TimestampsDecoder>()?;
    m.add_class::<ValuesEncoder>()?;
    m.add_class::<ValuesDecoder>()?;
    m.add_class::<PairsEncoder>()?;
    m.add_class::<PairsDecoder>()?;
    m.add_function(wrap_pyfunction!(estimate_ts_compression, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_val_compression, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(learn_profile, m)?)?;
    Ok(())
}
