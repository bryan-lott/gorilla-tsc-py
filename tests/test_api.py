"""Tests of the Python binding layer: argument handling, errors and metadata.

Bit-level format details are tested in the Rust crate; these check that the
bindings forward configuration faithfully and map errors to Python exceptions.
"""

import struct

import pytest

from gorilla_tsc_py import (
    DecodeError,
    encode,
    PairsDecoder,
    PairsEncoder,
    TimestampsDecoder,
    TimestampsEncoder,
    ValuesDecoder,
    ValuesEncoder,
    decode,
    estimate_ts_compression,
    estimate_val_compression,
    inspect,
)

MODES = ["xor", "dod", "xor_dod", "dod_xor", "delta", "decimal", "raw", "dictionary", "decimal_split", "auto"]


# --- configuration -----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"precision": "int16"}, "timestamp precision must be 'int32' or 'int64' (got 'int16')"),
        ({"precision": "f64"}, "timestamp precision must be 'int32' or 'int64' (got 'f64')"),
        ({"checkpoint_interval": 0}, "checkpoint interval must be between 1 and 4294967295 (got 0)"),
        ({"checkpoint_interval": 2**32}, "checkpoint interval must be between 1 and 4294967295 (got 4294967296)"),
        ({"mode": "rle"}, "mode must be one of 'xor', 'dod', 'xor_dod', 'dod_xor', 'delta', 'decimal', 'raw', 'dictionary', 'decimal_split', 'auto' (got 'rle')"),
    ],
)
def test_timestamps_encoder_rejects_bad_config(kwargs, msg):
    with pytest.raises(ValueError) as e:
        TimestampsEncoder(**kwargs)
    assert str(e.value) == msg


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"precision": "f16"}, "value precision must be 'f32' or 'f64' (got 'f16')"),
        ({"precision": "int64"}, "value precision must be 'f32' or 'f64' (got 'int64')"),
        ({"mode": "rle"}, "mode must be one of 'xor', 'dod', 'xor_dod', 'dod_xor', 'delta', 'decimal', 'raw', 'dictionary', 'decimal_split', 'auto' (got 'rle')"),
        ({"checkpoint_interval": -1}, "checkpoint interval must be between 1 and 4294967295 (got -1)"),
    ],
)
def test_values_encoder_rejects_bad_config(kwargs, msg):
    with pytest.raises(ValueError) as e:
        ValuesEncoder(**kwargs)
    assert str(e.value) == msg


def test_pairs_encoder_rejects_bad_config():
    with pytest.raises(ValueError, match="timestamp precision"):
        PairsEncoder(ts_precision="f32")
    with pytest.raises(ValueError, match="value precision"):
        PairsEncoder(val_precision="int32")
    with pytest.raises(ValueError, match="mode must be"):
        PairsEncoder(val_mode="nope")
    with pytest.raises(ValueError, match="mode must be"):
        PairsEncoder(ts_mode="nope")
    with pytest.raises(ValueError, match="checkpoint interval"):
        PairsEncoder(checkpoint_interval=0)


def test_result_metadata_reflects_config():
    r = TimestampsEncoder("int32", 7, "xor").encode_all([1, 2, 3])
    assert {k: r[k] for k in r if k != "encoded"} == {
        "nb_timestamps": 3, "precision": "int32", "mode": "xor", "checkpoint_interval": 7,
    }
    r = ValuesEncoder("f32", "dod_xor", 9).encode_all([1.0])
    assert {k: r[k] for k in r if k != "encoded"} == {
        "nb_values": 1, "precision": "f32", "mode": "dod_xor", "checkpoint_interval": 9,
    }


def test_defaults():
    assert inspect(TimestampsEncoder().encode_all([1])) == {
        "version": 1, "precision": "int64", "mode": "auto", "checkpoint_interval": 100, "count": 1,
    }
    assert inspect(ValuesEncoder().encode_all([1.0])) == {
        "version": 1, "precision": "f64", "mode": "auto", "checkpoint_interval": 100, "count": 1,
    }
    r = PairsEncoder().encode_all([(1, 1.0)])
    assert (r["ts_precision"], r["val_precision"], r["ts_mode"], r["val_mode"], r["checkpoint_interval"]) == (
        "int64", "f64", "auto", "auto", 100,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_header_records_value_config(mode, precision):
    h = inspect(ValuesEncoder(precision, mode, 3).encode_all([1.0, 2.0]))
    assert (h["precision"], h["mode"], h["checkpoint_interval"], h["count"]) == (precision, mode, 3, 2)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("precision", ["int32", "int64"])
def test_header_records_timestamp_config(mode, precision):
    h = inspect(TimestampsEncoder(precision, 4, mode).encode_all([5, 6, 7]))
    assert (h["precision"], h["mode"], h["checkpoint_interval"], h["count"]) == (precision, mode, 4, 3)


def test_pairs_forward_config_to_both_streams():
    r = PairsEncoder("int32", "f32", "xor_dod", 5, "dod_xor").encode_all([(2, 2.0), (1, 1.0)])
    assert inspect(r["encoded_ts"]) == {
        "version": 1, "precision": "int32", "mode": "dod_xor", "checkpoint_interval": 5, "count": 2,
    }
    assert inspect(r["encoded_val"]) == {
        "version": 1, "precision": "f32", "mode": "xor_dod", "checkpoint_interval": 5, "count": 2,
    }
    assert r["nb_pairs"] == 2
    assert PairsDecoder(r).decode_all() == [(1, 1.0), (2, 2.0)]


@pytest.mark.parametrize("ts_p", ["int32", "int64"])
@pytest.mark.parametrize("val_p", ["f32", "f64"])
def test_pairs_all_precisions_round_trip(ts_p, val_p):
    pairs = [(3, 1.5), (-1, -2.25), (3, 0.5)]
    out = PairsDecoder(PairsEncoder(ts_p, val_p).encode_all(pairs)).decode_all()
    assert out == [(-1, -2.25), (3, 1.5), (3, 0.5)]


# --- input types -------------------------------------------------------------


def test_non_int_timestamps_rejected():
    with pytest.raises(TypeError, match="^All timestamps must be ints$"):
        TimestampsEncoder().encode_all([1, 2.0])
    with pytest.raises(TypeError, match="^Timestamp must be int$"):
        TimestampsEncoder().encode_next(1.0)
    with pytest.raises(TypeError, match="^All timestamps must be ints$"):
        TimestampsEncoder("int32").encode_all(["1"])


def test_non_float_values_rejected():
    with pytest.raises(TypeError, match="^All values must be floats$"):
        ValuesEncoder().encode_all([1.0, 2])
    with pytest.raises(TypeError, match="^Value must be float$"):
        ValuesEncoder().encode_next(1)


@pytest.mark.parametrize("bad", [[(1, 2.0, 3)], [[1, 2.0]], [("1", 2.0)], [(1, 2)], [(1,)], [1]])
def test_malformed_pairs_rejected(bad):
    with pytest.raises(TypeError, match=r"^All pairs must be \(int, float\) tuples$"):
        PairsEncoder().encode_all(bad)


def test_out_of_range_timestamps_overflow():
    with pytest.raises(OverflowError, match="out of range for int32"):
        TimestampsEncoder("int32").encode_all([2**31])
    with pytest.raises(OverflowError, match="out of range for int32"):
        PairsEncoder("int32").encode_all([(2**31, 1.0)])
    with pytest.raises(OverflowError):
        TimestampsEncoder().encode_all([2**63])
    assert TimestampsDecoder(TimestampsEncoder("int32").encode_all([-(2**31), 2**31 - 1])).decode_all() == [
        -(2**31), 2**31 - 1,
    ]


def test_f32_values_are_rounded_to_f32():
    out = ValuesDecoder(ValuesEncoder("f32").encode_all([0.1])).decode_all()
    assert out == [struct.unpack("f", struct.pack("f", 0.1))[0]]


def test_accepts_any_iterable():
    assert decode(TimestampsEncoder().encode_all(range(5))) == [0, 1, 2, 3, 4]
    assert decode(ValuesEncoder().encode_all(x / 2 for x in range(3))) == [0.0, 0.5, 1.0]


# --- streaming ---------------------------------------------------------------


def test_streaming_matches_bulk():
    e = TimestampsEncoder(checkpoint_interval=2)
    assert all(e.encode_next(t) for t in [1, 5, 2, 8])
    assert e.finish() == TimestampsEncoder(checkpoint_interval=2).encode_all([1, 5, 2, 8])
    v = ValuesEncoder(mode="dod")
    for x in [1.0, 2.5]:
        assert v.encode_next(x) is True
    assert v.finish() == ValuesEncoder(mode="dod").encode_all([1.0, 2.5])


def test_encode_all_includes_streamed_items():
    e = ValuesEncoder()
    e.encode_next(1.0)
    r = e.encode_all([2.0])
    assert r["nb_values"] == 2
    assert decode(r) == [1.0, 2.0]


def test_encoder_cannot_be_reused_after_finishing():
    for e, item in [(TimestampsEncoder(), 1), (ValuesEncoder(), 1.0)]:
        e.finish()
        for call in (lambda: e.encode_next(item), lambda: e.encode_all([item]), e.finish):
            with pytest.raises(RuntimeError, match="^encoder already finished$"):
                call()


# --- decoding ----------------------------------------------------------------


def test_decoders_accept_bytes_or_result_dict():
    r = TimestampsEncoder().encode_all([1, 2])
    assert TimestampsDecoder(r["encoded"]).decode_all() == TimestampsDecoder(r).decode_all() == [1, 2]
    v = ValuesEncoder().encode_all([1.5])
    assert ValuesDecoder(v["encoded"]).decode_all() == [1.5]


def test_decoder_input_errors():
    with pytest.raises(ValueError, match="^missing key 'encoded'$"):
        TimestampsDecoder({})
    with pytest.raises(ValueError, match="^missing key 'encoded_val'$"):
        PairsDecoder({"encoded_ts": b""})
    with pytest.raises(TypeError, match="^expected bytes or an encoder result dict$"):
        ValuesDecoder(42)


def test_decode_error_is_a_value_error():
    assert issubclass(DecodeError, ValueError)


@pytest.mark.parametrize(
    "data, msg",
    [
        (b"", "unexpected end of stream"),
        (b"X", "not a gorilla-tsc stream (bad magic)"),
        (b"G\x09", "unsupported format version 9"),
        (b"G\x01\x40\x01\x00", "invalid header: reserved descriptor bits set"),
        (b"G\x01\x0f\x01\x00", "invalid header: unknown mode"),
    ],
)
def test_malformed_streams_raise_decode_error(data, msg):
    with pytest.raises(DecodeError) as e:
        decode(data)
    assert str(e.value) == msg
    with pytest.raises(DecodeError):
        inspect(data)


def test_truncated_and_trailing_streams_raise():
    enc = ValuesEncoder().encode_all([1.0, 2.0, 3.0])["encoded"]
    with pytest.raises(DecodeError, match="^unexpected end of stream$"):
        ValuesDecoder(enc[:-1]).decode_all()
    with pytest.raises(DecodeError, match="^corrupt stream: trailing data after last item$"):
        ValuesDecoder(enc + b"\x00").decode_all()


def test_decoders_check_stream_kind():
    ts = TimestampsEncoder().encode_all([1])
    vals = ValuesEncoder().encode_all([1.0])
    with pytest.raises(DecodeError, match="^expected a value \\(float\\) stream, found integers$"):
        ValuesDecoder(ts).decode_all()
    with pytest.raises(DecodeError, match="^expected a timestamp \\(integer\\) stream, found floats$"):
        TimestampsDecoder(vals).decode_all()


def test_pair_streams_of_different_lengths():
    data = {
        "encoded_ts": TimestampsEncoder().encode_all([1, 2])["encoded"],
        "encoded_val": ValuesEncoder().encode_all([1.0])["encoded"],
    }
    with pytest.raises(DecodeError, match="^pair streams differ in length$"):
        PairsDecoder(data).decode_all()


def test_decode_returns_native_python_types():
    assert all(type(x) is int for x in decode(TimestampsEncoder("int32").encode_all([1, 2])))
    assert all(type(x) is float for x in decode(ValuesEncoder("f32").encode_all([1.0])))


# --- estimates ---------------------------------------------------------------


def test_estimate_ts_compression():
    # payload: 64 bits + 9 for the first delta-of-delta + 8 x 1 bit = 81 bits -> 11 bytes
    assert estimate_ts_compression(list(range(1000, 1010)), mode="dod") == {
        "ratio": 80 / 11, "compressed_size": 11, "bits_per_item": 8.1,
    }


def test_estimates_default_to_auto():
    ts = list(range(1000, 1010))
    assert estimate_ts_compression(ts) == estimate_ts_compression(ts, mode="auto")
    vals = [1.0] * 9
    assert estimate_val_compression(vals) == estimate_val_compression(vals, mode="auto")


def test_estimate_ts_compression_options():
    assert estimate_ts_compression([5, 5], precision="int32", checkpoint_interval=1, mode="dod") == {
        "ratio": 1.0, "compressed_size": 8, "bits_per_item": 32.0,
    }
    xor = estimate_ts_compression(list(range(0, 300, 3)), mode="xor")
    dod = estimate_ts_compression(list(range(0, 300, 3)), mode="dod")
    assert xor["compressed_size"] > dod["compressed_size"]


def test_estimate_val_compression():
    assert estimate_val_compression([1.0] * 9, mode="xor") == {"ratio": 8.0, "compressed_size": 9, "bits_per_item": 8.0}
    assert estimate_val_compression([1.0, 1.0], mode="dod", precision="f32", checkpoint_interval=1) == {
        "ratio": 1.0, "compressed_size": 8, "bits_per_item": 32.0,
    }


def test_estimate_empty():
    assert estimate_ts_compression([]) == {"ratio": float("inf"), "compressed_size": 0, "bits_per_item": 0.0}
    assert estimate_val_compression([]) == {"ratio": float("inf"), "compressed_size": 0, "bits_per_item": 0.0}


def test_estimate_type_checks():
    with pytest.raises(TypeError):
        estimate_ts_compression([1.0])
    with pytest.raises(TypeError):
        estimate_val_compression([1])


# --- one-shot encode ---------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 7])
def test_encode_matches_encoder_classes(threads):
    vals = [x / 10 for x in range(2500)]
    assert encode(vals, threads=threads) == ValuesEncoder().encode_all(vals)["encoded"]
    ts = list(range(0, 25_000, 10))
    assert encode(ts, precision="int64", threads=threads) == TimestampsEncoder().encode_all(ts)["encoded"]


def test_encode_options():
    data = encode([1.5, 2.5], precision="f32", mode="decimal", checkpoint_interval=7)
    assert inspect(data) == {"version": 1, "precision": "f32", "mode": "decimal", "checkpoint_interval": 7, "count": 2}
    assert decode(data) == [1.5, 2.5]
    assert decode(encode([3, -4], precision="int32")) == [3, -4]


def test_encode_infers_nothing_and_validates():
    with pytest.raises(TypeError, match="^All values must be floats$"):
        encode([1, 2])  # default precision is f64
    with pytest.raises(TypeError, match="^All timestamps must be ints$"):
        encode([1.0], precision="int64")
    with pytest.raises(ValueError, match="^precision must be one of 'int32', 'int64', 'f32', 'f64' \\(got 'f16'\\)$"):
        encode([1.0], precision="f16")
    with pytest.raises(ValueError, match="^threads must be at least 1$"):
        encode([1.0], threads=0)


def test_decimal_and_auto_compress_decimal_data_far_better_than_xor():
    vals = [round(20 + (i * 37 % 23) / 10, 1) for i in range(5000)]
    xor = len(encode(vals, mode="xor"))
    assert len(encode(vals, mode="decimal")) * 4 < xor
    assert len(encode(vals)) * 4 < xor  # auto is the default
