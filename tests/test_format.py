"""Tests for the encoded format: sizes, field widths, validation and helpers.

Round-trip tests cannot see a codec that is correct but wasteful (a value put in a
bigger bucket than needed, an XOR window not reused, an extra padding byte). These
pin the exact bit cost of each encoding path.
"""

import math
import struct

import pytest

from gorilla_tsc_py import (
    PairsDecoder,
    PairsEncoder,
    TimestampsDecoder,
    TimestampsEncoder,
    ValuesDecoder,
    ValuesEncoder,
    estimate_ts_compression,
    estimate_val_compression,
)


def nbytes(bits: int) -> int:
    return math.ceil(bits / 8)


def f64(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def f32(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits))[0]


# --- bit stream layout -------------------------------------------------------


def test_full_timestamp_is_big_endian_64_bits():
    assert TimestampsEncoder().encode_all([1000])["encoded"] == (1000).to_bytes(8, "big")


def test_full_int32_timestamp_is_big_endian_32_bits():
    enc = TimestampsEncoder(precision="int32").encode_all([-2])["encoded"]
    assert enc == (-2 & 0xFFFFFFFF).to_bytes(4, "big")


def test_full_value_is_ieee754_big_endian():
    assert ValuesEncoder().encode_all([1.5])["encoded"] == struct.pack(">d", 1.5)
    assert ValuesEncoder(precision="f32").encode_all([1.5])["encoded"] == struct.pack(">f", 1.5)


def test_padding_is_zero_bits_and_only_to_the_next_byte():
    # 64-bit checkpoint + one '0' bit (dod == 0) -> 65 bits -> 9 bytes, 7 zero pad bits.
    enc = TimestampsEncoder().encode_all([1000, 1000])["encoded"]
    assert enc == (1000).to_bytes(8, "big") + b"\x00"
    # Byte-aligned stream gets no padding byte.
    assert len(TimestampsEncoder().encode_all([7])["encoded"]) == 8


def test_padding_bits_after_set_bits_are_zero():
    # 64 + '10' + 7-bit(-1 -> 0x7F) = 73 bits; last byte is 1 data bit + 7 zero bits.
    enc = TimestampsEncoder().encode_all([1000, 999])["encoded"]
    assert enc[8:] == bytes([0b10111111, 0b10000000])


# --- timestamp delta-of-delta buckets ----------------------------------------

N_DODS = 8


def ts_with_constant_dod(dod: int, n: int = N_DODS):
    ts, delta = [10**9], 0
    for _ in range(n):
        delta += dod
        ts.append(ts[-1] + delta)
    return ts


@pytest.mark.parametrize(
    "dod, cost_bits",
    [
        (0, 1),
        (-64, 9), (63, 9), (-1, 9), (1, 9),
        (-65, 12), (64, 12), (-256, 12), (255, 12),
        (-257, 16), (256, 16), (-2048, 16), (2047, 16),
        (-2049, 4 + 66), (2048, 4 + 66),
    ],
)
def test_timestamp_dod_bucket_costs(dod, cost_bits):
    ts = ts_with_constant_dod(dod)
    result = TimestampsEncoder().encode_all(ts)
    assert len(result["encoded"]) == nbytes(64 + N_DODS * cost_bits)
    assert TimestampsDecoder(result).decode_all() == ts


@pytest.mark.parametrize("dod, cost_bits", [(-2049, 4 + 34), (2048, 4 + 34), (-64, 9)])
def test_int32_timestamp_dod_escape_cost(dod, cost_bits):
    ts = ts_with_constant_dod(dod, n=4)
    result = TimestampsEncoder(precision="int32").encode_all(ts)
    assert len(result["encoded"]) == nbytes(32 + 4 * cost_bits)
    assert TimestampsDecoder(result).decode_all() == ts


def test_timestamp_checkpoint_resets_delta():
    # With interval 2 each pair is a fresh checkpoint: 64 bits + one dod of the raw delta.
    ts = [0, 100, 1000, 1100]
    result = TimestampsEncoder(checkpoint_interval=2).encode_all(ts)
    assert len(result["encoded"]) == nbytes(2 * (64 + 12))  # delta 100 -> 12-bit bucket
    assert TimestampsDecoder(result).decode_all() == ts


# --- value delta-of-delta (on packed bits) -----------------------------------

BASE = struct.unpack(">Q", struct.pack(">d", 1.0))[0]


def vals_with_packed_dod(first_delta: int, dod: int, n: int, to_float=f64, base=BASE):
    packed, delta = [base, base + first_delta], first_delta
    for _ in range(n):
        delta += dod
        packed.append(packed[-1] + delta)
    return [to_float(p) for p in packed]


@pytest.mark.parametrize(
    "dod, cost_bits",
    [(0, 1), (-1, 9), (63, 9), (-64, 9), (64, 2 + 66), (-65, 2 + 66)],
)
def test_value_dod_bucket_costs(dod, cost_bits):
    vals = vals_with_packed_dod(first_delta=5, dod=dod, n=N_DODS)
    result = ValuesEncoder(mode="dod").encode_all(vals)
    assert len(result["encoded"]) == nbytes(64 + 65 + N_DODS * cost_bits)
    assert ValuesDecoder(result).decode_all() == vals


def test_value_dod_f32_widths():
    base = struct.unpack(">I", struct.pack(">f", 1.0))[0]
    vals = vals_with_packed_dod(first_delta=-3, dod=1, n=4, to_float=f32, base=base)
    result = ValuesEncoder(mode="dod", precision="f32").encode_all(vals)
    assert len(result["encoded"]) == nbytes(32 + 33 + 4 * 9)
    assert ValuesDecoder(result).decode_all() == vals


def test_dod_xor_small_dods_use_short_bucket():
    # XORs 1, 2, 3, ... -> dod of XOR is 1 each step -> 9 bits each.
    packed = [BASE]
    for x in range(1, 6):
        packed.append(packed[-1] ^ x)
    vals = [f64(p) for p in packed]
    result = ValuesEncoder(mode="dod_xor").encode_all(vals)
    assert len(result["encoded"]) == nbytes(64 + 64 + 4 * 9)
    assert ValuesDecoder(result).decode_all() == vals


# --- XOR windows -------------------------------------------------------------


def test_xor_header_field_widths_f64():
    # XOR = 0b1 << 10: 53 leading zeros (capped to 31), 10 trailing zeros, len = 64-31-10 = 23.
    vals = [f64(BASE), f64(BASE ^ (1 << 10))]
    result = ValuesEncoder().encode_all(vals)
    assert len(result["encoded"]) == nbytes(64 + 2 + 5 + 6 + 23)
    assert ValuesDecoder(result).decode_all() == vals


def test_xor_header_field_widths_f32():
    base = struct.unpack(">I", struct.pack(">f", 1.0))[0]
    # XOR = 1 << 3: 28 leading zeros (capped to 15), len = 32-15-3 = 14.
    vals = [f32(base), f32(base ^ (1 << 3))]
    result = ValuesEncoder(precision="f32").encode_all(vals)
    assert len(result["encoded"]) == nbytes(32 + 2 + 4 + 5 + 14)
    assert ValuesDecoder(result).decode_all() == vals


def test_xor_with_32_or_more_leading_zeros_round_trips():
    for shift in (0, 1, 31):
        vals = [f64(BASE), f64(BASE ^ (1 << shift)), f64(BASE ^ (3 << shift))]
        assert ValuesDecoder(ValuesEncoder().encode_all(vals)).decode_all() == vals


def test_xor_reuses_window_when_it_fits():
    # First XOR sets window lz=31(cap) tz=4 len=29; following XORs fit it exactly or within.
    xs = [0b1111 << 4, 0b1 << 4, 0b1 << 7]
    packed = [BASE]
    for x in xs:
        packed.append(packed[-1] ^ x)
    vals = [f64(p) for p in packed]
    result = ValuesEncoder().encode_all(vals)
    header = 2 + 5 + 6 + 29
    reuse = 2 + 29
    assert len(result["encoded"]) == nbytes(64 + header + 2 * reuse)
    assert ValuesDecoder(result).decode_all() == vals


def test_xor_same_window_bounds_are_reused():
    # Identical lz and tz must count as "fits" (>=, not >).
    packed = [BASE, BASE ^ (1 << 4), BASE]
    vals = [f64(p) for p in packed]
    result = ValuesEncoder().encode_all(vals)
    assert len(result["encoded"]) == nbytes(64 + (2 + 5 + 6 + 29) + (2 + 29))


def test_xor_dod_mode_round_trips_negative_deltas():
    vals = [f64(BASE + d) for d in (0, -10, -25, -25, 40, 0)]
    assert ValuesDecoder(ValuesEncoder(mode="xor_dod").encode_all(vals)).decode_all() == vals


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize("kwargs, msg", [
    ({"precision": "int16"}, "Precision must be 'int32' or 'int64'"),
    ({"checkpoint_interval": 0}, "Checkpoint interval must be >= 1"),
])
def test_timestamps_encoder_rejects_bad_config(kwargs, msg):
    with pytest.raises(ValueError, match=f"^{msg}$"):
        TimestampsEncoder(**kwargs)


@pytest.mark.parametrize("kwargs, msg", [
    ({"precision": "f16"}, "Precision must be 'f32' or 'f64'"),
    ({"mode": "rle"}, "Mode must be 'xor', 'dod', 'xor_dod', or 'dod_xor'"),
    ({"checkpoint_interval": 0}, "Checkpoint interval must be >= 1"),
])
def test_values_encoder_rejects_bad_config(kwargs, msg):
    with pytest.raises(ValueError, match=f"^{msg}$"):
        ValuesEncoder(**kwargs)


def test_checkpoint_interval_of_one_is_allowed():
    TimestampsEncoder(checkpoint_interval=1)
    ValuesEncoder(checkpoint_interval=1)


def test_non_int_timestamps_rejected():
    with pytest.raises(TypeError, match="^All timestamps must be ints$"):
        TimestampsEncoder().encode_all([1, 2.0])
    with pytest.raises(TypeError, match="^Timestamp must be int$"):
        TimestampsEncoder().encode_next(1.0)


def test_non_float_values_rejected():
    with pytest.raises(TypeError, match="^All values must be floats$"):
        ValuesEncoder().encode_all([1.0, 2])


@pytest.mark.parametrize("bad", [[(1, 2.0, 3)], [[1, 2.0]], [("1", 2.0)], [(1, 2)], [(1,)]])
def test_malformed_pairs_rejected(bad):
    with pytest.raises(TypeError, match=r"^All pairs must be \(int, float\) tuples$"):
        PairsEncoder().encode_all(bad)


def test_encode_next_reports_success():
    assert TimestampsEncoder().encode_next(1) is True
    assert ValuesEncoder().encode_next(1.0) is True


def test_corrupt_xor_stream_raises():
    # checkpoint value, then '1' (nonzero) '0' (reuse window) with no window set.
    data = {"encoded": struct.pack(">d", 1.0) + b"\x80", "nb_values": 2,
            "precision": "f64", "mode": "xor", "checkpoint_interval": 100}
    with pytest.raises(ValueError, match="^Corrupt stream: XOR window reused before being set$"):
        ValuesDecoder(data).decode_all()


# --- pairs -------------------------------------------------------------------


def test_pairs_defaults():
    result = PairsEncoder().encode_all([(1, 1.0)])
    assert result["checkpoint_interval"] == 100
    assert (result["ts_precision"], result["val_precision"], result["val_mode"]) == ("int64", "f64", "xor")


def test_pairs_forward_checkpoint_interval_to_both_streams():
    pairs = [(i * 10, float(i)) for i in range(7)]
    result = PairsEncoder(checkpoint_interval=3).encode_all(pairs)
    ts, vals = zip(*pairs)
    assert result["encoded_ts"] == TimestampsEncoder(checkpoint_interval=3).encode_all(list(ts))["encoded"]
    assert result["encoded_val"] == ValuesEncoder(checkpoint_interval=3).encode_all(list(vals))["encoded"]
    assert PairsDecoder(result).decode_all() == pairs


def test_default_checkpoint_interval_is_100():
    assert TimestampsEncoder().encode_all([1])["checkpoint_interval"] == 100
    assert ValuesEncoder().encode_all([1.0])["checkpoint_interval"] == 100


# --- estimates ---------------------------------------------------------------


def test_estimate_ts_compression():
    data = list(range(1000, 1010))  # 64 bits + '10'+7 for first dod, then 8 x '0' = 81 bits
    stats = estimate_ts_compression(data)
    assert stats == {"ratio": 80 / 11, "compressed_size": 11, "bits_per_item": 8.8}


def test_estimate_ts_compression_int32_and_interval():
    stats = estimate_ts_compression([5, 5], precision="int32", checkpoint_interval=1)
    assert stats == {"ratio": 1.0, "compressed_size": 8, "bits_per_item": 32.0}


def test_estimate_ts_samples_first_1000():
    assert estimate_ts_compression(list(range(5000)))["compressed_size"] == \
        estimate_ts_compression(list(range(1000)))["compressed_size"]


def test_estimate_ts_empty():
    assert estimate_ts_compression([]) == {"ratio": float("inf"), "compressed_size": 0, "bits_per_item": 0}


def test_estimate_val_compression():
    stats = estimate_val_compression([1.0] * 9)  # 64 + 8 x '0' = 72 bits = 9 bytes
    assert stats == {"ratio": 8.0, "compressed_size": 9, "bits_per_item": 8.0}


def test_estimate_val_compression_options():
    stats = estimate_val_compression([1.0, 1.0], mode="dod", precision="f32", checkpoint_interval=1)
    assert stats == {"ratio": 1.0, "compressed_size": 8, "bits_per_item": 32.0}


def test_estimate_val_samples_first_1000():
    data = [float(i) for i in range(3000)]
    assert estimate_val_compression(data)["compressed_size"] == \
        estimate_val_compression(data[:1000])["compressed_size"]


def test_estimate_val_empty():
    assert estimate_val_compression([]) == {"ratio": float("inf"), "compressed_size": 0, "bits_per_item": 0}


# --- debug logging -----------------------------------------------------------


@pytest.mark.parametrize("mode", ["xor", "dod", "xor_dod", "dod_xor"])
def test_debug_mode_logs_and_still_round_trips(mode, caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger="gorilla_tsc_py.gorilla_compression")
    vals = [1.0, -2.5, 3.25, 3.25, 1e10]
    ts = [0, 10, 5, 5, 10**12]
    v = ValuesDecoder(ValuesEncoder(mode=mode, checkpoint_interval=3, debug=True).encode_all(vals), debug=True)
    t = TimestampsDecoder(TimestampsEncoder(checkpoint_interval=3, debug=True).encode_all(ts), debug=True)
    assert v.decode_all() == vals
    assert t.decode_all() == ts
    assert "Checkpoint" in caplog.text


def test_estimates_default_to_checkpoint_every_100():
    ts = list(range(0, 201 * 7, 7))  # 201 items: 3 checkpoints at 100, 2 at 101
    assert estimate_ts_compression(ts) == estimate_ts_compression(ts, checkpoint_interval=100)
    assert estimate_ts_compression(ts) != estimate_ts_compression(ts, checkpoint_interval=101)
    vals = [1.0] * 201
    assert estimate_val_compression(vals) == estimate_val_compression(vals, checkpoint_interval=100)
    assert estimate_val_compression(vals) != estimate_val_compression(vals, checkpoint_interval=101)


def test_xor_dod_f32_header_uses_33_bit_widths():
    base = struct.unpack(">I", struct.pack(">f", 1.0))[0]
    # deltas 1, 2 -> XOR of 33-bit deltas = 0b11: lz=31 capped to 31 (5 bits), len=33-31-0=2 (6 bits)
    vals = [f32(base), f32(base + 1), f32(base + 3)]
    result = ValuesEncoder(mode="xor_dod", precision="f32").encode_all(vals)
    assert len(result["encoded"]) == nbytes(32 + 33 + 2 + 5 + 6 + 2)
    assert ValuesDecoder(result).decode_all() == vals
