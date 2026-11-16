"""Focused regression tests for bugs found in the codec.

Each test pins one bug; they were written failing before the fix.
"""

import pytest
from hypothesis import given, strategies as st



# --- timestamps ------------------------------------------------------------

from gorilla_tsc_py import (  # noqa: E402
    TimestampsEncoder,
    TimestampsDecoder,
    ValuesEncoder,
    ValuesDecoder,
    PairsEncoder,
    PairsDecoder,
)

INT_RANGES = {"int32": (-(2**31), 2**31 - 1), "int64": (-(2**63), 2**63 - 1)}


def ts_round_trip(ts, precision="int64", checkpoint_interval=100):
    enc = TimestampsEncoder(precision=precision, checkpoint_interval=checkpoint_interval)
    return TimestampsDecoder(enc.encode_all(ts)).decode_all()


@pytest.mark.parametrize(
    "ts",
    [
        [0, 2**31],  # delta-of-delta needs > 32 bits
        [0, 2**40, 0],
        [0, 2**63 - 1, -(2**63)],  # extreme swing
    ],
)
def test_int64_large_delta_of_delta(ts):
    assert ts_round_trip(ts) == ts


@pytest.mark.parametrize("ts", [[0, 2**20], [0, 2**31 - 1, -(2**31)]])
def test_int32_large_delta_of_delta(ts):
    assert ts_round_trip(ts, precision="int32") == ts


@pytest.mark.parametrize("precision", ["int32", "int64"])
def test_negative_timestamps(precision):
    ts = [-5, -4, 0, 7]
    assert ts_round_trip(ts, precision=precision, checkpoint_interval=2) == ts


@pytest.mark.parametrize("precision", ["int32", "int64"])
@given(data=st.data())
def test_timestamps_round_trip_full_range(precision, data):
    lo, hi = INT_RANGES[precision]
    ts = data.draw(st.lists(st.integers(lo, hi), max_size=50))
    ci = data.draw(st.integers(1, 10))
    assert ts_round_trip(ts, precision, ci) == ts


@pytest.mark.parametrize("dod", [-2048, -2047, -256, -255, -64, -63, 63, 64, 255, 256, 2047, 2048, 2049])
def test_delta_of_delta_bucket_boundaries(dod):
    ts = [1000, 1000 + dod]  # first delta-of-delta == dod
    assert ts_round_trip(ts) == ts


# --- values ----------------------------------------------------------------

MODES = ["xor", "dod", "xor_dod", "dod_xor", "delta", "decimal", "raw", "auto"]
PRECISIONS = {"f32": 32, "f64": 64}


def val_round_trip(vals, mode="xor", precision="f64", checkpoint_interval=100):
    enc = ValuesEncoder(precision=precision, mode=mode, checkpoint_interval=checkpoint_interval)
    return ValuesDecoder(enc.encode_all(vals)).decode_all()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "vals",
    [
        [0.0, 1.0],  # first XOR after a checkpoint has no window to reuse
        [1.0, 0.0, 1.0],  # delta does not fit in 32 bits
        [1.0, 3.0, 1.0, -7.5, 1e300, -1e-300, 0.0],
        [1.0, 1.5, 1.25, 1.125],  # XOR window reuse with different trailing zeros
        [5e-324, -5e-324, 1.7976931348623157e308],  # leading zeros > 31, 64 meaningful bits
    ],
)
def test_values_known_round_trips(mode, vals):
    assert val_round_trip(vals, mode=mode) == vals


@pytest.mark.parametrize("mode", MODES)
def test_f32_xor_window_fits_field_widths(mode):
    vals = [1.0, 1.5, 2.0, -3.25, 9.99994610111476e-41, 3.4028234663852886e38]  # all exact in f32
    assert val_round_trip(vals, mode=mode, precision="f32") == vals


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("precision", list(PRECISIONS))
@given(data=st.data())
def test_values_round_trip_property(mode, precision, data):
    floats = st.floats(allow_nan=False, width=PRECISIONS[precision])
    vals = data.draw(st.lists(floats, max_size=50))
    ci = data.draw(st.integers(1, 10))
    out = val_round_trip(vals, mode, precision, ci)
    # Compare bit patterns so -0.0 vs 0.0 is not glossed over.
    pack = "f" if precision == "f32" else "d"
    import struct
    assert [struct.pack(pack, v) for v in out] == [struct.pack(pack, v) for v in vals]


# --- pairs -----------------------------------------------------------------


def test_pairs_with_duplicate_timestamps_keep_insertion_order():
    pairs = [(5, 2.0), (1, 9.0), (5, 1.0)]
    out = PairsDecoder(PairsEncoder().encode_all(pairs)).decode_all()
    assert out == [(1, 9.0), (5, 2.0), (5, 1.0)]
