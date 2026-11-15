"""Gorilla-style time series compression, implemented in Rust.

All logic lives in the ``gorilla-tsc`` Rust crate; this package only re-exports
its Python bindings.
"""

from gorilla_tsc_py._native import (
    DecodeError,
    PairsDecoder,
    PairsEncoder,
    TimestampsDecoder,
    TimestampsEncoder,
    ValuesDecoder,
    ValuesEncoder,
    decode,
    encode,
    estimate_ts_compression,
    estimate_val_compression,
    inspect,
)

__all__ = [
    "DecodeError",
    "PairsDecoder",
    "PairsEncoder",
    "TimestampsDecoder",
    "TimestampsEncoder",
    "ValuesDecoder",
    "ValuesEncoder",
    "decode",
    "encode",
    "estimate_ts_compression",
    "estimate_val_compression",
    "inspect",
]
