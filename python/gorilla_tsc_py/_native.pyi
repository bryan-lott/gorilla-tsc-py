from collections.abc import Iterable
from typing import Literal, TypedDict

IntPrecision = Literal["int32", "int64"]
FloatPrecision = Literal["f32", "f64"]
Mode = Literal["xor", "dod", "xor_dod", "dod_xor"]

class DecodeError(ValueError): ...

class TimestampsResult(TypedDict):
    encoded: bytes
    nb_timestamps: int
    precision: IntPrecision
    mode: Mode
    checkpoint_interval: int

class ValuesResult(TypedDict):
    encoded: bytes
    nb_values: int
    precision: FloatPrecision
    mode: Mode
    checkpoint_interval: int

class PairsResult(TypedDict):
    encoded_ts: bytes
    encoded_val: bytes
    nb_pairs: int
    ts_precision: IntPrecision
    val_precision: FloatPrecision
    ts_mode: Mode
    val_mode: Mode
    checkpoint_interval: int

class Estimate(TypedDict):
    ratio: float
    compressed_size: int
    bits_per_item: float

class Header(TypedDict):
    version: int
    precision: IntPrecision | FloatPrecision
    mode: Mode
    checkpoint_interval: int
    count: int

class TimestampsEncoder:
    def __init__(self, precision: IntPrecision = "int64", checkpoint_interval: int = 100, mode: Mode = "dod") -> None: ...
    def encode_next(self, ts: int) -> bool: ...
    def encode_all(self, timestamps: Iterable[int]) -> TimestampsResult: ...
    def finish(self) -> TimestampsResult: ...

class TimestampsDecoder:
    def __init__(self, data: bytes | TimestampsResult) -> None: ...
    def decode_all(self) -> list[int]: ...

class ValuesEncoder:
    def __init__(self, precision: FloatPrecision = "f64", mode: Mode = "xor", checkpoint_interval: int = 100) -> None: ...
    def encode_next(self, value: float) -> bool: ...
    def encode_all(self, values: Iterable[float]) -> ValuesResult: ...
    def finish(self) -> ValuesResult: ...

class ValuesDecoder:
    def __init__(self, data: bytes | ValuesResult) -> None: ...
    def decode_all(self) -> list[float]: ...

class PairsEncoder:
    def __init__(
        self,
        ts_precision: IntPrecision = "int64",
        val_precision: FloatPrecision = "f64",
        val_mode: Mode = "xor",
        checkpoint_interval: int = 100,
        ts_mode: Mode = "dod",
    ) -> None: ...
    def encode_all(self, pairs: Iterable[tuple[int, float]]) -> PairsResult: ...

class PairsDecoder:
    def __init__(self, data: PairsResult) -> None: ...
    def decode_all(self) -> list[tuple[int, float]]: ...

def estimate_ts_compression(
    data: Iterable[int], precision: IntPrecision = "int64", checkpoint_interval: int = 100, mode: Mode = "dod"
) -> Estimate: ...
def estimate_val_compression(
    data: Iterable[float], mode: Mode = "xor", precision: FloatPrecision = "f64", checkpoint_interval: int = 100
) -> Estimate: ...
def decode(data: bytes | TimestampsResult | ValuesResult) -> list[int] | list[float]: ...
def inspect(data: bytes | TimestampsResult | ValuesResult) -> Header: ...
