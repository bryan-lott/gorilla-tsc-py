"""
Gorilla Time Series Compression Library

This library implements a variant of the Gorilla compression algorithm for time series data,
with delta-of-delta encoding for timestamps and configurable XOR or delta-of-delta for values.
Checkpoints are inserted every n items for error recovery and random access.

Key Features:
- Configurable precision for timestamps (int32/int64) and values (f32/f64).
- Choice between XOR (better for irregular floats) and DoD (simpler for trends).
- Automatic sorting of timestamps in pairs.
- Convenience functions to estimate compression on sample data.

Why XOR vs. DoD for Values:
- XOR: Exploits bit-level similarities in floats (e.g., shared mantissa bits). Best for noisy,
  irregular data like sensor readings. Achieves higher compression (5-10x) but more complex.
- DoD: Arithmetic differencing. Simpler, best for predictable trends or integer-like data.
  May compress worse (2-7x) for bit-noisy floats.

Usage:
    from gorilla_tsc_py import TimestampsEncoder, ValuesEncoder, estimate_compression_ratio

    # Encode timestamps
    encoder = TimestampsEncoder(precision='int64', checkpoint_interval=100)
    result = encoder.encode_all([1000, 1001, 1002])

    # Estimate on sample
    stats = estimate_compression_ratio([1.0, 2.0, 3.0], 'xor', 'f64', 100)
"""

import struct
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)


def zigzag_encode(value: int) -> int:
    """Encode signed int to unsigned using zig-zag."""
    return (value << 1) ^ (value >> 63) if value < 0 else value << 1


def zigzag_decode(value: int) -> int:
    """Decode unsigned to signed using zig-zag."""
    return -(value >> 1) if value & 1 else value >> 1


class BitStream:
    """Handles variable-bit reading/writing to/from bytes."""

    def __init__(self):
        self.buffer = bytearray()
        self.bit_pos = 0

    def write_bits(self, value: int, num_bits: int):
        """Write num_bits of value to the stream."""
        if num_bits == 0:
            return
        # Extend buffer if needed
        while len(self.buffer) * 8 < self.bit_pos + num_bits:
            self.buffer.append(0)
        for i in range(num_bits):
            bit = (value >> (num_bits - 1 - i)) & 1
            byte_idx = self.bit_pos // 8
            bit_idx = self.bit_pos % 8
            if bit:
                self.buffer[byte_idx] |= 1 << (7 - bit_idx)
            self.bit_pos += 1

    def read_bits(self, num_bits: int) -> int:
        """Read num_bits from the stream."""
        if num_bits == 0:
            return 0
        value = 0
        for i in range(num_bits):
            byte_idx = self.bit_pos // 8
            bit_idx = self.bit_pos % 8
            bit = (self.buffer[byte_idx] >> (7 - bit_idx)) & 1
            value = (value << 1) | bit
            self.bit_pos += 1
        return value

    def to_bytes(self) -> bytes:
        """Pad to byte boundary and return bytes."""
        # Pad remaining bits in last byte with zeros
        if self.bit_pos % 8 != 0:
            pad_bits = 8 - (self.bit_pos % 8)
            self.write_bits(0, pad_bits)
        return bytes(self.buffer)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Create BitStream from bytes."""
        bs = cls()
        bs.buffer = bytearray(data)
        return bs


class TimestampsEncoder:
    """Encodes timestamps using delta-of-delta with checkpoints."""

    def __init__(
        self,
        precision: str = "int64",
        checkpoint_interval: int = 100,
        debug: bool = False,
    ):
        """Initialize encoder.

        Args:
            precision: 'int32' or 'int64' for timestamp bit width.
            checkpoint_interval: Full timestamp every n items (must be >=1).
            debug: If True, print intermediate steps.

        Raises:
            ValueError: If precision or checkpoint_interval is invalid.
        """
        if precision not in ("int32", "int64"):
            raise ValueError("Precision must be 'int32' or 'int64'")
        if checkpoint_interval < 1:
            raise ValueError("Checkpoint interval must be >= 1")
        self.precision = precision
        self.checkpoint_interval = checkpoint_interval
        self.bitstream = BitStream()
        self.prev_ts = None
        self.prev_delta = None
        self.count = 0
        self.debug = debug

    def _write_full_timestamp(self, ts: int):
        """Write full timestamp."""
        if self.precision == "int32":
            self.bitstream.write_bits(ts & 0xFFFFFFFF, 32)
        else:
            self.bitstream.write_bits(ts & 0xFFFFFFFFFFFFFFFF, 64)

    def _write_delta_of_delta(self, dod: int):
        """Write delta-of-delta with variable bits."""
        if dod == 0:
            self.bitstream.write_bits(0b0, 1)
        elif -63 <= dod <= 64:
            self.bitstream.write_bits(0b10, 2)
            self.bitstream.write_bits(dod & 0x7F, 7)  # 7 bits signed
        elif -255 <= dod <= 256:
            self.bitstream.write_bits(0b110, 3)
            self.bitstream.write_bits(dod & 0x1FF, 9)
        elif -2047 <= dod <= 2048:
            self.bitstream.write_bits(0b1110, 4)
            self.bitstream.write_bits(dod & 0xFFF, 12)
        else:
            self.bitstream.write_bits(0b1111, 4)
            max_bits = 32 if self.precision == "int64" else 16
            self.bitstream.write_bits(dod & ((1 << max_bits) - 1), max_bits)

    def encode_next(self, ts: int) -> bool:
        if not isinstance(ts, int):
            raise TypeError("Timestamp must be int")
        if self.debug:
            logger.debug(f"Encoding timestamp {self.count}: {ts}")
        if self.count % self.checkpoint_interval == 0:
            if self.debug:
                bits = 64 if self.precision == "int64" else 32
                logger.debug(
                    f"  Checkpoint: Writing full timestamp ({bits} bits) - no compression"
                )
            self._write_full_timestamp(ts)
            self.prev_ts = ts
            self.prev_delta = None
        else:
            if self.prev_ts is None:
                return False
            delta = ts - self.prev_ts
            dod = delta - (self.prev_delta if self.prev_delta is not None else 0)
            if self.debug:
                bits = 64 if self.precision == "int64" else 32
                delta_bin = (
                    format(abs(delta), f"0{bits}b")
                    if delta >= 0
                    else format((1 << bits) + delta, f"0{bits}b")
                )
                dod_bin = (
                    format(abs(dod), f"0{bits}b")
                    if dod >= 0
                    else format((1 << bits) + dod, f"0{bits}b")
                )
                logger.debug(
                    f"  Delta: {delta} (binary: {delta_bin}), DoD: {dod} (binary: {dod_bin}) - variable bits stored"
                )
            self._write_delta_of_delta(dod)
            self.prev_delta = delta
            self.prev_ts = ts
        self.count += 1
        return True

    def encode_all(self, timestamps: List[int]) -> Dict:
        """Encode all timestamps at once.

        Args:
            timestamps: List of int timestamps.

        Returns:
            Dict with 'encoded' bytes, 'nb_timestamps', 'precision', 'checkpoint_interval'.

        Raises:
            TypeError: If timestamps are not all ints.
        """
        if not all(isinstance(t, int) for t in timestamps):
            raise TypeError("All timestamps must be ints")
        """Encode all timestamps."""
        for ts in timestamps:
            self.encode_next(ts)
        return {
            "encoded": self.bitstream.to_bytes(),
            "nb_timestamps": len(timestamps),
            "precision": self.precision,
            "checkpoint_interval": self.checkpoint_interval,
        }


class TimestampsDecoder:
    """Decodes timestamps."""

    def __init__(self, data: Dict, debug: bool = False):
        self.bitstream = BitStream.from_bytes(data["encoded"])
        self.precision = data["precision"]
        self.checkpoint_interval = data["checkpoint_interval"]
        self.count = data["nb_timestamps"]
        self.debug = debug

    def _read_full_timestamp(self) -> int:
        if self.precision == "int32":
            return self.bitstream.read_bits(32)
        else:
            return self.bitstream.read_bits(64)

    def _read_delta_of_delta(self) -> int:
        prefix = self.bitstream.read_bits(1)
        if prefix == 0:
            return 0
        prefix2 = self.bitstream.read_bits(1)
        if prefix2 == 0:  # 10
            return self._signed_bits(7)
        prefix3 = self.bitstream.read_bits(1)
        if prefix3 == 0:  # 110
            return self._signed_bits(9)
        prefix4 = self.bitstream.read_bits(1)
        if prefix4 == 0:  # 1110
            return self._signed_bits(12)
        else:  # 1111
            max_bits = 32 if self.precision == "int64" else 16
            return self._signed_bits(max_bits)

    def _signed_bits(self, bits: int) -> int:
        val = self.bitstream.read_bits(bits)
        if val & (1 << (bits - 1)):  # Negative
            val -= 1 << bits
        return val

    def decode_all(self) -> List[int]:
        if self.debug:
            logger.debug(
                f"Decoding {self.count} timestamps, precision: {self.precision}"
            )
        bits = 64 if self.precision == "int64" else 32
        timestamps = []
        prev_ts = None
        prev_delta = None
        for i in range(self.count):
            if self.debug:
                logger.debug(f"Decoding timestamp {i}:")
            if i % self.checkpoint_interval == 0:
                if self.debug:
                    logger.debug(f"  Checkpoint: Reading full timestamp ({bits} bits)")
                ts = self._read_full_timestamp()
                if self.debug:
                    ts_bin = format(ts, f"0{bits}b")
                    logger.debug(
                        f"  Full timestamp: {ts} (full binary: {ts_bin}) - read all {bits} bits"
                    )
                prev_ts = ts
                prev_delta = None
            else:
                if prev_ts is None:
                    raise ValueError("Invalid state")
                dod = self._read_delta_of_delta()
                delta = (prev_delta if prev_delta is not None else 0) + dod
                ts = prev_ts + delta
                if self.debug:
                    dod_bin = (
                        format(abs(dod), f"0{bits}b")
                        if dod >= 0
                        else format((1 << bits) + dod, f"0{bits}b")
                    )
                    delta_bin = (
                        format(abs(delta), f"0{bits}b")
                        if delta >= 0
                        else format((1 << bits) + delta, f"0{bits}b")
                    )
                    ts_bin = format(ts, f"0{bits}b")
                    logger.debug(
                        f"  DoD: {dod} (meaningful: {dod_bin}) - variable bits read, accumulated delta: {delta} ({delta_bin}), timestamp: {ts} ({ts_bin})"
                    )
                prev_delta = delta
                prev_ts = ts
            timestamps.append(ts)
        return timestamps


class ValuesEncoder:
    """Encodes values using XOR or DoD with checkpoints."""

    def __init__(
        self,
        precision: str = "f64",
        mode: str = "xor",
        checkpoint_interval: int = 100,
        debug: bool = False,
    ):
        if precision not in ("f32", "f64"):
            raise ValueError("Precision must be 'f32' or 'f64'")
        if mode not in ("xor", "dod", "xor_dod", "dod_xor"):
            raise ValueError("Mode must be 'xor', 'dod', 'xor_dod', or 'dod_xor'")
        if checkpoint_interval < 1:
            raise ValueError("Checkpoint interval must be >= 1")
        self.precision = precision
        self.mode = mode
        self.checkpoint_interval = checkpoint_interval
        self.bitstream = BitStream()
        self.prev_val = None
        self.prev_delta = None
        self.prev_xor = None  # For stacked modes
        self.prev_lz = 0
        self.prev_tz = 0
        self.prev_len = 1  # dummy
        self.count = 0
        self.debug = debug

    def _pack_float(self, val: float) -> int:
        if self.precision == "f32":
            return struct.unpack(">I", struct.pack(">f", val))[0]
        else:
            return struct.unpack(">Q", struct.pack(">d", val))[0]

    def _write_full_value(self, val: float):
        packed = self._pack_float(val)
        bits = 32 if self.precision == "f32" else 64
        self.bitstream.write_bits(packed, bits)

    def _write_xor(self, xor_val: int):
        if xor_val == 0:
            self.bitstream.write_bits(0, 1)
            return
        self.bitstream.write_bits(1, 1)
        # Find leading/trailing zeros
        leading_zeros = 0
        while leading_zeros < 64 and (xor_val & (1 << (63 - leading_zeros))) == 0:
            leading_zeros += 1
        trailing_zeros = 0
        while (
            trailing_zeros < 64 - leading_zeros
            and (xor_val & (1 << trailing_zeros)) == 0
        ):
            trailing_zeros += 1
        meaningful_bits = 64 - leading_zeros - trailing_zeros
        if (
            self.prev_lz is not None
            and self.prev_tz is not None
            and leading_zeros >= self.prev_lz
            and trailing_zeros >= self.prev_tz
        ):
            # Case a: reuse prev
            self.bitstream.write_bits(0, 1)
            meaningful = xor_val >> trailing_zeros
            self.bitstream.write_bits(meaningful, self.prev_len)
        else:
            # Case b
            self.bitstream.write_bits(1, 1)
            lz_bits = 5 if self.precision == "f64" else 4
            len_bits = 6 if self.precision == "f64" else 5
            self.bitstream.write_bits(leading_zeros, lz_bits)
            self.bitstream.write_bits(meaningful_bits, len_bits)
            meaningful = xor_val >> trailing_zeros
            self.bitstream.write_bits(meaningful, meaningful_bits)
        self.prev_lz = leading_zeros
        self.prev_tz = trailing_zeros
        self.prev_len = meaningful_bits

    def _write_dod(self, dod: int):
        # Zig-zag encode the signed dod
        zz_dod = zigzag_encode(dod)
        if zz_dod == 0:
            self.bitstream.write_bits(0, 1)
        elif zz_dod <= 127:  # -63 to 64 mapped
            self.bitstream.write_bits(0b10, 2)
            self.bitstream.write_bits(zz_dod, 7)
        # Simplified, add more ranges if needed
        else:
            self.bitstream.write_bits(0b1111, 4)
            bits = 32 if self.precision == "f64" else 16
            self.bitstream.write_bits(zz_dod & ((1 << bits) - 1), bits)

    def encode_next(self, val: float) -> bool:
        if self.debug:
            logger.debug(f"Encoding value {self.count}: {val}")
        packed = self._pack_float(val)
        bits = 64 if self.precision == "f64" else 32
        if self.debug:
            packed_bin = format(packed, f"0{bits}b")
            logger.debug(f"  Packed float: {packed} (full binary: {packed_bin})")
        if self.count % self.checkpoint_interval == 0:
            if self.debug:
                logger.debug(
                    f"  Checkpoint: Writing full value ({bits} bits) - no compression"
                )
            self._write_full_value(val)
            self.prev_val = packed
            self.prev_delta = None
            # For XOR, reset to force case b next
            self.prev_lz = 0
        else:
            if self.prev_val is None:
                return False
            if self.mode == "xor":
                xor_val = packed ^ self.prev_val
                if self.debug:
                    xor_bin = format(xor_val, f"0{bits}b")
                    meaningful_start = xor_bin.lstrip("0") or "0"
                    logger.debug(
                        f"  XOR with prev: {xor_val} (full: {xor_bin}) - meaningful bits stored: {meaningful_start} (skips leading zeros)"
                    )
                self._write_xor(xor_val)
            elif self.mode == "dod":
                delta = packed - self.prev_val
                if self.debug:
                    delta_bin = (
                        format(abs(delta), f"0{bits}b")
                        if delta >= 0
                        else format((1 << bits) + delta, f"0{bits}b")
                    )
                    logger.debug(f"  Delta: {delta} (binary: {delta_bin})")
                if self.prev_delta is not None:
                    dod = delta - self.prev_delta
                    if self.debug:
                        dod_bin = (
                            format(abs(dod), f"0{bits}b")
                            if dod >= 0
                            else format((1 << bits) + dod, f"0{bits}b")
                        )
                        logger.debug(
                            f"  DoD: {dod} (binary: {dod_bin}) - variable bits stored"
                        )
                        zz_dod = zigzag_encode(dod)
                        zz_bin = format(zz_dod, f"0{bits}b").lstrip("0") or "0"
                        logger.debug(
                            f"  Zig-zag DoD: {zz_dod} - meaningful: {zz_bin} (no leading zeros)"
                        )
                    self._write_dod(dod)
                else:
                    # First delta after checkpoint, write as full delta zig-zagged
                    bits_delta = 32 if self.precision == "f64" else 16
                    zz_delta = zigzag_encode(delta)
                    if self.debug:
                        zz_bin = format(zz_delta, f"0{bits_delta}b").lstrip("0") or "0"
                        logger.debug(
                            f"  Full delta zig-zag: {zz_delta} - meaningful: {zz_bin} (no leading zeros)"
                        )
                    self.bitstream.write_bits(
                        zz_delta & ((1 << bits_delta) - 1), bits_delta
                    )
                self.prev_delta = delta
            elif self.mode == "xor_dod":
                # DoD first, then XOR on deltas
                delta = packed - self.prev_val
                if self.prev_delta is not None:
                    xor_val = delta ^ self.prev_delta
                    if self.debug:
                        xor_bin = format(xor_val, f"0{bits}b").lstrip("0") or "0"
                        logger.debug(
                            f"  XOR on DoD: {xor_val} - meaningful: {xor_bin} (skips leading zeros)"
                        )
                    self._write_xor(xor_val)
                else:
                    # First delta, write full
                    bits_delta = 32 if self.precision == "f64" else 16
                    zz_delta = zigzag_encode(delta)
                    if self.debug:
                        zz_bin = format(zz_delta, f"0{bits_delta}b").lstrip("0") or "0"
                        logger.debug(
                            f"  Full delta zig-zag: {zz_delta} - meaningful: {zz_bin}"
                        )
                    self.bitstream.write_bits(
                        zz_delta & ((1 << bits_delta) - 1), bits_delta
                    )
                self.prev_delta = delta
            elif self.mode == "dod_xor":
                # XOR first, then DoD on XORs
                xor_val = packed ^ self.prev_val
                if self.prev_xor is not None:
                    dod = xor_val - self.prev_xor
                    if self.debug:
                        dod_bin = (
                            format(abs(dod), f"0{bits}b")
                            if dod >= 0
                            else format((1 << bits) + dod, f"0{bits}b")
                        )
                        logger.debug(
                            f"  DoD on XOR: {dod} (binary: {dod_bin}) - variable bits stored"
                        )
                        zz_dod = zigzag_encode(dod)
                        zz_bin = format(zz_dod, f"0{bits}b").lstrip("0") or "0"
                        logger.debug(f"  Zig-zag DoD: {zz_dod} - meaningful: {zz_bin}")
                    self._write_dod(dod)
                else:
                    # First XOR, write full
                    if self.debug:
                        xor_bin = format(xor_val, f"0{bits}b").lstrip("0") or "0"
                        logger.debug(f"  Full XOR: {xor_val} - meaningful: {xor_bin}")
                    self.bitstream.write_bits(xor_val, bits)
                self.prev_xor = xor_val
            self.prev_val = packed
        self.count += 1
        return True

    def encode_all(self, values: List[float]) -> Dict:
        """Encode all values at once.

        Args:
            values: List of float values.

        Returns:
            Dict with 'encoded' bytes, 'nb_values', 'precision', 'mode', 'checkpoint_interval'.

        Raises:
            TypeError: If values are not all floats.
        """
        if not all(isinstance(v, float) for v in values):
            raise TypeError("All values must be floats")
        for val in values:
            self.encode_next(val)
        return {
            "encoded": self.bitstream.to_bytes(),
            "nb_values": len(values),
            "precision": self.precision,
            "mode": self.mode,
            "checkpoint_interval": self.checkpoint_interval,
        }


class ValuesDecoder:
    """Decodes values."""

    def __init__(self, data: Dict, debug: bool = False):
        self.bitstream = BitStream.from_bytes(data["encoded"])
        self.precision = data["precision"]
        self.mode = data["mode"]
        self.checkpoint_interval = data["checkpoint_interval"]
        self.count = data["nb_values"]
        self.prev_lz = 0
        self.prev_tz = 0
        self.prev_len = 1
        self.prev_xor = None  # For stacked modes
        self.debug = debug

    def _pack_float(self, val: float) -> int:
        if self.precision == "f32":
            return struct.unpack(">I", struct.pack(">f", val))[0]
        else:
            return struct.unpack(">Q", struct.pack(">d", val))[0]

    def _signed_bits(self, bits: int) -> int:
        val = self.bitstream.read_bits(bits)
        if val & (1 << (bits - 1)):  # Negative
            val -= 1 << bits
        return val

    def _unpack_float(self, packed: int) -> float:
        if self.precision == "f32":
            return struct.unpack(">f", struct.pack(">I", packed))[0]
        else:
            return struct.unpack(">d", struct.pack(">Q", packed))[0]

    def _read_full_value(self) -> float:
        bits = 32 if self.precision == "f32" else 64
        packed = self.bitstream.read_bits(bits)
        return self._unpack_float(packed)

    def _read_xor(self) -> int:
        if self.bitstream.read_bits(1) == 0:
            return 0
        control = self.bitstream.read_bits(1)
        if control == 0:
            # Reuse prev
            meaningful = self.bitstream.read_bits(self.prev_len)
            xor_val = meaningful << self.prev_tz
        else:
            lz_bits = 5 if self.precision == "f64" else 4
            len_bits = 6 if self.precision == "f64" else 5
            leading_zeros = self.bitstream.read_bits(lz_bits)
            meaningful_bits = self.bitstream.read_bits(len_bits)
            meaningful = self.bitstream.read_bits(meaningful_bits)
            xor_val = meaningful << leading_zeros
            self.prev_lz = leading_zeros
            self.prev_tz = 64 - leading_zeros - meaningful_bits
            self.prev_len = meaningful_bits
        return xor_val

    def _read_dod(self) -> int:
        prefix = self.bitstream.read_bits(1)
        if prefix == 0:
            zz_val = 0
        else:
            prefix2 = self.bitstream.read_bits(1)
            if prefix2 == 0:  # 10
                zz_val = self.bitstream.read_bits(7)
            # Simplified, assume no more prefixes for now
            else:
                bits = 32 if self.precision == "f64" else 16
                zz_val = self.bitstream.read_bits(bits)
        return zigzag_decode(zz_val)

    def decode_all(self) -> List[float]:
        if self.debug:
            logger.debug(
                f"Decoding {self.count} values, mode: {self.mode}, precision: {self.precision}"
            )
        bits = 64 if self.precision == "f64" else 32
        values = []
        prev_packed = None
        prev_delta = None
        for i in range(self.count):
            if self.debug:
                logger.debug(f"Decoding value {i}:")
            if i % self.checkpoint_interval == 0:
                if self.debug:
                    logger.debug(f"  Checkpoint: Reading full value ({bits} bits)")
                val = self._read_full_value()
                packed = self._pack_float(val)
                if self.debug:
                    packed_bin = format(packed, f"0{bits}b")
                    logger.debug(
                        f"  Full value: {val}, packed: {packed} (full binary: {packed_bin}) - read all {bits} bits"
                    )
                prev_packed = packed
                prev_delta = None
            else:
                if prev_packed is None:
                    raise ValueError("Invalid state")
                if self.mode == "xor":
                    xor_val = self._read_xor()
                    packed = prev_packed ^ xor_val
                    if self.debug:
                        xor_bin = format(xor_val, f"0{bits}b").lstrip("0") or "0"
                        packed_bin = format(packed, f"0{bits}b")
                        logger.debug(
                            f"  XOR val: {xor_val} (meaningful: {xor_bin}) - only meaningful bits read, resulting packed: {packed} ({packed_bin})"
                        )
                elif self.mode == "dod":
                    if prev_delta is None:
                        bits_delta = 32 if self.precision == "f64" else 16
                        zz_delta = self.bitstream.read_bits(bits_delta)
                        delta = zigzag_decode(zz_delta)
                        if self.debug:
                            zz_bin = format(zz_delta, f"0{bits_delta}b")
                            delta_bin = (
                                format(abs(delta), f"0{bits_delta}b")
                                if delta >= 0
                                else format(
                                    (1 << bits_delta) + delta, f"0{bits_delta}b"
                                )
                            )
                            logger.debug(
                                f"  Full delta zig-zag: {zz_delta} (meaningful: {zz_bin}) - only meaningful bits read, decoded delta: {delta} ({delta_bin})"
                            )
                    else:
                        dod = self._read_dod()
                        delta = prev_delta + dod
                        if self.debug:
                            dod_bin = (
                                format(abs(dod), f"0{bits}b")
                                if dod >= 0
                                else format((1 << bits) + dod, f"0{bits}b")
                            )
                            delta_bin = (
                                format(abs(delta), f"0{bits}b")
                                if delta >= 0
                                else format((1 << bits) + delta, f"0{bits}b")
                            )
                            logger.debug(
                                f"  DoD: {dod} (meaningful: {dod_bin}) - variable bits read, accumulated delta: {delta} ({delta_bin})"
                            )
                    packed = prev_packed + delta
                    prev_delta = delta
                    if self.debug:
                        packed_bin = format(packed, f"0{bits}b")
                        logger.debug(f"  Packed: {packed} ({packed_bin})")
                elif self.mode == "xor_dod":
                    # Read XOR on deltas
                    if prev_delta is None:
                        bits_delta = 32 if self.precision == "f64" else 16
                        zz_delta = self.bitstream.read_bits(bits_delta)
                        delta = zigzag_decode(zz_delta)
                        if self.debug:
                            zz_bin = (
                                format(zz_delta, f"0{bits_delta}b").lstrip("0") or "0"
                            )
                            delta_bin = (
                                format(abs(delta), f"0{bits_delta}b")
                                if delta >= 0
                                else format(
                                    (1 << bits_delta) + delta, f"0{bits_delta}b"
                                )
                            )
                            logger.debug(f"  Full delta: {delta} ({delta_bin})")
                    else:
                        xor_val = self._read_xor()
                        delta = prev_delta ^ xor_val
                        if self.debug:
                            xor_bin = format(xor_val, f"0{bits}b").lstrip("0") or "0"
                            delta_bin = (
                                format(abs(delta), f"0{bits}b")
                                if delta >= 0
                                else format((1 << bits) + delta, f"0{bits}b")
                            )
                            logger.debug(
                                f"  XOR val: {xor_val} (meaningful: {xor_bin}) - decoded delta: {delta} ({delta_bin})"
                            )
                    packed = prev_packed + delta
                    prev_delta = delta
                    if self.debug:
                        packed_bin = format(packed, f"0{bits}b")
                        logger.debug(f"  Packed: {packed} ({packed_bin})")
                elif self.mode == "dod_xor":
                    # Read DoD on XORs
                    if self.prev_xor is None:
                        xor_val = self.bitstream.read_bits(bits)
                        if self.debug:
                            xor_bin = format(xor_val, f"0{bits}b").lstrip("0") or "0"
                            logger.debug(
                                f"  Full XOR: {xor_val} (meaningful: {xor_bin})"
                            )
                    else:
                        dod = self._read_dod()
                        xor_val = self.prev_xor + dod
                        if self.debug:
                            dod_bin = (
                                format(abs(dod), f"0{bits}b")
                                if dod >= 0
                                else format((1 << bits) + dod, f"0{bits}b")
                            )
                            xor_bin = format(xor_val, f"0{bits}b").lstrip("0") or "0"
                            logger.debug(
                                f"  DoD: {dod} (meaningful: {dod_bin}) - accumulated XOR: {xor_val} ({xor_bin})"
                            )
                    packed = prev_packed ^ xor_val
                    self.prev_xor = xor_val
                    if self.debug:
                        packed_bin = format(packed, f"0{bits}b")
                        logger.debug(f"  Packed: {packed} ({packed_bin})")
                val = self._unpack_float(packed)
                prev_packed = packed
            values.append(val)
            if self.debug:
                logger.debug(f"  Final value: {val}")
        return values


class PairsEncoder:
    """Encodes timestamp-value pairs."""

    def __init__(
        self,
        ts_precision="int64",
        val_precision="f64",
        val_mode="xor",
        checkpoint_interval=100,
    ):
        self.ts_encoder = TimestampsEncoder(ts_precision, checkpoint_interval)
        self.val_encoder = ValuesEncoder(val_precision, val_mode, checkpoint_interval)

    def encode_all(self, pairs: List[tuple]) -> Dict:
        """Encode all timestamp-value pairs at once.

        Args:
            pairs: List of (int, float) tuples.

        Returns:
            Dict with encoded data and metadata.

        Raises:
            TypeError: If pairs are not tuples of (int, float).
        """
        if not all(
            isinstance(p, tuple)
            and len(p) == 2
            and isinstance(p[0], int)
            and isinstance(p[1], float)
            for p in pairs
        ):
            raise TypeError("All pairs must be (int, float) tuples")
        if not pairs:
            timestamps, values = [], []
        else:
            timestamps, values = zip(*sorted(pairs))  # Sort by ts
        ts_result = self.ts_encoder.encode_all(list(timestamps))
        val_result = self.val_encoder.encode_all(list(values))
        return {
            "encoded_ts": ts_result["encoded"],
            "encoded_val": val_result["encoded"],
            "nb_pairs": len(pairs),
            "nb_timestamps": ts_result["nb_timestamps"],
            "nb_values": val_result["nb_values"],
            "ts_precision": ts_result["precision"],
            "val_precision": val_result["precision"],
            "val_mode": val_result["mode"],
            "checkpoint_interval": ts_result["checkpoint_interval"],
        }


class PairsDecoder:
    """Decodes pairs."""

    def __init__(self, data: Dict):
        self.ts_decoder = TimestampsDecoder(
            {
                "encoded": data["encoded_ts"],
                "nb_timestamps": data["nb_timestamps"],
                "precision": data["ts_precision"],
                "checkpoint_interval": data["checkpoint_interval"],
            }
        )
        self.val_decoder = ValuesDecoder(
            {
                "encoded": data["encoded_val"],
                "nb_values": data["nb_values"],
                "precision": data["val_precision"],
                "mode": data["val_mode"],
                "checkpoint_interval": data["checkpoint_interval"],
            }
        )
        self.ts_decoder.count = data["nb_pairs"]
        self.val_decoder.count = data["nb_pairs"]

    def decode_all(self) -> List[tuple]:
        timestamps = self.ts_decoder.decode_all()
        values = self.val_decoder.decode_all()
        return list(zip(timestamps, values))


def estimate_ts_compression(
    data: List[int], precision: str = "int64", checkpoint_interval: int = 100
) -> Dict[str, Union[float, int]]:
    """Estimate compression ratio, size, bits per item for timestamps."""
    encoder = TimestampsEncoder(precision, checkpoint_interval)
    sample = data[:1000]
    result = encoder.encode_all(sample)
    compressed_size = len(result["encoded"])
    uncompressed_size = len(sample) * (4 if precision == "int32" else 8)
    ratio = uncompressed_size / compressed_size if compressed_size > 0 else float("inf")
    bits_per_item = (compressed_size * 8) / len(sample) if sample else 0
    return {
        "ratio": ratio,
        "compressed_size": compressed_size,
        "bits_per_item": bits_per_item,
    }


def estimate_val_compression(
    data: List[float],
    mode: str = "xor",
    precision: str = "f64",
    checkpoint_interval: int = 100,
) -> Dict[str, Union[float, int]]:
    """Estimate compression ratio, size, bits per item for values."""
    encoder = ValuesEncoder(precision, mode, checkpoint_interval)
    sample = data[:1000]
    result = encoder.encode_all(sample)
    compressed_size = len(result["encoded"])
    uncompressed_size = len(sample) * (4 if precision == "f32" else 8)
    ratio = uncompressed_size / compressed_size if compressed_size > 0 else float("inf")
    bits_per_item = (compressed_size * 8) / len(sample) if sample else 0
    return {
        "ratio": ratio,
        "compressed_size": compressed_size,
        "bits_per_item": bits_per_item,
    }
