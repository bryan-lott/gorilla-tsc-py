//! MSB-first bit packing.

use crate::DecodeError;

/// Appends bit fields to a byte buffer, most significant bit first.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    /// Bits held in `acc` (always < 8 between calls).
    pending: u32,
}

impl BitWriter {
    /// Empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the low `n` bits of `value` (`n <= 64`). `value` must fit in `n` bits.
    #[allow(clippy::cast_possible_truncation)] // each pushed byte is masked to 8 bits
    pub fn write(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64, "field wider than 64 bits");
        debug_assert!(n == 64 || value >> n == 0, "value {value:#x} does not fit in {n} bits");
        if n == 0 {
            return;
        }
        let mut acc = (u128::from(self.acc) << n) | u128::from(value);
        let mut pending = self.pending + n;
        while pending >= 8 {
            pending -= 8;
            self.bytes.push((acc >> pending) as u8);
        }
        acc &= (1u128 << pending) - 1;
        self.acc = acc as u64;
        self.pending = pending;
    }

    /// Append everything written to `other`.
    pub fn append(&mut self, other: &Self) {
        for &b in &other.bytes {
            self.write(u64::from(b), 8);
        }
        self.write(other.acc, other.pending);
    }

    /// Number of bits written so far.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.pending as usize
    }

    /// Pad the final byte with zero bits and return the buffer.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // acc holds < 8 bits
    pub fn finish(mut self) -> Vec<u8> {
        if self.pending > 0 {
            self.bytes.push((self.acc << (8 - self.pending)) as u8);
        }
        self.bytes
    }
}

/// Reads bit fields written by [`BitWriter`].
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Reader over `bytes`, starting at bit 0.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Read an `n`-bit field (`n <= 64`).
    ///
    /// # Errors
    /// [`DecodeError::UnexpectedEof`] if fewer than `n` bits remain.
    #[allow(clippy::cast_possible_truncation)] // result is masked to n <= 64 bits
    pub fn read(&mut self, n: u32) -> Result<u64, DecodeError> {
        debug_assert!(n <= 64, "field wider than 64 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.pos + n as usize > self.bytes.len() * 8 {
            return Err(DecodeError::UnexpectedEof);
        }
        let start = self.pos / 8;
        let offset = (self.pos % 8) as u32;
        // offset + n <= 71 bits, so at most 9 bytes are needed.
        let mut window = [0u8; 16];
        let take = (self.bytes.len() - start).min(9);
        window[..take].copy_from_slice(&self.bytes[start..start + take]);
        let v = u128::from_be_bytes(window) >> (128 - offset - n);
        self.pos += n as usize;
        Ok((v & ((1u128 << n) - 1)) as u64)
    }

    /// Bits left unread.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.bytes.len() * 8 - self.pos
    }

    /// A `Vec` capacity for `n` decoded items that the remaining input can
    /// back: at most one item per remaining bit. A header may claim any
    /// count, so reserving `n` outright lets a few bytes demand gigabytes.
    /// (Items can cost less than a bit, so this is only a hint; vectors
    /// still grow as items are actually decoded.)
    #[must_use]
    pub fn capacity_hint(&self, n: usize) -> usize {
        n.min(self.remaining())
    }

    /// Read a single bit as a bool.
    ///
    /// # Errors
    /// [`DecodeError::UnexpectedEof`] at end of stream.
    pub fn read_bit(&mut self) -> Result<bool, DecodeError> {
        Ok(self.read(1)? == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn capacity_hint_is_bounded_by_remaining_bits() {
        let bytes = [0u8; 3];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.capacity_hint(5), 5);
        assert_eq!(r.capacity_hint(1 << 40), 24);
        r.read(20).unwrap();
        assert_eq!(r.capacity_hint(usize::MAX), 4);
        assert_eq!(r.capacity_hint(0), 0);
    }

    #[test]
    fn msb_first_layout() {
        let mut w = BitWriter::new();
        w.write(0b1, 1);
        w.write(0b01, 2);
        w.write(0xABCD, 16);
        assert_eq!(w.bit_len(), 19);
        // 1 01 1010101111001101 -> 1011 0101 0111 1001 101(0 0000)
        assert_eq!(w.finish(), vec![0b1011_0101, 0b0111_1001, 0b1010_0000]);
    }

    #[test]
    fn full_width_fields() {
        let mut w = BitWriter::new();
        w.write(1, 1);
        w.write(u64::MAX, 64);
        w.write(0, 64);
        w.write(0x8000_0000_0000_0001, 64);
        let bytes = w.finish();
        assert_eq!(bytes.len(), 25); // 193 bits
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(1), Ok(1));
        assert_eq!(r.read(64), Ok(u64::MAX));
        assert_eq!(r.read(64), Ok(0));
        assert_eq!(r.read(64), Ok(0x8000_0000_0000_0001));
    }

    #[test]
    fn append_concatenates_bit_streams() {
        let mut a = BitWriter::new();
        a.write(0b101, 3);
        let mut b = BitWriter::new();
        b.write(0xABCD, 16);
        b.write(0b11, 2);
        a.append(&b);
        assert_eq!(a.bit_len(), 21);
        let mut expected = BitWriter::new();
        expected.write(0b101, 3);
        expected.write(0xABCD, 16);
        expected.write(0b11, 2);
        assert_eq!(a.finish(), expected.finish());
        let mut empty = BitWriter::new();
        empty.append(&BitWriter::new());
        assert_eq!(empty.bit_len(), 0);
    }

    #[test]
    fn zero_width_is_a_no_op() {
        let mut w = BitWriter::new();
        w.write(0, 0);
        assert_eq!(w.bit_len(), 0);
        assert_eq!(w.finish(), Vec::<u8>::new());
        let mut r = BitReader::new(&[]);
        assert_eq!(r.read(0), Ok(0));
    }

    #[test]
    fn aligned_stream_gets_no_padding_byte() {
        let mut w = BitWriter::new();
        w.write(0xAB, 8);
        assert_eq!(w.finish(), vec![0xAB]);
    }

    #[test]
    fn reading_past_the_end_is_an_error() {
        let mut r = BitReader::new(&[0xFF]);
        assert_eq!(r.read(4), Ok(0xF));
        assert_eq!(r.read(5), Err(DecodeError::UnexpectedEof));
        let mut r = BitReader::new(&[0xFF]);
        assert_eq!(r.read(8), Ok(0xFF));
        assert_eq!(r.read_bit(), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn read_bit_values() {
        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.read_bit(), Ok(true));
        assert_eq!(r.read_bit(), Ok(false));
    }

    proptest! {
        #[test]
        fn fields_round_trip(fields in prop::collection::vec((any::<u64>(), 0u32..=64), 0..200)) {
            let fields: Vec<(u64, u32)> = fields
                .into_iter()
                .map(|(v, n)| (if n == 64 { v } else { v & ((1u64 << n) - 1) }, n))
                .collect();
            let mut w = BitWriter::new();
            for &(v, n) in &fields { w.write(v, n); }
            let total: u32 = fields.iter().map(|f| f.1).sum();
            prop_assert_eq!(w.bit_len(), total as usize);
            let bytes = w.finish();
            prop_assert_eq!(bytes.len(), total.div_ceil(8) as usize);
            let mut r = BitReader::new(&bytes);
            for &(v, n) in &fields { prop_assert_eq!(r.read(n), Ok(v)); }
            // Only zero padding remains.
            let rest = u32::try_from(bytes.len() * 8).unwrap() - total;
            prop_assert_eq!(r.read(rest), Ok(0));
            prop_assert_eq!(r.read(1), Err(DecodeError::UnexpectedEof));
        }
    }
}
