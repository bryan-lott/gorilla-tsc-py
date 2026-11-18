//! Binary adaptive range coder (the LZMA design), embedded in the bit stream.
//!
//! Each coded bit uses a [`Prob`] that adapts towards the bits it sees. The
//! coder's bytes are written as 8-bit fields of the surrounding bit stream, and
//! the decoder consumes exactly as many bytes as the encoder produced, so the
//! stream continues seamlessly after a coded section.

use crate::{BitReader, BitWriter, DecodeError};

/// Probability scale: probabilities are `p / 2^PROB_BITS`.
const PROB_BITS: u32 = 11;
/// Adaptation rate: larger is slower and steadier.
const ADAPT_SHIFT: u32 = 5;
/// Below this the range is renormalised by a byte.
const TOP: u32 = 1 << 24;

/// Adaptive probability that the next bit is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prob(u16);

impl Default for Prob {
    fn default() -> Self {
        Self(1 << (PROB_BITS - 1))
    }
}

impl Prob {
    fn update(&mut self, bit: bool) {
        if bit {
            self.0 -= self.0 >> ADAPT_SHIFT;
        } else {
            self.0 += ((1 << PROB_BITS) - self.0) >> ADAPT_SHIFT;
        }
    }

    fn bound(self, range: u32) -> u32 {
        (range >> PROB_BITS) * u32::from(self.0)
    }
}

/// Range encoder writing into a [`BitWriter`].
#[derive(Debug)]
pub struct RangeEncoder<'a> {
    out: &'a mut BitWriter,
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    /// LZMA's first output byte is always zero; it is not written.
    first: bool,
}

impl<'a> RangeEncoder<'a> {
    /// Start a coded section.
    pub fn new(out: &'a mut BitWriter) -> Self {
        Self { out, low: 0, range: u32::MAX, cache: 0, cache_size: 1, first: true }
    }

    /// Code `bit` with the adaptive probability `p`.
    pub fn bit(&mut self, p: &mut Prob, bit: bool) {
        let bound = p.bound(self.range);
        if bit {
            self.low += u64::from(bound);
            self.range -= bound;
        } else {
            self.range = bound;
        }
        p.update(bit);
        self.normalize();
    }

    /// Code the low `n` bits of `v` (MSB first) at probability one half.
    pub fn direct(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.range >>= 1;
            if (v >> i) & 1 == 1 {
                self.low += u64::from(self.range);
            }
            self.normalize();
        }
    }

    /// End the coded section.
    pub fn finish(mut self) {
        for _ in 0..5 {
            self.shift_low();
        }
    }

    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    #[allow(clippy::cast_possible_truncation)] // bytes are masked
    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            let mut byte = self.cache;
            loop {
                self.emit(byte.wrapping_add(carry));
                byte = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    fn emit(&mut self, byte: u8) {
        if self.first {
            debug_assert_eq!(byte, 0, "LZMA's first byte is always zero");
            self.first = false;
        } else {
            self.out.write(u64::from(byte), 8);
        }
    }
}

/// Range decoder reading from a [`BitReader`].
#[derive(Debug)]
pub struct RangeDecoder<'r, 'a> {
    input: &'r mut BitReader<'a>,
    code: u32,
    range: u32,
}

impl<'r, 'a> RangeDecoder<'r, 'a> {
    /// Start reading a coded section.
    ///
    /// # Errors
    /// On truncated input.
    #[allow(clippy::cast_possible_truncation)] // 8-bit fields
    pub fn new(input: &'r mut BitReader<'a>) -> Result<Self, DecodeError> {
        let mut code = 0u32;
        for _ in 0..4 {
            code = (code << 8) | input.read(8)? as u32;
        }
        Ok(Self { input, code, range: u32::MAX })
    }

    #[allow(clippy::cast_possible_truncation)] // 8-bit fields
    fn normalize(&mut self) -> Result<(), DecodeError> {
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.input.read(8)? as u32;
        }
        Ok(())
    }

    /// Decode a bit coded with `p` (which adapts as on the encoder side).
    ///
    /// # Errors
    /// On truncated input.
    pub fn bit(&mut self, p: &mut Prob) -> Result<bool, DecodeError> {
        let bound = p.bound(self.range);
        let bit = self.code >= bound;
        if bit {
            self.code -= bound;
            self.range -= bound;
        } else {
            self.range = bound;
        }
        p.update(bit);
        self.normalize()?;
        Ok(bit)
    }

    /// Decode `n` direct bits.
    ///
    /// # Errors
    /// On truncated input.
    pub fn direct(&mut self, n: u32) -> Result<u64, DecodeError> {
        let mut v = 0u64;
        for _ in 0..n {
            self.range >>= 1;
            let bit = self.code >= self.range;
            if bit {
                self.code -= self.range;
            }
            v = (v << 1) | u64::from(bit);
            self.normalize()?;
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Encode bits (each with its own context index) and direct fields, return bytes.
    fn encode(events: &[Event], contexts: usize) -> Vec<u8> {
        let mut w = BitWriter::new();
        let mut probs = vec![Prob::default(); contexts];
        let mut e = RangeEncoder::new(&mut w);
        for ev in events {
            match *ev {
                Event::Bit(c, b) => e.bit(&mut probs[c], b),
                Event::Direct(v, n) => e.direct(v, n),
            }
        }
        e.finish();
        w.write(0b101, 3); // trailing data after the section
        w.finish()
    }

    fn decode(bytes: &[u8], events: &[Event], contexts: usize) -> Vec<Event> {
        let mut r = BitReader::new(bytes);
        let mut probs = vec![Prob::default(); contexts];
        let mut out = Vec::new();
        {
            let mut d = RangeDecoder::new(&mut r).unwrap();
            for ev in events {
                out.push(match *ev {
                    Event::Bit(c, _) => Event::Bit(c, d.bit(&mut probs[c]).unwrap()),
                    Event::Direct(_, n) => Event::Direct(d.direct(n).unwrap(), n),
                });
            }
        }
        assert_eq!(r.read(3), Ok(0b101), "section must end exactly where the encoder stopped");
        out
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Event {
        Bit(usize, bool),
        Direct(u64, u32),
    }

    #[test]
    fn probability_adapts_towards_observed_bits() {
        let mut p = Prob::default();
        assert_eq!(p, Prob(1024));
        p.update(false);
        assert_eq!(p, Prob(1024 + (2048 - 1024) / 32));
        let mut q = Prob::default();
        q.update(true);
        assert_eq!(q, Prob(1024 - 1024 / 32));
        for _ in 0..1000 {
            p.update(false);
        }
        assert!(p.0 < 2048 && p.0 > 2000);
    }

    #[test]
    fn empty_section_round_trips() {
        assert_eq!(decode(&encode(&[], 1), &[], 1), Vec::<Event>::new());
    }

    #[test]
    fn empty_section_costs_four_bytes() {
        let mut w = BitWriter::new();
        RangeEncoder::new(&mut w).finish();
        assert_eq!(w.bit_len(), 32);
    }

    #[test]
    fn skewed_bits_compress_well() {
        let events: Vec<Event> = (0..10_000).map(|i| Event::Bit(0, i % 50 == 0)).collect();
        let bytes = encode(&events, 1);
        assert!(bytes.len() * 8 < 10_000 / 5, "{} bytes", bytes.len()); // < 0.2 bits per bit
        assert_eq!(decode(&bytes, &events, 1), events);
    }

    #[test]
    fn direct_bits_cost_about_one_bit_each() {
        let events: Vec<Event> = (0..1000u64).map(|i| Event::Direct(i * 2_654_435_761 % (1 << 20), 20)).collect();
        let bytes = encode(&events, 1);
        assert!(bytes.len() * 8 <= 20_000 + 64, "{} bytes", bytes.len());
        assert_eq!(decode(&bytes, &events, 1), events);
    }

    #[test]
    fn wide_direct_fields() {
        let events =
            [Event::Direct(u64::MAX, 64), Event::Direct(0, 64), Event::Direct(1 << 63, 64), Event::Direct(0, 0)];
        assert_eq!(decode(&encode(&events, 1), &events, 1), events);
    }

    /// Bytes emitted by `shift_low` from a given state, then by a carrying shift.
    fn shift_bytes(low: u64) -> Vec<u8> {
        let mut out = BitWriter::new();
        let mut e = RangeEncoder::new(&mut out);
        e.first = false;
        e.cache = 0x12;
        e.low = low;
        e.shift_low();
        // Now force a carry out of the pending bytes.
        e.low = 1 << 32;
        e.shift_low();
        out.finish()
    }

    #[test]
    fn top_byte_ff_is_deferred_until_the_carry_is_known() {
        // 0xFF in the top byte might still receive a carry, so the cached byte
        // must wait: the carry then lands in 0x12 and the 0xFF wraps to 0x00.
        assert_eq!(shift_bytes(0xFF00_0000), [0x13, 0x00]);
        assert_eq!(shift_bytes(0xFFFF_FFFF), [0x13, 0x00]);
        // Below 0xFF00_0000 no carry can reach the cached byte: flush it now.
        assert_eq!(shift_bytes(0xFEFF_FFFF), [0x12, 0xFF]);
    }

    #[test]
    fn carry_propagation_round_trips() {
        // Long runs of highly probable bits followed by improbable ones exercise carries.
        let mut events = Vec::new();
        for round in 0..50 {
            events.extend((0..300).map(|_| Event::Bit(0, false)));
            events.extend((0..=(round % 7)).map(|_| Event::Bit(0, true)));
            events.push(Event::Direct(0xFF_FFFF, 24));
        }
        assert_eq!(decode(&encode(&events, 1), &events, 1), events);
    }

    #[test]
    fn truncated_section_is_eof() {
        let events: Vec<Event> = (0..100).map(|i| Event::Bit(0, i % 3 == 0)).collect();
        let mut writer = BitWriter::new();
        let mut prob = Prob::default();
        let mut enc = RangeEncoder::new(&mut writer);
        for ev in &events {
            if let Event::Bit(_, bit) = *ev {
                enc.bit(&mut prob, bit);
            }
        }
        enc.finish();
        let bytes = writer.finish();
        let short = &bytes[..bytes.len() - 2];
        let mut reader = BitReader::new(short);
        let mut prob = Prob::default();
        let res: Result<Vec<bool>, DecodeError> = (|| {
            let mut dec = RangeDecoder::new(&mut reader)?;
            (0..events.len()).map(|_| dec.bit(&mut prob)).collect()
        })();
        assert_eq!(res, Err(DecodeError::UnexpectedEof));
        assert_eq!(RangeDecoder::new(&mut BitReader::new(&[1, 2, 3])).err(), Some(DecodeError::UnexpectedEof));
    }

    fn event() -> impl Strategy<Value = Event> {
        prop_oneof![
            (0usize..4, any::<bool>()).prop_map(|(c, b)| Event::Bit(c, b)),
            (any::<u64>(), 0u32..=64).prop_map(|(v, n)| Event::Direct(if n == 64 { v } else { v & ((1 << n) - 1) }, n)),
        ]
    }

    proptest! {
        #[test]
        fn round_trips(events in prop::collection::vec(event(), 0..400)) {
            prop_assert_eq!(decode(&encode(&events, 4), &events, 4), events);
        }
        #[test]
        fn biased_round_trips(bits in prop::collection::vec(prop::bool::weighted(0.97), 0..2000)) {
            let events: Vec<Event> = bits.into_iter().map(|b| Event::Bit(0, b)).collect();
            prop_assert_eq!(decode(&encode(&events, 1), &events, 1), events);
        }
    }
}
