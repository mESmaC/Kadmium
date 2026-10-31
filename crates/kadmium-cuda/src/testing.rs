//! Deterministic test data shared by the unit tests, the runtime's parity tests and the GPU
//! parity harness, so all three exercise the same inputs from the same seed.

/// SplitMix64: a tiny seeded generator with no dependencies.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in `[-1, 1)`.
    pub fn unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    /// A finite value of either sign with an exponent spread over `2^-20 ..= 2^20`, so sums
    /// that associate differently round differently.
    pub fn wide_f32(&mut self) -> f32 {
        let bits = self.next_u64();
        let sign = ((bits >> 63) as u32) << 31;
        let exponent = (127 - 20 + (bits >> 32) as u32 % 41) << 23;
        let mantissa = bits as u32 & 0x007f_ffff;
        f32::from_bits(sign | exponent | mantissa)
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }

    /// Activations shaped like a hidden state: mostly unit scale, a few outliers, and the odd
    /// all-zero block so the zero-scale path is exercised.
    pub fn activations(&mut self, len: usize) -> Vec<f32> {
        let mut values: Vec<f32> = (0..len)
            .map(|_| {
                let v = self.unit_f32();
                if self.next_u32().is_multiple_of(61) {
                    v * 40.0
                } else {
                    v
                }
            })
            .collect();
        if len >= 64 {
            let zero_block = (self.next_u32() as usize % (len / 32)) * 32;
            values[zero_block..zero_block + 32].fill(0.0);
        }
        values
    }
}
