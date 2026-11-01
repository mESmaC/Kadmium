//! Arithmetic for the native decoder, under the RepOps conventions.
//!
//! - Weights keep the precision they were stored in. A BF16 or F16 element is widened to `f32`
//!   when it is multiplied, and that widening is exact: every BF16 and every F16 value is an
//!   `f32` value.
//! - Every dot product is the RepOps product-then-tree: each product rounded once in `f32`, then
//!   the canonical pairwise tree of [`crate::repops::tree_sum`]. The leaves are computed as the
//!   tree reads them instead of being collected first; the tree is the same, so the bits are.
//! - RoPE angles and the inverse frequencies use `f64` exp, ln, sin and cos written here on IEEE
//!   basic operations only, like the `f32` functions in `repops`, so a platform `libm` cannot
//!   change a bit. The `f32` sine in `repops` is accurate only for `|x|` below about 6,400; a
//!   position times a frequency reaches the model's full context, so these reduce in `f64` and
//!   stay accurate for every angle below `2^20`, the decoder's position bound.

use crate::repops;

/// The conventions this module adds to RepOps, in words.
pub const NATIVE_DECODER_CONVENTIONS: &[&str] = &[
    "weights: stored F32, F16 or BF16 bits kept; each element widened exactly to f32 at use",
    "linear: per output feature, RepOps product-then-tree dot over the input, then + bias",
    "rmsnorm: f64 tree sum of exact squares, f64 IEEE sqrt, normalized value rounded to f32, times weight",
    "rope: inverse frequency via f64 exp(-(2i/d) ln theta) rounded to f32; angle position * frequency exact in f64; f64 sin/cos rounded to f32; rotate-half layout",
    "attention: f32 RepOps dot times 1/sqrt(head_dim), RepOps f32 exp softmax with tree sums, value mix as one tree per dimension",
    "mlp: RepOps silu(gate) * up, then down",
];

/// Exact widening of an IEEE binary16 value to `f32`.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    let out = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            // Subnormal: fraction * 2^-24. Normalise so the leading one reaches bit 10.
            let mut biased = 113u32;
            let mut f = fraction;
            while f & 0x400 == 0 {
                f <<= 1;
                biased -= 1;
            }
            sign | (biased << 23) | ((f & 0x3ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(out)
}

/// Exact widening of a bfloat16 value to `f32`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The RepOps tree over `n` leaves, with leaf `j` computed on demand by `leaf(j)`. Equal bit
/// for bit to `repops::tree_sum` of the collected leaves.
pub fn lazy_tree_sum<F: Fn(usize) -> f32>(n: usize, leaf: &F) -> f32 {
    if n == 0 {
        return 0.0;
    }
    lazy_node(leaf, n, 0, n.next_power_of_two())
}

fn lazy_node<F: Fn(usize) -> f32>(leaf: &F, n: usize, lo: usize, size: usize) -> f32 {
    if size == 1 {
        return leaf(lo);
    }
    if size == 8 && lo + 8 <= n {
        let a = leaf(lo) + leaf(lo + 1);
        let b = leaf(lo + 2) + leaf(lo + 3);
        let c = leaf(lo + 4) + leaf(lo + 5);
        let d = leaf(lo + 6) + leaf(lo + 7);
        return (a + b) + (c + d);
    }
    let half = size / 2;
    let left = lazy_node(leaf, n, lo, half);
    let mid = lo + half;
    if mid >= n {
        left
    } else {
        left + lazy_node(leaf, n, mid, half)
    }
}

/// Weights in the precision they were stored in.
#[derive(Debug, Clone)]
pub enum WeightData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Bf16(Vec<u16>),
}

impl WeightData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) | Self::Bf16(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The stored dtype's name, as SafeTensors writes it.
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::F32(_) => "F32",
            Self::F16(_) => "F16",
            Self::Bf16(_) => "BF16",
        }
    }

    /// Bytes held in memory, which equal the bytes stored on disk.
    pub fn stored_bytes(&self) -> usize {
        match self {
            Self::F32(values) => values.len() * 4,
            Self::F16(values) | Self::Bf16(values) => values.len() * 2,
        }
    }

    /// Every element widened exactly to `f32`.
    pub fn widen(&self) -> Vec<f32> {
        match self {
            Self::F32(values) => values.clone(),
            Self::F16(values) => values.iter().map(|&b| f16_to_f32(b)).collect(),
            Self::Bf16(values) => values.iter().map(|&b| bf16_to_f32(b)).collect(),
        }
    }
}

/// A row-major `rows x cols` weight matrix, PyTorch `Linear` layout `[out_features, in_features]`.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: WeightData,
}

impl Matrix {
    /// Row `row` widened exactly to `f32`. The caller guarantees `row < rows`.
    pub fn row_f32(&self, row: usize) -> Vec<f32> {
        let start = row * self.cols;
        let end = start + self.cols;
        match &self.data {
            WeightData::F32(values) => values[start..end].to_vec(),
            WeightData::F16(values) => values[start..end].iter().map(|&b| f16_to_f32(b)).collect(),
            WeightData::Bf16(values) => {
                values[start..end].iter().map(|&b| bf16_to_f32(b)).collect()
            }
        }
    }

    /// RepOps dot of row `row` with `x`, `x.len() == cols`.
    pub fn row_dot(&self, row: usize, x: &[f32]) -> f32 {
        let start = row * self.cols;
        let end = start + self.cols;
        match &self.data {
            WeightData::F32(values) => {
                let w = &values[start..end];
                lazy_tree_sum(self.cols, &|j| w[j] * x[j])
            }
            WeightData::F16(values) => {
                let w = &values[start..end];
                lazy_tree_sum(self.cols, &|j| f16_to_f32(w[j]) * x[j])
            }
            WeightData::Bf16(values) => {
                let w = &values[start..end];
                lazy_tree_sum(self.cols, &|j| bf16_to_f32(w[j]) * x[j])
            }
        }
    }
}

/// Work below this many multiply-adds runs on the calling thread.
const PARALLEL_WORK: usize = 1 << 15;

/// `input` holds `rows` vectors of length `weight.cols`; returns `rows` vectors of length
/// `weight.rows`, each output feature the RepOps dot plus its bias. Output features are split
/// across threads; each feature is computed by one thread in one fixed order, so the thread
/// count never changes a bit.
pub fn linear(weight: &Matrix, bias: Option<&[f32]>, input: &[f32], rows: usize) -> Vec<f32> {
    let features = weight.rows;
    let cols = weight.cols;
    let feature = |f: usize, out: &mut [f32]| {
        for (r, slot) in out.iter_mut().enumerate() {
            let x = &input[r * cols..(r + 1) * cols];
            let mut value = weight.row_dot(f, x);
            if let Some(bias) = bias {
                value += bias[f];
            }
            *slot = value;
        }
    };
    // Feature-major scratch: feature f's outputs for every row are contiguous.
    let mut by_feature = vec![0.0f32; features * rows];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let work = features.saturating_mul(cols).saturating_mul(rows);
    if threads <= 1 || work < PARALLEL_WORK || features < 2 || rows == 0 {
        for (f, out) in by_feature.chunks_mut(rows.max(1)).enumerate() {
            feature(f, out);
        }
    } else {
        let per_thread = features.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, chunk) in by_feature.chunks_mut(per_thread * rows).enumerate() {
                let feature = &feature;
                scope.spawn(move || {
                    for (offset, out) in chunk.chunks_mut(rows).enumerate() {
                        feature(chunk_index * per_thread + offset, out);
                    }
                });
            }
        });
    }
    let mut out = vec![0.0f32; rows * features];
    for f in 0..features {
        for r in 0..rows {
            out[r * features + f] = by_feature[f * rows + r];
        }
    }
    out
}

/// RMSNorm of one vector: `weight * round_f32(x / sqrt(mean(x^2) + eps))`, the mean an `f64`
/// tree sum of exact squares.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f64, out: &mut [f32]) {
    let mean = repops::tree_sum_squares_f64(x) / x.len() as f64;
    let inverse = 1.0 / (mean + eps).sqrt();
    for ((slot, &value), &w) in out.iter_mut().zip(x).zip(weight) {
        *slot = w * ((f64::from(value) * inverse) as f32);
    }
}

// ================================================================================================
// Deterministic f64 transcendentals for RoPE
// ================================================================================================

fn pow2_f64(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

fn ldexp_f64(x: f64, k: i32) -> f64 {
    if k > 1023 {
        (x * pow2_f64(1023)) * pow2_f64((k - 1023).min(1023))
    } else if k < -1022 {
        (x * pow2_f64(-1022)) * pow2_f64((k + 1022).max(-1022))
    } else {
        x * pow2_f64(k)
    }
}

const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);

/// `e^x` in `f64`: `x = k ln2 + r`, `|r| <= ln2 / 2`, a degree-13 Taylor polynomial in Horner
/// order, scaled by `2^k`. Basic IEEE operations only.
pub fn exp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x < -745.133_219_101_941_1 {
        return 0.0;
    }
    let k = (x * std::f64::consts::LOG2_E).round();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut p = 1.0 / 6_227_020_800.0;
    for divisor in [
        479_001_600.0,
        39_916_800.0,
        3_628_800.0,
        362_880.0,
        40_320.0,
        5_040.0,
        720.0,
        120.0,
        24.0,
        6.0,
        2.0,
        1.0,
        1.0,
    ] {
        p = 1.0 / divisor + r * p;
    }
    ldexp_f64(p, k as i32)
}

/// Natural logarithm of a positive finite `f64`: `m 2^k` with `m` in `[sqrt(1/2), sqrt(2))`,
/// `ln m = 2 atanh(s)`, `s = (m - 1) / (m + 1)`, as an odd series to `s^25`.
pub fn ln_f64(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    let mut bits = x.to_bits();
    let mut k: i32 = 0;
    if bits < 0x0010_0000_0000_0000 {
        bits = (x * pow2_f64(54)).to_bits();
        k -= 54;
    }
    k += ((bits >> 52) as i32) - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let mut series = 1.0 / 25.0;
    for odd in [
        23.0, 21.0, 19.0, 17.0, 15.0, 13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0,
    ] {
        series = 1.0 / odd + z * series;
    }
    let kf = f64::from(k);
    kf * LN2_HI + (kf * LN2_LO + 2.0 * s * series)
}

// fdlibm's pi/2 in 33-bit parts and the tail: `n * PIO2_k` is exact while `|n| < 2^20`.
const PIO2_1: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
const PIO2_2: f64 = f64::from_bits(0x3dd0_b461_1a60_0000);
const PIO2_3: f64 = f64::from_bits(0x3ba3_198a_2e00_0000);
const PIO2_3T: f64 = f64::from_bits(0x397b_839a_2520_49c1);

/// The largest angle magnitude [`sin_cos_f64`] reduces accurately.
pub const MAX_ANGLE: f64 = 1_048_576.0;

fn sin_kernel(r: f64) -> f64 {
    let z = r * r;
    let mut p = 1.0 / 51_090_942_171_709_440_000.0;
    for (divisor, sign) in [
        (121_645_100_408_832_000.0, -1.0),
        (355_687_428_096_000.0, 1.0),
        (1_307_674_368_000.0, -1.0),
        (6_227_020_800.0, 1.0),
        (39_916_800.0, -1.0),
        (362_880.0, 1.0),
        (5_040.0, -1.0),
        (120.0, 1.0),
        (6.0, -1.0),
        (1.0, 1.0),
    ] {
        p = sign / divisor + z * p;
    }
    r * p
}

fn cos_kernel(r: f64) -> f64 {
    let z = r * r;
    let mut p = 1.0 / 2_432_902_008_176_640_000.0;
    for (divisor, sign) in [
        (6_402_373_705_728_000.0, -1.0),
        (20_922_789_888_000.0, 1.0),
        (87_178_291_200.0, -1.0),
        (479_001_600.0, 1.0),
        (3_628_800.0, -1.0),
        (40_320.0, 1.0),
        (720.0, -1.0),
        (24.0, 1.0),
        (2.0, -1.0),
        (1.0, 1.0),
    ] {
        p = sign / divisor + z * p;
    }
    p
}

/// Sine and cosine of `x`, `|x| <= MAX_ANGLE`, on IEEE basic operations: Cody-Waite reduction
/// by `pi/2` in four parts, then Taylor kernels to degree 21 on `[-pi/4, pi/4]`.
pub fn sin_cos_f64(x: f64) -> (f64, f64) {
    let n = (x * std::f64::consts::FRAC_2_PI).round();
    let r = (((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3) - n * PIO2_3T;
    let (s, c) = (sin_kernel(r), cos_kernel(r));
    match (n as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_widening_is_exact_for_every_value() {
        for bits in 0..=u16::MAX {
            let sign = if bits >> 15 == 1 { -1.0f64 } else { 1.0 };
            let exponent = i32::from((bits >> 10) & 0x1f);
            let fraction = f64::from(bits & 0x3ff);
            let widened = f16_to_f32(bits);
            if exponent == 0x1f {
                if fraction == 0.0 {
                    assert_eq!(f64::from(widened), sign * f64::INFINITY);
                } else {
                    assert!(widened.is_nan());
                }
                continue;
            }
            let expected = if exponent == 0 {
                sign * fraction * 2f64.powi(-24)
            } else {
                sign * (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15)
            };
            assert_eq!(f64::from(widened), expected, "bits {bits:#06x}");
            assert_eq!(widened.is_sign_negative(), bits >> 15 == 1);
        }
    }

    #[test]
    fn bf16_widening_keeps_the_high_half() {
        for bits in [0u16, 0x3f80, 0xbf80, 0x7f7f, 0x0001, 0x8000] {
            assert_eq!(bf16_to_f32(bits).to_bits() >> 16, u32::from(bits));
        }
    }

    #[test]
    fn lazy_tree_equals_repops_tree_for_every_length() {
        for n in 0..300 {
            let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.61).sin() * 1e3).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.7).cos() + 1e-3).collect();
            let lazy = lazy_tree_sum(n, &|j| a[j] * b[j]);
            assert_eq!(
                lazy.to_bits(),
                repops::tree_dot_f32(&a, &b).to_bits(),
                "n {n}"
            );
        }
    }

    #[test]
    fn linear_is_identical_on_one_thread_and_many() {
        let rows = 3;
        let weight = Matrix {
            rows: 300,
            cols: 129,
            data: WeightData::Bf16(
                (0..300 * 129)
                    .map(|i| (i * 7919 % 65_000) as u16 & 0xbfff)
                    .collect(),
            ),
        };
        let input: Vec<f32> = (0..rows * 129).map(|i| (i as f32 * 0.37).sin()).collect();
        let bias: Vec<f32> = (0..300).map(|i| i as f32 * 0.01).collect();
        let fast = linear(&weight, Some(&bias), &input, rows);
        for r in 0..rows {
            for f in 0..300 {
                let x = &input[r * 129..(r + 1) * 129];
                let expected = repops::tree_dot_f32(&weight.row_f32(f), x) + bias[f];
                assert_eq!(fast[r * 300 + f].to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn exp_and_ln_match_the_platform_to_a_few_ulps() {
        for i in -2000..2000 {
            let x = f64::from(i) * 0.173;
            let ours = exp_f64(x);
            let platform = x.exp();
            assert!(((ours - platform) / platform).abs() < 1e-15, "exp({x})");
        }
        for i in 1..4000 {
            let x = f64::from(i) * 2.5e3 + 1e-3;
            let ours = ln_f64(x);
            let platform = x.ln();
            assert!(
                (ours - platform).abs() <= 1e-15 * platform.abs().max(1.0),
                "ln({x})"
            );
        }
        assert_eq!(ln_f64(1.0), 0.0);
    }

    #[test]
    fn sin_cos_stay_accurate_up_to_the_position_bound() {
        let mut x = 0.0f64;
        while x <= MAX_ANGLE {
            for angle in [x, -x] {
                let (s, c) = sin_cos_f64(angle);
                assert!((s - angle.sin()).abs() < 2e-15, "sin({angle})");
                assert!((c - angle.cos()).abs() < 2e-15, "cos({angle})");
            }
            x = x * 1.37 + 0.731;
        }
        let (s, c) = sin_cos_f64(MAX_ANGLE);
        assert!((s - MAX_ANGLE.sin()).abs() < 2e-15 && (c - MAX_ANGLE.cos()).abs() < 2e-15);
    }
}
