//! RepOps: reproducible operators for the reference math.
//!
//! Gensyn's RepOps makes the same work produce the same bits on different hardware by taking
//! away the one freedom a backend has: the order of floating-point operations. Addition is not
//! associative in IEEE 754, so a CUDA warp reduction, a Metal SIMD-group reduction and one CPU
//! thread round differently. The primitives here fix the order once, in the index space rather
//! than the execution order, so any chunking, thread count or reassociation a backend applies
//! still produces the same bits:
//!
//! - every reduction is a pairwise tree over a power-of-two capacity, split at the midpoint of
//!   each node, with the leaves past the end absent rather than added as zeros;
//! - a dot product is the product of every pair, each rounded once, then that tree; there is no
//!   fused multiply-add, which Rust never contracts on its own;
//! - a norm is the square root, which IEEE 754 rounds correctly, of the tree sum of squares;
//! - accumulation over INT8 boundary codes is done in integers, which is exact in any order;
//! - `exp`, `ln`, `pow`, `sin` and `cos` are fixed-order polynomials on IEEE basic operations
//!   only, because the platform `libm` may differ in the last bit between CUDA, Metal, glibc
//!   and the MSVC runtime.
//!
//! A [`RepOpsManifest`] names these conventions and carries a version. Its id is folded into
//! every digest the runtime produces, so two results are only ever compared when they were
//! computed under the same conventions. Digests are FNV-1a 64 today, the primitive the rest of
//! Kadmium content-addresses with; nothing in the protocol depends on which hash it is.
//!
//! What is borrowed from Gensyn and what is not: the fixed reduction order and the re-implemented
//! transcendentals are RepOps' two ideas; the single multiply-add convention is open-1b's.
//! RepOps covers CUDA kernels for whole model families in FP32; this module covers the synthetic
//! reference kernels of this runtime, in f32 and f64, and makes no claim about real-model
//! numerics until real kernels are rebuilt on it.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Version of the conventions below. Bump it whenever a convention changes, so digests computed
/// under the old one stop comparing equal to digests computed under the new one.
pub const REPOPS_VERSION: u32 = 1;

/// The conventions, in words. The manifest id is the FNV-1a 64 of this text with the version,
/// so editing a line changes every digest, which is what a changed convention must do.
pub const REPOPS_CONVENTIONS: &[&str] = &[
    "reduce: pairwise tree over next_power_of_two(n) leaves, split at the node midpoint, absent leaves skipped",
    "dot: per-element product rounded once, then the reduce tree; no fused multiply-add",
    "norm: IEEE sqrt of the tree sum of squares",
    "int8: sums of INT8 codes and their squares accumulated in i64, exact; one f64 tree over block scales",
    "exp: Cody-Waite ln2 reduction, degree-7 Taylor polynomial in Horner order, scaling by the exponent field",
    "ln: mantissa in [sqrt(1/2), sqrt(2)), fdlibm logf rational kernel in fixed order",
    "pow: exp(y * ln(x)) for x > 0",
    "sin, cos: Cody-Waite three-part pi/2 reduction, Cephes kernels, quadrant by n mod 4",
    "digest: FNV-1a 64 over the manifest id, a tag, then the exact little-endian bytes of the operands",
];

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The conventions a digest was computed under.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepOpsManifest {
    pub version: u32,
    /// FNV-1a 64 of the version line and the conventions; folded into every digest.
    pub id: u64,
    pub conventions: Vec<String>,
}

impl RepOpsManifest {
    /// The manifest this build computes under.
    pub fn current() -> &'static RepOpsManifest {
        static MANIFEST: OnceLock<RepOpsManifest> = OnceLock::new();
        MANIFEST.get_or_init(|| {
            let text = format!(
                "kadmium-repops/v{REPOPS_VERSION}\n{}",
                REPOPS_CONVENTIONS.join("\n")
            );
            RepOpsManifest {
                version: REPOPS_VERSION,
                id: fnv1a(text.as_bytes()),
                conventions: REPOPS_CONVENTIONS.iter().map(|s| s.to_string()).collect(),
            }
        })
    }

    /// The id of the current manifest.
    pub fn id() -> u64 {
        Self::current().id
    }
}

// ================================================================================================
// Digests
// ================================================================================================

/// An FNV-1a 64 digest seeded with the manifest id and a tag, fed the exact bytes of operands.
#[derive(Debug, Clone)]
pub struct Digest {
    state: u64,
}

impl Digest {
    pub fn new(tag: &str) -> Self {
        let mut digest = Self { state: FNV_OFFSET };
        digest.update(&RepOpsManifest::id().to_le_bytes());
        digest.update(tag.as_bytes());
        digest.update(&[0]);
        digest
    }

    pub fn update(&mut self, bytes: &[u8]) -> &mut Self {
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
        self
    }

    pub fn update_u32(&mut self, value: u32) -> &mut Self {
        self.update(&value.to_le_bytes())
    }

    pub fn update_u64(&mut self, value: u64) -> &mut Self {
        self.update(&value.to_le_bytes())
    }

    /// The exact bits of every value, little-endian: no rounding, no quantization.
    pub fn update_f32s(&mut self, values: &[f32]) -> &mut Self {
        for v in values {
            self.update(&v.to_le_bytes());
        }
        self
    }

    pub fn update_f64s(&mut self, values: &[f64]) -> &mut Self {
        for v in values {
            self.update(&v.to_le_bytes());
        }
        self
    }

    pub fn update_i8s(&mut self, values: &[i8]) -> &mut Self {
        for v in values {
            self.update(&v.to_le_bytes());
        }
        self
    }

    pub fn finish(&self) -> u64 {
        self.state
    }
}

/// One digest over a tagged byte string.
pub fn digest_bytes(tag: &str, bytes: &[u8]) -> u64 {
    Digest::new(tag).update(bytes).finish()
}

/// Folds a chain of digests, in order, into one: the digest of a unit from its layer digests.
pub fn fold_chain(tag: &str, digests: &[u64]) -> u64 {
    let mut digest = Digest::new(tag);
    digest.update_u32(digests.len() as u32);
    for d in digests {
        digest.update_u64(*d);
    }
    digest.finish()
}

// ================================================================================================
// Fixed-order reductions
// ================================================================================================

/// A value the reduce tree can add. Integers are exact in any order; floats are not, which is
/// why the tree exists.
pub trait Summand: Copy {
    const ZERO: Self;
    fn add(self, rhs: Self) -> Self;
}

impl Summand for f32 {
    const ZERO: Self = 0.0;
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Summand for f64 {
    const ZERO: Self = 0.0;
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Summand for i64 {
    const ZERO: Self = 0;
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

fn tree_node<T: Summand>(values: &[T], lo: usize, size: usize) -> T {
    if size == 1 {
        return values[lo];
    }
    let half = size / 2;
    let left = tree_node(values, lo, half);
    let mid = lo + half;
    if mid >= values.len() {
        left
    } else {
        left.add(tree_node(values, mid, half))
    }
}

/// The canonical pairwise tree sum: a perfect binary tree over `next_power_of_two(n)` leaves,
/// each node the sum of its left and right children, a child whose leaves all lie past the end
/// absent (its sibling's value passes up unchanged, so padding never adds a zero). The result
/// depends only on the values and their indices, never on how they were chunked or scheduled.
pub fn tree_sum<T: Summand>(values: &[T]) -> T {
    if values.is_empty() {
        return T::ZERO;
    }
    tree_node(values, 0, values.len().next_power_of_two())
}

pub fn tree_sum_f32(values: &[f32]) -> f32 {
    tree_sum(values)
}

pub fn tree_sum_f64(values: &[f64]) -> f64 {
    tree_sum(values)
}

/// Dot product: every product rounded once, then the tree. No fused multiply-add.
pub fn tree_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let products: Vec<f32> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    tree_sum(&products)
}

pub fn tree_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let products: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    tree_sum(&products)
}

/// Sum of squares of `f32` values accumulated in `f64`, every square exact, then the tree.
pub fn tree_sum_squares_f64(values: &[f32]) -> f64 {
    let squares: Vec<f64> = values.iter().map(|&v| (v as f64) * (v as f64)).collect();
    tree_sum(&squares)
}

pub fn tree_sum_squares_f32(values: &[f32]) -> f32 {
    tree_dot_f32(values, values)
}

/// The L2 norm: IEEE square root of the tree sum of squares.
pub fn tree_norm_f32(values: &[f32]) -> f32 {
    tree_sum_squares_f32(values).sqrt()
}

pub fn tree_norm_f64(values: &[f64]) -> f64 {
    tree_dot_f64(values, values).sqrt()
}

/// The reduce tree as a streaming accumulator: leaves arrive in any order and any chunking, on
/// any number of threads, and a node is formed the moment both of its children are complete.
/// Because a node only ever adds its own two children, no arrival order can change any operand
/// of any addition, and [`TreeReducer::finish`] is bit-identical to [`tree_sum`].
#[derive(Debug, Clone)]
pub struct TreeReducer<T: Summand> {
    len: usize,
    capacity: usize,
    /// Heap layout: node 1 is the root, the leaves sit at `capacity..2 * capacity`.
    nodes: Vec<Option<T>>,
    pushed: usize,
}

impl<T: Summand> TreeReducer<T> {
    pub fn new(len: usize) -> Self {
        let capacity = len.next_power_of_two().max(1);
        Self {
            len,
            capacity,
            nodes: vec![None; 2 * capacity],
            pushed: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when every leaf has arrived.
    pub fn is_complete(&self) -> bool {
        self.pushed == self.len
    }

    /// Whether every leaf under `node` lies past the end of the input.
    fn absent(&self, node: usize) -> bool {
        let mut leftmost = node;
        while leftmost < self.capacity {
            leftmost *= 2;
        }
        leftmost - self.capacity >= self.len
    }

    /// Places the leaf at `index` and forms every ancestor that is now complete.
    pub fn push(&mut self, index: usize, value: T) -> Result<(), String> {
        if index >= self.len {
            return Err(format!(
                "TreeReducer: leaf {index} is past the end of {} leaves",
                self.len
            ));
        }
        let mut node = self.capacity + index;
        if self.nodes[node].is_some() {
            return Err(format!("TreeReducer: leaf {index} pushed twice"));
        }
        self.nodes[node] = Some(value);
        self.pushed += 1;
        while node > 1 {
            let parent = node / 2;
            let left = 2 * parent;
            let right = left + 1;
            let Some(l) = self.nodes[left] else {
                break;
            };
            let combined = if self.absent(right) {
                l
            } else if let Some(r) = self.nodes[right] {
                l.add(r)
            } else {
                break;
            };
            self.nodes[parent] = Some(combined);
            node = parent;
        }
        Ok(())
    }

    /// Places a contiguous chunk of leaves starting at `start`: what one thread or one block of
    /// a backend contributes.
    pub fn push_chunk(&mut self, start: usize, values: &[T]) -> Result<(), String> {
        for (offset, &v) in values.iter().enumerate() {
            self.push(start + offset, v)?;
        }
        Ok(())
    }

    /// The root, once every leaf has arrived.
    pub fn finish(&self) -> Result<T, String> {
        if self.len == 0 {
            return Ok(T::ZERO);
        }
        if !self.is_complete() {
            return Err(format!(
                "TreeReducer: {} of {} leaves arrived",
                self.pushed, self.len
            ));
        }
        self.nodes[1].ok_or_else(|| "TreeReducer: root not formed".to_string())
    }
}

// ================================================================================================
// Integer-exact accumulation over INT8 boundary codes
// ================================================================================================

/// Sum of squares of INT8 codes, exact: an `i64` holds 2^47 squared codes without overflow.
pub fn int8_sum_squares(codes: &[i8]) -> i64 {
    codes.iter().map(|&q| (q as i64) * (q as i64)).sum()
}

/// Dot product of two INT8 code vectors, exact.
pub fn int8_dot(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i64) * (y as i64))
        .sum()
}

/// Energy of a blockwise INT8 tensor: per block the exact integer sum of squared codes scaled
/// by the block's squared scale, then one tree over the blocks. The only roundings are the two
/// multiplications per block and the tree.
pub fn int8_block_energy(codes: &[i8], scales: &[f32], block: usize) -> f64 {
    let block = block.max(1);
    let terms: Vec<f64> = codes
        .chunks(block)
        .zip(scales)
        .map(|(chunk, &scale)| {
            let scale = scale as f64;
            (scale * scale) * (int8_sum_squares(chunk) as f64)
        })
        .collect();
    tree_sum(&terms)
}

/// The L2 norm of a blockwise INT8 tensor from its integer-exact energy.
pub fn int8_block_l2(codes: &[i8], scales: &[f32], block: usize) -> f64 {
    int8_block_energy(codes, scales, block).sqrt()
}

// ================================================================================================
// Transcendentals on IEEE basic operations only
// ================================================================================================

/// `x * 2^k` through the exponent field, in at most two steps so every factor is a normal number.
fn ldexp_f32(x: f32, k: i32) -> f32 {
    fn pow2(k: i32) -> f32 {
        f32::from_bits(((k + 127) as u32) << 23)
    }
    if k > 127 {
        (x * pow2(127)) * pow2((k - 127).min(127))
    } else if k < -126 {
        (x * pow2(-126)) * pow2((k + 126).max(-126))
    } else {
        x * pow2(k)
    }
}

/// `e^x` in `f32`: `x = k ln2 + r` with `|r| <= ln2 / 2`, a degree-7 Taylor polynomial of `r`
/// in Horner order, then scaling by `2^k`. Overflows to infinity above 88.72 and underflows
/// to zero below -103.97, like the platform function.
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.722_84 {
        return f32::INFINITY;
    }
    if x < -103.972_08 {
        return 0.0;
    }
    // The high 16 bits of ln 2, so `k * LN2_HI` is exact for |k| < 256, and the remainder.
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
    const LN2_LO: f32 = 1.428_606_8e-6_f32;
    let k = (x * std::f32::consts::LOG2_E).round();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67
                    + r * (0.041_666_668
                        + r * (0.008_333_334 + r * (0.001_388_889 + r * 1.984_127e-4))))));
    ldexp_f32(p, k as i32)
}

/// Natural logarithm in `f32`: the fdlibm `logf` kernel in fixed order.
pub fn ln(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    // The fdlibm `logf` constants, by their bits so there is no question which `f32` they are.
    const LN2_HI: f32 = f32::from_bits(0x3f31_7180);
    const LN2_LO: f32 = f32::from_bits(0x3717_f7d1);
    const LG1: f32 = f32::from_bits(0x3f2a_aaaa);
    const LG2: f32 = f32::from_bits(0x3ecc_ce13);
    const LG3: f32 = f32::from_bits(0x3e91_e9ee);
    const LG4: f32 = f32::from_bits(0x3e78_9e26);
    const SQRT2: f32 = std::f32::consts::SQRT_2;

    let mut bits = x.to_bits();
    let mut k: i32 = 0;
    if bits < 0x0080_0000 {
        // Subnormal: scale into the normal range first.
        bits = (x * 33_554_432.0).to_bits();
        k -= 25;
    }
    k += ((bits >> 23) as i32) - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > SQRT2 {
        m *= 0.5;
        k += 1;
    }
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * LG4);
    let t2 = z * (LG1 + w * LG3);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let kf = k as f32;
    kf * LN2_HI - ((hfsq - (s * (hfsq + r) + kf * LN2_LO)) - f)
}

/// `x^y` for `x > 0` as `exp(y ln x)`.
pub fn pow(x: f32, y: f32) -> f32 {
    if y == 0.0 || x == 1.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 { 0.0 } else { f32::INFINITY };
    }
    exp(y * ln(x))
}

/// `pi / 2` in three parts: 1.5703125 (8 significant bits), 4.8375129e-4 (12 significant
/// bits) and the remainder rounded once, so `n * PIO2_1` and `n * PIO2_2` are exact products.
const PIO2_1: f32 = f32::from_bits(0x3fc9_0000);
const PIO2_2: f32 = f32::from_bits(0x39fd_a000);
const PIO2_3: f32 = (std::f64::consts::FRAC_PI_2 - PIO2_1 as f64 - PIO2_2 as f64) as f32;

/// Reduces `x` to `r` in `[-pi/4, pi/4]` and the quadrant `n mod 4`. The three-part constant
/// keeps `n * PIO2_1` and `n * PIO2_2` exact while `|n| < 4096`, so the reduction is accurate
/// for `|x|` below about 6,400 and merely deterministic beyond it.
fn reduce_pio2(x: f32) -> (f32, u32) {
    let n = (x * std::f32::consts::FRAC_2_PI).round();
    let r = ((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3;
    let quadrant = ((n as i64).rem_euclid(4)) as u32;
    (r, quadrant)
}

fn sin_kernel(r: f32) -> f32 {
    let z = r * r;
    r + r * z * (-0.166_666_55 + z * (0.008_332_161 + z * -1.951_529_6e-4))
}

fn cos_kernel(r: f32) -> f32 {
    let z = r * r;
    1.0 - 0.5 * z + z * z * (0.041_666_646 + z * (-0.001_388_731_6 + z * 2.443_315_7e-5))
}

/// Sine in `f32` on IEEE basic operations.
pub fn sin(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (r, quadrant) = reduce_pio2(x);
    match quadrant {
        0 => sin_kernel(r),
        1 => cos_kernel(r),
        2 => -sin_kernel(r),
        _ => -cos_kernel(r),
    }
}

/// Cosine in `f32` on IEEE basic operations.
pub fn cos(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (r, quadrant) = reduce_pio2(x);
    match quadrant {
        0 => cos_kernel(r),
        1 => -sin_kernel(r),
        2 => -cos_kernel(r),
        _ => sin_kernel(r),
    }
}

pub fn sin_cos(x: f32) -> (f32, f32) {
    (sin(x), cos(x))
}

/// The logistic function `1 / (1 + e^-x)`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// SiLU, also called swish: `x / (1 + e^-x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32 * 1e3))
            .collect()
    }

    /// A reduction the way a backend might run it: a chunk size, and an order in which the
    /// chunks (threads, warps, SIMD groups) finish.
    #[derive(Debug, Clone, Copy)]
    enum SimulatedBackend {
        CpuSerial,
        CudaWarps,
        MetalSimdGroups,
        VulkanSubgroups,
        ArmNeonLanes,
        ThreadsFinishingBackwards,
    }

    impl SimulatedBackend {
        fn reduce(self, values: &[f32]) -> f32 {
            let (chunk, backwards, interleave) = match self {
                Self::CpuSerial => (values.len().max(1), false, false),
                Self::CudaWarps => (32, false, true),
                Self::MetalSimdGroups => (16, true, false),
                Self::VulkanSubgroups => (64, false, true),
                Self::ArmNeonLanes => (4, true, true),
                Self::ThreadsFinishingBackwards => (7, true, false),
            };
            let mut chunks: Vec<(usize, &[f32])> = values
                .chunks(chunk)
                .enumerate()
                .map(|(i, c)| (i * chunk, c))
                .collect();
            if backwards {
                chunks.reverse();
            }
            if interleave {
                let (even, odd): (Vec<_>, Vec<_>) = chunks
                    .iter()
                    .copied()
                    .partition(|(s, _)| (s / chunk) % 2 == 0);
                chunks = odd.into_iter().chain(even).collect();
            }
            let mut reducer = TreeReducer::new(values.len());
            for (start, c) in chunks {
                reducer.push_chunk(start, c).unwrap();
            }
            reducer.finish().unwrap()
        }
    }

    #[test]
    fn tree_sum_is_bit_identical_under_any_chunking_and_arrival_order() {
        for n in [
            0usize, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 257,
        ] {
            let values = ramp(n);
            let canonical = tree_sum_f32(&values);
            for chunk in [1usize, 2, 3, 5, 7, 8, 13, 16, 64, n.max(1)] {
                let mut forward = TreeReducer::new(n);
                let mut backward = TreeReducer::new(n);
                let chunks: Vec<(usize, &[f32])> = values
                    .chunks(chunk)
                    .enumerate()
                    .map(|(i, c)| (i * chunk, c))
                    .collect();
                for (start, c) in &chunks {
                    forward.push_chunk(*start, c).unwrap();
                }
                for (start, c) in chunks.iter().rev() {
                    backward.push_chunk(*start, c).unwrap();
                }
                assert_eq!(
                    forward.finish().unwrap().to_bits(),
                    canonical.to_bits(),
                    "n={n} chunk={chunk} forward"
                );
                assert_eq!(
                    backward.finish().unwrap().to_bits(),
                    canonical.to_bits(),
                    "n={n} chunk={chunk} backward"
                );
            }
            // Leaves arriving one at a time in a scrambled order.
            let mut scrambled = TreeReducer::new(n);
            let mut order: Vec<usize> = (0..n).collect();
            let mut seed = 0x9e37_79b9u64;
            for i in (1..n).rev() {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let j = (seed >> 33) as usize % (i + 1);
                order.swap(i, j);
            }
            for i in order {
                scrambled.push(i, values[i]).unwrap();
            }
            assert_eq!(scrambled.finish().unwrap().to_bits(), canonical.to_bits());
            for backend in [
                SimulatedBackend::CpuSerial,
                SimulatedBackend::CudaWarps,
                SimulatedBackend::MetalSimdGroups,
                SimulatedBackend::VulkanSubgroups,
                SimulatedBackend::ArmNeonLanes,
                SimulatedBackend::ThreadsFinishingBackwards,
            ] {
                assert_eq!(
                    backend.reduce(&values).to_bits(),
                    canonical.to_bits(),
                    "n={n} {backend:?}"
                );
            }
        }
    }

    #[test]
    fn the_tree_differs_from_a_sequential_sum_and_the_reducer_rejects_misuse() {
        // A sequential left fold loses the two ones; the tree keeps them paired.
        let values = [1.0f32, 1e8, -1e8, 1.0];
        let sequential = values.iter().fold(0.0f32, |acc, v| acc + v);
        assert_eq!(sequential, 1.0);
        assert_eq!(tree_sum_f32(&values), 0.0);
        assert_eq!(tree_sum_f32(&[]), 0.0);
        assert_eq!(tree_sum_f32(&[2.5]), 2.5);
        assert_eq!(tree_sum(&[1i64, 2, 3]), 6);

        let mut reducer = TreeReducer::new(3);
        assert!(reducer.push(3, 1.0f32).is_err());
        reducer.push(1, 1.0).unwrap();
        assert!(reducer.push(1, 2.0).is_err());
        assert!(reducer.finish().is_err());
        reducer.push(0, 1.0).unwrap();
        reducer.push(2, 1.0).unwrap();
        assert!(reducer.is_complete());
        assert_eq!(reducer.finish().unwrap(), 3.0);
        assert_eq!(TreeReducer::<f64>::new(0).finish().unwrap(), 0.0);
    }

    #[test]
    fn dots_norms_and_int8_energy_follow_the_tree() {
        let a = ramp(50);
        let b: Vec<f32> = ramp(50).iter().map(|v| v * 0.5 + 1.0).collect();
        let products: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        assert_eq!(
            tree_dot_f32(&a, &b).to_bits(),
            tree_sum_f32(&products).to_bits()
        );
        assert_eq!(
            tree_norm_f32(&a).to_bits(),
            tree_dot_f32(&a, &a).sqrt().to_bits()
        );
        let squares: Vec<f64> = a.iter().map(|&v| (v as f64) * (v as f64)).collect();
        assert_eq!(
            tree_sum_squares_f64(&a).to_bits(),
            tree_sum_f64(&squares).to_bits()
        );

        let codes: Vec<i8> = (0..64).map(|i| ((i * 37) % 255) as i8).collect();
        let scales = [0.01f32, 0.02, 0.03, 0.04];
        assert_eq!(int8_sum_squares(&codes[..4]), {
            let c = &codes[..4];
            c.iter().map(|&q| (q as i64) * (q as i64)).sum::<i64>()
        });
        assert_eq!(
            int8_dot(&codes[..3], &codes[..3]),
            int8_sum_squares(&codes[..3])
        );
        let dequantized: Vec<f32> = codes
            .chunks(16)
            .zip(scales)
            .flat_map(|(c, s)| c.iter().map(move |&q| (q as f32) * s))
            .collect();
        let exact = int8_block_l2(&codes, &scales, 16);
        let floating = tree_norm_f32(&dequantized) as f64;
        assert!(
            (exact - floating).abs() <= floating * 1e-5,
            "{exact} vs {floating}"
        );
    }

    #[test]
    fn transcendentals_track_the_platform_functions_and_hit_exact_points() {
        // The three parts of pi/2 add up to it, and the first two have the short significands
        // that keep their products with the quadrant count exact.
        let pio2 = PIO2_1 as f64 + PIO2_2 as f64 + PIO2_3 as f64;
        assert!((pio2 - std::f64::consts::FRAC_PI_2).abs() < 1e-14, "{pio2}");
        assert_eq!(PIO2_1.to_bits() & 0xffff, 0);
        assert_eq!(PIO2_2.to_bits() & 0xfff, 0);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert_eq!(pow(3.0, 0.0), 1.0);
        assert_eq!(pow(1.0, 7.5), 1.0);
        assert!(exp(f32::NAN).is_nan());
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(-200.0), 0.0);
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert!(sin(f32::INFINITY).is_nan());

        let mut worst_exp = 0.0f32;
        let mut worst_ln = 0.0f32;
        let mut worst_trig = 0.0f32;
        for i in -2000..=2000 {
            let x = i as f32 * 0.05; // -100..=100
            if x > -87.0 && x < 87.0 {
                let ours = exp(x);
                let theirs = x.exp();
                worst_exp = worst_exp.max(((ours - theirs) / theirs).abs());
            }
            worst_trig = worst_trig.max((sin(x) - x.sin()).abs());
            worst_trig = worst_trig.max((cos(x) - x.cos()).abs());
            let y = (i + 2001) as f32 * 0.37 + 1e-3; // positive
            worst_ln = worst_ln.max((ln(y) - y.ln()).abs() / y.ln().abs().max(1.0));
        }
        assert!(worst_exp < 4e-7, "exp relative error {worst_exp}");
        assert!(worst_ln < 4e-7, "ln relative error {worst_ln}");
        assert!(worst_trig < 2e-6, "sin/cos absolute error {worst_trig}");
        // Small and subnormal inputs of ln, large and tiny outputs of exp.
        assert!((ln(1e-40) - (1e-40f32).ln()).abs() < 1e-4);
        assert!(((exp(-100.0) - (-100.0f32).exp()) / (-100.0f32).exp()).abs() < 1e-5);
        assert!(((exp(88.5) - (88.5f32).exp()) / (88.5f32).exp()).abs() < 4e-7);
        for t in 0..=16 {
            let y = t as f32 / 16.0;
            let ours = pow(10_000.0, y);
            let theirs = 10_000f32.powf(y);
            assert!(((ours - theirs) / theirs).abs() < 4e-6, "pow {y}");
        }
        assert!((silu(1.5) - 1.5 / (1.0 + (-1.5f32).exp())).abs() < 1e-6);
        assert!((sigmoid(-2.0) - 1.0 / (1.0 + 2.0f32.exp())).abs() < 1e-7);
    }

    #[test]
    fn the_manifest_id_folds_into_every_digest() {
        let manifest = RepOpsManifest::current();
        assert_eq!(manifest.version, REPOPS_VERSION);
        assert_ne!(manifest.id, 0);
        assert_eq!(manifest.conventions.len(), REPOPS_CONVENTIONS.len());
        assert_eq!(RepOpsManifest::id(), manifest.id);

        let a = digest_bytes("layer", &[1, 2, 3]);
        assert_eq!(a, Digest::new("layer").update(&[1, 2, 3]).finish());
        assert_ne!(a, digest_bytes("trunk", &[1, 2, 3]));
        assert_ne!(a, fnv1a(&[1, 2, 3]), "the tag and manifest change the hash");
        assert_ne!(fold_chain("unit", &[a, 7]), fold_chain("unit", &[7, a]));
        assert_ne!(fold_chain("unit", &[a]), fold_chain("unit", &[a, a]));
        assert_eq!(
            Digest::new("x").update_f32s(&[1.5]).finish(),
            Digest::new("x").update(&1.5f32.to_le_bytes()).finish()
        );
    }
}
