//! Fine-grained block-scaled and `M`-grouped MoE matrix kernels, after DeepSeek's `DeepGEMM`.
//!
//! `DeepGEMM` (`deepseek-ai/DeepGEMM`) is a compact FP8 GEMM library built for DeepSeek-V3 and R1.
//! Rather than parameterising over every layout and precision, it implements four things cleanly:
//!
//! 1. **Fine-grained sub-block scaling (`per_token_cast_to_fp8`, `per_block_cast_to_fp8`):**
//!    Standard per-tensor 8-bit quantization loses accuracy on MoE activation outliers. DeepGEMM
//!    scales LHS activations per `1 x G` row tile and RHS weights per `G x G` 2D sub-block
//!    (`G = 128` on full models, configurable down to `16` on small reference tensors), optionally
//!    snapping scales upward to lossless power-of-two `UE8M0` exponents (`ceil_to_ue8m0`) so the
//!    scale multiplication is an exponent shift free of rounding drift.
//! 2. **TMA-aligned column-major scale layout (`get_col_major_tma_aligned_tensor`):**
//!    The `[M, ceil_div(K, G)]` activation scale matrix is transposed into column-major order with
//!    its leading `M` dimension padded to a multiple of 4 (`TMA_ALIGN_ROWS = 4`, 16-byte aligned
//!    for `f32` scales).
//! 3. **Two-level accumulation (`gemm_fp8_fp8_bf16_nt`):**
//!    Each `G`-element tile along `K` accumulates in inner integer/tensor-core order, scales by
//!    `lhs_scale[m, kb] * rhs_scale[nb, kb]`, and promotes the tile sum into an outer high-precision
//!    accumulator across `0..ceil_div(K, G)`. Here both levels reduce through [`crate::repops`],
//!    so the two-level sum is bit-identical on every backend.
//! 4. **`M`-grouped GEMMs for MoE (`m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` and `_masked`):**
//!    Every routed expert in an MoE layer shares the same `(N, K)` shape, so DeepGEMM groups
//!    exclusively along the token axis `M`:
//!    - **Contiguous layout** packs the rows routed to each expert one after another, padded per
//!      expert to a multiple of `block_m` with `m_indices[row] = -1` on padding rows, for prefill
//!      and training.
//!    - **Masked layout** takes a fixed `[num_groups, max_m, K]` buffer and a device-side
//!      `masked_m: &[u32]` array giving each expert's valid row count, for decode under fixed-shape
//!      execution graphs where the host scheduler does not know how many tokens landed on each
//!      expert ahead of time.
//!
//! Left out, deliberately: Hopper SM90 WGMMA/TMA hardware descriptors, warp-group specialization,
//! and post-compilation SASS `FFMA` reuse-bit interleaving (`interleave_ffma.py`), which are
//! specific to NVIDIA Hopper assembly. What is kept is the scaling math, the `UE8M0` exponent snap,
//! the column-major aligned scale layout, the two-level RepOps accumulation, and both `M`-grouped
//! MoE schedules, with tests proving that `contiguous` (at any `block_m` padding), `masked` (at
//! any `max_m`), and single-row execution produce bit-identical bytes and RepOps digests.

use crate::{MoERoutedExpertSelection, repops};
use serde::{Deserialize, Serialize};

/// Default sub-block granularity on reference tensors (`16`), matching [`crate::LAYER_BOUNDARY_BLOCK`].
pub const DEFAULT_SUBBLOCK_SIZE: usize = 16;

/// DeepGEMM's full-model sub-block granularity (`128` elements along `K` and `N`).
pub const DEEPGEMM_MODEL_SUBBLOCK_SIZE: usize = 128;

/// Row alignment required by `get_col_major_tma_aligned_scales` (`4` `f32` scales = 16 bytes).
pub const TMA_ALIGN_ROWS: usize = 4;

/// Maximum finite magnitude represented in signed 8-bit E4M3 (`448.0`), mapped onto integer codes
/// `[-127, 127]` in the reference math so the inner tile product is integer-exact on every CPU/GPU.
pub const FP8_E4M3_MAX: f32 = 448.0;

/// Ceiling integer division `ceil(a / b)`, DeepGEMM's `ceil_div`.
#[inline]
pub const fn ceil_div(a: usize, b: usize) -> usize {
    if b == 0 { 0 } else { a.div_ceil(b) }
}

/// Rounds `a` up to the next multiple of `align`.
#[inline]
pub const fn align_up(a: usize, align: usize) -> usize {
    if align == 0 {
        a
    } else {
        ceil_div(a, align) * align
    }
}

/// How sub-block scaling factors are represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleFormat {
    /// Exact IEEE `f32` `absmax / 127.0` per sub-block (Hopper SM90 convention).
    ExactF32,
    /// Power-of-two `UE8M0` exponent scale (`2^{ceil(log2(absmax / 127.0))}`), Blackwell / DeepGEMM
    /// convention that prevents mantissa rounding drift in block scale multiplication.
    Ue8m0PowerOfTwo,
}

/// Snaps a positive scale factor upward to the nearest power of two (`UE8M0`: unsigned 8-bit
/// exponent, 0-bit mantissa), clamping to IEEE `f32` normal exponents `[-126, 127]`.
#[inline]
pub fn ceil_to_ue8m0(scale: f32) -> f32 {
    if !scale.is_finite() || scale <= 0.0 {
        return f32::from_bits(1 << 23); // 2^-126
    }
    let bits = scale.to_bits();
    let biased_exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    let ceil_biased = if mantissa == 0 {
        biased_exp
    } else {
        biased_exp + 1
    }
    .clamp(1, 254);
    f32::from_bits((ceil_biased as u32) << 23)
}

/// Extracts the unbiased 8-bit `UE8M0` biased exponent byte (`1..=254`) from a scale factor.
#[inline]
pub fn ue8m0_biased_byte(scale: f32) -> u8 {
    let snapped = ceil_to_ue8m0(scale);
    ((snapped.to_bits() >> 23) & 0xff) as u8
}

/// Reconstructs the `f32` power-of-two scale from a `UE8M0` biased exponent byte.
#[inline]
pub fn ue8m0_to_f32(byte: u8) -> f32 {
    let clamped = byte.clamp(1, 254) as u32;
    f32::from_bits(clamped << 23)
}

#[inline]
fn compute_scale(absmax: f32, format: ScaleFormat) -> f32 {
    let raw = if absmax > 1e-12 {
        absmax / 127.0
    } else {
        1.0 / 127.0
    };
    match format {
        ScaleFormat::ExactF32 => raw,
        ScaleFormat::Ue8m0PowerOfTwo => ceil_to_ue8m0(raw),
    }
}

/// Column-major TMA-aligned scale tensor for `[rows, k_blocks]` activation scales, after
/// DeepGEMM's `get_col_major_tma_aligned_tensor`.
///
/// The logical shape is `[rows, k_blocks]`, stored in column-major order with the row stride
/// aligned to `TMA_ALIGN_ROWS` (`4`), so column `kb` starts at `kb * aligned_rows`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColMajorAlignedScales {
    pub rows: usize,
    pub aligned_rows: usize,
    pub k_blocks: usize,
    pub data: Vec<f32>,
}

impl ColMajorAlignedScales {
    /// Converts a row-major `[rows, k_blocks]` scale buffer into a column-major TMA-aligned buffer.
    pub fn from_row_major(row_major: &[f32], rows: usize, k_blocks: usize) -> Self {
        let aligned_rows = align_up(rows, TMA_ALIGN_ROWS);
        let mut data = vec![0.0f32; aligned_rows * k_blocks];
        for r in 0..rows {
            for kb in 0..k_blocks {
                if let Some(&val) = row_major.get(r * k_blocks + kb) {
                    data[kb * aligned_rows + r] = val;
                }
            }
        }
        Self {
            rows,
            aligned_rows,
            k_blocks,
            data,
        }
    }

    /// Reads the scale for logical `(row, k_block)`.
    #[inline]
    pub fn get(&self, row: usize, k_block: usize) -> f32 {
        if row < self.rows && k_block < self.k_blocks {
            self.data[k_block * self.aligned_rows + row]
        } else {
            0.0
        }
    }

    /// Converts back to row-major `[rows, k_blocks]`.
    pub fn to_row_major(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.k_blocks);
        for r in 0..self.rows {
            for kb in 0..self.k_blocks {
                out.push(self.get(r, kb));
            }
        }
        out
    }
}

/// LHS activation matrix `[rows, k]` quantized with fine-grained `1 x block_size` per-token
/// sub-block scaling (`per_token_cast_to_fp8`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerTokenBlockQuantized {
    pub rows: usize,
    pub k: usize,
    pub block_size: usize,
    pub format: ScaleFormat,
    /// Quantized codes `[rows * k]` in row-major order.
    pub codes: Vec<i8>,
    /// Row-major scales `[rows * ceil_div(k, block_size)]`.
    pub scales: Vec<f32>,
}

impl PerTokenBlockQuantized {
    /// Quantizes a `[rows, k]` row-major `f32` activation matrix using `1 x block_size` tiles.
    pub fn quantize(
        data: &[f32],
        rows: usize,
        k: usize,
        block_size: usize,
        format: ScaleFormat,
    ) -> Self {
        let block_size = block_size.max(1);
        let k_blocks = ceil_div(k, block_size);
        let mut codes = vec![0i8; rows * k];
        let mut scales = vec![0.0f32; rows * k_blocks];

        for r in 0..rows {
            let row_offset = r * k;
            for kb in 0..k_blocks {
                let k_start = kb * block_size;
                let k_end = (k_start + block_size).min(k);
                let slice = &data[row_offset + k_start..row_offset + k_end];
                let absmax = slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = compute_scale(absmax, format);
                scales[r * k_blocks + kb] = scale;
                let inv_scale = 1.0 / scale;
                for (idx, &v) in slice.iter().enumerate() {
                    let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    codes[row_offset + k_start + idx] = q;
                }
            }
        }

        Self {
            rows,
            k,
            block_size,
            format,
            codes,
            scales,
        }
    }

    #[inline]
    pub fn k_blocks(&self) -> usize {
        ceil_div(self.k, self.block_size)
    }

    /// Returns the column-major TMA-aligned view of this activation batch's scales.
    pub fn tma_aligned_scales(&self) -> ColMajorAlignedScales {
        ColMajorAlignedScales::from_row_major(&self.scales, self.rows, self.k_blocks())
    }

    #[inline]
    pub fn row_codes(&self, row: usize) -> &[i8] {
        &self.codes[row * self.k..(row + 1) * self.k]
    }

    #[inline]
    pub fn scale(&self, row: usize, kb: usize) -> f32 {
        self.scales[row * self.k_blocks() + kb]
    }

    /// Dequantizes back to `[rows * k]` `f32`.
    pub fn dequantize(&self) -> Vec<f32> {
        let k_blocks = self.k_blocks();
        let mut out = vec![0.0f32; self.rows * self.k];
        for r in 0..self.rows {
            for kb in 0..k_blocks {
                let scale = self.scales[r * k_blocks + kb];
                let k_start = kb * self.block_size;
                let k_end = (k_start + self.block_size).min(self.k);
                for col in k_start..k_end {
                    out[r * self.k + col] = (self.codes[r * self.k + col] as f32) * scale;
                }
            }
        }
        out
    }

    /// Wire footprint in bytes (`1` byte per code + `1` byte per UE8M0 scale or `4` bytes per f32 scale).
    pub fn wire_bytes(&self) -> usize {
        let scale_bytes = match self.format {
            ScaleFormat::Ue8m0PowerOfTwo => self.scales.len(),
            ScaleFormat::ExactF32 => self.scales.len() * 4,
        };
        16 + self.codes.len() + scale_bytes
    }
}

/// RHS weight matrix `[n, k]` quantized with fine-grained `block_size x block_size` 2D sub-block
/// scaling (`per_block_cast_to_fp8`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerBlockQuantizedMatrix {
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
    pub format: ScaleFormat,
    /// Quantized codes `[n * k]` in row-major (`[n, k]`) layout for `NT` GEMM (`x @ W^T`).
    pub codes: Vec<i8>,
    /// 2D block scales `[ceil_div(n, block_size) * ceil_div(k, block_size)]`.
    pub scales: Vec<f32>,
}

impl PerBlockQuantizedMatrix {
    /// Quantizes a `[n, k]` row-major `f32` weight matrix over `block_size x block_size` 2D tiles.
    pub fn quantize(
        weights: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        format: ScaleFormat,
    ) -> Self {
        let block_size = block_size.max(1);
        let n_blocks = ceil_div(n, block_size);
        let k_blocks = ceil_div(k, block_size);
        let mut codes = vec![0i8; n * k];
        let mut scales = vec![0.0f32; n_blocks * k_blocks];

        for nb in 0..n_blocks {
            let n_start = nb * block_size;
            let n_end = (n_start + block_size).min(n);
            for kb in 0..k_blocks {
                let k_start = kb * block_size;
                let k_end = (k_start + block_size).min(k);

                let mut absmax = 0.0f32;
                for r in n_start..n_end {
                    for c in k_start..k_end {
                        absmax = absmax.max(weights[r * k + c].abs());
                    }
                }
                let scale = compute_scale(absmax, format);
                scales[nb * k_blocks + kb] = scale;
                let inv_scale = 1.0 / scale;

                for r in n_start..n_end {
                    for c in k_start..k_end {
                        let v = weights[r * k + c];
                        codes[r * k + c] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }

        Self {
            n,
            k,
            block_size,
            format,
            codes,
            scales,
        }
    }

    #[inline]
    pub fn n_blocks(&self) -> usize {
        ceil_div(self.n, self.block_size)
    }

    #[inline]
    pub fn k_blocks(&self) -> usize {
        ceil_div(self.k, self.block_size)
    }

    #[inline]
    pub fn row_codes(&self, row_n: usize) -> &[i8] {
        &self.codes[row_n * self.k..(row_n + 1) * self.k]
    }

    #[inline]
    pub fn block_scale(&self, nb: usize, kb: usize) -> f32 {
        self.scales[nb * self.k_blocks() + kb]
    }
}

/// DeepGEMM's two-level accumulation dot product for one output element `(m, n)`:
///
/// - **Level 1 (Inner Tile MMA):** Within each `k_block` of width `block_size`, the product of
///   signed 8-bit codes is summed in exact `i32` integer arithmetic (`sum_{c in tile} a[c] * b[c]`),
///   which is associative and commutative.
/// - **Level 2 (Outer CUDA-Core Promotion):** Each tile's integer sum is promoted to `f32`,
///   multiplied by the combined sub-block scale `lhs_scale[m, kb] * rhs_scale[nb, kb]`, and all
///   `k_blocks` tile sums are reduced through [`repops::tree_sum_f32`].
#[inline]
pub fn two_level_scaled_dot(
    lhs_row: &[i8],
    lhs_row_scales: &[f32],
    rhs_row: &[i8],
    rhs_nb_scales: &[f32],
    k: usize,
    block_size: usize,
) -> f32 {
    let k_blocks = ceil_div(k, block_size);
    let mut tile_promotions = Vec::with_capacity(k_blocks);
    for kb in 0..k_blocks {
        let k_start = kb * block_size;
        let k_end = (k_start + block_size).min(k);
        let mut inner_acc: i32 = 0;
        for c in k_start..k_end {
            inner_acc += (lhs_row[c] as i32) * (rhs_row[c] as i32);
        }
        let combined_scale = lhs_row_scales[kb] * rhs_nb_scales[kb];
        tile_promotions.push((inner_acc as f32) * combined_scale);
    }
    repops::tree_sum_f32(&tile_promotions)
}

/// Standard fine-grained block-scaled `NT` GEMM (`D = A @ B^T`), after DeepGEMM's
/// `gemm_fp8_fp8_bf16_nt`. Returns a row-major `[lhs.rows, rhs.n]` matrix.
pub fn gemm_fp8_fp8_bf16_nt(
    lhs: &PerTokenBlockQuantized,
    rhs: &PerBlockQuantizedMatrix,
) -> Vec<f32> {
    assert_eq!(lhs.k, rhs.k, "DeepGEMM: K dimension mismatch");
    assert_eq!(
        lhs.block_size, rhs.block_size,
        "DeepGEMM: block_size mismatch"
    );
    let m = lhs.rows;
    let n = rhs.n;
    let k = lhs.k;
    let block_size = lhs.block_size;
    let k_blocks = lhs.k_blocks();
    let aligned_lhs_scales = lhs.tma_aligned_scales();

    let mut out = vec![0.0f32; m * n];
    let mut lhs_scales_row = vec![0.0f32; k_blocks];
    for row_m in 0..m {
        for (kb, dst) in lhs_scales_row.iter_mut().enumerate() {
            *dst = aligned_lhs_scales.get(row_m, kb);
        }
        let a_codes = lhs.row_codes(row_m);
        for col_n in 0..n {
            let nb = col_n / block_size;
            let rhs_scales = &rhs.scales[nb * k_blocks..(nb + 1) * k_blocks];
            let b_codes = rhs.row_codes(col_n);
            out[row_m * n + col_n] =
                two_level_scaled_dot(a_codes, &lhs_scales_row, b_codes, rhs_scales, k, block_size);
        }
    }
    out
}

/// Contiguous `M`-grouped GEMM (`m_grouped_gemm_fp8_fp8_bf16_nt_contiguous`):
///
/// - `lhs`: `[total_m, K]` quantized activations where tokens routed to the same expert occupy
///   contiguous rows (optionally padded to a multiple of `block_m` per expert).
/// - `rhs_groups`: slice of `[N, K]` quantized weight matrices, one per expert group.
/// - `m_indices`: length `total_m`; entry `r` is the expert group index `0..rhs_groups.len()`
///   for row `r`, or `-1` for alignment padding rows (which produce zeros).
pub fn m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
    lhs: &PerTokenBlockQuantized,
    rhs_groups: &[PerBlockQuantizedMatrix],
    m_indices: &[i32],
) -> Vec<f32> {
    assert_eq!(
        lhs.rows,
        m_indices.len(),
        "DeepGEMM contiguous: lhs.rows must equal m_indices.len()"
    );
    let Some(first_rhs) = rhs_groups.first() else {
        return Vec::new();
    };
    let n = first_rhs.n;
    let k = lhs.k;
    let block_size = lhs.block_size;
    let k_blocks = lhs.k_blocks();
    let aligned_lhs_scales = lhs.tma_aligned_scales();

    let mut out = vec![0.0f32; lhs.rows * n];
    let mut lhs_scales_row = vec![0.0f32; k_blocks];

    for (row_m, &group_idx) in m_indices.iter().enumerate() {
        if group_idx < 0 {
            continue;
        }
        let Some(rhs) = rhs_groups.get(group_idx as usize) else {
            continue;
        };
        assert_eq!(rhs.n, n, "DeepGEMM contiguous: uniform N required");
        assert_eq!(rhs.k, k, "DeepGEMM contiguous: uniform K required");

        for (kb, dst) in lhs_scales_row.iter_mut().enumerate() {
            *dst = aligned_lhs_scales.get(row_m, kb);
        }
        let a_codes = lhs.row_codes(row_m);
        for col_n in 0..n {
            let nb = col_n / block_size;
            let rhs_scales = &rhs.scales[nb * k_blocks..(nb + 1) * k_blocks];
            let b_codes = rhs.row_codes(col_n);
            out[row_m * n + col_n] =
                two_level_scaled_dot(a_codes, &lhs_scales_row, b_codes, rhs_scales, k, block_size);
        }
    }
    out
}

/// Masked `M`-grouped GEMM (`m_grouped_gemm_fp8_fp8_bf16_nt_masked`):
///
/// Designed for low-latency decode with fixed execution graphs.
/// - `lhs_groups`: one `[max_m, K]` quantized activation tensor per expert group (`num_groups`).
/// - `rhs_groups`: one `[N, K]` quantized weight matrix per expert group (`num_groups`).
/// - `masked_m`: length `num_groups`; entry `g` is the dynamic count of valid tokens `<= max_m`
///   routed to group `g`. Rows `masked_m[g]..max_m` are masked out and written as `0.0`.
///
/// Returns `[num_groups * max_m * N]` in row-major `[group, m, n]` layout.
pub fn m_grouped_gemm_fp8_fp8_bf16_nt_masked(
    lhs_groups: &[PerTokenBlockQuantized],
    rhs_groups: &[PerBlockQuantizedMatrix],
    masked_m: &[u32],
) -> Vec<f32> {
    assert_eq!(lhs_groups.len(), rhs_groups.len());
    assert_eq!(lhs_groups.len(), masked_m.len());
    let Some(first_lhs) = lhs_groups.first() else {
        return Vec::new();
    };
    let Some(first_rhs) = rhs_groups.first() else {
        return Vec::new();
    };
    let num_groups = lhs_groups.len();
    let max_m = first_lhs.rows;
    let n = first_rhs.n;
    let k = first_lhs.k;
    let block_size = first_lhs.block_size;
    let k_blocks = first_lhs.k_blocks();

    let mut out = vec![0.0f32; num_groups * max_m * n];
    let mut lhs_scales_row = vec![0.0f32; k_blocks];

    for g in 0..num_groups {
        let lhs = &lhs_groups[g];
        let rhs = &rhs_groups[g];
        assert_eq!(lhs.rows, max_m);
        assert_eq!(lhs.k, k);
        assert_eq!(rhs.n, n);
        assert_eq!(rhs.k, k);

        let valid_m = (masked_m[g] as usize).min(max_m);
        let aligned_scales = lhs.tma_aligned_scales();
        let group_out_offset = g * max_m * n;

        for row_m in 0..valid_m {
            for (kb, dst) in lhs_scales_row.iter_mut().enumerate() {
                *dst = aligned_scales.get(row_m, kb);
            }
            let a_codes = lhs.row_codes(row_m);
            let row_out_offset = group_out_offset + row_m * n;
            for col_n in 0..n {
                let nb = col_n / block_size;
                let rhs_scales = &rhs.scales[nb * k_blocks..(nb + 1) * k_blocks];
                let b_codes = rhs.row_codes(col_n);
                out[row_out_offset + col_n] = two_level_scaled_dot(
                    a_codes,
                    &lhs_scales_row,
                    b_codes,
                    rhs_scales,
                    k,
                    block_size,
                );
            }
        }
    }
    out
}

/// One expert's three SwiGLU matrices stored as fine-grained 2D block-scaled tensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepGemmSwiGluExpert {
    pub expert_id: u32,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub w_gate: PerBlockQuantizedMatrix,
    pub w_up: PerBlockQuantizedMatrix,
    pub w_down: PerBlockQuantizedMatrix,
}

impl DeepGemmSwiGluExpert {
    // Keep the public constructor aligned with the three projection matrices and their shape.
    #[allow(clippy::too_many_arguments)]
    pub fn from_fp32_weights(
        expert_id: u32,
        hidden_dim: usize,
        intermediate_dim: usize,
        block_size: usize,
        format: ScaleFormat,
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
    ) -> Self {
        Self {
            expert_id,
            hidden_dim,
            intermediate_dim,
            w_gate: PerBlockQuantizedMatrix::quantize(
                w_gate,
                intermediate_dim,
                hidden_dim,
                block_size,
                format,
            ),
            w_up: PerBlockQuantizedMatrix::quantize(
                w_up,
                intermediate_dim,
                hidden_dim,
                block_size,
                format,
            ),
            w_down: PerBlockQuantizedMatrix::quantize(
                w_down,
                hidden_dim,
                intermediate_dim,
                block_size,
                format,
            ),
        }
    }

    /// Deterministic synthetic expert weights for `(layer, expert_id)` using RepOps transcendentals.
    pub fn synthetic(
        layer: u32,
        expert_id: u32,
        hidden_dim: usize,
        intermediate_dim: usize,
        block_size: usize,
        format: ScaleFormat,
    ) -> Self {
        let seed_base = (layer as usize) * 31 + (expert_id as usize) * 17;
        let w_gate: Vec<f32> = (0..(intermediate_dim * hidden_dim))
            .map(|i| repops::sin(((i + seed_base) as f32) * 0.017) * 0.25)
            .collect();
        let w_up: Vec<f32> = (0..(intermediate_dim * hidden_dim))
            .map(|i| repops::cos(((i + seed_base) as f32) * 0.023) * 0.25)
            .collect();
        let w_down: Vec<f32> = (0..(hidden_dim * intermediate_dim))
            .map(|i| repops::sin(((i + seed_base) as f32) * 0.013) * 0.25)
            .collect();
        Self::from_fp32_weights(
            expert_id,
            hidden_dim,
            intermediate_dim,
            block_size,
            format,
            &w_gate,
            &w_up,
            &w_down,
        )
    }

    /// Runs a single row `x` (`[hidden_dim]`) through `gate * Down(SiLU(Gate(x)) * Up(x))`.
    pub fn forward_row(&self, x: &[f32], gating_weight: f32) -> Vec<f32> {
        let q_x = PerTokenBlockQuantized::quantize(
            x,
            1,
            self.hidden_dim,
            self.w_gate.block_size,
            self.w_gate.format,
        );
        let gate_proj = gemm_fp8_fp8_bf16_nt(&q_x, &self.w_gate);
        let up_proj = gemm_fp8_fp8_bf16_nt(&q_x, &self.w_up);
        let inter: Vec<f32> = gate_proj
            .into_iter()
            .zip(up_proj)
            .map(|(g, u)| repops::silu(g) * u)
            .collect();
        let q_inter = PerTokenBlockQuantized::quantize(
            &inter,
            1,
            self.intermediate_dim,
            self.w_down.block_size,
            self.w_down.format,
        );
        gemm_fp8_fp8_bf16_nt(&q_inter, &self.w_down)
            .into_iter()
            .map(|v| v * gating_weight)
            .collect()
    }
}

/// A pool of resident MoE experts on one worker executed via DeepGEMM's `M`-grouped kernels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepGemmExpertPool {
    pub layer: u32,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub block_size: usize,
    pub format: ScaleFormat,
    pub experts: Vec<DeepGemmSwiGluExpert>,
}

impl DeepGemmExpertPool {
    pub fn new(
        layer: u32,
        hidden_dim: usize,
        intermediate_dim: usize,
        block_size: usize,
        format: ScaleFormat,
        experts: Vec<DeepGemmSwiGluExpert>,
    ) -> Self {
        Self {
            layer,
            hidden_dim,
            intermediate_dim,
            block_size,
            format,
            experts,
        }
    }

    /// Builds a synthetic pool for `expert_ids` on `layer`.
    pub fn synthetic(
        layer: u32,
        expert_ids: &[u32],
        hidden_dim: usize,
        intermediate_dim: usize,
        block_size: usize,
        format: ScaleFormat,
    ) -> Self {
        let experts = expert_ids
            .iter()
            .map(|&id| {
                DeepGemmSwiGluExpert::synthetic(
                    layer,
                    id,
                    hidden_dim,
                    intermediate_dim,
                    block_size,
                    format,
                )
            })
            .collect();
        Self::new(
            layer,
            hidden_dim,
            intermediate_dim,
            block_size,
            format,
            experts,
        )
    }

    /// Reference per-row, per-expert execution (`gemm_fp8_fp8_bf16_nt` per assignment).
    pub fn forward_individual(
        &self,
        rows: &[Vec<f32>],
        routing: &[Vec<MoERoutedExpertSelection>],
    ) -> Vec<(u32, u32, Vec<f32>)> {
        let mut results = Vec::new();
        for (row_idx, selections) in routing.iter().enumerate() {
            let Some(x) = rows.get(row_idx) else {
                continue;
            };
            for sel in selections {
                if let Some(expert) = self.experts.iter().find(|e| e.expert_id == sel.expert_id) {
                    results.push((
                        row_idx as u32,
                        sel.expert_id,
                        expert.forward_row(x, sel.gating_weight),
                    ));
                }
            }
        }
        results.sort_by_key(|(r, e, _)| (*r, *e));
        results
    }

    /// Executes all routed experts across `rows` using DeepGEMM's **contiguous `M`-grouped**
    /// layout (`m_grouped_gemm_fp8_fp8_bf16_nt_contiguous`), padding each expert's contiguous
    /// row segment to a multiple of `block_m` with `m_indices = -1`.
    pub fn forward_contiguous_grouped(
        &self,
        rows: &[Vec<f32>],
        routing: &[Vec<MoERoutedExpertSelection>],
        block_m: usize,
    ) -> Vec<(u32, u32, Vec<f32>)> {
        let block_m = block_m.max(1);
        let w_gates: Vec<PerBlockQuantizedMatrix> =
            self.experts.iter().map(|e| e.w_gate.clone()).collect();
        let w_ups: Vec<PerBlockQuantizedMatrix> =
            self.experts.iter().map(|e| e.w_up.clone()).collect();
        let w_downs: Vec<PerBlockQuantizedMatrix> =
            self.experts.iter().map(|e| e.w_down.clone()).collect();

        // Group assignments by local expert index `0..self.experts.len()`.
        let mut packed_x: Vec<f32> = Vec::new();
        let mut m_indices: Vec<i32> = Vec::new();
        // Track `(packed_row_slot, original_row, expert_id, gating_weight)`.
        let mut valid_slots: Vec<(usize, u32, u32, f32)> = Vec::new();

        for (group_idx, expert) in self.experts.iter().enumerate() {
            let mut group_count = 0usize;
            for (row_idx, selections) in routing.iter().enumerate() {
                if let Some(sel) = selections.iter().find(|s| s.expert_id == expert.expert_id)
                    && let Some(x) = rows.get(row_idx)
                {
                    let slot = m_indices.len();
                    packed_x.extend_from_slice(x);
                    m_indices.push(group_idx as i32);
                    valid_slots.push((slot, row_idx as u32, expert.expert_id, sel.gating_weight));
                    group_count += 1;
                }
            }
            if group_count > 0 {
                let padded = align_up(group_count, block_m);
                for _ in group_count..padded {
                    packed_x.extend(std::iter::repeat_n(0.0f32, self.hidden_dim));
                    m_indices.push(-1);
                }
            }
        }

        if m_indices.is_empty() {
            return Vec::new();
        }

        let total_m = m_indices.len();
        let q_lhs = PerTokenBlockQuantized::quantize(
            &packed_x,
            total_m,
            self.hidden_dim,
            self.block_size,
            self.format,
        );

        // 1. Grouped Gate & Up projections.
        let gate_out = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(&q_lhs, &w_gates, &m_indices);
        let up_out = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(&q_lhs, &w_ups, &m_indices);

        // 2. Elementwise RepOps SiLU(gate) * up.
        let inter: Vec<f32> = gate_out
            .into_iter()
            .zip(up_out)
            .map(|(g, u)| repops::silu(g) * u)
            .collect();
        let q_inter = PerTokenBlockQuantized::quantize(
            &inter,
            total_m,
            self.intermediate_dim,
            self.block_size,
            self.format,
        );

        // 3. Grouped Down projection.
        let down_out = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(&q_inter, &w_downs, &m_indices);

        let mut results: Vec<(u32, u32, Vec<f32>)> = valid_slots
            .into_iter()
            .map(|(slot, row_idx, expert_id, gate)| {
                let slice = &down_out[slot * self.hidden_dim..(slot + 1) * self.hidden_dim];
                let scaled = slice.iter().map(|&v| v * gate).collect();
                (row_idx, expert_id, scaled)
            })
            .collect();
        results.sort_by_key(|(r, e, _)| (*r, *e));
        results
    }

    /// Executes all routed experts across `rows` using DeepGEMM's **masked `M`-grouped** layout
    /// (`m_grouped_gemm_fp8_fp8_bf16_nt_masked`) with a fixed `max_m` capacity per expert group
    /// and a device-side `masked_m` token count array.
    pub fn forward_masked_grouped(
        &self,
        rows: &[Vec<f32>],
        routing: &[Vec<MoERoutedExpertSelection>],
        max_m: usize,
    ) -> Vec<(u32, u32, Vec<f32>)> {
        let max_m = max_m.max(rows.len()).max(1);
        let num_groups = self.experts.len();
        if num_groups == 0 {
            return Vec::new();
        }

        let w_gates: Vec<PerBlockQuantizedMatrix> =
            self.experts.iter().map(|e| e.w_gate.clone()).collect();
        let w_ups: Vec<PerBlockQuantizedMatrix> =
            self.experts.iter().map(|e| e.w_up.clone()).collect();
        let w_downs: Vec<PerBlockQuantizedMatrix> =
            self.experts.iter().map(|e| e.w_down.clone()).collect();

        let mut lhs_groups = Vec::with_capacity(num_groups);
        let mut masked_m = Vec::with_capacity(num_groups);
        // Track `(group_idx, local_m, original_row, expert_id, gating_weight)`.
        let mut valid_slots: Vec<(usize, usize, u32, u32, f32)> = Vec::new();

        for (group_idx, expert) in self.experts.iter().enumerate() {
            let mut group_x = vec![0.0f32; max_m * self.hidden_dim];
            let mut count = 0usize;
            for (row_idx, selections) in routing.iter().enumerate() {
                if let Some(sel) = selections.iter().find(|s| s.expert_id == expert.expert_id)
                    && let Some(x) = rows.get(row_idx)
                    && count < max_m
                {
                    group_x[count * self.hidden_dim..(count + 1) * self.hidden_dim]
                        .copy_from_slice(x);
                    valid_slots.push((
                        group_idx,
                        count,
                        row_idx as u32,
                        expert.expert_id,
                        sel.gating_weight,
                    ));
                    count += 1;
                }
            }
            lhs_groups.push(PerTokenBlockQuantized::quantize(
                &group_x,
                max_m,
                self.hidden_dim,
                self.block_size,
                self.format,
            ));
            masked_m.push(count as u32);
        }

        let gate_out = m_grouped_gemm_fp8_fp8_bf16_nt_masked(&lhs_groups, &w_gates, &masked_m);
        let up_out = m_grouped_gemm_fp8_fp8_bf16_nt_masked(&lhs_groups, &w_ups, &masked_m);

        let mut inter_groups = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let offset = g * max_m * self.intermediate_dim;
            let len = max_m * self.intermediate_dim;
            let inter_slice: Vec<f32> = gate_out[offset..offset + len]
                .iter()
                .zip(&up_out[offset..offset + len])
                .map(|(&gv, &uv)| repops::silu(gv) * uv)
                .collect();
            inter_groups.push(PerTokenBlockQuantized::quantize(
                &inter_slice,
                max_m,
                self.intermediate_dim,
                self.block_size,
                self.format,
            ));
        }

        let down_out = m_grouped_gemm_fp8_fp8_bf16_nt_masked(&inter_groups, &w_downs, &masked_m);

        let mut results: Vec<(u32, u32, Vec<f32>)> = valid_slots
            .into_iter()
            .map(|(g, local_m, row_idx, expert_id, gate)| {
                let offset = (g * max_m + local_m) * self.hidden_dim;
                let slice = &down_out[offset..offset + self.hidden_dim];
                let scaled = slice.iter().map(|&v| v * gate).collect();
                (row_idx, expert_id, scaled)
            })
            .collect();
        results.sort_by_key(|(r, e, _)| (*r, *e));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rows(count: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|r| {
                (0..dim)
                    .map(|i| repops::sin(((r * 19 + i * 7 + 3) as f32) * 0.07) * 0.9)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn ue8m0_snaps_upward_to_exact_powers_of_two() {
        assert_eq!(ceil_to_ue8m0(1.0), 1.0);
        assert_eq!(ceil_to_ue8m0(0.5), 0.5);
        assert_eq!(ceil_to_ue8m0(0.5001), 1.0);
        assert_eq!(ceil_to_ue8m0(0.26), 0.5);
        assert_eq!(ceil_to_ue8m0(3.1), 4.0);
        for &val in &[0.00390625f32, 0.125, 0.75, 1.5, 12.3] {
            let snapped = ceil_to_ue8m0(val);
            assert!(snapped >= val);
            let byte = ue8m0_biased_byte(val);
            assert_eq!(ue8m0_to_f32(byte), snapped);
        }
    }

    #[test]
    fn tma_col_major_scales_align_and_roundtrip() {
        // 5 rows -> aligned to 8 rows (multiple of 4), 3 k_blocks.
        let row_major: Vec<f32> = (0..15).map(|i| (i + 1) as f32 * 0.25).collect();
        let aligned = ColMajorAlignedScales::from_row_major(&row_major, 5, 3);
        assert_eq!(aligned.rows, 5);
        assert_eq!(aligned.aligned_rows, 8);
        assert_eq!(aligned.k_blocks, 3);
        assert_eq!(aligned.data.len(), 24);
        assert_eq!(aligned.to_row_major(), row_major);
        // Padding slots 5..8 in every column are zero.
        for kb in 0..3 {
            for r in 5..8 {
                assert_eq!(aligned.data[kb * 8 + r], 0.0);
            }
        }
    }

    #[test]
    fn contiguous_and_masked_grouped_gemm_match_individual_and_are_batch_invariant() {
        use crate::ExpertResidencyTier;
        let hidden_dim = 32usize;
        let inter_dim = 64usize;
        let rows = sample_rows(4, hidden_dim);
        let routing = vec![
            vec![
                MoERoutedExpertSelection {
                    expert_id: 2,
                    gating_weight: 0.65,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                },
                MoERoutedExpertSelection {
                    expert_id: 0,
                    gating_weight: 0.35,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                },
            ],
            vec![
                MoERoutedExpertSelection {
                    expert_id: 1,
                    gating_weight: 0.55,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                },
                MoERoutedExpertSelection {
                    expert_id: 2,
                    gating_weight: 0.45,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                },
            ],
            vec![
                MoERoutedExpertSelection {
                    expert_id: 0,
                    gating_weight: 0.80,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                },
                MoERoutedExpertSelection {
                    expert_id: 3,
                    gating_weight: 0.20,
                    residency: ExpertResidencyTier::WarmSystemRam,
                },
            ],
            vec![
                MoERoutedExpertSelection {
                    expert_id: 2,
                    gating_weight: 0.50,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                },
                MoERoutedExpertSelection {
                    expert_id: 1,
                    gating_weight: 0.50,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                },
            ],
        ];

        for format in [ScaleFormat::ExactF32, ScaleFormat::Ue8m0PowerOfTwo] {
            let pool = DeepGemmExpertPool::synthetic(
                5,
                &[0, 1, 2, 3],
                hidden_dim,
                inter_dim,
                DEFAULT_SUBBLOCK_SIZE,
                format,
            );

            let individual = pool.forward_individual(&rows, &routing);
            let contiguous_b1 = pool.forward_contiguous_grouped(&rows, &routing, 1);
            let contiguous_b4 = pool.forward_contiguous_grouped(&rows, &routing, 4);
            let contiguous_b8 = pool.forward_contiguous_grouped(&rows, &routing, 8);
            let masked_m4 = pool.forward_masked_grouped(&rows, &routing, 4);
            let masked_m12 = pool.forward_masked_grouped(&rows, &routing, 12);

            assert_eq!(individual.len(), 8);
            assert_eq!(contiguous_b1, individual);
            assert_eq!(contiguous_b4, individual);
            assert_eq!(contiguous_b8, individual);
            assert_eq!(masked_m4, individual);
            assert_eq!(masked_m12, individual);

            // Batch invariance: decoding each row alone through contiguous or masked grouped
            // kernels produces the exact same bytes as decoding it inside the 4-row batch.
            for r in 0..rows.len() {
                let solo_row = std::slice::from_ref(&rows[r]);
                let solo_routing = std::slice::from_ref(&routing[r]);
                let solo_contig = pool.forward_contiguous_grouped(solo_row, solo_routing, 4);
                let solo_masked = pool.forward_masked_grouped(solo_row, solo_routing, 4);
                assert_eq!(solo_contig, solo_masked);
                for (_, expert_id, solo_bytes) in solo_contig {
                    let batched = individual
                        .iter()
                        .find(|(br, be, _)| *br == r as u32 && *be == expert_id)
                        .expect("matching (row, expert) in batch");
                    assert_eq!(&batched.2, &solo_bytes);
                }
            }
        }
    }
}
