//! # `kadmium_runtime::quant_lora`
//!
//! 4-bit group-quantized matrix-vector kernels (`Q4KBlockTensor`), Ollama/GGML-style `Q8_K`
//! activation block quantization with integer dot-product kernels (`Q8KBlockVector`,
//! `vec_dot_q4_K_q8_K`), Colibri double-buffered NVMe DMA prefetching (`ColibriNvmeDoubleBuffer`),
//! and distributed QLoRA forward/backward adapter updates (`DistributedQLoRAAdapter`).

use crate::repops;
use serde::{Deserialize, Serialize};

/// Ollama / GGML `block_q8_K` (32-element sub-block) quantized activation vector.
///
/// When multiplying a `Q4_K` weight matrix by an activation vector $x \in \mathbb{R}^K$,
/// Ollama's GGML backend (`vec_dot_q4_K_q8_K`) quantizes $x$ **once** into 32-element `i8`
/// blocks with scale $d_{x,b}$ and precomputed integer sum $\text{bsum}_{x,b} = \sum_{j=0}^{31} q_{x,b,j}$.
/// Every row $r$ of the weight matrix then evaluates each 32-element block via an exact `i32`
/// integer dot product without unpacking weights to `f32`:
/// $$\sum_{j=0}^{31} (\text{min}_{r,b} + \text{scale}_{r,b} q_{r,b,j}) (d_{x,b} q_{x,b,j})
///   = d_{x,b} \left(\text{min}_{r,b} \cdot \text{bsum}_{x,b} + \text{scale}_{r,b} \sum_{j=0}^{31} q_{r,b,j} q_{x,b,j}\right)$$
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Q8KBlockVector {
    pub len: usize,
    pub block_size: usize,
    /// Per-block symmetric scale $d_{x,b} = \max_j |x_{b,j}| / 127$.
    pub scales: Vec<f32>,
    /// Quantized signed `i8` codes in `[-127, 127]`.
    pub qs: Vec<i8>,
    /// Precomputed block sums $\text{bsum}_{x,b} = \sum_{j \in b} q_{x,b,j}$ in `i32`
    /// so the `Q4_K` affine `min_val` offset is evaluated in $O(1)$ per 32-element block.
    pub bsums: Vec<i32>,
}

impl Q8KBlockVector {
    pub const BLOCK_SIZE: usize = 32;

    /// Quantizes a slice `input` into 32-element `Q8_K` blocks with precomputed `bsums`.
    pub fn quantize(input: &[f32]) -> Self {
        let block_size = Self::BLOCK_SIZE;
        let num_blocks = input.len().div_ceil(block_size);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut qs = Vec::with_capacity(input.len());
        let mut bsums = Vec::with_capacity(num_blocks);

        for chunk in input.chunks(block_size) {
            let absmax = chunk
                .iter()
                .copied()
                .fold(0.0f32, |acc, v| acc.max(v.abs()));
            if absmax == 0.0 || !absmax.is_finite() {
                scales.push(0.0);
                bsums.push(0);
                qs.extend(std::iter::repeat_n(0i8, chunk.len()));
            } else {
                let d = absmax / 127.0;
                let inv_d = 127.0 / absmax;
                let mut bsum = 0i32;
                for &v in chunk {
                    let q = (v * inv_d).round().clamp(-127.0, 127.0) as i8;
                    bsum += q as i32;
                    qs.push(q);
                }
                scales.push(d);
                bsums.push(bsum);
            }
        }

        Self {
            len: input.len(),
            block_size,
            scales,
            qs,
            bsums,
        }
    }

    /// Reconstructs the `f32` vector $x_j = d_{x,b} \cdot q_{x,b,j}$.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        for (b, chunk) in self.qs.chunks(self.block_size).enumerate() {
            let d = self.scales[b];
            for &q in chunk {
                out.push(d * (q as f32));
            }
        }
        out
    }
}

/// 4-bit Group-Quantized (`Q4_K` / 32-element block) Tensor for `.ktb` weight banks.
///
/// Packs every 2 quantized 4-bit values (`0..=15`) into a single byte (`16 bytes` per 32 weights)
/// alongside per-block `(scale, min_val)` affine parameters:
/// $$w_i \approx \text{min\_val}_b + \text{scale}_b \cdot q_i, \quad q_i \in \{0, \dots, 15\}$$
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q4KBlockTensor {
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    /// Packed 4-bit nibbles: `low_nibble = byte & 0x0F`, `high_nibble = (byte >> 4) & 0x0F`.
    pub packed_nibbles: Vec<u8>,
    pub block_scales: Vec<f32>,
    pub block_mins: Vec<f32>,
}

impl Q4KBlockTensor {
    /// Quantizes a row-major `rows x cols` FP32 matrix into 4-bit 32-element blocks.
    pub fn quantize_matrix(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);
        let block_size = 32usize;
        let num_blocks = weights.len().div_ceil(block_size);
        let mut packed_nibbles = Vec::with_capacity(weights.len().div_ceil(2));
        let mut block_scales = Vec::with_capacity(num_blocks);
        let mut block_mins = Vec::with_capacity(num_blocks);

        for chunk in weights.chunks(block_size) {
            let min_v = chunk.iter().copied().fold(f32::INFINITY, f32::min);
            let max_v = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = (max_v - min_v).max(1e-8);
            let scale = range / 15.0;
            block_mins.push(min_v);
            block_scales.push(scale);

            for pair in chunk.chunks(2) {
                let q0 = (((pair[0] - min_v) / scale).round().clamp(0.0, 15.0)) as u8;
                let q1 = if pair.len() == 2 {
                    (((pair[1] - min_v) / scale).round().clamp(0.0, 15.0)) as u8
                } else {
                    0
                };
                packed_nibbles.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            }
        }

        Self {
            rows,
            cols,
            block_size,
            packed_nibbles,
            block_scales,
            block_mins,
        }
    }

    /// Computes fused on-the-fly 4-bit dequantization + matrix-vector multiplication $y = W x$
    /// without materializing the full FP32 matrix in memory. Each row is a RepOps dot product:
    /// one rounding per product, then the fixed reduce tree.
    pub fn matvec_f32(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols);
        let mut output = vec![0.0f32; self.rows];

        for (r, out_val) in output.iter_mut().enumerate() {
            let base = r * self.cols;
            let products: Vec<f32> = input
                .iter()
                .enumerate()
                .map(|(c, &x_val)| {
                    let global_idx = base + c;
                    let block_idx = global_idx / self.block_size;
                    let scale = self.block_scales[block_idx];
                    let min_v = self.block_mins[block_idx];
                    let byte = self.packed_nibbles[global_idx / 2];
                    let q = if global_idx.is_multiple_of(2) {
                        (byte & 0x0F) as f32
                    } else {
                        ((byte >> 4) & 0x0F) as f32
                    };
                    let w = min_v + scale * q;
                    w * x_val
                })
                .collect();
            *out_val = repops::tree_sum_f32(&products);
        }
        output
    }

    /// Ollama / GGML `vec_dot_q4_K_q8_K` kernel for a single row `row` against a pre-quantized
    /// `Q8KBlockVector`.
    ///
    /// When `self.cols` is a multiple of `block_size` (`32`), each 32-element block is aligned
    /// with `x_q8`'s blocks, so the inner 32-element reduction runs entirely in `i32` integer
    /// arithmetic (`q_w * q_x`) and uses the precomputed `x_q8.bsums[b]` for the affine offset,
    /// reducing the number of `f32` tree-reduction leaves by `32x`.
    pub fn vec_dot_q4k_q8k(&self, row: usize, x_q8: &Q8KBlockVector) -> f32 {
        assert!(row < self.rows);
        assert_eq!(x_q8.len, self.cols);
        assert_eq!(self.block_size, Q8KBlockVector::BLOCK_SIZE);

        let base = row * self.cols;
        let bs = self.block_size;

        // Fast aligned path (`cols % 32 == 0` and `base % 32 == 0`), matching GGML's row-aligned
        // `block_q4_K` x `block_q8_K` integer dot product.
        if self.cols.is_multiple_of(bs) {
            let num_row_blocks = self.cols / bs;
            let first_block = base / bs;
            let mut block_terms = Vec::with_capacity(num_row_blocks);

            for b in 0..num_row_blocks {
                let d_x = x_q8.scales[b];
                if d_x == 0.0 {
                    block_terms.push(0.0);
                    continue;
                }
                let w_block = first_block + b;
                let scale_w = self.block_scales[w_block];
                let min_w = self.block_mins[w_block];
                let bsum_x = x_q8.bsums[b];

                let byte_base = (w_block * bs) / 2;
                let x_base = b * bs;
                let mut q_dot_i32 = 0i32;
                for pair_idx in 0..(bs / 2) {
                    let byte = self.packed_nibbles[byte_base + pair_idx];
                    let q0 = (byte & 0x0F) as i32;
                    let q1 = ((byte >> 4) & 0x0F) as i32;
                    let x0 = x_q8.qs[x_base + pair_idx * 2] as i32;
                    let x1 = x_q8.qs[x_base + pair_idx * 2 + 1] as i32;
                    q_dot_i32 += q0 * x0 + q1 * x1;
                }

                let term = d_x * (min_w * (bsum_x as f32) + scale_w * (q_dot_i32 as f32));
                block_terms.push(term);
            }
            return repops::tree_sum_f32(&block_terms);
        }

        // Unaligned tail fallback for arbitrary test dimensions where `cols % 32 != 0`.
        let num_row_blocks = self.cols.div_ceil(bs);
        let mut block_terms = Vec::with_capacity(num_row_blocks);
        for b in 0..num_row_blocks {
            let d_x = x_q8.scales[b];
            if d_x == 0.0 {
                block_terms.push(0.0);
                continue;
            }
            let col_start = b * bs;
            let col_end = (col_start + bs).min(self.cols);
            let mut sub_sum = 0.0f32;
            for c in col_start..col_end {
                let global_idx = base + c;
                let w_block = global_idx / bs;
                let byte = self.packed_nibbles[global_idx / 2];
                let q_w = if global_idx.is_multiple_of(2) {
                    (byte & 0x0F) as i32
                } else {
                    ((byte >> 4) & 0x0F) as i32
                };
                let w = self.block_mins[w_block] + self.block_scales[w_block] * (q_w as f32);
                sub_sum += w * (x_q8.qs[c] as f32);
            }
            block_terms.push(d_x * sub_sum);
        }
        repops::tree_sum_f32(&block_terms)
    }

    /// Matrix-vector product $y = W x$ using Ollama / GGML `vec_dot_q4_K_q8_K` across all rows,
    /// reusing a single pre-quantized `Q8KBlockVector`.
    pub fn matvec_q8k(&self, x_q8: &Q8KBlockVector) -> Vec<f32> {
        assert_eq!(x_q8.len, self.cols);
        (0..self.rows)
            .map(|r| self.vec_dot_q4k_q8k(r, x_q8))
            .collect()
    }

    /// Memory compression ratio compared to raw `f32` (`4 bytes` per element).
    pub fn compression_ratio(&self) -> f64 {
        let fp32_bytes = (self.rows * self.cols * 4) as f64;
        let q4_bytes = (self.packed_nibbles.len()
            + (self.block_scales.len() + self.block_mins.len()) * 4) as f64;
        ((fp32_bytes / q4_bytes.max(1.0)) * 100.0).round() / 100.0
    }
}

/// Telemetry trace for a single double-buffered Colibri NVMe-to-RAM prefetch step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColibriPrefetchStepTrace {
    pub layer_index: u32,
    pub active_compute_slot: &'static str,
    pub background_prefetch_slot: &'static str,
    pub compute_duration_ms: f64,
    pub nvme_dma_duration_ms: f64,
    pub effective_pipeline_stall_ms: f64,
}

/// Colibri Double-Buffered NVMe-to-RAM Prefetch Pipeline.
///
/// Overlaps execution of layer $L$ in `Slot A` with asynchronous 64 KiB page-aligned
/// DMA prefetching of layer $L+1$'s cold routed experts into `Slot B`.
pub struct ColibriNvmeDoubleBuffer;

impl ColibriNvmeDoubleBuffer {
    /// Simulates a multi-layer double-buffered execution stream and returns per-layer overlap traces.
    pub fn execute_double_buffered_stream(
        num_layers: u32,
        cold_expert_bytes_per_layer: usize,
        nvme_bw_gbps: f64,
        compute_ms_per_layer: f64,
    ) -> Vec<ColibriPrefetchStepTrace> {
        let mut traces = Vec::with_capacity(num_layers as usize);
        let bw_bytes_per_ms = (nvme_bw_gbps.max(0.5) * 1e9) / 1000.0;

        for l in 0..num_layers {
            let (active_slot, prefetch_slot) = if l % 2 == 0 {
                ("Slot-A (Compute)", "Slot-B (NVMe DMA)")
            } else {
                ("Slot-B (Compute)", "Slot-A (NVMe DMA)")
            };
            let dma_ms =
                (((cold_expert_bytes_per_layer as f64) / bw_bytes_per_ms) * 100.0).round() / 100.0;
            let stall_ms = ((dma_ms - compute_ms_per_layer).max(0.0) * 100.0).round() / 100.0;

            traces.push(ColibriPrefetchStepTrace {
                layer_index: l,
                active_compute_slot: active_slot,
                background_prefetch_slot: prefetch_slot,
                compute_duration_ms: compute_ms_per_layer,
                nvme_dma_duration_ms: dma_ms,
                effective_pipeline_stall_ms: stall_ms,
            });
        }

        traces
    }
}

/// Distributed QLoRA Adapter Layer combining a frozen 4-bit `Q4KBlockTensor` base weight matrix
/// with trainable low-rank FP32 matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$:
/// $$y = W_{\text{Q4K}} x + \frac{\alpha}{r} B (A x)$$
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedQLoRAAdapter {
    pub d_out: usize,
    pub d_in: usize,
    pub rank: usize,
    pub alpha: f32,
    pub frozen_base_q4k: Q4KBlockTensor,
    /// Row-major `rank x d_in` adapter down-projection.
    pub lora_a: Vec<f32>,
    /// Row-major `d_out x rank` adapter up-projection.
    pub lora_b: Vec<f32>,
}

impl DistributedQLoRAAdapter {
    pub fn new(base_weights: &[f32], d_out: usize, d_in: usize, rank: usize, alpha: f32) -> Self {
        let frozen_base_q4k = Q4KBlockTensor::quantize_matrix(base_weights, d_out, d_in);
        let lora_a = (0..(rank * d_in))
            .map(|i| (((i * 13 + 7) % 31) as f32 - 15.0) * 0.01)
            .collect();
        let lora_b = (0..(d_out * rank))
            .map(|i| (((i * 19 + 3) % 29) as f32 - 14.0) * 0.01)
            .collect();
        Self {
            d_out,
            d_in,
            rank,
            alpha,
            frozen_base_q4k,
            lora_a,
            lora_b,
        }
    }

    /// Executes forward pass $y = W_{\text{Q4K}} x + \frac{\alpha}{r} B (A x)$ and returns `(y, h_mid = A x)`.
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), self.d_in);
        let mut y = self.frozen_base_q4k.matvec_f32(x);
        let mut h = vec![0.0f32; self.rank];

        for (r, h_val) in h.iter_mut().enumerate() {
            let row = &self.lora_a[r * self.d_in..(r + 1) * self.d_in];
            *h_val = repops::tree_dot_f32(row, x);
        }

        let scale = self.alpha / (self.rank.max(1) as f32);
        for (o, y_val) in y.iter_mut().enumerate() {
            let row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
            *y_val += scale * repops::tree_dot_f32(row, &h);
        }

        (y, h)
    }

    /// Executes one analytical backward + SGD step on MSE loss $\mathcal{L} = \frac{1}{2} \|y - y_{\text{target}}\|_2^2$,
    /// returning `(loss_before, flattened_lora_gradients)`.
    pub fn backward_and_step(
        &mut self,
        x: &[f32],
        y_target: &[f32],
        learning_rate: f32,
    ) -> (f32, Vec<f32>) {
        let (y, h) = self.forward(x);
        let scale = self.alpha / (self.rank.max(1) as f32);

        let mut dy = vec![0.0f32; self.d_out];
        let mut loss_terms = vec![0.0f32; self.d_out];
        for o in 0..self.d_out {
            let diff = y[o] - y_target[o];
            dy[o] = diff;
            loss_terms[o] = 0.5 * diff * diff;
        }
        let loss = repops::tree_sum_f32(&loss_terms);

        let mut grad_b = vec![0.0f32; self.d_out * self.rank];
        for (o, &dy_val) in dy.iter().enumerate() {
            for (r, &h_val) in h.iter().enumerate() {
                grad_b[o * self.rank + r] = scale * dy_val * h_val;
            }
        }

        let mut dh = vec![0.0f32; self.rank];
        let mut column = vec![0.0f32; self.d_out];
        for (r, dh_val) in dh.iter_mut().enumerate() {
            for (o, slot) in column.iter_mut().enumerate() {
                *slot = self.lora_b[o * self.rank + r];
            }
            *dh_val = scale * repops::tree_dot_f32(&column, &dy);
        }

        let mut grad_a = vec![0.0f32; self.rank * self.d_in];
        for (r, &dh_val) in dh.iter().enumerate() {
            for (c, &x_val) in x.iter().enumerate() {
                grad_a[r * self.d_in + c] = dh_val * x_val;
            }
        }

        for (b_val, &gb) in self.lora_b.iter_mut().zip(grad_b.iter()) {
            *b_val -= learning_rate * gb;
        }
        for (a_val, &ga) in self.lora_a.iter_mut().zip(grad_a.iter()) {
            *a_val -= learning_rate * ga;
        }

        let mut flat_grads = grad_a;
        flat_grads.extend_from_slice(&grad_b);
        (loss, flat_grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_dot_q4k_q8k_matches_block_algebra_and_tracks_fp32_matvec() {
        let rows = 16usize;
        let cols = 128usize;
        let weights: Vec<f32> = (0..(rows * cols))
            .map(|i| repops::sin((i as f32) * 0.037) * 0.8 - 0.15)
            .collect();
        let x: Vec<f32> = (0..cols)
            .map(|c| repops::cos((c as f32) * 0.053) * 0.9)
            .collect();

        let q4 = Q4KBlockTensor::quantize_matrix(&weights, rows, cols);
        let x_q8 = Q8KBlockVector::quantize(&x);

        let y_f32 = q4.matvec_f32(&x);
        let y_q8k = q4.matvec_q8k(&x_q8);

        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&a, &b) in y_f32.iter().zip(y_q8k.iter()) {
            dot += (a as f64) * (b as f64);
            norm_a += (a as f64) * (a as f64);
            norm_b += (b as f64) * (b as f64);
        }
        let cos_sim = dot / (norm_a.sqrt() * norm_b.sqrt());
        assert!(
            cos_sim > 0.9999,
            "expected cosine similarity > 0.9999 between matvec_f32 and vec_dot_q4k_q8k, got {cos_sim}"
        );
    }
}
