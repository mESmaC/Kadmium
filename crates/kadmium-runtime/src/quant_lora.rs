//! # `kadmium_runtime::quant_lora`
//!
//! 4-bit group-quantized matrix-vector kernels (`Q4KBlockTensor`), Colibri double-buffered
//! NVMe DMA prefetching (`ColibriNvmeDoubleBuffer`), and distributed QLoRA forward/backward
//! adapter updates (`DistributedQLoRAAdapter`).

use crate::repops;
use serde::{Deserialize, Serialize};

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
            .map(|i| (((i * 17 + 3) % 19) as f32 - 9.0) * 0.002)
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

    /// Computes forward output $y = W_{\text{Q4K}} x + \frac{\alpha}{r} B (A x)$ and intermediate $h = A x$.
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), self.d_in);
        let mut y = self.frozen_base_q4k.matvec_f32(x);
        let scale = self.alpha / (self.rank.max(1) as f32);

        let mut h = vec![0.0f32; self.rank];
        for (r, h_val) in h.iter_mut().enumerate() {
            *h_val = repops::tree_dot_f32(&self.lora_a[r * self.d_in..(r + 1) * self.d_in], x);
        }

        for (o, y_val) in y.iter_mut().enumerate() {
            let delta = repops::tree_dot_f32(&self.lora_b[o * self.rank..(o + 1) * self.rank], &h);
            *y_val += scale * delta;
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
