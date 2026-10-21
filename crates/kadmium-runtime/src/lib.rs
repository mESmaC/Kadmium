//! # `kadmium-runtime`
//!
//! Unified runtime orchestrating:
//! 1. **Distributed Inference:** Asymmetric pipeline + MoE expert routing with SBC speculative drafts.
//! 2. **Distributed Training:** 1F1B asymmetric micro-batch scheduling, QLoRA, and DiLoCo sync.
//! 3. **BOINC-Inspired Grid Execution:** Workunit queues, polite host yield, and canary verification.

use kadmium_core::{GradientCompressionCodec, NodeId, TrainingStrategy};
use kadmium_sharding::ClusterShardPlan;
use serde::{Deserialize, Serialize};

pub mod exl3_qtip;
pub mod fraud_proof;
pub mod inference_compute;
pub mod kernels;
pub mod live_worker;
pub mod throughput_boost;
pub mod training_compute;
pub mod training_framework;
pub use exl3_qtip::*;
pub use fraud_proof::*;
pub use inference_compute::*;
pub use kernels::*;
pub use live_worker::*;
pub use throughput_boost::*;
pub use training_compute::*;
pub use training_framework::*;

/// Telemetry snapshot for a running distributed & decentralized training job in Kadmium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobTelemetry {
    pub job_id: String,
    pub model_id: String,
    pub current_step: u64,
    pub total_steps: u64,
    pub loss: f64,
    pub learning_rate: f64,
    /// Number of in-flight 1F1B micro-batches hiding inter-node pipeline latency.
    pub micro_batches_in_flight: u32,
    /// Pipeline bubble overhead percentage (lower is better; asymmetric sharding minimizes this).
    pub pipeline_bubble_pct: f64,
    /// Number of Sharded-Island DiLoCo sub-clusters participating across LAN/WAN.
    pub sharded_diloco_islands: u32,
    /// DiLoCo inner-loop steps per outer-loop Nesterov pseudo-gradient sync.
    pub diloco_inner_step_interval: u32,
    /// Active gradient/momentum wire compression codec (e.g., Nous DeMo DCT or Hivemind 8-bit).
    pub compression_codec: GradientCompressionCodec,
    /// Effective wire bandwidth compression factor (e.g., 864x with DeMo DCT).
    pub gradient_compression_ratio: f64,
    /// Whether Gensyn Verde / RepOps cross-silicon reproducible verification is passing.
    pub verde_repops_verified: bool,
    /// Last step archived to the Home NAS `.ktb` checkpoint vault.
    pub last_nas_checkpoint_step: u64,
}

impl TrainingJobTelemetry {
    pub fn sample_for_plan(plan: &ClusterShardPlan) -> Self {
        let active_stages = plan
            .assignments
            .iter()
            .filter(|a| a.layer_count > 0)
            .count() as u32;
        let micro_batches = (active_stages * 3).max(4);
        // Asymmetric sharding equalizes stage times, keeping the 1F1B bubble small (~6-9%).
        let bubble_pct = (((active_stages.saturating_sub(1) as f64)
            / (micro_batches as f64 + active_stages as f64))
            * 42.0
            * 10.0)
            .round()
            / 10.0;

        let is_hybrid_wan = plan.training_strategy == Some(TrainingStrategy::HybridElasticMesh);
        let (codec, ratio, islands) = if is_hybrid_wan {
            (GradientCompressionCodec::DeMoDctFrequencyTopK, 864.0, 2)
        } else {
            (GradientCompressionCodec::HivemindBlockwise8Bit, 14.8, 1)
        };

        Self {
            job_id: "train-job-llama70b-qlora-01".to_string(),
            model_id: plan.model_id.clone(),
            current_step: 1_420,
            total_steps: 5_000,
            loss: 1.184,
            learning_rate: 1.5e-4,
            micro_batches_in_flight: micro_batches,
            pipeline_bubble_pct: bubble_pct,
            sharded_diloco_islands: islands,
            diloco_inner_step_interval: 50,
            compression_codec: codec,
            gradient_compression_ratio: ratio,
            verde_repops_verified: true,
            last_nas_checkpoint_step: 1_400,
        }
    }
}

/// BOINC + Gensyn Verde / RL-Swarm inspired workunit for verification, RL sandboxes, and batch AI tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoincWorkunit {
    pub workunit_id: String,
    pub title: String,
    pub assigned_node: NodeId,
    pub progress_pct: f64,
    /// Cosine similarity score of RepOps/Verde spot-verification against reference logits ($1.0 = \text{exact}$).
    pub canary_cosine_similarity: f64,
    pub verified: bool,
    pub credits_awarded: f64,
}

/// Generates sample BOINC, Gensyn Verde, and RL-Swarm workunits running across edge/utility nodes.
pub fn sample_boinc_workunits() -> Vec<BoincWorkunit> {
    vec![
        BoincWorkunit {
            workunit_id: "wu-verde-audit-0941".into(),
            title: "Gensyn Verde RepOps Cross-Silicon Audit (CUDA <-> Metal Layer Bisection)"
                .into(),
            assigned_node: NodeId("node-legacy-laptop".into()),
            progress_pct: 100.0,
            canary_cosine_similarity: 0.99998,
            verified: true,
            credits_awarded: 420.5,
        },
        BoincWorkunit {
            workunit_id: "wu-rlswarm-sbx-0942".into(),
            title: "RL-Swarm GRPO Verifiable Code/Math Sandbox (128 Rollout Unit Tests)".into(),
            assigned_node: NodeId("node-sbc-swarm".into()),
            progress_pct: 96.0,
            canary_cosine_similarity: 1.0,
            verified: true,
            credits_awarded: 385.0,
        },
        BoincWorkunit {
            workunit_id: "wu-dataset-pack-0943".into(),
            title: "FineWeb-Edu Tokenization & 64KB Page-Aligned KTB Packing".into(),
            assigned_node: NodeId("node-home-nas".into()),
            progress_pct: 88.0,
            canary_cosine_similarity: 1.0,
            verified: true,
            credits_awarded: 610.0,
        },
    ]
}

// ================================================================================================
// 1. Nous Research `DisTrO / DeMo` Fast-Frequency DCT Momentum Compressor with Error Feedback
// ================================================================================================

/// Sparse frequency-domain packet transmitted over WAN/LAN during a DeMo / DisTrO sync step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseDctPacket {
    pub block_len: usize,
    pub indices: Vec<u16>,
    pub values: Vec<f32>,
}

impl SparseDctPacket {
    /// Wire compression ratio compared to raw `f32` tensor transmission.
    pub fn compression_ratio(&self) -> f64 {
        let raw_bytes = self.block_len * 4;
        let packed_bytes = (self.indices.len() * 2 + self.values.len() * 4).max(1);
        ((raw_bytes as f64) / (packed_bytes as f64) * 10.0).round() / 10.0
    }
}

/// Stateful **DeMo (Decoupled Momentum)** compressor implementing orthonormal `DCT-II` / `IDCT-III`,
/// Top-$k$ fast frequency extraction, and local residual error-feedback accumulation ($e_t$).
#[derive(Debug, Clone)]
pub struct DeMoDctCompressor {
    pub residual_error: Vec<f32>,
    pub momentum: Vec<f32>,
    pub beta: f32,
}

impl DeMoDctCompressor {
    pub fn new(size: usize, beta: f32) -> Self {
        Self {
            residual_error: vec![0.0; size],
            momentum: vec![0.0; size],
            beta,
        }
    }

    /// Orthonormal 1D Discrete Cosine Transform Type-II (`DCT-II`).
    pub fn dct_ii(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        let n_f = n as f32;
        let mut out = vec![0.0f32; n];
        for (k, out_k) in out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let k_f = k as f32;
            for (i, &x) in input.iter().enumerate() {
                let angle = std::f32::consts::PI * ((i as f32) + 0.5) * k_f / n_f;
                sum += x * angle.cos();
            }
            let scale = if k == 0 {
                (1.0 / n_f).sqrt()
            } else {
                (2.0 / n_f).sqrt()
            };
            *out_k = sum * scale;
        }
        out
    }

    /// Orthonormal 1D Inverse Discrete Cosine Transform (`IDCT-III`, exact inverse of `dct_ii`).
    pub fn idct_iii(coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len();
        if n == 0 {
            return Vec::new();
        }
        let n_f = n as f32;
        let scale0 = (1.0 / n_f).sqrt();
        let scale_k = (2.0 / n_f).sqrt();
        let mut out = vec![0.0f32; n];

        for (i, out_i) in out.iter_mut().enumerate() {
            let mut sum = coeffs[0] * scale0;
            let i_half = (i as f32) + 0.5;
            for (k, &c) in coeffs.iter().enumerate().skip(1) {
                if c.abs() > 0.0 {
                    let angle = std::f32::consts::PI * i_half * (k as f32) / n_f;
                    sum += c * scale_k * angle.cos();
                }
            }
            *out_i = sum;
        }
        out
    }

    /// Executes one DeMo step:
    /// 1. Updates momentum $m_t = \beta m_{t-1} + g_t$
    /// 2. Adds accumulated residual error $u_t = m_t + e_{t-1}$
    /// 3. Transforms $u_t$ via `DCT-II` and extracts Top-$k$ highest-magnitude frequencies
    /// 4. Updates residual error $e_t = u_t - \text{IDCT}(c_t)$ so zero gradient energy is lost!
    pub fn step_and_compress(&mut self, grad: &[f32], top_k: usize) -> SparseDctPacket {
        assert_eq!(grad.len(), self.momentum.len());
        let n = grad.len();
        let mut u = vec![0.0f32; n];

        for i in 0..n {
            self.momentum[i] = self.beta * self.momentum[i] + grad[i];
            u[i] = self.momentum[i] + self.residual_error[i];
        }

        let freq = Self::dct_ii(&u);
        let mut ranked: Vec<(usize, f32)> = freq.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let k = top_k.min(n).max(1);
        let mut sparse_freq = vec![0.0f32; n];
        let mut indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);

        for &(idx, val) in ranked.iter().take(k) {
            sparse_freq[idx] = val;
            indices.push(idx as u16);
            values.push(val);
        }

        // Reconstruct transmitted spatial signal and store exact residual in error buffer
        let reconstructed = Self::idct_iii(&sparse_freq);
        for i in 0..n {
            self.residual_error[i] = u[i] - reconstructed[i];
        }

        SparseDctPacket {
            block_len: n,
            indices,
            values,
        }
    }

    /// Decompresses a received `SparseDctPacket` back into a dense spatial momentum update.
    pub fn decompress_packet(packet: &SparseDctPacket) -> Vec<f32> {
        let mut sparse_freq = vec![0.0f32; packet.block_len];
        for (&idx, &val) in packet.indices.iter().zip(packet.values.iter()) {
            if (idx as usize) < packet.block_len {
                sparse_freq[idx as usize] = val;
            }
        }
        Self::idct_iii(&sparse_freq)
    }
}

// ================================================================================================
// 2. Hivemind 8-Bit Blockwise Dynamic Activation Quantizer
// ================================================================================================

/// Blockwise `INT8` quantized activation tensor for low-bandwidth pipeline stage boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockwiseInt8Tensor {
    pub block_size: usize,
    pub total_elements: usize,
    pub absmax_scales: Vec<f32>,
    pub quantized_data: Vec<i8>,
}

impl BlockwiseInt8Tensor {
    /// Quantizes an `f32` activation slice into 8-bit blocks of `block_size` elements.
    pub fn quantize(input: &[f32], block_size: usize) -> Self {
        let bs = block_size.max(8);
        let num_blocks = input.len().div_ceil(bs);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut qdata = Vec::with_capacity(input.len());

        for chunk in input.chunks(bs) {
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            for &v in chunk {
                let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                qdata.push(q);
            }
        }

        Self {
            block_size: bs,
            total_elements: input.len(),
            absmax_scales: scales,
            quantized_data: qdata,
        }
    }

    /// Dequantizes the `INT8` blocks back into `f32` activations.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.total_elements);
        for (block_idx, chunk) in self.quantized_data.chunks(self.block_size).enumerate() {
            let scale = self.absmax_scales.get(block_idx).copied().unwrap_or(1.0);
            for &q in chunk {
                out.push((q as f32) * scale);
            }
        }
        out
    }
}

// ================================================================================================
// 3. Gensyn `Verde / RepOps` Reproducible Tree Reduction & O(log L) Refereed Layer Bisection
// ================================================================================================

/// Outcome of a Gensyn `Verde` interactive $O(\log_2 L)$ refereed layer-bisection audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerdeBisectionOutcome {
    /// Total layers in the audited pipeline shard.
    pub total_layers: usize,
    /// Exact layer index $l^*$ where the worker's output first diverged from the referee's `RepOps` output.
    pub faulty_layer_index: usize,
    /// Number of single-layer referee evaluations performed ($\le \lceil \log_2 L \rceil + 1$).
    pub referee_evaluations_used: usize,
}

/// Cross-silicon reproducible execution (`RepOps`) and $O(\log L)$ layer-bisection verifier.
pub struct VerdeVerificationEngine;

impl VerdeVerificationEngine {
    /// Deterministic binary-tree pairwise summation (`RepOps`), eliminating floating-point
    /// non-associativity drift across CUDA warp reductions, Apple Metal SIMD groups, and CPU threads.
    pub fn repops_pairwise_tree_sum(values: &[f64]) -> f64 {
        match values.len() {
            0 => 0.0,
            1 => values[0],
            2 => values[0] + values[1],
            n => {
                let mid = n.next_power_of_two() / 2;
                let mid = if mid >= n { n / 2 } else { mid };
                Self::repops_pairwise_tree_sum(&values[..mid])
                    + Self::repops_pairwise_tree_sum(&values[mid..])
            }
        }
    }

    /// Runs the Gensyn **Verde Refereed Delegation Game** via $O(\log_2 L)$ binary search over $L$ layers:
    /// 1. Binary-searches the monotone state trajectory `worker_states[0..=L]` against the canary
    ///    reference trajectory `reference_states[0..=L]` in $\lceil \log_2 L \rceil$ steps to find the
    ///    exact first layer $l^*$ where the output state `worker_states[l* + 1]` diverged while
    ///    the input state `worker_states[l*]` still matched.
    /// 2. Executes a single-layer `RepOps` referee evaluation on layer $l^*$ (`referee_layer_fn(l*, &worker_states[l*])`)
    ///    to conclusively verify the faulty transition.
    pub fn bisect_faulty_layer<F>(
        worker_states: &[Vec<f64>],
        reference_states: &[Vec<f64>],
        mut referee_layer_fn: F,
        tolerance_eps: f64,
    ) -> Option<VerdeBisectionOutcome>
    where
        F: FnMut(usize, &[f64]) -> Vec<f64>,
    {
        if worker_states.len() < 2 || worker_states.len() != reference_states.len() {
            return None;
        }
        let total_layers = worker_states.len() - 1;

        let states_diverge = |idx: usize| -> bool {
            worker_states[idx]
                .iter()
                .zip(reference_states[idx].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
                > tolerance_eps
        };

        // If the final output state doesn't diverge at all, all layers passed!
        if !states_diverge(total_layers) {
            return None;
        }

        // Binary search in 1..=total_layers for the first state index `high` that diverges.
        let mut low = 1usize;
        let mut high = total_layers;
        let mut steps = 0usize;

        while low < high {
            let mid = low + (high - low) / 2;
            steps += 1;
            if states_diverge(mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        let faulty_layer = low - 1;
        // Execute the single-layer Referee operator on `faulty_layer` to confirm the dispute:
        steps += 1;
        let referee_out = referee_layer_fn(faulty_layer, &worker_states[faulty_layer]);
        let confirmed_diff = referee_out
            .iter()
            .zip(worker_states[faulty_layer + 1].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        if confirmed_diff > tolerance_eps {
            Some(VerdeBisectionOutcome {
                total_layers,
                faulty_layer_index: faulty_layer,
                referee_evaluations_used: steps,
            })
        } else {
            None
        }
    }
}

pub mod pipeline;
pub mod quant_lora;
pub mod scheduler;

pub use pipeline::*;
pub use quant_lora::*;
pub use scheduler::*;

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::{ModelSpec, WorkloadKind, sample_personal_cluster};
    use kadmium_sharding::ShardPlanner;

    #[test]
    fn test_training_telemetry_computes_low_pipeline_bubble() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::llama_3_70b_lora();
        let plan =
            ShardPlanner::compute_plan(1, &model, WorkloadKind::DistributedTraining, &cluster);
        let telemetry = TrainingJobTelemetry::sample_for_plan(&plan);

        assert!(telemetry.pipeline_bubble_pct < 15.0);
        assert!(telemetry.micro_batches_in_flight >= 8);
    }

    #[test]
    fn test_demo_dct_compressor_and_lossless_error_feedback() {
        let n = 64;
        let mut demo = DeMoDctCompressor::new(n, 0.9);
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).cos() * 0.5).collect();

        // Compress keeping only Top-4 out of 64 DCT frequency components:
        let packet = demo.step_and_compress(&grad, 4);
        assert_eq!(packet.indices.len(), 4);
        assert!(packet.compression_ratio() >= 10.0);

        let reconstructed = DeMoDctCompressor::decompress_packet(&packet);
        assert_eq!(reconstructed.len(), n);

        // Verify exact DeMo energy conservation: reconstructed[i] + residual_error[i] == momentum[i]
        for (i, &rec) in reconstructed.iter().enumerate().take(n) {
            let sum = rec + demo.residual_error[i];
            assert!((sum - demo.momentum[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_hivemind_blockwise_int8_quantization() {
        let input: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) * 0.03).collect();
        let q = BlockwiseInt8Tensor::quantize(&input, 32);
        assert_eq!(q.absmax_scales.len(), 4);

        let deq = q.dequantize();
        for (a, b) in input.iter().zip(deq.iter()) {
            assert!((a - b).abs() < 0.02);
        }
    }

    #[test]
    fn test_verde_repops_and_log_l_layer_bisection() {
        let total_layers = 64usize;
        let faulty_target = 41usize;

        let referee_step = |layer: usize, state: &[f64]| -> Vec<f64> {
            state
                .iter()
                .map(|&v| v * 1.01 + (layer as f64) * 0.005)
                .collect()
        };

        let mut reference_states = Vec::with_capacity(total_layers + 1);
        let mut worker_states = Vec::with_capacity(total_layers + 1);
        reference_states.push(vec![1.0, 2.0, 3.0, 4.0]);
        worker_states.push(vec![1.0, 2.0, 3.0, 4.0]);

        for l in 0..total_layers {
            let next_ref = referee_step(l, &reference_states[l]);
            reference_states.push(next_ref);

            let mut next_worker = referee_step(l, &worker_states[l]);
            if l == faulty_target {
                // Inject hardware corruption / bit flip at Layer 41!
                next_worker[2] += 0.75;
            }
            worker_states.push(next_worker);
        }

        let outcome = VerdeVerificationEngine::bisect_faulty_layer(
            &worker_states,
            &reference_states,
            referee_step,
            1e-6,
        )
        .expect("should isolate faulty layer 41");

        assert_eq!(outcome.faulty_layer_index, faulty_target);
        // Binary search over 64 layers requires at most log2(64) + 1 = 7 checks!
        assert!(outcome.referee_evaluations_used <= 7);

        let cert = VerdeFraudProofEngine::generate_fraud_proof(
            "fp-cert-0041",
            "untrusted-supply-node-99",
            &worker_states,
            &reference_states,
            referee_step,
            1e-6,
            1.52,
        )
        .expect("generate fraud proof certificate");
        assert_eq!(cert.faulty_layer_index, faulty_target);
        assert!(VerdeFraudProofEngine::verify_certificate(&cert));
    }

    #[test]
    fn test_asymmetric_1f1b_pipeline_scheduler_and_diloco_outer_step() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::llama_3_70b_lora();
        let plan =
            ShardPlanner::compute_plan(1, &model, WorkloadKind::DistributedTraining, &cluster);

        let schedule = Pipeline1F1BScheduler::build_schedule(&plan, 12);
        assert!(schedule.num_stages >= 4);
        assert_eq!(
            schedule.events.len(),
            schedule.num_stages * (schedule.num_micro_batches as usize) * 2
        );
        assert!(schedule.pipeline_bubble_pct < 25.0);

        // Verify Sharded-Island DiLoCo Outer Nesterov step moves master weights toward island consensus:
        let mut master = vec![1.0f32, 2.0, 3.0, 4.0];
        let island_a = vec![0.8f32, 1.8, 2.8, 3.8];
        let island_b = vec![0.9f32, 1.9, 2.9, 3.9];

        let mut opt = DiLoCoOuterOptimizer::new(4, 0.7, 0.9);
        opt.step_outer_nesterov(&mut master, &[island_a, island_b]);
        assert!(master[0] < 1.0 && master[0] > 0.7);
    }

    #[test]
    fn test_distributed_moe_pipeline_executor_and_speculative_verification() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);

        let report = DistributedMoEPipelineExecutor::execute_forward_pass(
            &plan,
            "Explain Kadmium asymmetric MoE sharding",
        );

        assert!(!report.stages.is_empty());
        for stage in &report.stages {
            assert_eq!(stage.routed_experts_sample.len(), 4);
            let weight_sum: f32 = stage
                .routed_experts_sample
                .iter()
                .map(|r| r.gating_weight)
                .sum();
            assert!((weight_sum - 1.0).abs() < 0.01);
            assert!(stage.boundary_int8_payload_bytes > 0);
            assert!(stage.output_activation_l2_norm > 0.0);
        }
        assert_eq!(report.speculative_verification.proposed_draft_tokens, 4);
        assert!(report.speculative_verification.accepted_draft_tokens >= 2);
        assert!(report.speculative_verification.effective_speedup_multiplier > 1.5);
    }

    #[test]
    fn test_q4k_block_matvec_and_colibri_double_buffer_prefetch() {
        let rows = 8usize;
        let cols = 32usize;
        let weights: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i as f32) * 0.05).sin() * 0.8)
            .collect();
        let x: Vec<f32> = (0..cols).map(|j| ((j as f32) * 0.1).cos() * 0.5).collect();

        // Exact FP32 reference y = W * x
        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += weights[r * cols + c] * x[c];
            }
            expected[r] = acc;
        }

        let q4 = Q4KBlockTensor::quantize_matrix(&weights, rows, cols);
        assert!(q4.compression_ratio() > 5.0);
        let actual = q4.matvec_f32(&x);

        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!((e - a).abs() < 0.18, "expected {e}, got {a}");
        }

        // Verify Colibri Double-Buffered NVMe DMA overlaps completely when compute >= DMA
        let traces = ColibriNvmeDoubleBuffer::execute_double_buffered_stream(
            4,
            4 * 1024 * 1024, // 4 MiB cold expert page
            7.4,             // 7.4 GB/s PCIe Gen4 NVMe (~0.57 ms DMA)
            1.20,            // 1.20 ms compute window
        );
        assert_eq!(traces.len(), 4);
        for t in &traces {
            assert_eq!(t.effective_pipeline_stall_ms, 0.0);
        }
    }

    #[test]
    fn test_distributed_qlora_adapter_forward_backward_reduces_loss() {
        let d_out = 8usize;
        let d_in = 32usize;
        let base_w: Vec<f32> = (0..(d_out * d_in))
            .map(|i| ((i as f32) * 0.04).sin() * 0.3)
            .collect();
        let x: Vec<f32> = (0..d_in).map(|j| ((j as f32) * 0.1).cos() * 0.6).collect();
        let target: Vec<f32> = vec![0.5, -0.4, 0.3, -0.2, 0.6, -0.5, 0.2, 0.1];

        let mut qlora = DistributedQLoRAAdapter::new(&base_w, d_out, d_in, 4, 8.0);
        let (initial_loss, grads_0) = qlora.backward_and_step(&x, &target, 0.05);
        assert!(!grads_0.is_empty());

        let mut last_loss = initial_loss;
        for _ in 0..8 {
            let (loss, _) = qlora.backward_and_step(&x, &target, 0.05);
            last_loss = loss;
        }
        assert!(
            last_loss < initial_loss,
            "QLoRA backward step must reduce loss ({last_loss} < {initial_loss})"
        );
    }

    #[test]
    fn test_fused_transformer_kernels_and_cross_silicon_dispatch() {
        let cluster = sample_personal_cluster();
        let report = HardwareKernelDispatcher::run_cluster_kernel_benchmark(&cluster);
        assert_eq!(report.node_profiles.len(), cluster.len());
        assert!(report.cross_silicon_bit_identical);
        assert!(report.sample_output_norm > 0.0);
        assert!(
            report
                .node_profiles
                .iter()
                .all(|p| p.repops_digest_fnv1a == report.canonical_repops_digest_fnv1a)
        );
    }

    #[tokio::test]
    async fn test_live_multi_node_tcp_wireframe_inference_and_training() {
        use kadmium_network::LiveWorkerRegistration;
        use std::collections::BTreeMap;

        let cluster = sample_personal_cluster();
        // Spawn 2 real TCP WireFrame worker daemons (Main PC + M4 Mac Mini) on ephemeral loopback ports:
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cluster[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cluster[1].clone(), false)
            .await
            .expect("spawn worker 1");

        let (pong0, rtt0) = DistributedLanExecutor::ping_worker_tcp(&w0.local_addr.to_string(), 1)
            .await
            .expect("ping worker 0");
        assert_eq!(pong0.node.id, cluster[0].id);
        assert!(rtt0 >= 0.0);

        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            cluster[0].id.0.clone(),
            LiveWorkerRegistration {
                node: cluster[0].clone(),
                worker_tcp_addr: w0.local_addr.to_string(),
                worker_lan_ip: "127.0.0.1".into(),
                worker_tcp_port: w0.local_addr.port(),
                os_name: "windows".into(),
                cpu_arch: "x86_64".into(),
                board_or_model: "RTX 4090 Desktop".into(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: rtt0,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );
        live_workers.insert(
            cluster[1].id.0.clone(),
            LiveWorkerRegistration {
                node: cluster[1].clone(),
                worker_tcp_addr: w1.local_addr.to_string(),
                worker_lan_ip: "127.0.0.1".into(),
                worker_tcp_port: w1.local_addr.port(),
                os_name: "macos".into(),
                cpu_arch: "aarch64".into(),
                board_or_model: "Mac Mini M4 Pro".into(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: 0.2,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );

        let sub_cluster = vec![cluster[0].clone(), cluster[1].clone(), cluster[5].clone()];
        let model = ModelSpec::deepseek_v3_671b_moe();
        let inf_plan =
            ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &sub_cluster);

        let (inf_report, hops) = DistributedLanExecutor::execute_live_pipeline(
            &inf_plan,
            "Test multi-node LAN WireFrame inference",
            &live_workers,
        )
        .await;
        assert_eq!(hops.len(), 2);
        assert!(
            hops.iter()
                .all(|h| h.transport_mode.contains("Live TCP WireFrame"))
        );
        assert!(inf_report.effective_tokens_per_sec > 0.0);

        let trn_plan = ShardPlanner::compute_plan(
            1,
            &ModelSpec::llama_3_70b_lora(),
            WorkloadKind::DistributedTraining,
            &sub_cluster,
        );
        let trn_report = DistributedLanExecutor::execute_live_training_step(
            &trn_plan,
            1,
            0.05,
            &live_workers,
        )
        .await;
        assert_eq!(trn_report.stages.len(), 2);
        assert!(trn_report.mean_loss_after < trn_report.mean_loss_before);
        assert!(trn_report.mean_wire_compression_ratio >= 10.0);
        assert!(trn_report.all_stages_repops_verified);

        // Also verify Over-The-Air (OTA) FleetUpdatePush over live TCP WireFrame & LocalFleetSpawner:
        use kadmium_network::{FleetUpdateManifest, UpdatePayloadKind, UpdateTargetScope};
        let payload = b"KDM_OTA_TEST_PAYLOAD".to_vec();
        let expected_merkle = 0x9988_7766_5544_3322u64;
        let mac = FleetUpdateManifest::compute_signature_mac(
            "ota-test-1",
            "v0.2.0-cd48",
            expected_merkle,
            &payload,
        );
        let manifest = FleetUpdateManifest {
            update_id: "ota-test-1".into(),
            target_version: "v0.2.0-cd48".into(),
            payload_kind: UpdatePayloadKind::BinaryRelease,
            target_scope: UpdateTargetScope::AllNodes,
            rolling_canary: true,
            issued_at_unix_ms: 2000,
            expected_ktb_merkle_root_fnv64: expected_merkle,
            artifact_size_bytes: payload.len(),
            payload_bytes: payload,
            signature_mac_fnv64: mac,
            release_notes: "Unit test OTA push".into(),
        };
        let ota_ack =
            DistributedLanExecutor::push_update_tcp(&w0.local_addr.to_string(), 2, &manifest)
                .await
                .expect("push OTA update over TCP WireFrame");
        assert!(ota_ack.accepted);
        assert_eq!(ota_ack.installed_version, "v0.2.0-cd48");
        assert_eq!(ota_ack.applied_ktb_merkle_root_fnv64, expected_merkle);
        assert_ne!(ota_ack.repops_post_check_digest, 0);

        let spawned = LocalFleetSpawner::spawn_heterogeneous_local_fleet(19100, false)
            .await
            .expect("spawn 5-node local + online TCP fleet");
        assert_eq!(spawned.len(), 5);
        assert!(spawned.iter().any(|s| s.network_channel == "OnlineWAN"));
        assert!(spawned.iter().any(|s| s.network_channel == "LocalLAN"));
    }

    #[test]
    fn test_exl3_qtip_fwht_incoherence_and_viterbi_trellis_quantization() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_exl3();
        let report =
            DistributedExl3ConversionEngine::run_conversion_and_benchmark(&model, 2.55, &cluster);

        // FWHT incoherence must suppress heavy-tailed outlier peak-to-RMS ratio
        assert!(
            report.post_fwht_peak_to_rms < report.pre_fwht_peak_to_rms,
            "FWHT must reduce outlier peak-to-RMS ({} < {})",
            report.post_fwht_peak_to_rms,
            report.pre_fwht_peak_to_rms
        );
        // EXL3 Viterbi + FWHT must outperform naive scalar quantization on outlier matrices
        assert!(report.snr_gain_over_scalar_db > 1.5);
        assert!(report.vram_savings_vs_q4k_gb > 120.0);
        assert!(!report.node_conversion_shards.is_empty());

        // Also verify Hadamard-domain matvec_f32 matches dequantize() * x
        let rows = 8usize;
        let cols = 16usize;
        let weights: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i as f32) * 0.13).sin() * 0.4)
            .collect();
        let x: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.21).cos() * 0.5).collect();
        let q = Exl3TrellisQuantizer::quantize_matrix(&weights, rows, cols, 4.0, None, 42);
        let y_fast = q.matvec_f32(&x);
        let deq = q.dequantize();
        for (r, &yf) in y_fast.iter().enumerate() {
            let y_ref: f32 = (0..cols).map(|c| deq[r * cols + c] * x[c]).sum();
            assert!((yf - y_ref).abs() < 1e-3, "matvec mismatch: {yf} vs {y_ref}");
        }
    }
}
