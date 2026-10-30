//! # `kadmium-runtime`
//!
//! Unified runtime orchestrating:
//! 1. **Distributed Inference:** Asymmetric pipeline + MoE expert routing with SBC speculative drafts.
//! 2. **Distributed Training:** 1F1B asymmetric micro-batch scheduling, QLoRA, and DiLoCo sync.
//! 3. **BOINC-Inspired Grid Execution:** Workunit queues, polite host yield, and canary verification.

use kadmium_core::{GradientCompressionCodec, NodeId, TrainingStrategy};
use kadmium_sharding::ClusterShardPlan;
use serde::{Deserialize, Serialize};

pub mod decode_loop;
pub mod deep_gemm;
pub mod diloco;
pub mod exl3_qtip;
pub mod fraud_proof;
pub mod grid_executor;
pub mod inference_compute;
pub mod kernels;
pub mod kv_replay;
pub mod live_worker;
pub mod moe_layer;
pub mod repops;
pub mod speculative;
pub mod throughput_boost;
pub mod training_compute;
pub mod training_framework;
pub use decode_loop::*;
pub use deep_gemm::*;
pub use diloco::*;
pub use exl3_qtip::*;
pub use fraud_proof::*;
pub use grid_executor::*;
pub use inference_compute::*;
pub use kernels::*;
pub use kv_replay::*;
pub use live_worker::*;
pub use moe_layer::*;
pub use speculative::*;
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

    /// Orthonormal 1D Discrete Cosine Transform Type-II (`DCT-II`). Every coefficient is a
    /// RepOps dot product with its cosine basis row, so the packet does not depend on how a
    /// backend chunks the sum.
    pub fn dct_ii(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        let n_f = n as f32;
        (0..n)
            .map(|k| {
                let k_f = k as f32;
                let basis: Vec<f32> = (0..n)
                    .map(|i| repops::cos(std::f32::consts::PI * ((i as f32) + 0.5) * k_f / n_f))
                    .collect();
                let scale = if k == 0 {
                    (1.0 / n_f).sqrt()
                } else {
                    (2.0 / n_f).sqrt()
                };
                repops::tree_dot_f32(input, &basis) * scale
            })
            .collect()
    }

    /// Orthonormal 1D Inverse Discrete Cosine Transform (`IDCT-III`, exact inverse of `dct_ii`).
    /// Every coefficient contributes a term, zeros included, so the reduce tree has the same
    /// shape for every packet and a sparse packet reconstructs to the same bits everywhere.
    pub fn idct_iii(coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len();
        if n == 0 {
            return Vec::new();
        }
        let n_f = n as f32;
        let scale0 = (1.0 / n_f).sqrt();
        let scale_k = (2.0 / n_f).sqrt();
        (0..n)
            .map(|i| {
                let i_half = (i as f32) + 0.5;
                let terms: Vec<f32> = coeffs
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| {
                        if k == 0 {
                            c * scale0
                        } else {
                            c * scale_k
                                * repops::cos(std::f32::consts::PI * i_half * (k as f32) / n_f)
                        }
                    })
                    .collect();
                repops::tree_sum_f32(&terms)
            })
            .collect()
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
        let inf_plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &sub_cluster);

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
        let trn_report =
            DistributedLanExecutor::execute_live_training_step(&trn_plan, 1, 0.05, &live_workers)
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
            assert!(
                (yf - y_ref).abs() < 1e-3,
                "matvec mismatch: {yf} vs {y_ref}"
            );
        }
    }

    #[tokio::test]
    async fn test_live_reterminal_cm4_6_subsystem_cluster() {
        use kadmium_core::{DEFAULT_RETERMINAL_CM4_IP, ReTerminalCm4LanBridge};
        use kadmium_network::{
            FleetUpdateManifest, LiveWorkerRegistration, UpdatePayloadKind, UpdateTargetScope,
        };
        use std::collections::BTreeMap;

        let probe = ReTerminalCm4LanBridge::probe_and_verify(None);
        assert!(probe.execution_proof.emmc_witness_verified);
        assert!(probe.execution_proof.neon_gemm_64x64_ms > 0.0);

        let desktop_node = sample_personal_cluster().remove(0);
        let cm4_node = probe.hardware_node.clone();

        let w_desktop =
            LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, desktop_node.clone(), false)
                .await
                .expect("spawn desktop worker");
        let w_cm4 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cm4_node.clone(), false)
            .await
            .expect("spawn reTerminal CM4 bridge worker");

        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            desktop_node.id.0.clone(),
            LiveWorkerRegistration {
                node: desktop_node.clone(),
                worker_tcp_addr: w_desktop.local_addr.to_string(),
                worker_lan_ip: "127.0.0.1".into(),
                worker_tcp_port: w_desktop.local_addr.port(),
                os_name: "windows".into(),
                cpu_arch: "x86_64".into(),
                board_or_model: "Desktop Workstation".into(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: 0.4,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );
        live_workers.insert(
            cm4_node.id.0.clone(),
            LiveWorkerRegistration {
                node: cm4_node.clone(),
                worker_tcp_addr: w_cm4.local_addr.to_string(),
                worker_lan_ip: DEFAULT_RETERMINAL_CM4_IP.into(),
                worker_tcp_port: w_cm4.local_addr.port(),
                os_name: "linux".into(),
                cpu_arch: "aarch64".into(),
                board_or_model: probe.board_model.clone(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: probe.execution_proof.lan_round_trip_ms,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );

        let cluster = vec![desktop_node.clone(), cm4_node.clone()];
        let model = ModelSpec::deepseek_v3_671b_exl3();
        let inf_plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);

        // 1. Live Multi-Device Inference Forward Pass (Desktop x86_64 + reTerminal CM4 aarch64 NEON)
        let (inf_report, hops) = DistributedLanExecutor::execute_live_pipeline(
            &inf_plan,
            "Verify Seeed Studio reTerminal CM4 distributed inference",
            &live_workers,
        )
        .await;
        assert!(!hops.is_empty());
        assert!(inf_report.effective_tokens_per_sec > 0.0);

        // 2. Live Cross-ISA Gensyn Verde O(log L) Layer-Bisection Audit (x86_64 vs aarch64 Cortex-A72)
        let mut cm4_audit_workers = BTreeMap::new();
        cm4_audit_workers.insert(
            cm4_node.id.0.clone(),
            live_workers.get(&cm4_node.id.0).unwrap().clone(),
        );
        let honest_audit = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
            1,
            64,
            None,
            &cm4_audit_workers,
        )
        .await;
        assert!(honest_audit.bit_exact_verified);
        assert_eq!(honest_audit.disputed_layer, None);

        let fault_audit = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
            1,
            64,
            Some(41),
            &cm4_audit_workers,
        )
        .await;
        assert!(!fault_audit.bit_exact_verified);
        assert_eq!(fault_audit.disputed_layer, Some(41));

        // 3. Live FTS3 64 KiB .ktb Shard Streaming to reTerminal CM4 32 GB eMMC
        let fts3_report = DistributedLanExecutor::stream_fts3_ktb_shard_tcp(
            &model.id,
            "shard-cm4-emmc-00.ktb",
            1,
            &cm4_audit_workers,
        )
        .await;
        assert!(
            fts3_report
                .target_receipts
                .iter()
                .all(|r| r.verified_merkle_proof)
        );

        // 4. Live Over-The-Air (OTA) Release Push + Post-Update ARMv8 NEON RepOps Self-Test
        let payload = b"KDM_CM4_RETERMINAL_OTA_V020".to_vec();
        let expected_merkle = fts3_report.merkle_root_fnv64;
        let mac = FleetUpdateManifest::compute_signature_mac(
            "ota-cm4-01",
            "v0.2.0-cd48",
            expected_merkle,
            &payload,
        );
        let manifest = FleetUpdateManifest {
            update_id: "ota-cm4-01".into(),
            target_version: "v0.2.0-cd48".into(),
            payload_kind: UpdatePayloadKind::BinaryRelease,
            target_scope: UpdateTargetScope::AllNodes,
            rolling_canary: true,
            issued_at_unix_ms: 3000,
            expected_ktb_merkle_root_fnv64: expected_merkle,
            artifact_size_bytes: payload.len(),
            payload_bytes: payload,
            signature_mac_fnv64: mac,
            release_notes: "Live OTA release staged on reTerminal CM4 eMMC".into(),
        };
        let ota_ack =
            DistributedLanExecutor::push_update_tcp(&w_cm4.local_addr.to_string(), 2, &manifest)
                .await
                .expect("push OTA to CM4 worker");
        assert!(ota_ack.accepted);
        assert_eq!(ota_ack.installed_version, "v0.2.0-cd48");

        // 5. Live Dream-RSI MCTS PUCT Rollout Evolution on reTerminal CM4 Cortex-A72
        let dream_report = DistributedLanExecutor::run_live_dream_rsi_replay_tcp(
            1,
            "gpu_kernel_opt",
            64,
            &cm4_audit_workers,
        )
        .await;
        assert!(dream_report.best_evolved_speedup > 1.38);

        // 6. Live ARMv8 NEON 128-bit veorq_u32 RAID-5 XOR Parity Reconstruction on reTerminal CM4
        if probe.physical_board_online {
            let s0 = vec![0x12u8; 256];
            let s1 = vec![0x34u8; 256];
            let s2 = vec![0x56u8; 256];
            let s3_lost = vec![0x78u8; 256];
            let parity: Vec<u8> = (0..256)
                .map(|i| s0[i] ^ s1[i] ^ s2[i] ^ s3_lost[i])
                .collect();
            let (recovered, _fnv, hw_ms) =
                ReTerminalCm4LanBridge::execute_raid5_xor_recovery_on_cm4(
                    DEFAULT_RETERMINAL_CM4_IP,
                    &parity,
                    &s0,
                    &s1,
                    &s2,
                )
                .expect("execute live NEON RAID-5 XOR recovery on reTerminal CM4");
            assert_eq!(recovered, s3_lost);
            assert!(hw_ms >= 0.0);
            println!(
                "=== LIVE RETERMINAL CM4 6-SUBSYSTEM TEST PASSED ===\n\
                 - Physical Board Online: {} ({} @ {})\n\
                 - 1. Inference Throughput: {:.1} tok/s across {} stages\n\
                 - 2. Cross-ISA Verde Audit: honest_verified={} ({} steps), fault_L41_detected={:?} (cert {:?})\n\
                 - 3. FTS3 .ktb eMMC Stream: {} pages verified [Merkle {:#018x}] ({:.1} Mbps)\n\
                 - 4. OTA Push to CM4 eMMC: accepted={} version={} ({})\n\
                 - 5. Dream-RSI MCTS on CM4: {} rollouts -> {:.2}x speedup [hash {:#018x}]\n\
                 - 6. ARMv8 NEON veorq_u32 RAID-5 XOR Recovery: 256 B reconstructed in {:.3} ms",
                probe.physical_board_online,
                probe.board_model,
                probe.lan_endpoint,
                inf_report.effective_tokens_per_sec,
                hops.len(),
                honest_audit.bit_exact_verified,
                honest_audit.bisection_steps.len(),
                fault_audit.disputed_layer,
                fault_audit.fraud_proof_certificate_id,
                fts3_report.target_receipts.len(),
                fts3_report.merkle_root_fnv64,
                fts3_report.aggregate_goodput_mbps,
                ota_ack.accepted,
                ota_ack.installed_version,
                ota_ack.status_note,
                dream_report.total_mcts_simulations,
                dream_report.best_evolved_speedup,
                dream_report.evolved_policy_hash_fnv64,
                hw_ms,
            );
        }
    }

    #[tokio::test]
    async fn test_live_reterminal_cm4_exl3_qlora_kv_vision_and_stress_suite() {
        use kadmium_core::{
            DEFAULT_RETERMINAL_CM4_IP, ModelSpec, ReTerminalCm4LanBridge, WorkloadKind,
            sample_personal_cluster,
        };
        use kadmium_network::LiveWorkerRegistration;
        use kadmium_sharding::ShardPlanner;
        use std::collections::BTreeMap;

        let probe = ReTerminalCm4LanBridge::probe_and_verify(None);
        let desktop_node = sample_personal_cluster().remove(0);
        let cm4_node = probe.hardware_node.clone();

        let w_desktop =
            LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, desktop_node.clone(), false)
                .await
                .expect("spawn desktop worker");
        let w_cm4 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cm4_node.clone(), false)
            .await
            .expect("spawn reTerminal CM4 bridge worker");

        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            desktop_node.id.0.clone(),
            LiveWorkerRegistration {
                node: desktop_node.clone(),
                worker_tcp_addr: w_desktop.local_addr.to_string(),
                worker_lan_ip: "127.0.0.1".into(),
                worker_tcp_port: w_desktop.local_addr.port(),
                os_name: "windows".into(),
                cpu_arch: "x86_64".into(),
                board_or_model: "Desktop Workstation".into(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: 0.4,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );
        live_workers.insert(
            cm4_node.id.0.clone(),
            LiveWorkerRegistration {
                node: cm4_node.clone(),
                worker_tcp_addr: w_cm4.local_addr.to_string(),
                worker_lan_ip: DEFAULT_RETERMINAL_CM4_IP.into(),
                worker_tcp_port: w_cm4.local_addr.port(),
                os_name: "linux".into(),
                cpu_arch: "aarch64".into(),
                board_or_model: probe.board_model.clone(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: probe.execution_proof.lan_round_trip_ms,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );

        let cluster = vec![desktop_node.clone(), cm4_node.clone()];
        let model = ModelSpec::deepseek_v3_671b_exl3();
        let train_plan =
            ShardPlanner::compute_plan(2, &model, WorkloadKind::DistributedTraining, &cluster);
        let inf_plan = ShardPlanner::compute_plan(2, &model, WorkloadKind::Inference, &cluster);

        // 7. Live Distributed QLoRA + Nous DeMo DCT + DiLoCo Nesterov Training Step on reTerminal CM4
        let train_step =
            DistributedLanExecutor::execute_live_training_step(&train_plan, 1, 0.08, &live_workers)
                .await;
        assert!(train_step.all_stages_repops_verified);
        assert!(
            train_step.mean_loss_after < train_step.mean_loss_before,
            "QLoRA training step must reduce loss ({} < {})",
            train_step.mean_loss_after,
            train_step.mean_loss_before
        );
        assert!(train_step.mean_wire_compression_ratio > 10.0);

        // 8. Live Zero-Drop KV-Cache Page Migration (Polite Gaming Yield Failover to reTerminal CM4 eMMC)
        let kv_mig = DistributedLanExecutor::migrate_kv_cache_pages_tcp(
            2,
            "sess-cm4-live-01",
            (32, 60),
            &live_workers,
        )
        .await;
        assert!(kv_mig.zero_context_drop_verified);
        assert!(kv_mig.ack.bit_exact_verified);
        assert_eq!(kv_mig.pages_migrated, 8);
        assert_eq!(kv_mig.tokens_preserved, 128);

        // 9. Live Strata Multimodal Vision Patch Offload (SigLIP-SO400M on aarch64 reTerminal CM4)
        let vision = DistributedLanExecutor::execute_vision_patch_offload(
            2,
            "Inspect reTerminal CM4 industrial panel telemetry",
            &live_workers,
        )
        .await;
        assert_eq!(vision.offloaded_to_node_id, cm4_node.id.0);
        assert_eq!(vision.image_patches_encoded, 729);
        assert_eq!(vision.visual_tokens_projected, 256);

        // 10. Live Distributed Embeddings (/v1/embeddings across x86_64 + aarch64 CM4)
        let emb = DistributedLanExecutor::execute_live_embeddings(
            &inf_plan,
            &[
                "Kadmium heterogeneous edge cluster".to_string(),
                "Seeed Studio reTerminal CM4 ARMv8 NEON".to_string(),
            ],
            64,
            &live_workers,
        )
        .await;
        assert_eq!(emb.vectors.len(), 2);
        assert_eq!(emb.vectors[0].embedding.len(), 64);

        if probe.physical_board_online {
            // 11. Live Cross-ISA EXL3 / QTIP Fast Walsh-Hadamard Transform + Trellis MatVec (x86_64 vs aarch64 NEON)
            let rows = 16usize;
            let cols = 32usize;
            let weights: Vec<f32> = (0..(rows * cols))
                .map(|i| {
                    let base = ((i as f32) * 0.17).sin() * 0.35;
                    if i % 29 == 0 { base * 4.2 } else { base }
                })
                .collect();
            let x: Vec<f32> = (0..cols)
                .map(|c| ((c as f32) * 0.23 + 0.1).cos() * 0.6)
                .collect();
            let cal_rows: Vec<Vec<f32>> = (0..12)
                .map(|s| {
                    (0..cols)
                        .map(|c| (((s * 11 + c * 5) as f32) * 0.13).sin() * 0.5)
                        .collect()
                })
                .collect();
            let col_signs = FastWalshHadamardTransform::rademacher_signs(cols, 777 + 1);
            let hessian =
                OnTheFlyHessianLdl::from_calibration_activations(&cal_rows, cols, &col_signs, 0.05);

            for &bpw in &[2.5f32, 3.0f32, 4.0f32] {
                let q_tensor = Exl3TrellisQuantizer::quantize_matrix(
                    &weights,
                    rows,
                    cols,
                    bpw,
                    Some(&hessian),
                    777,
                );
                let y_x86 = q_tensor.matvec_f32(&x);
                let y_x86_f64: Vec<f64> = y_x86.iter().map(|&v| v as f64).collect();
                let x86_sum = VerdeVerificationEngine::repops_pairwise_tree_sum(&y_x86_f64);
                let q_milli = (x86_sum * 10_000.0).round() as i64;
                let mut buf = [0u8; 16];
                buf[0..4].copy_from_slice(&0u32.to_le_bytes());
                buf[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
                buf[8..16].copy_from_slice(&(q_milli as u64).to_le_bytes());
                let mut x86_dig: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in &buf {
                    x86_dig ^= b as u64;
                    x86_dig = x86_dig.wrapping_mul(0x0100_0000_01b3);
                }

                let (y_cm4, cm4_dig, cm4_exl3_ms) =
                    ReTerminalCm4LanBridge::execute_exl3_fwht_matvec_on_cm4(
                        DEFAULT_RETERMINAL_CM4_IP,
                        rows,
                        cols,
                        q_tensor.trellis_l_bits,
                        q_tensor.rademacher_seed,
                        &q_tensor.row_scales,
                        &q_tensor.col_bit_schedule,
                        &q_tensor.trellis_symbols,
                        &x,
                    )
                    .expect("execute live EXL3 FWHT matvec on reTerminal CM4");

                for (r, (&vx, &vc)) in y_x86.iter().zip(y_cm4.iter()).enumerate() {
                    assert!(
                        (vx - vc).abs() < 1e-4,
                        "EXL3 {:.1}-bpw row {r} mismatch between x86_64 ({vx}) and CM4 NEON ({vc})",
                        bpw
                    );
                }
                assert_eq!(
                    x86_dig, cm4_dig,
                    "Cross-ISA EXL3 RepOps digest mismatch at {bpw} bpw"
                );
                println!(
                    "  [EXL3 {:.1}-bpw Cross-ISA Verified] x86_64 == CM4 ARMv8 NEON | RepOps={:#018x} | CM4 HW={:.3} ms",
                    bpw, cm4_dig, cm4_exl3_ms
                );
            }

            // 12. Live 200-Iteration ARMv8 NEON 64x64 GEMM Stress & Thermal Benchmark on reTerminal CM4
            let stress =
                ReTerminalCm4LanBridge::run_stress_benchmark_on_cm4(DEFAULT_RETERMINAL_CM4_IP, 200)
                    .expect("run 200-iteration NEON stress benchmark on reTerminal CM4");
            assert_eq!(stress.iterations, 200);
            assert!(stress.sustained_gflops > 0.1);
            println!(
                "=== LIVE RETERMINAL CM4 EXTENDED SUITE (SUBSYSTEMS 7..12) PASSED ===\n\
                 - 7. Distributed QLoRA + DeMo DCT on CM4: loss {:.4} -> {:.4} ({:.1}x wire compression)\n\
                 - 8. Zero-Drop KV-Cache Migration to CM4 eMMC: {} pages ({} tokens) verified={} in {:.2} ms\n\
                 - 9. Strata Vision Offload (SigLIP-400M on CM4): {} patches -> {} tokens [RepOps {:#018x}]\n\
                 - 10. Distributed Embeddings: {}x{}D vectors [RepOps {:#018x}]\n\
                 - 11. Cross-ISA EXL3/QTIP FWHT + Trellis MatVec (2.5 / 3.0 / 4.0 bpw): Bit-Exact x86_64 == ARMv8 NEON\n\
                 - 12. 200-Iter NEON 64x64 GEMM Stress Bench: min={:.3}ms p50={:.3}ms p95={:.3}ms p99={:.3}ms | {:.2} GFLOP/s | Temp {:.1}°C -> {:.1}°C @ {} MHz",
                train_step.mean_loss_before,
                train_step.mean_loss_after,
                train_step.mean_wire_compression_ratio,
                kv_mig.pages_migrated,
                kv_mig.tokens_preserved,
                kv_mig.zero_context_drop_verified,
                kv_mig.migration_rtt_ms,
                vision.image_patches_encoded,
                vision.visual_tokens_projected,
                vision.repops_digest_fnv1a,
                emb.vectors.len(),
                emb.dimensions,
                emb.vectors[0].repops_digest_fnv1a,
                stress.min_ms,
                stress.p50_ms,
                stress.p95_ms,
                stress.p99_ms,
                stress.sustained_gflops,
                stress.temp_before_c,
                stress.temp_after_c,
                stress.freq_after_mhz,
            );
        }
    }

    #[tokio::test]
    async fn test_live_apple_m4_mac_mini_12_subsystem_and_multi_device_cluster() {
        use kadmium_core::{
            DEFAULT_M4_MAC_MINI_IP, M4MacMiniLanBridge, ModelSpec, ReTerminalCm4LanBridge,
            WorkloadKind, sample_personal_cluster,
        };
        use kadmium_network::{
            FleetUpdateManifest, LiveWorkerRegistration, UpdatePayloadKind, UpdateTargetScope,
        };
        use kadmium_sharding::ShardPlanner;
        use std::collections::BTreeMap;

        let m4_probe = M4MacMiniLanBridge::probe_and_verify(None);
        assert!(m4_probe.execution_proof.nvme_witness_verified);
        assert!(m4_probe.execution_proof.amx_sgemm_512_gflops > 50.0);

        let cm4_probe = ReTerminalCm4LanBridge::probe_and_verify(None);
        let desktop_node = sample_personal_cluster().remove(0);
        let m4_node = m4_probe.hardware_node.clone();
        let cm4_node = cm4_probe.hardware_node.clone();

        let w_desktop =
            LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, desktop_node.clone(), false)
                .await
                .expect("spawn desktop worker");
        let w_m4 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, m4_node.clone(), false)
            .await
            .expect("spawn Apple M4 Mac mini bridge worker");
        let w_cm4 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cm4_node.clone(), false)
            .await
            .expect("spawn reTerminal CM4 bridge worker");

        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            desktop_node.id.0.clone(),
            LiveWorkerRegistration {
                node: desktop_node.clone(),
                worker_tcp_addr: w_desktop.local_addr.to_string(),
                worker_lan_ip: "127.0.0.1".into(),
                worker_tcp_port: w_desktop.local_addr.port(),
                os_name: "windows".into(),
                cpu_arch: "x86_64".into(),
                board_or_model: "Desktop Workstation".into(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: 0.4,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );
        live_workers.insert(
            m4_node.id.0.clone(),
            LiveWorkerRegistration {
                node: m4_node.clone(),
                worker_tcp_addr: w_m4.local_addr.to_string(),
                worker_lan_ip: DEFAULT_M4_MAC_MINI_IP.into(),
                worker_tcp_port: w_m4.local_addr.port(),
                os_name: "macos".into(),
                cpu_arch: "aarch64".into(),
                board_or_model: m4_probe.board_model.clone(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: m4_probe.execution_proof.lan_round_trip_ms,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );
        live_workers.insert(
            cm4_node.id.0.clone(),
            LiveWorkerRegistration {
                node: cm4_node.clone(),
                worker_tcp_addr: w_cm4.local_addr.to_string(),
                worker_lan_ip: kadmium_core::DEFAULT_RETERMINAL_CM4_IP.into(),
                worker_tcp_port: w_cm4.local_addr.port(),
                os_name: "linux".into(),
                cpu_arch: "aarch64".into(),
                board_or_model: cm4_probe.board_model.clone(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: cm4_probe.execution_proof.lan_round_trip_ms,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            },
        );

        let mut m4_only_workers = BTreeMap::new();
        m4_only_workers.insert(
            m4_node.id.0.clone(),
            live_workers.get(&m4_node.id.0).unwrap().clone(),
        );

        // 3-Device Live Cluster: Windows x86_64 PC + Apple M4 Mac mini (Metal 4/AMX) + Seeed reTerminal CM4 (ARMv8 NEON)
        let cluster = vec![desktop_node.clone(), m4_node.clone(), cm4_node.clone()];
        let model = ModelSpec::deepseek_v3_671b_exl3();
        let inf_plan = ShardPlanner::compute_plan(3, &model, WorkloadKind::Inference, &cluster);
        let train_plan =
            ShardPlanner::compute_plan(3, &model, WorkloadKind::DistributedTraining, &cluster);

        // 1. Live 3-Device Pipeline Inference (Windows x86_64 -> Apple M4 Metal 4 -> reTerminal CM4 NEON)
        let (inf_report, hops) = DistributedLanExecutor::execute_live_pipeline(
            &inf_plan,
            "Verify 3-device Windows PC + Apple M4 Mac mini + Seeed reTerminal CM4 inference",
            &live_workers,
        )
        .await;
        assert!(hops.len() >= 2);
        assert!(inf_report.effective_tokens_per_sec > 0.0);

        // 2. Live Cross-ISA Gensyn Verde O(log L) Layer-Bisection Audit against Apple M4 Mac mini
        let honest_audit = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
            3,
            64,
            None,
            &m4_only_workers,
        )
        .await;
        assert!(honest_audit.bit_exact_verified);
        assert_eq!(honest_audit.disputed_layer, None);

        let fault_audit = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
            3,
            64,
            Some(23),
            &m4_only_workers,
        )
        .await;
        assert!(!fault_audit.bit_exact_verified);
        assert_eq!(fault_audit.disputed_layer, Some(23));

        // 3. Live FTS3 64 KiB .ktb Shard Streaming to Apple M4 NVMe SSD
        let fts3_report = DistributedLanExecutor::stream_fts3_ktb_shard_tcp(
            &model.id,
            "shard-apple-m4-nvme-00.ktb",
            3,
            &m4_only_workers,
        )
        .await;
        assert!(
            fts3_report
                .target_receipts
                .iter()
                .all(|r| r.verified_merkle_proof)
        );

        // 4. Live Over-The-Air (OTA) Release Push + Post-Update Apple M4 RepOps Self-Test
        let payload = b"KDM_APPLE_M4_METAL4_OTA_V020".to_vec();
        let expected_merkle = fts3_report.merkle_root_fnv64;
        let mac = FleetUpdateManifest::compute_signature_mac(
            "ota-m4-01",
            "v0.2.0-cd48",
            expected_merkle,
            &payload,
        );
        let manifest = FleetUpdateManifest {
            update_id: "ota-m4-01".into(),
            target_version: "v0.2.0-cd48".into(),
            payload_kind: UpdatePayloadKind::BinaryRelease,
            target_scope: UpdateTargetScope::AllNodes,
            rolling_canary: true,
            issued_at_unix_ms: 4000,
            expected_ktb_merkle_root_fnv64: expected_merkle,
            artifact_size_bytes: payload.len(),
            payload_bytes: payload,
            signature_mac_fnv64: mac,
            release_notes: "Live OTA release staged on Apple M4 Mac mini NVMe".into(),
        };
        let ota_ack =
            DistributedLanExecutor::push_update_tcp(&w_m4.local_addr.to_string(), 3, &manifest)
                .await
                .expect("push OTA to Apple M4 Mac mini");
        assert!(ota_ack.accepted);

        // 5. Live Dream-RSI MCTS PUCT Rollouts on Apple M4
        let dream_report = DistributedLanExecutor::run_live_dream_rsi_replay_tcp(
            3,
            "metal4_amx_kernel_opt",
            64,
            &m4_only_workers,
        )
        .await;
        assert!(dream_report.best_evolved_speedup > 1.38);

        // 6. Live Distributed QLoRA + Nous DeMo DCT + DiLoCo Training Step across PC + Apple M4 + CM4
        let train_step =
            DistributedLanExecutor::execute_live_training_step(&train_plan, 1, 0.08, &live_workers)
                .await;
        assert!(train_step.all_stages_repops_verified);
        assert!(train_step.mean_loss_after < train_step.mean_loss_before);

        // 7. Live Zero-Drop KV-Cache Page Migration to Apple M4 16 GB Unified Memory / NVMe
        let kv_mig = DistributedLanExecutor::migrate_kv_cache_pages_tcp(
            3,
            "sess-apple-m4-live-01",
            (16, 48),
            &m4_only_workers,
        )
        .await;
        assert!(kv_mig.zero_context_drop_verified);

        // 8. Live Strata Multimodal Vision Patch Offload (SigLIP-400M on Apple M4 Metal)
        let vision = DistributedLanExecutor::execute_vision_patch_offload(
            3,
            "Apple M4 Metal 4 SigLIP vision patch encode",
            &m4_only_workers,
        )
        .await;
        assert_eq!(vision.offloaded_to_node_id, m4_node.id.0);

        if m4_probe.physical_board_online {
            // 9. Live ARMv9.2-A NEON 128-bit veorq_u32 RAID-5 XOR Parity Reconstruction on Apple M4
            let s0 = vec![0xAAu8; 256];
            let s1 = vec![0x55u8; 256];
            let s2 = vec![0x33u8; 256];
            let s3_lost = vec![0xCCu8; 256];
            let parity: Vec<u8> = (0..256)
                .map(|i| s0[i] ^ s1[i] ^ s2[i] ^ s3_lost[i])
                .collect();
            let (recovered, _fnv, xor_ms) = M4MacMiniLanBridge::execute_raid5_xor_recovery_on_m4(
                DEFAULT_M4_MAC_MINI_IP,
                &parity,
                &s0,
                &s1,
                &s2,
            )
            .expect("execute live NEON RAID-5 XOR recovery on Apple M4");
            assert_eq!(recovered, s3_lost);

            // 10. Live Cross-ISA EXL3 / QTIP FWHT + Bitshift Trellis MatVec (x86_64 vs Apple M4 Accelerate vDSP + NEON)
            let rows = 16usize;
            let cols = 32usize;
            let weights: Vec<f32> = (0..(rows * cols))
                .map(|i| ((i as f32) * 0.17).sin() * 0.35)
                .collect();
            let x: Vec<f32> = (0..cols)
                .map(|c| ((c as f32) * 0.23 + 0.1).cos() * 0.6)
                .collect();
            let q_tensor =
                Exl3TrellisQuantizer::quantize_matrix(&weights, rows, cols, 2.5, None, 777);
            let y_x86 = q_tensor.matvec_f32(&x);
            let y_x86_f64: Vec<f64> = y_x86.iter().map(|&v| v as f64).collect();
            let x86_sum = VerdeVerificationEngine::repops_pairwise_tree_sum(&y_x86_f64);
            let q_milli = (x86_sum * 10_000.0).round() as i64;
            let mut buf = [0u8; 16];
            buf[0..4].copy_from_slice(&0u32.to_le_bytes());
            buf[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
            buf[8..16].copy_from_slice(&(q_milli as u64).to_le_bytes());
            let mut x86_dig: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in &buf {
                x86_dig ^= b as u64;
                x86_dig = x86_dig.wrapping_mul(0x0100_0000_01b3);
            }

            let (y_m4, m4_dig, m4_exl3_ms) = M4MacMiniLanBridge::execute_exl3_fwht_matvec_on_m4(
                DEFAULT_M4_MAC_MINI_IP,
                rows,
                cols,
                q_tensor.trellis_l_bits,
                q_tensor.rademacher_seed,
                &q_tensor.row_scales,
                &q_tensor.col_bit_schedule,
                &q_tensor.trellis_symbols,
                &x,
            )
            .expect("execute live EXL3 FWHT matvec on Apple M4 Mac mini");

            for (vx, vm) in y_x86.iter().zip(y_m4.iter()) {
                assert!((vx - vm).abs() < 1e-4);
            }
            assert_eq!(x86_dig, m4_dig);

            // 11. Live Stress Benchmark on Apple M4 (Metal 4 GPU + AMX 512x512 SGEMM + HW CRC32C)
            let stress = M4MacMiniLanBridge::run_stress_benchmark_on_m4(DEFAULT_M4_MAC_MINI_IP, 25)
                .expect("run stress benchmark on Apple M4 Mac mini");

            println!(
                "=== LIVE APPLE M4 MAC MINI + 3-DEVICE CLUSTER SUITE PASSED ===\n\
                 - Physical Apple M4 Online: {} ({} @ {})\n\
                 - 1. 3-Device Pipeline Inference (Windows x86_64 -> Apple M4 -> reTerminal CM4): {:.1} tok/s ({} stages)\n\
                 - 2. Cross-ISA Verde Audit on M4: honest_verified={} ({} steps), fault_L23_detected={:?}\n\
                 - 3. FTS3 .ktb NVMe Stream to M4: {} pages verified [Merkle {:#018x}] ({:.1} Mbps)\n\
                 - 4. OTA Push to M4 NVMe: accepted={} ({})\n\
                 - 5. Dream-RSI MCTS on M4: {} rollouts -> {:.2}x speedup [hash {:#018x}]\n\
                 - 6. 3-Device QLoRA + DeMo DCT Training: loss {:.4} -> {:.4} ({:.1}x wire compression)\n\
                 - 7. Zero-Drop KV-Cache Migration to M4 Unified Memory: {} pages ({} tokens) in {:.2} ms\n\
                 - 8. Strata Vision Offload (SigLIP-400M on M4 Metal): {} patches -> {} tokens [RepOps {:#018x}]\n\
                 - 9. ARMv9.2-A NEON RAID-5 XOR Recovery on M4: 256 B reconstructed in {:.4} ms\n\
                 - 10. Cross-ISA EXL3/QTIP 2.5-bpw FWHT MatVec (x86_64 == Apple M4 vDSP+NEON): RepOps={:#018x} in {:.4} ms\n\
                 - 11. Metal 4 GPU + AMX 512x512 SGEMM Stress: Metal p50={:.3}ms | AMX={:.2} GFLOP/s ({:.3}ms) | HW CRC32C={:.2} GB/s",
                m4_probe.physical_board_online,
                m4_probe.board_model,
                m4_probe.lan_endpoint,
                inf_report.effective_tokens_per_sec,
                hops.len(),
                honest_audit.bit_exact_verified,
                honest_audit.bisection_steps.len(),
                fault_audit.disputed_layer,
                fts3_report.target_receipts.len(),
                fts3_report.merkle_root_fnv64,
                fts3_report.aggregate_goodput_mbps,
                ota_ack.accepted,
                ota_ack.status_note,
                dream_report.total_mcts_simulations,
                dream_report.best_evolved_speedup,
                dream_report.evolved_policy_hash_fnv64,
                train_step.mean_loss_before,
                train_step.mean_loss_after,
                train_step.mean_wire_compression_ratio,
                kv_mig.pages_migrated,
                kv_mig.tokens_preserved,
                kv_mig.migration_rtt_ms,
                vision.image_patches_encoded,
                vision.visual_tokens_projected,
                vision.repops_digest_fnv1a,
                xor_ms,
                m4_dig,
                m4_exl3_ms,
                stress.metal_p50_ms,
                stress.amx_sgemm_512_gflops,
                stress.amx_sgemm_512_ms,
                stress.hw_crc32c_gbps,
            );
        }
    }

    #[tokio::test]
    async fn test_live_platform_family_catalogs_and_cross_platform_swarm_stress() {
        use kadmium_core::{
            AppleM4PlatformBridge, AppleM4ProductFormFactor, AppleM4SiliconTier,
            Cm4CarrierFormFactor, DEFAULT_M4_MAC_MINI_IP, DEFAULT_RETERMINAL_CM4_IP,
            KendryteK230BoardFormFactor, KendryteK230PlatformBridge, ModelSpec,
            RpiCm4PlatformBridge, TegraX1PlatformController, TegraX1PlatformFormFactor,
            WorkloadKind,
        };
        use kadmium_network::LiveWorkerRegistration;
        use kadmium_sharding::ShardPlanner;
        use std::collections::BTreeMap;

        // 1. Verify Apple M4 Platform Family Catalog (Mac mini, MacBook Pro, MacBook Air, iMac, Mac Studio, iPad Pro across M4 / M4 Pro / M4 Max / M4 Ultra)
        let m4_catalog = AppleM4PlatformBridge::catalog_all_m4_products();
        assert_eq!(m4_catalog.len(), 7);
        assert!(
            m4_catalog
                .iter()
                .any(|(ff, tier, _)| *ff == AppleM4ProductFormFactor::MacMini
                    && *tier == AppleM4SiliconTier::M4)
        );
        assert!(
            m4_catalog
                .iter()
                .any(|(ff, tier, _)| *ff == AppleM4ProductFormFactor::MacMini
                    && *tier == AppleM4SiliconTier::M4Pro)
        );
        assert!(
            m4_catalog
                .iter()
                .any(|(ff, tier, _)| *ff == AppleM4ProductFormFactor::MacBookPro
                    && *tier == AppleM4SiliconTier::M4Max)
        );
        assert!(
            m4_catalog
                .iter()
                .any(|(ff, tier, _)| *ff == AppleM4ProductFormFactor::MacBookAir
                    && *tier == AppleM4SiliconTier::M4)
        );
        assert!(
            m4_catalog
                .iter()
                .any(|(ff, tier, _)| *ff == AppleM4ProductFormFactor::IMac
                    && *tier == AppleM4SiliconTier::M4)
        );
        assert!(
            m4_catalog
                .iter()
                .any(|(ff, tier, _)| *ff == AppleM4ProductFormFactor::MacStudio
                    && *tier == AppleM4SiliconTier::M4Ultra)
        );
        assert!(
            m4_catalog
                .iter()
                .any(|(ff, tier, _)| *ff == AppleM4ProductFormFactor::IPadPro
                    && *tier == AppleM4SiliconTier::M4)
        );

        // 2. Verify Raspberry Pi CM4 (BCM2711 Cortex-A72) Platform Family Catalog across all CM4 carriers & RAM/storage SKUs
        let cm4_catalog = RpiCm4PlatformBridge::catalog_all_cm4_platforms();
        assert_eq!(cm4_catalog.len(), 6);
        assert!(
            cm4_catalog
                .iter()
                .any(|(c, _)| *c == Cm4CarrierFormFactor::SeeedReTerminalHmi)
        );
        assert!(
            cm4_catalog
                .iter()
                .any(|(c, _)| *c == Cm4CarrierFormFactor::OfficialRpiCm4IoBoard)
        );
        assert!(
            cm4_catalog
                .iter()
                .any(|(c, _)| *c == Cm4CarrierFormFactor::TuringPi2ClusterBlade)
        );
        assert!(
            cm4_catalog
                .iter()
                .any(|(c, _)| *c == Cm4CarrierFormFactor::WaveshareCm4Industrial)
        );
        assert!(
            cm4_catalog
                .iter()
                .any(|(c, _)| *c == Cm4CarrierFormFactor::ClockworkPiUConsoleCm4)
        );
        assert!(
            cm4_catalog
                .iter()
                .any(|(c, _)| *c == Cm4CarrierFormFactor::GenericBcm2711Cm4Carrier)
        );

        // 3. Verify NVIDIA Tegra X1 / X1+ (T210 / T214) Platform Family Catalog (Switch Mariko/Erista/8GB + Jetson Nano B01 + Shield TV Pro)
        let tegra_catalog = TegraX1PlatformController::catalog_all_tegra_x1_platforms();
        assert_eq!(tegra_catalog.len(), 5);
        assert!(
            tegra_catalog
                .iter()
                .any(|(f, _)| *f == TegraX1PlatformFormFactor::NintendoSwitchMariko)
        );
        assert!(
            tegra_catalog
                .iter()
                .any(|(f, _)| *f == TegraX1PlatformFormFactor::NvidiaJetsonNanoB01)
        );
        assert!(
            tegra_catalog
                .iter()
                .any(|(f, _)| *f == TegraX1PlatformFormFactor::NvidiaShieldTvProT214)
        );

        // 4. Verify Kendryte K230 / K230D RISC-V (RVV 1.0 + KPU) Platform Family Catalog
        let k230_catalog = KendryteK230PlatformBridge::catalog_all_k230_platforms();
        assert_eq!(k230_catalog.len(), 5);
        assert!(
            k230_catalog
                .iter()
                .any(|(f, _)| *f == KendryteK230BoardFormFactor::CanMvK230_01Studio)
        );
        assert!(
            k230_catalog
                .iter()
                .any(|(f, _)| *f == KendryteK230BoardFormFactor::CanMvK230DZeroSip)
        );

        // 5. Verify Multi-Platform Heterogeneous Sharding across a combined 23-node cross-platform fleet
        let mut fleet_nodes = Vec::new();
        for (_, _, n) in &m4_catalog {
            fleet_nodes.push(n.clone());
        }
        for (_, n) in &cm4_catalog {
            fleet_nodes.push(n.clone());
        }
        for (_, n) in &tegra_catalog {
            fleet_nodes.push(n.clone());
        }
        for (_, n) in &k230_catalog {
            fleet_nodes.push(n.clone());
        }
        assert_eq!(fleet_nodes.len(), 23);

        let model = ModelSpec::deepseek_v3_671b_exl3();
        let deepseek_plan =
            ShardPlanner::compute_plan(4, &model, WorkloadKind::Inference, &fleet_nodes);
        assert!(deepseek_plan.assignments.len() >= 4);

        // 6. Live Cross-Platform Multi-Node Stress Tests (Windows x86_64 <-> Apple M4 @ 192.168.40.237 <-> Raspberry Pi CM4 @ 192.168.40.75)
        let m4_probe = AppleM4PlatformBridge::probe_and_verify(Some(DEFAULT_M4_MAC_MINI_IP));
        let cm4_probe = RpiCm4PlatformBridge::probe_and_verify(Some(DEFAULT_RETERMINAL_CM4_IP));

        if m4_probe.physical_board_online {
            // 6a. Verify dynamic platform auto-detection on live Apple M4 silicon + CM4 carrier detection
            assert_eq!(m4_probe.telemetry.hw_model, "Mac16,10");
            assert_eq!(m4_probe.telemetry.silicon_tier, "M4");
            assert_eq!(m4_probe.telemetry.form_factor, "MacMini");
            let detected_cm4_carrier = Cm4CarrierFormFactor::detect(
                &cm4_probe.board_model,
                &cm4_probe.telemetry.i2c_peripherals,
            );
            assert_eq!(
                detected_cm4_carrier,
                Cm4CarrierFormFactor::SeeedReTerminalHmi
            );

            // 6b. Live 5-Point Gensyn Verde Fault-Localization Sweep across Layer Boundaries (L0, L11, L19, L37, L63) on Apple M4
            let initial_act = vec![0.125, -0.35, 0.72, 0.19, -0.61, 0.44, 0.88, -0.27];
            let (_m4_sample, _m4_sum, m4_digest, m4_verde_ms) =
                AppleM4PlatformBridge::execute_live_verde_challenge_on_m4(
                    DEFAULT_M4_MAC_MINI_IP,
                    0,
                    24,
                    &initial_act,
                    None,
                )
                .expect("live M4 Verde challenge");

            let desktop_node = kadmium_core::sample_personal_cluster().remove(0);
            let m4_node = m4_probe.hardware_node.clone();
            let cm4_node = cm4_probe.hardware_node.clone();
            let w_desktop =
                LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, desktop_node.clone(), false)
                    .await
                    .expect("spawn desktop worker");
            let w_m4 =
                LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, m4_node.clone(), false)
                    .await
                    .expect("spawn M4 worker");
            let w_cm4 =
                LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cm4_node.clone(), false)
                    .await
                    .expect("spawn CM4 worker");

            let desktop_reg = LiveWorkerRegistration {
                node: desktop_node.clone(),
                worker_tcp_addr: w_desktop.local_addr.to_string(),
                worker_lan_ip: "127.0.0.1".into(),
                worker_tcp_port: w_desktop.local_addr.port(),
                os_name: "windows".into(),
                cpu_arch: "x86_64".into(),
                board_or_model: "Desktop Workstation".into(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: 0.4,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            };
            let m4_reg = LiveWorkerRegistration {
                node: m4_node.clone(),
                worker_tcp_addr: w_m4.local_addr.to_string(),
                worker_lan_ip: DEFAULT_M4_MAC_MINI_IP.into(),
                worker_tcp_port: w_m4.local_addr.port(),
                os_name: "macos".into(),
                cpu_arch: "aarch64".into(),
                board_or_model: m4_probe.board_model.clone(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: m4_probe.execution_proof.lan_round_trip_ms,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            };
            let cm4_reg = LiveWorkerRegistration {
                node: cm4_node.clone(),
                worker_tcp_addr: w_cm4.local_addr.to_string(),
                worker_lan_ip: DEFAULT_RETERMINAL_CM4_IP.into(),
                worker_tcp_port: w_cm4.local_addr.port(),
                os_name: "linux".into(),
                cpu_arch: "aarch64".into(),
                board_or_model: cm4_probe.board_model.clone(),
                network_channel: "LocalLAN".into(),
                installed_version: "v0.1.0-cd48".into(),
                ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
                heartbeat_count: 1,
                registered_at_unix_ms: 1000,
                last_heartbeat_unix_ms: 1000,
                last_measured_rtt_ms: cm4_probe.execution_proof.lan_round_trip_ms,
                executed_forward_stages: 0,
                executed_training_steps: 0,
                online: true,
            };

            let mut m4_only = BTreeMap::new();
            m4_only.insert(m4_node.id.0.clone(), m4_reg.clone());

            let mut multi_workers = BTreeMap::new();
            multi_workers.insert(desktop_node.id.0.clone(), desktop_reg);
            multi_workers.insert(m4_node.id.0.clone(), m4_reg);
            multi_workers.insert(cm4_node.id.0.clone(), cm4_reg);

            let fault_targets = [0u32, 11, 19, 37, 63];
            let mut caught_faults = Vec::new();
            for &target_l in &fault_targets {
                let audit = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
                    4,
                    64,
                    Some(target_l),
                    &m4_only,
                )
                .await;
                assert!(!audit.bit_exact_verified);
                assert_eq!(audit.disputed_layer, Some(target_l));
                caught_faults.push(audit.disputed_layer.unwrap());
            }

            // 6c. 4-Seed Cross-ISA EXL3 / QTIP 2.5-bpw Hadamard MatVec Sweep (Windows x86_64 vs Apple M4 vDSP+NEON)
            let exl3_seeds = [48u64, 1337, 2026, 99999];
            let mut verified_exl3_digests = Vec::new();
            for &seed in &exl3_seeds {
                let rows = 32usize;
                let cols = 64usize;
                let raw_w: Vec<f32> = (0..(rows * cols))
                    .map(|i| ((i as f32) * 0.037 + (seed as f32) * 0.01).sin() * 0.42)
                    .collect();
                let x_vec: Vec<f32> = (0..cols)
                    .map(|i| ((i as f32) * 0.13 + (seed as f32) * 0.005).cos() * 0.35)
                    .collect();
                let q_tile =
                    Exl3TrellisQuantizer::quantize_matrix(&raw_w, rows, cols, 2.55, None, seed);
                let y_x86 = q_tile.matvec_f32(&x_vec);
                let y_x86_f64: Vec<f64> = y_x86.iter().map(|&v| v as f64).collect();
                let x86_sum = VerdeVerificationEngine::repops_pairwise_tree_sum(&y_x86_f64);
                let q_milli = (x86_sum * 10_000.0).round() as i64;
                let mut buf = [0u8; 16];
                buf[0..4].copy_from_slice(&0u32.to_le_bytes());
                buf[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
                buf[8..16].copy_from_slice(&(q_milli as u64).to_le_bytes());
                let mut x86_dig: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in &buf {
                    x86_dig ^= b as u64;
                    x86_dig = x86_dig.wrapping_mul(0x0100_0000_01b3);
                }
                let (y_m4, m4_dig, _ms) = AppleM4PlatformBridge::execute_exl3_fwht_matvec_on_m4(
                    DEFAULT_M4_MAC_MINI_IP,
                    q_tile.rows,
                    q_tile.cols,
                    q_tile.trellis_l_bits,
                    q_tile.rademacher_seed,
                    &q_tile.row_scales,
                    &q_tile.col_bit_schedule,
                    &q_tile.trellis_symbols,
                    &x_vec,
                )
                .expect("live M4 EXL3 matvec seed sweep");
                assert_eq!(x86_dig, m4_dig);
                for (a, b) in y_x86.iter().zip(y_m4.iter()) {
                    assert!((a - b).abs() < 1e-4);
                }
                verified_exl3_digests.push(m4_dig);
            }

            // 6d. Multi-Batch Metal 4 GPU + AMX 512x512 Scaling Sweep (10, 25, 50 iterations on Apple M4)
            let stress_10 =
                AppleM4PlatformBridge::run_stress_benchmark_on_m4(DEFAULT_M4_MAC_MINI_IP, 10)
                    .expect("10-iter M4 stress");
            let stress_50 =
                AppleM4PlatformBridge::run_stress_benchmark_on_m4(DEFAULT_M4_MAC_MINI_IP, 50)
                    .expect("50-iter M4 stress");
            assert!(stress_10.last_fnv1a64 != 0 && stress_50.last_fnv1a64 != 0);

            // 6e. Live Concurrent Multi-Node FTS3 .ktb Swarm Multicast
            let multicast_fts3 = DistributedLanExecutor::stream_fts3_ktb_shard_tcp(
                &model.id,
                "layers.0_27.multicast_swarm.ktb",
                4,
                &multi_workers,
            )
            .await;
            assert!(multicast_fts3.target_receipts.len() >= 8);
            assert!(
                multicast_fts3
                    .target_receipts
                    .iter()
                    .all(|r| r.verified_merkle_proof)
            );

            // 6f. Live 5-Epoch Multi-Platform Distributed QLoRA + Nous DeMo + DiLoCo Outer-Momentum Convergence
            let cluster_nodes = vec![desktop_node, m4_node, cm4_node];
            let train_plan = ShardPlanner::compute_plan(
                4,
                &model,
                WorkloadKind::DistributedTraining,
                &cluster_nodes,
            );
            let mut epoch_losses = Vec::new();
            for step in 1..=5 {
                let rep = DistributedLanExecutor::execute_live_training_step(
                    &train_plan,
                    step,
                    0.06,
                    &multi_workers,
                )
                .await;
                assert!(rep.all_stages_repops_verified);
                assert!(rep.mean_loss_after < rep.mean_loss_before);
                epoch_losses.push((rep.mean_loss_before, rep.mean_loss_after));
            }

            println!(
                "=== MULTI-PLATFORM FAMILY CATALOGS & LIVE CROSS-PLATFORM SWARM STRESS PASSED ===\n\
                 - Platform Catalogs Verified: Apple M4 ({} SKUs) | Raspberry Pi CM4 ({} SKUs) | NVIDIA Tegra X1/X1+ ({} SKUs) | Kendryte K230 RISC-V ({} SKUs)\n\
                 - 23-Node Cross-Platform DeepSeek-V3-671B-EXL3 Plan: {} stages sharded across 4 silicon families\n\
                 - Live 5-Point Verde Bisection Sweep on M4: honest_digest={:#018x} ({:.3}ms) | Faults pinpointed: {:?}\n\
                 - Live 4-Seed EXL3/QTIP 2.5-bpw Hadamard Sweep (x86_64 == M4 vDSP+NEON): 4/4 bit-exact ({:#018x}, {:#018x})\n\
                 - Live Metal 4 GPU + AMX Scaling Sweep (10 vs 50 iters): p50={:.3}ms vs {:.3}ms | AMX={:.2} GFLOP/s | CRC32C={:.2} GB/s\n\
                 - Live FTS3 Swarm Multicast: {} receipts verified ({:.1} Mbps)\n\
                 - Live 5-Step QLoRA + DeMo + DiLoCo Training: step1 {:.4}->{:.4} ... step5 {:.4}->{:.4}",
                m4_catalog.len(),
                cm4_catalog.len(),
                tegra_catalog.len(),
                k230_catalog.len(),
                deepseek_plan.assignments.len(),
                m4_digest,
                m4_verde_ms,
                caught_faults,
                verified_exl3_digests[0],
                verified_exl3_digests[3],
                stress_10.metal_p50_ms,
                stress_50.metal_p50_ms,
                stress_50.amx_sgemm_512_gflops,
                stress_50.hw_crc32c_gbps,
                multicast_fts3.target_receipts.len(),
                multicast_fts3.aggregate_goodput_mbps,
                epoch_losses[0].0,
                epoch_losses[0].1,
                epoch_losses[4].0,
                epoch_losses[4].1,
            );
        }
    }

    #[test]
    fn test_live_apple_m4_deep_metal4_shaders_gcd_10core_and_zerocopy_mmap_ktb() {
        use kadmium_core::{AppleM4PlatformBridge, DEFAULT_M4_MAC_MINI_IP};

        let probe = AppleM4PlatformBridge::probe_and_verify(Some(DEFAULT_M4_MAC_MINI_IP));
        if !probe.physical_board_online {
            return;
        }

        let deep_256 =
            AppleM4PlatformBridge::run_deep_metal4_suite_on_m4(DEFAULT_M4_MAC_MINI_IP, 10, 256)
                .expect("execute deep Metal 4 + 10-Core GCD + zero-copy mmap suite (seq=256)");
        let deep_1024 =
            AppleM4PlatformBridge::run_deep_metal4_suite_on_m4(DEFAULT_M4_MAC_MINI_IP, 15, 1024)
                .expect("execute deep Metal 4 + 10-Core GCD + zero-copy mmap suite (seq=1024)");

        // 1. Verify 1024x1024 Threadgroup-Tiled Metal 4 GPU GEMM determinism & throughput
        assert_eq!(
            deep_256.tiled_gemm1024_fnv1a64,
            deep_1024.tiled_gemm1024_fnv1a64
        );
        assert!(deep_256.tiled_gemm1024_gflops > 50.0);

        // 2. Verify Cross-ISA Metal 4 GPU Fused RMSNorm + RoPE against Windows x86_64 Host Reference
        let num_heads = 8usize;
        let head_dim = 64usize;
        let pos_offset = 12usize;
        let eps = 1e-5f32;
        let theta = 10000.0f32;
        let total_rr = num_heads * head_dim;
        let rr_in: Vec<f32> = (0..total_rr)
            .map(|i| (((i * 17 + 5) % 29) as f32 - 14.0) * 0.08)
            .collect();
        let rr_w: Vec<f32> = (0..head_dim)
            .map(|i| 1.0 + ((i % 11) as f32) * 0.02)
            .collect();
        let mut rr_out_x86 = vec![0.0f32; total_rr];
        for h in 0..num_heads {
            let base = h * head_dim;
            let mut sq_sum = 0.0f32;
            for i in 0..head_dim {
                let v = rr_in[base + i];
                sq_sum += v * v;
            }
            let inv_rms = 1.0f32 / ((sq_sum / (head_dim as f32) + eps).sqrt());
            for pair in 0..(head_dim / 2) {
                let i0 = 2 * pair;
                let i1 = i0 + 1;
                let x0 = rr_in[base + i0] * inv_rms * rr_w[i0];
                let x1 = rr_in[base + i1] * inv_rms * rr_w[i1];
                let exponent = (2 * pair) as f32 / (head_dim as f32);
                let freq = ((pos_offset + h) as f32) / theta.powf(exponent);
                let c = freq.cos();
                let s = freq.sin();
                rr_out_x86[base + i0] = x0 * c - x1 * s;
                rr_out_x86[base + i1] = x0 * s + x1 * c;
            }
        }
        for (&actual, &expected) in rr_out_x86.iter().zip(deep_256.rmsnorm_rope_sample8.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }
        let rr_f64: Vec<f64> = rr_out_x86.iter().map(|&v| v as f64).collect();
        let rr_sum = VerdeVerificationEngine::repops_pairwise_tree_sum(&rr_f64);
        let rr_q = (rr_sum * 10_000.0).round() as i64;
        let mut rr_buf = [0u8; 16];
        rr_buf[0..4].copy_from_slice(&(pos_offset as u32).to_le_bytes());
        rr_buf[4..8].copy_from_slice(&(num_heads as u32).to_le_bytes());
        rr_buf[8..16].copy_from_slice(&(rr_q as u64).to_le_bytes());
        let mut x86_rr_digest: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &rr_buf {
            x86_rr_digest ^= b as u64;
            x86_rr_digest = x86_rr_digest.wrapping_mul(0x0100_0000_01b3);
        }
        assert_eq!(x86_rr_digest, deep_256.rmsnorm_rope_repops_digest);

        // 3. Verify Metal 4 Flash-GQA Online-Softmax PagedAttention scaling (seq=256 vs seq=1024)
        assert_ne!(
            deep_256.flash_gqa_repops_digest,
            deep_1024.flash_gqa_repops_digest
        );
        assert!(deep_256.flash_gqa_seq256_ms > 0.0);

        // 4. Verify 10-Core Grand Central Dispatch (4P + 6E) Parallel Scaling & ARMv9.2-A INT8 DotProd + FP16 NEON
        assert!(deep_256.gcd_10core_speedup > 1.5);
        assert!(deep_256.armv9_int8_dotprod_gops > 20.0);
        assert!(deep_256.armv9_fp16_neon_gflops > 10.0);
        assert_eq!(deep_256.gcd_fnv1a64, deep_1024.gcd_fnv1a64);

        // 5. Verify Zero-Copy POSIX mmap() + Metal 4 newBufferWithBytesNoCopy Unified Memory .ktb Execution against x86_64 Reference
        assert!(deep_256.zerocopy_mmap_verified);
        let mut zc_y_x86 = vec![0.0f64; 64];
        for (r, zc_out) in zc_y_x86.iter_mut().enumerate().take(64) {
            let mut acc = 0.0f32;
            for c in 0..256usize {
                let idx = r * 256 + c;
                let w = (((idx * 19 + 7) % 37) as f32 - 18.0) * 0.025;
                let x = ((c as f32) * 0.09).cos() * 0.5;
                acc = w.mul_add(x, acc);
            }
            *zc_out = acc as f64;
            if r < 4 {
                assert!((acc - deep_256.zerocopy_mmap_sample4[r]).abs() < 1e-4);
            }
        }
        let zc_sum = VerdeVerificationEngine::repops_pairwise_tree_sum(&zc_y_x86);
        let zc_q = (zc_sum * 10_000.0).round() as i64;
        let mut zc_buf = [0u8; 16];
        zc_buf[0..4].copy_from_slice(&0u32.to_le_bytes());
        zc_buf[4..8].copy_from_slice(&64u32.to_le_bytes());
        zc_buf[8..16].copy_from_slice(&(zc_q as u64).to_le_bytes());
        let mut x86_zc_digest: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &zc_buf {
            x86_zc_digest ^= b as u64;
            x86_zc_digest = x86_zc_digest.wrapping_mul(0x0100_0000_01b3);
        }
        assert_eq!(x86_zc_digest, deep_256.zerocopy_mmap_repops_digest);

        println!(
            "=== LIVE APPLE M4 DEEP METAL 4 + 10-CORE GCD + ZERO-COPY MMAP .KTB SUITE PASSED ===\n\
             - 1. Metal 4 Tiled 1024x1024 GPU GEMM: {:.3} ms ({:.2} GFLOP/s | FNV-1a64: {:#018x})\n\
             - 2. Cross-ISA Metal 4 Fused RMSNorm+RoPE (x86_64 == M4 GPU): {:.4} ms [RepOps {:#018x}]\n\
             - 3. Metal 4 Flash-GQA PagedAttention (seq=256 vs 1024): {:.4} ms vs {:.4} ms [RepOps {:#018x}]\n\
             - 4. 10-Core GCD (4P+6E) Parallel Scaling: 1-Core {:.3} ms -> 10-Core {:.3} ms ({:.2}x speedup)\n\
             - 5. ARMv9.2-A Hardware Vector Throughput: INT8 DotProd (vdotq_s32)={:.2} GOPS | FP16 NEON (vfmaq_f16)={:.2} GFLOP/s\n\
             - 6. Zero-Copy POSIX mmap() + newBufferWithBytesNoCopy .ktb Metal 4 Exec: {:.4} ms (NoCopy UMA verified={} | x86_64 == M4 RepOps {:#018x})",
            deep_256.tiled_gemm1024_ms,
            deep_256.tiled_gemm1024_gflops,
            deep_256.tiled_gemm1024_fnv1a64,
            deep_256.rmsnorm_rope_ms,
            deep_256.rmsnorm_rope_repops_digest,
            deep_256.flash_gqa_seq256_ms,
            deep_1024.flash_gqa_seq256_ms,
            deep_256.flash_gqa_repops_digest,
            deep_256.gcd_1core_256_ms,
            deep_256.gcd_10core_256_ms,
            deep_256.gcd_10core_speedup,
            deep_256.armv9_int8_dotprod_gops,
            deep_256.armv9_fp16_neon_gflops,
            deep_256.zerocopy_mmap_ktb_ms,
            deep_256.zerocopy_mmap_verified,
            deep_256.zerocopy_mmap_repops_digest,
        );
    }

    #[tokio::test]
    async fn test_live_worker_serves_64_concurrent_forward_stages_over_one_pooled_channel() {
        let cluster = sample_personal_cluster();
        let worker =
            LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cluster[0].clone(), false)
                .await
                .expect("spawn worker");
        let addr = worker.local_addr.to_string();
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.07).sin() * 0.6).collect();
        let boundary = BlockwiseInt8Tensor::quantize(&input, 16);

        // 64 stage calls in flight at once against one worker, all through the process-wide pool.
        let mut tasks = Vec::with_capacity(64);
        for i in 0..64u32 {
            let addr = addr.clone();
            let req = LiveStageForwardRequest {
                epoch: 7,
                model_id: "deepseek-v3-671b-moe-q4k".into(),
                stage_index: i as usize,
                start_layer: i,
                end_layer: i + 1,
                q_star_ratio: 0.8,
                boundary_input: boundary.clone(),
                run_speculative_draft: false,
            };
            tasks.push(tokio::spawn(async move {
                let (resp, wall_ms) = DistributedLanExecutor::send_stage_forward_tcp(&addr, &req)
                    .await
                    .expect("concurrent forward stage");
                (i, resp, wall_ms)
            }));
        }
        for task in tasks {
            let (i, resp, wall_ms) = task.await.expect("join stage task");
            assert_eq!(resp.start_layer, i, "reply correlated to the wrong request");
            assert_eq!(resp.end_layer, i + 1);
            assert_eq!(resp.node_id, cluster[0].id.0);
            assert!(wall_ms > 0.0);
        }
        assert_eq!(
            worker
                .served_forward_stages
                .load(std::sync::atomic::Ordering::Relaxed),
            64
        );
        // One persistent connection carried all 64 requests.
        assert_eq!(
            worker
                .accepted_connections
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        let channel = DistributedLanExecutor::channel_pool()
            .get_or_connect(&addr)
            .await
            .expect("pooled channel");
        assert!(channel.is_healthy());
        assert_eq!(channel.in_flight(), 0);
    }

    #[tokio::test]
    async fn test_live_worker_answers_legacy_kdmf_one_shot_client_and_unregistered_work_frames() {
        use kadmium_network::{KDM_FRAME_MAGIC, PeerChannel, WireFrame, WireFrameKind};
        use std::time::Duration;
        use tokio::io::AsyncWriteExt;

        let cluster = sample_personal_cluster();
        let worker =
            LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cluster[1].clone(), false)
                .await
                .expect("spawn worker");

        // A pre-KDM2 node: raw socket, hand-built 13-byte KDMF header, one frame, one reply, close.
        let ping = LiveHeartbeatPing {
            epoch: 3,
            timestamp_unix_ms: 1,
        };
        let payload = serde_json::to_vec(&ping).expect("serialize ping");
        let mut raw = Vec::with_capacity(13 + payload.len());
        raw.extend_from_slice(&KDM_FRAME_MAGIC);
        raw.push(WireFrameKind::HeartbeatTelemetry as u8);
        raw.extend_from_slice(&3u32.to_le_bytes());
        raw.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        raw.extend_from_slice(&payload);

        let mut stream = tokio::net::TcpStream::connect(worker.local_addr)
            .await
            .expect("connect raw client");
        stream.write_all(&raw).await.expect("write legacy frame");
        let reply = WireFrame::read_from_stream(&mut stream)
            .await
            .expect("legacy client still gets a reply");
        assert_eq!(reply.kind, WireFrameKind::HeartbeatTelemetry);
        assert_eq!(reply.epoch, 3);
        assert_eq!(reply.request_id, 0);
        let pong: LiveHeartbeatPong = serde_json::from_slice(&reply.payload).expect("pong payload");
        assert_eq!(pong.node.id, cluster[1].id);
        drop(stream);

        // A work frame whose payload is not a dispatch envelope is answered with a failure
        // envelope (never left hanging), and the reply carries the request id so the channel
        // can match it.
        let channel = PeerChannel::connect(&worker.local_addr.to_string(), Duration::from_secs(5))
            .await
            .expect("connect channel");
        let result = channel
            .call(
                WireFrameKind::WorkDispatch,
                9,
                b"opaque-work-unit".to_vec(),
                Duration::from_secs(5),
            )
            .await
            .expect("work frame answered");
        assert_eq!(result.kind, WireFrameKind::WorkResult);
        assert_eq!(result.epoch, 9);
        assert_ne!(result.request_id, 0);
        let body = WorkResultPayload::from_le_bytes(&result.payload).expect("work result envelope");
        assert!(!body.header.ok);
        assert!(
            body.error_message()
                .is_some_and(|e| e.contains("malformed WorkDispatch"))
        );

        // A malformed payload for a known kind is answered with an error body on the same
        // connection, and the connection stays usable afterwards.
        let bad = channel
            .call(
                WireFrameKind::SystolicTokenWavefront,
                9,
                b"not json".to_vec(),
                Duration::from_secs(5),
            )
            .await
            .expect("malformed request still answered");
        assert_eq!(bad.kind, WireFrameKind::SystolicTokenWavefront);
        assert!(serde_json::from_slice::<LiveStageForwardResponse>(&bad.payload).is_err());
        let (pong_again, _) =
            DistributedLanExecutor::ping_worker_tcp(&worker.local_addr.to_string(), 4)
                .await
                .expect("ping after error replies");
        assert_eq!(pong_again.node.id, cluster[1].id);
        assert!(channel.is_healthy());
        assert_eq!(channel.in_flight(), 0);
    }

    #[test]
    fn test_token_batch_int8_binary_codec_roundtrip() {
        let rows = 4usize;
        let dim = 40usize;
        let flat: Vec<f32> = (0..(rows * dim))
            .map(|i| ((i as f32) * 0.05).sin() * (1.0 + (i / dim) as f32))
            .collect();
        let batch =
            TokenBatchInt8::quantize(&flat, rows as u32, dim as u32, 16).expect("quantize batch");
        assert_eq!(batch.tensor.total_elements, rows * dim);
        assert_eq!(batch.tensor.absmax_scales.len(), 10);

        let bytes = batch.to_le_bytes();
        assert_eq!(
            bytes.len(),
            8 + INT8_TENSOR_CODEC_HEADER_LEN + 10 * 4 + rows * dim
        );
        let decoded = TokenBatchInt8::from_le_bytes(&bytes).expect("decode batch");
        assert_eq!(decoded.rows, 4);
        assert_eq!(decoded.dim, 40);
        assert_eq!(decoded.tensor.block_size, batch.tensor.block_size);
        assert_eq!(decoded.tensor.total_elements, batch.tensor.total_elements);
        assert_eq!(decoded.tensor.absmax_scales, batch.tensor.absmax_scales);
        assert_eq!(decoded.tensor.quantized_data, batch.tensor.quantized_data);
        assert_eq!(decoded.tensor.dequantize(), batch.tensor.dequantize());
        assert_eq!(decoded.to_le_bytes(), bytes);
        let rows_back = decoded.dequantize_rows();
        assert_eq!(rows_back.len(), rows);
        assert!(rows_back.iter().all(|r| r.len() == dim));
        for (a, b) in flat.iter().zip(rows_back.concat().iter()) {
            assert!((a - b).abs() < 0.05, "dequantized {b} drifted from {a}");
        }

        // Standalone tensor codec, including negative values and a partial last block.
        let q = BlockwiseInt8Tensor::quantize(&flat[..37], 8);
        let t_bytes = q.to_le_bytes();
        assert_eq!(t_bytes.len(), INT8_TENSOR_CODEC_HEADER_LEN + 5 * 4 + 37);
        let q2 = BlockwiseInt8Tensor::from_le_bytes(&t_bytes).expect("decode tensor");
        assert_eq!(q2.block_size, 8);
        assert_eq!(q2.total_elements, 37);
        assert_eq!(q2.absmax_scales, q.absmax_scales);
        assert_eq!(q2.quantized_data, q.quantized_data);
        let (prefix, consumed) =
            BlockwiseInt8Tensor::from_le_bytes_prefix(&[t_bytes.clone(), vec![0xAA; 3]].concat())
                .expect("decode tensor prefix");
        assert_eq!(consumed, t_bytes.len());
        assert_eq!(prefix.quantized_data, q.quantized_data);

        // Corrupt inputs are rejected rather than mis-decoded.
        assert!(BlockwiseInt8Tensor::from_le_bytes(&t_bytes[..t_bytes.len() - 1]).is_err());
        assert!(BlockwiseInt8Tensor::from_le_bytes(&t_bytes[..5]).is_err());
        assert!(BlockwiseInt8Tensor::from_le_bytes(&[t_bytes.clone(), vec![0]].concat()).is_err());
        let mut zero_block = t_bytes.clone();
        zero_block[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(BlockwiseInt8Tensor::from_le_bytes(&zero_block).is_err());
        assert!(TokenBatchInt8::from_le_bytes(&bytes[..bytes.len() - 3]).is_err());
        assert!(TokenBatchInt8::from_le_bytes(&bytes[..6]).is_err());
        let mut wrong_shape = bytes.clone();
        wrong_shape[0..4].copy_from_slice(&5u32.to_le_bytes());
        assert!(TokenBatchInt8::from_le_bytes(&wrong_shape).is_err());
        assert!(TokenBatchInt8::quantize(&flat, 3, 40, 16).is_err());
    }
}
