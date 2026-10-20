//! TPU/Compute-in-Memory Dataflow Stream Planner & Sub-Compute Environment Partitioner (`kadmium-sharding::slices`).

use crate::{ClusterShardPlan, ShardPlanner};
use kadmium_core::{DataflowMode, HardwareNode, ModelSpec, SlicePurpose, WorkloadKind};
use serde::{Deserialize, Serialize};

/// Performance & bandwidth profile for a specific TPU/CIM-inspired `DataflowMode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowModeProfile {
    pub mode: DataflowMode,
    pub stationary_tensor: String,
    pub streaming_tensor: String,
    /// Arithmetic intensity in FLOPs per streamed byte.
    pub arithmetic_intensity_flops_per_byte: f64,
    /// Per-step or per-layer wire/DMA volume streamed across tiers/nodes in MiB.
    pub streamed_volume_per_step_mb: f64,
    /// Percentage of streaming latency hidden behind compute via double-buffering ($0..=100\%$).
    pub compute_stream_overlap_pct: f64,
    /// Estimated effective throughput in tokens/sec for the evaluated batch size.
    pub estimated_tokens_per_sec: f64,
    pub best_suited_for: String,
}

/// Comparative analysis between Compute-in-Memory Dynamic Token Streaming and Dynamic Weight Streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowComparisonReport {
    pub model_id: String,
    pub evaluated_batch_tokens: u32,
    pub recommended_low_batch_mode: DataflowMode,
    pub recommended_high_batch_mode: DataflowMode,
    /// Batch size (in concurrent tokens) where Token-Stationary Weight Streaming overtakes
    /// Weight-Stationary Token Streaming.
    pub crossover_batch_tokens: u32,
    pub profiles: Vec<DataflowModeProfile>,
}

/// Planner evaluating TPU / Compute-in-Memory inspired dataflow modes on a Kadmium cluster.
pub struct DataflowStreamPlanner;

impl DataflowStreamPlanner {
    pub fn evaluate_dataflows(
        model: &ModelSpec,
        nodes: &[HardwareNode],
        batch_tokens: u32,
    ) -> DataflowComparisonReport {
        let total_tflops: f64 = nodes.iter().map(|n| n.effective_fp16_tflops()).sum();
        let total_mem_bw_gbps: f64 = nodes
            .iter()
            .map(|n| n.memory.effective_bandwidth_gbps())
            .sum();
        let total_nvme_bw_gbps: f64 = nodes.iter().map(|n| n.memory.nvme_read_bw_gbps).sum();

        let active_weight_gb = if model.is_moe {
            (model.num_layers as f64)
                * (model.dense_layer_size_gb
                    + (model.active_experts_per_token as f64) * model.single_expert_size_gb)
        } else {
            model.total_weight_footprint_gb()
        };

        // 1. Compute-in-Memory / Weight-Stationary Dynamic Token Streaming:
        let token_packet_mb =
            ((batch_tokens as f64) * 0.016 * (nodes.len() as f64) * 100.0).round() / 100.0;
        let cim_tok_s =
            ((total_mem_bw_gbps / active_weight_gb.max(1.0)) * 1.15 * 10.0).round() / 10.0;

        // 2. Token-Stationary Dynamic Weight Streaming:
        let weight_stream_mb = (active_weight_gb * 1024.0 * 10.0).round() / 10.0;
        let arith_intensity = (batch_tokens as f64) * 2.0;
        let overlap_pct = ((batch_tokens as f64 / 512.0) * 92.0).clamp(18.0, 96.5);
        let weight_stream_tok_s = (((total_nvme_bw_gbps * (batch_tokens as f64).sqrt() * 0.85)
            + (total_tflops * 0.12))
            * 10.0)
            .round()
            / 10.0;

        // 3. Disaggregated Dual-Stream (Token-Stationary Prefill + CIM Weight-Stationary Decode):
        let dual_tok_s = ((cim_tok_s.max(weight_stream_tok_s) * 1.22) * 10.0).round() / 10.0;

        // 4. Adaptive Tiered Hybrid (FreeToken q* + Colibri NVMe):
        let hybrid_tok_s = ((cim_tok_s * 0.92 + weight_stream_tok_s * 0.35) * 10.0).round() / 10.0;

        let profiles = vec![
            DataflowModeProfile {
                mode: DataflowMode::ComputeInMemoryTokenStreaming,
                stationary_tensor: "Model Weights (Pinned in VRAM / Unified / DDR RAM Banks)"
                    .into(),
                streaming_tensor: "Systolic Token Wavefront Packets (O(d) KB per hop)".into(),
                arithmetic_intensity_flops_per_byte: 64.0,
                streamed_volume_per_step_mb: token_packet_mb,
                compute_stream_overlap_pct: 94.0,
                estimated_tokens_per_sec: cim_tok_s,
                best_suited_for:
                    "Low-latency interactive decode (Batch 1..8) & Speculative Decoding".into(),
            },
            DataflowModeProfile {
                mode: DataflowMode::TokenStationaryWeightStreaming,
                stationary_tensor: "Token Activation Batch (Pinned in Fast GPU VRAM / SRAM)".into(),
                streaming_tensor: "Double-Buffered .ktb Weight Tiles (NVMe / NAS -> GPU DMA)"
                    .into(),
                arithmetic_intensity_flops_per_byte: arith_intensity,
                streamed_volume_per_step_mb: weight_stream_mb,
                compute_stream_overlap_pct: (overlap_pct * 10.0).round() / 10.0,
                estimated_tokens_per_sec: weight_stream_tok_s,
                best_suited_for:
                    "High-throughput batch prefill, evals & running 400B+ models on 1-2 GPUs".into(),
            },
            DataflowModeProfile {
                mode: DataflowMode::DisaggregatedDualStream,
                stationary_tensor:
                    "Dual: Stationary Tokens (Prefill) + Stationary Weights (Decode)".into(),
                streaming_tensor: "Weight-Streamed Prefill -> CIM Token-Streamed Decode".into(),
                arithmetic_intensity_flops_per_byte: arith_intensity.max(64.0),
                streamed_volume_per_step_mb: (token_packet_mb * 2.5 * 100.0).round() / 100.0,
                compute_stream_overlap_pct: 95.5,
                estimated_tokens_per_sec: dual_tok_s,
                best_suited_for: "Production multi-user serving & long-context agent workflows"
                    .into(),
            },
            DataflowModeProfile {
                mode: DataflowMode::AdaptiveTieredHybrid,
                stationary_tensor: "Dense Trunk + Hot Experts (VRAM/Unified)".into(),
                streaming_tensor: "Cold MoE Experts (NVMe) + q* Host/GPU Split".into(),
                arithmetic_intensity_flops_per_byte: 48.0,
                streamed_volume_per_step_mb: (weight_stream_mb * 0.18 * 10.0).round() / 10.0,
                compute_stream_overlap_pct: 89.0,
                estimated_tokens_per_sec: hybrid_tok_s,
                best_suited_for: "General-purpose mixed MoE inference and QLoRA fine-tuning".into(),
            },
        ];

        DataflowComparisonReport {
            model_id: model.id.clone(),
            evaluated_batch_tokens: batch_tokens,
            recommended_low_batch_mode: DataflowMode::ComputeInMemoryTokenStreaming,
            recommended_high_batch_mode: DataflowMode::TokenStationaryWeightStreaming,
            crossover_batch_tokens: 256,
            profiles,
        }
    }
}

/// An isolated **Sub-Compute Environment (Cluster Slice)** carved out of a larger node pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeSubCluster {
    pub slice_id: String,
    pub display_name: String,
    pub purpose: SlicePurpose,
    pub workload: WorkloadKind,
    pub dataflow_mode: DataflowMode,
    pub node_count: usize,
    pub aggregate_fp16_tflops: f64,
    pub aggregate_fast_memory_gb: f64,
    pub aggregate_system_ram_gb: f64,
    pub aggregate_nvme_gb: f64,
    pub api_namespace: String,
    /// Whether idle nodes in this slice can be elastically borrowed by training/batch slices.
    pub allow_elastic_borrowing: bool,
    pub shard_plan: ClusterShardPlan,
}

/// Complete multi-environment partitioning layout for a global compute pool (e.g., 100 nodes -> 5 slices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPoolSliceLayout {
    pub total_pool_nodes: usize,
    pub num_slices: usize,
    pub nodes_per_slice: usize,
    pub unassigned_reserve_nodes: usize,
    pub slices: Vec<ComputeSubCluster>,
}

/// Partitioner that sections a global pool of nodes (e.g., 100 nodes) into isolated
/// Sub-Compute Environments.
pub struct SubClusterPartitioner;

impl SubClusterPartitioner {
    /// Partitions `pool` into `num_slices` capability-balanced sub-compute environments.
    pub fn partition_pool(
        pool: &[HardwareNode],
        num_slices: usize,
        model: &ModelSpec,
    ) -> ClusterPoolSliceLayout {
        let k = num_slices.max(1).min(pool.len().max(1));
        let nodes_per_slice = pool.len() / k;
        let unassigned_reserve_nodes = pool.len() % k;

        let slice_templates = [
            (
                "slice-01-prod-deploy",
                "Production Inference Cluster",
                SlicePurpose::ProductionDeployment,
                WorkloadKind::Inference,
                DataflowMode::ComputeInMemoryTokenStreaming,
                false,
            ),
            (
                "slice-02-staging-test",
                "Staging & Canary Test Environment",
                SlicePurpose::StagingAndTesting,
                WorkloadKind::Inference,
                DataflowMode::DisaggregatedDualStream,
                true,
            ),
            (
                "slice-03-train-diloco",
                "Distributed Training Lab (1F1B + DeMo)",
                SlicePurpose::DistributedTrainingLab,
                WorkloadKind::DistributedTraining,
                DataflowMode::AdaptiveTieredHybrid,
                true,
            ),
            (
                "slice-04-rl-dream-rsi",
                "RL-Swarm & Dream-RSI Discovery Arena",
                SlicePurpose::RlAndDreamRsiArena,
                WorkloadKind::DistributedTraining,
                DataflowMode::ComputeInMemoryTokenStreaming,
                true,
            ),
            (
                "slice-05-batch-eval",
                "High-Throughput Weight-Streamed Batch Pool",
                SlicePurpose::BatchAndEvalPool,
                WorkloadKind::BatchPipeline,
                DataflowMode::TokenStationaryWeightStreaming,
                true,
            ),
        ];

        let mut slice_buckets: Vec<Vec<HardwareNode>> = vec![Vec::new(); k];
        for (idx, node) in pool.iter().take(k * nodes_per_slice).enumerate() {
            slice_buckets[idx % k].push(node.clone());
        }

        let mut slices = Vec::with_capacity(k);
        for (s_idx, bucket) in slice_buckets.into_iter().enumerate() {
            let (base_id, base_name, purpose, workload, dataflow_mode, elastic) =
                slice_templates[s_idx % slice_templates.len()];
            let slice_id = if s_idx < slice_templates.len() {
                base_id.to_string()
            } else {
                format!("slice-{:02}-custom", s_idx + 1)
            };

            let agg_tflops: f64 = bucket.iter().map(|n| n.effective_fp16_tflops()).sum();
            let agg_fast_mem: f64 = bucket.iter().map(|n| n.available_fast_memory_gb()).sum();
            let agg_ram: f64 = bucket.iter().map(|n| n.memory.system_ram_gb).sum();
            let agg_nvme: f64 = bucket.iter().map(|n| n.memory.nvme_cache_gb).sum();

            let shard_plan = ShardPlanner::compute_plan(1, model, workload, &bucket);

            slices.push(ComputeSubCluster {
                slice_id: slice_id.clone(),
                display_name: base_name.to_string(),
                purpose,
                workload,
                dataflow_mode,
                node_count: bucket.len(),
                aggregate_fp16_tflops: (agg_tflops * 10.0).round() / 10.0,
                aggregate_fast_memory_gb: (agg_fast_mem * 10.0).round() / 10.0,
                aggregate_system_ram_gb: (agg_ram * 10.0).round() / 10.0,
                aggregate_nvme_gb: (agg_nvme * 10.0).round() / 10.0,
                api_namespace: format!("/v1/slices/{}", slice_id),
                allow_elastic_borrowing: elastic,
                shard_plan,
            });
        }

        ClusterPoolSliceLayout {
            total_pool_nodes: pool.len(),
            num_slices: k,
            nodes_per_slice,
            unassigned_reserve_nodes,
            slices,
        }
    }
}
