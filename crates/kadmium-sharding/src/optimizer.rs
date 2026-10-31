//! # `kadmium_sharding::optimizer`
//!
//! Min-Max Pipeline Stage Latency Optimizer (`MinMaxStageLatencyOptimizer`) and
//! Distributed PagedAttention KV-Cache Planner (`DistributedPagedKvPlanner`).

use crate::ClusterShardPlan;
use kadmium_core::{HardwareNode, ModelSpec};
use serde::{Deserialize, Serialize};

/// Report from the Min-Max Pipeline Bottleneck Optimizer (`MinMaxStageLatencyOptimizer`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageLatencyOptimizationReport {
    pub initial_bottleneck_ms: f64,
    pub optimized_bottleneck_ms: f64,
    pub latency_reduction_pct: f64,
    pub refined_stage_layers: Vec<(String, u32, f64)>,
}

/// Min-Max Pipeline Stage Latency Optimizer.
///
/// Refines integer layer counts $(L_1, \dots, L_K)$ across pipeline stages so that
/// $\max_k (L_k \cdot \tau_k + \text{link\_lat}_k)$ is minimized subject to $\sum_k L_k = L_{\text{total}}$.
pub struct MinMaxStageLatencyOptimizer;

impl MinMaxStageLatencyOptimizer {
    pub fn optimize_stage_boundaries(
        plan: &ClusterShardPlan,
        nodes: &[HardwareNode],
    ) -> StageLatencyOptimizationReport {
        let mut stages: Vec<(String, u32, f64, f64)> = Vec::new();
        for assign in &plan.assignments {
            if assign.layer_count == 0 {
                continue;
            }
            let node = nodes.iter().find(|n| n.id == assign.node_id);
            let tflops = node.map(|n| n.fp16_tflops.max(1.5)).unwrap_or(10.0);
            let nvme_bw = node
                .map(|n| n.memory.nvme_read_bw_gbps.max(1.0))
                .unwrap_or(3.0);
            let link_ms = node.map(|n| n.network_latency_ms).unwrap_or(1.0);

            let per_layer_ms = (18.0 / tflops) + (0.45 / nvme_bw);
            stages.push((
                assign.hostname.clone(),
                assign.layer_count,
                per_layer_ms,
                link_ms,
            ));
        }

        let calc_max_lat = |st: &[(String, u32, f64, f64)]| -> f64 {
            st.iter()
                .map(|(_, count, per_l, link)| (*count as f64) * per_l + link)
                .fold(0.0_f64, f64::max)
        };

        let initial_bottleneck = calc_max_lat(&stages);

        for _step in 0..64 {
            if stages.len() < 2 {
                break;
            }
            let current_max = calc_max_lat(&stages);
            let slowest_idx = stages
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let la = (a.1 as f64) * a.2 + a.3;
                    let lb = (b.1 as f64) * b.2 + b.3;
                    la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            if stages[slowest_idx].1 <= 1 {
                break;
            }

            let fastest_idx = stages
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let la = ((a.1 + 1) as f64) * a.2 + a.3;
                    let lb = ((b.1 + 1) as f64) * b.2 + b.3;
                    la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            if slowest_idx == fastest_idx {
                break;
            }

            stages[slowest_idx].1 -= 1;
            stages[fastest_idx].1 += 1;
            let next_max = calc_max_lat(&stages);
            if next_max >= current_max {
                stages[slowest_idx].1 += 1;
                stages[fastest_idx].1 -= 1;
                break;
            }
        }

        let optimized_bottleneck = calc_max_lat(&stages);
        let reduction = if initial_bottleneck > 0.0 {
            ((initial_bottleneck - optimized_bottleneck) / initial_bottleneck * 100.0).max(0.0)
        } else {
            0.0
        };

        let refined_stage_layers = stages
            .into_iter()
            .map(|(host, cnt, per_l, link)| {
                let stage_lat = (((cnt as f64) * per_l + link) * 100.0).round() / 100.0;
                (host, cnt, stage_lat)
            })
            .collect();

        StageLatencyOptimizationReport {
            initial_bottleneck_ms: (initial_bottleneck * 100.0).round() / 100.0,
            optimized_bottleneck_ms: (optimized_bottleneck * 100.0).round() / 100.0,
            latency_reduction_pct: (reduction * 10.0).round() / 10.0,
            refined_stage_layers,
        }
    }
}

/// Per-node PagedAttention KV-Cache allocation entry (`16 tokens/page`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePagedKvAllocation {
    pub hostname: String,
    pub assigned_layers: u32,
    pub kv_budget_gb: f64,
    pub warm_ram_kv_budget_gb: f64,
    pub nvme_kv_budget_gb: f64,
    pub page_size_tokens: u32,
    pub allocated_kv_pages: u32,
    pub max_context_tokens_supported: u32,
    pub max_tiered_context_tokens_supported: u32,
}

/// Cluster-wide full-attention context capacity report combining fast VRAM/Unified memory,
/// remaining host System RAM, and remaining NVMe `.ktb` cache capacity, bounded by the model's
/// native pretrained context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterContextCapacityReport {
    pub model_id: String,
    pub model_native_context_tokens: u32,
    pub active_pipeline_stages: usize,
    pub zigzag_ring_cp_degree: usize,
    pub in_memory_stage_context_tokens: u32,
    pub tiered_stage_context_tokens: u32,
    pub cluster_in_memory_context_tokens: u32,
    pub cluster_max_full_attention_context_tokens: u32,
    pub total_hot_kv_budget_gb: f64,
    pub total_warm_ram_kv_budget_gb: f64,
    pub total_nvme_kv_budget_gb: f64,
    pub planning_note: String,
    pub node_allocations: Vec<NodePagedKvAllocation>,
}

/// Distributed PagedAttention KV-Cache Planner across heterogeneous VRAM and Unified Memory.
pub struct DistributedPagedKvPlanner;

impl DistributedPagedKvPlanner {
    pub fn plan_kv_cache_pages(
        plan: &ClusterShardPlan,
        model: &ModelSpec,
        nodes: &[HardwareNode],
    ) -> Vec<NodePagedKvAllocation> {
        let page_tokens = 16u32;
        let native_cap = model.native_max_context_tokens().max(page_tokens);
        let min_floor = 512u32.min(native_cap);
        let kv_per_tok_per_layer_gb =
            (model.fp16_kv_bytes_per_token_per_layer() as f64) / 1_073_741_824.0;
        let mut out = Vec::new();

        for assign in &plan.assignments {
            if assign.layer_count == 0 {
                continue;
            }
            let node = nodes.iter().find(|n| n.id == assign.node_id);

            // Fast accelerator memory remaining after hot resident weights (never double-counting weights).
            let avail_fast_gb = node.map(|n| n.available_fast_memory_gb()).unwrap_or(8.0);
            let free_fast_for_kv_gb = (avail_fast_gb - assign.tiered_memory.hot_vram_or_unified_gb)
                .max(assign.tiered_memory.state_and_cache_gb)
                .max(0.25);

            // System RAM remaining in the polite resource offer after warm weight placement.
            let offered_ram_gb = node
                .map(|n| {
                    kadmium_core::ResourceOffer::default_for(n)
                        .under_polite_state(n, n.polite_state)
                        .system_ram_gb
                })
                .unwrap_or(4.0);
            let free_ram_after_weights_gb =
                (offered_ram_gb - assign.tiered_memory.warm_system_ram_gb).max(0.0);
            let warm_ram_kv_gb = free_ram_after_weights_gb * 0.50;

            // NVMe cache remaining after cold weight streaming banks.
            let free_nvme_after_weights_gb = node
                .map(|n| {
                    (n.memory.nvme_cache_gb - assign.tiered_memory.cold_nvme_stream_gb).max(0.0)
                })
                .unwrap_or(0.0);
            let nvme_kv_gb = (free_nvme_after_weights_gb * 0.25).min(64.0);

            let stage_kv_per_tok_gb =
                (kv_per_tok_per_layer_gb * assign.layer_count as f64).max(1e-9);

            let raw_hot_tokens = ((free_fast_for_kv_gb + warm_ram_kv_gb * 0.35)
                / stage_kv_per_tok_gb)
                .floor() as u32;
            let pages = raw_hot_tokens.clamp(min_floor, native_cap) / page_tokens;
            let max_tokens = pages * page_tokens;

            let total_tiered_kv_gb = free_fast_for_kv_gb + warm_ram_kv_gb + nvme_kv_gb;
            let raw_tiered_tokens = (total_tiered_kv_gb / stage_kv_per_tok_gb).floor() as u32;
            let tiered_pages = raw_tiered_tokens.clamp(max_tokens, native_cap) / page_tokens;
            let max_tiered_tokens = tiered_pages * page_tokens;

            out.push(NodePagedKvAllocation {
                hostname: assign.hostname.clone(),
                assigned_layers: assign.layer_count,
                kv_budget_gb: (free_fast_for_kv_gb * 100.0).round() / 100.0,
                warm_ram_kv_budget_gb: (warm_ram_kv_gb * 100.0).round() / 100.0,
                nvme_kv_budget_gb: (nvme_kv_gb * 100.0).round() / 100.0,
                page_size_tokens: page_tokens,
                allocated_kv_pages: pages,
                max_context_tokens_supported: max_tokens,
                max_tiered_context_tokens_supported: max_tiered_tokens,
            });
        }

        out
    }

    /// Computes the end-to-end cluster context capacity report across all active pipeline stages,
    /// bounded by the model's native pretrained context window.
    pub fn plan_cluster_context_capacity(
        plan: &ClusterShardPlan,
        model: &ModelSpec,
        nodes: &[HardwareNode],
    ) -> ClusterContextCapacityReport {
        let native_cap = model.native_max_context_tokens();
        let allocs = Self::plan_kv_cache_pages(plan, model, nodes);
        let active_stages = allocs.len();

        // Zig-Zag Ring Context Parallelism is only applicable when the node pool has multiple
        // full pipeline replicas beyond the active pipeline stages in `plan`.
        let cp_degree = if active_stages > 0 && nodes.len() >= active_stages * 2 {
            (nodes.len() / active_stages).min(4)
        } else {
            1
        };

        let in_memory_stage = allocs
            .iter()
            .map(|a| a.max_context_tokens_supported)
            .min()
            .unwrap_or(0)
            .min(native_cap);
        let tiered_stage = allocs
            .iter()
            .map(|a| a.max_tiered_context_tokens_supported)
            .min()
            .unwrap_or(0)
            .min(native_cap);

        let cluster_in_memory = in_memory_stage
            .saturating_mul(cp_degree as u32)
            .min(native_cap);
        let cluster_max_full = tiered_stage
            .saturating_mul(cp_degree as u32)
            .min(native_cap);

        let total_hot: f64 = allocs.iter().map(|a| a.kv_budget_gb).sum();
        let total_warm: f64 = allocs.iter().map(|a| a.warm_ram_kv_budget_gb).sum();
        let total_nvme: f64 = allocs.iter().map(|a| a.nvme_kv_budget_gb).sum();

        ClusterContextCapacityReport {
            model_id: model.id.clone(),
            model_native_context_tokens: native_cap,
            active_pipeline_stages: active_stages,
            zigzag_ring_cp_degree: cp_degree,
            in_memory_stage_context_tokens: in_memory_stage,
            tiered_stage_context_tokens: tiered_stage,
            cluster_in_memory_context_tokens: cluster_in_memory,
            cluster_max_full_attention_context_tokens: cluster_max_full,
            total_hot_kv_budget_gb: (total_hot * 100.0).round() / 100.0,
            total_warm_ram_kv_budget_gb: (total_warm * 100.0).round() / 100.0,
            total_nvme_kv_budget_gb: (total_nvme * 100.0).round() / 100.0,
            planning_note: "Theoretical KV capacity planner estimate bounded by model native context window; warm RAM and NVMe KV tiers incur PCIe/host transfer latency and do not indicate live model readiness.".to_string(),
            node_allocations: allocs,
        }
    }
}
