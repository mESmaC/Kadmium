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
    pub page_size_tokens: u32,
    pub allocated_kv_pages: u32,
    pub max_context_tokens_supported: u32,
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
        let mut out = Vec::new();

        for assign in &plan.assignments {
            if assign.layer_count == 0 {
                continue;
            }
            let node = nodes.iter().find(|n| n.id == assign.node_id);
            let total_fast_and_ram = node
                .map(|n| n.available_fast_memory_gb() + n.memory.system_ram_gb * 0.35)
                .unwrap_or(8.0);
            let kv_budget_gb = (total_fast_and_ram * 0.22).max(0.5);
            let kv_per_tok_per_layer_gb = (model.kv_cache_per_layer_gb / 4096.0).max(1e-6);
            let max_tokens =
                ((kv_budget_gb / (kv_per_tok_per_layer_gb * assign.layer_count as f64)).floor()
                    as u32)
                    .clamp(2048, 131_072);
            let pages = max_tokens / page_tokens;

            out.push(NodePagedKvAllocation {
                hostname: assign.hostname.clone(),
                assigned_layers: assign.layer_count,
                kv_budget_gb: (kv_budget_gb * 100.0).round() / 100.0,
                page_size_tokens: page_tokens,
                allocated_kv_pages: pages,
                max_context_tokens_supported: pages * page_tokens,
            });
        }

        out
    }
}
