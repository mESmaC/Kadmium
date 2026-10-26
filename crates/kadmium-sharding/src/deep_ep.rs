//! Expert-parallel dispatch layouts, asymmetric-domain forwarding and hook-overlap accounting,
//! after DeepSeek's `DeepEP`.
//!
//! `DeepEP` (`deepseek-ai/DeepEP`) is the communication library underneath DeepSeek-V3's expert
//! parallelism. It separates a layer's all-to-all into three explicit pieces:
//!
//! 1. **Layout inspection (`Buffer::get_dispatch_layout`):**
//!    Before moving hidden states, `get_dispatch_layout(topk_idx, num_experts)` scans the router's
//!    selections and returns:
//!    - `num_tokens_per_rank`: how many distinct rows of the batch actually need to visit each
//!      expert holder. When one row routes to three experts that sit on the same holder, the row's
//!      activation crosses the wire to that holder once (`is_token_in_rank[rank][row] = true`),
//!      not three times; when a holder serves none of a row's selected experts (or another EPLB
//!      replica serves that row via [`crate::ExpertRowShare`]), the row is not sent to that holder
//!      at all.
//!    - `num_tokens_per_rdma_rank` (here `num_tokens_per_segment` over [`crate::LinkSegments`]):
//!      how many distinct rows cross into each network segment. In DeepEP's asymmetric-domain
//!      forwarding, a token crossing from the local NVLink domain to a remote RDMA node travels the
//!      inter-node link once to that node and fans out over the node's fast intra-node links. Here
//!      a row needed by several holders on the same remote [`crate::LinkSegments`] domain crosses
//!      the slow inter-segment link once and relays inside the segment.
//!    - `num_tokens_per_expert`: how many rows landed on each expert `0..num_experts`, which is
//!      the device-side `masked_m` vector consumed by `DeepGEMM`'s masked grouped GEMM.
//!    - `is_token_in_rank`: the `[holder][row]` incidence table.
//! 2. **Low-latency quantized `dispatch` and `combine` (`low_latency_dispatch`, `low_latency_combine`):**
//!    For inference decoding, DeepEP dispatches activations quantized to 8-bit (`FP8` with
//!    fine-grained `1 x G` sub-block scales) and combines expert partials back in reduced
//!    precision, cutting wire bytes by nearly 4x compared to `f32` while routing only matched rows
//!    to each holder.
//! 3. **Zero-SM hook-based overlap (`return_recv_hook=True`):**
//!    Both `low_latency_dispatch` and `low_latency_combine` return a completion hook (`recv_hook`)
//!    that drives RDMA/network progression in the background without occupying GPU Streaming
//!    Multiprocessors (`0 SM` occupation). While the network transfer is in flight, the accelerator
//!    runs local work that does not depend on the routed experts—either the layer's shared expert
//!    or the other micro-batch in a two-microbatch ping-pong—and calls `recv_hook()` immediately
//!    before consuming the received buffer.
//!
//! Left out, deliberately: NVSHMEM symmetric heaps, InfiniBand verbs Queue Pairs (`ibgda`), and
//! PTX `ld.global.nc` warp polling. What is kept is `DispatchLayout` (`get_dispatch_layout` with
//! EPLB replica awareness and `LinkSegments` domain coalescing), the selective/quantized wire
//! envelopes in `kadmium-runtime`, the `RecvHook` overlap execution helper, and the modelled byte
//! and latency accounting (`DeepEpStepAccount`) attached to [`crate::StrategyChoice`].

use kadmium_core::{HardwareNode, ModelSpec, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ExpertPlacementPlan, ExpertPopularity, ExpertSliceParams, LinkSegments, StepCostEstimate,
};

/// Sub-block size used when accounting for `1 x G` fine-grained FP8/INT8 dispatch payloads.
pub const DEEP_EP_DISPATCH_BLOCK_SIZE: usize = 128;

/// Routing layout computed prior to token dispatch at one MoE layer, after DeepEP's
/// `Buffer::get_dispatch_layout`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchLayout {
    pub layer: u32,
    pub rows: u32,
    pub num_experts: u32,
    /// Distinct rows of the batch that must be sent to each holder node (`0..=rows`).
    pub num_tokens_per_rank: BTreeMap<NodeId, u32>,
    /// Distinct rows of the batch that cross into each [`LinkSegments`] domain (`0..=rows`),
    /// DeepEP's `num_tokens_per_rdma_rank` for asymmetric-domain forwarding.
    pub num_tokens_per_segment: BTreeMap<String, u32>,
    /// Rows routed to each expert `0..num_experts`; feeds `m_grouped_gemm_masked`'s `masked_m`.
    pub num_tokens_per_expert: Vec<u32>,
    /// Per-holder row incidence mask of length `rows`: `is_token_in_rank[holder][row]` is true
    /// iff `holder` serves at least one selected expert for `row`.
    pub is_token_in_rank: BTreeMap<NodeId, Vec<bool>>,
    /// Total `(row, expert)` assignments served across all holders (`rows * top_k` when every
    /// selected expert has a holder).
    pub total_expert_assignments: u32,
}

impl DispatchLayout {
    /// Computes the DeepEP dispatch layout for `layer` from each row's selected expert ids,
    /// each holder's [`ExpertSliceParams`] (including EPLB's [`crate::ExpertRowShare`]), and the
    /// mesh's [`LinkSegments`].
    pub fn compute(
        layer: u32,
        num_experts: u32,
        routed_experts_per_row: &[Vec<u32>],
        slices: &[ExpertSliceParams],
        segments: &LinkSegments,
    ) -> Self {
        let rows = routed_experts_per_row.len() as u32;
        let mut num_tokens_per_expert = vec![0u32; num_experts as usize];
        for selections in routed_experts_per_row {
            for &expert_id in selections {
                if let Some(slot) = num_tokens_per_expert.get_mut(expert_id as usize) {
                    *slot += 1;
                }
            }
        }

        let mut is_token_in_rank: BTreeMap<NodeId, Vec<bool>> = BTreeMap::new();
        let mut num_tokens_per_rank: BTreeMap<NodeId, u32> = BTreeMap::new();
        let mut segment_row_hits: BTreeMap<String, Vec<bool>> = BTreeMap::new();
        let mut total_expert_assignments = 0u32;

        for slice in slices {
            let mut mask = vec![false; rows as usize];
            let seg = segments.segment_of(&slice.holder);
            let seg_mask = segment_row_hits
                .entry(seg)
                .or_insert_with(|| vec![false; rows as usize]);

            for (row_idx, selections) in routed_experts_per_row.iter().enumerate() {
                let row = row_idx as u32;
                let mut row_hit = false;
                for &expert_id in selections {
                    if slice.serves(expert_id, row) {
                        row_hit = true;
                        total_expert_assignments += 1;
                    }
                }
                if row_hit {
                    mask[row_idx] = true;
                    seg_mask[row_idx] = true;
                }
            }

            let count = mask.iter().filter(|&&b| b).count() as u32;
            num_tokens_per_rank.insert(slice.holder.clone(), count);
            is_token_in_rank.insert(slice.holder.clone(), mask);
        }

        let num_tokens_per_segment = segment_row_hits
            .into_iter()
            .map(|(seg, mask)| {
                let count = mask.iter().filter(|&&b| b).count() as u32;
                (seg, count)
            })
            .collect();

        Self {
            layer,
            rows,
            num_experts,
            num_tokens_per_rank,
            num_tokens_per_segment,
            num_tokens_per_expert,
            is_token_in_rank,
            total_expert_assignments,
        }
    }

    /// Returns the ordered list of row indices `0..rows` that must be sent to `holder`.
    pub fn rows_for_holder(&self, holder: &NodeId) -> Vec<u32> {
        self.is_token_in_rank
            .get(holder)
            .map(|mask| {
                mask.iter()
                    .enumerate()
                    .filter_map(|(r, &hit)| if hit { Some(r as u32) } else { None })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total `(holder, row)` activations sent across all holders after per-holder deduplication,
    /// compared against `naive_broadcast_rows = slices.len() * rows`.
    pub fn total_dispatched_holder_rows(&self) -> u32 {
        self.num_tokens_per_rank.values().sum()
    }

    /// Total `(segment, row)` activations sent when forwarding once per remote segment
    /// (`trunk_segment` excluded because it is intra-segment).
    pub fn remote_segment_forwarded_rows(&self, trunk_segment: &str) -> u32 {
        self.num_tokens_per_segment
            .iter()
            .filter(|(seg, _)| seg.as_str() != trunk_segment)
            .map(|(_, &c)| c)
            .sum()
    }
}

/// Wire payload bytes for one activation row of width `dim` in `f32` (`4 * dim`).
#[inline]
pub fn f32_row_wire_bytes(dim: usize) -> usize {
    dim * 4
}

/// Wire payload bytes for one activation row of width `dim` under DeepEP's fine-grained `1 x G`
/// FP8/INT8 block scaling with `UE8M0` 1-byte exponent scales (`dim + ceil_div(dim, G)`).
#[inline]
pub fn fp8_ue8m0_row_wire_bytes(dim: usize, block_size: usize) -> usize {
    let block_size = block_size.max(1);
    dim + dim.div_ceil(block_size)
}

/// Modelled accounting of how DeepEP's layout deduplication, asymmetric-domain segment forwarding,
/// low-latency FP8 quantization, and Zero-SM hook overlap affect an expert-parallel decode step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepEpStepAccount {
    pub rows: u32,
    pub hidden_dim: u32,
    pub moe_layers: u32,
    /// Naive broadcast dispatch bytes across the step (`holders * rows * 4 * dim` per MoE layer).
    pub broadcast_f32_dispatch_bytes: u64,
    /// Selective `f32` dispatch bytes (`is_token_in_rank` deduplication, exact `f32` rows).
    pub selective_f32_dispatch_bytes: u64,
    /// Low-latency `1 x 128` FP8 + `UE8M0` selective dispatch bytes.
    pub low_latency_fp8_dispatch_bytes: u64,
    /// Asymmetric-domain forwarded FP8 dispatch bytes (one inter-segment copy per distinct remote
    /// [`LinkSegments`] domain, then intra-segment fan-out).
    pub segment_forwarded_fp8_dispatch_bytes: u64,
    /// Standard `f32` combine return bytes (`PART` envelopes) across the step.
    pub f32_combine_bytes: u64,
    /// Low-latency FP8/INT8 combine return bytes (`DPCM` envelopes) across the step.
    pub low_latency_combine_bytes: u64,
    /// Overall wire-byte reduction factor (`(broadcast_dispatch + f32_combine) / (segment_fp8_dispatch + low_latency_combine)`).
    pub wire_compression_ratio: f64,
    /// Un-overlapped network hop time across all MoE layers of the step, in ms.
    pub raw_hop_ms: f64,
    /// Local compute available per step to overlap behind `recv_hook` (shared expert + trunk /
    /// dual-microbatch ping-pong), in ms.
    pub overlapped_local_compute_ms: f64,
    /// Exposed network hop time remaining after Zero-SM hook overlap, in ms.
    pub exposed_hop_ms: f64,
    /// Modelled expert-parallel step time (`ms`) when Zero-SM hook overlap is active.
    pub overlapped_step_ms: f64,
    /// Modelled expert-parallel ms/token when Zero-SM hook overlap is active.
    pub overlapped_ms_per_token: f64,
    /// Always true: these figures come from the analytical cost model, not a live hardware run.
    pub modelled: bool,
    pub rationale: String,
}

impl DeepEpStepAccount {
    /// Estimates DeepEP's wire and overlap savings for an expert-parallel placement on `nodes`.
    pub fn estimate(
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        popularity: Option<&ExpertPopularity>,
        ep_estimate: &StepCostEstimate,
        segments: &LinkSegments,
    ) -> Option<Self> {
        if !model.is_moe || model.experts_per_layer == 0 {
            return None;
        }
        let rows = ep_estimate.rows.max(1);
        let rows_f = rows as f64;
        // Derive nominal hidden dimension from `dense_layer_size_gb` (~24 MiB -> D=2048, else 4096).
        let hidden_dim: usize = if model.dense_layer_size_gb <= 0.05 {
            2048
        } else if model.dense_layer_size_gb <= 0.15 {
            4096
        } else {
            7168
        };
        let f32_row = f32_row_wire_bytes(hidden_dim) as f64;
        let fp8_row = fp8_ue8m0_row_wire_bytes(hidden_dim, DEEP_EP_DISPATCH_BLOCK_SIZE) as f64;

        let k = model.active_experts_per_token.max(1) as f64;
        let mut broadcast_dispatch = 0.0f64;
        let mut selective_f32_dispatch = 0.0f64;
        let mut low_latency_fp8_dispatch = 0.0f64;
        let mut segment_fp8_dispatch = 0.0f64;
        let mut f32_combine = 0.0f64;
        let mut low_latency_combine = 0.0f64;
        let mut moe_layers = 0u32;

        for range in &placement.trunk_ranges {
            let trunk_seg = segments.segment_of(&range.owner);
            for layer in range.start_layer..=range.end_layer {
                moe_layers += 1;
                let probs = popularity
                    .map(|p| p.probabilities(layer, model.experts_per_layer))
                    .unwrap_or_else(|| {
                        vec![
                            1.0 / model.experts_per_layer.max(1) as f64;
                            model.experts_per_layer as usize
                        ]
                    });

                // Per holder and per segment, compute the probability that a single row lands on
                // at least one expert served by that holder / segment:
                // `P(row hits holder H) = 1 - prod_{e in H} (1 - min(1, k * p_e * share))`.
                let mut holder_miss_prob: BTreeMap<NodeId, f64> = BTreeMap::new();
                let mut segment_miss_prob: BTreeMap<String, f64> = BTreeMap::new();
                let mut holders_in_layer: BTreeSet<NodeId> = BTreeSet::new();
                let mut remote_partial_entries = 0.0f64;

                for home in placement.experts.iter().filter(|h| h.layer == layer) {
                    let p_e = probs.get(home.expert as usize).copied().unwrap_or(0.0);
                    let copies = home.holders.len().max(1);
                    for (idx, holder) in home.holders.iter().enumerate() {
                        if holder == &range.owner {
                            continue;
                        }
                        holders_in_layer.insert(holder.clone());
                        let share = ExpertPlacementPlan::row_share(rows, copies, idx);
                        let p_hit = (k * p_e * share).clamp(0.0, 1.0);
                        remote_partial_entries += rows_f * p_hit;

                        let entry = holder_miss_prob.entry(holder.clone()).or_insert(1.0);
                        *entry *= (1.0 - p_hit).clamp(0.0, 1.0);

                        let seg = segments.segment_of(holder);
                        if seg != trunk_seg {
                            let seg_entry = segment_miss_prob.entry(seg).or_insert(1.0);
                            *seg_entry *= (1.0 - p_hit).clamp(0.0, 1.0);
                        }
                    }
                }

                let remote_holders = holders_in_layer.len() as f64;
                if remote_holders == 0.0 {
                    continue;
                }

                // 1. Naive broadcast sends all `rows` in `f32` to every remote holder in the layer.
                broadcast_dispatch += remote_holders * rows_f * f32_row;

                // 2. Selective `is_token_in_rank` sends row `r` to holder `H` only when `H` serves
                //    at least one expert for row `r`.
                let expected_holder_rows: f64 = holder_miss_prob
                    .values()
                    .map(|&miss| rows_f * (1.0 - miss))
                    .sum();
                selective_f32_dispatch += expected_holder_rows * f32_row;
                low_latency_fp8_dispatch += expected_holder_rows * fp8_row;

                // 3. Asymmetric-domain forwarding sends row `r` once per remote segment, plus
                //    same-segment holders directly.
                let same_seg_holder_rows: f64 = holder_miss_prob
                    .iter()
                    .filter(|(h, _)| segments.segment_of(h) == trunk_seg)
                    .map(|(_, &miss)| rows_f * (1.0 - miss))
                    .sum();
                let remote_seg_rows: f64 = segment_miss_prob
                    .values()
                    .map(|&miss| rows_f * (1.0 - miss))
                    .sum();
                segment_fp8_dispatch += (same_seg_holder_rows + remote_seg_rows) * fp8_row;

                // 4. Combine return path: one partial per served `(row, expert)` on remote holders,
                //    or pre-aggregated per `(holder, row)` in low-latency FP8 combine mode.
                f32_combine += remote_partial_entries * (8.0 + f32_row);
                low_latency_combine += expected_holder_rows * (8.0 + fp8_row);
            }
        }

        let _ = nodes;
        let baseline_total = (broadcast_dispatch + f32_combine).max(1.0);
        let optimized_total = (segment_fp8_dispatch + low_latency_combine).max(1.0);
        let compression = ((baseline_total / optimized_total) * 100.0).round() / 100.0;

        // Zero-SM hook overlap: while remote expert slices and network hops (`raw_hop_ms`) are in
        // flight, the trunk owner overlaps shared-expert / dense-trunk work (`weight_read_ms - expert_read_ms`).
        // With wider batches (`rows >= 2`), two-microbatch ping-pong also overlaps up to half of
        // the remaining expert compute.
        let non_routed_local_ms =
            (ep_estimate.weight_read_ms - ep_estimate.expert_read_ms).max(0.0);
        let dual_microbatch_overlap_ms = if rows >= 2 {
            ep_estimate.expert_read_ms * 0.35
        } else {
            0.0
        };
        let overlap_budget_ms = non_routed_local_ms + dual_microbatch_overlap_ms;
        let raw_hop_ms = ep_estimate.hop_ms;
        let overlapped_local_compute_ms = raw_hop_ms.min(overlap_budget_ms);
        let exposed_hop_ms = (raw_hop_ms - overlapped_local_compute_ms).max(0.0);
        let overlapped_step_ms = (ep_estimate.step_ms - overlapped_local_compute_ms).max(0.05);
        let overlapped_ms_per_token = overlapped_step_ms / rows_f;

        let rationale = format!(
            "DeepEP layout deduplication + 1x{DEEP_EP_DISPATCH_BLOCK_SIZE} FP8/UE8M0 dispatch & combine compresses fan-out wire traffic by {compression:.2}x ({:.1} KB -> {:.1} KB/step); Zero-SM recv_hook hides {:.2} ms of the {:.2} ms hop latency behind local trunk/shared-expert compute ({:.2} ms/token modelled)",
            baseline_total / 1024.0,
            optimized_total / 1024.0,
            overlapped_local_compute_ms,
            raw_hop_ms,
            overlapped_ms_per_token,
        );

        Some(Self {
            rows,
            hidden_dim: hidden_dim as u32,
            moe_layers,
            broadcast_f32_dispatch_bytes: broadcast_dispatch.round() as u64,
            selective_f32_dispatch_bytes: selective_f32_dispatch.round() as u64,
            low_latency_fp8_dispatch_bytes: low_latency_fp8_dispatch.round() as u64,
            segment_forwarded_fp8_dispatch_bytes: segment_fp8_dispatch.round() as u64,
            f32_combine_bytes: f32_combine.round() as u64,
            low_latency_combine_bytes: low_latency_combine.round() as u64,
            wire_compression_ratio: compression,
            raw_hop_ms: (raw_hop_ms * 100.0).round() / 100.0,
            overlapped_local_compute_ms: (overlapped_local_compute_ms * 100.0).round() / 100.0,
            exposed_hop_ms: (exposed_hop_ms * 100.0).round() / 100.0,
            overlapped_step_ms: (overlapped_step_ms * 100.0).round() / 100.0,
            overlapped_ms_per_token: (overlapped_ms_per_token * 100.0).round() / 100.0,
            modelled: true,
            rationale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExpertRowShare;

    #[test]
    fn get_dispatch_layout_deduplicates_rows_per_rank_and_segment_and_respects_eplb_shares() {
        // 4 rows, 8 experts, 3 holders across 2 link segments:
        // - `pc` on `lan:192.168.40` holds experts [0, 1] (and shares expert 0 with `mac` 2-way)
        // - `mac` on `lan:192.168.40` holds experts [0, 2, 3]
        // - `cloud` on `remote:cloud` holds experts [4, 5, 6, 7]
        let slices = vec![
            ExpertSliceParams {
                model_id: "m".into(),
                epoch: 1,
                session_id: "s".into(),
                layer: 0,
                rows: 4,
                holder: NodeId("pc".into()),
                resident_experts: vec![0, 1],
                row_shares: vec![ExpertRowShare {
                    expert: 0,
                    index: 0,
                    count: 2,
                }],
            },
            ExpertSliceParams {
                model_id: "m".into(),
                epoch: 1,
                session_id: "s".into(),
                layer: 0,
                rows: 4,
                holder: NodeId("mac".into()),
                resident_experts: vec![0, 2, 3],
                row_shares: vec![ExpertRowShare {
                    expert: 0,
                    index: 1,
                    count: 2,
                }],
            },
            ExpertSliceParams {
                model_id: "m".into(),
                epoch: 1,
                session_id: "s".into(),
                layer: 0,
                rows: 4,
                holder: NodeId("cloud".into()),
                resident_experts: vec![4, 5, 6, 7],
                row_shares: vec![],
            },
        ];

        let mut seg_map = BTreeMap::new();
        seg_map.insert(NodeId("pc".into()), "lan:192.168.40".to_string());
        seg_map.insert(NodeId("mac".into()), "lan:192.168.40".to_string());
        seg_map.insert(NodeId("cloud".into()), "remote:cloud".to_string());
        let segments = LinkSegments::from_map(seg_map);

        // Row 0 -> [0, 1] (both served by `pc` since row 0 % 2 == 0; `mac` and `cloud` get nothing!)
        // Row 1 -> [0, 2] (expert 0 at row 1 % 2 == 1 goes to `mac`, expert 2 also on `mac`; `pc` gets nothing!)
        // Row 2 -> [4, 5, 6] (all three on `cloud` -> deduplicated to 1 row sent to `cloud`!)
        // Row 3 -> [1, 3, 7] (1 on `pc`, 3 on `mac`, 7 on `cloud` -> all 3 holders get row 3)
        let routed = vec![vec![0, 1], vec![0, 2], vec![4, 5, 6], vec![1, 3, 7]];

        let layout = DispatchLayout::compute(0, 8, &routed, &slices, &segments);

        // `pc` only needs rows 0 and 3 (2 tokens, even though row 0 selected two experts on `pc`).
        assert_eq!(layout.num_tokens_per_rank[&NodeId("pc".into())], 2);
        assert_eq!(layout.rows_for_holder(&NodeId("pc".into())), vec![0, 3]);

        // `mac` only needs rows 1 and 3 (row 0's expert 0 was served by `pc`, and row 1's two
        // experts [0, 2] both sit on `mac` and deduplicate to 1 token).
        assert_eq!(layout.num_tokens_per_rank[&NodeId("mac".into())], 2);
        assert_eq!(layout.rows_for_holder(&NodeId("mac".into())), vec![1, 3]);

        // `cloud` only needs rows 2 and 3 (row 2 hit three experts [4, 5, 6] on `cloud` -> 1 token).
        assert_eq!(layout.num_tokens_per_rank[&NodeId("cloud".into())], 2);
        assert_eq!(layout.rows_for_holder(&NodeId("cloud".into())), vec![2, 3]);

        // Total holder-rows dispatched is 2 + 2 + 2 = 6, versus 3 * 4 = 12 for naive broadcast!
        assert_eq!(layout.total_dispatched_holder_rows(), 6);

        // Segment coalescing: `lan:192.168.40` needs rows 0, 1, 3 (3 distinct rows across `pc` + `mac`),
        // and `remote:cloud` needs rows 2, 3 (2 distinct rows).
        assert_eq!(layout.num_tokens_per_segment["lan:192.168.40"], 3);
        assert_eq!(layout.num_tokens_per_segment["remote:cloud"], 2);
        assert_eq!(layout.remote_segment_forwarded_rows("lan:192.168.40"), 2);

        // Per-expert counts (`masked_m` for DeepGEMM):
        assert_eq!(layout.num_tokens_per_expert, vec![2, 2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(layout.total_expert_assignments, 10);
    }
}
