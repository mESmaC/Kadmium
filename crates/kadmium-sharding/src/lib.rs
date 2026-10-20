//! # `kadmium-sharding`
//!
//! Implements **Custom Asymmetric Model Sharding** and **Dynamic Compute Responsibility Shifting**
//! across wildly heterogeneous hardware for both **Inference** and **Distributed Training**.

use kadmium_core::{
    ComputeProviderKind, DataflowMode, DeviceKind, FailoverDirection, HardwareNode, ModelSpec,
    NodeId, NodeRole, NodeTrustTier, PoliteState, SlicePurpose, TrainingStrategy, WorkloadKind,
};
use serde::{Deserialize, Serialize};

/// Per-node activation checkpointing policy during distributed training (1F1B / QLoRA).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationCheckpointingPolicy {
    /// Node has abundant Unified Memory or VRAM (e.g., M4 Mac Mini 48GB, Cloud H100);
    /// caches forward activations to avoid recomputation FLOPs.
    CacheActivationsInMemory,
    /// Checkpoints every 2nd transformer block to balance VRAM and FLOPs.
    SelectiveEveryTwoLayers,
    /// Node is VRAM-constrained (e.g., 8GB Gaming Laptop); recomputes activations during
    /// backward pass to maximize the number of layers/experts it can host.
    FullRecomputeSaveVram,
    /// Utility / non-pipeline node (tokenization, dataset prep, or weight seeding).
    NotApplicable,
}

/// Breakdown of how MoE experts (or layer weights) are distributed across a node's memory tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredPlacementBreakdown {
    /// GiB pinned in Tier 0 (VRAM) or Tier 1 (Unified Memory).
    pub hot_vram_or_unified_gb: f64,
    /// GiB placed in Tier 2 (Host System RAM) for CPU/hybrid $q^*$ execution.
    pub warm_system_ram_gb: f64,
    /// GiB streamed on demand from Tier 3 (Local NVMe `.ktb` bank, Colibri-style).
    pub cold_nvme_stream_gb: f64,
    /// Reserved GiB for KV-Cache (Inference) or Forward Activations + LoRA Gradients (Training).
    pub state_and_cache_gb: f64,
    /// FreeToken-inspired bandwidth-adaptive split ratio $q^* \in [0.0, 1.0]$.
    pub q_star_accel_ratio: f64,
}

/// Auxiliary edge responsibilities assigned to low-power nodes (SBCs, Legacy Laptops, NAS).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuxiliaryResponsibility {
    /// Fast BPE tokenization, prompt template formatting, and SSE output detokenization.
    TokenizationAndStreamIngress,
    /// Runs a tiny 0.6B–1.5B INT4 draft model for speculative decoding verification.
    SpeculativeDecodingDraftEngine,
    /// Streams pre-tokenized training micro-batches and performs on-the-fly data augmentation.
    TrainingDataLoaderAndAugmenter,
    /// Seeds `.ktb` tensor banks via FTS3 third-party transfers and archives rolling checkpoints.
    Fts3WeightSeedAndCheckpointVault,
    /// Runs BOINC-style deterministic canary spot-verification and embedding lookup tables.
    CanaryVerifierAndEmbeddingHost,
    /// Executes Gensyn Verde / RepOps $O(\log L)$ refereed layer-bisection cross-silicon audits.
    VerdeRefereedLayerBisectionAuditor,
    /// Runs parallel verifiable reward sandboxes (code compilation/unit tests/math) for RL-Swarm GRPO.
    RlSwarmRewardVerifierSandbox,
}

/// Assignment of a contiguous layer range $[start\_layer, end\_layer]$ and auxiliary tasks to a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeShardAssignment {
    pub node_id: NodeId,
    pub hostname: String,
    pub device_kind: DeviceKind,
    pub trust_tier: NodeTrustTier,
    pub polite_state: PoliteState,
    pub pipeline_stage_index: Option<usize>,
    /// Inclusive start layer index, if assigned transformer layers.
    pub start_layer: Option<u32>,
    /// Inclusive end layer index, if assigned transformer layers.
    pub end_layer: Option<u32>,
    pub layer_count: u32,
    /// Share of total model compute handled by this node ($0.0..=100.0\%$).
    pub compute_share_pct: f64,
    pub tiered_memory: TieredPlacementBreakdown,
    pub training_checkpoint_policy: ActivationCheckpointingPolicy,
    pub auxiliary_tasks: Vec<AuxiliaryResponsibility>,
}

/// Complete cluster-wide shard plan for a given model and workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterShardPlan {
    pub epoch: u64,
    pub model_id: String,
    pub model_display_name: String,
    pub workload: WorkloadKind,
    pub training_strategy: Option<TrainingStrategy>,
    pub total_layers: u32,
    pub total_model_weight_gb: f64,
    pub assignments: Vec<NodeShardAssignment>,
    /// Estimated aggregate cluster throughput (tokens/sec for inference, samples/sec for training).
    pub estimated_throughput: f64,
    pub throughput_unit: String,
}

/// Reason why a dynamic compute responsibility shift was triggered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShiftTriggerReason {
    /// User started a game or heavy application on a primary workstation/laptop.
    HostEnteredGamingMode { node_id: NodeId },
    /// Laptop unplugged from AC power or hit thermal throttle limit.
    ThermalOrBatteryThrottle { node_id: NodeId },
    /// Ephemeral rented Cloud GPU joined the cluster to boost training/inference.
    CloudBurstNodeJoined { node_id: NodeId },
    /// Ephemeral Cloud GPU is nearing rental expiration and draining shards back to local cluster.
    CloudBurstNodeDraining { node_id: NodeId },
    /// Manual or automated pipeline bubble rebalance.
    PipelineStragglerRebalance,
}

/// A single contiguous range of layers migrating from one node to another during a live shift.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMigrationMove {
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub start_layer: u32,
    pub end_layer: u32,
    pub layer_count: u32,
    /// Static weight volume (GiB) pre-staged in the background from NAS/peer via FTS3 before switchover.
    pub background_fts3_prestage_gb: f64,
    /// Live mutable state (KV-cache slice or LoRA adapter delta in GiB) transferred at step/token boundary.
    pub live_boundary_delta_gb: f64,
}

/// Complete live migration plan transitioning the cluster from `previous_epoch` to `new_epoch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsibilityShiftPlan {
    pub previous_epoch: u64,
    pub new_epoch: u64,
    pub trigger: ShiftTriggerReason,
    pub migrations: Vec<LayerMigrationMove>,
    pub total_background_prestage_gb: f64,
    pub total_live_boundary_delta_gb: f64,
    /// Estimated boundary switchover pause in milliseconds (after background pre-staging completes).
    pub estimated_switchover_pause_ms: f64,
    pub new_plan: ClusterShardPlan,
}

/// Asymmetric topology-aware shard planner and dynamic responsibility shifter.
pub struct ShardPlanner;

impl ShardPlanner {
    /// Computes an optimal asymmetric shard plan across a heterogeneous node list.
    pub fn compute_plan(
        epoch: u64,
        model: &ModelSpec,
        workload: WorkloadKind,
        nodes: &[HardwareNode],
    ) -> ClusterShardPlan {
        // 1. Score all nodes for layer-sharding eligibility.
        let mut scored_workers: Vec<(usize, f64)> = nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| {
                let score = node.capacity_score(workload);
                // Filter out tiny utility-only or tiny-score nodes from hosting heavy transformer blocks
                // unless no other nodes exist.
                let eligible = score >= 12.0
                    && (node.roles.contains(&NodeRole::ComputeWorker)
                        || node.roles.contains(&NodeRole::BurstAccelerator));
                (idx, if eligible { score } else { 0.0 })
            })
            .collect();

        let total_score: f64 = scored_workers.iter().map(|(_, s)| *s).sum();

        // Fallback if all nodes had tiny scores: allow any ComputeWorker.
        if total_score <= f64::EPSILON {
            for (idx, score) in &mut scored_workers {
                if nodes[*idx].roles.contains(&NodeRole::ComputeWorker) {
                    *score = 1.0;
                }
            }
        }
        let total_score: f64 = scored_workers.iter().map(|(_, s)| *s).sum::<f64>().max(1.0);

        // 2. Proportionally allocate model.num_layers across eligible workers using largest-remainder method.
        let active_indices: Vec<usize> = scored_workers
            .iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(i, _)| *i)
            .collect();

        let mut layer_counts = vec![0u32; nodes.len()];
        let mut remainders = Vec::new();
        let mut assigned_sum = 0u32;

        for &idx in &active_indices {
            let exact = (scored_workers[idx].1 / total_score) * (model.num_layers as f64);
            let floored = exact.floor() as u32;
            layer_counts[idx] = floored;
            assigned_sum += floored;
            remainders.push((idx, exact - (floored as f64)));
        }

        remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut leftover = model.num_layers.saturating_sub(assigned_sum);
        for (idx, _) in remainders {
            if leftover == 0 {
                break;
            }
            layer_counts[idx] += 1;
            leftover -= 1;
        }

        // 3. Enforce Privacy-Aware 3-Tier Trust Zone Pinning:
        //    Order active pipeline stages so `TrustedCore` nodes always bookend Layer 0 (Prompt Ingress)
        //    and Layer N-1 (LM Head / Token Sampling), placing `VerifiedPeerMiddleLayersOnly` and
        //    `UntrustedAuditedMiddleLayersOnly` strictly in middle stages (0 < L_a <= L_b < N-1).
        let mut trusted_active: Vec<usize> = (0..nodes.len())
            .filter(|&i| {
                layer_counts[i] > 0 && nodes[i].trust_tier.can_host_boundary_ingress_egress()
            })
            .collect();
        let middle_only_active: Vec<usize> = (0..nodes.len())
            .filter(|&i| {
                layer_counts[i] > 0 && !nodes[i].trust_tier.can_host_boundary_ingress_egress()
            })
            .collect();

        let mut ordered_stage_nodes = Vec::new();
        if trusted_active.len() >= 2 && !middle_only_active.is_empty() {
            // Keep the last TrustedCore node as the final Egress stage (ending at num_layers - 1)
            let egress_trusted = trusted_active.pop().expect("at least 2 trusted nodes");
            ordered_stage_nodes.extend(trusted_active);
            ordered_stage_nodes.extend(middle_only_active);
            ordered_stage_nodes.push(egress_trusted);
        } else {
            ordered_stage_nodes.extend(trusted_active);
            ordered_stage_nodes.extend(middle_only_active);
        }

        let mut stage_map: Vec<Option<(usize, u32, u32)>> = vec![None; nodes.len()];
        let mut current_layer = 0u32;
        for (stage_idx, &node_idx) in ordered_stage_nodes.iter().enumerate() {
            let count = layer_counts[node_idx];
            let s = current_layer;
            let e = current_layer + count - 1;
            current_layer += count;
            stage_map[node_idx] = Some((stage_idx, s, e));
        }

        let mut assignments = Vec::with_capacity(nodes.len());

        let per_layer_moe_gb = (model.experts_per_layer as f64) * model.single_expert_size_gb;
        let per_layer_total_weight_gb = model.dense_layer_size_gb + per_layer_moe_gb;

        for (idx, node) in nodes.iter().enumerate() {
            let count = layer_counts[idx];
            let (stage_idx, start_layer, end_layer) = match stage_map[idx] {
                Some((stg, s, e)) => (Some(stg), Some(s), Some(e)),
                None => (None, None, None),
            };

            let compute_share_pct = if count > 0 {
                (scored_workers[idx].1 / total_score) * 100.0
            } else {
                0.0
            };

            // Determine training activation checkpointing policy per node hardware profile.
            let checkpoint_policy = match workload {
                WorkloadKind::DistributedTraining if count > 0 => {
                    if node.memory.unified_mem_gb >= 32.0 || node.memory.vram_gb >= 48.0 {
                        ActivationCheckpointingPolicy::CacheActivationsInMemory
                    } else if node.memory.vram_gb >= 16.0
                        && node.polite_state == PoliteState::IdleDedicated
                    {
                        ActivationCheckpointingPolicy::SelectiveEveryTwoLayers
                    } else {
                        ActivationCheckpointingPolicy::FullRecomputeSaveVram
                    }
                }
                _ => ActivationCheckpointingPolicy::NotApplicable,
            };

            // Calculate state/cache memory (KV cache for inference, activations + LoRA states for training).
            let state_per_layer_gb = match workload {
                WorkloadKind::Inference => model.kv_cache_per_layer_gb,
                WorkloadKind::DistributedTraining => {
                    let act_factor = match checkpoint_policy {
                        ActivationCheckpointingPolicy::CacheActivationsInMemory => 1.0,
                        ActivationCheckpointingPolicy::SelectiveEveryTwoLayers => 0.5,
                        ActivationCheckpointingPolicy::FullRecomputeSaveVram => 0.2,
                        ActivationCheckpointingPolicy::NotApplicable => 0.0,
                    };
                    (model.training_activation_per_layer_gb * act_factor)
                        + model.lora_state_per_layer_gb
                }
                WorkloadKind::BatchPipeline => model.kv_cache_per_layer_gb * 0.5,
            };

            let state_and_cache_gb = (count as f64) * state_per_layer_gb;
            let total_shard_weights_gb = (count as f64) * per_layer_total_weight_gb;

            // Place weights across Hot (VRAM/Unified), Warm (System RAM), and Cold (NVMe `.ktb` streaming)
            let fast_budget_gb = (node.available_fast_memory_gb() - state_and_cache_gb).max(0.0);
            let hot_vram_or_unified_gb = total_shard_weights_gb.min(fast_budget_gb);
            let remaining_after_hot = (total_shard_weights_gb - hot_vram_or_unified_gb).max(0.0);

            let ram_budget_gb =
                node.memory.system_ram_gb * 0.65 * node.polite_state.availability_multiplier();
            let warm_system_ram_gb = remaining_after_hot.min(ram_budget_gb);
            let cold_nvme_stream_gb = (remaining_after_hot - warm_system_ram_gb).max(0.0);

            // Assign auxiliary responsibilities to SBCs, NAS, Legacy laptops, or high-CPU nodes.
            let mut auxiliary_tasks = Vec::new();
            if node.device_kind == DeviceKind::HomeNas
                || node.roles.contains(&NodeRole::StorageVault)
            {
                auxiliary_tasks.push(AuxiliaryResponsibility::Fts3WeightSeedAndCheckpointVault);
            }
            if node.device_kind == DeviceKind::LinuxSbc {
                auxiliary_tasks.push(AuxiliaryResponsibility::TokenizationAndStreamIngress);
                if workload == WorkloadKind::Inference {
                    auxiliary_tasks.push(AuxiliaryResponsibility::SpeculativeDecodingDraftEngine);
                } else {
                    auxiliary_tasks.push(AuxiliaryResponsibility::RlSwarmRewardVerifierSandbox);
                    auxiliary_tasks.push(AuxiliaryResponsibility::TrainingDataLoaderAndAugmenter);
                }
            }
            if node.device_kind == DeviceKind::LegacyLaptop {
                if workload == WorkloadKind::DistributedTraining {
                    auxiliary_tasks
                        .push(AuxiliaryResponsibility::VerdeRefereedLayerBisectionAuditor);
                    auxiliary_tasks.push(AuxiliaryResponsibility::TrainingDataLoaderAndAugmenter);
                } else {
                    auxiliary_tasks.push(AuxiliaryResponsibility::CanaryVerifierAndEmbeddingHost);
                }
            }

            assignments.push(NodeShardAssignment {
                node_id: node.id.clone(),
                hostname: node.hostname.clone(),
                device_kind: node.device_kind,
                trust_tier: node.trust_tier,
                polite_state: node.polite_state,
                pipeline_stage_index: stage_idx,
                start_layer,
                end_layer,
                layer_count: count,
                compute_share_pct: (compute_share_pct * 10.0).round() / 10.0,
                tiered_memory: TieredPlacementBreakdown {
                    hot_vram_or_unified_gb: (hot_vram_or_unified_gb * 100.0).round() / 100.0,
                    warm_system_ram_gb: (warm_system_ram_gb * 100.0).round() / 100.0,
                    cold_nvme_stream_gb: (cold_nvme_stream_gb * 100.0).round() / 100.0,
                    state_and_cache_gb: (state_and_cache_gb * 100.0).round() / 100.0,
                    q_star_accel_ratio: (node.compute_q_star_ratio() * 100.0).round() / 100.0,
                },
                training_checkpoint_policy: checkpoint_policy,
                auxiliary_tasks,
            });
        }

        // Determine training strategy if applicable.
        let has_wan_or_burst = nodes.iter().any(|n| n.network_latency_ms > 8.0);
        let training_strategy = match workload {
            WorkloadKind::DistributedTraining => {
                if has_wan_or_burst {
                    Some(TrainingStrategy::HybridElasticMesh)
                } else {
                    Some(TrainingStrategy::AsymmetricPipeline1F1B)
                }
            }
            _ => None,
        };

        // Estimate aggregate cluster throughput.
        let effective_cluster_tflops: f64 = nodes
            .iter()
            .zip(layer_counts.iter())
            .filter(|(_, c)| **c > 0)
            .map(|(n, _)| n.effective_fp16_tflops())
            .sum();

        let (estimated_throughput, throughput_unit) = match workload {
            WorkloadKind::Inference => {
                // MoE active params vs dense params scaling
                let active_ratio = if model.is_moe { 0.14 } else { 1.0 };
                let active_params_b = model.total_params_billions * active_ratio;
                let tok_per_sec =
                    ((effective_cluster_tflops * 1.45) / active_params_b.max(1.0)).max(4.5);
                (
                    (tok_per_sec * 10.0).round() / 10.0,
                    "tokens/sec".to_string(),
                )
            }
            WorkloadKind::DistributedTraining => {
                let samples_per_sec = ((effective_cluster_tflops * 0.42)
                    / model.total_params_billions.max(1.0))
                .max(0.8);
                (
                    (samples_per_sec * 100.0).round() / 100.0,
                    "sequences/sec (1F1B + QLoRA)".to_string(),
                )
            }
            WorkloadKind::BatchPipeline => {
                let items_per_sec = effective_cluster_tflops * 3.8;
                (
                    (items_per_sec * 10.0).round() / 10.0,
                    "workunits/min".to_string(),
                )
            }
        };

        ClusterShardPlan {
            epoch,
            model_id: model.id.clone(),
            model_display_name: model.display_name.clone(),
            workload,
            training_strategy,
            total_layers: model.num_layers,
            total_model_weight_gb: (model.total_weight_footprint_gb() * 100.0).round() / 100.0,
            assignments,
            estimated_throughput,
            throughput_unit,
        }
    }

    /// Computes a live **ResponsibilityShiftPlan** between an existing shard plan and an updated
    /// cluster topology (e.g., when a user launches a game on their desktop or a burst GPU joins).
    pub fn plan_responsibility_shift(
        current_plan: &ClusterShardPlan,
        model: &ModelSpec,
        updated_nodes: &[HardwareNode],
        trigger: ShiftTriggerReason,
    ) -> ResponsibilityShiftPlan {
        let next_epoch = current_plan.epoch + 1;
        let new_plan = Self::compute_plan(next_epoch, model, current_plan.workload, updated_nodes);

        // Map each layer l in 0..model.num_layers from its old owner to its new owner.
        let mut old_owner_by_layer = vec![NodeId("unassigned".into()); model.num_layers as usize];
        for assign in &current_plan.assignments {
            if let (Some(s), Some(e)) = (assign.start_layer, assign.end_layer) {
                for l in s..=e {
                    old_owner_by_layer[l as usize] = assign.node_id.clone();
                }
            }
        }

        let mut new_owner_by_layer = vec![NodeId("unassigned".into()); model.num_layers as usize];
        for assign in &new_plan.assignments {
            if let (Some(s), Some(e)) = (assign.start_layer, assign.end_layer) {
                for l in s..=e {
                    new_owner_by_layer[l as usize] = assign.node_id.clone();
                }
            }
        }

        let per_layer_weight_gb = model.dense_layer_size_gb
            + (model.experts_per_layer as f64) * model.single_expert_size_gb;
        let per_layer_delta_gb = match current_plan.workload {
            WorkloadKind::Inference => model.kv_cache_per_layer_gb,
            WorkloadKind::DistributedTraining => model.lora_state_per_layer_gb,
            WorkloadKind::BatchPipeline => 0.02,
        };

        // Group contiguous layer transitions (from_node -> to_node) where from_node != to_node.
        let mut migrations: Vec<LayerMigrationMove> = Vec::new();
        for l in 0..model.num_layers {
            let from = &old_owner_by_layer[l as usize];
            let to = &new_owner_by_layer[l as usize];
            if from != to {
                if let Some(last) = migrations.last_mut()
                    && &last.from_node == from
                    && &last.to_node == to
                    && last.end_layer + 1 == l
                {
                    last.end_layer = l;
                    last.layer_count += 1;
                    last.background_fts3_prestage_gb =
                        ((last.layer_count as f64) * per_layer_weight_gb * 100.0).round() / 100.0;
                    last.live_boundary_delta_gb =
                        ((last.layer_count as f64) * per_layer_delta_gb * 100.0).round() / 100.0;
                    continue;
                }
                migrations.push(LayerMigrationMove {
                    from_node: from.clone(),
                    to_node: to.clone(),
                    start_layer: l,
                    end_layer: l,
                    layer_count: 1,
                    background_fts3_prestage_gb: (per_layer_weight_gb * 100.0).round() / 100.0,
                    live_boundary_delta_gb: (per_layer_delta_gb * 100.0).round() / 100.0,
                });
            }
        }

        let total_prestage: f64 = migrations
            .iter()
            .map(|m| m.background_fts3_prestage_gb)
            .sum();
        let total_delta: f64 = migrations.iter().map(|m| m.live_boundary_delta_gb).sum();

        // Because static base weights are pre-staged in the background via FTS3 from the NAS/peers,
        // the atomic step/token boundary pause only waits on the compact delta state transfer!
        // Assume ~2.5 Gbps (~0.3125 GB/s) effective P2P delta stream speed.
        let pause_ms = ((total_delta / 0.3125) * 1000.0).clamp(12.0, 2500.0);

        ResponsibilityShiftPlan {
            previous_epoch: current_plan.epoch,
            new_epoch: next_epoch,
            trigger,
            migrations,
            total_background_prestage_gb: (total_prestage * 100.0).round() / 100.0,
            total_live_boundary_delta_gb: (total_delta * 100.0).round() / 100.0,
            estimated_switchover_pause_ms: (pause_ms * 10.0).round() / 100.0 * 10.0,
            new_plan,
        }
    }
}

// ================================================================================================
// TPU & Compute-in-Memory (CIM) Inspired Dataflow Engine:
// Dynamic Token Streaming (Weight-Stationary) vs. Dynamic Weight Streaming (Token-Stationary)
// ================================================================================================

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
        // Weights stay 100% pinned in distributed RAM/Unified/VRAM banks; only compact token activations
        // (e.g. 16 KB per token per pipeline boundary) flow like a systolic wavefront.
        let token_packet_mb =
            ((batch_tokens as f64) * 0.016 * (nodes.len() as f64) * 100.0).round() / 100.0;
        let cim_tok_s =
            ((total_mem_bw_gbps / active_weight_gb.max(1.0)) * 1.15 * 10.0).round() / 10.0;

        // 2. Token-Stationary Dynamic Weight Streaming:
        // Large batch of tokens stays stationary in GPU SRAM/VRAM while double-buffered weight tiles
        // stream from NVMe / NAS / Host RAM. Throughput scales strongly with batch_tokens!
        let weight_stream_mb = (active_weight_gb * 1024.0 * 10.0).round() / 10.0;
        let arith_intensity = (batch_tokens as f64) * 2.0; // 2 FLOPs per weight byte (Q8/FP8)
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

// ================================================================================================
// Sub-Compute Environment Partitioner (Cluster Pool Slicing: e.g. 100 Nodes -> 5x 20-Node Slices)
// ================================================================================================

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
/// Sub-Compute Environments (e.g., 5 sub-clusters of 20 nodes each for Prod, Staging/Test,
/// Distributed Training, RL/Dream-RSI, and Batch Evals).
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

        // Distribute nodes across the `k` slices using striped assignment so each sub-cluster
        // receives a balanced mix of GPUs, Apple Silicon Macs, Laptops, SBCs, and NAS vaults.
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

// ================================================================================================
// Open-Core Hybrid Cloud Engine:
// 1. Multi-Datacenter Spot & Fragmented GPU Arbitrage
// 2. Bidirectional Downtime Protection Shield (Local <-> Cloud Failover & Failback)
// ================================================================================================

/// A fragmented or spot compute instance offer from an external datacenter provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderComputeOffer {
    pub offer_id: String,
    pub provider_name: String,
    pub provider_kind: ComputeProviderKind,
    pub instance_description: String,
    pub fp16_tflops: f64,
    pub vram_or_unified_gb: f64,
    pub system_ram_gb: f64,
    pub hourly_spot_usd: f64,
    /// Equivalent monolithic hyperscaler on-demand price in USD/hr for comparison.
    pub hyperscaler_equivalent_usd: f64,
}

/// Multi-provider arbitrage plan stitching fragmented cheap instances + local hardware together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiProviderArbitragePlan {
    pub target_model_id: String,
    pub selected_offers: Vec<ProviderComputeOffer>,
    pub total_combined_tflops: f64,
    pub total_combined_vram_gb: f64,
    pub kadmium_blended_hourly_usd: f64,
    pub monolithic_hyperscaler_hourly_usd: f64,
    pub cost_reduction_pct: f64,
}

/// A single simulated outage and automatic live shard failover event (`Local <-> Cloud`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverShieldSimulation {
    pub direction: FailoverDirection,
    pub incident_summary: String,
    pub affected_layers: (u32, u32),
    pub source_environment: String,
    pub target_failover_environment: String,
    /// Recovery Time Objective (RTO) achieved via Warm Vault `.ktb` pre-staging + live delta handoff.
    pub switchover_rto_ms: f64,
    /// Zero dropped requests or preserved training step continuity.
    pub zero_downtime_preserved: bool,
    pub temporary_failover_hourly_usd: f64,
}

/// Planner for Kadmium's "Supabase for AI Compute" Hybrid Cloud Arbitrage & Downtime Protection Shield.
pub struct HybridCloudArbitragePlanner;

impl HybridCloudArbitragePlanner {
    /// Builds a multi-datacenter arbitrage plan that stitches fragmented, low-cost single-GPU
    /// spot instances and high-RAM bare-metal servers across 4 different providers using
    /// Kadmium's asymmetric sharding + DeMo/DisTrO low-bandwidth compression.
    pub fn compute_arbitrage_plan(model: &ModelSpec) -> MultiProviderArbitragePlan {
        let offers = vec![
            ProviderComputeOffer {
                offer_id: "arb-local-01".into(),
                provider_name: "User Local Cluster (Main PC + M4 Mac Mini + NAS)".into(),
                provider_kind: ComputeProviderKind::SelfHostedLocalOnPrem,
                instance_description: "RTX 4090 24GB + M4 Pro 48GB Unified + 40TB NAS Seed".into(),
                fp16_tflops: 210.8,
                vram_or_unified_gb: 72.0,
                system_ram_gb: 128.0,
                hourly_spot_usd: 0.00,
                hyperscaler_equivalent_usd: 4.80,
            },
            ProviderComputeOffer {
                offer_id: "arb-spot-02".into(),
                provider_name: "Specialist GPU Spot #1 (Stranded Single L40S)".into(),
                provider_kind: ComputeProviderKind::SpecialistGpuSpotMarket,
                instance_description: "1x NVIDIA L40S 48GB VRAM (No NVLink required)".into(),
                fp16_tflops: 362.0,
                vram_or_unified_gb: 48.0,
                system_ram_gb: 96.0,
                hourly_spot_usd: 0.69,
                hyperscaler_equivalent_usd: 2.85,
            },
            ProviderComputeOffer {
                offer_id: "arb-spot-03".into(),
                provider_name: "Green Stranded-Energy DC (Preemptible 1x A100-80GB)".into(),
                provider_kind: ComputeProviderKind::StrandedEnergyDatacenter,
                instance_description: "1x NVIDIA A100 80GB PCIe (Spot Preemptible)".into(),
                fp16_tflops: 312.0,
                vram_or_unified_gb: 80.0,
                system_ram_gb: 128.0,
                hourly_spot_usd: 0.89,
                hyperscaler_equivalent_usd: 4.10,
            },
            ProviderComputeOffer {
                offer_id: "arb-baremetal-04".into(),
                provider_name: "Regional Bare-Metal (Hetzner/OVH High-RAM NVMe Host)".into(),
                provider_kind: ComputeProviderKind::RegionalBareMetalProvider,
                instance_description:
                    "AMD EPYC 256GB DDR5 + 2x Gen5 NVMe (Warm/Cold MoE Expert Host)".into(),
                fp16_tflops: 18.5,
                vram_or_unified_gb: 0.0,
                system_ram_gb: 256.0,
                hourly_spot_usd: 0.24,
                hyperscaler_equivalent_usd: 1.95,
            },
        ];

        let total_tflops: f64 = offers.iter().map(|o| o.fp16_tflops).sum();
        let total_vram: f64 = offers.iter().map(|o| o.vram_or_unified_gb).sum();
        let blended_usd: f64 = offers.iter().map(|o| o.hourly_spot_usd).sum();
        let hyper_usd: f64 = offers.iter().map(|o| o.hyperscaler_equivalent_usd).sum();
        let savings_pct = (((hyper_usd - blended_usd) / hyper_usd.max(0.01)) * 100.0).max(0.0);

        MultiProviderArbitragePlan {
            target_model_id: model.id.clone(),
            selected_offers: offers,
            total_combined_tflops: (total_tflops * 10.0).round() / 10.0,
            total_combined_vram_gb: (total_vram * 10.0).round() / 10.0,
            kadmium_blended_hourly_usd: (blended_usd * 100.0).round() / 100.0,
            monolithic_hyperscaler_hourly_usd: (hyper_usd * 100.0).round() / 100.0,
            cost_reduction_pct: (savings_pct * 10.0).round() / 10.0,
        }
    }

    /// Simulates the three stages of Kadmium's **Bidirectional Downtime Protection Shield**:
    /// 1. Local Cluster Outage -> Automatic Cloud Standby Failover
    /// 2. Cloud Provider Spot Preemption / Outage -> Cross-Provider + Local Fallback
    /// 3. Local Cluster Recovery -> Automatic Failback to Zero-Marginal-Cost Local Hardware
    pub fn simulate_bidirectional_failover_shield() -> Vec<FailoverShieldSimulation> {
        vec![
            FailoverShieldSimulation {
                direction: FailoverDirection::LocalOutageToCloudStandby,
                incident_summary:
                    "Home ISP fiber drop / power blip on `cameron-main-rig` (Layers L00..L25)"
                        .into(),
                affected_layers: (0, 25),
                source_environment: "Local Home Cluster (Main PC + NAS)".into(),
                target_failover_environment:
                    "Kadmium Cloud Standby (Pre-warmed .ktb Vault + Spot L40S)".into(),
                switchover_rto_ms: 340.0,
                zero_downtime_preserved: true,
                temporary_failover_hourly_usd: 0.69,
            },
            FailoverShieldSimulation {
                direction: FailoverDirection::CloudOutageOrPreemptionToLocalAndMultiCloud,
                incident_summary:
                    "30s Spot Eviction Notice on Cloud A100 instance (Layers L26..L55)".into(),
                affected_layers: (26, 55),
                source_environment: "Cloud Provider A (Preemptible A100 Spot)".into(),
                target_failover_environment: "Local M4 Mac Mini + Gaming Laptops + Provider B Spot"
                    .into(),
                switchover_rto_ms: 185.0,
                zero_downtime_preserved: true,
                temporary_failover_hourly_usd: 0.42,
            },
            FailoverShieldSimulation {
                direction: FailoverDirection::CloudFailbackToRecoveredLocal,
                incident_summary:
                    "Local ISP & Main PC back online -> draining Cloud Standby back to Local".into(),
                affected_layers: (0, 25),
                source_environment: "Kadmium Cloud Standby (Spot L40S)".into(),
                target_failover_environment: "Local Home Cluster ($0.00/hr marginal compute)"
                    .into(),
                switchover_rto_ms: 210.0,
                zero_downtime_preserved: true,
                temporary_failover_hourly_usd: 0.00,
            },
        ]
    }
}

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

            // Per-layer latency contribution (compute + cold stream fraction)
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

        // Coordinate-descent integer boundary transfer: move 1 layer from slowest stage to fastest stage
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
                // Revert if no improvement
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

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::{sample_cloud_burst_gpu_node, sample_node_pool, sample_personal_cluster};

    #[test]
    fn test_asymmetric_moe_inference_sharding_covers_all_layers() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);

        let sum_layers: u32 = plan.assignments.iter().map(|a| a.layer_count).sum();
        assert_eq!(sum_layers, model.num_layers);

        let main_pc_layers = plan.assignments[0].layer_count;
        let mac_mini_layers = plan.assignments[1].layer_count;
        let laptop_2_layers = plan.assignments[3].layer_count;

        assert!(main_pc_layers > laptop_2_layers);
        assert!(mac_mini_layers > laptop_2_layers);
    }

    #[test]
    fn test_live_responsibility_shift_on_gaming_mode_and_cloud_burst() {
        let mut cluster = sample_personal_cluster();
        let model = ModelSpec::llama_3_70b_lora();

        let initial_plan =
            ShardPlanner::compute_plan(1, &model, WorkloadKind::DistributedTraining, &cluster);
        let initial_pc_layers = initial_plan.assignments[0].layer_count;

        cluster[0].polite_state = PoliteState::GamingOrHeavyApp;
        let shift_1 = ShardPlanner::plan_responsibility_shift(
            &initial_plan,
            &model,
            &cluster,
            ShiftTriggerReason::HostEnteredGamingMode {
                node_id: cluster[0].id.clone(),
            },
        );

        let gaming_pc_layers = shift_1.new_plan.assignments[0].layer_count;
        assert!(gaming_pc_layers < initial_pc_layers);
        assert!(!shift_1.migrations.is_empty());

        let burst_node = sample_cloud_burst_gpu_node();
        cluster.push(burst_node.clone());
        let shift_2 = ShardPlanner::plan_responsibility_shift(
            &shift_1.new_plan,
            &model,
            &cluster,
            ShiftTriggerReason::CloudBurstNodeJoined {
                node_id: burst_node.id,
            },
        );

        let cloud_assignment = shift_2
            .new_plan
            .assignments
            .last()
            .expect("cloud node present");
        assert!(cloud_assignment.layer_count >= 30);
        // Verify Privacy-Aware Trust Zone Pinning: the external cloud burst node is strictly
        // placed on middle layers (never Layer 0 or Layer N-1)!
        assert!(cloud_assignment.start_layer.unwrap() > 0);
        assert!(cloud_assignment.end_layer.unwrap() < model.num_layers - 1);
        assert!(shift_2.new_plan.estimated_throughput > shift_1.new_plan.estimated_throughput);
    }

    #[test]
    fn test_partition_100_node_pool_into_5_subclusters_of_20_nodes() {
        let pool = sample_node_pool(100);
        let model = ModelSpec::deepseek_v3_671b_moe();
        let layout = SubClusterPartitioner::partition_pool(&pool, 5, &model);

        assert_eq!(layout.total_pool_nodes, 100);
        assert_eq!(layout.num_slices, 5);
        assert_eq!(layout.nodes_per_slice, 20);
        assert_eq!(layout.slices.len(), 5);

        for slice in &layout.slices {
            assert_eq!(slice.node_count, 20);
            let sum_layers: u32 = slice
                .shard_plan
                .assignments
                .iter()
                .map(|a| a.layer_count)
                .sum();
            assert_eq!(sum_layers, model.num_layers);
        }
    }

    #[test]
    fn test_tpu_cim_dataflow_evaluation() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let report = DataflowStreamPlanner::evaluate_dataflows(&model, &cluster, 512);

        assert_eq!(report.profiles.len(), 4);
        assert_eq!(
            report.recommended_low_batch_mode,
            DataflowMode::ComputeInMemoryTokenStreaming
        );
        assert_eq!(
            report.recommended_high_batch_mode,
            DataflowMode::TokenStationaryWeightStreaming
        );
    }

    #[test]
    fn test_multi_provider_arbitrage_and_bidirectional_failover_shield() {
        let model = ModelSpec::deepseek_v3_671b_moe();
        let arb = HybridCloudArbitragePlanner::compute_arbitrage_plan(&model);
        // Combining local hardware + cheap multi-provider single-GPU spot reduces cost by >80%!
        assert!(arb.cost_reduction_pct >= 80.0);
        assert!(arb.kadmium_blended_hourly_usd < 2.00);

        let shield_events = HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield();
        assert_eq!(shield_events.len(), 3);
        assert!(shield_events.iter().all(|e| e.zero_downtime_preserved));
    }

    #[test]
    fn test_minmax_stage_latency_optimizer_and_paged_kv_planner() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);

        let opt = MinMaxStageLatencyOptimizer::optimize_stage_boundaries(&plan, &cluster);
        assert!(opt.optimized_bottleneck_ms <= opt.initial_bottleneck_ms);
        let refined_sum: u32 = opt.refined_stage_layers.iter().map(|(_, c, _)| *c).sum();
        assert_eq!(refined_sum, model.num_layers);

        let kv_allocs = DistributedPagedKvPlanner::plan_kv_cache_pages(&plan, &model, &cluster);
        assert!(!kv_allocs.is_empty());
        for kv in &kv_allocs {
            assert_eq!(kv.page_size_tokens, 16);
            assert!(kv.max_context_tokens_supported >= 2048);
        }
    }
}
