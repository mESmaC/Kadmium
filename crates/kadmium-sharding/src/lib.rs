//! # `kadmium-sharding`
//!
//! Implements **Custom Asymmetric Model Sharding** and **Dynamic Compute Responsibility Shifting**
//! across wildly heterogeneous hardware for both **Inference** and **Distributed Training**.

use kadmium_core::{
    DeviceKind, HardwareNode, ModelSpec, NodeId, NodeRole, NodeTrustTier, PoliteState,
    TrainingStrategy, WorkloadKind,
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
            if matches!(
                node.device_kind,
                DeviceKind::LinuxSbc | DeviceKind::MobilePhone | DeviceKind::PhoneFarmChassis
            ) {
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

pub mod arbitrage;
pub mod cloud_shield_watcher;
pub mod hierarchical_pool;
pub mod optimizer;
pub mod slices;
pub mod spot_provisioner;

pub use arbitrage::*;
pub use cloud_shield_watcher::*;
pub use hierarchical_pool::*;
pub use optimizer::*;
pub use slices::*;
pub use spot_provisioner::*;

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::{
        DataflowMode, sample_cloud_burst_gpu_node, sample_node_pool, sample_personal_cluster,
    };

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

        let watcher = CloudShieldCutoverWatcher::evaluate_incident_timeline();
        assert!(watcher.measured_cutover_rto_ms < 400.0);
        assert!(watcher.zero_dropped_requests);
        assert!((watcher.total_prorated_incident_cost_usd - 0.27).abs() < 0.01);
        assert_eq!(
            watcher.steps.last().map(|s| s.state),
            Some(CloudShieldState::HealthyLocalZeroBurn)
        );

        let provisioned =
            SpotOrderbookBidEngine::provision_optimal_middle_shard(48.0, 1.50, (22, 48))
                .expect("provision optimal spot middle shard");
        assert!(provisioned.selected_quote.spot_usd_per_hr <= 1.50);
        assert_eq!(provisioned.pinned_middle_layers, (22, 48));
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
