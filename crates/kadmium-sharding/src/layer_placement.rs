//! Layer placement across machines for native checkpoint inference.
//!
//! Decides which machine runs which contiguous range of decoder layers using each part's real
//! in-memory weight bytes (`model.embed_tokens.weight`, each layer's matrices and widened norm or
//! bias vectors, `model.norm.weight`, and `lm_head.weight` or the shared tied embedding) plus the
//! `f32` KV cache bytes required for the model's full `max_position_embeddings` context window.
//!
//! Quality and safety contract:
//!
//! - Every layer `0 .. num_layers` is placed in order across a chain of machines without gaps or
//!   overlaps, with the token embedding on the first stage and the final norm and LM head on the
//!   last stage.
//! - Every stage's weight bytes plus its full-context KV cache bytes fit within that machine's
//!   offered resident memory (`fast_memory_gb + system_ram_gb` under the node's polite state,
//!   never touching the owner's RAM reserve or cold NVMe/NAS storage).
//! - Stage 0 (prompt ingress) and the final stage (LM head and sampling) are placed only on
//!   [`NodeTrustTier::TrustedCore`] machines.
//! - If the machines cannot hold all layers and the full native context, placement fails with a
//!   [`LayerPlacementError`] naming the required and offered bytes rather than shrinking the
//!   context window, dropping layers, or swapping to a smaller model.
//!
//! Proof coverage (**Proved (model)**):
//!
//! - `Kadmium.Sharding.LayerPlacement.planFrom_valid`
//! - `Kadmium.Sharding.LayerPlacement.stages_contiguous_cover`
//! - `Kadmium.Sharding.LayerPlacement.stages_within_budget`
//! - `Kadmium.Sharding.LayerPlacement.stages_boundary_trusted`
//! - `Kadmium.Sharding.LayerPlacement.single_stage_shares_tied_embedding`
//! - `Kadmium.Sharding.LayerPlacement.planFrom_complete`
//! - `Kadmium.Sharding.LayerPlacement.planFrom_none_iff_infeasible`
//! - `Kadmium.Sharding.LayerPlacement.planFrom_optimal`
//! - `Kadmium.Sharding.LayerPlacement.infeasible_when_total_exceeds_cluster_capacity`

use crate::ModelContentNames;
use kadmium_core::{HardwareNode, NodeId, NodeRole, NodeTrustTier, PoliteState, ResourceOffer};
use kadmium_grid::{ContentKind, ContentRef, PlacementRule, ResidencyTier};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use thiserror::Error;

const GIB: f64 = 1_073_741_824.0;

/// Errors returned when a checkpoint cannot be placed across the available machines at its full
/// native context length.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum LayerPlacementError {
    #[error("invalid checkpoint memory profile: {0}")]
    InvalidProfile(String),
    #[error("no eligible compute machines offer resident memory for model placement")]
    NoEligibleMachines,
    #[error(
        "none of the {untrusted_candidates} eligible machines is TrustedCore, which is required to host layer 0 (token embedding) and layer {last_layer} (LM head)"
    )]
    NoTrustedBoundaryMachine {
        untrusted_candidates: usize,
        last_layer: usize,
    },
    #[error(
        "the {eligible_machines} eligible machine(s) ({total_available_bytes} bytes offered across all machines, largest trusted machine {largest_trusted_machine_bytes} bytes) cannot hold all {num_layers} layers plus the {context_length}-position KV cache ({single_machine_bytes} bytes on one machine, {split_total_bytes} bytes across multiple machines, including {full_context_kv_bytes} bytes of full-context KV cache): {reason}; refusing to shorten the context or use a smaller model"
    )]
    InsufficientClusterMemory {
        num_layers: usize,
        context_length: usize,
        single_machine_bytes: u64,
        split_total_bytes: u64,
        full_context_kv_bytes: u64,
        total_available_bytes: u64,
        largest_trusted_machine_bytes: u64,
        eligible_machines: usize,
        trusted_machines: usize,
        reason: String,
    },
}

/// Exact in-memory byte profile of a checkpoint and its full-context KV cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMemoryProfile {
    /// Total decoder transformer layers (`num_hidden_layers`).
    pub num_layers: usize,
    /// Native context window (`max_position_embeddings`) served without truncation.
    pub max_position_embeddings: usize,
    /// In-memory bytes of `model.embed_tokens.weight`.
    pub embedding_bytes: u64,
    /// In-memory weight bytes of each decoder layer `0 .. num_layers`.
    pub layer_weight_bytes: Vec<u64>,
    /// In-memory bytes of `model.norm.weight` (widened to `f32`).
    pub final_norm_bytes: u64,
    /// In-memory bytes of `lm_head.weight` (or `model.embed_tokens.weight` when tied).
    pub lm_head_bytes: u64,
    /// Whether `lm_head` shares `model.embed_tokens.weight` when a single slice holds both ends.
    pub tie_word_embeddings: bool,
    /// Full-context `f32` KV cache bytes for one layer (`max_position_embeddings * 2 * kv_width * 4`).
    pub kv_bytes_per_layer: u64,
}

impl CheckpointMemoryProfile {
    /// Validates layer counts, context length, and byte totals against overflow.
    pub fn validate(&self) -> Result<(), LayerPlacementError> {
        if self.num_layers == 0 {
            return Err(LayerPlacementError::InvalidProfile(
                "`num_layers` must be at least 1".into(),
            ));
        }
        if self.max_position_embeddings == 0 {
            return Err(LayerPlacementError::InvalidProfile(
                "`max_position_embeddings` must be at least 1".into(),
            ));
        }
        if self.layer_weight_bytes.len() != self.num_layers {
            return Err(LayerPlacementError::InvalidProfile(format!(
                "`layer_weight_bytes` has {} entries for {} layers",
                self.layer_weight_bytes.len(),
                self.num_layers
            )));
        }
        let _ = self.split_total_bytes()?;
        Ok(())
    }

    /// In-memory weight bytes held by a slice running `layers` with `embedding` and `head`.
    ///
    /// When `embedding && head && self.tie_word_embeddings`, the tied token embedding matrix is
    /// shared in memory between the embedding lookup and the LM head and is counted once.
    pub fn slice_weight_bytes(
        &self,
        layers: Range<usize>,
        embedding: bool,
        head: bool,
    ) -> Result<u64, LayerPlacementError> {
        if layers.start > layers.end || layers.end > self.num_layers {
            return Err(LayerPlacementError::InvalidProfile(format!(
                "layer range {}..{} is outside 0..{}",
                layers.start, layers.end, self.num_layers
            )));
        }
        if self.layer_weight_bytes.len() != self.num_layers {
            return Err(LayerPlacementError::InvalidProfile(format!(
                "`layer_weight_bytes` has {} entries for {} layers",
                self.layer_weight_bytes.len(),
                self.num_layers
            )));
        }
        let emb = if embedding { self.embedding_bytes } else { 0 };
        let mut layer_sum = 0u64;
        for &w in &self.layer_weight_bytes[layers] {
            layer_sum = layer_sum.checked_add(w).ok_or_else(|| {
                LayerPlacementError::InvalidProfile("layer weight byte sum overflows u64".into())
            })?;
        }
        let hd = if head {
            let lm = if embedding && self.tie_word_embeddings {
                0
            } else {
                self.lm_head_bytes
            };
            self.final_norm_bytes.checked_add(lm).ok_or_else(|| {
                LayerPlacementError::InvalidProfile("head weight byte sum overflows u64".into())
            })?
        } else {
            0
        };
        emb.checked_add(layer_sum)
            .and_then(|v| v.checked_add(hd))
            .ok_or_else(|| {
                LayerPlacementError::InvalidProfile("slice weight byte sum overflows u64".into())
            })
    }

    /// Full-context `f32` KV cache bytes required for `layers` at `max_position_embeddings`.
    pub fn slice_kv_bytes(&self, layers: Range<usize>) -> Result<u64, LayerPlacementError> {
        if layers.start > layers.end || layers.end > self.num_layers {
            return Err(LayerPlacementError::InvalidProfile(format!(
                "layer range {}..{} is outside 0..{}",
                layers.start, layers.end, self.num_layers
            )));
        }
        let count = (layers.end - layers.start) as u64;
        count.checked_mul(self.kv_bytes_per_layer).ok_or_else(|| {
            LayerPlacementError::InvalidProfile("slice KV cache byte total overflows u64".into())
        })
    }

    /// Total resident memory bytes (`weight_bytes + kv_bytes`) required by a slice at full context.
    pub fn slice_total_bytes(
        &self,
        layers: Range<usize>,
        embedding: bool,
        head: bool,
    ) -> Result<u64, LayerPlacementError> {
        let w = self.slice_weight_bytes(layers.clone(), embedding, head)?;
        let kv = self.slice_kv_bytes(layers)?;
        w.checked_add(kv).ok_or_else(|| {
            LayerPlacementError::InvalidProfile(
                "slice total memory byte requirement overflows u64".into(),
            )
        })
    }

    /// Total `f32` KV cache bytes across all `num_layers` at `max_position_embeddings`.
    pub fn full_context_kv_bytes(&self) -> Result<u64, LayerPlacementError> {
        self.slice_kv_bytes(0..self.num_layers)
    }

    /// Total resident bytes when a single machine holds the entire model (`0..num_layers`).
    pub fn whole_model_bytes(&self) -> Result<u64, LayerPlacementError> {
        self.slice_total_bytes(0..self.num_layers, true, true)
    }

    /// Total resident bytes across stages when the model is split across two or more machines
    /// (so stage 0 holds `embedding_bytes` and the last stage holds `final_norm_bytes + lm_head_bytes`).
    pub fn split_total_bytes(&self) -> Result<u64, LayerPlacementError> {
        let all_layers_weights = self.slice_weight_bytes(0..self.num_layers, false, false)?;
        let kv = self.full_context_kv_bytes()?;
        self.embedding_bytes
            .checked_add(all_layers_weights)
            .and_then(|v| v.checked_add(self.final_norm_bytes))
            .and_then(|v| v.checked_add(self.lm_head_bytes))
            .and_then(|v| v.checked_add(kv))
            .ok_or_else(|| {
                LayerPlacementError::InvalidProfile("split model byte total overflows u64".into())
            })
    }
}

/// Resident memory budget offered by a single candidate machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineMemoryBudget {
    pub node_id: NodeId,
    pub hostname: String,
    pub trust_tier: NodeTrustTier,
    /// Offered fast memory (VRAM or Unified Memory) under the node's polite state, in bytes.
    pub fast_bytes: u64,
    /// Offered host system RAM under the node's polite state (excluding `reserved_ram_gb`), in bytes.
    pub warm_bytes: u64,
    /// Total resident memory available for slice weights and KV cache (`fast_bytes + warm_bytes`).
    pub capacity_bytes: u64,
    /// Primary fast residency tier when `fast_bytes > 0`.
    pub preferred_tier: ResidencyTier,
    /// Whether the donor requires manual approval before staging.
    pub requires_approval: bool,
}

impl MachineMemoryBudget {
    /// Creates an explicit byte budget for `node_id`.
    pub fn new(node_id: impl Into<NodeId>, capacity_bytes: u64, trust_tier: NodeTrustTier) -> Self {
        let id = node_id.into();
        let hostname = id.0.clone();
        Self {
            node_id: id,
            hostname,
            trust_tier,
            fast_bytes: capacity_bytes,
            warm_bytes: 0,
            capacity_bytes,
            preferred_tier: ResidencyTier::Vram,
            requires_approval: trust_tier != NodeTrustTier::TrustedCore,
        }
    }

    /// Derives a machine's resident memory budget from its hardware profile and [`ResourceOffer`]
    /// under its current [`PoliteState`]. Cold NVMe and NAS tiers are never counted as resident
    /// memory. Returns `None` if the node is not an active compute worker or offers zero bytes.
    pub fn from_node_and_offer(node: &HardwareNode, offer: Option<&ResourceOffer>) -> Option<Self> {
        if node.polite_state == PoliteState::Draining
            || (!node.roles.contains(&NodeRole::ComputeWorker)
                && !node.roles.contains(&NodeRole::BurstAccelerator))
        {
            return None;
        }
        let base_offer = offer.cloned().unwrap_or_else(|| node.default_offer());
        let live = base_offer
            .clamped_to(node)
            .under_polite_state(node, node.polite_state);
        if live.slots == 0 || live.max_allocations == 0 {
            return None;
        }
        let fast_bytes = (live.fast_memory_gb.max(0.0) * GIB).floor() as u64;
        let warm_bytes = (live.system_ram_gb.max(0.0) * GIB).floor() as u64;
        let capacity_bytes = fast_bytes.saturating_add(warm_bytes);
        if capacity_bytes == 0 {
            return None;
        }
        let preferred_tier = if node.memory.unified_mem_gb > 0.0 && fast_bytes > 0 {
            ResidencyTier::UnifiedMemory
        } else if node.memory.vram_gb > 0.0 && fast_bytes > 0 {
            ResidencyTier::Vram
        } else {
            ResidencyTier::SystemRam
        };
        Some(Self {
            node_id: node.id.clone(),
            hostname: node.hostname.clone(),
            trust_tier: node.trust_tier,
            fast_bytes,
            warm_bytes,
            capacity_bytes,
            preferred_tier,
            requires_approval: live.requires_approval,
        })
    }

    /// Collects the resident memory budgets of all eligible compute machines in `nodes`.
    pub fn from_cluster(
        nodes: &[HardwareNode],
        offers: &BTreeMap<NodeId, ResourceOffer>,
    ) -> Vec<Self> {
        nodes
            .iter()
            .filter_map(|node| Self::from_node_and_offer(node, offers.get(&node.id)))
            .collect()
    }
}

/// One machine's assigned contiguous layer slice and its verified memory accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedLayerStage {
    pub stage_index: usize,
    pub node_id: NodeId,
    pub hostname: String,
    pub trust_tier: NodeTrustTier,
    /// Contiguous half-open layer range `start..end` executed by this machine.
    pub layers: Range<usize>,
    /// True for stage 0 (`layers.start == 0`), which holds `model.embed_tokens.weight`.
    pub embedding: bool,
    /// True for the final stage (`layers.end == num_layers`), which holds `model.norm.weight` and `lm_head`.
    pub head: bool,
    /// Exact in-memory weight bytes held by this stage.
    pub weight_bytes: u64,
    /// Exact full-context `f32` KV cache bytes held by this stage (`layers.len() * kv_bytes_per_layer`).
    pub kv_bytes: u64,
    /// Total resident memory required by this stage (`weight_bytes + kv_bytes`).
    pub total_bytes: u64,
    /// Resident memory capacity offered by this machine.
    pub capacity_bytes: u64,
    /// Offered resident memory left free after holding `total_bytes`.
    pub free_bytes: u64,
    /// Residency tier on this machine (`Vram`, `UnifiedMemory`, or `SystemRam`; never cold storage).
    pub residency_tier: ResidencyTier,
    /// Whether staging onto this machine requires donor approval.
    pub requires_approval: bool,
}

/// Complete multi-machine layer placement plan for a checkpoint at its native context length.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerPlacementPlan {
    pub num_layers: usize,
    pub context_length: usize,
    pub stages: Vec<PlannedLayerStage>,
    pub total_weight_bytes: u64,
    pub total_kv_bytes: u64,
    pub total_bytes: u64,
}

impl LayerPlacementPlan {
    /// True when the model is split across two or more machines.
    pub fn is_split(&self) -> bool {
        self.stages.len() > 1
    }

    /// Returns the stage that owns `layer`.
    pub fn stage_for_layer(&self, layer: usize) -> Option<&PlannedLayerStage> {
        self.stages.iter().find(|s| s.layers.contains(&layer))
    }

    /// Emits pinned [`PlacementRule`]s for `kadmium-grid` covering each stage's weight slice and
    /// full-context KV cache reservation on its assigned machine.
    pub fn placement_rules(&self, model_id: &str) -> Vec<PlacementRule> {
        let mut rules = Vec::with_capacity(self.stages.len() * 2);
        for stage in &self.stages {
            let start = stage.layers.start as u32;
            let end = (stage.layers.end - 1) as u32;
            rules.push(PlacementRule {
                content: ModelContentNames::trunk(model_id, start, end, stage.weight_bytes),
                min_replicas: 1,
                preferred_tier: stage.residency_tier,
                candidate_nodes: vec![stage.node_id.clone()],
                pin: true,
            });
            if stage.kv_bytes > 0 {
                let kv_id = format!("{model_id}/kv-full-context/L{start:03}-L{end:03}");
                rules.push(PlacementRule {
                    content: ContentRef::named(ContentKind::KvPage, &kv_id, stage.kv_bytes),
                    min_replicas: 1,
                    preferred_tier: stage.residency_tier,
                    candidate_nodes: vec![stage.node_id.clone()],
                    pin: true,
                });
            }
        }
        rules
    }
}

/// Orders candidate machines so [`NodeTrustTier::TrustedCore`] machines bookend middle-only peers
/// when at least two trusted machines and at least one middle-only machine are present.
pub fn order_candidate_machines(budgets: &[MachineMemoryBudget]) -> Vec<MachineMemoryBudget> {
    let mut trusted: Vec<MachineMemoryBudget> = budgets
        .iter()
        .filter(|b| b.capacity_bytes > 0 && b.trust_tier.can_host_boundary_ingress_egress())
        .cloned()
        .collect();
    let middle_only: Vec<MachineMemoryBudget> = budgets
        .iter()
        .filter(|b| b.capacity_bytes > 0 && !b.trust_tier.can_host_boundary_ingress_egress())
        .cloned()
        .collect();

    let mut ordered = Vec::with_capacity(trusted.len() + middle_only.len());
    if trusted.len() >= 2 && !middle_only.is_empty() {
        let egress_trusted = trusted.pop();
        ordered.extend(trusted);
        ordered.extend(middle_only);
        if let Some(last) = egress_trusted {
            ordered.push(last);
        }
    } else {
        ordered.extend(trusted);
        ordered.extend(middle_only);
    }
    ordered
}

/// Rational comparison `a.0 / a.1 <= b.0 / b.1` via cross-multiplication in `u128`.
fn load_le(a: (u64, u64), b: (u64, u64)) -> bool {
    (a.0 as u128) * (b.1 as u128) <= (b.0 as u128) * (a.1 as u128)
}

fn max_load(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    if load_le(b, a) { a } else { b }
}

fn plan_load(stages: &[PlannedLayerStage]) -> (u64, u64) {
    stages.iter().rev().fold((0u64, 1u64), |acc, st| {
        max_load((st.total_bytes, st.capacity_bytes), acc)
    })
}

fn pick_better(
    best: Option<Vec<PlannedLayerStage>>,
    cand: Vec<PlannedLayerStage>,
) -> Option<Vec<PlannedLayerStage>> {
    match best {
        None => Some(cand),
        Some(b) => {
            if load_le(plan_load(&b), plan_load(&cand)) {
                Some(b)
            } else {
                Some(cand)
            }
        }
    }
}

fn build_stage_if_valid(
    profile: &CheckpointMemoryProfile,
    machine: &MachineMemoryBudget,
    start: usize,
    stop: usize,
) -> Option<PlannedLayerStage> {
    if start >= stop || stop > profile.num_layers || machine.capacity_bytes == 0 {
        return None;
    }
    let embedding = start == 0;
    let head = stop == profile.num_layers;
    if (embedding || head) && !machine.trust_tier.can_host_boundary_ingress_egress() {
        return None;
    }
    let weight_bytes = profile
        .slice_weight_bytes(start..stop, embedding, head)
        .ok()?;
    let kv_bytes = profile.slice_kv_bytes(start..stop).ok()?;
    let total_bytes = weight_bytes.checked_add(kv_bytes)?;
    if total_bytes > machine.capacity_bytes {
        return None;
    }
    let residency_tier = if machine.fast_bytes >= total_bytes {
        machine.preferred_tier
    } else {
        ResidencyTier::SystemRam
    };
    Some(PlannedLayerStage {
        stage_index: 0,
        node_id: machine.node_id.clone(),
        hostname: machine.hostname.clone(),
        trust_tier: machine.trust_tier,
        layers: start..stop,
        embedding,
        head,
        weight_bytes,
        kv_bytes,
        total_bytes,
        capacity_bytes: machine.capacity_bytes,
        free_bytes: machine.capacity_bytes - total_bytes,
        residency_tier,
        requires_approval: machine.requires_approval,
    })
}

/// Planner that assigns contiguous ranges of decoder layers across machines using real tensor
/// weight sizes and full-context KV cache memory.
pub struct LayerPlacementPlanner;

impl LayerPlacementPlanner {
    /// Plans layer placement across `nodes` using their [`ResourceOffer`]s under each node's
    /// current [`PoliteState`].
    pub fn plan_for_cluster(
        profile: &CheckpointMemoryProfile,
        nodes: &[HardwareNode],
        offers: &BTreeMap<NodeId, ResourceOffer>,
    ) -> Result<LayerPlacementPlan, LayerPlacementError> {
        let budgets = MachineMemoryBudget::from_cluster(nodes, offers);
        Self::plan_with_budgets(profile, &budgets)
    }

    /// Plans layer placement across explicit machine resident memory budgets.
    pub fn plan_with_budgets(
        profile: &CheckpointMemoryProfile,
        budgets: &[MachineMemoryBudget],
    ) -> Result<LayerPlacementPlan, LayerPlacementError> {
        profile.validate()?;
        let ordered = order_candidate_machines(budgets);
        if ordered.is_empty() {
            return Err(LayerPlacementError::NoEligibleMachines);
        }
        let trusted_count = ordered
            .iter()
            .filter(|b| b.trust_tier.can_host_boundary_ingress_egress())
            .count();
        if trusted_count == 0 {
            return Err(LayerPlacementError::NoTrustedBoundaryMachine {
                untrusted_candidates: ordered.len(),
                last_layer: profile.num_layers - 1,
            });
        }

        // Dynamic programming matching `Kadmium.Sharding.LayerPlacement.planFrom`:
        // `dp[start]` holds the optimal suffix chain covering `start..num_layers`.
        let n = profile.num_layers;
        let mut dp: Vec<Option<Vec<PlannedLayerStage>>> = vec![None; n + 1];
        dp[n] = Some(Vec::new());

        for machine in ordered.iter().rev() {
            let mut next_dp: Vec<Option<Vec<PlannedLayerStage>>> = vec![None; n + 1];
            next_dp[n] = Some(Vec::new());
            for start in 0..n {
                let mut with_machine: Option<Vec<PlannedLayerStage>> = None;
                for (stop, tail_opt) in dp.iter().enumerate().take(n + 1).skip(start + 1) {
                    if let Some(stage) = build_stage_if_valid(profile, machine, start, stop)
                        && let Some(tail) = tail_opt
                    {
                        let mut cand = Vec::with_capacity(1 + tail.len());
                        cand.push(stage);
                        cand.extend(tail.iter().cloned());
                        with_machine = pick_better(with_machine, cand);
                    }
                }
                next_dp[start] = match &dp[start] {
                    Some(skip_plan) => pick_better(with_machine, skip_plan.clone()),
                    None => with_machine,
                };
            }
            dp = next_dp;
        }

        if let Some(mut stages) = dp[0].take() {
            let mut total_weight_bytes = 0u64;
            let mut total_kv_bytes = 0u64;
            let mut total_bytes = 0u64;
            for (idx, stage) in stages.iter_mut().enumerate() {
                stage.stage_index = idx;
                total_weight_bytes = total_weight_bytes.saturating_add(stage.weight_bytes);
                total_kv_bytes = total_kv_bytes.saturating_add(stage.kv_bytes);
                total_bytes = total_bytes.saturating_add(stage.total_bytes);
            }
            return Ok(LayerPlacementPlan {
                num_layers: profile.num_layers,
                context_length: profile.max_position_embeddings,
                stages,
                total_weight_bytes,
                total_kv_bytes,
                total_bytes,
            });
        }

        let single_machine_bytes = profile.whole_model_bytes()?;
        let split_total_bytes = profile.split_total_bytes()?;
        let full_context_kv_bytes = profile.full_context_kv_bytes()?;
        let total_available_bytes: u64 = ordered
            .iter()
            .fold(0u64, |acc, b| acc.saturating_add(b.capacity_bytes));
        let largest_trusted_machine_bytes = ordered
            .iter()
            .filter(|b| b.trust_tier.can_host_boundary_ingress_egress())
            .map(|b| b.capacity_bytes)
            .max()
            .unwrap_or(0);

        let reason = if trusted_count == 1 && ordered.len() > 1 {
            format!(
                "only 1 machine is TrustedCore ({largest_trusted_machine_bytes} bytes offered, {single_machine_bytes} bytes required on one machine), and middle-only peers cannot host layer 0 or layer {} without a second TrustedCore boundary machine",
                profile.num_layers - 1
            )
        } else if total_available_bytes < single_machine_bytes {
            format!(
                "combined offered memory across all eligible machines ({total_available_bytes} bytes) is less than the model weights and full-context KV cache ({single_machine_bytes} bytes)"
            )
        } else if ordered.len() == 1 || profile.num_layers == 1 {
            format!(
                "no single TrustedCore machine offers the {single_machine_bytes} bytes needed to hold all {} layers, the embedding, the LM head, and the full-context KV cache",
                profile.num_layers
            )
        } else {
            format!(
                "no contiguous cut of the {} layers fits each machine's individual memory offer alongside the token embedding ({} bytes), LM head ({} bytes), and full-context KV cache ({} bytes/layer)",
                profile.num_layers,
                profile.embedding_bytes,
                profile
                    .final_norm_bytes
                    .saturating_add(profile.lm_head_bytes),
                profile.kv_bytes_per_layer
            )
        };

        Err(LayerPlacementError::InsufficientClusterMemory {
            num_layers: profile.num_layers,
            context_length: profile.max_position_embeddings,
            single_machine_bytes,
            split_total_bytes,
            full_context_kv_bytes,
            total_available_bytes,
            largest_trusted_machine_bytes,
            eligible_machines: ordered.len(),
            trusted_machines: trusted_count,
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::{ComputeBackend, DeviceKind, MemoryTierProfile, PoliteState};

    fn sample_profile(tied: bool) -> CheckpointMemoryProfile {
        CheckpointMemoryProfile {
            num_layers: 4,
            max_position_embeddings: 4096,
            embedding_bytes: 500,
            layer_weight_bytes: vec![1_000, 1_000, 1_000, 1_000],
            final_norm_bytes: 100,
            lm_head_bytes: 500,
            tie_word_embeddings: tied,
            kv_bytes_per_layer: 500,
        }
    }

    fn make_node(
        id: &str,
        vram_gb: f64,
        system_ram_gb: f64,
        nvme_cache_gb: f64,
        trust_tier: NodeTrustTier,
        polite_state: PoliteState,
    ) -> HardwareNode {
        HardwareNode {
            id: NodeId(id.into()),
            hostname: id.into(),
            device_kind: DeviceKind::DesktopWorkstation,
            backend: ComputeBackend::Cuda,
            roles: vec![NodeRole::ComputeWorker],
            trust_tier,
            gpu_name: "Test GPU".into(),
            cpu_name: "Test CPU".into(),
            npu_name: None,
            npu_tops: 0.0,
            fp16_tflops: 40.0,
            int4_tops: 80.0,
            cpu_threads: 8,
            cpu_tflops: 1.0,
            memory: MemoryTierProfile {
                vram_gb,
                vram_bw_gbps: 500.0,
                unified_mem_gb: 0.0,
                unified_mem_bw_gbps: 0.0,
                system_ram_gb,
                system_ram_bw_gbps: 60.0,
                interconnect_bw_gbps: 32.0,
                nvme_cache_gb,
                nvme_read_bw_gbps: 7.0,
            },
            polite_state,
            network_uplink_gbps: 10.0,
            network_latency_ms: 0.3,
        }
    }

    #[test]
    fn single_machine_shares_tied_embedding_while_multi_machine_counts_both_ends() {
        let tied = sample_profile(true);
        assert_eq!(tied.whole_model_bytes().unwrap(), 6_600);
        assert_eq!(tied.split_total_bytes().unwrap(), 7_100);

        let one = [MachineMemoryBudget::new(
            NodeId("box-a".into()),
            6_600,
            NodeTrustTier::TrustedCore,
        )];
        let plan_one = LayerPlacementPlanner::plan_with_budgets(&tied, &one).unwrap();
        assert_eq!(plan_one.stages.len(), 1);
        assert_eq!(plan_one.stages[0].layers, 0..4);
        assert!(plan_one.stages[0].embedding);
        assert!(plan_one.stages[0].head);
        assert_eq!(plan_one.stages[0].weight_bytes, 4_600);
        assert_eq!(plan_one.stages[0].kv_bytes, 2_000);
        assert_eq!(plan_one.total_bytes, 6_600);

        let untied = sample_profile(false);
        assert_eq!(untied.whole_model_bytes().unwrap(), 7_100);
        let err = LayerPlacementPlanner::plan_with_budgets(&untied, &one).unwrap_err();
        assert!(matches!(
            err,
            LayerPlacementError::InsufficientClusterMemory {
                single_machine_bytes: 7_100,
                total_available_bytes: 6_600,
                ..
            }
        ));

        let two = [
            MachineMemoryBudget::new(NodeId("box-a".into()), 3_600, NodeTrustTier::TrustedCore),
            MachineMemoryBudget::new(NodeId("box-b".into()), 3_600, NodeTrustTier::TrustedCore),
        ];
        let plan_two = LayerPlacementPlanner::plan_with_budgets(&tied, &two).unwrap();
        assert_eq!(plan_two.stages.len(), 2);
        assert_eq!(plan_two.stages[0].layers, 0..2);
        assert_eq!(plan_two.stages[0].weight_bytes, 2_500);
        assert_eq!(plan_two.stages[0].kv_bytes, 1_000);
        assert_eq!(plan_two.stages[1].layers, 2..4);
        assert_eq!(plan_two.stages[1].weight_bytes, 2_600);
        assert_eq!(plan_two.stages[1].kv_bytes, 1_000);
        assert_eq!(plan_two.total_bytes, 7_100);
    }

    #[test]
    fn trusted_machines_bookend_middle_only_peers_and_placement_rules_pin_stages() {
        let profile = sample_profile(true);
        let budgets = [
            MachineMemoryBudget::new(
                NodeId("trusted-1".into()),
                2_200,
                NodeTrustTier::TrustedCore,
            ),
            MachineMemoryBudget::new(
                NodeId("trusted-2".into()),
                2_200,
                NodeTrustTier::TrustedCore,
            ),
            MachineMemoryBudget::new(
                NodeId("peer-mid".into()),
                3_200,
                NodeTrustTier::VerifiedPeerMiddleLayersOnly,
            ),
        ];
        let plan = LayerPlacementPlanner::plan_with_budgets(&profile, &budgets).unwrap();
        assert_eq!(plan.stages.len(), 3);
        assert_eq!(plan.stages[0].node_id, NodeId("trusted-1".into()));
        assert_eq!(plan.stages[0].layers, 0..1);
        assert!(plan.stages[0].embedding && !plan.stages[0].head);

        assert_eq!(plan.stages[1].node_id, NodeId("peer-mid".into()));
        assert_eq!(plan.stages[1].layers, 1..3);
        assert!(!plan.stages[1].embedding && !plan.stages[1].head);

        assert_eq!(plan.stages[2].node_id, NodeId("trusted-2".into()));
        assert_eq!(plan.stages[2].layers, 3..4);
        assert!(!plan.stages[2].embedding && plan.stages[2].head);

        let rules = plan.placement_rules("qwen2");
        assert_eq!(rules.len(), 6);
        assert!(rules.iter().all(|r| r.pin && r.min_replicas == 1));
        assert_eq!(rules[0].candidate_nodes, vec![NodeId("trusted-1".into())]);
        assert_eq!(rules[2].candidate_nodes, vec![NodeId("peer-mid".into())]);
        assert_eq!(rules[4].candidate_nodes, vec![NodeId("trusted-2".into())]);
    }

    #[test]
    fn resource_offers_and_polite_state_exclude_cold_storage_and_draining_nodes() {
        let active = make_node(
            "workstation",
            8.0,
            16.0,
            200.0,
            NodeTrustTier::TrustedCore,
            PoliteState::IdleDedicated,
        );
        let draining = make_node(
            "draining",
            24.0,
            64.0,
            500.0,
            NodeTrustTier::TrustedCore,
            PoliteState::Draining,
        );
        let mut offers = BTreeMap::new();
        offers.insert(
            active.id.clone(),
            ResourceOffer {
                fast_memory_gb: 2.0,
                system_ram_gb: 4.0,
                nvme_gb: 100.0,
                reserved_ram_gb: 12.0,
                slots: 1,
                max_allocations: 4,
                requires_approval: false,
            },
        );
        let budgets = MachineMemoryBudget::from_cluster(&[active, draining], &offers);
        assert_eq!(budgets.len(), 1);
        assert_eq!(budgets[0].node_id, NodeId("workstation".into()));
        assert_eq!(budgets[0].fast_bytes, 2 * 1_073_741_824);
        assert_eq!(budgets[0].warm_bytes, 4 * 1_073_741_824);
        assert_eq!(budgets[0].capacity_bytes, 6 * 1_073_741_824);
    }

    #[test]
    fn fails_when_full_context_kv_cache_overflows_even_if_weights_alone_fit() {
        let profile = CheckpointMemoryProfile {
            num_layers: 2,
            max_position_embeddings: 32_768,
            embedding_bytes: 500,
            layer_weight_bytes: vec![1_000, 1_000],
            final_norm_bytes: 100,
            lm_head_bytes: 500,
            tie_word_embeddings: true,
            kv_bytes_per_layer: 2_000,
        };
        // Weights alone are 500 + 2000 + 100 = 2600 bytes, which would fit in 4000 bytes,
        // but full-context KV cache requires 2 * 2000 = 4000 additional bytes (total 6600).
        let budgets = [MachineMemoryBudget::new(
            NodeId("small-box".into()),
            4_000,
            NodeTrustTier::TrustedCore,
        )];
        let err = LayerPlacementPlanner::plan_with_budgets(&profile, &budgets).unwrap_err();
        match err {
            LayerPlacementError::InsufficientClusterMemory {
                context_length,
                single_machine_bytes,
                full_context_kv_bytes,
                total_available_bytes,
                ..
            } => {
                assert_eq!(context_length, 32_768);
                assert_eq!(single_machine_bytes, 6_600);
                assert_eq!(full_context_kv_bytes, 4_000);
                assert_eq!(total_available_bytes, 4_000);
            }
            other => panic!("expected InsufficientClusterMemory, got {other:?}"),
        }
    }
}
