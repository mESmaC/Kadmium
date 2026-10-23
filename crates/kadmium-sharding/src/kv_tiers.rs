//! KV pages as catalog content with residency tiers, and the bounded replay that rebuilds a
//! session's decoder state when its pages have sunk (ARCHITECTURE 9.12).
//!
//! DeepInfer's deployment keeps the in-flight KV in HBM, the sliding-window state in host
//! memory, and the paged block KV on a dedicated SSD cluster with days of retention; a hit whose
//! window state is missing backs off one block and recomputes the last 128 tokens, which is
//! what its `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` setting (the window plus the maximum
//! backtracking length) sizes. Here the same shape runs over the mesh's tiers: a session's KV
//! page per layer is content in the replica catalog, resident in fast memory on the trunk
//! owner while the session is active; an idle session's pages sink to the owner's NVMe and
//! then to the vault under a [`KvTierPolicy`]; and a session that resumes from a sunk page
//! rebuilds its state with a [`AppKind::KvReplay`] unit over a bounded suffix on the trunk
//! owner. Everything about size here is modelled from the model spec's reference KV figure;
//! no claim is made about a real model's KV.
//!
//! Left out: 3FS and its erasure coding, time-to-live retention, GPUDirect swap-in, and the
//! prefix-cache hit policy across sessions.

use kadmium_core::{HardwareNode, ModelSpec, NodeId, NodeRole};
use kadmium_grid::{
    AppKind, ContentKind, ContentRef, CostEstimate, PlacementConstraints, PlacementRule,
    RedundancyPolicy, Replica, ReplicaCatalog, ResidencyTier, StagingJob, WorkGraph, WorkUnit,
    WorkUnitId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{ExpertPlacementPlan, StepGraphBuilder};

const GIB: f64 = 1_073_741_824.0;

/// The context window the model spec's `kv_cache_per_layer_gb` is taken at, for the modelled
/// bytes per token and layer. An assumption, stated here once.
pub const KV_REFERENCE_CONTEXT_TOKENS: u64 = 32_768;

/// Stable names for a session's KV pages.
pub struct KvPageNames;

impl KvPageNames {
    pub fn page_id(model_id: &str, session_id: &str, layer: u32) -> String {
        format!("{model_id}/kv/{session_id}/L{layer:03}")
    }

    /// The `(model, session, layer)` a page id names, if it is one.
    pub fn parse(content_id: &str) -> Option<(String, String, u32)> {
        let (rest, layer) = content_id.rsplit_once("/L")?;
        let (model, session) = rest.split_once("/kv/")?;
        if model.is_empty() || session.is_empty() || layer.len() < 3 {
            return None;
        }
        Some((model.to_string(), session.to_string(), layer.parse().ok()?))
    }

    /// Modelled bytes of one layer's page for `tokens` of context.
    pub fn page_bytes(model: &ModelSpec, tokens: u64) -> u64 {
        let per_token = model.kv_cache_per_layer_gb * GIB / KV_REFERENCE_CONTEXT_TOKENS as f64;
        (per_token * tokens as f64).ceil() as u64
    }

    pub fn page(model: &ModelSpec, session_id: &str, layer: u32, tokens: u64) -> ContentRef {
        ContentRef::named(
            ContentKind::KvPage,
            &Self::page_id(&model.id, session_id, layer),
            Self::page_bytes(model, tokens),
        )
    }
}

/// Where a session's pages should sit, by how long it has been idle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvTierPolicy {
    /// Idle longer than this: pages sink from fast memory to the owner's NVMe.
    pub idle_to_nvme_ms: u64,
    /// Idle longer than this: pages sink to the vault.
    pub idle_to_vault_ms: u64,
    /// The attention window a resumed session must see again.
    pub window_tokens: u32,
    /// The longest draft a resumed session may have to back out of, which the replayed suffix
    /// also covers (DeepInfer keeps the window plus the maximum backtracking length).
    pub backtrack_tokens: u32,
}

impl Default for KvTierPolicy {
    fn default() -> Self {
        Self {
            idle_to_nvme_ms: 60_000,
            idle_to_vault_ms: 10 * 60_000,
            window_tokens: 128,
            backtrack_tokens: 12,
        }
    }
}

impl KvTierPolicy {
    /// Tokens a replay recomputes: the window plus the backtrack, bounded by the context.
    pub fn replay_tokens(&self, tokens_total: u64) -> u32 {
        (self.window_tokens as u64 + self.backtrack_tokens as u64).min(tokens_total.max(1)) as u32
    }

    /// The tier a session idle for `idle_ms` belongs in, given whether its owner offers NVMe.
    pub fn tier_for_idle(
        &self,
        idle_ms: u64,
        owner_has_nvme: bool,
        fast: ResidencyTier,
    ) -> ResidencyTier {
        if idle_ms >= self.idle_to_vault_ms {
            ResidencyTier::NasVault
        } else if idle_ms >= self.idle_to_nvme_ms {
            if owner_has_nvme {
                ResidencyTier::LocalNvme
            } else {
                ResidencyTier::NasVault
            }
        } else {
            fast
        }
    }
}

/// A decode session as the KV tiering sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvSessionState {
    pub session_id: String,
    pub model_id: String,
    /// Context tokens the session's pages hold.
    pub tokens: u64,
    pub last_active_unix_ms: u64,
    /// The tier of every page at the last tiering pass (all pages move together).
    pub tier: ResidencyTier,
    /// Tokens a resume replayed so far, summed.
    #[serde(default)]
    pub replayed_tokens: u64,
    #[serde(default)]
    pub replays: u32,
}

impl KvSessionState {
    pub fn new(session_id: &str, model_id: &str, now: u64) -> Self {
        Self {
            session_id: session_id.to_string(),
            model_id: model_id.to_string(),
            tokens: 0,
            last_active_unix_ms: now,
            tier: ResidencyTier::Vram,
            replayed_tokens: 0,
            replays: 0,
        }
    }

    /// Whether a decode step must replay before it can run: the pages have sunk to NVMe or
    /// the vault (a CPU-only owner keeps its pages in system RAM, which is not sunk).
    pub fn needs_replay(&self) -> bool {
        KvTierPlanner::is_sunk(self.tier)
    }
}

/// One page's move between tiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvTierMove {
    pub session_id: String,
    pub layer: u32,
    pub node: NodeId,
    pub from: ResidencyTier,
    pub to: ResidencyTier,
    pub bytes_modelled: u64,
}

/// What a tiering pass decided.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KvTierPlan {
    pub moves: Vec<KvTierMove>,
    /// Sessions whose pages sink to NVMe or the vault this pass.
    pub sunk: Vec<String>,
    /// Sessions whose pages come back to fast memory this pass (a resume).
    pub raised: Vec<String>,
    /// Modelled page bytes per tier after the pass.
    pub bytes_by_tier: BTreeMap<String, u64>,
    /// Always true: every byte figure comes from the model spec's reference KV size.
    pub modelled: bool,
}

pub struct KvTierPlanner;

impl KvTierPlanner {
    /// Whether pages in a tier have left their owner's memory: NVMe or the vault.
    pub fn is_sunk(tier: ResidencyTier) -> bool {
        matches!(tier, ResidencyTier::LocalNvme | ResidencyTier::NasVault)
    }

    /// The fast tier of a node: unified memory where it has any, VRAM where it has that, RAM
    /// otherwise.
    pub fn fast_tier_of(node: &HardwareNode) -> ResidencyTier {
        if node.memory.unified_mem_gb > 0.0 {
            ResidencyTier::UnifiedMemory
        } else if node.memory.vram_gb > 0.0 {
            ResidencyTier::Vram
        } else {
            ResidencyTier::SystemRam
        }
    }

    /// Records a session's pages as resident in fast memory on each layer's trunk owner, as
    /// a decode step leaves them; the replica's bytes are the modelled page size.
    pub fn record_resident(
        catalog: &mut ReplicaCatalog,
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        session: &KvSessionState,
        now: u64,
    ) -> usize {
        let mut pages = 0;
        for range in &placement.trunk_ranges {
            let tier = nodes
                .iter()
                .find(|n| n.id == range.owner)
                .map(Self::fast_tier_of)
                .unwrap_or(ResidencyTier::SystemRam);
            for layer in range.start_layer..=range.end_layer {
                let page = KvPageNames::page(model, &session.session_id, layer, session.tokens);
                // The vault keeps no copy while the session is active: one replica per page.
                let holders = catalog.holder_nodes(&page.content_id);
                for holder in holders {
                    if holder != range.owner {
                        catalog.remove(&holder, &page.content_id);
                    }
                }
                catalog.record(Replica::new(page, range.owner.clone(), tier, now).pinned());
                pages += 1;
            }
        }
        pages
    }

    /// One tiering pass over `sessions` at `now`: every page of a session idle past the
    /// policy's thresholds moves down to the owner's NVMe or to the vault (the vault node when
    /// the mesh has one), and a session marked active comes back up. The moves are applied to
    /// the catalog and the sessions' tiers updated.
    #[allow(clippy::too_many_arguments)]
    pub fn tier(
        catalog: &mut ReplicaCatalog,
        sessions: &mut [KvSessionState],
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        policy: &KvTierPolicy,
        now: u64,
    ) -> KvTierPlan {
        let vault = nodes
            .iter()
            .find(|n| n.roles.contains(&NodeRole::StorageVault))
            .map(|n| n.id.clone());
        let mut plan = KvTierPlan {
            modelled: true,
            ..KvTierPlan::default()
        };
        for session in sessions.iter_mut() {
            if session.model_id != model.id {
                continue;
            }
            let idle = now.saturating_sub(session.last_active_unix_ms);
            let mut new_tier = session.tier;
            for range in &placement.trunk_ranges {
                let owner = nodes.iter().find(|n| n.id == range.owner);
                let fast = owner
                    .map(Self::fast_tier_of)
                    .unwrap_or(ResidencyTier::SystemRam);
                let has_nvme = owner.is_some_and(|n| n.memory.nvme_cache_gb > 0.0);
                let target = policy.tier_for_idle(idle, has_nvme, fast);
                let target_node = if target == ResidencyTier::NasVault {
                    vault.clone().unwrap_or_else(|| range.owner.clone())
                } else {
                    range.owner.clone()
                };
                for layer in range.start_layer..=range.end_layer {
                    let page = KvPageNames::page(model, &session.session_id, layer, session.tokens);
                    let current = catalog
                        .holders(&page.content_id)
                        .first()
                        .map(|r| (r.node_id.clone(), r.tier));
                    let Some((node, from)) = current else {
                        continue;
                    };
                    if from == target && node == target_node {
                        continue;
                    }
                    catalog.remove(&node, &page.content_id);
                    let mut replica = Replica::new(page.clone(), target_node.clone(), target, now);
                    replica.pinned = !Self::is_sunk(target);
                    catalog.record(replica);
                    plan.moves.push(KvTierMove {
                        session_id: session.session_id.clone(),
                        layer,
                        node: target_node.clone(),
                        from,
                        to: target,
                        bytes_modelled: page.bytes,
                    });
                }
                new_tier = target;
            }
            if new_tier != session.tier {
                if !Self::is_sunk(new_tier) {
                    plan.raised.push(session.session_id.clone());
                } else if !Self::is_sunk(session.tier) || new_tier.rank() > session.tier.rank() {
                    plan.sunk.push(session.session_id.clone());
                }
                session.tier = new_tier;
            }
        }
        for session in sessions.iter().filter(|s| s.model_id == model.id) {
            for range in &placement.trunk_ranges {
                for layer in range.start_layer..=range.end_layer {
                    let id = KvPageNames::page_id(&model.id, &session.session_id, layer);
                    for r in catalog.holders(&id) {
                        *plan
                            .bytes_by_tier
                            .entry(format!("{:?}", r.tier))
                            .or_default() += r.content.bytes;
                    }
                }
            }
        }
        plan
    }

    /// The rules a placement would carry for a session's pages at their current tier, so the
    /// staging planner can diff them like any other content.
    pub fn rules_for(
        catalog: &ReplicaCatalog,
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        session: &KvSessionState,
    ) -> Vec<PlacementRule> {
        let mut rules = Vec::new();
        for range in &placement.trunk_ranges {
            for layer in range.start_layer..=range.end_layer {
                let page = KvPageNames::page(model, &session.session_id, layer, session.tokens);
                let (tier, node) = catalog
                    .holders(&page.content_id)
                    .first()
                    .map(|r| (r.tier, r.node_id.clone()))
                    .unwrap_or((ResidencyTier::Vram, range.owner.clone()));
                rules.push(PlacementRule {
                    content: page,
                    min_replicas: 1,
                    preferred_tier: tier,
                    candidate_nodes: vec![node],
                    pin: !Self::is_sunk(tier),
                });
            }
        }
        rules
    }

    /// Transfers that bring a session's pages back to fast memory on their owners: what a
    /// resume stages before, or instead of, replaying.
    pub fn raise_jobs(
        catalog: &ReplicaCatalog,
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        session: &KvSessionState,
    ) -> Vec<StagingJob> {
        let mut jobs = Vec::new();
        for range in &placement.trunk_ranges {
            let fast = nodes
                .iter()
                .find(|n| n.id == range.owner)
                .map(Self::fast_tier_of)
                .unwrap_or(ResidencyTier::SystemRam);
            for layer in range.start_layer..=range.end_layer {
                let page = KvPageNames::page(model, &session.session_id, layer, session.tokens);
                if let Some(r) = catalog.holders(&page.content_id).first()
                    && (r.node_id != range.owner || Self::is_sunk(r.tier))
                {
                    jobs.push(StagingJob {
                        content: page.clone(),
                        source_node: r.node_id.clone(),
                        destination_node: range.owner.clone(),
                        target_tier: fast,
                        bytes: page.bytes,
                        pin: true,
                    });
                }
            }
        }
        jobs
    }
}

/// Parameters of a [`AppKind::KvReplay`] unit: rebuild the pages of one trunk range for a
/// session from a bounded suffix of its context, on the range's owner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvReplayParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub stage_index: usize,
    pub start_layer: u32,
    pub end_layer: u32,
    /// Tokens recomputed: the window plus the backtrack, bounded by the context.
    pub suffix_tokens: u32,
    /// The session's context, which sizes the pages the replay re-registers.
    pub tokens_total: u64,
    /// Modelled bytes of one layer's page at `tokens_total`, for the pages the unit reports.
    pub page_bytes_modelled: u64,
    pub q_star_ratio: f64,
    pub experts_per_layer: u32,
    pub active_experts_per_token: u32,
}

impl StepGraphBuilder {
    /// The input of a replay graph: the suffix's rows, one per token.
    pub fn replay_input_ref(graph_id: &str) -> ContentRef {
        ContentRef::named(ContentKind::Activation, &format!("{graph_id}/suffix"), 0)
    }

    /// A replay graph: one [`AppKind::KvReplay`] unit per trunk range on its owner, chained
    /// like the decode step, each running the suffix through its layers and re-registering its
    /// pages. The suffix rows are the graph's seed.
    pub fn kv_replay_graph(
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        session_id: &str,
        epoch: u64,
        tokens_total: u64,
        policy: &KvTierPolicy,
    ) -> WorkGraph {
        let graph_id = format!("{session_id}/replay-{epoch}");
        let suffix_tokens = policy.replay_tokens(tokens_total);
        let mut units = Vec::new();
        let last_layer = model.num_layers.saturating_sub(1);
        for (stage_index, range) in placement.trunk_ranges.iter().enumerate() {
            let trunk = crate::ModelContentNames::trunk(
                &model.id,
                range.start_layer,
                range.end_layer,
                range.bytes,
            );
            let boundary = range.start_layer == 0 || range.end_layer == last_layer;
            let previous = if stage_index == 0 {
                Self::replay_input_ref(&graph_id)
            } else {
                Self::output_ref(
                    &graph_id,
                    &WorkUnitId::qualified(&graph_id, &format!("replay-{:02}", stage_index - 1)),
                )
            };
            let mut placement_constraints = if boundary {
                PlacementConstraints::trusted_only()
            } else {
                PlacementConstraints::default()
            };
            placement_constraints.requires_resident = vec![trunk.clone()];
            placement_constraints.preferred_nodes = vec![range.owner.clone()];
            let layers = (range.end_layer - range.start_layer + 1) as f64;
            let mut unit = WorkUnit::new(
                WorkUnitId::qualified(&graph_id, &format!("replay-{stage_index:02}")),
                AppKind::KvReplay,
            )
            .with_params_json(&KvReplayParams {
                model_id: model.id.clone(),
                epoch,
                session_id: session_id.to_string(),
                stage_index,
                start_layer: range.start_layer,
                end_layer: range.end_layer,
                suffix_tokens,
                tokens_total,
                page_bytes_modelled: KvPageNames::page_bytes(model, tokens_total),
                q_star_ratio: range.q_star_ratio,
                experts_per_layer: model.experts_per_layer,
                active_experts_per_token: model.active_experts_per_token,
            })
            .expect("params serialize")
            .with_placement(placement_constraints)
            .with_cost(
                CostEstimate::new(
                    2.0 * range.bytes as f64 * suffix_tokens as f64,
                    range.bytes,
                    (8 * 1024 * suffix_tokens) as u64,
                    0.5 * layers * suffix_tokens as f64,
                )
                .with_work_units(suffix_tokens as u64),
            )
            .with_redundancy(RedundancyPolicy::trusted());
            unit.inputs = vec![trunk, previous];
            units.push(unit);
        }
        WorkGraph::chain(graph_id, units)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionStrategy, ExpertPlacementPlanner};
    use kadmium_core::sample_personal_cluster;

    fn toy_moe() -> ModelSpec {
        ModelSpec {
            id: "toy-moe".into(),
            display_name: "Toy MoE".into(),
            total_params_billions: 4.0,
            num_layers: 4,
            is_moe: true,
            dense_layer_size_gb: 0.05,
            experts_per_layer: 8,
            active_experts_per_token: 2,
            single_expert_size_gb: 0.01,
            kv_cache_per_layer_gb: 0.01,
            training_activation_per_layer_gb: 0.02,
            lora_state_per_layer_gb: 0.01,
        }
    }

    #[test]
    fn pages_are_named_sized_and_parsed_and_idle_sessions_sink_then_come_back() {
        let model = toy_moe();
        let id = KvPageNames::page_id(&model.id, "sess-1", 3);
        assert_eq!(id, "toy-moe/kv/sess-1/L003");
        assert_eq!(
            KvPageNames::parse(&id),
            Some(("toy-moe".into(), "sess-1".into(), 3))
        );
        assert_eq!(KvPageNames::parse("toy-moe/L003/E004"), None);
        let bytes = KvPageNames::page_bytes(&model, KV_REFERENCE_CONTEXT_TOKENS);
        assert!(
            (bytes as i64 - (0.01 * GIB) as i64).abs() <= 1,
            "the whole reference window is the spec's figure: {bytes}"
        );
        assert!(KvPageNames::page_bytes(&model, 100) < bytes / 100);

        let policy = KvTierPolicy::default();
        assert_eq!(policy.replay_tokens(10_000), 140);
        assert_eq!(policy.replay_tokens(50), 50);
        assert_eq!(
            policy.tier_for_idle(0, true, ResidencyTier::Vram),
            ResidencyTier::Vram
        );
        assert_eq!(
            policy.tier_for_idle(60_000, true, ResidencyTier::Vram),
            ResidencyTier::LocalNvme
        );
        assert_eq!(
            policy.tier_for_idle(60_000, false, ResidencyTier::Vram),
            ResidencyTier::NasVault
        );
        assert_eq!(
            policy.tier_for_idle(600_000, true, ResidencyTier::Vram),
            ResidencyTier::NasVault
        );

        let nodes = sample_personal_cluster();
        let placement =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None);
        let mut catalog = ReplicaCatalog::new();
        let mut sessions = vec![
            KvSessionState::new("busy", &model.id, 1_000),
            KvSessionState::new("idle", &model.id, 1_000),
        ];
        sessions[0].tokens = 300;
        sessions[1].tokens = 300;
        for s in &sessions {
            let pages =
                KvTierPlanner::record_resident(&mut catalog, &model, &placement, &nodes, s, 1_000);
            assert_eq!(pages, model.num_layers as usize);
        }
        assert_eq!(catalog.content_count(), 2 * model.num_layers as usize);
        assert!(
            catalog
                .resident_on(&placement.trunk_ranges[0].owner)
                .iter()
                .all(|r| !KvTierPlanner::is_sunk(r.tier) && r.pinned)
        );

        // Nothing idle yet: no moves.
        let quiet = KvTierPlanner::tier(
            &mut catalog,
            &mut sessions,
            &model,
            &placement,
            &nodes,
            &policy,
            1_000,
        );
        assert!(quiet.moves.is_empty());
        assert!(quiet.modelled);

        // One session idles past the NVMe threshold.
        sessions[0].last_active_unix_ms = 100_000;
        let pass = KvTierPlanner::tier(
            &mut catalog,
            &mut sessions,
            &model,
            &placement,
            &nodes,
            &policy,
            100_000,
        );
        assert_eq!(pass.sunk, vec!["idle".to_string()]);
        assert_eq!(pass.moves.len(), model.num_layers as usize);
        assert!(
            pass.moves
                .iter()
                .all(|m| m.to == ResidencyTier::LocalNvme && m.from.is_fast())
        );
        assert!(sessions[1].needs_replay());
        assert!(!sessions[0].needs_replay());
        assert!(pass.bytes_by_tier.contains_key("LocalNvme"));
        let rules = KvTierPlanner::rules_for(&catalog, &model, &placement, &sessions[1]);
        assert!(
            rules
                .iter()
                .all(|r| r.preferred_tier == ResidencyTier::LocalNvme && !r.pin)
        );

        // Longer still: the vault, on the vault node; the busy session keeps decoding.
        let vault = nodes
            .iter()
            .find(|n| n.roles.contains(&NodeRole::StorageVault))
            .unwrap();
        sessions[0].last_active_unix_ms = 800_000;
        let later = KvTierPlanner::tier(
            &mut catalog,
            &mut sessions,
            &model,
            &placement,
            &nodes,
            &policy,
            800_000,
        );
        assert_eq!(later.sunk, vec!["idle".to_string()]);
        assert!(
            later
                .moves
                .iter()
                .all(|m| m.to == ResidencyTier::NasVault && m.node == vault.id)
        );
        assert_eq!(sessions[1].tier, ResidencyTier::NasVault);
        let jobs = KvTierPlanner::raise_jobs(&catalog, &model, &placement, &nodes, &sessions[1]);
        assert_eq!(jobs.len(), model.num_layers as usize);
        assert!(
            jobs.iter().all(|j| j.source_node == vault.id
                && j.pin
                && !KvTierPlanner::is_sunk(j.target_tier))
        );

        // Active again: the pages come back up and the session no longer needs a replay.
        sessions[1].last_active_unix_ms = 800_000;
        let back = KvTierPlanner::tier(
            &mut catalog,
            &mut sessions,
            &model,
            &placement,
            &nodes,
            &policy,
            800_001,
        );
        assert_eq!(back.raised, vec!["idle".to_string()]);
        assert!(!sessions[1].needs_replay());
        assert!(
            KvTierPlanner::raise_jobs(&catalog, &model, &placement, &nodes, &sessions[1])
                .is_empty()
        );
    }

    #[test]
    fn a_replay_graph_is_a_chain_over_the_trunk_ranges_sized_by_the_bounded_suffix() {
        let model = toy_moe();
        let nodes = sample_personal_cluster();
        let placement =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None);
        let policy = KvTierPolicy::default();
        let graph =
            StepGraphBuilder::kv_replay_graph(&model, &placement, "sess-9", 4, 5_000, &policy);
        assert_eq!(graph.id, "sess-9/replay-4");
        assert_eq!(graph.len(), placement.trunk_ranges.len());
        assert_eq!(
            graph.wavefronts().unwrap().len(),
            placement.trunk_ranges.len()
        );
        let first = graph.ready();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].app, AppKind::KvReplay);
        assert!(first[0].app.is_latency_critical());
        let params: KvReplayParams = first[0].params_json().unwrap();
        assert_eq!(params.suffix_tokens, 140);
        assert_eq!(params.tokens_total, 5_000);
        assert_eq!(params.start_layer, 0);
        assert!(first[0].placement.boundary_layer_access);
        assert_eq!(
            first[0].placement.preferred_nodes,
            vec![placement.trunk_ranges[0].owner.clone()]
        );
        assert!(
            first[0]
                .inputs
                .iter()
                .any(|c| c.content_id == "sess-9/replay-4/suffix")
        );
        assert_eq!(first[0].cost.work_units, 140);
        let short = StepGraphBuilder::kv_replay_graph(&model, &placement, "sess-9", 5, 20, &policy);
        let p: KvReplayParams = short.ready()[0].params_json().unwrap();
        assert_eq!(p.suffix_tokens, 20, "a short context replays all of it");
    }
}
