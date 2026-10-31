//! Learned usage and live re-pin: the planner's memory of what the mesh actually routes to.
//!
//! colibri keeps a usage file beside the model, updated every turn, and seeds its pinned hot set
//! from it at start-up; under its balanced policy it then re-pins at safe boundaries from a
//! decaying session heat map, with hysteresis and a cap on swaps per boundary, so placement
//! follows what the last few turns routed to without churning. The same two signals live here:
//! a [`UsageLedger`] per model with a long-term count that is never decayed and a session count
//! that halves every half-life, persisted as a sidecar; and a [`RepinPlanner`] that proposes at
//! most a few swaps between the hottest streamed experts and the coldest pinned ones when the
//! heat difference clears the hysteresis. Placement changes speed, never bytes: the reference
//! math is the same wherever an expert lives.

use kadmium_core::{HardwareNode, ModelSpec, NodeId};
use kadmium_grid::{ContentAccess, ResidencyTier, StagingJob};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::{
    BalanceMode, BalanceReport, ExpertLoadBalancer, ExpertPlacementPlan, ExpertPopularity,
    LinkSegments, ModelContentNames, ReplicaBudget, ReplicaChange,
};

/// A session hit counts this many long-term hits while it lasts.
pub const SESSION_HEAT_WEIGHT: f64 = 3.0;
/// Session heat halves this often.
pub const DEFAULT_SESSION_HALF_LIFE_MS: u64 = 10 * 60 * 1000;

/// One expert's heat, for summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotExpert {
    pub layer: u32,
    pub expert: u32,
    pub hits: u64,
}

/// What the ledger knows, for a status line or an API response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageSummary {
    pub model_id: String,
    pub turns: u64,
    pub routed_hits: u64,
    pub experts_seen: usize,
    pub session_hits: u64,
    pub hottest: Vec<HotExpert>,
    /// False until any routing has been recorded; then the planner's popularity comes from here.
    pub learned: bool,
}

/// Routing heat learned from live use, kept beside the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageLedger {
    pub model_id: String,
    /// Every routed hit since the ledger was created; never decayed.
    pub long_term: ExpertPopularity,
    /// Recent hits, halved every half-life.
    pub session: ExpertPopularity,
    pub session_half_life_ms: u64,
    pub turns: u64,
    pub routed_hits: u64,
    pub last_decay_unix_ms: u64,
    pub updated_unix_ms: u64,
}

impl UsageLedger {
    pub fn new(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            long_term: ExpertPopularity::default(),
            session: ExpertPopularity::default(),
            session_half_life_ms: DEFAULT_SESSION_HALF_LIFE_MS,
            turns: 0,
            routed_hits: 0,
            last_decay_unix_ms: 0,
            updated_unix_ms: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.routed_hits == 0
    }

    /// Records one content access when it names an expert of this model; other content is not
    /// the ledger's business.
    pub fn record_content(&mut self, content_id: &str, hits: u64, now: u64) -> bool {
        let Some((model, layer, expert)) = ModelContentNames::parse_expert(content_id) else {
            return false;
        };
        if model != self.model_id || hits == 0 {
            return false;
        }
        self.long_term.record(layer, expert, hits);
        self.session.record(layer, expert, hits);
        self.routed_hits += hits;
        self.updated_unix_ms = self.updated_unix_ms.max(now);
        true
    }

    /// Records the accesses of one turn (one decode step's validated results), decaying the
    /// session heat first. Returns how many accesses named experts of this model.
    pub fn record_turn<'a>(
        &mut self,
        accesses: impl IntoIterator<Item = &'a ContentAccess>,
        now: u64,
    ) -> usize {
        self.decay(now);
        let mut recorded = 0;
        for access in accesses {
            if self.record_content(&access.content_id, access.hits, now) {
                recorded += 1;
            }
        }
        if recorded > 0 {
            self.turns += 1;
        }
        recorded
    }

    /// Halves the session heat once per elapsed half-life.
    pub fn decay(&mut self, now: u64) {
        if self.last_decay_unix_ms == 0 {
            self.last_decay_unix_ms = now;
            return;
        }
        let half_life = self.session_half_life_ms.max(1);
        let elapsed = now.saturating_sub(self.last_decay_unix_ms);
        let halvings = (elapsed / half_life).min(64);
        for _ in 0..halvings {
            self.session.halve();
        }
        if halvings > 0 {
            self.last_decay_unix_ms += halvings * half_life;
        }
    }

    /// The planner's popularity: long-term hits plus the weighted session heat.
    pub fn popularity(&self) -> ExpertPopularity {
        self.long_term
            .merged_with(&self.session, SESSION_HEAT_WEIGHT)
    }

    pub fn hottest(&self, n: usize) -> Vec<HotExpert> {
        self.popularity()
            .hottest(n)
            .into_iter()
            .map(|(layer, expert, hits)| HotExpert {
                layer,
                expert,
                hits,
            })
            .collect()
    }

    pub fn summary(&self) -> UsageSummary {
        UsageSummary {
            model_id: self.model_id.clone(),
            turns: self.turns,
            routed_hits: self.routed_hits,
            experts_seen: self.long_term.iter().count(),
            session_hits: self.session.total_hits(),
            hottest: self.hottest(8),
            learned: !self.is_empty(),
        }
    }

    /// Whether this ledger belongs to the model.
    pub fn sidecar_matches(&self, model_id: &str) -> bool {
        self.model_id == model_id
    }

    /// Where the ledger lives beside a model: `<dir>/<model id>.usage.json`, with the id made
    /// safe for a file name.
    pub fn sidecar_path(dir: &Path, model_id: &str) -> PathBuf {
        let safe: String = model_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        dir.join(format!("{safe}.usage.json"))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let tmp = path.with_extension(format!("{}.{nanos}.tmp", std::process::id()));
        std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Option<Self>, String> {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json)
                .map(Some)
                .map_err(|e| e.to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    /// The saved ledger for a model, or a new one.
    pub fn load_or_new(dir: &Path, model_id: &str) -> Self {
        Self::load(&Self::sidecar_path(dir, model_id))
            .ok()
            .flatten()
            .filter(|l| l.model_id == model_id)
            .unwrap_or_else(|| Self::new(model_id))
    }
}

/// When and how much the live re-pin may move.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepinPolicy {
    /// Swaps per boundary at most.
    pub max_swaps: usize,
    /// A streamed expert must be this much hotter than the pinned one it would replace.
    pub hysteresis: f64,
    /// Heat a streamed expert needs before it is considered at all.
    pub min_hits: u64,
    /// When set, a swap must also pay for its transfers: the hits the promoted expert should
    /// earn beyond the demoted one, times what a fast-tier hit saves over a streamed read, must
    /// cover the two transfers the swap makes (InferSSD's adaptive VRAM admission, where an
    /// expert is admitted only if its expected extra hits pay back its upload).
    #[serde(default)]
    pub payback: Option<RepinPayback>,
}

impl Default for RepinPolicy {
    fn default() -> Self {
        Self {
            max_swaps: 4,
            hysteresis: 0.25,
            min_hits: 1,
            payback: None,
        }
    }
}

/// What a swap costs and what a hit on the promoted expert saves, in the units the re-pin
/// planner compares: `admission_hits = transfer_ms / (miss_ms - hit_ms)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RepinPayback {
    pub expert_bytes: u64,
    /// Read bandwidth of the fast tier an expert is promoted into, in GB/s.
    pub fast_gbps: f64,
    /// Read bandwidth of the tier it is streamed from today, in GB/s.
    pub streamed_gbps: f64,
    /// The link the promotion and the demotion cross, in GB/s.
    pub link_gbps: f64,
}

impl RepinPayback {
    /// A payback for a mesh: the median fast-tier and system-RAM bandwidths of the nodes, and
    /// the slowest uplink among them as the link both transfers cross.
    pub fn for_mesh(model: &ModelSpec, nodes: &[HardwareNode]) -> Self {
        fn median(mut values: Vec<f64>, fallback: f64) -> f64 {
            values.retain(|v| v.is_finite() && *v > 0.0);
            if values.is_empty() {
                return fallback;
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values[values.len() / 2]
        }
        let fast = median(
            nodes
                .iter()
                .map(|n| {
                    if n.memory.unified_mem_gb > 0.0 {
                        n.memory.unified_mem_bw_gbps
                    } else {
                        n.memory.vram_bw_gbps
                    }
                })
                .collect(),
            ResidencyTier::Vram.nominal_bandwidth_gbps(),
        );
        let streamed = median(
            nodes.iter().map(|n| n.memory.system_ram_bw_gbps).collect(),
            ResidencyTier::SystemRam.nominal_bandwidth_gbps(),
        );
        let link = nodes
            .iter()
            .map(|n| n.network_uplink_gbps / 8.0)
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        Self {
            expert_bytes: (model.single_expert_size_gb * 1_073_741_824.0) as u64,
            fast_gbps: fast,
            streamed_gbps: streamed,
            link_gbps: if link.is_finite() { link } else { 0.125 },
        }
    }

    fn ms_at(&self, gbps: f64) -> f64 {
        self.expert_bytes as f64 / (gbps.max(1e-6) * 1e9) * 1000.0
    }

    /// A read of the expert from the fast tier.
    pub fn hit_ms(&self) -> f64 {
        self.ms_at(self.fast_gbps)
    }

    /// A read of the expert from where it is streamed today.
    pub fn miss_ms(&self) -> f64 {
        self.ms_at(self.streamed_gbps)
    }

    /// The promotion and the demotion, both across the link.
    pub fn transfer_ms(&self) -> f64 {
        self.transfer_ms_for(false)
    }

    /// The two transfers of a swap: across the link between two nodes, or two copies through
    /// the streamed tier's memory when the promoted and demoted experts share a node.
    pub fn transfer_ms_for(&self, same_node: bool) -> f64 {
        if same_node {
            2.0 * self.ms_at(self.streamed_gbps)
        } else {
            2.0 * self.ms_at(self.link_gbps)
        }
    }

    /// Hits the promoted expert must earn beyond the demoted one to repay the transfers across
    /// the link; infinite when a fast hit saves nothing.
    pub fn admission_hits(&self) -> f64 {
        self.admission_hits_for(false)
    }

    /// [`RepinPayback::admission_hits`] for a swap on one node or across the link.
    pub fn admission_hits_for(&self, same_node: bool) -> f64 {
        let saved = self.miss_ms() - self.hit_ms();
        if saved <= 0.0 {
            f64::INFINITY
        } else {
            self.transfer_ms_for(same_node) / saved
        }
    }
}

/// One proposed swap: a hot streamed expert takes the fast-tier place of a cold pinned one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepinSwap {
    pub layer: u32,
    pub promote: u32,
    pub promote_hits: u64,
    pub demote: u32,
    pub demote_hits: u64,
    /// The node whose fast tier changes hands.
    pub node: NodeId,
    pub fast_tier: ResidencyTier,
    /// Where the promoted expert came from, and the tier the demoted one goes to.
    pub from_node: NodeId,
    pub from_tier: ResidencyTier,
    /// Extra hits the swap had to earn to pay for its transfers, 0 without a payback rule.
    #[serde(default)]
    pub required_hits: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RepinProposal {
    pub swaps: Vec<RepinSwap>,
    pub layers_considered: usize,
    pub rationale: String,
    /// Copies the load balancer adds or drops at this boundary (its incremental mode), each
    /// one change against the same cap as the swaps.
    #[serde(default)]
    pub replica_changes: Vec<ReplicaChange>,
    /// The balancer's account of the boundary: the busiest node per layer before and after.
    #[serde(default)]
    pub balance: Option<BalanceReport>,
}

impl RepinProposal {
    /// Swaps plus copies added or dropped.
    pub fn change_count(&self) -> usize {
        self.swaps.len() + self.replica_changes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.change_count() == 0
    }
}

pub struct RepinPlanner;

impl RepinPlanner {
    /// One re-pin boundary with the load balancer in its incremental mode: first the tier
    /// swaps of [`RepinPlanner::propose`] (a hot streamed expert takes a cold pinned one's fast
    /// slot on the same node), then, within whatever of `policy.max_swaps` is left, copies
    /// added where they lower a layer's busiest node by more than the same hysteresis, packed
    /// across `segments` first. Both kinds of change are bounded by the one cap.
    pub fn propose_balanced(
        placement: &ExpertPlacementPlan,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        heat: &ExpertPopularity,
        segments: &LinkSegments,
        policy: &RepinPolicy,
    ) -> RepinProposal {
        let mut proposal = Self::propose(placement, heat, policy);
        let remaining = policy.max_swaps.saturating_sub(proposal.swaps.len());
        if remaining == 0 || placement.strategy != crate::ExecutionStrategy::ExpertParallel {
            return proposal;
        }
        let mut trial = placement.clone();
        let budget = ReplicaBudget::from_plan(&trial, model);
        let report = ExpertLoadBalancer::balance(
            &mut trial,
            model,
            nodes,
            heat,
            &budget,
            segments,
            true,
            BalanceMode::Incremental {
                max_changes: remaining,
                hysteresis: policy.hysteresis,
            },
        );
        proposal.replica_changes = report.changes.clone();
        if !report.changes.is_empty() {
            proposal.rationale = format!("{}; {}", proposal.rationale, report.rationale);
        }
        proposal.balance = Some(report);
        proposal
    }
    /// Proposes at most `policy.max_swaps` swaps: per layer, the hottest streamed expert against
    /// the coldest pinned one on a node with a fast tier, taken only when the streamed expert is
    /// hotter by more than the hysteresis. Experts that already have a fast replica are never
    /// promoted again.
    pub fn propose(
        placement: &ExpertPlacementPlan,
        heat: &ExpertPopularity,
        policy: &RepinPolicy,
    ) -> RepinProposal {
        let mut layers: Vec<u32> = placement.experts.iter().map(|h| h.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        let mut candidates: Vec<(u64, RepinSwap)> = Vec::new();
        let mut unpaid = 0usize;

        for layer in &layers {
            let homes: Vec<_> = placement
                .experts
                .iter()
                .filter(|h| h.layer == *layer && !h.holders.is_empty())
                .collect();
            let hottest_streamed = homes
                .iter()
                .filter(|h| !h.tier.is_fast() && h.holders.len() == 1)
                .map(|h| (heat.hits(h.layer, h.expert), *h))
                .filter(|(hits, _)| *hits >= policy.min_hits)
                .max_by_key(|(hits, h)| (*hits, std::cmp::Reverse(h.expert)));
            let coldest_pinned = homes
                .iter()
                .filter(|h| h.tier.is_fast())
                .map(|h| (heat.hits(h.layer, h.expert), *h))
                .min_by_key(|(hits, h)| (*hits, h.expert));
            let (Some((hot_hits, hot)), Some((cold_hits, cold))) =
                (hottest_streamed, coldest_pinned)
            else {
                continue;
            };
            let threshold = cold_hits as f64 * (1.0 + policy.hysteresis.max(0.0));
            if hot_hits <= cold_hits || (hot_hits as f64) <= threshold {
                continue;
            }
            let same_node = hot.holders[0] == cold.holders[0];
            let required_hits = policy
                .payback
                .map_or(0.0, |p| p.admission_hits_for(same_node));
            if ((hot_hits - cold_hits) as f64) < required_hits {
                unpaid += 1;
                continue;
            }
            candidates.push((
                hot_hits - cold_hits,
                RepinSwap {
                    layer: *layer,
                    promote: hot.expert,
                    promote_hits: hot_hits,
                    demote: cold.expert,
                    demote_hits: cold_hits,
                    node: cold.holders[0].clone(),
                    fast_tier: cold.tier,
                    from_node: hot.holders[0].clone(),
                    from_tier: hot.tier,
                    required_hits,
                },
            ));
        }

        candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.layer.cmp(&b.1.layer)));
        let swaps: Vec<RepinSwap> = candidates
            .into_iter()
            .take(policy.max_swaps)
            .map(|(_, s)| s)
            .collect();
        let mut rationale = if swaps.is_empty() {
            format!(
                "no streamed expert is more than {:.0}% hotter than the coldest pinned expert of its layer",
                policy.hysteresis * 100.0
            )
        } else {
            format!(
                "{} swap{} (at most {} per boundary): streamed experts hotter than the coldest pinned expert of their layer by more than {:.0}%",
                swaps.len(),
                if swaps.len() == 1 { "" } else { "s" },
                policy.max_swaps,
                policy.hysteresis * 100.0
            )
        };
        if let Some(payback) = policy.payback
            && unpaid > 0
        {
            rationale = format!(
                "{rationale}; {unpaid} hotter expert{} left streamed because their extra hits would not repay the two transfers (a swap needs {:.1} extra hits across the link at {:.2} ms, {:.1} on one node at {:.2} ms)",
                if unpaid == 1 { "" } else { "s" },
                payback.admission_hits_for(false),
                payback.transfer_ms_for(false),
                payback.admission_hits_for(true),
                payback.transfer_ms_for(true)
            );
        }
        RepinProposal {
            swaps,
            layers_considered: layers.len(),
            rationale,
            replica_changes: Vec::new(),
            balance: None,
        }
    }

    /// Applies a proposal to the placement and returns the transfers it implies: the promoted
    /// expert into the node's fast tier, pinned, and the demoted expert out to where the promoted
    /// one came from (fast memory accounting is unchanged, because the two experts are the same
    /// size and trade places); then every copy the balancer added, from the expert's primary
    /// into the node's fast tier, pinned, with dropped copies taken off their holders.
    pub fn apply(
        placement: &mut ExpertPlacementPlan,
        proposal: &RepinProposal,
        expert_bytes: u64,
    ) -> Vec<StagingJob> {
        let mut jobs = Self::apply_swaps(placement, proposal, expert_bytes);
        if proposal.replica_changes.is_empty() {
            return jobs;
        }
        jobs.extend(ExpertLoadBalancer::staging_for(
            placement,
            &proposal.replica_changes,
            expert_bytes,
        ));
        let gib = expert_bytes as f64 / 1_073_741_824.0;
        for change in &proposal.replica_changes {
            let Some(home) = placement
                .experts
                .iter_mut()
                .find(|h| h.layer == change.layer && h.expert == change.expert)
            else {
                continue;
            };
            if change.add {
                if !home.holders.contains(&change.node) {
                    home.holders.push(change.node.clone());
                }
            } else {
                home.holders.retain(|n| n != &change.node);
            }
            let content_id =
                ModelContentNames::expert_id(&placement.model_id, change.layer, change.expert);
            if let Some(rule) = placement
                .rules
                .iter_mut()
                .find(|r| r.content.content_id == content_id)
            {
                if change.add {
                    if !rule.candidate_nodes.contains(&change.node) {
                        rule.candidate_nodes.push(change.node.clone());
                    }
                } else {
                    rule.candidate_nodes.retain(|n| n != &change.node);
                }
                rule.min_replicas = rule.candidate_nodes.len().max(1) as u32;
            }
            if let Some(fit) = placement
                .fit
                .per_node
                .iter_mut()
                .find(|f| f.node == change.node)
            {
                if change.add {
                    fit.fast_used_gb = ((fit.fast_used_gb + gib) * 100.0).round() / 100.0;
                    fit.fast_free_bytes = fit.fast_free_bytes.saturating_sub(expert_bytes);
                } else {
                    fit.fast_used_gb = ((fit.fast_used_gb - gib).max(0.0) * 100.0).round() / 100.0;
                    fit.fast_free_bytes = fit.fast_free_bytes.saturating_add(expert_bytes);
                }
            }
        }
        placement.replicated_experts = placement
            .experts
            .iter()
            .map(|h| h.holders.len().saturating_sub(1))
            .sum();
        jobs
    }

    fn apply_swaps(
        placement: &mut ExpertPlacementPlan,
        proposal: &RepinProposal,
        expert_bytes: u64,
    ) -> Vec<StagingJob> {
        let mut jobs = Vec::new();
        for swap in &proposal.swaps {
            let Some(hot_idx) = placement
                .experts
                .iter()
                .position(|h| h.layer == swap.layer && h.expert == swap.promote)
            else {
                continue;
            };
            let Some(cold_idx) = placement
                .experts
                .iter()
                .position(|h| h.layer == swap.layer && h.expert == swap.demote)
            else {
                continue;
            };
            placement.experts[hot_idx].holders = vec![swap.node.clone()];
            placement.experts[hot_idx].tier = swap.fast_tier;
            placement.experts[cold_idx].holders = vec![swap.from_node.clone()];
            placement.experts[cold_idx].tier = swap.from_tier;

            let hot_ref = ModelContentNames::expert(
                &placement.model_id,
                swap.layer,
                swap.promote,
                expert_bytes,
            );
            let cold_ref = ModelContentNames::expert(
                &placement.model_id,
                swap.layer,
                swap.demote,
                expert_bytes,
            );
            for rule in placement.rules.iter_mut() {
                if rule.content.content_id == hot_ref.content_id {
                    rule.candidate_nodes = vec![swap.node.clone()];
                    rule.min_replicas = 1;
                    rule.preferred_tier = swap.fast_tier;
                    rule.pin = true;
                } else if rule.content.content_id == cold_ref.content_id {
                    rule.candidate_nodes = vec![swap.from_node.clone()];
                    rule.min_replicas = 1;
                    rule.preferred_tier = swap.from_tier;
                    rule.pin = false;
                }
            }
            jobs.push(StagingJob {
                content: hot_ref,
                source_node: swap.from_node.clone(),
                destination_node: swap.node.clone(),
                target_tier: swap.fast_tier,
                bytes: expert_bytes,
                pin: true,
            });
            jobs.push(StagingJob {
                content: cold_ref,
                source_node: swap.node.clone(),
                destination_node: swap.from_node.clone(),
                target_tier: swap.from_tier,
                bytes: expert_bytes,
                pin: false,
            });
        }
        jobs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionStrategy, ExpertPlacementPlanner};
    use kadmium_core::{ModelSpec, ResourceOffer, WorkloadKind, sample_personal_cluster};
    use std::collections::BTreeMap;

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
    fn ledger_records_this_models_experts_decays_the_session_and_roundtrips() {
        let mut ledger = UsageLedger::new("toy-moe");
        assert!(ledger.is_empty());
        let accesses = vec![
            ContentAccess {
                content_id: ModelContentNames::expert_id("toy-moe", 2, 5),
                hits: 3,
            },
            ContentAccess {
                content_id: ModelContentNames::expert_id("toy-moe", 2, 1),
                hits: 1,
            },
            ContentAccess {
                content_id: ModelContentNames::expert_id("other-model", 2, 1),
                hits: 9,
            },
            ContentAccess {
                content_id: "graph/stage-00/out".into(),
                hits: 1,
            },
        ];
        assert_eq!(ledger.record_turn(&accesses, 1_000), 2);
        assert_eq!(ledger.turns, 1);
        assert_eq!(ledger.routed_hits, 4);
        assert_eq!(ledger.long_term.hits(2, 5), 3);
        assert_eq!(ledger.session.hits(2, 5), 3);
        let summary = ledger.summary();
        assert!(summary.learned);
        assert_eq!(summary.hottest[0].expert, 5);

        // Session heat halves per half-life; the long-term count never moves.
        ledger.decay(1_000 + 2 * ledger.session_half_life_ms);
        assert_eq!(ledger.session.hits(2, 5), 0);
        assert_eq!(ledger.session.hits(2, 1), 0);
        assert_eq!(ledger.long_term.hits(2, 5), 3);
        let pop = ledger.popularity();
        assert_eq!(pop.ranked(2, 8)[0], 5);

        let dir = std::env::temp_dir().join(format!("kadmium-usage-{}", std::process::id()));
        let path = UsageLedger::sidecar_path(&dir, "toy-moe");
        assert!(path.ends_with("toy-moe.usage.json"));
        ledger.save(&path).unwrap();
        assert_eq!(UsageLedger::load(&path).unwrap(), Some(ledger.clone()));
        assert_eq!(UsageLedger::load_or_new(&dir, "toy-moe"), ledger);
        assert!(UsageLedger::load_or_new(&dir, "unknown").is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn repin_swaps_hot_streamed_experts_for_cold_pinned_ones_under_hysteresis_and_a_cap() {
        let nodes = sample_personal_cluster();
        let model = toy_moe();
        // Starve fast memory so that only a few experts per layer are pinned; the rest sit warm.
        let offers: BTreeMap<_, ResourceOffer> = nodes
            .iter()
            .map(|n| {
                let mut offer = n.default_offer();
                offer.fast_memory_gb = 0.0;
                (n.id.clone(), offer)
            })
            .collect();
        let mut placement = ExpertPlacementPlanner::plan_with_offers(
            &model,
            &nodes,
            ExecutionStrategy::PipelineChain,
            None,
            WorkloadKind::Inference,
            &offers,
        );
        // Pin the first two experts of every layer by hand, as a loaded plan would have.
        for home in placement.experts.iter_mut() {
            if home.expert < 2 {
                home.tier = ResidencyTier::Vram;
            } else {
                home.tier = ResidencyTier::SystemRam;
            }
        }
        let expert_bytes = (model.single_expert_size_gb * 1_073_741_824.0) as u64;

        let mut heat = ExpertPopularity::default();
        for layer in 0..4 {
            heat.record(layer, 0, 10);
            heat.record(layer, 1, 10);
        }
        // Layer 0: expert 7 is far hotter than the pinned ones; layer 1: expert 6 barely hotter;
        // layer 2: expert 5 within the hysteresis; layer 3: nothing streamed is hot.
        heat.record(0, 7, 40);
        heat.record(1, 6, 13);
        heat.record(2, 5, 12);

        let policy = RepinPolicy::default();
        let proposal = RepinPlanner::propose(&placement, &heat, &policy);
        assert_eq!(proposal.layers_considered, 4);
        assert_eq!(proposal.swaps.len(), 2, "{:?}", proposal.swaps);
        assert_eq!(
            proposal.swaps[0].layer, 0,
            "the largest heat gap comes first"
        );
        assert_eq!(proposal.swaps[0].promote, 7);
        assert!(proposal.swaps[0].demote < 2);
        assert_eq!(proposal.swaps[1].layer, 1);
        assert_eq!(proposal.swaps[1].promote, 6);
        assert!(proposal.rationale.contains("25%"));

        let capped = RepinPlanner::propose(
            &placement,
            &heat,
            &RepinPolicy {
                max_swaps: 1,
                ..RepinPolicy::default()
            },
        );
        assert_eq!(capped.swaps.len(), 1);
        assert_eq!(capped.swaps[0].layer, 0);

        let digest_before = placement.digest();
        let jobs = RepinPlanner::apply(&mut placement, &proposal, expert_bytes);
        assert_eq!(jobs.len(), 4, "a promotion and a demotion per swap");
        assert!(jobs[0].pin && !jobs[1].pin);
        assert_eq!(jobs[0].target_tier, ResidencyTier::Vram);
        let promoted = placement
            .experts
            .iter()
            .find(|h| h.layer == 0 && h.expert == 7)
            .unwrap();
        assert_eq!(promoted.tier, ResidencyTier::Vram);
        let demoted = placement
            .experts
            .iter()
            .find(|h| h.layer == 0 && h.expert == proposal.swaps[0].demote)
            .unwrap();
        assert_eq!(demoted.tier, ResidencyTier::SystemRam);
        let rule = placement
            .rules
            .iter()
            .find(|r| r.content.content_id == ModelContentNames::expert_id("toy-moe", 0, 7))
            .unwrap();
        assert!(rule.pin);
        assert_eq!(rule.preferred_tier, ResidencyTier::Vram);
        assert_eq!(
            placement.digest(),
            digest_before,
            "a tier swap on the same node keeps the placement's identity"
        );

        // Nothing left to swap at this heat.
        let again = RepinPlanner::propose(&placement, &heat, &policy);
        assert!(again.swaps.is_empty());
    }

    #[test]
    fn payback_keeps_a_swap_whose_extra_hits_would_not_repay_its_transfers() {
        let nodes = sample_personal_cluster();
        let model = toy_moe();
        let offers: BTreeMap<_, ResourceOffer> = nodes
            .iter()
            .map(|n| {
                let mut offer = n.default_offer();
                offer.fast_memory_gb = 0.0;
                (n.id.clone(), offer)
            })
            .collect();
        let mut placement = ExpertPlacementPlanner::plan_with_offers(
            &model,
            &nodes,
            ExecutionStrategy::PipelineChain,
            None,
            WorkloadKind::Inference,
            &offers,
        );
        for home in placement.experts.iter_mut() {
            home.tier = if home.expert < 2 {
                ResidencyTier::Vram
            } else {
                ResidencyTier::SystemRam
            };
        }
        let mut heat = ExpertPopularity::default();
        for layer in 0..4 {
            heat.record(layer, 0, 10);
            heat.record(layer, 1, 10);
        }
        heat.record(0, 7, 40); // 30 extra hits
        heat.record(1, 6, 13); // 3 extra hits

        // Across a slow link, 10 MiB experts over 1 Gb/s (0.125 GB/s) cost 168 ms for the two
        // transfers; a fast hit saves 10 MiB at 60 GB/s against 800 GB/s, about 0.16 ms, so a
        // cross-node swap would need about a thousand extra hits. The chain placement swaps
        // within one node, where the two transfers are memory copies at the streamed rate
        // (0.35 ms) and a couple of extra hits pay.
        let payback = RepinPayback {
            expert_bytes: (model.single_expert_size_gb * 1_073_741_824.0) as u64,
            fast_gbps: 800.0,
            streamed_gbps: 60.0,
            link_gbps: 0.125,
        };
        assert!(
            payback.admission_hits() > 500.0,
            "{}",
            payback.admission_hits()
        );
        let on_node = payback.transfer_ms_for(true);
        assert!((on_node - 2.0 * payback.miss_ms()).abs() < 1e-9);
        assert!(payback.admission_hits_for(true) < 5.0);
        assert!(payback.admission_hits_for(true) < payback.admission_hits_for(false));

        // When a fast hit saves nothing over the streamed read, no swap can ever repay.
        let strict = RepinPolicy {
            payback: Some(RepinPayback {
                fast_gbps: 60.0,
                ..payback
            }),
            ..RepinPolicy::default()
        };
        let none = RepinPlanner::propose(&placement, &heat, &strict);
        assert!(none.swaps.is_empty(), "{:?}", none.swaps);
        assert!(
            none.rationale.contains("would not repay"),
            "{}",
            none.rationale
        );

        // A slow streamed tier: a hit saves 10 MiB at 5 GB/s (NVMe) against 800 GB/s, about
        // 2 ms, and the two on-node copies cost 4.2 ms, so two extra hits pay. Both the
        // 30-hit and the 3-hit swap qualify.
        let cheap = RepinPolicy {
            payback: Some(RepinPayback {
                streamed_gbps: 5.0,
                link_gbps: 40.0,
                ..payback
            }),
            ..RepinPolicy::default()
        };
        let both = RepinPlanner::propose(&placement, &heat, &cheap);
        assert_eq!(both.swaps.len(), 2, "{:?}", both.swaps);
        assert!(both.swaps[0].required_hits > 1.0 && both.swaps[0].required_hits < 3.0);

        // A mesh-derived payback is finite and positive.
        let mesh = RepinPayback::for_mesh(&model, &nodes);
        assert!(mesh.expert_bytes > 0 && mesh.link_gbps > 0.0);
        assert!(mesh.admission_hits().is_finite());
        let _ = RepinPlanner::propose(
            &placement,
            &heat,
            &RepinPolicy {
                payback: Some(mesh),
                ..RepinPolicy::default()
            },
        );
    }

    #[test]
    fn a_balanced_boundary_adds_copies_under_the_same_cap_and_applies_them() {
        let nodes = sample_personal_cluster();
        let model = toy_moe();
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 1.0);
        let mut placement =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        assert_eq!(placement.replicated_experts, 0);
        let segments = LinkSegments::derive(&nodes);
        let policy = RepinPolicy::default();
        let proposal =
            RepinPlanner::propose_balanced(&placement, &model, &nodes, &heat, &segments, &policy);
        assert!(proposal.change_count() <= policy.max_swaps);
        assert!(
            !proposal.replica_changes.is_empty(),
            "{}",
            proposal.rationale
        );
        assert!(proposal.replica_changes.iter().all(|c| c.add));
        let balance = proposal.balance.as_ref().unwrap();
        assert!(matches!(balance.mode, BalanceMode::Incremental { .. }));
        assert!(balance.busiest_bytes_after < balance.busiest_bytes_before);

        let expert_bytes = (model.single_expert_size_gb * 1_073_741_824.0) as u64;
        let digest_before = placement.digest();
        let jobs = RepinPlanner::apply(&mut placement, &proposal, expert_bytes);
        assert_eq!(jobs.len(), proposal.replica_changes.len());
        assert!(jobs.iter().all(|j| j.pin && j.target_tier.is_fast()));
        assert_eq!(placement.replicated_experts, proposal.replica_changes.len());
        assert_ne!(
            placement.digest(),
            digest_before,
            "a copy on another node is a new placement"
        );
        for change in &proposal.replica_changes {
            assert!(
                placement
                    .holders_of(change.layer, change.expert)
                    .contains(&change.node)
            );
            let rule = placement
                .rules
                .iter()
                .find(|r| {
                    r.content.content_id
                        == ModelContentNames::expert_id(&model.id, change.layer, change.expert)
                })
                .unwrap();
            assert!(rule.candidate_nodes.contains(&change.node));
        }

        // The chain shape has no copies to balance: the boundary is the swaps alone.
        let chain =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None);
        let chain_proposal =
            RepinPlanner::propose_balanced(&chain, &model, &nodes, &heat, &segments, &policy);
        assert!(chain_proposal.replica_changes.is_empty());
        assert!(chain_proposal.balance.is_none());
    }
}
