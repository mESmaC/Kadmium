//! AI on the grid substrate, planner half.
//!
//! Three questions, answered with `kadmium-grid` types so the generic scheduler can act on them:
//!
//! 1. **Where should a model's weights live?** [`ExpertPlacementPlanner`] turns a [`ModelSpec`]
//!    and the mesh into [`PlacementRule`]s: dense trunk ranges on their owners, every routed
//!    expert on a node chosen by remaining capacity and bandwidth, hot experts replicated when
//!    there is room. Two shapes are supported: [`ExecutionStrategy::PipelineChain`], where a
//!    layer's experts live with the layer's trunk owner, and
//!    [`ExecutionStrategy::ExpertParallel`], where experts spread across the whole mesh so its
//!    pooled fast memory holds a model no single node can.
//! 2. **Which shape is faster for this model on this mesh?** [`StrategyCostModel`] estimates the
//!    bandwidth-bound time per token of both shapes, counting one round trip per MoE layer for
//!    the fan-out and one per stage boundary for the chain, and picks the lower. Every figure it
//!    produces is modelled, not measured.
//! 3. **What work does one decode step consist of?** [`StepGraphBuilder`] emits the
//!    [`WorkGraph`] for a step: a chain of stage units, or per layer a trunk unit, one expert
//!    slice per holder, and a combine. Token streaming is therefore one schedule of the same
//!    units that expert parallelism schedules differently.

use kadmium_core::{
    ComputeBackend, HardwareNode, ModelSpec, NodeId, NodeRole, NodeTrustTier, ResourceOffer,
    WorkloadKind,
};
use kadmium_grid::{
    AppKind, ContentKind, ContentRef, CostEstimate, PlacementConstraints, PlacementRule,
    RedundancyPolicy, ResidencyTier, WorkGraph, WorkUnit, WorkUnitId, fnv1a_64,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    BalanceMode, BalanceReport, CalibrationRecord, DeepEpStepAccount, ExpertLoadBalancer,
    LinkSegments, ReplicaBudget, ShardPlanner,
};

const GIB: f64 = 1_073_741_824.0;

/// Stable names for model content on the grid.
pub struct ModelContentNames;

impl ModelContentNames {
    pub fn trunk_id(model_id: &str, start_layer: u32, end_layer: u32) -> String {
        format!("{model_id}/trunk/L{start_layer:03}-L{end_layer:03}")
    }

    pub fn trunk(model_id: &str, start_layer: u32, end_layer: u32, bytes: u64) -> ContentRef {
        ContentRef::named(
            ContentKind::KtbWeightBank,
            &Self::trunk_id(model_id, start_layer, end_layer),
            bytes,
        )
    }

    /// The `(model, layer, expert)` an expert content id names, if it is one.
    pub fn parse_expert(content_id: &str) -> Option<(String, u32, u32)> {
        let (rest, expert) = content_id.rsplit_once("/E")?;
        let (model, layer) = rest.rsplit_once("/L")?;
        if model.is_empty() || expert.len() < 3 || layer.len() < 3 {
            return None;
        }
        Some((model.to_string(), layer.parse().ok()?, expert.parse().ok()?))
    }

    pub fn expert_id(model_id: &str, layer: u32, expert: u32) -> String {
        format!("{model_id}/L{layer:03}/E{expert:03}")
    }

    pub fn expert(model_id: &str, layer: u32, expert: u32, bytes: u64) -> ContentRef {
        ContentRef::named(
            ContentKind::ExpertBank,
            &Self::expert_id(model_id, layer, expert),
            bytes,
        )
    }
}

/// The shape of a decode step across the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Each node holds whole layers with their experts; one unit per stage; a hop only at stage
    /// boundaries. Today's pipeline.
    PipelineChain,
    /// Experts spread across the mesh by capacity and bandwidth; per MoE layer the trunk owner
    /// fans the hidden rows out to the holders of the selected experts and combines their partial
    /// sums. Nodes compute at the same time on the same step.
    ExpertParallel,
}

/// Which node owns the dense trunk (attention, norms, router) of a layer range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrunkRange {
    pub start_layer: u32,
    pub end_layer: u32,
    pub owner: NodeId,
    pub bytes: u64,
    /// The owner's bandwidth-adaptive split, carried into stage units so kernels route exactly
    /// as the legacy planner's assignment did.
    pub q_star_ratio: f64,
}

/// Where one routed expert lives: its holders, fastest first, and the tier of the primary copy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertHome {
    pub layer: u32,
    pub expert: u32,
    pub holders: Vec<NodeId>,
    pub tier: ResidencyTier,
}

/// Routing popularity of experts, from live routing samples or a modelled skew.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExpertPopularity {
    hits: BTreeMap<u64, u64>,
}

impl ExpertPopularity {
    fn key(layer: u32, expert: u32) -> u64 {
        ((layer as u64) << 32) | expert as u64
    }

    pub fn record(&mut self, layer: u32, expert: u32, hits: u64) {
        *self.hits.entry(Self::key(layer, expert)).or_default() += hits;
    }

    pub fn hits(&self, layer: u32, expert: u32) -> u64 {
        self.hits
            .get(&Self::key(layer, expert))
            .copied()
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Every `(layer, expert, hits)` with a non-zero count.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, u64)> + '_ {
        self.hits
            .iter()
            .filter(|(_, hits)| **hits > 0)
            .map(|(key, hits)| ((key >> 32) as u32, (key & 0xffff_ffff) as u32, *hits))
    }

    pub fn total_hits(&self) -> u64 {
        self.hits.values().sum()
    }

    /// Halves every count and forgets the ones that reach zero.
    pub fn halve(&mut self) {
        for hits in self.hits.values_mut() {
            *hits /= 2;
        }
        self.hits.retain(|_, hits| *hits > 0);
    }

    /// This popularity plus `other` scaled by `weight`.
    pub fn merged_with(&self, other: &Self, weight: f64) -> Self {
        let mut merged = self.clone();
        for (layer, expert, hits) in other.iter() {
            let scaled = (hits as f64 * weight.max(0.0)).round() as u64;
            if scaled > 0 {
                merged.record(layer, expert, scaled);
            }
        }
        merged
    }

    /// The `n` hottest experts over all layers, hottest first, layer then id as tie-breaks.
    pub fn hottest(&self, n: usize) -> Vec<(u32, u32, u64)> {
        let mut all: Vec<(u32, u32, u64)> = self.iter().collect();
        all.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        all.truncate(n);
        all
    }

    /// Experts of a layer, most popular first, ids as the tie-break.
    pub fn ranked(&self, layer: u32, num_experts: u32) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..num_experts).collect();
        ids.sort_by_key(|e| (std::cmp::Reverse(self.hits(layer, *e)), *e));
        ids
    }

    /// Probability that a routed slot lands on each expert of the layer; uniform when unknown.
    pub fn probabilities(&self, layer: u32, num_experts: u32) -> Vec<f64> {
        let total: u64 = (0..num_experts).map(|e| self.hits(layer, e)).sum();
        if total == 0 {
            return vec![1.0 / num_experts.max(1) as f64; num_experts as usize];
        }
        (0..num_experts)
            .map(|e| self.hits(layer, e) as f64 / total as f64)
            .collect()
    }

    /// A modelled power-law skew: the expert of rank `r` gets `(r + 1)^-alpha` of the traffic.
    /// `alpha = 0.25` reproduces the learned-pin skew in the R4 review; `alpha = 0` is uniform.
    pub fn power_law(num_layers: u32, num_experts: u32, alpha: f64) -> Self {
        let mut pop = Self::default();
        for layer in 0..num_layers {
            for rank in 0..num_experts {
                let weight = ((rank + 1) as f64).powf(-alpha) * 1_000_000.0;
                // Rotate which expert is hottest per layer so no node collects every hot expert.
                let expert = (rank + layer * 7) % num_experts;
                pop.record(layer, expert, weight.round() as u64);
            }
        }
        pop
    }
}

/// What a plan asks of one node against what the node offers, in GiB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeFit {
    pub node: NodeId,
    pub fast_offered_gb: f64,
    pub fast_used_gb: f64,
    pub warm_offered_gb: f64,
    pub warm_used_gb: f64,
    pub cold_offered_gb: f64,
    pub cold_used_gb: f64,
    /// The owner reviews the plan before anything is staged here.
    pub requires_approval: bool,
    /// Offered fast memory the plan left free, in bytes: the replica budget's raw material.
    #[serde(default)]
    pub fast_free_bytes: u64,
}

/// Whether the plan fits the offers, and what is missing where. colibri refuses to start when
/// the always-resident part plus a minimum expert cache does not fit; lumabri shows each donor
/// the RAM and disk a plan asks of it before anything loads. This is that review for the mesh.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementFit {
    /// Every trunk range and its KV reservation sits in fast or warm memory on its owner, and no
    /// content was left unplaced.
    pub fits: bool,
    /// Some experts fit in no offer and stream from the vault on every step they are routed to.
    pub streams_from_vault: bool,
    /// Bytes that fit nowhere, with no vault to fall back on.
    pub unplaced_gb: f64,
    /// Bytes placed on the vault because no offer had room.
    pub vault_streamed_gb: f64,
    pub per_node: Vec<NodeFit>,
    /// One line per shortfall, for the owner reviewing the plan.
    pub shortfalls: Vec<String>,
}

impl PlacementFit {
    pub fn summary(&self) -> String {
        if self.fits && !self.streams_from_vault {
            "fits the offers".to_string()
        } else if self.fits {
            format!(
                "fits the offers; {:.1} GB of experts stream from the vault",
                self.vault_streamed_gb
            )
        } else {
            format!(
                "does not fit: {} shortfall{}, {:.1} GB unplaced, {:.1} GB streaming from the vault",
                self.shortfalls.len(),
                if self.shortfalls.len() == 1 { "" } else { "s" },
                self.unplaced_gb,
                self.vault_streamed_gb
            )
        }
    }
}

/// The complete placement of a model on the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPlacementPlan {
    pub model_id: String,
    pub strategy: ExecutionStrategy,
    pub trunk_ranges: Vec<TrunkRange>,
    /// Every `(layer, expert)` of the model exactly once.
    pub experts: Vec<ExpertHome>,
    pub rules: Vec<PlacementRule>,
    pub fast_memory_used_gb: BTreeMap<NodeId, f64>,
    pub warm_memory_used_gb: BTreeMap<NodeId, f64>,
    /// Expert bytes that fit neither fast nor warm memory anywhere and stream from NVMe or NAS.
    pub cold_gb: f64,
    pub replicated_experts: usize,
    /// How the plan fits the offers it was placed against.
    #[serde(default)]
    pub fit: PlacementFit,
    /// What the load balancer did with the free fast memory, when it ran: the busiest node of
    /// every layer before and after its copies, modelled.
    #[serde(default)]
    pub balance: Option<BalanceReport>,
}

impl ExpertPlacementPlan {
    /// This placement with every extra copy removed: what the balancer started from, for a
    /// before-and-after estimate.
    pub fn primaries_only(&self) -> Self {
        let mut plan = self.clone();
        for home in plan.experts.iter_mut() {
            home.holders.truncate(1);
        }
        for rule in plan.rules.iter_mut() {
            if ModelContentNames::parse_expert(&rule.content.content_id).is_some() {
                rule.candidate_nodes.truncate(1);
                rule.min_replicas = rule.min_replicas.min(1);
            }
        }
        plan.replicated_experts = 0;
        plan.balance = None;
        plan
    }

    /// The share of a step's rows each copy of an expert serves when the rows are dealt round
    /// robin across `copies` holders: holder `index` gets rows `index, index + copies, ...`.
    pub fn row_share(rows: u32, copies: usize, index: usize) -> f64 {
        let rows = rows.max(1);
        let copies = copies.max(1) as u32;
        if index as u32 >= rows {
            return 0.0;
        }
        let mine = (rows - index as u32).div_ceil(copies);
        mine as f64 / rows as f64
    }
    /// Content-addressed identity of the placement: the shape, every trunk range with its owner,
    /// every expert with its holders and tier. Approvals and calibration records bind to it, so
    /// a changed placement asks again and measures again.
    pub fn digest(&self) -> u64 {
        let mut text = format!("{}|{:?}", self.model_id, self.strategy);
        for r in &self.trunk_ranges {
            text.push_str(&format!(
                "|T{}-{}@{}:{}",
                r.start_layer, r.end_layer, r.owner, r.bytes
            ));
        }
        // Tiers are left out: moving an expert between tiers on the same node changes speed,
        // not which node holds it, so it needs neither a new approval nor a new calibration.
        for h in &self.experts {
            text.push_str(&format!("|E{}/{}@{:?}", h.layer, h.expert, h.holders));
        }
        fnv1a_64(text.as_bytes())
    }

    /// For every replicated expert, the holder that should serve it this step: the one with the
    /// lowest `cost` among holders that have one, the primary when none has. Lumabri ranks a
    /// segment's replicas by measured completion cost; this is that choice for an expert's copies.
    pub fn serving_assignment(
        &self,
        cost: impl Fn(&NodeId) -> Option<f64>,
    ) -> BTreeMap<(u32, u32), NodeId> {
        let mut serving = BTreeMap::new();
        for home in self.experts.iter().filter(|h| h.holders.len() > 1) {
            let primary = &home.holders[0];
            let best = home
                .holders
                .iter()
                .filter_map(|n| cost(n).map(|c| (c, n)))
                .min_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| (a.1 != primary).cmp(&(b.1 != primary)))
                })
                .map(|(_, n)| n.clone());
            if let Some(node) = best
                && &node != primary
            {
                serving.insert((home.layer, home.expert), node);
            }
        }
        serving
    }

    pub fn trunk_owner(&self, layer: u32) -> Option<&NodeId> {
        self.trunk_ranges
            .iter()
            .find(|r| r.start_layer <= layer && layer <= r.end_layer)
            .map(|r| &r.owner)
    }

    pub fn holders_of(&self, layer: u32, expert: u32) -> &[NodeId] {
        self.experts
            .iter()
            .find(|h| h.layer == layer && h.expert == expert)
            .map(|h| h.holders.as_slice())
            .unwrap_or(&[])
    }

    /// Experts of a layer whose primary copy is on the node.
    pub fn experts_on(&self, node: &NodeId, layer: u32) -> Vec<u32> {
        self.experts
            .iter()
            .filter(|h| h.layer == layer && h.holders.first() == Some(node))
            .map(|h| h.expert)
            .collect()
    }

    /// Nodes holding any expert of a layer, primary copies only, in node-id order.
    pub fn holder_nodes(&self, layer: u32) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self
            .experts
            .iter()
            .filter(|h| h.layer == layer)
            .filter_map(|h| h.holders.first().cloned())
            .collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    /// Fraction of experts whose primary copy sits in VRAM or unified memory.
    pub fn hot_fraction(&self) -> f64 {
        if self.experts.is_empty() {
            return 1.0;
        }
        let hot = self.experts.iter().filter(|h| h.tier.is_fast()).count();
        hot as f64 / self.experts.len() as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "{:?}: {} trunk ranges, {} experts ({:.0}% in fast memory, {} replicated), {:.1} GB cold",
            self.strategy,
            self.trunk_ranges.len(),
            self.experts.len(),
            self.hot_fraction() * 100.0,
            self.replicated_experts,
            self.cold_gb
        )
    }
}

/// Per-node budget the planner spends while placing content: the node's offer under its polite
/// state, by tier.
#[derive(Debug, Clone)]
struct NodeBudget {
    node_id: NodeId,
    trust_tier: NodeTrustTier,
    fast_tier: Option<ResidencyTier>,
    fast_bw_gbps: f64,
    fast_left: f64,
    fast_used: f64,
    fast_offered: f64,
    warm_bw_gbps: f64,
    warm_left: f64,
    warm_used: f64,
    warm_offered: f64,
    cold_tier: ResidencyTier,
    cold_bw_gbps: f64,
    cold_left: f64,
    cold_used: f64,
    cold_offered: f64,
    requires_approval: bool,
    hop_ms: f64,
}

impl NodeBudget {
    fn from_node(node: &HardwareNode) -> Self {
        Self::from_node_and_offer(node, &node.default_offer())
    }

    /// The budget an offer gives the planner on a node: the offered fast memory and RAM under the
    /// host's polite state, the offered NVMe for cold banks.
    fn from_node_and_offer(node: &HardwareNode, offer: &ResourceOffer) -> Self {
        let offer = offer.clone().clamped_to(node);
        let live = offer.under_polite_state(node, node.polite_state);
        let (fast_tier, fast_bw_gbps) = if node.memory.unified_mem_gb > 0.0 {
            (
                Some(ResidencyTier::UnifiedMemory),
                node.memory.unified_mem_bw_gbps,
            )
        } else if node.memory.vram_gb > 0.0 {
            (Some(ResidencyTier::Vram), node.memory.vram_bw_gbps)
        } else {
            (None, 0.0)
        };
        let (cold_tier, cold_bw_gbps) = if offer.nvme_gb > 0.0 {
            (
                ResidencyTier::LocalNvme,
                node.memory.nvme_read_bw_gbps.max(0.5),
            )
        } else {
            (ResidencyTier::NasVault, 0.3)
        };
        Self {
            node_id: node.id.clone(),
            trust_tier: node.trust_tier,
            fast_tier,
            fast_bw_gbps,
            fast_left: live.fast_memory_gb * GIB,
            fast_used: 0.0,
            fast_offered: live.fast_memory_gb * GIB,
            warm_bw_gbps: node.memory.system_ram_bw_gbps.max(1.0),
            warm_left: live.system_ram_gb * GIB,
            warm_used: 0.0,
            warm_offered: live.system_ram_gb * GIB,
            cold_tier,
            cold_bw_gbps,
            cold_left: offer.nvme_gb * GIB,
            cold_used: 0.0,
            cold_offered: offer.nvme_gb * GIB,
            requires_approval: offer.requires_approval,
            hop_ms: node.network_latency_ms,
        }
    }

    fn fast_cost_ms(&self, bytes: f64) -> f64 {
        (self.fast_used + bytes) / (self.fast_bw_gbps.max(1.0) * 1e9) * 1000.0
    }

    fn warm_cost_ms(&self, bytes: f64) -> f64 {
        (self.warm_used + bytes) / (self.warm_bw_gbps.max(1.0) * 1e9) * 1000.0
    }

    fn cold_cost_ms(&self, bytes: f64) -> f64 {
        (self.cold_used + bytes) / (self.cold_bw_gbps.max(0.1) * 1e9) * 1000.0
    }

    fn take_cold(&mut self, bytes: f64) -> bool {
        if self.cold_tier == ResidencyTier::LocalNvme && self.cold_left >= bytes {
            self.cold_left -= bytes;
            self.cold_used += bytes;
            true
        } else {
            false
        }
    }

    fn take_fast(&mut self, bytes: f64) -> bool {
        if self.fast_tier.is_some() && self.fast_left >= bytes {
            self.fast_left -= bytes;
            self.fast_used += bytes;
            true
        } else {
            false
        }
    }

    fn take_warm(&mut self, bytes: f64) -> bool {
        if self.warm_left >= bytes {
            self.warm_left -= bytes;
            self.warm_used += bytes;
            true
        } else {
            false
        }
    }
}

pub struct ExpertPlacementPlanner;

impl ExpertPlacementPlanner {
    /// Places a model on the mesh in the given shape for inference. With `popularity`, hot
    /// experts are replicated into leftover fast memory under the expert-parallel shape.
    pub fn plan(
        model: &ModelSpec,
        nodes: &[HardwareNode],
        strategy: ExecutionStrategy,
        popularity: Option<&ExpertPopularity>,
    ) -> ExpertPlacementPlan {
        Self::plan_for_workload(model, nodes, strategy, popularity, WorkloadKind::Inference)
    }

    /// Like [`ExpertPlacementPlanner::plan`], with the trunk ranges sized for the given workload,
    /// so a training preview's stages match the training planner's layer split.
    pub fn plan_for_workload(
        model: &ModelSpec,
        nodes: &[HardwareNode],
        strategy: ExecutionStrategy,
        popularity: Option<&ExpertPopularity>,
        workload: WorkloadKind,
    ) -> ExpertPlacementPlan {
        Self::plan_with_offers(
            model,
            nodes,
            strategy,
            popularity,
            workload,
            &BTreeMap::new(),
        )
    }

    /// The planner proper: places the model against each node's offer (the node's default offer
    /// where `offers` has none) and reports how the plan fits them. Link segments for the
    /// balancer are derived from the nodes' round trips.
    pub fn plan_with_offers(
        model: &ModelSpec,
        nodes: &[HardwareNode],
        strategy: ExecutionStrategy,
        popularity: Option<&ExpertPopularity>,
        workload: WorkloadKind,
        offers: &BTreeMap<NodeId, ResourceOffer>,
    ) -> ExpertPlacementPlan {
        let segments = LinkSegments::derive(nodes);
        Self::plan_with_offers_segmented(
            model, nodes, strategy, popularity, workload, offers, &segments,
        )
    }

    /// [`ExpertPlacementPlanner::plan_with_offers`] with the link segments the balancer packs
    /// across first.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_with_offers_segmented(
        model: &ModelSpec,
        nodes: &[HardwareNode],
        strategy: ExecutionStrategy,
        popularity: Option<&ExpertPopularity>,
        workload: WorkloadKind,
        offers: &BTreeMap<NodeId, ResourceOffer>,
        segments: &LinkSegments,
    ) -> ExpertPlacementPlan {
        let shard_plan = ShardPlanner::compute_plan(0, model, workload, nodes);
        let mut trunk_ranges: Vec<TrunkRange> = shard_plan
            .assignments
            .iter()
            .filter(|a| a.layer_count > 0)
            .filter_map(|a| {
                let (s, e) = (a.start_layer?, a.end_layer?);
                Some(TrunkRange {
                    start_layer: s,
                    end_layer: e,
                    owner: a.node_id.clone(),
                    bytes: (model.dense_layer_size_gb * (e - s + 1) as f64 * GIB) as u64,
                    q_star_ratio: a.tiered_memory.q_star_accel_ratio,
                })
            })
            .collect();
        trunk_ranges.sort_by_key(|r| r.start_layer);

        let mut budgets: Vec<NodeBudget> = nodes
            .iter()
            .filter(|n| {
                (n.roles.contains(&NodeRole::ComputeWorker)
                    || n.roles.contains(&NodeRole::BurstAccelerator))
                    && n.capacity_score(WorkloadKind::Inference) > 0.0
            })
            .map(|n| match offers.get(&n.id) {
                Some(offer) => NodeBudget::from_node_and_offer(n, offer),
                None => NodeBudget::from_node(n),
            })
            .collect();
        let seed_node = nodes
            .iter()
            .find(|n| n.roles.contains(&NodeRole::StorageVault))
            .map(|n| n.id.clone());

        let mut rules = Vec::new();
        let mut cold_bytes = 0.0f64;
        let mut shortfalls: Vec<String> = Vec::new();
        let mut trunk_shortfalls = 0usize;
        let mut vault_bytes = 0.0f64;
        let mut unplaced_bytes = 0.0f64;
        let mut unplaced_experts = 0usize;
        let mut vault_experts: BTreeMap<String, (usize, f64)> = BTreeMap::new();

        // 1. Trunks and KV reservations on their owners.
        for range in &trunk_ranges {
            let layers = (range.end_layer - range.start_layer + 1) as f64;
            let kv_bytes = model.kv_cache_per_layer_gb * layers * GIB;
            let (tier, cold) = match budgets.iter_mut().find(|b| b.node_id == range.owner) {
                Some(b) => {
                    let trunk_bytes = range.bytes as f64;
                    let _ = b.take_fast(kv_bytes) || b.take_warm(kv_bytes);
                    if b.take_fast(trunk_bytes) {
                        (b.fast_tier.unwrap_or(ResidencyTier::SystemRam), false)
                    } else if b.take_warm(trunk_bytes) {
                        (ResidencyTier::SystemRam, false)
                    } else {
                        shortfalls.push(format!(
                            "{} cannot hold trunk L{:03}-L{:03} ({:.1} GB) in its offered {:.1} GB fast memory and {:.1} GB RAM; it streams from {:?}",
                            b.node_id,
                            range.start_layer,
                            range.end_layer,
                            trunk_bytes / GIB,
                            b.fast_offered / GIB,
                            b.warm_offered / GIB,
                            b.cold_tier
                        ));
                        trunk_shortfalls += 1;
                        (b.cold_tier, true)
                    }
                }
                None => {
                    shortfalls.push(format!(
                        "trunk L{:03}-L{:03} ({:.1} GB) is owned by {}, which offers no compute budget; it streams from the vault",
                        range.start_layer,
                        range.end_layer,
                        range.bytes as f64 / GIB,
                        range.owner
                    ));
                    trunk_shortfalls += 1;
                    (ResidencyTier::NasVault, true)
                }
            };
            if cold {
                cold_bytes += range.bytes as f64;
            }
            if tier == ResidencyTier::NasVault {
                vault_bytes += range.bytes as f64;
            }
            rules.push(PlacementRule {
                content: ModelContentNames::trunk(
                    &model.id,
                    range.start_layer,
                    range.end_layer,
                    range.bytes,
                ),
                min_replicas: 1,
                preferred_tier: tier,
                candidate_nodes: vec![range.owner.clone()],
                pin: true,
            });
        }

        // 2. Experts.
        let expert_bytes = model.single_expert_size_gb * GIB;
        let mut experts = Vec::new();
        if model.is_moe && model.experts_per_layer > 0 {
            for layer in 0..model.num_layers {
                let boundary = layer == 0 || layer + 1 == model.num_layers;
                let owner = trunk_ranges
                    .iter()
                    .find(|r| r.start_layer <= layer && layer <= r.end_layer)
                    .map(|r| r.owner.clone());
                let order = popularity
                    .map(|p| p.ranked(layer, model.experts_per_layer))
                    .unwrap_or_else(|| (0..model.experts_per_layer).collect());

                for expert in order {
                    let (holder, tier, cold) = match strategy {
                        ExecutionStrategy::PipelineChain => {
                            Self::place_with_owner(&mut budgets, owner.as_ref(), expert_bytes)
                        }
                        ExecutionStrategy::ExpertParallel => {
                            Self::place_spread(&mut budgets, boundary, owner.as_ref(), expert_bytes)
                        }
                    };
                    let owner_label = owner
                        .as_ref()
                        .map(|o| o.0.clone())
                        .unwrap_or_else(|| "no owner".to_string());
                    let holder = match holder {
                        Some(h) => {
                            if tier == ResidencyTier::NasVault {
                                vault_bytes += expert_bytes;
                                let entry = vault_experts.entry(owner_label).or_default();
                                entry.0 += 1;
                                entry.1 += expert_bytes;
                            }
                            h
                        }
                        None => match &seed_node {
                            Some(seed) => {
                                vault_bytes += expert_bytes;
                                let entry = vault_experts.entry(owner_label).or_default();
                                entry.0 += 1;
                                entry.1 += expert_bytes;
                                seed.clone()
                            }
                            None => {
                                unplaced_bytes += expert_bytes;
                                unplaced_experts += 1;
                                NodeId("unplaced".into())
                            }
                        },
                    };
                    if cold {
                        cold_bytes += expert_bytes;
                    }
                    experts.push(ExpertHome {
                        layer,
                        expert,
                        holders: vec![holder],
                        tier,
                    });
                }
            }
        }
        if unplaced_experts > 0 {
            shortfalls.push(format!(
                "{unplaced_experts} experts ({:.1} GB) fit in no offer and the mesh has no vault to stream them from",
                unplaced_bytes / GIB
            ));
        }
        for (owner, (count, bytes)) in &vault_experts {
            shortfalls.push(format!(
                "{count} experts of layers owned by {owner} ({:.1} GB) exceed the offers and stream from the vault on every step they are routed to",
                bytes / GIB
            ));
        }

        // 3. Rules for the primary copies; the balancer adds its copies below, into whatever
        // fast memory the offers leave free.
        for home in &experts {
            rules.push(PlacementRule {
                content: ModelContentNames::expert(
                    &model.id,
                    home.layer,
                    home.expert,
                    expert_bytes as u64,
                ),
                min_replicas: home.holders.len() as u32,
                preferred_tier: home.tier,
                candidate_nodes: home.holders.clone(),
                pin: home.tier.is_fast(),
            });
        }

        let round = |v: f64| (v * 100.0).round() / 100.0;
        let fit = PlacementFit {
            fits: unplaced_bytes <= 0.0 && trunk_shortfalls == 0,
            streams_from_vault: vault_bytes > 0.0,
            unplaced_gb: round(unplaced_bytes / GIB),
            vault_streamed_gb: round(vault_bytes / GIB),
            per_node: budgets
                .iter()
                .map(|b| NodeFit {
                    node: b.node_id.clone(),
                    fast_offered_gb: round(b.fast_offered / GIB),
                    fast_used_gb: round(b.fast_used / GIB),
                    warm_offered_gb: round(b.warm_offered / GIB),
                    warm_used_gb: round(b.warm_used / GIB),
                    cold_offered_gb: round(b.cold_offered / GIB),
                    cold_used_gb: round(b.cold_used / GIB),
                    requires_approval: b.requires_approval,
                    fast_free_bytes: if b.fast_tier.is_some() {
                        b.fast_left.max(0.0) as u64
                    } else {
                        0
                    },
                })
                .collect(),
            shortfalls,
        };
        let mut plan = ExpertPlacementPlan {
            model_id: model.id.clone(),
            strategy,
            trunk_ranges,
            experts,
            rules,
            fast_memory_used_gb: budgets
                .iter()
                .map(|b| (b.node_id.clone(), round(b.fast_used / GIB)))
                .collect(),
            warm_memory_used_gb: budgets
                .iter()
                .map(|b| (b.node_id.clone(), round(b.warm_used / GIB)))
                .collect(),
            cold_gb: round(cold_bytes / GIB),
            replicated_experts: 0,
            fit,
            balance: None,
        };

        // 4. Load-proportional replication (EPLB) under the expert-parallel shape, from the
        // learned heat, into the slots the offers leave free, packed across link segments and
        // then nodes so the busiest node of every layer reads the least.
        if strategy == ExecutionStrategy::ExpertParallel
            && let Some(pop) = popularity
            && !pop.is_empty()
        {
            let budget = ReplicaBudget::from_plan(&plan, model);
            let report = ExpertLoadBalancer::balance(
                &mut plan,
                model,
                nodes,
                pop,
                &budget,
                segments,
                true,
                BalanceMode::Full,
            );
            plan.balance = Some(report);
        }
        plan
    }

    /// Chain shape: the expert goes to its layer's trunk owner, fast then warm then cold.
    fn place_with_owner(
        budgets: &mut [NodeBudget],
        owner: Option<&NodeId>,
        bytes: f64,
    ) -> (Option<NodeId>, ResidencyTier, bool) {
        let Some(owner) = owner else {
            return (None, ResidencyTier::NasVault, true);
        };
        let Some(b) = budgets.iter_mut().find(|b| &b.node_id == owner) else {
            return (Some(owner.clone()), ResidencyTier::NasVault, true);
        };
        if b.take_fast(bytes) {
            (
                Some(owner.clone()),
                b.fast_tier.unwrap_or(ResidencyTier::SystemRam),
                false,
            )
        } else if b.take_warm(bytes) {
            (Some(owner.clone()), ResidencyTier::SystemRam, false)
        } else if b.take_cold(bytes) {
            (Some(owner.clone()), ResidencyTier::LocalNvme, true)
        } else {
            (Some(owner.clone()), ResidencyTier::NasVault, true)
        }
    }

    /// Expert-parallel shape: the node whose fast memory finishes the extra bytes soonest, then
    /// the same over warm memory, then the same over the mesh's NVMe drives so cold experts
    /// stream from several drives at once, then the NAS vault behind the trunk owner.
    fn place_spread(
        budgets: &mut [NodeBudget],
        boundary: bool,
        owner: Option<&NodeId>,
        bytes: f64,
    ) -> (Option<NodeId>, ResidencyTier, bool) {
        let allowed = |b: &NodeBudget| !boundary || b.trust_tier.can_host_boundary_ingress_egress();
        let fast = budgets
            .iter()
            .enumerate()
            .filter(|(_, b)| allowed(b) && b.fast_tier.is_some() && b.fast_left >= bytes)
            .min_by(|(_, a), (_, b)| {
                a.fast_cost_ms(bytes)
                    .partial_cmp(&b.fast_cost_ms(bytes))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        if let Some(i) = fast {
            let b = &mut budgets[i];
            b.take_fast(bytes);
            return (
                Some(b.node_id.clone()),
                b.fast_tier.unwrap_or(ResidencyTier::SystemRam),
                false,
            );
        }
        let warm = budgets
            .iter()
            .enumerate()
            .filter(|(_, b)| allowed(b) && b.warm_left >= bytes)
            .min_by(|(_, a), (_, b)| {
                a.warm_cost_ms(bytes)
                    .partial_cmp(&b.warm_cost_ms(bytes))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        if let Some(i) = warm {
            let b = &mut budgets[i];
            b.take_warm(bytes);
            return (Some(b.node_id.clone()), ResidencyTier::SystemRam, false);
        }
        let cold = budgets
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                allowed(b) && b.cold_tier == ResidencyTier::LocalNvme && b.cold_left >= bytes
            })
            .min_by(|(_, a), (_, b)| {
                a.cold_cost_ms(bytes)
                    .partial_cmp(&b.cold_cost_ms(bytes))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        if let Some(i) = cold {
            let b = &mut budgets[i];
            b.take_cold(bytes);
            return (Some(b.node_id.clone()), ResidencyTier::LocalNvme, true);
        }
        Self::place_with_owner(budgets, owner, bytes)
    }
}

/// Modelled time per token of one step shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepCostEstimate {
    pub strategy: ExecutionStrategy,
    pub rows: u32,
    pub step_ms: f64,
    pub ms_per_token: f64,
    pub tokens_per_sec: f64,
    pub weight_read_ms: f64,
    pub hop_ms: f64,
    pub hops_per_step: u32,
    pub cold_stream_ms: f64,
    pub busiest_node: Option<NodeId>,
    /// Weight reads summed on the busiest node over the step, in ms.
    #[serde(default)]
    pub busiest_node_ms: f64,
    /// Expert reads on the critical path: per layer the slowest holder, summed over layers.
    /// This is the figure the load balancer lowers.
    #[serde(default)]
    pub expert_read_ms: f64,
    /// Always true: these figures come from a model, never from a measurement.
    pub modelled: bool,
}

/// The load balancer's effect on the expert-parallel shape, before and after its copies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BalanceSummary {
    pub replicas_added: usize,
    pub replicas_dropped: usize,
    pub segments: usize,
    pub hierarchical: bool,
    /// The balancer's own account, under its even-split assumption: per layer the busiest
    /// node's expected expert bytes and read time, summed over layers.
    pub busiest_bytes_before: f64,
    pub busiest_bytes_after: f64,
    pub busiest_ms_before: f64,
    pub busiest_ms_after: f64,
    /// The cost model's account at `rows`: expert reads on the critical path and the whole
    /// step. At one row every expert's rows go to its primary, so a copy changes nothing
    /// until the batch is wider.
    pub expert_read_ms_before: f64,
    pub expert_read_ms_after: f64,
    pub step_ms_before: f64,
    pub step_ms_after: f64,
    pub rows: u32,
    /// Always true.
    pub modelled: bool,
    pub rationale: String,
}

/// What settled a [`StrategyChoice`].
pub const DECIDED_BY_MODELLED: &str = "modelled";
pub const DECIDED_BY_MEASURED: &str = "measured";
pub const DECIDED_BY_DENSE: &str = "dense";

fn default_decided_by() -> String {
    DECIDED_BY_MODELLED.to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyChoice {
    pub chosen: ExecutionStrategy,
    pub chain: StepCostEstimate,
    pub expert_parallel: StepCostEstimate,
    pub rationale: String,
    /// A calibration of the chain shape that binds to this mesh and placement, if one exists.
    #[serde(default)]
    pub measured_chain: Option<CalibrationRecord>,
    /// The same for the expert-parallel shape.
    #[serde(default)]
    pub measured_expert_parallel: Option<CalibrationRecord>,
    /// `measured` when both shapes carry a trusted calibration and the faster one was chosen;
    /// `dense` when the model has no routed experts; else `modelled`.
    #[serde(default = "default_decided_by")]
    pub decided_by: String,
    /// What the load balancer did to the expert-parallel placement, when learned heat let it
    /// run: the busiest node before and after, modelled.
    #[serde(default)]
    pub balance: Option<BalanceSummary>,
    /// What DeepEP's selective dispatch layout, asymmetric-domain forwarding, low-latency FP8
    /// quantization and Zero-SM hook overlap do to the expert-parallel step, modelled.
    #[serde(default)]
    pub deep_ep: Option<DeepEpStepAccount>,
    /// What the strongest single node in the mesh would achieve running the model alone under
    /// Ollama's `Q4_K_M` + `q8_0` KV cache hybrid GPU+DDR5 split (`EstimateGpuLayout` +
    /// `vec_dot_q4_K_q8_K` / `MulMatID`), modelled.
    #[serde(default)]
    pub ollama_single_box_baseline: Option<OllamaSingleBoxBaseline>,
}

/// Analytical (`modelled: true`) single-box baseline modelled after `ollama/ollama`
/// (`llm/memory.go` `EstimateGpuLayout` + `ml/backend/ggml` `vec_dot_q4_K_q8_K` & `MulMatID`):
///
/// Evaluates every node in the mesh as a standalone hybrid GPU+CPU runner with `Q4_K_M` weights
/// (`4.85 bits/weight`, `0.303125x` of 16-bit weight size) and `q8_0` FlashAttention KV cache
/// (`34 bytes` per 32 elements), placing as much of the model as fits into fast VRAM/UMA and
/// streaming the overflow in place from host DDR5 (`SystemRam`) with zero network hops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OllamaSingleBoxBaseline {
    pub best_node: NodeId,
    pub weight_quant: String,
    pub kv_cache_type: String,
    pub q4km_total_weight_gb: f64,
    pub q4km_active_weight_gb_per_step: f64,
    pub fast_tier_resident_gb: f64,
    pub system_ram_spill_gb: f64,
    pub nvme_spill_gb: f64,
    pub single_box_ms_per_token: f64,
    pub single_box_tokens_per_sec: f64,
    pub chosen_distributed_ms_per_token: f64,
    pub distributed_speedup_vs_single_box_q4km: f64,
    pub beats_single_box_ollama_q4km: bool,
    pub modelled: bool,
}

impl OllamaSingleBoxBaseline {
    /// Ratio of `Q4_K_M` (`~4.85 bits/weight`) to 16-bit (`16 bits/weight`) checkpoint bytes.
    pub const Q4_K_M_TO_BF16_RATIO: f64 = 4.85 / 16.0;

    pub fn estimate(
        model: &ModelSpec,
        nodes: &[HardwareNode],
        cost_model: &StrategyCostModel,
        chosen_distributed_ms_per_token: f64,
    ) -> Option<Self> {
        if nodes.is_empty() {
            return None;
        }
        let eta = cost_model.bandwidth_efficiency.clamp(0.05, 1.0);
        let rows = (cost_model.rows.max(1)) as f64;
        let distinct_scale = rows.powf(cost_model.batch_expert_growth);

        let layers = (model.num_layers.max(1)) as f64;
        let bf16_trunk_gb = layers * model.dense_layer_size_gb;
        let bf16_total_experts_gb = if model.is_moe {
            layers * (model.experts_per_layer as f64) * model.single_expert_size_gb
        } else {
            0.0
        };
        let bf16_active_experts_gb = if model.is_moe {
            let k_eff = ((model.active_experts_per_token as f64) * distinct_scale)
                .min(model.experts_per_layer as f64);
            layers * k_eff * model.single_expert_size_gb
        } else {
            0.0
        };

        let spec_total_gb = (bf16_trunk_gb + bf16_total_experts_gb).max(1e-6);
        let target_q4km_gb = model.total_params_billions * (4.85 / 8.0);
        // If the ModelSpec is given in BF16/FP16 (larger than 4.85 bits/param), scale down to
        // Q4_K_M; if the ModelSpec is already quantized at or below Q4_K_M (e.g. Q4/FP8 .ktb),
        // keep its native Q4/FP8 footprint without double-quantizing.
        let q4km_scale = (target_q4km_gb / spec_total_gb).clamp(Self::Q4_K_M_TO_BF16_RATIO, 1.0);
        let q4km_total_gb = spec_total_gb * q4km_scale;
        let q4km_active_gb = (bf16_trunk_gb + bf16_active_experts_gb) * q4km_scale;
        // Reserve `q8_0` KV cache (`34/64` of FP16 KV) + 0.5 GiB GGML compute buffer in fast memory
        let kv_q8_gb = (layers * model.kv_cache_per_layer_gb * (34.0 / 64.0)).min(2.0);
        let graph_scratch_gb = 0.5;

        let round2 = |v: f64| (v * 100.0).round() / 100.0;
        let mut best: Option<Self> = None;

        for node in nodes {
            let budget = NodeBudget::from_node(node);
            let fast_cap_gb = (budget.fast_left / GIB - kv_q8_gb - graph_scratch_gb).max(0.0);
            let warm_cap_gb = budget.warm_left / GIB;

            let fast_res_gb = q4km_total_gb.min(fast_cap_gb);
            let rem_after_fast = (q4km_total_gb - fast_res_gb).max(0.0);
            let ram_spill_gb = rem_after_fast.min(warm_cap_gb);
            let nvme_spill_gb = (rem_after_fast - ram_spill_gb).max(0.0);

            let total_denom = q4km_total_gb.max(1e-6);
            let fast_frac = fast_res_gb / total_denom;
            let ram_frac = ram_spill_gb / total_denom;
            let nvme_frac = nvme_spill_gb / total_denom;

            let fast_bw = (budget.fast_bw_gbps.max(1.0) * eta).max(0.5);
            let warm_bw = cost_model
                .measured_stream_gbps
                .get(&node.id)
                .copied()
                .unwrap_or_else(|| (budget.warm_bw_gbps * eta).max(0.5))
                .max(0.5);
            let cold_bw = (budget.cold_bw_gbps * eta).max(0.1);

            let step_sec = (q4km_active_gb * fast_frac) / fast_bw
                + (q4km_active_gb * ram_frac) / warm_bw
                + (q4km_active_gb * nvme_frac) / cold_bw;
            let step_ms = step_sec * 1000.0;
            let ms_per_tok = (step_ms / rows).max(0.01);
            let tok_s = 1000.0 / ms_per_tok;

            let dist_ms = chosen_distributed_ms_per_token.max(0.01);
            let speedup = ms_per_tok / dist_ms;

            let candidate = Self {
                best_node: node.id.clone(),
                weight_quant: "Q4_K_M".to_string(),
                kv_cache_type: "q8_0".to_string(),
                q4km_total_weight_gb: round2(q4km_total_gb),
                q4km_active_weight_gb_per_step: round2(q4km_active_gb),
                fast_tier_resident_gb: round2(fast_res_gb),
                system_ram_spill_gb: round2(ram_spill_gb),
                nvme_spill_gb: round2(nvme_spill_gb),
                single_box_ms_per_token: round2(ms_per_tok),
                single_box_tokens_per_sec: round2(tok_s),
                chosen_distributed_ms_per_token: round2(chosen_distributed_ms_per_token),
                distributed_speedup_vs_single_box_q4km: round2(speedup),
                beats_single_box_ollama_q4km: chosen_distributed_ms_per_token < ms_per_tok,
                modelled: true,
            };

            if best
                .as_ref()
                .is_none_or(|b| candidate.single_box_ms_per_token < b.single_box_ms_per_token)
            {
                best = Some(candidate);
            }
        }

        best
    }
}

impl StrategyChoice {
    /// Attaches the calibrations that bind to the two shapes. When both are present and trusted,
    /// the measured figures decide and the rationale says so; with one or none, the modelled
    /// decision stands and the records ride along for the reader. Returns whether the chosen
    /// shape changed.
    pub fn apply_measurements(
        &mut self,
        chain: Option<CalibrationRecord>,
        expert_parallel: Option<CalibrationRecord>,
    ) -> bool {
        self.measured_chain = chain.filter(CalibrationRecord::is_trusted);
        self.measured_expert_parallel = expert_parallel.filter(CalibrationRecord::is_trusted);
        if self.decided_by == DECIDED_BY_DENSE {
            return false;
        }
        let (Some(c), Some(e)) = (&self.measured_chain, &self.measured_expert_parallel) else {
            return false;
        };
        let measured_choice = if c.measured_ms_per_token <= e.measured_ms_per_token {
            ExecutionStrategy::PipelineChain
        } else {
            ExecutionStrategy::ExpertParallel
        };
        let changed = measured_choice != self.chosen;
        self.chosen = measured_choice;
        self.decided_by = DECIDED_BY_MEASURED.to_string();
        self.rationale = format!(
            "measured {:.2} ms/token for the chain over {} step{} against {:.2} ms/token expert-parallel over {} step{} on the {} kernels; the measurements decide. Modelled: {}",
            c.measured_ms_per_token,
            c.steps,
            if c.steps == 1 { "" } else { "s" },
            e.measured_ms_per_token,
            e.steps,
            if e.steps == 1 { "" } else { "s" },
            c.kernels,
            self.rationale
        );
        changed
    }
}

/// Bandwidth-bound cost model for a decode step, after the R4 review's envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCostModel {
    /// Fraction of peak memory bandwidth real kernels reach.
    pub bandwidth_efficiency: f64,
    /// Rows in the step's token batch.
    pub rows: u32,
    /// Measured round trips to nodes, overriding the capability vector's latency.
    pub measured_rtt_ms: BTreeMap<NodeId, f64>,
    /// Exponent of distinct-expert growth with batch size.
    pub batch_expert_growth: f64,
    /// Memory streaming rates nodes measured on themselves, in GB/s, which replace the nominal
    /// system-RAM bandwidth times the efficiency factor for those nodes.
    #[serde(default)]
    pub measured_stream_gbps: BTreeMap<NodeId, f64>,
    /// Link segments the load balancer packs across first; derived from the nodes when absent.
    #[serde(default)]
    pub segments: Option<LinkSegments>,
}

impl Default for StrategyCostModel {
    fn default() -> Self {
        Self {
            bandwidth_efficiency: 0.6,
            rows: 1,
            measured_rtt_ms: BTreeMap::new(),
            batch_expert_growth: 0.7,
            measured_stream_gbps: BTreeMap::new(),
            segments: None,
        }
    }
}

impl StrategyCostModel {
    pub fn with_rows(mut self, rows: u32) -> Self {
        self.rows = rows.max(1);
        self
    }

    pub fn with_segments(mut self, segments: LinkSegments) -> Self {
        self.segments = Some(segments);
        self
    }

    pub fn with_measured_rtt(mut self, node: NodeId, rtt_ms: f64) -> Self {
        self.measured_rtt_ms.insert(node, rtt_ms);
        self
    }

    pub fn with_measured_stream(mut self, node: NodeId, gbps: f64) -> Self {
        if gbps.is_finite() && gbps > 0.0 {
            self.measured_stream_gbps.insert(node, gbps);
        }
        self
    }

    /// Effective bandwidth of a tier on a node in GB/s: the node's own measured streaming rate
    /// for system RAM when it reported one, else the nominal figure scaled by the efficiency.
    fn effective_bw_gbps(
        &self,
        node: &NodeId,
        budget: &NodeBudget,
        tier: ResidencyTier,
        eta: f64,
    ) -> f64 {
        if tier == ResidencyTier::SystemRam
            && let Some(measured) = self.measured_stream_gbps.get(node)
        {
            return measured.max(0.1);
        }
        Self::tier_bw_gbps(budget, tier) * eta
    }

    fn rtt_ms(&self, node: &NodeId, budgets: &BTreeMap<NodeId, NodeBudget>) -> f64 {
        self.measured_rtt_ms
            .get(node)
            .copied()
            .or_else(|| budgets.get(node).map(|b| b.hop_ms))
            .unwrap_or(1.0)
    }

    fn tier_bw_gbps(budget: &NodeBudget, tier: ResidencyTier) -> f64 {
        match tier {
            ResidencyTier::Vram | ResidencyTier::UnifiedMemory => budget.fast_bw_gbps.max(1.0),
            ResidencyTier::SystemRam => budget.warm_bw_gbps,
            ResidencyTier::LocalNvme => budget.cold_bw_gbps,
            ResidencyTier::NasVault => 0.3,
        }
    }

    /// Estimates one step of the placement on the mesh.
    pub fn estimate(
        &self,
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        popularity: Option<&ExpertPopularity>,
    ) -> StepCostEstimate {
        let budgets: BTreeMap<NodeId, NodeBudget> = nodes
            .iter()
            .map(|n| (n.id.clone(), NodeBudget::from_node(n)))
            .collect();
        let eta = self.bandwidth_efficiency.clamp(0.05, 1.0);
        let rows = self.rows.max(1) as f64;
        let expert_bytes = model.single_expert_size_gb * GIB;
        let k = model.active_experts_per_token as f64;
        let distinct_scale = rows.powf(self.batch_expert_growth);

        let mut weight_ms = 0.0f64;
        let mut expert_ms = 0.0f64;
        let mut cold_ms = 0.0f64;
        let mut hop_ms = 0.0f64;
        let mut hops = 0u32;
        let mut per_node_ms: BTreeMap<NodeId, f64> = BTreeMap::new();

        for range in &placement.trunk_ranges {
            let Some(owner_budget) = budgets.get(&range.owner) else {
                continue;
            };
            let trunk_tier = placement
                .rules
                .iter()
                .find(|r| {
                    r.content.content_id
                        == ModelContentNames::trunk_id(
                            &model.id,
                            range.start_layer,
                            range.end_layer,
                        )
                })
                .map(|r| r.preferred_tier)
                .unwrap_or(ResidencyTier::SystemRam);
            let trunk_bw =
                self.effective_bw_gbps(&range.owner, owner_budget, trunk_tier, eta) * 1e9;
            let trunk_ms = range.bytes as f64 / trunk_bw * 1000.0;
            weight_ms += trunk_ms;
            *per_node_ms.entry(range.owner.clone()).or_default() += trunk_ms;
            if !trunk_tier.is_fast() && trunk_tier != ResidencyTier::SystemRam {
                cold_ms += trunk_ms;
            }

            for layer in range.start_layer..=range.end_layer {
                if !model.is_moe || model.experts_per_layer == 0 {
                    continue;
                }
                let probs = popularity
                    .map(|p| p.probabilities(layer, model.experts_per_layer))
                    .unwrap_or_else(|| {
                        vec![1.0 / model.experts_per_layer as f64; model.experts_per_layer as usize]
                    });
                // Expected bytes each node reads for this layer, by tier. An expert's copies
                // share its rows round robin, as the step graph deals them: one row goes to the
                // primary alone, a wide batch is spread evenly (EPLB's assumption).
                let mut node_tier_bytes: BTreeMap<(NodeId, ResidencyTier), f64> = BTreeMap::new();
                for home in placement.experts.iter().filter(|h| h.layer == layer) {
                    let p = probs.get(home.expert as usize).copied().unwrap_or(0.0);
                    let expected = (k * p * distinct_scale).min(1.0) * expert_bytes;
                    if home.holders.is_empty() {
                        *node_tier_bytes
                            .entry((range.owner.clone(), home.tier))
                            .or_default() += expected;
                        continue;
                    }
                    for (index, holder) in home.holders.iter().enumerate() {
                        let share =
                            ExpertPlacementPlan::row_share(self.rows, home.holders.len(), index);
                        if share > 0.0 {
                            *node_tier_bytes
                                .entry((holder.clone(), home.tier))
                                .or_default() += expected * share;
                        }
                    }
                }
                let mut layer_node_ms: BTreeMap<NodeId, f64> = BTreeMap::new();
                let mut layer_cold_ms: BTreeMap<NodeId, f64> = BTreeMap::new();
                for ((node, tier), bytes) in &node_tier_bytes {
                    let bw = budgets
                        .get(node)
                        .map(|b| self.effective_bw_gbps(node, b, *tier, eta))
                        .unwrap_or(0.3 * eta)
                        * 1e9;
                    let ms = bytes / bw * 1000.0;
                    *layer_node_ms.entry(node.clone()).or_default() += ms;
                    if matches!(tier, ResidencyTier::LocalNvme | ResidencyTier::NasVault) {
                        *layer_cold_ms.entry(node.clone()).or_default() += ms;
                    }
                }
                // Holders work at the same time: the slowest bounds the layer, and the cold
                // streaming on the critical path is the slowest drive, not the sum of drives.
                let layer_ms = layer_node_ms.values().copied().fold(0.0, f64::max);
                weight_ms += layer_ms;
                expert_ms += layer_ms;
                cold_ms += layer_cold_ms.values().copied().fold(0.0, f64::max);
                for (node, ms) in &layer_node_ms {
                    *per_node_ms.entry(node.clone()).or_default() += ms;
                }
                let remote: Vec<&NodeId> = layer_node_ms
                    .keys()
                    .filter(|n| **n != range.owner)
                    .collect();
                if !remote.is_empty() {
                    let worst = remote
                        .iter()
                        .map(|n| {
                            self.rtt_ms(n, &budgets)
                                .max(self.rtt_ms(&range.owner, &budgets))
                        })
                        .fold(0.0, f64::max);
                    hop_ms += worst;
                    hops += 1;
                }
            }
        }

        // One hop per stage boundary.
        if placement.trunk_ranges.len() > 1 {
            for pair in placement.trunk_ranges.windows(2) {
                hop_ms += self
                    .rtt_ms(&pair[0].owner, &budgets)
                    .max(self.rtt_ms(&pair[1].owner, &budgets));
                hops += 1;
            }
        }

        let step_ms = weight_ms + hop_ms;
        let ms_per_token = step_ms / rows;
        let round = |v: f64| (v * 100.0).round() / 100.0;
        let busiest = per_node_ms
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(n, ms)| (n.clone(), *ms));
        StepCostEstimate {
            strategy: placement.strategy,
            rows: self.rows.max(1),
            step_ms: round(step_ms),
            ms_per_token: round(ms_per_token),
            tokens_per_sec: round(if ms_per_token > 0.0 {
                1000.0 / ms_per_token
            } else {
                0.0
            }),
            weight_read_ms: round(weight_ms),
            hop_ms: round(hop_ms),
            hops_per_step: hops,
            cold_stream_ms: round(cold_ms),
            busiest_node: busiest.as_ref().map(|(n, _)| n.clone()),
            busiest_node_ms: round(busiest.map_or(0.0, |(_, ms)| ms)),
            expert_read_ms: (expert_ms * 10_000.0).round() / 10_000.0,
            modelled: true,
        }
    }

    /// The balancer's effect on an expert-parallel placement: the placement with its copies
    /// against the same placement with primaries only, in the balancer's own figures and in
    /// this model's at its batch width. `None` when the balancer did not run on the placement.
    pub fn balance_summary(
        &self,
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        popularity: Option<&ExpertPopularity>,
    ) -> Option<BalanceSummary> {
        let report = placement.balance.as_ref()?;
        let after = self.estimate(model, placement, nodes, popularity);
        let before = self.estimate(model, &placement.primaries_only(), nodes, popularity);
        Some(BalanceSummary {
            replicas_added: report.replicas_added,
            replicas_dropped: report.replicas_dropped,
            segments: report.segments,
            hierarchical: report.hierarchical,
            busiest_bytes_before: report.busiest_bytes_before,
            busiest_bytes_after: report.busiest_bytes_after,
            busiest_ms_before: report.busiest_ms_before,
            busiest_ms_after: report.busiest_ms_after,
            expert_read_ms_before: before.expert_read_ms,
            expert_read_ms_after: after.expert_read_ms,
            step_ms_before: before.step_ms,
            step_ms_after: after.step_ms,
            rows: self.rows.max(1),
            modelled: true,
            rationale: report.rationale.clone(),
        })
    }

    /// Places the model both ways, estimates both, and returns the faster shape with its placement.
    pub fn choose(
        &self,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        popularity: Option<&ExpertPopularity>,
    ) -> (StrategyChoice, ExpertPlacementPlan) {
        self.choose_with_offers(model, nodes, popularity, &BTreeMap::new())
    }

    /// [`StrategyCostModel::choose`] against each node's offer.
    pub fn choose_with_offers(
        &self,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        popularity: Option<&ExpertPopularity>,
        offers: &BTreeMap<NodeId, ResourceOffer>,
    ) -> (StrategyChoice, ExpertPlacementPlan) {
        let (choice, chain_plan, ep_plan) =
            self.choose_both_with_offers(model, nodes, popularity, offers);
        let plan = if choice.chosen == ExecutionStrategy::PipelineChain {
            chain_plan
        } else {
            ep_plan
        };
        (choice, plan)
    }

    /// Both placements with the choice between them, for callers that attach measurements
    /// afterwards and may end up running the other shape.
    pub fn choose_both_with_offers(
        &self,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        popularity: Option<&ExpertPopularity>,
        offers: &BTreeMap<NodeId, ResourceOffer>,
    ) -> (StrategyChoice, ExpertPlacementPlan, ExpertPlacementPlan) {
        let segments = self
            .segments
            .clone()
            .unwrap_or_else(|| LinkSegments::derive(nodes));
        let chain_plan = ExpertPlacementPlanner::plan_with_offers_segmented(
            model,
            nodes,
            ExecutionStrategy::PipelineChain,
            popularity,
            WorkloadKind::Inference,
            offers,
            &segments,
        );
        let ep_plan = ExpertPlacementPlanner::plan_with_offers_segmented(
            model,
            nodes,
            ExecutionStrategy::ExpertParallel,
            popularity,
            WorkloadKind::Inference,
            offers,
            &segments,
        );
        let chain = self.estimate(model, &chain_plan, nodes, popularity);
        let ep = self.estimate(model, &ep_plan, nodes, popularity);
        let balance = self.balance_summary(model, &ep_plan, nodes, popularity);
        let deep_ep =
            DeepEpStepAccount::estimate(model, &ep_plan, nodes, popularity, &ep, &segments);

        let dense = !model.is_moe || model.experts_per_layer == 0;
        let chosen = if dense || chain.ms_per_token <= ep.ms_per_token {
            ExecutionStrategy::PipelineChain
        } else {
            ExecutionStrategy::ExpertParallel
        };
        let rationale = if dense {
            "dense model: no routed experts to spread, so the chain has nothing to gain from fan-out"
                .to_string()
        } else {
            format!(
                "modelled {:.1} ms/token for the chain ({} hops, {:.1} ms cold streaming) against {:.1} ms/token expert-parallel ({} hops, {:.1} ms cold streaming); {:.0}% of experts in fast memory under expert-parallel versus {:.0}% in the chain",
                chain.ms_per_token,
                chain.hops_per_step,
                chain.cold_stream_ms,
                ep.ms_per_token,
                ep.hops_per_step,
                ep.cold_stream_ms,
                ep_plan.hot_fraction() * 100.0,
                chain_plan.hot_fraction() * 100.0
            )
        };
        let chosen_ms_per_tok = if chosen == ExecutionStrategy::PipelineChain {
            chain.ms_per_token
        } else {
            ep.ms_per_token
        };
        let ollama_single_box_baseline =
            OllamaSingleBoxBaseline::estimate(model, nodes, self, chosen_ms_per_tok);
        (
            StrategyChoice {
                chosen,
                chain,
                expert_parallel: ep,
                rationale,
                measured_chain: None,
                measured_expert_parallel: None,
                decided_by: if dense {
                    DECIDED_BY_DENSE.to_string()
                } else {
                    DECIDED_BY_MODELLED.to_string()
                },
                balance,
                deep_ep,
                ollama_single_box_baseline,
            },
            chain_plan,
            ep_plan,
        )
    }
}

/// Parameters of a [`AppKind::StageForward`] unit: a contiguous layer range in the chain shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageForwardParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub stage_index: usize,
    pub start_layer: u32,
    pub end_layer: u32,
    pub rows: u32,
    /// The stage owner's bandwidth-adaptive split ratio, which steers expert routing.
    pub q_star_ratio: f64,
    pub experts_per_layer: u32,
    pub active_experts_per_token: u32,
}

/// Parameters of a [`AppKind::TrunkForward`] unit: attention, norms and the router of one layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrunkForwardParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub layer: u32,
    pub rows: u32,
    pub experts_per_layer: u32,
    pub active_experts_per_token: u32,
    /// The trunk owner's bandwidth-adaptive split ratio, which steers the router's locality prior.
    pub q_star_ratio: f64,
    /// The layer's expert slices, when known at planning time, so the trunk unit can run
    /// DeepEP's `get_dispatch_layout` and emit per-holder selective dispatches containing only
    /// the rows `r` where `is_token_in_rank[holder][r]` is true.
    #[serde(default)]
    pub slices: Vec<ExpertSliceParams>,
}

/// An expert whose rows a holder shares with the expert's other copies: this holder serves
/// the rows `index, index + count, ...` of the step's batch and leaves the rest to them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRowShare {
    pub expert: u32,
    pub index: u32,
    pub count: u32,
}

impl ExpertRowShare {
    /// Whether this holder serves `row` for the expert.
    pub fn serves(&self, row: u32) -> bool {
        self.count <= 1 || row % self.count == self.index
    }
}

/// Parameters of an [`AppKind::ExpertSlice`] unit: the experts of one layer resident on one holder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertSliceParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub layer: u32,
    pub rows: u32,
    pub holder: NodeId,
    /// Experts this holder may be asked for; the trunk's routing picks which are needed.
    pub resident_experts: Vec<u32>,
    /// Experts among them whose rows are dealt across their copies (EPLB's even split); an
    /// expert not listed is served by this holder for every row.
    #[serde(default)]
    pub row_shares: Vec<ExpertRowShare>,
}

impl ExpertSliceParams {
    /// Whether this holder serves `expert` for `row`.
    pub fn serves(&self, expert: u32, row: u32) -> bool {
        self.resident_experts.contains(&expert)
            && self
                .row_shares
                .iter()
                .find(|s| s.expert == expert)
                .is_none_or(|s| s.serves(row))
    }
}

/// Parameters of a [`AppKind::Combine`] unit: the fixed-order sum of one layer's partials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CombineParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub layer: u32,
    pub rows: u32,
    pub contributors: Vec<NodeId>,
}

/// Parameters of a [`AppKind::DraftVerify`] unit: the acceptance test at the end of a
/// speculative step (ARCHITECTURE 9.11). The step's stage chain or fan-out verified a batch of
/// `1 + draft_length` rows: the session's own row first, then the drafter's rows. The unit
/// accepts the longest prefix of draft positions that pass the modelled acceptance test,
/// emits the row at that position as the next step's input, and appends the next step's
/// `next_draft_length` draft rows to it, so the next step's first unit reads a verify batch
/// again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DraftVerifyParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    /// The last layer of the model, whose boundary the unit reads.
    pub layer: u32,
    /// Draft positions verified this step.
    pub draft_length: u32,
    /// The drafter's conditional survival per position, `c_1 .. c_draft_length`, modelled.
    pub confidences: Vec<f64>,
    /// Draft rows to append for the next step.
    pub next_draft_length: u32,
    /// The step index within the session, which seeds the next step's draft rows.
    pub step: u64,
    /// The session's seed, which the modelled drafter and acceptance test both use.
    pub seed: u64,
}

pub struct StepGraphBuilder;

impl StepGraphBuilder {
    fn graph_id(session_id: &str, epoch: u64) -> String {
        format!("{session_id}/step-{epoch}")
    }

    /// [`StepGraphBuilder::decode_step_from`] over a verify batch of `1 + params.draft_length`
    /// rows, with a [`AppKind::DraftVerify`] unit on the last stage's owner after the sink:
    /// the acceptance test, whose output is the next step's input.
    #[allow(clippy::too_many_arguments)]
    pub fn speculative_step_from(
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        epoch: u64,
        session_id: &str,
        serving: &BTreeMap<(u32, u32), NodeId>,
        input: &ContentRef,
        params: &DraftVerifyParams,
    ) -> WorkGraph {
        let rows = 1 + params.draft_length;
        let mut graph = Self::decode_step_from(
            model, placement, nodes, epoch, session_id, rows, serving, input,
        );
        let Some(sink) = graph.sinks().last().map(|u| u.id.clone()) else {
            return graph;
        };
        let owner = placement
            .trunk_ranges
            .last()
            .map(|r| r.owner.clone())
            .unwrap_or_else(|| NodeId("coordinator".into()));
        let verify_id = WorkUnitId::qualified(&graph.id, "verify");
        let mut unit = WorkUnit::new(verify_id.clone(), AppKind::DraftVerify)
            .with_params_json(params)
            .expect("params serialize")
            .with_placement(Self::placement_for(&owner, Vec::new(), true))
            .with_cost(
                CostEstimate::new(1e5 * rows as f64, 0, (8 * 1024 * rows) as u64, 0.05)
                    .with_work_units(rows as u64),
            )
            .with_redundancy(RedundancyPolicy::trusted());
        unit.inputs = vec![Self::output_ref(&graph.id, &sink)];
        graph.add_unit(unit);
        graph
            .add_dependency(&verify_id, &sink)
            .expect("the verify unit follows the sink");
        graph
    }

    /// The activation the first unit of a step consumes: the prompt (or previous token) batch.
    pub fn prompt_ref(graph_id: &str) -> ContentRef {
        ContentRef::named(ContentKind::Activation, &format!("{graph_id}/prompt"), 0)
    }

    /// The routing a trunk unit produces for its layer, which the layer's expert slices consume.
    /// Named by the unit's graph-qualified id (`<graph>/<unit>/routing`), whether the id handed
    /// in is qualified or bare.
    pub fn routing_ref(graph_id: &str, trunk_unit_id: &WorkUnitId) -> ContentRef {
        ContentRef::named(
            ContentKind::Activation,
            &format!("{}/routing", trunk_unit_id.in_graph(graph_id)),
            0,
        )
    }

    /// The per-holder selective dispatch packet a trunk unit produces for `holder` after
    /// DeepEP's `get_dispatch_layout`: contains only the rows `r` where `holder` serves at
    /// least one routed expert (`<graph>/<unit>/dispatch/<holder>`).
    pub fn holder_dispatch_ref(
        graph_id: &str,
        trunk_unit_id: &WorkUnitId,
        holder: &NodeId,
    ) -> ContentRef {
        ContentRef::named(
            ContentKind::Activation,
            &format!("{}/dispatch/{}", trunk_unit_id.in_graph(graph_id), holder.0),
            0,
        )
    }

    /// The activation a unit produces, which dependent units list among their inputs. Named by
    /// the unit's graph-qualified id (`<graph>/<unit>/out`), whether the id handed in is
    /// qualified or bare.
    pub fn output_ref(graph_id: &str, unit_id: &WorkUnitId) -> ContentRef {
        ContentRef::named(
            ContentKind::Activation,
            &format!("{}/out", unit_id.in_graph(graph_id)),
            0,
        )
    }

    fn placement_for(
        owner: &NodeId,
        resident: Vec<ContentRef>,
        boundary: bool,
    ) -> PlacementConstraints {
        let mut placement = if boundary {
            PlacementConstraints::trusted_only()
        } else {
            PlacementConstraints::default()
        };
        placement.requires_resident = resident;
        placement.preferred_nodes = vec![owner.clone()];
        placement
    }

    fn cost(bytes: f64, bw_gbps: f64, hop_ms: f64, rows: u32) -> CostEstimate {
        let ms = bytes / (bw_gbps.max(1.0) * 0.6 * 1e9) * 1000.0 + hop_ms;
        CostEstimate::new(
            2.0 * bytes * rows as f64,
            bytes as u64,
            (8 * 1024 * rows) as u64,
            (ms * 100.0).round() / 100.0,
        )
        .with_work_units(rows.max(1) as u64)
    }

    /// The work graph of one decode step for the placement's shape, every expert served by its
    /// primary holder.
    pub fn decode_step(
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        epoch: u64,
        session_id: &str,
        rows: u32,
    ) -> WorkGraph {
        Self::decode_step_served(
            model,
            placement,
            nodes,
            epoch,
            session_id,
            rows,
            &BTreeMap::new(),
        )
    }

    /// [`StepGraphBuilder::decode_step`] with replicated experts served by the holders `serving`
    /// names (see [`ExpertPlacementPlan::serving_assignment`]); experts it does not name are
    /// served by their primary holder. The bytes out are the same whichever copy serves.
    pub fn decode_step_served(
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        epoch: u64,
        session_id: &str,
        rows: u32,
        serving: &BTreeMap<(u32, u32), NodeId>,
    ) -> WorkGraph {
        let graph_id = Self::graph_id(session_id, epoch);
        let input = Self::prompt_ref(&graph_id);
        Self::decode_step_from(
            model, placement, nodes, epoch, session_id, rows, serving, &input,
        )
    }

    /// [`StepGraphBuilder::decode_step_served`] with the step's first unit reading `input`
    /// instead of the graph's own prompt reference: the sink output of the previous step, so
    /// that successive steps of a session chain inside one queue.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_from(
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        epoch: u64,
        session_id: &str,
        rows: u32,
        serving: &BTreeMap<(u32, u32), NodeId>,
        input: &ContentRef,
    ) -> WorkGraph {
        let rows = rows.max(1);
        let budgets: BTreeMap<NodeId, NodeBudget> = nodes
            .iter()
            .map(|n| (n.id.clone(), NodeBudget::from_node(n)))
            .collect();
        let bw_of = |node: &NodeId, tier: ResidencyTier| {
            budgets
                .get(node)
                .map(|b| StrategyCostModel::tier_bw_gbps(b, tier))
                .unwrap_or(1.0)
        };
        let hop_of = |node: &NodeId| budgets.get(node).map_or(1.0, |b| b.hop_ms);
        let expert_bytes = model.single_expert_size_gb * GIB;
        let k = model.active_experts_per_token.max(1) as f64;
        let last_layer = model.num_layers.saturating_sub(1);
        let graph_id = Self::graph_id(session_id, epoch);

        match placement.strategy {
            ExecutionStrategy::PipelineChain => {
                let mut units = Vec::new();
                for (stage_index, range) in placement.trunk_ranges.iter().enumerate() {
                    let layers = (range.end_layer - range.start_layer + 1) as f64;
                    let trunk = ModelContentNames::trunk(
                        &model.id,
                        range.start_layer,
                        range.end_layer,
                        range.bytes,
                    );
                    let boundary = range.start_layer == 0 || range.end_layer == last_layer;
                    let tier = placement
                        .rules
                        .iter()
                        .find(|r| r.content.content_id == trunk.content_id)
                        .map(|r| r.preferred_tier)
                        .unwrap_or(ResidencyTier::SystemRam);
                    let bytes = range.bytes as f64 + k * expert_bytes * layers;
                    let previous_activation = if stage_index == 0 {
                        input.clone()
                    } else {
                        Self::output_ref(
                            &graph_id,
                            &WorkUnitId::qualified(
                                &graph_id,
                                &format!("stage-{:02}", stage_index - 1),
                            ),
                        )
                    };
                    let unit_id =
                        WorkUnitId::qualified(&graph_id, &format!("stage-{stage_index:02}"));
                    let mut unit = WorkUnit::new(unit_id, AppKind::StageForward)
                        .with_params_json(&StageForwardParams {
                            model_id: model.id.clone(),
                            epoch,
                            session_id: session_id.to_string(),
                            stage_index,
                            start_layer: range.start_layer,
                            end_layer: range.end_layer,
                            rows,
                            q_star_ratio: range.q_star_ratio,
                            experts_per_layer: model.experts_per_layer,
                            active_experts_per_token: model.active_experts_per_token,
                        })
                        .expect("params serialize")
                        .with_placement(Self::placement_for(
                            &range.owner,
                            vec![trunk.clone()],
                            boundary,
                        ))
                        .with_cost(Self::cost(
                            bytes,
                            bw_of(&range.owner, tier),
                            if stage_index == 0 {
                                0.0
                            } else {
                                hop_of(&range.owner)
                            },
                            rows,
                        ))
                        .with_redundancy(RedundancyPolicy::trusted());
                    unit.inputs = vec![trunk, previous_activation];
                    units.push(unit);
                }
                WorkGraph::chain(graph_id, units)
            }
            ExecutionStrategy::ExpertParallel => {
                let mut graph = WorkGraph::new(graph_id.clone());
                let mut previous: Option<WorkUnitId> = None;
                let mut previous_activation = input.clone();
                for range in &placement.trunk_ranges {
                    let trunk = ModelContentNames::trunk(
                        &model.id,
                        range.start_layer,
                        range.end_layer,
                        range.bytes,
                    );
                    let trunk_tier = placement
                        .rules
                        .iter()
                        .find(|r| r.content.content_id == trunk.content_id)
                        .map(|r| r.preferred_tier)
                        .unwrap_or(ResidencyTier::SystemRam);
                    let layers = (range.end_layer - range.start_layer + 1) as f64;
                    for layer in range.start_layer..=range.end_layer {
                        let boundary = layer == 0 || layer == last_layer;
                        let trunk_id =
                            WorkUnitId::qualified(&graph_id, &format!("L{layer:03}-trunk"));
                        let trunk_out = Self::output_ref(&graph_id, &trunk_id);

                        // Which holder serves each expert of the layer this step: the copy
                        // `serving` names, else for a batch wider than one row every copy
                        // with the rows dealt round robin (EPLB's even split), else the primary.
                        let mut by_node: BTreeMap<NodeId, Vec<u32>> = BTreeMap::new();
                        let mut shares: BTreeMap<NodeId, Vec<ExpertRowShare>> = BTreeMap::new();
                        if model.is_moe && model.experts_per_layer > 0 {
                            for home in placement.experts.iter().filter(|h| h.layer == layer) {
                                let Some(primary) = home.holders.first() else {
                                    continue;
                                };
                                if let Some(node) = serving
                                    .get(&(layer, home.expert))
                                    .filter(|n| home.holders.contains(n))
                                {
                                    by_node.entry(node.clone()).or_default().push(home.expert);
                                } else if rows > 1 && home.holders.len() > 1 {
                                    let count = home.holders.len().min(rows as usize) as u32;
                                    for (index, node) in
                                        home.holders.iter().take(count as usize).enumerate()
                                    {
                                        by_node.entry(node.clone()).or_default().push(home.expert);
                                        shares.entry(node.clone()).or_default().push(
                                            ExpertRowShare {
                                                expert: home.expert,
                                                index: index as u32,
                                                count,
                                            },
                                        );
                                    }
                                } else {
                                    by_node
                                        .entry(primary.clone())
                                        .or_default()
                                        .push(home.expert);
                                }
                            }
                        }
                        let holders: Vec<NodeId> = by_node.keys().cloned().collect();
                        let layer_slices: Vec<ExpertSliceParams> = holders
                            .iter()
                            .map(|holder| {
                                let mut resident = by_node.get(holder).cloned().unwrap_or_default();
                                resident.sort_unstable();
                                let mut row_shares =
                                    shares.get(holder).cloned().unwrap_or_default();
                                row_shares.sort_by_key(|s| s.expert);
                                ExpertSliceParams {
                                    model_id: model.id.clone(),
                                    epoch,
                                    session_id: session_id.to_string(),
                                    layer,
                                    rows,
                                    holder: holder.clone(),
                                    resident_experts: resident,
                                    row_shares,
                                }
                            })
                            .collect();

                        let mut trunk_unit = WorkUnit::new(trunk_id.clone(), AppKind::TrunkForward)
                            .with_params_json(&TrunkForwardParams {
                                model_id: model.id.clone(),
                                epoch,
                                session_id: session_id.to_string(),
                                layer,
                                rows,
                                experts_per_layer: model.experts_per_layer,
                                active_experts_per_token: model.active_experts_per_token,
                                q_star_ratio: range.q_star_ratio,
                                slices: layer_slices.clone(),
                            })
                            .expect("params serialize")
                            .with_placement(Self::placement_for(
                                &range.owner,
                                vec![trunk.clone()],
                                boundary,
                            ))
                            .with_cost(Self::cost(
                                range.bytes as f64 / layers,
                                bw_of(&range.owner, trunk_tier),
                                0.0,
                                rows,
                            ))
                            .with_redundancy(RedundancyPolicy::trusted());
                        trunk_unit.inputs = vec![trunk.clone(), previous_activation.clone()];
                        graph.add_unit(trunk_unit);
                        if let Some(prev) = &previous {
                            graph
                                .add_dependency(&trunk_id, prev)
                                .expect("step graphs are acyclic");
                        }

                        if !model.is_moe || model.experts_per_layer == 0 {
                            previous_activation = trunk_out;
                            previous = Some(trunk_id);
                            continue;
                        }

                        let mut leaf_ids = Vec::new();
                        for slice_params in &layer_slices {
                            let holder = &slice_params.holder;
                            let refs: Vec<ContentRef> = slice_params
                                .resident_experts
                                .iter()
                                .map(|e| {
                                    ModelContentNames::expert(
                                        &model.id,
                                        layer,
                                        *e,
                                        expert_bytes as u64,
                                    )
                                })
                                .collect();
                            let tier = placement
                                .experts
                                .iter()
                                .find(|h| h.layer == layer && h.holders.first() == Some(holder))
                                .map(|h| h.tier)
                                .unwrap_or(ResidencyTier::SystemRam);
                            let expected_bytes = k
                                * (slice_params.resident_experts.len() as f64
                                    / model.experts_per_layer as f64)
                                * expert_bytes;
                            let leaf_id = WorkUnitId::qualified(
                                &graph_id,
                                &format!("L{layer:03}-experts-{}", holder.0),
                            );
                            let mut leaf = WorkUnit::new(leaf_id.clone(), AppKind::ExpertSlice)
                                .with_params_json(slice_params)
                                .expect("params serialize")
                                .with_placement(Self::placement_for(holder, refs.clone(), boundary))
                                .with_cost(Self::cost(
                                    expected_bytes,
                                    bw_of(holder, tier),
                                    if holder == &range.owner {
                                        0.0
                                    } else {
                                        hop_of(holder)
                                    },
                                    rows,
                                ))
                                .with_redundancy(RedundancyPolicy::trusted());
                            leaf.inputs = {
                                let mut inputs = refs;
                                inputs
                                    .push(Self::holder_dispatch_ref(&graph_id, &trunk_id, holder));
                                inputs
                            };
                            graph.add_unit(leaf);
                            graph
                                .add_dependency(&leaf_id, &trunk_id)
                                .expect("step graphs are acyclic");
                            leaf_ids.push(leaf_id);
                        }

                        let combine_id =
                            WorkUnitId::qualified(&graph_id, &format!("L{layer:03}-combine"));
                        let mut combine = WorkUnit::new(combine_id.clone(), AppKind::Combine)
                            .with_params_json(&CombineParams {
                                model_id: model.id.clone(),
                                epoch,
                                session_id: session_id.to_string(),
                                layer,
                                rows,
                                contributors: holders.clone(),
                            })
                            .expect("params serialize")
                            .with_placement(Self::placement_for(&range.owner, Vec::new(), boundary))
                            .with_cost(
                                CostEstimate::new(
                                    1e6 * rows as f64,
                                    0,
                                    (8 * 1024 * rows * holders.len() as u32) as u64,
                                    0.05,
                                )
                                .with_work_units(rows as u64),
                            )
                            .with_redundancy(RedundancyPolicy::trusted());
                        combine.inputs = leaf_ids
                            .iter()
                            .map(|leaf| Self::output_ref(&graph_id, leaf))
                            .chain(std::iter::once(trunk_out.clone()))
                            .collect();
                        graph.add_unit(combine);
                        for leaf in &leaf_ids {
                            graph
                                .add_dependency(&combine_id, leaf)
                                .expect("step graphs are acyclic");
                        }
                        if leaf_ids.is_empty() {
                            graph
                                .add_dependency(&combine_id, &trunk_id)
                                .expect("step graphs are acyclic");
                        }
                        previous_activation = Self::output_ref(&graph_id, &combine_id);
                        previous = Some(combine_id);
                    }
                }
                graph
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::{DeviceKind, sample_personal_cluster};

    fn cluster() -> Vec<HardwareNode> {
        sample_personal_cluster()
    }

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
    fn expert_ids_parse_back_and_replicas_can_serve() {
        let id = ModelContentNames::expert_id("deepseek-v3-671b-moe-q4k", 25, 45);
        assert_eq!(
            ModelContentNames::parse_expert(&id),
            Some(("deepseek-v3-671b-moe-q4k".to_string(), 25, 45))
        );
        assert_eq!(ModelContentNames::parse_expert("graph/stage-00/out"), None);
        assert_eq!(
            ModelContentNames::parse_expert(&ModelContentNames::trunk_id("m", 0, 3)),
            None
        );

        let nodes = cluster();
        let model = toy_moe();
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 0.5);
        let placement = ExpertPlacementPlanner::plan(
            &model,
            &nodes,
            ExecutionStrategy::ExpertParallel,
            Some(&heat),
        );
        assert!(
            placement.replicated_experts > 0,
            "the toy mesh has room for replicas"
        );
        let replicated = placement
            .experts
            .iter()
            .find(|h| h.holders.len() > 1)
            .expect("a replicated expert");
        let replica = replicated.holders[1].clone();

        // Unmeasured holders: the primary serves. Measured: the cheaper copy serves.
        assert!(placement.serving_assignment(|_| None).is_empty());
        let serving =
            placement.serving_assignment(|node| Some(if *node == replica { 1.0 } else { 50.0 }));
        assert_eq!(
            serving.get(&(replicated.layer, replicated.expert)),
            Some(&replica)
        );

        let default_graph = StepGraphBuilder::decode_step(&model, &placement, &nodes, 1, "s", 1);
        let served_graph =
            StepGraphBuilder::decode_step_served(&model, &placement, &nodes, 1, "s", 1, &serving);
        let leaf = |g: &WorkGraph, node: &NodeId| {
            g.unit(&WorkUnitId::new(format!(
                "L{:03}-experts-{}",
                replicated.layer, node.0
            )))
            .and_then(|u| u.params_json::<ExpertSliceParams>().ok())
            .map(|p| p.resident_experts)
            .unwrap_or_default()
        };
        assert!(leaf(&default_graph, &replicated.holders[0]).contains(&replicated.expert));
        assert!(!leaf(&default_graph, &replica).contains(&replicated.expert));
        assert!(leaf(&served_graph, &replica).contains(&replicated.expert));
        assert!(!leaf(&served_graph, &replicated.holders[0]).contains(&replicated.expert));
        assert!(default_graph.len() <= served_graph.len() + 1);
    }

    #[test]
    fn content_names_are_stable() {
        assert_eq!(
            ModelContentNames::expert("m", 25, 45, 1).content_id,
            "m/L025/E045"
        );
        assert_eq!(
            ModelContentNames::trunk("m", 0, 24, 1).content_id,
            "m/trunk/L000-L024"
        );
    }

    #[test]
    fn chain_keeps_experts_with_their_trunk_owner_and_spread_uses_the_mesh() {
        let model = ModelSpec::deepseek_v3_671b_moe();
        let nodes = cluster();
        let chain =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None);
        let spread =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);

        let expected = (model.num_layers * model.experts_per_layer) as usize;
        assert_eq!(chain.experts.len(), expected);
        assert_eq!(spread.experts.len(), expected);
        assert_eq!(chain.rules.len(), expected + chain.trunk_ranges.len());

        for home in &chain.experts {
            assert_eq!(home.holders.len(), 1);
            assert_eq!(Some(&home.holders[0]), chain.trunk_owner(home.layer));
        }
        let chain_holders: std::collections::BTreeSet<_> =
            chain.experts.iter().map(|h| h.holders[0].clone()).collect();
        let spread_holders: std::collections::BTreeSet<_> = spread
            .experts
            .iter()
            .map(|h| h.holders[0].clone())
            .collect();
        assert!(spread_holders.len() >= chain_holders.len());
        assert!(spread.hot_fraction() >= chain.hot_fraction());
        assert!(spread.summary().contains("ExpertParallel"));

        // Budgets are respected: fast memory used never exceeds what the node offers.
        for node in &nodes {
            let used = spread
                .fast_memory_used_gb
                .get(&node.id)
                .copied()
                .unwrap_or(0.0);
            assert!(
                used <= node.available_fast_memory_gb() + 1e-6,
                "{}",
                node.id
            );
        }
    }

    #[test]
    fn boundary_layer_experts_stay_on_owned_hardware() {
        let model = toy_moe();
        let mut nodes = cluster();
        nodes[2].trust_tier = NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
        nodes[3].trust_tier = NodeTrustTier::VerifiedPeerMiddleLayersOnly;
        let plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        let last = model.num_layers - 1;
        for home in plan
            .experts
            .iter()
            .filter(|h| h.layer == 0 || h.layer == last)
        {
            for holder in &home.holders {
                let node = nodes
                    .iter()
                    .find(|n| &n.id == holder)
                    .expect("holder is a node");
                assert_eq!(node.trust_tier, NodeTrustTier::TrustedCore);
            }
        }
        assert_eq!(plan.holders_of(0, 0).len(), 1);
        assert!(plan.holders_of(99, 0).is_empty());
    }

    #[test]
    fn offers_bound_the_plan_and_the_fit_report_names_what_is_missing() {
        let nodes = cluster();
        let model = toy_moe();
        let plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None);
        assert!(plan.fit.fits, "{:?}", plan.fit.shortfalls);
        assert_eq!(plan.fit.per_node.len(), plan.fast_memory_used_gb.len());
        assert!(
            plan.fit
                .per_node
                .iter()
                .all(|n| n.fast_used_gb <= n.fast_offered_gb + 1e-6
                    && n.warm_used_gb <= n.warm_offered_gb + 1e-6)
        );
        let digest = plan.digest();
        assert_ne!(digest, 0);
        assert_eq!(
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None)
                .digest(),
            digest,
            "the same placement has the same digest"
        );
        assert_ne!(
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None)
                .digest(),
            digest
        );

        // Owners who offer almost nothing: trunks no longer fit, and the fit report says where.
        let offers: BTreeMap<NodeId, ResourceOffer> = nodes
            .iter()
            .map(|n| {
                let mut offer = n.default_offer();
                offer.fast_memory_gb = 0.0;
                offer.system_ram_gb = 0.01;
                offer.nvme_gb = 0.0;
                offer.requires_approval = true;
                (n.id.clone(), offer)
            })
            .collect();
        let starved = ExpertPlacementPlanner::plan_with_offers(
            &model,
            &nodes,
            ExecutionStrategy::PipelineChain,
            None,
            WorkloadKind::Inference,
            &offers,
        );
        assert!(!starved.fit.fits);
        assert!(!starved.fit.shortfalls.is_empty());
        assert!(starved.fit.shortfalls[0].contains("cannot hold trunk"));
        assert!(starved.fit.per_node.iter().all(|n| n.requires_approval));
        assert!(
            starved
                .fit
                .per_node
                .iter()
                .all(|n| n.fast_offered_gb == 0.0)
        );
        assert!(starved.fit.summary().starts_with("does not fit"));
        assert_eq!(
            starved.digest(),
            digest,
            "the same experts on the same owners: tiers do not change the placement's identity"
        );

        // A generous explicit offer is clamped to the machine.
        let node = &nodes[0];
        let huge = ResourceOffer {
            fast_memory_gb: 1e6,
            system_ram_gb: 1e6,
            nvme_gb: 1e6,
            reserved_ram_gb: 0.0,
            slots: 0,
            max_allocations: 0,
            requires_approval: false,
        }
        .clamped_to(node);
        assert!(huge.fast_memory_gb <= node.memory.fast_accelerator_memory_gb());
        assert!(huge.system_ram_gb + huge.reserved_ram_gb <= node.memory.system_ram_gb + 1e-9);
        assert!(huge.reserved_ram_gb >= ResourceOffer::ram_reserve_gb(node.memory.system_ram_gb));
        assert_eq!(huge.slots, 1);
    }

    #[test]
    fn popularity_replicates_hot_experts_only_with_room() {
        let model = toy_moe();
        let nodes = cluster();
        let pop = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 1.0);
        assert_eq!(pop.ranked(0, 8)[0], 0);
        assert_eq!(pop.ranked(1, 8)[0], 7);
        let probs = pop.probabilities(0, 8);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(probs[0] > probs[7]);

        let plain =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        let hot = ExpertPlacementPlanner::plan(
            &model,
            &nodes,
            ExecutionStrategy::ExpertParallel,
            Some(&pop),
        );
        assert_eq!(plain.replicated_experts, 0);
        assert!(plain.balance.is_none());
        assert!(hot.replicated_experts > 0);
        let balance = hot.balance.as_ref().expect("the balancer ran");
        assert!(balance.modelled);
        assert_eq!(balance.replicas_added, hot.replicated_experts);
        assert!(balance.busiest_bytes_after < balance.busiest_bytes_before);
        assert!(
            hot.experts
                .iter()
                .all(|h| h.holders.len() <= 1 + balance.per_layer_budget as usize)
        );
        let replicated_rule = hot
            .rules
            .iter()
            .find(|r| r.min_replicas >= 2)
            .expect("a replicated expert has a multi-replica rule");
        assert_eq!(
            replicated_rule.candidate_nodes.len(),
            replicated_rule.min_replicas as usize
        );
        // Copies land in free fast memory and are counted against it.
        for change in &balance.changes {
            let fit = hot
                .fit
                .per_node
                .iter()
                .find(|f| f.node == change.node)
                .unwrap();
            assert!(fit.fast_used_gb <= fit.fast_offered_gb + 1e-6);
        }

        // The cost model reports the busiest node before and after, and the even split only
        // pays for a batch wider than one row.
        let cost = StrategyCostModel::default();
        let one_row = cost
            .balance_summary(&model, &hot, &nodes, Some(&pop))
            .unwrap();
        assert!(one_row.modelled);
        assert_eq!(one_row.rows, 1);
        assert!(one_row.busiest_ms_after < one_row.busiest_ms_before);
        assert_eq!(
            one_row.expert_read_ms_after, one_row.expert_read_ms_before,
            "one row goes to the primary alone, so copies change nothing yet"
        );
        let eight = StrategyCostModel::default()
            .with_rows(8)
            .balance_summary(&model, &hot, &nodes, Some(&pop))
            .unwrap();
        assert!(
            eight.expert_read_ms_after < eight.expert_read_ms_before,
            "{eight:?}"
        );
        assert!(eight.step_ms_after < eight.step_ms_before, "{eight:?}");
        assert_eq!(ExpertPlacementPlan::row_share(1, 2, 0), 1.0);
        assert_eq!(ExpertPlacementPlan::row_share(1, 2, 1), 0.0);
        assert_eq!(ExpertPlacementPlan::row_share(3, 2, 0), 2.0 / 3.0);
        assert_eq!(ExpertPlacementPlan::row_share(3, 2, 1), 1.0 / 3.0);
        assert!(cost.balance_summary(&model, &plain, &nodes, None).is_none());

        // A wide step deals a replicated expert's rows across its copies, one holder per row.
        let graph = StepGraphBuilder::decode_step(&model, &hot, &nodes, 1, "shares", 4);
        let replicated = hot
            .experts
            .iter()
            .find(|h| h.holders.len() > 1)
            .expect("a replicated expert");
        let slices: Vec<ExpertSliceParams> = graph
            .units()
            .filter(|u| u.app == AppKind::ExpertSlice)
            .filter_map(|u| u.params_json::<ExpertSliceParams>().ok())
            .filter(|p| p.layer == replicated.layer)
            .collect();
        for row in 0..4 {
            let serving: Vec<&NodeId> = slices
                .iter()
                .filter(|p| p.serves(replicated.expert, row))
                .map(|p| &p.holder)
                .collect();
            assert_eq!(
                serving.len(),
                1,
                "row {row} of expert {:?}",
                replicated.expert
            );
        }
        let share = slices
            .iter()
            .flat_map(|p| p.row_shares.iter())
            .find(|s| s.expert == replicated.expert)
            .expect("a row share");
        assert_eq!(share.count as usize, replicated.holders.len().min(4));
        let single = StepGraphBuilder::decode_step(&model, &hot, &nodes, 1, "shares", 1);
        assert!(
            single
                .units()
                .filter_map(|u| u.params_json::<ExpertSliceParams>().ok())
                .all(|p| p.row_shares.is_empty()),
            "one row goes to the primary alone"
        );
    }

    #[test]
    fn cost_model_prefers_fan_out_for_a_model_no_node_can_hold_and_chain_for_dense() {
        let nodes = cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let (choice, plan) = StrategyCostModel::default().choose(&model, &nodes, None);
        assert!(choice.chain.modelled && choice.expert_parallel.modelled);
        assert!(choice.chain.cold_stream_ms > choice.expert_parallel.cold_stream_ms);
        assert!(choice.expert_parallel.hops_per_step >= model.num_layers);
        assert_eq!(choice.chosen, ExecutionStrategy::ExpertParallel);
        assert_eq!(plan.strategy, ExecutionStrategy::ExpertParallel);
        assert!(choice.rationale.contains("ms/token"));
        assert!(choice.expert_parallel.tokens_per_sec > 0.0);

        let dense = ModelSpec::llama_3_70b_lora();
        let (dense_choice, dense_plan) = StrategyCostModel::default().choose(&dense, &nodes, None);
        assert_eq!(dense_choice.chosen, ExecutionStrategy::PipelineChain);
        assert_eq!(dense_plan.experts.len(), 0);
        assert!(dense_choice.rationale.contains("dense"));
    }

    #[test]
    fn measurements_decide_the_shape_only_when_both_shapes_are_measured() {
        use crate::{CalibrationBinding, CalibrationRecord};
        let nodes = cluster();
        let model = toy_moe();
        let cost = StrategyCostModel::default();
        let (mut choice, chain_plan, ep_plan) =
            cost.choose_both_with_offers(&model, &nodes, None, &BTreeMap::new());
        assert_eq!(choice.decided_by, DECIDED_BY_MODELLED);
        let modelled = choice.chosen;
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.0.clone()).collect();
        let bind = |plan: &ExpertPlacementPlan| {
            CalibrationBinding::new(
                &model.id,
                plan.strategy,
                plan.digest(),
                node_ids.clone(),
                BTreeMap::new(),
                1,
            )
        };
        let record = |plan: &ExpertPlacementPlan, ms: f64| {
            CalibrationRecord::from_steps(bind(plan), &[ms], BTreeMap::new(), 1, None, 1)
        };

        // One measurement rides along without deciding.
        assert!(!choice.apply_measurements(Some(record(&chain_plan, 9.0)), None));
        assert_eq!(choice.decided_by, DECIDED_BY_MODELLED);
        assert_eq!(choice.chosen, modelled);
        assert!(choice.measured_chain.is_some());

        // Both measured: the faster shape wins whatever the model said.
        let changed =
            choice.apply_measurements(Some(record(&chain_plan, 9.0)), Some(record(&ep_plan, 3.0)));
        assert_eq!(choice.chosen, ExecutionStrategy::ExpertParallel);
        assert_eq!(changed, modelled != ExecutionStrategy::ExpertParallel);
        assert_eq!(choice.decided_by, DECIDED_BY_MEASURED);
        assert!(choice.rationale.starts_with("measured 9.00 ms/token"));

        let changed =
            choice.apply_measurements(Some(record(&chain_plan, 2.0)), Some(record(&ep_plan, 3.0)));
        assert!(changed);
        assert_eq!(choice.chosen, ExecutionStrategy::PipelineChain);

        // A record that failed the digest gate is dropped, and the decision falls back.
        let mut failed = record(&ep_plan, 1.0);
        failed.digest_matches_chain = Some(false);
        choice.apply_measurements(Some(record(&chain_plan, 2.0)), Some(failed));
        assert!(choice.measured_expert_parallel.is_none());
        assert_eq!(choice.chosen, ExecutionStrategy::PipelineChain);

        // A measured streaming rate replaces the nominal RAM bandwidth in the estimate.
        let slow_ram = StrategyCostModel::default().with_measured_stream(nodes[0].id.clone(), 2.0);
        let nominal = cost.estimate(&model, &chain_plan, &nodes, None);
        let measured = slow_ram.estimate(&model, &chain_plan, &nodes, None);
        assert!(measured.modelled);
        let any_ram = chain_plan
            .rules
            .iter()
            .any(|r| r.preferred_tier == ResidencyTier::SystemRam);
        if any_ram {
            assert!(measured.weight_read_ms >= nominal.weight_read_ms);
        }
        assert!(
            StrategyCostModel::default()
                .with_measured_stream(nodes[0].id.clone(), 0.0)
                .measured_stream_gbps
                .is_empty(),
            "a zero probe is ignored"
        );
    }

    #[test]
    fn measured_round_trips_and_batch_rows_change_the_estimate() {
        let nodes = cluster();
        let model = toy_moe();
        let plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        let base = StrategyCostModel::default().estimate(&model, &plan, &nodes, None);
        let mut slow = StrategyCostModel::default();
        for node in &nodes {
            slow = slow.with_measured_rtt(node.id.clone(), 50.0);
        }
        let slow_est = slow.estimate(&model, &plan, &nodes, None);
        assert!(slow_est.hop_ms > base.hop_ms);
        assert!(slow_est.ms_per_token > base.ms_per_token);

        let batched = StrategyCostModel::default()
            .with_rows(8)
            .estimate(&model, &plan, &nodes, None);
        assert_eq!(batched.rows, 8);
        assert!(batched.ms_per_token < base.ms_per_token);
        assert!(batched.step_ms >= base.step_ms);
    }

    #[test]
    fn chain_step_graph_is_one_unit_per_stage() {
        let nodes = cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None);
        let graph = StepGraphBuilder::decode_step(&model, &plan, &nodes, 7, "sess-a", 1);
        assert_eq!(graph.len(), plan.trunk_ranges.len());
        assert_eq!(graph.wavefronts().unwrap().len(), plan.trunk_ranges.len());
        assert_eq!(graph.id, "sess-a/step-7");
        let first = graph.ready();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].id.0, "sess-a/step-7/stage-00");
        assert!(
            first[0]
                .inputs
                .iter()
                .any(|c| c.content_id == "sess-a/step-7/prompt")
        );
        assert_eq!(
            StepGraphBuilder::output_ref(&graph.id, &first[0].id).content_id,
            "sess-a/step-7/stage-00/out"
        );
        assert_eq!(
            StepGraphBuilder::output_ref(&graph.id, &"stage-00".into()).content_id,
            "sess-a/step-7/stage-00/out",
            "a bare name and the qualified id name the same output"
        );
        // Step 8 chained to step 7 reads step 7's sink output instead of its own prompt.
        let sink = StepGraphBuilder::output_ref(
            &graph.id,
            &graph.sinks().last().map(|u| u.id.clone()).unwrap(),
        );
        let next = StepGraphBuilder::decode_step_from(
            &model,
            &plan,
            &nodes,
            8,
            "sess-a",
            1,
            &BTreeMap::new(),
            &sink,
        );
        assert_eq!(next.id, "sess-a/step-8");
        assert!(
            next.ready()[0]
                .inputs
                .iter()
                .any(|c| c.content_id == sink.content_id)
        );
        let params: StageForwardParams = first[0].params_json().unwrap();
        assert_eq!(params.start_layer, 0);
        assert_eq!(params.stage_index, 0);
        assert!(first[0].placement.boundary_layer_access);
        assert_eq!(
            first[0].placement.preferred_nodes,
            vec![plan.trunk_ranges[0].owner.clone()]
        );
        assert!(graph.critical_path_ms() > 0.0);
    }

    #[test]
    fn expert_parallel_step_graph_fans_out_per_layer_and_fans_back_in() {
        let nodes = cluster();
        let model = toy_moe();
        let plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        let graph = StepGraphBuilder::decode_step(&model, &plan, &nodes, 1, "sess-b", 4);

        let mut trunks = 0;
        let mut slices = 0;
        let mut combines = 0;
        for unit in graph.units() {
            match unit.app {
                AppKind::TrunkForward => trunks += 1,
                AppKind::ExpertSlice => slices += 1,
                AppKind::Combine => combines += 1,
                _ => panic!("unexpected unit kind"),
            }
        }
        assert_eq!(trunks, model.num_layers as usize);
        assert_eq!(combines, model.num_layers as usize);
        assert!(slices >= model.num_layers as usize);

        let ready = graph.ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(
            ready[0].id,
            WorkUnitId::qualified("sess-b/step-1", "L000-trunk")
        );
        assert_eq!(ready[0].id.name(), "L000-trunk");
        assert!(ready[0].placement.boundary_layer_access);
        let levels = graph.wavefronts().unwrap();
        assert_eq!(levels.len(), 3 * model.num_layers as usize);
        assert!(!levels[1].is_empty());

        let slice = graph
            .units()
            .find(|u| u.app == AppKind::ExpertSlice)
            .expect("slice");
        let params: ExpertSliceParams = slice.params_json().unwrap();
        assert_eq!(params.rows, 4);
        assert!(!params.resident_experts.is_empty());
        assert_eq!(slice.placement.preferred_nodes, vec![params.holder.clone()]);
        assert_eq!(
            slice.placement.requires_resident.len(),
            params.resident_experts.len()
        );
        let combine = graph
            .units()
            .find(|u| u.app == AppKind::Combine)
            .expect("combine");
        let cparams: CombineParams = combine.params_json().unwrap();
        assert!(!cparams.contributors.is_empty());
        assert!(graph.critical_path_ms() < graph.total_est_wall_ms());
        assert!(nodes.iter().any(|n| n.device_kind == DeviceKind::HomeNas));
    }
}

// ================================================================================================
// Training, replay and audit graphs
// ================================================================================================

/// Parameters of a [`AppKind::TrainForwardBackward`] unit: one stage's forward and backward
/// pass over its layer range for one micro-batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainStageParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub stage_index: usize,
    pub start_layer: u32,
    pub end_layer: u32,
    pub step_index: u64,
    pub micro_batch_id: u32,
    pub learning_rate: f32,
}

/// Parameters of a legacy [`AppKind::DeMoReduce`] unit: independent per-stage gradient previews.
/// Distinct layer ranges are not replica islands; this graph does not perform a DiLoCo sync.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeMoReduceParams {
    pub model_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub step_index: u64,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    pub contributors: Vec<NodeId>,
}

/// Parameters of a [`AppKind::DreamReplay`] unit: one offline replay over a recorded world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamReplayParams {
    pub replay_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub world_id: String,
    pub mcts_simulations: u32,
    pub puct_exploration_c: f64,
    pub archive_trajectories_count: usize,
}

/// Parameters of a [`AppKind::VerdeAudit`] unit: a refereed layer-bisection audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerdeAuditParams {
    pub audit_id: String,
    pub epoch: u64,
    pub session_id: String,
    pub start_layer: u32,
    pub target_layer: u32,
    pub inject_fault_at_layer: Option<u32>,
}

impl StepGraphBuilder {
    pub fn training_input_ref(graph_id: &str) -> ContentRef {
        ContentRef::named(
            ContentKind::Activation,
            &format!("{graph_id}/train-input"),
            0,
        )
    }

    pub fn training_target_ref(graph_id: &str) -> ContentRef {
        ContentRef::named(
            ContentKind::Activation,
            &format!("{graph_id}/train-target"),
            0,
        )
    }

    /// Legacy stateless training preview: every trunk range trains independently on the same
    /// micro-batch, and a summary unit combines telemetry from the separate gradient previews.
    /// This graph neither persists local optimizer state nor averages replica island deltas.
    pub fn training_step(
        model: &ModelSpec,
        placement: &ExpertPlacementPlan,
        nodes: &[HardwareNode],
        epoch: u64,
        session_id: &str,
        step_index: u64,
        learning_rate: f32,
    ) -> WorkGraph {
        let graph_id = format!("{session_id}/train-{step_index}");
        let mut graph = WorkGraph::new(graph_id.clone());
        let last_layer = model.num_layers.saturating_sub(1);
        let input = Self::training_input_ref(&graph_id);
        let target = Self::training_target_ref(&graph_id);
        let budgets: BTreeMap<NodeId, NodeBudget> = nodes
            .iter()
            .map(|n| (n.id.clone(), NodeBudget::from_node(n)))
            .collect();
        let mut stage_ids = Vec::new();

        for (stage_index, range) in placement.trunk_ranges.iter().enumerate() {
            let trunk = ModelContentNames::trunk(
                &model.id,
                range.start_layer,
                range.end_layer,
                range.bytes,
            );
            let boundary = range.start_layer == 0 || range.end_layer == last_layer;
            let bw = budgets
                .get(&range.owner)
                .map(|b| {
                    if b.fast_tier.is_some() {
                        b.fast_bw_gbps
                    } else {
                        b.warm_bw_gbps
                    }
                })
                .unwrap_or(1.0);
            let unit_id = WorkUnitId::qualified(&graph_id, &format!("stage-{stage_index:02}"));
            let mut unit = WorkUnit::new(unit_id.clone(), AppKind::TrainForwardBackward)
                .with_params_json(&TrainStageParams {
                    model_id: model.id.clone(),
                    epoch,
                    session_id: session_id.to_string(),
                    stage_index,
                    start_layer: range.start_layer,
                    end_layer: range.end_layer,
                    step_index,
                    micro_batch_id: (step_index % 12) as u32,
                    learning_rate,
                })
                .expect("params serialize")
                .with_placement(Self::placement_for(
                    &range.owner,
                    vec![trunk.clone()],
                    boundary,
                ))
                .with_cost(Self::cost(range.bytes as f64 * 3.0, bw, 0.0, 1))
                .with_redundancy(RedundancyPolicy::trusted());
            unit.inputs = vec![trunk, input.clone(), target.clone()];
            graph.add_unit(unit);
            stage_ids.push(unit_id);
        }

        let owners: Vec<NodeId> = placement
            .trunk_ranges
            .iter()
            .map(|r| r.owner.clone())
            .collect();
        let reduce_owner = owners
            .first()
            .cloned()
            .unwrap_or_else(|| NodeId("coordinator".into()));
        let reduce_id = WorkUnitId::qualified(&graph_id, "demo-reduce");
        let mut reduce = WorkUnit::new(reduce_id.clone(), AppKind::DeMoReduce)
            .with_params_json(&DeMoReduceParams {
                model_id: model.id.clone(),
                epoch,
                session_id: session_id.to_string(),
                step_index,
                outer_lr: 0.7,
                outer_momentum: 0.9,
                contributors: owners,
            })
            .expect("params serialize")
            .with_placement(Self::placement_for(&reduce_owner, Vec::new(), true))
            .with_cost(
                CostEstimate::new(1e6, 0, 64 * 1024, 0.5)
                    .with_work_units(stage_ids.len().max(1) as u64),
            )
            .with_redundancy(RedundancyPolicy::trusted());
        reduce.inputs = stage_ids
            .iter()
            .map(|s| Self::output_ref(&graph_id, s))
            .collect();
        graph.add_unit(reduce);
        for stage in &stage_ids {
            graph
                .add_dependency(&reduce_id, stage)
                .expect("training graphs are acyclic");
        }
        graph
    }

    /// A batch of independent Dream-RSI replays: the BOINC shape, a bag with no edges, steered
    /// towards CPU nodes when the mesh has any.
    pub fn replay_batch(
        session_id: &str,
        epoch: u64,
        world_id: &str,
        rollouts: u32,
        mcts_simulations: u32,
        nodes: &[HardwareNode],
    ) -> WorkGraph {
        let graph_id = format!("{session_id}/replay-{epoch}");
        let mut graph = WorkGraph::new(graph_id.clone());
        let cpu_backends: Vec<ComputeBackend> = [
            ComputeBackend::CpuAvx512OrAvx2,
            ComputeBackend::CpuArmNeonNpu,
        ]
        .into_iter()
        .filter(|b| nodes.iter().any(|n| &n.backend == b))
        .collect();
        for i in 0..rollouts.max(1) {
            let placement = PlacementConstraints {
                allowed_backends: cpu_backends.clone(),
                ..PlacementConstraints::default()
            };
            let unit_id = WorkUnitId::qualified(&graph_id, &format!("replay-{i:03}"));
            let unit = WorkUnit::new(unit_id, AppKind::DreamReplay)
                .with_params_json(&DreamReplayParams {
                    replay_id: format!("{session_id}-{epoch}-{i:03}"),
                    epoch,
                    session_id: session_id.to_string(),
                    world_id: world_id.to_string(),
                    mcts_simulations,
                    puct_exploration_c: 1.4,
                    archive_trajectories_count: 256,
                })
                .expect("params serialize")
                .with_placement(placement)
                .with_cost(
                    CostEstimate::new(
                        1e9 * mcts_simulations as f64,
                        0,
                        4096,
                        0.2 * mcts_simulations as f64,
                    )
                    .with_work_units(mcts_simulations.max(1) as u64),
                )
                .with_priority(1)
                .with_redundancy(RedundancyPolicy::trusted());
            graph.add_unit(unit);
        }
        graph
    }

    /// A refereed audit of a layer range, run redundantly on untrusted nodes: disagreement opens
    /// a dispute the dispatcher settles with a trusted referee.
    pub fn audit_unit(
        session_id: &str,
        epoch: u64,
        audit_id: &str,
        start_layer: u32,
        target_layer: u32,
        inject_fault_at_layer: Option<u32>,
    ) -> WorkUnit {
        WorkUnit::new(format!("audit-{audit_id}"), AppKind::VerdeAudit)
            .with_params_json(&VerdeAuditParams {
                audit_id: audit_id.to_string(),
                epoch,
                session_id: session_id.to_string(),
                start_layer,
                target_layer,
                inject_fault_at_layer,
            })
            .expect("params serialize")
            .with_cost(
                CostEstimate::new(
                    1e8 * (target_layer.saturating_sub(start_layer) + 1) as f64,
                    0,
                    2048,
                    2.0,
                )
                .with_work_units((target_layer.saturating_sub(start_layer) + 1) as u64),
            )
            .with_redundancy(RedundancyPolicy::untrusted())
    }
}

#[cfg(test)]
mod phase5_tests {
    use super::*;
    use kadmium_core::sample_personal_cluster;

    #[test]
    fn training_preview_is_a_bag_of_stages_fanning_into_one_summary() {
        let nodes = sample_personal_cluster();
        let model = ModelSpec::llama_3_70b_lora();
        let placement =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::PipelineChain, None);
        let graph = StepGraphBuilder::training_step(&model, &placement, &nodes, 2, "lab", 17, 1e-4);
        let stages = placement.trunk_ranges.len();
        assert_eq!(graph.len(), stages + 1);
        let levels = graph.wavefronts().unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), stages);
        assert_eq!(
            levels[1],
            vec![WorkUnitId::qualified("lab/train-17", "demo-reduce")]
        );
        let reduce = graph.unit(&"demo-reduce".into()).unwrap();
        assert_eq!(reduce.inputs.len(), stages);
        assert!(
            reduce
                .inputs
                .iter()
                .all(|c| c.content_id.starts_with("lab/train-17/stage-"))
        );
        let params: DeMoReduceParams = reduce.params_json().unwrap();
        assert_eq!(params.contributors.len(), stages);
        let first = graph.unit(&"stage-00".into()).unwrap();
        let sp: TrainStageParams = first.params_json().unwrap();
        assert_eq!((sp.step_index, sp.micro_batch_id), (17, 5));
        assert_eq!(first.inputs.len(), 3);
        assert!(first.placement.boundary_layer_access);
    }

    #[test]
    fn replay_batch_is_a_bag_steered_to_cpu_nodes_and_audits_are_redundant() {
        let nodes = sample_personal_cluster();
        let graph = StepGraphBuilder::replay_batch("lab", 1, "gpu-kernel-world", 6, 32, &nodes);
        assert_eq!(graph.len(), 6);
        assert_eq!(graph.wavefronts().unwrap().len(), 1);
        let unit = graph.unit(&"replay-003".into()).unwrap();
        assert!(!unit.placement.allowed_backends.is_empty());
        assert_eq!(unit.priority, 1);
        let params: DreamReplayParams = unit.params_json().unwrap();
        assert_eq!(params.replay_id, "lab-1-003");

        let audit = StepGraphBuilder::audit_unit("lab", 1, "a1", 0, 7, Some(5));
        assert_eq!(audit.redundancy.min_quorum, 2);
        assert_eq!(
            audit.redundancy.validation,
            kadmium_grid::ValidationRule::VerdeBisection
        );
        let ap: VerdeAuditParams = audit.params_json().unwrap();
        assert_eq!(ap.inject_fault_at_layer, Some(5));
    }

    #[test]
    fn ollama_single_box_q4km_baseline_discloses_when_single_node_vram_ddr5_wins_and_when_mesh_wins()
     {
        let nodes = sample_personal_cluster();
        let cost = StrategyCostModel::default();

        // 1. Small 35B-A3B MoE (~19 GB in Q4_K_M, ~3B active): fits entirely in a single PC's
        //    VRAM + DDR5 with zero NVMe spill and 20-50 tok/s single-box decode.
        let mut ornith_35b_a3b = ModelSpec::deepseek_v3_671b_moe();
        ornith_35b_a3b.id = "ornith-1.5-35b-a3b".into();
        ornith_35b_a3b.total_params_billions = 35.0;
        ornith_35b_a3b.num_layers = 40;
        ornith_35b_a3b.dense_layer_size_gb = 0.08;
        ornith_35b_a3b.experts_per_layer = 64;
        ornith_35b_a3b.active_experts_per_token = 4;
        ornith_35b_a3b.single_expert_size_gb = 0.023;
        ornith_35b_a3b.kv_cache_per_layer_gb = 0.01;

        let (small_choice, _) = cost.choose(&ornith_35b_a3b, &nodes, None);
        let small_base = small_choice
            .ollama_single_box_baseline
            .as_ref()
            .expect("baseline present");
        assert!(small_base.modelled);
        assert_eq!(small_base.nvme_spill_gb, 0.0);
        assert!(small_base.single_box_tokens_per_sec >= 20.0);

        // 2. Massive 671B MoE (~400+ GB in Q4_K_M): spills heavily on any single node, so the
        //    multi-node mesh beats the single-box Q4_K_M baseline.
        let deepseek_671b = ModelSpec::deepseek_v3_671b_moe();
        let (large_choice, _) = cost.choose(&deepseek_671b, &nodes, None);
        let large_base = large_choice
            .ollama_single_box_baseline
            .as_ref()
            .expect("baseline present");
        assert!(large_base.nvme_spill_gb > 100.0);
        assert!(large_base.beats_single_box_ollama_q4km);
        assert!(large_base.distributed_speedup_vs_single_box_q4km > 1.0);
    }
}
