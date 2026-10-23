//! Replica catalog and placement rules: the grid's map of where every piece of content lives,
//! in which memory tier, and which transfers would bring the mesh to the placement the rules
//! describe. Modelled on grid data management: Rucio-style replica rules diffed against the
//! catalog into FTS3-style third-party staging jobs.
//!
//! The scheduler reads the catalog to score locality, so "weights stay, activations move" is a
//! consequence of placement rather than a special case in an executor.

use kadmium_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::ContentRef;

/// Where a replica sits on its node. Ordered fastest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ResidencyTier {
    Vram,
    UnifiedMemory,
    SystemRam,
    LocalNvme,
    NasVault,
}

impl ResidencyTier {
    /// 0 is fastest.
    pub fn rank(self) -> u8 {
        match self {
            Self::Vram => 0,
            Self::UnifiedMemory => 1,
            Self::SystemRam => 2,
            Self::LocalNvme => 3,
            Self::NasVault => 4,
        }
    }

    /// Order-of-magnitude read bandwidth of the tier in GB/s, for cost estimates when a node has
    /// reported no measurement.
    pub fn nominal_bandwidth_gbps(self) -> f64 {
        match self {
            Self::Vram => 800.0,
            Self::UnifiedMemory => 200.0,
            Self::SystemRam => 60.0,
            Self::LocalNvme => 5.0,
            Self::NasVault => 0.3,
        }
    }

    pub fn is_fast(self) -> bool {
        matches!(self, Self::Vram | Self::UnifiedMemory)
    }
}

/// One copy of one piece of content on one node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replica {
    pub content: ContentRef,
    pub node_id: NodeId,
    pub tier: ResidencyTier,
    /// Pinned replicas are never offered for eviction.
    pub pinned: bool,
    pub verified_unix_ms: u64,
    pub last_access_unix_ms: u64,
}

impl Replica {
    pub fn new(
        content: ContentRef,
        node_id: NodeId,
        tier: ResidencyTier,
        now_unix_ms: u64,
    ) -> Self {
        Self {
            content,
            node_id,
            tier,
            pinned: false,
            verified_unix_ms: now_unix_ms,
            last_access_unix_ms: now_unix_ms,
        }
    }

    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }
}

/// Desired placement for one piece of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementRule {
    pub content: ContentRef,
    pub min_replicas: u32,
    pub preferred_tier: ResidencyTier,
    /// Destinations in order of preference.
    pub candidate_nodes: Vec<NodeId>,
    pub pin: bool,
}

/// A third-party transfer the catalog needs in order to satisfy a rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StagingJob {
    pub content: ContentRef,
    pub source_node: NodeId,
    pub destination_node: NodeId,
    pub target_tier: ResidencyTier,
    pub bytes: u64,
    pub pin: bool,
}

/// An owner's approval of one placement plan for their node. Nothing is staged on a node whose
/// offer requires approval until its owner has approved the plan that puts content there
/// (lumabri: no model loads until every donor accepts). The approval binds to the placement's
/// digest, so a changed plan asks again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanApproval {
    pub node: NodeId,
    pub model_id: String,
    pub placement_digest: u64,
    pub approved_unix_ms: u64,
}

impl PlanApproval {
    pub fn covers(&self, model_id: &str, placement_digest: u64) -> bool {
        self.model_id == model_id && self.placement_digest == placement_digest
    }
}

/// Staging jobs split into the ones that can start now and the ones held for an owner's approval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StagingPlan {
    pub ready: Vec<StagingJob>,
    pub awaiting_approval: Vec<StagingJob>,
}

impl StagingPlan {
    pub fn bytes_ready(&self) -> u64 {
        self.ready.iter().map(|j| j.bytes).sum()
    }

    pub fn bytes_awaiting(&self) -> u64 {
        self.awaiting_approval.iter().map(|j| j.bytes).sum()
    }

    /// Nodes with staging held back, in node-id order.
    pub fn nodes_awaiting(&self) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self
            .awaiting_approval
            .iter()
            .map(|j| j.destination_node.clone())
            .collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    /// Whether every job can start.
    pub fn is_clear(&self) -> bool {
        self.awaiting_approval.is_empty()
    }
}

/// Which node holds which content, in which tier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplicaCatalog {
    by_content: BTreeMap<String, Vec<Replica>>,
}

impl ReplicaCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a replica, replacing any earlier record of the same content on the same node.
    pub fn record(&mut self, replica: Replica) {
        let entries = self
            .by_content
            .entry(replica.content.content_id.clone())
            .or_default();
        if let Some(existing) = entries.iter_mut().find(|r| r.node_id == replica.node_id) {
            *existing = replica;
        } else {
            entries.push(replica);
        }
    }

    pub fn remove(&mut self, node_id: &NodeId, content_id: &str) -> bool {
        let Some(entries) = self.by_content.get_mut(content_id) else {
            return false;
        };
        let before = entries.len();
        entries.retain(|r| &r.node_id != node_id);
        let removed = entries.len() != before;
        if entries.is_empty() {
            self.by_content.remove(content_id);
        }
        removed
    }

    /// Forgets everything a node held, for example when it leaves the mesh. Returns how many
    /// replicas were dropped.
    pub fn remove_node(&mut self, node_id: &NodeId) -> usize {
        let mut dropped = 0;
        self.by_content.retain(|_, entries| {
            let before = entries.len();
            entries.retain(|r| &r.node_id != node_id);
            dropped += before - entries.len();
            !entries.is_empty()
        });
        dropped
    }

    pub fn holders(&self, content_id: &str) -> &[Replica] {
        self.by_content
            .get(content_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn holder_nodes(&self, content_id: &str) -> Vec<NodeId> {
        self.holders(content_id)
            .iter()
            .map(|r| r.node_id.clone())
            .collect()
    }

    pub fn is_resident(&self, node_id: &NodeId, content_id: &str) -> bool {
        self.holders(content_id)
            .iter()
            .any(|r| &r.node_id == node_id)
    }

    pub fn resident_on(&self, node_id: &NodeId) -> Vec<&Replica> {
        self.by_content
            .values()
            .flatten()
            .filter(|r| &r.node_id == node_id)
            .collect()
    }

    pub fn bytes_resident_on(&self, node_id: &NodeId) -> u64 {
        self.resident_on(node_id)
            .iter()
            .map(|r| r.content.bytes)
            .sum()
    }

    /// The holder in the fastest tier.
    pub fn fastest_holder(&self, content_id: &str) -> Option<&Replica> {
        self.holders(content_id)
            .iter()
            .min_by_key(|r| r.tier.rank())
    }

    /// Fraction of the inputs' bytes already resident on the node, in `0.0..=1.0`. A unit with
    /// no inputs scores 1.0 everywhere.
    pub fn locality_score(&self, node_id: &NodeId, inputs: &[ContentRef]) -> f64 {
        let total: u64 = inputs.iter().map(|c| c.bytes).sum();
        if total == 0 {
            return 1.0;
        }
        let resident: u64 = inputs
            .iter()
            .filter(|c| self.is_resident(node_id, &c.content_id))
            .map(|c| c.bytes)
            .sum();
        resident as f64 / total as f64
    }

    pub fn missing_inputs(&self, node_id: &NodeId, inputs: &[ContentRef]) -> Vec<ContentRef> {
        inputs
            .iter()
            .filter(|c| !self.is_resident(node_id, &c.content_id))
            .cloned()
            .collect()
    }

    pub fn touch(&mut self, node_id: &NodeId, content_id: &str, now_unix_ms: u64) {
        if let Some(entries) = self.by_content.get_mut(content_id)
            && let Some(r) = entries.iter_mut().find(|r| &r.node_id == node_id)
        {
            r.last_access_unix_ms = now_unix_ms;
        }
    }

    /// Least recently used, unpinned replicas on a node whose bytes add up to at least
    /// `bytes_needed`, for a worker that must make room.
    pub fn evict_candidates(&self, node_id: &NodeId, bytes_needed: u64) -> Vec<ContentRef> {
        let mut candidates: Vec<&Replica> = self
            .resident_on(node_id)
            .into_iter()
            .filter(|r| !r.pinned)
            .collect();
        candidates.sort_by_key(|r| r.last_access_unix_ms);
        let mut freed = 0u64;
        let mut out = Vec::new();
        for r in candidates {
            if freed >= bytes_needed {
                break;
            }
            freed += r.content.bytes;
            out.push(r.content.clone());
        }
        out
    }

    pub fn content_count(&self) -> usize {
        self.by_content.len()
    }

    pub fn replica_count(&self) -> usize {
        self.by_content.values().map(Vec::len).sum()
    }

    /// Transfers needed to satisfy the rules, with no capacity limit on destinations.
    pub fn plan_staging(&self, rules: &[PlacementRule]) -> Vec<StagingJob> {
        self.plan_staging_with_capacity(rules, &BTreeMap::new())
    }

    /// Transfers needed to satisfy the rules. A destination listed in `capacity_bytes` is skipped
    /// once its resident bytes plus the bytes already planned for it would exceed its capacity.
    /// Content with no holder anywhere cannot be staged and yields no job.
    pub fn plan_staging_with_capacity(
        &self,
        rules: &[PlacementRule],
        capacity_bytes: &BTreeMap<NodeId, u64>,
    ) -> Vec<StagingJob> {
        let mut jobs = Vec::new();
        let mut planned: BTreeMap<NodeId, u64> = BTreeMap::new();

        for rule in rules {
            let content_id = &rule.content.content_id;
            let holders = self.holders(content_id);
            let Some(source) = self.fastest_holder(content_id) else {
                continue;
            };
            let have = holders.len() as u32;
            let mut needed = rule.min_replicas.saturating_sub(have);

            for dest in &rule.candidate_nodes {
                if needed == 0 {
                    break;
                }
                if holders.iter().any(|r| &r.node_id == dest) {
                    continue;
                }
                if let Some(cap) = capacity_bytes.get(dest) {
                    let used =
                        self.bytes_resident_on(dest) + planned.get(dest).copied().unwrap_or(0);
                    if used + rule.content.bytes > *cap {
                        continue;
                    }
                }
                *planned.entry(dest.clone()).or_default() += rule.content.bytes;
                jobs.push(StagingJob {
                    content: rule.content.clone(),
                    source_node: source.node_id.clone(),
                    destination_node: dest.clone(),
                    target_tier: rule.preferred_tier,
                    bytes: rule.content.bytes,
                    pin: rule.pin,
                });
                needed -= 1;
            }
        }
        jobs
    }

    /// Like [`ReplicaCatalog::plan_staging_with_capacity`], with every job bound for a node in
    /// `needs_approval` held back until its owner approves the plan.
    pub fn plan_staging_gated(
        &self,
        rules: &[PlacementRule],
        capacity_bytes: &BTreeMap<NodeId, u64>,
        needs_approval: &BTreeSet<NodeId>,
    ) -> StagingPlan {
        let (awaiting_approval, ready): (Vec<StagingJob>, Vec<StagingJob>) = self
            .plan_staging_with_capacity(rules, capacity_bytes)
            .into_iter()
            .partition(|j| needs_approval.contains(&j.destination_node));
        StagingPlan {
            ready,
            awaiting_approval,
        }
    }

    /// Records the replica a completed staging job created.
    pub fn apply_staging(&mut self, job: &StagingJob, now_unix_ms: u64) {
        let mut replica = Replica::new(
            job.content.clone(),
            job.destination_node.clone(),
            job.target_tier,
            now_unix_ms,
        );
        replica.pinned = job.pin;
        self.record(replica);
    }

    /// Records every rule as already satisfied: each candidate node holds the content in the
    /// rule's preferred tier. For a catalog seeded from a placement whose staging has completed,
    /// or for tests that want locality to follow a plan.
    pub fn assume_placed(rules: &[PlacementRule], now_unix_ms: u64) -> Self {
        let mut catalog = Self::new();
        for rule in rules {
            for node in &rule.candidate_nodes {
                let mut replica =
                    Replica::new(rule.content.clone(), node.clone(), rule.preferred_tier, now_unix_ms);
                replica.pinned = rule.pin;
                catalog.record(replica);
            }
        }
        catalog
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContentKind;

    fn node(id: &str) -> NodeId {
        NodeId(id.to_string())
    }

    fn expert(name: &str, bytes: u64) -> ContentRef {
        ContentRef::named(ContentKind::ExpertBank, name, bytes)
    }

    #[test]
    fn records_dedupe_per_node_and_report_holders() {
        let mut cat = ReplicaCatalog::new();
        let e45 = expert("m/L25/E045", 6_300_000);
        cat.record(Replica::new(
            e45.clone(),
            node("nas"),
            ResidencyTier::NasVault,
            1,
        ));
        cat.record(Replica::new(
            e45.clone(),
            node("pc"),
            ResidencyTier::SystemRam,
            2,
        ));
        cat.record(Replica::new(
            e45.clone(),
            node("pc"),
            ResidencyTier::Vram,
            3,
        ));

        assert_eq!(cat.holders(&e45.content_id).len(), 2);
        assert_eq!(cat.replica_count(), 2);
        assert_eq!(
            cat.fastest_holder(&e45.content_id).unwrap().node_id,
            node("pc")
        );
        assert_eq!(
            cat.fastest_holder(&e45.content_id).unwrap().tier,
            ResidencyTier::Vram
        );
        assert!(cat.is_resident(&node("pc"), &e45.content_id));
        assert!(!cat.is_resident(&node("mac"), &e45.content_id));
        assert!(cat.remove(&node("nas"), &e45.content_id));
        assert_eq!(cat.holder_nodes(&e45.content_id), vec![node("pc")]);
        assert_eq!(cat.remove_node(&node("pc")), 1);
        assert_eq!(cat.content_count(), 0);
    }

    #[test]
    fn locality_is_the_resident_fraction_of_input_bytes() {
        let mut cat = ReplicaCatalog::new();
        let a = expert("a", 300);
        let b = expert("b", 100);
        cat.record(Replica::new(a.clone(), node("pc"), ResidencyTier::Vram, 1));
        let inputs = vec![a.clone(), b.clone()];
        assert!((cat.locality_score(&node("pc"), &inputs) - 0.75).abs() < 1e-9);
        assert_eq!(cat.locality_score(&node("mac"), &inputs), 0.0);
        assert_eq!(cat.locality_score(&node("mac"), &[]), 1.0);
        assert_eq!(cat.missing_inputs(&node("pc"), &inputs), vec![b]);
    }

    #[test]
    fn staging_plan_reaches_min_replicas_from_the_fastest_holder() {
        let mut cat = ReplicaCatalog::new();
        let e = expert("m/L25/E045", 6_300_000);
        cat.record(Replica::new(
            e.clone(),
            node("nas"),
            ResidencyTier::NasVault,
            1,
        ));
        let rule = PlacementRule {
            content: e.clone(),
            min_replicas: 3,
            preferred_tier: ResidencyTier::Vram,
            candidate_nodes: vec![node("pc"), node("mac"), node("laptop")],
            pin: true,
        };
        let jobs = cat.plan_staging(std::slice::from_ref(&rule));
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.source_node == node("nas")));
        assert_eq!(jobs[0].destination_node, node("pc"));
        assert_eq!(jobs[1].destination_node, node("mac"));

        for j in &jobs {
            cat.apply_staging(j, 5);
        }
        assert!(cat.plan_staging(&[rule]).is_empty());
        assert!(
            cat.holders(&e.content_id)
                .iter()
                .any(|r| r.pinned && r.node_id == node("pc"))
        );

        let unknown = PlacementRule {
            content: expert("nowhere", 1),
            min_replicas: 1,
            preferred_tier: ResidencyTier::SystemRam,
            candidate_nodes: vec![node("pc")],
            pin: false,
        };
        assert!(cat.plan_staging(&[unknown]).is_empty());
    }

    #[test]
    fn staging_plan_respects_destination_capacity() {
        let mut cat = ReplicaCatalog::new();
        let big = expert("big", 10_000);
        let small = expert("small", 100);
        cat.record(Replica::new(
            big.clone(),
            node("nas"),
            ResidencyTier::NasVault,
            1,
        ));
        cat.record(Replica::new(
            small.clone(),
            node("nas"),
            ResidencyTier::NasVault,
            1,
        ));
        let rules = vec![
            PlacementRule {
                content: big.clone(),
                min_replicas: 2,
                preferred_tier: ResidencyTier::Vram,
                candidate_nodes: vec![node("tiny"), node("pc")],
                pin: false,
            },
            PlacementRule {
                content: small,
                min_replicas: 2,
                preferred_tier: ResidencyTier::Vram,
                candidate_nodes: vec![node("tiny")],
                pin: false,
            },
        ];
        let mut cap = BTreeMap::new();
        cap.insert(node("tiny"), 500u64);
        let jobs = cat.plan_staging_with_capacity(&rules, &cap);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].destination_node, node("pc"));
        assert_eq!(jobs[0].content, big);
        assert_eq!(jobs[1].destination_node, node("tiny"));
    }

    #[test]
    fn gated_staging_holds_jobs_for_nodes_awaiting_approval() {
        let mut catalog = ReplicaCatalog::new();
        let bank = expert("L003/E007", 6_000_000);
        catalog.record(Replica::new(
            bank.clone(),
            node("vault"),
            ResidencyTier::NasVault,
            0,
        ));
        let rules = vec![PlacementRule {
            content: bank,
            min_replicas: 3,
            preferred_tier: ResidencyTier::SystemRam,
            candidate_nodes: vec![node("owned"), node("peer")],
            pin: false,
        }];
        let gated = catalog.plan_staging_gated(&rules, &BTreeMap::new(), &[node("peer")].into());
        assert_eq!(gated.ready.len(), 1);
        assert_eq!(gated.ready[0].destination_node, node("owned"));
        assert_eq!(gated.awaiting_approval.len(), 1);
        assert_eq!(gated.nodes_awaiting(), vec![node("peer")]);
        assert_eq!(gated.bytes_awaiting(), 6_000_000);
        assert!(!gated.is_clear());

        let approval = PlanApproval {
            node: node("peer"),
            model_id: "toy".into(),
            placement_digest: 42,
            approved_unix_ms: 1,
        };
        assert!(approval.covers("toy", 42));
        assert!(!approval.covers("toy", 43), "a changed plan asks again");
        let clear = catalog.plan_staging_gated(&rules, &BTreeMap::new(), &BTreeSet::new());
        assert!(clear.is_clear());
        assert_eq!(clear.ready.len(), 2);
    }

    #[test]
    fn eviction_offers_least_recently_used_unpinned_replicas() {
        let mut cat = ReplicaCatalog::new();
        cat.record(Replica::new(
            expert("old", 100),
            node("pc"),
            ResidencyTier::Vram,
            1,
        ));
        cat.record(
            Replica::new(expert("pinned", 100), node("pc"), ResidencyTier::Vram, 2).pinned(),
        );
        cat.record(Replica::new(
            expert("new", 100),
            node("pc"),
            ResidencyTier::Vram,
            3,
        ));
        cat.touch(&node("pc"), "old", 10);
        let victims = cat.evict_candidates(&node("pc"), 150);
        let names: Vec<_> = victims.iter().map(|c| c.content_id.as_str()).collect();
        assert_eq!(names, vec!["new", "old"]);
        assert_eq!(cat.bytes_resident_on(&node("pc")), 300);
    }
}
