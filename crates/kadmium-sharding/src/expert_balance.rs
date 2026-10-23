//! Load-proportional expert replication and placement, after DeepSeek's EPLB.
//!
//! EPLB (Expert Parallelism Load Balancer) takes the load every routed expert carried over a
//! window of steps, gives heavy experts extra copies (`replicate_experts`: each extra slot goes
//! to the expert with the highest load per copy), and packs the copies onto accelerators so
//! that the busiest one carries the least (`balanced_packing`: heaviest first, each to the
//! least-loaded pack with room). When the node count divides the expert groups it balances in
//! two levels, first across nodes and then across the GPUs of each node, so that a group's
//! traffic stays on one node's fast links. DeepInfer keeps a budget of extra expert slots per
//! rank (`num_extra_experts`, rounded so the slots divide evenly), collects loads over the
//! last `N` steps and reassigns every `M` steps.
//!
//! Here the same pieces run over a mesh of unequal machines. The load is the usage ledger's
//! heat per expert; the budget is the fast memory each node's offer leaves free after the trunks
//! and the primary copies, in expert-sized slots, capped at a quarter of a layer's experts; the
//! two levels are link segments (nodes that share a LAN or a fast link) and then the nodes of
//! the chosen segment. EPLB balances bytes because its ranks are identical; the nodes here are
//! not, so the objective is the busiest node's modelled read time per layer, each node's
//! expected bytes divided by the bandwidth of the tier they sit in, and the bytes are reported
//! beside it. The balancer runs in two modes: in full at planning time, and incrementally at a
//! re-pin boundary, where it makes at most a few changes and only when each clears a
//! hysteresis, which is the mode the existing re-pin boundary becomes. Every figure it reports
//! is modelled, under EPLB's assumption that the copies of an expert share its rows evenly,
//! which the step graph makes true for batches wider than one row.
//!
//! Left out, deliberately: expert groups and group-limited routing (Kadmium's models have no
//! groups, so the segment level packs copies rather than groups), the NVLink weight shuffle
//! (a replica here is a staging transfer the catalog records), the periodic reassignment timer
//! (a re-pin boundary is asked for explicitly, as before), and the measured streaming rates,
//! which the cost model applies on top of the same placement.

use kadmium_core::{HardwareNode, ModelSpec, NodeId, NodeTrustTier};
use kadmium_grid::{ResidencyTier, StagingJob};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{ExpertPlacementPlan, ExpertPopularity, ModelContentNames};

const GIB: f64 = 1_073_741_824.0;

/// Extra copies per layer at most, as a fraction of the layer's experts. DeepInfer runs decode
/// with one redundant expert per rank at EP64 over 256 experts (a quarter) and prefill with
/// four per rank at EP8 (an eighth); a quarter is the ceiling here.
pub const EPLB_MAX_REDUNDANCY_FRACTION: f64 = 0.25;

/// A node whose round trip to the coordinator is at most this sits on the coordinator's LAN
/// segment when nothing better is known.
pub const LAN_SEGMENT_MAX_LATENCY_MS: f64 = 2.5;

/// Which nodes share a fast link: the first level of the hierarchical packing. Derived from
/// the workers' LAN addresses where the coordinator knows them (one segment per `/24`), from
/// the capability vector's round trip otherwise (the LAN, or a remote segment per node), or
/// set explicitly.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LinkSegments {
    pub by_node: BTreeMap<NodeId, String>,
}

impl LinkSegments {
    pub fn derive(nodes: &[HardwareNode]) -> Self {
        Self::derive_with_lan_ips(nodes, &BTreeMap::new())
    }

    /// Segments from LAN addresses where known, round trips otherwise.
    pub fn derive_with_lan_ips(nodes: &[HardwareNode], lan_ips: &BTreeMap<NodeId, String>) -> Self {
        let by_node = nodes
            .iter()
            .map(|n| {
                let subnet = lan_ips
                    .get(&n.id)
                    .and_then(|ip| ip.rsplit_once('.').map(|(prefix, _)| prefix.to_string()))
                    .filter(|prefix| !prefix.is_empty());
                let key = match subnet {
                    Some(prefix) => format!("lan:{prefix}"),
                    None if n.network_latency_ms <= LAN_SEGMENT_MAX_LATENCY_MS => "lan".to_string(),
                    None => format!("remote:{}", n.id),
                };
                (n.id.clone(), key)
            })
            .collect();
        Self { by_node }
    }

    pub fn from_map(by_node: BTreeMap<NodeId, String>) -> Self {
        Self { by_node }
    }

    /// The segment of a node; a node the map does not know is a segment of its own.
    pub fn segment_of(&self, node: &NodeId) -> String {
        self.by_node
            .get(node)
            .cloned()
            .unwrap_or_else(|| node.0.clone())
    }

    /// Distinct segments, in name order.
    pub fn segments(&self) -> Vec<String> {
        let set: BTreeSet<&String> = self.by_node.values().collect();
        set.into_iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.segments().len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_node.is_empty()
    }
}

/// The extra expert slots the offers leave free, in expert-sized units of fast memory, and the
/// most of them any one layer may spend.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplicaBudget {
    /// Free slots per node, from the plan's fit against the offers.
    pub slots: BTreeMap<NodeId, u32>,
    /// Extra copies one layer may take at most.
    pub per_layer: u32,
}

impl ReplicaBudget {
    /// The budget a placed plan leaves: each node's offered fast memory minus what the trunks,
    /// KV reservations and primary copies took, in expert-sized slots; the per-layer share is
    /// the total spread over the layers, capped by [`EPLB_MAX_REDUNDANCY_FRACTION`].
    pub fn from_plan(plan: &ExpertPlacementPlan, model: &ModelSpec) -> Self {
        let expert_bytes = (model.single_expert_size_gb * GIB).max(1.0);
        let slots: BTreeMap<NodeId, u32> = plan
            .fit
            .per_node
            .iter()
            .map(|f| {
                (
                    f.node.clone(),
                    (f.fast_free_bytes as f64 / expert_bytes).floor() as u32,
                )
            })
            .filter(|(_, slots)| *slots > 0)
            .collect();
        let total: u32 = slots.values().sum();
        let layers = if model.is_moe {
            model.num_layers.max(1)
        } else {
            1
        };
        let cap = ((model.experts_per_layer as f64 * EPLB_MAX_REDUNDANCY_FRACTION).floor() as u32)
            .max(u32::from(model.experts_per_layer > 1));
        Self {
            slots,
            per_layer: total.div_ceil(layers).min(cap),
        }
    }

    pub fn total_slots(&self) -> u32 {
        self.slots.values().sum()
    }

    pub fn with_per_layer(mut self, per_layer: u32) -> Self {
        self.per_layer = per_layer;
        self
    }
}

/// How the balancer may change a placement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BalanceMode {
    /// Planning time: spend the budget wherever a copy lowers the busiest node.
    Full,
    /// A re-pin boundary: at most `max_changes` changes, each only when it lowers the layer's
    /// busiest node by more than `hysteresis` of its read time. A change is one copy added, or
    /// one cold copy dropped to make room for a hot one (two changes).
    Incremental { max_changes: usize, hysteresis: f64 },
}

/// One copy added to or dropped from a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicaChange {
    pub layer: u32,
    pub expert: u32,
    pub node: NodeId,
    pub tier: ResidencyTier,
    /// True for a copy added, false for one dropped.
    pub add: bool,
    /// The layer's busiest node before and after this change: its expected bytes and the
    /// modelled time to read them.
    pub busiest_bytes_before: f64,
    pub busiest_bytes_after: f64,
    pub busiest_ms_before: f64,
    pub busiest_ms_after: f64,
}

/// The busiest node of one layer before and after balancing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerBalance {
    pub layer: u32,
    pub busiest_node_before: Option<NodeId>,
    pub busiest_bytes_before: f64,
    pub busiest_ms_before: f64,
    pub busiest_node_after: Option<NodeId>,
    pub busiest_bytes_after: f64,
    pub busiest_ms_after: f64,
    pub replicas_added: u32,
    pub replicas_dropped: u32,
}

/// What the balancer did and what it expects, all modelled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BalanceReport {
    pub mode: BalanceMode,
    pub hierarchical: bool,
    pub segments: usize,
    pub budget_slots: u32,
    pub per_layer_budget: u32,
    pub layers: Vec<LayerBalance>,
    pub changes: Vec<ReplicaChange>,
    pub replicas_added: usize,
    pub replicas_dropped: usize,
    /// Sum over layers of the busiest node's expected bytes and modelled read time: the
    /// expert reads on the step's critical path.
    pub busiest_bytes_before: f64,
    pub busiest_bytes_after: f64,
    pub busiest_ms_before: f64,
    pub busiest_ms_after: f64,
    /// Always true: the figures assume an expert's copies share its rows evenly and read at
    /// their tier's nominal bandwidth.
    pub modelled: bool,
    pub rationale: String,
}

impl BalanceReport {
    /// The relief the balancer expects on the critical path, as a fraction of the read time
    /// before.
    pub fn relief(&self) -> f64 {
        if self.busiest_ms_before <= 0.0 {
            0.0
        } else {
            1.0 - self.busiest_ms_after / self.busiest_ms_before
        }
    }
}

/// A node as the balancer sees it.
#[derive(Debug, Clone)]
struct NodeSlot {
    node: NodeId,
    segment: String,
    fast_tier: ResidencyTier,
    boundary_ok: bool,
    free_slots: u32,
}

/// Expected bytes on a node for one layer and the modelled time to read them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct NodeLoad {
    bytes: f64,
    ms: f64,
}

/// A node's fast tier and its bandwidth.
fn fast_tier_of(node: &HardwareNode) -> (ResidencyTier, f64) {
    if node.memory.unified_mem_gb > 0.0 {
        (
            ResidencyTier::UnifiedMemory,
            node.memory.unified_mem_bw_gbps.max(1.0),
        )
    } else if node.memory.vram_gb > 0.0 {
        (ResidencyTier::Vram, node.memory.vram_bw_gbps.max(1.0))
    } else {
        (
            ResidencyTier::SystemRam,
            node.memory.system_ram_bw_gbps.max(1.0),
        )
    }
}

/// Nominal bandwidth of a tier on a node, in GB/s, as the cost model counts it before the
/// efficiency factor and measured rates.
pub fn tier_bandwidth_gbps(node: &HardwareNode, tier: ResidencyTier) -> f64 {
    match tier {
        ResidencyTier::Vram | ResidencyTier::UnifiedMemory => fast_tier_of(node).1,
        ResidencyTier::SystemRam => node.memory.system_ram_bw_gbps.max(1.0),
        ResidencyTier::LocalNvme => node.memory.nvme_read_bw_gbps.max(0.5),
        ResidencyTier::NasVault => 0.3,
    }
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

pub struct ExpertLoadBalancer;

impl ExpertLoadBalancer {
    /// EPLB's `replicate_experts`: `extra` further copies, each to the expert with the highest
    /// load per copy at that moment (ties to the lowest id). Returns the copy count per expert.
    pub fn replica_counts(loads: &[f64], extra: u32) -> Vec<u32> {
        let mut counts = vec![1u32; loads.len()];
        if loads.is_empty() {
            return counts;
        }
        for _ in 0..extra {
            let best = (0..loads.len())
                .map(|e| (loads[e] / counts[e] as f64, e))
                .fold(None, |acc: Option<(f64, usize)>, (per_copy, e)| match acc {
                    Some((best, _)) if best >= per_copy => acc,
                    _ => Some((per_copy, e)),
                })
                .map(|(_, e)| e);
            if let Some(e) = best {
                counts[e] += 1;
            }
        }
        counts
    }

    /// EPLB's `balanced_packing`: items (by load) go heaviest first, each to the pack with the
    /// least load that still has room. Returns the pack index per item, or `None` for an item
    /// that fit nowhere.
    pub fn balanced_packing(loads: &[f64], capacity: &[u32]) -> Vec<Option<usize>> {
        let mut order: Vec<usize> = (0..loads.len()).collect();
        order.sort_by(|a, b| {
            loads[*b]
                .partial_cmp(&loads[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        let mut pack_load = vec![0.0f64; capacity.len()];
        let mut used = vec![0u32; capacity.len()];
        let mut out = vec![None; loads.len()];
        for item in order {
            let pack = (0..capacity.len())
                .filter(|p| used[*p] < capacity[*p])
                .min_by(|a, b| {
                    pack_load[*a]
                        .partial_cmp(&pack_load[*b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(b))
                });
            if let Some(p) = pack {
                out[item] = Some(p);
                pack_load[p] += loads[item];
                used[p] += 1;
            }
        }
        out
    }

    /// Expected bytes each expert of `layer` reads per step under the heat: `min(1, k p_e)`
    /// times the expert's bytes, with `p_e` the expert's share of the layer's hits (uniform
    /// when the layer has none).
    pub fn expected_bytes(
        model: &ModelSpec,
        heat: &ExpertPopularity,
        layer: u32,
        experts: &[u32],
    ) -> Vec<f64> {
        let expert_bytes = model.single_expert_size_gb * GIB;
        let k = model.active_experts_per_token.max(1) as f64;
        let hits: Vec<f64> = experts
            .iter()
            .map(|e| heat.hits(layer, *e) as f64)
            .collect();
        let total: f64 = hits.iter().sum();
        experts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let p = if total > 0.0 {
                    hits[i] / total
                } else {
                    1.0 / experts.len().max(1) as f64
                };
                (k * p).min(1.0) * expert_bytes
            })
            .collect()
    }

    /// Expected bytes and read time per node for one layer when every expert's copies share
    /// its rows evenly: the primary reads from the tier the plan placed it in, a copy from the
    /// node's fast tier.
    fn node_loads(
        holders: &[Vec<NodeId>],
        loads: &[f64],
        tiers: &[ResidencyTier],
        bw: &dyn Fn(&NodeId, ResidencyTier) -> f64,
    ) -> BTreeMap<NodeId, NodeLoad> {
        let mut out: BTreeMap<NodeId, NodeLoad> = BTreeMap::new();
        for (i, nodes) in holders.iter().enumerate() {
            if nodes.is_empty() {
                continue;
            }
            let share = loads[i] / nodes.len() as f64;
            for (index, n) in nodes.iter().enumerate() {
                let tier = if index == 0 {
                    tiers[i]
                } else {
                    ResidencyTier::Vram
                };
                let entry = out.entry(n.clone()).or_default();
                entry.bytes += share;
                entry.ms += share / (bw(n, tier) * 1e9) * 1000.0;
            }
        }
        out
    }

    /// The node with the longest modelled read time (ties: more bytes, then the id).
    fn busiest(loads: &BTreeMap<NodeId, NodeLoad>) -> (Option<NodeId>, NodeLoad) {
        let mut best: Option<(&NodeId, NodeLoad)> = None;
        for (n, l) in loads {
            let better = match best {
                None => true,
                Some((_, b)) => {
                    l.ms > b.ms + 1e-12 || ((l.ms - b.ms).abs() <= 1e-12 && l.bytes > b.bytes)
                }
            };
            if better {
                best = Some((n, *l));
            }
        }
        match best {
            Some((n, l)) => (Some(n.clone()), l),
            None => (None, NodeLoad::default()),
        }
    }

    /// Runs the balancer over every MoE layer of `plan`, adding (and in incremental mode
    /// dropping) copies in place: holders, rules, replica count and the fit's free fast memory
    /// are updated. `budget.slots` are consumed as copies land.
    #[allow(clippy::too_many_arguments)]
    pub fn balance(
        plan: &mut ExpertPlacementPlan,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        heat: &ExpertPopularity,
        budget: &ReplicaBudget,
        segments: &LinkSegments,
        hierarchical: bool,
        mode: BalanceMode,
    ) -> BalanceReport {
        let expert_bytes = (model.single_expert_size_gb * GIB) as u64;
        let last_layer = model.num_layers.saturating_sub(1);
        let by_id: BTreeMap<&NodeId, &HardwareNode> = nodes.iter().map(|n| (&n.id, n)).collect();
        let bw = |node: &NodeId, tier: ResidencyTier| -> f64 {
            by_id
                .get(node)
                .map(|n| tier_bandwidth_gbps(n, tier))
                .unwrap_or(0.3)
        };
        let mut slots: Vec<NodeSlot> = nodes
            .iter()
            .filter_map(|n| {
                let free = budget.slots.get(&n.id).copied().unwrap_or(0);
                let (fast_tier, _) = fast_tier_of(n);
                (free > 0 || matches!(mode, BalanceMode::Incremental { .. })).then(|| NodeSlot {
                    node: n.id.clone(),
                    segment: segments.segment_of(&n.id),
                    fast_tier,
                    boundary_ok: n.trust_tier == NodeTrustTier::TrustedCore,
                    free_slots: free,
                })
            })
            .collect();
        let (max_changes, hysteresis) = match mode {
            BalanceMode::Full => (usize::MAX, 0.0),
            BalanceMode::Incremental {
                max_changes,
                hysteresis,
            } => (max_changes, hysteresis.max(0.0)),
        };
        let mut layers: Vec<u32> = plan.experts.iter().map(|h| h.layer).collect();
        layers.sort_unstable();
        layers.dedup();

        // The most skewed layers first (busiest over mean read time), so a scarce budget
        // relieves the worst imbalance.
        let layer_data = |plan: &ExpertPlacementPlan, layer: u32| {
            let idx: Vec<usize> = (0..plan.experts.len())
                .filter(|&i| plan.experts[i].layer == layer)
                .collect();
            let experts: Vec<u32> = idx.iter().map(|&i| plan.experts[i].expert).collect();
            let loads = Self::expected_bytes(model, heat, layer, &experts);
            let holders: Vec<Vec<NodeId>> = idx
                .iter()
                .map(|&i| plan.experts[i].holders.clone())
                .collect();
            let tiers: Vec<ResidencyTier> = idx.iter().map(|&i| plan.experts[i].tier).collect();
            (idx, experts, loads, holders, tiers)
        };
        let mut skew: Vec<(f64, u32)> = layers
            .iter()
            .map(|&layer| {
                let (_, _, loads, holders, tiers) = layer_data(plan, layer);
                let node_loads = Self::node_loads(&holders, &loads, &tiers, &bw);
                let (_, busiest) = Self::busiest(&node_loads);
                let mean = if node_loads.is_empty() {
                    0.0
                } else {
                    node_loads.values().map(|l| l.ms).sum::<f64>() / node_loads.len() as f64
                };
                (if mean > 0.0 { busiest.ms / mean } else { 0.0 }, layer)
            })
            .collect();
        skew.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });

        let mut changes: Vec<ReplicaChange> = Vec::new();
        let mut layer_reports: Vec<LayerBalance> = Vec::new();
        let mut total_changes = 0usize;

        for (_, layer) in skew {
            let (idx, experts, loads, mut holders, tiers) = layer_data(plan, layer);
            let boundary = layer == 0 || layer == last_layer;
            let owner_segment = plan
                .trunk_owner(layer)
                .map(|o| segments.segment_of(o))
                .unwrap_or_default();
            let (before_node, before) =
                Self::busiest(&Self::node_loads(&holders, &loads, &tiers, &bw));
            // The budget is cumulative, like DeepInfer's extra slots: copies the layer already
            // holds count against it, so a second pass adds nothing a first one filled.
            let existing: u32 = holders
                .iter()
                .map(|h| h.len().saturating_sub(1) as u32)
                .sum();
            let mut added = 0u32;
            let mut dropped = 0u32;

            while existing + added < budget.per_layer && total_changes < max_changes {
                let node_loads = Self::node_loads(&holders, &loads, &tiers, &bw);
                let (_, cur) = Self::busiest(&node_loads);
                if cur.ms <= 0.0 {
                    break;
                }
                // Every (expert, node) pair that could take a copy, scored by the busiest node's
                // read time afterwards; EPLB's load-per-copy order and then the segment level
                // break ties.
                let mut best: Option<(f64, f64, usize, usize, Option<usize>)> = None;
                for pos in 0..idx.len() {
                    let per_copy = loads[pos] / holders[pos].len().max(1) as f64;
                    for (s, slot) in slots.iter().enumerate() {
                        if holders[pos].contains(&slot.node) || (boundary && !slot.boundary_ok) {
                            continue;
                        }
                        // A copy to drop on this node when it has no free slot: the extra copy
                        // (never a primary) with the least load share, incremental mode only.
                        let drop: Option<usize> = if slot.free_slots > 0 {
                            None
                        } else if matches!(mode, BalanceMode::Incremental { .. })
                            && total_changes + 2 <= max_changes
                        {
                            (0..holders.len())
                                .filter(|&q| {
                                    q != pos
                                        && holders[q].len() > 1
                                        && holders[q][0] != slot.node
                                        && holders[q].contains(&slot.node)
                                })
                                .min_by(|&a, &b| {
                                    let la = loads[a] / holders[a].len() as f64;
                                    let lb = loads[b] / holders[b].len() as f64;
                                    la.partial_cmp(&lb)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                        .then(a.cmp(&b))
                                })
                        } else {
                            continue;
                        };
                        if slot.free_slots == 0 && drop.is_none() {
                            continue;
                        }
                        let mut trial = holders.clone();
                        if let Some(q) = drop {
                            trial[q].retain(|n| n != &slot.node);
                        }
                        trial[pos].push(slot.node.clone());
                        let (_, new) =
                            Self::busiest(&Self::node_loads(&trial, &loads, &tiers, &bw));
                        let gain = cur.ms - new.ms;
                        let better = match &best {
                            None => true,
                            Some((g, pc, _, bs, _)) => {
                                gain > *g + 1e-12
                                    || ((gain - g).abs() <= 1e-12
                                        && (per_copy > *pc + 1e-9
                                            || ((per_copy - pc).abs() <= 1e-9
                                                && Self::segment_rank(
                                                    &slots[s],
                                                    &node_loads,
                                                    &slots,
                                                    &owner_segment,
                                                    hierarchical,
                                                ) < Self::segment_rank(
                                                    &slots[*bs],
                                                    &node_loads,
                                                    &slots,
                                                    &owner_segment,
                                                    hierarchical,
                                                ))))
                            }
                        };
                        if better {
                            best = Some((gain, per_copy, pos, s, drop));
                        }
                    }
                }
                let Some((gain, _, pos, s, drop)) = best else {
                    break;
                };
                if gain <= 0.0 || gain <= hysteresis * cur.ms {
                    break;
                }
                let node = slots[s].node.clone();
                let tier = slots[s].fast_tier;
                if let Some(q) = drop {
                    holders[q].retain(|n| n != &node);
                    changes.push(ReplicaChange {
                        layer,
                        expert: experts[q],
                        node: node.clone(),
                        tier,
                        add: false,
                        busiest_bytes_before: cur.bytes.round(),
                        busiest_bytes_after: cur.bytes.round(),
                        busiest_ms_before: round4(cur.ms),
                        busiest_ms_after: round4(cur.ms),
                    });
                    dropped += 1;
                    total_changes += 1;
                    slots[s].free_slots += 1;
                }
                holders[pos].push(node.clone());
                slots[s].free_slots -= 1;
                let (_, new) = Self::busiest(&Self::node_loads(&holders, &loads, &tiers, &bw));
                changes.push(ReplicaChange {
                    layer,
                    expert: experts[pos],
                    node,
                    tier,
                    add: true,
                    busiest_bytes_before: cur.bytes.round(),
                    busiest_bytes_after: new.bytes.round(),
                    busiest_ms_before: round4(cur.ms),
                    busiest_ms_after: round4(new.ms),
                });
                added += 1;
                total_changes += 1;
            }

            let (after_node, after) =
                Self::busiest(&Self::node_loads(&holders, &loads, &tiers, &bw));
            for (pos, &i) in idx.iter().enumerate() {
                plan.experts[i].holders = holders[pos].clone();
            }
            layer_reports.push(LayerBalance {
                layer,
                busiest_node_before: before_node,
                busiest_bytes_before: before.bytes.round(),
                busiest_ms_before: round4(before.ms),
                busiest_node_after: after_node,
                busiest_bytes_after: after.bytes.round(),
                busiest_ms_after: round4(after.ms),
                replicas_added: added,
                replicas_dropped: dropped,
            });
        }
        layer_reports.sort_by_key(|l| l.layer);

        // Rules and the fit follow the holders.
        let gb = expert_bytes as f64 / GIB;
        for change in &changes {
            let content_id =
                ModelContentNames::expert_id(&plan.model_id, change.layer, change.expert);
            if let Some(rule) = plan
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
            if let Some(fit) = plan.fit.per_node.iter_mut().find(|f| f.node == change.node) {
                if change.add {
                    fit.fast_used_gb = ((fit.fast_used_gb + gb) * 100.0).round() / 100.0;
                    fit.fast_free_bytes = fit.fast_free_bytes.saturating_sub(expert_bytes);
                } else {
                    fit.fast_used_gb = ((fit.fast_used_gb - gb).max(0.0) * 100.0).round() / 100.0;
                    fit.fast_free_bytes = fit.fast_free_bytes.saturating_add(expert_bytes);
                }
            }
            let used = plan
                .fast_memory_used_gb
                .entry(change.node.clone())
                .or_default();
            *used = if change.add {
                ((*used + gb) * 100.0).round() / 100.0
            } else {
                ((*used - gb).max(0.0) * 100.0).round() / 100.0
            };
        }
        plan.replicated_experts = plan
            .experts
            .iter()
            .map(|h| h.holders.len().saturating_sub(1))
            .sum();

        let replicas_added = changes.iter().filter(|c| c.add).count();
        let replicas_dropped = changes.iter().filter(|c| !c.add).count();
        let busiest_bytes_before: f64 = layer_reports.iter().map(|l| l.busiest_bytes_before).sum();
        let busiest_bytes_after: f64 = layer_reports.iter().map(|l| l.busiest_bytes_after).sum();
        let busiest_ms_before: f64 = layer_reports.iter().map(|l| l.busiest_ms_before).sum();
        let busiest_ms_after: f64 = layer_reports.iter().map(|l| l.busiest_ms_after).sum();
        let rationale = if replicas_added == 0 && replicas_dropped == 0 {
            match mode {
                BalanceMode::Full => format!(
                    "no copy lowers the busiest node of any layer within {} free slot{} and {} per layer",
                    budget.total_slots(),
                    if budget.total_slots() == 1 { "" } else { "s" },
                    budget.per_layer
                ),
                BalanceMode::Incremental { hysteresis, .. } => format!(
                    "no copy lowers a layer's busiest node by more than {:.0}% of its read time",
                    hysteresis * 100.0
                ),
            }
        } else {
            format!(
                "{replicas_added} cop{} added{} across {} segment{}: the busiest node's expert reads over the step fall from {:.1} MB in {:.4} ms to {:.1} MB in {:.4} ms, modelled with each expert's copies sharing its rows evenly at their tiers' nominal bandwidth",
                if replicas_added == 1 { "y" } else { "ies" },
                if replicas_dropped > 0 {
                    format!(", {replicas_dropped} dropped")
                } else {
                    String::new()
                },
                segments.len().max(1),
                if segments.len() == 1 { "" } else { "s" },
                busiest_bytes_before / 1e6,
                busiest_ms_before,
                busiest_bytes_after / 1e6,
                busiest_ms_after
            )
        };
        BalanceReport {
            mode,
            hierarchical,
            segments: segments.len(),
            budget_slots: budget.total_slots(),
            per_layer_budget: budget.per_layer,
            layers: layer_reports,
            changes,
            replicas_added,
            replicas_dropped,
            busiest_bytes_before: busiest_bytes_before.round(),
            busiest_bytes_after: busiest_bytes_after.round(),
            busiest_ms_before: round4(busiest_ms_before),
            busiest_ms_after: round4(busiest_ms_after),
            modelled: true,
            rationale,
        }
    }

    /// The first level of the packing: lower is better. Hierarchical, a node in the segment
    /// with the least read time for this layer ranks first, the trunk owner's segment breaking
    /// ties; flat, every node ranks the same and the node's own load decides.
    fn segment_rank(
        slot: &NodeSlot,
        node_loads: &BTreeMap<NodeId, NodeLoad>,
        slots: &[NodeSlot],
        owner_segment: &str,
        hierarchical: bool,
    ) -> (f64, u8, f64, String) {
        let own = node_loads.get(&slot.node).map_or(0.0, |l| l.ms);
        if !hierarchical {
            return (0.0, 0, own, slot.node.0.clone());
        }
        let segment_load: f64 = slots
            .iter()
            .filter(|s| s.segment == slot.segment)
            .map(|s| node_loads.get(&s.node).map_or(0.0, |l| l.ms))
            .sum();
        (
            segment_load,
            u8::from(slot.segment != owner_segment),
            own,
            slot.node.0.clone(),
        )
    }

    /// The staging a set of changes implies: every copy added is a transfer from the expert's
    /// primary holder into the node's fast tier, pinned. A drop is an eviction the catalog
    /// records; it moves no bytes.
    pub fn staging_for(
        plan: &ExpertPlacementPlan,
        changes: &[ReplicaChange],
        expert_bytes: u64,
    ) -> Vec<StagingJob> {
        changes
            .iter()
            .filter(|c| c.add)
            .filter_map(|c| {
                let primary = plan.holders_of(c.layer, c.expert).first()?.clone();
                Some(StagingJob {
                    content: ModelContentNames::expert(
                        &plan.model_id,
                        c.layer,
                        c.expert,
                        expert_bytes,
                    ),
                    source_node: primary,
                    destination_node: c.node.clone(),
                    target_tier: c.tier,
                    bytes: expert_bytes,
                    pin: true,
                })
            })
            .collect()
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
    fn replica_counts_follow_load_per_copy_and_packing_is_heaviest_first_to_the_least_loaded() {
        // EPLB's rule: the heaviest expert takes the extra copies until its load per copy drops
        // below the next expert's.
        let loads = [10.0, 4.0, 3.0, 1.0];
        assert_eq!(
            ExpertLoadBalancer::replica_counts(&loads, 0),
            vec![1, 1, 1, 1]
        );
        assert_eq!(
            ExpertLoadBalancer::replica_counts(&loads, 1),
            vec![2, 1, 1, 1]
        );
        assert_eq!(
            ExpertLoadBalancer::replica_counts(&loads, 2),
            vec![3, 1, 1, 1]
        );
        assert_eq!(
            ExpertLoadBalancer::replica_counts(&loads, 3),
            vec![3, 2, 1, 1]
        );
        assert_eq!(
            ExpertLoadBalancer::replica_counts(&[], 3),
            Vec::<u32>::new()
        );
        // Equal loads: the lowest id takes the copy.
        assert_eq!(
            ExpertLoadBalancer::replica_counts(&[2.0, 2.0], 1),
            vec![2, 1]
        );

        let packs = ExpertLoadBalancer::balanced_packing(&[5.0, 1.0, 4.0, 2.0], &[2, 2]);
        // 5 -> pack 0, 4 -> pack 1, 2 -> pack 1 (load 4 < 5), 1 -> pack 0.
        assert_eq!(packs, vec![Some(0), Some(0), Some(1), Some(1)]);
        let short = ExpertLoadBalancer::balanced_packing(&[5.0, 1.0, 4.0], &[1, 1]);
        assert_eq!(short, vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn segments_come_from_lan_addresses_then_round_trips() {
        let mut nodes = sample_personal_cluster();
        nodes.push(kadmium_core::sample_cloud_burst_gpu_node());
        let derived = LinkSegments::derive(&nodes);
        assert_eq!(derived.segment_of(&nodes[0].id), "lan");
        assert_eq!(
            derived.segment_of(&nodes[7].id),
            format!("remote:{}", nodes[7].id),
            "a 14.5 ms round trip is not the LAN"
        );
        assert_eq!(derived.len(), 2);
        let ips: BTreeMap<NodeId, String> = [
            (nodes[0].id.clone(), "192.168.1.10".to_string()),
            (nodes[1].id.clone(), "192.168.1.11".to_string()),
            (nodes[2].id.clone(), "10.0.0.5".to_string()),
        ]
        .into_iter()
        .collect();
        let by_ip = LinkSegments::derive_with_lan_ips(&nodes, &ips);
        assert_eq!(by_ip.segment_of(&nodes[0].id), "lan:192.168.1");
        assert_eq!(by_ip.segment_of(&nodes[1].id), "lan:192.168.1");
        assert_eq!(by_ip.segment_of(&nodes[2].id), "lan:10.0.0");
        assert_eq!(by_ip.segment_of(&nodes[3].id), "lan");
        assert_eq!(
            LinkSegments::default().segment_of(&NodeId("x".into())),
            "x",
            "an unknown node is a segment of its own"
        );
    }

    #[test]
    fn the_balancer_lowers_the_busiest_node_and_stops_at_the_budget() {
        let nodes = sample_personal_cluster();
        let model = toy_moe();
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 1.0);
        let mut plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        assert_eq!(plan.replicated_experts, 0, "no heat, no copies");
        let budget = ReplicaBudget::from_plan(&plan, &model);
        assert!(budget.total_slots() > 0, "{budget:?}");
        assert_eq!(budget.per_layer, 2, "a quarter of eight experts");
        let segments = LinkSegments::derive(&nodes);

        let report = ExpertLoadBalancer::balance(
            &mut plan,
            &model,
            &nodes,
            &heat,
            &budget,
            &segments,
            true,
            BalanceMode::Full,
        );
        assert!(report.modelled);
        assert!(report.replicas_added > 0, "{}", report.rationale);
        assert!(report.replicas_added <= 2 * model.num_layers as usize);
        assert!(
            report.busiest_ms_after < report.busiest_ms_before,
            "{}",
            report.rationale
        );
        assert!(report.relief() > 0.0);
        assert_eq!(report.layers.len(), model.num_layers as usize);
        for layer in &report.layers {
            assert!(layer.busiest_ms_after <= layer.busiest_ms_before);
            assert!(layer.replicas_added <= budget.per_layer);
        }
        for change in &report.changes {
            assert!(
                change.busiest_ms_after < change.busiest_ms_before,
                "{change:?}"
            );
        }
        assert_eq!(plan.replicated_experts, report.replicas_added);
        // The hottest expert of layer 0 (expert 0 under this skew) got a copy.
        let hottest = plan
            .experts
            .iter()
            .find(|h| h.layer == 0 && h.expert == 0)
            .unwrap();
        assert!(hottest.holders.len() >= 2, "{:?}", hottest.holders);
        // Rules and the fit followed the copies.
        for change in report.changes.iter().filter(|c| c.add) {
            let rule = plan
                .rules
                .iter()
                .find(|r| {
                    r.content.content_id
                        == ModelContentNames::expert_id(&model.id, change.layer, change.expert)
                })
                .unwrap();
            assert!(rule.candidate_nodes.contains(&change.node));
            assert_eq!(rule.min_replicas as usize, rule.candidate_nodes.len());
            assert!(change.tier.is_fast());
        }
        let jobs = ExpertLoadBalancer::staging_for(&plan, &report.changes, 10_000);
        assert_eq!(jobs.len(), report.replicas_added);
        assert!(jobs.iter().all(|j| j.pin && j.target_tier.is_fast()));
        // Balancing again changes nothing: every copy that would help has landed.
        let leftover = ReplicaBudget::from_plan(&plan, &model);
        let again = ExpertLoadBalancer::balance(
            &mut plan,
            &model,
            &nodes,
            &heat,
            &leftover,
            &segments,
            true,
            BalanceMode::Full,
        );
        assert_eq!(again.replicas_added, 0, "{}", again.rationale);
    }

    #[test]
    fn copies_go_where_the_read_time_falls_not_merely_where_bytes_do() {
        // One fast node holds everything; the only free slots are on a node whose fast memory
        // is thirty times slower. Half of a hot expert's rows would take longer there than the
        // whole layer takes on the fast node, so the hot experts stay put; the balancer finds
        // only the small relief of moving half of a cold expert, which a boundary's hysteresis
        // rejects as not worth a transfer.
        let cluster = sample_personal_cluster();
        let mut fast = cluster[0].clone();
        fast.memory.vram_gb = 64.0;
        let mut slow = cluster[5].clone();
        slow.memory.unified_mem_bw_gbps = fast.memory.vram_bw_gbps / 30.0;
        slow.roles = fast.roles.clone();
        let nodes = vec![fast.clone(), slow.clone()];
        let model = toy_moe();
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 1.0);
        let mut plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        for home in plan.experts.iter_mut() {
            home.holders = vec![fast.id.clone()];
            home.tier = ResidencyTier::Vram;
        }
        let budget = ReplicaBudget {
            slots: [(slow.id.clone(), 8u32)].into_iter().collect(),
            per_layer: 2,
        };
        let segments = LinkSegments::derive(&nodes);
        let mut full = plan.clone();
        let report = ExpertLoadBalancer::balance(
            &mut full,
            &model,
            &nodes,
            &heat,
            &budget,
            &segments,
            true,
            BalanceMode::Full,
        );
        for change in &report.changes {
            let experts: Vec<u32> = (0..model.experts_per_layer).collect();
            let loads = ExpertLoadBalancer::expected_bytes(&model, &heat, change.layer, &experts);
            let hottest = loads.iter().cloned().fold(0.0, f64::max);
            assert!(
                loads[change.expert as usize] < hottest / 4.0,
                "only a cold expert's rows are worth sending to a slow node: {change:?}"
            );
        }
        for layer in &report.layers {
            assert_eq!(layer.busiest_node_after.as_ref(), Some(&fast.id));
        }
        assert!(report.relief() < 0.05, "{}", report.rationale);
        let mut boundary = plan.clone();
        let gated = ExpertLoadBalancer::balance(
            &mut boundary,
            &model,
            &nodes,
            &heat,
            &budget,
            &segments,
            true,
            BalanceMode::Incremental {
                max_changes: 4,
                hysteresis: 0.05,
            },
        );
        assert_eq!(gated.replicas_added, 0, "{}", gated.rationale);

        // Give the slow node the fast node's bandwidth and the hot experts' copies land there.
        let mut peer = slow.clone();
        peer.memory.unified_mem_bw_gbps = fast.memory.vram_bw_gbps;
        let nodes = vec![fast.clone(), peer.clone()];
        let report = ExpertLoadBalancer::balance(
            &mut plan,
            &model,
            &nodes,
            &heat,
            &budget,
            &LinkSegments::derive(&nodes),
            true,
            BalanceMode::Full,
        );
        assert!(report.replicas_added > 0, "{}", report.rationale);
        assert!(report.changes.iter().all(|c| c.node == peer.id));
        assert!(report.relief() > 0.2, "{}", report.rationale);
        let first = &report.changes[0];
        let experts: Vec<u32> = (0..model.experts_per_layer).collect();
        let loads = ExpertLoadBalancer::expected_bytes(&model, &heat, first.layer, &experts);
        assert_eq!(
            loads[first.expert as usize],
            loads.iter().cloned().fold(0.0, f64::max),
            "the hottest expert of the most skewed layer goes first"
        );
    }

    #[test]
    fn incremental_mode_is_capped_and_gated_by_hysteresis() {
        let nodes = sample_personal_cluster();
        let model = toy_moe();
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 1.0);
        let base =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        let budget = ReplicaBudget::from_plan(&base, &model);
        let segments = LinkSegments::derive(&nodes);

        let mut capped = base.clone();
        let report = ExpertLoadBalancer::balance(
            &mut capped,
            &model,
            &nodes,
            &heat,
            &budget,
            &segments,
            true,
            BalanceMode::Incremental {
                max_changes: 2,
                hysteresis: 0.25,
            },
        );
        assert!(report.changes.len() <= 2, "{:?}", report.changes);
        assert!(report.replicas_added >= 1, "{}", report.rationale);
        for change in &report.changes {
            assert!(
                change.busiest_ms_before - change.busiest_ms_after
                    > 0.25 * change.busiest_ms_before - 0.011,
                "{change:?}"
            );
        }

        let mut strict = base.clone();
        let nothing = ExpertLoadBalancer::balance(
            &mut strict,
            &model,
            &nodes,
            &heat,
            &budget,
            &segments,
            true,
            BalanceMode::Incremental {
                max_changes: 4,
                hysteresis: 0.99,
            },
        );
        assert_eq!(nothing.replicas_added, 0);
        assert!(nothing.rationale.contains("99%"));

        // No budget: nothing moves in full mode either.
        let mut starved = base.clone();
        let empty = ReplicaBudget::default();
        let none = ExpertLoadBalancer::balance(
            &mut starved,
            &model,
            &nodes,
            &heat,
            &empty,
            &segments,
            false,
            BalanceMode::Full,
        );
        assert_eq!(none.replicas_added, 0);
        assert_eq!(starved.digest(), base.digest());
    }

    #[test]
    fn boundary_layers_only_gain_copies_on_owned_hardware_and_the_objective_is_segment_blind() {
        let mut nodes = sample_personal_cluster();
        nodes[2].trust_tier = NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
        nodes[3].trust_tier = NodeTrustTier::VerifiedPeerMiddleLayersOnly;
        let model = toy_moe();
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 1.0);
        let mut plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        let budget = ReplicaBudget::from_plan(&plan, &model);
        // Two segments: the laptops on their own link.
        let mut map: BTreeMap<NodeId, String> = nodes
            .iter()
            .map(|n| (n.id.clone(), "lan".to_string()))
            .collect();
        map.insert(nodes[2].id.clone(), "laptops".into());
        map.insert(nodes[3].id.clone(), "laptops".into());
        let segments = LinkSegments::from_map(map);
        assert_eq!(segments.len(), 2);
        let report = ExpertLoadBalancer::balance(
            &mut plan,
            &model,
            &nodes,
            &heat,
            &budget,
            &segments,
            true,
            BalanceMode::Full,
        );
        assert!(report.hierarchical);
        assert_eq!(report.segments, 2);
        let last = model.num_layers - 1;
        for change in report
            .changes
            .iter()
            .filter(|c| c.layer == 0 || c.layer == last)
        {
            let node = nodes.iter().find(|n| n.id == change.node).unwrap();
            assert_eq!(node.trust_tier, NodeTrustTier::TrustedCore, "{change:?}");
        }
        let mut flat_plan =
            ExpertPlacementPlanner::plan(&model, &nodes, ExecutionStrategy::ExpertParallel, None);
        let flat = ExpertLoadBalancer::balance(
            &mut flat_plan,
            &model,
            &nodes,
            &heat,
            &budget,
            &segments,
            false,
            BalanceMode::Full,
        );
        assert!(!flat.hierarchical);
        assert_eq!(
            flat.busiest_ms_after, report.busiest_ms_after,
            "the objective is the same; the segment level only orders ties"
        );
    }
}
