//! Credit ledger: grid credit granted only for validated work, in the BOINC tradition.
//!
//! Credit is the accounting unit of the grid: a function of the validated unit's cost estimate
//! (FLOPs and bytes), not of time spent, so a slow node earns the same credit as a fast one for
//! the same work and an invalid result earns nothing. The supply engine in `kadmium-core` turns
//! credit into money or cloud credits; this ledger only keeps the books.

use kadmium_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{AppKind, WorkResultRecord, WorkUnit, WorkUnitId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreditPolicy {
    pub credits_per_pflop: f64,
    pub credits_per_gb_read: f64,
    pub credits_per_gb_moved: f64,
    /// Deducted from a node's recent average when one of its results is invalidated.
    pub invalid_penalty_credits: f64,
    /// Half-life of the recent-average decay.
    pub recent_half_life_ms: u64,
}

impl Default for CreditPolicy {
    fn default() -> Self {
        Self {
            credits_per_pflop: 100.0,
            credits_per_gb_read: 0.5,
            credits_per_gb_moved: 2.0,
            invalid_penalty_credits: 10.0,
            recent_half_life_ms: 7 * 24 * 3_600 * 1_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreditEntry {
    pub unit_id: WorkUnitId,
    pub node_id: NodeId,
    pub app: AppKind,
    pub credits: f64,
    pub validated_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCreditSummary {
    pub node_id: NodeId,
    pub total_credits: f64,
    pub validated_units: u64,
    pub invalid_units: u64,
    pub flops_total: f64,
    pub bytes_read_total: u64,
    pub bytes_moved_total: u64,
    /// Exponentially decayed credit, the node's recent contribution rate.
    pub recent_average_credit: f64,
    /// When the recent average was last decayed; `None` until the first entry.
    pub last_update_unix_ms: Option<u64>,
}

impl NodeCreditSummary {
    fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            total_credits: 0.0,
            validated_units: 0,
            invalid_units: 0,
            flops_total: 0.0,
            bytes_read_total: 0,
            bytes_moved_total: 0,
            recent_average_credit: 0.0,
            last_update_unix_ms: None,
        }
    }

    fn decay(&mut self, now: u64, half_life_ms: u64) {
        if let Some(last) = self.last_update_unix_ms
            && half_life_ms > 0
        {
            let elapsed = now.saturating_sub(last) as f64;
            let factor = 0.5f64.powf(elapsed / half_life_ms as f64);
            self.recent_average_credit *= factor;
        }
        self.last_update_unix_ms = Some(now);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreditLedger {
    policy: CreditPolicy,
    entries: Vec<CreditEntry>,
    nodes: BTreeMap<NodeId, NodeCreditSummary>,
}

impl CreditLedger {
    pub fn new(policy: CreditPolicy) -> Self {
        Self {
            policy,
            entries: Vec::new(),
            nodes: BTreeMap::new(),
        }
    }

    pub fn policy(&self) -> &CreditPolicy {
        &self.policy
    }

    /// Credit a unit is worth once validated, from its cost estimate.
    pub fn credit_for(&self, unit: &WorkUnit) -> f64 {
        let flops_credit = unit.cost.flops / 1e15 * self.policy.credits_per_pflop;
        let read_credit = unit.cost.bytes_read as f64 / 1e9 * self.policy.credits_per_gb_read;
        let moved_credit = unit.cost.bytes_moved as f64 / 1e9 * self.policy.credits_per_gb_moved;
        ((flops_credit + read_credit + moved_credit) * 1000.0).round() / 1000.0
    }

    /// Credits the node that produced the validated result. Returns the credit granted.
    pub fn credit_validated(
        &mut self,
        unit: &WorkUnit,
        result: &WorkResultRecord,
        now: u64,
    ) -> f64 {
        let credits = self.credit_for(unit);
        let half_life = self.policy.recent_half_life_ms;
        let summary = self
            .nodes
            .entry(result.node_id.clone())
            .or_insert_with(|| NodeCreditSummary::new(result.node_id.clone()));
        summary.decay(now, half_life);
        summary.total_credits += credits;
        summary.validated_units += 1;
        summary.flops_total += unit.cost.flops;
        summary.bytes_read_total += unit.cost.bytes_read;
        summary.bytes_moved_total += result.bytes_moved.max(unit.cost.bytes_moved);
        summary.recent_average_credit += credits;
        self.entries.push(CreditEntry {
            unit_id: unit.id.clone(),
            node_id: result.node_id.clone(),
            app: unit.app.clone(),
            credits,
            validated_unix_ms: now,
        });
        credits
    }

    /// Records an invalidated result: no credit, and the node's recent average takes the penalty.
    pub fn penalize_invalid(&mut self, node_id: &NodeId, now: u64) {
        let half_life = self.policy.recent_half_life_ms;
        let penalty = self.policy.invalid_penalty_credits;
        let summary = self
            .nodes
            .entry(node_id.clone())
            .or_insert_with(|| NodeCreditSummary::new(node_id.clone()));
        summary.decay(now, half_life);
        summary.invalid_units += 1;
        summary.recent_average_credit = (summary.recent_average_credit - penalty).max(0.0);
    }

    pub fn node(&self, node_id: &NodeId) -> Option<&NodeCreditSummary> {
        self.nodes.get(node_id)
    }

    /// Nodes by total credit, highest first.
    pub fn leaderboard(&self) -> Vec<&NodeCreditSummary> {
        let mut nodes: Vec<&NodeCreditSummary> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.total_credits
                .partial_cmp(&a.total_credits)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node_id.cmp(&b.node_id))
        });
        nodes
    }

    pub fn total_credits(&self) -> f64 {
        self.entries.iter().map(|e| e.credits).sum()
    }

    pub fn entries(&self) -> &[CreditEntry] {
        &self.entries
    }

    pub fn entries_for_node(&self, node_id: &NodeId) -> Vec<&CreditEntry> {
        self.entries
            .iter()
            .filter(|e| &e.node_id == node_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CostEstimate, InstanceId};

    fn unit(id: &str, flops: f64, read: u64, moved: u64) -> WorkUnit {
        WorkUnit::new(id, AppKind::ExpertSlice)
            .with_cost(CostEstimate::new(flops, read, moved, 1.0))
    }

    fn result(unit: &WorkUnit, node: &str, now: u64) -> WorkResultRecord {
        WorkResultRecord::new(
            unit.id.clone(),
            InstanceId(format!("{}#1", unit.id)),
            NodeId(node.to_string()),
            1,
            1.0,
            now,
        )
    }

    #[test]
    fn credit_follows_the_cost_estimate_and_only_validated_work_earns() {
        let mut ledger = CreditLedger::new(CreditPolicy::default());
        let u = unit("a", 2e15, 4_000_000_000, 500_000_000);
        assert_eq!(ledger.credit_for(&u), 200.0 + 2.0 + 1.0);
        let granted = ledger.credit_validated(&u, &result(&u, "pc", 1_000), 1_000);
        assert_eq!(granted, 203.0);
        ledger.penalize_invalid(&NodeId("peer".into()), 1_000);

        let pc = ledger.node(&NodeId("pc".into())).unwrap();
        assert_eq!(pc.validated_units, 1);
        assert_eq!(pc.total_credits, 203.0);
        let peer = ledger.node(&NodeId("peer".into())).unwrap();
        assert_eq!(peer.invalid_units, 1);
        assert_eq!(peer.total_credits, 0.0);
        assert_eq!(ledger.total_credits(), 203.0);
        assert_eq!(ledger.entries_for_node(&NodeId("pc".into())).len(), 1);
        assert_eq!(ledger.leaderboard()[0].node_id, NodeId("pc".into()));
    }

    #[test]
    fn recent_average_decays_with_the_half_life() {
        let policy = CreditPolicy {
            recent_half_life_ms: 1_000,
            ..CreditPolicy::default()
        };
        let mut ledger = CreditLedger::new(policy);
        let u = unit("a", 1e15, 0, 0);
        ledger.credit_validated(&u, &result(&u, "pc", 0), 0);
        let fresh = ledger
            .node(&NodeId("pc".into()))
            .unwrap()
            .recent_average_credit;
        assert_eq!(fresh, 100.0);

        let u2 = unit("b", 1e15, 0, 0);
        ledger.credit_validated(&u2, &result(&u2, "pc", 1_000), 1_000);
        let after = ledger.node(&NodeId("pc".into())).unwrap();
        assert!((after.recent_average_credit - 150.0).abs() < 1e-9);
        assert_eq!(after.total_credits, 200.0);

        ledger.penalize_invalid(&NodeId("pc".into()), 1_000);
        assert!(
            (ledger
                .node(&NodeId("pc".into()))
                .unwrap()
                .recent_average_credit
                - 140.0)
                .abs()
                < 1e-9
        );
    }
}
