//! Work graphs: DAGs of work units joined by data dependencies.
//!
//! A pipeline decode step is a chain; expert parallelism is a fan-out and fan-in; a Dream-RSI
//! replay batch is a bag with no edges; a training step is a chain ending in a reduce. The
//! scheduler consumes the ready set, which is the wavefront: every unit whose prerequisites
//! have all been validated.
//!
//! Every unit a graph holds carries a graph-qualified id (`<graph id>/<name>`), so the steps of
//! one session and the graphs of several sessions can share one queue. A bare name handed to any
//! method here is read as the name of this graph's unit.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{GridError, WorkUnit, WorkUnitId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkGraph {
    pub id: String,
    units: BTreeMap<WorkUnitId, WorkUnit>,
    /// unit -> its prerequisites
    prereqs: BTreeMap<WorkUnitId, BTreeSet<WorkUnitId>>,
    completed: BTreeSet<WorkUnitId>,
    /// Insertion order, so ready sets and wavefronts are deterministic.
    order: Vec<WorkUnitId>,
}

impl WorkGraph {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            units: BTreeMap::new(),
            prereqs: BTreeMap::new(),
            completed: BTreeSet::new(),
            order: Vec::new(),
        }
    }

    /// The id a unit name or id has inside this graph: `<graph id>/<name>`, unchanged when it
    /// is already qualified by this graph.
    pub fn qualify(&self, id: &WorkUnitId) -> WorkUnitId {
        id.in_graph(&self.id)
    }

    /// Adds or replaces a unit, qualifying its id by this graph and tagging it with the graph's
    /// id. A unit added under a bare name is thereafter known as `<graph id>/<name>`.
    pub fn add_unit(&mut self, mut unit: WorkUnit) -> &mut Self {
        unit.id = self.qualify(&unit.id);
        unit.graph_id = Some(self.id.clone());
        if !self.units.contains_key(&unit.id) {
            self.order.push(unit.id.clone());
        }
        self.prereqs.entry(unit.id.clone()).or_default();
        self.units.insert(unit.id.clone(), unit);
        self
    }

    /// Declares that `unit` cannot start until `prereq` has been validated.
    pub fn add_dependency(
        &mut self,
        unit: &WorkUnitId,
        prereq: &WorkUnitId,
    ) -> Result<(), GridError> {
        let unit = self.qualify(unit);
        let prereq = self.qualify(prereq);
        if !self.units.contains_key(&unit) {
            return Err(GridError::UnknownUnit(unit));
        }
        if !self.units.contains_key(&prereq) {
            return Err(GridError::UnknownUnit(prereq));
        }
        if unit == prereq || self.depends_on(&prereq, &unit) {
            return Err(GridError::DependencyCycle { unit, prereq });
        }
        self.prereqs.entry(unit).or_default().insert(prereq);
        Ok(())
    }

    /// True when `from` transitively depends on `target`.
    fn depends_on(&self, from: &WorkUnitId, target: &WorkUnitId) -> bool {
        let mut stack = vec![from.clone()];
        let mut seen = BTreeSet::new();
        while let Some(current) = stack.pop() {
            if &current == target {
                return true;
            }
            if !seen.insert(current.clone()) {
                continue;
            }
            if let Some(ps) = self.prereqs.get(&current) {
                stack.extend(ps.iter().cloned());
            }
        }
        false
    }

    pub fn unit(&self, id: &WorkUnitId) -> Option<&WorkUnit> {
        self.units.get(&self.qualify(id))
    }

    /// Units in insertion order.
    pub fn units(&self) -> impl Iterator<Item = &WorkUnit> {
        self.order.iter().filter_map(|id| self.units.get(id))
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    pub fn prerequisites(&self, id: &WorkUnitId) -> Vec<WorkUnitId> {
        self.prereqs
            .get(&self.qualify(id))
            .map(|ps| ps.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn dependents(&self, id: &WorkUnitId) -> Vec<WorkUnitId> {
        let id = self.qualify(id);
        self.order
            .iter()
            .filter(|u| self.prereqs.get(*u).is_some_and(|ps| ps.contains(&id)))
            .cloned()
            .collect()
    }

    /// The units nothing in the graph depends on: where a step's final activation comes out.
    pub fn sinks(&self) -> Vec<&WorkUnit> {
        self.units()
            .filter(|u| self.dependents(&u.id).is_empty())
            .collect()
    }

    /// The wavefront: units not yet completed whose prerequisites all are.
    pub fn ready(&self) -> Vec<&WorkUnit> {
        self.units()
            .filter(|u| !self.completed.contains(&u.id))
            .filter(|u| {
                self.prereqs
                    .get(&u.id)
                    .map(|ps| ps.iter().all(|p| self.completed.contains(p)))
                    .unwrap_or(true)
            })
            .collect()
    }

    pub fn mark_completed(&mut self, id: &WorkUnitId) -> Result<(), GridError> {
        let id = self.qualify(id);
        if !self.units.contains_key(&id) {
            return Err(GridError::UnknownUnit(id));
        }
        self.completed.insert(id);
        Ok(())
    }

    pub fn is_completed(&self, id: &WorkUnitId) -> bool {
        self.completed.contains(&self.qualify(id))
    }

    pub fn is_complete(&self) -> bool {
        self.units.keys().all(|id| self.completed.contains(id))
    }

    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Topological levels: every unit in level `k` depends only on units in levels `< k`.
    pub fn wavefronts(&self) -> Result<Vec<Vec<WorkUnitId>>, GridError> {
        let mut indegree: BTreeMap<WorkUnitId, usize> = self
            .order
            .iter()
            .map(|id| (id.clone(), self.prereqs.get(id).map_or(0, BTreeSet::len)))
            .collect();
        let mut dependents: BTreeMap<WorkUnitId, Vec<WorkUnitId>> = BTreeMap::new();
        for (unit, ps) in &self.prereqs {
            for p in ps {
                dependents.entry(p.clone()).or_default().push(unit.clone());
            }
        }

        let mut levels = Vec::new();
        let mut frontier: VecDeque<WorkUnitId> = self
            .order
            .iter()
            .filter(|id| indegree.get(*id) == Some(&0))
            .cloned()
            .collect();
        let mut placed = 0usize;

        while !frontier.is_empty() {
            let level: Vec<WorkUnitId> = frontier.drain(..).collect();
            placed += level.len();
            let mut next = Vec::new();
            for id in &level {
                if let Some(ds) = dependents.get(id) {
                    for d in ds {
                        let entry = indegree.get_mut(d).expect("dependent is a known unit");
                        *entry -= 1;
                        if *entry == 0 {
                            next.push(d.clone());
                        }
                    }
                }
            }
            // Keep insertion order inside a level for determinism.
            next.sort_by_key(|id| self.order.iter().position(|o| o == id));
            frontier.extend(next);
            levels.push(level);
        }

        if placed != self.units.len() {
            let stuck = self
                .order
                .iter()
                .find(|id| indegree.get(*id).is_some_and(|d| *d > 0))
                .cloned()
                .unwrap_or_else(|| WorkUnitId::new("?"));
            return Err(GridError::DependencyCycle {
                unit: stuck.clone(),
                prereq: stuck,
            });
        }
        Ok(levels)
    }

    /// Longest path through the graph by estimated wall time: the step time if every ready unit
    /// ran in parallel.
    pub fn critical_path_ms(&self) -> f64 {
        let Ok(levels) = self.wavefronts() else {
            return self.total_est_wall_ms();
        };
        let mut finish: BTreeMap<WorkUnitId, f64> = BTreeMap::new();
        for level in levels {
            for id in level {
                let own = self.units.get(&id).map_or(0.0, |u| u.cost.est_wall_ms);
                let start = self
                    .prereqs
                    .get(&id)
                    .map(|ps| {
                        ps.iter()
                            .map(|p| finish.get(p).copied().unwrap_or(0.0))
                            .fold(0.0, f64::max)
                    })
                    .unwrap_or(0.0);
                finish.insert(id, start + own);
            }
        }
        finish.values().copied().fold(0.0, f64::max)
    }

    /// Sum of estimated wall time: the step time if everything ran serially.
    pub fn total_est_wall_ms(&self) -> f64 {
        self.units.values().map(|u| u.cost.est_wall_ms).sum()
    }

    /// A chain: each unit depends on the one before it (a pipeline).
    pub fn chain(id: impl Into<String>, units: Vec<WorkUnit>) -> Self {
        let mut graph = Self::new(id);
        let mut previous: Option<WorkUnitId> = None;
        for unit in units {
            let current = unit.id.clone();
            graph.add_unit(unit);
            if let Some(prev) = previous {
                graph
                    .add_dependency(&current, &prev)
                    .expect("fresh ids cannot form a cycle");
            }
            previous = Some(current);
        }
        graph
    }

    /// A fan-out and fan-in: every leaf depends on `source`, and `sink` depends on every leaf
    /// (a trunk step, its expert slices, and their combine).
    pub fn fan_out_fan_in(
        id: impl Into<String>,
        source: WorkUnit,
        leaves: Vec<WorkUnit>,
        sink: WorkUnit,
    ) -> Self {
        let mut graph = Self::new(id);
        let source_id = source.id.clone();
        let sink_id = sink.id.clone();
        graph.add_unit(source);
        let leaf_ids: Vec<WorkUnitId> = leaves.iter().map(|l| l.id.clone()).collect();
        for leaf in leaves {
            graph.add_unit(leaf);
        }
        graph.add_unit(sink);
        for leaf in &leaf_ids {
            graph
                .add_dependency(leaf, &source_id)
                .expect("fresh ids cannot form a cycle");
            graph
                .add_dependency(&sink_id, leaf)
                .expect("fresh ids cannot form a cycle");
        }
        if leaf_ids.is_empty() {
            graph
                .add_dependency(&sink_id, &source_id)
                .expect("fresh ids cannot form a cycle");
        }
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AppKind, CostEstimate};

    fn unit(id: &str, ms: f64) -> WorkUnit {
        WorkUnit::new(id, AppKind::Custom("t".into())).with_cost(CostEstimate::new(0.0, 0, 0, ms))
    }

    #[test]
    fn chain_releases_one_unit_at_a_time() {
        let mut g = WorkGraph::chain("decode", vec![unit("s0", 10.0), unit("s1", 20.0)]);
        // Units are known by their graph-qualified ids; bare names resolve to them.
        let ready: Vec<_> = g.ready().iter().map(|u| u.id.0.clone()).collect();
        assert_eq!(ready, vec!["decode/s0"]);
        assert_eq!(g.ready()[0].id.name(), "s0");
        assert_eq!(g.qualify(&"s0".into()), WorkUnitId::new("decode/s0"));
        assert_eq!(g.qualify(&"decode/s0".into()), WorkUnitId::new("decode/s0"));
        g.mark_completed(&"s0".into()).unwrap();
        let ready: Vec<_> = g.ready().iter().map(|u| u.id.name().to_string()).collect();
        assert_eq!(ready, vec!["s1"]);
        assert!(!g.is_complete());
        g.mark_completed(&WorkUnitId::qualified("decode", "s1"))
            .unwrap();
        assert!(g.is_complete());
        assert_eq!(g.critical_path_ms(), 30.0);
        assert_eq!(
            g.unit(&"s1".into()).unwrap().graph_id.as_deref(),
            Some("decode")
        );
        assert_eq!(g.sinks().len(), 1);
        assert_eq!(g.sinks()[0].id.name(), "s1");
        assert!(g.unit(&"nowhere".into()).is_none());
        assert!(matches!(
            g.mark_completed(&"nowhere".into()),
            Err(GridError::UnknownUnit(id)) if id.0 == "decode/nowhere"
        ));
    }

    #[test]
    fn fan_out_fan_in_has_three_wavefronts_and_parallel_critical_path() {
        let g = WorkGraph::fan_out_fan_in(
            "layer25",
            unit("trunk", 5.0),
            vec![unit("e45", 8.0), unit("e40", 3.0), unit("e211", 6.0)],
            unit("combine", 1.0),
        );
        let levels = g.wavefronts().unwrap();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], vec![WorkUnitId::qualified("layer25", "trunk")]);
        assert_eq!(levels[1].len(), 3);
        assert_eq!(levels[2], vec![g.qualify(&"combine".into())]);
        assert_eq!(g.critical_path_ms(), 5.0 + 8.0 + 1.0);
        assert_eq!(g.total_est_wall_ms(), 23.0);
        assert_eq!(g.dependents(&"trunk".into()).len(), 3);
        assert_eq!(g.prerequisites(&"combine".into()).len(), 3);
        assert_eq!(g.sinks().len(), 1);
    }

    #[test]
    fn cycles_are_rejected() {
        let mut g = WorkGraph::new("c");
        g.add_unit(unit("a", 1.0)).add_unit(unit("b", 1.0));
        g.add_dependency(&"b".into(), &"a".into()).unwrap();
        let err = g.add_dependency(&"a".into(), &"b".into()).unwrap_err();
        assert!(matches!(err, GridError::DependencyCycle { .. }));
        assert!(matches!(
            g.add_dependency(&"a".into(), &"a".into()),
            Err(GridError::DependencyCycle { .. })
        ));
        assert!(matches!(
            g.add_dependency(&"a".into(), &"zz".into()),
            Err(GridError::UnknownUnit(_))
        ));
        // Qualified and bare names are the same unit.
        assert!(matches!(
            g.add_dependency(&"c/a".into(), &"c/b".into()),
            Err(GridError::DependencyCycle { .. })
        ));
    }

    #[test]
    fn bag_of_independent_units_is_one_wavefront() {
        let mut g = WorkGraph::new("replay");
        for i in 0..5 {
            g.add_unit(unit(&format!("r{i}"), 4.0));
        }
        assert_eq!(g.wavefronts().unwrap().len(), 1);
        assert_eq!(g.ready().len(), 5);
        assert_eq!(g.critical_path_ms(), 4.0);
    }
}
