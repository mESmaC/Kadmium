//! The sweep harness: calibration over batch widths and node counts, written as a Pareto table
//! of throughput against per-user speed into the benchmark ledger (ARCHITECTURE 9.13).
//!
//! DeepInfer's `examples/sweep/run_sweep.sh` runs one benchmark point per decode batch size
//! under one tag, writes every point as a JSON line with its whole configuration, and plots the
//! points of a tag as one Pareto curve of output throughput against per-user speed; its plotter
//! refuses a file whose points disagree on any fixed field (model, EP size, context length,
//! draft length, redundant experts), because such points are not one curve. The same
//! discipline holds here: every point carries its full configuration, points form a curve only
//! when every field but the swept batch width matches, and the Pareto frontier is marked within
//! a curve and nowhere else. Node counts are a second swept axis and each node set is its own
//! curve, as each EP size is DeepInfer's.
//!
//! Every figure a point carries is measured on the kernels the workers ran, which today are the
//! synthetic references; the table says so in every row.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::ExecutionStrategy;
use kadmium_grid::fnv1a_64;

/// Everything that fixes a point, so two points compare only when all of it matches except
/// the batch width.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepConfiguration {
    pub model_id: String,
    pub strategy: ExecutionStrategy,
    /// Rows per session per step: the swept axis within a curve.
    pub rows: u32,
    pub sessions: u32,
    pub steps: u32,
    pub node_count: usize,
    /// Every node of the mesh the point ran on, sorted.
    pub node_ids: Vec<String>,
    pub placement_digest: u64,
    /// Installed version per node (`in-process` for a node the coordinator ran itself).
    pub versions: BTreeMap<String, String>,
    pub kernels: String,
    pub pipelined: bool,
    /// `None` for a plain loop; the drafter and draft length otherwise.
    pub speculative: Option<SweepSpeculation>,
    /// Branch or commit the point was taken on, as the caller knows it.
    pub repository_state: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepSpeculation {
    pub draft_profile: String,
    pub max_draft: u32,
}

impl SweepConfiguration {
    /// The fields a curve shares: everything but the batch width.
    pub fn curve_key(&self) -> String {
        let mut text = format!(
            "{}|{:?}|sessions={}|steps={}|nodes={}|{:?}|digest={}|kernels={}|pipelined={}|{:?}|{}",
            self.model_id,
            self.strategy,
            self.sessions,
            self.steps,
            self.node_count,
            self.node_ids,
            self.placement_digest,
            self.kernels,
            self.pipelined,
            self.speculative,
            self.repository_state
        );
        for (n, v) in &self.versions {
            text.push_str(&format!("|{n}={v}"));
        }
        text
    }

    /// A short id of the curve, for the table.
    pub fn curve_id(&self) -> String {
        format!("{:08x}", fnv1a_64(self.curve_key().as_bytes()) as u32)
    }

    /// Whether two points belong to one curve.
    pub fn comparable_with(&self, other: &Self) -> bool {
        self.curve_key() == other.curve_key()
    }
}

/// What a point measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepMeasurement {
    /// Mean wall of a step across sessions, first dispatch to last return.
    pub step_ms_mean: f64,
    pub steps_per_second: f64,
    /// Tokens over the loop's wall across every session.
    pub tokens_per_sec: f64,
    /// One row's tokens over its session's wall, averaged over sessions: a user is one row of
    /// a session's batch, as in DeepInfer's plot.
    pub per_user_tokens_per_sec: f64,
    pub wall_ms: f64,
    pub tokens: u64,
    pub mean_inter_step_gap_ms: f64,
    /// For a speculative point, the full-acceptance throughput beside the accepted one above.
    pub tokens_per_sec_full_acceptance: Option<f64>,
    pub acceptance_rate: Option<f64>,
    pub sampled_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepPoint {
    pub configuration: SweepConfiguration,
    pub measured: SweepMeasurement,
}

/// Every point of a sweep, with the curves they form and the frontier within each.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SweepTable {
    pub tag: String,
    pub points: Vec<SweepPoint>,
}

impl SweepTable {
    pub fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            points: Vec::new(),
        }
    }

    pub fn push(&mut self, point: SweepPoint) {
        self.points.push(point);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Curve id of every point, in point order.
    pub fn curve_ids(&self) -> Vec<String> {
        self.points
            .iter()
            .map(|p| p.configuration.curve_id())
            .collect()
    }

    /// Distinct curves, in first-appearance order, with the indices of their points.
    pub fn curves(&self) -> Vec<(String, Vec<usize>)> {
        let mut out: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, p) in self.points.iter().enumerate() {
            let id = p.configuration.curve_id();
            match out.iter_mut().find(|(c, _)| *c == id) {
                Some((_, idx)) => idx.push(i),
                None => out.push((id, vec![i])),
            }
        }
        out
    }

    /// Whether each point is on its curve's Pareto frontier: no other point of the same curve
    /// has both higher throughput and higher per-user speed.
    pub fn pareto(&self) -> Vec<bool> {
        let mut on = vec![false; self.points.len()];
        for (_, idx) in self.curves() {
            for &i in &idx {
                let me = &self.points[i].measured;
                let dominated = idx.iter().any(|&j| {
                    if i == j {
                        return false;
                    }
                    let other = &self.points[j].measured;
                    other.tokens_per_sec >= me.tokens_per_sec
                        && other.per_user_tokens_per_sec >= me.per_user_tokens_per_sec
                        && (other.tokens_per_sec > me.tokens_per_sec
                            || other.per_user_tokens_per_sec > me.per_user_tokens_per_sec)
                });
                on[i] = !dominated;
            }
        }
        on
    }

    /// The table in the ledger's form: one row per point with its full configuration, the
    /// curve it belongs to and whether it is on that curve's frontier, under a note that points
    /// compare only within a curve.
    pub fn to_markdown(&self, title: &str) -> String {
        let pareto = self.pareto();
        let mut out = String::new();
        out.push_str(&format!("### {title}\n\n"));
        out.push_str(
            "Every row carries its full configuration. Points compare only within one curve: every field but the batch width (rows) must match, as DeepInfer's plotter requires of a sweep tag. `pareto` marks the points of a curve that no other point of the same curve beats on both throughput and per-user speed. Every figure is measured on the kernels named in the row.\n\n",
        );
        out.push_str("| curve | model | shape | nodes | node set | placement | rows | sessions | steps | pipelined | speculative | kernels | step ms (mean) | steps/s | tok/s | per-user tok/s | gap ms | pareto |\n");
        out.push_str("| :--- | :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |\n");
        for (i, p) in self.points.iter().enumerate() {
            let c = &p.configuration;
            let m = &p.measured;
            let spec = match &c.speculative {
                Some(s) => format!("{} x{}", s.draft_profile, s.max_draft),
                None => "no".to_string(),
            };
            let throughput = match m.tokens_per_sec_full_acceptance {
                Some(full) => format!(
                    "`{:.1}` (accepted), `{full:.1}` (full acceptance)",
                    m.tokens_per_sec
                ),
                None => format!("`{:.1}`", m.tokens_per_sec),
            };
            out.push_str(&format!(
                "| `{}` | `{}` | {:?} | {} | {} | `{:016x}` | {} | {} | {} | {} | {} | {} | `{:.2}` | `{:.1}` | {} | `{:.1}` | `{:.3}` | {} |\n",
                c.curve_id(),
                c.model_id,
                c.strategy,
                c.node_count,
                c.node_ids.join(", "),
                c.placement_digest,
                c.rows,
                c.sessions,
                c.steps,
                if c.pipelined { "yes" } else { "no" },
                spec,
                c.kernels,
                m.step_ms_mean,
                m.steps_per_second,
                throughput,
                m.per_user_tokens_per_sec,
                m.mean_inter_step_gap_ms,
                if pareto[i] { "yes" } else { "" },
            ));
        }
        out.push('\n');
        for (id, idx) in self.curves() {
            let c = &self.points[idx[0]].configuration;
            out.push_str(&format!(
                "- Curve `{id}`: `{}` {:?} on {} node{} ({}), {} session{}, {} steps, {}, {}, {}; versions {}; repository {}; {} point{}.\n",
                c.model_id,
                c.strategy,
                c.node_count,
                if c.node_count == 1 { "" } else { "s" },
                c.node_ids.join(", "),
                c.sessions,
                if c.sessions == 1 { "" } else { "s" },
                c.steps,
                if c.pipelined { "pipelined" } else { "sequential" },
                match &c.speculative {
                    Some(s) => format!("speculative ({} drafter, {} positions)", s.draft_profile, s.max_draft),
                    None => "plain".to_string(),
                },
                c.kernels,
                c.versions
                    .iter()
                    .map(|(n, v)| format!("{n}={v}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                c.repository_state,
                idx.len(),
                if idx.len() == 1 { "" } else { "s" },
            ));
        }
        out
    }

    pub fn to_jsonl(&self) -> String {
        self.points
            .iter()
            .filter_map(|p| serde_json::to_string(p).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Appends the table under `title` to the ledger file at `path`, creating it with a header
    /// when it does not exist. Returns the bytes written.
    pub fn append_to_ledger(&self, path: &Path, title: &str) -> Result<usize, String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let existing = std::fs::read_to_string(path).unwrap_or_default();
        let mut text = existing;
        if text.is_empty() {
            text.push_str(&format!(
                "# Sweep ledger `{}`\n\nMeasured on the kernels each row names. Points compare only within a curve.\n\n",
                self.tag
            ));
        }
        let section = self.to_markdown(title);
        text.push_str(&section);
        text.push('\n');
        std::fs::write(path, &text).map_err(|e| e.to_string())?;
        Ok(section.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(rows: u32, nodes: usize, tok: f64, user: f64) -> SweepPoint {
        SweepPoint {
            configuration: SweepConfiguration {
                model_id: "toy-moe".into(),
                strategy: ExecutionStrategy::PipelineChain,
                rows,
                sessions: 1,
                steps: 4,
                node_count: nodes,
                node_ids: (0..nodes).map(|i| format!("n{i}")).collect(),
                placement_digest: nodes as u64,
                versions: BTreeMap::new(),
                kernels: "reference-synthetic".into(),
                pipelined: true,
                speculative: None,
                repository_state: "test".into(),
            },
            measured: SweepMeasurement {
                step_ms_mean: 1000.0 / user * rows as f64,
                steps_per_second: user / rows as f64,
                tokens_per_sec: tok,
                per_user_tokens_per_sec: user,
                wall_ms: 10.0,
                tokens: 4 * rows as u64,
                mean_inter_step_gap_ms: 0.1,
                tokens_per_sec_full_acceptance: None,
                acceptance_rate: None,
                sampled_unix_ms: 1,
            },
        }
    }

    #[test]
    fn points_form_curves_by_configuration_and_the_frontier_is_marked_within_each() {
        let mut table = SweepTable::new("t");
        // One node: wider batches raise throughput and lower per-user speed, a frontier of all.
        table.push(point(1, 1, 100.0, 100.0));
        table.push(point(2, 1, 150.0, 75.0));
        table.push(point(4, 1, 200.0, 50.0));
        // A dominated point on the same curve.
        table.push(point(8, 1, 190.0, 40.0));
        // Two nodes: another curve, never compared with the first.
        table.push(point(1, 2, 90.0, 90.0));
        assert_eq!(table.len(), 5);
        let curves = table.curves();
        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0].1, vec![0, 1, 2, 3]);
        assert_eq!(curves[1].1, vec![4]);
        assert!(
            table.points[0]
                .configuration
                .comparable_with(&table.points[1].configuration)
        );
        assert!(
            !table.points[0]
                .configuration
                .comparable_with(&table.points[4].configuration)
        );
        assert_eq!(table.pareto(), vec![true, true, true, false, true]);

        let md = table.to_markdown("A sweep");
        assert!(md.contains("### A sweep"));
        assert!(md.contains("| rows |"));
        assert!(md.contains("reference-synthetic"));
        assert!(md.matches("| yes |\n").count() >= 4);
        assert!(md.contains("Curve `"));
        assert!(md.contains("compare only within one curve"));
        assert_eq!(table.to_jsonl().lines().count(), 5);

        let dir = std::env::temp_dir().join(format!("kadmium-sweep-{}", std::process::id()));
        let path = dir.join("ledger.md");
        let written = table.append_to_ledger(&path, "first").unwrap();
        assert!(written > 0);
        table.append_to_ledger(&path, "second").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# Sweep ledger `t`"));
        assert_eq!(text.matches("### ").count(), 2);
        let _ = std::fs::remove_dir_all(dir);

        let mut spec = point(1, 1, 50.0, 50.0);
        spec.configuration.speculative = Some(SweepSpeculation {
            draft_profile: "code".into(),
            max_draft: 5,
        });
        spec.measured.tokens_per_sec_full_acceptance = Some(120.0);
        let mut other = SweepTable::new("s");
        other.push(spec);
        let md = other.to_markdown("spec");
        assert!(md.contains("(full acceptance)"));
        assert!(md.contains("code x5"));
    }
}
