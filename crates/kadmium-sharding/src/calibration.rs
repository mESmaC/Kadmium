//! Calibration: measured speed bound to the exact thing that produced it.
//!
//! lumabri's calibrate runs at most eight test tokens under a time limit, saves the observed
//! speed, and binds it to the resident workload that produced it, so a changed binary, adapter
//! or allocation makes the number inapplicable instead of silently stale; colibri's tune keeps a
//! measured profile only when output, hit rate and latency pass their gates. A
//! [`CalibrationRecord`] here binds measured milliseconds per token to a [`CalibrationBinding`]:
//! the model, the shape, the placement's digest, the node set and the versions that ran, and
//! the batch width. A lookup matches the whole binding or nothing. "Not calibrated" means no
//! matching measurement exists, never zero. Every record also says which kernels it measured;
//! today that is the synthetic reference math, so the mechanism is real and the numbers are not
//! yet about any model.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::ExecutionStrategy;

/// Decode steps a calibration runs at most.
pub const CALIBRATION_MAX_STEPS: u32 = 8;
/// Wall-clock budget of a calibration, in ms.
pub const CALIBRATION_MAX_WALL_MS: u64 = 20_000;
/// What the records measure today.
pub const CALIBRATION_KERNELS_REFERENCE: &str = "reference-synthetic";

/// What a measurement is bound to: change any of it and the measurement no longer applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBinding {
    pub model_id: String,
    pub strategy: ExecutionStrategy,
    pub placement_digest: u64,
    /// Every node of the mesh the step ran on, sorted.
    pub node_ids: Vec<String>,
    /// Installed version per node (`in-process` for a node the coordinator ran itself).
    pub versions: BTreeMap<String, String>,
    pub rows: u32,
}

impl CalibrationBinding {
    pub fn new(
        model_id: &str,
        strategy: ExecutionStrategy,
        placement_digest: u64,
        mut node_ids: Vec<String>,
        versions: BTreeMap<String, String>,
        rows: u32,
    ) -> Self {
        node_ids.sort();
        node_ids.dedup();
        Self {
            model_id: model_id.to_string(),
            strategy,
            placement_digest,
            node_ids,
            versions,
            rows: rows.max(1),
        }
    }
}

/// A measured speed and what it was measured on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRecord {
    pub binding: CalibrationBinding,
    pub steps: u32,
    /// Mean wall time of a step divided by its rows.
    pub measured_ms_per_token: f64,
    pub measured_tokens_per_sec: f64,
    pub step_wall_ms: Vec<f64>,
    /// Mean compute time per step on each node that ran units, from the workers' own clocks.
    pub per_node_exec_ms: BTreeMap<String, f64>,
    /// Digest of the last step's final boundary.
    pub output_digest_fnv1a: u64,
    /// For a fan-out shape, whether its bytes matched a chain step on the same prompt; `None`
    /// when the gate does not apply. A record that failed the gate never steers a choice.
    pub digest_matches_chain: Option<bool>,
    /// The kernels the measurement ran on.
    pub kernels: String,
    pub sampled_unix_ms: u64,
}

impl CalibrationRecord {
    pub fn from_steps(
        binding: CalibrationBinding,
        step_wall_ms: &[f64],
        per_node_exec_ms: BTreeMap<String, f64>,
        output_digest_fnv1a: u64,
        digest_matches_chain: Option<bool>,
        now: u64,
    ) -> Self {
        let steps = step_wall_ms.len() as u32;
        let mean_wall = if steps == 0 {
            0.0
        } else {
            step_wall_ms.iter().sum::<f64>() / steps as f64
        };
        let ms_per_token = mean_wall / binding.rows.max(1) as f64;
        let round = |v: f64| (v * 100.0).round() / 100.0;
        Self {
            measured_ms_per_token: round(ms_per_token),
            measured_tokens_per_sec: round(if ms_per_token > 0.0 {
                1000.0 / ms_per_token
            } else {
                0.0
            }),
            binding,
            steps,
            step_wall_ms: step_wall_ms.iter().map(|v| round(*v)).collect(),
            per_node_exec_ms: per_node_exec_ms
                .into_iter()
                .map(|(n, v)| (n, round(v)))
                .collect(),
            output_digest_fnv1a,
            digest_matches_chain,
            kernels: CALIBRATION_KERNELS_REFERENCE.to_string(),
            sampled_unix_ms: now,
        }
    }

    /// Whether the record may steer a choice: it measured at least one step and did not fail the
    /// digest gate.
    pub fn is_trusted(&self) -> bool {
        self.steps > 0 && self.digest_matches_chain != Some(false)
    }

    pub fn summary(&self) -> String {
        format!(
            "{:?}: {:.2} ms/token measured over {} step{} on {} nodes ({} kernels){}",
            self.binding.strategy,
            self.measured_ms_per_token,
            self.steps,
            if self.steps == 1 { "" } else { "s" },
            self.binding.node_ids.len(),
            self.kernels,
            match self.digest_matches_chain {
                Some(true) => ", bytes match the chain",
                Some(false) => ", bytes differ from the chain: not trusted",
                None => "",
            }
        )
    }
}

/// Every calibration the coordinator has kept.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationStore {
    pub records: Vec<CalibrationRecord>,
}

impl CalibrationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The trusted record for exactly this binding, if one exists.
    pub fn lookup(&self, binding: &CalibrationBinding) -> Option<&CalibrationRecord> {
        self.records
            .iter()
            .find(|r| &r.binding == binding && r.is_trusted())
    }

    /// Keeps a record, replacing any earlier one with the same binding.
    pub fn record(&mut self, record: CalibrationRecord) {
        self.records.retain(|r| r.binding != record.binding);
        self.records.push(record);
    }

    pub fn for_model(&self, model_id: &str) -> Vec<&CalibrationRecord> {
        self.records
            .iter()
            .filter(|r| r.binding.model_id == model_id)
            .collect()
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let tmp = path.with_extension(format!("{}.{nanos}.tmp", std::process::id()));
        std::fs::write(&tmp, self.to_json()?).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    /// Reads a saved store; a missing file is an empty store.
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(json) => Self::from_json(&json),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(digest: u64, version: &str, rows: u32) -> CalibrationBinding {
        CalibrationBinding::new(
            "toy-moe",
            ExecutionStrategy::ExpertParallel,
            digest,
            vec!["b".into(), "a".into()],
            [("a".to_string(), version.to_string())]
                .into_iter()
                .collect(),
            rows,
        )
    }

    #[test]
    fn records_bind_to_the_whole_identity_and_untrusted_ones_never_steer() {
        let mut store = CalibrationStore::new();
        let record = CalibrationRecord::from_steps(
            binding(7, "v1", 1),
            &[4.0, 6.0],
            [("a".to_string(), 3.0)].into_iter().collect(),
            99,
            Some(true),
            1_000,
        );
        assert_eq!(record.steps, 2);
        assert!((record.measured_ms_per_token - 5.0).abs() < 1e-9);
        assert!((record.measured_tokens_per_sec - 200.0).abs() < 1e-9);
        assert_eq!(
            record.binding.node_ids,
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(record.is_trusted());
        assert!(record.summary().contains("2 steps"));
        store.record(record.clone());

        assert_eq!(store.lookup(&binding(7, "v1", 1)), Some(&record));
        assert!(
            store.lookup(&binding(8, "v1", 1)).is_none(),
            "another placement"
        );
        assert!(
            store.lookup(&binding(7, "v2", 1)).is_none(),
            "another version"
        );
        assert!(
            store.lookup(&binding(7, "v1", 2)).is_none(),
            "another batch width"
        );

        // A later record for the same binding replaces the earlier one.
        let faster = CalibrationRecord::from_steps(
            binding(7, "v1", 1),
            &[2.0],
            BTreeMap::new(),
            99,
            Some(true),
            2_000,
        );
        store.record(faster.clone());
        assert_eq!(store.len(), 1);
        assert_eq!(store.lookup(&binding(7, "v1", 1)), Some(&faster));

        // A record that failed the digest gate is kept for the record, but never looked up.
        let tampered = CalibrationRecord::from_steps(
            binding(9, "v1", 1),
            &[1.0],
            BTreeMap::new(),
            1,
            Some(false),
            3_000,
        );
        assert!(!tampered.is_trusted());
        store.record(tampered);
        assert_eq!(store.len(), 2);
        assert!(store.lookup(&binding(9, "v1", 1)).is_none());
        assert_eq!(store.for_model("toy-moe").len(), 2);

        let dir = std::env::temp_dir().join(format!("kadmium-calibration-{}", std::process::id()));
        let path = dir.join("grid_calibrations.json");
        store.save(&path).unwrap();
        assert_eq!(CalibrationStore::load(&path).unwrap(), store);
        assert!(
            CalibrationStore::load(&dir.join("none.json"))
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
