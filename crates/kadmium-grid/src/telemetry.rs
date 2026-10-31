//! The measured side of matchmaking: what the coordinator observes about each worker and keeps
//! between runs, next to the capability vector's modelled side.
//!
//! Workers return `exec_ms` with every result. The queue turns the validated ones into
//! [`ComputeSample`]s (never a failed, invalid or cancelled instance), and a worker's profile
//! folds them into one bounded [`ComputeEwma`] per kind of work. Once an estimate has enough
//! samples the matchmaker ranks by expected completion: the hop latency plus the measured cost
//! of the unit and of the work already queued on the worker, the way lumabri's segment routing
//! ranks replicas by measured run cost. A [`TelemetryStore`] is the coordinator's memory of these
//! estimates across runs and restarts.

use kadmium_core::{DEFAULT_LINE_WINDOW, NodeId, WindowedLine};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::WorkerProfile;

/// Floor on a sample, so a result that reports a rounded-down zero still counts as work.
const MIN_MS_PER_UNIT: f64 = 0.001;

/// A bounded exponentially weighted moving average of a worker's measured compute cost for one
/// kind of work, in milliseconds per unit of work (a row, a rollout, a simulation, a layer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeEwma {
    pub ms_per_unit: f64,
    pub samples: u32,
    pub last_sample_unix_ms: u64,
}

impl ComputeEwma {
    pub fn first(ms_per_unit: f64, now: u64) -> Self {
        Self {
            ms_per_unit: ms_per_unit.max(MIN_MS_PER_UNIT),
            samples: 1,
            last_sample_unix_ms: now,
        }
    }

    /// Folds one sample in; `alpha` is the weight of the new sample.
    pub fn observe(&mut self, ms_per_unit: f64, alpha: f64, now: u64) {
        let alpha = alpha.clamp(0.01, 1.0);
        let sample = ms_per_unit.max(MIN_MS_PER_UNIT);
        self.ms_per_unit = if self.samples == 0 {
            sample
        } else {
            alpha * sample + (1.0 - alpha) * self.ms_per_unit
        };
        self.samples = self.samples.saturating_add(1);
        self.last_sample_unix_ms = self.last_sample_unix_ms.max(now);
    }

    /// Whether the estimate saw a sample within `ttl_ms`; a `ttl_ms` of 0 never expires.
    pub fn is_fresh(&self, now: u64, ttl_ms: u64) -> bool {
        ttl_ms == 0 || now.saturating_sub(self.last_sample_unix_ms) <= ttl_ms
    }

    /// Predicted compute time for `work` units of work.
    pub fn predict_ms(&self, work: f64) -> f64 {
        self.ms_per_unit * work.max(1.0)
    }
}

/// Validated hops a worker needs before its learned line prices its hops.
pub const MIN_HOP_SAMPLES: u32 = 4;

/// A learned line for the cost of one hop to a worker, `ms = fixed + ms_per_mb * MB` over the
/// bytes the hop carried both ways, fitted by least squares toward a prior over a window of the
/// last [`DEFAULT_LINE_WINDOW`] hops (InferSSD's cost lines: a fixed term and a per-byte term,
/// each learned from the executor's own timings rather than taken from a probe, over a window
/// so a link that degrades is followed). The sample is the coordinator's wall time for a
/// dispatch less the worker's reported execution time: everything the wire, the framing and
/// the queue cost. A worker that over-reports `exec_ms` lowers its own apparent hop cost; the
/// line is a routing hint, not a settlement figure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HopLine {
    pub line: WindowedLine,
    pub fixed_ms: f64,
    pub ms_per_mb: f64,
    pub samples: u32,
    pub last_sample_unix_ms: u64,
}

impl HopLine {
    /// A line with its prior: the node's nominal round trip and its uplink's per-megabyte cost.
    pub fn with_prior(rtt_ms: f64, uplink_gbps: f64) -> Self {
        let per_mb = if uplink_gbps > 0.0 {
            8.0 / uplink_gbps // 8e6 bit per MB over Gb/s: ms per MB
        } else {
            8.0
        };
        Self {
            line: WindowedLine::new(rtt_ms.max(0.0), per_mb, DEFAULT_LINE_WINDOW),
            fixed_ms: rtt_ms.max(0.0),
            ms_per_mb: per_mb,
            samples: 0,
            last_sample_unix_ms: 0,
        }
    }

    pub fn prior_ms_per_mb(&self) -> f64 {
        self.line.prior_slope
    }

    /// Folds one hop in: `bytes` carried both ways, `ms` outside the worker's execution.
    pub fn observe(&mut self, bytes: u64, ms: f64, now: u64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        self.line.observe(bytes as f64 / 1e6, ms);
        self.fixed_ms = self.line.fixed;
        self.ms_per_mb = self.line.slope;
        self.samples = self.samples.saturating_add(1);
        self.last_sample_unix_ms = self.last_sample_unix_ms.max(now);
    }

    pub fn predict_ms(&self, bytes: u64) -> f64 {
        self.fixed_ms + self.ms_per_mb * bytes as f64 / 1e6
    }

    pub fn is_learned(&self) -> bool {
        self.samples >= MIN_HOP_SAMPLES
    }
}

/// One validated result as a sample of its worker's compute cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeSample {
    pub node: NodeId,
    /// The unit's app kind label.
    pub app: String,
    pub ms_per_unit: f64,
    pub at_unix_ms: u64,
}

/// Everything the coordinator remembers about one worker between runs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerTelemetry {
    /// Measured compute cost per kind of work.
    #[serde(default)]
    pub compute: BTreeMap<String, ComputeEwma>,
    /// Memory streaming rate the worker measured on itself, if it reported one.
    #[serde(default)]
    pub measured_stream_gbps: Option<f64>,
    /// System RAM the worker last reported free, in GiB.
    #[serde(default)]
    pub free_system_ram_gb: Option<f64>,
    /// The learned cost of a hop to the worker, from the coordinator's own dispatch timings.
    #[serde(default)]
    pub hop: Option<HopLine>,
    #[serde(default)]
    pub updated_unix_ms: u64,
}

impl WorkerTelemetry {
    pub fn total_samples(&self) -> u64 {
        self.compute.values().map(|e| e.samples as u64).sum()
    }

    /// Whether any kind of work on the worker has at least `min_samples` samples.
    pub fn is_measured(&self, min_samples: u32) -> bool {
        self.compute.values().any(|e| e.samples >= min_samples)
    }
}

/// The coordinator's memory of worker telemetry across runs and restarts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TelemetryStore {
    pub workers: BTreeMap<NodeId, WorkerTelemetry>,
}

impl TelemetryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, node: &NodeId) -> Option<&WorkerTelemetry> {
        self.workers.get(node)
    }

    /// Copies what the store knows about a worker's node into its profile.
    pub fn apply_to(&self, worker: &mut WorkerProfile) {
        if let Some(t) = self.workers.get(worker.id()) {
            worker.compute = t.compute.clone();
            if t.measured_stream_gbps.is_some() {
                worker.measured_stream_gbps = t.measured_stream_gbps;
            }
            if t.free_system_ram_gb.is_some() {
                worker.free_system_ram_gb = t.free_system_ram_gb;
            }
            if let Some(hop) = &t.hop
                && worker.hop.as_ref().is_none_or(|h| h.samples < hop.samples)
            {
                worker.hop = Some(hop.clone());
            }
        }
    }

    pub fn apply_to_all(&self, workers: &mut [WorkerProfile]) {
        for w in workers {
            self.apply_to(w);
        }
    }

    /// Records what a profile learned during a run. An estimate replaces the stored one when it
    /// has seen more samples or a later sample, so two runs absorbed in either order keep the
    /// better-informed figure.
    pub fn absorb(&mut self, worker: &WorkerProfile, now: u64) {
        let entry = self.workers.entry(worker.id().clone()).or_default();
        for (app, ewma) in &worker.compute {
            let replace = match entry.compute.get(app) {
                Some(existing) => {
                    ewma.samples > existing.samples
                        || ewma.last_sample_unix_ms > existing.last_sample_unix_ms
                }
                None => true,
            };
            if replace {
                entry.compute.insert(app.clone(), ewma.clone());
            }
        }
        if worker.measured_stream_gbps.is_some() {
            entry.measured_stream_gbps = worker.measured_stream_gbps;
        }
        if worker.free_system_ram_gb.is_some() {
            entry.free_system_ram_gb = worker.free_system_ram_gb;
        }
        if let Some(hop) = &worker.hop
            && entry.hop.as_ref().is_none_or(|h| {
                hop.samples > h.samples || hop.last_sample_unix_ms > h.last_sample_unix_ms
            })
        {
            entry.hop = Some(hop.clone());
        }
        entry.updated_unix_ms = entry.updated_unix_ms.max(now);
    }

    pub fn absorb_all(&mut self, workers: &[WorkerProfile], now: u64) {
        for w in workers {
            self.absorb(w, now);
        }
    }

    /// Workers with at least `min_samples` samples for some kind of work.
    pub fn measured_workers(&self, min_samples: u32) -> usize {
        self.workers
            .values()
            .filter(|t| t.is_measured(min_samples))
            .count()
    }

    pub fn total_samples(&self) -> u64 {
        self.workers
            .values()
            .map(WorkerTelemetry::total_samples)
            .sum()
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// Writes the store as JSON, through a temporary file so a reader never sees a partial write.
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

    /// Reads a store written by [`TelemetryStore::save`]; a missing file is an empty store.
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
    use crate::SchedulerPolicy;
    use kadmium_core::sample_personal_cluster;

    #[test]
    fn ewma_folds_samples_and_expires() {
        let mut e = ComputeEwma::first(10.0, 1_000);
        assert_eq!(e.samples, 1);
        e.observe(20.0, 0.5, 2_000);
        assert_eq!(e.samples, 2);
        assert!((e.ms_per_unit - 15.0).abs() < 1e-9);
        assert_eq!(e.last_sample_unix_ms, 2_000);
        assert!(e.is_fresh(2_500, 1_000));
        assert!(!e.is_fresh(3_001, 1_000));
        assert!(e.is_fresh(1_000_000, 0));
        assert!((e.predict_ms(4.0) - 60.0).abs() < 1e-9);
        e.observe(0.0, 1.0, 2_100);
        assert!(e.ms_per_unit > 0.0, "a zero sample is floored, not dropped");
    }

    #[test]
    fn store_roundtrips_json_and_applies_to_profiles() {
        let policy = SchedulerPolicy::default();
        let node = sample_personal_cluster().remove(0);
        let mut worker = WorkerProfile::from_node(node.clone());
        for i in 0..9 {
            worker.record_compute("ExpertSlice", 2.0 + i as f64, &policy, 1_000 + i);
        }
        worker.measured_stream_gbps = Some(42.0);
        let mut store = TelemetryStore::new();
        store.absorb(&worker, 5_000);
        assert_eq!(store.measured_workers(policy.min_compute_samples), 1);
        assert_eq!(store.total_samples(), 9);

        let json = store.to_json().unwrap();
        let back = TelemetryStore::from_json(&json).unwrap();
        assert_eq!(back, store);

        let mut fresh = WorkerProfile::from_node(node);
        assert!(fresh.compute.is_empty());
        back.apply_to(&mut fresh);
        assert_eq!(fresh.compute["ExpertSlice"].samples, 9);
        assert_eq!(fresh.measured_stream_gbps, Some(42.0));

        // A profile with fewer samples never overwrites a better-informed estimate.
        let mut stale = WorkerProfile::from_node(sample_personal_cluster().remove(0));
        stale.record_compute("ExpertSlice", 99.0, &policy, 10);
        let mut merged = back.clone();
        merged.absorb(&stale, 6_000);
        assert_eq!(merged.workers[stale.id()].compute["ExpertSlice"].samples, 9);

        let dir = std::env::temp_dir().join(format!("kadmium-telemetry-{}", std::process::id()));
        let path = dir.join("grid_telemetry.json");
        store.save(&path).unwrap();
        assert_eq!(TelemetryStore::load(&path).unwrap(), store);
        assert_eq!(
            TelemetryStore::load(&dir.join("missing.json")).unwrap(),
            TelemetryStore::default()
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod hop_line_tests {
    use super::*;
    use kadmium_core::sample_personal_cluster;

    #[test]
    fn hop_line_learns_fixed_and_per_megabyte_terms_and_is_absorbed_by_the_store() {
        let mut line = HopLine::with_prior(2.0, 1.0);
        assert!(
            (line.prior_ms_per_mb() - 8.0).abs() < 1e-9,
            "1 Gb/s is 8 ms per MB"
        );
        assert!(!line.is_learned());
        // A hop that costs 1 ms fixed plus 4 ms per MB.
        for (i, mb) in [0.5f64, 1.0, 2.0, 4.0, 1.0, 3.0].iter().enumerate() {
            line.observe((mb * 1e6) as u64, 1.0 + 4.0 * mb, 100 + i as u64);
        }
        assert!(line.is_learned());
        assert!((line.ms_per_mb - 4.0).abs() < 0.3, "{}", line.ms_per_mb);
        assert!(
            (line.predict_ms(2_000_000) - 9.0).abs() < 0.6,
            "{}",
            line.predict_ms(2_000_000)
        );
        assert_eq!(line.last_sample_unix_ms, 105);

        // The window forgets: after many slow hops the line follows them, old fast ones gone.
        for i in 0..200 {
            line.observe(1_000_000, 20.0 + 4.0, 200 + i);
        }
        assert!(
            (line.predict_ms(1_000_000) - 24.0).abs() < 1.0,
            "{}",
            line.predict_ms(1_000_000)
        );
        assert_eq!(line.line.samples.len(), DEFAULT_LINE_WINDOW);

        let node = sample_personal_cluster().remove(0);
        let mut worker = WorkerProfile::from_node(node);
        assert!(worker.learned_hop().is_none());
        // Until a line is learned the bytes are not priced: the choice between workers is
        // what it was before hop lines existed.
        let nominal = worker.hop_estimate_ms(1_000_000);
        assert_eq!(nominal, worker.hop_latency_ms());
        for i in 0..MIN_HOP_SAMPLES as u64 {
            worker.record_hop(1_000_000, 3.0, i);
        }
        assert!(worker.learned_hop().is_some());
        assert!((worker.hop_estimate_ms(1_000_000) - 3.0).abs() < 0.5);

        let mut store = TelemetryStore::new();
        store.absorb(&worker, 10);
        let stored = store.get(worker.id()).unwrap().hop.as_ref().unwrap();
        assert_eq!(stored.samples, MIN_HOP_SAMPLES);
        let mut fresh = WorkerProfile::from_node(worker.node.clone());
        store.apply_to(&mut fresh);
        assert_eq!(fresh.hop, worker.hop);
        // A worker with fewer samples never overwrites the store's line.
        let mut stale = WorkerProfile::from_node(worker.node.clone());
        stale.record_hop(1, 1.0, 1);
        store.absorb(&stale, 11);
        assert_eq!(
            store
                .get(worker.id())
                .unwrap()
                .hop
                .as_ref()
                .unwrap()
                .samples,
            MIN_HOP_SAMPLES
        );
    }
}
