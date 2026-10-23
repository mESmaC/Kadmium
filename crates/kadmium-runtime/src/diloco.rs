//! Synchronous DiLoCo outer rounds over replicated, named parameter shards.
//!
//! An island submits its final local weights after exactly `inner_steps` updates from the
//! supplied snapshot. Only replicas of the same shard are averaged; different pipeline
//! stages are never averaged together. The all-island barrier applies persistent Nesterov
//! momentum and commits every shard together. Local optimizers and transport live outside
//! this module. Byte counters estimate uncompressed FP32 parameter payloads only.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const CHECKPOINT_VERSION: u32 = 1;
const MAX_ISLANDS: usize = 64;
const MAX_SHARDS: usize = 64;
const MAX_PARAMETERS: u64 = 262_144;
const MAX_RESIDENT_SCALARS: u64 = 262_144;
const MAX_INNER_STEPS: u32 = 1_000_000;
const MAX_ID_BYTES: usize = 96;
const MAX_CHECKPOINT_BYTES: usize = 8 * 1024 * 1024;

/// Outer-loop settings. The inner optimizer belongs to each island.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiLoCoConfig {
    pub inner_steps: u32,
    pub outer_learning_rate: f32,
    pub outer_momentum: f32,
}

impl Default for DiLoCoConfig {
    fn default() -> Self {
        Self {
            inner_steps: 500,
            outer_learning_rate: 0.7,
            outer_momentum: 0.9,
        }
    }
}

impl DiLoCoConfig {
    fn validate(&self) -> Result<(), String> {
        if !(1..=MAX_INNER_STEPS).contains(&self.inner_steps) {
            return Err(format!("inner_steps must be in 1..={MAX_INNER_STEPS}"));
        }
        if !self.outer_learning_rate.is_finite() || self.outer_learning_rate <= 0.0 {
            return Err("outer_learning_rate must be finite and positive".into());
        }
        if !self.outer_momentum.is_finite() || !(0.0..1.0).contains(&self.outer_momentum) {
            return Err("outer_momentum must be finite and in [0, 1)".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterShard {
    pub shard_id: String,
    pub values: Vec<f32>,
}

/// Final local weights, never gradients or deltas, for one island's current round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IslandUpdate {
    pub island_id: String,
    pub round: u64,
    /// Equality marker from the snapshot used to start the local updates; not authentication.
    pub baseline_id: String,
    pub local_steps: u32,
    pub shards: Vec<ParameterShard>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiLoCoSnapshot {
    pub round: u64,
    /// Deterministic FNV-1a marker over round, shard layout and FP32 weight bits.
    /// This detects a mismatched starting snapshot; it is not a security proof.
    pub baseline_id: String,
    pub shards: Vec<ParameterShard>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiLoCoProgress {
    /// Current session round, incremented if this submission completed the barrier.
    pub round: u64,
    pub synchronized: bool,
    pub received_islands: Vec<String>,
    pub pending_islands: Vec<String>,
}

/// Estimates for sending every final parameter shard to the coordinator, then broadcasting
/// the new parameters to every island. Excludes protocol framing, serialization, retries,
/// local optimizer state, and initial snapshot distribution. These are not measured traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiLoCoParameterPayloadEstimates {
    pub values_per_island: u64,
    pub bytes_per_island: u64,
    pub upload_bytes_per_round: u64,
    pub broadcast_bytes_per_round: u64,
    pub total_bytes_per_round: u64,
    pub received_upload_bytes_this_round: u64,
    pub completed_round_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiLoCoStatus {
    pub config: DiLoCoConfig,
    pub round: u64,
    pub baseline_id: String,
    pub island_ids: Vec<String>,
    pub received_islands: Vec<String>,
    pub pending_islands: Vec<String>,
    pub synchronization_count: u64,
    pub parameter_payload_estimates: DiLoCoParameterPayloadEstimates,
}

#[derive(Debug, Clone)]
struct ShardState {
    values: Vec<f32>,
    momentum: Vec<f32>,
}

/// A validated session. Mutable optimizer and barrier state are private; restart goes through
/// `from_checkpoint_json` rather than unchecked deserialization of the live session.
#[derive(Debug, Clone)]
pub struct DiLoCoSession {
    config: DiLoCoConfig,
    round: u64,
    island_ids: BTreeSet<String>,
    shards: BTreeMap<String, ShardState>,
    pending: BTreeMap<String, BTreeMap<String, Vec<f32>>>,
    payload: DiLoCoParameterPayloadEstimates,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    version: u32,
    config: DiLoCoConfig,
    round: u64,
    synchronization_count: u64,
    island_ids: Vec<String>,
    shards: Vec<CheckpointShard>,
    pending: Vec<IslandUpdate>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointShard {
    shard_id: String,
    values: Vec<f32>,
    momentum: Vec<f32>,
}

impl DiLoCoSession {
    /// Creates round zero with zero outer momentum. Reference-engine limits are 64 islands,
    /// 64 shards, 96-byte IDs, and 262,144 scalar values counting master weights, outer momentum
    /// and one submission per island: `parameters_per_island * (islands + 2)`.
    pub fn new(
        config: DiLoCoConfig,
        island_ids: Vec<String>,
        shards: Vec<ParameterShard>,
    ) -> Result<Self, String> {
        config.validate()?;
        if island_ids.is_empty() || island_ids.len() > MAX_ISLANDS {
            return Err(format!("island count must be in 1..={MAX_ISLANDS}"));
        }
        let mut islands = BTreeSet::new();
        for id in island_ids {
            validate_id(&id, "island")?;
            if !islands.insert(id.clone()) {
                return Err(format!("duplicate island `{id}`"));
            }
        }
        let normalized = normalize_shards(shards)?;
        let values = normalized.values().map(|v| v.len() as u64).sum::<u64>();
        let replica_values = values
            .checked_mul(islands.len() as u64)
            .ok_or("replica parameter count overflow")?;
        let resident_scalars = values
            .checked_mul(islands.len() as u64 + 2)
            .ok_or("resident parameter count overflow")?;
        if resident_scalars > MAX_RESIDENT_SCALARS {
            return Err(format!(
                "resident scalar count including master, momentum and islands exceeds {MAX_RESIDENT_SCALARS}"
            ));
        }
        let bytes_per_island = values * 4;
        let upload_bytes_per_round = replica_values * 4;
        let payload = DiLoCoParameterPayloadEstimates {
            values_per_island: values,
            bytes_per_island,
            upload_bytes_per_round,
            broadcast_bytes_per_round: upload_bytes_per_round,
            total_bytes_per_round: upload_bytes_per_round * 2,
            received_upload_bytes_this_round: 0,
            completed_round_bytes: 0,
        };
        Ok(Self {
            config,
            round: 0,
            island_ids: islands,
            shards: normalized
                .into_iter()
                .map(|(id, values)| {
                    let momentum = vec![0.0; values.len()];
                    (id, ShardState { values, momentum })
                })
                .collect(),
            pending: BTreeMap::new(),
            payload,
        })
    }

    /// Accepts one island atomically. At the barrier, all shard updates are checked before any
    /// master weight, momentum, counter, or pending submission is changed. The declared local
    /// step count and baseline are protocol checks, not proof that a remote learner executed H.
    pub fn submit(&mut self, update: IslandUpdate) -> Result<DiLoCoProgress, String> {
        let island_id = update.island_id.clone();
        let local = self.validate_update(update)?;
        if self.pending.len() + 1 < self.island_ids.len() {
            self.pending.insert(island_id, local);
            self.payload.received_upload_bytes_this_round += self.payload.bytes_per_island;
            return Ok(self.progress(false));
        }

        let next_round = self.round.checked_add(1).ok_or("round counter overflow")?;
        let completed_bytes = self
            .payload
            .completed_round_bytes
            .checked_add(self.payload.total_bytes_per_round)
            .ok_or("completed parameter payload estimate overflow")?;
        let mut candidate = BTreeMap::new();
        let mu = f64::from(self.config.outer_momentum);
        let lr = f64::from(self.config.outer_learning_rate);
        for (shard_id, master) in &self.shards {
            let mut values = Vec::with_capacity(master.values.len());
            let mut momentum = Vec::with_capacity(master.values.len());
            for (coordinate, &before) in master.values.iter().enumerate() {
                let mut sum = 0.0f64;
                // Ordered accumulation makes arrival order and input shard order irrelevant.
                for id in &self.island_ids {
                    let replica = if id == &island_id {
                        &local
                    } else {
                        self.pending.get(id).ok_or("incomplete island barrier")?
                    };
                    let final_values = replica.get(shard_id).ok_or("missing replica shard")?;
                    sum += f64::from(before) - f64::from(final_values[coordinate]);
                }
                let gradient = checked_f32(sum / self.island_ids.len() as f64, "pseudo-gradient")?;
                let velocity = checked_f32(
                    mu * f64::from(master.momentum[coordinate]) + f64::from(gradient),
                    "outer momentum",
                )?;
                let direction = checked_f32(
                    f64::from(gradient) + mu * f64::from(velocity),
                    "Nesterov direction",
                )?;
                let value = checked_f32(
                    f64::from(before) - lr * f64::from(direction),
                    "master parameter",
                )?;
                values.push(value);
                momentum.push(velocity);
            }
            candidate.insert(shard_id.clone(), ShardState { values, momentum });
        }
        self.shards = candidate;
        self.round = next_round;
        self.pending.clear();
        self.payload.received_upload_bytes_this_round = 0;
        self.payload.completed_round_bytes = completed_bytes;
        Ok(self.progress(true))
    }

    fn validate_update(&self, update: IslandUpdate) -> Result<BTreeMap<String, Vec<f32>>, String> {
        if !self.island_ids.contains(&update.island_id) {
            return Err(format!("unknown island `{}`", update.island_id));
        }
        if update.round != self.round {
            return Err(format!(
                "expected round {}, got {}",
                self.round, update.round
            ));
        }
        if update.baseline_id != self.baseline_id() {
            return Err("baseline_id does not match the current master snapshot".into());
        }
        if self.pending.contains_key(&update.island_id) {
            return Err(format!("duplicate submission from `{}`", update.island_id));
        }
        if update.local_steps != self.config.inner_steps {
            return Err(format!(
                "expected exactly {} local steps, got {}",
                self.config.inner_steps, update.local_steps
            ));
        }
        let local = normalize_shards(update.shards)?;
        if local.len() != self.shards.len() {
            return Err("submission must contain every master shard exactly once".into());
        }
        for (id, values) in &local {
            let master = self
                .shards
                .get(id)
                .ok_or_else(|| format!("unknown shard `{id}`"))?;
            if values.len() != master.values.len() {
                return Err(format!("wrong parameter dimensions for shard `{id}`"));
            }
        }
        Ok(local)
    }

    /// The master snapshot stays unchanged until all islands finish the current round.
    pub fn snapshot(&self) -> DiLoCoSnapshot {
        DiLoCoSnapshot {
            round: self.round,
            baseline_id: self.baseline_id(),
            shards: self
                .shards
                .iter()
                .map(|(id, s)| ParameterShard {
                    shard_id: id.clone(),
                    values: s.values.clone(),
                })
                .collect(),
        }
    }

    pub fn status(&self) -> DiLoCoStatus {
        DiLoCoStatus {
            config: self.config.clone(),
            round: self.round,
            baseline_id: self.baseline_id(),
            island_ids: self.island_ids.iter().cloned().collect(),
            received_islands: self.pending.keys().cloned().collect(),
            pending_islands: self.missing_islands(),
            synchronization_count: self.round,
            parameter_payload_estimates: self.payload.clone(),
        }
    }

    fn missing_islands(&self) -> Vec<String> {
        self.island_ids
            .iter()
            .filter(|id| !self.pending.contains_key(*id))
            .cloned()
            .collect()
    }

    fn progress(&self, synchronized: bool) -> DiLoCoProgress {
        DiLoCoProgress {
            round: self.round,
            synchronized,
            received_islands: self.pending.keys().cloned().collect(),
            pending_islands: self.missing_islands(),
        }
    }

    fn baseline_id(&self) -> String {
        let mut hash = 0xcbf29ce484222325u64;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        feed(b"kadmium-diloco-baseline-v1");
        feed(&self.round.to_le_bytes());
        feed(&(self.shards.len() as u64).to_le_bytes());
        for (id, shard) in &self.shards {
            feed(&(id.len() as u64).to_le_bytes());
            feed(id.as_bytes());
            feed(&(shard.values.len() as u64).to_le_bytes());
            for value in &shard.values {
                feed(&value.to_bits().to_le_bytes());
            }
        }
        format!("fnv1a64-{hash:016x}")
    }

    /// Saves master weights, per-shard outer momentum, configuration and partial barrier state.
    /// Local learners must checkpoint their own inner optimizer/data-loader state separately.
    pub fn checkpoint_json(&self) -> Result<String, String> {
        let baseline_id = self.baseline_id();
        let checkpoint = Checkpoint {
            version: CHECKPOINT_VERSION,
            config: self.config.clone(),
            round: self.round,
            synchronization_count: self.round,
            island_ids: self.island_ids.iter().cloned().collect(),
            shards: self
                .shards
                .iter()
                .map(|(id, s)| CheckpointShard {
                    shard_id: id.clone(),
                    values: s.values.clone(),
                    momentum: s.momentum.clone(),
                })
                .collect(),
            pending: self
                .pending
                .iter()
                .map(|(id, shards)| IslandUpdate {
                    island_id: id.clone(),
                    round: self.round,
                    baseline_id: baseline_id.clone(),
                    local_steps: self.config.inner_steps,
                    shards: shards
                        .iter()
                        .map(|(id, values)| ParameterShard {
                            shard_id: id.clone(),
                            values: values.clone(),
                        })
                        .collect(),
                })
                .collect(),
        };
        let json =
            serde_json::to_string(&checkpoint).map_err(|e| format!("checkpoint encoding: {e}"))?;
        if json.len() > MAX_CHECKPOINT_BYTES {
            return Err("checkpoint exceeds 8 MiB limit".into());
        }
        Ok(json)
    }

    /// Restores only validated state. Duplicate IDs are rejected before maps are constructed;
    /// a completed barrier may not be stored as pending. Untrusted JSON is size bounded.
    pub fn from_checkpoint_json(json: &str) -> Result<Self, String> {
        if json.len() > MAX_CHECKPOINT_BYTES {
            return Err("checkpoint exceeds 8 MiB limit".into());
        }
        let checkpoint: Checkpoint =
            serde_json::from_str(json).map_err(|e| format!("checkpoint decoding: {e}"))?;
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(format!(
                "unsupported checkpoint version {}",
                checkpoint.version
            ));
        }
        if checkpoint.round != checkpoint.synchronization_count {
            return Err("checkpoint round and synchronization count disagree".into());
        }
        let mut session = Self::new(
            checkpoint.config,
            checkpoint.island_ids,
            checkpoint
                .shards
                .iter()
                .map(|s| ParameterShard {
                    shard_id: s.shard_id.clone(),
                    values: s.values.clone(),
                })
                .collect(),
        )?;
        session.payload.completed_round_bytes = checkpoint
            .round
            .checked_mul(session.payload.total_bytes_per_round)
            .ok_or("checkpoint parameter payload estimate overflow")?;
        session.round = checkpoint.round;
        for shard in checkpoint.shards {
            let target = session
                .shards
                .get_mut(&shard.shard_id)
                .ok_or("unknown checkpoint shard")?;
            if shard.momentum.len() != target.values.len()
                || shard.momentum.iter().any(|v| !v.is_finite())
            {
                return Err(format!("invalid momentum for shard `{}`", shard.shard_id));
            }
            if session.round == 0 && shard.momentum.iter().any(|&v| v != 0.0) {
                return Err("round zero checkpoint must have zero momentum".into());
            }
            target.momentum = shard.momentum;
        }
        if checkpoint.pending.len() >= session.island_ids.len() {
            return Err("checkpoint may contain only an incomplete pending barrier".into());
        }
        for update in checkpoint.pending {
            let id = update.island_id.clone();
            let normalized = session.validate_update(update)?;
            session.pending.insert(id, normalized);
            session.payload.received_upload_bytes_this_round += session.payload.bytes_per_island;
        }
        Ok(session)
    }
}

fn validate_id(id: &str, kind: &str) -> Result<(), String> {
    if id.trim().is_empty() || id.len() > MAX_ID_BYTES || id.chars().any(char::is_control) {
        return Err(format!(
            "{kind} ID must be nonblank, contain no control characters, and be at most {MAX_ID_BYTES} bytes"
        ));
    }
    Ok(())
}

fn normalize_shards(shards: Vec<ParameterShard>) -> Result<BTreeMap<String, Vec<f32>>, String> {
    if shards.is_empty() || shards.len() > MAX_SHARDS {
        return Err(format!("shard count must be in 1..={MAX_SHARDS}"));
    }
    let mut normalized = BTreeMap::new();
    let mut count = 0u64;
    for shard in shards {
        validate_id(&shard.shard_id, "shard")?;
        if shard.values.is_empty() || shard.values.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "shard `{}` must have nonempty finite values",
                shard.shard_id
            ));
        }
        count = count
            .checked_add(shard.values.len() as u64)
            .ok_or("parameter count overflow")?;
        if count > MAX_PARAMETERS {
            return Err(format!("master parameter count exceeds {MAX_PARAMETERS}"));
        }
        if normalized
            .insert(shard.shard_id.clone(), shard.values)
            .is_some()
        {
            return Err(format!("duplicate shard `{}`", shard.shard_id));
        }
    }
    Ok(normalized)
}

fn checked_f32(value: f64, label: &str) -> Result<f32, String> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() {
        return Err(format!("{label} overflowed finite FP32 range"));
    }
    Ok(narrowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard(id: &str, values: &[f32]) -> ParameterShard {
        ParameterShard {
            shard_id: id.into(),
            values: values.to_vec(),
        }
    }

    fn session() -> DiLoCoSession {
        DiLoCoSession::new(
            DiLoCoConfig {
                inner_steps: 3,
                outer_learning_rate: 0.5,
                outer_momentum: 0.5,
            },
            vec!["island-b".into(), "island-a".into()],
            vec![shard("trunk", &[10.0, -2.0]), shard("head", &[100.0])],
        )
        .unwrap()
    }

    fn update(s: &DiLoCoSession, island: &str, trunk: &[f32], head: f32) -> IslandUpdate {
        IslandUpdate {
            island_id: island.into(),
            round: s.round,
            baseline_id: s.snapshot().baseline_id,
            local_steps: s.config.inner_steps,
            shards: vec![shard("trunk", trunk), shard("head", &[head])],
        }
    }

    #[test]
    fn two_round_numerical_oracle_preserves_momentum_and_shard_identity() {
        let mut s = session();
        // Round 1: mean pseudo-gradients trunk=[3,-3], head=[-3].
        // v1=g1; theta1=theta0-.5*(g1+.5*v1).
        s.submit(update(&s, "island-a", &[8.0, 0.0], 104.0))
            .unwrap();
        assert!(
            s.submit(update(&s, "island-b", &[6.0, 2.0], 102.0))
                .unwrap()
                .synchronized
        );
        assert_eq!(
            s.snapshot().shards,
            vec![shard("head", &[102.25]), shard("trunk", &[7.75, 0.25])]
        );
        // Round 2: g2=[2,3] / [-2], v2=[3.5,1.5] / [-3.5].
        // Independent hand calculation: theta2=[5.875,-1.625] / [104.125].
        s.submit(update(&s, "island-b", &[4.75, -3.75], 105.25))
            .unwrap();
        s.submit(update(&s, "island-a", &[6.75, -1.75], 103.25))
            .unwrap();
        assert_eq!(
            s.snapshot().shards,
            vec![shard("head", &[104.125]), shard("trunk", &[5.875, -1.625])]
        );
        let status = s.status();
        assert_eq!(status.round, 2);
        assert_eq!(status.synchronization_count, 2);
        assert_eq!(status.parameter_payload_estimates.total_bytes_per_round, 48);
        assert_eq!(status.parameter_payload_estimates.completed_round_bytes, 96);
    }

    #[test]
    fn exactly_h_steps_and_matching_baseline_are_required_without_mutation() {
        let mut s = session();
        for steps in [0, 2, 4] {
            let before = s.checkpoint_json().unwrap();
            let mut u = update(&s, "island-a", &[1.0, 2.0], 3.0);
            u.local_steps = steps;
            assert!(s.submit(u).is_err());
            assert_eq!(s.checkpoint_json().unwrap(), before);
        }
        let before = s.checkpoint_json().unwrap();
        let mut u = update(&s, "island-a", &[1.0, 2.0], 3.0);
        u.baseline_id = "different-model-with-same-round".into();
        assert!(s.submit(u).unwrap_err().contains("baseline"));
        assert_eq!(s.checkpoint_json().unwrap(), before);
    }

    #[test]
    fn snapshot_stays_stable_mid_round_and_order_does_not_matter() {
        let mut first = session();
        let mut second = session();
        let initial = first.snapshot();
        let a = update(&first, "island-a", &[8.0, 0.0], 104.0);
        let b = update(&first, "island-b", &[6.0, 2.0], 102.0);
        let progress = first.submit(a.clone()).unwrap();
        assert!(!progress.synchronized);
        assert_eq!(first.snapshot(), initial);
        assert_eq!(progress.pending_islands, vec!["island-b"]);
        assert_eq!(
            first
                .status()
                .parameter_payload_estimates
                .received_upload_bytes_this_round,
            12
        );
        first.submit(b.clone()).unwrap();
        let mut b_reversed = b;
        b_reversed.shards.reverse();
        second.submit(b_reversed).unwrap();
        second.submit(a).unwrap();
        assert_eq!(
            first.checkpoint_json().unwrap(),
            second.checkpoint_json().unwrap()
        );
        assert_ne!(first.snapshot().baseline_id, initial.baseline_id);
    }

    #[test]
    fn malformed_submissions_are_rejected_atomically() {
        let mut s = session();
        let good = update(&s, "island-a", &[8.0, 0.0], 104.0);
        let mut cases = Vec::new();
        let mut bad = good.clone();
        bad.island_id = "unknown".into();
        cases.push(bad);
        let mut bad = good.clone();
        bad.round += 1;
        cases.push(bad);
        let mut bad = good.clone();
        bad.shards.pop();
        cases.push(bad);
        let mut bad = good.clone();
        bad.shards[1].shard_id = "unknown".into();
        cases.push(bad);
        let mut bad = good.clone();
        bad.shards[1] = bad.shards[0].clone();
        cases.push(bad);
        let mut bad = good.clone();
        bad.shards[0].values.pop();
        cases.push(bad);
        let mut bad = good.clone();
        bad.shards[0].values[0] = f32::NAN;
        cases.push(bad);
        let mut bad = good.clone();
        bad.shards[0].values[0] = f32::INFINITY;
        cases.push(bad);
        for bad in cases {
            let before = s.checkpoint_json().unwrap();
            assert!(s.submit(bad).is_err());
            assert_eq!(s.checkpoint_json().unwrap(), before);
        }
        s.submit(good.clone()).unwrap();
        let before = s.checkpoint_json().unwrap();
        assert!(s.submit(good.clone()).unwrap_err().contains("duplicate"));
        assert_eq!(s.checkpoint_json().unwrap(), before);
        s.submit(update(&s, "island-b", &[6.0, 2.0], 102.0))
            .unwrap();
        assert!(s.submit(good).unwrap_err().contains("round"));
    }

    #[test]
    fn overflowing_final_submission_keeps_all_shards_and_pending_state() {
        let mut s = DiLoCoSession::new(
            DiLoCoConfig {
                inner_steps: 1,
                outer_learning_rate: 1.0,
                outer_momentum: 0.5,
            },
            vec!["a".into(), "b".into()],
            vec![shard("a-normal", &[2.0]), shard("z-overflow", &[f32::MAX])],
        )
        .unwrap();
        let make = |s: &DiLoCoSession, id: &str| IslandUpdate {
            island_id: id.into(),
            round: 0,
            baseline_id: s.snapshot().baseline_id,
            local_steps: 1,
            shards: vec![shard("a-normal", &[1.0]), shard("z-overflow", &[-f32::MAX])],
        };
        s.submit(make(&s, "a")).unwrap();
        let before = s.checkpoint_json().unwrap();
        assert!(s.submit(make(&s, "b")).unwrap_err().contains("overflow"));
        assert_eq!(s.checkpoint_json().unwrap(), before);
        assert_eq!(s.status().received_islands, vec!["a"]);
    }

    #[test]
    fn checkpoint_restores_momentum_and_pending_round_identically() {
        let mut original = session();
        original
            .submit(update(&original, "island-a", &[8.0, 0.0], 104.0))
            .unwrap();
        original
            .submit(update(&original, "island-b", &[6.0, 2.0], 102.0))
            .unwrap();
        original
            .submit(update(&original, "island-b", &[4.75, -3.75], 105.25))
            .unwrap();
        let saved = original.checkpoint_json().unwrap();
        let mut restored = DiLoCoSession::from_checkpoint_json(&saved).unwrap();
        assert_eq!(original.status(), restored.status());
        assert_eq!(saved, restored.checkpoint_json().unwrap());
        let last = update(&original, "island-a", &[6.75, -1.75], 103.25);
        original.submit(last.clone()).unwrap();
        restored.submit(last).unwrap();
        assert_eq!(
            original.checkpoint_json().unwrap(),
            restored.checkpoint_json().unwrap()
        );
    }

    #[test]
    fn invalid_checkpoints_and_configs_return_errors() {
        let base: serde_json::Value =
            serde_json::from_str(&session().checkpoint_json().unwrap()).unwrap();
        let mut invalid = Vec::new();
        let mut v = base.clone();
        v["version"] = 99.into();
        invalid.push(v);
        let mut v = base.clone();
        v["config"]["inner_steps"] = 0.into();
        invalid.push(v);
        let mut v = base.clone();
        v["config"]["outer_momentum"] = 1.0.into();
        invalid.push(v);
        let mut v = base.clone();
        v["round"] = 1.into();
        invalid.push(v);
        let mut v = base.clone();
        v["shards"][0]["momentum"] = serde_json::json!([]);
        invalid.push(v);
        let mut v = base.clone();
        v["shards"][0]["momentum"][0] = 1.0.into();
        invalid.push(v);
        let mut v = base.clone();
        v["shards"][0]["values"][0] = serde_json::Value::Null;
        invalid.push(v);
        let mut v = base.clone();
        v["shards"][1] = v["shards"][0].clone();
        invalid.push(v);
        let mut v = base.clone();
        v["island_ids"] = serde_json::json!(["a", "a"]);
        invalid.push(v);
        let mut v = base;
        v["round"] = u64::MAX.into();
        v["synchronization_count"] = u64::MAX.into();
        invalid.push(v);
        for v in invalid {
            assert!(DiLoCoSession::from_checkpoint_json(&v.to_string()).is_err());
        }
        assert!(DiLoCoSession::from_checkpoint_json("not JSON").is_err());
        assert!(
            DiLoCoSession::new(DiLoCoConfig::default(), vec![], vec![shard("x", &[1.0])]).is_err()
        );
        let config = DiLoCoConfig {
            outer_learning_rate: f32::NAN,
            ..DiLoCoConfig::default()
        };
        assert!(DiLoCoSession::new(config, vec!["a".into()], vec![shard("x", &[1.0])]).is_err());
    }

    #[test]
    fn pending_checkpoint_validation_and_resource_limits_are_enforced() {
        let mut s = session();
        s.submit(update(&s, "island-a", &[8.0, 0.0], 104.0))
            .unwrap();
        let base: serde_json::Value = serde_json::from_str(&s.checkpoint_json().unwrap()).unwrap();
        let mut invalid = Vec::new();
        let mut v = base.clone();
        v["pending"][0]["round"] = 1.into();
        invalid.push(v);
        let mut v = base.clone();
        v["pending"][0]["baseline_id"] = "wrong".into();
        invalid.push(v);
        let mut v = base.clone();
        v["pending"][0]["local_steps"] = 2.into();
        invalid.push(v);
        let mut v = base.clone();
        v["pending"][0]["island_id"] = "unknown".into();
        invalid.push(v);
        let mut v = base.clone();
        v["pending"][0]["shards"][0]["values"] = serde_json::json!([]);
        invalid.push(v);
        let mut v = base.clone();
        v["pending"] = serde_json::json!([v["pending"][0], v["pending"][0]]);
        invalid.push(v.clone());
        v["island_ids"] = serde_json::json!(["island-a", "island-b", "island-c"]);
        invalid.push(v);
        let mut v = base;
        v["shards"][0]["shard_id"] = "x".repeat(MAX_ID_BYTES + 1).into();
        invalid.push(v);
        for v in invalid {
            assert!(DiLoCoSession::from_checkpoint_json(&v.to_string()).is_err());
        }
        assert!(
            DiLoCoSession::new(
                DiLoCoConfig::default(),
                vec!["a".into(), "b".into()],
                vec![shard(
                    "too-large",
                    &vec![0.0; MAX_RESIDENT_SCALARS as usize / 4 + 1]
                )],
            )
            .is_err()
        );
        assert!(
            DiLoCoSession::new(
                DiLoCoConfig::default(),
                (0..=MAX_ISLANDS).map(|i| i.to_string()).collect(),
                vec![shard("x", &[1.0])],
            )
            .is_err()
        );
    }

    #[test]
    fn checkpoint_preserves_fp32_bits_and_baseline_marker() {
        let mut values = vec![f32::from_bits(1), f32::MAX, f32::MIN_POSITIVE, -0.0];
        let mut bits = 17u32;
        for _ in 0..512 {
            bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = f32::from_bits(bits);
            if value.is_finite() {
                values.push(value);
            }
        }
        let s = DiLoCoSession::new(
            DiLoCoConfig::default(),
            vec!["a".into()],
            vec![shard("x", &values)],
        )
        .unwrap();
        let restored = DiLoCoSession::from_checkpoint_json(&s.checkpoint_json().unwrap()).unwrap();
        assert_eq!(s.snapshot().baseline_id, restored.snapshot().baseline_id);
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            restored.snapshot().shards[0]
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );
    }
}
