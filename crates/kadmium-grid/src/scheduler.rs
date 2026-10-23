//! The grid's process scheduler: matchmaking, the work-unit lifecycle, redundancy and quorum
//! validation, straggler duplication, retries, and the pool inventory.
//!
//! Dispatch works both ways, with one matchmaker behind both. A worker can *pull* work that
//! matches its capabilities and resident content (BOINC's scheduler request, the pilot-job
//! late-binding pattern), which suits replay batches, sandboxes and staging. The coordinator can
//! *push* latency-critical units to the best-ranked worker over a persistent channel, which
//! suits the units of a decode or training step.
//!
//! The matchmaker never looks inside a unit's parameters. It checks constraints (trust tier,
//! boundary-layer access, backend, fast memory, politeness, hop latency) and scores preferences
//! (locality of inputs, hop latency, capacity, availability). Once a worker has returned enough
//! validated results of a kind of work, its measured cost replaces the modelled latency and
//! capacity terms with an expected completion time: the hop, plus the measured cost of the unit
//! behind everything already queued on the worker. Residency stays the first criterion, so a
//! measured speed never pulls a unit away from the node that holds its weights, and a failed,
//! invalid or cancelled instance is never a sample.

use kadmium_core::{HardwareNode, NodeId, ResourceOffer, WorkloadKind};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    AppKind, ComputeEwma, ComputeSample, GridError, InstanceId, ReplicaCatalog, ValidationRule,
    WorkGraph, WorkInstance, WorkResultRecord, WorkState, WorkUnit, WorkUnitId, trust_satisfies,
};

/// A worker as the scheduler sees it: the capability vector plus live load and measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerProfile {
    pub node: HardwareNode,
    pub online: bool,
    /// Instances currently assigned or running on the worker.
    pub in_flight: u32,
    pub max_concurrent: u32,
    /// Measured coordinator round trip; 0 means unknown and the capability vector's figure is used.
    pub measured_rtt_ms: f64,
    /// Measured memory streaming rate from the worker's probe or heartbeats, if any.
    pub measured_stream_gbps: Option<f64>,
    pub free_fast_memory_gb: f64,
    pub last_seen_unix_ms: u64,
    /// System RAM the worker last reported free, in GiB, if it reports one.
    #[serde(default)]
    pub free_system_ram_gb: Option<f64>,
    /// Measured compute cost per kind of work, folded from validated results.
    #[serde(default)]
    pub compute: BTreeMap<String, ComputeEwma>,
    /// The owner's explicit offer, if the worker sent one; `None` means the node's default offer.
    #[serde(default)]
    pub offer: Option<ResourceOffer>,
}

impl WorkerProfile {
    pub fn from_node(node: HardwareNode) -> Self {
        let free_fast_memory_gb = node.available_fast_memory_gb();
        Self {
            node,
            online: true,
            in_flight: 0,
            max_concurrent: 4,
            measured_rtt_ms: 0.0,
            measured_stream_gbps: None,
            free_fast_memory_gb,
            last_seen_unix_ms: 0,
            free_system_ram_gb: None,
            compute: BTreeMap::new(),
            offer: None,
        }
    }

    /// Applies the owner's offer: its slots become the concurrency limit, and its fast memory
    /// under the host's polite state becomes the free fast memory the matchmaker checks.
    pub fn with_offer(mut self, offer: ResourceOffer) -> Self {
        let offer = offer.clamped_to(&self.node);
        self.max_concurrent = offer.slots.max(1);
        self.free_fast_memory_gb = offer
            .under_polite_state(&self.node, self.node.polite_state)
            .fast_memory_gb;
        self.offer = Some(offer);
        self
    }

    /// The offer in force: the owner's, clamped to the machine, or the node's default.
    pub fn effective_offer(&self) -> ResourceOffer {
        match &self.offer {
            Some(offer) => offer.clone().clamped_to(&self.node),
            None => self.node.default_offer(),
        }
    }

    pub fn with_max_concurrent(mut self, max_concurrent: u32) -> Self {
        self.max_concurrent = max_concurrent.max(1);
        self
    }

    pub fn with_measured_rtt_ms(mut self, rtt_ms: f64) -> Self {
        self.measured_rtt_ms = rtt_ms;
        self
    }

    pub fn id(&self) -> &NodeId {
        &self.node.id
    }

    /// Polite availability multiplier of the host, 1.0 when idle and 0.15 when its owner is gaming.
    pub fn availability(&self) -> f64 {
        self.node.polite_state.availability_multiplier()
    }

    pub fn hop_latency_ms(&self) -> f64 {
        if self.measured_rtt_ms > 0.0 {
            self.measured_rtt_ms
        } else {
            self.node.network_latency_ms
        }
    }

    pub fn has_capacity(&self) -> bool {
        self.online && self.in_flight < self.max_concurrent
    }

    /// Folds a validated result's measured cost into the worker's estimate for that kind of work.
    pub fn record_compute(
        &mut self,
        app: &str,
        ms_per_unit: f64,
        policy: &SchedulerPolicy,
        now: u64,
    ) {
        match self.compute.get_mut(app) {
            Some(ewma) => ewma.observe(ms_per_unit, policy.compute_ewma_alpha, now),
            None => {
                self.compute
                    .insert(app.to_string(), ComputeEwma::first(ms_per_unit, now));
            }
        }
    }

    /// The worker's measured estimate for a kind of work, once it has enough fresh samples to
    /// steer routing.
    pub fn compute_estimate(
        &self,
        app: &str,
        policy: &SchedulerPolicy,
        now: u64,
    ) -> Option<&ComputeEwma> {
        self.compute.get(app).filter(|e| {
            e.samples >= policy.min_compute_samples
                && e.is_fresh(now, policy.compute_sample_ttl_ms)
        })
    }

    /// Wall time the unit should take on this worker: measured when the worker is measured for
    /// the unit's kind of work, else the unit's own modelled estimate.
    pub fn expected_wall_ms(&self, unit: &WorkUnit, policy: &SchedulerPolicy, now: u64) -> f64 {
        self.compute_estimate(&unit.app.label(), policy, now)
            .map(|e| e.predict_ms(unit.cost.work()))
            .unwrap_or(unit.cost.est_wall_ms)
    }

    /// Expected time to completion if the unit were queued on this worker now: the hop latency,
    /// plus the measured cost of the unit behind every instance already in flight. `None` until
    /// the worker is measured for the unit's kind of work.
    pub fn completion_estimate_ms(
        &self,
        unit: &WorkUnit,
        policy: &SchedulerPolicy,
        now: u64,
    ) -> Option<f64> {
        self.completion_estimate_for(&unit.app.label(), unit.cost.work(), policy, now)
    }

    /// [`WorkerProfile::completion_estimate_ms`] for `work` units of a kind of work, for callers
    /// choosing between holders before a unit exists.
    pub fn completion_estimate_for(
        &self,
        app: &str,
        work: f64,
        policy: &SchedulerPolicy,
        now: u64,
    ) -> Option<f64> {
        let ewma = self.compute_estimate(app, policy, now)?;
        let queued = self.in_flight as f64 + 1.0;
        Some(self.hop_latency_ms() + ewma.predict_ms(work) * queued)
    }
}

/// Weights and limits of the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerPolicy {
    pub locality_weight: f64,
    pub latency_weight: f64,
    pub capacity_weight: f64,
    pub availability_weight: f64,
    pub preferred_node_bonus: f64,
    /// Workers below this availability take no new work; a host whose owner is gaming sits at 0.15.
    pub min_availability: f64,
    /// An instance running longer than `est_wall_ms * straggler_factor` gets a duplicate.
    pub straggler_factor: f64,
    /// Floor for the straggler timeout, so tiny units are not duplicated on jitter.
    pub straggler_min_ms: f64,
    pub max_attempts: u32,
    /// Units handed to one pull request at most.
    pub max_pull_batch: usize,
    /// Capacity score that counts as 1.0 when scoring.
    pub capacity_score_scale: f64,
    /// Rank eligible workers by the share of a unit's inputs they already hold before anything
    /// else, so a measured speed never pulls a unit away from the node that holds its weights.
    #[serde(default = "default_residency_first")]
    pub residency_first: bool,
    /// Validated results a worker needs for one kind of work before its measured cost steers
    /// routing and straggler timeouts.
    #[serde(default = "default_min_compute_samples")]
    pub min_compute_samples: u32,
    /// Weight of a new sample in a worker's compute average.
    #[serde(default = "default_compute_ewma_alpha")]
    pub compute_ewma_alpha: f64,
    /// An estimate without a sample this recent falls back to the modelled terms; 0 never expires.
    #[serde(default = "default_compute_sample_ttl_ms")]
    pub compute_sample_ttl_ms: u64,
    /// Completion estimate, in ms, that scores half marks on the measured term.
    #[serde(default = "default_completion_scale_ms")]
    pub completion_scale_ms: f64,
}

fn default_residency_first() -> bool {
    true
}

fn default_min_compute_samples() -> u32 {
    8
}

fn default_compute_ewma_alpha() -> f64 {
    0.2
}

fn default_compute_sample_ttl_ms() -> u64 {
    15 * 60 * 1000
}

fn default_completion_scale_ms() -> f64 {
    50.0
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self {
            locality_weight: 0.45,
            latency_weight: 0.20,
            capacity_weight: 0.20,
            availability_weight: 0.15,
            preferred_node_bonus: 0.10,
            min_availability: 0.20,
            straggler_factor: 2.5,
            straggler_min_ms: 500.0,
            max_attempts: 3,
            max_pull_batch: 4,
            capacity_score_scale: 120.0,
            residency_first: default_residency_first(),
            min_compute_samples: default_min_compute_samples(),
            compute_ewma_alpha: default_compute_ewma_alpha(),
            compute_sample_ttl_ms: default_compute_sample_ttl_ms(),
            completion_scale_ms: default_completion_scale_ms(),
        }
    }
}

/// How well one worker suits one unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchScore {
    pub node_id: NodeId,
    pub eligible: bool,
    pub score: f64,
    /// Fraction of the unit's resident inputs the worker already holds.
    pub locality: f64,
    /// Measured expected completion in ms, once the worker is measured for the unit's kind of
    /// work; `None` means the modelled latency and capacity terms scored it.
    #[serde(default)]
    pub completion_estimate_ms: Option<f64>,
    pub reason: String,
}

/// Which workload class a unit's capacity score should be taken for.
pub fn workload_for(app: &AppKind) -> WorkloadKind {
    match app {
        AppKind::TrainForwardBackward | AppKind::DeMoReduce => WorkloadKind::DistributedTraining,
        AppKind::Fts3Stage
        | AppKind::Checksum
        | AppKind::WasiSandbox
        | AppKind::DreamReplay
        | AppKind::VerdeAudit
        | AppKind::TokenizePack
        | AppKind::Custom(_) => WorkloadKind::BatchPipeline,
        _ => WorkloadKind::Inference,
    }
}

pub struct Matchmaker;

impl Matchmaker {
    pub fn score(
        unit: &WorkUnit,
        worker: &WorkerProfile,
        catalog: &ReplicaCatalog,
        policy: &SchedulerPolicy,
        now: u64,
    ) -> MatchScore {
        let node_id = worker.node.id.clone();
        let locality = catalog.locality_score(&node_id, &unit.placement.requires_resident);
        let ineligible = |reason: &str| MatchScore {
            node_id: node_id.clone(),
            eligible: false,
            score: 0.0,
            locality,
            completion_estimate_ms: None,
            reason: reason.to_string(),
        };

        if !worker.online {
            return ineligible("offline");
        }
        if !worker.has_capacity() {
            return ineligible("no free slot");
        }
        if unit.placement.excluded_nodes.contains(&node_id) {
            return ineligible("excluded by the unit");
        }
        if !trust_satisfies(worker.node.trust_tier, unit.placement.min_trust_tier) {
            return ineligible("trust tier too low");
        }
        if unit.placement.boundary_layer_access
            && !worker.node.trust_tier.can_host_boundary_ingress_egress()
        {
            return ineligible("boundary layers stay on owned hardware");
        }
        if !unit.placement.allowed_backends.is_empty()
            && !unit
                .placement
                .allowed_backends
                .contains(&worker.node.backend)
        {
            return ineligible("backend not allowed");
        }
        if worker.free_fast_memory_gb < unit.placement.min_fast_memory_gb {
            return ineligible("not enough fast memory");
        }
        let availability = worker.availability();
        if availability < policy.min_availability {
            return ineligible("host busy, polite yield");
        }
        if let Some(max_hop) = unit.placement.max_hop_latency_ms
            && worker.hop_latency_ms() > max_hop
        {
            return ineligible("hop latency too high");
        }

        let latency_term = 1.0 / (1.0 + worker.hop_latency_ms() / 10.0);
        let capacity_term = (worker.node.capacity_score(workload_for(&unit.app))
            / policy.capacity_score_scale)
            .clamp(0.0, 1.0);
        let slot_term = 1.0 - (worker.in_flight as f64 / worker.max_concurrent.max(1) as f64);
        let availability_term = availability * (0.5 + 0.5 * slot_term);

        // Measured when the worker has enough fresh samples of this kind of work: the latency
        // and capacity weights go to one term that falls with the expected completion time.
        let completion_estimate_ms = worker.completion_estimate_ms(unit, policy, now);
        let speed_terms = match completion_estimate_ms {
            Some(est) => {
                (policy.latency_weight + policy.capacity_weight)
                    / (1.0 + est / policy.completion_scale_ms.max(0.01))
            }
            None => policy.latency_weight * latency_term + policy.capacity_weight * capacity_term,
        };

        let mut score = policy.locality_weight * locality
            + speed_terms
            + policy.availability_weight * availability_term;
        if unit.placement.preferred_nodes.contains(&node_id) {
            score += policy.preferred_node_bonus;
        }

        MatchScore {
            node_id,
            eligible: true,
            score,
            locality,
            completion_estimate_ms,
            reason: "ok".to_string(),
        }
    }

    /// All workers scored for a unit: eligible first, then (under `residency_first`) the share of
    /// the unit's inputs already held, then the best score, with the node id as the tie-break.
    pub fn rank(
        unit: &WorkUnit,
        workers: &[WorkerProfile],
        catalog: &ReplicaCatalog,
        policy: &SchedulerPolicy,
        now: u64,
    ) -> Vec<MatchScore> {
        let mut scores: Vec<MatchScore> = workers
            .iter()
            .map(|w| Self::score(unit, w, catalog, policy, now))
            .collect();
        scores.sort_by(|a, b| {
            b.eligible
                .cmp(&a.eligible)
                .then_with(|| {
                    if policy.residency_first {
                        b.locality
                            .partial_cmp(&a.locality)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .then_with(|| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.node_id.cmp(&b.node_id))
        });
        scores
    }
}

/// Everything the queue knows about one unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitStatus {
    pub unit: WorkUnit,
    pub state: WorkState,
    pub instances: Vec<InstanceId>,
    pub returned: Vec<WorkResultRecord>,
    pub validated: Option<WorkResultRecord>,
    /// Failed or offline attempts so far.
    pub attempts: u32,
    /// Prerequisites not yet validated.
    pub blocked_on: BTreeSet<WorkUnitId>,
    /// A Verde dispute is open; the runtime resolves it with [`WorkQueue::resolve_dispute`].
    pub dispute_open: bool,
    /// Replicas wanted beyond the policy's target, raised for stragglers and tie-breaks.
    pub extra_replicas_wanted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchedulerEvent {
    Submitted {
        unit: WorkUnitId,
    },
    /// All prerequisites validated; the unit is now ready.
    Released {
        unit: WorkUnitId,
    },
    Assigned {
        unit: WorkUnitId,
        instance: InstanceId,
        node: NodeId,
    },
    Started {
        instance: InstanceId,
    },
    Returned {
        instance: InstanceId,
        node: NodeId,
    },
    Validated {
        unit: WorkUnitId,
        node: NodeId,
    },
    Invalidated {
        instance: InstanceId,
        node: NodeId,
        reason: String,
    },
    DisputeOpened {
        unit: WorkUnitId,
    },
    /// A Verde dispute settled: `winner` validated, or `None` when the referee reproduced no
    /// disputant and the unit was rejected. `reason` says what the referee found.
    DisputeSettled {
        unit: WorkUnitId,
        winner: Option<NodeId>,
        reason: String,
    },
    Duplicated {
        unit: WorkUnitId,
        original: InstanceId,
        reason: String,
    },
    Retried {
        unit: WorkUnitId,
        attempt: u32,
        reason: String,
    },
    Failed {
        unit: WorkUnitId,
        reason: String,
    },
    Expired {
        unit: WorkUnitId,
    },
    Assimilated {
        unit: WorkUnitId,
    },
    Cancelled {
        unit: WorkUnitId,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerEventRecord {
    pub at_unix_ms: u64,
    pub event: SchedulerEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationOutcome {
    Pending { returned: u32, needed: u32 },
    Validated { node: NodeId },
    DisputeOpened { nodes: Vec<NodeId> },
    Invalid { reason: String },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueueStats {
    pub total: usize,
    pub ready: usize,
    pub blocked: usize,
    pub unsent: usize,
    pub assigned: usize,
    pub running: usize,
    pub returned: usize,
    pub validated: usize,
    pub invalid: usize,
    pub assimilated: usize,
    pub expired: usize,
    pub failed: usize,
    pub disputes_open: usize,
    pub instances_live: usize,
}

/// The work queue: every unit the grid has been asked to run, every instance of each, and the
/// events that moved them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkQueue {
    policy: SchedulerPolicy,
    units: BTreeMap<WorkUnitId, UnitStatus>,
    instances: BTreeMap<InstanceId, WorkInstance>,
    /// prerequisite -> units waiting on it
    dependents: BTreeMap<WorkUnitId, Vec<WorkUnitId>>,
    straggler_flagged: BTreeSet<InstanceId>,
    next_instance_seq: u64,
    events: Vec<SchedulerEventRecord>,
    /// Measured costs of results validated since the last drain.
    #[serde(default)]
    compute_samples: Vec<ComputeSample>,
}

impl WorkQueue {
    pub fn new(policy: SchedulerPolicy) -> Self {
        Self {
            policy,
            units: BTreeMap::new(),
            instances: BTreeMap::new(),
            dependents: BTreeMap::new(),
            straggler_flagged: BTreeSet::new(),
            next_instance_seq: 0,
            events: Vec::new(),
            compute_samples: Vec::new(),
        }
    }

    pub fn policy(&self) -> &SchedulerPolicy {
        &self.policy
    }

    /// Keeps a validated result as a sample of its worker's compute cost per unit of work. Only
    /// validated results get here: a failed, invalid or cancelled instance is never a sample.
    fn sample_compute(&mut self, app: &AppKind, record: &WorkResultRecord, now: u64) {
        if record.exec_ms <= 0.0 || !record.exec_ms.is_finite() {
            return;
        }
        self.compute_samples.push(ComputeSample {
            node: record.node_id.clone(),
            app: app.label(),
            ms_per_unit: record.exec_ms / record.units_processed.max(1) as f64,
            at_unix_ms: now,
        });
    }

    /// Compute samples of the results validated since the last drain.
    pub fn drain_compute_samples(&mut self) -> Vec<ComputeSample> {
        std::mem::take(&mut self.compute_samples)
    }

    /// Folds the samples validated since the last call into the matching worker profiles and
    /// returns how many there were.
    pub fn feed_workers(&mut self, workers: &mut [WorkerProfile], now: u64) -> usize {
        let samples = self.drain_compute_samples();
        let count = samples.len();
        for sample in samples {
            if let Some(worker) = workers.iter_mut().find(|w| w.id() == &sample.node) {
                worker.record_compute(&sample.app, sample.ms_per_unit, &self.policy, now);
            }
        }
        count
    }

    fn emit(&mut self, now: u64, event: SchedulerEvent) {
        self.events.push(SchedulerEventRecord {
            at_unix_ms: now,
            event,
        });
    }

    pub fn submit(&mut self, unit: WorkUnit, now: u64) -> Result<(), GridError> {
        if self.units.contains_key(&unit.id) {
            return Err(GridError::DuplicateUnit(unit.id));
        }
        let id = unit.id.clone();
        self.units.insert(
            id.clone(),
            UnitStatus {
                unit,
                state: WorkState::Unsent,
                instances: Vec::new(),
                returned: Vec::new(),
                validated: None,
                attempts: 0,
                blocked_on: BTreeSet::new(),
                dispute_open: false,
                extra_replicas_wanted: 0,
            },
        );
        self.emit(now, SchedulerEvent::Submitted { unit: id });
        Ok(())
    }

    /// Submits every unit of a graph; units with prerequisites stay blocked until those validate.
    /// Unit ids are graph-qualified, so the graphs of several sessions, or the successive steps
    /// of one, sit in the queue together.
    pub fn submit_graph(&mut self, graph: &WorkGraph, now: u64) -> Result<(), GridError> {
        self.submit_graph_after(graph, &[], now)
    }

    /// [`WorkQueue::submit_graph`] with prerequisites outside the graph: each `(unit, prereq)`
    /// pair blocks the graph's `unit` on a unit already in the queue, which is how step `t + 1`
    /// of a session waits for the sink of step `t` while both sit in the queue. A prerequisite
    /// that has validated blocks nothing; one that has failed, expired or been cancelled is an
    /// error, because the dependent could never run; an unknown one is an error too.
    pub fn submit_graph_after(
        &mut self,
        graph: &WorkGraph,
        external: &[(WorkUnitId, WorkUnitId)],
        now: u64,
    ) -> Result<(), GridError> {
        if let Some(existing) = graph.units().find(|u| self.units.contains_key(&u.id)) {
            return Err(GridError::DuplicateUnit(existing.id.clone()));
        }
        let mut external_blocks: BTreeMap<WorkUnitId, BTreeSet<WorkUnitId>> = BTreeMap::new();
        for (unit, prereq) in external {
            let unit = graph.qualify(unit);
            if graph.unit(&unit).is_none() {
                return Err(GridError::UnknownUnit(unit));
            }
            let Some(status) = self.units.get(prereq) else {
                return Err(GridError::UnknownUnit(prereq.clone()));
            };
            if status.validated.is_some() {
                continue;
            }
            if status.state.is_terminal() {
                return Err(GridError::PrerequisiteFailed {
                    unit,
                    prereq: prereq.clone(),
                    state: status.state,
                });
            }
            external_blocks
                .entry(unit)
                .or_default()
                .insert(prereq.clone());
        }
        for unit in graph.units() {
            let mut prereqs: BTreeSet<WorkUnitId> = graph
                .prerequisites(&unit.id)
                .into_iter()
                .filter(|p| !graph.is_completed(p))
                .collect();
            if let Some(blocks) = external_blocks.remove(&unit.id) {
                prereqs.extend(blocks);
            }
            for p in &prereqs {
                self.dependents
                    .entry(p.clone())
                    .or_default()
                    .push(unit.id.clone());
            }
            self.submit(unit.clone(), now)?;
            if let Some(status) = self.units.get_mut(&unit.id) {
                status.blocked_on = prereqs;
            }
        }
        Ok(())
    }

    /// Whether every unit of `graph` has reached a terminal state or validated: nothing of it
    /// is still queued, running or awaiting a verdict.
    pub fn graph_settled(&self, graph: &WorkGraph) -> bool {
        graph.units().all(|u| {
            self.units
                .get(&u.id)
                .is_some_and(|s| s.validated.is_some() || s.state.is_terminal())
        })
    }

    fn live_instances(&self, status: &UnitStatus) -> usize {
        status
            .instances
            .iter()
            .filter(|i| {
                self.instances
                    .get(*i)
                    .is_some_and(|inst| inst.state.is_live())
            })
            .count()
    }

    fn needs_instances(&self, status: &UnitStatus) -> bool {
        if status.state.is_terminal()
            || status.state == WorkState::Validated
            || status.validated.is_some()
            || status.dispute_open
            || !status.blocked_on.is_empty()
        {
            return false;
        }
        let policy = &status.unit.redundancy;
        let wanted = (policy.target_replicas + status.extra_replicas_wanted)
            .min(policy.max_replicas)
            .max(1) as usize;
        let ceiling = (policy.max_replicas + self.policy.max_attempts) as usize;
        self.live_instances(status) < wanted && status.instances.len() < ceiling
    }

    /// Units that need an instance, earliest deadline first, then priority, then age.
    pub fn ready_units(&self) -> Vec<&WorkUnit> {
        let mut ready: Vec<&UnitStatus> = self
            .units
            .values()
            .filter(|s| self.needs_instances(s))
            .collect();
        ready.sort_by_key(|s| {
            (
                s.unit.deadline_unix_ms.unwrap_or(u64::MAX),
                Reverse(s.unit.priority),
                s.unit.created_unix_ms,
                s.unit.id.clone(),
            )
        });
        ready.into_iter().map(|s| &s.unit).collect()
    }

    fn has_live_instance_on(&self, status: &UnitStatus, node: &NodeId) -> bool {
        status.instances.iter().any(|i| {
            self.instances
                .get(i)
                .is_some_and(|inst| &inst.node_id == node && inst.state.is_live())
        })
    }

    fn straggling_original(&self, status: &UnitStatus) -> Option<InstanceId> {
        status
            .instances
            .iter()
            .find(|i| self.straggler_flagged.contains(*i))
            .cloned()
    }

    fn assign(
        &mut self,
        unit_id: &WorkUnitId,
        node: &NodeId,
        now: u64,
        duplicate_of: Option<InstanceId>,
    ) -> WorkInstance {
        self.next_instance_seq += 1;
        let instance_id = InstanceId(format!("{unit_id}#{}", self.next_instance_seq));
        let status = self.units.get_mut(unit_id).expect("unit exists");
        let instance = WorkInstance {
            instance_id: instance_id.clone(),
            unit_id: unit_id.clone(),
            node_id: node.clone(),
            attempt: status.instances.len() as u32 + 1,
            state: WorkState::Assigned,
            assigned_unix_ms: now,
            started_unix_ms: None,
            returned_unix_ms: None,
            est_wall_ms: status.unit.cost.est_wall_ms,
            duplicate_of,
        };
        status.instances.push(instance_id.clone());
        if status.state == WorkState::Unsent {
            status.state = WorkState::Assigned;
        }
        self.instances.insert(instance_id.clone(), instance.clone());
        self.emit(
            now,
            SchedulerEvent::Assigned {
                unit: unit_id.clone(),
                instance: instance_id,
                node: node.clone(),
            },
        );
        instance
    }

    /// Pull dispatch: hands the worker the best-matching ready units it can run, at most
    /// `max_units` and never more than the policy's pull batch or the worker's free slots.
    pub fn request_work(
        &mut self,
        worker: &mut WorkerProfile,
        catalog: &ReplicaCatalog,
        now: u64,
        max_units: usize,
    ) -> Vec<WorkInstance> {
        worker.last_seen_unix_ms = now;
        let limit = max_units.min(self.policy.max_pull_batch);
        let candidates: Vec<WorkUnitId> = self.ready_units().iter().map(|u| u.id.clone()).collect();
        let mut handed = Vec::new();

        for unit_id in candidates {
            if handed.len() >= limit || !worker.has_capacity() {
                break;
            }
            let status = self.units.get(&unit_id).expect("ready unit exists");
            if self.has_live_instance_on(status, worker.id()) {
                continue;
            }
            let score = Matchmaker::score(&status.unit, worker, catalog, &self.policy, now);
            if !score.eligible {
                continue;
            }
            let duplicate_of = self.straggling_original(status);
            let node = worker.id().clone();
            let instance = self.assign(&unit_id, &node, now, duplicate_of);
            worker.in_flight += 1;
            handed.push(instance);
        }
        handed
    }

    /// Push dispatch: assigns every ready unit to its best-ranked eligible worker, filling
    /// redundancy targets with distinct nodes.
    pub fn dispatch_push(
        &mut self,
        workers: &mut [WorkerProfile],
        catalog: &ReplicaCatalog,
        now: u64,
    ) -> Vec<WorkInstance> {
        let candidates: Vec<WorkUnitId> = self.ready_units().iter().map(|u| u.id.clone()).collect();
        let mut assigned = Vec::new();

        for unit_id in candidates {
            loop {
                let status = self.units.get(&unit_id).expect("ready unit exists");
                if !self.needs_instances(status) {
                    break;
                }
                let ranked = Matchmaker::rank(&status.unit, workers, catalog, &self.policy, now);
                let chosen = ranked
                    .iter()
                    .filter(|m| m.eligible)
                    .find(|m| !self.has_live_instance_on(status, &m.node_id))
                    .map(|m| m.node_id.clone());
                let Some(node) = chosen else {
                    break;
                };
                let duplicate_of = self.straggling_original(status);
                let instance = self.assign(&unit_id, &node, now, duplicate_of);
                if let Some(w) = workers.iter_mut().find(|w| w.id() == &node) {
                    w.in_flight += 1;
                    w.last_seen_unix_ms = now;
                }
                assigned.push(instance);
            }
        }
        assigned
    }

    pub fn report_started(&mut self, instance_id: &InstanceId, now: u64) -> Result<(), GridError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or_else(|| GridError::UnknownInstance(instance_id.clone()))?;
        if instance.state != WorkState::Assigned {
            return Err(GridError::InvalidTransition {
                unit: instance.unit_id.clone(),
                from: instance.state,
                to: WorkState::Running,
            });
        }
        instance.state = WorkState::Running;
        instance.started_unix_ms = Some(now);
        let unit_id = instance.unit_id.clone();
        if let Some(status) = self.units.get_mut(&unit_id)
            && status.state == WorkState::Assigned
        {
            status.state = WorkState::Running;
        }
        self.emit(
            now,
            SchedulerEvent::Started {
                instance: instance_id.clone(),
            },
        );
        Ok(())
    }

    /// Records a returned result and runs validation for its unit.
    pub fn report_result(
        &mut self,
        instance_id: &InstanceId,
        result: WorkResultRecord,
        now: u64,
    ) -> Result<ValidationOutcome, GridError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or_else(|| GridError::UnknownInstance(instance_id.clone()))?;
        if !matches!(instance.state, WorkState::Assigned | WorkState::Running) {
            return Err(GridError::InvalidTransition {
                unit: instance.unit_id.clone(),
                from: instance.state,
                to: WorkState::Returned,
            });
        }
        instance.state = WorkState::Returned;
        instance.returned_unix_ms = Some(now);
        let unit_id = instance.unit_id.clone();
        let node = instance.node_id.clone();

        let status = self
            .units
            .get_mut(&unit_id)
            .ok_or_else(|| GridError::UnknownUnit(unit_id.clone()))?;
        if status.state.is_terminal() {
            return Ok(ValidationOutcome::Invalid {
                reason: format!("unit is {:?}", status.state),
            });
        }
        let mut record = result;
        record.unit_id = unit_id.clone();
        record.instance_id = instance_id.clone();
        record.node_id = node.clone();
        record.returned_unix_ms = now;
        status.returned.push(record);
        if matches!(status.state, WorkState::Assigned | WorkState::Running) {
            status.state = WorkState::Returned;
        }
        self.emit(
            now,
            SchedulerEvent::Returned {
                instance: instance_id.clone(),
                node,
            },
        );
        Ok(self.validate(&unit_id, now))
    }

    /// Records a failed or abandoned instance and retries the unit while attempts remain.
    pub fn report_failure(
        &mut self,
        instance_id: &InstanceId,
        reason: &str,
        now: u64,
    ) -> Result<(), GridError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or_else(|| GridError::UnknownInstance(instance_id.clone()))?;
        if !instance.state.is_live() {
            return Ok(());
        }
        instance.state = WorkState::Failed;
        let unit_id = instance.unit_id.clone();
        self.straggler_flagged.remove(instance_id);

        let max_attempts = self.policy.max_attempts;
        let live = {
            let status = self.units.get(&unit_id).expect("unit exists");
            self.live_instances(status)
        };
        let status = self.units.get_mut(&unit_id).expect("unit exists");
        status.attempts += 1;
        if status.validated.is_some() || status.state.is_terminal() {
            return Ok(());
        }
        if status.attempts >= max_attempts && live == 0 {
            status.state = WorkState::Failed;
            let attempts = status.attempts;
            self.emit(
                now,
                SchedulerEvent::Failed {
                    unit: unit_id,
                    reason: format!("{reason} after {attempts} attempts"),
                },
            );
        } else {
            if live == 0 {
                status.state = WorkState::Unsent;
            }
            let attempt = status.attempts;
            self.emit(
                now,
                SchedulerEvent::Retried {
                    unit: unit_id,
                    attempt,
                    reason: reason.to_string(),
                },
            );
        }
        Ok(())
    }

    fn validate(&mut self, unit_id: &WorkUnitId, now: u64) -> ValidationOutcome {
        let status = self.units.get(unit_id).expect("unit exists");
        let app = status.unit.app.clone();
        let rule = status.unit.redundancy.validation.clone();
        let min_quorum = status.unit.redundancy.min_quorum.max(1) as usize;
        let max_replicas = status.unit.redundancy.max_replicas as usize;
        let returned = status.returned.clone();

        if let Some(winner) = status.validated.clone() {
            // A late result after validation: agreement is fine, disagreement is invalid.
            let mut invalid = Vec::new();
            for r in &returned {
                if r.instance_id != winner.instance_id
                    && self
                        .instances
                        .get(&r.instance_id)
                        .is_some_and(|i| i.state == WorkState::Returned)
                {
                    let agrees = match &rule {
                        ValidationRule::Trusted => true,
                        ValidationRule::ScalarTolerance { max_abs_diff } => {
                            (r.check_value - winner.check_value).abs() <= *max_abs_diff
                        }
                        _ => r.digest_fnv1a == winner.digest_fnv1a,
                    };
                    invalid.push((r.clone(), agrees));
                }
            }
            for (record, agrees) in invalid {
                self.settle_instance(&record.instance_id, &record.node_id, agrees, now);
                if agrees {
                    self.sample_compute(&app, &record, now);
                }
            }
            return ValidationOutcome::Validated {
                node: winner.node_id,
            };
        }

        if returned.is_empty() {
            return ValidationOutcome::Pending {
                returned: 0,
                needed: min_quorum as u32,
            };
        }

        let (winners, losers): (Vec<WorkResultRecord>, Vec<WorkResultRecord>) = match &rule {
            ValidationRule::Trusted => (vec![returned[0].clone()], Vec::new()),
            ValidationRule::BitExactDigest | ValidationRule::VerdeBisection => {
                let mut groups: BTreeMap<u64, Vec<WorkResultRecord>> = BTreeMap::new();
                for r in &returned {
                    groups.entry(r.digest_fnv1a).or_default().push(r.clone());
                }
                let best = groups
                    .values()
                    .max_by_key(|g| (g.len(), Reverse(g[0].returned_unix_ms)))
                    .cloned()
                    .unwrap_or_default();
                if best.len() >= min_quorum {
                    let losers = returned
                        .iter()
                        .filter(|r| r.digest_fnv1a != best[0].digest_fnv1a)
                        .cloned()
                        .collect();
                    (best, losers)
                } else {
                    (Vec::new(), Vec::new())
                }
            }
            ValidationRule::ScalarTolerance { max_abs_diff } => {
                let mut values: Vec<f64> = returned.iter().map(|r| r.check_value).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = values[values.len() / 2];
                let (agree, disagree): (Vec<_>, Vec<_>) = returned
                    .iter()
                    .cloned()
                    .partition(|r| (r.check_value - median).abs() <= *max_abs_diff);
                if agree.len() >= min_quorum {
                    (agree, disagree)
                } else {
                    (Vec::new(), Vec::new())
                }
            }
        };

        if !winners.is_empty() {
            let winner = winners
                .iter()
                .min_by_key(|r| r.returned_unix_ms)
                .cloned()
                .expect("non-empty");
            for w in &winners {
                self.settle_instance(&w.instance_id, &w.node_id, true, now);
                self.sample_compute(&app, w, now);
            }
            for l in &losers {
                self.settle_instance(&l.instance_id, &l.node_id, false, now);
            }
            let status = self.units.get_mut(unit_id).expect("unit exists");
            status.validated = Some(winner.clone());
            status.state = WorkState::Validated;
            status.dispute_open = false;
            self.emit(
                now,
                SchedulerEvent::Validated {
                    unit: unit_id.clone(),
                    node: winner.node_id.clone(),
                },
            );
            self.release_dependents(unit_id, now);
            return ValidationOutcome::Validated {
                node: winner.node_id,
            };
        }

        // No quorum yet.
        let distinct_outcomes = match &rule {
            ValidationRule::ScalarTolerance { .. } => 0,
            _ => returned
                .iter()
                .map(|r| r.digest_fnv1a)
                .collect::<BTreeSet<_>>()
                .len(),
        };
        let disagreement = returned.len() >= 2 && distinct_outcomes > 1;

        if disagreement && rule == ValidationRule::VerdeBisection {
            if let Some(status) = self.units.get_mut(unit_id) {
                status.dispute_open = true;
            }
            let nodes: Vec<NodeId> = returned.iter().map(|r| r.node_id.clone()).collect();
            self.emit(
                now,
                SchedulerEvent::DisputeOpened {
                    unit: unit_id.clone(),
                },
            );
            return ValidationOutcome::DisputeOpened { nodes };
        }

        let (instance_count, live) = {
            let status = self.units.get(unit_id).expect("unit exists");
            (status.instances.len(), self.live_instances(status))
        };

        if instance_count >= max_replicas {
            // Nothing else is outstanding, so no quorum can ever form.
            if live == returned.len() || live == 0 {
                if let Some(status) = self.units.get_mut(unit_id) {
                    status.state = WorkState::Invalid;
                }
                for r in &returned {
                    self.settle_instance(&r.instance_id, &r.node_id, false, now);
                }
                self.emit(
                    now,
                    SchedulerEvent::Failed {
                        unit: unit_id.clone(),
                        reason: "no quorum within the replica ceiling".to_string(),
                    },
                );
                return ValidationOutcome::Invalid {
                    reason: "no quorum within the replica ceiling".to_string(),
                };
            }
        } else if disagreement && let Some(status) = self.units.get_mut(unit_id) {
            // Ask for a tie-breaker.
            status.extra_replicas_wanted += 1;
        }

        ValidationOutcome::Pending {
            returned: returned.len() as u32,
            needed: min_quorum as u32,
        }
    }

    fn settle_instance(&mut self, instance_id: &InstanceId, node: &NodeId, valid: bool, now: u64) {
        self.settle_instance_with_reason(
            instance_id,
            node,
            valid,
            "disagrees with the validated result",
            now,
        );
    }

    fn settle_instance_with_reason(
        &mut self,
        instance_id: &InstanceId,
        node: &NodeId,
        valid: bool,
        reason: &str,
        now: u64,
    ) {
        if let Some(instance) = self.instances.get_mut(instance_id) {
            if instance.state != WorkState::Returned {
                return;
            }
            instance.state = if valid {
                WorkState::Validated
            } else {
                WorkState::Invalid
            };
        }
        self.straggler_flagged.remove(instance_id);
        if !valid {
            self.emit(
                now,
                SchedulerEvent::Invalidated {
                    instance: instance_id.clone(),
                    node: node.clone(),
                    reason: reason.to_string(),
                },
            );
        }
    }

    fn release_dependents(&mut self, unit_id: &WorkUnitId, now: u64) {
        let Some(waiting) = self.dependents.remove(unit_id) else {
            return;
        };
        for dependent in waiting {
            let released = if let Some(status) = self.units.get_mut(&dependent) {
                status.blocked_on.remove(unit_id);
                status.blocked_on.is_empty()
            } else {
                false
            };
            if released {
                self.emit(now, SchedulerEvent::Released { unit: dependent });
            }
        }
    }

    /// Settles a Verde dispute in favour of `winning_node`, whose returned result validates.
    pub fn resolve_dispute(
        &mut self,
        unit_id: &WorkUnitId,
        winning_node: &NodeId,
        now: u64,
    ) -> Result<ValidationOutcome, GridError> {
        self.resolve_dispute_with_reason(
            unit_id,
            winning_node,
            "disagrees with the validated result",
            now,
        )
    }

    /// [`WorkQueue::resolve_dispute`] with what the referee found recorded against every loser,
    /// such as the layer whose digest it could not reproduce.
    pub fn resolve_dispute_with_reason(
        &mut self,
        unit_id: &WorkUnitId,
        winning_node: &NodeId,
        reason: &str,
        now: u64,
    ) -> Result<ValidationOutcome, GridError> {
        let status = self
            .units
            .get(unit_id)
            .ok_or_else(|| GridError::UnknownUnit(unit_id.clone()))?;
        if !status.dispute_open {
            return Err(GridError::InvalidTransition {
                unit: unit_id.clone(),
                from: status.state,
                to: WorkState::Validated,
            });
        }
        let Some(winner) = status
            .returned
            .iter()
            .find(|r| &r.node_id == winning_node)
            .cloned()
        else {
            return Err(GridError::UnknownUnit(unit_id.clone()));
        };
        let losers: Vec<WorkResultRecord> = status
            .returned
            .iter()
            .filter(|r| r.instance_id != winner.instance_id)
            .cloned()
            .collect();
        let app = status.unit.app.clone();
        self.settle_instance(&winner.instance_id, &winner.node_id, true, now);
        self.sample_compute(&app, &winner, now);
        for l in &losers {
            self.settle_instance_with_reason(&l.instance_id, &l.node_id, false, reason, now);
        }
        let status = self.units.get_mut(unit_id).expect("unit exists");
        status.validated = Some(winner.clone());
        status.state = WorkState::Validated;
        status.dispute_open = false;
        self.emit(
            now,
            SchedulerEvent::DisputeSettled {
                unit: unit_id.clone(),
                winner: Some(winner.node_id.clone()),
                reason: reason.to_string(),
            },
        );
        self.emit(
            now,
            SchedulerEvent::Validated {
                unit: unit_id.clone(),
                node: winner.node_id.clone(),
            },
        );
        self.release_dependents(unit_id, now);
        Ok(ValidationOutcome::Validated {
            node: winner.node_id,
        })
    }

    /// Settles a Verde dispute against every disputant, because the referee reproduced none of
    /// them. Their instances are invalidated and their nodes excluded from the unit, which goes
    /// back to the queue while the replica ceiling allows another attempt and fails otherwise.
    pub fn reject_dispute(
        &mut self,
        unit_id: &WorkUnitId,
        reason: &str,
        now: u64,
    ) -> Result<ValidationOutcome, GridError> {
        let status = self
            .units
            .get(unit_id)
            .ok_or_else(|| GridError::UnknownUnit(unit_id.clone()))?;
        if !status.dispute_open {
            return Err(GridError::InvalidTransition {
                unit: unit_id.clone(),
                from: status.state,
                to: WorkState::Invalid,
            });
        }
        let returned = status.returned.clone();
        let instance_count = status.instances.len();
        let max_replicas = status.unit.redundancy.max_replicas as usize;
        let min_quorum = status.unit.redundancy.min_quorum.max(1);
        for r in &returned {
            self.settle_instance_with_reason(&r.instance_id, &r.node_id, false, reason, now);
        }
        let status = self.units.get_mut(unit_id).expect("unit exists");
        status.dispute_open = false;
        status.returned.clear();
        for r in &returned {
            if !status.unit.placement.excluded_nodes.contains(&r.node_id) {
                status.unit.placement.excluded_nodes.push(r.node_id.clone());
            }
        }
        let outcome = if instance_count < max_replicas {
            status.state = WorkState::Unsent;
            status.attempts += 1;
            let attempt = status.attempts;
            self.emit(
                now,
                SchedulerEvent::Retried {
                    unit: unit_id.clone(),
                    attempt,
                    reason: reason.to_string(),
                },
            );
            ValidationOutcome::Pending {
                returned: 0,
                needed: min_quorum,
            }
        } else {
            status.state = WorkState::Invalid;
            self.emit(
                now,
                SchedulerEvent::Failed {
                    unit: unit_id.clone(),
                    reason: format!("{reason}; the replica ceiling allows no further attempt"),
                },
            );
            ValidationOutcome::Invalid {
                reason: reason.to_string(),
            }
        };
        self.emit(
            now,
            SchedulerEvent::DisputeSettled {
                unit: unit_id.clone(),
                winner: None,
                reason: reason.to_string(),
            },
        );
        Ok(outcome)
    }

    /// Periodic housekeeping: expires units past their deadline, fails instances on workers that
    /// went offline, and flags stragglers for duplication. Returns the events it produced.
    pub fn tick(&mut self, now: u64, workers: &[WorkerProfile]) -> Vec<SchedulerEventRecord> {
        let first_event = self.events.len();

        // 1. Deadlines.
        let expired: Vec<WorkUnitId> = self
            .units
            .values()
            .filter(|s| {
                s.validated.is_none()
                    && !s.state.is_terminal()
                    && s.unit.deadline_unix_ms.is_some_and(|d| d < now)
            })
            .map(|s| s.unit.id.clone())
            .collect();
        for unit_id in expired {
            if let Some(status) = self.units.get_mut(&unit_id) {
                status.state = WorkState::Expired;
                let instances = status.instances.clone();
                for i in instances {
                    if let Some(inst) = self.instances.get_mut(&i)
                        && inst.state.is_live()
                    {
                        inst.state = WorkState::Expired;
                    }
                    self.straggler_flagged.remove(&i);
                }
            }
            self.emit(now, SchedulerEvent::Expired { unit: unit_id });
        }

        // 2. Workers that went away.
        let offline: Vec<InstanceId> = self
            .instances
            .values()
            .filter(|i| matches!(i.state, WorkState::Assigned | WorkState::Running))
            .filter(|i| !workers.iter().any(|w| w.id() == &i.node_id && w.online))
            .map(|i| i.instance_id.clone())
            .collect();
        for instance_id in offline {
            let _ = self.report_failure(&instance_id, "worker offline", now);
        }

        // 3. Stragglers.
        let mut to_flag = Vec::new();
        for status in self.units.values() {
            if status.validated.is_some() || status.state.is_terminal() || status.dispute_open {
                continue;
            }
            if status.instances.len() >= status.unit.redundancy.max_replicas as usize {
                continue;
            }
            for instance_id in &status.instances {
                let Some(inst) = self.instances.get(instance_id) else {
                    continue;
                };
                if !matches!(inst.state, WorkState::Assigned | WorkState::Running)
                    || self.straggler_flagged.contains(instance_id)
                {
                    continue;
                }
                // A measured worker is judged against its own measured speed for this kind of
                // work, an unmeasured one against the unit's modelled estimate.
                let expected_ms = workers
                    .iter()
                    .find(|w| w.id() == &inst.node_id)
                    .map(|w| w.expected_wall_ms(&status.unit, &self.policy, now))
                    .unwrap_or(status.unit.cost.est_wall_ms);
                let timeout_ms = (expected_ms * self.policy.straggler_factor)
                    .max(self.policy.straggler_min_ms);
                let since = inst.started_unix_ms.unwrap_or(inst.assigned_unix_ms);
                if now.saturating_sub(since) as f64 > timeout_ms {
                    to_flag.push((status.unit.id.clone(), instance_id.clone()));
                    break;
                }
            }
        }
        for (unit_id, instance_id) in to_flag {
            self.straggler_flagged.insert(instance_id.clone());
            if let Some(status) = self.units.get_mut(&unit_id) {
                status.extra_replicas_wanted += 1;
            }
            self.emit(
                now,
                SchedulerEvent::Duplicated {
                    unit: unit_id,
                    original: instance_id,
                    reason: "straggler timeout".to_string(),
                },
            );
        }

        self.events[first_event..].to_vec()
    }

    /// Marks a validated unit as consumed and returns its validated result.
    pub fn assimilate(
        &mut self,
        unit_id: &WorkUnitId,
        now: u64,
    ) -> Result<WorkResultRecord, GridError> {
        let status = self
            .units
            .get_mut(unit_id)
            .ok_or_else(|| GridError::UnknownUnit(unit_id.clone()))?;
        let Some(result) = status.validated.clone() else {
            return Err(GridError::InvalidTransition {
                unit: unit_id.clone(),
                from: status.state,
                to: WorkState::Assimilated,
            });
        };
        status.state = WorkState::Assimilated;
        self.emit(
            now,
            SchedulerEvent::Assimilated {
                unit: unit_id.clone(),
            },
        );
        Ok(result)
    }

    /// Fails a unit and every live instance of it.
    pub fn cancel(&mut self, unit_id: &WorkUnitId, now: u64) -> Result<(), GridError> {
        let status = self
            .units
            .get_mut(unit_id)
            .ok_or_else(|| GridError::UnknownUnit(unit_id.clone()))?;
        if status.state.is_terminal() {
            return Ok(());
        }
        status.state = WorkState::Failed;
        let instances = status.instances.clone();
        for i in instances {
            if let Some(inst) = self.instances.get_mut(&i)
                && inst.state.is_live()
            {
                inst.state = WorkState::Failed;
            }
            self.straggler_flagged.remove(&i);
        }
        self.emit(
            now,
            SchedulerEvent::Cancelled {
                unit: unit_id.clone(),
            },
        );
        Ok(())
    }

    pub fn status(&self, unit_id: &WorkUnitId) -> Option<&UnitStatus> {
        self.units.get(unit_id)
    }

    pub fn instance(&self, instance_id: &InstanceId) -> Option<&WorkInstance> {
        self.instances.get(instance_id)
    }

    pub fn units(&self) -> impl Iterator<Item = &UnitStatus> {
        self.units.values()
    }

    pub fn events(&self) -> &[SchedulerEventRecord] {
        &self.events
    }

    pub fn drain_events(&mut self) -> Vec<SchedulerEventRecord> {
        std::mem::take(&mut self.events)
    }

    pub fn stats(&self) -> QueueStats {
        let mut stats = QueueStats {
            total: self.units.len(),
            ready: self.ready_units().len(),
            ..QueueStats::default()
        };
        for s in self.units.values() {
            if !s.blocked_on.is_empty() {
                stats.blocked += 1;
            }
            if s.dispute_open {
                stats.disputes_open += 1;
            }
            match s.state {
                WorkState::Unsent => stats.unsent += 1,
                WorkState::Assigned => stats.assigned += 1,
                WorkState::Running => stats.running += 1,
                WorkState::Returned => stats.returned += 1,
                WorkState::Validated => stats.validated += 1,
                WorkState::Invalid => stats.invalid += 1,
                WorkState::Assimilated => stats.assimilated += 1,
                WorkState::Expired => stats.expired += 1,
                WorkState::Failed => stats.failed += 1,
            }
        }
        stats.instances_live = self
            .instances
            .values()
            .filter(|i| i.state.is_live())
            .count();
        stats
    }
}

/// The mesh summed up as one machine: the spec sheet of the computer the nodes assemble.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PoolInventory {
    pub nodes_total: usize,
    pub nodes_online: usize,
    pub total_fp16_tflops: f64,
    /// FP16 TFLOPS on online nodes, weighted by polite availability.
    pub available_fp16_tflops: f64,
    pub total_npu_tops: f64,
    /// VRAM plus unified memory across online nodes, in GiB.
    pub total_fast_memory_gb: f64,
    pub total_system_ram_gb: f64,
    pub total_nvme_gb: f64,
    pub total_cpu_threads: u32,
    pub free_slots: u32,
    pub min_hop_latency_ms: f64,
    pub max_hop_latency_ms: f64,
    pub backends: Vec<String>,
    /// Online workers with enough validated samples of some kind of work for their measured cost
    /// to steer routing.
    #[serde(default)]
    pub measured_workers: usize,
    /// Validated compute samples folded into the workers' estimates.
    #[serde(default)]
    pub compute_samples: u64,
}

impl PoolInventory {
    pub fn from_workers(workers: &[WorkerProfile]) -> Self {
        let min_samples = SchedulerPolicy::default().min_compute_samples;
        let online: Vec<&WorkerProfile> = workers.iter().filter(|w| w.online).collect();
        let mut backends: Vec<String> = online
            .iter()
            .map(|w| format!("{:?}", w.node.backend))
            .collect();
        backends.sort();
        backends.dedup();
        let hops: Vec<f64> = online.iter().map(|w| w.hop_latency_ms()).collect();
        let round = |v: f64| (v * 100.0).round() / 100.0;
        Self {
            nodes_total: workers.len(),
            nodes_online: online.len(),
            total_fp16_tflops: round(online.iter().map(|w| w.node.fp16_tflops).sum()),
            available_fp16_tflops: round(
                online
                    .iter()
                    .map(|w| w.node.fp16_tflops * w.availability())
                    .sum(),
            ),
            total_npu_tops: round(online.iter().map(|w| w.node.npu_tops).sum()),
            total_fast_memory_gb: round(
                online
                    .iter()
                    .map(|w| w.node.memory.fast_accelerator_memory_gb())
                    .sum(),
            ),
            total_system_ram_gb: round(online.iter().map(|w| w.node.memory.system_ram_gb).sum()),
            total_nvme_gb: round(online.iter().map(|w| w.node.memory.nvme_cache_gb).sum()),
            total_cpu_threads: online.iter().map(|w| w.node.cpu_threads).sum(),
            free_slots: online
                .iter()
                .map(|w| w.max_concurrent.saturating_sub(w.in_flight))
                .sum(),
            min_hop_latency_ms: hops.iter().copied().fold(f64::INFINITY, f64::min),
            max_hop_latency_ms: hops.iter().copied().fold(0.0, f64::max),
            backends,
            measured_workers: online
                .iter()
                .filter(|w| w.compute.values().any(|e| e.samples >= min_samples))
                .count(),
            compute_samples: workers
                .iter()
                .flat_map(|w| w.compute.values())
                .map(|e| e.samples as u64)
                .sum(),
        }
        .finite()
    }

    fn finite(mut self) -> Self {
        if !self.min_hop_latency_ms.is_finite() {
            self.min_hop_latency_ms = 0.0;
        }
        self
    }

    /// One line for a status bar: the machine the mesh adds up to.
    pub fn summary_line(&self) -> String {
        format!(
            "{} of {} nodes online · {:.0} TFLOPS FP16 ({:.0} available) · {:.0} GB fast memory · {:.0} GB RAM · {:.1} TB NVMe · {} threads · {} free slots",
            self.nodes_online,
            self.nodes_total,
            self.total_fp16_tflops,
            self.available_fp16_tflops,
            self.total_fast_memory_gb,
            self.total_system_ram_gb,
            self.total_nvme_gb / 1024.0,
            self.total_cpu_threads,
            self.free_slots
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ContentKind, ContentRef, CostEstimate, PlacementConstraints, RedundancyPolicy, Replica,
        ResidencyTier,
    };
    use kadmium_core::{ComputeBackend, NodeTrustTier, PoliteState, sample_personal_cluster};

    fn node(id: &str, tier: NodeTrustTier) -> HardwareNode {
        let mut n = sample_personal_cluster()
            .into_iter()
            .next()
            .expect("sample cluster has nodes");
        n.id = NodeId(id.to_string());
        n.hostname = id.to_string();
        n.trust_tier = tier;
        n.polite_state = PoliteState::IdleDedicated;
        n
    }

    fn worker(id: &str, tier: NodeTrustTier) -> WorkerProfile {
        WorkerProfile::from_node(node(id, tier))
    }

    fn expert(name: &str) -> ContentRef {
        ContentRef::named(ContentKind::ExpertBank, name, 6_300_000)
    }

    fn result(instance: &WorkInstance, digest: u64, check: f64, now: u64) -> WorkResultRecord {
        WorkResultRecord::new(
            instance.unit_id.clone(),
            instance.instance_id.clone(),
            instance.node_id.clone(),
            digest,
            5.0,
            now,
        )
        .with_check_value(check)
    }

    #[test]
    fn locality_wins_between_otherwise_equal_workers() {
        let e45 = expert("m/L25/E045");
        let mut catalog = ReplicaCatalog::new();
        catalog.record(Replica::new(
            e45.clone(),
            NodeId("b".into()),
            ResidencyTier::Vram,
            0,
        ));
        let mut workers = vec![
            worker("a", NodeTrustTier::TrustedCore),
            worker("b", NodeTrustTier::TrustedCore),
        ];
        let unit = WorkUnit::new("slice", AppKind::ExpertSlice).with_inputs(vec![e45]);
        let ranked = Matchmaker::rank(&unit, &workers, &catalog, &SchedulerPolicy::default(), 0);
        assert_eq!(ranked[0].node_id, NodeId("b".into()));
        assert_eq!(ranked[0].locality, 1.0);
        assert_eq!(ranked[1].locality, 0.0);

        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit(unit, 0).unwrap();
        let assigned = queue.dispatch_push(&mut workers, &catalog, 1);
        assert_eq!(assigned.len(), 1);
        assert_eq!(assigned[0].node_id, NodeId("b".into()));
        assert_eq!(workers[1].in_flight, 1);
        assert_eq!(workers[0].in_flight, 0);
    }

    #[test]
    fn boundary_units_never_leave_owned_hardware() {
        let catalog = ReplicaCatalog::new();
        let mut workers = vec![
            worker("peer", NodeTrustTier::UntrustedAuditedMiddleLayersOnly),
            worker("friend", NodeTrustTier::VerifiedPeerMiddleLayersOnly),
            worker("mine", NodeTrustTier::TrustedCore),
        ];
        let unit = WorkUnit::new("ingress", AppKind::TrunkForward)
            .with_placement(PlacementConstraints::trusted_only());
        let ranked = Matchmaker::rank(&unit, &workers, &catalog, &SchedulerPolicy::default(), 0);
        assert_eq!(ranked[0].node_id, NodeId("mine".into()));
        assert!(ranked[1..].iter().all(|m| !m.eligible));
        assert!(ranked[1].reason.contains("trust") || ranked[1].reason.contains("boundary"));

        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit(unit, 0).unwrap();
        let assigned = queue.dispatch_push(&mut workers, &catalog, 1);
        assert_eq!(assigned[0].node_id, NodeId("mine".into()));
    }

    #[test]
    fn bit_exact_quorum_validates_agreement_and_invalidates_the_minority() {
        let catalog = ReplicaCatalog::new();
        let mut workers: Vec<WorkerProfile> = ["a", "b", "c"]
            .iter()
            .map(|id| worker(id, NodeTrustTier::UntrustedAuditedMiddleLayersOnly))
            .collect();
        let unit = WorkUnit::new("audit", AppKind::VerdeAudit).with_redundancy(RedundancyPolicy {
            min_quorum: 2,
            target_replicas: 2,
            max_replicas: 4,
            validation: ValidationRule::BitExactDigest,
        });
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit(unit, 0).unwrap();

        let assigned = queue.dispatch_push(&mut workers, &catalog, 1);
        assert_eq!(assigned.len(), 2);
        assert_ne!(assigned[0].node_id, assigned[1].node_id);

        let first = queue
            .report_result(
                &assigned[0].instance_id,
                result(&assigned[0], 0xAAAA, 1.0, 10),
                10,
            )
            .unwrap();
        assert_eq!(
            first,
            ValidationOutcome::Pending {
                returned: 1,
                needed: 2
            }
        );

        // Disagreement: a tie-breaker is requested and dispatched to a third node.
        let second = queue
            .report_result(
                &assigned[1].instance_id,
                result(&assigned[1], 0xBBBB, 1.0, 11),
                11,
            )
            .unwrap();
        assert_eq!(
            second,
            ValidationOutcome::Pending {
                returned: 2,
                needed: 2
            }
        );
        let tie_break = queue.dispatch_push(&mut workers, &catalog, 12);
        assert_eq!(tie_break.len(), 1);
        assert!(
            ![assigned[0].node_id.clone(), assigned[1].node_id.clone()]
                .contains(&tie_break[0].node_id)
        );

        let third = queue
            .report_result(
                &tie_break[0].instance_id,
                result(&tie_break[0], 0xAAAA, 1.0, 13),
                13,
            )
            .unwrap();
        assert_eq!(
            third,
            ValidationOutcome::Validated {
                node: assigned[0].node_id.clone()
            }
        );
        let status = queue.status(&"audit".into()).unwrap();
        assert_eq!(status.state, WorkState::Validated);
        assert_eq!(
            queue.instance(&assigned[1].instance_id).unwrap().state,
            WorkState::Invalid
        );
        assert_eq!(
            queue.instance(&assigned[0].instance_id).unwrap().state,
            WorkState::Validated
        );
        assert!(queue.events().iter().any(|e| matches!(
            e.event,
            SchedulerEvent::Invalidated { ref node, .. } if *node == assigned[1].node_id
        )));
        assert_eq!(
            queue.assimilate(&"audit".into(), 14).unwrap().digest_fnv1a,
            0xAAAA
        );
        assert_eq!(queue.stats().assimilated, 1);
    }

    #[test]
    fn verde_disagreement_opens_a_dispute_the_runtime_resolves() {
        let catalog = ReplicaCatalog::new();
        let mut workers = vec![
            worker("a", NodeTrustTier::UntrustedAuditedMiddleLayersOnly),
            worker("b", NodeTrustTier::UntrustedAuditedMiddleLayersOnly),
        ];
        let unit = WorkUnit::new("slice", AppKind::ExpertSlice)
            .with_redundancy(RedundancyPolicy::untrusted());
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit(unit, 0).unwrap();
        let assigned = queue.dispatch_push(&mut workers, &catalog, 1);
        assert_eq!(assigned.len(), 2);
        queue
            .report_result(&assigned[0].instance_id, result(&assigned[0], 1, 0.0, 2), 2)
            .unwrap();
        let outcome = queue
            .report_result(&assigned[1].instance_id, result(&assigned[1], 2, 0.0, 3), 3)
            .unwrap();
        assert!(
            matches!(outcome, ValidationOutcome::DisputeOpened { ref nodes } if nodes.len() == 2)
        );
        assert!(queue.status(&"slice".into()).unwrap().dispute_open);
        assert!(queue.ready_units().is_empty());
        assert_eq!(queue.stats().disputes_open, 1);

        let resolved = queue
            .resolve_dispute(&"slice".into(), &assigned[1].node_id, 4)
            .unwrap();
        assert_eq!(
            resolved,
            ValidationOutcome::Validated {
                node: assigned[1].node_id.clone()
            }
        );
        assert_eq!(
            queue.instance(&assigned[0].instance_id).unwrap().state,
            WorkState::Invalid
        );
        assert!(
            queue
                .resolve_dispute(&"slice".into(), &assigned[1].node_id, 5)
                .is_err()
        );
        assert!(queue.events().iter().any(|e| matches!(
            &e.event,
            SchedulerEvent::DisputeSettled { winner: Some(node), .. } if *node == assigned[1].node_id
        )));
    }

    #[test]
    fn a_dispute_the_referee_reproduces_for_nobody_excludes_both_and_retries_elsewhere() {
        let catalog = ReplicaCatalog::new();
        let mut workers = vec![
            worker("a", NodeTrustTier::UntrustedAuditedMiddleLayersOnly),
            worker("b", NodeTrustTier::UntrustedAuditedMiddleLayersOnly),
            worker("c", NodeTrustTier::UntrustedAuditedMiddleLayersOnly),
            worker("d", NodeTrustTier::UntrustedAuditedMiddleLayersOnly),
        ];
        let unit = WorkUnit::new("stage", AppKind::StageForward)
            .with_redundancy(RedundancyPolicy::untrusted());
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit(unit, 0).unwrap();
        let first = queue.dispatch_push(&mut workers, &catalog, 1);
        assert_eq!(first.len(), 2);
        queue
            .report_result(&first[0].instance_id, result(&first[0], 1, 0.0, 2), 2)
            .unwrap();
        let outcome = queue
            .report_result(&first[1].instance_id, result(&first[1], 2, 0.0, 3), 3)
            .unwrap();
        assert!(matches!(outcome, ValidationOutcome::DisputeOpened { .. }));

        let rejected = queue
            .reject_dispute(&"stage".into(), "Verde: layer 3 reproduced for neither", 4)
            .unwrap();
        assert_eq!(
            rejected,
            ValidationOutcome::Pending {
                returned: 0,
                needed: 2
            }
        );
        let status = queue.status(&"stage".into()).unwrap();
        assert!(!status.dispute_open);
        assert_eq!(status.state, WorkState::Unsent);
        assert_eq!(status.unit.placement.excluded_nodes.len(), 2);
        assert!(status.returned.is_empty());
        for i in &first {
            assert_eq!(queue.instance(&i.instance_id).unwrap().state, WorkState::Invalid);
            workers
                .iter_mut()
                .find(|w| w.id() == &i.node_id)
                .unwrap()
                .in_flight = 0;
        }
        let invalidated = queue
            .events()
            .iter()
            .filter(|e| matches!(&e.event, SchedulerEvent::Invalidated { reason, .. } if reason.contains("layer 3")))
            .count();
        assert_eq!(invalidated, 2);
        assert!(queue.events().iter().any(|e| matches!(
            &e.event,
            SchedulerEvent::DisputeSettled { winner: None, .. }
        )));

        // The retry goes to the two nodes that were not disputants.
        let second = queue.dispatch_push(&mut workers, &catalog, 5);
        assert_eq!(second.len(), 2);
        let disputants: Vec<&NodeId> = first.iter().map(|i| &i.node_id).collect();
        assert!(second.iter().all(|i| !disputants.contains(&&i.node_id)));
        queue
            .report_result(&second[0].instance_id, result(&second[0], 9, 0.0, 6), 6)
            .unwrap();
        let validated = queue
            .report_result(&second[1].instance_id, result(&second[1], 9, 0.0, 7), 7)
            .unwrap();
        assert!(matches!(validated, ValidationOutcome::Validated { .. }));
        assert!(queue.reject_dispute(&"stage".into(), "again", 8).is_err());
    }

    #[test]
    fn scalar_tolerance_accepts_cross_silicon_drift() {
        let catalog = ReplicaCatalog::new();
        let mut workers = vec![
            worker("cuda", NodeTrustTier::VerifiedPeerMiddleLayersOnly),
            worker("metal", NodeTrustTier::VerifiedPeerMiddleLayersOnly),
        ];
        let unit =
            WorkUnit::new("stage", AppKind::StageForward).with_redundancy(RedundancyPolicy {
                min_quorum: 2,
                target_replicas: 2,
                max_replicas: 3,
                validation: ValidationRule::ScalarTolerance { max_abs_diff: 1e-3 },
            });
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit(unit, 0).unwrap();
        let assigned = queue.dispatch_push(&mut workers, &catalog, 1);
        queue
            .report_result(
                &assigned[0].instance_id,
                result(&assigned[0], 7, 1.0000, 2),
                2,
            )
            .unwrap();
        let outcome = queue
            .report_result(
                &assigned[1].instance_id,
                result(&assigned[1], 9, 1.0004, 3),
                3,
            )
            .unwrap();
        assert!(matches!(outcome, ValidationOutcome::Validated { .. }));
        assert_eq!(
            queue.instance(&assigned[1].instance_id).unwrap().state,
            WorkState::Validated
        );
    }

    #[test]
    fn stragglers_are_duplicated_and_the_first_result_wins() {
        let catalog = ReplicaCatalog::new();
        let mut workers = vec![
            worker("a", NodeTrustTier::TrustedCore),
            worker("b", NodeTrustTier::TrustedCore),
        ];
        let unit = WorkUnit::new("slow", AppKind::DreamReplay)
            .with_cost(CostEstimate::new(1e9, 0, 0, 100.0));
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit(unit, 0).unwrap();
        let first = queue.dispatch_push(&mut workers, &catalog, 0);
        assert_eq!(first.len(), 1);
        queue.report_started(&first[0].instance_id, 0).unwrap();

        assert!(queue.tick(300, &workers).is_empty());
        let events = queue.tick(600, &workers);
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, SchedulerEvent::Duplicated { .. }))
        );

        let dup = queue.dispatch_push(&mut workers, &catalog, 601);
        assert_eq!(dup.len(), 1);
        assert_ne!(dup[0].node_id, first[0].node_id);
        assert_eq!(dup[0].duplicate_of, Some(first[0].instance_id.clone()));
        assert_eq!(dup[0].attempt, 2);

        let outcome = queue
            .report_result(&dup[0].instance_id, result(&dup[0], 42, 0.0, 700), 700)
            .unwrap();
        assert!(
            matches!(outcome, ValidationOutcome::Validated { ref node } if *node == dup[0].node_id)
        );
        // The late original agrees, so it is not invalidated.
        let late = queue
            .report_result(&first[0].instance_id, result(&first[0], 42, 0.0, 900), 900)
            .unwrap();
        assert!(matches!(late, ValidationOutcome::Validated { .. }));
        assert_eq!(
            queue.instance(&first[0].instance_id).unwrap().state,
            WorkState::Validated
        );
    }

    #[test]
    fn deadlines_order_the_queue_and_expire_units() {
        let catalog = ReplicaCatalog::new();
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue
            .submit(
                WorkUnit::new("later", AppKind::Checksum).with_deadline(5_000),
                0,
            )
            .unwrap();
        queue
            .submit(
                WorkUnit::new("soon", AppKind::Checksum).with_deadline(1_000),
                0,
            )
            .unwrap();
        queue
            .submit(
                WorkUnit::new("urgent-no-deadline", AppKind::Checksum).with_priority(9),
                0,
            )
            .unwrap();
        let order: Vec<_> = queue.ready_units().iter().map(|u| u.id.0.clone()).collect();
        assert_eq!(order, vec!["soon", "later", "urgent-no-deadline"]);

        let mut workers = vec![worker("a", NodeTrustTier::TrustedCore)];
        let assigned = queue.dispatch_push(&mut workers, &catalog, 10);
        assert_eq!(assigned[0].unit_id, WorkUnitId::new("soon"));

        let events = queue.tick(6_000, &workers);
        let expired: Vec<_> = events
            .iter()
            .filter(|e| matches!(e.event, SchedulerEvent::Expired { .. }))
            .collect();
        assert_eq!(expired.len(), 2);
        assert_eq!(queue.stats().expired, 2);
        assert_eq!(
            queue.instance(&assigned[0].instance_id).unwrap().state,
            WorkState::Expired
        );
        assert!(matches!(
            queue.report_result(
                &assigned[0].instance_id,
                result(&assigned[0], 1, 0.0, 7_000),
                7_000
            ),
            Err(GridError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn polite_yield_blocks_new_work_and_offline_workers_trigger_retries() {
        let catalog = ReplicaCatalog::new();
        let mut gaming = worker("gaming", NodeTrustTier::TrustedCore);
        gaming.node.polite_state = PoliteState::GamingOrHeavyApp;
        let idle = worker("idle", NodeTrustTier::TrustedCore);
        let mut workers = vec![gaming, idle];

        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue
            .submit(WorkUnit::new("u", AppKind::WasiSandbox), 0)
            .unwrap();
        let assigned = queue.dispatch_push(&mut workers, &catalog, 1);
        assert_eq!(assigned[0].node_id, NodeId("idle".into()));

        workers[1].online = false;
        let events = queue.tick(2, &workers);
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, SchedulerEvent::Retried { attempt: 1, .. }))
        );
        assert_eq!(
            queue.instance(&assigned[0].instance_id).unwrap().state,
            WorkState::Failed
        );
        assert_eq!(queue.status(&"u".into()).unwrap().state, WorkState::Unsent);

        workers[0].node.polite_state = PoliteState::LightDesktopUse;
        let again = queue.dispatch_push(&mut workers, &catalog, 3);
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].node_id, NodeId("gaming".into()));
        assert_eq!(again[0].attempt, 2);
    }

    #[test]
    fn failures_retry_until_max_attempts() {
        let catalog = ReplicaCatalog::new();
        let policy = SchedulerPolicy {
            max_attempts: 2,
            ..SchedulerPolicy::default()
        };
        let mut queue = WorkQueue::new(policy);
        queue
            .submit(WorkUnit::new("flaky", AppKind::Checksum), 0)
            .unwrap();
        let mut workers = vec![worker("a", NodeTrustTier::TrustedCore)];

        let first = queue.dispatch_push(&mut workers, &catalog, 1);
        queue
            .report_failure(&first[0].instance_id, "crash", 2)
            .unwrap();
        assert_eq!(
            queue.status(&"flaky".into()).unwrap().state,
            WorkState::Unsent
        );
        workers[0].in_flight = 0;
        let second = queue.dispatch_push(&mut workers, &catalog, 3);
        assert_eq!(second.len(), 1);
        queue
            .report_failure(&second[0].instance_id, "crash", 4)
            .unwrap();
        assert_eq!(
            queue.status(&"flaky".into()).unwrap().state,
            WorkState::Failed
        );
        assert!(queue.ready_units().is_empty());
        assert!(
            queue
                .events()
                .iter()
                .any(|e| matches!(e.event, SchedulerEvent::Failed { .. }))
        );
    }

    #[test]
    fn graph_units_release_as_prerequisites_validate() {
        let catalog = ReplicaCatalog::new();
        let graph = WorkGraph::chain(
            "decode",
            vec![
                WorkUnit::new("s0", AppKind::StageForward),
                WorkUnit::new("s1", AppKind::StageForward),
            ],
        );
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit_graph(&graph, 0).unwrap();
        assert!(queue.submit_graph(&graph, 0).is_err());
        assert_eq!(queue.stats().blocked, 1);
        let ready: Vec<_> = queue.ready_units().iter().map(|u| u.id.0.clone()).collect();
        assert_eq!(ready, vec!["decode/s0"]);

        let mut workers = vec![worker("a", NodeTrustTier::TrustedCore)];
        let assigned = queue.dispatch_push(&mut workers, &catalog, 1);
        assert_eq!(assigned.len(), 1);
        queue
            .report_result(&assigned[0].instance_id, result(&assigned[0], 1, 0.0, 2), 2)
            .unwrap();
        assert!(queue.events().iter().any(|e| matches!(
            e.event,
            SchedulerEvent::Released { ref unit } if unit.name() == "s1"
        )));
        let ready: Vec<_> = queue.ready_units().iter().map(|u| u.id.0.clone()).collect();
        assert_eq!(ready, vec!["decode/s1"]);
        assert_eq!(queue.stats().blocked, 0);
        assert!(!queue.graph_settled(&graph));
        let s1 = graph.qualify(&"s1".into());
        assert!(queue.cancel(&s1, 3).is_ok());
        assert_eq!(queue.status(&s1).unwrap().state, WorkState::Failed);
        assert!(queue.graph_settled(&graph));
    }

    #[test]
    fn graphs_with_the_same_unit_names_share_a_queue_and_chain_across_graphs() {
        let catalog = ReplicaCatalog::new();
        let step = |t: u64| {
            WorkGraph::chain(
                format!("sess/step-{t}"),
                vec![
                    WorkUnit::new("stage-00", AppKind::StageForward),
                    WorkUnit::new("stage-01", AppKind::StageForward),
                ],
            )
        };
        let first = step(1);
        let second = step(2);
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        queue.submit_graph(&first, 0).unwrap();
        // The same names again in another graph are other units.
        queue
            .submit_graph_after(
                &second,
                &[("stage-00".into(), first.qualify(&"stage-01".into()))],
                0,
            )
            .unwrap();
        assert_eq!(queue.stats().total, 4);
        let ready: Vec<_> = queue.ready_units().iter().map(|u| u.id.0.clone()).collect();
        assert_eq!(ready, vec!["sess/step-1/stage-00"], "step 2 waits for step 1's sink");

        let mut workers = vec![worker("a", NodeTrustTier::TrustedCore)];
        let finish = |queue: &mut WorkQueue, workers: &mut Vec<WorkerProfile>, now: u64| {
            let assigned = queue.dispatch_push(workers, &catalog, now);
            assert_eq!(assigned.len(), 1, "one unit ready at a time along the chain");
            queue
                .report_result(
                    &assigned[0].instance_id,
                    result(&assigned[0], 1, 0.0, now),
                    now,
                )
                .unwrap();
            workers[0].in_flight = 0;
            assigned[0].unit_id.clone()
        };
        assert_eq!(finish(&mut queue, &mut workers, 1).0, "sess/step-1/stage-00");
        assert!(!queue.graph_settled(&first));
        assert_eq!(finish(&mut queue, &mut workers, 2).0, "sess/step-1/stage-01");
        assert!(queue.graph_settled(&first));
        assert!(queue.events().iter().any(|e| matches!(
            e.event,
            SchedulerEvent::Released { ref unit } if unit.0 == "sess/step-2/stage-00"
        )));
        assert_eq!(finish(&mut queue, &mut workers, 3).0, "sess/step-2/stage-00");
        assert_eq!(finish(&mut queue, &mut workers, 4).0, "sess/step-2/stage-01");
        assert!(queue.graph_settled(&second));

        // A prerequisite that already validated blocks nothing; a failed one is refused, and so
        // is an unknown one.
        let third = step(3);
        queue
            .submit_graph_after(
                &third,
                &[("stage-00".into(), second.qualify(&"stage-01".into()))],
                5,
            )
            .unwrap();
        assert!(queue.ready_units().iter().any(|u| u.id.0 == "sess/step-3/stage-00"));
        queue.cancel(&third.qualify(&"stage-01".into()), 6).unwrap();
        let fourth = step(4);
        assert!(matches!(
            queue.submit_graph_after(
                &fourth,
                &[("stage-00".into(), third.qualify(&"stage-01".into()))],
                7,
            ),
            Err(GridError::PrerequisiteFailed { .. })
        ));
        let fifth = step(5);
        assert!(matches!(
            queue.submit_graph_after(&fifth, &[("stage-00".into(), "nowhere".into())], 8),
            Err(GridError::UnknownUnit(_))
        ));
    }

    #[test]
    fn pull_requests_respect_capacity_and_batch_limits() {
        let catalog = ReplicaCatalog::new();
        let mut queue = WorkQueue::new(SchedulerPolicy::default());
        for i in 0..5 {
            queue
                .submit(WorkUnit::new(format!("r{i}"), AppKind::DreamReplay), 0)
                .unwrap();
        }
        let mut w = worker("sbc", NodeTrustTier::TrustedCore).with_max_concurrent(2);
        let handed = queue.request_work(&mut w, &catalog, 1, 4);
        assert_eq!(handed.len(), 2);
        assert_eq!(w.in_flight, 2);
        assert!(queue.request_work(&mut w, &catalog, 2, 4).is_empty());
        assert_eq!(queue.stats().assigned, 2);
        assert_eq!(queue.stats().instances_live, 2);
        assert_eq!(queue.drain_events().len(), 7);
        assert!(queue.events().is_empty());
    }

    #[test]
    fn pool_inventory_adds_the_mesh_up_as_one_machine() {
        let mut workers: Vec<WorkerProfile> = sample_personal_cluster()
            .into_iter()
            .map(WorkerProfile::from_node)
            .collect();
        workers[2].online = false;
        let inventory = PoolInventory::from_workers(&workers);
        assert_eq!(inventory.nodes_total, workers.len());
        assert_eq!(inventory.nodes_online, workers.len() - 1);
        assert!(inventory.total_fp16_tflops > 0.0);
        assert!(inventory.available_fp16_tflops <= inventory.total_fp16_tflops);
        assert!(inventory.total_fast_memory_gb > 0.0);
        assert!(inventory.total_system_ram_gb > 0.0);
        assert!(inventory.min_hop_latency_ms <= inventory.max_hop_latency_ms);
        assert!(
            inventory
                .backends
                .contains(&format!("{:?}", ComputeBackend::Cuda))
        );
        assert!(inventory.summary_line().contains("nodes online"));
        assert_eq!(PoolInventory::from_workers(&[]).min_hop_latency_ms, 0.0);
    }

    fn resident_catalog(nodes: &[&str], content: &[ContentRef]) -> ReplicaCatalog {
        let mut catalog = ReplicaCatalog::new();
        for n in nodes {
            for c in content {
                catalog.record(Replica::new(
                    c.clone(),
                    NodeId(n.to_string()),
                    ResidencyTier::Vram,
                    0,
                ));
            }
        }
        catalog
    }

    fn feed(worker: &mut WorkerProfile, app: &str, ms: f64, samples: u32, policy: &SchedulerPolicy) {
        for i in 0..samples {
            worker.record_compute(app, ms, policy, 1_000 + i as u64);
        }
    }

    #[test]
    fn measured_compute_cost_ranks_equal_latency_residents_once_sampled() {
        let policy = SchedulerPolicy::default();
        let bank = expert("L010/E001");
        let catalog = resident_catalog(&["a", "b"], std::slice::from_ref(&bank));
        let mut a = worker("a", NodeTrustTier::TrustedCore);
        let mut b = worker("b", NodeTrustTier::TrustedCore);
        a.measured_rtt_ms = 1.0;
        b.measured_rtt_ms = 1.0;
        let unit = WorkUnit::new("slice", AppKind::ExpertSlice)
            .with_inputs(vec![bank])
            .with_cost(CostEstimate::new(1e9, 0, 0, 20.0).with_work_units(1));
        let now = 2_000;

        // Same capability vector, same modelled score: the node id breaks the tie towards `a`.
        let ranked = Matchmaker::rank(&unit, &[a.clone(), b.clone()], &catalog, &policy, now);
        assert_eq!(ranked[0].node_id.0, "a");
        assert!(ranked[0].completion_estimate_ms.is_none());

        // `a` is slow and `b` fast, but seven samples change nothing.
        feed(&mut a, "ExpertSlice", 5.0, 7, &policy);
        feed(&mut b, "ExpertSlice", 2.0, 7, &policy);
        let ranked = Matchmaker::rank(&unit, &[a.clone(), b.clone()], &catalog, &policy, now);
        assert_eq!(ranked[0].node_id.0, "a", "seven samples keep the modelled ranking");
        assert!(ranked.iter().all(|m| m.completion_estimate_ms.is_none()));

        // The eighth sample lets the measured cost decide.
        feed(&mut a, "ExpertSlice", 5.0, 1, &policy);
        feed(&mut b, "ExpertSlice", 2.0, 1, &policy);
        let ranked = Matchmaker::rank(&unit, &[a.clone(), b.clone()], &catalog, &policy, now);
        assert_eq!(ranked[0].node_id.0, "b");
        let est_b = ranked[0].completion_estimate_ms.expect("measured");
        let est_a = ranked[1].completion_estimate_ms.expect("measured");
        assert!((est_b - 3.0).abs() < 1e-9, "hop plus one unit of work: {est_b}");
        assert!((est_a - 6.0).abs() < 1e-9, "{est_a}");

        // Queue pressure counts: three instances already on `b` put it behind `a`.
        b.in_flight = 3;
        let ranked = Matchmaker::rank(&unit, &[a.clone(), b.clone()], &catalog, &policy, now);
        assert_eq!(ranked[0].node_id.0, "a");
        assert!((ranked[1].completion_estimate_ms.unwrap() - 9.0).abs() < 1e-9);

        // A unit of a kind nobody has measured is still ranked on the modelled terms.
        let other = WorkUnit::new("audit", AppKind::VerdeAudit);
        let ranked = Matchmaker::rank(&other, &[a.clone(), b.clone()], &catalog, &policy, now);
        assert!(ranked.iter().all(|m| m.completion_estimate_ms.is_none()));
    }

    #[test]
    fn residency_outranks_measured_speed_and_stale_samples_expire() {
        let policy = SchedulerPolicy::default();
        let banks = [expert("L011/E002"), expert("L011/E003")];
        // The resident node holds half of the unit's input bytes; the remote node holds none.
        let catalog = resident_catalog(&["slow-resident"], &banks[..1]);
        let mut slow = worker("slow-resident", NodeTrustTier::TrustedCore);
        let mut fast = worker("fast-remote", NodeTrustTier::TrustedCore);
        feed(&mut slow, "ExpertSlice", 500.0, 8, &policy);
        feed(&mut fast, "ExpertSlice", 1.0, 8, &policy);
        let unit = WorkUnit::new("slice", AppKind::ExpertSlice)
            .with_inputs(banks.to_vec())
            .with_cost(CostEstimate::new(1e9, 0, 0, 20.0));
        let workers = [fast.clone(), slow.clone()];

        let ranked = Matchmaker::rank(&unit, &workers, &catalog, &policy, 2_000);
        assert_eq!(ranked[0].node_id.0, "slow-resident");
        assert!(ranked[0].score < ranked[1].score, "the remote node scores higher, yet loses");
        assert!(ranked[0].completion_estimate_ms > ranked[1].completion_estimate_ms);

        let mut lenient = policy.clone();
        lenient.residency_first = false;
        let ranked = Matchmaker::rank(&unit, &workers, &catalog, &lenient, 2_000);
        assert_eq!(ranked[0].node_id.0, "fast-remote");

        // Stale samples fall back to the modelled terms.
        let later = 2_000 + policy.compute_sample_ttl_ms + 1;
        assert!(fast.completion_estimate_ms(&unit, &policy, 2_000).is_some());
        assert!(fast.completion_estimate_ms(&unit, &policy, later).is_none());
        let ranked = Matchmaker::rank(&unit, &workers, &catalog, &lenient, later);
        assert!(ranked.iter().all(|m| m.completion_estimate_ms.is_none()));
    }

    #[test]
    fn only_validated_results_become_compute_samples() {
        let catalog = ReplicaCatalog::new();
        let mut workers = vec![
            worker("a", NodeTrustTier::TrustedCore),
            worker("b", NodeTrustTier::TrustedCore),
            worker("c", NodeTrustTier::TrustedCore),
        ];
        let mut queue = WorkQueue::new(SchedulerPolicy::default());

        // A failed attempt is not a sample; the validated retry is, per unit of work.
        let flaky = WorkUnit::new("flaky", AppKind::DreamReplay)
            .with_cost(CostEstimate::new(1e9, 0, 0, 10.0).with_work_units(4))
            .with_placement(PlacementConstraints {
                excluded_nodes: vec![NodeId("c".into())],
                ..PlacementConstraints::default()
            });
        queue.submit(flaky, 0).unwrap();
        let first = queue.dispatch_push(&mut workers, &catalog, 0);
        assert_eq!(first.len(), 1);
        queue
            .report_failure(&first[0].instance_id, "crash", 1)
            .unwrap();
        assert_eq!(queue.feed_workers(&mut workers, 1), 0);
        let retry = queue.dispatch_push(&mut workers, &catalog, 2);
        assert_eq!(retry.len(), 1);
        let record = result(&retry[0], 7, 0.0, 3).with_units_processed(4);
        queue
            .report_result(&retry[0].instance_id, record, 3)
            .unwrap();
        assert_eq!(queue.feed_workers(&mut workers, 3), 1);
        let measured = workers
            .iter()
            .find(|w| w.id() == &retry[0].node_id)
            .unwrap();
        let ewma = &measured.compute["DreamReplay"];
        assert_eq!(ewma.samples, 1);
        assert!((ewma.ms_per_unit - 1.25).abs() < 1e-9, "5 ms over 4 units");
        assert!(
            workers
                .iter()
                .filter(|w| w.id() != &retry[0].node_id)
                .all(|w| w.compute.is_empty())
        );

        // Under a bit-exact quorum the minority result is invalidated and never sampled.
        for w in workers.iter_mut() {
            w.compute.clear();
        }
        let audited = WorkUnit::new("audited", AppKind::VerdeAudit).with_redundancy(
            RedundancyPolicy {
                min_quorum: 2,
                target_replicas: 3,
                max_replicas: 3,
                validation: ValidationRule::BitExactDigest,
            },
        );
        queue.submit(audited, 10).unwrap();
        let trio = queue.dispatch_push(&mut workers, &catalog, 10);
        assert_eq!(trio.len(), 3);
        queue
            .report_result(&trio[0].instance_id, result(&trio[0], 1, 0.0, 11), 11)
            .unwrap();
        queue
            .report_result(&trio[1].instance_id, result(&trio[1], 2, 0.0, 12), 12)
            .unwrap();
        let outcome = queue
            .report_result(&trio[2].instance_id, result(&trio[2], 1, 0.0, 13), 13)
            .unwrap();
        assert!(matches!(outcome, ValidationOutcome::Validated { .. }));
        assert_eq!(queue.feed_workers(&mut workers, 13), 2);
        let sampled: Vec<&NodeId> = workers
            .iter()
            .filter(|w| !w.compute.is_empty())
            .map(|w| w.id())
            .collect();
        assert_eq!(sampled.len(), 2);
        assert!(!sampled.contains(&&trio[1].node_id), "the disagreeing node is not sampled");
    }

    #[test]
    fn straggler_timeout_follows_the_measured_speed_of_the_worker() {
        let catalog = ReplicaCatalog::new();
        let policy = SchedulerPolicy::default();
        let mut workers = vec![
            worker("a", NodeTrustTier::TrustedCore),
            worker("b", NodeTrustTier::TrustedCore),
        ];
        // `a` is measured at 400 ms per unit of replay work, far above the unit's 100 ms estimate.
        feed(&mut workers[0], "DreamReplay", 400.0, 8, &policy);
        let unit = WorkUnit::new("slow", AppKind::DreamReplay)
            .with_cost(CostEstimate::new(1e9, 0, 0, 100.0))
            .with_placement(PlacementConstraints {
                excluded_nodes: vec![NodeId("b".into())],
                ..PlacementConstraints::default()
            });
        let mut queue = WorkQueue::new(policy);
        queue.submit(unit, 0).unwrap();
        let first = queue.dispatch_push(&mut workers, &catalog, 0);
        assert_eq!(first[0].node_id.0, "a");
        queue.report_started(&first[0].instance_id, 0).unwrap();

        // Modelled, the timeout would be max(100 x 2.5, 500) = 500 ms; measured it is 1000 ms.
        assert!(queue.tick(600, &workers).is_empty());
        assert!(
            queue
                .tick(1_100, &workers)
                .iter()
                .any(|e| matches!(e.event, SchedulerEvent::Duplicated { .. }))
        );
    }
}
