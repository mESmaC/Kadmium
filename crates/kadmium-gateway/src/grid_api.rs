//! Grid substrate endpoints: the mesh as one machine (`/api/v1/grid/inventory`), the planner's
//! shape choice (`/api/v1/grid/plan`), work graphs run on the live mesh (`/api/v1/grid/run`,
//! `/api/v1/grid/train`, `/api/v1/grid/replay`) and the credit ledger (`/api/v1/grid/credits`).
//! The OpenAI, Anthropic and pipeline endpoints route their decode step through the same
//! dispatcher via [`execute_step_or_legacy`], falling back to the legacy stage loop only when a
//! grid step cannot be built.

use crate::GatewayState;
use axum::{
    Json,
    extract::{Query, State},
};
use kadmium_core::{
    HardwareNode, ModelSpec, NodeId, NodeRole, ResourceOffer, VerifiedComputeReceipt, WorkloadKind,
};
use kadmium_grid::{
    AppKind, CreditEntry, PlanApproval, PoolInventory, Replica, ReplicaCatalog, ResidencyTier,
    SchedulerPolicy, StagingJob, TelemetryStore, WorkGraph, WorkUnit, WorkerProfile, fnv1a_64,
};
use kadmium_network::LiveWorkerRegistration;
use kadmium_runtime::{
    DeMoReduceReport, DecodeLoopOptions, DecodeLoopReport, DecodeSessionPlan,
    DistributedLanExecutor, DistributedMoEPipelineExecutor, DistributedPipelineExecutionReport,
    ExecutorContext, ExecutorRegistry, GridDispatcher, GridRunConfig, GridRunReport,
    LiveHopTelemetry, LocalExecutor, SessionSpeculation, TokenBatchInt8, VerdeDisputeReport,
    VerificationSetup, build_initial_resident_ktb_bank, prompt_seed_hidden, run_decode_loop,
    suffix_batch,
};
use kadmium_sharding::{
    AcceptanceAccounting, BalanceReport, CALIBRATION_MAX_STEPS, CALIBRATION_MAX_WALL_MS,
    CalibrationBinding, CalibrationRecord, CalibrationStore, ClusterShardPlan, CombineParams,
    DraftProfile, ExecutionStrategy, ExpertPlacementPlan, ExpertPlacementPlanner, ExpertPopularity,
    ExpertSliceParams, KvPageNames, KvSessionState, KvTierPlan, KvTierPlanner, KvTierPolicy,
    LinkSegments, PlacementFit, RepinPlanner, RepinPolicy, RepinProposal, RepinSwap, ReplicaChange,
    StageForwardParams, StepGraphBuilder, StepsPerSecondProfile, StrategyChoice, StrategyCostModel,
    SweepPoint, TrainStageParams, TrunkForwardParams, TrunkRange, UsageSummary, VerificationPolicy,
    records_for_placement,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Summary of the last grid run, kept in the gateway state and shown on the inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLastRun {
    pub graph_id: String,
    /// `decode`, `train` or `replay`.
    pub kind: String,
    pub strategy: Option<ExecutionStrategy>,
    pub units_total: usize,
    pub units_validated: usize,
    pub success: bool,
    pub wall_ms: f64,
    /// The cost model's figure for the shape that ran; modelled, never measured.
    pub modelled_ms_per_token: Option<f64>,
    pub credits_granted: f64,
    pub nodes_used: Vec<String>,
    pub epoch: u64,
    /// Validated results that became measured compute samples on their workers.
    #[serde(default)]
    pub compute_samples: usize,
    /// Verde disputes the run settled.
    #[serde(default)]
    pub disputes: usize,
}

/// The placements the coordinator keeps between runs for a model: both shapes, planned once
/// against the mesh and the offers, so the pins the planner learned stay put, the live re-pin
/// adapts them a few swaps at a time, and a calibration binds to a digest that holds still.
/// Re-planned only when the key no longer matches the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPlacement {
    pub choice: StrategyChoice,
    pub chain_plan: ExpertPlacementPlan,
    pub ep_plan: ExpertPlacementPlan,
    pub key: PlacementKey,
    pub created_unix_ms: u64,
    /// Re-pin boundaries applied to the active placement, and the swaps they made in total.
    pub repins_applied: usize,
    pub swaps_applied: usize,
    /// Copies the load balancer added or dropped at those boundaries, in total.
    #[serde(default)]
    pub replica_changes_applied: usize,
}

impl CachedPlacement {
    pub fn plan_for(&self, strategy: ExecutionStrategy) -> &ExpertPlacementPlan {
        match strategy {
            ExecutionStrategy::PipelineChain => &self.chain_plan,
            ExecutionStrategy::ExpertParallel => &self.ep_plan,
        }
    }

    pub fn plan_for_mut(&mut self, strategy: ExecutionStrategy) -> &mut ExpertPlacementPlan {
        match strategy {
            ExecutionStrategy::PipelineChain => &mut self.chain_plan,
            ExecutionStrategy::ExpertParallel => &mut self.ep_plan,
        }
    }

    /// The placement the chosen shape runs on.
    pub fn active(&self) -> &ExpertPlacementPlan {
        self.plan_for(self.choice.chosen)
    }
}

/// What a cached placement was planned against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementKey {
    pub epoch: u64,
    pub node_ids: Vec<String>,
    pub offers_digest: u64,
}

struct MeshSnapshot {
    epoch: u64,
    nodes: Vec<HardwareNode>,
    live_workers: BTreeMap<String, LiveWorkerRegistration>,
    local_node: HardwareNode,
    inference_model: ModelSpec,
    training_model: ModelSpec,
    telemetry: TelemetryStore,
    /// The offer in force on every node: the owner's where one was sent, else the default.
    offers: BTreeMap<NodeId, ResourceOffer>,
    approvals: BTreeMap<String, PlanApproval>,
    calibrations: CalibrationStore,
}

/// The offer in force on every node of the mesh.
pub fn effective_offers(
    nodes: &[HardwareNode],
    explicit: &BTreeMap<String, ResourceOffer>,
) -> BTreeMap<NodeId, ResourceOffer> {
    nodes
        .iter()
        .map(|n| {
            let offer = match explicit.get(&n.id.0) {
                Some(offer) => offer.clone().clamped_to(n),
                None => n.default_offer(),
            };
            (n.id.clone(), offer)
        })
        .collect()
}

fn snapshot(state: &Arc<Mutex<GatewayState>>) -> MeshSnapshot {
    let guard = state.lock().expect("gateway state lock");
    let local_node = guard
        .nodes
        .iter()
        .find(|n| guard.live_workers.contains_key(&n.id.0))
        .or_else(|| guard.nodes.first())
        .cloned()
        .unwrap_or_else(|| kadmium_core::sample_personal_cluster().remove(0));
    MeshSnapshot {
        epoch: guard.epoch,
        nodes: guard.nodes.clone(),
        live_workers: guard.live_workers.clone(),
        local_node,
        inference_model: guard.inference_model.clone(),
        training_model: guard.training_model.clone(),
        telemetry: guard.grid_telemetry.clone(),
        offers: effective_offers(&guard.nodes, &guard.grid_offers),
        approvals: guard.grid_plan_approvals.clone(),
        calibrations: guard.grid_calibrations.clone(),
    }
}

/// The identity a measurement of `strategy` on `placement_digest` binds to on this mesh: every
/// node, with the version each live worker runs (`in-process` for the coordinator's own nodes).
fn calibration_binding(
    snap: &MeshSnapshot,
    model_id: &str,
    strategy: ExecutionStrategy,
    placement_digest: u64,
    rows: u32,
) -> CalibrationBinding {
    let node_ids: Vec<String> = snap.nodes.iter().map(|n| n.id.0.clone()).collect();
    let versions: BTreeMap<String, String> = snap
        .nodes
        .iter()
        .map(|n| {
            let version = snap
                .live_workers
                .get(&n.id.0)
                .map(|r| r.installed_version.clone())
                .unwrap_or_else(|| "in-process".to_string());
            (n.id.0.clone(), version)
        })
        .collect();
    CalibrationBinding::new(
        model_id,
        strategy,
        placement_digest,
        node_ids,
        versions,
        rows,
    )
}

/// Attaches the calibrations that bind to the cached shapes, letting measurements decide the
/// shape when both have one.
fn refresh_measurements(cached: &mut CachedPlacement, snap: &MeshSnapshot, model_id: &str) {
    let chain = snap
        .calibrations
        .lookup(&calibration_binding(
            snap,
            model_id,
            ExecutionStrategy::PipelineChain,
            cached.chain_plan.digest(),
            1,
        ))
        .cloned();
    let ep = snap
        .calibrations
        .lookup(&calibration_binding(
            snap,
            model_id,
            ExecutionStrategy::ExpertParallel,
            cached.ep_plan.digest(),
            1,
        ))
        .cloned();
    cached.choice.apply_measurements(chain, ep);
}

/// Worker profiles for every node of the mesh: live registrations where they exist, in-process
/// profiles for simulated nodes so the scheduler can still place work on them, and on each the
/// measured telemetry the coordinator has kept from earlier runs.
pub fn mesh_workers(
    nodes: &[HardwareNode],
    live_workers: &BTreeMap<String, LiveWorkerRegistration>,
    telemetry: &TelemetryStore,
    offers: &BTreeMap<NodeId, ResourceOffer>,
) -> Vec<WorkerProfile> {
    let mut workers = GridDispatcher::workers_from_registrations(live_workers);
    for node in nodes {
        if !live_workers.contains_key(&node.id.0) {
            workers.push(WorkerProfile::from_node(node.clone()));
        }
    }
    workers = workers
        .into_iter()
        .map(|w| match offers.get(w.id()) {
            Some(offer) => w.with_offer(offer.clone()),
            None => w,
        })
        .collect();
    telemetry.apply_to_all(&mut workers);
    workers
}

fn local_executor(local_node: &HardwareNode) -> LocalExecutor {
    let (bank, _) = build_initial_resident_ktb_bank("deepseek-v3-671b-moe-q4k", 1);
    LocalExecutor {
        registry: ExecutorRegistry::with_defaults(),
        ctx: ExecutorContext::new(
            &local_node.id.0,
            &local_node.hostname,
            Arc::new(Mutex::new(bank)),
        ),
    }
}

fn placement_key(snap: &MeshSnapshot) -> PlacementKey {
    let mut node_ids: Vec<String> = snap.nodes.iter().map(|n| n.id.0.clone()).collect();
    node_ids.sort();
    let offers_json = serde_json::to_vec(&snap.offers).unwrap_or_default();
    PlacementKey {
        epoch: snap.epoch,
        node_ids,
        offers_digest: fnv1a_64(&offers_json),
    }
}

/// Forces a shape onto a choice without re-planning: both shapes are cached.
fn force_shape(choice: &mut StrategyChoice, forced: Option<ExecutionStrategy>) {
    if let Some(strategy) = forced
        && strategy != choice.chosen
    {
        choice.chosen = strategy;
        choice.decided_by = "forced".to_string();
        choice.rationale = format!("shape forced to {strategy:?}; {}", choice.rationale);
    }
}

/// The placement in force for a model: the cached shapes while the mesh they were planned
/// against is unchanged, else fresh plans seeded from the model's learned routing heat and
/// cached. Calibrations that bind to the cached shapes are attached every time, so measurements
/// decide the shape as soon as both shapes have one. The boolean says whether the placement came
/// from the cache.
fn current_placement(
    state: &Arc<Mutex<GatewayState>>,
    snap: &MeshSnapshot,
    model: &ModelSpec,
    forced: Option<ExecutionStrategy>,
) -> (StrategyChoice, ExpertPlacementPlan, bool) {
    let key = placement_key(snap);
    let popularity = {
        let mut guard = state.lock().expect("gateway state lock");
        if let Some(cached) = guard.grid_placements.get_mut(&model.id)
            && cached.key == key
        {
            refresh_measurements(cached, snap, &model.id);
            let mut choice = cached.choice.clone();
            force_shape(&mut choice, forced);
            let plan = cached.plan_for(choice.chosen).clone();
            return (choice, plan, true);
        }
        let ledger = guard.usage_for(&model.id);
        (!ledger.is_empty()).then(|| ledger.popularity())
    };
    let (choice, chain_plan, ep_plan) = cost_model(
        &snap.nodes,
        &snap.live_workers,
        &snap.telemetry,
    )
    .choose_both_with_offers(model, &snap.nodes, popularity.as_ref(), &snap.offers);
    let mut cached = CachedPlacement {
        choice,
        chain_plan,
        ep_plan,
        key,
        created_unix_ms: now_unix_ms(),
        repins_applied: 0,
        swaps_applied: 0,
        replica_changes_applied: 0,
    };
    refresh_measurements(&mut cached, snap, &model.id);
    let mut choice = cached.choice.clone();
    force_shape(&mut choice, forced);
    let plan = cached.plan_for(choice.chosen).clone();
    let mut guard = state.lock().expect("gateway state lock");
    guard.grid_placements.insert(model.id.clone(), cached);
    (choice, plan, false)
}

/// One re-pin boundary on the model's cached placement: the proposal from the learned heat
/// (tier swaps, and the load balancer's copies in its incremental mode), and (with `apply`) the
/// changes made and the transfers they imply, with the cost model's critical-path expert reads
/// before and after, modelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridRepinOutcome {
    pub model_id: String,
    pub proposal: RepinProposal,
    pub applied: bool,
    pub staging_jobs: Vec<StagingJob>,
    pub placement_digest: u64,
    pub repins_applied: usize,
    pub swaps_applied: usize,
    /// Copies the balancer added or dropped across every boundary so far.
    pub replica_changes_applied: usize,
    /// Expert reads on the step's critical path at one row, modelled, before and after.
    pub expert_read_ms_before: f64,
    pub expert_read_ms_after: f64,
}

fn repin_now(
    state: &Arc<Mutex<GatewayState>>,
    model: &ModelSpec,
    policy: &RepinPolicy,
    apply: bool,
) -> Result<GridRepinOutcome, String> {
    let mut guard = state.lock().expect("gateway state lock");
    let heat = guard.usage_for(&model.id).popularity();
    let nodes = guard.nodes.clone();
    let segments = link_segments(&nodes, &guard.live_workers);
    let cost = cost_model(&nodes, &guard.live_workers, &guard.grid_telemetry);
    let Some(cached) = guard.grid_placements.get_mut(&model.id) else {
        return Err(format!(
            "no placement in force for `{}`: ask for the plan or run a step first",
            model.id
        ));
    };
    let active = cached.choice.chosen;
    let proposal = RepinPlanner::propose_balanced(
        cached.plan_for(active),
        model,
        &nodes,
        &heat,
        &segments,
        policy,
    );
    let before = cost
        .estimate(model, cached.plan_for(active), &nodes, Some(&heat))
        .expert_read_ms;
    let staging_jobs = if apply && !proposal.is_empty() {
        let expert_bytes = (model.single_expert_size_gb * 1_073_741_824.0) as u64;
        cached.repins_applied += 1;
        cached.swaps_applied += proposal.swaps.len();
        cached.replica_changes_applied += proposal.replica_changes.len();
        RepinPlanner::apply(cached.plan_for_mut(active), &proposal, expert_bytes)
    } else {
        Vec::new()
    };
    let after = cost
        .estimate(model, cached.plan_for(active), &nodes, Some(&heat))
        .expert_read_ms;
    Ok(GridRepinOutcome {
        model_id: model.id.clone(),
        applied: apply && !proposal.is_empty(),
        placement_digest: cached.active().digest(),
        repins_applied: cached.repins_applied,
        swaps_applied: cached.swaps_applied,
        replica_changes_applied: cached.replica_changes_applied,
        expert_read_ms_before: before,
        expert_read_ms_after: after,
        proposal,
        staging_jobs,
    })
}

/// Which nodes share a fast link: one segment per `/24` of the live workers' LAN addresses,
/// the round-trip rule for nodes without a registration.
fn link_segments(
    nodes: &[HardwareNode],
    live_workers: &BTreeMap<String, LiveWorkerRegistration>,
) -> LinkSegments {
    let lan_ips: BTreeMap<NodeId, String> = live_workers
        .values()
        .filter(|r| !r.worker_lan_ip.trim().is_empty())
        .map(|r| (r.node.id.clone(), r.worker_lan_ip.clone()))
        .collect();
    LinkSegments::derive_with_lan_ips(nodes, &lan_ips)
}

/// The cost model with what the mesh has measured: round trips from the registrations, memory
/// streaming rates from the workers' own probes, and the link segments the balancer packs
/// across first.
fn cost_model(
    nodes: &[HardwareNode],
    live_workers: &BTreeMap<String, LiveWorkerRegistration>,
    telemetry: &TelemetryStore,
) -> StrategyCostModel {
    let mut model = StrategyCostModel::default().with_segments(link_segments(nodes, live_workers));
    for reg in live_workers.values() {
        if reg.last_measured_rtt_ms > 0.0 {
            model = model.with_measured_rtt(reg.node.id.clone(), reg.last_measured_rtt_ms);
        }
    }
    for (node, t) in &telemetry.workers {
        if let Some(gbps) = t.measured_stream_gbps {
            model = model.with_measured_stream(node.clone(), gbps);
        }
    }
    model
}

fn parse_strategy(value: Option<&str>) -> Option<ExecutionStrategy> {
    match value.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("chain") | Some("pipeline") | Some("pipeline-chain") => {
            Some(ExecutionStrategy::PipelineChain)
        }
        Some("ep") | Some("expert-parallel") | Some("fan-out") | Some("fanout") => {
            Some(ExecutionStrategy::ExpertParallel)
        }
        _ => None,
    }
}

fn now_unix_ms() -> u64 {
    LiveWorkerRegistration::now_unix_ms()
}

fn layer_span(unit: &WorkUnit) -> (u32, u32) {
    match unit.app {
        AppKind::StageForward => unit
            .params_json::<StageForwardParams>()
            .map(|p| (p.start_layer, p.end_layer))
            .unwrap_or((0, 0)),
        AppKind::TrunkForward => unit
            .params_json::<TrunkForwardParams>()
            .map(|p| (p.layer, p.layer))
            .unwrap_or((0, 0)),
        AppKind::ExpertSlice => unit
            .params_json::<ExpertSliceParams>()
            .map(|p| (p.layer, p.layer))
            .unwrap_or((0, 0)),
        AppKind::Combine => unit
            .params_json::<CombineParams>()
            .map(|p| (p.layer, p.layer))
            .unwrap_or((0, 0)),
        AppKind::TrainForwardBackward => unit
            .params_json::<TrainStageParams>()
            .map(|p| (p.start_layer, p.end_layer))
            .unwrap_or((0, 0)),
        _ => (0, 0),
    }
}

/// The grid run's traces as the hop telemetry the Control Room and Companion already render.
pub fn hops_from_traces(
    graph: &WorkGraph,
    report: &GridRunReport,
    nodes: &[HardwareNode],
    live_workers: &BTreeMap<String, LiveWorkerRegistration>,
) -> Vec<LiveHopTelemetry> {
    report
        .traces
        .iter()
        .filter(|t| t.ok)
        .enumerate()
        .map(|(i, t)| {
            let (start_layer, end_layer) = graph.unit(&t.unit_id).map(layer_span).unwrap_or((0, 0));
            let registration = live_workers.get(&t.node_id.0);
            let hostname = registration
                .map(|r| r.node.hostname.clone())
                .or_else(|| {
                    nodes
                        .iter()
                        .find(|n| n.id == t.node_id)
                        .map(|n| n.hostname.clone())
                })
                .unwrap_or_else(|| t.node_id.0.clone());
            LiveHopTelemetry {
                stage_index: i,
                node_id: t.node_id.0.clone(),
                hostname,
                transport_mode: format!("grid {} via {}", t.app, t.transport),
                worker_tcp_addr: registration.map(|r| r.worker_tcp_addr.clone()),
                start_layer,
                end_layer,
                wire_plus_compute_ms: t.wall_ms,
                compute_exec_ms: t.exec_ms,
                boundary_int8_bytes: 0,
                repops_digest_fnv1a: t.digest_fnv1a,
            }
        })
        .collect()
}

/// Everything a grid decode step produced.
pub struct GridStepOutcome {
    pub strategy: ExecutionStrategy,
    pub choice: StrategyChoice,
    pub placement: ExpertPlacementPlan,
    pub graph_id: String,
    pub report: GridRunReport,
    pub hops: Vec<LiveHopTelemetry>,
    pub modelled_ms_per_token: f64,
    /// RepOps digest of the last unit that ran; its form depends on the unit's kind.
    pub final_digest_fnv1a: u64,
    /// FNV-1a of the bytes the step's sink units produced: the same for every shape that
    /// produces the same bytes, which is what calibration's digest gate compares.
    pub final_output_fnv1a: u64,
    /// Whether the step ran on the placement already in force rather than a fresh plan.
    pub placement_cached: bool,
    /// Routing accesses folded into the model's usage ledger.
    pub routing_recorded: usize,
    /// The re-pin boundary applied after the step, when one was asked for.
    pub repin: Option<GridRepinOutcome>,
}

/// Books a finished run: credits and receipts, the last-run record, and the measured telemetry
/// the workers accumulated, absorbed into the store and saved for the next coordinator.
#[allow(clippy::too_many_arguments)]
fn record_run(
    state: &Arc<Mutex<GatewayState>>,
    kind: &str,
    graph_id: &str,
    strategy: Option<ExecutionStrategy>,
    report: &GridRunReport,
    modelled_ms_per_token: Option<f64>,
    epoch: u64,
    workers: &[WorkerProfile],
) {
    let mut nodes_used: Vec<String> = report
        .traces
        .iter()
        .filter(|t| t.ok)
        .map(|t| t.node_id.0.clone())
        .collect();
    nodes_used.sort();
    nodes_used.dedup();
    let mut guard = state.lock().expect("gateway state lock");
    guard.grid_runs += 1;
    guard
        .grid_credit_entries
        .extend(report.credits.iter().cloned());
    guard.grid_receipts.extend(report.receipts.iter().cloned());
    guard.grid_telemetry.absorb_all(workers, now_unix_ms());
    let _ = guard.save_grid_telemetry();
    guard.grid_last_run = Some(GridLastRun {
        graph_id: graph_id.to_string(),
        kind: kind.to_string(),
        strategy,
        units_total: report.units_total,
        units_validated: report.units_validated,
        success: report.success,
        wall_ms: report.wall_ms,
        modelled_ms_per_token,
        credits_granted: report.credits.iter().map(|c| c.credits).sum(),
        nodes_used,
        epoch,
        compute_samples: report.compute_samples,
        disputes: report.disputes.len(),
    });
}

/// Runs one decode step of the inference model as a work graph over the mesh: the planner picks
/// the shape (or `forced` sets it), the dispatcher runs the graph over live workers with an
/// in-process executor for simulated nodes, and the run is recorded in the gateway state.
pub async fn run_grid_decode_step(
    state: &Arc<Mutex<GatewayState>>,
    prompt: &str,
    rows: u32,
    forced: Option<ExecutionStrategy>,
) -> Result<GridStepOutcome, String> {
    run_grid_decode_step_with(state, prompt, rows, forced, None).await
}

/// [`run_grid_decode_step`] with an optional re-pin boundary applied to the placement in force
/// once the step has been recorded.
pub async fn run_grid_decode_step_with(
    state: &Arc<Mutex<GatewayState>>,
    prompt: &str,
    rows: u32,
    forced: Option<ExecutionStrategy>,
    repin: Option<RepinPolicy>,
) -> Result<GridStepOutcome, String> {
    let snap = snapshot(state);
    let rows = rows.clamp(1, 64);
    let model = snap.inference_model.clone();
    let (choice, placement, placement_cached) = current_placement(state, &snap, &model, forced);
    if placement.trunk_ranges.is_empty() {
        return Err("no node in the mesh can hold the model's trunk".to_string());
    }
    let strategy = choice.chosen;
    let modelled_ms_per_token = match strategy {
        ExecutionStrategy::PipelineChain => choice.chain.ms_per_token,
        ExecutionStrategy::ExpertParallel => choice.expert_parallel.ms_per_token,
    };

    let session = format!("gateway-e{}", snap.epoch);
    let mut workers = mesh_workers(
        &snap.nodes,
        &snap.live_workers,
        &snap.telemetry,
        &snap.offers,
    );
    // Replicated experts are served by the copy with the lowest measured completion estimate,
    // the primary until the holders are measured.
    let policy = SchedulerPolicy::default();
    let now = now_unix_ms();
    let serving = placement.serving_assignment(|node| {
        workers
            .iter()
            .find(|w| w.id() == node)
            .and_then(|w| w.completion_estimate_for("ExpertSlice", rows as f64, &policy, now))
    });
    let graph = StepGraphBuilder::decode_step_served(
        &model,
        &placement,
        &snap.nodes,
        snap.epoch,
        &session,
        rows,
        &serving,
    );
    let catalog = ReplicaCatalog::assume_placed(&placement.rules, now_unix_ms());
    let seed_row = prompt_seed_hidden(prompt, 64);
    let seed_values: Vec<f32> = (0..rows).flat_map(|_| seed_row.iter().copied()).collect();
    let seed = TokenBatchInt8::quantize(&seed_values, rows, 64, 16)?;
    let config = GridRunConfig {
        epoch: snap.epoch,
        ..GridRunConfig::default()
    }
    .with_seed(StepGraphBuilder::prompt_ref(&graph.id), seed.to_le_bytes());
    let local = local_executor(&snap.local_node);
    let report = GridDispatcher::run_graph(
        &graph,
        &mut workers,
        &catalog,
        &snap.live_workers,
        Some(&local),
        &config,
    )
    .await;
    let hops = hops_from_traces(&graph, &report, &snap.nodes, &snap.live_workers);
    let final_digest_fnv1a = report
        .traces
        .iter()
        .rev()
        .find(|t| t.ok)
        .map(|t| t.digest_fnv1a)
        .unwrap_or(0);
    let final_output_fnv1a = report
        .outputs
        .iter()
        .map(|(_, bytes)| fnv1a_64(bytes))
        .fold(0u64, |acc, d| acc ^ d);
    record_run(
        state,
        "decode",
        &graph.id,
        Some(strategy),
        &report,
        Some(modelled_ms_per_token),
        snap.epoch,
        &workers,
    );
    let routing_recorded = state
        .lock()
        .expect("gateway state lock")
        .record_grid_access(&model.id, &report.accesses);
    if !report.success {
        return Err(format!(
            "grid step `{}` validated {} of {} units",
            graph.id, report.units_validated, report.units_total
        ));
    }
    let repin = match repin {
        Some(policy) => repin_now(state, &model, &policy, true).ok(),
        None => None,
    };
    Ok(GridStepOutcome {
        strategy,
        choice,
        placement,
        graph_id: graph.id.clone(),
        report,
        hops,
        modelled_ms_per_token,
        final_digest_fnv1a,
        final_output_fnv1a,
        placement_cached,
        routing_recorded,
        repin,
    })
}

/// What a decode loop ran on and produced, beside the loop's own report.
pub struct GridDecodeLoopOutcome {
    pub strategy: ExecutionStrategy,
    pub choice: StrategyChoice,
    pub placement: ExpertPlacementPlan,
    pub placement_cached: bool,
    pub modelled_ms_per_token: f64,
    pub routing_recorded: usize,
    pub loop_report: DecodeLoopReport,
    /// The steps-per-second profile the verification scheduler used, when sessions drafted.
    pub profile: Option<StepsPerSecondProfile>,
    /// The replay a named session needed before its steps could run, when its pages had sunk.
    pub resumed: Option<GridReplaySummary>,
    /// The sessions as the registry holds them after the loop.
    pub sessions: Vec<KvSessionState>,
}

/// A replay of one session's bounded suffix, as a resume or on request (ARCHITECTURE 9.12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridReplaySummary {
    pub session_id: String,
    pub graph_id: String,
    /// Tokens recomputed: the window plus the backtrack, bounded by the context.
    pub suffix_tokens: u32,
    pub tokens_total: u64,
    pub window_tokens: u32,
    pub backtrack_tokens: u32,
    /// The tier the pages came back from.
    pub from_tier: ResidencyTier,
    pub units_total: usize,
    pub units_validated: usize,
    /// First dispatch to last return, measured on the kernels the workers run.
    pub wall_ms: f64,
    pub pages_rebuilt: usize,
    /// From the model spec's reference KV figure.
    pub bytes_modelled: u64,
    pub final_digest_fnv1a: u64,
    pub success: bool,
    pub kernels: String,
}

/// Runs the replay graph of a session whose pages left fast memory: one `KvReplay` unit per
/// trunk range on its owner over the bounded suffix, then the pages are recorded resident
/// again and the session marked fast.
pub async fn run_kv_replay(
    state: &Arc<Mutex<GatewayState>>,
    session_id: &str,
) -> Result<GridReplaySummary, String> {
    let snap = snapshot(state);
    let model = snap.inference_model.clone();
    let (session, policy) = {
        let guard = state.lock().expect("gateway state lock");
        let session = guard
            .grid_sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| format!("no session `{session_id}` is known to the grid"))?;
        (session, guard.grid_kv_policy.clone())
    };
    if session.model_id != model.id {
        return Err(format!(
            "session `{session_id}` decoded `{}`, not the model in force `{}`",
            session.model_id, model.id
        ));
    }
    let (_, placement, _) = current_placement(state, &snap, &model, None);
    let graph = StepGraphBuilder::kv_replay_graph(
        &model,
        &placement,
        session_id,
        snap.epoch,
        session.tokens.max(1),
        &policy,
    );
    let suffix_tokens = policy.replay_tokens(session.tokens.max(1));
    let seed = suffix_batch(session_id, session.tokens.max(1), suffix_tokens, 64)?;
    let mut workers = mesh_workers(
        &snap.nodes,
        &snap.live_workers,
        &snap.telemetry,
        &snap.offers,
    );
    let catalog = ReplicaCatalog::assume_placed(&placement.rules, now_unix_ms());
    let config = GridRunConfig {
        epoch: snap.epoch,
        ..GridRunConfig::default()
    }
    .with_seed(
        StepGraphBuilder::replay_input_ref(&graph.id),
        seed.to_le_bytes(),
    );
    let local = local_executor(&snap.local_node);
    let report = GridDispatcher::run_graph(
        &graph,
        &mut workers,
        &catalog,
        &snap.live_workers,
        Some(&local),
        &config,
    )
    .await;
    record_run(
        state,
        "kv-replay",
        &graph.id,
        None,
        &report,
        None,
        snap.epoch,
        &workers,
    );
    let _ = state
        .lock()
        .expect("gateway state lock")
        .record_grid_access(&model.id, &report.accesses);
    let mut pages_rebuilt = 0usize;
    let mut bytes_modelled = 0u64;
    let from_tier = session.tier;
    let now = now_unix_ms();
    {
        let mut guard = state.lock().expect("gateway state lock");
        if report.success {
            let nodes = guard.nodes.clone();
            let mut catalog = std::mem::take(&mut guard.grid_kv_catalog);
            if let Some(s) = guard.grid_sessions.get_mut(session_id) {
                s.last_active_unix_ms = now;
                s.replays += 1;
                s.replayed_tokens += suffix_tokens as u64;
                s.tier = nodes
                    .iter()
                    .find(|n| {
                        placement
                            .trunk_ranges
                            .first()
                            .is_some_and(|r| r.owner == n.id)
                    })
                    .map(KvTierPlanner::fast_tier_of)
                    .unwrap_or(ResidencyTier::Vram);
                pages_rebuilt = KvTierPlanner::record_resident(
                    &mut catalog,
                    &model,
                    &placement,
                    &nodes,
                    s,
                    now,
                );
                bytes_modelled = pages_rebuilt as u64 * KvPageNames::page_bytes(&model, s.tokens);
            }
            guard.grid_kv_catalog = catalog;
        }
    }
    Ok(GridReplaySummary {
        session_id: session_id.to_string(),
        graph_id: graph.id.clone(),
        suffix_tokens,
        tokens_total: session.tokens,
        window_tokens: policy.window_tokens,
        backtrack_tokens: policy.backtrack_tokens,
        from_tier,
        units_total: report.units_total,
        units_validated: report.units_validated,
        wall_ms: report.wall_ms,
        pages_rebuilt,
        bytes_modelled,
        final_digest_fnv1a: report
            .traces
            .iter()
            .rev()
            .find(|t| t.ok)
            .map_or(0, |t| t.digest_fnv1a),
        success: report.success,
        kernels: kadmium_sharding::CALIBRATION_KERNELS_REFERENCE.to_string(),
    })
}

/// Books the sessions a decode loop ran: tokens, activity, and their pages resident in fast
/// memory on the trunk owners.
fn record_sessions(
    state: &Arc<Mutex<GatewayState>>,
    model: &ModelSpec,
    placement: &ExpertPlacementPlan,
    loop_report: &DecodeLoopReport,
) -> Vec<KvSessionState> {
    let now = now_unix_ms();
    let mut guard = state.lock().expect("gateway state lock");
    let nodes = guard.nodes.clone();
    let mut catalog = std::mem::take(&mut guard.grid_kv_catalog);
    let mut out = Vec::new();
    for s in &loop_report.sessions {
        let entry = guard
            .grid_sessions
            .entry(s.session_id.clone())
            .or_insert_with(|| KvSessionState::new(&s.session_id, &model.id, now));
        entry.tokens += s.tokens;
        entry.last_active_unix_ms = now;
        entry.tier = nodes
            .iter()
            .find(|n| {
                placement
                    .trunk_ranges
                    .first()
                    .is_some_and(|r| r.owner == n.id)
            })
            .map(KvTierPlanner::fast_tier_of)
            .unwrap_or(ResidencyTier::Vram);
        KvTierPlanner::record_resident(&mut catalog, model, placement, &nodes, entry, now);
        out.push(entry.clone());
    }
    guard.grid_kv_catalog = catalog;
    out
}

/// How a decode loop's sessions draft: the modelled drafter's profile and DSpark's policy.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeSpeculation {
    pub draft_profile: DraftProfile,
    pub policy: VerificationPolicy,
}

/// The widths the modelled profile is taken at when fewer than two are measured.
const MODELLED_PROFILE_WIDTHS: [u32; 5] = [1, 2, 4, 8, 16];

/// DSpark's `SPS(B)` for a placement: the trusted calibration records of exactly this
/// placement across batch widths when at least two widths are measured, else the cost model's
/// step time at a few widths, labelled modelled.
fn steps_per_second_profile(
    snap: &MeshSnapshot,
    model: &ModelSpec,
    strategy: ExecutionStrategy,
    placement: &ExpertPlacementPlan,
    popularity: Option<&ExpertPopularity>,
) -> StepsPerSecondProfile {
    let by_width = records_for_placement(
        snap.calibrations.records.iter(),
        &model.id,
        strategy,
        placement.digest(),
    );
    if by_width.len() >= 2 {
        return StepsPerSecondProfile::from_calibrations(by_width.values().copied());
    }
    let cost = cost_model(&snap.nodes, &snap.live_workers, &snap.telemetry);
    let steps: Vec<(u32, f64)> = MODELLED_PROFILE_WIDTHS
        .iter()
        .map(|&w| {
            (
                w,
                cost.clone()
                    .with_rows(w)
                    .estimate(model, placement, &snap.nodes, popularity)
                    .step_ms,
            )
        })
        .collect();
    StepsPerSecondProfile::from_modelled_steps(&steps)
}

/// What a decode loop runs for each session: the prompt, rows, steps, mode and drafting.
pub struct LoopShape {
    pub prompt: String,
    pub rows: u32,
    pub steps: u32,
    pub pipelined: bool,
    pub speculation: Option<DecodeSpeculation>,
    pub session_ids: Vec<String>,
}

/// The decode loop over an explicit mesh and placement: the workers of `nodes` (live where
/// registered, in-process otherwise), one session plan per id, and the verification profile
/// when the sessions draft. Returns the loop's report, the workers with what they learned, and
/// the profile used.
async fn decode_loop_on_mesh(
    state: &Arc<Mutex<GatewayState>>,
    snap: &MeshSnapshot,
    model: &ModelSpec,
    nodes: &[HardwareNode],
    placement: &ExpertPlacementPlan,
    strategy: ExecutionStrategy,
    shape: &LoopShape,
) -> Result<
    (
        DecodeLoopReport,
        Vec<WorkerProfile>,
        Option<StepsPerSecondProfile>,
    ),
    String,
> {
    let live_workers: BTreeMap<String, LiveWorkerRegistration> = snap
        .live_workers
        .iter()
        .filter(|(id, _)| nodes.iter().any(|n| n.id.0 == **id))
        .map(|(id, r)| (id.clone(), r.clone()))
        .collect();
    let mut workers = mesh_workers(nodes, &live_workers, &snap.telemetry, &snap.offers);
    let policy = SchedulerPolicy::default();
    let now = now_unix_ms();
    let rows = shape.rows;
    let serving = placement.serving_assignment(|node| {
        workers
            .iter()
            .find(|w| w.id() == node)
            .and_then(|w| w.completion_estimate_for("ExpertSlice", rows as f64, &policy, now))
    });
    let mut plans = Vec::with_capacity(shape.session_ids.len());
    for (i, session_id) in shape.session_ids.iter().enumerate() {
        let seed_row = prompt_seed_hidden(&format!("{} [session {i}]", shape.prompt), 64);
        let seed_values: Vec<f32> = (0..rows).flat_map(|_| seed_row.iter().copied()).collect();
        plans.push(DecodeSessionPlan {
            speculation: shape.speculation.as_ref().map(|s| SessionSpeculation {
                profile: s.draft_profile.clone(),
                seed: fnv1a_64(session_id.as_bytes()),
            }),
            session_id: session_id.clone(),
            model: model.clone(),
            placement: placement.clone(),
            nodes: nodes.to_vec(),
            rows,
            steps: shape.steps,
            first_epoch: snap.epoch,
            serving: serving.clone(),
            seed: TokenBatchInt8::quantize(&seed_values, rows, 64, 16)?,
        });
    }
    let mut options = DecodeLoopOptions::pipelined(shape.pipelined);
    let mut profile = None;
    if let Some(spec) = &shape.speculation {
        let popularity = {
            let mut guard = state.lock().expect("gateway state lock");
            let ledger = guard.usage_for(&model.id);
            (!ledger.is_empty()).then(|| ledger.popularity())
        };
        let sps = steps_per_second_profile(snap, model, strategy, placement, popularity.as_ref());
        options = options.with_verification(VerificationSetup {
            policy: spec.policy.clone(),
            profile: sps.clone(),
        });
        profile = Some(sps);
    }
    let catalog = ReplicaCatalog::assume_placed(&placement.rules, now_unix_ms());
    let config = GridRunConfig {
        epoch: snap.epoch,
        ..GridRunConfig::default()
    };
    let local = local_executor(&snap.local_node);
    let loop_report = run_decode_loop(
        &plans,
        &mut workers,
        &catalog,
        &live_workers,
        Some(&local),
        &config,
        &options,
    )
    .await;
    Ok((loop_report, workers, profile))
}

/// Runs `steps` decode steps for `sessions` sessions of the inference model through one
/// [`run_decode_loop`]: pipelined, each session's next step is planned and queued while its
/// current one runs and the sessions' graphs interleave in one queue; sequential, one graph at
/// a time, the old shape. Every session starts from the prompt with its index appended, so the
/// sessions' bytes differ and a row that crossed sessions would show. With `speculation` every
/// session drafts (one row each) and the verification scheduler sizes its verify batch per
/// step from the profile and the modelled confidences.
#[allow(clippy::too_many_arguments)]
pub async fn run_grid_decode_loop(
    state: &Arc<Mutex<GatewayState>>,
    prompt: &str,
    rows: u32,
    steps: u32,
    sessions: u32,
    forced: Option<ExecutionStrategy>,
    pipelined: bool,
    speculation: Option<DecodeSpeculation>,
    session: Option<String>,
) -> Result<GridDecodeLoopOutcome, String> {
    // A named session continues: when its pages have sunk, it replays its suffix first.
    let mut resumed = None;
    if let Some(id) = session.as_deref() {
        let needs_replay = state
            .lock()
            .expect("gateway state lock")
            .grid_sessions
            .get(id)
            .is_some_and(KvSessionState::needs_replay);
        if needs_replay {
            resumed = Some(run_kv_replay(state, id).await?);
        }
    }
    let snap = snapshot(state);
    let rows = if speculation.is_some() {
        1
    } else {
        rows.clamp(1, 64)
    };
    let steps = steps.clamp(1, 64);
    let sessions = if session.is_some() {
        1
    } else {
        sessions.clamp(1, 8)
    };
    let model = snap.inference_model.clone();
    let (choice, placement, placement_cached) = current_placement(state, &snap, &model, forced);
    if placement.trunk_ranges.is_empty() {
        return Err("no node in the mesh can hold the model's trunk".to_string());
    }
    let strategy = choice.chosen;
    let modelled_ms_per_token = match strategy {
        ExecutionStrategy::PipelineChain => choice.chain.ms_per_token,
        ExecutionStrategy::ExpertParallel => choice.expert_parallel.ms_per_token,
    };
    let session_ids: Vec<String> = (0..sessions)
        .map(|i| match &session {
            Some(id) => id.clone(),
            None => format!("gateway-e{}-s{i}", snap.epoch),
        })
        .collect();
    let (loop_report, workers, profile) = decode_loop_on_mesh(
        state,
        &snap,
        &model,
        &snap.nodes,
        &placement,
        strategy,
        &LoopShape {
            prompt: prompt.to_string(),
            rows,
            steps,
            pipelined,
            speculation: speculation.clone(),
            session_ids,
        },
    )
    .await?;
    record_run(
        state,
        if pipelined {
            "decode-loop"
        } else {
            "decode-loop-sequential"
        },
        &loop_report.report.graph_id,
        Some(strategy),
        &loop_report.report,
        Some(modelled_ms_per_token),
        snap.epoch,
        &workers,
    );
    let routing_recorded = state
        .lock()
        .expect("gateway state lock")
        .record_grid_access(&model.id, &loop_report.report.accesses);
    if !loop_report.success {
        return Err(format!(
            "decode loop validated {} of {} units",
            loop_report.report.units_validated, loop_report.report.units_total
        ));
    }
    let sessions = record_sessions(state, &model, &placement, &loop_report);
    Ok(GridDecodeLoopOutcome {
        strategy,
        choice,
        placement,
        placement_cached,
        modelled_ms_per_token,
        routing_recorded,
        loop_report,
        profile,
        resumed,
        sessions,
    })
}

/// The decode step every inference endpoint runs: the grid dispatcher, or the legacy stage loop
/// when a grid step cannot be built. The report keeps the shape the Control Room reads: modelled
/// stages and speculative figures from the reference executor, with the step's measured wall time
/// and the cost model's modelled tokens per second.
pub async fn execute_step_or_legacy(
    state: &Arc<Mutex<GatewayState>>,
    plan: &ClusterShardPlan,
    prompt: &str,
    live_workers: &BTreeMap<String, LiveWorkerRegistration>,
) -> (DistributedPipelineExecutionReport, Vec<LiveHopTelemetry>) {
    match run_grid_decode_step(state, prompt, 1, None).await {
        Ok(outcome) => {
            let mut report = DistributedMoEPipelineExecutor::execute_forward_pass(plan, prompt);
            report.total_pipeline_latency_ms = outcome.report.wall_ms;
            report.effective_tokens_per_sec = match outcome.strategy {
                ExecutionStrategy::PipelineChain => outcome.choice.chain.tokens_per_sec,
                ExecutionStrategy::ExpertParallel => outcome.choice.expert_parallel.tokens_per_sec,
            };
            (report, outcome.hops)
        }
        Err(_) => DistributedLanExecutor::execute_live_pipeline(plan, prompt, live_workers).await,
    }
}

// ================================================================================================
// Endpoints
// ================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridInventoryResponse {
    pub inventory: PoolInventory,
    pub summary: String,
    pub live_workers: usize,
    pub simulated_nodes: usize,
    pub runs: u64,
    pub last_run: Option<GridLastRun>,
    /// Measured compute cost per worker and kind of work, kept across runs; every figure in it
    /// was measured on the kernels the workers run, which today are the synthetic references.
    pub telemetry: TelemetryStore,
}

pub async fn get_grid_inventory(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<GridInventoryResponse> {
    let (nodes, live_workers, runs, last_run, telemetry, offers) = {
        let guard = state.lock().expect("gateway state lock");
        (
            guard.nodes.clone(),
            guard.live_workers.clone(),
            guard.grid_runs,
            guard.grid_last_run.clone(),
            guard.grid_telemetry.clone(),
            effective_offers(&guard.nodes, &guard.grid_offers),
        )
    };
    let workers = mesh_workers(&nodes, &live_workers, &telemetry, &offers);
    let inventory = PoolInventory::from_workers(&workers);
    Json(GridInventoryResponse {
        summary: inventory.summary_line(),
        inventory,
        live_workers: live_workers.len(),
        simulated_nodes: nodes
            .iter()
            .filter(|n| !live_workers.contains_key(&n.id.0))
            .count(),
        runs,
        last_run,
        telemetry,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridPlanQuery {
    pub model: Option<String>,
    pub strategy: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPlacementSummary {
    pub strategy: ExecutionStrategy,
    pub trunk_ranges: Vec<TrunkRange>,
    pub experts: usize,
    pub hot_fraction_pct: f64,
    pub replicated_experts: usize,
    pub cold_gb: f64,
    pub rules: usize,
    pub fast_memory_used_gb: BTreeMap<String, f64>,
    pub summary: String,
    /// The load balancer's account of this placement's copies, when it ran: every layer's
    /// busiest node before and after, modelled.
    #[serde(default)]
    pub balance: Option<BalanceReport>,
}

impl GridPlacementSummary {
    pub fn from_placement(placement: &ExpertPlacementPlan) -> Self {
        Self {
            strategy: placement.strategy,
            trunk_ranges: placement.trunk_ranges.clone(),
            experts: placement.experts.len(),
            hot_fraction_pct: (placement.hot_fraction() * 1000.0).round() / 10.0,
            replicated_experts: placement.replicated_experts,
            cold_gb: placement.cold_gb,
            rules: placement.rules.len(),
            fast_memory_used_gb: placement
                .fast_memory_used_gb
                .iter()
                .map(|(n, gb)| (n.0.clone(), *gb))
                .collect(),
            summary: placement.summary(),
            balance: placement.balance.clone(),
        }
    }
}

/// The initial staging a placement needs, as the owners reviewing it see it: what can start now
/// and what waits for an owner's approval.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GridStagingPreview {
    /// The vault the preview stages from.
    pub source: String,
    pub ready_jobs: usize,
    pub ready_gb: f64,
    /// Transfers held until the owner of their destination approves this exact placement.
    pub awaiting_jobs: usize,
    pub awaiting_gb: f64,
    pub nodes_awaiting_approval: Vec<String>,
    /// Nodes whose offers require approval at all.
    pub approval_required: Vec<String>,
    /// Nodes whose owners approved this exact placement.
    pub approved_nodes: Vec<String>,
    pub ready_to_stage: bool,
}

fn staging_preview(
    placement: &ExpertPlacementPlan,
    nodes: &[HardwareNode],
    offers: &BTreeMap<NodeId, ResourceOffer>,
    approvals: &BTreeMap<String, PlanApproval>,
    now: u64,
) -> GridStagingPreview {
    let digest = placement.digest();
    let Some(source) = nodes
        .iter()
        .find(|n| n.roles.contains(&NodeRole::StorageVault))
        .or_else(|| nodes.first())
        .map(|n| n.id.clone())
    else {
        return GridStagingPreview::default();
    };
    let mut catalog = ReplicaCatalog::new();
    for rule in &placement.rules {
        catalog.record(Replica::new(
            rule.content.clone(),
            source.clone(),
            ResidencyTier::NasVault,
            now,
        ));
    }
    let approval_required: Vec<NodeId> = offers
        .iter()
        .filter(|(_, o)| o.requires_approval)
        .map(|(n, _)| n.clone())
        .collect();
    let approved: Vec<NodeId> = approval_required
        .iter()
        .filter(|n| {
            approvals
                .get(&n.0)
                .is_some_and(|a| a.covers(&placement.model_id, digest))
        })
        .cloned()
        .collect();
    let needs: BTreeSet<NodeId> = approval_required
        .iter()
        .filter(|n| !approved.contains(n))
        .cloned()
        .collect();
    // The vault's copy is a replica too, so each rule asks for one copy per candidate on top.
    let rules: Vec<_> = placement
        .rules
        .iter()
        .map(|rule| {
            let mut rule = rule.clone();
            let extra = u32::from(!rule.candidate_nodes.contains(&source));
            rule.min_replicas = rule.candidate_nodes.len() as u32 + extra;
            rule
        })
        .collect();
    let plan = catalog.plan_staging_gated(&rules, &BTreeMap::new(), &needs);
    let gib = |bytes: u64| (bytes as f64 / 1_073_741_824.0 * 100.0).round() / 100.0;
    GridStagingPreview {
        source: source.0,
        ready_jobs: plan.ready.len(),
        ready_gb: gib(plan.bytes_ready()),
        awaiting_jobs: plan.awaiting_approval.len(),
        awaiting_gb: gib(plan.bytes_awaiting()),
        nodes_awaiting_approval: plan.nodes_awaiting().into_iter().map(|n| n.0).collect(),
        approval_required: approval_required.into_iter().map(|n| n.0).collect(),
        approved_nodes: approved.into_iter().map(|n| n.0).collect(),
        ready_to_stage: plan.is_clear(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPlanResponse {
    pub model_id: String,
    pub choice: StrategyChoice,
    pub placement: GridPlacementSummary,
    /// Identity of the placement; approvals and calibration records bind to it.
    pub placement_digest: u64,
    /// How the placement fits the offers in force, and what is missing where.
    pub fit: PlacementFit,
    /// The offer in force on every node.
    pub offers: BTreeMap<String, ResourceOffer>,
    pub staging: GridStagingPreview,
    /// What the planner learned from live routing for this model; `learned` is false until any
    /// step has run, and then the placement's expert order and replicas come from it.
    pub usage: UsageSummary,
    /// Whether this is the placement already in force rather than a fresh plan.
    pub placement_cached: bool,
    pub repins_applied: usize,
    pub swaps_applied: usize,
    /// Always true: the cost model's figures are projections until a run is measured.
    pub modelled: bool,
}

fn plan_response(
    state: &Arc<Mutex<GatewayState>>,
    snap: &MeshSnapshot,
    model: ModelSpec,
    forced: Option<ExecutionStrategy>,
) -> GridPlanResponse {
    let (choice, placement, placement_cached) = current_placement(state, snap, &model, forced);
    let (usage, repins_applied, swaps_applied) = {
        let mut guard = state.lock().expect("gateway state lock");
        let usage = guard.usage_for(&model.id).summary();
        let cached = guard.grid_placements.get(&model.id);
        (
            usage,
            cached.map_or(0, |c| c.repins_applied),
            cached.map_or(0, |c| c.swaps_applied),
        )
    };
    let staging = staging_preview(
        &placement,
        &snap.nodes,
        &snap.offers,
        &snap.approvals,
        now_unix_ms(),
    );
    GridPlanResponse {
        model_id: model.id.clone(),
        choice,
        placement_digest: placement.digest(),
        fit: placement.fit.clone(),
        offers: snap
            .offers
            .iter()
            .map(|(n, o)| (n.0.clone(), o.clone()))
            .collect(),
        staging,
        placement: GridPlacementSummary::from_placement(&placement),
        usage,
        placement_cached,
        repins_applied,
        swaps_applied,
        modelled: true,
    }
}

fn resolve_model(snap: &MeshSnapshot, requested: Option<&str>) -> ModelSpec {
    requested
        .filter(|m| !m.trim().is_empty())
        .map(ModelSpec::from_id)
        .unwrap_or_else(|| snap.inference_model.clone())
}

pub async fn get_grid_plan(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Query(query): Query<GridPlanQuery>,
) -> Json<GridPlanResponse> {
    let snap = snapshot(&state);
    let model = resolve_model(&snap, query.model.as_deref());
    let forced = parse_strategy(query.strategy.as_deref());
    Json(plan_response(&state, &snap, model, forced))
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridRepinRequest {
    pub model: Option<String>,
    /// Swaps at most for this boundary; default 4.
    pub max_swaps: Option<usize>,
    /// Apply the swaps to the placement in force (default) or only propose them.
    pub apply: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridRepinResponse {
    pub model_id: String,
    pub swaps: Vec<RepinSwap>,
    /// Copies the load balancer added or dropped at this boundary (its incremental mode),
    /// counted against the same cap as the swaps.
    #[serde(default)]
    pub replica_changes: Vec<ReplicaChange>,
    /// The balancer's account of the boundary: every layer's busiest node before and after,
    /// modelled; absent for the chain shape, which has no copies to balance.
    #[serde(default)]
    pub balance: Option<BalanceReport>,
    pub rationale: String,
    pub applied: bool,
    pub staging_jobs: Vec<StagingJob>,
    pub placement_digest: u64,
    pub repins_applied: usize,
    pub swaps_applied: usize,
    #[serde(default)]
    pub replica_changes_applied: usize,
    /// Expert reads on the step's critical path at one row, modelled, before and after.
    #[serde(default)]
    pub expert_read_ms_before: f64,
    #[serde(default)]
    pub expert_read_ms_after: f64,
    /// Always true for the figures above.
    #[serde(default)]
    pub modelled: bool,
    pub error: Option<String>,
}

/// One live re-pin boundary (colibri's balanced policy, with EPLB's balancer in its
/// incremental mode): at most a few changes, each a swap between a hot streamed expert and a
/// cold pinned one or a copy of a hot expert onto a node with room, when the learned heat
/// clears the hysteresis.
pub async fn post_grid_repin(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridRepinRequest>>,
) -> Json<GridRepinResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let snap = snapshot(&state);
    let model = resolve_model(&snap, req.model.as_deref());
    // Make sure a placement is in force for the model before adapting it.
    let _ = current_placement(&state, &snap, &model, None);
    let policy = RepinPolicy {
        max_swaps: req.max_swaps.unwrap_or(4).clamp(1, 64),
        ..RepinPolicy::default()
    };
    match repin_now(&state, &model, &policy, req.apply.unwrap_or(true)) {
        Ok(outcome) => Json(GridRepinResponse {
            model_id: outcome.model_id,
            swaps: outcome.proposal.swaps,
            replica_changes: outcome.proposal.replica_changes,
            balance: outcome.proposal.balance,
            rationale: outcome.proposal.rationale,
            applied: outcome.applied,
            staging_jobs: outcome.staging_jobs,
            placement_digest: outcome.placement_digest,
            repins_applied: outcome.repins_applied,
            swaps_applied: outcome.swaps_applied,
            replica_changes_applied: outcome.replica_changes_applied,
            expert_read_ms_before: outcome.expert_read_ms_before,
            expert_read_ms_after: outcome.expert_read_ms_after,
            modelled: true,
            error: None,
        }),
        Err(error) => Json(GridRepinResponse {
            model_id: model.id,
            swaps: Vec::new(),
            replica_changes: Vec::new(),
            balance: None,
            rationale: String::new(),
            applied: false,
            staging_jobs: Vec::new(),
            placement_digest: 0,
            repins_applied: 0,
            swaps_applied: 0,
            replica_changes_applied: 0,
            expert_read_ms_before: 0.0,
            expert_read_ms_after: 0.0,
            modelled: true,
            error: Some(error),
        }),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GridCalibrateRequest {
    /// Shape to measure: `chain` or `expert-parallel`; the planner's choice when absent.
    pub strategy: Option<String>,
    /// Decode steps to run, at most 8.
    pub steps: Option<u32>,
    pub rows: Option<u32>,
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCalibrateResponse {
    pub model_id: String,
    pub strategy: Option<ExecutionStrategy>,
    pub record: Option<CalibrationRecord>,
    pub steps_run: u32,
    pub wall_ms: f64,
    /// For the fan-out shape, whether its bytes matched a chain step on the same prompt.
    pub digest_matches_chain: Option<bool>,
    /// Whether the record binds to the placement now in force for the shape it measured.
    pub applies_to_plan_in_force: bool,
    /// What now decides the shape for this model: `measured`, `modelled`, `dense` or `forced`.
    pub decided_by: String,
    pub rationale: String,
    pub error: Option<String>,
}

/// Measures the placement in force: at most eight decode steps of one shape within a 20 second
/// budget, recorded as milliseconds per token bound to the placement's digest, the node set and
/// the versions that ran. A fan-out measurement also runs one chain step on the same prompt and
/// keeps whether the bytes matched; a record that fails that gate is kept but never steers the
/// choice. Once both shapes are measured on the same mesh, the measurements decide the shape.
pub async fn post_grid_calibrate(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridCalibrateRequest>>,
) -> Json<GridCalibrateResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let steps = req
        .steps
        .unwrap_or(CALIBRATION_MAX_STEPS)
        .clamp(1, CALIBRATION_MAX_STEPS);
    let rows = req.rows.unwrap_or(1).clamp(1, 64);
    let prompt = req
        .prompt
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| "calibration: the same prompt on every step".to_string());
    let forced = parse_strategy(req.strategy.as_deref());
    let model_id = snapshot(&state).inference_model.id.clone();
    let failure = |error: String| GridCalibrateResponse {
        model_id: model_id.clone(),
        strategy: forced,
        record: None,
        steps_run: 0,
        wall_ms: 0.0,
        digest_matches_chain: None,
        applies_to_plan_in_force: false,
        decided_by: String::new(),
        rationale: String::new(),
        error: Some(error),
    };

    let t0 = Instant::now();
    let mut walls = Vec::new();
    let mut exec_by_node: BTreeMap<String, f64> = BTreeMap::new();
    let mut last: Option<GridStepOutcome> = None;
    for _ in 0..steps {
        if t0.elapsed().as_millis() as u64 > CALIBRATION_MAX_WALL_MS {
            break;
        }
        match run_grid_decode_step(&state, &prompt, rows, forced).await {
            Ok(outcome) => {
                walls.push(outcome.report.wall_ms);
                for trace in outcome.report.traces.iter().filter(|t| t.ok) {
                    *exec_by_node.entry(trace.node_id.0.clone()).or_default() += trace.exec_ms;
                }
                last = Some(outcome);
            }
            Err(e) => return Json(failure(format!("calibration step failed: {e}"))),
        }
    }
    let Some(outcome) = last else {
        return Json(failure(
            "no calibration step ran within the time budget".to_string(),
        ));
    };
    let strategy = outcome.strategy;
    let digest_matches_chain = match strategy {
        ExecutionStrategy::ExpertParallel => {
            match run_grid_decode_step(
                &state,
                &prompt,
                rows,
                Some(ExecutionStrategy::PipelineChain),
            )
            .await
            {
                Ok(chain) => Some(chain.final_output_fnv1a == outcome.final_output_fnv1a),
                Err(_) => Some(false),
            }
        }
        ExecutionStrategy::PipelineChain => None,
    };
    let steps_run = walls.len() as u32;
    for v in exec_by_node.values_mut() {
        *v /= steps_run.max(1) as f64;
    }
    let snap = snapshot(&state);
    let binding = calibration_binding(&snap, &model_id, strategy, outcome.placement.digest(), rows);
    let record = CalibrationRecord::from_steps(
        binding,
        &walls,
        exec_by_node,
        outcome.final_output_fnv1a,
        digest_matches_chain,
        now_unix_ms(),
    );
    {
        let mut guard = state.lock().expect("gateway state lock");
        guard.grid_calibrations.record(record.clone());
        let _ = guard.save_grid_calibrations();
    }
    let snap = snapshot(&state);
    let (choice, _, _) = current_placement(&state, &snap, &snap.inference_model.clone(), None);
    let in_force = state
        .lock()
        .expect("gateway state lock")
        .grid_placements
        .get(&model_id)
        .map(|c| c.plan_for(strategy).digest());
    Json(GridCalibrateResponse {
        model_id,
        strategy: Some(strategy),
        applies_to_plan_in_force: in_force == Some(record.binding.placement_digest),
        record: Some(record),
        steps_run,
        wall_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
        digest_matches_chain,
        decided_by: choice.decided_by,
        rationale: choice.rationale,
        error: None,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridApproveRequest {
    pub model: Option<String>,
    /// The node whose owner approves (or, with `approve: false`, withdraws approval of) the plan.
    pub node: String,
    pub approve: Option<bool>,
}

/// An owner's approval of the current placement plan for their node. The approval binds to the
/// plan's digest, so a plan that changes afterwards asks again.
pub async fn post_grid_plan_approve(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<GridApproveRequest>,
) -> Json<GridPlanResponse> {
    let snap = snapshot(&state);
    let model = resolve_model(&snap, req.model.as_deref());
    let (_, placement, _) = current_placement(&state, &snap, &model, None);
    let digest = placement.digest();
    {
        let mut guard = state.lock().expect("gateway state lock");
        if req.approve.unwrap_or(true) {
            guard.grid_plan_approvals.insert(
                req.node.clone(),
                PlanApproval {
                    node: NodeId(req.node.clone()),
                    model_id: model.id.clone(),
                    placement_digest: digest,
                    approved_unix_ms: now_unix_ms(),
                },
            );
        } else {
            guard.grid_plan_approvals.remove(&req.node);
        }
    }
    let snap = snapshot(&state);
    Json(plan_response(&state, &snap, model, None))
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridRunRequest {
    pub prompt: Option<String>,
    pub rows: Option<u32>,
    pub strategy: Option<String>,
    /// Apply one re-pin boundary to the placement in force after the step.
    pub repin: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridRunApiResponse {
    pub strategy: Option<ExecutionStrategy>,
    pub graph_id: String,
    pub units_total: usize,
    pub units_validated: usize,
    pub success: bool,
    pub wall_ms: f64,
    pub modelled_ms_per_token: f64,
    pub final_digest_fnv1a: u64,
    pub hops: Vec<LiveHopTelemetry>,
    pub credits_granted: f64,
    pub receipts: usize,
    pub rationale: String,
    pub placement_cached: bool,
    pub routing_recorded: usize,
    pub repin_swaps: usize,
    /// Verde disputes the step settled: the first divergent layer, the referee's work and who
    /// was penalised, one per disputed unit.
    #[serde(default)]
    pub disputes: Vec<VerdeDisputeReport>,
    pub error: Option<String>,
}

pub async fn post_grid_run(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridRunRequest>>,
) -> Json<GridRunApiResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let prompt = req
        .prompt
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| "Run one decode step as a work graph across the mesh.".to_string());
    let forced = parse_strategy(req.strategy.as_deref());
    let repin = req.repin.unwrap_or(false).then(RepinPolicy::default);
    match run_grid_decode_step_with(&state, &prompt, req.rows.unwrap_or(1), forced, repin).await {
        Ok(outcome) => Json(GridRunApiResponse {
            strategy: Some(outcome.strategy),
            graph_id: outcome.graph_id,
            units_total: outcome.report.units_total,
            units_validated: outcome.report.units_validated,
            success: outcome.report.success,
            wall_ms: outcome.report.wall_ms,
            modelled_ms_per_token: outcome.modelled_ms_per_token,
            final_digest_fnv1a: outcome.final_digest_fnv1a,
            hops: outcome.hops,
            credits_granted: outcome.report.credits.iter().map(|c| c.credits).sum(),
            receipts: outcome.report.receipts.len(),
            rationale: outcome.choice.rationale,
            placement_cached: outcome.placement_cached,
            routing_recorded: outcome.routing_recorded,
            repin_swaps: outcome
                .repin
                .as_ref()
                .map_or(0, |r| r.proposal.change_count()),
            disputes: outcome.report.disputes.clone(),
            error: None,
        }),
        Err(error) => {
            let last = state
                .lock()
                .expect("gateway state lock")
                .grid_last_run
                .clone();
            Json(GridRunApiResponse {
                strategy: last.as_ref().and_then(|l| l.strategy),
                graph_id: last
                    .as_ref()
                    .map(|l| l.graph_id.clone())
                    .unwrap_or_default(),
                units_total: last.as_ref().map_or(0, |l| l.units_total),
                units_validated: last.as_ref().map_or(0, |l| l.units_validated),
                success: false,
                wall_ms: last.as_ref().map_or(0.0, |l| l.wall_ms),
                modelled_ms_per_token: 0.0,
                final_digest_fnv1a: 0,
                hops: Vec::new(),
                credits_granted: 0.0,
                receipts: 0,
                rationale: String::new(),
                placement_cached: false,
                routing_recorded: 0,
                repin_swaps: 0,
                disputes: Vec::new(),
                error: Some(error),
            })
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridDecodeRequest {
    pub prompt: Option<String>,
    pub rows: Option<u32>,
    /// Decode steps per session, at most 64.
    pub steps: Option<u32>,
    /// Sessions decoded together, at most 8.
    pub sessions: Option<u32>,
    pub strategy: Option<String>,
    /// Plan and queue each session's next step while its current one runs, and interleave
    /// the sessions in one queue (default); `false` runs one graph at a time.
    pub pipelined: Option<bool>,
    /// Draft and verify (ARCHITECTURE 9.11): every session decodes one row with a modelled
    /// drafter and the verification scheduler sizes its verify batch per step.
    pub speculative: Option<bool>,
    /// Draft positions per step at most; DeepInfer's default is 5.
    pub max_draft: Option<u32>,
    /// The modelled drafter: `code` (often right), `chat` (less so) or `none` (never).
    pub draft_profile: Option<String>,
    /// DSpark's causal early stop (default) or DeepInfer's search of every pair.
    pub early_stop: Option<bool>,
    /// Which step's confidences size a step's verification: 1 (default), 2 (DeepInfer) or 0.
    pub confidence_lag: Option<u32>,
    /// Continue this session (one session; its KV pages replay first when they have sunk).
    pub session: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridDecodeSessionSummary {
    pub session_id: String,
    pub rows: u32,
    pub steps: usize,
    pub tokens: u64,
    pub wall_ms: f64,
    pub mean_step_ms: f64,
    pub inter_step_gap_ms: Vec<f64>,
    pub mean_gap_ms: f64,
    pub max_gap_ms: f64,
    pub final_output_fnv1a: u64,
    pub final_digest_fnv1a: u64,
    pub success: bool,
    /// Present for a speculative session: proposed and accepted positions, both accountings.
    #[serde(default)]
    pub accounting: Option<AcceptanceAccounting>,
}

/// How a speculative loop's verification was sized and how its tokens are counted, disclosed
/// the way DeepInfer's benchmark README discloses it: full acceptance beside the scaled figure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridVerificationSummary {
    pub draft_profile: String,
    pub max_draft: u32,
    pub early_stop: bool,
    pub confidence_lag: u32,
    /// `measured` from calibration records across batch widths, or `modelled`.
    pub profile_source: String,
    pub profile_widths: Vec<u32>,
    pub profile: String,
    /// Over the steps: rows verified per step, expected accepted tokens, and the scheduler's
    /// expected gain over verifying nothing.
    pub mean_batch_tokens: f64,
    pub mean_expected_accepted: f64,
    pub mean_expected_gain: f64,
    pub proposed: u64,
    pub accepted: u64,
    /// `accepted / proposed`; the acceptance is modelled, not a drafter's.
    pub acceptance_rate: f64,
    pub tokens_full_acceptance: u64,
    pub tokens_accepted: u64,
    /// DeepInfer's plotter factor, `1 + nextn x acceptance`.
    pub tau: f64,
    /// Tokens per second over the loop's wall under each accounting.
    pub measured_tokens_per_sec_full_acceptance: f64,
    pub measured_tokens_per_sec_accepted: f64,
    pub modelled_acceptance: bool,
    pub accounting_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridDecodeApiResponse {
    pub strategy: Option<ExecutionStrategy>,
    pub pipelined: bool,
    pub sessions: Vec<GridDecodeSessionSummary>,
    pub steps_total: u64,
    pub tokens_total: u64,
    /// First dispatch to last return across every session, measured on the coordinator.
    pub wall_ms: f64,
    /// `tokens_total` over the wall: the mesh's throughput on the kernels the workers run.
    pub measured_tokens_per_sec: f64,
    /// One row's tokens over its session's wall, averaged over sessions: one user's speed, a
    /// user being one row of a session's batch (DeepInfer's per-user speed).
    pub measured_per_user_tokens_per_sec: f64,
    /// The next step's first dispatch minus this step's last return, over every boundary.
    pub mean_inter_step_gap_ms: f64,
    pub max_inter_step_gap_ms: f64,
    /// Whether units of two sessions were in flight at the same time.
    pub interleaved: bool,
    pub modelled_ms_per_token: f64,
    /// The kernels the measured figures ran on: `reference-synthetic` today.
    pub kernels: String,
    pub units_total: usize,
    pub units_validated: usize,
    pub credits_granted: f64,
    pub routing_recorded: usize,
    pub placement_cached: bool,
    pub disputes: Vec<VerdeDisputeReport>,
    /// Present when the sessions drafted.
    #[serde(default)]
    pub verification: Option<GridVerificationSummary>,
    /// Present when a named session replayed its suffix before decoding.
    #[serde(default)]
    pub resumed: Option<GridReplaySummary>,
    /// The sessions as the registry holds them after the loop: tokens and KV tier.
    #[serde(default)]
    pub session_states: Vec<KvSessionState>,
    pub error: Option<String>,
}

fn decode_failure(error: String, pipelined: bool) -> GridDecodeApiResponse {
    GridDecodeApiResponse {
        strategy: None,
        pipelined,
        sessions: Vec::new(),
        steps_total: 0,
        tokens_total: 0,
        wall_ms: 0.0,
        measured_tokens_per_sec: 0.0,
        measured_per_user_tokens_per_sec: 0.0,
        mean_inter_step_gap_ms: 0.0,
        max_inter_step_gap_ms: 0.0,
        interleaved: false,
        modelled_ms_per_token: 0.0,
        kernels: kadmium_sharding::CALIBRATION_KERNELS_REFERENCE.to_string(),
        units_total: 0,
        units_validated: 0,
        credits_granted: 0.0,
        routing_recorded: 0,
        placement_cached: false,
        disputes: Vec::new(),
        verification: None,
        resumed: None,
        session_states: Vec::new(),
        error: Some(error),
    }
}

/// One session as the sessions endpoint lists it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSessionView {
    pub state: KvSessionState,
    pub pages: usize,
    /// Modelled bytes of the session's pages, by tier.
    pub bytes_by_tier: BTreeMap<String, u64>,
    pub needs_replay: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSessionsResponse {
    pub sessions: Vec<GridSessionView>,
    pub policy: KvTierPolicy,
    /// Modelled page bytes across every session, by tier.
    pub bytes_by_tier: BTreeMap<String, u64>,
    /// Always true: every byte figure comes from the model spec's reference KV size.
    pub modelled: bool,
}

fn session_views(guard: &GatewayState) -> (Vec<GridSessionView>, BTreeMap<String, u64>) {
    let mut total: BTreeMap<String, u64> = BTreeMap::new();
    let views = guard
        .grid_sessions
        .values()
        .map(|s| {
            let mut by_tier: BTreeMap<String, u64> = BTreeMap::new();
            let mut pages = 0usize;
            for replica in guard.grid_kv_catalog.resident_on_any() {
                if let Some((model, session, _)) = KvPageNames::parse(&replica.content.content_id)
                    && model == s.model_id
                    && session == s.session_id
                {
                    pages += 1;
                    *by_tier.entry(format!("{:?}", replica.tier)).or_default() +=
                        replica.content.bytes;
                    *total.entry(format!("{:?}", replica.tier)).or_default() +=
                        replica.content.bytes;
                }
            }
            GridSessionView {
                needs_replay: s.needs_replay(),
                state: s.clone(),
                pages,
                bytes_by_tier: by_tier,
            }
        })
        .collect();
    (views, total)
}

/// The decode sessions the grid has run and where their KV pages sit (ARCHITECTURE 9.12).
pub async fn get_grid_sessions(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<GridSessionsResponse> {
    let guard = state.lock().expect("gateway state lock");
    let (sessions, bytes_by_tier) = session_views(&guard);
    Json(GridSessionsResponse {
        sessions,
        policy: guard.grid_kv_policy.clone(),
        bytes_by_tier,
        modelled: true,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridSessionsTierRequest {
    /// The moment the pass judges idleness at; now when absent. A later time lets a caller
    /// see what the policy would do after that much idleness.
    pub at_unix_ms: Option<u64>,
    pub idle_to_nvme_ms: Option<u64>,
    pub idle_to_vault_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSessionsTierResponse {
    pub at_unix_ms: u64,
    pub plan: KvTierPlan,
    pub sessions: Vec<GridSessionView>,
    pub policy: KvTierPolicy,
}

/// One tiering pass: every session idle past the policy's thresholds has its pages sink to
/// the owner's NVMe and then to the vault; a session active again comes back up.
pub async fn post_grid_sessions_tier(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridSessionsTierRequest>>,
) -> Json<GridSessionsTierResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let snap = snapshot(&state);
    let model = snap.inference_model.clone();
    let (_, placement, _) = current_placement(&state, &snap, &model, None);
    let at = req.at_unix_ms.unwrap_or_else(now_unix_ms);
    let mut guard = state.lock().expect("gateway state lock");
    if let Some(ms) = req.idle_to_nvme_ms {
        guard.grid_kv_policy.idle_to_nvme_ms = ms;
    }
    if let Some(ms) = req.idle_to_vault_ms {
        guard.grid_kv_policy.idle_to_vault_ms = ms.max(guard.grid_kv_policy.idle_to_nvme_ms);
    }
    let policy = guard.grid_kv_policy.clone();
    let nodes = guard.nodes.clone();
    let mut catalog = std::mem::take(&mut guard.grid_kv_catalog);
    let mut sessions: Vec<KvSessionState> = guard.grid_sessions.values().cloned().collect();
    let plan = KvTierPlanner::tier(
        &mut catalog,
        &mut sessions,
        &model,
        &placement,
        &nodes,
        &policy,
        at,
    );
    for s in sessions {
        guard.grid_sessions.insert(s.session_id.clone(), s);
    }
    guard.grid_kv_catalog = catalog;
    let (views, _) = session_views(&guard);
    Json(GridSessionsTierResponse {
        at_unix_ms: at,
        plan,
        sessions: views,
        policy,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridSessionsReplayRequest {
    pub session: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSessionsReplayResponse {
    pub replay: Option<GridReplaySummary>,
    pub session: Option<KvSessionState>,
    pub error: Option<String>,
}

/// Replays a session's bounded suffix on the trunk owners and brings its pages back to fast
/// memory, whatever tier they sat in.
pub async fn post_grid_sessions_replay(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<GridSessionsReplayRequest>,
) -> Json<GridSessionsReplayResponse> {
    match run_kv_replay(&state, &req.session).await {
        Ok(replay) => {
            let session = state
                .lock()
                .expect("gateway state lock")
                .grid_sessions
                .get(&req.session)
                .cloned();
            Json(GridSessionsReplayResponse {
                replay: Some(replay),
                session,
                error: None,
            })
        }
        Err(error) => Json(GridSessionsReplayResponse {
            replay: None,
            session: None,
            error: Some(error),
        }),
    }
}

/// What the request asks the drafter and the scheduler to do, when it asks at all.
fn decode_speculation(req: &GridDecodeRequest) -> Result<Option<DecodeSpeculation>, String> {
    if !req.speculative.unwrap_or(false) {
        return Ok(None);
    }
    let name = req
        .draft_profile
        .clone()
        .unwrap_or_else(|| "code".to_string());
    let draft_profile = DraftProfile::named(&name)
        .ok_or_else(|| format!("unknown draft profile `{name}`: use code, chat or none"))?;
    Ok(Some(DecodeSpeculation {
        draft_profile,
        policy: VerificationPolicy {
            max_draft: req
                .max_draft
                .unwrap_or(kadmium_sharding::DEFAULT_MAX_DRAFT)
                .clamp(1, 16),
            early_stop: req.early_stop.unwrap_or(true),
            confidence_lag: req.confidence_lag.unwrap_or(1).min(4),
            max_batch_tokens: None,
        },
    }))
}

/// Several decode steps of one or several sessions through one runner (ARCHITECTURE 9.10),
/// drafting and verifying when asked (9.11).
pub async fn post_grid_decode(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridDecodeRequest>>,
) -> Json<GridDecodeApiResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let prompt = req
        .prompt
        .clone()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| {
            "Decode several steps as chained work graphs across the mesh.".to_string()
        });
    let forced = parse_strategy(req.strategy.as_deref());
    let pipelined = req.pipelined.unwrap_or(true);
    let speculation = match decode_speculation(&req) {
        Ok(s) => s,
        Err(e) => return Json(decode_failure(e, pipelined)),
    };
    let draft_profile_name = req
        .draft_profile
        .clone()
        .unwrap_or_else(|| "code".to_string());
    match run_grid_decode_loop(
        &state,
        &prompt,
        req.rows.unwrap_or(1),
        req.steps.unwrap_or(4),
        req.sessions.unwrap_or(1),
        forced,
        pipelined,
        speculation.clone(),
        req.session.clone().filter(|s| !s.trim().is_empty()),
    )
    .await
    {
        Ok(outcome) => {
            let round = |v: f64| (v * 100.0).round() / 100.0;
            let lp = &outcome.loop_report;
            let sessions: Vec<GridDecodeSessionSummary> = lp
                .sessions
                .iter()
                .map(|s| GridDecodeSessionSummary {
                    session_id: s.session_id.clone(),
                    rows: s.rows,
                    steps: s.steps.len(),
                    tokens: s.tokens,
                    wall_ms: s.wall_ms,
                    mean_step_ms: round(if s.steps.is_empty() {
                        0.0
                    } else {
                        s.steps.iter().map(|st| st.wall_ms).sum::<f64>() / s.steps.len() as f64
                    }),
                    inter_step_gap_ms: s.inter_step_gap_ms.clone(),
                    mean_gap_ms: s.mean_gap_ms,
                    max_gap_ms: s.max_gap_ms,
                    final_output_fnv1a: s.steps.last().map_or(0, |st| st.output_fnv1a),
                    final_digest_fnv1a: s.steps.last().map_or(0, |st| st.final_digest_fnv1a),
                    success: s.success,
                    accounting: s.accounting.clone(),
                })
                .collect();
            let verification = match (&speculation, &lp.accounting, &outcome.profile) {
                (Some(spec), Some(acc), Some(profile)) => {
                    let n = lp.schedules.len().max(1) as f64;
                    let per_sec = |tokens: u64| {
                        round(if lp.wall_ms > 0.0 {
                            tokens as f64 / lp.wall_ms * 1000.0
                        } else {
                            0.0
                        })
                    };
                    Some(GridVerificationSummary {
                        draft_profile: draft_profile_name.to_ascii_lowercase(),
                        max_draft: spec.policy.max_draft,
                        early_stop: spec.policy.early_stop,
                        confidence_lag: spec.policy.confidence_lag,
                        profile_source: profile.source.clone(),
                        profile_widths: profile.widths(),
                        profile: profile.summary(),
                        mean_batch_tokens: round(
                            lp.schedules.iter().map(|s| s.batch_tokens as f64).sum::<f64>() / n,
                        ),
                        mean_expected_accepted: round(
                            lp.schedules.iter().map(|s| s.expected_accepted).sum::<f64>() / n,
                        ),
                        mean_expected_gain: round(
                            lp.schedules.iter().map(|s| s.gain()).sum::<f64>() / n,
                        ),
                        proposed: acc.proposed,
                        accepted: acc.accepted,
                        acceptance_rate: (acc.acceptance_rate * 10_000.0).round() / 10_000.0,
                        tokens_full_acceptance: acc.tokens_full_acceptance,
                        tokens_accepted: acc.tokens_accepted,
                        tau: round(acc.tau()),
                        measured_tokens_per_sec_full_acceptance: per_sec(acc.tokens_full_acceptance),
                        measured_tokens_per_sec_accepted: per_sec(acc.tokens_accepted),
                        modelled_acceptance: true,
                        accounting_note: "tokens_full_acceptance counts one golden token plus every draft position per step whatever was accepted, as DeepInfer's benchmark does; tokens_accepted counts the golden token plus the accepted prefix. The acceptance is a modelled coin, not a drafter's: no draft model exists here.".to_string(),
                    })
                }
                _ => None,
            };
            let per_user = if sessions.is_empty() {
                0.0
            } else {
                sessions
                    .iter()
                    .map(|s| {
                        if s.wall_ms > 0.0 {
                            s.tokens as f64 / s.rows.max(1) as f64 / s.wall_ms * 1000.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / sessions.len() as f64
            };
            Json(GridDecodeApiResponse {
                strategy: Some(outcome.strategy),
                pipelined: lp.pipelined,
                steps_total: lp.steps,
                tokens_total: lp.tokens,
                wall_ms: lp.wall_ms,
                measured_tokens_per_sec: round(if lp.wall_ms > 0.0 {
                    lp.tokens as f64 / lp.wall_ms * 1000.0
                } else {
                    0.0
                }),
                measured_per_user_tokens_per_sec: round(per_user),
                mean_inter_step_gap_ms: lp.mean_gap_ms,
                max_inter_step_gap_ms: lp.max_gap_ms,
                interleaved: lp.interleaved,
                modelled_ms_per_token: outcome.modelled_ms_per_token,
                kernels: kadmium_sharding::CALIBRATION_KERNELS_REFERENCE.to_string(),
                units_total: lp.report.units_total,
                units_validated: lp.report.units_validated,
                credits_granted: lp.report.credits.iter().map(|c| c.credits).sum(),
                routing_recorded: outcome.routing_recorded,
                placement_cached: outcome.placement_cached,
                disputes: lp.report.disputes.clone(),
                sessions,
                verification,
                resumed: outcome.resumed,
                session_states: outcome.sessions,
                error: None,
            })
        }
        Err(error) => Json(decode_failure(error, pipelined)),
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridSweepRequest {
    /// Batch widths to sweep; `[1, 2, 4, 8]` when absent, at most 8 widths of at most 64 rows.
    pub rows: Option<Vec<u32>>,
    /// Mesh sizes to sweep, each the first `n` nodes of the mesh; the whole mesh when absent.
    pub node_counts: Option<Vec<usize>>,
    /// Decode steps per point, at most 16.
    pub steps: Option<u32>,
    pub sessions: Option<u32>,
    pub strategy: Option<String>,
    pub pipelined: Option<bool>,
    pub speculative: Option<bool>,
    pub max_draft: Option<u32>,
    pub draft_profile: Option<String>,
    pub prompt: Option<String>,
    /// One tag is one ledger file; points of different configurations in it form different
    /// curves.
    pub tag: Option<String>,
    /// Append the table to `target/kadmium-state/sweeps/<tag>.md` (default true).
    pub write_ledger: Option<bool>,
    /// Keep a calibration record per plain single-session point (default true), so the
    /// verification scheduler's profile becomes measured at these widths.
    pub record_calibrations: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSweepResponse {
    pub tag: String,
    pub points: Vec<SweepPoint>,
    /// The curve each point belongs to, and whether it is on that curve's Pareto frontier.
    pub curve_ids: Vec<String>,
    pub pareto: Vec<bool>,
    pub curves: usize,
    /// The table in the ledger's form.
    pub markdown: String,
    pub ledger_path: Option<String>,
    pub calibration_records_added: usize,
    /// Points that could not run, with why.
    pub skipped: Vec<String>,
    pub kernels: String,
    pub error: Option<String>,
}

/// The sweep harness (ARCHITECTURE 9.13): a decode loop per batch width and node count, each
/// point a row of the Pareto table with its full configuration, points comparable only within
/// a curve. Plain single-session points also become calibration records of their placement at
/// their width.
pub async fn post_grid_sweep(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridSweepRequest>>,
) -> Json<GridSweepResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let snap = snapshot(&state);
    let model = snap.inference_model.clone();
    let tag = req
        .tag
        .clone()
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| format!("sweep-e{}", snap.epoch));
    let failure = |error: String| GridSweepResponse {
        tag: tag.clone(),
        points: Vec::new(),
        curve_ids: Vec::new(),
        pareto: Vec::new(),
        curves: 0,
        markdown: String::new(),
        ledger_path: None,
        calibration_records_added: 0,
        skipped: Vec::new(),
        kernels: kadmium_sharding::CALIBRATION_KERNELS_REFERENCE.to_string(),
        error: Some(error),
    };
    let mut rows: Vec<u32> = req
        .rows
        .clone()
        .unwrap_or_else(|| vec![1, 2, 4, 8])
        .into_iter()
        .map(|r| r.clamp(1, 64))
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows.truncate(8);
    let mut node_counts: Vec<usize> = req
        .node_counts
        .clone()
        .unwrap_or_else(|| vec![snap.nodes.len()])
        .into_iter()
        .map(|n| n.clamp(1, snap.nodes.len().max(1)))
        .collect();
    node_counts.sort_unstable();
    node_counts.dedup();
    node_counts.truncate(8);
    let steps = req.steps.unwrap_or(4).clamp(1, 16);
    let sessions = req.sessions.unwrap_or(1).clamp(1, 8);
    let pipelined = req.pipelined.unwrap_or(true);
    let speculation = match decode_speculation(&GridDecodeRequest {
        speculative: req.speculative,
        max_draft: req.max_draft,
        draft_profile: req.draft_profile.clone(),
        ..GridDecodeRequest::default()
    }) {
        Ok(s) => s,
        Err(e) => return Json(failure(e)),
    };
    let forced = parse_strategy(req.strategy.as_deref());
    let prompt = req
        .prompt
        .clone()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| "sweep: the same prompt at every point".to_string());
    let popularity = {
        let mut guard = state.lock().expect("gateway state lock");
        let ledger = guard.usage_for(&model.id);
        (!ledger.is_empty()).then(|| ledger.popularity())
    };
    let record_calibrations = req.record_calibrations.unwrap_or(true);
    let repository_state = format!(
        "kadmium {}",
        kadmium_network::DEFAULT_KADMIUM_RUNTIME_VERSION
    );

    let mut table = kadmium_sharding::SweepTable::new(&tag);
    let mut skipped = Vec::new();
    let mut records_added = 0usize;
    for &n in &node_counts {
        let nodes: Vec<HardwareNode> = snap.nodes[..n].to_vec();
        let offers: BTreeMap<NodeId, ResourceOffer> = snap
            .offers
            .iter()
            .filter(|(id, _)| nodes.iter().any(|nd| nd.id == **id))
            .map(|(id, o)| (id.clone(), o.clone()))
            .collect();
        let live: BTreeMap<String, LiveWorkerRegistration> = snap
            .live_workers
            .iter()
            .filter(|(id, _)| nodes.iter().any(|nd| nd.id.0 == **id))
            .map(|(id, r)| (id.clone(), r.clone()))
            .collect();
        let (mut choice, chain_plan, ep_plan) = cost_model(&nodes, &live, &snap.telemetry)
            .choose_both_with_offers(&model, &nodes, popularity.as_ref(), &offers);
        force_shape(&mut choice, forced);
        let strategy = choice.chosen;
        let placement = match strategy {
            ExecutionStrategy::PipelineChain => chain_plan,
            ExecutionStrategy::ExpertParallel => ep_plan,
        };
        if placement.trunk_ranges.is_empty() {
            skipped.push(format!(
                "{n} node{}: no node can hold the model's trunk",
                if n == 1 { "" } else { "s" }
            ));
            continue;
        }
        let mut node_ids: Vec<String> = nodes.iter().map(|nd| nd.id.0.clone()).collect();
        node_ids.sort();
        let versions: BTreeMap<String, String> = nodes
            .iter()
            .map(|nd| {
                let version = live
                    .get(&nd.id.0)
                    .map(|r| r.installed_version.clone())
                    .unwrap_or_else(|| "in-process".to_string());
                (nd.id.0.clone(), version)
            })
            .collect();
        for &r in &rows {
            let width = if speculation.is_some() { 1 } else { r };
            let shape = LoopShape {
                prompt: prompt.clone(),
                rows: width,
                steps,
                pipelined,
                speculation: speculation.clone(),
                session_ids: (0..sessions)
                    .map(|i| format!("sweep-{tag}-n{n}-r{r}-s{i}"))
                    .collect(),
            };
            let (report, workers, _) = match decode_loop_on_mesh(
                &state, &snap, &model, &nodes, &placement, strategy, &shape,
            )
            .await
            {
                Ok(run) => run,
                Err(e) => {
                    skipped.push(format!("{n} nodes, {r} rows: {e}"));
                    continue;
                }
            };
            record_run(
                &state,
                "sweep",
                &report.report.graph_id,
                Some(strategy),
                &report.report,
                None,
                snap.epoch,
                &workers,
            );
            if !report.success {
                skipped.push(format!(
                    "{n} nodes, {r} rows: validated {} of {} units",
                    report.report.units_validated, report.report.units_total
                ));
                continue;
            }
            let now = now_unix_ms();
            if record_calibrations && sessions == 1 && speculation.is_none() {
                let walls: Vec<f64> = report.sessions[0].steps.iter().map(|s| s.wall_ms).collect();
                let mut exec_by_node: BTreeMap<String, f64> = BTreeMap::new();
                for trace in report.report.traces.iter().filter(|t| t.ok) {
                    *exec_by_node.entry(trace.node_id.0.clone()).or_default() += trace.exec_ms;
                }
                for v in exec_by_node.values_mut() {
                    *v /= walls.len().max(1) as f64;
                }
                let binding = CalibrationBinding::new(
                    &model.id,
                    strategy,
                    placement.digest(),
                    node_ids.clone(),
                    versions.clone(),
                    width,
                );
                let record = CalibrationRecord::from_steps(
                    binding,
                    &walls,
                    exec_by_node,
                    report.sessions[0]
                        .steps
                        .last()
                        .map_or(0, |s| s.output_fnv1a),
                    None,
                    now,
                );
                let mut guard = state.lock().expect("gateway state lock");
                guard.grid_calibrations.record(record);
                let _ = guard.save_grid_calibrations();
                records_added += 1;
            }
            table.push(kadmium_sharding::SweepPoint {
                configuration: kadmium_sharding::SweepConfiguration {
                    model_id: model.id.clone(),
                    strategy,
                    rows: width,
                    sessions,
                    steps,
                    node_count: n,
                    node_ids: node_ids.clone(),
                    placement_digest: placement.digest(),
                    versions: versions.clone(),
                    kernels: kadmium_sharding::CALIBRATION_KERNELS_REFERENCE.to_string(),
                    pipelined,
                    speculative: speculation
                        .as_ref()
                        .map(|s| kadmium_sharding::SweepSpeculation {
                            draft_profile: req
                                .draft_profile
                                .clone()
                                .unwrap_or_else(|| "code".to_string())
                                .to_ascii_lowercase(),
                            max_draft: s.policy.max_draft,
                        }),
                    repository_state: repository_state.clone(),
                },
                measured: report.sweep_measurement(now),
            });
        }
    }
    let title = format!(
        "Sweep `{tag}`: {} point{}",
        table.len(),
        if table.len() == 1 { "" } else { "s" }
    );
    let markdown = table.to_markdown(&title);
    let ledger_path = if req.write_ledger.unwrap_or(true) && !table.is_empty() {
        let safe: String = tag
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let path = std::path::Path::new("target/kadmium-state/sweeps").join(format!("{safe}.md"));
        table
            .append_to_ledger(&path, &title)
            .ok()
            .map(|_| path.display().to_string())
    } else {
        None
    };
    Json(GridSweepResponse {
        tag: tag.clone(),
        curve_ids: table.curve_ids(),
        pareto: table.pareto(),
        curves: table.curves().len(),
        points: table.points,
        markdown,
        ledger_path,
        calibration_records_added: records_added,
        skipped,
        kernels: kadmium_sharding::CALIBRATION_KERNELS_REFERENCE.to_string(),
        error: None,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridTrainRequest {
    pub step_index: Option<u64>,
    pub learning_rate: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridTrainApiResponse {
    pub graph_id: String,
    pub islands: usize,
    pub success: bool,
    pub wall_ms: f64,
    pub reduce: Option<DeMoReduceReport>,
    pub credits_granted: f64,
    pub hops: Vec<LiveHopTelemetry>,
    /// Verde disputes the step settled, one per disputed unit.
    #[serde(default)]
    pub disputes: Vec<VerdeDisputeReport>,
    pub error: Option<String>,
}

pub async fn post_grid_train(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridTrainRequest>>,
) -> Json<GridTrainApiResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let snap = snapshot(&state);
    let step_index = req.step_index.unwrap_or(snap.epoch);
    let learning_rate = req
        .learning_rate
        .filter(|lr| lr.is_finite() && *lr > 0.0)
        .map_or(0.05, |lr| lr.min(1.0));
    let model = snap.training_model.clone();
    let placement = ExpertPlacementPlanner::plan_with_offers(
        &model,
        &snap.nodes,
        ExecutionStrategy::PipelineChain,
        None,
        WorkloadKind::DistributedTraining,
        &snap.offers,
    );
    let session = format!("gateway-e{}", snap.epoch);
    let graph = StepGraphBuilder::training_step(
        &model,
        &placement,
        &snap.nodes,
        snap.epoch,
        &session,
        step_index,
        learning_rate,
    );
    let islands = placement.trunk_ranges.len();
    // The same synthetic micro-batch the legacy training step uses for this step index.
    let input: Vec<f32> = (0..32)
        .map(|i| (((i as u64 + step_index) as f32) * 0.11).cos() * 0.6)
        .collect();
    let target: Vec<f32> = (0..16)
        .map(|j| (((j * 3 + 1) as f32) * 0.17).sin() * 0.5)
        .collect();
    let seeds = match (
        TokenBatchInt8::quantize(&input, 1, 32, 16),
        TokenBatchInt8::quantize(&target, 1, 16, 16),
    ) {
        (Ok(i), Ok(t)) => (i, t),
        (Err(e), _) | (_, Err(e)) => {
            return Json(GridTrainApiResponse {
                graph_id: graph.id.clone(),
                islands,
                success: false,
                wall_ms: 0.0,
                reduce: None,
                credits_granted: 0.0,
                hops: Vec::new(),
                disputes: Vec::new(),
                error: Some(e),
            });
        }
    };
    let catalog = ReplicaCatalog::assume_placed(&placement.rules, now_unix_ms());
    let mut workers = mesh_workers(
        &snap.nodes,
        &snap.live_workers,
        &snap.telemetry,
        &snap.offers,
    );
    let config = GridRunConfig {
        epoch: snap.epoch,
        ..GridRunConfig::default()
    }
    .with_seed(
        StepGraphBuilder::training_input_ref(&graph.id),
        seeds.0.to_le_bytes(),
    )
    .with_seed(
        StepGraphBuilder::training_target_ref(&graph.id),
        seeds.1.to_le_bytes(),
    );
    let local = local_executor(&snap.local_node);
    let report = GridDispatcher::run_graph(
        &graph,
        &mut workers,
        &catalog,
        &snap.live_workers,
        Some(&local),
        &config,
    )
    .await;
    let hops = hops_from_traces(&graph, &report, &snap.nodes, &snap.live_workers);
    let reduce_ref = StepGraphBuilder::output_ref(&graph.id, &"demo-reduce".into());
    let reduce = report
        .output(&reduce_ref.content_id)
        .and_then(|bytes| serde_json::from_slice::<DeMoReduceReport>(bytes).ok());
    record_run(
        &state, "train", &graph.id, None, &report, None, snap.epoch, &workers,
    );
    Json(GridTrainApiResponse {
        graph_id: graph.id.clone(),
        islands,
        success: report.success,
        wall_ms: report.wall_ms,
        reduce,
        credits_granted: report.credits.iter().map(|c| c.credits).sum(),
        hops,
        disputes: report.disputes.clone(),
        error: (!report.success).then(|| {
            format!(
                "validated {} of {} units",
                report.units_validated, report.units_total
            )
        }),
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridReplayRequest {
    pub rollouts: Option<u32>,
    pub mcts_simulations: Option<u32>,
    pub world_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridReplayApiResponse {
    pub graph_id: String,
    pub rollouts: u32,
    pub success: bool,
    pub wall_ms: f64,
    pub nodes_used: Vec<String>,
    pub credits_granted: f64,
    pub receipts: Vec<VerifiedComputeReceipt>,
    pub error: Option<String>,
}

pub async fn post_grid_replay(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<GridReplayRequest>>,
) -> Json<GridReplayApiResponse> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let snap = snapshot(&state);
    let rollouts = req.rollouts.unwrap_or(8).clamp(1, 64);
    let sims = req.mcts_simulations.unwrap_or(32).clamp(8, 256);
    let world_id = req
        .world_id
        .filter(|w| !w.trim().is_empty())
        .unwrap_or_else(|| "gpu-kernel-world".to_string());
    let session = format!("gateway-e{}", snap.epoch);
    let graph = StepGraphBuilder::replay_batch(
        &session,
        snap.epoch,
        &world_id,
        rollouts,
        sims,
        &snap.nodes,
    );
    let catalog = ReplicaCatalog::new();
    let mut workers = mesh_workers(
        &snap.nodes,
        &snap.live_workers,
        &snap.telemetry,
        &snap.offers,
    );
    let config = GridRunConfig {
        epoch: snap.epoch,
        ..GridRunConfig::default()
    };
    let local = local_executor(&snap.local_node);
    let report = GridDispatcher::run_graph(
        &graph,
        &mut workers,
        &catalog,
        &snap.live_workers,
        Some(&local),
        &config,
    )
    .await;
    let mut nodes_used: Vec<String> = report
        .traces
        .iter()
        .filter(|t| t.ok)
        .map(|t| t.node_id.0.clone())
        .collect();
    nodes_used.sort();
    nodes_used.dedup();
    record_run(
        &state, "replay", &graph.id, None, &report, None, snap.epoch, &workers,
    );
    Json(GridReplayApiResponse {
        graph_id: graph.id.clone(),
        rollouts,
        success: report.success,
        wall_ms: report.wall_ms,
        nodes_used,
        credits_granted: report.credits.iter().map(|c| c.credits).sum(),
        receipts: report.receipts.clone(),
        error: (!report.success).then(|| {
            format!(
                "validated {} of {} units",
                report.units_validated, report.units_total
            )
        }),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCreditsResponse {
    pub runs: u64,
    pub entries: Vec<CreditEntry>,
    pub totals_by_node: Vec<(String, f64)>,
    pub receipts: Vec<VerifiedComputeReceipt>,
    pub total_credits: f64,
    pub total_usd: f64,
    pub total_cloud_credits_usd: f64,
}

pub async fn get_grid_credits(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<GridCreditsResponse> {
    let guard = state.lock().expect("gateway state lock");
    let mut totals: BTreeMap<NodeId, f64> = BTreeMap::new();
    for entry in &guard.grid_credit_entries {
        *totals.entry(entry.node_id.clone()).or_default() += entry.credits;
    }
    let mut totals_by_node: Vec<(String, f64)> = totals
        .into_iter()
        .map(|(n, c)| (n.0, (c * 1000.0).round() / 1000.0))
        .collect();
    totals_by_node.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let round4 = |v: f64| (v * 10_000.0).round() / 10_000.0;
    Json(GridCreditsResponse {
        runs: guard.grid_runs,
        entries: guard.grid_credit_entries.clone(),
        totals_by_node,
        total_credits: round4(guard.grid_credit_entries.iter().map(|e| e.credits).sum()),
        total_usd: round4(guard.grid_receipts.iter().map(|r| r.earned_usd).sum()),
        total_cloud_credits_usd: round4(
            guard
                .grid_receipts
                .iter()
                .map(|r| r.earned_cloud_credits_usd)
                .sum(),
        ),
        receipts: guard.grid_receipts.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_state() -> Arc<Mutex<GatewayState>> {
        Arc::new(Mutex::new(GatewayState::new()))
    }

    #[tokio::test]
    async fn decode_step_runs_in_process_across_the_demo_mesh_and_keeps_books() {
        let state = demo_state();
        let outcome = run_grid_decode_step(
            &state,
            "grid decode on the demo fleet",
            1,
            Some(ExecutionStrategy::PipelineChain),
        )
        .await
        .expect("chain step runs");
        assert_eq!(outcome.strategy, ExecutionStrategy::PipelineChain);
        assert!(outcome.report.success);
        assert_eq!(outcome.hops.len(), outcome.placement.trunk_ranges.len());
        assert!(
            outcome
                .hops
                .iter()
                .all(|h| h.transport_mode.contains("in-process"))
        );
        assert!(
            outcome
                .hops
                .windows(2)
                .all(|w| w[0].end_layer + 1 == w[1].start_layer)
        );
        assert_ne!(outcome.final_digest_fnv1a, 0);
        assert!(outcome.modelled_ms_per_token > 0.0);

        let inventory = get_grid_inventory(State(state.clone())).await.0;
        assert_eq!(inventory.runs, 1);
        assert_eq!(inventory.live_workers, 0);
        assert!(inventory.simulated_nodes > 0);
        assert!(inventory.inventory.total_fast_memory_gb > 0.0);
        let last = inventory.last_run.expect("last run recorded");
        assert_eq!(last.kind, "decode");
        assert!(last.credits_granted > 0.0);

        // Every validated unit left a measured compute sample on its worker, and the samples
        // survive into the next run's worker profiles.
        assert_eq!(last.compute_samples, outcome.report.units_total);
        assert_eq!(
            inventory.inventory.compute_samples,
            outcome.report.units_total as u64
        );
        assert_eq!(
            inventory.telemetry.total_samples(),
            outcome.report.units_total as u64
        );

        let credits = get_grid_credits(State(state.clone())).await.0;
        assert_eq!(credits.entries.len(), outcome.report.units_total);
        assert!(credits.total_credits > 0.0);
        assert!(!credits.totals_by_node.is_empty());

        // The same prompt again: the same bytes come out, and the samples double.
        let second = run_grid_decode_step(
            &state,
            "grid decode on the demo fleet",
            1,
            Some(ExecutionStrategy::PipelineChain),
        )
        .await
        .expect("second chain step runs");
        assert_eq!(second.final_digest_fnv1a, outcome.final_digest_fnv1a);
        let after = get_grid_inventory(State(state.clone())).await.0;
        assert_eq!(
            after.telemetry.total_samples(),
            2 * outcome.report.units_total as u64
        );
    }

    #[tokio::test]
    async fn plan_endpoint_reports_both_shapes_and_forced_shapes_are_honoured() {
        let state = demo_state();
        let auto = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert!(auto.modelled);
        assert!(auto.choice.chain.ms_per_token > 0.0);
        assert!(auto.choice.expert_parallel.ms_per_token > 0.0);
        assert_eq!(auto.placement.strategy, auto.choice.chosen);

        let forced = get_grid_plan(
            State(state.clone()),
            Query(GridPlanQuery {
                model: Some("llama-3-70b-lora".into()),
                strategy: Some("expert-parallel".into()),
            }),
        )
        .await
        .0;
        assert_eq!(forced.model_id, "llama-3-70b-lora");
        assert_eq!(forced.placement.strategy, ExecutionStrategy::ExpertParallel);
        assert!(forced.choice.rationale.contains("forced"));
        assert_eq!(
            parse_strategy(Some("chain")),
            Some(ExecutionStrategy::PipelineChain)
        );
        assert_eq!(parse_strategy(Some("nonsense")), None);

        // The plan is reviewed against every node's offer: the fit report, the offers, and the
        // staging that waits for owners whose offers require approval.
        assert!(auto.fit.fits, "{:?}", auto.fit.shortfalls);
        assert_ne!(auto.placement_digest, 0);
        let node_count = state.lock().unwrap().nodes.len();
        assert_eq!(auto.offers.len(), node_count);
        assert!(!auto.fit.per_node.is_empty());
        assert_eq!(
            auto.staging.ready_to_stage,
            auto.staging.nodes_awaiting_approval.is_empty()
        );
        assert!(auto.staging.ready_jobs + auto.staging.awaiting_jobs > 0);
    }

    #[tokio::test]
    async fn decode_steps_learn_routing_and_the_placement_in_force_adapts_by_repin() {
        let state = demo_state();
        let before = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert!(!before.usage.learned);
        assert!(!before.placement_cached, "the first plan is fresh");
        let again = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert!(again.placement_cached, "the plan in force is reused");
        assert_eq!(again.placement_digest, before.placement_digest);

        let outcome = run_grid_decode_step(&state, "learn from this step", 1, None)
            .await
            .expect("step runs");
        assert!(outcome.placement_cached);
        assert!(outcome.routing_recorded > 0, "stage units reported routing");
        let model_id = outcome.placement.model_id.clone();
        {
            let mut guard = state.lock().unwrap();
            let ledger = guard.usage_for(&model_id);
            assert_eq!(ledger.turns, 1);
            assert!(ledger.routed_hits > 0);
            assert!(ledger.sidecar_matches(&model_id));
        }
        let learned = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert!(learned.usage.learned);
        assert_eq!(learned.usage.turns, 1);
        assert!(!learned.usage.hottest.is_empty());

        // A re-pin boundary on the placement in force: proposals never error, and whatever it
        // swaps stays swapped in the cached placement.
        let repin = post_grid_repin(
            State(state.clone()),
            Some(Json(GridRepinRequest {
                model: None,
                max_swaps: Some(2),
                apply: Some(true),
            })),
        )
        .await
        .0;
        assert!(repin.error.is_none(), "{:?}", repin.error);
        assert_eq!(repin.model_id, model_id);
        assert!(repin.swaps.len() + repin.replica_changes.len() <= 2);
        let adds = repin.replica_changes.iter().filter(|c| c.add).count();
        assert_eq!(repin.staging_jobs.len(), 2 * repin.swaps.len() + adds);
        assert!(repin.modelled);
        assert!(repin.expert_read_ms_after <= repin.expert_read_ms_before + 1e-9);
        let after = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert_eq!(after.swaps_applied, repin.swaps.len());
        assert_eq!(
            after.repins_applied,
            usize::from(!(repin.swaps.is_empty() && repin.replica_changes.is_empty()))
        );

        // The expert-parallel shape records routing from its trunk units as well, and a run can
        // ask for a re-pin boundary at its end.
        let ep = run_grid_decode_step_with(
            &state,
            "learn from this step",
            1,
            Some(ExecutionStrategy::ExpertParallel),
            Some(RepinPolicy::default()),
        )
        .await
        .expect("expert-parallel step runs");
        assert_eq!(ep.final_output_fnv1a, outcome.final_output_fnv1a);
        assert_ne!(ep.final_output_fnv1a, 0);
        assert!(ep.routing_recorded > 0);
        let ep_repin = ep.repin.as_ref().expect("a boundary after the step");
        assert!(
            ep_repin.proposal.balance.is_some(),
            "the expert-parallel boundary runs the balancer in its incremental mode"
        );
        assert!(ep_repin.expert_read_ms_after <= ep_repin.expert_read_ms_before + 1e-9);
        // The plan in force reports the balancer's copies and the choice carries the summary
        // when the balancer ran at planning time.
        let plan_after = get_grid_plan(
            State(state.clone()),
            Query(GridPlanQuery {
                model: None,
                strategy: Some("expert-parallel".into()),
            }),
        )
        .await
        .0;
        assert_eq!(
            plan_after.placement.replicated_experts,
            plan_after
                .placement
                .balance
                .as_ref()
                .map_or(0, |b| b.replicas_added)
                + ep_repin
                    .proposal
                    .replica_changes
                    .iter()
                    .filter(|c| c.add)
                    .count()
        );
        assert!(ep.report.accesses.iter().any(|a| a.app == "TrunkForward"));
        assert_eq!(state.lock().unwrap().usage_for(&model_id).turns, 2);
    }

    #[tokio::test]
    async fn calibrations_bind_to_the_plan_in_force_and_decide_once_both_shapes_are_measured() {
        let state = demo_state();
        let chain = post_grid_calibrate(
            State(state.clone()),
            Some(Json(GridCalibrateRequest {
                strategy: Some("chain".into()),
                steps: Some(2),
                rows: None,
                prompt: None,
            })),
        )
        .await
        .0;
        assert!(chain.error.is_none(), "{:?}", chain.error);
        assert_eq!(chain.steps_run, 2);
        assert_eq!(chain.strategy, Some(ExecutionStrategy::PipelineChain));
        let record = chain.record.clone().expect("a record");
        assert_eq!(record.steps, 2);
        assert!(record.measured_ms_per_token > 0.0);
        assert_eq!(record.kernels, "reference-synthetic");
        assert!(chain.applies_to_plan_in_force);
        assert_eq!(chain.digest_matches_chain, None);
        assert_ne!(
            chain.decided_by, "measured",
            "one shape measured decides nothing"
        );

        let plan = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert_eq!(
            plan.choice
                .measured_chain
                .as_ref()
                .map(|r| r.binding.placement_digest),
            Some(record.binding.placement_digest)
        );
        assert!(plan.choice.measured_expert_parallel.is_none());

        let ep = post_grid_calibrate(
            State(state.clone()),
            Some(Json(GridCalibrateRequest {
                strategy: Some("expert-parallel".into()),
                steps: Some(2),
                rows: None,
                prompt: None,
            })),
        )
        .await
        .0;
        assert!(ep.error.is_none(), "{:?}", ep.error);
        assert_eq!(
            ep.digest_matches_chain,
            Some(true),
            "the fan-out bytes match the chain"
        );
        assert!(ep.applies_to_plan_in_force);
        assert_eq!(ep.decided_by, "measured");
        assert!(ep.rationale.starts_with("measured"));

        let decided = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert_eq!(decided.choice.decided_by, "measured");
        assert!(decided.choice.measured_chain.is_some());
        assert!(decided.choice.measured_expert_parallel.is_some());
        let faster = if decided
            .choice
            .measured_chain
            .as_ref()
            .unwrap()
            .measured_ms_per_token
            <= decided
                .choice
                .measured_expert_parallel
                .as_ref()
                .unwrap()
                .measured_ms_per_token
        {
            ExecutionStrategy::PipelineChain
        } else {
            ExecutionStrategy::ExpertParallel
        };
        assert_eq!(decided.choice.chosen, faster);
        assert_eq!(decided.placement.strategy, faster);
        assert_eq!(state.lock().unwrap().grid_calibrations.len(), 2);
    }

    #[tokio::test]
    async fn owners_approve_plans_before_staging_reaches_their_nodes() {
        let state = demo_state();
        // One owner asks to review plans: their node's staging waits for approval.
        let (node_id, mut offer) = {
            let guard = state.lock().unwrap();
            let node = guard
                .nodes
                .iter()
                .find(|n| n.roles.contains(&NodeRole::ComputeWorker))
                .expect("a compute node");
            (node.id.0.clone(), node.default_offer())
        };
        offer.requires_approval = true;
        offer.slots = 2;
        state
            .lock()
            .unwrap()
            .set_worker_offer(&node_id, Some(offer), Some(12.0), None);

        let plan = get_grid_plan(State(state.clone()), Query(GridPlanQuery::default()))
            .await
            .0;
        assert!(plan.offers[&node_id].requires_approval);
        assert_eq!(plan.offers[&node_id].slots, 2);
        assert!(plan.staging.approval_required.contains(&node_id));
        let holds_content = plan.staging.nodes_awaiting_approval.contains(&node_id);
        assert_eq!(
            plan.fit
                .per_node
                .iter()
                .find(|n| n.node.0 == node_id)
                .map(|n| n.requires_approval),
            Some(true)
        );

        let approved = post_grid_plan_approve(
            State(state.clone()),
            Json(GridApproveRequest {
                model: None,
                node: node_id.clone(),
                approve: Some(true),
            }),
        )
        .await
        .0;
        assert_eq!(approved.placement_digest, plan.placement_digest);
        assert!(approved.staging.approved_nodes.contains(&node_id));
        assert!(!approved.staging.nodes_awaiting_approval.contains(&node_id));
        if holds_content {
            assert!(approved.staging.ready_jobs > plan.staging.ready_jobs);
        }

        // The worker profile the scheduler sees carries the offer's slots.
        let workers = {
            let guard = state.lock().unwrap();
            mesh_workers(
                &guard.nodes,
                &guard.live_workers,
                &guard.grid_telemetry,
                &effective_offers(&guard.nodes, &guard.grid_offers),
            )
        };
        let profile = workers.iter().find(|w| w.id().0 == node_id).unwrap();
        assert_eq!(profile.max_concurrent, 2);
        assert_eq!(profile.free_system_ram_gb, Some(12.0));

        // Withdrawing the approval puts the node back on hold, and a changed offer drops it too.
        let withdrawn = post_grid_plan_approve(
            State(state.clone()),
            Json(GridApproveRequest {
                model: None,
                node: node_id.clone(),
                approve: Some(false),
            }),
        )
        .await
        .0;
        assert!(!withdrawn.staging.approved_nodes.contains(&node_id));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn decode_loop_pipelines_steps_and_interleaves_sessions_on_the_demo_mesh() {
        let state = demo_state();
        let request = |pipelined: bool| GridDecodeRequest {
            prompt: Some("decode loop".into()),
            rows: Some(2),
            steps: Some(3),
            sessions: Some(2),
            strategy: Some("chain".into()),
            pipelined: Some(pipelined),
            ..GridDecodeRequest::default()
        };
        let sequential = post_grid_decode(State(state.clone()), Some(Json(request(false))))
            .await
            .0;
        assert!(sequential.error.is_none(), "{:?}", sequential.error);
        assert!(!sequential.pipelined);
        assert!(
            !sequential.interleaved,
            "one graph at a time never overlaps"
        );
        let pipelined = post_grid_decode(State(state.clone()), Some(Json(request(true))))
            .await
            .0;
        assert!(pipelined.error.is_none(), "{:?}", pipelined.error);
        assert!(pipelined.pipelined);
        assert_eq!(pipelined.strategy, Some(ExecutionStrategy::PipelineChain));
        assert_eq!(pipelined.sessions.len(), 2);
        assert_eq!(pipelined.steps_total, 6);
        assert_eq!(pipelined.tokens_total, 12);
        assert_eq!(pipelined.kernels, "reference-synthetic");
        assert!(pipelined.measured_tokens_per_sec > 0.0);
        assert!(pipelined.measured_per_user_tokens_per_sec > 0.0);
        assert!(pipelined.interleaved, "two sessions share the queue");
        for (s, p) in sequential.sessions.iter().zip(&pipelined.sessions) {
            assert_eq!(s.session_id, p.session_id);
            assert_eq!(p.steps, 3);
            assert_eq!(p.inter_step_gap_ms.len(), 2);
            assert!(p.inter_step_gap_ms.iter().all(|g| *g >= 0.0));
            assert_eq!(
                s.final_output_fnv1a, p.final_output_fnv1a,
                "pipelining does not change a session's bytes"
            );
            assert!(p.success);
        }
        assert_ne!(
            pipelined.sessions[0].final_output_fnv1a, pipelined.sessions[1].final_output_fnv1a,
            "sessions start from different prompts and stay apart"
        );
        assert_eq!(pipelined.units_validated, pipelined.units_total);
        assert!(pipelined.credits_granted > 0.0);
        assert!(pipelined.routing_recorded > 0);

        let inventory = get_grid_inventory(State(state.clone())).await.0;
        assert_eq!(inventory.runs, 2);
        let last = inventory.last_run.expect("recorded");
        assert_eq!(last.kind, "decode-loop");
        assert_eq!(last.units_total, pipelined.units_total);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn speculative_decode_sizes_verification_from_the_profile_and_discloses_both_accountings()
    {
        let state = demo_state();
        let request = |profile: &str| GridDecodeRequest {
            prompt: Some("speculative".into()),
            rows: Some(4),
            steps: Some(3),
            sessions: Some(2),
            strategy: Some("chain".into()),
            pipelined: Some(true),
            speculative: Some(true),
            max_draft: Some(3),
            draft_profile: Some(profile.into()),
            early_stop: None,
            confidence_lag: None,
            session: None,
        };
        // No calibration yet: the profile is modelled from the cost model.
        let code = post_grid_decode(State(state.clone()), Some(Json(request("code"))))
            .await
            .0;
        assert!(code.error.is_none(), "{:?}", code.error);
        let v = code.verification.as_ref().expect("a verification summary");
        assert_eq!(v.profile_source, "modelled");
        assert_eq!(v.profile_widths, vec![1, 2, 4, 8, 16]);
        assert_eq!(v.max_draft, 3);
        assert!(v.early_stop);
        assert_eq!(v.confidence_lag, 1);
        assert!(v.modelled_acceptance);
        assert!(v.proposed > 0, "{v:?}");
        assert!(v.accepted <= v.proposed);
        assert!(v.tokens_full_acceptance >= v.tokens_accepted);
        assert_eq!(v.tokens_full_acceptance, 6 + v.proposed);
        assert_eq!(v.tokens_accepted, 6 + v.accepted);
        assert_eq!(
            code.tokens_total, v.tokens_accepted,
            "throughput counts accepted tokens"
        );
        assert!(v.measured_tokens_per_sec_full_acceptance >= v.measured_tokens_per_sec_accepted);
        assert!(v.tau >= 1.0);
        assert!(v.accounting_note.contains("modelled"));
        assert!(
            code.sessions.iter().all(|s| s.rows == 1),
            "a speculative session decodes one row"
        );
        assert!(code.sessions.iter().all(|s| s.accounting.is_some()));
        assert!(v.mean_batch_tokens >= 2.0, "{v:?}");

        // A drafter that is never right verifies nothing and counts the same both ways.
        let none = post_grid_decode(State(state.clone()), Some(Json(request("none"))))
            .await
            .0;
        let v = none.verification.as_ref().unwrap();
        assert_eq!((v.proposed, v.accepted), (0, 0));
        assert_eq!(v.tokens_full_acceptance, v.tokens_accepted);
        assert_eq!(v.mean_batch_tokens, 2.0, "one golden row per session");

        let unknown = post_grid_decode(State(state.clone()), Some(Json(request("poetry"))))
            .await
            .0;
        assert!(
            unknown
                .error
                .as_deref()
                .is_some_and(|e| e.contains("unknown draft profile"))
        );

        // Two calibrated widths of the chain placement make the profile measured.
        for rows in [1u32, 2] {
            let cal = post_grid_calibrate(
                State(state.clone()),
                Some(Json(GridCalibrateRequest {
                    strategy: Some("chain".into()),
                    steps: Some(1),
                    rows: Some(rows),
                    prompt: None,
                })),
            )
            .await
            .0;
            assert!(cal.error.is_none(), "{:?}", cal.error);
        }
        let measured = post_grid_decode(State(state.clone()), Some(Json(request("chat"))))
            .await
            .0;
        assert!(measured.error.is_none(), "{:?}", measured.error);
        let v = measured.verification.as_ref().unwrap();
        assert_eq!(v.profile_source, "measured", "{}", v.profile);
        assert_eq!(v.profile_widths, vec![1, 2]);
        assert_eq!(v.draft_profile, "chat");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn kv_pages_follow_sessions_through_tiers_and_a_sunk_session_replays_before_decoding() {
        let state = demo_state();
        let decode = |session: &str| GridDecodeRequest {
            prompt: Some("kv tiers".into()),
            rows: Some(1),
            steps: Some(2),
            sessions: Some(1),
            strategy: Some("chain".into()),
            pipelined: Some(true),
            session: Some(session.into()),
            ..GridDecodeRequest::default()
        };
        let first = post_grid_decode(State(state.clone()), Some(Json(decode("alpha"))))
            .await
            .0;
        assert!(first.error.is_none(), "{:?}", first.error);
        assert!(
            first.resumed.is_none(),
            "a new session has nothing to replay"
        );
        assert_eq!(first.session_states.len(), 1);
        let alpha = &first.session_states[0];
        assert_eq!(alpha.session_id, "alpha");
        assert_eq!(alpha.tokens, 2);
        assert!(!alpha.needs_replay());
        let layers = state.lock().unwrap().inference_model.num_layers as usize;

        let listed = get_grid_sessions(State(state.clone())).await.0;
        assert!(listed.modelled);
        assert_eq!(listed.sessions.len(), 1);
        let view = &listed.sessions[0];
        assert_eq!(view.pages, layers, "one page per layer");
        assert!(!view.needs_replay);
        assert!(
            view.bytes_by_tier
                .keys()
                .all(|t| t != "LocalNvme" && t != "NasVault"),
            "{:?}",
            view.bytes_by_tier
        );
        let page_bytes = KvPageNames::page_bytes(&state.lock().unwrap().inference_model, 2);
        assert_eq!(
            view.bytes_by_tier.values().sum::<u64>(),
            layers as u64 * page_bytes
        );

        // Idle past the NVMe threshold: the pages sink to the owners' NVMe.
        let now = now_unix_ms();
        let sunk = post_grid_sessions_tier(
            State(state.clone()),
            Some(Json(GridSessionsTierRequest {
                at_unix_ms: Some(now + 2 * 60_000),
                idle_to_nvme_ms: None,
                idle_to_vault_ms: None,
            })),
        )
        .await
        .0;
        assert!(sunk.plan.modelled);
        assert_eq!(sunk.plan.sunk, vec!["alpha".to_string()]);
        assert_eq!(sunk.plan.moves.len(), layers);
        assert!(
            sunk.plan
                .moves
                .iter()
                .all(|m| m.to == ResidencyTier::LocalNvme)
        );
        assert!(sunk.sessions[0].needs_replay);
        assert_eq!(sunk.sessions[0].state.tier, ResidencyTier::LocalNvme);

        // Continuing the session replays its bounded suffix first, then decodes.
        let second = post_grid_decode(State(state.clone()), Some(Json(decode("alpha"))))
            .await
            .0;
        assert!(second.error.is_none(), "{:?}", second.error);
        let replay = second
            .resumed
            .as_ref()
            .expect("a sunk session replays before decoding");
        assert_eq!(replay.session_id, "alpha");
        assert_eq!(replay.from_tier, ResidencyTier::LocalNvme);
        assert_eq!(
            replay.suffix_tokens, 2,
            "a two-token context replays all of it"
        );
        assert_eq!(replay.tokens_total, 2);
        assert_eq!(replay.window_tokens, 128);
        assert_eq!(replay.backtrack_tokens, 12);
        assert!(replay.success);
        assert_eq!(replay.pages_rebuilt, layers);
        assert_eq!(replay.bytes_modelled, layers as u64 * page_bytes);
        assert_eq!(replay.kernels, "reference-synthetic");
        assert!(replay.wall_ms >= 0.0);
        assert_ne!(replay.final_digest_fnv1a, 0);
        let alpha = &second.session_states[0];
        assert_eq!(alpha.tokens, 4);
        assert_eq!(alpha.replays, 1);
        assert_eq!(alpha.replayed_tokens, 2);
        assert!(!alpha.needs_replay());

        // Idle for much longer: the vault; an explicit replay brings the pages back.
        let vaulted = post_grid_sessions_tier(
            State(state.clone()),
            Some(Json(GridSessionsTierRequest {
                at_unix_ms: Some(now + 60 * 60_000),
                idle_to_nvme_ms: None,
                idle_to_vault_ms: None,
            })),
        )
        .await
        .0;
        assert!(
            vaulted
                .plan
                .moves
                .iter()
                .all(|m| m.to == ResidencyTier::NasVault)
        );
        assert_eq!(vaulted.sessions[0].state.tier, ResidencyTier::NasVault);
        let replayed = post_grid_sessions_replay(
            State(state.clone()),
            Json(GridSessionsReplayRequest {
                session: "alpha".into(),
            }),
        )
        .await
        .0;
        assert!(replayed.error.is_none(), "{:?}", replayed.error);
        let r = replayed.replay.unwrap();
        assert_eq!(r.from_tier, ResidencyTier::NasVault);
        assert_eq!(r.suffix_tokens, 4);
        assert!(!replayed.session.unwrap().needs_replay());
        let unknown = post_grid_sessions_replay(
            State(state.clone()),
            Json(GridSessionsReplayRequest {
                session: "nobody".into(),
            }),
        )
        .await
        .0;
        assert!(
            unknown
                .error
                .as_deref()
                .is_some_and(|e| e.contains("no session"))
        );

        let inventory = get_grid_inventory(State(state.clone())).await.0;
        assert_eq!(
            inventory.runs, 4,
            "two loops, the resume's replay and the explicit one"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_calibrates_batch_widths_and_node_counts_into_a_pareto_table() {
        let state = demo_state();
        let node_count = state.lock().unwrap().nodes.len();
        let sweep = post_grid_sweep(
            State(state.clone()),
            Some(Json(GridSweepRequest {
                rows: Some(vec![4, 1, 2]),
                node_counts: Some(vec![node_count, 2]),
                steps: Some(2),
                sessions: Some(1),
                strategy: Some("chain".into()),
                pipelined: Some(true),
                tag: Some("test-sweep".into()),
                write_ledger: Some(true),
                ..GridSweepRequest::default()
            })),
        )
        .await
        .0;
        assert!(sweep.error.is_none(), "{:?}", sweep.error);
        assert!(sweep.skipped.is_empty(), "{:?}", sweep.skipped);
        assert_eq!(sweep.points.len(), 6);
        assert_eq!(sweep.curves, 2, "one curve per node set");
        assert_eq!(sweep.curve_ids.len(), 6);
        assert_eq!(sweep.pareto.len(), 6);
        assert!(sweep.pareto.iter().filter(|p| **p).count() >= 2);
        assert_eq!(sweep.calibration_records_added, 6);
        assert_eq!(sweep.kernels, "reference-synthetic");
        let widths: Vec<u32> = sweep.points.iter().map(|p| p.configuration.rows).collect();
        assert_eq!(
            widths,
            vec![1, 2, 4, 1, 2, 4],
            "widths ascending within each node set"
        );
        for p in &sweep.points {
            assert_eq!(p.configuration.steps, 2);
            assert!(p.configuration.node_count == 2 || p.configuration.node_count == node_count);
            assert_eq!(p.configuration.node_ids.len(), p.configuration.node_count);
            assert!(p.configuration.versions.values().all(|v| v == "in-process"));
            assert!(p.measured.tokens_per_sec > 0.0);
            assert!(p.measured.per_user_tokens_per_sec > 0.0);
            assert_eq!(p.measured.tokens, 2 * p.configuration.rows as u64);
        }
        assert!(sweep.markdown.contains("Sweep `test-sweep`: 6 points"));
        assert!(sweep.markdown.contains("compare only within one curve"));
        let ledger = sweep.ledger_path.as_ref().expect("the ledger was written");
        assert!(ledger.ends_with("test-sweep.md"));
        let text = std::fs::read_to_string(ledger).unwrap();
        assert!(text.contains("reference-synthetic"));
        let _ = std::fs::remove_file(ledger);

        // The whole-mesh chain placement is now calibrated at three widths, so the verification
        // scheduler's profile is measured there.
        assert_eq!(state.lock().unwrap().grid_calibrations.len(), 6);
        let spec = post_grid_decode(
            State(state.clone()),
            Some(Json(GridDecodeRequest {
                steps: Some(2),
                strategy: Some("chain".into()),
                speculative: Some(true),
                max_draft: Some(2),
                draft_profile: Some("code".into()),
                ..GridDecodeRequest::default()
            })),
        )
        .await
        .0;
        assert!(spec.error.is_none(), "{:?}", spec.error);
        let v = spec.verification.unwrap();
        assert_eq!(v.profile_source, "measured", "{}", v.profile);
        assert_eq!(v.profile_widths, vec![1, 2, 4]);

        // A speculative sweep decodes one row per point and reports both accountings.
        let spec_sweep = post_grid_sweep(
            State(state.clone()),
            Some(Json(GridSweepRequest {
                rows: Some(vec![1]),
                node_counts: None,
                steps: Some(2),
                strategy: Some("chain".into()),
                speculative: Some(true),
                max_draft: Some(2),
                draft_profile: Some("chat".into()),
                tag: Some("test-sweep-spec".into()),
                write_ledger: Some(false),
                ..GridSweepRequest::default()
            })),
        )
        .await
        .0;
        assert!(spec_sweep.error.is_none(), "{:?}", spec_sweep.error);
        assert_eq!(spec_sweep.points.len(), 1);
        assert!(spec_sweep.ledger_path.is_none());
        assert_eq!(
            spec_sweep.calibration_records_added, 0,
            "speculative points are not calibrations"
        );
        let p = &spec_sweep.points[0];
        assert_eq!(
            p.configuration
                .speculative
                .as_ref()
                .map(|s| s.draft_profile.as_str()),
            Some("chat")
        );
        assert!(p.measured.tokens_per_sec_full_acceptance.is_some());
        assert!(p.measured.acceptance_rate.is_some());
        assert!(spec_sweep.markdown.contains("(full acceptance)"));
    }

    #[tokio::test]
    async fn training_and_replay_endpoints_run_graphs_on_the_demo_mesh() {
        let state = demo_state();
        let train = post_grid_train(
            State(state.clone()),
            Some(Json(GridTrainRequest {
                step_index: Some(3),
                learning_rate: Some(0.01),
            })),
        )
        .await
        .0;
        assert!(train.success, "{:?}", train.error);
        let reduce = train.reduce.expect("reduce report");
        assert_eq!(reduce.contributors, train.islands);
        assert!(train.credits_granted > 0.0);

        let replay = post_grid_replay(
            State(state.clone()),
            Some(Json(GridReplayRequest {
                rollouts: Some(5),
                mcts_simulations: Some(16),
                world_id: None,
            })),
        )
        .await
        .0;
        assert!(replay.success, "{:?}", replay.error);
        assert_eq!(replay.rollouts, 5);
        assert_eq!(replay.receipts.len(), 5);
        assert!(!replay.nodes_used.is_empty());

        let credits = get_grid_credits(State(state)).await.0;
        assert_eq!(credits.runs, 2);
        assert!(credits.total_usd > 0.0);
    }
}
