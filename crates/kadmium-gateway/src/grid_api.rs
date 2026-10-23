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
    HardwareNode, ModelSpec, NodeId, NodeRole, ResourceOffer, VerifiedComputeReceipt,
    WorkloadKind,
};
use kadmium_grid::{
    AppKind, CreditEntry, PlanApproval, PoolInventory, Replica, ReplicaCatalog, ResidencyTier,
    TelemetryStore, WorkGraph, WorkUnit, WorkerProfile,
};
use kadmium_network::LiveWorkerRegistration;
use kadmium_runtime::{
    DeMoReduceReport, DistributedLanExecutor, DistributedMoEPipelineExecutor,
    DistributedPipelineExecutionReport, ExecutorContext, ExecutorRegistry, GridDispatcher,
    GridRunConfig, GridRunReport, LiveHopTelemetry, LocalExecutor, TokenBatchInt8,
    build_initial_resident_ktb_bank, prompt_seed_hidden,
};
use kadmium_sharding::{
    ClusterShardPlan, CombineParams, ExecutionStrategy, ExpertPlacementPlan,
    ExpertPlacementPlanner, ExpertSliceParams, PlacementFit, StageForwardParams,
    StepGraphBuilder, StrategyChoice, StrategyCostModel, TrainStageParams, TrunkForwardParams,
    TrunkRange,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

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
    }
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

fn cost_model(live_workers: &BTreeMap<String, LiveWorkerRegistration>) -> StrategyCostModel {
    let mut model = StrategyCostModel::default();
    for reg in live_workers.values() {
        if reg.last_measured_rtt_ms > 0.0 {
            model = model.with_measured_rtt(reg.node.id.clone(), reg.last_measured_rtt_ms);
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
    pub final_digest_fnv1a: u64,
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
    let snap = snapshot(state);
    let rows = rows.clamp(1, 64);
    let model = snap.inference_model.clone();
    let cost = cost_model(&snap.live_workers);
    let (mut choice, placement) = cost.choose_with_offers(&model, &snap.nodes, None, &snap.offers);
    let placement = match forced {
        Some(strategy) if strategy != choice.chosen => {
            choice.chosen = strategy;
            choice.rationale = format!("shape forced to {strategy:?}; {}", choice.rationale);
            ExpertPlacementPlanner::plan_with_offers(
                &model,
                &snap.nodes,
                strategy,
                None,
                WorkloadKind::Inference,
                &snap.offers,
            )
        }
        _ => placement,
    };
    if placement.trunk_ranges.is_empty() {
        return Err("no node in the mesh can hold the model's trunk".to_string());
    }
    let strategy = choice.chosen;
    let modelled_ms_per_token = match strategy {
        ExecutionStrategy::PipelineChain => choice.chain.ms_per_token,
        ExecutionStrategy::ExpertParallel => choice.expert_parallel.ms_per_token,
    };

    let session = format!("gateway-e{}", snap.epoch);
    let graph =
        StepGraphBuilder::decode_step(&model, &placement, &snap.nodes, snap.epoch, &session, rows);
    let catalog = ReplicaCatalog::assume_placed(&placement.rules, now_unix_ms());
    let mut workers = mesh_workers(
        &snap.nodes,
        &snap.live_workers,
        &snap.telemetry,
        &snap.offers,
    );
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
    if !report.success {
        return Err(format!(
            "grid step `{}` validated {} of {} units",
            graph.id, report.units_validated, report.units_total
        ));
    }
    Ok(GridStepOutcome {
        strategy,
        choice,
        placement,
        graph_id: graph.id.clone(),
        report,
        hops,
        modelled_ms_per_token,
        final_digest_fnv1a,
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
    /// Always true: the cost model's figures are projections until a run is measured.
    pub modelled: bool,
}

fn plan_response(
    snap: &MeshSnapshot,
    model: ModelSpec,
    forced: Option<ExecutionStrategy>,
) -> GridPlanResponse {
    let (mut choice, mut placement) =
        cost_model(&snap.live_workers).choose_with_offers(&model, &snap.nodes, None, &snap.offers);
    if let Some(strategy) = forced
        && strategy != choice.chosen
    {
        choice.chosen = strategy;
        choice.rationale = format!("shape forced to {strategy:?}; {}", choice.rationale);
        placement = ExpertPlacementPlanner::plan_with_offers(
            &model,
            &snap.nodes,
            strategy,
            None,
            WorkloadKind::Inference,
            &snap.offers,
        );
    }
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
    Json(plan_response(&snap, model, forced))
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
    let (_, placement) =
        cost_model(&snap.live_workers).choose_with_offers(&model, &snap.nodes, None, &snap.offers);
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
    Json(plan_response(&snap, model, None))
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GridRunRequest {
    pub prompt: Option<String>,
    pub rows: Option<u32>,
    pub strategy: Option<String>,
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
    match run_grid_decode_step(&state, &prompt, req.rows.unwrap_or(1), forced).await {
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
                error: Some(error),
            })
        }
    }
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
        &state,
        "train",
        &graph.id,
        None,
        &report,
        None,
        snap.epoch,
        &workers,
    );
    Json(GridTrainApiResponse {
        graph_id: graph.id.clone(),
        islands,
        success: report.success,
        wall_ms: report.wall_ms,
        reduce,
        credits_granted: report.credits.iter().map(|c| c.credits).sum(),
        hops,
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
        &state,
        "replay",
        &graph.id,
        None,
        &report,
        None,
        snap.epoch,
        &workers,
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
        assert_eq!(inventory.telemetry.total_samples(), outcome.report.units_total as u64);

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
        assert_eq!(auto.fit.per_node.len() > 0, true);
        assert_eq!(
            auto.staging.ready_to_stage,
            auto.staging.nodes_awaiting_approval.is_empty()
        );
        assert!(auto.staging.ready_jobs + auto.staging.awaiting_jobs > 0);
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
            plan.fit.per_node.iter().find(|n| n.node.0 == node_id).map(|n| n.requires_approval),
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
