//! Generic work execution on the grid.
//!
//! Three pieces. The [`WorkExecutor`] trait and the [`ExecutorRegistry`] a worker uses to run
//! work units by [`AppKind`]. The binary envelopes that carry a unit and its activation blobs
//! over `WorkDispatch` and `WorkResult` frames. And [`GridDispatcher`], the coordinator-side
//! loop that runs a whole [`WorkGraph`] across the mesh through the `kadmium-grid` queue and the
//! persistent channel pool, carrying the activations one unit produces to the units that depend
//! on them. A pipeline is therefore a chain graph and expert parallelism a fan-out graph, with no
//! special executor for either.
//!
//! The first executor, [`StageForwardExecutor`], runs today's contiguous-layer stage through the
//! same kernels the legacy pipeline calls, so a chain graph reproduces the legacy pipeline
//! digest for digest.

use crate::{
    BlockwiseInt8Tensor, DistributedLanExecutor, LiveStageForwardRequest, TokenBatchInt8,
    execute_stage_kernels_with_ktb,
};
use kadmium_core::NodeId;
use kadmium_grid::{
    AppKind, ContentKind, ContentRef, InstanceId, ReplicaCatalog, SchedulerEventRecord,
    SchedulerPolicy, WorkGraph, WorkQueue, WorkResultRecord, WorkUnit, WorkUnitId, WorkerProfile,
    fnv1a_64,
};
use kadmium_network::{LiveWorkerRegistration, WireFrameKind};
use kadmium_sharding::{StageForwardParams, StepGraphBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

fn now_unix_ms() -> u64 {
    LiveWorkerRegistration::now_unix_ms()
}

/// Activation blobs handed to an executor, keyed by content id.
#[derive(Debug, Clone, Default)]
pub struct WorkInputs {
    blobs: BTreeMap<String, Vec<u8>>,
}

impl WorkInputs {
    pub fn from_blobs(blobs: Vec<(String, Vec<u8>)>) -> Self {
        Self {
            blobs: blobs.into_iter().collect(),
        }
    }

    pub fn blob(&self, content_id: &str) -> Option<&[u8]> {
        self.blobs.get(content_id).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }

    pub fn total_bytes(&self) -> u64 {
        self.blobs.values().map(|b| b.len() as u64).sum()
    }

    /// Every activation input of the unit that was delivered, in the unit's input order.
    pub fn activation_batches(
        &self,
        unit: &WorkUnit,
    ) -> Result<Vec<(String, TokenBatchInt8)>, String> {
        let mut out = Vec::new();
        for input in unit
            .inputs
            .iter()
            .filter(|c| c.kind == ContentKind::Activation)
        {
            if let Some(bytes) = self.blob(&input.content_id) {
                let batch = TokenBatchInt8::from_le_bytes(bytes)
                    .map_err(|e| format!("activation `{}`: {e}", input.content_id))?;
                out.push((input.content_id.clone(), batch));
            }
        }
        Ok(out)
    }

    /// The first activation input of the unit.
    pub fn activation_batch(&self, unit: &WorkUnit) -> Result<TokenBatchInt8, String> {
        self.activation_batches(unit)?
            .into_iter()
            .next()
            .map(|(_, b)| b)
            .ok_or_else(|| format!("unit `{}` received no activation input", unit.id))
    }
}

/// What an executor produced: named output blobs, a small telemetry payload, and the values the
/// scheduler validates on.
#[derive(Debug, Clone, Default)]
pub struct WorkOutput {
    pub outputs: Vec<(ContentRef, Vec<u8>)>,
    pub payload: Vec<u8>,
    pub digest_fnv1a: u64,
    pub check_value: f64,
}

/// Worker-local state an executor may use.
#[derive(Clone)]
pub struct ExecutorContext {
    pub node_id: String,
    pub hostname: String,
    pub resident_ktb_bank: Arc<Mutex<Vec<u8>>>,
}

impl ExecutorContext {
    pub fn new(node_id: &str, hostname: &str, resident_ktb_bank: Arc<Mutex<Vec<u8>>>) -> Self {
        Self {
            node_id: node_id.to_string(),
            hostname: hostname.to_string(),
            resident_ktb_bank,
        }
    }
}

/// Runs one kind of work unit on a worker.
pub trait WorkExecutor: Send + Sync {
    fn app_kind(&self) -> AppKind;
    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String>;
}

/// Executors by app kind, shared by every connection of a worker.
#[derive(Clone, Default)]
pub struct ExecutorRegistry {
    inner: Arc<RwLock<HashMap<AppKind, Arc<dyn WorkExecutor>>>>,
}

impl std::fmt::Debug for ExecutorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutorRegistry")
            .field("kinds", &self.kinds())
            .finish()
    }
}

impl ExecutorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// The executors every worker ships with.
    pub fn with_defaults() -> Self {
        let registry = Self::new();
        registry.register(Arc::new(StageForwardExecutor));
        registry
    }

    pub fn register(&self, executor: Arc<dyn WorkExecutor>) {
        self.inner
            .write()
            .expect("executor registry lock")
            .insert(executor.app_kind(), executor);
    }

    pub fn get(&self, app: &AppKind) -> Option<Arc<dyn WorkExecutor>> {
        self.inner
            .read()
            .expect("executor registry lock")
            .get(app)
            .cloned()
    }

    pub fn kinds(&self) -> Vec<AppKind> {
        let mut kinds: Vec<AppKind> = self
            .inner
            .read()
            .expect("executor registry lock")
            .keys()
            .cloned()
            .collect();
        kinds.sort_by_key(AppKind::label);
        kinds
    }

    pub fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        match self.get(&unit.app) {
            Some(executor) => executor.execute(unit, inputs, ctx),
            None => Err(format!("no executor registered for {}", unit.app.label())),
        }
    }
}

// ================================================================================================
// Wire envelopes: `json_len u32 | json | blob_count u32 | (id_len u32 | id | blob_len u32 | blob)*`
// ================================================================================================

fn encode_envelope(json: &[u8], blobs: &[(String, Vec<u8>)]) -> Vec<u8> {
    let blob_bytes: usize = blobs.iter().map(|(id, b)| 8 + id.len() + b.len()).sum();
    let mut out = Vec::with_capacity(8 + json.len() + blob_bytes);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json);
    out.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
    for (id, blob) in blobs {
        out.extend_from_slice(&(id.len() as u32).to_le_bytes());
        out.extend_from_slice(id.as_bytes());
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(blob);
    }
    out
}

/// Named blobs carried next to a unit or a result, keyed by content id.
pub type WorkBlobs = Vec<(String, Vec<u8>)>;

fn decode_envelope(bytes: &[u8]) -> Result<(Vec<u8>, WorkBlobs), String> {
    let mut at = 0usize;
    let take_u32 = |at: &mut usize| -> Result<usize, String> {
        if bytes.len() < *at + 4 {
            return Err("work envelope truncated".to_string());
        }
        let v = u32::from_le_bytes([bytes[*at], bytes[*at + 1], bytes[*at + 2], bytes[*at + 3]]);
        *at += 4;
        Ok(v as usize)
    };
    let take_bytes = |at: &mut usize, len: usize| -> Result<&[u8], String> {
        if bytes.len() < *at + len {
            return Err("work envelope truncated".to_string());
        }
        let slice = &bytes[*at..*at + len];
        *at += len;
        Ok(slice)
    };

    let json_len = take_u32(&mut at)?;
    let json = take_bytes(&mut at, json_len)?.to_vec();
    let blob_count = take_u32(&mut at)?;
    let mut blobs = Vec::with_capacity(blob_count.min(1024));
    for _ in 0..blob_count {
        let id_len = take_u32(&mut at)?;
        let id = String::from_utf8(take_bytes(&mut at, id_len)?.to_vec())
            .map_err(|e| format!("work envelope blob id is not UTF-8: {e}"))?;
        let blob_len = take_u32(&mut at)?;
        let blob = take_bytes(&mut at, blob_len)?.to_vec();
        blobs.push((id, blob));
    }
    if at != bytes.len() {
        return Err(format!(
            "work envelope has {} trailing bytes",
            bytes.len() - at
        ));
    }
    Ok((json, blobs))
}

/// JSON part of a `WorkDispatch` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkDispatchHeader {
    pub unit: WorkUnit,
    pub instance_id: InstanceId,
    pub epoch: u64,
}

/// A unit on its way to a worker, with the activation blobs it needs.
#[derive(Debug, Clone)]
pub struct WorkDispatchPayload {
    pub header: WorkDispatchHeader,
    pub blobs: Vec<(String, Vec<u8>)>,
}

impl WorkDispatchPayload {
    pub fn new(
        unit: WorkUnit,
        instance_id: InstanceId,
        epoch: u64,
        blobs: Vec<(String, Vec<u8>)>,
    ) -> Self {
        Self {
            header: WorkDispatchHeader {
                unit,
                instance_id,
                epoch,
            },
            blobs,
        }
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let json = serde_json::to_vec(&self.header).expect("serialize WorkDispatchHeader");
        encode_envelope(&json, &self.blobs)
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (json, blobs) = decode_envelope(bytes)?;
        let header: WorkDispatchHeader = serde_json::from_slice(&json)
            .map_err(|e| format!("malformed WorkDispatchHeader: {e}"))?;
        Ok(Self { header, blobs })
    }
}

/// JSON part of a `WorkResult` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResultHeader {
    pub ok: bool,
    pub error: Option<String>,
    pub result: Option<WorkResultRecord>,
}

/// A worker's answer to a dispatch: the result record and the output blobs it produced.
#[derive(Debug, Clone)]
pub struct WorkResultPayload {
    pub header: WorkResultHeader,
    pub blobs: Vec<(String, Vec<u8>)>,
}

impl WorkResultPayload {
    pub fn success(result: WorkResultRecord, blobs: Vec<(String, Vec<u8>)>) -> Self {
        Self {
            header: WorkResultHeader {
                ok: true,
                error: None,
                result: Some(result),
            },
            blobs,
        }
    }

    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            header: WorkResultHeader {
                ok: false,
                error: Some(error.into()),
                result: None,
            },
            blobs: Vec::new(),
        }
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let json = serde_json::to_vec(&self.header).expect("serialize WorkResultHeader");
        encode_envelope(&json, &self.blobs)
    }

    /// Parses an envelope. A bare JSON object with an `error` field, which is what a worker
    /// without an executor registry answers, is accepted as a failure.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        match decode_envelope(bytes) {
            Ok((json, blobs)) => {
                let header: WorkResultHeader = serde_json::from_slice(&json)
                    .map_err(|e| format!("malformed WorkResultHeader: {e}"))?;
                Ok(Self { header, blobs })
            }
            Err(envelope_err) => {
                if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes)
                    && let Some(error) = value.get("error").and_then(|e| e.as_str())
                {
                    return Ok(Self::failure(error));
                }
                Err(envelope_err)
            }
        }
    }

    pub fn error_message(&self) -> Option<&str> {
        self.header.error.as_deref()
    }
}

/// Runs a dispatched unit through the registry and packages the answer. Shared by the worker's
/// frame handler and the dispatcher's in-process fallback.
pub fn execute_dispatch(
    payload: WorkDispatchPayload,
    registry: &ExecutorRegistry,
    ctx: &ExecutorContext,
) -> WorkResultPayload {
    let t0 = Instant::now();
    let WorkDispatchHeader {
        unit, instance_id, ..
    } = payload.header;
    let inputs = WorkInputs::from_blobs(payload.blobs);
    let bytes_in = inputs.total_bytes();
    match registry.execute(&unit, &inputs, ctx) {
        Ok(output) => {
            let exec_ms = (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
            let bytes_out: u64 = output.outputs.iter().map(|(_, b)| b.len() as u64).sum();
            let record = WorkResultRecord::new(
                unit.id.clone(),
                instance_id,
                NodeId(ctx.node_id.clone()),
                output.digest_fnv1a,
                exec_ms.max(0.01),
                now_unix_ms(),
            )
            .with_check_value(output.check_value)
            .with_payload(output.payload)
            .with_outputs(output.outputs.iter().map(|(r, _)| r.clone()).collect())
            .with_bytes_moved(bytes_in + bytes_out);
            let blobs = output
                .outputs
                .into_iter()
                .map(|(r, b)| (r.content_id, b))
                .collect();
            WorkResultPayload::success(record, blobs)
        }
        Err(e) => WorkResultPayload::failure(e),
    }
}

// ================================================================================================
// Stage executor: today's contiguous-layer stage as a work unit
// ================================================================================================

/// Runs a [`AppKind::StageForward`] unit through the same kernels the legacy pipeline uses, one
/// row at a time, and emits the output batch as the unit's activation output.
pub struct StageForwardExecutor;

impl WorkExecutor for StageForwardExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::StageForward
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: StageForwardParams = unit.params_json().map_err(|e| e.to_string())?;
        let batch = inputs.activation_batch(unit)?;
        let rows = batch.dequantize_rows();
        let block_size = batch.tensor.block_size.max(1);
        let bank = ctx.resident_ktb_bank.lock().expect("resident bank lock");

        let mut out_values: Vec<f32> = Vec::with_capacity(rows.len() * batch.dim as usize);
        let mut digests: Vec<u64> = Vec::with_capacity(rows.len());
        let mut l2_sum = 0.0f64;
        let mut last_response = None;

        for row in &rows {
            // A single-row batch keeps the exact tensor it was sent, so the digest matches the
            // legacy path bit for bit; wider batches re-quantize each row.
            let boundary_input = if batch.rows == 1 {
                batch.tensor.clone()
            } else {
                BlockwiseInt8Tensor::quantize(row, block_size)
            };
            let request = LiveStageForwardRequest {
                epoch: params.epoch,
                model_id: params.model_id.clone(),
                stage_index: params.stage_index,
                start_layer: params.start_layer,
                end_layer: params.end_layer,
                q_star_ratio: params.q_star_ratio,
                boundary_input,
                run_speculative_draft: false,
            };
            let response =
                execute_stage_kernels_with_ktb(&ctx.node_id, &ctx.hostname, &request, Some(&bank));
            out_values.extend(response.boundary_output.dequantize());
            digests.push(response.repops_digest_fnv1a);
            l2_sum += response.output_activation_l2_norm as f64;
            last_response = Some(response);
        }
        drop(bank);

        let last = last_response.ok_or_else(|| "stage unit received an empty batch".to_string())?;
        let out_batch = if batch.rows == 1 {
            TokenBatchInt8 {
                rows: 1,
                dim: batch.dim,
                tensor: last.boundary_output.clone(),
            }
        } else {
            TokenBatchInt8::quantize(&out_values, batch.rows, batch.dim, block_size)?
        };
        let digest = if digests.len() == 1 {
            digests[0]
        } else {
            let mut bytes = Vec::with_capacity(digests.len() * 8);
            for d in &digests {
                bytes.extend_from_slice(&d.to_le_bytes());
            }
            fnv1a_64(&bytes)
        };
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                out_batch.to_le_bytes(),
            )],
            payload: serde_json::to_vec(&last).unwrap_or_default(),
            digest_fnv1a: digest,
            check_value: l2_sum / rows.len().max(1) as f64,
        })
    }
}

// ================================================================================================
// Coordinator-side dispatcher
// ================================================================================================

/// How a graph run behaves.
#[derive(Debug, Clone)]
pub struct GridRunConfig {
    pub epoch: u64,
    pub policy: SchedulerPolicy,
    /// Reply deadline per dispatched unit.
    pub call_timeout: Duration,
    /// Activation blobs available before any unit runs, such as the prompt batch.
    pub seed_blobs: Vec<(ContentRef, Vec<u8>)>,
}

impl Default for GridRunConfig {
    fn default() -> Self {
        Self {
            epoch: 1,
            policy: SchedulerPolicy::default(),
            call_timeout: Duration::from_secs(30),
            seed_blobs: Vec::new(),
        }
    }
}

impl GridRunConfig {
    pub fn with_seed(mut self, content: ContentRef, bytes: Vec<u8>) -> Self {
        self.seed_blobs.push((content, bytes));
        self
    }
}

/// One dispatched instance and how it went.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridDispatchTrace {
    pub unit_id: WorkUnitId,
    pub instance_id: InstanceId,
    pub node_id: NodeId,
    pub app: String,
    pub transport: String,
    pub wall_ms: f64,
    pub exec_ms: f64,
    pub digest_fnv1a: u64,
    pub check_value: f64,
    pub ok: bool,
    pub error: Option<String>,
}

/// Outcome of running a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridRunReport {
    pub graph_id: String,
    pub units_total: usize,
    pub units_validated: usize,
    pub units_failed: usize,
    pub success: bool,
    pub wall_ms: f64,
    pub traces: Vec<GridDispatchTrace>,
    pub events: Vec<SchedulerEventRecord>,
    /// Outputs of the graph's sink units, by content id.
    pub outputs: Vec<(String, Vec<u8>)>,
}

impl GridRunReport {
    pub fn output(&self, content_id: &str) -> Option<&[u8]> {
        self.outputs
            .iter()
            .find(|(id, _)| id == content_id)
            .map(|(_, b)| b.as_slice())
    }

    pub fn trace_for(&self, unit_id: &WorkUnitId) -> Option<&GridDispatchTrace> {
        self.traces.iter().rev().find(|t| &t.unit_id == unit_id)
    }
}

/// In-process execution for nodes that have no live worker, such as the coordinator's own node
/// before its worker registers, or a simulated node.
#[derive(Clone)]
pub struct LocalExecutor {
    pub registry: ExecutorRegistry,
    pub ctx: ExecutorContext,
}

struct DispatchOutcome {
    instance_id: InstanceId,
    unit_id: WorkUnitId,
    node_id: NodeId,
    app: String,
    transport: String,
    wall_ms: f64,
    result: Result<WorkResultPayload, String>,
}

pub struct GridDispatcher;

impl GridDispatcher {
    /// Scheduler-facing worker profiles from the live registrations the coordinator keeps.
    pub fn workers_from_registrations(
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
    ) -> Vec<WorkerProfile> {
        live_workers
            .values()
            .map(|reg| {
                let mut worker = WorkerProfile::from_node(reg.node.clone());
                worker.online = reg.online;
                worker.measured_rtt_ms = reg.last_measured_rtt_ms;
                worker.last_seen_unix_ms = reg.last_heartbeat_unix_ms;
                worker
            })
            .collect()
    }

    /// Runs every unit of `graph` to validation: ready units are matched to workers by the
    /// queue, dispatched over the channel pool to their live worker (or run in-process through
    /// `local` when the node has none), and their activation outputs are handed to the units
    /// that depend on them.
    pub async fn run_graph(
        graph: &WorkGraph,
        workers: &mut [WorkerProfile],
        catalog: &ReplicaCatalog,
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        local: Option<&LocalExecutor>,
        config: &GridRunConfig,
    ) -> GridRunReport {
        let t_start = Instant::now();
        let mut queue = WorkQueue::new(config.policy.clone());
        let mut traces = Vec::new();
        let mut blobs: BTreeMap<String, Vec<u8>> = config
            .seed_blobs
            .iter()
            .map(|(r, b)| (r.content_id.clone(), b.clone()))
            .collect();

        if let Err(e) = queue.submit_graph(graph, now_unix_ms()) {
            return GridRunReport {
                graph_id: graph.id.clone(),
                units_total: graph.len(),
                units_validated: 0,
                units_failed: graph.len(),
                success: false,
                wall_ms: 0.0,
                traces,
                events: vec![],
                outputs: vec![],
            }
            .with_error_trace(e.to_string());
        }

        let mut in_flight: JoinSet<DispatchOutcome> = JoinSet::new();

        loop {
            let assigned = queue.dispatch_push(workers, catalog, now_unix_ms());
            let progressed = !assigned.is_empty();
            for instance in assigned {
                let unit = match queue.status(&instance.unit_id) {
                    Some(status) => status.unit.clone(),
                    None => continue,
                };
                let mut attach = Vec::new();
                let mut missing: Option<String> = None;
                for input in unit
                    .inputs
                    .iter()
                    .filter(|c| c.kind == ContentKind::Activation)
                {
                    match blobs.get(&input.content_id) {
                        Some(bytes) => attach.push((input.content_id.clone(), bytes.clone())),
                        None => {
                            missing = Some(input.content_id.clone());
                            break;
                        }
                    }
                }
                let app = unit.app.label();
                if let Some(id) = missing {
                    let error = format!("activation input `{id}` is not available");
                    Self::release_slot(workers, &instance.node_id);
                    let _ = queue.report_failure(&instance.instance_id, &error, now_unix_ms());
                    traces.push(GridDispatchTrace {
                        unit_id: instance.unit_id.clone(),
                        instance_id: instance.instance_id.clone(),
                        node_id: instance.node_id.clone(),
                        app,
                        transport: "not dispatched".into(),
                        wall_ms: 0.0,
                        exec_ms: 0.0,
                        digest_fnv1a: 0,
                        check_value: 0.0,
                        ok: false,
                        error: Some(error),
                    });
                    continue;
                }

                let payload = WorkDispatchPayload::new(
                    unit,
                    instance.instance_id.clone(),
                    config.epoch,
                    attach,
                );
                let (instance_id, unit_id, node_id) = (
                    instance.instance_id.clone(),
                    instance.unit_id.clone(),
                    instance.node_id.clone(),
                );

                if let Some(reg) = live_workers.get(&node_id.0).filter(|w| w.online) {
                    let addr = reg.worker_tcp_addr.clone();
                    let bytes = payload.to_le_bytes();
                    let timeout = config.call_timeout;
                    let epoch = config.epoch as u32;
                    in_flight.spawn(async move {
                        let t0 = Instant::now();
                        let result = DistributedLanExecutor::channel_pool()
                            .call(&addr, WireFrameKind::WorkDispatch, epoch, bytes, timeout)
                            .await
                            .and_then(|frame| WorkResultPayload::from_le_bytes(&frame.payload));
                        DispatchOutcome {
                            instance_id,
                            unit_id,
                            node_id,
                            app,
                            transport: format!("pooled channel {addr}"),
                            wall_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                            result,
                        }
                    });
                } else if let Some(local) = local {
                    let registry = local.registry.clone();
                    let ctx = local.ctx.clone();
                    in_flight.spawn(async move {
                        let t0 = Instant::now();
                        let result = tokio::task::spawn_blocking(move || {
                            execute_dispatch(payload, &registry, &ctx)
                        })
                        .await
                        .map_err(|e| format!("in-process executor task failed: {e}"));
                        DispatchOutcome {
                            instance_id,
                            unit_id,
                            node_id,
                            app,
                            transport: "in-process".into(),
                            wall_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                            result,
                        }
                    });
                } else {
                    let error =
                        format!("node `{node_id}` has no live worker and no local executor");
                    Self::release_slot(workers, &node_id);
                    let _ = queue.report_failure(&instance_id, &error, now_unix_ms());
                    traces.push(GridDispatchTrace {
                        unit_id,
                        instance_id,
                        node_id,
                        app,
                        transport: "not dispatched".into(),
                        wall_ms: 0.0,
                        exec_ms: 0.0,
                        digest_fnv1a: 0,
                        check_value: 0.0,
                        ok: false,
                        error: Some(error),
                    });
                }
            }

            if in_flight.is_empty() {
                // Nothing is running. Failures handled inline may have queued retries, so go
                // round again; when nothing could be assigned either, the run is finished or stuck.
                if progressed {
                    continue;
                }
                break;
            }
            let Some(joined) = in_flight.join_next().await else {
                break;
            };
            let outcome = match joined {
                Ok(outcome) => outcome,
                Err(_) => break,
            };
            Self::release_slot(workers, &outcome.node_id);

            match outcome.result {
                Ok(payload) if payload.header.ok => {
                    let record = payload.header.result.clone().unwrap_or_else(|| {
                        WorkResultRecord::new(
                            outcome.unit_id.clone(),
                            outcome.instance_id.clone(),
                            outcome.node_id.clone(),
                            0,
                            outcome.wall_ms,
                            now_unix_ms(),
                        )
                    });
                    for (id, bytes) in payload.blobs {
                        blobs.insert(id, bytes);
                    }
                    traces.push(GridDispatchTrace {
                        unit_id: outcome.unit_id.clone(),
                        instance_id: outcome.instance_id.clone(),
                        node_id: outcome.node_id.clone(),
                        app: outcome.app,
                        transport: outcome.transport,
                        wall_ms: outcome.wall_ms,
                        exec_ms: record.exec_ms,
                        digest_fnv1a: record.digest_fnv1a,
                        check_value: record.check_value,
                        ok: true,
                        error: None,
                    });
                    let _ = queue.report_result(&outcome.instance_id, record, now_unix_ms());
                }
                Ok(payload) => {
                    let error = payload
                        .error_message()
                        .unwrap_or("worker reported failure without a message")
                        .to_string();
                    traces.push(GridDispatchTrace {
                        unit_id: outcome.unit_id.clone(),
                        instance_id: outcome.instance_id.clone(),
                        node_id: outcome.node_id.clone(),
                        app: outcome.app,
                        transport: outcome.transport,
                        wall_ms: outcome.wall_ms,
                        exec_ms: 0.0,
                        digest_fnv1a: 0,
                        check_value: 0.0,
                        ok: false,
                        error: Some(error.clone()),
                    });
                    let _ = queue.report_failure(&outcome.instance_id, &error, now_unix_ms());
                }
                Err(error) => {
                    traces.push(GridDispatchTrace {
                        unit_id: outcome.unit_id.clone(),
                        instance_id: outcome.instance_id.clone(),
                        node_id: outcome.node_id.clone(),
                        app: outcome.app,
                        transport: outcome.transport,
                        wall_ms: outcome.wall_ms,
                        exec_ms: 0.0,
                        digest_fnv1a: 0,
                        check_value: 0.0,
                        ok: false,
                        error: Some(error.clone()),
                    });
                    let _ = queue.report_failure(&outcome.instance_id, &error, now_unix_ms());
                }
            }
            queue.tick(now_unix_ms(), workers);
        }

        let stats = queue.stats();
        let units_validated = stats.validated + stats.assimilated;
        let units_failed = stats.failed + stats.invalid + stats.expired;
        let mut outputs = Vec::new();
        for unit in graph.units() {
            if !graph.dependents(&unit.id).is_empty() {
                continue;
            }
            if let Some(status) = queue.status(&unit.id)
                && let Some(record) = &status.validated
            {
                for out in &record.outputs {
                    if let Some(bytes) = blobs.get(&out.content_id) {
                        outputs.push((out.content_id.clone(), bytes.clone()));
                    }
                }
            }
        }

        GridRunReport {
            graph_id: graph.id.clone(),
            units_total: graph.len(),
            units_validated,
            units_failed,
            success: units_validated == graph.len(),
            wall_ms: (t_start.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
            traces,
            events: queue.drain_events(),
            outputs,
        }
    }

    fn release_slot(workers: &mut [WorkerProfile], node_id: &NodeId) {
        if let Some(worker) = workers.iter_mut().find(|w| w.id() == node_id) {
            worker.in_flight = worker.in_flight.saturating_sub(1);
        }
    }
}

impl GridRunReport {
    fn with_error_trace(mut self, error: String) -> Self {
        self.traces.push(GridDispatchTrace {
            unit_id: WorkUnitId::new("graph"),
            instance_id: InstanceId("graph#0".into()),
            node_id: NodeId("coordinator".into()),
            app: "submit".into(),
            transport: "not dispatched".into(),
            wall_ms: 0.0,
            exec_ms: 0.0,
            digest_fnv1a: 0,
            check_value: 0.0,
            ok: false,
            error: Some(error),
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DEFAULT_INITIAL_KTB_MERKLE_ROOT, LiveNodeWorkerServer, build_initial_resident_ktb_bank,
        prompt_seed_hidden,
    };
    use kadmium_core::{HardwareNode, ModelSpec, WorkloadKind, sample_personal_cluster};
    use kadmium_grid::{CostEstimate, RedundancyPolicy};
    use kadmium_sharding::{
        ExecutionStrategy, ExpertPlacementPlanner, ShardPlanner, StepGraphBuilder,
    };

    fn registration(node: &HardwareNode, addr: &str, rtt_ms: f64) -> LiveWorkerRegistration {
        LiveWorkerRegistration {
            node: node.clone(),
            worker_tcp_addr: addr.to_string(),
            worker_lan_ip: "127.0.0.1".into(),
            worker_tcp_port: addr
                .rsplit(':')
                .next()
                .and_then(|p| p.parse().ok())
                .unwrap_or(0),
            os_name: "test".into(),
            cpu_arch: "x86_64".into(),
            board_or_model: "test".into(),
            network_channel: "LocalLAN".into(),
            installed_version: "v0.1.0-cd48".into(),
            ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
            heartbeat_count: 1,
            registered_at_unix_ms: 1000,
            last_heartbeat_unix_ms: 1000,
            last_measured_rtt_ms: rtt_ms,
            executed_forward_stages: 0,
            executed_training_steps: 0,
            online: true,
        }
    }

    struct EchoReverse;

    impl WorkExecutor for EchoReverse {
        fn app_kind(&self) -> AppKind {
            AppKind::Custom("echo-reverse".into())
        }

        fn execute(
            &self,
            unit: &WorkUnit,
            inputs: &WorkInputs,
            _ctx: &ExecutorContext,
        ) -> Result<WorkOutput, String> {
            let input = unit
                .inputs
                .first()
                .and_then(|c| inputs.blob(&c.content_id))
                .ok_or("no input")?;
            let mut reversed = input.to_vec();
            reversed.reverse();
            let graph_id = unit.graph_id.as_deref().unwrap_or("g");
            Ok(WorkOutput {
                outputs: vec![(
                    StepGraphBuilder::output_ref(graph_id, &unit.id),
                    reversed.clone(),
                )],
                payload: Vec::new(),
                digest_fnv1a: fnv1a_64(&reversed),
                check_value: reversed.len() as f64,
            })
        }
    }

    fn local_executor(node_id: &str) -> LocalExecutor {
        let (bank, _) = build_initial_resident_ktb_bank("deepseek-v3-671b-moe-q4k", 1);
        let registry = ExecutorRegistry::with_defaults();
        registry.register(Arc::new(EchoReverse));
        LocalExecutor {
            registry,
            ctx: ExecutorContext::new(node_id, node_id, Arc::new(Mutex::new(bank))),
        }
    }

    #[test]
    fn envelopes_roundtrip_and_bare_error_json_is_a_failure() {
        let unit = WorkUnit::new("u", AppKind::StageForward).with_params_bytes(b"{}".to_vec());
        let dispatch = WorkDispatchPayload::new(
            unit,
            InstanceId("u#1".into()),
            7,
            vec![("a".into(), vec![1, 2, 3]), ("b".into(), Vec::new())],
        );
        let decoded = WorkDispatchPayload::from_le_bytes(&dispatch.to_le_bytes()).unwrap();
        assert_eq!(decoded.header.epoch, 7);
        assert_eq!(decoded.header.unit.id, WorkUnitId::new("u"));
        assert_eq!(decoded.blobs, dispatch.blobs);
        assert!(WorkDispatchPayload::from_le_bytes(&dispatch.to_le_bytes()[..10]).is_err());

        let record = WorkResultRecord::new(
            WorkUnitId::new("u"),
            InstanceId("u#1".into()),
            NodeId("n".into()),
            42,
            1.5,
            9,
        );
        let result = WorkResultPayload::success(record, vec![("out".into(), vec![9, 9])]);
        let decoded = WorkResultPayload::from_le_bytes(&result.to_le_bytes()).unwrap();
        assert!(decoded.header.ok);
        assert_eq!(decoded.header.result.unwrap().digest_fnv1a, 42);
        assert_eq!(decoded.blobs[0].1, vec![9, 9]);

        let bare =
            WorkResultPayload::from_le_bytes(br#"{"error":"no executor registered"}"#).unwrap();
        assert!(!bare.header.ok);
        assert_eq!(bare.error_message(), Some("no executor registered"));
        assert!(WorkResultPayload::from_le_bytes(b"not an envelope").is_err());
    }

    #[test]
    fn registry_runs_registered_kinds_and_reports_missing_ones() {
        let local = local_executor("node-a");
        assert!(local.registry.get(&AppKind::StageForward).is_some());
        assert_eq!(local.registry.kinds().len(), 2);

        let input = ContentRef::named(ContentKind::Activation, "g/in", 3);
        let mut unit = WorkUnit::new("rev", AppKind::Custom("echo-reverse".into()));
        unit.inputs = vec![input.clone()];
        unit.graph_id = Some("g".into());
        let payload = WorkDispatchPayload::new(
            unit,
            InstanceId("rev#1".into()),
            1,
            vec![(input.content_id.clone(), vec![1, 2, 3])],
        );
        let result = execute_dispatch(payload, &local.registry, &local.ctx);
        assert!(result.header.ok);
        let record = result.header.result.unwrap();
        assert_eq!(record.node_id, NodeId("node-a".into()));
        assert_eq!(record.check_value, 3.0);
        assert_eq!(result.blobs[0].0, "g/rev/out");
        assert_eq!(result.blobs[0].1, vec![3, 2, 1]);
        assert_eq!(record.bytes_moved, 6);

        let unknown = WorkDispatchPayload::new(
            WorkUnit::new("x", AppKind::DreamReplay),
            InstanceId("x#1".into()),
            1,
            vec![],
        );
        let failed = execute_dispatch(unknown, &local.registry, &local.ctx);
        assert!(!failed.header.ok);
        assert!(
            failed
                .error_message()
                .unwrap()
                .contains("no executor registered")
        );
    }

    #[tokio::test]
    async fn chain_graph_reproduces_the_legacy_pipeline_digest_for_digest() {
        let cluster = sample_personal_cluster();
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cluster[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, cluster[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            cluster[0].id.0.clone(),
            registration(&cluster[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            cluster[1].id.0.clone(),
            registration(&cluster[1], &w1.local_addr.to_string(), 0.2),
        );

        let sub_cluster = vec![cluster[0].clone(), cluster[1].clone(), cluster[5].clone()];
        let model = ModelSpec::deepseek_v3_671b_moe();
        let prompt = "Chain graph equivalence against the legacy pipeline";

        // Legacy path.
        let legacy_plan =
            ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &sub_cluster);
        let (legacy_report, hops) =
            DistributedLanExecutor::execute_live_pipeline(&legacy_plan, prompt, &live_workers)
                .await;
        assert_eq!(hops.len(), 2);

        // Grid path: the same stages as a chain graph.
        let placement = ExpertPlacementPlanner::plan(
            &model,
            &sub_cluster,
            ExecutionStrategy::PipelineChain,
            None,
        );
        assert_eq!(placement.trunk_ranges.len(), 2);
        let graph = StepGraphBuilder::decode_step(&model, &placement, &sub_cluster, 1, "sess", 1);
        let catalog = ReplicaCatalog::assume_placed(&placement.rules, 1);
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let seed = TokenBatchInt8::quantize(&prompt_seed_hidden(prompt, 64), 1, 64, 16).unwrap();
        let config = GridRunConfig::default()
            .with_seed(StepGraphBuilder::prompt_ref(&graph.id), seed.to_le_bytes());
        let report =
            GridDispatcher::run_graph(&graph, &mut workers, &catalog, &live_workers, None, &config)
                .await;
        assert!(report.success, "traces: {:?}", report.traces);
        assert_eq!(report.units_validated, 2);
        assert!(
            report
                .traces
                .iter()
                .all(|t| t.transport.starts_with("pooled channel"))
        );

        for (i, hop) in hops.iter().enumerate() {
            let unit_id = WorkUnitId::new(format!("stage-{i:02}"));
            let trace = report.trace_for(&unit_id).expect("trace per stage");
            assert_eq!(
                trace.node_id.0, hop.node_id,
                "stage {i} ran on the planned node"
            );
            assert_eq!(
                trace.digest_fnv1a, hop.repops_digest_fnv1a,
                "stage {i} RepOps digest matches the legacy hop"
            );
        }
        let last_stage = legacy_report.stages.last().unwrap();
        let out_ref = StepGraphBuilder::output_ref(&graph.id, &WorkUnitId::new("stage-01"));
        let out =
            TokenBatchInt8::from_le_bytes(report.output(&out_ref.content_id).unwrap()).unwrap();
        let l2: f32 = out
            .tensor
            .dequantize()
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!((l2 - last_stage.output_activation_l2_norm).abs() < 1e-2);
        assert!(workers.iter().all(|w| w.in_flight == 0));
        assert_eq!(
            w0.served_work_units
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            w1.served_work_units
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn nodes_without_workers_run_in_process_and_missing_inputs_fail_cleanly() {
        let cluster = sample_personal_cluster();
        let node = cluster[0].clone();
        let live_workers: BTreeMap<String, LiveWorkerRegistration> = BTreeMap::new();
        let mut workers = vec![WorkerProfile::from_node(node.clone())];
        let catalog = ReplicaCatalog::new();
        let local = local_executor(&node.id.0);

        let input = ContentRef::named(ContentKind::Activation, "g/step-1/prompt", 4);
        let mut unit = WorkUnit::new("rev", AppKind::Custom("echo-reverse".into()))
            .with_cost(CostEstimate::new(1.0, 0, 4, 1.0))
            .with_redundancy(RedundancyPolicy::trusted());
        unit.inputs = vec![input.clone()];
        let mut graph = WorkGraph::new("g/step-1");
        graph.add_unit(unit);

        let config = GridRunConfig::default().with_seed(input.clone(), vec![1, 2, 3, 4]);
        let report = GridDispatcher::run_graph(
            &graph,
            &mut workers,
            &catalog,
            &live_workers,
            Some(&local),
            &config,
        )
        .await;
        assert!(report.success, "{:?}", report.traces);
        assert_eq!(report.traces[0].transport, "in-process");
        assert_eq!(report.output("g/step-1/rev/out").unwrap(), &[4, 3, 2, 1]);
        assert_eq!(workers[0].in_flight, 0);

        let no_seed = GridRunConfig::default();
        let failed = GridDispatcher::run_graph(
            &graph,
            &mut workers,
            &catalog,
            &live_workers,
            Some(&local),
            &no_seed,
        )
        .await;
        assert!(!failed.success);
        assert!(failed.units_failed >= 1);
        assert!(failed.traces.iter().any(|t| {
            t.error
                .as_deref()
                .is_some_and(|e| e.contains("not available"))
        }));

        let nowhere =
            GridDispatcher::run_graph(&graph, &mut workers, &catalog, &live_workers, None, &config)
                .await;
        assert!(!nowhere.success);
        assert!(nowhere.traces.iter().any(|t| {
            t.error
                .as_deref()
                .is_some_and(|e| e.contains("no live worker"))
        }));
    }
}
