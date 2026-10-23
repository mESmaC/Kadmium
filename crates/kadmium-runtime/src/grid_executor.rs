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
//!
//! Every result carries the RepOps digest of each layer boundary it ran. When two untrusted
//! copies of a unit disagree, the dispatcher runs the Verde protocol in
//! [`GridDispatcher::referee_dispute`]: it finds the first divergent layer from the two chains,
//! has a disputant open the boundary before it, checks the opening against the digest both
//! committed to, and re-runs that one layer on a trusted referee through a [`VerdeProbe`].

use crate::{
    BlockwiseInt8Tensor, DistributedLanExecutor, ExpertPartial, ExpertPartialsBatch,
    LAYER_BOUNDARY_BLOCK, LayerHook, LayerReference, LayerRouting, LiveStageForwardRequest,
    LiveStageForwardResponse, MoERoutedExpertSelection, TokenBatchF32, TokenBatchInt8,
    VERDE_CHALLENGE_INITIAL_ACTIVATION, execute_stage_layers_with_ktb_routed_hooked, repops,
    verde_challenge_layers,
};
use kadmium_core::{NodeId, VerifiedComputeReceipt};
use kadmium_grid::{
    AppKind, ContentAccess, ContentKind, ContentRef, CreditEntry, CreditLedger, CreditPolicy,
    InstanceId, LayerDigest, RecordedAccess, ReplicaCatalog, SchedulerEvent, SchedulerEventRecord,
    SchedulerPolicy, ValidationOutcome, WorkGraph, WorkQueue, WorkResultRecord, WorkUnit,
    WorkUnitId, WorkerProfile,
};
use kadmium_network::{LiveWorkerRegistration, WireFrameKind};
use kadmium_sharding::{
    CombineParams, ExpertSliceParams, ModelContentNames, StageForwardParams, StepGraphBuilder,
    TrunkForwardParams,
};
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
    /// The RepOps digest of every layer boundary the unit ran, in layer order: what a Verde
    /// bisection compares between disputants. One entry for a unit that is one step, none for
    /// a unit that runs no layers.
    pub layer_digests: Vec<LayerDigest>,
    pub check_value: f64,
    /// Rows, rollouts or simulations the unit covered, for credit and receipts.
    pub units_processed: u64,
    /// Content the unit touched, with hit counts: the experts a layer routed to, for the usage
    /// ledger and frequency-aware eviction.
    pub accessed: Vec<ContentAccess>,
}

/// Content accesses as `(layer, expert) -> hits`, turned into the content ids of a model.
fn expert_accesses(model_id: &str, hits: &BTreeMap<(u32, u32), u64>) -> Vec<ContentAccess> {
    hits.iter()
        .map(|((layer, expert), n)| ContentAccess {
            content_id: ModelContentNames::expert_id(model_id, *layer, *expert),
            hits: *n,
        })
        .collect()
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

/// A Verde probe: the layers a bisection asks a node to run, and from what.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerdeProbe {
    pub from_layer: u32,
    pub to_layer: u32,
    /// Content id of the boundary to start from, an opening delivered with the dispatch. `None`
    /// starts from the unit's own input, which is only valid from the unit's first layer.
    pub state_in: Option<String>,
    /// Return the boundary after `to_layer` as a blob: an opening for the referee.
    pub want_state_out: bool,
}

impl VerdeProbe {
    /// The content id an opened boundary travels under.
    pub fn state_id(unit_id: &WorkUnitId, layer: u32) -> String {
        format!("{unit_id}/verde/state@{layer}")
    }
}

/// What a probe produced.
#[derive(Debug, Clone, Default)]
pub struct VerdeProbeOutput {
    /// One digest per layer run, in layer order.
    pub layer_digests: Vec<LayerDigest>,
    /// The boundary after the last layer run, when the probe asked for it.
    pub state_out: Option<Vec<u8>>,
    pub layers_run: u32,
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

    /// Verde: runs only `probe.from_layer..=probe.to_layer` of the unit, from an opened boundary
    /// or the unit's own input, and reports the digest of every layer it ran, with the boundary
    /// after the last one when asked. Executors whose unit is one indivisible step keep this
    /// default, and a dispute over such a unit re-runs it whole on the referee.
    fn probe(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
        probe: &VerdeProbe,
    ) -> Result<VerdeProbeOutput, String> {
        let _ = (unit, inputs, ctx, probe);
        Err(format!(
            "{} units are not bisectable",
            self.app_kind().label()
        ))
    }
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
        registry.register(Arc::new(StageForwardExecutor::honest()));
        registry.register(Arc::new(TrunkForwardExecutor));
        registry.register(Arc::new(ExpertSliceExecutor));
        registry.register(Arc::new(CombineExecutor));
        registry.register(Arc::new(TrainForwardBackwardExecutor));
        registry.register(Arc::new(DeMoReduceExecutor));
        registry.register(Arc::new(DreamReplayExecutor));
        registry.register(Arc::new(VerdeAuditExecutor::default()));
        registry.register(Arc::new(crate::DraftVerifyExecutor));
        registry.register(Arc::new(crate::KvReplayExecutor));
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

    pub fn probe(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
        probe: &VerdeProbe,
    ) -> Result<VerdeProbeOutput, String> {
        match self.get(&unit.app) {
            Some(executor) => executor.probe(unit, inputs, ctx, probe),
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
    /// Set on a Verde probe: run only these layers of the unit instead of the whole unit.
    #[serde(default)]
    pub probe: Option<VerdeProbe>,
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
                probe: None,
            },
            blobs,
        }
    }

    /// Turns the dispatch into a Verde probe of the unit.
    pub fn with_probe(mut self, probe: VerdeProbe) -> Self {
        self.header.probe = Some(probe);
        self
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
        unit,
        instance_id,
        probe,
        ..
    } = payload.header;
    let inputs = WorkInputs::from_blobs(payload.blobs);
    let bytes_in = inputs.total_bytes();
    if let Some(probe) = probe {
        // A Verde probe: the layers asked for, their digests, and the opened boundary if wanted.
        return match registry.probe(&unit, &inputs, ctx, &probe) {
            Ok(output) => {
                let exec_ms = (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
                let mut outputs = Vec::new();
                let mut blobs = Vec::new();
                if let Some(state) = output.state_out {
                    let state_id = VerdeProbe::state_id(&unit.id, probe.to_layer);
                    outputs.push(ContentRef::named(
                        ContentKind::Activation,
                        &state_id,
                        state.len() as u64,
                    ));
                    blobs.push((state_id, state));
                }
                let bytes_out: u64 = blobs.iter().map(|(_, b)| b.len() as u64).sum();
                let record = WorkResultRecord::new(
                    unit.id.clone(),
                    instance_id,
                    NodeId(ctx.node_id.clone()),
                    LayerReference::unit_digest(&output.layer_digests),
                    exec_ms.max(0.01),
                    now_unix_ms(),
                )
                .with_layer_digests(output.layer_digests)
                .with_outputs(outputs)
                .with_bytes_moved(bytes_in + bytes_out)
                .with_units_processed(output.layers_run as u64)
                .with_payload(serde_json::to_vec(&probe).unwrap_or_default());
                WorkResultPayload::success(record, blobs)
            }
            Err(e) => WorkResultPayload::failure(e),
        };
    }
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
            .with_bytes_moved(bytes_in + bytes_out)
            .with_units_processed(output.units_processed)
            .with_layer_digests(output.layer_digests)
            .with_accessed(output.accessed);
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

/// A fault a broken or dishonest stage would introduce: an offset added to one element of one
/// layer's output before the boundary is quantized. Tests and demos register a stage executor
/// carrying one; every layer after it then diverges, and the bisection must still name this one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InjectedLayerFault {
    pub layer: u32,
    pub element: usize,
    pub offset: f32,
}

/// Runs a [`AppKind::StageForward`] unit through the same kernels the legacy pipeline uses, one
/// row at a time, and emits the output batch as the unit's activation output with the RepOps
/// digest of every layer boundary. Bisectable: a probe runs any sub-range of its layers from an
/// opened boundary.
#[derive(Debug, Clone, Default)]
pub struct StageForwardExecutor {
    /// The fault this executor injects, if it is the dishonest one of a test or a demo.
    pub fault: Option<InjectedLayerFault>,
}

impl StageForwardExecutor {
    pub fn honest() -> Self {
        Self { fault: None }
    }

    pub fn with_injected_fault(fault: InjectedLayerFault) -> Self {
        Self { fault: Some(fault) }
    }

    /// The per-row input tensors of a batch as the stage consumes them: a single-row batch
    /// keeps the exact tensor it was sent, so the digest matches the legacy path bit for bit,
    /// and a wider batch whose rows are whole blocks is split into its rows' exact codes and
    /// scales, so a row reads the same bits alone or beside any neighbours (batch invariance).
    /// Only a batch whose rows straddle blocks re-quantizes each row.
    fn row_inputs(batch: &TokenBatchInt8) -> Vec<BlockwiseInt8Tensor> {
        if batch.rows == 1 {
            return vec![batch.tensor.clone()];
        }
        if let Some(rows) = batch.row_tensors() {
            return rows;
        }
        let block_size = batch.tensor.block_size.max(1);
        batch
            .dequantize_rows()
            .iter()
            .map(|row| BlockwiseInt8Tensor::quantize(row, block_size))
            .collect()
    }

    /// The stage's output batch from the rows' boundary tensors: the exact codes and scales
    /// each row produced, concatenated, when rows are whole blocks; a flat re-quantization of
    /// the dequantized rows otherwise.
    fn output_batch(batch: &TokenBatchInt8, runs: &[StageRowRun]) -> Result<TokenBatchInt8, String> {
        let boundaries: Vec<BlockwiseInt8Tensor> =
            runs.iter().map(|(r, _)| r.boundary_output.clone()).collect();
        if batch.rows == 1 || (batch.dim as usize).is_multiple_of(LAYER_BOUNDARY_BLOCK) {
            return TokenBatchInt8::from_row_tensors(batch.dim, &boundaries);
        }
        let out_values: Vec<f32> = boundaries.iter().flat_map(|b| b.dequantize()).collect();
        TokenBatchInt8::quantize(
            &out_values,
            batch.rows,
            batch.dim,
            batch.tensor.block_size.max(1),
        )
    }

    /// Runs layers `start..=end` for every row, each from its own INT8 tensor, through the
    /// stage kernels with this executor's fault hook.
    fn run_rows(
        &self,
        params: &StageForwardParams,
        row_inputs: Vec<BlockwiseInt8Tensor>,
        start: u32,
        end: u32,
        ctx: &ExecutorContext,
    ) -> Vec<StageRowRun> {
        let bank = ctx.resident_ktb_bank.lock().expect("resident bank lock");
        let fault = self.fault;
        let mut fault_hook = move |layer: u32, out: &mut Vec<f32>| {
            if let Some(f) = fault
                && f.layer == layer
                && let Some(v) = out.get_mut(f.element)
            {
                *v += f.offset;
            }
        };
        let mut runs = Vec::with_capacity(row_inputs.len());
        for boundary_input in row_inputs {
            let request = LiveStageForwardRequest {
                epoch: params.epoch,
                model_id: params.model_id.clone(),
                stage_index: params.stage_index,
                start_layer: start,
                end_layer: end,
                q_star_ratio: params.q_star_ratio,
                boundary_input,
                run_speculative_draft: false,
            };
            let hook: Option<LayerHook<'_>> = if fault.is_some() {
                Some(&mut fault_hook)
            } else {
                None
            };
            runs.push(execute_stage_layers_with_ktb_routed_hooked(
                &ctx.node_id,
                &ctx.hostname,
                &request,
                Some(&bank),
                params.experts_per_layer,
                params.active_experts_per_token,
                hook,
            ));
        }
        runs
    }
}

/// One row of a stage run: the stage response and the experts routed per layer.
type StageRowRun = (LiveStageForwardResponse, Vec<(u32, Vec<u32>)>);

/// Folds per-row layer chains into one chain for the batch: per layer, the rows' digests in
/// row order, as [`LayerReference::batch_boundary_digest`] folds a batch.
pub(crate) fn fold_layer_chains<'a>(rows: impl Iterator<Item = &'a [LayerDigest]>) -> Vec<LayerDigest> {
    let rows: Vec<&[LayerDigest]> = rows.collect();
    let Some(first) = rows.first() else {
        return Vec::new();
    };
    (0..first.len())
        .map(|j| {
            let digests: Vec<u64> = rows
                .iter()
                .filter_map(|r| r.get(j))
                .map(|d| d.digest)
                .collect();
            LayerDigest::new(first[j].layer, LayerReference::fold_row_digests(&digests))
        })
        .collect()
}

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
        let runs = self.run_rows(
            &params,
            Self::row_inputs(&batch),
            params.start_layer,
            params.end_layer,
            ctx,
        );

        let mut l2_sum = 0.0f64;
        let mut routed_hits: BTreeMap<(u32, u32), u64> = BTreeMap::new();
        for (response, routed) in &runs {
            for (layer, experts) in routed {
                for expert in experts {
                    *routed_hits.entry((*layer, *expert)).or_default() += 1;
                }
            }
            l2_sum += response.output_activation_l2_norm as f64;
        }
        let (last, _) = runs
            .last()
            .ok_or_else(|| "stage unit received an empty batch".to_string())?;
        let out_batch = Self::output_batch(&batch, &runs)?;
        // One digest per layer for the batch, then the unit's digest from the chain: for a
        // single row this is the legacy stage digest, bit for bit.
        let layer_digests = fold_layer_chains(runs.iter().map(|(r, _)| r.layer_digests.as_slice()));
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                out_batch.to_le_bytes(),
            )],
            payload: serde_json::to_vec(last).unwrap_or_default(),
            digest_fnv1a: LayerReference::unit_digest(&layer_digests),
            layer_digests,
            check_value: l2_sum / runs.len() as f64,
            units_processed: runs.len() as u64,
            accessed: expert_accesses(&params.model_id, &routed_hits),
        })
    }

    fn probe(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
        probe: &VerdeProbe,
    ) -> Result<VerdeProbeOutput, String> {
        let params: StageForwardParams = unit.params_json().map_err(|e| e.to_string())?;
        if probe.from_layer > probe.to_layer
            || probe.from_layer < params.start_layer
            || probe.to_layer > params.end_layer
        {
            return Err(format!(
                "probe {}..={} is outside the stage's layers {}..={}",
                probe.from_layer, probe.to_layer, params.start_layer, params.end_layer
            ));
        }
        let (dim, row_inputs) = match &probe.state_in {
            Some(id) => {
                let bytes = inputs
                    .blob(id)
                    .ok_or_else(|| format!("probe is missing the opened boundary `{id}`"))?;
                let opened = TokenBatchInt8::from_le_bytes(bytes)?;
                let rows = match opened.row_tensors() {
                    Some(rows) => rows,
                    None if opened.rows == 1 => vec![opened.tensor.clone()],
                    None => {
                        return Err("an opened boundary with several rows needs a width that is a whole number of blocks".to_string());
                    }
                };
                (opened.dim, rows)
            }
            None => {
                if probe.from_layer != params.start_layer {
                    return Err("a probe that does not start at the stage's first layer needs an opened boundary".to_string());
                }
                let batch = inputs.activation_batch(unit)?;
                (batch.dim, Self::row_inputs(&batch))
            }
        };
        let runs = self.run_rows(&params, row_inputs, probe.from_layer, probe.to_layer, ctx);
        if runs.is_empty() {
            return Err("probe received an empty batch".to_string());
        }
        let layer_digests = fold_layer_chains(runs.iter().map(|(r, _)| r.layer_digests.as_slice()));
        let state_out = if probe.want_state_out {
            let boundaries: Vec<BlockwiseInt8Tensor> = runs
                .iter()
                .map(|(r, _)| r.boundary_output.clone())
                .collect();
            Some(TokenBatchInt8::from_row_tensors(dim, &boundaries)?.to_le_bytes())
        } else {
            None
        };
        Ok(VerdeProbeOutput {
            layer_digests,
            state_out,
            layers_run: probe.to_layer - probe.from_layer + 1,
        })
    }
}

// ================================================================================================
// Fan-out executors: one layer as trunk, expert slices and combine on different nodes
// ================================================================================================

fn l2_norm(values: &[f32]) -> f64 {
    repops::tree_sum_squares_f64(values).sqrt()
}

fn activation_input<'a>(
    unit: &'a WorkUnit,
    inputs: &'a WorkInputs,
    pick: impl Fn(&str) -> bool,
) -> Result<(&'a str, &'a [u8]), String> {
    unit.inputs
        .iter()
        .filter(|c| c.kind == ContentKind::Activation && pick(&c.content_id))
        .find_map(|c| {
            inputs
                .blob(&c.content_id)
                .map(|b| (c.content_id.as_str(), b))
        })
        .ok_or_else(|| format!("unit `{}` is missing an activation input", unit.id))
}

/// Multi-row batches are quantized as one flat tensor, which equals per-row quantization only
/// when a row is a whole number of blocks.
fn boundary_batch(rows: &[Vec<f32>]) -> Result<TokenBatchInt8, String> {
    let dim = rows.first().map_or(0, Vec::len);
    if rows.len() > 1 && !dim.is_multiple_of(LAYER_BOUNDARY_BLOCK) {
        return Err(format!(
            "layer boundary batches need a width that is a multiple of {LAYER_BOUNDARY_BLOCK}, got {dim}"
        ));
    }
    if rows.len() == 1 {
        return Ok(TokenBatchInt8 {
            rows: 1,
            dim: dim as u32,
            tensor: LayerReference::boundary_quantize(&rows[0]),
        });
    }
    TokenBatchInt8::quantize(
        &rows.concat(),
        rows.len() as u32,
        dim as u32,
        LAYER_BOUNDARY_BLOCK,
    )
}

/// Runs a [`AppKind::TrunkForward`] unit: the layer's trunk on every row of the INT8 input batch.
/// With experts, the trunk output leaves as exact `f32` for the slices and the combine, with the
/// routing beside it; without, the trunk is the whole layer and its output is boundary-quantized.
pub struct TrunkForwardExecutor;

impl WorkExecutor for TrunkForwardExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::TrunkForward
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: TrunkForwardParams = unit.params_json().map_err(|e| e.to_string())?;
        let batch = inputs.activation_batch(unit)?;
        let rows = batch.dequantize_rows();
        let gamma = {
            let bank = ctx.resident_ktb_bank.lock().expect("resident bank lock");
            LayerReference::gamma_from_bank(Some(&bank), (batch.dim as usize).max(16))
        };
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");

        let mut trunk_rows = Vec::with_capacity(rows.len());
        let mut routing_rows = Vec::with_capacity(rows.len());
        for row in &rows {
            let (h, routing) = LayerReference::trunk_forward(
                row,
                params.layer,
                &gamma,
                params.experts_per_layer,
                params.active_experts_per_token,
                params.q_star_ratio,
            );
            trunk_rows.push(h);
            routing_rows.push(routing);
        }

        if params.experts_per_layer == 0 || params.active_experts_per_token == 0 {
            // A dense layer: the trunk is the whole layer, so this is the layer boundary.
            let combined: Vec<Vec<f32>> = trunk_rows
                .iter()
                .map(|h| LayerReference::combine(h, &[]))
                .collect();
            let out = boundary_batch(&combined)?;
            // The trunk is the layer boundary here, so its digest is the layer's boundary
            // digest: the value a chain stage records for the same layer.
            let digest = LayerReference::batch_boundary_digest(params.layer, &out);
            return Ok(WorkOutput {
                outputs: vec![(
                    StepGraphBuilder::output_ref(graph_id, &unit.id),
                    out.to_le_bytes(),
                )],
                payload: Vec::new(),
                digest_fnv1a: digest,
                layer_digests: vec![LayerDigest::new(params.layer, digest)],
                check_value: l2_norm(&combined.concat()),
                units_processed: rows.len() as u64,
                accessed: Vec::new(),
            });
        }

        let out = TokenBatchF32::from_rows(&trunk_rows)?;
        let mut routed_hits: BTreeMap<(u32, u32), u64> = BTreeMap::new();
        for row in &routing_rows {
            for selection in row {
                *routed_hits
                    .entry((params.layer, selection.expert_id))
                    .or_default() += 1;
            }
        }
        let routing = LayerRouting {
            layer: params.layer,
            rows: routing_rows,
        };
        let digest = LayerReference::trunk_digest(params.layer, &out);
        Ok(WorkOutput {
            outputs: vec![
                (
                    StepGraphBuilder::output_ref(graph_id, &unit.id),
                    out.to_le_bytes(),
                ),
                (
                    StepGraphBuilder::routing_ref(graph_id, &unit.id),
                    routing.to_json_bytes(),
                ),
            ],
            payload: routing.to_json_bytes(),
            digest_fnv1a: digest,
            layer_digests: vec![LayerDigest::new(params.layer, digest)],
            check_value: l2_norm(&out.data),
            units_processed: rows.len() as u64,
            accessed: expert_accesses(&params.model_id, &routed_hits),
        })
    }
}

/// Runs an [`AppKind::ExpertSlice`] unit: for every row, the contribution of each selected expert
/// resident on this holder, from the trunk output and routing the trunk unit produced.
pub struct ExpertSliceExecutor;

impl WorkExecutor for ExpertSliceExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::ExpertSlice
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        _ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: ExpertSliceParams = unit.params_json().map_err(|e| e.to_string())?;
        let (_, trunk_bytes) = activation_input(unit, inputs, |id| id.ends_with("/out"))?;
        let (_, routing_bytes) = activation_input(unit, inputs, |id| id.ends_with("/routing"))?;
        let trunk = TokenBatchF32::from_le_bytes(trunk_bytes)?;
        let routing = LayerRouting::from_json_bytes(routing_bytes)?;
        if routing.rows.len() != trunk.rows as usize {
            return Err(format!(
                "routing covers {} rows but the trunk output has {}",
                routing.rows.len(),
                trunk.rows
            ));
        }

        let mut partials = Vec::new();
        for (row, selections) in routing.rows.iter().enumerate() {
            let h = trunk.row(row);
            // The experts this holder serves for this row: its resident experts, minus the rows
            // of a replicated expert that another copy takes (EPLB's even split).
            let mut mine: Vec<&MoERoutedExpertSelection> = selections
                .iter()
                .filter(|s| params.serves(s.expert_id, row as u32))
                .collect();
            mine.sort_by_key(|s| s.expert_id);
            for selection in mine {
                partials.push(ExpertPartial {
                    row: row as u32,
                    expert: selection.expert_id,
                    data: LayerReference::expert_contribution(
                        h,
                        params.layer,
                        selection.expert_id,
                        selection.gating_weight,
                    ),
                });
            }
        }
        let applied = partials.len();
        let mut served_hits: BTreeMap<(u32, u32), u64> = BTreeMap::new();
        for partial in &partials {
            *served_hits
                .entry((params.layer, partial.expert))
                .or_default() += 1;
        }
        let batch = ExpertPartialsBatch {
            layer: params.layer,
            rows: trunk.rows,
            dim: trunk.dim,
            partials,
        };
        let digest = LayerReference::partials_digest(&batch);
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                batch.to_le_bytes(),
            )],
            payload: serde_json::to_vec(&serde_json::json!({
                "layer": params.layer,
                "holder": params.holder.0,
                "experts_applied": applied,
            }))
            .unwrap_or_default(),
            digest_fnv1a: digest,
            layer_digests: vec![LayerDigest::new(params.layer, digest)],
            check_value: applied as f64,
            units_processed: trunk.rows as u64,
            accessed: expert_accesses(&params.model_id, &served_hits),
        })
    }
}

/// Runs an [`AppKind::Combine`] unit: the residual plus every holder's partials in expert-id
/// order, clamped and quantized at the layer boundary, which the next trunk consumes.
pub struct CombineExecutor;

impl WorkExecutor for CombineExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::Combine
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        _ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: CombineParams = unit.params_json().map_err(|e| e.to_string())?;
        let (_, trunk_bytes) = activation_input(unit, inputs, |id| id.contains("-trunk/"))?;
        let trunk = TokenBatchF32::from_le_bytes(trunk_bytes)?;
        let mut partial_batches = Vec::new();
        for input in unit
            .inputs
            .iter()
            .filter(|c| c.kind == ContentKind::Activation && !c.content_id.contains("-trunk/"))
        {
            let bytes = inputs.blob(&input.content_id).ok_or_else(|| {
                format!(
                    "combine `{}` is missing partials `{}`",
                    unit.id, input.content_id
                )
            })?;
            partial_batches.push(ExpertPartialsBatch::from_le_bytes(bytes)?);
        }

        let mut out_rows = Vec::with_capacity(trunk.rows as usize);
        for row in 0..trunk.rows {
            let partials: Vec<(u32, Vec<f32>)> = partial_batches
                .iter()
                .flat_map(|b| b.row_partials(row))
                .collect();
            out_rows.push(LayerReference::combine(trunk.row(row as usize), &partials));
        }
        let out = boundary_batch(&out_rows)?;
        // The layer boundary: the same digest a chain stage records for this layer.
        let digest = LayerReference::batch_boundary_digest(params.layer, &out);
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                out.to_le_bytes(),
            )],
            payload: serde_json::to_vec(&serde_json::json!({
                "layer": params.layer,
                "contributors": params.contributors.len(),
                "partials": partial_batches.iter().map(|b| b.partials.len()).sum::<usize>(),
            }))
            .unwrap_or_default(),
            digest_fnv1a: digest,
            layer_digests: vec![LayerDigest::new(params.layer, digest)],
            check_value: l2_norm(&out_rows.concat()),
            units_processed: trunk.rows as u64,
            accessed: Vec::new(),
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
    /// Credit policy for the run's ledger; `None` keeps no books.
    pub credit_policy: Option<CreditPolicy>,
    /// The instant every trace's `dispatched_at_ms` and `returned_at_ms` count from, so the
    /// traces of several runs under one config compare on one clock.
    pub clock_origin: Instant,
}

impl Default for GridRunConfig {
    fn default() -> Self {
        Self {
            epoch: 1,
            policy: SchedulerPolicy::default(),
            call_timeout: Duration::from_secs(30),
            seed_blobs: Vec::new(),
            credit_policy: Some(CreditPolicy::default()),
            clock_origin: Instant::now(),
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
    /// The RepOps digest of every layer the unit ran, from the result record.
    #[serde(default)]
    pub layer_digests: Vec<LayerDigest>,
    pub check_value: f64,
    pub ok: bool,
    pub error: Option<String>,
    /// When the unit left the coordinator and when its answer arrived, in ms on the run
    /// config's clock: the inter-step gap is the next step's first `dispatched_at_ms` minus
    /// this step's last `returned_at_ms`. Measured on the coordinator.
    #[serde(default)]
    pub dispatched_at_ms: f64,
    #[serde(default)]
    pub returned_at_ms: f64,
}

/// One Verde dispute and how the referee settled it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerdeDisputeReport {
    pub unit_id: WorkUnitId,
    pub app: String,
    pub disputants: Vec<NodeId>,
    pub referee: Option<NodeId>,
    /// `layer-bisection` when the disputants' digest chains diverge; `whole-unit` when they do
    /// not (a one-step unit, or a result whose digest disagrees while its chain does not) or
    /// when the executor refused a probe.
    pub method: String,
    /// Layers the unit ran, from the length of its digest chain; 1 for a one-step unit.
    pub layers_in_unit: u32,
    /// Digest comparisons made to find the first divergent layer: that layer's position plus
    /// one, never more than `layers_in_unit`. The chains travel with the results, so this is
    /// what the interactive rounds of a bisection cost here: comparisons, not re-execution.
    pub digest_comparisons: u32,
    /// Openings of the divergent layer's input asked of the disputants, and how many of them
    /// reproduced the digest both had committed to.
    pub openings_requested: u32,
    pub openings_verified: u32,
    pub divergent_layer: Option<u32>,
    /// Layers the referee itself ran: one when an opening verified, the layers before the
    /// divergent one as well when it had to reconstruct the input, the whole unit otherwise.
    pub referee_layers_run: u32,
    pub winner: Option<NodeId>,
    pub losers: Vec<NodeId>,
    pub outcome: String,
    pub wall_ms: f64,
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
    /// Verde disputes the run settled, one per disputed unit.
    #[serde(default)]
    pub disputes: Vec<VerdeDisputeReport>,
    /// Validated results folded into the workers' measured compute estimates during the run.
    #[serde(default)]
    pub compute_samples: usize,
    /// Content the validated units touched, summed per node, kind of work and content id: the
    /// experts each layer routed to and each holder served.
    #[serde(default)]
    pub accesses: Vec<RecordedAccess>,
    /// Outputs of the graph's sink units, by content id.
    pub outputs: Vec<(String, Vec<u8>)>,
    /// Credit granted for validated units, one entry per unit.
    pub credits: Vec<CreditEntry>,
    /// The credits as supply-grid receipts at the canonical clearing rates.
    pub receipts: Vec<VerifiedComputeReceipt>,
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
    dispatched_at_ms: f64,
    returned_at_ms: f64,
    result: Result<WorkResultPayload, String>,
}

fn ms_since(origin: Instant) -> f64 {
    (origin.elapsed().as_secs_f64() * 1000.0 * 1000.0).round() / 1000.0
}

/// Where a referee action runs: a trusted worker over its channel, or the in-process executor.
enum RefereeTarget {
    Worker {
        node: NodeId,
        addr: String,
    },
    Local {
        node: NodeId,
        executor: LocalExecutor,
    },
}

impl RefereeTarget {
    fn node(&self) -> &NodeId {
        match self {
            Self::Worker { node, .. } | Self::Local { node, .. } => node,
        }
    }
}

/// What the referee concluded.
enum Verdict {
    Undecided,
    /// The executor refused the probe: the unit is one indivisible step.
    NotBisectable,
    Winner(NodeId, String),
    Neither(String),
    Failed(String),
}

impl Verdict {
    fn from_failure(message: Option<&str>) -> Self {
        let message = message.unwrap_or("worker reported failure without a message");
        if message.contains("not bisectable") {
            Self::NotBisectable
        } else {
            Self::Failed(message.to_string())
        }
    }
}

fn elapsed_ms(t0: Instant) -> f64 {
    (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0
}

/// The trace of one referee action: a probe, an opening or a whole-unit re-run.
#[allow(clippy::too_many_arguments)]
fn referee_trace(
    unit_id: &WorkUnitId,
    instance_id: InstanceId,
    node_id: NodeId,
    app: &str,
    transport: String,
    wall_ms: f64,
    outcome: &Result<WorkResultPayload, String>,
    returned_at_ms: f64,
) -> GridDispatchTrace {
    let record = outcome
        .as_ref()
        .ok()
        .filter(|p| p.header.ok)
        .and_then(|p| p.header.result.as_ref());
    GridDispatchTrace {
        unit_id: unit_id.clone(),
        instance_id,
        node_id,
        app: app.to_string(),
        transport,
        wall_ms,
        exec_ms: record.map_or(0.0, |r| r.exec_ms),
        digest_fnv1a: record.map_or(0, |r| r.digest_fnv1a),
        layer_digests: record.map(|r| r.layer_digests.clone()).unwrap_or_default(),
        check_value: record.map_or(0.0, |r| r.check_value),
        ok: record.is_some(),
        error: match outcome {
            Ok(p) if p.header.ok => None,
            Ok(p) => Some(
                p.error_message()
                    .unwrap_or("worker reported failure without a message")
                    .to_string(),
            ),
            Err(e) => Some(e.clone()),
        },
        dispatched_at_ms: ((returned_at_ms - wall_ms).max(0.0) * 1000.0).round() / 1000.0,
        returned_at_ms,
    }
}

/// Which unit a scheduler event is about.
fn event_unit_id(event: &SchedulerEvent) -> Option<WorkUnitId> {
    let from_instance = |i: &InstanceId| {
        WorkUnitId::new(i.0.rsplit_once('#').map_or(i.0.as_str(), |(u, _)| u))
    };
    Some(match event {
        SchedulerEvent::Submitted { unit }
        | SchedulerEvent::Released { unit }
        | SchedulerEvent::Assigned { unit, .. }
        | SchedulerEvent::Validated { unit, .. }
        | SchedulerEvent::DisputeOpened { unit }
        | SchedulerEvent::DisputeSettled { unit, .. }
        | SchedulerEvent::Duplicated { unit, .. }
        | SchedulerEvent::Retried { unit, .. }
        | SchedulerEvent::Failed { unit, .. }
        | SchedulerEvent::Expired { unit }
        | SchedulerEvent::Assimilated { unit }
        | SchedulerEvent::Cancelled { unit } => unit.clone(),
        SchedulerEvent::Started { instance }
        | SchedulerEvent::Returned { instance, .. }
        | SchedulerEvent::Invalidated { instance, .. } => from_instance(instance),
    })
}

/// The dispatcher's loop kept open across graphs: one queue, one blob store and one set of
/// in-flight calls, into which graphs are submitted while others run. The successive steps of
/// a session chain through it (step `t + 1` waits on the sink of step `t` and is dispatched the
/// moment that sink validates, with nothing planned in between), and the graphs of several
/// sessions interleave in its queue, so the stages of different sessions run on different
/// nodes at the same time. DeepInfer's asynchronous scheduling overlaps the CPU's scheduling
/// of step `N + 1` with the GPU's forward of step `N`; this is that overlap for a mesh, with
/// the coordinator's planning of the next graph overlapping the workers' execution of the
/// current one.
pub struct GridRunner<'a> {
    queue: WorkQueue,
    workers: &'a mut [WorkerProfile],
    catalog: &'a ReplicaCatalog,
    live_workers: &'a BTreeMap<String, LiveWorkerRegistration>,
    local: Option<&'a LocalExecutor>,
    config: &'a GridRunConfig,
    blobs: BTreeMap<String, Vec<u8>>,
    /// Every returned instance's outputs, so a dispute can put the winner's back in
    /// circulation whichever disputant returned last.
    returned_blobs: BTreeMap<InstanceId, WorkBlobs>,
    in_flight: JoinSet<DispatchOutcome>,
    traces: Vec<GridDispatchTrace>,
    disputes: Vec<VerdeDisputeReport>,
    compute_samples: usize,
    graphs: Vec<WorkGraph>,
}

impl<'a> GridRunner<'a> {
    pub fn new(
        workers: &'a mut [WorkerProfile],
        catalog: &'a ReplicaCatalog,
        live_workers: &'a BTreeMap<String, LiveWorkerRegistration>,
        local: Option<&'a LocalExecutor>,
        config: &'a GridRunConfig,
    ) -> Self {
        Self {
            queue: WorkQueue::new(config.policy.clone()),
            workers,
            catalog,
            live_workers,
            local,
            config,
            blobs: config
                .seed_blobs
                .iter()
                .map(|(r, b)| (r.content_id.clone(), b.clone()))
                .collect(),
            returned_blobs: BTreeMap::new(),
            in_flight: JoinSet::new(),
            traces: Vec::new(),
            disputes: Vec::new(),
            compute_samples: 0,
            graphs: Vec::new(),
        }
    }

    /// Milliseconds on the run config's clock.
    pub fn now_ms(&self) -> f64 {
        ms_since(self.config.clock_origin)
    }

    /// Submits a graph into the queue, with `external` prerequisites on units already there
    /// (the sink of the previous step) and `seeds` the graph's units read. Nothing is sent
    /// until [`GridRunner::dispatch_ready`].
    pub fn submit(
        &mut self,
        graph: &WorkGraph,
        external: &[(WorkUnitId, WorkUnitId)],
        seeds: Vec<(ContentRef, Vec<u8>)>,
    ) -> Result<(), kadmium_grid::GridError> {
        self.queue
            .submit_graph_after(graph, external, now_unix_ms())?;
        for (content, bytes) in seeds {
            self.blobs.insert(content.content_id, bytes);
        }
        self.graphs.push(graph.clone());
        Ok(())
    }

    /// A blob the runner holds, such as a step's sink output.
    pub fn blob(&self, content_id: &str) -> Option<&[u8]> {
        self.blobs.get(content_id).map(Vec::as_slice)
    }

    /// The validated result's payload of a unit, such as a verify unit's acceptance report.
    pub fn payload_of(&self, unit_id: &WorkUnitId) -> Option<&[u8]> {
        self.queue
            .status(unit_id)
            .and_then(|s| s.validated.as_ref())
            .map(|r| r.payload.as_slice())
    }

    pub fn graph_settled(&self, graph: &WorkGraph) -> bool {
        self.queue.graph_settled(graph)
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    pub fn traces(&self) -> &[GridDispatchTrace] {
        &self.traces
    }

    /// Compute samples the runner fed to its workers so far.
    pub fn compute_samples(&self) -> usize {
        self.compute_samples
    }

    fn failed_trace(
        instance: &kadmium_grid::WorkInstance,
        app: String,
        error: String,
        now_ms: f64,
    ) -> GridDispatchTrace {
        GridDispatchTrace {
            unit_id: instance.unit_id.clone(),
            instance_id: instance.instance_id.clone(),
            node_id: instance.node_id.clone(),
            app,
            transport: "not dispatched".into(),
            wall_ms: 0.0,
            exec_ms: 0.0,
            digest_fnv1a: 0,
            layer_digests: Vec::new(),
            check_value: 0.0,
            ok: false,
            error: Some(error),
            dispatched_at_ms: now_ms,
            returned_at_ms: now_ms,
        }
    }

    /// Assigns every ready unit to its best worker and sends it: over the channel pool to a
    /// live worker, in-process through the local executor otherwise. Returns how many
    /// instances the queue assigned, dispatched or not.
    pub fn dispatch_ready(&mut self) -> usize {
        let assigned = self
            .queue
            .dispatch_push(self.workers, self.catalog, now_unix_ms());
        let count = assigned.len();
        for instance in assigned {
            let unit = match self.queue.status(&instance.unit_id) {
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
                match self.blobs.get(&input.content_id) {
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
                GridDispatcher::release_slot(self.workers, &instance.node_id);
                let _ = self
                    .queue
                    .report_failure(&instance.instance_id, &error, now_unix_ms());
                let now_ms = self.now_ms();
                self.traces
                    .push(Self::failed_trace(&instance, app, error, now_ms));
                continue;
            }

            let payload = WorkDispatchPayload::new(
                unit,
                instance.instance_id.clone(),
                self.config.epoch,
                attach,
            );
            let (instance_id, unit_id, node_id) = (
                instance.instance_id.clone(),
                instance.unit_id.clone(),
                instance.node_id.clone(),
            );
            let origin = self.config.clock_origin;

            if let Some(reg) = self.live_workers.get(&node_id.0).filter(|w| w.online) {
                let addr = reg.worker_tcp_addr.clone();
                let bytes = payload.to_le_bytes();
                let timeout = self.config.call_timeout;
                let epoch = self.config.epoch as u32;
                let dispatched_at_ms = ms_since(origin);
                self.in_flight.spawn(async move {
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
                        dispatched_at_ms,
                        returned_at_ms: ms_since(origin),
                        result,
                    }
                });
            } else if let Some(local) = self.local {
                let registry = local.registry.clone();
                let ctx = local.ctx.clone();
                let dispatched_at_ms = ms_since(origin);
                self.in_flight.spawn(async move {
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
                        dispatched_at_ms,
                        returned_at_ms: ms_since(origin),
                        result,
                    }
                });
            } else {
                let error = format!("node `{node_id}` has no live worker and no local executor");
                GridDispatcher::release_slot(self.workers, &node_id);
                let _ = self
                    .queue
                    .report_failure(&instance_id, &error, now_unix_ms());
                let now_ms = self.now_ms();
                self.traces
                    .push(Self::failed_trace(&instance, app, error, now_ms));
            }
        }
        count
    }

    /// Waits for one in-flight unit and settles it: the result is validated (a disagreement
    /// is refereed), its outputs join the blob store, its measured cost feeds its worker, and
    /// the queue is ticked. False when nothing is in flight.
    pub async fn settle_one(&mut self) -> bool {
        let Some(joined) = self.in_flight.join_next().await else {
            return false;
        };
        let outcome = match joined {
            Ok(outcome) => outcome,
            Err(_) => return false,
        };
        GridDispatcher::release_slot(self.workers, &outcome.node_id);

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
                for (id, bytes) in &payload.blobs {
                    self.blobs.insert(id.clone(), bytes.clone());
                }
                self.returned_blobs
                    .insert(outcome.instance_id.clone(), payload.blobs);
                self.traces.push(GridDispatchTrace {
                    unit_id: outcome.unit_id.clone(),
                    instance_id: outcome.instance_id.clone(),
                    node_id: outcome.node_id.clone(),
                    app: outcome.app,
                    transport: outcome.transport,
                    wall_ms: outcome.wall_ms,
                    exec_ms: record.exec_ms,
                    digest_fnv1a: record.digest_fnv1a,
                    layer_digests: record.layer_digests.clone(),
                    check_value: record.check_value,
                    ok: true,
                    error: None,
                    dispatched_at_ms: outcome.dispatched_at_ms,
                    returned_at_ms: outcome.returned_at_ms,
                });
                let validation = self
                    .queue
                    .report_result(&outcome.instance_id, record, now_unix_ms());
                if let Ok(ValidationOutcome::DisputeOpened { nodes }) = validation {
                    GridDispatcher::referee_dispute(
                        &mut self.queue,
                        &mut self.traces,
                        &mut self.disputes,
                        &self.blobs,
                        &outcome.unit_id,
                        &nodes,
                        self.workers,
                        self.live_workers,
                        self.local,
                        self.config,
                    )
                    .await;
                    // The outputs in circulation are the winner's, not whichever disputant
                    // returned last.
                    if let Some(record) = self
                        .queue
                        .status(&outcome.unit_id)
                        .and_then(|s| s.validated.as_ref())
                        && let Some(winner_blobs) = self.returned_blobs.get(&record.instance_id)
                    {
                        for (id, bytes) in winner_blobs {
                            self.blobs.insert(id.clone(), bytes.clone());
                        }
                    }
                }
            }
            Ok(payload) => {
                let error = payload
                    .error_message()
                    .unwrap_or("worker reported failure without a message")
                    .to_string();
                self.traces.push(GridDispatchTrace {
                    unit_id: outcome.unit_id.clone(),
                    instance_id: outcome.instance_id.clone(),
                    node_id: outcome.node_id.clone(),
                    app: outcome.app,
                    transport: outcome.transport,
                    wall_ms: outcome.wall_ms,
                    exec_ms: 0.0,
                    digest_fnv1a: 0,
                    layer_digests: Vec::new(),
                    check_value: 0.0,
                    ok: false,
                    error: Some(error.clone()),
                    dispatched_at_ms: outcome.dispatched_at_ms,
                    returned_at_ms: outcome.returned_at_ms,
                });
                let _ = self
                    .queue
                    .report_failure(&outcome.instance_id, &error, now_unix_ms());
            }
            Err(error) => {
                self.traces.push(GridDispatchTrace {
                    unit_id: outcome.unit_id.clone(),
                    instance_id: outcome.instance_id.clone(),
                    node_id: outcome.node_id.clone(),
                    app: outcome.app,
                    transport: outcome.transport,
                    wall_ms: outcome.wall_ms,
                    exec_ms: 0.0,
                    digest_fnv1a: 0,
                    layer_digests: Vec::new(),
                    check_value: 0.0,
                    ok: false,
                    error: Some(error.clone()),
                    dispatched_at_ms: outcome.dispatched_at_ms,
                    returned_at_ms: outcome.returned_at_ms,
                });
                let _ = self
                    .queue
                    .report_failure(&outcome.instance_id, &error, now_unix_ms());
            }
        }
        // Validated results become compute samples on their workers, so the next dispatch
        // in this run and every later run rank by measured completion.
        self.compute_samples += self.queue.feed_workers(self.workers, now_unix_ms());
        self.queue.tick(now_unix_ms(), self.workers);
        true
    }

    /// Dispatches and settles until nothing is in flight and nothing can be assigned.
    pub async fn run_until_settled(&mut self) {
        loop {
            let progressed = self.dispatch_ready() > 0;
            if self.in_flight.is_empty() {
                // Nothing is running. Failures handled inline may have queued retries, so go
                // round again; when nothing could be assigned either, the run is finished or
                // stuck.
                if progressed {
                    continue;
                }
                break;
            }
            if !self.settle_one().await {
                break;
            }
        }
    }

    /// Dispatches and settles until every unit of `graph` has validated or failed, keeping
    /// every other graph in the queue moving meanwhile. Returns false when the graph cannot
    /// settle because nothing is running and nothing can be assigned.
    pub async fn run_until_graph_settled(&mut self, graph: &WorkGraph) -> bool {
        loop {
            let progressed = self.dispatch_ready() > 0;
            if self.graph_settled(graph) {
                return true;
            }
            if self.in_flight.is_empty() {
                if progressed {
                    continue;
                }
                return false;
            }
            if !self.settle_one().await {
                return self.graph_settled(graph);
            }
        }
    }

    /// The report for one graph in the runner: its traces, events, disputes, outputs, accesses
    /// and credits, and a wall time from its first dispatch to its last return.
    pub fn report_for(&self, graph: &WorkGraph) -> GridRunReport {
        self.report_over(
            &graph.id,
            graph.units().map(|u| u.id.clone()).collect(),
            graph.len(),
        )
    }

    /// One report over every graph the runner ran, under `graph_id`.
    pub fn report_all(&self, graph_id: &str) -> GridRunReport {
        let ids: Vec<WorkUnitId> = self
            .graphs
            .iter()
            .flat_map(|g| g.units().map(|u| u.id.clone()))
            .collect();
        let total = ids.len();
        self.report_over(graph_id, ids, total)
    }

    fn report_over(
        &self,
        graph_id: &str,
        ids: Vec<WorkUnitId>,
        units_total: usize,
    ) -> GridRunReport {
        let id_set: std::collections::BTreeSet<&WorkUnitId> = ids.iter().collect();
        let in_set = |id: &WorkUnitId| id_set.contains(id);
        let statuses: Vec<&kadmium_grid::UnitStatus> =
            ids.iter().filter_map(|id| self.queue.status(id)).collect();
        let units_validated = statuses.iter().filter(|s| s.validated.is_some()).count();
        let units_failed = statuses
            .iter()
            .filter(|s| s.validated.is_none() && s.state.is_terminal())
            .count();
        let traces: Vec<GridDispatchTrace> = self
            .traces
            .iter()
            .filter(|t| in_set(&t.unit_id))
            .cloned()
            .collect();
        let events: Vec<SchedulerEventRecord> = self
            .queue
            .events()
            .iter()
            .filter(|e| event_unit_id(&e.event).is_some_and(|u| in_set(&u)))
            .cloned()
            .collect();
        let disputes: Vec<VerdeDisputeReport> = self
            .disputes
            .iter()
            .filter(|d| in_set(&d.unit_id))
            .cloned()
            .collect();

        // Outputs of the sink units: the units nothing in their graph depends on.
        let mut outputs = Vec::new();
        for graph in &self.graphs {
            for unit in graph.sinks() {
                if !in_set(&unit.id) {
                    continue;
                }
                if let Some(status) = self.queue.status(&unit.id)
                    && let Some(record) = &status.validated
                {
                    for out in &record.outputs {
                        if let Some(bytes) = self.blobs.get(&out.content_id) {
                            outputs.push((out.content_id.clone(), bytes.clone()));
                        }
                    }
                }
            }
        }

        // What the validated units touched, summed per node, kind of work and content.
        let mut access_sums: BTreeMap<(NodeId, String, String), u64> = BTreeMap::new();
        let mut compute_samples = 0usize;
        for status in &statuses {
            if let Some(record) = &status.validated {
                if record.exec_ms > 0.0 && record.exec_ms.is_finite() {
                    compute_samples += 1;
                }
                for access in &record.accessed {
                    *access_sums
                        .entry((
                            record.node_id.clone(),
                            status.unit.app.label(),
                            access.content_id.clone(),
                        ))
                        .or_default() += access.hits;
                }
            }
        }
        let accesses: Vec<RecordedAccess> = access_sums
            .into_iter()
            .map(|((node, app, content_id), hits)| RecordedAccess {
                node,
                app,
                content_id,
                hits,
            })
            .collect();

        // Books: credit for every validated unit, a penalty for every invalidated instance.
        let mut ledger =
            CreditLedger::new(self.config.credit_policy.clone().unwrap_or_default());
        if self.config.credit_policy.is_some() {
            let now = now_unix_ms();
            for status in &statuses {
                if let Some(record) = &status.validated {
                    ledger.credit_validated(&status.unit, record, now);
                }
            }
            for event in &events {
                if let SchedulerEvent::Invalidated { node, .. } = &event.event {
                    ledger.penalize_invalid(node, now);
                }
            }
        }

        let first = traces
            .iter()
            .map(|t| t.dispatched_at_ms)
            .fold(f64::INFINITY, f64::min);
        let last = traces
            .iter()
            .map(|t| t.returned_at_ms)
            .fold(f64::NEG_INFINITY, f64::max);
        let wall_ms = if first.is_finite() && last.is_finite() && last >= first {
            ((last - first) * 1000.0).round() / 1000.0
        } else {
            0.0
        };

        GridRunReport {
            graph_id: graph_id.to_string(),
            units_total,
            units_validated,
            units_failed,
            success: units_total > 0 && units_validated == units_total,
            wall_ms,
            traces,
            events,
            disputes,
            compute_samples,
            accesses,
            outputs,
            credits: ledger.entries().to_vec(),
            receipts: ledger.receipts(),
        }
    }
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
    /// that depend on them. One graph, run to the end: [`GridRunner`] is the same loop kept open
    /// across several graphs.
    pub async fn run_graph(
        graph: &WorkGraph,
        workers: &mut [WorkerProfile],
        catalog: &ReplicaCatalog,
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        local: Option<&LocalExecutor>,
        config: &GridRunConfig,
    ) -> GridRunReport {
        let mut runner = GridRunner::new(workers, catalog, live_workers, local, config);
        if let Err(e) = runner.submit(graph, &[], Vec::new()) {
            return runner.report_for(graph).with_error_trace(e.to_string());
        }
        runner.run_until_settled().await;
        runner.report_for(graph)
    }

    /// Where an action for `node` runs: its live worker, else the in-process executor.
    fn target_for(
        node: &NodeId,
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        local: Option<&LocalExecutor>,
    ) -> Option<RefereeTarget> {
        if let Some(reg) = live_workers.get(&node.0).filter(|w| w.online) {
            return Some(RefereeTarget::Worker {
                node: node.clone(),
                addr: reg.worker_tcp_addr.clone(),
            });
        }
        local.map(|executor| RefereeTarget::Local {
            node: node.clone(),
            executor: executor.clone(),
        })
    }

    /// The referee: a TrustedCore worker that is online and was not a disputant, else the
    /// in-process executor.
    fn referee_for(
        disputants: &[NodeId],
        workers: &[WorkerProfile],
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        local: Option<&LocalExecutor>,
    ) -> Option<RefereeTarget> {
        let worker = workers.iter().find(|w| {
            w.online
                && w.node.trust_tier.can_host_boundary_ingress_egress()
                && !disputants.contains(w.id())
                && live_workers.get(&w.id().0).is_some_and(|r| r.online)
        });
        if let Some(worker) = worker {
            return Some(RefereeTarget::Worker {
                node: worker.id().clone(),
                addr: live_workers[&worker.id().0].worker_tcp_addr.clone(),
            });
        }
        local.map(|executor| RefereeTarget::Local {
            node: NodeId(executor.ctx.node_id.clone()),
            executor: executor.clone(),
        })
    }

    /// Sends a dispatch to a target and waits for the answer. Returns the transport label, the
    /// result and the wall time.
    async fn dispatch_to(
        target: &RefereeTarget,
        payload: WorkDispatchPayload,
        config: &GridRunConfig,
        label: &str,
    ) -> (String, Result<WorkResultPayload, String>, f64) {
        let t0 = Instant::now();
        match target {
            RefereeTarget::Worker { addr, .. } => {
                let result = DistributedLanExecutor::channel_pool()
                    .call(
                        addr,
                        WireFrameKind::WorkDispatch,
                        config.epoch as u32,
                        payload.to_le_bytes(),
                        config.call_timeout,
                    )
                    .await
                    .and_then(|frame| WorkResultPayload::from_le_bytes(&frame.payload));
                (format!("{label} {addr}"), result, elapsed_ms(t0))
            }
            RefereeTarget::Local { executor, .. } => {
                let registry = executor.registry.clone();
                let ctx = executor.ctx.clone();
                let result =
                    tokio::task::spawn_blocking(move || execute_dispatch(payload, &registry, &ctx))
                        .await
                        .map_err(|e| format!("{label} task failed: {e}"));
                (format!("{label} in-process"), result, elapsed_ms(t0))
            }
        }
    }

    /// Dispatches a Verde probe of `unit` to a target.
    async fn dispatch_probe(
        target: &RefereeTarget,
        unit: &WorkUnit,
        probe: &VerdeProbe,
        blobs: WorkBlobs,
        config: &GridRunConfig,
        label: &str,
    ) -> (InstanceId, String, Result<WorkResultPayload, String>, f64) {
        let instance_id = InstanceId(format!(
            "{}#{label}-L{}-L{}",
            unit.id, probe.from_layer, probe.to_layer
        ));
        let payload =
            WorkDispatchPayload::new(unit.clone(), instance_id.clone(), config.epoch, blobs)
                .with_probe(probe.clone());
        let (transport, result, wall_ms) = Self::dispatch_to(target, payload, config, label).await;
        (
            instance_id,
            format!(
                "{transport}: layers {}..={}",
                probe.from_layer, probe.to_layer
            ),
            result,
            wall_ms,
        )
    }

    /// Settles a Verde dispute.
    ///
    /// When the disputants' results carry digest chains that diverge, the referee finds the
    /// first divergent layer by comparing the chains (Verde's first phase: comparisons, not
    /// re-execution), obtains that layer's input as an opening of the boundary before it from
    /// a disputant, checked against the digest both committed to, and re-runs that one layer
    /// through a [`VerdeProbe`]. The disputant whose layer digest it reproduces wins; the loser
    /// is penalised with the layer named. When no opening verifies, the referee reconstructs
    /// the input itself from the unit's input. When the chains do not diverge (a one-step unit,
    /// or a result whose digest disagrees while its chain does not), or the executor refuses a
    /// probe, the referee re-runs the whole unit and sides with the digest it reproduces. The
    /// referee is a TrustedCore worker that was not a disputant, else the in-process executor;
    /// with neither, or when the referee fails, the unit is cancelled. When the referee
    /// reproduces no disputant, both are penalised and the unit retries elsewhere.
    #[allow(clippy::too_many_arguments)]
    async fn referee_dispute(
        queue: &mut WorkQueue,
        traces: &mut Vec<GridDispatchTrace>,
        disputes: &mut Vec<VerdeDisputeReport>,
        blobs: &BTreeMap<String, Vec<u8>>,
        unit_id: &WorkUnitId,
        disputants: &[NodeId],
        workers: &[WorkerProfile],
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        local: Option<&LocalExecutor>,
        config: &GridRunConfig,
    ) {
        let t0 = Instant::now();
        let Some(status) = queue.status(unit_id).cloned() else {
            return;
        };
        let unit = status.unit;
        let records = status.returned;
        let app = unit.app.label();
        let inputs: WorkBlobs = unit
            .inputs
            .iter()
            .filter(|c| c.kind == ContentKind::Activation)
            .filter_map(|c| {
                blobs
                    .get(&c.content_id)
                    .map(|b| (c.content_id.clone(), b.clone()))
            })
            .collect();
        let mut report = VerdeDisputeReport {
            unit_id: unit_id.clone(),
            app: app.clone(),
            disputants: disputants.to_vec(),
            referee: None,
            method: "whole-unit".to_string(),
            layers_in_unit: records
                .iter()
                .map(|r| r.layer_digests.len())
                .max()
                .unwrap_or(0)
                .max(1) as u32,
            digest_comparisons: 0,
            openings_requested: 0,
            openings_verified: 0,
            divergent_layer: None,
            referee_layers_run: 0,
            winner: None,
            losers: Vec::new(),
            outcome: String::new(),
            wall_ms: 0.0,
        };
        let first = records.first().cloned();
        let other = first.as_ref().and_then(|a| {
            records
                .iter()
                .find(|r| r.digest_fnv1a != a.digest_fnv1a)
                .cloned()
        });
        let Some((a, b)) = first.zip(other) else {
            report.outcome = "no two returned results disagree; unit cancelled".to_string();
            let _ = queue.cancel(unit_id, now_unix_ms());
            report.wall_ms = elapsed_ms(t0);
            disputes.push(report);
            return;
        };
        let Some(referee) = Self::referee_for(disputants, workers, live_workers, local) else {
            report.outcome = "no trusted referee is online; unit cancelled".to_string();
            let _ = queue.cancel(unit_id, now_unix_ms());
            report.wall_ms = elapsed_ms(t0);
            disputes.push(report);
            return;
        };
        report.referee = Some(referee.node().clone());

        let mut verdict = Verdict::Undecided;
        if let Some(index) = a.first_divergent_index(&b) {
            report.method = "layer-bisection".to_string();
            report.digest_comparisons = (index + 1) as u32;
            let layer = a.layer_digests[index].layer;
            report.divergent_layer = Some(layer);
            let first_layer = a.layer_digests[0].layer;
            let mut probe_blobs = inputs.clone();
            let mut state_in: Option<Vec<u8>> = None;
            if index > 0 {
                // The divergent layer's input is the boundary both disputants committed to
                // at the layer before. Either may open it; the opening is checked against the
                // agreed digest, so an opening that lies is caught here, not later.
                let agreed = a.layer_digests[index - 1];
                let state_id = VerdeProbe::state_id(unit_id, agreed.layer);
                let opening = VerdeProbe {
                    from_layer: first_layer,
                    to_layer: agreed.layer,
                    state_in: None,
                    want_state_out: true,
                };
                for record in [&a, &b] {
                    let Some(target) = Self::target_for(&record.node_id, live_workers, local)
                    else {
                        continue;
                    };
                    report.openings_requested += 1;
                    let (instance_id, transport, outcome, wall_ms) = Self::dispatch_probe(
                        &target,
                        &unit,
                        &opening,
                        inputs.clone(),
                        config,
                        "verde-opening",
                    )
                    .await;
                    traces.push(referee_trace(
                        unit_id,
                        instance_id,
                        target.node().clone(),
                        &app,
                        transport,
                        wall_ms,
                        &outcome,
                    ms_since(config.clock_origin),
                    ));
                    if let Ok(payload) = &outcome
                        && payload.header.ok
                        && payload
                            .header
                            .result
                            .as_ref()
                            .and_then(|r| r.layer_digests.last())
                            == Some(&agreed)
                        && let Some((_, bytes)) =
                            payload.blobs.iter().find(|(id, _)| *id == state_id)
                    {
                        report.openings_verified += 1;
                        state_in = Some(bytes.clone());
                        break;
                    }
                }
                if state_in.is_none() {
                    // No disputant opened the boundary: the referee reconstructs it from the
                    // unit's input, which costs it the layers before the divergent one.
                    let rebuild = VerdeProbe {
                        from_layer: first_layer,
                        to_layer: agreed.layer,
                        state_in: None,
                        want_state_out: true,
                    };
                    report.referee_layers_run += index as u32;
                    let (instance_id, transport, outcome, wall_ms) = Self::dispatch_probe(
                        &referee,
                        &unit,
                        &rebuild,
                        inputs.clone(),
                        config,
                        "referee",
                    )
                    .await;
                    traces.push(referee_trace(
                        unit_id,
                        instance_id,
                        referee.node().clone(),
                        &app,
                        transport,
                        wall_ms,
                        &outcome,
                    ms_since(config.clock_origin),
                    ));
                    match outcome {
                        Ok(payload) if payload.header.ok => {
                            state_in = payload
                                .blobs
                                .into_iter()
                                .find(|(id, _)| *id == state_id)
                                .map(|(_, bytes)| bytes);
                            if state_in.is_none() {
                                verdict = Verdict::Failed(
                                    "referee returned no boundary for the divergent layer"
                                        .to_string(),
                                );
                            }
                        }
                        Ok(payload) => verdict = Verdict::from_failure(payload.error_message()),
                        Err(e) => verdict = Verdict::Failed(e),
                    }
                }
                if let Some(bytes) = &state_in {
                    probe_blobs.push((state_id, bytes.clone()));
                }
            }
            if matches!(verdict, Verdict::Undecided) {
                // The referee runs the divergent layer alone, from the verified boundary.
                let probe = VerdeProbe {
                    from_layer: layer,
                    to_layer: layer,
                    state_in: (index > 0)
                        .then(|| VerdeProbe::state_id(unit_id, a.layer_digests[index - 1].layer)),
                    want_state_out: false,
                };
                report.referee_layers_run += 1;
                let (instance_id, transport, outcome, wall_ms) =
                    Self::dispatch_probe(&referee, &unit, &probe, probe_blobs, config, "referee")
                        .await;
                traces.push(referee_trace(
                    unit_id,
                    instance_id,
                    referee.node().clone(),
                    &app,
                    transport,
                    wall_ms,
                    &outcome,
                ms_since(config.clock_origin),
                ));
                verdict = match outcome {
                    Ok(payload) if payload.header.ok => {
                        let reproduced = payload
                            .header
                            .result
                            .as_ref()
                            .and_then(|r| r.layer_digests.first())
                            .map(|d| d.digest);
                        let winner = reproduced.and_then(|digest| {
                            records.iter().find(|r| {
                                r.layer_digests
                                    .get(index)
                                    .is_some_and(|d| d.digest == digest)
                            })
                        });
                        match winner {
                            Some(w) => Verdict::Winner(
                                w.node_id.clone(),
                                format!(
                                    "Verde: layer {layer} digest not reproduced by referee `{}`",
                                    referee.node()
                                ),
                            ),
                            None => Verdict::Neither(format!(
                                "Verde: layer {layer} digest reproduced for no disputant by referee `{}`",
                                referee.node()
                            )),
                        }
                    }
                    Ok(payload) => Verdict::from_failure(payload.error_message()),
                    Err(e) => Verdict::Failed(e),
                };
            }
        }

        if matches!(verdict, Verdict::Undecided | Verdict::NotBisectable) {
            // The whole unit again on the referee: the disputant whose digest it reproduces wins.
            report.method = "whole-unit".to_string();
            report.referee_layers_run += report.layers_in_unit;
            let instance_id = InstanceId(format!("{unit_id}#referee"));
            let payload = WorkDispatchPayload::new(
                unit.clone(),
                instance_id.clone(),
                config.epoch,
                inputs.clone(),
            );
            let (transport, outcome, wall_ms) =
                Self::dispatch_to(&referee, payload, config, "referee").await;
            traces.push(referee_trace(
                unit_id,
                instance_id,
                referee.node().clone(),
                &app,
                transport,
                wall_ms,
                &outcome,
            ms_since(config.clock_origin),
            ));
            verdict = match outcome {
                Ok(payload) if payload.header.ok => {
                    let digest = payload.header.result.as_ref().map(|r| r.digest_fnv1a);
                    match digest.and_then(|d| records.iter().find(|r| r.digest_fnv1a == d)) {
                        Some(w) => Verdict::Winner(
                            w.node_id.clone(),
                            format!(
                                "Verde: result digest not reproduced by referee `{}`",
                                referee.node()
                            ),
                        ),
                        None => Verdict::Neither(format!(
                            "Verde: result digest reproduced for no disputant by referee `{}`",
                            referee.node()
                        )),
                    }
                }
                Ok(payload) => Verdict::Failed(
                    payload
                        .error_message()
                        .unwrap_or("referee reported failure without a message")
                        .to_string(),
                ),
                Err(e) => Verdict::Failed(e),
            };
        }

        let now = now_unix_ms();
        match verdict {
            Verdict::Winner(node, reason) => {
                let _ = queue.resolve_dispute_with_reason(unit_id, &node, &reason, now);
                report.losers = disputants.iter().filter(|d| **d != node).cloned().collect();
                report.winner = Some(node);
                report.outcome = reason;
            }
            Verdict::Neither(reason) => {
                let _ = queue.reject_dispute(unit_id, &reason, now);
                report.losers = disputants.to_vec();
                report.outcome = reason;
            }
            Verdict::Failed(error) => {
                let _ = queue.cancel(unit_id, now);
                report.outcome = format!("referee failed: {error}; unit cancelled");
            }
            Verdict::Undecided | Verdict::NotBisectable => {
                let _ = queue.cancel(unit_id, now);
                report.outcome = "referee reached no verdict; unit cancelled".to_string();
            }
        }
        report.wall_ms = elapsed_ms(t0);
        disputes.push(report);
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
            layer_digests: Vec::new(),
            check_value: 0.0,
            ok: false,
            error: Some(error),
            dispatched_at_ms: 0.0,
            returned_at_ms: 0.0,
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
    use kadmium_grid::{CostEstimate, RedundancyPolicy, fnv1a_64};
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
                layer_digests: Vec::new(),
                check_value: reversed.len() as f64,
                units_processed: 1,
                accessed: Vec::new(),
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
        assert_eq!(local.registry.kinds().len(), 11);

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
            WorkUnit::new("x", AppKind::Checksum),
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
            let unit_id = WorkUnitId::qualified(&graph.id, &format!("stage-{i:02}"));
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

#[cfg(test)]
mod fanout_tests {
    use super::*;
    use crate::{DEFAULT_INITIAL_KTB_MERKLE_ROOT, LiveNodeWorkerServer, prompt_seed_hidden};
    use kadmium_core::{HardwareNode, ModelSpec, sample_personal_cluster};
    use kadmium_grid::fnv1a_64;
    use kadmium_sharding::{
        ExecutionStrategy, ExpertPlacementPlanner, ExpertPopularity, StepGraphBuilder,
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

    fn toy_moe() -> ModelSpec {
        ModelSpec {
            id: "toy-moe".into(),
            display_name: "Toy MoE".into(),
            total_params_billions: 4.0,
            num_layers: 4,
            is_moe: true,
            dense_layer_size_gb: 0.05,
            experts_per_layer: 8,
            active_experts_per_token: 2,
            single_expert_size_gb: 0.01,
            kv_cache_per_layer_gb: 0.01,
            training_activation_per_layer_gb: 0.02,
            lora_state_per_layer_gb: 0.01,
        }
    }

    async fn run_shape(
        strategy: ExecutionStrategy,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        seed: &TokenBatchInt8,
        session: &str,
    ) -> (GridRunReport, Vec<u8>) {
        run_shape_with_heat(strategy, model, nodes, live_workers, seed, session, None).await
    }

    /// [`run_shape`] on a placement planned from `heat`, so the expert-parallel shape carries
    /// the balancer's copies and a wide batch deals rows across them.
    async fn run_shape_with_heat(
        strategy: ExecutionStrategy,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        seed: &TokenBatchInt8,
        session: &str,
        heat: Option<&ExpertPopularity>,
    ) -> (GridRunReport, Vec<u8>) {
        let placement = ExpertPlacementPlanner::plan(model, nodes, strategy, heat);
        let graph = StepGraphBuilder::decode_step(model, &placement, nodes, 1, session, seed.rows);
        let catalog = ReplicaCatalog::assume_placed(&placement.rules, 1);
        let mut workers = GridDispatcher::workers_from_registrations(live_workers);
        let config = GridRunConfig::default()
            .with_seed(StepGraphBuilder::prompt_ref(&graph.id), seed.to_le_bytes());
        let report =
            GridDispatcher::run_graph(&graph, &mut workers, &catalog, live_workers, None, &config)
                .await;
        assert!(
            report.success,
            "{strategy:?} run failed: {:?}",
            report.traces
        );
        let last_unit = match strategy {
            ExecutionStrategy::PipelineChain => {
                WorkUnitId::new(format!("stage-{:02}", placement.trunk_ranges.len() - 1))
            }
            ExecutionStrategy::ExpertParallel => {
                WorkUnitId::new(format!("L{:03}-combine", model.num_layers - 1))
            }
        };
        let out_ref = StepGraphBuilder::output_ref(&graph.id, &last_unit);
        let bytes = report
            .output(&out_ref.content_id)
            .expect("final output present")
            .to_vec();
        (report, bytes)
    }

    #[tokio::test]
    async fn replica_served_fan_out_matches_the_chain_bytes_and_reports_routing() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            nodes[0].id.0.clone(),
            registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        let model = toy_moe();
        let seed =
            TokenBatchInt8::quantize(&prompt_seed_hidden("replica serving", 64), 1, 64, 16)
                .unwrap();
        let (chain_report, chain_bytes) = run_shape(
            ExecutionStrategy::PipelineChain,
            &model,
            &nodes,
            &live_workers,
            &seed,
            "replica-chain",
        )
        .await;
        // The chain's stage units report which experts every layer routed to.
        let routed: Vec<&RecordedAccess> = chain_report
            .accesses
            .iter()
            .filter(|a| a.app == "StageForward")
            .collect();
        assert!(!routed.is_empty());
        assert!(
            routed
                .iter()
                .all(|a| ModelContentNames::parse_expert(&a.content_id)
                    .is_some_and(|(m, _, _)| m == model.id))
        );
        let routed_hits: u64 = routed.iter().map(|a| a.hits).sum();
        assert_eq!(
            routed_hits,
            (model.num_layers * model.active_experts_per_token) as u64,
            "one row routes top-k experts on every layer"
        );

        // A placement with replicated hot experts, served from the replica rather than the
        // primary: the same bytes come out of the last combine.
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 0.5);
        let placement = ExpertPlacementPlanner::plan(
            &model,
            &nodes,
            ExecutionStrategy::ExpertParallel,
            Some(&heat),
        );
        let replicated = placement
            .experts
            .iter()
            .find(|h| h.holders.len() > 1)
            .expect("a replicated expert")
            .clone();
        let replica = replicated.holders[1].clone();
        let serving = placement
            .serving_assignment(|n| Some(if *n == replica { 1.0 } else { 9.0 }));
        assert!(serving.contains_key(&(replicated.layer, replicated.expert)));
        let graph = StepGraphBuilder::decode_step_served(
            &model,
            &placement,
            &nodes,
            1,
            "replica-ep",
            1,
            &serving,
        );
        let catalog = ReplicaCatalog::assume_placed(&placement.rules, 1);
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let config = GridRunConfig::default()
            .with_seed(StepGraphBuilder::prompt_ref(&graph.id), seed.to_le_bytes());
        let report =
            GridDispatcher::run_graph(&graph, &mut workers, &catalog, &live_workers, None, &config)
                .await;
        assert!(report.success, "{:?}", report.traces);
        let out_ref = StepGraphBuilder::output_ref(
            &graph.id,
            &WorkUnitId::new(format!("L{:03}-combine", model.num_layers - 1)),
        );
        assert_eq!(
            report.output(&out_ref.content_id).expect("final output"),
            chain_bytes.as_slice(),
            "the replica serves the same bytes as the primary would"
        );
        let leaf = WorkUnitId::qualified(
            &graph.id,
            &format!("L{:03}-experts-{}", replicated.layer, replica.0),
        );
        let trace = report.trace_for(&leaf).expect("the replica ran a slice unit");
        assert!(trace.ok);
        assert_eq!(trace.node_id, replica);
        assert!(
            report
                .accesses
                .iter()
                .any(|a| a.app == "TrunkForward" && a.hits > 0)
        );
        // Every expert the trunks routed to was served by exactly one holder.
        let routed_total: u64 = report
            .accesses
            .iter()
            .filter(|a| a.app == "TrunkForward")
            .map(|a| a.hits)
            .sum();
        let served_total: u64 = report
            .accesses
            .iter()
            .filter(|a| a.app == "ExpertSlice")
            .map(|a| a.hits)
            .sum();
        assert_eq!(served_total, routed_total);
        assert!(
            report
                .accesses
                .iter()
                .all(|a| ModelContentNames::parse_expert(&a.content_id).is_some())
        );
    }

    #[tokio::test]
    async fn fan_out_graph_matches_the_chain_graph_byte_for_byte() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            nodes[0].id.0.clone(),
            registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        let model = toy_moe();

        // One row.
        let seed =
            TokenBatchInt8::quantize(&prompt_seed_hidden("fan-out equivalence", 64), 1, 64, 16)
                .unwrap();
        let (chain, chain_bytes) = run_shape(
            ExecutionStrategy::PipelineChain,
            &model,
            &nodes,
            &live_workers,
            &seed,
            "one-row",
        )
        .await;
        let (fan_out, fan_out_bytes) = run_shape(
            ExecutionStrategy::ExpertParallel,
            &model,
            &nodes,
            &live_workers,
            &seed,
            "one-row",
        )
        .await;
        assert_eq!(
            chain_bytes, fan_out_bytes,
            "layer-3 boundary differs between shapes"
        );
        assert_eq!(fnv1a_64(&chain_bytes), fnv1a_64(&fan_out_bytes));
        assert!(fan_out.units_total > chain.units_total);
        let slice_nodes: std::collections::BTreeSet<_> = fan_out
            .traces
            .iter()
            .filter(|t| t.app == "ExpertSlice")
            .map(|t| t.node_id.clone())
            .collect();
        assert!(!slice_nodes.is_empty(), "no expert slices ran");
        assert!(
            fan_out
                .traces
                .iter()
                .all(|t| t.transport.starts_with("pooled channel")),
            "every unit crossed a real channel"
        );
        let decoded = TokenBatchInt8::from_le_bytes(&fan_out_bytes).unwrap();
        assert_eq!((decoded.rows, decoded.dim), (1, 64));

        // Two rows in one step: the batch path must agree too.
        let rows: Vec<f32> = [
            prompt_seed_hidden("row one", 64),
            prompt_seed_hidden("row two", 64),
        ]
        .concat();
        let seed2 = TokenBatchInt8::quantize(&rows, 2, 64, 16).unwrap();
        let (_, chain2) = run_shape(
            ExecutionStrategy::PipelineChain,
            &model,
            &nodes,
            &live_workers,
            &seed2,
            "two-rows",
        )
        .await;
        let (_, fan_out2) = run_shape(
            ExecutionStrategy::ExpertParallel,
            &model,
            &nodes,
            &live_workers,
            &seed2,
            "two-rows",
        )
        .await;
        assert_eq!(chain2, fan_out2, "two-row batch differs between shapes");
        assert_eq!(TokenBatchInt8::from_le_bytes(&fan_out2).unwrap().rows, 2);
        assert_ne!(chain2, chain_bytes);
    }

    #[tokio::test]
    async fn every_layer_boundary_carries_the_same_digest_in_both_shapes() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            nodes[0].id.0.clone(),
            registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        let model = toy_moe();

        for prompts in [vec!["layer digests"], vec!["row one", "row two"]] {
            let values: Vec<f32> = prompts
                .iter()
                .flat_map(|p| prompt_seed_hidden(p, 64))
                .collect();
            let seed = TokenBatchInt8::quantize(&values, prompts.len() as u32, 64, 16).unwrap();
            let (chain, _) = run_shape(
                ExecutionStrategy::PipelineChain,
                &model,
                &nodes,
                &live_workers,
                &seed,
                "digests-chain",
            )
            .await;
            let (fan_out, _) = run_shape(
                ExecutionStrategy::ExpertParallel,
                &model,
                &nodes,
                &live_workers,
                &seed,
                "digests-ep",
            )
            .await;

            // The chain's stage results carry one digest per layer, in layer order.
            let mut chain_layers: Vec<LayerDigest> = Vec::new();
            for stage in 0..8 {
                let id = WorkUnitId::qualified(&chain.graph_id, &format!("stage-{stage:02}"));
                match chain.trace_for(&id) {
                    Some(trace) => chain_layers.extend(trace.layer_digests.iter().copied()),
                    None => break,
                }
            }
            assert_eq!(chain_layers.len(), model.num_layers as usize, "{prompts:?}");
            for (i, d) in chain_layers.iter().enumerate() {
                assert_eq!(d.layer, i as u32);
            }
            // The fan-out's combine unit for each layer produced the same boundary digest.
            for d in &chain_layers {
                let combine = fan_out
                    .trace_for(&WorkUnitId::qualified(
                        &fan_out.graph_id,
                        &format!("L{:03}-combine", d.layer),
                    ))
                    .expect("combine trace");
                assert_eq!(
                    combine.digest_fnv1a, d.digest,
                    "{prompts:?}: layer {} boundary digest differs between shapes",
                    d.layer
                );
                assert_eq!(combine.layer_digests, vec![*d]);
            }
            // A stage's digest is the fold of its layer digests under the manifest.
            let stage = chain
                .trace_for(&WorkUnitId::qualified(&chain.graph_id, "stage-00"))
                .unwrap();
            assert_eq!(
                stage.digest_fnv1a,
                LayerReference::unit_digest(&stage.layer_digests)
            );
            assert_ne!(stage.digest_fnv1a, LayerReference::unit_digest(&[]));
        }
    }

    /// DeepInfer's batch invariance as a property over real TCP workers: a row decoded alone
    /// produces the same bytes as the same row inside any batch of other rows, at any position,
    /// in both shapes. The reference math runs rows independently; what the property checks is
    /// that nothing on the way in, between units or on the way out re-quantizes a row because
    /// of its neighbours.
    #[tokio::test]
    async fn a_row_decodes_to_the_same_bytes_alone_and_inside_any_batch() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live_workers = BTreeMap::new();
        live_workers.insert(
            nodes[0].id.0.clone(),
            registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
        );
        live_workers.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        let model = toy_moe();
        let rows: Vec<Vec<f32>> = ["row a", "row b", "row c", "row d"]
            .iter()
            .map(|p| prompt_seed_hidden(p, 64))
            .collect();
        let batches: [&[usize]; 6] = [
            &[0, 1],
            &[1, 0],
            &[0, 1, 2],
            &[2, 1, 0],
            &[1, 0, 2, 3],
            &[3, 3, 0, 2],
        ];

        // The third shape is the fan-out over a placement with the balancer's copies, so a
        // batch deals a replicated expert's rows across its holders: the bytes must still be
        // the row's own.
        let heat = ExpertPopularity::power_law(model.num_layers, model.experts_per_layer, 1.0);
        let balanced = ExpertPlacementPlanner::plan(
            &model,
            &nodes,
            ExecutionStrategy::ExpertParallel,
            Some(&heat),
        );
        assert!(balanced.replicated_experts > 0, "the toy mesh has room for copies");
        for (strategy, heat) in [
            (ExecutionStrategy::PipelineChain, None),
            (ExecutionStrategy::ExpertParallel, None),
            (ExecutionStrategy::ExpertParallel, Some(&heat)),
        ] {
            let mut alone: Vec<Vec<u8>> = Vec::new();
            for (i, row) in rows.iter().enumerate() {
                let seed = TokenBatchInt8::quantize(row, 1, 64, 16).unwrap();
                let (_, bytes) = run_shape_with_heat(
                    strategy,
                    &model,
                    &nodes,
                    &live_workers,
                    &seed,
                    &format!("alone-{i}"),
                    heat,
                )
                .await;
                let out = TokenBatchInt8::from_le_bytes(&bytes).unwrap();
                assert_eq!(out.rows, 1);
                alone.push(out.tensor.to_le_bytes());
            }
            assert!(alone.windows(2).all(|w| w[0] != w[1]), "distinct rows decode differently");

            for (b, batch) in batches.iter().enumerate() {
                let values: Vec<f32> = batch
                    .iter()
                    .flat_map(|&i| rows[i].iter().copied())
                    .collect();
                let seed =
                    TokenBatchInt8::quantize(&values, batch.len() as u32, 64, 16).unwrap();
                let (report, bytes) = run_shape_with_heat(
                    strategy,
                    &model,
                    &nodes,
                    &live_workers,
                    &seed,
                    &format!("batch-{b}"),
                    heat,
                )
                .await;
                if heat.is_some() {
                    // Both holders of a replicated expert served some of its rows.
                    let served: u64 = report
                        .accesses
                        .iter()
                        .filter(|a| a.app == "ExpertSlice")
                        .map(|a| a.hits)
                        .sum();
                    let routed: u64 = report
                        .accesses
                        .iter()
                        .filter(|a| a.app == "TrunkForward")
                        .map(|a| a.hits)
                        .sum();
                    assert_eq!(served, routed, "every routed row was served exactly once");
                }
                let out = TokenBatchInt8::from_le_bytes(&bytes).unwrap();
                let per_row = out.row_tensors().expect("rows are whole blocks");
                assert_eq!(per_row.len(), batch.len());
                for (pos, &i) in batch.iter().enumerate() {
                    assert_eq!(
                        per_row[pos].to_le_bytes(),
                        alone[i],
                        "{strategy:?} (heat {}): row {i} at position {pos} of batch {batch:?} differs from the row decoded alone",
                        heat.is_some()
                    );
                }
            }
        }
    }
}

// ================================================================================================
// Training, replay and audit executors
// ================================================================================================

/// Summary of independent stateless stage previews, output of an [`AppKind::DeMoReduce`] unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeMoReduceReport {
    pub step_index: u64,
    pub contributors: usize,
    pub mean_loss_before: f32,
    pub mean_loss_after: f32,
    pub mean_wire_compression_ratio: f64,
    /// Legacy JSON key: norm of concatenated independent stage previews, not replica consensus.
    pub diloco_outer_consensus_norm: f32,
    /// Legacy JSON key: digest of independent synthetic stage previews, not a saved master model.
    pub master_weight_digest_fnv1a: u64,
    #[serde(default)]
    pub execution_mode: String,
}

/// Runs a [`AppKind::TrainForwardBackward`] unit through the same stage training kernel the
/// legacy step uses, on the micro-batch delivered as the unit's input and target blobs.
pub struct TrainForwardBackwardExecutor;

impl WorkExecutor for TrainForwardBackwardExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::TrainForwardBackward
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: kadmium_sharding::TrainStageParams =
            unit.params_json().map_err(|e| e.to_string())?;
        let (_, input_bytes) = activation_input(unit, inputs, |id| id.ends_with("/train-input"))?;
        let (_, target_bytes) = activation_input(unit, inputs, |id| id.ends_with("/train-target"))?;
        let input = TokenBatchInt8::from_le_bytes(input_bytes)?;
        let target = TokenBatchInt8::from_le_bytes(target_bytes)?;
        let request = crate::LiveStageTrainingRequest {
            epoch: params.epoch,
            model_id: params.model_id.clone(),
            stage_index: params.stage_index,
            start_layer: params.start_layer,
            end_layer: params.end_layer,
            step_index: params.step_index,
            micro_batch_id: params.micro_batch_id,
            learning_rate: params.learning_rate,
            boundary_input: input.tensor,
            target_signal: target.tensor,
        };
        let response = crate::execute_stage_training_local(&ctx.node_id, &ctx.hostname, &request);
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                serde_json::to_vec(&response).map_err(|e| e.to_string())?,
            )],
            payload: serde_json::to_vec(&serde_json::json!({
                "loss_before": response.loss_before,
                "loss_after": response.loss_after,
                "wire_compression_ratio": response.wire_compression_ratio,
            }))
            .unwrap_or_default(),
            digest_fnv1a: response.repops_digest_fnv1a,
            layer_digests: Vec::new(),
            check_value: response.loss_after as f64,
            units_processed: input.rows.max(1) as u64,
            accessed: Vec::new(),
        })
    }
}

/// Legacy stateless gradient preview: decompresses each stage's DeMo packet and applies its
/// update to an independent synthetic baseline. Distinct layer ranges are never averaged as
/// replicas. Stateful DiLoCo training sessions use shard-aligned parameter deltas separately.
/// Stage ranges are ordered canonically; RepOps summaries and digests ignore arrival order.
pub struct DeMoReduceExecutor;

impl WorkExecutor for DeMoReduceExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::DeMoReduce
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        _ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: kadmium_sharding::DeMoReduceParams =
            unit.params_json().map_err(|e| e.to_string())?;
        let mut responses: Vec<crate::LiveStageTrainingResponse> = Vec::new();
        for input in unit
            .inputs
            .iter()
            .filter(|c| c.kind == ContentKind::Activation)
        {
            let bytes = inputs
                .blob(&input.content_id)
                .ok_or_else(|| format!("reduce `{}` is missing `{}`", unit.id, input.content_id))?;
            responses.push(serde_json::from_slice(bytes).map_err(|e| e.to_string())?);
        }
        if responses.is_empty() {
            return Err("preview received no stage packets".to_string());
        }
        responses.sort_by_key(|r| (r.start_layer, r.end_layer));
        let losses_before: Vec<f32> = responses.iter().map(|r| r.loss_before).collect();
        let losses_after: Vec<f32> = responses.iter().map(|r| r.loss_after).collect();
        let compressions: Vec<f64> = responses.iter().map(|r| r.wire_compression_ratio).collect();
        let stage_gradients: Vec<Vec<f32>> = responses
            .iter()
            .map(|r| crate::DeMoDctCompressor::decompress_packet(&r.demo_dct_packet))
            .collect();
        let (preview_norm, preview_digest) = crate::scheduler::stateless_stage_gradient_preview(
            &stage_gradients,
            params.outer_lr,
            params.outer_momentum,
        )?;
        let count = responses.len() as f32;
        let report = DeMoReduceReport {
            step_index: params.step_index,
            contributors: responses.len(),
            mean_loss_before: ((repops::tree_sum_f32(&losses_before) / count) * 10_000.0).round()
                / 10_000.0,
            mean_loss_after: ((repops::tree_sum_f32(&losses_after) / count) * 10_000.0).round()
                / 10_000.0,
            mean_wire_compression_ratio: ((repops::tree_sum_f64(&compressions) / count as f64)
                * 10.0)
                .round()
                / 10.0,
            diloco_outer_consensus_norm: (preview_norm * 1000.0).round() / 1000.0,
            master_weight_digest_fnv1a: preview_digest,
            execution_mode: crate::scheduler::STATELESS_STAGE_PREVIEW_MODE.to_string(),
        };
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                serde_json::to_vec(&report).map_err(|e| e.to_string())?,
            )],
            payload: serde_json::to_vec(&report).unwrap_or_default(),
            digest_fnv1a: report.master_weight_digest_fnv1a,
            layer_digests: Vec::new(),
            check_value: preview_norm as f64,
            units_processed: responses.len() as u64,
            accessed: Vec::new(),
        })
    }
}

/// Runs a [`AppKind::DreamReplay`] unit: one offline Dream-RSI replay on this node.
pub struct DreamReplayExecutor;

impl WorkExecutor for DreamReplayExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::DreamReplay
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        _inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: kadmium_sharding::DreamReplayParams =
            unit.params_json().map_err(|e| e.to_string())?;
        let request = crate::LiveDreamRsiReplayRequest {
            replay_id: params.replay_id.clone(),
            epoch: params.epoch,
            world_id: params.world_id.clone(),
            mcts_simulations: params.mcts_simulations,
            puct_exploration_c: params.puct_exploration_c,
            archive_trajectories_count: params.archive_trajectories_count,
        };
        let response = crate::execute_dream_rsi_replay_local(&ctx.node_id, &ctx.hostname, &request);
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                serde_json::to_vec(&response).map_err(|e| e.to_string())?,
            )],
            payload: serde_json::to_vec(&serde_json::json!({
                "world_id": response.world_id,
                "evolved_peak_speedup": response.evolved_peak_speedup,
                "mcts_simulations_run": response.mcts_simulations_run,
            }))
            .unwrap_or_default(),
            digest_fnv1a: response.evolved_policy_hash_fnv64,
            layer_digests: Vec::new(),
            check_value: response.evolved_peak_speedup,
            units_processed: response.mcts_simulations_run as u64,
            accessed: Vec::new(),
        })
    }
}

/// Runs a [`AppKind::VerdeAudit`] unit: a layer-bisection challenge on this node, whose RepOps
/// digest is what redundant copies are compared on. Bisectable: a probe runs any sub-range of
/// the challenge's layers from an opened state. `tamper_at_layer` makes this the executor a
/// dishonest node would run, for tests and demos: it corrupts that layer whatever the unit
/// asked for, and its openings lie the same way its result did.
#[derive(Debug, Clone, Default)]
pub struct VerdeAuditExecutor {
    pub tamper_at_layer: Option<u32>,
}

impl VerdeAuditExecutor {
    fn fault_for(&self, params: &kadmium_sharding::VerdeAuditParams) -> Option<u32> {
        self.tamper_at_layer.or(params.inject_fault_at_layer)
    }
}

impl WorkExecutor for VerdeAuditExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::VerdeAudit
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        _inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: kadmium_sharding::VerdeAuditParams =
            unit.params_json().map_err(|e| e.to_string())?;
        let request = crate::LiveVerdeBisectionRequest {
            audit_id: params.audit_id.clone(),
            epoch: params.epoch,
            start_layer: params.start_layer,
            target_layer: params.target_layer,
            initial_activation: Vec::new(),
            inject_fault_at_layer: self.fault_for(&params),
        };
        let response =
            crate::execute_verde_layer_challenge_local(&ctx.node_id, &ctx.hostname, &request);
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                serde_json::to_vec(&response).map_err(|e| e.to_string())?,
            )],
            payload: serde_json::to_vec(&serde_json::json!({
                "audit_id": response.audit_id,
                "queried_layer": response.queried_layer,
                "repops_tree_sum": response.repops_tree_sum,
            }))
            .unwrap_or_default(),
            digest_fnv1a: response.repops_digest_fnv1a,
            check_value: response.repops_tree_sum,
            units_processed: (params.target_layer.saturating_sub(params.start_layer) + 1) as u64,
            accessed: Vec::new(),
            layer_digests: response.layer_digests,
        })
    }

    fn probe(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        _ctx: &ExecutorContext,
        probe: &VerdeProbe,
    ) -> Result<VerdeProbeOutput, String> {
        let params: kadmium_sharding::VerdeAuditParams =
            unit.params_json().map_err(|e| e.to_string())?;
        if probe.from_layer > probe.to_layer
            || probe.from_layer < params.start_layer
            || probe.to_layer > params.target_layer
        {
            return Err(format!(
                "probe {}..={} is outside the audit's layers {}..={}",
                probe.from_layer, probe.to_layer, params.start_layer, params.target_layer
            ));
        }
        let mut state: Vec<f64> = match &probe.state_in {
            Some(id) => {
                let bytes = inputs
                    .blob(id)
                    .ok_or_else(|| format!("probe is missing the opened state `{id}`"))?;
                if !bytes.len().is_multiple_of(8) {
                    return Err("an opened audit state is a vector of f64".to_string());
                }
                bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect()
            }
            None => {
                if probe.from_layer != params.start_layer {
                    return Err("a probe that does not start at the audit's first layer needs an opened state".to_string());
                }
                VERDE_CHALLENGE_INITIAL_ACTIVATION.to_vec()
            }
        };
        let layer_digests = verde_challenge_layers(
            &mut state,
            probe.from_layer,
            probe.to_layer,
            self.fault_for(&params),
        );
        let state_out = probe
            .want_state_out
            .then(|| state.iter().flat_map(|v| v.to_le_bytes()).collect());
        Ok(VerdeProbeOutput {
            layer_digests,
            state_out,
            layers_run: probe.to_layer - probe.from_layer + 1,
        })
    }
}

#[cfg(test)]
mod phase5_tests {
    use super::*;
    use crate::{DEFAULT_INITIAL_KTB_MERKLE_ROOT, LiveNodeWorkerServer};
    use kadmium_core::{
        HardwareNode, ModelSpec, NodeTrustTier, WorkloadKind, sample_personal_cluster,
    };
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

    async fn two_workers(
        nodes: &[HardwareNode],
    ) -> (
        LiveNodeWorkerServer,
        LiveNodeWorkerServer,
        BTreeMap<String, LiveWorkerRegistration>,
    ) {
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn worker 0");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn worker 1");
        let mut live = BTreeMap::new();
        live.insert(
            nodes[0].id.0.clone(),
            registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
        );
        live.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        (w0, w1, live)
    }

    #[tokio::test]
    async fn training_step_graph_matches_the_stateless_stage_previews() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let (_w0, _w1, live_workers) = two_workers(&nodes).await;
        let model = ModelSpec::llama_3_70b_lora();

        let legacy_plan =
            ShardPlanner::compute_plan(1, &model, WorkloadKind::DistributedTraining, &nodes);
        let legacy = DistributedLanExecutor::execute_live_training_step(
            &legacy_plan,
            7,
            1e-4,
            &live_workers,
        )
        .await;
        assert_eq!(legacy.stages.len(), 2);

        let placement = ExpertPlacementPlanner::plan_for_workload(
            &model,
            &nodes,
            ExecutionStrategy::PipelineChain,
            None,
            WorkloadKind::DistributedTraining,
        );
        let graph =
            StepGraphBuilder::training_step(&model, &placement, &nodes, 1, "train", 7, 1e-4);
        let catalog = ReplicaCatalog::assume_placed(&placement.rules, 1);
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        // The same micro-batch the legacy loop synthesizes for step 7.
        let input: Vec<f32> = (0..32)
            .map(|i| (((i as u64 + 7) as f32) * 0.11).cos() * 0.6)
            .collect();
        let target: Vec<f32> = (0..16)
            .map(|j| (((j * 3 + 1) as f32) * 0.17).sin() * 0.5)
            .collect();
        let config = GridRunConfig::default()
            .with_seed(
                StepGraphBuilder::training_input_ref(&graph.id),
                TokenBatchInt8::quantize(&input, 1, 32, 16)
                    .unwrap()
                    .to_le_bytes(),
            )
            .with_seed(
                StepGraphBuilder::training_target_ref(&graph.id),
                TokenBatchInt8::quantize(&target, 1, 16, 16)
                    .unwrap()
                    .to_le_bytes(),
            );
        let report =
            GridDispatcher::run_graph(&graph, &mut workers, &catalog, &live_workers, None, &config)
                .await;
        assert!(report.success, "{:?}", report.traces);

        for (i, stage) in legacy.stages.iter().enumerate() {
            let trace = report
                .trace_for(&WorkUnitId::qualified(&graph.id, &format!("stage-{i:02}")))
                .expect("stage trace");
            assert_eq!(trace.node_id.0, stage.node_id);
            assert_eq!(
                trace.digest_fnv1a, stage.repops_digest_fnv1a,
                "stage {i} digest"
            );
        }
        let reduce_ref = StepGraphBuilder::output_ref(&graph.id, &WorkUnitId::new("demo-reduce"));
        let reduce: DeMoReduceReport =
            serde_json::from_slice(report.output(&reduce_ref.content_id).unwrap()).unwrap();
        assert_eq!(reduce.contributors, 2);
        assert_eq!(
            reduce.execution_mode,
            crate::scheduler::STATELESS_STAGE_PREVIEW_MODE
        );
        assert_eq!(reduce.execution_mode, legacy.execution_mode);
        assert_eq!(
            reduce.master_weight_digest_fnv1a,
            legacy.stage_preview_digest_fnv1a
        );
        assert!(legacy.preview_error.is_none());
        assert_eq!(reduce.mean_loss_after, legacy.mean_loss_after);
        assert_eq!(
            reduce.diloco_outer_consensus_norm,
            legacy.diloco_outer_consensus_norm
        );
        assert_eq!(report.credits.len(), 3);
        assert!(report.receipts.iter().all(|r| r.earned_usd >= 0.0));
    }

    #[tokio::test]
    async fn replay_batch_spreads_across_workers_and_earns_credit() {
        let cluster = sample_personal_cluster();
        let nodes = vec![cluster[0].clone(), cluster[1].clone()];
        let (_w0, _w1, live_workers) = two_workers(&nodes).await;
        let graph = StepGraphBuilder::replay_batch("dream", 3, "gpu-kernel-world", 6, 32, &nodes);
        let catalog = ReplicaCatalog::new();
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let report = GridDispatcher::run_graph(
            &graph,
            &mut workers,
            &catalog,
            &live_workers,
            None,
            &GridRunConfig::default(),
        )
        .await;
        assert!(report.success, "{:?}", report.traces);
        assert_eq!(report.units_validated, 6);
        let nodes_used: std::collections::BTreeSet<_> =
            report.traces.iter().map(|t| t.node_id.clone()).collect();
        assert_eq!(nodes_used.len(), 2, "both workers took replay units");
        // Every validated result is a measured compute sample on the worker that ran it.
        assert_eq!(report.compute_samples, 6);
        let sampled: u32 = workers
            .iter()
            .filter_map(|w| w.compute.get("DreamReplay"))
            .map(|e| e.samples)
            .sum();
        assert_eq!(sampled, 6);
        assert!(
            workers
                .iter()
                .filter_map(|w| w.compute.get("DreamReplay"))
                .all(|e| e.ms_per_unit > 0.0 && e.ms_per_unit < 1_000.0),
            "cost is per simulation, not per unit"
        );
        assert_eq!(report.credits.len(), 6);
        assert_eq!(report.receipts.len(), 6);
        assert!(report.receipts.iter().all(|r| r.earned_usd > 0.0));
        assert!(
            report
                .receipts
                .iter()
                .all(|r| r.tokens_or_rollouts_processed >= 8)
        );
        assert!(report.credits.iter().all(|c| c.credits > 0.0));
    }

    struct TamperingAudit;

    impl WorkExecutor for TamperingAudit {
        fn app_kind(&self) -> AppKind {
            AppKind::VerdeAudit
        }

        fn execute(
            &self,
            unit: &WorkUnit,
            inputs: &WorkInputs,
            ctx: &ExecutorContext,
        ) -> Result<WorkOutput, String> {
            let mut honest = VerdeAuditExecutor::default().execute(unit, inputs, ctx)?;
            honest.digest_fnv1a = honest.digest_fnv1a.wrapping_add(1);
            honest.check_value += 1.0;
            Ok(honest)
        }
    }

    #[tokio::test]
    async fn untrusted_disagreement_is_refereed_and_only_the_honest_node_is_credited() {
        let cluster = sample_personal_cluster();
        let mut nodes = vec![cluster[0].clone(), cluster[1].clone()];
        nodes[1].trust_tier = NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
        let (_w0, w1, live_workers) = two_workers(&nodes).await;
        w1.executors.register(Arc::new(TamperingAudit));

        let unit = StepGraphBuilder::audit_unit("audit", 1, "a1", 0, 7, None);
        let mut graph = WorkGraph::new("audit/a1");
        graph.add_unit(unit);
        let catalog = ReplicaCatalog::new();
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let (bank, _) = crate::build_initial_resident_ktb_bank("deepseek-v3-671b-moe-q4k", 1);
        let referee = LocalExecutor {
            registry: ExecutorRegistry::with_defaults(),
            ctx: ExecutorContext::new(
                "node-coordinator",
                "coordinator",
                Arc::new(Mutex::new(bank)),
            ),
        };
        let report = GridDispatcher::run_graph(
            &graph,
            &mut workers,
            &catalog,
            &live_workers,
            Some(&referee),
            &GridRunConfig::default(),
        )
        .await;
        assert!(report.success, "{:?}", report.traces);
        assert!(
            report
                .traces
                .iter()
                .any(|t| t.transport.starts_with("referee"))
        );
        let honest = nodes[0].id.clone();
        let tamperer = nodes[1].id.clone();
        assert_eq!(report.credits.len(), 1);
        assert_eq!(report.credits[0].node_id, honest);
        assert!(report.events.iter().any(|e| matches!(
            &e.event,
            kadmium_grid::SchedulerEvent::Invalidated { node, .. } if *node == tamperer
        )));
        assert!(
            report
                .events
                .iter()
                .any(|e| matches!(&e.event, kadmium_grid::SchedulerEvent::DisputeOpened { .. }))
        );
        assert_eq!(report.receipts.len(), 1);
        assert_eq!(report.receipts[0].provider_node_id, honest);
        // The tamperer changed its digest but not its layer chain, so there was nothing to
        // bisect: the referee re-ran the whole unit.
        assert_eq!(report.disputes.len(), 1);
        assert_eq!(report.disputes[0].method, "whole-unit");
        assert_eq!(report.disputes[0].referee_layers_run, 8);
        assert_eq!(report.disputes[0].winner.as_ref(), Some(&honest));
        assert_eq!(report.disputes[0].losers, vec![tamperer]);
    }

    /// Two untrusted disputants, both real TCP workers, over one eight-layer stage; the second
    /// disputant's stage executor corrupts `fault_layer`. The referee is a third TCP worker on
    /// owned hardware, or the coordinator's in-process executor.
    async fn disputed_stage(
        fault_layer: u32,
        in_process_referee: bool,
    ) -> (GridRunReport, NodeId, NodeId, Vec<u8>) {
        let cluster = sample_personal_cluster();
        let mut nodes = [cluster[0].clone(), cluster[2].clone(), cluster[3].clone()];
        nodes[1].trust_tier = NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
        nodes[2].trust_tier = NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
        let w0 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[0].clone(), false)
            .await
            .expect("spawn referee worker");
        let w1 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[1].clone(), false)
            .await
            .expect("spawn honest worker");
        let w2 = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, nodes[2].clone(), false)
            .await
            .expect("spawn tampering worker");
        w2.executors
            .register(Arc::new(StageForwardExecutor::with_injected_fault(
                InjectedLayerFault {
                    layer: fault_layer,
                    element: 0,
                    offset: 0.45,
                },
            )));
        let mut live_workers = BTreeMap::new();
        if !in_process_referee {
            live_workers.insert(
                nodes[0].id.0.clone(),
                registration(&nodes[0], &w0.local_addr.to_string(), 0.3),
            );
        }
        live_workers.insert(
            nodes[1].id.0.clone(),
            registration(&nodes[1], &w1.local_addr.to_string(), 0.2),
        );
        live_workers.insert(
            nodes[2].id.0.clone(),
            registration(&nodes[2], &w2.local_addr.to_string(), 0.2),
        );

        let graph_id = "verde/step-1";
        let prompt = StepGraphBuilder::prompt_ref(graph_id);
        let mut unit = WorkUnit::new("stage-00", AppKind::StageForward)
            .with_params_json(&kadmium_sharding::StageForwardParams {
                model_id: "toy-moe".into(),
                epoch: 1,
                session_id: "verde".into(),
                stage_index: 0,
                start_layer: 0,
                end_layer: 7,
                rows: 1,
                q_star_ratio: 0.7,
                experts_per_layer: 8,
                active_experts_per_token: 2,
            })
            .unwrap()
            .with_placement(kadmium_grid::PlacementConstraints {
                excluded_nodes: vec![nodes[0].id.clone()],
                ..kadmium_grid::PlacementConstraints::default()
            })
            .with_cost(kadmium_grid::CostEstimate::new(1e9, 0, 4096, 2.0))
            .with_redundancy(kadmium_grid::RedundancyPolicy::untrusted());
        unit.inputs = vec![prompt.clone()];
        let mut graph = WorkGraph::new(graph_id);
        graph.add_unit(unit);
        let unit = graph.unit(&WorkUnitId::new("stage-00")).unwrap().clone();

        let seed =
            TokenBatchInt8::quantize(&crate::prompt_seed_hidden("verde bisection", 64), 1, 64, 16)
                .unwrap();
        let config = GridRunConfig::default().with_seed(prompt.clone(), seed.to_le_bytes());
        let (bank, _) = crate::build_initial_resident_ktb_bank("deepseek-v3-671b-moe-q4k", 1);
        let referee = LocalExecutor {
            registry: ExecutorRegistry::with_defaults(),
            ctx: ExecutorContext::new(
                "node-coordinator",
                "coordinator",
                Arc::new(Mutex::new(bank)),
            ),
        };
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let report = GridDispatcher::run_graph(
            &graph,
            &mut workers,
            &ReplicaCatalog::new(),
            &live_workers,
            in_process_referee.then_some(&referee),
            &config,
        )
        .await;

        // What an honest node produces for the same unit, computed in-process.
        let oracle = execute_dispatch(
            WorkDispatchPayload::new(
                unit,
                InstanceId("oracle".into()),
                1,
                vec![(prompt.content_id.clone(), seed.to_le_bytes())],
            ),
            &referee.registry,
            &referee.ctx,
        );
        assert!(oracle.header.ok);
        let honest_output = oracle.blobs.into_iter().next().unwrap().1;
        (
            report,
            nodes[1].id.clone(),
            nodes[2].id.clone(),
            honest_output,
        )
    }

    #[tokio::test]
    async fn verde_bisection_locates_an_injected_layer_fault_over_tcp() {
        for (fault_layer, in_process_referee) in [(5u32, false), (0u32, true), (7u32, false)] {
            let (report, honest, tamperer, honest_output) =
                disputed_stage(fault_layer, in_process_referee).await;
            assert!(
                report.success,
                "fault at layer {fault_layer}: {:?}",
                report.traces
            );
            assert_eq!(report.disputes.len(), 1, "fault at layer {fault_layer}");
            let dispute = &report.disputes[0];
            assert_eq!(dispute.method, "layer-bisection");
            assert_eq!(dispute.layers_in_unit, 8);
            assert_eq!(
                dispute.divergent_layer,
                Some(fault_layer),
                "the first divergent layer is the one the fault was injected at"
            );
            assert_eq!(dispute.digest_comparisons, fault_layer + 1);
            assert_eq!(
                dispute.referee_layers_run, 1,
                "the referee ran one layer, not the unit"
            );
            if fault_layer == 0 {
                assert_eq!(dispute.openings_requested, 0);
            } else {
                assert!(dispute.openings_verified >= 1);
                assert!(
                    report
                        .traces
                        .iter()
                        .any(|t| t.ok && t.transport.starts_with("verde-opening")),
                    "a disputant opened the boundary over its channel"
                );
            }
            assert_eq!(dispute.winner.as_ref(), Some(&honest));
            assert_eq!(dispute.losers, vec![tamperer.clone()]);
            assert!(dispute.outcome.contains(&format!("layer {fault_layer}")));
            let referee_transport = if in_process_referee {
                "referee in-process"
            } else {
                "referee 127.0.0.1"
            };
            assert!(
                report
                    .traces
                    .iter()
                    .any(|t| t.ok && t.transport.starts_with(referee_transport)),
                "{:?}",
                report.traces
            );
            assert!(report.events.iter().any(|e| matches!(
                &e.event,
                SchedulerEvent::Invalidated { node, reason, .. }
                    if *node == tamperer && reason.contains(&format!("layer {fault_layer}"))
            )));
            assert!(report.events.iter().any(|e| matches!(
                &e.event,
                SchedulerEvent::DisputeSettled { winner: Some(w), .. } if *w == honest
            )));
            assert_eq!(report.credits.len(), 1);
            assert_eq!(report.credits[0].node_id, honest);
            assert_eq!(
                report.output("verde/step-1/stage-00/out").unwrap(),
                honest_output.as_slice(),
                "the output in circulation is the honest one"
            );
        }
    }

    #[tokio::test]
    async fn audit_disputes_bisect_to_the_tampered_layer_too() {
        let cluster = sample_personal_cluster();
        let mut nodes = vec![cluster[0].clone(), cluster[1].clone()];
        nodes[0].trust_tier = NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
        nodes[1].trust_tier = NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
        let (_w0, w1, live_workers) = two_workers(&nodes).await;
        w1.executors.register(Arc::new(VerdeAuditExecutor {
            tamper_at_layer: Some(4),
        }));
        let unit = StepGraphBuilder::audit_unit("audit", 1, "a2", 0, 7, None);
        let mut graph = WorkGraph::new("audit/a2");
        graph.add_unit(unit);
        let mut workers = GridDispatcher::workers_from_registrations(&live_workers);
        let (bank, _) = crate::build_initial_resident_ktb_bank("deepseek-v3-671b-moe-q4k", 1);
        let referee = LocalExecutor {
            registry: ExecutorRegistry::with_defaults(),
            ctx: ExecutorContext::new(
                "node-coordinator",
                "coordinator",
                Arc::new(Mutex::new(bank)),
            ),
        };
        let report = GridDispatcher::run_graph(
            &graph,
            &mut workers,
            &ReplicaCatalog::new(),
            &live_workers,
            Some(&referee),
            &GridRunConfig::default(),
        )
        .await;
        assert!(report.success, "{:?}", report.traces);
        assert_eq!(report.disputes.len(), 1);
        let dispute = &report.disputes[0];
        assert_eq!(dispute.method, "layer-bisection");
        assert_eq!(dispute.divergent_layer, Some(4));
        assert_eq!(dispute.referee_layers_run, 1);
        assert!(dispute.openings_verified >= 1);
        assert_eq!(dispute.winner.as_ref(), Some(&nodes[0].id));
        assert_eq!(dispute.losers, vec![nodes[1].id.clone()]);
        assert_eq!(report.credits.len(), 1);
        assert_eq!(report.credits[0].node_id, nodes[0].id);
    }

    #[test]
    fn the_demo_reduce_is_bit_identical_in_any_contributor_order() {
        let (bank, _) = crate::build_initial_resident_ktb_bank("deepseek-v3-671b-moe-q4k", 1);
        let local = LocalExecutor {
            registry: ExecutorRegistry::with_defaults(),
            ctx: ExecutorContext::new("node-a", "a", Arc::new(Mutex::new(bank))),
        };
        let input = TokenBatchInt8::quantize(
            &(0..32)
                .map(|i| ((i as f32) * 0.11).cos() * 0.6)
                .collect::<Vec<_>>(),
            1,
            32,
            16,
        )
        .unwrap();
        let target = TokenBatchInt8::quantize(
            &(0..16)
                .map(|j| ((j as f32) * 0.17).sin() * 0.5)
                .collect::<Vec<_>>(),
            1,
            16,
            16,
        )
        .unwrap();
        // Three independent stages over different layer ranges, as the preview graph runs them.
        let islands: Vec<crate::LiveStageTrainingResponse> = [(0u32, 9u32), (10, 19), (20, 29)]
            .iter()
            .map(|(start, end)| {
                crate::execute_stage_training_local(
                    "node-a",
                    "a",
                    &crate::LiveStageTrainingRequest {
                        epoch: 1,
                        model_id: "llama-3-70b-lora".into(),
                        stage_index: (*start / 10) as usize,
                        start_layer: *start,
                        end_layer: *end,
                        step_index: 3,
                        micro_batch_id: 3,
                        learning_rate: 1e-4,
                        boundary_input: input.tensor.clone(),
                        target_signal: target.tensor.clone(),
                    },
                )
            })
            .collect();
        let refs: Vec<ContentRef> = (0..3)
            .map(|i| ContentRef::named(ContentKind::Activation, &format!("g/island-{i}/out"), 0))
            .collect();
        let mut unit = WorkUnit::new("demo-reduce", AppKind::DeMoReduce)
            .with_params_json(&kadmium_sharding::DeMoReduceParams {
                model_id: "llama-3-70b-lora".into(),
                epoch: 1,
                session_id: "g".into(),
                step_index: 3,
                outer_lr: 0.7,
                outer_momentum: 0.9,
                contributors: vec![],
            })
            .unwrap();
        unit.inputs = refs.clone();
        unit.graph_id = Some("g".into());
        let run = |order: [usize; 3]| {
            let blobs: Vec<(String, Vec<u8>)> = order
                .iter()
                .enumerate()
                .map(|(slot, &island)| {
                    (
                        refs[slot].content_id.clone(),
                        serde_json::to_vec(&islands[island]).unwrap(),
                    )
                })
                .collect();
            let result = execute_dispatch(
                WorkDispatchPayload::new(unit.clone(), InstanceId("r".into()), 1, blobs),
                &local.registry,
                &local.ctx,
            );
            assert!(result.header.ok, "{:?}", result.header.error);
            let record = result.header.result.unwrap();
            (
                record.digest_fnv1a,
                record.check_value,
                result.blobs[0].1.clone(),
            )
        };
        let forward = run([0, 1, 2]);
        let backward = run([2, 1, 0]);
        let rotated = run([1, 2, 0]);
        assert_eq!(forward, backward);
        assert_eq!(forward, rotated);
        let report: DeMoReduceReport = serde_json::from_slice(&forward.2).unwrap();
        assert_eq!(report.contributors, 3);
        assert_ne!(report.master_weight_digest_fnv1a, 0);
    }
}
