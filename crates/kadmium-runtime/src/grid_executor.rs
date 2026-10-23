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
    BlockwiseInt8Tensor, DistributedLanExecutor, ExpertPartial, ExpertPartialsBatch,
    LAYER_BOUNDARY_BLOCK, LayerReference, LayerRouting, LiveStageForwardRequest,
    MoERoutedExpertSelection, TokenBatchF32, TokenBatchInt8, execute_stage_layers_with_ktb_routed,
};
use kadmium_core::{NodeId, VerifiedComputeReceipt};
use kadmium_grid::{
    AppKind, ContentAccess, ContentKind, ContentRef, CreditEntry, CreditLedger, CreditPolicy,
    InstanceId, RecordedAccess, ReplicaCatalog, SchedulerEvent, SchedulerEventRecord,
    SchedulerPolicy, ValidationOutcome, WorkGraph, WorkQueue, WorkResultRecord, WorkUnit,
    WorkUnitId, WorkerProfile, fnv1a_64,
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
        registry.register(Arc::new(TrunkForwardExecutor));
        registry.register(Arc::new(ExpertSliceExecutor));
        registry.register(Arc::new(CombineExecutor));
        registry.register(Arc::new(TrainForwardBackwardExecutor));
        registry.register(Arc::new(DeMoReduceExecutor));
        registry.register(Arc::new(DreamReplayExecutor));
        registry.register(Arc::new(VerdeAuditExecutor));
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
            .with_bytes_moved(bytes_in + bytes_out)
            .with_units_processed(output.units_processed)
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
        let mut routed_hits: BTreeMap<(u32, u32), u64> = BTreeMap::new();

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
            let (response, routed) = execute_stage_layers_with_ktb_routed(
                &ctx.node_id,
                &ctx.hostname,
                &request,
                Some(&bank),
                params.experts_per_layer,
                params.active_experts_per_token,
            );
            for (layer, experts) in routed {
                for expert in experts {
                    *routed_hits.entry((layer, expert)).or_default() += 1;
                }
            }
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
            units_processed: rows.len() as u64,
            accessed: expert_accesses(&params.model_id, &routed_hits),
        })
    }
}

// ================================================================================================
// Fan-out executors: one layer as trunk, expert slices and combine on different nodes
// ================================================================================================

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt()
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
            let bytes = out.to_le_bytes();
            return Ok(WorkOutput {
                outputs: vec![(
                    StepGraphBuilder::output_ref(graph_id, &unit.id),
                    bytes.clone(),
                )],
                payload: Vec::new(),
                digest_fnv1a: fnv1a_64(&bytes),
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
        let bytes = out.to_le_bytes();
        Ok(WorkOutput {
            outputs: vec![
                (
                    StepGraphBuilder::output_ref(graph_id, &unit.id),
                    bytes.clone(),
                ),
                (
                    StepGraphBuilder::routing_ref(graph_id, &unit.id),
                    routing.to_json_bytes(),
                ),
            ],
            payload: routing.to_json_bytes(),
            digest_fnv1a: fnv1a_64(&bytes),
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
            let mut mine: Vec<&MoERoutedExpertSelection> = selections
                .iter()
                .filter(|s| params.resident_experts.contains(&s.expert_id))
                .collect();
            mine.sort_by_key(|s| s.expert_id);
            for selection in mine {
                partials.push(ExpertPartial {
                    row: row as u32,
                    expert: selection.expert_id,
                    data: LayerReference::expert_contribution_from_bank(
                        h,
                        params.layer,
                        selection.expert_id,
                        selection.gating_weight,
                        Some(&_ctx.resident_ktb_bank.lock().expect("resident bank lock")),
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
        let bytes = batch.to_le_bytes();
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                bytes.clone(),
            )],
            payload: serde_json::to_vec(&serde_json::json!({
                "layer": params.layer,
                "holder": params.holder.0,
                "experts_applied": applied,
            }))
            .unwrap_or_default(),
            digest_fnv1a: fnv1a_64(&bytes),
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
        let bytes = out.to_le_bytes();
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                bytes.clone(),
            )],
            payload: serde_json::to_vec(&serde_json::json!({
                "layer": params.layer,
                "contributors": params.contributors.len(),
                "partials": partial_batches.iter().map(|b| b.partials.len()).sum::<usize>(),
            }))
            .unwrap_or_default(),
            digest_fnv1a: fnv1a_64(&bytes),
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
}

impl Default for GridRunConfig {
    fn default() -> Self {
        Self {
            epoch: 1,
            policy: SchedulerPolicy::default(),
            call_timeout: Duration::from_secs(30),
            seed_blobs: Vec::new(),
            credit_policy: Some(CreditPolicy::default()),
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
        let mut compute_samples = 0usize;
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
                compute_samples: 0,
                accesses: vec![],
                outputs: vec![],
                credits: vec![],
                receipts: vec![],
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
                    let validation =
                        queue.report_result(&outcome.instance_id, record, now_unix_ms());
                    if let Ok(ValidationOutcome::DisputeOpened { nodes }) = validation {
                        Self::referee_dispute(
                            &mut queue,
                            &mut traces,
                            &blobs,
                            &outcome.unit_id,
                            &nodes,
                            workers,
                            live_workers,
                            local,
                            config,
                        )
                        .await;
                    }
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
            // Validated results become compute samples on their workers, so the next dispatch
            // in this run and every later run rank by measured completion.
            compute_samples += queue.feed_workers(workers, now_unix_ms());
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

        // What the validated units touched, summed per node, kind of work and content.
        let mut access_sums: BTreeMap<(NodeId, String, String), u64> = BTreeMap::new();
        for status in queue.units() {
            if let Some(record) = &status.validated {
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
        let mut ledger = CreditLedger::new(config.credit_policy.clone().unwrap_or_default());
        if config.credit_policy.is_some() {
            let now = now_unix_ms();
            for status in queue.units() {
                if let Some(record) = &status.validated {
                    ledger.credit_validated(&status.unit, record, now);
                }
            }
            for event in queue.events() {
                if let SchedulerEvent::Invalidated { node, .. } = &event.event {
                    ledger.penalize_invalid(node, now);
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
            compute_samples,
            accesses,
            outputs,
            credits: ledger.entries().to_vec(),
            receipts: ledger.receipts(),
        }
    }

    /// Settles a Verde dispute: the disputed unit is re-run by a trusted referee (a TrustedCore
    /// worker that was not a disputant, else the in-process executor), and the disputant whose
    /// digest the referee reproduces wins. With no referee or no match, the unit is cancelled.
    #[allow(clippy::too_many_arguments)]
    async fn referee_dispute(
        queue: &mut WorkQueue,
        traces: &mut Vec<GridDispatchTrace>,
        blobs: &BTreeMap<String, Vec<u8>>,
        unit_id: &WorkUnitId,
        disputants: &[NodeId],
        workers: &[WorkerProfile],
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        local: Option<&LocalExecutor>,
        config: &GridRunConfig,
    ) {
        let Some(unit) = queue.status(unit_id).map(|s| s.unit.clone()) else {
            return;
        };
        let attach: Vec<(String, Vec<u8>)> = unit
            .inputs
            .iter()
            .filter(|c| c.kind == ContentKind::Activation)
            .filter_map(|c| {
                blobs
                    .get(&c.content_id)
                    .map(|b| (c.content_id.clone(), b.clone()))
            })
            .collect();
        let instance_id = InstanceId(format!("{unit_id}#referee"));
        let payload =
            WorkDispatchPayload::new(unit.clone(), instance_id.clone(), config.epoch, attach);

        let referee_worker = workers.iter().find(|w| {
            w.online
                && w.node.trust_tier.can_host_boundary_ingress_egress()
                && !disputants.contains(w.id())
                && live_workers.get(&w.id().0).is_some_and(|r| r.online)
        });
        let t0 = Instant::now();
        let (referee_node, transport, result) = if let Some(worker) = referee_worker {
            let addr = live_workers[&worker.id().0].worker_tcp_addr.clone();
            let result = DistributedLanExecutor::channel_pool()
                .call(
                    &addr,
                    WireFrameKind::WorkDispatch,
                    config.epoch as u32,
                    payload.to_le_bytes(),
                    config.call_timeout,
                )
                .await
                .and_then(|frame| WorkResultPayload::from_le_bytes(&frame.payload));
            (worker.id().clone(), format!("referee {addr}"), result)
        } else if let Some(local) = local {
            let registry = local.registry.clone();
            let ctx = local.ctx.clone();
            let node = NodeId(ctx.node_id.clone());
            let result =
                tokio::task::spawn_blocking(move || execute_dispatch(payload, &registry, &ctx))
                    .await
                    .map_err(|e| format!("referee task failed: {e}"));
            (node, "referee in-process".to_string(), result)
        } else {
            let _ = queue.cancel(unit_id, now_unix_ms());
            return;
        };
        let wall_ms = (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;

        let referee_digest = match &result {
            Ok(payload) if payload.header.ok => {
                payload.header.result.as_ref().map(|r| r.digest_fnv1a)
            }
            _ => None,
        };
        traces.push(GridDispatchTrace {
            unit_id: unit_id.clone(),
            instance_id,
            node_id: referee_node,
            app: unit.app.label(),
            transport,
            wall_ms,
            exec_ms: result
                .as_ref()
                .ok()
                .and_then(|p| p.header.result.as_ref().map(|r| r.exec_ms))
                .unwrap_or(0.0),
            digest_fnv1a: referee_digest.unwrap_or(0),
            check_value: 0.0,
            ok: referee_digest.is_some(),
            error: result.as_ref().err().cloned(),
        });

        let winner = referee_digest.and_then(|digest| {
            queue
                .status(unit_id)
                .and_then(|s| s.returned.iter().find(|r| r.digest_fnv1a == digest))
                .map(|r| r.node_id.clone())
        });
        match winner {
            Some(node) => {
                let _ = queue.resolve_dispute(unit_id, &node, now_unix_ms());
            }
            None => {
                let _ = queue.cancel(unit_id, now_unix_ms());
            }
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
        assert_eq!(local.registry.kinds().len(), 9);

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

#[cfg(test)]
mod fanout_tests {
    use super::*;
    use crate::{DEFAULT_INITIAL_KTB_MERKLE_ROOT, LiveNodeWorkerServer, prompt_seed_hidden};
    use kadmium_core::{HardwareNode, ModelSpec, sample_personal_cluster};
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

    pub(super) fn toy_moe() -> ModelSpec {
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

    pub(super) async fn run_shape(
        strategy: ExecutionStrategy,
        model: &ModelSpec,
        nodes: &[HardwareNode],
        live_workers: &BTreeMap<String, LiveWorkerRegistration>,
        seed: &TokenBatchInt8,
        session: &str,
    ) -> (GridRunReport, Vec<u8>) {
        let placement = ExpertPlacementPlanner::plan(model, nodes, strategy, None);
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
        let seed = TokenBatchInt8::quantize(&prompt_seed_hidden("replica serving", 64), 1, 64, 16)
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
        assert!(routed.iter().all(|a| {
            ModelContentNames::parse_expert(&a.content_id).is_some_and(|(m, _, _)| m == model.id)
        }));
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
        let serving = placement.serving_assignment(|n| Some(if *n == replica { 1.0 } else { 9.0 }));
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
        let leaf = WorkUnitId::new(format!("L{:03}-experts-{}", replicated.layer, replica.0));
        let trace = report
            .trace_for(&leaf)
            .expect("the replica ran a slice unit");
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
}

// ================================================================================================
// Training, replay and audit executors
// ================================================================================================

/// The DiLoCo outer step's summary, the output of a [`AppKind::DeMoReduce`] unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeMoReduceReport {
    pub step_index: u64,
    pub contributors: usize,
    pub mean_loss_before: f32,
    pub mean_loss_after: f32,
    pub mean_wire_compression_ratio: f64,
    pub diloco_outer_consensus_norm: f32,
    pub master_weight_digest_fnv1a: u64,
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
            check_value: response.loss_after as f64,
            units_processed: input.rows.max(1) as u64,
            accessed: Vec::new(),
        })
    }
}

/// Runs a [`AppKind::DeMoReduce`] unit: decompresses every island's DeMo packet and applies the
/// DiLoCo outer Nesterov step to the master weights.
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
            return Err("reduce received no island packets".to_string());
        }
        let island_updates: Vec<Vec<f32>> = responses
            .iter()
            .map(|r| crate::DeMoDctCompressor::decompress_packet(&r.demo_dct_packet))
            .collect();
        let param_len = island_updates.first().map_or(16, Vec::len);
        let mut master_weights = vec![0.5f32; param_len];
        let mut diloco =
            crate::DiLoCoOuterOptimizer::new(param_len, params.outer_lr, params.outer_momentum);
        diloco.step_outer_nesterov(&mut master_weights, &island_updates);
        let consensus_norm = master_weights.iter().map(|v| v * v).sum::<f32>().sqrt();
        let count = responses.len() as f32;
        let mut weight_bytes = Vec::with_capacity(param_len * 4);
        for w in &master_weights {
            weight_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let report = DeMoReduceReport {
            step_index: params.step_index,
            contributors: responses.len(),
            mean_loss_before: ((responses.iter().map(|r| r.loss_before).sum::<f32>() / count)
                * 10_000.0)
                .round()
                / 10_000.0,
            mean_loss_after: ((responses.iter().map(|r| r.loss_after).sum::<f32>() / count)
                * 10_000.0)
                .round()
                / 10_000.0,
            mean_wire_compression_ratio: ((responses
                .iter()
                .map(|r| r.wire_compression_ratio)
                .sum::<f64>()
                / count as f64)
                * 10.0)
                .round()
                / 10.0,
            diloco_outer_consensus_norm: (consensus_norm * 1000.0).round() / 1000.0,
            master_weight_digest_fnv1a: fnv1a_64(&weight_bytes),
        };
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                serde_json::to_vec(&report).map_err(|e| e.to_string())?,
            )],
            payload: serde_json::to_vec(&report).unwrap_or_default(),
            digest_fnv1a: report.master_weight_digest_fnv1a,
            check_value: consensus_norm as f64,
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
            check_value: response.evolved_peak_speedup,
            units_processed: response.mcts_simulations_run as u64,
            accessed: Vec::new(),
        })
    }
}

/// Runs a [`AppKind::VerdeAudit`] unit: a layer-bisection challenge on this node, whose RepOps
/// digest is what redundant copies are compared on.
pub struct VerdeAuditExecutor;

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
            inject_fault_at_layer: params.inject_fault_at_layer,
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
    async fn training_step_graph_matches_the_legacy_islands_and_reduces() {
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
                .trace_for(&WorkUnitId::new(format!("stage-{i:02}")))
                .expect("stage trace");
            assert_eq!(trace.node_id.0, stage.node_id);
            assert_eq!(
                trace.digest_fnv1a, stage.repops_digest_fnv1a,
                "island {i} digest"
            );
        }
        let reduce_ref = StepGraphBuilder::output_ref(&graph.id, &WorkUnitId::new("demo-reduce"));
        let reduce: DeMoReduceReport =
            serde_json::from_slice(report.output(&reduce_ref.content_id).unwrap()).unwrap();
        assert_eq!(reduce.contributors, 2);
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
            let mut honest = VerdeAuditExecutor.execute(unit, inputs, ctx)?;
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
    }

    #[tokio::test]
    async fn premerged_ktb_expert_banks_execute_identically_across_chain_and_expert_parallel_over_tcp()
     {
        use super::fanout_tests::{run_shape, toy_moe};
        use crate::prompt_seed_hidden;
        use kadmium_data::{KtbPreMergedBankPacker, TensorEncoding};

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
            TokenBatchInt8::quantize(&prompt_seed_hidden("premerged ftwx bank", 64), 1, 64, 16)
                .unwrap();

        // First run without pre-merged expert banks (synthetic fallback):
        let (_fallback_report, fallback_bytes) = run_shape(
            ExecutionStrategy::PipelineChain,
            &model,
            &nodes,
            &live_workers,
            &seed,
            "fallback-chain",
        )
        .await;

        // Now build a custom 64 KiB page-aligned pre-merged .ktb bank with custom SwiGLU weights:
        let dim = 64usize;
        let mut packer = KtbPreMergedBankPacker::new(&model.id);
        let mut gamma_bytes = Vec::with_capacity(dim * 4);
        for i in 0..dim {
            let g = 1.0f32 + (i as f32) * 0.002;
            gamma_bytes.extend_from_slice(&g.to_le_bytes());
        }
        packer.add_dense_trunk_tensor("layer.rmsnorm.gamma", 0, TensorEncoding::Fp16, gamma_bytes);

        for layer in 0..model.num_layers {
            for expert in 0..model.experts_per_layer {
                let gate: Vec<f32> = (0..dim)
                    .map(|i| 0.85 + ((layer as usize + i) % 11) as f32 * 0.03)
                    .collect();
                let up: Vec<f32> = (0..dim)
                    .map(|i| 0.65 + ((expert as usize + i) % 13) as f32 * 0.04)
                    .collect();
                let down: Vec<f32> = (0..dim)
                    .map(|i| {
                        (((layer as usize * 31 + expert as usize * 17 + i * 5) % 47) as f32 - 23.0)
                            * 0.009
                    })
                    .collect();
                packer.add_premerged_f32_expert(layer, expert, &gate, &up, &down);
            }
        }
        let custom_premerged_ktb = packer.pack_to_bytes();

        *w0.resident_ktb_bank.lock().expect("w0 bank") = custom_premerged_ktb.clone();
        *w1.resident_ktb_bank.lock().expect("w1 bank") = custom_premerged_ktb;

        let (_chain_report, chain_bytes) = run_shape(
            ExecutionStrategy::PipelineChain,
            &model,
            &nodes,
            &live_workers,
            &seed,
            "premerged-chain",
        )
        .await;
        let (_ep_report, ep_bytes) = run_shape(
            ExecutionStrategy::ExpertParallel,
            &model,
            &nodes,
            &live_workers,
            &seed,
            "premerged-ep",
        )
        .await;

        // PipelineChain and ExpertParallel over TCP both read the custom pre-merged FTWX .ktb
        // banks and produce byte-for-byte identical outputs, distinct from the synthetic fallback:
        assert_eq!(chain_bytes, ep_bytes);
        assert_ne!(chain_bytes, fallback_bytes);
    }
}
