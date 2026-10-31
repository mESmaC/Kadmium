//! User-facing jobs require actual TCP workers. Demonstration/in-process workers cannot
//! satisfy this route, including when a live worker fails during execution.
use super::*;
use axum::http::StatusCode;
use kadmium_grid::{CostEstimate, PlacementConstraints};
use kadmium_runtime::matrix_job::MatrixProduct;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static NEXT_JOB: AtomicU64 = AtomicU64::new(1);
type ApiError = (StatusCode, Json<Value>);
fn bad(message: impl Into<String>) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error":message.into()})),
    )
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComputeJob {
    MatrixMultiply { matrix: MatrixProduct },
    TrainingPreview { step_index: u64, learning_rate: f32 },
    Replay { rollouts: u32, simulations: u32 },
}

fn live_snapshot(state: &Arc<Mutex<GatewayState>>) -> Result<MeshSnapshot, ApiError> {
    let mut snap = snapshot(state);
    snap.live_workers.retain(|_, w| w.online);
    snap.nodes
        .retain(|n| snap.live_workers.contains_key(&n.id.0));
    // These jobs carry the user's entire numerical input. Restrict them to owned workers.
    snap.nodes
        .retain(|n| n.trust_tier == kadmium_core::NodeTrustTier::TrustedCore);
    snap.nodes.retain(|n| {
        snap.offers
            .get(&n.id)
            .is_none_or(|offer| !offer.requires_approval)
    });
    let ids: BTreeSet<_> = snap.nodes.iter().map(|n| n.id.0.clone()).collect();
    snap.live_workers.retain(|id, _| ids.contains(id));
    if snap.nodes.len() < 2 {
        return Err(bad(
            "Connect at least two online trusted TCP workers with kadmium join. Demo nodes do not count.",
        ));
    }
    Ok(snap)
}

pub async fn run(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(job): Json<ComputeJob>,
) -> Result<Json<Value>, ApiError> {
    let snap = live_snapshot(&state)?;
    let id = format!(
        "compute-{}-{}",
        now_unix_ms(),
        NEXT_JOB.fetch_add(1, Ordering::Relaxed)
    );
    let mut config = GridRunConfig {
        epoch: snap.epoch,
        credit_policy: None,
        call_timeout: Duration::from_secs(15),
        ..GridRunConfig::default()
    };
    let mut matrix_shape = None;
    let (graph, catalog, kind, execution) = match job {
        ComputeJob::MatrixMultiply { matrix } => {
            matrix.validate().map_err(bad)?;
            if matrix.rows < 2 {
                return Err(bad(
                    "Distributed matrix multiplication requires at least two rows",
                ));
            }
            matrix_shape = Some((matrix.rows, matrix.columns));
            let mut graph = WorkGraph::new(&id);
            let count = snap.nodes.len().min(matrix.rows);
            for i in 0..count {
                let start = i * matrix.rows / count;
                let end = (i + 1) * matrix.rows / count;
                let part = MatrixProduct {
                    rows: end - start,
                    left: matrix.left[start * matrix.inner..end * matrix.inner].to_vec(),
                    ..matrix.clone()
                };
                let mut placement = PlacementConstraints::trusted_only();
                // A pinned row range makes multi-worker execution observable, not an estimate.
                placement.excluded_nodes = snap
                    .nodes
                    .iter()
                    .filter(|n| n.id != snap.nodes[i].id)
                    .map(|n| n.id.clone())
                    .collect();
                let unit = WorkUnit::new(
                    format!("rows-{start:06}"),
                    AppKind::Custom("matrix-product-f32-v1".into()),
                )
                .with_params_json(&part)
                .map_err(|e| bad(e.to_string()))?
                .with_placement(placement)
                .with_cost(CostEstimate::new(
                    (2 * part.rows * part.inner * part.columns) as f64,
                    ((part.left.len() + part.right.len()) * 4) as u64,
                    0,
                    1.,
                ));
                graph.add_unit(unit);
            }
            (
                graph,
                ReplicaCatalog::new(),
                "matrix_multiply",
                "FP32 CPU matrix product over supplied data; RepOps reduction; no quantized transport",
            )
        }
        ComputeJob::TrainingPreview {
            step_index,
            learning_rate,
        } => {
            if !learning_rate.is_finite()
                || !(0.0..=1.0).contains(&learning_rate)
                || learning_rate == 0.
            {
                return Err(bad("Learning rate must be finite and in (0, 1]"));
            }
            let model = &snap.training_model;
            let placement = ExpertPlacementPlanner::plan_with_offers(
                model,
                &snap.nodes,
                ExecutionStrategy::PipelineChain,
                None,
                WorkloadKind::DistributedTraining,
                &snap.offers,
            );
            if placement.trunk_ranges.len() < 2 {
                return Err(bad(
                    "The training preview plan does not span two live workers; inspect capacity and resource offers",
                ));
            }
            let graph = StepGraphBuilder::training_step(
                model,
                &placement,
                &snap.nodes,
                snap.epoch,
                &id,
                step_index,
                learning_rate,
            );
            let input: Vec<f32> = (0..32)
                .map(|i| (((i as u64).wrapping_add(step_index) as f32) * 0.11).cos() * 0.6)
                .collect();
            let target: Vec<f32> = (0..16)
                .map(|j| ((j * 3 + 1) as f32 * 0.17).sin() * 0.5)
                .collect();
            config = config
                .with_seed(
                    StepGraphBuilder::training_input_ref(&graph.id),
                    TokenBatchInt8::quantize(&input, 1, 32, 16)
                        .map_err(bad)?
                        .to_le_bytes(),
                )
                .with_seed(
                    StepGraphBuilder::training_target_ref(&graph.id),
                    TokenBatchInt8::quantize(&target, 1, 16, 16)
                        .map_err(bad)?
                        .to_le_bytes(),
                );
            (
                graph,
                ReplicaCatalog::assume_placed(&placement.rules, now_unix_ms()),
                "training_preview",
                "Stateless synthetic per-stage gradient preview with reference Int8 boundaries; does not train the imported checkpoint",
            )
        }
        ComputeJob::Replay {
            rollouts,
            simulations,
        } => {
            if !(2..=64).contains(&rollouts) || !(8..=256).contains(&simulations) {
                return Err(bad("Replay requires 2–64 rollouts and 8–256 simulations"));
            }
            (
                StepGraphBuilder::replay_batch(
                    &id,
                    snap.epoch,
                    "gpu-kernel-world",
                    rollouts,
                    simulations,
                    &snap.nodes,
                ),
                ReplicaCatalog::new(),
                "replay",
                "Bundled in-memory Dream-RSI reference replay; no model learning",
            )
        }
    };
    let mut workers = mesh_workers(
        &snap.nodes,
        &snap.live_workers,
        &snap.telemetry,
        &snap.offers,
    );
    let report = tokio::time::timeout(Duration::from_secs(60),GridDispatcher::run_graph(&graph,&mut workers,&catalog,&snap.live_workers,None,&config))
        .await.map_err(|_| (StatusCode::GATEWAY_TIMEOUT,Json(json!({"error":"Job exceeded 60 seconds; inspect workers before retrying. Already dispatched work may finish on a worker."}))))?;
    record_run(
        &state, kind, &graph.id, None, &report, None, snap.epoch, &workers,
    );
    let nodes_used: BTreeSet<_> = report
        .traces
        .iter()
        .filter(|t| t.ok)
        .map(|t| t.node_id.0.clone())
        .collect();
    let mut result = Value::Null;
    if report.success
        && let Some((rows, columns)) = matrix_shape
    {
        let mut outputs = report.outputs.clone();
        outputs.sort_by(|a, b| a.0.cmp(&b.0));
        if outputs
            .iter()
            .any(|(_, bytes)| !bytes.len().is_multiple_of(4))
        {
            return Err((
                StatusCode::BAD_GATEWAY,
                Json(json!({"error":"Worker returned a malformed FP32 payload"})),
            ));
        }
        let values: Vec<f32> = outputs
            .iter()
            .flat_map(|(_, b)| b.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)))
            .collect();
        if values.len() != rows * columns || values.iter().any(|v| !v.is_finite()) {
            return Err((
                StatusCode::BAD_GATEWAY,
                Json(json!({"error":"Workers returned an incomplete or nonfinite matrix"})),
            ));
        }
        result = json!({"rows":rows,"columns":columns,"values":values});
    }
    Ok(Json(json!({
        "graph_id":graph.id,"kind":kind,"execution":execution,"success":report.success,
        "distributed":nodes_used.len()>1,"nodes_used":nodes_used,"wall_ms":report.wall_ms,
        "units_total":report.units_total,"units_validated":report.units_validated,
        "traces":report.traces,"result":result,
        "error":(!report.success).then_some("One or more work units failed; see traces. No local fallback was used.")
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_runtime::LiveNodeWorkerServer;

    fn matrix_job() -> ComputeJob {
        ComputeJob::MatrixMultiply {
            matrix: MatrixProduct {
                rows: 3,
                inner: 3,
                columns: 2,
                left: vec![1., 2., 3., 4., 5., 6., 0.125, -3., 2.],
                right: vec![7., 8., 9., 10., 11., 12.],
            },
        }
    }

    #[tokio::test]
    async fn actual_tcp_workers_compute_user_data_and_demo_workers_cannot_substitute() {
        let mut state = GatewayState::new();
        let demo = Arc::new(Mutex::new(GatewayState::new()));
        assert_eq!(
            run(State(demo), Json(matrix_job())).await.unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
        let mut nodes = kadmium_core::sample_personal_cluster();
        nodes.truncate(2);
        for node in &mut nodes {
            node.trust_tier = kadmium_core::NodeTrustTier::TrustedCore;
            node.polite_state = kadmium_core::PoliteState::IdleDedicated;
        }
        let mut servers = Vec::new();
        for node in nodes {
            let server = LiveNodeWorkerServer::spawn_auto_port("127.0.0.1", 0, node.clone(), false)
                .await
                .unwrap();
            state.register_live_worker(node, server.local_addr.to_string(), 0.1, true);
            servers.push(server);
        }
        let state = Arc::new(Mutex::new(state));
        let report = run(State(state.clone()), Json(matrix_job()))
            .await
            .unwrap()
            .0;
        assert_eq!(report["success"], true, "{report}");
        assert_eq!(report["distributed"], true);
        assert_eq!(report["nodes_used"].as_array().unwrap().len(), 2);
        assert_eq!(
            report["result"]["values"],
            json!([58., 64., 139., 154., -4.125, -5.])
        );
        assert!(
            report["traces"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["transport"].as_str().unwrap().contains("pooled channel"))
        );
        assert!(
            servers
                .iter()
                .all(|s| s.served_work_units.load(Ordering::Relaxed) > 0)
        );
        // A worker can disappear after its last heartbeat. A failed TCP dispatch must
        // remain a failed job even though the coordinator has a local executor.
        let unavailable = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = unavailable.local_addr().unwrap().to_string();
        drop(unavailable);
        state
            .lock()
            .unwrap()
            .live_workers
            .values_mut()
            .next()
            .unwrap()
            .worker_tcp_addr = address;
        let failed = run(State(state.clone()), Json(matrix_job()))
            .await
            .unwrap()
            .0;
        assert_eq!(failed["success"], false, "{failed}");
        assert!(failed["result"].is_null());
        assert!(
            failed["traces"]
                .as_array()
                .unwrap()
                .iter()
                .any(|trace| trace["ok"] == false)
        );
        // Offline registrations are excluded even if demonstration hardware remains in the plan.
        state
            .lock()
            .unwrap()
            .live_workers
            .values_mut()
            .next()
            .unwrap()
            .online = false;
        assert!(run(State(state), Json(matrix_job())).await.is_err());
    }

    #[test]
    fn rejects_unknown_work_and_silent_extra_parameters() {
        assert!(
            serde_json::from_value::<ComputeJob>(json!({"kind":"shell","command":"x"})).is_err()
        );
        assert!(serde_json::from_value::<ComputeJob>(json!({"kind":"training_preview","step_index":1,"learning_rate":0.1,"checkpoint":"ignored"})).is_err());
    }
}
