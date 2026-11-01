//! Read-only bridge between inference clients and the compute planner.

use crate::GatewayState;
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use kadmium_core::{ModelSpec, WorkloadKind};
use kadmium_data::checkpoint::{
    CheckpointLibrary,
    placement::{
        CheckpointPlacementCatalog, CheckpointPlacementView, EligibleMachineView,
        InspectedCheckpointProfile, PlanCheckpointRequest, plan_inspected_checkpoint,
    },
};
use kadmium_sharding::{DistributedPagedKvPlanner, MachineMemoryBudget, ShardPlanner};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashSet};
use std::sync::{Arc, Mutex};

pub(crate) fn planning_catalog() -> Vec<ModelSpec> {
    vec![
        ModelSpec::llama_3_2_3b_instruct(),
        ModelSpec::qwen2_5_coder_7b(),
        ModelSpec::qwen2_5_coder_14b(),
        ModelSpec::qwen2_5_coder_32b(),
        ModelSpec::llama_3_70b_lora(),
        ModelSpec::qwen3_235b_moe(),
        ModelSpec::deepseek_v3_671b_moe(),
        ModelSpec::deepseek_v3_671b_exl3(),
    ]
}

#[derive(Default, Deserialize)]
pub(crate) struct WorkspaceQuery {
    model_id: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct PlacementQuery {
    checkpoint_id: Option<String>,
}

pub(crate) async fn inspect(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Query(query): Query<WorkspaceQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let guard = state.lock().expect("gateway state lock");
    snapshot(&guard, query.model_id.as_deref())
        .map(Json)
        .map_err(|error| (StatusCode::BAD_REQUEST, Json(json!({"error": error}))))
}

pub(crate) async fn get_checkpoint_placement(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Query(query): Query<PlacementQuery>,
) -> Json<CheckpointPlacementCatalog> {
    let (nodes, offers, online_node_ids) = {
        let guard = state.lock().expect("gateway state lock");
        let offers = crate::grid_api::effective_offers(&guard.nodes, &guard.grid_offers);
        let online: BTreeSet<String> = guard
            .live_workers
            .values()
            .filter(|worker| worker.online)
            .map(|worker| worker.node.id.0.clone())
            .collect();
        (guard.nodes.clone(), offers, online)
    };
    let budgets = MachineMemoryBudget::from_cluster(&nodes, &offers);
    let eligible_machines: Vec<EligibleMachineView> = budgets
        .iter()
        .map(|b| EligibleMachineView::from_budget(b, online_node_ids.contains(&b.node_id.0)))
        .collect();
    let total_cluster_capacity_bytes = budgets
        .iter()
        .fold(0u64, |acc, b| acc.saturating_add(b.capacity_bytes));

    let wanted = query.checkpoint_id.filter(|id| !id.trim().is_empty());
    let (checkpoints, selected_checkpoint_id, placement, placement_error) =
        tokio::task::spawn_blocking(move || {
            let library = CheckpointLibrary::default();
            let checkpoints = library.list().unwrap_or_default();
            let selected = wanted
                .or_else(|| {
                    checkpoints
                        .iter()
                        .find(|c| c.source_format == "safetensors" && c.pre_quantized.is_none())
                        .map(|c| c.id.clone())
                })
                .or_else(|| checkpoints.first().map(|c| c.id.clone()));
            let Some(ref id) = selected else {
                return (checkpoints, None, None, None);
            };
            match library.inspect_placement_profile(id) {
                Ok(inspected) => {
                    match plan_inspected_checkpoint(&inspected, &nodes, &offers, &online_node_ids) {
                        Ok(view) => (checkpoints, selected, Some(view), None),
                        Err(err) => (checkpoints, selected, None, Some(err.to_string())),
                    }
                }
                Err(err) => (checkpoints, selected, None, Some(err)),
            }
        })
        .await
        .unwrap_or_else(|e| (Vec::new(), None, None, Some(e.to_string())));

    Json(CheckpointPlacementCatalog {
        checkpoints,
        eligible_machines,
        total_cluster_capacity_bytes,
        selected_checkpoint_id,
        placement,
        placement_error,
    })
}

pub(crate) async fn post_checkpoint_placement(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(request): Json<PlanCheckpointRequest>,
) -> Result<Json<CheckpointPlacementView>, (StatusCode, Json<Value>)> {
    let (nodes, offers, online_node_ids) = {
        let guard = state.lock().expect("gateway state lock");
        let offers = crate::grid_api::effective_offers(&guard.nodes, &guard.grid_offers);
        let online: BTreeSet<String> = guard
            .live_workers
            .values()
            .filter(|worker| worker.online)
            .map(|worker| worker.node.id.0.clone())
            .collect();
        (guard.nodes.clone(), offers, online)
    };
    let inspected = if let Some(profile) = request.profile {
        profile.validate().map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": e.to_string() })),
            )
        })?;
        let checkpoint_id = request.checkpoint_id.trim().to_string();
        if checkpoint_id.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "`checkpoint_id` must not be empty" })),
            ));
        }
        InspectedCheckpointProfile {
            checkpoint_name: request
                .checkpoint_name
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| checkpoint_id.clone()),
            checkpoint_id,
            architecture: request
                .architecture
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "LlamaForCausalLM".into()),
            source_format: request
                .source_format
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "safetensors".into()),
            profile,
        }
    } else {
        let id = request.checkpoint_id.clone();
        tokio::task::spawn_blocking(move || {
            CheckpointLibrary::default().inspect_placement_profile(&id)
        })
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
        })?
        .map_err(|err| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err })),
            )
        })?
    };

    plan_inspected_checkpoint(&inspected, &nodes, &offers, &online_node_ids)
        .map(Json)
        .map_err(|err| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
        })
}

fn snapshot(state: &GatewayState, model_id: Option<&str>) -> Result<Value, String> {
    let catalog = planning_catalog();
    let online_nodes: HashSet<_> = state
        .live_workers
        .values()
        .filter(|worker| worker.online)
        .map(|worker| &worker.node.id)
        .collect();
    let model = match model_id {
        None => &state.inference_model,
        Some(id) => catalog
            .iter()
            .find(|model| model.id == id)
            .or_else(|| (state.inference_model.id == id).then_some(&state.inference_model))
            .ok_or_else(|| {
                "Unknown planning model; choose an id from the workspace catalog".to_string()
            })?,
    };
    let plan =
        ShardPlanner::compute_plan(state.epoch, model, WorkloadKind::Inference, &state.nodes);
    let context_capacity =
        DistributedPagedKvPlanner::plan_cluster_context_capacity(&plan, model, &state.nodes);
    let assigned_layers: u32 = plan.assignments.iter().map(|node| node.layer_count).sum();
    let mut warnings = vec![
        "Placement and throughput are planner estimates, not a loaded-model or generation readiness check.".to_string(),
        "Kadmium's /v1 routes execute reference diagnostics. Use a configured inference provider for generated answers.".to_string(),
    ];
    if !state.live_only_mode {
        warnings.push(
            "This fleet includes demonstration hardware; it is not a live capacity inventory."
                .into(),
        );
    }
    if assigned_layers < model.num_layers {
        warnings.push(format!(
            "Only {assigned_layers} of {} layers were assigned.",
            model.num_layers
        ));
    }
    Ok(json!({
        "schema_version": 1,
        "epoch": state.epoch,
        "fleet_mode": if state.live_only_mode { "live" } else { "demo_and_live" },
        "generation_supported": false,
        "selected_model_id": state.user_selected_model.then_some(&state.inference_model.id),
        "online_workers": state.live_workers.values().filter(|worker| worker.online).count(),
        "catalog": catalog,
        "nodes": state.nodes.iter().map(|node| json!({
            "id": node.id,
            "hostname": node.hostname,
            "backend": node.backend,
            "polite_state": node.polite_state,
            "trust_tier": node.trust_tier,
            "available_fast_memory_gb": node.available_fast_memory_gb(),
            "registered_online": online_nodes.contains(&node.id),
        })).collect::<Vec<_>>(),
        "plan": plan,
        "context_capacity": context_capacity,
        "assigned_layers": assigned_layers,
        "warnings": warnings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_uses_requested_model_without_changing_cluster_selection() {
        let state = GatewayState::new();
        let before = serde_json::to_value(&state.inference_plan).unwrap();
        let result = snapshot(&state, Some("llama-3.2-3b-instruct")).unwrap();
        assert_eq!(result["plan"]["model_id"], "llama-3.2-3b-instruct");
        assert_eq!(result["assigned_layers"], result["plan"]["total_layers"]);
        assert_eq!(
            result["context_capacity"]["model_native_context_tokens"],
            131_072
        );
        assert!(
            result["context_capacity"]["cluster_max_full_attention_context_tokens"]
                .as_u64()
                .unwrap()
                <= 131_072
        );
        assert_eq!(result["generation_supported"], false);
        assert_eq!(result["fleet_mode"], "demo_and_live");
        assert_eq!(serde_json::to_value(&state.inference_plan).unwrap(), before);
        assert!(snapshot(&state, Some("not-a-model")).is_err());
    }

    #[test]
    fn empty_live_fleet_reports_unassigned_layers_without_readiness_claim() {
        let mut state = GatewayState::new();
        state.nodes.clear();
        state.live_only_mode = true;
        let result = snapshot(&state, None).unwrap();
        assert_eq!(result["assigned_layers"], 0);
        assert_eq!(result["online_workers"], 0);
        assert_eq!(result["fleet_mode"], "live");
        assert_eq!(result["generation_supported"], false);
        assert!(
            result["warnings"]
                .as_array()
                .unwrap()
                .iter()
                .any(|warning| warning.as_str().unwrap().contains("0 of"))
        );
    }

    #[tokio::test]
    async fn checkpoint_placement_plans_stages_and_refuses_when_machines_cannot_fit_full_context() {
        use kadmium_sharding::CheckpointMemoryProfile;

        let state = Arc::new(Mutex::new(GatewayState::new()));
        let gib = 1_073_741_824u64;
        // Build a 4-layer profile where each layer + KV cache is 28 GiB (total ~113 GiB), so no
        // single machine in the sample personal cluster fits all 4 layers, forcing a split across
        // machines.
        let profile = CheckpointMemoryProfile {
            num_layers: 4,
            max_position_embeddings: 131_072,
            embedding_bytes: 512 * 1024 * 1024,
            layer_weight_bytes: vec![20 * gib; 4],
            final_norm_bytes: 65_536,
            lm_head_bytes: 512 * 1024 * 1024,
            tie_word_embeddings: false,
            kv_bytes_per_layer: 8 * gib,
        };
        let Json(view) = post_checkpoint_placement(
            State(state.clone()),
            Json(PlanCheckpointRequest {
                checkpoint_id: "checkpoint-split-test".into(),
                checkpoint_name: Some("qwen2.5-coder-32b".into()),
                architecture: Some("Qwen2ForCausalLM".into()),
                source_format: Some("safetensors".into()),
                profile: Some(profile),
            }),
        )
        .await
        .unwrap();
        assert!(view.is_split);
        assert!(view.stages.len() >= 2);
        assert_eq!(view.context_length, 131_072);
        assert_eq!(view.stages.first().unwrap().start_layer, 0);
        assert!(view.stages.first().unwrap().embedding);
        assert_eq!(view.stages.last().unwrap().end_layer, 4);
        assert!(view.stages.last().unwrap().head);

        // Now exceed total cluster capacity and verify 422 refusal without shortening context.
        let oversized = CheckpointMemoryProfile {
            num_layers: 4,
            max_position_embeddings: 131_072,
            embedding_bytes: gib,
            layer_weight_bytes: vec![200 * gib; 4],
            final_norm_bytes: 65_536,
            lm_head_bytes: gib,
            tie_word_embeddings: false,
            kv_bytes_per_layer: 64 * gib,
        };
        let (status, Json(body)) = post_checkpoint_placement(
            State(state),
            Json(PlanCheckpointRequest {
                checkpoint_id: "checkpoint-oversized".into(),
                checkpoint_name: Some("oversized-model".into()),
                architecture: Some("LlamaForCausalLM".into()),
                source_format: Some("safetensors".into()),
                profile: Some(oversized),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let msg = body["error"].as_str().unwrap_or_default();
        assert!(
            msg.contains("refusing to shorten the context or use a smaller model"),
            "{msg}"
        );
    }
}
