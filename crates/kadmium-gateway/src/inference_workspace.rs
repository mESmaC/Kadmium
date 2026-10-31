//! Read-only bridge between inference clients and the compute planner.

use crate::GatewayState;
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use kadmium_core::{ModelSpec, WorkloadKind};
use kadmium_sharding::{DistributedPagedKvPlanner, ShardPlanner};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashSet;
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

pub(crate) async fn inspect(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Query(query): Query<WorkspaceQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let guard = state.lock().expect("gateway state lock");
    snapshot(&guard, query.model_id.as_deref())
        .map(Json)
        .map_err(|error| (StatusCode::BAD_REQUEST, Json(json!({"error": error}))))
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
}
