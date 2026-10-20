//! Axum HTTP router, static asset handlers, REST API endpoints, and OpenAI/Anthropic SSE
//! gateway handlers (`kadmium-gateway::routes`).

use crate::{
    CloudShieldApiResponse, ClusterOverviewResponse, CompanionAppStatus, CompanionConnectionMode,
    DocumentationPage, DreamRsiApiResponse, GatewayState, HostProbeApiResponse, KadmiumMcpManifest,
    KadmiumMcpServer, McpJsonRpcRequest, McpJsonRpcResponse, build_documentation_catalog,
};
use axum::{
    Json, Router,
    extract::State,
    http::header,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use kadmium_core::{
    LiveHardwareProfiler, ModelSpec, NodeTrustTier, OsHookHysteresisReport,
    OsSwapchainHysteresisEngine, PoliteState, ScopedApiKey, SubClusterQuotaPolicy,
    SubClusterRbacEngine, SupplyLeasePolicy, SupplyMarketplaceSnapshot, WorkloadKind,
    sample_cloud_burst_gpu_node, sample_node_pool,
};
use kadmium_data::{EcosystemIntegrationsOverview, Fts3TransferScheduler};
use kadmium_network::{
    HolePunchNegotiationResult, JoinTicket, LanPairingCandidate, NatTraversalCoordinator,
    ZeroClickLanPairingEngine,
};
use kadmium_rl::{
    DiscoveryTree, DreamRsiWorldRegistry, GrpoEpochExecutor, GrpoVerifiedStepReport,
    RegisteredDreamRsiWorld, RlEnvironmentSuiteSpec, RlTopologyPlanner,
};
use kadmium_runtime::{
    DistributedMoEPipelineExecutor, DistributedPipelineExecutionReport, HardwareKernelDispatcher,
    HardwareKernelSuiteReport, Pipeline1F1BSchedule, Pipeline1F1BScheduler,
    VerdeFraudProofCertificate, VerdeFraudProofEngine,
};
use kadmium_sharding::{
    CloudShieldCutoverWatcher, CloudShieldWatcherReport, ClusterPoolSliceLayout,
    DataflowComparisonReport, DataflowStreamPlanner, HybridCloudArbitragePlanner,
    ProvisionedSpotContainer, ShardPlanner, ShiftTriggerReason, SpotOrderbookBidEngine,
    SpotOrderbookQuote, SubClusterPartitioner,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

const DASHBOARD_HTML: &str = include_str!("dashboard.html");
const DOCS_PORTAL_HTML: &str = include_str!("docs_portal.html");
const MARKETING_SITE_HTML: &str = include_str!("marketing_site.html");
const COMPANION_APP_HTML: &str = include_str!("companion_app.html");
const DESIGN_SYSTEM_DEMO_HTML: &str = include_str!("../../../design-system/demo.html");

const DS_TOKENS_CSS: &str = include_str!("../../../design-system/tokens.css");
const DS_TOKENS_JSON: &str = include_str!("../../../design-system/tokens.json");
const DS_BUNDLE_CSS: &str = include_str!("../../../design-system/components/bundle.css");
const DS_BUNDLE_JS: &str = include_str!("../../../design-system/components/bundle.js");
const DS_REACT_JS: &str =
    include_str!("../../../design-system/components/lib/react.production.min.js");
const DS_REACT_DOM_JS: &str =
    include_str!("../../../design-system/components/lib/react-dom.production.min.js");

const DS_FONT_ARCHIVO: &[u8] =
    include_bytes!("../../../design-system/fonts/Archivo-Variable.woff2");
const DS_FONT_JETBRAINS_MONO: &[u8] =
    include_bytes!("../../../design-system/fonts/JetBrainsMono-Variable.woff2");

const DS_LOGO_FAVICON: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-favicon.svg");
const DS_LOGO_WORDMARK_DARK: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-wordmark-dark.svg");
const DS_LOGO_WORDMARK_LIGHT: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-wordmark-light.svg");
const DS_LOGO_MARK_DARK: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-mark-dark.svg");
const DS_LOGO_MARK_LIGHT: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-mark-light.svg");
const DS_LOGO_TILE_GRAPHITE: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-tile-graphite.svg");
const DS_LOGO_TILE_CADMIUM: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-tile-cadmium.svg");

#[derive(Debug, Deserialize)]
pub struct ShiftRequest {
    pub scenario: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionPairingApiResponse {
    pub lan_candidates: Vec<LanPairingCandidate>,
    pub os_hysteresis: OsHookHysteresisReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaGovernanceApiResponse {
    pub spot_orderbook: Vec<SpotOrderbookQuote>,
    pub provisioned_spot_container: Option<ProvisionedSpotContainer>,
    pub slice_quotas: Vec<SubClusterQuotaPolicy>,
    pub scoped_api_keys: Vec<ScopedApiKey>,
    pub dream_rsi_world_registry: Vec<RegisteredDreamRsiWorld>,
    pub sample_fraud_proof: Option<VerdeFraudProofCertificate>,
}

/// Builds the Axum router serving both the Unified Dashboard and all Backend REST/AI Gateway APIs.
pub fn build_router(state: Arc<Mutex<GatewayState>>) -> Router {
    Router::new()
        .route("/", get(serve_dashboard))
        .route("/site", get(serve_marketing_site))
        .route("/companion", get(serve_companion_app))
        .route("/docs", get(serve_docs_portal))
        .route("/design-system", get(serve_design_system_demo))
        .route("/tokens.css", get(serve_tokens_css))
        .route("/tokens.json", get(serve_tokens_json))
        .route("/components/bundle.css", get(serve_bundle_css))
        .route("/components/bundle.js", get(serve_bundle_js))
        .route(
            "/components/lib/react.production.min.js",
            get(serve_react_js),
        )
        .route(
            "/components/lib/react-dom.production.min.js",
            get(serve_react_dom_js),
        )
        .route("/fonts/Archivo-Variable.woff2", get(serve_font_archivo))
        .route(
            "/fonts/JetBrainsMono-Variable.woff2",
            get(serve_font_jetbrains),
        )
        .route("/assets/Logos/kadmium-favicon.svg", get(serve_logo_favicon))
        .route(
            "/assets/Logos/kadmium-wordmark-dark.svg",
            get(serve_logo_wordmark_dark),
        )
        .route(
            "/assets/Logos/kadmium-wordmark-light.svg",
            get(serve_logo_wordmark_light),
        )
        .route(
            "/assets/Logos/kadmium-mark-dark.svg",
            get(serve_logo_mark_dark),
        )
        .route(
            "/assets/Logos/kadmium-mark-light.svg",
            get(serve_logo_mark_light),
        )
        .route(
            "/assets/Logos/kadmium-tile-graphite.svg",
            get(serve_logo_tile_graphite),
        )
        .route(
            "/assets/Logos/kadmium-tile-cadmium.svg",
            get(serve_logo_tile_cadmium),
        )
        .route("/api/v1/docs", get(get_documentation_catalog))
        .route("/api/v1/cluster/overview", get(get_cluster_overview))
        .route("/api/v1/cluster/shift", post(post_cluster_shift))
        .route("/api/v1/slices", get(get_pool_slices))
        .route("/api/v1/dataflow", get(get_dataflow_comparison))
        .route("/api/v1/training/schedule", get(get_training_1f1b_schedule))
        .route("/api/v1/rl/dream-rsi", get(get_dream_rsi_status))
        .route("/api/v1/rl/grpo-step", get(get_grpo_verified_step))
        .route("/api/v1/cloud-shield", get(get_cloud_shield_status))
        .route("/api/v1/host/probe", get(get_local_host_probe))
        .route("/api/v1/companion/status", get(get_companion_status))
        .route("/api/v1/companion/mode", post(post_companion_mode))
        .route("/api/v1/supply/marketplace", get(get_supply_marketplace))
        .route("/api/v1/supply/configure", post(post_supply_configure))
        .route("/api/v1/integrations", get(get_integrations_overview))
        .route("/api/v1/mcp", get(get_mcp_manifest))
        .route("/mcp", post(post_mcp_rpc))
        .route("/metrics", get(get_prometheus_metrics))
        .route("/api/v1/telemetry/wandb", get(get_wandb_summary))
        .route("/api/v1/pipeline/execute", get(get_pipeline_execution))
        .route("/api/v1/runtime/kernels", get(get_runtime_kernels))
        .route("/api/v1/network/nat", get(get_network_nat_matrix))
        .route("/api/v1/companion/pairing", get(get_companion_pairing))
        .route(
            "/api/v1/cloud-shield/watcher",
            get(get_cloud_shield_watcher),
        )
        .route("/api/v1/ga/governance", get(get_ga_governance))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state)
}

async fn serve_dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn serve_marketing_site() -> Html<&'static str> {
    Html(MARKETING_SITE_HTML)
}

async fn serve_companion_app() -> Html<&'static str> {
    Html(COMPANION_APP_HTML)
}

async fn serve_docs_portal() -> Html<&'static str> {
    Html(DOCS_PORTAL_HTML)
}

async fn serve_design_system_demo() -> Html<&'static str> {
    Html(DESIGN_SYSTEM_DEMO_HTML)
}

async fn serve_tokens_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        DS_TOKENS_CSS,
    )
}

async fn serve_tokens_json() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
        DS_TOKENS_JSON,
    )
}

async fn serve_bundle_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        DS_BUNDLE_CSS,
    )
}

async fn serve_bundle_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        DS_BUNDLE_JS,
    )
}

async fn serve_react_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        DS_REACT_JS,
    )
}

async fn serve_react_dom_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        DS_REACT_DOM_JS,
    )
}

async fn serve_font_archivo() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "font/woff2")], DS_FONT_ARCHIVO)
}

async fn serve_font_jetbrains() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "font/woff2")],
        DS_FONT_JETBRAINS_MONO,
    )
}

async fn serve_logo_favicon() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        DS_LOGO_FAVICON,
    )
}

async fn serve_logo_wordmark_dark() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        DS_LOGO_WORDMARK_DARK,
    )
}

async fn serve_logo_wordmark_light() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        DS_LOGO_WORDMARK_LIGHT,
    )
}

async fn serve_logo_mark_dark() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        DS_LOGO_MARK_DARK,
    )
}

async fn serve_logo_mark_light() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        DS_LOGO_MARK_LIGHT,
    )
}

async fn serve_logo_tile_graphite() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        DS_LOGO_TILE_GRAPHITE,
    )
}

async fn serve_logo_tile_cadmium() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        DS_LOGO_TILE_CADMIUM,
    )
}

async fn get_documentation_catalog() -> Json<Vec<DocumentationPage>> {
    Json(build_documentation_catalog())
}

async fn get_cluster_overview(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<ClusterOverviewResponse> {
    let guard = state.lock().expect("gateway state lock");
    Json(guard.snapshot())
}

async fn get_pool_slices() -> Json<ClusterPoolSliceLayout> {
    let pool = sample_node_pool(100);
    let model = ModelSpec::deepseek_v3_671b_moe();
    Json(SubClusterPartitioner::partition_pool(&pool, 5, &model))
}

async fn get_dataflow_comparison(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<DataflowComparisonReport> {
    let guard = state.lock().expect("gateway state lock");
    Json(DataflowStreamPlanner::evaluate_dataflows(
        &guard.inference_model,
        &guard.nodes,
        512,
    ))
}

async fn get_training_1f1b_schedule(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<Pipeline1F1BSchedule> {
    let guard = state.lock().expect("gateway state lock");
    Json(Pipeline1F1BScheduler::build_schedule(
        &guard.training_plan,
        12,
    ))
}

async fn get_dream_rsi_status(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<DreamRsiApiResponse> {
    let guard = state.lock().expect("gateway state lock");
    let tree = DiscoveryTree::sample_gpu_kernel_discovery_tree();
    let env_suite = RlEnvironmentSuiteSpec::dream_rsi_evolving_world();
    let rl_plan = RlTopologyPlanner::compute_rl_plan(
        guard.epoch,
        &guard.training_model,
        &env_suite,
        &guard.nodes,
    );
    Json(DreamRsiApiResponse {
        discovery_tree: tree,
        evolution_report: rl_plan.dream_rsi_report.clone(),
        rl_plan,
    })
}

async fn get_cloud_shield_status(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<CloudShieldApiResponse> {
    let guard = state.lock().expect("gateway state lock");
    Json(CloudShieldApiResponse {
        arbitrage_plan: HybridCloudArbitragePlanner::compute_arbitrage_plan(&guard.inference_model),
        failover_simulations: HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield(),
    })
}

async fn get_local_host_probe() -> Json<HostProbeApiResponse> {
    let local_node = LiveHardwareProfiler::probe_local_host();
    let q_star_ratio = local_node.compute_q_star_ratio();
    let ticket = JoinTicket::generate(
        local_node.id.clone(),
        vec!["127.0.0.1:7848".into(), "relay.kadmium.ai:4433".into()],
        Some("slice-01-prod-deploy".into()),
        NodeTrustTier::VerifiedPeerMiddleLayersOnly,
    );
    Json(HostProbeApiResponse {
        local_node,
        q_star_ratio,
        join_ticket_token: ticket.encode_token(),
    })
}

async fn get_companion_status(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<CompanionAppStatus> {
    let guard = state.lock().expect("gateway state lock");
    Json(guard.companion_status())
}

async fn post_companion_mode(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(new_mode): Json<CompanionConnectionMode>,
) -> Json<CompanionAppStatus> {
    let mut guard = state.lock().expect("gateway state lock");
    guard.companion_mode = new_mode;
    Json(guard.companion_status())
}

async fn get_supply_marketplace(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<SupplyMarketplaceSnapshot> {
    let guard = state.lock().expect("gateway state lock");
    Json(guard.supply_marketplace())
}

async fn post_supply_configure(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(new_policy): Json<SupplyLeasePolicy>,
) -> Json<SupplyMarketplaceSnapshot> {
    let mut guard = state.lock().expect("gateway state lock");
    guard.supply_policy = new_policy;
    Json(guard.supply_marketplace())
}

async fn get_pipeline_execution(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<DistributedPipelineExecutionReport> {
    let guard = state.lock().expect("gateway state lock");
    Json(DistributedMoEPipelineExecutor::execute_forward_pass(
        &guard.inference_plan,
        "Verify asymmetric MoE routing and speculative draft verification across the cluster.",
    ))
}

async fn get_runtime_kernels(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<HardwareKernelSuiteReport> {
    let guard = state.lock().expect("gateway state lock");
    Json(HardwareKernelDispatcher::run_cluster_kernel_benchmark(
        &guard.nodes,
    ))
}

async fn get_network_nat_matrix(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<Vec<HolePunchNegotiationResult>> {
    let guard = state.lock().expect("gateway state lock");
    Json(NatTraversalCoordinator::probe_cluster_nat_mesh(
        &guard.nodes,
    ))
}

async fn get_companion_pairing(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<CompanionPairingApiResponse> {
    let guard = state.lock().expect("gateway state lock");
    let coord_id = &guard.coordinator_quorum.active_leader;
    let lan_candidates = ZeroClickLanPairingEngine::discover_lan_candidates(coord_id);
    let os_hysteresis =
        OsSwapchainHysteresisEngine::sample_desktop_gaming_timeline(&guard.nodes[0]);
    Json(CompanionPairingApiResponse {
        lan_candidates,
        os_hysteresis,
    })
}

async fn get_cloud_shield_watcher() -> Json<CloudShieldWatcherReport> {
    Json(CloudShieldCutoverWatcher::evaluate_incident_timeline())
}

async fn get_ga_governance() -> Json<GaGovernanceApiResponse> {
    let (slice_quotas, scoped_api_keys) = SubClusterRbacEngine::sample_lab_rbac_catalog();
    let referee_fn = |layer: usize, state: &[f64]| -> Vec<f64> {
        state
            .iter()
            .map(|&v| v * 1.01 + (layer as f64) * 0.002)
            .collect()
    };
    let mut ref_states = vec![vec![1.0, 2.0, 3.0]];
    let mut wrk_states = vec![vec![1.0, 2.0, 3.0]];
    for l in 0..32 {
        ref_states.push(referee_fn(l, &ref_states[l]));
        let mut w = referee_fn(l, &wrk_states[l]);
        if l == 19 {
            w[1] += 0.55;
        }
        wrk_states.push(w);
    }
    let sample_fraud_proof = VerdeFraudProofEngine::generate_fraud_proof(
        "fp-cert-cd48-0019",
        "untrusted-supply-gpu-19",
        &wrk_states,
        &ref_states,
        referee_fn,
        1e-6,
        1.52,
    );

    Json(GaGovernanceApiResponse {
        spot_orderbook: SpotOrderbookBidEngine::query_live_orderbook(),
        provisioned_spot_container: SpotOrderbookBidEngine::provision_optimal_middle_shard(
            48.0,
            1.50,
            (22, 48),
        ),
        slice_quotas,
        scoped_api_keys,
        dream_rsi_world_registry: DreamRsiWorldRegistry::public_catalog(),
        sample_fraud_proof,
    })
}

async fn get_grpo_verified_step() -> Json<GrpoVerifiedStepReport> {
    Json(GrpoEpochExecutor::run_verified_grpo_step())
}

async fn get_integrations_overview(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<EcosystemIntegrationsOverview> {
    let guard = state.lock().expect("gateway state lock");
    Json(EcosystemIntegrationsOverview::build_for_plan(
        &guard.inference_plan,
    ))
}

async fn get_mcp_manifest() -> Json<KadmiumMcpManifest> {
    Json(KadmiumMcpServer::manifest())
}

async fn post_mcp_rpc(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<McpJsonRpcRequest>,
) -> Json<McpJsonRpcResponse> {
    let mut guard = state.lock().expect("gateway state lock");
    Json(KadmiumMcpServer::handle_request(&mut guard, req))
}

async fn get_prometheus_metrics(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> impl IntoResponse {
    let guard = state.lock().expect("gateway state lock");
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        guard.export_prometheus_metrics(),
    )
}

async fn get_wandb_summary(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<serde_json::Value> {
    let guard = state.lock().expect("gateway state lock");
    Json(guard.export_wandb_summary())
}

async fn post_cluster_shift(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<ShiftRequest>,
) -> Json<ClusterOverviewResponse> {
    let mut guard = state.lock().expect("gateway state lock");
    guard.epoch += 1;

    match req.scenario.as_str() {
        "gaming_mode" => {
            if let Some(pc) = guard.nodes.first_mut() {
                pc.polite_state = PoliteState::GamingOrHeavyApp;
            }
            let trigger = ShiftTriggerReason::HostEnteredGamingMode {
                node_id: guard.nodes[0].id.clone(),
            };
            let shift = ShardPlanner::plan_responsibility_shift(
                &guard.inference_plan,
                &guard.inference_model,
                &guard.nodes,
                trigger,
            );
            guard.fts3_transfers =
                Fts3TransferScheduler::plan_live_shift_transfers(&shift, &guard.nodes);
            guard.inference_plan = shift.new_plan.clone();
            guard.training_plan = ShardPlanner::compute_plan(
                guard.epoch,
                &guard.training_model,
                WorkloadKind::DistributedTraining,
                &guard.nodes,
            );
            guard.last_shift = Some(shift);
        }
        "cloud_burst" => {
            let burst = sample_cloud_burst_gpu_node();
            if !guard.nodes.iter().any(|n| n.id == burst.id) {
                guard.nodes.push(burst.clone());
            }
            let trigger = ShiftTriggerReason::CloudBurstNodeJoined { node_id: burst.id };
            let shift = ShardPlanner::plan_responsibility_shift(
                &guard.inference_plan,
                &guard.inference_model,
                &guard.nodes,
                trigger,
            );
            guard.fts3_transfers =
                Fts3TransferScheduler::plan_live_shift_transfers(&shift, &guard.nodes);
            guard.inference_plan = shift.new_plan.clone();
            guard.training_plan = ShardPlanner::compute_plan(
                guard.epoch,
                &guard.training_model,
                WorkloadKind::DistributedTraining,
                &guard.nodes,
            );
            guard.last_shift = Some(shift);
        }
        "coordinator_failover" => {
            let current_leader = guard.coordinator_quorum.active_leader.clone();
            guard.coordinator_quorum.trigger_failover(&current_leader);
        }
        _ => {
            *guard = GatewayState::new();
        }
    }

    Json(guard.snapshot())
}

/// Formats an OpenAI-compatible `text/event-stream` SSE payload (`chat.completion.chunk` + `[DONE]`).
pub fn format_openai_sse_stream(epoch: u64, model_name: &str, tokens: &[&str]) -> String {
    let mut sse = String::new();
    let req_id = format!("chatcmpl-kadmium-e{epoch}");

    for (idx, tok) in tokens.iter().enumerate() {
        let delta = if idx == 0 {
            serde_json::json!({ "role": "assistant", "content": tok })
        } else {
            serde_json::json!({ "content": tok })
        };
        let chunk = serde_json::json!({
            "id": req_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": serde_json::Value::Null
            }]
        });
        sse.push_str(&format!("data: {chunk}\n\n"));
    }

    let final_chunk = serde_json::json!({
        "id": req_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    });
    sse.push_str(&format!("data: {final_chunk}\n\ndata: [DONE]\n\n"));
    sse
}

async fn openai_chat_completions(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let guard = state.lock().expect("gateway state lock");
    let model_name = req
        .model
        .unwrap_or_else(|| guard.inference_plan.model_id.clone());
    let prompt_sample = req
        .messages
        .last()
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("Kadmium distributed inference");
    let exec_report =
        DistributedMoEPipelineExecutor::execute_forward_pass(&guard.inference_plan, prompt_sample);

    if req.stream == Some(true) {
        let sse_body = format_openai_sse_stream(
            guard.epoch,
            &model_name,
            &[
                "Executed ",
                "across ",
                "the ",
                "Kadmium ",
                "(Cd-48) ",
                "heterogeneous ",
                "fabric ",
                "(Asymmetric ",
                "Q* ",
                "Sharding ",
                "+ ",
                "Colibri ",
                "Double-Buffered ",
                "NVMe).",
            ],
        );
        return (
            [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            sse_body,
        )
            .into_response();
    }

    let active_stages: Vec<String> = guard
        .inference_plan
        .assignments
        .iter()
        .filter(|a| a.layer_count > 0)
        .map(|a| {
            format!(
                "{}[L{}..L{}]",
                a.hostname,
                a.start_layer.unwrap_or(0),
                a.end_layer.unwrap_or(0)
            )
        })
        .collect();

    Json(serde_json::json!({
        "id": format!("chatcmpl-kadmium-e{}", guard.epoch),
        "object": "chat.completion",
        "model": model_name,
        "kadmium_fabric": {
            "epoch": guard.epoch,
            "pipeline_route": active_stages,
            "estimated_tokens_per_sec": exec_report.effective_tokens_per_sec,
            "speculative_draft_node": exec_report.speculative_verification.draft_node_id,
            "speculative_speedup": exec_report.speculative_verification.effective_speedup_multiplier,
            "execution_trace": exec_report
        },
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Executed across the Kadmium heterogeneous mesh (Asymmetric Sharding + FreeToken q* + Colibri NVMe MoE Streaming)."
            },
            "finish_reason": "stop"
        }]
    }))
    .into_response()
}

async fn anthropic_messages(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<serde_json::Value> {
    let guard = state.lock().expect("gateway state lock");
    let model_name = req
        .model
        .unwrap_or_else(|| guard.inference_plan.model_id.clone());

    Json(serde_json::json!({
        "id": format!("msg_kadmium_e{}", guard.epoch),
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": [{
            "type": "text",
            "text": "Executed via Kadmium's Strata-compatible Anthropic /v1/messages gateway across your distributed home/lab cluster."
        }],
        "stop_reason": "end_turn"
    }))
}
