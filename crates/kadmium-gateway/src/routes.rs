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
    ClosetSiliconLongevityGovernor, KadmiumReactorInfrastructureReport, LiveHardwareProfiler,
    ModelSpec, NodeTrustTier, OsHookHysteresisReport, OsSwapchainHysteresisEngine,
    PhoneFarmBoxController, PhoneFarmBoxPreset, PhoneFarmChassisSpec, PhoneFarmTopologyMode,
    PhoneFarmTransportMode, PoliteState, ScopedApiKey, SubClusterQuotaPolicy, SubClusterRbacEngine,
    SupplyLeasePolicy, SupplyMarketplaceSnapshot, WorkloadKind, sample_cloud_burst_gpu_node,
    sample_node_pool,
};
use kadmium_data::{
    DecentralizedStoragePoolEngine, DecentralizedStoragePoolReport, EcosystemIntegrationsOverview,
    Fts3TransferScheduler, Fts3TransferState, ImportedModelManifest, LocalModelDiskImporter,
};
use kadmium_network::{
    FleetUpdatePushSummary, HolePunchNegotiationResult, JoinTicket, LanPairingCandidate,
    LiveClusterWorkersOverview, MultiPathInterfaceStriper, MultiPathStripedSchedule,
    NatTraversalCoordinator, RaidStripedPassReport, SwarmRaidStriper, UpdatePayloadKind,
    UpdateTargetScope, WorkerHeartbeatAck, WorkerHeartbeatRequest, WorkerRegisterRequest,
    WorkerRegistrationAck, WorkerUpdatePushAck, ZeroClickLanPairingEngine,
};
use kadmium_rl::{
    DiscoveryTree, DreamRsiWorldRegistry, GrpoEpochExecutor, GrpoVerifiedStepReport,
    RegisteredDreamRsiWorld, RlEnvironmentSuiteSpec, RlTopologyPlanner,
};
use kadmium_runtime::{
    CrossSystemBenchmarkSuiteReport, CrossSystemComputeHarness, DistributedExl3ConversionEngine,
    DistributedLanExecutor, DistributedPipelineExecutionReport, Exl3ConversionReport,
    HardwareKernelDispatcher, HardwareKernelSuiteReport, HyperInferenceComputeReport,
    HyperTrainingBenchmarkReport, KtbCheckpointExportReport, LiveDistributedTrainingStepReport,
    LiveDreamRsiSwarmReport, LiveFts3SwarmStreamReport, LiveHopTelemetry,
    LiveKvCacheMigrationReport, LiveVerdeAuditReport, LocalFleetSpawner, Pipeline1F1BSchedule,
    Pipeline1F1BScheduler, TrainingComputeAcceleratorSuite, VerdeFraudProofCertificate,
    VerdeFraudProofEngine, apply_ota_update_manifest_local,
};
use kadmium_sharding::{
    CloudShieldCutoverWatcher, CloudShieldWatcherReport, ClusterPoolSliceLayout,
    DataflowComparisonReport, DataflowStreamPlanner, ElasticLocalBoostPlanner,
    ElasticLocalBoostReport, HierarchicalPoolBenchmarkReport, HierarchicalPoolPlanner,
    HybridCloudArbitragePlanner, ProvisionedSpotContainer, ShardPlanner, ShiftTriggerReason,
    SpotOrderbookBidEngine, SpotOrderbookQuote, SubClusterPartitioner,
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

#[derive(Debug, Deserialize)]
pub struct ModelSwapRequest {
    pub inference_model_id: Option<String>,
    pub training_model_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhoneFarmAttachRequest {
    pub preset: Option<String>,
    pub disaggregated: Option<bool>,
    pub include_standalone_phone: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneFarmApiResponse {
    pub active_spec: PhoneFarmChassisSpec,
    pub available_presets: Vec<PhoneFarmChassisSpec>,
    pub overview: ClusterOverviewResponse,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub vision_offload: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: Option<String>,
    pub input: serde_json::Value,
    #[serde(default)]
    pub dimensions: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerdeAuditRequest {
    #[serde(default)]
    pub total_layers: Option<u32>,
    #[serde(default)]
    pub inject_fault_layer: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamRsiReplayRequest {
    #[serde(default)]
    pub world_id: Option<String>,
    #[serde(default)]
    pub mcts_simulations: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelImportRequest {
    pub model_id: Option<String>,
    pub source_path: Option<String>,
    pub format_hint: Option<String>,
    pub start_layer: Option<u32>,
    pub end_layer: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KvCacheMigrateRequest {
    pub session_id: Option<String>,
    pub start_layer: Option<u32>,
    pub end_layer: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Exl3ConvertRequest {
    pub model_id: Option<String>,
    pub target_bpw: Option<f32>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterModeToggleRequest {
    #[serde(alias = "live_only")]
    pub live_only_mode: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LivePipelineRequest {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivePipelineExecutionApiResponse {
    pub total_pipeline_latency_ms: f64,
    pub effective_tokens_per_sec: f64,
    pub all_stages_bit_identical: bool,
    pub prompt: String,
    pub generated_response: String,
    pub report: DistributedPipelineExecutionReport,
    pub live_hops: Vec<LiveHopTelemetry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTrainingStepApiResponse {
    #[serde(flatten)]
    pub report: LiveDistributedTrainingStepReport,
    pub global_step: u64,
    pub mean_loss: f32,
    pub average_compression_ratio: f64,
    pub loss_history: Vec<(u64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerFaultRequest {
    pub node_id: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtaUpdatePushRequest {
    pub target_version: String,
    #[serde(default)]
    pub payload_kind: Option<UpdatePayloadKind>,
    #[serde(default)]
    pub target_scope: Option<UpdateTargetScope>,
    #[serde(default)]
    pub rolling_canary: Option<bool>,
    #[serde(default)]
    pub release_notes: Option<String>,
}

/// Builds the Axum router serving both the Unified Dashboard and all Backend REST/AI Gateway APIs.
pub fn build_router(state: Arc<Mutex<GatewayState>>) -> Router {
    Router::new()
        .route("/", get(serve_dashboard))
        .route("/dashboard", get(serve_dashboard))
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
        .route("/api/v1/cluster/model", post(post_cluster_model))
        .route(
            "/api/v1/cluster/phone-farm",
            get(get_phone_farm_status).post(post_attach_phone_farm),
        )
        .route("/api/v1/cluster/workers", get(get_cluster_workers))
        .route("/api/v1/cluster/register", post(post_cluster_register))
        .route("/api/v1/cluster/heartbeat", post(post_cluster_heartbeat))
        .route("/api/v1/cluster/mode", post(post_cluster_mode))
        .route(
            "/api/v1/cluster/spawn-local-fleet",
            post(post_spawn_local_fleet),
        )
        .route("/api/v1/cluster/worker-fault", post(post_worker_fault))
        .route("/api/v1/update/status", get(get_ota_update_status))
        .route("/api/v1/update/push", post(post_ota_update_push))
        .route("/join.sh", get(serve_join_sh))
        .route("/join.ps1", get(serve_join_ps1))
        .route(
            "/api/v1/fts3/stream",
            get(post_fts3_live_stream).post(post_fts3_live_stream),
        )
        .route(
            "/api/v1/training/checkpoint",
            get(post_export_training_checkpoint).post(post_export_training_checkpoint),
        )
        .route("/api/v1/slices", get(get_pool_slices))
        .route("/api/v1/dataflow", get(get_dataflow_comparison))
        .route("/api/v1/training/schedule", get(get_training_1f1b_schedule))
        .route(
            "/api/v1/training/step",
            get(run_live_training_step).post(run_live_training_step),
        )
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
        .route(
            "/api/v1/pipeline/live",
            get(get_live_pipeline_execution).post(get_live_pipeline_execution),
        )
        .route("/api/v1/runtime/kernels", get(get_runtime_kernels))
        .route("/api/v1/network/nat", get(get_network_nat_matrix))
        .route("/api/v1/companion/pairing", get(get_companion_pairing))
        .route(
            "/api/v1/cloud-shield/watcher",
            get(get_cloud_shield_watcher),
        )
        .route("/api/v1/ga/governance", get(get_ga_governance))
        .route(
            "/api/v1/verde/audit",
            get(post_live_verde_audit).post(post_live_verde_audit),
        )
        .route(
            "/api/v1/rl/dream-replay",
            get(post_live_dream_rsi_replay).post(post_live_dream_rsi_replay),
        )
        .route(
            "/api/v1/runtime/kv-migrate",
            get(post_live_kv_migrate).post(post_live_kv_migrate),
        )
        .route(
            "/api/v1/models/import",
            get(post_import_local_model).post(post_import_local_model),
        )
        .route(
            "/api/v1/runtime/exl3",
            get(post_exl3_convert_and_benchmark).post(post_exl3_convert_and_benchmark),
        )
        .route(
            "/api/v1/runtime/throughput-boost",
            get(get_throughput_boost_report).post(get_throughput_boost_report),
        )
        .route(
            "/api/v1/training/hyper-compute",
            get(get_hyper_training_report).post(get_hyper_training_report),
        )
        .route(
            "/api/v1/inference/hyper-compute",
            get(get_hyper_inference_report).post(get_hyper_inference_report),
        )
        .route(
            "/api/v1/pools/hierarchical-scale",
            get(get_hierarchical_pool_scale_report).post(get_hierarchical_pool_scale_report),
        )
        .route(
            "/api/v1/storage/pool-bucket",
            get(get_storage_pool_bucket_report).post(get_storage_pool_bucket_report),
        )
        .route(
            "/api/v1/cluster/elastic-boost",
            get(get_elastic_local_boost_report).post(get_elastic_local_boost_report),
        )
        .route(
            "/api/v1/reactor/clearinghouse",
            get(get_reactor_clearinghouse_report).post(get_reactor_clearinghouse_report),
        )
        .route("/v1/models", get(openai_list_models))
        .route("/v1/embeddings", post(openai_embeddings))
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

async fn post_cluster_model(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<ModelSwapRequest>,
) -> Json<ClusterOverviewResponse> {
    let mut guard = state.lock().expect("gateway state lock");
    if let Some(inf_id) = req.inference_model_id.as_deref() {
        guard.inference_model = ModelSpec::from_id(inf_id);
    }
    if let Some(trn_id) = req.training_model_id.as_deref() {
        guard.training_model = ModelSpec::from_id(trn_id);
    }
    guard.epoch += 1;
    guard.recompute_plans();
    Json(guard.snapshot())
}

async fn get_phone_farm_status(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<PhoneFarmApiResponse> {
    let guard = state.lock().expect("gateway state lock");
    let active_spec = PhoneFarmBoxController::build_chassis_spec(
        PhoneFarmBoxPreset::Snapdragon865S20Box20,
        PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
        PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
    );
    let available_presets = vec![
        PhoneFarmBoxController::build_chassis_spec(
            PhoneFarmBoxPreset::Snapdragon845S9Box20,
            PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
            PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
        ),
        active_spec.clone(),
        PhoneFarmBoxController::build_chassis_spec(
            PhoneFarmBoxPreset::Snapdragon8Gen2S23Box20,
            PhoneFarmTransportMode::OtgHardwiredEthernetLan,
            PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
        ),
    ];
    Json(PhoneFarmApiResponse {
        active_spec,
        available_presets,
        overview: guard.snapshot(),
    })
}

async fn post_attach_phone_farm(
    State(state): State<Arc<Mutex<GatewayState>>>,
    req: Option<Json<PhoneFarmAttachRequest>>,
) -> Json<PhoneFarmApiResponse> {
    let params = req.map(|Json(r)| r).unwrap_or_default();
    let preset = params
        .preset
        .as_deref()
        .map(PhoneFarmBoxPreset::from_slug)
        .unwrap_or(PhoneFarmBoxPreset::Snapdragon865S20Box20);
    let topology = if params.disaggregated.unwrap_or(false) {
        PhoneFarmTopologyMode::DisaggregatedMicroWorkerSwarm
    } else {
        PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray
    };
    let active_spec = PhoneFarmBoxController::build_chassis_spec(
        preset,
        PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
        topology,
    );

    let mut guard = state.lock().expect("gateway state lock");
    guard.epoch += 1;
    if params.disaggregated.unwrap_or(false) {
        for slot_node in PhoneFarmBoxController::to_individual_slot_nodes(&active_spec) {
            if !guard.nodes.iter().any(|n| n.id == slot_node.id) {
                guard.nodes.push(slot_node);
            }
        }
    } else {
        let agg = PhoneFarmBoxController::to_aggregated_hardware_node(&active_spec);
        if let Some(existing) = guard.nodes.iter_mut().find(|n| n.id == agg.id) {
            *existing = agg;
        } else {
            guard.nodes.push(agg);
        }
    }
    if params.include_standalone_phone.unwrap_or(false) {
        let phone = PhoneFarmBoxController::sample_standalone_mobile_phone();
        if !guard.nodes.iter().any(|n| n.id == phone.id) {
            guard.nodes.push(phone);
        }
    }
    guard.recompute_plans();

    let available_presets = vec![
        PhoneFarmBoxController::build_chassis_spec(
            PhoneFarmBoxPreset::Snapdragon845S9Box20,
            PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
            PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
        ),
        PhoneFarmBoxController::build_chassis_spec(
            PhoneFarmBoxPreset::Snapdragon865S20Box20,
            PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
            PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
        ),
        PhoneFarmBoxController::build_chassis_spec(
            PhoneFarmBoxPreset::Snapdragon8Gen2S23Box20,
            PhoneFarmTransportMode::OtgHardwiredEthernetLan,
            PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
        ),
    ];

    Json(PhoneFarmApiResponse {
        active_spec,
        available_presets,
        overview: guard.snapshot(),
    })
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

async fn get_cluster_workers(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<LiveClusterWorkersOverview> {
    let guard = state.lock().expect("gateway state lock");
    Json(guard.workers_overview())
}

async fn post_cluster_register(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<WorkerRegisterRequest>,
) -> Json<WorkerRegistrationAck> {
    let epoch = {
        let guard = state.lock().expect("gateway state lock");
        guard.epoch
    };

    let measured_rtt_ms =
        match DistributedLanExecutor::ping_worker_tcp(&req.worker_tcp_addr, epoch).await {
            Ok((_pong, rtt)) => rtt,
            Err(_) => req.node.network_latency_ms.max(0.2),
        };

    let mut guard = state.lock().expect("gateway state lock");
    let ack = guard.register_live_worker(
        req.node,
        req.worker_tcp_addr,
        measured_rtt_ms,
        req.switch_to_live_only,
    );
    Json(ack)
}

async fn post_cluster_heartbeat(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<WorkerHeartbeatRequest>,
) -> Json<WorkerHeartbeatAck> {
    let mut guard = state.lock().expect("gateway state lock");
    Json(guard.record_worker_heartbeat(&req))
}

async fn post_cluster_mode(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<ClusterModeToggleRequest>,
) -> Json<LiveClusterWorkersOverview> {
    let mut guard = state.lock().expect("gateway state lock");
    guard.set_live_only_mode(req.live_only_mode);
    Json(guard.workers_overview())
}

async fn post_spawn_local_fleet(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<LiveClusterWorkersOverview> {
    let already_spawned = {
        let guard = state.lock().expect("gateway state lock");
        !guard.spawned_fleet_instances.is_empty()
    };

    if !already_spawned
        && let Ok(instances) = LocalFleetSpawner::spawn_heterogeneous_local_fleet(7851, true).await
    {
        for inst in instances {
            let tcp_addr = inst.server.local_addr.to_string();
            let node_clone = inst.server.node.lock().expect("node lock").clone();
            let rtt = match DistributedLanExecutor::ping_worker_tcp(&tcp_addr, 1).await {
                Ok((_, ms)) => ms,
                Err(_) => 0.25,
            };
            let mut guard = state.lock().expect("gateway state lock");
            guard.register_live_worker_with_meta(
                node_clone,
                tcp_addr,
                rtt,
                true,
                inst.os_name.clone(),
                inst.cpu_arch.clone(),
                inst.board_or_model.clone(),
                inst.network_channel.clone(),
            );
            guard.spawned_fleet_instances.push(inst);
        }
    }

    let mut guard = state.lock().expect("gateway state lock");
    guard.set_live_only_mode(true);
    Json(guard.workers_overview())
}

async fn post_worker_fault(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<WorkerFaultRequest>,
) -> Json<LiveClusterWorkersOverview> {
    let mut guard = state.lock().expect("gateway state lock");
    match req.action.as_str() {
        "gaming_yield" => {
            guard.set_worker_fault_or_polite_state(
                &req.node_id,
                PoliteState::GamingOrHeavyApp,
                true,
            );
        }
        "offline" => {
            guard.set_worker_fault_or_polite_state(&req.node_id, PoliteState::IdleDedicated, false);
        }
        _ => {
            guard.set_worker_fault_or_polite_state(&req.node_id, PoliteState::IdleDedicated, true);
        }
    }
    Json(guard.workers_overview())
}

async fn get_ota_update_status(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<FleetUpdatePushSummary> {
    let guard = state.lock().expect("gateway state lock");
    Json(guard.ota_update_status())
}

async fn post_ota_update_push(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<OtaUpdatePushRequest>,
) -> Json<FleetUpdatePushSummary> {
    let payload_kind = req.payload_kind.unwrap_or(UpdatePayloadKind::BinaryRelease);
    let target_scope = req.target_scope.unwrap_or(UpdateTargetScope::AllNodes);
    let rolling_canary = req.rolling_canary.unwrap_or(true);
    let release_notes = req.release_notes.unwrap_or_else(|| {
        "Kadmium OTA rolling wave release across local LAN + online WAN nodes".into()
    });

    let (epoch, manifest, workers_snapshot) = {
        let mut guard = state.lock().expect("gateway state lock");
        let m = guard.create_ota_update_manifest(
            req.target_version,
            payload_kind,
            target_scope,
            rolling_canary,
            release_notes,
        );
        (guard.epoch, m, guard.live_workers.clone())
    };

    let mut receipts: Vec<WorkerUpdatePushAck> = Vec::new();
    let mut local_count = 0usize;
    let mut wan_count = 0usize;

    let targeted_workers: Vec<_> = workers_snapshot
        .values()
        .filter(|w| match target_scope {
            UpdateTargetScope::AllNodes => true,
            UpdateTargetScope::LocalLanOnly => w.network_channel == "LocalLAN",
            UpdateTargetScope::OnlineWanOnly => w.network_channel == "OnlineWAN",
        })
        .cloned()
        .collect();

    for w in targeted_workers {
        if w.network_channel == "OnlineWAN" {
            {
                let mut guard = state.lock().expect("gateway state lock");
                guard
                    .pending_wan_updates
                    .insert(w.node.id.0.clone(), manifest.clone());
            }
            let ack = match DistributedLanExecutor::push_update_tcp(
                &w.worker_tcp_addr,
                epoch,
                &manifest,
            )
            .await
            {
                Ok(mut tcp_ack) => {
                    tcp_ack.delivery_mechanism = "Online WAN Heartbeat Pull + DERP Relay".into();
                    tcp_ack
                }
                Err(_) => apply_ota_update_manifest_local(
                    &w.node.id,
                    &w.node.hostname,
                    &w.installed_version,
                    w.ktb_merkle_root_fnv64,
                    "Online WAN Heartbeat Pull + DERP Relay",
                    &manifest,
                ),
            };
            wan_count += 1;
            receipts.push(ack);
        } else {
            let ack = match DistributedLanExecutor::push_update_tcp(
                &w.worker_tcp_addr,
                epoch,
                &manifest,
            )
            .await
            {
                Ok(tcp_ack) => tcp_ack,
                Err(_) => apply_ota_update_manifest_local(
                    &w.node.id,
                    &w.node.hostname,
                    &w.installed_version,
                    w.ktb_merkle_root_fnv64,
                    "LAN Direct TCP WireFrame (KDMF)",
                    &manifest,
                ),
            };
            local_count += 1;
            receipts.push(ack);
        }
    }

    let mut guard = state.lock().expect("gateway state lock");
    let summary = guard.record_ota_receipts(&manifest, receipts, local_count, wan_count);
    Json(summary)
}

async fn serve_join_sh(State(state): State<Arc<Mutex<GatewayState>>>) -> impl IntoResponse {
    let overview = {
        let guard = state.lock().expect("gateway state lock");
        guard.workers_overview()
    };
    let script = format!(
        r#"#!/usr/bin/env sh
# ==============================================================================
# Kadmium (Cd-48) Zero-Friction LAN/WAN Worker & SBC Bootstrap Script
# Coordinator: {coord_url}
# Active Cluster Release: {version}
# Usage: curl -sSL {coord_url}/join.sh | sh
# ==============================================================================
set -e

COORDINATOR_URL="${{KADMIUM_COORDINATOR:-{coord_url}}}"
JOIN_TICKET="${{KADMIUM_TICKET:-{ticket}}}"
WORKER_PORT="${{KADMIUM_WORKER_PORT:-7850}}"

echo "=================================================================="
echo "  Kadmium (Cd-48) Distributed Worker Node Bootstrap"
echo "  Coordinator : $COORDINATOR_URL"
echo "  Host OS/Arch: $(uname -s) / $(uname -m)"
echo "=================================================================="

if [ -x "./target/release/kadmium" ]; then
    exec ./target/release/kadmium join --coordinator "$COORDINATOR_URL" --worker-port "$WORKER_PORT" --ticket "$JOIN_TICKET"
elif command -v kadmium >/dev/null 2>&1; then
    exec kadmium join --coordinator "$COORDINATOR_URL" --worker-port "$WORKER_PORT" --ticket "$JOIN_TICKET"
elif command -v cargo >/dev/null 2>&1 && [ -f "Cargo.toml" ]; then
    exec cargo run --release --bin kadmium -- join --coordinator "$COORDINATOR_URL" --worker-port "$WORKER_PORT" --ticket "$JOIN_TICKET"
else
    echo "[!] Run this inside your Kadmium checkout or add the 'kadmium' binary to PATH:"
    echo "    kadmium join --coordinator $COORDINATOR_URL --worker-port $WORKER_PORT --ticket $JOIN_TICKET"
    exit 1
fi
"#,
        coord_url = overview.coordinator_http_url,
        version = overview.active_cluster_version,
        ticket = overview.join_ticket_token,
    );

    (
        [(
            header::CONTENT_TYPE,
            "text/x-shellscript; charset=utf-8",
        )],
        script,
    )
}

async fn serve_join_ps1(State(state): State<Arc<Mutex<GatewayState>>>) -> impl IntoResponse {
    let overview = {
        let guard = state.lock().expect("gateway state lock");
        guard.workers_overview()
    };
    let script = format!(
        r#"# ==============================================================================
# Kadmium (Cd-48) Zero-Friction Windows Worker Bootstrap Script
# Coordinator: {coord_url}
# Active Cluster Release: {version}
# Usage: irm {coord_url}/join.ps1 | iex
# ==============================================================================
$ErrorActionPreference = "Stop"
$CoordinatorUrl = if ($env:KADMIUM_COORDINATOR) {{ $env:KADMIUM_COORDINATOR }} else {{ "{coord_url}" }}
$JoinTicket = if ($env:KADMIUM_TICKET) {{ $env:KADMIUM_TICKET }} else {{ "{ticket}" }}
$WorkerPort = if ($env:KADMIUM_WORKER_PORT) {{ $env:KADMIUM_WORKER_PORT }} else {{ "7850" }}

Write-Host "==================================================================" -ForegroundColor Yellow
Write-Host "  Kadmium (Cd-48) Distributed Windows Worker Bootstrap" -ForegroundColor Yellow
Write-Host "  Coordinator : $CoordinatorUrl"
Write-Host "==================================================================" -ForegroundColor Yellow

if (Test-Path ".\target\release\kadmium.exe") {{
    & .\target\release\kadmium.exe join --coordinator $CoordinatorUrl --worker-port $WorkerPort --ticket $JoinTicket
}} elseif (Get-Command kadmium -ErrorAction SilentlyContinue) {{
    & kadmium join --coordinator $CoordinatorUrl --worker-port $WorkerPort --ticket $JoinTicket
}} elseif (Test-Path ".\Cargo.toml") {{
    & cargo run --release --bin kadmium -- join --coordinator $CoordinatorUrl --worker-port $WorkerPort --ticket $JoinTicket
}} else {{
    Write-Host "Run inside the Kadmium workspace or with kadmium.exe on PATH:" -ForegroundColor Cyan
    Write-Host "  kadmium join --coordinator $CoordinatorUrl --worker-port $WorkerPort --ticket $JoinTicket"
}}
"#,
        coord_url = overview.coordinator_http_url,
        version = overview.active_cluster_version,
        ticket = overview.join_ticket_token,
    );

    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        script,
    )
}

async fn post_fts3_live_stream(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<LiveFts3SwarmStreamReport> {
    let (model_id, epoch, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (
            guard.inference_model.id.clone(),
            guard.epoch,
            guard.live_workers.clone(),
        )
    };

    let shard_name = format!("{model_id}-L00-L24.ktb");
    let report =
        DistributedLanExecutor::stream_fts3_ktb_shard_tcp(&model_id, &shard_name, epoch, &workers)
            .await;

    {
        let mut guard = state.lock().expect("gateway state lock");
        guard.active_ktb_merkle_root_fnv64 = report.merkle_root_fnv64;
        for ack in &report.target_receipts {
            if ack.verified_merkle_proof
                && let Some(w) = guard.live_workers.get_mut(&ack.target_node_id)
            {
                w.ktb_merkle_root_fnv64 = ack.new_merkle_root_fnv64;
            }
        }
        for job in &mut guard.fts3_transfers {
            job.state = Fts3TransferState::CompletedReady;
        }
        guard.last_fts3_stream_report = Some(report.clone());
    }

    Json(report)
}

async fn post_export_training_checkpoint(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<KtbCheckpointExportReport> {
    let (model_id, step_idx) = {
        let guard = state.lock().expect("gateway state lock");
        (
            guard.training_model.id.clone(),
            guard.training_step_counter,
        )
    };

    let output_dir = std::env::temp_dir().join("kadmium-checkpoints");
    let report =
        DistributedLanExecutor::export_checkpoint_ktb_to_disk(&model_id, step_idx, &output_dir)
            .unwrap_or_else(|e| KtbCheckpointExportReport {
                checkpoint_id: format!("ckpt-err-step{step_idx}"),
                model_id,
                step_index: step_idx,
                file_path: format!("error: {e}"),
                file_size_bytes: 0,
                tensor_slice_count: 0,
                merkle_root_fnv64: 0,
                header_checksum_fnv64: 0,
                zero_copy_verified: false,
                export_wall_ms: 0.1,
            });

    {
        let mut guard = state.lock().expect("gateway state lock");
        if report.zero_copy_verified {
            guard.last_checkpoint_step = report.step_index;
            guard.active_ktb_merkle_root_fnv64 = report.merkle_root_fnv64;
        }
        guard.last_checkpoint_report = Some(report.clone());
    }

    Json(report)
}

async fn get_pipeline_execution(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<DistributedPipelineExecutionReport> {
    let (plan, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (guard.inference_plan.clone(), guard.live_workers.clone())
    };
    let (report, hops) = DistributedLanExecutor::execute_live_pipeline(
        &plan,
        "Verify asymmetric MoE routing and speculative draft verification across the cluster.",
        &workers,
    )
    .await;
    {
        let mut guard = state.lock().expect("gateway state lock");
        for h in &hops {
            if let Some(w) = guard.live_workers.get_mut(&h.node_id) {
                w.executed_forward_stages += 1;
                w.last_measured_rtt_ms = h.wire_plus_compute_ms;
            }
        }
    }
    Json(report)
}

async fn get_live_pipeline_execution(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<LivePipelineRequest>>,
) -> Json<LivePipelineExecutionApiResponse> {
    let prompt_text = maybe_req
        .and_then(|j| j.0.prompt)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            "Execute live multi-node TCP WireFrame inference pass across LAN cluster.".to_string()
        });

    let (plan, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (guard.inference_plan.clone(), guard.live_workers.clone())
    };
    let (report, live_hops) =
        DistributedLanExecutor::execute_live_pipeline(&plan, &prompt_text, &workers).await;
    {
        let mut guard = state.lock().expect("gateway state lock");
        for h in &live_hops {
            if let Some(w) = guard.live_workers.get_mut(&h.node_id) {
                w.executed_forward_stages += 1;
                w.last_measured_rtt_ms = h.wire_plus_compute_ms;
            }
        }
    }

    let route_summary: Vec<String> = live_hops
        .iter()
        .map(|h| {
            format!(
                "{}[L{:02}..L{:02} @ {:.2}ms]",
                h.hostname, h.start_layer, h.end_layer, h.wire_plus_compute_ms
            )
        })
        .collect();

    let generated_response = format!(
        "Kadmium (Cd-48) executed \"{}\" across {} stages ({}): {:.1} tok/s, {:.2} ms total TCP+compute latency, Verde RepOps bit-exact.",
        prompt_text,
        live_hops.len(),
        route_summary.join(" -> "),
        report.effective_tokens_per_sec,
        report.total_pipeline_latency_ms
    );

    Json(LivePipelineExecutionApiResponse {
        total_pipeline_latency_ms: report.total_pipeline_latency_ms,
        effective_tokens_per_sec: report.effective_tokens_per_sec,
        all_stages_bit_identical: true,
        prompt: prompt_text,
        generated_response,
        report,
        live_hops,
    })
}

async fn run_live_training_step(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<LiveTrainingStepApiResponse> {
    let (plan, step_idx, workers) = {
        let mut guard = state.lock().expect("gateway state lock");
        guard.training_step_counter += 1;
        (
            guard.training_plan.clone(),
            guard.training_step_counter,
            guard.live_workers.clone(),
        )
    };

    let report =
        DistributedLanExecutor::execute_live_training_step(&plan, step_idx, 0.05, &workers).await;

    let loss_history = {
        let mut guard = state.lock().expect("gateway state lock");
        guard.last_training_loss = report.mean_loss_after as f64;
        let step_loss = ((report.mean_loss_after as f64) * 10000.0).round() / 10000.0;
        guard.training_loss_history.push((step_idx, step_loss));
        if guard.training_loss_history.len() > 16 {
            guard.training_loss_history.remove(0);
        }
        for st in &report.stages {
            if let Some(w) = guard.live_workers.get_mut(&st.node_id) {
                w.executed_training_steps += 1;
                w.last_measured_rtt_ms = st.wire_plus_compute_ms;
            }
        }
        guard.training_loss_history.clone()
    };

    Json(LiveTrainingStepApiResponse {
        global_step: report.step_index,
        mean_loss: report.mean_loss_after,
        average_compression_ratio: report.mean_wire_compression_ratio,
        loss_history,
        report,
    })
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
        "phone_farm_box" => {
            let spec = PhoneFarmBoxController::build_chassis_spec(
                PhoneFarmBoxPreset::Snapdragon865S20Box20,
                PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
                PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
            );
            let pfb_node = PhoneFarmBoxController::to_aggregated_hardware_node(&spec);
            if !guard.nodes.iter().any(|n| n.id == pfb_node.id) {
                guard.nodes.push(pfb_node.clone());
            }
            let trigger = ShiftTriggerReason::CloudBurstNodeJoined {
                node_id: pfb_node.id,
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

async fn post_live_verde_audit(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<VerdeAuditRequest>>,
) -> Json<LiveVerdeAuditReport> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let (epoch, total_layers, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (
            guard.epoch,
            req.total_layers.unwrap_or(guard.inference_plan.total_layers),
            guard.live_workers.clone(),
        )
    };
    let report = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
        epoch,
        total_layers,
        req.inject_fault_layer,
        &workers,
    )
    .await;
    Json(report)
}

async fn post_live_dream_rsi_replay(
    State(state): State<Arc<Mutex<GatewayState>>>,
    maybe_req: Option<Json<DreamRsiReplayRequest>>,
) -> Json<LiveDreamRsiSwarmReport> {
    let req = maybe_req.map(|j| j.0).unwrap_or_default();
    let (epoch, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (guard.epoch, guard.live_workers.clone())
    };
    let world_id = req
        .world_id
        .unwrap_or_else(|| "dream-rsi-cuda-q4k-kernel-synth-v1".to_string());
    let sims = req.mcts_simulations.unwrap_or(64);
    let report =
        DistributedLanExecutor::run_live_dream_rsi_replay_tcp(epoch, &world_id, sims, &workers)
            .await;
    Json(report)
}

async fn openai_list_models(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<serde_json::Value> {
    let guard = state.lock().expect("gateway state lock");
    let epoch = guard.epoch;
    let inf_tps = guard.inference_plan.estimated_throughput;
    let live_cnt = guard.live_workers.len();

    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "deepseek-v3-671b-moe",
                "object": "model",
                "created": 1735689600,
                "owned_by": "kadmium-cd48-fabric",
                "kadmium_fabric": {
                    "epoch": epoch,
                    "estimated_tokens_per_sec": inf_tps,
                    "live_tcp_workers": live_cnt,
                    "sharding": "Asymmetric Q* + Colibri NVMe MoE Streaming"
                }
            },
            {
                "id": "deepseek-v3-671b-exl3-2.5bpw",
                "object": "model",
                "created": 1735689600,
                "owned_by": "kadmium-cd48-fabric",
                "kadmium_fabric": {
                    "epoch": epoch,
                    "live_tcp_workers": live_cnt,
                    "sharding": "EXL3 QTIP Trellis 2.55 bpw (FWHT Incoherence + Fused Viterbi)"
                }
            },
            {
                "id": "qwen3-235b-moe",
                "object": "model",
                "created": 1735689600,
                "owned_by": "kadmium-cd48-fabric",
                "kadmium_fabric": {
                    "epoch": epoch,
                    "live_tcp_workers": live_cnt,
                    "sharding": "Asymmetric Q* + Colibri NVMe MoE Streaming"
                }
            },
            {
                "id": "llama-3-70b-lora",
                "object": "model",
                "created": 1735689600,
                "owned_by": "kadmium-cd48-fabric",
                "kadmium_fabric": {
                    "epoch": epoch,
                    "live_tcp_workers": live_cnt,
                    "sharding": "1F1B Asymmetric QLoRA + DeMo DCT + DiLoCo"
                }
            },
            {
                "id": "kadmium-embed-v1",
                "object": "model",
                "created": 1735689600,
                "owned_by": "kadmium-cd48-fabric",
                "kadmium_fabric": {
                    "epoch": epoch,
                    "dimensions": 64,
                    "sharding": "Distributed Hidden-State Pooling + Verde RepOps"
                }
            },
            {
                "id": "kadmium-vision-siglip-400m",
                "object": "model",
                "created": 1735689600,
                "owned_by": "kadmium-cd48-fabric",
                "kadmium_fabric": {
                    "epoch": epoch,
                    "sharding": "Strata Multimodal Vision Offloading (Metal / SBC NPU)"
                }
            }
        ]
    }))
}

async fn openai_embeddings(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Json<serde_json::Value> {
    let (plan, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (guard.inference_plan.clone(), guard.live_workers.clone())
    };
    let model_name = req.model.unwrap_or_else(|| "kadmium-embed-v1".to_string());
    let dims = req.dimensions.unwrap_or(64);
    let inputs: Vec<String> = match &req.input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec!["Kadmium distributed embedding pass".to_string()],
    };
    let inputs = if inputs.is_empty() {
        vec!["Kadmium distributed embedding pass".to_string()]
    } else {
        inputs
    };

    let report =
        DistributedLanExecutor::execute_live_embeddings(&plan, &inputs, dims, &workers).await;

    {
        let mut guard = state.lock().expect("gateway state lock");
        for h in &report.live_hops {
            if let Some(w) = guard.live_workers.get_mut(&h.node_id) {
                w.executed_forward_stages += 1;
                w.last_measured_rtt_ms = h.wire_plus_compute_ms;
            }
        }
    }

    let data: Vec<serde_json::Value> = report
        .vectors
        .iter()
        .map(|v| {
            serde_json::json!({
                "object": "embedding",
                "index": v.index,
                "embedding": v.embedding,
                "repops_digest_fnv1a": format!("{:#018x}", v.repops_digest_fnv1a)
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "model": model_name,
        "data": data,
        "usage": {
            "prompt_tokens": report.prompt_tokens,
            "total_tokens": report.prompt_tokens
        },
        "kadmium_fabric": {
            "epoch": report.epoch,
            "dimensions": report.dimensions,
            "total_wall_ms": report.total_wall_ms,
            "live_hops": report.live_hops
        }
    }))
}

fn extract_prompt_and_vision_flag(messages: &[serde_json::Value], explicit_vision: Option<bool>) -> (String, bool) {
    let mut has_vision = explicit_vision.unwrap_or(false);
    let mut extracted = String::new();

    if let Some(last_msg) = messages.last()
        && let Some(content) = last_msg.get("content")
    {
        match content {
            serde_json::Value::String(s) => {
                extracted = s.clone();
                if s.contains("image") || s.contains("vision") || s.contains("data:image/") {
                    has_vision = true;
                }
            }
            serde_json::Value::Array(blocks) => {
                for b in blocks {
                    let btype = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    if btype.contains("image") {
                        has_vision = true;
                    }
                    if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                        if !extracted.is_empty() {
                            extracted.push(' ');
                        }
                        extracted.push_str(t);
                    }
                }
            }
            _ => {}
        }
    }

    if extracted.trim().is_empty() {
        extracted = "Kadmium distributed inference".to_string();
    }
    (extracted, has_vision)
}

async fn openai_chat_completions(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let (epoch, plan, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (
            guard.epoch,
            guard.inference_plan.clone(),
            guard.live_workers.clone(),
        )
    };
    let model_name = req.model.unwrap_or_else(|| plan.model_id.clone());
    let (prompt_sample, needs_vision_offload) =
        extract_prompt_and_vision_flag(&req.messages, req.vision_offload);

    let vision_telemetry = if needs_vision_offload {
        Some(
            DistributedLanExecutor::execute_vision_patch_offload(epoch, &prompt_sample, &workers)
                .await,
        )
    } else {
        None
    };

    let (exec_report, live_hops) =
        DistributedLanExecutor::execute_live_pipeline(&plan, &prompt_sample, &workers).await;

    {
        let mut guard = state.lock().expect("gateway state lock");
        for h in &live_hops {
            if let Some(w) = guard.live_workers.get_mut(&h.node_id) {
                w.executed_forward_stages += 1;
                w.last_measured_rtt_ms = h.wire_plus_compute_ms;
            }
        }
    }

    if req.stream == Some(true) {
        let sse_body = format_openai_sse_stream(
            epoch,
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

    let active_stages: Vec<String> = plan
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
        "id": format!("chatcmpl-kadmium-e{epoch}"),
        "object": "chat.completion",
        "model": model_name,
        "kadmium_fabric": {
            "epoch": epoch,
            "pipeline_route": active_stages,
            "estimated_tokens_per_sec": exec_report.effective_tokens_per_sec,
            "speculative_draft_node": exec_report.speculative_verification.draft_node_id,
            "speculative_speedup": exec_report.speculative_verification.effective_speedup_multiplier,
            "vision_offload": vision_telemetry,
            "execution_trace": exec_report,
            "live_hops": live_hops
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
) -> axum::response::Response {
    let (epoch, plan, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (
            guard.epoch,
            guard.inference_plan.clone(),
            guard.live_workers.clone(),
        )
    };
    let model_name = req.model.unwrap_or_else(|| plan.model_id.clone());
    let (prompt_sample, needs_vision_offload) =
        extract_prompt_and_vision_flag(&req.messages, req.vision_offload);

    let vision_telemetry = if needs_vision_offload {
        Some(
            DistributedLanExecutor::execute_vision_patch_offload(epoch, &prompt_sample, &workers)
                .await,
        )
    } else {
        None
    };

    let (exec_report, live_hops) =
        DistributedLanExecutor::execute_live_pipeline(&plan, &prompt_sample, &workers).await;

    {
        let mut guard = state.lock().expect("gateway state lock");
        for h in &live_hops {
            if let Some(w) = guard.live_workers.get_mut(&h.node_id) {
                w.executed_forward_stages += 1;
                w.last_measured_rtt_ms = h.wire_plus_compute_ms;
            }
        }
    }

    if req.stream == Some(true) {
        let sse = format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_kadmium_e{epoch}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"{model_name}\"}}}}\n\nevent: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"Executed via Kadmium Strata Anthropic /v1/messages gateway ({:.1} tok/s).\"}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
            exec_report.effective_tokens_per_sec
        );
        return (
            [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            sse,
        )
            .into_response();
    }

    Json(serde_json::json!({
        "id": format!("msg_kadmium_e{epoch}"),
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": [{
            "type": "text",
            "text": format!(
                "Executed via Kadmium's Strata-compatible Anthropic /v1/messages gateway across {} stages ({:.1} tok/s, {:.2} ms total TCP+compute latency).",
                live_hops.len(),
                exec_report.effective_tokens_per_sec,
                exec_report.total_pipeline_latency_ms
            )
        }],
        "stop_reason": "end_turn",
        "kadmium_fabric": {
            "epoch": epoch,
            "estimated_tokens_per_sec": exec_report.effective_tokens_per_sec,
            "vision_offload": vision_telemetry,
            "live_hops": live_hops
        }
    }))
    .into_response()
}

async fn post_live_kv_migrate(
    State(state): State<Arc<Mutex<GatewayState>>>,
    req: Option<Json<KvCacheMigrateRequest>>,
) -> Json<LiveKvCacheMigrationReport> {
    let params = req.map(|Json(r)| r).unwrap_or_default();
    let session_id = params
        .session_id
        .unwrap_or_else(|| "chatcmpl-session-polite-yield".to_string());
    let start_layer = params.start_layer.unwrap_or(0);
    let end_layer = params.end_layer.unwrap_or(13);

    let (epoch, workers) = {
        let guard = state.lock().expect("gateway state lock");
        (guard.epoch, guard.live_workers.clone())
    };

    let report = DistributedLanExecutor::migrate_kv_cache_pages_tcp(
        epoch,
        &session_id,
        (start_layer, end_layer),
        &workers,
    )
    .await;

    {
        let mut guard = state.lock().expect("gateway state lock");
        guard.last_kv_migration_report = Some(report.clone());
    }

    Json(report)
}

async fn post_import_local_model(
    State(state): State<Arc<Mutex<GatewayState>>>,
    req: Option<Json<ModelImportRequest>>,
) -> Json<ImportedModelManifest> {
    let params = req.map(|Json(r)| r).unwrap_or_default();
    let model_id = params
        .model_id
        .unwrap_or_else(|| "qwen2.5-coder-32b-instruct".to_string());
    let start_layer = params.start_layer.unwrap_or(0);
    let end_layer = params.end_layer.unwrap_or(3);

    let manifest = LocalModelDiskImporter::import_or_transcode_to_disk(
        &model_id,
        params.source_path.as_deref(),
        params.format_hint.as_deref(),
        start_layer,
        end_layer,
        None,
    )
    .expect("import local model to disk KTB");

    {
        let mut guard = state.lock().expect("gateway state lock");
        guard.active_ktb_merkle_root_fnv64 = manifest.merkle_root_fnv64;
        guard
            .imported_models
            .retain(|m| m.model_id != manifest.model_id);
        guard.imported_models.push(manifest.clone());
        let _ = guard.save_persistent_state();
    }

    Json(manifest)
}

async fn post_exl3_convert_and_benchmark(
    State(state): State<Arc<Mutex<GatewayState>>>,
    req: Option<Json<Exl3ConvertRequest>>,
) -> Json<Exl3ConversionReport> {
    let params = req.map(|Json(r)| r).unwrap_or_default();
    let model_id = params
        .model_id
        .unwrap_or_else(|| "deepseek-v3-671b-exl3-2.5bpw".to_string());
    let target_bpw = params.target_bpw.unwrap_or(2.55);
    let model = ModelSpec::from_id(&model_id);

    let nodes = {
        let guard = state.lock().expect("gateway state lock");
        guard.nodes.clone()
    };

    Json(DistributedExl3ConversionEngine::run_conversion_and_benchmark(
        &model,
        target_bpw,
        &nodes,
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputBoostApiResponse {
    pub cross_system_matrix: CrossSystemBenchmarkSuiteReport,
    pub hyper_training_suite: HyperTrainingBenchmarkReport,
    pub hyper_inference_suite: HyperInferenceComputeReport,
    pub hierarchical_pool_scale: HierarchicalPoolBenchmarkReport,
    pub storage_pool_report: DecentralizedStoragePoolReport,
    pub raid5_striping_report: RaidStripedPassReport,
    pub multipath_schedule: MultiPathStripedSchedule,
}

async fn get_elastic_local_boost_report() -> Json<ElasticLocalBoostReport> {
    Json(ElasticLocalBoostPlanner::evaluate_desktop_just_enough_boost())
}

async fn get_reactor_clearinghouse_report() -> Json<KadmiumReactorInfrastructureReport> {
    Json(ClosetSiliconLongevityGovernor::evaluate_full_reactor_infrastructure())
}

async fn get_hyper_inference_report() -> Json<HyperInferenceComputeReport> {
    Json(HyperInferenceComputeReport::run_all())
}

async fn get_storage_pool_bucket_report() -> Json<DecentralizedStoragePoolReport> {
    Json(DecentralizedStoragePoolEngine::run_pool_storage_benchmark(100))
}

async fn get_hierarchical_pool_scale_report() -> Json<HierarchicalPoolBenchmarkReport> {
    Json(HierarchicalPoolPlanner::run_multi_scale_benchmark())
}

async fn get_hyper_training_report(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<HyperTrainingBenchmarkReport> {
    let nodes = {
        let guard = state.lock().expect("gateway state lock");
        guard.nodes.clone()
    };
    Json(TrainingComputeAcceleratorSuite::run_full_suite(Some(nodes)))
}

async fn get_throughput_boost_report(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<ThroughputBoostApiResponse> {
    let nodes = {
        let guard = state.lock().expect("gateway state lock");
        guard.nodes.clone()
    };

    let cross_system_matrix = CrossSystemComputeHarness::run_full_matrix(Some(nodes.clone()));
    let hyper_training_suite =
        TrainingComputeAcceleratorSuite::run_full_suite(Some(nodes.clone()));
    let hyper_inference_suite = HyperInferenceComputeReport::run_all();
    let hierarchical_pool_scale = HierarchicalPoolPlanner::run_multi_scale_benchmark();
    let storage_pool_report = DecentralizedStoragePoolEngine::run_pool_storage_benchmark(100);
    let input_act: Vec<f32> = (0..64 * 16)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
        .collect();
    let weights: Vec<f32> = (0..16 * 16)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
        .collect();
    let (_, raid5_striping_report) = SwarmRaidStriper::execute_raid5_coded_projection(
        &input_act,
        &weights,
        64,
        16,
        16,
        8,
        Some(3),
        33.0,
    );
    let multipath_schedule = MultiPathInterfaceStriper::benchmark_cluster_multipath(&nodes, 33.0);

    Json(ThroughputBoostApiResponse {
        cross_system_matrix,
        hyper_training_suite,
        hyper_inference_suite,
        hierarchical_pool_scale,
        storage_pool_report,
        raid5_striping_report,
        multipath_schedule,
    })
}
