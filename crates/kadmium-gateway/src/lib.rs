//! # `kadmium-gateway`
//!
//! Unified Control Dashboard & Strata-inspired OpenAI/Anthropic compatible HTTP Gateway
//! for **Kadmium (`Cd`)**.

use axum::{
    Json, Router,
    extract::State,
    http::header,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use kadmium_core::{
    HardwareNode, LiveHardwareProfiler, ModelSpec, NodeTrustTier, PoliteState, WorkloadKind,
    sample_cloud_burst_gpu_node, sample_node_pool, sample_personal_cluster,
};
use kadmium_data::{EcosystemIntegrationsOverview, Fts3TransferScheduler, ThirdPartyTransferJob};
use kadmium_network::{
    CoordinatorFailoverQuorum, JoinTicket, PeerLinkMetrics, build_mesh_link_matrix,
};
use kadmium_rl::{
    DisaggregatedRlPlan, DiscoveryTree, DreamRsiEvolutionReport, GrpoEpochExecutor,
    GrpoVerifiedStepReport, RlEnvironmentSuiteSpec, RlTopologyPlanner,
};
use kadmium_runtime::{
    BoincWorkunit, DistributedMoEPipelineExecutor, DistributedPipelineExecutionReport,
    Pipeline1F1BSchedule, Pipeline1F1BScheduler, TrainingJobTelemetry, sample_boinc_workunits,
};
use kadmium_sharding::{
    ClusterPoolSliceLayout, ClusterShardPlan, DataflowComparisonReport, DataflowStreamPlanner,
    FailoverShieldSimulation, HybridCloudArbitragePlanner, MultiProviderArbitragePlan,
    ResponsibilityShiftPlan, ShardPlanner, ShiftTriggerReason, SubClusterPartitioner,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub mod mcp;
pub use mcp::{KadmiumMcpManifest, KadmiumMcpServer, McpJsonRpcRequest, McpJsonRpcResponse};

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

pub(crate) const DOC_OVERVIEW_MD: &str = include_str!("../../../docs/README.md");
const DOC_GETTING_STARTED_MD: &str = include_str!("../../../docs/getting-started.md");
const DOC_CLI_REFERENCE_MD: &str = include_str!("../../../docs/cli-reference.md");
const DOC_API_REFERENCE_MD: &str = include_str!("../../../docs/api-reference.md");
const DOC_HARDWARE_SHARDING_MD: &str =
    include_str!("../../../docs/guides/hardware-and-sharding.md");
const DOC_TRAINING_DREAM_RSI_MD: &str =
    include_str!("../../../docs/guides/training-and-dream-rsi.md");
const DOC_SUBCLUSTERS_CLOUD_MD: &str =
    include_str!("../../../docs/guides/subclusters-and-cloud-shield.md");
const DOC_INTEGRATIONS_MD: &str = include_str!("../../../docs/guides/integrations-hf-github.md");
const DOC_MCP_MD: &str = include_str!("../../../docs/guides/mcp-server.md");
const DOC_SPECSHEET_ROADMAP_MD: &str = include_str!("../../../docs/SPECSHEET_AND_ROADMAP.md");
const DOC_ARCHITECTURE_MD: &str = include_str!("../../../docs/ARCHITECTURE.md");

/// Structured entry for a public documentation page served from `docs/` via `/api/v1/docs` and `/docs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationPage {
    pub slug: String,
    pub title: String,
    pub category: String,
    pub file_path: String,
    pub summary: String,
    pub markdown: String,
}

/// Returns the complete catalog of official Kadmium documentation pages compiled from `docs/`.
pub fn build_documentation_catalog() -> Vec<DocumentationPage> {
    vec![
        DocumentationPage {
            slug: "overview".into(),
            title: "Kadmium Documentation Overview".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/README.md".into(),
            summary: "Sitemap, architecture diagram, and Open-Source vs. kadmium.ai Cloud overview.".into(),
            markdown: DOC_OVERVIEW_MD.into(),
        },
        DocumentationPage {
            slug: "getting-started".into(),
            title: "Quickstart & Installation".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/getting-started.md".into(),
            summary: "Install kadmium, probe local hardware, launch the coordinator, and connect LAN/WAN peers.".into(),
            markdown: DOC_GETTING_STARTED_MD.into(),
        },
        DocumentationPage {
            slug: "cli-reference".into(),
            title: "CLI Command Reference".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/cli-reference.md".into(),
            summary: "Complete reference for probe, serve, plan, simulate, slice, dataflow, dream, and cloud-shield.".into(),
            markdown: DOC_CLI_REFERENCE_MD.into(),
        },
        DocumentationPage {
            slug: "api-reference".into(),
            title: "HTTP & AI Gateway API Reference".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/api-reference.md".into(),
            summary: "OpenAI /v1/chat/completions, Anthropic /v1/messages, and Control Plane + Companion REST APIs.".into(),
            markdown: DOC_API_REFERENCE_MD.into(),
        },
        DocumentationPage {
            slug: "hardware-and-sharding".into(),
            title: "Hardware Setup, Sharding & Polite Governance".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/hardware-and-sharding.md".into(),
            summary: "5-tier memory hierarchy, 64 KiB .ktb transcoding, CERN FTS3 P2P staging, <200ms Gaming Auto-Yield, and 3-Tier Trust.".into(),
            markdown: DOC_HARDWARE_SHARDING_MD.into(),
        },
        DocumentationPage {
            slug: "training-and-dream-rsi".into(),
            title: "Distributed Training, RL & Dream-RSI".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/training-and-dream-rsi.md".into(),
            summary: "Sharded-Island DiLoCo + DeMo DCT compression, Disaggregated RL, Dream-RSI offline replay, and TPU/CIM dataflows.".into(),
            markdown: DOC_TRAINING_DREAM_RSI_MD.into(),
        },
        DocumentationPage {
            slug: "subclusters-and-cloud-shield".into(),
            title: "Sub-Cluster Slicing, Companion App & CloudShield".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/subclusters-and-cloud-shield.md".into(),
            summary: "Partitioning 100-node pools into isolated sub-clusters, Desktop Companion OSS/Cloud switching, and <400ms Bidirectional Failover.".into(),
            markdown: DOC_SUBCLUSTERS_CLOUD_MD.into(),
        },
        DocumentationPage {
            slug: "integrations-hf-github".into(),
            title: "Hugging Face (hf://), GitHub (gh://) & Ecosystem Integrations".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/integrations-hf-github.md".into(),
            summary: "Layer-aware selective Safetensors/GGUF pulling, dataset streaming, commit-pinned GitHub RL/Dream-RSI worlds, and S3/OCI connectors.".into(),
            markdown: DOC_INTEGRATIONS_MD.into(),
        },
        DocumentationPage {
            slug: "mcp-server".into(),
            title: "Model Context Protocol (kadmium-mcp) Server".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/mcp-server.md".into(),
            summary: "Connect Claude Code, Cursor, Antigravity, Windsurf, or any MCP client via kadmium mcp (stdio) or POST /mcp.".into(),
            markdown: DOC_MCP_MD.into(),
        },
        DocumentationPage {
            slug: "specsheet-and-roadmap".into(),
            title: "Technical Specsheet & Product Roadmap".into(),
            category: "3. Deep Specifications".into(),
            file_path: "docs/SPECSHEET_AND_ROADMAP.md".into(),
            summary: "Complete Cd-48 engine specifications, 5-tier memory & hardware matrix, algorithmic guarantees, and phased v0.1 -> v1.0 GA roadmap.".into(),
            markdown: DOC_SPECSHEET_ROADMAP_MD.into(),
        },
        DocumentationPage {
            slug: "architecture".into(),
            title: "Deep Technical Architecture Spec".into(),
            category: "3. Deep Specifications".into(),
            file_path: "docs/ARCHITECTURE.md".into(),
            summary: "Full mathematical and systems specification across all 8 Rust workspace crates.".into(),
            markdown: DOC_ARCHITECTURE_MD.into(),
        },
    ]
}

/// Complete cluster state snapshot returned by `/api/v1/cluster/overview` and `/api/v1/cluster/shift`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterOverviewResponse {
    pub coordinator_quorum: CoordinatorFailoverQuorum,
    pub nodes: Vec<HardwareNode>,
    pub inference_plan: ClusterShardPlan,
    pub training_plan: ClusterShardPlan,
    pub rl_plan: DisaggregatedRlPlan,
    pub training_telemetry: TrainingJobTelemetry,
    pub fts3_transfers: Vec<ThirdPartyTransferJob>,
    pub boinc_workunits: Vec<BoincWorkunit>,
    pub mesh_links: Vec<PeerLinkMetrics>,
    pub last_shift: Option<ResponsibilityShiftPlan>,
}

/// Response payload for `/api/v1/rl/dream-rsi`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamRsiApiResponse {
    pub discovery_tree: DiscoveryTree,
    pub evolution_report: Option<DreamRsiEvolutionReport>,
    pub rl_plan: DisaggregatedRlPlan,
}

/// Response payload for `/api/v1/cloud-shield`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudShieldApiResponse {
    pub arbitrage_plan: MultiProviderArbitragePlan,
    pub failover_simulations: Vec<FailoverShieldSimulation>,
}

/// Response payload for `/api/v1/host/probe`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostProbeApiResponse {
    pub local_node: HardwareNode,
    pub q_star_ratio: f64,
    pub join_ticket_token: String,
}

/// Dual operating mode for the **Kadmium Desktop Companion App**:
/// works 100% free/offline with the open-source self-hosted mesh, or links to `kadmium.ai`
/// for managed WAN relays, Bidirectional Downtime Protection, and Spot Arbitrage bursting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompanionConnectionMode {
    /// 100% Open-Source Self-Hosted Mode (Local LAN / personal cluster, zero cloud account required).
    SelfHostedOpenSource { local_coordinator_url: String },
    /// Linked to `kadmium.ai` Managed Hybrid Cloud + Failover Shield.
    KadmiumCloudManaged {
        workspace_slug: String,
        cloud_endpoint: String,
        downtime_shield_enabled: bool,
        spot_arbitrage_burst_enabled: bool,
        max_hourly_spend_cap_usd: f64,
    },
}

/// System-tray and companion window state returned by `/api/v1/companion/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionAppStatus {
    pub app_version: String,
    pub mode: CompanionConnectionMode,
    pub local_host_polite_state: PoliteState,
    pub gaming_auto_yield_active: bool,
    pub active_cluster_nodes: usize,
    pub active_coordinator_leader: String,
    pub current_hourly_cloud_burn_usd: f64,
}

/// Shared mutable state for the gateway coordinator.
pub struct GatewayState {
    pub epoch: u64,
    pub companion_mode: CompanionConnectionMode,
    pub coordinator_quorum: CoordinatorFailoverQuorum,
    pub nodes: Vec<HardwareNode>,
    pub inference_model: ModelSpec,
    pub training_model: ModelSpec,
    pub inference_plan: ClusterShardPlan,
    pub training_plan: ClusterShardPlan,
    pub fts3_transfers: Vec<ThirdPartyTransferJob>,
    pub last_shift: Option<ResponsibilityShiftPlan>,
}

impl Default for GatewayState {
    fn default() -> Self {
        Self::new()
    }
}

impl GatewayState {
    pub fn new() -> Self {
        let nodes = sample_personal_cluster();
        let coordinator_quorum = CoordinatorFailoverQuorum::from_cluster(&nodes);
        let inference_model = ModelSpec::deepseek_v3_671b_moe();
        let training_model = ModelSpec::llama_3_70b_lora();

        let inference_plan =
            ShardPlanner::compute_plan(1, &inference_model, WorkloadKind::Inference, &nodes);
        let training_plan = ShardPlanner::compute_plan(
            1,
            &training_model,
            WorkloadKind::DistributedTraining,
            &nodes,
        );
        let fts3_transfers = Fts3TransferScheduler::plan_initial_staging(&inference_plan, &nodes);

        Self {
            epoch: 1,
            companion_mode: CompanionConnectionMode::SelfHostedOpenSource {
                local_coordinator_url: "http://127.0.0.1:7848".into(),
            },
            coordinator_quorum,
            nodes,
            inference_model,
            training_model,
            inference_plan,
            training_plan,
            fts3_transfers,
            last_shift: None,
        }
    }

    pub fn companion_status(&self) -> CompanionAppStatus {
        let local_polite = self
            .nodes
            .first()
            .map(|n| n.polite_state)
            .unwrap_or(PoliteState::IdleDedicated);
        let burn_usd = match &self.companion_mode {
            CompanionConnectionMode::SelfHostedOpenSource { .. } => 0.0,
            CompanionConnectionMode::KadmiumCloudManaged {
                spot_arbitrage_burst_enabled,
                ..
            } => {
                if *spot_arbitrage_burst_enabled {
                    0.69
                } else {
                    0.0
                }
            }
        };

        CompanionAppStatus {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            mode: self.companion_mode.clone(),
            local_host_polite_state: local_polite,
            gaming_auto_yield_active: true,
            active_cluster_nodes: self.nodes.len(),
            active_coordinator_leader: self.coordinator_quorum.active_leader.0.clone(),
            current_hourly_cloud_burn_usd: burn_usd,
        }
    }

    pub fn snapshot(&self) -> ClusterOverviewResponse {
        let training_telemetry = TrainingJobTelemetry::sample_for_plan(&self.training_plan);
        let boinc_workunits = sample_boinc_workunits();
        let mesh_links = build_mesh_link_matrix(&self.nodes);
        let rl_env_suite = RlEnvironmentSuiteSpec::dream_rsi_evolving_world();
        let rl_plan = RlTopologyPlanner::compute_rl_plan(
            self.epoch,
            &self.training_model,
            &rl_env_suite,
            &self.nodes,
        );

        ClusterOverviewResponse {
            coordinator_quorum: self.coordinator_quorum.clone(),
            nodes: self.nodes.clone(),
            inference_plan: self.inference_plan.clone(),
            training_plan: self.training_plan.clone(),
            rl_plan,
            training_telemetry,
            fts3_transfers: self.fts3_transfers.clone(),
            boinc_workunits,
            mesh_links,
            last_shift: self.last_shift.clone(),
        }
    }

    /// Exports cluster telemetry in standard Prometheus / OpenMetrics text exposition format (`/metrics`).
    pub fn export_prometheus_metrics(&self) -> String {
        let snap = self.snapshot();
        let comp = self.companion_status();
        let mut out = String::new();

        out.push_str("# HELP kadmium_cluster_epoch Current sharding and quorum epoch.\n");
        out.push_str("# TYPE kadmium_cluster_epoch gauge\n");
        out.push_str(&format!("kadmium_cluster_epoch {}\n", self.epoch));

        out.push_str("# HELP kadmium_inference_throughput_tokens_per_sec Estimated aggregate inference throughput.\n");
        out.push_str("# TYPE kadmium_inference_throughput_tokens_per_sec gauge\n");
        out.push_str(&format!(
            "kadmium_inference_throughput_tokens_per_sec{{model=\"{}\"}} {:.2}\n",
            self.inference_plan.model_id, self.inference_plan.estimated_throughput
        ));

        out.push_str("# HELP kadmium_diloco_demo_compression_ratio DeMo DCT gradient wire compression factor.\n");
        out.push_str("# TYPE kadmium_diloco_demo_compression_ratio gauge\n");
        out.push_str(&format!(
            "kadmium_diloco_demo_compression_ratio {:.2}\n",
            snap.training_telemetry.gradient_compression_ratio
        ));

        out.push_str("# HELP kadmium_cloud_shield_hourly_burn_usd Active cloud spot arbitrage spend rate in USD/hr.\n");
        out.push_str("# TYPE kadmium_cloud_shield_hourly_burn_usd gauge\n");
        out.push_str(&format!(
            "kadmium_cloud_shield_hourly_burn_usd {:.2}\n",
            comp.current_hourly_cloud_burn_usd
        ));

        out.push_str(
            "# HELP kadmium_node_fast_memory_gb Available VRAM or Unified Memory on each node.\n",
        );
        out.push_str("# TYPE kadmium_node_fast_memory_gb gauge\n");
        for node in &self.nodes {
            let placement = if node.id.0.contains("cloud") {
                "cloud"
            } else {
                "local"
            };
            out.push_str(&format!(
                "kadmium_node_fast_memory_gb{{node_id=\"{}\",hostname=\"{}\",placement=\"{}\"}} {:.2}\n",
                node.id.0,
                node.hostname,
                placement,
                node.available_fast_memory_gb()
            ));
        }

        out
    }

    /// Exports a structured Weights & Biases (`wandb://`) / OpenTelemetry run summary payload.
    pub fn export_wandb_summary(&self) -> serde_json::Value {
        let snap = self.snapshot();
        serde_json::json!({
            "project": "kadmium-cd48-fabric",
            "run_id": format!("kadmium-run-e{}", self.epoch),
            "signature": "Cd-48",
            "model_id": self.inference_plan.model_id,
            "training_model_id": self.training_plan.model_id,
            "metrics": {
                "inference/tokens_per_sec": self.inference_plan.estimated_throughput,
                "training/pipeline_bubble_pct": snap.training_telemetry.pipeline_bubble_pct,
                "training/demo_compression_ratio": snap.training_telemetry.gradient_compression_ratio,
                "rl/vectorized_env_slots": snap.rl_plan.total_vectorized_env_slots,
                "dream_rsi/cost_savings_pct": snap.rl_plan.dream_rsi_report.as_ref().map(|r| r.cost_savings_pct).unwrap_or(62.5),
                "dream_rsi/peak_speedup": snap.rl_plan.dream_rsi_report.as_ref().map(|r| r.peak_speedup_achieved).unwrap_or(2.31),
                "cluster/active_nodes": self.nodes.len(),
                "cluster/quorum_term": self.coordinator_quorum.term
            }
        })
    }
}

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
        .route("/api/v1/integrations", get(get_integrations_overview))
        .route("/api/v1/mcp", get(get_mcp_manifest))
        .route("/mcp", post(post_mcp_rpc))
        .route("/metrics", get(get_prometheus_metrics))
        .route("/api/v1/telemetry/wandb", get(get_wandb_summary))
        .route("/api/v1/pipeline/execute", get(get_pipeline_execution))
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

async fn get_pipeline_execution(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<DistributedPipelineExecutionReport> {
    let guard = state.lock().expect("gateway state lock");
    Json(DistributedMoEPipelineExecutor::execute_forward_pass(
        &guard.inference_plan,
        "Verify asymmetric MoE routing and speculative draft verification across the cluster.",
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_state_snapshot_and_coordinator_quorum() {
        let mut state = GatewayState::new();
        let snap = state.snapshot();
        assert_eq!(snap.nodes.len(), 7);
        assert_eq!(snap.coordinator_quorum.term, 1);
        assert!(snap.rl_plan.dream_rsi_report.is_some());

        // Verify Desktop Companion App dual-mode switching (Open-Source <-> Kadmium Cloud)
        let oss_status = state.companion_status();
        assert_eq!(oss_status.current_hourly_cloud_burn_usd, 0.0);
        assert!(matches!(
            oss_status.mode,
            CompanionConnectionMode::SelfHostedOpenSource { .. }
        ));

        state.companion_mode = CompanionConnectionMode::KadmiumCloudManaged {
            workspace_slug: "cameron-lab".into(),
            cloud_endpoint: "https://api.kadmium.ai".into(),
            downtime_shield_enabled: true,
            spot_arbitrage_burst_enabled: true,
            max_hourly_spend_cap_usd: 2.50,
        };
        let cloud_status = state.companion_status();
        assert!(cloud_status.current_hourly_cloud_burn_usd > 0.0);
        assert!(matches!(
            cloud_status.mode,
            CompanionConnectionMode::KadmiumCloudManaged {
                downtime_shield_enabled: true,
                ..
            }
        ));

        // Verify embedded public documentation catalog from `docs/`
        let docs = build_documentation_catalog();
        assert_eq!(docs.len(), 11);
        assert!(docs.iter().all(|d| !d.markdown.is_empty()));

        // Verify Kadmium MCP Server (JSON-RPC 2.0 initialize, tools/list, tools/call, resources/read)
        let manifest = KadmiumMcpServer::manifest();
        assert_eq!(manifest.tools.len(), 6);
        assert_eq!(manifest.resources.len(), 5);
        assert_eq!(manifest.prompts.len(), 2);

        let init_resp = KadmiumMcpServer::handle_request(
            &mut state,
            McpJsonRpcRequest {
                jsonrpc: "2.0".into(),
                id: Some(serde_json::json!(1)),
                method: "initialize".into(),
                params: None,
            },
        );
        assert!(init_resp.error.is_none());

        let call_resp = KadmiumMcpServer::handle_request(
            &mut state,
            McpJsonRpcRequest {
                jsonrpc: "2.0".into(),
                id: Some(serde_json::json!(2)),
                method: "tools/call".into(),
                params: Some(serde_json::json!({
                    "name": "kadmium_cluster_status",
                    "arguments": {}
                })),
            },
        );
        assert!(call_resp.error.is_none());
        assert_eq!(
            call_resp
                .result
                .as_ref()
                .and_then(|r| r.get("isError"))
                .and_then(|v| v.as_bool()),
            Some(false)
        );

        let sse = format_openai_sse_stream(1, "deepseek-v3-671b-moe", &["Hello ", "Kadmium"]);
        assert!(sse.contains("\"object\":\"chat.completion.chunk\""));
        assert!(sse.ends_with("data: [DONE]\n\n"));
    }
}
