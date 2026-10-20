//! # `kadmium-gateway`
//!
//! Unified Control Dashboard & Strata-inspired OpenAI/Anthropic compatible HTTP Gateway
//! for **Kadmium (`Cd`)**.

use kadmium_core::{
    HardwareNode, ModelSpec, PoliteState, SupplyLeasePolicy, SupplyMarketplaceSnapshot,
    SupplySettlementEngine, WorkloadKind, sample_personal_cluster,
};
use kadmium_data::{Fts3TransferScheduler, ThirdPartyTransferJob};
use kadmium_network::{CoordinatorFailoverQuorum, PeerLinkMetrics, build_mesh_link_matrix};
use kadmium_rl::{
    DisaggregatedRlPlan, DiscoveryTree, DreamRsiEvolutionReport, RlEnvironmentSuiteSpec,
    RlTopologyPlanner,
};
use kadmium_runtime::{BoincWorkunit, TrainingJobTelemetry, sample_boinc_workunits};
use kadmium_sharding::{
    ClusterShardPlan, FailoverShieldSimulation, MultiProviderArbitragePlan,
    ResponsibilityShiftPlan, ShardPlanner,
};
use serde::{Deserialize, Serialize};

pub mod docs;
pub mod mcp;
pub mod routes;

pub(crate) use docs::DOC_OVERVIEW_MD;
pub use docs::{DocumentationPage, build_documentation_catalog};
pub use mcp::{KadmiumMcpManifest, KadmiumMcpServer, McpJsonRpcRequest, McpJsonRpcResponse};
pub use routes::*;

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
    pub supply_policy: SupplyLeasePolicy,
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
            supply_policy: SupplyLeasePolicy::default(),
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

    pub fn supply_marketplace(&self) -> SupplyMarketplaceSnapshot {
        SupplySettlementEngine::evaluate_cluster_supply(&self.nodes, &self.supply_policy)
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
        assert_eq!(docs.len(), 12);
        assert!(docs.iter().all(|d| !d.markdown.is_empty()));

        // Verify Kadmium MCP Server (JSON-RPC 2.0 initialize, tools/list, tools/call, resources/read)
        let manifest = KadmiumMcpServer::manifest();
        assert_eq!(manifest.tools.len(), 7);
        assert_eq!(manifest.resources.len(), 5);
        assert_eq!(manifest.prompts.len(), 2);

        let supply = state.supply_marketplace();
        assert!(supply.total_hourly_usd > 0.0);
        assert!(supply.total_hourly_cloud_credits_usd > supply.total_hourly_usd);

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
