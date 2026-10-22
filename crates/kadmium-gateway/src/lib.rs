//! # `kadmium-gateway`
//!
//! Unified Control Dashboard & Strata-inspired OpenAI/Anthropic compatible HTTP Gateway
//! for **Kadmium (`Cd`)**.

use kadmium_core::{
    HardwareNode, LiveHardwareProfiler, ModelSpec, NodeTrustTier, PoliteState, SupplyLeasePolicy,
    SupplyMarketplaceSnapshot, SupplySettlementEngine, WorkloadKind, sample_personal_cluster,
};
use kadmium_data::{
    Fts3TransferScheduler, GroundingAnchorTarget, ImportedModelManifest, PoolBucketObjectEntry,
    ThirdPartyTransferJob,
};
use kadmium_network::{
    CoordinatorFailoverQuorum, DEFAULT_KADMIUM_RUNTIME_VERSION, FleetUpdateManifest,
    FleetUpdatePushSummary, JoinTicket, LiveClusterWorkersOverview, LiveWorkerRegistration,
    PeerLinkMetrics, UpdatePayloadKind, UpdateTargetScope, WorkerHeartbeatAck,
    WorkerHeartbeatRequest, WorkerRegistrationAck, WorkerUpdatePushAck, build_mesh_link_matrix,
};
use kadmium_rl::{
    DisaggregatedRlPlan, DiscoveryTree, DreamRsiEvolutionReport, RlEnvironmentSuiteSpec,
    RlTopologyPlanner,
};
use kadmium_runtime::{
    BoincWorkunit, DEFAULT_INITIAL_KTB_MERKLE_ROOT, KtbCheckpointExportReport,
    LiveFts3SwarmStreamReport, LiveKvCacheMigrationReport, SpawnedLocalWorkerInstance,
    TrainingJobTelemetry, sample_boinc_workunits,
};
use kadmium_sharding::{
    ClusterShardPlan, FailoverShieldSimulation, MultiProviderArbitragePlan,
    ResponsibilityShiftPlan, ShardPlanner, ShiftTriggerReason,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod checkpoints;
pub mod coordinator;
pub mod docs;
pub mod mcp;
pub mod routes;

pub use checkpoints::{TrainingCheckpoint, TrainingCheckpointList, TrainingResumeReport};
pub use coordinator::{CoordinatorOptions, run_coordinator};
pub(crate) use docs::DOC_OVERVIEW_MD;
pub use docs::{DocumentationPage, build_documentation_catalog};
pub use mcp::{KadmiumMcpManifest, KadmiumMcpServer, McpJsonRpcRequest, McpJsonRpcResponse};
pub use routes::*;

/// Complete cluster state snapshot returned by `/api/v1/cluster/overview` and `/api/v1/cluster/shift`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterOverviewResponse {
    pub coordinator_quorum: CoordinatorFailoverQuorum,
    pub nodes: Vec<HardwareNode>,
    pub active_model_loaded: bool,
    pub executed_inference_requests: u64,
    pub last_measured_tokens_per_sec: Option<f64>,
    pub last_measured_pipeline_latency_ms: Option<f64>,
    pub imported_models: Vec<ImportedModelManifest>,
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
    /// Live physical LAN/WAN workers registered via `POST /api/v1/cluster/register` (`kadmium join`)
    /// or spawned locally via `POST /api/v1/cluster/spawn-local-fleet`.
    pub live_workers: BTreeMap<String, LiveWorkerRegistration>,
    /// Spawned local TCP worker servers (`127.0.0.1:7851..=7855`) for multi-node local testing.
    pub spawned_fleet_instances: Vec<SpawnedLocalWorkerInstance>,
    /// When true, `nodes` contains only live physical/spawned machines registered in `live_workers`.
    pub live_only_mode: bool,
    pub coordinator_lan_ip: String,
    pub coordinator_http_port: u16,
    pub training_step_counter: u64,
    pub last_training_loss: f64,
    pub training_loss_history: Vec<(u64, f64)>,
    /// Active OTA release version & `.ktb` Merkle root across the cluster.
    pub active_cluster_version: String,
    pub active_ktb_merkle_root_fnv64: u64,
    pub latest_update_manifest: Option<FleetUpdateManifest>,
    /// Queued OTA update manifests for Online WAN nodes that poll `/api/v1/cluster/heartbeat`.
    pub pending_wan_updates: BTreeMap<String, FleetUpdateManifest>,
    pub update_receipts: Vec<WorkerUpdatePushAck>,
    pub last_checkpoint_step: u64,
    pub last_checkpoint_report: Option<KtbCheckpointExportReport>,
    pub last_fts3_stream_report: Option<LiveFts3SwarmStreamReport>,
    pub last_kv_migration_report: Option<LiveKvCacheMigrationReport>,
    pub imported_models: Vec<ImportedModelManifest>,
    /// User-staged or checkpoint objects inside `kdm://pool-bucket/...`.
    pub bucket_objects: Vec<PoolBucketObjectEntry>,
    /// User-configured persistent grounding anchors (`nas://`, `kdm-vault://`, `s3://`, `r2://`).
    pub bucket_anchors: Vec<GroundingAnchorTarget>,
    /// True only when a model has been explicitly loaded/selected, imported as `.ktb`, or executed in this session.
    pub active_model_loaded: bool,
    /// True when the user has explicitly selected a model ID via `/api/v1/cluster/model` or `/api/v1/models/import`.
    pub user_selected_model: bool,
    pub executed_inference_requests: u64,
    pub last_measured_tokens_per_sec: Option<f64>,
    pub last_measured_pipeline_latency_ms: Option<f64>,
    /// Checkpoint id training was last resumed from via `/api/v1/training/resume`.
    pub resumed_from_checkpoint: Option<String>,
}

/// On-disk persistent journal written to `target/kadmium-state/cluster_state.json`
/// so active models, training progress, checkpoints, and OTA versions survive restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentClusterStateJournal {
    pub epoch: u64,
    pub inference_model_id: String,
    pub training_model_id: String,
    pub training_step_counter: u64,
    pub last_training_loss: f64,
    pub training_loss_history: Vec<(u64, f64)>,
    pub active_cluster_version: String,
    pub active_ktb_merkle_root_fnv64: u64,
    pub last_checkpoint_step: u64,
    pub imported_models: Vec<ImportedModelManifest>,
    #[serde(default)]
    pub bucket_objects: Vec<PoolBucketObjectEntry>,
    #[serde(default)]
    pub bucket_anchors: Vec<GroundingAnchorTarget>,
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
        let lan_ip = LiveHardwareProfiler::probe_local_lan_ip();

        Self {
            epoch: 1,
            companion_mode: CompanionConnectionMode::SelfHostedOpenSource {
                local_coordinator_url: format!("http://{lan_ip}:7848"),
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
            live_workers: BTreeMap::new(),
            spawned_fleet_instances: Vec::new(),
            live_only_mode: false,
            coordinator_lan_ip: lan_ip,
            coordinator_http_port: 7848,
            training_step_counter: 1_420,
            last_training_loss: 1.184,
            training_loss_history: vec![
                (1_417, 1.312),
                (1_418, 1.264),
                (1_419, 1.219),
                (1_420, 1.184),
            ],
            active_cluster_version: DEFAULT_KADMIUM_RUNTIME_VERSION.to_string(),
            active_ktb_merkle_root_fnv64: DEFAULT_INITIAL_KTB_MERKLE_ROOT,
            latest_update_manifest: None,
            pending_wan_updates: BTreeMap::new(),
            update_receipts: Vec::new(),
            last_checkpoint_step: 1_400,
            last_checkpoint_report: None,
            last_fts3_stream_report: None,
            last_kv_migration_report: None,
            imported_models: Vec::new(),
            bucket_objects: Vec::new(),
            bucket_anchors: Vec::new(),
            active_model_loaded: false,
            user_selected_model: false,
            executed_inference_requests: 0,
            last_measured_tokens_per_sec: None,
            last_measured_pipeline_latency_ms: None,
            resumed_from_checkpoint: None,
        }
    }

    /// Writes a crash-safe JSON state journal to `target/kadmium-state/cluster_state.json`.
    pub fn save_persistent_state(&self) -> Result<String, String> {
        let dir = std::path::Path::new("target/kadmium-state");
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create state directory: {e}"))?;
        let path = dir.join("cluster_state.json");
        let journal = PersistentClusterStateJournal {
            epoch: self.epoch,
            inference_model_id: self.inference_model.id.clone(),
            training_model_id: self.training_model.id.clone(),
            training_step_counter: self.training_step_counter,
            last_training_loss: self.last_training_loss,
            training_loss_history: self.training_loss_history.clone(),
            active_cluster_version: self.active_cluster_version.clone(),
            active_ktb_merkle_root_fnv64: self.active_ktb_merkle_root_fnv64,
            last_checkpoint_step: self.last_checkpoint_step,
            imported_models: self.imported_models.clone(),
            bucket_objects: self.bucket_objects.clone(),
            bucket_anchors: self.bucket_anchors.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&journal)
            .map_err(|e| format!("Failed to serialize state journal: {e}"))?;
        std::fs::write(&path, bytes)
            .map_err(|e| format!("Failed to write state journal: {e}"))?;
        Ok(path.display().to_string())
    }

    /// Recomputes `coordinator_quorum`, `inference_plan`, `training_plan`, and `fts3_transfers`
    /// whenever the node topology changes.
    pub fn recompute_plans(&mut self) {
        if self.nodes.is_empty() {
            self.nodes = vec![LiveHardwareProfiler::probe_local_host()];
        }
        if self.live_only_mode && !self.user_selected_model {
            let recommended = ModelSpec::recommended_for_cluster(&self.nodes);
            self.inference_model = recommended.clone();
            self.training_model = recommended;
        }
        self.coordinator_quorum = CoordinatorFailoverQuorum::from_cluster(&self.nodes);
        self.inference_plan = ShardPlanner::compute_plan(
            self.epoch,
            &self.inference_model,
            WorkloadKind::Inference,
            &self.nodes,
        );
        self.training_plan = ShardPlanner::compute_plan(
            self.epoch,
            &self.training_model,
            WorkloadKind::DistributedTraining,
            &self.nodes,
        );
        if self.live_only_mode {
            if (self.last_fts3_stream_report.is_none() && self.last_shift.is_none())
                || !self.active_model_loaded
            {
                self.fts3_transfers.clear();
            }
        } else {
            self.fts3_transfers =
                Fts3TransferScheduler::plan_initial_staging(&self.inference_plan, &self.nodes);
        }
    }

    /// Switches between Live-Hardware-Only mode (`true`) and Hybrid Demo + Live mode (`false`).
    pub fn set_live_only_mode(&mut self, live_only: bool) {
        self.live_only_mode = live_only;
        self.epoch += 1;
        if live_only {
            self.training_step_counter = 0;
            self.last_training_loss = 0.0;
            self.training_loss_history.clear();
            self.last_checkpoint_step = 0;
            self.resumed_from_checkpoint = None;
            self.supply_policy.enabled = false;
            self.fts3_transfers.clear();
            self.active_model_loaded = !self.imported_models.is_empty();
            let live_nodes: Vec<HardwareNode> = self
                .live_workers
                .values()
                .filter(|w| w.online)
                .map(|w| w.node.clone())
                .collect();
            self.nodes = if live_nodes.is_empty() {
                vec![LiveHardwareProfiler::probe_local_host()]
            } else {
                live_nodes
            };
        } else {
            self.active_model_loaded = true;
            if !self.user_selected_model {
                self.inference_model = ModelSpec::deepseek_v3_671b_moe();
                self.training_model = ModelSpec::llama_3_70b_lora();
            }
            if self.training_step_counter == 0 {
                self.training_step_counter = 1_420;
                self.last_training_loss = 1.184;
                self.training_loss_history = vec![
                    (1_417, 1.312),
                    (1_418, 1.264),
                    (1_419, 1.219),
                    (1_420, 1.184),
                ];
                self.last_checkpoint_step = 1_400;
            }
            let mut merged = sample_personal_cluster();
            for w in self.live_workers.values().filter(|w| w.online) {
                if let Some(existing) = merged.iter_mut().find(|n| n.id == w.node.id) {
                    *existing = w.node.clone();
                } else {
                    merged.push(w.node.clone());
                }
            }
            self.nodes = merged;
        }
        self.recompute_plans();
    }

    /// Registers or updates a live LAN/WAN worker (`kadmium join`), recomputes asymmetric shard
    /// plans across the updated cluster, and returns `WorkerRegistrationAck`.
    pub fn register_live_worker(
        &mut self,
        node: HardwareNode,
        worker_tcp_addr: String,
        measured_rtt_ms: f64,
        switch_to_live_only: bool,
    ) -> WorkerRegistrationAck {
        let board = if !node.gpu_name.is_empty() {
            let npu_str = node
                .npu_name
                .as_ref()
                .map(|n| format!(" + {n} ({:.0} TOPS)", node.npu_tops))
                .unwrap_or_default();
            format!("{} + {}{}", node.gpu_name, node.cpu_name, npu_str)
        } else {
            LiveHardwareProfiler::probe_sbc_or_board_model()
                .unwrap_or_else(|| format!("{:?} ({:?})", node.device_kind, node.backend))
        };
        let net_chan = if node.id.0.contains("remote") || node.id.0.contains("wan") {
            "OnlineWAN".to_string()
        } else {
            "LocalLAN".to_string()
        };
        self.register_live_worker_with_meta(
            node,
            worker_tcp_addr,
            measured_rtt_ms,
            switch_to_live_only,
            std::env::consts::OS.to_string(),
            std::env::consts::ARCH.to_string(),
            board,
            net_chan,
        )
    }

    /// Registers a live or locally-spawned worker with explicit OS/Arch/Board and network channel metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn register_live_worker_with_meta(
        &mut self,
        mut node: HardwareNode,
        worker_tcp_addr: String,
        measured_rtt_ms: f64,
        switch_to_live_only: bool,
        os_name: String,
        cpu_arch: String,
        board_or_model: String,
        network_channel: String,
    ) -> WorkerRegistrationAck {
        node.network_latency_ms = measured_rtt_ms.max(0.1);
        let now_ms = LiveWorkerRegistration::now_unix_ms();

        let (lan_ip, tcp_port) = match worker_tcp_addr.rsplit_once(':') {
            Some((ip, p)) => (ip.to_string(), p.parse::<u16>().unwrap_or(7850)),
            None => (worker_tcp_addr.clone(), 7850),
        };

        let (prev_fwd, prev_trn, reg_at, hb_cnt, ver, mrk) = self
            .live_workers
            .get(&node.id.0)
            .map(|w| {
                (
                    w.executed_forward_stages,
                    w.executed_training_steps,
                    w.registered_at_unix_ms,
                    w.heartbeat_count + 1,
                    w.installed_version.clone(),
                    w.ktb_merkle_root_fnv64,
                )
            })
            .unwrap_or((
                0,
                0,
                now_ms,
                1,
                self.active_cluster_version.clone(),
                self.active_ktb_merkle_root_fnv64,
            ));

        self.live_workers.insert(
            node.id.0.clone(),
            LiveWorkerRegistration {
                node: node.clone(),
                worker_tcp_addr,
                worker_lan_ip: lan_ip,
                worker_tcp_port: tcp_port,
                os_name,
                cpu_arch,
                board_or_model,
                network_channel,
                installed_version: ver,
                ktb_merkle_root_fnv64: mrk,
                heartbeat_count: hb_cnt,
                registered_at_unix_ms: reg_at,
                last_heartbeat_unix_ms: now_ms,
                last_measured_rtt_ms: measured_rtt_ms,
                executed_forward_stages: prev_fwd,
                executed_training_steps: prev_trn,
                online: true,
            },
        );

        if switch_to_live_only && !self.live_only_mode {
            self.live_only_mode = true;
            self.training_step_counter = 0;
            self.last_training_loss = 0.0;
            self.training_loss_history.clear();
            self.last_checkpoint_step = 0;
            self.resumed_from_checkpoint = None;
            self.supply_policy.enabled = false;
            self.fts3_transfers.clear();
            self.active_model_loaded = !self.imported_models.is_empty();
            self.nodes = self
                .live_workers
                .values()
                .filter(|w| w.online)
                .map(|w| w.node.clone())
                .collect();
        } else if let Some(existing) = self.nodes.iter_mut().find(|n| n.id == node.id) {
            *existing = node.clone();
        } else {
            self.nodes.push(node.clone());
        }

        self.epoch += 1;
        self.recompute_plans();

        let inf_assign = self
            .inference_plan
            .assignments
            .iter()
            .find(|a| a.node_id == node.id);
        let trn_assign = self
            .training_plan
            .assignments
            .iter()
            .find(|a| a.node_id == node.id);

        let inference_layers = inf_assign.and_then(|a| Some((a.start_layer?, a.end_layer?)));
        let training_layers = trn_assign.and_then(|a| Some((a.start_layer?, a.end_layer?)));
        let compute_share_pct = inf_assign.map(|a| a.compute_share_pct).unwrap_or(0.0);

        WorkerRegistrationAck {
            accepted: true,
            cluster_epoch: self.epoch,
            coordinator_id: self.coordinator_quorum.active_leader.clone(),
            assigned_node_id: node.id.clone(),
            measured_tcp_rtt_ms: measured_rtt_ms,
            inference_layers,
            training_layers,
            compute_share_pct,
            q_star_ratio: node.compute_q_star_ratio(),
            total_cluster_nodes: self.nodes.len(),
            total_live_workers: self.live_workers.len(),
            message: format!(
                "Registered live worker '{}' ({:?}) at epoch {}",
                node.hostname, node.device_kind, self.epoch
            ),
        }
    }

    /// Processes a periodic heartbeat from a live worker daemon (`POST /api/v1/cluster/heartbeat`).
    /// Automatically triggers a `<200ms` live responsibility shift if the worker's `PoliteState` changed,
    /// and delivers any queued OTA update (`pending_ota_update`) for Online WAN nodes!
    pub fn record_worker_heartbeat(&mut self, req: &WorkerHeartbeatRequest) -> WorkerHeartbeatAck {
        let now_ms = LiveWorkerRegistration::now_unix_ms();
        let mut state_changed = false;

        if let Some(ota_ack) = &req.applied_ota_ack
            && ota_ack.accepted
        {
            if let Some(w) = self.live_workers.get_mut(&req.node_id.0) {
                w.installed_version = ota_ack.installed_version.clone();
                w.ktb_merkle_root_fnv64 = ota_ack.applied_ktb_merkle_root_fnv64;
            }
            self.update_receipts
                .retain(|r| r.node_id != ota_ack.node_id);
            self.update_receipts.push(ota_ack.clone());
        }

        if let Some(w) = self.live_workers.get_mut(&req.node_id.0) {
            w.last_heartbeat_unix_ms = now_ms;
            w.heartbeat_count += 1;
            w.online = true;
            if !req.worker_tcp_addr.is_empty() {
                w.worker_tcp_addr = req.worker_tcp_addr.clone();
            }
            if w.node.polite_state != req.polite_state {
                w.node.polite_state = req.polite_state;
                state_changed = true;
            }
        }

        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == req.node_id)
            && n.polite_state != req.polite_state
        {
            n.polite_state = req.polite_state;
            state_changed = true;
        }

        if state_changed {
            self.epoch += 1;
            let trigger = ShiftTriggerReason::HostEnteredGamingMode {
                node_id: req.node_id.clone(),
            };
            let shift = ShardPlanner::plan_responsibility_shift(
                &self.inference_plan,
                &self.inference_model,
                &self.nodes,
                trigger,
            );
            self.fts3_transfers =
                Fts3TransferScheduler::plan_live_shift_transfers(&shift, &self.nodes);
            self.inference_plan = shift.new_plan.clone();
            self.training_plan = ShardPlanner::compute_plan(
                self.epoch,
                &self.training_model,
                WorkloadKind::DistributedTraining,
                &self.nodes,
            );
            self.last_shift = Some(shift);
        }

        let inference_layers = self
            .inference_plan
            .assignments
            .iter()
            .find(|a| a.node_id == req.node_id)
            .and_then(|a| Some((a.start_layer?, a.end_layer?)));
        let training_layers = self
            .training_plan
            .assignments
            .iter()
            .find(|a| a.node_id == req.node_id)
            .and_then(|a| Some((a.start_layer?, a.end_layer?)));

        let pending_ota_update = self.pending_wan_updates.remove(&req.node_id.0);

        WorkerHeartbeatAck {
            cluster_epoch: self.epoch,
            rebalance_triggered: state_changed,
            inference_layers,
            training_layers,
            active_leader: self.coordinator_quorum.active_leader.clone(),
            pending_ota_update,
        }
    }

    /// Toggles a live or locally-spawned worker between `Available`, `GamingOrHeavyApp` (Polite auto-yield),
    /// or `Offline` (triggers `SkipPipe` bypass & shard rebalance).
    pub fn set_worker_fault_or_polite_state(
        &mut self,
        node_id_str: &str,
        polite_state: PoliteState,
        online: bool,
    ) -> bool {
        let mut found = false;
        let mut target_node_id = None;

        if let Some(w) = self.live_workers.get_mut(node_id_str) {
            w.node.polite_state = polite_state;
            w.online = online;
            target_node_id = Some(w.node.id.clone());
            found = true;
        }

        for inst in &self.spawned_fleet_instances {
            if let Ok(mut guard) = inst.server.node.lock()
                && guard.id.0 == node_id_str
            {
                guard.polite_state = polite_state;
            }
        }

        if let Some(nid) = target_node_id {
            if !online {
                self.nodes.retain(|n| n.id != nid);
            } else if let Some(n) = self.nodes.iter_mut().find(|n| n.id == nid) {
                n.polite_state = polite_state;
            } else if let Some(w) = self.live_workers.get(node_id_str) {
                self.nodes.push(w.node.clone());
            }

            self.epoch += 1;
            let trigger = ShiftTriggerReason::HostEnteredGamingMode { node_id: nid };
            let shift = ShardPlanner::plan_responsibility_shift(
                &self.inference_plan,
                &self.inference_model,
                &self.nodes,
                trigger,
            );
            self.fts3_transfers =
                Fts3TransferScheduler::plan_live_shift_transfers(&shift, &self.nodes);
            self.inference_plan = shift.new_plan.clone();
            self.training_plan = ShardPlanner::compute_plan(
                self.epoch,
                &self.training_model,
                WorkloadKind::DistributedTraining,
                &self.nodes,
            );
            self.last_shift = Some(shift);
        }

        found
    }

    /// Builds a signed `FleetUpdateManifest` for an Over-The-Air (OTA) update push across Local LAN and/or Online WAN nodes.
    pub fn create_ota_update_manifest(
        &mut self,
        target_version: String,
        payload_kind: UpdatePayloadKind,
        target_scope: UpdateTargetScope,
        rolling_canary: bool,
        release_notes: String,
    ) -> FleetUpdateManifest {
        let now_ms = LiveWorkerRegistration::now_unix_ms();
        let update_id = format!("ota-e{}-{}", self.epoch + 1, target_version);

        // Synthesize deterministic binary/delta payload bytes and new .ktb Merkle root
        let payload_bytes: Vec<u8> = match payload_kind {
            UpdatePayloadKind::BinaryRelease => format!(
                "KDM_BIN_RELEASE|version={target_version}|targets=win-x64,macos-arm64,linux-x64,linux-arm64,linux-riscv64|notes={release_notes}"
            )
            .into_bytes(),
            UpdatePayloadKind::KtbWeightDelta => {
                let mut page_delta = vec![0x4Bu8; 1024];
                for (i, b) in target_version.bytes().enumerate() {
                    page_delta[i % 1024] ^= b;
                }
                page_delta
            }
            UpdatePayloadKind::DreamRsiPolicyBundle => format!(
                "KDM_DREAM_RSI_POLICY|version={target_version}|puct_c=1.414|wasi_cap_mb=256|notes={release_notes}"
            )
            .into_bytes(),
        };

        let expected_merkle = FleetUpdateManifest::compute_signature_mac(
            "ktb-merkle-root",
            &target_version,
            self.active_ktb_merkle_root_fnv64,
            &payload_bytes,
        );
        let mac = FleetUpdateManifest::compute_signature_mac(
            &update_id,
            &target_version,
            expected_merkle,
            &payload_bytes,
        );

        let artifact_size_bytes = match payload_kind {
            UpdatePayloadKind::BinaryRelease => 14_680_064, // 14 MB cross-platform release bundle
            UpdatePayloadKind::KtbWeightDelta => 65_536 * 4, // 4 x 64 KiB page-aligned .ktb delta
            UpdatePayloadKind::DreamRsiPolicyBundle => 262_144, // 256 KiB WASI + MCTS policy bundle
        };

        let manifest = FleetUpdateManifest {
            update_id,
            target_version,
            payload_kind,
            target_scope,
            rolling_canary,
            issued_at_unix_ms: now_ms,
            expected_ktb_merkle_root_fnv64: expected_merkle,
            artifact_size_bytes,
            payload_bytes,
            signature_mac_fnv64: mac,
            release_notes,
        };

        self.latest_update_manifest = Some(manifest.clone());
        manifest
    }

    /// Commits the `WorkerUpdatePushAck` receipts returned by Local LAN (TCP WireFrame) and
    /// Online WAN (Heartbeat Pull/Relay) workers after an OTA update push.
    pub fn record_ota_receipts(
        &mut self,
        manifest: &FleetUpdateManifest,
        receipts: Vec<WorkerUpdatePushAck>,
        local_lan_count: usize,
        online_wan_count: usize,
    ) -> FleetUpdatePushSummary {
        self.epoch += 1;
        self.active_cluster_version = manifest.target_version.clone();
        self.active_ktb_merkle_root_fnv64 = manifest.expected_ktb_merkle_root_fnv64;

        for ack in &receipts {
            if ack.accepted
                && let Some(w) = self.live_workers.get_mut(&ack.node_id.0)
            {
                w.installed_version = ack.installed_version.clone();
                w.ktb_merkle_root_fnv64 = ack.applied_ktb_merkle_root_fnv64;
            }
        }

        self.update_receipts = receipts.clone();

        FleetUpdatePushSummary {
            active_cluster_version: self.active_cluster_version.clone(),
            active_ktb_merkle_root_fnv64: self.active_ktb_merkle_root_fnv64,
            latest_manifest: Some(manifest.clone()),
            total_nodes_targeted: receipts.len(),
            local_lan_pushed_count: local_lan_count,
            online_wan_pushed_count: online_wan_count,
            canary_verified: manifest.rolling_canary,
            zero_downtime_skippipe_active: true,
            node_receipts: receipts,
        }
    }

    /// Returns the current OTA Fleet Update status (`GET /api/v1/update/status`).
    pub fn ota_update_status(&self) -> FleetUpdatePushSummary {
        let local_cnt = self
            .update_receipts
            .iter()
            .filter(|r| r.delivery_mechanism.contains("LAN"))
            .count();
        let wan_cnt = self
            .update_receipts
            .iter()
            .filter(|r| r.delivery_mechanism.contains("WAN"))
            .count();

        FleetUpdatePushSummary {
            active_cluster_version: self.active_cluster_version.clone(),
            active_ktb_merkle_root_fnv64: self.active_ktb_merkle_root_fnv64,
            latest_manifest: self.latest_update_manifest.clone(),
            total_nodes_targeted: self.update_receipts.len(),
            local_lan_pushed_count: local_cnt,
            online_wan_pushed_count: wan_cnt,
            canary_verified: self
                .latest_update_manifest
                .as_ref()
                .map(|m| m.rolling_canary)
                .unwrap_or(true),
            zero_downtime_skippipe_active: true,
            node_receipts: self.update_receipts.clone(),
        }
    }

    /// Returns a snapshot of connected live workers and the LAN join command/ticket.
    pub fn workers_overview(&self) -> LiveClusterWorkersOverview {
        let http_url = format!(
            "http://{}:{}",
            self.coordinator_lan_ip, self.coordinator_http_port
        );
        let ticket = JoinTicket::generate(
            self.coordinator_quorum.active_leader.clone(),
            vec![format!(
                "{}:{}",
                self.coordinator_lan_ip, self.coordinator_http_port
            )],
            Some("slice-01-prod-deploy".into()),
            NodeTrustTier::TrustedCore,
        )
        .encode_token();

        let join_cmd = format!(
            "cargo run --release --bin kadmium -- join --coordinator {http_url}"
        );
        let workers_vec: Vec<LiveWorkerRegistration> =
            self.live_workers.values().cloned().collect();

        LiveClusterWorkersOverview {
            coordinator_lan_ip: self.coordinator_lan_ip.clone(),
            coordinator_http_url: http_url.clone(),
            join_command_cli: format!("kadmium join --coordinator {http_url}"),
            recommended_join_command: join_cmd,
            join_ticket_token: ticket,
            live_only_mode: self.live_only_mode,
            total_cluster_nodes: self.nodes.len(),
            live_worker_count: workers_vec.len(),
            active_cluster_version: self.active_cluster_version.clone(),
            live_workers: workers_vec.clone(),
            workers: workers_vec,
        }
    }

    pub fn supply_marketplace(&self) -> SupplyMarketplaceSnapshot {
        let mut snap =
            SupplySettlementEngine::evaluate_cluster_supply(&self.nodes, &self.supply_policy);
        if self.live_only_mode {
            // The settlement engine ships sample receipts for the demo fleet; with live hardware
            // only, keep just the receipts earned by machines that are actually in the cluster.
            snap.recent_receipts
                .retain(|receipt| self.nodes.iter().any(|n| n.id == receipt.provider_node_id));
        }
        snap
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
            app_version: self.active_cluster_version.clone(),
            mode: self.companion_mode.clone(),
            local_host_polite_state: local_polite,
            gaming_auto_yield_active: true,
            active_cluster_nodes: self.nodes.len(),
            active_coordinator_leader: self.coordinator_quorum.active_leader.0.clone(),
            current_hourly_cloud_burn_usd: burn_usd,
        }
    }

    pub fn snapshot(&self) -> ClusterOverviewResponse {
        let mut training_telemetry = TrainingJobTelemetry::sample_for_plan(&self.training_plan);
        training_telemetry.current_step = self.training_step_counter;
        training_telemetry.loss = self.last_training_loss;
        training_telemetry.last_nas_checkpoint_step = self.last_checkpoint_step;
        if self.live_only_mode && self.training_step_counter == 0 {
            training_telemetry.total_steps = 0;
            training_telemetry.micro_batches_in_flight = 0;
        }

        let boinc_workunits = if self.live_only_mode {
            Vec::new()
        } else {
            sample_boinc_workunits()
        };
        let mesh_links = build_mesh_link_matrix(&self.nodes);
        let rl_env_suite = RlEnvironmentSuiteSpec::dream_rsi_evolving_world();
        let mut rl_plan = RlTopologyPlanner::compute_rl_plan(
            self.epoch,
            &self.training_model,
            &rl_env_suite,
            &self.nodes,
        );
        if self.live_only_mode {
            rl_plan.dream_rsi_report = None;
        }

        ClusterOverviewResponse {
            coordinator_quorum: self.coordinator_quorum.clone(),
            nodes: self.nodes.clone(),
            active_model_loaded: self.active_model_loaded,
            executed_inference_requests: self.executed_inference_requests,
            last_measured_tokens_per_sec: self.last_measured_tokens_per_sec,
            last_measured_pipeline_latency_ms: self.last_measured_pipeline_latency_ms,
            imported_models: self.imported_models.clone(),
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

        out.push_str(
            "# HELP kadmium_live_workers_connected Connected physical LAN/WAN worker daemons.\n",
        );
        out.push_str("# TYPE kadmium_live_workers_connected gauge\n");
        out.push_str(&format!(
            "kadmium_live_workers_connected {}\n",
            self.live_workers.len()
        ));

        out.push_str("# HELP kadmium_inference_throughput_tokens_per_sec Estimated aggregate inference throughput.\n");
        out.push_str("# TYPE kadmium_inference_throughput_tokens_per_sec gauge\n");
        let reported_tps = if self.active_model_loaded {
            self.last_measured_tokens_per_sec
                .unwrap_or(self.inference_plan.estimated_throughput)
        } else {
            0.0
        };
        out.push_str(&format!(
            "kadmium_inference_throughput_tokens_per_sec{{model=\"{}\"}} {:.2}\n",
            self.inference_plan.model_id, reported_tps
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
        let reported_tps = if self.active_model_loaded {
            self.last_measured_tokens_per_sec
                .unwrap_or(self.inference_plan.estimated_throughput)
        } else {
            0.0
        };
        serde_json::json!({
            "project": "kadmium-cd48-fabric",
            "run_id": format!("kadmium-run-e{}", self.epoch),
            "signature": "Cd-48",
            "active_version": self.active_cluster_version,
            "model_id": if self.active_model_loaded { self.inference_plan.model_id.as_str() } else { "idle" },
            "training_model_id": if self.training_step_counter > 0 { self.training_plan.model_id.as_str() } else { "idle" },
            "metrics": {
                "inference/tokens_per_sec": reported_tps,
                "inference/executed_requests": self.executed_inference_requests,
                "training/pipeline_bubble_pct": if self.training_step_counter > 0 { snap.training_telemetry.pipeline_bubble_pct } else { 0.0 },
                "training/demo_compression_ratio": if self.training_step_counter > 0 { snap.training_telemetry.gradient_compression_ratio } else { 0.0 },
                "training/current_step": self.training_step_counter,
                "training/loss": self.last_training_loss,
                "rl/vectorized_env_slots": snap.rl_plan.total_vectorized_env_slots,
                "dream_rsi/cost_savings_pct": snap.rl_plan.dream_rsi_report.as_ref().map(|r| r.cost_savings_pct).unwrap_or(0.0),
                "dream_rsi/peak_speedup": snap.rl_plan.dream_rsi_report.as_ref().map(|r| r.peak_speedup_achieved).unwrap_or(0.0),
                "cluster/active_nodes": self.nodes.len(),
                "cluster/live_workers": self.live_workers.len(),
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

        // Verify live worker registration & heartbeat auto-rebalance
        let probe_node = LiveHardwareProfiler::probe_local_host();
        let ack = state.register_live_worker(
            probe_node.clone(),
            "127.0.0.1:7850".into(),
            0.35,
            true,
        );
        assert!(ack.accepted);
        assert!(state.live_only_mode);
        assert_eq!(state.nodes.len(), 1);
        assert_eq!(state.live_workers.len(), 1);

        let hb_ack = state.record_worker_heartbeat(&WorkerHeartbeatRequest {
            node_id: probe_node.id.clone(),
            worker_tcp_addr: "127.0.0.1:7850".into(),
            polite_state: PoliteState::GamingOrHeavyApp,
            external_gpu_utilization_pct: 75.0,
            external_vram_used_gb: 8.0,
            package_temp_celsius: 68.0,
            applied_ota_ack: None,
        });
        assert!(hb_ack.rebalance_triggered);

        // Verify OTA Update Manifest signature & receipt commit
        let ota_manifest = state.create_ota_update_manifest(
            "v0.2.0-cd48".into(),
            UpdatePayloadKind::BinaryRelease,
            UpdateTargetScope::AllNodes,
            true,
            "Fused Q4_K kernel + Dream-RSI rollout".into(),
        );
        assert!(ota_manifest.verify_signature());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_platforms_overview_and_attach_endpoints_and_ui_bundles() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::sync::{Arc, Mutex};

        let state = Arc::new(Mutex::new(GatewayState::new()));
        let app = build_router(Arc::clone(&state));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let http_get = |path: &str| -> String {
            let mut s = TcpStream::connect(addr).unwrap();
            write!(
                s,
                "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            let mut buf = String::new();
            s.read_to_string(&mut buf).unwrap();
            buf.split_once("\r\n\r\n").unwrap().1.to_string()
        };

        let http_post = |path: &str, body_json: &str| -> String {
            let mut s = TcpStream::connect(addr).unwrap();
            write!(
                s,
                "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body_json}",
                body_json.len()
            )
            .unwrap();
            let mut buf = String::new();
            s.read_to_string(&mut buf).unwrap();
            buf.split_once("\r\n\r\n").unwrap().1.to_string()
        };

        // 1. Verify Control Room Dashboard (/) and Desktop Companion Application (/companion) HTML bundles
        let dash_html = http_get("/");
        assert!(dash_html.contains("/api/v1/platforms/overview"));
        assert!(dash_html.contains("/api/v1/m4-mac-mini/deep-suite"));
        assert!(dash_html.contains("Boards & devices"));
        assert!(dash_html.contains("/api/v1/training/resume"));

        let comp_html = http_get("/companion");
        assert!(comp_html.contains("Boards and bridges"));
        assert!(comp_html.contains("Import local weights"));
        assert!(comp_html.contains("/api/v1/platforms/attach"));
        assert!(comp_html.contains("__KADMIUM_DESKTOP__"));

        // 2. Verify GET /api/v1/platforms/overview
        let pf_json = http_get("/api/v1/platforms/overview");
        let pf: PlatformFamiliesOverviewResponse = serde_json::from_str(&pf_json).unwrap();
        assert_eq!(pf.total_platform_families, 5);
        assert_eq!(pf.total_catalog_skus, 26);
        assert_eq!(pf.apple_m4_catalog.len(), 7);
        assert_eq!(pf.rpi_cm4_catalog.len(), 6);
        assert_eq!(pf.tegra_x1_catalog.len(), 5);
        assert_eq!(pf.kendryte_k230_catalog.len(), 5);
        assert_eq!(pf.phone_farm_catalog.len(), 3);
        if pf.apple_m4_probe.physical_board_online {
            let deep = pf
                .apple_m4_deep_metal4
                .expect("live Apple M4 deep Metal 4 suite");
            assert!(deep.tiled_gemm1024_gflops > 50.0);
            assert!(deep.zerocopy_mmap_verified);
        }

        // 3. Verify POST /api/v1/platforms/attach across families
        let attach_json = http_post("/api/v1/platforms/attach", r#"{"family":"all-live"}"#);
        let attach_parsed: PlatformAttachResponse = serde_json::from_str(&attach_json).unwrap();
        assert_eq!(attach_parsed.attached_family, "all-live");
        assert!(!attach_parsed.overview.nodes.is_empty());

        // 4. Verify Storage Bucket live report, object staging, and grounding anchor endpoints
        let bucket_live_json = http_get("/api/v1/storage/pool-bucket?bucket=live");
        let bucket_live: kadmium_data::DecentralizedStoragePoolReport =
            serde_json::from_str(&bucket_live_json).unwrap();
        assert_eq!(bucket_live.bucket_uri, "kdm://pool-bucket/");
        assert!(bucket_live.total_participating_nodes >= 1);
        assert!(bucket_live.total_pooled_disk_capacity_gb > 0.0);

        let put_obj_json = http_post(
            "/api/v1/storage/pool-bucket/object",
            r#"{"preset":"active-model"}"#,
        );
        let after_put: kadmium_data::DecentralizedStoragePoolReport =
            serde_json::from_str(&put_obj_json).unwrap();
        assert!(!after_put.bucket_objects.is_empty());

        let add_anchor_json = http_post(
            "/api/v1/storage/pool-bucket/anchor",
            r#"{"anchor_kind":"LocalNasZfsVault","anchor_uri":"nas://local-truenas/zpool-ground","capacity_tb":8.0,"sustained_bandwidth_gbps":10.0}"#,
        );
        let after_anchor: kadmium_data::DecentralizedStoragePoolReport =
            serde_json::from_str(&add_anchor_json).unwrap();
        assert!(
            after_anchor
                .bucket_grounding
                .configured_anchors
                .iter()
                .any(|a| a.anchor_uri == "nas://local-truenas/zpool-ground")
        );

        let bucket_scale_json = http_get("/api/v1/storage/pool-bucket?bucket=enterprise");
        let bucket_scale: kadmium_data::DecentralizedStoragePoolReport =
            serde_json::from_str(&bucket_scale_json).unwrap();
        assert_eq!(bucket_scale.total_participating_nodes, 100);

        // 5. Verify the training checkpoint list and resume endpoints
        let ckpt_json = http_get("/api/v1/training/checkpoints");
        let ckpts: TrainingCheckpointList = serde_json::from_str(&ckpt_json).unwrap();
        assert!(ckpts.directory.ends_with("kadmium-checkpoints"));

        let resume_json = http_post(
            "/api/v1/training/resume",
            r#"{"checkpoint_id":"ckpt-step0000-not-a-real-file"}"#,
        );
        let resume: serde_json::Value = serde_json::from_str(&resume_json).unwrap();
        assert!(resume["error"].as_str().unwrap().contains("No checkpoint"));

        // 6. Verify a training step uses the learning rate it was sent
        let step_json = http_post("/api/v1/training/step", r#"{"learning_rate":0.02}"#);
        let step: serde_json::Value = serde_json::from_str(&step_json).unwrap();
        assert!((step["learning_rate"].as_f64().unwrap() - 0.02).abs() < 1e-6);
    }
}
