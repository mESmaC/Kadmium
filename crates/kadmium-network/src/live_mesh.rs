//! # `kadmium_network::live_mesh`
//!
//! Live LAN/WAN worker registration, UDP zero-config coordinator auto-discovery (`KDMD`),
//! Over-The-Air (OTA) Fleet Update Pushing (both direct TCP `WireFrame` push for local LAN
//! nodes and heartbeat-queued pull/relay for online WAN nodes), and lightweight async HTTP JSON
//! client over `tokio::net::TcpStream`.

use crate::LAN_BEACON_MAGIC;
use kadmium_core::{HardwareNode, NodeId, PoliteState};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Default UDP port used by `kadmium serve` and `kadmium join` for zero-config LAN discovery.
pub const DEFAULT_LAN_DISCOVERY_UDP_PORT: u16 = 7849;

/// Default base TCP port for a `LiveNodeWorkerServer` listening for `WireFrame` pipeline hops.
pub const DEFAULT_WORKER_TCP_PORT: u16 = 7850;

/// Default initial runtime version reported by nodes on startup.
pub const DEFAULT_KADMIUM_RUNTIME_VERSION: &str = "v0.1.0-cd48";

/// Magic probe payload sent inside a `"KDMD"` UDP datagram when a worker searches for a coordinator.
pub const LAN_DISCOVER_COORDINATOR_MAGIC: &[u8] = b"KDMP_DISCOVER_COORDINATOR_V1";

/// Announcement returned over UDP by a running `kadmium serve` coordinator when a worker
/// broadcasts a discovery probe on the LAN.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanCoordinatorAnnouncement {
    pub coordinator_id: NodeId,
    pub coordinator_hostname: String,
    pub coordinator_http_url: String,
    pub join_ticket_token: String,
    pub active_model_id: String,
    pub cluster_epoch: u64,
}

/// Target scope for an Over-The-Air (OTA) Fleet Update push.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateTargetScope {
    /// Push to both Local LAN nodes (via direct TCP `WireFrame`) and Online WAN nodes (via Heartbeat Pull/Relay).
    AllNodes,
    /// Push strictly to Local LAN nodes over direct TCP `WireFrame` streams.
    LocalLanOnly,
    /// Push strictly to Online / WAN nodes via Heartbeat manifest queuing and chunked HTTP/DERP pull.
    OnlineWanOnly,
}

/// Kind of artifact delivered by the Fleet Update Pushing system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdatePayloadKind {
    /// Cross-platform `kadmium` worker binary / runtime hot-reload (`windows/x86_64`, `macos/aarch64`, `linux/aarch64`, `linux/riscv64`, `linux/x86_64`).
    BinaryRelease,
    /// 64 KiB page-aligned `.ktb` weight bank & QLoRA adapter delta patch verified via `KtbMerkleTree`.
    KtbWeightDelta,
    /// Updated `Dream-RSI` MCTS exploration policy & WASI verifier sandbox rules.
    DreamRsiPolicyBundle,
}

/// Signed OTA Update Manifest pushed over `WireFrameKind::FleetUpdatePush` (for Local LAN nodes)
/// or embedded in `WorkerHeartbeatAck` / `GET /api/v1/update/artifact` (for Online WAN nodes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetUpdateManifest {
    pub update_id: String,
    pub target_version: String,
    pub payload_kind: UpdatePayloadKind,
    pub target_scope: UpdateTargetScope,
    /// If true, verifies the first node (canary) and its Verde RepOps digest before rolling out to remaining stages.
    pub rolling_canary: bool,
    pub issued_at_unix_ms: u64,
    pub expected_ktb_merkle_root_fnv64: u64,
    pub artifact_size_bytes: usize,
    pub payload_bytes: Vec<u8>,
    pub signature_mac_fnv64: u64,
    pub release_notes: String,
}

impl FleetUpdateManifest {
    /// Computes a deterministic 64-bit FNV-1a MAC signature over the update manifest header and payload.
    pub fn compute_signature_mac(
        update_id: &str,
        target_version: &str,
        expected_merkle_root: u64,
        payload_bytes: &[u8],
    ) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for b in update_id
            .as_bytes()
            .iter()
            .chain(target_version.as_bytes().iter())
            .chain(expected_merkle_root.to_le_bytes().iter())
            .chain(payload_bytes.iter())
        {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub fn verify_signature(&self) -> bool {
        let expected = Self::compute_signature_mac(
            &self.update_id,
            &self.target_version,
            self.expected_ktb_merkle_root_fnv64,
            &self.payload_bytes,
        );
        self.signature_mac_fnv64 == expected
    }
}

/// Receipt returned by a worker (over TCP `WireFrame` or via `WorkerHeartbeatRequest`) after
/// verifying and applying a `FleetUpdateManifest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerUpdatePushAck {
    pub update_id: String,
    pub node_id: NodeId,
    pub hostname: String,
    pub accepted: bool,
    pub previous_version: String,
    pub installed_version: String,
    pub delivery_mechanism: String,
    pub verified_mac_fnv64: u64,
    pub applied_ktb_merkle_root_fnv64: u64,
    pub repops_post_check_digest: u64,
    pub apply_latency_ms: f64,
    pub status_note: String,
}

/// Summary report returned by `POST /api/v1/update/push` and `GET /api/v1/update/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetUpdatePushSummary {
    pub active_cluster_version: String,
    pub active_ktb_merkle_root_fnv64: u64,
    pub latest_manifest: Option<FleetUpdateManifest>,
    pub total_nodes_targeted: usize,
    pub local_lan_pushed_count: usize,
    pub online_wan_pushed_count: usize,
    pub canary_verified: bool,
    pub zero_downtime_skippipe_active: bool,
    pub node_receipts: Vec<WorkerUpdatePushAck>,
}

/// Request payload sent by `kadmium join` to `POST /api/v1/cluster/register`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegisterRequest {
    pub node: HardwareNode,
    /// Reachable `ip:port` where this worker's `LiveNodeWorkerServer` listens for `WireFrame` TCP streams.
    pub worker_tcp_addr: String,
    /// Optional `kdm1_...` join ticket token.
    pub join_ticket: Option<String>,
    /// If true, replaces the simulated demo fleet with live hardware nodes only.
    pub switch_to_live_only: bool,
}

/// Response returned by `POST /api/v1/cluster/register` after the coordinator verifies the worker's
/// TCP `WireFrame` port and recomputes the cluster shard plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistrationAck {
    pub accepted: bool,
    pub cluster_epoch: u64,
    pub coordinator_id: NodeId,
    pub assigned_node_id: NodeId,
    pub measured_tcp_rtt_ms: f64,
    pub inference_layers: Option<(u32, u32)>,
    pub training_layers: Option<(u32, u32)>,
    pub compute_share_pct: f64,
    pub q_star_ratio: f64,
    pub total_cluster_nodes: usize,
    pub total_live_workers: usize,
    pub message: String,
}

/// Periodic heartbeat sent by a running `kadmium join` worker daemon to `POST /api/v1/cluster/heartbeat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeatRequest {
    pub node_id: NodeId,
    pub worker_tcp_addr: String,
    pub polite_state: PoliteState,
    pub external_gpu_utilization_pct: f64,
    pub external_vram_used_gb: f64,
    pub package_temp_celsius: f64,
    /// Optional receipt if the worker just applied an OTA update queued on the previous heartbeat.
    #[serde(default)]
    pub applied_ota_ack: Option<WorkerUpdatePushAck>,
}

/// Response returned by `POST /api/v1/cluster/heartbeat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeatAck {
    pub cluster_epoch: u64,
    pub rebalance_triggered: bool,
    pub inference_layers: Option<(u32, u32)>,
    pub training_layers: Option<(u32, u32)>,
    pub active_leader: NodeId,
    /// If an OTA update is queued for this node (e.g. an Online WAN node behind NAT), the coordinator
    /// delivers the signed manifest in the heartbeat response so the worker can apply it immediately.
    #[serde(default)]
    pub pending_ota_update: Option<FleetUpdateManifest>,
}

/// Live registration record stored on the coordinator for each connected physical or spawned worker node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveWorkerRegistration {
    pub node: HardwareNode,
    pub worker_tcp_addr: String,
    pub worker_lan_ip: String,
    pub worker_tcp_port: u16,
    pub os_name: String,
    pub cpu_arch: String,
    pub board_or_model: String,
    /// `"LocalLAN"` (direct TCP WireFrame push) or `"OnlineWAN"` (NAT hole-punch / heartbeat pull relay).
    pub network_channel: String,
    pub installed_version: String,
    pub ktb_merkle_root_fnv64: u64,
    pub heartbeat_count: u64,
    pub registered_at_unix_ms: u64,
    pub last_heartbeat_unix_ms: u64,
    pub last_measured_rtt_ms: f64,
    pub executed_forward_stages: u64,
    pub executed_training_steps: u64,
    pub online: bool,
}

impl LiveWorkerRegistration {
    pub fn now_unix_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// Summary returned by `GET /api/v1/cluster/workers` for the Control Room UI and CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveClusterWorkersOverview {
    pub coordinator_lan_ip: String,
    pub coordinator_http_url: String,
    pub join_command_cli: String,
    pub recommended_join_command: String,
    pub join_ticket_token: String,
    pub live_only_mode: bool,
    pub total_cluster_nodes: usize,
    pub live_worker_count: usize,
    pub active_cluster_version: String,
    pub live_workers: Vec<LiveWorkerRegistration>,
    pub workers: Vec<LiveWorkerRegistration>,
}

/// UDP LAN Auto-Discovery service enabling `kadmium join` to find a `kadmium serve` coordinator
/// on the local network without typing IP addresses.
pub struct LanCoordinatorDiscovery;

impl LanCoordinatorDiscovery {
    /// Encodes a coordinator announcement into a `"KDMD"` UDP datagram.
    pub fn encode_announcement(ann: &LanCoordinatorAnnouncement) -> Vec<u8> {
        let json = serde_json::to_vec(ann).expect("serialize LanCoordinatorAnnouncement");
        let mut pkt = Vec::with_capacity(6 + json.len());
        pkt.extend_from_slice(&LAN_BEACON_MAGIC);
        pkt.extend_from_slice(&(json.len() as u16).to_le_bytes());
        pkt.extend_from_slice(&json);
        pkt
    }

    /// Decodes a `"KDMD"` UDP coordinator announcement datagram.
    pub fn decode_announcement(pkt: &[u8]) -> Result<LanCoordinatorAnnouncement, String> {
        if pkt.len() < 6 || pkt[0..4] != LAN_BEACON_MAGIC {
            return Err("Invalid KDMD announcement header".into());
        }
        let len = u16::from_le_bytes([pkt[4], pkt[5]]) as usize;
        if pkt.len() < 6 + len {
            return Err("Truncated KDMD announcement payload".into());
        }
        serde_json::from_slice(&pkt[6..6 + len])
            .map_err(|e| format!("Invalid announcement JSON: {e}"))
    }

    /// Spawns a background UDP responder task on `bind_port` (typically `7849`) that replies to
    /// worker discovery probes with `announcement`.
    pub async fn spawn_responder(
        bind_port: u16,
        announcement: LanCoordinatorAnnouncement,
    ) -> Result<SocketAddr, String> {
        let addr = format!("0.0.0.0:{bind_port}");
        let sock = tokio::net::UdpSocket::bind(&addr)
            .await
            .map_err(|e| format!("Failed to bind UDP discovery responder on {addr}: {e}"))?;
        let local_addr = sock
            .local_addr()
            .map_err(|e| format!("UDP local_addr error: {e}"))?;

        let encoded_reply = Self::encode_announcement(&announcement);
        tokio::spawn(async move {
            let mut buf = [0u8; 2048];
            while let Ok((n, peer)) = sock.recv_from(&mut buf).await {
                if n >= LAN_DISCOVER_COORDINATOR_MAGIC.len()
                    && &buf[..LAN_DISCOVER_COORDINATOR_MAGIC.len()]
                        == LAN_DISCOVER_COORDINATOR_MAGIC
                {
                    let _ = sock.send_to(&encoded_reply, peer).await;
                }
            }
        });

        Ok(local_addr)
    }

    /// Broadcasts a UDP discovery probe on the local subnet and loopback, returning the first
    /// `LanCoordinatorAnnouncement` that responds within `timeout_ms`.
    pub async fn discover_on_lan(
        discovery_port: u16,
        timeout_ms: u64,
    ) -> Option<LanCoordinatorAnnouncement> {
        let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.ok()?;
        let _ = sock.set_broadcast(true);

        let targets = [
            format!("127.0.0.1:{discovery_port}"),
            format!("255.255.255.255:{discovery_port}"),
        ];
        for t in &targets {
            if let Ok(target_addr) = t.parse::<SocketAddr>() {
                let _ = sock
                    .send_to(LAN_DISCOVER_COORDINATOR_MAGIC, target_addr)
                    .await;
            }
        }

        let mut buf = [0u8; 4096];
        match tokio::time::timeout(Duration::from_millis(timeout_ms), sock.recv_from(&mut buf))
            .await
        {
            Ok(Ok((n, _peer))) => Self::decode_announcement(&buf[..n]).ok(),
            _ => None,
        }
    }
}

/// Minimal zero-dependency async HTTP/1.1 JSON client over `tokio::net::TcpStream`
/// for worker-to-coordinator control plane calls (`register`, `heartbeat`, `workers`, `update`).
pub struct MeshHttpClient;

impl MeshHttpClient {
    fn parse_http_url(url: &str) -> Result<(String, u16, String), String> {
        let stripped = url
            .trim()
            .strip_prefix("http://")
            .or_else(|| url.trim().strip_prefix("https://"))
            .unwrap_or(url.trim());

        let (host_port, path) = match stripped.split_once('/') {
            Some((hp, p)) => (hp, format!("/{p}")),
            None => (stripped, "/".to_string()),
        };

        let (host, port) = match host_port.rsplit_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse::<u16>()
                    .map_err(|_| format!("Invalid port in URL '{url}'"))?,
            ),
            None => (host_port.to_string(), 7848),
        };

        Ok((host, port, path))
    }

    /// Sends an HTTP `GET` request to `base_url + path` and deserializes the JSON response.
    pub async fn get_json<Resp: DeserializeOwned>(
        base_url: &str,
        path: &str,
    ) -> Result<Resp, String> {
        let full_url = format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        let (host, port, uri_path) = Self::parse_http_url(&full_url)?;

        let connect_addr = format!("{host}:{port}");
        let mut stream = tokio::time::timeout(
            Duration::from_secs(5),
            tokio::net::TcpStream::connect(&connect_addr),
        )
        .await
        .map_err(|_| format!("Connection timed out to {connect_addr}"))?
        .map_err(|e| format!("Failed to connect to {connect_addr}: {e}"))?;

        let request_header = format!(
            "GET {uri_path} HTTP/1.1\r\nHost: {host}:{port}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
        );

        stream
            .write_all(request_header.as_bytes())
            .await
            .map_err(|e| format!("HTTP write header error: {e}"))?;
        stream
            .flush()
            .await
            .map_err(|e| format!("HTTP flush error: {e}"))?;

        let mut raw_resp = Vec::new();
        stream
            .read_to_end(&mut raw_resp)
            .await
            .map_err(|e| format!("HTTP read error: {e}"))?;

        let resp_str = String::from_utf8_lossy(&raw_resp);
        let (_, body_str) = resp_str
            .split_once("\r\n\r\n")
            .ok_or_else(|| "Malformed HTTP response (missing header separator)".to_string())?;

        serde_json::from_str::<Resp>(body_str.trim()).map_err(|e| {
            format!(
                "Failed to parse JSON response from {full_url}: {e} (body: {})",
                &body_str.trim()[..body_str.trim().len().min(200)]
            )
        })
    }

    /// Sends an HTTP `POST` request with a JSON body to `base_url + path` and deserializes the JSON response.
    pub async fn post_json<Req: Serialize, Resp: DeserializeOwned>(
        base_url: &str,
        path: &str,
        payload: &Req,
    ) -> Result<Resp, String> {
        let full_url = format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        let (host, port, uri_path) = Self::parse_http_url(&full_url)?;
        let body_bytes =
            serde_json::to_vec(payload).map_err(|e| format!("JSON encode error: {e}"))?;

        let connect_addr = format!("{host}:{port}");
        let mut stream = tokio::time::timeout(
            Duration::from_secs(5),
            tokio::net::TcpStream::connect(&connect_addr),
        )
        .await
        .map_err(|_| format!("Connection timed out to {connect_addr}"))?
        .map_err(|e| format!("Failed to connect to {connect_addr}: {e}"))?;

        let request_header = format!(
            "POST {uri_path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body_bytes.len()
        );

        stream
            .write_all(request_header.as_bytes())
            .await
            .map_err(|e| format!("HTTP write header error: {e}"))?;
        stream
            .write_all(&body_bytes)
            .await
            .map_err(|e| format!("HTTP write body error: {e}"))?;
        stream
            .flush()
            .await
            .map_err(|e| format!("HTTP flush error: {e}"))?;

        let mut raw_resp = Vec::new();
        stream
            .read_to_end(&mut raw_resp)
            .await
            .map_err(|e| format!("HTTP read error: {e}"))?;

        let resp_str = String::from_utf8_lossy(&raw_resp);
        let (_, body_str) = resp_str
            .split_once("\r\n\r\n")
            .ok_or_else(|| "Malformed HTTP response (missing header separator)".to_string())?;

        serde_json::from_str::<Resp>(body_str.trim()).map_err(|e| {
            format!(
                "Failed to parse JSON response from {full_url}: {e} (body: {})",
                &body_str.trim()[..body_str.trim().len().min(200)]
            )
        })
    }
}
