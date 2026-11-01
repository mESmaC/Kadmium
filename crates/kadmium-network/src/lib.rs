//! # `kadmium-network`
//!
//! P2P mesh discovery and pairwise link profiling over QUIC for Kadmium.
//! Distinguishes high-speed local bridges (Thunderbolt / 10GbE / 2.5GbE LAN) from Wi-Fi
//! and WAN hole-punched links so `kadmium-sharding` can place tensor-parallel vs.
//! pipeline-parallel boundaries appropriately.

use kadmium_core::{HardwareNode, NodeId, NodeRole, NodeTrustTier, PoliteState};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub mod delta_link;
pub mod lan_pairing;
pub mod live_mesh;
pub mod nat_traversal;
pub mod peer_channel;
pub mod raid_striper;
pub mod usbc_bridge;
pub mod usbc_interconnect;
pub use delta_link::{
    ChunkPlan, DeltaDecoder, DeltaEncoder, DeltaHello, DeltaLinkConfig, DeltaLinkSnapshot,
    DeltaLinkStats, FrameReader, FrameWriter, accept_delta_hello,
};
pub use lan_pairing::*;
pub use live_mesh::*;
pub use nat_traversal::*;
pub use peer_channel::*;
pub use raid_striper::*;
pub use usbc_bridge::{
    BRIDGE_ADDR_SCHEME, BridgeConfig, BridgeLane, BridgeLaneSnapshot, BridgeListenerConfig,
    BridgeReader, BridgeSession, BridgeSnapshot, BridgeWriter, MAX_BRIDGE_LANES, UsbCBridge,
    UsbCBridgeListener,
};
pub use usbc_interconnect::*;

/// Transport path classification between two peers in the Kadmium mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportKind {
    /// Direct 10GbE / 2.5GbE / 1GbE wired LAN over QUIC.
    WiredLanQuic,
    /// Direct USB4 / Thunderbolt IP bridge (e.g., Mac Mini <-> Desktop/NAS).
    ThunderboltBridge,
    /// Local Wi-Fi 6 / 6E mesh link.
    WirelessLanQuic,
    /// NAT hole-punched peer-to-peer WAN link (Iroh / WireGuard / Tailscale style).
    WanDirectHolepunch,
}

/// Pairwise link quality metrics between two nodes in the Kadmium mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerLinkMetrics {
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub transport: TransportKind,
    pub bandwidth_gbps: f64,
    pub rtt_latency_ms: f64,
}

/// Builds a realistic pairwise link matrix for a list of `HardwareNode`s.
pub fn build_mesh_link_matrix(nodes: &[HardwareNode]) -> Vec<PeerLinkMetrics> {
    let mut links = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let a = &nodes[i];
            let b = &nodes[j];
            let bw = a.network_uplink_gbps.min(b.network_uplink_gbps);
            let rtt = ((a.network_latency_ms + b.network_latency_ms) * 10.0).round() / 10.0;
            let transport = if rtt > 8.0 {
                TransportKind::WanDirectHolepunch
            } else if bw >= 10.0 {
                TransportKind::ThunderboltBridge
            } else if rtt >= 1.4 && bw <= 1.0 {
                TransportKind::WirelessLanQuic
            } else {
                TransportKind::WiredLanQuic
            };

            links.push(PeerLinkMetrics {
                from_node: a.id.clone(),
                to_node: b.id.clone(),
                transport,
                bandwidth_gbps: bw,
                rtt_latency_ms: rtt,
            });
        }
    }
    links
}

// ================================================================================================
// Iroh-Style Single-Token Mesh Join Ticket (`kadmium join kdm1_...`) & Binary Wire Framing
// ================================================================================================

/// Payload embedded inside a single-token `kdm1_...` mesh join ticket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinTicket {
    pub coordinator_id: NodeId,
    pub endpoints: Vec<String>,
    /// Target Sub-Compute Environment slice (`None` joins the default pool).
    pub target_slice_id: Option<String>,
    /// Trust tier granted to the joining node (`TrustedCore`, `VerifiedPeerMiddleLayersOnly`, etc.).
    pub granted_trust_tier: NodeTrustTier,
    pub ticket_mac_fnv64: u64,
}

impl JoinTicket {
    pub fn generate(
        coordinator_id: NodeId,
        endpoints: Vec<String>,
        target_slice_id: Option<String>,
        granted_trust_tier: NodeTrustTier,
    ) -> Self {
        let mut ticket = Self {
            coordinator_id,
            endpoints,
            target_slice_id,
            granted_trust_tier,
            ticket_mac_fnv64: 0,
        };
        ticket.ticket_mac_fnv64 = ticket.compute_mac();
        ticket
    }

    fn compute_mac(&self) -> u64 {
        let canonical = format!(
            "{}|{:?}|{:?}|{:?}",
            self.coordinator_id.0, self.endpoints, self.target_slice_id, self.granted_trust_tier
        );
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in canonical.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Encodes the ticket as a compact `kdm1_<hex>` string for CLI/dashboard copy-paste.
    pub fn encode_token(&self) -> String {
        let json = serde_json::to_vec(self).expect("serialize JoinTicket");
        let mut hex = String::with_capacity(5 + json.len() * 2);
        hex.push_str("kdm1_");
        for b in json {
            hex.push_str(&format!("{:02x}", b));
        }
        hex
    }

    /// Decodes and verifies a `kdm1_<hex>` join token string.
    pub fn decode_and_verify(token: &str) -> Result<Self, String> {
        let hex_body = token
            .strip_prefix("kdm1_")
            .ok_or_else(|| "Invalid ticket prefix (expected 'kdm1_')".to_string())?;
        if hex_body.len() % 2 != 0 {
            return Err("Invalid hex length in join ticket".into());
        }
        let mut bytes = Vec::with_capacity(hex_body.len() / 2);
        for i in (0..hex_body.len()).step_by(2) {
            let b = u8::from_str_radix(&hex_body[i..i + 2], 16)
                .map_err(|_| "Invalid hex character in join ticket".to_string())?;
            bytes.push(b);
        }
        let ticket: Self = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Malformed JoinTicket payload: {e}"))?;
        if ticket.compute_mac() != ticket.ticket_mac_fnv64 {
            return Err("JoinTicket MAC integrity check failed".into());
        }
        Ok(ticket)
    }
}

/// Magic bytes of the legacy 13-byte binary wire frame header (`"KDMF"`), which carried no
/// request id. Still accepted on read so pre-`KDM2` nodes can talk to this one; never emitted.
pub const KDM_FRAME_MAGIC: [u8; 4] = *b"KDMF";

/// Magic bytes of the current 21-byte binary wire frame header (`"KDM2"`) carrying a `u64` request id.
pub const KDM_FRAME_MAGIC_V2: [u8; 4] = *b"KDM2";

/// Legacy `KDMF` header length: magic (4) + kind `u8` (1) + epoch `u32` LE (4) + len `u32` LE (4).
pub const KDM_FRAME_HEADER_LEN_V1: usize = 13;

/// `KDM2` header length: magic (4) + kind `u8` (1) + epoch `u32` LE (4) + request_id `u64` LE (8)
/// + len `u32` LE (4).
pub const KDM_FRAME_HEADER_LEN_V2: usize = 21;

/// Upper bound on a single frame payload accepted from the wire (1 GiB). A corrupt or hostile
/// length field must not make a node allocate gigabytes before the payload even arrives.
pub const KDM_FRAME_MAX_PAYLOAD_BYTES: usize = 1 << 30;

/// Multiplexed QUIC stream message kind in the Kadmium P2P data/control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WireFrameKind {
    HeartbeatTelemetry = 1,
    ShardPlanEpochUpdate = 2,
    SystolicTokenWavefront = 3,
    DeMoCompressedGradient = 4,
    Fts3KtbChunkStream = 5,
    VerdeBisectionChallenge = 6,
    FleetUpdatePush = 7,
    DreamRsiRolloutReplay = 8,
    KvCachePageMigration = 9,
    /// Generic grid work unit pushed to a worker. The payload is opaque bytes whose layout
    /// `kadmium-grid` defines; the worker answers with a `WorkResult` carrying the same `request_id`.
    WorkDispatch = 10,
    /// Result (or a small `{"error":"..."}` JSON body) for a `WorkDispatch` or `WorkRequest`,
    /// matched to it by `request_id`.
    WorkResult = 11,
    /// A worker asking a scheduler for work. Opaque payload defined by `kadmium-grid`.
    WorkRequest = 12,
    /// Opens a delta link on a connection (see [`delta_link`]); answered by a `DeltaHello`.
    DeltaHello = 13,
    /// Any other frame, sent as only the chunks the peer does not already hold (see
    /// [`delta_link`]). Read loops rebuild it into the frame it wraps before dispatching.
    DeltaFrame = 14,
    /// Batched, checkpoint-bound expert reads. Payload is a binary `KDEX` request/response
    /// defined by `kadmium-data`; replies reuse this kind and the request id.
    ExpertLoad = 15,
    /// Authenticated native decoder slice session (`KDNS` open, exact-`f32` forward, close and
    /// cancel). Replies reuse this kind and the request id.
    NativeDecoderSession = 16,
}

impl WireFrameKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::HeartbeatTelemetry),
            2 => Some(Self::ShardPlanEpochUpdate),
            3 => Some(Self::SystolicTokenWavefront),
            4 => Some(Self::DeMoCompressedGradient),
            5 => Some(Self::Fts3KtbChunkStream),
            6 => Some(Self::VerdeBisectionChallenge),
            7 => Some(Self::FleetUpdatePush),
            8 => Some(Self::DreamRsiRolloutReplay),
            9 => Some(Self::KvCachePageMigration),
            10 => Some(Self::WorkDispatch),
            11 => Some(Self::WorkResult),
            12 => Some(Self::WorkRequest),
            13 => Some(Self::DeltaHello),
            14 => Some(Self::DeltaFrame),
            15 => Some(Self::ExpertLoad),
            16 => Some(Self::NativeDecoderSession),
            _ => None,
        }
    }
}

/// Binary length-prefixed wire frame for Kadmium P2P streams.
///
/// Encodes as the 21-byte `KDM2` header followed by the payload:
///
/// ```text
/// offset  size  field
///      0     4  magic "KDM2"
///      4     1  kind (WireFrameKind as u8)
///      5     4  epoch, u32 LE
///      9     8  request_id, u64 LE
///     17     4  payload length, u32 LE
///     21     n  payload
/// ```
///
/// Decoding also accepts the legacy 13-byte `KDMF` header, which has no `request_id` field (it
/// reads as 0). The sender (`PeerChannel`) allocates `request_id` per connection and the receiver
/// echoes it on the reply, so many requests can be in flight on one socket and replies may come
/// back in any order. `0` means "legacy or unsolicited".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireFrame {
    pub kind: WireFrameKind,
    pub epoch: u32,
    pub request_id: u64,
    pub payload: Vec<u8>,
}

impl WireFrame {
    /// A frame with `request_id` 0; tag it with [`WireFrame::with_request_id`].
    pub fn new(kind: WireFrameKind, epoch: u32, payload: Vec<u8>) -> Self {
        Self {
            kind,
            epoch,
            request_id: 0,
            payload,
        }
    }

    /// Returns the frame tagged with `request_id`.
    pub fn with_request_id(mut self, request_id: u64) -> Self {
        self.request_id = request_id;
        self
    }

    /// Encodes as a `KDM2` frame.
    pub fn encode(&self) -> Vec<u8> {
        self.encode_with_request_id(self.request_id)
    }

    /// Encodes as a `KDM2` frame carrying `request_id` instead of `self.request_id`, without
    /// cloning the payload (lets a caller resend one prepared frame under a fresh id).
    pub fn encode_with_request_id(&self, request_id: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(KDM_FRAME_HEADER_LEN_V2 + self.payload.len());
        out.extend_from_slice(&KDM_FRAME_MAGIC_V2);
        out.push(self.kind as u8);
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&request_id.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Header length implied by the 4 magic bytes, or an error for anything that is not a frame.
    fn header_len_for_magic(magic: &[u8]) -> Result<usize, String> {
        if magic == KDM_FRAME_MAGIC_V2 {
            Ok(KDM_FRAME_HEADER_LEN_V2)
        } else if magic == KDM_FRAME_MAGIC {
            Ok(KDM_FRAME_HEADER_LEN_V1)
        } else {
            Err("Invalid WireFrame magic (expected 'KDM2' or legacy 'KDMF')".into())
        }
    }

    /// Parses a complete v1 or v2 header into `(kind, epoch, request_id, payload_len)`.
    fn parse_header(header: &[u8]) -> Result<(WireFrameKind, u32, u64, usize), String> {
        let kind = WireFrameKind::from_u8(header[4])
            .ok_or_else(|| format!("Unknown WireFrameKind byte: {}", header[4]))?;
        let epoch = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
        let (request_id, len_at) = if header.len() == KDM_FRAME_HEADER_LEN_V2 {
            let mut id = [0u8; 8];
            id.copy_from_slice(&header[9..17]);
            (u64::from_le_bytes(id), 17)
        } else {
            (0, 9)
        };
        let len = u32::from_le_bytes([
            header[len_at],
            header[len_at + 1],
            header[len_at + 2],
            header[len_at + 3],
        ]) as usize;
        if len > KDM_FRAME_MAX_PAYLOAD_BYTES {
            return Err(format!(
                "WireFrame payload length {len} exceeds the {KDM_FRAME_MAX_PAYLOAD_BYTES}-byte limit"
            ));
        }
        Ok((kind, epoch, request_id, len))
    }

    /// Decodes a `KDM2` or legacy `KDMF` frame from a byte slice.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 {
            return Err("WireFrame shorter than 4-byte magic".into());
        }
        let header_len = Self::header_len_for_magic(&bytes[0..4])?;
        if bytes.len() < header_len {
            return Err(format!("WireFrame shorter than {header_len}-byte header"));
        }
        let (kind, epoch, request_id, len) = Self::parse_header(&bytes[..header_len])?;
        if bytes.len() < header_len + len {
            return Err("WireFrame payload truncated".into());
        }
        Ok(Self {
            kind,
            epoch,
            request_id,
            payload: bytes[header_len..header_len + len].to_vec(),
        })
    }

    /// Asynchronously writes this binary `WireFrame` onto any Tokio `AsyncWrite` stream.
    pub async fn write_to_stream<W: AsyncWriteExt + Unpin>(
        &self,
        stream: &mut W,
    ) -> Result<(), String> {
        let encoded = self.encode();
        stream
            .write_all(&encoded)
            .await
            .map_err(|e| format!("Stream write error: {e}"))?;
        stream
            .flush()
            .await
            .map_err(|e| format!("Stream flush error: {e}"))
    }

    /// Asynchronously reads a single `KDM2` (or legacy `KDMF`) `WireFrame` from any Tokio
    /// `AsyncRead` stream.
    pub async fn read_from_stream<R: AsyncReadExt + Unpin>(stream: &mut R) -> Result<Self, String> {
        let mut header = [0u8; KDM_FRAME_HEADER_LEN_V2];
        stream
            .read_exact(&mut header[..4])
            .await
            .map_err(|e| format!("Stream header read error: {e}"))?;
        let header_len = Self::header_len_for_magic(&header[..4]).map_err(|_| {
            "Invalid WireFrame magic on stream (expected 'KDM2' or legacy 'KDMF')".to_string()
        })?;
        stream
            .read_exact(&mut header[4..header_len])
            .await
            .map_err(|e| format!("Stream header read error: {e}"))?;
        let (kind, epoch, request_id, len) = Self::parse_header(&header[..header_len])?;
        let mut payload = vec![0u8; len];
        if len > 0 {
            stream
                .read_exact(&mut payload)
                .await
                .map_err(|e| format!("Stream payload read error: {e}"))?;
        }
        Ok(Self {
            kind,
            epoch,
            request_id,
            payload,
        })
    }
}

/// Consensus and automatic coordinator failover state for a cluster or Sub-Compute Slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorFailoverQuorum {
    pub term: u64,
    pub active_leader: NodeId,
    pub standby_candidates: Vec<NodeId>,
}

impl CoordinatorFailoverQuorum {
    /// Initializes the coordinator failover quorum from eligible `TrustedCore` nodes holding `NodeRole::Coordinator`
    /// or `NodeRole::StorageVault` (e.g. Main PC, M4 Mac Mini, Home NAS).
    pub fn from_cluster(nodes: &[HardwareNode]) -> Self {
        if nodes.is_empty() {
            return Self {
                term: 0,
                active_leader: NodeId("unconnected".into()),
                standby_candidates: Vec::new(),
            };
        }

        let mut candidates: Vec<NodeId> = nodes
            .iter()
            .filter(|n| {
                n.trust_tier == NodeTrustTier::TrustedCore
                    && n.polite_state != PoliteState::Draining
                    && (n.roles.contains(&NodeRole::Coordinator)
                        || n.roles.contains(&NodeRole::StorageVault))
            })
            .map(|n| n.id.clone())
            .collect();

        let active_leader = if !candidates.is_empty() {
            candidates.remove(0)
        } else {
            nodes[0].id.clone()
        };

        Self {
            term: 1,
            active_leader,
            standby_candidates: candidates,
        }
    }

    /// Triggers an automatic leader failover when `failed_node` goes offline or reboots,
    /// promoting the highest-priority standby node and incrementing `term`.
    pub fn trigger_failover(&mut self, failed_node: &NodeId) -> NodeId {
        if &self.active_leader == failed_node && !self.standby_candidates.is_empty() {
            self.term += 1;
            let old_leader = self.active_leader.clone();
            self.active_leader = self.standby_candidates.remove(0);
            self.standby_candidates.push(old_leader);
        }
        self.active_leader.clone()
    }
}

/// 4-byte UDP LAN mDNS/beacon magic (`"KDMD"` = Kadmium Discovery).
pub const LAN_BEACON_MAGIC: [u8; 4] = *b"KDMD";

/// Zero-config LAN discovery beacon broadcast over UDP (`_kadmium._udp.local` / port `7849`)
/// so nodes on the same Wi-Fi/Ethernet subnet auto-discover each other without manual IPs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanDiscoveryBeacon {
    pub node_id: NodeId,
    pub hostname: String,
    pub quic_port: u16,
    pub q_star_ratio: f64,
    pub vram_or_uma_gb: f64,
    pub polite_state: PoliteState,
    pub trust_tier: NodeTrustTier,
}

impl LanDiscoveryBeacon {
    /// Constructs a LAN discovery beacon from a `HardwareNode`.
    pub fn from_node(node: &HardwareNode, quic_port: u16) -> Self {
        Self {
            node_id: node.id.clone(),
            hostname: node.hostname.clone(),
            quic_port,
            q_star_ratio: node.compute_q_star_ratio(),
            vram_or_uma_gb: node.memory.vram_gb,
            polite_state: node.polite_state,
            trust_tier: node.trust_tier,
        }
    }

    /// Encodes the beacon into a compact `"KDMD"` UDP datagram.
    pub fn encode_datagram(&self) -> Vec<u8> {
        let json = serde_json::to_vec(self).expect("serialize LanDiscoveryBeacon");
        let mut pkt = Vec::with_capacity(6 + json.len());
        pkt.extend_from_slice(&LAN_BEACON_MAGIC);
        pkt.extend_from_slice(&(json.len() as u16).to_le_bytes());
        pkt.extend_from_slice(&json);
        pkt
    }

    /// Decodes and validates a `"KDMD"` UDP datagram.
    pub fn decode_datagram(pkt: &[u8]) -> Result<Self, String> {
        if pkt.len() < 6 {
            return Err("UDP datagram shorter than 6-byte KDMD header".into());
        }
        if pkt[0..4] != LAN_BEACON_MAGIC {
            return Err("Invalid LAN beacon magic (expected 'KDMD')".into());
        }
        let len = u16::from_le_bytes([pkt[4], pkt[5]]) as usize;
        if pkt.len() < 6 + len {
            return Err("Truncated LAN beacon JSON payload".into());
        }
        serde_json::from_slice(&pkt[6..6 + len]).map_err(|e| format!("Invalid beacon JSON: {e}"))
    }

    /// Sends this beacon over an async Tokio `UdpSocket` to `target_addr`.
    pub async fn broadcast_once(
        &self,
        socket: &tokio::net::UdpSocket,
        target_addr: std::net::SocketAddr,
    ) -> Result<usize, String> {
        let dgram = self.encode_datagram();
        socket
            .send_to(&dgram, target_addr)
            .await
            .map_err(|e| format!("UDP send_to error: {e}"))
    }

    /// Receives and decodes a single `LanDiscoveryBeacon` from an async Tokio `UdpSocket`.
    pub async fn recv_once(
        socket: &tokio::net::UdpSocket,
    ) -> Result<(Self, std::net::SocketAddr), String> {
        let mut buf = [0u8; 2048];
        let (n, peer_addr) = socket
            .recv_from(&mut buf)
            .await
            .map_err(|e| format!("UDP recv_from error: {e}"))?;
        let beacon = Self::decode_datagram(&buf[..n])?;
        Ok((beacon, peer_addr))
    }
}

/// Last-Write-Wins (LWW) CRDT register for a single node's replicated state in the gossip mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrdtNodeRegister {
    pub node_id: String,
    pub lamport_ts: u64,
    pub writer_peer: String,
    pub polite_state: PoliteState,
    pub assigned_layers: Option<(u32, u32)>,
    pub online: bool,
}

impl CrdtNodeRegister {
    /// Deterministic LWW ordering: higher `lamport_ts` wins; ties broken lexicographically by `writer_peer`.
    pub fn supersedes(&self, other: &Self) -> bool {
        (self.lamport_ts, &self.writer_peer) > (other.lamport_ts, &other.writer_peer)
    }
}

/// Conflict-Free Replicated Data Type (LWW-Element-Map + Vector Clock) for decentralized
/// Kadmium cluster topology and shard assignment replication.
///
/// Guarantees strong eventual consistency ($A \sqcup B = B \sqcup A$, $(A \sqcup B) \sqcup C = A \sqcup (B \sqcup C)$, $A \sqcup A = A$).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterStateCrdt {
    pub vector_clock: std::collections::BTreeMap<String, u64>,
    pub registers: std::collections::BTreeMap<String, CrdtNodeRegister>,
}

impl ClusterStateCrdt {
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates a node's state in the CRDT, advancing `writer_peer`'s logical clock.
    pub fn upsert_node_state(
        &mut self,
        writer_peer: impl Into<String>,
        node_id: impl Into<String>,
        polite_state: PoliteState,
        assigned_layers: Option<(u32, u32)>,
        online: bool,
    ) {
        let writer = writer_peer.into();
        let nid = node_id.into();
        let max_seen = self.vector_clock.values().copied().max().unwrap_or(0);
        let next_ts = max_seen + 1;
        self.vector_clock.insert(writer.clone(), next_ts);

        let candidate = CrdtNodeRegister {
            node_id: nid.clone(),
            lamport_ts: next_ts,
            writer_peer: writer,
            polite_state,
            assigned_layers,
            online,
        };

        match self.registers.get(&nid) {
            Some(existing) if !candidate.supersedes(existing) => {}
            _ => {
                self.registers.insert(nid, candidate);
            }
        }
    }

    /// Commutative, associative, and idempotent join ($\sqcup$) with a remote peer's `ClusterStateCrdt`.
    pub fn merge(&mut self, remote: &Self) {
        for (peer, &remote_ts) in &remote.vector_clock {
            let entry = self.vector_clock.entry(peer.clone()).or_insert(0);
            if remote_ts > *entry {
                *entry = remote_ts;
            }
        }

        for (nid, remote_reg) in &remote.registers {
            match self.registers.get(nid) {
                Some(local_reg) => {
                    if remote_reg.supersedes(local_reg) {
                        self.registers.insert(nid.clone(), remote_reg.clone());
                    }
                }
                None => {
                    self.registers.insert(nid.clone(), remote_reg.clone());
                }
            }
        }
    }

    /// Packs this CRDT snapshot into a `WireFrame` for P2P gossip transmission.
    pub fn to_wire_frame(&self, epoch: u32) -> WireFrame {
        WireFrame::new(
            WireFrameKind::ShardPlanEpochUpdate,
            epoch,
            serde_json::to_vec(self).expect("serialize ClusterStateCrdt"),
        )
    }

    /// Unpacks a `ClusterStateCrdt` from a received `WireFrame`.
    pub fn from_wire_frame(frame: &WireFrame) -> Result<Self, String> {
        serde_json::from_slice(&frame.payload).map_err(|e| format!("Invalid CRDT payload: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::sample_personal_cluster;

    #[test]
    fn test_join_ticket_roundtrip_and_mac_verification() {
        let ticket = JoinTicket::generate(
            NodeId("node-m4-mac-mini".into()),
            vec!["192.168.1.48:7848".into(), "relay.kadmium.ai:4433".into()],
            Some("slice-03-train-diloco".into()),
            NodeTrustTier::VerifiedPeerMiddleLayersOnly,
        );

        let token = ticket.encode_token();
        assert!(token.starts_with("kdm1_"));

        let decoded = JoinTicket::decode_and_verify(&token).expect("valid ticket");
        assert_eq!(decoded, ticket);
    }

    /// Encodes a frame with the pre-request-id 13-byte `KDMF` header exactly as old nodes did.
    fn encode_legacy_kdmf(frame: &WireFrame) -> Vec<u8> {
        let mut out = Vec::with_capacity(KDM_FRAME_HEADER_LEN_V1 + frame.payload.len());
        out.extend_from_slice(&KDM_FRAME_MAGIC);
        out.push(frame.kind as u8);
        out.extend_from_slice(&frame.epoch.to_le_bytes());
        out.extend_from_slice(&(frame.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&frame.payload);
        out
    }

    #[test]
    fn test_binary_wire_frame_encode_decode() {
        // Current layout: 21-byte `KDM2` header with a u64 request id.
        let frame = WireFrame::new(
            WireFrameKind::SystolicTokenWavefront,
            4,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        )
        .with_request_id(0xDEAD_BEEF_0000_0042);
        let raw = frame.encode();
        assert_eq!(raw.len(), KDM_FRAME_HEADER_LEN_V2 + 8);
        assert_eq!(&raw[0..4], &KDM_FRAME_MAGIC_V2);
        assert_eq!(raw[4], WireFrameKind::SystolicTokenWavefront as u8);
        assert_eq!(u32::from_le_bytes([raw[5], raw[6], raw[7], raw[8]]), 4);
        let mut id = [0u8; 8];
        id.copy_from_slice(&raw[9..17]);
        assert_eq!(u64::from_le_bytes(id), 0xDEAD_BEEF_0000_0042);
        assert_eq!(u32::from_le_bytes([raw[17], raw[18], raw[19], raw[20]]), 8);
        let decoded = WireFrame::decode(&raw).expect("decode KDM2 frame");
        assert_eq!(decoded, frame);
        assert_eq!(
            WireFrame::decode(&frame.encode_with_request_id(7))
                .expect("decode re-tagged frame")
                .request_id,
            7
        );

        // Legacy layout: 13-byte `KDMF` header still decodes, with request_id 0.
        let legacy = WireFrame::new(WireFrameKind::DeMoCompressedGradient, 9, b"legacy".to_vec());
        let raw_legacy = encode_legacy_kdmf(&legacy);
        assert_eq!(raw_legacy.len(), KDM_FRAME_HEADER_LEN_V1 + 6);
        let decoded_legacy = WireFrame::decode(&raw_legacy).expect("decode legacy KDMF frame");
        assert_eq!(decoded_legacy, legacy);
        assert_eq!(decoded_legacy.request_id, 0);

        // Truncation, unknown kinds, and bad magic are rejected.
        assert!(WireFrame::decode(&raw[..raw.len() - 1]).is_err());
        assert!(WireFrame::decode(&raw[..KDM_FRAME_HEADER_LEN_V2 - 1]).is_err());
        assert!(WireFrame::decode(&raw_legacy[..5]).is_err());
        assert!(WireFrame::decode(b"KD").is_err());
        let mut bad_magic = raw.clone();
        bad_magic[0..4].copy_from_slice(b"KDMX");
        assert!(WireFrame::decode(&bad_magic).is_err());
        let mut bad_kind = raw.clone();
        bad_kind[4] = 200;
        assert!(WireFrame::decode(&bad_kind).is_err());

        // Generic grid work kinds round-trip through the byte codec.
        for (kind, byte) in [
            (WireFrameKind::WorkDispatch, 10u8),
            (WireFrameKind::WorkResult, 11u8),
            (WireFrameKind::WorkRequest, 12u8),
            (WireFrameKind::DeltaHello, 13u8),
            (WireFrameKind::DeltaFrame, 14u8),
            (WireFrameKind::ExpertLoad, 15u8),
            (WireFrameKind::NativeDecoderSession, 16u8),
        ] {
            assert_eq!(kind as u8, byte);
            assert_eq!(WireFrameKind::from_u8(byte), Some(kind));
        }
        assert_eq!(WireFrameKind::from_u8(17), None);
    }

    #[tokio::test]
    async fn test_wire_frame_stream_reader_accepts_v2_and_legacy_headers() {
        // `&[u8]` is an `AsyncRead`: feed both layouts back to back through one reader.
        let v2 = WireFrame::new(WireFrameKind::WorkDispatch, 1, vec![9, 9, 9]).with_request_id(77);
        let legacy = WireFrame::new(WireFrameKind::HeartbeatTelemetry, 2, Vec::new());
        let mut bytes = v2.encode();
        bytes.extend(encode_legacy_kdmf(&legacy));

        let mut cursor: &[u8] = &bytes;
        let first = WireFrame::read_from_stream(&mut cursor)
            .await
            .expect("read KDM2 frame");
        assert_eq!(first, v2);
        let second = WireFrame::read_from_stream(&mut cursor)
            .await
            .expect("read legacy KDMF frame");
        assert_eq!(second, legacy);
        assert!(cursor.is_empty());
        assert!(WireFrame::read_from_stream(&mut cursor).await.is_err());
    }

    #[tokio::test]
    async fn test_async_tokio_p2p_wire_frame_exchange_and_coordinator_failover() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");

        let server_task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept peer");
            let received = WireFrame::read_from_stream(&mut socket)
                .await
                .expect("read frame");
            assert_eq!(received.kind, WireFrameKind::DeMoCompressedGradient);
            assert_eq!(received.epoch, 7);

            let reply = WireFrame::new(
                WireFrameKind::ShardPlanEpochUpdate,
                8,
                b"ACK-EPOCH-8".to_vec(),
            );
            reply
                .write_to_stream(&mut socket)
                .await
                .expect("write reply");
        });

        let mut client = tokio::net::TcpStream::connect(addr)
            .await
            .expect("connect loopback");
        let out_frame = WireFrame::new(WireFrameKind::DeMoCompressedGradient, 7, vec![42, 99, 128]);
        out_frame
            .write_to_stream(&mut client)
            .await
            .expect("client write");

        let ack = WireFrame::read_from_stream(&mut client)
            .await
            .expect("client read ack");
        assert_eq!(ack.kind, WireFrameKind::ShardPlanEpochUpdate);
        assert_eq!(ack.epoch, 8);
        assert_eq!(ack.payload, b"ACK-EPOCH-8");
        server_task.await.expect("server task complete");

        // Also test automatic Coordinator Failover Quorum (Main PC -> M4 Mac Mini -> Home NAS):
        let cluster = sample_personal_cluster();
        let mut quorum = CoordinatorFailoverQuorum::from_cluster(&cluster);
        assert_eq!(quorum.active_leader, NodeId("node-main-pc".into()));
        assert_eq!(quorum.term, 1);

        let new_leader = quorum.trigger_failover(&NodeId("node-main-pc".into()));
        assert_eq!(new_leader, NodeId("node-m4-mac-mini".into()));
        assert_eq!(quorum.term, 2);

        // Also test async UDP LAN Discovery Beacon ("KDMD") broadcast and receive:
        let rx_sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind rx udp");
        let rx_addr = rx_sock.local_addr().expect("rx addr");
        let tx_sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind tx udp");

        let beacon = LanDiscoveryBeacon::from_node(&cluster[1], 7848);
        beacon
            .broadcast_once(&tx_sock, rx_addr)
            .await
            .expect("broadcast udp beacon");
        let (received_beacon, _) = LanDiscoveryBeacon::recv_once(&rx_sock)
            .await
            .expect("recv udp beacon");
        assert_eq!(received_beacon, beacon);

        // Verify ZeroClickLanPairingEngine 6-digit SAS code & one-click ticket approval
        let mut cands = ZeroClickLanPairingEngine::discover_lan_candidates(&cluster[0].id);
        assert_eq!(cands.len(), 3);
        assert_eq!(cands[0].status, PairingApprovalStatus::ApprovedTrustedCore);
        assert_eq!(cands[1].short_auth_code.len(), 7); // "XXX-YYY"
        let issued = ZeroClickLanPairingEngine::approve_candidate(
            &cluster[0].id,
            &mut cands[2],
            NodeTrustTier::VerifiedPeerMiddleLayersOnly,
        );
        assert_eq!(
            cands[2].status,
            PairingApprovalStatus::ApprovedVerifiedMiddleOnly
        );
        assert!(JoinTicket::decode_and_verify(&issued.encode_token()).is_ok());
    }

    #[test]
    fn test_cluster_state_crdt_commutativity_idempotence_and_wire_sync() {
        let mut crdt_a = ClusterStateCrdt::new();
        crdt_a.upsert_node_state(
            "node-main-pc",
            "node-main-pc",
            PoliteState::IdleDedicated,
            Some((0, 25)),
            true,
        );
        crdt_a.upsert_node_state(
            "node-main-pc",
            "node-m4-mac-mini",
            PoliteState::IdleDedicated,
            Some((26, 42)),
            true,
        );

        let mut crdt_b = crdt_a.clone();
        // Concurrent update on node A: main PC enters gaming mode
        crdt_a.upsert_node_state(
            "node-main-pc",
            "node-main-pc",
            PoliteState::GamingOrHeavyApp,
            Some((0, 12)),
            true,
        );
        // Concurrent update on node B: Mac Mini absorbs layers 13..42
        crdt_b.upsert_node_state(
            "node-m4-mac-mini",
            "node-m4-mac-mini",
            PoliteState::IdleDedicated,
            Some((13, 42)),
            true,
        );

        let mut ab = crdt_a.clone();
        ab.merge(&crdt_b);

        let mut ba = crdt_b.clone();
        ba.merge(&crdt_a);

        // Commutativity: A ⊔ B == B ⊔ A
        assert_eq!(ab, ba);

        // Idempotence: (A ⊔ B) ⊔ B == A ⊔ B
        let mut abb = ab.clone();
        abb.merge(&crdt_b);
        assert_eq!(abb, ab);

        // WireFrame round-trip
        let frame = ab.to_wire_frame(9);
        let decoded = ClusterStateCrdt::from_wire_frame(&frame).expect("decode CRDT frame");
        assert_eq!(decoded, ab);
    }

    #[tokio::test]
    async fn test_nat_traversal_holepunch_and_derp_fallback() {
        let cluster = sample_personal_cluster();
        let matrix = NatTraversalCoordinator::probe_cluster_nat_mesh(&cluster);
        assert!(matrix.len() >= 7);
        assert!(
            matrix
                .iter()
                .any(|r| r.selected_path == CandidateEndpointKind::LanDirect)
        );
        assert!(
            matrix
                .iter()
                .any(|r| r.selected_path == CandidateEndpointKind::StunPublicReflexive)
        );
        assert!(
            matrix
                .iter()
                .any(|r| r.selected_path == CandidateEndpointKind::DerpRelayFallback)
        );

        let (from_a, to_b, nonce) = NatTraversalCoordinator::execute_udp_holepunch_loopback()
            .await
            .expect("udp hole-punch loopback");
        assert_ne!(from_a.port(), to_b.port());
        assert_eq!(nonce, 0xCD48_2026_F5B7_0001);
    }
}
