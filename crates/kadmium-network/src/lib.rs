//! # `kadmium-network`
//!
//! P2P mesh discovery and pairwise link profiling over QUIC for Kadmium.
//! Distinguishes high-speed local bridges (Thunderbolt / 10GbE / 2.5GbE LAN) from Wi-Fi
//! and WAN hole-punched links so `kadmium-sharding` can place tensor-parallel vs.
//! pipeline-parallel boundaries appropriately.

use kadmium_core::{HardwareNode, NodeId, NodeRole, NodeTrustTier, PoliteState};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

/// Magic bytes for Kadmium's binary QUIC wire frame header (`"KDMF"`).
pub const KDM_FRAME_MAGIC: [u8; 4] = *b"KDMF";

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
            _ => None,
        }
    }
}

/// Binary length-prefixed wire frame for Kadmium P2P streams (`13-byte header + payload`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireFrame {
    pub kind: WireFrameKind,
    pub epoch: u32,
    pub payload: Vec<u8>,
}

impl WireFrame {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(13 + self.payload.len());
        out.extend_from_slice(&KDM_FRAME_MAGIC);
        out.push(self.kind as u8);
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 13 {
            return Err("WireFrame shorter than 13-byte header".into());
        }
        if bytes[0..4] != KDM_FRAME_MAGIC {
            return Err("Invalid WireFrame magic (expected 'KDMF')".into());
        }
        let kind = WireFrameKind::from_u8(bytes[4])
            .ok_or_else(|| format!("Unknown WireFrameKind byte: {}", bytes[4]))?;
        let epoch = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        let len = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
        if bytes.len() < 13 + len {
            return Err("WireFrame payload truncated".into());
        }
        Ok(Self {
            kind,
            epoch,
            payload: bytes[13..13 + len].to_vec(),
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

    /// Asynchronously reads a single binary `WireFrame` from any Tokio `AsyncRead` stream.
    pub async fn read_from_stream<R: AsyncReadExt + Unpin>(stream: &mut R) -> Result<Self, String> {
        let mut header = [0u8; 13];
        stream
            .read_exact(&mut header)
            .await
            .map_err(|e| format!("Stream header read error: {e}"))?;
        if header[0..4] != KDM_FRAME_MAGIC {
            return Err("Invalid WireFrame magic on stream (expected 'KDMF')".into());
        }
        let kind = WireFrameKind::from_u8(header[4])
            .ok_or_else(|| format!("Unknown WireFrameKind byte: {}", header[4]))?;
        let epoch = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
        let len = u32::from_le_bytes([header[9], header[10], header[11], header[12]]) as usize;
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
            NodeId("node-m4-mac-mini".into())
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

    #[test]
    fn test_binary_wire_frame_encode_decode() {
        let frame = WireFrame {
            kind: WireFrameKind::SystolicTokenWavefront,
            epoch: 4,
            payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };
        let raw = frame.encode();
        let decoded = WireFrame::decode(&raw).expect("decode frame");
        assert_eq!(decoded, frame);
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

            let reply = WireFrame {
                kind: WireFrameKind::ShardPlanEpochUpdate,
                epoch: 8,
                payload: b"ACK-EPOCH-8".to_vec(),
            };
            reply
                .write_to_stream(&mut socket)
                .await
                .expect("write reply");
        });

        let mut client = tokio::net::TcpStream::connect(addr)
            .await
            .expect("connect loopback");
        let out_frame = WireFrame {
            kind: WireFrameKind::DeMoCompressedGradient,
            epoch: 7,
            payload: vec![42, 99, 128],
        };
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
    }
}
