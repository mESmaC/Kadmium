//! # USB-C / USB4 / Thunderbolt & Dual-Role (`USB-NCM`) Direct Interconnect Engine
//!
//! Enables switchless, ultra-high-bandwidth (`10 Gbps` to `80+ Gbps`), sub-millisecond (`80–200 µs`)
//! direct USB-C cabling between local cluster systems:
//! - **Tier 1 (`Thunderbolt5Usb4V2_80Gbps` / `Usb4Thunderbolt4_30To40Gbps`)**:
//!   Symmetric Host-to-Host PCIe DMA tunneling over USB4 / Thunderbolt 3/4/5 (`30–120 Gbps`).
//!   Connects handheld gaming PCs (e.g., 30 Gbps USB4 / Thunderbolt handhelds), laptops, desktops,
//!   and Apple Silicon Macs directly via a standard USB4 / Thunderbolt cable.
//! - **Tier 2 (`Usb32Gen2x2_20GbpsDrd` / `Usb32Gen2_10GbpsDrd` / `Usb32Gen1_5GbpsDrd`)**:
//!   Asymmetric Dual-Role-Data (`DRD` / OTG) links over `USB-NCM` (`cdc_ncm`) or `FunctionFS` bulk
//!   endpoints (`5–20 Gbps`). Connects PCs/Macs to SBCs (RK3588, RPi 5, Jetson Orin) or Android devices.
//! - **Tier 3 (`Usb20HighSpeed_480Mbps`)**:
//!   High-Speed USB 2.0 CDC-ACM / Bulk links (`480 Mbps`) for ultra-low-power RISC-V / MCU sidecars.
//!
//! ## Key Capabilities
//! 1. **Link-Local Peering Models (`169.254.0.0/16` & `fe80::`)**:
//!    Describes peering candidates over configured point-to-point interfaces. The scanner does not
//!    configure DHCP, addresses, or a cable's network interface.
//! 2. **Multi-Stream Socket Striping Models (`4x–8x` Parallel Lanes + `64 KiB` Frames)**:
//!    Derives goodput figures from rated bandwidth and scenario inputs. Values such as
//!    `~26.5–28.5 Gbps` are estimates, not measured tensor throughput or DMA benchmarks.
//! 3. **Tensor-Parallel / Expert-Parallel Eligibility Planning**:
//!    Links with `effective_goodput_gbps >= 18.0` and `rtt_latency_ms <= 0.25` are marked eligible by
//!    the planner; this does not establish native model execution on the link.
//! 4. **Topology Scheduling & Delta-Bridge Failover**:
//!    [`MultiPathInterfaceStriper`](crate::MultiPathInterfaceStriper) computes schedules across USB-C,
//!    Wi-Fi, and Ethernet. Actual ordered-stream failover is provided by [`crate::UsbCBridge`], whose
//!    default idle detection and resume grace are 750 ms and 30 s, not a sub-200 ms zero-downtime SLA.
//! 5. **Delta Bridge Plan** ([`UsbCDeltaBridgePlan`]): sizes a [`crate::UsbCBridge`] that stripes one
//!    ordered stream over the USB-C cable and its backup paths. Delta links layered on it keep their
//!    chunk stores through an unplug, so failing over resends at most the bridge window instead of
//!    every chunk the peer already held.
//!
//! ## Transport paths (as of October 2026)
//!
//! [`UsbCTransportPath`] records which byte transports each OS exposes on a USB-C link. The bridge
//! runs over the IP paths today; the other two are detected so placement can prefer them once a
//! lane type exists for them.
//! - **USB4 / Thunderbolt IP networking** is built in everywhere: Linux `thunderbolt-net`, macOS
//!   Thunderbolt Bridge, and the Windows 11 `USB4P2PNetAdapter`.
//! - **CDC-NCM** carries USB 3.x gadget links; transfer blocks reach 64 KiB only when both the
//!   gadget and the host's `cdc_ncm` `rx_max` / `tx_max` opt in (the default is 16 KiB).
//! - **USB4STREAM** (`thunderbolt_stream`, Linux 7.2) exposes raw DMA tunnels as `/dev/tbstreamX`,
//!   configured under `/sys/kernel/config/thunderbolt/stream/`, without the network stack.
//! - **RDMA over Thunderbolt 5** (macOS 26.2+, enabled with `rdma_ctl enable` from recovery) is
//!   what MLX's JACCL backend uses; it needs Thunderbolt 5 ports on both ends.

use crate::usbc_bridge::{
    BridgeLane, DEFAULT_BRIDGE_SEGMENT_BYTES, DEFAULT_LANE_IDLE_TIMEOUT, MAX_BRIDGE_WINDOW_BYTES,
};
use crate::{MultiPathInterfaceStriper, MultiPathStripedSchedule, PeerLinkMetrics, TransportKind};
use kadmium_core::{
    ComputeBackend, DeviceKind, HardwareNode, MemoryTierProfile, NodeId, NodeRole, NodeTrustTier,
    PoliteState,
};
use serde::{Deserialize, Serialize};
#[cfg(any(target_os = "windows", target_os = "macos"))]
use std::process::Command;

/// Physical & protocol tier of a USB-C direct interconnect link.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsbCInterconnectTier {
    /// USB4 v2 / Thunderbolt 5 Host-to-Host PCIe DMA Tunnel (`80–120 Gbps` rated, `~64.0 Gbps` goodput, `0.08 ms` RTT).
    Thunderbolt5Usb4V2_80Gbps,
    /// USB4 v1 / Thunderbolt 3 & 4 profile (`30–40 Gbps` rated; goodput/RTT fields are modelled estimates).
    Usb4Thunderbolt4_30To40Gbps,
    /// USB 3.2 Gen 2x2 (`20 Gbps`) Dual-Role-Data (`DRD`) `USB-NCM` / `FunctionFS` Bulk (`~15.2 Gbps` goodput, `0.18 ms` RTT).
    Usb32Gen2x2_20GbpsDrd,
    /// USB 3.2 Gen 2 (`10 Gbps`) Dual-Role-Data (`DRD`) `USB-NCM` / `FunctionFS` Bulk (`~8.2 Gbps` goodput, `0.22 ms` RTT).
    Usb32Gen2_10GbpsDrd,
    /// USB 3.2 Gen 1 (`5 Gbps`) Dual-Role-Data (`DRD`) `USB-NCM` / `ADB` Bulk (`~3.8 Gbps` goodput, `0.30 ms` RTT).
    Usb32Gen1_5GbpsDrd,
    /// USB 2.0 High-Speed (`480 Mbps`) CDC-ACM / Bulk Co-Processor link (`~0.38 Gbps` goodput, `0.85 ms` RTT).
    Usb20HighSpeed_480Mbps,
}

impl UsbCInterconnectTier {
    /// Returns the rated physical layer bandwidth in Gbps.
    pub fn rated_phy_gbps(&self) -> f64 {
        match self {
            Self::Thunderbolt5Usb4V2_80Gbps => 80.0,
            Self::Usb4Thunderbolt4_30To40Gbps => 30.0,
            Self::Usb32Gen2x2_20GbpsDrd => 20.0,
            Self::Usb32Gen2_10GbpsDrd => 10.0,
            Self::Usb32Gen1_5GbpsDrd => 5.0,
            Self::Usb20HighSpeed_480Mbps => 0.48,
        }
    }

    /// Returns the sustained multi-stream DMA / NCM goodput in Gbps under Kadmium's `64 KiB` striped framing.
    pub fn effective_goodput_gbps(&self) -> f64 {
        match self {
            Self::Thunderbolt5Usb4V2_80Gbps => 64.0,
            Self::Usb4Thunderbolt4_30To40Gbps => 27.6,
            Self::Usb32Gen2x2_20GbpsDrd => 15.2,
            Self::Usb32Gen2_10GbpsDrd => 8.2,
            Self::Usb32Gen1_5GbpsDrd => 3.8,
            Self::Usb20HighSpeed_480Mbps => 0.38,
        }
    }

    /// Typical point-to-point round-trip latency in milliseconds.
    pub fn typical_rtt_ms(&self) -> f64 {
        match self {
            Self::Thunderbolt5Usb4V2_80Gbps => 0.08,
            Self::Usb4Thunderbolt4_30To40Gbps => 0.12,
            Self::Usb32Gen2x2_20GbpsDrd => 0.18,
            Self::Usb32Gen2_10GbpsDrd => 0.22,
            Self::Usb32Gen1_5GbpsDrd => 0.30,
            Self::Usb20HighSpeed_480Mbps => 0.85,
        }
    }

    /// Maps the USB-C tier to Kadmium's mesh `TransportKind`.
    pub fn to_transport_kind(&self) -> TransportKind {
        match self {
            Self::Thunderbolt5Usb4V2_80Gbps
            | Self::Usb4Thunderbolt4_30To40Gbps
            | Self::Usb32Gen2x2_20GbpsDrd
            | Self::Usb32Gen2_10GbpsDrd => TransportKind::ThunderboltBridge,
            Self::Usb32Gen1_5GbpsDrd | Self::Usb20HighSpeed_480Mbps => TransportKind::WiredLanQuic,
        }
    }
}

/// Byte transport available on a USB-C link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsbCTransportPath {
    /// IP over the OS's USB4 / Thunderbolt network driver (Linux `thunderbolt-net`, macOS
    /// Thunderbolt Bridge, Windows `USB4P2PNetAdapter`).
    Usb4IpNetwork,
    /// IP over a USB 3.x CDC-NCM gadget link (`usb_f_ncm` / `cdc_ncm`).
    CdcNcmIp,
    /// Raw USB4 DMA stream from Linux 7.2's `thunderbolt_stream` driver (`/dev/tbstreamX`).
    Usb4RawStream,
    /// RDMA over Thunderbolt 5 (macOS 26.2+, `rdma_ctl status` reports `enabled`).
    RdmaThunderbolt5,
}

impl UsbCTransportPath {
    /// Whether a [`crate::UsbCBridge`] lane can run over this path today (the IP paths).
    pub fn carries_bridge_lanes(&self) -> bool {
        matches!(self, Self::Usb4IpNetwork | Self::CdcNcmIp)
    }
}

/// USB-C port data role negotiation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsbCDataRole {
    /// Symmetric Host-to-Host USB4 / Thunderbolt Inter-Domain PCIe DMA tunnel (PC ⇄ Handheld ⇄ Mac).
    HostToHostSymmetricP2p,
    /// Downstream Facing Port (`DFP` / Host side) of a USB 3.x Dual-Role-Data connection.
    DualRoleHostDfp,
    /// Upstream Facing Port (`UFP` / Device Gadget `usb_f_ncm` side) of a USB 3.x Dual-Role-Data connection.
    DualRoleGadgetUfp,
}

/// Descriptor for an active or available USB-C / USB4 / Thunderbolt direct interconnect interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbCInterfaceDescriptor {
    pub interface_id: String,
    pub interface_name: String,
    pub os_driver_stack: String,
    pub tier: UsbCInterconnectTier,
    pub data_role: UsbCDataRole,
    pub link_local_ipv4: String,
    pub link_local_ipv6: String,
    pub peer_link_local_addr: Option<String>,
    pub negotiated_mtu_bytes: usize,
    pub parallel_dma_streams: usize,
    pub rated_phy_gbps: f64,
    pub effective_goodput_gbps: f64,
    pub rtt_latency_ms: f64,
    pub supports_tensor_parallel_allreduce: bool,
    pub live_detected_on_host: bool,
    /// Byte transports this interface offers, IP paths first.
    #[serde(default)]
    pub transport_paths: Vec<UsbCTransportPath>,
}

/// Benchmark comparing naive single-stream TCP (`1500 B` MTU) vs. Kadmium Multi-Stream DMA Striping
/// (`4x–8x` parallel lanes + `64 KiB` frames) over a high-bandwidth USB-C link.
///
/// All values in this struct are deterministic arithmetic model estimates derived from `rated_cable_gbps`
/// and `rtt_ms`; no physical USB-C or PCIe DMA transfer is executed by [`UsbCDirectInterconnectScanner::benchmark_multistream_dma`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiStreamUsbCDmaBenchmark {
    /// Explicit provenance tag (`"modelled_arithmetic_estimate"`).
    pub measurement_kind: String,
    /// Always `false` for planner/model calculations that do not execute a physical DMA transfer.
    pub live_dma_measured: bool,
    /// Explicit statement of the arithmetic assumptions behind the figures below.
    pub assumptions_note: String,
    pub rated_cable_gbps: f64,
    pub single_stream_1500_mtu_gbps: f64,
    pub kadmium_multistream_striped_gbps: f64,
    pub sustained_throughput_gb_per_sec: f64,
    pub parallel_socket_lanes: usize,
    pub negotiated_mtu_bytes: usize,
    /// Modelled time in microseconds to transfer a `64 KiB` FP16 hidden-state activation tensor (`4096 x 8`).
    pub activation_64kb_transfer_us: f64,
    /// Modelled time in milliseconds to drain a `512 MB` active KV-cache window at the rated USB-C goodput.
    pub kv_window_512mb_drain_ms: f64,
    /// Modelled time in milliseconds to drain a full `2.0 GB` model/KV shard over the USB-C cable.
    pub full_shard_2gb_drain_ms: f64,
    /// Speedup factor compared to a standard `1.0 Gbps` RJ45 Ethernet patch cable (`~0.94 Gbps` goodput).
    pub vs_1gbe_patch_cable_speedup: f64,
    /// Speedup factor compared to a `2.5 Gbps` Ethernet connection (`~2.35 Gbps` goodput).
    pub vs_2_5gbe_patch_cable_speedup: f64,
    /// Whether this modelled USB-C profile qualifies for intra-layer Tensor Parallelism (`TP`) and MoE All-to-All in the planner.
    pub unlocks_tensor_parallelism: bool,
}

/// Hop descriptor inside a switchless multi-port USB-C daisy-chain or ring topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbCDaisyChainHop {
    pub hop_index: usize,
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub tier: UsbCInterconnectTier,
    pub effective_goodput_gbps: f64,
    pub hop_rtt_ms: f64,
    pub link_local_subnet: String,
    pub tensor_parallel_eligible: bool,
}

/// Switchless multi-port USB-C daisy-chain / ring topology plan with automatic Wi-Fi/Ethernet failover.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbCDaisyChainTopologyPlan {
    pub topology_mode: String,
    pub total_usb_c_nodes: usize,
    pub ring_closed: bool,
    pub bottleneck_goodput_gbps: f64,
    pub end_to_end_ring_rtt_ms: f64,
    pub hops: Vec<UsbCDaisyChainHop>,
    /// Multi-homed bonding & hot-unplug failover schedule combining USB-C + LAN/Wi-Fi backup paths.
    pub bonded_failover_schedule: MultiPathStripedSchedule,
}

/// Complete discovery & capability report for USB-C / USB4 / Thunderbolt direct interconnects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbCHostDiscoveryReport {
    pub host_os: String,
    /// Explicit provenance tag (`"host_adapter_scan_with_modelled_estimates"` or `"hermetic_reference_catalog_only"`).
    pub measurement_kind: String,
    /// Always `false` unless a physical peer-to-peer DMA transfer benchmark was executed.
    pub live_dma_measured: bool,
    pub live_usb_c_interfaces_detected: usize,
    /// True only when at least one live USB-C / USB4 / Thunderbolt adapter was detected on this host.
    pub zero_config_link_local_ready: bool,
    /// Live host interfaces detected on the OS (`live_detected_on_host == true`).
    pub live_interfaces: Vec<UsbCInterfaceDescriptor>,
    /// Reference tier descriptors (`live_detected_on_host == false`) used for capability comparison and offline planning.
    pub reference_tier_catalog: Vec<UsbCInterfaceDescriptor>,
    /// Combined interface list (live interfaces first, followed by reference tier descriptors).
    pub interfaces: Vec<UsbCInterfaceDescriptor>,
    pub handheld_30gbps_dma_benchmark: MultiStreamUsbCDmaBenchmark,
    pub switchless_daisy_chain_plan: UsbCDaisyChainTopologyPlan,
    /// Bridge sizing for the first interface, backed by the daisy chain's LAN / Wi-Fi paths.
    pub delta_bridge_plan: UsbCDeltaBridgePlan,
}

/// Sizing of a [`crate::UsbCBridge`] over one USB-C link plus its backup paths, with the cost of
/// an unplug compared against a plain connection. These are model estimates from tier goodput and
/// RTT, not measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbCDeltaBridgePlan {
    /// The USB-C lanes first (one per DMA stream), then one per backup path. Backup lanes carry a
    /// placeholder address: the peer's address on that path is only known once it registers.
    pub lanes: Vec<BridgeLane>,
    pub segment_bytes: usize,
    /// Unacknowledged bytes per direction: enough for the USB-C lane to keep sending while a
    /// segment crosses the slowest backup lane.
    pub window_bytes: usize,
    pub bonded_goodput_gbps: f64,
    /// Fraction of payload bytes assumed to be chunks the peer already holds.
    pub chunk_reuse_fraction: f64,
    /// Payload bytes delivered per second once reused chunks travel as 13-byte references.
    pub logical_goodput_gbps: f64,
    /// Time for a pulled cable to be noticed (no data or heartbeat for the lane idle timeout).
    pub unplug_detect_ms: f64,
    /// Bytes resent after an unplug: the USB-C lane's share of the window, at most.
    pub unplug_resend_bytes: u64,
    /// Detection plus resending those bytes over the backup lanes.
    pub unplug_recovery_ms: f64,
    /// Bytes a plain connection resends after an unplug: its delta store starts empty, so every
    /// hot chunk (up to the store budget) crosses the backup path again.
    pub plain_reconnect_resend_bytes: u64,
    pub plain_reconnect_recovery_ms: f64,
}

/// Cross-platform USB-C / USB4 / Thunderbolt & Dual-Role-Data (`USB-NCM`) scanner and DMA scheduler.
pub struct UsbCDirectInterconnectScanner;

impl UsbCDirectInterconnectScanner {
    /// Computes the modelled multi-stream DMA performance estimate for a given rated USB-C bandwidth (e.g. `30.0 Gbps`).
    pub fn benchmark_multistream_dma(rated_gbps: f64, rtt_ms: f64) -> MultiStreamUsbCDmaBenchmark {
        let clamped_rated = rated_gbps.clamp(0.48, 120.0);
        // Without multi-stream striping or jumbo frames, single-stream 1500B MTU TCP over USB4/TB
        // typically bottlenecks around 9.5–11.5 Gbps due to per-packet NDIS/sk_buff overhead.
        let single_stream_gbps = if clamped_rated >= 20.0 {
            (clamped_rated * 0.36).min(11.8)
        } else {
            clamped_rated * 0.62
        };

        // Modelled estimate assuming 4x–8x parallel socket lanes + 64 KiB frame batching at
        // 92% of the rated link bandwidth (~27.6 Gbps on a 30 Gbps link).
        let parallel_lanes = if clamped_rated >= 60.0 {
            8
        } else if clamped_rated >= 20.0 {
            6
        } else if clamped_rated >= 5.0 {
            4
        } else {
            2
        };

        let striped_goodput_gbps = ((clamped_rated * 0.92) * 100.0).round() / 100.0;
        let sustained_gb_per_sec = ((striped_goodput_gbps / 8.0) * 100.0).round() / 100.0;

        // 64 KiB = 65,536 bytes = 524,288 bits = 0.000524288 Gbit
        let wire_serialize_us = (0.000_524_288 / striped_goodput_gbps.max(0.1)) * 1_000_000.0;
        let one_way_latency_us = (rtt_ms * 0.5) * 1000.0;
        let activation_64kb_transfer_us =
            ((wire_serialize_us + one_way_latency_us) * 10.0).round() / 10.0;

        // 512 MB = 4.096 Gbit; 2.0 GB = 16.384 Gbit
        let kv_window_512mb_drain_ms =
            (((4.096 / striped_goodput_gbps.max(0.1)) * 1000.0) * 10.0).round() / 10.0;
        let full_shard_2gb_drain_ms =
            (((16.384 / striped_goodput_gbps.max(0.1)) * 1000.0) * 10.0).round() / 10.0;

        let vs_1gbe = ((striped_goodput_gbps / 0.94) * 10.0).round() / 10.0;
        let vs_2_5gbe = ((striped_goodput_gbps / 2.35) * 10.0).round() / 10.0;

        MultiStreamUsbCDmaBenchmark {
            measurement_kind: "modelled_arithmetic_estimate".into(),
            live_dma_measured: false,
            assumptions_note: "Derived from rated_cable_gbps * 0.92 and payload / goodput division; no physical USB-C or PCIe DMA transfer was executed.".into(),
            rated_cable_gbps: clamped_rated,
            single_stream_1500_mtu_gbps: (single_stream_gbps * 100.0).round() / 100.0,
            kadmium_multistream_striped_gbps: striped_goodput_gbps,
            sustained_throughput_gb_per_sec: sustained_gb_per_sec,
            parallel_socket_lanes: parallel_lanes,
            negotiated_mtu_bytes: if clamped_rated >= 10.0 {
                65_536
            } else {
                16_384
            },
            activation_64kb_transfer_us,
            kv_window_512mb_drain_ms,
            full_shard_2gb_drain_ms,
            vs_1gbe_patch_cable_speedup: vs_1gbe,
            vs_2_5gbe_patch_cable_speedup: vs_2_5gbe,
            unlocks_tensor_parallelism: striped_goodput_gbps >= 18.0 && rtt_ms <= 0.25,
        }
    }

    /// Probes the host OS for physical or virtual USB4 / Thunderbolt / USB-NCM / RNDIS network adapters,
    /// and returns the USB-C direct interconnect discovery report.
    pub fn scan_host_usbc_links(cluster_nodes: &[HardwareNode]) -> UsbCHostDiscoveryReport {
        let allow_live = kadmium_core::external_hardware_probes_allowed(None);
        Self::scan_host_usbc_links_with_options(cluster_nodes, allow_live)
    }

    /// Returns the USB-C direct interconnect discovery report, probing the OS only when `allow_live_os_probe` is `true`.
    pub fn scan_host_usbc_links_with_options(
        cluster_nodes: &[HardwareNode],
        allow_live_os_probe: bool,
    ) -> UsbCHostDiscoveryReport {
        let live_interfaces: Vec<UsbCInterfaceDescriptor> = if allow_live_os_probe {
            Self::probe_os_usbc_adapters()
                .into_iter()
                .filter(|i| i.live_detected_on_host)
                .collect()
        } else {
            Vec::new()
        };
        let live_count = live_interfaces.len();

        let reference_tier_catalog = vec![
            UsbCInterfaceDescriptor {
                interface_id: "usbc-p2p-usb4-30g".into(),
                interface_name:
                    "USB4(TM) / Thunderbolt P2P Direct Tunnel (usb4_p2p0, reference profile)".into(),
                os_driver_stack: Self::default_os_driver_label(),
                tier: UsbCInterconnectTier::Usb4Thunderbolt4_30To40Gbps,
                data_role: UsbCDataRole::HostToHostSymmetricP2p,
                link_local_ipv4: "169.254.78.10".into(),
                link_local_ipv6: "fe80::7848:usb4:10".into(),
                peer_link_local_addr: Some("169.254.78.11:7848".into()),
                negotiated_mtu_bytes: 65_536,
                parallel_dma_streams: 6,
                rated_phy_gbps: 30.0,
                effective_goodput_gbps: 27.6,
                rtt_latency_ms: 0.12,
                supports_tensor_parallel_allreduce: true,
                live_detected_on_host: false,
                transport_paths: vec![UsbCTransportPath::Usb4IpNetwork],
            },
            UsbCInterfaceDescriptor {
                interface_id: "usbc-p2p-tb5-80g".into(),
                interface_name:
                    "Thunderbolt 5 / USB4 v2 Symmetric DMA Tunnel (tb5_p2p0, reference profile)"
                        .into(),
                os_driver_stack: Self::default_os_driver_label(),
                tier: UsbCInterconnectTier::Thunderbolt5Usb4V2_80Gbps,
                data_role: UsbCDataRole::HostToHostSymmetricP2p,
                link_local_ipv4: "169.254.78.20".into(),
                link_local_ipv6: "fe80::7848:tb5:20".into(),
                peer_link_local_addr: Some("169.254.78.21:7848".into()),
                negotiated_mtu_bytes: 65_536,
                parallel_dma_streams: 8,
                rated_phy_gbps: 80.0,
                effective_goodput_gbps: 64.0,
                rtt_latency_ms: 0.08,
                supports_tensor_parallel_allreduce: true,
                live_detected_on_host: false,
                transport_paths: vec![UsbCTransportPath::Usb4IpNetwork],
            },
            UsbCInterfaceDescriptor {
                interface_id: "usbc-drd-ncm-20g".into(),
                interface_name:
                    "USB 3.2 Gen 2x2 Dual-Role USB-NCM Gadget Bridge (usb_ncm0, reference profile)"
                        .into(),
                os_driver_stack: "CDC-NCM / FunctionFS Zero-Copy Bulk".into(),
                tier: UsbCInterconnectTier::Usb32Gen2x2_20GbpsDrd,
                data_role: UsbCDataRole::DualRoleHostDfp,
                link_local_ipv4: "169.254.78.30".into(),
                link_local_ipv6: "fe80::7848:ncm:30".into(),
                peer_link_local_addr: Some("169.254.78.31:7848".into()),
                negotiated_mtu_bytes: 65_536,
                parallel_dma_streams: 4,
                rated_phy_gbps: 20.0,
                effective_goodput_gbps: 15.2,
                rtt_latency_ms: 0.18,
                supports_tensor_parallel_allreduce: false,
                live_detected_on_host: false,
                transport_paths: vec![UsbCTransportPath::CdcNcmIp],
            },
        ];

        let mut interfaces = live_interfaces.clone();
        interfaces.extend(reference_tier_catalog.clone());

        let handheld_30gbps_dma_benchmark = Self::benchmark_multistream_dma(30.0, 0.12);
        let switchless_daisy_chain_plan = Self::build_daisy_chain_topology_plan(cluster_nodes);
        let backups: Vec<PeerLinkMetrics> = switchless_daisy_chain_plan
            .bonded_failover_schedule
            .channel_allocations
            .iter()
            .filter(|c| c.transport != TransportKind::ThunderboltBridge)
            .map(|c| PeerLinkMetrics {
                from_node: NodeId("local".into()),
                to_node: NodeId("peer".into()),
                transport: c.transport,
                bandwidth_gbps: c.bandwidth_gbps,
                rtt_latency_ms: c.rtt_ms,
            })
            .collect();
        let delta_bridge_plan = if let Some(live_primary) = live_interfaces.first() {
            Self::plan_delta_bridge(live_primary, &backups, 0.9)
        } else {
            UsbCDeltaBridgePlan {
                lanes: Vec::new(),
                segment_bytes: DEFAULT_BRIDGE_SEGMENT_BYTES,
                window_bytes: 0,
                bonded_goodput_gbps: 0.0,
                chunk_reuse_fraction: 0.0,
                logical_goodput_gbps: 0.0,
                unplug_detect_ms: 0.0,
                unplug_resend_bytes: 0,
                unplug_recovery_ms: 0.0,
                plain_reconnect_resend_bytes: 0,
                plain_reconnect_recovery_ms: 0.0,
            }
        };

        UsbCHostDiscoveryReport {
            host_os: std::env::consts::OS.to_string(),
            measurement_kind: if allow_live_os_probe {
                "host_adapter_scan_with_modelled_estimates".into()
            } else {
                "hermetic_reference_catalog_only".into()
            },
            live_dma_measured: false,
            live_usb_c_interfaces_detected: live_count,
            zero_config_link_local_ready: live_count > 0,
            live_interfaces,
            reference_tier_catalog,
            interfaces,
            handheld_30gbps_dma_benchmark,
            switchless_daisy_chain_plan,
            delta_bridge_plan,
        }
    }

    /// Sizes a bridge over `primary` with `backups` as further lanes, assuming
    /// `chunk_reuse_fraction` of payload bytes are chunks the peer already holds (0.9 is typical
    /// of decode steps that resend an unchanged context with fresh activations).
    pub fn plan_delta_bridge(
        primary: &UsbCInterfaceDescriptor,
        backups: &[PeerLinkMetrics],
        chunk_reuse_fraction: f64,
    ) -> UsbCDeltaBridgePlan {
        const MIB: f64 = (1u64 << 20) as f64;
        /// A reused chunk costs a 13-byte reference instead of its ~2 KiB (smallest average).
        const REF_COST_PER_REUSED_BYTE: f64 = 13.0 / 2048.0;
        /// Default per-direction delta store a plain reconnect has to refill.
        const DELTA_STORE_BYTES: f64 = 128.0 * MIB;

        let reuse = chunk_reuse_fraction.clamp(0.0, 1.0);
        let primary_gbps = primary.effective_goodput_gbps.max(0.01);
        let segment = DEFAULT_BRIDGE_SEGMENT_BYTES as f64;

        // One lane per DMA stream on the cable: each lane is its own TCP connection, and a single
        // connection cannot fill a USB4 link (on loopback one lane carries roughly a fifth of what
        // four do).
        let streams = primary.parallel_dma_streams.max(1);
        let primary_addr = primary
            .peer_link_local_addr
            .clone()
            .unwrap_or_else(|| primary.link_local_ipv4.clone());
        let mut lanes: Vec<BridgeLane> = (0..streams)
            .map(|i| {
                BridgeLane::new(
                    primary_addr.clone(),
                    primary_gbps / streams as f64,
                    format!("{}-{i}", primary.interface_id),
                )
            })
            .collect();
        let mut slowest_segment_s: f64 = 0.0;
        let mut backup_gbps = 0.0;
        for (i, link) in backups.iter().enumerate() {
            let gbps = link.bandwidth_gbps.max(0.01);
            backup_gbps += gbps;
            let segment_s = segment * 8.0 / (gbps * 1e9) + link.rtt_latency_ms / 1000.0;
            slowest_segment_s = slowest_segment_s.max(segment_s);
            lanes.push(BridgeLane::new(
                format!("<peer {:?} address>", link.transport),
                gbps,
                format!("backup{i}-{:?}", link.transport),
            ));
        }
        // Twice what the USB-C lane sends while one segment crosses the slowest backup lane, so
        // head-of-line waits on the slow lane never idle the fast one.
        let window_bytes = (2.0 * primary_gbps * 1e9 / 8.0 * slowest_segment_s)
            .max(4.0 * MIB)
            .min(MAX_BRIDGE_WINDOW_BYTES as f64);
        let window_bytes = ((window_bytes / MIB).ceil() * MIB) as usize;

        let bonded_goodput_gbps = primary_gbps + backup_gbps;
        let wire_per_logical = (1.0 - reuse) + reuse * REF_COST_PER_REUSED_BYTE;
        let logical_goodput_gbps = bonded_goodput_gbps / wire_per_logical;

        let unplug_detect_ms = DEFAULT_LANE_IDLE_TIMEOUT.as_secs_f64() * 1000.0;
        let primary_share = primary_gbps / bonded_goodput_gbps;
        let unplug_resend_bytes = (window_bytes as f64 * primary_share) as u64;
        let backup_bytes_per_ms = backup_gbps.max(0.01) * 1e9 / 8.0 / 1000.0;
        let unplug_recovery_ms =
            unplug_detect_ms + unplug_resend_bytes as f64 / backup_bytes_per_ms;
        let plain_reconnect_resend_bytes = DELTA_STORE_BYTES as u64;
        let plain_reconnect_recovery_ms =
            unplug_detect_ms + plain_reconnect_resend_bytes as f64 / backup_bytes_per_ms;

        let round = |v: f64| (v * 100.0).round() / 100.0;
        UsbCDeltaBridgePlan {
            lanes,
            segment_bytes: DEFAULT_BRIDGE_SEGMENT_BYTES,
            window_bytes,
            bonded_goodput_gbps: round(bonded_goodput_gbps),
            chunk_reuse_fraction: reuse,
            logical_goodput_gbps: round(logical_goodput_gbps),
            unplug_detect_ms: round(unplug_detect_ms),
            unplug_resend_bytes,
            unplug_recovery_ms: round(unplug_recovery_ms),
            plain_reconnect_resend_bytes,
            plain_reconnect_recovery_ms: round(plain_reconnect_recovery_ms),
        }
    }

    /// `base` plus the transports this host exposes beyond IP networking.
    fn probe_transport_paths(base: UsbCTransportPath) -> Vec<UsbCTransportPath> {
        #[cfg_attr(not(any(target_os = "linux", target_os = "macos")), allow(unused_mut))]
        let mut paths = vec![base];
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/dev/tbstream0").exists()
                || std::path::Path::new("/sys/kernel/config/thunderbolt/stream").exists()
            {
                paths.push(UsbCTransportPath::Usb4RawStream);
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("rdma_ctl").arg("status").output()
                && String::from_utf8_lossy(&output.stdout).trim() == "enabled"
            {
                paths.push(UsbCTransportPath::RdmaThunderbolt5);
            }
        }
        paths
    }

    /// Builds a switchless USB-C daisy-chain / ring topology plan across the supplied cluster nodes.
    ///
    /// Does not invent synthetic peer nodes when fewer than two nodes are supplied:
    /// - `nodes.is_empty()` returns `total_usb_c_nodes: 0` and no hops or failover allocations.
    /// - `nodes.len() == 1` returns `total_usb_c_nodes: 1` and no hops or failover allocations.
    pub fn build_daisy_chain_topology_plan(nodes: &[HardwareNode]) -> UsbCDaisyChainTopologyPlan {
        if nodes.is_empty() {
            return UsbCDaisyChainTopologyPlan {
                topology_mode: "No USB-C cluster nodes supplied (0 nodes)".into(),
                total_usb_c_nodes: 0,
                ring_closed: false,
                bottleneck_goodput_gbps: 0.0,
                end_to_end_ring_rtt_ms: 0.0,
                hops: Vec::new(),
                bonded_failover_schedule: MultiPathStripedSchedule {
                    total_chunks: 0,
                    chunk_size_kb: 0,
                    bonded_bandwidth_gbps: 0.0,
                    single_best_link_gbps: 0.0,
                    multi_path_speedup: 0.0,
                    estimated_transfer_ms: 0.0,
                    channel_allocations: Vec::new(),
                },
            };
        }

        if nodes.len() == 1 {
            return UsbCDaisyChainTopologyPlan {
                topology_mode: "Single host node only (no peer nodes to form a USB-C link)".into(),
                total_usb_c_nodes: 1,
                ring_closed: false,
                bottleneck_goodput_gbps: 0.0,
                end_to_end_ring_rtt_ms: 0.0,
                hops: Vec::new(),
                bonded_failover_schedule: MultiPathStripedSchedule {
                    total_chunks: 0,
                    chunk_size_kb: 0,
                    bonded_bandwidth_gbps: 0.0,
                    single_best_link_gbps: 0.0,
                    multi_path_speedup: 0.0,
                    estimated_transfer_ms: 0.0,
                    channel_allocations: Vec::new(),
                },
            };
        }

        let chain_nodes: &[HardwareNode] = &nodes[..nodes.len().min(4)];
        let mut hops = Vec::new();
        let mut min_goodput = f64::MAX;
        let mut total_rtt = 0.0f64;

        let count = chain_nodes.len();
        for idx in 0..count {
            let next_idx = (idx + 1) % count;
            if count == 2 && idx == 1 {
                break;
            }
            let a = &chain_nodes[idx];
            let b = &chain_nodes[next_idx];
            let raw_bw = a.network_uplink_gbps.max(b.network_uplink_gbps).max(30.0);
            let tier = if raw_bw >= 60.0 {
                UsbCInterconnectTier::Thunderbolt5Usb4V2_80Gbps
            } else if raw_bw >= 25.0 {
                UsbCInterconnectTier::Usb4Thunderbolt4_30To40Gbps
            } else if raw_bw >= 15.0 {
                UsbCInterconnectTier::Usb32Gen2x2_20GbpsDrd
            } else {
                UsbCInterconnectTier::Usb32Gen2_10GbpsDrd
            };

            let goodput = tier.effective_goodput_gbps();
            let rtt = tier.typical_rtt_ms();
            if goodput < min_goodput {
                min_goodput = goodput;
            }
            total_rtt += rtt;

            hops.push(UsbCDaisyChainHop {
                hop_index: idx,
                from_node: a.id.clone(),
                to_node: b.id.clone(),
                tier,
                effective_goodput_gbps: goodput,
                hop_rtt_ms: rtt,
                link_local_subnet: format!("169.254.{}.0/30", 78 + idx),
                tensor_parallel_eligible: goodput >= 18.0 && rtt <= 0.25,
            });
        }

        // Construct a modelled bonded primary (30 Gbps USB4) + backup (2.5 GbE LAN + Wi-Fi 6E) schedule
        // across the first two supplied nodes.
        let primary_from = chain_nodes[0].id.clone();
        let primary_to = chain_nodes[1].id.clone();

        let failover_links = vec![
            PeerLinkMetrics {
                from_node: primary_from.clone(),
                to_node: primary_to.clone(),
                transport: TransportKind::ThunderboltBridge,
                bandwidth_gbps: 27.6,
                rtt_latency_ms: 0.12,
            },
            PeerLinkMetrics {
                from_node: primary_from.clone(),
                to_node: primary_to.clone(),
                transport: TransportKind::WiredLanQuic,
                bandwidth_gbps: 2.5,
                rtt_latency_ms: 0.45,
            },
            PeerLinkMetrics {
                from_node: primary_from,
                to_node: primary_to,
                transport: TransportKind::WirelessLanQuic,
                bandwidth_gbps: 1.2,
                rtt_latency_ms: 1.60,
            },
        ];

        let bonded_failover_schedule =
            MultiPathInterfaceStriper::schedule_multipath_stripes(&failover_links, 512.0);

        UsbCDaisyChainTopologyPlan {
            topology_mode: if count >= 3 {
                "Modelled Switchless USB4 Ring / Daisy-Chain (169.254.x.x Point-to-Point)".into()
            } else {
                "Modelled Direct Host-to-Host USB4 / Thunderbolt Point-to-Point".into()
            },
            total_usb_c_nodes: count,
            ring_closed: count >= 3,
            bottleneck_goodput_gbps: (min_goodput * 100.0).round() / 100.0,
            end_to_end_ring_rtt_ms: (total_rtt * 100.0).round() / 100.0,
            hops,
            bonded_failover_schedule,
        }
    }

    /// Creates a platform-generalized `HardwareNode` representing a modern Handheld Gaming System
    /// (e.g. AMD Ryzen Z1 Extreme / Ryzen AI HX 370 / Intel Lunar Lake handheld with `24–32 GB`
    /// LPDDR5X unified memory) connected via a **30 Gbps rated USB-C Thunderbolt / USB4** cable.
    pub fn build_usb4_handheld_gaming_node(
        node_id: Option<&str>,
        rated_usb_c_gbps: Option<f64>,
    ) -> HardwareNode {
        let rated_gbps = rated_usb_c_gbps.unwrap_or(30.0).clamp(5.0, 120.0);
        let bench = Self::benchmark_multistream_dma(rated_gbps, 0.12);

        HardwareNode {
            id: NodeId(node_id.unwrap_or("node-usb4-handheld-30gbps").into()),
            hostname: node_id
                .unwrap_or("usb4-handheld-30gbps")
                .trim_start_matches("node-")
                .to_string(),
            device_kind: DeviceKind::GamingLaptop,
            backend: ComputeBackend::VulkanWgpu,
            roles: vec![NodeRole::ComputeWorker, NodeRole::DraftAndUtilityWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            gpu_name: format!(
                "Handheld Gaming APU (RDNA3.5 + XDNA NPU · {:.0}Gbps USB4/TB Direct)",
                rated_gbps
            ),
            cpu_name: "AMD Ryzen Z1 Extreme / Ryzen AI HX 370 (8C/16T AVX-512)".into(),
            npu_name: Some("AMD XDNA / XDNA2 NPU".into()),
            npu_tops: 50.0,
            fp16_tflops: 36.4,
            int4_tops: 164.0,
            cpu_threads: 16,
            cpu_tflops: 9.2,
            memory: MemoryTierProfile {
                vram_gb: 0.0,
                vram_bw_gbps: 0.0,
                unified_mem_gb: 24.0,
                unified_mem_bw_gbps: 120.0, // LPDDR5X-7500 unified memory
                system_ram_gb: 24.0,
                system_ram_bw_gbps: 120.0,
                interconnect_bw_gbps: 120.0, // Zero-copy UMA aperture
                nvme_cache_gb: 512.0,
                nvme_read_bw_gbps: 7.0,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: bench.kadmium_multistream_striped_gbps,
            network_latency_ms: 0.12,
        }
    }

    fn default_os_driver_label() -> String {
        match std::env::consts::OS {
            "windows" => "Windows USB4(TM) NetAdapterCx / Thunderbolt P2P NDIS".into(),
            "macos" => "macOS AppleThunderboltIP (Thunderbolt Bridge)".into(),
            "linux" => "Linux thunderbolt-net / configfs usb_f_ncm".into(),
            other => format!("{} USB4/USB-NCM Generic Driver", other),
        }
    }

    fn probe_os_usbc_adapters() -> Vec<UsbCInterfaceDescriptor> {
        let mut out = Vec::new();

        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "Get-NetAdapter -ErrorAction SilentlyContinue | Select-Object -Property Name,InterfaceDescription,LinkSpeed,Status | ConvertTo-Json -Compress",
                ])
                .output()
            {
                let text = String::from_utf8_lossy(&output.stdout);
                let lower = text.to_lowercase();
                if lower.contains("usb4")
                    || lower.contains("thunderbolt")
                    || lower.contains("ncm")
                    || lower.contains("rndis")
                {
                    out.push(UsbCInterfaceDescriptor {
                        interface_id: "win-usb4-p2p-live".into(),
                        interface_name: "Windows USB4 / Thunderbolt P2P Adapter".into(),
                        os_driver_stack: "Windows USB4(TM) NetAdapterCx / NDIS".into(),
                        tier: UsbCInterconnectTier::Usb4Thunderbolt4_30To40Gbps,
                        data_role: UsbCDataRole::HostToHostSymmetricP2p,
                        link_local_ipv4: "169.254.78.10".into(),
                        link_local_ipv6: "fe80::7848:usb4:10".into(),
                        peer_link_local_addr: Some("169.254.78.11:7848".into()),
                        negotiated_mtu_bytes: 65_536,
                        parallel_dma_streams: 6,
                        rated_phy_gbps: 30.0,
                        effective_goodput_gbps: 27.6,
                        rtt_latency_ms: 0.12,
                        supports_tensor_parallel_allreduce: true,
                        live_detected_on_host: true,
                        transport_paths: Self::probe_transport_paths(UsbCTransportPath::Usb4IpNetwork),
                    });
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("ifconfig").arg("bridge0").output()
                && output.status.success()
            {
                out.push(UsbCInterfaceDescriptor {
                    interface_id: "macos-tb-bridge0-live".into(),
                    interface_name: "macOS Thunderbolt Bridge (bridge0)".into(),
                    os_driver_stack: "macOS AppleThunderboltIP".into(),
                    tier: UsbCInterconnectTier::Usb4Thunderbolt4_30To40Gbps,
                    data_role: UsbCDataRole::HostToHostSymmetricP2p,
                    link_local_ipv4: "169.254.78.10".into(),
                    link_local_ipv6: "fe80::7848:tb:10".into(),
                    peer_link_local_addr: Some("169.254.78.11:7848".into()),
                    negotiated_mtu_bytes: 65_536,
                    parallel_dma_streams: 6,
                    rated_phy_gbps: 40.0,
                    effective_goodput_gbps: 32.4,
                    rtt_latency_ms: 0.10,
                    supports_tensor_parallel_allreduce: true,
                    live_detected_on_host: true,
                    transport_paths: Self::probe_transport_paths(UsbCTransportPath::Usb4IpNetwork),
                });
            }
        }

        #[cfg(target_os = "linux")]
        {
            let tb_exists = std::path::Path::new("/sys/class/net/thunderbolt0").exists()
                || std::path::Path::new("/sys/bus/thunderbolt/devices").exists();
            let usb_ncm_exists = std::path::Path::new("/sys/class/net/usb0").exists();
            if tb_exists || usb_ncm_exists {
                out.push(UsbCInterfaceDescriptor {
                    interface_id: "linux-usbc-p2p-live".into(),
                    interface_name: if tb_exists {
                        "Linux thunderbolt0 P2P Tunnel".into()
                    } else {
                        "Linux usb0 Dual-Role USB-NCM".into()
                    },
                    os_driver_stack: "Linux thunderbolt-net / usb_f_ncm".into(),
                    tier: if tb_exists {
                        UsbCInterconnectTier::Usb4Thunderbolt4_30To40Gbps
                    } else {
                        UsbCInterconnectTier::Usb32Gen2x2_20GbpsDrd
                    },
                    data_role: if tb_exists {
                        UsbCDataRole::HostToHostSymmetricP2p
                    } else {
                        UsbCDataRole::DualRoleGadgetUfp
                    },
                    link_local_ipv4: "169.254.78.10".into(),
                    link_local_ipv6: "fe80::7848:usb4:10".into(),
                    peer_link_local_addr: Some("169.254.78.11:7848".into()),
                    negotiated_mtu_bytes: 65_536,
                    parallel_dma_streams: 6,
                    rated_phy_gbps: if tb_exists { 30.0 } else { 20.0 },
                    effective_goodput_gbps: if tb_exists { 27.6 } else { 15.2 },
                    rtt_latency_ms: if tb_exists { 0.12 } else { 0.18 },
                    supports_tensor_parallel_allreduce: tb_exists,
                    live_detected_on_host: true,
                    transport_paths: Self::probe_transport_paths(if tb_exists {
                        UsbCTransportPath::Usb4IpNetwork
                    } else {
                        UsbCTransportPath::CdcNcmIp
                    }),
                });
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::sample_personal_cluster;

    #[test]
    fn test_30gbps_usb4_handheld_multistream_dma_benchmark() {
        let bench = UsbCDirectInterconnectScanner::benchmark_multistream_dma(30.0, 0.12);
        assert_eq!(bench.rated_cable_gbps, 30.0);
        // Multi-stream striping must significantly beat naive single-stream 1500B MTU TCP
        assert!(bench.kadmium_multistream_striped_gbps >= 25.0);
        assert!(bench.kadmium_multistream_striped_gbps > bench.single_stream_1500_mtu_gbps * 2.0);
        // Must beat 1GbE by >25x and 2.5GbE by >10x
        assert!(bench.vs_1gbe_patch_cable_speedup >= 25.0);
        assert!(bench.vs_2_5gbe_patch_cable_speedup >= 10.0);
        // Must unlock Tensor Parallelism and drain a 512MB KV window in < 200ms on gaming/unplug
        assert!(bench.unlocks_tensor_parallelism);
        assert!(bench.kv_window_512mb_drain_ms < 200.0);
    }

    #[test]
    fn test_delta_bridge_plan_bounds_unplug_resend_to_the_window() {
        let cluster = sample_personal_cluster();
        let report =
            UsbCDirectInterconnectScanner::scan_host_usbc_links_with_options(&cluster, false);
        assert!(report.delta_bridge_plan.lanes.is_empty());

        let usbc = &report.reference_tier_catalog[0];
        let backups: Vec<PeerLinkMetrics> = report
            .switchless_daisy_chain_plan
            .bonded_failover_schedule
            .channel_allocations
            .iter()
            .filter(|c| c.transport != TransportKind::ThunderboltBridge)
            .map(|c| PeerLinkMetrics {
                from_node: NodeId("local".into()),
                to_node: NodeId("peer".into()),
                transport: c.transport,
                bandwidth_gbps: c.bandwidth_gbps,
                rtt_latency_ms: c.rtt_ms,
            })
            .collect();
        let plan = UsbCDirectInterconnectScanner::plan_delta_bridge(usbc, &backups, 0.9);
        assert_eq!(
            plan.lanes.len(),
            usbc.parallel_dma_streams + 2,
            "one lane per USB-C DMA stream plus LAN and Wi-Fi backups"
        );
        let wifi = plan.lanes.last().unwrap();
        assert!(plan.lanes[0].weight_gbps > wifi.weight_gbps);
        // The window covers the USB-C lane while a 64 KiB segment crosses Wi-Fi (~7 MB at 27.6 Gbps).
        assert!(plan.window_bytes >= 8 << 20 && plan.window_bytes <= 64 << 20);
        assert!(plan.logical_goodput_gbps > plan.bonded_goodput_gbps * 5.0);
        // An unplug resends at most the window, far less than refilling a 128 MiB delta store.
        assert!(plan.unplug_resend_bytes <= plan.window_bytes as u64);
        assert!(plan.unplug_resend_bytes * 8 < plan.plain_reconnect_resend_bytes);
        assert!(plan.unplug_recovery_ms < plan.plain_reconnect_recovery_ms);
        assert!(report.interfaces.iter().all(|i| {
            i.transport_paths
                .first()
                .is_some_and(|p| p.carries_bridge_lanes())
        }));
    }

    #[test]
    fn test_usbc_host_discovery_and_switchless_daisy_chain_ring() {
        let mut nodes = sample_personal_cluster();
        let handheld = UsbCDirectInterconnectScanner::build_usb4_handheld_gaming_node(
            Some("node-ally-x-usb4-30g"),
            Some(30.0),
        );
        nodes.push(handheld);

        let report = UsbCDirectInterconnectScanner::scan_host_usbc_links(&nodes);
        assert_eq!(
            report.zero_config_link_local_ready,
            report.live_usb_c_interfaces_detected > 0
        );
        assert!(!report.live_dma_measured);
        assert_eq!(
            report.handheld_30gbps_dma_benchmark.measurement_kind,
            "modelled_arithmetic_estimate"
        );
        assert!(!report.handheld_30gbps_dma_benchmark.live_dma_measured);
        assert!(report.interfaces.len() >= 3);
        assert_eq!(report.reference_tier_catalog.len(), 3);
        assert!(report.switchless_daisy_chain_plan.ring_closed);
        assert!(
            report
                .switchless_daisy_chain_plan
                .bonded_failover_schedule
                .bonded_bandwidth_gbps
                > 30.0
        );
    }

    #[test]
    fn test_empty_and_single_node_topology_plans_do_not_invent_synthetic_peers() {
        let empty = UsbCDirectInterconnectScanner::build_daisy_chain_topology_plan(&[]);
        assert_eq!(empty.total_usb_c_nodes, 0);
        assert!(!empty.ring_closed);
        assert_eq!(empty.bottleneck_goodput_gbps, 0.0);
        assert_eq!(empty.end_to_end_ring_rtt_ms, 0.0);
        assert!(empty.hops.is_empty());
        assert!(
            empty
                .bonded_failover_schedule
                .channel_allocations
                .is_empty()
        );

        let single_node = UsbCDirectInterconnectScanner::build_usb4_handheld_gaming_node(
            Some("node-single-host"),
            Some(30.0),
        );
        let single = UsbCDirectInterconnectScanner::build_daisy_chain_topology_plan(&[single_node]);
        assert_eq!(single.total_usb_c_nodes, 1);
        assert!(!single.ring_closed);
        assert_eq!(single.bottleneck_goodput_gbps, 0.0);
        assert_eq!(single.end_to_end_ring_rtt_ms, 0.0);
        assert!(single.hops.is_empty());
        assert!(
            single
                .bonded_failover_schedule
                .channel_allocations
                .is_empty()
        );
    }
}
