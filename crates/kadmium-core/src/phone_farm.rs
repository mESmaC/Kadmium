//! # `kadmium_core::phone_farm`
//!
//! First-Class Mobile Phone & 20-Slot **Phone Farm Box** (`phonefarmbox.com`) Cluster Support.
//!
//! Enables Kadmium (`Cd-48`) to harvest both:
//! 1. **20-Slot Phone Farm Box Chassis (`DeviceKind::PhoneFarmChassis`)**:
//!    Industrial enclosures housing 20 stripped ARM64 Qualcomm Snapdragon motherboards
//!    (screens, cameras, and physical lithium-ion batteries removed), powered continuously 24/7
//!    via regulated `5V / 40A–60A` (`200W–300W`) power supplies with **BMS battery-eliminator cables**,
//!    cooled by 3–4 chassis fans, and connected via a built-in **20-port USB 3.0 ADB hub** or
//!    **USB-OTG Ethernet switch**.
//! 2. **Standalone Mobile Phones (`DeviceKind::MobilePhone`)**:
//!    Everyday Android (Termux / Native CLI / ADB) or iOS devices governed by a strict
//!    **Pocket-Polite Governor** (`MobilePocketPolitePolicy`) that only executes shards,
//!    speculative draft tokens, or Dream-RSI MCTS rollouts when plugged into a charger with the
//!    screen off and battery/skin temperature below `39.0 °C`.

use crate::{
    ComputeBackend, DeviceKind, HardwareNode, MemoryTierProfile, NodeId, NodeRole, NodeTrustTier,
    PoliteState,
};
use serde::{Deserialize, Serialize};
use std::process::Command;

/// Hardware generation presets for a 20-Slot Phone Farm Box (`phonefarmbox.com`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhoneFarmBoxPreset {
    /// Entry-level 20× Snapdragon 845 (e.g., Galaxy S9 / Note 9 boards):
    /// 160 Kryo 385 cores, 120 GB LPDDR4X, 1.28 TB UFS 2.1, ~24 FP16 TFLOPS / 120 INT4 TOPS, ~95W wall power.
    Snapdragon845S9Box20,
    /// Sweet-spot price/performance 20× Snapdragon 865 (e.g., Galaxy S20 / OnePlus 8T boards):
    /// 160 Kryo 585 cores, 160 GB LPDDR5, 2.56 TB UFS 3.0, Adreno 650 + Hexagon 698 NPU (~46 FP16 TFLOPS / 300 INT4 TOPS), ~115W wall power.
    Snapdragon865S20Box20,
    /// Flagship high-density 20× Snapdragon 8 Gen 2 (e.g., Galaxy S23 / Xiaomi 13 boards):
    /// 160 ARMv9 cores, 240 GB LPDDR5X, 5.12 TB UFS 4.0, Adreno 740 + Hexagon HTP INT4 (~88 FP16 TFLOPS / 680 INT4 TOPS), ~145W wall power.
    Snapdragon8Gen2S23Box20,
}

impl PhoneFarmBoxPreset {
    pub fn from_slug(slug: &str) -> Self {
        match slug.to_ascii_lowercase().as_str() {
            "sd845" | "s9" | "snapdragon845" | "snapdragon845s9box20" => Self::Snapdragon845S9Box20,
            "sd8gen2" | "s23" | "snapdragon8gen2" | "snapdragon8gen2s23box20" => {
                Self::Snapdragon8Gen2S23Box20
            }
            _ => Self::Snapdragon865S20Box20,
        }
    }
}

/// Physical interconnect between the 20-Slot Phone Farm Box and the Kadmium Coordinator / Host PC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhoneFarmTransportMode {
    /// Single USB 3.0 uplink from the box's internal 20-port USB hub to the host PC,
    /// multiplexing TCP `KDMF` frames over `adb forward tcp:7861..7880 tcp:7850` (`~380 MB/s` zero-jitter bulk transfer).
    Usb3AdbMultiplexedTunnel,
    /// Hardwired RJ45 Ethernet into the box's internal USB-OTG LAN switch; each motherboard gets a
    /// dedicated LAN IP and speaks native `KDMF` TCP `WireFrame` on port `7850`.
    OtgHardwiredEthernetLan,
    /// USB power/ADB control combined with Wi-Fi 6 / 6E data plane.
    HybridUsbAndWiFi6,
}

/// How the 20 motherboards inside a Phone Farm Box are exposed to the Kadmium scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhoneFarmTopologyMode {
    /// Exposes the 20-slot chassis as a single high-capacity logical node (`160–240 GB Unified LPDDR5`,
    /// `2.5–5.1 TB UFS Flash`, `160 ARM64 threads`) that internally stripes MoE experts (`256 / 20 = 12–13 experts/slot`),
    /// speculative draft heads, and Dream-RSI MCTS rollouts across all 20 boards in parallel.
    CoherentMoEAndDreamRsiArray,
    /// Exposes all 20 motherboard slots as independent micro-worker nodes (`slot-01..slot-20`) for
    /// disaggregated WASI RL sandboxes, BOINC workunits, and `/v1/embeddings` sharding.
    DisaggregatedMicroWorkerSwarm,
}

/// Pocket-Polite battery, thermal, and screen-state governor for standalone mobile phones.
/// (Automatically bypassed on battery-free `PhoneFarmChassis` boards running on 5V BMS dummy cables).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobilePocketPolitePolicy {
    /// True when the board is powered by a 5V BMS battery-eliminator cable inside a Phone Farm Box
    /// (zero lithium-ion battery degradation or fire risk -> 24/7 100% duty cycle).
    pub battery_eliminator_bms_cable: bool,
    /// For standalone phones: only harvest compute when connected to AC / USB-PD / MagSafe charger.
    pub require_charger_connected: bool,
    /// For standalone phones: yield immediately when the user turns on the screen / unlocks the phone.
    pub require_screen_off: bool,
    /// Minimum battery state-of-charge (`%`) required before accepting shards on a standalone phone.
    pub min_battery_soc_pct: u8,
    /// Maximum SoC / battery skin temperature (`°C`) before throttling or yielding shards.
    pub max_skin_temp_c: f64,
}

impl MobilePocketPolitePolicy {
    pub fn for_phone_farm_chassis() -> Self {
        Self {
            battery_eliminator_bms_cable: true,
            require_charger_connected: false,
            require_screen_off: false,
            min_battery_soc_pct: 0,
            max_skin_temp_c: 68.0, // Actively cooled by 3-4 chassis fans, no battery pouch
        }
    }

    pub fn for_standalone_smartphone() -> Self {
        Self {
            battery_eliminator_bms_cable: false,
            require_charger_connected: true,
            require_screen_off: true,
            min_battery_soc_pct: 80,
            max_skin_temp_c: 38.5,
        }
    }
}

/// Telemetry and shard assignment for a single stripped ARM64 motherboard slot (`1..=20`) inside a Phone Farm Box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneSlotDescriptor {
    pub slot_index: u8,
    pub adb_serial: String,
    pub hostname: String,
    pub soc_model: String,
    pub cpu_cores: u32,
    pub accelerator_signature: String,
    pub lpddr_unified_gb: f64,
    pub lpddr_bw_gbps: f64,
    pub ufs_flash_gb: f64,
    pub ufs_read_bw_gbps: f64,
    pub fp16_tflops: f64,
    pub int4_tops: f64,
    pub host_forwarded_tcp_port: u16,
    pub bms_simulated_voltage_mv: u16,
    pub board_temp_c: f64,
    /// Range of routed MoE experts (`start_expert..=end_expert` out of 256) pinned in this slot's LPDDR/UFS bank.
    pub pinned_moe_expert_range: (u32, u32),
    /// Concurrent Dream-RSI MCTS / WASI RL verifier threads assigned to this board.
    pub dream_rsi_mcts_workers: u32,
}

/// Live USB-ADB discovery report from running `adb devices -l` on the host machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdbBridgeDiscoveryReport {
    pub adb_binary_detected: bool,
    pub connected_physical_devices: Vec<String>,
    pub bootstrap_script_preview: String,
}

/// Complete specification, ROI analysis, and per-slot breakdown of a 20-Slot Phone Farm Box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneFarmChassisSpec {
    pub chassis_id: String,
    pub display_name: String,
    pub preset: PhoneFarmBoxPreset,
    pub slot_count: usize,
    pub transport_mode: PhoneFarmTransportMode,
    pub topology_mode: PhoneFarmTopologyMode,
    pub psu_rating: String,
    pub average_wall_power_watts: f64,
    pub cooling_system: String,
    pub estimated_hardware_cost_usd: f64,
    pub aggregate_cpu_cores: u32,
    pub aggregate_lpddr_unified_gb: f64,
    /// Aggregate internal memory bandwidth when all 20 boards evaluate striped MoE experts or rollouts in parallel.
    pub aggregate_parallel_mem_bw_gbps: f64,
    pub aggregate_ufs_flash_gb: f64,
    pub aggregate_ufs_read_bw_gbps: f64,
    pub aggregate_fp16_tflops: f64,
    pub aggregate_int4_tops: f64,
    pub polite_policy: MobilePocketPolitePolicy,
    pub ideal_kadmium_workloads: Vec<String>,
    pub adb_discovery: AdbBridgeDiscoveryReport,
    pub slots: Vec<PhoneSlotDescriptor>,
}

/// Controller for profiling, discovering (`adb devices`), and attaching Mobile Phones and
/// 20-Slot Phone Farm Boxes into a Kadmium cluster.
pub struct PhoneFarmBoxController;

impl PhoneFarmBoxController {
    /// Probes the local host for an installed `adb` binary and lists any physical Android phones
    /// or Phone Farm Box motherboards currently attached over USB/TCP.
    pub fn scan_local_adb_devices() -> AdbBridgeDiscoveryReport {
        let output = Command::new("adb").arg("devices").arg("-l").output();
        let (adb_binary_detected, connected_physical_devices) = match output {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let devices: Vec<String> = stdout
                    .lines()
                    .skip(1)
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('*'))
                    .map(String::from)
                    .collect();
                (true, devices)
            }
            _ => (false, Vec::new()),
        };

        let bootstrap_script_preview = [
            "# Zero-touch provisioning for a 20-Slot Phone Farm Box over USB-ADB Hub:",
            "for slot in $(seq 1 20); do",
            "  SERIAL=$(adb devices | awk 'NR=='$((slot+1))'{print $1}')",
            "  PORT=$((7860 + slot))",
            "  adb -s $SERIAL push target/aarch64-linux-android/release/kadmium /data/local/tmp/kadmium",
            "  adb -s $SERIAL forward tcp:$PORT tcp:7850",
            "  adb -s $SERIAL reverse tcp:7848 tcp:7848",
            "  adb -s $SERIAL shell '/data/local/tmp/kadmium join --coordinator http://127.0.0.1:7848 --port 7850 &'",
            "done",
        ]
        .join("\n");

        AdbBridgeDiscoveryReport {
            adb_binary_detected,
            connected_physical_devices,
            bootstrap_script_preview,
        }
    }

    /// Builds a detailed 20-Slot Phone Farm Box specification for the requested hardware preset.
    pub fn build_chassis_spec(
        preset: PhoneFarmBoxPreset,
        transport_mode: PhoneFarmTransportMode,
        topology_mode: PhoneFarmTopologyMode,
    ) -> PhoneFarmChassisSpec {
        let slot_count = 20usize;
        let (
            display_name,
            soc_model,
            accel_sig,
            ram_per_slot_gb,
            bw_per_slot_gbps,
            ufs_per_slot_gb,
            ufs_bw_per_slot_gbps,
            fp16_per_slot,
            int4_per_slot,
            watts,
            cost_usd,
        ) = match preset {
            PhoneFarmBoxPreset::Snapdragon845S9Box20 => (
                "PhoneFarmBox 20× Snapdragon 845 (Galaxy S9 Stripped Mainboards)",
                "Qualcomm SDM845 (8× Kryo 385 ARMv8.2)",
                "Adreno 630 Vulkan + Hexagon 685HVX",
                6.0,
                29.9,
                64.0,
                0.85,
                1.2,
                6.0,
                95.0,
                540.0,
            ),
            PhoneFarmBoxPreset::Snapdragon865S20Box20 => (
                "PhoneFarmBox 20× Snapdragon 865 (Galaxy S20 5G Stripped Mainboards)",
                "Qualcomm SM8250 (8× Kryo 585 2.84GHz)",
                "Adreno 650 Vulkan Subgroup + Hexagon 698 NPU (15 TOPS/board)",
                8.0,
                44.0,
                128.0,
                1.50,
                2.3,
                15.0,
                115.0,
                920.0,
            ),
            PhoneFarmBoxPreset::Snapdragon8Gen2S23Box20 => (
                "PhoneFarmBox 20× Snapdragon 8 Gen 2 (Galaxy S23 Stripped Mainboards)",
                "Qualcomm SM8550-AB (8× Kryo ARMv9 3.2GHz)",
                "Adreno 740 CooperativeMatrix + Hexagon HTP INT4 (34 TOPS/board)",
                12.0,
                67.2,
                256.0,
                3.80,
                4.4,
                34.0,
                145.0,
                1780.0,
            ),
        };

        let total_experts = 256u32;
        let experts_per_slot = total_experts / (slot_count as u32);
        let mut slots = Vec::with_capacity(slot_count);

        for idx in 0..slot_count {
            let slot_num = (idx + 1) as u8;
            let start_exp = (idx as u32) * experts_per_slot;
            let end_exp = if idx + 1 == slot_count {
                total_experts - 1
            } else {
                start_exp + experts_per_slot - 1
            };
            slots.push(PhoneSlotDescriptor {
                slot_index: slot_num,
                adb_serial: format!("PFB20-SLOT{:02}-9889A{:02X}", slot_num, slot_num * 5),
                hostname: format!("pfb-slot-{:02}", slot_num),
                soc_model: soc_model.to_string(),
                cpu_cores: 8,
                accelerator_signature: accel_sig.to_string(),
                lpddr_unified_gb: ram_per_slot_gb,
                lpddr_bw_gbps: bw_per_slot_gbps,
                ufs_flash_gb: ufs_per_slot_gb,
                ufs_read_bw_gbps: ufs_bw_per_slot_gbps,
                fp16_tflops: fp16_per_slot,
                int4_tops: int4_per_slot,
                host_forwarded_tcp_port: 7860 + (slot_num as u16),
                bms_simulated_voltage_mv: 4200,
                board_temp_c: 36.0 + ((idx % 5) as f64) * 0.8,
                pinned_moe_expert_range: (start_exp, end_exp),
                dream_rsi_mcts_workers: 6,
            });
        }

        let sc = slot_count as f64;
        PhoneFarmChassisSpec {
            chassis_id: "phone-farm-box-20x".to_string(),
            display_name: display_name.to_string(),
            preset,
            slot_count,
            transport_mode,
            topology_mode,
            psu_rating: "Integrated 250W Regulated 5V/50A DC Bus (Battery-Eliminator BMS Harnesses)"
                .to_string(),
            average_wall_power_watts: watts,
            cooling_system: "3× High-Static-Pressure 120mm Chassis Fans (36–39 °C sustained, zero thermal throttle)".to_string(),
            estimated_hardware_cost_usd: cost_usd,
            aggregate_cpu_cores: (slot_count as u32) * 8,
            aggregate_lpddr_unified_gb: ram_per_slot_gb * sc,
            aggregate_parallel_mem_bw_gbps: ((bw_per_slot_gbps * sc) * 10.0).round() / 10.0,
            aggregate_ufs_flash_gb: ufs_per_slot_gb * sc,
            aggregate_ufs_read_bw_gbps: ((ufs_bw_per_slot_gbps * sc) * 10.0).round() / 10.0,
            aggregate_fp16_tflops: ((fp16_per_slot * sc) * 10.0).round() / 10.0,
            aggregate_int4_tops: ((int4_per_slot * sc) * 10.0).round() / 10.0,
            polite_policy: MobilePocketPolitePolicy::for_phone_farm_chassis(),
            ideal_kadmium_workloads: vec![
                "Colibri MoE Expert Bank Striping: Pins all 256 routed MoE experts across 160–240 GB LPDDR5 + 2.5 TB UFS 3.0 flash (~30 GB/s aggregate flash read)".to_string(),
                "Dream-RSI Offline World Evolution: 160 ARM64 cores run 120 concurrent MCTS rollout simulations & WASI RL sandboxes at 0 GPU cost (~115W wall power)".to_string(),
                "Speculative Decoding Draft Swarm: 20 parallel Hexagon NPU / Adreno Vulkan draft heads proposing 5-token speculative trees".to_string(),
                "Strata Vision Patch & /v1/embeddings Offload: 20-way data-parallel ViT image tile encoding and batch document embedding".to_string(),
            ],
            adb_discovery: Self::scan_local_adb_devices(),
            slots,
        }
    }

    /// Converts a `PhoneFarmChassisSpec` into a coherent aggregated `HardwareNode` (`DeviceKind::PhoneFarmChassis`)
    /// so the global `ShardPlanner` and `RlTopologyPlanner` can schedule MoE expert banks, middle layers,
    /// speculative draft heads, and 160-core Dream-RSI MCTS rollouts onto the box.
    pub fn to_aggregated_hardware_node(spec: &PhoneFarmChassisSpec) -> HardwareNode {
        let uplink_gbps = match spec.transport_mode {
            PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel => 3.8, // USB 3.0 5Gbps bulk ADB bridge
            PhoneFarmTransportMode::OtgHardwiredEthernetLan => 2.5,  // Switched OTG LAN
            PhoneFarmTransportMode::HybridUsbAndWiFi6 => 4.8,
        };
        let latency_ms = match spec.transport_mode {
            PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel => 0.45,
            PhoneFarmTransportMode::OtgHardwiredEthernetLan => 0.65,
            PhoneFarmTransportMode::HybridUsbAndWiFi6 => 1.20,
        };

        HardwareNode {
            id: NodeId(spec.chassis_id.clone()),
            hostname: "phonefarmbox-20x-snapdragon".to_string(),
            device_kind: DeviceKind::PhoneFarmChassis,
            backend: ComputeBackend::CpuArmNeonNpu,
            roles: vec![
                NodeRole::ComputeWorker,
                NodeRole::DraftAndUtilityWorker,
                NodeRole::StorageVault,
            ],
            trust_tier: NodeTrustTier::TrustedCore,
            gpu_name: "20x Qualcomm Adreno Mobile GPU Array".into(),
            cpu_name: format!(
                "20x Snapdragon Kryo ({} ARM64 Cores)",
                spec.aggregate_cpu_cores
            ),
            npu_name: Some("20x Qualcomm Hexagon NPU Array".into()),
            npu_tops: (spec.aggregate_int4_tops * 0.65 * 10.0).round() / 10.0,
            fp16_tflops: spec.aggregate_fp16_tflops,
            int4_tops: spec.aggregate_int4_tops,
            cpu_threads: spec.aggregate_cpu_cores,
            cpu_tflops: ((spec.aggregate_cpu_cores as f64) * 0.14 * 10.0).round() / 10.0,
            memory: MemoryTierProfile {
                vram_gb: 0.0,
                vram_bw_gbps: 0.0,
                // Effective striped Unified LPDDR memory across the 20 motherboards
                unified_mem_gb: spec.aggregate_lpddr_unified_gb,
                unified_mem_bw_gbps: (spec.aggregate_parallel_mem_bw_gbps * 0.42).round(),
                system_ram_gb: 0.0,
                system_ram_bw_gbps: 0.0,
                interconnect_bw_gbps: uplink_gbps * 4.0,
                nvme_cache_gb: spec.aggregate_ufs_flash_gb,
                nvme_read_bw_gbps: spec.aggregate_ufs_read_bw_gbps,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: uplink_gbps,
            network_latency_ms: latency_ms,
        }
    }

    /// Converts a `PhoneFarmChassisSpec` into 20 individual `HardwareNode`s (`DeviceKind::MobilePhone`),
    /// useful when running disaggregated micro-worker swarms.
    pub fn to_individual_slot_nodes(spec: &PhoneFarmChassisSpec) -> Vec<HardwareNode> {
        spec.slots
            .iter()
            .map(|slot| HardwareNode {
                id: NodeId(format!("node-{}", slot.hostname)),
                hostname: slot.hostname.clone(),
                device_kind: DeviceKind::MobilePhone,
                backend: ComputeBackend::CpuArmNeonNpu,
                roles: vec![NodeRole::ComputeWorker, NodeRole::DraftAndUtilityWorker],
                trust_tier: NodeTrustTier::TrustedCore,
                gpu_name: format!("{} Adreno GPU", slot.soc_model),
                cpu_name: format!("{} ({}C Kryo)", slot.soc_model, slot.cpu_cores),
                npu_name: Some(format!("{} Hexagon NPU", slot.soc_model)),
                npu_tops: (slot.int4_tops * 0.65 * 10.0).round() / 10.0,
                fp16_tflops: slot.fp16_tflops,
                int4_tops: slot.int4_tops,
                cpu_threads: slot.cpu_cores,
                cpu_tflops: 1.1,
                memory: MemoryTierProfile {
                    vram_gb: 0.0,
                    vram_bw_gbps: 0.0,
                    unified_mem_gb: slot.lpddr_unified_gb,
                    unified_mem_bw_gbps: slot.lpddr_bw_gbps,
                    system_ram_gb: 0.0,
                    system_ram_bw_gbps: 0.0,
                    interconnect_bw_gbps: slot.lpddr_bw_gbps,
                    nvme_cache_gb: slot.ufs_flash_gb,
                    nvme_read_bw_gbps: slot.ufs_read_bw_gbps,
                },
                polite_state: PoliteState::IdleDedicated,
                network_uplink_gbps: 1.0,
                network_latency_ms: 0.6,
            })
            .collect()
    }

    /// Builds a representative standalone smartphone node (e.g., Galaxy S24 Ultra / Pixel 9 Pro on nightly charger).
    pub fn sample_standalone_mobile_phone() -> HardwareNode {
        HardwareNode {
            id: NodeId("node-mobile-s24-ultra".into()),
            hostname: "galaxy-s24-ultra-pocket".into(),
            device_kind: DeviceKind::MobilePhone,
            backend: ComputeBackend::CpuArmNeonNpu,
            roles: vec![NodeRole::DraftAndUtilityWorker, NodeRole::ComputeWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            gpu_name: "Qualcomm Adreno 750".into(),
            cpu_name: "Snapdragon 8 Gen 3 (8C Kryo)".into(),
            npu_name: Some("Qualcomm Hexagon NPU".into()),
            npu_tops: 34.0,
            fp16_tflops: 5.8,
            int4_tops: 45.0,
            cpu_threads: 8,
            cpu_tflops: 1.4,
            memory: MemoryTierProfile {
                vram_gb: 0.0,
                vram_bw_gbps: 0.0,
                unified_mem_gb: 12.0,
                unified_mem_bw_gbps: 76.8,
                system_ram_gb: 0.0,
                system_ram_bw_gbps: 0.0,
                interconnect_bw_gbps: 76.8,
                nvme_cache_gb: 256.0,
                nvme_read_bw_gbps: 4.0,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 1.2,
            network_latency_ms: 1.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phone_farm_box_20_slot_spec_and_node_conversion() {
        let spec = PhoneFarmBoxController::build_chassis_spec(
            PhoneFarmBoxPreset::Snapdragon865S20Box20,
            PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
            PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray,
        );
        assert_eq!(spec.slot_count, 20);
        assert_eq!(spec.aggregate_cpu_cores, 160);
        assert!((spec.aggregate_lpddr_unified_gb - 160.0).abs() < 1e-6);
        assert!((spec.aggregate_ufs_flash_gb - 2560.0).abs() < 1e-6);
        assert!(spec.polite_policy.battery_eliminator_bms_cable);

        let agg_node = PhoneFarmBoxController::to_aggregated_hardware_node(&spec);
        assert_eq!(agg_node.device_kind, DeviceKind::PhoneFarmChassis);
        assert_eq!(agg_node.cpu_threads, 160);
        assert!(agg_node.compute_q_star_ratio() > 0.80);

        let slot_nodes = PhoneFarmBoxController::to_individual_slot_nodes(&spec);
        assert_eq!(slot_nodes.len(), 20);
        assert_eq!(slot_nodes[0].device_kind, DeviceKind::MobilePhone);
    }
}
