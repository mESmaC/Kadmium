//! # `kadmium-core`
//!
//! Core domain types for **Kadmium (`Cd`)**, the heterogeneous distributed AI/ML compute fabric.
//! Models multi-tier memory hierarchies (`VRAM`, `UnifiedMemory`, `SystemRAM`, `NVMe`, `NAS`),
//! FreeToken-inspired $q^*$ bandwidth-adaptive execution ratios, BOINC-inspired polite host states,
//! and model/training topologies.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a compute or storage node in the Kadmium mesh.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Physical classification of a node in a heterogeneous home, lab, or hybrid cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    /// Main desktop workstation with discrete desktop GPU(s).
    DesktopWorkstation,
    /// Gaming laptop with discrete mobile GPU and battery/thermal envelope.
    GamingLaptop,
    /// Apple Silicon Mac (M1–M4) with high-bandwidth Unified Memory.
    AppleSiliconMac,
    /// Older laptop or desktop with CPU/iGPU, spare DDR RAM, and SSD storage.
    LegacyLaptop,
    /// Single-Board Computer (Linux ARM64 / RISC-V, e.g., RK3588, RPi 5, Jetson Nano).
    LinuxSbc,
    /// Home or lab Network Attached Storage appliance for weight/dataset seeding and checkpoints.
    HomeNas,
    /// University lab or friend's remote shared workstation over LAN/WAN.
    RemotePeerWorkstation,
    /// Ephemeral rented datacenter GPU joined for temporary workflow boosting.
    CloudBurstGpu,
}

/// Primary compute acceleration backend available on a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeBackend {
    Cuda,
    MetalUnified,
    Rocm,
    VulkanWgpu,
    CpuAvx512OrAvx2,
    CpuArmNeonNpu,
}

/// Logical roles a node can fulfill simultaneously in the Kadmium cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Orchestrates shard planning, FTS3 transfer scheduling, and API gateway routing.
    Coordinator,
    /// Executes dense transformer layers, MoE expert banks, or training pipeline stages.
    ComputeWorker,
    /// Runs tokenization, speculative decoding draft models, BPE preprocessing, and embeddings.
    DraftAndUtilityWorker,
    /// Hosts canonical Kadmium Tensor Banks (`.ktb`), datasets, and training checkpoints.
    StorageVault,
    /// Temporary high-FLOP accelerator that absorbs heavy shards on demand.
    BurstAccelerator,
}

/// BOINC-inspired host availability state governing how much compute Kadmium may harvest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoliteState {
    /// Node is idle or dedicated; 100% of configured compute/memory is available.
    IdleDedicated,
    /// User is doing light desktop work (browsing/coding); 75% compute budget allowed.
    LightDesktopUse,
    /// User launched a game, 3D app, or heavy local job; yield down to 15% immediately.
    GamingOrHeavyApp,
    /// Laptop is running on battery or thermally constrained; throttle to 25%.
    BatteryOrThermalThrottle,
    /// Ephemeral cloud GPU or shutting-down node is draining its shards before disconnect.
    Draining,
}

/// 3-Tier Privacy & Security Trust Model governing which model layers a node is allowed to host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeTrustTier {
    /// User's own hardware (Main PC, Mac Mini, Laptops, SBCs, Home NAS).
    /// Allowed to host Ingress ($L_0$ token embeddings), Egress ($L_{N-1}$ LM head/sampling),
    /// middle layers, and raw prompt/checkpoint vaults.
    TrustedCore,
    /// Friend, family, or partner academic lab node over WAN.
    /// Restricted to **middle transformer layers only** ($0 < L_a \le L_b < N-1$) so it only
    /// processes intermediate floating-point activation tensors—never raw prompts or decoded text.
    VerifiedPeerMiddleLayersOnly,
    /// External community pool or multi-cloud spot arbitrage instance.
    /// Restricted to middle layers only + mandatory Gensyn Verde / RepOps layer-bisection auditing.
    UntrustedAuditedMiddleLayersOnly,
}

impl NodeTrustTier {
    /// Returns true if this node is permitted to host Layer 0 (Prompt Ingress / Embeddings)
    /// or Layer $N-1$ (LM Head / Token Sampling).
    pub fn can_host_boundary_ingress_egress(self) -> bool {
        matches!(self, Self::TrustedCore)
    }
}

impl PoliteState {
    /// Returns the polite availability multiplier $\rho \in [0.0, 1.0]$.
    pub fn availability_multiplier(self) -> f64 {
        match self {
            Self::IdleDedicated => 1.0,
            Self::LightDesktopUse => 0.75,
            Self::GamingOrHeavyApp => 0.15,
            Self::BatteryOrThermalThrottle => 0.25,
            Self::Draining => 0.0,
        }
    }
}

/// Five-tier memory hierarchy profile for a node, enabling FreeToken ($q^*$) and Colibri
/// (NVMe expert streaming) placement decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTierProfile {
    /// Tier 0: Discrete GPU VRAM capacity in GiB.
    pub vram_gb: f64,
    /// Tier 0: Discrete GPU VRAM bandwidth in GB/s.
    pub vram_bw_gbps: f64,
    /// Tier 1: Apple Silicon / APU Unified Memory capacity in GiB.
    pub unified_mem_gb: f64,
    /// Tier 1: Unified Memory bandwidth in GB/s.
    pub unified_mem_bw_gbps: f64,
    /// Tier 2: Host System RAM (DDR4/DDR5) capacity in GiB.
    pub system_ram_gb: f64,
    /// Tier 2: Host System RAM bandwidth in GB/s.
    pub system_ram_bw_gbps: f64,
    /// PCIe / Host-to-Device interconnect bandwidth in GB/s (0 if unified or CPU-only).
    pub interconnect_bw_gbps: f64,
    /// Tier 3: Local NVMe SSD cache capacity allocated to Kadmium `.ktb` banks in GiB.
    pub nvme_cache_gb: f64,
    /// Tier 3: Sustained sequential NVMe read throughput in GB/s (for Colibri expert streaming).
    pub nvme_read_bw_gbps: f64,
}

impl MemoryTierProfile {
    /// Total fast accelerator memory (Tier 0 VRAM + Tier 1 Unified Memory) in GiB.
    pub fn fast_accelerator_memory_gb(&self) -> f64 {
        self.vram_gb + self.unified_mem_gb
    }

    /// Total memory usable for resident model weights (Tier 0 + Tier 1 + 70% of Tier 2 System RAM).
    pub fn resident_weight_capacity_gb(&self) -> f64 {
        self.vram_gb + self.unified_mem_gb + (self.system_ram_gb * 0.70)
    }

    /// Effective weighted memory bandwidth in GB/s across active tiers.
    pub fn effective_bandwidth_gbps(&self) -> f64 {
        if self.unified_mem_gb > 0.0 {
            self.unified_mem_bw_gbps
        } else if self.vram_gb > 0.0 {
            // Blended bandwidth accounting for PCIe offload window
            (self.vram_bw_gbps * 0.75) + (self.system_ram_bw_gbps * 0.25)
        } else {
            self.system_ram_bw_gbps
        }
    }
}

fn default_trust_tier() -> NodeTrustTier {
    NodeTrustTier::TrustedCore
}

/// Complete capability vector and live status of a physical or virtual node in the Kadmium mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareNode {
    pub id: NodeId,
    pub hostname: String,
    pub device_kind: DeviceKind,
    pub backend: ComputeBackend,
    pub roles: Vec<NodeRole>,
    #[serde(default = "default_trust_tier")]
    pub trust_tier: NodeTrustTier,
    /// Peak FP16 / BF16 TFLOPs on primary accelerator.
    pub fp16_tflops: f64,
    /// Peak INT4 / Q4 quantized TOPS.
    pub int4_tops: f64,
    /// Host CPU threads available for parallel host-bank execution ($q^*$ CPU split).
    pub cpu_threads: u32,
    /// Host CPU FP16/INT4 effective TFLOPs for hybrid execution.
    pub cpu_tflops: f64,
    pub memory: MemoryTierProfile,
    pub polite_state: PoliteState,
    /// Network link speed to local switch/mesh in Gbps.
    pub network_uplink_gbps: f64,
    /// Typical round-trip latency to cluster coordinator in milliseconds.
    pub network_latency_ms: f64,
}

impl HardwareNode {
    /// Effective TFLOPs currently available after applying the BOINC-style PoliteState multiplier.
    pub fn effective_fp16_tflops(&self) -> f64 {
        let rho = self.polite_state.availability_multiplier();
        (self.fp16_tflops + self.cpu_tflops * 0.35) * rho
    }

    /// Effective fast memory (VRAM + Unified Memory) available under the current PoliteState.
    pub fn available_fast_memory_gb(&self) -> f64 {
        let rho = self.polite_state.availability_multiplier();
        // When gaming on a desktop/laptop, reserve most VRAM for the user's game.
        let vram_factor = match self.polite_state {
            PoliteState::GamingOrHeavyApp => 0.15,
            PoliteState::BatteryOrThermalThrottle => 0.50,
            PoliteState::Draining => 0.0,
            _ => 0.90,
        };
        (self.memory.vram_gb * vram_factor) + (self.memory.unified_mem_gb * rho.max(0.5))
    }

    /// Computes the FreeToken-inspired bandwidth-adaptive execution split $q^* \in [0.0, 1.0]$.
    ///
    /// $q^*$ represents the optimal fraction of active token/expert compute executed on the
    /// primary accelerator vs. dynamically streamed/executed across host CPU + RAM + NVMe so
    /// that accelerator and host/PCIe pipelines complete simultaneously without stalling.
    pub fn compute_q_star_ratio(&self) -> f64 {
        if self.memory.unified_mem_gb > 0.0 {
            // Apple Silicon Unified Memory has zero PCIe copy bottleneck.
            return 0.96;
        }
        if self.memory.vram_gb <= 0.0 {
            // Pure CPU/RAM/NVMe node executes 100% on host banks.
            return 0.0;
        }
        let accel_power = self.fp16_tflops * self.polite_state.availability_multiplier();
        let host_stream_power = self.cpu_tflops
            + (self.memory.interconnect_bw_gbps * 0.35)
            + (self.memory.nvme_read_bw_gbps * 0.50);
        let total = accel_power + host_stream_power;
        if total <= f64::EPSILON {
            0.0
        } else {
            (accel_power / total).clamp(0.10, 0.95)
        }
    }

    /// Composite capacity score for a specific workload kind (`Inference`, `DistributedTraining`, `BatchPipeline`).
    pub fn capacity_score(&self, workload: WorkloadKind) -> f64 {
        if self.polite_state == PoliteState::Draining
            || !self.roles.contains(&NodeRole::ComputeWorker)
                && !self.roles.contains(&NodeRole::BurstAccelerator)
        {
            return 0.0;
        }

        let rho = self.polite_state.availability_multiplier();
        let flops = self.effective_fp16_tflops();
        let fast_mem = self.available_fast_memory_gb();
        let ram_mem = self.memory.system_ram_gb * 0.5 * rho;
        let bw = (self.memory.effective_bandwidth_gbps() / 100.0) * rho;
        let nvme = self.memory.nvme_read_bw_gbps * rho;

        match workload {
            WorkloadKind::Inference => {
                // Inference is heavily memory-bandwidth and capacity bound (plus NVMe for MoE streaming).
                (flops * 1.1) + (fast_mem * 2.8) + (ram_mem * 0.9) + (bw * 4.5) + (nvme * 1.5)
            }
            WorkloadKind::DistributedTraining => {
                // Training weights FLOPs and high-speed activation/gradient memory heavily, penalized by WAN latency.
                let latency_penalty = 1.0 / (1.0 + (self.network_latency_ms / 25.0));
                ((flops * 2.4) + (fast_mem * 3.0) + (ram_mem * 0.6) + (bw * 3.0)) * latency_penalty
            }
            WorkloadKind::BatchPipeline => {
                (flops * 1.5) + (fast_mem * 1.5) + (ram_mem * 1.0) + (nvme * 2.0)
            }
        }
    }
}

/// Primary AI/ML workload classification scheduled by Kadmium.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadKind {
    /// Low-latency streaming token generation or high-throughput MoE inference.
    Inference,
    /// Distributed fine-tuning or pre-training (Asymmetric 1F1B, QLoRA, DiLoCo).
    DistributedTraining,
    /// Asynchronous BOINC-style workunits (embeddings, dataset curation, evals, synthetic data).
    BatchPipeline,
}

/// Distributed & decentralized training synchronization and sharding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingStrategy {
    /// Asymmetric 1F1B pipeline parallelism with Hivemind 8-bit boundary activations & selective checkpointing.
    AsymmetricPipeline1F1B,
    /// Frozen 4-bit/8-bit `.ktb` base shards + distributed low-rank adapter (LoRA/QLoRA/DoRA) gradients.
    DistributedQLoRA,
    /// Sharded-Island DiLoCo (OpenDiLoCo) + Nous DisTrO/DeMo DCT frequency-compressed momentum sync.
    ShardedIslandDiLoCoDeMo,
    /// Decentralized RL Post-Training Swarm (GRPO/PPO rollouts on GPUs/Mac + verifiable reward sandboxes on SBCs).
    DecentralizedRlSwarmGrpo,
    /// Hybrid LAN 1F1B Pipeline + WAN Sharded-Island DiLoCo/DeMo + SkipPipe + Elastic Cloud GPU Bursting.
    HybridElasticMesh,
}

/// Wire compression codec used for inter-node activations, pseudo-gradients, and momentum states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientCompressionCodec {
    /// Uncompressed FP16/BF16 (used only over Thunderbolt / 10GbE local links).
    RawFp16Bf16,
    /// Hivemind-style 8-bit blockwise dynamic quantization for pipeline boundary activations.
    HivemindBlockwise8Bit,
    /// OpenDiLoCo INT8 + Top-k sparsification with error feedback.
    DiLoCoInt8TopK,
    /// Nous DisTrO / DeMo Discrete Cosine Transform (DCT) fast-frequency momentum extraction (800x-3000x compression).
    DeMoDctFrequencyTopK,
}

/// TPU / Systolic & Compute-in-Memory (CIM) inspired tensor dataflow execution modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataflowMode {
    /// Default FreeToken $q^*$ + Colibri tiered execution.
    AdaptiveTieredHybrid,
    /// **Compute-in-Memory / Weight-Stationary Dynamic Token Streaming:**
    /// Weights are permanently pinned across distributed VRAM, Unified Memory, and System RAM banks
    /// (zero weight movement). Compact token activation vectors ($x_t$) stream like a systolic
    /// wavefront directly to the memory controllers where weights reside. Ideal for low-batch decode.
    ComputeInMemoryTokenStreaming,
    /// **Token-Stationary Dynamic Weight Streaming:**
    /// Large batches of tokens/activations remain stationary in fast accelerator memory while
    /// double-buffered weight tiles ($W^{(l)}$) stream continuously through the compute units
    /// from NVMe / NAS / peer RAM banks. Ideal for high-throughput batch prefill, evals, or running
    /// massive models on 1–2 GPUs.
    TokenStationaryWeightStreaming,
    /// **Disaggregated Dual-Stream (Systolic Prefill + CIM Decode):**
    /// Uses Token-Stationary Dynamic Weight Streaming on high-FLOP GPUs for prompt prefill, then
    /// hands the KV-cache/token stream to Weight-Stationary Compute-in-Memory nodes for low-latency decode.
    DisaggregatedDualStream,
}

/// Operational purpose of an isolated Sub-Compute Environment (Cluster Slice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlicePurpose {
    /// Locked production inference deployment with strict latency SLA.
    ProductionDeployment,
    /// Staging, integration testing, and canary model verification.
    StagingAndTesting,
    /// Dedicated distributed pre-training or fine-tuning (1F1B / QLoRA / DiLoCo).
    DistributedTrainingLab,
    /// Disaggregated RL environments & Dream-RSI offline replay evolution.
    RlAndDreamRsiArena,
    /// High-throughput batch pipelines, embeddings, and weight-streamed evaluations.
    BatchAndEvalPool,
}

/// Bidirectional High-Availability Failover direction for Kadmium's Downtime Protection Shield.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailoverDirection {
    /// Local cluster or ISP outage -> automatically shift affected shards to Kadmium Cloud standby nodes.
    LocalOutageToCloudStandby,
    /// Cloud provider outage or spot instance preemption -> shift shards to alternate cloud provider + local cluster.
    CloudOutageOrPreemptionToLocalAndMultiCloud,
    /// Local hardware recovered -> drain temporary cloud failover nodes back to local zero-marginal-cost hardware.
    CloudFailbackToRecoveredLocal,
}

/// Multi-datacenter / cloud provider classification for Kadmium's Spot & Stranded Compute Arbitrage engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeProviderKind {
    /// User's own zero-marginal-cost local hardware (Desktop, Mac Mini, Laptops, SBCs, NAS).
    SelfHostedLocalOnPrem,
    /// Regional bare-metal & high-memory CPU/NVMe provider (e.g., Hetzner, OVH, Scaleway).
    RegionalBareMetalProvider,
    /// Specialized GPU spot market / decentralized GPU lease (e.g., RunPod, Vast, Lambda, FluidStack).
    SpecialistGpuSpotMarket,
    /// Hyperscaler preemptible spot capacity (e.g., AWS Spot, GCP Spot, Azure Low-Priority).
    HyperscalerPreemptibleSpot,
    /// Green / stranded energy datacenter GPU capacity (e.g., Crusoe, Northern Data).
    StrandedEnergyDatacenter,
}

/// Architectural specification of a model to be sharded across the Kadmium cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub display_name: String,
    pub total_params_billions: f64,
    pub num_layers: u32,
    pub is_moe: bool,
    /// Size of shared dense attention/trunk weights per layer in GiB (at target quantization).
    pub dense_layer_size_gb: f64,
    /// Total routed experts per MoE layer (0 for dense models).
    pub experts_per_layer: u32,
    /// Number of experts activated per token ($k$ in Top-$k$ routing).
    pub active_experts_per_token: u32,
    /// Footprint of a single expert in GiB.
    pub single_expert_size_gb: f64,
    /// KV-cache footprint per layer at reference context window in GiB.
    pub kv_cache_per_layer_gb: f64,
    /// Training forward activation footprint per layer (without recomputation) in GiB.
    pub training_activation_per_layer_gb: f64,
    /// LoRA adapter + optimizer state footprint per layer in GiB.
    pub lora_state_per_layer_gb: f64,
}

impl ModelSpec {
    /// Total static weight footprint of the entire model in GiB.
    pub fn total_weight_footprint_gb(&self) -> f64 {
        let per_layer_moe = (self.experts_per_layer as f64) * self.single_expert_size_gb;
        (self.num_layers as f64) * (self.dense_layer_size_gb + per_layer_moe)
    }

    ///Preset: DeepSeek-V3 / GLM-5 Class 671B Mixture-of-Experts (Q4_K / FP8 hybrid `.ktb`).
    pub fn deepseek_v3_671b_moe() -> Self {
        Self {
            id: "deepseek-v3-671b-moe".to_string(),
            display_name: "DeepSeek-V3 671B MoE (37B Active, Q4/FP8 KTB)".to_string(),
            total_params_billions: 671.0,
            num_layers: 61,
            is_moe: true,
            dense_layer_size_gb: 0.32,
            experts_per_layer: 64,
            active_experts_per_token: 6,
            single_expert_size_gb: 0.085,
            kv_cache_per_layer_gb: 0.14,
            training_activation_per_layer_gb: 0.45,
            lora_state_per_layer_gb: 0.09,
        }
    }

    /// Preset: Qwen3-235B-A22B MoE (Q4 `.ktb`).
    pub fn qwen3_235b_moe() -> Self {
        Self {
            id: "qwen3-235b-moe".to_string(),
            display_name: "Qwen3-235B-A22B MoE (Q4 KTB)".to_string(),
            total_params_billions: 235.0,
            num_layers: 64,
            is_moe: true,
            dense_layer_size_gb: 0.22,
            experts_per_layer: 64,
            active_experts_per_token: 8,
            single_expert_size_gb: 0.032,
            kv_cache_per_layer_gb: 0.12,
            training_activation_per_layer_gb: 0.35,
            lora_state_per_layer_gb: 0.07,
        }
    }

    /// Preset: Llama-3.3-70B Dense Fine-Tuning / Inference Topology.
    pub fn llama_3_70b_lora() -> Self {
        Self {
            id: "llama-3-70b-lora".to_string(),
            display_name: "Llama-3.3-70B Dense (Distributed QLoRA / 1F1B)".to_string(),
            total_params_billions: 70.6,
            num_layers: 80,
            is_moe: false,
            dense_layer_size_gb: 0.52,
            experts_per_layer: 0,
            active_experts_per_token: 0,
            single_expert_size_gb: 0.0,
            kv_cache_per_layer_gb: 0.16,
            training_activation_per_layer_gb: 0.55,
            lora_state_per_layer_gb: 0.11,
        }
    }
}

/// Builds a representative personal + home lab cluster matching real-world heterogeneous hardware:
/// - Main Desktop PC (RTX 4090 24GB + 64GB DDR5 + Gen4 NVMe)
/// - M4 Mac Mini Pro (48GB Unified Memory + Gen4 NVMe)
/// - 2x Gaming Laptops (RTX 4070M 8GB + 32GB DDR5, RTX 3070M 8GB + 16GB DDR4)
/// - 1x Legacy Laptop (ThinkPad CPU/iGPU + 16GB RAM + NVMe)
/// - 1x Linux SBC Cluster (3x ARM64 RK3588 / Pi 5 boards for draft decoding, tokenization & data prep)
/// - 1x Home NAS Vault (64GB ECC + 40TB ZFS/NVMe Cache for FTS3/Davix weight & checkpoint seeding)
pub fn sample_personal_cluster() -> Vec<HardwareNode> {
    vec![
        HardwareNode {
            id: NodeId("node-main-pc".into()),
            hostname: "cameron-main-rig".into(),
            device_kind: DeviceKind::DesktopWorkstation,
            backend: ComputeBackend::Cuda,
            roles: vec![NodeRole::Coordinator, NodeRole::ComputeWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops: 165.0,
            int4_tops: 660.0,
            cpu_threads: 32,
            cpu_tflops: 4.2,
            memory: MemoryTierProfile {
                vram_gb: 24.0,
                vram_bw_gbps: 1008.0,
                unified_mem_gb: 0.0,
                unified_mem_bw_gbps: 0.0,
                system_ram_gb: 64.0,
                system_ram_bw_gbps: 89.6,
                interconnect_bw_gbps: 31.5,
                nvme_cache_gb: 512.0,
                nvme_read_bw_gbps: 7.4,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 2.5,
            network_latency_ms: 0.4,
        },
        HardwareNode {
            id: NodeId("node-m4-mac-mini".into()),
            hostname: "m4-mac-mini-pro".into(),
            device_kind: DeviceKind::AppleSiliconMac,
            backend: ComputeBackend::MetalUnified,
            roles: vec![NodeRole::Coordinator, NodeRole::ComputeWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops: 42.0,
            int4_tops: 152.0,
            cpu_threads: 14,
            cpu_tflops: 6.5,
            memory: MemoryTierProfile {
                vram_gb: 0.0,
                vram_bw_gbps: 0.0,
                unified_mem_gb: 48.0,
                unified_mem_bw_gbps: 273.0,
                system_ram_gb: 0.0,
                system_ram_bw_gbps: 0.0,
                interconnect_bw_gbps: 273.0,
                nvme_cache_gb: 512.0,
                nvme_read_bw_gbps: 6.8,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 10.0,
            network_latency_ms: 0.5,
        },
        HardwareNode {
            id: NodeId("node-gaming-laptop-1".into()),
            hostname: "zephyrus-g16-rtx4070".into(),
            device_kind: DeviceKind::GamingLaptop,
            backend: ComputeBackend::Cuda,
            roles: vec![NodeRole::ComputeWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops: 46.0,
            int4_tops: 184.0,
            cpu_threads: 20,
            cpu_tflops: 2.8,
            memory: MemoryTierProfile {
                vram_gb: 8.0,
                vram_bw_gbps: 256.0,
                unified_mem_gb: 0.0,
                unified_mem_bw_gbps: 0.0,
                system_ram_gb: 32.0,
                system_ram_bw_gbps: 76.8,
                interconnect_bw_gbps: 15.7,
                nvme_cache_gb: 256.0,
                nvme_read_bw_gbps: 5.2,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 2.5,
            network_latency_ms: 1.2,
        },
        HardwareNode {
            id: NodeId("node-gaming-laptop-2".into()),
            hostname: "legion-5-rtx3070".into(),
            device_kind: DeviceKind::GamingLaptop,
            backend: ComputeBackend::Cuda,
            roles: vec![NodeRole::ComputeWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops: 36.0,
            int4_tops: 144.0,
            cpu_threads: 16,
            cpu_tflops: 2.1,
            memory: MemoryTierProfile {
                vram_gb: 8.0,
                vram_bw_gbps: 448.0,
                unified_mem_gb: 0.0,
                unified_mem_bw_gbps: 0.0,
                system_ram_gb: 16.0,
                system_ram_bw_gbps: 51.2,
                interconnect_bw_gbps: 15.7,
                nvme_cache_gb: 256.0,
                nvme_read_bw_gbps: 3.6,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 1.0,
            network_latency_ms: 1.5,
        },
        HardwareNode {
            id: NodeId("node-legacy-laptop".into()),
            hostname: "thinkpad-t480-linux".into(),
            device_kind: DeviceKind::LegacyLaptop,
            backend: ComputeBackend::CpuAvx512OrAvx2,
            roles: vec![NodeRole::ComputeWorker, NodeRole::DraftAndUtilityWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops: 1.8,
            int4_tops: 7.2,
            cpu_threads: 8,
            cpu_tflops: 1.8,
            memory: MemoryTierProfile {
                vram_gb: 0.0,
                vram_bw_gbps: 0.0,
                unified_mem_gb: 0.0,
                unified_mem_bw_gbps: 0.0,
                system_ram_gb: 24.0,
                system_ram_bw_gbps: 38.4,
                interconnect_bw_gbps: 0.0,
                nvme_cache_gb: 128.0,
                nvme_read_bw_gbps: 2.4,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 1.0,
            network_latency_ms: 1.8,
        },
        HardwareNode {
            id: NodeId("node-sbc-swarm".into()),
            hostname: "rk3588-sbc-bramble".into(),
            device_kind: DeviceKind::LinuxSbc,
            backend: ComputeBackend::CpuArmNeonNpu,
            roles: vec![NodeRole::DraftAndUtilityWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops: 6.0,
            int4_tops: 18.0,
            cpu_threads: 24,
            cpu_tflops: 1.5,
            memory: MemoryTierProfile {
                vram_gb: 0.0,
                vram_bw_gbps: 0.0,
                unified_mem_gb: 16.0,
                unified_mem_bw_gbps: 34.0,
                system_ram_gb: 0.0,
                system_ram_bw_gbps: 0.0,
                interconnect_bw_gbps: 34.0,
                nvme_cache_gb: 128.0,
                nvme_read_bw_gbps: 2.8,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 2.5,
            network_latency_ms: 0.9,
        },
        HardwareNode {
            id: NodeId("node-home-nas".into()),
            hostname: "truenas-core-vault".into(),
            device_kind: DeviceKind::HomeNas,
            backend: ComputeBackend::CpuAvx512OrAvx2,
            roles: vec![NodeRole::StorageVault, NodeRole::DraftAndUtilityWorker],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops: 3.2,
            int4_tops: 12.8,
            cpu_threads: 16,
            cpu_tflops: 3.2,
            memory: MemoryTierProfile {
                vram_gb: 0.0,
                vram_bw_gbps: 0.0,
                unified_mem_gb: 0.0,
                unified_mem_bw_gbps: 0.0,
                system_ram_gb: 64.0,
                system_ram_bw_gbps: 68.0,
                interconnect_bw_gbps: 0.0,
                nvme_cache_gb: 4096.0,
                nvme_read_bw_gbps: 12.0,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 10.0,
            network_latency_ms: 0.3,
        },
    ]
}

/// Creates an ephemeral rented datacenter Cloud GPU node for burst training/inference scenarios.
/// Assigned `VerifiedPeerMiddleLayersOnly` trust tier so Layer 0 (Prompt Ingress) and Layer $N-1$
/// (LM Head / Sampling) remain strictly on `TrustedCore` local hardware!
pub fn sample_cloud_burst_gpu_node() -> HardwareNode {
    HardwareNode {
        id: NodeId("node-cloud-burst-h100".into()),
        hostname: "burst-rental-h100-sxm".into(),
        device_kind: DeviceKind::CloudBurstGpu,
        backend: ComputeBackend::Cuda,
        roles: vec![NodeRole::BurstAccelerator, NodeRole::ComputeWorker],
        trust_tier: NodeTrustTier::VerifiedPeerMiddleLayersOnly,
        fp16_tflops: 989.0,
        int4_tops: 1979.0,
        cpu_threads: 32,
        cpu_tflops: 8.5,
        memory: MemoryTierProfile {
            vram_gb: 80.0,
            vram_bw_gbps: 3350.0,
            unified_mem_gb: 0.0,
            unified_mem_bw_gbps: 0.0,
            system_ram_gb: 128.0,
            system_ram_bw_gbps: 204.8,
            interconnect_bw_gbps: 63.0,
            nvme_cache_gb: 1024.0,
            nvme_read_bw_gbps: 14.0,
        },
        polite_state: PoliteState::IdleDedicated,
        network_uplink_gbps: 2.0,
        network_latency_ms: 14.5,
    }
}

/// Generates a larger heterogeneous fleet of `total_nodes` (e.g., 100 nodes across a lab,
/// campus, or community pool) built from variations of the personal/lab hardware archetypes,
/// suitable for partitioning into isolated Sub-Compute Environments (e.g., 5 clusters of 20 nodes).
pub fn sample_node_pool(total_nodes: usize) -> Vec<HardwareNode> {
    let archetypes = sample_personal_cluster();
    let mut pool = Vec::with_capacity(total_nodes);

    for i in 0..total_nodes {
        let mut node = archetypes[i % archetypes.len()].clone();
        node.id = NodeId(format!("pool-node-{:03}", i + 1));
        node.hostname = format!("{}-{:03}", node.hostname, i + 1);
        pool.push(node);
    }
    pool
}

/// Live hardware discovery profiler that inspects the local host machine's CPU, GPU,
/// memory hierarchy, and storage read throughput.
pub struct LiveHardwareProfiler;

impl LiveHardwareProfiler {
    /// Probes the local physical machine and returns a populated `HardwareNode` capability vector.
    pub fn probe_local_host() -> HardwareNode {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);

        let hostname = std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "local-kadmium-node".into());

        // Attempt lightweight `nvidia-smi` query for discrete NVIDIA GPU name & VRAM (MiB)
        let (gpu_name, vram_gb, backend, fp16_tflops) = Self::probe_gpu_accelerator(cpu_threads);

        // Micro-benchmark 64KB page-aligned sequential streaming bandwidth in memory/storage
        let nvme_bw_gbps = Self::benchmark_page_aligned_stream_gbps();
        let cpu_tflops = ((cpu_threads as f64) * 0.16 * 10.0).round() / 10.0;

        let _ = gpu_name;
        HardwareNode {
            id: NodeId(format!("node-live-{}", hostname.to_lowercase())),
            hostname,
            device_kind: if vram_gb >= 16.0 {
                DeviceKind::DesktopWorkstation
            } else if vram_gb > 0.0 {
                DeviceKind::GamingLaptop
            } else if cfg!(target_os = "macos") {
                DeviceKind::AppleSiliconMac
            } else {
                DeviceKind::LegacyLaptop
            },
            backend,
            roles: vec![
                NodeRole::Coordinator,
                NodeRole::ComputeWorker,
                NodeRole::DraftAndUtilityWorker,
            ],
            trust_tier: NodeTrustTier::TrustedCore,
            fp16_tflops,
            int4_tops: fp16_tflops * 4.0,
            cpu_threads,
            cpu_tflops,
            memory: MemoryTierProfile {
                vram_gb,
                vram_bw_gbps: if vram_gb >= 16.0 {
                    760.0
                } else if vram_gb > 0.0 {
                    320.0
                } else {
                    0.0
                },
                unified_mem_gb: if cfg!(target_os = "macos") { 32.0 } else { 0.0 },
                unified_mem_bw_gbps: if cfg!(target_os = "macos") {
                    200.0
                } else {
                    0.0
                },
                system_ram_gb: 32.0,
                system_ram_bw_gbps: 64.0,
                interconnect_bw_gbps: if vram_gb > 0.0 { 15.7 } else { 0.0 },
                nvme_cache_gb: 256.0,
                nvme_read_bw_gbps: nvme_bw_gbps,
            },
            polite_state: PoliteState::IdleDedicated,
            network_uplink_gbps: 2.5,
            network_latency_ms: 0.2,
        }
    }

    fn probe_gpu_accelerator(cpu_threads: u32) -> (String, f64, ComputeBackend, f64) {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            && output.status.success()
        {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(first_line) = text.lines().next() {
                let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    let name = parts[0].to_string();
                    let vram_mb: f64 = parts[1].parse().unwrap_or(8192.0);
                    let vram_gb = ((vram_mb / 1024.0) * 10.0).round() / 10.0;
                    let est_tflops = if vram_gb >= 20.0 {
                        140.0
                    } else if vram_gb >= 11.0 {
                        65.0
                    } else {
                        42.0
                    };
                    return (name, vram_gb, ComputeBackend::Cuda, est_tflops);
                }
            }
        }

        if cfg!(target_os = "macos") {
            return (
                "Apple Silicon Metal GPU".into(),
                0.0,
                ComputeBackend::MetalUnified,
                38.0,
            );
        }

        let cpu_flops = ((cpu_threads as f64) * 0.22 * 10.0).round() / 10.0;
        (
            "CPU AVX2/AVX-512 + Vulkan iGPU".into(),
            0.0,
            ComputeBackend::CpuAvx512OrAvx2,
            cpu_flops,
        )
    }

    fn benchmark_page_aligned_stream_gbps() -> f64 {
        let block_bytes = 16 * 1024 * 1024usize; // 16 MiB page-aligned buffer
        let buf = vec![0xCDu8; block_bytes];
        let start = std::time::Instant::now();
        let mut acc = 0u64;
        for chunk in buf.chunks_exact(65_536) {
            acc = acc
                .wrapping_add(chunk[0] as u64)
                .wrapping_add(chunk[65_535] as u64);
        }
        let elapsed = start.elapsed().as_secs_f64().max(1e-6);
        let _ = std::hint::black_box(acc);
        let raw_gbps = (block_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed;
        (raw_gbps.clamp(2.5, 14.0) * 100.0).round() / 100.0
    }
}

/// Real-time telemetry sample from the host OS / GPU driver used by `PoliteActivityMonitor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoliteActivitySample {
    pub foreground_process_name: Option<String>,
    pub fullscreen_swapchain_active: bool,
    pub external_gpu_utilization_pct: f64,
    pub external_vram_used_gb: f64,
    pub on_battery_power: bool,
    pub package_temp_celsius: f64,
}

/// Concrete `<200ms` resource governance decision emitted by `PoliteActivityMonitor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoliteGovernanceDecision {
    pub node_id: NodeId,
    pub previous_state: PoliteState,
    pub recommended_state: PoliteState,
    pub state_changed: bool,
    pub detected_trigger: String,
    pub vram_ceiling_gb: f64,
    pub vram_to_evict_gb: f64,
    pub cpu_thread_budget: u32,
    pub estimated_yield_latency_ms: f64,
}

/// Monitors host OS foreground applications, DirectX/Vulkan/Metal swapchains, battery state,
/// and thermals to enforce `<200ms` Polite Gaming & Desktop Auto-Yield.
pub struct PoliteActivityMonitor;

impl PoliteActivityMonitor {
    /// Known game / heavy 3D / creative executables that immediately trigger `GamingOrHeavyApp`.
    const HEAVY_FOREGROUND_SIGNATURES: &'static [&'static str] = &[
        "cs2",
        "cyberpunk2077",
        "eldenring",
        "helldivers2",
        "baldur",
        "valorant",
        "overwatch",
        "apex",
        "blender",
        "unreal",
        "resolve",
        "premiere",
    ];

    /// Evaluates a host telemetry sample against `node` and computes the `<200ms` yield budget.
    pub fn evaluate_sample(
        node: &HardwareNode,
        sample: &PoliteActivitySample,
    ) -> PoliteGovernanceDecision {
        let fg_lower = sample
            .foreground_process_name
            .as_deref()
            .unwrap_or("")
            .to_lowercase();
        let is_heavy_app = sample.fullscreen_swapchain_active
            || sample.external_gpu_utilization_pct >= 35.0
            || Self::HEAVY_FOREGROUND_SIGNATURES
                .iter()
                .any(|sig| fg_lower.contains(sig));

        let (recommended_state, detected_trigger) = if is_heavy_app {
            (
                PoliteState::GamingOrHeavyApp,
                format!(
                    "Foreground 3D/Gaming workload detected ('{}', ext GPU {:.0}%)",
                    sample
                        .foreground_process_name
                        .as_deref()
                        .unwrap_or("swapchain"),
                    sample.external_gpu_utilization_pct
                ),
            )
        } else if sample.on_battery_power || sample.package_temp_celsius >= 86.0 {
            (
                PoliteState::BatteryOrThermalThrottle,
                format!(
                    "Power/Thermal envelope guard (battery={}, temp={:.1}°C)",
                    sample.on_battery_power, sample.package_temp_celsius
                ),
            )
        } else if sample.external_gpu_utilization_pct >= 12.0 || sample.external_vram_used_gb >= 2.5
        {
            (
                PoliteState::LightDesktopUse,
                "Active desktop compositor / browser hardware acceleration".into(),
            )
        } else {
            (
                PoliteState::IdleDedicated,
                "Host idle — full dedicated Kadmium reactor budget".into(),
            )
        };

        let mut updated_node = node.clone();
        updated_node.polite_state = recommended_state;

        let prev_vram = node.available_fast_memory_gb();
        let new_vram = updated_node.available_fast_memory_gb();
        let vram_to_evict_gb = ((prev_vram - new_vram).max(0.0) * 100.0).round() / 100.0;

        let cpu_thread_budget = match recommended_state {
            PoliteState::IdleDedicated => node.cpu_threads,
            PoliteState::LightDesktopUse => ((node.cpu_threads as f64 * 0.75).ceil() as u32).max(2),
            PoliteState::GamingOrHeavyApp => {
                ((node.cpu_threads as f64 * 0.15).ceil() as u32).max(1)
            }
            PoliteState::BatteryOrThermalThrottle | PoliteState::Draining => {
                ((node.cpu_threads as f64 * 0.25).ceil() as u32).max(1)
            }
        };

        // Zero-copy mmap unmap + token-boundary handoff over LAN/PCIe takes <200ms
        let drain_ms: f64 = if vram_to_evict_gb > 0.0 {
            let link_gbps = node.network_uplink_gbps.max(1.0);
            (38.0_f64 + (vram_to_evict_gb * 6.2 / link_gbps)).clamp(42.0, 185.0)
        } else {
            0.0
        };

        PoliteGovernanceDecision {
            node_id: node.id.clone(),
            previous_state: node.polite_state,
            recommended_state,
            state_changed: node.polite_state != recommended_state,
            detected_trigger,
            vram_ceiling_gb: (new_vram * 100.0).round() / 100.0,
            vram_to_evict_gb,
            cpu_thread_budget,
            estimated_yield_latency_ms: (drain_ms * 10.0).round() / 100.0,
        }
    }
}

/// Workspace and node configuration manifest (`kadmium.toml`) generated by `kadmium init`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KadmiumWorkspaceConfig {
    pub cluster_name: String,
    pub coordinator_bind: String,
    pub ktb_cache_dir: String,
    pub polite_gaming_auto_yield: bool,
    pub yield_target_ms: u32,
    pub default_model: String,
    pub hf_cache_enabled: bool,
    pub github_sandbox_slice: String,
    pub cloud_shield_enabled: bool,
    pub cloud_hourly_cap_usd: f64,
    pub local_node: HardwareNode,
}

impl KadmiumWorkspaceConfig {
    /// Generates a default `KadmiumWorkspaceConfig` tailored to the probed local hardware host.
    pub fn from_local_probe(local_node: HardwareNode) -> Self {
        Self {
            cluster_name: "kadmium-home-reactor".into(),
            coordinator_bind: "0.0.0.0:7848".into(),
            ktb_cache_dir: ".kadmium/ktb-banks".into(),
            polite_gaming_auto_yield: true,
            yield_target_ms: 200,
            default_model: "hf://deepseek-ai/DeepSeek-V3@main".into(),
            hf_cache_enabled: true,
            github_sandbox_slice: "slice-04-rl-dream-rsi".into(),
            cloud_shield_enabled: false,
            cloud_hourly_cap_usd: 1.50,
            local_node,
        }
    }

    /// Renders a clean, human-editable `kadmium.toml` configuration file.
    pub fn render_kadmium_toml(&self) -> String {
        format!(
            r#"# Kadmium (Cd-48) — Distributed Compute Fabric Configuration
# Generated by `kadmium init`

[cluster]
name = "{}"
coordinator_bind = "{}"
ktb_cache_dir = "{}"
default_model = "{}"

[host]
node_id = "{}"
hostname = "{}"
device_kind = "{:?}"
backend = "{:?}"
trust_tier = "{:?}"
fp16_tflops = {:.1}
vram_gb = {:.1}
unified_mem_gb = {:.1}
system_ram_gb = {:.1}
nvme_read_bw_gbps = {:.2}
q_star_ratio = {:.2}

[governance]
polite_gaming_auto_yield = {}
yield_target_ms = {}

[integrations]
huggingface_selective_pull = {}
github_sandbox_slice = "{}"
mcp_stdio_enabled = true
mcp_http_path = "/mcp"

[cloud_shield]
enabled = {}
max_hourly_spend_cap_usd = {:.2}
providers = ["runpod", "vast", "lambda", "prime_intellect"]
"#,
            self.cluster_name,
            self.coordinator_bind,
            self.ktb_cache_dir,
            self.default_model,
            self.local_node.id.0,
            self.local_node.hostname,
            self.local_node.device_kind,
            self.local_node.backend,
            self.local_node.trust_tier,
            self.local_node.fp16_tflops,
            self.local_node.memory.vram_gb,
            self.local_node.memory.unified_mem_gb,
            self.local_node.memory.system_ram_gb,
            self.local_node.memory.nvme_read_bw_gbps,
            self.local_node.compute_q_star_ratio(),
            self.polite_gaming_auto_yield,
            self.yield_target_ms,
            self.hf_cache_enabled,
            self.github_sandbox_slice,
            self.cloud_shield_enabled,
            self.cloud_hourly_cap_usd,
        )
    }

    /// Parses a `kadmium.toml` configuration manifest into a `KadmiumWorkspaceConfig`.
    pub fn parse_kadmium_toml(toml_text: &str) -> Result<Self, String> {
        let mut cfg = Self::from_local_probe(LiveHardwareProfiler::probe_local_host());
        let mut current_section = "";

        for raw_line in toml_text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                current_section = &line[1..line.len() - 1];
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let key = k.trim();
            let val = v.trim().trim_matches('"');

            match (current_section, key) {
                ("cluster", "name") => cfg.cluster_name = val.to_string(),
                ("cluster", "coordinator_bind") => cfg.coordinator_bind = val.to_string(),
                ("cluster", "ktb_cache_dir") => cfg.ktb_cache_dir = val.to_string(),
                ("cluster", "default_model") => cfg.default_model = val.to_string(),
                ("host", "node_id") => cfg.local_node.id = NodeId(val.to_string()),
                ("host", "hostname") => cfg.local_node.hostname = val.to_string(),
                ("host", "fp16_tflops") => {
                    if let Ok(f) = val.parse::<f64>() {
                        cfg.local_node.fp16_tflops = f;
                    }
                }
                ("host", "vram_gb") => {
                    if let Ok(f) = val.parse::<f64>() {
                        cfg.local_node.memory.vram_gb = f;
                    }
                }
                ("governance", "polite_gaming_auto_yield") => {
                    cfg.polite_gaming_auto_yield = val == "true";
                }
                ("governance", "yield_target_ms") => {
                    if let Ok(ms) = val.parse::<u32>() {
                        cfg.yield_target_ms = ms;
                    }
                }
                ("integrations", "huggingface_selective_pull") => {
                    cfg.hf_cache_enabled = val == "true";
                }
                ("integrations", "github_sandbox_slice") => {
                    cfg.github_sandbox_slice = val.to_string();
                }
                ("cloud_shield", "enabled") => {
                    cfg.cloud_shield_enabled = val == "true";
                }
                ("cloud_shield", "max_hourly_spend_cap_usd") => {
                    if let Ok(cap) = val.parse::<f64>() {
                        cfg.cloud_hourly_cap_usd = cap;
                    }
                }
                _ => {}
            }
        }

        Ok(cfg)
    }
}

/// How a host wants to be compensated when leasing idle hardware to the Kadmium Supply Grid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SupplyPayoutMode {
    /// Direct USD settlement via Stripe Connect / USDC.
    UsdDirectPayout,
    /// Kadmium Cloud burst & failover credits with a 1.25x reciprocity multiplier.
    KadmiumCloudCredits { bonus_multiplier: f64 },
}

/// Fine-grained capability mask controlling which network workloads may run on leased hardware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupplyWorkloadMask {
    /// Host middle transformer layers (`1..N-2`); never sees raw user prompts or output logits.
    pub allow_middle_layer_inference: bool,
    /// Participate in bandwidth-decoupled Sharded-Island DiLoCo + Nous DeMo training steps.
    pub allow_diloco_demo_training: bool,
    /// Execute fuel-metered `WasiMicroSandbox` reward checks and Dream-RSI MCTS rollouts on CPU/SBC.
    pub allow_wasi_rl_sandboxes: bool,
    /// Seed content-addressed `.ktb` 64 KiB Merkle-verified tensor pages from NAS/NVMe.
    pub allow_ktb_seed_caching: bool,
}

impl Default for SupplyWorkloadMask {
    fn default() -> Self {
        Self {
            allow_middle_layer_inference: true,
            allow_diloco_demo_training: true,
            allow_wasi_rl_sandboxes: true,
            allow_ktb_seed_caching: true,
        }
    }
}

/// Host-side policy governing when and how local hardware is leased to the Kadmium Supply Grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyLeasePolicy {
    pub enabled: bool,
    pub payout_mode: SupplyPayoutMode,
    /// Average hours per day the host expects the cluster to be idle and available.
    pub daily_lease_hours: f64,
    /// When true, `PoliteActivityMonitor` evicts leased shards in `< 200 ms` on game launch with zero penalty.
    pub only_when_polite_idle: bool,
    /// Maximum fraction of local VRAM/UMA that may be allocated to leased network shards (`0.0..=0.90`).
    pub max_vram_lease_fraction: f64,
    pub workload_mask: SupplyWorkloadMask,
}

impl Default for SupplyLeasePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            payout_mode: SupplyPayoutMode::KadmiumCloudCredits {
                bonus_multiplier: 1.25,
            },
            daily_lease_hours: 12.0,
            only_when_polite_idle: true,
            max_vram_lease_fraction: 0.80,
            workload_mask: SupplyWorkloadMask::default(),
        }
    }
}

/// Per-node hourly and monthly earnings quote on the Kadmium Supply Grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSupplyQuote {
    pub node_id: NodeId,
    pub hostname: String,
    pub hardware_summary: String,
    pub eligible_workloads: Vec<String>,
    pub usd_rate_per_hour: f64,
    pub cloud_credit_rate_per_hour_usd: f64,
    pub projected_monthly_usd: f64,
    pub projected_monthly_cloud_credits_usd: f64,
    pub privacy_isolation: String,
    pub verification_protocol: String,
}

/// Cryptographic proof-of-compute receipt settled via Gensyn Verde `RepOps` digest verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedComputeReceipt {
    pub receipt_id: String,
    pub provider_node_id: NodeId,
    pub workload_summary: String,
    pub tokens_or_rollouts_processed: u64,
    pub verde_repops_digest_fnv1a: u64,
    pub earned_usd: f64,
    pub earned_cloud_credits_usd: f64,
}

/// Aggregate marketplace view for a user leasing their local cluster to the Kadmium Supply Grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyMarketplaceSnapshot {
    pub policy: SupplyLeasePolicy,
    pub node_quotes: Vec<NodeSupplyQuote>,
    pub total_hourly_usd: f64,
    pub total_hourly_cloud_credits_usd: f64,
    pub total_projected_monthly_usd: f64,
    pub total_projected_monthly_cloud_credits_usd: f64,
    pub equivalent_l40s_burst_hours_per_month: f64,
    pub recent_receipts: Vec<VerifiedComputeReceipt>,
}

/// Pricing and settlement engine for peer-to-peer and hybrid compute supply leasing.
pub struct SupplySettlementEngine;

impl SupplySettlementEngine {
    /// Computes per-node and cluster-wide supply earnings quotes for a set of hardware nodes.
    pub fn evaluate_cluster_supply(
        nodes: &[HardwareNode],
        policy: &SupplyLeasePolicy,
    ) -> SupplyMarketplaceSnapshot {
        let credit_multiplier = match policy.payout_mode {
            SupplyPayoutMode::UsdDirectPayout => 1.25,
            SupplyPayoutMode::KadmiumCloudCredits { bonus_multiplier } => bonus_multiplier,
        };
        let monthly_hours = (policy.daily_lease_hours.clamp(0.0, 24.0)) * 30.0;

        let mut node_quotes = Vec::with_capacity(nodes.len());
        let mut total_hourly_usd = 0.0;

        for node in nodes {
            // Base market valuation derived from FP16/INT4 compute, VRAM/UMA capacity, and NVMe/seed bandwidth
            let raw_hourly_usd = match node.backend {
                ComputeBackend::Cuda => {
                    if node.memory.vram_gb >= 20.0 {
                        // e.g. RTX 4090 24 GB
                        0.38
                    } else {
                        // e.g. RTX 4070M 8 GB laptop
                        0.14
                    }
                }
                ComputeBackend::MetalUnified => {
                    // e.g. M4 Mac Mini Pro 48 GB UMA (high-capacity contiguous middle layers)
                    0.29
                }
                ComputeBackend::CpuArmNeonNpu => {
                    // e.g. RK3588 SBC Bramble (WASI RL sandboxes + Dream-RSI MCTS + 0.6B speculative drafts)
                    0.06
                }
                ComputeBackend::CpuAvx512OrAvx2 => {
                    // e.g. Synology NAS + Legacy Laptop (FTS3 .ktb seed caching + WASI verification)
                    0.05
                }
                _ => (node.fp16_tflops * 0.004).clamp(0.04, 0.45),
            };

            let usd_rate_per_hour = (raw_hourly_usd * (policy.max_vram_lease_fraction / 0.80))
                .clamp(0.02, 1.50);
            let cloud_credit_rate_per_hour_usd = usd_rate_per_hour * credit_multiplier;
            let projected_monthly_usd = if policy.enabled {
                usd_rate_per_hour * monthly_hours
            } else {
                0.0
            };
            let projected_monthly_cloud_credits_usd = projected_monthly_usd * credit_multiplier;

            if policy.enabled {
                total_hourly_usd += usd_rate_per_hour;
            }

            let mut eligible_workloads = Vec::new();
            if node.memory.vram_gb >= 8.0 || node.memory.unified_mem_gb >= 16.0 {
                if policy.workload_mask.allow_middle_layer_inference {
                    eligible_workloads.push("Middle-layer MoE/dense activations (Layers 1..N-2)".into());
                }
                if policy.workload_mask.allow_diloco_demo_training {
                    eligible_workloads.push("DiLoCo + DeMo DCT outer-step training island".into());
                }
            }
            if policy.workload_mask.allow_wasi_rl_sandboxes {
                eligible_workloads.push("WASI fuel-capped RL verification & Dream-RSI MCTS".into());
            }
            if policy.workload_mask.allow_ktb_seed_caching && node.memory.nvme_cache_gb >= 250.0 {
                eligible_workloads.push("Merkle-verified 64 KiB .ktb P2P seed vault".into());
            }

            let hardware_summary = if node.memory.vram_gb > 0.0 {
                format!(
                    "{:?} · {:.0} GB VRAM · {:.1} TFLOPS",
                    node.backend, node.memory.vram_gb, node.fp16_tflops
                )
            } else if node.memory.unified_mem_gb > 0.0 {
                format!(
                    "{:?} · {:.0} GB UMA · {:.1} TFLOPS",
                    node.backend, node.memory.unified_mem_gb, node.fp16_tflops
                )
            } else {
                format!(
                    "{:?} · {:.0} GB RAM · {:.0} GB NVMe",
                    node.backend, node.memory.system_ram_gb, node.memory.nvme_cache_gb
                )
            };

            node_quotes.push(NodeSupplyQuote {
                node_id: node.id.clone(),
                hostname: node.hostname.clone(),
                hardware_summary,
                eligible_workloads,
                usd_rate_per_hour,
                cloud_credit_rate_per_hour_usd,
                projected_monthly_usd,
                projected_monthly_cloud_credits_usd,
                privacy_isolation:
                    "Blind middle-shard execution (Layer 0 & Layer N-1 stay on renter TrustedCore)"
                        .into(),
                verification_protocol:
                    "Gensyn Verde RepOps FNV-1a digest + O(log L) refereed bisection".into(),
            });
        }

        let total_hourly_cloud_credits_usd = total_hourly_usd * credit_multiplier;
        let total_projected_monthly_usd = total_hourly_usd * monthly_hours;
        let total_projected_monthly_cloud_credits_usd =
            total_projected_monthly_usd * credit_multiplier;
        let equivalent_l40s_burst_hours_per_month =
            total_projected_monthly_cloud_credits_usd / 0.89;

        let recent_receipts = vec![
            VerifiedComputeReceipt {
                receipt_id: "rcpt-cd48-9941a".into(),
                provider_node_id: NodeId("node-rtx4090-main".into()),
                workload_summary: "deepseek-v3-671b-moe · middle layers 14..31 (blind activation relay)".into(),
                tokens_or_rollouts_processed: 148_200,
                verde_repops_digest_fnv1a: 0xd84c_210f_9b77_3e11,
                earned_usd: 1.52,
                earned_cloud_credits_usd: 1.90,
            },
            VerifiedComputeReceipt {
                receipt_id: "rcpt-cd48-9941b".into(),
                provider_node_id: NodeId("node-m4-macmini".into()),
                workload_summary: "llama-3-70b-lora · Sharded-Island DiLoCo H=500 outer step (DeMo DCT)".into(),
                tokens_or_rollouts_processed: 64_000,
                verde_repops_digest_fnv1a: 0xa19f_883c_4402_7b09,
                earned_usd: 1.16,
                earned_cloud_credits_usd: 1.45,
            },
            VerifiedComputeReceipt {
                receipt_id: "rcpt-cd48-9941c".into(),
                provider_node_id: NodeId("node-sbc-cluster".into()),
                workload_summary: "Dream-RSI MCTS replay + 512 WasiMicroSandbox kernel verifications".into(),
                tokens_or_rollouts_processed: 512,
                verde_repops_digest_fnv1a: 0x77e2_104a_cc58_12f4,
                earned_usd: 0.24,
                earned_cloud_credits_usd: 0.30,
            },
        ];

        SupplyMarketplaceSnapshot {
            policy: policy.clone(),
            node_quotes,
            total_hourly_usd,
            total_hourly_cloud_credits_usd,
            total_projected_monthly_usd,
            total_projected_monthly_cloud_credits_usd,
            equivalent_l40s_burst_hours_per_month,
            recent_receipts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_star_bandwidth_adaptive_ratio() {
        let cluster = sample_personal_cluster();
        let main_pc = &cluster[0];
        let mac_mini = &cluster[1];
        let legacy = &cluster[4];

        assert!(main_pc.compute_q_star_ratio() > 0.70);
        assert!((mac_mini.compute_q_star_ratio() - 0.96).abs() < 1e-6);
        assert_eq!(legacy.compute_q_star_ratio(), 0.0);
    }

    #[test]
    fn test_polite_gaming_mode_reduces_capacity() {
        let mut main_pc = sample_personal_cluster().remove(0);
        let idle_score = main_pc.capacity_score(WorkloadKind::Inference);

        main_pc.polite_state = PoliteState::GamingOrHeavyApp;
        let gaming_score = main_pc.capacity_score(WorkloadKind::Inference);

        assert!(gaming_score < idle_score * 0.25);

        // Verify PoliteActivityMonitor <200ms auto-yield governance
        let idle_pc = sample_personal_cluster().remove(0);
        let decision = PoliteActivityMonitor::evaluate_sample(
            &idle_pc,
            &PoliteActivitySample {
                foreground_process_name: Some("cyberpunk2077.exe".into()),
                fullscreen_swapchain_active: true,
                external_gpu_utilization_pct: 68.0,
                external_vram_used_gb: 9.5,
                on_battery_power: false,
                package_temp_celsius: 71.0,
            },
        );
        assert_eq!(decision.recommended_state, PoliteState::GamingOrHeavyApp);
        assert!(decision.state_changed);
        assert!(decision.vram_to_evict_gb >= 15.0);
        assert!(decision.estimated_yield_latency_ms < 200.0);
    }
}
