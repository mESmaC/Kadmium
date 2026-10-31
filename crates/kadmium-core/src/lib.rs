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
    /// Individual Android / iOS smartphone (Termux / Native Worker / ADB) with LPDDR Unified Memory & NPU.
    MobilePhone,
    /// High-density 20-Slot Phone Farm Box (stripped ARM64 Snapdragon motherboards, battery-eliminator PSU, USB/OTG switch).
    PhoneFarmChassis,
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

    /// Total memory usable for resident model weights: Tier 0 + Tier 1 + the offered share of
    /// Tier 2 system RAM ([`canonical_numbers::OFFER_SYSTEM_RAM_FRAC`]).
    pub fn resident_weight_capacity_gb(&self) -> f64 {
        self.vram_gb
            + self.unified_mem_gb
            + (self.system_ram_gb * canonical_numbers::OFFER_SYSTEM_RAM_FRAC)
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

/// What a node's owner lets the grid use: the budgets the planner and scheduler spend. Lumabri's
/// donor settings: the owner sets what is offered, a reserve of RAM is never offered, at most a
/// few model allocations share the node, and a peer's plan is reviewed before anything is staged.
/// The default offer comes from the capability vector and the canonical fractions; an explicit
/// offer is clamped to the machine with [`ResourceOffer::clamped_to`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceOffer {
    /// VRAM or unified memory the grid may fill with weights and KV, in GiB, when the host is idle.
    pub fast_memory_gb: f64,
    /// System RAM the grid may fill, in GiB; never reaches into `reserved_ram_gb`.
    pub system_ram_gb: f64,
    /// Local NVMe the grid may fill with cold banks, in GiB.
    pub nvme_gb: f64,
    /// RAM kept for the owner, in GiB, whatever `system_ram_gb` says.
    pub reserved_ram_gb: f64,
    /// Work instances the node runs at once.
    pub slots: u32,
    /// Distinct model allocations the node accepts at once.
    pub max_allocations: u32,
    /// The owner reviews a placement plan before anything is staged on the node.
    pub requires_approval: bool,
}

impl ResourceOffer {
    /// RAM kept for the owner: a quarter of the machine's RAM, bounded to 1-4 GiB.
    pub fn ram_reserve_gb(total_ram_gb: f64) -> f64 {
        (total_ram_gb * canonical_numbers::OFFER_RAM_RESERVE_FRAC)
            .clamp(
                canonical_numbers::OFFER_RAM_RESERVE_MIN_GB,
                canonical_numbers::OFFER_RAM_RESERVE_MAX_GB,
            )
            .min(total_ram_gb.max(0.0))
    }

    /// The offer a node makes before its owner edits it: the canonical share of discrete VRAM and
    /// all of the unified memory, the canonical share of RAM, all of the NVMe cache, the default
    /// slots, and approval required unless the hardware is the owner's own.
    pub fn default_for(node: &HardwareNode) -> Self {
        let total_ram = node.memory.system_ram_gb.max(0.0);
        let reserve = Self::ram_reserve_gb(total_ram);
        Self {
            fast_memory_gb: node.memory.vram_gb * canonical_numbers::OFFER_FAST_MEMORY_FRAC
                + node.memory.unified_mem_gb,
            system_ram_gb: (total_ram * canonical_numbers::OFFER_SYSTEM_RAM_FRAC)
                .min((total_ram - reserve).max(0.0)),
            nvme_gb: node.memory.nvme_cache_gb.max(0.0),
            reserved_ram_gb: reserve,
            slots: canonical_numbers::OFFER_DEFAULT_SLOTS,
            max_allocations: canonical_numbers::OFFER_MAX_ALLOCATIONS,
            requires_approval: node.trust_tier != NodeTrustTier::TrustedCore,
        }
    }

    /// Clamps an owner's offer to the machine: never more memory than it has, never into the
    /// reserve, at least one slot.
    pub fn clamped_to(mut self, node: &HardwareNode) -> Self {
        let total_ram = node.memory.system_ram_gb.max(0.0);
        self.reserved_ram_gb = self
            .reserved_ram_gb
            .max(Self::ram_reserve_gb(total_ram))
            .min(total_ram);
        self.fast_memory_gb = self
            .fast_memory_gb
            .clamp(0.0, node.memory.fast_accelerator_memory_gb());
        self.system_ram_gb = self
            .system_ram_gb
            .clamp(0.0, (total_ram - self.reserved_ram_gb).max(0.0));
        self.nvme_gb = self.nvme_gb.clamp(0.0, node.memory.nvme_cache_gb.max(0.0));
        self.slots = self.slots.max(1);
        self.max_allocations = self.max_allocations.max(1);
        self
    }

    /// The offer under the host's polite state. VRAM goes back to the owner's game first, unified
    /// memory and RAM scale with the availability multiplier, slots scale with it too, and a
    /// draining host offers nothing.
    pub fn under_polite_state(&self, node: &HardwareNode, state: PoliteState) -> Self {
        let rho = state.availability_multiplier();
        let vram_factor = match state {
            PoliteState::GamingOrHeavyApp => 0.15,
            PoliteState::BatteryOrThermalThrottle => 0.50,
            PoliteState::Draining => 0.0,
            _ => 1.0,
        };
        let fast_total = node.memory.fast_accelerator_memory_gb();
        // Split the offered fast memory between VRAM and unified memory in the machine's ratio.
        let (vram_share, unified_share) = if fast_total > 0.0 {
            (
                self.fast_memory_gb * node.memory.vram_gb / fast_total,
                self.fast_memory_gb * node.memory.unified_mem_gb / fast_total,
            )
        } else {
            (0.0, 0.0)
        };
        let unified_factor = if state == PoliteState::Draining {
            0.0
        } else {
            rho.max(0.5)
        };
        Self {
            fast_memory_gb: vram_share * vram_factor + unified_share * unified_factor,
            system_ram_gb: self.system_ram_gb * rho,
            nvme_gb: self.nvme_gb,
            reserved_ram_gb: self.reserved_ram_gb,
            slots: if state == PoliteState::Draining {
                0
            } else {
                ((self.slots as f64 * rho).ceil() as u32).max(1)
            },
            max_allocations: self.max_allocations,
            requires_approval: self.requires_approval,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "{:.1} GB fast, {:.1} GB RAM ({:.1} GB reserved), {:.0} GB NVMe, {} slots, {} allocations{}",
            self.fast_memory_gb,
            self.system_ram_gb,
            self.reserved_ram_gb,
            self.nvme_gb,
            self.slots,
            self.max_allocations,
            if self.requires_approval {
                ", approval required"
            } else {
                ""
            }
        )
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
    /// Detected GPU or primary accelerator model name (e.g., `"NVIDIA GeForce RTX 5070"` or `"Apple M4 10-Core Metal 4 GPU"`).
    #[serde(default)]
    pub gpu_name: String,
    /// Detected host CPU model name (e.g., `"Intel Core Ultra 7 265F"` or `"Apple M4 (4P+6E + AMX)"`).
    #[serde(default)]
    pub cpu_name: String,
    /// Optional dedicated Neural Processing Unit model name (e.g., `"Intel AI Boost NPU"` or `"Apple M4 16-Core Neural Engine"`).
    #[serde(default)]
    pub npu_name: Option<String>,
    /// Dedicated NPU INT8/FP16 TOPS (e.g., `13.0` for Intel AI Boost Arrow Lake, `38.0` for Apple M4 16-Core ANE, `6.0` for K230 KPU).
    #[serde(default)]
    pub npu_tops: f64,
    /// Peak FP16 / BF16 TFLOPs on primary GPU/accelerator.
    pub fp16_tflops: f64,
    /// Peak GPU INT4 / FP4 quantized TOPS.
    pub int4_tops: f64,
    /// Host CPU threads available for parallel host-bank execution ($q^*$ CPU split).
    pub cpu_threads: u32,
    /// Host CPU FP16/FP32 vector TFLOPs for hybrid execution.
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
        (self.fp16_tflops + self.cpu_tflops * 0.35 + self.npu_tops * 0.25) * rho
    }

    /// Effective fast memory (VRAM + Unified Memory) available under the current PoliteState:
    /// the default offer's fast memory under that state.
    pub fn available_fast_memory_gb(&self) -> f64 {
        self.offered_now().fast_memory_gb
    }

    /// The offer this node makes before its owner edits it; see [`ResourceOffer::default_for`].
    pub fn default_offer(&self) -> ResourceOffer {
        ResourceOffer::default_for(self)
    }

    /// The default offer under the node's current polite state.
    pub fn offered_now(&self) -> ResourceOffer {
        self.default_offer()
            .under_polite_state(self, self.polite_state)
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

    /// Preset: Llama-3.2-3B-Instruct Dense (Q4_K `.ktb`, 2.0 GiB across 28 layers for single SBC / edge / laptop).
    pub fn llama_3_2_3b_instruct() -> Self {
        Self {
            id: "llama-3.2-3b-instruct".to_string(),
            display_name: "Llama-3.2-3B-Instruct (Q4_K KTB · 2.0 GB)".to_string(),
            total_params_billions: 3.2,
            num_layers: 28,
            is_moe: false,
            dense_layer_size_gb: 0.072,
            experts_per_layer: 0,
            active_experts_per_token: 0,
            single_expert_size_gb: 0.0,
            kv_cache_per_layer_gb: 0.018,
            training_activation_per_layer_gb: 0.06,
            lora_state_per_layer_gb: 0.015,
        }
    }

    /// Preset: Qwen2.5-Coder-7B-Instruct Dense (Q4_K `.ktb`, 4.7 GiB across 28 layers).
    pub fn qwen2_5_coder_7b() -> Self {
        Self {
            id: "qwen2.5-coder-7b-instruct".to_string(),
            display_name: "Qwen2.5-Coder-7B-Instruct (Q4_K KTB · 4.7 GB)".to_string(),
            total_params_billions: 7.6,
            num_layers: 28,
            is_moe: false,
            dense_layer_size_gb: 0.168,
            experts_per_layer: 0,
            active_experts_per_token: 0,
            single_expert_size_gb: 0.0,
            kv_cache_per_layer_gb: 0.032,
            training_activation_per_layer_gb: 0.11,
            lora_state_per_layer_gb: 0.025,
        }
    }

    /// Preset: Qwen2.5-Coder-14B-Instruct Dense (Q4_K `.ktb`, 9.0 GiB across 48 layers).
    pub fn qwen2_5_coder_14b() -> Self {
        Self {
            id: "qwen2.5-coder-14b-instruct".to_string(),
            display_name: "Qwen2.5-Coder-14B-Instruct (Q4_K KTB · 9.0 GB)".to_string(),
            total_params_billions: 14.7,
            num_layers: 48,
            is_moe: false,
            dense_layer_size_gb: 0.188,
            experts_per_layer: 0,
            active_experts_per_token: 0,
            single_expert_size_gb: 0.0,
            kv_cache_per_layer_gb: 0.048,
            training_activation_per_layer_gb: 0.18,
            lora_state_per_layer_gb: 0.04,
        }
    }

    /// Preset: Qwen2.5-Coder-32B-Instruct Dense (Q4_K `.ktb`, 18.6 GiB across 64 layers for consumer LAN/M4/GPU clusters).
    pub fn qwen2_5_coder_32b() -> Self {
        Self {
            id: "qwen2.5-coder-32b-instruct".to_string(),
            display_name: "Qwen2.5-Coder-32B-Instruct (Q4_K KTB · 18.6 GB)".to_string(),
            total_params_billions: 32.5,
            num_layers: 64,
            is_moe: false,
            dense_layer_size_gb: 0.29,
            experts_per_layer: 0,
            active_experts_per_token: 0,
            single_expert_size_gb: 0.0,
            kv_cache_per_layer_gb: 0.08,
            training_activation_per_layer_gb: 0.28,
            lora_state_per_layer_gb: 0.06,
        }
    }

    /// Preset: DeepSeek-V3 671B MoE quantized with EXL3 / QTIP Trellis Quantization (`2.55 bpw` hybrid `.ktb`).
    /// Attention trunks at `4.0 bpw` EXL3; 256 routed MoE experts at `2.4 bpw` EXL3 (~194 GB total vs 351 GB Q4_K).
    pub fn deepseek_v3_671b_exl3() -> Self {
        Self {
            id: "deepseek-v3-671b-exl3-2.5bpw".to_string(),
            display_name: "DeepSeek-V3 671B MoE (EXL3/QTIP 2.55 bpw Trellis KTB)".to_string(),
            total_params_billions: 671.0,
            num_layers: 61,
            is_moe: true,
            dense_layer_size_gb: 0.24,
            experts_per_layer: 64,
            active_experts_per_token: 6,
            single_expert_size_gb: 0.046,
            kv_cache_per_layer_gb: 0.10,
            training_activation_per_layer_gb: 0.42,
            lora_state_per_layer_gb: 0.08,
        }
    }

    /// Selects a realistic model preset sized to fit the actual fast memory (VRAM + Unified Memory)
    /// of the connected cluster nodes without ungrounded overflow.
    pub fn recommended_for_cluster(nodes: &[HardwareNode]) -> Self {
        let fast_mem_gb: f64 = nodes
            .iter()
            .map(|n| n.memory.vram_gb + n.memory.unified_mem_gb)
            .sum();
        if fast_mem_gb < 6.0 {
            Self::llama_3_2_3b_instruct()
        } else if fast_mem_gb < 11.5 {
            Self::qwen2_5_coder_7b()
        } else if fast_mem_gb < 20.0 {
            Self::qwen2_5_coder_14b()
        } else if fast_mem_gb < 52.0 {
            Self::qwen2_5_coder_32b()
        } else if fast_mem_gb < 120.0 {
            Self::llama_3_70b_lora()
        } else {
            Self::deepseek_v3_671b_moe()
        }
    }

    /// Resolves a model preset by ID or returns `deepseek_v3_671b_moe()` by default.
    pub fn from_id(id: &str) -> Self {
        Self::try_from_id(id).unwrap_or_else(Self::deepseek_v3_671b_moe)
    }

    /// Resolves only known presets and aliases, without silently selecting a default.
    pub fn try_from_id(id: &str) -> Option<Self> {
        Some(match id {
            "llama-3.2-3b-instruct" | "llama-3.2-3b" => Self::llama_3_2_3b_instruct(),
            "qwen2.5-coder-7b-instruct" | "qwen2.5-coder-7b" => Self::qwen2_5_coder_7b(),
            "qwen2.5-coder-14b-instruct" | "qwen2.5-coder-14b" => Self::qwen2_5_coder_14b(),
            "qwen2.5-coder-32b-instruct" | "qwen2.5-coder-32b" => Self::qwen2_5_coder_32b(),
            "qwen3-235b-moe" => Self::qwen3_235b_moe(),
            "llama-3-70b-lora" | "llama-3.3-70b-instruct" => Self::llama_3_70b_lora(),
            "deepseek-v3-671b-exl3-2.5bpw" | "deepseek-v3-exl3" | "exl3" => {
                Self::deepseek_v3_671b_exl3()
            }
            "deepseek-v3-671b-moe" => Self::deepseek_v3_671b_moe(),
            _ => return None,
        })
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
            gpu_name: "NVIDIA GeForce RTX 4090 (24 GB)".into(),
            cpu_name: "Intel Core i9-14900K (24C/32T)".into(),
            npu_name: None,
            npu_tops: 0.0,
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
            gpu_name: "Apple M4 Pro 20-Core Metal 4 GPU".into(),
            cpu_name: "Apple M4 Pro (10P+4E + AMX)".into(),
            npu_name: Some("Apple M4 16-Core Neural Engine".into()),
            npu_tops: 38.0,
            fp16_tflops: 17.0,
            int4_tops: 34.0,
            cpu_threads: 14,
            cpu_tflops: 4.4,
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
            gpu_name: "NVIDIA GeForce RTX 4070 Laptop GPU".into(),
            cpu_name: "Intel Core Ultra 9 185H (16C/22T)".into(),
            npu_name: Some("Intel AI Boost NPU".into()),
            npu_tops: 11.5,
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
            gpu_name: "NVIDIA GeForce RTX 3070 Laptop GPU".into(),
            cpu_name: "AMD Ryzen 7 5800H (8C/16T)".into(),
            npu_name: None,
            npu_tops: 0.0,
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
            gpu_name: "Intel UHD Graphics 620".into(),
            cpu_name: "Intel Core i7-8550U (4C/8T)".into(),
            npu_name: None,
            npu_tops: 0.0,
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
            gpu_name: "ARM Mali-G610 MP4".into(),
            cpu_name: "3x Rockchip RK3588 (24x Cortex-A76/A55)".into(),
            npu_name: Some("3x Rockchip RK3588 6-TOPS NPU".into()),
            npu_tops: 18.0,
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
            gpu_name: "Headless Server CPU".into(),
            cpu_name: "AMD EPYC 4344P (8C/16T AVX-512)".into(),
            npu_name: None,
            npu_tops: 0.0,
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
        gpu_name: "NVIDIA H100 SXM5 80GB".into(),
        cpu_name: "Intel Xeon Platinum 8480+".into(),
        npu_name: None,
        npu_tops: 0.0,
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

pub mod canonical_numbers;
pub mod governance;
pub mod host_ledger;
pub mod k230_bridge;
pub mod learning;
pub mod m4_mac_mini;
pub mod os_hooks;
pub mod phone_farm;
pub mod platform;
pub mod rbac;
pub mod reactor_clearinghouse;
pub mod reterminal_cm4;
pub mod supply;
pub mod tegra_switch;

pub use canonical_numbers::*;
pub use governance::*;
pub use host_ledger::*;
pub use k230_bridge::*;
pub use learning::*;
pub use m4_mac_mini::*;
pub use os_hooks::*;
pub use phone_farm::*;
pub use platform::*;
pub use rbac::*;
pub use reactor_clearinghouse::*;
pub use reterminal_cm4::*;
pub use supply::*;
pub use tegra_switch::*;

/// Returns `true` when external LAN/USB/ADB hardware probes are permitted.
///
/// Loopback endpoints (`127.0.0.1` or `localhost`) are always allowed so local mock listeners
/// in unit tests work deterministically. Non-loopback LAN/USB/ADB probes are disabled under
/// `cfg(test)` or when `KADMIUM_DISABLE_LIVE_PROBES` is set in the environment.
pub fn external_hardware_probes_allowed(custom_endpoint: Option<&str>) -> bool {
    if let Some(ep) = custom_endpoint {
        let trimmed = ep.trim();
        if trimmed == "127.0.0.1"
            || trimmed.starts_with("127.0.0.1:")
            || trimmed == "localhost"
            || trimmed.starts_with("localhost:")
        {
            return true;
        }
    }
    if cfg!(test)
        || std::env::var_os("KADMIUM_DISABLE_LIVE_PROBES").is_some()
        || std::env::var_os("RUST_TEST_THREADS").is_some()
        || std::env::var_os("NEXTEST_EXECUTION_MODE").is_some()
    {
        return false;
    }
    true
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

        let (quotas, keys) = SubClusterRbacEngine::sample_lab_rbac_catalog();
        assert!(
            SubClusterRbacEngine::authorize_slice_request(
                &keys[0],
                &quotas[0],
                WorkloadKind::Inference,
                1.25,
                true
            )
            .is_ok()
        );
        // BlindSupplyProvider must be rejected if requesting Layer 0 / N-1 ingress/egress
        assert!(
            SubClusterRbacEngine::authorize_slice_request(
                &keys[2],
                &quotas[2],
                WorkloadKind::Inference,
                0.0,
                true
            )
            .is_err()
        );
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

        // Verify OsSwapchainHysteresisEngine fast-yield (0ms delay) and slow-reclaim (15,000ms cooldown)
        let hyst = OsSwapchainHysteresisEngine::sample_desktop_gaming_timeline(&idle_pc);
        assert_eq!(hyst.timeline[1].new_state, PoliteState::GamingOrHeavyApp);
        // Step 2 (at 8,000ms, 6s after game launch) holds GamingOrHeavyApp due to 15s hysteresis window!
        assert_eq!(hyst.timeline[2].new_state, PoliteState::GamingOrHeavyApp);
        assert!(hyst.timeline[2].cooldown_remaining_ms > 0);
        // Step 3 (at 25,000ms, >15s after game activity) reclaims IdleDedicated
        assert_eq!(hyst.timeline[3].new_state, PoliteState::IdleDedicated);
    }
}
