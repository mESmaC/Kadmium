//! # FreeToken Bandwidth-Adaptive Execution ($q^*$ Policy) & Elastic Slot Balancer
//!
//! Implements the core mathematical solver and memory-hierarchy balancer from
//! **[`FlashML-org/FreeToken`](https://github.com/FlashML-org/FreeToken)** for heterogeneous
//! edge/desktop/cluster Mixture-of-Experts (MoE) and dense transformer execution:
//!
//! 1. **Real-Time Interconnect Telemetry ([`FreeTokenInterconnectTelemetry`]):**
//!    Captures live or calibrated throughput across `VRAM`, `PCIe / USB4 / OCuLink`, `System RAM`
//!    (including worker-probed `measured_ram_stream_gbps`), and `NVMe`, alongside `GPU`, `CPU`,
//!    and `NPU` compute capabilities under the host's current [`PoliteState`].
//!
//! 2. **Closed-Form Zero-Bubble $q^*$ Co-Execution Solver ([`FreeTokenQStarSolver`]):**
//!    When an MoE layer routes to $K_{\text{active}}$ experts and $K_{\text{miss}}$ of those
//!    experts miss the fast VRAM cache (residing in Warm System RAM or Cold NVMe `.ktb` banks),
//!    naive execution stalls the GPU while streaming all $K_{\text{miss}}$ expert banks across
//!    PCIe. FreeToken eliminates this pipeline bubble by splitting the $K_{\text{miss}}$ experts
//!    across:
//!    - **GPU via PCIe DMA Stream ($q^*_{\text{gpu\_dma}}$):** Streams a fraction of missed
//!      pre-merged `.ktb` expert banks over PCIe to GPU tensor cores,
//!    - **Host CPU In-Place over System RAM ($q^*_{\text{cpu\_ram}}$):** Executes a fraction of
//!      missed experts directly in-place on host AVX2/AVX-512/NEON threads reading zero-copy
//!      memory-mapped `.ktb` banks in System RAM, and
//!    - **Host NPU In-Place over System RAM ($q^*_{\text{npu\_ram}}$):** Dispatches a fraction of
//!      missed experts to the on-die Neural Processing Unit (Apple ANE, Intel AI Boost, Hexagon,
//!      Rockchip NPU) over shared system memory.
//!
//!    Solving $T_{\text{pcie}}(q_{\text{gpu}}) = T_{\text{cpu}}(q_{\text{cpu}}) = T_{\text{npu}}(q_{\text{npu}})$
//!    with shared memory-controller contention factor $\gamma_{\text{mc}}$ yields the closed-form
//!    zero-bubble co-execution policy.
//!
//! 3. **Elastic KV-Cache vs. Resident Expert Slot Balancer ([`ElasticVramSlotBalancer`]):**
//!    Dynamically trades off fast accelerator memory (VRAM / Unified Memory) between growing
//!    KV-cache page allocations and resident LRU/LFRU `.ktb` expert slots as sequence length
//!    scales, without reloading model weights.

use crate::{HardwareNode, PoliteState};
use serde::{Deserialize, Serialize};

const GIB_TO_BYTES: f64 = 1_073_741_824.0;

/// Target execution engine for a single routed MoE expert under the FreeToken $q^*$ policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertExecutionEngine {
    /// Expert is already resident in Tier 0 VRAM or Tier 1 Unified Memory (zero PCIe transfer).
    FastVramOrUnifiedResident,
    /// Expert is streamed from a 64 KiB page-aligned `.ktb` bank in Host RAM over PCIe DMA to GPU.
    GpuPcieDmaStream,
    /// Expert is executed in-place on Host CPU vector threads directly from zero-copy System RAM.
    CpuInPlaceSystemRam,
    /// Expert is executed in-place on Host NPU directly from System RAM.
    NpuInPlaceSystemRam,
    /// Cold expert is prefetched from local NVMe into a double-buffered RAM slot and co-executed.
    NvmeDoubleBufferedStream,
}

/// Real-time or calibrated interconnect and compute telemetry for a node, driving FreeToken's
/// bandwidth-adaptive $q^*$ solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreeTokenInterconnectTelemetry {
    /// Whether CPU, GPU, and NPU share Unified Memory (zero PCIe copy overhead).
    pub is_unified_memory: bool,
    /// Effective fast accelerator memory bandwidth (VRAM or Unified Memory) in GB/s.
    pub fast_mem_bw_gbps: f64,
    /// Sustained Host-to-Device PCIe / Thunderbolt / OCuLink DMA bandwidth in GB/s.
    pub pcie_h2d_bw_gbps: f64,
    /// Sustained System RAM streaming bandwidth in GB/s (worker-probed measurement when available).
    pub ram_stream_bw_gbps: f64,
    /// Sustained sequential NVMe read bandwidth in GB/s for cold `.ktb` banks.
    pub nvme_read_bw_gbps: f64,
    /// Available GPU FP16/INT4 effective TFLOPs under current [`PoliteState`].
    pub gpu_tflops: f64,
    /// Available Host CPU vector TFLOPs under current [`PoliteState`].
    pub cpu_tflops: f64,
    /// Available NPU TOPS under current [`PoliteState`].
    pub npu_tops: f64,
    /// Shared host memory-controller contention factor $\gamma_{\text{mc}} \in (0.0, 1.0]$ when
    /// PCIe DMA and CPU/NPU simultaneously stream from System RAM.
    pub memory_controller_contention_factor: f64,
}

impl FreeTokenInterconnectTelemetry {
    /// Builds telemetry from a [`HardwareNode`] capability vector, applying its [`PoliteState`]
    /// availability multiplier and nominal bandwidth efficiency.
    pub fn from_node(node: &HardwareNode) -> Self {
        let rho = node.polite_state.availability_multiplier();
        let is_unified = node.memory.unified_mem_gb > 0.0;
        let fast_bw = if is_unified {
            node.memory.unified_mem_bw_gbps
        } else {
            node.memory.vram_bw_gbps
        };
        // Real kernels sustain ~60-75% of theoretical peak DRAM/PCIe bandwidth unless overridden
        // by a live worker stream probe.
        let nominal_ram_eff = 0.65;
        let nominal_pcie_eff = 0.78;
        let contention = if is_unified {
            0.92
        } else if node.memory.interconnect_bw_gbps > 0.0 {
            // PCIe DMA reading system RAM steals a fraction of DDR channels from CPU/NPU:
            let pcie_to_ram_ratio = (node.memory.interconnect_bw_gbps
                / node.memory.system_ram_bw_gbps.max(1.0))
            .clamp(0.0, 0.5);
            (1.0 - 0.35 * pcie_to_ram_ratio).clamp(0.65, 0.95)
        } else {
            1.0
        };

        Self {
            is_unified_memory: is_unified,
            fast_mem_bw_gbps: (fast_bw * 0.72 * rho.max(0.25)).max(0.0),
            pcie_h2d_bw_gbps: if is_unified {
                fast_bw * 0.85
            } else {
                (node.memory.interconnect_bw_gbps * nominal_pcie_eff).max(0.0)
            },
            ram_stream_bw_gbps: (node.memory.system_ram_bw_gbps * nominal_ram_eff).max(0.5),
            nvme_read_bw_gbps: (node.memory.nvme_read_bw_gbps * 0.80).max(0.1),
            gpu_tflops: if node.memory.vram_gb > 0.0 || is_unified {
                (node.fp16_tflops * rho).max(0.0)
            } else {
                0.0
            },
            cpu_tflops: (node.cpu_tflops * rho).max(0.05),
            npu_tops: (node.npu_tops * rho).max(0.0),
            memory_controller_contention_factor: contention,
        }
    }

    /// Overrides the nominal System RAM bandwidth with a live worker-probed streaming rate in GB/s.
    pub fn with_measured_ram_stream_gbps(mut self, measured_gbps: f64) -> Self {
        if measured_gbps.is_finite() && measured_gbps > 0.0 {
            self.ram_stream_bw_gbps = measured_gbps;
            // Recompute memory-controller contention against actual measured RAM throughput
            if !self.is_unified_memory && self.pcie_h2d_bw_gbps > 0.0 {
                let ratio = (self.pcie_h2d_bw_gbps / measured_gbps.max(1.0)).clamp(0.0, 0.6);
                self.memory_controller_contention_factor = (1.0 - 0.35 * ratio).clamp(0.60, 0.95);
            }
        }
        self
    }

    /// Overrides the PCIe Host-to-Device DMA bandwidth with a live measured throughput in GB/s.
    pub fn with_measured_pcie_bw_gbps(mut self, measured_pcie_gbps: f64) -> Self {
        if measured_pcie_gbps.is_finite() && measured_pcie_gbps >= 0.0 {
            self.pcie_h2d_bw_gbps = measured_pcie_gbps;
        }
        self
    }

    /// Overrides the NVMe sequential read bandwidth with a live measured throughput in GB/s.
    pub fn with_measured_nvme_bw_gbps(mut self, measured_nvme_gbps: f64) -> Self {
        if measured_nvme_gbps.is_finite() && measured_nvme_gbps > 0.0 {
            self.nvme_read_bw_gbps = measured_nvme_gbps;
        }
        self
    }
}

/// Assignment of a specific routed expert to its optimal execution engine under $q^*$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertCoExecutionAssignment {
    pub expert_id: u32,
    pub engine: ExpertExecutionEngine,
    /// Estimated execution + transfer latency for this expert in milliseconds.
    pub estimated_latency_ms: f64,
}

/// Complete output of the FreeToken $q^*$ Bandwidth-Adaptive Co-Execution Solver for one layer
/// (or node stage).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QStarCoExecutionPlan {
    /// Total active experts routed per token/batch at this layer ($K_{\text{active}}$).
    pub active_experts: u32,
    /// Number of routed experts already hot-resident in VRAM or Unified Memory ($K_{\text{hot}}$).
    pub hot_resident_experts: u32,
    /// Number of routed experts that missed fast memory and reside in System RAM / NVMe ($K_{\text{miss}}$).
    pub cache_missed_experts: u32,
    /// Continuous zero-bubble fraction $q^*_{\text{gpu\_dma}} \in [0.0, 1.0]$ of cache-missed
    /// work streamed over PCIe DMA to GPU.
    pub q_star_miss_gpu_dma_fraction: f64,
    /// Continuous zero-bubble fraction $q^*_{\text{cpu\_ram}} \in [0.0, 1.0]$ of cache-missed
    /// work executed in-place on Host CPU over System RAM.
    pub q_star_miss_cpu_ram_fraction: f64,
    /// Continuous zero-bubble fraction $q^*_{\text{npu\_ram}} \in [0.0, 1.0]$ of cache-missed
    /// work executed in-place on Host NPU over System RAM.
    pub q_star_miss_npu_ram_fraction: f64,
    /// Overall fraction $q^* \in [0.0, 1.0]$ of total active expert compute executed on the
    /// primary GPU/Unified accelerator (combining hot resident experts + PCIe-streamed experts).
    pub overall_q_star_ratio: f64,
    /// Integer count of cache-missed experts streamed over PCIe DMA to GPU.
    pub missed_assigned_to_gpu_dma: u32,
    /// Integer count of cache-missed experts executed in-place on Host CPU + RAM.
    pub missed_assigned_to_cpu_ram: u32,
    /// Integer count of cache-missed experts executed in-place on Host NPU + RAM.
    pub missed_assigned_to_npu_ram: u32,
    /// Per-expert latency (ms) when hot-resident in VRAM / Unified Memory ($\tau_{\text{hot}}$).
    pub tau_hot_vram_ms: f64,
    /// Per-expert latency (ms) when streamed over PCIe DMA and executed on GPU ($\tau_{\text{pcie}}$).
    pub tau_pcie_gpu_ms: f64,
    /// Per-expert latency (ms) when executed in-place on Host CPU over System RAM ($\tau_{\text{cpu}}$).
    pub tau_cpu_ram_ms: f64,
    /// Per-expert latency (ms) when executed in-place on Host NPU over System RAM ($\tau_{\text{npu}}$).
    pub tau_npu_ram_ms: f64,
    /// Baseline latency (ms) if all $K_{\text{miss}}$ experts were naively streamed over PCIe
    /// while CPU/NPU sat idle.
    pub naive_serial_pcie_ms: f64,
    /// Overlapped completion makespan (ms) under FreeToken $q^*$ co-execution.
    pub co_execution_makespan_ms: f64,
    /// Idle pipeline bubble eliminated per layer (ms).
    pub pipeline_bubble_saved_ms: f64,
    /// Speedup multiplier vs. naive serial PCIe streaming for this layer (`>= 1.0`).
    pub speedup_vs_naive_x: f64,
    /// Effective blended bandwidth (GB/s) achieved on cache-missed experts via co-execution.
    pub effective_coexecution_bw_gbps: f64,
}

/// FreeToken closed-form and discrete makespan solver for bandwidth-adaptive co-execution.
pub struct FreeTokenQStarSolver;

impl FreeTokenQStarSolver {
    /// Computes per-expert execution latencies $(\tau_{\text{hot}}, \tau_{\text{pcie}}, \tau_{\text{cpu}}, \tau_{\text{npu}})$
    /// in milliseconds for an expert of size `expert_bytes` and `batch_tokens` active rows.
    ///
    /// For SwiGLU MoE experts (`gate_proj`, `up_proj`, `down_proj`), each weight byte (at Q4/FP8
    /// ~0.5–1.0 bytes/param) corresponds to roughly $2 \times \text{batch\_tokens}$ FLOPs per
    /// parameter. At low batch decode ($B = 1 \dots 8$), execution is strongly memory-bandwidth
    /// bound; at larger prefill batches ($B \ge 32$), compute FLOPs increasingly dominate.
    pub fn compute_per_expert_latencies_ms(
        telemetry: &FreeTokenInterconnectTelemetry,
        expert_bytes: u64,
        batch_tokens: u32,
    ) -> (f64, f64, f64, f64) {
        let bytes = (expert_bytes.max(1024)) as f64;
        let rows = (batch_tokens.max(1)) as f64;
        // Approximate parameters in expert (assuming ~0.625 bytes/param for Q4_K / EXL3):
        let approx_params = bytes / 0.625;
        let expert_flops = 2.0 * approx_params * rows;

        // 1. Hot resident in VRAM / Unified Memory:
        let tau_hot_ms = if telemetry.gpu_tflops > 0.0 && telemetry.fast_mem_bw_gbps > 0.0 {
            let mem_ms = (bytes / (telemetry.fast_mem_bw_gbps * 1e9)) * 1000.0;
            let cmp_ms = (expert_flops / (telemetry.gpu_tflops * 1e12)) * 1000.0;
            mem_ms + cmp_ms
        } else {
            f64::INFINITY
        };

        // 2. Warm expert streamed over PCIe DMA -> GPU:
        let tau_pcie_ms = if telemetry.is_unified_memory && telemetry.gpu_tflops > 0.0 {
            // Unified memory has no PCIe bus; warm and hot share the unified memory controller.
            tau_hot_ms
        } else if telemetry.gpu_tflops > 0.0 && telemetry.pcie_h2d_bw_gbps > 0.0 {
            // Must read from System RAM across PCIe DMA and then run on GPU:
            let eff_dma_bw = telemetry.pcie_h2d_bw_gbps.min(telemetry.ram_stream_bw_gbps);
            let dma_ms = (bytes / (eff_dma_bw.max(0.1) * 1e9)) * 1000.0;
            let gpu_cmp_ms = (expert_flops / (telemetry.gpu_tflops * 1e12)) * 1000.0;
            dma_ms + gpu_cmp_ms
        } else {
            f64::INFINITY
        };

        // 3. Warm expert executed in-place on Host CPU over System RAM:
        let eff_ram_bw =
            (telemetry.ram_stream_bw_gbps * telemetry.memory_controller_contention_factor).max(0.1);
        let tau_cpu_ms = if telemetry.cpu_tflops > 0.0 {
            let ram_ms = (bytes / (eff_ram_bw * 1e9)) * 1000.0;
            let cpu_cmp_ms = (expert_flops / (telemetry.cpu_tflops * 1e12)) * 1000.0;
            ram_ms + cpu_cmp_ms
        } else {
            f64::INFINITY
        };

        // 4. Warm expert executed in-place on Host NPU over System RAM:
        let tau_npu_ms = if telemetry.npu_tops > 0.0 {
            // NPU shares System RAM controller; INT8/INT4 matmul efficiency ~0.75 of peak TOPS
            let ram_ms = (bytes / (eff_ram_bw * 1e9)) * 1000.0;
            let npu_cmp_ms = (expert_flops / (telemetry.npu_tops * 0.75 * 1e12)) * 1000.0;
            ram_ms + npu_cmp_ms
        } else {
            f64::INFINITY
        };

        (tau_hot_ms, tau_pcie_ms, tau_cpu_ms, tau_npu_ms)
    }

    /// Solves the closed-form zero-bubble $q^*$ co-execution split and discrete expert partition
    /// for a layer with `active_experts` ($K_{\text{active}}$) total routed experts, of which
    /// `hot_resident_experts` ($K_{\text{hot}}$) are already in VRAM/Unified Memory and the
    /// remaining $K_{\text{miss}} = K_{\text{active}} - K_{\text{hot}}$ reside in System RAM.
    pub fn solve_for_layer(
        telemetry: &FreeTokenInterconnectTelemetry,
        expert_bytes: u64,
        batch_tokens: u32,
        active_experts: u32,
        hot_resident_experts: u32,
    ) -> QStarCoExecutionPlan {
        let k_active = active_experts.max(1);
        let k_hot = hot_resident_experts.min(k_active);
        let k_miss = k_active.saturating_sub(k_hot);

        let (tau_hot, tau_pcie, tau_cpu, tau_npu) =
            Self::compute_per_expert_latencies_ms(telemetry, expert_bytes, batch_tokens);

        // Expert completion rates (experts per ms) for the three cache-miss execution paths:
        let rate_pcie = if tau_pcie.is_finite() && tau_pcie > 0.0 {
            1.0 / tau_pcie
        } else {
            0.0
        };
        let rate_cpu = if tau_cpu.is_finite() && tau_cpu > 0.0 {
            1.0 / tau_cpu
        } else {
            0.0
        };
        // On Unified Memory (e.g., Apple Silicon), GPU + AMX/CPU + ANE share the same memory
        // controller; NPU + CPU can still assist when compute-bound or via ANE offload, scaled by
        // contention.
        let rate_npu = if tau_npu.is_finite() && tau_npu > 0.0 {
            // Weight NPU contribution by whether it has dedicated SRAM/DMA alongside RAM
            0.65 / tau_npu
        } else {
            0.0
        };

        let total_rate = rate_pcie + rate_cpu + rate_npu;
        let (q_gpu_dma, q_cpu_ram, q_npu_ram) = if total_rate <= f64::EPSILON {
            (0.0, 1.0, 0.0)
        } else {
            (
                rate_pcie / total_rate,
                rate_cpu / total_rate,
                rate_npu / total_rate,
            )
        };

        // Discrete makespan-minimizing assignment for the `k_miss` missed experts:
        // Each step assigns the next missed expert to whichever engine finishes earliest if given
        // one more expert: `argmin_e (count_e + 1) * tau_e`.
        // Furthermore, GPU already has `k_hot * tau_hot` of hot-resident work queued on the stream!
        let hot_gpu_ms = if tau_hot.is_finite() {
            (k_hot as f64) * tau_hot
        } else {
            0.0
        };

        let mut n_gpu = 0u32;
        let mut n_cpu = 0u32;
        let mut n_npu = 0u32;

        for _ in 0..k_miss {
            let next_gpu_finish = if tau_pcie.is_finite() {
                hot_gpu_ms + ((n_gpu + 1) as f64) * tau_pcie
            } else {
                f64::INFINITY
            };
            let next_cpu_finish = if tau_cpu.is_finite() {
                ((n_cpu + 1) as f64) * tau_cpu
            } else {
                f64::INFINITY
            };
            // NPU + CPU share system RAM bandwidth; account for combined RAM stream load:
            let next_npu_finish = if tau_npu.is_finite() && telemetry.npu_tops > 0.0 {
                ((n_npu + 1) as f64) * (tau_npu / 0.65)
            } else {
                f64::INFINITY
            };

            if next_gpu_finish <= next_cpu_finish && next_gpu_finish <= next_npu_finish {
                n_gpu += 1;
            } else if next_cpu_finish <= next_npu_finish {
                n_cpu += 1;
            } else {
                n_npu += 1;
            }
        }

        let gpu_total_ms = hot_gpu_ms
            + if tau_pcie.is_finite() {
                (n_gpu as f64) * tau_pcie
            } else {
                0.0
            };
        let cpu_total_ms = if tau_cpu.is_finite() {
            (n_cpu as f64) * tau_cpu
        } else {
            0.0
        };
        let npu_total_ms = if tau_npu.is_finite() {
            (n_npu as f64) * tau_npu
        } else {
            0.0
        };

        // Co-execution makespan is the max across parallel hardware pipelines:
        let co_makespan_ms = gpu_total_ms.max(cpu_total_ms).max(npu_total_ms);

        // Baseline naive latency: if GPU present, streaming all `k_miss` over PCIe (or if CPU-only,
        // running all on CPU):
        let naive_ms = if tau_pcie.is_finite() {
            hot_gpu_ms + (k_miss as f64) * tau_pcie
        } else if tau_cpu.is_finite() {
            (k_active as f64) * tau_cpu
        } else {
            co_makespan_ms
        };

        let bubble_saved_ms = (naive_ms - co_makespan_ms).max(0.0);
        let speedup_x = if co_makespan_ms > 1e-9 {
            (naive_ms / co_makespan_ms).max(1.0)
        } else {
            1.0
        };

        // Overall fraction of active experts executed on primary GPU/Unified accelerator:
        let overall_q_star = if k_miss == 0 {
            if telemetry.is_unified_memory {
                0.96
            } else if telemetry.gpu_tflops > 0.0 {
                ((k_hot as f64 + q_gpu_dma) / (k_active as f64 + 1.0)).clamp(0.10, 0.95)
            } else {
                0.0
            }
        } else {
            ((k_hot as f64 + (k_miss as f64) * q_gpu_dma) / (k_active as f64)).clamp(
                if telemetry.gpu_tflops > 0.0 {
                    0.10
                } else {
                    0.0
                },
                if telemetry.is_unified_memory {
                    0.96
                } else {
                    0.95
                },
            )
        };

        // Effective blended bandwidth (GB/s) for the missed experts under co-execution:
        let missed_bytes_total = (k_miss as f64) * (expert_bytes as f64);
        let miss_only_makespan_ms = ((n_gpu as f64)
            * if tau_pcie.is_finite() { tau_pcie } else { 0.0 })
        .max(cpu_total_ms)
        .max(npu_total_ms);
        let eff_coexec_bw_gbps = if k_miss > 0 && miss_only_makespan_ms > 1e-9 {
            (missed_bytes_total / (miss_only_makespan_ms / 1000.0)) / 1e9
        } else {
            telemetry.ram_stream_bw_gbps + telemetry.pcie_h2d_bw_gbps
        };

        QStarCoExecutionPlan {
            active_experts: k_active,
            hot_resident_experts: k_hot,
            cache_missed_experts: k_miss,
            q_star_miss_gpu_dma_fraction: q_gpu_dma,
            q_star_miss_cpu_ram_fraction: q_cpu_ram,
            q_star_miss_npu_ram_fraction: q_npu_ram,
            overall_q_star_ratio: overall_q_star,
            missed_assigned_to_gpu_dma: n_gpu,
            missed_assigned_to_cpu_ram: n_cpu,
            missed_assigned_to_npu_ram: n_npu,
            tau_hot_vram_ms: tau_hot,
            tau_pcie_gpu_ms: tau_pcie,
            tau_cpu_ram_ms: tau_cpu,
            tau_npu_ram_ms: tau_npu,
            naive_serial_pcie_ms: naive_ms,
            co_execution_makespan_ms: co_makespan_ms,
            pipeline_bubble_saved_ms: bubble_saved_ms,
            speedup_vs_naive_x: speedup_x,
            effective_coexecution_bw_gbps: eff_coexec_bw_gbps,
        }
    }

    /// Assigns concrete `expert_ids` from a layer's routing selection to their target
    /// [`ExpertExecutionEngine`]s according to the solved [`QStarCoExecutionPlan`].
    pub fn assign_routed_experts(
        plan: &QStarCoExecutionPlan,
        routed_expert_ids: &[u32],
        hot_resident_cutoff_or_set: &[u32],
    ) -> Vec<ExpertCoExecutionAssignment> {
        let mut assignments = Vec::with_capacity(routed_expert_ids.len());
        let mut rem_gpu_dma = plan.missed_assigned_to_gpu_dma;
        let mut rem_cpu_ram = plan.missed_assigned_to_cpu_ram;

        for &expert_id in routed_expert_ids {
            if hot_resident_cutoff_or_set.contains(&expert_id) && plan.tau_hot_vram_ms.is_finite() {
                assignments.push(ExpertCoExecutionAssignment {
                    expert_id,
                    engine: ExpertExecutionEngine::FastVramOrUnifiedResident,
                    estimated_latency_ms: plan.tau_hot_vram_ms,
                });
            } else if rem_gpu_dma > 0 && plan.tau_pcie_gpu_ms.is_finite() {
                rem_gpu_dma -= 1;
                assignments.push(ExpertCoExecutionAssignment {
                    expert_id,
                    engine: ExpertExecutionEngine::GpuPcieDmaStream,
                    estimated_latency_ms: plan.tau_pcie_gpu_ms,
                });
            } else if rem_cpu_ram > 0 || !plan.tau_npu_ram_ms.is_finite() {
                rem_cpu_ram = rem_cpu_ram.saturating_sub(1);
                assignments.push(ExpertCoExecutionAssignment {
                    expert_id,
                    engine: ExpertExecutionEngine::CpuInPlaceSystemRam,
                    estimated_latency_ms: plan.tau_cpu_ram_ms,
                });
            } else {
                assignments.push(ExpertCoExecutionAssignment {
                    expert_id,
                    engine: ExpertExecutionEngine::NpuInPlaceSystemRam,
                    estimated_latency_ms: plan.tau_npu_ram_ms,
                });
            }
        }

        assignments
    }
}

// ================================================================================================
// Elastic VRAM / Unified Memory Balancer: KV-Cache Pages vs. Resident `.ktb` Expert Slots
// ================================================================================================

/// Snapshot produced by [`ElasticVramSlotBalancer`] when context length or [`PoliteState`] changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElasticSlotBalanceSnapshot {
    pub context_tokens: u32,
    pub polite_state: PoliteState,
    /// Total fast memory offered under the current [`PoliteState`] in GiB.
    pub offered_fast_mem_gb: f64,
    /// Fast memory pinned by dense attention/router trunk weights in GiB.
    pub pinned_trunk_gb: f64,
    /// Fast memory consumed by active KV-cache pages at `context_tokens` in GiB.
    pub active_kv_cache_gb: f64,
    /// Remaining fast memory available for hot resident `.ktb` expert slots in GiB.
    pub expert_slot_budget_gb: f64,
    /// Number of 64 KiB page-aligned `.ktb` expert banks that fit in fast memory slots.
    pub resident_fast_expert_slots: u32,
    /// Number of experts demoted to Warm System RAM `.ktb` banks as KV cache grew.
    pub warm_ram_expert_slots: u32,
    /// Expected per-layer fast-memory expert cache hit rate $\in [0.0, 1.0]$ under power-law
    /// routing locality ($\alpha$).
    pub expected_fast_hit_rate: f64,
    /// Updated FreeToken $q^*$ co-execution plan for a representative layer at this operating point.
    pub layer_q_star_plan: QStarCoExecutionPlan,
}

/// FreeToken's **Elastic KV-Cache vs. Resident Expert Slot Balancer**.
///
/// Rather than statically partitioning GPU VRAM at model load time (which either starves long
/// contexts or wastes VRAM during short prompts), `ElasticVramSlotBalancer` dynamically trades off
/// 64 KiB page-aligned VRAM blocks between growing KV-cache pages and resident `.ktb` MoE expert
/// slots, updating $q^*$ on the fly as experts are promoted/demoted between VRAM and System RAM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticVramSlotBalancer {
    pub num_layers_on_node: u32,
    pub experts_per_layer: u32,
    pub active_experts_per_token: u32,
    pub single_expert_size_gb: f64,
    pub pinned_trunk_size_gb: f64,
    /// KV-cache memory per token per layer in bytes (e.g., 2,048 B for MLA or 16,384 B for GQA).
    pub kv_bytes_per_token_per_layer: u64,
    /// Power-law routing popularity skew $\alpha$ (e.g., `0.25` to `0.65`).
    pub routing_skew_alpha: f64,
}

impl ElasticVramSlotBalancer {
    pub fn new(
        num_layers_on_node: u32,
        experts_per_layer: u32,
        active_experts_per_token: u32,
        single_expert_size_gb: f64,
        pinned_trunk_size_gb: f64,
        kv_bytes_per_token_per_layer: u64,
    ) -> Self {
        Self {
            num_layers_on_node: num_layers_on_node.max(1),
            experts_per_layer: experts_per_layer.max(1),
            active_experts_per_token: active_experts_per_token.max(1),
            single_expert_size_gb: single_expert_size_gb.max(0.001),
            pinned_trunk_size_gb: pinned_trunk_size_gb.max(0.0),
            kv_bytes_per_token_per_layer: kv_bytes_per_token_per_layer.max(256),
            routing_skew_alpha: 0.45,
        }
    }

    /// Computes the cumulative routing probability captured by the top `resident_per_layer`
    /// most popular experts out of `experts_per_layer` under a power-law distribution
    /// $P(r) \propto (r + 1)^{-\alpha}$.
    pub fn power_law_hit_rate(&self, resident_per_layer: u32) -> f64 {
        let n = self.experts_per_layer;
        let k = resident_per_layer.min(n);
        if k == 0 {
            return 0.0;
        }
        if k >= n {
            return 1.0;
        }
        let alpha = self.routing_skew_alpha.max(0.0);
        let mut top_sum = 0.0f64;
        let mut total_sum = 0.0f64;
        for r in 0..n {
            let w = ((r + 1) as f64).powf(-alpha);
            if r < k {
                top_sum += w;
            }
            total_sum += w;
        }
        if total_sum <= f64::EPSILON {
            (k as f64) / (n as f64)
        } else {
            (top_sum / total_sum).clamp(0.0, 1.0)
        }
    }

    /// Evaluates the elastic VRAM/UMA slot balance and updated $q^*$ co-execution plan at a given
    /// `context_tokens` length and optional measured RAM streaming bandwidth.
    pub fn balance_at_context(
        &self,
        node: &HardwareNode,
        context_tokens: u32,
        measured_ram_stream_gbps: Option<f64>,
    ) -> ElasticSlotBalanceSnapshot {
        let offer = node.offered_now();
        let offered_fast_gb = offer.fast_memory_gb;
        let total_experts_on_node = self.num_layers_on_node * self.experts_per_layer;

        let kv_total_bytes = (context_tokens as u64)
            .saturating_mul(self.num_layers_on_node as u64)
            .saturating_mul(self.kv_bytes_per_token_per_layer);
        let active_kv_gb = (kv_total_bytes as f64) / GIB_TO_BYTES;

        let remaining_fast_gb =
            (offered_fast_gb - self.pinned_trunk_size_gb - active_kv_gb).max(0.0);
        let max_fast_slots = ((remaining_fast_gb / self.single_expert_size_gb).floor() as u32)
            .min(total_experts_on_node);
        let warm_ram_slots = total_experts_on_node.saturating_sub(max_fast_slots);

        let slots_per_layer = max_fast_slots / self.num_layers_on_node.max(1);
        let hit_rate = self.power_law_hit_rate(slots_per_layer);
        let expected_hot_active = ((self.active_experts_per_token as f64) * hit_rate)
            .round()
            .clamp(0.0, self.active_experts_per_token as f64)
            as u32;

        let mut telemetry = FreeTokenInterconnectTelemetry::from_node(node);
        if let Some(measured_bw) = measured_ram_stream_gbps {
            telemetry = telemetry.with_measured_ram_stream_gbps(measured_bw);
        }
        let expert_bytes = (self.single_expert_size_gb * GIB_TO_BYTES).round() as u64;
        let layer_q_star_plan = FreeTokenQStarSolver::solve_for_layer(
            &telemetry,
            expert_bytes,
            1,
            self.active_experts_per_token,
            expected_hot_active,
        );

        ElasticSlotBalanceSnapshot {
            context_tokens,
            polite_state: node.polite_state,
            offered_fast_mem_gb: offered_fast_gb,
            pinned_trunk_gb: self.pinned_trunk_size_gb,
            active_kv_cache_gb: active_kv_gb,
            expert_slot_budget_gb: remaining_fast_gb,
            resident_fast_expert_slots: max_fast_slots,
            warm_ram_expert_slots: warm_ram_slots,
            expected_fast_hit_rate: hit_rate,
            layer_q_star_plan,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PoliteState, sample_personal_cluster};

    #[test]
    fn freetoken_q_star_solver_eliminates_pcie_bubble_via_coexecution() {
        let cluster = sample_personal_cluster();
        let desktop = &cluster[0]; // RTX 5070 12GB VRAM + 64GB DDR5 + Intel AI Boost NPU
        let telemetry =
            FreeTokenInterconnectTelemetry::from_node(desktop).with_measured_ram_stream_gbps(58.0);

        // Suppose a 32 MiB expert bank, 8 active experts routed, 2 hot in VRAM, 6 cache-missed in RAM:
        let expert_bytes = 32 * 1024 * 1024;
        let plan = FreeTokenQStarSolver::solve_for_layer(&telemetry, expert_bytes, 1, 8, 2);

        assert_eq!(plan.active_experts, 8);
        assert_eq!(plan.hot_resident_experts, 2);
        assert_eq!(plan.cache_missed_experts, 6);
        assert_eq!(
            plan.missed_assigned_to_gpu_dma
                + plan.missed_assigned_to_cpu_ram
                + plan.missed_assigned_to_npu_ram,
            6
        );
        // Both GPU PCIe DMA and Host CPU/NPU in-place RAM engines participate:
        assert!(plan.missed_assigned_to_gpu_dma > 0);
        assert!(plan.missed_assigned_to_cpu_ram + plan.missed_assigned_to_npu_ram > 0);
        // Co-execution must reduce latency compared to naive serial PCIe streaming:
        assert!(plan.co_execution_makespan_ms < plan.naive_serial_pcie_ms);
        assert!(plan.pipeline_bubble_saved_ms > 0.0);
        assert!(plan.speedup_vs_naive_x > 1.15);
        assert!(plan.effective_coexecution_bw_gbps > telemetry.pcie_h2d_bw_gbps);
    }

    #[test]
    fn freetoken_q_star_adapts_to_live_interconnect_degradation() {
        let cluster = sample_personal_cluster();
        let desktop = &cluster[0];
        let expert_bytes = 16 * 1024 * 1024;

        // Fast PCIe x16 (28 GB/s) vs constrained PCIe x4 / OCuLink (6 GB/s):
        let fast_pcie = FreeTokenInterconnectTelemetry::from_node(desktop)
            .with_measured_pcie_bw_gbps(28.0)
            .with_measured_ram_stream_gbps(60.0);
        let slow_pcie = FreeTokenInterconnectTelemetry::from_node(desktop)
            .with_measured_pcie_bw_gbps(6.0)
            .with_measured_ram_stream_gbps(60.0);

        let plan_fast = FreeTokenQStarSolver::solve_for_layer(&fast_pcie, expert_bytes, 1, 8, 0);
        let plan_slow = FreeTokenQStarSolver::solve_for_layer(&slow_pcie, expert_bytes, 1, 8, 0);

        // When PCIe bandwidth drops from 28 GB/s to 6 GB/s, q* automatically shifts missed experts
        // away from PCIe DMA toward in-place Host CPU/NPU RAM execution!
        assert!(
            plan_slow.q_star_miss_gpu_dma_fraction < plan_fast.q_star_miss_gpu_dma_fraction,
            "expected lower GPU DMA fraction on slow PCIe: slow={} vs fast={}",
            plan_slow.q_star_miss_gpu_dma_fraction,
            plan_fast.q_star_miss_gpu_dma_fraction
        );
        assert!(
            plan_slow.missed_assigned_to_cpu_ram + plan_slow.missed_assigned_to_npu_ram
                >= plan_fast.missed_assigned_to_cpu_ram + plan_fast.missed_assigned_to_npu_ram
        );
    }

    #[test]
    fn elastic_vram_slot_balancer_trades_expert_slots_for_kv_cache_without_reload() {
        let cluster = sample_personal_cluster();
        let mut desktop = cluster[0].clone();
        // 16 layers on this node, 64 experts/layer (1,024 total experts = 25.6 GiB), top-8 active,
        // 0.025 GiB (25.6 MiB) per expert, 2.5 GiB pinned trunk, 16 KiB/token/layer KV cache:
        let balancer = ElasticVramSlotBalancer::new(16, 64, 8, 0.025, 2.5, 16_384);

        let short_ctx = balancer.balance_at_context(&desktop, 512, Some(56.0));
        let long_ctx = balancer.balance_at_context(&desktop, 32_768, Some(56.0));

        // As context grows from 512 -> 32,768 tokens, KV cache grows and elastic fast expert slots
        // smoothly demote to warm RAM without dropping below 0:
        assert!(long_ctx.active_kv_cache_gb > short_ctx.active_kv_cache_gb);
        assert!(long_ctx.resident_fast_expert_slots < short_ctx.resident_fast_expert_slots);
        assert!(long_ctx.warm_ram_expert_slots > short_ctx.warm_ram_expert_slots);
        assert!(long_ctx.expected_fast_hit_rate <= short_ctx.expected_fast_hit_rate);

        // If user starts gaming (`PoliteState::GamingOrHeavyApp`), fast VRAM yields immediately
        // and FreeToken shifts missed experts to warm RAM co-execution:
        desktop.polite_state = PoliteState::GamingOrHeavyApp;
        let gaming_snap = balancer.balance_at_context(&desktop, 4_096, Some(56.0));
        assert!(gaming_snap.offered_fast_mem_gb < short_ctx.offered_fast_mem_gb);
        assert!(gaming_snap.resident_fast_expert_slots < short_ctx.resident_fast_expert_slots);
    }
}
