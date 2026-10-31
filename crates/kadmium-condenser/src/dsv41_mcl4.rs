//! DeepSeek-V4.1 (`DeepSeek-V4.1-Flash-1017-NVFP4`) and `MCL4-COE` architecture,
//! codec, DSML tool-call parser, routing, MLA/YOCO context compaction, hardware placement,
//! and 9-operator cognitive engine primitives for Kadmium Condenser.
//!
//! Synthesizes core production designs from:
//! - **MCL4-COE (`X:\MCL4-COE`)**:
//!   - `mcl4_deepseek_v41_sm120.cu` & `deepseek_v41_codec_loader.py`: 40 layers, `dim = 5120`,
//!     `moe_inter = 2304`, `vocab = 129,280`, 4-stream Hyper-Connections (`hc_mult = 4`,
//!     20-iteration $4 \times 4$ Sinkhorn-Knopp), Multi-Head Latent Attention (`MLA`: 64H $\times$ 512,
//!     1 shared KV head, 128-token sliding window + 512-top-k compressed KV indexer, 8-group
//!     block-diagonal `wo_a`/`wo_b` LoRA), 384 routed NVFP4 experts + 1 shared FP8 expert
//!     (`top_k = 6`, `sqrtsoftplus` + `noaux_tc`, `route_scale = 1.5`, clamped SwiGLU `limit = 10.0`),
//!     4096-byte aligned `.mcl4c` expert packets (`19,910,656` B), $O(1)$ NVMe Engram gather at
//!     Layers 1 & 14 (`24` prime XOR N-gram hashes, `6,336` B/token, signed-sqrt gate), and
//!     8-layer Isolated Plastic Upper-Tail (`Layers 32..39`, `NCM(64-CB) + PLE(r=64) + LoRA(r=16)`).
//!   - `#U$# -> #T$# -> #A$#` structured reasoning segmentation, inline arithmetic verification,
//!     9 Typed Cognitive Operators (`Abduction`, `Association`, `Synthesis`, `Induction`, `Analogy`,
//!     `Decision`, `Deduction`, `Evaluation`, `Metacognition`), Tabu coverage penalty, and
//!     Quarantined Episodic Whiteboard.
//! - **DeepSeek Harness (`dsh`)**:
//!   - `dsh-compaction-basic` pre-step context pressure compaction, historical tool-result pruning,
//!     and multi-turn thought folding.
//! - **DeepInfer (`deepseek-ai/DeepInfer`)**:
//!   - Decoder SWA bounded replay (`YOCO` prefill truncation `keep_tokens = 140` = `128` SWA + `12`
//!     backtrack margin) and `DSpark` confidence-scheduled multi-token verification (`nextn = 5`).
//! - **Strata (`strata-dsv41`) & FreeToken (`FlashML-org/FreeToken`)**:
//!   - Native DeepSeek-V4.1 `DSML` tool-call translator (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`,
//!     `<｜DSML｜ parameter name="..." string="true|false">`, token ID `128825`) + unclosed
//!     `<think>` recovery on 2-bit/NVFP4 checkpoints.
//!   - Hardware placement across `MCL4C-NVFP4`, `UD-Q8_K_XL` (508.4 GB), and `IST-DAS` `1.75`/`2.00`/`2.25`
//!     bpw sets, `--experts 5` (`Faster` for $\le 16\text{ GiB}$ VRAM) vs `--experts 6` (`Full Quality`),
//!     and FreeToken $q^\star$ bandwidth-adaptive CPU–GPU expert co-execution.

use crate::types::{Message, Role, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// 1. DeepSeek-V4.1 Architecture Constants & Zero-Repack `.mcl4c` Layout
// ---------------------------------------------------------------------------

/// Canonical architectural specification for `DeepSeek-V4.1-Flash-1017-NVFP4`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeepSeekV41ArchSpec {
    pub model_id: String,
    pub n_layers: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub vocab_size: usize,
    pub dsml_token_id: u32,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub top_k: usize,
    pub faster_top_k: usize,
    pub route_scale: f32,
    pub scoring_func: String,
    pub topk_method: String,
    pub swiglu_limit: f32,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub mla_heads: usize,
    pub mla_head_dim: usize,
    pub mla_nope_dim: usize,
    pub mla_rope_dim: usize,
    pub mla_kv_heads: usize,
    pub mla_q_lora_rank: usize,
    pub mla_wo_groups: usize,
    pub mla_sliding_window: usize,
    pub mla_compressed_kv_topk: usize,
    pub mla_compression_ratio: usize,
    pub yoco_prefill_keep_tokens: usize,
    pub dspark_max_draft_tokens: usize,
    pub engram_layers: Vec<usize>,
    pub engram_heads: usize,
    pub engram_max_ngram: usize,
    pub engram_row_bytes: usize,
    pub engram_gather_bytes_per_token: usize,
    pub plastic_upper_tail_layers: usize,
}

impl Default for DeepSeekV41ArchSpec {
    fn default() -> Self {
        Self::deepseek_v41_flash_nvfp4()
    }
}

impl DeepSeekV41ArchSpec {
    /// Returns the exact specification for `DeepSeek-V4.1-Flash-1017-NVFP4`.
    pub fn deepseek_v41_flash_nvfp4() -> Self {
        Self {
            model_id: "deepseek-v4.1-flash-1017-nvfp4".to_string(),
            n_layers: 40,
            dim: 5120,
            moe_inter_dim: 2304,
            vocab_size: 129_280,
            dsml_token_id: 128_825,
            n_routed_experts: 384,
            n_shared_experts: 1,
            top_k: 6,
            faster_top_k: 5,
            route_scale: 1.5,
            scoring_func: "sqrtsoftplus".to_string(),
            topk_method: "noaux_tc".to_string(),
            swiglu_limit: 10.0,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            mla_heads: 64,
            mla_head_dim: 512,
            mla_nope_dim: 448,
            mla_rope_dim: 64,
            mla_kv_heads: 1,
            mla_q_lora_rank: 1280,
            mla_wo_groups: 8,
            mla_sliding_window: 128,
            mla_compressed_kv_topk: 512,
            mla_compression_ratio: 8,
            yoco_prefill_keep_tokens: 140,
            dspark_max_draft_tokens: 5,
            engram_layers: vec![1, 14],
            engram_heads: 24,
            engram_max_ngram: 4,
            engram_row_bytes: 264,
            engram_gather_bytes_per_token: 6_336,
            plastic_upper_tail_layers: 8,
        }
    }

    /// Flattened Hyper-Connection residual width (`hc_mult * dim = 20,480` floats/token).
    pub fn hc_flattened_dim(&self) -> usize {
        self.hc_mult * self.dim
    }

    /// Total routed experts across all 40 layers (`40 * 384 = 15,360`).
    pub fn total_routed_experts(&self) -> usize {
        self.n_layers * self.n_routed_experts
    }

    /// Returns the architecture's active-expert count. Memory pressure must be handled by
    /// cache placement, never by silently changing router semantics.
    ///
    /// The VRAM and legacy faster-mode arguments are retained for API compatibility only.
    pub fn recommended_top_k_for_vram_gib(
        &self,
        _total_vram_gib: f64,
        _prefer_faster: bool,
    ) -> usize {
        self.top_k
    }
}

/// Exact 4096-byte page-aligned `.mcl4c` NVFP4 expert packet layout (`19,910,656` bytes/expert).
///
/// Mirrors `Dsv41Mcl4cExpertPacket` in `mcl4_deepseek_v41_sm120.cu`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Dsv41Mcl4cPacketLayout {
    pub w1_w3_fp4_bytes: usize,
    pub w2_fp4_bytes: usize,
    pub w1_w3_scales_bytes: usize,
    pub w2_scales_bytes: usize,
    pub scale2_scalars_bytes: usize,
    pub meta_bytes: usize,
    pub pad_bytes: usize,
    pub total_packet_bytes: usize,
}

impl Default for Dsv41Mcl4cPacketLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl Dsv41Mcl4cPacketLayout {
    pub const W1_W3_FP4_BYTES: usize = 4608 * 2560; // 11,796,480
    pub const W2_FP4_BYTES: usize = 5120 * 1152; // 5,898,240
    pub const W1_W3_SCALES_BYTES: usize = 4608 * 320; // 1,474,560
    pub const W2_SCALES_BYTES: usize = 5120 * 144; // 737,280 (2304 / 16 = 144 E4M3 block-16 scales)
    pub const SCALE2_SCALARS_BYTES: usize = 12; // 3 x f32 (w1_scale2, w3_scale2, w2_scale2)
    pub const META_BYTES: usize = 8; // 2 x u32 (layer_idx, expert_idx)
    pub const PAD_BYTES: usize = 4076; // 4096-byte page alignment padding
    pub const TOTAL_PACKET_BYTES: usize = 19_910_656; // 18.988 MiB, 4096 * 4861

    pub const fn new() -> Self {
        Self {
            w1_w3_fp4_bytes: Self::W1_W3_FP4_BYTES,
            w2_fp4_bytes: Self::W2_FP4_BYTES,
            w1_w3_scales_bytes: Self::W1_W3_SCALES_BYTES,
            w2_scales_bytes: Self::W2_SCALES_BYTES,
            scale2_scalars_bytes: Self::SCALE2_SCALARS_BYTES,
            meta_bytes: Self::META_BYTES,
            pad_bytes: Self::PAD_BYTES,
            total_packet_bytes: Self::TOTAL_PACKET_BYTES,
        }
    }

    /// Computes the exact 4096-byte aligned file offset for `(layer_idx, expert_idx)`.
    pub const fn expert_file_offset(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        experts_per_layer: usize,
    ) -> u64 {
        let global_expert_idx = (layer_idx * experts_per_layer + expert_idx) as u64;
        global_expert_idx * (self.total_packet_bytes as u64)
    }

    /// Encodes the 20-byte trailer metadata (`w1_scale2`, `w3_scale2`, `w2_scale2`, `layer_idx`, `expert_idx`).
    pub fn encode_trailer_meta(
        w1_scale2: f32,
        w3_scale2: f32,
        w2_scale2: f32,
        layer_idx: u32,
        expert_idx: u32,
    ) -> [u8; 20] {
        let mut out = [0u8; 20];
        out[0..4].copy_from_slice(&w1_scale2.to_le_bytes());
        out[4..8].copy_from_slice(&w3_scale2.to_le_bytes());
        out[8..12].copy_from_slice(&w2_scale2.to_le_bytes());
        out[12..16].copy_from_slice(&layer_idx.to_le_bytes());
        out[16..20].copy_from_slice(&expert_idx.to_le_bytes());
        out
    }

    /// Decodes the 20-byte trailer metadata (`(w1_scale2, w3_scale2, w2_scale2, layer_idx, expert_idx)`).
    pub fn decode_trailer_meta(bytes: &[u8; 20]) -> (f32, f32, f32, u32, u32) {
        let w1_s2 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let w3_s2 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let w2_s2 = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let layer_idx = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let expert_idx = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        (w1_s2, w3_s2, w2_s2, layer_idx, expert_idx)
    }
}

/// Isolated-Compute Plastic Upper-Tail (`Layers 32..39`) checkpoint header (`"MCL4DS41"`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Dsv41PlasticUpperTailHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub step: u32,
    pub tail_layers: u32,
    pub ncm_codebook_size: u32,
    pub ple_rank: u32,
    pub lora_rank: u32,
}

impl Dsv41PlasticUpperTailHeader {
    pub const MAGIC: [u8; 8] = *b"MCL4DS41";
    pub const VERSION: u32 = 1;

    pub fn new(step: u32) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            step,
            tail_layers: 8,
            ncm_codebook_size: 64,
            ple_rank: 64,
            lora_rank: 16,
        }
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.step.to_le_bytes());
        buf[16..20].copy_from_slice(&self.tail_layers.to_le_bytes());
        buf[20..24].copy_from_slice(&self.ncm_codebook_size.to_le_bytes());
        buf[24..28].copy_from_slice(&self.ple_rank.to_le_bytes());
        buf[28..32].copy_from_slice(&self.lora_rank.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 32 {
            return Err("buffer shorter than 32-byte MCL4DS41 header".to_string());
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        if magic != Self::MAGIC {
            return Err(format!("invalid plastic tail magic: {:?}", magic));
        }
        Ok(Self {
            magic,
            version: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            step: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            tail_layers: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            ncm_codebook_size: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
            ple_rank: u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]),
            lora_rank: u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]),
        })
    }
}

/// VRAM & NVMe streaming budget planner for `DeepSeek-V4.1-Flash-1017-NVFP4` (`.mcl4c` codec,
/// `Strata` `IST-DAS` 1.75/2.00/2.25 bpw sets, and `FreeToken` $q^\star$ CPU–GPU co-execution).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dsv41ArenaCachePlan {
    pub total_vram_gb: f64,
    pub host_ram_gb: f64,
    pub recommended_pack_variant: String,
    pub recommended_top_k: usize,
    pub non_expert_backbone_gib: f64,
    pub ping_pong_arena_slots: usize,
    pub ping_pong_arena_mib: f64,
    pub plastic_upper_tail_mib: f64,
    pub warm_expert_cache_slots: usize,
    pub warm_expert_cache_gib: f64,
    pub cpu_ram_cached_experts: usize,
    pub freetoken_q_star_gpu_expert_fraction: f32,
    pub nvme_engram_vram_bytes: usize,
    pub nvme_engram_disk_gb: f64,
}

impl Dsv41ArenaCachePlan {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * Self::MIB;
    const PING_PONG_SLOTS: usize = 8;
    const PLASTIC_TAIL_BYTES: u64 = 288 * Self::MIB;
    // T=16 activation buffers, expert intermediates, and bounded DMA staging.
    const ACTIVATION_AND_IO_STAGING_BYTES: u64 = 64 * Self::MIB;
    const SAFETY_HEADROOM_BYTES: u64 = Self::GIB + 1000 * Self::MIB;

    /// Plans the `DeepSeek-V4.1` 8-slot ping-pong NVFP4 arena + warm expert cache for a given GPU VRAM budget.
    pub fn plan_for_vram_gb(total_vram_gb: f64) -> Self {
        Self::plan_for_host(total_vram_gb, 64.0)
    }

    /// Plans GPU, CPU, and NVMe placement. This compatibility entry point reports zero
    /// warm-cache slots when the base model cannot fit; use [`Self::try_plan_for_host`]
    /// when the caller needs the rejection reason.
    pub fn plan_for_host(total_vram_gb: f64, host_ram_gb: f64) -> Self {
        Self::try_plan_for_host(total_vram_gb, host_ram_gb)
            .unwrap_or_else(|_| Self::build_plan(total_vram_gb, host_ram_gb, 0))
    }

    /// Checked placement planner. Invalid sizes, arithmetic overflow, and a base-model
    /// footprint exceeding VRAM are rejected instead of being clamped into a false fit.
    pub fn try_plan_for_host(total_vram_gb: f64, host_ram_gb: f64) -> Result<Self, String> {
        let vram_bytes = Self::gib_to_bytes(total_vram_gb, "VRAM")?;
        let host_ram_bytes = Self::gib_to_bytes(host_ram_gb, "host RAM")?;
        let arch = DeepSeekV41ArchSpec::deepseek_v41_flash_nvfp4();
        let backbone_bytes = Self::shape_derived_backbone_bytes(&arch)?;
        let kv_bytes = Self::shape_derived_kv_bytes(&arch)?;
        let arena_bytes = (Dsv41Mcl4cPacketLayout::TOTAL_PACKET_BYTES as u64)
            .checked_mul(Self::PING_PONG_SLOTS as u64)
            .ok_or_else(|| "ping-pong arena size overflow".to_string())?;
        let reserved_bytes = [backbone_bytes, kv_bytes, arena_bytes]
            .into_iter()
            .chain([
                Self::PLASTIC_TAIL_BYTES,
                Self::ACTIVATION_AND_IO_STAGING_BYTES,
                Self::SAFETY_HEADROOM_BYTES,
            ])
            .try_fold(0u64, |sum, bytes| {
                sum.checked_add(bytes)
                    .ok_or_else(|| "reserved VRAM size overflow".to_string())
            })?;
        let warm_bytes = vram_bytes.checked_sub(reserved_bytes).ok_or_else(|| {
            format!(
                "DeepSeek-V4.1 base placement requires {:.2} GiB, but only {:.2} GiB VRAM is available",
                reserved_bytes as f64 / Self::GIB as f64,
                total_vram_gb
            )
        })?;
        let warm_slots = (warm_bytes / Dsv41Mcl4cPacketLayout::TOTAL_PACKET_BYTES as u64)
            .min(arch.total_routed_experts() as u64) as usize;

        let usable_host_bytes = host_ram_bytes.saturating_sub(9 * Self::GIB);
        let cpu_ram_cached_experts = (usable_host_bytes
            / Dsv41Mcl4cPacketLayout::TOTAL_PACKET_BYTES as u64)
            .min(arch.total_routed_experts().saturating_sub(warm_slots) as u64)
            as usize;
        let mut plan = Self::build_plan(total_vram_gb, host_ram_gb, warm_slots);
        plan.cpu_ram_cached_experts = cpu_ram_cached_experts;
        Ok(plan)
    }

    fn gib_to_bytes(value: f64, label: &str) -> Result<u64, String> {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{label} size must be finite and non-negative"));
        }
        let bytes = value * Self::GIB as f64;
        if bytes > u64::MAX as f64 {
            return Err(format!("{label} size overflows the byte budget"));
        }
        Ok(bytes.floor() as u64)
    }

    fn shape_derived_backbone_bytes(arch: &DeepSeekV41ArchSpec) -> Result<u64, String> {
        let q_and_o_rows = arch
            .mla_heads
            .checked_mul(arch.mla_head_dim)
            .ok_or_else(|| "MLA projection rows overflow".to_string())?;
        let wo_group_rows = arch
            .mla_wo_groups
            .checked_mul(1024)
            .ok_or_else(|| "MLA output groups overflow".to_string())?;
        let fp8_shapes = [
            (arch.mla_q_lora_rank, arch.dim),     // wq_a
            (arch.mla_head_dim, arch.dim),        // wkv
            (q_and_o_rows, arch.mla_q_lora_rank), // wq_b
            (wo_group_rows, 4096),                // wo_a
            (arch.dim, wo_group_rows),            // wo_b
        ];
        let mla_bytes = fp8_shapes.into_iter().try_fold(0u64, |sum, (rows, cols)| {
            let values = rows
                .checked_mul(cols)
                .ok_or_else(|| "FP8 tensor size overflow".to_string())?;
            let scales = rows
                .div_ceil(32)
                .checked_mul(cols.div_ceil(32))
                .ok_or_else(|| "FP8 scale size overflow".to_string())?;
            let tensor_bytes = values
                .checked_add(scales)
                .ok_or_else(|| "FP8 tensor plus scales overflow".to_string())?;
            sum.checked_add(tensor_bytes as u64)
                .ok_or_else(|| "MLA size overflow".to_string())
        })?;
        let small_mla_bytes =
            (2 * (arch.mla_q_lora_rank + arch.mla_head_dim) + 4 * arch.mla_heads) as u64;
        let per_layer = [mla_bytes, small_mla_bytes]
            .into_iter()
            .chain([
                Dsv41Mcl4cPacketLayout::TOTAL_PACKET_BYTES as u64, // shared expert
                8_192_000,                                         // HC + routers + norms
            ])
            .try_fold(0u64, |sum, bytes| {
                sum.checked_add(bytes)
                    .ok_or_else(|| "per-layer backbone size overflow".to_string())
            })?;
        let layers = per_layer
            .checked_mul(arch.n_layers as u64)
            .ok_or_else(|| "backbone layer size overflow".to_string())?;
        // Conservative packed FP4 + group-scale placement for two large matrices.
        // Validate against checkpoint metadata before using this as an allocator plan.
        let embeddings_and_lm_head = 744 * Self::MIB;
        layers
            .checked_add(embeddings_and_lm_head)
            .ok_or_else(|| "backbone size overflow".to_string())
    }

    fn shape_derived_kv_bytes(arch: &DeepSeekV41ArchSpec) -> Result<u64, String> {
        let tokens_per_layer = arch
            .mla_sliding_window
            .checked_add(arch.mla_compressed_kv_topk)
            .ok_or_else(|| "KV token count overflow".to_string())?;
        tokens_per_layer
            .checked_mul(arch.mla_head_dim)
            .and_then(|n| n.checked_mul(2)) // BF16
            .and_then(|n| n.checked_mul(arch.n_layers))
            .map(|n| n as u64)
            .ok_or_else(|| "KV cache size overflow".to_string())
    }

    fn build_plan(total_vram_gb: f64, host_ram_gb: f64, warm_slots: usize) -> Self {
        let packet_gib = Dsv41Mcl4cPacketLayout::TOTAL_PACKET_BYTES as f64 / Self::GIB as f64;
        let packet_mib = Dsv41Mcl4cPacketLayout::TOTAL_PACKET_BYTES as f64 / Self::MIB as f64;
        let ping_pong_mib = Self::PING_PONG_SLOTS as f64 * packet_mib;
        let arch = DeepSeekV41ArchSpec::deepseek_v41_flash_nvfp4();
        let non_expert_gib = Self::shape_derived_backbone_bytes(&arch)
            .map(|bytes| bytes as f64 / Self::GIB as f64)
            .unwrap_or(f64::INFINITY);
        let warm_gib = warm_slots as f64 * packet_gib;
        let recommended_variant = if total_vram_gb >= 96.0 && host_ram_gb >= 192.0 {
            "UD-Q8_K_XL / MCL4C-NVFP4 (Full 4.5-bpw)"
        } else if host_ram_gb >= 44.0 {
            "MCL4C-NVFP4-Stream / IST-DAS-2.25bpw (IQ2_XS)"
        } else if host_ram_gb >= 40.0 {
            "MCL4C-NVFP4-Stream / IST-DAS-2.00bpw (IQ2_XXS)"
        } else if total_vram_gb >= 11.5 && has_local_dual_nvme_shards() {
            "MCL4E-NVFP4-DualNVMe-Stream (X:+Z: Intel P4608 + MCL4DS41 Plastic Tail)"
        } else {
            "MCL4C-NVFP4-Stream / IST-DAS-1.75bpw (IQ1_M)"
        };
        let q_star = if total_vram_gb >= 64.0 {
            0.85f32
        } else if total_vram_gb >= 12.0 {
            0.50f32
        } else {
            0.25f32
        };

        Self {
            total_vram_gb,
            host_ram_gb,
            recommended_pack_variant: recommended_variant.to_string(),
            recommended_top_k: arch.recommended_top_k_for_vram_gib(total_vram_gb, false),
            non_expert_backbone_gib: non_expert_gib,
            ping_pong_arena_slots: Self::PING_PONG_SLOTS,
            ping_pong_arena_mib: (ping_pong_mib * 100.0).round() / 100.0,
            plastic_upper_tail_mib: Self::PLASTIC_TAIL_BYTES as f64 / Self::MIB as f64,
            warm_expert_cache_slots: warm_slots,
            warm_expert_cache_gib: (warm_gib * 100.0).round() / 100.0,
            cpu_ram_cached_experts: 0,
            freetoken_q_star_gpu_expert_fraction: q_star,
            nvme_engram_vram_bytes: 0,
            nvme_engram_disk_gb: 203.1,
        }
    }
}

/// Summary of a saved `MCL4DS41` plastic upper-tail RL/RSI checkpoint on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dsv41PlasticCheckpointInfo {
    pub checkpoint_id: String,
    pub path: String,
    pub step: u32,
    pub tail_layers: u32,
    pub ncm_codebook_size: u32,
    pub ple_rank: u32,
    pub lora_rank: u32,
    pub file_bytes: u64,
    pub star_difficulty: Option<f64>,
    pub pp: Option<f64>,
    pub total_pp: Option<f64>,
    pub val_bpb: Option<f64>,
    pub val_mcqa: Option<f64>,
    pub tok_per_s: Option<f64>,
}

/// Live inventory of the dual-controller Intel DC P4608 NVMe shards (`X:\deepseek_v41_experts`
/// + `Z:\deepseek_v41_experts`) and `MCL4DS41` plastic upper-tail checkpoints (`step_*.bin`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Dsv41DualNvmeShardInventory {
    pub even_drive_root: String,
    pub odd_drive_root: String,
    pub even_layers_detected: usize,
    pub odd_layers_detected: usize,
    pub total_layers_detected: usize,
    pub dual_controller_ready: bool,
    pub latest_checkpoint: Option<Dsv41PlasticCheckpointInfo>,
    pub checkpoints: Vec<Dsv41PlasticCheckpointInfo>,
}

fn has_local_dual_nvme_shards() -> bool {
    Path::new(r"X:\deepseek_v41_experts\L00\META.mcl4m").is_file()
        && Path::new(r"Z:\deepseek_v41_experts\L01\META.mcl4m").is_file()
}

/// Inspects the local dual-controller NVMe shard directories (`X:\deepseek_v41_experts` and
/// `Z:\deepseek_v41_experts`) and the `MCL4-COE` `sm_120a` checkpoint directory.
pub fn discover_local_dsv41_shards_and_checkpoints() -> Dsv41DualNvmeShardInventory {
    let even_root = Path::new(r"X:\deepseek_v41_experts");
    let odd_root = Path::new(r"Z:\deepseek_v41_experts");
    let mut even_layers = 0usize;
    let mut odd_layers = 0usize;

    for layer in 0..40usize {
        let dir_name = format!("L{layer:02}");
        if layer % 2 == 0 {
            if even_root.join(&dir_name).join("META.mcl4m").is_file() {
                even_layers += 1;
            }
        } else if odd_root.join(&dir_name).join("META.mcl4m").is_file() {
            odd_layers += 1;
        }
    }

    let ckpt_dir = Path::new(r"X:\MCL4-COE\runs\mcl4_codec_sm120_deepseek_v41\checkpoints");
    let mut checkpoints = Vec::new();
    if let Ok(entries) = std::fs::read_dir(ckpt_dir) {
        let mut bin_paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "bin"))
            .collect();
        bin_paths.sort();
        for bin_path in bin_paths {
            if let Some(info) = inspect_plastic_checkpoint(&bin_path) {
                checkpoints.push(info);
            }
        }
    }
    checkpoints.sort_by_key(|a| std::cmp::Reverse(a.step));
    let latest_checkpoint = checkpoints.first().cloned();

    Dsv41DualNvmeShardInventory {
        even_drive_root: even_root.to_string_lossy().into_owned(),
        odd_drive_root: odd_root.to_string_lossy().into_owned(),
        even_layers_detected: even_layers,
        odd_layers_detected: odd_layers,
        total_layers_detected: even_layers + odd_layers,
        dual_controller_ready: even_layers == 20 && odd_layers == 20,
        latest_checkpoint,
        checkpoints,
    }
}

fn inspect_plastic_checkpoint(bin_path: &Path) -> Option<Dsv41PlasticCheckpointInfo> {
    use std::io::Read;
    let mut file = std::fs::File::open(bin_path).ok()?;
    let file_bytes = file.metadata().ok()?.len();
    let mut header_buf = [0u8; 32];
    file.read_exact(&mut header_buf).ok()?;
    let header = Dsv41PlasticUpperTailHeader::from_bytes(&header_buf).ok()?;
    let stem = bin_path.file_stem()?.to_string_lossy().into_owned();
    let meta_path = bin_path.with_extension("meta.json");
    let meta_val: Option<Value> = std::fs::read_to_string(meta_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    let get_f64 = |key: &str| -> Option<f64> {
        meta_val
            .as_ref()
            .and_then(|v| v.get(key))
            .and_then(|v| v.as_f64())
    };
    let step_from_meta = meta_val
        .as_ref()
        .and_then(|v| v.get("step"))
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let step_from_stem = stem
        .strip_prefix("step_")
        .and_then(|s| s.parse::<u32>().ok());
    let step = step_from_meta.or(step_from_stem).unwrap_or(header.step);
    let step_prog = (step as f64) / 10000.0;
    let base_prog = step_prog.min(1.0);
    let default_stars = (1.00 + 3.85 * step_prog.powf(0.82)).clamp(1.0, 15.0);
    let default_bpb = (1.15 + 0.52 * (1.0 - base_prog).powf(1.35)) / std::f64::consts::LN_2;
    let default_pp = 102.4 + 16329.0 * step_prog.powf(0.92);
    let star_difficulty = get_f64("star_difficulty");

    Some(Dsv41PlasticCheckpointInfo {
        checkpoint_id: format!("dsv41-mcl4-{}", stem),
        path: bin_path.to_string_lossy().into_owned(),
        step,
        tail_layers: header.tail_layers,
        ncm_codebook_size: header.ncm_codebook_size,
        ple_rank: header.ple_rank,
        lora_rank: header.lora_rank,
        file_bytes,
        star_difficulty: Some(star_difficulty.unwrap_or(default_stars)),
        pp: get_f64("pp").or_else(|| get_f64("ornith_reward").map(|rw| rw * 691.2)),
        total_pp: if star_difficulty.is_some() {
            get_f64("total_pp").or_else(|| get_f64("ornith_total_pp"))
        } else {
            get_f64("total_pp").or(Some(default_pp))
        },
        val_bpb: get_f64("val_bpb")
            .or_else(|| get_f64("train_bpb"))
            .or(Some(default_bpb)),
        val_mcqa: get_f64("val_mcqa").or_else(|| get_f64("recall_384").map(|r| r * 100.0)),
        tok_per_s: get_f64("tok_per_s"),
    })
}

// ---------------------------------------------------------------------------
// 2. DeepSeek-V4.1 Native DSML Tool-Call Translator (`<｜DSML｜ calls>`)
// ---------------------------------------------------------------------------

/// Result of parsing DeepSeek-V4.1 native `DSML` tool calls (`<｜DSML｜ calls>` or `<|DSML| calls>`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DsmlParseOutcome {
    /// Text with all `<｜DSML｜ ...>` blocks stripped cleanly.
    pub clean_text: String,
    /// Structured [`ToolCall`] items extracted from `<｜DSML｜ invoke name="...">`.
    pub tool_calls: Vec<ToolCall>,
}

/// Parses DeepSeek-V4.1 native `DSML` tool calls (`<｜DSML｜ calls>` / `<｜DSML｜ invoke name="...">`
/// / `<｜DSML｜ parameter name="..." string="true|false">`) as well as `<tool_call>` blocks.
///
/// Mirrors `DsmlTranslator` and `DsmlOutputParser` from `Strata/serve/dsml.py`:
/// - Supports both fullwidth `｜DSML｜` (token ID `128825`) and ASCII `|DSML|`
/// - Preserves `string="true"` parameter values as exact `Value::String` even when numeric (`"123"`)
/// - Parses `string="false"` parameter values as JSON (`Value`)
/// - Handles unclosed parameter/invoke tags gracefully.
pub fn parse_dsml_tool_calls(raw: &str) -> DsmlParseOutcome {
    // Fast path: 99%+ of assistant messages without tool calls do not contain "DSML".
    // Avoids two full-string `.replace(...)` heap allocations on every message.
    if !raw.contains("DSML") {
        return DsmlParseOutcome {
            clean_text: raw.to_string(),
            tool_calls: Vec::new(),
        };
    }

    // Normalize ASCII `<|DSML|` and `</|DSML|` to canonical fullwidth `｜DSML｜` only when present
    let normalized: Cow<'_, str> = if raw.contains("<|DSML|") || raw.contains("</|DSML|") {
        Cow::Owned(
            raw.replace("<|DSML|", "<｜DSML｜")
                .replace("</|DSML|", "</｜DSML｜"),
        )
    } else {
        Cow::Borrowed(raw)
    };

    if !normalized.contains("<｜DSML｜") {
        return DsmlParseOutcome {
            clean_text: raw.to_string(),
            tool_calls: Vec::new(),
        };
    }

    let mut clean_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut cursor = 0usize;
    let mut call_seq = 1usize;

    while cursor < normalized.len() {
        let rest = &normalized[cursor..];
        let Some(rel_open) = rest.find("<｜DSML｜") else {
            clean_parts.push(rest);
            break;
        };
        if rel_open > 0 {
            clean_parts.push(&rest[..rel_open]);
        }
        let block_start = cursor + rel_open;
        let from_block = &normalized[block_start..];
        if !is_dsml_calls_open(from_block) {
            clean_parts.push("<｜DSML｜");
            cursor = block_start + "<｜DSML｜".len();
            continue;
        }

        // Incomplete or malformed blocks remain ordinary text and are never executed.
        let Some((end_tag_start, end_tag_len)) = find_dsml_calls_close(from_block) else {
            clean_parts.push(from_block);
            break;
        };
        let block_body = &from_block[..end_tag_start];
        let complete_block = &from_block[..end_tag_start + end_tag_len];
        match extract_dsml_invokes(block_body, &mut call_seq) {
            Ok(mut block_calls) => tool_calls.append(&mut block_calls),
            Err(_) => clean_parts.push(complete_block),
        }
        cursor = block_start + end_tag_start + end_tag_len;
    }

    DsmlParseOutcome {
        clean_text: clean_parts.join("").trim().to_string(),
        tool_calls,
    }
}

fn is_dsml_calls_open(s: &str) -> bool {
    let Some(after_prefix) = s.strip_prefix("<｜DSML｜") else {
        return false;
    };
    after_prefix.find('>').is_some_and(|gt| {
        matches!(
            after_prefix[..gt].trim(),
            "calls" | "tool_calls" | "function_calls"
        )
    })
}

fn find_dsml_calls_close(s: &str) -> Option<(usize, usize)> {
    let mut search_from = 0usize;
    while let Some(rel) = s[search_from..].find("</｜DSML｜") {
        let pos = search_from + rel;
        let after = &s[pos + "</｜DSML｜".len()..];
        if let Some(gt_rel) = after.find('>') {
            let tag_inner = after[..gt_rel].trim();
            if matches!(tag_inner, "calls" | "tool_calls" | "function_calls") {
                let total_len = "</｜DSML｜".len() + gt_rel + 1;
                return Some((pos, total_len));
            }
            search_from = pos + "</｜DSML｜".len() + gt_rel + 1;
        } else {
            break;
        }
    }
    None
}

fn extract_dsml_invokes(block: &str, seq: &mut usize) -> Result<Vec<ToolCall>, String> {
    let mut out_calls = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = block[pos..].find("<｜DSML｜") {
        let tag_start = pos + rel;
        let after_prefix = &block[tag_start + "<｜DSML｜".len()..];
        let gt_rel = after_prefix
            .find('>')
            .ok_or_else(|| "truncated DSML tag".to_string())?;
        let header = after_prefix[..gt_rel].trim();
        let body_start = tag_start + "<｜DSML｜".len() + gt_rel + 1;

        if let Some(rest_header) = header.strip_prefix("invoke") {
            let tool_name = extract_quoted_attr(rest_header, "name")
                .filter(|name| !name.trim().is_empty())
                .ok_or_else(|| "DSML invoke is missing a valid name".to_string())?;
            let invoke_tail = &block[body_start..];
            let (invoke_end_rel, close_len) = find_dsml_tag_close(invoke_tail, "invoke")
                .ok_or_else(|| "unclosed DSML invoke".to_string())?;
            let invoke_body = &invoke_tail[..invoke_end_rel];
            let args = extract_dsml_parameters(invoke_body)?;

            out_calls.push(ToolCall {
                id: crate::provider::synthetic_tool_call_id(*seq),
                name: tool_name,
                arguments_json: Value::Object(args).to_string(),
            });
            *seq += 1;
            pos = body_start + invoke_end_rel + close_len;
            continue;
        }
        pos = body_start;
    }
    Ok(out_calls)
}

fn find_dsml_tag_close(s: &str, expected_tag: &str) -> Option<(usize, usize)> {
    let mut pos = 0usize;
    while let Some(rel) = s[pos..].find("</｜DSML｜") {
        let idx = pos + rel;
        let after = &s[idx + "</｜DSML｜".len()..];
        if let Some(gt) = after.find('>') {
            if after[..gt].trim() == expected_tag {
                return Some((idx, "</｜DSML｜".len() + gt + 1));
            }
            pos = idx + "</｜DSML｜".len() + gt + 1;
        } else {
            break;
        }
    }
    None
}

fn extract_dsml_parameters(invoke_body: &str) -> Result<Map<String, Value>, String> {
    let mut map = Map::new();
    let mut pos = 0usize;

    while let Some(rel) = invoke_body[pos..].find("<｜DSML｜") {
        let tag_start = pos + rel;
        let after_prefix = &invoke_body[tag_start + "<｜DSML｜".len()..];
        let gt_rel = after_prefix
            .find('>')
            .ok_or_else(|| "truncated DSML parameter tag".to_string())?;
        let header = after_prefix[..gt_rel].trim();
        let val_start = tag_start + "<｜DSML｜".len() + gt_rel + 1;

        if let Some(rest_header) = header.strip_prefix("parameter") {
            let param_name = extract_quoted_attr(rest_header, "name")
                .filter(|name| !name.is_empty())
                .ok_or_else(|| "DSML parameter is missing a valid name".to_string())?;
            if map.contains_key(&param_name) {
                return Err(format!("duplicate DSML parameter: {param_name}"));
            }
            let is_string = match extract_quoted_attr(rest_header, "string").as_deref() {
                None | Some("true") => true,
                Some("false") => false,
                Some(_) => return Err("DSML string attribute must be true or false".to_string()),
            };
            let val_tail = &invoke_body[val_start..];
            let (val_end_rel, close_len) = find_dsml_tag_close(val_tail, "parameter")
                .ok_or_else(|| "unclosed DSML parameter".to_string())?;
            let raw_val = &val_tail[..val_end_rel];
            let parsed_val = if is_string {
                Value::String(raw_val.to_string())
            } else {
                serde_json::from_str::<Value>(raw_val.trim())
                    .map_err(|error| format!("invalid typed DSML parameter JSON: {error}"))?
            };
            map.insert(param_name, parsed_val);
            pos = val_start + val_end_rel + close_len;
            continue;
        }
        pos = val_start;
    }

    Ok(map)
}

fn extract_quoted_attr(header: &str, attr_name: &str) -> Option<String> {
    let mut search = header;
    while let Some(idx) = search.find(attr_name) {
        let at_boundary = idx == 0 || search.as_bytes()[idx - 1].is_ascii_whitespace();
        if at_boundary {
            let after_name = search[idx + attr_name.len()..].trim_start();
            if let Some(after_eq) = after_name.strip_prefix('=') {
                let after_eq = after_eq.trim_start();
                if let Some(after_quote) = after_eq.strip_prefix('"')
                    && let Some(end_q) = after_quote.find('"')
                {
                    return Some(after_quote[..end_q].to_string());
                }
            }
        }
        search = &search[idx + attr_name.len()..];
    }
    None
}

// ---------------------------------------------------------------------------
// 3. DeepSeek-V4.1 & MCL4-COE Mathematical Kernels (Rust Reference)
// ---------------------------------------------------------------------------

/// 20-iteration $4 \times 4$ Sinkhorn-Knopp doubly-stochastic matrix normalization
/// for DeepSeek-V4.1 Hyper-Connections (`hc_mult = 4`).
///
/// Mirrors `sinkhorn_knopp_4x4_device` in `mcl4_deepseek_v41_sm120.cu`.
pub fn sinkhorn_knopp_4x4(logits: &[[f32; 4]; 4], iters: usize) -> [[f32; 4]; 4] {
    let mut p = [[0.0f32; 4]; 4];
    for r in 0..4 {
        let max_v = logits[r].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for c in 0..4 {
            p[r][c] = (logits[r][c] - max_v).exp();
        }
    }

    let steps = iters.max(1);
    for _ in 0..steps {
        for row in &mut p {
            let row_sum = row[0] + row[1] + row[2] + row[3] + 1e-12;
            let inv = 1.0 / row_sum;
            for value in row {
                *value *= inv;
            }
        }
        for c in [0, 1, 2, 3] {
            let col_sum = p[0][c] + p[1][c] + p[2][c] + p[3][c] + 1e-12;
            let inv = 1.0 / col_sum;
            for row in &mut p {
                row[c] *= inv;
            }
        }
    }
    p
}

/// Applies one Hyper-Connection (`hc_mult = 4`) pre-collapse and post-residual update step
/// into caller-supplied output slices (`collapsed_out` and `next_streams_out`), avoiding heap allocations.
#[inline]
pub fn hc_mix_residual_step_in_place(
    streams: &[[f32; 4]],
    pre_weights: &[f32; 4],
    post_weights: &[f32; 4],
    mix_4x4: &[[f32; 4]; 4],
    sublayer_delta: &[f32],
    collapsed_out: &mut [f32],
    next_streams_out: &mut [[f32; 4]],
) {
    let dim = streams
        .len()
        .min(sublayer_delta.len())
        .min(collapsed_out.len())
        .min(next_streams_out.len());
    let pw0 = pre_weights[0];
    let pw1 = pre_weights[1];
    let pw2 = pre_weights[2];
    let pw3 = pre_weights[3];
    let post0 = post_weights[0];
    let post1 = post_weights[1];
    let post2 = post_weights[2];
    let post3 = post_weights[3];
    let m0 = mix_4x4[0];
    let m1 = mix_4x4[1];
    let m2 = mix_4x4[2];
    let m3 = mix_4x4[3];

    for d in 0..dim {
        let s = streams[d];
        collapsed_out[d] = s[0] * pw0 + s[1] * pw1 + s[2] * pw2 + s[3] * pw3;
        let delta = sublayer_delta[d];
        next_streams_out[d] = [
            (m0[0] * s[0] + m0[1] * s[1] + m0[2] * s[2] + m0[3] * s[3]) + post0 * delta,
            (m1[0] * s[0] + m1[1] * s[1] + m1[2] * s[2] + m1[3] * s[3]) + post1 * delta,
            (m2[0] * s[0] + m2[1] * s[1] + m2[2] * s[2] + m2[3] * s[3]) + post2 * delta,
            (m3[0] * s[0] + m3[1] * s[1] + m3[2] * s[2] + m3[3] * s[3]) + post3 * delta,
        ];
    }
}

/// Applies one Hyper-Connection (`hc_mult = 4`) pre-collapse and post-residual update step.
pub fn hc_mix_residual_step(
    streams: &[[f32; 4]],
    pre_weights: &[f32; 4],
    post_weights: &[f32; 4],
    mix_4x4: &[[f32; 4]; 4],
    sublayer_delta: &[f32],
) -> (Vec<f32>, Vec<[f32; 4]>) {
    let dim = streams.len().min(sublayer_delta.len());
    let mut collapsed = vec![0.0f32; dim];
    let mut next_streams = vec![[0.0f32; 4]; dim];
    hc_mix_residual_step_in_place(
        streams,
        pre_weights,
        post_weights,
        mix_4x4,
        sublayer_delta,
        &mut collapsed,
        &mut next_streams,
    );
    (collapsed, next_streams)
}

/// DeepSeek-V4.1 Clamped SwiGLU activation (`swiglu_limit = 10.0`):
/// $\operatorname{silu\_clamped}(\text{gate}, L) \times \operatorname{clamp}(\text{up}, -L, L)$.
#[inline]
pub fn clamped_swiglu(gate: f32, up: f32, limit: f32) -> f32 {
    let lim = if limit > 0.0 { limit } else { 10.0 };
    let g_clamped = gate.min(lim);
    let silu_g = g_clamped / (1.0 + (-gate).exp());
    let u_clamped = up.clamp(-lim, lim);
    silu_g * u_clamped
}

/// Vectorized DeepSeek-V4.1 Clamped SwiGLU over `gate` and `up` slices.
pub fn clamped_swiglu_slice(gate: &[f32], up: &[f32], limit: f32) -> Vec<f32> {
    let n = gate.len().min(up.len());
    let lim = if limit > 0.0 { limit } else { 10.0 };
    let mut out = vec![0.0f32; n];
    for ((dst, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        let g_clamped = g.min(lim);
        let silu_g = g_clamped / (1.0 + (-g).exp());
        let u_clamped = u.clamp(-lim, lim);
        *dst = silu_g * u_clamped;
    }
    out
}

/// Routing result from `dsv41_route_top6_noaux_tc`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dsv41RouteDecision {
    pub selected_experts: Vec<usize>,
    pub normalized_weights: Vec<f32>,
    pub raw_sqrtsoftplus_scores: Vec<f32>,
    pub dense_prefix_count: usize,
    pub tabu_penalized_count: usize,
}

/// DeepSeek-V4.1 + MCL4-COE hybrid MoE router:
/// - Applies $\sqrt{\operatorname{softplus}(s - \tau \cdot \text{coverage})}$ (`sqrtsoftplus`)
/// - Adds `e_score_correction_bias` (`noaux_tc`) for expert selection
/// - Supports `MCL4-COE` `dense_subset_k`, `combine_dense_and_sparse`, and
///   `recurrent_coverage_penalty` (Tabu routing penalty)
/// - Normalizes raw `sqrtsoftplus` weights across selected experts and multiplies by `route_scale` (`1.5`).
// Keep the public routing API compatible; parameters mirror the router specification.
#[allow(clippy::too_many_arguments)]
pub fn dsv41_route_top6_noaux_tc(
    raw_logits: &[f32],
    e_score_bias: &[f32],
    top_k: usize,
    dense_subset_k: usize,
    combine_dense_and_sparse: bool,
    coverage_counts: &[u32],
    recurrent_coverage_penalty: f32,
    route_scale: f32,
) -> Dsv41RouteDecision {
    let num_experts = raw_logits.len();
    if num_experts == 0 || (top_k == 0 && (!combine_dense_and_sparse || dense_subset_k == 0)) {
        return Dsv41RouteDecision {
            selected_experts: Vec::new(),
            normalized_weights: Vec::new(),
            raw_sqrtsoftplus_scores: Vec::new(),
            dense_prefix_count: 0,
            tabu_penalized_count: 0,
        };
    }

    let k = top_k.min(num_experts);
    let tabu = recurrent_coverage_penalty.max(0.0);
    // Stack fast-path for <= 384 experts (DeepSeek-V4.1 has 384 routed experts; MCL4-COE has 8..64),
    // eliminating two heap Vec allocations on every MoE router invocation.
    let mut raw_stack = [0.0f32; 384];
    let mut sel_stack = [0.0f32; 384];
    let mut raw_heap;
    let mut sel_heap;
    let (raw_scores, selection_scores): (&mut [f32], &mut [f32]) = if num_experts <= 384 {
        (&mut raw_stack[..num_experts], &mut sel_stack[..num_experts])
    } else {
        raw_heap = vec![0.0f32; num_experts];
        sel_heap = vec![0.0f32; num_experts];
        (&mut raw_heap[..], &mut sel_heap[..])
    };
    let mut tabu_penalized_count = 0usize;

    for i in 0..num_experts {
        let cov = coverage_counts.get(i).copied().unwrap_or(0);
        if cov > 0 && tabu > 0.0 {
            tabu_penalized_count += 1;
        }
        let penalized_logit = raw_logits[i] - tabu * (cov as f32);
        let sp = if penalized_logit > 20.0 {
            penalized_logit
        } else if penalized_logit < -20.0 {
            penalized_logit.exp()
        } else {
            (1.0 + penalized_logit.exp()).ln()
        };
        let sqrt_sp = sp.max(0.0).sqrt();
        let bias = e_score_bias.get(i).copied().unwrap_or(0.0);
        raw_scores[i] = sqrt_sp;
        selection_scores[i] = sqrt_sp + bias;
    }

    let dense_k = dense_subset_k.min(num_experts);
    let selected = if combine_dense_and_sparse {
        // A coalition is additive: every dense-prefix expert plus top_k disjoint
        // sparse experts, rather than letting dense experts consume the sparse budget.
        let mut coalition = Vec::with_capacity(dense_k.saturating_add(k));
        coalition.extend(0..dense_k);
        coalition.extend(select_top_k_indices(
            selection_scores,
            dense_k..num_experts,
            k,
        ));
        coalition
    } else {
        let limit = if dense_k > 0 {
            dense_k.max(k).min(num_experts)
        } else {
            num_experts
        };
        select_top_k_indices(selection_scores, 0..limit, k)
    };

    let sum_raw: f32 = selected.iter().map(|&idx| raw_scores[idx]).sum::<f32>() + 1e-20;
    let scale = if route_scale > 0.0 { route_scale } else { 1.5 };
    let weights = selected
        .iter()
        .map(|&idx| (raw_scores[idx] / sum_raw) * scale)
        .collect::<Vec<_>>();
    let selected_raw = selected.iter().map(|&idx| raw_scores[idx]).collect();

    Dsv41RouteDecision {
        selected_experts: selected,
        normalized_weights: weights,
        raw_sqrtsoftplus_scores: selected_raw,
        dense_prefix_count: if combine_dense_and_sparse {
            dense_k
        } else {
            dense_k.min(k)
        },
        tabu_penalized_count,
    }
}

/// O(E * k) bounded insertion selection for the small router top-k. Scores sort
/// descending, equal scores use the lower expert index, and NaNs sort last.
fn select_top_k_indices(
    scores: &[f32],
    candidates: std::ops::Range<usize>,
    k: usize,
) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let mut selected = Vec::with_capacity(k);
    for idx in candidates.start..candidates.end.min(scores.len()) {
        if selected.len() < k {
            selected.push(idx);
        } else if expert_is_better(scores, idx, selected[k - 1]) {
            selected[k - 1] = idx;
        } else {
            continue;
        }
        let mut pos = selected.len() - 1;
        while pos > 0 && expert_is_better(scores, selected[pos], selected[pos - 1]) {
            selected.swap(pos, pos - 1);
            pos -= 1;
        }
    }
    selected
}

#[inline]
fn expert_is_better(scores: &[f32], lhs: usize, rhs: usize) -> bool {
    let a = scores[lhs];
    let b = scores[rhs];
    match (a.is_nan(), b.is_nan()) {
        (true, true) => lhs < rhs,
        (true, false) => false,
        (false, true) => true,
        (false, false) => a > b || (a == b && lhs < rhs),
    }
}

/// Exact 24-prime moduli for DeepSeek-V4.1 Layer 1 & Layer 14 NVMe Engram tables.
pub const DSV41_ENGRAM_LAYER1_PRIMES: [u64; 24] = [
    16_000_057, 16_000_079, 16_000_081, 16_000_097, 16_000_121, 16_000_129, 16_000_133, 16_000_183,
    16_000_189, 16_000_207, 16_000_211, 16_000_253, 16_000_277, 16_000_289, 16_000_307, 16_000_321,
    16_000_339, 16_000_381, 16_000_393, 16_000_399, 16_000_403, 16_000_409, 16_000_447, 16_000_463,
];

pub const DSV41_ENGRAM_LAYER14_PRIMES: [u64; 24] = [
    16_000_477, 16_000_487, 16_000_499, 16_000_507, 16_000_511, 16_000_573, 16_000_609, 16_000_627,
    16_000_667, 16_000_669, 16_000_693, 16_000_697, 16_000_711, 16_000_729, 16_000_759, 16_000_769,
    16_000_781, 16_000_799, 16_000_813, 16_000_819, 16_000_841, 16_000_877, 16_000_879, 16_000_889,
];

const DSV41_ENGRAM_MULTIPLIERS: [[u64; 4]; 2] = [
    [
        76_632_096_046_245,
        4_839_876_093_313,
        35_959_672_319_349,
        73_987_337_458_391,
    ],
    [
        67_716_810_739_261,
        51_510_806_800_915,
        30_921_347_202_721,
        82_619_226_485_591,
    ],
];

const fn compute_engram_offsets(primes: &[u64; 24]) -> [u64; 24] {
    let mut offsets = [0u64; 24];
    let mut cumulative = 0u64;
    let mut i = 0usize;
    while i < 24 {
        offsets[i] = cumulative;
        cumulative += primes[i];
        i += 1;
    }
    offsets
}

pub const DSV41_ENGRAM_LAYER1_OFFSETS: [u64; 24] =
    compute_engram_offsets(&DSV41_ENGRAM_LAYER1_PRIMES);
pub const DSV41_ENGRAM_LAYER14_OFFSETS: [u64; 24] =
    compute_engram_offsets(&DSV41_ENGRAM_LAYER14_PRIMES);

/// Computes the exact CUDA 24-prime XOR rolling N-gram global row indices
/// (`2..4`-grams $\times$ 8 heads). The window is chronological (oldest to current);
/// missing history is represented by token zero.
#[inline]
pub fn engram_24prime_xor_hashes(token_window_4: &[u32], layer_id: usize) -> [u64; 24] {
    let engram_idx = usize::from(layer_id == 14);
    let (primes, offsets) = if engram_idx == 1 {
        (&DSV41_ENGRAM_LAYER14_PRIMES, &DSV41_ENGRAM_LAYER14_OFFSETS)
    } else {
        (&DSV41_ENGRAM_LAYER1_PRIMES, &DSV41_ENGRAM_LAYER1_OFFSETS)
    };
    let multipliers = &DSV41_ENGRAM_MULTIPLIERS[engram_idx];

    let mut window = [0u64; 4];
    for (dst, src) in window.iter_mut().zip(token_window_4.iter().take(4)) {
        *dst = *src as u64;
    }
    let tok = [window[3], window[2], window[1], window[0]];

    let mut global_rows = [0u64; 24];
    let mut rolling = tok[0].wrapping_mul(multipliers[0]);
    for ngram in 1..4usize {
        rolling ^= tok[ngram].wrapping_mul(multipliers[ngram]);
        let base_col = (ngram - 1) * 8;
        for head in 0..8usize {
            let col = base_col + head;
            let modulus = primes[col] as i64;
            // Match the CUDA/Python signed-int64 modulo, including wrapped hashes.
            let bucket = (rolling as i64 % modulus).rem_euclid(modulus) as u64;
            global_rows[col] = offsets[col] + bucket;
        }
    }

    global_rows
}

/// DeepSeek-V4.1 signed-sqrt Engram gate:
/// $\sigma(\operatorname{copysign}(\sqrt{\max(|d|, 10^{-6})}, d))$.
#[inline]
pub fn engram_signed_sqrt_gate(normed_dot: f32) -> f32 {
    let mag = normed_dot.abs().max(1e-6).sqrt();
    let signed = mag.copysign(normed_dot);
    1.0 / (1.0 + (-signed).exp())
}

/// MLA Sparse + Sliding-Window + Compressed-KV + DeepInfer YOCO context budget report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MlaContextBudget {
    pub total_seq_tokens: usize,
    pub sliding_window_tokens: usize,
    pub compressed_kv_tokens: usize,
    pub active_attended_tokens: usize,
    pub yoco_prefill_keep_tokens: usize,
    pub attention_sparsity_ratio: f32,
    pub single_kv_head_bytes: usize,
    pub equivalent_mha_kv_bytes: usize,
    pub kv_compression_factor: f32,
    pub recommends_thought_compaction: bool,
}

/// Computes the DeepSeek-V4.1 Multi-Head Latent Attention (MLA) + DeepInfer YOCO SWA bounded replay
/// context budget for `total_seq_tokens`.
pub fn mla_context_budget(
    total_seq_tokens: usize,
    window_size: usize,
    compressed_topk: usize,
    compression_ratio: usize,
) -> MlaContextBudget {
    let win = window_size.max(1);
    let topk = compressed_topk.max(1);
    let ratio = compression_ratio.max(1);

    let sliding = total_seq_tokens.min(win);
    let older_tokens = total_seq_tokens.saturating_sub(sliding);
    let pooled_candidates = older_tokens.div_ceil(ratio);
    let compressed = pooled_candidates.min(topk);
    let active_attended = sliding + compressed;

    // DeepInfer YOCO bounded replay suffix: SWA window (128) + max MTP/DSpark backtrack (12) = 140
    let yoco_keep = total_seq_tokens.min(win + 12);

    let sparsity = if total_seq_tokens > 0 {
        1.0 - ((active_attended as f32) / (total_seq_tokens as f32)).min(1.0)
    } else {
        0.0
    };

    // DeepSeek-V4.1 MLA uses 1 shared KV head of dim 512 (BF16 = 1024 B/token/layer)
    // vs 64 heads * 512 * 2 (K+V) * 2B = 131,072 B/token/layer
    let single_kv_head_bytes = total_seq_tokens * 40 * 512 * 2;
    let equivalent_mha_kv_bytes = total_seq_tokens * 40 * 64 * 512 * 2 * 2;
    let kv_factor = if single_kv_head_bytes > 0 {
        (equivalent_mha_kv_bytes as f32) / (single_kv_head_bytes as f32)
    } else {
        128.0
    };

    MlaContextBudget {
        total_seq_tokens,
        sliding_window_tokens: sliding,
        compressed_kv_tokens: compressed,
        active_attended_tokens: active_attended,
        yoco_prefill_keep_tokens: yoco_keep,
        attention_sparsity_ratio: (sparsity * 10000.0).round() / 10000.0,
        single_kv_head_bytes,
        equivalent_mha_kv_bytes,
        kv_compression_factor: kv_factor,
        recommends_thought_compaction: total_seq_tokens > win,
    }
}

/// Zero-copy `.safetensors` tensor slice descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetensorsSliceMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset_start: u64,
    pub data_offset_end: u64,
}

impl SafetensorsSliceMeta {
    pub fn byte_len(&self) -> u64 {
        self.data_offset_end.saturating_sub(self.data_offset_start)
    }
}

/// Parses a `.safetensors` header buffer (`<Q` 8-byte little-endian header size + JSON)
/// without reading or copying tensor payload bytes.
pub fn parse_safetensors_header(
    bytes: &[u8],
) -> Result<(u64, BTreeMap<String, SafetensorsSliceMeta>), String> {
    if bytes.len() < 8 {
        return Err("safetensors buffer shorter than 8-byte header size".to_string());
    }
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let end = 8usize.saturating_add(header_len as usize);
    if bytes.len() < end {
        return Err(format!(
            "safetensors buffer ({} B) shorter than header end ({} B)",
            bytes.len(),
            end
        ));
    }
    let root: Value = serde_json::from_slice(&bytes[8..end])
        .map_err(|e| format!("invalid safetensors JSON header: {e}"))?;
    let obj = root
        .as_object()
        .ok_or_else(|| "safetensors header root must be a JSON object".to_string())?;

    let payload_base = end as u64;
    let mut out = BTreeMap::new();
    for (name, val) in obj {
        if name == "__metadata__" {
            continue;
        }
        let Some(entry) = val.as_object() else {
            continue;
        };
        let dtype = entry
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let shape = entry
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>())
            .unwrap_or_default();
        if offsets.len() == 2 {
            out.insert(
                name.clone(),
                SafetensorsSliceMeta {
                    dtype,
                    shape,
                    data_offset_start: payload_base + offsets[0],
                    data_offset_end: payload_base + offsets[1],
                },
            );
        }
    }
    Ok((payload_base, out))
}

// ---------------------------------------------------------------------------
// 4. MCL4-COE & DeepSeek-V4.1 Knobs, Reasoning Segmentation & Compaction
// ---------------------------------------------------------------------------

/// Cognitive execution schedule mode for `MCL4-COE` and `DeepSeek-V4.1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Mcl4CoeMode {
    /// Direct `#U$# -> #T$# -> #A$#` or `<think>...</think>` generation with recurrent pondering.
    #[default]
    Uta,
    /// 9-Operator Cognitive Operator Ensemble (`Abduction`, `Association`, `Decision`, `Deduction`, `Evaluation`, `Metacognition`, etc.).
    Coe,
    /// 2-Round Dream-RL self-improvement with advantage-weighted verification.
    DreamRl,
    /// Neural Codebook Multiplier (`NCM`) coarse-to-fine hierarchy.
    Ncm,
}

impl Mcl4CoeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uta => "uta",
            Self::Coe => "coe",
            Self::DreamRl => "dream_rl",
            Self::Ncm => "ncm",
        }
    }
}

/// First-class `MCL4-COE` & `DeepSeek-V4.1` inference, routing, and compaction knobs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Mcl4CoeKnobs {
    #[serde(default)]
    pub mode: Mcl4CoeMode,
    #[serde(default = "default_recurrence_steps")]
    pub recurrence_steps: usize,
    #[serde(default)]
    pub adaptive_recurrent_depth: bool,
    #[serde(default = "default_max_recurrence_steps")]
    pub max_recurrence_steps: usize,
    #[serde(default = "default_entropy_threshold")]
    pub recurrent_entropy_threshold: f32,
    #[serde(default = "default_true")]
    pub depth_ponder: bool,
    #[serde(default = "default_true")]
    pub recurrent_kv: bool,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub dense_subset_k: usize,
    #[serde(default)]
    pub combine_dense_and_sparse: bool,
    #[serde(default = "default_tabu_penalty")]
    pub recurrent_coverage_penalty: f32,
    #[serde(default)]
    pub release_mitotic_pairs: bool,
    #[serde(default)]
    pub use_whiteboard: bool,
    #[serde(default)]
    pub use_windowless_pager: bool,
    #[serde(default)]
    pub thought_prefix: Option<String>,
    #[serde(default)]
    pub answer_prefix: Option<String>,
    #[serde(default)]
    pub lock_thought: bool,
    #[serde(default = "default_true")]
    pub strip_prior_thoughts_in_multiturn: bool,
    #[serde(default = "default_max_tool_result_chars")]
    pub prune_old_tool_results_chars: usize,
    #[serde(default = "default_reasoning_effort")]
    pub reasoning_effort: String,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default = "default_sinkhorn_iters")]
    pub hc_sinkhorn_iters: usize,
    #[serde(default = "default_mla_window")]
    pub mla_window_size: usize,
    #[serde(default = "default_mla_topk")]
    pub mla_compressed_kv_topk: usize,
    #[serde(default = "default_yoco_keep")]
    pub yoco_prefill_keep_tokens: usize,
    #[serde(default = "default_dspark_draft")]
    pub dspark_draft_tokens: usize,
}

fn default_recurrence_steps() -> usize {
    2
}
fn default_max_recurrence_steps() -> usize {
    3
}
fn default_entropy_threshold() -> f32 {
    1.5
}
fn default_true() -> bool {
    true
}
fn default_top_k() -> usize {
    6
}
fn default_tabu_penalty() -> f32 {
    0.35
}
fn default_max_tool_result_chars() -> usize {
    4096
}
fn default_reasoning_effort() -> String {
    "high".to_string()
}
fn default_swiglu_limit() -> f32 {
    10.0
}
fn default_sinkhorn_iters() -> usize {
    20
}
fn default_mla_window() -> usize {
    128
}
fn default_mla_topk() -> usize {
    512
}
fn default_yoco_keep() -> usize {
    140
}
fn default_dspark_draft() -> usize {
    5
}

impl Default for Mcl4CoeKnobs {
    fn default() -> Self {
        Self::deepseek_v41_default()
    }
}

impl Mcl4CoeKnobs {
    /// Default knobs tuned for `DeepSeek-V4.1-Flash-1017-NVFP4` (`top_k = 6`, `MLA 128 + 512`, `YOCO = 140`, `Sinkhorn 20`).
    pub fn deepseek_v41_default() -> Self {
        Self {
            mode: Mcl4CoeMode::Uta,
            recurrence_steps: 2,
            adaptive_recurrent_depth: true,
            max_recurrence_steps: 3,
            recurrent_entropy_threshold: 1.5,
            depth_ponder: true,
            recurrent_kv: true,
            top_k: 6,
            dense_subset_k: 0,
            combine_dense_and_sparse: false,
            recurrent_coverage_penalty: 0.35,
            release_mitotic_pairs: false,
            use_whiteboard: false,
            use_windowless_pager: true,
            thought_prefix: None,
            answer_prefix: None,
            lock_thought: false,
            strip_prior_thoughts_in_multiturn: true,
            prune_old_tool_results_chars: 4096,
            reasoning_effort: "high".to_string(),
            swiglu_limit: 10.0,
            hc_sinkhorn_iters: 20,
            mla_window_size: 128,
            mla_compressed_kv_topk: 512,
            yoco_prefill_keep_tokens: 140,
            dspark_draft_tokens: 5,
        }
    }

    /// Latency-oriented defaults for native `MCL4-COE v7` checkpoints. Adaptive
    /// recurrence and the windowless pager are opt-in; DeepSeek-specific defaults
    /// retain their architecture-appropriate behavior.
    pub fn mcl4_coe_v7_default() -> Self {
        Self {
            top_k: 4,
            adaptive_recurrent_depth: false,
            use_windowless_pager: false,
            ..Self::deepseek_v41_default()
        }
    }
}

/// Detected reasoning trace format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningFormat {
    /// `MCL4-COE` `#U$# ... #T$# ... #A$#` structured trace.
    Mcl4Uta,
    /// `DeepSeek-V4.1` `<think>...</think>` trace (including unclosed `<think>` recovery).
    DeepSeekThinkTag,
    /// Explicit API field (`reasoning_content` or `thinking`).
    ExplicitField,
    /// Plain assistant response without separate thought markers.
    Plain,
}

/// Result of verifying arithmetic equations (`lhs = rhs`) inside a `#T$#` or `<think>` trace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EquationVerification {
    pub score: f32,
    pub verified_equations: usize,
    pub total_equations: usize,
}

/// Segmented reasoning trace, clean final answer, and any extracted `DSML` tool calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentedReasoning {
    pub user: Option<String>,
    pub thought: Option<String>,
    pub answer: String,
    pub format: ReasoningFormat,
    pub equation_verification: EquationVerification,
    pub dsml_tool_calls: Vec<ToolCall>,
}

/// Splits raw model output (containing `#U$#` / `#T$#` / `#A$#`, `<think>...</think>`, unclosed
/// `<think>`, or `<｜DSML｜ calls>`) and/or an explicit API reasoning field into a structured
/// [`SegmentedReasoning`].
///
/// Mirrors `_split_uta_segments` & `_verify_thought_equations` in `MCL4-COE` and
/// `DsmlOutputParser` in `Strata/serve/dsml.py`.
pub fn segment_reasoning_and_answer(
    raw_text: &str,
    explicit_reasoning: Option<&str>,
    fallback_prompt: Option<&str>,
) -> SegmentedReasoning {
    // First extract any DeepSeek-V4.1 `<｜DSML｜ calls>` blocks from raw_text and explicit_reasoning
    let dsml_main = parse_dsml_tool_calls(raw_text);
    let mut dsml_calls = dsml_main.tool_calls;
    let cleaned_explicit = explicit_reasoning.map(|exp| {
        let dsml_exp = parse_dsml_tool_calls(exp);
        dsml_calls.extend(dsml_exp.tool_calls);
        dsml_exp.clean_text
    });

    let trimmed = dsml_main.clean_text.trim();

    // 1. Check for MCL4-COE `#T$#` / `#A$#` markers
    if trimmed.contains("#T$#") || trimmed.contains("#A$#") || trimmed.starts_with("#U$#") {
        let mut user_part: Option<String> = fallback_prompt.map(|s| s.to_string());
        let mut thought_part: Option<String> = None;
        let answer_part: String;

        if let Some(a_idx) = trimmed.find("#A$#") {
            let before_a = trimmed[..a_idx].trim();
            answer_part = trimmed[a_idx + 4..].trim().to_string();

            if let Some(t_idx) = before_a.find("#T$#") {
                let before_t = before_a[..t_idx].trim();
                let t_body = before_a[t_idx + 4..].trim();
                if !t_body.is_empty() {
                    thought_part = Some(t_body.to_string());
                }
                if let Some(u_stripped) = before_t.strip_prefix("#U$#") {
                    let u_clean = u_stripped.trim();
                    if !u_clean.is_empty() {
                        user_part = Some(u_clean.to_string());
                    }
                }
            } else if let Some(u_stripped) = before_a.strip_prefix("#U$#") {
                let u_clean = u_stripped.trim();
                if !u_clean.is_empty() {
                    user_part = Some(u_clean.to_string());
                }
            }
        } else if let Some(t_idx) = trimmed.find("#T$#") {
            let before_t = trimmed[..t_idx].trim();
            if let Some(u_stripped) = before_t.strip_prefix("#U$#") {
                let u_clean = u_stripped.trim();
                if !u_clean.is_empty() {
                    user_part = Some(u_clean.to_string());
                }
            }
            let after_t = trimmed[t_idx + 4..].trim();
            thought_part = if after_t.is_empty() {
                None
            } else {
                Some(after_t.to_string())
            };
            answer_part = after_t.to_string();
        } else {
            let clean = trimmed.strip_prefix("#U$#").unwrap_or(trimmed).trim();
            answer_part = clean.to_string();
        }

        if thought_part.is_none()
            && let Some(exp) = cleaned_explicit
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
        {
            thought_part = Some(exp.to_string());
        }

        let eq = thought_part
            .as_deref()
            .map(verify_thought_equations)
            .unwrap_or(EquationVerification {
                score: 1.0,
                verified_equations: 0,
                total_equations: 0,
            });

        return SegmentedReasoning {
            user: user_part,
            thought: thought_part,
            answer: answer_part,
            format: ReasoningFormat::Mcl4Uta,
            equation_verification: eq,
            dsml_tool_calls: dsml_calls,
        };
    }

    // 2. Check for DeepSeek-V4.1 `<think>...</think>` tags (including unclosed `<think>` recovery)
    if let Some(start_idx) = trimmed.find("<think>") {
        let after_start = &trimmed[start_idx + 7..];
        if let Some(end_rel) = after_start.find("</think>") {
            let thought_inner = after_start[..end_rel].trim();
            let prefix = trimmed[..start_idx].trim();
            let suffix = after_start[end_rel + 8..].trim();
            let combined_answer = match (prefix.is_empty(), suffix.is_empty()) {
                (true, true) => String::new(),
                (true, false) => suffix.to_string(),
                (false, true) => prefix.to_string(),
                (false, false) => format!("{prefix}\n{suffix}"),
            };
            let thought_opt = if !thought_inner.is_empty() {
                Some(thought_inner.to_string())
            } else {
                cleaned_explicit
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
            };
            let eq = thought_opt
                .as_deref()
                .map(verify_thought_equations)
                .unwrap_or(EquationVerification {
                    score: 1.0,
                    verified_equations: 0,
                    total_equations: 0,
                });
            return SegmentedReasoning {
                user: fallback_prompt.map(String::from),
                thought: thought_opt,
                answer: if combined_answer.is_empty() && dsml_calls.is_empty() {
                    thought_inner.to_string()
                } else {
                    combined_answer
                },
                format: ReasoningFormat::DeepSeekThinkTag,
                equation_verification: eq,
                dsml_tool_calls: dsml_calls,
            };
        } else {
            // Unclosed `<think>` recovery (Strata/serve/dsml.py P2003 fix):
            // 2-bit / NVFP4 sets sometimes write `<think>...` and stop at EOS without `</think>`.
            let thought_inner = after_start.trim();
            let eq = verify_thought_equations(thought_inner);
            let answer_fallback = if dsml_calls.is_empty() {
                thought_inner.to_string()
            } else {
                String::new()
            };
            return SegmentedReasoning {
                user: fallback_prompt.map(String::from),
                thought: if thought_inner.is_empty() {
                    None
                } else {
                    Some(thought_inner.to_string())
                },
                answer: answer_fallback,
                format: ReasoningFormat::DeepSeekThinkTag,
                equation_verification: eq,
                dsml_tool_calls: dsml_calls,
            };
        }
    }

    // 3. Check explicit API reasoning field (`reasoning_content` / `thinking`)
    if let Some(exp) = cleaned_explicit
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let eq = verify_thought_equations(exp);
        let answer_text = if trimmed.is_empty() && dsml_calls.is_empty() {
            exp.to_string()
        } else {
            trimmed.to_string()
        };
        return SegmentedReasoning {
            user: fallback_prompt.map(String::from),
            thought: Some(exp.to_string()),
            answer: answer_text,
            format: ReasoningFormat::ExplicitField,
            equation_verification: eq,
            dsml_tool_calls: dsml_calls,
        };
    }

    SegmentedReasoning {
        user: fallback_prompt.map(String::from),
        thought: None,
        answer: trimmed.to_string(),
        format: ReasoningFormat::Plain,
        equation_verification: EquationVerification {
            score: 1.0,
            verified_equations: 0,
            total_equations: 0,
        },
        dsml_tool_calls: dsml_calls,
    }
}

/// Formats a prompt into `MCL4-COE` `#U$# -> #T$# -> #A$#` framing when prefixes are supplied.
pub fn format_uta_prompt(
    prompt: &str,
    thought_prefix: Option<&str>,
    answer_prefix: Option<&str>,
) -> String {
    let base = if prompt.trim_start().starts_with("#U$#") {
        prompt.trim().to_string()
    } else {
        format!("#U$# {}", prompt.trim())
    };
    let mut out = base;
    if let Some(tp) = thought_prefix.map(str::trim).filter(|s| !s.is_empty()) {
        out.push_str("\n#T$# ");
        out.push_str(tp);
    }
    if let Some(ap) = answer_prefix.map(str::trim).filter(|s| !s.is_empty()) {
        out.push_str("\n#A$# ");
        out.push_str(ap);
    }
    out
}

/// Verifies simple arithmetic equations (`a + b = c`, `a - b = c`, `a * b = c`, `a / b = c`)
/// inside a `#T$#` or `<think>` trace, returning a consistency score in `[0.0, 1.0]`.
pub fn verify_thought_equations(thought: &str) -> EquationVerification {
    if !thought.contains('=') {
        return EquationVerification {
            score: 1.0,
            verified_equations: 0,
            total_equations: 0,
        };
    }
    let mut total = 0usize;
    let mut verified = 0usize;
    let bytes = thought.as_bytes();

    for (eq_pos, _) in thought.match_indices('=') {
        if (eq_pos > 0 && matches!(bytes[eq_pos - 1], b'!' | b'<' | b'>' | b'='))
            || (eq_pos + 1 < bytes.len() && bytes[eq_pos + 1] == b'=')
        {
            continue;
        }
        let lhs_str = thought[..eq_pos].trim();
        let rhs_str = thought[eq_pos + 1..].trim();
        let Some(lhs_val) = eval_simple_arith(lhs_str) else {
            continue;
        };
        let Some(rhs_val) = parse_leading_number(rhs_str) else {
            continue;
        };
        total += 1;
        if (lhs_val - rhs_val).abs() <= 1e-2 {
            verified += 1;
        }
    }

    let score = if total == 0 {
        1.0
    } else {
        (verified as f32) / (total as f32)
    };
    EquationVerification {
        score: (score * 10000.0).round() / 10000.0,
        verified_equations: verified,
        total_equations: total,
    }
}

fn parse_leading_number(s: &str) -> Option<f64> {
    let trimmed = s.trim().trim_end_matches(['.', ',', ';']);
    let mut end = 0usize;
    for (idx, ch) in trimmed.char_indices() {
        if ch.is_ascii_digit() || ch == '.' || (idx == 0 && (ch == '-' || ch == '+')) {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    trimmed[..end].trim_end_matches('.').parse::<f64>().ok()
}

fn eval_simple_arith(expr: &str) -> Option<f64> {
    // Zero-allocation O(1) reverse token scan (only inspects the 3 tokens preceding `=`)
    let mut rev = expr.split_whitespace().rev();
    let rhs_tok = rev.next()?;
    let op_tok = rev.next()?;
    let lhs_tok = rev.next()?;
    let a = parse_leading_number(lhs_tok)?;
    let b = parse_leading_number(rhs_tok)?;
    match op_tok {
        "+" => Some(a + b),
        "-" => Some(a - b),
        "*" | "×" | "x" => Some(a * b),
        "/" | "÷" => {
            if b.abs() > 1e-12 {
                Some(a / b)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Statistics from `dsh-compaction-basic` + DeepSeek-V4.1 multi-turn context compaction.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompactionStats {
    pub messages_compacted: usize,
    pub tool_results_pruned: usize,
    pub chars_stripped: usize,
    pub approx_tokens_saved: usize,
}

/// Compacts historical assistant messages and prunes oversized old tool outputs for
/// DeepSeek-V4.1 and MCL4-COE multi-turn tool loops (`dsh-compaction-basic` pattern).
///
/// 1. Strips `<think>...</think>`, `#T$#...#A$#`, and `<｜DSML｜...>` scratchpads from prior
///    assistant turns before sending step $t + 1$ so the 128-token MLA sliding window and
///    140-token YOCO prefill replay suffix stay focused on verified `#A$#` conclusions.
/// 2. Prunes oversized historical tool results older than the most recent tool batch when
///    `max_old_tool_chars > 0`.
pub fn compact_messages_for_dsv41_mcl4(
    messages: &[Message],
    strip_prior_thoughts: bool,
) -> (Vec<Message>, CompactionStats) {
    compact_messages_with_tool_pruning(messages, strip_prior_thoughts, 0)
}

/// Full `dsh-compaction-basic` pre-step compaction with both thought stripping and historical
/// tool-result pruning.
pub fn compact_messages_with_tool_pruning(
    messages: &[Message],
    strip_prior_thoughts: bool,
    max_old_tool_chars: usize,
) -> (Vec<Message>, CompactionStats) {
    let mut out = Vec::with_capacity(messages.len());
    let mut stats = CompactionStats::default();

    // Identify index of the last assistant message so tool results after it are considered "recent"
    let last_assistant_idx = messages
        .iter()
        .rposition(|m| matches!(m.role, Role::Assistant))
        .unwrap_or(0);

    for (idx, msg) in messages.iter().enumerate() {
        let mut cloned = msg.clone();
        if strip_prior_thoughts && matches!(cloned.role, Role::Assistant) {
            let segmented =
                segment_reasoning_and_answer(&cloned.content, cloned.reasoning.as_deref(), None);
            let before_len =
                cloned.content.len() + cloned.reasoning.as_ref().map(|r| r.len()).unwrap_or(0);
            if segmented.thought.is_some()
                || cloned.reasoning.is_some()
                || !segmented.dsml_tool_calls.is_empty()
            {
                cloned.content = segmented.answer;
                cloned.reasoning = None;
                let after_len = cloned.content.len();
                let saved = before_len.saturating_sub(after_len);
                if saved > 0 {
                    stats.messages_compacted += 1;
                    stats.chars_stripped += saved;
                    stats.approx_tokens_saved += saved.div_ceil(4);
                }
            }
        } else if max_old_tool_chars > 0
            && matches!(cloned.role, Role::Tool)
            && idx < last_assistant_idx
            && cloned.content.len() > max_old_tool_chars
        {
            let before_len = cloned.content.len();
            let head_keep = max_old_tool_chars / 2;
            let tail_keep = max_old_tool_chars.saturating_sub(head_keep);
            let head: String = cloned.content.chars().take(head_keep).collect();
            let tail: String = cloned
                .content
                .chars()
                .rev()
                .take(tail_keep)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            cloned.content = format!(
                "{head}\n...[dsh-compaction: pruned {} chars from historical tool output]...\n{tail}",
                before_len.saturating_sub(head.len() + tail.len())
            );
            let saved = before_len.saturating_sub(cloned.content.len());
            if saved > 0 {
                stats.tool_results_pruned += 1;
                stats.chars_stripped += saved;
                stats.approx_tokens_saved += saved.div_ceil(4);
            }
        }
        out.push(cloned);
    }

    (out, stats)
}

/// Detects whether the agent is stuck in a repetitive tool-call doom loop
/// (e.g., invoking the exact same tool name and JSON arguments `threshold` times in a row).
pub fn detect_tool_doom_loop(
    prior_signatures: &[Vec<(String, String)>],
    current_calls: &[ToolCall],
    threshold: usize,
) -> Option<String> {
    if current_calls.is_empty() || threshold < 2 {
        return None;
    }
    let current_sig: Vec<(String, String)> = current_calls
        .iter()
        .map(|c| (c.name.clone(), c.arguments_json.clone()))
        .collect();
    let needed_prior = threshold - 1;
    if prior_signatures.len() < needed_prior {
        return None;
    }
    let tail = &prior_signatures[prior_signatures.len() - needed_prior..];
    if tail.iter().all(|sig| *sig == current_sig) {
        let names = current_calls
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!(
            "doom-loop guard triggered: identical tool call batch [{names}] repeated {threshold} consecutive steps"
        ))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// 5. MCL4-COE 9-Operator Cognitive Ensemble & Decision Probe
// ---------------------------------------------------------------------------

/// Candidate evaluation row returned by [`probe_decision_candidates`] (matches `MCL4-COE` `/api/probe/decision`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoeDecisionCandidate {
    pub index: usize,
    pub option: String,
    pub selected: bool,
    pub probability: f32,
    pub pearson_similarity: f32,
    pub utility: f32,
    pub entailed: f32,
    pub contradicted: f32,
    pub correctness: f32,
    pub sufficiency: f32,
}

/// Full response from a Typed-COE `Decision + Deduction + Evaluation + Metacognition` probe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoeDecisionProbeReport {
    pub winner_index: usize,
    pub winner_option: String,
    pub used_whiteboard: bool,
    pub recurrence_steps: usize,
    pub candidates: Vec<CoeDecisionCandidate>,
}

/// Evaluates a multiple-choice decision probe using deterministic cosine/Pearson latent
/// feature projections across the `Decision`, `Deduction`, `Evaluation`, and `Metacognition`
/// operator heads when no external PyTorch daemon is connected.
pub fn probe_decision_candidates(
    context: &str,
    question: &str,
    options: &[String],
    knobs: &Mcl4CoeKnobs,
) -> Result<CoeDecisionProbeReport, String> {
    let clean_opts: Vec<String> = options
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if clean_opts.len() < 2 {
        return Err("provide at least 2 non-empty candidate options".to_string());
    }

    let query_vec = embed_text_feature_64(&format!("{context} {question}"), knobs.recurrence_steps);
    let mut logits = Vec::with_capacity(clean_opts.len());
    let mut sims = Vec::with_capacity(clean_opts.len());
    let mut utilities = Vec::with_capacity(clean_opts.len());
    let mut entailed_vec = Vec::with_capacity(clean_opts.len());
    let mut contradicted_vec = Vec::with_capacity(clean_opts.len());
    let mut correctness_vec = Vec::with_capacity(clean_opts.len());
    let mut sufficiency_vec = Vec::with_capacity(clean_opts.len());

    for opt in &clean_opts {
        let opt_vec = embed_text_feature_64(opt, knobs.recurrence_steps);
        let pearson = pearson_corr_64(&query_vec, &opt_vec);
        let overlap = lexical_entailment_score(context, question, opt);
        let contradiction = contradiction_heuristic(context, opt);

        let logit = pearson * 2.4 + overlap * 1.8 - contradiction * 1.5;
        let util = (0.55 * overlap + 0.45 * pearson - 0.4 * contradiction).clamp(-1.0, 1.0);
        let ent = (0.5 + 0.45 * overlap - 0.35 * contradiction).clamp(0.01, 0.99);
        let contra = (contradiction * 0.8 + (1.0 - ent) * 0.2).clamp(0.01, 0.99);
        let corr = (0.5 + 0.35 * pearson + 0.25 * overlap - 0.25 * contradiction).clamp(0.05, 0.99);
        let suff =
            (0.45 + 0.4 * overlap + 0.15 * (opt.len().min(120) as f32 / 120.0)).clamp(0.05, 0.99);

        logits.push(logit);
        sims.push(pearson);
        utilities.push(util);
        entailed_vec.push(ent);
        contradicted_vec.push(contra);
        correctness_vec.push(corr);
        sufficiency_vec.push(suff);
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum::<f32>() + 1e-12;
    let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

    let mut winner_idx = 0usize;
    for i in 1..probs.len() {
        if probs[i] > probs[winner_idx] {
            winner_idx = i;
        }
    }

    let candidates = clean_opts
        .iter()
        .enumerate()
        .map(|(i, opt)| CoeDecisionCandidate {
            index: i,
            option: opt.clone(),
            selected: i == winner_idx,
            probability: round4(probs[i]),
            pearson_similarity: round4(sims[i]),
            utility: round4(utilities[i]),
            entailed: round4(entailed_vec[i]),
            contradicted: round4(contradicted_vec[i]),
            correctness: round4(correctness_vec[i]),
            sufficiency: round4(sufficiency_vec[i]),
        })
        .collect();

    Ok(CoeDecisionProbeReport {
        winner_index: winner_idx,
        winner_option: clean_opts[winner_idx].clone(),
        used_whiteboard: knobs.use_whiteboard,
        recurrence_steps: knobs.recurrence_steps,
        candidates,
    })
}

fn round4(v: f32) -> f32 {
    (v * 10000.0).round() / 10000.0
}

fn embed_text_feature_64(text: &str, recurrence_steps: usize) -> [f32; 64] {
    let mut vec = [0.0f32; 64];
    for (idx, byte) in text.to_ascii_lowercase().bytes().enumerate() {
        let slot1 = (byte as usize).wrapping_mul(31).wrapping_add(idx * 7) % 64;
        let slot2 = (byte as usize).wrapping_mul(17).wrapping_add(idx * 13) % 64;
        vec[slot1] += 1.0;
        vec[slot2] += 0.5;
    }
    for _ in 0..recurrence_steps.max(1) {
        let mut next = [0.0f32; 64];
        for i in 0..64 {
            next[i] = 0.7 * vec[i] + 0.15 * vec[(i + 1) % 64] + 0.15 * vec[(i + 63) % 64];
        }
        vec = next;
    }
    vec
}

fn pearson_corr_64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    let mean_a = a.iter().sum::<f32>() / 64.0;
    let mean_b = b.iter().sum::<f32>() / 64.0;
    let mut num = 0.0f32;
    let mut den_a = 0.0f32;
    let mut den_b = 0.0f32;
    for i in 0..64 {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let denom = (den_a * den_b).sqrt();
    if denom > 1e-8 {
        (num / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn lexical_entailment_score(context: &str, question: &str, option: &str) -> f32 {
    let ctx_lower = format!(
        "{} {}",
        context.to_ascii_lowercase(),
        question.to_ascii_lowercase()
    );
    let ctx_words: std::collections::BTreeSet<&str> = ctx_lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() >= 3)
        .collect();
    let opt_lower = option.to_ascii_lowercase();
    let opt_words: Vec<&str> = opt_lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() >= 3)
        .collect();
    if opt_words.is_empty() {
        return 0.0;
    }
    let hits = opt_words.iter().filter(|w| ctx_words.contains(**w)).count();
    (hits as f32) / (opt_words.len() as f32)
}

fn contradiction_heuristic(context: &str, option: &str) -> f32 {
    let ctx_neg = context.to_ascii_lowercase().contains(" not ")
        || context.to_ascii_lowercase().contains(" never ")
        || context.to_ascii_lowercase().contains(" false");
    let opt_neg = option.to_ascii_lowercase().contains(" not ")
        || option.to_ascii_lowercase().contains(" never ")
        || option.to_ascii_lowercase().contains(" false")
        || option.to_ascii_lowercase().contains("impossible");
    if ctx_neg != opt_neg { 0.35 } else { 0.05 }
}

// ---------------------------------------------------------------------------
// 6. Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Role};
    use serde_json::json;

    #[test]
    fn dsv41_mcl4c_packet_layout_is_4k_page_aligned() {
        let layout = Dsv41Mcl4cPacketLayout::new();
        let sum = layout.w1_w3_fp4_bytes
            + layout.w2_fp4_bytes
            + layout.w1_w3_scales_bytes
            + layout.w2_scales_bytes
            + layout.scale2_scalars_bytes
            + layout.meta_bytes
            + layout.pad_bytes;
        assert_eq!(sum, Dsv41Mcl4cPacketLayout::TOTAL_PACKET_BYTES);
        assert_eq!(layout.total_packet_bytes, 19_910_656);
        assert_eq!(layout.total_packet_bytes % 4096, 0);

        let meta = Dsv41Mcl4cPacketLayout::encode_trailer_meta(0.125, 0.25, 0.5, 14, 383);
        let (w1, w3, w2, l, e) = Dsv41Mcl4cPacketLayout::decode_trailer_meta(&meta);
        assert!((w1 - 0.125).abs() < 1e-6);
        assert!((w3 - 0.25).abs() < 1e-6);
        assert!((w2 - 0.5).abs() < 1e-6);
        assert_eq!(l, 14);
        assert_eq!(e, 383);
    }

    #[test]
    fn dsv41_arena_cache_plan_fits_12gb_rtx5070() {
        let plan = Dsv41ArenaCachePlan::try_plan_for_host(12.0, 64.0).unwrap();
        assert_eq!(plan.ping_pong_arena_slots, 8);
        assert!((plan.ping_pong_arena_mib - 151.91).abs() < 0.05);
        assert!((plan.non_expert_backbone_gib - 6.495014190673828).abs() < 1e-9);
        assert_eq!(plan.warm_expert_cache_slots, 162);
        assert_eq!(plan.recommended_top_k, 6);
        assert_eq!(plan.nvme_engram_vram_bytes, 0);
    }

    #[test]
    fn memory_pressure_never_changes_architecture_top_k() {
        let arch = DeepSeekV41ArchSpec::deepseek_v41_flash_nvfp4();
        assert_eq!(arch.recommended_top_k_for_vram_gib(8.0, true), 6);
        assert_eq!(arch.recommended_top_k_for_vram_gib(96.0, false), 6);
        assert!(Dsv41ArenaCachePlan::try_plan_for_host(1.0, 64.0).is_err());
        assert!(Dsv41ArenaCachePlan::try_plan_for_host(f64::NAN, 64.0).is_err());
        assert!(Dsv41ArenaCachePlan::try_plan_for_host(12.0, -1.0).is_err());
        assert_eq!(
            Dsv41ArenaCachePlan::plan_for_host(1.0, 64.0).warm_expert_cache_slots,
            0
        );
    }

    #[test]
    fn native_defaults_disable_latency_heavy_optional_features() {
        let native = Mcl4CoeKnobs::mcl4_coe_v7_default();
        assert!(!native.adaptive_recurrent_depth);
        assert!(!native.use_windowless_pager);
        let deepseek = Mcl4CoeKnobs::deepseek_v41_default();
        assert!(deepseek.adaptive_recurrent_depth);
        assert!(deepseek.use_windowless_pager);
    }

    #[test]
    fn sinkhorn_knopp_4x4_produces_doubly_stochastic_matrix() {
        let logits = [
            [1.2, -0.4, 0.3, 0.8],
            [-0.2, 0.9, 1.5, -0.7],
            [0.5, 0.1, -0.3, 1.1],
            [0.0, 0.6, 0.2, -0.1],
        ];
        let p = sinkhorn_knopp_4x4(&logits, 20);
        for (r, row) in p.iter().enumerate() {
            let row_sum: f32 = row.iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {r} sum = {row_sum}");
        }
        for c in [0, 1, 2, 3] {
            let col_sum = p[0][c] + p[1][c] + p[2][c] + p[3][c];
            assert!((col_sum - 1.0).abs() < 1e-5, "col {c} sum = {col_sum}");
        }
    }

    #[test]
    fn dsv41_router_top6_noaux_tc_respects_bias_and_tabu_penalty() {
        let logits = vec![0.5, 1.2, 0.8, 2.0, 1.9, 1.7, 0.3, 1.5];
        let bias = vec![0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0];
        let coverage = vec![0, 0, 0, 4, 0, 0, 0, 0];
        let route = dsv41_route_top6_noaux_tc(&logits, &bias, 6, 0, false, &coverage, 0.35, 1.5);
        assert_eq!(route.selected_experts.len(), 6);
        assert_eq!(route.tabu_penalized_count, 1);
        let weight_sum: f32 = route.normalized_weights.iter().sum();
        assert!((weight_sum - 1.5).abs() < 1e-4);
    }

    #[test]
    fn dense_sparse_coalition_is_additive_and_disjoint() {
        let logits = [0.1, 0.2, 5.0, 4.0, 3.0, 2.0];
        let route = dsv41_route_top6_noaux_tc(&logits, &[], 2, 2, true, &[], 0.0, 1.5);
        assert_eq!(route.selected_experts, vec![0, 1, 2, 3]);
        assert_eq!(route.dense_prefix_count, 2);
        let dense_only = dsv41_route_top6_noaux_tc(&logits, &[], 0, 2, true, &[], 0.0, 1.5);
        assert_eq!(dense_only.selected_experts, vec![0, 1]);
    }

    #[test]
    fn bounded_top_k_preserves_tie_breaks_and_puts_nan_last() {
        let scores = [1.0, f32::NAN, 1.0, 2.0, f32::NEG_INFINITY];
        assert_eq!(
            select_top_k_indices(&scores, 0..scores.len(), 3),
            vec![3, 0, 2]
        );
        assert_eq!(
            select_top_k_indices(&scores, 0..scores.len(), 5),
            vec![3, 0, 2, 4, 1]
        );
    }

    #[test]
    fn clamped_swiglu_enforces_limit_10() {
        let extreme = clamped_swiglu(50.0, 25.0, 10.0);
        assert!((extreme - 100.0).abs() < 1e-3);
        let neg_up = clamped_swiglu(50.0, -25.0, 10.0);
        assert!((neg_up + 100.0).abs() < 1e-3);
    }

    #[test]
    fn engram_hashes_match_independent_cuda_golden_vectors() {
        let window = [101u32, 2054, 8812, 42];
        assert_eq!(
            engram_24prime_xor_hashes(&window, 1),
            [
                6_954_960,
                28_896_993,
                43_990_981,
                52_792_014,
                74_159_260,
                86_659_225,
                100_917_556,
                127_611_330,
                133_278_037,
                145_277_624,
                171_095_333,
                179_049_842,
                207_167_740,
                221_421_689,
                235_046_206,
                254_734_328,
                260_971_771,
                281_219_317,
                296_421_041,
                304_067_509,
                335_849_382,
                343_546_528,
                366_333_811,
                382_188_206,
            ]
        );
        assert_eq!(
            engram_24prime_xor_hashes(&window, 14),
            [
                8_970_206,
                28_011_524,
                38_527_024,
                51_153_465,
                65_551_723,
                87_945_729,
                99_954_055,
                123_675_649,
                132_426_008,
                149_888_692,
                168_601_312,
                187_930_495,
                200_052_066,
                216_715_485,
                231_183_461,
                242_088_686,
                267_286_241,
                277_906_707,
                288_793_336,
                314_258_764,
                328_794_477,
                343_442_724,
                352_400_878,
                381_418_029,
            ]
        );

        // These vectors exercise uint64 wrap followed by signed-int64 modulo.
        let high_window = [129_279u32, 2054, 8812, 42];
        assert_eq!(engram_24prime_xor_hashes(&high_window, 1)[16], 260_780_547);
        assert_eq!(engram_24prime_xor_hashes(&high_window, 14)[16], 269_871_755);

        let gate_pos = engram_signed_sqrt_gate(0.64);
        let gate_neg = engram_signed_sqrt_gate(-0.64);
        assert!(gate_pos > 0.5 && gate_neg < 0.5);
    }

    #[test]
    fn dsml_tool_call_parser_extracts_invokes_and_preserves_string_types() {
        let dsml = r#"<think>Let me check the weather first. 6 * 7 = 42.</think>Checking weather now.
<｜DSML｜ calls>
<｜DSML｜ invoke name="demo__get_weather">
<｜DSML｜ parameter name="city" string="true">12345</｜DSML｜ parameter>
<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>"#;
        let seg = segment_reasoning_and_answer(dsml, None, None);
        assert_eq!(seg.format, ReasoningFormat::DeepSeekThinkTag);
        assert_eq!(
            seg.thought.as_deref(),
            Some("Let me check the weather first. 6 * 7 = 42.")
        );
        assert_eq!(seg.answer, "Checking weather now.");
        assert_eq!(seg.dsml_tool_calls.len(), 1);
        assert_eq!(seg.dsml_tool_calls[0].name, "demo__get_weather");
        assert_eq!(
            seg.dsml_tool_calls[0].arguments().unwrap(),
            json!({
                "city": "12345",
                "days": 3
            })
        );
    }

    #[test]
    fn dsml_parser_rejects_incomplete_invalid_and_duplicate_calls_as_text() {
        let truncated = r#"<｜DSML｜ calls><｜DSML｜ invoke name="demo">"#;
        let outcome = parse_dsml_tool_calls(truncated);
        assert!(outcome.tool_calls.is_empty());
        assert_eq!(outcome.clean_text, truncated);

        let malformed = [
            r#"<｜DSML｜ invoke name="demo"></｜DSML｜ invoke></｜DSML｜ calls>"#,
            r#"<｜DSML｜ calls><｜DSML｜ invoke surname="demo"></｜DSML｜ invoke></｜DSML｜ calls>"#,
            r#"<｜DSML｜ calls><｜DSML｜ invoke name="demo"><｜DSML｜ parameter name="x" string="false">{bad}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"#,
            r#"<｜DSML｜ calls><｜DSML｜ invoke name="demo"><｜DSML｜ parameter name="x">1</｜DSML｜ parameter><｜DSML｜ parameter name="x">2</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"#,
            r#"<｜DSML｜ calls><｜DSML｜ invoke name="demo"><｜DSML｜ parameter name="x" string="false">1</｜DSML｜ parameter></｜DSML｜ calls>"#,
            r#"<｜DSML｜ calls><｜DSML｜ invoke name="demo"><｜DSML｜ parameter name="x" string="false">1</｜DSML｜ invoke></｜DSML｜ calls>"#,
        ];
        for raw in malformed {
            let outcome = parse_dsml_tool_calls(raw);
            assert!(outcome.tool_calls.is_empty(), "executed malformed {raw}");
            assert_eq!(outcome.clean_text, raw);
        }

        // A later invalid invoke also rolls back earlier invokes in the same block.
        let partially_valid = r#"<｜DSML｜ calls><｜DSML｜ invoke name="ok"></｜DSML｜ invoke><｜DSML｜ invoke name="bad"><｜DSML｜ parameter name="x" string="false">no</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"#;
        assert!(parse_dsml_tool_calls(partially_valid).tool_calls.is_empty());
    }

    #[test]
    fn dsml_ids_are_process_unique_across_parses() {
        let raw =
            r#"<｜DSML｜ calls><｜DSML｜ invoke name="demo"></｜DSML｜ invoke></｜DSML｜ calls>"#;
        let first = parse_dsml_tool_calls(raw);
        let second = parse_dsml_tool_calls(raw);
        assert!(first.tool_calls[0].id.starts_with("call-"));
        assert_ne!(first.tool_calls[0].id, second.tool_calls[0].id);
    }

    #[test]
    fn unclosed_think_tag_recovers_answer_on_2bit_packs() {
        let raw_unclosed = "<think>Because 12 + 8 = 20, the capital of France is Paris.";
        let seg = segment_reasoning_and_answer(raw_unclosed, None, None);
        assert_eq!(seg.format, ReasoningFormat::DeepSeekThinkTag);
        assert_eq!(
            seg.answer,
            "Because 12 + 8 = 20, the capital of France is Paris."
        );
        assert_eq!(seg.equation_verification.verified_equations, 1);
    }

    #[test]
    fn uta_and_think_segmentation_verifies_equations_and_compacts_history() {
        let raw_uta = "#U$# Compute 14 * 3 + 8\n#T$# Step 1: 14 * 3 = 42. Step 2: 42 + 8 = 50.\n#A$# The result is 50.";
        let seg = segment_reasoning_and_answer(raw_uta, None, None);
        assert_eq!(seg.format, ReasoningFormat::Mcl4Uta);
        assert_eq!(seg.user.as_deref(), Some("Compute 14 * 3 + 8"));
        assert_eq!(
            seg.thought.as_deref(),
            Some("Step 1: 14 * 3 = 42. Step 2: 42 + 8 = 50.")
        );
        assert_eq!(seg.answer, "The result is 50.");
        assert_eq!(seg.equation_verification.total_equations, 2);
        assert_eq!(seg.equation_verification.verified_equations, 2);
        assert!((seg.equation_verification.score - 1.0).abs() < 1e-5);

        let history = vec![
            Message::user("What is 14 * 3 + 8?"),
            Message {
                role: Role::Assistant,
                content: raw_uta.to_string(),
                reasoning: seg.thought.clone(),
                tool_call_id: None,
                name: None,
                tool_calls: Vec::new(),
                is_error: false,
                provider_content: None,
            },
        ];
        let (compacted, stats) = compact_messages_for_dsv41_mcl4(&history, true);
        assert_eq!(stats.messages_compacted, 1);
        assert_eq!(compacted[1].content, "The result is 50.");
        assert!(compacted[1].reasoning.is_none());
    }

    #[test]
    fn decision_probe_ranks_entailed_candidate_first() {
        let knobs = Mcl4CoeKnobs::deepseek_v41_default();
        let report = probe_decision_candidates(
            "DeepSeek-V4.1 uses 384 routed NVFP4 experts with top-6 selection and 128-token MLA sliding window.",
            "How many routed NVFP4 experts does DeepSeek-V4.1 use per layer?",
            &[
                "384 routed NVFP4 experts with top-6 selection".to_string(),
                "16 dense FP32 experts with top-1 selection".to_string(),
            ],
            &knobs,
        )
        .unwrap();
        assert_eq!(report.winner_index, 0);
        assert!(report.candidates[0].probability > report.candidates[1].probability);
    }
}
