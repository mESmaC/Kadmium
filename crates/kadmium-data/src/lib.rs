//! # `kadmium-data`
//!
//! Data plane for **Kadmium (`Cd`)**, synthesizing two key inspirations:
//! 1. **FreeToken (`FTW`) & Colibri:** The **Kadmium Tensor Bank (`.ktb`)** zero-copy,
//!    page-aligned weight and MoE expert bank layout for direct DMA/NVMe streaming.
//! 2. **CERN FTS3 & Davix:** Third-Party Transfer orchestration where the Home NAS or
//!    seed peers stream chunked `.ktb` banks directly peer-to-peer over parallel QUIC
//!    streams without bottlenecking the coordinator.

use kadmium_core::{HardwareNode, NodeId, NodeRole};
use kadmium_sharding::{ClusterShardPlan, ResponsibilityShiftPlan};
use serde::{Deserialize, Serialize};

/// Magic header bytes for the Kadmium Tensor Bank (`.ktb`) file format (`"KDTB"`).
pub const KTB_MAGIC_BYTES: [u8; 4] = *b"KDTB";

/// Default page alignment (64 KiB) for zero-copy `mmap` / `io_uring` / `IoRing` NVMe streaming.
pub const KTB_PAGE_ALIGNMENT_BYTES: u64 = 65_536;

/// Quantization and tensor encoding format inside a `.ktb` bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorEncoding {
    Fp16,
    Bf16,
    Fp8E4M3,
    Int8Symmetric,
    Q4KBlock,
    Nf4DoubleQuant,
}

/// Metadata descriptor for a pre-merged **Kadmium Tensor Bank (`.ktb`)** shard file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KadmiumTensorBankManifest {
    pub magic: [u8; 4],
    pub format_version: u16,
    pub model_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub dense_encoding: TensorEncoding,
    pub expert_encoding: TensorEncoding,
    pub page_alignment_bytes: u64,
    pub total_chunks: u32,
    pub total_size_gb: f64,
}

impl KadmiumTensorBankManifest {
    pub fn new_for_shard(model_id: &str, start_layer: u32, end_layer: u32, size_gb: f64) -> Self {
        let chunks = ((size_gb * 1024.0) / 64.0).ceil().max(1.0) as u32; // 64 MiB Davix-style chunks
        Self {
            magic: KTB_MAGIC_BYTES,
            format_version: 1,
            model_id: model_id.to_string(),
            start_layer,
            end_layer,
            dense_encoding: TensorEncoding::Fp8E4M3,
            expert_encoding: TensorEncoding::Q4KBlock,
            page_alignment_bytes: KTB_PAGE_ALIGNMENT_BYTES,
            total_chunks: chunks,
            total_size_gb: (size_gb * 100.0).round() / 100.0,
        }
    }
}

/// State machine stages for a CERN FTS3-inspired Third-Party Transfer job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fts3TransferState {
    /// Queued in FTS3 scheduler awaiting source/destination bandwidth slot.
    Scheduled,
    /// Active direct peer-to-peer multi-stream QUIC transfer (Davix chunk mode).
    ActiveDirectP2PStream,
    /// Verifying chunk Blake3 checksums on destination NVMe/RAM tier.
    VerifyingBlake3Checksums,
    /// Completed and memory-mapped ready for compute.
    CompletedReady,
    /// Transient link drop; retrying specific failed 64 MiB chunks with backoff.
    RetryingFailedChunks,
}

/// Kind of payload transferred by the FTS3 data plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferPayloadKind {
    /// Static `.ktb` model weight & MoE expert banks (NAS/Peer -> Worker).
    StaticWeightBankKtb,
    /// Live boundary delta state (KV-cache slice or LoRA adapter gradients during a shift).
    LiveBoundaryStateDelta,
    /// Pre-tokenized training dataset shard or rolling checkpoint archive.
    DatasetOrCheckpointArchive,
}

/// A CERN FTS3-style Third-Party Transfer job between two peers in the Kadmium mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdPartyTransferJob {
    pub transfer_id: String,
    pub source_node: NodeId,
    pub destination_node: NodeId,
    pub payload_kind: TransferPayloadKind,
    pub start_layer: u32,
    pub end_layer: u32,
    pub volume_gb: f64,
    pub parallel_quic_streams: u8,
    pub allocated_bandwidth_gbps: f64,
    pub estimated_duration_sec: f64,
    pub state: Fts3TransferState,
}

/// FTS3-inspired transfer scheduler that plans non-bottlenecked P2P transfers.
pub struct Fts3TransferScheduler;

impl Fts3TransferScheduler {
    /// Identifies the primary `StorageVault` (e.g., Home NAS) or fallback seed node in the cluster.
    pub fn select_seed_vault(nodes: &[HardwareNode]) -> NodeId {
        nodes
            .iter()
            .find(|n| n.roles.contains(&NodeRole::StorageVault))
            .or_else(|| nodes.first())
            .map(|n| n.id.clone())
            .unwrap_or_else(|| NodeId("node-home-nas".into()))
    }

    /// Generates initial FTS3 third-party `.ktb` weight staging transfers from the Home NAS
    /// (or peer seed) directly to each assigned compute worker.
    pub fn plan_initial_staging(
        plan: &ClusterShardPlan,
        nodes: &[HardwareNode],
    ) -> Vec<ThirdPartyTransferJob> {
        let seed_node = Self::select_seed_vault(nodes);
        let mut jobs = Vec::new();

        for (idx, assign) in plan.assignments.iter().enumerate() {
            if let (Some(s), Some(e)) = (assign.start_layer, assign.end_layer) {
                if assign.node_id == seed_node {
                    continue;
                }
                let total_gb = assign.tiered_memory.hot_vram_or_unified_gb
                    + assign.tiered_memory.warm_system_ram_gb
                    + assign.tiered_memory.cold_nvme_stream_gb;

                let dest_uplink = nodes
                    .iter()
                    .find(|n| n.id == assign.node_id)
                    .map(|n| n.network_uplink_gbps)
                    .unwrap_or(1.0);

                let bw_gbps = dest_uplink.min(10.0) * 0.85;
                let duration_sec = (total_gb * 8.0) / bw_gbps.max(0.1);

                jobs.push(ThirdPartyTransferJob {
                    transfer_id: format!("fts3-init-e{}-{:02}", plan.epoch, idx + 1),
                    source_node: seed_node.clone(),
                    destination_node: assign.node_id.clone(),
                    payload_kind: TransferPayloadKind::StaticWeightBankKtb,
                    start_layer: s,
                    end_layer: e,
                    volume_gb: (total_gb * 100.0).round() / 100.0,
                    parallel_quic_streams: if bw_gbps >= 2.0 { 8 } else { 4 },
                    allocated_bandwidth_gbps: (bw_gbps * 100.0).round() / 100.0,
                    estimated_duration_sec: (duration_sec * 10.0).round() / 10.0,
                    state: Fts3TransferState::CompletedReady,
                });
            }
        }

        jobs
    }

    /// Generates two-phase FTS3 transfer jobs for a live `ResponsibilityShiftPlan`:
    /// 1. Background pre-staging of static `.ktb` weights from the Home NAS to the relief node.
    /// 2. Atomic step/token boundary P2P transfer of the compact live delta state from the yielding node.
    pub fn plan_live_shift_transfers(
        shift: &ResponsibilityShiftPlan,
        nodes: &[HardwareNode],
    ) -> Vec<ThirdPartyTransferJob> {
        let seed_node = Self::select_seed_vault(nodes);
        let mut jobs = Vec::new();

        for (idx, mig) in shift.migrations.iter().enumerate() {
            // Phase 1: Background FTS3 pre-stage of static weights from NAS -> relief node
            jobs.push(ThirdPartyTransferJob {
                transfer_id: format!("fts3-prestage-e{}-{:02}", shift.new_epoch, idx + 1),
                source_node: seed_node.clone(),
                destination_node: mig.to_node.clone(),
                payload_kind: TransferPayloadKind::StaticWeightBankKtb,
                start_layer: mig.start_layer,
                end_layer: mig.end_layer,
                volume_gb: mig.background_fts3_prestage_gb,
                parallel_quic_streams: 8,
                allocated_bandwidth_gbps: 2.1,
                estimated_duration_sec: ((mig.background_fts3_prestage_gb * 8.0) / 2.1 * 10.0)
                    .round()
                    / 10.0,
                state: Fts3TransferState::ActiveDirectP2PStream,
            });

            // Phase 2: Atomic boundary delta handoff (KV cache / LoRA gradients) from yielding -> relief node
            jobs.push(ThirdPartyTransferJob {
                transfer_id: format!("fts3-delta-e{}-{:02}", shift.new_epoch, idx + 1),
                source_node: mig.from_node.clone(),
                destination_node: mig.to_node.clone(),
                payload_kind: TransferPayloadKind::LiveBoundaryStateDelta,
                start_layer: mig.start_layer,
                end_layer: mig.end_layer,
                volume_gb: mig.live_boundary_delta_gb,
                parallel_quic_streams: 4,
                allocated_bandwidth_gbps: 2.5,
                estimated_duration_sec: ((mig.live_boundary_delta_gb * 8.0) / 2.5 * 100.0).round()
                    / 100.0,
                state: Fts3TransferState::Scheduled,
            });
        }

        jobs
    }
}

// ================================================================================================
// Milestone 1: Real Binary `.ktb` (Kadmium Tensor Bank) 64 KB Page-Aligned Packer & Zero-Copy Reader
// ================================================================================================

/// Metadata entry for a single 64 KB page-aligned tensor or MoE expert bank inside a `.ktb` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KtbPageAlignedTensorEntry {
    pub tensor_name: String,
    pub layer_index: u32,
    /// `None` for shared dense attention/trunk tensors; `Some(expert_id)` for routed MoE experts.
    pub expert_index: Option<u32>,
    pub encoding: TensorEncoding,
    /// Exact byte offset in the `.ktb` file (guaranteed to be a multiple of 65,536 / 64 KiB).
    pub page_offset_bytes: u64,
    /// Unpadded payload size in bytes.
    pub payload_bytes: u64,
    /// Total 64 KiB page-padded reservation in bytes.
    pub padded_reservation_bytes: u64,
    /// Deterministic 64-bit FNV-1a content integrity checksum.
    pub checksum_fnv64: u64,
}

/// Complete header embedded inside Page 0 (`0..65536`) of a binary `.ktb` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KtbBinaryArchiveHeader {
    pub model_id: String,
    pub format_version: u16,
    pub page_alignment_bytes: u64,
    pub entries: Vec<KtbPageAlignedTensorEntry>,
}

/// Computes a fast, deterministic 64-bit FNV-1a checksum over a tensor byte slice.
pub fn compute_fnv1a64_checksum(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Rounds `size` up to the next multiple of `KTB_PAGE_ALIGNMENT_BYTES` (64 KiB = 65,536 bytes).
pub fn align_to_64kb_page(size: usize) -> usize {
    let align = KTB_PAGE_ALIGNMENT_BYTES as usize;
    size.div_ceil(align) * align
}

struct RawTensorBankInput {
    name: String,
    layer: u32,
    expert: Option<u32>,
    encoding: TensorEncoding,
    data: Vec<u8>,
}

/// Binary packer that compiles raw tensors (from Safetensors / GGUF) into a 64 KB page-aligned
/// `.ktb` byte stream ready for zero-copy `mmap` NVMe streaming and FTS3 chunk transfers.
pub struct KtbBankPacker {
    model_id: String,
    tensors: Vec<RawTensorBankInput>,
}

impl KtbBankPacker {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            tensors: Vec::new(),
        }
    }

    /// Adds a dense layer tensor or routed MoE expert bank to be packed at a 64 KiB page boundary.
    pub fn add_tensor_bank(
        &mut self,
        tensor_name: impl Into<String>,
        layer_index: u32,
        expert_index: Option<u32>,
        encoding: TensorEncoding,
        payload: Vec<u8>,
    ) {
        self.tensors.push(RawTensorBankInput {
            name: tensor_name.into(),
            layer: layer_index,
            expert: expert_index,
            encoding,
            data: payload,
        });
    }

    /// Packs all added tensors into a 64 KiB page-aligned `.ktb` binary image.
    /// Page 0 (`0..65536`) contains the `"KDTB"` magic, version, and JSON header directory;
    /// subsequent 64 KiB pages hold each aligned tensor bank.
    pub fn pack_to_bytes(&self) -> Vec<u8> {
        let page_size = KTB_PAGE_ALIGNMENT_BYTES as usize;
        let mut current_offset = page_size as u64; // Page 0 reserved for header directory
        let mut entries = Vec::with_capacity(self.tensors.len());

        for t in &self.tensors {
            let payload_len = t.data.len();
            let padded_len = align_to_64kb_page(payload_len.max(1));
            let checksum = compute_fnv1a64_checksum(&t.data);

            entries.push(KtbPageAlignedTensorEntry {
                tensor_name: t.name.clone(),
                layer_index: t.layer,
                expert_index: t.expert,
                encoding: t.encoding,
                page_offset_bytes: current_offset,
                payload_bytes: payload_len as u64,
                padded_reservation_bytes: padded_len as u64,
                checksum_fnv64: checksum,
            });

            current_offset += padded_len as u64;
        }

        let header = KtbBinaryArchiveHeader {
            model_id: self.model_id.clone(),
            format_version: 1,
            page_alignment_bytes: KTB_PAGE_ALIGNMENT_BYTES,
            entries,
        };

        let header_json = serde_json::to_vec(&header).expect("serialize KtbBinaryArchiveHeader");
        assert!(
            10 + header_json.len() <= page_size,
            "KTB header directory exceeds 64 KiB Page 0"
        );

        let mut out = Vec::with_capacity(current_offset as usize);
        // Write Page 0: Magic (4B) + Version (2B) + JSON len (4B) + JSON + 0x00 padding to 64 KiB
        out.extend_from_slice(&KTB_MAGIC_BYTES);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&header_json);
        out.resize(page_size, 0u8);

        // Write each 64 KiB page-aligned tensor bank
        for t in &self.tensors {
            let padded_len = align_to_64kb_page(t.data.len().max(1));
            out.extend_from_slice(&t.data);
            let pad_bytes = padded_len - t.data.len();
            if pad_bytes > 0 {
                out.extend(std::iter::repeat_n(0u8, pad_bytes));
            }
        }

        out
    }
}

/// Zero-copy reader that parses a `.ktb` binary image (or memory-mapped file slice) and provides
/// $O(1)$ page-aligned slice access and checksum verification for dense layers and MoE experts.
pub struct KtbBankReader<'a> {
    raw: &'a [u8],
    pub header: KtbBinaryArchiveHeader,
}

impl<'a> KtbBankReader<'a> {
    pub fn from_slice(raw: &'a [u8]) -> Result<Self, String> {
        if raw.len() < KTB_PAGE_ALIGNMENT_BYTES as usize {
            return Err("Buffer smaller than 64 KiB KTB Page 0".into());
        }
        if raw[0..4] != KTB_MAGIC_BYTES {
            return Err("Invalid KTB magic header (expected 'KDTB')".into());
        }
        let json_len = u32::from_le_bytes([raw[6], raw[7], raw[8], raw[9]]) as usize;
        let header: KtbBinaryArchiveHeader = serde_json::from_slice(&raw[10..10 + json_len])
            .map_err(|e| format!("Invalid KTB header JSON: {e}"))?;

        Ok(Self { raw, header })
    }

    /// Returns a zero-copy slice to the requested tensor/expert entry after verifying 64 KiB
    /// page alignment and 64-bit checksum integrity.
    pub fn get_verified_tensor_slice(
        &self,
        entry: &KtbPageAlignedTensorEntry,
    ) -> Result<&'a [u8], String> {
        if !entry
            .page_offset_bytes
            .is_multiple_of(KTB_PAGE_ALIGNMENT_BYTES)
        {
            return Err(format!(
                "Tensor {} offset {} is not 64 KiB page-aligned",
                entry.tensor_name, entry.page_offset_bytes
            ));
        }
        let start = entry.page_offset_bytes as usize;
        let end = start + (entry.payload_bytes as usize);
        if end > self.raw.len() {
            return Err("Tensor slice exceeds KTB archive bounds".into());
        }
        let slice = &self.raw[start..end];
        let actual_hash = compute_fnv1a64_checksum(slice);
        if actual_hash != entry.checksum_fnv64 {
            return Err(format!(
                "Checksum mismatch for {}: expected {:#x}, got {:#x}",
                entry.tensor_name, entry.checksum_fnv64, actual_hash
            ));
        }
        Ok(slice)
    }
}

// ================================================================================================
// Safetensors Binary Parser & Automatic Shard-Filtered `.ktb` Transcoder
// ================================================================================================

/// Parsed descriptor of a single tensor inside a HuggingFace `.safetensors` binary file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorFieldSpec {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Transcoder that parses HuggingFace `.safetensors` binary buffers/files, automatically
/// classifies dense vs. MoE expert tensors by layer index, filters for a specific shard range
/// $[start\_layer, end\_layer]$, and compiles them into a 64 KiB page-aligned `.ktb` bank.
pub struct SafetensorsToKtbTranscoder;

impl SafetensorsToKtbTranscoder {
    /// Builds a valid `.safetensors` binary buffer in memory from a list of `(name, dtype, shape, raw_bytes)`.
    pub fn build_safetensors_bytes(tensors: &[(&str, &str, Vec<usize>, &[u8])]) -> Vec<u8> {
        let mut header_map = serde_json::Map::new();
        let mut current_offset = 0usize;
        for (name, dtype, shape, bytes) in tensors {
            let start = current_offset;
            let end = start + bytes.len();
            current_offset = end;
            header_map.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header_json = serde_json::to_vec(&header_map).expect("serialize safetensors header");
        let mut out = Vec::with_capacity(8 + header_json.len() + current_offset);
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_json);
        for (_, _, _, bytes) in tensors {
            out.extend_from_slice(bytes);
        }
        out
    }

    /// Parses layer index and optional MoE expert index from standard HuggingFace tensor names
    /// such as `"model.layers.14.mlp.experts.7.up_proj.weight"` or `"blk.3.attn_q.weight"`.
    pub fn parse_layer_and_expert(tensor_name: &str) -> (u32, Option<u32>) {
        let parts: Vec<&str> = tensor_name.split('.').collect();
        let mut layer_idx = 0u32;
        let mut expert_idx = None;

        for i in 0..parts.len().saturating_sub(1) {
            if (parts[i] == "layers" || parts[i] == "blk" || parts[i] == "h")
                && let Ok(l) = parts[i + 1].parse::<u32>()
            {
                layer_idx = l;
            }
            if (parts[i] == "experts" || parts[i] == "routed_experts")
                && let Ok(e) = parts[i + 1].parse::<u32>()
            {
                expert_idx = Some(e);
            }
        }
        (layer_idx, expert_idx)
    }

    /// Transcodes a `.safetensors` binary slice into a 64 KiB page-aligned `.ktb` bank,
    /// filtering only tensors belonging to `start_layer..=end_layer`.
    pub fn transcode_shard_to_ktb(
        model_id: &str,
        safetensors_bytes: &[u8],
        start_layer: u32,
        end_layer: u32,
    ) -> Result<Vec<u8>, String> {
        if safetensors_bytes.len() < 8 {
            return Err("Safetensors buffer too short (< 8 bytes)".into());
        }
        let header_len = u64::from_le_bytes(
            safetensors_bytes[0..8]
                .try_into()
                .map_err(|_| "Invalid header length bytes")?,
        ) as usize;
        let data_base = 8 + header_len;
        if safetensors_bytes.len() < data_base {
            return Err("Safetensors header exceeds buffer length".into());
        }

        let raw_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&safetensors_bytes[8..data_base])
                .map_err(|e| format!("Invalid Safetensors JSON header: {e}"))?;

        let mut packer = KtbBankPacker::new(model_id);

        // Sort keys deterministically so tensors are grouped by (layer_index, expert_index, name).
        let mut keys: Vec<&String> = raw_map.keys().filter(|k| *k != "__metadata__").collect();
        keys.sort_by_key(|k| {
            let (l, e) = Self::parse_layer_and_expert(k);
            (l, e.unwrap_or(0), (*k).clone())
        });

        for key in keys {
            let val = &raw_map[key];
            let spec: SafetensorFieldSpec = serde_json::from_value(val.clone())
                .map_err(|e| format!("Invalid tensor entry {key}: {e}"))?;

            let (layer_idx, expert_idx) = Self::parse_layer_and_expert(key);
            if layer_idx < start_layer || layer_idx > end_layer {
                continue;
            }

            let [start_off, end_off] = spec.data_offsets;
            let abs_start = data_base + start_off;
            let abs_end = data_base + end_off;
            if abs_end > safetensors_bytes.len() || abs_start > abs_end {
                return Err(format!("Out-of-bounds tensor offsets for {key}"));
            }

            let encoding = match spec.dtype.as_str() {
                "BF16" => TensorEncoding::Bf16,
                "F8_E4M3" => TensorEncoding::Fp8E4M3,
                "I8" | "U8" => TensorEncoding::Int8Symmetric,
                _ => TensorEncoding::Fp16,
            };

            packer.add_tensor_bank(
                key.clone(),
                layer_idx,
                expert_idx,
                encoding,
                safetensors_bytes[abs_start..abs_end].to_vec(),
            );
        }

        Ok(packer.pack_to_bytes())
    }
}

// ================================================================================================
// GGUF v3 Binary Header Parser, Shard Transcoder, & FTS3 Content-Addressed Chunk Slicer
// ================================================================================================

/// Magic bytes for a standard GGUF model file (`"GGUF"`).
pub const GGUF_MAGIC_BYTES: [u8; 4] = *b"GGUF";

/// Simplified GGML tensor type codes found in GGUF v3 headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum GgmlTypeCode {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q4K = 12,
    Bf16 = 30,
}

impl GgmlTypeCode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            30 => Self::Bf16,
            _ => Self::Q4K,
        }
    }

    pub fn to_ktb_encoding(self) -> TensorEncoding {
        match self {
            Self::F32 | Self::F16 => TensorEncoding::Fp16,
            Self::Bf16 => TensorEncoding::Bf16,
            Self::Q8_0 => TensorEncoding::Int8Symmetric,
            Self::Q4_0 | Self::Q4K => TensorEncoding::Q4KBlock,
        }
    }
}

/// Transcoder that parses GGUF v3 binary streams (`blk.{l}.*`), filters for an assigned
/// layer shard $[start\_layer, end\_layer]$, and compiles a 64 KiB page-aligned `.ktb` bank.
pub struct GgufToKtbTranscoder;

impl GgufToKtbTranscoder {
    /// Builds a compact GGUF v3 binary image in memory (0 metadata KV pairs, `N` tensor descriptors,
    /// 32-byte aligned data section) for testing and local shard transcoding.
    pub fn build_gguf_v3_bytes(tensors: &[(&str, GgmlTypeCode, u64, &[u8])]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(&GGUF_MAGIC_BYTES);
        header.extend_from_slice(&3u32.to_le_bytes()); // GGUF v3
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        header.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0

        let mut running_data_offset = 0u64;
        for (name, ggml_type, num_elements, raw_bytes) in tensors {
            let name_bytes = name.as_bytes();
            header.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            header.extend_from_slice(name_bytes);
            header.extend_from_slice(&1u32.to_le_bytes()); // 1D flattened shape for compact descriptor
            header.extend_from_slice(&num_elements.to_le_bytes());
            header.extend_from_slice(&(*ggml_type as u32).to_le_bytes());
            header.extend_from_slice(&running_data_offset.to_le_bytes());
            running_data_offset += raw_bytes.len() as u64;
        }

        // GGUF v3 pads the end of the tensor info header to 32-byte alignment
        let align = 32usize;
        let padded_header_len = header.len().div_ceil(align) * align;
        header.resize(padded_header_len, 0u8);

        for (_, _, _, raw_bytes) in tensors {
            header.extend_from_slice(raw_bytes);
        }
        header
    }

    /// Parses a GGUF v3 binary buffer (with 0 KV pairs) and transcodes tensors in
    /// `start_layer..=end_layer` into a 64 KiB page-aligned `.ktb` bank.
    pub fn transcode_shard_to_ktb(
        model_id: &str,
        gguf_bytes: &[u8],
        start_layer: u32,
        end_layer: u32,
    ) -> Result<Vec<u8>, String> {
        if gguf_bytes.len() < 24 {
            return Err("GGUF buffer shorter than 24-byte header".into());
        }
        if gguf_bytes[0..4] != GGUF_MAGIC_BYTES {
            return Err("Invalid GGUF magic bytes (expected 'GGUF')".into());
        }
        let version =
            u32::from_le_bytes([gguf_bytes[4], gguf_bytes[5], gguf_bytes[6], gguf_bytes[7]]);
        if version != 3 {
            return Err(format!("Unsupported GGUF version {version} (expected v3)"));
        }
        let tensor_count = u64::from_le_bytes(gguf_bytes[8..16].try_into().unwrap()) as usize;
        let kv_count = u64::from_le_bytes(gguf_bytes[16..24].try_into().unwrap()) as usize;
        if kv_count != 0 {
            return Err("Expected compact 0-KV GGUF stream".into());
        }

        let mut cursor = 24usize;
        let mut parsed_descriptors = Vec::with_capacity(tensor_count);

        for _ in 0..tensor_count {
            if cursor + 8 > gguf_bytes.len() {
                return Err("Truncated GGUF tensor name length".into());
            }
            let name_len =
                u64::from_le_bytes(gguf_bytes[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
            let name = std::str::from_utf8(&gguf_bytes[cursor..cursor + name_len])
                .map_err(|e| format!("Invalid UTF-8 tensor name: {e}"))?
                .to_string();
            cursor += name_len;

            let n_dims =
                u32::from_le_bytes(gguf_bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            for _ in 0..n_dims {
                cursor += 8;
            }
            let ggml_code = u32::from_le_bytes(gguf_bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let rel_offset =
                u64::from_le_bytes(gguf_bytes[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;

            parsed_descriptors.push((name, GgmlTypeCode::from_u32(ggml_code), rel_offset));
        }

        let data_base = cursor.div_ceil(32) * 32;
        let mut packer = KtbBankPacker::new(model_id);

        for i in 0..parsed_descriptors.len() {
            let (ref name, ggml_type, rel_start) = parsed_descriptors[i];
            let rel_end = if i + 1 < parsed_descriptors.len() {
                parsed_descriptors[i + 1].2
            } else {
                gguf_bytes.len().saturating_sub(data_base)
            };

            let (layer_idx, expert_idx) = SafetensorsToKtbTranscoder::parse_layer_and_expert(name);
            if layer_idx < start_layer || layer_idx > end_layer {
                continue;
            }

            let abs_start = data_base + rel_start;
            let abs_end = data_base + rel_end;
            packer.add_tensor_bank(
                name.clone(),
                layer_idx,
                expert_idx,
                ggml_type.to_ktb_encoding(),
                gguf_bytes[abs_start..abs_end].to_vec(),
            );
        }

        Ok(packer.pack_to_bytes())
    }
}

/// Content-addressed chunk descriptor used by CERN FTS3 / Davix peer-to-peer delta transfers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fts3ChunkDescriptor {
    pub chunk_index: u32,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub checksum_fnv64: u64,
}

/// Slices a `.ktb` binary image into content-addressed Davix/FTS3 chunks and computes delta
/// transfer lists so destination peers only fetch chunks not already in their local NVMe cache.
pub struct Fts3ChunkSlicer;

impl Fts3ChunkSlicer {
    pub fn compute_chunk_manifest(
        ktb_bytes: &[u8],
        chunk_size_bytes: usize,
    ) -> Vec<Fts3ChunkDescriptor> {
        let cs = chunk_size_bytes.max(65_536);
        let mut manifest = Vec::new();
        let mut offset = 0u64;

        for (idx, slice) in ktb_bytes.chunks(cs).enumerate() {
            manifest.push(Fts3ChunkDescriptor {
                chunk_index: idx as u32,
                byte_offset: offset,
                byte_len: slice.len() as u64,
                checksum_fnv64: compute_fnv1a64_checksum(slice),
            });
            offset += slice.len() as u64;
        }
        manifest
    }

    /// Returns only the `Fts3ChunkDescriptor`s from `target_manifest` whose checksums are missing
    /// or modified compared to `cached_manifest` on the destination peer.
    pub fn diff_missing_chunks(
        cached_manifest: &[Fts3ChunkDescriptor],
        target_manifest: &[Fts3ChunkDescriptor],
    ) -> Vec<Fts3ChunkDescriptor> {
        target_manifest
            .iter()
            .filter(|target| {
                !cached_manifest.iter().any(|cached| {
                    cached.chunk_index == target.chunk_index
                        && cached.checksum_fnv64 == target.checksum_fnv64
                        && cached.byte_len == target.byte_len
                })
            })
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::{ModelSpec, WorkloadKind, sample_personal_cluster};
    use kadmium_sharding::ShardPlanner;

    #[test]
    fn test_fts3_initial_staging_uses_home_nas_as_seed() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);
        let transfers = Fts3TransferScheduler::plan_initial_staging(&plan, &cluster);

        assert!(!transfers.is_empty());
        for job in &transfers {
            assert_eq!(job.source_node, NodeId("node-home-nas".into()));
            assert_ne!(job.destination_node, NodeId("node-home-nas".into()));
        }
    }

    #[test]
    fn test_ktb_binary_64kb_page_aligned_pack_and_zero_copy_read() {
        let mut packer = KtbBankPacker::new("deepseek-v3-671b-moe");
        packer.add_tensor_bank(
            "layers.0.attn.qkv_proj",
            0,
            None,
            TensorEncoding::Fp8E4M3,
            vec![0xAA; 12_345],
        );
        packer.add_tensor_bank(
            "layers.0.moe.experts.0.up_proj",
            0,
            Some(0),
            TensorEncoding::Q4KBlock,
            vec![0xBB; 70_000],
        );
        packer.add_tensor_bank(
            "layers.0.moe.experts.1.up_proj",
            0,
            Some(1),
            TensorEncoding::Q4KBlock,
            vec![0xCC; 32_768],
        );

        let image = packer.pack_to_bytes();
        // Total image size must be an exact multiple of 64 KiB (65,536 bytes).
        assert_eq!(image.len() % (KTB_PAGE_ALIGNMENT_BYTES as usize), 0);

        let reader = KtbBankReader::from_slice(&image).expect("valid KTB archive");
        assert_eq!(reader.header.entries.len(), 3);

        for entry in &reader.header.entries {
            // Every tensor entry must start on an exact 64 KiB page boundary!
            assert_eq!(entry.page_offset_bytes % KTB_PAGE_ALIGNMENT_BYTES, 0);
            let slice = reader
                .get_verified_tensor_slice(entry)
                .expect("verified zero-copy slice");
            assert_eq!(slice.len() as u64, entry.payload_bytes);
        }
    }

    #[test]
    fn test_safetensors_to_ktb_shard_transcoder() {
        let t0 = vec![0x11u8; 4096];
        let t1 = vec![0x22u8; 8192];
        let t2 = vec![0x33u8; 4096];

        let st_bytes = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
            (
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                vec![64, 32],
                &t0,
            ),
            (
                "model.layers.1.mlp.experts.3.up_proj.weight",
                "F8_E4M3",
                vec![128, 64],
                &t1,
            ),
            (
                "model.layers.2.self_attn.q_proj.weight",
                "BF16",
                vec![64, 32],
                &t2,
            ),
        ]);

        // Transcode only Layer 1..=1 into a .ktb shard bank:
        let ktb_bytes = SafetensorsToKtbTranscoder::transcode_shard_to_ktb(
            "deepseek-v3-671b-moe",
            &st_bytes,
            1,
            1,
        )
        .expect("transcode safetensors shard");

        let reader = KtbBankReader::from_slice(&ktb_bytes).expect("parse transcoded KTB");
        assert_eq!(reader.header.entries.len(), 1);
        let entry = &reader.header.entries[0];
        assert_eq!(entry.layer_index, 1);
        assert_eq!(entry.expert_index, Some(3));
        assert_eq!(entry.encoding, TensorEncoding::Fp8E4M3);
        let verified = reader
            .get_verified_tensor_slice(entry)
            .expect("verify slice");
        assert_eq!(verified, &t1[..]);
    }

    #[test]
    fn test_gguf_v3_to_ktb_transcoder_and_fts3_delta_chunks() {
        let b0 = vec![0x44u8; 65_536];
        let b1 = vec![0x55u8; 65_536];
        let gguf_bytes = GgufToKtbTranscoder::build_gguf_v3_bytes(&[
            ("blk.0.attn_q.weight", GgmlTypeCode::Q8_0, 65_536, &b0),
            (
                "blk.5.experts.2.ffn_up.weight",
                GgmlTypeCode::Q4K,
                65_536,
                &b1,
            ),
        ]);

        let ktb_bytes =
            GgufToKtbTranscoder::transcode_shard_to_ktb("qwen3-235b-moe", &gguf_bytes, 5, 5)
                .expect("transcode GGUF v3 shard");

        let reader = KtbBankReader::from_slice(&ktb_bytes).expect("parse GGUF->KTB archive");
        assert_eq!(reader.header.entries.len(), 1);
        assert_eq!(reader.header.entries[0].layer_index, 5);
        assert_eq!(reader.header.entries[0].expert_index, Some(2));
        assert_eq!(reader.header.entries[0].encoding, TensorEncoding::Q4KBlock);

        // Verify FTS3 chunk manifest & delta diffing
        let manifest_a = Fts3ChunkSlicer::compute_chunk_manifest(&ktb_bytes, 65_536);
        assert_eq!(manifest_a.len(), 2); // Page 0 header + Page 1 tensor bank

        let mut modified_ktb = ktb_bytes.clone();
        modified_ktb[65_536 + 10] ^= 0xFF; // mutate 1 byte in chunk #1
        let manifest_b = Fts3ChunkSlicer::compute_chunk_manifest(&modified_ktb, 65_536);

        let delta = Fts3ChunkSlicer::diff_missing_chunks(&manifest_a, &manifest_b);
        // Only chunk #1 changed; chunk #0 (Page 0 header) is already cached!
        assert_eq!(delta.len(), 1);
        assert_eq!(delta[0].chunk_index, 1);
    }
}
