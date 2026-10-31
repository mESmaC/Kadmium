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

pub mod storage_pool;
pub use storage_pool::*;
pub mod checkpoint;

/// Magic header bytes for the Kadmium Tensor Bank (`.ktb`) file format (`"KDTB"`).
pub const KTB_MAGIC_BYTES: [u8; 4] = *b"KDTB";

/// Default page alignment (64 KiB) for zero-copy `mmap` / `io_uring` / `IoRing` NVMe streaming.
pub const KTB_PAGE_ALIGNMENT_BYTES: u64 = 65_536;

/// Quantization and tensor encoding format inside a `.ktb` bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorEncoding {
    Fp32,
    Fp16,
    Bf16,
    Fp8E4M3,
    Int8Symmetric,
    Q4KBlock,
    Nf4DoubleQuant,
    /// EXL3 / QTIP (arXiv:2406.11235) Hadamard Incoherence + Bitshift Trellis-Coded Quantization (`2.0–6.0 bpw`).
    Exl3QtpTrellis,
    /// Original GGML block type. The bytes are not Kadmium's reference Q4/Int8 codec.
    Ggml(u32),
    /// Unsigned bytes, distinct from symmetric quantized weights.
    Uint8,
    /// Raw signed bytes, without an implied quantization scale.
    Int8,
    Fp64,
    Fp8E5M2,
    /// Power-of-two scale bytes used by microscaling (MX) block formats.
    Fp8E8M0,
    /// Six-bit floats packed without padding (four elements per three bytes).
    Fp6E2M3,
    Fp6E3M2,
    /// Four-bit floats packed two per byte (SafeTensors `F4`).
    Fp4E2M1,
    Bool,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Complex64,
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
    pub fn pack_to_bytes(&self) -> Result<Vec<u8>, String> {
        let page_size = KTB_PAGE_ALIGNMENT_BYTES as usize;
        let mut current_offset = page_size as u64; // Page 0 reserved for header directory
        let mut entries = Vec::with_capacity(self.tensors.len());

        for t in &self.tensors {
            let payload_len = t.data.len();
            let padded_len = payload_len
                .max(1)
                .checked_add(page_size - 1)
                .ok_or("KTB tensor size overflow")?
                / page_size
                * page_size;
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

            current_offset = current_offset
                .checked_add(padded_len as u64)
                .ok_or("KTB archive size overflow")?;
        }

        let header = KtbBinaryArchiveHeader {
            model_id: self.model_id.clone(),
            format_version: 1,
            page_alignment_bytes: KTB_PAGE_ALIGNMENT_BYTES,
            entries,
        };

        let header_json = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
        if header_json.len() > page_size - 10 {
            return Err(
                "KTB header directory exceeds 64 KiB Page 0; import fewer tensors per shard".into(),
            );
        }

        let capacity = usize::try_from(current_offset).map_err(|_| "KTB archive too large")?;
        let mut out = Vec::new();
        out.try_reserve_exact(capacity)
            .map_err(|e| format!("Cannot allocate KTB archive: {e}"))?;
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

        Ok(out)
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
        let header_end = 10usize
            .checked_add(json_len)
            .filter(|end| *end <= KTB_PAGE_ALIGNMENT_BYTES as usize)
            .ok_or("KTB header exceeds Page 0 bounds")?;
        let header: KtbBinaryArchiveHeader =
            serde_json::from_slice(raw.get(10..header_end).ok_or("Truncated KTB header")?)
                .map_err(|e| format!("Invalid KTB header JSON: {e}"))?;
        for entry in &header.entries {
            Self::tensor_range(entry, raw.len())?;
        }
        Ok(Self { raw, header })
    }

    fn tensor_range(
        entry: &KtbPageAlignedTensorEntry,
        archive_len: usize,
    ) -> Result<std::ops::Range<usize>, String> {
        let start = usize::try_from(entry.page_offset_bytes)
            .map_err(|_| "Tensor offset exceeds address space")?;
        let len = usize::try_from(entry.payload_bytes)
            .map_err(|_| "Tensor length exceeds address space")?;
        let end = start.checked_add(len).ok_or("Tensor range overflow")?;
        if start < KTB_PAGE_ALIGNMENT_BYTES as usize
            || end > archive_len
            || !entry
                .page_offset_bytes
                .is_multiple_of(KTB_PAGE_ALIGNMENT_BYTES)
        {
            return Err("Tensor slice exceeds KTB archive bounds or alignment".into());
        }
        Ok(start..end)
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
        let range = Self::tensor_range(entry, self.raw.len())?;
        let slice = self
            .raw
            .get(range)
            .ok_or("Tensor slice exceeds KTB archive bounds")?;
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

pub mod batch_read;
pub mod expert;
pub mod importer;
pub mod integrations;
pub mod interdevice;
pub mod merkle;
pub mod moe_expert_pack;
pub mod range_pull;
pub mod safetensors_manifest;
pub mod transcoder;

pub use batch_read::*;
pub use expert::*;
pub use importer::*;
pub use integrations::*;
pub use merkle::*;
pub use moe_expert_pack::*;
pub use range_pull::*;
pub use safetensors_manifest::*;
pub use transcoder::*;

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

        let image = packer
            .pack_to_bytes()
            .expect("bounded reference KTB fixture");
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
        assert_eq!(reader.header.entries[0].encoding, TensorEncoding::Ggml(12));

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

    #[test]
    fn test_huggingface_and_github_ecosystem_integrations() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);

        let hf_plan = HuggingFaceHubConnector::plan_selective_model_pull(
            "hf://deepseek-ai/DeepSeek-V3@main",
            &plan,
        )
        .expect("plan hf selective pull");
        assert_eq!(hf_plan.hf_ref.repo_id, "deepseek-ai/DeepSeek-V3");
        assert!(hf_plan.plan_only);
        assert_eq!(hf_plan.files_downloaded, 0);
        assert_eq!(hf_plan.ktb_banks_compiled, 0);
        assert_eq!(hf_plan.index_source, "unverified_missing_index_manifest");
        assert_eq!(hf_plan.seed_vault_node, plan.assignments[0].hostname);
        assert!(!hf_plan.node_pulls.is_empty());
        for np in &hf_plan.node_pulls {
            assert!(np.required_safetensors_files.is_empty());
            assert!(np.full_repo_avoided_gb > 150.0);
            assert!(np.output_ktb_bank_path.ends_with(".ktb"));
        }

        // Dataset URIs must not be planned as .safetensors/.ktb models and must not invent split filenames
        assert!(
            HuggingFaceHubConnector::plan_selective_model_pull(
                "hf://datasets/open-r1/OpenR1-Math-220k@main",
                &plan
            )
            .is_err()
        );
        let ds_plan = HuggingFaceHubConnector::plan_dataset_pull(
            "hf://datasets/open-r1/OpenR1-Math-220k@main",
        )
        .expect("plan dataset pull");
        assert_eq!(ds_plan.hf_ref.kind, HfRepoKind::Dataset);
        assert!(ds_plan.dataset_info_url.contains("dataset_infos.json"));
        assert!(ds_plan.parquet_split_urls.is_empty());
        assert_eq!(ds_plan.files_downloaded, 0);

        let ds_with_splits = HuggingFaceHubConnector::plan_dataset_pull_with_splits(
            "hf://datasets/open-r1/OpenR1-Math-220k@main",
            &["data/train-00000-of-00001.parquet"],
        )
        .expect("plan dataset pull with explicit split");
        assert_eq!(ds_with_splits.parquet_split_urls.len(), 1);

        // Real parsed index manifest populates exact shard files
        let single_shard_index = SafetensorsIndexManifest {
            total_size_bytes: 14_000_000_000,
            weight_map: (0..=plan.total_layers)
                .map(|l| {
                    (
                        format!("model.layers.{l}.self_attn.q_proj.weight"),
                        "model.safetensors".to_string(),
                    )
                })
                .collect(),
        };
        let indexed_plan = HuggingFaceHubConnector::plan_selective_model_pull_with_index(
            "hf://Qwen/Qwen2.5-7B-Instruct@main",
            &plan,
            Some(&single_shard_index),
        )
        .expect("plan with parsed index");
        assert_eq!(indexed_plan.index_source, "parsed_safetensors_index");
        for np in &indexed_plan.node_pulls {
            assert_eq!(
                np.required_safetensors_files,
                vec!["model.safetensors".to_string()]
            );
        }

        let gh_run = GitHubConnector::evaluate_repo_on_slice(
            "gh://zhengkid/Dream-RSI@a4f91c2#worlds/gpu_kernel_opt",
        )
        .expect("evaluate github repo on slice");
        assert_eq!(gh_run.gh_ref.owner, "zhengkid");
        assert_eq!(gh_run.gh_ref.repo, "Dream-RSI");
        assert_eq!(gh_run.gh_ref.git_ref, "a4f91c2");
        assert_eq!(gh_run.pinned_commit_sha, "unverified_unmounted");
        assert_eq!(gh_run.assigned_slice_id, "slice-04-rl-dream-rsi");
        assert!(!gh_run.mounted);
        assert_eq!(gh_run.execution_mode, "plan_only_unmounted");
        assert_eq!(gh_run.verified_node_count, 0);
        assert!(!gh_run.verde_verification_passed);
        assert_eq!(gh_run.measured_tokens_per_sec, 0.0);
        assert_eq!(gh_run.dream_rsi_best_speedup, 0.0);
        assert!(
            GitHubConnector::stage_local_repo(
                "gh://nonexistent-owner-404/nonexistent-repo-404@deadbeef",
                None
            )
            .is_err()
        );
    }

    #[test]
    fn test_ktb_merkle_tree_proof_and_in_place_page_patching() {
        let mut packer = KtbBankPacker::new("deepseek-v3-671b-moe");
        for i in 0..4u32 {
            packer.add_tensor_bank(
                format!("layers.{i}.moe.experts.0.up_proj"),
                i,
                Some(0),
                TensorEncoding::Q4KBlock,
                vec![(i as u8).wrapping_add(0x30); 65_536],
            );
        }
        let ktb_image = packer
            .pack_to_bytes()
            .expect("bounded reference KTB fixture");
        let tree = KtbMerkleTree::from_ktb_bytes(&ktb_image);
        assert_eq!(tree.leaf_hashes.len(), 5); // 1 header page + 4 tensor pages

        // Prove and verify page #3
        let proof = tree.prove_page(3).expect("proof for page 3");
        let page_size = KTB_PAGE_ALIGNMENT_BYTES as usize;
        let page_3_slice = &ktb_image[3 * page_size..4 * page_size];
        assert!(KtbMerkleTree::verify_page(
            page_3_slice,
            &proof,
            tree.root_hash
        ));

        // Corrupt 1 byte and verify Merkle proof rejects tampered peer chunk
        let mut corrupted_page = page_3_slice.to_vec();
        corrupted_page[1024] ^= 0xFF;
        assert!(!KtbMerkleTree::verify_page(
            &corrupted_page,
            &proof,
            tree.root_hash
        ));

        // Verify Fts3PagePatchEngine in-place patching
        let mut worker_buf = vec![0u8; ktb_image.len()];
        Fts3PagePatchEngine::verify_and_patch_page(
            &mut worker_buf,
            page_3_slice,
            &proof,
            tree.root_hash,
        )
        .expect("patch verified page 3");
        assert_eq!(&worker_buf[3 * page_size..4 * page_size], page_3_slice);
    }

    #[test]
    fn test_hf_http_byte_range_slicer_and_index_manifest_parser() {
        let index_json = r#"{
            "metadata": { "total_size": 685000000000 },
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00016.safetensors",
                "model.layers.14.mlp.experts.2.up_proj.weight": "model-00004-of-00016.safetensors"
            }
        }"#;
        let idx = HfByteRangeSlicer::parse_index_manifest(index_json).expect("parse index json");
        let files = HfByteRangeSlicer::filter_required_shard_files(&idx, 14, 15);
        assert_eq!(files, vec!["model-00004-of-00016.safetensors"]);

        let raw_st = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
            (
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                vec![16, 16],
                &vec![0x11; 512],
            ),
            (
                "model.layers.14.mlp.experts.2.up_proj.weight",
                "F8_E4M3",
                vec![32, 32],
                &vec![0x42; 1024],
            ),
        ]);
        let slices = HfByteRangeSlicer::plan_http_range_requests(
            "deepseek-ai/DeepSeek-V3",
            "main",
            "model-00004-of-00016.safetensors",
            &raw_st,
            14,
            14,
        )
        .expect("plan range slices");
        assert_eq!(slices.len(), 1);
        assert!(slices[0].http_range_header.starts_with("bytes="));
        assert_eq!(slices[0].payload_len_bytes, 1024);

        // Two adjacent tensors of one file share one range request; a far one does not.
        let both = HfByteRangeSlicer::plan_http_range_requests(
            "deepseek-ai/DeepSeek-V3",
            "main",
            "model-00004-of-00016.safetensors",
            &raw_st,
            0,
            14,
        )
        .expect("plan both");
        assert_eq!(both.len(), 2);
        let merged = HfByteRangeSlicer::coalesce_range_requests(&both, 0);
        assert_eq!(merged.len(), 1, "{merged:?}");
        assert_eq!(merged[0].members.len(), 2);
        assert_eq!(merged[0].len(), 512 + 1024);
        assert_eq!(merged[0].gap_bytes, 0);
        let (a, b) = merged[0].slice_bounds(&both[merged[0].members[1]]);
        assert_eq!(b - a, both[merged[0].members[1]].payload_len_bytes);
        let mut far = both.clone();
        far[1].byte_start += 100_000;
        far[1].byte_end_inclusive += 100_000;
        far[1].http_range_header =
            format!("bytes={}-{}", far[1].byte_start, far[1].byte_end_inclusive);
        assert_eq!(
            HfByteRangeSlicer::coalesce_range_requests(&far, 1024).len(),
            2
        );
        let bridged = HfByteRangeSlicer::coalesce_range_requests(&far, 200_000);
        assert_eq!(bridged.len(), 1);
        assert_eq!(bridged[0].gap_bytes, 100_000);

        let ktb = HfByteRangeSlicer::assemble_range_slices_to_ktb(
            "deepseek-v3-671b-moe",
            &raw_st,
            &slices,
        )
        .expect("assemble ktb from byte ranges");
        let reader = KtbBankReader::from_slice(&ktb).expect("read ktb");
        assert_eq!(reader.header.entries.len(), 1);
        assert_eq!(reader.header.entries[0].layer_index, 14);
    }
}
