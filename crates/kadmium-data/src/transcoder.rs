//! Safetensors & GGUF v3 Binary Shard Transcoders and CERN FTS3 Content-Addressed Chunk Slicer
//! (`kadmium-data::transcoder`).

use crate::{KtbBankPacker, TensorEncoding, compute_fnv1a64_checksum};
use serde::{Deserialize, Serialize};

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
