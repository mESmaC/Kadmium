//! Live HTTP Byte-Range (`Range: bytes=start-end`) Safetensors Slicer & Index Manifest Parser (`kadmium-data::range_pull`).
//!
//! Enables a Kadmium worker node assigned layers `[start_layer..=end_layer]` to:
//! 1. Parse `model.safetensors.index.json` to identify which `.safetensors` shard files contain its assigned layers.
//! 2. Fetch only the initial header bytes (`Range: bytes=0-65535`) of each relevant `.safetensors` file.
//! 3. Compute exact HTTP byte-range headers (`Range: bytes=A-B`) for individual layer/expert tensors
//!    and stream them directly into a 64 KiB page-aligned `.ktb` Tensor Bank without downloading unwanted layers.

use crate::{KtbBankPacker, SafetensorsToKtbTranscoder, TensorEncoding};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Parsed Hugging Face `model.safetensors.index.json` manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsIndexManifest {
    pub total_size_bytes: u64,
    /// Maps tensor name (e.g., `"model.layers.14.self_attn.q_proj.weight"`) -> shard file (`"model-00004-of-00016.safetensors"`).
    pub weight_map: BTreeMap<String, String>,
}

/// Exact HTTP `Range: bytes=start-end` descriptor for a single tensor slice inside a remote `.safetensors` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpByteRangeSlice {
    pub tensor_name: String,
    pub layer_index: u32,
    pub expert_index: Option<u32>,
    pub shard_filename: String,
    pub url: String,
    /// RFC 9110 HTTP Range header value (e.g., `"bytes=248-8439"`).
    pub http_range_header: String,
    pub byte_start: u64,
    pub byte_end_inclusive: u64,
    pub payload_len_bytes: usize,
    pub target_encoding: TensorEncoding,
}

/// One HTTP request covering several planned slices whose byte ranges are contiguous or within
/// `max_gap` of each other: one round trip and one TCP ramp-up instead of one per tensor, the
/// way InferSSD merges contiguous expert ranges into one read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRangeRequest {
    pub url: String,
    pub http_range_header: String,
    pub byte_start: u64,
    pub byte_end_inclusive: u64,
    /// Positions in the planned slice list this request serves, in byte order.
    pub members: Vec<usize>,
    /// Bytes inside the request no slice asked for.
    pub gap_bytes: u64,
}

impl HttpRangeRequest {
    pub fn len(&self) -> u64 {
        self.byte_end_inclusive - self.byte_start + 1
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// The bytes of `slice` (one of `members`) inside this request's body.
    pub fn slice_bounds(&self, slice: &HttpByteRangeSlice) -> (usize, usize) {
        let start = (slice.byte_start - self.byte_start) as usize;
        (start, start + slice.payload_len_bytes)
    }
}

/// Plans and executes selective HTTP `Range: bytes=...` pulls from Hugging Face `.safetensors` repositories.
pub struct HfByteRangeSlicer;

impl HfByteRangeSlicer {
    /// Parses a Hugging Face `model.safetensors.index.json` document and filters which shard filenames
    /// are needed for `[start_layer..=end_layer]`.
    pub fn parse_index_manifest(json_str: &str) -> Result<SafetensorsIndexManifest, String> {
        let root: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("Invalid index JSON: {e}"))?;

        let total_size_bytes = root
            .get("metadata")
            .and_then(|m| m.get("total_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let wm_obj = root
            .get("weight_map")
            .and_then(|w| w.as_object())
            .ok_or("Missing weight_map object in model.safetensors.index.json")?;

        let mut weight_map = BTreeMap::new();
        for (tensor_name, file_val) in wm_obj {
            if let Some(fname) = file_val.as_str() {
                weight_map.insert(tensor_name.clone(), fname.to_string());
            }
        }

        Ok(SafetensorsIndexManifest {
            total_size_bytes,
            weight_map,
        })
    }

    /// Returns the deduplicated list of `.safetensors` shard filenames required by a node hosting `[start_layer..=end_layer]`.
    pub fn filter_required_shard_files(
        manifest: &SafetensorsIndexManifest,
        start_layer: u32,
        end_layer: u32,
    ) -> Vec<String> {
        let mut files = Vec::new();
        for (tensor_name, shard_file) in &manifest.weight_map {
            let (layer_idx, _) = SafetensorsToKtbTranscoder::parse_layer_and_expert(tensor_name);
            if layer_idx >= start_layer && layer_idx <= end_layer && !files.contains(shard_file) {
                files.push(shard_file.clone());
            }
        }
        files
    }

    /// Inspects the binary header of a `.safetensors` file (`8-byte LE header_len + JSON header`)
    /// and computes exact `Range: bytes=start-end` HTTP slice descriptors for layers `[start_layer..=end_layer]`.
    pub fn plan_http_range_requests(
        repo_id: &str,
        revision: &str,
        shard_filename: &str,
        safetensors_header_bytes: &[u8],
        start_layer: u32,
        end_layer: u32,
    ) -> Result<Vec<HttpByteRangeSlice>, String> {
        if safetensors_header_bytes.len() < 8 {
            return Err("Safetensors header probe shorter than 8 bytes".into());
        }
        let header_len =
            u64::from_le_bytes(safetensors_header_bytes[0..8].try_into().unwrap()) as usize;
        let data_base_offset = (8 + header_len) as u64;

        if safetensors_header_bytes.len() < 8 + header_len {
            return Err("Safetensors header probe does not contain full JSON header".into());
        }

        let header_json: serde_json::Value =
            serde_json::from_slice(&safetensors_header_bytes[8..8 + header_len])
                .map_err(|e| format!("Invalid Safetensors JSON header: {e}"))?;
        let obj = header_json
            .as_object()
            .ok_or("Safetensors header root must be a JSON object")?;

        let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/{shard_filename}");
        let mut slices = Vec::new();

        for (tensor_name, meta) in obj {
            if tensor_name == "__metadata__" {
                continue;
            }
            let (layer_idx, expert_idx) =
                SafetensorsToKtbTranscoder::parse_layer_and_expert(tensor_name);
            if layer_idx < start_layer || layer_idx > end_layer {
                continue;
            }

            let offsets = meta
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or("Missing data_offsets array")?;
            if offsets.len() != 2 {
                return Err("Malformed data_offsets length".into());
            }
            let rel_start = offsets[0].as_u64().ok_or("Invalid start offset")?;
            let rel_end = offsets[1].as_u64().ok_or("Invalid end offset")?;
            if rel_end <= rel_start {
                continue;
            }

            let abs_start = data_base_offset + rel_start;
            let abs_end_inclusive = data_base_offset + rel_end - 1;
            let payload_len = (rel_end - rel_start) as usize;

            let dtype = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("F16");
            let target_encoding = match dtype {
                "BF16" => TensorEncoding::Bf16,
                "F8_E4M3" => TensorEncoding::Fp8E4M3,
                "I8" | "U8" => TensorEncoding::Q4KBlock,
                _ => TensorEncoding::Fp16,
            };

            slices.push(HttpByteRangeSlice {
                tensor_name: tensor_name.clone(),
                layer_index: layer_idx,
                expert_index: expert_idx,
                shard_filename: shard_filename.to_string(),
                url: url.clone(),
                http_range_header: format!("bytes={abs_start}-{abs_end_inclusive}"),
                byte_start: abs_start,
                byte_end_inclusive: abs_end_inclusive,
                payload_len_bytes: payload_len,
                target_encoding,
            });
        }

        Ok(slices)
    }

    /// Merges planned slices of one file into as few HTTP range requests as possible: slices
    /// whose byte ranges touch or sit within `max_gap_bytes` of each other share a request.
    /// Slices of different URLs never merge. Requests come back in byte order per URL.
    pub fn coalesce_range_requests(
        slices: &[HttpByteRangeSlice],
        max_gap_bytes: u64,
    ) -> Vec<HttpRangeRequest> {
        let mut order: Vec<usize> = (0..slices.len()).collect();
        order.sort_by(|&a, &b| {
            slices[a]
                .url
                .cmp(&slices[b].url)
                .then(slices[a].byte_start.cmp(&slices[b].byte_start))
        });
        let mut requests: Vec<HttpRangeRequest> = Vec::new();
        for index in order {
            let s = &slices[index];
            match requests.last_mut() {
                Some(last)
                    if last.url == s.url
                        && s.byte_start
                            <= last.byte_end_inclusive.saturating_add(1 + max_gap_bytes) =>
                {
                    if s.byte_start > last.byte_end_inclusive + 1 {
                        last.gap_bytes += s.byte_start - last.byte_end_inclusive - 1;
                    }
                    last.byte_end_inclusive = last.byte_end_inclusive.max(s.byte_end_inclusive);
                    last.http_range_header =
                        format!("bytes={}-{}", last.byte_start, last.byte_end_inclusive);
                    last.members.push(index);
                }
                _ => requests.push(HttpRangeRequest {
                    url: s.url.clone(),
                    http_range_header: format!("bytes={}-{}", s.byte_start, s.byte_end_inclusive),
                    byte_start: s.byte_start,
                    byte_end_inclusive: s.byte_end_inclusive,
                    members: vec![index],
                    gap_bytes: 0,
                }),
            }
        }
        requests
    }

    /// Extracts the planned `[byte_start..=byte_end_inclusive]` byte ranges from `safetensors_bytes`
    /// and packs them directly into a 64 KiB page-aligned `.ktb` Tensor Bank image.
    pub fn assemble_range_slices_to_ktb(
        model_id: &str,
        safetensors_bytes: &[u8],
        slices: &[HttpByteRangeSlice],
    ) -> Result<Vec<u8>, String> {
        let mut packer = KtbBankPacker::new(model_id);
        for s in slices {
            let start = s.byte_start as usize;
            let end_excl = (s.byte_end_inclusive + 1) as usize;
            if end_excl > safetensors_bytes.len() {
                return Err(format!(
                    "Range slice {}..{} exceeds buffer length {}",
                    start,
                    end_excl,
                    safetensors_bytes.len()
                ));
            }
            packer.add_tensor_bank(
                s.tensor_name.clone(),
                s.layer_index,
                s.expert_index,
                s.target_encoding,
                safetensors_bytes[start..end_excl].to_vec(),
            );
        }
        packer.pack_to_bytes()
    }
}
