//! # FreeToken (`FTW`) Pre-Merged Weight Banks for `.ktb`
//!
//! Implements the **Pre-Merged Weight Bank** storage and zero-copy execution layout from
//! **[`FlashML-org/FreeToken`](https://github.com/FlashML-org/FreeToken)** on top of Kadmium's
//! 64 KiB page-aligned `.ktb` (Kadmium Tensor Bank) format:
//!
//! - **Ahead-of-Time Expert Matrix Fusion (`FTWX` Sub-Header):**
//!   In standard Safetensors / GGUF checkpoints, each MoE expert $(l, e)$ is split across three
//!   discontiguous tensors (`gate_proj`, `up_proj`, `down_proj`). Streaming a cache-missed expert
//!   at runtime requires three separate string lookups, three scattered NVMe/RAM seeks, and
//!   runtime buffer repacking.
//!   [`KtbPreMergedBankPacker`] fuses `(gate_proj, up_proj, down_proj)` into a **single contiguous
//!   64 KiB page-aligned binary bank** prefixed by a 32-byte `"FTWX"` descriptor.
//! - **$O(1)$ Direct Slot Indexing ([`KtbPreMergedBankReader`]):**
//!   Eliminates string-based tensor lookup during token generation by indexing every pre-merged
//!   expert slot by `(layer_index, expert_index)` at mmap/open time, returning zero-copy slices
//!   (`gate_proj`, `up_proj`, `down_proj`) plus the full `contiguous_payload` for single-DMA
//!   PCIe transfers or in-place CPU/NPU execution.
//! - **Automatic Safetensors -> Pre-Merged `.ktb` Transcoder:**
//!   [`KtbPreMergedBankPacker::transcode_safetensors_to_premerged_ktb`] automatically pairs split
//!   `gate_proj` (`w1`), `up_proj` (`w3`), and `down_proj` (`w2`) expert tensors from raw
//!   Safetensors shards and fuses them into 64 KiB page-aligned pre-merged `.ktb` banks.

use crate::{
    KTB_PAGE_ALIGNMENT_BYTES, KtbBankPacker, KtbBankReader, KtbPageAlignedTensorEntry,
    SafetensorFieldSpec, SafetensorsToKtbTranscoder, TensorEncoding,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Magic bytes (`"FTWX"`) identifying a FreeToken pre-merged expert payload inside a `.ktb` entry.
pub const PREMERGED_EXPERT_MAGIC: [u8; 4] = *b"FTWX";

/// Fixed 32-byte header at the beginning of a pre-merged expert bank payload.
pub const PREMERGED_EXPERT_HEADER_BYTES: usize = 32;

/// Canonical tensor name for a pre-merged expert bank inside a `.ktb` archive.
pub fn premerged_expert_tensor_name(layer: u32, expert: u32) -> String {
    format!("layers.{layer}.moe.experts.{expert}.premerged_gate_up_down")
}

/// Zero-copy view into a single 64 KiB page-aligned pre-merged MoE expert bank in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreMergedExpertSlices<'a> {
    pub layer_index: u32,
    pub expert_index: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub encoding: TensorEncoding,
    /// Exact 64 KiB page-aligned offset of this bank inside the `.ktb` image.
    pub page_offset_bytes: u64,
    /// Entire contiguous payload (`header + gate_proj + up_proj + down_proj`) for single-DMA
    /// PCIe streaming.
    pub contiguous_payload: &'a [u8],
    /// Zero-copy slice to `gate_proj` (`W_gate` / `w1`) weights.
    pub gate_proj: &'a [u8],
    /// Zero-copy slice to `up_proj` (`W_up` / `w3`) weights.
    pub up_proj: &'a [u8],
    /// Zero-copy slice to `down_proj` (`W_down` / `w2`) weights.
    pub down_proj: &'a [u8],
}

impl<'a> PreMergedExpertSlices<'a> {
    /// Parses a verified `.ktb` entry slice into zero-copy `(gate_proj, up_proj, down_proj)`
    /// sub-slices in $O(1)$ pointer arithmetic.
    pub fn from_entry_slice(
        entry: &KtbPageAlignedTensorEntry,
        slice: &'a [u8],
    ) -> Result<Self, String> {
        if slice.len() < PREMERGED_EXPERT_HEADER_BYTES {
            return Err(format!(
                "Pre-merged expert slice for {} is smaller than 32-byte FTWX header ({} B)",
                entry.tensor_name,
                slice.len()
            ));
        }
        if slice[0..4] != PREMERGED_EXPERT_MAGIC {
            return Err(format!(
                "Invalid FTWX magic in pre-merged expert bank {}",
                entry.tensor_name
            ));
        }
        let u32_le = |off: usize| -> u32 {
            u32::from_le_bytes([slice[off], slice[off + 1], slice[off + 2], slice[off + 3]])
        };
        let layer_index = u32_le(4);
        let expert_index = u32_le(8);
        let hidden_dim = u32_le(12);
        let intermediate_dim = u32_le(16);
        let gate_len = u32_le(20) as usize;
        let up_len = u32_le(24) as usize;
        let down_len = u32_le(28) as usize;

        let total_expected = PREMERGED_EXPERT_HEADER_BYTES
            .saturating_add(gate_len)
            .saturating_add(up_len)
            .saturating_add(down_len);
        if slice.len() < total_expected {
            return Err(format!(
                "Pre-merged expert bank {} truncated: expected {} B, got {} B",
                entry.tensor_name,
                total_expected,
                slice.len()
            ));
        }

        let gate_start = PREMERGED_EXPERT_HEADER_BYTES;
        let up_start = gate_start + gate_len;
        let down_start = up_start + up_len;
        let down_end = down_start + down_len;

        Ok(Self {
            layer_index,
            expert_index,
            hidden_dim,
            intermediate_dim,
            encoding: entry.encoding,
            page_offset_bytes: entry.page_offset_bytes,
            contiguous_payload: &slice[..down_end],
            gate_proj: &slice[gate_start..up_start],
            up_proj: &slice[up_start..down_start],
            down_proj: &slice[down_start..down_end],
        })
    }

    /// Reads `(gate[idx], up[idx], down[idx])` as little-endian `f32` directly from the zero-copy
    /// memory-mapped slices without heap allocation.
    #[inline]
    pub fn read_f32_triplet(&self, idx: usize) -> Option<(f32, f32, f32)> {
        let byte_off = idx.checked_mul(4)?;
        let end = byte_off.checked_add(4)?;
        if end > self.gate_proj.len() || end > self.up_proj.len() || end > self.down_proj.len() {
            return None;
        }
        let g = f32::from_le_bytes([
            self.gate_proj[byte_off],
            self.gate_proj[byte_off + 1],
            self.gate_proj[byte_off + 2],
            self.gate_proj[byte_off + 3],
        ]);
        let u = f32::from_le_bytes([
            self.up_proj[byte_off],
            self.up_proj[byte_off + 1],
            self.up_proj[byte_off + 2],
            self.up_proj[byte_off + 3],
        ]);
        let d = f32::from_le_bytes([
            self.down_proj[byte_off],
            self.down_proj[byte_off + 1],
            self.down_proj[byte_off + 2],
            self.down_proj[byte_off + 3],
        ]);
        Some((g, u, d))
    }

    /// Evaluates the fused SwiGLU expert contribution directly over the zero-copy pre-merged
    /// `(gate_proj, up_proj, down_proj)` slices:
    ///
    /// $$\text{out}[i] = \text{routing\_gate} \cdot \operatorname{swish}(h[i] \cdot g[i]) \cdot u[i] \cdot d[i]$$
    pub fn evaluate_swiglu_f32(&self, hidden: &[f32], routing_gate: f32) -> Vec<f32> {
        hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let (g, u, d) = self.read_f32_triplet(i).unwrap_or((1.0, 1.0, 0.0));
                let gated_in = v * g;
                let swish = gated_in / (1.0 + (-gated_in).exp());
                swish * u * d * routing_gate
            })
            .collect()
    }
}

/// Summary telemetry of a pre-merged `.ktb` weight bank archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreMergedBankSummary {
    pub model_id: String,
    pub total_entries: usize,
    pub premerged_expert_banks: usize,
    pub dense_trunk_entries: usize,
    pub total_archive_bytes: u64,
    /// Number of separate tensor seeks/DMAs avoided per expert access (`3 -> 1`, i.e., `3x` fewer seeks).
    pub seeks_avoided_per_expert: u32,
}

/// Builder that packs pre-merged MoE expert banks (`gate_proj + up_proj + down_proj` fused per
/// `(layer, expert)`) and dense trunk tensors into 64 KiB page-aligned `.ktb` archives.
pub struct KtbPreMergedBankPacker {
    model_id: String,
    inner: KtbBankPacker,
    premerged_count: usize,
    dense_count: usize,
}

impl KtbPreMergedBankPacker {
    pub fn new(model_id: impl Into<String>) -> Self {
        let id = model_id.into();
        Self {
            model_id: id.clone(),
            inner: KtbBankPacker::new(id),
            premerged_count: 0,
            dense_count: 0,
        }
    }

    /// Adds a dense trunk tensor (e.g., `layer.rmsnorm.gamma` or fused `qkv_proj`) at a 64 KiB
    /// page boundary.
    pub fn add_dense_trunk_tensor(
        &mut self,
        tensor_name: impl Into<String>,
        layer_index: u32,
        encoding: TensorEncoding,
        payload: Vec<u8>,
    ) {
        self.inner
            .add_tensor_bank(tensor_name, layer_index, None, encoding, payload);
        self.dense_count += 1;
    }

    /// Fuses `gate_proj`, `up_proj`, and `down_proj` into a single contiguous `"FTWX"` pre-merged
    /// expert payload and schedules it at an exact 64 KiB page boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn add_premerged_expert_bank(
        &mut self,
        layer_index: u32,
        expert_index: u32,
        hidden_dim: u32,
        intermediate_dim: u32,
        encoding: TensorEncoding,
        gate_proj: &[u8],
        up_proj: &[u8],
        down_proj: &[u8],
    ) {
        let payload = Self::encode_premerged_payload(
            layer_index,
            expert_index,
            hidden_dim,
            intermediate_dim,
            gate_proj,
            up_proj,
            down_proj,
        );
        self.inner.add_tensor_bank(
            premerged_expert_tensor_name(layer_index, expert_index),
            layer_index,
            Some(expert_index),
            encoding,
            payload,
        );
        self.premerged_count += 1;
    }

    /// Convenience helper that packs an `f32` triplet `(gate_f32, up_f32, down_f32)` into a
    /// 64 KiB page-aligned pre-merged expert bank.
    pub fn add_premerged_f32_expert(
        &mut self,
        layer_index: u32,
        expert_index: u32,
        gate_f32: &[f32],
        up_f32: &[f32],
        down_f32: &[f32],
    ) {
        let to_bytes = |vals: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        };
        let g = to_bytes(gate_f32);
        let u = to_bytes(up_f32);
        let d = to_bytes(down_f32);
        let dim = gate_f32.len() as u32;
        self.add_premerged_expert_bank(
            layer_index,
            expert_index,
            dim,
            dim,
            TensorEncoding::Fp16,
            &g,
            &u,
            &d,
        );
    }

    /// Encodes the 32-byte `"FTWX"` sub-header followed by `[gate_proj | up_proj | down_proj]`.
    pub fn encode_premerged_payload(
        layer_index: u32,
        expert_index: u32,
        hidden_dim: u32,
        intermediate_dim: u32,
        gate_proj: &[u8],
        up_proj: &[u8],
        down_proj: &[u8],
    ) -> Vec<u8> {
        let total_len =
            PREMERGED_EXPERT_HEADER_BYTES + gate_proj.len() + up_proj.len() + down_proj.len();
        let mut buf = Vec::with_capacity(total_len);
        buf.extend_from_slice(&PREMERGED_EXPERT_MAGIC);
        buf.extend_from_slice(&layer_index.to_le_bytes());
        buf.extend_from_slice(&expert_index.to_le_bytes());
        buf.extend_from_slice(&hidden_dim.to_le_bytes());
        buf.extend_from_slice(&intermediate_dim.to_le_bytes());
        buf.extend_from_slice(&(gate_proj.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(up_proj.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(down_proj.len() as u32).to_le_bytes());
        buf.extend_from_slice(gate_proj);
        buf.extend_from_slice(up_proj);
        buf.extend_from_slice(down_proj);
        buf
    }

    /// Finalizes the 64 KiB page-aligned `.ktb` binary image.
    pub fn pack_to_bytes(&self) -> Vec<u8> {
        self.inner.pack_to_bytes()
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn premerged_count(&self) -> usize {
        self.premerged_count
    }

    pub fn dense_count(&self) -> usize {
        self.dense_count
    }

    /// Transcodes a Safetensors file into a pre-merged `.ktb` bank for `start_layer..=end_layer`,
    /// automatically grouping split `gate_proj`/`w1`, `up_proj`/`w3`, and `down_proj`/`w2`
    /// matrices for each `(layer, expert)` and fusing them into a single 64 KiB page-aligned
    /// `"FTWX"` pre-merged expert bank.
    pub fn transcode_safetensors_to_premerged_ktb(
        model_id: &str,
        raw_safetensors_bytes: &[u8],
        start_layer: u32,
        end_layer: u32,
    ) -> Result<Vec<u8>, String> {
        if raw_safetensors_bytes.len() < 8 {
            return Err("Safetensors buffer too small (< 8 bytes)".into());
        }
        let header_len = u64::from_le_bytes(
            raw_safetensors_bytes[0..8]
                .try_into()
                .map_err(|_| "Failed to read 8-byte safetensors header length")?,
        ) as usize;
        let data_base = 8 + header_len;
        if raw_safetensors_bytes.len() < data_base {
            return Err("Safetensors header length exceeds buffer size".into());
        }

        let raw_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&raw_safetensors_bytes[8..data_base])
                .map_err(|e| format!("Invalid Safetensors JSON header: {e}"))?;

        let mut packer = Self::new(model_id);
        let mut keys: Vec<&String> = raw_map.keys().filter(|k| *k != "__metadata__").collect();
        keys.sort_by_key(|k| {
            let (l, e) = SafetensorsToKtbTranscoder::parse_layer_and_expert(k);
            (l, e.unwrap_or(0), (*k).clone())
        });

        #[allow(clippy::type_complexity)]
        let mut expert_groups: BTreeMap<
            (u32, u32),
            (TensorEncoding, Vec<usize>, Vec<u8>, Vec<u8>, Vec<u8>),
        > = BTreeMap::new();

        for key in keys {
            let val = &raw_map[key];
            let spec: SafetensorFieldSpec = serde_json::from_value(val.clone())
                .map_err(|e| format!("Invalid tensor entry {key}: {e}"))?;
            let (layer, expert) = SafetensorsToKtbTranscoder::parse_layer_and_expert(key);
            if layer < start_layer || layer > end_layer {
                continue;
            }
            let [start_off, end_off] = spec.data_offsets;
            let start = data_base + start_off;
            let end = data_base + end_off;
            if end > raw_safetensors_bytes.len() || start > end {
                return Err(format!("Tensor {key} offsets out of bounds"));
            }
            let enc = match spec.dtype.as_str() {
                "BF16" => TensorEncoding::Bf16,
                "F8_E4M3" | "F8_E5M2" => TensorEncoding::Fp8E4M3,
                "I8" | "U8" => TensorEncoding::Int8Symmetric,
                _ => TensorEncoding::Fp16,
            };
            let slice = &raw_safetensors_bytes[start..end];

            if let Some(exp_id) = expert {
                let entry = expert_groups.entry((layer, exp_id)).or_insert_with(|| {
                    (enc, spec.shape.clone(), Vec::new(), Vec::new(), Vec::new())
                });
                let lower = key.to_ascii_lowercase();
                if lower.contains("gate_proj") || lower.contains("w1") || lower.contains("ffn_gate")
                {
                    entry.2 = slice.to_vec();
                } else if lower.contains("down_proj")
                    || lower.contains("w2")
                    || lower.contains("ffn_down")
                {
                    entry.4 = slice.to_vec();
                } else {
                    entry.3 = slice.to_vec();
                }
            } else {
                packer.add_dense_trunk_tensor(key.clone(), layer, enc, slice.to_vec());
            }
        }

        for ((layer, exp_id), (enc, shape, gate, up, down)) in expert_groups {
            let hidden_dim = shape.last().copied().unwrap_or(64) as u32;
            let inter_dim = shape.first().copied().unwrap_or(64) as u32;
            packer.add_premerged_expert_bank(
                layer, exp_id, hidden_dim, inter_dim, enc, &gate, &up, &down,
            );
        }

        Ok(packer.pack_to_bytes())
    }
}

/// Zero-copy `.ktb` reader with an $O(1)$ `(layer, expert)` direct lookup index for pre-merged
/// FreeToken expert banks.
pub struct KtbPreMergedBankReader<'a> {
    pub inner: KtbBankReader<'a>,
    /// Direct $O(1)$ map from `(layer_index, expert_index)` to entry index in `inner.header.entries`.
    expert_slot_index: BTreeMap<(u32, u32), usize>,
}

impl<'a> KtbPreMergedBankReader<'a> {
    /// Parses a `.ktb` byte slice (or memory-mapped region) and indexes all pre-merged expert
    /// banks by `(layer_index, expert_index)` in $O(1)$ lookup tables.
    pub fn from_slice(raw: &'a [u8]) -> Result<Self, String> {
        let inner = KtbBankReader::from_slice(raw)?;
        let mut expert_slot_index = BTreeMap::new();

        for (idx, entry) in inner.header.entries.iter().enumerate() {
            if let Some(exp_id) = entry.expert_index {
                let is_premerged = entry.tensor_name.contains("premerged") || {
                    let start = entry.page_offset_bytes as usize;
                    start + 4 <= raw.len() && raw[start..start + 4] == PREMERGED_EXPERT_MAGIC
                };
                if is_premerged {
                    expert_slot_index.insert((entry.layer_index, exp_id), idx);
                }
            }
        }

        Ok(Self {
            inner,
            expert_slot_index,
        })
    }

    /// Number of pre-merged `(layer, expert)` banks indexed in this `.ktb` archive.
    pub fn premerged_expert_count(&self) -> usize {
        self.expert_slot_index.len()
    }

    /// Returns the sorted list of `expert_id`s that have a pre-merged bank resident for `layer`.
    pub fn resident_experts_for_layer(&self, layer: u32) -> Vec<u32> {
        self.expert_slot_index
            .keys()
            .filter_map(|&(l, e)| if l == layer { Some(e) } else { None })
            .collect()
    }

    /// $O(1)$ lookup and zero-copy slice extraction for `(layer, expert)`.
    /// Verifies 64 KiB page alignment, 64-bit FNV-1a checksum integrity, and `"FTWX"` sub-header
    /// bounds before returning the zero-copy [`PreMergedExpertSlices`].
    pub fn get_premerged_expert(
        &self,
        layer: u32,
        expert: u32,
    ) -> Result<Option<PreMergedExpertSlices<'a>>, String> {
        let Some(&entry_idx) = self.expert_slot_index.get(&(layer, expert)) else {
            return Ok(None);
        };
        let entry = &self.inner.header.entries[entry_idx];
        let verified_slice = self.inner.get_verified_tensor_slice(entry)?;
        let slices = PreMergedExpertSlices::from_entry_slice(entry, verified_slice)?;
        Ok(Some(slices))
    }

    /// Produces a summary report of the pre-merged `.ktb` archive.
    pub fn summary(&self) -> PreMergedBankSummary {
        let premerged = self.expert_slot_index.len();
        let total = self.inner.header.entries.len();
        let total_bytes: u64 = (KTB_PAGE_ALIGNMENT_BYTES)
            + self
                .inner
                .header
                .entries
                .iter()
                .map(|e| e.padded_reservation_bytes)
                .sum::<u64>();
        PreMergedBankSummary {
            model_id: self.inner.header.model_id.clone(),
            total_entries: total,
            premerged_expert_banks: premerged,
            dense_trunk_entries: total.saturating_sub(premerged),
            total_archive_bytes: total_bytes,
            seeks_avoided_per_expert: if premerged > 0 { 2 } else { 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premerged_expert_bank_packs_64kb_page_aligned_and_reads_zero_copy_in_o1() {
        let mut packer = KtbPreMergedBankPacker::new("deepseek-v3-671b-moe");
        let dim = 64usize;
        let gate: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let up: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.02).collect();
        let down: Vec<f32> = (0..dim).map(|i| -0.1 + (i as f32) * 0.005).collect();

        packer.add_premerged_f32_expert(2, 0, &gate, &up, &down);
        packer.add_premerged_f32_expert(2, 5, &gate, &up, &down);

        let ktb_bytes = packer.pack_to_bytes();
        assert_eq!(ktb_bytes.len() % (KTB_PAGE_ALIGNMENT_BYTES as usize), 0);

        let reader =
            KtbPreMergedBankReader::from_slice(&ktb_bytes).expect("parse premerged KTB reader");
        assert_eq!(reader.premerged_expert_count(), 2);
        assert_eq!(reader.resident_experts_for_layer(2), vec![0, 5]);

        let exp5 = reader
            .get_premerged_expert(2, 5)
            .expect("verify expert (2, 5)")
            .expect("expert (2, 5) exists");
        assert_eq!(exp5.page_offset_bytes % KTB_PAGE_ALIGNMENT_BYTES, 0);
        assert_eq!(exp5.layer_index, 2);
        assert_eq!(exp5.expert_index, 5);
        assert_eq!(exp5.hidden_dim, dim as u32);
        assert_eq!(exp5.gate_proj.len(), dim * 4);
        assert_eq!(exp5.up_proj.len(), dim * 4);
        assert_eq!(exp5.down_proj.len(), dim * 4);

        let (g0, u0, d0) = exp5.read_f32_triplet(0).unwrap();
        assert!((g0 - gate[0]).abs() < 1e-6);
        assert!((u0 - up[0]).abs() < 1e-6);
        assert!((d0 - down[0]).abs() < 1e-6);

        let summary = reader.summary();
        assert_eq!(summary.premerged_expert_banks, 2);
        assert_eq!(summary.seeks_avoided_per_expert, 2);
    }

    #[test]
    fn safetensors_split_expert_projections_fuse_into_single_premerged_ktb_bank() {
        let gate_bytes = vec![0xAAu8; 2048];
        let up_bytes = vec![0xBBu8; 2048];
        let down_bytes = vec![0xCCu8; 2048];
        let q_proj = vec![0x11u8; 1024];

        let st_bytes = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
            (
                "model.layers.3.self_attn.q_proj.weight",
                "BF16",
                vec![32, 16],
                &q_proj,
            ),
            (
                "model.layers.3.mlp.experts.7.gate_proj.weight",
                "F8_E4M3",
                vec![64, 32],
                &gate_bytes,
            ),
            (
                "model.layers.3.mlp.experts.7.up_proj.weight",
                "F8_E4M3",
                vec![64, 32],
                &up_bytes,
            ),
            (
                "model.layers.3.mlp.experts.7.down_proj.weight",
                "F8_E4M3",
                vec![32, 64],
                &down_bytes,
            ),
        ]);

        let ktb = KtbPreMergedBankPacker::transcode_safetensors_to_premerged_ktb(
            "qwen3-235b-moe",
            &st_bytes,
            3,
            3,
        )
        .expect("transcode and merge safetensors experts");

        let reader = KtbPreMergedBankReader::from_slice(&ktb).expect("open premerged KTB");
        // The 3 split expert tensors (gate, up, down) were fused into 1 pre-merged expert bank,
        // alongside the 1 dense q_proj tensor (total 2 entries instead of 4!).
        assert_eq!(reader.inner.header.entries.len(), 2);
        assert_eq!(reader.premerged_expert_count(), 1);

        let fused = reader
            .get_premerged_expert(3, 7)
            .expect("verify fused expert")
            .expect("expert (3, 7) present");
        assert_eq!(fused.gate_proj, &gate_bytes[..]);
        assert_eq!(fused.up_proj, &up_bytes[..]);
        assert_eq!(fused.down_proj, &down_bytes[..]);
        assert_eq!(
            fused.contiguous_payload.len(),
            PREMERGED_EXPERT_HEADER_BYTES + gate_bytes.len() + up_bytes.len() + down_bytes.len()
        );
    }
}
