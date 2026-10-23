//! Pre-merged MoE expert weight banks (`FTWX` sub-format) inside `.ktb` archives.
//!
//! Standard Safetensors and GGUF checkpoints store each routed MoE expert $(l, e)$ as three
//! separate tensors (`gate_proj`, `up_proj`, `down_proj`). [`KtbPreMergedBankPacker`] concatenates
//! those three projections into a single 64 KiB page-aligned `.ktb` entry prefixed by a 32-byte
//! `"FTWX"` descriptor so an expert can be verified and sliced from one contiguous page range.
//!
//! # On-Disk `FTWX` Payload Layout
//!
//! Each pre-merged expert entry is stored in the `.ktb` directory under the name
//! `layers.{layer}.moe.experts.{expert}.premerged_gate_up_down` with `expert_index = Some(expert)`
//! and `page_offset_bytes` aligned to [`KTB_PAGE_ALIGNMENT_BYTES`] (64 KiB).
//!
//! Within the entry's verified payload slice, all header fields are little-endian `u32`:
//!
//! | Byte Range | Field | Description |
//! | :--- | :--- | :--- |
//! | `0..4` | `magic` | ASCII `"FTWX"` ([`PREMERGED_EXPERT_MAGIC`]) |
//! | `4..8` | `layer_index` | Transformer layer index; must match the `.ktb` directory entry |
//! | `8..12` | `expert_index` | Routed expert index; must match the `.ktb` directory entry |
//! | `12..16` | `hidden_dim` | Model hidden dimension $d_{\text{model}}$ (`> 0`) |
//! | `16..20` | `intermediate_dim` | Expert intermediate dimension $d_{\text{ff}}$ (`> 0`) |
//! | `20..24` | `gate_len_bytes` | Byte length of `gate_proj` ($W_{\text{gate}}$) |
//! | `24..28` | `up_len_bytes` | Byte length of `up_proj` ($W_{\text{up}}$) |
//! | `28..32` | `down_len_bytes` | Byte length of `down_proj` ($W_{\text{down}}$) |
//! | `32..32 + gate_len` | `gate_proj` | Raw `gate_proj` weight bytes |
//! | `.. + up_len` | `up_proj` | Raw `up_proj` weight bytes |
//! | `.. + down_len` | `down_proj` | Raw `down_proj` weight bytes |
//!
//! The entry payload length must equal `32 + gate_len_bytes + up_len_bytes + down_len_bytes`.

use crate::{
    KTB_PAGE_ALIGNMENT_BYTES, KtbBankPacker, KtbBankReader, KtbPageAlignedTensorEntry,
    SafetensorFieldSpec, SafetensorsToKtbTranscoder, TensorEncoding,
};
use std::collections::BTreeMap;

/// Magic bytes (`"FTWX"`) at the start of a pre-merged expert payload inside a `.ktb` entry.
pub const PREMERGED_EXPERT_MAGIC: [u8; 4] = *b"FTWX";

/// Fixed 32-byte header length at the start of a pre-merged expert payload.
pub const PREMERGED_EXPERT_HEADER_BYTES: usize = 32;

/// Canonical `.ktb` entry name for a pre-merged expert bank at `(layer, expert)`.
pub fn premerged_expert_tensor_name(layer: u32, expert: u32) -> String {
    format!("layers.{layer}.moe.experts.{expert}.premerged_gate_up_down")
}

/// Borrowed slices into a verified 64 KiB page-aligned pre-merged expert entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreMergedExpertSlices<'a> {
    pub layer_index: u32,
    pub expert_index: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub encoding: TensorEncoding,
    pub page_offset_bytes: u64,
    pub contiguous_payload: &'a [u8],
    pub gate_proj: &'a [u8],
    pub up_proj: &'a [u8],
    pub down_proj: &'a [u8],
}

impl<'a> PreMergedExpertSlices<'a> {
    /// Parses and validates a verified `.ktb` entry slice into `(gate_proj, up_proj, down_proj)`
    /// sub-slices.
    pub fn from_entry_slice(
        entry: &KtbPageAlignedTensorEntry,
        slice: &'a [u8],
    ) -> Result<Self, String> {
        if slice.len() < PREMERGED_EXPERT_HEADER_BYTES {
            return Err(format!(
                "pre-merged expert {} is {} bytes, smaller than the 32-byte FTWX header",
                entry.tensor_name,
                slice.len()
            ));
        }
        if slice[0..4] != PREMERGED_EXPERT_MAGIC {
            return Err(format!(
                "pre-merged expert {} is missing the FTWX header magic",
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

        if layer_index != entry.layer_index || entry.expert_index != Some(expert_index) {
            return Err(format!(
                "pre-merged expert {} header (L{layer_index} E{expert_index}) disagrees with directory entry (L{} E{:?})",
                entry.tensor_name, entry.layer_index, entry.expert_index
            ));
        }
        if hidden_dim == 0 || intermediate_dim == 0 {
            return Err(format!(
                "pre-merged expert {} has zero dimension ({hidden_dim} x {intermediate_dim})",
                entry.tensor_name
            ));
        }

        let gate_start = PREMERGED_EXPERT_HEADER_BYTES;
        let up_start = gate_start
            .checked_add(gate_len)
            .ok_or_else(|| format!("pre-merged expert {} gate_len overflows usize", entry.tensor_name))?;
        let down_start = up_start
            .checked_add(up_len)
            .ok_or_else(|| format!("pre-merged expert {} up_len overflows usize", entry.tensor_name))?;
        let down_end = down_start
            .checked_add(down_len)
            .ok_or_else(|| format!("pre-merged expert {} down_len overflows usize", entry.tensor_name))?;

        if slice.len() != down_end {
            return Err(format!(
                "pre-merged expert {} payload is {} bytes, header expects {} bytes",
                entry.tensor_name,
                slice.len(),
                down_end
            ));
        }

        if entry.encoding == TensorEncoding::Fp32 {
            let expected_gate = (hidden_dim as usize)
                .checked_mul(4)
                .ok_or_else(|| format!("pre-merged expert {} hidden_dim overflows", entry.tensor_name))?;
            let expected_up = (intermediate_dim as usize)
                .checked_mul(4)
                .ok_or_else(|| format!("pre-merged expert {} intermediate_dim overflows", entry.tensor_name))?;
            let expected_down = expected_gate;
            if gate_len != expected_gate || up_len != expected_up || down_len != expected_down {
                return Err(format!(
                    "pre-merged Fp32 expert {} projection byte lengths ({gate_len}, {up_len}, {down_len}) do not match dimensions ({hidden_dim}, {intermediate_dim})",
                    entry.tensor_name
                ));
            }
        }

        Ok(Self {
            layer_index,
            expert_index,
            hidden_dim,
            intermediate_dim,
            encoding: entry.encoding,
            page_offset_bytes: entry.page_offset_bytes,
            contiguous_payload: slice,
            gate_proj: &slice[gate_start..up_start],
            up_proj: &slice[up_start..down_start],
            down_proj: &slice[down_start..down_end],
        })
    }

    /// Reads `(gate[idx], up[idx], down[idx])` as little-endian `f32`.
    #[inline]
    pub fn read_f32_triplet(&self, idx: usize) -> Option<(f32, f32, f32)> {
        let byte_off = idx.checked_mul(4)?;
        let end = byte_off.checked_add(4)?;
        let g_bytes = self.gate_proj.get(byte_off..end)?;
        let u_bytes = self.up_proj.get(byte_off..end)?;
        let d_bytes = self.down_proj.get(byte_off..end)?;
        let g = f32::from_le_bytes([g_bytes[0], g_bytes[1], g_bytes[2], g_bytes[3]]);
        let u = f32::from_le_bytes([u_bytes[0], u_bytes[1], u_bytes[2], u_bytes[3]]);
        let d = f32::from_le_bytes([d_bytes[0], d_bytes[1], d_bytes[2], d_bytes[3]]);
        Some((g, u, d))
    }

    /// Evaluates the elementwise SwiGLU expert contribution from an `Fp32` pre-merged bank:
    /// `routing_gate * swish(hidden[i] * gate[i]) * up[i] * down[i]`.
    ///
    /// Returns an error if the bank encoding is not [`TensorEncoding::Fp32`] or if the activation
    /// width does not match `hidden_dim` and `intermediate_dim`.
    pub fn evaluate_swiglu_f32(
        &self,
        hidden: &[f32],
        routing_gate: f32,
    ) -> Result<Vec<f32>, String> {
        if self.encoding != TensorEncoding::Fp32 {
            return Err(format!(
                "pre-merged expert L{} E{} has encoding {:?}, expected Fp32",
                self.layer_index, self.expert_index, self.encoding
            ));
        }
        let dim = hidden.len();
        if self.hidden_dim as usize != dim || self.intermediate_dim as usize != dim {
            return Err(format!(
                "pre-merged expert L{} E{} dimensions ({}, {}) do not match activation width {dim}",
                self.layer_index, self.expert_index, self.hidden_dim, self.intermediate_dim
            ));
        }
        let mut out = Vec::with_capacity(dim);
        for (i, &v) in hidden.iter().enumerate() {
            let (g, u, d) = self.read_f32_triplet(i).ok_or_else(|| {
                format!(
                    "pre-merged expert L{} E{} truncated at element {i}",
                    self.layer_index, self.expert_index
                )
            })?;
            let gated_in = v * g;
            let swish = gated_in / (1.0 + (-gated_in).exp());
            out.push(swish * u * d * routing_gate);
        }
        Ok(out)
    }
}

/// Builder that packs dense trunk tensors and pre-merged `FTWX` MoE expert banks into a 64 KiB
/// page-aligned `.ktb` archive.
pub struct KtbPreMergedBankPacker {
    inner: KtbBankPacker,
}

impl KtbPreMergedBankPacker {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            inner: KtbBankPacker::new(model_id),
        }
    }

    /// Adds a dense trunk tensor (such as `layer.rmsnorm.gamma`) at a 64 KiB page boundary.
    pub fn add_dense_trunk_tensor(
        &mut self,
        tensor_name: impl Into<String>,
        layer_index: u32,
        encoding: TensorEncoding,
        payload: Vec<u8>,
    ) {
        self.inner
            .add_tensor_bank(tensor_name, layer_index, None, encoding, payload);
    }

    /// Concatenates `gate_proj`, `up_proj`, and `down_proj` behind a 32-byte `"FTWX"` header and
    /// schedules the combined payload at a 64 KiB page boundary.
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
    }

    /// Packs an `f32` triplet `(gate_f32, up_f32, down_f32)` of equal width into an `Fp32`
    /// pre-merged expert bank.
    pub fn add_premerged_f32_expert(
        &mut self,
        layer_index: u32,
        expert_index: u32,
        gate_f32: &[f32],
        up_f32: &[f32],
        down_f32: &[f32],
    ) -> Result<(), String> {
        if gate_f32.len() != up_f32.len() || gate_f32.len() != down_f32.len() {
            return Err(format!(
                "pre-merged Fp32 expert L{layer_index} E{expert_index} slices differ in length ({}, {}, {})",
                gate_f32.len(),
                up_f32.len(),
                down_f32.len()
            ));
        }
        if gate_f32.is_empty() {
            return Err(format!(
                "pre-merged Fp32 expert L{layer_index} E{expert_index} must be non-empty"
            ));
        }
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
        let dim = u32::try_from(gate_f32.len())
            .map_err(|_| "pre-merged Fp32 expert dimension overflows u32".to_string())?;
        self.add_premerged_expert_bank(
            layer_index,
            expert_index,
            dim,
            dim,
            TensorEncoding::Fp32,
            &g,
            &u,
            &d,
        );
        Ok(())
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
        let total_len = PREMERGED_EXPERT_HEADER_BYTES
            + gate_proj.len()
            + up_proj.len()
            + down_proj.len();
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

    /// Fallibly packs all added tensors into a 64 KiB page-aligned `.ktb` binary image.
    pub fn try_pack_to_bytes(&self) -> Result<Vec<u8>, String> {
        self.inner.try_pack_to_bytes()
    }

    /// Packs all added tensors into a 64 KiB page-aligned `.ktb` binary image.
    pub fn pack_to_bytes(&self) -> Vec<u8> {
        self.inner.pack_to_bytes()
    }

    /// Transcodes a Safetensors buffer for `start_layer..=end_layer` into a `.ktb` archive,
    /// combining each routed expert's `gate_proj` (`w1` / `ffn_gate`), `up_proj` (`w3` / `ffn_up`),
    /// and `down_proj` (`w2` / `ffn_down`) tensors into one `FTWX` pre-merged expert bank.
    pub fn transcode_safetensors_to_premerged_ktb(
        model_id: &str,
        raw_safetensors_bytes: &[u8],
        start_layer: u32,
        end_layer: u32,
    ) -> Result<Vec<u8>, String> {
        if raw_safetensors_bytes.len() < 8 {
            return Err("Safetensors buffer too small (< 8 bytes)".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(
            raw_safetensors_bytes[0..8]
                .try_into()
                .map_err(|_| "Failed to read 8-byte safetensors header length")?,
        ))
        .map_err(|_| "Safetensors header length overflows usize")?;
        let data_base = 8usize
            .checked_add(header_len)
            .ok_or_else(|| "Safetensors header offset overflows usize".to_string())?;
        let header_slice = raw_safetensors_bytes
            .get(8..data_base)
            .ok_or_else(|| "Safetensors header length exceeds buffer size".to_string())?;

        let raw_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(header_slice)
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
            if start_off > end_off {
                return Err(format!("Tensor {key} has inverted offsets"));
            }
            let start = data_base
                .checked_add(start_off)
                .ok_or_else(|| format!("Tensor {key} start offset overflows usize"))?;
            let end = data_base
                .checked_add(end_off)
                .ok_or_else(|| format!("Tensor {key} end offset overflows usize"))?;
            let slice = raw_safetensors_bytes
                .get(start..end)
                .ok_or_else(|| format!("Tensor {key} offsets out of bounds"))?;

            let enc = match spec.dtype.as_str() {
                "F32" => TensorEncoding::Fp32,
                "BF16" => TensorEncoding::Bf16,
                "F8_E4M3" | "F8_E5M2" => TensorEncoding::Fp8E4M3,
                "I8" | "U8" => TensorEncoding::Int8Symmetric,
                _ => TensorEncoding::Fp16,
            };

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
                } else if lower.contains("up_proj")
                    || lower.contains("w3")
                    || lower.contains("ffn_up")
                {
                    entry.3 = slice.to_vec();
                } else {
                    return Err(format!(
                        "Unrecognized expert projection tensor {key} for L{layer} E{exp_id}"
                    ));
                }
            } else {
                packer.add_dense_trunk_tensor(key.clone(), layer, enc, slice.to_vec());
            }
        }

        for ((layer, exp_id), (enc, shape, gate, up, down)) in expert_groups {
            if gate.is_empty() || up.is_empty() || down.is_empty() {
                return Err(format!(
                    "Incomplete expert projection triplet for L{layer} E{exp_id} (gate={} B, up={} B, down={} B)",
                    gate.len(),
                    up.len(),
                    down.len()
                ));
            }
            let hidden_dim = shape.last().copied().unwrap_or(64).max(1) as u32;
            let inter_dim = shape.first().copied().unwrap_or(64).max(1) as u32;
            packer.add_premerged_expert_bank(
                layer,
                exp_id,
                hidden_dim,
                inter_dim,
                enc,
                &gate,
                &up,
                &down,
            );
        }

        packer.try_pack_to_bytes()
    }
}

/// Reader over a `.ktb` slice that indexes all `FTWX` pre-merged expert entries by `(layer, expert)`.
pub struct KtbPreMergedBankReader<'a> {
    pub inner: KtbBankReader<'a>,
    expert_slot_index: BTreeMap<(u32, u32), usize>,
}

impl<'a> KtbPreMergedBankReader<'a> {
    /// Parses a `.ktb` archive, bounds-checking every directory entry and indexing all `FTWX`
    /// pre-merged expert entries by `(layer_index, expert_index)`.
    pub fn from_slice(raw: &'a [u8]) -> Result<Self, String> {
        let inner = KtbBankReader::from_slice(raw)?;
        let mut expert_slot_index = BTreeMap::new();

        for (idx, entry) in inner.header.entries.iter().enumerate() {
            if !entry
                .page_offset_bytes
                .is_multiple_of(KTB_PAGE_ALIGNMENT_BYTES)
                || entry.page_offset_bytes < KTB_PAGE_ALIGNMENT_BYTES
            {
                return Err(format!(
                    "Tensor {} offset {} is not 64 KiB page-aligned",
                    entry.tensor_name, entry.page_offset_bytes
                ));
            }
            let start = usize::try_from(entry.page_offset_bytes)
                .map_err(|_| format!("Tensor {} offset overflows usize", entry.tensor_name))?;
            let len = usize::try_from(entry.payload_bytes)
                .map_err(|_| format!("Tensor {} length overflows usize", entry.tensor_name))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| format!("Tensor {} span overflows usize", entry.tensor_name))?;
            let entry_slice = raw.get(start..end).ok_or_else(|| {
                format!(
                    "Tensor {} slice ({start}..{end}) exceeds KTB archive bounds ({})",
                    entry.tensor_name,
                    raw.len()
                )
            })?;

            let has_ftwx_magic = entry_slice.get(0..4) == Some(&PREMERGED_EXPERT_MAGIC);
            let has_ftwx_name = entry.tensor_name.ends_with(".premerged_gate_up_down");
            if has_ftwx_magic || has_ftwx_name {
                let exp_id = entry.expert_index.ok_or_else(|| {
                    format!(
                        "Pre-merged expert entry {} is missing expert_index",
                        entry.tensor_name
                    )
                })?;
                if expert_slot_index
                    .insert((entry.layer_index, exp_id), idx)
                    .is_some()
                {
                    return Err(format!(
                        "Duplicate pre-merged expert entry for L{} E{exp_id}",
                        entry.layer_index
                    ));
                }
            }
        }

        Ok(Self {
            inner,
            expert_slot_index,
        })
    }

    /// Number of pre-merged `(layer, expert)` banks indexed in this archive.
    pub fn premerged_expert_count(&self) -> usize {
        self.expert_slot_index.len()
    }

    /// Sorted list of `expert_id`s with a pre-merged bank resident for `layer`.
    pub fn resident_experts_for_layer(&self, layer: u32) -> Vec<u32> {
        self.expert_slot_index
            .keys()
            .filter_map(|&(l, e)| if l == layer { Some(e) } else { None })
            .collect()
    }

    /// Looks up `(layer, expert)`, verifies its 64 KiB page alignment and FNV-1a checksum, and
    /// validates the `"FTWX"` sub-header and projection slice bounds.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KTB_MAGIC_BYTES, KtbBinaryArchiveHeader};

    #[test]
    fn premerged_expert_bank_packs_page_aligned_and_validates_on_read() {
        let mut packer = KtbPreMergedBankPacker::new("deepseek-v3-671b-moe");
        let dim = 64usize;
        let gate: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let up: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.02).collect();
        let down: Vec<f32> = (0..dim).map(|i| -0.1 + (i as f32) * 0.005).collect();

        packer.add_premerged_f32_expert(2, 0, &gate, &up, &down).unwrap();
        packer.add_premerged_f32_expert(2, 5, &gate, &up, &down).unwrap();

        let ktb_bytes = packer.try_pack_to_bytes().unwrap();
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
        assert_eq!(exp5.encoding, TensorEncoding::Fp32);
        assert_eq!(exp5.gate_proj.len(), dim * 4);
        assert_eq!(exp5.up_proj.len(), dim * 4);
        assert_eq!(exp5.down_proj.len(), dim * 4);

        let (g0, u0, d0) = exp5.read_f32_triplet(0).unwrap();
        assert!((g0 - gate[0]).abs() < 1e-6);
        assert!((u0 - up[0]).abs() < 1e-6);
        assert!((d0 - down[0]).abs() < 1e-6);

        // Wrong activation width is rejected rather than padded:
        assert!(exp5.evaluate_swiglu_f32(&[0.5; 32], 1.0).is_err());

        // Checksum corruption is surfaced as an error:
        let mut corrupted = ktb_bytes.clone();
        let corrupt_at = exp5.page_offset_bytes as usize + PREMERGED_EXPERT_HEADER_BYTES + 4;
        corrupted[corrupt_at] ^= 0xFF;
        let bad_reader = KtbPreMergedBankReader::from_slice(&corrupted).unwrap();
        assert!(bad_reader.get_premerged_expert(2, 5).is_err());
    }

    #[test]
    fn reader_rejects_overflowing_header_offsets_without_panicking() {
        let crafted = KtbBinaryArchiveHeader {
            model_id: "crafted".into(),
            format_version: 1,
            page_alignment_bytes: KTB_PAGE_ALIGNMENT_BYTES,
            entries: vec![KtbPageAlignedTensorEntry {
                tensor_name: premerged_expert_tensor_name(0, 0),
                layer_index: 0,
                expert_index: Some(0),
                encoding: TensorEncoding::Fp32,
                page_offset_bytes: u64::MAX - (u64::MAX % KTB_PAGE_ALIGNMENT_BYTES),
                payload_bytes: 128,
                padded_reservation_bytes: KTB_PAGE_ALIGNMENT_BYTES,
                checksum_fnv64: 0,
            }],
        };
        let json = serde_json::to_vec(&crafted).unwrap();
        let mut buf = Vec::with_capacity(KTB_PAGE_ALIGNMENT_BYTES as usize);
        buf.extend_from_slice(&KTB_MAGIC_BYTES);
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        buf.resize(KTB_PAGE_ALIGNMENT_BYTES as usize, 0u8);

        assert!(KtbPreMergedBankReader::from_slice(&buf).is_err());
        let base_reader = KtbBankReader::from_slice(&buf).unwrap();
        assert!(
            base_reader
                .get_verified_tensor_slice(&base_reader.header.entries[0])
                .is_err()
        );
    }

    #[test]
    fn multi_page_header_directory_packs_hundreds_of_experts_without_panicking() {
        let mut packer = KtbPreMergedBankPacker::new("large-moe");
        let weights = [1.0f32; 16];
        // 512 entries produce a >100 KiB JSON directory, spanning multiple 64 KiB header pages.
        for layer in 0..8u32 {
            for expert in 0..64u32 {
                packer
                    .add_premerged_f32_expert(layer, expert, &weights, &weights, &weights)
                    .unwrap();
            }
        }
        let ktb = packer.try_pack_to_bytes().expect("pack 512 experts");
        assert_eq!(ktb.len() % (KTB_PAGE_ALIGNMENT_BYTES as usize), 0);

        let reader = KtbPreMergedBankReader::from_slice(&ktb).expect("read multi-page header KTB");
        assert_eq!(reader.premerged_expert_count(), 512);
        let last = reader
            .get_premerged_expert(7, 63)
            .expect("verify L7 E63")
            .expect("L7 E63 present");
        assert_eq!(last.page_offset_bytes % KTB_PAGE_ALIGNMENT_BYTES, 0);
        assert!(last.page_offset_bytes >= 2 * KTB_PAGE_ALIGNMENT_BYTES);
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
