//! Per-layer reference math shared by every execution shape.
//!
//! A transformer layer is split into the three pieces the grid schedules separately: the trunk
//! (RMSNorm, RoPE, attention over the layer's KV page, and the router), the expert contributions
//! (one per selected expert, each a function of the trunk output alone), and the combine (the
//! residual plus the contributions in expert-id order, clamped, then quantized to INT8 at the
//! layer boundary). A chain stage folds the three per layer on one node; the fan-out shape runs
//! them as three kinds of work unit on different nodes. The boundary is quantized in both shapes
//! and partials cross the wire as exact `f32`, so the two shapes produce identical bytes, which
//! is what the equivalence tests check.
//!
//! The kernels are reference implementations on small synthetic tensors, like the rest of the
//! runtime today; real weights plug in behind the same three functions.
//!
//! Every reduction and transcendental in the layer goes through [`crate::repops`], so the bytes
//! at a boundary do not depend on the backend that produced them, and every boundary carries a
//! RepOps digest ([`LayerReference::boundary_digest`]): the commitment a Verde bisection
//! compares layer by layer.

use crate::{
    BlockwiseInt8Tensor, FusedRmsNormRopeKernel, KvPageBlock, MoEGatingRouter,
    MoERoutedExpertSelection, PagedGqaAttentionKernel, TokenBatchInt8, repops,
};
use kadmium_data::KtbBankReader;
use kadmium_grid::LayerDigest;
use serde::{Deserialize, Serialize};

/// Block size of the INT8 quantization applied at every layer boundary.
pub const LAYER_BOUNDARY_BLOCK: usize = 16;
pub const ROPE_THETA: f32 = 10_000.0;
pub const RMSNORM_EPS: f32 = 1e-5;
/// Residual weight of the attention output.
pub const ATTENTION_RESIDUAL_SCALE: f32 = 0.25;

pub struct LayerReference;

impl LayerReference {
    /// The RMSNorm scale from the resident `.ktb` bank, or the synthetic ramp when the bank has
    /// none.
    pub fn gamma_from_bank(bank: Option<&[u8]>, dim: usize) -> Vec<f32> {
        let mut gamma: Vec<f32> = Vec::new();
        if let Some(ktb_buf) = bank
            && let Ok(reader) = KtbBankReader::from_slice(ktb_buf)
            && let Some(entry) = reader
                .header
                .entries
                .iter()
                .find(|e| e.tensor_name == "layer.rmsnorm.gamma")
            && let Ok(raw_slice) = reader.get_verified_tensor_slice(entry)
        {
            for chunk in raw_slice.chunks_exact(4).take(dim) {
                gamma.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        if gamma.len() < dim {
            gamma = (0..dim).map(|i| 1.0 + (i as f32) * 0.002).collect();
        }
        gamma
    }

    /// Hot and warm expert cutoffs for the router's locality prior, from the node's q* ratio.
    pub fn routing_cutoffs(experts_per_layer: u32, q_star: f64) -> (u32, u32) {
        let q = q_star.clamp(0.15, 0.96);
        let total = experts_per_layer as f64;
        (
            (total * q * 0.55).round() as u32,
            (total * (q * 0.85).min(0.92)).round() as u32,
        )
    }

    /// The layer's synthetic 4-token KV page.
    pub fn kv_page(layer: u32, dim: usize) -> KvPageBlock {
        KvPageBlock {
            page_id: layer,
            tokens_stored: 4,
            head_dim: dim,
            keys: (0..(4 * dim))
                .map(|i| repops::cos((i as f32) * 0.03 + (layer as f32) * 0.1) * 0.4)
                .collect(),
            values: (0..(4 * dim))
                .map(|i| repops::sin((i as f32) * 0.05 + (layer as f32) * 0.07) * 0.4)
                .collect(),
        }
    }

    /// Trunk of one layer: RMSNorm and RoPE at position `layer + 1`, attention over the layer's KV
    /// page with a residual, then the router. Returns the trunk output and the routing.
    pub fn trunk_forward(
        hidden: &[f32],
        layer: u32,
        gamma: &[f32],
        experts_per_layer: u32,
        top_k: u32,
        q_star: f64,
    ) -> (Vec<f32>, Vec<MoERoutedExpertSelection>) {
        let dim = hidden.len().max(16);
        let mut h = hidden.to_vec();
        if h.len() < dim {
            h.resize(dim, 0.1);
        }
        let gamma: Vec<f32> = if gamma.len() >= dim {
            gamma[..dim].to_vec()
        } else {
            let mut g = gamma.to_vec();
            g.extend((g.len()..dim).map(|i| 1.0 + (i as f32) * 0.002));
            g
        };
        let mut h = FusedRmsNormRopeKernel::forward(
            &h,
            &gamma,
            layer as usize + 1,
            ROPE_THETA,
            RMSNORM_EPS,
        );
        let attention =
            PagedGqaAttentionKernel::forward_flash_paged(&h, &[Self::kv_page(layer, dim)]);
        for (x, a) in h.iter_mut().zip(attention.iter()) {
            *x += a * ATTENTION_RESIDUAL_SCALE;
        }
        let routing = if experts_per_layer > 0 && top_k > 0 {
            let (hot, warm) = Self::routing_cutoffs(experts_per_layer, q_star);
            MoEGatingRouter::route_top_k(&h, layer, experts_per_layer, top_k, hot, warm)
        } else {
            Vec::new()
        };
        (h, routing)
    }

    /// Synthetic weight of expert `expert` at `layer` for hidden index `idx`.
    pub fn expert_weight(layer: u32, expert: u32, idx: usize) -> f32 {
        ((((layer as usize) * 19 + (expert as usize) * 23 + idx * 7) % 53) as f32 - 26.0) * 0.004
    }

    /// One expert's contribution to the layer output: `gate * swish(h) * w_expert`.
    pub fn expert_contribution(hidden: &[f32], layer: u32, expert: u32, gate: f32) -> Vec<f32> {
        hidden
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let swish = repops::silu(v);
                swish * Self::expert_weight(layer, expert, i) * gate
            })
            .collect()
    }

    /// The residual plus every contribution in expert-id order, clamped. Per element the terms
    /// `[residual, partial of the lowest expert id, ...]` go through the RepOps reduce tree, so
    /// the sum is the same whether the partials came from one node or many, in any order.
    pub fn combine(hidden: &[f32], partials: &[(u32, Vec<f32>)]) -> Vec<f32> {
        let mut ordered: Vec<&(u32, Vec<f32>)> = partials.iter().collect();
        ordered.sort_by_key(|(expert, _)| *expert);
        let mut terms: Vec<f32> = Vec::with_capacity(ordered.len() + 1);
        hidden
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                terms.clear();
                terms.push(h);
                terms.extend(ordered.iter().filter_map(|(_, p)| p.get(i).copied()));
                repops::tree_sum_f32(&terms).clamp(-8.0, 8.0)
            })
            .collect()
    }

    /// Quantization applied at every layer boundary, on the wire or not.
    pub fn boundary_quantize(out: &[f32]) -> BlockwiseInt8Tensor {
        BlockwiseInt8Tensor::quantize(out, LAYER_BOUNDARY_BLOCK)
    }

    /// Trunk, every selected expert, and combine, on one node. Returns the layer output before
    /// boundary quantization and the routing.
    pub fn layer_forward(
        hidden_in: &[f32],
        layer: u32,
        gamma: &[f32],
        experts_per_layer: u32,
        top_k: u32,
        q_star: f64,
    ) -> (Vec<f32>, Vec<MoERoutedExpertSelection>) {
        let (h, routing) =
            Self::trunk_forward(hidden_in, layer, gamma, experts_per_layer, top_k, q_star);
        let partials: Vec<(u32, Vec<f32>)> = routing
            .iter()
            .map(|r| {
                (
                    r.expert_id,
                    Self::expert_contribution(&h, layer, r.expert_id, r.gating_weight),
                )
            })
            .collect();
        (Self::combine(&h, &partials), routing)
    }

    /// Runs layers `start..=end` from a dequantized boundary, quantizing at every layer boundary
    /// and recording each boundary's digest. `hook` sees every layer's output before the
    /// boundary quantization; it is how a test or a demo injects a fault at a known layer, and
    /// how a dishonest node would.
    #[allow(clippy::too_many_arguments)]
    pub fn run_layers(
        hidden_in: &[f32],
        start: u32,
        end: u32,
        gamma: &[f32],
        experts_per_layer: u32,
        top_k: u32,
        q_star: f64,
        mut hook: Option<LayerHook<'_>>,
    ) -> LayerRun {
        let mut hidden = hidden_in.to_vec();
        let mut boundary = None;
        let mut layer_digests = Vec::new();
        let mut routed = Vec::new();
        for layer in start..=end {
            let (mut out, routing) =
                Self::layer_forward(&hidden, layer, gamma, experts_per_layer, top_k, q_star);
            if let Some(hook) = hook.as_mut() {
                hook(layer, &mut out);
            }
            routed.push((layer, routing.iter().map(|r| r.expert_id).collect()));
            let q = Self::boundary_quantize(&out);
            layer_digests.push(LayerDigest::new(layer, Self::boundary_digest(layer, &q)));
            hidden = q.dequantize();
            boundary = Some(q);
        }
        LayerRun {
            hidden,
            boundary,
            layer_digests,
            routed,
        }
    }

    /// The RepOps digest of one layer boundary: the manifest id, the layer, and the exact INT8
    /// codes and block scales. No floating-point reduction is involved, so two nodes that
    /// produced the same bytes produce the same digest, and nothing else does.
    pub fn boundary_digest(layer: u32, boundary: &BlockwiseInt8Tensor) -> u64 {
        let mut digest = repops::Digest::new("layer-boundary");
        digest
            .update_u32(layer)
            .update_u32(boundary.block_size as u32)
            .update_u32(boundary.total_elements as u32)
            .update_f32s(&boundary.absmax_scales)
            .update_i8s(&boundary.quantized_data);
        digest.finish()
    }

    /// The boundary digest of a batch: one row's digest, or the fold of the rows' digests in row
    /// order, which is what a stage running the rows one at a time records for the same layer.
    pub fn batch_boundary_digest(layer: u32, batch: &TokenBatchInt8) -> u64 {
        match batch.row_tensors() {
            Some(rows) if rows.len() > 1 => {
                let digests: Vec<u64> = rows
                    .iter()
                    .map(|row| Self::boundary_digest(layer, row))
                    .collect();
                Self::fold_row_digests(&digests)
            }
            _ => Self::boundary_digest(layer, &batch.tensor),
        }
    }

    /// Folds per-row boundary digests of one layer into the layer's digest for the batch.
    pub fn fold_row_digests(row_digests: &[u64]) -> u64 {
        if row_digests.len() == 1 {
            row_digests[0]
        } else {
            repops::fold_chain("layer-rows", row_digests)
        }
    }

    /// The digest of a trunk output: exact `f32` bytes, tagged with the layer.
    pub fn trunk_digest(layer: u32, batch: &TokenBatchF32) -> u64 {
        let mut digest = repops::Digest::new("layer-trunk");
        digest
            .update_u32(layer)
            .update_u32(batch.rows)
            .update_u32(batch.dim)
            .update_f32s(&batch.data);
        digest.finish()
    }

    /// The digest of one holder's expert partials for a layer.
    pub fn partials_digest(batch: &ExpertPartialsBatch) -> u64 {
        repops::digest_bytes("layer-partials", &batch.to_le_bytes())
    }

    /// The digest of a whole unit from its layer digests, in order.
    pub fn unit_digest(layer_digests: &[LayerDigest]) -> u64 {
        let digests: Vec<u64> = layer_digests.iter().map(|d| d.digest).collect();
        repops::fold_chain("unit", &digests)
    }

    /// The L2 norm of a boundary from its integer-exact energy: the squared codes are summed in
    /// integers per block, then scaled and reduced once.
    pub fn boundary_l2(boundary: &BlockwiseInt8Tensor) -> f64 {
        repops::int8_block_l2(
            &boundary.quantized_data,
            &boundary.absmax_scales,
            boundary.block_size,
        )
    }
}

/// A per-layer hook: the layer index and its output before boundary quantization.
pub type LayerHook<'a> = &'a mut dyn FnMut(u32, &mut Vec<f32>);

/// What [`LayerReference::run_layers`] produced.
#[derive(Debug, Clone)]
pub struct LayerRun {
    /// The last boundary, dequantized: the input of the next layer.
    pub hidden: Vec<f32>,
    /// The last boundary as quantized; `None` when no layer ran.
    pub boundary: Option<BlockwiseInt8Tensor>,
    /// One digest per layer, in layer order.
    pub layer_digests: Vec<LayerDigest>,
    /// The experts the router selected on every layer, `(layer, expert ids)`.
    pub routed: Vec<(u32, Vec<u32>)>,
}

/// A batch of `rows` activations of width `dim` in exact `f32`, for the trunk output that expert
/// slices and the combine consume. Layout: `"F32B" | rows u32 LE | dim u32 LE | data f32 LE ...`.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenBatchF32 {
    pub rows: u32,
    pub dim: u32,
    pub data: Vec<f32>,
}

pub const F32_BATCH_MAGIC: [u8; 4] = *b"F32B";

impl TokenBatchF32 {
    pub fn new(rows: u32, dim: u32, data: Vec<f32>) -> Result<Self, String> {
        if data.len() != (rows as usize) * (dim as usize) {
            return Err(format!(
                "TokenBatchF32: {} values do not form a {rows} x {dim} batch",
                data.len()
            ));
        }
        Ok(Self { rows, dim, data })
    }

    pub fn from_rows(rows: &[Vec<f32>]) -> Result<Self, String> {
        let dim = rows.first().map_or(0, Vec::len) as u32;
        if rows.iter().any(|r| r.len() != dim as usize) {
            return Err("TokenBatchF32: rows differ in width".to_string());
        }
        Ok(Self {
            rows: rows.len() as u32,
            dim,
            data: rows.concat(),
        })
    }

    pub fn row(&self, row: usize) -> &[f32] {
        let dim = self.dim as usize;
        &self.data[row * dim..(row + 1) * dim]
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.data.len() * 4);
        out.extend_from_slice(&F32_BATCH_MAGIC);
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.dim.to_le_bytes());
        for v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || bytes[0..4] != F32_BATCH_MAGIC {
            return Err("TokenBatchF32: missing F32B header".to_string());
        }
        let rows = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let dim = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let expected = (rows as usize) * (dim as usize);
        let body = &bytes[12..];
        if body.len() != expected * 4 {
            return Err(format!(
                "TokenBatchF32: {} data bytes for a {rows} x {dim} batch",
                body.len()
            ));
        }
        let data = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Self { rows, dim, data })
    }
}

/// The router's choice for every row of a batch at one layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerRouting {
    pub layer: u32,
    pub rows: Vec<Vec<MoERoutedExpertSelection>>,
}

impl LayerRouting {
    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("serialize LayerRouting")
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("LayerRouting: {e}"))
    }
}

/// One expert's contribution for one row.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertPartial {
    pub row: u32,
    pub expert: u32,
    pub data: Vec<f32>,
}

/// Every contribution one holder computed for a batch at one layer. Layout:
/// `"PART" | layer u32 | rows u32 | dim u32 | count u32 | (row u32 | expert u32 | dim f32) ...`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertPartialsBatch {
    pub layer: u32,
    pub rows: u32,
    pub dim: u32,
    pub partials: Vec<ExpertPartial>,
}

pub const PARTIALS_MAGIC: [u8; 4] = *b"PART";

impl ExpertPartialsBatch {
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(20 + self.partials.len() * (8 + self.dim as usize * 4));
        out.extend_from_slice(&PARTIALS_MAGIC);
        out.extend_from_slice(&self.layer.to_le_bytes());
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.extend_from_slice(&(self.partials.len() as u32).to_le_bytes());
        for p in &self.partials {
            out.extend_from_slice(&p.row.to_le_bytes());
            out.extend_from_slice(&p.expert.to_le_bytes());
            for v in &p.data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 20 || bytes[0..4] != PARTIALS_MAGIC {
            return Err("ExpertPartialsBatch: missing PART header".to_string());
        }
        let u32_at = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let layer = u32_at(4);
        let rows = u32_at(8);
        let dim = u32_at(12) as usize;
        let count = u32_at(16) as usize;
        let entry_len = 8 + dim * 4;
        if bytes.len() != 20 + count * entry_len {
            return Err(format!(
                "ExpertPartialsBatch: {} bytes do not hold {count} partials of width {dim}",
                bytes.len()
            ));
        }
        let mut partials = Vec::with_capacity(count);
        for i in 0..count {
            let at = 20 + i * entry_len;
            let row = u32_at(at);
            let expert = u32_at(at + 4);
            let data = bytes[at + 8..at + entry_len]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            partials.push(ExpertPartial { row, expert, data });
        }
        Ok(Self {
            layer,
            rows,
            dim: dim as u32,
            partials,
        })
    }

    /// Partials of one row as `(expert, data)` pairs.
    pub fn row_partials(&self, row: u32) -> Vec<(u32, Vec<f32>)> {
        self.partials
            .iter()
            .filter(|p| p.row == row)
            .map(|p| (p.expert, p.data.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hidden() -> Vec<f32> {
        (0..64).map(|i| ((i as f32) * 0.11).sin() * 0.8).collect()
    }

    #[test]
    fn layer_forward_is_the_composition_of_trunk_experts_and_combine() {
        let gamma = LayerReference::gamma_from_bank(None, 64);
        let (whole, routing) = LayerReference::layer_forward(&hidden(), 3, &gamma, 8, 2, 0.7);
        assert_eq!(routing.len(), 2);

        let (trunk, routing_again) = LayerReference::trunk_forward(&hidden(), 3, &gamma, 8, 2, 0.7);
        assert_eq!(routing, routing_again);
        // Partials gathered in reverse order still combine to the same bytes.
        let mut partials: Vec<(u32, Vec<f32>)> = routing
            .iter()
            .map(|r| {
                (
                    r.expert_id,
                    LayerReference::expert_contribution(&trunk, 3, r.expert_id, r.gating_weight),
                )
            })
            .collect();
        partials.reverse();
        let composed = LayerReference::combine(&trunk, &partials);
        assert_eq!(composed, whole);
        assert!(whole.iter().all(|v| v.abs() <= 8.0));

        let (dense, no_routing) = LayerReference::layer_forward(&hidden(), 3, &gamma, 0, 0, 0.7);
        assert!(no_routing.is_empty());
        assert_eq!(dense, LayerReference::combine(&trunk, &[]));
        assert_eq!(
            LayerReference::boundary_quantize(&whole).block_size,
            LAYER_BOUNDARY_BLOCK
        );
    }

    #[test]
    fn f32_batch_and_partials_codecs_roundtrip() {
        let batch = TokenBatchF32::from_rows(&[hidden(), hidden()]).unwrap();
        let decoded = TokenBatchF32::from_le_bytes(&batch.to_le_bytes()).unwrap();
        assert_eq!(decoded, batch);
        assert_eq!(decoded.row(1), hidden().as_slice());
        assert!(TokenBatchF32::from_le_bytes(b"nope").is_err());
        assert!(TokenBatchF32::new(2, 3, vec![0.0; 5]).is_err());

        let partials = ExpertPartialsBatch {
            layer: 7,
            rows: 2,
            dim: 3,
            partials: vec![
                ExpertPartial {
                    row: 0,
                    expert: 5,
                    data: vec![1.0, 2.0, 3.0],
                },
                ExpertPartial {
                    row: 1,
                    expert: 2,
                    data: vec![-1.5, 0.0, 9.0],
                },
            ],
        };
        let decoded = ExpertPartialsBatch::from_le_bytes(&partials.to_le_bytes()).unwrap();
        assert_eq!(decoded, partials);
        assert_eq!(decoded.row_partials(1), vec![(2, vec![-1.5, 0.0, 9.0])]);
        assert!(ExpertPartialsBatch::from_le_bytes(&partials.to_le_bytes()[..30]).is_err());

        let routing = LayerRouting {
            layer: 1,
            rows: vec![vec![], vec![]],
        };
        assert_eq!(
            LayerRouting::from_json_bytes(&routing.to_json_bytes()).unwrap(),
            routing
        );
        assert_eq!(LayerReference::routing_cutoffs(256, 0.9), (127, 196));
    }
}
