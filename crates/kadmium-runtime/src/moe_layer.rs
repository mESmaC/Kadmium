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

use crate::expert_cache::{ExpertCache, lock_expert_cache};
use crate::{
    BlockwiseInt8Tensor, DEFAULT_SUBBLOCK_SIZE, FusedRmsNormRopeKernel, KvPageBlock,
    MoEGatingRouter, MoERoutedExpertSelection, PagedGqaAttentionKernel, PerTokenBlockQuantized,
    ScaleFormat, TokenBatchInt8, repops, ue8m0_biased_byte, ue8m0_to_f32,
};
use kadmium_data::KtbBankReader;
use kadmium_grid::LayerDigest;
use kadmium_sharding::ExpertSliceParams;
use serde::{Deserialize, Serialize};
use std::time::Instant;

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
            for chunk in raw_slice.as_chunks::<4>().0.iter().take(dim) {
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

    /// The weight vector of an expert over `dim` hidden indices: what a weight bank holds
    /// for the expert, and what the expert cache keeps resident.
    pub fn expert_weights(layer: u32, expert: u32, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| Self::expert_weight(layer, expert, i))
            .collect()
    }

    /// [`LayerReference::expert_weights`] as little-endian `f32` bytes: the expert's slot.
    pub fn expert_weight_bytes(layer: u32, expert: u32, dim: usize) -> Vec<u8> {
        Self::expert_weights(layer, expert, dim)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    /// The weights back from an entry's bytes, or `None` when the bytes are not exactly `dim`
    /// little-endian `f32`s: a caller then computes the weights from their source rather than
    /// pad or truncate, so a cached entry of the wrong width is never used.
    pub fn expert_weights_from_bytes(bytes: &[u8], dim: usize) -> Option<Vec<f32>> {
        if bytes.len() != dim * 4 {
            return None;
        }
        Some(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        )
    }

    /// The revision term of an [`crate::ExpertKey`] for a node's weights: the resident bank's root
    /// with the vector width folded in, so a width change or a new checkpoint never aliases.
    pub fn expert_revision(bank_root_fnv64: u64, dim: usize) -> u64 {
        bank_root_fnv64 ^ (dim as u64).wrapping_mul(0x9e3779b97f4a7c15)
    }

    /// One expert's contribution from its weight vector: `gate * swish(h) * w_expert`, in the
    /// same operation order whether the weights came from a slot or were computed on the spot.
    pub fn expert_contribution_from_weights(
        hidden: &[f32],
        weights: &[f32],
        gate: f32,
    ) -> Vec<f32> {
        hidden
            .iter()
            .zip(weights)
            .map(|(&v, &w)| {
                let swish = repops::silu(v);
                swish * w * gate
            })
            .collect()
    }

    /// One expert's contribution to the layer output: `gate * swish(h) * w_expert`.
    pub fn expert_contribution(hidden: &[f32], layer: u32, expert: u32, gate: f32) -> Vec<f32> {
        let weights = Self::expert_weights(layer, expert, hidden.len());
        Self::expert_contribution_from_weights(hidden, &weights, gate)
    }

    /// A speculative run of the next layer's router on this layer's trunk output: the next
    /// layer's norm at its position, then its routing (InferSSD's `build_moe_prefetch_hint`,
    /// which applies layer `l + 1`'s norm and router to layer `l`'s FFN input). The hint is
    /// advisory. It fills cache slots for layer `next_layer` and never changes what that
    /// layer computes, so the output is the same with the hint and without it.
    pub fn prefetch_hint(
        trunk_out: &[f32],
        next_layer: u32,
        gamma: &[f32],
        experts_per_layer: u32,
        top_k: u32,
        q_star: f64,
    ) -> Vec<u32> {
        if experts_per_layer == 0 || top_k == 0 {
            return Vec::new();
        }
        let dim = trunk_out.len().max(16);
        let mut h = trunk_out.to_vec();
        h.resize(dim, 0.1);
        let gamma: Vec<f32> = if gamma.len() >= dim {
            gamma[..dim].to_vec()
        } else {
            let mut g = gamma.to_vec();
            g.extend((g.len()..dim).map(|i| 1.0 + (i as f32) * 0.002));
            g
        };
        let probe = FusedRmsNormRopeKernel::forward(
            &h,
            &gamma,
            next_layer as usize + 1,
            ROPE_THETA,
            RMSNORM_EPS,
        );
        let (hot, warm) = Self::routing_cutoffs(experts_per_layer, q_star);
        MoEGatingRouter::route_top_k(&probe, next_layer, experts_per_layer, top_k, hot, warm)
            .into_iter()
            .map(|r| r.expert_id)
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

    /// [`LayerReference::run_layers`] with the experts' weights served through an
    /// [`ExpertCache`]: every layer first asks the cache for the experts it routed to (a miss
    /// loads the expert's weights and admits them under the cache's policy), and when
    /// `prefetch_depth > 0` runs the next layer's router on this layer's trunk output and
    /// prefetches the predicted experts into the next layer's entries. The prefetch costs a
    /// second router run per layer, so it is off unless asked for. The cache is locked only
    /// around the lookups and the prefetch, never across the layer's arithmetic. The
    /// arithmetic is the uncached one: the bytes at every boundary are identical, which the
    /// tests check. `revision` is the key term for these weights ([`Self::expert_revision`]).
    #[allow(clippy::too_many_arguments)]
    pub fn run_layers_cached(
        model_id: &str,
        revision: u64,
        hidden_in: &[f32],
        start: u32,
        end: u32,
        gamma: &[f32],
        experts_per_layer: u32,
        top_k: u32,
        q_star: f64,
        mut hook: Option<LayerHook<'_>>,
        cache: &std::sync::Mutex<ExpertCache>,
        prefetch_depth: usize,
    ) -> (LayerRun, LayerCacheTrace) {
        let mut hidden = hidden_in.to_vec();
        let mut boundary = None;
        let mut layer_digests = Vec::new();
        let mut routed = Vec::new();
        let mut trace = LayerCacheTrace::default();
        let mut pending_hint: Option<Vec<u32>> = None;
        lock_expert_cache(cache).begin_token(None);
        for layer in start..=end {
            let (h, routing) =
                Self::trunk_forward(&hidden, layer, gamma, experts_per_layer, top_k, q_star);
            let dim = h.len();
            let ids: Vec<u32> = routing.iter().map(|r| r.expert_id).collect();
            if let Some(hint) = pending_hint.take() {
                trace.hint_predicted += hint.len() as u64;
                trace.hint_correct += hint.iter().filter(|e| ids.contains(e)).count() as u64;
            }
            let expert_bytes = dim as u64 * 4;
            let fetched = {
                let mut cache = lock_expert_cache(cache);
                let before = cache.stats();
                let fetched =
                    cache.fetch(model_id, revision, layer, &ids, expert_bytes, |expert| {
                        Self::expert_weight_bytes(layer, expert, dim)
                    });
                let after = cache.stats();
                trace.hits += after.hits - before.hits;
                trace.misses += after.misses - before.misses;
                trace.bypassed += after.bypassed - before.bypassed;
                trace.prefetch_hits += after.prefetch_hits - before.prefetch_hits;
                fetched
            };
            let partials: Vec<(u32, Vec<f32>)> = routing
                .iter()
                .map(|r| {
                    let weights = fetched
                        .iter()
                        .find(|(e, _)| *e == r.expert_id)
                        .and_then(|(_, b)| Self::expert_weights_from_bytes(b, dim))
                        .unwrap_or_else(|| Self::expert_weights(layer, r.expert_id, dim));
                    (
                        r.expert_id,
                        Self::expert_contribution_from_weights(&h, &weights, r.gating_weight),
                    )
                })
                .collect();
            let mut out = Self::combine(&h, &partials);
            if prefetch_depth > 0 && layer < end {
                let hint =
                    Self::prefetch_hint(&h, layer + 1, gamma, experts_per_layer, top_k, q_star);
                let mut cache = lock_expert_cache(cache);
                let (_, requests) = cache.plan_prefetch(
                    model_id,
                    revision,
                    layer + 1,
                    &hint,
                    prefetch_depth,
                    expert_bytes,
                );
                for request in requests {
                    let bytes = Self::expert_weight_bytes(layer + 1, request.key.expert, dim);
                    if cache.prefetch_done(request, bytes) {
                        trace.prefetched += 1;
                    }
                }
                pending_hint = Some(hint);
            }
            if let Some(hook) = hook.as_mut() {
                hook(layer, &mut out);
            }
            routed.push((layer, ids));
            let q = Self::boundary_quantize(&out);
            layer_digests.push(LayerDigest::new(layer, Self::boundary_digest(layer, &q)));
            hidden = q.dequantize();
            boundary = Some(q);
        }
        (
            LayerRun {
                hidden,
                boundary,
                layer_digests,
                routed,
            },
            trace,
        )
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

/// What the expert cache did for one run of layers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct LayerCacheTrace {
    pub hits: u64,
    pub misses: u64,
    pub bypassed: u64,
    /// Experts the run found resident because a prefetch put them there.
    pub prefetch_hits: u64,
    /// Experts prefetched during the run.
    pub prefetched: u64,
    /// Experts the next-layer hint predicted, and how many the next layer then routed to.
    pub hint_predicted: u64,
    pub hint_correct: u64,
}

impl LayerCacheTrace {
    pub fn hit_rate(&self) -> f64 {
        let served = self.hits + self.misses;
        if served == 0 {
            0.0
        } else {
            self.hits as f64 / served as f64
        }
    }

    pub fn hint_precision(&self) -> f64 {
        if self.hint_predicted == 0 {
            0.0
        } else {
            self.hint_correct as f64 / self.hint_predicted as f64
        }
    }

    pub fn add(&mut self, other: &LayerCacheTrace) {
        self.hits += other.hits;
        self.misses += other.misses;
        self.bypassed += other.bypassed;
        self.prefetch_hits += other.prefetch_hits;
        self.prefetched += other.prefetched;
        self.hint_predicted += other.hint_predicted;
        self.hint_correct += other.hint_correct;
    }
}

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
            .as_chunks::<4>()
            .0
            .iter()
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
                .as_chunks::<4>()
                .0
                .iter()
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

// ================================================================================================
// DeepEP Holder-Selective Dispatch, Low-Latency Quantized Combine & Zero-SM Hook Overlap
// ================================================================================================

/// Precision of a holder-selective dispatch packet (`HolderDispatchBatch`), after DeepEP's
/// high-throughput (`ExactF32`) and low-latency decode (`LowLatencyFp8*`) modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DispatchPrecision {
    /// Exact `f32` rows (`4` bytes/element), selective to `is_token_in_rank[holder]`:
    /// preserves bit-for-bit equivalence with full-broadcast fan-out and chain execution.
    ExactF32,
    /// Fine-grained `1 x DEFAULT_SUBBLOCK_SIZE` 8-bit quantization with `f32` sub-block scales.
    LowLatencyFp8F32Scale,
    /// Fine-grained `1 x DEFAULT_SUBBLOCK_SIZE` 8-bit quantization with 1-byte `UE8M0`
    /// power-of-two exponent scales (`ceil_to_ue8m0`), cutting wire bytes by nearly 4x.
    LowLatencyFp8Ue8m0,
}

impl DispatchPrecision {
    fn tag(self) -> u8 {
        match self {
            Self::ExactF32 => 0,
            Self::LowLatencyFp8F32Scale => 1,
            Self::LowLatencyFp8Ue8m0 => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::ExactF32),
            1 => Ok(Self::LowLatencyFp8F32Scale),
            2 => Ok(Self::LowLatencyFp8Ue8m0),
            other => Err(format!(
                "HolderDispatchBatch: unknown precision tag {other}"
            )),
        }
    }
}

/// One `(row, expert, gating_weight)` assignment targeted at a specific expert-slice holder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HolderExpertAssignment {
    pub row: u32,
    pub expert: u32,
    pub gating_weight: f32,
}

pub const HOLDER_DISPATCH_MAGIC: [u8; 4] = *b"DPDS";

/// Per-holder selective token dispatch batch (`"DPDS"`), after DeepEP's `get_dispatch_layout` +
/// `dispatch` / `low_latency_dispatch`.
///
/// Instead of broadcasting every row of `TokenBatchF32` and the full `LayerRouting` JSON to every
/// holder in the mesh, `HolderDispatchBatch` packs **only the deduplicated rows `r` where
/// `is_token_in_rank[holder][r]` is true**, along with the holder's filtered `(row, expert, gate)`
/// assignments. When `holder` serves 0 rows on a step, `included_rows` and `row_data` are empty.
#[derive(Debug, Clone, PartialEq)]
pub struct HolderDispatchBatch {
    pub layer: u32,
    pub total_rows: u32,
    pub dim: u32,
    pub precision: DispatchPrecision,
    /// Deduplicated batch row indices `0..total_rows` included in this holder's payload, ascending.
    pub included_rows: Vec<u32>,
    /// Row vectors for `included_rows` (`included_rows.len() * dim`), exact `f32` or dequantized.
    pub row_data: Vec<f32>,
    /// Filtered `(row, expert, gate)` assignments served by this holder, in `(row, expert)` order.
    pub assignments: Vec<HolderExpertAssignment>,
}

impl HolderDispatchBatch {
    /// Builds the selective dispatch packet for `slice` from the trunk output and routing.
    pub fn for_slice(
        trunk: &TokenBatchF32,
        routing: &LayerRouting,
        slice: &ExpertSliceParams,
        precision: DispatchPrecision,
    ) -> Self {
        let dim = trunk.dim as usize;
        let mut included_rows = Vec::new();
        let mut raw_rows = Vec::new();
        let mut assignments = Vec::new();

        for (row_idx, selections) in routing.rows.iter().enumerate() {
            let row = row_idx as u32;
            let mut mine: Vec<&MoERoutedExpertSelection> = selections
                .iter()
                .filter(|s| slice.serves(s.expert_id, row))
                .collect();
            if mine.is_empty() {
                continue;
            }
            mine.sort_by_key(|s| s.expert_id);
            included_rows.push(row);
            raw_rows.extend_from_slice(trunk.row(row_idx));
            for sel in mine {
                assignments.push(HolderExpertAssignment {
                    row,
                    expert: sel.expert_id,
                    gating_weight: sel.gating_weight,
                });
            }
        }

        let row_data = match precision {
            DispatchPrecision::ExactF32 => raw_rows,
            DispatchPrecision::LowLatencyFp8F32Scale => PerTokenBlockQuantized::quantize(
                &raw_rows,
                included_rows.len(),
                dim,
                DEFAULT_SUBBLOCK_SIZE,
                ScaleFormat::ExactF32,
            )
            .dequantize(),
            DispatchPrecision::LowLatencyFp8Ue8m0 => PerTokenBlockQuantized::quantize(
                &raw_rows,
                included_rows.len(),
                dim,
                DEFAULT_SUBBLOCK_SIZE,
                ScaleFormat::Ue8m0PowerOfTwo,
            )
            .dequantize(),
        };

        Self {
            layer: slice.layer,
            total_rows: trunk.rows,
            dim: trunk.dim,
            precision,
            included_rows,
            row_data,
            assignments,
        }
    }

    /// Looks up the hidden state slice for `original_row` (`0..total_rows`), if included.
    pub fn row_slice(&self, original_row: u32) -> Option<&[f32]> {
        let local_idx = self.included_rows.binary_search(&original_row).ok()?;
        let dim = self.dim as usize;
        Some(&self.row_data[local_idx * dim..(local_idx + 1) * dim])
    }

    /// Evaluates every assignment in this dispatch packet using `LayerReference::expert_contribution`.
    pub fn evaluate_partials(&self) -> ExpertPartialsBatch {
        self.evaluate_partials_with(&std::collections::BTreeMap::new())
    }

    /// The distinct experts the packet's assignments ask for, ascending.
    pub fn expert_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.assignments.iter().map(|a| a.expert).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// [`HolderDispatchBatch::evaluate_partials`] with weight vectors a cache served for some
    /// experts; an expert missing from `weights` is computed from its source. Same operation
    /// order either way, so the bytes do not depend on what was cached.
    pub fn evaluate_partials_with(
        &self,
        weights: &std::collections::BTreeMap<u32, Vec<f32>>,
    ) -> ExpertPartialsBatch {
        let mut partials = Vec::with_capacity(self.assignments.len());
        for assign in &self.assignments {
            if let Some(h) = self.row_slice(assign.row) {
                let data = match weights.get(&assign.expert) {
                    Some(w) => {
                        LayerReference::expert_contribution_from_weights(h, w, assign.gating_weight)
                    }
                    None => LayerReference::expert_contribution(
                        h,
                        self.layer,
                        assign.expert,
                        assign.gating_weight,
                    ),
                };
                partials.push(ExpertPartial {
                    row: assign.row,
                    expert: assign.expert,
                    data,
                });
            }
        }
        ExpertPartialsBatch {
            layer: self.layer,
            rows: self.total_rows,
            dim: self.dim,
            partials,
        }
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let inc = self.included_rows.len();
        let dim = self.dim as usize;
        let mut out = Vec::new();
        out.extend_from_slice(&HOLDER_DISPATCH_MAGIC);
        out.extend_from_slice(&[self.precision.tag(), 0, 0, 0]);
        out.extend_from_slice(&self.layer.to_le_bytes());
        out.extend_from_slice(&self.total_rows.to_le_bytes());
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.extend_from_slice(&(inc as u32).to_le_bytes());
        out.extend_from_slice(&(self.assignments.len() as u32).to_le_bytes());

        for &r in &self.included_rows {
            out.extend_from_slice(&r.to_le_bytes());
        }
        for a in &self.assignments {
            out.extend_from_slice(&a.row.to_le_bytes());
            out.extend_from_slice(&a.expert.to_le_bytes());
            out.extend_from_slice(&a.gating_weight.to_le_bytes());
        }

        match self.precision {
            DispatchPrecision::ExactF32 => {
                for &v in &self.row_data {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            DispatchPrecision::LowLatencyFp8F32Scale => {
                let q = PerTokenBlockQuantized::quantize(
                    &self.row_data,
                    inc,
                    dim,
                    DEFAULT_SUBBLOCK_SIZE,
                    ScaleFormat::ExactF32,
                );
                for &s in &q.scales {
                    out.extend_from_slice(&s.to_le_bytes());
                }
                for &c in &q.codes {
                    out.push(c as u8);
                }
            }
            DispatchPrecision::LowLatencyFp8Ue8m0 => {
                let q = PerTokenBlockQuantized::quantize(
                    &self.row_data,
                    inc,
                    dim,
                    DEFAULT_SUBBLOCK_SIZE,
                    ScaleFormat::Ue8m0PowerOfTwo,
                );
                for &s in &q.scales {
                    out.push(ue8m0_biased_byte(s));
                }
                for &c in &q.codes {
                    out.push(c as u8);
                }
            }
        }
        out
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 28 || bytes[0..4] != HOLDER_DISPATCH_MAGIC {
            return Err("HolderDispatchBatch: missing DPDS header".to_string());
        }
        let u32_at = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let precision = DispatchPrecision::from_tag(bytes[4])?;
        let layer = u32_at(8);
        let total_rows = u32_at(12);
        let dim = u32_at(16) as usize;
        let inc = u32_at(20) as usize;
        let assign_count = u32_at(24) as usize;

        let meta_end = 28 + inc * 4 + assign_count * 12;
        if bytes.len() < meta_end {
            return Err("HolderDispatchBatch: truncated metadata".to_string());
        }
        let mut at = 28usize;
        let mut included_rows = Vec::with_capacity(inc);
        for _ in 0..inc {
            included_rows.push(u32_at(at));
            at += 4;
        }
        let mut assignments = Vec::with_capacity(assign_count);
        for _ in 0..assign_count {
            let row = u32_at(at);
            let expert = u32_at(at + 4);
            let gating_weight =
                f32::from_le_bytes([bytes[at + 8], bytes[at + 9], bytes[at + 10], bytes[at + 11]]);
            assignments.push(HolderExpertAssignment {
                row,
                expert,
                gating_weight,
            });
            at += 12;
        }

        let body = &bytes[at..];
        let row_data = match precision {
            DispatchPrecision::ExactF32 => {
                let expected = inc * dim * 4;
                if body.len() != expected {
                    return Err(format!(
                        "HolderDispatchBatch: expected {expected} f32 bytes, got {}",
                        body.len()
                    ));
                }
                body.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            DispatchPrecision::LowLatencyFp8F32Scale => {
                let k_blocks = dim.div_ceil(DEFAULT_SUBBLOCK_SIZE);
                let scale_bytes = inc * k_blocks * 4;
                let code_bytes = inc * dim;
                if body.len() != scale_bytes + code_bytes {
                    return Err("HolderDispatchBatch: truncated FP8+f32 payload".to_string());
                }
                let scales: Vec<f32> = body[..scale_bytes]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let codes: Vec<i8> = body[scale_bytes..].iter().map(|&b| b as i8).collect();
                PerTokenBlockQuantized {
                    rows: inc,
                    k: dim,
                    block_size: DEFAULT_SUBBLOCK_SIZE,
                    format: ScaleFormat::ExactF32,
                    codes,
                    scales,
                }
                .dequantize()
            }
            DispatchPrecision::LowLatencyFp8Ue8m0 => {
                let k_blocks = dim.div_ceil(DEFAULT_SUBBLOCK_SIZE);
                let scale_bytes = inc * k_blocks;
                let code_bytes = inc * dim;
                if body.len() != scale_bytes + code_bytes {
                    return Err("HolderDispatchBatch: truncated FP8+UE8M0 payload".to_string());
                }
                let scales: Vec<f32> = body[..scale_bytes]
                    .iter()
                    .map(|&b| ue8m0_to_f32(b))
                    .collect();
                let codes: Vec<i8> = body[scale_bytes..].iter().map(|&b| b as i8).collect();
                PerTokenBlockQuantized {
                    rows: inc,
                    k: dim,
                    block_size: DEFAULT_SUBBLOCK_SIZE,
                    format: ScaleFormat::Ue8m0PowerOfTwo,
                    codes,
                    scales,
                }
                .dequantize()
            }
        };

        Ok(Self {
            layer,
            total_rows,
            dim: dim as u32,
            precision,
            included_rows,
            row_data,
            assignments,
        })
    }
}

pub const LOW_LATENCY_COMBINE_MAGIC: [u8; 4] = *b"DPCM";

/// Low-latency quantized combine envelope (`"DPCM"`), after DeepEP's `low_latency_combine`.
///
/// Packs an [`ExpertPartialsBatch`] using fine-grained `1 x DEFAULT_SUBBLOCK_SIZE` quantization
/// (`ScaleFormat::Ue8m0PowerOfTwo` or `ScaleFormat::ExactF32`), reducing return wire bytes by
/// nearly 4x compared to `"PART"` `f32` partials.
#[derive(Debug, Clone, PartialEq)]
pub struct LowLatencyCombineBatch {
    pub layer: u32,
    pub rows: u32,
    pub dim: u32,
    pub format: ScaleFormat,
    /// `(row, expert)` pairs for each partial row in `quantized`.
    pub meta: Vec<(u32, u32)>,
    pub quantized: PerTokenBlockQuantized,
}

impl LowLatencyCombineBatch {
    pub fn from_partials(batch: &ExpertPartialsBatch, format: ScaleFormat) -> Self {
        let count = batch.partials.len();
        let dim = batch.dim as usize;
        let mut flat = Vec::with_capacity(count * dim);
        let mut meta = Vec::with_capacity(count);
        for p in &batch.partials {
            meta.push((p.row, p.expert));
            flat.extend_from_slice(&p.data);
        }
        let quantized =
            PerTokenBlockQuantized::quantize(&flat, count, dim, DEFAULT_SUBBLOCK_SIZE, format);
        Self {
            layer: batch.layer,
            rows: batch.rows,
            dim: batch.dim,
            format,
            meta,
            quantized,
        }
    }

    pub fn into_partials_batch(&self) -> ExpertPartialsBatch {
        let deq = self.quantized.dequantize();
        let dim = self.dim as usize;
        let partials = self
            .meta
            .iter()
            .enumerate()
            .map(|(idx, &(row, expert))| ExpertPartial {
                row,
                expert,
                data: deq[idx * dim..(idx + 1) * dim].to_vec(),
            })
            .collect();
        ExpertPartialsBatch {
            layer: self.layer,
            rows: self.rows,
            dim: self.dim,
            partials,
        }
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let count = self.meta.len();
        let fmt_tag = match self.format {
            ScaleFormat::ExactF32 => 0u8,
            ScaleFormat::Ue8m0PowerOfTwo => 1u8,
        };
        let mut out = Vec::new();
        out.extend_from_slice(&LOW_LATENCY_COMBINE_MAGIC);
        out.extend_from_slice(&[fmt_tag, 0, 0, 0]);
        out.extend_from_slice(&self.layer.to_le_bytes());
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.extend_from_slice(&(count as u32).to_le_bytes());
        for &(row, expert) in &self.meta {
            out.extend_from_slice(&row.to_le_bytes());
            out.extend_from_slice(&expert.to_le_bytes());
        }
        match self.format {
            ScaleFormat::ExactF32 => {
                for &s in &self.quantized.scales {
                    out.extend_from_slice(&s.to_le_bytes());
                }
            }
            ScaleFormat::Ue8m0PowerOfTwo => {
                for &s in &self.quantized.scales {
                    out.push(ue8m0_biased_byte(s));
                }
            }
        }
        for &c in &self.quantized.codes {
            out.push(c as u8);
        }
        out
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24 || bytes[0..4] != LOW_LATENCY_COMBINE_MAGIC {
            return Err("LowLatencyCombineBatch: missing DPCM header".to_string());
        }
        let u32_at = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let format = match bytes[4] {
            0 => ScaleFormat::ExactF32,
            1 => ScaleFormat::Ue8m0PowerOfTwo,
            other => return Err(format!("LowLatencyCombineBatch: unknown format {other}")),
        };
        let layer = u32_at(8);
        let rows = u32_at(12);
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let meta_end = 24 + count * 8;
        if bytes.len() < meta_end {
            return Err("LowLatencyCombineBatch: truncated meta".to_string());
        }
        let mut meta = Vec::with_capacity(count);
        for i in 0..count {
            let at = 24 + i * 8;
            meta.push((u32_at(at), u32_at(at + 4)));
        }
        let body = &bytes[meta_end..];
        let k_blocks = dim.div_ceil(DEFAULT_SUBBLOCK_SIZE);
        let scale_bytes = match format {
            ScaleFormat::ExactF32 => count * k_blocks * 4,
            ScaleFormat::Ue8m0PowerOfTwo => count * k_blocks,
        };
        let code_bytes = count * dim;
        if body.len() != scale_bytes + code_bytes {
            return Err("LowLatencyCombineBatch: truncated payload".to_string());
        }
        let scales: Vec<f32> = match format {
            ScaleFormat::ExactF32 => body[..scale_bytes]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            ScaleFormat::Ue8m0PowerOfTwo => body[..scale_bytes]
                .iter()
                .map(|&b| ue8m0_to_f32(b))
                .collect(),
        };
        let codes: Vec<i8> = body[scale_bytes..].iter().map(|&b| b as i8).collect();
        Ok(Self {
            layer,
            rows,
            dim: dim as u32,
            format,
            meta,
            quantized: PerTokenBlockQuantized {
                rows: count,
                k: dim,
                block_size: DEFAULT_SUBBLOCK_SIZE,
                format,
                codes,
                scales,
            },
        })
    }
}

/// Telemetry from overlapping a background communication/slice task (`RecvHook`) with local
/// non-dependent compute (e.g., a shared expert or a second micro-batch), after DeepEP's
/// `return_recv_hook=True`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecvHookOverlapTrace {
    pub comm_ms: f64,
    pub local_compute_ms: f64,
    pub total_wall_ms: f64,
    pub overlapped_ms: f64,
    pub exposed_comm_ms: f64,
    pub zero_sm_occupancy: bool,
}

/// A background completion hook returned by `spawn_recv_hook`, after DeepEP's `recv_hook`.
pub struct RecvHook<T> {
    handle: std::thread::JoinHandle<(T, f64)>,
}

impl<T: Send + 'static> RecvHook<T> {
    /// Spawns background dispatch/combine work onto an independent worker thread (`0 SM`
    /// occupation on the caller's compute stream) and returns a `RecvHook` to complete it.
    pub fn spawn(background_comm: impl FnOnce() -> T + Send + 'static) -> Self {
        let handle = std::thread::spawn(move || {
            let t0 = Instant::now();
            let out = background_comm();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            (out, ms)
        });
        Self { handle }
    }

    /// Completes the background transfer and returns `(result, background_ms)`.
    pub fn complete(self) -> Result<(T, f64), String> {
        self.handle
            .join()
            .map_err(|_| "RecvHook background worker panicked".to_string())
    }
}

/// Runs `background_comm` behind a `RecvHook` while `local_compute` (such as the layer's shared
/// expert or the other micro-batch's attention/trunk) executes concurrently on the caller thread,
/// then invokes `hook.complete()`.
pub fn run_with_recv_hook<T, U>(
    background_comm: impl FnOnce() -> T + Send + 'static,
    local_compute: impl FnOnce() -> U,
) -> Result<(T, U, RecvHookOverlapTrace), String>
where
    T: Send + 'static,
{
    let wall_t0 = Instant::now();
    let hook = RecvHook::spawn(background_comm);

    let local_t0 = Instant::now();
    let local_out = local_compute();
    let local_compute_ms = local_t0.elapsed().as_secs_f64() * 1000.0;

    let (comm_out, comm_ms) = hook.complete()?;
    let total_wall_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;
    let overlapped_ms = comm_ms.min(local_compute_ms);
    let exposed_comm_ms = (total_wall_ms - local_compute_ms).max(0.0);

    Ok((
        comm_out,
        local_out,
        RecvHookOverlapTrace {
            comm_ms,
            local_compute_ms,
            total_wall_ms,
            overlapped_ms,
            exposed_comm_ms,
            zero_sm_occupancy: true,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_cache::ExpertCacheAllocation;

    /// The reviewer's benchmark shape: 60 layers, width 1024, 256 experts, top-8, release
    /// build. Run with `cargo test --release -p kadmium-runtime cached_path_overhead --
    /// --ignored --nocapture`. Without the prefetch the cached path must stay within a few
    /// percent of the uncached one once warm; with it, the second router run per layer shows.
    #[test]
    #[ignore = "timing; run in release on a quiet machine"]
    fn cached_path_overhead_stays_small_without_prefetch() {
        let hidden: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.07).sin() * 0.6).collect();
        let gamma = LayerReference::gamma_from_bank(None, 1024);
        let cache = std::sync::Mutex::new(ExpertCache::new(1 << 30, ExpertCacheAllocation::Global));
        let time = |f: &dyn Fn()| {
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                f();
            }
            t0.elapsed().as_secs_f64() * 1000.0 / 5.0
        };
        // Warm both paths once.
        LayerReference::run_layers(&hidden, 0, 59, &gamma, 256, 8, 0.8, None);
        LayerReference::run_layers_cached(
            "bench", 1, &hidden, 0, 59, &gamma, 256, 8, 0.8, None, &cache, 0,
        );
        let plain = time(&|| {
            LayerReference::run_layers(&hidden, 0, 59, &gamma, 256, 8, 0.8, None);
        });
        let cached = time(&|| {
            LayerReference::run_layers_cached(
                "bench", 1, &hidden, 0, 59, &gamma, 256, 8, 0.8, None, &cache, 0,
            );
        });
        let prefetching = time(&|| {
            LayerReference::run_layers_cached(
                "bench", 1, &hidden, 0, 59, &gamma, 256, 8, 0.8, None, &cache, 8,
            );
        });
        let stats = lock_expert_cache(&cache).stats();
        eprintln!(
            "per token: uncached {plain:.1} ms, cached {cached:.1} ms, cached + prefetch {prefetching:.1} ms; {stats:?}"
        );
        assert!(stats.hit_rate() > 0.9, "{stats:?}");
        assert!(
            cached <= plain * 1.15,
            "cached {cached:.1} ms vs uncached {plain:.1} ms"
        );
    }

    #[test]
    fn cached_layers_produce_the_uncached_bytes_and_prefetch_next_layer_experts() {
        let hidden: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.07).sin() * 0.6).collect();
        let gamma = LayerReference::gamma_from_bank(None, 64);
        let plain = LayerReference::run_layers(&hidden, 0, 5, &gamma, 8, 2, 0.8, None);
        let cache = std::sync::Mutex::new(ExpertCache::new(
            64 * 4096,
            ExpertCacheAllocation::Uniform { layers: 6 },
        ));
        let (cached, trace) = LayerReference::run_layers_cached(
            "toy", 1, &hidden, 0, 5, &gamma, 8, 2, 0.8, None, &cache, 2,
        );
        assert_eq!(cached.hidden, plain.hidden);
        assert_eq!(cached.layer_digests, plain.layer_digests);
        assert_eq!(cached.routed, plain.routed);
        assert_eq!(trace.hits + trace.misses, 12, "{trace:?}");
        assert!(trace.prefetched > 0, "{trace:?}");
        assert_eq!(
            trace.hint_predicted, 10,
            "two predictions for each of layers 1..=5"
        );
        assert!(trace.hint_precision() <= 1.0);

        // A second pass of the same token routes the same way: every expert is a hit now.
        let (again, trace2) = LayerReference::run_layers_cached(
            "toy", 1, &hidden, 0, 5, &gamma, 8, 2, 0.8, None, &cache, 0,
        );
        assert_eq!(again.hidden, plain.hidden);
        assert_eq!(trace2.misses, 0, "{trace2:?}");
        assert_eq!(trace2.hits, 12);
        assert_eq!(trace2.hit_rate(), 1.0);

        // With a hook the fault lands the same way in both paths.
        let mut fault = |layer: u32, out: &mut Vec<f32>| {
            if layer == 3 {
                out[0] += 0.5;
            }
        };
        let faulty_plain =
            LayerReference::run_layers(&hidden, 0, 5, &gamma, 8, 2, 0.8, Some(&mut fault));
        let (faulty_cached, _) = LayerReference::run_layers_cached(
            "toy",
            1,
            &hidden,
            0,
            5,
            &gamma,
            8,
            2,
            0.8,
            Some(&mut fault),
            &cache,
            2,
        );
        assert_eq!(faulty_cached.hidden, faulty_plain.hidden);
        assert_ne!(faulty_cached.hidden, plain.hidden);

        // A global cache of one expert's bytes bypasses most experts and still produces the
        // same bytes; the other model's experts never alias this one's.
        let tiny = std::sync::Mutex::new(ExpertCache::new(256, ExpertCacheAllocation::Global));
        let (small, trace3) = LayerReference::run_layers_cached(
            "toy", 1, &hidden, 0, 5, &gamma, 8, 2, 0.8, None, &tiny, 2,
        );
        assert_eq!(small.hidden, plain.hidden);
        assert!(trace3.misses > 0);
        let (other, trace4) = LayerReference::run_layers_cached(
            "other-model",
            1,
            &hidden,
            0,
            5,
            &gamma,
            8,
            2,
            0.8,
            None,
            &cache,
            0,
        );
        assert_eq!(other.hidden, plain.hidden);
        assert_eq!(trace4.hits, 0, "{trace4:?}");
        // Another revision of the same model's weights is a different entry too.
        let (revised, trace5) = LayerReference::run_layers_cached(
            "toy", 2, &hidden, 0, 5, &gamma, 8, 2, 0.8, None, &cache, 0,
        );
        assert_eq!(revised.hidden, plain.hidden);
        assert_eq!(trace5.hits, 0, "{trace5:?}");
        // An entry of the wrong width is never decoded.
        assert!(LayerReference::expert_weights_from_bytes(&[0u8; 12], 4).is_none());
        assert_eq!(
            LayerReference::expert_weights_from_bytes(&[0u8; 16], 4)
                .unwrap()
                .len(),
            4
        );
        assert_ne!(
            LayerReference::expert_revision(1, 64),
            LayerReference::expert_revision(1, 128)
        );

        // The hint itself is a top-k of the next layer's router.
        let hint = LayerReference::prefetch_hint(&hidden, 1, &gamma, 8, 2, 0.8);
        assert_eq!(hint.len(), 2);
        assert!(LayerReference::prefetch_hint(&hidden, 1, &gamma, 0, 2, 0.8).is_empty());
    }

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

    #[test]
    fn holder_selective_dispatch_and_low_latency_combine_roundtrip_and_overlap_hook() {
        use crate::ExpertResidencyTier;
        use kadmium_core::NodeId;
        use kadmium_sharding::ExpertRowShare;

        let r0 = hidden();
        let r1: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.07).cos() * 0.6).collect();
        let r2: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
        let trunk = TokenBatchF32::from_rows(&[r0.clone(), r1, r2.clone()]).unwrap();
        let routing = LayerRouting {
            layer: 4,
            rows: vec![
                vec![
                    MoERoutedExpertSelection {
                        expert_id: 1,
                        gating_weight: 0.6,
                        residency: ExpertResidencyTier::HotVramOrUnified,
                    },
                    MoERoutedExpertSelection {
                        expert_id: 3,
                        gating_weight: 0.4,
                        residency: ExpertResidencyTier::HotVramOrUnified,
                    },
                ],
                vec![MoERoutedExpertSelection {
                    expert_id: 7,
                    gating_weight: 1.0,
                    residency: ExpertResidencyTier::WarmSystemRam,
                }],
                vec![MoERoutedExpertSelection {
                    expert_id: 1,
                    gating_weight: 0.9,
                    residency: ExpertResidencyTier::HotVramOrUnified,
                }],
            ],
        };

        // Holder serves experts [1, 3]; row 1 selected only expert 7, so row 1 is omitted!
        // Row 0 selected both 1 and 3 on this holder -> deduplicated to a single row 0 vector!
        let slice = ExpertSliceParams {
            model_id: "m".into(),
            epoch: 1,
            session_id: "s".into(),
            layer: 4,
            rows: 3,
            holder: NodeId("node-a".into()),
            resident_experts: vec![1, 3],
            row_shares: Vec::<ExpertRowShare>::new(),
        };

        for precision in [
            DispatchPrecision::ExactF32,
            DispatchPrecision::LowLatencyFp8F32Scale,
            DispatchPrecision::LowLatencyFp8Ue8m0,
        ] {
            let pkt = HolderDispatchBatch::for_slice(&trunk, &routing, &slice, precision);
            assert_eq!(pkt.included_rows, vec![0, 2]);
            assert_eq!(pkt.assignments.len(), 3);
            assert!(pkt.row_slice(1).is_none());
            let wire = pkt.to_le_bytes();
            let decoded = HolderDispatchBatch::from_le_bytes(&wire).unwrap();
            assert_eq!(decoded, pkt);

            if precision == DispatchPrecision::ExactF32 {
                assert_eq!(decoded.row_slice(0).unwrap(), r0.as_slice());
                assert_eq!(decoded.row_slice(2).unwrap(), r2.as_slice());
                let partials = decoded.evaluate_partials();
                assert_eq!(partials.partials.len(), 3);

                // Low-latency combine codec round-trip on the partials:
                for fmt in [ScaleFormat::ExactF32, ScaleFormat::Ue8m0PowerOfTwo] {
                    let ll_combine = LowLatencyCombineBatch::from_partials(&partials, fmt);
                    let ll_bytes = ll_combine.to_le_bytes();
                    assert!(ll_bytes.len() < partials.to_le_bytes().len());
                    let ll_decoded = LowLatencyCombineBatch::from_le_bytes(&ll_bytes).unwrap();
                    assert_eq!(ll_decoded, ll_combine);
                    let recovered = ll_decoded.into_partials_batch();
                    assert_eq!(recovered.partials.len(), 3);
                }
            }
        }

        // Zero-SM recv_hook overlap: background dispatch/combine runs concurrently with local work.
        let (comm_val, local_val, trace) =
            run_with_recv_hook(|| 42u32, || LayerReference::combine(&r0, &[])).unwrap();
        assert_eq!(comm_val, 42);
        assert_eq!(local_val.len(), 64);
        assert!(trace.zero_sm_occupancy);
        assert!(trace.total_wall_ms >= 0.0);
    }
}
