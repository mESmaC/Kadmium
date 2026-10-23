//! The [`AppKind::KvReplay`] unit: a session's decoder state rebuilt from a bounded suffix of
//! its context on the trunk owner (ARCHITECTURE 9.12).
//!
//! DeepInfer's bounded replay recomputes the last tokens of a context (the attention window
//! plus the longest draft it may back out of) when a session's window state is not where the
//! KV it hit is. Here a replay graph is the decode step's chain of trunk ranges with one row
//! per suffix token: each unit runs the suffix through its layers on the reference kernels,
//! as a stage unit would, and reports the pages it rebuilt with their modelled bytes and the
//! digest of the layer's boundary. The suffix rows themselves are a stand-in for the tokens'
//! hidden rows, which no real model produced; the work is real, the sizes are modelled.

use crate::{
    BlockwiseInt8Tensor, ExecutorContext, LiveStageForwardRequest, TokenBatchInt8, WorkExecutor,
    WorkInputs, WorkOutput, execute_stage_layers_with_ktb_routed_hooked, prompt_seed_hidden,
};
use crate::{LayerReference, grid_executor::fold_layer_chains};
use kadmium_grid::{AppKind, ContentAccess, WorkUnit};
use kadmium_sharding::{KvPageNames, KvReplayParams, ModelContentNames, StepGraphBuilder};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One page a replay rebuilt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvPageRecord {
    pub content_id: String,
    pub layer: u32,
    /// From the model spec's reference KV figure, not measured.
    pub bytes_modelled: u64,
    /// The RepOps digest of the layer's boundary over the suffix.
    pub digest: u64,
}

/// What a replay unit reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvReplayReport {
    pub session_id: String,
    pub stage_index: usize,
    pub start_layer: u32,
    pub end_layer: u32,
    pub suffix_tokens: u32,
    pub tokens_total: u64,
    pub pages: Vec<KvPageRecord>,
    pub bytes_modelled_total: u64,
    /// Always true: the page sizes come from the spec, the suffix rows from a seed.
    pub modelled: bool,
}

/// The suffix rows a replay runs, one per token of the suffix, from the session and the token
/// positions: the same wherever they are built.
pub fn suffix_rows(
    session_id: &str,
    tokens_total: u64,
    suffix_tokens: u32,
    dim: usize,
) -> Vec<Vec<f32>> {
    let first = tokens_total.saturating_sub(suffix_tokens as u64);
    (0..suffix_tokens as u64)
        .map(|i| prompt_seed_hidden(&format!("{session_id}/token/{}", first + i), dim))
        .collect()
}

/// The suffix as the replay graph's seed batch.
pub fn suffix_batch(
    session_id: &str,
    tokens_total: u64,
    suffix_tokens: u32,
    dim: u32,
) -> Result<TokenBatchInt8, String> {
    let rows = suffix_rows(session_id, tokens_total, suffix_tokens, dim as usize);
    let tensors: Vec<BlockwiseInt8Tensor> = rows
        .iter()
        .map(|r| BlockwiseInt8Tensor::quantize(r, crate::LAYER_BOUNDARY_BLOCK))
        .collect();
    TokenBatchInt8::from_row_tensors(dim, &tensors)
}

pub struct KvReplayExecutor;

impl WorkExecutor for KvReplayExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::KvReplay
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: KvReplayParams = unit.params_json().map_err(|e| e.to_string())?;
        let batch = inputs.activation_batch(unit)?;
        let rows = match batch.row_tensors() {
            Some(rows) => rows,
            None if batch.rows == 1 => vec![batch.tensor.clone()],
            None => return Err("a replay suffix needs rows that are whole blocks".to_string()),
        };
        let bank = ctx.resident_ktb_bank.lock().expect("resident bank lock");
        let mut responses = Vec::with_capacity(rows.len());
        let mut routed_hits: BTreeMap<(u32, u32), u64> = BTreeMap::new();
        for row in rows {
            let request = LiveStageForwardRequest {
                epoch: params.epoch,
                model_id: params.model_id.clone(),
                stage_index: params.stage_index,
                start_layer: params.start_layer,
                end_layer: params.end_layer,
                q_star_ratio: params.q_star_ratio,
                boundary_input: row,
                run_speculative_draft: false,
            };
            let (response, routed) = execute_stage_layers_with_ktb_routed_hooked(
                &ctx.node_id,
                &ctx.hostname,
                &request,
                Some(&bank),
                params.experts_per_layer,
                params.active_experts_per_token,
                None,
            );
            for (layer, experts) in routed {
                for expert in experts {
                    *routed_hits.entry((layer, expert)).or_default() += 1;
                }
            }
            responses.push(response);
        }
        drop(bank);
        if responses.is_empty() {
            return Err("replay received an empty suffix".to_string());
        }
        let boundaries: Vec<BlockwiseInt8Tensor> = responses
            .iter()
            .map(|r| r.boundary_output.clone())
            .collect();
        let out = TokenBatchInt8::from_row_tensors(batch.dim, &boundaries)?;
        let layer_digests = fold_layer_chains(responses.iter().map(|r| r.layer_digests.as_slice()));
        let pages: Vec<KvPageRecord> = layer_digests
            .iter()
            .map(|d| KvPageRecord {
                content_id: KvPageNames::page_id(&params.model_id, &params.session_id, d.layer),
                layer: d.layer,
                bytes_modelled: params.page_bytes_modelled,
                digest: d.digest,
            })
            .collect();
        let report = KvReplayReport {
            session_id: params.session_id.clone(),
            stage_index: params.stage_index,
            start_layer: params.start_layer,
            end_layer: params.end_layer,
            suffix_tokens: responses.len() as u32,
            tokens_total: params.tokens_total,
            bytes_modelled_total: pages.iter().map(|p| p.bytes_modelled).sum(),
            pages,
            modelled: true,
        };
        let accessed: Vec<ContentAccess> = routed_hits
            .iter()
            .map(|((layer, expert), n)| ContentAccess {
                content_id: ModelContentNames::expert_id(&params.model_id, *layer, *expert),
                hits: *n,
            })
            .collect();
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                out.to_le_bytes(),
            )],
            payload: serde_json::to_vec(&report).unwrap_or_default(),
            digest_fnv1a: LayerReference::unit_digest(&layer_digests),
            layer_digests,
            check_value: responses
                .iter()
                .map(|r| r.output_activation_l2_norm as f64)
                .sum::<f64>()
                / responses.len() as f64,
            units_processed: responses.len() as u64,
            accessed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_rows_are_the_last_tokens_and_the_batch_splits_back() {
        let a = suffix_rows("sess", 1_000, 3, 64);
        let b = suffix_rows("sess", 1_000, 3, 64);
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
        // The last row of a longer suffix is the same token as the last row of a shorter one.
        let long = suffix_rows("sess", 1_000, 5, 64);
        assert_eq!(long[4], a[2]);
        assert_ne!(suffix_rows("other", 1_000, 3, 64)[0], a[0]);
        let batch = suffix_batch("sess", 1_000, 3, 64).unwrap();
        assert_eq!(batch.rows, 3);
        assert_eq!(batch.row_tensors().unwrap().len(), 3);
        assert_eq!(suffix_batch("sess", 2, 2, 64).unwrap().rows, 2);
    }
}
