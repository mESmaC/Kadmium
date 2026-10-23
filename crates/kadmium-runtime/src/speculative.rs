//! Speculative steps on the reference math: the draft rows a modelled drafter proposes and
//! the [`AppKind::DraftVerify`] unit that accepts a prefix of them (ARCHITECTURE 9.11).
//!
//! A speculative step runs the stage chain or fan-out over a verify batch: the session's own
//! row first, then `draft_length` draft rows. That forward is the real cost of verification,
//! and because the reference math is batch invariant (9.8) every row's boundary is the row's
//! own, whichever rows sit beside it. What is modelled is the drafter: no draft model exists
//! here, so the draft rows are deterministic perturbations of the session's row and the
//! acceptance test is a coin seeded by the verified row's digest and the drafter's modelled
//! confidence. The accounting reports full-acceptance and accepted tokens side by side, as
//! DeepInfer's benchmark README does.

use crate::{
    BlockwiseInt8Tensor, ExecutorContext, LAYER_BOUNDARY_BLOCK, LayerReference, TokenBatchInt8,
    WorkExecutor, WorkInputs, WorkOutput, prompt_seed_hidden,
};
use kadmium_grid::{AppKind, LayerDigest, WorkUnit};
use kadmium_sharding::{DraftVerifyParams, ModelledDrafter, SessionDraft, StepGraphBuilder};
use serde::{Deserialize, Serialize};

/// The draft rows a modelled drafter proposes after `base` at `step`: deterministic
/// perturbations of the row, the same wherever they are computed, so the coordinator can seed
/// a session's first verify batch and the verify unit can build the next one.
pub fn draft_rows(base: &[f32], seed: u64, step: u64, count: u32) -> Vec<Vec<f32>> {
    (1..=count)
        .map(|k| {
            let noise = prompt_seed_hidden(&format!("draft/{seed}/{step}/{k}"), base.len());
            base.iter()
                .zip(noise.iter())
                .map(|(b, n)| (b * 0.85 + n * 0.35).clamp(-8.0, 8.0))
                .collect()
        })
        .collect()
}

/// A verify batch: the session's row (its exact codes and scales) followed by `draft_length`
/// draft rows, each quantized on its own, so every row is a whole number of blocks and the
/// batch splits back into the same rows.
pub fn verify_batch(
    row: &BlockwiseInt8Tensor,
    dim: u32,
    seed: u64,
    step: u64,
    draft_length: u32,
) -> Result<TokenBatchInt8, String> {
    let base = row.dequantize();
    let mut rows = vec![row.clone()];
    for draft in draft_rows(&base, seed, step, draft_length) {
        rows.push(BlockwiseInt8Tensor::quantize(&draft, LAYER_BOUNDARY_BLOCK));
    }
    TokenBatchInt8::from_row_tensors(dim, &rows)
}

/// What a verify unit reports beside its output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DraftVerifyReport {
    pub session_id: String,
    pub step: u64,
    pub proposed: u32,
    pub accepted: u32,
    pub confidences: Vec<f64>,
    pub survival: Vec<f64>,
    /// `sum a_j` over the verified positions.
    pub expected_accepted: f64,
    pub next_draft_length: u32,
    /// Tokens this step under full-acceptance accounting (`1 + proposed`) and under the
    /// modelled acceptance (`1 + accepted`).
    pub tokens_full_acceptance: u32,
    pub tokens_accepted: u32,
    /// Always true: the drafter and its acceptance are modelled.
    pub modelled: bool,
}

/// Runs a [`AppKind::DraftVerify`] unit: the acceptance test over a verified batch, the next
/// step's input as its output.
pub struct DraftVerifyExecutor;

impl WorkExecutor for DraftVerifyExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::DraftVerify
    }

    fn execute(
        &self,
        unit: &WorkUnit,
        inputs: &WorkInputs,
        _ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: DraftVerifyParams = unit.params_json().map_err(|e| e.to_string())?;
        let batch = inputs.activation_batch(unit)?;
        let rows = batch
            .row_tensors()
            .ok_or_else(|| "a verify batch needs rows that are whole blocks".to_string())?;
        if rows.len() != 1 + params.draft_length as usize {
            return Err(format!(
                "verify batch holds {} rows but the unit drafted {}",
                rows.len(),
                params.draft_length
            ));
        }
        if params.confidences.len() < params.draft_length as usize {
            return Err(format!(
                "{} confidences for {} draft positions",
                params.confidences.len(),
                params.draft_length
            ));
        }
        // The longest prefix of draft positions the modelled acceptance test passes, each coin
        // a function of the verified row's own bytes.
        let mut accepted = 0u32;
        for k in 1..=params.draft_length {
            let digest = LayerReference::boundary_digest(params.layer, &rows[k as usize]);
            if ModelledDrafter::accepts(params.confidences[k as usize - 1], digest, params.seed, k)
            {
                accepted = k;
            } else {
                break;
            }
        }
        let chosen = &rows[accepted as usize];
        let next = verify_batch(
            chosen,
            batch.dim,
            params.seed,
            params.step + 1,
            params.next_draft_length,
        )?;
        let draft = SessionDraft {
            session_id: params.session_id.clone(),
            confidences: params.confidences[..params.draft_length as usize].to_vec(),
        };
        let report = DraftVerifyReport {
            session_id: params.session_id.clone(),
            step: params.step,
            proposed: params.draft_length,
            accepted,
            survival: draft.survival(),
            expected_accepted: draft.expected_accepted(),
            confidences: draft.confidences,
            next_draft_length: params.next_draft_length,
            tokens_full_acceptance: 1 + params.draft_length,
            tokens_accepted: 1 + accepted,
            modelled: true,
        };
        let digest = LayerReference::boundary_digest(params.layer, chosen);
        let graph_id = unit.graph_id.as_deref().unwrap_or("graph");
        Ok(WorkOutput {
            outputs: vec![(
                StepGraphBuilder::output_ref(graph_id, &unit.id),
                next.to_le_bytes(),
            )],
            payload: serde_json::to_vec(&report).unwrap_or_default(),
            digest_fnv1a: digest,
            layer_digests: vec![LayerDigest::new(params.layer, digest)],
            check_value: accepted as f64,
            units_processed: rows.len() as u64,
            accessed: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_rows_are_deterministic_and_the_verify_batch_splits_back_into_its_rows() {
        let base = prompt_seed_hidden("a session row", 64);
        let a = draft_rows(&base, 7, 3, 3);
        let b = draft_rows(&base, 7, 3, 3);
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
        assert_ne!(a[0], a[1]);
        assert_ne!(
            draft_rows(&base, 7, 4, 1)[0],
            a[0],
            "another step drafts other rows"
        );
        assert_ne!(
            draft_rows(&base, 8, 3, 1)[0],
            a[0],
            "another seed drafts other rows"
        );
        assert!(a.iter().flatten().all(|v| v.abs() <= 8.0));

        // The verify batch drafts from the row as the executor has it: dequantized.
        let row = BlockwiseInt8Tensor::quantize(&base, LAYER_BOUNDARY_BLOCK);
        let from_row = draft_rows(&row.dequantize(), 7, 3, 3);
        let batch = verify_batch(&row, 64, 7, 3, 3).unwrap();
        assert_eq!(batch.rows, 4);
        let rows = batch.row_tensors().unwrap();
        assert_eq!(
            rows[0].to_le_bytes(),
            row.to_le_bytes(),
            "the session's row is exact"
        );
        assert_eq!(
            rows[1].to_le_bytes(),
            BlockwiseInt8Tensor::quantize(&from_row[0], LAYER_BOUNDARY_BLOCK).to_le_bytes()
        );
        let none = verify_batch(&row, 64, 7, 3, 0).unwrap();
        assert_eq!(none.rows, 1);
        assert_eq!(none.tensor.to_le_bytes(), row.to_le_bytes());
    }
}
