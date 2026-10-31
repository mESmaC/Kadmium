//! Host reference for every kernel: the same lane functions, evaluated in the same order a warp
//! evaluates them, so a kernel's output must match these bit for bit.
//!
//! A reference here is the contract, not an approximation: the GPU parity harness in
//! `device/` fails on a single differing bit, and the runtime's tests hold these references to
//! the runtime's own `Q8KBlockVector`, `Q4KBlockTensor` and `repops` paths.

use crate::formats::{
    self, BLOCK, MXFP4_BLOCK_BYTES, NIBBLE_BYTES, q4k_code_dot, q4k_q8k_block_term,
};
use crate::fp::{Fp, HostFp};
use crate::math::{max_ignore_nan, silu};
use crate::routing::{better, eligible, router_score, row_mask_value};
use crate::tree::chunked_tree_sum;

/// `Q8_K` quantized activations, field for field Kadmium's `Q8KBlockVector`.
#[derive(Debug, Clone, PartialEq)]
pub struct Q8KActivations {
    pub scales: Vec<f32>,
    pub qs: Vec<i8>,
    pub bsums: Vec<i32>,
}

impl Q8KActivations {
    pub fn view(&self) -> Q8KView<'_> {
        Q8KView {
            scales: &self.scales,
            qs: &self.qs,
            bsums: &self.bsums,
        }
    }
}

/// Borrowed `Q8_K` activations.
#[derive(Debug, Clone, Copy)]
pub struct Q8KView<'a> {
    pub scales: &'a [f32],
    pub qs: &'a [i8],
    pub bsums: &'a [i32],
}

/// Borrowed `Q4_K` weight matrix, field for field Kadmium's `Q4KBlockTensor`.
#[derive(Debug, Clone, Copy)]
pub struct Q4KView<'a> {
    pub rows: usize,
    pub cols: usize,
    pub packed_nibbles: &'a [u8],
    pub block_scales: &'a [f32],
    pub block_mins: &'a [f32],
}

/// Borrowed `MXFP4` weight matrix: `rows * cols / 32` blocks of 17 bytes, row-major.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp4View<'a> {
    pub rows: usize,
    pub cols: usize,
    pub blocks: &'a [u8],
}

/// `q8k_quantize`: one warp per 32-element block; the absmax is an order-free NaN-ignoring
/// max and the code sum is an integer sum, so only the per-element rounding is pinned.
pub fn q8k_quantize(x: &[f32]) -> Q8KActivations {
    let blocks = x.len().div_ceil(BLOCK);
    let mut out = Q8KActivations {
        scales: Vec::with_capacity(blocks),
        qs: Vec::with_capacity(x.len()),
        bsums: Vec::with_capacity(blocks),
    };
    for chunk in x.chunks(BLOCK) {
        let absmax = chunk
            .iter()
            .fold(0.0f32, |acc, v| max_ignore_nan(acc, v.abs()));
        match formats::q8k_scales::<HostFp>(absmax) {
            None => {
                out.scales.push(0.0);
                out.bsums.push(0);
                out.qs.extend(core::iter::repeat_n(0i8, chunk.len()));
            }
            Some((d, inv_d)) => {
                let mut bsum = 0i32;
                for &v in chunk {
                    let q = formats::q8k_code::<HostFp>(v, inv_d);
                    bsum += q as i32;
                    out.qs.push(q);
                }
                out.scales.push(d);
                out.bsums.push(bsum);
            }
        }
    }
    out
}

/// `q4k_q8k_matvec`: one warp per row, lane `j` computing block `32c + j`'s term, the terms
/// reduced by the RepOps tree. Requires `cols % 32 == 0`, the aligned layout real checkpoints
/// use; the runtime keeps its scalar path for anything else.
pub fn q4k_q8k_matvec(w: Q4KView<'_>, x: Q8KView<'_>) -> Vec<f32> {
    assert!(
        w.cols.is_multiple_of(BLOCK),
        "q4k_q8k_matvec needs cols % 32 == 0"
    );
    let blocks_per_row = w.cols / BLOCK;
    (0..w.rows)
        .map(|row| {
            let terms: Vec<f32> = (0..blocks_per_row)
                .map(|b| {
                    let wb = row * blocks_per_row + b;
                    let dot = q4k_code_dot(
                        &w.packed_nibbles[wb * NIBBLE_BYTES..(wb + 1) * NIBBLE_BYTES],
                        &x.qs[b * BLOCK..(b + 1) * BLOCK],
                    );
                    q4k_q8k_block_term::<HostFp>(
                        x.scales[b],
                        w.block_scales[wb],
                        w.block_mins[wb],
                        x.bsums[b],
                        dot,
                    )
                })
                .collect();
            chunked_tree_sum::<HostFp>(&terms)
        })
        .collect()
}

/// `mxfp4_matvec`: the RepOps dot product of each dequantized row with `x`: every product
/// rounded once, then the tree. One `MXFP4` block is exactly one warp chunk.
pub fn mxfp4_matvec(w: Mxfp4View<'_>, x: &[f32]) -> Vec<f32> {
    assert!(
        w.cols.is_multiple_of(BLOCK),
        "mxfp4_matvec needs cols % 32 == 0"
    );
    assert_eq!(x.len(), w.cols);
    let blocks_per_row = w.cols / BLOCK;
    (0..w.rows)
        .map(|row| {
            let products: Vec<f32> = (0..w.cols)
                .map(|c| {
                    let block = row * blocks_per_row + c / BLOCK;
                    let bytes =
                        &w.blocks[block * MXFP4_BLOCK_BYTES..(block + 1) * MXFP4_BLOCK_BYTES];
                    HostFp::mul(formats::mxfp4_weight::<HostFp>(bytes, c % BLOCK), x[c])
                })
                .collect();
            chunked_tree_sum::<HostFp>(&products)
        })
        .collect()
}

/// `swiglu`: `SiLU(gate) * up`, elementwise, with the RepOps `exp`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| HostFp::mul(silu::<HostFp>(g), u))
        .collect()
}

/// `moe_top_k`: per row of `n_experts` logits, the `k` best experts in rank order (score
/// descending, then id ascending), with their canonical scores.
pub fn moe_top_k(logits: &[f32], n_experts: usize, k: usize) -> (Vec<i32>, Vec<f32>) {
    assert!(n_experts > 0 && logits.len().is_multiple_of(n_experts));
    assert!(k <= n_experts);
    let rows = logits.len() / n_experts;
    let mut ids = Vec::with_capacity(rows * k);
    let mut scores = Vec::with_capacity(rows * k);
    for row in logits.chunks(n_experts) {
        let mut previous = None;
        for _ in 0..k {
            let mut best = None;
            for (e, &logit) in row.iter().enumerate() {
                let candidate = (router_score(logit), e as u32);
                if eligible(candidate, previous) {
                    best = better(best, Some(candidate));
                }
            }
            let (score, id) = best.expect("k <= n_experts leaves a candidate every round");
            ids.push(id as i32);
            scores.push(score);
            previous = best;
        }
    }
    (ids, scores)
}

/// `row_mask`: `rows x width` additive mask, row `r` visible up to `visible[r]` columns.
pub fn row_mask(visible: &[i32], width: usize) -> Vec<f32> {
    visible
        .iter()
        .flat_map(|&v| (0..width).map(move |c| row_mask_value(c as u32, v)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::SplitMix64;
    use crate::tree::tree_sum;

    #[test]
    fn q4k_matvec_is_the_canonical_tree_over_block_terms() {
        let mut rng = SplitMix64::new(11);
        let (rows, cols) = (5, 32 * 40);
        let x = q8k_quantize(&rng.activations(cols));
        let packed = rng.bytes(rows * cols / 2);
        let scales: Vec<f32> = (0..rows * cols / 32)
            .map(|_| rng.unit_f32().abs())
            .collect();
        let mins: Vec<f32> = (0..rows * cols / 32).map(|_| rng.unit_f32()).collect();
        let w = Q4KView {
            rows,
            cols,
            packed_nibbles: &packed,
            block_scales: &scales,
            block_mins: &mins,
        };
        let y = q4k_q8k_matvec(w, x.view());
        // Recompute row 3 with the recursive tree directly.
        let terms: Vec<f32> = (0..cols / 32)
            .map(|b| {
                let wb = 3 * (cols / 32) + b;
                let dot = q4k_code_dot(&packed[wb * 16..wb * 16 + 16], &x.qs[b * 32..b * 32 + 32]);
                q4k_q8k_block_term::<HostFp>(x.scales[b], scales[wb], mins[wb], x.bsums[b], dot)
            })
            .collect();
        assert_eq!(y[3].to_bits(), tree_sum::<HostFp>(&terms).to_bits());
    }

    #[test]
    fn mxfp4_matvec_matches_dequantized_dot() {
        let mut rng = SplitMix64::new(3);
        let (rows, cols) = (3, 96);
        let mut blocks = rng.bytes(rows * cols / 32 * MXFP4_BLOCK_BYTES);
        for block in blocks.chunks_mut(MXFP4_BLOCK_BYTES) {
            block[0] = 120 + (block[0] % 16);
        }
        let x: Vec<f32> = (0..cols).map(|_| rng.unit_f32()).collect();
        let y = mxfp4_matvec(
            Mxfp4View {
                rows,
                cols,
                blocks: &blocks,
            },
            &x,
        );
        for (row, &value) in y.iter().enumerate() {
            let want: f64 = (0..cols)
                .map(|c| {
                    let b = row * cols / 32 + c / 32;
                    let w = formats::mxfp4_weight::<HostFp>(&blocks[b * 17..b * 17 + 17], c % 32);
                    w as f64 * x[c] as f64
                })
                .sum();
            assert!((value as f64 - want).abs() <= 1e-4 * want.abs().max(1.0));
        }
    }

    #[test]
    fn top_k_ranks_ties_by_id_and_nan_last_among_finite() {
        let logits = [1.0, 3.0, f32::NAN, 3.0, f32::NEG_INFINITY, 2.0];
        let (ids, scores) = moe_top_k(&logits, 6, 6);
        assert_eq!(ids, vec![1, 3, 5, 0, 2, 4]);
        assert_eq!(scores[4], -f32::MAX);
        assert_eq!(scores[5], f32::NEG_INFINITY);
    }

    #[test]
    fn row_mask_marks_visible_prefix() {
        let mask = row_mask(&[0, 2, 5], 3);
        let inf = f32::NEG_INFINITY;
        assert_eq!(mask, vec![inf, inf, inf, 0.0, 0.0, inf, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn q8k_zero_and_partial_blocks() {
        let mut x = vec![0.0f32; 40];
        x[33] = -4.0;
        let q = q8k_quantize(&x);
        assert_eq!(q.scales[0], 0.0);
        assert_eq!(q.bsums, vec![0, -127]);
        assert_eq!(q.qs.len(), 40);
        assert_eq!(q.qs[33], -127);
    }
}
