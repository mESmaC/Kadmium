//! Kadmium's CUDA kernels in pure Rust, compiled to PTX by NVIDIA cuda-oxide, and the harness
//! that holds each one to the RepOps host reference in `kadmium_cuda::reference` bit for bit.
//!
//! Every kernel body is a thin SIMT shell (which warp and lane does what, and the shuffles
//! between them) around lane functions from `kadmium_cuda`, instantiated with [`DeviceFp`] so
//! each floating-point operation is an explicit round-to-nearest PTX instruction that `ptxas`
//! may not fuse into an `fma`.
//!
//! ```text
//! cargo oxide run              # parity on the default device, exit status 1 on any differing bit
//! cargo oxide run -- --bench   # also time the two mat-vecs at 4096 x 4096
//! ```

mod parity;

use cuda_device::{DisjointSlice, float, kernel, thread, warp};
use cuda_host::cuda_module;
use kadmium_cuda::fp::Fp;

/// Device arithmetic: `add.rn.f32`, `mul.rn.f32`, `div.rn.f32`.
#[derive(Clone, Copy)]
pub struct DeviceFp;

impl Fp for DeviceFp {
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        float::add_rn_f32(a, b)
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        float::mul_rn_f32(a, b)
    }

    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        float::div_rn_f32(a, b)
    }
}

#[cuda_module]
mod kernels {
    use super::*;
    use kadmium_cuda::formats::{self, BLOCK, MXFP4_BLOCK_BYTES, NIBBLE_BYTES};
    use kadmium_cuda::math::{max_ignore_nan, silu};
    use kadmium_cuda::routing::{better, eligible, router_score, row_mask_value};
    use kadmium_cuda::tree::{TreeStack, combine_lanes};

    /// This thread's warp across the whole 1-D grid, and its lane.
    #[inline(always)]
    fn warp_and_lane() -> (usize, u32) {
        let thread = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        (thread / 32, (thread % 32) as u32)
    }

    /// The RepOps tree over one 32-leaf chunk, lane `j` holding leaf `j`: butterfly distances
    /// 1, 2, 4, 8, 16, so each step joins exactly two sibling subtrees of the canonical tree.
    /// Every lane returns the chunk sum.
    #[inline(always)]
    fn warp_tree(lane: u32, value: f32, present: bool) -> f32 {
        let mut mine = (value, present);
        let mut distance = 1u32;
        while distance < 32 {
            let partner = (
                warp::shuffle_xor_f32(mine.0, distance),
                warp::shuffle_xor(mine.1 as u32, distance) != 0,
            );
            mine = combine_lanes::<DeviceFp>(lane, distance, mine, partner);
            distance <<= 1;
        }
        mine.0
    }

    /// [`warp_tree`] over four full chunks at once, each reduced by its own canonical tree.
    #[inline(always)]
    fn warp_tree_x4(lane: u32, values: [f32; 4]) -> [f32; 4] {
        let mut sums = values;
        let mut distance = 1u32;
        while distance < 32 {
            let partners = [
                warp::shuffle_xor_f32(sums[0], distance),
                warp::shuffle_xor_f32(sums[1], distance),
                warp::shuffle_xor_f32(sums[2], distance),
                warp::shuffle_xor_f32(sums[3], distance),
            ];
            let mut i = 0;
            while i < 4 {
                sums[i] =
                    combine_lanes::<DeviceFp>(lane, distance, (sums[i], true), (partners[i], true))
                        .0;
                i += 1;
            }
            distance <<= 1;
        }
        sums
    }

    /// Integer sum across the warp; exact in any order.
    #[inline(always)]
    fn warp_sum_i32(value: i32) -> i32 {
        let mut sum = value;
        let mut distance = 16u32;
        while distance > 0 {
            sum = sum.wrapping_add(warp::shuffle_xor(sum as u32, distance) as i32);
            distance >>= 1;
        }
        sum
    }

    /// `|x|` by clearing the sign bit.
    #[inline(always)]
    fn abs(x: f32) -> f32 {
        f32::from_bits(x.to_bits() & 0x7fff_ffff)
    }

    /// `sum(q_w * q_x)` for one `Q4_K` block with `dp4a`: each 32-bit weight word holds the
    /// codes of eight elements, even elements in the low nibbles; `prmt` gathers the matching
    /// even and odd activation bytes. Integer, so it equals the scalar loop exactly.
    #[inline(always)]
    fn q4k_code_dot_dp4a(packed: &[u8], weight_byte: usize, codes: &[i8], code: usize) -> i32 {
        // SAFETY: the caller bounds both blocks; device allocations are 256-byte aligned and
        // the offsets are multiples of 16 (weights) and 32 (codes), so the words are aligned.
        unsafe {
            let w = packed.as_ptr().add(weight_byte) as *const u32;
            let x = codes.as_ptr().add(code) as *const u32;
            let mut dot = 0i32;
            let mut word = 0usize;
            while word < NIBBLE_BYTES / 4 {
                let weights = *w.add(word);
                let lo = *x.add(2 * word);
                let hi = *x.add(2 * word + 1);
                let even = cuda_device::prmt::prmt(lo, hi, 0x6420);
                let odd = cuda_device::prmt::prmt(lo, hi, 0x7531);
                dot = cuda_device::dotprod::dp4a_s32(weights & 0x0f0f_0f0f, even, dot);
                dot = cuda_device::dotprod::dp4a_s32((weights >> 4) & 0x0f0f_0f0f, odd, dot);
                word += 1;
            }
            dot
        }
    }

    /// Quantizes activations to `Q8_K`, one warp per 32-element block.
    #[kernel]
    pub fn q8k_quantize(
        x: &[f32],
        mut qs: DisjointSlice<i8>,
        mut scales: DisjointSlice<f32>,
        mut bsums: DisjointSlice<i32>,
    ) {
        let (block, lane) = warp_and_lane();
        if block >= scales.len() {
            return;
        }
        let idx = block * BLOCK + lane as usize;
        let present = idx < x.len();
        let value = if present { x[idx] } else { 0.0 };

        let mut absmax = abs(value);
        let mut distance = 16u32;
        while distance > 0 {
            absmax = max_ignore_nan(absmax, warp::shuffle_xor_f32(absmax, distance));
            distance >>= 1;
        }

        let (d, code) = match formats::q8k_scales::<DeviceFp>(absmax) {
            None => (0.0, 0i8),
            Some((d, inv_d)) => (d, formats::q8k_code::<DeviceFp>(value, inv_d)),
        };
        let code = if present { code } else { 0 };
        let bsum = warp_sum_i32(code as i32);

        // SAFETY: `idx < x.len() == qs.len()` and `block < scales.len() == bsums.len()`; each
        // index is written by exactly one thread.
        unsafe {
            if present && idx < qs.len() {
                *qs.get_unchecked_mut(idx) = code;
            }
            if lane == 0 {
                *scales.get_unchecked_mut(block) = d;
                if block < bsums.len() {
                    *bsums.get_unchecked_mut(block) = bsum;
                }
            }
        }
    }

    /// `y = W x` for `Q4_K` weights and `Q8_K` activations, one warp per row.
    #[kernel]
    pub fn q4k_q8k_matvec(
        packed: &[u8],
        w_scales: &[f32],
        w_mins: &[f32],
        x_scales: &[f32],
        x_qs: &[i8],
        x_bsums: &[i32],
        mut y: DisjointSlice<f32>,
    ) {
        let (row, lane) = warp_and_lane();
        if row >= y.len() {
            return;
        }
        let blocks_per_row = x_scales.len();
        let mut stack = TreeStack::new();
        let mut chunk = 0usize;
        while chunk * 32 < blocks_per_row {
            let b = chunk * 32 + lane as usize;
            let present = b < blocks_per_row;
            let term = if present {
                let wb = row * blocks_per_row + b;
                let dot = q4k_code_dot_dp4a(packed, wb * NIBBLE_BYTES, x_qs, b * BLOCK);
                formats::q4k_q8k_block_term::<DeviceFp>(
                    x_scales[b],
                    w_scales[wb],
                    w_mins[wb],
                    x_bsums[b],
                    dot,
                )
            } else {
                0.0
            };
            stack.push::<DeviceFp>(warp_tree(lane, term, present));
            chunk += 1;
        }
        if lane == 0 {
            // SAFETY: `row < y.len()`, one writer per row.
            unsafe { *y.get_unchecked_mut(row) = stack.finish::<DeviceFp>() };
        }
    }

    /// `y = W x` for `MXFP4` weights, one warp per row, one block per warp step.
    #[kernel]
    pub fn mxfp4_matvec(blocks: &[u8], x: &[f32], mut y: DisjointSlice<f32>) {
        let (row, lane) = warp_and_lane();
        if row >= y.len() {
            return;
        }
        let blocks_per_row = x.len() / BLOCK;
        let product = |b: usize| {
            let start = (row * blocks_per_row + b) * MXFP4_BLOCK_BYTES;
            let block = &blocks[start..start + MXFP4_BLOCK_BYTES];
            let w = formats::mxfp4_weight::<DeviceFp>(block, lane as usize);
            DeviceFp::mul(w, x[b * BLOCK + lane as usize])
        };
        // Two levels of the canonical tree in registers: each block's 32 products reduce to a
        // block sum, lane `g` keeps the sum of block `g` of a 32-block group, and the group's
        // 32 sums reduce again, so the stack sees one aligned 1024-element subtree per group.
        // Four block trees per step keep four shuffle chains in flight.
        let mut stack = TreeStack::new();
        let mut group = 0usize;
        while group < blocks_per_row {
            let len = (blocks_per_row - group).min(32);
            let mut mine = 0.0f32;
            let mut g = 0usize;
            while g + 4 <= len {
                let b = group + g;
                let sums = warp_tree_x4(
                    lane,
                    [product(b), product(b + 1), product(b + 2), product(b + 3)],
                );
                let mut i = 0;
                while i < 4 {
                    if lane as usize == g + i {
                        mine = sums[i];
                    }
                    i += 1;
                }
                g += 4;
            }
            while g < len {
                let sum = warp_tree(lane, product(group + g), true);
                if lane as usize == g {
                    mine = sum;
                }
                g += 1;
            }
            stack.push::<DeviceFp>(warp_tree(lane, mine, (lane as usize) < len));
            group += len;
        }
        if lane == 0 {
            // SAFETY: `row < y.len()`, one writer per row.
            unsafe { *y.get_unchecked_mut(row) = stack.finish::<DeviceFp>() };
        }
    }

    /// `out = SiLU(gate) * up`, one thread per element.
    #[kernel]
    pub fn swiglu(gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = DeviceFp::mul(silu::<DeviceFp>(gate[i]), up[i]);
        }
    }

    /// The `k` best experts of each row of `logits`, in rank order, one warp per row.
    #[kernel]
    pub fn moe_top_k(
        logits: &[f32],
        n_experts: u32,
        k: u32,
        mut ids: DisjointSlice<i32>,
        mut scores: DisjointSlice<f32>,
    ) {
        let (row, lane) = warp_and_lane();
        let n = n_experts as usize;
        if n == 0 || row >= logits.len() / n {
            return;
        }
        let base = row * n;
        let mut previous: Option<(f32, u32)> = None;
        let mut round = 0u32;
        while round < k {
            let mut best: Option<(f32, u32)> = None;
            let mut e = lane;
            while (e as usize) < n {
                let candidate = (router_score(logits[base + e as usize]), e);
                if eligible(candidate, previous) {
                    best = better(best, Some(candidate));
                }
                e += 32;
            }
            // `better` is a strict total order on (score, id), so the butterfly's pairing
            // order cannot change the winner.
            let mut distance = 16u32;
            while distance > 0 {
                let (score, id, has) = match best {
                    Some((s, i)) => (s, i, 1u32),
                    None => (0.0, 0, 0),
                };
                let partner_score = warp::shuffle_xor_f32(score, distance);
                let partner_id = warp::shuffle_xor(id, distance);
                let partner_has = warp::shuffle_xor(has, distance) != 0;
                if partner_has {
                    best = better(best, Some((partner_score, partner_id)));
                }
                distance >>= 1;
            }
            if lane == round {
                if let Some((score, id)) = best {
                    let out = row * k as usize + round as usize;
                    // SAFETY: bounds checked; lane `round` alone writes slot `round`.
                    unsafe {
                        if out < ids.len() && out < scores.len() {
                            *ids.get_unchecked_mut(out) = id as i32;
                            *scores.get_unchecked_mut(out) = score;
                        }
                    }
                }
            }
            previous = best;
            round += 1;
        }
    }

    /// Additive attention mask: row `r` visible for its first `visible[r]` columns.
    #[kernel]
    pub fn row_mask(visible: &[i32], width: u32, mut out: DisjointSlice<f32>) {
        let row = thread::blockIdx_y() as usize;
        let column = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if row >= visible.len() || column >= width {
            return;
        }
        let idx = row * width as usize + column as usize;
        if idx < out.len() {
            // SAFETY: in bounds; one thread per (row, column).
            unsafe { *out.get_unchecked_mut(idx) = row_mask_value(column, visible[row]) };
        }
    }
}

fn main() {
    let bench = std::env::args().any(|arg| arg == "--bench");
    match parity::run(bench) {
        Ok(true) => println!("\nall kernels bit-identical to the RepOps reference"),
        Ok(false) => {
            eprintln!("\nparity FAILED: a kernel differs from the RepOps reference");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("CUDA error: {error}");
            std::process::exit(2);
        }
    }
}
