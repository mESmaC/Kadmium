//! The RepOps pairwise tree, as a warp computes it.
//!
//! RepOps sums `n` values as a perfect binary tree over `next_power_of_two(n)` leaves, each node
//! the sum `left + right` of its children, a child whose leaves all lie past the end absent
//! (its sibling passes up unchanged, so padding never adds a zero, which would turn a `-0.0`
//! into `+0.0`). A kernel reaches the same bits in two stages:
//!
//! 1. **Within a warp**, lane `j` holds leaf `32c + j` of chunk `c`. A butterfly over
//!    `shuffle_xor` distances `1, 2, 4, 8, 16`, in that order, combines exactly the sibling
//!    subtrees of the canonical tree, so after five steps every lane holds the chunk's
//!    subtree sum. [`combine_lanes`] is one step of it.
//! 2. **Across chunks**, the aligned 32-leaf subtrees are the leaves of the tree's upper levels.
//!    [`TreeStack`] folds them as they stream in, merging equal-sized subtrees like a binary
//!    counter and folding the remainder from the right, which is the canonical shape for any
//!    chunk count.
//!
//! [`tree_sum`] is the host reference both stages are tested against; it is the same recursion
//! as `kadmium_runtime::repops::tree_sum`.

use crate::fp::Fp;

/// Lanes in an NVIDIA warp.
pub const WARP: usize = 32;

/// One butterfly step at shuffle distance `distance`: `mine` is this lane's subtree, `partner`
/// the subtree of lane `lane ^ distance`, each with whether any of its leaves is present.
/// The lane with the `distance` bit clear holds the left subtree.
#[inline(always)]
pub fn combine_lanes<F: Fp>(
    lane: u32,
    distance: u32,
    mine: (f32, bool),
    partner: (f32, bool),
) -> (f32, bool) {
    let (left, right) = if lane & distance == 0 {
        (mine, partner)
    } else {
        (partner, mine)
    };
    match (left.1, right.1) {
        (true, true) => (F::add(left.0, right.0), true),
        (true, false) => left,
        (false, _) => right,
    }
}

/// Folds aligned power-of-two subtrees, streamed left to right, into the canonical tree.
/// Holds at most one subtree per size, so 32 slots cover any `u32` count of subtrees.
#[derive(Debug, Clone, Copy)]
pub struct TreeStack {
    sums: [f32; 32],
    /// `log2` of the number of streamed subtrees each slot covers.
    ranks: [u32; 32],
    depth: usize,
}

impl Default for TreeStack {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeStack {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            sums: [0.0; 32],
            ranks: [0; 32],
            depth: 0,
        }
    }

    /// Appends the next subtree to the right of everything pushed so far.
    #[inline(always)]
    pub fn push<F: Fp>(&mut self, value: f32) {
        let mut sum = value;
        let mut rank = 0u32;
        while self.depth > 0 && self.ranks[self.depth - 1] == rank {
            self.depth -= 1;
            sum = F::add(self.sums[self.depth], sum);
            rank += 1;
        }
        self.sums[self.depth] = sum;
        self.ranks[self.depth] = rank;
        self.depth += 1;
    }

    /// The tree sum of everything pushed; `0.0` when nothing was.
    #[inline(always)]
    pub fn finish<F: Fp>(&self) -> f32 {
        if self.depth == 0 {
            return 0.0;
        }
        let mut depth = self.depth - 1;
        let mut sum = self.sums[depth];
        while depth > 0 {
            depth -= 1;
            sum = F::add(self.sums[depth], sum);
        }
        sum
    }
}

/// The canonical RepOps tree sum, by recursion over the index space.
pub fn tree_sum<F: Fp>(values: &[f32]) -> f32 {
    fn node<F: Fp>(values: &[f32], lo: usize, size: usize) -> f32 {
        if size == 1 {
            return values[lo];
        }
        let half = size / 2;
        let left = node::<F>(values, lo, half);
        let mid = lo + half;
        if mid >= values.len() {
            left
        } else {
            F::add(left, node::<F>(values, mid, half))
        }
    }
    if values.is_empty() {
        return 0.0;
    }
    node::<F>(values, 0, values.len().next_power_of_two())
}

/// Host model of one warp's butterfly over a 32-leaf chunk, `present` leaves from the left.
pub fn warp_tree<F: Fp>(leaves: &[f32; WARP], present: usize) -> f32 {
    let mut lanes: [(f32, bool); WARP] = core::array::from_fn(|j| (leaves[j], j < present));
    let mut distance = 1u32;
    while distance < WARP as u32 {
        let before = lanes;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let partner = before[lane ^ distance as usize];
            *slot = combine_lanes::<F>(lane as u32, distance, before[lane], partner);
        }
        distance <<= 1;
    }
    lanes[0].0
}

/// Host model of a whole kernel reduction: warp trees over 32-leaf chunks, then the stack.
pub fn chunked_tree_sum<F: Fp>(values: &[f32]) -> f32 {
    let mut stack = TreeStack::new();
    for chunk in values.chunks(WARP) {
        let mut leaves = [0.0f32; WARP];
        leaves[..chunk.len()].copy_from_slice(chunk);
        stack.push::<F>(warp_tree::<F>(&leaves, chunk.len()));
    }
    stack.finish::<F>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::HostFp;
    use crate::testing::SplitMix64;

    #[test]
    fn warp_and_stack_reproduce_the_canonical_tree() {
        let mut rng = SplitMix64::new(7);
        for n in (0..=200).chain([255, 256, 257, 1000, 1023, 1024, 1025, 4097]) {
            // Wide exponent spread so a different association would change the bits.
            let values: Vec<f32> = (0..n).map(|_| rng.wide_f32()).collect();
            let canonical = tree_sum::<HostFp>(&values);
            let chunked = chunked_tree_sum::<HostFp>(&values);
            assert_eq!(canonical.to_bits(), chunked.to_bits(), "n = {n}");
        }
    }

    #[test]
    fn absent_leaves_never_add_a_zero() {
        // A lone -0.0 must survive: adding a padding +0.0 would flip its sign.
        assert_eq!(tree_sum::<HostFp>(&[-0.0]).to_bits(), (-0.0f32).to_bits());
        assert_eq!(
            chunked_tree_sum::<HostFp>(&[-0.0]).to_bits(),
            (-0.0f32).to_bits()
        );
        let values = vec![-0.0f32; 33];
        assert_eq!(
            chunked_tree_sum::<HostFp>(&values).to_bits(),
            (-0.0f32).to_bits()
        );
    }

    #[test]
    fn empty_sums_to_zero() {
        assert_eq!(tree_sum::<HostFp>(&[]), 0.0);
        assert_eq!(chunked_tree_sum::<HostFp>(&[]), 0.0);
    }
}
