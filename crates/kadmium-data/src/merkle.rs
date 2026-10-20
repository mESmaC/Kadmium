//! # `kadmium_data::merkle`
//!
//! 64 KiB page-aligned binary Merkle tree (`KtbMerkleTree`), $O(\log N)$ inclusion proofs
//! (`MerkleInclusionProof`), and in-place FTS3 P2P page patching (`Fts3PagePatchEngine`).

use crate::{KTB_PAGE_ALIGNMENT_BYTES, compute_fnv1a64_checksum};
use serde::{Deserialize, Serialize};

/// Cryptographic/FNV-1a Merkle inclusion proof for a single 64 KiB `.ktb` page.
/// Allows a worker node to verify a streamed 64 KiB page from any peer immediately upon arrival.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleInclusionProof {
    pub page_index: usize,
    pub leaf_hash: u64,
    /// Pairs of `(sibling_hash, sibling_is_right)` from leaf to root.
    pub audit_path: Vec<(u64, bool)>,
}

/// Binary Merkle Tree over 64 KiB-aligned pages of a `.ktb` tensor bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KtbMerkleTree {
    pub page_size_bytes: usize,
    pub leaf_hashes: Vec<u64>,
    pub levels: Vec<Vec<u64>>,
    pub root_hash: u64,
}

impl KtbMerkleTree {
    fn hash_pair(left: u64, right: u64) -> u64 {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&left.to_le_bytes());
        buf[8..].copy_from_slice(&right.to_le_bytes());
        compute_fnv1a64_checksum(&buf)
    }

    /// Constructs a binary Merkle tree over `ktb_bytes` sliced at `KTB_PAGE_ALIGNMENT_BYTES` (64 KiB).
    pub fn from_ktb_bytes(ktb_bytes: &[u8]) -> Self {
        let page_size = KTB_PAGE_ALIGNMENT_BYTES as usize;
        let leaf_hashes: Vec<u64> = ktb_bytes
            .chunks(page_size)
            .map(compute_fnv1a64_checksum)
            .collect();

        if leaf_hashes.is_empty() {
            return Self {
                page_size_bytes: page_size,
                leaf_hashes: vec![],
                levels: vec![],
                root_hash: 0,
            };
        }

        let mut levels = vec![leaf_hashes.clone()];
        while levels.last().expect("non-empty levels").len() > 1 {
            let current = levels.last().expect("non-empty levels");
            let mut next_level = Vec::with_capacity(current.len().div_ceil(2));
            for pair in current.chunks(2) {
                let l = pair[0];
                let r = if pair.len() == 2 { pair[1] } else { pair[0] };
                next_level.push(Self::hash_pair(l, r));
            }
            levels.push(next_level);
        }

        let root_hash = levels.last().and_then(|l| l.first()).copied().unwrap_or(0);
        Self {
            page_size_bytes: page_size,
            leaf_hashes,
            levels,
            root_hash,
        }
    }

    /// Generates an $O(\log N)$ Merkle inclusion proof for 64 KiB page `page_index`.
    pub fn prove_page(&self, page_index: usize) -> Option<MerkleInclusionProof> {
        if page_index >= self.leaf_hashes.len() {
            return None;
        }
        let mut idx = page_index;
        let mut audit_path = Vec::new();

        for level in &self.levels[..self.levels.len().saturating_sub(1)] {
            let is_right_child = idx % 2 == 1;
            let sibling_idx = if is_right_child {
                idx - 1
            } else if idx + 1 < level.len() {
                idx + 1
            } else {
                idx
            };
            audit_path.push((level[sibling_idx], !is_right_child));
            idx /= 2;
        }

        Some(MerkleInclusionProof {
            page_index,
            leaf_hash: self.leaf_hashes[page_index],
            audit_path,
        })
    }

    /// Verifies that `page_bytes` at `proof.page_index` hashes to `expected_root`.
    pub fn verify_page(
        page_bytes: &[u8],
        proof: &MerkleInclusionProof,
        expected_root: u64,
    ) -> bool {
        let computed_leaf = compute_fnv1a64_checksum(page_bytes);
        if computed_leaf != proof.leaf_hash {
            return false;
        }
        let mut acc = computed_leaf;
        for &(sibling, sibling_is_right) in &proof.audit_path {
            acc = if sibling_is_right {
                Self::hash_pair(acc, sibling)
            } else {
                Self::hash_pair(sibling, acc)
            };
        }
        acc == expected_root
    }
}

/// Resumable 64 KiB page-level P2P patching engine for FTS3 `.ktb` shard streaming.
pub struct Fts3PagePatchEngine;

impl Fts3PagePatchEngine {
    /// Verifies an incoming 64 KiB page against `expected_merkle_root` and patches it in-place
    /// into `target_ktb_buffer`.
    pub fn verify_and_patch_page(
        target_ktb_buffer: &mut [u8],
        incoming_page_bytes: &[u8],
        proof: &MerkleInclusionProof,
        expected_merkle_root: u64,
    ) -> Result<(), String> {
        if !KtbMerkleTree::verify_page(incoming_page_bytes, proof, expected_merkle_root) {
            return Err(format!(
                "Merkle inclusion proof failed for 64 KiB page #{}",
                proof.page_index
            ));
        }
        let page_size = KTB_PAGE_ALIGNMENT_BYTES as usize;
        let start = proof.page_index * page_size;
        let end = start + incoming_page_bytes.len();
        if end > target_ktb_buffer.len() {
            return Err("Target .ktb buffer too small for page offset".into());
        }
        target_ktb_buffer[start..end].copy_from_slice(incoming_page_bytes);
        Ok(())
    }
}
