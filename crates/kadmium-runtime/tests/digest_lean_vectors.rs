//! Differential check against the Lean model of RepOps digests in
//! `formal/KadmiumFormal/RepOps/Digest.lean`.
//!
//! The model says `digest_bytes(tag, bytes)` is FNV-1a 64 of
//! `manifest id (8 bytes LE) ++ tag ++ [0] ++ bytes` (`digestBytes_eq_fnv1a`), that `fold_chain`
//! hashes the count as `u32` LE followed by every digest as `u64` LE (`chainBytes`), and that
//! splitting the input across `update` calls never changes a digest (`digestUpdate_append`).
//! The FNV-1a fold itself is tied to the model by the `fnv` vectors `lake exe kadmium_vectors`
//! writes; the structural identities are then checked through the public `Digest` API with the
//! manifest id the running binary reports.

use kadmium_runtime::repops::{Digest, RepOpsManifest, digest_bytes, fold_chain};

const VECTORS: &str = include_str!("fixtures/fnv1a_lean_vectors.txt");

/// FNV-1a 64 as the Lean model folds it; checked below against the model's vectors before it is
/// used as the reference for the digest identities.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn hex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex byte"))
        .collect()
}

fn payloads() -> Vec<Vec<u8>> {
    VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(f[0], "fnv");
            let bytes = hex(f[1]);
            let expected = u64::from_str_radix(f[2], 16).expect("hash");
            assert_eq!(fnv1a(&bytes), expected, "reference fold differs for {line}");
            bytes
        })
        .collect()
}

/// `manifest id (8 bytes LE) ++ tag ++ [0]`, the prefix `Digest::new` hashes.
fn preimage(tag: &str) -> Vec<u8> {
    let mut bytes = RepOpsManifest::id().to_le_bytes().to_vec();
    bytes.extend_from_slice(tag.as_bytes());
    bytes.push(0);
    bytes
}

#[test]
fn digest_bytes_is_fnv1a_of_the_modelled_preimage() {
    let payloads = payloads();
    assert!(
        payloads.len() >= 10,
        "regenerate with `lake exe kadmium_vectors`"
    );
    for tag in ["", "unit", "layer-digest", "tag with spaces"] {
        for payload in &payloads {
            let mut bytes = preimage(tag);
            bytes.extend_from_slice(payload);
            assert_eq!(
                digest_bytes(tag, payload),
                fnv1a(&bytes),
                "digest_bytes differs for tag {tag:?} and {} payload bytes",
                payload.len()
            );
        }
    }
}

#[test]
fn chunking_never_changes_a_digest() {
    for payload in payloads() {
        for split in [0, 1, payload.len() / 2, payload.len()] {
            let (a, b) = payload.split_at(split.min(payload.len()));
            assert_eq!(
                Digest::new("chunked").update(a).update(b).finish(),
                digest_bytes("chunked", &payload),
                "split at {split} changes the digest of {} bytes",
                payload.len()
            );
        }
        let mut typed = Digest::new("typed");
        let mut plain = Digest::new("typed");
        let words: Vec<u32> = (0..payload.len() / 4)
            .map(|i| {
                u32::from_le_bytes([
                    payload[4 * i],
                    payload[4 * i + 1],
                    payload[4 * i + 2],
                    payload[4 * i + 3],
                ])
            })
            .collect();
        for w in &words {
            typed.update_u32(*w);
        }
        plain.update(&payload[..words.len() * 4]);
        assert_eq!(typed.finish(), plain.finish());
    }
}

#[test]
fn fold_chain_hashes_the_modelled_chain_bytes() {
    let chains: Vec<Vec<u64>> = vec![
        vec![],
        vec![0],
        vec![u64::MAX],
        payloads().iter().map(|p| fnv1a(p)).collect(),
    ];
    for chain in &chains {
        let mut bytes = preimage("chain");
        bytes.extend_from_slice(&(chain.len() as u32).to_le_bytes());
        for d in chain {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        assert_eq!(
            fold_chain("chain", chain),
            fnv1a(&bytes),
            "fold_chain differs for {} digests",
            chain.len()
        );
    }
}
