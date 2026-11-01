//! Differential check against the Lean model of `ReplicaCatalog::evict_candidates` in
//! `formal/KadmiumFormal/Grid/Eviction.lean`, which proves the loop offers the shortest prefix of
//! the sorted unpinned replicas that frees the requested bytes, or every unpinned replica. This
//! fixture, written by `lake exe kadmium_vectors`, ties the model to the Rust.

use kadmium_core::NodeId;
use kadmium_grid::{ContentKind, ContentRef, Replica, ReplicaCatalog, ResidencyTier};

const VECTORS: &str = include_str!("fixtures/eviction_lean_vectors.txt");

#[test]
fn eviction_prefixes_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "evict");
        let need: u64 = f[1].parse().expect("bytes needed");
        let sizes: Vec<u64> = if f[2] == "-" {
            Vec::new()
        } else {
            f[2].split(',').map(|n| n.parse().expect("size")).collect()
        };
        let count: usize = f[3].parse().expect("count");

        // Replica `i` sorts `i`-th: `(hits, last_access)` grows with the index. A pinned replica
        // with no bytes sits in front and must never be offered.
        let node = NodeId("worker".to_string());
        let mut catalog = ReplicaCatalog::new();
        let pinned = ContentRef::named(ContentKind::ExpertBank, "pinned", 1 << 40);
        catalog.record(Replica::new(pinned, node.clone(), ResidencyTier::Vram, 0).pinned());
        for (i, &bytes) in sizes.iter().enumerate() {
            let content = ContentRef::named(ContentKind::ExpertBank, &format!("c{i}"), bytes);
            let mut replica = Replica::new(content, node.clone(), ResidencyTier::Vram, 0);
            replica.hits = i as u64;
            replica.last_access_unix_ms = i as u64;
            catalog.record(replica);
        }

        let offered = catalog.evict_candidates(&node, need);
        let ids: Vec<String> = offered.iter().map(|c| c.content_id.clone()).collect();
        let expected: Vec<String> = (0..count).map(|i| format!("c{i}")).collect();
        assert_eq!(ids, expected, "offered replicas differ for {line}");
        let freed: u64 = offered.iter().map(|c| c.bytes).sum();
        assert!(
            freed >= need || count == sizes.len(),
            "{freed} bytes offered for {need} needed in {line}"
        );
        if count > 0 {
            let shorter: u64 = sizes[..count - 1].iter().sum();
            assert!(shorter < need, "a shorter prefix would do for {line}");
        }
        checked += 1;
    }
    assert!(checked >= 11, "regenerate with `lake exe kadmium_vectors`");
}
