//! Differential check against the Lean model of `RedundancyPolicy::normalized` in
//! `formal/KadmiumFormal/Grid/Redundancy.lean`, which proves normalisation establishes
//! `1 <= min_quorum <= target_replicas <= max_replicas`, is idempotent, and never lowers a count.
//! This fixture, written by `lake exe kadmium_vectors`, ties the model to the Rust.

use kadmium_grid::{RedundancyPolicy, ValidationRule};

const VECTORS: &str = include_str!("fixtures/redundancy_lean_vectors.txt");

#[test]
fn normalisation_matches_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "policy");
        let nums: Vec<u32> = f[1..7].iter().map(|n| n.parse().expect("count")).collect();
        let policy = RedundancyPolicy {
            min_quorum: nums[0],
            target_replicas: nums[1],
            max_replicas: nums[2],
            validation: ValidationRule::BitExactDigest,
        };
        let normalized = policy.clone().normalized();
        assert_eq!(
            normalized.min_quorum, nums[3],
            "min_quorum differs for {line}"
        );
        assert_eq!(
            normalized.target_replicas, nums[4],
            "target differs for {line}"
        );
        assert_eq!(normalized.max_replicas, nums[5], "max differs for {line}");
        assert!(1 <= normalized.min_quorum);
        assert!(normalized.min_quorum <= normalized.target_replicas);
        assert!(normalized.target_replicas <= normalized.max_replicas);
        let twice = normalized.clone().normalized();
        assert_eq!(twice.min_quorum, normalized.min_quorum);
        assert_eq!(twice.target_replicas, normalized.target_replicas);
        assert_eq!(twice.max_replicas, normalized.max_replicas);
        assert!(policy.min_quorum <= normalized.min_quorum);
        assert!(policy.target_replicas <= normalized.target_replicas);
        assert!(policy.max_replicas <= normalized.max_replicas);
        checked += 1;
    }
    assert!(checked >= 8, "regenerate with `lake exe kadmium_vectors`");
}
