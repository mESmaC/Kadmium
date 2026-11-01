//! Differential check against the Lean model of checkpoint layer placement in
//! `formal/KadmiumFormal/Sharding/LayerPlacement.lean`, which proves contiguous non-empty
//! full-model layer coverage, exact weight + full-context KV byte accounting within each
//! machine's resident capacity, boundary `TrustedCore` pinning, completeness (failure if and
//! only if no valid placement exists), and minimax relative memory load optimality.
//!
//! This fixture, written by `lake exe kadmium_vectors`, ties the model to
//! `CheckpointMemoryProfile`, `MachineMemoryBudget`, and
//! `LayerPlacementPlanner::plan_with_budgets`.

use kadmium_core::{NodeId, NodeTrustTier};
use kadmium_sharding::layer_placement::{
    CheckpointMemoryProfile, LayerPlacementPlanner, MachineMemoryBudget,
};

const VECTORS: &str = include_str!("fixtures/layer_placement_lean_vectors.txt");

#[test]
fn layer_placement_matches_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "place");
        let max_pos: usize = f[1].parse().expect("max_pos");
        let emb: u64 = f[2].parse().expect("emb");
        let layer_weights: Vec<u64> = f[3]
            .split(',')
            .map(|n| n.parse().expect("layer_weight"))
            .collect();
        let norm: u64 = f[4].parse().expect("norm");
        let head: u64 = f[5].parse().expect("head");
        let tied: bool = match f[6] {
            "1" => true,
            "0" => false,
            other => panic!("unexpected tied flag: {other}"),
        };
        let kv_per_layer: u64 = f[7].parse().expect("kv_per_layer");

        let profile = CheckpointMemoryProfile {
            num_layers: layer_weights.len(),
            max_position_embeddings: max_pos,
            embedding_bytes: emb,
            layer_weight_bytes: layer_weights,
            final_norm_bytes: norm,
            lm_head_bytes: head,
            tie_word_embeddings: tied,
            kv_bytes_per_layer: kv_per_layer,
        };
        profile.validate().expect("profile should be valid");

        let machines: Vec<MachineMemoryBudget> = if f[8] == "-" {
            Vec::new()
        } else {
            f[8].split(',')
                .map(|m| {
                    let parts: Vec<&str> = m.split(':').collect();
                    assert_eq!(parts.len(), 3);
                    let id = parts[0].to_string();
                    let cap: u64 = parts[1].parse().expect("cap");
                    let trust = match parts[2] {
                        "1" => NodeTrustTier::TrustedCore,
                        "0" => NodeTrustTier::VerifiedPeerMiddleLayersOnly,
                        other => panic!("unexpected trusted flag: {other}"),
                    };
                    MachineMemoryBudget::new(NodeId(id), cap, trust)
                })
                .collect()
        };

        let result = LayerPlacementPlanner::plan_with_budgets(&profile, &machines);
        if f[9] == "none" {
            assert!(
                result.is_err(),
                "expected placement failure for line `{line}`, got {result:?}"
            );
        } else {
            let plan = result.unwrap_or_else(|e| {
                panic!("expected placement success for line `{line}`, got error: {e}")
            });
            let actual = plan
                .stages
                .iter()
                .map(|st| {
                    format!(
                        "{}:{}-{}:{}:{}:{}:{}:{}",
                        st.node_id.0,
                        st.layers.start,
                        st.layers.end,
                        u8::from(st.embedding),
                        u8::from(st.head),
                        st.weight_bytes,
                        st.kv_bytes,
                        st.total_bytes
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            assert_eq!(actual, f[9], "mismatch on line `{line}`");
        }
        checked += 1;
    }
    assert!(checked >= 10, "regenerate with `lake exe kadmium_vectors`");
}
