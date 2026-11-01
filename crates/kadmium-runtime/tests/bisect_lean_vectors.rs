//! Differential check against the Lean model of Verde layer bisection in
//! `formal/KadmiumFormal/Runtime/Bisect.lean`, which proves the search returns a layer whose
//! output diverges while its input does not, in at most `ceil(log2 L) + 1` referee evaluations,
//! without assuming divergence is monotone along the trajectory. This fixture, written by
//! `lake exe kadmium_vectors`, ties the model to `VerdeVerificationEngine::bisect_faulty_layer`.

use kadmium_runtime::VerdeVerificationEngine;

const VECTORS: &str = include_str!("fixtures/bisect_lean_vectors.txt");

#[test]
fn bisection_matches_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "bisect");
        let layers: usize = f[1].parse().expect("layer count");
        let bits: Vec<bool> = f[2].chars().map(|c| c == '1').collect();
        assert_eq!(bits.len(), layers + 1);

        // State `i` diverges exactly when its bit is set: the worker holds `1.0` where the
        // reference holds `0.0`.
        let worker: Vec<Vec<f64>> = bits
            .iter()
            .map(|&d| vec![if d { 1.0 } else { 0.0 }])
            .collect();
        let reference: Vec<Vec<f64>> = vec![vec![0.0]; layers + 1];
        // The referee recomputes the reference, so it confirms every reported divergence.
        let referee = |_layer: usize, _input: &[f64]| vec![0.0];
        let outcome =
            VerdeVerificationEngine::bisect_faulty_layer(&worker, &reference, referee, 0.5);

        if f[3] == "none" {
            assert!(outcome.is_none(), "model finds nothing for {line}");
        } else {
            let faulty: usize = f[3].parse().expect("faulty layer");
            let evaluations: usize = f[4].parse().expect("evaluations");
            let outcome = outcome.expect("model finds a layer");
            assert_eq!(outcome.total_layers, layers);
            assert_eq!(
                outcome.faulty_layer_index, faulty,
                "layer differs for {line}"
            );
            assert_eq!(
                outcome.referee_evaluations_used, evaluations,
                "evaluation count differs for {line}"
            );
            assert!(bits[faulty + 1], "the reported layer's output must diverge");
            assert!(
                faulty == 0 || !bits[faulty],
                "the reported layer's input must match"
            );
            let bound = usize::BITS - (layers - 1).max(1).leading_zeros();
            assert!(
                evaluations <= bound as usize + 1 || layers == 1,
                "{evaluations} evaluations exceed ceil(log2 {layers}) + 1"
            );
        }
        checked += 1;
    }
    assert!(checked >= 20, "regenerate with `lake exe kadmium_vectors`");
}
