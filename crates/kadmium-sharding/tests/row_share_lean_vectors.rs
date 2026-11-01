//! Differential check against the Lean model of round-robin row shares in
//! `formal/KadmiumFormal/Sharding/RowShare.lean`, which proves that every row is served by exactly
//! one copy of a replicated expert and that the copies' shares add up to the batch. This fixture,
//! written by `lake exe kadmium_vectors`, ties the model to `ExpertPlacementPlan::row_share` and
//! `ExpertRowShare::serves`.

use kadmium_sharding::grid_plan::{ExpertPlacementPlan, ExpertRowShare};

const VECTORS: &str = include_str!("fixtures/row_share_lean_vectors.txt");

#[test]
fn shares_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "share");
        let rows: u32 = f[1].parse().expect("rows");
        let copies: usize = f[2].parse().expect("copies");
        let mine: Vec<u32> = f[3].split(',').map(|n| n.parse().expect("count")).collect();
        assert_eq!(mine.len(), copies);

        for (index, &expected) in mine.iter().enumerate() {
            let share = ExpertRowShare {
                expert: 0,
                index: index as u32,
                count: copies as u32,
            };
            let served = (0..rows).filter(|&row| share.serves(row)).count() as u32;
            assert_eq!(
                served, expected,
                "copy {index} of {copies} over {rows} rows"
            );
            let fraction = ExpertPlacementPlan::row_share(rows, copies, index);
            assert_eq!(
                (fraction * f64::from(rows)).round() as u32,
                expected,
                "row_share fraction for copy {index} of {copies} over {rows} rows"
            );
        }
        assert_eq!(
            mine.iter().sum::<u32>(),
            rows,
            "shares do not add up for {line}"
        );
        for row in 0..rows {
            let servers = (0..copies)
                .filter(|&index| {
                    ExpertRowShare {
                        expert: 0,
                        index: index as u32,
                        count: copies as u32,
                    }
                    .serves(row)
                })
                .count();
            assert_eq!(
                servers, 1,
                "row {row} has {servers} servers among {copies} copies"
            );
        }
        checked += 1;
    }
    assert!(checked >= 10, "regenerate with `lake exe kadmium_vectors`");
}
