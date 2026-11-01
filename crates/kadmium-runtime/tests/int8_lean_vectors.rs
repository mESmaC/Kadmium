//! Differential check against the Lean model of INT8 accumulation in
//! `formal/KadmiumFormal/RepOps/Int8.lean`, which proves the sums are order-independent and fit an
//! `i64`. The model computes in unbounded integers; this fixture, written by
//! `lake exe kadmium_vectors`, ties it to `int8_dot` and `int8_sum_squares`.

use kadmium_runtime::repops::{int8_dot, int8_sum_squares};

const VECTORS: &str = include_str!("fixtures/int8_lean_vectors.txt");

/// SplitMix64 of `i + 1`, as `splitmix` in `formal/Fixtures.lean`.
fn splitmix(i: u64) -> u64 {
    let mut z = i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `byteAt` in `formal/Fixtures.lean`, read as an `i8`.
fn code_at(seed: u64, j: u64) -> i8 {
    ((splitmix(seed * 1_000_003 + j) % 256) as u8) as i8
}

#[test]
fn integer_sums_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "int8");
        let n: u64 = f[1].parse().expect("count");
        let seed: u64 = f[2].parse().expect("seed");
        let dot: i64 = f[3].parse().expect("dot");
        let squares: i64 = f[4].parse().expect("sum of squares");

        let a: Vec<i8> = (0..n).map(|j| code_at(seed, j)).collect();
        let b: Vec<i8> = (0..n).map(|j| code_at(seed + 1, j)).collect();
        assert_eq!(int8_dot(&a, &b), dot, "dot differs for {line}");
        assert_eq!(
            int8_sum_squares(&a),
            squares,
            "sum of squares differs for {line}"
        );

        // Order independence, as `dot_perm` states: a reversed and a chunked accumulation agree.
        let reversed: i64 = a
            .iter()
            .zip(&b)
            .rev()
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum();
        assert_eq!(reversed, dot);
        let chunked: i64 = a
            .chunks(7)
            .zip(b.chunks(7))
            .map(|(ca, cb)| int8_dot(ca, cb))
            .sum();
        assert_eq!(chunked, dot);
        checked += 1;
    }
    assert!(checked >= 6, "regenerate with `lake exe kadmium_vectors`");
}
