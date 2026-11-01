//! Differential check against the Lean model of FNV-1a 64 in
//! `formal/KadmiumFormal/RepOps/Digest.lean` for the grid's own copy of the function, `fnv1a_64`,
//! which content-addresses work units. This fixture is written by `lake exe kadmium_vectors`.

use kadmium_grid::fnv1a_64;

const VECTORS: &str = include_str!("fixtures/fnv1a_lean_vectors.txt");

fn hex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex byte"))
        .collect()
}

#[test]
fn work_unit_hashes_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "fnv");
        let bytes = hex(f[1]);
        let expected = u64::from_str_radix(f[2], 16).expect("hash");
        assert_eq!(fnv1a_64(&bytes), expected, "hash differs for {line}");
        checked += 1;
    }
    assert!(checked >= 10, "regenerate with `lake exe kadmium_vectors`");
}
