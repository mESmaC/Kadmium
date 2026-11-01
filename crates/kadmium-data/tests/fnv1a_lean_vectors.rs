//! Differential check against the Lean model of FNV-1a 64 in
//! `formal/KadmiumFormal/RepOps/Digest.lean`. The model's constants are checked there against the
//! published reference vectors; this fixture, written by `lake exe kadmium_vectors`, ties the model
//! to `compute_fnv1a64_checksum`, the function every `.ktb` checksum and content address uses.

use kadmium_data::compute_fnv1a64_checksum;

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
fn checksums_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "fnv");
        let bytes = hex(f[1]);
        let expected = u64::from_str_radix(f[2], 16).expect("hash");
        assert_eq!(
            compute_fnv1a64_checksum(&bytes),
            expected,
            "hash differs for {line}"
        );
        checked += 1;
    }
    assert!(checked >= 10, "regenerate with `lake exe kadmium_vectors`");
}

/// The published FNV-1a 64 vectors the Lean model proves by `decide`.
#[test]
fn reference_vectors() {
    assert_eq!(compute_fnv1a64_checksum(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(compute_fnv1a64_checksum(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(compute_fnv1a64_checksum(b"foobar"), 0x8594_4171_f739_67e8);
}
