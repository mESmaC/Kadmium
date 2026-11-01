//! Differential check against the Lean model of XOR page parity in
//! `formal/KadmiumFormal/Network/XorParity.lean`, which proves that the parity XOR the other pages
//! is the missing page for every stripe. This fixture, written by `lake exe kadmium_vectors`, ties
//! the model to `encode_xor_parity_page` and `reconstruct_missing_page_xor`.

use kadmium_network::SwarmRaidStriper;

const VECTORS: &str = include_str!("fixtures/xor_parity_lean_vectors.txt");

/// SplitMix64 of `i + 1`, as `splitmix` in `formal/Fixtures.lean`.
fn splitmix(i: u64) -> u64 {
    let mut z = i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `byteAt` in `formal/Fixtures.lean`.
fn byte_at(seed: u64, j: u64) -> u8 {
    (splitmix(seed * 1_000_003 + j) % 256) as u8
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

#[test]
fn parity_and_recovery_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "stripe");
        let pages: usize = f[1].parse().expect("page count");
        let len: u64 = f[2].parse().expect("page length");
        let missing: usize = f[3].parse().expect("missing index");
        let seed: u64 = f[4].parse().expect("seed");
        let expected_parity = hex(f[5]);
        let expected_recovered = hex(f[6]);

        let stripe: Vec<Vec<u8>> = (0..pages as u64)
            .map(|p| (0..len).map(|j| byte_at(seed + p, j)).collect())
            .collect();
        let parity = SwarmRaidStriper::encode_xor_parity_page(&stripe);
        assert_eq!(parity, expected_parity, "parity differs for {line}");

        let with_hole: Vec<Option<Vec<u8>>> = stripe
            .iter()
            .enumerate()
            .map(|(i, p)| if i == missing { None } else { Some(p.clone()) })
            .collect();
        let recovered =
            SwarmRaidStriper::reconstruct_missing_page_xor(&with_hole, &parity, missing);
        assert_eq!(recovered, expected_recovered, "recovery differs for {line}");
        assert_eq!(recovered, stripe[missing], "recovery is not the lost page");
        checked += 1;
    }
    assert!(checked >= 6, "regenerate with `lake exe kadmium_vectors`");
}
