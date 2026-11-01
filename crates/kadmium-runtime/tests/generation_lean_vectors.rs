//! Differential check against the Lean model of streamed-text holdback in
//! `formal/KadmiumFormal/Native/Generation.lean`, which proves that what is sent only grows as the
//! text grows and never reaches into a stop string that a later token completes. This fixture,
//! written by `lake exe kadmium_vectors`, ties `heldBack` to `native_generation::held_back_len`.

use kadmium_runtime::native_generation::held_back_len;

const VECTORS: &str = include_str!("fixtures/generation_lean_vectors.txt");

fn hex(field: &str) -> Vec<u8> {
    if field == "-" {
        return Vec::new();
    }
    (0..field.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&field[i..i + 2], 16).expect("hex byte"))
        .collect()
}

#[test]
fn held_back_len_matches_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!((f[0], f.len()), ("held", 5), "{line}");
        let patterns: Vec<Vec<u8>> = if f[1] == "-" {
            Vec::new()
        } else {
            f[1].split(',').map(hex).collect()
        };
        let text = hex(f[2]);
        let held: usize = f[3].parse().expect("held back");
        let emitted: usize = f[4].parse().expect("emitted");
        assert_eq!(held_back_len(&text, &patterns), held, "{line}");
        assert_eq!(text.len() - held, emitted, "{line}");
        checked += 1;
    }
    assert_eq!(checked, 59);
}
