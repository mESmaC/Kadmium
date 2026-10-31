//! Differential check against the Lean model of the RepOps reduction in `formal/`.
//!
//! `formal/KadmiumFormal/RepOps/TreeReducer.lean` proves that `TreeReducer` returns `tree_sum`
//! for every push order and every addition, and `TreeSum.lean` proves that
//! `VerdeVerificationEngine::repops_pairwise_tree_sum` builds the same tree. Those theorems are
//! about a Lean model of this code. This test ties the model to the code: `lake exe
//! repops_vectors` evaluates the model and writes `tests/fixtures/repops_tree_vectors.txt`, and
//! every Rust implementation must reproduce each value bit for bit.
//!
//! Two additions are checked. `Mix` is neither associative nor commutative, so it matches only
//! when the tree shape and the operand order are identical. `f64` checks IEEE addition with
//! operands whose rounding depends on the order.

use kadmium_runtime::VerdeVerificationEngine;
use kadmium_runtime::repops::{Summand, TreeReducer, tree_sum};

const VECTORS: &str = include_str!("fixtures/repops_tree_vectors.txt");

/// SplitMix64 of `i + 1`, as `splitmix` in `formal/Vectors.lean`.
fn splitmix(i: u64) -> u64 {
    let mut z = i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `mix` in `formal/Vectors.lean`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mix(u64);

impl Summand for Mix {
    const ZERO: Self = Mix(0);
    fn add(self, rhs: Self) -> Self {
        let z = (self.0 ^ rhs.0.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_mul(0xBF58_476D_1CE4_E5B9)
            .wrapping_add(rhs.0);
        Mix(z ^ (z >> 29))
    }
}

/// `leafF64` in `formal/Vectors.lean`: finite, normal values from `2^-30` to `2^34`.
fn leaf_f64(i: usize) -> f64 {
    let h = splitmix(i as u64 + 0x1000);
    let sign = h & 0x8000_0000_0000_0000;
    let exponent = 993 + ((h >> 52) & 0x3F);
    let mantissa = h & 0x000F_FFFF_FFFF_FFFF;
    f64::from_bits(sign | (exponent << 52) | mantissa)
}

struct Vector {
    n: usize,
    mix: u64,
    f64_bits: u64,
}

fn vectors() -> Vec<Vector> {
    VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 3, "malformed fixture line: {line}");
            Vector {
                n: fields[0].parse().expect("leaf count"),
                mix: u64::from_str_radix(fields[1], 16).expect("mix value"),
                f64_bits: u64::from_str_radix(fields[2], 16).expect("f64 bits"),
            }
        })
        .collect()
}

/// Arrival orders a backend might produce: in order, backwards, odd lanes before even lanes,
/// chunks of seven finishing last first, and a seeded shuffle.
fn orders(n: usize) -> Vec<Vec<usize>> {
    let forward: Vec<usize> = (0..n).collect();
    let backward: Vec<usize> = (0..n).rev().collect();
    let odd_then_even: Vec<usize> = (0..n)
        .filter(|i| i % 2 == 1)
        .chain((0..n).filter(|i| i % 2 == 0))
        .collect();
    let chunks_backwards: Vec<usize> = forward
        .chunks(7)
        .rev()
        .flat_map(|chunk| chunk.iter().copied())
        .collect();
    let mut shuffled = forward.clone();
    for i in (1..n).rev() {
        let j = (splitmix((n * 31 + i) as u64) % (i as u64 + 1)) as usize;
        shuffled.swap(i, j);
    }
    vec![forward, backward, odd_then_even, chunks_backwards, shuffled]
}

fn reduce_in_order<T: Summand>(values: &[T], order: &[usize]) -> T {
    let mut reducer = TreeReducer::new(values.len());
    for &i in order {
        reducer
            .push(i, values[i])
            .expect("each leaf is pushed once");
    }
    reducer.finish().expect("every leaf arrived")
}

/// Pushes contiguous chunks of `chunk` leaves with `push_chunk`, last chunk first.
fn reduce_in_chunks<T: Summand>(values: &[T], chunk: usize) -> T {
    let mut reducer = TreeReducer::new(values.len());
    let starts: Vec<usize> = (0..values.len()).step_by(chunk).collect();
    for &start in starts.iter().rev() {
        let end = (start + chunk).min(values.len());
        reducer
            .push_chunk(start, &values[start..end])
            .expect("chunks are disjoint");
    }
    reducer.finish().expect("every leaf arrived")
}

#[test]
fn fixture_covers_the_generated_sizes() {
    let vectors = vectors();
    assert_eq!(
        vectors.len(),
        142,
        "regenerate with `lake exe repops_vectors`"
    );
    assert!(vectors.iter().any(|v| v.n == 0));
    assert!(vectors.iter().any(|v| v.n == 4097));
}

#[test]
fn tree_sum_matches_the_lean_model() {
    for v in vectors() {
        let mix: Vec<Mix> = (0..v.n).map(|i| Mix(splitmix(i as u64))).collect();
        let floats: Vec<f64> = (0..v.n).map(leaf_f64).collect();
        assert_eq!(
            tree_sum(&mix).0,
            v.mix,
            "tree_sum shape differs at n = {}",
            v.n
        );
        assert_eq!(
            tree_sum(&floats).to_bits(),
            v.f64_bits,
            "tree_sum f64 differs at n = {}",
            v.n
        );
    }
}

#[test]
fn tree_reducer_matches_the_lean_model_in_every_order() {
    for v in vectors() {
        let mix: Vec<Mix> = (0..v.n).map(|i| Mix(splitmix(i as u64))).collect();
        let floats: Vec<f64> = (0..v.n).map(leaf_f64).collect();
        for order in orders(v.n) {
            assert_eq!(reduce_in_order(&mix, &order).0, v.mix, "n = {}", v.n);
            assert_eq!(
                reduce_in_order(&floats, &order).to_bits(),
                v.f64_bits,
                "n = {}",
                v.n
            );
        }
        for chunk in [1, 3, 32] {
            assert_eq!(reduce_in_chunks(&mix, chunk).0, v.mix, "n = {}", v.n);
            assert_eq!(
                reduce_in_chunks(&floats, chunk).to_bits(),
                v.f64_bits,
                "n = {}",
                v.n
            );
        }
    }
}

#[test]
fn legacy_verde_tree_sum_matches_the_lean_model() {
    for v in vectors() {
        let floats: Vec<f64> = (0..v.n).map(leaf_f64).collect();
        assert_eq!(
            VerdeVerificationEngine::repops_pairwise_tree_sum(&floats).to_bits(),
            v.f64_bits,
            "repops_pairwise_tree_sum differs at n = {}",
            v.n
        );
    }
}
