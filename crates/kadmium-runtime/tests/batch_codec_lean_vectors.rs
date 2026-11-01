//! Differential check against the Lean model of the MoE batch codecs in
//! `formal/KadmiumFormal/Runtime/BatchCodec.lean`. The model proves that a header is accepted
//! exactly when the body has the length the header implies, in checked arithmetic, and exhibits
//! the two headers that the wrapping `usize` arithmetic of a release build would have accepted
//! with no body at all. This fixture, written by `lake exe kadmium_vectors`, ties the model to
//! `TokenBatchF32::from_le_bytes` and `ExpertPartialsBatch::from_le_bytes`.

use kadmium_runtime::moe_layer::{
    ExpertPartialsBatch, F32_BATCH_MAGIC, PARTIALS_MAGIC, TokenBatchF32,
};

const VECTORS: &str = include_str!("fixtures/batch_codec_lean_vectors.txt");

fn lines() -> impl Iterator<Item = Vec<&'static str>> {
    VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect())
}

#[test]
fn f32_headers_are_accepted_exactly_when_the_model_says() {
    let mut checked = 0;
    for f in lines().filter(|f| f[0] == "f32") {
        let rows: u32 = f[1].parse().expect("rows");
        let dim: u32 = f[2].parse().expect("dim");
        let body: usize = f[3].parse().expect("body bytes");
        let mut bytes = Vec::with_capacity(12 + body);
        bytes.extend_from_slice(&F32_BATCH_MAGIC);
        bytes.extend_from_slice(&rows.to_le_bytes());
        bytes.extend_from_slice(&dim.to_le_bytes());
        bytes.resize(12 + body, 0);
        let decoded = TokenBatchF32::from_le_bytes(&bytes);
        match f[4] {
            "accept" => {
                let batch = decoded.unwrap_or_else(|e| {
                    panic!("model accepts {rows} x {dim} with {body} bytes: {e}")
                });
                assert_eq!(batch.rows, rows);
                assert_eq!(batch.dim, dim);
                assert_eq!(batch.data.len() * 4, body);
            }
            "reject" => {
                assert!(
                    decoded.is_err(),
                    "model rejects {rows} x {dim} with {body} bytes, Rust accepted it"
                );
            }
            other => panic!("unknown verdict {other}"),
        }
        checked += 1;
    }
    assert!(checked >= 8, "regenerate with `lake exe kadmium_vectors`");
}

#[test]
fn partials_headers_are_accepted_exactly_when_the_model_says() {
    let mut checked = 0;
    for f in lines().filter(|f| f[0] == "part") {
        let count: u32 = f[1].parse().expect("count");
        let dim: u32 = f[2].parse().expect("dim");
        let total: usize = f[3].parse().expect("total bytes");
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(&PARTIALS_MAGIC);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&dim.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.resize(total, 0);
        let decoded = ExpertPartialsBatch::from_le_bytes(&bytes);
        match f[4] {
            "accept" => {
                let batch = decoded.unwrap_or_else(|e| {
                    panic!("model accepts {count} partials of width {dim} in {total} bytes: {e}")
                });
                assert_eq!(batch.dim, dim);
                assert_eq!(batch.partials.len(), count as usize);
                assert!(batch.partials.iter().all(|p| p.data.len() == dim as usize));
            }
            "reject" => {
                assert!(
                    decoded.is_err(),
                    "model rejects {count} partials of width {dim} in {total} bytes, Rust accepted it"
                );
            }
            other => panic!("unknown verdict {other}"),
        }
        checked += 1;
    }
    assert!(checked >= 8, "regenerate with `lake exe kadmium_vectors`");
}
