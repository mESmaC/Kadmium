//! Differential check against the Lean model of the MoE batch codecs in
//! `formal/KadmiumFormal/Runtime/BatchCodec.lean`. The model proves that the encoders' output
//! decodes back to the input, that a header is accepted exactly when the body has the length the
//! header implies, in checked arithmetic, and exhibits the two headers that the wrapping `usize`
//! arithmetic of a release build would have accepted with no body at all. This fixture, written
//! by `lake exe kadmium_vectors`, ties the model to `TokenBatchF32::{to_le_bytes, from_le_bytes}`
//! and `ExpertPartialsBatch::{to_le_bytes, from_le_bytes}`: acceptance decisions, and encoded
//! batches that must decode to every field and element the model encoded and re-encode to the
//! same bytes.

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

/// The `f32` values whose little-endian bytes are `body`.
fn f32s(body: &[u8]) -> Vec<f32> {
    body.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn encoded_f32_batches_decode_to_the_model_and_re_encode_to_the_same_bytes() {
    let mut checked = 0;
    for f in lines().filter(|f| f[0] == "f32enc") {
        let rows: u32 = f[1].parse().expect("rows");
        let dim: u32 = f[2].parse().expect("dim");
        let seed: u64 = f[3].parse().expect("seed");
        let encoded = hex(f[4]);
        let body: Vec<u8> = (0..u64::from(rows) * u64::from(dim) * 4)
            .map(|j| byte_at(seed, j))
            .collect();
        let batch = TokenBatchF32::from_le_bytes(&encoded)
            .unwrap_or_else(|e| panic!("model encoding of {rows} x {dim} does not decode: {e}"));
        assert_eq!(batch.rows, rows);
        assert_eq!(batch.dim, dim);
        assert_eq!(
            bits(&batch.data),
            bits(&f32s(&body)),
            "elements differ for {rows} x {dim}"
        );
        assert_eq!(
            batch.to_le_bytes(),
            encoded,
            "re-encoding differs for {rows} x {dim}"
        );
        checked += 1;
    }
    assert!(checked >= 5, "regenerate with `lake exe kadmium_vectors`");
}

#[test]
fn encoded_partials_decode_to_the_model_and_re_encode_to_the_same_bytes() {
    let mut checked = 0;
    for f in lines().filter(|f| f[0] == "partenc") {
        let layer: u32 = f[1].parse().expect("layer");
        let rows: u32 = f[2].parse().expect("rows");
        let dim: u32 = f[3].parse().expect("dim");
        let seed: u64 = f[4].parse().expect("seed");
        let entries: Vec<(u32, u32)> = if f[5] == "-" {
            Vec::new()
        } else {
            f[5].split(',')
                .map(|e| {
                    let (row, expert) = e.split_once(':').expect("row:expert");
                    (row.parse().expect("row"), expert.parse().expect("expert"))
                })
                .collect()
        };
        let encoded = hex(f[6]);
        let batch = ExpertPartialsBatch::from_le_bytes(&encoded)
            .unwrap_or_else(|e| panic!("model encoding of layer {layer} does not decode: {e}"));
        assert_eq!(batch.layer, layer);
        assert_eq!(batch.rows, rows);
        assert_eq!(batch.dim, dim);
        assert_eq!(batch.partials.len(), entries.len());
        for (i, (partial, (row, expert))) in batch.partials.iter().zip(&entries).enumerate() {
            assert_eq!(partial.row, *row, "row of partial {i}");
            assert_eq!(partial.expert, *expert, "expert of partial {i}");
            let data: Vec<u8> = (0..u64::from(dim) * 4)
                .map(|j| byte_at(seed + i as u64, j))
                .collect();
            assert_eq!(
                bits(&partial.data),
                bits(&f32s(&data)),
                "data of partial {i}"
            );
        }
        assert_eq!(
            batch.to_le_bytes(),
            encoded,
            "re-encoding differs for layer {layer}"
        );
        checked += 1;
    }
    assert!(checked >= 4, "regenerate with `lake exe kadmium_vectors`");
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
