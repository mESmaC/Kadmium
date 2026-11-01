//! The native decoder against Hugging Face transformers, and its split-execution contract.
//!
//! Fixtures come from `scripts/native_decoder_fixtures.py`: tiny randomly initialised Llama 3
//! (BF16, `llama3` RoPE scaling), Qwen2 (F32, q/k/v bias, tied embeddings), Qwen3 (BF16, q/k
//! head norms, head_dim not hidden/heads, one KV head) and Mistral (F16) checkpoints, saved by
//! transformers, with the float32 logits transformers computes from the stored weights.

use kadmium_runtime::native_decoder::{
    Activations, DecoderConfig, DecoderSlice, KvCache, NativeDecoderError, SlicePlan,
};
use serde_json::Value;
use std::path::{Path, PathBuf};

const FIXTURES: [&str; 4] = ["llama3_bf16", "qwen2_f32", "qwen3_bf16", "mistral_f16"];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/native_decoder")
        .join(name)
}

struct Reference {
    tokens: Vec<u32>,
    vocab: usize,
    logits: Vec<f32>,
}

fn reference(name: &str) -> Reference {
    let text = std::fs::read_to_string(fixture(name).join("reference.json")).unwrap();
    let json: Value = serde_json::from_str(&text).unwrap();
    Reference {
        tokens: json["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .collect(),
        vocab: json["vocab_size"].as_u64().unwrap() as usize,
        logits: json["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    }
}

fn whole(name: &str) -> DecoderSlice {
    let dir = fixture(name);
    let config = DecoderConfig::from_model_dir(&dir).unwrap();
    DecoderSlice::load(&dir, SlicePlan::whole(&config)).unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Prefill of the whole sequence through one slice.
fn prefill_logits(slice: &DecoderSlice, tokens: &[u32]) -> Vec<f32> {
    let mut cache = slice.new_cache();
    let hidden = slice.embed(tokens, 0).unwrap();
    let out = slice.forward(&hidden, &mut cache).unwrap();
    slice.logits(&out).unwrap()
}

#[test]
fn matches_transformers_logits() {
    for name in FIXTURES {
        let reference = reference(name);
        let logits = prefill_logits(&whole(name), &reference.tokens);
        assert_eq!(logits.len(), reference.tokens.len() * reference.vocab);
        let mut worst = 0.0f32;
        for (ours, theirs) in logits.iter().zip(&reference.logits) {
            let error = (ours - theirs).abs() / theirs.abs().max(1.0);
            worst = worst.max(error);
        }
        eprintln!("{name}: worst relative logit error {worst:e}");
        // Both sides compute in f32 from the same stored weights; they differ only in summation
        // order and transcendental rounding.
        assert!(worst < 2e-5, "{name}: worst relative logit error {worst}");
        // The argmax, which greedy decoding emits, agrees at every position.
        for row in 0..reference.tokens.len() {
            let span = row * reference.vocab..(row + 1) * reference.vocab;
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .fold(
                        (0, f32::NEG_INFINITY),
                        |b, (i, &x)| if x > b.1 { (i, x) } else { b },
                    )
                    .0
            };
            assert_eq!(
                argmax(&logits[span.clone()]),
                argmax(&reference.logits[span]),
                "{name} row {row}"
            );
        }
    }
}

#[test]
fn decoding_one_token_at_a_time_equals_prefill_bit_for_bit() {
    for name in FIXTURES {
        let reference = reference(name);
        let slice = whole(name);
        let expected = prefill_logits(&slice, &reference.tokens);
        let mut cache = slice.new_cache();
        let split = 5;
        let first = slice.embed(&reference.tokens[..split], 0).unwrap();
        let mut logits = slice
            .logits(&slice.forward(&first, &mut cache).unwrap())
            .unwrap();
        for (offset, &token) in reference.tokens[split..].iter().enumerate() {
            let step = slice.embed(&[token], split + offset).unwrap();
            logits.extend(
                slice
                    .logits(&slice.forward(&step, &mut cache).unwrap())
                    .unwrap(),
            );
        }
        assert_eq!(bits(&logits), bits(&expected), "{name}");
        assert_eq!(cache.positions(), reference.tokens.len());
    }
}

/// Runs a chain of slices, each with its own cache, prefill then token-by-token decode, and
/// returns every logit row.
fn run_chain(slices: &[DecoderSlice], tokens: &[u32], prefill: usize) -> Vec<f32> {
    let mut caches: Vec<KvCache> = slices.iter().map(DecoderSlice::new_cache).collect();
    let mut step = |chunk: &[u32], start: usize| -> Vec<f32> {
        let mut hidden = slices[0].embed(chunk, start).unwrap();
        for (slice, cache) in slices.iter().zip(caches.iter_mut()) {
            // What crosses a machine boundary is the activations' bytes; rebuild from them.
            let carried = Activations::new(
                hidden.start_position(),
                hidden.hidden_size(),
                hidden.data().to_vec(),
            )
            .unwrap();
            hidden = slice.forward(&carried, cache).unwrap();
        }
        slices.last().unwrap().logits(&hidden).unwrap()
    };
    let mut logits = step(&tokens[..prefill], 0);
    for (i, &token) in tokens[prefill..].iter().enumerate() {
        logits.extend(step(&[token], prefill + i));
    }
    logits
}

fn slice(name: &str, layers: std::ops::Range<usize>, embedding: bool, head: bool) -> DecoderSlice {
    DecoderSlice::load(
        &fixture(name),
        SlicePlan {
            layers,
            embedding,
            head,
        },
    )
    .unwrap()
}

#[test]
fn splitting_the_layers_at_every_cut_is_bit_identical() {
    for name in FIXTURES {
        let reference = reference(name);
        let n = whole(name).config().num_layers;
        let unsplit = run_chain(&[whole(name)], &reference.tokens, 4);
        for a in 0..=n {
            for b in a..=n {
                let chain = [
                    slice(name, 0..a, true, false),
                    slice(name, a..b, false, false),
                    slice(name, b..n, false, true),
                ];
                assert_eq!(
                    bits(&run_chain(&chain, &reference.tokens, 4)),
                    bits(&unsplit),
                    "{name}: cuts at {a} and {b}"
                );
            }
        }
    }
}

#[test]
fn a_slice_holds_only_its_part_of_the_weights() {
    let name = "llama3_bf16";
    let total = whole(name).stored_weight_bytes();
    let n = whole(name).config().num_layers;
    let parts: usize = [
        slice(name, 0..1, true, false),
        slice(name, 1..n, false, true),
    ]
    .iter()
    .map(DecoderSlice::stored_weight_bytes)
    .sum();
    assert_eq!(parts, total);
    // BF16 matrices stay two bytes per weight.
    let safetensors = std::fs::metadata(fixture(name).join("model.safetensors"))
        .unwrap()
        .len() as usize;
    assert!(total < safetensors * 2, "weights were widened at load");
}

#[test]
fn a_request_past_the_native_context_names_both_numbers() {
    let slice = whole("qwen2_f32");
    let maximum = slice.config().max_position_embeddings;
    let tokens = vec![1u32; maximum + 1];
    let hidden = slice.embed(&tokens, 0).unwrap();
    let mut cache = slice.new_cache();
    let error = slice.forward(&hidden, &mut cache).unwrap_err();
    assert!(matches!(
        error,
        NativeDecoderError::ContextExceeded { requested, maximum: m } if requested == maximum + 1 && m == maximum
    ));
    assert_eq!(
        cache.positions(),
        0,
        "a refused request leaves the cache untouched"
    );
    // Exactly the native context is served.
    let hidden = slice.embed(&tokens[..maximum], 0).unwrap();
    slice.forward(&hidden, &mut cache).unwrap();
    assert_eq!(cache.positions(), maximum);
}

#[test]
fn misuse_is_an_error_not_a_guess() {
    let name = "mistral_f16";
    let full = whole(name);
    let mut cache = full.new_cache();
    let hidden = full.embed(&[1, 2, 3], 0).unwrap();
    full.forward(&hidden, &mut cache).unwrap();
    // Out-of-order activations.
    let stale = full.embed(&[4], 1).unwrap();
    assert!(matches!(
        full.forward(&stale, &mut cache),
        Err(NativeDecoderError::CachePosition {
            expected: 3,
            actual: 1
        })
    ));
    // A token outside the vocabulary.
    assert!(matches!(
        full.embed(&[96], 0),
        Err(NativeDecoderError::TokenOutOfRange { token: 96, .. })
    ));
    // A cache from another slice.
    let middle = slice(name, 1..2, false, false);
    let mut wrong = full.new_cache();
    let input = Activations::new(0, 32, vec![0.5; 32]).unwrap();
    assert!(matches!(
        middle.forward(&input, &mut wrong),
        Err(NativeDecoderError::Shape(_))
    ));
    // Parts the slice does not hold.
    assert!(matches!(
        middle.embed(&[1], 0),
        Err(NativeDecoderError::MissingPart(_))
    ));
    assert!(matches!(
        middle.logits(&input),
        Err(NativeDecoderError::MissingPart(_))
    ));
    // Layers beyond the model.
    let error = DecoderSlice::load(
        &fixture(name),
        SlicePlan {
            layers: 2..9,
            embedding: false,
            head: false,
        },
    )
    .err()
    .unwrap();
    assert!(matches!(
        error,
        NativeDecoderError::LayerRange {
            start: 2,
            end: 9,
            layers: 3
        }
    ));
}

/// Writes a single-tensor-file checkpoint with the fixture's config and one tensor replaced.
fn checkpoint_with(dir: &Path, replaced: &str, dtype: &str, bytes_per_element: usize) {
    let source = fixture("mistral_f16");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::copy(source.join("config.json"), dir.join("config.json")).unwrap();
    let original = std::fs::read(source.join("model.safetensors")).unwrap();
    let header_len = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&original[8..8 + header_len]).unwrap();
    let data = &original[8 + header_len..];
    let mut entries: Vec<(String, Value)> = header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| *k != "__metadata__")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    entries.sort_by_key(|(_, v)| v["data_offsets"][0].as_u64().unwrap());
    let mut out_header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, entry) in entries {
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let mut entry = entry.clone();
        let bytes = if name == replaced {
            let elements: u64 = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap())
                .product();
            entry["dtype"] = Value::from(dtype);
            vec![0x11u8; elements as usize * bytes_per_element]
        } else {
            data[start..end].to_vec()
        };
        entry["data_offsets"] = serde_json::json!([payload.len(), payload.len() + bytes.len()]);
        payload.extend(bytes);
        out_header.insert(name, entry);
    }
    let header_bytes = serde_json::to_vec(&Value::Object(out_header)).unwrap();
    let mut file = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file.extend(header_bytes);
    file.extend(payload);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

#[test]
fn quantized_tensors_are_refused_not_dequantized() {
    let root = std::env::temp_dir().join(format!("kadmium-native-decoder-{}", std::process::id()));
    for (dtype, width) in [("F8_E4M3", 1), ("I8", 1), ("F64", 8)] {
        let dir = root.join(dtype);
        checkpoint_with(&dir, "model.layers.1.mlp.up_proj.weight", dtype, width);
        let config = DecoderConfig::from_model_dir(&dir).unwrap();
        let error = DecoderSlice::load(&dir, SlicePlan::whole(&config))
            .err()
            .unwrap();
        assert!(
            matches!(&error, NativeDecoderError::UnsupportedDtype { tensor, .. } if tensor == "model.layers.1.mlp.up_proj.weight"),
            "{dtype}: {error}"
        );
        // A slice that does not hold the layer loads, because it never reads it.
        DecoderSlice::load(
            &dir,
            SlicePlan {
                layers: 0..1,
                embedding: true,
                head: false,
            },
        )
        .unwrap();
    }
    let _ = std::fs::remove_dir_all(root);
}

/// A real checkpoint against transformers, split across two slices. Run with
/// `KADMIUM_NATIVE_DECODER_MODEL=<model dir> KADMIUM_NATIVE_DECODER_REFERENCE=<json>
/// cargo test --release -p kadmium-runtime --test native_decoder -- --ignored`, the reference
/// written by `scripts/native_decoder_fixtures.py --reference`.
#[test]
#[ignore = "needs a downloaded checkpoint and its transformers reference"]
fn real_checkpoint_matches_transformers() {
    let (Ok(model), Ok(reference)) = (
        std::env::var("KADMIUM_NATIVE_DECODER_MODEL"),
        std::env::var("KADMIUM_NATIVE_DECODER_REFERENCE"),
    ) else {
        panic!("set KADMIUM_NATIVE_DECODER_MODEL and KADMIUM_NATIVE_DECODER_REFERENCE");
    };
    let model = PathBuf::from(model);
    let json: Value = serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
    let numbers = |key: &str| -> Vec<f64> {
        json[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let tokens: Vec<u32> = numbers("tokens").iter().map(|&t| t as u32).collect();
    let argmax: Vec<usize> = numbers("argmax").iter().map(|&t| t as usize).collect();
    let last: Vec<f32> = numbers("last_logits").iter().map(|&v| v as f32).collect();

    let config = DecoderConfig::from_model_dir(&model).unwrap();
    let n = config.num_layers;
    let started = std::time::Instant::now();
    let unsplit = DecoderSlice::load(&model, SlicePlan::whole(&config)).unwrap();
    eprintln!(
        "loaded {} weight bytes in {:?}",
        unsplit.stored_weight_bytes(),
        started.elapsed()
    );
    let started = std::time::Instant::now();
    let logits = prefill_logits(&unsplit, &tokens);
    eprintln!(
        "prefill of {} tokens in {:?}",
        tokens.len(),
        started.elapsed()
    );
    let vocab = config.vocab_size;
    let mut disagreements = 0;
    for (row, &expected) in argmax.iter().enumerate() {
        let span = &logits[row * vocab..(row + 1) * vocab];
        let ours = span
            .iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |b, (i, &x)| if x > b.1 { (i, x) } else { b },
            )
            .0;
        if ours != expected {
            disagreements += 1;
            eprintln!("row {row}: argmax {ours}, transformers {expected}");
        }
    }
    let ours_last = &logits[(tokens.len() - 1) * vocab..];
    let worst = ours_last
        .iter()
        .zip(&last)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f32, f32::max);
    eprintln!(
        "last-row worst relative logit error {worst:e}; argmax disagreements {disagreements}"
    );
    assert!(worst < 1e-3, "worst relative logit error {worst}");
    assert_eq!(disagreements, 0);
    drop(unsplit);

    let cut = n / 2;
    let chain = [
        DecoderSlice::load(
            &model,
            SlicePlan {
                layers: 0..cut,
                embedding: true,
                head: false,
            },
        )
        .unwrap(),
        DecoderSlice::load(
            &model,
            SlicePlan {
                layers: cut..n,
                embedding: false,
                head: true,
            },
        )
        .unwrap(),
    ];
    let split = run_chain(&chain, &tokens, tokens.len());
    assert_eq!(bits(&split), bits(&logits), "split at layer {cut}");
}

/// Copies a fixture with some `config.json` fields replaced.
fn fixture_with_config(name: &str, dir: &Path, patch: Value) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let source = fixture(name);
    std::fs::copy(
        source.join("model.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    let mut config: Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
            .unwrap();
    for (key, value) in patch.as_object().unwrap() {
        config[key] = value.clone();
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    dir.to_path_buf()
}

#[test]
fn config_values_never_size_an_allocation_the_checkpoint_does_not_back() {
    let root = std::env::temp_dir().join(format!(
        "kadmium-native-decoder-config-{}",
        std::process::id()
    ));
    // A layer count far beyond the checkpoint: an error at the first missing layer, not a
    // capacity overflow or an abort.
    let dir = fixture_with_config(
        "mistral_f16",
        &root.join("layers"),
        serde_json::json!({"num_hidden_layers": 1_152_921_504_606_846_976u64}),
    );
    let config = DecoderConfig::from_model_dir(&dir).unwrap();
    let error = DecoderSlice::load(&dir, SlicePlan::whole(&config))
        .err()
        .unwrap();
    assert!(
        matches!(&error, NativeDecoderError::MissingTensor(name) if name.starts_with("model.layers.3.")),
        "{error}"
    );

    // Fewer layers than the checkpoint stores would silently skip one.
    let dir = fixture_with_config(
        "mistral_f16",
        &root.join("fewer"),
        serde_json::json!({"num_hidden_layers": 2}),
    );
    let config = DecoderConfig::from_model_dir(&dir).unwrap();
    let error = DecoderSlice::load(&dir, SlicePlan::whole(&config))
        .err()
        .unwrap();
    assert!(
        matches!(&error, NativeDecoderError::Config(m) if m.contains("layer 2")),
        "{error}"
    );

    // A head dimension the projections do not have is refused before anything is sized by it,
    // even for a slice that holds no layers.
    let dir = fixture_with_config(
        "mistral_f16",
        &root.join("head_dim"),
        serde_json::json!({"num_attention_heads": 1, "num_key_value_heads": 1,
                           "head_dim": 1_099_511_627_776u64}),
    );
    let config = DecoderConfig::from_model_dir(&dir).unwrap();
    let error = DecoderSlice::load(
        &dir,
        SlicePlan {
            layers: 0..0,
            embedding: true,
            head: true,
        },
    )
    .err()
    .unwrap();
    assert!(
        matches!(&error, NativeDecoderError::TensorShape { tensor, .. } if tensor.ends_with("q_proj.weight")),
        "{error}"
    );
    assert_eq!(config.head_dim, 1_099_511_627_776);
    let _ = std::fs::remove_dir_all(root);
}
