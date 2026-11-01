//! The native decoder against Hugging Face transformers, and its split-execution contract.
//!
//! Fixtures come from `scripts/native_decoder_fixtures.py`: tiny randomly initialised Llama 3
//! (BF16, `llama3` RoPE scaling), Qwen2 (F32, q/k/v bias, tied embeddings), Qwen3 (BF16, q/k
//! head norms, head_dim not hidden/heads, one KV head) and Mistral (F16, full and sliding-window
//! attention) checkpoints, saved by
//! transformers, with the float32 logits transformers computes from the stored weights.

use kadmium_core::{
    ComputeBackend, DeviceKind, HardwareNode, MemoryTierProfile, NodeId, NodeRole, NodeTrustTier,
    PoliteState, ResourceOffer, sample_personal_cluster,
};
use kadmium_data::SafetensorsManifest;
use kadmium_network::WireFrameKind;
use kadmium_runtime::native_decoder::{
    Activations, DecoderConfig, DecoderSlice, ForwardInput, ForwardOutput, KvCache,
    LayerPlacementError, MachineMemoryBudget, NATIVE_SESSION_HEADER_BYTES, NativeDecoderError,
    NativeSessionHost, NativeSessionLimits, SessionAuthKey, SessionForwardRequest,
    SessionForwardResponse, SessionOpenRequest, SessionRequest, SessionResponse, SlicePlan,
    checkpoint_memory_profile, decode_f32_tensor, encode_f32_tensor, fnv1a64_f32_le,
    plan_checkpoint_placement, plan_checkpoint_placement_with_budgets,
};
use kadmium_runtime::{DistributedLanExecutor, LiveNodeWorkerServer};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

const FIXTURES: [&str; 5] = [
    "llama3_bf16",
    "qwen2_f32",
    "qwen3_bf16",
    "mistral_f16",
    "mistral_sliding_f16",
];

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
fn sliding_attention_crosses_the_window_at_every_prefill_boundary() {
    let name = "mistral_sliding_f16";
    let tokens = reference(name).tokens;
    let model = whole(name);
    let expected = bits(&prefill_logits(&model, &tokens));
    let chain = [
        slice(name, 0..1, true, false),
        slice(name, 1..3, false, true),
    ];
    for prefill in 1..=tokens.len() {
        assert_eq!(
            bits(&run_chain(&chain, &tokens, prefill)),
            expected,
            "prefill {prefill}"
        );
    }
}

#[test]
fn sliding_attention_masks_only_keys_outside_the_architectural_window() {
    let dir = fixture("mistral_sliding_f16");
    let manifest = SafetensorsManifest::open(&dir).unwrap();
    let original = DecoderConfig::from_model_dir(&dir).unwrap();
    let tokens = reference("mistral_sliding_f16").tokens;
    let mut changed = tokens.clone();
    changed[0] = 2;
    // One layer isolates the mask from information carried forward through earlier layers.
    let plan = SlicePlan {
        layers: 0..1,
        embedding: true,
        head: true,
    };
    let mut full = original.clone();
    full.sliding_window = None;
    let full = DecoderSlice::load_from(&manifest, full, plan.clone()).unwrap();
    let full_logits = bits(&prefill_logits(&full, &tokens));
    for window in [
        1,
        4,
        original.max_position_embeddings,
        original.max_position_embeddings + 1,
    ] {
        let mut config = original.clone();
        config.sliding_window = Some(window);
        let model = DecoderSlice::load_from(&manifest, config, plan.clone()).unwrap();
        let logits = bits(&prefill_logits(&model, &tokens));
        let changed_logits = bits(&prefill_logits(&model, &changed));
        let prefix = window.min(tokens.len()) * original.vocab_size;
        assert_eq!(logits[..prefix], full_logits[..prefix], "window {window}");
        assert_ne!(
            logits[..original.vocab_size],
            changed_logits[..original.vocab_size]
        );
        if window < tokens.len() {
            assert_ne!(logits[prefix..], full_logits[prefix..]);
            assert_eq!(
                logits[prefix..],
                changed_logits[prefix..],
                "window {window}"
            );
        }
    }
}

#[test]
fn sliding_attention_keeps_the_full_context_and_cache() {
    let model = whole("mistral_sliding_f16");
    let c = model.config();
    let maximum = c.max_position_embeddings;
    assert_eq!(c.sliding_window, Some(4));
    let mut cache = model.new_cache();
    let tokens = vec![1; maximum];
    let hidden = model.embed(&tokens, 0).unwrap();
    model.forward(&hidden, &mut cache).unwrap();
    assert_eq!(cache.positions(), maximum);
    assert_eq!(cache.bytes(), c.num_layers * maximum * c.kv_width() * 2 * 4);
    let before = cache.bytes();
    let extra = model.embed(&[1], maximum).unwrap();
    assert!(
        matches!(model.forward(&extra, &mut cache), Err(NativeDecoderError::ContextExceeded {
        requested, maximum: limit
    }) if requested == maximum + 1 && limit == maximum)
    );
    assert_eq!(cache.positions(), maximum);
    assert_eq!(cache.bytes(), before);
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

#[test]
fn checkpoint_memory_profile_matches_loaded_weights_and_full_context_kv_bytes() {
    for name in FIXTURES {
        let dir = fixture(name);
        let config = DecoderConfig::from_model_dir(&dir).unwrap();
        let manifest = SafetensorsManifest::open(&dir).unwrap();
        let profile = checkpoint_memory_profile(&manifest, &config).unwrap();
        assert_eq!(profile.num_layers, config.num_layers);
        assert_eq!(
            profile.max_position_embeddings,
            config.max_position_embeddings
        );

        // Single-machine whole-model slice.
        let full = whole(name);
        assert_eq!(
            profile
                .slice_weight_bytes(0..config.num_layers, true, true)
                .unwrap(),
            full.stored_weight_bytes() as u64,
            "{name}: whole-model weight bytes"
        );
        let mut full_cache = full.new_cache();
        let all_tokens = vec![1u32; config.max_position_embeddings];
        let hidden = full.embed(&all_tokens, 0).unwrap();
        full.forward(&hidden, &mut full_cache).unwrap();
        assert_eq!(full_cache.positions(), config.max_position_embeddings);
        assert_eq!(
            profile.full_context_kv_bytes().unwrap(),
            full_cache.bytes() as u64,
            "{name}: full-context KV cache bytes"
        );

        // Every 3-stage split across 0..=num_layers.
        let n = config.num_layers;
        for a in 1..n {
            for b in a..n {
                let plans = [
                    SlicePlan {
                        layers: 0..a,
                        embedding: true,
                        head: false,
                    },
                    SlicePlan {
                        layers: a..b,
                        embedding: false,
                        head: false,
                    },
                    SlicePlan {
                        layers: b..n,
                        embedding: false,
                        head: true,
                    },
                ];
                for plan in plans {
                    let loaded = DecoderSlice::load(&dir, plan.clone()).unwrap();
                    let expected_weights = profile
                        .slice_weight_bytes(plan.layers.clone(), plan.embedding, plan.head)
                        .unwrap();
                    assert_eq!(
                        expected_weights,
                        loaded.stored_weight_bytes() as u64,
                        "{name}: slice {plan:?} weight bytes"
                    );
                }
            }
        }
    }
}

#[test]
fn planned_multi_machine_slices_respect_memory_budgets_and_decode_bit_identically() {
    for name in FIXTURES {
        let dir = fixture(name);
        let reference = reference(name);
        let config = DecoderConfig::from_model_dir(&dir).unwrap();
        let manifest = SafetensorsManifest::open(&dir).unwrap();
        let profile = checkpoint_memory_profile(&manifest, &config).unwrap();

        // Give two TrustedCore machines and one middle-only machine just enough memory for a
        // 3-way split (0..1 on node-a, 1..2 on node-mid, 2..3 on node-b) so no single machine
        // can hold the whole checkpoint.
        let s0 = profile.slice_total_bytes(0..1, true, false).unwrap();
        let s1 = profile.slice_total_bytes(1..2, false, false).unwrap();
        let s2 = profile
            .slice_total_bytes(2..config.num_layers, false, true)
            .unwrap();
        let whole_bytes = profile.whole_model_bytes().unwrap();
        assert!(s0 < whole_bytes && s1 < whole_bytes && s2 < whole_bytes);

        let budgets = [
            MachineMemoryBudget::new(NodeId("node-a".into()), s0, NodeTrustTier::TrustedCore),
            MachineMemoryBudget::new(
                NodeId("node-mid".into()),
                s1,
                NodeTrustTier::VerifiedPeerMiddleLayersOnly,
            ),
            MachineMemoryBudget::new(NodeId("node-b".into()), s2, NodeTrustTier::TrustedCore),
        ];
        let placement = plan_checkpoint_placement_with_budgets(&dir, &budgets).unwrap();
        assert!(placement.is_split());
        assert_eq!(placement.context_length(), config.max_position_embeddings);
        assert_eq!(placement.num_layers(), config.num_layers);
        assert_eq!(placement.stages.len(), 3);
        assert_eq!(placement.stages[0].node_id, NodeId("node-a".into()));
        assert_eq!(placement.stages[0].plan.layers, 0..1);
        assert!(placement.stages[0].plan.embedding && !placement.stages[0].plan.head);
        assert_eq!(placement.stages[1].node_id, NodeId("node-mid".into()));
        assert_eq!(placement.stages[1].plan.layers, 1..2);
        assert!(!placement.stages[1].plan.embedding && !placement.stages[1].plan.head);
        assert_eq!(placement.stages[2].node_id, NodeId("node-b".into()));
        assert_eq!(placement.stages[2].plan.layers, 2..config.num_layers);
        assert!(!placement.stages[2].plan.embedding && placement.stages[2].plan.head);

        let mut slices = Vec::with_capacity(placement.stages.len());
        for stage in &placement.stages {
            assert!(stage.total_bytes <= stage.capacity_bytes);
            let loaded = DecoderSlice::load(&dir, stage.plan.clone()).unwrap();
            assert_eq!(loaded.stored_weight_bytes() as u64, stage.weight_bytes);
            slices.push(loaded);
        }

        let expected = run_chain(&[whole(name)], &reference.tokens, 4);
        let actual = run_chain(&slices, &reference.tokens, 4);
        assert_eq!(
            bits(&actual),
            bits(&expected),
            "{name}: planned 3-stage chain"
        );
    }
}

#[test]
fn placement_fails_when_machines_cannot_hold_model_and_full_context() {
    let dir = fixture("qwen2_f32");
    let config = DecoderConfig::from_model_dir(&dir).unwrap();
    let manifest = SafetensorsManifest::open(&dir).unwrap();
    let profile = checkpoint_memory_profile(&manifest, &config).unwrap();
    let whole_bytes = profile.whole_model_bytes().unwrap();
    let weights_only = profile
        .slice_weight_bytes(0..config.num_layers, true, true)
        .unwrap();
    assert!(weights_only < whole_bytes);

    // 1 byte short of whole_bytes on a single machine fails with InsufficientClusterMemory
    // rather than shrinking the context window.
    let one_short = [MachineMemoryBudget::new(
        NodeId("box-1".into()),
        whole_bytes - 1,
        NodeTrustTier::TrustedCore,
    )];
    let err = plan_checkpoint_placement_with_budgets(&dir, &one_short).unwrap_err();
    assert!(matches!(
        err,
        NativeDecoderError::Placement(LayerPlacementError::InsufficientClusterMemory {
            context_length,
            single_machine_bytes,
            total_available_bytes,
            ..
        }) if context_length == config.max_position_embeddings
            && single_machine_bytes == whole_bytes
            && total_available_bytes == whole_bytes - 1
    ));

    // Enough memory for weights alone (at a shorter context), but not for the full context.
    let weights_budget = [MachineMemoryBudget::new(
        NodeId("box-weights-only".into()),
        weights_only,
        NodeTrustTier::TrustedCore,
    )];
    let err = plan_checkpoint_placement_with_budgets(&dir, &weights_budget).unwrap_err();
    assert!(matches!(
        err,
        NativeDecoderError::Placement(LayerPlacementError::InsufficientClusterMemory {
            context_length,
            ..
        }) if context_length == config.max_position_embeddings
    ));

    // Untrusted/middle-only machines with plenty of memory still fail when no TrustedCore
    // boundary machine is present.
    let untrusted_only = [MachineMemoryBudget::new(
        NodeId("peer-1".into()),
        whole_bytes * 4,
        NodeTrustTier::VerifiedPeerMiddleLayersOnly,
    )];
    let err = plan_checkpoint_placement_with_budgets(&dir, &untrusted_only).unwrap_err();
    assert!(matches!(
        err,
        NativeDecoderError::Placement(LayerPlacementError::NoTrustedBoundaryMachine {
            untrusted_candidates: 1,
            last_layer: 2,
        })
    ));

    // Cluster placement from HardwareNode + ResourceOffer succeeds when resident memory suffices.
    let node = HardwareNode {
        id: NodeId("main-pc".into()),
        hostname: "main-pc".into(),
        device_kind: DeviceKind::DesktopWorkstation,
        backend: ComputeBackend::Cuda,
        roles: vec![NodeRole::ComputeWorker],
        trust_tier: NodeTrustTier::TrustedCore,
        gpu_name: "Test GPU".into(),
        cpu_name: "Test CPU".into(),
        npu_name: None,
        npu_tops: 0.0,
        fp16_tflops: 50.0,
        int4_tops: 100.0,
        cpu_threads: 8,
        cpu_tflops: 1.0,
        memory: MemoryTierProfile {
            vram_gb: 8.0,
            vram_bw_gbps: 500.0,
            unified_mem_gb: 0.0,
            unified_mem_bw_gbps: 0.0,
            system_ram_gb: 16.0,
            system_ram_bw_gbps: 60.0,
            interconnect_bw_gbps: 32.0,
            nvme_cache_gb: 100.0,
            nvme_read_bw_gbps: 7.0,
        },
        polite_state: PoliteState::IdleDedicated,
        network_uplink_gbps: 10.0,
        network_latency_ms: 0.2,
    };
    let offers: BTreeMap<NodeId, ResourceOffer> = BTreeMap::new();
    let cluster_plan = plan_checkpoint_placement(&dir, &[node], &offers).unwrap();
    assert_eq!(cluster_plan.stages.len(), 1);
    assert_eq!(cluster_plan.stages[0].plan, SlicePlan::whole(&config));
    assert_eq!(cluster_plan.total_bytes, whole_bytes);
}

fn test_auth_key(label: &[u8]) -> SessionAuthKey {
    SessionAuthKey::new(label).unwrap()
}

fn sample_worker_node(id: &str) -> HardwareNode {
    let mut node = sample_personal_cluster().into_iter().next().unwrap();
    node.id = NodeId(id.to_string());
    node.hostname = format!("{id}.lan");
    node
}

#[test]
fn session_wire_preserves_every_f32_bit_pattern() {
    let key = test_auth_key(b"kadmium-test-secret-key-0001");
    let limits = NativeSessionLimits::default();
    let special_bits: [u32; 12] = [
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        0x0000_0001, // smallest positive subnormal
        0x8000_0001, // smallest negative subnormal
        0x007f_ffff, // largest positive subnormal
        0x0080_0000, // smallest positive normal
        0x3f80_0000, // +1.0
        0xbf80_0000, // -1.0
        0x7f7f_ffff, // f32::MAX
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_1234, // quiet NaN with custom payload bits
    ];
    let values: Vec<f32> = special_bits.iter().copied().map(f32::from_bits).collect();
    let act = Activations::new(7, 4, values.clone()).unwrap();
    let req = SessionRequest::Forward(SessionForwardRequest {
        session_id: "sess-bit-exact".to_string(),
        input: ForwardInput::Activations(act),
        emit_logits: false,
    });
    let wire = req.encode(&key, &limits).unwrap();
    let decoded = SessionRequest::decode(&wire, &key, &limits).unwrap();
    let SessionRequest::Forward(decoded_req) = decoded else {
        panic!("expected Forward request");
    };
    let ForwardInput::Activations(decoded_act) = decoded_req.input else {
        panic!("expected Activations input");
    };
    assert_eq!(decoded_act.start_position(), 7);
    assert_eq!(decoded_act.rows(), 3);
    assert_eq!(decoded_act.hidden_size(), 4);
    assert_eq!(bits(decoded_act.data()), special_bits.to_vec());

    let resp = SessionResponse::Forward(SessionForwardResponse {
        session_id: "sess-bit-exact".to_string(),
        output: ForwardOutput::Logits {
            start_position: 7,
            rows: 3,
            vocab_size: 4,
            data: values,
        },
        cache_positions: 10,
        kv_cache_bytes: 128,
        repops_digest_fnv1a: fnv1a64_f32_le(&special_bits.map(f32::from_bits)),
    });
    let resp_wire = resp.encode(&key, &limits).unwrap();
    let decoded_resp = SessionResponse::decode(&resp_wire, &key, &limits).unwrap();
    let SessionResponse::Forward(decoded_fwd) = decoded_resp else {
        panic!("expected Forward response");
    };
    let ForwardOutput::Logits { data, .. } = decoded_fwd.output else {
        panic!("expected Logits output");
    };
    assert_eq!(bits(&data), special_bits.to_vec());
}

#[test]
fn session_authentication_rejects_missing_wrong_tampered_and_replayed_frames() {
    let limits = NativeSessionLimits::default();
    let key_a = test_auth_key(b"kadmium-session-key-alpha-1234");
    let key_b = test_auth_key(b"kadmium-session-key-bravo-5678");

    // Keys shorter than 16 bytes are refused at construction.
    assert!(matches!(
        SessionAuthKey::new(b"short-key"),
        Err(NativeDecoderError::Authentication(_))
    ));
    // Debug formatting redacts key material.
    assert!(
        !format!("{key_a:?}").contains("alpha"),
        "Debug leaked secret material"
    );

    let open = SessionRequest::Open(SessionOpenRequest {
        session_id: "sess-auth".to_string(),
        model_id: "llama3_bf16".to_string(),
        start_layer: 0,
        end_layer: 1,
        embedding: true,
        head: false,
    });
    let wire = open.encode(&key_a, &limits).unwrap();

    // Wrong key is rejected.
    assert!(matches!(
        SessionRequest::decode(&wire, &key_b, &limits),
        Err(NativeDecoderError::Authentication(_))
    ));

    // Replaying a signed request frame into SessionResponse::decode fails domain separation.
    assert!(matches!(
        SessionResponse::decode(&wire, &key_a, &limits),
        Err(NativeDecoderError::Authentication(_))
    ));

    // Flipping a bit in the metadata or payload or tag is rejected before parsing JSON.
    for idx in [5usize, 14, NATIVE_SESSION_HEADER_BYTES] {
        let mut tampered = wire.clone();
        tampered[idx] ^= 0x01;
        assert!(
            matches!(
                SessionRequest::decode(&tampered, &key_a, &limits),
                Err(NativeDecoderError::Authentication(_))
            ),
            "tampered byte at offset {idx} was not rejected as unauthenticated"
        );
    }

    // A worker without an auth key configured refuses the frame.
    let unconfigured = NativeSessionHost::new();
    unconfigured.clear_auth_key();
    let (reply_bytes, counted) = unconfigured.handle_frame_payload(&wire);
    assert!(!counted);
    assert!(matches!(
        SessionResponse::decode(&reply_bytes, &key_a, &limits),
        Err(NativeDecoderError::Authentication(_))
    ));
}

#[test]
fn session_wire_enforces_size_limits_and_rejects_dimension_overflow() {
    let key = test_auth_key(b"kadmium-bounds-check-secret-99");
    let tight = NativeSessionLimits {
        max_frame_bytes: 256,
        max_metadata_bytes: 180,
        max_payload_bytes: 64,
        max_active_sessions: 2,
    };

    // Payload larger than max_payload_bytes is rejected on encode and decode.
    let big_act = Activations::new(0, 8, vec![1.0f32; 32]).unwrap(); // 128 B > 64 B
    let big_req = SessionRequest::Forward(SessionForwardRequest {
        session_id: "sess-big".to_string(),
        input: ForwardInput::Activations(big_act),
        emit_logits: false,
    });
    assert!(matches!(
        big_req.encode(&key, &tight),
        Err(NativeDecoderError::FrameTooLarge {
            actual: 128,
            limit: 64
        })
    ));

    // A frame encoded with default limits is rejected by a receiver with tighter frame limits.
    let default_wire = big_req
        .encode(&key, &NativeSessionLimits::default())
        .unwrap();
    assert!(matches!(
        SessionRequest::decode(&default_wire, &key, &tight),
        Err(NativeDecoderError::FrameTooLarge { .. })
    ));

    // Overflowing dimensions (2^31 * 2^31 * 4 wraps 64-bit arithmetic to 0) are rejected.
    let half = 1usize << 31;
    assert!(matches!(
        encode_f32_tensor(half, half, &[], usize::MAX),
        Err(NativeDecoderError::Wire(_))
    ));
    assert!(matches!(
        decode_f32_tensor(half, half, &[], usize::MAX),
        Err(NativeDecoderError::Wire(_))
    ));

    // Truncated or trailing tensor bytes are rejected.
    assert!(matches!(
        decode_f32_tensor(2, 4, &[0u8; 28], 1024),
        Err(NativeDecoderError::Wire(_))
    ));
}

#[test]
fn cancelling_an_in_flight_forward_stops_between_layers_and_rolls_back_kv_cache() {
    let slice = whole("llama3_bf16");
    let mut cache = slice.new_cache();
    let prefill = slice.embed(&[1, 2], 0).unwrap();
    slice.forward(&prefill, &mut cache).unwrap();
    assert_eq!(cache.positions(), 2);

    // Pre-cancelled flag aborts before completing the slice and rolls KV cache back to 2 positions.
    let cancelled = AtomicBool::new(true);
    let step = slice.embed(&[3, 4], 2).unwrap();
    let err = slice
        .forward_until(&step, &mut cache, &cancelled)
        .unwrap_err();
    assert!(matches!(err, NativeDecoderError::Cancelled(_)));
    assert_eq!(
        cache.positions(),
        2,
        "cancelled forward must roll back appended KV positions"
    );

    // Host-level cancel removes the session and reports released KV bytes.
    let host = NativeSessionHost::new();
    host.register_model_dir("llama3_bf16", fixture("llama3_bf16"));
    let SessionResponse::Open(opened) = host.execute(SessionRequest::Open(SessionOpenRequest {
        session_id: "sess-cancel".to_string(),
        model_id: "llama3_bf16".to_string(),
        start_layer: 0,
        end_layer: 2,
        embedding: true,
        head: false,
    })) else {
        panic!("expected OpenResponse");
    };
    assert_eq!(opened.hidden_size, 32);
    assert_eq!(host.active_session_count(), 1);

    let SessionResponse::Forward(fwd) =
        host.execute(SessionRequest::Forward(SessionForwardRequest {
            session_id: "sess-cancel".to_string(),
            input: ForwardInput::Tokens {
                start_position: 0,
                tokens: vec![1, 2, 3],
            },
            emit_logits: false,
        }))
    else {
        panic!("expected ForwardResponse");
    };
    assert_eq!(fwd.cache_positions, 3);
    assert!(fwd.kv_cache_bytes > 0);

    let SessionResponse::Cancel(cancelled_resp) = host.execute(SessionRequest::Cancel(
        kadmium_runtime::native_decoder::SessionCancelRequest {
            session_id: "sess-cancel".to_string(),
        },
    )) else {
        panic!("expected CancelResponse");
    };
    assert_eq!(cancelled_resp.released_kv_bytes, fwd.kv_cache_bytes);
    assert_eq!(host.active_session_count(), 0);
}

#[tokio::test]
async fn live_worker_tcp_chain_is_bit_identical_to_unsplit_and_supports_cancel() {
    let key = test_auth_key(b"kadmium-live-tcp-chain-auth-key-2026");
    let limits = NativeSessionLimits::default();

    let mut workers = Vec::new();
    for idx in 0..3 {
        let server = LiveNodeWorkerServer::spawn_auto_port(
            "127.0.0.1",
            0,
            sample_worker_node(&format!("node-native-slice-{idx}")),
            false,
        )
        .await
        .unwrap();
        server.native_sessions.set_auth_key(key.clone());
        for name in FIXTURES {
            server
                .native_sessions
                .register_model_dir(name, fixture(name));
        }
        workers.push(server);
    }

    for (epoch, name) in FIXTURES.iter().enumerate() {
        let epoch = (epoch + 1) as u64;
        let reference = reference(name);
        let unsplit = whole(name);
        let n = unsplit.config().num_layers;
        assert_eq!(n, 3);
        let expected = run_chain(&[unsplit], &reference.tokens, 4);

        let s0 = format!("{name}-stage-0");
        let s1 = format!("{name}-stage-1");
        let s2 = format!("{name}-stage-2");

        let addr0 = workers[0].local_addr.to_string();
        let addr1 = workers[1].local_addr.to_string();
        let addr2 = workers[2].local_addr.to_string();

        DistributedLanExecutor::open_native_session_tcp(
            &addr0,
            epoch,
            SessionOpenRequest {
                session_id: s0.clone(),
                model_id: (*name).to_string(),
                start_layer: 0,
                end_layer: 1,
                embedding: true,
                head: false,
            },
            &key,
            &limits,
        )
        .await
        .unwrap();
        DistributedLanExecutor::open_native_session_tcp(
            &addr1,
            epoch,
            SessionOpenRequest {
                session_id: s1.clone(),
                model_id: (*name).to_string(),
                start_layer: 1,
                end_layer: 2,
                embedding: false,
                head: false,
            },
            &key,
            &limits,
        )
        .await
        .unwrap();
        DistributedLanExecutor::open_native_session_tcp(
            &addr2,
            epoch,
            SessionOpenRequest {
                session_id: s2.clone(),
                model_id: (*name).to_string(),
                start_layer: 2,
                end_layer: n,
                embedding: false,
                head: true,
            },
            &key,
            &limits,
        )
        .await
        .unwrap();

        let run_remote_step = |tokens: Vec<u32>, start_position: usize| {
            let addr0 = addr0.clone();
            let addr1 = addr1.clone();
            let addr2 = addr2.clone();
            let s0 = s0.clone();
            let s1 = s1.clone();
            let s2 = s2.clone();
            let key = key.clone();
            async move {
                let r0 = DistributedLanExecutor::forward_native_session_tcp(
                    &addr0,
                    epoch,
                    SessionForwardRequest {
                        session_id: s0,
                        input: ForwardInput::Tokens {
                            start_position,
                            tokens,
                        },
                        emit_logits: false,
                    },
                    &key,
                    &limits,
                )
                .await
                .unwrap();
                let ForwardOutput::Activations(a0) = r0.output else {
                    panic!("stage 0 must return activations");
                };

                let r1 = DistributedLanExecutor::forward_native_session_tcp(
                    &addr1,
                    epoch,
                    SessionForwardRequest {
                        session_id: s1,
                        input: ForwardInput::Activations(a0),
                        emit_logits: false,
                    },
                    &key,
                    &limits,
                )
                .await
                .unwrap();
                let ForwardOutput::Activations(a1) = r1.output else {
                    panic!("stage 1 must return activations");
                };

                let r2 = DistributedLanExecutor::forward_native_session_tcp(
                    &addr2,
                    epoch,
                    SessionForwardRequest {
                        session_id: s2,
                        input: ForwardInput::Activations(a1),
                        emit_logits: true,
                    },
                    &key,
                    &limits,
                )
                .await
                .unwrap();
                let ForwardOutput::Logits { data, .. } = r2.output else {
                    panic!("stage 2 must return logits");
                };
                data
            }
        };

        let prefill = 4usize;
        let mut remote_logits = run_remote_step(reference.tokens[..prefill].to_vec(), 0).await;
        for (i, &tok) in reference.tokens[prefill..].iter().enumerate() {
            remote_logits.extend(run_remote_step(vec![tok], prefill + i).await);
        }

        assert_eq!(
            bits(&remote_logits),
            bits(&expected),
            "{name}: 3-worker TCP chain differed from unsplit execution"
        );

        let closed0 =
            DistributedLanExecutor::close_native_session_tcp(&addr0, epoch, &s0, &key, &limits)
                .await
                .unwrap();
        let closed1 =
            DistributedLanExecutor::close_native_session_tcp(&addr1, epoch, &s1, &key, &limits)
                .await
                .unwrap();
        let closed2 =
            DistributedLanExecutor::close_native_session_tcp(&addr2, epoch, &s2, &key, &limits)
                .await
                .unwrap();
        assert_eq!(closed0.final_positions, reference.tokens.len());
        assert_eq!(closed1.final_positions, reference.tokens.len());
        assert_eq!(closed2.final_positions, reference.tokens.len());
        assert!(closed0.released_kv_bytes > 0);
    }

    // Verify that Cancel bypasses a saturated forward semaphore on the worker connection.
    let addr = workers[0].local_addr.to_string();
    DistributedLanExecutor::open_native_session_tcp(
        &addr,
        99,
        SessionOpenRequest {
            session_id: "sess-priority-cancel".to_string(),
            model_id: "llama3_bf16".to_string(),
            start_layer: 0,
            end_layer: 1,
            embedding: true,
            head: false,
        },
        &key,
        &limits,
    )
    .await
    .unwrap();

    // Hold all native_session_slots permits on worker 0 so any non-priority frame would block.
    let all_permits = Arc::clone(&workers[0].native_session_slots)
        .acquire_many_owned(
            kadmium_runtime::native_decoder::MAX_IN_FLIGHT_NATIVE_SESSION_FRAMES as u32,
        )
        .await
        .unwrap();

    let cancel_ack = tokio::time::timeout(
        Duration::from_secs(2),
        DistributedLanExecutor::cancel_native_session_tcp(
            &addr,
            99,
            "sess-priority-cancel",
            &key,
            &limits,
        ),
    )
    .await
    .expect("Cancel frame must bypass saturated native_session_slots")
    .unwrap();
    assert_eq!(cancel_ack.session_id, "sess-priority-cancel");
    drop(all_permits);

    // Wrong key over TCP is rejected with NativeDecoderError::Authentication.
    let bad_key = test_auth_key(b"kadmium-wrong-tcp-auth-key-00000");
    let auth_err = DistributedLanExecutor::open_native_session_tcp(
        &addr,
        100,
        SessionOpenRequest {
            session_id: "sess-unauth".to_string(),
            model_id: "llama3_bf16".to_string(),
            start_layer: 0,
            end_layer: 1,
            embedding: true,
            head: false,
        },
        &bad_key,
        &limits,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(auth_err, NativeDecoderError::Authentication(_)),
        "expected Authentication error, got {auth_err:?}"
    );
    let _ = WireFrameKind::NativeDecoderSession;
}
