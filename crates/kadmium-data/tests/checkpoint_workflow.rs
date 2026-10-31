use kadmium_data::checkpoint::CheckpointLibrary;
use kadmium_data::{KtbBankReader, SafetensorsToKtbTranscoder, TensorEncoding};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(1);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "checkpoint-workflow-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(path.join("source")).unwrap();
        Self(path)
    }
    fn source(&self) -> PathBuf {
        self.0.join("source")
    }
    fn library(&self) -> CheckpointLibrary {
        CheckpointLibrary::new(self.0.join("library"))
    }
    fn metadata(&self) {
        fs::write(
            self.source().join("config.json"),
            br#"{"model_type":"llama","max_position_embeddings":8192}"#,
        )
        .unwrap();
        fs::write(self.source().join("tokenizer.json"), b"{}").unwrap();
        fs::write(
            self.source().join("tokenizer_config.json"),
            br#"{"chat_template":"keep all reasoning"}"#,
        )
        .unwrap();
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn normalize_preserves_f32_bf16_metadata_and_all_layers_across_banks() {
    let temp = Scratch::new();
    temp.metadata();
    let payload = [0u8, 0, 128, 63, 1, 0, 128, 63]; // Adjacent F32 values, distinct in F32 only.
    let names: Vec<_> = (0..70)
        .map(|i| format!("model.layers.{i}.self_attn.q_proj.weight"))
        .collect();
    let tensors: Vec<_> = names
        .iter()
        .map(|n| (n.as_str(), "F32", vec![1, 2], payload.as_slice()))
        .chain([
            (
                "model.embed_tokens.weight",
                "BF16",
                vec![2, 2],
                payload.as_slice(),
            ),
            ("lm_head.weight", "F16", vec![2, 2], payload.as_slice()),
        ])
        .collect();
    let original = SafetensorsToKtbTranscoder::build_safetensors_bytes(&tensors);
    fs::write(temp.source().join("model.safetensors"), &original).unwrap();
    let lib = temp.library();
    let manifest = lib.import(&temp.source()).unwrap();
    assert_eq!(manifest.tensors.len(), 72);
    assert_eq!(manifest.banks.len(), 2);
    assert_eq!(manifest.context_length, 8192);
    let dir = lib.package_dir(&manifest.id).unwrap();
    assert_eq!(
        fs::read(dir.join("source/model.safetensors")).unwrap(),
        original
    );
    assert_eq!(
        fs::read(dir.join("source/tokenizer_config.json")).unwrap(),
        fs::read(temp.source().join("tokenizer_config.json")).unwrap()
    );
    for bank in &manifest.banks {
        let bytes = fs::read(dir.join("banks").join(&bank.name)).unwrap();
        let reader = KtbBankReader::from_slice(&bytes).unwrap();
        for entry in &reader.header.entries {
            assert_eq!(reader.get_verified_tensor_slice(entry).unwrap(), payload);
            if entry.tensor_name.contains("self_attn") {
                assert_eq!(entry.encoding, TensorEncoding::Fp32);
            }
        }
    }
    assert_eq!(lib.verify(&manifest.id).unwrap().id, manifest.id);
    assert_eq!(lib.import(&temp.source()).unwrap().id, manifest.id);
    assert_eq!(lib.list().unwrap().len(), 1);
    let manifest_path = dir.join("manifest.json");
    let original_manifest = fs::read(&manifest_path).unwrap();
    let mut changed = manifest.clone();
    changed.tensors[0].bank_offset += 65536;
    fs::write(&manifest_path, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(
        lib.verify(&manifest.id)
            .unwrap_err()
            .contains("bank location")
    );
    fs::write(&manifest_path, original_manifest).unwrap();
    let bank_path = dir.join("banks").join(&manifest.banks[0].name);
    let mut bytes = fs::read(&bank_path).unwrap();
    bytes[65536] ^= 1;
    fs::write(bank_path, bytes).unwrap();
    assert!(lib.verify(&manifest.id).unwrap_err().contains("Integrity"));
}

#[test]
fn malformed_and_incomplete_shards_publish_nothing() {
    let temp = Scratch::new();
    temp.metadata();
    let bytes =
        SafetensorsToKtbTranscoder::build_safetensors_bytes(&[("weight", "F32", vec![2], &[0; 8])]);
    fs::write(temp.source().join("shard-1.safetensors"), bytes).unwrap();
    fs::write(
        temp.source().join("model.safetensors.index.json"),
        json!({"weight_map":{"weight":"shard-2.safetensors"}}).to_string(),
    )
    .unwrap();
    let lib = temp.library();
    assert!(lib.import(&temp.source()).is_err());
    assert!(lib.list().unwrap().is_empty());
    fs::write(
        temp.source().join("model.safetensors.index.json"),
        r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
    )
    .unwrap();
    assert!(lib.import(&temp.source()).is_err());
    fs::remove_file(temp.source().join("model.safetensors.index.json")).unwrap();
    fs::rename(
        temp.source().join("shard-1.safetensors"),
        temp.source().join("model.safetensors"),
    )
    .unwrap();
    let malformed =
        SafetensorsToKtbTranscoder::build_safetensors_bytes(&[("weight", "F32", vec![9], &[0; 8])]);
    fs::write(temp.source().join("model.safetensors"), malformed).unwrap();
    assert!(lib.import(&temp.source()).is_err());
    assert!(lib.list().unwrap().is_empty());
    assert!(lib.read("../../outside").is_err());
}

#[test]
fn sharded_checkpoint_keeps_the_complete_index_and_detects_source_damage() {
    let temp = Scratch::new();
    temp.metadata();
    let mut map = serde_json::Map::new();
    for i in 0..2 {
        let name = format!("model.layers.{i}.weight");
        let file = format!("model-{i}.safetensors");
        fs::write(
            temp.source().join(&file),
            SafetensorsToKtbTranscoder::build_safetensors_bytes(&[(
                &name,
                "BF16",
                vec![2],
                &[0; 4],
            )]),
        )
        .unwrap();
        map.insert(name, json!(file));
    }
    fs::write(
        temp.source().join("model.safetensors.index.json"),
        json!({"weight_map":map,"metadata":{"total_size":8}}).to_string(),
    )
    .unwrap();
    let lib = temp.library();
    let m = lib.import(&temp.source()).unwrap();
    assert_eq!(m.tensors.len(), 2);
    fs::write(
        lib.package_dir(&m.id).unwrap().join("source/config.json"),
        b"{}",
    )
    .unwrap();
    assert!(lib.verify(&m.id).is_err());
}

// GGUF v3 metadata, tokenizer array, nondefault alignment and unordered tensor offsets.
fn gguf(dtype: u32, splits: u32) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    out.extend(6u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", "llama"),
        ("tokenizer.ggml.model", "llama"),
    ] {
        string(&mut out, key);
        out.extend(8u32.to_le_bytes());
        string(&mut out, value);
    }
    for (key, value) in [
        ("general.alignment", 64),
        ("llama.context_length", 8192),
        ("split.count", splits),
    ] {
        string(&mut out, key);
        out.extend(4u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    }
    string(&mut out, "tokenizer.ggml.tokens");
    out.extend(9u32.to_le_bytes());
    out.extend(8u32.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    string(&mut out, "a");
    string(&mut out, "b");
    for (name, offset) in [("output.weight", 64u64), ("token_embd.weight", 0)] {
        string(&mut out, name);
        out.extend(1u32.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        out.extend(dtype.to_le_bytes());
        out.extend(offset.to_le_bytes());
    }
    out.resize(out.len().div_ceil(64) * 64, 0);
    out.extend([1u8; 8]);
    out.extend([0u8; 56]);
    out.extend([2u8; 8]);
    out
}

#[test]
fn gguf_metadata_alignment_and_f32_are_retained_without_guessing_unknown_types() {
    let temp = Scratch::new();
    let path = temp.source().join("model.gguf");
    let bytes = gguf(0, 1);
    fs::write(&path, &bytes).unwrap();
    let lib = temp.library();
    let m = lib.import(&path).unwrap();
    assert_eq!(m.context_length, 8192);
    assert!(m.tensors.iter().all(|t| t.encoding == TensorEncoding::Fp32));
    for t in &m.tensors {
        let bank = fs::read(lib.package_dir(&m.id).unwrap().join("banks").join(&t.bank)).unwrap();
        assert_eq!(
            &bank[t.bank_offset as usize..(t.bank_offset + t.bytes) as usize],
            &bytes[t.source_offset as usize..(t.source_offset + t.bytes) as usize]
        );
    }
    fs::write(&path, gguf(999, 1)).unwrap();
    assert!(lib.import(&path).unwrap_err().contains("Unsupported GGML"));
    fs::write(&path, gguf(0, 2)).unwrap();
    assert!(lib.import(&path).unwrap_err().contains("requires 2 shards"));
    assert_eq!(lib.list().unwrap().len(), 1);
}

fn split_gguf(split: u32) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    out.extend((if split == 0 { 6u64 } else { 3 }).to_le_bytes());
    if split == 0 {
        for (key, value) in [
            ("general.architecture", "llama"),
            ("tokenizer.ggml.model", "llama"),
        ] {
            string(&mut out, key);
            out.extend(8u32.to_le_bytes());
            string(&mut out, value);
        }
        string(&mut out, "llama.context_length");
        out.extend(4u32.to_le_bytes());
        out.extend(4096u32.to_le_bytes());
    }
    for (key, value) in [
        ("split.count", 2),
        ("split.no", split),
        ("split.tensors.count", 2),
    ] {
        string(&mut out, key);
        out.extend(4u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    }
    string(&mut out, &format!("blk.{split}.weight"));
    out.extend(1u32.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.resize(out.len().div_ceil(32) * 32, 0);
    out.extend([split as u8; 8]);
    out
}

#[test]
fn complete_gguf_splits_accept_metadata_in_first_shard_only() {
    let temp = Scratch::new();
    for split in 0..2 {
        fs::write(
            temp.source().join(format!("model-{split}.gguf")),
            split_gguf(split),
        )
        .unwrap();
    }
    let lib = temp.library();
    let manifest = lib.import(&temp.source()).unwrap();
    assert_eq!(manifest.context_length, 4096);
    assert_eq!(manifest.tensors.len(), 2);
    lib.verify(&manifest.id).unwrap();
    fs::write(temp.source().join("model-1.gguf"), split_gguf(0)).unwrap();
    assert!(
        lib.import(&temp.source())
            .unwrap_err()
            .contains("split number")
    );
    assert_eq!(lib.list().unwrap().len(), 1);
}
