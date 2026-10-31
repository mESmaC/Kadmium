//! Pre-quantized sources import only with explicit operator acceptance, and every stored
//! precision is copied byte for byte.

use kadmium_data::SafetensorsToKtbTranscoder;
use kadmium_data::checkpoint::{CheckpointLibrary, CheckpointManifest, ImportOptions};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);
const ACCEPT: ImportOptions = ImportOptions {
    accept_pre_quantized: true,
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "pre-quantized-{}-{}",
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
    fn safetensors(&self, config: &str, tensors: &[(&str, &str, Vec<usize>, &[u8])]) {
        fs::write(self.source().join("config.json"), config).unwrap();
        fs::write(self.source().join("tokenizer.json"), b"{}").unwrap();
        fs::write(
            self.source().join("model.safetensors"),
            SafetensorsToKtbTranscoder::build_safetensors_bytes(tensors),
        )
        .unwrap();
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `(code, name, elements per block, bytes per block)` from ggml's `type_traits`.
const GGML: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (31, "Q4_0_4_4", 32, 18),
    (32, "Q4_0_4_8", 32, 18),
    (33, "Q4_0_8_8", 32, 18),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (36, "IQ4_NL_4_4", 32, 18),
    (37, "IQ4_NL_4_8", 32, 18),
    (38, "IQ4_NL_8_8", 32, 18),
    (39, "MXFP4", 32, 17),
];

/// A GGUF v3 file with one two-block tensor per listed type and distinct payload bytes.
fn gguf(types: &[(u32, &str, u64, u64)]) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((types.len() as u64).to_le_bytes());
    out.extend(3u64.to_le_bytes());
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
    let mut payload = Vec::new();
    for (code, name, block, bytes) in types {
        string(&mut out, &format!("blk.0.{name}.weight"));
        out.extend(1u32.to_le_bytes());
        out.extend((block * 2).to_le_bytes());
        out.extend(code.to_le_bytes());
        out.extend((payload.len() as u64).to_le_bytes());
        payload.extend((0..bytes * 2).map(|i| (i as u8) ^ (*code as u8).wrapping_mul(37)));
        payload.resize(payload.len().div_ceil(32) * 32, 0);
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    out.extend(payload);
    out
}

fn assert_banks_match_source(
    lib: &CheckpointLibrary,
    manifest: &CheckpointManifest,
    source: &Path,
) {
    let dir = lib.package_dir(&manifest.id).unwrap();
    for t in &manifest.tensors {
        let original = fs::read(source.join(&t.source)).unwrap();
        let bank = fs::read(dir.join("banks").join(&t.bank)).unwrap();
        assert_eq!(
            &bank[t.bank_offset as usize..(t.bank_offset + t.bytes) as usize],
            &original[t.source_offset as usize..(t.source_offset + t.bytes) as usize],
            "{} changed during normalization",
            t.name
        );
    }
}

#[test]
fn every_ggml_type_is_copied_exactly_after_the_operator_accepts_quantization() {
    let temp = Scratch::new();
    let path = temp.source().join("model.gguf");
    fs::write(&path, gguf(GGML)).unwrap();
    let lib = temp.library();

    let refused = lib.import(&path).unwrap_err();
    assert!(refused.contains("already quantized"), "{refused}");
    assert!(
        refused.contains("Q8_1") && refused.contains("MXFP4"),
        "{refused}"
    );
    assert!(lib.list().unwrap().is_empty());

    let manifest = lib.import_with(&path, ACCEPT).unwrap();
    assert_eq!(manifest.tensors.len(), GGML.len());
    for (code, name, _, bytes) in GGML {
        let tensor = manifest
            .tensors
            .iter()
            .find(|t| t.name == format!("blk.0.{name}.weight"))
            .unwrap();
        assert_eq!(tensor.bytes, bytes * 2, "{name}");
        assert_eq!(tensor.dtype, format!("GGML_{code}"));
        assert_eq!(kadmium_data::checkpoint::ggml_type_name(*code), Some(*name));
    }
    let quantized = manifest.pre_quantized.clone().unwrap();
    assert_eq!(quantized.method.as_deref(), Some("gguf"));
    for exact in ["F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"] {
        assert!(
            !quantized.tensor_formats.iter().any(|f| f == exact),
            "{exact}"
        );
    }
    assert_eq!(quantized.tensor_formats.len(), GGML.len() - 8);
    assert!(
        manifest
            .warnings
            .iter()
            .any(|w| w.contains("pre-quantized"))
    );
    assert_banks_match_source(&lib, &manifest, &temp.source());

    let verified = lib.verify(&manifest.id).unwrap();
    assert_eq!(verified.pre_quantized, manifest.pre_quantized);
    assert_eq!(lib.list().unwrap()[0].pre_quantized, manifest.pre_quantized);
    // Acceptance is per import: reimporting the same files without it is refused again.
    assert!(lib.import(&path).is_err());
}

#[test]
fn sixteen_bit_and_wider_sources_need_no_acceptance() {
    let temp = Scratch::new();
    let path = temp.source().join("model.gguf");
    let exact: Vec<_> = GGML
        .iter()
        .copied()
        .filter(|(code, ..)| matches!(code, 0 | 1 | 24..=28 | 30))
        .collect();
    fs::write(&path, gguf(&exact)).unwrap();
    let manifest = temp.library().import(&path).unwrap();
    assert_eq!(manifest.pre_quantized, None);

    let st = Scratch::new();
    st.safetensors(
        r#"{"model_type":"llama","max_position_embeddings":4096}"#,
        &[
            ("a.weight", "F64", vec![1], &[0; 8]),
            ("b.weight", "BF16", vec![2], &[1; 4]),
            ("position_ids", "I64", vec![1], &[2; 8]),
        ],
    );
    let manifest = st.library().import(&st.source()).unwrap();
    assert_eq!(manifest.pre_quantized, None);
    assert!(manifest.warnings.iter().any(|w| w.contains("integer")));
}

#[test]
fn declared_and_undeclared_safetensors_quantization_is_detected() {
    // Transformers `quantization_config`: GPTQ packs 4-bit weights into I32 words.
    let gptq = Scratch::new();
    gptq.safetensors(
        r#"{"model_type":"llama","max_position_embeddings":4096,
            "quantization_config":{"quant_method":"gptq","bits":4,"group_size":128}}"#,
        &[
            ("model.layers.0.mlp.qweight", "I32", vec![2], &[7; 8]),
            ("model.layers.0.mlp.scales", "F16", vec![2], &[3; 4]),
        ],
    );
    let lib = gptq.library();
    assert!(
        lib.import(&gptq.source())
            .unwrap_err()
            .contains("gptq 4-bit (I32)")
    );
    let manifest = lib.import_with(&gptq.source(), ACCEPT).unwrap();
    let quantized = manifest.pre_quantized.clone().unwrap();
    assert_eq!(quantized.method.as_deref(), Some("gptq"));
    assert_eq!(quantized.bits, Some(4));
    assert_eq!(quantized.tensor_formats, ["I32"]);
    assert_banks_match_source(&lib, &manifest, &gptq.source());
    lib.verify(&manifest.id).unwrap();

    // FP8 weights with no declared method are still below 16 bits.
    let fp8 = Scratch::new();
    fp8.safetensors(
        r#"{"model_type":"deepseek_v3","max_position_embeddings":4096}"#,
        &[
            ("model.layers.0.mlp.weight", "F8_E4M3", vec![4], &[9; 4]),
            (
                "model.layers.0.mlp.weight_scale_inv",
                "F32",
                vec![1],
                &[1; 4],
            ),
        ],
    );
    let refused = fp8.library().import(&fp8.source()).unwrap_err();
    assert!(refused.contains("undeclared method (F8_E4M3)"), "{refused}");
    let manifest = fp8.library().import_with(&fp8.source(), ACCEPT).unwrap();
    assert_eq!(manifest.pre_quantized.unwrap().method, None);

    // compressed-tensors exports may declare their scheme as `compression_config`.
    let compressed = Scratch::new();
    compressed.safetensors(
        r#"{"model_type":"llama","max_position_embeddings":4096,
            "compression_config":{"quant_method":"compressed-tensors",
                "config_groups":{"group_0":{"weights":{"num_bits":4}}}}}"#,
        &[("model.layers.0.mlp.weight_packed", "I32", vec![1], &[5; 4])],
    );
    let refused = compressed
        .library()
        .import(&compressed.source())
        .unwrap_err();
    assert!(
        refused.contains("compressed-tensors 4-bit (I32)"),
        "{refused}"
    );

    // Integer weights with no declared method are quantized storage too.
    for (name, dtype) in [
        ("model.layers.0.mlp.down_proj.weight", "I8"),
        ("model.layers.0.mlp.qweight", "I32"),
        ("model.layers.0.mlp.qzeros", "U32"),
    ] {
        let bytes = if dtype == "I8" { 4 } else { 16 };
        let undeclared = Scratch::new();
        undeclared.safetensors(
            r#"{"model_type":"llama","max_position_embeddings":4096}"#,
            &[
                (name, dtype, vec![4], &[3; 16][..bytes]),
                ("position_ids", "I64", vec![1], &[0; 8]),
            ],
        );
        let refused = undeclared
            .library()
            .import(&undeclared.source())
            .unwrap_err();
        assert!(
            refused.contains(&format!("undeclared method ({dtype})")),
            "{refused}"
        );
    }

    // AutoGPTQ's sidecar is copied into the package and read for the method.
    let sidecar = Scratch::new();
    sidecar.safetensors(
        r#"{"model_type":"llama","max_position_embeddings":4096}"#,
        &[("model.layers.0.mlp.qweight", "I32", vec![1], &[5; 4])],
    );
    fs::write(
        sidecar.source().join("quantize_config.json"),
        br#"{"bits":4,"group_size":128}"#,
    )
    .unwrap();
    // Chat template and processor files travel with the weights.
    for name in [
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
    ] {
        fs::write(sidecar.source().join(name), b"{}").unwrap();
    }
    let lib = sidecar.library();
    let manifest = lib.import_with(&sidecar.source(), ACCEPT).unwrap();
    assert_eq!(
        manifest.pre_quantized.as_ref().unwrap().description(),
        "gptq 4-bit (I32)"
    );
    for name in [
        "quantize_config.json",
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
    ] {
        let copied = lib
            .package_dir(&manifest.id)
            .unwrap()
            .join("source")
            .join(name);
        assert!(copied.is_file(), "{name} was not copied");
    }

    // Packed FP4 described by an NVIDIA ModelOpt sidecar.
    let fp4 = Scratch::new();
    fp4.safetensors(
        r#"{"model_type":"llama","max_position_embeddings":4096}"#,
        &[
            (
                "model.layers.0.mlp.weight",
                "F4",
                vec![2, 3],
                &[0x21, 0x43, 0x65],
            ),
            ("model.layers.0.mlp.weight_scale", "F8_E4M3", vec![1], &[1]),
        ],
    );
    fs::write(
        fp4.source().join("hf_quant_config.json"),
        br#"{"producer":{"name":"modelopt"},"quantization":{"quant_algo":"NVFP4"}}"#,
    )
    .unwrap();
    let lib = fp4.library();
    let manifest = lib.import_with(&fp4.source(), ACCEPT).unwrap();
    let quantized = manifest.pre_quantized.clone().unwrap();
    assert_eq!(quantized.method.as_deref(), Some("modelopt nvfp4"));
    assert_eq!(quantized.tensor_formats, ["F4", "F8_E4M3"]);
    let packed = manifest.tensors.iter().find(|t| t.dtype == "F4").unwrap();
    assert_eq!(packed.bytes, 3);
    assert_banks_match_source(&lib, &manifest, &fp4.source());
}

#[test]
fn legacy_packages_record_their_precision_and_current_ones_cannot_hide_it() {
    let temp = Scratch::new();
    let path = temp.source().join("model.gguf");
    fs::write(&path, gguf(&GGML[..9])).unwrap();
    let lib = temp.library();
    let manifest = lib.import_with(&path, ACCEPT).unwrap();
    let manifest_path = lib.package_dir(&manifest.id).unwrap().join("manifest.json");

    let mut legacy = manifest.clone();
    legacy.schema_version = 2;
    legacy.pre_quantized = None;
    fs::write(&manifest_path, serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(lib.list().unwrap()[0].pre_quantized, None);
    legacy.warnings.retain(|w| !w.contains("pre-quantized"));
    fs::write(&manifest_path, serde_json::to_vec(&legacy).unwrap()).unwrap();
    let upgraded = lib.verify(&manifest.id).unwrap();
    assert_eq!(upgraded.schema_version, 3);
    assert_eq!(upgraded.pre_quantized, manifest.pre_quantized);
    assert_eq!(upgraded.warnings, manifest.warnings);
    assert_eq!(lib.list().unwrap()[0].pre_quantized, manifest.pre_quantized);

    let mut hidden = manifest.clone();
    hidden.pre_quantized = None;
    fs::write(&manifest_path, serde_json::to_vec(&hidden).unwrap()).unwrap();
    assert!(
        lib.verify(&manifest.id)
            .unwrap_err()
            .contains("source precision")
    );
}
