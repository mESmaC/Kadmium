use kadmium_data::SafetensorsManifest;
use kadmium_runtime::native_decoder::{DecoderConfig, SlicePlan};
use std::path::Path;

#[test]
fn slice_tensor_names_match_lean_selection() {
    let vectors = include_str!("fixtures/native_staging_lean_vectors.txt");
    let mut checked = 0;
    for model in ["llama3_bf16", "qwen2_f32", "qwen3_bf16", "mistral_f16"] {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/native_decoder")
            .join(model);
        let config = DecoderConfig::from_model_dir(&directory).unwrap();
        let manifest = SafetensorsManifest::open(&directory).unwrap();
        for line in vectors
            .lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 6);
            let tied: bool = fields[4].parse().unwrap();
            if tied != config.tie_word_embeddings {
                continue;
            }
            let plan = SlicePlan {
                layers: fields[0].parse().unwrap()..fields[1].parse().unwrap(),
                embedding: fields[2].parse().unwrap(),
                head: fields[3].parse().unwrap(),
            };
            let flags: Vec<_> = fields[5]
                .split(',')
                .map(|value| match value {
                    "0" => false,
                    "1" => true,
                    _ => panic!("invalid selection flag"),
                })
                .collect();
            assert_eq!(flags.len(), 6);
            let selected = plan.tensor_names(&manifest, &config).unwrap();
            for tensor in manifest.tensors() {
                let role = match tensor.name.as_str() {
                    "model.embed_tokens.weight" => 0,
                    "model.norm.weight" => 1,
                    "lm_head.weight" => 2,
                    name => {
                        3 + name
                            .strip_prefix("model.layers.")
                            .unwrap()
                            .split('.')
                            .next()
                            .unwrap()
                            .parse::<usize>()
                            .unwrap()
                    }
                };
                assert_eq!(
                    selected.contains(&tensor.name),
                    flags[role],
                    "{model}: {line}: {}",
                    tensor.name
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 24);
}
