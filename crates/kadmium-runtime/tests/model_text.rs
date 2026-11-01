use kadmium_runtime::model_text::{ModelText, TemplateValue};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/model_text")
        .join(name)
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn write(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "kadmium-text-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        for entry in std::fs::read_dir(fixture("byte_bpe")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), path.join(entry.file_name())).unwrap();
        }
        Self(path)
    }
    fn edit(&self, name: &str, edit: impl FnOnce(&mut Value)) {
        let path = self.0.join(name);
        let mut value = read(&path);
        edit(&mut value);
        write(&path, &value);
    }
    fn load(&self) -> ModelText {
        ModelText::from_model_dir(&self.0).unwrap()
    }
    fn error(&self) -> String {
        ModelText::from_model_dir(&self.0)
            .err()
            .unwrap()
            .to_string()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn compare(model: &ModelText, references: &Path) {
    let reference = read(references);
    for case in reference["plain"].as_array().unwrap() {
        let ids = model
            .encode(
                case["text"].as_str().unwrap(),
                case["special"].as_bool().unwrap(),
            )
            .unwrap();
        assert_eq!(json!(ids), case["ids"], "{case}");
        assert_eq!(
            model.decode(&ids, false).unwrap(),
            case["decoded"].as_str().unwrap()
        );
        assert_eq!(
            model.decode(&ids, true).unwrap(),
            case["skipped"].as_str().unwrap()
        );
    }
    #[derive(Deserialize)]
    struct Chat {
        messages: Vec<TemplateValue>,
        generation: bool,
        variables: BTreeMap<String, TemplateValue>,
        prompt: String,
        ids: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct Reference {
        chats: Vec<Chat>,
    }
    let chats: Reference = serde_json::from_slice(&std::fs::read(references).unwrap()).unwrap();
    for case in chats.chats {
        let messages = &case.messages;
        let generation = case.generation;
        let variables = &case.variables;
        let prompt = model
            .render_chat(messages, generation, None, variables, 1_000_000)
            .unwrap();
        assert_eq!(prompt, case.prompt);
        assert_eq!(
            json!(
                model
                    .encode_chat(messages, generation, None, variables, 1_000_000)
                    .unwrap()
            ),
            json!(case.ids)
        );
    }
}

#[test]
fn token_ids_decoding_and_chat_match_transformers() {
    for name in ["byte_bpe", "unigram"] {
        compare(
            &ModelText::from_model_dir(&fixture(name)).unwrap(),
            &fixture(name).join("reference.json"),
        );
    }
}

#[test]
fn sampling_and_stops_match_checkpoint_defaults_without_applying_length_caps() {
    let model = ModelText::from_model_dir(&fixture("byte_bpe")).unwrap();
    let defaults = model.generation_defaults();
    let expected = read(&fixture("byte_bpe").join("generation_reference.json"));
    let sampling = &defaults.sampling;
    assert_eq!(
        json!({"do_sample": sampling.do_sample, "temperature": sampling.temperature,
        "top_k": sampling.top_k, "top_p": sampling.top_p, "min_p": sampling.min_p,
        "typical_p": sampling.typical_p, "repetition_penalty": sampling.repetition_penalty,
        "eos_token_id": defaults.eos_token_ids}),
        expected
    );
    assert!(defaults.is_eos(2));
    assert!(defaults.is_eos(4));
    assert!(!defaults.is_eos(3));
    assert_eq!(defaults.find_stop("abcEN"), None);
    assert_eq!(defaults.find_stop("abcENDtail"), Some(3));
    assert_eq!(defaults.find_stop("é结束tailEND"), Some(2));
    assert_eq!(defaults.declared["max_new_tokens"], 17);
    assert!(model.encode(&"a".repeat(100), false).unwrap().len() > 17);
}

#[test]
fn empty_generation_config_is_not_legacy_config_and_explicit_null_disables_eos() {
    let temp = Temp::new();
    temp.edit("config.json", |v| {
        v["temperature"] = json!(0.2);
        v["do_sample"] = json!(true);
    });
    write(
        &temp.0.join("generation_config.json"),
        &json!({"eos_token_id":null}),
    );
    let model = temp.load();
    assert_eq!(model.generation_defaults().sampling.temperature, 1.0);
    assert!(!model.generation_defaults().sampling.do_sample);
    assert!(model.generation_defaults().eos_token_ids.is_empty());
    std::fs::remove_file(temp.0.join("generation_config.json")).unwrap();
    let model = temp.load();
    assert_eq!(model.generation_defaults().sampling.temperature, 0.2);
    assert!(model.generation_defaults().sampling.do_sample);
    assert_eq!(model.generation_defaults().eos_token_ids, [2]);
}

#[test]
fn serialized_truncation_and_padding_cannot_cut_or_extend_the_prompt() {
    let temp = Temp::new();
    let original = temp
        .load()
        .encode("first second third fourth", false)
        .unwrap();
    temp.edit("tokenizer.json", |v| {
        v["truncation"] =
            json!({"direction":"Right", "max_length":2, "strategy":"LongestFirst", "stride":0});
        v["padding"] = json!({"strategy":{"Fixed":80},"direction":"Right","pad_to_multiple_of":null,
            "pad_id":2,"pad_type_id":0,"pad_token":"</s>"});
    });
    assert_eq!(
        temp.load()
            .encode("first second third fourth", false)
            .unwrap(),
        original
    );
    temp.edit("config.json", |v| {
        v["max_position_embeddings"] = json!(original.len())
    });
    assert_eq!(
        temp.load()
            .encode("first second third fourth", false)
            .unwrap(),
        original
    );
    temp.edit("config.json", |v| {
        v["max_position_embeddings"] = json!(original.len() - 1)
    });
    let error = temp
        .load()
        .encode("first second third fourth", false)
        .unwrap_err()
        .to_string();
    assert!(error.contains(&format!("position {}", original.len())));
    assert!(error.contains(&format!("is {} positions", original.len() - 1)));
}

#[test]
fn named_templates_precedence_missing_assets_and_template_failures() {
    let temp = Temp::new();
    std::fs::remove_file(temp.0.join("chat_template.jinja")).unwrap();
    temp.edit("tokenizer_config.json", |v| {
        v["chat_template"] = json!([
            {"name":"default", "template":"{{ messages[0].content }}"},
            {"name":"tool_use", "template":"tools: {{ messages[0].content }}"},
            {"name":"broken", "template":"{{ raise_exception('bad roles') }}"},
            {"name":"missing", "template":"{{ unknown_required_argument }}"}
        ])
    });
    let messages = [TemplateValue::from_serialize(
        json!({"role":"user", "content":"hello"}),
    )];
    let model = temp.load();
    assert_eq!(
        model
            .render_chat(&messages, true, None, &BTreeMap::new(), 1000)
            .unwrap(),
        "hello"
    );
    let variables = serde_json::from_str(r#"{"tools":[]}"#).unwrap();
    assert_eq!(
        model
            .render_chat(&messages, true, None, &variables, 1000)
            .unwrap(),
        "tools: hello"
    );
    for name in ["broken", "missing", "absent"] {
        assert!(
            model
                .render_chat(&messages, true, Some(name), &BTreeMap::new(), 1000)
                .is_err()
        );
    }
    assert!(
        model
            .render_chat(&messages, true, None, &BTreeMap::new(), 1)
            .is_err()
    );
    let overrides = serde_json::from_str(r#"{"messages":[]}"#).unwrap();
    assert!(
        model
            .render_chat(&messages, true, None, &overrides, 1000)
            .is_err()
    );
    std::fs::write(temp.0.join("chat_template.jinja"), "standalone").unwrap();
    assert_eq!(
        temp.load()
            .render_chat(&messages, true, None, &BTreeMap::new(), 1000)
            .unwrap(),
        "standalone"
    );
    std::fs::remove_file(temp.0.join("tokenizer.json")).unwrap();
    assert!(temp.error().contains("missing tokenizer.json"));
}

#[test]
fn malformed_settings_and_vocabulary_mismatches_fail() {
    for (key, value) in [
        ("temperature", json!(-1)),
        ("top_p", json!(1.1)),
        ("top_k", json!(-1)),
        ("eos_token_id", json!([900000])),
        ("stop_strings", json!([""])),
    ] {
        let temp = Temp::new();
        temp.edit("generation_config.json", |v| v[key] = value);
        assert!(!temp.error().is_empty());
    }
    let temp = Temp::new();
    assert!(temp.load().decode(&[900000], true).is_err());
    temp.edit("config.json", |v| v["vocab_size"] = json!(2));
    assert!(temp.error().contains("outside the vocabulary"));
    let temp = Temp::new();
    temp.edit("tokenizer_config.json", |v| {
        v["added_tokens_decoder"]["2"]["special"] = json!(false)
    });
    assert!(temp.error().contains("added token 2 differs"));
    let temp = Temp::new();
    temp.edit("tokenizer_config.json", |v| {
        v["add_bos_token"] = json!(false)
    });
    assert!(temp.error().contains("BOS/EOS flags differ"));
    temp.edit("tokenizer_config.json", |v| {
        v["add_bos_token"] = json!(true)
    });
    temp.load();
}

#[test]
fn context_gate_and_eos_match_lean_vectors() {
    let temp = Temp::new();
    let unrestricted = temp.load();
    for line in include_str!("fixtures/model_text_contract.txt").lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        match fields[0] {
            "context" => {
                let maximum: usize = fields[1].parse().unwrap();
                let count: usize = fields[2].parse().unwrap();
                temp.edit("config.json", |v| {
                    v["max_position_embeddings"] = json!(maximum)
                });
                let text = "a".repeat(count);
                let original = unrestricted.encode(&text, false).unwrap();
                assert_eq!(original.len(), count);
                let result = temp.load().encode(&text, false);
                assert_eq!(result.is_ok(), fields[3] == "true", "{line}");
                if let Ok(tokens) = result {
                    assert_eq!(tokens, original);
                }
            }
            "eos" => assert_eq!(
                unrestricted
                    .generation_defaults()
                    .is_eos(fields[1].parse().unwrap()),
                fields[2] == "true",
                "{line}"
            ),
            _ => panic!("unknown fixture line: {line}"),
        }
    }
}

#[test]
fn unsupported_json_formatting_and_nontext_content_are_errors() {
    let temp = Temp::new();
    let messages = [TemplateValue::from_serialize(
        json!({"role":"user", "content":"hello"}),
    )];
    for source in [
        "{{ 0.00001 | tojson }}",
        "{{ tools | tojson(sort_keys=true) }}",
        "{{ tools | tojson(ensure_ascii=true) }}",
        "{{ tools | tojson(separators=[',', ':']) }}",
    ] {
        std::fs::write(temp.0.join("chat_template.jinja"), source).unwrap();
        assert!(
            temp.load()
                .render_chat(&messages, true, None, &BTreeMap::new(), 1000)
                .is_err()
        );
    }
    let multimodal = [TemplateValue::from_serialize(
        json!({"role":"user", "content":[{"type":"image"}]}),
    )];
    assert!(
        temp.load()
            .render_chat(&multimodal, true, None, &BTreeMap::new(), 1000)
            .is_err()
    );
}

/// Generate reference.json with scripts/model_text_fixtures.py --model DIR --output DIR.
#[test]
#[ignore = "requires a downloaded checkpoint's tokenizer assets and Transformers reference"]
fn real_checkpoint_matches_transformers() {
    let directory = PathBuf::from(std::env::var("KADMIUM_MODEL_TEXT_DIR").unwrap());
    compare(
        &ModelText::from_model_dir(&directory).unwrap(),
        &directory.join("reference.json"),
    );
}
