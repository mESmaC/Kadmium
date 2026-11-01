//! Checkpoint-owned text processing for the native decoder. No generation loop lives here.
//!
//! Tokenization uses the checkpoint's `tokenizer.json`; chat uses its Jinja template. Runtime
//! loading is local and entirely Rust. Missing assets and unsupported template operations fail.
//! Sampling metadata is retained for the future generator, including unrecognised fields.

mod generation;
mod specials;
mod template;

pub use generation::{GenerationDefaults, SamplingDefaults};
/// Deserialize chat input directly into this value to retain JSON object insertion order.
/// Converting through an unordered/sorted map first loses tool-schema formatting information.
pub use minijinja::Value as TemplateValue;

use crate::native_decoder::{DecoderConfig, NativeDecoderError};
use serde_json::{Map, Value};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer;

// Storage parser guards, not token/context limits. Oversized artifacts are rejected, never cut.
const TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;
const METADATA_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ModelTextError {
    #[error("model text metadata: {0}")]
    Metadata(String),
    #[error("model tokenizer: {0}")]
    Tokenizer(String),
    #[error("model chat template: {0}")]
    Template(String),
    #[error(transparent)]
    Decoder(#[from] NativeDecoderError),
}

pub type Result<T> = std::result::Result<T, ModelTextError>;

fn invalid(message: impl Into<String>) -> ModelTextError {
    ModelTextError::Metadata(message.into())
}

fn read_optional(path: &Path, limit: u64) -> Result<Option<Vec<u8>>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(invalid(format!("cannot open {}: {e}", path.display()))),
    };
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?;
    if bytes.len() as u64 > limit {
        return Err(invalid(format!(
            "{} exceeds the artifact parser limit of {limit} bytes",
            path.display()
        )));
    }
    Ok(Some(bytes))
}

fn read_json(dir: &Path, name: &str, required: bool) -> Result<Map<String, Value>> {
    match read_optional(&dir.join(name), METADATA_BYTES)? {
        Some(bytes) => serde_json::from_slice::<Map<String, Value>>(&bytes)
            .map_err(|e| invalid(format!("{name}: {e}"))),
        None if required => Err(invalid(format!("missing {name}"))),
        None => Ok(Map::new()),
    }
}

/// Text assets bound to a validated native decoder's vocabulary and context.
pub struct ModelText {
    tokenizer: Tokenizer,
    tokenizer_config: Map<String, Value>,
    special_tokens: Map<String, Value>,
    templates: Map<String, Value>,
    config: DecoderConfig,
    generation: GenerationDefaults,
}

impl ModelText {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let raw_config = read_json(dir, "config.json", true)?;
        let config = DecoderConfig::from_json(&Value::Object(raw_config.clone()))?;
        let bytes = read_optional(&dir.join("tokenizer.json"), TOKENIZER_BYTES)?
            .ok_or_else(|| invalid("missing tokenizer.json; tokenizer.model-only checkpoints need a tokenizer.json export"))?;
        let mut tokenizer =
            Tokenizer::from_bytes(&bytes).map_err(|e| ModelTextError::Tokenizer(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| ModelTextError::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(None);
        for id in tokenizer.get_vocab(true).values() {
            if *id as usize >= config.vocab_size {
                return Err(NativeDecoderError::TokenOutOfRange {
                    token: *id,
                    vocab_size: config.vocab_size,
                }
                .into());
            }
        }
        let mut tokenizer_config = read_json(dir, "tokenizer_config.json", true)?;
        let special_tokens = specials::resolve(dir, &mut tokenizer_config, &tokenizer)?;
        // These options alter the HF wrapper before it calls the tokenizer backend.
        for key in ["add_prefix_space", "split_special_tokens"] {
            if tokenizer_config
                .get(key)
                .is_some_and(|v| v != &Value::Bool(false) && !v.is_null())
            {
                return Err(invalid(format!(
                    "tokenizer_config.json `{key}` is not supported by this text adapter"
                )));
            }
        }
        validate_tokenizer_config(&tokenizer, &tokenizer_config)?;
        let templates = template::load(dir, &tokenizer_config)?;
        let declared = match read_optional(&dir.join("generation_config.json"), METADATA_BYTES)? {
            Some(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| invalid(format!("generation_config.json: {e}")))?,
            None => raw_config.clone(),
        };
        let generation =
            GenerationDefaults::load(&raw_config, declared, &tokenizer, config.vocab_size)?;
        Ok(Self {
            tokenizer,
            tokenizer_config,
            special_tokens,
            templates,
            config,
            generation,
        })
    }

    pub fn generation_defaults(&self) -> &GenerationDefaults {
        &self.generation
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Plain text can request the tokenizer's own postprocessor (BOS/EOS). Chat must not add
    /// special tokens again: the checkpoint template already writes them.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| ModelTextError::Tokenizer(e.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        self.validate_ids(&ids)?;
        if ids.len() > self.config.max_position_embeddings {
            return Err(NativeDecoderError::ContextExceeded {
                requested: ids.len(),
                maximum: self.config.max_position_embeddings,
            }
            .into());
        }
        Ok(ids)
    }

    /// Decode the complete token sequence, including byte-fallback sequences spanning tokens.
    /// This is not an incremental streaming decoder; callers must not emit independent tokens.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.validate_ids(ids)?;
        let text = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| ModelTextError::Tokenizer(e.to_string()))?;
        match self.tokenizer_config.get("clean_up_tokenization_spaces") {
            Some(Value::Bool(true)) => Ok(clean_up(&text)),
            Some(Value::Bool(false)) | Some(Value::Null) | None => Ok(text),
            Some(_) => Err(invalid("clean_up_tokenization_spaces must be a boolean")),
        }
    }

    fn validate_ids(&self, ids: &[u32]) -> Result<()> {
        for &id in ids {
            if id as usize >= self.config.vocab_size || self.tokenizer.id_to_token(id).is_none() {
                return Err(ModelTextError::Tokenizer(format!(
                    "token {id} has no entry in the model tokenizer (decoder vocabulary {})",
                    self.config.vocab_size
                )));
            }
        }
        Ok(())
    }

    /// Render the complete conversation. `variables` carries model-specific arguments such as
    /// `tools`, `enable_thinking`, or an explicit date. `fuel` is the caller's template execution
    /// budget; exhaustion is an error, never a partially rendered prompt.
    pub fn render_chat(
        &self,
        messages: &[TemplateValue],
        add_generation_prompt: bool,
        template_name: Option<&str>,
        variables: &std::collections::BTreeMap<String, TemplateValue>,
        fuel: u64,
    ) -> Result<String> {
        template::render(
            &self.templates,
            &self.special_tokens,
            messages,
            add_generation_prompt,
            template_name,
            variables,
            fuel,
        )
    }

    pub fn encode_chat(
        &self,
        messages: &[TemplateValue],
        add_generation_prompt: bool,
        template_name: Option<&str>,
        variables: &std::collections::BTreeMap<String, TemplateValue>,
        fuel: u64,
    ) -> Result<Vec<u32>> {
        let prompt = self.render_chat(
            messages,
            add_generation_prompt,
            template_name,
            variables,
            fuel,
        )?;
        self.encode(&prompt, false)
    }
}

fn validate_tokenizer_config(tokenizer: &Tokenizer, config: &Map<String, Value>) -> Result<()> {
    if let Some(entries) = config.get("added_tokens_decoder") {
        let entries = entries
            .as_object()
            .ok_or_else(|| invalid("added_tokens_decoder must be an object"))?;
        let actual = tokenizer.get_added_tokens_decoder();
        for (id, value) in entries {
            let id: u32 = id
                .parse()
                .map_err(|_| invalid("added token id must be a u32 integer"))?;
            let expected: tokenizers::AddedToken = serde_json::from_value(value.clone())
                .map_err(|e| invalid(format!("added token {id}: {e}")))?;
            if actual.get(&id) != Some(&expected) {
                return Err(invalid(format!(
                    "added token {id} differs between tokenizer_config.json and tokenizer.json; export matching tokenizer assets"
                )));
            }
        }
    }
    if ["add_bos_token", "add_eos_token"]
        .iter()
        .any(|key| config.contains_key(*key))
    {
        let raw = tokenizer
            .encode("a", false)
            .map_err(|e| ModelTextError::Tokenizer(e.to_string()))?;
        let mut expected = Vec::new();
        for (flag, token, content) in [
            ("add_bos_token", "bos_token", &[][..]),
            ("add_eos_token", "eos_token", raw.get_ids()),
        ] {
            expected.extend_from_slice(content);
            let enabled = match config.get(flag) {
                None => false,
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| invalid(format!("{flag} must be a boolean")))?,
            };
            if enabled {
                let token = config
                    .get(token)
                    .and_then(|v| {
                        v.as_str()
                            .or_else(|| v.get("content").and_then(Value::as_str))
                    })
                    .and_then(|text| tokenizer.token_to_id(text))
                    .ok_or_else(|| invalid(format!("{flag} requires a matching {token}")))?;
                expected.push(token);
            }
        }
        let processed = tokenizer
            .encode("a", true)
            .map_err(|e| ModelTextError::Tokenizer(e.to_string()))?;
        if processed.get_ids() != expected {
            return Err(invalid(
                "BOS/EOS flags differ from the tokenizer.json postprocessor; export matching tokenizer assets",
            ));
        }
    }
    Ok(())
}

// Hugging Face PreTrainedTokenizerBase.clean_up_tokenization, in the same replacement order.
fn clean_up(text: &str) -> String {
    let mut text = text.to_owned();
    for (from, to) in [
        (" .", "."),
        (" ?", "?"),
        (" !", "!"),
        (" ,", ","),
        (" ' ", "'"),
        (" n't", "n't"),
        (" 'm", "'m"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" 're", "'re"),
    ] {
        text = text.replace(from, to);
    }
    text
}
