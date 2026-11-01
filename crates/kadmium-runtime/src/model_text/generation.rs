use super::{Result, invalid};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

/// Common Hugging Face sampling parameters. Missing values use Transformers GenerationConfig's
/// documented sampling defaults, not a Kadmium speed/quality preset. This does not sample logits.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingDefaults {
    pub do_sample: bool,
    pub temperature: f64,
    pub top_k: u64,
    pub top_p: f64,
    pub min_p: Option<f64>,
    pub typical_p: f64,
    pub repetition_penalty: f64,
}

/// Model-owned generation metadata. The future generator must implement or reject every active
/// setting in `declared`; these common fields alone are not permission to ignore the remainder.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationDefaults {
    pub sampling: SamplingDefaults,
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub stop_strings: Vec<String>,
    /// Exact source settings (generation_config.json, or legacy config.json when absent).
    /// Includes declared max_length/max_new_tokens; they are metadata, not an output cap.
    pub declared: Map<String, Value>,
}

impl GenerationDefaults {
    pub(super) fn load(
        config: &Map<String, Value>,
        declared: Map<String, Value>,
        tokenizer: &Tokenizer,
        vocab_size: usize,
    ) -> Result<Self> {
        let number = |key: &str, default: f64| -> Result<f64> {
            match declared.get(key) {
                None => Ok(default),
                Some(value) => value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| invalid(format!("generation `{key}` must be a finite number"))),
            }
        };
        let probability = |key: &str, default: f64| -> Result<f64> {
            let value = number(key, default)?;
            if !(0.0..=1.0).contains(&value) {
                return Err(invalid(format!(
                    "generation `{key}` must be between 0 and 1"
                )));
            }
            Ok(value)
        };
        let do_sample = match declared.get("do_sample") {
            None => false,
            Some(value) => value
                .as_bool()
                .ok_or_else(|| invalid("do_sample must be a boolean"))?,
        };
        let temperature = number("temperature", 1.0)?;
        if temperature < 0.0 || (do_sample && temperature == 0.0) {
            return Err(invalid(
                "temperature must be positive when sampling and nonnegative otherwise",
            ));
        }
        let repetition_penalty = number("repetition_penalty", 1.0)?;
        if repetition_penalty <= 0.0 {
            return Err(invalid("repetition_penalty must be positive"));
        }
        let sampling = SamplingDefaults {
            do_sample,
            temperature,
            repetition_penalty,
            top_k: match declared.get("top_k") {
                None => 50,
                Some(value) => value
                    .as_u64()
                    .ok_or_else(|| invalid("top_k must be a nonnegative integer"))?,
            },
            top_p: probability("top_p", 1.0)?,
            typical_p: probability("typical_p", 1.0)?,
            min_p: match declared.get("min_p") {
                None | Some(Value::Null) => None,
                Some(_) => Some(probability("min_p", 0.0)?),
            },
        };
        let id = |value: &Value| -> Result<u32> {
            let id = value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| invalid("special token ids must be nonnegative u32 integers"))?;
            if id as usize >= vocab_size || tokenizer.id_to_token(id).is_none() {
                return Err(invalid(format!(
                    "special token {id} is outside the model tokenizer vocabulary"
                )));
            }
            Ok(id)
        };
        let special = |key: &str| declared.get(key).or_else(|| config.get(key));
        let single = |key: &str| -> Result<Option<u32>> {
            special(key).filter(|v| !v.is_null()).map(id).transpose()
        };
        let eos_token_ids = match special("eos_token_id") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(values)) => values.iter().map(&id).collect::<Result<Vec<_>>>()?,
            Some(value) => vec![id(value)?],
        };
        let stop_strings = match declared.get("stop_strings") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(value)) => vec![value.clone()],
            Some(Value::Array(values)) => values
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid("stop_strings must contain strings"))
                })
                .collect::<Result<Vec<_>>>()?,
            Some(_) => return Err(invalid("stop_strings must be a string or array of strings")),
        };
        if stop_strings.iter().any(String::is_empty) {
            return Err(invalid("stop strings cannot be empty"));
        }
        Ok(Self {
            bos_token_id: single("bos_token_id")?,
            pad_token_id: single("pad_token_id")?,
            sampling,
            eos_token_ids,
            stop_strings,
            declared,
        })
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token_ids.contains(&token)
    }

    /// Byte offset of the earliest literal stop in the entire decoded generated text. The caller
    /// excludes the prompt and re-decodes the accumulated output so stops can cross token edges.
    pub fn find_stop(&self, generated_text: &str) -> Option<usize> {
        self.stop_strings
            .iter()
            .filter_map(|stop| generated_text.find(stop))
            .min()
    }
}
