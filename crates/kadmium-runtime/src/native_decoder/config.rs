//! Hugging Face `config.json` for the dense decoder families the native runtime executes.
//!
//! A field the decoder cannot honour exactly is an error, never an approximation: sliding-window
//! attention, a RoPE scaling type other than default or `llama3`, an activation other than SiLU,
//! and a `quantization_config` (a checkpoint quantized before import) are all refused with the
//! reason. Defaults are the format's own documented defaults (`num_key_value_heads` equals
//! `num_attention_heads`, `rope_theta` 10000, untied embeddings for these families), not
//! limits chosen here.

use super::math::MAX_ANGLE;
use super::{NativeDecoderError, Result};
use serde_json::Value;
use std::path::Path;

/// `config.json` is read whole; a file larger than this is not a model config.
const CONFIG_FILE_LIMIT: u64 = 4 * 1024 * 1024;

/// The decoder families this runtime executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// `LlamaForCausalLM`: Llama 2, 3, 3.1, 3.2 and Llama-architecture derivatives.
    Llama,
    /// `MistralForCausalLM` without sliding-window attention.
    Mistral,
    /// `Qwen2ForCausalLM`: Qwen2 and Qwen2.5, bias on the q, k and v projections.
    Qwen2,
    /// `Qwen3ForCausalLM`: dense Qwen3, RMSNorm on each query and key head.
    Qwen3,
}

impl Architecture {
    pub fn hf_name(self) -> &'static str {
        match self {
            Self::Llama => "LlamaForCausalLM",
            Self::Mistral => "MistralForCausalLM",
            Self::Qwen2 => "Qwen2ForCausalLM",
            Self::Qwen3 => "Qwen3ForCausalLM",
        }
    }
}

/// How the RoPE inverse frequencies are derived from `rope_theta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// `theta^(-2i/d)`.
    Default,
    /// Llama 3.1's frequency-dependent rescaling.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: f64,
    },
}

/// A validated decoder configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderConfig {
    pub architecture: Architecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    /// The model's native context length: the decoder serves every position below it.
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub qkv_bias: bool,
    pub o_bias: bool,
    pub mlp_bias: bool,
    pub qk_norm: bool,
}

fn unsupported(feature: impl Into<String>) -> NativeDecoderError {
    NativeDecoderError::Unsupported(feature.into())
}

fn invalid(message: impl Into<String>) -> NativeDecoderError {
    NativeDecoderError::Config(message.into())
}

fn positive(json: &Value, key: &str) -> Result<usize> {
    json.get(key)
        .and_then(Value::as_u64)
        .filter(|&n| n > 0)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid(format!("`{key}` must be a positive integer")))
}

fn optional_positive(json: &Value, key: &str) -> Result<Option<usize>> {
    match json.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => positive(json, key).map(Some),
    }
}

fn number(json: &Value, key: &str) -> Result<f64> {
    json.get(key)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .ok_or_else(|| invalid(format!("`{key}` must be a finite number")))
}

fn flag(json: &Value, key: &str, default: bool) -> Result<bool> {
    match json.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(invalid(format!("`{key}` must be true or false"))),
    }
}

impl DecoderConfig {
    /// Reads `config.json` from a model directory.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let length = std::fs::metadata(&path)
            .map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?
            .len();
        if length > CONFIG_FILE_LIMIT {
            return Err(invalid(format!(
                "{} is {length} bytes, more than a model config can be ({CONFIG_FILE_LIMIT})",
                path.display()
            )));
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let json: Value =
            serde_json::from_str(text).map_err(|e| invalid(format!("not valid JSON: {e}")))?;
        Self::from_json(&json)
    }

    pub fn from_json(json: &Value) -> Result<Self> {
        if json
            .get("quantization_config")
            .is_some_and(|v| !v.is_null())
        {
            return Err(unsupported(
                "a checkpoint with `quantization_config`: it was quantized before import, and the native decoder runs 16-bit or wider weights only",
            ));
        }
        let name = json
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|list| list.first())
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("`architectures` must name the model class"))?;
        let architecture = match name {
            "LlamaForCausalLM" => Architecture::Llama,
            "MistralForCausalLM" => Architecture::Mistral,
            "Qwen2ForCausalLM" => Architecture::Qwen2,
            "Qwen3ForCausalLM" => Architecture::Qwen3,
            other => return Err(unsupported(format!("the `{other}` architecture"))),
        };
        if let Some(act) = json.get("hidden_act").filter(|v| !v.is_null())
            && act.as_str() != Some("silu")
        {
            return Err(unsupported(format!(
                "the activation {act}; only SiLU is implemented"
            )));
        }

        let vocab_size = positive(json, "vocab_size")?;
        let hidden_size = positive(json, "hidden_size")?;
        let intermediate_size = positive(json, "intermediate_size")?;
        let num_layers = positive(json, "num_hidden_layers")?;
        let num_heads = positive(json, "num_attention_heads")?;
        let num_kv_heads = optional_positive(json, "num_key_value_heads")?.unwrap_or(num_heads);
        let head_dim = match optional_positive(json, "head_dim")? {
            Some(dim) => dim,
            None if hidden_size % num_heads == 0 => hidden_size / num_heads,
            None => {
                return Err(invalid(format!(
                    "`hidden_size` {hidden_size} is not divisible by `num_attention_heads` {num_heads}"
                )));
            }
        };
        if num_heads % num_kv_heads != 0 {
            return Err(invalid(format!(
                "`num_attention_heads` {num_heads} is not a multiple of `num_key_value_heads` {num_kv_heads}"
            )));
        }
        if head_dim % 2 != 0 {
            return Err(invalid(format!(
                "head dimension {head_dim} must be even for RoPE"
            )));
        }
        for (what, a, b) in [
            ("query width", num_heads, head_dim),
            ("embedding size", vocab_size, hidden_size),
            ("MLP size", intermediate_size, hidden_size),
        ] {
            if a.checked_mul(b).is_none() {
                return Err(invalid(format!("{what} {a} x {b} overflows")));
            }
        }
        let rms_norm_eps = number(json, "rms_norm_eps")?;
        if rms_norm_eps <= 0.0 {
            return Err(invalid("`rms_norm_eps` must be positive"));
        }
        let max_position_embeddings = positive(json, "max_position_embeddings")?;
        if max_position_embeddings as f64 > MAX_ANGLE {
            return Err(unsupported(format!(
                "a native context of {max_position_embeddings} positions; RoPE angles are exact up to {} positions",
                MAX_ANGLE as u64
            )));
        }

        if let Some(types) = json.get("layer_types").and_then(Value::as_array)
            && types.iter().any(|t| t.as_str() != Some("full_attention"))
        {
            return Err(unsupported(
                "sliding-window attention layers (`layer_types`)",
            ));
        }
        match architecture {
            Architecture::Mistral => {
                if let Some(window) = optional_positive(json, "sliding_window")?
                    && window < max_position_embeddings
                {
                    return Err(unsupported(format!(
                        "sliding-window attention ({window} of {max_position_embeddings} positions)"
                    )));
                }
            }
            Architecture::Qwen2 | Architecture::Qwen3 => {
                if flag(json, "use_sliding_window", false)? {
                    return Err(unsupported(
                        "sliding-window attention (`use_sliding_window`)",
                    ));
                }
            }
            Architecture::Llama => {}
        }

        let (rope_theta, rope_scaling) = rope(json)?;

        let attention_bias = flag(json, "attention_bias", false)?;
        let (qkv_bias, o_bias, mlp_bias, qk_norm) = match architecture {
            Architecture::Llama => (
                attention_bias,
                attention_bias,
                flag(json, "mlp_bias", false)?,
                false,
            ),
            Architecture::Mistral => (false, false, false, false),
            Architecture::Qwen2 => (true, false, false, false),
            Architecture::Qwen3 => (attention_bias, attention_bias, false, true),
        };

        Ok(Self {
            architecture,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            rope_scaling,
            max_position_embeddings,
            tie_word_embeddings: flag(json, "tie_word_embeddings", false)?,
            qkv_bias,
            o_bias,
            mlp_bias,
            qk_norm,
        })
    }

    pub fn q_width(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// Reads RoPE settings from transformers 5's `rope_parameters` or the older top-level
/// `rope_theta` and `rope_scaling`.
fn rope(json: &Value) -> Result<(f64, RopeScaling)> {
    let (theta, scaling) = match json.get("rope_parameters").filter(|v| !v.is_null()) {
        Some(parameters) => {
            if !parameters.is_object() {
                return Err(invalid("`rope_parameters` must be an object"));
            }
            (number(parameters, "rope_theta")?, Some(parameters))
        }
        None => {
            let theta = match json.get("rope_theta") {
                None | Some(Value::Null) => 10_000.0,
                Some(_) => number(json, "rope_theta")?,
            };
            (theta, json.get("rope_scaling").filter(|v| !v.is_null()))
        }
    };
    if theta <= 1.0 {
        return Err(invalid(format!(
            "`rope_theta` {theta} must be greater than 1"
        )));
    }
    let Some(scaling) = scaling else {
        return Ok((theta, RopeScaling::Default));
    };
    let kind = scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    match kind {
        "default" => Ok((theta, RopeScaling::Default)),
        "llama3" => {
            let factor = number(scaling, "factor")?;
            let low_freq_factor = number(scaling, "low_freq_factor")?;
            let high_freq_factor = number(scaling, "high_freq_factor")?;
            let original = number(scaling, "original_max_position_embeddings")?;
            if factor <= 0.0
                || low_freq_factor <= 0.0
                || high_freq_factor <= low_freq_factor
                || original <= 0.0
            {
                return Err(invalid("`llama3` RoPE scaling parameters are out of range"));
            }
            Ok((
                theta,
                RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings: original,
                },
            ))
        }
        other => Err(unsupported(format!("`{other}` RoPE scaling"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn llama() -> Value {
        json!({
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 128256, "hidden_size": 3072, "intermediate_size": 8192,
            "num_hidden_layers": 28, "num_attention_heads": 24, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5, "max_position_embeddings": 131072, "rope_theta": 500000.0,
            "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                             "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
            "tie_word_embeddings": true, "hidden_act": "silu"
        })
    }

    #[test]
    fn reads_a_legacy_llama_3_2_config() {
        let config = DecoderConfig::from_json(&llama()).unwrap();
        assert_eq!(config.architecture, Architecture::Llama);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.max_position_embeddings, 131072);
        assert!(config.tie_word_embeddings);
        assert!(
            matches!(config.rope_scaling, RopeScaling::Llama3 { factor, .. } if factor == 32.0)
        );
        assert_eq!(config.rope_theta, 500000.0);
    }

    #[test]
    fn refuses_what_it_cannot_serve_exactly() {
        let cases: Vec<(&str, Value)> = vec![
            (
                "yarn",
                json!({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}),
            ),
            (
                "quantization",
                json!({"quantization_config": {"quant_method": "awq"}}),
            ),
            ("activation", json!({"hidden_act": "gelu"})),
            (
                "architecture",
                json!({"architectures": ["GPT2LMHeadModel"]}),
            ),
            ("context", json!({"max_position_embeddings": 2_000_000})),
            ("layer types", json!({"layer_types": ["sliding_attention"]})),
        ];
        for (what, patch) in cases {
            let mut config = llama();
            for (k, v) in patch.as_object().unwrap() {
                config[k] = v.clone();
            }
            let error = DecoderConfig::from_json(&config).unwrap_err();
            assert!(
                matches!(error, NativeDecoderError::Unsupported(_)),
                "{what}: {error}"
            );
        }
        let mistral = json!({
            "architectures": ["MistralForCausalLM"], "vocab_size": 32000, "hidden_size": 4096,
            "intermediate_size": 14336, "num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 8, "rms_norm_eps": 1e-5, "max_position_embeddings": 32768,
            "sliding_window": 4096
        });
        let error = DecoderConfig::from_json(&mistral).unwrap_err().to_string();
        assert!(error.contains("4096") && error.contains("32768"), "{error}");
    }

    #[test]
    fn rejects_malformed_dimensions() {
        for (key, value) in [
            ("num_key_value_heads", json!(5)),
            ("hidden_size", json!(0)),
            ("rms_norm_eps", json!(-1.0)),
            ("vocab_size", json!("many")),
        ] {
            let mut config = llama();
            config[key] = value;
            assert!(matches!(
                DecoderConfig::from_json(&config),
                Err(NativeDecoderError::Config(_))
            ));
        }
    }
}
