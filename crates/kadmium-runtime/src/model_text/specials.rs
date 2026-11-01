//! Hugging Face's legacy special-token overlay and the attributes exposed to chat templates.
use super::{Result, invalid, read_json};
use serde_json::{Map, Value};
use std::path::Path;
use tokenizers::Tokenizer;

const ATTRIBUTES: [&str; 8] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
];

/// Resolve before the BOS/EOS postprocessor check, so it uses the same metadata as the template.
pub(super) fn resolve(
    dir: &Path,
    config: &mut Map<String, Value>,
    tokenizer: &Tokenizer,
) -> Result<Map<String, Value>> {
    if !config.contains_key("added_tokens_decoder") {
        for (key, value) in read_json(dir, "special_tokens_map.json", false)? {
            if !ATTRIBUTES.contains(&key.as_str()) {
                return Err(invalid(format!(
                    "unsupported legacy special-token attribute `{key}`"
                )));
            }
            if key == "additional_special_tokens" && value.is_array() {
                let mut combined = match config.get(&key) {
                    None | Some(Value::Null) => Vec::new(),
                    Some(Value::Array(tokens)) => tokens.clone(),
                    Some(_) => return Err(invalid("additional_special_tokens must be an array")),
                };
                if let Value::Array(tokens) = value {
                    // HF compares Python AddedToken objects and strings before conversion. Refuse
                    // ambiguous mixed descriptor merges rather than deduplicating by token text.
                    if !combined.is_empty()
                        && !tokens.is_empty()
                        && combined.iter().chain(&tokens).any(Value::is_object)
                    {
                        return Err(invalid(
                            "merging legacy additional special-token descriptors is unsupported; export matching modern tokenizer assets",
                        ));
                    }
                    for token in tokens {
                        if !combined.contains(&token) {
                            combined.push(token);
                        }
                    }
                }
                config.insert(key, Value::Array(combined));
            } else {
                config.insert(key, value);
            }
        }
    }
    let mut context = Map::new();
    for key in ATTRIBUTES {
        let Some(value) = config.get(key).filter(|v| !v.is_null()) else {
            continue;
        };
        if key == "additional_special_tokens" {
            let values = value
                .as_array()
                .ok_or_else(|| invalid("additional_special_tokens must be an array"))?;
            if !values.is_empty() {
                context.insert(
                    key.into(),
                    Value::Array(
                        values
                            .iter()
                            .map(|v| token_content(v, tokenizer))
                            .collect::<Result<Vec<_>>>()?,
                    ),
                );
            }
        } else {
            let content = token_content(value, tokenizer)?;
            if content.as_str().is_some_and(|text| !text.is_empty()) {
                context.insert(key.into(), content);
            }
        }
    }
    Ok(context)
}

fn token_content(value: &Value, tokenizer: &Tokenizer) -> Result<Value> {
    let text = value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
        .ok_or_else(|| {
            invalid("special tokens require a string or a descriptor with string content")
        })?;
    // Transformers may register missing special tokens. The native decoder cannot grow its
    // vocabulary; require a matching exported tokenizer instead of changing token ids here.
    if !text.is_empty() {
        let token = tokenizer
            .token_to_id(text)
            .and_then(|id| tokenizer.get_added_tokens_decoder().get(&id).cloned());
        if !token.is_some_and(|token| token.special) {
            return Err(invalid(
                "resolved special token is not registered as special in tokenizer.json; export matching tokenizer assets",
            ));
        }
    }
    Ok(Value::String(text.into()))
}
