use super::TemplateValue;
use super::{METADATA_BYTES, ModelTextError, Result, invalid, read_optional};
use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

mod json;

pub(super) fn load(dir: &Path, config: &Map<String, Value>) -> Result<Map<String, Value>> {
    let mut templates = Map::new();
    match config.get("chat_template") {
        None | Some(Value::Null) => (),
        Some(Value::String(template)) => {
            templates.insert("default".into(), Value::String(template.clone()));
        }
        Some(Value::Array(entries)) => {
            for entry in entries {
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("chat_template entry requires a name"))?;
                let value = entry
                    .get("template")
                    .filter(|v| v.is_string())
                    .ok_or_else(|| invalid("chat_template entry requires a template string"))?;
                if templates.insert(name.into(), value.clone()).is_some() {
                    return Err(invalid(format!("duplicate chat template `{name}`")));
                }
            }
        }
        Some(Value::Object(entries)) => templates = entries.clone(),
        Some(_) => return Err(invalid("chat_template must be a string or named templates")),
    }
    // Transformers gives the standalone template precedence over tokenizer_config.json.
    if let Some(bytes) = read_optional(&dir.join("chat_template.jinja"), METADATA_BYTES)? {
        templates.insert(
            "default".into(),
            Value::String(
                String::from_utf8(bytes)
                    .map_err(|e| invalid(format!("chat_template.jinja: {e}")))?,
            ),
        );
    }
    for (name, value) in &templates {
        if !value.is_string() {
            return Err(invalid(format!("chat template `{name}` must be a string")));
        }
    }
    Ok(templates)
}

pub(super) fn render(
    templates: &Map<String, Value>,
    config: &Map<String, Value>,
    messages: &[TemplateValue],
    add_generation_prompt: bool,
    name: Option<&str>,
    variables: &BTreeMap<String, TemplateValue>,
    fuel: u64,
) -> Result<String> {
    if fuel == 0 {
        return Err(invalid("template execution fuel must be positive"));
    }
    if messages.is_empty() {
        return Err(invalid("a chat requires at least one message"));
    }
    for message in messages {
        if message
            .get_attr("role")
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .is_none()
        {
            return Err(invalid("each chat message requires a string role"));
        }
        if message
            .get_attr("content")
            .ok()
            .is_some_and(|v| !v.is_none() && !v.is_undefined() && v.as_str().is_none())
        {
            return Err(invalid(
                "the native text adapter does not support multimodal message content",
            ));
        }
    }
    let name = name.unwrap_or_else(|| {
        if variables.get("tools").is_some_and(|v| !v.is_none())
            && templates.contains_key("tool_use")
        {
            "tool_use"
        } else {
            "default"
        }
    });
    let source = templates.get(name).and_then(Value::as_str)
        .ok_or_else(|| ModelTextError::Template(format!("checkpoint has no `{name}` chat template; select an available template or use plain text")))?;
    // Jinja's lexer normalizes template-source newlines, including files exported on Windows.
    // Message content is kept as supplied.
    let source = source.replace("\r\n", "\n").replace('\r', "\n");
    let mut context = BTreeMap::new();
    context.insert("tools".to_owned(), TemplateValue::from(()));
    for (key, value) in config {
        if key.ends_with("_token") {
            let value = if value.is_object() {
                value
                    .get("content")
                    .cloned()
                    .ok_or_else(|| invalid(format!("special token `{key}` requires content")))?
            } else {
                value.clone()
            };
            context.insert(key.clone(), TemplateValue::from_serialize(value));
        }
    }
    for (key, value) in variables {
        if key == "messages"
            || key == "add_generation_prompt"
            || (key != "tools" && context.contains_key(key))
        {
            return Err(invalid(format!(
                "template variable `{key}` would replace conversation or checkpoint metadata"
            )));
        }
        context.insert(key.clone(), value.clone());
    }
    context.insert("messages".into(), TemplateValue::from_serialize(messages));
    context.insert(
        "add_generation_prompt".into(),
        TemplateValue::from(add_generation_prompt),
    );
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // HF templates test absent optional fields for truthiness (for example message.tool_calls).
    // Printing or iterating a missing required value still fails rather than dropping content.
    env.set_undefined_behavior(UndefinedBehavior::SemiStrict);
    env.set_fuel(Some(fuel));
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", json::tojson);
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env.add_template("chat", &source)
        .and_then(|()| env.get_template("chat"))
        .and_then(|template| template.render(context))
        // Alternate display includes referenced variables, which can contain private messages.
        .map_err(|e| ModelTextError::Template(e.to_string()))
}
