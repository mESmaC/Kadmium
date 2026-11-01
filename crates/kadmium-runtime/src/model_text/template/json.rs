//! Transformers replaces Jinja's HTML-safe JSON filter with json.dumps(ensure_ascii=False).
use minijinja::{Error, ErrorKind, Value, value::Kwargs};
use serde::Serialize;
use std::io::{self, Write};

struct PythonCompact;
impl serde_json::ser::Formatter for PythonCompact {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

pub(super) fn tojson(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    // Rust and Python do not share all floating-point JSON spellings. Until that formatter has
    // differential coverage, refuse this input instead of changing the model's tool prompt.
    let check = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    fn has_float(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Number(n) => n.is_f64(),
            serde_json::Value::Array(values) => values.iter().any(has_float),
            serde_json::Value::Object(values) => values.values().any(has_float),
            _ => false,
        }
    }
    if has_float(&check) {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson floating-point values require a Python-compatible formatter that is not implemented",
        ));
    }
    // Fail explicitly for options not implemented here rather than rendering different bytes.
    let indent: Option<usize> = kwargs.get("indent")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    if ensure_ascii == Some(true) || sort_keys == Some(true) {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson ensure_ascii=true and sort_keys=true are not supported",
        ));
    }
    let mut output = Vec::new();
    if let Some(indent) = indent {
        // A parser resource bound; rejecting an unsupported formatter never changes the prompt.
        if indent > 16 {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "tojson indent above 16 is not supported",
            ));
        }
        let spaces = vec![b' '; indent];
        let mut serializer = serde_json::Serializer::with_formatter(
            &mut output,
            serde_json::ser::PrettyFormatter::with_indent(&spaces),
        );
        value.serialize(&mut serializer)
    } else {
        value.serialize(&mut serde_json::Serializer::with_formatter(
            &mut output,
            PythonCompact,
        ))
    }
    .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    String::from_utf8(output).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))
}
