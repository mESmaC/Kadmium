use super::*;
use serde_json::Value;
use std::collections::BTreeMap;

pub(super) fn inspect(
    dir: &Path,
    files: &[PackageFile],
) -> Result<(String, Option<String>, u64, Vec<CheckpointTensor>)> {
    let safe: Vec<_> = files
        .iter()
        .filter(|f| f.name.ends_with(".safetensors"))
        .collect();
    let gguf: Vec<_> = files.iter().filter(|f| f.name.ends_with(".gguf")).collect();
    let mut tensors = Vec::new();
    let architecture;
    let format;
    let context_length;
    if !safe.is_empty() && gguf.is_empty() {
        let config: Value = read_json(&dir.join("config.json"))
            .map_err(|e| format!("A SafeTensors model requires config.json: {e}"))?;
        architecture = config["model_type"].as_str().map(String::from);
        context_length = config["max_position_embeddings"].as_u64().unwrap_or(0);
        if architecture.is_none() {
            return Err("config.json must identify model_type".into());
        }
        if !dir.join("tokenizer.json").is_file() && !dir.join("tokenizer.model").is_file() {
            return Err("SafeTensors model requires tokenizer.json or tokenizer.model".into());
        }
        let parsed = crate::SafetensorsManifest::open(dir).map_err(io_error)?;
        let supplied: BTreeSet<_> = safe.iter().map(|f| f.name.as_str()).collect();
        let referenced: BTreeSet<_> = parsed.shards().iter().map(|s| s.name.as_str()).collect();
        if supplied != referenced {
            return Err("Directory contains SafeTensors files outside the model index".into());
        }
        for t in parsed.tensors() {
            let (encoding, _) = safetensors_encoding(t.dtype.as_str())?;
            tensors.push(tensor(
                t.name.clone(),
                t.dtype.as_str().into(),
                t.shape.clone(),
                encoding,
                &t.shard_name,
                t.absolute_offset,
                t.byte_len,
            ));
        }
        format = "safetensors".into();
    } else if !gguf.is_empty() && safe.is_empty() {
        let count = gguf.len() as u64;
        let mut architectures = BTreeSet::new();
        let mut splits = BTreeSet::new();
        let mut expected_tensors = None;
        let mut contexts = BTreeSet::new();
        let mut has_tokenizer = false;
        for file in gguf {
            let (mut parsed, metadata) = read_gguf(dir, file)?;
            if let Some(arch) = metadata.get("general.architecture").and_then(Value::as_str) {
                architectures.insert(arch.to_string());
                if let Some(context) = metadata
                    .get(&format!("{arch}.context_length"))
                    .and_then(Value::as_u64)
                {
                    contexts.insert(context);
                }
            }
            has_tokenizer |= metadata.contains_key("tokenizer.ggml.model");
            let declared_count = metadata
                .get("split.count")
                .and_then(Value::as_u64)
                .unwrap_or(1);
            if declared_count != count {
                return Err(format!(
                    "GGUF requires {declared_count} shards; select the directory containing all shards"
                ));
            }
            if count > 1 {
                let split = metadata
                    .get("split.no")
                    .and_then(Value::as_u64)
                    .ok_or("GGUF shard missing split.no")?;
                if split >= count || !splits.insert(split) {
                    return Err("Duplicate or invalid GGUF split number".into());
                }
                let total = metadata
                    .get("split.tensors.count")
                    .and_then(Value::as_u64)
                    .ok_or("GGUF shard missing split.tensors.count")?;
                if expected_tensors.is_some_and(|previous| previous != total) {
                    return Err("GGUF shards disagree on tensor count".into());
                }
                expected_tensors = Some(total);
            }
            tensors.append(&mut parsed);
        }
        if architectures.len() != 1
            || contexts.len() > 1
            || !has_tokenizer
            || expected_tensors.is_some_and(|n| n != tensors.len() as u64)
        {
            return Err("GGUF shards do not describe one complete architecture".into());
        }
        architecture = architectures.into_iter().next();
        context_length = contexts.into_iter().next().unwrap_or(0);
        format = "gguf".into();
    } else {
        return Err(
            "Select exactly one checkpoint format: GGUF or a SafeTensors model directory".into(),
        );
    }
    unique_tensors(&tensors)?;
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((format, architecture, context_length, tensors))
}

fn tensor(
    name: String,
    dtype: String,
    shape: Vec<u64>,
    encoding: TensorEncoding,
    source: &str,
    offset: u64,
    bytes: u64,
) -> CheckpointTensor {
    CheckpointTensor {
        name,
        dtype,
        shape,
        encoding,
        source: source.into(),
        source_offset: offset,
        bytes,
        bank: String::new(),
        bank_offset: 0,
    }
}

pub(crate) fn safetensors_encoding(dtype: &str) -> Result<(TensorEncoding, u64)> {
    Ok(match dtype {
        "F32" => (TensorEncoding::Fp32, 4),
        "F16" => (TensorEncoding::Fp16, 2),
        "BF16" => (TensorEncoding::Bf16, 2),
        "F8_E4M3" => (TensorEncoding::Fp8E4M3, 1),
        "I8" => (TensorEncoding::Int8, 1),
        "U8" => (TensorEncoding::Uint8, 1),
        _ => {
            return Err(format!(
                "Unsupported SafeTensors dtype {dtype}; no precision conversion was performed"
            ));
        }
    })
}

fn u32_le(file: &mut File) -> Result<u32> {
    let mut b = [0; 4];
    file.read_exact(&mut b).map_err(io_error)?;
    Ok(u32::from_le_bytes(b))
}
fn u64_le(file: &mut File) -> Result<u64> {
    let mut b = [0; 8];
    file.read_exact(&mut b).map_err(io_error)?;
    Ok(u64::from_le_bytes(b))
}
fn string(file: &mut File) -> Result<String> {
    let len = u64_le(file)?;
    if len > MAX_MANIFEST {
        return Err("GGUF string exceeds metadata budget".into());
    }
    let mut bytes = vec![0; len as usize];
    file.read_exact(&mut bytes).map_err(io_error)?;
    String::from_utf8(bytes).map_err(io_error)
}

// Read selected scalar metadata and skip tokenizer arrays without materializing them.
fn value(file: &mut File, kind: u32, depth: u32, budget: &mut u64) -> Result<Value> {
    if depth > 2 || *budget == 0 {
        return Err("GGUF metadata exceeds bounds".into());
    }
    *budget -= 1;
    let fixed = match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        8 | 9 => 0,
        _ => return Err(format!("Unknown GGUF metadata type {kind}")),
    };
    if fixed > 0 {
        let mut bytes = [0u8; 8];
        file.read_exact(&mut bytes[..fixed]).map_err(io_error)?;
        let unsigned = u64::from_le_bytes(bytes);
        return Ok(match kind {
            1 => Value::from(bytes[0] as i8),
            3 => Value::from(i16::from_le_bytes(bytes[..2].try_into().unwrap())),
            5 => Value::from(i32::from_le_bytes(bytes[..4].try_into().unwrap())),
            11 => Value::from(i64::from_le_bytes(bytes)),
            6 => Value::from(f32::from_le_bytes(bytes[..4].try_into().unwrap())),
            12 => Value::from(f64::from_le_bytes(bytes)),
            7 => Value::from(unsigned != 0),
            _ => Value::from(unsigned),
        });
    }
    if kind == 8 {
        return string(file).map(Value::from);
    }
    let item_type = u32_le(file)?;
    let count = u64_le(file)?;
    if count > *budget {
        return Err("GGUF array exceeds metadata budget".into());
    }
    for _ in 0..count {
        value(file, item_type, depth + 1, budget)?;
    }
    Ok(Value::Null)
}

fn ggml_layout(code: u32) -> Result<(u64, u64, TensorEncoding)> {
    let (block, bytes) = match code {
        0 => (1, 4),
        1 | 30 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        _ => {
            return Err(format!(
                "Unsupported GGML tensor type {code}; no replacement format was selected"
            ));
        }
    };
    let encoding = match code {
        0 => TensorEncoding::Fp32,
        1 => TensorEncoding::Fp16,
        30 => TensorEncoding::Bf16,
        _ => TensorEncoding::Ggml(code),
    };
    Ok((block, bytes, encoding))
}

fn read_gguf(
    dir: &Path,
    source: &PackageFile,
) -> Result<(Vec<CheckpointTensor>, BTreeMap<String, Value>)> {
    let mut file = File::open(dir.join(&source.name)).map_err(io_error)?;
    let mut magic = [0; 4];
    file.read_exact(&mut magic).map_err(io_error)?;
    if &magic != b"GGUF" || u32_le(&mut file)? != 3 {
        return Err("Expected GGUF version 3".into());
    }
    let count = u64_le(&mut file)?;
    let kv_count = u64_le(&mut file)?;
    if count == 0 || count > 100_000 || kv_count > 100_000 {
        return Err("GGUF descriptor counts exceed bounds".into());
    }
    let mut metadata = BTreeMap::new();
    let mut budget = 2_000_000;
    for _ in 0..kv_count {
        let key = string(&mut file)?;
        let kind = u32_le(&mut file)?;
        let val = value(&mut file, kind, 0, &mut budget)?;
        if metadata.insert(key, val).is_some() {
            return Err("Duplicate GGUF metadata key".into());
        }
        if file.stream_position().map_err(io_error)? > MAX_MANIFEST {
            return Err("GGUF metadata exceeds 64 MiB".into());
        }
    }
    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() || alignment > PAGE {
        return Err("Invalid GGUF alignment".into());
    }
    let mut tensors = Vec::new();
    for _ in 0..count {
        let name = string(&mut file)?;
        let dimensions = u32_le(&mut file)?;
        if dimensions == 0 || dimensions > 4 {
            return Err("GGUF tensor must have 1 to 4 dimensions".into());
        }
        let mut shape = Vec::new();
        for _ in 0..dimensions {
            shape.push(u64_le(&mut file)?);
        }
        let code = u32_le(&mut file)?;
        let offset = u64_le(&mut file)?;
        let (block, block_bytes, encoding) = ggml_layout(code)?;
        if shape[0] == 0 || shape[0] % block != 0 || offset % alignment != 0 {
            return Err(format!("Invalid GGUF block shape or alignment for {name}"));
        }
        let elements = shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or("GGUF tensor shape overflow")?;
        let bytes = (elements / block)
            .checked_mul(block_bytes)
            .ok_or("GGUF tensor size overflow")?;
        if bytes == 0 {
            return Err("Empty GGUF tensor".into());
        }
        tensors.push(tensor(
            name,
            format!("GGML_{code}"),
            shape,
            encoding,
            &source.name,
            offset,
            bytes,
        ));
    }
    let pos = file.stream_position().map_err(io_error)?;
    let base = pos
        .checked_add(alignment - 1)
        .ok_or("GGUF offset overflow")?
        / alignment
        * alignment;
    let mut ranges = Vec::new();
    for t in &mut tensors {
        t.source_offset = t
            .source_offset
            .checked_add(base)
            .ok_or("GGUF offset overflow")?;
        let end = t
            .source_offset
            .checked_add(t.bytes)
            .ok_or("GGUF offset overflow")?;
        if end > source.bytes {
            return Err(format!("Truncated GGUF tensor {}", t.name));
        }
        ranges.push((t.source_offset, end));
    }
    ranges.sort();
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err("Overlapping GGUF tensor payloads".into());
        }
    }
    Ok((tensors, metadata))
}
