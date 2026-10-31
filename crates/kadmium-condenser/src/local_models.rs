//! Explicit local checkpoint -> lossless package -> local Ollama -> Condenser handoff.
//! The backend consumes the preserved originals, not the storage-only KTB banks.

use crate::provider::decode_json;
use crate::server::{ApiError, ServerState};
use crate::{CondenserError, ProviderConfig, ProviderKind, Result};
use axum::{Json, extract::State};
use kadmium_data::checkpoint::{
    CheckpointLibrary, CheckpointManifest, CheckpointSummary, ImportOptions, ggml_type_name,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Deserialize)]
pub(crate) struct ImportRequest {
    source_path: PathBuf,
    /// Operator acceptance of a checkpoint that was quantized before import.
    #[serde(default)]
    accept_pre_quantized: bool,
}
#[derive(Deserialize)]
pub(crate) struct LoadRequest {
    checkpoint_id: String,
    base_url: String,
    context_length: Option<u32>,
}

#[derive(Serialize)]
pub struct LoadedCheckpoint {
    pub checkpoint: CheckpointSummary,
    pub provider: ProviderConfig,
    pub tensor_metadata_verified: bool,
    pub execution: &'static str,
    pub context_length: u32,
}

#[derive(Deserialize)]
pub(crate) struct BrowseRequest {
    path: Option<PathBuf>,
}

/// Host-side picker: browsers cannot expose local absolute paths through file inputs.
pub(crate) async fn browse(
    Json(request): Json<BrowseRequest>,
) -> std::result::Result<Json<Value>, ApiError> {
    let listing = tokio::task::spawn_blocking(move || -> Result<Value> {
        let mut path = request.path.filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
                .map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")));
        if path.is_file() { path.pop(); }
        let path = user_facing_path(std::fs::canonicalize(path)?);
        let mut entries = Vec::new();
        let mut truncated = false;
        // One unreadable entry must not hide the rest of the directory.
        for entry in std::fs::read_dir(&path)?.flatten() {
            let candidate = entry.path();
            // `metadata` follows symlinks, so linked model folders stay browsable.
            let Ok(metadata) = std::fs::metadata(&candidate) else { continue };
            if metadata.is_dir() || (metadata.is_file() && candidate.extension().is_some_and(|ext| ext == "gguf" || ext == "safetensors")) {
                if entries.len() == 500 { truncated = true; break; }
                entries.push(json!({"name":entry.file_name().to_string_lossy(), "path":candidate, "directory":metadata.is_dir()}));
            }
        }
        entries.sort_by_key(|entry| (!entry["directory"].as_bool().unwrap_or(false), entry["name"].as_str().unwrap_or("").to_lowercase()));
        Ok(json!({"path":path,"parent":path.parent(),"entries":entries,"truncated":truncated}))
    }).await.map_err(|e| CondenserError::Config(e.to_string()))??;
    Ok(Json(listing))
}

/// Windows `canonicalize` returns verbatim `\\?\C:\...` paths; show and reuse the
/// ordinary drive or UNC form, which names the same location.
fn user_facing_path(path: PathBuf) -> PathBuf {
    let plain = {
        let text = path.to_string_lossy();
        match text.strip_prefix(r"\\?\UNC\") {
            Some(unc) => Some(format!(r"\\{unc}")),
            None => text
                .strip_prefix(r"\\?\")
                .filter(|rest| rest.as_bytes().get(1) == Some(&b':'))
                .map(str::to_owned),
        }
    };
    plain.map(PathBuf::from).unwrap_or(path)
}

pub(crate) async fn list(
    State(state): State<ServerState>,
) -> std::result::Result<Json<Value>, ApiError> {
    let library = state.model_library.clone();
    let directory = library.root().display().to_string();
    let models = tokio::task::spawn_blocking(move || library.list())
        .await
        .map_err(|e| CondenserError::Config(e.to_string()))?
        .map_err(CondenserError::Config)?;
    Ok(Json(json!({"directory":directory, "checkpoints":models})))
}

pub(crate) async fn import(
    State(state): State<ServerState>,
    Json(request): Json<ImportRequest>,
) -> std::result::Result<Json<CheckpointSummary>, ApiError> {
    let permit = state
        .model_operations
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            CondenserError::Config("A checkpoint import or load is already running".into())
        })?;
    let library = state.model_library.clone();
    let manifest = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        library.import_with(
            &request.source_path,
            ImportOptions {
                accept_pre_quantized: request.accept_pre_quantized,
            },
        )
    })
    .await
    .map_err(|e| CondenserError::Config(e.to_string()))?
    .map_err(CondenserError::Config)?;
    Ok(Json(manifest.summary()))
}

pub(crate) async fn load(
    State(state): State<ServerState>,
    Json(request): Json<LoadRequest>,
) -> std::result::Result<Json<LoadedCheckpoint>, ApiError> {
    let _permit = state
        .model_operations
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            CondenserError::Config("A checkpoint import or load is already running".into())
        })?;
    Ok(Json(
        load_ollama_with_context(
            &state.model_library,
            &request.checkpoint_id,
            &request.base_url,
            request.context_length,
        )
        .await?,
    ))
}

fn local_url(base: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(base.trim())
        .map_err(|_| CondenserError::Config("Enter a local Ollama HTTP URL".into()))?;
    if url.scheme() != "http"
        || !matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(CondenserError::Config("Checkpoint loading requires a loopback Ollama HTTP URL without credentials or query parameters".into()));
    }
    if url.host_str() == Some("localhost") {
        url.set_host(Some("127.0.0.1"))
            .map_err(|e| CondenserError::Config(e.to_string()))?;
    }
    Ok(url)
}

fn endpoint(base: &reqwest::Url, suffix: &str) -> reqwest::Url {
    let mut url = base.clone();
    url.set_path(&format!(
        "{}/api/{suffix}",
        base.path().trim_end_matches('/')
    ));
    url
}

fn backend_error(error: impl std::fmt::Display) -> CondenserError {
    CondenserError::provider(
        "Ollama",
        format!(
            "Local checkpoint load failed: {error}. Check that Ollama is running and supports this architecture and original precision."
        ),
    )
}

/// Imports only local bytes; no model pull, quantization, context reduction, or backend fallback.
/// A backend that rewrites tensor names, shapes or dtypes fails the check before activation.
pub async fn load_ollama(
    library: &CheckpointLibrary,
    id: &str,
    base: &str,
) -> Result<LoadedCheckpoint> {
    load_ollama_with_context(library, id, base, None).await
}

pub async fn load_ollama_with_context(
    library: &CheckpointLibrary,
    id: &str,
    base: &str,
    context_length: Option<u32>,
) -> Result<LoadedCheckpoint> {
    let base = local_url(base)?;
    let owned = library.clone();
    let wanted = id.to_string();
    let manifest = tokio::task::spawn_blocking(move || owned.verify(&wanted))
        .await
        .map_err(backend_error)?
        .map_err(CondenserError::Config)?;
    if manifest.context_length == 0 || manifest.context_length > i32::MAX as u64 {
        return Err(CondenserError::Config("Checkpoint lacks a supported native context length; refusing to choose a smaller context automatically".into()));
    }
    // An explicit owner choice wins; otherwise the native (or supported scaled) maximum.
    let requested = context_length.map(u64::from);
    if requested.is_some_and(|c| c == 0 || c > manifest.context_length) {
        return Err(CondenserError::Config(
            "Context must be between 1 and the checkpoint's declared maximum".into(),
        ));
    }
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(backend_error)?;
    // Fail before streaming large files when no local backend is reachable.
    let health = client
        .get(endpoint(&base, "version"))
        .send()
        .await
        .map_err(backend_error)?;
    let _: Value = decode_json(ProviderKind::Ollama, health).await?;
    let source_dir = library
        .package_dir(id)
        .map_err(CondenserError::Config)?
        .join("source");
    let mut files = BTreeMap::new();
    for file in &manifest.source_files {
        let digest = format!("sha256:{}", file.sha256);
        let url = endpoint(&base, &format!("blobs/{digest}"));
        let check = client
            .head(url.clone())
            .send()
            .await
            .map_err(backend_error)?;
        if check.status() == reqwest::StatusCode::NOT_FOUND {
            let source = tokio::fs::File::open(source_dir.join(&file.name)).await?;
            let body = reqwest::Body::wrap_stream(tokio_util::io::ReaderStream::new(source));
            let response = client
                .post(url)
                .header(reqwest::header::CONTENT_LENGTH, file.bytes)
                .body(body)
                .send()
                .await
                .map_err(backend_error)?;
            if !response.status().is_success() {
                return Err(backend_error(format!(
                    "blob upload returned HTTP {}",
                    response.status()
                )));
            }
        } else if !check.status().is_success() {
            return Err(backend_error(format!(
                "blob inspection returned HTTP {}",
                check.status()
            )));
        }
        files.insert(file.name.clone(), digest);
    }
    let model = format!("kadmium-{}:latest", manifest.id);
    let response = client
        .post(endpoint(&base, "create"))
        .json(&json!({
            "model":model, "files":files, "stream":false,
            "parameters":{"num_ctx":requested.unwrap_or(manifest.context_length)}
        }))
        .send()
        .await
        .map_err(backend_error)?;
    let created: Value = decode_json(ProviderKind::Ollama, response).await?;
    if created["status"] != "success" {
        return Err(backend_error(
            created
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("backend did not finish model creation"),
        ));
    }
    let response = client
        .post(endpoint(&base, "show"))
        .json(&json!({"model":model,"verbose":true}))
        .send()
        .await
        .map_err(backend_error)?;
    let shown: Value = decode_json(ProviderKind::Ollama, response).await?;
    verify_backend_tensors(&manifest, &shown)?;
    let has_backend_rope_scaling = shown
        .get("model_info")
        .and_then(Value::as_object)
        .is_some_and(|info| info.keys().any(|k| k.contains(".rope.scaling.")));
    let backend_ctx = crate::provider::advertised_context_length(&shown);
    if let Some(ctx) = backend_ctx
        && ctx < manifest.context_length
    {
        return Err(backend_error(
            "backend reported a context length smaller than the checkpoint native context length",
        ));
    }
    let active_context = match (requested, manifest.scaled_context_length) {
        (Some(chosen), _) => chosen,
        (None, Some(scaled))
            if has_backend_rope_scaling && backend_ctx.is_some_and(|ctx| ctx >= scaled) =>
        {
            scaled
        }
        _ => manifest.context_length,
    };
    // Empty generate preloads the actual model. A create success alone does not prove it fits.
    let response = client
        .post(endpoint(&base, "generate"))
        .json(&json!({
            "model":model,"stream":false,"keep_alive":"5m","options":{"num_ctx":active_context}
        }))
        .send()
        .await
        .map_err(backend_error)?;
    let loaded: Value = decode_json(ProviderKind::Ollama, response).await?;
    if loaded["done"] != true || loaded.get("error").is_some() {
        return Err(backend_error("backend did not confirm model loading"));
    }
    let response = client
        .get(endpoint(&base, "ps"))
        .send()
        .await
        .map_err(backend_error)?;
    let running: Value = decode_json(ProviderKind::Ollama, response).await?;
    let full_context = running["models"].as_array().is_some_and(|models| {
        models.iter().any(|entry| {
            (entry["name"] == model || entry["model"] == model)
                && entry["context_length"]
                    .as_u64()
                    .is_some_and(|length| length >= active_context)
        })
    });
    if !full_context {
        return Err(backend_error(
            "backend did not confirm the loaded model at the requested context length",
        ));
    }
    let mut provider = ProviderConfig::new(ProviderKind::Ollama);
    provider.model = model;
    provider.base_url = Some(base.as_str().trim_end_matches('/').to_string());
    Ok(LoadedCheckpoint {
        checkpoint: manifest.summary(),
        provider,
        tensor_metadata_verified: true,
        context_length: u32::try_from(active_context).map_err(backend_error)?,
        execution: "Local Ollama using preserved source files; KTB banks are storage artifacts. Backend arithmetic and KV precision are not RepOps-certified.",
    })
}

fn verify_backend_tensors(manifest: &CheckpointManifest, shown: &Value) -> Result<()> {
    let tensors = shown["tensors"].as_array().ok_or_else(|| {
        backend_error("backend did not report tensor metadata; cannot verify source precision")
    })?;
    if tensors.len() != manifest.tensors.len() {
        return Err(backend_error("backend changed the tensor count"));
    }
    let mut actual = BTreeMap::new();
    for t in tensors {
        let name = t["name"]
            .as_str()
            .ok_or_else(|| backend_error("tensor name is missing"))?;
        if actual.insert(name, t).is_some() {
            return Err(backend_error("duplicate backend tensor"));
        }
    }
    for t in &manifest.tensors {
        let found = actual
            .get(t.name.as_str())
            .ok_or_else(|| backend_error(format!("backend omitted or renamed {}", t.name)))?;
        let dtype = found["type"].as_str().unwrap_or("").to_ascii_uppercase();
        let expected = match t.encoding {
            kadmium_data::TensorEncoding::Ggml(code) => {
                ggml_type_name(code).unwrap_or("unsupported").to_string()
            }
            kadmium_data::TensorEncoding::Fp32 => "F32".into(),
            kadmium_data::TensorEncoding::Fp16 => "F16".into(),
            kadmium_data::TensorEncoding::Bf16 => "BF16".into(),
            _ => t.dtype.clone(),
        };
        let shape: Vec<u64> =
            serde_json::from_value(found["shape"].clone()).map_err(backend_error)?;
        let normalized = |s: &[u64]| {
            let mut v = s.to_vec();
            while manifest.source_format == "gguf" && v.len() > 1 && v.last() == Some(&1) {
                v.pop();
            }
            v
        };
        if dtype != expected || normalized(&shape) != normalized(&t.shape) {
            return Err(backend_error(format!(
                "backend changed the dtype or shape of {} (expected {expected}, received {dtype}); task route was not changed",
                t.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{user_facing_path, verify_backend_tensors};
    use kadmium_data::TensorEncoding;
    use kadmium_data::checkpoint::{CheckpointManifest, CheckpointTensor};
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn backend_must_report_each_ggml_type_by_its_ggml_name() {
        let tensor = |name: &str, code: u32| CheckpointTensor {
            name: name.into(),
            dtype: format!("GGML_{code}"),
            shape: vec![256],
            encoding: TensorEncoding::Ggml(code),
            source: "model.gguf".into(),
            source_offset: 0,
            bytes: 0,
            bank: String::new(),
            bank_offset: 0,
        };
        let manifest = CheckpointManifest {
            schema_version: 3,
            id: String::new(),
            name: "fixture".into(),
            source_format: "gguf".into(),
            architecture: Some("llama".into()),
            context_length: 4096,
            scaled_context_length: None,
            source_files: Vec::new(),
            banks: Vec::new(),
            tensors: vec![tensor("experts", 39), tensor("attn", 16)],
            quality_policy: "preserve_source".into(),
            pre_quantized: None,
            warnings: Vec::new(),
        };
        let shown = |expert: &str| {
            json!({"tensors":[
                {"name":"experts","shape":[256],"type":expert},
                {"name":"attn","shape":[256],"type":"iq2_xxs"},
            ]})
        };
        verify_backend_tensors(&manifest, &shown("MXFP4")).unwrap();
        let changed = verify_backend_tensors(&manifest, &shown("BF16")).unwrap_err();
        assert!(
            changed
                .to_string()
                .contains("expected MXFP4, received BF16")
        );
    }

    #[test]
    fn verbatim_windows_prefixes_are_removed_for_display() {
        for (verbatim, plain) in [
            (r"\\?\C:\Models\llama", r"C:\Models\llama"),
            (r"\\?\UNC\nas\models", r"\\nas\models"),
            (r"\\?\Volume{0}\models", r"\\?\Volume{0}\models"),
            ("/home/me/models", "/home/me/models"),
        ] {
            assert_eq!(
                user_facing_path(PathBuf::from(verbatim)),
                PathBuf::from(plain)
            );
        }
    }
}
