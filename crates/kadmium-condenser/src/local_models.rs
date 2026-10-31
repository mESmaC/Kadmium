//! Explicit local checkpoint -> lossless package -> local Ollama -> Condenser handoff.
//! The backend consumes the preserved originals, not the storage-only KTB banks.

use crate::provider::decode_json;
use crate::server::{ApiError, ServerState};
use crate::{CondenserError, ProviderConfig, ProviderKind, Result};
use axum::{Json, extract::State};
use kadmium_data::checkpoint::{CheckpointLibrary, CheckpointManifest, CheckpointSummary};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Deserialize)]
pub(crate) struct ImportRequest {
    source_path: PathBuf,
}
#[derive(Deserialize)]
pub(crate) struct LoadRequest {
    checkpoint_id: String,
    base_url: String,
}

#[derive(Serialize)]
pub struct LoadedCheckpoint {
    pub checkpoint: CheckpointSummary,
    pub provider: ProviderConfig,
    pub tensor_metadata_verified: bool,
    pub execution: &'static str,
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
        library.import(&request.source_path)
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
        load_ollama(
            &state.model_library,
            &request.checkpoint_id,
            &request.base_url,
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
            "parameters":{"num_ctx":manifest.context_length}
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
    // Empty generate preloads the actual model. A create success alone does not prove it fits.
    let response = client.post(endpoint(&base, "generate")).json(&json!({
        "model":model,"stream":false,"keep_alive":"5m","options":{"num_ctx":manifest.context_length}
    })).send().await.map_err(backend_error)?;
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
                    .is_some_and(|length| length >= manifest.context_length)
        })
    });
    if !full_context {
        return Err(backend_error(
            "backend did not confirm the loaded model at its full declared context length",
        ));
    }
    let mut provider = ProviderConfig::new(ProviderKind::Ollama);
    provider.model = model;
    provider.base_url = Some(base.as_str().trim_end_matches('/').to_string());
    Ok(LoadedCheckpoint {
        checkpoint: manifest.summary(),
        provider,
        tensor_metadata_verified: true,
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
            kadmium_data::TensorEncoding::Ggml(code) => match code {
                2 => "Q4_0",
                3 => "Q4_1",
                6 => "Q5_0",
                7 => "Q5_1",
                8 => "Q8_0",
                9 => "Q8_1",
                10 => "Q2_K",
                11 => "Q3_K",
                12 => "Q4_K",
                13 => "Q5_K",
                14 => "Q6_K",
                15 => "Q8_K",
                _ => "unsupported",
            }
            .to_string(),
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
