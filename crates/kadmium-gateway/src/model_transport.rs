//! Receives prepared checkpoints from Condenser and serves them from this coordinator's Ollama.
//! Chunks are verified on arrival, the whole package is verified in the background before it is
//! published, and the model loads at its full context by default. Received models run on this
//! coordinator's single Ollama backend, not across the grid.
//!
//! Access: every upload, load and chat request needs `Authorization: Bearer <token>` matching
//! `KADMIUM_TRANSPORT_TOKEN`. Without a token the coordinator does not receive or serve models.
//! A loopback peer is not treated as proof of the owner: a local reverse proxy or another local
//! user also connects from loopback. Only the read-only listing is also open to a same-origin
//! browser on this computer, through the same check as the embedded Condenser workspace.

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use kadmium_data::checkpoint::CheckpointLibrary;
use kadmium_data::checkpoint::transport::{TRANSPORT_CHUNK_BYTES, TransportManifest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

const MAX_CHAT_BODY: usize = 64 * 1024 * 1024;
const MAX_MANIFEST_BODY: usize = 64 * 1024 * 1024;
/// Native Ollama API for received models; Condenser's Ollama provider uses it so every request
/// carries explicit full context, no truncation and no context shifting.
pub const OLLAMA_PROXY_PATH: &str = "/api/v1/model-transport/ollama";

/// A background step (verification or loading) for one checkpoint.
enum Step {
    Running,
    Done(Value),
    Failed(String),
}

#[derive(Clone)]
pub struct TransportState {
    library: CheckpointLibrary,
    ollama: Arc<str>,
    token: Option<Arc<str>>,
    max_bytes: Option<u64>,
    /// Loaded model name -> the Ollama base URL serving it.
    served: Arc<Mutex<BTreeMap<String, Arc<str>>>>,
    finishing: Arc<Mutex<HashMap<String, Step>>>,
    loading: Arc<Mutex<HashMap<String, Step>>>,
}

impl TransportState {
    pub fn from_env() -> Self {
        let mut state = Self::new(
            CheckpointLibrary::default(),
            std::env::var("KADMIUM_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            std::env::var("KADMIUM_TRANSPORT_TOKEN")
                .ok()
                .filter(|t| !t.trim().is_empty()),
        );
        state.max_bytes = std::env::var("KADMIUM_TRANSPORT_MAX_BYTES")
            .ok()
            .and_then(|v| v.trim().parse().ok());
        state
    }

    pub fn new(library: CheckpointLibrary, ollama: String, token: Option<String>) -> Self {
        Self {
            library,
            ollama: ollama.into(),
            token: token.map(|t| t.trim().into()),
            max_bytes: None,
            served: Default::default(),
            finishing: Default::default(),
            loading: Default::default(),
        }
    }

    /// Caps the size of a single received model.
    pub fn with_max_bytes(mut self, max_bytes: Option<u64>) -> Self {
        self.max_bytes = max_bytes;
        self
    }
}

pub fn router(state: TransportState) -> Router {
    Router::new()
        .route(
            "/api/v1/model-transport",
            // Chunk maps for very large models run to megabytes; match the manifest bound.
            post(begin)
                .layer(DefaultBodyLimit::max(MAX_MANIFEST_BODY))
                .get(library),
        )
        .route(
            "/api/v1/model-transport/{id}/files/{file}/chunks/{chunk}",
            put(chunk).layer(DefaultBodyLimit::max(TRANSPORT_CHUNK_BYTES as usize + 4096)),
        )
        .route("/api/v1/model-transport/{id}/complete", post(complete))
        .route("/api/v1/model-transport/{id}/load", post(load))
        .route(
            &format!("{OLLAMA_PROXY_PATH}/api/chat"),
            post(|s, b| native(s, "api/chat", b)).layer(DefaultBodyLimit::max(MAX_CHAT_BODY)),
        )
        .route(
            &format!("{OLLAMA_PROXY_PATH}/api/show"),
            post(|s, b| native(s, "api/show", b)),
        )
        .route(&format!("{OLLAMA_PROXY_PATH}/api/tags"), get(tags))
        .route_layer(axum::middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

fn authorized(state: &TransportState, request: &Request) -> bool {
    state.token.as_deref().is_some_and(|token| {
        request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|given| constant_time_eq(given.trim().as_bytes(), token.as_bytes()))
    })
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn denied(state: &TransportState) -> Response {
    if state.token.is_some() {
        failed(
            StatusCode::UNAUTHORIZED,
            "A valid access token is required (the coordinator's KADMIUM_TRANSPORT_TOKEN)",
        )
    } else {
        failed(
            StatusCode::FORBIDDEN,
            "This coordinator does not receive models. Set KADMIUM_TRANSPORT_TOKEN on it and restart Kadmium.",
        )
    }
}

async fn guard(State(state): State<TransportState>, request: Request, next: Next) -> Response {
    let listing =
        request.method() == Method::GET && request.uri().path() == "/api/v1/model-transport";
    if authorized(&state, &request)
        || (listing && crate::condenser_mount::local_browser_request(&request))
    {
        next.run(request).await
    } else {
        denied(&state)
    }
}

fn failed(status: StatusCode, error: impl std::fmt::Display) -> Response {
    (status, Json(json!({ "error": error.to_string() }))).into_response()
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, Box<Response>> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| Box::new(failed(StatusCode::INTERNAL_SERVER_ERROR, e)))?
        .map_err(|e| Box::new(failed(StatusCode::BAD_REQUEST, e)))
}

async fn library(State(state): State<TransportState>) -> Response {
    let library = state.library.clone();
    match blocking(move || library.list()).await {
        Ok(checkpoints) => {
            let served: Vec<_> = state
                .served
                .lock()
                .expect("served")
                .keys()
                .cloned()
                .collect();
            Json(json!({ "checkpoints": checkpoints, "served": served })).into_response()
        }
        Err(response) => *response,
    }
}

async fn begin(
    State(state): State<TransportState>,
    Json(transport): Json<TransportManifest>,
) -> Response {
    if let Some(limit) = state.max_bytes
        && transport.total_bytes > limit
    {
        return failed(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "This coordinator accepts models up to {limit} bytes (KADMIUM_TRANSPORT_MAX_BYTES)"
            ),
        );
    }
    let library = state.library.clone();
    match blocking(move || library.receive_begin(&transport)).await {
        Ok(status) => Json(status).into_response(),
        Err(response) => *response,
    }
}

async fn chunk(
    State(state): State<TransportState>,
    Path((id, file, chunk)): Path<(String, u32, u32)>,
    body: Bytes,
) -> Response {
    let library = state.library.clone();
    match blocking(move || library.receive_chunk(&id, file, chunk, &body)).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(response) => *response,
    }
}

/// Starts (or reports) whole-package verification. Verification re-hashes every byte, which can
/// take longer than any reasonable HTTP request, so it runs in the background: callers repeat
/// this request until it answers 200 with the published checkpoint, or an error.
async fn complete(State(state): State<TransportState>, Path(id): Path<String>) -> Response {
    let library = state.library.clone();
    let wanted = id.clone();
    match blocking(move || library.read(&wanted)).await {
        Ok(manifest) => {
            return Json(json!({ "state": "published", "checkpoint": manifest.summary() }))
                .into_response();
        }
        Err(_) if state.library.package_dir(&id).is_err() => {
            return failed(StatusCode::BAD_REQUEST, "Invalid checkpoint ID");
        }
        Err(_) => {}
    }
    let mut finishing = state.finishing.lock().expect("finishing");
    match finishing.remove(&id) {
        Some(Step::Running) => {
            finishing.insert(id, Step::Running);
            return (StatusCode::ACCEPTED, Json(json!({ "state": "verifying" }))).into_response();
        }
        Some(Step::Failed(error)) => return failed(StatusCode::BAD_REQUEST, error),
        Some(Step::Done(_)) | None => {}
    }
    // Each verification re-reads a whole model; run one at a time so concurrent senders cannot
    // multiply the disk load.
    if finishing.values().any(|step| matches!(step, Step::Running)) {
        return busy("Another model is being verified");
    }
    finishing.insert(id.clone(), Step::Running);
    drop(finishing);
    let library = state.library.clone();
    let tracker = state.finishing.clone();
    tokio::task::spawn_blocking(move || {
        let outcome = library.receive_finish(&id);
        let mut finishing = tracker.lock().expect("finishing");
        match outcome {
            Ok(_) => finishing.remove(&id),
            Err(error) => finishing.insert(id, Step::Failed(error)),
        };
    });
    (StatusCode::ACCEPTED, Json(json!({ "state": "verifying" }))).into_response()
}

#[derive(Deserialize, Default)]
struct LoadRequest {
    context_length: Option<u32>,
}

fn busy(what: &str) -> Response {
    (
        StatusCode::CONFLICT,
        Json(json!({ "state": "busy", "error": format!("{what}; try again shortly") })),
    )
        .into_response()
}

/// Starts (or reports) loading into this coordinator's Ollama. Loading streams every source file
/// to the backend and preloads the model, which can outlast an HTTP request, so it runs in the
/// background: callers repeat this request until it answers 200 with the loaded model.
async fn load(
    State(state): State<TransportState>,
    Path(id): Path<String>,
    request: Option<Json<LoadRequest>>,
) -> Response {
    let context = request.map(|Json(r)| r).unwrap_or_default().context_length;
    let key = format!(
        "{id}@{}",
        context.map_or_else(|| "max".into(), |c| c.to_string())
    );
    let mut loading = state.loading.lock().expect("loading");
    match loading.remove(&key) {
        Some(Step::Running) => {
            loading.insert(key, Step::Running);
            return (StatusCode::ACCEPTED, Json(json!({ "state": "loading" }))).into_response();
        }
        Some(Step::Done(loaded)) => return Json(loaded).into_response(),
        Some(Step::Failed(error)) => return failed(StatusCode::BAD_GATEWAY, error),
        None => {}
    }
    if loading.values().any(|step| matches!(step, Step::Running)) {
        return busy("Another model is loading");
    }
    loading.insert(key.clone(), Step::Running);
    drop(loading);
    let task = state.clone();
    tokio::spawn(async move {
        let outcome = kadmium_condenser::local_models::load_ollama_with_context(
            &task.library,
            &id,
            &task.ollama,
            context,
        )
        .await;
        let step = match outcome {
            Ok(loaded) => {
                let base: Arc<str> = loaded
                    .provider
                    .base_url
                    .clone()
                    .unwrap_or_else(|| task.ollama.to_string())
                    .into();
                task.served
                    .lock()
                    .expect("served")
                    .insert(loaded.provider.model.clone(), base);
                Step::Done(json!({
                    "model": loaded.provider.model,
                    "context_length": loaded.context_length,
                    "checkpoint": loaded.checkpoint,
                    "execution": "This coordinator's Ollama, using the preserved source files. Backend arithmetic and KV precision are not RepOps-certified.",
                    "ollama_path": OLLAMA_PROXY_PATH,
                }))
            }
            Err(error) => Step::Failed(error.to_string()),
        };
        task.loading.lock().expect("loading").insert(key, step);
    });
    (StatusCode::ACCEPTED, Json(json!({ "state": "loading" }))).into_response()
}

/// `kadmium-<64 lowercase hex>` with any tag or none: a name only received models use.
fn transported_id(model: &str) -> Option<&str> {
    let id = model.strip_prefix("kadmium-")?;
    let id = id.split_once(':').map_or(id, |(id, _)| id);
    (id.len() == 64
        && id
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()))
    .then_some(id)
}

/// Finds the backend for a received model. After a restart the in-memory table is empty, so a
/// published checkpoint whose model still exists in Ollama is registered again. Anything else is
/// an explicit "not loaded" error, never a diagnostic reply.
async fn backend_for(state: &TransportState, model: &str) -> Result<Arc<str>, Box<Response>> {
    if let Some(base) = state.served.lock().expect("served").get(model).cloned() {
        return Ok(base);
    }
    let not_loaded = || {
        Box::new(failed(
            StatusCode::CONFLICT,
            format!(
                "{model} is not loaded on this coordinator. Send it again from Condenser, or POST /api/v1/model-transport/<id>/load."
            ),
        ))
    };
    let id = transported_id(model).ok_or_else(not_loaded)?.to_string();
    // Loading creates only `:latest`; any other tag was never created here.
    let name = format!("kadmium-{id}:latest");
    if model != name && model != format!("kadmium-{id}") {
        return Err(not_loaded());
    }
    let library = state.library.clone();
    if blocking(move || library.read(&id)).await.is_err() {
        return Err(not_loaded());
    }
    if !kadmium_condenser::model_transport::ollama_has_model(&state.ollama, &name).await {
        return Err(not_loaded());
    }
    let mut served = state.served.lock().expect("served");
    served.insert(name.clone(), state.ollama.clone());
    if name != model {
        served.insert(model.to_string(), state.ollama.clone());
    }
    Ok(state.ollama.clone())
}

fn requested_model(bytes: &Bytes) -> Option<String> {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|v| v["model"].as_str().map(str::to_owned))
}

/// Native Ollama endpoints, restricted to received models.
async fn native(State(state): State<TransportState>, path: &'static str, body: Bytes) -> Response {
    let Some(model) = requested_model(&body) else {
        return failed(StatusCode::BAD_REQUEST, "Request must name a model");
    };
    match backend_for(&state, &model).await {
        Ok(base) => kadmium_condenser::model_transport::forward_json(&base, path, body).await,
        Err(response) => *response,
    }
}

/// Lists only received models, in Ollama's `/api/tags` shape.
async fn tags(State(state): State<TransportState>) -> Response {
    let models: Vec<_> = state
        .served
        .lock()
        .expect("served")
        .keys()
        .filter(|name| name.ends_with(":latest"))
        .map(|name| json!({ "name": name, "model": name }))
        .collect();
    Json(json!({ "models": models })).into_response()
}

/// Keeps received-model names away from the diagnostic `/v1` handlers. Ollama's
/// OpenAI-compatible endpoint cannot turn off prompt truncation or context shifting, so a
/// received model is never served there: both `/v1` chat routes answer with an explicit error
/// that names the native route, which carries full context and no truncation on every request.
/// Other requests are unchanged.
pub async fn serve_loaded_models(
    State(state): State<TransportState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if request.method() != Method::POST || !matches!(path, "/v1/chat/completions" | "/v1/messages")
    {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, MAX_CHAT_BODY).await {
        Ok(bytes) => bytes,
        Err(error) => return failed(StatusCode::PAYLOAD_TOO_LARGE, error),
    };
    let request = Request::from_parts(parts, Body::from(bytes.clone()));
    let Some(model) = requested_model(&bytes) else {
        return next.run(request).await;
    };
    let received = transported_id(&model).is_some()
        || state.served.lock().expect("served").contains_key(&model);
    if !received {
        return next.run(request).await;
    }
    if !authorized(&state, &request) {
        return denied(&state);
    }
    if let Err(response) = backend_for(&state, &model).await {
        return *response;
    }
    failed(
        StatusCode::BAD_REQUEST,
        format!(
            "Received models are served only through {OLLAMA_PROXY_PATH}/api/chat, which keeps full context with no truncation. Ollama's OpenAI-compatible endpoint cannot guarantee that, so Kadmium does not serve {model} on {}. Condenser uses the native route automatically.",
            path_of(&request)
        ),
    )
}

fn path_of(request: &Request) -> String {
    request.uri().path().to_string()
}

#[cfg(test)]
mod tests {
    use super::transported_id;

    #[test]
    fn only_content_addressed_names_are_received_models() {
        let id = "a".repeat(64);
        assert_eq!(
            transported_id(&format!("kadmium-{id}:latest")),
            Some(id.as_str())
        );
        assert_eq!(transported_id(&format!("kadmium-{id}")), Some(id.as_str()));
        assert_eq!(
            transported_id(&format!("kadmium-{id}:other")),
            Some(id.as_str())
        );
        assert_eq!(transported_id("kadmium-deepseek"), None);
        assert_eq!(transported_id("llama3.2"), None);
    }
}
