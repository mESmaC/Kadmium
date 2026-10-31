//! Receives prepared checkpoints from Condenser and serves them from this coordinator's Ollama.
//! Chunks are verified on arrival, the whole package is verified in the background before it is
//! published, and the model loads at its full context by default. Received models run on this
//! coordinator's single Ollama backend, not across the grid.
//!
//! Access: set `KADMIUM_TRANSPORT_TOKEN` to accept bearer-authenticated requests from other
//! computers (required behind a reverse proxy). Without it, only same-origin requests from this
//! computer that carry no forwarding headers are accepted.

use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use kadmium_data::checkpoint::CheckpointLibrary;
use kadmium_data::checkpoint::transport::{TRANSPORT_CHUNK_BYTES, TransportManifest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

const MAX_CHAT_BODY: usize = 64 * 1024 * 1024;
const MAX_MANIFEST_BODY: usize = 64 * 1024 * 1024;
/// Native Ollama API for received models; Condenser's Ollama provider uses it so every request
/// carries explicit full context, no truncation and no context shifting.
pub const OLLAMA_PROXY_PATH: &str = "/api/v1/model-transport/ollama";

enum Finishing {
    Running,
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
    finishing: Arc<Mutex<HashMap<String, Finishing>>>,
    loads: Arc<tokio::sync::Semaphore>,
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
            loads: Arc::new(tokio::sync::Semaphore::new(1)),
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
    match &state.token {
        Some(token) => request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|given| constant_time_eq(given.trim().as_bytes(), token.as_bytes())),
        None => {
            request
                .extensions()
                .get::<ConnectInfo<SocketAddr>>()
                .is_some_and(|peer| peer.0.ip().is_loopback())
                && local_same_origin(request.headers())
        }
    }
}

/// A loopback peer is not enough on its own: a local reverse proxy makes every request look
/// local, and a browser on this computer can be steered here by a rebound DNS name.
fn local_same_origin(headers: &HeaderMap) -> bool {
    if [
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-real-ip",
    ]
    .iter()
    .any(|name| headers.contains_key(*name))
    {
        return false;
    }
    let Some(host) = headers.get(header::HOST).and_then(|h| h.to_str().ok()) else {
        return false;
    };
    let name = host
        .rsplit_once(':')
        .filter(|(_, port)| port.chars().all(|c| c.is_ascii_digit()))
        .map_or(host, |(name, _)| name);
    if !matches!(name, "127.0.0.1" | "localhost" | "[::1]") {
        return false;
    }
    headers.get(header::ORIGIN).is_none_or(|origin| {
        origin
            .to_str()
            .is_ok_and(|origin| origin == format!("http://{host}"))
    })
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn denied(state: &TransportState) -> Response {
    let message = if state.token.is_some() {
        "A valid access token is required (the coordinator's KADMIUM_TRANSPORT_TOKEN)"
    } else {
        "Requests from other computers or through a proxy need KADMIUM_TRANSPORT_TOKEN set on this coordinator"
    };
    (StatusCode::UNAUTHORIZED, Json(json!({ "error": message }))).into_response()
}

async fn guard(State(state): State<TransportState>, request: Request, next: Next) -> Response {
    if authorized(&state, &request) {
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
        Some(Finishing::Running) => {
            finishing.insert(id, Finishing::Running);
            return (StatusCode::ACCEPTED, Json(json!({ "state": "verifying" }))).into_response();
        }
        Some(Finishing::Failed(error)) => return failed(StatusCode::BAD_REQUEST, error),
        None => {}
    }
    finishing.insert(id.clone(), Finishing::Running);
    drop(finishing);
    let library = state.library.clone();
    let tracker = state.finishing.clone();
    tokio::task::spawn_blocking(move || {
        let outcome = library.receive_finish(&id);
        let mut finishing = tracker.lock().expect("finishing");
        match outcome {
            Ok(_) => finishing.remove(&id),
            Err(error) => finishing.insert(id, Finishing::Failed(error)),
        };
    });
    (StatusCode::ACCEPTED, Json(json!({ "state": "verifying" }))).into_response()
}

#[derive(Deserialize, Default)]
struct LoadRequest {
    context_length: Option<u32>,
}

async fn load(
    State(state): State<TransportState>,
    Path(id): Path<String>,
    request: Option<Json<LoadRequest>>,
) -> Response {
    let Ok(_permit) = state.loads.clone().try_acquire_owned() else {
        return failed(
            StatusCode::CONFLICT,
            "Another model is loading; try again shortly",
        );
    };
    let context = request.map(|Json(r)| r).unwrap_or_default().context_length;
    match kadmium_condenser::local_models::load_ollama_with_context(
        &state.library,
        &id,
        &state.ollama,
        context,
    )
    .await
    {
        Ok(loaded) => {
            let base: Arc<str> = loaded
                .provider
                .base_url
                .clone()
                .unwrap_or_else(|| state.ollama.to_string())
                .into();
            state
                .served
                .lock()
                .expect("served")
                .insert(loaded.provider.model.clone(), base);
            Json(json!({
                "model": loaded.provider.model,
                "context_length": loaded.context_length,
                "checkpoint": loaded.checkpoint,
                "execution": "This coordinator's Ollama, using the preserved source files. Backend arithmetic and KV precision are not RepOps-certified.",
                "ollama_path": OLLAMA_PROXY_PATH,
            }))
            .into_response()
        }
        Err(error) => failed(StatusCode::BAD_GATEWAY, error),
    }
}

/// `kadmium-<64 lowercase hex>` with an optional `:latest` tag: a name only received models use.
fn transported_id(model: &str) -> Option<&str> {
    let id = model.strip_prefix("kadmium-")?;
    let id = id.strip_suffix(":latest").unwrap_or(id);
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
    let library = state.library.clone();
    if blocking(move || library.read(&id)).await.is_err() {
        return Err(not_loaded());
    }
    let name = format!("{}:latest", model.trim_end_matches(":latest"));
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

/// Routes OpenAI-compatible chat for received models to the backend that loaded them, and
/// refuses received-model names the coordinator cannot serve (including on `/v1/messages`)
/// instead of letting the diagnostic handler answer for them. Other requests are unchanged.
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
    let messages = path == "/v1/messages";
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
    if messages {
        return failed(
            StatusCode::BAD_REQUEST,
            format!(
                "Received models are served through /v1/chat/completions and {OLLAMA_PROXY_PATH}, not /v1/messages"
            ),
        );
    }
    match backend_for(&state, &model).await {
        Ok(base) => {
            kadmium_condenser::model_transport::forward_json(&base, "v1/chat/completions", bytes)
                .await
        }
        Err(response) => *response,
    }
}

#[cfg(test)]
mod tests {
    use super::{local_same_origin, transported_id};
    use axum::http::HeaderMap;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, value.parse().unwrap());
        }
        map
    }

    #[test]
    fn local_access_requires_a_loopback_host_without_forwarding_or_foreign_origin() {
        assert!(local_same_origin(&headers(&[("host", "127.0.0.1:7848")])));
        assert!(local_same_origin(&headers(&[
            ("host", "localhost:7848"),
            ("origin", "http://localhost:7848")
        ])));
        assert!(!local_same_origin(&headers(&[(
            "host",
            "evil.example:7848"
        )])));
        assert!(!local_same_origin(&headers(&[
            ("host", "127.0.0.1:7848"),
            ("origin", "http://evil.example")
        ])));
        assert!(!local_same_origin(&headers(&[
            ("host", "127.0.0.1:7848"),
            ("x-forwarded-for", "203.0.113.9")
        ])));
        assert!(!local_same_origin(&HeaderMap::new()));
    }

    #[test]
    fn only_content_addressed_names_are_received_models() {
        let id = "a".repeat(64);
        assert_eq!(
            transported_id(&format!("kadmium-{id}:latest")),
            Some(id.as_str())
        );
        assert_eq!(transported_id(&format!("kadmium-{id}")), Some(id.as_str()));
        assert_eq!(transported_id("kadmium-deepseek"), None);
        assert_eq!(transported_id("llama3.2"), None);
    }
}
