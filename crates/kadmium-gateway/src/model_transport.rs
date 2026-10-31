//! Receives prepared checkpoints from Condenser and serves them from this coordinator.
//! Kadmium Cloud runs the same endpoints. Chunks are verified on arrival, the whole package is
//! verified before it is published, and the model loads at its full context by default.
//!
//! Access: set `KADMIUM_TRANSPORT_TOKEN` to accept bearer-authenticated uploads over the
//! network. Without it, only requests from this computer are accepted.

use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, put};
use axum::{Json, Router};
use kadmium_data::checkpoint::CheckpointLibrary;
use kadmium_data::checkpoint::transport::{TRANSPORT_CHUNK_BYTES, TransportManifest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

const MAX_CHAT_BODY: usize = 64 * 1024 * 1024;
const MAX_MANIFEST_BODY: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct TransportState {
    library: CheckpointLibrary,
    ollama: Arc<str>,
    token: Option<Arc<str>>,
    /// Loaded model name -> the Ollama base URL serving it.
    served: Arc<Mutex<BTreeMap<String, Arc<str>>>>,
    loads: Arc<tokio::sync::Semaphore>,
}

impl TransportState {
    pub fn from_env() -> Self {
        Self::new(
            CheckpointLibrary::default(),
            std::env::var("KADMIUM_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            std::env::var("KADMIUM_TRANSPORT_TOKEN")
                .ok()
                .filter(|t| !t.trim().is_empty()),
        )
    }

    pub fn new(library: CheckpointLibrary, ollama: String, token: Option<String>) -> Self {
        Self {
            library,
            ollama: ollama.into(),
            token: token.map(|t| t.trim().into()),
            served: Default::default(),
            loads: Arc::new(tokio::sync::Semaphore::new(1)),
        }
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
        None => request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .is_some_and(|peer| peer.0.ip().is_loopback()),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn denied(state: &TransportState) -> Response {
    let message = if state.token.is_some() {
        "A valid Kadmium API key is required"
    } else {
        "Uploads from other computers need KADMIUM_TRANSPORT_TOKEN set on this coordinator"
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

async fn complete(State(state): State<TransportState>, Path(id): Path<String>) -> Response {
    let library = state.library.clone();
    match blocking(move || library.receive_finish(&id)).await {
        Ok(manifest) => Json(json!({ "checkpoint": manifest.summary() })).into_response(),
        Err(response) => *response,
    }
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
                "execution": loaded.execution,
            }))
            .into_response()
        }
        Err(error) => failed(StatusCode::BAD_GATEWAY, error),
    }
}

/// Routes chat requests for transported models to the backend that loaded them. Every other
/// request continues to the coordinator's own handler unchanged.
pub async fn serve_loaded_models(
    State(state): State<TransportState>,
    request: Request,
    next: Next,
) -> Response {
    if request.method() != Method::POST || request.uri().path() != "/v1/chat/completions" {
        return next.run(request).await;
    }
    if state.served.lock().expect("served").is_empty() {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, MAX_CHAT_BODY).await {
        Ok(bytes) => bytes,
        Err(error) => return failed(StatusCode::PAYLOAD_TOO_LARGE, error),
    };
    let model = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| v["model"].as_str().map(str::to_owned));
    let backend = model.and_then(|m| state.served.lock().expect("served").get(&m).cloned());
    let request = Request::from_parts(parts, Body::from(bytes.clone()));
    match backend {
        Some(_) if !authorized(&state, &request) => denied(&state),
        Some(base) => kadmium_condenser::model_transport::forward_chat(&base, bytes).await,
        None => next.run(request).await,
    }
}
