//! Sends a prepared checkpoint to a Kadmium coordinator using the
//! verified, resumable chunk protocol in `kadmium_data::checkpoint::transport`. The receiver
//! loads the preserved originals into its Ollama at the checkpoint's full context by default,
//! and tasks then reach it through the receiver's native Ollama proxy, so every request keeps
//! explicit full context, unlimited output, tools, and no truncation or context shifting.

use crate::server::{ApiError, ServerState};
use crate::{CondenserError, ProviderConfig, ProviderKind, Result};
use axum::{Json, extract::State};
use kadmium_data::checkpoint::CheckpointLibrary;
use kadmium_data::checkpoint::transport::ReceiveStatus;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{Read, Seek, SeekFrom};
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const CHUNK_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Default, Serialize)]
pub struct PublishProgress {
    /// idle, preparing, uploading, verifying, loading, ready, or failed.
    pub state: &'static str,
    pub checkpoint_id: Option<String>,
    pub endpoint: Option<String>,
    pub total_bytes: u64,
    pub sent_bytes: u64,
    pub skipped_bytes: u64,
    pub message: String,
    pub result: Option<Value>,
}

pub(crate) type SharedProgress = Arc<Mutex<PublishProgress>>;

#[derive(Deserialize)]
pub(crate) struct StatusQuery {
    checkpoint_id: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct TransportRequest {
    checkpoint_id: String,
}

#[derive(Deserialize)]
pub(crate) struct PublishRequest {
    checkpoint_id: String,
    endpoint: String,
    api_key: Option<String>,
    context_length: Option<u32>,
}

/// Builds (or reads the cached) transport manifest for a prepared checkpoint.
pub(crate) async fn transport(
    State(state): State<ServerState>,
    Json(request): Json<TransportRequest>,
) -> std::result::Result<Json<Value>, ApiError> {
    let library = state.model_library.clone();
    let manifest = tokio::task::spawn_blocking(move || library.transport(&request.checkpoint_id))
        .await
        .map_err(|e| CondenserError::Config(e.to_string()))?
        .map_err(CondenserError::Config)?;
    Ok(Json(json!(manifest.summary())))
}

/// Progress of the latest send. With `?checkpoint_id=`, another model's send reads as idle, so
/// a second tab never shows a different model's progress as its own.
pub(crate) async fn publish_status(
    State(state): State<ServerState>,
    axum::extract::Query(query): axum::extract::Query<StatusQuery>,
) -> Json<PublishProgress> {
    let progress = state.publish.lock().expect("publish progress").clone();
    let other = query
        .checkpoint_id
        .is_some_and(|id| progress.checkpoint_id.as_deref() != Some(id.as_str()));
    if progress.state.is_empty() || other {
        return Json(PublishProgress {
            state: "idle",
            ..Default::default()
        });
    }
    Json(progress)
}

pub(crate) async fn publish(
    State(state): State<ServerState>,
    Json(request): Json<PublishRequest>,
) -> std::result::Result<Json<PublishProgress>, ApiError> {
    let endpoint = remote_url(&request.endpoint)?;
    let permit = state
        .model_operations
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            CondenserError::Config("A model is already being prepared, loaded or sent".into())
        })?;
    state
        .model_library
        .package_dir(&request.checkpoint_id)
        .map_err(CondenserError::Config)?;
    let started = PublishProgress {
        state: "preparing",
        checkpoint_id: Some(request.checkpoint_id.clone()),
        endpoint: Some(endpoint.as_str().trim_end_matches('/').into()),
        message: "Checking every chunk before sending".into(),
        ..Default::default()
    };
    *state.publish.lock().expect("publish progress") = started.clone();
    let progress = state.publish.clone();
    let library = state.model_library.clone();
    tokio::spawn(async move {
        let _permit = permit;
        let outcome = send(&library, &progress, &request, endpoint).await;
        let mut current = progress.lock().expect("publish progress");
        match outcome {
            Ok(result) => {
                current.state = "ready";
                current.message = "Verified and loaded in the Kadmium coordinator's Ollama".into();
                current.result = Some(result);
            }
            Err(error) => {
                current.state = "failed";
                current.message = error.to_string();
            }
        }
    });
    Ok(Json(started))
}

async fn send(
    library: &CheckpointLibrary,
    progress: &SharedProgress,
    request: &PublishRequest,
    endpoint: reqwest::Url,
) -> Result<Value> {
    let owned = library.clone();
    let id = request.checkpoint_id.clone();
    let transport = tokio::task::spawn_blocking(move || owned.transport(&id))
        .await
        .map_err(failure)?
        .map_err(CondenserError::Config)?;
    let client = reqwest::Client::builder()
        .user_agent(concat!("Condenser/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(failure)?;
    let authorize = |builder: reqwest::RequestBuilder| match request.api_key.as_deref() {
        Some(key) if !key.trim().is_empty() => builder.bearer_auth(key.trim()),
        _ => builder,
    };
    let base = endpoint.as_str().trim_end_matches('/').to_string();
    let id = &transport.checkpoint_id;
    let status: ReceiveStatus = decode(
        authorize(client.post(format!("{base}/api/v1/model-transport")))
            .timeout(Duration::from_secs(120))
            .json(&transport)
            .send()
            .await,
    )
    .await?;
    update(progress, |p| {
        p.state = "uploading";
        p.total_bytes = status.total_bytes;
        p.skipped_bytes = status.received_bytes;
        p.message = if status.received_bytes > 0 {
            "Resuming; chunks the coordinator already has are skipped".into()
        } else {
            "Sending verified 16 MiB chunks".into()
        };
    });
    let root = library.package_dir(id).map_err(CondenserError::Config)?;
    for missing in &status.missing {
        let file = transport
            .files
            .get(missing.file as usize)
            .ok_or_else(|| failure("receiver asked for an unknown file"))?;
        let chunk = file
            .chunks
            .get(missing.chunk as usize)
            .ok_or_else(|| failure("receiver asked for an unknown chunk"))?;
        let path = root.join(&file.path);
        let (offset, length) = (chunk.offset, chunk.bytes as usize);
        let bytes = tokio::task::spawn_blocking(move || -> std::io::Result<Vec<u8>> {
            let mut input = std::fs::File::open(path)?;
            input.seek(SeekFrom::Start(offset))?;
            let mut buffer = vec![0; length];
            input.read_exact(&mut buffer)?;
            Ok(buffer)
        })
        .await
        .map_err(failure)??;
        let url = format!(
            "{base}/api/v1/model-transport/{id}/files/{}/chunks/{}",
            missing.file, missing.chunk
        );
        let mut attempt = 0;
        loop {
            attempt += 1;
            let sent = authorize(client.put(&url))
                .timeout(Duration::from_secs(600))
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(bytes.clone())
                .send()
                .await;
            match sent {
                Ok(response) if response.status().is_success() => break,
                Ok(response)
                    if attempt >= CHUNK_ATTEMPTS || response.status().is_client_error() =>
                {
                    return Err(remote_error(response).await);
                }
                Err(error) if attempt >= CHUNK_ATTEMPTS => return Err(failure(error)),
                _ => tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await,
            }
        }
        update(progress, |p| p.sent_bytes += chunk.bytes);
    }
    update(progress, |p| {
        p.state = "verifying";
        p.message = "The coordinator is verifying every byte before publishing".into();
    });
    // Verification runs in the background on the receiver; poll until it publishes or fails.
    loop {
        let response =
            authorize(client.post(format!("{base}/api/v1/model-transport/{id}/complete")))
                .timeout(Duration::from_secs(120))
                .send()
                .await
                .map_err(failure)?;
        if !poll(response.status()) {
            if response.status().is_success() {
                break;
            }
            return Err(remote_error(response).await);
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    update(progress, |p| {
        p.state = "loading";
        p.message = "Loading at the model's full context".into();
    });
    // Loading also runs in the background on the receiver; poll until it reports the model.
    let loaded: Value = loop {
        let response = authorize(client.post(format!("{base}/api/v1/model-transport/{id}/load")))
            .timeout(Duration::from_secs(120))
            .json(&json!({ "context_length": request.context_length }))
            .send()
            .await;
        match response {
            Ok(response) if poll(response.status()) => {
                tokio::time::sleep(Duration::from_secs(2)).await
            }
            response => break decode(response).await?,
        }
    };
    let model = loaded["model"]
        .as_str()
        .ok_or_else(|| failure("the coordinator did not report the loaded model"))?;
    let ollama_path = loaded["ollama_path"]
        .as_str()
        .filter(|path| path.starts_with('/') && !path.contains(".."))
        .ok_or_else(|| failure("the coordinator does not serve received models"))?;
    let mut provider = ProviderConfig::new(ProviderKind::Ollama);
    provider.model = model.into();
    provider.base_url = Some(format!("{base}{ollama_path}"));
    Ok(json!({
        "checkpoint": loaded["checkpoint"],
        "context_length": loaded["context_length"],
        "execution": loaded["execution"],
        "provider": provider,
        "transport": transport.summary(),
    }))
}

/// The receiver is still working (202) or busy with another model (409): ask again shortly.
fn poll(status: reqwest::StatusCode) -> bool {
    matches!(
        status,
        reqwest::StatusCode::ACCEPTED | reqwest::StatusCode::CONFLICT
    )
}

fn backend_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("HTTP client")
    })
}

/// Whether a model was created in the Ollama at `ollama_base` (used to re-register received
/// models after a coordinator restart).
pub async fn ollama_has_model(ollama_base: &str, model: &str) -> bool {
    let url = format!("{}/api/show", ollama_base.trim_end_matches('/'));
    backend_client()
        .post(url)
        .timeout(Duration::from_secs(30))
        .json(&json!({ "model": model }))
        .send()
        .await
        .is_ok_and(|response| response.status().is_success())
}

/// Forwards a JSON request to a model the receiver loaded into its local Ollama, streaming the
/// reply unchanged.
pub async fn forward_json(
    ollama_base: &str,
    path: &str,
    body: axum::body::Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let client = backend_client();
    let url = format!(
        "{}/{}",
        ollama_base.trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    match client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(response) => {
            let status = axum::http::StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();
            (
                status,
                [(axum::http::header::CONTENT_TYPE, content_type)],
                axum::body::Body::from_stream(response.bytes_stream()),
            )
                .into_response()
        }
        Err(error) => (
            axum::http::StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("The model host is not reachable: {error}")})),
        )
            .into_response(),
    }
}

fn update(progress: &SharedProgress, change: impl FnOnce(&mut PublishProgress)) {
    change(&mut progress.lock().expect("publish progress"));
}

async fn decode<T: serde::de::DeserializeOwned>(
    response: reqwest::Result<reqwest::Response>,
) -> Result<T> {
    let response = response.map_err(failure)?;
    if !response.status().is_success() {
        return Err(remote_error(response).await);
    }
    response.json().await.map_err(failure)
}

async fn remote_error(response: reqwest::Response) -> CondenserError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v["error"].as_str().map(str::to_owned))
        .unwrap_or(body);
    let hint = match status.as_u16() {
        401 => "Check the access token (the coordinator's KADMIUM_TRANSPORT_TOKEN). ",
        404 => "This address does not accept model uploads; check it, or update the coordinator. ",
        403 => "This coordinator does not receive models; set KADMIUM_TRANSPORT_TOKEN on it. ",
        409 => "The model is not loaded on the coordinator. ",
        413 => "The coordinator does not accept a model this large. ",
        _ => "",
    };
    failure(format!(
        "{hint}HTTP {status}: {}",
        detail.chars().take(400).collect::<String>()
    ))
}

fn failure(error: impl std::fmt::Display) -> CondenserError {
    CondenserError::provider("Kadmium", format!("Sending the model failed: {error}"))
}

/// HTTPS for anything public. Plain HTTP only reaches this computer or a private network.
pub(crate) fn remote_url(base: &str) -> Result<reqwest::Url> {
    let invalid = || {
        CondenserError::Config(
            "Enter a coordinator address such as https://coordinator.example:7848 or http://192.168.1.48:7848"
                .into(),
        )
    };
    let url = reqwest::Url::parse(base.trim()).map_err(|_| invalid())?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.host_str().is_none()
    {
        return Err(invalid());
    }
    let host = url.host_str().unwrap_or_default();
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let private = host
        .parse::<IpAddr>()
        .map(private_ip)
        .unwrap_or_else(|_| host == "localhost" || host.ends_with(".local"));
    match url.scheme() {
        "https" => Ok(url),
        "http" if private => Ok(url),
        "http" => Err(CondenserError::Config(
            "Use HTTPS for Kadmium endpoints outside your private network".into(),
        )),
        _ => Err(invalid()),
    }
}

fn private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        IpAddr::V6(ip) => ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00,
    }
}

#[cfg(test)]
mod tests {
    use super::remote_url;

    #[test]
    fn endpoints_require_https_outside_private_networks() {
        assert!(remote_url("https://api.kadmium.ai").is_ok());
        assert!(remote_url("http://192.168.1.48:7848").is_ok());
        assert!(remote_url("http://127.0.0.1:7848").is_ok());
        assert!(remote_url("http://studio.local:7848").is_ok());
        assert!(remote_url("http://api.kadmium.ai").is_err());
        assert!(remote_url("https://user:pw@api.kadmium.ai").is_err());
        assert!(remote_url("ftp://api.kadmium.ai").is_err());
    }
}
