//! Explicit access to a Kadmium coordinator, independent of the
//! provider selected for a task. Credentials live only for this request.

use crate::server::ApiError;
use crate::{CondenserError, Result};
use axum::Json;
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;

const RESPONSE_LIMIT: usize = 2 * 1024 * 1024;

#[derive(Deserialize)]
pub(crate) struct InspectRequest {
    base_url: String,
    api_key: Option<String>,
    model_id: Option<String>,
}

fn workspace_url(base: &str, model_id: Option<&str>) -> Result<reqwest::Url> {
    let mut url = coordinator_url(base, "inference/workspace")?;
    if let Some(id) = model_id {
        url.query_pairs_mut().append_pair("model_id", id);
    }
    Ok(url)
}

fn coordinator_url(base: &str, route: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(base.trim())
        .map_err(|_| CondenserError::Config("Enter a valid coordinator URL".into()))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(CondenserError::Config(
            "Coordinator URL must use HTTP(S) without credentials, query parameters, or a fragment"
                .into(),
        ));
    }
    url.set_path(&format!(
        "{}/api/v1/{route}",
        url.path().trim_end_matches('/')
    ));
    Ok(url)
}

#[derive(Deserialize)]
pub(crate) struct JobRequest {
    base_url: String,
    api_key: Option<String>,
    job: Value,
}

pub(crate) async fn run_job(Json(request): Json<JobRequest>) -> Result<Json<Value>, ApiError> {
    let url = coordinator_url(&request.base_url, "compute/jobs")?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(75))
        .build()
        .map_err(|e| CondenserError::provider("Kadmium", e.to_string()))?;
    let mut upstream = client.post(url).json(&request.job);
    if let Some(key) = request.api_key.filter(|key| !key.is_empty()) {
        upstream = upstream.bearer_auth(key);
    }
    let response = upstream.send().await.map_err(|_| CondenserError::provider("Kadmium","Job connection failed or timed out. The request was not retried; dispatched work may still finish."))?;
    let report: Value =
        crate::provider::decode_json(crate::ProviderKind::Kadmium, response).await?;
    if !report["success"].is_boolean() || !report["traces"].is_array() {
        return Err(CondenserError::provider(
            "Kadmium",
            "Coordinator returned an unsupported job report",
        )
        .into());
    }
    Ok(Json(report))
}

pub(crate) async fn inspect(Json(request): Json<InspectRequest>) -> Result<Json<Value>, ApiError> {
    let url = workspace_url(&request.base_url, request.model_id.as_deref())?;
    // Do not forward the coordinator credential through an upstream redirect.
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|_| {
            CondenserError::provider("Kadmium", "Could not create the coordinator client")
        })?;
    let mut upstream = client.get(url);
    if let Some(key) = request.api_key.filter(|key| !key.is_empty()) {
        upstream = upstream.bearer_auth(key);
    }
    let mut response = upstream.send().await.map_err(|_| {
        CondenserError::provider(
            "Kadmium",
            "Coordinator unavailable. Check its URL and that Kadmium is running.",
        )
    })?;
    if !response.status().is_success() {
        return Err(CondenserError::provider("Kadmium", format!(
            "Coordinator returned HTTP {}. Check access and that it supports the inference workspace API.", response.status().as_u16()
        )).into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| CondenserError::provider("Kadmium", "Coordinator response was interrupted"))?
    {
        if bytes.len() + chunk.len() > RESPONSE_LIMIT {
            return Err(
                CondenserError::provider("Kadmium", "Coordinator response exceeds 2 MiB").into(),
            );
        }
        bytes.extend_from_slice(&chunk);
    }
    let snapshot: Value = serde_json::from_slice(&bytes).map_err(|_| {
        CondenserError::provider(
            "Kadmium",
            "Coordinator did not return a valid workspace snapshot",
        )
    })?;
    if snapshot["schema_version"] != 1
        || !snapshot["plan"].is_object()
        || !snapshot["nodes"].is_array()
        || !snapshot["catalog"].is_array()
    {
        return Err(CondenserError::provider(
            "Kadmium",
            "Unsupported coordinator workspace response",
        )
        .into());
    }
    Ok(Json(snapshot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinator_address_preserves_proxy_prefix_and_encodes_model_id() {
        let url = workspace_url("http://localhost:7848/fabric/", Some("a/b & c")).unwrap();
        assert_eq!(url.path(), "/fabric/api/v1/inference/workspace");
        assert_eq!(url.query_pairs().next().unwrap().1, "a/b & c");
        for base in [
            "file:///tmp/a",
            "https://key@example.com",
            "http://localhost/?key=secret",
            "http://localhost/#token",
        ] {
            assert!(workspace_url(base, None).is_err());
        }
    }
}
