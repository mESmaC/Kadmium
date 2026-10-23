//! Transparent inference gateway for a separately configured compatible backend.
//!
//! OpenAI routes require the corresponding native backend routes. Anthropic requests
//! require a native `/v1/messages` backend; this module performs no protocol translation.
//! Status describes configuration and admission capacity, not backend health.

use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::Stream;
use serde_json::{Value, json};
use std::{
    fmt,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const REQUEST_BODY_LIMIT: usize = 2 * 1024 * 1024;
static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Settings for a native OpenAI/Anthropic-compatible inference backend.
#[derive(Clone)]
pub struct InferenceConfig {
    /// HTTP(S) origin, or a path ending in `/v1`; credentials and queries are forbidden.
    pub base_url: Option<String>,
    /// Backend credential. Caller credentials are never forwarded.
    pub api_key: Option<String>,
    pub max_in_flight: usize,
    /// Total upstream request deadline, including streamed response consumption.
    pub timeout_secs: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            api_key: None,
            max_in_flight: 4,
            timeout_secs: 120,
        }
    }
}

impl fmt::Debug for InferenceConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Invalid URLs can contain credentials too; never format the raw URL.
        formatter
            .debug_struct("InferenceConfig")
            .field("base_url", &self.base_url.as_ref().map(|_| "<configured>"))
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("max_in_flight", &self.max_in_flight)
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

impl InferenceConfig {
    /// Validate before startup side effects. Errors never include configuration secrets.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_in_flight == 0 || self.max_in_flight > Semaphore::MAX_PERMITS {
            return Err("Inference max_in_flight must be a supported positive count".into());
        }
        if self.timeout_secs == 0
            || std::time::Instant::now()
                .checked_add(Duration::from_secs(self.timeout_secs))
                .is_none()
        {
            return Err("Inference timeout_secs must be a supported positive duration".into());
        }
        if let Some(url) = &self.base_url {
            normalize_base_url(url)?;
        }
        if let Some(key) = &self.api_key {
            if key.trim().is_empty() || key.chars().any(char::is_whitespace) {
                return Err("Inference API key must be nonempty and contain no whitespace".into());
            }
            HeaderValue::from_str(key)
                .map_err(|_| "Inference API key contains invalid header characters".to_owned())?;
            HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|_| "Inference API key contains invalid header characters".to_owned())?;
        }
        Ok(())
    }
}

fn normalize_base_url(input: &str) -> Result<reqwest::Url, String> {
    let invalid = || {
        "Inference base URL must be an absolute HTTP(S) origin or path ending in /v1, without credentials, query, or fragment".to_owned()
    };
    if input.trim() != input || input.chars().any(char::is_control) || input.contains('\\') {
        return Err(invalid());
    }
    // Check raw authority too: a URL parser may discard an empty userinfo component.
    let authority = input
        .split_once("://")
        .map(|(_, rest)| rest.split(['/', '?', '#']).next().unwrap_or_default())
        .ok_or_else(invalid)?;
    let mut url = reqwest::Url::parse(input).map_err(|_| invalid())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || authority.contains('@')
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid());
    }
    let path = url.path().trim_end_matches('/');
    let path = if path.is_empty() {
        "/v1".to_owned()
    } else if path.ends_with("/v1") {
        path.to_owned()
    } else {
        return Err(invalid());
    };
    url.set_path(&path);
    Ok(url)
}

struct InferenceState {
    base_url: Option<reqwest::Url>,
    bearer_key: Option<HeaderValue>,
    native_key: Option<HeaderValue>,
    max_in_flight: usize,
    timeout_secs: u64,
    permits: Arc<Semaphore>,
    client: reqwest::Client,
}

/// Build inference routes with independent state, ready to merge into a gateway router.
pub fn router(config: InferenceConfig) -> Result<Router, String> {
    config.validate()?;
    let base_url = config
        .base_url
        .as_deref()
        .map(normalize_base_url)
        .transpose()?;
    let native_key = config.api_key.as_ref().map(|key| {
        let mut header = HeaderValue::from_str(key).expect("validated key");
        header.set_sensitive(true);
        header
    });
    let bearer_key = config.api_key.as_ref().map(|key| {
        let mut header = HeaderValue::from_str(&format!("Bearer {key}")).expect("validated key");
        header.set_sensitive(true);
        header
    });
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(config.timeout_secs))
        .connect_timeout(Duration::from_secs(config.timeout_secs.min(10)))
        .pool_max_idle_per_host(config.max_in_flight)
        .build()
        .map_err(|_| "Could not initialize inference backend HTTP client".to_owned())?;
    let state = Arc::new(InferenceState {
        base_url,
        bearer_key,
        native_key,
        max_in_flight: config.max_in_flight,
        timeout_secs: config.timeout_secs,
        permits: Arc::new(Semaphore::new(config.max_in_flight)),
        client,
    });
    Ok(Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/messages", post(messages))
        .route("/api/v1/inference/status", get(status))
        .with_state(state))
}

#[derive(Clone, Copy)]
enum Endpoint {
    Models,
    Chat,
    Embeddings,
    Messages,
}

impl Endpoint {
    fn path(self) -> &'static str {
        match self {
            Self::Models => "/models",
            Self::Chat => "/chat/completions",
            Self::Embeddings => "/embeddings",
            Self::Messages => "/messages",
        }
    }

    fn is_anthropic(self) -> bool {
        matches!(self, Self::Messages)
    }
}

async fn models(State(state): State<Arc<InferenceState>>, request: Request) -> Response {
    forward(state, request, Endpoint::Models).await
}

async fn chat(State(state): State<Arc<InferenceState>>, request: Request) -> Response {
    forward(state, request, Endpoint::Chat).await
}

async fn embeddings(State(state): State<Arc<InferenceState>>, request: Request) -> Response {
    forward(state, request, Endpoint::Embeddings).await
}

async fn messages(State(state): State<Arc<InferenceState>>, request: Request) -> Response {
    forward(state, request, Endpoint::Messages).await
}

async fn status(State(state): State<Arc<InferenceState>>) -> Response {
    let available = state.permits.available_permits();
    let configured = state.base_url.is_some();
    let mut response = Json(json!({
        "configured": configured,
        "mode": if configured { "compatible_backend" } else { "unconfigured" },
        "endpoint": state.base_url.as_ref().map(reqwest::Url::as_str),
        "max_in_flight": state.max_in_flight,
        "available_slots": available,
        "in_flight": state.max_in_flight.saturating_sub(available),
        "timeout_secs": state.timeout_secs,
        "anthropic_mode": "native_passthrough",
        "readiness": "configuration_only"
    }))
    .into_response();
    set_request_id(&mut response, &new_request_id());
    response
}

fn new_request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("kdm-{:x}-{nanos:x}-{sequence:x}", std::process::id())
}

fn set_request_id(response: &mut Response, request_id: &str) {
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(request_id).expect("generated ASCII request ID"),
    );
}

fn error_response(
    endpoint: Endpoint,
    status: StatusCode,
    code: &'static str,
    message: &'static str,
    param: Option<&str>,
    request_id: &str,
) -> Response {
    let error_type = match status {
        StatusCode::BAD_REQUEST | StatusCode::PAYLOAD_TOO_LARGE => "invalid_request_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        _ => "api_error",
    };
    let mut value = json!({
        "error": { "message": message, "type": error_type, "code": code, "param": param },
        "request_id": request_id
    });
    if endpoint.is_anthropic() {
        value["type"] = json!("error");
    }
    let mut response = (status, Json(value)).into_response();
    set_request_id(&mut response, request_id);
    response
}

async fn forward(state: Arc<InferenceState>, request: Request, endpoint: Endpoint) -> Response {
    let request_id = new_request_id();
    let Some(base_url) = &state.base_url else {
        return error_response(
            endpoint,
            StatusCode::SERVICE_UNAVAILABLE,
            "backend_not_configured",
            "No inference backend is configured.",
            None,
            &request_id,
        );
    };
    let permit = match Arc::clone(&state.permits).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let mut response = error_response(
                endpoint,
                StatusCode::TOO_MANY_REQUESTS,
                "admission_full",
                "The inference gateway is at capacity. Retry later.",
                None,
                &request_id,
            );
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
            return response;
        }
    };
    let (parts, body) = request.into_parts();
    let payload = if matches!(endpoint, Endpoint::Models) {
        None
    } else {
        let bytes = match tokio::time::timeout(
            Duration::from_secs(state.timeout_secs),
            to_bytes(body, REQUEST_BODY_LIMIT),
        )
        .await
        {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(error)) => {
                use std::error::Error;
                let too_large = error
                    .source()
                    .is_some_and(|source| source.is::<http_body_util::LengthLimitError>());
                return error_response(
                    endpoint,
                    if too_large {
                        StatusCode::PAYLOAD_TOO_LARGE
                    } else {
                        StatusCode::BAD_REQUEST
                    },
                    if too_large {
                        "request_too_large"
                    } else {
                        "invalid_json"
                    },
                    if too_large {
                        "Request body exceeds the 2 MiB limit."
                    } else {
                        "Could not read the JSON request body."
                    },
                    None,
                    &request_id,
                );
            }
            Err(_) => {
                return error_response(
                    endpoint,
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    "Timed out reading the request body.",
                    None,
                    &request_id,
                );
            }
        };
        let value = match serde_json::from_slice::<Value>(&bytes) {
            Ok(value) => value,
            Err(_) => {
                return error_response(
                    endpoint,
                    StatusCode::BAD_REQUEST,
                    "invalid_json",
                    "Request body must be valid JSON.",
                    None,
                    &request_id,
                );
            }
        };
        if let Err(problem) = validate_payload(endpoint, &value) {
            return error_response(
                endpoint,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                problem.message,
                Some(problem.param),
                &request_id,
            );
        }
        Some(value)
    };
    let mut upstream_url = base_url.clone();
    upstream_url.set_path(&format!("{}{}", base_url.path(), endpoint.path()));
    let mut upstream_request = if matches!(endpoint, Endpoint::Models) {
        state.client.get(upstream_url)
    } else {
        state
            .client
            .post(upstream_url)
            .json(payload.as_ref().expect("POST payload"))
    };
    if endpoint.is_anthropic() {
        if let Some(key) = &state.native_key {
            upstream_request = upstream_request.header("x-api-key", key.clone());
        }
        upstream_request = upstream_request.header(
            "anthropic-version",
            parts
                .headers
                .get("anthropic-version")
                .cloned()
                .unwrap_or_else(|| HeaderValue::from_static("2023-06-01")),
        );
        if let Some(beta) = parts.headers.get("anthropic-beta") {
            upstream_request = upstream_request.header("anthropic-beta", beta.clone());
        }
    } else if let Some(key) = &state.bearer_key {
        upstream_request = upstream_request.header(header::AUTHORIZATION, key.clone());
    }
    upstream_request = upstream_request.header("x-request-id", &request_id);
    let streaming = payload
        .as_ref()
        .is_some_and(|value| value.get("stream") == Some(&Value::Bool(true)));
    if streaming {
        upstream_request = upstream_request.header(header::ACCEPT, "text/event-stream");
    } else {
        upstream_request = upstream_request.header(header::ACCEPT, "application/json");
    }
    let upstream = match upstream_request.send().await {
        Ok(response) => response,
        Err(error) => {
            let timed_out = error.is_timeout();
            return error_response(
                endpoint,
                if timed_out {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                },
                if timed_out {
                    "upstream_timeout"
                } else {
                    "upstream_unavailable"
                },
                if timed_out {
                    "The inference backend timed out."
                } else {
                    "The inference backend is unavailable."
                },
                None,
                &request_id,
            );
        }
    };
    if !upstream.status().is_success() {
        let redirect = upstream.status().is_redirection();
        let mut response = error_response(
            endpoint,
            if redirect {
                StatusCode::BAD_GATEWAY
            } else {
                upstream.status()
            },
            if redirect {
                "upstream_redirect"
            } else {
                "upstream_error"
            },
            if redirect {
                "The inference backend returned a redirect; redirects are disabled."
            } else {
                "The inference backend rejected the request."
            },
            None,
            &request_id,
        );
        // Never echo the backend's diagnostic body, cookies, or authentication headers.
        copy_retry_after(upstream.headers(), response.headers_mut());
        return response;
    }
    if !valid_response_content_type(upstream.headers(), streaming) {
        return error_response(
            endpoint,
            StatusCode::BAD_GATEWAY,
            "invalid_upstream_response",
            "The inference backend returned an incompatible response content type.",
            None,
            &request_id,
        );
    }
    let status = upstream.status();
    let mut headers = HeaderMap::new();
    for name in [
        header::CONTENT_TYPE,
        header::CACHE_CONTROL,
        header::HeaderName::from_static("x-accel-buffering"),
    ] {
        if let Some(value) = upstream.headers().get(&name) {
            headers.insert(name, value.clone());
        }
    }
    if streaming {
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    }
    copy_retry_after(upstream.headers(), &mut headers);
    let stream = UpstreamBody {
        inner: Box::pin(upstream.bytes_stream()),
        permit: Some(permit),
    };
    let mut response = Body::from_stream(stream).into_response();
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    set_request_id(&mut response, &request_id);
    response
}

fn valid_response_content_type(headers: &HeaderMap, streaming: bool) -> bool {
    let Some(content_type) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if streaming {
        media_type == "text/event-stream"
    } else {
        media_type == "application/json"
            || (media_type.starts_with("application/") && media_type.ends_with("+json"))
    }
}

fn copy_retry_after(source: &HeaderMap, target: &mut HeaderMap) {
    // Numeric seconds only: do not copy arbitrary text from an upstream error response.
    if let Some(value) = source.get(header::RETRY_AFTER)
        && value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .is_some()
    {
        target.insert(header::RETRY_AFTER, value.clone());
    }
}

/// Pull-based transport with no producer task or unbounded queue. Dropping the downstream
/// body drops the upstream stream/connection and its admission permit together.
struct UpstreamBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    permit: Option<OwnedSemaphorePermit>,
}

impl Stream for UpstreamBody {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(context) {
            Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(bytes))),
            Poll::Ready(Some(Err(_))) => {
                self.permit.take();
                // Headers have already been sent; terminate the body without leaking a URL
                // or inventing a protocol-specific token/error event.
                Poll::Ready(Some(Err(std::io::Error::other(
                    "Inference backend response interrupted",
                ))))
            }
            Poll::Ready(None) => {
                self.permit.take();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

struct ValidationProblem {
    param: &'static str,
    message: &'static str,
}

fn invalid(param: &'static str, message: &'static str) -> ValidationProblem {
    ValidationProblem { param, message }
}

fn nonempty_string(value: &Value) -> bool {
    value.as_str().is_some_and(|value| !value.trim().is_empty())
}

fn content_blocks(value: &Value) -> bool {
    value.as_array().is_some_and(|blocks| {
        !blocks.is_empty()
            && blocks.iter().all(|block| {
                block
                    .as_object()
                    .is_some_and(|block| block.get("type").is_some_and(nonempty_string))
            })
    })
}

fn positive_integer(value: &Value) -> bool {
    value.as_u64().is_some_and(|value| value > 0)
}

fn token_array(values: &[Value]) -> bool {
    !values.is_empty()
        && values.iter().all(|value| {
            value
                .as_u64()
                .is_some_and(|token| u32::try_from(token).is_ok())
        })
}

fn valid_embedding_input(value: &Value) -> bool {
    if value.is_string() {
        return nonempty_string(value);
    }
    let Some(values) = value.as_array() else {
        return false;
    };
    !values.is_empty()
        && (values.iter().all(nonempty_string)
            || token_array(values)
            || values
                .iter()
                .all(|value| value.as_array().is_some_and(|tokens| token_array(tokens))))
}

fn validate_payload(endpoint: Endpoint, value: &Value) -> Result<(), ValidationProblem> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("body", "Request body must be a JSON object."))?;
    if !object.get("model").is_some_and(nonempty_string) {
        return Err(invalid("model", "model must be a nonempty string."));
    }
    if let Some(stream) = object.get("stream")
        && !stream.is_boolean()
    {
        return Err(invalid("stream", "stream must be a boolean."));
    }
    if matches!(endpoint, Endpoint::Embeddings) {
        if !object.get("input").is_some_and(valid_embedding_input) {
            return Err(invalid(
                "input",
                "input must be a nonempty string, string array, token array, or array of token arrays; mixed shapes are invalid.",
            ));
        }
        if let Some(dimensions) = object.get("dimensions")
            && !positive_integer(dimensions)
        {
            return Err(invalid(
                "dimensions",
                "dimensions must be a positive integer.",
            ));
        }
        if let Some(encoding) = object.get("encoding_format")
            && !matches!(encoding.as_str(), Some("float" | "base64"))
        {
            return Err(invalid(
                "encoding_format",
                "encoding_format must be float or base64.",
            ));
        }
        return Ok(());
    }
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .ok_or_else(|| {
            invalid(
                "messages",
                "messages must be a nonempty array of message objects.",
            )
        })?;
    for message in messages {
        let message = message.as_object().ok_or_else(|| {
            invalid(
                "messages",
                "Each message must be an object with a role and content.",
            )
        })?;
        let role = message.get("role").and_then(Value::as_str);
        let valid_role = if endpoint.is_anthropic() {
            matches!(role, Some("user" | "assistant"))
        } else {
            matches!(
                role,
                Some("system" | "developer" | "user" | "assistant" | "tool" | "function")
            )
        };
        if !valid_role {
            return Err(invalid(
                "messages",
                "Each message must have a supported role.",
            ));
        }
        let content = message.get("content");
        let assistant_tool_call = !endpoint.is_anthropic()
            && role == Some("assistant")
            && (message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty() && calls.iter().all(Value::is_object))
                || message.get("function_call").is_some_and(Value::is_object));
        let valid_content = content
            .is_some_and(|content| content.is_string() || content_blocks(content))
            || (assistant_tool_call && content.is_none_or(Value::is_null));
        if !valid_content {
            return Err(invalid(
                "messages",
                "Message content must be a string or an array of typed content blocks; assistant tool calls may omit content.",
            ));
        }
    }
    for (field, param) in [
        ("max_tokens", "max_tokens"),
        ("max_completion_tokens", "max_completion_tokens"),
    ] {
        if let Some(value) = object.get(field)
            && !positive_integer(value)
        {
            return Err(invalid(param, "Token limits must be positive integers."));
        }
    }
    if endpoint.is_anthropic() {
        if !object.get("max_tokens").is_some_and(positive_integer) {
            return Err(invalid(
                "max_tokens",
                "Anthropic messages require a positive integer max_tokens.",
            ));
        }
        if let Some(system) = object.get("system")
            && !(system.is_string() || content_blocks(system))
        {
            return Err(invalid(
                "system",
                "system must be a string or an array of typed content blocks.",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_response_type_must_match_the_requested_protocol() {
        let mut headers = HeaderMap::new();
        assert!(!valid_response_content_type(&headers, false));
        assert!(!valid_response_content_type(&headers, true));
        for content_type in [
            "application/json",
            "application/json; charset=utf-8",
            "Application/Problem+JSON",
        ] {
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
            assert!(valid_response_content_type(&headers, false));
            assert!(!valid_response_content_type(&headers, true));
        }
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );
        assert!(valid_response_content_type(&headers, true));
        assert!(!valid_response_content_type(&headers, false));
        for content_type in ["text/html", "text/plain", "application/octet-stream"] {
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
            assert!(!valid_response_content_type(&headers, false));
            assert!(!valid_response_content_type(&headers, true));
        }
    }

    #[test]
    fn base_url_normalization_and_rejection() {
        for (input, expected) in [
            ("http://localhost:8080", "http://localhost:8080/v1"),
            ("https://backend.example/", "https://backend.example/v1"),
            ("https://backend.example/v1/", "https://backend.example/v1"),
            (
                "https://backend.example/prefix/v1",
                "https://backend.example/prefix/v1",
            ),
        ] {
            assert_eq!(normalize_base_url(input).unwrap().as_str(), expected);
        }
        for invalid in [
            "",
            "localhost:8080",
            "ftp://backend.example/v1",
            "https://user:secret@backend.example/v1",
            "https://@backend.example/v1",
            "https://backend.example/v1?key=secret",
            "https://backend.example/v1#secret",
            "https://backend.example/completions",
            " https://backend.example/v1",
            "https://backend.example\\@evil.example/v1",
        ] {
            assert!(normalize_base_url(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn configuration_is_validated_and_debug_redacted() {
        assert!(InferenceConfig::default().validate().is_ok());
        let config = InferenceConfig {
            base_url: Some("https://user:url-secret@backend.example".into()),
            api_key: Some("api-secret".into()),
            ..Default::default()
        };
        let diagnostic = format!("{config:?} {}", config.validate().unwrap_err());
        assert!(!diagnostic.contains("url-secret"));
        assert!(!diagnostic.contains("api-secret"));
        assert!(
            InferenceConfig {
                max_in_flight: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            InferenceConfig {
                max_in_flight: usize::MAX,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            InferenceConfig {
                timeout_secs: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        for key in ["", " ", "secret\r\ninjected: value"] {
            assert!(
                InferenceConfig {
                    api_key: Some(key.into()),
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn embedding_shapes_are_homogeneous_and_dimensions_are_not_clamped() {
        for input in [
            json!("hello"),
            json!(["hello", "world"]),
            json!([1, 2]),
            json!([[1, 2], [3]]),
        ] {
            assert!(
                validate_payload(
                    Endpoint::Embeddings,
                    &json!({"model":"embeddings", "input":input, "dimensions":4096})
                )
                .is_ok()
            );
        }
        for input in [
            json!(""),
            json!([]),
            json!(["text", 1]),
            json!([[1], "text"]),
            json!([[]]),
            json!([-1]),
            json!([1.5]),
            json!([null]),
            json!([[1], ["x"]]),
        ] {
            assert!(
                validate_payload(
                    Endpoint::Embeddings,
                    &json!({"model":"embeddings", "input":input})
                )
                .is_err()
            );
        }
        for dimensions in [json!(0), json!(-1), json!(1.5), json!("16")] {
            assert!(
                validate_payload(
                    Endpoint::Embeddings,
                    &json!({"model":"embeddings", "input":"hello", "dimensions":dimensions})
                )
                .is_err()
            );
        }
    }

    #[test]
    fn conversations_preserve_tools_multimodal_blocks_and_unknown_controls() {
        let value = json!({
            "model":"real-model", "stream":true,
            "messages":[
                {"role":"developer", "content":"policy"},
                {"role":"user", "content":[{"type":"text", "text":"hello"}, {"type":"image_url", "image_url":{"url":"data:image/png;base64,AA"}}]},
                {"role":"assistant", "content":null, "tool_calls":[{"id":"call-1", "type":"function", "function":{"name":"weather", "arguments":"{}"}}]},
                {"role":"tool", "tool_call_id":"call-1", "content":"sunny"}
            ],
            "temperature":0.5, "tools":[{"type":"function", "function":{"name":"weather"}}],
            "vendor_option":{"keep":true}
        });
        let before = value.clone();
        assert!(validate_payload(Endpoint::Chat, &value).is_ok());
        assert_eq!(value, before);
        for value in [
            json!([]),
            json!({}),
            json!({"model":"m", "messages":[]}),
            json!({"model":"m", "messages":["hello"]}),
            json!({"model":"m", "messages":[{"role":"user", "content":42}]}),
            json!({"model":"m", "messages":[{"role":"user", "content":"hello"}], "stream":"true"}),
        ] {
            assert!(validate_payload(Endpoint::Chat, &value).is_err());
        }
    }

    #[test]
    fn anthropic_uses_native_messages_contract() {
        assert!(validate_payload(Endpoint::Messages, &json!({"model":"claude", "max_tokens":32, "system":[{"type":"text", "text":"policy"}], "messages":[{"role":"user", "content":[{"type":"text", "text":"hello"}]}], "stop_sequences":["END"]})).is_ok());
        assert!(
            validate_payload(
                Endpoint::Messages,
                &json!({"model":"claude", "messages":[{"role":"user", "content":"hello"}]})
            )
            .is_err()
        );
        assert!(validate_payload(Endpoint::Messages, &json!({"model":"claude", "max_tokens":32, "messages":[{"role":"system", "content":"hello"}]})).is_err());
    }
}
