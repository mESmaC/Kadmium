//! Protocol checks use real HTTP sockets and a controlled backend; no GPU or external service.

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::any,
};
use kadmium_gateway::inference_api::{InferenceConfig, router};
use serde_json::{Value, json};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::{Notify, mpsc};

const BACKEND_KEY: &str = "backend-secret-never-return";
const OPENAI_FIRST: &str = "data: {\"id\":\"upstream-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello \\\"世界\\\"\\n\"},\"finish_reason\":null}]}\n\n";
const OPENAI_LAST: &str = "data: {\"id\":\"upstream-chat\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
const ANTHROPIC_FIRST: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"upstream-message\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"test-model\",\"content\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n";
const ANTHROPIC_LAST: &str = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Native reply\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

struct Captured {
    path: String,
    headers: HeaderMap,
    body: Value,
}

#[derive(Clone)]
struct Backend {
    requests: mpsc::UnboundedSender<Captured>,
    finish_stream: Arc<Notify>,
}

async fn backend(
    State(state): State<Backend>,
    uri: Uri,
    headers: HeaderMap,
    bytes: Bytes,
) -> Response {
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let _ = state.requests.send(Captured {
        path: uri.path().into(),
        headers,
        body: body.clone(),
    });
    if body["model"] == "upstream-error" {
        return (
            StatusCode::UNAUTHORIZED,
            [("set-cookie", "secret=session"), ("retry-after", "7")],
            format!("provider diagnostics containing {BACKEND_KEY}"),
        )
            .into_response();
    }
    if body["model"] == "upstream-redirect" {
        return (
            StatusCode::TEMPORARY_REDIRECT,
            [("location", "http://127.0.0.1:1/do-not-follow")],
        )
            .into_response();
    }
    if body["model"] == "upstream-html" {
        return (
            [("content-type", "text/html")],
            format!("<html>private backend error {BACKEND_KEY}</html>"),
        )
            .into_response();
    }
    if body["stream"] == true {
        let native_anthropic = uri.path() == "/v1/messages";
        let first = if native_anthropic {
            ANTHROPIC_FIRST
        } else {
            OPENAI_FIRST
        };
        let last = if native_anthropic {
            ANTHROPIC_LAST
        } else {
            OPENAI_LAST
        };
        let gate = state.finish_stream;
        let chunks = futures_util::stream::unfold(0, move |index| {
            let gate = gate.clone();
            async move {
                match index {
                    0 => Some((Ok::<_, Infallible>(Bytes::from_static(first.as_bytes())), 1)),
                    1 => {
                        gate.notified().await;
                        Some((Ok(Bytes::from_static(last.as_bytes())), 2))
                    }
                    _ => None,
                }
            }
        });
        return Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("set-cookie", "must-not-propagate=1")
            .body(Body::from_stream(chunks))
            .unwrap();
    }
    let response = match uri.path() {
        "/v1/models" => {
            json!({"object":"list","data":[{"id":"test-model","object":"model","created":1,"owned_by":"test"}]})
        }
        "/v1/embeddings" => {
            json!({"object":"list","model":"test-model","data":[{"object":"embedding","index":0,"embedding":[0.25,-0.5]}],"usage":{"prompt_tokens":3,"total_tokens":3}})
        }
        "/v1/messages" => {
            json!({"id":"upstream-message","type":"message","role":"assistant","model":"test-model","content":[{"type":"text","text":"Native reply"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":2}})
        }
        _ => {
            json!({"id":"upstream-chat","object":"chat.completion","created":1,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"Hello \"世界\"\n"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5},"provider_extension":{"kept":true}})
        }
    };
    axum::Json(response).into_response()
}

struct Server {
    base: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(app: Router) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Server { base, task }
}

async fn fixture(
    max_in_flight: usize,
) -> (
    Server,
    Server,
    mpsc::UnboundedReceiver<Captured>,
    Arc<Notify>,
) {
    let (requests, captured) = mpsc::unbounded_channel();
    let finish_stream = Arc::new(Notify::new());
    let upstream = serve(
        Router::new()
            .route("/v1/{*path}", any(backend))
            .with_state(Backend {
                requests,
                finish_stream: finish_stream.clone(),
            }),
    )
    .await;
    let gateway = serve(
        router(InferenceConfig {
            base_url: Some(format!("{}/v1", upstream.base)),
            api_key: Some(BACKEND_KEY.into()),
            max_in_flight,
            timeout_secs: 5,
        })
        .unwrap(),
    )
    .await;
    (gateway, upstream, captured, finish_stream)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap()
}

fn chat(model: &str) -> Value {
    json!({"model":model,"messages":[{"role":"user","content":"hello"}]})
}

async fn read_first_event(response: &mut reqwest::Response, expected: &str) {
    let mut received = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), async {
        while received.len() < expected.len() {
            received
                .extend_from_slice(&response.chunk().await.unwrap().expect("stream ended early"));
        }
    })
    .await
    .expect("gateway buffered a stream waiting for backend completion");
    assert_eq!(received, expected.as_bytes());
}

#[tokio::test]
async fn preserves_conversation_controls_native_protocols_and_configured_auth() {
    let (gateway, _upstream, mut requests, _) = fixture(4).await;
    let client = client();
    let models = client
        .get(format!("{}/v1/models", gateway.base))
        .send()
        .await
        .unwrap();
    assert_eq!(models.status(), 200);
    let listed: Value = models.json().await.unwrap();
    assert_eq!(listed["data"][0]["id"], "test-model");
    assert_eq!(requests.recv().await.unwrap().path, "/v1/models");

    let conversation = json!({
        "model":"test-model", "messages":[
            {"role":"system","content":"Preserve all turns"},
            {"role":"user","content":"First turn"},
            {"role":"assistant","content":"Previous reply"},
            {"role":"user","content":[{"type":"text","text":"Hello \"世界\"\n"}]}
        ], "max_tokens":17,"temperature":0.25,"top_p":0.8,"seed":42,"stop":["END"],
        "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
        "tool_choice":"auto","response_format":{"type":"json_object"},"provider_option":{"enabled":true}
    });
    let response = client
        .post(format!("{}/v1/chat/completions", gateway.base))
        .bearer_auth("caller-key-do-not-forward")
        .header("x-api-key", "caller-other-key")
        .json(&conversation)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().contains_key("x-request-id"));
    let response: Value = response.json().await.unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "Hello \"世界\"\n"
    );
    assert_eq!(response["usage"]["total_tokens"], 5);
    assert_eq!(response["provider_extension"]["kept"], true);
    let sent = requests.recv().await.unwrap();
    assert_eq!(sent.body, conversation);
    assert_eq!(
        sent.headers["authorization"],
        format!("Bearer {BACKEND_KEY}")
    );
    assert_ne!(
        sent.headers.get("x-api-key").and_then(|v| v.to_str().ok()),
        Some("caller-other-key")
    );

    let embedding = json!({"model":"test-model","input":["one","two"],"dimensions":2,"encoding_format":"float"});
    let response = client
        .post(format!("{}/v1/embeddings", gateway.base))
        .json(&embedding)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response: Value = response.json().await.unwrap();
    assert_eq!(response["data"][0]["embedding"], json!([0.25, -0.5]));
    assert_eq!(requests.recv().await.unwrap().body, embedding);

    let native = json!({"model":"test-model","max_tokens":12,"system":[{"type":"text","text":"native system"}],"messages":[{"role":"user","content":[{"type":"text","text":"native input"}]}],"stop_sequences":["END"]});
    let response = client
        .post(format!("{}/v1/messages", gateway.base))
        .header("x-api-key", "caller-key")
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "test-beta")
        .json(&native)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response: Value = response.json().await.unwrap();
    assert_eq!(response["stop_reason"], "end_turn");
    assert_eq!(response["usage"]["input_tokens"], 3);
    let sent = requests.recv().await.unwrap();
    assert_eq!(sent.body, native);
    assert_eq!(sent.headers["x-api-key"], BACKEND_KEY);
    assert_eq!(sent.headers["anthropic-version"], "2023-06-01");
    assert_eq!(sent.headers["anthropic-beta"], "test-beta");
}

#[tokio::test]
async fn streams_native_sse_before_completion_and_holds_admission_until_body_finishes() {
    for (path, first, last) in [
        ("chat/completions", OPENAI_FIRST, OPENAI_LAST),
        ("messages", ANTHROPIC_FIRST, ANTHROPIC_LAST),
    ] {
        let (gateway, _upstream, _requests, finish) = fixture(1).await;
        let client = client();
        let mut payload = chat("test-model");
        payload["stream"] = true.into();
        payload["max_tokens"] = 20.into();
        let mut response = client
            .post(format!("{}/v1/{path}", gateway.base))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("text/event-stream")
        );
        assert!(!response.headers().contains_key("set-cookie"));
        assert_eq!(response.headers()["cache-control"], "no-cache");
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        read_first_event(&mut response, first).await;
        let busy = client
            .get(format!("{}/v1/models", gateway.base))
            .send()
            .await
            .unwrap();
        assert_eq!(busy.status(), 429);
        assert!(busy.headers().contains_key("retry-after"));
        finish.notify_one();
        let remaining = response.text().await.unwrap();
        assert_eq!(remaining, last);
        for data in format!("{first}{remaining}")
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
        {
            if data != "[DONE]" {
                serde_json::from_str::<Value>(data).unwrap();
            }
        }
        assert_eq!(
            client
                .get(format!("{}/v1/models", gateway.base))
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
    }
}

#[tokio::test]
async fn downstream_cancellation_releases_stream_permit() {
    let (gateway, _upstream, _requests, _finish) = fixture(1).await;
    let client = client();
    let mut payload = chat("test-model");
    payload["stream"] = true.into();
    let mut response = client
        .post(format!("{}/v1/chat/completions", gateway.base))
        .json(&payload)
        .send()
        .await
        .unwrap();
    response.chunk().await.unwrap().unwrap();
    drop(response);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let response = client
                .get(format!("{}/v1/models", gateway.base))
                .send()
                .await
                .unwrap();
            if response.status() == 200 {
                break;
            }
            assert_eq!(response.status(), 429);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("cancelled downstream retained admission permit");
}

#[tokio::test]
async fn errors_are_typed_bounded_and_do_not_expose_provider_secrets() {
    let unconfigured = serve(router(InferenceConfig::default()).unwrap()).await;
    let client = client();
    let response = client
        .post(format!("{}/v1/chat/completions", unconfigured.base))
        .json(&chat("test-model"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    assert_eq!(
        response.json::<Value>().await.unwrap()["error"]["code"],
        "backend_not_configured"
    );
    let status: Value = client
        .get(format!("{}/api/v1/inference/status", unconfigured.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["configured"], false);

    let (gateway, _upstream, mut requests, _) = fixture(2).await;
    for (body, code) in [("{".to_string(),400), (json!({"model":"test-model","messages":[]}).to_string(),400), (json!({"model":"test-model","messages":[{"role":"user","content":"x".repeat(2 * 1024 * 1024)}]}).to_string(),413)] {
        let response = client.post(format!("{}/v1/chat/completions", gateway.base)).header("content-type","application/json").body(body).send().await.unwrap();
        assert_eq!(response.status(), code);
        assert!(response.json::<Value>().await.unwrap()["error"]["message"].is_string());
    }
    assert!(
        requests.try_recv().is_err(),
        "rejected input reached backend"
    );

    let mut ids = Vec::new();
    for model in ["upstream-error", "upstream-redirect"] {
        let response = client
            .post(format!("{}/v1/chat/completions", gateway.base))
            .json(&chat(model))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if model == "upstream-error" { 401 } else { 502 }
        );
        assert!(!response.headers().contains_key("set-cookie"));
        assert!(!response.headers().contains_key("location"));
        ids.push(
            response.headers()["x-request-id"]
                .to_str()
                .unwrap()
                .to_string(),
        );
        let text = response.text().await.unwrap();
        assert!(!text.contains(BACKEND_KEY));
        assert!(!text.contains("provider diagnostics"));
        assert!(serde_json::from_str::<Value>(&text).unwrap()["error"]["code"].is_string());
    }
    assert_ne!(ids[0], ids[1]);
    for stream in [false, true] {
        let mut payload = chat("upstream-html");
        payload["stream"] = stream.into();
        let response = client
            .post(format!("{}/v1/chat/completions", gateway.base))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 502);
        let text = response.text().await.unwrap();
        assert!(!text.contains(BACKEND_KEY));
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap()["error"]["code"],
            "invalid_upstream_response"
        );
    }
    let status = client
        .get(format!("{}/api/v1/inference/status", gateway.base))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!status.contains(BACKEND_KEY));
}

#[tokio::test]
async fn coordinator_separates_actual_inference_from_reference_diagnostics() {
    let state = Arc::new(std::sync::Mutex::new(kadmium_gateway::GatewayState::new()));
    let gateway = serve(kadmium_gateway::build_router(state.clone())).await;
    let client = client();
    let response = client
        .get(format!("{}/v1/models", gateway.base))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    let response = client
        .post(format!("{}/v1/chat/completions", gateway.base))
        .json(&chat("test-model"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    let response = client.post(format!("{}/v1/messages", gateway.base))
        .json(&json!({"model":"test-model","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 503);
    assert_eq!(response.json::<Value>().await.unwrap()["type"], "error");
    assert_eq!(state.lock().unwrap().executed_inference_requests, 0);
    let reference = client
        .get(format!("{}/api/v1/reference/models", gateway.base))
        .send()
        .await
        .unwrap();
    assert_eq!(reference.status(), 200);
    assert!(
        !reference.json::<Value>().await.unwrap()["data"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}
