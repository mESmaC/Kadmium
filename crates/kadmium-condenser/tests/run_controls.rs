use axum::extract::Query;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use kadmium_condenser::server::{ServerOptions, bind};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

async fn serve_mock(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, task)
}

#[tokio::test]
async fn cancellation_returns_one_final_report_and_releases_run_identity() {
    let started = Arc::new(tokio::sync::Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let (provider, upstream) = serve_mock(Router::new().route(
        "/v1/chat/completions",
        post({
            let started = started.clone();
            let calls = calls.clone();
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                started.notify_one();
                std::future::pending::<Json<Value>>()
            }
        }),
    ))
    .await;
    let (address, server) = bind(ServerOptions {
        port: 0,
        plugin_roots: vec![],
        ..ServerOptions::default()
    })
    .await
    .unwrap();
    let server = tokio::spawn(server);
    let client = reqwest::Client::new();
    let payload = json!({
        "run_id": "cancel-me",
        "provider": { "kind": "openai", "model": "test-model", "base_url": provider },
        "messages": [{"role": "user", "content": "Keep this prompt"}],
        "max_run_seconds": 10
    });
    let response = client
        .post(format!("http://{address}/api/v1/chat/stream"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    tokio::time::timeout(Duration::from_secs(3), started.notified())
        .await
        .unwrap();
    let duplicate = client
        .post(format!("http://{address}/api/v1/chat/stream"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(duplicate.status(), 400);
    let cancel = client
        .post(format!("http://{address}/api/v1/runs/cancel-me/cancel"))
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 200);
    let stream = tokio::time::timeout(Duration::from_secs(3), response.text())
        .await
        .unwrap()
        .unwrap();
    let finished: Vec<Value> = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event"] == "finished")
        .collect();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0]["report"]["status"], "cancelled");
    assert_eq!(
        finished[0]["report"]["messages"][0]["content"],
        "Keep this prompt"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let mut budgeted = payload;
    budgeted["max_output_tokens"] = json!(0);
    let next: Value = client
        .post(format!("http://{address}/api/v1/chat"))
        .json(&budgeted)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(next["status"], "budget_exceeded");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        client
            .post(format!("http://{address}/api/v1/runs/cancel-me/cancel"))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    upstream.abort();
    server.abort();
}

#[tokio::test]
async fn both_chat_endpoints_enforce_tool_selection_before_inference() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (provider, upstream) = serve_mock(Router::new().route("/v1/chat/completions", post({
        let calls = calls.clone();
        move |Json(request): Json<Value>| {
            calls.fetch_add(1, Ordering::SeqCst);
            async move {
                assert!(request.get("tools").is_none_or(|tools| tools.as_array().unwrap().is_empty()));
                Json(json!({"model":"test", "choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}))
            }
        }
    }))).await;
    let (address, server) = bind(ServerOptions {
        port: 0,
        plugin_roots: vec![],
        ..ServerOptions::default()
    })
    .await
    .unwrap();
    let server = tokio::spawn(server);
    let client = reqwest::Client::new();
    for streaming in [false, true] {
        for (enabled, allowed, rejected) in [
            (true, json!(["missing__tool"]), true),
            (true, json!([]), false),
            (false, json!(["missing__tool"]), false),
        ] {
            let payload = json!({
                "provider": {"kind":"openai", "model":"test", "base_url":provider},
                "messages":[{"role":"user","content":"hello"}],
                "enable_tools":enabled, "allowed_tools":allowed
            });
            let path = if streaming { "chat/stream" } else { "chat" };
            let response = client
                .post(format!("http://{address}/api/v1/{path}"))
                .json(&payload)
                .send()
                .await
                .unwrap();
            if rejected && !streaming {
                assert_eq!(response.status(), 400);
            } else {
                assert_eq!(response.status(), 200);
            }
            let body = response.text().await.unwrap();
            if rejected {
                assert!(body.contains("exact plugin__tool name"));
            } else {
                assert!(body.contains("completed"));
            }
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    upstream.abort();
    server.abort();
}

#[tokio::test]
async fn compute_bridge_preserves_selection_auth_and_upstream_failures() {
    let app = Router::new()
        .route("/fabric/api/v1/inference/workspace", get(|headers: HeaderMap, Query(query): Query<HashMap<String, String>>| async move {
            assert_eq!(headers.get("authorization").unwrap(), "Bearer test-only-key");
            assert_eq!(query.get("model_id").unwrap(), "model/with spaces");
            Json(json!({ "schema_version": 1, "plan": {"model_id": query["model_id"]}, "catalog": [], "nodes": [] }))
        }))
        .route("/redirect/api/v1/inference/workspace", get(|| async {
            axum::response::Redirect::temporary("/must-not-follow")
        }))
        .route("/large/api/v1/inference/workspace", get(|| async { "x".repeat(2 * 1024 * 1024 + 1) }));
    let (coordinator, upstream) = serve_mock(app).await;
    let (address, server) = bind(ServerOptions {
        port: 0,
        plugin_roots: vec![],
        ..ServerOptions::default()
    })
    .await
    .unwrap();
    let server = tokio::spawn(server);
    let client = reqwest::Client::new();
    let inspect = format!("http://{address}/api/v1/compute/inspect");
    let snapshot: Value = client.post(&inspect).json(&json!({
        "base_url": format!("{coordinator}/fabric/"), "model_id": "model/with spaces", "api_key": "test-only-key"
    })).send().await.unwrap().json().await.unwrap();
    assert_eq!(snapshot["plan"]["model_id"], "model/with spaces");
    for path in ["redirect", "large"] {
        let response = client
            .post(&inspect)
            .json(&json!({
                "base_url": format!("{coordinator}/{path}"), "api_key": "test-only-key"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 502);
        assert!(!response.text().await.unwrap().contains("test-only-key"));
    }
    upstream.abort();
    server.abort();
}
