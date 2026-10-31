use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use kadmium_condenser::{
    ChatRequest, Message, ProviderConfig, ProviderFactory, ProviderKind, SamplingParams,
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn providers_use_advertised_maxima_and_keep_explicit_limits() {
    let requests = Arc::new(Mutex::new(Vec::<Value>::new()));
    let app = Router::new()
        .route("/api/show", post(|| async { Json(json!({"model_info":{"general.architecture":"llama","llama.context_length":131072}})) }))
        .route("/v1/models", get(|| async { Json(json!({"data":[{"id":"model","max_output_tokens":64000}]})) }))
        .route("/v1/models/model", get(|| async { Json(json!({"id":"model","max_tokens":64000})) }))
        .route("/api/chat", post(|State(requests): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>| async move {
            requests.lock().unwrap().push(body);
            Json(json!({"model":"model","message":{"role":"assistant","content":"ok"},"done":true}))
        }))
        .route("/v1/chat/completions", post(|State(requests): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>| async move {
            requests.lock().unwrap().push(body);
            Json(json!({"model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}))
        }))
        .route("/v1/messages", post(|State(requests): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>| async move {
            requests.lock().unwrap().push(body);
            Json(json!({"model":"model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}))
        })).with_state(requests.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    for kind in [
        ProviderKind::Ollama,
        ProviderKind::OpenAi,
        ProviderKind::Anthropic,
        ProviderKind::Mcl4Coe,
    ] {
        let mut config = ProviderConfig::new(kind);
        config.model = "model".into();
        config.base_url = Some(base.clone());
        config.api_key = Some("fixture".into());
        let provider = ProviderFactory::build(config).unwrap();
        let mut request = ChatRequest {
            model: "model".into(),
            messages: vec![Message::user("Keep all of this input")],
            sampling: SamplingParams::default(),
            tools: vec![],
        };
        provider.complete(&request).await.unwrap();
        let body = requests.lock().unwrap().last().unwrap().clone();
        if kind == ProviderKind::Ollama {
            // The Modelfile parameter carries the full context; only an explicit choice is sent.
            assert!(body["options"].get("num_ctx").is_none(), "{body}");
            assert_eq!(body["options"]["num_predict"], -1);
            assert_eq!(body["truncate"], false);
            assert_eq!(body["shift"], false);
            request.sampling.context_window = Some(8192);
        } else {
            assert_eq!(body["max_tokens"], 64000);
        }
        assert_eq!(body["messages"][0]["content"], "Keep all of this input");
        request.sampling.max_tokens = Some(512);
        provider.complete(&request).await.unwrap();
        let body = requests.lock().unwrap().last().unwrap().clone();
        if kind == ProviderKind::Ollama {
            assert_eq!(body["options"]["num_ctx"], 8192);
            assert_eq!(body["options"]["num_predict"], 512);
            request.sampling.context_window = Some(131073);
            assert!(provider.complete(&request).await.is_err());
        } else {
            assert_eq!(body["max_tokens"], 512);
            request.sampling.context_window = Some(8192);
            assert!(provider.complete(&request).await.is_err());
        }
    }
    server.abort();
}

#[tokio::test]
async fn missing_maximum_never_submits_a_small_default() {
    let requests = Arc::new(Mutex::new(Vec::<Value>::new()));
    let app = Router::new()
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data":[{"id":"unknown"}]})) }),
        )
        .route(
            "/api/show",
            post(|| async { Json(json!({"model_info":{}})) }),
        )
        .route(
            "/v1/models/unknown",
            get(|| async { Json(json!({"id":"unknown"})) }),
        )
        .route("/api/chat", post(|State(requests): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>| async move {
            requests.lock().unwrap().push(body);
            Json(json!({"model":"unknown","message":{"role":"assistant","content":"ok"},"done":true}))
        }))
        .route("/v1/chat/completions", post(|State(requests): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>| async move {
            requests.lock().unwrap().push(body);
            Json(json!({"model":"unknown","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}))
        }))
        .with_state(requests.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    for kind in [
        ProviderKind::Ollama,
        ProviderKind::OpenAi,
        ProviderKind::Anthropic,
        ProviderKind::Mcl4Coe,
    ] {
        let mut config = ProviderConfig::new(kind);
        config.model = "unknown".into();
        config.base_url = Some(base.clone());
        config.api_key = Some("fixture".into());
        let request = ChatRequest {
            model: "unknown".into(),
            messages: vec![Message::user("hello")],
            sampling: SamplingParams::default(),
            tools: vec![],
        };
        let result = ProviderFactory::build(config)
            .unwrap()
            .complete(&request)
            .await;
        if kind == ProviderKind::Ollama {
            // Unlimited generation and the model's own context; no small guess is sent.
            result.unwrap();
            let body = requests.lock().unwrap().last().unwrap().clone();
            assert_eq!(body["options"]["num_predict"], -1);
            assert!(body["options"].get("num_ctx").is_none(), "{body}");
        } else if matches!(kind, ProviderKind::OpenAi | ProviderKind::Mcl4Coe) {
            // Omitting max_tokens is these APIs' own maximum, so nothing is guessed.
            result.unwrap();
            let body = requests.lock().unwrap().last().unwrap().clone();
            assert!(body.get("max_tokens").is_none(), "{body}");
        } else {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("did not"), "{error}");
        }
    }
    server.abort();
}

#[tokio::test]
async fn successful_generations_are_not_rejected_by_the_metadata_byte_cap() {
    let full = "x".repeat(16 * 1024 * 1024 + 1);
    let output = full.clone();
    let app = Router::new().route("/v1/chat/completions",post(move || {
        let output = output.clone();
        async move { Json(json!({"model":"model","choices":[{"message":{"role":"assistant","content":output},"finish_reason":"stop"}]})) }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut config = ProviderConfig::new(ProviderKind::OpenAi);
    config.model = "model".into();
    config.base_url = Some(format!("http://{}", listener.local_addr().unwrap()));
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let request = ChatRequest {
        model: "model".into(),
        messages: vec![Message::user("Full output")],
        sampling: SamplingParams {
            max_tokens: Some(8_000_000),
            ..Default::default()
        },
        tools: vec![],
    };
    let result = ProviderFactory::build(config)
        .unwrap()
        .complete(&request)
        .await
        .unwrap();
    assert_eq!(result.message.content, full);
    server.abort();
}

#[tokio::test]
async fn both_chat_transports_preserve_input_beyond_two_mib() {
    use kadmium_condenser::server::{ServerOptions, bind};
    let full = "Long context 🦀 ".repeat(200_000);
    let expected = full.clone();
    let app = Router::new().route("/v1/chat/completions",post(move |Json(body):Json<Value>| {
        let expected = expected.clone();
        async move {
            assert_eq!(body["messages"].as_array().unwrap().last().unwrap()["content"],expected);
            Json(json!({"model":"model","choices":[{"message":{"role":"assistant","content":"all input received"},"finish_reason":"stop"}]}))
        }
    })).layer(axum::extract::DefaultBodyLimit::disable());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let upstream = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (address, server) = bind(ServerOptions {
        port: 0,
        plugin_roots: vec![],
        ..Default::default()
    })
    .await
    .unwrap();
    let server = tokio::spawn(server);
    let client = reqwest::Client::new();
    for path in ["chat", "chat/stream"] {
        let response = client.post(format!("http://{address}/api/v1/{path}"))
            .json(&json!({"provider":{"kind":"openai","model":"model","base_url":base},"messages":[{"role":"user","content":full}],"sampling":{"max_tokens":100}}))
            .send().await.unwrap();
        assert!(response.status().is_success(), "{}", response.status());
        assert!(
            response
                .text()
                .await
                .unwrap()
                .contains("all input received")
        );
    }
    upstream.abort();
    server.abort();
}
