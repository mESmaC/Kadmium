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
        .route("/v1/models/model", get(|| async { Json(json!({"id":"model","capabilities":{"max_tokens":64000}})) }))
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
            assert_eq!(body["options"]["num_ctx"], 131072);
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
        );
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
        let error = ProviderFactory::build(config)
            .unwrap()
            .complete(&request)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not"), "{error}");
    }
    server.abort();
}
