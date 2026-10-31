//! Condenser prepares a checkpoint, sends it to a Kadmium receiver over the verified chunk
//! protocol, the receiver loads it at full context, and chat reaches the loaded model.
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, head, post},
};
use kadmium_condenser::server::{ServerOptions, bind};
use kadmium_data::{SafetensorsToKtbTranscoder, checkpoint::CheckpointLibrary};
use kadmium_gateway::model_transport::{
    OLLAMA_PROXY_PATH, TransportState, router, serve_loaded_models,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct Ollama {
    blobs: Arc<Mutex<BTreeMap<String, usize>>>,
    model: Arc<Mutex<String>>,
    context: Arc<Mutex<u64>>,
}

async fn serve(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap()
    });
    base
}

#[tokio::test]
async fn prepared_model_is_sent_verified_loaded_and_served() {
    let root = std::env::temp_dir().join(format!("kadmium-transport-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("source")).unwrap();
    std::fs::write(
        root.join("source/config.json"),
        br#"{"model_type":"llama","max_position_embeddings":131072}"#,
    )
    .unwrap();
    std::fs::write(root.join("source/tokenizer.json"), b"{}").unwrap();
    std::fs::write(
        root.join("source/model.safetensors"),
        SafetensorsToKtbTranscoder::build_safetensors_bytes(&[(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "BF16",
            vec![2],
            &[1, 2, 3, 4],
        )]),
    )
    .unwrap();

    let ollama = Ollama::default();
    let backend = serve(
        Router::new()
            .route("/api/version", get(|| async { Json(json!({"version":"fixture"})) }))
            .route(
                "/api/blobs/{digest}",
                head(|| async { StatusCode::NOT_FOUND }).post(
                    |State(o): State<Ollama>, Path(d): Path<String>, b: Bytes| async move {
                        o.blobs.lock().unwrap().insert(d, b.len());
                        StatusCode::CREATED
                    },
                ),
            )
            .route(
                "/api/create",
                post(|State(o): State<Ollama>, Json(body): Json<Value>| async move {
                    assert!(body.get("quantize").is_none());
                    *o.context.lock().unwrap() = body["parameters"]["num_ctx"].as_u64().unwrap();
                    *o.model.lock().unwrap() = body["model"].as_str().unwrap().into();
                    Json(json!({"status":"success"}))
                }),
            )
            .route(
                "/api/show",
                post(|State(o): State<Ollama>, Json(body): Json<Value>| async move {
                    // Only models this backend actually created exist.
                    if body["model"] != *o.model.lock().unwrap() {
                        return (StatusCode::NOT_FOUND, Json(json!({"error":"model not found"})));
                    }
                    (StatusCode::OK, Json(json!({
                        "model_info":{"general.architecture":"llama","llama.context_length":131072},
                        "tensors":[{"name":"model.layers.0.mlp.experts.0.up_proj.weight","shape":[2],"type":"BF16"}]
                    })))
                }),
            )
            .route(
                "/api/chat",
                post(|Json(body): Json<Value>| async move {
                    // Received models keep local quality guarantees on every request.
                    assert_eq!(body["options"]["num_ctx"], 131072);
                    assert_eq!(body["options"]["num_predict"], -1);
                    assert_eq!(body["truncate"], false);
                    assert_eq!(body["shift"], false);
                    Json(json!({"model":body["model"],"message":{"role":"assistant","content":"native answer from the transported model"},"done":true,"eval_count":2,"prompt_eval_count":3}))
                }),
            )
            .route("/api/generate", post(|| async { Json(json!({"done":true})) }))
            .route(
                "/api/ps",
                get(|State(o): State<Ollama>| async move {
                    Json(json!({"models":[{"name":*o.model.lock().unwrap(),"context_length":*o.context.lock().unwrap()}]}))
                }),
            )
            .route(
                "/v1/chat/completions",
                post(|Json(body): Json<Value>| async move {
                    Json(json!({"model":body["model"],"choices":[{"message":{"role":"assistant","content":"served by the transported model"},"finish_reason":"stop"}]}))
                }),
            )
            .with_state(ollama.clone()),
    )
    .await;

    let receiver_library = CheckpointLibrary::new(root.join("receiver"));
    let transport = TransportState::new(
        receiver_library.clone(),
        backend.clone(),
        Some("cloud-key".into()),
    );
    let coordinator = |transport: TransportState| {
        router(transport.clone())
            .route(
                "/v1/chat/completions",
                post(|| async { Json(json!({"choices":[{"message":{"content":"coordinator"}}]})) }),
            )
            .route(
                "/v1/messages",
                post(|| async { Json(json!({"content":[{"text":"coordinator"}]})) }),
            )
            .layer(axum::middleware::from_fn_with_state(
                transport,
                serve_loaded_models,
            ))
    };
    let kadmium = serve(coordinator(transport.clone())).await;

    let (address, condenser) = bind(ServerOptions {
        port: 0,
        model_library: root.join("sender"),
        ..Default::default()
    })
    .await
    .unwrap();
    tokio::spawn(condenser);
    let condenser = format!("http://{address}");
    let client = reqwest::Client::new();

    let prepared: Value = client
        .post(format!("{condenser}/api/v1/checkpoints/import"))
        .json(&json!({"source_path": root.join("source")}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = prepared["id"].as_str().unwrap().to_string();
    assert_eq!(prepared["transport"]["alignment_bytes"], 4096);
    assert_eq!(prepared["transport"]["chunk_bytes"], 16 * 1024 * 1024);

    // Without the key the receiver refuses, and the failure says why.
    let denied = client
        .post(format!("{kadmium}/api/v1/model-transport"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 401);

    let publish = |key: &'static str| {
        client
            .post(format!("{condenser}/api/v1/checkpoints/publish"))
            .json(&json!({"checkpoint_id": id, "endpoint": kadmium, "api_key": key}))
            .send()
    };
    let wait = || async {
        loop {
            let progress: Value = client
                .get(format!(
                    "{condenser}/api/v1/checkpoints/publish?checkpoint_id={id}"
                ))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if matches!(progress["state"].as_str(), Some("ready" | "failed")) {
                return progress;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    };
    assert_eq!(publish("wrong").await.unwrap().status(), 200);
    let failed = wait().await;
    assert_eq!(failed["state"], "failed");
    assert!(
        failed["message"].as_str().unwrap().contains("access token"),
        "{failed}"
    );

    assert_eq!(publish("cloud-key").await.unwrap().status(), 200);
    let ready = wait().await;
    assert_eq!(ready["state"], "ready", "{ready}");
    assert_eq!(ready["result"]["context_length"], 131072);
    assert_eq!(ready["result"]["provider"]["kind"], "ollama");
    assert_eq!(
        ready["result"]["provider"]["base_url"],
        format!("{kadmium}{OLLAMA_PROXY_PATH}")
    );
    // Another model's progress never reads as this one's.
    let other: Value = client
        .get(format!(
            "{condenser}/api/v1/checkpoints/publish?checkpoint_id={}",
            "b".repeat(64)
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(other["state"], "idle");
    let model = ready["result"]["provider"]["model"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(receiver_library.verify(&id).unwrap().id, id);
    assert_eq!(ollama.blobs.lock().unwrap().len(), 3);

    // Sending again skips everything the receiver already verified.
    assert_eq!(publish("cloud-key").await.unwrap().status(), 200);
    let again = wait().await;
    assert_eq!(again["state"], "ready");
    assert_eq!(again["sent_bytes"], 0);

    let chat = |key: Option<&'static str>, model: String| {
        let request = client
            .post(format!("{kadmium}/v1/chat/completions"))
            .json(&json!({"model": model, "messages":[{"role":"user","content":"hi"}]}));
        match key {
            Some(key) => request.bearer_auth(key).send(),
            None => request.send(),
        }
    };
    let answer: Value = chat(Some("cloud-key"), model.clone())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        answer["choices"][0]["message"]["content"],
        "served by the transported model"
    );
    assert_eq!(chat(None, model.clone()).await.unwrap().status(), 401);
    let other: Value = chat(None, "reference".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(other["choices"][0]["message"]["content"], "coordinator");

    // A Condenser task routed to the received model runs through the native Ollama proxy with
    // full context, unlimited output and no truncation, exactly like a local model.
    let mut provider = ready["result"]["provider"].clone();
    provider["api_key"] = json!("cloud-key");
    let task: Value = client
        .post(format!("{condenser}/api/v1/chat"))
        .json(&json!({"provider": provider, "messages":[{"role":"user","content":"hi"}], "enable_tools": false}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        task.to_string()
            .contains("native answer from the transported model"),
        "{task}"
    );

    // After a coordinator restart the published model is found again, not answered by the
    // diagnostic handler; an unknown received-model name is refused outright.
    let restarted = serve(coordinator(TransportState::new(
        receiver_library.clone(),
        backend.clone(),
        Some("cloud-key".into()),
    )))
    .await;
    let restarted_chat: Value = client
        .post(format!("{restarted}/v1/chat/completions"))
        .bearer_auth("cloud-key")
        .json(&json!({"model": model, "messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        restarted_chat["choices"][0]["message"]["content"],
        "served by the transported model"
    );
    let unknown = format!("kadmium-{}:latest", "c".repeat(64));
    let missing = client
        .post(format!("{restarted}/v1/chat/completions"))
        .bearer_auth("cloud-key")
        .json(&json!({"model": unknown, "messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 409);
    assert!(missing.text().await.unwrap().contains("not loaded"));
    let messages = client
        .post(format!("{restarted}/v1/messages"))
        .bearer_auth("cloud-key")
        .json(&json!({"model": model, "messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(messages.status(), 400);
    assert!(!messages.text().await.unwrap().contains("coordinator\""));
    let _ = std::fs::remove_dir_all(&root);
}
