use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, head, post},
};
use kadmium_condenser::server::{ServerOptions, bind};
use kadmium_data::{SafetensorsToKtbTranscoder, checkpoint::CheckpointLibrary};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Clone, Default)]
struct Backend {
    blobs: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    corrupt: Arc<AtomicBool>,
    fail_preload: Arc<AtomicBool>,
    shorten_context: Arc<AtomicBool>,
    model: Arc<Mutex<String>>,
    loads: Arc<AtomicUsize>,
    context: Arc<AtomicUsize>,
}

#[tokio::test]
async fn import_verify_load_and_chat_rejects_backend_precision_changes() {
    let root =
        std::env::temp_dir().join(format!("condenser-checkpoint-flow-{}", std::process::id()));
    std::fs::create_dir_all(root.join("source")).unwrap();
    std::fs::write(
        root.join("source/config.json"),
        br#"{"model_type":"llama","max_position_embeddings":8192}"#,
    )
    .unwrap();
    std::fs::write(root.join("source/tokenizer.json"), b"{}").unwrap();
    std::fs::write(
        root.join("source/model.safetensors"),
        SafetensorsToKtbTranscoder::build_safetensors_bytes(&[(
            "model.weight",
            "F32",
            vec![2],
            &[0, 0, 128, 63, 1, 0, 128, 63],
        )]),
    )
    .unwrap();
    let backend = Backend::default();
    let app=Router::new()
        .route("/prefix/api/version",get(||async{Json(json!({"version":"fixture"}))}))
        .route("/prefix/api/blobs/{digest}",head(||async{StatusCode::NOT_FOUND}).post(|State(state):State<Backend>,Path(digest):Path<String>,bytes:Bytes|async move{
            state.blobs.lock().unwrap().insert(digest,bytes.to_vec());StatusCode::CREATED
        }))
        .route("/prefix/api/create",post(|State(state):State<Backend>,Json(body):Json<Value>|async move{
            assert!(body.get("quantize").is_none()); assert!(body.get("from").is_none());
            state.context.store(body["parameters"]["num_ctx"].as_u64().unwrap() as usize, Ordering::Relaxed);
            assert_eq!(body["files"].as_object().unwrap().len(),3);
            *state.model.lock().unwrap()=body["model"].as_str().unwrap().into();
            Json(json!({"status":"success"}))
        }))
        .route("/prefix/api/show",post(|State(state):State<Backend>|async move{
            Json(json!({"model_info":{"general.architecture":"llama","llama.context_length":8192},"tensors":[{"name":"model.weight","shape":[2],"type":if state.corrupt.load(Ordering::Relaxed){"F16"}else{"F32"}}]}))
        }))
        .route("/prefix/api/generate",post(|State(state):State<Backend>,Json(body):Json<Value>|async move{
            assert_eq!(body["options"]["num_ctx"],state.context.load(Ordering::Relaxed));state.loads.fetch_add(1,Ordering::Relaxed);
            if state.fail_preload.load(Ordering::Relaxed) { Json(json!({"error":"model does not fit"})) } else { Json(json!({"done":true})) }
        }))
        .route("/prefix/api/ps",get(|State(state):State<Backend>|async move{
            Json(json!({"models":[{"name":*state.model.lock().unwrap(),"context_length":if state.shorten_context.load(Ordering::Relaxed){4096}else{state.context.load(Ordering::Relaxed)}}]}))
        }))
        .route("/prefix/api/chat",post(|Json(body):Json<Value>|async move{
            assert!(body["model"].as_str().unwrap().starts_with("kadmium-"));
            // Full context comes from the created model's `num_ctx` parameter (asserted above).
            assert!(body["options"].get("num_ctx").is_none());
            assert_eq!(body["options"]["num_predict"],-1);
            assert_eq!(body["truncate"],false);
            Json(json!({"model":body["model"],"message":{"role":"assistant","content":"Checkpoint answer"},"done":true,"eval_count":2,"prompt_eval_count":3}))
        })).with_state(backend.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}/prefix", listener.local_addr().unwrap());
    let mock = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (addr, server) = bind(ServerOptions {
        port: 0,
        model_library: root.join("library"),
        ..Default::default()
    })
    .await
    .unwrap();
    let server = tokio::spawn(server);
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");
    let imported = client
        .post(format!("{base}/api/v1/checkpoints/import"))
        .json(&json!({"source_path":root.join("source")}))
        .send()
        .await
        .unwrap();
    assert_eq!(imported.status(), 200);
    let imported: Value = imported.json().await.unwrap();
    let id = imported["id"].as_str().unwrap();
    let loaded = client
        .post(format!("{base}/api/v1/checkpoints/load"))
        .json(&json!({"checkpoint_id":id,"base_url":upstream}))
        .send()
        .await
        .unwrap();
    let status = loaded.status();
    let loaded: Value = loaded.json().await.unwrap();
    assert_eq!(status, 200, "{loaded}");
    assert_eq!(loaded["tensor_metadata_verified"], true);
    let library = CheckpointLibrary::new(root.join("library"));
    let manifest = library.verify(id).unwrap();
    for file in manifest.source_files {
        assert_eq!(
            backend.blobs.lock().unwrap()[&format!("sha256:{}", file.sha256)],
            std::fs::read(root.join("source").join(file.name)).unwrap()
        );
    }
    let chat=client.post(format!("{base}/api/v1/chat")).json(&json!({"provider":loaded["provider"],"messages":[{"role":"user","content":"Answer with these weights"}],"enable_tools":false})).send().await.unwrap();
    assert_eq!(chat.status(), 200);
    let answer: Value = chat.json().await.unwrap();
    assert!(answer.to_string().contains("Checkpoint answer"));
    backend.corrupt.store(true, Ordering::Relaxed);
    let failed = client
        .post(format!("{base}/api/v1/checkpoints/load"))
        .json(&json!({"checkpoint_id":id,"base_url":upstream}))
        .send()
        .await
        .unwrap();
    assert!(!failed.status().is_success());
    assert!(failed.text().await.unwrap().contains("changed the dtype"));
    assert_eq!(backend.loads.load(Ordering::Relaxed), 1);
    backend.corrupt.store(false, Ordering::Relaxed);
    backend.fail_preload.store(true, Ordering::Relaxed);
    let failed = client
        .post(format!("{base}/api/v1/checkpoints/load"))
        .json(&json!({"checkpoint_id":id,"base_url":upstream}))
        .send()
        .await
        .unwrap();
    assert!(!failed.status().is_success());
    assert!(
        failed
            .text()
            .await
            .unwrap()
            .contains("did not confirm model loading")
    );
    backend.fail_preload.store(false, Ordering::Relaxed);
    backend.shorten_context.store(true, Ordering::Relaxed);
    let failed = client
        .post(format!("{base}/api/v1/checkpoints/load"))
        .json(&json!({"checkpoint_id":id,"base_url":upstream}))
        .send()
        .await
        .unwrap();
    assert!(!failed.status().is_success());
    assert!(
        failed
            .text()
            .await
            .unwrap()
            .contains("requested context length")
    );
    let remote = client
        .post(format!("{base}/api/v1/checkpoints/load"))
        .json(&json!({"checkpoint_id":id,"base_url":"https://example.com"}))
        .send()
        .await
        .unwrap();
    assert_eq!(remote.status(), 400);
    backend.shorten_context.store(false, Ordering::Relaxed);
    let lowered = client
        .post(format!("{base}/api/v1/checkpoints/load"))
        .json(&json!({"checkpoint_id":id,"base_url":upstream,"context_length":4096}))
        .send()
        .await
        .unwrap();
    assert_eq!(lowered.status(), 200);
    assert_eq!(
        lowered.json::<Value>().await.unwrap()["context_length"],
        4096
    );
    assert_eq!(backend.context.load(Ordering::Relaxed), 4096);
    let invalid = client
        .post(format!("{base}/api/v1/checkpoints/load"))
        .json(&json!({"checkpoint_id":id,"base_url":upstream,"context_length":8193}))
        .send()
        .await
        .unwrap();
    assert_eq!(invalid.status(), 400);
    let forbidden = client
        .post(format!("{base}/api/v1/checkpoints/import"))
        .header("origin", "https://example.com")
        .json(&json!({"source_path":root.join("source")}))
        .send()
        .await
        .unwrap();
    assert_eq!(forbidden.status(), 403);
    server.abort();
    mock.abort();
    let _ = std::fs::remove_dir_all(root);
}
