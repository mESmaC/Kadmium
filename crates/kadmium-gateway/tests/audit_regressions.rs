use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use kadmium_core::ModelSpec;
use kadmium_data::{DecentralizedStoragePoolEngine as Storage, LocalModelDiskImporter};
use kadmium_gateway::{GatewayState, build_router};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tower::ServiceExt;

fn gateway() -> (Router, Arc<Mutex<GatewayState>>) {
    let state = Arc::new(Mutex::new(GatewayState::new()));
    (build_router(state.clone()), state)
}

async fn request(app: &Router, method: &str, uri: &str, body: Value) -> (u16, String) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "kadmium-audit-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn invalid_shard_requests_leave_all_state_routes_usable() {
    let (app, state) = gateway();
    for (k, m) in [
        (0u64, 2u64),
        (8, 0),
        (256, 1),
        (u64::MAX, 2),
        (1 << 63, 1 << 63),
    ] {
        let (status, body) = request(
            &app,
            "POST",
            "/api/v1/storage/pool-bucket/object",
            json!({"logical_size_gb":0.01,"data_shards_k":k,"parity_shards_m":m}),
        )
        .await;
        assert!((400..500).contains(&status), "{body}");
        assert!(serde_json::from_str::<Value>(&body).unwrap()["error"].is_object());
        assert!(state.lock().unwrap().bucket_objects.is_empty());
        for path in [
            "/api/v1/cluster/overview",
            "/v1/models",
            "/api/v1/cluster/workers",
        ] {
            assert_eq!(request(&app, "GET", path, json!(null)).await.0, 200);
        }
    }
    for (k, m) in [(8, 2), (255, 1)] {
        assert!(
            Storage::create_bucket_object("test", "test", "test", 0.01, k, m, None, &[]).is_ok()
        );
    }
}

#[tokio::test]
async fn model_selection_is_validated_atomically_and_accepts_aliases() {
    let (app, state) = gateway();
    let before = serde_json::to_value(state.lock().unwrap().snapshot()).unwrap();
    let epoch = state.lock().unwrap().epoch;
    for payload in [
        json!({"inference_model_id":""}),
        json!({"training_model_id":"missing"}),
        json!({"inference_model_id":"llama-3.2-3b","training_model_id":"missing"}),
        json!({"inference_model_id":"missing","training_model_id":"llama-3.2-3b"}),
    ] {
        let (status, body) = request(&app, "POST", "/api/v1/cluster/model", payload).await;
        assert_eq!(status, 400, "{body}");
        let guard = state.lock().unwrap();
        assert_eq!(guard.epoch, epoch);
        assert_eq!(serde_json::to_value(guard.snapshot()).unwrap(), before);
    }
    assert_eq!(
        request(
            &app,
            "POST",
            "/api/v1/cluster/model",
            json!({"inference_model_id":"llama-3.2-3b"})
        )
        .await
        .0,
        200
    );
    assert_eq!(
        state.lock().unwrap().inference_model.id,
        "llama-3.2-3b-instruct"
    );
}

#[tokio::test]
async fn embeddings_reject_whole_invalid_batches_and_preserve_indices() {
    let (app, _) = gateway();
    for payload in [
        json!({}),
        json!({"input":null}),
        json!({"input":[]}),
        json!({"input":""}),
        json!({"input":{"x":"text"}}),
        json!({"input":[1,"keep",null]}),
    ] {
        let (status, body) = request(&app, "POST", "/v1/embeddings", payload).await;
        assert_eq!(status, 400, "{body}");
        assert!(
            serde_json::from_str::<Value>(&body)
                .unwrap()
                .get("data")
                .is_none()
        );
    }
    for (input, len) in [(json!("hello"), 1), (json!(["hello", "world"]), 2)] {
        let (status, body) = request(
            &app,
            "POST",
            "/v1/embeddings",
            json!({"input":input,"dimensions":16}),
        )
        .await;
        assert_eq!(status, 200, "{body}");
        let data: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(data["data"].as_array().unwrap().len(), len);
        for i in 0..len {
            assert_eq!(data["data"][i]["index"], i);
        }
    }
}

#[tokio::test]
async fn diagnostic_sse_serializes_arbitrary_model_labels() {
    let (app, _) = gateway();
    for model in [
        "quote\"model",
        "back\\slash",
        "line\r\nevent: injected",
        "模型🧪",
    ] {
        let payload =
            json!({"model":model,"messages":[{"role":"user","content":"hello"}],"stream":true});
        let (status, body) = request(&app, "POST", "/v1/messages", payload).await;
        assert_eq!(status, 200);
        let blocks = body.trim().split("\n\n").collect::<Vec<_>>();
        assert_eq!(blocks.len(), 3, "{body}");
        for (i, block) in blocks.iter().enumerate() {
            let lines = block.lines().collect::<Vec<_>>();
            assert_eq!(lines.len(), 2);
            let data: Value =
                serde_json::from_str(lines[1].strip_prefix("data: ").unwrap()).unwrap();
            if i == 0 {
                assert_eq!(data["message"]["model"], model);
            }
        }
        let (_, body) = request(
            &app,
            "POST",
            "/v1/messages",
            json!({"model":model,"messages":[]}),
        )
        .await;
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["model"],
            model
        );
    }
}

#[tokio::test]
async fn missing_and_corrupt_imports_return_errors_without_state_changes() {
    let (app, state) = gateway();
    let dir = temp_dir("imports");
    let before = serde_json::to_value(state.lock().unwrap().snapshot()).unwrap();
    let mut paths = vec![
        dir.join("missing.safetensors"),
        dir.join("missing-parent/model.gguf"),
        dir.clone(),
    ];
    for (name, bytes) in [
        ("corrupt.safetensors", b"not a safetensors file".as_slice()),
        ("truncated.safetensors", &[0, 0]),
        ("corrupt.gguf", b"bad"),
        ("truncated.gguf", b"GGUF"),
    ] {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        paths.push(path);
    }
    for path in paths {
        let (status, body) = request(
            &app,
            "POST",
            "/api/v1/models/import",
            json!({"model_id":"audit-corrupt","source_path":path,"start_layer":0,"end_layer":0}),
        )
        .await;
        assert_eq!(status, 400, "{body}");
        assert!(serde_json::from_str::<Value>(&body).unwrap()["error"]["message"].is_string());
        assert_eq!(
            serde_json::to_value(state.lock().unwrap().snapshot()).unwrap(),
            before
        );
        assert_eq!(
            request(&app, "GET", "/api/v1/cluster/overview", json!(null))
                .await
                .0,
            200
        );
    }
    assert_eq!(
        request(&app, "POST", "/api/v1/models/import", json!({}))
            .await
            .0,
        400
    );
    assert!(!PathBuf::from("target/kadmium-models/audit-corrupt.safetensors").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn journal_reconstruction_preserves_objects_imports_anchors_and_progress() {
    let dir = temp_dir("journal");
    let path = dir.join("cluster_state.json");
    let source = dir.join("valid.safetensors");
    std::fs::write(
        &source,
        kadmium_data::SafetensorsToKtbTranscoder::build_safetensors_bytes(&[(
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            vec![2],
            &[1, 2, 3, 4],
        )]),
    )
    .unwrap();
    let manifest = LocalModelDiskImporter::import_or_transcode_to_disk(
        "llama-3.2-3b-instruct",
        Some(source.to_str().unwrap()),
        None,
        0,
        0,
        Some(&dir),
    )
    .unwrap();
    assert!(!manifest.synthetic_fixture);
    let mut first = GatewayState::new();
    assert!(!first.load_persistent_state_from(&path).unwrap());
    first.inference_model = ModelSpec::llama_3_2_3b_instruct();
    first.training_step_counter = 42;
    first.imported_models.push(manifest);
    first.bucket_objects.push(
        Storage::create_bucket_object("persist", "Persist", "test", 0.01, 8, 2, None, &first.nodes)
            .unwrap(),
    );
    first.bucket_anchors =
        Storage::verify_bucket_grounding_and_mass_dropout_recovery(8, 128).configured_anchors;
    first.save_persistent_state_to(&path).unwrap();
    let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    drop(first);
    let mut restarted = GatewayState::new();
    restarted.set_live_only_mode(true);
    assert!(restarted.load_persistent_state_from(&path).unwrap());
    assert!(!restarted.active_model_loaded);
    assert_eq!(restarted.inference_plan.model_id, "llama-3.2-3b-instruct");
    restarted.save_persistent_state_to(&path).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&std::fs::read(&path).unwrap()).unwrap(),
        saved
    );
    for bad in [b"{corrupt".to_vec(), {
        let mut future = saved.clone();
        future["schema_version"] = json!(999);
        serde_json::to_vec(&future).unwrap()
    }] {
        std::fs::write(&path, &bad).unwrap();
        let mut state = GatewayState::new();
        assert!(state.load_persistent_state_from(&path).is_err());
        assert!(state.imported_models.is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), bad);
    }
    std::fs::remove_dir_all(dir).unwrap();
}
