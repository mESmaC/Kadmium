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
    let mut req = Request::builder()
        .method(method)
        .uri(uri)
        .header("host", "127.0.0.1:7848")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    req.extensions_mut()
        .insert(axum::extract::ConnectInfo(std::net::SocketAddr::from((
            [127, 0, 0, 1],
            7848,
        ))));
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn loopback_raw_get(
    app: &Router,
    uri: &str,
    range_header: Option<&str>,
) -> axum::response::Response {
    let mut builder = Request::builder()
        .method("GET")
        .uri(uri)
        .header("host", "127.0.0.1:7848");
    if let Some(r) = range_header {
        builder = builder.header("range", r);
    }
    let mut req = builder.body(Body::empty()).unwrap();
    req.extensions_mut()
        .insert(axum::extract::ConnectInfo(std::net::SocketAddr::from((
            [127, 0, 0, 1],
            7848,
        ))));
    app.clone().oneshot(req).await.unwrap()
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
    let mut huge_count = vec![0u8; 40];
    huge_count[..4].copy_from_slice(b"GGUF");
    huge_count[4..8].copy_from_slice(&3u32.to_le_bytes());
    huge_count[8..16].copy_from_slice(&(1u64 << 50).to_le_bytes());
    let mut huge_header = vec![0u8; 40];
    huge_header[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let names: Vec<_> = (0..800)
        .map(|i| format!("model.layers.0.experts.{i}.weight"))
        .collect();
    let tensors: Vec<_> = names
        .iter()
        .map(|name| (name.as_str(), "BF16", vec![1], &[1u8, 2][..]))
        .collect();
    let large_directory =
        kadmium_data::SafetensorsToKtbTranscoder::build_safetensors_bytes(&tensors);
    for (name, bytes) in [
        ("huge-count.gguf", huge_count),
        ("huge-header.safetensors", huge_header),
        ("large-directory.safetensors", large_directory),
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
            json!({"model_id":"llama-3.2-3b-instruct","source_path":path,"start_layer":0,"end_layer":0}),
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
    first.set_live_only_mode(true);
    first.user_selected_model = true;
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
    // Legacy custom imports must not strand supported models or unrelated saved state.
    for (custom_inference, custom_training) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let mut legacy = saved.clone();
        let mut custom = legacy["imported_models"][0].clone();
        custom["model_id"] = json!("my-custom-model");
        custom["derived_model_spec"]["id"] = json!("my-custom-model");
        legacy["imported_models"]
            .as_array_mut()
            .unwrap()
            .push(custom);
        if custom_inference {
            legacy["inference_model_id"] = json!("my-custom-model");
        }
        if custom_training {
            legacy["training_model_id"] = json!("my-custom-model");
        }
        let original = serde_json::to_vec(&legacy).unwrap();
        std::fs::write(&path, &original).unwrap();
        let mut restored = GatewayState::new();
        restored.set_live_only_mode(true);
        assert!(restored.load_persistent_state_from(&path).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(restored.imported_models.len(), 1);
        assert!(restored.resolve_model("my-custom-model").is_none());
        assert_eq!(restored.bucket_objects.len(), 1);
        assert_eq!(
            serde_json::to_value(&restored.bucket_anchors).unwrap(),
            saved["bucket_anchors"]
        );
        assert_eq!(restored.user_selected_model, !custom_inference);
        assert_eq!(
            restored.training_step_counter,
            if custom_training || custom_inference {
                0
            } else {
                42
            }
        );
        assert_ne!(restored.inference_plan.model_id, "my-custom-model");
        assert_ne!(restored.training_plan.model_id, "my-custom-model");
        restored.save_persistent_state_to(&path).unwrap();
        let mut next = GatewayState::new();
        next.set_live_only_mode(true);
        assert!(next.load_persistent_state_from(&path).unwrap());
        assert_eq!(next.imported_models.len(), 1);
    }
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

#[tokio::test]
async fn overflowing_sizes_preserve_journal_and_restart_state() {
    let (app, state) = gateway();
    let dir = temp_dir("overflow");
    let path = dir.join("cluster_state.json");
    state
        .lock()
        .unwrap()
        .save_persistent_state_to(&path)
        .unwrap();
    let saved = std::fs::read(&path).unwrap();
    for size in [1e307, 1e100, 1e12] {
        let (status, body) = request(
            &app,
            "POST",
            "/api/v1/storage/pool-bucket/object",
            json!({"logical_size_gb":size,"data_shards_k":8,"parity_shards_m":2}),
        )
        .await;
        assert_eq!(status, 400, "{body}");
        assert!(state.lock().unwrap().bucket_objects.is_empty());
        state
            .lock()
            .unwrap()
            .save_persistent_state_to(&path)
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), saved);
    }
    let mut restarted = GatewayState::new();
    assert!(restarted.load_persistent_state_from(&path).unwrap());
    assert!(restarted.bucket_objects.is_empty());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn journals_preserve_selection_and_separate_live_from_demo_progress() {
    let dir = temp_dir("provenance");
    let path = dir.join("cluster_state.json");
    for selected in [false, true] {
        let mut demo = GatewayState::new();
        demo.user_selected_model = selected;
        demo.save_persistent_state_to(&path).unwrap();
        for legacy in [false, true] {
            if legacy {
                let mut value: Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                value.as_object_mut().unwrap().remove("live_only_mode");
                value.as_object_mut().unwrap().remove("user_selected_model");
                std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            }
            let mut live = GatewayState::new();
            live.set_live_only_mode(true);
            live.load_persistent_state_from(&path).unwrap();
            assert_eq!(live.user_selected_model, selected && !legacy);
            assert_eq!(live.training_step_counter, 0);
            assert_eq!(live.last_checkpoint_step, 0);
            assert!(live.training_loss_history.is_empty());
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn demo_saves_never_overwrite_live_training_progress() {
    let dir = temp_dir("mode-journals");
    let mut live = GatewayState::new();
    live.persistent_state_dir = dir.clone();
    live.set_live_only_mode(true);
    live.training_step_counter = 500;
    live.last_training_loss = 0.25;
    live.training_loss_history = vec![(500, 0.25)];
    live.last_checkpoint_step = 400;
    live.save_persistent_state().unwrap();
    let live_bytes = std::fs::read(live.persistent_state_path()).unwrap();

    let migrated_dir = dir.join("migrated");
    let mut demo = GatewayState::new();
    demo.persistent_state_dir = migrated_dir.clone();
    // Upgrading in demo mode must also migrate the live journal unchanged.
    assert!(demo.load_persistent_state_with_legacy(&dir).unwrap());
    assert_eq!(
        std::fs::read(migrated_dir.join("cluster_state.json")).unwrap(),
        live_bytes
    );
    assert_ne!(demo.training_step_counter, 500);
    demo.training_step_counter = 2000;
    demo.save_persistent_state().unwrap();
    assert_ne!(demo.persistent_state_path(), live.persistent_state_path());
    assert_eq!(
        std::fs::read(live.persistent_state_path()).unwrap(),
        live_bytes
    );

    let mut restarted = GatewayState::new();
    restarted.persistent_state_dir = migrated_dir;
    restarted.set_live_only_mode(true);
    assert!(restarted.load_persistent_state().unwrap());
    assert_eq!(restarted.training_step_counter, 500);
    assert_eq!(restarted.last_training_loss, 0.25);
    assert_eq!(restarted.training_loss_history, vec![(500, 0.25)]);
    assert_eq!(restarted.last_checkpoint_step, 400);
    restarted.set_live_only_mode(false);
    assert!(restarted.load_persistent_state().unwrap());
    assert_eq!(restarted.training_step_counter, 2000);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn journal_write_failure_preserves_original_and_cleans_temporary_file() {
    let dir = temp_dir("write-failure");
    let path = dir.join("cluster_state.json");
    // A directory at the destination forces replacement to fail on every OS.
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("original"), b"preserve").unwrap();
    assert!(GatewayState::new().save_persistent_state_to(&path).is_err());
    assert_eq!(std::fs::read(path.join("original")).unwrap(), b"preserve");
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn bucket_upload_download_verify_and_multi_medium_inventory_routes_work() {
    let (app, state) = gateway();
    let dir = temp_dir("bucket-routes");
    state.lock().unwrap().persistent_state_dir = dir.clone();

    let (status, body) = request(
        &app,
        "GET",
        "/api/v1/storage/pool-bucket/mediums",
        json!(null),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let mediums: Value = serde_json::from_str(&body).unwrap();
    assert!(mediums["cluster_mediums"].as_array().unwrap().len() >= 4);

    let (status, body) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/upload",
        json!({
            "object_uri": "kdm://pool-bucket/tests/route-payload.bin",
            "storage_class": "ModelCheckpointKtb",
            "content_utf8": "Kadmium multi-medium erasure-coded payload across all storage tiers!"
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let uploaded: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        uploaded["manifest"]["object_uri"],
        "kdm://pool-bucket/tests/route-payload.bin"
    );
    assert!(
        !uploaded["manifest"]["shards"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        uploaded["access_guide"]["cli_download_cmd"]
            .as_str()
            .unwrap()
            .contains("kadmium bucket get")
    );

    let (status, body) = request(
        &app,
        "GET",
        "/api/v1/storage/pool-bucket/download?uri=kdm://pool-bucket/tests/route-payload.bin",
        json!(null),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let downloaded: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        downloaded["content_utf8"],
        "Kadmium multi-medium erasure-coded payload across all storage tiers!"
    );

    let (status, body) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/verify",
        json!({"uri": "kdm://pool-bucket/tests/route-payload.bin", "simulate_drop_count": 2}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let verified: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(verified["bit_identical_verified"], true);
    assert_eq!(verified["sha256_verified"], true);

    // Arbitrary server-side source_path reads and output_path writes over HTTP must be rejected
    let (status, _) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/upload",
        json!({
            "object_uri": "kdm://pool-bucket/tests/forbidden-path.bin",
            "source_path": "/etc/passwd"
        }),
    )
    .await;
    assert_eq!(status, 400);

    let (status, _) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/download",
        json!({
            "uri": "kdm://pool-bucket/tests/route-payload.bin",
            "output_path": "/tmp/pwn.bin"
        }),
    )
    .await;
    assert_eq!(status, 400);

    // Unbounded k+m amplification on stored payloads must be rejected
    let (status, _) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/upload",
        json!({
            "object_uri": "kdm://pool-bucket/tests/amp.bin",
            "content_utf8": "test",
            "data_shards_k": 1,
            "parity_shards_m": 255
        }),
    )
    .await;
    assert_eq!(status, 400);

    // Cross-origin browser mutation without token must be rejected with 403
    let cross_origin_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/storage/pool-bucket/upload")
                .header("content-type", "application/json")
                .header("origin", "https://evil.example")
                .body(Body::from(
                    json!({
                        "object_uri": "kdm://pool-bucket/tests/csrf.bin",
                        "content_utf8": "csrf"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_origin_resp.status().as_u16(), 403);

    // Valid byte range returns 206 Partial Content + Content-Range + nosniff
    let range_resp = loopback_raw_get(
        &app,
        "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/route-payload.bin",
        Some("bytes=0-6"),
    )
    .await;
    assert_eq!(range_resp.status().as_u16(), 206);
    assert_eq!(
        range_resp
            .headers()
            .get("content-range")
            .unwrap()
            .to_str()
            .unwrap(),
        "bytes 0-6/68"
    );
    assert_eq!(
        range_resp
            .headers()
            .get("x-content-type-options")
            .unwrap()
            .to_str()
            .unwrap(),
        "nosniff"
    );
    let partial_bytes = to_bytes(range_resp.into_body(), 1024).await.unwrap();
    assert_eq!(&partial_bytes[..], b"Kadmium");

    // Query offset/length without an HTTP Range header returns 200 OK (not 206 Partial Content)
    let offset_resp = loopback_raw_get(
        &app,
        "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/route-payload.bin&offset=0&length=7",
        None,
    )
    .await;
    assert_eq!(offset_resp.status().as_u16(), 200);
    assert!(offset_resp.headers().get("content-range").is_none());
    let offset_bytes = to_bytes(offset_resp.into_body(), 1024).await.unwrap();
    assert_eq!(&offset_bytes[..], b"Kadmium");

    // Unsatisfiable range returns 416 Range Not Satisfiable with Content-Range: bytes */68
    let bad_range_resp = loopback_raw_get(
        &app,
        "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/route-payload.bin",
        Some("bytes=999-1000"),
    )
    .await;
    assert_eq!(bad_range_resp.status().as_u16(), 416);
    assert_eq!(
        bad_range_resp
            .headers()
            .get("content-range")
            .unwrap()
            .to_str()
            .unwrap(),
        "bytes */68"
    );

    // Malformed Range header must be ignored per RFC 9110 §14.2 and serve 200 OK full payload
    for ignored_range in ["bytes=invalid", "items=0-5", "bytes=10-5"] {
        let ignored_resp = loopback_raw_get(
            &app,
            "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/route-payload.bin",
            Some(ignored_range),
        )
        .await;
        assert_eq!(ignored_resp.status().as_u16(), 200, "range={ignored_range}");
    }

    // 0-byte object without Range header serves 200 OK; with Range: bytes=0-0 returns 416
    let (status, empty_body) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/upload",
        json!({
            "object_uri": "kdm://pool-bucket/tests/empty.bin",
            "content_utf8": ""
        }),
    )
    .await;
    assert_eq!(status, 200, "{empty_body}");
    let empty_get = loopback_raw_get(
        &app,
        "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/empty.bin",
        None,
    )
    .await;
    assert_eq!(empty_get.status().as_u16(), 200);
    assert!(
        to_bytes(empty_get.into_body(), 1024)
            .await
            .unwrap()
            .is_empty()
    );

    let empty_range_416 = loopback_raw_get(
        &app,
        "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/empty.bin",
        Some("bytes=0-0"),
    )
    .await;
    assert_eq!(empty_range_416.status().as_u16(), 416);
    assert_eq!(
        empty_range_416
            .headers()
            .get("content-range")
            .unwrap()
            .to_str()
            .unwrap(),
        "bytes */0"
    );

    // Multi-byte UTF-8 payload ("€".repeat(2000), 6,000 bytes) must not panic at byte index 4096
    let euro_payload = "€".repeat(2000);
    let (euro_up_status, euro_up_body) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/upload",
        json!({
            "object_uri": "kdm://pool-bucket/tests/utf8-euro.txt",
            "content_utf8": euro_payload
        }),
    )
    .await;
    assert_eq!(euro_up_status, 200, "{euro_up_body}");
    let (euro_dl_status, euro_dl_body) = request(
        &app,
        "GET",
        "/api/v1/storage/pool-bucket/download?uri=kdm://pool-bucket/tests/utf8-euro.txt",
        json!(null),
    )
    .await;
    assert_eq!(euro_dl_status, 200, "{euro_dl_body}");
    let euro_dl: Value = serde_json::from_str(&euro_dl_body).unwrap();
    assert_eq!(
        euro_dl["content_utf8"].as_str(),
        Some(euro_payload.as_str())
    );
    assert_eq!(
        euro_dl["utf8_preview"].as_str().map(|s| s.len()),
        Some(4098)
    );
    assert!(euro_dl["utf8_preview"].as_str().unwrap().ends_with("..."));

    // Payload > 2 MiB (Axum's default limit) and 8 MiB data_hex upload succeed on /api/v1/storage/pool-bucket/upload
    let large_payload = "A".repeat(2 * 1024 * 1024 + 128 * 1024);
    let (large_status, large_body) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/upload",
        json!({
            "object_uri": "kdm://pool-bucket/tests/large-2m.bin",
            "content_utf8": large_payload
        }),
    )
    .await;
    assert_eq!(large_status, 200, "{large_body}");

    let hex_8m = "ab".repeat(8 * 1024 * 1024);
    let (hex_8m_status, hex_8m_body) = request(
        &app,
        "POST",
        "/api/v1/storage/pool-bucket/upload",
        json!({
            "object_uri": "kdm://pool-bucket/tests/hex-8m.bin",
            "data_hex": hex_8m
        }),
    )
    .await;
    assert_eq!(hex_8m_status, 200, "{hex_8m_body}");

    // Missing ConnectInfo (maybe_peer == None) without token must fail closed with 403 Forbidden
    for (method, uri, host_hdr) in [
        (
            "GET",
            "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/route-payload.bin",
            Some("127.0.0.1:7848"),
        ),
        ("GET", "/api/v1/storage/pool-bucket/objects", None),
        (
            "POST",
            "/api/v1/models/quality-tradeoffs",
            Some("localhost"),
        ),
        ("POST", "/mcp", Some("127.0.0.1:7848")),
    ] {
        let mut builder = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json");
        if let Some(h) = host_hdr {
            builder = builder.header("host", h);
        }
        let req = builder
            .body(Body::from(if uri == "/mcp" {
                r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#
            } else {
                "{}"
            }))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status().as_u16(),
            403,
            "missing ConnectInfo {method} {uri}"
        );
    }

    // Non-loopback peer socket (even with spoofed Host: localhost:7848 or empty Host) must be rejected without token
    let remote_peer: std::net::SocketAddr = "198.51.100.42:54321".parse().unwrap();
    for (method, uri, host_hdr) in [
        (
            "GET",
            "/api/v1/storage/pool-bucket/raw?uri=kdm://pool-bucket/tests/route-payload.bin",
            "localhost:7848",
        ),
        ("GET", "/api/v1/storage/pool-bucket/objects", ""),
        ("POST", "/api/v1/models/quality-tradeoffs", "127.0.0.1:7848"),
        ("POST", "/mcp", "localhost"),
    ] {
        let mut req = Request::builder()
            .method(method)
            .uri(uri)
            .header("host", host_hdr)
            .header("content-type", "application/json")
            .body(Body::from(if uri == "/mcp" {
                r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#
            } else {
                "{}"
            }))
            .unwrap();
        req.extensions_mut()
            .insert(axum::extract::ConnectInfo(remote_peer));
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status().as_u16(), 403, "{method} {uri}");
    }

    // With valid bearer token, remote peer is authorized
    let token = state.lock().unwrap().bucket_write_token.clone();
    let mut authed_req = Request::builder()
        .method("GET")
        .uri("/api/v1/storage/pool-bucket/objects")
        .header("host", "example.com")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    authed_req
        .extensions_mut()
        .insert(axum::extract::ConnectInfo(remote_peer));
    let authed_resp = app.clone().oneshot(authed_req).await.unwrap();
    assert_eq!(authed_resp.status().as_u16(), 200);

    // Fresh install token persistence: load_persistent_state persists bucket-access-token even when cluster_state.json is absent
    let fresh_dir = temp_dir("fresh-token-install");
    let mut fresh_state_1 = GatewayState::new();
    fresh_state_1.persistent_state_dir = fresh_dir.clone();
    assert!(!fresh_state_1.load_persistent_state().unwrap());
    assert!(fresh_dir.join("bucket-store/bucket-access-token").exists());
    let mut fresh_state_2 = GatewayState::new();
    fresh_state_2.persistent_state_dir = fresh_dir.clone();
    assert!(!fresh_state_2.load_persistent_state().unwrap());
    assert_eq!(
        fresh_state_1.bucket_write_token,
        fresh_state_2.bucket_write_token
    );
    std::fs::remove_dir_all(fresh_dir).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn model_quality_limits_default_to_uncompromised_and_enforce_manual_tradeoffs() {
    let (app, _) = gateway();
    let (status, body) =
        request(&app, "GET", "/api/v1/models/quality-tradeoffs", json!(null)).await;
    assert_eq!(status, 200, "{body}");
    let report: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(report["is_uncompromised_maximum_quality"], true);
    assert_eq!(report["effective_context_window_tokens"], 163_840);
    assert_eq!(report["effective_max_input_tokens"], 163_840);
    assert!(report["effective_max_output_tokens"].is_null());
    assert!(report.get("estimated_quality_retention_pct").is_none());
    assert!(report.get("estimated_prefill_speedup_x").is_none());
    assert!(
        report["tradeoffs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|t| t["manually_lowered"] == false)
    );

    // By default (uncapped output tokens), requesting 100,000 max_tokens within the 163,840 context window succeeds
    let (chat_ok, chat_ok_body) = request(
        &app,
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "deepseek-ai/DeepSeek-V3",
            "max_tokens": 100_000,
            "messages": [{"role": "user", "content": "Hello"}]
        }),
    )
    .await;
    assert_eq!(chat_ok, 200, "{chat_ok_body}");

    // Baseline dataflow seq_len depends on context window, not max_output_tokens
    let (_, df_before_body) = request(&app, "GET", "/api/v1/dataflow", json!(null)).await;
    let df_before: Value = serde_json::from_str(&df_before_body).unwrap();

    // Lower only max_output_tokens and verify dataflow report is unchanged
    let (status, _) = request(
        &app,
        "POST",
        "/api/v1/models/quality-tradeoffs",
        json!({ "max_output_tokens": 2_048 }),
    )
    .await;
    assert_eq!(status, 200);
    let (_, df_after_body) = request(&app, "GET", "/api/v1/dataflow", json!(null)).await;
    let df_after: Value = serde_json::from_str(&df_after_body).unwrap();
    assert_eq!(df_before["profiles"], df_after["profiles"]);

    // Unknown fields such as compact_history must be rejected (400/422)
    let (unknown_status, _) = request(
        &app,
        "POST",
        "/api/v1/models/quality-tradeoffs",
        json!({ "compact_history": true }),
    )
    .await;
    assert!((400..500).contains(&unknown_status));

    let (status, body) = request(
        &app,
        "POST",
        "/api/v1/models/quality-tradeoffs",
        json!({
            "context_window_tokens": 32_768,
            "max_input_tokens": 4,
            "max_output_tokens": 4_096,
            "active_experts_per_token": 4
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let lowered: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(lowered["is_uncompromised_maximum_quality"], false);
    assert_eq!(lowered["effective_max_output_tokens"], 4_096);
    assert_eq!(
        lowered["tradeoffs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| t["manually_lowered"] == true)
            .count(),
        4
    );

    // Lowered max_input_tokens (4 tokens) must actively reject longer prompts on /v1/chat/completions for DeepSeek
    let (chat_status, chat_body) = request(
        &app,
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": "This prompt has well over four estimated tokens and must be rejected when manual max_input_tokens=4"}]
        }),
    )
    .await;
    assert_eq!(chat_status, 400, "{chat_body}");

    // Per-request model override isolation: calling /v1/chat/completions with qwen2.5-coder-14b must NOT inherit DeepSeek's max_input_tokens=4 override
    let (qwen_chat_status, qwen_chat_body) = request(
        &app,
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "qwen2.5-coder-14b",
            "messages": [{"role": "user", "content": "This prompt has well over four estimated tokens and must succeed on qwen2.5-coder-14b because overrides are scoped per model"}]
        }),
    )
    .await;
    assert_eq!(qwen_chat_status, 200, "{qwen_chat_body}");

    let (status, body) = request(
        &app,
        "POST",
        "/api/v1/models/quality-tradeoffs",
        json!({"reset_to_native_maximum": true}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let restored: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(restored["is_uncompromised_maximum_quality"], true);
    assert!(restored["effective_max_output_tokens"].is_null());

    // Lower DeepSeek to 65,536, switch to qwen2.5-coder-7b (native 32,768), verify Qwen starts at 32,768 uncompromised,
    // and switch back to DeepSeek to verify 65,536 was preserved and never clamped to 32,768
    let (ds_set_status, _) = request(
        &app,
        "POST",
        "/api/v1/models/quality-tradeoffs",
        json!({"context_window_tokens": 65_536}),
    )
    .await;
    assert_eq!(ds_set_status, 200);

    let (qwen_sw_status, qwen_sw_body) = request(
        &app,
        "POST",
        "/api/v1/models/quality-tradeoffs",
        json!({"model_id": "qwen2.5-coder-7b"}),
    )
    .await;
    assert_eq!(qwen_sw_status, 200, "{qwen_sw_body}");
    let qwen_report: Value = serde_json::from_str(&qwen_sw_body).unwrap();
    assert_eq!(qwen_report["model_id"], "qwen2.5-coder-7b-instruct");
    assert_eq!(qwen_report["is_uncompromised_maximum_quality"], true);
    assert_eq!(qwen_report["effective_context_window_tokens"], 32_768);

    let (ds_back_status, _) = request(
        &app,
        "POST",
        "/api/v1/cluster/model",
        json!({"inference_model_id": "deepseek-v3-671b-moe"}),
    )
    .await;
    assert_eq!(ds_back_status, 200);
    let (_, ds_back_q_body) =
        request(&app, "GET", "/api/v1/models/quality-tradeoffs", json!(null)).await;
    let ds_back_q: Value = serde_json::from_str(&ds_back_q_body).unwrap();
    assert_eq!(ds_back_q["model_id"], "deepseek-v3-671b-moe");
    assert_eq!(ds_back_q["effective_context_window_tokens"], 65_536);

    // Switching the active inference model to llama-3.2-3b-instruct starts at its native 131,072 context window
    let (switch_status, _) = request(
        &app,
        "POST",
        "/api/v1/cluster/model",
        json!({"inference_model_id": "llama-3.2-3b-instruct"}),
    )
    .await;
    assert_eq!(switch_status, 200);
    let (_, switched_q_body) =
        request(&app, "GET", "/api/v1/models/quality-tradeoffs", json!(null)).await;
    let switched_q: Value = serde_json::from_str(&switched_q_body).unwrap();
    assert_eq!(switched_q["model_id"], "llama-3.2-3b-instruct");
    assert_eq!(switched_q["is_uncompromised_maximum_quality"], true);
    assert_eq!(switched_q["effective_context_window_tokens"], 131_072);

    // Unknown model IDs report Option::None ("backend-reported") rather than a synthetic 32,768 fallback
    let mut unknown_spec = kadmium_core::ModelSpec::llama_3_2_3b_instruct();
    unknown_spec.id = "custom-unknown-7b".into();
    assert_eq!(unknown_spec.try_native_max_context_tokens(), None);
    let mut unknown_limits = kadmium_gateway::ModelQualityLimits::for_model(&unknown_spec);
    assert_eq!(unknown_limits.native_max_context_window_tokens, None);
    assert_eq!(unknown_limits.native_max_input_tokens, None);
    assert_eq!(unknown_limits.effective_context_window_tokens(), None);
    assert_eq!(unknown_limits.effective_max_input_tokens(), None);
    let unknown_report = unknown_limits.evaluate_tradeoffs(&unknown_spec);
    assert!(unknown_report.is_uncompromised_maximum_quality);
    assert_eq!(unknown_report.effective_context_window_tokens, None);
    assert_eq!(unknown_report.effective_max_input_tokens, None);
    assert_eq!(
        unknown_report.tradeoffs[0].native_maximum,
        "backend-reported"
    );
    assert_eq!(
        unknown_report.tradeoffs[0].active_setting,
        "backend-reported"
    );
    unknown_limits.user_override_context_window_tokens = Some(16_384);
    unknown_limits.user_override_max_input_tokens = Some(8_192);
    unknown_limits.sync_native_with_model(&unknown_spec);
    assert_eq!(
        unknown_limits.effective_context_window_tokens(),
        Some(16_384)
    );
    assert_eq!(unknown_limits.effective_max_input_tokens(), Some(8_192));
}

#[tokio::test]
async fn owner_resource_offers_and_draining_nodes_bound_bucket_mediums_and_persist() {
    let dir = temp_dir("bucket-offer-bounds");
    let state = Arc::new(Mutex::new({
        let mut s = GatewayState::new();
        s.persistent_state_dir = dir.clone();
        s.nodes[0].polite_state = kadmium_core::PoliteState::Draining;
        let custom_offer = kadmium_core::ResourceOffer {
            fast_memory_gb: 1.5,
            system_ram_gb: 3.0,
            nvme_gb: 8.0,
            reserved_ram_gb: 4.0,
            slots: 1,
            max_allocations: 1,
            requires_approval: false,
        };
        let node1_id = s.nodes[1].id.0.clone();
        s.grid_offers.insert(node1_id, custom_offer);
        s.save_persistent_state().unwrap();
        s
    }));
    let app = build_router(state.clone());

    let (med_status, med_body) = request(
        &app,
        "GET",
        "/api/v1/storage/pool-bucket/mediums",
        json!(null),
    )
    .await;
    assert_eq!(med_status, 200, "{med_body}");
    let med_json: Value = serde_json::from_str(&med_body).unwrap();
    let node_mediums = med_json["node_mediums"].as_array().unwrap();
    // Node 0 is PoliteState::Draining -> every medium has 0.0 available and 0.0 pledged
    for slice in node_mediums[0][2].as_array().unwrap() {
        assert_eq!(slice["available_capacity_gb"].as_f64(), Some(0.0));
        assert_eq!(slice["pledged_bucket_gb"].as_f64(), Some(0.0));
    }
    // Node 1 has custom ResourceOffer (fast_memory_gb=1.5, system_ram_gb=3.0, nvme_gb=8.0)
    for slice in node_mediums[1][2].as_array().unwrap() {
        let avail = slice["available_capacity_gb"].as_f64().unwrap();
        let pledged = slice["pledged_bucket_gb"].as_f64().unwrap();
        assert!(pledged <= avail);
        assert!(avail <= 8.0);
    }

    // Verify grid_offers survive journal reload
    let mut reloaded = GatewayState::new();
    reloaded.persistent_state_dir = dir.clone();
    assert!(reloaded.load_persistent_state().unwrap());
    let node1_id = reloaded.nodes[1].id.0.clone();
    assert_eq!(
        reloaded.grid_offers.get(&node1_id).map(|o| o.nvme_gb),
        Some(8.0)
    );
    std::fs::remove_dir_all(dir).unwrap();
}
