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

#[tokio::test]
async fn pre_quantized_import_requires_the_operator_to_accept_it() {
    let root = std::env::temp_dir().join(format!("condenser-pre-quantized-{}", std::process::id()));
    std::fs::create_dir_all(root.join("source")).unwrap();
    std::fs::write(
        root.join("source/config.json"),
        br#"{"model_type":"llama","max_position_embeddings":8192,"quantization_config":{"quant_method":"awq","bits":4}}"#,
    )
    .unwrap();
    std::fs::write(root.join("source/tokenizer.json"), b"{}").unwrap();
    std::fs::write(
        root.join("source/model.safetensors"),
        SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
            ("model.layers.0.mlp.qweight", "I32", vec![2], &[7; 8]),
            ("model.layers.0.mlp.scales", "F16", vec![2], &[1; 4]),
        ]),
    )
    .unwrap();
    let (addr, server) = bind(ServerOptions {
        port: 0,
        model_library: root.join("library"),
        ..Default::default()
    })
    .await
    .unwrap();
    let server = tokio::spawn(server);
    let client = reqwest::Client::new();
    let url = format!("http://{addr}/api/v1/checkpoints/import");
    let refused = client
        .post(&url)
        .json(&json!({"source_path":root.join("source")}))
        .send()
        .await
        .unwrap();
    assert!(!refused.status().is_success());
    let text = refused.text().await.unwrap();
    assert!(
        text.contains("already quantized") && text.contains("awq 4-bit (I32)"),
        "{text}"
    );
    let accepted = client
        .post(&url)
        .json(&json!({"source_path":root.join("source"),"accept_pre_quantized":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(accepted.status(), 200);
    let summary: Value = accepted.json().await.unwrap();
    assert_eq!(summary["pre_quantized"]["method"], "awq");
    assert_eq!(summary["pre_quantized"]["bits"], 4);
    assert_eq!(summary["pre_quantized"]["tensor_formats"], json!(["I32"]));
    let listed: Value = client
        .get(format!("http://{addr}/api/v1/checkpoints"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["checkpoints"][0]["pre_quantized"]["method"], "awq");
    server.abort();
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn split_across_machines_plans_stages_and_surfaces_insufficient_cluster_memory() {
    use kadmium_core::{NodeId, NodeTrustTier};
    use kadmium_data::checkpoint::placement::{
        InspectedCheckpointProfile, MachineMemoryBudget, PlanCheckpointRequest,
        plan_inspected_checkpoint_with_budgets,
    };
    use std::collections::BTreeSet;

    let root = std::env::temp_dir().join(format!("condenser-split-flow-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let source = root.join("source");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(
        source.join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "hidden_act": "silu"
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(source.join("tokenizer.json"), "{}").unwrap();
    let mut specs: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![16, 8]),
        ("model.norm.weight".into(), vec![8]),
        ("lm_head.weight".into(), vec![16, 8]),
    ];
    for layer in 0..4 {
        let p = format!("model.layers.{layer}.");
        specs.push((format!("{p}input_layernorm.weight"), vec![8]));
        specs.push((format!("{p}post_attention_layernorm.weight"), vec![8]));
        specs.push((format!("{p}self_attn.q_proj.weight"), vec![8, 8]));
        specs.push((format!("{p}self_attn.k_proj.weight"), vec![8, 8]));
        specs.push((format!("{p}self_attn.v_proj.weight"), vec![8, 8]));
        specs.push((format!("{p}self_attn.o_proj.weight"), vec![8, 8]));
        specs.push((format!("{p}mlp.gate_proj.weight"), vec![16, 8]));
        specs.push((format!("{p}mlp.up_proj.weight"), vec![16, 8]));
        specs.push((format!("{p}mlp.down_proj.weight"), vec![8, 16]));
    }
    let payloads: Vec<Vec<u8>> = specs
        .iter()
        .map(|(_, shape)| vec![0u8; shape.iter().product::<usize>() * 2])
        .collect();
    let entries: Vec<(&str, &str, Vec<usize>, &[u8])> = specs
        .iter()
        .zip(payloads.iter())
        .map(|((name, shape), bytes)| (name.as_str(), "BF16", shape.clone(), bytes.as_slice()))
        .collect();
    std::fs::write(
        source.join("model.safetensors"),
        SafetensorsToKtbTranscoder::build_safetensors_bytes(&entries),
    )
    .unwrap();

    let reject_budget = Arc::new(AtomicBool::new(false));
    let reject_for_coordinator = reject_budget.clone();
    let coordinator = Router::new().route(
        "/api/v1/checkpoints/placement",
        post(
            move |headers: axum::http::HeaderMap, Json(req): Json<PlanCheckpointRequest>| {
                let reject = reject_for_coordinator.clone();
                async move {
                    assert_eq!(
                        headers.get("authorization").and_then(|v| v.to_str().ok()),
                        Some("Bearer split-secret")
                    );
                    let profile = req.profile.expect("profile must be supplied");
                    let inspected = InspectedCheckpointProfile {
                        checkpoint_id: req.checkpoint_id,
                        checkpoint_name: req.checkpoint_name.unwrap_or_else(|| "source".into()),
                        architecture: req
                            .architecture
                            .unwrap_or_else(|| "LlamaForCausalLM".into()),
                        source_format: req.source_format.unwrap_or_else(|| "safetensors".into()),
                        profile: profile.clone(),
                    };
                    let per_layer = profile.layer_weight_bytes[0] + profile.kv_bytes_per_layer;
                    let s0 = profile.embedding_bytes + 2 * per_layer;
                    let s1 = 2 * per_layer + profile.final_norm_bytes + profile.lm_head_bytes;
                    let mut m0 = MachineMemoryBudget::new(
                        NodeId("node-a".into()),
                        s0,
                        NodeTrustTier::TrustedCore,
                    );
                    m0.hostname = "machine-alpha".into();
                    let mut m1 = MachineMemoryBudget::new(
                        NodeId("node-b".into()),
                        s1,
                        NodeTrustTier::TrustedCore,
                    );
                    m1.hostname = "machine-beta".into();
                    let budgets = if reject.load(Ordering::Relaxed) {
                        vec![m0]
                    } else {
                        vec![m0, m1]
                    };
                    let mut online = BTreeSet::new();
                    online.insert("node-a".to_string());
                    match plan_inspected_checkpoint_with_budgets(&inspected, &budgets, &online) {
                        Ok(view) => (StatusCode::OK, Json(serde_json::to_value(view).unwrap())),
                        Err(err) => (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            Json(json!({ "error": err.to_string() })),
                        ),
                    }
                }
            },
        ),
    );
    let coord_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let coord_url = format!("http://{}", coord_listener.local_addr().unwrap());
    let coord_task = tokio::spawn(async move {
        axum::serve(coord_listener, coordinator).await.unwrap();
    });

    let (addr, server) = bind(ServerOptions {
        port: 0,
        model_library: root.join("library"),
        ..Default::default()
    })
    .await
    .unwrap();
    let server_task = tokio::spawn(server);
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let imported: Value = client
        .post(format!("{base}/api/v1/checkpoints/import"))
        .json(&json!({ "source_path": &source }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = imported["id"].as_str().unwrap();

    let resp = client
        .post(format!("{base}/api/v1/checkpoints/split"))
        .json(&json!({
            "checkpoint_id": id,
            "endpoint": &coord_url,
            "api_key": "split-secret"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let view: Value = resp.json().await.unwrap();
    assert_eq!(view["is_split"], true);
    assert_eq!(view["context_length"], 64);
    assert_eq!(view["stages"].as_array().unwrap().len(), 2);
    assert_eq!(view["stages"][0]["hostname"], "machine-alpha");
    assert_eq!(view["stages"][0]["start_layer"], 0);
    assert_eq!(view["stages"][0]["end_layer"], 2);
    assert_eq!(view["stages"][0]["embedding"], true);
    assert_eq!(view["stages"][1]["hostname"], "machine-beta");
    assert_eq!(view["stages"][1]["start_layer"], 2);
    assert_eq!(view["stages"][1]["end_layer"], 4);
    assert_eq!(view["stages"][1]["head"], true);

    reject_budget.store(true, Ordering::Relaxed);
    let failed = client
        .post(format!("{base}/api/v1/checkpoints/split"))
        .json(&json!({
            "checkpoint_id": id,
            "endpoint": &coord_url,
            "api_key": "split-secret"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(failed.status(), 422);
    let text = failed.text().await.unwrap();
    assert!(
        text.contains("refusing to shorten the context or use a smaller model"),
        "{text}"
    );

    server_task.abort();
    coord_task.abort();
    let _ = std::fs::remove_dir_all(root);
}
