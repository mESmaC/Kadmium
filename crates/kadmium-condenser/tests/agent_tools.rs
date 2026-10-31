use async_trait::async_trait;
use axum::{
    Json, Router,
    routing::{get, post},
};
use kadmium_condenser::agent::{AgentConfig, AgentTools, SubagentProfile};
use kadmium_condenser::harness::ToolExecutor;
use kadmium_condenser::plugin::PluginManager;
use kadmium_condenser::provider::InferenceProvider;
use kadmium_condenser::{ChatRequest, ChatResponse, ModelInfo, Result};
use kadmium_condenser::{
    Harness, HarnessOptions, Message, ProviderConfig, ProviderKind, RunControl, RunStatus,
    SamplingParams, ToolCall, Usage,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "condenser-agent-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("workspace")).unwrap();
        Self { root }
    }
    fn tools(
        &self,
        session: Option<&str>,
        research: bool,
        profiles: BTreeMap<String, SubagentProfile>,
    ) -> Arc<AgentTools> {
        Arc::new(
            AgentTools::new(
                PluginManager::new(vec![]),
                self.root.join("workspace"),
                self.root.join("memory"),
                AgentConfig {
                    session_id: session.map(str::to_owned),
                    research_required: research,
                },
                profiles,
            )
            .unwrap(),
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

async fn invoke(tools: &AgentTools, name: &str, args: Value) -> Value {
    let (catalog, _) = tools.registered_tools().unwrap();
    let tool = catalog
        .iter()
        .find(|t| t.unqualified_name() == name)
        .unwrap();
    assert!(
        jsonschema::validator_for(&tool.definition.input_schema)
            .unwrap()
            .is_valid(&args)
    );
    let result = tools.execute(tool, args).await.unwrap();
    assert!(!result.is_error, "{}", result.content);
    serde_json::from_str(&result.content).unwrap()
}

struct Scripted {
    replies: Mutex<VecDeque<Message>>,
    seen: Arc<Mutex<Vec<ChatRequest>>>,
}
#[async_trait]
impl InferenceProvider for Scripted {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![])
    }
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(request.clone());
        Ok(ChatResponse {
            model: request.model.clone(),
            message: self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected model call"),
            finish_reason: Some("stop".into()),
            usage: Usage {
                input_tokens: 2,
                output_tokens: 3,
                total_tokens: 5,
            },
            telemetry: None,
        })
    }
}
fn call(name: &str, args: Value) -> Message {
    let mut message = Message::assistant("");
    message.tool_calls.push(ToolCall {
        id: format!("call-{name}"),
        name: format!("condenser__{name}"),
        arguments_json: args.to_string(),
    });
    message
}
fn harness(
    tools: Arc<AgentTools>,
    replies: Vec<Message>,
) -> (Harness, Arc<Mutex<Vec<ChatRequest>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(Scripted {
        replies: Mutex::new(replies.into()),
        seen: seen.clone(),
    });
    let mut config = ProviderConfig::new(ProviderKind::OpenAi);
    config.model = "parent-exact-model".into();
    (
        Harness::from_components(config, provider, tools).unwrap(),
        seen,
    )
}
fn options(names: &[&str]) -> HarnessOptions {
    HarnessOptions {
        enable_tools: true,
        allowed_tools: Some(names.iter().map(|s| format!("condenser__{s}")).collect()),
        ..HarnessOptions::default()
    }
}

#[tokio::test]
async fn lossless_indexed_memory_survives_restart_and_is_workspace_scoped() {
    let fixture = Fixture::new();
    let tools = fixture.tools(Some("task-a"), false, BTreeMap::new());
    let text = "Memory boundary 🦀\n".repeat(2000);
    let saved = invoke(
        &tools,
        "notes_write",
        json!({"title":"Decision","content":text}),
    )
    .await;
    let id = saved["id"].as_str().unwrap();
    drop(tools);
    let tools = fixture.tools(Some("task-b"), false, BTreeMap::new());
    let search = invoke(&tools, "memory_search", json!({"query":"MEM bound"})).await;
    assert_eq!(search["matches"][0]["id"], id);
    assert_eq!(search["matches"][0]["session"], "task-a");
    assert!(search["matches"][0].get("content").is_none());
    let mut restored = String::new();
    let mut offset = 0;
    loop {
        let read = invoke(&tools, "memory_read", json!({"id":id,"offset":offset})).await;
        restored.push_str(read["page"]["content"].as_str().unwrap());
        match read["page"]["next_offset"].as_u64() {
            Some(next) => offset = next,
            None => break,
        }
    }
    assert_eq!(restored, text);
    std::fs::create_dir(fixture.root.join("other")).unwrap();
    let other = AgentTools::new(
        PluginManager::new(vec![]),
        fixture.root.join("other"),
        fixture.root.join("memory"),
        AgentConfig {
            session_id: Some("task-a".into()),
            ..Default::default()
        },
        BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(
        invoke(&other, "memory_search", json!({"query":""})).await["total"],
        0
    );
    let ephemeral = fixture.tools(None, false, BTreeMap::new());
    assert_eq!(
        invoke(&ephemeral, "memory_search", json!({"query":""})).await["total"],
        0
    );
    invoke(
        &ephemeral,
        "notes_write",
        json!({"title":"Volatile","content":"Only this turn"}),
    )
    .await;
    assert_eq!(
        invoke(
            &fixture.tools(None, false, BTreeMap::new()),
            "memory_search",
            json!({"query":""})
        )
        .await["total"],
        0
    );
}

#[tokio::test]
async fn research_gate_requires_successful_source_and_preserves_full_context() {
    let fixture = Fixture::new();
    std::fs::write(fixture.root.join("workspace/evidence.txt"), "Verified fact").unwrap();
    let tools = fixture.tools(None, true, BTreeMap::new());
    let (agent, seen) = harness(
        tools,
        vec![
            call("file_read", json!({"path":"evidence.txt"})),
            Message::assistant("Verified fact [evidence.txt]"),
            Message::assistant("An unverified new turn"),
        ],
    );
    let input = Message::user("A long prompt ".repeat(5000));
    let report = agent
        .run(vec![input.clone()], options(&["file_read"]))
        .await
        .unwrap();
    assert_eq!(report.status, RunStatus::Completed);
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0].messages[1], input);
    assert_eq!(seen[0].sampling, SamplingParams::default());
    assert_eq!(seen[0].tools.len(), 1);
    assert_eq!(seen[1].messages[1], input);
    assert_eq!(report.messages[0], input); // policy is not duplicated into stored history
    let report = agent
        .run(vec![Message::user("New fact")], options(&["file_read"]))
        .await
        .unwrap();
    assert_eq!(report.status, RunStatus::Failed); // prior-turn evidence cannot satisfy this turn
    let (blocked, seen) = harness(fixture.tools(None, true, BTreeMap::new()), vec![]);
    assert!(
        blocked
            .run(vec![Message::user("Research")], options(&[]))
            .await
            .is_err()
    );
    assert!(seen.lock().unwrap().is_empty());
    let (agent, _) = harness(
        fixture.tools(None, true, BTreeMap::new()),
        vec![
            call("file_read", json!({"path":"missing.txt"})),
            Message::assistant("Unsupported claim"),
        ],
    );
    let report = agent
        .run(vec![Message::user("Verify")], options(&["file_read"]))
        .await
        .unwrap();
    assert_eq!(report.status, RunStatus::Failed);
    assert!(report.error.unwrap().contains("Research required"));
    assert!(report.steps[0].tool_executions[0].result.is_error);
}

#[tokio::test]
async fn excluded_tools_cannot_execute_and_edits_require_an_exact_match() {
    let fixture = Fixture::new();
    std::fs::write(fixture.root.join("workspace/code.txt"), "old value").unwrap();
    std::fs::write(fixture.root.join("workspace/.env"), "secret").unwrap();
    let (agent, _) = harness(
        fixture.tools(None, false, BTreeMap::new()),
        vec![
            call(
                "file_edit",
                json!({"path":"code.txt","old_text":"old","new_text":"new"}),
            ),
            Message::assistant("done"),
        ],
    );
    let report = agent
        .run(vec![Message::user("Inspect")], options(&["file_read"]))
        .await
        .unwrap();
    assert!(report.steps[0].tool_executions[0].result.is_error);
    assert_eq!(
        std::fs::read_to_string(fixture.root.join("workspace/code.txt")).unwrap(),
        "old value"
    );
    let tools = fixture.tools(None, false, BTreeMap::new());
    invoke(
        &tools,
        "file_edit",
        json!({"path":"code.txt","old_text":"old","new_text":"new"}),
    )
    .await;
    let (catalog, _) = tools.registered_tools().unwrap();
    let read = catalog
        .iter()
        .find(|t| t.unqualified_name() == "file_read")
        .unwrap();
    for path in [".env", "../outside"] {
        assert!(tools.execute(read, json!({"path":path})).await.is_err());
    }
    let edit = catalog
        .iter()
        .find(|t| t.unqualified_name() == "file_edit")
        .unwrap();
    assert!(
        tools
            .execute(
                edit,
                json!({"path":"code.txt","old_text":"old","new_text":"stale"})
            )
            .await
            .is_err()
    );
    assert_eq!(
        std::fs::read_to_string(fixture.root.join("workspace/code.txt")).unwrap(),
        "new value"
    );
}

async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    (
        url,
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
    )
}
fn profile(url: String) -> SubagentProfile {
    SubagentProfile {
        provider: ProviderKind::OpenAi,
        model: "user-selected-child".into(),
        base_url: url,
        api_key_env: None,
        sampling: SamplingParams::default(),
        timeout_secs: 0,
        allowed_tools: vec![
            "condenser__file_read".into(),
            "condenser__command".into(),
            "condenser__delegate".into(),
        ],
        max_steps: 8,
    }
}

#[tokio::test]
async fn delegate_uses_exact_target_native_output_limit_and_intersects_tools() {
    let fixture = Fixture::new();
    let requests = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured = requests.clone();
    let (url,server)=serve(Router::new().route("/v1/models",get(||async {Json(json!({"data":[{"id":"user-selected-child","max_output_tokens":65536}]}))}))
        .route("/v1/chat/completions",post(move |Json(body):Json<Value>| {let captured=captured.clone();async move {
            captured.lock().unwrap().push(body);
            Json(json!({"model":"user-selected-child","choices":[{"message":{"role":"assistant","content":"Child result"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28}}))
        }}))).await;
    let tools = fixture.tools(
        Some("parent"),
        false,
        BTreeMap::from([("local-review".into(), profile(url))]),
    );
    let (agent, _) = harness(
        tools.clone(),
        vec![
            call(
                "delegate",
                json!({"target":"local-review","task":"Explicit child brief"}),
            ),
            Message::assistant("Checked result"),
        ],
    );
    let report = agent
        .run(
            vec![Message::user("Parent-only context")],
            options(&["delegate", "file_read"]),
        )
        .await
        .unwrap();
    assert_eq!(report.status, RunStatus::Completed);
    assert_eq!(report.usage.output_tokens, 23);
    assert_eq!(report.usage.total_tokens, 38);
    let requests = requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["model"], "user-selected-child");
    assert_eq!(requests[0]["max_tokens"], 65536);
    assert_eq!(
        requests[0]["messages"][1]["content"],
        "Explicit child brief"
    );
    let child_tools = requests[0]["tools"].as_array().unwrap();
    assert_eq!(child_tools.len(), 1);
    assert_eq!(child_tools[0]["function"]["name"], "condenser__file_read");
    let saved = invoke(&tools, "memory_search", json!({"query":"Child result"})).await;
    assert!(saved["total"].as_u64().unwrap() > 0);
    server.abort();
}

#[tokio::test]
async fn child_usage_exhausts_parent_budget_and_cancellation_stops_waiting() {
    let fixture = Fixture::new();
    let (url,server)=serve(Router::new().route("/v1/chat/completions",post(||async {Json(json!({"model":"user-selected-child","choices":[{"message":{"role":"assistant","content":"long"},"finish_reason":"stop"}],"usage":{"completion_tokens":20,"total_tokens":20}}))}))).await;
    let mut p = profile(url);
    p.sampling.max_tokens = Some(50);
    let (agent, seen) = harness(
        fixture.tools(None, false, BTreeMap::from([("child".into(), p)])),
        vec![call("delegate", json!({"target":"child","task":"Brief"}))],
    );
    let report = agent
        .run_with_control(
            vec![Message::user("Task")],
            options(&["delegate"]),
            RunControl {
                max_output_tokens: Some(10),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    assert_eq!(report.status, RunStatus::BudgetExceeded);
    assert_eq!(report.usage.output_tokens, 23);
    assert_eq!(seen.lock().unwrap().len(), 1);
    server.abort();

    let started = Arc::new(tokio::sync::Notify::new());
    let signal = started.clone();
    let (url, server) = serve(Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let signal = signal.clone();
            async move {
                signal.notify_one();
                std::future::pending::<Json<Value>>().await
            }
        }),
    ))
    .await;
    let mut p = profile(url);
    p.sampling.max_tokens = Some(50);
    let (agent, _) = harness(
        fixture.tools(None, false, BTreeMap::from([("child".into(), p)])),
        vec![call("delegate", json!({"target":"child","task":"Wait"}))],
    );
    let control = RunControl::default();
    let cancel = control.cancellation.clone();
    let run = tokio::spawn(async move {
        agent
            .run_with_control(
                vec![Message::user("Task")],
                options(&["delegate"]),
                control,
                None,
            )
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(3), started.notified())
        .await
        .unwrap();
    cancel.cancel();
    let report = tokio::time::timeout(Duration::from_secs(3), run)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(report.status, RunStatus::Cancelled);
    assert_eq!(
        report.messages.last().unwrap().role,
        kadmium_condenser::Role::Tool
    );
    server.abort();
}

#[tokio::test]
async fn fetched_sources_remain_complete_and_failed_fetches_do_not_satisfy_research() {
    let fixture = Fixture::new();
    let full = "verified source 🦀 ".repeat(2000);
    let source = full.clone();
    let (url, server) = serve(Router::new().route(
        "/source",
        get(move || {
            let source = source.clone();
            async move { source }
        }),
    ))
    .await;
    let tools = fixture.tools(None, true, BTreeMap::new());
    let result = invoke(&tools, "web_fetch", json!({"url":format!("{url}/source")})).await;
    assert!(tools.completion_error().is_none());
    assert!(result["page"]["next_offset"].is_number());
    let mut content = result["page"]["content"].as_str().unwrap().to_owned();
    let mut offset = result["page"]["next_offset"].as_u64();
    while let Some(next) = offset {
        let read = invoke(
            &tools,
            "memory_read",
            json!({"id":result["record_id"],"offset":next}),
        )
        .await;
        content.push_str(read["page"]["content"].as_str().unwrap());
        offset = read["page"]["next_offset"].as_u64();
    }
    assert_eq!(content, full);
    let tools = fixture.tools(None, true, BTreeMap::new());
    let (catalog, _) = tools.registered_tools().unwrap();
    let fetch = catalog
        .iter()
        .find(|t| t.unqualified_name() == "web_fetch")
        .unwrap();
    assert!(
        tools
            .execute(fetch, json!({"url":format!("{url}/missing")}))
            .await
            .is_err()
    );
    assert!(tools.completion_error().is_some());
    server.abort();
}

#[tokio::test]
async fn ephemeral_child_notes_remain_retrievable_by_the_parent() {
    let fixture = Fixture::new();
    let count = Arc::new(AtomicU64::new(0));
    let (url, server) = serve(Router::new().route("/v1/chat/completions", post(move || {
        let count = count.clone();
        async move {
            let message = if count.fetch_add(1, Ordering::Relaxed) == 0 {
                json!({"role":"assistant","content":"","tool_calls":[{"id":"note","type":"function","function":{"name":"condenser__notes_write","arguments":"{\"title\":\"Child evidence\",\"content\":\"retained-exact-evidence\"}"}}]})
            } else { json!({"role":"assistant","content":"Evidence is saved"}) };
            Json(json!({"model":"user-selected-child","choices":[{"message":message,"finish_reason":"stop"}],"usage":{"completion_tokens":1,"total_tokens":1}}))
        }
    }))).await;
    let mut p = profile(url);
    p.sampling.max_tokens = Some(50);
    p.allowed_tools = vec!["condenser__notes_write".into()];
    let tools = fixture.tools(None, false, BTreeMap::from([("child".into(), p)]));
    let (agent, _) = harness(
        tools.clone(),
        vec![
            call("delegate", json!({"target":"child","task":"Save evidence"})),
            Message::assistant("Done"),
        ],
    );
    let report = agent
        .run(
            vec![Message::user("Task")],
            options(&["delegate", "notes_write"]),
        )
        .await
        .unwrap();
    assert_eq!(report.status, RunStatus::Completed);
    let search = invoke(&tools, "memory_search", json!({"query":"retained exact"})).await;
    let note = search["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["kind"] == "note")
        .unwrap();
    let read = invoke(&tools, "memory_read", json!({"id":note["id"]})).await;
    assert_eq!(read["page"]["content"], "retained-exact-evidence");
    assert!(!fixture.root.join("memory").exists());
    server.abort();
}
