use crate::dsv41_mcl4::{
    CoeDecisionProbeReport, DeepSeekV41ArchSpec, Dsv41ArenaCachePlan, Dsv41Mcl4cPacketLayout,
    Mcl4CoeKnobs, discover_local_dsv41_shards_and_checkpoints, mla_context_budget,
    probe_decision_candidates,
};
use crate::plugin::{PluginInventory, PluginManager, default_plugin_roots};
use crate::provider::ProviderFactory;
use crate::types::{Message, ModelInfo, ProviderConfig, SamplingParams};
use crate::{CondenserError, DEFAULT_HTTP_PORT, Harness, HarnessOptions, Result, RunControl};
use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use kadmium_core::LiveHardwareProfiler;
use serde::Deserialize;
use serde_json::{Value, json};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

const CONDENSER_HTML: &str = include_str!("assets/condenser.html");
const CONDENSER_CSS: &str = include_str!("assets/condenser.css");
const CONDENSER_JS: &str = include_str!("assets/condenser.js");
const WORKFLOW_JS: &str = include_str!("assets/workflow.js");
const CONDENSER_MARK: &str = include_str!("assets/condenser-mark.svg");
const CONDENSER_FAVICON: &str = include_str!("assets/condenser-favicon.svg");
const CONDENSER_FONT_ARCHIVO: &[u8] = include_bytes!("assets/fonts/Archivo-Variable.woff2");
const CONDENSER_FONT_MONO: &[u8] = include_bytes!("assets/fonts/JetBrainsMono-Variable.woff2");

#[derive(Debug, Clone)]
pub struct ServerOptions {
    /// Zero asks the OS for an available loopback port.
    pub port: u16,
    pub plugin_roots: Vec<PathBuf>,
    /// Bounds resident agent turns so local backends are not overwhelmed.
    pub max_concurrent_turns: usize,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            port: DEFAULT_HTTP_PORT,
            plugin_roots: default_plugin_roots(),
            max_concurrent_turns: 2,
        }
    }
}

#[derive(Clone)]
pub(crate) struct ServerState {
    plugin_roots: Arc<Vec<PathBuf>>,
    client: reqwest::Client,
    turn_permits: Arc<tokio::sync::Semaphore>,
    active_runs: crate::runs::ActiveRuns,
    pub(crate) workspace: crate::workspace::WorkspaceState,
}

pub async fn serve(options: ServerOptions) -> Result<()> {
    let (address, server) = bind(options).await?;
    println!("Condenser {} is ready", env!("CARGO_PKG_VERSION"));
    println!("  Local UI:        http://{address}/");
    println!("  Health:          http://{address}/api/v1/health");
    println!("  Provider API:    http://{address}/api/v1/providers");
    println!("  Plugin API:      http://{address}/api/v1/plugins");
    println!("  DSV4.1 Profile:  http://{address}/api/v1/dsv41/profile");
    println!("  MCL4 Probe API:  http://{address}/api/v1/mcl4/probe-decision");
    server.await
}

/// Binds only the loopback interface. API keys are accepted by the request API
/// but are never written to disk, logs, or the persistent Kadmium state.
pub async fn bind(
    options: ServerOptions,
) -> Result<(SocketAddr, impl std::future::Future<Output = Result<()>>)> {
    if options.max_concurrent_turns == 0 {
        return Err(CondenserError::Config(
            "max_concurrent_turns must be at least 1".into(),
        ));
    }
    let client = reqwest::Client::builder()
        .user_agent(concat!("Condenser/", env!("CARGO_PKG_VERSION")))
        .tcp_nodelay(true)
        .pool_max_idle_per_host(32)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .build()
        .map_err(|error| CondenserError::provider("server", error.to_string()))?;
    let state = ServerState {
        plugin_roots: Arc::new(options.plugin_roots),
        client,
        turn_permits: Arc::new(tokio::sync::Semaphore::new(options.max_concurrent_turns)),
        active_runs: crate::runs::ActiveRuns::default(),
        workspace: crate::workspace::WorkspaceState::from_current_dir()?,
    };
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, options.port)).await?;
    let address = listener.local_addr()?;
    let app = router(state);
    Ok((address, async move {
        axum::serve(listener, app)
            .await
            .map_err(CondenserError::from)
    }))
}

fn router(state: ServerState) -> Router {
    let api = Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/system", get(system))
        .route("/api/v1/providers", get(providers))
        .route("/api/v1/models", post(models))
        .route("/api/v1/plugins", get(plugins))
        .route("/api/v1/plugins/invoke", post(invoke_plugin))
        .route("/api/v1/chat", post(chat))
        .route("/api/v1/chat/stream", post(chat_stream))
        .route("/api/v1/dsv41/profile", get(dsv41_profile))
        .route("/api/v1/mcl4/probe-decision", post(mcl4_probe_decision))
        .route("/api/v1/workspace", get(crate::workspace::workspace))
        .route("/api/v1/compute/inspect", post(crate::compute::inspect))
        .route("/api/v1/runs/{id}/cancel", post(cancel_run))
        .route("/api/v1/files/list", post(crate::workspace::list_files))
        .route("/api/v1/files/read", post(crate::workspace::read_file))
        .route("/api/v1/files/write", post(crate::workspace::write_file))
        .route("/api/v1/files/create", post(crate::workspace::create_file))
        .route("/api/v1/files/rename", post(crate::workspace::rename_file))
        .route("/api/v1/files/delete", post(crate::workspace::delete_file))
        .route(
            "/api/v1/terminal/execute",
            post(crate::workspace::execute_terminal),
        )
        .layer(middleware::from_fn(loopback_request_guard));
    Router::new()
        .route("/", get(index))
        .route("/condenser.css", get(condenser_css))
        .route("/condenser.js", get(condenser_js))
        .route(
            "/workflow.js",
            get(|| async { text_asset("application/javascript; charset=utf-8", WORKFLOW_JS) }),
        )
        .route("/assets/condenser-mark.svg", get(condenser_mark))
        .route("/assets/condenser-favicon.svg", get(condenser_favicon))
        .route("/fonts/Archivo-Variable.woff2", get(font_archivo))
        .route("/fonts/JetBrainsMono-Variable.woff2", get(font_mono))
        .merge(api)
        .with_state(state)
}

async fn loopback_request_guard(request: Request, next: Next) -> Response {
    if loopback_headers(request.headers()) {
        next.run(request).await
    } else {
        StatusCode::FORBIDDEN.into_response()
    }
}

fn loopback_headers(headers: &HeaderMap) -> bool {
    let mut hosts = headers.get_all(header::HOST).iter();
    let valid_host = hosts.next().is_some_and(|host| {
        hosts.next().is_none()
            && host
                .to_str()
                .ok()
                .and_then(|host| host.parse::<axum::http::uri::Authority>().ok())
                .is_some_and(|authority| loopback_authority(&authority))
    });
    if !valid_host {
        return false;
    }

    let mut origins = headers.get_all(header::ORIGIN).iter();
    let Some(origin) = origins.next() else {
        return true;
    };
    if origins.next().is_some() {
        return false;
    }
    origin
        .to_str()
        .ok()
        .and_then(|origin| origin.parse::<axum::http::Uri>().ok())
        .is_some_and(|uri| {
            uri.scheme_str()
                .is_some_and(|scheme| scheme.eq_ignore_ascii_case("http"))
                && uri.path_and_query().is_none_or(|path| path.as_str() == "/")
                && uri.authority().is_some_and(|authority| {
                    !authority.as_str().contains('@') && loopback_authority(authority)
                })
        })
}

fn loopback_authority(authority: &axum::http::uri::Authority) -> bool {
    let authority = authority.as_str();
    match authority.rsplit_once(':') {
        Some((host, port)) => {
            loopback_host(host)
                && !port.is_empty()
                && port.bytes().all(|byte| byte.is_ascii_digit())
                && port.parse::<u16>().is_ok()
        }
        None => loopback_host(authority),
    }
}

fn loopback_host(host: &str) -> bool {
    host == "127.0.0.1" || host.eq_ignore_ascii_case("localhost")
}

async fn index() -> Html<&'static str> {
    Html(CONDENSER_HTML)
}

async fn condenser_css() -> Response {
    text_asset("text/css; charset=utf-8", CONDENSER_CSS)
}

async fn condenser_js() -> Response {
    text_asset("application/javascript; charset=utf-8", CONDENSER_JS)
}

async fn condenser_mark() -> Response {
    text_asset("image/svg+xml", CONDENSER_MARK)
}

async fn condenser_favicon() -> Response {
    text_asset("image/svg+xml", CONDENSER_FAVICON)
}

async fn font_archivo() -> Response {
    binary_asset("font/woff2", CONDENSER_FONT_ARCHIVO)
}

async fn font_mono() -> Response {
    binary_asset("font/woff2", CONDENSER_FONT_MONO)
}

fn binary_asset(content_type: &'static str, body: &'static [u8]) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        body,
    )
        .into_response()
}

fn text_asset(content_type: &'static str, body: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        body,
    )
        .into_response()
}

async fn health() -> Json<Value> {
    Json(json!({
        "name": "Condenser",
        "suite": "Kadmium Power Plant",
        "version": env!("CARGO_PKG_VERSION"),
        "plugin_protocol_version": crate::PLUGIN_PROTOCOL_VERSION,
        "status": "ok"
    }))
}

async fn system() -> Result<Json<Value>, ApiError> {
    let node = tokio::task::spawn_blocking(LiveHardwareProfiler::probe_local_host)
        .await
        .map_err(|error| CondenserError::provider("system", error.to_string()))?;
    let vram_gb = node
        .memory
        .fast_accelerator_memory_gb()
        .max(node.available_fast_memory_gb());
    let ram_gb = node.memory.system_ram_gb;
    let dsv41_plan = Dsv41ArenaCachePlan::plan_for_host(vram_gb, ram_gb);
    let dual_nvme = discover_local_dsv41_shards_and_checkpoints();
    Ok(Json(json!({
        "hostname": node.hostname,
        "device_kind": format!("{:?}", node.device_kind),
        "backend": format!("{:?}", node.backend),
        "fast_memory_gb": vram_gb,
        "system_ram_gb": ram_gb,
        "fp16_tflops": node.effective_fp16_tflops(),
        "dsv41_arena_plan": dsv41_plan,
        "dual_nvme_shard_inventory": dual_nvme
    })))
}

async fn dsv41_profile() -> Result<Json<Value>, ApiError> {
    let node = tokio::task::spawn_blocking(LiveHardwareProfiler::probe_local_host)
        .await
        .map_err(|error| CondenserError::provider("system", error.to_string()))?;
    let arch = DeepSeekV41ArchSpec::deepseek_v41_flash_nvfp4();
    let layout = Dsv41Mcl4cPacketLayout::new();
    let vram_gb = node
        .memory
        .fast_accelerator_memory_gb()
        .max(node.available_fast_memory_gb());
    let arena_plan = Dsv41ArenaCachePlan::plan_for_host(vram_gb, node.memory.system_ram_gb);
    let dual_nvme = discover_local_dsv41_shards_and_checkpoints();
    let mla_8k = mla_context_budget(
        8192,
        arch.mla_sliding_window,
        arch.mla_compressed_kv_topk,
        arch.mla_compression_ratio,
    );
    Ok(Json(json!({
        "architecture": arch,
        "mcl4c_packet_layout": layout,
        "host_arena_cache_plan": arena_plan,
        "dual_nvme_shard_inventory": dual_nvme,
        "mla_budget_8k": mla_8k,
        "default_knobs": Mcl4CoeKnobs::deepseek_v41_default()
    })))
}

#[derive(Debug, Deserialize)]
struct ProbeDecisionApiRequest {
    #[serde(default)]
    context: String,
    #[serde(default = "default_probe_question")]
    question: String,
    options: Vec<String>,
    #[serde(default)]
    knobs: Option<Mcl4CoeKnobs>,
}

fn default_probe_question() -> String {
    "Select the best option.".to_string()
}

async fn mcl4_probe_decision(
    Json(request): Json<ProbeDecisionApiRequest>,
) -> Result<Json<CoeDecisionProbeReport>, ApiError> {
    let knobs = request
        .knobs
        .unwrap_or_else(Mcl4CoeKnobs::deepseek_v41_default);
    let report = tokio::task::spawn_blocking(move || {
        probe_decision_candidates(
            &request.context,
            &request.question,
            &request.options,
            &knobs,
        )
    })
    .await
    .map_err(|error| CondenserError::provider("mcl4-probe", error.to_string()))?
    .map_err(CondenserError::Config)?;
    Ok(Json(report))
}

async fn providers() -> Json<Value> {
    Json(json!({"providers": ProviderFactory::descriptors()}))
}

#[derive(Debug, Deserialize)]
struct ModelsRequest {
    provider: ProviderConfig,
}

async fn models(
    State(state): State<ServerState>,
    Json(request): Json<ModelsRequest>,
) -> Result<Json<Vec<ModelInfo>>, ApiError> {
    let provider_cfg = request.provider;
    let timeout = provider_cfg.timeout_secs;
    let kind = provider_cfg.kind;
    let provider = ProviderFactory::build_with_client(provider_cfg, state.client)?;
    let models = tokio::time::timeout(
        std::time::Duration::from_secs(timeout),
        provider.list_models(),
    )
    .await
    .map_err(|_| {
        CondenserError::provider(
            kind.label(),
            format!("model discovery timed out after {timeout} seconds"),
        )
    })??;
    Ok(Json(models))
}

async fn plugins(State(state): State<ServerState>) -> Result<Json<PluginInventory>, ApiError> {
    let manager = PluginManager::new(state.plugin_roots.as_ref().clone());
    Ok(Json(manager.discover()?))
}

#[derive(Debug, Deserialize)]
struct PluginInvokeRequest {
    plugin_id: String,
    tool: String,
    #[serde(default)]
    arguments: Value,
}

async fn invoke_plugin(
    State(state): State<ServerState>,
    Json(request): Json<PluginInvokeRequest>,
) -> Result<Json<crate::ToolResult>, ApiError> {
    let manager = PluginManager::new(state.plugin_roots.as_ref().clone());
    Ok(Json(
        manager
            .invoke(&request.plugin_id, &request.tool, request.arguments)
            .await?,
    ))
}

#[derive(Debug, Deserialize)]
struct ChatApiRequest {
    #[serde(default)]
    run_id: Option<String>,
    provider: ProviderConfig,
    messages: Vec<Message>,
    #[serde(default)]
    enable_tools: bool,
    #[serde(default)]
    allowed_tools: Option<Vec<String>>,
    #[serde(default = "default_max_steps")]
    max_steps: usize,
    #[serde(default = "default_max_calls_per_step")]
    max_calls_per_step: usize,
    #[serde(default)]
    max_run_seconds: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    max_tool_calls: Option<usize>,
    #[serde(default)]
    sampling: SamplingParams,
}

async fn cancel_run(State(state): State<ServerState>, Path(id): Path<String>) -> Response {
    if state.active_runs.cancel(&id) {
        Json(json!({"status": "cancelling", "run_id": id})).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Run is no longer active"})),
        )
            .into_response()
    }
}

fn default_max_steps() -> usize {
    8
}

fn default_max_calls_per_step() -> usize {
    16
}

async fn chat(
    State(state): State<ServerState>,
    Json(request): Json<ChatApiRequest>,
) -> Result<Response, ApiError> {
    let _permit = match state.turn_permits.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            return Ok((
                StatusCode::TOO_MANY_REQUESTS,
                [(header::RETRY_AFTER, "1")],
                Json(json!({
                    "error": "Condenser is at its concurrent-turn limit; retry shortly"
                })),
            )
                .into_response());
        }
    };
    if request
        .max_run_seconds
        .is_some_and(|seconds| seconds == 0 || seconds > 86_400)
    {
        return Err(
            CondenserError::Config("max_run_seconds must be between 1 and 86400".into()).into(),
        );
    }
    let run = state.active_runs.register(request.run_id)?;
    let control = RunControl {
        cancellation: run.token.clone(),
        deadline: request
            .max_run_seconds
            .map(|seconds| tokio::time::Instant::now() + std::time::Duration::from_secs(seconds)),
        max_output_tokens: request.max_output_tokens,
        max_tool_calls: request.max_tool_calls,
    };
    let plugins = PluginManager::new(state.plugin_roots.as_ref().clone());
    let provider_cfg = request.provider;
    let harness = Harness::new_with_client(provider_cfg, plugins, state.client)?;
    let report = harness
        .run_with_control(
            request.messages,
            HarnessOptions {
                enable_tools: request.enable_tools,
                allowed_tools: request.allowed_tools,
                max_steps: request.max_steps,
                max_calls_per_step: request.max_calls_per_step,
                sampling: request.sampling,
            },
            control,
            None,
        )
        .await?;
    Ok(Json(report).into_response())
}

struct StreamCancelOnDrop(tokio_util::sync::CancellationToken);

impl Drop for StreamCancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

async fn chat_stream(
    State(state): State<ServerState>,
    Json(request): Json<ChatApiRequest>,
) -> Result<Response, ApiError> {
    let permit = match state.turn_permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return Ok((
                StatusCode::TOO_MANY_REQUESTS,
                [(header::RETRY_AFTER, "1")],
                Json(json!({
                    "error": "Condenser is at its concurrent-turn limit; retry shortly"
                })),
            )
                .into_response());
        }
    };
    if request
        .max_run_seconds
        .is_some_and(|seconds| seconds == 0 || seconds > 86_400)
    {
        return Err(
            CondenserError::Config("max_run_seconds must be between 1 and 86400".into()).into(),
        );
    }
    if request.messages.is_empty() {
        return Err(
            CondenserError::Config("a harness turn needs at least one message".into()).into(),
        );
    }
    if !(1..=32).contains(&request.max_steps) {
        return Err(CondenserError::Config("max_steps must be between 1 and 32".into()).into());
    }
    if request.max_calls_per_step == 0 {
        return Err(CondenserError::Config("max_calls_per_step must be at least 1".into()).into());
    }
    request.sampling.validate()?;

    let run = state.active_runs.register(request.run_id)?;
    let cancel_token = run.token.clone();
    let control = RunControl {
        cancellation: cancel_token.clone(),
        deadline: request
            .max_run_seconds
            .map(|seconds| tokio::time::Instant::now() + std::time::Duration::from_secs(seconds)),
        max_output_tokens: request.max_output_tokens,
        max_tool_calls: request.max_tool_calls,
    };
    let plugins = PluginManager::new(state.plugin_roots.as_ref().clone());
    let provider_cfg = request.provider;
    let harness = Harness::new_with_client(provider_cfg, plugins, state.client)?;
    let (tx, rx) = tokio::sync::mpsc::channel::<crate::harness::HarnessEvent>(1024);
    let options = HarnessOptions {
        enable_tools: request.enable_tools,
        allowed_tools: request.allowed_tools,
        max_steps: request.max_steps,
        max_calls_per_step: request.max_calls_per_step,
        sampling: request.sampling,
    };
    let messages = request.messages;
    let task = tokio::spawn(async move {
        let _permit = permit;
        let _run = run;
        harness
            .run_with_control(messages, options, control, Some(tx))
            .await
    });

    let cancel_guard = StreamCancelOnDrop(cancel_token);
    let stream = futures_util::stream::unfold(
        (rx, Some(task), cancel_guard),
        |(mut rx, mut task, cancel_guard)| async move {
            let event = loop {
                match rx.recv().await {
                    // The task result is authoritative even if the bounded
                    // progress channel was full when the harness finished.
                    Some(crate::HarnessEvent::Finished { .. }) => continue,
                    Some(event) => break serde_json::to_value(event).unwrap_or_default(),
                    None => {
                        let result = task.take()?.await;
                        break match result {
                            Ok(Ok(report)) => json!({"event": "finished", "report": report}),
                            Ok(Err(error)) => json!({"event": "error", "error": error.to_string()}),
                            Err(_) => {
                                json!({"event": "error", "error": "Run task terminated unexpectedly"})
                            }
                        };
                    }
                }
            };
            let line = format!(
                "data: {}\n\n",
                serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string())
            );
            Some((
                Ok::<String, std::convert::Infallible>(line),
                (rx, task, cancel_guard),
            ))
        },
    );

    Ok((
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        axum::body::Body::from_stream(stream),
    )
        .into_response())
}

pub(crate) struct ApiError(CondenserError);

impl<E> From<E> for ApiError
where
    E: Into<CondenserError>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            CondenserError::Config(_)
            | CondenserError::UnsupportedOption { .. }
            | CondenserError::UnknownTool(_)
            | CondenserError::StepLimit(_) => StatusCode::BAD_REQUEST,
            CondenserError::Plugin(_) | CondenserError::Provider { .. } => StatusCode::BAD_GATEWAY,
            CondenserError::Serialization(_) | CondenserError::Io(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        (status, Json(json!({"error": self.0.to_string()}))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn guarded_headers(host: &str, origin: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_str(host).unwrap());
        if let Some(origin) = origin {
            headers.insert(header::ORIGIN, HeaderValue::from_str(origin).unwrap());
        }
        headers
    }

    #[test]
    fn request_guard_requires_loopback_host_and_origin() {
        assert!(loopback_headers(&guarded_headers("127.0.0.1:7856", None)));
        assert!(loopback_headers(&guarded_headers("localhost", None)));
        assert!(loopback_headers(&guarded_headers(
            "LOCALHOST:7856",
            Some("http://127.0.0.1:7856")
        )));
        assert!(!loopback_headers(&HeaderMap::new()));
        assert!(!loopback_headers(&guarded_headers("example.com", None)));
        assert!(!loopback_headers(&guarded_headers(
            "127.0.0.1:invalid",
            None
        )));
        assert!(!loopback_headers(&guarded_headers(
            "localhost",
            Some("https://localhost")
        )));
        assert!(!loopback_headers(&guarded_headers(
            "localhost",
            Some("http://example.com")
        )));
        assert!(!loopback_headers(&guarded_headers(
            "localhost",
            Some("null")
        )));
    }

    #[tokio::test]
    async fn loopback_server_exposes_health_and_plugin_inventory() {
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("condenser-server-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let (address, future) = bind(ServerOptions {
            port: 0,
            plugin_roots: vec![root.clone()],
            ..ServerOptions::default()
        })
        .await
        .unwrap();
        let task = tokio::spawn(future);
        let client = reqwest::Client::new();
        let health: Value = client
            .get(format!("http://{address}/api/v1/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(health["name"], "Condenser");
        assert_eq!(health["plugin_protocol_version"], 1);
        let plugins: Value = client
            .get(format!("http://{address}/api/v1/plugins"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(plugins["plugins"].as_array().unwrap().len(), 0);
        let dsv41: Value = client
            .get(format!("http://{address}/api/v1/dsv41/profile"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(dsv41["architecture"]["n_layers"], 40);
        assert_eq!(dsv41["architecture"]["n_routed_experts"], 384);
        assert_eq!(
            dsv41["mcl4c_packet_layout"]["total_packet_bytes"],
            19_910_656
        );
        let probe: Value = client
            .post(format!("http://{address}/api/v1/mcl4/probe-decision"))
            .json(&json!({
                "context": "DeepSeek-V4.1 uses 384 routed NVFP4 experts per layer with top-6 routing.",
                "question": "Which configuration matches DeepSeek-V4.1?",
                "options": [
                    "384 routed NVFP4 experts with top-6 routing",
                    "8 dense FP32 experts with top-1 routing"
                ]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(probe["winner_index"], 0);
        let page = client
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(page.contains("Condenser"));
        assert!(page.contains("Kadmium Power Plant"));
        assert!(page.contains("id=\"timeline\""));
        assert!(page.contains("id=\"runbook\""));
        assert!(page.contains("id=\"settings-layer\""));
        let css = client
            .get(format!("http://{address}/condenser.css"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(css.contains(".agent-stage"));
        assert!(css.contains(".runbook"));
        let js = client
            .get(format!("http://{address}/condenser.js"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(js.contains("condenser.agent.tasks.v1"));
        assert!(js.contains("function renderRunbook"));
        task.abort();
        let _ = std::fs::remove_dir_all(root);
    }
}
