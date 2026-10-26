use crate::plugin::{PluginInventory, PluginManager, default_plugin_roots};
use crate::provider::ProviderFactory;
use crate::types::{Message, ModelInfo, ProviderConfig, SamplingParams};
use crate::{CondenserError, DEFAULT_HTTP_PORT, Harness, HarnessOptions, HarnessReport, Result};
use axum::extract::State;
use axum::http::{StatusCode, header};
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
const DS_TOKENS_CSS: &str = include_str!("../../../design-system/tokens.css");
const DS_BUNDLE_CSS: &str = include_str!("../../../design-system/components/bundle.css");
const DS_WORDMARK: &str =
    include_str!("../../../design-system/assets/Logos/kadmium-wordmark-dark.svg");
const DS_MARK: &str = include_str!("../../../design-system/assets/Logos/kadmium-mark-dark.svg");
const DS_FAVICON: &str = include_str!("../../../design-system/assets/Logos/kadmium-favicon.svg");
const DS_FONT_ARCHIVO: &[u8] =
    include_bytes!("../../../design-system/fonts/Archivo-Variable.woff2");
const DS_FONT_MONO: &[u8] =
    include_bytes!("../../../design-system/fonts/JetBrainsMono-Variable.woff2");

#[derive(Debug, Clone)]
pub struct ServerOptions {
    /// Zero asks the OS for an available loopback port.
    pub port: u16,
    pub plugin_roots: Vec<PathBuf>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            port: DEFAULT_HTTP_PORT,
            plugin_roots: default_plugin_roots(),
        }
    }
}

#[derive(Clone)]
struct ServerState {
    plugin_roots: Arc<Vec<PathBuf>>,
}

pub async fn serve(options: ServerOptions) -> Result<()> {
    let (address, server) = bind(options).await?;
    println!("Condenser {} is ready", env!("CARGO_PKG_VERSION"));
    println!("  Local UI:      http://{address}/");
    println!("  Health:        http://{address}/api/v1/health");
    println!("  Provider API:  http://{address}/api/v1/providers");
    println!("  Plugin API:    http://{address}/api/v1/plugins");
    server.await
}

/// Binds only the loopback interface. API keys are accepted by the request API
/// but are never written to disk, logs, or the persistent Kadmium state.
pub async fn bind(
    options: ServerOptions,
) -> Result<(SocketAddr, impl std::future::Future<Output = Result<()>>)> {
    let state = ServerState {
        plugin_roots: Arc::new(options.plugin_roots),
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
    Router::new()
        .route("/", get(index))
        .route("/tokens.css", get(tokens_css))
        .route("/components/bundle.css", get(bundle_css))
        .route("/condenser.css", get(condenser_css))
        .route("/condenser.js", get(condenser_js))
        .route("/assets/Logos/kadmium-wordmark-dark.svg", get(wordmark))
        .route("/assets/Logos/kadmium-mark-dark.svg", get(mark))
        .route("/assets/Logos/kadmium-favicon.svg", get(favicon))
        .route("/fonts/Archivo-Variable.woff2", get(font_archivo))
        .route("/fonts/JetBrainsMono-Variable.woff2", get(font_mono))
        .route("/api/v1/health", get(health))
        .route("/api/v1/system", get(system))
        .route("/api/v1/providers", get(providers))
        .route("/api/v1/models", post(models))
        .route("/api/v1/plugins", get(plugins))
        .route("/api/v1/plugins/invoke", post(invoke_plugin))
        .route("/api/v1/chat", post(chat))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(CONDENSER_HTML)
}

async fn tokens_css() -> Response {
    text_asset("text/css; charset=utf-8", DS_TOKENS_CSS)
}

async fn bundle_css() -> Response {
    text_asset("text/css; charset=utf-8", DS_BUNDLE_CSS)
}

async fn condenser_css() -> Response {
    text_asset("text/css; charset=utf-8", CONDENSER_CSS)
}

async fn condenser_js() -> Response {
    text_asset("application/javascript; charset=utf-8", CONDENSER_JS)
}

async fn wordmark() -> Response {
    text_asset("image/svg+xml", DS_WORDMARK)
}

async fn mark() -> Response {
    text_asset("image/svg+xml", DS_MARK)
}

async fn favicon() -> Response {
    text_asset("image/svg+xml", DS_FAVICON)
}

async fn font_archivo() -> Response {
    binary_asset("font/woff2", DS_FONT_ARCHIVO)
}

async fn font_mono() -> Response {
    binary_asset("font/woff2", DS_FONT_MONO)
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
    Ok(Json(json!({
        "hostname": node.hostname,
        "device_kind": format!("{:?}", node.device_kind),
        "backend": format!("{:?}", node.backend),
        "fast_memory_gb": node.available_fast_memory_gb(),
        "system_ram_gb": node.memory.system_ram_gb,
        "fp16_tflops": node.effective_fp16_tflops()
    })))
}

async fn providers() -> Json<Value> {
    Json(json!({"providers": ProviderFactory::descriptors()}))
}

#[derive(Debug, Deserialize)]
struct ModelsRequest {
    provider: ProviderConfig,
}

async fn models(Json(request): Json<ModelsRequest>) -> Result<Json<Vec<ModelInfo>>, ApiError> {
    let provider = ProviderFactory::build(request.provider)?;
    Ok(Json(provider.list_models().await?))
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
    provider: ProviderConfig,
    messages: Vec<Message>,
    #[serde(default)]
    enable_tools: bool,
    #[serde(default = "default_max_steps")]
    max_steps: usize,
    #[serde(default)]
    sampling: SamplingParams,
}

fn default_max_steps() -> usize {
    8
}

async fn chat(
    State(state): State<ServerState>,
    Json(request): Json<ChatApiRequest>,
) -> Result<Json<HarnessReport>, ApiError> {
    let plugins = PluginManager::new(state.plugin_roots.as_ref().clone());
    let harness = Harness::new(request.provider, plugins)?;
    let report = harness
        .run(
            request.messages,
            HarnessOptions {
                enable_tools: request.enable_tools,
                max_steps: request.max_steps,
                sampling: request.sampling,
            },
        )
        .await?;
    Ok(Json(report))
}

struct ApiError(CondenserError);

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
