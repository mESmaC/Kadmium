//! Manifest-discovered, process-isolated Condenser plugins.
//!
//! Discovery only reads manifests; it never executes plugin code. A plugin is
//! launched only for an explicit invocation or when the caller has opted into
//! model tool use. The wire protocol is JSON-RPC 2.0 with one message per line,
//! so plugins can be written in any language without a native Rust ABI.

use crate::types::{ToolDefinition, ToolResult};
use crate::{CondenserError, PLUGIN_PROTOCOL_VERSION, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

pub const PLUGIN_MANIFEST_FILE: &str = "condenser-plugin.json";
const DEFAULT_PLUGIN_TIMEOUT: Duration = Duration::from_secs(30);
const PLUGIN_SHUTDOWN_GRACE: Duration = Duration::from_millis(250);
const MAX_PLUGIN_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub protocol_version: u32,
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
}

impl PluginManifest {
    pub fn validate(&self) -> Result<()> {
        if self.protocol_version != PLUGIN_PROTOCOL_VERSION {
            return Err(CondenserError::Plugin(format!(
                "plugin `{}` speaks protocol {}, but this Condenser speaks {}",
                self.id, self.protocol_version, PLUGIN_PROTOCOL_VERSION
            )));
        }
        validate_identifier(&self.id, "plugin id")?;
        if self.name.trim().is_empty() || self.version.trim().is_empty() {
            return Err(CondenserError::Plugin(format!(
                "plugin `{}` needs a display name and version",
                self.id
            )));
        }
        if self.command.trim().is_empty() {
            return Err(CondenserError::Plugin(format!(
                "plugin `{}` has an empty command",
                self.id
            )));
        }
        let mut names = BTreeSet::new();
        for tool in &self.tools {
            validate_identifier(&tool.name, "tool name")?;
            if !names.insert(tool.name.as_str()) {
                return Err(CondenserError::Plugin(format!(
                    "plugin `{}` declares tool `{}` more than once",
                    self.id, tool.name
                )));
            }
            if !tool.input_schema.is_object() {
                return Err(CondenserError::Plugin(format!(
                    "tool `{}/{}` needs an object JSON Schema",
                    self.id, tool.name
                )));
            }
            jsonschema::validator_for(&tool.input_schema).map_err(|error| {
                CondenserError::Plugin(format!(
                    "tool `{}/{}` has an invalid JSON Schema: {error}",
                    self.id, tool.name
                ))
            })?;
        }
        Ok(())
    }
}

fn validate_identifier(value: &str, kind: &str) -> Result<()> {
    let valid = !value.is_empty()
        && value.len() <= 64
        && value
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric())
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'));
    if valid {
        Ok(())
    } else {
        Err(CondenserError::Plugin(format!(
            "invalid {kind} `{value}`; use 1-64 ASCII letters, digits, hyphens, or underscores"
        )))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Plugin {
    pub manifest: PluginManifest,
    pub root: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct PluginDiagnostic {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PluginInventory {
    pub plugins: Vec<Plugin>,
    pub diagnostics: Vec<PluginDiagnostic>,
}

#[derive(Debug, Clone)]
pub struct RegisteredTool {
    pub plugin_id: String,
    pub plugin_name: String,
    pub qualified_name: String,
    pub definition: ToolDefinition,
    plugin: Plugin,
}

impl RegisteredTool {
    pub fn new(plugin: Plugin, definition: ToolDefinition) -> Self {
        Self {
            plugin_id: plugin.manifest.id.clone(),
            plugin_name: plugin.manifest.name.clone(),
            qualified_name: format!("{}__{}", plugin.manifest.id, definition.name),
            definition,
            plugin,
        }
    }

    pub fn unqualified_name(&self) -> &str {
        &self.definition.name
    }
}

#[derive(Debug, Clone)]
pub struct PluginManager {
    roots: Vec<PathBuf>,
    timeout: Duration,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new(default_plugin_roots())
    }
}

impl PluginManager {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self {
            roots,
            timeout: DEFAULT_PLUGIN_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    pub fn discover(&self) -> Result<PluginInventory> {
        let mut inventory = PluginInventory::default();
        let mut seen = BTreeMap::<String, PathBuf>::new();
        for root in &self.roots {
            if !root.exists() {
                continue;
            }
            let entries = std::fs::read_dir(root).map_err(|error| {
                CondenserError::Plugin(format!("could not read {}: {error}", root.display()))
            })?;
            let mut paths = entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .collect::<Vec<_>>();
            paths.sort();
            for path in paths {
                let manifest_path = if path.is_dir() {
                    path.join(PLUGIN_MANIFEST_FILE)
                } else if path
                    .file_name()
                    .is_some_and(|name| name == PLUGIN_MANIFEST_FILE)
                {
                    path.clone()
                } else {
                    continue;
                };
                if !manifest_path.is_file() {
                    continue;
                }
                match load_plugin(&manifest_path) {
                    Ok(plugin) => {
                        if let Some(previous) = seen.get(&plugin.manifest.id) {
                            inventory.diagnostics.push(PluginDiagnostic {
                                path: manifest_path,
                                message: format!(
                                    "duplicate plugin id `{}`; first declaration at {} wins",
                                    plugin.manifest.id,
                                    previous.display()
                                ),
                            });
                        } else {
                            seen.insert(plugin.manifest.id.clone(), manifest_path);
                            inventory.plugins.push(plugin);
                        }
                    }
                    Err(error) => inventory.diagnostics.push(PluginDiagnostic {
                        path: manifest_path,
                        message: error.to_string(),
                    }),
                }
            }
        }
        Ok(inventory)
    }

    pub fn registered_tools(&self) -> Result<(Vec<RegisteredTool>, Vec<PluginDiagnostic>)> {
        let inventory = self.discover()?;
        let mut diagnostics = inventory.diagnostics;
        let mut tools = Vec::new();
        let mut names = BTreeMap::<String, String>::new();
        for plugin in inventory.plugins {
            for definition in &plugin.manifest.tools {
                let qualified_name = format!("{}__{}", plugin.manifest.id, definition.name);
                if let Some(previous) = names.get(&qualified_name) {
                    diagnostics.push(PluginDiagnostic {
                        path: plugin.root.join(PLUGIN_MANIFEST_FILE),
                        message: format!(
                            "tool name collision `{qualified_name}` with plugin `{previous}`; skipping"
                        ),
                    });
                    continue;
                }
                names.insert(qualified_name.clone(), plugin.manifest.id.clone());
                tools.push(RegisteredTool::new(plugin.clone(), definition.clone()));
            }
        }
        Ok((tools, diagnostics))
    }

    pub async fn invoke(
        &self,
        plugin_id: &str,
        tool_name: &str,
        arguments: Value,
    ) -> Result<ToolResult> {
        let inventory = self.discover()?;
        let plugin = inventory
            .plugins
            .into_iter()
            .find(|plugin| plugin.manifest.id == plugin_id)
            .ok_or_else(|| CondenserError::Plugin(format!("unknown plugin `{plugin_id}`")))?;
        if !plugin
            .manifest
            .tools
            .iter()
            .any(|tool| tool.name == tool_name)
        {
            return Err(CondenserError::Plugin(format!(
                "plugin `{plugin_id}` does not declare tool `{tool_name}`"
            )));
        }
        self.invoke_process(&plugin, tool_name, arguments).await
    }

    pub async fn invoke_registered(
        &self,
        tool: &RegisteredTool,
        arguments: Value,
    ) -> Result<ToolResult> {
        self.invoke_process(&tool.plugin, tool.unqualified_name(), arguments)
            .await
    }

    async fn invoke_process(
        &self,
        plugin: &Plugin,
        tool_name: &str,
        arguments: Value,
    ) -> Result<ToolResult> {
        let manifest = &plugin.manifest;
        let (command, args) = resolve_command_and_args(&plugin.root, manifest);
        let mut child = Command::new(&command)
            .args(args)
            .current_dir(&plugin.root)
            .env("CONDENSER_PLUGIN_DIR", &plugin.root)
            .env("CONDENSER_PLUGIN_ID", &manifest.id)
            .env(
                "CONDENSER_PROTOCOL_VERSION",
                PLUGIN_PROTOCOL_VERSION.to_string(),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Protocol stdout is captured; diagnostics belong on stderr and
            // remain visible without being mistaken for JSON-RPC messages.
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                CondenserError::Plugin(format!(
                    "could not launch `{}` with `{command}`: {error}",
                    manifest.id
                ))
            })?;

        let mut stdin = child.stdin.take().expect("piped plugin stdin");
        let mut stdout = child.stdout.take().expect("piped plugin stdout");
        let operation = async {
            write_rpc(
                &mut stdin,
                &json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments}
                }),
            )
            .await?;
            let response = read_rpc(&mut stdout, 1).await?;
            parse_tool_response(response, &manifest.id, tool_name)
        };

        let result = match tokio::time::timeout(self.timeout, operation).await {
            Ok(result) => result,
            Err(_) => Err(CondenserError::Plugin(format!(
                "{}.{} exceeded the {:?} plugin timeout",
                manifest.id, tool_name, self.timeout
            ))),
        };

        match result {
            Ok(tool_result) => {
                // The invocation is already complete. Shutdown gets its own short
                // grace period and can never turn a valid result into a timeout.
                let graceful = tokio::time::timeout(PLUGIN_SHUTDOWN_GRACE, async {
                    let _ = write_rpc(
                        &mut stdin,
                        &json!({"jsonrpc": "2.0", "id": 2, "method": "shutdown"}),
                    )
                    .await;
                    drop(stdin);
                    let _ = child.wait().await;
                })
                .await;
                if graceful.is_err() {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                }
                Ok(tool_result)
            }
            Err(error) => {
                drop(stdin);
                let _ = child.kill().await;
                let _ = child.wait().await;
                Err(error)
            }
        }
    }
}

fn parse_tool_response(response: Value, plugin_id: &str, tool_name: &str) -> Result<ToolResult> {
    let object = response
        .as_object()
        .ok_or_else(|| CondenserError::Plugin("plugin response is not a JSON-RPC object".into()))?;
    let has_result = object.contains_key("result");
    let has_error = object.contains_key("error");
    if has_result && has_error {
        return Err(CondenserError::Plugin(
            "plugin response contains both `result` and `error`".into(),
        ));
    }
    if has_error {
        let error = &response["error"];
        return Err(CondenserError::Plugin(format!(
            "{plugin_id}.{tool_name} failed: {}",
            error["message"]
                .as_str()
                .unwrap_or_else(|| error.as_str().unwrap_or("unknown error"))
        )));
    }
    if !has_result || response["result"].is_null() {
        return Err(CondenserError::Plugin(
            "plugin response is missing a non-null `result`".into(),
        ));
    }
    parse_tool_result(response["result"].clone())
}

fn parse_tool_result(value: Value) -> Result<ToolResult> {
    if let Some(text) = value.as_str() {
        return Ok(ToolResult::success(text));
    }
    if !value.is_object() {
        return Err(CondenserError::Plugin(
            "plugin tool result must be a string or an object".into(),
        ));
    }
    serde_json::from_value::<ToolResult>(value).map_err(|error| {
        CondenserError::Plugin(format!("plugin returned a malformed tool result: {error}"))
    })
}

fn substitute_plugin_dir(value: &str, root: &Path) -> String {
    value.replace("{plugin_dir}", &root.to_string_lossy())
}

fn resolve_command(root: &Path, command: &str) -> String {
    let command = substitute_plugin_dir(command, root);
    let path = Path::new(&command);
    if path.is_relative() && (command.contains('/') || command.contains('\\')) {
        root.join(path).to_string_lossy().into_owned()
    } else {
        command
    }
}

/// Fast-path resolver for Rust plugins configured with `command = "cargo"` and `args = ["run", ...]`.
/// When a pre-compiled binary exists in `{plugin_dir}/target/release/` or `{plugin_dir}/target/debug/`
/// and is at least as new as `src/main.rs` and `Cargo.toml`, invokes the compiled binary directly
/// without spawning `cargo` on every tool call.
fn resolve_command_and_args(root: &Path, manifest: &PluginManifest) -> (String, Vec<String>) {
    let default_cmd = resolve_command(root, &manifest.command);
    let default_args: Vec<String> = manifest
        .args
        .iter()
        .map(|arg| substitute_plugin_dir(arg, root))
        .collect();

    if manifest.command.trim() == "cargo"
        && default_args.first().is_some_and(|a| a == "run")
    {
        let bin_stem = format!("condenser-plugin-{}", manifest.id);
        let exe_name = if cfg!(windows) {
            format!("{bin_stem}.exe")
        } else {
            bin_stem
        };
        let src_mtime = root
            .join("src")
            .join("main.rs")
            .metadata()
            .and_then(|m| m.modified())
            .ok();
        let toml_mtime = root
            .join("Cargo.toml")
            .metadata()
            .and_then(|m| m.modified())
            .ok();

        for profile in ["release", "debug"] {
            let candidate = root.join("target").join(profile).join(&exe_name);
            if let Ok(meta) = candidate.metadata() {
                if meta.is_file() {
                    let bin_fresh = match (meta.modified().ok(), src_mtime, toml_mtime) {
                        (Some(bm), Some(sm), Some(tm)) => bm >= sm && bm >= tm,
                        (Some(bm), Some(sm), None) => bm >= sm,
                        _ => true,
                    };
                    if bin_fresh {
                        let passthrough_args = default_args
                            .iter()
                            .position(|a| a == "--")
                            .map(|pos| default_args[pos + 1..].to_vec())
                            .unwrap_or_default();
                        return (candidate.to_string_lossy().into_owned(), passthrough_args);
                    }
                }
            }
        }
    }

    (default_cmd, default_args)
}

async fn write_rpc<W>(writer: &mut W, value: &Value) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_rpc<R>(reader: &mut R, expected_id: u64) -> Result<Value>
where
    R: AsyncRead + Unpin,
{
    let mut frame = Vec::new();
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            return Err(CondenserError::Plugin(
                "plugin closed stdout before answering".into(),
            ));
        }
        let mut start = 0;
        while start < count {
            if let Some(relative_end) = chunk[start..count].iter().position(|byte| *byte == b'\n') {
                let end = start + relative_end;
                append_response_bytes(&mut frame, &chunk[start..end])?;
                if !frame.iter().all(u8::is_ascii_whitespace) {
                    let value: Value = serde_json::from_slice(&frame).map_err(|error| {
                        CondenserError::Plugin(format!("plugin returned non-JSON stdout: {error}"))
                    })?;
                    if value["jsonrpc"] != "2.0" {
                        return Err(CondenserError::Plugin(
                            "plugin response is not JSON-RPC 2.0".into(),
                        ));
                    }
                    if value["id"] == expected_id {
                        return Ok(value);
                    }
                }
                frame.clear();
                start = end + 1;
            } else {
                append_response_bytes(&mut frame, &chunk[start..count])?;
                break;
            }
        }
    }
}

fn append_response_bytes(frame: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_PLUGIN_RESPONSE_BYTES.saturating_sub(frame.len()) {
        return Err(CondenserError::Plugin(format!(
            "plugin response exceeds {MAX_PLUGIN_RESPONSE_BYTES} bytes"
        )));
    }
    frame.extend_from_slice(bytes);
    Ok(())
}

fn load_plugin(manifest_path: &Path) -> Result<Plugin> {
    let raw = std::fs::read_to_string(manifest_path)?;
    let manifest: PluginManifest = serde_json::from_str(&raw).map_err(|error| {
        CondenserError::Plugin(format!(
            "could not parse {}: {error}",
            manifest_path.display()
        ))
    })?;
    manifest.validate()?;
    let root = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()?;
    Ok(Plugin { manifest, root })
}

pub fn condenser_home() -> PathBuf {
    if let Some(path) = std::env::var_os("CONDENSER_HOME") {
        return PathBuf::from(path);
    }
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        return PathBuf::from(home).join(".condenser");
    }
    PathBuf::from(".condenser")
}

pub fn default_plugin_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(current) = std::env::current_dir() {
        roots.push(current.join("plugins"));
    }
    roots.push(condenser_home().join("plugins"));
    roots
}

/// Creates a standalone Rust echo plugin that an agent or user can modify.
/// Existing non-empty directories are never overwritten.
pub fn scaffold_plugin(name: &str, parent: Option<&Path>) -> Result<PathBuf> {
    let slug = slugify(name);
    if slug.is_empty() {
        return Err(CondenserError::Config(
            "plugin name needs at least one ASCII letter or digit".into(),
        ));
    }
    validate_identifier(&slug, "plugin id")?;
    let parent = match parent {
        Some(path) => path.to_path_buf(),
        None => std::env::current_dir()?.join("plugins"),
    };
    let root = parent.join(&slug);
    if root.exists() && std::fs::read_dir(&root)?.next().is_some() {
        return Err(CondenserError::Plugin(format!(
            "refusing to overwrite non-empty plugin directory {}",
            root.display()
        )));
    }
    std::fs::create_dir_all(root.join("src"))?;
    let manifest = PluginManifest {
        protocol_version: PLUGIN_PROTOCOL_VERSION,
        id: slug.clone(),
        name: name.trim().to_string(),
        version: "0.1.0".into(),
        description: "A Condenser process plugin generated by `condenser plugin scaffold`.".into(),
        command: "cargo".into(),
        args: vec![
            "run".into(),
            "--quiet".into(),
            "--manifest-path".into(),
            "{plugin_dir}/Cargo.toml".into(),
            "--".into(),
        ],
        tools: vec![ToolDefinition {
            name: "echo".into(),
            description: "Return the supplied text unchanged.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to return"}},
                "required": ["text"],
                "additionalProperties": false
            }),
        }],
    };
    std::fs::write(
        root.join(PLUGIN_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    std::fs::write(
        root.join("Cargo.toml"),
        format!(
            "[package]\nname = \"condenser-plugin-{slug}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\nserde = {{ version = \"1\", features = [\"derive\"] }}\nserde_json = \"1\"\n\n# Keep generated plugins standalone even inside the Kadmium repository.\n[workspace]\n"
        ),
    )?;
    std::fs::write(root.join("src/main.rs"), SCAFFOLD_SOURCE)?;
    Ok(root)
}

fn slugify(name: &str) -> String {
    let mut slug = String::new();
    let mut separator = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            separator = false;
        } else if !slug.is_empty() && !separator {
            slug.push('-');
            separator = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    slug
}

const SCAFFOLD_SOURCE: &str = r#"use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
struct Request {
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = match line { Ok(line) => line, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => {
                eprintln!("invalid request: {error}");
                continue;
            }
        };
        let shutdown = request.method == "shutdown";
        let response = match request.method.as_str() {
            "ping" => json!({"jsonrpc": "2.0", "id": request.id, "result": {"content": "pong", "is_error": false}}),
            "tools/call" => {
                let name = request.params["name"].as_str().unwrap_or_default();
                let arguments = &request.params["arguments"];
                let result = match (name, arguments["text"].as_str()) {
                    ("echo", Some(text)) => json!({"content": text, "is_error": false}),
                    ("echo", None) => json!({"content": "missing string argument `text`", "is_error": true}),
                    _ => json!({"content": format!("unknown tool `{name}`"), "is_error": true}),
                };
                json!({"jsonrpc": "2.0", "id": request.id, "result": result})
            }
            "shutdown" => json!({"jsonrpc": "2.0", "id": request.id, "result": {"content": "stopping", "is_error": false}}),
            _ => json!({"jsonrpc": "2.0", "id": request.id, "error": {"code": -32601, "message": "method not found"}}),
        };
        if writeln!(stdout, "{}", response).and_then(|_| stdout.flush()).is_err() { break; }
        if shutdown { break; }
    }
}
"#;

/// Small server used by Rust plugins, including Condenser's example echo
/// plugin. Larger plugins can implement the same line protocol directly.
pub fn run_stdio_plugin<F>(plugin_id: &str, handler: F) -> std::io::Result<()>
where
    F: Fn(&str, Value) -> Result<ToolResult>,
{
    use std::io::{BufRead, Write};

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => {
                eprintln!("{plugin_id}: invalid JSON input: {error}");
                continue;
            }
        };
        let id = request["id"].clone();
        let method = request["method"].as_str().unwrap_or_default();
        let shutdown = method == "shutdown";
        let response = match method {
            "ping" => json!({"jsonrpc": "2.0", "id": id, "result": ToolResult::success("pong")}),
            "tools/call" => {
                let name = request["params"]["name"].as_str().unwrap_or_default();
                let arguments = request["params"]["arguments"].clone();
                match handler(name, arguments) {
                    Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                    Err(error) => {
                        json!({"jsonrpc": "2.0", "id": id, "result": ToolResult::failure(error.to_string())})
                    }
                }
            }
            "shutdown" => {
                json!({"jsonrpc": "2.0", "id": id, "result": ToolResult::success("stopping")})
            }
            _ => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "method not found"}})
            }
        };
        writeln!(stdout, "{}", response)?;
        stdout.flush()?;
        if shutdown {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> PluginManifest {
        PluginManifest {
            protocol_version: PLUGIN_PROTOCOL_VERSION,
            id: "example".into(),
            name: "Example".into(),
            version: "0.1.0".into(),
            description: "Test".into(),
            command: "example".into(),
            args: Vec::new(),
            tools: vec![ToolDefinition {
                name: "echo".into(),
                description: "Echo".into(),
                input_schema: json!({"type": "object"}),
            }],
        }
    }

    #[test]
    fn manifests_validate_protocol_and_tool_schema() {
        assert!(manifest().validate().is_ok());
        let mut wrong = manifest();
        wrong.protocol_version += 1;
        assert!(wrong.validate().is_err());
        let mut bad_schema = manifest();
        bad_schema.tools[0].input_schema = json!(true);
        assert!(bad_schema.validate().is_err());
        let mut invalid_schema = manifest();
        invalid_schema.tools[0].input_schema = json!({"type": 42});
        assert!(invalid_schema.validate().is_err());
    }

    #[test]
    fn scaffold_is_non_destructive_and_loadable() {
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("condenser-scaffold-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let plugin = scaffold_plugin("My Example", Some(&root)).unwrap();
        let loaded = load_plugin(&plugin.join(PLUGIN_MANIFEST_FILE)).unwrap();
        assert_eq!(loaded.manifest.id, "my-example");
        assert!(plugin.join("src/main.rs").is_file());
        assert!(scaffold_plugin("My Example", Some(&root)).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn plugin_dir_substitution_is_relative_to_manifest() {
        assert_eq!(
            substitute_plugin_dir("{plugin_dir}/plugin.exe", Path::new("/tmp/example")),
            "/tmp/example/plugin.exe"
        );
    }

    #[test]
    fn slugs_are_stable_and_non_destructive() {
        assert_eq!(slugify("  My Useful Plugin  "), "my-useful-plugin");
        assert_eq!(slugify("---"), "");
    }

    #[test]
    fn rpc_responses_require_exactly_one_outcome() {
        for response in [
            json!({"jsonrpc": "2.0", "id": 1}),
            json!({"jsonrpc": "2.0", "id": 1, "result": null}),
            json!({"jsonrpc": "2.0", "id": 1, "result": "ok", "error": null}),
        ] {
            assert!(parse_tool_response(response, "example", "echo").is_err());
        }
        let error = parse_tool_response(
            json!({"jsonrpc": "2.0", "id": 1, "error": {"message": "failed"}}),
            "example",
            "echo",
        )
        .unwrap_err();
        assert!(error.to_string().contains("example.echo failed: failed"));
    }

    #[test]
    fn tool_results_accept_only_the_protocol_shapes() {
        let text = parse_tool_result(json!("plain text")).unwrap();
        assert_eq!(text.content, "plain text");
        assert!(!text.is_error);

        let object = parse_tool_result(json!({"content": "failed", "is_error": true})).unwrap();
        assert_eq!(object.content, "failed");
        assert!(object.is_error);
        assert!(
            !parse_tool_result(json!({"content": "default"}))
                .unwrap()
                .is_error
        );

        for malformed in [
            json!(null),
            json!(42),
            json!(["content"]),
            json!({}),
            json!({"content": 42}),
            json!({"content": "bad flag", "is_error": "no"}),
        ] {
            assert!(parse_tool_result(malformed).is_err());
        }
    }
}
