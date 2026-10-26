use clap::{Args, Parser, Subcommand};
use kadmium_condenser::plugin::{PluginManager, default_plugin_roots, scaffold_plugin};
use kadmium_condenser::provider::ProviderFactory;
use kadmium_condenser::server::{self, ServerOptions};
use kadmium_condenser::types::{Message, ProviderConfig, ProviderKind, SamplingParams};
use kadmium_condenser::ui::{self, UiOptions};
use kadmium_condenser::{CondenserError, DEFAULT_HTTP_PORT, Harness, HarnessOptions, Result};
use serde_json::Value;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser)]
#[command(
    name = "condenser",
    version,
    about = "Condenser — Kadmium Power Plant's modular inference harness",
    long_about = "Run local, hosted, OpenAI-compatible, Anthropic, or Kadmium inference through one Rust CLI and native UI. With no subcommand, Condenser opens its native app (or the browser fallback)."
)]
struct Cli {
    /// Override plugin search roots. Defaults: ./plugins and ~/.condenser/plugins.
    #[arg(long = "plugin-dir", global = true, value_name = "DIR")]
    plugin_dirs: Vec<PathBuf>,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Open the Condenser native UI (default command).
    Ui(UiArgs),
    /// Serve the same UI and JSON API on loopback without a native window.
    Serve(ServeArgs),
    /// Run one inference turn, including tool calls when explicitly enabled.
    Chat(ChatArgs),
    /// List models exposed by a provider.
    Models(ProviderArgs),
    /// Describe Condenser's built-in provider adapters.
    Providers,
    /// Discover, invoke, or scaffold process-isolated plugins.
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
}

#[derive(Debug, Clone, Args, Default)]
struct UiArgs {
    /// Loopback port. Defaults to an OS-selected available port.
    #[arg(short, long)]
    port: Option<u16>,
    /// Use the system browser even when a native window is available.
    #[arg(long)]
    browser: bool,
}

#[derive(Debug, Clone, Args)]
struct ServeArgs {
    /// Loopback port for the Condenser UI and API.
    #[arg(short, long, default_value_t = DEFAULT_HTTP_PORT)]
    port: u16,
    /// Open the local UI in the default browser after binding.
    #[arg(long)]
    open: bool,
}

#[derive(Debug, Clone, Args)]
struct ProviderArgs {
    /// Provider: kadmium, ollama, openai, or anthropic.
    #[arg(long, default_value = "ollama")]
    provider: String,
    /// Provider model identifier. Uses the provider default when present.
    #[arg(short, long)]
    model: Option<String>,
    /// Override the provider base URL (with or without a trailing /v1 where applicable).
    #[arg(long)]
    base_url: Option<String>,
    /// API key for this process only. Prefer --api-key-env to avoid process-list exposure.
    #[arg(long, conflicts_with = "api_key_env")]
    api_key: Option<String>,
    /// Environment variable containing the API key.
    #[arg(long, value_name = "NAME")]
    api_key_env: Option<String>,
    /// End-to-end provider request timeout in seconds.
    #[arg(long, default_value_t = 300)]
    timeout_secs: u64,
}

impl ProviderArgs {
    fn resolve(&self) -> Result<ProviderConfig> {
        let kind = ProviderKind::from_str(&self.provider)?;
        let mut config = ProviderConfig::new(kind);
        if let Some(model) = &self.model {
            config.model = model.clone();
        }
        if let Some(base_url) = &self.base_url {
            config.base_url = Some(base_url.clone());
        }
        config.timeout_secs = self.timeout_secs;
        config.api_key = self
            .api_key
            .clone()
            .or_else(|| {
                self.api_key_env
                    .as_ref()
                    .and_then(|name| std::env::var(name).ok())
            })
            .or_else(|| std::env::var(kind.default_api_key_env()).ok());
        Ok(config)
    }
}

#[derive(Debug, Clone, Args)]
struct ChatArgs {
    /// User message to send.
    message: String,
    #[command(flatten)]
    provider: ProviderArgs,
    /// Optional system prompt prepended to this turn.
    #[arg(long)]
    system: Option<String>,
    /// Allow the model to call tools from discovered plugins.
    #[arg(long)]
    tools: bool,
    /// Maximum model/tool steps before failing explicitly.
    #[arg(long, default_value_t = 8)]
    max_steps: usize,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<u32>,
    #[arg(long)]
    max_tokens: Option<u32>,
    #[arg(long)]
    seed: Option<u64>,
    /// Print the complete harness report as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum PluginCommand {
    /// List manifests and diagnostics without executing plugin code.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Explicitly invoke one declared plugin tool.
    Invoke {
        plugin_id: String,
        tool: String,
        /// JSON object passed as the tool arguments.
        #[arg(long, default_value = "{}")]
        arguments: String,
    },
    /// Generate a non-destructive standalone Rust plugin project.
    Scaffold {
        name: String,
        /// Parent directory. Defaults to ./plugins in the current workspace.
        #[arg(long)]
        parent: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();
    let roots = if cli.plugin_dirs.is_empty() {
        default_plugin_roots()
    } else {
        cli.plugin_dirs
    };
    let result = match cli.command.unwrap_or(Commands::Ui(UiArgs::default())) {
        Commands::Ui(args) => ui::run(UiOptions {
            port: args.port,
            browser: args.browser,
            plugin_roots: roots,
        }),
        Commands::Serve(args) => block_on(async move {
            let (address, future) = server::bind(ServerOptions {
                port: args.port,
                plugin_roots: roots,
            })
            .await?;
            println!("Condenser is ready at http://{address}/");
            if args.open {
                ui::open_in_browser(&format!("http://{address}/"));
            }
            future.await
        }),
        Commands::Chat(args) => block_on(run_chat(args, roots)),
        Commands::Models(args) => block_on(async move {
            let provider = ProviderFactory::build(args.resolve()?)?;
            let models = provider.list_models().await?;
            for model in models {
                println!("{}", model.id);
            }
            Ok(())
        }),
        Commands::Providers => {
            for descriptor in ProviderFactory::descriptors() {
                println!("{} ({})", descriptor.label, descriptor.kind);
                println!("  URL:     {}", descriptor.default_base_url);
                println!("  Key env: {}", descriptor.default_api_key_env);
                println!("  {}", descriptor.note);
            }
            Ok(())
        }
        Commands::Plugin { command } => block_on(run_plugin(command, roots)),
    };
    if let Err(error) = result {
        eprintln!("condenser: {error}");
        std::process::exit(1);
    }
}

fn block_on<F>(future: F) -> Result<()>
where
    F: std::future::Future<Output = Result<()>>,
{
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| CondenserError::Config(format!("could not start runtime: {error}")))?
        .block_on(future)
}

async fn run_chat(args: ChatArgs, roots: Vec<PathBuf>) -> Result<()> {
    let config = args.provider.resolve()?;
    let mut messages = Vec::new();
    if let Some(system) = args.system.filter(|system| !system.trim().is_empty()) {
        messages.push(Message::system(system));
    }
    messages.push(Message::user(args.message));
    let harness = Harness::new(config, PluginManager::new(roots))?;
    let report = harness
        .run(
            messages,
            HarnessOptions {
                enable_tools: args.tools,
                max_steps: args.max_steps,
                sampling: SamplingParams {
                    temperature: args.temperature,
                    top_p: args.top_p,
                    top_k: args.top_k,
                    max_tokens: args.max_tokens,
                    seed: args.seed,
                    stop: Vec::new(),
                },
            },
        )
        .await?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    for step in &report.steps {
        for execution in &step.tool_executions {
            eprintln!(
                "tool {} {} ({:.1} ms)",
                execution.tool_name,
                if execution.result.is_error {
                    "failed"
                } else {
                    "completed"
                },
                execution.duration_ms
            );
        }
    }
    for diagnostic in &report.diagnostics {
        eprintln!("plugin diagnostic: {diagnostic}");
    }
    println!("{}", report.answer.content);
    if report.usage.total_tokens > 0 {
        eprintln!(
            "usage: {} input + {} output = {} tokens",
            report.usage.input_tokens, report.usage.output_tokens, report.usage.total_tokens
        );
    }
    Ok(())
}

async fn run_plugin(command: PluginCommand, roots: Vec<PathBuf>) -> Result<()> {
    let manager = PluginManager::new(roots);
    match command {
        PluginCommand::List { json } => {
            let inventory = manager.discover()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&inventory)?);
            } else {
                if inventory.plugins.is_empty() {
                    println!("No Condenser plugins discovered.");
                }
                for plugin in inventory.plugins {
                    println!(
                        "{} {} ({}) — {}",
                        plugin.manifest.id,
                        plugin.manifest.version,
                        plugin.manifest.name,
                        plugin.manifest.description
                    );
                    println!("  root: {}", plugin.root.display());
                    for tool in plugin.manifest.tools {
                        println!(
                            "  tool {}__{}: {}",
                            plugin.manifest.id, tool.name, tool.description
                        );
                    }
                }
                for diagnostic in inventory.diagnostics {
                    eprintln!("{}: {}", diagnostic.path.display(), diagnostic.message);
                }
            }
        }
        PluginCommand::Invoke {
            plugin_id,
            tool,
            arguments,
        } => {
            let arguments: Value = serde_json::from_str(&arguments).map_err(|error| {
                CondenserError::Config(format!("--arguments must be valid JSON: {error}"))
            })?;
            let result = manager.invoke(&plugin_id, &tool, arguments).await?;
            println!("{}", result.content);
            if result.is_error {
                return Err(CondenserError::Plugin(format!(
                    "{plugin_id}.{tool} returned an error result"
                )));
            }
        }
        PluginCommand::Scaffold { name, parent } => {
            let root = scaffold_plugin(&name, parent.as_deref())?;
            println!("Created Condenser plugin at {}", root.display());
            println!(
                "Try it: condenser plugin invoke {} echo --arguments '{{\"text\":\"hello\"}}'",
                root.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }
    Ok(())
}
