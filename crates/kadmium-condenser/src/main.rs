use clap::{Args, Parser, Subcommand};
use kadmium_condenser::dsv41_mcl4::{
    DeepSeekV41ArchSpec, Dsv41ArenaCachePlan, Dsv41Mcl4cPacketLayout, Mcl4CoeKnobs, Mcl4CoeMode,
    discover_local_dsv41_shards_and_checkpoints, mla_context_budget, probe_decision_candidates,
};
use kadmium_condenser::plugin::{PluginManager, default_plugin_roots, scaffold_plugin};
use kadmium_condenser::provider::ProviderFactory;
use kadmium_condenser::server::{self, ServerOptions};
use kadmium_condenser::types::{Message, ProviderConfig, ProviderKind, SamplingParams};
use kadmium_condenser::ui::{self, UiOptions};
use kadmium_condenser::{CondenserError, DEFAULT_HTTP_PORT, Harness, HarnessOptions, Result};
use kadmium_core::LiveHardwareProfiler;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser)]
#[command(
    name = "condenser",
    version,
    about = "Condenser — Kadmium Power Plant's modular inference harness (optimized for DeepSeek-V4.1 & MCL4-COE)",
    long_about = "Run local, hosted, MCL4-COE, DeepSeek-V4.1, OpenAI-compatible, Anthropic, or Kadmium inference through one Rust CLI and native UI. With no subcommand, Condenser opens its native app (or the browser fallback)."
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
    /// Inspect DeepSeek-V4.1 & MCL4-COE architecture, .mcl4c packet layout, and host VRAM/RAM placement plan.
    Dsv41Profile {
        /// Print full profile as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run a Typed-COE 9-Operator Decision + Deduction + Evaluation + Metacognition probe.
    ProbeDecision(ProbeDecisionArgs),
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
    /// Provider: kadmium, mcl4-coe (or dsv41), ollama, openai, or anthropic.
    #[arg(long, default_value = "mcl4-coe")]
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
    /// MCL4-COE / DeepSeek-V4.1 cognitive mode: uta, coe, dream_rl, or ncm.
    #[arg(long)]
    mode: Option<String>,
    /// Recurrent pondering steps per turn (default 2 for MCL4-COE / DeepSeek-V4.1).
    #[arg(long)]
    recurrence_steps: Option<usize>,
    /// Dense expert subset prefix size (`dense_subset_k`).
    #[arg(long)]
    dense_subset_k: Option<usize>,
    /// Tabu recurrent expert coverage penalty (`recurrent_coverage_penalty`, default 0.35).
    #[arg(long)]
    tabu_penalty: Option<f32>,
    /// Optional `#T$#` reasoning prefix for guided thought completion.
    #[arg(long)]
    thought_prefix: Option<String>,
    /// Optional `#A$#` answer prefix for constrained answer completion.
    #[arg(long)]
    answer_prefix: Option<String>,
    /// Lock the `#T$#` thought prefix and decode `#A$#` directly.
    #[arg(long)]
    lock_thought: bool,
    /// DeepSeek-V4.1 reasoning effort: none, low, medium, high, max.
    #[arg(long)]
    reasoning_effort: Option<String>,
    /// Print the complete harness report as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct ProbeDecisionArgs {
    /// Context passage (`#U$#`).
    #[arg(long)]
    context: String,
    /// Decision question (`#Q$#`).
    #[arg(long, default_value = "Select the best option.")]
    question: String,
    /// Candidate options (`--option "A" --option "B"`).
    #[arg(long = "option", required = true)]
    options: Vec<String>,
    /// Recurrent pondering steps.
    #[arg(long, default_value_t = 2)]
    recurrence_steps: usize,
    /// Print report as JSON.
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
                ..ServerOptions::default()
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
        Commands::Dsv41Profile { json } => run_dsv41_profile(json),
        Commands::ProbeDecision(args) => run_probe_decision(args),
        Commands::Plugin { command } => block_on(run_plugin(command, roots)),
    };
    if let Err(error) = result {
        eprintln!("condenser: {error}");
        std::process::exit(1);
    }
}

fn run_dsv41_profile(as_json: bool) -> Result<()> {
    let node = LiveHardwareProfiler::probe_local_host();
    let arch = DeepSeekV41ArchSpec::deepseek_v41_flash_nvfp4();
    let layout = Dsv41Mcl4cPacketLayout::new();
    let vram_gb = node
        .memory
        .fast_accelerator_memory_gb()
        .max(node.available_fast_memory_gb());
    let plan = Dsv41ArenaCachePlan::plan_for_host(vram_gb, node.memory.system_ram_gb);
    let dual_nvme = discover_local_dsv41_shards_and_checkpoints();
    let mla_8k = mla_context_budget(
        8192,
        arch.mla_sliding_window,
        arch.mla_compressed_kv_topk,
        arch.mla_compression_ratio,
    );
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "architecture": arch,
                "mcl4c_packet_layout": layout,
                "host_arena_cache_plan": plan,
                "dual_nvme_shard_inventory": dual_nvme,
                "mla_budget_8k": mla_8k,
            }))?
        );
    } else {
        println!(
            "DeepSeek-V4.1 & MCL4-COE Condenser Profile ({})",
            arch.model_id
        );
        println!(
            "  Backbone:         {} layers, dim={}, moe_inter={}, vocab={}",
            arch.n_layers, arch.dim, arch.moe_inter_dim, arch.vocab_size
        );
        println!(
            "  Hyper-Connect:    hc_mult={} ({} floats/tok), {}-iter 4x4 Sinkhorn-Knopp",
            arch.hc_mult,
            arch.hc_flattened_dim(),
            arch.hc_sinkhorn_iters
        );
        println!(
            "  MLA + YOCO:       {}H x {}d (1 KV head, {:.0}x KV compression), window={} + topk={} (YOCO keep={})",
            arch.mla_heads,
            arch.mla_head_dim,
            mla_8k.kv_compression_factor,
            arch.mla_sliding_window,
            arch.mla_compressed_kv_topk,
            arch.yoco_prefill_keep_tokens
        );
        println!(
            "  MoE Experts:      {} routed NVFP4 + {} shared FP8, top_k={} (Faster={}), SwiGLU limit={:.1}",
            arch.n_routed_experts,
            arch.n_shared_experts,
            arch.top_k,
            arch.faster_top_k,
            arch.swiglu_limit
        );
        println!(
            "  .mcl4c Packet:    {} B ({:.3} MiB, 4K-aligned), 8-slot arena={:.2} MiB, warm cache={} slots ({:.2} GiB)",
            layout.total_packet_bytes,
            (layout.total_packet_bytes as f64) / (1024.0 * 1024.0),
            plan.ping_pong_arena_mib,
            plan.warm_expert_cache_slots,
            plan.warm_expert_cache_gib
        );
        println!(
            "  NVMe Engram:      Layers {:?}, 24-prime XOR N-gram, {} B/token gather, 0 B VRAM ({:.1} GB disk)",
            arch.engram_layers, arch.engram_gather_bytes_per_token, plan.nvme_engram_disk_gb
        );
        println!(
            "  Host Fit Plan:    {} (top_k={}, CPU RAM cached experts={}, FreeToken q*={:.2})",
            plan.recommended_pack_variant,
            plan.recommended_top_k,
            plan.cpu_ram_cached_experts,
            plan.freetoken_q_star_gpu_expert_fraction
        );
        println!(
            "  Dual-NVMe Shards: {} ({}/20 even on {}, {}/20 odd on {}, {} MCL4DS41 checkpoints)",
            if dual_nvme.dual_controller_ready {
                "READY"
            } else {
                "PARTIAL"
            },
            dual_nvme.even_layers_detected,
            dual_nvme.even_drive_root,
            dual_nvme.odd_layers_detected,
            dual_nvme.odd_drive_root,
            dual_nvme.checkpoints.len()
        );
        if let Some(ref latest) = dual_nvme.latest_checkpoint {
            println!(
                "  Latest Checkpt:   {} (step={}, d*={:.2}★, pp={:.1}, total_pp={:.1}, val_bpb={:.4}, val_mcqa={:.2}%)",
                latest.checkpoint_id,
                latest.step,
                latest.star_difficulty.unwrap_or(0.0),
                latest.pp.unwrap_or(0.0),
                latest.total_pp.unwrap_or(0.0),
                latest.val_bpb.unwrap_or(0.0),
                latest.val_mcqa.unwrap_or(0.0)
            );
        }
    }
    Ok(())
}

fn run_probe_decision(args: ProbeDecisionArgs) -> Result<()> {
    let mut knobs = Mcl4CoeKnobs::deepseek_v41_default();
    knobs.recurrence_steps = args.recurrence_steps;
    let report = probe_decision_candidates(&args.context, &args.question, &args.options, &knobs)
        .map_err(CondenserError::Config)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "MCL4-COE Decision Probe Winner: [{}] {}",
            report.winner_index, report.winner_option
        );
        for c in &report.candidates {
            println!(
                "  {} [{}] prob={:.4} pearson={:.4} util={:.4} entailed={:.4} corr={:.4} — {}",
                if c.selected { "*" } else { " " },
                c.index,
                c.probability,
                c.pearson_similarity,
                c.utility,
                c.entailed,
                c.correctness,
                c.option
            );
        }
    }
    Ok(())
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

    let has_mcl4_flags = config.kind == ProviderKind::Mcl4Coe
        || args.mode.is_some()
        || args.recurrence_steps.is_some()
        || args.dense_subset_k.is_some()
        || args.tabu_penalty.is_some()
        || args.thought_prefix.is_some()
        || args.answer_prefix.is_some()
        || args.lock_thought
        || args.reasoning_effort.is_some();

    let mcl4_coe = if has_mcl4_flags {
        let mut knobs = Mcl4CoeKnobs::deepseek_v41_default();
        if let Some(ref m) = args.mode {
            knobs.mode = match m.trim().to_ascii_lowercase().as_str() {
                "coe" => Mcl4CoeMode::Coe,
                "dream_rl" | "dream-rl" | "dream" => Mcl4CoeMode::DreamRl,
                "ncm" => Mcl4CoeMode::Ncm,
                _ => Mcl4CoeMode::Uta,
            };
        }
        if let Some(steps) = args.recurrence_steps {
            knobs.recurrence_steps = steps.max(1);
        }
        if let Some(dk) = args.dense_subset_k {
            knobs.dense_subset_k = dk;
        }
        if let Some(tabu) = args.tabu_penalty {
            knobs.recurrent_coverage_penalty = tabu.max(0.0);
        }
        if let Some(k) = args.top_k {
            knobs.top_k = k as usize;
        }
        knobs.thought_prefix = args.thought_prefix.clone();
        knobs.answer_prefix = args.answer_prefix.clone();
        knobs.lock_thought = args.lock_thought;
        if let Some(ref effort) = args.reasoning_effort {
            knobs.reasoning_effort = effort.clone();
        }
        Some(knobs)
    } else {
        None
    };

    let harness = Harness::new(config, PluginManager::new(roots))?;
    let report = harness
        .run(
            messages,
            HarnessOptions {
                enable_tools: args.tools,
                max_steps: args.max_steps,
                max_calls_per_step: 16,
                sampling: SamplingParams {
                    temperature: args.temperature,
                    top_p: args.top_p,
                    top_k: args.top_k,
                    max_tokens: args.max_tokens,
                    seed: args.seed,
                    stop: Vec::new(),
                    mcl4_coe,
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
    if report.status != kadmium_condenser::RunStatus::Completed {
        eprintln!(
            "run ended with status {:?}: {}",
            report.status,
            report.error.as_deref().unwrap_or("no additional detail")
        );
    }
    if let Some(ref thought) = report.answer.reasoning {
        eprintln!("#T$# {thought}");
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
