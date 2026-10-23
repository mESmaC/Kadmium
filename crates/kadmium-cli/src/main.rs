use clap::{Parser, Subcommand};
use kadmium_core::{
    HardwareNode, LiveHardwareProfiler, ModdedSwitchController, ModelSpec,
    PhoneFarmBoxController, PhoneFarmBoxPreset, PhoneFarmTopologyMode, PhoneFarmTransportMode,
    PoliteState, ReTerminalCm4LanBridge, SwitchRuntimeEnvironment, SwitchSiliconRevision,
    WorkloadKind, sample_cloud_burst_gpu_node, sample_node_pool, sample_personal_cluster,
};
use kadmium_data::{Fts3TransferScheduler, LocalModelDiskImporter};
use kadmium_gateway::{CoordinatorOptions, run_coordinator};
use kadmium_network::DEFAULT_WORKER_TCP_PORT;
use kadmium_rl::{
    DisaggregatedRlPlan, DiscoveryTree, DreamRsiEvolutionReport, RlEnvironmentSuiteSpec,
    RlTopologyPlanner,
};
use kadmium_runtime::{
    DistributedExl3ConversionEngine, TrainingJobTelemetry, sample_boinc_workunits,
};
use kadmium_sharding::{
    ClusterShardPlan, DataflowStreamPlanner, DistributedPagedKvPlanner,
    MinMaxStageLatencyOptimizer, ResponsibilityShiftPlan, ShardPlanner, ShiftTriggerReason,
    SubClusterPartitioner,
};

mod grid_ops;
mod ops;
use grid_ops::*;
use ops::*;

#[derive(Parser)]
#[command(
    name = "kadmium",
    version,
    about = "Kadmium (Cd-48) — Consolidated, Aggregate & Distributed AI/ML Compute Fabric in Pure Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a full simulation of asymmetric model sharding, training, RL, Dream-RSI, FTS3 staging, and live responsibility shifting
    Simulate,
    /// Run a Dream-RSI (Recursive Self-Improvement through Evolving Worlds) offline replay simulation cycle
    Dream,
    /// Section a global compute pool (e.g., 100 nodes) into isolated Sub-Compute Environments (e.g., 5 clusters of 20 nodes)
    Slice {
        /// Total nodes in the global compute pool
        #[arg(short, long, default_value_t = 100)]
        pool_size: usize,
        /// Number of isolated sub-compute clusters to carve out
        #[arg(short, long, default_value_t = 5)]
        slices: usize,
    },
    /// Compare Google TPU / Compute-in-Memory inspired Dynamic Token Streaming vs. Dynamic Weight Streaming
    Dataflow {
        /// Model preset: deepseek-v3-671b-moe | qwen3-235b-moe | llama-3-70b-lora
        #[arg(short, long, default_value = "deepseek-v3-671b-moe")]
        model: String,
        /// Active batch size in tokens
        #[arg(short, long, default_value_t = 512)]
        batch_tokens: u32,
    },
    /// Simulate Kadmium Cloud's Multi-Datacenter Spot Arbitrage & Bidirectional Local <-> Cloud Downtime Protection Shield
    CloudShield,
    /// Probe the live local host hardware (CPU, GPU, RAM, NVMe bandwidth, q* ratio) and benchmark 64KB page-aligned .ktb packing
    Probe,
    /// Compute and inspect a custom asymmetric shard plan for a model and workload
    Plan {
        /// Model preset: deepseek-v3-671b-moe | qwen3-235b-moe | llama-3-70b-lora
        #[arg(short, long, default_value = "deepseek-v3-671b-moe")]
        model: String,
        /// Workload kind: inference | training | rl | dream-rsi | batch
        #[arg(short, long, default_value = "inference")]
        workload: String,
    },
    /// Launch the Unified Web Dashboard & OpenAI/Anthropic compatible API Gateway
    Serve {
        /// Port to bind the HTTP dashboard and API gateway
        #[arg(short, long, default_value_t = 7848)]
        port: u16,
        /// Optional path to `kadmium.toml` workspace configuration
        #[arg(short, long)]
        config: Option<String>,
        /// Start with only real probed hardware nodes (default: true unless --demo is passed)
        #[arg(long)]
        live: bool,
        /// Include the 7-node simulated demo fleet alongside live hardware
        #[arg(long)]
        demo: bool,
        /// Spawn a 5-node local + online TCP worker test fleet on ports 7851..=7855 for single-machine testing
        #[arg(long)]
        spawn_fleet: bool,
        /// Local TCP port for the coordinator's built-in `WireFrame` stage worker listener
        #[arg(long, default_value_t = DEFAULT_WORKER_TCP_PORT)]
        worker_port: u16,
    },
    /// Join a live Kadmium cluster as a persistent LAN/WAN `WireFrame` compute worker daemon (or verify a `kdm1_...` ticket)
    Join {
        /// Coordinator HTTP gateway URL (e.g., `http://192.168.1.48:7848`); auto-discovered via UDP LAN broadcast if omitted
        #[arg(short, long)]
        coordinator: Option<String>,
        /// Optional `kdm1_...` Join Ticket token
        #[arg(short, long)]
        ticket: Option<String>,
        /// Preferred TCP port to bind this machine's `WireFrame` (`KDMF`) worker listener
        #[arg(long, default_value_t = DEFAULT_WORKER_TCP_PORT)]
        worker_port: u16,
        /// Optional custom hostname / node label (useful when testing multiple workers)
        #[arg(long)]
        hostname: Option<String>,
        /// Switch the coordinator to Live-Hardware-Only mode upon registration
        #[arg(long)]
        live_only: bool,
        /// Register and run a single verification pass, then exit instead of staying in the daemon heartbeat loop
        #[arg(long)]
        once: bool,
    },
    /// Pull and stage a Hugging Face model/dataset (hf://...) or GitHub RL/Dream-RSI world (gh://...)
    Pull {
        /// Ecosystem URI (e.g., hf://deepseek-ai/DeepSeek-V3@main or gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt)
        #[arg(default_value = "hf://deepseek-ai/DeepSeek-V3@main")]
        uri: String,
    },
    /// Launch the Kadmium Model Context Protocol (MCP) Server over stdio (or --inspect to view manifest & IDE config)
    Mcp {
        /// Print the MCP Server manifest, tools, resources, prompts, and .mcp.json snippet instead of blocking on stdio
        #[arg(short, long)]
        inspect: bool,
    },
    /// Initialize a local Kadmium workspace configuration (`kadmium.toml`) tailored to the probed host
    Init {
        /// Optional output path to write `kadmium.toml` to disk
        #[arg(short, long)]
        out: Option<String>,
    },
    /// Compile and verify a 64 KiB page-aligned `.ktb` (Kadmium Tensor Bank) shard archive
    Compile {
        /// Model identifier
        #[arg(short, long, default_value = "deepseek-v3-671b-moe")]
        model: String,
        /// Start layer index (inclusive)
        #[arg(long, default_value_t = 0)]
        start_layer: u32,
        /// End layer index (inclusive)
        #[arg(long, default_value_t = 3)]
        end_layer: u32,
        /// Optional output `.ktb` file path to write to disk
        #[arg(short, long)]
        out: Option<String>,
    },
    /// Inspect a `.ktb` (Kadmium Tensor Bank) archive and verify its 64 KiB page Merkle proofs
    Inspect {
        /// Optional path to a `.ktb` file on disk (compiles an in-memory archive if omitted)
        path: Option<String>,
    },
    /// Quote and configure leasing idle local compute to the Kadmium Supply Grid (earn USD or 1.25x Cloud Credits)
    Lease {
        /// Daily idle lease hours (0.0 to 24.0)
        #[arg(long, default_value_t = 12.0)]
        hours: f64,
        /// Compensation mode: credits (1.25x Kadmium Cloud credits) | usd (Direct USD payout)
        #[arg(short, long, default_value = "credits")]
        payout: String,
    },
    /// Benchmark Phase 2 fused Transformer/MoE kernels (RMSNorm+RoPE, Flash-GQA PagedAttention, Q4_K SwiGLU), HTTP Range Slicer, and STUN/DERP NAT traversal
    Bench,
    /// Inspect Phase 3/4 Zero-Click LAN Pairing (6-digit SAS), OS Swapchain Hysteresis (<200ms yield / 15s cooldown), and <400ms CloudShield Watcher
    Pairing,
    /// Inspect v1.0 GA Multi-Provider Spot Orderbook, Multi-Tenant Sub-Cluster RBAC, Verde Fraud Proofs, and the Public Dream-RSI World Registry
    /// Inspect v1.0 GA Multi-Provider Spot Orderbook, Multi-Tenant Sub-Cluster RBAC, Verde Fraud Proofs, and the Public Dream-RSI World Registry
    Registry,
    /// Push a signed Over-The-Air (OTA) update (binary, ktb-delta, or policy) across local LAN TCP workers and online WAN nodes
    Update {
        /// Coordinator HTTP gateway URL (defaults to http://127.0.0.1:7848)
        #[arg(short, long)]
        coordinator: Option<String>,
        /// Target release version string (e.g. v0.2.0-cd48)
        #[arg(short, long, default_value = "v0.2.0-cd48")]
        version: String,
        /// Payload kind: binary | ktb-delta | policy
        #[arg(short, long, default_value = "binary")]
        kind: String,
        /// Target scope: all | local | online
        #[arg(short, long, default_value = "all")]
        scope: String,
    },
    /// Profile, scan USB-ADB (`adb devices -l`), and simulate attaching a 20-Slot Phone Farm Box (`phonefarmbox.com`) or mobile phone swarm
    PhoneFarm {
        /// Hardware preset: sd845 (Galaxy S9 20x) | sd865 (Galaxy S20 20x) | sd8gen2 (Galaxy S23 20x)
        #[arg(short, long, default_value = "sd865")]
        preset: String,
        /// Expose each of the 20 motherboards as an individual micro-worker instead of a coherent 160 GB UMA MoE array
        #[arg(short, long)]
        disaggregated: bool,
    },
    /// Import and transcode a local `.gguf` v3 or `.safetensors` model file into a 64 KiB page-aligned `.ktb` weight bank on disk
    Import {
        /// Model identifier
        #[arg(short, long, default_value = "qwen2.5-coder-32b-instruct")]
        model: String,
        /// Optional path to a `.gguf` or `.safetensors` file on disk (synthesizes a reference binary in target/kadmium-models/ if omitted)
        #[arg(short, long)]
        path: Option<String>,
        /// Format hint when synthesizing a reference file: gguf | safetensors
        #[arg(short, long, default_value = "gguf")]
        format: String,
        /// Start layer index (inclusive)
        #[arg(long, default_value_t = 0)]
        start_layer: u32,
        /// End layer index (inclusive)
        #[arg(long, default_value_t = 3)]
        end_layer: u32,
    },
    /// Convert and benchmark EXL3 / QTIP Trellis Quantization (FWHT Incoherence + On-the-Fly Hessian + Fused Viterbi)
    Exl3 {
        /// Model identifier (e.g. deepseek-v3-671b-exl3-2.5bpw | qwen3-235b-moe | llama-3-70b-lora)
        #[arg(short, long, default_value = "deepseek-v3-671b-exl3-2.5bpw")]
        model: String,
        /// Target average bitrate in bits per weight (2.0 to 6.0 bpw)
        #[arg(short, long, default_value_t = 2.55)]
        bpw: f32,
    },
    /// Profile and attach an NVIDIA Tegra X1 / X1+ platform node (Modded Nintendo Switch Mariko/Erista, Jetson Nano B01, Shield TV Pro)
    #[command(visible_alias = "tegra-x1")]
    Switch {
        /// Silicon revision: mariko (16nm T210B01 4GB) | erista (20nm T210 4GB) | mariko-8gb (8GB LPDDR4X BGA mod)
        #[arg(short, long, default_value = "mariko")]
        revision: String,
    },
    /// Probe and benchmark a live Raspberry Pi Compute Module 4 (BCM2711 Cortex-A72 aarch64 NEON) platform node over LAN TCP
    #[command(visible_alias = "cm4", visible_alias = "rpi-cm4")]
    Reterminal {
        /// Optional LAN IP of the Raspberry Pi CM4 node (defaults to 192.168.40.75)
        #[arg(short, long)]
        ip: Option<String>,
    },
    /// Probe and benchmark a live Apple M4 platform node (M4 / M4 Pro / M4 Max / M4 Ultra · Metal 4 GPU + AMX Accelerate) over LAN TCP
    #[command(visible_alias = "apple-m4")]
    M4 {
        /// Optional LAN IP of the Apple M4 node (defaults to 192.168.40.237)
        #[arg(short, long)]
        ip: Option<String>,
    },
    /// Open the Kadmium desktop app (the Companion in its own window) with a local coordinator
    #[command(visible_alias = "app")]
    Desktop {
        /// Port to bind the local HTTP coordinator and API gateway
        #[arg(short, long, default_value_t = 7848)]
        port: u16,
        /// Start with only real probed hardware nodes (default: true unless --demo is passed)
        #[arg(long)]
        live: bool,
        /// Include the 7-node simulated demo fleet alongside live hardware
        #[arg(long)]
        demo: bool,
        /// Also open the Control Room in your browser
        #[arg(long)]
        control_room: bool,
        /// Print the hardware summary and check for the desktop app without opening a window
        #[arg(long)]
        check_only: bool,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Simulate) {
        Commands::Simulate => run_simulation(),
        Commands::Dream => run_dream_rsi(),
        Commands::Slice { pool_size, slices } => run_slice(pool_size, slices),
        Commands::Dataflow {
            model,
            batch_tokens,
        } => run_dataflow(&model, batch_tokens),
        Commands::CloudShield => run_cloud_shield(),
        Commands::Probe => run_probe(),
        Commands::Plan { model, workload } => run_plan(&model, &workload),
        Commands::Serve {
            port,
            config,
            live,
            demo,
            worker_port,
            spawn_fleet,
        } => run_server(port, config.as_deref(), live || !demo, worker_port, spawn_fleet).await,
        Commands::Join {
            coordinator,
            ticket,
            worker_port,
            hostname,
            live_only,
            once,
        } => {
            run_join(
                coordinator,
                ticket,
                worker_port,
                hostname,
                live_only,
                once,
            )
            .await
        }
        Commands::Pull { uri } => run_pull(&uri),
        Commands::Mcp { inspect } => run_mcp(inspect),
        Commands::Init { out } => run_init(out.as_deref()),
        Commands::Compile {
            model,
            start_layer,
            end_layer,
            out,
        } => run_compile(&model, start_layer, end_layer, out.as_deref()),
        Commands::Inspect { path } => run_inspect(path.as_deref()),
        Commands::Lease { hours, payout } => run_lease(hours, &payout),
        Commands::Bench => run_bench().await,
        Commands::Pairing => run_pairing(),
        Commands::Registry => run_registry(),
        Commands::Update {
            coordinator,
            version,
            kind,
            scope,
        } => run_update(coordinator, version, kind, scope).await,
        Commands::PhoneFarm {
            preset,
            disaggregated,
        } => run_phone_farm(&preset, disaggregated),
        Commands::Import {
            model,
            path,
            format,
            start_layer,
            end_layer,
        } => run_import(&model, path.as_deref(), &format, start_layer, end_layer),
        Commands::Exl3 { model, bpw } => run_exl3(&model, bpw),
        Commands::Switch { revision } => run_switch(&revision),
        Commands::Reterminal { ip } => run_reterminal(ip.as_deref()),
        Commands::M4 { ip } => run_m4_mac_mini(ip.as_deref()),
        Commands::Desktop {
            port,
            live,
            demo,
            control_room,
            check_only,
        } => run_desktop(port, live || !demo, control_room, check_only).await,
    }
}

fn resolve_model(name: &str) -> ModelSpec {
    ModelSpec::from_id(name)
}

fn resolve_workload(name: &str) -> WorkloadKind {
    match name.to_lowercase().as_str() {
        "training" | "train" => WorkloadKind::DistributedTraining,
        "batch" | "boinc" => WorkloadKind::BatchPipeline,
        _ => WorkloadKind::Inference,
    }
}

fn print_hardware_table(nodes: &[HardwareNode]) {
    println!(
        "\n{:<22} {:<20} {:<13} {:>10} {:>12} {:>10} {:>8}",
        "NODE ID", "HOSTNAME", "BACKEND", "FP16 TFLOP", "FAST MEM", "SYS RAM", "q* SPLIT"
    );
    println!("{}", "-".repeat(101));
    for n in nodes {
        println!(
            "{:<22} {:<20} {:<13} {:>10.1} {:>9.1} GB {:>7.1} GB {:>8.2}",
            n.id.0,
            n.hostname,
            format!("{:?}", n.backend),
            n.effective_fp16_tflops(),
            n.available_fast_memory_gb(),
            n.memory.system_ram_gb,
            n.compute_q_star_ratio()
        );
    }
}

pub(crate) fn print_shard_plan(plan: &ClusterShardPlan) {
    println!(
        "\n====================================================================================================="
    );
    println!(
        " KADMIUM ASYMMETRIC SHARD PLAN (Epoch #{}) — {} [{:?}]",
        plan.epoch, plan.model_display_name, plan.workload
    );
    println!(
        " Total Layers: {} | Static Footprint: {:.2} GiB | Est. Throughput: {:.2} {}",
        plan.total_layers,
        plan.total_model_weight_gb,
        plan.estimated_throughput,
        plan.throughput_unit
    );
    if let Some(strat) = plan.training_strategy {
        println!(" Distributed Training Strategy: {:?}", strat);
    }
    println!(
        "====================================================================================================="
    );
    println!(
        "{:<22} {:<14} {:<12} {:>7} {:>10} {:>10} {:>10} {:<24}",
        "HOSTNAME",
        "POLITE STATE",
        "LAYER SHARD",
        "SHARE",
        "HOT (GB)",
        "WARM (GB)",
        "NVME (GB)",
        "CHECKPOINT / EDGE ROLE"
    );
    println!("{}", "-".repeat(115));

    for a in &plan.assignments {
        let layer_str = match (a.start_layer, a.end_layer) {
            (Some(s), Some(e)) => format!("L{:02}..L{:02} ({:02})", s, e, a.layer_count),
            _ => "Edge / Aux".to_string(),
        };
        let role_or_ckpt =
            if a.layer_count > 0 && plan.workload == WorkloadKind::DistributedTraining {
                format!("{:?}", a.training_checkpoint_policy)
            } else if !a.auxiliary_tasks.is_empty() {
                format!("{:?}", a.auxiliary_tasks[0])
            } else {
                format!("q*={:.2}", a.tiered_memory.q_star_accel_ratio)
            };

        println!(
            "{:<22} {:<14} {:<12} {:>6.1}% {:>10.2} {:>10.2} {:>10.2} {:<24}",
            a.hostname,
            format!("{:?}", a.polite_state),
            layer_str,
            a.compute_share_pct,
            a.tiered_memory.hot_vram_or_unified_gb,
            a.tiered_memory.warm_system_ram_gb,
            a.tiered_memory.cold_nvme_stream_gb,
            role_or_ckpt
        );
    }
}

fn print_shift_summary(shift: &ResponsibilityShiftPlan) {
    println!(
        "\n>>> [LIVE COMPUTE RESPONSIBILITY SHIFT: Epoch #{} -> Epoch #{}] Trigger: {:?}",
        shift.previous_epoch, shift.new_epoch, shift.trigger
    );
    println!(
        "    Background FTS3 Pre-Stage from NAS: {:.2} GiB | Atomic Boundary Delta Handoff: {:.2} GiB | Pause: {:.1} ms",
        shift.total_background_prestage_gb,
        shift.total_live_boundary_delta_gb,
        shift.estimated_switchover_pause_ms
    );
    for m in &shift.migrations {
        println!(
            "    * Migrate Layers L{:02}..L{:02} ({:02} layers): {} ---> {} (Pre-stage: {:.2} GiB, Live Delta: {:.2} GiB)",
            m.start_layer,
            m.end_layer,
            m.layer_count,
            m.from_node,
            m.to_node,
            m.background_fts3_prestage_gb,
            m.live_boundary_delta_gb
        );
    }
}

fn print_rl_plan(rl_plan: &DisaggregatedRlPlan) {
    println!(
        "\n====================================================================================================="
    );
    println!(
        " KADMIUM DISAGGREGATED RL ENVIRONMENT & ROLLOUT PLAN (Epoch #{}) — {}",
        rl_plan.epoch, rl_plan.env_suite.display_name
    );
    println!(
        " Algorithm: {:?} | Vectorized Env Slots: {} | Policy Sync Interval: {} steps | Est: {:.1} rollouts/min",
        rl_plan.algorithm,
        rl_plan.total_vectorized_env_slots,
        rl_plan.policy_sync_interval_steps,
        rl_plan.estimated_cluster_episodes_per_min
    );
    println!(
        "====================================================================================================="
    );
    println!(
        "{:<22} {:<14} {:<16} {:>12} {:>16} {:<30}",
        "HOSTNAME",
        "POLITE STATE",
        "POLICY LAYERS",
        "ENV SLOTS",
        "ROLLOUTS/MIN",
        "DISAGGREGATED RL ROLES"
    );
    println!("{}", "-".repeat(115));
    for a in &rl_plan.node_assignments {
        let layers = match a.policy_layer_range {
            Some((s, e)) => format!("L{:02}..L{:02}", s, e),
            None => "Env Sandbox Only".to_string(),
        };
        let roles = a
            .rl_roles
            .iter()
            .map(|r| format!("{:?}", r))
            .collect::<Vec<_>>()
            .join(" + ");
        println!(
            "{:<22} {:<14} {:<16} {:>12} {:>16.1} {:<30}",
            a.hostname,
            format!("{:?}", a.polite_state),
            layers,
            a.parallel_env_slots,
            a.rollouts_per_min,
            roles
        );
    }
    if let Some(report) = &rl_plan.dream_rsi_report {
        print_dream_rsi_report(report);
    }
}

fn print_dream_rsi_report(report: &DreamRsiEvolutionReport) {
    let replay_hosts: Vec<String> = report
        .executed_on_replay_nodes
        .iter()
        .map(|n| n.0.clone())
        .collect();
    println!(
        "\n--- 💭 Dream-RSI Offline Replay Simulation (Gen #{} | Domain: {:?} | Tree: {}) ---",
        report.generation, report.domain, report.tree_id
    );
    println!(
        "    Replay World Hosts (Zero GPU Cost): {}",
        replay_hosts.join(", ")
    );
    println!(
        "    Promoted Exploration Policy: {} (Peak Discovery: {:.2}x | Online Compute Saved: {:.1}%)",
        report.promoted_policy_id, report.peak_speedup_achieved, report.cost_savings_pct
    );
    for eval in &report.evaluated_policies {
        let marker = if eval.policy.policy_id == report.promoted_policy_id {
            "★ WINNER"
        } else {
            "        "
        };
        println!(
            "      {} {:<24} | Peak: {:.2}x | Cost-to-Peak: {:>4.1} evals | Pruned: {} | Saved: {:>5.1}% | Order: {:?}",
            marker,
            eval.policy.policy_id,
            eval.peak_discovery_score,
            eval.cost_to_reach_peak,
            eval.pruned_branches,
            eval.discovery_cost_savings_pct,
            eval.simulated_traversal_order
        );
    }
}

fn run_dream_rsi() {
    let nodes = sample_personal_cluster();
    let model = ModelSpec::llama_3_70b_lora();
    let env_suite = RlEnvironmentSuiteSpec::dream_rsi_evolving_world();
    let rl_plan = RlTopologyPlanner::compute_rl_plan(1, &model, &env_suite, &nodes);

    let tree = DiscoveryTree::sample_gpu_kernel_discovery_tree();
    println!(
        "====================================================================================================="
    );
    println!(
        " 💭 KADMIUM DREAM-RSI ENGINE — Recursive Self-Improvement through Evolving Worlds (arXiv:2609.14858)"
    );
    println!(
        "====================================================================================================="
    );
    println!(
        "\n[1/2] Historical Discovery Tree Loaded from Home NAS ({} recorded nodes):",
        tree.nodes.len()
    );
    for n in &tree.nodes {
        println!(
            "  * Node #{:02} (Parent: {:>4}, Depth: {}) | Score: {:.2}x | Prior Novelty: {:.2} | {}",
            n.node_id,
            n.parent_id
                .map(|p| format!("#{:02}", p))
                .unwrap_or_else(|| "ROOT".into()),
            n.depth,
            n.realized_score,
            n.prior_novelty,
            n.proposal_label
        );
    }

    println!("\n[2/2] Disaggregated Dream-RSI Topology & Offline Policy Evolution:");
    print_rl_plan(&rl_plan);
}

fn run_slice(pool_size: usize, num_slices: usize) {
    let pool = sample_node_pool(pool_size);
    let model = ModelSpec::deepseek_v3_671b_moe();
    let layout = SubClusterPartitioner::partition_pool(&pool, num_slices, &model);

    println!(
        "==================================================================================================================="
    );
    println!(
        " 🧩 KADMIUM SUB-COMPUTE ENVIRONMENT PARTITIONER — {} Global Nodes -> {} Isolated Clusters ({} Nodes/Cluster)",
        layout.total_pool_nodes, layout.num_slices, layout.nodes_per_slice
    );
    println!(
        "==================================================================================================================="
    );
    println!(
        "{:<24} {:<30} {:>6} {:>11} {:>10} {:>10} {:<32}",
        "SUB-CLUSTER ID",
        "ENVIRONMENT PURPOSE",
        "NODES",
        "FP16 TFLOP",
        "FAST MEM",
        "SYS RAM",
        "DATAFLOW MODE / API NAMESPACE"
    );
    println!("{}", "-".repeat(129));
    for s in &layout.slices {
        println!(
            "{:<24} {:<30} {:>6} {:>11.1} {:>7.1} GB {:>7.1} GB {:?} ({})",
            s.slice_id,
            format!("{:?}", s.purpose),
            s.node_count,
            s.aggregate_fp16_tflops,
            s.aggregate_fast_memory_gb,
            s.aggregate_system_ram_gb,
            s.dataflow_mode,
            s.api_namespace
        );
    }
    println!(
        "\nEach Sub-Compute Environment runs an isolated ShardPlanner, FTS3 Transfer Queue, and API Gateway Namespace."
    );
}

fn run_dataflow(model_name: &str, batch_tokens: u32) {
    let nodes = sample_personal_cluster();
    let model = resolve_model(model_name);
    let report = DataflowStreamPlanner::evaluate_dataflows(&model, &nodes, batch_tokens);

    println!(
        "==================================================================================================================="
    );
    println!(
        " ⚡ KADMIUM TPU / COMPUTE-IN-MEMORY (CIM) DATAFLOW PLANNER — {} (Batch: {} tokens)",
        model.display_name, report.evaluated_batch_tokens
    );
    println!(
        " Low-Batch Recommendation: {:?} | High-Batch Recommendation: {:?} (Crossover: {} tokens)",
        report.recommended_low_batch_mode,
        report.recommended_high_batch_mode,
        report.crossover_batch_tokens
    );
    println!(
        "==================================================================================================================="
    );
    for p in &report.profiles {
        println!(
            "\n  * Mode: {:?}  -->  Est. Throughput: {:.1} tokens/sec (Overlap: {:.1}%)",
            p.mode, p.estimated_tokens_per_sec, p.compute_stream_overlap_pct
        );
        println!("    - Stationary Tensor: {}", p.stationary_tensor);
        println!(
            "    - Streaming Tensor:  {} ({:.2} MiB/step | Arith. Intensity: {:.1} FLOPs/Byte)",
            p.streaming_tensor,
            p.streamed_volume_per_step_mb,
            p.arithmetic_intensity_flops_per_byte
        );
        println!("    - Optimal Use Case:  {}", p.best_suited_for);
    }
}

fn run_plan(model_name: &str, workload_name: &str) {
    let nodes = sample_personal_cluster();
    let model = resolve_model(model_name);
    if workload_name.eq_ignore_ascii_case("rl")
        || workload_name.eq_ignore_ascii_case("dream-rsi")
        || workload_name.eq_ignore_ascii_case("dream")
    {
        print_hardware_table(&nodes);
        let env_suite = if workload_name.to_lowercase().contains("dream") {
            RlEnvironmentSuiteSpec::dream_rsi_evolving_world()
        } else {
            RlEnvironmentSuiteSpec::rust_python_code_sandbox()
        };
        let rl_plan = RlTopologyPlanner::compute_rl_plan(1, &model, &env_suite, &nodes);
        print_rl_plan(&rl_plan);
        return;
    }
    let workload = resolve_workload(workload_name);
    let plan = ShardPlanner::compute_plan(1, &model, workload, &nodes);

    print_hardware_table(&nodes);
    print_shard_plan(&plan);

    let (choice, placement) =
        kadmium_sharding::StrategyCostModel::default().choose(&model, &nodes, None);
    println!("\n--- Grid Execution Shape (modelled from the capability vectors, not measured) ---");
    println!("  * Chosen shape: {:?}", choice.chosen);
    println!(
        "  * Pipeline chain:  {:>8.2} ms/token | {:>3} hops | {:>8.2} ms cold streaming",
        choice.chain.ms_per_token, choice.chain.hops_per_step, choice.chain.cold_stream_ms
    );
    println!(
        "  * Expert-parallel: {:>8.2} ms/token | {:>3} hops | {:>8.2} ms cold streaming",
        choice.expert_parallel.ms_per_token,
        choice.expert_parallel.hops_per_step,
        choice.expert_parallel.cold_stream_ms
    );
    println!("  * Placement: {}", placement.summary());
    println!("  * Why: {}", choice.rationale);

    let transfers = Fts3TransferScheduler::plan_initial_staging(&plan, &nodes);
    println!(
        "\n--- CERN FTS3 / Davix Third-Party P2P Staging Transfers ({} Direct QUIC Streams) ---",
        transfers.len()
    );
    for t in transfers {
        println!(
            "  [{}] {} ==> {} | Layers L{:02}..L{:02} | {:.2} GiB @ {:.2} Gbps ({} streams, ~{:.1}s)",
            t.transfer_id,
            t.source_node,
            t.destination_node,
            t.start_layer,
            t.end_layer,
            t.volume_gb,
            t.allocated_bandwidth_gbps,
            t.parallel_quic_streams,
            t.estimated_duration_sec
        );
    }

    let opt = MinMaxStageLatencyOptimizer::optimize_stage_boundaries(&plan, &nodes);
    println!(
        "\n--- Min-Max Pipeline Bottleneck Optimizer (Initial {:.2} ms -> Optimized {:.2} ms, -{:.1}%) ---",
        opt.initial_bottleneck_ms, opt.optimized_bottleneck_ms, opt.latency_reduction_pct
    );
    for (host, cnt, lat) in &opt.refined_stage_layers {
        println!(
            "  * {:<22} {:>2} layers | Stage Latency: {:>5.2} ms",
            host, cnt, lat
        );
    }

    let kv_allocs = DistributedPagedKvPlanner::plan_kv_cache_pages(&plan, &model, &nodes);
    println!("\n--- Distributed PagedAttention KV-Cache Planner (16 tokens/page) ---");
    for kv in &kv_allocs {
        println!(
            "  * {:<22} Layers: {:>2} | KV Budget: {:>5.2} GB | Pages: {:>4} | Max Context: {:>6} tokens",
            kv.hostname,
            kv.assigned_layers,
            kv.kv_budget_gb,
            kv.allocated_kv_pages,
            kv.max_context_tokens_supported
        );
    }
}

fn run_simulation() {
    println!(
        "====================================================================================================="
    );
    println!(" 🧪 KADMIUM (Cd-48) — HETEROGENEOUS AI/ML COMPUTE FABRIC SIMULATION");
    println!(
        "====================================================================================================="
    );

    let mut nodes = sample_personal_cluster();
    println!(
        "\n[1/5] Discovered Personal + Home Lab Mesh ({} heterogeneous nodes):",
        nodes.len()
    );
    print_hardware_table(&nodes);

    // 1. MoE Inference Plan
    let moe_model = ModelSpec::deepseek_v3_671b_moe();
    let moe_plan = ShardPlanner::compute_plan(1, &moe_model, WorkloadKind::Inference, &nodes);
    println!(
        "\n[2/5] Initial Asymmetric Inference Shard Plan (FreeToken q* + Colibri NVMe MoE Streaming):"
    );
    print_shard_plan(&moe_plan);

    // 2. Distributed Training Plan
    let train_model = ModelSpec::llama_3_70b_lora();
    let train_plan =
        ShardPlanner::compute_plan(1, &train_model, WorkloadKind::DistributedTraining, &nodes);
    println!("\n[3/5] Initial Distributed Training Plan (Asymmetric 1F1B Pipeline + QLoRA):");
    print_shard_plan(&train_plan);

    let telemetry = TrainingJobTelemetry::sample_for_plan(&train_plan);
    println!(
        "\n      Training Telemetry: Step {}/{} | Loss: {:.3} | 1F1B Micro-Batches: {} | Bubble: {:.1}% | Codec: {:?} ({:.1}x) | Verde RepOps: {}",
        telemetry.current_step,
        telemetry.total_steps,
        telemetry.loss,
        telemetry.micro_batches_in_flight,
        telemetry.pipeline_bubble_pct,
        telemetry.compression_codec,
        telemetry.gradient_compression_ratio,
        telemetry.verde_repops_verified
    );

    // 3. Live Compute Responsibility Shift #1: User starts gaming on Main PC!
    println!(
        "\n[4/5] Simulating Event: User launches a game on Main Desktop PC (cameron-main-rig)!"
    );
    nodes[0].polite_state = PoliteState::GamingOrHeavyApp;
    let shift_gaming = ShardPlanner::plan_responsibility_shift(
        &train_plan,
        &train_model,
        &nodes,
        ShiftTriggerReason::HostEnteredGamingMode {
            node_id: nodes[0].id.clone(),
        },
    );
    print_shift_summary(&shift_gaming);
    print_shard_plan(&shift_gaming.new_plan);

    // 4. Live Compute Responsibility Shift #2: User rents a Cloud H100 GPU for 3 hours to boost training!
    println!(
        "\n[5/5] Simulating Event: Ephemeral Rented Cloud H100 GPU joins mesh to boost training!"
    );
    let burst_node = sample_cloud_burst_gpu_node();
    nodes.push(burst_node.clone());
    let shift_burst = ShardPlanner::plan_responsibility_shift(
        &shift_gaming.new_plan,
        &train_model,
        &nodes,
        ShiftTriggerReason::CloudBurstNodeJoined {
            node_id: burst_node.id,
        },
    );
    print_shift_summary(&shift_burst);
    print_shard_plan(&shift_burst.new_plan);

    // 5. Disaggregated RL Environment & Rollout Fabric Plan
    let rl_env_suite = RlEnvironmentSuiteSpec::rust_python_code_sandbox();
    let rl_plan = RlTopologyPlanner::compute_rl_plan(3, &train_model, &rl_env_suite, &nodes);
    print_rl_plan(&rl_plan);

    let workunits = sample_boinc_workunits();
    println!("\n--- Active BOINC-Style Polite Workunits & Deterministic Canary Verifications ---");
    for wu in workunits {
        println!(
            "  * [{}] {} on {} | Progress: {:.1}% | Canary Cosine: {:.5} (Verified: {})",
            wu.workunit_id,
            wu.title,
            wu.assigned_node,
            wu.progress_pct,
            wu.canary_cosine_similarity,
            wu.verified
        );
    }
    println!(
        "\nSimulation complete. Run `cargo run --bin kadmium -- serve` to open the live Web Dashboard!"
    );
}

async fn run_server(
    port: u16,
    config_path: Option<&str>,
    live_only: bool,
    worker_port: u16,
    spawn_fleet: bool,
) {
    let options = CoordinatorOptions {
        port,
        config_path: config_path.map(str::to_string),
        live_only,
        worker_port,
        spawn_fleet,
    };
    if let Err(err) = run_coordinator(options).await {
        eprintln!("kadmium: {err}");
        std::process::exit(1);
    }
}

fn run_phone_farm(preset_slug: &str, disaggregated: bool) {
    let preset = PhoneFarmBoxPreset::from_slug(preset_slug);
    let topology = if disaggregated {
        PhoneFarmTopologyMode::DisaggregatedMicroWorkerSwarm
    } else {
        PhoneFarmTopologyMode::CoherentMoEAndDreamRsiArray
    };
    let spec = PhoneFarmBoxController::build_chassis_spec(
        preset,
        PhoneFarmTransportMode::Usb3AdbMultiplexedTunnel,
        topology,
    );

    println!("====================================================================================");
    println!("  KADMIUM (Cd-48) — 20-SLOT PHONE FARM BOX & MOBILE SWARM PROFILER");
    println!("====================================================================================");
    println!("  Chassis Model:        {}", spec.display_name);
    println!("  Slot Count:           {} stripped ARM64 motherboards (battery-free BMS cables)", spec.slot_count);
    println!("  Power & Cooling:      {} | {:.0}W avg wall draw | {}", spec.psu_rating, spec.average_wall_power_watts, spec.cooling_system);
    println!("  Est. Hardware Cost:   ~${:.0} USD", spec.estimated_hardware_cost_usd);
    println!("  Aggregate CPU:        {} ARM64 cores", spec.aggregate_cpu_cores);
    println!(
        "  Aggregate Memory:     {:.0} GB Unified LPDDR ({:.1} GB/s aggregate parallel MoE bandwidth)",
        spec.aggregate_lpddr_unified_gb, spec.aggregate_parallel_mem_bw_gbps
    );
    println!(
        "  Aggregate UFS Flash:  {:.0} GB UFS ({:.1} GB/s aggregate sequential read for Colibri MoE)",
        spec.aggregate_ufs_flash_gb, spec.aggregate_ufs_read_bw_gbps
    );
    println!(
        "  Aggregate Compute:    {:.1} FP16 TFLOPS | {:.1} INT4/NPU TOPS",
        spec.aggregate_fp16_tflops, spec.aggregate_int4_tops
    );
    println!(
        "  USB-ADB Bridge Scan:  adb_installed={} | physical_usb_devices={}",
        spec.adb_discovery.adb_binary_detected,
        spec.adb_discovery.connected_physical_devices.len()
    );
    println!("------------------------------------------------------------------------------------");
    println!("  Sample Motherboard Slots (1..=5 of 20):");
    for slot in spec.slots.iter().take(5) {
        println!(
            "    [Slot {:02}] {} ({}) | {:.0}GB LPDDR + {:.0}GB UFS | TCP :{} | MoE Experts E{:03}..E{:03} | {:.1}°C",
            slot.slot_index,
            slot.hostname,
            slot.adb_serial,
            slot.lpddr_unified_gb,
            slot.ufs_flash_gb,
            slot.host_forwarded_tcp_port,
            slot.pinned_moe_expert_range.0,
            slot.pinned_moe_expert_range.1,
            slot.board_temp_c
        );
    }

    let mut cluster = sample_personal_cluster();
    if disaggregated {
        cluster.extend(PhoneFarmBoxController::to_individual_slot_nodes(&spec));
    } else {
        cluster.push(PhoneFarmBoxController::to_aggregated_hardware_node(&spec));
    }
    cluster.push(PhoneFarmBoxController::sample_standalone_mobile_phone());

    let model = ModelSpec::deepseek_v3_671b_moe();
    let inf_plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);
    let env_suite = RlEnvironmentSuiteSpec::dream_rsi_evolving_world();
    let rl_plan = RlTopologyPlanner::compute_rl_plan(1, &model, &env_suite, &cluster);

    println!("------------------------------------------------------------------------------------");
    println!(
        "  DeepSeek-V3 671B MoE Shard Plan with Phone Farm Box Attached ({:.1} tok/s):",
        inf_plan.estimated_throughput
    );
    for assign in &inf_plan.assignments {
        println!(
            "    - {:<28} ({:?}) -> Layers {:?}..={:?} ({} layers, {:.1}% share) | Aux: {:?}",
            assign.hostname,
            assign.device_kind,
            assign.start_layer,
            assign.end_layer,
            assign.layer_count,
            assign.compute_share_pct,
            assign.auxiliary_tasks
        );
    }
    println!(
        "  Dream-RSI & RL Arena Capacity: {} parallel WASI/MCTS env slots ({:.1} rollouts/min)",
        rl_plan.total_vectorized_env_slots, rl_plan.estimated_cluster_episodes_per_min
    );
    println!("====================================================================================");
}

fn run_import(
    model_id: &str,
    source_path: Option<&str>,
    format_hint: &str,
    start_layer: u32,
    end_layer: u32,
) {
    match LocalModelDiskImporter::import_or_transcode_to_disk(
        model_id,
        source_path,
        Some(format_hint),
        start_layer,
        end_layer,
        None,
    ) {
        Ok(m) => {
            println!("====================================================================================");
            println!("  KADMIUM (Cd-48) — LOCAL GGUF / SAFETENSORS -> 64 KiB .KTB DISK IMPORTER");
            println!("====================================================================================");
            println!("  Model Identifier:     {} ({})", m.model_id, m.derived_model_spec.display_name);
            println!("  Source Format:        {:?}", m.source_format);
            println!("  Source File Path:     {} ({} KiB)", m.source_file_path, m.source_file_bytes / 1024);
            println!("  Compiled .ktb Path:   {} ({} KiB, {} x 64KiB pages)", m.compiled_ktb_path, m.ktb_size_bytes / 1024, m.ktb_page_count);
            println!("  Transcoded Layers:    L{}..=L{} ({} tensor banks)", m.start_layer, m.end_layer, m.parsed_tensor_count);
            println!("  Merkle Root (FNV64):  {:#018x}", m.merkle_root_fnv64);
            println!("  64 KiB Alignment:     {}", if m.zero_copy_alignment_verified { "VERIFIED (mmap / io_uring / IoRing ready)" } else { "FAILED" });
            println!("====================================================================================");
        }
        Err(e) => {
            eprintln!("Failed to import model '{}': {}", model_id, e);
        }
    }
}

fn run_exl3(model_name: &str, target_bpw: f32) {
    let cluster = sample_personal_cluster();
    let model = resolve_model(model_name);
    let report = DistributedExl3ConversionEngine::run_conversion_and_benchmark(
        &model,
        target_bpw,
        &cluster,
    );

    println!("====================================================================================");
    println!("  KADMIUM (Cd-48) — EXL3 / QTIP TRELLIS QUANTIZER & DISTRIBUTED CONVERTER");
    println!("====================================================================================");
    println!("  Model:                {} ({})", report.model_name, report.model_id);
    println!(
        "  Bitrate:              {:.2} bpw target -> {:.2} bpw effective (Bitshift Trellis L=4, 16 states)",
        report.target_average_bpw, report.achieved_effective_bpw
    );
    println!(
        "  Footprint Reduction:  {:.1} GB (FP16) -> {:.1} GB (Q4_K) -> {:.1} GB (EXL3 .ktb) [-{:.1} GB vs Q4_K]",
        report.original_fp16_size_gb,
        report.legacy_q4k_size_gb,
        report.exl3_ktb_size_gb,
        report.vram_savings_vs_q4k_gb
    );
    println!(
        "  FWHT Incoherence:     Outlier Peak/RMS {:.2}x (pre-FWHT) -> {:.2}x (post-FWHT Rademacher + Hadamard)",
        report.pre_fwht_peak_to_rms, report.post_fwht_peak_to_rms
    );
    println!(
        "  Fused Viterbi SNR:    {:.2} dB (EXL3 Trellis) vs {:.2} dB (Scalar) [+{:.2} dB gain | Hessian Loss: {:.5}]",
        report.exl3_viterbi_snr_db,
        report.naive_scalar_snr_db,
        report.snr_gain_over_scalar_db,
        report.hessian_proxy_loss
    );
    println!(
        "  Single-Step Cluster:  ~{:.1} minutes across {} active conversion nodes",
        report.estimated_cluster_conversion_minutes,
        report.node_conversion_shards.len()
    );
    println!("------------------------------------------------------------------------------------");
    println!("  Distributed Single-Step HF -> EXL3 .ktb Node Conversion Schedule:");
    for shard in &report.node_conversion_shards {
        println!(
            "    - {:<24} ({:<10}) | Layers L{:02}..=L{:02} ({:>2} layers) | Attn {:.2} bpw / MoE {:.2} bpw | {:>6.1} GB .ktb (~{:.1} min)",
            shard.node_name,
            shard.hardware_backend,
            shard.assigned_layer_start,
            shard.assigned_layer_start + shard.assigned_layer_count.saturating_sub(1),
            shard.assigned_layer_count,
            shard.attention_trunk_bpw,
            shard.routed_experts_bpw,
            shard.output_ktb_shard_gb,
            shard.estimated_conversion_seconds / 60.0
        );
    }
    println!("====================================================================================");
}

fn run_switch(revision_slug: &str) {
    let rev = SwitchSiliconRevision::from_slug(revision_slug);
    let spec =
        ModdedSwitchController::build_spec(rev, SwitchRuntimeEnvironment::SwitchrootL4tJetPackCuda);
    let node = ModdedSwitchController::to_hardware_node(&spec);

    println!("====================================================================================");
    println!("  KADMIUM (Cd-48) — MODDED NINTENDO SWITCH (TEGRA X1 / X1+) E-WASTE NODE PROFILER");
    println!("====================================================================================");
    println!("  Device Revision:      {} ({}nm TSMC)", spec.display_name, spec.process_node_nm);
    println!("  Jetson Equivalency:   {}", spec.jetson_equivalent);
    println!("  Runtime Stack:        {:?}", spec.runtime_env);
    println!(
        "  CPU & Maxwell GPU:    {}x ARM Cortex-A57 @ {} MHz | {} CUDA cores (sm_53 FP16) @ {} MHz",
        spec.cpu_cores_a57, spec.cpu_clock_mhz, spec.maxwell_cuda_cores, spec.gpu_clock_mhz
    );
    println!(
        "  Unified Memory:       {:.1} GB LPDDR4X @ {} MHz ({:.1} GB/s bandwidth | q* = {:.2})",
        spec.unified_lpddr_gb,
        spec.emc_memory_clock_mhz,
        spec.unified_mem_bw_gbps,
        node.compute_q_star_ratio()
    );
    println!(
        "  Compute & Efficiency: {:.2} FP16 TFLOPS ({:.2} FP32) @ {:.1}W docked ({:.2}x Tokens/Wh vs desktop)",
        spec.fp16_tflops,
        spec.fp32_tflops,
        spec.dock_safety.wall_power_watts,
        spec.tokens_per_watt_hour_advantage_vs_desktop
    );
    println!(
        "  Dock & Battery Guard: BQ24193 80% Charge-Limit/Bypass={} | SoC {:.1}°C | Fan {}%",
        spec.dock_safety.bq24193_charge_limit_or_bypass_active,
        spec.dock_safety.soc_temp_celsius,
        spec.dock_safety.fan_duty_pct
    );
    println!("------------------------------------------------------------------------------------");
    println!("  Optimal Kadmium Workloads for Switch 1 ({}):", node.hostname);
    for w in &spec.recommended_kadmium_workloads {
        println!("    * {}", w);
    }
    println!("====================================================================================");
}

fn run_reterminal(custom_ip: Option<&str>) {
    let report = ReTerminalCm4LanBridge::probe_and_verify(custom_ip);
    println!("====================================================================================");
    println!("  KADMIUM (Cd-48) — SEEED STUDIO reTERMINAL CM4 (ARMv8 NEON) LIVE LAN PROBE");
    println!("====================================================================================");
    println!(
        "  Physical Board Online: {} ({})",
        report.physical_board_online, report.lan_endpoint
    );
    println!("  Board Model:           {}", report.board_model);
    println!("  MAC Address / OUI:     {}", report.mac_address_oui);
    println!("  SoC Architecture:      {}", report.soc_architecture);
    println!("  Kernel Uname:          {}", report.execution_proof.kernel_uname);
    println!(
        "  Live SoC Telemetry:    {:.1}°C | {} MHz | eMMC {:.2}/{:.2} GiB free",
        report.telemetry.temp_c,
        report.telemetry.cpu_freq_mhz,
        report.telemetry.emmc_free_gib,
        report.telemetry.emmc_total_gib
    );
    println!(
        "  I2C HMI Peripherals:   {}",
        report.telemetry.i2c_peripherals.join(", ")
    );
    println!("------------------------------------------------------------------------------------");
    println!(
        "  LAN TCP Round-Trip:    {:.2} ms (persistent TCP_NODELAY socket)",
        report.execution_proof.lan_round_trip_ms
    );
    println!(
        "  NEON 64x64 FP32 GEMM:  {:.3} ms (vfmaq_f32 | FNV-1a64: {:#018x} | sample: {:?})",
        report.execution_proof.neon_gemm_64x64_ms,
        report.execution_proof.neon_gemm_fnv1a64,
        report.execution_proof.neon_sample4
    );
    println!(
        "  eMMC 64 KiB .ktb I/O:  write {:.3} ms | read {:.3} ms | witness_verified={}",
        report.execution_proof.emmc_64k_write_ms,
        report.execution_proof.emmc_64k_read_ms,
        report.execution_proof.emmc_witness_verified
    );
    println!(
        "  Front-Bezel LEDs:      usr_led0..2 pulsed={}",
        report.execution_proof.front_bezel_leds_pulsed
    );
    if report.physical_board_online {
        let ip = custom_ip.unwrap_or(kadmium_core::DEFAULT_RETERMINAL_CM4_IP);
        if let Some(stress) = ReTerminalCm4LanBridge::run_stress_benchmark_on_cm4(ip, 200) {
            println!("------------------------------------------------------------------------------------");
            println!(
                "  200-Iter NEON Stress:  min {:.3} ms | p50 {:.3} ms | p95 {:.3} ms | p99 {:.3} ms",
                stress.min_ms, stress.p50_ms, stress.p95_ms, stress.p99_ms
            );
            println!(
                "  1-Core vs 4-Core OMP:  128x128 GEMM 1c={:.3} ms -> 4c={:.3} ms ({:.2}x speedup | {:.2} GFLOP/s)",
                stress.gemm128_1core_ms, stress.gemm128_4core_ms, stress.quad_core_speedup, stress.quad_core_gflops
            );
            println!(
                "  HW ARMv8 CRC32C:       {:.2} GB/s (__crc32cd 64 KiB .ktb page verification)",
                stress.hw_crc32c_gbps
            );
            println!(
                "  Live Sensors & GPU:    LIS331DLH Accel={} mg | LTR-303 Light=(vis+ir={}, ir={}) | GPU={}",
                stress.accel_mg_xyz, stress.ltr303_vis_ir_raw, stress.ltr303_ir_raw, stress.v3d_gpu
            );
            println!(
                "  Thermal & Clock Delta: {:.1}°C -> {:.1}°C | {} -> {} MHz (FNV-1a64: {:#018x})",
                stress.temp_before_c, stress.temp_after_c, stress.freq_before_mhz, stress.freq_after_mhz, stress.last_fnv1a64
            );
        }
    }
    println!("====================================================================================");
}

fn run_m4_mac_mini(custom_ip: Option<&str>) {
    let report = kadmium_core::M4MacMiniLanBridge::probe_and_verify(custom_ip);
    println!("====================================================================================");
    println!("  KADMIUM (Cd-48) — APPLE M4 MAC MINI (METAL 4 GPU + AMX ACCELERATE) LIVE PROBE");
    println!("====================================================================================");
    println!(
        "  Physical Board Online: {} ({})",
        report.physical_board_online, report.lan_endpoint
    );
    println!("  Machine Model:         {}", report.board_model);
    println!("  MAC Address / OUI:     {}", report.mac_address_oui);
    println!("  SoC Architecture:      {}", report.soc_architecture);
    println!("  Kernel Uname:          {}", report.execution_proof.kernel_uname);
    println!(
        "  Metal 4 GPU & UMA:     {} (pipeline_ready={}) | {:.2}/{:.1} GiB Metal working set",
        report.telemetry.metal_gpu_name,
        report.telemetry.metal_pipeline_ready,
        report.telemetry.metal_recommended_working_set_gib,
        report.telemetry.unified_memory_gib
    );
    println!(
        "  Apple Silicon Cores:   {} CPU ({}P + {}E) | {} GPU | {} Neural Engine | NVMe {:.2}/{:.2} GiB free",
        report.telemetry.cpu_cores_total,
        report.telemetry.cpu_cores_perf,
        report.telemetry.cpu_cores_eff,
        report.telemetry.gpu_cores,
        report.telemetry.ane_cores,
        report.telemetry.ssd_free_gib,
        report.telemetry.ssd_total_gib
    );
    println!("------------------------------------------------------------------------------------");
    println!(
        "  LAN TCP Round-Trip:    {:.2} ms (persistent TCP_NODELAY socket)",
        report.execution_proof.lan_round_trip_ms
    );
    println!(
        "  Metal 4 64x64 GEMM:    {:.3} ms (10-Core GPU Shader | FNV-1a64: {:#018x} | sample: {:?})",
        report.execution_proof.metal_gpu_gemm_64x64_ms,
        report.execution_proof.metal_gpu_gemm_fnv1a64,
        report.execution_proof.metal_sample4
    );
    println!(
        "  Apple M4 AMX SGEMM:    {:.3} ms (512x512 cblas_sgemm -> {:.2} GFLOP/s | FNV-1a64: {:#018x})",
        report.execution_proof.amx_sgemm_512_ms,
        report.execution_proof.amx_sgemm_512_gflops,
        report.execution_proof.amx_sgemm_fnv1a64
    );
    println!(
        "  HW ARMv9.2-A CRC32C:   {:.2} GB/s (__crc32cd 64 KiB .ktb page verification)",
        report.execution_proof.hw_crc32c_gbps
    );
    println!(
        "  Apple NVMe 64 KiB I/O: write {:.3} ms | read {:.3} ms | witness_verified={}",
        report.execution_proof.nvme_64k_write_ms,
        report.execution_proof.nvme_64k_read_ms,
        report.execution_proof.nvme_witness_verified
    );
    if report.physical_board_online {
        let ip = custom_ip.unwrap_or(kadmium_core::DEFAULT_M4_MAC_MINI_IP);
        if let Some(stress) = kadmium_core::M4MacMiniLanBridge::run_stress_benchmark_on_m4(ip, 25) {
            println!("------------------------------------------------------------------------------------");
            println!(
                "  25-Iter Metal 4 GPU:   min {:.3} ms | p50 {:.3} ms | p95 {:.3} ms",
                stress.metal_min_ms, stress.metal_p50_ms, stress.metal_p95_ms
            );
            println!(
                "  Sustained AMX + CRC:   512x512 AMX={:.2} GFLOP/s ({:.3} ms) | HW CRC32C={:.2} GB/s",
                stress.amx_sgemm_512_gflops, stress.amx_sgemm_512_ms, stress.hw_crc32c_gbps
            );
        }
        if let Some(deep) = kadmium_core::M4MacMiniLanBridge::run_deep_metal4_suite_on_m4(ip, 10, 256) {
            println!("------------------------------------------------------------------------------------");
            println!(
                "  Metal 4 Tiled 1024x1024 GEMM:  {:.3} ms ({:.2} GFLOP/s | FNV-1a64: {:#018x})",
                deep.tiled_gemm1024_ms, deep.tiled_gemm1024_gflops, deep.tiled_gemm1024_fnv1a64
            );
            println!(
                "  Metal 4 Fused RMSNorm+RoPE:    {:.4} ms (RepOps: {:#018x} | sample: {:?})",
                deep.rmsnorm_rope_ms,
                deep.rmsnorm_rope_repops_digest,
                &deep.rmsnorm_rope_sample8[..4]
            );
            println!(
                "  Metal 4 Flash-GQA Paged (256): {:.4} ms (RepOps: {:#018x} | sample: {:?})",
                deep.flash_gqa_seq256_ms, deep.flash_gqa_repops_digest, deep.flash_gqa_sample4
            );
            println!(
                "  10-Core GCD (4P+6E) Scaling:   1-Core {:.3} ms -> 10-Core {:.3} ms ({:.2}x speedup)",
                deep.gcd_1core_256_ms, deep.gcd_10core_256_ms, deep.gcd_10core_speedup
            );
            println!(
                "  ARMv9.2-A INT8 DotProd & FP16: vdotq_s32={:.2} GOPS | vfmaq_f16={:.2} GFLOP/s",
                deep.armv9_int8_dotprod_gops, deep.armv9_fp16_neon_gflops
            );
            println!(
                "  Zero-Copy mmap .ktb -> Metal:  {:.4} ms (NoCopy UMA verified={} | RepOps: {:#018x})",
                deep.zerocopy_mmap_ktb_ms,
                deep.zerocopy_mmap_verified,
                deep.zerocopy_mmap_repops_digest
            );
        }
    }
    println!("====================================================================================");
}

/// Opens a page in the default browser. Used only when the desktop app is not built.
fn open_in_browser(url: &str) {
    #[cfg(target_os = "windows")]
    {
        // rundll32 hands the URL to the shell without `cmd` re-parsing characters such as `&`.
        let _ = std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

/// Finds the `kadmium-desktop` app: next to this executable first (cargo builds both into
/// `target/<profile>/`), then on `PATH`.
fn find_desktop_app() -> Option<std::path::PathBuf> {
    let name = if cfg!(windows) {
        "kadmium-desktop.exe"
    } else {
        "kadmium-desktop"
    };
    let beside = std::env::current_exe()
        .ok()
        .map(|exe| exe.with_file_name(name))
        .filter(|path| path.is_file());
    beside.or_else(|| {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(name))
                .find(|path| path.is_file())
        })
    })
}

async fn run_desktop(port: u16, live_only: bool, open_control_room: bool, check_only: bool) {
    let local_node = LiveHardwareProfiler::probe_local_host();
    let m4_probe = kadmium_core::M4MacMiniLanBridge::probe_and_verify(None);
    let cm4_probe = kadmium_core::ReTerminalCm4LanBridge::probe_and_verify(None);
    let k230_probe = kadmium_core::K230UsbComputeBridge::probe_attached_k230();

    println!("====================================================================================");
    println!("  KADMIUM (Cd-48) — NATIVE DESKTOP APPLICATION & MULTI-PLATFORM COORDINATOR");
    println!("====================================================================================");
    println!(
        "  Local Workstation:     {} ({} [{:.1} TF FP16 / {:.1} TF FP8 / {:.1} TOPS FP4] + {} [{:.1} TF]{}, {:.1} GB VRAM + {:.1} GB RAM, q*={:.2})",
        local_node.hostname,
        local_node.gpu_name,
        local_node.fp16_tflops,
        local_node.int4_tops * 0.5,
        local_node.int4_tops,
        local_node.cpu_name,
        local_node.cpu_tflops,
        local_node
            .npu_name
            .as_ref()
            .map(|n| format!(" + {} [{:.0} TOPS NPU]", n, local_node.npu_tops))
            .unwrap_or_default(),
        local_node.memory.vram_gb,
        local_node.memory.system_ram_gb,
        local_node.compute_q_star_ratio()
    );
    println!(
        "  Apple M4 LAN Bridge:   {} ({} @ {} | {:.1} TF Metal 4 GPU + {:.0} TOPS 16-Core ANE + AMX {:.1} GFLOP/s)",
        if m4_probe.physical_board_online {
            "ONLINE"
        } else {
            "STANDBY"
        },
        m4_probe.board_model,
        m4_probe.lan_endpoint,
        m4_probe.hardware_node.fp16_tflops,
        m4_probe.hardware_node.npu_tops,
        m4_probe.execution_proof.amx_sgemm_512_gflops
    );
    println!(
        "  RPi CM4 LAN Bridge:    {} ({} @ {})",
        if cm4_probe.physical_board_online {
            "ONLINE"
        } else {
            "STANDBY"
        },
        cm4_probe.board_model,
        cm4_probe.lan_endpoint
    );
    println!(
        "  Kendryte K230 Bridge:  {} ({} [{}])",
        if k230_probe.physical_board_detected {
            "ONLINE"
        } else {
            "STANDBY"
        },
        k230_probe.board_model,
        k230_probe.chip_unique_die_id
    );
    println!(
        "  Silicon Catalog SKUs:  26 platform configurations (7 Apple M4 + 6 CM4 + 5 Tegra X1 + 5 K230 + 3 Phone Farm)"
    );
    let desktop_app = find_desktop_app();
    println!(
        "  Desktop app:           {}",
        desktop_app
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "not built yet (cargo build -p kadmium-desktop)".to_string())
    );
    println!("  Companion:             http://127.0.0.1:{port}/companion");
    println!("  Control Room:          http://127.0.0.1:{port}/");
    println!("====================================================================================");

    if check_only {
        return;
    }

    // The desktop app opens the Companion in its own window and runs the coordinator itself
    // (or attaches to one already on the port), so closing the window ends this command too.
    match desktop_app {
        Some(app) => {
            let mut command = tokio::process::Command::new(&app);
            command.arg("--port").arg(port.to_string());
            if !live_only {
                command.arg("--demo");
            }
            if open_control_room {
                command.arg("--control-room");
            }
            match command.status().await {
                Ok(status) if status.success() => return,
                Ok(status) => std::process::exit(status.code().unwrap_or(1)),
                Err(err) => eprintln!(
                    "kadmium: could not start {} ({err}); opening the Companion in your browser.",
                    app.display()
                ),
            }
        }
        None => eprintln!(
            "kadmium: the desktop app is not built yet, so the Companion opens in your browser.\n\
             Build it once with `cargo build -p kadmium-desktop` (add `--release` for a build \
             without a console window), then run `kadmium desktop` again."
        ),
    }

    let companion_url = format!("http://127.0.0.1:{port}/companion");
    let control_room_url = format!("http://127.0.0.1:{port}/");
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(550)).await;
        open_in_browser(&companion_url);
        if open_control_room {
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            open_in_browser(&control_room_url);
        }
    });

    run_server(port, None, live_only, DEFAULT_WORKER_TCP_PORT, false).await;
}
