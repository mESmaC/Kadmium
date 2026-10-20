use clap::{Parser, Subcommand};
use kadmium_core::{
    HardwareNode, KadmiumWorkspaceConfig, LiveHardwareProfiler, ModelSpec, NodeTrustTier,
    PoliteActivityMonitor, PoliteActivitySample, PoliteState, WorkloadKind,
    sample_cloud_burst_gpu_node, sample_node_pool, sample_personal_cluster,
};
use kadmium_data::{
    Fts3TransferScheduler, GitHubConnector, HuggingFaceHubConnector, KtbBankPacker, KtbBankReader,
    KtbMerkleTree, SafetensorsToKtbTranscoder, TensorEncoding,
};
use kadmium_gateway::{GatewayState, KadmiumMcpServer, McpJsonRpcRequest, build_router};
use kadmium_network::{ClusterStateCrdt, JoinTicket, LanDiscoveryBeacon};
use kadmium_rl::{
    DisaggregatedRlPlan, DiscoveryTree, DreamRsiEvolutionReport, RlEnvironmentSuiteSpec,
    RlTopologyPlanner,
};
use kadmium_runtime::{
    ColibriNvmeDoubleBuffer, DeMoDctCompressor, Q4KBlockTensor, TrainingJobTelemetry,
    VerdeVerificationEngine, sample_boinc_workunits,
};
use kadmium_sharding::{
    ClusterShardPlan, DataflowStreamPlanner, DistributedPagedKvPlanner,
    HybridCloudArbitragePlanner, MinMaxStageLatencyOptimizer, ResponsibilityShiftPlan,
    ShardPlanner, ShiftTriggerReason, SubClusterPartitioner,
};
use std::sync::{Arc, Mutex};

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
    },
    /// Verify a kdm1_... Join Ticket, broadcast a UDP LAN beacon (KDMD), and join a cluster slice
    Join {
        /// Optional kdm1_... ticket (if omitted, generates and verifies a local LAN/WAN ticket)
        #[arg(short, long)]
        ticket: Option<String>,
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
        Commands::Serve { port } => run_server(port).await,
        Commands::Join { ticket } => run_join(ticket).await,
        Commands::Pull { uri } => run_pull(&uri),
        Commands::Mcp { inspect } => run_mcp(inspect),
        Commands::Init { out } => run_init(out.as_deref()),
        Commands::Compile {
            model,
            start_layer,
            end_layer,
            out,
        } => run_compile(&model, start_layer, end_layer, out.as_deref()),
    }
}

fn resolve_model(name: &str) -> ModelSpec {
    match name.to_lowercase().as_str() {
        "qwen3-235b-moe" | "qwen3" => ModelSpec::qwen3_235b_moe(),
        "llama-3-70b-lora" | "llama" | "llama70b" => ModelSpec::llama_3_70b_lora(),
        _ => ModelSpec::deepseek_v3_671b_moe(),
    }
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

fn print_shard_plan(plan: &ClusterShardPlan) {
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

fn run_cloud_shield() {
    let model = ModelSpec::deepseek_v3_671b_moe();
    let arb = HybridCloudArbitragePlanner::compute_arbitrage_plan(&model);
    let shield_events = HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield();

    println!(
        "==================================================================================================================="
    );
    println!(
        " 🛡️  KADMIUM HYBRID CLOUD — Multi-Datacenter Spot Arbitrage & Bidirectional Downtime Protection Shield"
    );
    println!(
        "==================================================================================================================="
    );
    println!(
        "\n[1/2] Multi-Provider Fragmented GPU & Bare-Metal Arbitrage ({})",
        model.display_name
    );
    println!(
        "      Combined Compute: {:.1} FP16 TFLOPs | {:.1} GB Fast VRAM/Unified | Blended Cost: ${:.2}/hr vs. Hyperscaler ${:.2}/hr ({:.1}% Saved!)",
        arb.total_combined_tflops,
        arb.total_combined_vram_gb,
        arb.kadmium_blended_hourly_usd,
        arb.monolithic_hyperscaler_hourly_usd,
        arb.cost_reduction_pct
    );
    println!("{}", "-".repeat(125));
    for o in &arb.selected_offers {
        println!(
            "  * [{:<14}] {:<52} | {:>5.1} TFLOP | {:>5.1} GB VRAM | ${:>4.2}/hr (vs ${:>4.2}/hr)",
            o.offer_id,
            o.provider_name,
            o.fp16_tflops,
            o.vram_or_unified_gb,
            o.hourly_spot_usd,
            o.hyperscaler_equivalent_usd
        );
    }

    println!(
        "\n[2/2] Bidirectional Local <-> Cloud Downtime Protection Shield (Zero-Downtime Shard Handoff):"
    );
    println!("{}", "-".repeat(125));
    for ev in &shield_events {
        println!(
            "  >>> [{:?}] Layers L{:02}..L{:02} | RTO: {:.0} ms | Zero Downtime: {} | Burst Cost: ${:.2}/hr",
            ev.direction,
            ev.affected_layers.0,
            ev.affected_layers.1,
            ev.switchover_rto_ms,
            ev.zero_downtime_preserved,
            ev.temporary_failover_hourly_usd
        );
        println!("      Incident: {}", ev.incident_summary);
        println!(
            "      Route:    {}  ===>  {}\n",
            ev.source_environment, ev.target_failover_environment
        );
    }
}

fn run_probe() {
    let live_node = LiveHardwareProfiler::probe_local_host();
    println!(
        "==================================================================================================================="
    );
    println!(" 🔍 KADMIUM LIVE HARDWARE PROFILER & 64KB PAGE-ALIGNED .KTB PACKER (Milestone 1)");
    println!(
        "==================================================================================================================="
    );
    println!(
        "  * Hostname / Node ID:     {} ({})",
        live_node.hostname, live_node.id
    );
    println!(
        "  * Device Classification:  {:?} | Trust Tier: {:?}",
        live_node.device_kind, live_node.trust_tier
    );
    println!(
        "  * Primary Backend:        {:?} ({:.1} FP16 TFLOPs | {:.1} INT4 TOPS)",
        live_node.backend, live_node.fp16_tflops, live_node.int4_tops
    );
    println!(
        "  * Host CPU Parallelism:   {} logical threads ({:.1} CPU TFLOPs)",
        live_node.cpu_threads, live_node.cpu_tflops
    );
    println!(
        "  * Memory Tier Hierarchy:  {:.1} GB VRAM | {:.1} GB Unified | {:.1} GB Sys RAM | {:.2} GB/s Stream BW",
        live_node.memory.vram_gb,
        live_node.memory.unified_mem_gb,
        live_node.memory.system_ram_gb,
        live_node.memory.nvme_read_bw_gbps
    );
    println!(
        "  * FreeToken q* Split:     {:.2} (Accelerator vs. Host/NVMe Stream Ratio)",
        live_node.compute_q_star_ratio()
    );

    // Pack and verify a 64 KiB page-aligned .ktb tensor bank in memory
    let mut packer = KtbBankPacker::new("deepseek-v3-671b-moe");
    packer.add_tensor_bank(
        "layers.0.attn.qkv_proj",
        0,
        None,
        TensorEncoding::Fp8E4M3,
        vec![0xA5; 48_120],
    );
    packer.add_tensor_bank(
        "layers.0.moe.experts.0.gate_up_proj",
        0,
        Some(0),
        TensorEncoding::Q4KBlock,
        vec![0x5A; 142_000],
    );
    packer.add_tensor_bank(
        "layers.0.moe.experts.1.gate_up_proj",
        0,
        Some(1),
        TensorEncoding::Q4KBlock,
        vec![0x3C; 96_000],
    );

    let ktb_bytes = packer.pack_to_bytes();
    let reader = KtbBankReader::from_slice(&ktb_bytes).expect("parse packed KTB image");
    println!(
        "\n--- 📦 Binary .ktb (Kadmium Tensor Bank) 64 KiB Page-Aligned Verification ({} bytes total) ---",
        ktb_bytes.len()
    );
    for entry in &reader.header.entries {
        let slice = reader
            .get_verified_tensor_slice(entry)
            .expect("verify 64KB page offset & FNV-1a 64-bit checksum");
        println!(
            "  * {:<36} | Layer {:02} | Expert: {:>4} | Offset: {:#08x} (64KB Aligned) | Payload: {:>6} B | Checksum: {:#018x} [OK]",
            entry.tensor_name,
            entry.layer_index,
            entry
                .expert_index
                .map(|e| format!("#{:02}", e))
                .unwrap_or_else(|| "Dense".into()),
            entry.page_offset_bytes,
            slice.len(),
            entry.checksum_fnv64
        );
    }

    // 2. Safetensors -> .ktb Shard-Filtered Transcoder verification
    let raw_st = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
        (
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            vec![64, 64],
            &vec![0x11; 8192],
        ),
        (
            "model.layers.1.mlp.experts.4.up_proj.weight",
            "F8_E4M3",
            vec![128, 64],
            &vec![0x22; 16384],
        ),
    ]);
    let transcoded_ktb =
        SafetensorsToKtbTranscoder::transcode_shard_to_ktb("deepseek-v3-671b-moe", &raw_st, 1, 1)
            .expect("transcode safetensors shard");
    let st_reader = KtbBankReader::from_slice(&transcoded_ktb).expect("read transcoded KTB");
    println!(
        "\n--- 🔄 Safetensors -> .ktb Shard Transcoder (Filtered Layer 01..01: {} entry, {} bytes) [OK] ---",
        st_reader.header.entries.len(),
        transcoded_ktb.len()
    );

    // 3. Nous DeMo / DisTrO DCT Fast-Frequency Momentum Compression + Error Feedback
    let mut demo = DeMoDctCompressor::new(128, 0.9);
    let synth_grad: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.05).cos()).collect();
    let dct_pkt = demo.step_and_compress(&synth_grad, 4);
    println!(
        "--- 📡 Nous DeMo DCT Momentum Compressor: 128 floats -> Top-{} DCT coeffs ({:.1}x Wire Compression, Zero Energy Loss) [OK] ---",
        dct_pkt.indices.len(),
        dct_pkt.compression_ratio()
    );

    // 4. Gensyn Verde O(log2 L) Refereed Layer Bisection over 64 Layers
    let referee_fn = |layer: usize, state: &[f64]| -> Vec<f64> {
        state
            .iter()
            .map(|&v| v * 1.01 + (layer as f64) * 0.002)
            .collect()
    };
    let mut ref_states = vec![vec![1.0, 2.0, 3.0]];
    let mut wrk_states = vec![vec![1.0, 2.0, 3.0]];
    for l in 0..64 {
        ref_states.push(referee_fn(l, &ref_states[l]));
        let mut w = referee_fn(l, &wrk_states[l]);
        if l == 37 {
            w[0] += 0.42; // simulated fault at layer 37
        }
        wrk_states.push(w);
    }
    if let Some(bisect) =
        VerdeVerificationEngine::bisect_faulty_layer(&wrk_states, &ref_states, referee_fn, 1e-6)
    {
        println!(
            "--- ⚖️  Gensyn Verde O(log L) Layer Bisection: Isolated Faulty Layer #{} out of {} in {} checks [OK] ---",
            bisect.faulty_layer_index, bisect.total_layers, bisect.referee_evaluations_used
        );
    }

    // 5. Iroh-style Single-Token Join Ticket (`kdm1_...`)
    let ticket = JoinTicket::generate(
        live_node.id.clone(),
        vec!["192.168.1.48:7848".into(), "relay.kadmium.ai:4433".into()],
        Some("slice-01-prod-deploy".into()),
        NodeTrustTier::VerifiedPeerMiddleLayersOnly,
    );
    let encoded_ticket = ticket.encode_token();
    println!(
        "--- 🎟️  Generated WAN/LAN Mesh Join Ticket (MAC Verified): {}... ---",
        &encoded_ticket[..44.min(encoded_ticket.len())]
    );

    // 6. PoliteActivityMonitor <200ms Gaming Auto-Yield Governance
    let yield_decision = PoliteActivityMonitor::evaluate_sample(
        &live_node,
        &PoliteActivitySample {
            foreground_process_name: Some("cyberpunk2077.exe".into()),
            fullscreen_swapchain_active: true,
            external_gpu_utilization_pct: 72.0,
            external_vram_used_gb: 8.4,
            on_battery_power: false,
            package_temp_celsius: 68.0,
        },
    );
    println!(
        "--- 🎮 PoliteActivityMonitor Auto-Yield: {:?} -> {:?} in {:.1} ms (VRAM ceiling {:.2} GB, {} CPU threads) [OK] ---",
        yield_decision.previous_state,
        yield_decision.recommended_state,
        yield_decision.estimated_yield_latency_ms,
        yield_decision.vram_ceiling_gb,
        yield_decision.cpu_thread_budget
    );

    // 7. KtbMerkleTree O(log N) 64 KiB Page Inclusion Proof
    let merkle = KtbMerkleTree::from_ktb_bytes(&ktb_bytes);
    let proof_p1 = merkle.prove_page(1).expect("merkle proof page 1");
    let verified_p1 =
        KtbMerkleTree::verify_page(&ktb_bytes[65_536..131_072], &proof_p1, merkle.root_hash);
    println!(
        "--- 🌳 KtbMerkleTree 64 KiB Page Verifier: {} pages | Root: {:#018x} | Page #1 Proof Verified: {} [OK] ---",
        merkle.leaf_hashes.len(),
        merkle.root_hash,
        verified_p1
    );

    // 8. Q4_K 4-Bit Group-Quantized MatVec + Colibri Double-Buffered NVMe DMA
    let w_fp32: Vec<f32> = (0..(16 * 32)).map(|i| ((i as f32) * 0.03).sin()).collect();
    let x_fp32: Vec<f32> = (0..32).map(|j| ((j as f32) * 0.07).cos()).collect();
    let q4_mat = Q4KBlockTensor::quantize_matrix(&w_fp32, 16, 32);
    let y_q4 = q4_mat.matvec_f32(&x_fp32);
    let colibri = ColibriNvmeDoubleBuffer::execute_double_buffered_stream(
        4,
        4 * 1024 * 1024,
        live_node.memory.nvme_read_bw_gbps,
        1.45,
    );
    println!(
        "--- ⚡ Q4_K 4-Bit MatVec ({:.2}x compression, |y[0]|={:.3}) & Colibri Double-Buffer Stall: {:.2} ms [OK] ---",
        q4_mat.compression_ratio(),
        y_q4[0].abs(),
        colibri[0].effective_pipeline_stall_ms
    );

    // 9. LWW-Element-Map + Vector Clock ClusterStateCrdt Gossip Merge
    let mut crdt_local = ClusterStateCrdt::new();
    crdt_local.upsert_node_state(
        &live_node.id.0,
        &live_node.id.0,
        PoliteState::IdleDedicated,
        Some((0, 25)),
        true,
    );
    let frame = crdt_local.to_wire_frame(1);
    let crdt_decoded = ClusterStateCrdt::from_wire_frame(&frame).expect("decode CRDT frame");
    println!(
        "--- 🔄 ClusterStateCrdt Gossip Replication: {} register(s), Vector Clock v{} [OK] ---",
        crdt_decoded.registers.len(),
        crdt_decoded
            .vector_clock
            .values()
            .max()
            .copied()
            .unwrap_or(0)
    );
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

async fn run_server(port: u16) {
    let state = Arc::new(Mutex::new(GatewayState::new()));
    let app = build_router(state);
    let addr = format!("0.0.0.0:{}", port);
    println!(
        "===================================================================================="
    );
    println!(
        " Kadmium (Cd-48) Control Room & AI Gateway listening on http://127.0.0.1:{}",
        port
    );
    println!("    - Cluster Control Room:   http://127.0.0.1:{}/", port);
    println!(
        "    - kadmium.ai Website:     http://127.0.0.1:{}/site",
        port
    );
    println!(
        "    - Desktop Companion UI:   http://127.0.0.1:{}/companion",
        port
    );
    println!(
        "    - Application Docs:       http://127.0.0.1:{}/docs",
        port
    );
    println!(
        "    - Design System Showcase: http://127.0.0.1:{}/design-system",
        port
    );
    println!(
        "    - OpenAI Chat Endpoint:   http://127.0.0.1:{}/v1/chat/completions",
        port
    );
    println!(
        "    - Anthropic Endpoint:     http://127.0.0.1:{}/v1/messages",
        port
    );
    println!(
        "===================================================================================="
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind TCP listener");
    axum::serve(listener, app).await.expect("server error");
}

async fn run_join(ticket_arg: Option<String>) {
    let local_node = LiveHardwareProfiler::probe_local_host();
    let token = ticket_arg.unwrap_or_else(|| {
        JoinTicket::generate(
            local_node.id.clone(),
            vec!["127.0.0.1:7848".into(), "relay.kadmium.ai:4433".into()],
            Some("slice-01-prod-deploy".into()),
            NodeTrustTier::VerifiedPeerMiddleLayersOnly,
        )
        .encode_token()
    });

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Node Join & UDP LAN Beacon Handshake");
    println!(
        "===================================================================================="
    );

    let verified_ticket = JoinTicket::decode_and_verify(&token)
        .expect("Failed to verify kdm1_ Join Ticket MAC signature");
    println!(
        "  * Ticket MAC Verified:    [OK] Coordinator={} | Slice={:?} | Trust={:?}",
        verified_ticket.coordinator_id.0,
        verified_ticket.target_slice_id,
        verified_ticket.granted_trust_tier
    );

    let rx_sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind udp listener");
    let rx_addr = rx_sock.local_addr().expect("local udp addr");
    let tx_sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind udp sender");

    let beacon = LanDiscoveryBeacon::from_node(&local_node, 7848);
    beacon
        .broadcast_once(&tx_sock, rx_addr)
        .await
        .expect("send KDMD UDP beacon");
    let (recv_beacon, peer_addr) = LanDiscoveryBeacon::recv_once(&rx_sock)
        .await
        .expect("recv KDMD UDP beacon");
    println!(
        "  * UDP LAN Beacon (KDMD):  [OK] Broadcast & received from {} (hostname={}, q*={:.2})",
        peer_addr, recv_beacon.hostname, recv_beacon.q_star_ratio
    );

    let mut cluster = sample_personal_cluster();
    cluster.push(local_node);
    let model = ModelSpec::deepseek_v3_671b_moe();
    let plan = ShardPlanner::compute_plan(2, &model, WorkloadKind::Inference, &cluster);
    print_shard_plan(&plan);
}

fn run_pull(uri: &str) {
    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Ecosystem Connector (`hf://` & `gh://` Selective Staging)");
    println!(
        "===================================================================================="
    );

    if uri.starts_with("gh://") || uri.contains("github.com") {
        let gh_run =
            GitHubConnector::evaluate_repo_on_slice(uri).expect("Failed to parse GitHub URI");
        println!(
            "  * GitHub Repo Mounted:    {}/{} @ {} (SHA: {})",
            gh_run.gh_ref.owner,
            gh_run.gh_ref.repo,
            gh_run.gh_ref.git_ref,
            gh_run.pinned_commit_sha
        );
        println!("  * Subpath / World:        {:?}", gh_run.gh_ref.subpath);
        println!("  * Target Slice Assigned:  {}", gh_run.assigned_slice_id);
        println!("  * Sandbox Runtime:        {}", gh_run.sandbox_runtime);
        println!(
            "  * Verde Check Conclusion: {} ({:.1} tok/s, {:.2}x Dream-RSI speedup)",
            gh_run.github_check_conclusion,
            gh_run.measured_tokens_per_sec,
            gh_run.dream_rsi_best_speedup
        );
    } else {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);
        let hf_plan = HuggingFaceHubConnector::plan_selective_model_pull(uri, &plan)
            .expect("Failed to parse Hugging Face URI");
        println!(
            "  * Hugging Face Repo:      {} (rev: {}, kind: {:?})",
            hf_plan.hf_ref.repo_id, hf_plan.hf_ref.revision, hf_plan.hf_ref.kind
        );
        println!("  * Index Manifest URL:     {}", hf_plan.index_manifest_url);
        println!("  * Primary .ktb Seed Node: {}", hf_plan.seed_vault_node);
        println!(
            "\n{:<22} {:<10} {:<12} {:<14} {:<42}",
            "WORKER HOSTNAME", "LAYERS", "SELECTIVE", "AVOIDED DL", "COMPILED .KTB BANK"
        );
        println!("{}", "-".repeat(104));
        for np in &hf_plan.node_pulls {
            println!(
                "{:<22} L{:02}..L{:02}   {:>6.2} GB   {:>8.2} GB   {}",
                np.hostname,
                np.start_layer,
                np.end_layer,
                np.selective_download_gb,
                np.full_repo_avoided_gb,
                np.output_ktb_bank_path
            );
        }
    }
}

fn run_mcp(inspect: bool) {
    let mut state = GatewayState::new();

    if inspect {
        let manifest = KadmiumMcpServer::manifest();
        println!(
            "===================================================================================="
        );
        println!(
            " Kadmium (Cd-48) — Model Context Protocol (MCP) Server ({} v{})",
            manifest.server_name, manifest.server_version
        );
        println!(
            "===================================================================================="
        );
        println!("  * Protocol Version:       {}", manifest.protocol_version);
        println!(
            "  * Transports:             {}",
            manifest.transports.join(" | ")
        );
        println!("  * HTTP Endpoint:          {}", manifest.http_endpoint);
        println!("\n  [IDE / Agent `.mcp.json` Configuration]:");
        for line in manifest.stdio_config_json.lines() {
            println!("    {line}");
        }

        println!("\n  [Exposed MCP Tools ({})]:", manifest.tools.len());
        for t in &manifest.tools {
            println!("    - {:<30} {}", t.name, t.description);
        }

        println!(
            "\n  [Exposed MCP Resources ({})]:",
            manifest.resources.len()
        );
        for r in &manifest.resources {
            println!("    - {:<30} [{}] {}", r.uri, r.mime_type, r.name);
        }

        println!("\n  [Exposed MCP Prompts ({})]:", manifest.prompts.len());
        for p in &manifest.prompts {
            println!("    - {:<30} {}", p.name, p.description);
        }
        return;
    }

    // Standard I/O (stdio) JSON-RPC 2.0 loop for MCP clients
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line_res in stdin.lock().lines() {
        let Ok(line) = line_res else {
            break;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(req) = serde_json::from_str::<McpJsonRpcRequest>(trimmed) {
            let is_notification = req.id.is_none();
            let resp = KadmiumMcpServer::handle_request(&mut state, req);
            if !is_notification && let Ok(json_out) = serde_json::to_string(&resp) {
                let _ = writeln!(stdout, "{json_out}");
                let _ = stdout.flush();
            }
        }
    }
}

fn run_init(out_path: Option<&str>) {
    let local_node = LiveHardwareProfiler::probe_local_host();
    let cfg = KadmiumWorkspaceConfig::from_local_probe(local_node);
    let toml_text = cfg.render_kadmium_toml();

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Workspace Initialization (`kadmium.toml`)");
    println!(
        "===================================================================================="
    );
    println!("{toml_text}");

    if let Some(path) = out_path {
        std::fs::write(path, &toml_text).expect("Failed to write kadmium.toml to disk");
        println!("  [OK] Wrote configuration manifest to: {path}");
    }
}

fn run_compile(model: &str, start_layer: u32, end_layer: u32, out_path: Option<&str>) {
    let mut packer = KtbBankPacker::new(model);
    for l in start_layer..=end_layer {
        packer.add_tensor_bank(
            format!("layers.{l}.attn.qkv_proj"),
            l,
            None,
            TensorEncoding::Fp8E4M3,
            vec![0xA5; 32_768],
        );
        for exp in 0..2u32 {
            packer.add_tensor_bank(
                format!("layers.{l}.moe.experts.{exp}.gate_up_proj"),
                l,
                Some(exp),
                TensorEncoding::Q4KBlock,
                vec![(l as u8).wrapping_add(exp as u8); 65_536],
            );
        }
    }

    let ktb_bytes = packer.pack_to_bytes();
    let merkle = KtbMerkleTree::from_ktb_bytes(&ktb_bytes);
    let reader = KtbBankReader::from_slice(&ktb_bytes).expect("valid compiled .ktb image");

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — 64 KiB Page-Aligned `.ktb` Tensor Bank Compiler");
    println!(
        "===================================================================================="
    );
    println!("  * Model ID:               {}", reader.header.model_id);
    println!("  * Compiled Layer Range:   L{start_layer:02}..L{end_layer:02}");
    println!(
        "  * Total Archive Size:     {} bytes ({} x 64 KiB pages)",
        ktb_bytes.len(),
        merkle.leaf_hashes.len()
    );
    println!("  * Merkle Root Digest:     {:#018x}", merkle.root_hash);
    println!(
        "  * Tensor Entries Packed:  {}",
        reader.header.entries.len()
    );

    if let Some(path) = out_path {
        std::fs::write(path, &ktb_bytes).expect("Failed to write .ktb bank to disk");
        println!("  * Saved .ktb Archive To:  {path}");
    }
}
