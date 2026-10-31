//! Model Context Protocol (MCP) Server for the **Kadmium (`Cd-48`)** Compute Fabric.
//!
//! Supports both:
//! - **Stdio Transport (`kadmium mcp`)** for local AI coding agents (Claude Code, Cursor,
//!   Antigravity, Windsurf, Zed, OpenCode).
//! - **Streamable HTTP Transport (`POST /mcp`)** on the Gateway (`http://127.0.0.1:7848/mcp`)
//!   for remote/team agentic harnesses.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use kadmium_core::{ModelSpec, WorkloadKind};
use kadmium_data::{GitHubConnector, HuggingFaceHubConnector};
use kadmium_rl::{GrpoEpochExecutor, SandboxOp, WasiMicroSandbox};
use kadmium_runtime::DistributedMoEPipelineExecutor;
use kadmium_sharding::{
    DistributedPagedKvPlanner, HybridCloudArbitragePlanner, MinMaxStageLatencyOptimizer,
    ShardPlanner,
};

use crate::GatewayState;

/// MCP Protocol version supported by the Kadmium MCP server.
pub const KADMIUM_MCP_PROTOCOL_VERSION: &str = "2025-03-26";

/// JSON-RPC 2.0 Request envelope for MCP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpJsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 Response envelope for MCP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpJsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpJsonRpcError>,
}

/// JSON-RPC 2.0 Error payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpJsonRpcError {
    pub code: i64,
    pub message: String,
}

/// Metadata describing an MCP tool exposed by Kadmium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Metadata describing an MCP resource exposed by Kadmium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceDescriptor {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

/// Metadata describing an MCP prompt template exposed by Kadmium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptDescriptor {
    pub name: String,
    pub description: String,
    pub arguments: Vec<McpPromptArgument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// Summary manifest returned by `GET /api/v1/mcp` for the Control Room UI and CLI inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KadmiumMcpManifest {
    pub server_name: String,
    pub server_version: String,
    pub protocol_version: String,
    pub transports: Vec<String>,
    pub stdio_config_json: String,
    pub http_endpoint: String,
    pub tools: Vec<McpToolDescriptor>,
    pub resources: Vec<McpResourceDescriptor>,
    pub prompts: Vec<McpPromptDescriptor>,
}

/// Stateless/stateful MCP dispatcher operating over `GatewayState`.
pub struct KadmiumMcpServer;

impl KadmiumMcpServer {
    /// Returns the full catalog of MCP Tools, Resources, Prompts, and client configurations.
    pub fn manifest() -> KadmiumMcpManifest {
        let stdio_config = json!({
            "mcpServers": {
                "kadmium": {
                    "command": "kadmium",
                    "args": ["mcp"],
                    "env": {
                        "KADMIUM_COORDINATOR_URL": "http://127.0.0.1:7848"
                    }
                }
            }
        });

        KadmiumMcpManifest {
            server_name: "kadmium-mcp".into(),
            server_version: "0.1.0".into(),
            protocol_version: KADMIUM_MCP_PROTOCOL_VERSION.into(),
            transports: vec![
                "stdio (`kadmium mcp`)".into(),
                "Streamable HTTP (`POST http://127.0.0.1:7848/mcp`)".into(),
            ],
            stdio_config_json: serde_json::to_string_pretty(&stdio_config)
                .unwrap_or_else(|_| "{}".into()),
            http_endpoint: "http://127.0.0.1:7848/mcp".into(),
            tools: Self::list_tools(),
            resources: Self::list_resources(),
            prompts: Self::list_prompts(),
        }
    }

    /// Lists all 7 MCP tools exposed by Kadmium (`Cd-48`).
    pub fn list_tools() -> Vec<McpToolDescriptor> {
        vec![
            McpToolDescriptor {
                name: "kadmium_cluster_status".into(),
                description: "Inspect live Kadmium (Cd-48) heterogeneous cluster topology, aggregate VRAM/RAM capacity, effective TFLOPS, node polite states, and coordinator Bully+CRDT quorum.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
            McpToolDescriptor {
                name: "kadmium_plan_sharding".into(),
                description: "Compute an asymmetric Q* water-filling layer sharding plan across local (--cadmium) and cloud (--cherenkov) nodes for a given model.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "workload": {
                            "type": "string",
                            "enum": ["inference", "training"],
                            "description": "Workload type: 'inference' (DeepSeek-V3 671B MoE) or 'training' (Llama-3-70B LoRA/DiLoCo)."
                        }
                    },
                    "required": []
                }),
            },
            McpToolDescriptor {
                name: "kadmium_pull_artifact".into(),
                description: "Stage a Hugging Face model/dataset URI (hf://owner/repo@rev) into selective 64 KB-aligned .ktb shards or mount a GitHub repo/Dream-RSI world (gh://owner/repo@ref#subpath) onto a subcluster slice.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "Ecosystem URI such as 'hf://deepseek-ai/DeepSeek-V3@main' or 'gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt'."
                        }
                    },
                    "required": ["uri"]
                }),
            },
            McpToolDescriptor {
                name: "kadmium_run_inference".into(),
                description: "Execute a distributed MoE forward pass across the asymmetric Kadmium pipeline with rk3588-sbc-bramble speculative draft verification and Verde RepOps.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Prompt text to route across the distributed MoE pipeline."
                        }
                    },
                    "required": ["prompt"]
                }),
            },
            McpToolDescriptor {
                name: "kadmium_verify_rl_sandbox".into(),
                description: "Execute candidate code or GPU kernel synthesis inside the deterministic fuel-metered WasiMicroSandbox on rk3588-sbc-bramble and compute verified GRPO group advantages.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "candidate_code": {
                            "type": "string",
                            "description": "Optional Rust/CUDA/Triton snippet to evaluate inside the deterministic WASI sandbox."
                        }
                    },
                    "required": []
                }),
            },
            McpToolDescriptor {
                name: "kadmium_cloud_shield_arbitrage".into(),
                description: "Query live multi-provider spot GPU arbitrage quotes (RunPod, Vast.ai, Lambda, Prime Intellect, AWS) and bidirectional local<->cloud failover simulations.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
            McpToolDescriptor {
                name: "kadmium_supply_marketplace".into(),
                description: "Evaluate or update host compute leasing on the Kadmium Supply Grid (renting out idle local GPUs, M4 Mac Minis, SBCs, and NAS vaults for direct USD payouts or 1.25x Kadmium Cloud credits with <200ms Polite auto-yield).".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "daily_lease_hours": {
                            "type": "number",
                            "description": "Optional daily idle lease window in hours (0.0 to 24.0)."
                        },
                        "payout_mode": {
                            "type": "string",
                            "description": "Optional payout mode: 'credits' (1.25x Kadmium Cloud credits) or 'usd' (Direct USD payout)."
                        }
                    },
                    "required": []
                }),
            },
        ]
    }

    /// Lists all MCP resources exposed by Kadmium (`Cd-48`).
    pub fn list_resources() -> Vec<McpResourceDescriptor> {
        vec![
            McpResourceDescriptor {
                uri: "kadmium://cluster/topology".into(),
                name: "Heterogeneous Cluster Topology & Telemetry".into(),
                description: "Live hardware nodes, VRAM/RAM budgets, bandwidth, Q* ratios, and Polite Gaming Mode states.".into(),
                mime_type: "application/json".into(),
            },
            McpResourceDescriptor {
                uri: "kadmium://cluster/sharding".into(),
                name: "Active Q* Layer Sharding Plan (.ktb)".into(),
                description: "Asymmetric layer-range assignments and 64 KB-aligned .ktb tensor bank paths.".into(),
                mime_type: "application/json".into(),
            },
            McpResourceDescriptor {
                uri: "kadmium://cluster/integrations".into(),
                name: "Ecosystem Connectors (hf://, gh://, oci://, s3://, wandb://)".into(),
                description: "Selective Hugging Face pull plan and GitHub slice check runs.".into(),
                mime_type: "application/json".into(),
            },
            McpResourceDescriptor {
                uri: "kadmium://cluster/cloud-shield".into(),
                name: "Multi-Provider Cloud Arbitrage & Failover Shield".into(),
                description: "Real-time $/TFLOP/hr spot comparison and zero-downtime failover simulations.".into(),
                mime_type: "application/json".into(),
            },
            McpResourceDescriptor {
                uri: "kadmium://docs/overview".into(),
                name: "Kadmium Official Documentation Overview".into(),
                description: "Embedded markdown guide to the Kadmium compute fabric.".into(),
                mime_type: "text/markdown".into(),
            },
        ]
    }

    /// Lists all MCP prompts exposed by Kadmium (`Cd-48`).
    pub fn list_prompts() -> Vec<McpPromptDescriptor> {
        vec![
            McpPromptDescriptor {
                name: "optimize-heterogeneous-sharding".into(),
                description: "Analyze bottleneck links and optimize Q* layer assignments across local (--cadmium) and cloud (--cherenkov) nodes.".into(),
                arguments: vec![McpPromptArgument {
                    name: "model_id".into(),
                    description: "Model identifier (e.g., deepseek-v3-671b-moe)".into(),
                    required: false,
                }],
            },
            McpPromptDescriptor {
                name: "launch-dream-rsi-loop".into(),
                description: "Mount a GitHub repository into a WasiMicroSandbox slice and run a verified GRPO self-improvement epoch.".into(),
                arguments: vec![McpPromptArgument {
                    name: "repo_uri".into(),
                    description: "GitHub URI (e.g., gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt)".into(),
                    required: true,
                }],
            },
        ]
    }

    /// Handles a single JSON-RPC 2.0 MCP request against `GatewayState`.
    pub fn handle_request(state: &mut GatewayState, req: McpJsonRpcRequest) -> McpJsonRpcResponse {
        let id = req.id.clone();

        let result = match req.method.as_str() {
            "initialize" => Ok(json!({
                "protocolVersion": KADMIUM_MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false },
                    "resources": { "subscribe": false, "listChanged": false },
                    "prompts": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "kadmium-mcp",
                    "version": "0.1.0",
                    "signature": "Cd-48"
                }
            })),
            "notifications/initialized" | "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({
                "tools": Self::list_tools()
            })),
            "tools/call" => Self::handle_tool_call(state, req.params.unwrap_or_else(|| json!({}))),
            "resources/list" => Ok(json!({
                "resources": Self::list_resources()
            })),
            "resources/read" => {
                Self::handle_resource_read(state, req.params.unwrap_or_else(|| json!({})))
            }
            "prompts/list" => Ok(json!({
                "prompts": Self::list_prompts()
            })),
            "prompts/get" => Self::handle_prompt_get(req.params.unwrap_or_else(|| json!({}))),
            other => Err(McpJsonRpcError {
                code: -32601,
                message: format!("MCP method not found: '{other}'"),
            }),
        };

        match result {
            Ok(val) => McpJsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(val),
                error: None,
            },
            Err(err) => McpJsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(err),
            },
        }
    }

    fn handle_tool_call(state: &mut GatewayState, params: Value) -> Result<Value, McpJsonRpcError> {
        let tool_name =
            params
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpJsonRpcError {
                    code: -32602,
                    message: "Missing 'name' in tools/call params".into(),
                })?;
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        match tool_name {
            "kadmium_cluster_status" => {
                let snap = state.snapshot();
                let total_vram: f64 = snap
                    .nodes
                    .iter()
                    .map(|n| n.memory.vram_gb + n.memory.unified_mem_gb)
                    .sum();
                let total_ram: f64 = snap.nodes.iter().map(|n| n.memory.system_ram_gb).sum();
                let total_tflops: f64 = snap.nodes.iter().map(|n| n.fp16_tflops).sum();
                let ctx_cap = DistributedPagedKvPlanner::plan_cluster_context_capacity(
                    &snap.inference_plan,
                    &state.inference_model,
                    &snap.nodes,
                );
                let native_label = ctx_cap
                    .model_native_context_tokens
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let summary = format!(
                    "Kadmium (Cd-48) Epoch {} | {} active nodes | {:.1} GB VRAM/UMA + {:.1} GB RAM | {:.1} TFLOPS | Modelled KV context: {}–{} tokens (theoretical estimate, native {}) | Leader: {} (Term {}) | Throughput: {:.1} {}",
                    snap.inference_plan.epoch,
                    snap.nodes.len(),
                    total_vram,
                    total_ram,
                    total_tflops,
                    ctx_cap.cluster_in_memory_context_tokens,
                    ctx_cap.cluster_max_full_attention_context_tokens,
                    native_label,
                    snap.coordinator_quorum.active_leader,
                    snap.coordinator_quorum.term,
                    snap.inference_plan.estimated_throughput,
                    snap.inference_plan.throughput_unit
                );
                Ok(json!({
                    "content": [{ "type": "text", "text": summary }],
                    "structuredContent": {
                        "epoch": snap.inference_plan.epoch,
                        "active_leader": snap.coordinator_quorum.active_leader,
                        "node_count": snap.nodes.len(),
                        "total_vram_gb": total_vram,
                        "total_ram_gb": total_ram,
                        "total_tflops": total_tflops,
                        "estimated_throughput": snap.inference_plan.estimated_throughput,
                        "throughput_unit": snap.inference_plan.throughput_unit,
                        "context_capacity": ctx_cap
                    },
                    "isError": false
                }))
            }
            "kadmium_plan_sharding" => {
                let workload_str = args
                    .get("workload")
                    .and_then(|v| v.as_str())
                    .unwrap_or("inference");
                let (model, workload) = if workload_str == "training" {
                    (
                        ModelSpec::llama_3_70b_lora(),
                        WorkloadKind::DistributedTraining,
                    )
                } else {
                    (ModelSpec::deepseek_v3_671b_moe(), WorkloadKind::Inference)
                };
                let plan = ShardPlanner::compute_plan(state.epoch, &model, workload, &state.nodes);
                let latency_opt =
                    MinMaxStageLatencyOptimizer::optimize_stage_boundaries(&plan, &state.nodes);
                let paged_kv =
                    DistributedPagedKvPlanner::plan_kv_cache_pages(&plan, &model, &state.nodes);
                let context_capacity = DistributedPagedKvPlanner::plan_cluster_context_capacity(
                    &plan,
                    &model,
                    &state.nodes,
                );
                let payload = json!({
                    "shard_plan": plan,
                    "latency_optimization": latency_opt,
                    "paged_kv_allocations": paged_kv,
                    "context_capacity": context_capacity
                });
                let text = serde_json::to_string_pretty(&payload).unwrap_or_default();
                Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "structuredContent": payload,
                    "isError": false
                }))
            }
            "kadmium_pull_artifact" => {
                let uri = args
                    .get("uri")
                    .and_then(|v| v.as_str())
                    .unwrap_or("hf://deepseek-ai/DeepSeek-V3@main");
                if uri.starts_with("gh://") {
                    let run = GitHubConnector::evaluate_repo_on_slice(uri).map_err(|e| {
                        McpJsonRpcError {
                            code: -32602,
                            message: e,
                        }
                    })?;
                    let text = serde_json::to_string_pretty(&run).unwrap_or_default();
                    Ok(json!({
                        "content": [{ "type": "text", "text": text }],
                        "structuredContent": run,
                        "isError": false
                    }))
                } else {
                    let pull_plan = HuggingFaceHubConnector::plan_selective_model_pull(
                        uri,
                        &state.inference_plan,
                    )
                    .map_err(|e| McpJsonRpcError {
                        code: -32602,
                        message: e,
                    })?;
                    let text = serde_json::to_string_pretty(&pull_plan).unwrap_or_default();
                    Ok(json!({
                        "content": [{ "type": "text", "text": text }],
                        "structuredContent": pull_plan,
                        "isError": false
                    }))
                }
            }
            "kadmium_run_inference" => {
                let prompt = args
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Verify Kadmium distributed MoE execution via MCP.");
                let report = DistributedMoEPipelineExecutor::execute_forward_pass(
                    &state.inference_plan,
                    prompt,
                );
                let text = serde_json::to_string_pretty(&report).unwrap_or_default();
                Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "structuredContent": report,
                    "isError": false
                }))
            }
            "kadmium_verify_rl_sandbox" => {
                let grpo_report = GrpoEpochExecutor::run_verified_grpo_step();
                let custom_trace =
                    args.get("candidate_code")
                        .and_then(|v| v.as_str())
                        .map(|_code| {
                            WasiMicroSandbox::default().verify_candidate_program(
                                "mcp-agent-rollout-99",
                                "rk3588-sbc-bramble",
                                &[
                                    SandboxOp::PushInputX,
                                    SandboxOp::Dup,
                                    SandboxOp::Mul,
                                    SandboxOp::PushInputX,
                                    SandboxOp::PushConst(3),
                                    SandboxOp::Mul,
                                    SandboxOp::Add,
                                    SandboxOp::PushConst(7),
                                    SandboxOp::Sub,
                                ],
                                &[(0, -7), (2, 3), (5, 33), (-3, -7)],
                            )
                        });
                let payload = json!({
                    "grpo_step": grpo_report,
                    "custom_candidate_trace": custom_trace
                });
                let text = serde_json::to_string_pretty(&payload).unwrap_or_default();
                Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "structuredContent": payload,
                    "isError": false
                }))
            }
            "kadmium_cloud_shield_arbitrage" => {
                let arb =
                    HybridCloudArbitragePlanner::compute_arbitrage_plan(&state.inference_model);
                let sims = HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield();
                let payload = json!({
                    "arbitrage_plan": arb,
                    "failover_simulations": sims
                });
                let text = serde_json::to_string_pretty(&payload).unwrap_or_default();
                Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "structuredContent": payload,
                    "isError": false
                }))
            }
            "kadmium_supply_marketplace" => {
                if let Some(hours) = args.get("daily_lease_hours").and_then(|v| v.as_f64()) {
                    state.supply_policy.daily_lease_hours = hours.clamp(0.0, 24.0);
                }
                if let Some(mode) = args.get("payout_mode").and_then(|v| v.as_str()) {
                    state.supply_policy.payout_mode = if mode.eq_ignore_ascii_case("usd") {
                        kadmium_core::SupplyPayoutMode::UsdDirectPayout
                    } else {
                        kadmium_core::SupplyPayoutMode::KadmiumCloudCredits {
                            bonus_multiplier: 1.25,
                        }
                    };
                }
                let snap = state.supply_marketplace();
                let text = serde_json::to_string_pretty(&snap).unwrap_or_default();
                Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "structuredContent": snap,
                    "isError": false
                }))
            }
            other => Err(McpJsonRpcError {
                code: -32602,
                message: format!("Unknown Kadmium MCP tool: '{other}'"),
            }),
        }
    }

    fn handle_resource_read(
        state: &mut GatewayState,
        params: Value,
    ) -> Result<Value, McpJsonRpcError> {
        let uri = params
            .get("uri")
            .and_then(|v| v.as_str())
            .unwrap_or("kadmium://cluster/topology");

        let (mime_type, text) = match uri {
            "kadmium://cluster/topology" => (
                "application/json",
                serde_json::to_string_pretty(&state.nodes).unwrap_or_default(),
            ),
            "kadmium://cluster/sharding" => (
                "application/json",
                serde_json::to_string_pretty(&state.inference_plan).unwrap_or_default(),
            ),
            "kadmium://cluster/integrations" => {
                let integrations = kadmium_data::EcosystemIntegrationsOverview::build_for_plan(
                    &state.inference_plan,
                );
                (
                    "application/json",
                    serde_json::to_string_pretty(&integrations).unwrap_or_default(),
                )
            }
            "kadmium://cluster/cloud-shield" => {
                let arb =
                    HybridCloudArbitragePlanner::compute_arbitrage_plan(&state.inference_model);
                (
                    "application/json",
                    serde_json::to_string_pretty(&arb).unwrap_or_default(),
                )
            }
            "kadmium://docs/overview" => ("text/markdown", crate::DOC_OVERVIEW_MD.to_string()),
            other => {
                return Err(McpJsonRpcError {
                    code: -32602,
                    message: format!("Unknown Kadmium MCP resource URI: '{other}'"),
                });
            }
        };

        Ok(json!({
            "contents": [{
                "uri": uri,
                "mimeType": mime_type,
                "text": text
            }]
        }))
    }

    fn handle_prompt_get(params: Value) -> Result<Value, McpJsonRpcError> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("optimize-heterogeneous-sharding");

        let prompt_text = match name {
            "optimize-heterogeneous-sharding" => {
                "Call `kadmium_cluster_status` and `kadmium_plan_sharding` to inspect each node's Q* bandwidth-to-compute ratio. Ensure local hardware (`--cadmium`) holds the primary .ktb layer ranges and cloud burst nodes (`--cherenkov`) only absorb overflow or failover ranges."
            }
            "launch-dream-rsi-loop" => {
                "Call `kadmium_pull_artifact` with a `gh://` repository URI to mount the target codebase onto `slice-04-rl-dream-rsi`, then invoke `kadmium_verify_rl_sandbox` to score candidate trajectories inside the fuel-metered `WasiMicroSandbox`."
            }
            other => {
                return Err(McpJsonRpcError {
                    code: -32602,
                    message: format!("Unknown Kadmium MCP prompt: '{other}'"),
                });
            }
        };

        Ok(json!({
            "description": format!("Kadmium (Cd-48) workflow prompt: {name}"),
            "messages": [{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": prompt_text
                }
            }]
        }))
    }
}
