# Model Context Protocol (MCP) Server (`kadmium-mcp`)

Kadmium (`Cd-48`) includes a built-in **Model Context Protocol (MCP) Server** (`2025-03-26` spec) that lets AI coding agents and agentic harnesses—such as **Claude Code, Cursor, Antigravity, Windsurf, Zed, and OpenCode**—directly inspect, shard, stage, and execute workloads across your heterogeneous compute fabric.

---

## 1. Dual Transports (`stdio` & Streamable HTTP)

### A. Local `stdio` Transport (`kadmium mcp`)
Add `kadmium mcp` to your `.mcp.json`, `claude_desktop_config.json`, or IDE MCP configuration:

```json
{
  "mcpServers": {
    "kadmium": {
      "command": "kadmium",
      "args": ["mcp"],
      "env": {
        "KADMIUM_COORDINATOR_URL": "http://127.0.0.1:7848"
      }
    }
  }
}
```

### B. Streamable HTTP JSON-RPC 2.0 (`POST /mcp`)
When `kadmium-gateway` is running (`kadmium up`), it exposes a live JSON-RPC 2.0 MCP endpoint at:
- **`POST http://127.0.0.1:7848/mcp`** — Execute MCP `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, and `prompts/get` over HTTP.
- **`GET http://127.0.0.1:7848/api/v1/mcp`** — Retrieve the complete `KadmiumMcpManifest` JSON schema catalog.

---

## 2. Exposed MCP Tools (`tools/list`)

| MCP Tool Name | Arguments | Description |
| :--- | :--- | :--- |
| **`kadmium_cluster_status`** | `{}` | Inspect live heterogeneous cluster topology, aggregate VRAM/RAM capacity, effective TFLOPS, node polite states, and Bully+CRDT quorum leader. |
| **`kadmium_plan_sharding`** | `{ "workload": "inference" \| "training" }` | Run the $Q^*$ water-filling sharding planner across local (`--cadmium`) and cloud (`--cherenkov`) nodes. |
| **`kadmium_pull_artifact`** | `{ "uri": "hf://..." \| "gh://..." }` | Stage a Hugging Face model into selective 64 KB-aligned `.ktb` banks or mount a GitHub repository/Dream-RSI world onto a subcluster slice. |
| **`kadmium_run_inference`** | `{ "prompt": "..." }` | Execute a distributed MoE forward pass with `rk3588-sbc-bramble` speculative draft verification and Verde RepOps checks. |
| **`kadmium_verify_rl_sandbox`** | `{ "candidate_code": "..." }` | Score a candidate trajectory or kernel inside the deterministic fuel-metered `WasiMicroSandbox` and compute GRPO group advantages. |
| **`kadmium_cloud_shield_arbitrage`** | `{}` | Query real-time multi-provider spot GPU arbitrage quotes (`RunPod`, `Vast.ai`, `Lambda`, `Prime Intellect`, `AWS`) and failover simulations. |

---

## 3. Exposed MCP Resources (`resources/list`)

- `kadmium://cluster/topology` — Live hardware nodes, VRAM/RAM budgets, bandwidth, and $Q^*$ ratios.
- `kadmium://cluster/sharding` — Active layer-range assignments and `.ktb` tensor bank paths.
- `kadmium://cluster/integrations` — Registered `hf://`, `gh://`, `oci://`, `s3://`, and `wandb://` connectors.
- `kadmium://cluster/cloud-shield` — Real-time spot GPU arbitrage quotes and failover shield simulations.
- `kadmium://docs/overview` — Embedded Markdown documentation for AI agents.

---

## 4. CLI Inspection & Self-Test

```bash
# Inspect the MCP server manifest, tools, resources, and client configuration JSON
cargo run -p kadmium-cli -- mcp --inspect

# Run the stdio JSON-RPC 2.0 MCP loop for IDE/agent attachment
cargo run -p kadmium-cli -- mcp
```
