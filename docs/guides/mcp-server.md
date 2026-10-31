# Model Context Protocol (MCP) Server (`kadmium-mcp`)

Kadmium includes a **Model Context Protocol (MCP) `2025-03-26`** JSON-RPC 2.0 server for inspecting planners and running reference workloads. Its model-related operations use synthetic tensors and simulations; they do not turn the gateway into a language-model serving endpoint.

---

## 1. Transports and state

### Local `stdio` transport

```json
{
  "mcpServers": {
    "kadmium": {
      "command": "kadmium",
      "args": ["mcp"]
    }
  }
}
```

`kadmium mcp` creates a **fresh in-process `GatewayState`**. It does not attach to an already running coordinator, and `KADMIUM_COORDINATOR_URL` is not consumed by this transport. Use it for an isolated demonstration, not to inspect live shared coordinator state.

### Streamable HTTP transport

Start the coordinator with `kadmium serve` or `kadmium launch`, then use:

- **`POST http://127.0.0.1:7848/mcp`** — `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, and `prompts/get` against the running coordinator's state.
- **`GET http://127.0.0.1:7848/api/v1/mcp`** — the tool, resource, prompt, and client-configuration manifest.

There is no `kadmium up` command.

## 2. Exposed MCP tools

| Tool | Arguments | Current behavior |
| :--- | :--- | :--- |
| `kadmium_cluster_status` | `{}` | Returns cluster topology and planning metadata from the transport's gateway state. |
| `kadmium_plan_sharding` | `{ "workload": "inference" \| "training" }` | Runs the asymmetric sharding planner. |
| `kadmium_pull_artifact` | `{ "uri": "hf://..." \| "gh://..." }` | Produces a selective-pull plan or simulated GitHub result; performs no remote download or mount. |
| `kadmium_run_inference` | `{ "prompt": "..." }` | Executes a synthetic reference pipeline, not language-model generation. |
| `kadmium_verify_rl_sandbox` | `{ "candidate_code": "..." }` | Runs a fixed sequence in the custom fuel-metered stack machine and computes reference GRPO advantages. The supplied source text is not compiled or executed. |
| `kadmium_cloud_shield_arbitrage` | `{}` | Returns fixed sample spot offers and simulated failover reports, not live quotes or provisioned capacity. |
| `kadmium_supply_marketplace` | `{ "daily_lease_hours": 12, "payout_mode": "credits" }` (both optional) | Updates those local supply-policy fields, then returns the sample marketplace/lease snapshot. Hours clamp to `0..=24`; `payout_mode` accepts `credits` or `usd`. It does not create a paid lease. |

The RL sandbox is a bounded custom instruction interpreter, not a general WebAssembly/WASI runtime. It does not execute arbitrary Rust, CUDA, or Triton candidate code and does not perform online model training.

## 3. Exposed MCP resources

- `kadmium://cluster/topology` — hardware and planning profiles.
- `kadmium://cluster/sharding` — active layer assignments and tensor-bank metadata.
- `kadmium://cluster/integrations` — sample integration descriptors, not verified remote connections.
- `kadmium://cluster/cloud-shield` — sample arbitrage and failover simulations.
- `kadmium://docs/overview` — embedded Markdown documentation.

## 4. CLI inspection

```bash
# Print the manifest and client configuration without blocking on stdio.
cargo run -p kadmium-cli -- mcp --inspect

# Run the isolated stdio server.
cargo run -p kadmium-cli -- mcp
```
