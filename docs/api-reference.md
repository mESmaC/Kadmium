# Kadmium HTTP & AI Gateway API Reference

When `kadmium serve` is running (default `http://127.0.0.1:7848`), [`kadmium-gateway`](../crates/kadmium-gateway/src/lib.rs) exposes two families of HTTP endpoints:
1. **Strata-Compatible AI Inference Gateway** (OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`)
2. **Cluster Control Plane & Desktop Companion REST API** (`/api/v1/...`)

---

## 1. AI Inference Gateway Endpoints

### `POST /v1/chat/completions` (OpenAI Compatible)
Drop-in replacement for the OpenAI Chat Completions API. Works out of the box with the official `openai` Python/TypeScript SDKs, Cursor, Continue, LangChain, and OpenWebUI.

**Request Body:**
```json
{
  "model": "deepseek-ai/DeepSeek-V3-671B-MoE-Q4_K_M",
  "messages": [
    { "role": "system", "content": "You are running on a Kadmium heterogeneous cluster." },
    { "role": "user", "content": "Summarize our active pipeline route." }
  ]
}
```

**Response Body (`200 OK`):**
Includes standard OpenAI `choices` plus a `kadmium_fabric` telemetry block showing the exact heterogeneous pipeline route and speculative draft node used:
```json
{
  "id": "chatcmpl-kadmium-e1",
  "object": "chat.completion",
  "model": "deepseek-ai/DeepSeek-V3-671B-MoE-Q4_K_M",
  "kadmium_fabric": {
    "epoch": 1,
    "pipeline_route": [
      "main-gaming-pc[L0..L19]",
      "mac-mini-m4-pro[L20..L44]",
      "gaming-laptop-rtx4070[L45..L55]",
      "friend-remote-pc[L56..L59]",
      "legacy-thinkpad-t480[L60..L60]"
    ],
    "estimated_tokens_per_sec": 38.4,
    "speculative_draft_node": "rk3588-sbc-bramble"
  },
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Executed across the Kadmium heterogeneous mesh (Asymmetric Sharding + FreeToken q* + Colibri NVMe MoE Streaming)."
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

### `POST /v1/messages` (Anthropic Compatible)
Drop-in replacement for the Anthropic Messages API (`claude` SDKs, Claude Code custom base URLs).

**Request Body:**
```json
{
  "model": "deepseek-ai/DeepSeek-V3-671B-MoE-Q4_K_M",
  "messages": [
    { "role": "user", "content": "Hello from Anthropic SDK" }
  ]
}
```

---

## 2. Cluster Control Plane REST API

### `GET /api/v1/cluster/overview`
Returns [`ClusterOverviewResponse`](../crates/kadmium-gateway/src/lib.rs), a complete snapshot of the active cluster:
- `coordinator_quorum`: Active leader, standby candidates, and election term.
- `nodes`: All `HardwareNode` profiles, 5-tier memory hierarchies, trust tiers, and polite states.
- `inference_plan` & `training_plan`: Current [`ClusterShardPlan`](../crates/kadmium-sharding/src/lib.rs) layer and MoE expert assignments.
- `rl_plan`: Disaggregated `Actor`/`Environment`/`Learner` assignments + `DreamRsiEvolutionReport`.
- `training_telemetry`: Live DiLoCo + `DeMo` DCT compression ratio and `Verde` verification status.
- `fts3_transfers`: Active CERN FTS3 P2P shard transfer jobs from `home-nas-vault`.
- `boinc_workunits` & `mesh_links`: Workunit canary verification status and P2P latency/bandwidth matrix.

---

### `POST /api/v1/cluster/shift`
Triggers a live responsibility shift or coordinator failover and returns the updated `ClusterOverviewResponse`.

**Request Body:**
```json
{
  "scenario": "gaming_mode"
}
```

| `scenario` Value | Behavior |
| :--- | :--- |
| `"gaming_mode"` | Transitions `main-gaming-pc` to `PoliteState::GamingOrHeavyApp` (`<200ms` auto-yield), migrating its layers & MoE experts to `mac-mini-m4-pro` and `gaming-laptop-rtx4070` via FTS3 background pre-staging + live KV boundary delta handoff. |
| `"cloud_burst"` | Joins a spot `H100-80GB` node into the cluster and rebalances the shard plan for higher throughput. |
| `"coordinator_failover"` | Simulates a coordinator outage on the primary node and promotes the highest-priority standby (`home-nas-vault`) to leader in a new election term. |
| `"reset"` | Resets the cluster to Epoch 1 baseline (`PoliteState::IdleDedicated`). |

---

### `GET /api/v1/slices`
Returns [`ClusterPoolSliceLayout`](../crates/kadmium-sharding/src/lib.rs), demonstrating a 100-node heterogeneous pool partitioned into 5 isolated 20-node sub-clusters (`ProductionDeployment`, `TestingAndStaging`, `DistributedTraining`, `RlAndDreamRsi`, and `BatchDataProcessing`).

---

### `GET /api/v1/dataflow`
Returns [`DataflowComparisonReport`](../crates/kadmium-sharding/src/lib.rs), comparing TPU/CIM **Weight-Stationary Dynamic Token Streaming** vs. **Token-Stationary Dynamic Weight Streaming** vs. **Hybrid CIM-Adaptive** execution across the cluster.

---

### `GET /api/v1/training/schedule`
Returns [`Pipeline1F1BSchedule`](../crates/kadmium-runtime/src/lib.rs), the exact micro-batch timeline (`Warmup`, `Steady 1F1B`, and `Cooldown` phases, pipeline bubble %, and `DeMo` DCT outer-step synchronization time) across heterogeneous pipeline stages.

---

### `GET /api/v1/rl/dream-rsi`
Returns [`DreamRsiApiResponse`](../crates/kadmium-gateway/src/lib.rs), including the full [`DiscoveryTree`](../crates/kadmium-rl/src/lib.rs) and the offline causal replay [`DreamRsiEvolutionReport`](../crates/kadmium-rl/src/lib.rs) executed on low-power SBCs/legacy laptops.

---

### `GET /api/v1/cloud-shield`
Returns [`CloudShieldApiResponse`](../crates/kadmium-gateway/src/lib.rs), containing the Multi-Datacenter Spot Arbitrage plan (`~62%` savings vs. single-cloud on-demand) and Bidirectional Failover Shield (`Local -> Cloud` and `Cloud -> Local`) simulations.

---

### `GET /api/v1/host/probe`
Probes the local machine running the gateway in real time using [`LiveHardwareProfiler::probe_local_host()`](../crates/kadmium-core/src/lib.rs), returning its detected hardware profile, FreeToken $q^*$ ratio, and a signed `kdm1_...` Join Ticket.

---

## 3. Desktop Companion App API (Open-Source $\leftrightarrow$ `kadmium.ai` Cloud)

### `GET /api/v1/companion/status`
Returns [`CompanionAppStatus`](../crates/kadmium-gateway/src/lib.rs) for the system tray / Desktop Companion UI:
```json
{
  "app_version": "0.1.0",
  "mode": {
    "SelfHostedOpenSource": {
      "local_coordinator_url": "http://127.0.0.1:7848"
    }
  },
  "local_host_polite_state": "IdleDedicated",
  "gaming_auto_yield_active": true,
  "active_cluster_nodes": 7,
  "active_coordinator_leader": "main-gaming-pc",
  "current_hourly_cloud_burn_usd": 0.0
}
```

### `POST /api/v1/companion/mode`
Hot-switches the Desktop Companion App between **100% Free Self-Hosted Open-Source Mode** and **`kadmium.ai` Cloud Managed Mode** without restarting the node:

```bash
curl -X POST http://127.0.0.1:7848/api/v1/companion/mode \
  -H "Content-Type: application/json" \
  -d '{
    "KadmiumCloudManaged": {
      "workspace_slug": "my-team-cluster",
      "cloud_endpoint": "https://api.kadmium.ai",
      "downtime_shield_enabled": true,
      "spot_arbitrage_burst_enabled": true,
      "max_hourly_spend_cap_usd": 2.50
    }
  }'
```

---

## 4. Model Context Protocol (`POST /mcp` & `GET /api/v1/mcp`)

### `POST /mcp`
Streamable HTTP JSON-RPC 2.0 endpoint implementing the **Model Context Protocol (`2025-03-26`)** (`initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`).

### `GET /api/v1/mcp`
Returns [`KadmiumMcpManifest`](../crates/kadmium-gateway/src/mcp.rs), including all 6 MCP tool JSON Schemas, 5 resources, 2 prompts, and the `.mcp.json` configuration snippet.

---

## 5. Ecosystem Integrations & Observability (`/api/v1/integrations`, `/metrics`, `/api/v1/telemetry/wandb`)

### `GET /api/v1/integrations`
Returns [`EcosystemIntegrationsOverview`](../crates/kadmium-data/src/lib.rs) covering `hf://` selective layer-range pulls, `gh://` slice check runs, `oci://`, `s3://`, and `wandb://`.

### `GET /metrics`
Exports live cluster gauges (`kadmium_cluster_epoch`, `kadmium_inference_throughput_tokens_per_sec`, `kadmium_diloco_demo_compression_ratio`, `kadmium_cloud_shield_hourly_burn_usd`, `kadmium_node_fast_memory_gb`) in standard **Prometheus / OpenMetrics (`text/plain; version=0.0.4`)** exposition format.

### `GET /api/v1/telemetry/wandb`
Returns a structured **Weights & Biases (`wandb://`) / OpenTelemetry** run summary JSON object for distributed training, GRPO, and Dream-RSI runs.
