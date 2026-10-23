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

### `POST /api/v1/training/step`
Runs one live training step across the machines that hold training layers. The body is optional: `{"learning_rate": 0.02}` sets the step size (default `0.05`, clamped to `(0, 1]`). Returns the step report (loss before and after on each stage, `DeMo` compression, RepOps digests) with `global_step` and the recent `loss_history`.

---

### `POST /api/v1/training/checkpoint`
Saves a `.ktb` checkpoint of the training model at the current step to `kadmium-checkpoints/kadmium-<model>-step-<n>.ktb` in the system temp folder, lists it in the bucket under `kdm://pool-bucket/checkpoints/`, and returns its `checkpoint_id`, `file_path`, `merkle_root_fnv64` and `zero_copy_verified` (every page matched its checksum).

---

### `GET /api/v1/training/checkpoints`
Lists the saved checkpoints, newest step first. Each file is read back and its page checksums checked, so the list survives a restart: `checkpoint_id`, `model_id`, `step_index`, `file_path`, `file_size_bytes`, `saved_at_unix_secs` and `verified`. The response also carries `directory`, `training_model_id`, `current_step` and `resumed_from`.

```bash
curl -s http://127.0.0.1:7848/api/v1/training/checkpoints
```

---

### `POST /api/v1/training/resume`
Points training at a saved checkpoint: its model, its step and its weights' Merkle root. The next step continues from `step_index + 1`, and loss history after that step is dropped. Unknown ids answer `404`; a damaged file or a model this coordinator has no profile for answers `422`, both as `{"error": "..."}`.

```bash
curl -s http://127.0.0.1:7848/api/v1/training/resume \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_id": "ckpt-step0042-1a2b3c4d"}'
```

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

---

## 6. Runtime Kernels, Supply Grid & `v1.0 GA` Governance Endpoints

### `GET /api/v1/pipeline/execute`
Executes a live forward pass through [`DistributedMoEPipelineExecutor`](../crates/kadmium-runtime/src/lib.rs), returning stage-by-stage Top-4 MoE softmax routing weights, Colibri double-buffered cold NVMe `.ktb` page fetches, Hivemind INT8 boundary payloads, and Leviathan speculative draft acceptance speedup.

### `GET /api/v1/runtime/kernels`
Runs [`HardwareKernelDispatcher::run_cluster_kernel_benchmark`](../crates/kadmium-runtime/src/kernels.rs) across all cluster nodes (`FusedRmsNormRopeKernel` $\to$ `PagedGqaAttentionKernel` $\to$ `FusedSwiGluExpertKernel`) and verifies bit-identical `Verde RepOps` FNV-1a digests across CUDA, Metal, Vulkan, AVX-512, and ARM NEON backends.

### `GET /api/v1/network/nat`
Returns the cluster-wide [`NatTraversalCoordinator`](../crates/kadmium-network/src/nat.rs) STUN/QUIC UDP hole-punch matrix and `kadmium.ai` DERP/TURN mTLS fallback routes.

### `GET /api/v1/companion/pairing`
Returns discovered UDP `KDMD` LAN peers awaiting 6-digit SAS approval (`ZeroClickLanPairingEngine`) and the live OS swapchain/thermal/battery hysteresis timeline (`OsSwapchainHysteresisEngine`).

### `GET /api/v1/cloud-shield/watcher`
Returns the [`CloudShieldCutoverWatcher`](../crates/kadmium-sharding/src/cloud_watcher.rs) heartbeat monitor simulation (`380 ms` Local $\to$ Cloud cutover RTO, `0` dropped requests, and automatic local failback).

### `GET /api/v1/rl/grpo-step`
Executes a verified online GRPO training step (`GrpoEpochExecutor`) using the deterministic [`WasiMicroSandbox`](../crates/kadmium-rl/src/wasi_sandbox.rs) and `DreamRsiMctsEngine` PUCT tree search.

### `GET /api/v1/supply/marketplace` & `POST /api/v1/supply/configure`
Inspects or updates the [`SupplyLeasePolicy`](../crates/kadmium-core/src/governance.rs) for leasing idle local silicon (`1.25×` Kadmium Cloud credits or direct USD payout).

### `GET /api/v1/ga/governance`
Returns the complete `v1.0 GA` governance bundle:
- `spot_orderbook` & `provisioned_spot_container`: Live multi-provider spot quotes (`SpotOrderbookBidEngine`) and ephemeral middle-layer container descriptor.
- `slice_quotas` & `scoped_api_keys`: Multi-tenant sub-cluster resource quotas and MAC-signed `kdm_live_...` API keys (`SubClusterRbacEngine`).
- `dream_rsi_world_registry`: Public catalog of commit-pinned `gh://` WASI RL worlds (`DreamRsiWorldRegistry`).
- `sample_fraud_proof`: Self-verifying $O(\log_2 L)$ `VerdeFraudProofCertificate` (`VerdeFraudProofEngine`).

## 7. Grid Substrate Endpoints (`/api/v1/grid/*`)

The grid substrate (`kadmium-grid`) runs every workload as a graph of work units scheduled across the mesh. These endpoints expose the mesh as one machine, the planner's shape choice, graph runs, and the credit ledger. Every throughput figure a planner endpoint returns is modelled; every wall time a run endpoint returns is measured on the synthetic reference kernels.

### `GET /api/v1/grid/inventory`
The mesh summed as one machine: online nodes, FP16 TFLOPS (total and polite-availability-weighted), fast memory, system RAM, NVMe, CPU threads, free slots, hop latency range, backends, how many workers are measured and how many compute samples they hold, plus a one-line summary, the number of grid runs, the last run's record (with the samples it added) and `telemetry`: per worker, the measured compute cost per kind of work (ms per unit of work, samples, last sample), the memory streaming rate it measured on itself and the RAM it last reported free.

### `GET /api/v1/grid/plan?model=<id>&strategy=<chain|expert-parallel>`
The placement in force for a model and the choice between the pipeline chain and expert-parallel shapes: both modelled estimates (ms per token, hops per step, cold streaming), the rationale, `decided_by` (`modelled`, `measured` once both shapes carry a calibration that binds to this mesh, `dense`, or `forced`), the calibration records that bind to each shape, and the placement summary. Beside it: `placement_digest` (approvals and calibrations bind to it), `fit` (whether every trunk fits its owner's offer, what streams from the vault, per-node offered and used memory, one line per shortfall), `offers` (the offer in force on every node), `staging` (the transfers that can start, the ones held for an owner's approval, which owners must approve and which have), `usage` (turns, routed hits, experts seen, the hottest experts; `learned` is false until a step has run), `placement_cached`, and the re-pin boundaries and swaps applied so far. `strategy` forces a shape without re-planning. `modelled` is always `true`.

### `POST /api/v1/grid/plan/approve`
```json
{ "model": "deepseek-v3-671b-moe-q4k", "node": "node-laptop", "approve": true }
```
Records (or with `approve: false` withdraws) the owner's approval of the placement in force for their node. The approval binds to the placement digest, so a changed placement asks again; a changed offer drops it too. Returns the plan response with the staging preview updated.

### `POST /api/v1/grid/run`
```json
{ "prompt": "text", "rows": 1, "strategy": "chain", "repin": false }
```
Runs one decode step as a work graph over live workers, with simulated nodes executed in-process on the coordinator, on the placement in force; replicated experts are served by the copy with the lowest measured completion estimate. Returns the shape, graph id, units total and validated, measured wall time, the modelled ms per token for that shape, the final boundary digest, per-unit hop telemetry, credit granted and receipts issued, whether the placement came from the cache, how many routing accesses the usage ledger recorded, and with `repin: true` how many swaps the re-pin boundary after the step made.

### `POST /api/v1/grid/repin`
```json
{ "model": "deepseek-v3-671b-moe-q4k", "max_swaps": 4, "apply": true }
```
One re-pin boundary on the placement in force: per layer, the hottest streamed expert against the coldest pinned one, swapped when the learned heat clears a 25% hysteresis, at most `max_swaps` per boundary. Returns the swaps, the staging transfers they imply (the promoted expert into the node's fast tier, pinned, and the demoted one out), the placement digest afterwards and the running totals. With `apply: false` it only proposes.

### `POST /api/v1/grid/calibrate`
```json
{ "strategy": "expert-parallel", "steps": 8, "rows": 1, "prompt": "text" }
```
Measures the placement in force: at most eight decode steps of one shape within a twenty second budget, recorded as ms per token bound to the placement digest, the node set, the versions that ran and the batch width. A fan-out calibration also runs one chain step on the same prompt and reports `digest_matches_chain`; a record that fails that gate is kept but never steers the choice. Returns the record (with per-node compute time per step from the workers' own clocks and the kernels it measured, today `reference-synthetic`), whether it applies to the plan in force, and what now decides the shape.

### `POST /api/v1/grid/train`
```json
{ "step_index": 7, "learning_rate": 0.05 }
```
Runs one distributed training step: every trunk range as an independent island unit fanning into one DeMo reduce. Returns the DiLoCo reduce report (mean losses, wire compression, consensus norm, master-weight digest), credit granted and hop telemetry.

### `POST /api/v1/grid/replay`
```json
{ "rollouts": 8, "mcts_simulations": 32, "world_id": "gpu-kernel-world" }
```
Runs a Dream-RSI replay batch as a bag of independent units. Returns the nodes used, credit granted and the supply-grid receipts for each validated unit.

### `GET /api/v1/grid/credits`
The cumulative credit ledger since the coordinator started: every entry, totals by node, every receipt, and the USD and cloud-credit totals at the canonical clearing rates.
