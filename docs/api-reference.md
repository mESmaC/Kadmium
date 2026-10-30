# Kadmium HTTP & AI Gateway API Reference

When `kadmium launch` or `kadmium serve` is running, [`kadmium-gateway`](../crates/kadmium-gateway/src/lib.rs) exposes two families of HTTP endpoints. The default port is `7848`; the coordinator listens on all interfaces and is accessible locally at `http://127.0.0.1:7848`.

1. **Reference compute diagnostics with OpenAI/Anthropic-shaped responses** (`/v1/...`)
2. **Cluster Control Plane & Desktop Companion REST API** (`/api/v1/...`)

---

## 1. Reference API Endpoints

### Validation, imports and saved state

`POST /api/v1/cluster/model` accepts known preset IDs/aliases or IDs with an imported manifest. Unknown or empty IDs return HTTP 400 with `{"error":{"type":"invalid_request_error","message":"..."}}`. Both inference and training IDs are validated before changing either plan, the epoch, or diagnostic state.

`POST /api/v1/models/import` requires a readable `source_path`. Missing, unreadable, truncated or malformed files return a structured HTTP 400 error without updating active/imported model state. To intentionally generate a small synthetic reference fixture, omit `source_path` and explicitly set `demo_fixture: true`. The returned manifest includes `synthetic_fixture`; compiling such a file does not import real model weights. EXL3 conversion currently supports synthetic fixtures only. The Companion requires an existing path; the Control Room's **Compile reference fixture** action explicitly requests demo data.

`POST /api/v1/storage/pool-bucket/object` requires positive data/parity shard counts whose checked sum is at most 256 (defaults: 8 + 2). Invalid counts return structured HTTP 400 errors before shared state is changed. Entries record metadata and planned placement, not uploaded object bytes. Live storage reports use `report_kind: "live_placement_estimate"`, `durability_verified: false`, and a `protection_status` explaining unverified protection or insufficient distinct holders. Configured anchors have zero verified coverage until byte synchronization and independent failure domains can be verified. Synthetic recovery exercises remain separate in the benchmark reports (`report_kind: "synthetic_benchmark"`).

The coordinator restores model plans, imports, object/anchor metadata, training progress, checkpoint step and runtime version from `target/kadmium-state/cluster_state.json` before serving requests. Restart from the same working directory to use the same journal. Missing journals start a fresh workspace; corrupt or unsupported journals stop startup with an error and preserve the original file for recovery. Journal replacement writes and syncs a temporary file first. Restored plans and imports do not establish that weights are resident or workers are running. Versionless journals from earlier releases are accepted as schema version 1.

These routes retain their existing behavior. They execute small reference workloads and return diagnostic output, not generated language-model responses. There is no external inference relay or backend URL/key setting. This setup change keeps the existing Cargo dependencies and does not replace these routes with an unconfigured-backend error.

The API shapes are useful for inspecting reference execution, but are not a promise of full OpenAI/Anthropic protocol or SDK compatibility. A model ID in a response or `active_loaded` in the catalog is diagnostic state, not evidence that production model weights were loaded. Throughput fields describe reference execution or estimates, not measured language-model generation.

### `POST /v1/chat/completions` (reference diagnostic)

Accepts `messages` plus optional `model`, `stream` and `vision_offload`. The final message supplies the text sample for reference execution; earlier conversation turns do not form a model context. The supplied model name labels the response and does not select or load model weights. Sampling controls, tools and other provider options are not implemented.

**Request Body:**
```json
{
  "messages": [
    { "role": "user", "content": "Run a reference step." }
  ]
}
```

**Response Body (`200 OK`):**
Includes an OpenAI-shaped `choices` array with a fixed assistant message plus a `kadmium_fabric` telemetry block. The following is an illustrative excerpt, not a throughput benchmark or a guarantee of this topology:
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

### `POST /v1/messages` (reference diagnostic)

Accepts the same limited request structure as the chat route and returns a fixed diagnostic message containing stage and timing information. It does not implement the full Anthropic Messages contract, tool use, conversation processing or model generation.

**Request Body:**
```json
{
  "messages": [
    { "role": "user", "content": "Run a reference step." }
  ]
}
```

### Streaming, models and embeddings

`stream: true` on either message route returns a prebuilt `text/event-stream` response after reference execution. The OpenAI-shaped stream contains fixed chunks and `[DONE]`; the Anthropic-shaped stream contains a limited diagnostic event sequence. Neither is incremental token generation or a complete provider streaming protocol.

| Endpoint | Current behavior |
| :--- | :--- |
| `GET /v1/models` | Returns Kadmium's planning catalog and reference activity metadata, not a list of loaded language models. |
| `POST /v1/embeddings` | Takes a non-empty string or non-empty array of non-empty strings, optional `model` and `dimensions` (default `64`), and returns reference hidden-state vectors with telemetry. Invalid or omitted input returns a structured HTTP 400 error; batches are accepted in full or rejected, preserving input indices. These are not pretrained semantic embeddings. |

Client API keys do not authenticate access to these routes. Keep the all-interface coordinator on a trusted network or provide external access controls. A future external inference relay requires a separate dependency and downstream-access design decision; none is included here. The MCP endpoint remains separately available at `/mcp`.

---

## 2. Cluster Control Plane REST API

### `GET /api/v1/cluster/overview`
Returns [`ClusterOverviewResponse`](../crates/kadmium-gateway/src/lib.rs), a complete snapshot of the active cluster:
- `coordinator_quorum`: Active leader, standby candidates, and election term.
- `nodes`: All `HardwareNode` profiles, 5-tier memory hierarchies, trust tiers, and polite states.
- `inference_plan` & `training_plan`: Current [`ClusterShardPlan`](../crates/kadmium-sharding/src/lib.rs) layer and MoE expert assignments.
- `rl_plan`: Disaggregated `Actor`/`Environment`/`Learner` assignments + `DreamRsiEvolutionReport`.
- `training_telemetry`: Planning/sample metadata with the current legacy preview counters, loss and checkpoint position. Its island count, local-step interval and `verde_repops_verified` flag are sampled metadata, not evidence of island synchronization or the last preview's digest check. Use `/api/v1/training/step`'s `all_stages_repops_verified` for payload consistency and the separate session API for actual barrier progress.
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
Runs a stateless reference training preview for each stage assigned training layers. Each stage starts from its own synthetic baseline on every request; different layer ranges are not model replicas and are not averaged into a shared model. The body is optional: `{"learning_rate": 0.02}` sets the step size (default `0.05`, clamped to `(0, 1]`). Returns per-stage loss before/after, `DeMo` compression, RepOps digests, `global_step` and the recent `loss_history`. These counters do not imply persistent model or local optimizer updates.

| Report field | Meaning |
| :--- | :--- |
| `execution_mode` | `"stateless_per_stage_gradient_preview"`; distinguishes this legacy preview from a stateful DiLoCo session. |
| `stage_preview_digest_fnv1a` | Manifest-bound digest of the canonical aggregate stage preview. It is not a checkpoint or a digest of committed DiLoCo master weights. |
| `preview_error` | `null` when aggregate preview arithmetic succeeds; otherwise a diagnostic error string. On error the preview norm and digest are zero placeholders, not valid preview results. Check this separately from stage digest verification. |
| `all_stages_repops_verified` | True only for a nonempty set of returned stages whose reported island-step digests match recomputation from the received payload under the current RepOps manifest, and whose layer ranges match the requested ranges. This is payload/digest consistency, not independent kernel re-execution or a referee proof. |

A payload may have a consistent digest yet fail preview arithmetic, or have valid preview arithmetic yet an inconsistent digest. Neither verification field certifies a stateful training round.

The compatibility field `diloco_outer_consensus_norm` is the norm of concatenated independent stage previews, not replica consensus. Use the [DiLoCo sessions](#stateful-diloco-sessions) for persistent master parameters, outer momentum and synchronization barriers.

---

### Stateful DiLoCo sessions

These endpoints manage independent, in-memory synchronous training sessions. They do not change `/api/v1/training/step`, the grid training endpoints, RL execution or the active inference model. Each island trains a complete replica with an identical named shard layout, takes exactly `config.inner_steps` local optimizer steps from the committed snapshot, and submits its final parameters. The master advances only after all registered islands submit for the current round.

A session `ParameterShard` is a named slice of the parameter vector inside **each complete replica**. Every island submits every shard; the session does not assign different layers to different islands or execute a pipeline within an island. Pipeline partitioning across devices inside each replica is a [target architecture](ARCHITECTURE.md#53-mode-c-sharded-island-diloco-target-architecture), not a capability provided by this session engine. Local learner execution and optimizer state belong to the caller.

| Method and path | Purpose |
| :--- | :--- |
| `POST /api/v1/training/diloco/sessions` | Create from `{session_id, config, island_ids, shards}`; `config` may be omitted for runtime defaults. |
| `GET /api/v1/training/diloco/sessions` | List sessions. |
| `GET /api/v1/training/diloco/sessions/{session_id}` | Inspect config, current round/baseline, received/pending islands, synchronization count and parameter payload estimates. |
| `POST /api/v1/training/diloco/sessions/{session_id}/updates` | Submit `{island_id, round, baseline_id, local_steps, shards}`. Returns round progress; the last submission commits the outer update. |
| `GET /api/v1/training/diloco/sessions/{session_id}/snapshot` | Get `{round, baseline_id, shards}` for committed master parameters. Pending island values are not exposed as committed weights. |
| `GET /api/v1/training/diloco/sessions/{session_id}/checkpoint` | Download the session checkpoint as a JSON object. |
| `POST /api/v1/training/diloco/restore` | Restore `{session_id, checkpoint}` under an unused session ID. |
| `DELETE /api/v1/training/diloco/sessions/{session_id}` | Release a session. |

Create and restore return `201` with `session_id` and flattened status fields; list returns an array of those status objects. Delete returns `204`. Unknown sessions return `404`; invalid create/restore data returns `400`; rejected updates, reused IDs or the 16-session limit return `409`. Session IDs, island IDs and parameter shard IDs each contain 1–96 ASCII letters, digits, hyphens or underscores. Request bodies are limited to 8 MiB.

This bounded reference engine accepts up to 64 islands and 64 named shards, with `inner_steps` in `1..=1000000`. The resident scalar budget is `parameters_per_island * (islands + 2) <= 262144`, counting local submissions, master parameters and momentum. It is intended for small adapters and reference workloads; full-model GPU training requires a backend with streamed or distributed parameter storage.

Example create body, with two islands and two scalar shards:

```json
{
  "session_id": "regression",
  "config": {"inner_steps": 20, "outer_learning_rate": 0.7, "outer_momentum": 0.9},
  "island_ids": ["a", "b"],
  "shards": [
    {"shard_id": "weight", "values": [0.0]},
    {"shard_id": "bias", "values": [0.0]}
  ]
}
```

Read the snapshot, then include its exact `round` and `baseline_id` in every island's update. For example, after 20 actual local steps, island `a` may submit:

```json
{
  "island_id": "a",
  "round": 0,
  "baseline_id": "COPY_THE_EXACT_SNAPSHOT_BASELINE_ID",
  "local_steps": 20,
  "shards": [
    {"shard_id": "weight", "values": [0.35]},
    {"shard_id": "bias", "values": [0.31]}
  ]
}
```

The values above illustrate the payload shape; clients must submit their own trained values. Shards match by ID, regardless of submission ordering. Unknown/duplicate islands, stale rounds, incorrect baseline markers or local-step counts, incompatible layouts and nonfinite values are rejected without replacing committed weights. The baseline marker checks matching round/starting weights; it is not cryptographic proof that a client performed its claimed training. The session forms FP32 master-minus-local pseudo-gradients and delegates the outer update to the shared `DiLoCoOuterOptimizer::try_step_pseudo_gradients`: contributors are ordered by pseudo-gradient bits, averaged with the RepOps FP64 tree reduction and applied through the existing FP32 Nesterov update with persistent momentum.

`parameter_payload_estimates` counts dense FP32 parameter uploads and broadcasts. It excludes transport metadata, optimizer state and retries, so it is not a measured network-throughput figure. Checkpoints preserve the master, outer momentum and already accepted updates within an incomplete round. They do **not** contain external learners' AdamW state, data cursors or RNG state: callers must checkpoint those separately. Sessions disappear on coordinator restart unless a caller downloads and restores a checkpoint. A restored session may continue its incomplete barrier; restore does not overwrite an existing session.

For a working local learner, see [`kadmium diloco`](cli-reference.md#kadmium-diloco) and the [training guide](guides/training-and-dream-rsi.md#b-stateful-diloco-rounds).

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

The grid substrate (`kadmium-grid`) runs every workload as a graph of work units scheduled across the mesh. These endpoints expose the mesh as one machine, the planner's shape choice and its load balancer, graph runs and multi-step decode loops, the sessions and their KV tiers, the sweep harness, and the credit ledger. Every throughput figure a planner endpoint returns is modelled; every wall time a run endpoint returns is measured on the synthetic reference kernels, and every response that carries a measured figure names the kernels (`reference-synthetic` today). Where a figure rests on a modelled quantity (the load balancer's expected bytes, a drafter's acceptance, a KV page's size) the response says `modelled`.

### `GET /api/v1/grid/inventory`
The mesh summed as one machine: online nodes, FP16 TFLOPS (total and polite-availability-weighted), fast memory, system RAM, NVMe, CPU threads, free slots, hop latency range, backends, how many workers are measured and how many compute samples they hold, plus a one-line summary, the number of grid runs, the last run's record (with the samples it added and the Verde disputes it settled) and `telemetry`: per worker, the measured compute cost per kind of work (ms per unit of work, samples, last sample), the memory streaming rate it measured on itself and the RAM it last reported free.

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
Runs one decode step as a work graph over live workers, with simulated nodes executed in-process on the coordinator, on the placement in force; replicated experts are served by the copy with the lowest measured completion estimate. Returns the shape, graph id, units total and validated, measured wall time, the modelled ms per token for that shape, the final boundary digest, per-unit hop telemetry, credit granted and receipts issued, whether the placement came from the cache, how many routing accesses the usage ledger recorded, with `repin: true` how many swaps the re-pin boundary after the step made, and `disputes`: every Verde dispute the step settled (see ARCHITECTURE 9.6), each with the unit, the disputants, the referee, the `method` (`layer-bisection` or `whole-unit`), `layers_in_unit`, `digest_comparisons`, `openings_requested` and `openings_verified`, the `divergent_layer`, `referee_layers_run`, the `winner`, the `losers` and the `outcome`. Digests in every response are RepOps digests under the current manifest: a layer boundary's digest covers the manifest id, the layer and the exact INT8 bytes, and a unit's digest is the fold of its layer digests.

### `POST /api/v1/grid/repin`
```json
{ "model": "deepseek-v3-671b-moe-q4k", "max_swaps": 4, "apply": true }
```
One re-pin boundary on the placement in force, with the load balancer in its incremental mode (ARCHITECTURE 9.9): per layer, the hottest streamed expert against the coldest pinned one, swapped when the learned heat clears a 25% hysteresis; then, within whatever of `max_swaps` is left, copies of hot experts onto nodes with free fast memory where a copy lowers the layer's busiest node by more than the same hysteresis. Returns the `swaps`, the `replica_changes` (each with the layer's busiest node before and after, in bytes and modelled ms), the `balance` report for the expert-parallel shape, the staging transfers they imply (a promoted expert into the node's fast tier, pinned, the demoted one out, and each copy from its primary), the placement digest afterwards, the running totals, and `expert_read_ms_before` and `_after`: the cost model's expert reads on the step's critical path at one row, modelled. With `apply: false` it only proposes.

The plan response carries the same balancer's account when it ran at planning time: `placement.balance` (per layer) and `choice.balance` (a summary with the cost model's figures at its batch width, where at one row every expert's rows go to its primary and a copy changes nothing yet).

### `POST /api/v1/grid/calibrate`
```json
{ "strategy": "expert-parallel", "steps": 8, "rows": 1, "prompt": "text" }
```
Measures the placement in force: at most eight decode steps of one shape within a twenty second budget, recorded as ms per token bound to the placement digest, the node set, the versions that ran and the batch width. A fan-out calibration also runs one chain step on the same prompt and reports `digest_matches_chain`; a record that fails that gate is kept but never steers the choice. Returns the record (with per-node compute time per step from the workers' own clocks and the kernels it measured, today `reference-synthetic`), whether it applies to the plan in force, and what now decides the shape.

### `POST /api/v1/grid/decode`
```json
{ "prompt": "text", "rows": 1, "steps": 4, "sessions": 1, "strategy": "chain", "pipelined": true }
```
Several decode steps of one or several sessions through one runner (ARCHITECTURE 9.10). Pipelined (the default), each session's next step is planned and queued behind its current one while that one runs on the workers, so its first unit leaves in the loop turn that validates the previous sink, and the sessions' graphs interleave in one queue; `pipelined: false` runs one graph at a time, the shape of `/api/v1/grid/run` repeated. Each session starts from the prompt with its index appended. Returns per session the steps, tokens, wall, mean step time, every inter-step gap (the next step's first dispatch minus this step's last return, on the coordinator's clock), the final output and digest; in all, `wall_ms`, `measured_tokens_per_sec` (tokens over the wall), `measured_per_user_tokens_per_sec` (one session's tokens over its own wall, averaged), the mean and maximum gap, `interleaved` (units of two sessions in flight at once), the modelled ms per token for the shape, `kernels` (the kernels the measured figures ran on, `reference-synthetic` today), units, credits, routing recorded and disputes. Steps are at most 64 and sessions at most 8.

```json
{ "prompt": "text", "steps": 4, "sessions": 2, "speculative": true, "max_draft": 5, "draft_profile": "code", "early_stop": true, "confidence_lag": 1 }
```
With `speculative: true` every session decodes one row and drafts (ARCHITECTURE 9.11): `draft_profile` picks the modelled drafter (`code`, `chat` or `none`), `max_draft` its positions per step (at most 16), and the verification scheduler sizes every session's verify batch per step by DSpark's rule from the steps-per-second profile and the confidences `confidence_lag` steps behind, with DSpark's causal early stop unless `early_stop: false` asks for DeepInfer's search of every pair. The response adds `verification`: the profile's source (`measured` from calibration records of this placement at two or more batch widths, else `modelled` from the cost model) and widths, the mean verify batch, expected accepted tokens and expected gain over verifying nothing, the positions `proposed` and `accepted`, the `acceptance_rate`, `tokens_full_acceptance` and `tokens_accepted` with each over the wall as tokens per second, `tau`, and `modelled_acceptance: true` with a note: the acceptance is a modelled coin, no drafter exists, and `tokens_total` and `measured_tokens_per_sec` count accepted tokens.

With `session: "<id>"` the loop continues one named session (ARCHITECTURE 9.12): its tokens accumulate in the session registry, its KV pages are recorded resident on the trunk owners, and when its pages have sunk the loop replays its bounded suffix first and reports it as `resumed`. Every response carries `session_states`: the sessions as the registry holds them afterwards, with tokens and KV tier.

### `POST /api/v1/grid/sweep`
```json
{ "rows": [1, 2, 4, 8], "node_counts": [7, 2], "steps": 4, "sessions": 1, "strategy": "chain", "pipelined": true, "tag": "chain-2026-10", "write_ledger": true, "record_calibrations": true }
```
The sweep harness (ARCHITECTURE 9.13): one pipelined decode loop per batch width and per node count (the first `n` nodes of the mesh, placed afresh), every point a row of a Pareto table of throughput against per-user speed that carries its full configuration (model, shape, rows, sessions, steps, node count and node set, placement digest, version per node, kernels, mode, drafter, repository state). Points compare only within a curve, every field but the batch width equal; each node set is its own curve; `pareto` marks the frontier within each. A plain single-session point is also recorded as a calibration of its placement at its width, so the verification scheduler's profile becomes measured at those widths. `speculative`, `max_draft` and `draft_profile` make every point draft one row and report the full-acceptance throughput beside the accepted one. Returns the points, their curve ids and frontier flags, the table as markdown, the ledger file it was appended to (`target/kadmium-state/sweeps/<tag>.md`), the calibration records added and the points skipped with why. At most 8 widths of at most 64 rows, 8 node counts, 16 steps and 8 sessions.

### `GET /api/v1/grid/sessions`
The decode sessions the grid has run: each with its state (tokens, last activity, KV tier, replays), its page count, its modelled page bytes by tier and whether a decode would replay first; the tier policy in force; and the modelled bytes across every session by tier. `modelled` is always `true`: every byte figure comes from the model spec's reference KV size at an assumed window.

### `POST /api/v1/grid/sessions/tier`
```json
{ "at_unix_ms": 1761200000000, "idle_to_nvme_ms": 60000, "idle_to_vault_ms": 600000 }
```
One tiering pass, judged at `at_unix_ms` (now when absent; a later time shows what the policy would do after that much idleness): every session idle past `idle_to_nvme_ms` has its pages moved to the owner's NVMe (the vault when the owner offers none), past `idle_to_vault_ms` to the vault node, and a session active again comes back up. The thresholds given replace the policy's. Returns the plan (every page's move with its modelled bytes, the sessions sunk and raised, bytes per tier afterwards) and the sessions.

### `POST /api/v1/grid/sessions/replay`
```json
{ "session": "alpha" }
```
Replays the session's bounded suffix (the window plus the backtrack, 140 tokens by default, or the whole context when shorter) as one `KvReplay` unit per trunk range on its owner, through the dispatcher, and records the pages resident in fast memory again. Returns the suffix and context sizes, the tier the pages came back from, units, the measured wall on the kernels the workers run (`reference-synthetic` today), the pages rebuilt with their modelled bytes, and the digest of the last boundary. An unknown session is an error.

### `POST /api/v1/grid/train`
```json
{ "step_index": 7, "learning_rate": 0.05 }
```
Runs a stateless reference training preview for each trunk range and summarizes its DeMo gradients. Each stage has its own synthetic baseline; different layer ranges are never averaged as model replicas. The reduce report declares `execution_mode: "stateless_per_stage_gradient_preview"` and returns mean losses, wire compression, an aggregate norm/digest, credit, hop telemetry and `disputes` as for `/api/v1/grid/run`. Stage ranges are ordered canonically and summaries use the RepOps tree and manifest-bound digests, so arrival order does not affect the report. The legacy `diloco_outer_consensus_norm` and `master_weight_digest_fnv1a` keys remain for compatibility; they describe the independent-stage preview, not a saved model or DiLoCo consensus. Use the stateful DiLoCo session endpoints for persistent outer rounds.

### `POST /api/v1/grid/replay`
```json
{ "rollouts": 8, "mcts_simulations": 32, "world_id": "gpu-kernel-world" }
```
Runs a Dream-RSI replay batch as a bag of independent units. Returns the nodes used, credit granted and the supply-grid receipts for each validated unit.

### `GET /api/v1/grid/credits`
The cumulative credit ledger since the coordinator started: every entry, totals by node, every receipt, and the USD and cloud-credit totals at the canonical clearing rates.
