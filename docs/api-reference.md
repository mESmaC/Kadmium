# Kadmium HTTP & AI Gateway API Reference

When `kadmium launch` or `kadmium serve` is running, [`kadmium-gateway`](../crates/kadmium-gateway/src/lib.rs) exposes two families of HTTP endpoints. The default port is `7848`; the coordinator listens on all interfaces and is accessible locally at `http://127.0.0.1:7848`.

1. **Reference compute diagnostics with OpenAI/Anthropic-shaped responses** (`/v1/...`)
2. **Cluster Control Plane & Desktop Companion REST API** (`/api/v1/...`)

Condenser is included at `/condenser/` in Kadmium and can also run separately on loopback port `7856`. Embedded APIs use `/condenser/api/v1/...`. They require a loopback transport peer, same-origin requests, and this launch's access cookie, which is set by the one-time link `kadmium serve` prints (also saved in `condenser-access-<port>` in the state directory). The embedded workspace root is `condenser-workspace` in the state directory, or `KADMIUM_CONDENSER_WORKSPACE`. Its [inference workspace guide](guides/condenser-workspace.md) covers provider routing, maximum-quality defaults, compute inspection, task history and run cancellation. Kadmium's read-only `GET /api/v1/inference/workspace?model_id=<catalog-id>` returns fleet and placement context without loading a model or changing cluster selection.

## Condenser checkpoints and compute jobs

For first-party tools, durable recall, research acquisition requirements and local/API subagent configuration, see [Condenser agent tools](guides/condenser-workspace.md#built-in-agent-tools). Both chat endpoints accept optional `agent: {"session_id": "task-id", "research_required": true}`. Tools remain opt-in and subject to `allowed_tools`; `GET /api/v1/plugins` includes the reserved `condenser` catalog. `POST /api/v1/plugins/invoke` accepts the same optional `agent` settings for explicit built-in invocations.

These routes run on standalone Condenser's loopback server, with its Host/Origin guard, or under `/condenser` in Kadmium:

| Method and path | Contract |
| --- | --- |
| `GET /api/v1/checkpoints` | Returns `{directory, checkpoints}` from the local package library. Listing is not an integrity check. |
| `POST /api/v1/checkpoints/import` | `{source_path}` returns a checkpoint summary after lossless normalization and atomic publication. |
| `POST /api/v1/checkpoints/browse` | `{path?}` lists model files and folders on the host (up to 500 entries), with the directory, parent and a truncation indicator. |
| `POST /api/v1/checkpoints/load` | `{checkpoint_id, base_url, context_length?}` verifies the package, imports preserved source files into loopback Ollama without quantization, checks tensor metadata and preloads the declared context or an explicitly smaller allocation. Returns `{checkpoint, provider, context_length, tensor_metadata_verified, execution}`. |
| `POST /api/v1/compute/jobs` | `{base_url, api_key?, job}` proxies a job to the coordinator below with a 75-second timeout. Credentials are request-scoped; redirects are disabled. |

Only one import/load runs per server. A load response does not change a task's route; the user explicitly chooses **Use in this task**. KTB banks are storage artifacts, and the external backend's arithmetic is not RepOps-certified. See [local checkpoints](guides/local-checkpoints.md) for exact formats and failure behavior.

Chat sampling accepts `context_window` (Ollama only), `max_tokens`, and `compact_history` (MCL4 opt-in). Omitted context/output values resolve maximums. OpenAI-compatible requests omit `max_tokens` (the server maximum) unless the catalog advertises an explicit output limit. Missing required metadata (Anthropic output, Ollama context) produces an error instead of a guessed cap. `provider.timeout_secs: 0` (default) removes the generation timeout. Catalog discovery remains bounded independently. Context and output share the model's total window; other providers' context allocation is configured on their server.

## Live worker jobs

The **coordinator** also exposes `POST /api/v1/compute/jobs`, accepting the job itself:

```json
{"kind":"matrix_multiply","matrix":{"rows":2,"inner":3,"columns":2,"left":[1,2,3,4,5,6],"right":[7,8,9,10,11,12]}}
```

Other shapes are `{"kind":"training_preview","step_index":1,"learning_rate":0.01}` and `{"kind":"replay","rollouts":8,"simulations":32}`. Matrix inputs use row-major FP32, dimensions 1–256 (at least two rows), and finite values. The kernel uses RepOps dot/reduction order without quantized transport. Training is a stateless synthetic stage preview, not imported-model learning. Replay uses the bundled in-memory reference world. Learning rate must be in `(0, 1]`; replay accepts 2–64 rollouts and 8–256 simulations. Unknown fields/kinds are rejected.

At least two online `TrustedCore` TCP workers without outstanding approval requirements are required. Matrix row ranges are pinned to distinct workers. Demo and in-process fallback are disabled, including after a worker failure. Responses include `graph_id`, `kind`, `execution`, `success`, `distributed`, `nodes_used`, `wall_ms`, `units_total`, `units_validated`, `traces`, `result` and `error`. A matrix result contains `{rows, columns, values}` in submitted row order. Execution failures can return HTTP 200 with `success:false`; inspect both the status and report. Validation failures return HTTP 400 and the 60-second deadline returns HTTP 504. Dispatches have 15-second per-call deadlines. No monetary credits are issued. See [distributed compute](guides/distributed-compute.md) for setup and real learner integration.

---

## 1. Reference API Endpoints

### Validation, imports and saved state

`POST /api/v1/cluster/model` accepts known preset IDs/aliases or IDs with an imported manifest. Unknown or empty IDs return HTTP 400 with `{"error":{"type":"invalid_request_error","message":"..."}}`. Both inference and training IDs are validated before changing either plan, the epoch, or diagnostic state.

`POST /api/v1/models/import` requires a readable `source_path`. Missing, unreadable, truncated or malformed files return a structured HTTP 400 error without updating active/imported model state. To intentionally generate a small synthetic reference fixture, omit `source_path` and explicitly set `demo_fixture: true`. The returned manifest includes `synthetic_fixture`; compiling such a file does not import real model weights. EXL3 conversion currently supports synthetic fixtures only. The Companion requires an existing path; the Control Room's **Compile reference fixture** action explicitly requests demo data.

`POST /api/v1/storage/pool-bucket/object` requires positive data/parity shard counts whose checked sum is at most 256 (defaults: 8 + 2). Invalid counts return structured HTTP 400 errors before shared state is changed. Entries record metadata and planned placement, not uploaded object bytes. Live storage reports use `report_kind: "live_placement_estimate"`, `durability_verified: false`, and a `protection_status` explaining unverified protection or insufficient distinct holders. Configured anchors have zero verified coverage until byte synchronization and independent failure domains can be verified. Synthetic recovery exercises remain separate in the benchmark reports (`report_kind: "synthetic_benchmark"`).

The coordinator restores model plans, imports, object/anchor metadata, training progress, checkpoint step and runtime version before serving requests. CLI and desktop coordinators store `cluster_state.json` (live) and `cluster_state.demo.json` (demo) separately, so demo saves cannot overwrite live progress. The journal directory is `%LOCALAPPDATA%\Kadmium\state` on Windows, `~/Library/Application Support/Kadmium` on macOS, and `$XDG_STATE_HOME/kadmium` (or `~/.local/state/kadmium`) on Linux. Set `KADMIUM_STATE_DIR` to an absolute writable directory to use a separate workspace. Without that override, the first launch migrates a journal found under `target/kadmium-state` in the working directory, leaving the original file intact. Launch once from the old working directory when upgrading. Normalized model packages now use the shared per-user directory documented in the local checkpoint guide. Training checkpoint and diagnostic artifact paths are unchanged.

Missing journals start a fresh workspace; corrupt JSON or unsupported schema versions stop startup with an error and preserve the original file for recovery. Unsupported legacy custom imports are skipped with a warning, without deleting their model files. Supported imports and unrelated state still restore. A selected custom model falls back to the startup model plan, and training progress tied to that unsupported model is not restored. New custom IDs remain rejected. Journal replacement writes and syncs a unique temporary file first. Restored plans and imports do not establish that weights are resident or workers are running. Versionless journals from earlier releases are accepted as schema version 1; progress with missing provenance is never treated as live training.

The gateway's `/v1` routes retain their existing behavior. They execute small reference workloads and return diagnostic output, not generated language-model responses. They do not relay requests to Condenser's inference providers.

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
- `fts3_transfers`: Scheduler-generated transfer planning records associated with the sample `node-home-nas` / `truenas-core-vault` profile, not active CERN FTS3 network jobs.
- `boinc_workunits` & `mesh_links`: Workunit canary verification status and P2P latency/bandwidth matrix.

---

### `POST /api/v1/cluster/shift`
Applies a responsibility-shift or coordinator-failover scenario to coordinator state and returns the updated `ClusterOverviewResponse`. Layer movement and timings are planner/reference behavior, not proof of uninterrupted model migration.

**Request Body:**
```json
{
  "scenario": "gaming_mode"
}
```

| `scenario` Value | Behavior |
| :--- | :--- |
| `"gaming_mode"` | Sets the first node (sample ID `node-main-pc`) to `PoliteState::GamingOrHeavyApp` and computes a relief placement. The sub-200 ms figure is a policy target/estimate. |
| `"cloud_burst"` | Adds a sample spot `H100-80GB` profile and rebalances the plan; it does not rent an instance. |
| `"coordinator_failover"` | Simulates a coordinator outage and promotes the highest-priority standby in coordinator state. |
| `"phone_farm_box"` | Applies the phone-farm platform scenario. |
| `"usb4_handheld"` | Applies the USB4 30 Gbps handheld scenario. |
| `"usbc_handheld"` | Alias for the same USB4 30 Gbps handheld branch, not a distinct profile. |
| `"reset"` | Resets the cluster to its baseline state. |

Unrecognized scenario strings currently take the same reset branch rather than returning `400`; clients should send only documented values.

---

### `GET /api/v1/slices`
Returns [`ClusterPoolSliceLayout`](../crates/kadmium-sharding/src/lib.rs). Outside live-only mode this is a 100-node/five-slice demonstration with purposes `ProductionDeployment`, `StagingAndTesting`, `DistributedTrainingLab`, `RlAndDreamRsiArena`, and `BatchAndEvalPool` (including `slice-03-train-diloco` and `slice-05-batch-eval`). Live-only mode returns no slices without an active model, otherwise one live partition. The partitions are planning data, not isolated network namespaces or separately authenticated coordinators.

---

### `GET /api/v1/dataflow`
Returns [`DataflowComparisonReport`](../crates/kadmium-sharding/src/lib.rs) with modelled profiles for `AdaptiveTieredHybrid`, `ComputeInMemoryTokenStreaming`, `TokenStationaryWeightStreaming`, and `DisaggregatedDualStream`.

---

### `GET /api/v1/training/schedule`
Returns [`Pipeline1F1BSchedule`](../crates/kadmium-runtime/src/lib.rs): stage/micro-batch counts, total modelled duration, bubble percentage, peak activations, and the `Warmup`, `Steady 1F1B`, and `Cooldown` event timeline. It does not contain a DeMo outer-step synchronization measurement.

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
Writes a deterministic **synthetic reference KTB** tagged with the current model and step to `kadmium-checkpoints/kadmium-<model>-step-<n>.ktb` in the system temp folder. It is not a snapshot of learned stage weights. The checkpoint is listed under `kdm://pool-bucket/checkpoints/`; the response includes `checkpoint_id`, `file_path`, `merkle_root_fnv64`, and `zero_copy_verified` (every page matched its checksum).

---

### `GET /api/v1/training/checkpoints`
Lists the saved checkpoints, newest step first. Each file is read back and its page checksums checked, so the list survives a restart: `checkpoint_id`, `model_id`, `step_index`, `file_path`, `file_size_bytes`, `saved_at_unix_secs` and `verified`. The response also carries `directory`, `training_model_id`, `current_step` and `resumed_from`.

```bash
curl -s http://127.0.0.1:7848/api/v1/training/checkpoints
```

---

### `POST /api/v1/training/resume`
Restores coordinator metadata from a saved reference checkpoint: model profile, step counter, history position, and Merkle root. The next preview uses `step_index + 1`, and later loss history is dropped. Persistent learner/stage weights are not restored. Unknown ids answer `404`; a damaged file or a model this coordinator has no profile for answers `422`, both as `{"error": "..."}`.

```bash
curl -s http://127.0.0.1:7848/api/v1/training/resume \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_id": "ckpt-step0042-1a2b3c4d"}'
```

---

### `GET /api/v1/rl/dream-rsi`
Returns [`DreamRsiApiResponse`](../crates/kadmium-gateway/src/lib.rs), including the sample [`DiscoveryTree`](../crates/kadmium-rl/src/lib.rs) and in-process [`DreamRsiEvolutionReport`](../crates/kadmium-rl/src/lib.rs). `executed_on_replay_nodes` lists node profiles selected as replay-host labels; the replay evaluations run in the coordinator process, not on those labelled SBCs/laptops.

---

### `GET /api/v1/cloud-shield`
Returns [`CloudShieldApiResponse`](../crates/kadmium-gateway/src/lib.rs) with fixed sample arbitrage and failover scenarios. Current sample offers total $1.82/hr versus $13.70/hr (86.7% modelled savings); simulated timings are 340 ms local-to-cloud, 185 ms cloud-to-local/multicloud, and 210 ms failback. These are not live quotes, provisioned capacity, or measured service guarantees.

---

### `GET /api/v1/host/probe`
Probes the gateway host using [`LiveHardwareProfiler::probe_local_host()`](../crates/kadmium-core/src/lib.rs), returning its hardware profile, $q^*$ heuristic, and a `kdm1_...` endpoint descriptor. The ticket has an unkeyed FNV-1a integrity checksum, not a signature; it does not authenticate registration or authorize a trust tier.

---

## 3. Desktop Companion App API (Open-Source $\leftrightarrow$ `kadmium.ai` Cloud)

### `GET /api/v1/companion/status`
Returns [`CompanionAppStatus`](../crates/kadmium-gateway/src/lib.rs) for the system tray / Desktop Companion UI:
```json
{
  "app_version": "v0.1.0-cd48",
  "mode": {
    "SelfHostedOpenSource": {
      "local_coordinator_url": "http://127.0.0.1:7848"
    }
  },
  "local_host_polite_state": "IdleDedicated",
  "gaming_auto_yield_active": true,
  "active_cluster_nodes": 7,
  "active_coordinator_leader": "node-main-pc",
  "current_hourly_cloud_burn_usd": 0.0
}
```

### `POST /api/v1/companion/mode`
Changes the local Companion mode/status value without restarting the node:

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

This endpoint only stores the supplied mode fields. It does not contact the cloud endpoint, establish WAN connectivity, provision capacity, enable failover, or enforce `max_hourly_spend_cap_usd`. Cloud-burn values in Companion status are sample state, not billing records.

---

## 4. Model Context Protocol (`POST /mcp` & `GET /api/v1/mcp`)

### `POST /mcp`
Streamable HTTP JSON-RPC 2.0 endpoint implementing the **Model Context Protocol (`2025-03-26`)** (`initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`).

### `GET /api/v1/mcp`
Returns [`KadmiumMcpManifest`](../crates/kadmium-gateway/src/mcp.rs), including all 7 MCP tool JSON Schemas, 5 resources, 2 prompts, and the `.mcp.json` configuration snippet.

`kadmium mcp` over stdio creates a fresh in-process demonstration state and does not read `KADMIUM_COORDINATOR_URL`. Use `POST /mcp` when tools must operate on a running coordinator's shared state.

---

## 5. Ecosystem Integrations & Observability (`/api/v1/integrations`, `/metrics`, `/api/v1/telemetry/wandb`)

### `GET /api/v1/integrations`
Returns [`EcosystemIntegrationsOverview`](../crates/kadmium-data/src/lib.rs) containing sample descriptors for `hf://`, `gh://`, `oci://`, `s3://`, and `wandb://`. These do not represent verified remote connections. Current `pull` support is HF planning and simulated GitHub evaluation only; no OCI/S3/W&B network connector is implemented.

### `GET /metrics`
Exports live cluster gauges (`kadmium_cluster_epoch`, `kadmium_inference_throughput_tokens_per_sec`, `kadmium_diloco_demo_compression_ratio`, `kadmium_cloud_shield_hourly_burn_usd`, `kadmium_node_fast_memory_gb`) in standard **Prometheus / OpenMetrics (`text/plain; version=0.0.4`)** exposition format.

### `GET /api/v1/telemetry/wandb`
Returns a structured **Weights & Biases (`wandb://`) / OpenTelemetry** run summary JSON object for distributed training, GRPO, and Dream-RSI runs.

---

## 6. Runtime Kernels, Supply Grid & `v1.0 GA` Governance Endpoints

### `GET /api/v1/pipeline/execute`
Executes a synthetic reference forward pass through [`DistributedMoEPipelineExecutor`](../crates/kadmium-runtime/src/lib.rs), returning stage routing, buffer/page metadata, boundary payloads, and modelled draft-acceptance figures. It is not language-model generation or a hardware benchmark.

### `GET /api/v1/runtime/kernels`
Runs [`HardwareKernelDispatcher::run_cluster_kernel_benchmark`](../crates/kadmium-runtime/src/kernels.rs) on a synthetic host-side tensor and labels results with sample node targets. It does not execute every labelled native backend; cross-silicon digest flags and performance figures are reference/model outputs. Native device parity is limited to the separately tested CUDA kernels.

### `GET /api/v1/network/nat`
Returns [`NatTraversalCoordinator`](../crates/kadmium-network/src/nat_traversal.rs) route-planning output and a loopback UDP self-test result. STUN/DERP addresses are descriptor strings; this endpoint does not establish a managed QUIC, DERP, or TURN connection.

### `GET` or `POST /api/v1/network/usbc`
Returns [`UsbCHostDiscoveryReport`](../crates/kadmium-network/src/usbc_interconnect.rs) from the current host/node profiles. `/api/v1/network/usbc-interconnect` is an alias. Scanner ratings and `delta_bridge_plan` values are modelled estimates, not cable benchmarks. Actual multi-lane sessions use `kdbr://` addresses and `KADMIUM_USBC_BRIDGE_BIND`; see the [USB-C delta bridge guide](guides/usbc-delta-bridge.md) for transport behavior and security limits.

### `GET /api/v1/companion/pairing`
Returns sample/discovered `KDMD` LAN pairing candidates and an `OsSwapchainHysteresisEngine` policy timeline. The timeline is not proof that native swapchain, thermal, or battery hooks are active on every platform.

### `GET /api/v1/cloud-shield/watcher`
Returns the [`CloudShieldCutoverWatcher`](../crates/kadmium-sharding/src/cloud_shield_watcher.rs) fixed incident-timeline simulation (340 ms local-to-cloud scenario time, zero dropped requests in the sample, and simulated local failback). It does not monitor or migrate a live service.

### `GET /api/v1/rl/grpo-step`
Runs `GrpoEpochExecutor::run_verified_grpo_step`: eight synthesized programs for a fixed polynomial are executed in the custom fuel-metered [`WasiMicroSandbox`](../crates/kadmium-rl/src/sandbox_grpo.rs), then reference group advantages are calculated. It does not invoke Dream MCTS/PUCT or train a model, and the sandbox is not a general WASI runtime.

### `GET /api/v1/supply/marketplace` & `POST /api/v1/supply/configure`
Inspects or updates the local [`SupplyLeasePolicy`](../crates/kadmium-core/src/governance.rs) and its illustrative credit/USD calculations. These endpoints do not create paid leases, settle funds, or issue redeemable credits.

### `GET /api/v1/ga/governance`
Returns a sample governance bundle:
- `spot_orderbook` & `provisioned_spot_container`: fixed quotes and a constructed container descriptor; no provider API or provisioning occurs.
- `slice_quotas` & `scoped_api_keys`: the sample `SubClusterRbacEngine` catalog, not endpoint authentication or enforced tenant quotas.
- `dream_rsi_world_registry`: a hard-coded world catalog, not a remote public registry or general WASI execution.
- `sample_fraud_proof`: a reference `VerdeFraudProofCertificate`. Its FNV-based digest/MAC is not cryptographic authentication, automatic slashing, or financial settlement.

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
Runs one decode step as a work graph over live workers, with simulated nodes executed in-process on the coordinator, on the placement in force; replicated experts are served by the copy with the lowest measured completion estimate. Returns the shape, graph id, units total and validated, measured wall time, the modelled ms per token for that shape, the final boundary digest, per-unit hop telemetry, credit granted and receipts issued, whether the placement came from the cache, how many routing accesses the usage ledger recorded, with `repin: true` the total re-pin change count (tier swaps plus replica additions/drops) after the step, and `disputes`: every Verde dispute the step settled (see ARCHITECTURE 9.6), each with the unit, the disputants, the referee, the `method` (`layer-bisection` or `whole-unit`), `layers_in_unit`, `digest_comparisons`, `openings_requested` and `openings_verified`, the `divergent_layer`, `referee_layers_run`, the `winner`, the `losers` and the `outcome`. Digests in every response are RepOps digests under the current manifest: a layer boundary's digest covers the manifest id, the layer and the exact INT8 bytes, and a unit's digest is the fold of its layer digests.

### `POST /api/v1/grid/repin`
```json
{ "model": "deepseek-v3-671b-moe-q4k", "max_swaps": 4, "apply": true, "payback": false }
```
One re-pin boundary on the placement in force, with the load balancer in its incremental mode (ARCHITECTURE 9.9): per layer, the hottest streamed expert against the coldest pinned one, swapped when the learned heat clears a 25% hysteresis; then, within whatever of `max_swaps` is left, copies of hot experts onto nodes with free fast memory where a copy lowers the layer's busiest node by more than the same hysteresis. Returns the `swaps`, the `replica_changes` (each with the layer's busiest node before and after, in bytes and modelled ms), the `balance` report for the expert-parallel shape, the staging transfers they imply (a promoted expert into the node's fast tier, pinned, the demoted one out, and each copy from its primary), the placement digest afterwards, the running totals, and `expert_read_ms_before` and `_after`: the cost model's expert reads on the step's critical path at one row, modelled. With `apply: false` it only proposes. With `payback: true` (default `false`, like the boundary a decode step asks for) a swap is proposed only when the extra hits the promoted expert should earn, beyond the demoted one, repay its two transfers against what a fast-tier hit saves over a streamed read: across the mesh's slowest uplink between two nodes, or two memory copies when both experts sit on one node (ARCHITECTURE 9.19, InferSSD's adaptive admission). On a slow link that keeps most swaps streamed; the rationale says how many hotter experts stayed streamed for that reason, and each swap reports its `required_hits`.

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
