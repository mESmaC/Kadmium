# Kadmium CLI Command Reference

The `kadmium` CLI (`crates/kadmium-cli`) is a single-binary control tool, hardware profiler, simulator, and HTTP/QUIC coordinator for the **Kadmium (`Cd-48`)** compute fabric.

```text
Usage: kadmium [COMMAND]

Commands:
  simulate      Run the full end-to-end heterogeneous cluster & live responsibility shift simulation
  probe         Probe and benchmark THIS local host machine (CPU, GPU, RAM, 64KB page-aligned bandwidth, and Join Ticket)
  dream         Run a standalone Dream-RSI (Recursive Self-Improvement through Evolving Worlds) offline replay simulation
  slice         Partition a large node pool (e.g., 100 nodes) into isolated sub-compute environments (Prod, Test, Train, RL)
  dataflow      Evaluate TPU / Compute-in-Memory (CIM) inspired Dynamic Token Streaming vs. Dynamic Weight Streaming
  cloud-shield  Evaluate Kadmium Cloud Business Model: Multi-Datacenter Spot Arbitrage + Bidirectional Downtime Protection
  plan          Compute and print the asymmetric sharding plan for a target workload
  launch        Inspect setup, create a missing configuration and start the existing coordinator
  serve         Launch the Control Room and reference API diagnostics
  desktop       Open the Kadmium desktop app (the Companion in its own window) with a local coordinator
```

---

## 1. `kadmium probe`

Probes the local machine's CPU cores, system RAM, GPU VRAM (`nvidia-smi` / Metal / Vulkan), and benchmarks `64 KiB` page-aligned memory/NVMe streaming throughput, then outputs a `kdm1_...` Join Ticket and runs a live local Tokio P2P `WireFrame` roundtrip self-test.

```bash
kadmium probe
# Or via Cargo:
cargo run -p kadmium-cli -- probe
```

**Example Output Highlights:**
- **Detected Hostname & OS/Arch:** e.g., `DESKTOP-F1M5LLE (windows/x86_64)`
- **Compute Backend:** `Cuda` / `MetalUnified` / `CpuAvx512OrAvx2`
- **FreeToken $q^*$ Ratio:** Bandwidth-adaptive split ratio between VRAM and system RAM.
- **Generated Join Ticket:** `kdm1_...` (with 64-bit FNV-1a MAC signature).

---

## 2. `kadmium serve`

Starts the **Kadmium Control Room** and existing reference API diagnostics. The `/v1` message routes return fixed diagnostic text and telemetry; they do not provide language-model generation or full OpenAI/Anthropic SDK compatibility.

```bash
kadmium serve --port 7848
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p`, `--port <PORT>` | `7848` | TCP port for the web dashboard, REST control plane, and `/v1/chat/completions` + `/v1/messages` endpoints. |
| `-c`, `--config <PATH>` | none | Read a workspace file using the existing tolerant parser. |
| `--demo` | off | Include the simulated seven-node fleet alongside live hardware. |
| `--spawn-fleet` | off | Start the existing worker test fleet. |
| `--worker-port <PORT>` | `7850` | Preferred built-in worker TCP port; `0` asks the OS for a free port. |

HTTP and worker listeners use all interfaces. `serve` behavior, its defaults and its parser are unchanged by the additive setup command. Configuration does not override the listener, host capabilities or active model; the existing coordinator consumes CloudShield settings and the workspace name for that mode. There are no new backend URL, key or inference-relay settings.

---

## 2a. `kadmium launch`

Probes this host, prints an informational planning recommendation, preserves an existing configuration or exclusively creates a missing one, then starts the existing coordinator. `--plan` performs only the probe and prints the plan: no files, listeners, bridge discovery or downloads.

```bash
kadmium launch --plan
kadmium launch --config kadmium.toml --port 7848
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--plan` | off | Print a human-readable setup plan and exit. |
| `-c`, `--config <PATH>` | `kadmium.toml` | Preserve an existing readable file; create a missing file without overwriting. |
| `-p`, `--port <PORT>` | `7848` | HTTP port, listening on all interfaces. |
| `--demo` | off | Include the simulated seven-node fleet. |
| `--spawn-fleet` | off | Start the existing worker test fleet. |
| `--worker-port <PORT>` | `7850` | Preferred built-in worker TCP port; `0` asks for a free port. |

The generated recommendation does not fix model selection, prove model fit or load weights; the coordinator recommends again at startup. Existing configuration bytes are left unchanged. See the [launch guide](guides/cli.md) for the dependency decision and network/configuration limitations. `init`, `serve` and `desktop` retain their existing behavior.

---

## 3. `kadmium plan`

Computes and prints the optimal asymmetric layer and MoE expert assignment across your heterogeneous cluster for a specific workload mode (`inference` or `training`).

```bash
# Plan DeepSeek-V3-671B-MoE inference across the cluster:
kadmium plan --workload inference

# Plan Llama-3-70B LoRA/DiLoCo distributed training across the cluster:
kadmium plan --workload training
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-w`, `--workload <MODE>` | `inference` | Target workload (`inference` or `training`). |

After the shard plan it prints the grid execution shape: both modelled estimates, the placement, the usage it learned from a coordinator's ledger when `target/kadmium-state/usage/<model>.usage.json` exists, every calibration record kept for the model, and the offers and fit table (fast memory, RAM and NVMe used against offered per node, and one line per shortfall).

---

## 3a. `kadmium calibrate`

Asks a running coordinator to measure the placement in force: up to 8 decode steps of one shape within a 20 second budget, recorded as ms per token bound to the placement digest, node set and versions that ran.

```bash
kadmium calibrate --coordinator http://127.0.0.1:7848 --strategy expert-parallel --steps 8
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-c`, `--coordinator <URL>` | `http://127.0.0.1:7848` | Coordinator HTTP gateway. |
| `-s`, `--strategy <SHAPE>` | planner's choice | `chain` or `expert-parallel`. |
| `--steps <N>` | `8` | Decode steps to run, at most 8. |
| `--rows <N>` | `1` | Rows per step. |

---

## 4. `kadmium simulate`

Runs the comprehensive 5-phase end-to-end fabric simulation on the terminal:
- **Phase 1:** Hardware Discovery, FreeToken $q^*$ Bandwidth Ratios, and Asymmetric MoE Sharding (`DeepSeek-V3-671B-MoE`, `61` layers, `256` routed experts/layer).
- **Phase 2:** CERN FTS3 Third-Party P2P Data Staging directly from the `home-nas-vault` seed node (`0.00 GB` coordinator relay bottleneck).
- **Phase 3:** Live Responsibility Shift (`Gaming Mode` `<200ms` Polite Auto-Yield on the main PC, followed by a `Cloud Burst GPU` join event).
- **Phase 4:** Fault-Tolerant Distributed Training (`Sharded-Island DiLoCo` + `Nous DisTrO/DeMo` DCT momentum compression + `Gensyn Verde` refereed verification).
- **Phase 5:** Disaggregated RL (`Actor` / `Environment` / `Learner`) + **Dream-RSI** (*Recursive Self-Improvement through Evolving Worlds*).

```bash
kadmium simulate
```

---

## 5. `kadmium slice`

Partitions a large pool of heterogeneous compute nodes into isolated sub-cluster environments (`ProductionDeployment`, `TestingAndStaging`, `DistributedTraining`, `RlAndDreamRsi`, and `BatchDataProcessing`), each with its own independent shard plan and `isolated_network_namespace`.

```bash
# Partition 100 nodes into 5 isolated sub-clusters of 20 nodes each:
kadmium slice --pool-size 100 --slices 5
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--pool-size <N>` | `100` | Total number of heterogeneous nodes in the master pool. |
| `--slices <K>` | `5` | Number of isolated sub-cluster slices to provision. |

---

## 6. `kadmium dataflow`

Evaluates Google TPU / Compute-in-Memory (CIM) inspired execution dataflows across your cluster for a given micro-batch token count:
1. **`WeightStationaryTokenStreaming` (TPU/CIM Systolic Mode):** Weights stay pinned in SRAM/VRAM/Unified RAM while micro-batches of tokens stream continuously through the stationary weight banks.
2. **`TokenStationaryWeightStreaming` (Colibri/FreeToken Edge Mode):** Activations stay pinned in L2/VRAM while `.ktb` weight banks stream dynamically from NVMe/DDR5 (`64 KiB` page-aligned).
3. **`HybridCimAdaptive` (Kadmium Optimal):** Assigns Weight-Stationary Token Streaming to high-$q^*$ nodes (Main GPU, Mac Mini) and Dynamic Weight Streaming to NVMe/CPU edge nodes.

```bash
kadmium dataflow --batch-tokens 512
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-b`, `--batch-tokens <TOKENS>` | `512` | Active token micro-batch size used to find the roofline crossover point. |

---

## 7. `kadmium dream`

Runs a standalone **Dream-RSI** ([*Recursive Self-Improvement through Evolving Worlds*](https://www.dream-rsi.com/), arXiv:2609.14858) offline causal replay simulation over recorded discovery trees (`DiscoveryTree`) using low-power CPU/SBC/NAS threads (`0` GPU hours required).

```bash
kadmium dream
```

Outputs a comparative table of candidate exploration policies ($\alpha$ priority weight, $\tau_{\text{prune}}$ pruning threshold, frontier width $W$), showing **compute savings %** to reach the global peak reward (`123.5 GB/s` GPU kernel throughput) and the synthesized meta-prompt injected into the next online RL epoch.

---

## 8. `kadmium cloud-shield`

Evaluates the **`kadmium.ai` Managed Hybrid Cloud** features:
- **Multi-Datacenter Spot Arbitrage:** Splits layers across local hardware (`$0.00/hr`) + verified spot GPU providers (Vast.ai, RunPod, Lambda, AWS, GCP) and compares hourly cost against single-cloud on-demand pricing.
- **Bidirectional Downtime Protection (`CloudShield`):** Simulates both **Local $\to$ Cloud** failover (e.g., home power/ISP outage) and **Cloud $\to$ Local** failover (e.g., spot GPU preemption preemption notice), reporting hot-standby FTS3 sync overhead and `<400ms` cutover latency.

```bash
kadmium cloud-shield
```

---

## 9. `kadmium init`

Probes the local host hardware (`VRAM`, `RAM`, `NVMe BW`, $q^*$) and generates a `kadmium.toml` workspace configuration manifest:

```bash
kadmium init --out kadmium.toml
```

---

## 10. `kadmium compile` & `kadmium inspect`

Compiles a contiguous layer range into a 64 KiB page-aligned `.ktb` (Kadmium Tensor Bank) binary archive, computes its `KtbMerkleTree` root digest, and inspects/verifies every 64 KiB page:

```bash
kadmium compile --model deepseek-v3-671b-moe --start-layer 0 --end-layer 3 --out shard.ktb
kadmium inspect shard.ktb
```

---

## 11. `kadmium pull`

Stages selective layer-range shards from Hugging Face Hub (`hf://`) or mounts commit-pinned RL / Dream-RSI repositories from GitHub (`gh://`):

```bash
kadmium pull hf://deepseek-ai/DeepSeek-V3@main
kadmium pull gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt
```

---

## 12. `kadmium join`

Verifies an HMAC-SHA256 `kdm1_...` Join Ticket, broadcasts a UDP LAN discovery beacon (`KDMD`), and attaches the local host to a cluster slice:

```bash
kadmium join --ticket kdm1_...
kadmium join --offer-ram-gb 24 --offer-fast-gb 10 --slots 2 --require-approval
```

The worker measures its memory streaming rate before registering (a single-thread copy, which replaces the nominal RAM figure in the coordinator's cost model for this node) and sends the owner's offer with every registration and heartbeat.

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--offer-fast-gb <GiB>` | 90% of VRAM plus all unified memory | VRAM or unified memory offered to the grid. |
| `--offer-ram-gb <GiB>` | 70% of RAM, never within the reserve | System RAM offered; the reserve (a quarter of RAM, bounded to 1 to 4 GiB) is never offered. |
| `--offer-nvme-gb <GiB>` | the whole Kadmium cache | NVMe offered for cold weight banks. |
| `--slots <N>` | `4` | Work instances this machine runs at once. |
| `--require-approval` | off for owned hardware | Review placement plans before anything is staged here. |

---

## 13. `kadmium mcp`

Starts the **Kadmium Model Context Protocol (`kadmium-mcp`) Server** over `stdio` for AI coding agents (Claude Code, Cursor, Antigravity, Windsurf, Zed), or prints the full MCP tool/resource/prompt catalog with `--inspect`:

```bash
kadmium mcp --inspect
kadmium mcp
```

---

## 14. `kadmium lease`

Configures and simulates the **Kadmium Supply Grid** on your local nodes, comparing direct USD payout rates against **1.25× Kadmium Cloud compute credits** while enforcing middle-layer-only privacy ($1 \dots N-2$) and `< 200 ms` gaming auto-eviction:

```bash
kadmium lease --hours 8 --credits true
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--hours <HOURS>` | `8.0` | Daily idle lease hours to simulate per node. |
| `--credits <BOOL>` | `true` | Choose `1.25×` Kadmium Cloud credits (`true`) or direct USD payout (`false`). |

---

## 15. `kadmium bench`

Executes the native hardware kernel suite (`FusedRmsNormRopeKernel` $\to$ `PagedGqaAttentionKernel` $\to$ `FusedSwiGluExpertKernel`), prints the active vendor intrinsic target (`CUDA PTX`, `ROCm HIP`, `Metal SIMD-group`, `Vulkan Subgroup`, `AVX-512 FMA`, `ARM NEON DotProd`), verifies deterministic `Verde RepOps` digest parity across nodes, and probes the UDP hole-punch + DERP relay NAT mesh:

```bash
kadmium bench
```

---

## 16. `kadmium pairing`

Runs the **Zero-Click LAN Pairing** (`ZeroClickLanPairingEngine` with 6-digit SAS verification codes), **OS Swapchain/Thermal Hysteresis** (`DXGI` / `IOKit` / `DRM` `< 200 ms` fast-yield + `15 s` anti-thrash cooldown), and **Automated `CloudShield` Cutover Watcher** (`380 ms` RTO, `0` dropped requests):

```bash
kadmium pairing
```

---

## 17. `kadmium registry`

Inspects the **`v1.0 GA` Multi-Provider Spot Orderbook**, **Sub-Cluster RBAC & Scoped API Keys (`kdm_live_...`)**, **Cryptographic `Verde` $O(\log_2 L)$ Fraud-Proof Certificates**, and the **Public `Dream-RSI` World Registry**:

```bash
kadmium registry
```

---

## 18. `kadmium desktop`

Opens the **desktop app** (`kadmium-desktop`): the Companion in its own native window, with a local coordinator running inside the app. If a coordinator already answers on the port (for example `kadmium serve`), the window attaches to it instead. Closing the window ends the command.

```bash
cargo build -p kadmium-desktop   # once, so the app sits next to the kadmium binary
kadmium desktop --control-room
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p`, `--port <PORT>` | `7848` | Coordinator port to attach to or start on. |
| `--demo` | off | Start the coordinator with the 7-node simulated demo fleet alongside live hardware. |
| `--control-room` | off | Also open the Control Room in your default browser. |
| `--check-only` | off | Print the hardware summary and where the desktop app was found, then exit. |

The app is looked up next to the `kadmium` binary first, then on `PATH`. If it is not built, `kadmium desktop` says so and opens the Companion in your browser. You can also start the app directly: `kadmium-desktop --port 7848`.
