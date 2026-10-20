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
  serve         Launch the Unified Web Control Dashboard & OpenAI/Anthropic compatible AI Gateway
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

Starts the **Kadmium Control Plane Dashboard** and **Strata-compatible OpenAI/Anthropic AI Gateway**.

```bash
kadmium serve --port 7848
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p`, `--port <PORT>` | `7848` | TCP port for the web dashboard, REST control plane, and `/v1/chat/completions` + `/v1/messages` endpoints. |

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
```

---

## 13. `kadmium mcp`

Starts the **Kadmium Model Context Protocol (`kadmium-mcp`) Server** over `stdio` for AI coding agents (Claude Code, Cursor, Antigravity, Windsurf, Zed), or prints the full MCP tool/resource/prompt catalog with `--inspect`:

```bash
kadmium mcp --inspect
kadmium mcp
```
