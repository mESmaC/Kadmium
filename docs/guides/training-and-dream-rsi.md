# Guide: Distributed Training, RL & Dream-RSI Evolving Worlds

Kadmium goes beyond distributed inference: it enables **fault-tolerant pre-training, LoRA/QLoRA fine-tuning, Disaggregated Reinforcement Learning (GRPO/PPO), and Dream-RSI self-improvement** across home broadband and heterogeneous hardware.

---

## 1. Fault-Tolerant Distributed Training (`Sharded-Island DiLoCo` + `DeMo`)

Standard `AllReduce` data-parallel training stalls completely over 100–300 Mbps home internet or Wi-Fi links. Kadmium combines three breakthroughs in [`kadmium-runtime`](../../crates/kadmium-runtime/src/lib.rs):

### A. Asymmetric 1F1B Pipeline Parallelism + Sharded Islands
When a model (`Llama-3-70B-LoRA` or `DeepSeek-V3`) is larger than any single consumer GPU:
- [`ShardPlanner`](../../crates/kadmium-sharding/src/lib.rs) partitions layers asymmetrically across your LAN devices (`main-gaming-pc` $\to$ `mac-mini-m4-pro` $\to$ `gaming-laptop-rtx4070` $\to$ `friend-remote-pc`).
- [`Pipeline1F1BScheduler`](../../crates/kadmium-runtime/src/lib.rs) schedules **Warmup**, **Steady-State 1F1B (One-Forward-One-Backward)**, and **Cooldown** micro-batches, keeping pipeline bubble overhead under `~19%`.
- Stage-boundary activations and gradients are compressed using **Hivemind 8-Bit Blockwise Dynamic Quantization** ([`BlockwiseInt8Tensor`](../../crates/kadmium-runtime/src/lib.rs)), cutting inter-node bandwidth by $4\times$.

### B. Nous Research `DisTrO` / `DeMo` Frequency-Domain Momentum Compression
Instead of syncing dense gradients on every step, nodes run $H = 50$ local inner steps (`DiLoCo`) and compress their outer momentum updates using **Orthonormal Discrete Cosine Transform (`DCT-II` / `IDCT-III`)**:
1. [`DeMoDctCompressor::step_and_compress`](../../crates/kadmium-runtime/src/lib.rs) transforms the momentum chunk into frequency space and extracts only the **Top-$k$ fastest-moving frequency components**.
2. Unsent frequencies are accumulated locally into a lossless residual buffer (`residual_error[i]`), guaranteeing exact energy conservation:
   $$\hat{m}_t + e_{t+1} = m_t$$
3. Achieves **`142x`–`800x` communication reduction**, shrinking outer-step sync over residential broadband to `<100ms`.

---

## 2. Disaggregated RL (`Actor` / `Environment` / `Learner`)

Reinforcement learning workloads (e.g., GRPO reasoning, tool use, and GPU kernel synthesis) waste up to 70% of GPU cycles if GPUs block waiting for CPU-bound code execution or environment rollouts.

[`RlTopologyPlanner`](../../crates/kadmium-rl/src/lib.rs) disaggregates RL across your hardware tiers:
- **Rollout Actors (`main-gaming-pc`, `mac-mini-m4-pro`):** High-throughput token generation for trajectory rollouts, assisted by speculative draft tokens from `rk3588-sbc-bramble`.
- **Environment Sandbox Workers (`legacy-thinkpad-t480`, `rk3588-sbc-bramble`, `home-nas-vault`):** Execute verifiable reward sandboxes (embedded **Wasmtime WASI** micro-sandboxes for deterministic unit/math/compiler verifiers + OCI container bridge for Gymnasium/Python environments).
- **Policy Learners (`main-gaming-pc` + `gaming-laptop-rtx4070`):** Consume verified rollout trajectories, compute zero-mean unit-variance group advantages ([`compute_grpo_group_advantages`](../../crates/kadmium-rl/src/lib.rs)), and apply `DeMo` + DiLoCo gradient updates.

---

## 3. First-Class Support for Dream-RSI (*Recursive Self-Improvement through Evolving Worlds*)

Kadmium includes native support for [**Dream-RSI**](https://www.dream-rsi.com/) ([`zhengkid/Dream-RSI`](https://github.com/zhengkid/Dream-RSI), arXiv:2609.14858) in [`DreamRsiSimulator`](../../crates/kadmium-rl/src/lib.rs):

### How Dream-RSI Works on Kadmium
1. **Online Discovery Tree Recording:** During online evolutionary search (e.g., optimizing a custom CUDA/Triton/CubeCL kernel from a `16.8 GB/s` baseline to `123.5 GB/s` peak), Kadmium records every parent-child mutation, evaluator score, and compute cost into a persistent [`DiscoveryTree`](../../crates/kadmium-rl/src/lib.rs) on your Home NAS.
2. **Zero-GPU Offline Causal Replay ("Dreaming"):**
   - While your GPUs are busy or while you are gaming, Kadmium schedules **offline causal replay simulations** onto your **ARM/RISC-V SBCs (`rk3588-sbc-bramble`), Legacy Laptops (`legacy-thinkpad-t480`), and idle Home NAS CPU threads**.
   - Why can low-power CPUs run this? Because in a recorded `DiscoveryTree`, the expensive GPU kernel benchmarks at each tree node have *already been evaluated and cached*! Replaying counterfactual exploration policies ($\alpha$ priority weight, $\tau_{\text{prune}}$ pruning ratio, frontier width $W$) is pure causal graph traversal (`0` GPU hours required).
3. **Self-Evolving Meta-Policy Injection:**
   - `DreamRsiSimulator::evolve_meta_policy` identifies the exploration policy that reaches the global peak (`123.5 GB/s`) while pruning dead-end regressions—achieving **`71.4%` compute savings** (`2.8x` discovery speedup)—and automatically injects the evolved meta-policy into the next online RL epoch.

Run the standalone Dream-RSI simulator anytime via:
```bash
kadmium dream
```

---

## 4. TPU / Compute-in-Memory (CIM) Inspired Dataflows

Different phases of inference and RL have radically different arithmetic intensities. [`DataflowStreamPlanner`](../../crates/kadmium-sharding/src/lib.rs) lets you inspect and switch between three execution dataflows:

1. **`WeightStationaryTokenStreaming` (Google TPU / CIM Systolic Mode):**
   - Keeps `.ktb` weight banks stationary in SRAM/VRAM/Unified Memory and streams micro-batches of tokens ($B \ge 64$) through the stationary weights, amortizing memory reads by $B\times$.
2. **`TokenStationaryWeightStreaming` (Colibri / FreeToken Edge Mode):**
   - Keeps the active token's KV-cache and hidden state stationary in L2/VRAM while streaming `64 KiB` page-aligned `.ktb` weight/expert banks dynamically from NVMe/DDR5 (`B = 1..8` low-latency chat).
3. **`HybridCimAdaptive` (Default Recommended):**
   - Automatically runs `WeightStationaryTokenStreaming` on high-$q^*$ nodes (`mac-mini-m4-pro`, `main-gaming-pc`) and `TokenStationaryWeightStreaming` on NVMe/CPU edge nodes.

Compare all three modes across your cluster with:
```bash
kadmium dataflow --batch-tokens 512
```
