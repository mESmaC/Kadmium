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

### B. Stateful DiLoCo rounds

[`DiLoCoSession`](../../crates/kadmium-runtime/src/diloco.rs) implements a synchronous outer training loop inspired by [OpenDiLoCo's pinned PyTorch implementation](https://github.com/PrimeIntellect-ai/OpenDiloco/blob/2d750e58a692ce1424d2a2366b2b3de1f42c9bf1/open_diloco/train_diloco_torch.py#L336-L353):

1. All islands begin from the same committed parameter snapshot. Each island contains the same complete model layout; a shard is an identified parameter partition within that replica, not a different model layer contributed by another island.
2. Each island takes exactly `H` local optimizer steps on its own data. Local AdamW moments survive outer synchronization.
3. Islands submit their final parameter shards with the round number and the snapshot's `baseline_id`. The coordinator waits for every registered island, rejecting unknown/duplicate islands, stale rounds, mismatched baselines/layouts and invalid values.
4. For master parameters `theta` and each locally trained replica `theta_i`, the outer pseudo-gradient is the equally weighted mean of `theta - theta_i`. A persistent Nesterov momentum optimizer updates the master once; the new snapshot starts the next round.

The barrier is synchronous even when submissions arrive at different times. A missing island prevents the round from committing. This implementation does not include OpenDiLoCo's networking or membership recovery, async reinforcement learning, or an inference acceleration technique. It adds an independent session API; existing generic training and RL endpoints do not automatically use this stateful loop.

Run the integrated reference workload:

```bash
cargo run -p kadmium-cli -- diloco --inner-steps 20 --rounds 25 --islands 3 --json
```

This trains a real small CPU regression objective (`y = 2*x + 1`) with different data ranges per island, two parameter shards, persistent local AdamW state, and an outer Nesterov update. The report includes loss history and 25 synchronizations versus a 500-sync per-step baseline for the same local step count. Payload numbers count FP32 parameters uploaded and broadcast; they exclude protocol overhead and are not measured traffic. This proves the reference training path and cadence, not LLM convergence, GPU throughput or WAN performance.

The [session HTTP API](../api-reference.md#stateful-diloco-sessions) exposes create, submit, status, committed snapshot and checkpoint/restore. Session checkpoints preserve master parameters, outer momentum and pending submissions. External learners must separately save their own model, local optimizer, data cursor and random state for a complete training restart. A committed snapshot can be consumed by a future inference adapter; no current inference model is automatically replaced.

### C. DeMo frequency-domain compression

[`DeMoDctCompressor::step_and_compress`](../../crates/kadmium-runtime/src/lib.rs) is a separate compression primitive: it transforms momentum chunks using DCT-II, selects top-k frequency components and retains unsent residuals. The new DiLoCo session API transports full FP32 parameter shards and does not enable this compressor. Communication reductions from sparse compression and from taking `H` local steps must be measured separately before combining claims.

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
