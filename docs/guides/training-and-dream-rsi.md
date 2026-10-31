# Guide: Distributed Training, RL & Dream-RSI Reference Workloads

Kadmium provides reference training and RL mechanisms plus a bounded stateful DiLoCo outer loop for small complete replicas. Pipeline-sharded large-model training and online policy learning are target designs, not current integrated capabilities.

---

## 1. Training mechanisms

### A. Pipeline planning and stateless stage previews

[`ShardPlanner`](../../crates/kadmium-sharding/src/lib.rs) assigns layer ranges, [`Pipeline1F1BScheduler`](../../crates/kadmium-runtime/src/lib.rs) models warmup/steady-state/cooldown, and [`BlockwiseInt8Tensor`](../../crates/kadmium-runtime/src/lib.rs) provides boundary quantization. In the [target sharded-island design](../ARCHITECTURE.md#53-mode-c-sharded-island-diloco-target-architecture), devices inside one island would collectively hold a full replica.

The current `/api/v1/training/step` and grid training endpoints run independent synthetic stage previews. Each request resets the stage's model state. Their summaries report `execution_mode: "stateless_per_stage_gradient_preview"`; counters and loss history do not make them persistent learners or synchronized replicas.

### B. Stateful DiLoCo rounds

[`DiLoCoSession`](../../crates/kadmium-runtime/src/diloco.rs) implements a synchronous outer loop inspired by [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiloco):

1. Every island starts from the same committed snapshot and contains the same complete model layout. A session shard is a named parameter partition within each replica, not a pipeline stage supplied by a different island.
2. Each caller performs exactly `H` local optimizer steps and owns its local optimizer state.
3. Islands submit final values with the round and `baseline_id`. The coordinator rejects stale rounds, mismatched layouts, duplicate submissions, and invalid values.
4. After all islands submit, the session forms FP32 master-minus-local pseudo-gradients, averages them with the RepOps tree reduction, and applies the shared Nesterov outer update with persistent momentum.

The barrier is synchronous: a missing island prevents commitment. The session does not provide pipeline execution, membership recovery, or automatic local learner checkpoints.

Run the integrated two-parameter CPU regression example:

```bash
cargo run -p kadmium-cli -- diloco --inner-steps 20 --rounds 25 --islands 3 --json
```

This uses nonidentical data ranges, persistent local AdamW state, and the shared outer optimizer. Its payload estimate excludes protocol overhead and is not measured network traffic. It demonstrates the reference cadence, not LLM convergence or GPU/WAN performance.

The [session HTTP API](../api-reference.md#stateful-diloco-sessions) exposes creation, updates, status, snapshots, and checkpoint/restore. Session checkpoints preserve master parameters, outer momentum, and pending submissions; callers must separately save local optimizer, data, and RNG state. Sessions do not automatically update an inference model.

### C. DeMo compression

[`DeMoDctCompressor::step_and_compress`](../../crates/kadmium-runtime/src/lib.rs) transforms momentum chunks with DCT-II, selects top-k components, and retains unsent residuals. Its ratio depends on tensor size and `top_k`. The DiLoCo session transports full FP32 shards and does not automatically enable this compressor. No universal compression ratio or convergence guarantee is established by the primitive.

## 2. RL topology and sandbox reference

[`RlTopologyPlanner`](../../crates/kadmium-rl/src/lib.rs) produces actor/environment/learner placement plans. The current `GrpoEpochExecutor` verifies eight synthesized programs for a fixed polynomial and computes normalized group advantages. `WasiMicroSandbox` is a custom fuel-metered integer stack machine, despite its historical name; it is not a WebAssembly/WASI runtime.

These paths do not generate model rollouts, execute arbitrary Rust/CUDA/Triton code, run PPO, or update a language model. Wasmtime/OCI execution and distributed online RL remain target integrations.

## 3. Dream-RSI offline replay

[`DreamRsiSimulator`](../../crates/kadmium-rl/src/lib.rs) replays an **in-memory sample discovery tree** with counterfactual exploration policies. Because node scores are already present, replay is CPU graph traversal and needs no GPU evaluation.

```bash
kadmium dream
```

`run_dream_cycle` reports normalized scores and evaluation costs. In the current sample, the promoted policy reaches the **2.31× peak** after **4 of 9 unit-cost evaluations**, a **55.6% evaluation-cost saving** relative to the baseline discovery order.

The CLI does not record a persistent tree on a NAS, benchmark a real GPU kernel, synthesize or inject a meta-prompt, or modify a subsequent online RL epoch. The figures describe only the bundled sample.

## 4. Dataflow comparison

[`DataflowStreamPlanner`](../../crates/kadmium-sharding/src/lib.rs) reports four planning variants:

- `AdaptiveTieredHybrid`
- `ComputeInMemoryTokenStreaming`
- `TokenStationaryWeightStreaming`
- `DisaggregatedDualStream`

For the current report, low-batch inference favors compute-in-memory token streaming, while high-batch inference favors token-stationary weight streaming. These are modelled planner recommendations, not measurements of a loaded model.

```bash
kadmium dataflow --model deepseek-v3-671b-moe --batch-tokens 512
```
