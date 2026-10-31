# Kadmium (`Cd-48`) Measured Benchmark Ledger (`2026-10`)

> **Commercial-use qualification — 2026-10-30:** Run 1 is a reported external-harness measurement with the configuration/provenance caveats below, not a reproducible shipped-product benchmark. Previous business/launch descriptions of Q4_K_M, 19.6 GiB, a 16 GB PC, wired 1GbE and zero swap did not match this ledger and have been withdrawn. Do not advertise 3.98× as an equal-condition production speedup. Runs 2–3 measure reference kernels on loopback. See the [evidence review](MARKETING_BUSINESS_REVIEW_2026-10-30.md).

> **Process Invariant (Measured Figures Replace Modelled Ones As They Land):**
> [`docs/NUMBERS_2026-10.md`](NUMBERS_2026-10.md) and the R4 review ([`docs/MARKET_RESEARCH_REVIEW_2026-10.md`](MARKET_RESEARCH_REVIEW_2026-10.md) section 10) label every throughput and ROI figure **Modelled** until measured logs exist. This ledger is where measured runs are recorded. Each run is dated, names the hardware and configuration it was taken on, and is labelled **Measured**. Anything derived from a run by arithmetic is labelled **Derived**, and anything projected from a run is labelled **Modelled**.
>
> - **Ledger Month:** `2026-10`
> - **First Entry:** Run 1, `2026-10-22`, `ornith-1.5-35B-A3B` across an RTX 5070 PC and an M4 Mac mini
> - **Repository State When Recorded:** `main` at `405e023` (`2026-10-23`)
> - **Companion Ledgers:** [`docs/NUMBERS_2026-10.md`](NUMBERS_2026-10.md) (market, energy and BOM constants), [`docs/MARKET_RESEARCH_REVIEW_2026-10.md`](MARKET_RESEARCH_REVIEW_2026-10.md) section 3 (modelled capability envelope)

---

## Run 1: `ornith-1.5-35B-A3B`, two-stage pipeline across a PC and an M4 Mac mini (`2026-10-22`)

**Status: Measured. First run, unoptimized baseline.** The run establishes the numbers to beat. It does not replace the modelled capability envelope, because the configuration below is far from the one the envelope assumes: the checkpoint is unquantized, the PC stage runs on the CPU, and the hop between machines is JSON over HTTP on Wi-Fi.

### 1.1 Model

| Field | Value |
| :--- | :--- |
| Checkpoint | `ornith-1.5-35B-A3B` at `X:\ornith-1.5-35B-A3B` on the PC |
| Architecture class | `Qwen3_5MoeForConditionalGeneration` |
| Files | 16 `.safetensors` shards, `71.90 GB`, unquantized `BF16` |
| Layers | 40: 30 `Qwen3_5MoeGatedDeltaNet` recurrent linear-attention layers and 10 `Qwen3_5MoeAttention` full-GQA layers |
| Experts | 256 per layer, top-8 routing plus 1 shared expert |
| Tokenizer | BPE; `<\|im_end\|>` is token id `248046` |

### 1.2 Hardware and configuration

| Node | Role in this run | Hardware | Execution path |
| :--- | :--- | :--- | :--- |
| `DEFINITELYNOTAI` (Windows PC) | Stage 0: layers `0..24`, embedding, `conv1d` and GatedDeltaNet recurrent state (`32 x 128 x 128` per layer), full-GQA KV cache | RTX 5070 `12 GB` GDDR7 (`672 GB/s`, present but **not used** by this harness), `32 GB` DDR5, checkpoint on `X:\` NVMe | PyTorch `torch 2.14.0+cpu` (CPU only), zero-copy `mmap` over all 16 shards; a separate Rust path on 20 AVX2 threads for the steady-state line in 1.5 |
| `apple-m4-mac-mini` (`192.168.40.237:7860`) | Stage 1: layers `25..39` and `final_norm` | Apple M4, 10 CPU cores, `16 GB` unified memory as stated in the run report (unverified, see 1.7), `120 GB/s` spec | `ornith_stage1_real.ktb` executed by `m4_ornith_stage1_forward` in `kadmium_m4_metal_live.m`: Grand Central Dispatch across 10 cores, ARM64 NEON `BF16 -> FP32` FMA on the CPU, served by `kadmium_m4_daemon_live.py` |
| Link | One hidden-state vector per hop, each direction | LAN `192.168.40.0/24`; the run report attributes hop variance to Wi-Fi | JSON over HTTP to the Python daemon (`exec_ms`, `streamed_gbps`, `routed_experts`), not the `KDMF` binary frames in `kadmium-network` |

Pipeline shape: two stages in series, one request at a time. A decode token costs stage 0 plus the hop plus stage 1; there is no overlap between stages.

Harness provenance: `run_ornith_live_generation.py` and `kadmium_m4_daemon_live.py` / `kadmium_m4_metal_live.m` live in an Antigravity scratch directory on the PC. The Rust evaluation test `test_live_ornith_1_5_35b_a3b_real_checkpoint_cross_device_evaluation` lives in that machine's checkout of `crates/kadmium-runtime/src/lib.rs`. None of these were in `origin/main` on `2026-10-23`.

### 1.3 Prompt 1, code generation (Measured)

- **Prompt:** `Write a one-line Python lambda that squares a number x.` (25 prompt tokens)
- **Generated:** 13 tokens to `<|im_end|>`, decoding to a fenced block containing `lambda x: x ** 2`

| Step | Phase | Token id | Piece | Stage 0 PC `L00..L24` (ms) | Hop (ms) | Stage 1 M4 `L25..L39` (ms) | M4 stream (GB/s) | Layer 25 top-8 experts |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 00 | prefill, 25 tok | 71093 | `` '```' `` | 2985.73 | 221.18 | 47.94 | 33.8 | 32, 134, 182, 93, 154, 89, 108, 138 |
| 01 | decode | 12305 | `'python'` | 196.52 | 114.32 | 44.64 | 36.3 | 134, 182, 32, 103, 93, 112, 220, 202 |
| 02 | decode | 198 | `'\n'` | 242.20 | 6.56 | 45.86 | 35.3 | 112, 45, 138, 212, 182, 11, 32, 161 |
| 03 | decode | 12564 | `'lambda'` | 245.18 | 30.63 | 45.70 | 35.5 | 45, 112, 212, 138, 3, 193, 161, 4 |
| 04 | decode | 830 | `' x'` | 324.38 | 22.90 | 46.99 | 34.5 | 45, 212, 180, 3, 161, 227, 254, 39 |
| 05 | decode | 25 | `':'` | 303.01 | 88.62 | 46.67 | 34.7 | 45, 212, 24, 233, 83, 132, 13, 70 |
| 06 | decode | 830 | `' x'` | 332.40 | 107.11 | 45.88 | 35.3 | 45, 254, 3, 161, 193, 59, 128, 132 |
| 07 | decode | 2972 | `' **'` | 340.04 | 113.86 | 48.43 | 33.5 | 193, 233, 132, 70, 45, 24, 115, 163 |
| 08 | decode | 220 | `' '` | 205.02 | 61.90 | 44.94 | 36.0 | 45, 83, 0, 189, 120, 161, 162, 56 |
| 09 | decode | 17 | `'2'` | 314.38 | 110.86 | 47.23 | 34.3 | 45, 52, 93, 15, 145, 86, 219, 120 |
| 10 | decode | 198 | `'\n'` | 281.42 | 35.79 | 45.54 | 35.6 | 193, 71, 45, 70, 188, 161, 152, 2 |
| 11 | decode | 71093 | `` '```' `` | 327.74 | 40.98 | 46.14 | 35.1 | 222, 145, 45, 102, 120, 250, 219, 226 |
| 12 | decode | 248046 | `'<\|im_end\|>'` | 352.11 | 62.78 | 56.21 | 28.8 | 182, 134, 32, 147, 152, 43, 130, 93 |

Prompt 1 summary (Measured, as reported; the per-step rows reproduce these to rounding):

| Metric | Value | Split |
| :--- | ---: | :--- |
| Time to first token, 25-token prefill | `3,254.86 ms` | `130.19 ms` per prompt token |
| Mean decode time per output token | `402.08 ms` (`2.49 tok/s`) | Stage 0 `288.70 ms`, hop `66.36 ms`, stage 1 `47.02 ms` at `34.6 GB/s` sustained |

### 1.4 Prompt 2, systems explanation (Measured)

- **Prompt:** `In one concise sentence, why split a 35B MoE model across a PC and a Mac mini?` (35 prompt tokens)
- **Generated:** 43 tokens to `<|im_end|>`: `To combine the PC's larger GPU VRAM with the Mac's unified memory pool, giving you enough total fast memory to fit a 35B MoE model that wouldn't fit on either machine alone.`

Rows spanning several steps are aggregates as reported (mean or p50 as marked); their expert list is one representative step.

| Step | Phase | Token id | Piece | Stage 0 PC `L00..L24` (ms) | Hop (ms) | Stage 1 M4 `L25..L39` (ms) | M4 stream (GB/s) | Layer 25 top-8 experts |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 00 | prefill, 35 tok | 1206 | `'To'` | 12901.79 | 38.56 | 46.36 | 35.0 | 134, 182, 112, 32, 108, 138, 154, 220 |
| 01 | decode | 15491 | `' combine'` | 489.10 | 85.84 | 47.51 | 34.1 | 28, 29, 166, 217, 140, 40, 51, 98 |
| 02 | decode | 279 | `' the'` | 360.08 | 153.17 | 47.64 | 34.0 | 211, 40, 28, 51, 29, 24, 224, 37 |
| 03 | decode | 6464 | `' PC'` | 338.45 | 24.90 | 45.25 | 35.8 | 211, 164, 40, 98, 217, 245, 46, 24 |
| 04 | decode | 579 | `"'s"` | 379.50 | 65.43 | 45.37 | 35.7 | 211, 230, 98, 40, 148, 13, 252, 24 |
| 05 | decode | 7875 | `' larger'` | 308.03 | 72.44 | 45.68 | 35.5 | 211, 40, 98, 164, 245, 174, 252, 24 |
| 06 | decode | 21966 | `' GPU'` | 274.94 | 95.34 | 48.15 | 33.6 | 40, 174, 211, 24, 98, 217, 245, 7 |
| 07 | decode | 18588 | `' VR'` | 270.65 | 87.70 | 45.52 | 35.6 | 40, 174, 211, 98, 0, 217, 222, 24 |
| 08 | decode | 1354 | `'AM'` | 249.35 | 94.02 | 44.74 | 36.2 | 98, 174, 40, 0, 117, 211, 101, 24 |
| 09 | decode | 440 | `' with'` | 246.66 | 61.11 | 47.31 | 34.2 | 211, 148, 101, 192, 19, 140, 91, 129 |
| 10..14 | decode | 279..4779 | `' the Mac's unified memory'` | 223.42 mean | 48.72 mean | 45.99 mean | 35.2 | 40, 98, 164, 21, 217, 35, 245, 24 |
| 15 | decode | 7087 | `' pool'` | 171.97 | 62.52 | 43.81 | 37.0 | 40, 21, 101, 192, 140, 98, 24, 46 |
| 16 | decode | 11 | `','` | 217.11 | 52.71 | 45.61 | 35.5 | 140, 192, 19, 222, 230, 129, 224, 207 |
| 17..26 | decode | 6866..220 | `' giving you enough total fast memory to fit a '` | 249.84 mean | 70.05 mean | 45.65 mean | 35.5 | 174, 40, 116, 98, 29, 24, 131, 221 |
| 27..32 | decode | 18..1558 | `'35B MoE model'` | 294.99 mean | 76.44 mean | 46.61 mean | 34.8 | 131, 174, 101, 42, 13, 116, 98, 40 |
| 33..40 | decode | 421..7249 | `' that wouldn't fit on either machine alone'` | 185.59 p50 | 94.52 p50 | 45.33 p50 | 35.8 | 76, 195, 192, 40, 19, 166, 36, 37 |
| 41 | decode | 13 | `'.'` | 204.04 | 20.61 | 47.81 | 33.9 | 97, 28, 40, 3, 140, 91, 24, 46 |
| 42 | decode | 248046 | `'<\|im_end\|>'` | 443.86 | 107.73 | 49.51 | 32.7 | 182, 147, 134, 43, 152, 32, 154, 52 |

Prompt 2 time to first token (Derived from row 00): `12,986.71 ms`, which is `371.05 ms` per prompt token, against `130.19 ms` per prompt token for prompt 1. The run report notes stage 0 spikes above `500 ms` at steps 20 and 35, inside the aggregated rows, when newly routed experts were first touched.

### 1.5 Headline figures (Measured, as reported)

| Figure | Value | Split |
| :--- | ---: | :--- |
| Best 32-token decode window | `2.73 tok/s` (`366.89 ms/tok`) | Stage 0 `268.59 ms`, hop `51.68 ms`, stage 1 `46.63 ms` |
| Fastest single decode step (warm page cache, low hop jitter) | `4.10 tok/s` (`244.11 ms/tok`) | Stage 0 `180.18 ms`, hop `19.17 ms`, stage 1 `44.76 ms` at `36.2 GB/s` |
| Rust path steady state, p50 (20 AVX2 threads on the PC, 10 NEON cores on the M4) | `3.25 tok/s` (`308.1 ms/tok`) | Stage 0 `222.64 ms`, hop `44.51 ms`, stage 1 `48.36 ms` |
| Single-PC reference, cold NVMe paging, no Mac | `0.82 tok/s` | The two-stage Rust path is **3.98×** this figure (**Derived**: 3.25 / 0.82) |
| M4 stage memory stream rate | `34.8 to 37.2 GB/s` | Across all 10 cores, about a third of the `120 GB/s` spec |

### 1.6 Where the time goes (Derived)

| Segment | Typical decode step | What bounds it |
| :--- | ---: | :--- |
| Stage 0, PC, 25 layers | `220 to 290 ms` | CPU-only torch. Stage 0 holds about `45 GB` of BF16 weights (`25/40` of `71.90 GB`) against `32 GB` of RAM, so cold experts page in from NVMe |
| Hop | `45 to 65 ms` mean, `20 to 150 ms` range | Serialization and daemon overhead on Wi-Fi. The payload is one hidden-state vector of a few kilobytes, so bandwidth is not the limit |
| Stage 1, Mac, 15 layers | `45 to 48 ms` | Memory-bound on the CPU NEON path at about `35 GB/s` |
| Whole token | `310 to 400 ms`, `2.5 to 3.3 tok/s` | Stages run in series |

Prefill scales with the number of distinct experts touched rather than with prompt length alone: 35 tokens through 25 layers at top-8 reach most of the 256 experts per layer, which streams most of stage 0's weights through NVMe once. This is why prompt 2's prefill cost `2.85x` more per token than prompt 1's.

### 1.7 Caveats and open checks

1. **Stage 1 footprint against Mac memory.** Fifteen of forty layers in BF16 is about `27 GB`, more than the `16 GB` the run report gives for the Mac mini. Either the mini has more memory than stated, `ornith_stage1_real.ktb` is already reduced, or the Mac is also paging through its file cache and the report's "zero disk paging" claim does not hold. Check the file size of `ornith_stage1_real.ktb` and the mini's installed memory before the next run.
2. **The proposed Q4 split leaves no VRAM headroom.** The run report proposes `~11.8 GB` of stage 0 weights in the RTX 5070's `12 GB`, which leaves nothing for the CUDA context, activations, the 30 recurrent states and the 10 KV caches. Let `ShardPlanner` in [`crates/kadmium-sharding/src/lib.rs`](../crates/kadmium-sharding/src/lib.rs) choose the split from memory, or use a lower bit rate, or keep warm experts in the PC's DDR5 as the q* tiering intends.
3. **The planner scores the Mac at spec bandwidth.** [`crates/kadmium-core/src/m4_mac_mini.rs`](../crates/kadmium-core/src/m4_mac_mini.rs) uses `120 GB/s`; this run achieved about `35 GB/s` on the CPU path. Until the Metal path exists the planner will over-assign layers to the Mac. Feeding the measured streaming bandwidth from `kadmium probe` into the capacity score corrects this.
4. **Single run, uncontrolled cache state.** Prompt 2 ran after prompt 1, so some pages were already warm; the OS page cache was not flushed between runs; no power or thermal readings were logged; no repetitions were taken. Treat every figure as one sample.
5. **The harness is not in the repository.** See 1.2. The numbers cannot be reproduced from `origin/main` as of `2026-10-23`.
6. **The run report's own figures are quoted as given.** Row-level values were cross-checked against the reported means for prompt 1 and agree to rounding.

### 1.8 Levers for Run 2 (Modelled, not measured)

The order is by expected payoff. Every figure in this table is a projection and must be replaced by a measured row when Run 2 lands.

| Lever | Expected effect per decode token |
| :--- | :--- |
| Quantize to roughly 3 to 4.5 bits per weight so the whole model sits in VRAM plus unified memory, with the split chosen by `ShardPlanner` | Removes NVMe paging entirely; prefill falls from seconds to well under one second |
| Run stage 0 on the RTX 5070 (the harness is CPU-only today) | Stage 0 from about `250 ms` to `15 to 25 ms` |
| Run stage 1 on Metal or MLX instead of CPU NEON | Stage 1 from about `46 ms` to `6 to 10 ms` at 4-bit |
| Carry the hop as `KDMF` binary frames with INT8 boundary activations ([`crates/kadmium-network/src/lib.rs`](../crates/kadmium-network/src/lib.rs), [`crates/kadmium-runtime/src/live_worker.rs`](../crates/kadmium-runtime/src/live_worker.rs)) over Ethernet or the USB4 link, instead of JSON over HTTP on Wi-Fi | Hop from about `50 ms` to `1 to 2 ms` |
| All four together | Roughly `25 to 35 ms` per token, `30 to 40 tok/s`, which sits inside the modelled envelope in the R4 review for models of this size |

---

## Run 2: inter-step gap of the decode loop on the reference kernels, two loopback TCP workers (`2026-10-23`)

**Status: Measured on the reference kernels. Not a model run.** The figures measure the coordinator's scheduling gap between decode steps before and after step pipelining (ARCHITECTURE 9.10). The kernels are the synthetic 64-float references, so every number here is about orchestration; nothing in this run says anything about model execution or replaces the modelled envelope.

### 2.1 What was run

- **Date and repository state:** `2026-10-23`, branch `claude/deepinfer-scheduling-7c30d3` over `main` at `e0ad4df`, uncommitted at the time of the run. The harness is in the repository: the test `pipelined_steps_match_sequential_bytes_and_close_the_inter_step_gap` in [`crates/kadmium-runtime/src/decode_loop.rs`](../crates/kadmium-runtime/src/decode_loop.rs), run as `cargo test -p kadmium-runtime --lib pipelined_steps_match_sequential`; it writes `target/kadmium-state/benchmarks/inter_step_gap.json`.
- **Model:** `toy-moe` (4 layers, 8 routed experts per layer, 2 active per token, 64-float hidden rows, INT8 layer boundaries in 16-wide blocks), the reference kernels of `kadmium-runtime::moe_layer`. No weights are read.
- **Hardware and configuration:** one machine, AMD Ryzen 5 7535HS, 32 GB RAM, Windows 11 Pro. Two worker servers on loopback TCP with the `node-main-pc` and `node-m4-mac-mini` capability profiles (the profiles steer placement; the arithmetic runs on the host CPU); `KDM2` frames over the persistent channel pool; the coordinator in the test process on a tokio multi-thread runtime with 4 worker threads. Per shape: 2 sessions, 6 steps each, 2 rows per step; a warm-up loop first; then the sequential loop (one graph per step through its own queue, sessions one after another) and the pipelined loop (one runner, the next step queued behind the current sink, sessions interleaved). Three runs, the whole test re-run each time.
- **What is measured:** per session, the inter-step gap: the next step's first dispatch minus this step's last return, both on the coordinator's clock (`GridDispatchTrace::dispatched_at_ms`, `returned_at_ms`); the wall from the first dispatch to the last return over both sessions.

### 2.2 Inter-step gap (Measured)

| Shape | Mode | Mean gap, ms (run 1 / 2 / 3) | Max gap, ms | Wall of 2 sessions x 6 steps, ms |
| :--- | :--- | ---: | ---: | ---: |
| Pipeline chain, 2 stages | Sequential | `0.390 / 0.502 / 0.568` | `0.68 / 0.72 / 0.90` | `44.0 / 54.7 / 62.4` |
| Pipeline chain, 2 stages | Pipelined | `0.173 / 0.260 / 0.247` | `0.24 / 0.38 / 0.44` | `21.6 / 28.3 / 28.2` |
| Expert parallel, 4 layers | Sequential | `1.957 / 1.594 / 1.842` | `3.23 / 2.00 / 2.45` | `162.1 / 134.0 / 157.5` |
| Expert parallel, 4 layers | Pipelined | `0.952 / 0.909 / 1.057` | `2.39 / 2.48 / 2.50` | `122.4 / 98.9 / 100.8` |

The sessions' bytes were identical between the two modes at every step, and identical to the one-graph dispatcher's, in every run (the test asserts it).

### 2.3 Reading (Derived)

- **Chain:** the gap falls by about half, from `0.39 to 0.57 ms` to `0.17 to 0.26 ms`, and the wall roughly halves, because the two sessions' stages overlap on the two nodes once they share a queue.
- **Expert parallel:** the gap falls from `1.6 to 2.0 ms` to `0.9 to 1.1 ms`. In the pipelined runs one session's gaps sit near `0.2 ms` and the other's near `1.5 to 2.5 ms`: a fan-out step is dozens of units of about a millisecond each, settled one at a time by the coordinator, so a session's next step waits behind the other session's returns in the coordinator's loop. The remaining gap is the coordinator, not the planning.
- A step on these kernels takes a few milliseconds end to end; a real model's step would be tens to hundreds, and the same gap would be a smaller share of it.

### 2.4 Caveats

1. One machine, loopback, and the coordinator in the same process as the workers; nothing crossed a wire.
2. Three runs with other processes on the machine and no control of cache or CPU state; treat the spread between runs as the noise floor. The test asserts only that the best pipelined mean gap of three pairs is at or below the best sequential one.
3. The gap excludes the time a dispatched frame spends before the worker reads it, since `dispatched_at_ms` is taken when the call is spawned.

---

## Run 3: sweep harness over batch widths and node counts on the reference kernels, two loopback TCP workers (`2026-10-23`)

**Status: Measured on the reference kernels. Not a model run.** The sweep harness (ARCHITECTURE 9.13) run once on the same machine and workers as Run 2: a pipelined decode loop per batch width on one node and on two, each point a row of the Pareto table with its full configuration. The figures describe the orchestration of the synthetic references and say nothing about model execution.

### 3.1 What was run

- **Date and repository state:** `2026-10-23`, branch `claude/deepinfer-scheduling-7c30d3` over `main` at `e0ad4df`, uncommitted at the time of the run. The harness is in the repository: the test `sweep_over_batch_widths_and_node_counts_writes_a_pareto_table` in [`crates/kadmium-runtime/src/decode_loop.rs`](../crates/kadmium-runtime/src/decode_loop.rs), run as `cargo test -p kadmium-runtime --lib sweep_over`; it writes `target/kadmium-state/benchmarks/sweep_loopback.md` and `.jsonl`. The gateway's `POST /api/v1/grid/sweep` produces the same table for a live mesh.
- **Model, hardware, configuration:** as Run 2 (`toy-moe`, AMD Ryzen 5 7535HS, two loopback TCP workers with the `node-main-pc` and `node-m4-mac-mini` profiles, `KDM2` frames). Per point: 1 session, 4 steps, pipelined, the pipeline chain shape; batch widths 1, 2, 4 and 8 rows; node sets of one node (one trunk range, no hop) and two (two trunk ranges, one hop per step). One run, after a warm-up loop per node set.
- **Figures:** `tok/s` is the point's tokens over the loop's wall (first dispatch to last return); `per-user tok/s` is one row's tokens over the session's wall, a user being one row of the batch, as DeepInfer plots it; `gap ms` the mean inter-step gap. Points compare only within a curve.

### 3.2 Pareto table (Measured)

| curve | shape | nodes | node set | rows | step ms (mean) | steps/s | tok/s | per-user tok/s | gap ms | pareto |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `9836d442` | Pipeline chain | 1 | `node-main-pc` | 1 | `1.80` | `555.4` | `488.4` | `488.4` | `0.329` | yes |
| `9836d442` | Pipeline chain | 1 | `node-main-pc` | 2 | `2.38` | `420.4` | `774.8` | `387.4` | `0.270` | yes |
| `9836d442` | Pipeline chain | 1 | `node-main-pc` | 4 | `3.90` | `256.4` | `952.6` | `238.2` | `0.399` | yes |
| `9836d442` | Pipeline chain | 1 | `node-main-pc` | 8 | `6.23` | `160.5` | `1244.0` | `155.5` | `0.268` | yes |
| `25bfc213` | Pipeline chain | 2 | `node-m4-mac-mini`, `node-main-pc` | 1 | `2.83` | `353.4` | `335.6` | `335.6` | `0.199` | yes |
| `25bfc213` | Pipeline chain | 2 | `node-m4-mac-mini`, `node-main-pc` | 2 | `3.80` | `263.0` | `504.2` | `252.1` | `0.218` | yes |
| `25bfc213` | Pipeline chain | 2 | `node-m4-mac-mini`, `node-main-pc` | 4 | `6.08` | `164.5` | `641.8` | `160.4` | `0.203` | yes |
| `25bfc213` | Pipeline chain | 2 | `node-m4-mac-mini`, `node-main-pc` | 8 | `11.64` | `85.9` | `671.9` | `84.0` | `0.353` | yes |

Full configuration of both curves: `toy-moe` (4 layers, 8 experts per layer, 2 active), placement digests `626c8e02e12221f3` (one node) and `e99a4db5565bea14` (two nodes), every node at `v0.1.0-cd48`, kernels `reference-synthetic`, 1 session, 4 steps, pipelined, no drafting, repository `claude/deepinfer-scheduling-7c30d3` over `e0ad4df`, uncommitted.

### 3.3 Reading (Derived)

- Within each curve the trade-off is the one the table exists to show: from 1 to 8 rows the throughput rises `2.5x` on one node and `2.0x` on two, while the per-user speed falls `3.1x` and `4.0x`. Every point is on its curve's frontier, so the width is a choice between the two axes, not a free improvement.
- The two-node curve sits below the one-node curve at every width: on these kernels a step's compute is a few milliseconds and the hop between the two stages costs more than the second node saves. On a model whose layers take longer than the hop, the second node would pay; the table cannot say so, which is the point of comparing only within a curve.
- The step time grows less than linearly with the rows (`1.80 ms` to `6.23 ms` for `8x` the rows on one node), which is the batching the throughput axis counts.

### 3.4 Caveats

1. One run per point, no repetitions; Run 2's three-run spread is the noise floor to read these against.
2. One machine, loopback, coordinator in the test process; nothing crossed a wire.
3. The one-node curve has no hop at all and the two-node curve one hop of loopback TCP; neither says anything about a LAN or a WAN.

---

## Run 4 — Delta-link transport (reported 2026-10-30)

### 4.1 Configuration and provenance

- **Harness:** `cargo run --release -p kadmium-network --example delta_link_bench -- serve` on the worker and `cargo run --release -p kadmium-network --example delta_link_bench -- run <host>:7870` on the coordinator.
- **Configuration reported in `docs/ARCHITECTURE.md` §9.3:** Wi-Fi to an M4 Mac mini, 12 steps per scenario.
- **Provenance limitation:** the peer hardware, exact checkout, and raw timing output are not recorded in the repository. Treat this as a reported harness result, not an independently reproducible ledger run.

### 4.2 Reported results

| Scenario | Result | Label |
| :--- | ---: | :--- |
| Re-sent stage weights | 8.0× | **Derived as reported** from scenario timings; raw values absent |
| Decode steps with unchanged context | 9.1× | **Derived as reported** |
| KV appends | 6.9× | **Derived as reported** |
| Sparse parameter updates | 5.8× | **Derived as reported** |
| Fresh random payloads | 0.9× | **Derived as reported**; reported worst case |

These ratios describe the delta-link example on one Wi-Fi setup. They are not KDM2-wide, internet, model-inference, or guaranteed bandwidth improvements.

## Run 5 — Native CUDA kernel parity and timings (entry recorded 2026-10-30)

### 5.1 Configuration and provenance

- **Harness:** `cargo oxide run` for parity and `cargo oxide run -- --bench` for timings from `crates/kadmium-cuda/device`.
- **Hardware/software:** RTX 5070 (`sm_120`), CUDA 13.3, driver 616.92, WSL2 Ubuntu; 4096 × 4096 mat-vecs.
- **Parity:** every reported case matched the host reference bits, including partial/zero blocks, NaN/infinite inputs, E8M0 exponents, tied router logits, and long reduction rows. NaNs compare as values, not as identical payloads.
- **Provenance limitation:** the underlying run date, exact checkout, repetition count, and raw logs are not recorded in `crates/kadmium-cuda/README.md`. Results below are **Measured as reported** and **Derived as reported**, not an independently reproduced run.

### 5.2 Reported results

| Kernel | Per-launch range (Measured as reported) | Weight bandwidth (Derived as reported) |
| :--- | ---: | ---: |
| `q4k_q8k_matvec` | 33–38 µs | 330–380 GB/s |
| `mxfp4_matvec` | 100–110 µs | 81–88 GB/s |

The `mxfp4_matvec` reduction shape, not memory alone, bounds the current kernel. These results cover two specific CUDA kernels on one GPU and do not establish general model-backend performance or parity on other backends.

---

## How to add a run

Copy the Run 1 structure and fill every field; leave nothing implied.

1. **Date and repository state:** run date, `main` commit hash, and whether the harness is committed.
2. **Model:** checkpoint path, architecture class, file count and bytes, precision or quantization, layer and expert counts.
3. **Hardware and configuration:** every node with its role, layer range, memory, accelerator, execution path and software versions; the link type and the wire format of the hop.
4. **Per-step telemetry:** one row per generated token with stage times, hop time, and any node-reported stream rate; mark aggregates as mean or p50.
5. **Headline figures:** time to first token, mean decode time per output token, best window, and the single-node reference the run is compared against.
6. **Labels:** Measured for anything logged by the harness, Derived for arithmetic on logged values, Modelled for projections.

> **Sweep harness limitation:** `crates/kadmium-runtime/src/decode_loop.rs` currently writes a hard-coded historical repository string into newly generated sweep output. Do not treat that generated provenance as the current checkout; record the actual commit and working-tree state separately until the harness is fixed.
