# Kadmium Market Research — October 2026

> **What this is.** The market and capability case for Kadmium, built from three kinds of evidence: the mechanisms implemented in the repository (`readme.md`, `docs/`, and all eight crates under `crates/`, read in full; `cargo test --workspace` run), the arithmetic those mechanisms imply on real hardware, and market data checked on **20 October 2026** (65 numbered sources, §14). It then audits the earlier plan, [`MARKET_RESEARCH_AND_LAUNCH_PLAN.md`](./MARKET_RESEARCH_AND_LAUNCH_PLAN.md), against the same evidence.
>
> **How to read it.** Part A (§1–§6) is written to be lifted into external material — a grant's market section, the Kickstarter page, investor notes. Part B (§7–§8) holds the evidence tables and the line‑by‑line audit of the plan. Part C (§9–§13) is for planning: readiness and critical path, a risk register, recommendations and line‑level edits to the plan. Figures from the plan's demo scripts and the marketing pages were placeholders written before the site went live; wherever they appear here they are replaced by modelled figures labelled as such, with the model reproducible from Appendix A. Measured figures from `kadmium bench` should replace the modelled ones as they land.
>
> **Not legal or financial advice.** Regulatory and tax notes summarise public sources; confirm with counsel before launch.
>
> **Revision history.** R1 (audit of the plan) → R2 (Kadmium's upstream engines, the no‑direct‑competitor finding) → R3 (capability envelope from mechanism math, Phase 6 arithmetic) → **R4, this version: rebuilt case‑first after a full read of the code; adds the designed‑behaviour and multi‑stream cases to the envelope, a code‑level description of what Kadmium is, and a readiness/critical‑path section.**

---

## Part A — The case for Kadmium

### 1. Executive summary

**What Kadmium is.** Kadmium applies the two systems ideas that made volunteer and grid computing work — BOINC's polite harvesting of machines people already own, and CERN's peer‑to‑peer data staging — to AI inference, training and RL, and pairs them with the memory‑tiering ideas from FreeToken and colibri (bandwidth‑adaptive CPU/GPU splits, NVMe expert streaming) so that a desk, a lab or a community of ordinary machines behaves as one accelerator [57][58][65]. It is one pure‑Rust binary, 18,782 lines across eight crates with no dependencies beyond tokio, serde, axum and clap, and the whole control plane is implemented and tested end to end: hardware profiling, the asymmetric shard planner with privacy‑tiered placement, the `.ktb` tensor‑bank format with Merkle proofs, FTS3‑style page streaming, a binary wire protocol over real sockets, polite governance, Verde verification with fraud proofs, DeMo/DiLoCo training math, the Supply Grid settlement engine, RBAC, and OpenAI/Anthropic/MCP gateways — 36 tests pass, including a live multi‑process TCP pipeline (§2). Nothing shipping combines these pieces; the nearest things each do one or two of them (§5).

**What the math says it can do.** The envelope in §3 derives single‑stream and multi‑user throughput from each mechanism's arithmetic and validates it against measured systems (EXO, AMD, colibri — within 13–40%). Capacity is never the limit: engines using the same tiering run GLM‑5.2 (744B) in 16–24 GB of RAM [58]. Speed is governed by one lever — how much of a model's routed experts the mesh holds in fast memory — and pooling is the cheapest way anyone can pull that lever in 2026. The plan's own "kitchen‑sink" desk (RTX 4090 + 64 GB, MacBook Pro 36 GB, NAS, SBC) runs gpt‑oss‑120b at **28–67 tok/s**, a dense 70B at ~7, Qwen3‑235B at **3–8** and DeepSeek‑V3 at **1–2.5**; a five‑developer office mesh (two 4090 PCs, two MacBook Pros, one Mac Studio, 10 GbE — about 400 GB of pooled fast memory) runs DeepSeek‑V3 at **4–16 tok/s per user and ~30 aggregate for five users**, Qwen3‑235B at 12–39, gpt‑oss‑120b at 38–85 (ranges are the uniform‑routing floor to the designed case; §3.3). No single machine on that LAN can host those models at those speeds; the mesh can.

**Why now.** The memory supercycle made large fast memory the scarcest thing in computing: consumer DRAM roughly doubled in H1 2026 with shortages expected through 2028 [38], Apple withdrew the 512 GB Mac Studio [39], NVIDIA's DGX Spark 128 GB went from $3,999 to $6,950 [41]. The open‑weight frontier went to 0.75–2.8T parameters [45][14], beyond any single consumer box. The audience is large and growing: colibri and FreeToken, the single‑machine engines Kadmium's tiering is modelled on, have 42.8k and 14.4k GitHub stars [57][58]; r/LocalLLaMA is 853k members, +55% in a year [36]; Steam's 43.5M concurrent record and VRAM survey imply roughly three million 24 GB+ and twelve million 16 GB GPUs online at peak [33][34]. Ollama's move to per‑token credits validates the hybrid local→cloud subscription [14]; Prime Intellect's $130M Series A at a $1B valuation validates compute aggregation as a venture‑scale business [26]; GPU rental and resale prices rose through the autumn [1][42], so owning and pooling compute is worth more, not less (§4).

**The business, in numbers.** Open‑core: free local pooling forever; paid governance, burst and failover (§6). Burst should be priced as a daily wholesale index plus a spread (today's index: 4090 $0.33–0.48, A100 $0.89–1.80, H100 $1.80–3.99 [1]) rather than the plan's fixed rates, which are now at or below cost. The Supply Grid is a market of its own: the code already pays a 4090‑class host $0.38/hour in USD or 1.25× that in credits ($0.475/hour) [65] — three to four times Salad's cash rate of $0.11–0.20 [7] — and a residential peer's 25 Mbps uplink carries about 1.6M tokens per hour of batch work, so ten peers are ~16M tokens/hour, worth $2.4–4.5/hour at serverless list prices for V3‑class tokens [13]. Credits are a liability to be modelled, not free (§6.3). Revenue cases on the plan's own prices: $16k, $36k and $111k per month at 90 days for bear, base and the plan's conversion assumptions (§6.5).

**Readiness, plainly.** The science is in code; the engineering ahead is integration. The stage kernels are correct reference implementations that run on small synthetic tensors, the chat endpoint returns real pipeline telemetry with placeholder text, and `bench` derives GFLOPs from spec sheets. Real model execution with GPU/Metal/CPU backends is the critical path to every demo in the plan, followed by QUIC/NAT traversal, OS yield hooks, a WebAssembly runtime and an HTTP client for `hf://` pulls (§9). Two routes exist: pure‑Rust engines, or FreeToken/colibri as per‑node execution backends under Kadmium's planner — both Apache‑2.0, both already doing the single‑box part [57][58]. The plan's week‑three demo should be re‑sequenced behind that work; the NSF pitch can go in now with the full proposal aimed at 4 March 2027 [48].

**What the plan gets wrong, briefly.** Wholesale GPU rates are stale, so the burst margin model inverts (§6.2); the heavy‑coding cost comparison quotes the wrong price tier (§6.4); the host‑credit arithmetic is inconsistent with the code (§6.3); the demo figures are placeholders (§3); the Kickstarter hardware BOM is exposed to DRAM (§6.6); and a handful of competitor facts need correcting — LM Studio "Bionic", Petals as live, Cloudflare's $10k, the 470–490 TWh figure (§7). All of it is fixable with the edits in §12.

### 2. What Kadmium is, from the code

The repository is the primary source for this section; the README and docs describe the same system but the code says what exists. Kadmium is a Cargo workspace of eight crates — `kadmium-core`, `-sharding`, `-data`, `-network`, `-runtime`, `-rl`, `-gateway`, `-cli` — totalling 18,782 lines of Rust (edition 2024) with no third‑party dependencies beyond `serde`, `serde_json`, `thiserror`, `tokio`, `axum`, `clap` and `tracing`. `cargo test --workspace` passes 36 tests, one of which spawns two worker daemons and a coordinator as separate TCP servers on loopback, plans a DeepSeek‑V3‑class MoE across them, executes a multi‑hop forward pass and a distributed QLoRA + DeMo training step over the wire, pushes a signed OTA update, and spawns a five‑node local fleet [65].

**The flow that exists end to end.** A node starts, probes itself (`LiveHardwareProfiler`: RAM from `/proc/meminfo`, `sysctl` or `wmic`; device‑tree board model; `nvidia-smi` or ROCm sysfs for VRAM; a page‑aligned streaming micro‑benchmark; LAN IP) and advertises a `HardwareNode` capability vector with a five‑tier `MemoryTierProfile`, a `PoliteState` and a `NodeTrustTier`. The `ShardPlanner` scores every node per workload (FLOPs, fast memory, bandwidth, NVMe, latency penalty), allocates layers in proportion by largest remainder, orders the pipeline so `TrustedCore` nodes always hold layer 0 and layer N−1 with peers and spot nodes confined to the middle, places each node's weights across hot (VRAM/unified), warm (RAM) and cold (NVMe stream) tiers with the node's q\* ratio, assigns auxiliary roles (NAS seed vault; SBC tokenisation, drafts, WASI verifiers; legacy laptops as Verde auditors), and chooses a training strategy from the link latencies. A `ResponsibilityShiftPlan` diffs two plans into layer migration moves when a node starts gaming, drains or a burst GPU joins. The coordinator then drives workers over `KDMF` binary frames: `SystolicTokenWavefront` for inference stages (INT8 blockwise boundary activations in both directions), `DeMoCompressedGradient` for training stages, `Fts3KtbChunkStream` to stream 64 KiB `.ktb` pages with Merkle inclusion proofs verified on arrival, `VerdeBisectionChallenge` for O(log L) audits, `DreamRsiRolloutReplay` for SBC/NAS replay work, `FleetUpdatePush` for MAC‑signed OTA updates with a post‑update RepOps self‑test. Results flow back to an Axum gateway that exposes OpenAI (`/v1/chat/completions`, SSE), Anthropic (`/v1/messages`), embeddings, an MCP server (stdio and HTTP), Prometheus metrics and W&B summaries, plus cluster, shift, slice, supply and governance APIs, and to the dashboard, companion and docs pages [65].

**What each crate implements and what its tests prove.**

| Crate | Implemented and exercised by tests | Reference‑level or modelled today | Integration still ahead |
|---|---|---|---|
| `kadmium-core` | Capability vector; five memory tiers; polite multipliers (idle 1.0 → gaming 0.15); trust tiers with boundary‑layer rule; q\* ratio; per‑workload capacity scores; model presets; `LiveHardwareProfiler` (real probes); `PoliteActivityMonitor` (<200 ms decision from a sample; reads `nvidia-smi` utilisation live); swapchain hysteresis engine (0 ms yield, 15 s reclaim); `SubClusterRbacEngine` (roles, scoped keys, quotas; `BlindSupplyProvider` barred from layer 0/N−1); `SupplySettlementEngine` (USD/credit rates, 1.25× credits, receipts, workload masks, lease policies); `kadmium.toml` | Capacity weights and TFLOPs tables are heuristics; the polite sampler sees GPU utilisation, not the foreground process | OS swapchain/foreground hooks (DXGI/NVML, IOKit/Metal, DRM) |
| `kadmium-sharding` | `ShardPlanner` and shift plans (above); `MinMaxStageLatencyOptimizer`; `DistributedPagedKvPlanner` (16‑token pages across tiers); `SubClusterPartitioner` with elastic borrowing; `HybridCloudArbitragePlanner`; `SpotOrderbookBidEngine`; `CloudShieldCutoverWatcher` | Throughput estimate is a FLOPs heuristic (§3 replaces it); spot orderbooks are modelled | Live RunPod/Vast/Lambda/Prime Intellect APIs |
| `kadmium-data` | `.ktb` packer/reader (`KDTB`, strict 64 KiB pages, FNV‑1a checksums); `KtbMerkleTree` with inclusion proofs; `Fts3PagePatchEngine`; safetensors header parser and shard‑filtered transcoder to `.ktb`; GGUF v3 parser/transcoder; FTS3 chunk manifests and diffs; HF `index.json` + byte‑range computation; `hf://`, `gh://`, `oci://`, `s3://`, `wandb://` URI handling | Transcoders exercised on in‑memory buffers | HTTP client for `hf://` pulls; `mmap`/`io_uring` read paths; GGUF metadata KV |
| `kadmium-network` | `KDMF` frame codec; `KDMD` LAN beacons over real UDP; HMAC join tickets with 6‑digit SAS; `ClusterStateCrdt` merge; coordinator failover quorum; live worker registry; `FleetUpdateManifest` with MAC | NAT classification and hole‑punch planning without STUN traffic; transport is TCP (docs say QUIC) | QUIC (`quinn`), STUN/DERP relay, `relay.kadmium.ai` |
| `kadmium-runtime` | Reference kernels: fused RMSNorm+RoPE, paged GQA with online softmax, `Q4KBlockTensor` 4‑bit block matvec (5–6× compression, tested against FP32), fused SwiGLU expert on Q4K; `MoEGatingRouter` with residency tiers; Colibri double‑buffer stall model; INT8 boundary quantiser; `DeMoDctCompressor` (orthonormal DCT‑II/IDCT, top‑k, exact error feedback — energy conservation tested); `DiLoCoOuterOptimizer`; `Pipeline1F1BScheduler` with bubble %; `DistributedQLoRAAdapter` (loss decreases in tests); Verde bisection + `VerdeFraudProofEngine` certificates; `LiveNodeWorkerServer` and `DistributedLanExecutor` (inference, training, FTS3 streaming, Verde audit, Dream‑RSI dispatch, embeddings, OTA); `LocalFleetSpawner` | Stage forward runs on tiny synthetic tensors (hidden state = the INT8 boundary vector; a 64‑float gamma; synthetic KV pages; hash‑based gating over 256 experts, top‑4; fixed draft strings); `bench` GFLOPs = spec TFLOPs × 0.78 | Real weights and GPU/Metal/CPU kernels at model scale; tokenizer and sampling; KV cache at scale |
| `kadmium-rl` | GRPO group advantages; fuel‑metered deterministic sandbox (six‑op stack machine with fuel and depth limits); PUCT/UCB1 MCTS Dream‑RSI engine; world registry; RL topology planner | The sandbox is a mini‑ISA, not WebAssembly; replay candidates are synthetic | `wasmtime`/WASI; real verifiers (compilers, test runners) |
| `kadmium-gateway`, `kadmium-cli` | All endpoints above; dashboard/companion/docs/marketing HTML; CLI: `node`, `join`, `pairing`, `simulate`, `plan`, `dream`, `dataflow`, `slice`, `cloud-shield`, `bench`, `serve`, `lease`, `pull`, `compile`, `inspect`, `registry`, `mcp` | Chat completion content is a fixed assistant message carrying real telemetry (hops, RTTs, digests, estimated tok/s) | Real generation once the execution layer exists |

**Design properties worth selling.** Four of them do not appear in the earlier plan at the prominence they deserve. *Privacy by construction:* `NodeTrustTier` keeps prompt ingress and token egress on hardware the user owns, peers and spot nodes only ever see INT8 mid‑network activations, and the RBAC engine refuses a `BlindSupplyProvider` any request that touches layer 0 or N−1 — "zero plaintext egress" is a property of the planner, not a promise. *Verifiability:* every stage returns a RepOps digest computed with fixed‑tree reductions, a disputed shard is bisected in ⌈log₂ L⌉ + 1 referee evaluations, and the outcome is a `VerdeFraudProofCertificate` that can slash and reassign a faulty community worker. *Operability for a volunteer swarm:* a single zero‑dependency binary, UDP discovery, SAS pairing, CRDT cluster state, coordinator failover, Merkle‑verified page streaming, MAC‑signed OTA with self‑test — the lessons of two decades of BOINC. *Economics in the binary:* lease policies, workload masks, USD or 1.25× credit payout, verified compute receipts and datacenter‑avoidance accounting are types in `kadmium-core`, not slides.

**Execution‑layer status, stated plainly.** The runtime's kernels are correct reference implementations; what they run on today are small synthetic tensors, and the pipeline's "tokens per second" is a modelled figure. None of this diminishes the design — it is the normal shape of a systems project that built its hardest distributed pieces first — but it fixes the critical path (§9): real model execution with hardware backends is what turns every number in §3 from modelled to measured, and every demo in the plan depends on it.

### 3. What it can do: the capability envelope

The plan's demo figures ("`DeepSeek‑V3‑671B` on a gaming PC + MacBook + NAS + RK3588", "`112 GB` pool … `34 tok/s`", "`12 → 32–42 tok/s`" with drafts) and the marketing site's example numbers are **placeholders** written before the site went live; they are not Kadmium performance claims and this review no longer treats them as such. What the plan needs in their place is a capability envelope derived from the mechanisms Kadmium actually implements — asymmetric sharding, the q\* split, Colibri double‑buffered NVMe streaming, weight‑stationary vs token‑stationary dataflow, INT8 boundaries, speculative drafts — and then measurements. This section derives that envelope. Every formula below is reproducible with the script in Appendix A; every parameter is stated so it can be corrected.

#### 3.1 The governing equation for single‑stream decode

One token must read every *active* weight once. Dense trunk layers (attention, shared experts, embeddings) are read in layer order, so their time adds across whichever tiers hold them; routed experts can be read concurrently when they sit on different devices (weight‑stationary mode, `ComputeInMemoryTokenStreaming` [65]), at the price of one activation round‑trip per MoE layer:

```
T_token = Σ_tiers trunk_bytes_i / (BW_i · η_i)                       # sequential, fastest tiers filled first
        + T_experts                                                  # see below
        + miss_bytes / (BW_nvme · η_nvme)                            # cold experts streamed (Colibri)
        + (nodes − 1) · t_hop                                        # pipeline boundary hops

pipeline only  (seq):  T_experts = Σ_i hit_bytes_i / (BW_i · η_i)
expert‑parallel (par): T_experts = max_i hit_bytes_i / (BW_i · η_i) + L_moe · t_rt
hit rate: c = cached_expert_bytes / routed_total ; hit = c^α   (α = 1 uniform routing; α ≈ 0.25 fits colibri's learned pins, §3.7)
```

`η` is the fraction of peak memory bandwidth real kernels achieve (0.6 for GPU/unified memory, 0.7 for CPU‑from‑RAM, 0.9 for sequential 64 KiB NVMe pages, 0.5 for ROCm on Strix Halo); `t_hop` is one LAN hop with an 8 KB INT8 activation (0.4 ms on 10 GbE, 0.6 ms on 5 GbE, 0.7 ms on 2.5 GbE); `L_moe` is the number of MoE layers. The planner should take the better of *seq* and *par* per model: for small‑active models the per‑layer round‑trip costs more than expert parallelism saves.

**Per‑token bytes (Kadmium's `Q4KBlockTensor` ≈ 0.5625 B/param; gpt‑oss MXFP4 ≈ 0.53):**

| Model | Trunk read per token | Routed experts per token | Routed experts total | MoE layers | Derivation |
|---|---|---|---|---|---|
| DeepSeek‑V3 671B | 9.1 GB (16.1B params: MLA attention 11.4B, shared experts 2.6B, dense FFN 1.2B, embed/head 0.9B) | 11.5 GB (8 experts × 44.0M params × 58 layers = 20.4B) | 368 GB (256 × 44.0M × 58 = 654B) | 58 | expert = 3 × 7168 × 2048 |
| GLM‑5.2 744B | 9.9 GB resident dense weights | ~11 GB cold expert reads | ~415 GB | ~70 | colibri's published figures [61] |
| Qwen3‑235B‑A22B | 4.5 GB (~8B params) | 8.0 GB (8 × 18.9M × 94 = 14.2B) | 128 GB (128 × 18.9M × 94 = 227B) | 94 | expert = 3 × 4096 × 1536 |
| gpt‑oss‑120b | ~4 GB (attention/embeddings, BF16 in the release) | 1.9 GB (4 × 24.9M × 36 = 3.6B) | 61 GB (128 × 24.9M × 36 = 115B) | 36 | expert = 3 × 2880 × 2880 |
| Llama‑3.3‑70B dense | 40 GB (all 70B) | — | — | 0 | dense |

The DeepSeek figure reproduces colibri's independent "~11 GB of expert reads per token" for a same‑class model [61], which is the check that the bookkeeping is right.

#### 3.2 Representative meshes

Usable bytes after KV cache, activations and OS; peak bandwidths from vendor specs:

| Mesh | Fast memory (bytes usable, peak GB/s, η) | NVMe | Links |
|---|---|---|---|
| M1 "kitchen‑sink" (the plan's demo) | RTX 4090: VRAM 20 GB @1000 (0.6), DDR5 dual‑channel 54 GB @60 (0.7); MacBook Pro M3 Max 30 GB @300 (0.6); NAS; RK3588 drafts | 7 GB/s + 5 GB/s | 2.5 GbE, t_hop 0.7 ms |
| M3 single PC | 4090 VRAM 20 GB @1000; DDR5 54 GB @60 | 7 GB/s | — |
| M3b single PC, 192 GB RAM | 4090 VRAM 20 GB; DDR5 180 GB @60 | 7 GB/s | — |
| M4 four Strix Halo (AMD's guide [22]) | 4 × 118 GB @256 (0.5) | 4 × 7 GB/s | 5 GbE, 0.6 ms |
| M2 "five‑developer office" | 2 × 4090 PCs (VRAM 20 + DDR5 54 each); 2 × MacBook Pro M4 Max 42 GB @410; Mac Studio M4 Max 115 GB @546 | 2 × 7 + 3 × 5 GB/s | 10 GbE, 0.4 ms |

#### 3.3 Single‑stream decode envelope

Three figures per cell. *Floor:* every expert equally likely (α = 1), pipeline only — the number a generic engine with no placement intelligence would see. *Designed:* the planner does what the code describes — pins the hottest experts in the fastest tiers, chooses pipeline‑only or expert‑parallel per model, overlaps GPU and CPU work through q\* — with routing skew α = 0.25, the value that reproduces colibri's measured learned‑pin result (§3.7). *Aggregate (5):* five concurrent users on the designed configuration, batched per step so the dense trunk is read once and expert reads grow as N^0.7 with overlap. Real systems should land between floor and designed until `α` and `η` are measured for the actual models and traffic; the aggregate column is the office‑mesh throughput argument.

| Model | M1 kitchen‑sink desk | M3 single PC (64 GB) | M3b single PC (192 GB) | M4 four Strix Halo | M2 five‑developer office |
|---|---|---|---|---|---|
| gpt‑oss‑120b (117B, 5.1B active) | **28 / 67 / 162** | 24 / 51 / 108 | 21 / 51 / 108 | 21 / 21 / 63 | **38 / 85 / 263** |
| Llama‑3.3‑70B dense | **6.9 / 6.9 / 34** | 2.0 / 2.0 / 10 | 2.0 / 2.0 / 10 | 3.2 / 3.2 / 16 | **15 / 15 / 73** |
| Qwen3‑235B‑A22B | 2.6 / 7.5 / 13 | 1.5 / 4.2 / 7 | 5.4 / 12 / 20 | 10 / 10 / 22 | **12 / 39 / 84** |
| DeepSeek‑V3 671B | **0.7 / 2.5 / 4.2** | 0.6 / 1.4 / 2.3 | 1.0 / 2.4 / 4.1 | 6.1 / 7.7 / 20 | **3.7 / 16 / 32** |
| GLM‑5.2 744B | 0.7 / 2.4 / 4.0 | 0.7 / 1.4 / 2.3 | 0.9 / 2.3 / 3.8 | 6.1 / 7.0 / 18 | 2.3 / 12.5 / 24 |

Cells are tok/s as floor / designed / aggregate for five streams; cached‑expert fractions and the seq/par split are printed by Appendix A. Reading it: the kitchen‑sink desk is an excellent gpt‑oss‑120b, Qwen3.6‑35B‑A3B and dense‑70B machine, a usable Qwen3‑235B machine, and a *runs‑it* machine for 671B‑class models at one to two and a half tokens per second — colibri's 42.8k stars say that last category is itself a draw [58]. The five‑developer office mesh is the argument: its ~400 GB of pooled fast memory holds DeepSeek‑V3's routed experts, so it reaches 4–16 tok/s per user and about 30 aggregate, Qwen3‑235B at 12–39, and gpt‑oss‑120b at 85 single‑stream and 260 aggregate — on hardware the office already owns, with prompts that never leave the LAN. Four Strix Halo boxes hold everything and are bounded by their 256 GB/s memory, which is why they look alike across models. The lever is the same in every column: put the routed experts in fast memory somewhere on the mesh. That is what asymmetric sharding across owned devices does and what no single machine can.

#### 3.4 Prefill (token‑stationary weight streaming)

For a prompt of S tokens, essentially every expert is touched, so the whole model streams once; time is the larger of streaming and compute, plus activation transfers at stage boundaries:

```
T_prefill ≈ max( non_resident_bytes / (BW_nvme · η) ,  2 · A_active · S / (FLOPS · MFU) ) + (nodes − 1) · S · d · 1 B / link_BW
crossover (Colibri double buffer fully hidden):  S* = η · FLOPS · b / (2 · BW_tier)
```

Crossover batch for a 4090 (80 TFLOPs effective, b = 0.5625 B/param): **~375 tokens from DDR5 (60 GB/s)** — which is the "B × S ≥ 256" rule in `ARCHITECTURE.md` §3.3, correct for the RAM tier — **~3,200 tokens from Gen4 NVMe (7 GB/s)**, ~1,900 from Gen5, ~19,000 from a 10 GbE peer and ~75,000 from a 2.5 GbE NAS. So NVMe streaming is hidden behind compute only for long prompts or batch jobs, and the NAS is a seed vault, not an execution tier — exactly the role split the architecture gives it [65]. Time‑to‑first‑token for a 4,096‑token prompt: DeepSeek‑V3 on M1 ≈ 25 s (streaming‑bound), on M2 ≈ 2.7 s, on M4 ≈ 3.5 s (compute‑bound at 50% MFU); gpt‑oss‑120b about a second or less everywhere. A 29 MB INT8 activation (4,096 × 7,168) crosses a 2.5 GbE hop in 0.1 s, a 1 GbE hop in 0.23 s and a 25 Mbps residential uplink in 9.4 s — the plan's line‑78 arithmetic is right.

#### 3.5 The remaining mechanisms, checked

| Mechanism (plan / spec) | The math | Verdict |
|---|---|---|
| WAN decode cap (plan §3.1) | tok/s ≤ 1 / (hops × RTT): 5 residential hops × 40 ms → ≤ 5 tok/s; 1 datacenter hop × 30 ms → ≤ 33 tok/s | **Holds**; the single‑hop rule is correct. Add: expert‑parallel across WAN costs L_moe × RTT = 58 × 30 ms = 1.7 s/token → never spread one model's experts across WAN peers. |
| Expert‑parallel on LAN | L_moe × t_hop = 58 × 0.5 ms ≈ 29 ms/token → caps ~35 tok/s on gigabit, fine for 671B‑class; for gpt‑oss (36 layers, 1.9 GB/token) the round‑trips cost more than they save | **Holds** with a planner that picks seq vs par per model. |
| Hivemind INT8 boundaries (plan line 139) | 16 KB BF16 → 8 KB INT8 per token per hop = 50%; "50–75%" needs 4‑bit boundaries | 50% **holds**; 75% is a stretch. |
| Speculative drafts, `K = 4` (plan line 85, "12 → 32–42 tok/s") | Dense weights are read once per verification pass, so gain ≈ accepted tokens per pass (2–3×). Routed experts are *not* shared across the K drafted tokens: the union of their top‑8 experts grows per pass, so expert bytes scale ~2–3× and the gain on NVMe misses ≈ 1×. | **Holds only for resident/dense‑dominated models**; near 1× for cold‑expert streaming. State it per model. |
| KV cache budget (`DistributedPagedKvPlanner`) | DeepSeek MLA ≈ 70 KB/token → 32K = 2.3 GB, 128K = 9.2 GB; GQA 70B ≈ 328 KB/token → 32K = 10.7 GB, 128K = 43 GB | Fits the fast‑memory reservations used above; dense 70B at 128K needs a Mac‑class tier. |
| `< 200 ms` polite yield (spec §3) | Bytes movable in 200 ms: 25 MB on 1 GbE, 62 MB on 2.5 GbE, 250 MB on 10 GbE. A 4K‑context MLA KV slice is ~280 MB | **Holds only if KV deltas stream continuously to the standby shard** (70–330 KB per generated token — trivial); a cold KV copy at handoff takes 0.2–2 s. Worth stating in the spec. |
| DeMo/DisTrO compression | `SPECSHEET` §3 says 10–85×; `readme.md` and `ARCHITECTURE.md` §5.4 say 800–3,000×. For a 70B BF16 pseudo‑gradient (140 GB): 85× → 1.65 GB → 8.8 min on a 25 Mbps uplink; 1,000× → 140 MB → 45 s. With H = 500 inner steps either fits DiLoCo | Workable either way; **fix the inconsistency** — quote the conservative spec figure externally. |
| Verde bisection (spec §3) | ⌈log₂ 64⌉ = 6 → "≤ 7 checks" | **Holds.** |
| Seeding/leeching, RAID‑0/5 striping, multi‑homed links (plan §15, roadmap Phase 6, unbuilt) | Staging, striped‑return, K/V‑exchange, stall‑probability and per‑peer throughput arithmetic in §3.6 | Swarming **holds** for staging (minutes, not hours); striping **holds** for independent sequences, not for one prompt (causal K/V); parity **holds** for data streams and linear stages. Roadmap, not shipped. |
| Cold‑expert cost (Colibri) | 8 × 44.0M × 58 × 0.5625 B = 11.5 GB per token for DeepSeek‑V3 | Matches colibri's independent ~11 GB for GLM‑5.2 [61]. |

#### 3.6 Seeding/leeching and striping (plan §15, roadmap Phase 6): what they change and what they don't

The envelope above leaves them out deliberately: they do not alter single‑stream decode on a LAN mesh, where the limits are memory bandwidth, NVMe and per‑hop latency. They matter in three places — getting weights onto nodes, moving long‑prompt activations across residential uplinks, and surviving stragglers — and the arithmetic for each follows (Appendix A, "Phase 6"). Residential links are taken as the plan does: 500–1,000 Mbps down, 25 Mbps up.

*Seeding and leeching (staging).* `T_stage = bytes / min(receiver_download, Σ seeder_uploads)`:

| Payload | 1 residential seeder | 8 seeders | 32 seeders | LAN NAS, 2.5 GbE | Cloud, 1 Gbps |
|---|---|---|---|---|---|
| One 1/16 layer slice of DeepSeek‑V3 Q4 (23 GB) | 2.0 h | 15 min | **3.8 min** | 1.2 min | 3.1 min |
| Full DeepSeek‑V3 Q4 (370 GB) | 33 h | 4.1 h | 62 min | 20 min | 49 min |
| One cold expert (24.8 MB) | 7.9 s | 1.0 s | 0.25 s | 0.08 s | 0.2 s |

Swarming turns a two‑hour slice fetch into four minutes and hits the receiver's download ceiling at 20–40 seeders (the plan's "16–32" is the right range); it saves the plan's $0.08–0.12/GB of CDN egress, i.e. $30–44 per full‑model pull; a seeder's uplink carries 11 GB/hour, so seeding one full model costs 33 uplink‑hours and seeding micro‑credits should be metered per GB served. What swarming does *not* make viable is pulling cold experts per token over WAN: 0.25 s per expert even from 32 seeders, times 8 experts × 58 layers. WAN peers must execute experts in place (weight‑stationary) or be pre‑staged — which is how lumabri works and what Kadmium's architecture already specifies. Seeding therefore sets time‑to‑ready and failover warmness, not tok/s, and the plan uses it for exactly that (CloudShield standby shards, new workers).

*Sequence striping of one prefill.* One boundary of a 4,096‑token prompt is S·d INT8 = 29.4 MB: 9.4 s over one 25 Mbps uplink, **1.17 s** striped eight ways, 0.59 s sixteen ways, 0.29 s at k = 32, with the receiver's gigabit download becoming the ceiling at k ≈ 40. The plan's "10.6 s → 1.3 s" holds for a boundary. What §15.2 omits is causal attention. FFN and MoE blocks are token‑wise independent, as the plan says (67–92% of FLOPs), but attention is not: chunk j needs the K/V of chunks 1..j−1 for every layer in the slice. There are two ways to satisfy that and both are bandwidth‑bound. If the peers exchange K/V, the first peer must upload its chunk's K/V to the seven later peers: with DeepSeek's MLA (70 KB/token) that is 7 × 36 MB = 251 MB → **80 s**; with a GQA 70B (328 KB/token) it is **376 s**. If attention stays on the trusted core and only FFN/MoE work is striped, the core must send every peer its chunk at every layer: 29.4 MB × 58 MoE layers = 1.7 GB of core *upload* per prompt (27 s on symmetric 500 Mbps fiber, 6.5 min on 35 Mbps cable) plus the peers' returns, 1.17 s × 58 = 68 s. Either way a single 4K prompt striped across residential peers takes one to seven minutes against ~6 s on a local 4090 (§3.4). Striping does not make *one* prompt fast over residential WAN; it makes *many independent sequences* fast — batch documents, RL rollouts, evaluation sets — where there is no K/V to exchange and only end‑of‑sequence results return. That is precisely where §3.1's two‑tier rule already routes residential hosts. The §15 line "rivals a `$2.50/hr` datacenter GPU" is defensible for throughput (tokens per hour), not for latency, and should say so.

*Per‑peer throughput bound (weight‑stationary batch work).* `uplink / bytes_per_token_per_boundary`: 25 Mbps ÷ 7 KB = **436 tokens/s ≈ 1.6M tokens/hour per residential peer per boundary** at d = 7,168 (2.7M/hour at d = 4,096). Ten peers ≈ 16M tokens/hour of boundary capacity. This is the honest "virtual accelerator" number, and it argues for metering Supply Grid work per token processed or per GB served rather than per GPU‑hour (§6.3).

*Expert‑channel striping over WAN.* One activation round‑trip per MoE layer: for decode, 58 × 30 ms = **1.7 s per token** — not interactive; for prefill, dispatching 8 token‑rows per token per layer is 235 MB per layer at S = 4,096, which saturates the dispatcher's uplink. Expert striping is a LAN or 10 GbE‑fleet technique (the studio and campus supply), not a residential one — again consistent with the two‑tier rule.

*RAID‑5/6 parity.* Overhead p/k: 12.5% for 8+1, 25% for 8+2. If each stripe independently stalls with probability q, the probability that a layer waits is:

| q per stripe | k = 8, no parity | 8 + 1 | 8 + 2 |
|---|---|---|---|
| 2% | 14.9% | 1.3% | 0.1% |
| 5% | 33.7% | 7.1% | 1.2% |
| 10% | 57.0% | 22.5% | 7.0% |

That is the mechanism behind the plan's "60–80% p99 reduction", and it is consistent with it for q in the 2–5% range. Linearity caveat: `Y_parity = (Σ X_j) W = Σ Y_j` holds for a linear projection; an FFN or expert is linear → nonlinear → linear, so coded computation applies per linear stage (with an intermediate exchange) or by replication, while Reed–Solomon over *data* streams — weight pages, KV blocks, activations in flight — has no such limit and is the stronger of the two ideas. All of this is Phase 6 and unbuilt; external material should call it roadmap.

*Multi‑homed link striping.* Sum of interface rates, LAN only: 2.5 GbE + Wi‑Fi 6E (~0.9 Gbps real) + Thunderbolt 4 (20 Gbps) ≈ 23 Gbps moves a 29 MB boundary in ~10 ms instead of 94 ms. Helps staging and prefill hops on the LAN; decode is RTT‑bound and unaffected.

*Net effect on the envelope.* §3.3's decode figures are unchanged. LAN prefill improves marginally with bonded links. Time‑to‑ready and failover warmness are where swarming pays — it is what makes WAN staging a matter of minutes. For the residential Supply Grid, the value is throughput on independent sequences and seeding bandwidth, not interactive latency; the datacenter‑burst path remains the interactive accelerator, and residential credits should be metered per token or per GB served.

#### 3.7 Validation against measured systems

The envelope is only useful if it predicts what others have measured. Single‑node checks: an M3 Ultra (819 GB/s) on DeepSeek‑V3.1 at 4‑bit reads 20.6 GB/token → predicted 23.9 tok/s vs **21.1 measured** by EXO [19]; one Strix Halo node (256 GB/s, η 0.5) on Kimi K2.5 at Q2 (~9.6 GB/token) → predicted 13.3 vs **9.45 measured** through llama.cpp RPC across four nodes [22], the gap being RPC's per‑op round trips; a 128 GB M5 Max (~550–600 GB/s unified memory, internal NVMe) running GLM‑5.2 with colibri's learned pins → predicted 1.8 tok/s (α = 0.25) vs **2.06 measured** [61], and 0.75 with uniform routing — which is why α = 0.25 is the "skewed" case above and uniform is the floor. The other published points fall in the same bands:

| System | Model | Fast memory | Decode | Source |
|---|---|---|---|---|
| colibri, laptop, cold cache | GLM‑5.2 (744B) | 25 GB RAM | 0.05–0.1 tok/s | [61] |
| colibri, Core Ultra 9 + RTX 5080, RAM capped | GLM‑5.2 | 25 GB | 0.31–0.38 | [61] |
| colibri, Mac mini M4 Pro | GLM‑5.2 | 48 GB | 0.30 | [61] |
| colibri, M5 Max, learned hot‑expert pin | GLM‑5.2 | 128 GB | 2.06 | [61] |
| colibri, 251 GiB host + six RTX 5090, full residency | GLM‑5.2 | ~250 GB + 192 GB VRAM | 5.8–6.8 (~9 NUMA) | [61] |
| colibri | DeepSeek V4‑Flash (284B) | 16–32 GB | 0.9–1.6 | [58] |
| colibri | Kimi K3 (2.8T) | 32–64 GB | ~0.1 | [58] |
| colibri | Qwen3.6‑35B‑A3B | ~17 GB | 12.8–15.7 | [61] |
| FreeToken, RTX 3090 + 192 GB DDR4 (128 GB would not load) | DeepSeek V4‑Flash | ~190 GB | 8.8–11 | [60] |
| lumabri, experts from disk / in RAM / on gigabit peers / over 30 ms internet | MoE test model | — | 0.04 / 1.92 / 5.97 / 1.13 | [59] |
| llama.cpp RPC, 4 × Strix Halo, 5 GbE | Kimi K2.5 (1T, Q2) | 480 GB | 9.45 | [22] |
| EXO, 1 → 4 × M3 Ultra 512 GB, TB5 | DeepSeek‑V3.1 (671B) | 512 GB → 2 TB | 21.1 → 32.5 | [19] |

Two things the validation says about the plan's language. NVMe streaming makes giant models *possible*: colibri moved from 1.5 to 8.8 GB/s of SSD bandwidth and gained only 0.10 → 0.28 tok/s because compute became the limit [61]. And the step‑change arrives when the routed experts live in fast memory across the mesh — FreeToken's 192 GB, the 250 GB + six‑5090 host, EXO's 2 TB, the M2 office mesh above — which is the Kadmium thesis stated as a number. The plan should carry §3.3's envelope for the demo hardware it actually uses, labelled as modelled, and swap in measurements as `kadmium bench` produces them.

### 4. Why now: tailwinds and demand

Every structural change in 2026 moved toward Kadmium. Fast memory became the scarce resource and the thing people already own became the cheapest supply of it; open models outgrew every single consumer box; the demand for running giant models on owned hardware showed up as star counts and subreddit growth rather than as a forecast; and the hosted alternatives got pricier and more usage‑capped, not less. The table and bullets below carry the evidence; §5 shows there is no one else positioned to catch it as a whole.

#### 4.1 What changed in 2026 and what it means

| Change | Evidence | Implication |
|---|---|---|
| **Memory supercycle.** Consumer DRAM +90–110% in Q1 2026, SSDs +147%; AI/HBM takes ~20% of DRAM output; shortage expected through 2027, possibly 2028 [38]. | Apple drops 512 GB Mac Studio, 256 GB upgrade $2,000 [39]; Framework 128 GB $1,999 → $2,459 [40]; DGX Spark 128 GB $3,999 → $6,950 [41]. | The cheapest 128 GB+ of fast memory most people will ever own is the memory they already have. Pooling is the only "upgrade" that got cheaper this year. Hardware‑tier Kickstarter margins are exposed. |
| **GPU prices rose on both sides of the counter.** 59 of 81 consumer GPUs up Aug→Sept; 5090 resale 2.4× MSRP; used 4090 ~$2,500 [42][43]. Cloud medians: 4090 $0.48, H100 $3.99, B200 $7.99 — all far above the plan's wholesale assumptions [1]. Lightning AI roughly doubled entry GPU rates in Sept 2026 [12]. | Burst COGS are higher and more volatile than the plan assumes. | Index retail to wholesale; favour community supply for 24–32 GB classes; keep datacenter burst for 80 GB+ classes only. |
| **Open‑weight frontier went to 0.75–2.8T params.** GLM‑5/5.1/5.2 ~750B (MIT), Kimi K2.6 1T, DeepSeek V4‑Pro 1.6T / V4‑Flash 284B, Qwen3.5‑397B‑A17B, Qwen3.6‑35B‑A3B, Gemma 4 31B, all permissive [45]; Kimi K3 2.8T [14]. | Tiered‑streaming engines run all of them on consumer boxes — GLM‑5.2 in 16–24 GB of RAM, Kimi K3 in 32–64 GB [58] — at 0.1–2 tok/s; interactive speed arrives when the routed experts sit in fast memory across the mesh (§3). | The strongest argument for Kadmium's pooling: aggregate RAM + VRAM across a LAN is the cheapest route to interactive 300B+ MoE. Publish the §3 envelope per mesh (modelled), then measured tok/s per tier. |
| **Hybrid local→cloud pricing standardised on per‑token.** Ollama's 31 Aug 2026 move cites unpredictable GPU‑time billing [14]. | Users expect "$ per million tokens", not "$ per GPU‑hour". | Meter burst in tokens (cost‑plus on GPU‑hr ÷ measured throughput) or show live $/M‑token estimates. |
| **Decentralised compute got funded — as aggregation.** Prime Intellect: $130M Series A at $1B (Jul 2026), >$100M ARR run‑rate (company‑reported), 6,000 clients, INTELLECT‑3 trained on 512 centralized H200s [26][27]. DePIN: $10B market cap vs $72M FY2025 on‑chain revenue; Akash ~$4.2M ARR; io.net ~$20M annualised; Aethir $127.8M FY2025 from enterprise containers [30]. | Money follows aggregation, RL tooling and enterprise‑grade supply; residential supply alone is a small business. | Lead with the aggregator ("one API over your LAN + every spot market") and with studio/campus fleets; treat residential hosts as a community feature, not the revenue engine. |
| **Local‑inference community keeps compounding.** r/LocalLLaMA 853k members, +55% YoY [36]; Steam 43.5M concurrent record [34]; 12 GB+ VRAM crossed 50% of Steam systems [32][33]. | Demand for running models at home is mainstream among gamers and developers. | The free tier's audience is real and growing; conversion, not reach, is the risk. |
| **Developers choose on quality, not cost.** Stack Overflow 2026 (30k respondents): 73% of AI users use it daily; Claude Code 66% and Copilot 59% among agent users; quality 75% vs cost 38% in model choice; 38% of orgs use usage limits [37]. | "Cheaper than OpenRouter" is a weak hook for the coding segment; "no rate limits, private, same quality" is the hook. | Rewrite Scenario B (line 98) and the outbound scripts. |

---

#### 4.2 Demand signals

- **Appetite for giant MoE on consumer hardware.** colibri (42.8k GitHub stars) and FreeToken (14.4k) — the engines Kadmium's tiering is modelled on — are among the most‑starred inference projects of 2026 [57][58]. People want to run 300B–2.8T models on hardware they own badly enough to accept single‑digit tok/s; a mesh that turns that into interactive speed has a ready audience.
- **Community size and growth.** r/LocalLLaMA: 853k members on 30 Sep 2026, +304k (+55%) in twelve months [36]. This is the plan's primary launch channel and it is roughly a third larger than when most 2025‑era plans were written.
- **Developer behaviour.** Stack Overflow 2026 (30k+ respondents): 73% of AI‑tool users use them daily, a third for 4+ hours a day; coding agents 66% / chatbots 63%; Claude Code 66% and Copilot 59% among agent users; quality (75%) outweighs cost (38%) in model choice; 38% of organisations impose usage limits and 21% say AI spend is rising [37]. Rate limits and spend caps are a felt pain; raw price is not the top criterion.
- **Willingness to pay for hybrid local/cloud.** Ollama's paid tiers and its switch to per‑token pricing [14]; Lightning AI's price increases with a shrunken free tier [12]; DGX Spark demand strong enough for NVIDIA to add a 64 GB SKU at $4,999 while raising the 128 GB to $6,950 [41].
- **Market sizing (for decks, not for planning).** Roots Analysis puts GPU‑as‑a‑Service at **$10.8B in 2026 → $132.4B by 2040 (19.6% CAGR)**, public cloud 63% share, fractional GPU services the fastest segment at 25.5% CAGR, AI/ML 41% of demand [31]. Treat as an order‑of‑magnitude reference only.

### 5. Category: no direct competitor, and a map of the adjacent ones

The original plan tears down Ollama, LM Studio, Tailscale, Salad and the serverless platforms. The tables below add the categories it skips and correct the ones it has. "Kadmium gap" is the capability the competitor lacks that Kadmium's README claims; "Kadmium exposure" is where the competitor is ahead.

#### 5.0 Is there a direct competitor? No — and that cuts both ways

Kadmium's README defines the product as BOINC‑style polite host harvesting plus CERN‑style peer‑to‑peer data staging applied to AI, built on FreeToken/colibri‑style tiered execution, Hivemind/Petals swarm routing, DiLoCo/DeMo training, Verde verification and a credits marketplace with cloud burst [65]. Nothing shipping combines those. The nearest things each do one or two pieces:

| Piece of Kadmium | Closest existing thing | Status |
|---|---|---|
| Single‑machine tiered MoE execution (q\* split, expert cache, NVMe streaming, coding‑agent API) | FreeToken (14.4k stars, Apache‑2.0, Anthropic/OpenAI APIs) [57]; colibri (42.8k stars, pure C, CPU/Vulkan/CUDA/Metal) [58] | Shipping, large communities |
| P2P swarm execution of MoE experts across ordinary machines | lumabri (colibri‑based; 11 stars; "working prototype") [59]; Petals (dormant since 2024) [25] | Prototype / dormant |
| LAN cluster of heterogeneous devices | EXO (Apple‑first) [18][19]; llama.cpp RPC [22][24]; GPUStack (Linux‑only workers) [20] | Shipping; none covers Windows + NVIDIA + Mac + SBC + NAS together |
| Community pool for LLM inference, paid in kudos | Kalavai (planned Llama‑3.1‑405B at full precision across ≥ 488 consumer GPUs, Oct 2024) [62]; AI Horde (community art/text cluster, active, kudos economy) [63] | Small; no frontier‑MoE sharding; no cash or credits |
| Paid consumer‑GPU supply network | Salad (60k+ daily GPUs, whole containers per host) [8]; Vast (17k GPUs, marketplace) [4] | Shipping; cash payouts; no LAN pooling, no privacy pinning |
| Decentralised training | Prime Intellect, Nous Psyche, Gensyn, Templar [26][28][29] | Shipping/testnets; token‑ or venture‑funded; flagship runs centralised |
| Polite background harvesting with credit accounting | BOINC; Folding@home peaked at 2.43 exaFLOPS in April 2020 with volunteer machines [64] | Decades of precedent; not ML‑aware |
| Local→cloud hybrid with credits | Ollama (per‑token credits, 31 Aug 2026) [14] | Shipping; all‑local or all‑cloud per request |

What "no 1:1 competitor" implies. There is no price anchor for the whole product, so pricing has to be built from costs and from the adjacent anchors (Ollama for hybrid credits, Lightning for seats, Salad/Vast for host rates, the GPU index for burst). Category creation carries an education cost, which is why measured demos matter more than feature lists. And every adjacent player can extend into the gap — FreeToken adding multi‑node, EXO adding Windows/NVIDIA, Ollama adding LAN pooling, Salad adding sharding — so the moat is how fast Kadmium ships the mesh features they lack. The volunteer‑computing precedent is the encouraging half: BOINC‑style projects have recruited hundreds of thousands of machines for credit, mission and community rather than cash [64], which supports a credits‑first Supply Grid over Salad's cash model, provided the credits are accounted for honestly (§6.3).

#### 5.1 Free local runtimes (the "pay $0 forever" baseline)

| Project | Status (Oct 2026) | What it does well | Kadmium gap it leaves | Kadmium exposure |
|---|---|---|---|---|
| **FreeToken** (FlashML‑org) | Apache‑2.0, 14.4k stars [57] | Single‑box giant MoE: q\* CPU–GPU co‑execution, global LRU expert cache, double‑buffered prefill, FTW weight format, MXFP4/NVFP4/FP8, RTX 30/40/50; Anthropic + OpenAI APIs for coding agents [57]; DeepSeek V4‑Flash at ~9–11 tok/s on an RTX 3090 with 192 GB DDR4 [60] | One machine only; no LAN pooling, polite yield, credits, burst or training | Owns the solo‑developer "frontier MoE on my gaming PC" story that the plan's free tier targets; same agent‑API surface |
| **colibri / lumabri** (JustVugg) | colibri Apache‑2.0, 42.8k stars; lumabri 11 stars, prototype [58][59] | Pure‑C expert streaming: GLM‑5.2 744B in 16–24 GB RAM, Kimi K3 2.8T in 32–64 GB; lumabri delegates expert execution to LAN/internet peers (~6 tok/s over gigabit) [58][59] | Speed: 0.05–2 tok/s for 744B on single machines [61]; no scheduler, metering, failover, privacy tiers or training | colibri's star count is the clearest demand signal for the category; lumabri is the nearest conceptual sibling to Kadmium's swarm |
| **EXO** (exo‑explore) | v1.x, Apache‑2.0 [19] | Auto‑discovery of Apple devices; MLX backend; disaggregated prefill on DGX Spark + decode on M3 Ultra, 2.8× on 8B over 10 GbE [18]; 4× M3 Ultra runs DeepSeek‑V3.1 671B at 32.5 tok/s (~1.5× one node) [19] | No official NVIDIA/AMD/Windows path (community `exo-cuda` tinygrad fork only) [19]; no gaming yield; no burst/cloud layer; no SBC/NAS roles | Owns the Mac‑cluster mindshare and the "frontier on your desk" headlines; the 512 GB Mac Studio's retirement [39] hurts its flagship config |
| **llama.cpp RPC** | Mainline feature; AMD publishes cluster guides on it [22] | Runs anywhere llama.cpp runs; 1T‑class Kimi K2.5 at Q2 across 4 × 128 GB Strix Halo boxes at 9.45 tok/s (5 GbE) [22]; zero cost | Capacity, not speed: split 72B decode falls 11.1 → 5.9 tok/s even on direct 10 GbE [24]; manual layer placement; no scheduler, auth, metering, failover or cloud burst | It is the default answer on r/LocalLLaMA to "how do I pool my machines"; Kadmium must publish head‑to‑head numbers on *mixed* hardware |
| **GPUStack** | Apache‑2.0, ~5.9k stars [21] | Kubernetes‑style cluster manager; vLLM/SGLang/TensorRT‑LLM; NVIDIA, AMD, Ascend, Hygon, MThreads… [20] | **Workers Linux‑only** [20] — no Mac, no Windows gaming PC; datacenter‑shaped | Strong for Linux labs/SMBs; the obvious comparison for the `Lab & Team` tier |
| **vLLM + Ray** | Production standard | Tensor + pipeline parallel; 4×4090 over 2 nodes ≈ 52–60 tok/s on Llama‑3.1‑70B [23] | Homogeneous CUDA assumption; NCCL setup pain; no consumer‑grade yield | Ceiling on raw throughput for NVIDIA‑only shops |
| **Petals** | Dormant: last push Sep 2024, last release Sep 2023 [25] | Proved BitTorrent‑style public swarms work for 100B+ models | — | None; cite as prior art and lessons (WAN decode latency) |
| **NVIDIA Dynamo** | Production framework | Disaggregated prefill/decode, KV‑aware routing [23] | Overkill for home/office; datacenter‑only | Sets the industry vocabulary ("disaggregated serving") that Kadmium's `DisaggregatedDualStream` shares; useful for positioning |

#### 5.2 Hosted local→cloud hybrids (the monetisation comparables)

| Company | Pricing (Oct 2026) | Notes |
|---|---|---|
| **Ollama** | Free (small monthly usage) · Pro $20/mo → $60 usage · Max $100 → $300 · Team $500 → $1,000 shared, unlimited users; per‑token billing; no rollover; ZDR; US/EU compute [14] | Moved off GPU‑time billing on 31 Aug 2026 because customers found it unpredictable [14]. Still all‑local or all‑cloud per request. |
| **LM Studio** | Free at home and at work (since Jul 2025); Teams (self‑serve) and Enterprise (SSO, model/MCP gating) paid [16] | No cloud credit product. "Bionic" is the app edition name. |
| **Lightning AI** | Free: one‑time 30 credits · Pro $50/mo ($20 annual, 240 credits/yr) · Teams $140/user/mo · H100 $4.50/hr, A100‑40GB $2.19, T4 $0.55 after the 21 Sep 2026 reprice [12] | The UI reference for Kadmium's dashboard; note it monetises **per seat** at $119–$140/user — Kadmium's $199/mo for a 25‑node lab is cheap by comparison. |
| **Tailscale** | Free personal; paid team governance; ~US$1.5B valuation (Apr 2025); 10,000+ customers [17] | The open‑core governance template the plan copies — still valid. |

#### 5.3 GPU marketplaces and neoclouds (wholesale sources and host‑side comparables)

| Provider | Scale / price points | Relevance |
|---|---|---|
| **Vast.ai** | 17,000+ GPUs from ~1,400 hosts in 500+ locations; 700k+ rentals/month; H100 $0.90–$1.87 (verified $1.50–$1.87), A100 $0.50–$0.80, 4090 $0.34–$0.50; unverified hosts carry a 20–40% reliability tax; no automatic failover [4]. Live prices ≈ 25% above host earnings [5]. | Cheapest wholesale, least reliable. A good spot source only behind Kadmium's own failover. Host take ≈ 20% of gross. |
| **SaladCloud** | 60k+ daily‑active GPUs, 100k+ monthly suppliers, 450k+ nodes lifetime, 191 countries, $7M+ paid to hosts lifetime [7][8]; customer 4090 $0.16–$0.33, 5090 $0.25–$0.50 (Sep 2026) [6]; host 4090 $0.11–$0.20, 5090 $0.18–$0.30, 3090 $0.06–$0.10 (Sep 2026 rate post) [7] | The direct residential‑supply comparable. $7M lifetime payouts across 450k nodes (~$15/node, derived) shows how little residential hosting pays. |
| **RunPod** | Community 4090 ~$0.34, secure ~$0.69; A100 ~$1.49; H100 $2.69–$2.99 [3] | Reliable interactive burst source; the plan's $0.29/$1.39 figures are stale. |
| **Spheron** | 4090 $0.48, 5090 $0.78, L40S $0.96, A100 $1.43, H100 $2.65 ($2.20 spot), B200 $9.95 [2] | A mid‑tier reference point. |
| **Modal** | T4 $0.59, L4 $0.80, L40S $1.95, A100‑80GB $2.50, H100 $3.95, H200 $4.54, B200 $6.25 per hour; region pinning ×1.5–1.75 [9] | The failover bridge in the plan; L40S is 2.4× what the plan assumes. |
| **Together / Baseten / Fireworks** | Dedicated H100 $6.49 / $6.50 / $7.00; B200 $9.95–$11.95; serverless DeepSeek V4‑Pro $1.74–$2.10 in, $3.48–$4.40 out [10][11] | The "$3.50/hr 24/7 endpoint" segment in the plan actually pays $6.49–$7.00 — the replacement pitch is stronger. |
| **Cross‑provider index** | 85 providers, 3,938 listings; cheapest / median: 4090 $0.33 / $0.48, 5090 $0.35 / $0.72, L40S $0.55 / $1.64, A100 $0.89 / $1.80, H100 $1.80 / $3.99, H200 $2.09 / $5.00, B200 $3.75 / $7.99 [1] | Use as the wholesale index for retail pricing (§6.2). |

#### 5.4 Decentralised training and compute networks

| Project | Oct 2026 status | Takeaway |
|---|---|---|
| **Prime Intellect** | $130M Series A at $1B (Jul 2026), led by Radical Ventures with NVIDIA, Intel and Dell ventures; >$100M ARR run‑rate (company‑reported, unaudited), 6,000+ clients, ~40 staff; INTELLECT‑3 (106B MoE) trained on 512 centralized H200s; compute marketplace + hosted training/RL + sandboxes [26][27] | Validates "one API over many GPU sources + RL tooling" as a venture‑scale business. Not a residential‑sharding story. Potential wholesale partner *and* the best‑funded adjacent competitor. |
| **Nous Research (Psyche)** | Consilience 40B pre‑training run over the internet; next phase trains Hermes 4 on Seed‑OSS‑36B; runs coordinated by Solana smart contracts [28]; contributors with a single 24 GB GPU earn token rewards [29] | Token incentives are the mechanism Kadmium explicitly rejects; Kadmium's non‑crypto credits are a compliance advantage and a recruiting disadvantage. |
| **Gensyn** | RL Swarm testnet (May 2026); mainnet claims appear only in promotional sources [29] | Watch; verification‑of‑learning ideas overlap with Kadmium's `Verde` referee. |
| **Pluralis / Templar (Bittensor) / Macrocosmos** | Protocol‑level or subnet pre‑training; no audited scale figures [29] | Research‑adjacent; not a commercial threat to the LAN‑first wedge. |
| **DePIN compute (Akash, io.net, Aethir, Render)** | Sector: ~$10B circulating market cap on ~$72M FY2025 on‑chain revenue. Akash ~$4.2M ARR at >80% utilisation; io.net ~$20M annualised; Aethir $127.8M FY2025 (enterprise containers, 440k+ GPU containers, 94 countries); Render AI inference 35–40% of jobs [30] | Consumer‑GPU marketplaces are structurally small; enterprise‑grade supply makes the money. Supports the plan's emphasis on studio/campus fleets over residential hosts. |

#### 5.5 Serverless open‑model APIs (the price floor Kadmium cannot beat on $/token)

DeepSeek‑V3.2 is offered by 14+ providers from **$0.209 / $0.31** (GMICloud) to $3.00 / $4.50 per 1M tokens, with OpenRouter at **$0.28 / $0.42** and cache reads at $0.028 [13]. gpt‑oss‑120b is $0.15 / $0.60 and gpt‑oss‑20b $0.07 / $0.30 on Fireworks [11]. The 2026 frontier‑open tier (DeepSeek V4‑Pro, Kimi K2.6, GLM‑5.1) sits at $0.95–$2.10 in and $3.48–$4.40 out [10][11]. Any Kadmium pitch that quotes a per‑token saving against the first group will lose; against the second it is credible.

---

### 6. The business, in numbers

#### 6.1 Model and anchors

Kadmium is open‑core in the Tailscale/Supabase sense the README states [65]: the single binary and everything needed to run a personal, lab or enterprise mesh are free forever; revenue comes from governance and convenience (WAN relay, RBAC, observability, encrypted checkpoint mirroring), from burst compute, from the CloudShield failover service, and from the Supply Grid. There is no direct competitor to price against (§5.0), so each layer takes its anchor from the nearest adjacent market: Ollama's per‑token tiers for the hybrid subscription ($20 → $60 usage, $100 → $300, $500 → $1,000 shared) [14]; Lightning AI's per‑seat pricing for team tiers ($119–140 per user per month) [12]; the 85‑provider GPU index for burst [1]; Salad's and Vast's host rates for supply [4][7]; serverless token prices for what Supply Grid batch capacity is worth [13]. The plan's tier structure (Local $0; Burst Packs; CloudShield Pro $29; Lab & Team $199; Private Office Fabric $499) fits those anchors — if anything Lab & Team is cheap against per‑seat norms — and the sections below re‑derive the two layers whose numbers changed: burst pricing and the Supply Grid.

#### 6.2 Burst pricing at October‑2026 wholesale

Wholesale columns from the 85‑provider index [1] and provider pages [2][3][4][9]. "Realistic verified" is a bookable, SLA‑worthy rate (RunPod secure/community, Spheron, Vast verified), not a marketplace floor.

| Class | Plan wholesale → retail (margin) | Cheapest listing | Realistic verified wholesale | Median | Margin at realistic wholesale |
|---|---|---|---|---|---|
| RTX 4090 24 GB | $0.29 → **$0.45** (35.5%) | $0.33 (Salad) | $0.34 (RunPod community) – $0.48 (Spheron) | $0.48 | **−7% to +24%** |
| L40S 48 GB | $0.52 → **$0.75** (30.7%) | $0.55 (Novita) | $0.96 (Spheron) – $1.95 (Modal) | $1.64 | **negative** |
| A100 80 GB | $0.82 → **$1.19** (31.1%) | $0.89 | $1.43 (Spheron) – $1.49 (RunPod) | $1.80 | **negative** |
| H100 80 GB | $1.65 → **$2.29** (27.9%) | $1.80 | $1.50–$1.87 (Vast verified) – $2.65–$2.99 (Spheron/RunPod) | $3.99 | **−30% to +34%** |

Two consequences. First, **fixed retail $/hr rates printed in the plan (and on the site) will either lose money or look expensive within a quarter** because the underlying market moved 20–60% in six months. Price burst as `index(class) × (1 + spread)` with a published floor, refreshed daily from the provider APIs the `SpotOrderbookBidEngine` already calls, or bill per token (§11). Second, **the 24–32 GB classes are where community supply must carry the load**: the datacenter price of a 4090‑hour ($0.33–$0.48) is within a few cents of a residential host's power bill plus a fair credit, so the only way to earn a margin on that class is to fulfil it on hosts paid in credits (with the liability modelled, §6.3). Keep paid datacenter burst for the 80 GB+ classes where the retail price has room.

#### 6.3 The Supply Grid: credits done honestly, and the market it is

The plan states three incompatible things: hosts receive "1.25× Kadmium Cloud Credits" (line 19), a 4090 host receives "$0.90–$1.15/hr in credits" (line 114), and a 4090 burst retails at $0.45/hr (line 128). Pick one definition and use it everywhere:

```
credit_rate(class, hour) = m × wholesale_index(class)        # m = 1.00–1.25
host_credits_per_hour     = credit_rate × availability_score   # 0–1, from telemetry
```

At today's index a 4090 host earns **$0.33–$0.41/hr in credits** (m = 1.0–1.25), i.e. 3–4× Salad's cash rate — the "feels generous" effect the plan wants, without promising $0.90–$1.15.

Credits are not free. Each credited hour is a burst hour Kadmium must later fulfil. Model it as:

```
credit_COGS = credits_issued × redemption_rate × blended_fulfilment_cost_per_credit_dollar
```

where fulfilment on community hosts costs Kadmium *more credits* (a 1.25× multiplier compounds 25% per hop, so credits need an expiry and a cash sink such as subscriptions) and fulfilment on datacenter spot costs cash at the wholesale index. Illustrative month with 1,000 hosts × 8 h × 30 nights = 240k host‑hours at $0.41 → **$98k of credits issued**; at 70% redemption with 60% of redemptions landing on community supply and 40% on datacenter spot at $0.40, cash COGS ≈ 0.7 × 98k × 0.4 × ($0.40/$0.41) ≈ **$27k**, plus a credit float that grows unless credits expire. The plan's "$0 cash leaves the company" (line 19, 114) is true only in the month the credit is issued. Suggested guard‑rails: 12‑month expiry, m = 1.0 for cash‑equivalent classes and 1.25 only for overnight windows, and a reported "credit liability" line in the finance model.

**What the code already does, and the market the Supply Grid actually is.** `SupplySettlementEngine` prices a CUDA host with ≥ 20 GB VRAM at **$0.38/hour** in USD, an Apple Silicon host at $0.29, an SBC at $0.06 and a CPU node at $0.05, scaled by the leased‑VRAM fraction, and pays credits at **1.25× the USD rate** — $0.475/hour for a 4090‑class host — with a default lease of 12 hours a day, verified compute receipts and per‑workload masks [65]. That is three to four times Salad's cash rate for the same card ($0.11–0.20 [7]) and consistent with this section's formula if the $0.38 base is indexed to the daily wholesale rate rather than hard‑coded. It also tells us what the Supply Grid sells. A residential peer in weight‑stationary batch mode carries about 1.6M tokens per hour of boundary traffic on a 25 Mbps uplink (§3.6); ten such peers are ~16M tokens/hour, which at serverless list prices for V3‑class tokens ($0.15–0.28 per million input [11][13]) is $2.4–4.5 of market value per hour, or $1,700–3,200 a month, against roughly $0.90/hour of electricity across the ten hosts (§8.2). Sold as batch tokens — RL rollouts, evaluation sweeps, document processing, DiLoCo islands, seeding — at a discount to serverless and fulfilled on credit‑paid hosts, the Supply Grid has a revenue line of its own that scales with peers, while the BOINC precedent (hundreds of thousands of machines for credit and mission [64]) says the supply can be recruited. Metering it per token processed and per GB served, rather than per GPU‑hour, is what makes the price legible to buyers and the credit honest to hosts.

#### 6.4 Who buys and why: scenarios re‑priced (5‑developer team, 30 days, 80/20 input/output)

| Daily tokens | DeepSeek‑V3.2 via OpenRouter ($0.28/$0.42) [13] | gpt‑oss‑120b ($0.15/$0.60) [11] | Qwen3.6 Plus ($0.50/$3.00) [11] | DeepSeek V4‑Pro ($1.74/$3.48) [11] | DeepSeek V4‑Pro via Together ($2.10/$4.40) [10] | Closed frontier (~$3/$15, illustrative) |
|---|---|---|---|---|---|---|
| 5M | $46 | $36 | $150 | $313 | $384 | ~$810 |
| 15M | $139 | $108 | $450 | $940 | $1,152 | ~$2,430 |

The plan's Kadmium bundle for this team is $199 + ~$120 burst = **$319/mo**. It loses on price to every model a 5‑developer LAN runs at interactive speed (the first three columns) and wins only against the 750B+ frontier‑open tier or closed models. The envelope in §3.3 shows what that LAN can do with those: a five‑machine office mesh with ~400 GB of pooled fast memory runs DeepSeek‑V3‑class models at 9–11 tok/s and Qwen3‑235B at ~13, GLM‑5.2‑class at 6–9 — usable for agents and batch work, below datacenter speed for chat; the 1T+ class is where the burst layer does the work. The defensible pitch is therefore: *(a)* zero per‑token cost and zero data egress for everything the mesh runs at interactive speed (gpt‑oss‑120b ~38 tok/s, dense 70B ~15, Qwen3‑235B ~13 on that office mesh, plus the small MoEs on any single box); *(b)* no 429s or org usage caps (a pain 38% of orgs report [37]); *(c)* the 671B–750B tier running on owned hardware at 6–11 tok/s for agents and overnight batches, with cost‑plus burst when a session needs datacenter speed or the 1T+ tier. Quality parity, not price, is the lead (75% vs 38% [37]).

#### 6.5 Revenue cases: the plan's arithmetic is right, its conversion is heroic

The Day‑90 cash of $110,637 (line 169) adds up from its components, but it assumes 6,000 active free clusters converting to **520 CloudShield Pro subscribers (8.7%), 1,150 burst packs (19%), 95 paying labs and 28 regulated SMBs within 90 days** — before a sales team exists. For comparison, Akash reached ~$4.2M ARR (~$350k/mo) after years at >80% utilisation [30]; Salad has paid hosts $7M in its lifetime [8]; Prime Intellect's $100M run‑rate came with $150M raised and 6,000 clients [26]. Two extra cases on the plan's own prices:

| Case | Pro conv. | Pack buyers | Labs (new setups) | SMBs (new setups) | Monthly cash |
|---|---|---|---|---|---|
| Plan | 8.7% (520) | 19% (1,150 × $45) | 95 (10) | 28 (6) | **$110,637** |
| Base | 3% (180) | 6% (360 × $45) | 30 (5) | 8 (2) | **≈ $35,900** |
| Bear | 1.5% (90) | 3% (180 × $45) | 10 (2) | 3 (1) | **≈ $16,200** |

Open‑source freemium conversion of 1–5% is the norm; the plan's rate would be exceptional. Keep the plan as the stretch case and budget on Base. The COGS line (line 170) should also carry the credit liability from §6.3.

---

#### 6.6 Kickstarter hardware tiers under 2026 memory prices

| Tier (plan line 616/618) | Plan BOM | 2026 evidence | Risk / fix |
|---|---|---|---|
| `Cd‑48 Control Rod` RK3588 32 GB + 2 TB NVMe, $699 pledge | $185 board + $95 NVMe/etch + $25 credits = $305 | 32 GB RK3588 boards launched at $159–$197 in Oct 2023 [44]; DRAM and NAND roughly doubled in H1 2026 [38] | Budget $240–$300 for the board and $110–$140 for 2 TB NVMe; margin falls from 48% to ~35–40% at 8–10% fees. Still viable; order boards before the campaign closes. |
| `Cd‑48 Reactor Core` Strix Halo 128 GB, $4,499 pledge | $3,150 unit + $90 = $3,240 | Framework 128 GB retail $2,459 (Jan 2026, up $460) [40]; DGX Spark 128 GB $6,950 [41]; shortages through 2027–28 [38] | BOM is conservative *today*; the risk is a further step‑up before fulfilment. Cap quantity, state "built to order, ships in 8–12 weeks", and keep a 15% price‑protection buffer. |
| Fees | 8% | 5% platform + 3–5% processing = 8–10% [52] | Use 9%. |

---

## Part B — Evidence and plan audit

### 7. Plan audit, claim by claim

Verdict key: **OK** = supported by a current source · **UPDATE** = directionally right, number stale or imprecise · **WRONG** = contradicted by evidence · **UNVERIFIED** = no public source found either way.

#### 7.1 GPU wholesale / retail pricing (lines 18, 35–40, 227–231)

| Line | Claim in plan | Evidence (20 Oct 2026) | Verdict |
|---|---|---|---|
| 37 | RTX 4090: RunPod `$0.34–$0.89`, Vast `$0.18–$0.45`, wholesale **$0.29**, retail **$0.45**, margin 35.5% | Cheapest on‑demand 4090 across 85 providers: **$0.33** (Salad), median **$0.48** [1]. RunPod community ~$0.34, secure ~$0.69; Vast $0.34–$0.50 on unverified hosts [3]. Spheron $0.48 [2]. Salad customer price $0.16 (batch) – $0.33 (high priority) [6]. | **UPDATE** — wholesale $0.29 is only reachable on batch/interruptible tiers; retail $0.45 is below the market median. |
| 38 | L40S: wholesale **$0.52**, retail **$0.75**, margin 30.7% | Cheapest **$0.55** (Novita), median **$1.64** [1]; Spheron $0.96 [2]; Modal $1.95 [9]. | **WRONG** at retail — $0.75 undercuts nearly every provider; margin is negative at any verified source. |
| 39 | A100 80 GB: wholesale **$0.82**, retail **$1.19**, margin 31.1% | Cheapest **$0.89**, median **$1.80** [1]; Vast $0.50–$0.80 unverified [3][4]; RunPod community $1.49 [3]; Spheron $1.43 [2]; Modal $2.50 [9]; Lightning A100‑40GB $2.19 [12]. | **UPDATE** — only Vast's unverified tier (with a 20–40% "reliability tax" [4]) gets under $0.82. |
| 40 | H100: hyperscalers `$4.50–$6.90`, RunPod `$1.99–$2.89`, Vast `$1.10–$2.40`, wholesale **$1.65**, retail **$2.29**, margin 27.9% | Cheapest **$1.80**, median **$3.99** [1]; Vast $0.90–$1.87 (verified hosts $1.50–$1.87) [4]; RunPod $2.69–$2.99 [3]; Spheron $2.65 on‑demand / $2.20 spot [2]; AWS p5 ~$6.88, Azure ~$12.29 [2]. | **UPDATE** — retail $2.29 is below RunPod secure; margin on $1.80 wholesale is 21%, not 28%. |
| 18 | Wholesale spot `$0.29–$0.82/hr` fulfils bursts at 28–35% gross margin | See above. At realistic verified wholesale the spread is 0–25%. | **WRONG** as a planning assumption. |
| 66 | Dedicated endpoints cost `$3.95–$6.50/hr` per H100 | Together $6.49 (endpoint) / $5.49 (cluster); Baseten $6.50; Fireworks $7.00; Modal $3.95 serverless [9][10][11]. | **UPDATE** — range is now $5.49–$7.00 for dedicated; Modal is the $3.95. The "85–92% cheaper" arithmetic on line 67 still holds against $6.49–$7.00. |
| 227 | RunPod `$0.29` (4090) / `$0.59` (L40S) / `$1.39` (H100) | 4090 ~$0.34 community; H100 $2.69–$2.99 [3]. | **WRONG** — H100 is off by ~2×. |
| 228 | Modal `$0.80` (L40S) / `$2.10` (A100) / `$3.95` (H100) | Modal Jul 2026: L40S **$1.95**, A100‑80GB **$2.50**, H100 $3.95, H200 $4.54, B200 $6.25; region pinning adds 1.5–1.75× [9]. | **WRONG** on L40S (2.4× off). |
| 229 | Vast verified datacenter `$0.18` (4090) / `$0.38` (L40S) / `$1.10` (H100) | Vast 4090 $0.34–$0.50; H100 verified $1.50–$1.87 [3][4]. | **UPDATE** — those are unverified‑host spot floors, not verified‑datacenter rates. |
| 230 | Prime Intellect `$0.25` (4090) / `$0.35–$1.49` (H100) | Prime Intellect is an aggregator ("meta‑cloud") over other providers [27]; no public rate card found. | **UNVERIFIED** |
| 231 | SF Compute `$1.00–$2.28` per H100 in 8‑GPU blocks | Not re‑verified in this pass. | **UNVERIFIED** |
| 35 | Table header: "Live … Benchmarks (Q3/Q4 2026)" | Consumer GPU prices rose Aug→Sept on 59 of 81 cards [42]; cloud index medians sit 20–120% above the plan's wholesale assumptions [1]; Lightning AI raised entry GPU rates in Sept [12]. | **UPDATE** — add a date stamp and a refresh cadence. |

#### 7.2 Competitor teardown (lines 11, 44–67, 588)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 46 | Ollama Cloud: Pro $20 (+$60 credits), Max $100 (+$300), Team $500 (+$1,000 shared) | Exactly Ollama's 31 Aug 2026 "transparent pricing" tiers; billing is now per‑token with included monthly usage, no rollover, zero data retention, US/EU compute [14]. (Pre‑Aug‑31 the tiers were "50× free usage" style with Max closed to new signups and Team at $25/seat ×5 min [15].) | **OK** — note the per‑token switch and its rationale. "3 / 10 concurrent" matched the old scheme [15]; not stated on the new page. |
| 47 | "LM Studio Bionic: free local runtime + paid team/organization cloud credit pools with ZDR" | LM Studio is free for work since 8 Jul 2025; paid tiers are Teams (self‑serve, private artifact sharing) and Enterprise (SSO, model/MCP gating). "Bionic" appears only as the app/edition name in download CTAs [16]. No cloud credit pool product found. | **WRONG** — rewrite as "LM Studio (Teams/Enterprise)". |
| 48 | Both are "binary (all‑local OR all‑cloud)" | Consistent with their product pages [14][16]. | **OK** |
| 52 | Tailscale "$1.5B+" open‑core business | US$160M Series C at ~CA$2B (~US$1.5B), April 2025; 10,000+ customers [17]. | **OK** — write "~$1.5B valuation (Apr 2025)". |
| 58 | Salad: "tens of thousands of gamers … 190 countries" | Salad self‑reports 60,000+ daily‑active GPUs, 100,000+ monthly‑active supplier accounts, 450,000+ nodes lifetime, 191 countries, **$7M+ paid to hosts lifetime** [7][8]. | **OK** (understated). Note: $7M over 450k nodes ≈ $15 per node lifetime (derived). |
| 60 | Salad "capped at 16–24 GB VRAM", can't shard | Salad lists 5090 (32 GB) but its own guidance says nodes top out around 24 GB and to avoid large‑model VRAM needs [8]. | **OK** (minor: 32 GB exists). |
| 11, 588 | EXO "required identical Macs"; EXO/llama.cpp RPC are the free‑local trap | EXO 1.0 (Apache‑2.0) discovers heterogeneous Apple devices and already pairs a **DGX Spark (prefill) with an M3 Ultra (decode)** for a 2.8× end‑to‑end gain on Llama‑3.1‑8B over 10 GbE [18]. NVIDIA consumer GPUs still only via the community `exo-cuda` tinygrad fork; scaling across 4 M3 Ultras on DeepSeek‑V3.1 671B is 21.1 → 32.5 tok/s (~1.5×) [19]. | **UPDATE** — EXO is heterogeneous *within Apple + DGX Spark*; its gap is Windows/NVIDIA/AMD/SBC. Don't present disaggregated prefill/decode as unique to Kadmium. |
| 77, 471, 588 | Petals as a live competitor | Last push 7 Sep 2024, last release v2.2.0 6 Sep 2023, ~10k stars, not archived [25]. | **UPDATE** — dormant; cite as prior art. |
| — | (not covered) GPUStack | Apache‑2.0, ~5.9k stars; server runs anywhere in Docker, **workers are Linux‑only**; backends vLLM/SGLang/TensorRT‑LLM; NVIDIA/AMD/Ascend/Hygon/MThreads etc. [20][21]. | **ADD** — closest "cluster manager" competitor; no Mac/Windows workers. |
| — | (not covered) llama.cpp RPC in 2026 | AMD's own Feb 2026 guide runs **Kimi K2.5 (1T, ~375 GB at Q2) across 4 Framework Desktops (4×128 GB) on 5 GbE at 9.45 tok/s** with llama.cpp RPC [22]. Independent Mac Studio + DGX Spark RPC test: 72B decode falls from 11.1 to 5.9 tok/s when split; "adds capacity, not speed" [24]. 10 GbE is the recommended floor [23]. | **ADD** — this is the free baseline Kadmium's LAN sharding must beat. |
| 65–67 | Modal/Baseten/Together/Fireworks | See §7.1 rows 66. Lightning AI (the UI reference for the dashboard) repriced on 21 Sep 2026: Pro $50/mo ($20 annual), Teams $140/user/mo, H100 $4.50/hr, A100‑40GB $2.19, T4 $0.55; free tier cut to a one‑time 30‑credit grant [12]. | **UPDATE/ADD** |

#### 7.3 Serverless API benchmarks and buyer scenarios (lines 92–100)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 92 | OpenRouter DeepSeek‑V3 / Qwen‑Coder `$0.20–$0.60` in, `$0.60–$1.80` out per 1M | DeepSeek‑V3.2 on OpenRouter: **$0.28 in / $0.42 out / $0.028 cache‑read**; cheapest provider $0.209 / $0.31 [13]. Frontier open models: DeepSeek V4‑Pro $1.74/$3.48 (Fireworks, Baseten), $2.10/$4.40 (Together); Qwen3.6 Plus $0.50/$3.00; gpt‑oss‑120b $0.15/$0.60; Kimi K2.6 $0.95/$4.00 [10][11]. | **UPDATE** — output prices for V3‑class fell ~3×; a new 2026 frontier‑open tier sits at $1.7–$4.4. |
| 96 | Casual solo chat: OpenRouter wins | Still true; ~40k tokens/day ≈ $0.40/mo at V3.2 rates (derived). | **OK** |
| 98 | Heavy multi‑agent coding, 5–15M tokens/day → "$900–$2,400/mo" on OpenRouter; Kadmium $319/mo wins by 70–85% | At V3.2 rates (80/20 in/out, 30 days): **$46–$139/mo**. At Qwen3.6 Plus: $150–$450. At DeepSeek V4‑Pro: $313–$940 (Fireworks) / $380–$1,150 (Together). Closed frontier (~$3/$15): ~$800–$2,400 (derived). | **WRONG** as stated — true only versus frontier open or closed models. Re‑pitch on privacy, rate limits, quality parity (§6.4). |
| 100 | GRPO rollouts: `$1.00 / 1M` training tokens + `$1,800+/mo` GPUs stalling | Not re‑verified. | **UNVERIFIED** |

#### 7.4 Supply‑side economics (lines 105–117, 128)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 107 | US average `$0.15/kWh`; 4090 rig $0.075–$0.098/hr; Europe/CA/NE `$0.32–$0.42/kWh` | EIA STEO Oct 2026: US residential **17.3 ¢ (2025), 18.2 ¢ (2026), 18.7 ¢ (2027)** [46]. 500 W → **$0.091/hr**; 650 W (5090) → **$0.118/hr** (derived). Regional range not re‑verified. | **UPDATE** |
| 107 | Salad cash payout `$0.18–$0.25/hr` wiped out by power | Salad "Earning Rate Changes: September 12" (posted 4 Sep 2026) raised 5090/4090 rates and cut 15 classes [7]; the rates read from that post in September: **5090 $0.175–$0.30, 4090 $0.112–$0.198, 3090 $0.063–$0.102 per hour**. Customer‑facing 4090 price is $0.16–$0.33 [6]. | **OK** — the thesis is stronger than written: a US‑average 4090 host nets $0.02–$0.11/hr; a 3090 host nets ≈ $0. |
| 111 | Mac Studio/mini draw 40–85 W → `$0.006–$0.013/hr` | At 18.2 ¢: $0.007–$0.015/hr (derived). "128–512 GB UMA" — the 512 GB Mac Studio was **discontinued in March 2026**; 256 GB upgrade now $2,000 [39]. | **UPDATE** — say 128–256 GB for new machines. |
| 114 | Hosts get "$0.90–$1.15/hr in 1.25× credits" vs "$0.22/hr raw USD"; "4× perceived value"; "$0 upfront cash" | Retail 4090 burst is $0.45/hr (line 128); 1.25× retail = $0.56, 1.25× wholesale ($0.33) = $0.41. Credits are deferred COGS (a redemption liability), not $0 (§6.3). | **WRONG** (arithmetic) — define the credit rate once and model redemption. |
| 116 | Commercial fleets on `$0.08/kWh` commercial power | Plausible for some regions; not verified here. Vast says live prices run ~25% above host earnings and a 4‑GPU 5090 rig at 80% utilisation earns $700–$1,400/mo (≈ $0.30–$0.61 per GPU‑hour, derived) [5]. | **UNVERIFIED** power rate; fleet economics plausible. |
| 19, 128 | Community supply lifts gross margin from ~30% to >65% | Directionally right *if* credits are modelled as a liability with breakage; the DePIN record (Akash ~$4.2M ARR at >80% utilisation; sector $10B market cap on $72M revenue [30]) says consumer‑GPU supply is cheap but the *demand* side is the constraint. | **UPDATE** |

#### 7.5 Regulatory and tax (lines 242–255, 333–337, 611)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 243–247 | FinCEN prepaid‑access rule; closed‑loop exemption at ≤ $2,000/day under `31 CFR § 1010.100(ff)(4)(iii)(A)` | FinCEN FAQ: closed‑loop prepaid access (31 CFR 1010.100(kkk)) usable only for goods/services at a defined merchant; the prepaid‑program definition excludes closed‑loop arrangements limited to **$2,000 or less that can be associated with a prepaid access device or vehicle on any day** [50]. | **OK** — citation correct. Nuance: the cap is per device/vehicle per day, not "per account"; state money‑transmitter law is separate; confirm with counsel. |
| 252–255 | 1099‑NEC threshold $600 → **$2,000** for 2026 | OBBBA raised the 1099‑NEC/MISC threshold to $2,000 for payments made after 31 Dec 2025, inflation‑indexed from 2027 (2025 base, rounded to $100) [49]. | **OK** |
| 337 | Reg CF up to **$5,000,000** per 12 months | SEC: up to $5M in a 12‑month period, via a registered funding portal or broker‑dealer, with Form C disclosure [51]. | **OK** |
| 611 | Kickstarter + Stripe fee `8%` | Kickstarter 5% platform fee + 3–5% payment processing ≈ 8–10% all‑in [52]. | **OK** (use 8–10%). |

#### 7.6 Hardware BOM (lines 346, 606, 616, 618)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 616 | RK3588 32 GB board `$185` | Launch prices Oct 2023: Orange Pi 5 32 GB $159–$175, 5B $179, 5 Plus $197 [44] — *before* DRAM roughly doubled in 2026 [38]. | **UPDATE** — get live quotes; budget +30–60%. |
| 618 | Strix Halo 128 GB unit `$3,150` wholesale | Framework Desktop 128 GB retail rose from $1,999 to **$2,459** in Jan 2026 (+$460, memory cost) [40]; DGX Spark 128 GB is now $6,950, 64 GB $4,999 [41]. | **OK/conservative** today — but DRAM supply stays tight through 2027–2028 [38]; quote‑at‑ship or cap units. |
| 606 | Pre‑certified host avoids FCC Part 15 testing | Not re‑verified here. | **UNVERIFIED** (reasonable). |

#### 7.7 Grants (lines 358, 379–388, 624–625)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 385, 624 | NSF SBIR/STTR Phase I **$305,000**, Phase II **$1,250,000**, Fast‑Track **$1,555,000** | NSF relaunched SBIR/STTR on 26 May 2026 with $250M (NSF 26‑510; NSF 26‑511 $40M instrumentation pilot). Phase I up to $305k for **6–18 months**; Phase II up to $1.25M (~24 months); Fast‑Track $400k + $1.155M = $1.555M; new "Strategic Breakthrough" up to $30M by invitation. Project Pitch mandatory; deadlines **27 Jul 2026, 4 Nov 2026, 4 Mar 2027**; invitation valid for two deadlines [48]. | **OK** — fix "6–12 mo" → 6–18; next realistic full‑proposal date is **4 Mar 2027** (Pitch first). |
| 385, 625 | STTR partner gets "30%+ … 30% to 40%"; "approved within ~3 weeks" | STTR: research‑institution subaward, 60/40 small‑business/institution split with ≥30% at the institution; PI must be ≥51% employed by the small business; one proposal per org and per PI at a time [48]. Pitch turnaround not stated. | **OK** on split; **UNVERIFIED** on 3 weeks. Flag the 51% PI employment rule. |
| 387 | DOE SBIR Phase I `$200–250k`, Phase II `$1.1–1.6M` | Not verified in this pass. | **UNVERIFIED** |

#### 7.8 Marketing channels and credits (lines 484–488, 530, 750–752)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 484 | Reddit `$7–$15` CPM, `$0.45–$1.20` CPC | 2026 benchmarks: average CPM $3–$12; high‑value B2B $12 ($6–$20); SaaS $10; developer tools $9; tech/dev‑subreddit CPC ~$1.80; $5/day minimum [54]. | **OK** on CPM; CPC is ~2× higher than written. |
| 485, 530 | YouTube creators `$28–$55` CPM; `$700–$1,800` per integration | Advertiser CPM for tech/software is $10–$30 (platform ads) [55]; sponsorship integrations are priced flat per video, not by CPM. | **UPDATE** — treat as flat‑fee sponsorships; verify with 2–3 creator rate cards. |
| 486–487 | Dev newsletters `$50–$95` CPM; mega‑newsletters `$5k–$15k` | Techpresso (AI/tech) primary placement $4,000 for 600–1,500 unique clicks; dedicated send $24,000 [56]. No vertical CPM table found. | **UNVERIFIED** — plausible; price per *click* is the better unit. |
| 750 | OVHcloud up to €100,000 | Start €10k + 6 engineer hours; Scale up to €100k; <5 years old, <50 staff, <€10M revenue; no partner required [53]. | **OK** |
| 751 | Cloudflare for Startups bootstrapped `$10,000` | Tiers $5k / $25k / $100k / $250k; bootstrappers can claim the **$5k** tier directly; $100k+ needs an approved VC/accelerator [53]. | **WRONG** — $5k, not $10k. |
| 752 | Vultr Open Source Program `$500–$30,000/yr` | Vultr Startup Program needs a recent Series A–E round and six months of invoices [53]; no public open‑source program found. | **UNVERIFIED** |
| — | DigitalOcean (not cited) | Direct/bootstrapped applicants typically get $1k–$5k; up to $100k via partners / AI‑native tier [53]. | **ADD** |

#### 7.9 Macro narrative (lines 272–274, 327–329, 574–575, 635, 764)

| Line | Claim | Evidence | Verdict |
|---|---|---|---|
| 272, 327, 574 | Hyperscale AI datacenters "`470–490 TWh`" | IEA *Energy and AI*: data centres ~**415 TWh in 2024** (~1.5% of global electricity), ~**945 TWh by 2030** (base case); in the US, data centres are nearly half of electricity‑demand growth to 2030 [47]. | **UPDATE** — cite the IEA numbers. |
| 274, 329, 575, 635 | "30 million high‑end GPUs … 85% idle" | JPR Q2 2026: 12.5M desktop AIBs shipped in the quarter (four‑year high), 24.3M in H1, NVIDIA ~90% [35]. Steam Sept 2026: 24 GB 5.50%, 16 GB 27.21%, 12 GB+ 49.61% [33]; 32 GB ~1.28% (Jun) [32]. Steam peak concurrency 43.5M (11 Oct 2026) [34]. | **UPDATE** — "30M high‑end" only if ≥12 GB; "85% idle" is unsourced. Use §8.1's derived figures. |

---

### 8. Supply‑side estimates

#### 8.1 How many idle GPUs are really out there?

Derived from the Steam Hardware Survey shares [32][33] applied to Steam's 43.5M concurrent‑user record (11 Oct 2026) [34]. The survey share is used as a proxy for the concurrent population; real numbers will differ, but the order of magnitude is defensible and sourced, which the plan's "30 million … 85% idle" is not.

| VRAM tier | Steam share (Sep 2026) | Systems online at the 43.5M peak (derived) | Comment |
|---|---|---|---|
| 32 GB (5090 class) | ~1.3% (Jun 2026 breakdown) [32] | ~0.56M | 5090 at 0.40% of systems in Jul 2026 [32] |
| 24 GB (3090/4090/7900XTX class) | 5.50% [33] | ~2.4M | The practical "middle layer" host class |
| 16 GB | 27.21% [33] | ~11.8M | Fastest‑growing tier; most 16 GB cards are mainstream, not halo [32] |
| 12 GB+ (cumulative) | 49.61% [33] | ~21.6M | First time over half of Steam systems |

Shipments corroborate the scale: **12.5M desktop add‑in boards in Q2 2026** (a four‑year high, +6.6% YoY), 24.3M in H1, NVIDIA ~90% [35]. Existing marketplaces already aggregate a sliver of this: Salad 60k+ daily‑active GPUs [8], Vast 17k GPUs [4].

**So the honest sentence is:** "There are a few million 24–32 GB GPUs and over ten million 16 GB GPUs online on Steam at any given evening, almost all of them idle for AI." That is smaller than "30 million high‑end", and more than enough.

#### 8.2 Host economics at 2026 electricity prices

EIA residential average: 18.2 ¢/kWh (2026 forecast), 18.7 ¢ (2027) [46]. Power draw figures are the plan's own (line 106).

| Host hardware | Draw | Power cost/hr | Salad host rate/hr (Sep 2026) [7] | Net cash/hr at US average | 8 h/night × 30 nights |
|---|---|---|---|---|---|
| RTX 5090 rig | 650 W | $0.118 | $0.175–$0.30 | $0.06–$0.18 | $14–$44 |
| RTX 4090 rig | 500 W | $0.091 | $0.112–$0.198 | $0.02–$0.11 | $5–$26 |
| RTX 3090 rig | ~450 W | ~$0.082 | $0.063–$0.102 | ≈ −$0.02 to +$0.02 | ≈ $0 |
| Mac Studio (128–256 GB) | ~85 W | $0.015 | n/a (Salad is Windows/CUDA) | — | — |
| Mac mini M4 Pro (64 GB) | ~40 W | $0.007 | n/a | — | — |

Vast's own framing agrees: a four‑GPU 5090 rig at 80% utilisation earns $700–$1,400/month, i.e. roughly $0.30–$0.61 per GPU‑hour before power (derived), and live prices run ~25% above host earnings [5]. Salad's lifetime payout of $7M+ across 450k+ nodes [8] works out to roughly $15 per node ever (derived). **Cash payouts to residential hosts cannot be the retention mechanism; credits redeemable for something hosts actually want (burst, big‑model access, Pro features) can be — provided the credit rate is defined honestly (§6.3).**

---

## Part C — Readiness, risks and recommendations

### 9. Readiness and critical path

The roadmap marks Phases 1–5 complete and Phase 6 open [65]. Read against the code (§2), "complete" means the control plane, protocols, formats, verification, compression, planner, governance and gateways exist and are tested; the execution layer is at reference level. That is the right order to have built things in — the distributed‑systems work is the hard, novel part and it is done — but it sets the critical path for every external commitment in the plan.

| Critical‑path item | Why it gates the plan | Options | What it unlocks |
|---|---|---|---|
| **Model execution on real weights with hardware backends** | Every demo, every tok/s figure, the free tier's first‑run experience, the burst product | (a) pure‑Rust engines — `candle`, `burn`, `mistral.rs` — behind the existing kernel dispatcher; (b) FreeToken or colibri as a per‑node execution backend under Kadmium's planner (both Apache‑2.0; both already do the single‑box q\*/expert‑streaming job [57][58]); (c) ggml via FFI (fast, but not pure Rust) | Measured `η` and `α` per backend and model; modelled cells in §3.3 become measured; the r/LocalLLaMA post can carry numbers |
| **Transport and WAN** | The docs say QUIC and STUN/DERP; the code speaks TCP and plans NAT traversal without sending STUN | `quinn` for QUIC; a STUN probe; `relay.kadmium.ai` DERP fallback | Friend/lab peers and the Supply Grid beyond one LAN |
| **OS yield hooks** | The <200 ms polite yield is the gaming‑PC promise; today the sampler sees GPU utilisation only | DXGI/NVML on Windows, IOKit/Metal on macOS, DRM/sysfs on Linux | The "launch a game, watch layers shift" demo |
| **`hf://` pulls and `.ktb` on disk** | Users need to load a real model in one command | HTTP client (`reqwest`/`hyper`) for byte‑range pulls; `mmap`/`io_uring` read paths; GGUF metadata | `kadmium pull hf://…` on a real checkpoint |
| **WASI runtime** | RL verifiers and Dream‑RSI worlds are a `gh://` ecosystem play | `wasmtime` with fuel metering | Real reward sandboxes on SBCs |
| **Measurement programme** | §3 is modelled; the two parameters that matter most are unmeasured | `kadmium bench` on the kitchen‑sink mesh and one office mesh: achieved‑bandwidth fraction per backend, expert hit‑rate exponent on real coding/chat traffic, hop latency, yield latency | Credible public numbers; pricing calibrated to real throughput |

**Sequencing that fits the evidence.** Single‑box execution first on models that decode interactively on one machine (Qwen3.6‑35B‑A3B, gpt‑oss‑120b) — this is also where the free tier meets its first users and where FreeToken and colibri set the bar [57][58]. Then the two‑node LAN pipeline (PC + Mac) on a dense 70B and Qwen3‑235B, which is the first thing no competitor's single box can do. Then expert‑parallel placement on the office mesh for DeepSeek‑V3‑class models, then burst. Publish each step's measurements as they land. The plan's week‑three public launch should be re‑sequenced behind the first two steps; the NSF Project Pitch can go in now, with the full proposal aimed at the 4 March 2027 deadline [48].

### 10. Risk register

| Risk | Evidence | Mitigation |
|---|---|---|
| Fixed burst prices lose money or look expensive within a quarter | Wholesale index moved 20–120% against the plan's assumptions in six months [1][12][42] | Index‑plus‑spread pricing refreshed daily; per‑token metering; community supply for 24–32 GB classes (§6.2) |
| Placeholder figures get read as claims | r/LocalLLaMA checks numbers [36]; the plan and site carried "671B at 34 tok/s on 112 GB" | Carry §3.3's modelled table labelled as such; replace with measurements; never publish an unmeasured tok/s |
| Execution‑layer timing slips the launch plan | §2, §9 | Re‑sequence launch behind single‑box and two‑node measured demos; consider FreeToken/colibri as backends |
| Memory prices raise hardware‑tier COGS before fulfilment | DRAM +90–110% H1 2026, tight through 2028 [38]; Framework +$460 [40]; DGX Spark +74% [41] | Quote‑at‑ship, capped quantities, 15% buffer; re‑quote RK3588 boards (§6.6) |
| Residential supply earns little in cash terms | Salad $0.11–0.20/hour host rates [7]; $7M lifetime payouts across 450k nodes [8]; DePIN $72M revenue on $10B cap [30] | Credits‑first; meter per token/GB; studio and campus fleets for SLA‑grade supply (§6.3) |
| Adjacent players extend into the gap | FreeToken (14.4k stars) adding multi‑node; EXO adding NVIDIA; Ollama adding LAN pooling; Salad adding sharding [14][19][57] | Ship the mesh features they lack first; publish head‑to‑heads on mixed hardware |
| Regulatory exposure of the credit system | FinCEN closed‑loop exemption is per device/day and federal; state money‑transmitter law is separate [50] | Counsel review before launch; no P2P credit transfer; no cash‑out of promotional credits |
| Conversion assumptions in the 90‑day model | 8.7% paid conversion in 90 days vs 1–5% norms (§6.5) | Budget on the base case; treat the plan as stretch |
| Privacy language over‑reaches | "Blind" middle layers are INT8 activations; activation inversion is an active research area | Say "zero plaintext egress" and "no prompt or output ever leaves your hardware"; avoid "cryptographic privacy" |
| Inconsistent compression claims | 10–85× in the spec vs 800–3,000× in the README and architecture doc [65] | Quote the spec figure externally until measured |
| Phase 6 described as shipped | Roadmap marks it open [65] | Label swarming/striping as roadmap in external material (§3.6) |

### 11. Recommendations, ranked by impact

1. **Make real model execution the next milestone and sequence the launch behind it (§9).** Single box first (Qwen3.6‑35B‑A3B, gpt‑oss‑120b), then the two‑node LAN pipeline, then expert‑parallel on an office mesh, then burst. Decide early between pure‑Rust engines and FreeToken/colibri as per‑node backends under Kadmium's planner; both routes keep the planner, protocol and governance that are Kadmium's actual differentiation [57][58].
2. **Publish the capability envelope now, labelled modelled, and replace cells with `kadmium bench` measurements as they land (§3.3).** Lead with what the kitchen‑sink desk does at interactive speed and with the office‑mesh numbers that no single machine can match; show the 671B–2.8T class honestly as "runs on this desk at 1–2.5 tok/s, 4–16 per user on an office mesh". Retire every placeholder figure; measure `η` and `α` first.
3. **Price burst as index × spread (or per token), never fixed $/hour (§6.2).** Refresh daily from the same provider APIs the bid engine will call; publish a floor; show an estimated $/M‑tokens. Route 24–32 GB classes to community supply by default and datacenter burst to 80 GB+ classes.
4. **Run the Supply Grid as a batch‑token market (§6.3).** Keep the code's 1.25× credit rule but index the $0.38 base to the wholesale rate; meter per token processed and GB served; expire credits at 12 months; report credit liability; sell the aggregate as batch capacity at a discount to serverless.
5. **Lead the coding‑team pitch with quality parity, privacy and no rate limits, not price (§6.4).** Kadmium is cheaper than the frontier‑open and closed tiers, not cheaper than DeepSeek‑V3.2 on OpenRouter; developers rank quality 75% vs cost 38% [37], and 38% of organisations live under usage caps.
6. **Publish head‑to‑heads on mixed hardware (Windows 4090 + Mac + Linux box) against llama.cpp RPC and EXO.** It is the one configuration neither serves [19][20][22] and the one most r/LocalLLaMA readers own.
7. **Keep the swarm as the product's core and book revenue where the evidence says it is (§5.0, §6.3).** Credits rather than cash for residential hosts; studio and campus fleets for SLA‑grade supply; the paid layers as the margin engine; Phase 6 swarming and striping scoped to staging and throughput (§3.6).
8. **Plan the NSF timeline around 4 March 2027; submit the Project Pitch now [48].** The 4 November 2026 deadline is two weeks away and the invitation is good for two deadlines; check the ≥ 51% PI‑employment rule before naming the PI.
9. **Hedge the hardware tiers (§6.6).** Quote‑at‑ship or capped quantities on the Strix Halo tier; re‑quote RK3588 32 GB boards and NVMe; use a 9% fee assumption [38][40][41][52].
10. **Align the documents.** DeMo compression 10–85× (spec) vs 800–3,000× (README/architecture) — quote the spec figure; say TCP now, QUIC on the roadmap; label Phase 6 as roadmap; scope "blind middle layers" as zero plaintext egress [65].
11. **Fix the factual errors before anyone external reads the plan (§7).** LM Studio "Bionic" (line 47), Petals as live (77/471/588), Cloudflare $10k → $5k (751), 470–490 TWh → IEA 415/945 TWh (272/327/574/635/764), "30 million … 85% idle" → §8.1, Vultr open‑source programme (752), the H100 and L40S provider rates (227–229).
12. **Decide the relationship with FreeToken and colibri explicitly in the plan.** Both are Apache‑2.0, both ship coding‑agent APIs, and together they own the single‑machine giant‑MoE story [57][58]. Kadmium's defensible layer is everything above one machine — mesh pooling, polite yield, live shifting, privacy‑tiered sharding, verification, credits, burst and training. Saying so turns "why not just use FreeToken?" into the pitch, and makes option (b) in §9 a strategy rather than a shortcut.

### 12. Suggested edits to the plan

| Line(s) | Current | Replace with |
|---|---|---|
| 4, 18, 128 | "28–35% gross margin" on wholesale spot `$0.29–$0.82/hr` | "retail = daily wholesale index × 1.30 (floor published); datacenter burst for 80 GB+ classes, community supply for 24–32 GB classes" |
| 19, 114 | "`$0.90–$1.15/hr` in 1.25× credits … `$0` in upfront cash" | "`1.0–1.25× the wholesale index` in credits (≈ $0.33–$0.41/hr for a 4090 in Oct 2026); credits expire in 12 months and are carried as a redemption liability" |
| 35–40 | Price table | Date‑stamp the table; replace with §6.2 figures; add 5090 ($0.35 / $0.72 / Salad $0.25–$0.50) and B200 ($3.75 / $7.99) |
| 44–48 | "`LM Studio Bionic` … paid team/organization cloud credit pools with ZDR" | "`LM Studio` (free for work since Jul 2025; paid Teams/Enterprise for SSO and model/MCP gating)" |
| 46 | Ollama tiers | Keep; add "per‑token billing with included monthly usage since 31 Aug 2026; no rollover; ZDR" |
| 52 | "$1.5B+" | "~US$1.5B valuation (Series C, Apr 2025); 10,000+ customers" |
| 58 | "Tens of thousands of gamers … 190 countries" | "60,000+ daily‑active GPUs, 100,000+ monthly‑active hosts, 191 countries; $7M+ paid to hosts lifetime" |
| 66 | "`$3.95/hr` to `$6.50/hr` per H100" | "`$5.49–$7.00/hr` per dedicated H100 (Together, Baseten, Fireworks); Modal serverless `$3.95`" |
| 77, 471, 588 | Petals as a live competitor | "Petals (dormant since 2024)" |
| 92 | "`$0.20–$0.60` in / `$0.60–$1.80` out" | "DeepSeek‑V3.2 `$0.28 / $0.42` (cache read `$0.028`); frontier‑open tier (DeepSeek V4‑Pro, Kimi K2.6, GLM‑5.1) `$0.95–$2.10 / $3.48–$4.40`" |
| 98 | "`$900 – $2,400 / mo` … Kadmium wins by 70–85%" | §6.4 table; "Kadmium wins on privacy, rate limits and quality parity; price parity against V3.2‑class APIs" |
| 107 | "`$0.15/kWh`… `$0.075–$0.098/hr`" | "`$0.182/kWh` (EIA 2026) … `$0.09–$0.12/hr`"; Salad host rates "4090 `$0.11–$0.20/hr`, 3090 `$0.06–$0.10/hr` (Sep 2026)" |
| 111 | "128–512 GB UMA" | "128–256 GB UMA (512 GB Mac Studio discontinued Mar 2026)" |
| 154, 278, 519, 565 | Placeholder demo figures ("`DeepSeek‑V3‑671B`" on the kitchen‑sink mesh, "`112 GB` pool … `34 tok/s`") | The §3.3 envelope for the demo mesh, labelled modelled: `gpt‑oss‑120b` ~28 tok/s, dense 70B ~7, `Qwen3‑235B` 3–5, `DeepSeek‑V3` 1–2 on the kitchen‑sink mesh; `DeepSeek‑V3` 9–11 on a five‑machine office mesh; replace cells with `kadmium bench` measurements as they land |
| 227–229 | RunPod / Modal / Vast rates | RunPod 4090 `$0.34–$0.69`, H100 `$2.69–$2.99`; Modal L40S `$1.95`, A100 `$2.50`, H100 `$3.95`; Vast H100 verified `$1.50–$1.87`, 4090 `$0.34–$0.50` |
| 230–231 | Prime Intellect / SF Compute rates | Mark "quote on request" until verified |
| 247 | "capped at `$2,000/day` per account" | "no more than `$2,000` associated with a prepaid access device/vehicle on any day (31 CFR 1010.100(ff)(4)(iii)(A)); confirm state MTL exposure with counsel" |
| 272, 327, 574, 635, 764 | "`470–490 TWh`" | "~415 TWh in 2024, ~945 TWh by 2030 (IEA base case)" |
| 274, 329, 575, 635 | "30 million high‑end GPUs … 85% idle" | "a few million 24–32 GB GPUs and 10M+ 16 GB GPUs online on Steam alone at peak (Steam Hardware Survey Sep 2026 × 43.5M concurrent record)" |
| 405–471 (§15) | "rivals a `$2.50/hr` datacenter GPU"; "`10.6 s` → `1.3 s`" | Scope to throughput workloads (independent sequences, rollouts, batch) and to staging; add the causal‑attention K/V‑exchange constraint for single‑prompt striping; keep parity for data streams and linear stages; label Phase 6 as roadmap (§3.6) |
| 385 | "Phase I … (6–12 mo)" | "(6–18 mo)"; add deadlines 4 Nov 2026 / 4 Mar 2027 and the ≥51% PI employment rule |
| 484 | "`$0.45 – $1.20` CPC" | "`$0.50 – $3.00` CPC (tech/dev subreddits ≈ `$1.80`)" |
| 485, 530 | "`$28 – $55` CPM" | "flat‑fee sponsorships; advertiser CPM for tech/software `$10–$30`" |
| 611–618 | "8%" fee; RK3588 `$185`; Strix Halo tier | "9%"; RK3588 `$240–$300` (re‑quote); Strix Halo "built to order, capped" |
| 751 | "`$10,000`" | "`$5,000` bootstrapped tier (`$25k / $100k / $250k` with partners)" |
| 752 | "`Vultr Open Source Program` `$500 – $30,000/yr`" | Remove or verify; Vultr's startup program requires a Series A–E round |

---

### 13. Still unverified (worth a second pass before external use)

- SF Compute block pricing (line 231) and Prime Intellect's rate card (line 230).
- DOE SBIR/STTR Phase I/II amounts (line 387) and the NSF Project Pitch turnaround time (line 385).
- GRPO/RL training cost figures (line 100).
- Regional electricity rates quoted for Europe/California/Northeast (line 107) and commercial `$0.08/kWh` (line 116).
- Newsletter CPMs (lines 486–487) — price per click is available [56]; CPM by vertical is not.
- FCC Part 15 treatment of a pre‑certified host with a laser‑etched enclosure (line 603–607).
- Whether the "Vultr Open Source Program" exists in 2026 (line 752).
- Salad's exact September‑2026 host rates: figures are as read from the 4 Sep 2026 "Earning Rate Changes" post [7]; the index page fetched for this review shows the post but not the table.
- Kadmium's own measured tok/s per configuration (the repo documents the mechanisms and simulator output, not benchmark runs on real meshes). §3's envelope stands in until `kadmium bench` numbers exist; the two parameters most worth measuring first are the achieved‑bandwidth fraction `η` per backend and the expert hit‑rate exponent `α` for real coding/chat traffic.

---

### 14. Sources

Checked 20 October 2026 unless a date is given. Third‑party trackers and blogs are marked (3P); vendor pages are marked (V); primary/official sources are marked (P).

1. (3P) getdeploying — Cloud GPU Index, 85 providers / 3,938 listings, updated 20 Oct 2026. https://getdeploying.com/gpus
2. (V/3P) Spheron — GPU cloud spot vs on‑demand vs hyperscaler, May 2026; Spheron rates live 20 Oct 2026. https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/
3. (3P) Spheron — RunPod vs Vast.ai 2026 (Jul 28, updated Sep 30, 2026). https://www.spheron.network/blog/runpod-vs-vastai-2026/
4. (3P) Fast.io — Vast.ai review 2026 (reviewed Jun 25, 2026). https://www.fast.io/resources/vast-ai-review-2026.md
5. (V) Vast.ai — How much money can you earn renting out your GPU (May 18, 2026). https://vast.ai/article/how-much-money-can-you-earn-renting-out-your-gpu-on-vast-ai
6. (3P) Spheron — SaladCloud pricing 2026 (Sep 27, updated Oct 19, 2026; Salad's Sep 12 price list). https://www.spheron.network/blog/saladcloud-pricing-2026-gpu-cost-vs-spheron/
7. (V) SaladCloud blog — "Earning Rate Changes: September 12" (Sep 4, 2026), "Upcoming SaladCloud Price Changes: September 12, 2026", "How SaladCloud Built a Global Distributed Compute Network" (Aug 20, 2026). https://blog.salad.com/?p=1534
8. (3P) Jared Watkins — Salad Technologies research note (Aug 15, 2026; self‑reported scale figures). https://www.jaredwatkins.com/research/compute-marketplaces/_to_delete/salad/
9. (3P) Spheron — Modal GPU pricing 2026 (Jul 14, updated Oct 19, 2026). https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/
10. (3P) Morph — Together AI vs Baseten pricing (Jun 9, 2026). https://www.morphllm.com/comparisons/together-vs-baseten
11. (3P) Morph — Fireworks vs Together AI pricing (Jun 9, 2026). https://www.morphllm.com/comparisons/fireworks-vs-together
12. (3P) UsagePricing — Lightning AI GPU rate‑card reprice (Sep 21, 2026). https://www.usagepricing.com/blueprint/activity/lightning-ai-2026-09-21-gpu-rate-card-reprice
13. (3P) PricePerToken — DeepSeek V3.2 pricing by provider (updated Oct 20, 2026). https://pricepertoken.com/pricing-page/model/deepseek-deepseek-v3.2
14. (V) Ollama — "Ollama's transparent pricing" (Aug 31, 2026). https://ollama.com/blog/transparent-pricing
15. (3P) Omid Saffari — Ollama pricing 2026 (verified Aug 23, 2026; pre‑change tiers). https://omidsaffari.com/blog/ollama-pricing
16. (V) LM Studio — "LM Studio is free for use at work" (Jul 8, 2025). https://www.lmstudio.ai/blog/free-for-work
17. (3P) NCFA Canada — Tailscale raises CA$230M Series C (Apr 9, 2025). https://ncfacanada.org/tailscale-raises-230m-to-power-identity-first-networking/
18. (V) EXO Labs — NVIDIA DGX Spark + Apple Mac Studio with EXO 1.0. https://blog.exolabs.net/nvidia-dgx-spark/
19. (3P) DEV Community — EXO framework in 2026: pooling RTX 3090s vs DGX Spark (Jun 26, 2026). https://dev.to/jovan_chan_9500711396d4e6/exo-framework-in-2026-can-you-pool-rtx-3090s-to-beat-a-dgx-spark-the-honest-distributed-inference-5ccm
20. (P) GPUStack — README (main branch). https://cdn.jsdelivr.net/gh/gpustack/gpustack@main/README.md
21. (3P) SourcePulse — GPUStack project page. https://www.sourcepulse.org/projects/1842224
22. (V) AMD — How to run a one‑trillion‑parameter LLM locally: Ryzen AI Max+ cluster guide (Feb 25, 2026). https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html
23. (3P) Fungies — Multi‑node local LLM inference guide (Jul 1, 2026). https://fungies.io/multi-node-local-llm-inference-guide-2026
24. (3P) kjaiswal — llama.cpp distributed benchmarks, Mac Studio + DGX Spark over 10 GbE. https://github.com/kjaiswal/llama-cpp-distributed-benchmarks
25. (3P) ecosyste.ms — bigscience‑workshop/petals repository record. https://summary.ecosyste.ms/api/v1/projects/124184
26. (3P) KuCoin / Foresight News — Prime Intellect $130M Series A at $1B (Jul 2026). https://www.kucoin.com/news/flash/prime-intellect-completes-130m-a-round-at-1b-valuation-backed-by-nvidia-intel-and-dell
27. (3P) Sacra — Prime Intellect research. https://sacra.com/research/prime-intellect
28. (V) Nous Research — "The Next Phase of Psyche" (Sep 29, 2025). https://nousresearch.com/the-next-phase-of-psyche
29. (3P) Internet Pros — Decentralized AI training in 2026 (May 27, 2026; promotional). https://internet-pros.com/blog/decentralized-ai-training-prime-intellect-nous-2026/
30. (3P) BlockEden — DePIN compute revenue pivot: Akash, io.net, Aethir (Mar 12, 2026). https://blockeden.xyz/blog/2026/03/12/depin-compute-revenue-pivot-akash-ionet-aethir/
31. (3P) Roots Analysis — GPU‑as‑a‑Service market report. https://rootsanalysis.com/gpu-as-a-service-market
32. (3P) Tech‑Insider — Steam Hardware Survey VRAM breakdown, Jun–Aug 2026. https://tech-insider.org/steam-hardware-survey-july-2026/
33. (3P) Digital Citizen — Steam Hardware Survey September 2026. https://www.digitalcitizen.life/nvidia-rtx-5070-most-popular-gpu-steam-hardware-survey-september-2026/
34. (3P) Dot Esports — Steam all‑time record 43.5M concurrent users (Oct 11, 2026). https://dotesports.com/general/news/steam-all-time-record-43-5-million-concurrent-users
35. (3P) TweakTown — JPR: desktop GPU shipments hit four‑year high, Q2 2026. https://www.tweaktown.com/news/113493/desktop-gpu-shipments-hit-four-year-high-despite-rising-prices-per-new-research/index.html
36. (3P) GummySearch — r/LocalLLaMA stats (updated Sep 30, 2026). https://gummysearch.com/r/LocalLLaMA/
37. (P) Stack Overflow — 2026 Developer Survey results (Oct 6, 2026). https://stackoverflow.blog/2026/10/06/the-results-of-the-2026-developer-survey-are-here/
38. (3P) Wccftech — RAM shortage 2026 roundup (updated Apr 23, 2026). https://wccftech.com/roundup/memory-crisis/
39. (3P) Notebookcheck — Apple raises M3 Ultra Mac Studio RAM pricing, retires 512 GB (Mar 6, 2026). https://www.notebookcheck.net/Apple-increases-RAM-upgrade-pricing-for-M3-Ultra-Mac-Studio-retires-512-GB-option.1244246.0.html
40. (3P) Notebookcheck — Framework Desktop now costs up to $460 more (Jan 13, 2026). https://www.notebookcheck.net/Framework-Desktop-now-cost-up-to-460-more-due-to-RAM-shortage.1203235.0.html
41. (3P) AI Weekly — NVIDIA launches 64 GB DGX Spark at $4,999; 128 GB to $6,950 (Oct 3, 2026). https://aiweekly.co/alerts/nvidia-launches-64gb-dgx-spark-at-4999-as-ram-shortage-pushes-128gb-model-to
42. (3P) GPUPoet — October 2026 GPU market report. https://gpupoet.com/gpu/market-report/gpu-market-report-october-2026
43. (3P) RunAIHome — Used RTX 4090 vs new RTX 5090 for local AI (Sep 17, 2026). https://runaihome.com/blog/used-rtx-4090-vs-new-rtx-5090-2026/
44. (3P) Notebookcheck — Orange Pi 5 / 5B / 5 Plus 32 GB launch prices (Oct 17, 2023). https://www.notebookcheck.net/Orange-Pi-5-Orange-Pi-5B-and-Orange-Pi-5-Plus-refreshed-with-new-32-GB-of-RAM-options.759877.0.html
45. (3P) Keenable — Permissively licensed open‑weight models released after January 2026. https://select.keenable.ai/r/permissively-licensed-open-weight-models-released-after-january-2026/X06y2UYTUiTXtYZ0uS8rRA
46. (P) US EIA — Short‑Term Energy Outlook, electricity (Oct 6, 2026). https://www.eia.gov/outlooks/steo/report/elec_coal_renew.php?sfnsn=mo
47. (P) IEA — Energy and AI, executive summary. https://www.iea.org/reports/energy-and-ai/executive-summary
48. (3P) Granted AI — NSF relaunches SBIR/STTR with $250M (May 31, 2026; summarises NSF 26‑510/26‑511). https://grantedai.com/news/nsf-sbir-sttr-250m-2026-05-31
49. (3P) NATP — New 1099 rules under OBBBA. https://www.natptax.com/news-insights/blog/new-1099-rules-under-obbba-what-tax-pros-need-to-know-for-2026-and-beyond/
50. (P) FinCEN — Frequently asked questions regarding prepaid access. https://www.fincen.gov/resources/statutes-regulations/guidance/frequently-asked-questions-regarding-prepaid-access
51. (P) US SEC — Regulation Crowdfunding. https://www.sec.gov/smallbusiness/exemptofferings/regcrowdfunding
52. (3P) PledgeBox — Kickstarter fee calculator 2026 (May 10, 2026). https://www.pledgebox.com/post/kickstarter-fee-calculator
53. (3P) Better Stack — Best startup cloud credit programs 2026 (Sep 14, 2026). https://betterstack.com/community/comparisons/best-startup-cloud-credit-programs/
54. (3P) Benly — Reddit Ads cost benchmarks (updated Mar 24, 2026). https://benly.ai/learn/reddit-ads/reddit-ads-cost-benchmarks
55. (3P) upGrowth — YouTube CPM rates 2026 (updated Sep 16, 2026). https://upgrowth.in/youtube-cpm-rates-in-2026
56. (3P) Dupple — Newsletter sponsorship rate card 2026. https://dupple.com/learn/cpm-benchmarks-b2b-newsletters
57. (P) FlashML‑org/FreeToken — GitHub README (edge‑native MoE serving; q\* policy; FTW format; 14.4k stars). https://github.com/FlashML-org/FreeToken
58. (P) JustVugg/colibri — GitHub README (model RAM/disk tables and self‑reported decode speeds; 42.8k stars). https://github.com/JustVugg/colibri
59. (P) JustVugg/lumabri — GitHub README (P2P expert execution; measured tok/s local vs LAN vs internet). https://github.com/JustVugg/lumabri
60. (3P) MindStudio — FreeToken running DeepSeek V4‑Flash on a single RTX 3090 (Aug 25, 2026). https://www.mindstudio.ai/blog/freetoken-deepseek-v4-flash-single-3090
61. (3P) Wavect — colibri: GLM‑5.2 on consumer hardware (Jul 14, 2026; reviewed Sep 24, 2026 against colibri v1.12.0). https://wavect.io/blog/colibri-glm-5-2-consumer-hardware/
62. (V) Kalavai — "Setting a new world record: the world's largest distributed deployment" (Oct 22, 2024). https://kalavainet.substack.com/p/setting-a-new-world-record-the-worlds
63. (3P) SourcePulse — AI‑Horde (Haidra‑Org) project page. https://www.sourcepulse.org/projects/1845393
64. (3P) Wikipedia — Volunteer computing (Folding@home 2.43 exaFLOPS, April 2020). https://en.wikipedia.org/wiki/Volunteer_computing
65. (P) Kadmium repository — `readme.md` (heritage table, core capabilities, dataflow modes), `docs/ARCHITECTURE.md` §2.2 five‑tier memory hierarchy, §3.2 q\* policy, §3.3 dataflow modes, §6 `.ktb`; `docs/SPECSHEET_AND_ROADMAP.md` §2 hardware matrix, §3 `Q4KBlockTensor` / `ColibriNvmeDoubleBuffer`, §4 Supply Grid.

---

### Appendix A — Capability envelope script

The model behind §3, §3.6 and §3.7. Pure Python, no dependencies; edit `MODELS` and `MESHES` to your hardware and run `python3 -I envelope.py`. Bandwidths are vendor peaks, `η` is achieved fraction, usable bytes exclude KV/activations/OS. It is a bound‑style estimate (no kernel launch overhead, no KV reads, no scheduling jitter): expect real systems at 0.7–1.0× of the *seq* figure and 0.5–1.0× of the *par* figure until measured.

```python
"""First-principles capability envelope for Kadmium-style tiered MoE decode/prefill.

Single-stream decode (B=1):
  trunk (attention/shared/dense) is read layer-by-layer -> sequential: T_trunk = sum_i bytes_i/(BW_i*eta_i)
  routed experts: cached fraction c = cached_bytes/routed_total; hit = c**alpha (alpha=1 uniform, 0.25 skewed, fit to colibri)
     mode "seq" (pipeline only, llama.cpp-RPC-like): T_exp = sum_i hit_bytes_i/(BW_i*eta_i); misses from ONE NVMe
     mode "par" (expert-parallel, weight-stationary peers): tiers work concurrently -> T_exp = max_i hit_bytes_i/(BW_i*eta_i)
                                                           + one combine round-trip per MoE layer; misses from ALL NVMe
  T_tok = T_trunk + T_exp + T_miss + pipeline_hops*t_hop
Prefill (S tokens, token-stationary streaming): T = max(stream_bytes/BW_stream, 2*A*S/(FLOPS*MFU)) + hop transfers
Run: python3 -I envelope.py
"""
GB = 1e9

MODELS = {  # trunk_bytes, routed_bytes_per_token, routed_total_bytes, active_params, d_model, moe_layers
    "DeepSeek-V3 671B (Q4K)":      dict(trunk=9.1*GB, rpt=11.5*GB, rtot=368*GB, A=37e9,  d=7168, Lmoe=58),
    "GLM-5.2 744B (colibri pkg)":  dict(trunk=9.9*GB, rpt=11.0*GB, rtot=415*GB, A=40e9,  d=6144, Lmoe=70),
    "Qwen3-235B-A22B (Q4K)":       dict(trunk=4.5*GB, rpt=8.0*GB,  rtot=128*GB, A=22e9,  d=4096, Lmoe=94),
    "gpt-oss-120b (MXFP4)":        dict(trunk=4.0*GB, rpt=1.9*GB,  rtot=61*GB,  A=5.1e9, d=2880, Lmoe=36),
    "Llama-3.3-70B dense (Q4K)":   dict(trunk=40*GB,  rpt=0.0,     rtot=0.0,    A=70e9,  d=8192, Lmoe=0),
}

def node(name, tiers, nvme=0.0, flops=0.0):
    # tiers: list of (label, usable_bytes, BW, eta)
    return dict(name=name, tiers=tiers, nvme=nvme, flops=flops)

MESHES = {
 "M1 kitchen-sink: 4090 PC (24+64 GB) + MBP M3 Max 36 GB + NAS, 2.5GbE": dict(nodes=[
    node("PC", [("VRAM", 20*GB, 1000*GB, 0.6), ("DDR5", 54*GB, 60*GB, 0.7)], nvme=7*GB, flops=80e12),
    node("MBP", [("UMA", 30*GB, 300*GB, 0.6)], nvme=5*GB, flops=15e12)], t_hop=0.7e-3, link=0.3*GB),
 "M3 single PC: 4090 + 64 GB DDR5 + Gen4 NVMe": dict(nodes=[
    node("PC", [("VRAM", 20*GB, 1000*GB, 0.6), ("DDR5", 54*GB, 60*GB, 0.7)], nvme=7*GB, flops=80e12)], t_hop=0, link=1),
 "M3b single PC: 4090 + 192 GB DDR5 + Gen4 NVMe": dict(nodes=[
    node("PC", [("VRAM", 20*GB, 1000*GB, 0.6), ("DDR5", 180*GB, 60*GB, 0.7)], nvme=7*GB, flops=80e12)], t_hop=0, link=1),
 "M4 Strix Halo x4: 4 x 128 GB LPDDR5X, 5GbE": dict(nodes=[
    node(f"SH{i}", [("UMA", 118*GB, 256*GB, 0.5)], nvme=7*GB, flops=45e12) for i in range(4)], t_hop=0.6e-3, link=0.6*GB),
 "M2 office: 2 x 4090 PCs (24+64 GB), 2 x MBP M4 Max 48 GB, Mac Studio M4 Max 128 GB, 10GbE": dict(nodes=[
    node("PC1", [("VRAM", 20*GB, 1000*GB, 0.6), ("DDR5", 54*GB, 60*GB, 0.7)], nvme=7*GB, flops=80e12),
    node("PC2", [("VRAM", 20*GB, 1000*GB, 0.6), ("DDR5", 54*GB, 60*GB, 0.7)], nvme=7*GB, flops=80e12),
    node("MBP1", [("UMA", 42*GB, 410*GB, 0.6)], nvme=5*GB, flops=20e12),
    node("MBP2", [("UMA", 42*GB, 410*GB, 0.6)], nvme=5*GB, flops=20e12),
    node("Studio", [("UMA", 115*GB, 546*GB, 0.6)], nvme=5*GB, flops=30e12)], t_hop=0.4e-3, link=1.2*GB),
}

def all_tiers(M):
    out = []
    for n in M["nodes"]:
        for (lab, cap, bw, eta) in n["tiers"]:
            out.append(dict(node=n["name"], lab=lab, cap=cap, bw=bw, eta=eta))
    return out

def decode(model, mesh, alpha, mode):
    m, M = MODELS[model], MESHES[mesh]
    tiers = sorted(all_tiers(M), key=lambda t: -t["bw"]*t["eta"])
    # 1) trunk: fill fastest tiers first, sequential reads
    left, t_trunk = m["trunk"], 0.0
    for t in tiers:
        take = min(left, t["cap"]); t["cap"] -= take; left -= take
        t_trunk += take/(t["bw"]*t["eta"])
        if left <= 0: break
    nvme_total = sum(n["nvme"] for n in M["nodes"])
    if left > 0: t_trunk += left/(nvme_total*0.9)   # trunk overflow streams
    hops = (len(M["nodes"])-1)*M["t_hop"]
    if m["rtot"] == 0:
        T = t_trunk + hops
        return 1/T, 1.0
    # 2) routed experts
    cached = sum(t["cap"] for t in tiers)
    c = min(1.0, cached/m["rtot"])
    hit = 1.0 if c >= 1 else c**alpha
    hit_bytes = hit*m["rpt"]; miss_bytes = (1-hit)*m["rpt"]
    if mode == "seq":
        t_exp = sum(hit_bytes*(t["cap"]/cached)/(t["bw"]*t["eta"]) for t in tiers if cached > 0)
        t_miss = miss_bytes/(7*GB*0.9)
        extra = 0.0
    else:  # expert-parallel: tiers/nodes work concurrently; capacity-proportional placement -> slowest tier bounds
        t_exp = max((hit_bytes*(t["cap"]/cached)/(t["bw"]*t["eta"]) for t in tiers if cached > 0), default=0.0)
        t_miss = miss_bytes/(nvme_total*0.9)
        extra = m["Lmoe"]*M["t_hop"] if len(M["nodes"]) > 1 else 0.0  # one combine round-trip per MoE layer
    T = t_trunk + t_exp + t_miss + hops + extra
    return 1/T, c

def prefill(model, mesh, S=4096, mfu=0.5):
    m, M = MODELS[model], MESHES[mesh]
    fast = sum(t["cap"] for t in all_tiers(M))
    stream = max(0.0, m["trunk"] + m["rtot"] - fast)
    nvme_total = sum(n["nvme"] for n in M["nodes"])
    t_stream = stream/(nvme_total*0.9)
    flops = sum(n["flops"] for n in M["nodes"])
    t_compute = 2*m["A"]*S/(flops*mfu)
    t_hops = (len(M["nodes"])-1)*(S*m["d"]*1)/M["link"]
    return max(t_stream, t_compute)+t_hops, t_stream, t_compute, t_hops

if __name__ == "__main__":
    print("SINGLE-STREAM DECODE tok/s  [seq = pipeline only | par = expert-parallel ideal]  (uniform alpha=1 / skewed alpha=0.25)")
    for mesh in MESHES:
        print("\n==", mesh)
        for model in MODELS:
            su,c = decode(model, mesh, 1.0, "seq"); ss,_ = decode(model, mesh, 0.25, "seq")
            pu,_ = decode(model, mesh, 1.0, "par"); ps,_ = decode(model, mesh, 0.25, "par")
            print(f"  {model:30s} cached={c:4.0%}  seq {su:5.1f}/{ss:5.1f}   par {pu:5.1f}/{ps:5.1f}")
    print("\nPREFILL TTFT, S=4096 tokens (s): total | stream | compute | hop transfers")
    for mesh in MESHES:
        print("\n==", mesh)
        for model in MODELS:
            T,ts,tc,th = prefill(model, mesh)
            print(f"  {model:30s} {T:6.1f} | {ts:5.1f} | {tc:4.1f} | {th:4.2f}")
    print("\nCrossover S* = eta*FLOPS*b/(2*BW): tile compute hides streaming beyond this batch (4090, 80 TFLOPs eff, b=0.5625)")
    for name, bw in [("DDR5 dual-channel 60 GB/s",60*GB),("Gen4 NVMe 7 GB/s",7*GB),("Gen5 NVMe 12 GB/s",12*GB),("10GbE peer 1.2 GB/s",1.2*GB),("2.5GbE NAS 0.3 GB/s",0.3*GB)]:
        print(f"  {name:26s} S* = {80e12*0.5625/(2*bw):7.0f} tokens")
    print("\nVALIDATION single-node bound vs measured: M3 Ultra 819 GB/s, DeepSeek-V3.1 Q4 20.6 GB/token:",
          f"{1/(20.6*GB/(819*GB*0.6)):.1f} tok/s predicted vs 21.1 measured (EXO, 1 node)")
    print("  Strix Halo, Kimi K2.5 Q2 ~9.6 GB/token sequential on one 256 GB/s node:",
          f"{1/(9.6*GB/(256*GB*0.5)):.1f} tok/s predicted vs 9.45 measured (AMD, llama.cpp RPC, 4 nodes)")

# ---------------------------------------------------------------------------------------------
# Phase 6: seeding/leeching and striping (plan §15, roadmap v1.1). Bits/bytes conventions: Mbps = 1e6 bit/s.
Mbps = 1e6/8   # bytes per second per Mbps

def staging_time(bytes_, sources):
    """bytes / min(receiver download, sum of seeder uploads). sources = (n_seeders, up_Mbps, down_Mbps)"""
    n, up, down = sources
    return bytes_ / min(down*Mbps, n*up*Mbps)

def striped_return(S, d, k, up_Mbps=25, down_Mbps=1000, bytes_per_act=1):
    """one layer-boundary return of S*d INT8 activations striped k ways over residential uplinks"""
    total = S*d*bytes_per_act
    return total/k/(up_Mbps*Mbps) if k*up_Mbps < down_Mbps else total/(down_Mbps*Mbps)

def kv_exchange_first_peer(S, k, kv_per_token, up_Mbps=25):
    """causal attention across k sequence chunks: chunk 1's K/V must reach the k-1 later peers (uplink-bound)"""
    chunk_kv = (S/k)*kv_per_token
    return (k-1)*chunk_kv/(up_Mbps*Mbps), chunk_kv*k*(k-1)/2

def stall_prob(k, p, q):
    """P(more than p of k+p stripes stall) when each stripe independently stalls with prob q"""
    from math import comb
    n = k+p
    return sum(comb(n, i)*q**i*(1-q)**(n-i) for i in range(p+1, n+1))

if __name__ == "__main__":
    print("\n--- Phase 6: seeding / leeching (staging) ---")
    for label, b in [("one 1/16 layer slice of DeepSeek-V3 Q4 (23 GB)", 23*GB), ("full DeepSeek-V3 Q4 (370 GB)", 370*GB), ("one cold expert (24.8 MB)", 24.8e6)]:
        print(f"  {label}")
        for src_label, src in [("1 residential seeder 25 Mbps", (1,25,1000)), ("8 seeders", (8,25,1000)), ("32 seeders", (32,25,1000)),
                               ("LAN NAS 2.5 GbE", (1,2500,2500)), ("cloud CDN 1 Gbps", (1,1000,1000))]:
            t = staging_time(b, src)
            print(f"      {src_label:30s} {t:10.1f} s  ({t/60:7.1f} min)")
    print("  seeder upload budget at 25 Mbps:", f"{25*Mbps*3600/GB:.1f} GB/hour")
    print("\n--- Phase 6: sequence striping of one 4,096-token prefill, d=7168, INT8 ---")
    for k in [1, 8, 16, 32]:
        print(f"  k={k:2d}: boundary return {striped_return(4096,7168,k):5.2f} s")
    t1, tot = kv_exchange_first_peer(4096, 8, 70e3)
    print(f"  causal K/V exchange, MLA 70 KB/token, k=8: first peer uploads {t1:5.1f} s; swarm total {tot/GB:4.2f} GB")
    t1, tot = kv_exchange_first_peer(4096, 8, 328e3)
    print(f"  causal K/V exchange, GQA-70B 328 KB/token, k=8: first peer uploads {t1:5.1f} s; swarm total {tot/GB:4.2f} GB")
    print(f"  attention-at-core alternative: core uplink per layer 29.4 MB -> 58 MoE layers = {58*29.4e6/GB:4.2f} GB per prefill;",
          f"at 500 Mbps up {58*29.4e6/(500*Mbps):5.1f} s, at 35 Mbps up {58*29.4e6/(35*Mbps)/60:5.1f} min; peers' return {58*striped_return(4096,7168,8):5.1f} s")
    print("\n--- Phase 6: RAID-5/6 parity, stall probability per layer (each stripe stalls w.p. q) ---")
    for q in [0.02, 0.05, 0.10]:
        print(f"  q={q:.2f}: k=8 p=0 -> {stall_prob(8,0,q):6.1%}   k=8 p=1 -> {stall_prob(8,1,q):6.1%}   k=8 p=2 -> {stall_prob(8,2,q):6.1%}")
    print("\n--- Phase 6: per-peer throughput bound for weight-stationary batch work ---")
    for d in [7168, 4096]:
        per_tok = d*1  # INT8 boundary activation per token
        print(f"  d={d}: 25 Mbps uplink carries {25*Mbps/per_tok:6.0f} tokens/s = {25*Mbps/per_tok*3600/1e6:4.1f} M tokens/hour per peer (one boundary)")
    print(f"  decode over WAN with expert striping: 58 layers x 30 ms RTT = {58*0.03:4.2f} s per token")

# ---------------------------------------------------------------------------------------------
# Designed-behavior case: the planner picks the best of (a) pipeline-only with popularity-aware
# placement (hottest experts in the fastest tiers, no per-layer round trip), (b) expert-parallel with
# balanced placement, (c) expert-parallel with popularity-aware placement. Also multi-stream throughput.
def _trunk_and_tiers(model, mesh):
    m, M = MODELS[model], MESHES[mesh]
    tiers = sorted(all_tiers(M), key=lambda t: -t["bw"]*t["eta"])
    left, t_trunk = m["trunk"], 0.0
    for t in tiers:
        take = min(left, t["cap"]); t["cap"] -= take; left -= take
        t_trunk += take/(t["bw"]*t["eta"])
        if left <= 0: break
    nvme_total = sum(n["nvme"] for n in M["nodes"])
    if left > 0: t_trunk += left/(nvme_total*0.9)
    return m, M, tiers, t_trunk, nvme_total

def _popularity_shares(m, tiers, alpha):
    cum, shares = 0.0, []
    for t in tiers:
        c_prev = cum; cum = min(1.0, cum + t["cap"]/m["rtot"])
        shares.append(max(0.0, cum**alpha - c_prev**alpha))
        if cum >= 1.0: break
    hit = cum**alpha if cum < 1.0 else 1.0
    return shares, hit

def decode_designed(model, mesh, alpha=0.25, streams=1):
    m, M, tiers, t_trunk, nvme_total = _trunk_and_tiers(model, mesh)
    hops = (len(M["nodes"])-1)*M["t_hop"]
    if m["rtot"] == 0:
        return 1/(t_trunk+hops), streams/(t_trunk+hops)
    shares, hit = _popularity_shares(m, tiers, alpha)
    miss_par = (1-hit)*m["rpt"]/(nvme_total*0.9)
    miss_seq = (1-hit)*m["rpt"]/(7*GB*0.9)
    # (a) pipeline-only, popularity-aware
    ta = sum(s*m["rpt"]/(t["bw"]*t["eta"]) for s, t in zip(shares, tiers)) + miss_seq
    # (b) expert-parallel, balanced (capacity-proportional) placement
    cached = sum(t["cap"] for t in tiers); c = min(1.0, cached/m["rtot"]); hit_b = c**alpha if c < 1 else 1.0
    tb = max((hit_b*m["rpt"]*(t["cap"]/cached)/(t["bw"]*t["eta"]) for t in tiers if cached > 0), default=0.0) \
         + (1-hit_b)*m["rpt"]/(nvme_total*0.9) + (m["Lmoe"]*M["t_hop"] if len(M["nodes"]) > 1 else 0.0)
    # (c) expert-parallel, popularity-aware
    tc = max((s*m["rpt"]/(t["bw"]*t["eta"]) for s, t in zip(shares, tiers)), default=0.0) + miss_par \
         + (m["Lmoe"]*M["t_hop"] if len(M["nodes"]) > 1 else 0.0)
    t_exp = min(ta, tb, tc)
    T1 = t_trunk + t_exp + hops
    TN = t_trunk + t_exp*(streams**0.7) + hops   # N streams batched per step: trunk read once; expert bytes grow ~N^0.7
    return 1/T1, streams/TN

if __name__ == "__main__":
    print("\nDESIGNED CASE (planner picks best strategy; alpha=0.25): floor | designed single-stream | aggregate with 5 concurrent streams")
    for mesh in MESHES:
        print("\n==", mesh)
        for model in MODELS:
            s1, agg5 = decode_designed(model, mesh, 0.25, 5)
            floor, _ = decode(model, mesh, 1.0, "seq")
            print(f"  {model:30s} floor {floor:5.1f}   designed {s1:5.1f}   aggregate(5) {agg5:6.1f}")
```
