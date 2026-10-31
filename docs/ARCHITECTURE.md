# Kadmium (`Cd`) — Technical Architecture Specification

**Status:** Draft v0.1.0  
**Language Stack:** Rust (`tokio`, `axum`, `serde`); live worker transport is TCP, with QUIC described as a target architecture

**Target Environments:** Windows, macOS (Apple Silicon Metal / AMX), Linux (x86_64, ARM64 SBCs, RISC-V), NAS appliances, and ephemeral Cloud GPU instances.

**Current execution boundary:** This specification includes target designs as well as implemented mechanisms. Current model-related workloads run on small reference tensors. `/v1/chat/completions` and `/v1/messages` return fixed diagnostic messages and telemetry; `/v1/embeddings` returns reference vectors. The API shapes do not establish general SDK compatibility or language-model generation. The additive `launch` setup workflow starts this existing coordinator and preserves its configuration/listener behavior; it does not add an inference engine or relay.

**Dependency decision:** Keep the existing dependency set, with no Cargo manifest/lockfile changes or new direct/transitive Rust dependencies for setup. An external relay is deferred pending a future owner decision and downstream access/authentication design. See the [launch guide](guides/cli.md) and [API contract](api-reference.md).

---

## 1. Design Philosophy & Motivation

Conventional distributed AI/ML frameworks (Megatron-LM, DeepSpeed, FSDP, vLLM multi-node) are built around three assumptions that fail outside hyperscale datacenters:
1. **Homogeneous Silicon:** Every worker is assumed to have an identical GPU with identical VRAM capacity and FLOPs.
2. **Uniform Low-Latency Fabrics:** Nodes are assumed to communicate over symmetric NVLink or RDMA/InfiniBand fabrics.
3. **Dedicated 100% Availability:** Nodes are assumed never to be shared with interactive desktop users, games, battery/thermal constraints, or intermittent WAN connections.

When run on real-world personal, academic, or community hardware—such as a desktop gaming PC, an M4 Mac Mini, two gaming laptops, older laptops, Linux SBCs, and a home NAS—traditional frameworks either refuse to initialize or bottleneck the entire cluster down to the speed of the slowest node.

**Kadmium** explores a **Heterogeneity-Native Compute Fabric** through four mechanisms:
1. **Asymmetric Topology-Aware Shard Planning** from hardware, memory, and link profiles.
2. **Dynamic Compute Responsibility Planning**, with production KV/training-state migration as target architecture.
3. **Tiered Execution Models ($q^*$ and expert streaming)** inspired by **FreeToken**, **Colibri**, and **Lumabri**, currently evaluated on reference workloads.
4. **Scientific Grid Data Staging & Polite Scheduling** inspired by **CERN FTS3**, **Davix**, and **BOINC**.

---

## 2. Node Roles & Capability Vector (`kadmium-core`)

Machines are represented by hardware profiles and polite-policy envelopes. Live workers register over reachable HTTP/TCP addresses and send heartbeats; discovery, native backend coverage, and policy sensors vary by platform.

### 2.1 Logical Node Roles
A single physical device can hold multiple roles simultaneously:
- **`Coordinator`:** Maintains topology/state, runs planners, schedules reference transfers, and exposes the Dashboard/API. Coordinator failover scenarios do not establish production quorum guarantees.
- **`ComputeWorker`:** Executes assigned synthetic reference stages over TCP; native execution is limited to specifically documented bridges/kernels.
- **`DraftAndUtilityWorker`:** Target role for tokenization, draft generation, data preparation, and scoring. Current speculative acceptance is modelled.
- **`StorageVault`:** Storage/catalog role for `.ktb` banks and reference checkpoints. Direct NAS-to-worker QUIC seeding is target architecture.
- **`BurstAccelerator`:** Planning role for ephemeral capacity. Provider procurement and live drain/failback are not implemented managed services.

### 2.2 Multi-Tier Memory Hierarchy
Rather than treating memory as a binary "fits in VRAM or OOM", Kadmium models five explicit memory tiers per node. Bandwidths below are representative hardware-profile inputs, not measured Kadmium throughput:

| Tier | Examples | Typical Bandwidth | Primary Kadmium Allocation |
| :--- | :--- | :--- | :--- |
| **Tier 0: Discrete VRAM** | NVIDIA RTX / AMD Radeon VRAM | 300 – 1,800 GB/s | Active dense transformer layers, hot MoE experts, active micro-batch activations |
| **Tier 1: Unified Memory** | Apple Silicon M1–M4 Pro/Max/Ultra | 120 – 800 GB/s | Large contiguous layer ranges, global KV-cache pools, un-checkpointed training activations |
| **Tier 2: System RAM** | DDR4 / DDR5 Host Memory | 30 – 90 GB/s | Warm MoE expert banks, frozen QLoRA base weights, pinned prefetch buffers, optimizer states |
| **Tier 3: Local NVMe** | PCIe Gen4/Gen5 SSD | 3.5 – 14 GB/s | Cold MoE expert banks (`colibri`-style on-demand streaming), activation swap, local shard cache |
| **Tier 4: Peer / NAS Fabric** | 1GbE / 2.5GbE / 10GbE / Thunderbolt | 0.1 – 5.0 GB/s | Third-party FTS3 shard staging, dataset streaming, asynchronous checkpoint archiving |

---

## 3. Custom Asymmetric Model Sharding (`kadmium-sharding`)

### 3.1 The Asymmetric Sharding Formulation
Given a model $M$ consisting of $L$ layers (where each layer $l \in \{1..L\}$ has dense parameter footprint $W_l^{\text{dense}}$, optional MoE routed expert pool $E_l = \{e_{l,1}, \dots, e_{l,K}\}$, and per-token FLOP requirement $F_l$) and a cluster of nodes $N = \{n_1, \dots, n_m\}$ with pairwise link bandwidth $B_{i,j}$ and latency $\lambda_{i,j}$:

Instead of assigning $L / m$ layers to each node, Kadmium computes an **Effective Capacity Score** $C(n_i)$ weighted by the workload type $w \in \{\text{Inference}, \text{Training}, \text{Batch}\}$:

$$C_{\text{eff}}(n_i, w) = \alpha_w \cdot \text{FLOPs}_{\text{avail}}(n_i) + \beta_w \cdot \text{Mem}_{\text{fast}}(n_i) + \gamma_w \cdot \text{BW}_{\text{tier}}(n_i)$$

Where:
- **`FLOPs_avail(n_i)`** is scaled by the node's current **Polite Availability Multiplier** $\rho(n_i) \in [0.0, 1.0]$ (e.g., $\rho = 1.0$ when idle, $\rho = 0.15$ when the user is actively gaming on their main PC).
- **`Mem_fast(n_i)`** combines Tier 0 (VRAM), Tier 1 (Unified Memory), and bandwidth-discounted Tier 2 (System RAM) via the **FreeToken-inspired $q^*$ policy**.

### 3.2 Tiered Placement Heuristic ($q^*$)

The FreeToken-inspired $q^*$ value guides placement between fast accelerator/unified memory and host/streaming tiers. The current `HardwareNode::compute_q_star_ratio` implementation is a heuristic, not a measured time-equality equation:

- unified-memory profiles return `0.96`;
- nodes without VRAM return `0.0`;
- discrete-GPU profiles combine polite-adjusted FP16 compute, CPU compute, interconnect, and NVMe terms, clamped to `0.10..=0.95`.

The value informs planner assignments; it does not prove that dense attention, expert streaming, draft models, or native Metal/CUDA execution is active on a node.

### 3.3 TPU / CIM-Inspired Dataflow Models (`DataflowStreamPlanner`)

For $Y = X \cdot W$, activations or weights must move across memory/network tiers. The planner compares four modelled variants:

1. **`ComputeInMemoryTokenStreaming`:** weight-stationary token streaming; the current report recommends it for low-batch cases.
2. **`TokenStationaryWeightStreaming`:** token-stationary weight streaming; the current report recommends it for high-batch cases.
3. **`DisaggregatedDualStream`:** models separate prefill and decode streaming arrangements.
4. **`AdaptiveTieredHybrid`:** models tier-dependent selection across nodes.

Reported crossover points, overlap, and throughput are model outputs, not evidence of a systolic/CIM hardware implementation or measured loaded-model execution.

### 3.4 Multi-Environment Pool Slicing (`SubClusterPartitioner`)
For larger pools, `SubClusterPartitioner` computes $K$ stratified **planning partitions (`ComputeSubCluster`)**, such as five groups of 20 nodes. Purposes are `ProductionDeployment`, `StagingAndTesting`, `DistributedTrainingLab`, `RlAndDreamRsiArena`, and `BatchAndEvalPool`.

Each record contains an API namespace, member nodes, an elastic-borrowing policy, and a shard plan. Current partitions do not create strict SLA routing, per-slice coordinators, hard resource/network fences, authenticated tenants, or automatic borrowing/re-slicing.

---

## 4. Dynamic Compute Responsibility Planning

`ShardPlanner::plan_responsibility_shift` recomputes placements for activity and membership scenarios, including gaming and a sample cloud-burst node joining. Shift reports describe layer changes and modelled transfers.

The target live-migration sequence is to pre-stage static banks, hand off mutable KV/optimizer state at a step boundary, and then change the routing epoch. That complete sequence is **not** implemented as a production guarantee: direct NAS-to-worker FTS3/QUIC staging, zero-downtime model migration, rented-node lifecycle, and uninterrupted training/API service remain integration work.

Grid KV-tier and replay mechanisms in section 9.12 are separate reference bookkeeping/execution paths; they do not turn shift reports into measured live failover.

---

## 5. Distributed & Decentralized Training Engine (`kadmium-runtime`)

Training on real-world personal, academic, and community hardware requires overcoming both **compute/memory asymmetry** (no single device fits a 70B+ model) and **network asymmetry** (Gigabit/Wi-Fi LANs and consumer WAN links cannot run datacenter `AllReduce`).

Kadmium contains reference mechanisms inspired by **Prime Intellect (`OpenDiLoCo`)**, **Nous Research (`DisTrO` / `DeMo`)**, **Learning@Home (`Hivemind` / `Petals`)**, and **Gensyn (`Verde` / `SkipPipe`)**. They are not yet a unified persistent large-model training architecture:

### 5.1 Mode A: Pipeline Scheduling and Boundary References

`Pipeline1F1BScheduler` models asymmetric warmup, steady-state, and cooldown schedules, while blockwise INT8 tensors provide reference boundary quantization. These mechanisms establish schedules and arithmetic on synthetic inputs, not persistent 1F1B training across real model weights. Traffic-reduction percentages are workload/model calculations unless separately recorded in the benchmark ledger.

### 5.2 Mode B: LoRA / QLoRA References

The runtime includes quantized tensor-bank and adapter primitives. Current legacy training previews are stateless per-stage calculations; they do not persist adapter/model weights or establish a measured memory/network reduction for a real fine-tune.

### 5.3 Mode C: Sharded-Island DiLoCo (target architecture)

DiLoCo synchronizes complete replicas after multiple local optimizer steps. When a replica is too large for one device, the target design is to partition it across devices **within each island**:

1. **Pipeline partition within an island:** A local group would collectively hold one complete replica split into layer ranges $[L_0..L_a], [L_{a+1}..L_b], \dots$. Every other island would collectively hold the same complete layout.
2. **Local optimization:** Each island would execute $H$ optimizer steps over that replica using its own data and a pipeline schedule.
3. **Outer synchronization:** Matching parameter ranges across complete replicas would contribute master-minus-local pseudo-gradients to a persistent Nesterov outer update. Pipeline execution and peer-to-peer transport for this design remain integration work.

The current [`DiLoCoSession`](../crates/kadmium-runtime/src/diloco.rs) implements the synchronous outer barrier, complete-replica layout validation, persistent master/momentum state and checkpoints. Its `ParameterShard` is a named parameter slice repeated on **every** island, not a pipeline stage owned by one island. It receives all shards from each caller after `H` local steps; it neither runs the local learner nor distributes a replica across devices. A separate single-process CPU regression experiment exercises this session API with complete local replicas. The existing `/api/v1/training/step` and grid training paths remain stateless per-stage previews and are not an implementation of the pipeline-sharded target above.

### 5.4 Mode D: DeMo / DisTrO Frequency-Domain Momentum Compression (`NousResearch/DisTrO`)
Even outer-loop synchronization or cross-node gradient averaging can bottleneck slow home internet uploads (e.g., 20–50 Mbps upstream).
Inspired by **Nous Research's DisTrO / DeMo (Decoupled Momentum Optimization)**:
- Optimizer momentum states $m_t$ are transformed into the frequency domain via chunked **Discrete Cosine Transform (`DCT`)**.
- Only the **Top-$k$ fastest-moving frequency components** ($q_{\text{dct}}$) are extracted and transmitted across peers, while unselected frequencies remain in a local **Residual Error Feedback Accumulator** $e_t$:
  $$m_t = \beta m_{t-1} + g_t, \quad c_t = \text{TopK}(\mathcal{F}_{\text{DCT}}(m_t + e_{t-1})), \quad e_t = (m_t + e_{t-1}) - \mathcal{F}_{\text{DCT}}^{-1}(c_t)$$
- The wire ratio depends on tensor size, selected `top_k`, and packet encoding. Residual error feedback preserves unsent information locally, but does not by itself prove convergence or a universal compression ratio. The stateful DiLoCo session currently sends full FP32 parameter shards rather than automatically applying DeMo.

### 5.5 Mode E: Routing and SkipPipe Reference Models

Grid matching, replica selection, and SkipPipe-inspired schedulers provide routing/recovery mechanisms over reference work units. They do not establish production micro-batch recovery or uninterrupted global training steps for opportunistic large-model swarms.

### 5.6 Mode F: RL Topology and GRPO References

`RlTopologyPlanner` models actor, environment, and learner placement. The current `GrpoEpochExecutor` verifies eight synthesized programs for a fixed polynomial and computes normalized group advantages. `WasiMicroSandbox` is a custom fuel-metered integer stack machine, not a general WASI runtime or code compiler.

Model rollout generation, arbitrary code execution, PPO/RLOO, and policy-weight updates remain target integrations.

### 5.7 Mode G: Dream-RSI Replay (`kadmium-rl`)

Inspired by **[Dream-RSI](https://www.dream-rsi.com/)**, `DreamRsiSimulator::run_dream_cycle` traverses an in-memory sample `DiscoveryTree` with candidate exploration policies. Cached node scores make this CPU graph traversal without new GPU evaluations.

The bundled sample reaches a normalized **2.31× peak** after **4 of 9 unit-cost evaluations**, a **55.6% evaluation-cost saving** relative to its baseline discovery order. Current CLI/API paths do not record persistent NAS trees, execute real kernel optimization, distribute thousands of rollouts, or inject an evolved policy into online training. Persistent replay-world construction and policy feedback are target architecture.

---

## 6. FTS3 / Davix Data Plane & `KTB` Weight Format (`kadmium-data`)

### 6.1 Kadmium Tensor Bank (`KTB`) Format
Inspired by **FreeToken (`FTW`)** and **Colibri**:
- Standard `safetensors` or `GGUF` files often store tensors in logical naming order, requiring expensive seeks or repacking when streaming specific MoE experts or layer shards.
- **`.ktb` (Kadmium Tensor Bank)** aligns pages to strict **64 KiB (65,536-byte)** boundaries and groups tensor metadata by layer/expert/encoding. Chunk checksums and Merkle proofs use FNV-1a 64, not Blake3; FNV is not collision-resistant authentication.
- Current readers provide zero-copy slices into already loaded byte buffers. The local importer uses whole-file `std::fs::read`; direct `mmap`, `io_uring`, and Windows `IoRing` file paths are planned.

### 6.2 Third-Party Transfer Scheduler (CERN FTS3 Pattern)
In naive distributed systems, the coordinator downloads and forwards weights to workers, becoming a severe bottleneck.
The current `Fts3TransferScheduler` constructs transfer plans and `Fts3ChunkSlicer` computes chunk manifests/missing-chunk diffs. In the live reference path, the coordinator sends synthesized bank pages to workers over TCP and verifies them.

The TCP sender uses an InferSSD-style completion-driven window of at most eight page requests across the swarm (`io/pool-ring.cpp` and `io/submit-window.h` in InferSSD). It refills a slot as soon as one request finishes rather than waiting one network round trip per page or draining an entire batch. Each page's proof and JSON frame are prepared once and shared across destinations on persistent, request-correlated peer channels. The window bounds retained transfer buffers independently of bank size and worker count; it does not reserve a global budget across separate staging calls. Only matching, verified receipts count toward goodput. Local demonstration receipts are used only when there are no online workers, never to mask failed remote transfers. This optimizes the existing reference transfer, not production NAS-to-worker object storage; no WAN throughput claim is implied.

Reproduce the controlled loopback latency experiment with `cargo test --release --locked -p kadmium-runtime fts3_latency_benchmark -- --ignored --nocapture`. It compares stop-and-wait with the eight-request window over 32 pages and an injected 20 ms reply delay. Both runs reuse the same channel; set `KADMIUM_DELTA_LINK=0` to exclude delta-store warming. This is a latency-isolation experiment, not a disk, WAN, erasure-coded bucket, or production throughput benchmark.

The following remains **target architecture**: a coordinator orders a source peer to open direct multi-stream QUIC transfers to a destination, and both peers report production progress/retry/checksum state. No current FTS3/Davix network service or direct NAS-to-worker stream is established by the planner.

---

## 7. BOINC & Gensyn Verde: Polite Governance & Cross-Silicon Verification

To make it effortless for friends, family, academic labs, and decentralized peers to contribute idle machines without degrading their daily usage or corrupting training runs:

### 7.1 Polite Resource Governor (`BOINC`-Inspired)
- Models CPU/GPU budgets, foreground activity, battery/thermal states, and schedules; live sensor coverage is limited and platform-dependent.
- Uses a **target/estimated sub-200 ms** fast-yield policy with hysteresis. This is not an observed cross-platform latency SLA, and native swapchain/foreground drivers plus guaranteed live rebalancing remain future work.

### 7.2 Cross-Silicon Reproducible Verification (`Gensyn Verde` & `REE` Inspired)
Classic BOINC verification relies on bit-exact output matching across identical CPU binaries. In AI/ML, an NVIDIA RTX 4090 (`CUDA`), an Apple M4 Mac Mini (`Metal`), and an AMD/Linux laptop (`ROCm`/`Vulkan`) produce slightly different IEEE-754 floating-point results due to non-associative parallel FMA reductions.
Kadmium applies the two mechanisms Gensyn published for this, RepOps and Verde, to its work units; section 9.6 has the protocol and what it proves today.
1. **RepOps (reproducible operators, `kadmium-runtime::repops`):** every reduction in the reference math is a pairwise tree over the index space, every transcendental a fixed-order polynomial on IEEE basic operations, and every product is rounded once with no fused multiply-add. A backend may chunk, thread or reassociate as it likes; the bits do not change. A versioned manifest names the conventions, and its id is folded into every digest.
2. **Verde (refereed layer bisection, `GridDispatcher::referee_dispute`):** every result commits to the RepOps digest of each layer boundary it ran. When two untrusted copies of a unit disagree, the referee finds the first divergent layer from the two chains, has a disputant open the boundary before it, checks the opening against the digest both committed to, and re-runs that one layer. The disputant whose digest it reproduces wins; the other is penalised with the layer named.
3. **Contribution Ledger:** Tracks verified FLOP-hours, bytes staged via FTS3, RL rollouts verified, and tokens/steps contributed per node across the cluster.

---

## 8. Open-Core and Target Hybrid-Cloud Architecture

The self-hosted repository provides planners, local interfaces, reference execution, and simulations. Managed cloud functionality is a target built around `HybridCloudArbitragePlanner` and related policy types, not an operating service.

### 8.1 Current Companion state vs. managed control plane

- **Self-hosted:** local coordinator, hardware profiles, reference workloads, and simulations need no cloud account.
- **Cloud-managed mode:** `POST /api/v1/companion/mode` only stores local mode fields. There is no `kadmium login` command, managed coordinator connection, automatic WAN relay, enforced tenant RBAC, or encrypted remote mirroring in this path.
- **Embedded Condenser:** `kadmium-gateway` depends on the GUI-free `kadmium-condenser` library and mounts its router at `/condenser/`. The `kadmium-condenser` command and its native window (`wry`/`tao`, plus `clap`) live in the separate `kadmium-condenser-app` package, so the headless coordinator and `kadmium` CLI link no webview stack. The mount requires a loopback peer, same-origin requests and a per-launch access secret. Its workspace root is an explicit per-user directory, never the coordinator's working directory.

### 8.2 Spot-arbitrage simulation

The current arbitrage fixture compares hard-coded local and provider offers: **$1.82/hr versus $13.70/hr**, an **86.7% modelled saving**. `query_live_orderbook` returns an in-memory array and `provision_optimal_middle_shard` constructs a descriptor; neither contacts a provider nor creates capacity. Prices and savings are illustrative, not live quotes.

### 8.3 Failover simulation and production requirements

Current scenario outputs are 340 ms local-to-cloud, 185 ms cloud-to-local/multicloud, and 210 ms failback; the watcher fixture reports 340 ms. Zero-dropped-request fields are hard-coded scenario results.

Production CloudShield still requires authenticated provider integrations, procurement and spend enforcement, warm state replication, liveness monitoring, live state migration, billing, and measured failure testing. No current RTO, zero-loss, or automatic-failback guarantee is made.

---

## 9. The Grid Substrate (`kadmium-grid`): One Computer Assembled From Parts

Everything above describes AI workloads. Underneath them sits a generic distributed-compute substrate, in the tradition of BOINC's work units and CERN's grid data management, so that a mesh of unequal machines behaves like one computer: nodes are the components, you add compute nodes for more compute and storage nodes for more storage, and computation tasks run on the whole.

| In the one-machine picture | In `kadmium-grid` | Heritage |
| :--- | :--- | :--- |
| A process | `WorkUnit`: one piece of computation with content-addressed inputs, placement constraints, a cost estimate, a deadline and a redundancy policy. Each attempt to run it is a `WorkInstance`. | BOINC workunits and results |
| A job with dependencies | `WorkGraph`: a DAG whose ready set is the wavefront the scheduler hands out | DAGMan, grid workflows |
| The disk map | `ReplicaCatalog`: which node holds which content in which memory tier, with `PlacementRule`s that diff against it into `StagingJob`s | Rucio rules, FTS3 staging |
| The process scheduler | `WorkQueue` and its `Matchmaker`: capability, locality, politeness, trust and deadline matching; pull or push dispatch; straggler duplication; retries | BOINC scheduler, pilot-job late binding |
| The audit trail | Quorum validation per trust tier (`Trusted`, `BitExactDigest`, `ScalarTolerance`, `VerdeBisection`) and a `CreditLedger` that credits only validated work and settles it into supply-grid receipts at the canonical clearing rates | BOINC redundancy and credit |
| The spec sheet | `PoolInventory`: the mesh summed as one machine | Grid information systems |

### 9.1 Lifecycle and Validation

A unit moves `Unsent -> Assigned -> Running -> Returned -> Validated -> Assimilated`, with `Invalid`, `Expired` and `Failed` as the other exits. Owned hardware (`TrustedCore`) validates on first return. Verified peers validate within a scalar tolerance, which absorbs cross-silicon floating-point drift. Untrusted nodes run two instances; agreement validates, disagreement opens a Verde dispute that the dispatcher settles with a trusted referee (a `TrustedCore` worker that was not a disputant, or the coordinator in-process). The referee locates the first divergent layer from the digest chains both results carry and re-runs that layer alone (section 9.6), or the whole unit when the unit is one step. The loser's instance is invalidated and penalised in the ledger with the layer named; when the referee reproduces neither disputant, both are penalised and the unit retries on other nodes. Stragglers running past their estimate are duplicated; the first validated result wins.

### 9.2 Execution Shapes Are Schedules, Not Executors

The per-layer reference in `kadmium-runtime::moe_layer` splits a transformer layer into the three pieces the grid schedules separately: the trunk (RMSNorm, RoPE, attention over the layer's KV page, the router), the expert contributions (each a function of the trunk output alone), and the combine (residual plus contributions in expert-id order, clamped, then INT8-quantized at the layer boundary whether or not the boundary crosses a wire). Partials cross the wire as exact `f32`.

- **Pipeline chain:** one `StageForward` unit per trunk range, each folding the three pieces over its layers; a hop only at stage boundaries.
- **Expert parallel:** per layer a `TrunkForward` unit on the trunk owner, one `ExpertSlice` unit per holder of the selected experts, and a `Combine` unit; nodes compute at the same time on the same step and the mesh's pooled fast memory holds a model no single node can.

Because both shapes run the same reference math with the same boundary quantization, they produce identical bytes, which the runtime's equivalence tests assert over real TCP workers for single rows and batches. The planner (`kadmium-sharding::grid_plan`) places weights by capacity and bandwidth, spreads cold experts across the mesh's NVMe drives so they stream concurrently, keeps boundary-layer experts on owned hardware, replicates hot experts into spare fast memory, and chooses the shape per model and link from a bandwidth-bound cost model whose figures are always labelled modelled.

Training is a bag of island units (`TrainForwardBackward`) fanning into one `DeMoReduce`; a Dream-RSI replay batch is a bag with no edges steered to CPU nodes; a Verde audit is a single unit under the untrusted redundancy policy.

### 9.3 Transport

Frames carry a request id (`KDM2` header, 21 bytes; the legacy 13-byte `KDMF` header still decodes). One persistent multiplexed channel per peer carries pipelined work dispatches and demultiplexes out-of-order replies, so a fan-out layer costs one round trip rather than one connection per unit. Work units travel as `WorkDispatch` frames whose envelope holds the unit as JSON plus named binary activation blobs; results come back as `WorkResult` envelopes. Workers execute units through an `ExecutorRegistry` keyed by app kind, and the coordinator's `GridDispatcher` carries the activation one unit produces to the units that depend on it.

**Delta links (`kadmium_network::delta_link`).** Channels from the dispatcher's pool open with a `DeltaHello`. Once the hello is accepted, every frame in both directions travels as a `DeltaFrame` that carries only the bytes the peer does not already hold. This generalizes InferSSD's ggml RPC transport, where `SET_TENSOR_HASH` skips tensors the remote GPU has cached and `GRAPH_RECOMPUTE` replays a stored graph, to any frame kind. The sender cuts each payload into content-defined chunks (Gear rolling hash, about 1,024 chunks per payload, 2 KiB to 1 MiB average). Each chunk is keyed by XXH64 and length, and a chunk the receiver holds goes out as a 13-byte reference. The sender keeps an exact mirror of the receiver's LRU store, so it never asks whether the receiver holds a chunk. Both sides apply the same operations in wire order, and a reference that does not resolve closes the connection; the pool then redials with empty stores. Workers that predate delta links close the connection on the unknown hello, and the channel reconnects as a plain `KDM2` channel. Set `KADMIUM_DELTA_LINK=0` to turn delta links off, and `KADMIUM_DELTA_LINK_STORE_MB` to change the per-direction store budget (default 128 MiB; each worker connection holds one store of chunk bytes). `cargo run --release -p kadmium-network --example delta_link_bench -- serve` on a worker and `-- run <host>:7870` on the coordinator measures a link. Over Wi-Fi to an M4 Mac mini on 2026-10-30 (12 steps per scenario), the bench reported 8.0× faster re-sent stage weights, 9.1× faster decode steps with unchanged context, 6.9× faster KV appends and 5.8× faster sparse parameter updates. Fresh random payloads ran at 0.9× (the worst case). These ratios are derived from scenario timings; the peer details, checkout, and raw logs are not recorded, as noted in the [benchmark ledger](BENCHMARKS_2026-10.md).

**USB-C delta bridges (`kadmium_network::usbc_bridge`).** A delta link's stores belong to its connection, so a pulled USB-C cable used to cost every chunk the peer already held: the pool redialed over Wi-Fi with empty stores. A bridge carries one ordered stream over several lanes (TCP connections over the USB4 / NCM cable and over backup paths) and survives losing any of them. Segments of up to 64 KiB, coalesced from small writes, go to the lane that would finish them first, `(queued + len) / weight`. The receiver reorders them and returns cumulative ACKs on every lane. The sender holds up to a 16 MiB window of unacknowledged segments and moves a failed lane's share to the others. A lane fails on an error or after 750 ms without data or a 100 ms heartbeat. The dialing side redials into the same session for a 30 s resume grace, in the way InferSSD's device staging sends only what lies beyond the commit a device already holds. A redial or added lane joins only by proving, through an HMAC-SHA256 challenge-response, that it holds the resume secret. The listener returns that secret only on the lane that created the session, normally the cable, so a session id sniffed on Wi-Fi lets nobody join. The delta link above never sees a gap. `PeerChannel` dials a bridge for any `kdbr://host:port@gbps,...` address, and workers accept bridge sessions with `KADMIUM_USBC_BRIDGE_BIND`. Over Wi-Fi to the M4 Mac mini on 2026-10-30, after a lost path a bridge sent 12.5× fewer bytes and finished 2.0–2.2× sooner for decode steps (44× fewer bytes, 2.7–3.0× sooner for re-sent weights) than a redialed channel. Both lanes used Wi-Fi rather than a USB cable, ratios are derived from reported timings, and exact checkout/raw logs are absent. See the [USB-C delta bridge guide](guides/usbc-delta-bridge.md) and [benchmark ledger](BENCHMARKS_2026-10.md).

### 9.4 Endpoints

`GET /api/v1/grid/inventory` reports the mesh as one machine, with the measured telemetry per worker. `GET /api/v1/grid/plan` reports the shape choice, the placement in force, how it fits every node's offer, the staging it needs, the usage it learned from and the load balancer's account of its copies. `POST /api/v1/grid/plan/approve` records an owner's approval of that placement. `POST /api/v1/grid/run`, `/train` and `/replay` run a decode step, a training step and a replay batch as graphs on the live mesh; `POST /api/v1/grid/decode` runs several steps of one or several sessions through one runner, pipelined, drafting and verifying when asked, and continues a named session. `POST /api/v1/grid/repin` applies one re-pin boundary (tier swaps and the balancer's copies) and `POST /api/v1/grid/calibrate` measures the placement in force. `GET /api/v1/grid/sessions` lists the decode sessions and the tier of their KV pages; `POST /api/v1/grid/sessions/tier` runs a tiering pass and `POST /api/v1/grid/sessions/replay` replays a session's bounded suffix. `POST /api/v1/grid/sweep` runs the sweep harness and writes its Pareto table. `GET /api/v1/grid/credits` returns the cumulative ledger and receipts. The OpenAI-shaped, Anthropic-shaped and pipeline endpoints run a reference step through the same dispatcher; message text remains diagnostic rather than model-generated.

### 9.5 Allocation: Measured, Learned and Approved

The matchmaker and the planner allocate from four sources, in this order of authority. The mechanisms come from the two projects Kadmium's tiering is modelled on: lumabri's segment routing, calibration and donor settings, and colibri's usage file, learned pins and live re-pin.

| Source | What it is | Where it lives |
| :--- | :--- | :--- |
| **Residency** | A unit goes to the node that already holds its weights before anything else is considered. A measured speed never pulls a unit away from the node that holds its inputs. | `Matchmaker::rank` under `SchedulerPolicy::residency_first` |
| **Measured completion cost** | Every validated result carries the worker's own `exec_ms`. The queue turns it into a sample per unit of work (never a failed, invalid or cancelled instance), the worker's profile folds it into a bounded moving average per kind of work, and once a worker has eight fresh samples the matchmaker scores it by expected completion: the hop latency plus the measured cost of the unit behind everything already queued on it. The same estimate sets straggler timeouts and chooses which copy of a replicated expert serves a step. The coordinator keeps the averages between runs and restarts. | `ComputeEwma`, `WorkerProfile::completion_estimate_ms`, `TelemetryStore`, `target/kadmium-state/grid_telemetry.json` |
| **Offers** | What a node's owner lets the grid use: fast memory, RAM, NVMe, slots and allocations, with a reserve of RAM (a quarter, bounded to 1 to 4 GiB) that is never offered. The default offer comes from the capability vector and the canonical fractions; an explicit offer is sent on `kadmium join` and clamped to the machine. The planner spends offers and reports the fit: whether every trunk and KV reservation sits in fast or warm memory on its owner, what streams from the vault, and one line per shortfall. A node whose offer requires approval has its staging held until its owner approves the exact placement. | `ResourceOffer`, `PlacementFit`, `PlanApproval`, `ReplicaCatalog::plan_staging_gated` |
| **Learned usage** | Stage and trunk units report which experts every layer routed to. The coordinator folds them into a per-model usage ledger under `target/kadmium-state/usage`: a long-term count that is never decayed and a session count that halves every ten minutes. A fresh plan orders and replicates experts by that heat, and the placement in force is kept between runs so the pins stay put; a re-pin boundary swaps at most four hot streamed experts for the coldest pinned ones of their layers, only when the heat difference clears a 25% hysteresis. Replica eviction in the catalog counts reads before recency. | `UsageLedger`, `RepinPlanner`, `CachedPlacement`, `Replica::hits`, `target/kadmium-state/usage/<model>.usage.json` |
| **Calibration** | A measurement of the placement in force: at most eight decode steps of one shape within twenty seconds, recorded as milliseconds per token bound to the placement's digest, the node set, the versions that ran and the batch width. A fan-out measurement also runs one chain step on the same prompt and keeps whether the bytes matched; a record that fails that gate never steers a choice. Once both shapes carry a trusted record on the same mesh, the measurements decide the shape and the plan says `decided_by: measured`. Every record names the kernels it measured, which today are the synthetic references. | `CalibrationRecord`, `CalibrationStore`, `StrategyChoice::apply_measurements`, `target/kadmium-state/grid_calibrations.json` |

The cluster journal uses a separate OS state directory: an absolute `KADMIUM_STATE_DIR` override, otherwise `%LOCALAPPDATA%\Kadmium\state` on Windows, `$HOME/Library/Application Support/Kadmium` on macOS, or `$XDG_STATE_HOME/kadmium` (falling back to `$HOME/.local/state/kadmium`) on other platforms. Usage, telemetry, calibration, and placement files remain under `target/kadmium-state` relative to the working directory.

A worker also measures its own memory streaming rate when it joins (a single-thread copy, a floor rather than the DRAM's peak), and the cost model uses that figure in place of the nominal RAM bandwidth for that node. All of this changes speed, never bytes: the equivalence tests still assert that chain, fan-out and replica-served fan-out produce the same output.

### 9.6 RepOps and Verde: What Is Deterministic, What Is Committed, What Is Audited

The two ideas come from Gensyn: RepOps, a library of operators that produce the same bits on any hardware by fixing the order of floating-point operations, and Verde, refereed delegation in which a neutral referee settles a dispute between two providers by re-running only the step where they first disagree. Kadmium borrows the fixed reduction order, the re-implemented transcendentals, the one-multiply-add convention, the commitment to intermediate states and the one-step referee. It leaves out the Merkle trees over operator graphs and Verde's two-level game, because a Kadmium unit is one forward over a layer range and the chain of layer digests is the whole commitment; it leaves out on-chain referees and token incentives, which it rejects elsewhere too.

**What is deterministic.** `kadmium-runtime::repops` is the numerics the reference math is built on:

| Primitive | Convention |
| :--- | :--- |
| Reduction | A pairwise tree over `next_power_of_two(n)` leaves, split at the midpoint of each node, leaves past the end absent rather than added as zeros. The value depends on the operands and their indices only, so a chunked, threaded or reassociated reduction merges its partials through the same tree and produces the same bits (`TreeReducer`). |
| Dot product and norm | Every product rounded once (Rust never contracts to a fused multiply-add), then the tree; the norm is the IEEE square root of the tree sum of squares. |
| INT8 boundary energy | Squared codes summed in integers per block, exact in any order; one tree over the block scales. |
| `exp`, `ln`, `pow`, `sin`, `cos` | Fixed-order polynomials on IEEE basic operations, because the platform `libm` is allowed to differ in the last bit between CUDA, Metal, glibc and the MSVC runtime. Measured against the platform functions in the unit tests on this machine: `exp` and `ln` within 4e-7 relative, `sin` and `cos` within 2e-6 absolute on `|x| <= 100`. |
| Manifest | Version 1 names these conventions; its id is the FNV-1a 64 of that text and is folded into every digest, so results computed under different conventions never compare equal. |

Rebuilt on it: the fused RMSNorm and RoPE kernel, the paged attention kernel (a two-pass softmax over tree reductions rather than an online rescaling recurrence, which a split-KV backend could not reproduce), the SwiGLU expert, the router, the per-layer reference (trunk, expert contributions, combine), the DeMo DCT and its inverse (every coefficient contributes a term, zeros included, so the tree has the same shape for every packet), the Q4K matvec and QLoRA adapter, the DiLoCo outer step (islands reduced in canonical order, their bytes ascending) and the reduce's summaries. The tests prove bit-exactness across chunk sizes 1 to n, arrival orders and six simulated backends; chain, fan-out and replica-served fan-out still produce identical bytes; every layer boundary carries the same digest in both shapes; a DeMo reduce over three islands is bit-identical in any contributor order; the legacy pipeline and the grid chain agree digest for digest.

**What is committed.** Every result record carries `layer_digests`: 12 bytes per layer, in layer order.

| Unit | Each digest covers |
| :--- | :--- |
| `StageForward` | The manifest id, the layer and the exact INT8 codes and block scales of the layer boundary; for a batch, the rows' digests folded in row order. |
| `TrunkForward` | The exact `f32` trunk output, or the boundary when the layer is dense. |
| `ExpertSlice` | The holder's partials. |
| `Combine` | The layer boundary, which equals the chain stage's digest for that layer. |
| `TrainForwardBackward`, `DeMoReduce` | The exact bits of the shipped update, and of the master weights. |
| `VerdeAudit` | The exact `f64` state after each layer of the synthetic challenge. |

A unit's digest is the fold of its chain. A hardware bridge (K230, reTerminal CM4, M4 Mac mini) runs its stage on the device and returns one boundary, so its chain has one entry and a dispute over it re-runs the whole stage.

**What is audited.** On a `VerdeBisection` disagreement the dispatcher runs the protocol below and settles through `WorkQueue::resolve_dispute_with_reason`; the run report lists every dispute.

1. The referee is a `TrustedCore` worker that is online and was not a disputant, else the coordinator's in-process executor; with neither, the unit is cancelled.
2. It compares the two chains and takes the first position where they differ (Verde's first phase, and the referee's only work so far: `digest_comparisons`, at most `L`). Chains of different lengths, or chains that agree while the result digests differ, are not bisectable and the referee re-runs the whole unit instead.
3. The divergent layer's input is the boundary both disputants committed to at the layer before. The referee asks a disputant to open it: a probe that re-runs the unit's layers up to that boundary and returns it. An opening that does not reproduce the agreed digest is discarded (`openings_requested`, `openings_verified`). With no verified opening the referee reconstructs the boundary itself from the unit's input, which costs it the layers before the divergent one. When the divergent layer is the unit's first, the input is the unit's own, which the coordinator holds.
4. The referee runs the divergent layer alone from that input, as a probe, and compares the digest it gets with each disputant's claim at that position: `referee_layers_run` is 1 when an opening verified.
5. The disputant whose claim the referee reproduces wins and is credited; the other is invalidated with the reason `Verde: layer <l> digest not reproduced by referee`, which the ledger penalises. When the referee reproduces neither claim, both are penalised, both nodes are excluded from the unit, and it retries while the replica ceiling allows. The winner's output replaces whichever disputant returned last.

The literature's `O(log L)` is the number of interactive rounds a bisection needs when the referee must query the parties for each intermediate hash. Here the chain travels with the result, so those rounds collapse into one message and what the referee re-executes is one layer. Measured in `cargo test` over three loopback TCP workers (two untrusted disputants, one trusted referee), an eight-layer stage with a fault injected at layer 5 settled with 6 digest comparisons, 1 opening and 1 referee layer, and faults at layers 0 and 7 were named exactly; the same holds for the synthetic audit unit with a tamperer at layer 4. These are counts of work, not timings; nothing here has been timed across real silicon.

**What it does not prove.** The kernels are synthetic references on 64-float tensors. RepOps proves the mechanism and the bit-exactness of these references under any chunking; it proves nothing about the numerics of a real model until real kernels are rebuilt on the same primitives, with the manifest version bumped. The digest is FNV-1a 64, the content-addressing primitive the rest of Kadmium uses; it is not collision resistant, and the protocol does not depend on which hash it is. The trust assumption is Verde's: at least one honest disputant. The legacy audit endpoint (`/api/v1/verde/audit`) and the challenge the device scripts mirror are unchanged.

### 9.7 DeepInfer, EPLB and DSpark: What Was Borrowed

Sections 9.8 to 9.13 take their mechanisms from three DeepSeek sources, read on 2026-10-23: the DeepInfer engine (its README, `examples/README.md`, `eval/README.md`, `examples/yoco/README.md`, `doc/inference_deployment_qa.md`, the sweep and plotting scripts, the `MTPVarlen` scheduler and the batch-invariance test), the EPLB repository (`eplb.py` and its README) and the DSpark paper (arXiv 2607.05147). DeepInfer's CUDA and Ascend kernels (FlashMLA, DeepGEMM, DeepEP, TileLang) are out of scope: the runtime stays on the reference math, and every figure below says whether it was measured on those references or modelled.

| Source | Borrowed | Left out |
| :--- | :--- | :--- |
| DeepInfer, batch invariance | The property as a test over real workers: a row decoded alone produces the same bytes as the same row inside any batch, at any position, in both shapes (9.8) | The CUDA kernels that make it hold on a GPU; the deterministic sampler |
| EPLB | Load-proportional copy counts (`replicate_experts`), heaviest-first packing to the least-loaded pack with room (`balanced_packing`), two-level packing, DeepInfer's extra-expert slot budget and periodic reassignment as the re-pin boundary (9.9) | Expert groups and group-limited routing (no groups here); the NVLink weight shuffle; the fixed reassignment timer |
| DeepInfer, asynchronous scheduling | Scheduling step `t + 1` while step `t` runs; several sessions' graphs in one queue; the inter-step gap as the figure of merit (9.10) | The GPU-side kernels that recompute per-request lengths; zero-overhead launch of a fixed-shape graph |
| DSpark | Prefix survival `a_{r,j}`, the objective `tau x SPS(B)`, greedy admission with early stop, the profiled steps-per-second curve, the two-steps-behind budget (9.11) | The semi-autoregressive drafter and confidence head (no drafter exists here; confidences are modelled), sequential temperature scaling, variable-length attention kernels |
| DeepInfer, benchmark accounting | Full-acceptance versus acceptance-scaled throughput disclosed side by side; one sweep tag per configuration; points comparable only when every fixed field matches (9.11, 9.13) | The prefill and vision-encoder sweeps; GPU-hour revenue figures |
| DeepInfer, KV tiers and bounded replay | KV pages as catalog content with residency tiers; an idle session's pages sink; a bounded suffix is recomputed on resume (9.12) | 3FS, erasure coding, TTLs, GPUDirect swap-in; any real-model KV size |
| DeepGEMM | Fine-grained `1 x G` per-token and `G x G` per-block scaling, `UE8M0` power-of-two exponent scales, TMA-aligned column-major scale layout, two-level RepOps accumulation, and both `M`-grouped MoE GEMM schedules (`contiguous` and `masked`) (9.14) | Hopper SM90 WGMMA/TMA hardware descriptors, warp-group specialization, post-compilation SASS `FFMA` interleaving (`interleave_ffma.py`) |
| DeepEP | Layout inspection (`get_dispatch_layout`: `num_tokens_per_rank`, `num_tokens_per_segment`, `num_tokens_per_expert`, `is_token_in_rank`), holder-selective dispatch (`DPDS`), low-latency `1 x G` FP8/UE8M0 dispatch and combine (`DPCM`), and Zero-SM `RecvHook` communication-compute overlap (9.15) | NVSHMEM symmetric heaps, InfiniBand verbs Queue Pairs (`ibgda`), PTX `ld.global.nc` warp polling |
| Ollama (`ml/backend/ggml`) | `Q8KBlockVector` activation block quantization (`bsums`) + `vec_dot_q4_K_q8_K` integer dot-product kernel, `MulMatIdExpertBank` (`ggml_mul_mat_id` + `ggml_swiglu_split`), `QuantizedKvPageBlock` (`F16`, `Q8_0`, `Q4_0`) + sliding-window GQA FlashAttention, `NeoX` + `YaRN` RoPE, and `OllamaSingleBoxBaseline` (`Q4_K_M` VRAM+DDR5 split roofline) (9.16) | CGo GGML C++ graph allocator, GGUF mmap loader, Go HTTP runner lifecycle |

### 9.8 Graph-Qualified Units and Batch Invariance

Unit ids are qualified by their graph (`sess-a/step-7/stage-00`), so the steps of one session and the graphs of several sessions sit in one `WorkQueue` without their `stage-00`s colliding; a graph qualifies a bare name on the way in, and a queue can block a unit of one graph on a unit of another (`submit_graph_after`), which is how step `t + 1` waits for the sink of step `t`. Output references keep their old form, `<graph>/<unit>/out`, since the qualified id is the graph plus the unit.

DeepInfer's batch invariance is a property of its NVIDIA kernels: the same request gives the same tokens alone or in a batch of 64. Here it is a property test over loopback TCP workers (`a_row_decodes_to_the_same_bytes_alone_and_inside_any_batch`): four rows decoded alone, then in six batches of two to four rows at every position, in the chain shape, the fan-out shape, and the fan-out over a placement with the balancer's copies, where a batch deals a replicated expert's rows across its holders. The reference math runs rows independently, so what the test guards is the plumbing: that nothing on the way in, between units or on the way out re-quantizes a row because of its neighbours. The property held before any change; a trial of 200,000 random INT8 blocks found the dequantize-and-requantize round trip bit-exact. The chain executor now splits a batch into its rows' exact codes and scales and concatenates the rows' boundaries back, so the property rests on construction rather than on that arithmetic.

### 9.9 Load-Proportional Replication: EPLB on an Unequal Mesh

EPLB gives heavy experts extra copies, each slot to the expert with the highest load per copy, and packs the copies heaviest first onto the least-loaded accelerator with room, in two levels when groups divide across nodes. `kadmium-sharding::expert_balance` runs the same pieces with three substitutions.

| EPLB | Here |
| :--- | :--- |
| Load per expert over a window of steps | The usage ledger's heat: long-term hits plus the decaying session heat, as `min(1, k p_e)` of an expert's bytes |
| `num_extra_experts` slots per rank, rounded to divide evenly | The fast memory each node's offer leaves free after the trunks, the KV reservations and the primary copies, in expert-sized slots, spread over the layers and capped at a quarter of a layer's experts (DeepInfer's decode ratio at EP64). The cap is cumulative: copies a layer already holds count against it |
| Nodes, then the GPUs of a node | Link segments (one per `/24` of the live workers' LAN addresses, the round-trip rule otherwise), then the nodes of the chosen segment; the trunk owner's segment breaks ties |
| The busiest rank's token count | The busiest node's modelled read time for the layer: its expected bytes divided by the bandwidth of the tier they sit in. EPLB balances bytes because its ranks are identical; a copy on a slow unified-memory node can lower the bytes on the busiest node and raise the step time, which a test pins down. Bytes are reported beside the time |

Each copy is the `(expert, node)` pair that lowers the layer's busiest node the most; EPLB's load-per-copy order and then the segment level break ties; the balancer stops when no copy lowers it. In full mode it runs at planning time under the expert-parallel shape whenever the ledger has learned any routing. In incremental mode it is the re-pin boundary: colibri's tier swaps first, then copies, at most `max_swaps` changes in all, each only when it lowers a layer's busiest node by more than the 25% hysteresis; when a node has no free slot the boundary may drop its coldest extra copy to make room, which counts as two changes. A copy is a staging transfer from the primary into the node's fast tier, pinned; a change of holders changes the placement digest, so approvals and calibrations ask again.

The copies carry load within a step, not only across steps: for a batch wider than one row the step graph deals a replicated expert's rows round robin across its holders (`ExpertRowShare`), each holder computes the partials of its rows only, and the combine sums partials by expert id, so the bytes are the row's own whichever holder served it. One row still goes to the copy with the lowest measured completion estimate, the primary until the holders are measured. The cost model splits expected bytes the same way, so at one row a copy changes nothing and the plan says so (`balance.expert_read_ms_before` equals `_after` at `rows: 1`); the balancer's own figures assume the even split EPLB is built for. All of it is modelled: `BalanceReport` per layer, `BalanceSummary` on the plan's choice, and the re-pin response's critical-path expert reads before and after.

### 9.10 Step Pipelining: The Runner Kept Open Across Graphs

DeepInfer's asynchronous scheduling runs the CPU's schedule of step `N + 1` during the GPU's forward of step `N`, so the accelerator never waits for the scheduler. Here the workers are the accelerator and the coordinator is the scheduler. `GridDispatcher::run_graph` ran one graph through one queue and returned; `GridRunner` is the same loop kept open: one queue, one blob store and one set of in-flight calls, into which graphs are submitted while others run, with `dispatch_ready` (assign and send every ready unit) and `settle_one` (wait for one answer, validate or referee it, feed the worker's measured cost, tick) as the two moves. `run_graph` is now a runner that submits one graph and runs until settled, so the equivalence, replica-served and Verde tests run unchanged through it.

`run_decode_loop` runs the steps of one or several sessions through it. Pipelined, each session keeps one step queued ahead of the one running: the coordinator builds step `t + 1`'s graph while step `t` executes on the workers and submits it behind step `t`'s sink (an external prerequisite in the queue), so the first unit of `t + 1` leaves in the same loop turn that validates the last unit of `t`, with nothing planned in between. Sequential is the old shape for comparison: one graph at a time through its own queue, the next step planned after the previous one has settled. Several sessions interleave in the same queue; their graphs are independent, so the matchmaker hands the stages of different sessions to different nodes at the same time. That is batching across sessions at the level of units, not the merging of their rows into one tensor: each session keeps its own rows, and the loop asserts that a session's bytes are the same pipelined as sequential and the same as the one-graph dispatcher's, and that two sessions' bytes stay apart.

The figure of merit is the inter-step gap: the next step's first `dispatched_at_ms` minus this step's last `returned_at_ms`, both on the run config's clock and both measured on the coordinator (`GridDispatchTrace` carries them). It is recorded in the benchmark ledger as measured on the reference kernels over two loopback TCP workers, three runs each: the pipelined loop roughly halves it in both shapes, and the wall of two sessions falls as their stages overlap across the two nodes. In the fan-out shape the coordinator itself is the bottleneck, since a step is dozens of millisecond units settled one at a time; the gap that remains is that queue, not the planning. The endpoint is `POST /api/v1/grid/decode`.

### 9.11 Confidence-Scheduled Verification: DSpark's Scheduler Without Its Drafter

Speculative decoding drafts `gamma` tokens and verifies them in one forward of the target. Under batching every drafted token costs batch capacity, and one that will be rejected costs it for nothing; DSpark calls this verification waste and sizes each request's verification from its drafter's confidence. Its scheduler takes for every request `r` the conditional survival `c_{r,k}` of each draft position, forms the prefix survival `a_{r,j} = prod_{i <= j} c_{r,i}`, sorts every `(r, j)` pair by `a_{r,j}`, and admits pairs while the objective `tau x SPS(B)` improves: `tau` the expected accepted tokens, `B` the batch in tokens, `SPS(B)` the engine's steps per second at that batch, profiled once. The greedy order is exact because `a` is monotone in `j`; the early stop keeps the decision causal. DeepInfer's production variant searches every pair and sets the budget from confidences two steps behind, because its zero-overhead scheduler must know the next batch before the current step ends.

`kadmium-sharding::verification` is that scheduler. What it is given, and what it is not:

| Piece | Here | Status |
| :--- | :--- | :--- |
| `SPS(B)` | `StepsPerSecondProfile`: one point per batch width from the trusted calibration records of exactly this placement (`/api/v1/grid/calibrate` at several `rows`, or the sweep of 9.13), linear between measured widths, the step time extended along the last slope beyond them. With fewer than two widths measured, the cost model's step time at 1, 2, 4, 8 and 16 rows | Measured on the reference kernels, or modelled; the response says which |
| `c_{r,k}` | `ModelledDrafter`: a per-session base and decay (`code` roughly DSpark's conditional acceptance on code, `chat` lower and decaying, `none` zero) with a deterministic jitter per step and position | Modelled. No drafter exists; these numbers stand in for one |
| The acceptance test | A coin seeded by the verified row's RepOps digest and the position's confidence, so every node agrees on it | Modelled |
| The verify forward | The decode step over a verify batch of `1 + l_r` rows, the session's row then the drafts, through the stage chain or the fan-out; the `DraftVerify` unit on the last stage's owner accepts the longest surviving prefix and emits the accepted row with the next step's draft rows appended | Real work on the reference kernels |
| The lag | `confidence_lag`: 1 by default (step `t + 1` is planned while `t` runs, so the latest settled confidences are `t - 1`'s), 2 for DeepInfer's setting, 0 for the paper's offline one | Mechanism |

Because the reference math is batch invariant (9.8), every row of a verify batch gets the bytes it would get alone, and the loop's test proves the consequence: at every step the next input is the plain decode of the row the step accepted, and a drafter that is never right decodes exactly as a plain session. The draft rows themselves are deterministic perturbations of the session's row, computed identically by the coordinator for the first step and by the verify unit for the next, so the batch is reproducible on any node.

The accounting is disclosed the way DeepInfer's benchmark README discloses it. Its decode throughput counts `1 + nextn` tokens per request per step "regardless of the true acceptance rate", and its plotter scales that by `tau = 1 + nextn x acceptance`. The decode endpoint reports both: `tokens_full_acceptance` (one golden token plus every draft position per step) beside `tokens_accepted` (the golden token plus the accepted prefix), each over the wall as tokens per second, with `tau`, and `modelled_acceptance: true` on all of it. The loop's own `tokens` and throughput count accepted tokens. Until a real drafter reports its own confidences, every acceptance figure describes the modelled coin and nothing about a model.

### 9.12 KV Tiers and Bounded Replay

DeepInfer's deployment keeps the in-flight KV in HBM, the sliding-window state in host memory and the paged block KV on a dedicated SSD cluster with days of retention; a hit whose window state is missing backs off one block and recomputes the last 128 tokens, which its `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` setting sizes as the window plus the longest draft a request may have to back out of. `kadmium-sharding::kv_tiers` runs that shape over the mesh's tiers, with every size modelled.

| Piece | Here |
| :--- | :--- |
| A KV page | Catalog content (`ContentKind::KvPage`, `<model>/kv/<session>/L<layer>`), one per layer, resident in fast memory on the layer's trunk owner while the session is active, pinned; its bytes are the model spec's reference KV figure per layer scaled to the session's tokens, with the reference window stated as an assumption (`KV_REFERENCE_CONTEXT_TOKENS`) |
| Sinking | `KvTierPolicy`: a session idle for a minute has its pages moved to the owner's NVMe (the vault when the owner offers none), after ten minutes to the vault node; a tiering pass applies the moves to the catalog and reports modelled bytes per tier. The thresholds are the policy's and change by request; nothing moves on a timer |
| Resume | A session that continues from sunk pages runs a replay graph first: one `KvReplay` unit per trunk range on its owner, chained like a decode step, over a bounded suffix of the context (the window plus the backtrack, `140` tokens by default, or the whole context when shorter), one row per token. Each unit runs the suffix through its layers on the reference kernels, exactly as a stage unit would, and reports the pages it rebuilt with their modelled bytes and the digest of its boundary; the pages are recorded resident again and the session marked fast |
| Raising without replay | `KvTierPlanner::raise_jobs` gives the staging transfers that would bring sunk pages back as they are; the catalog's rules for a session's pages diff like any other content's |

What is real here is the work of the replay and the bookkeeping of where pages sit; what is modelled is every byte figure and the suffix rows themselves, which stand in for the tokens' hidden rows no real model produced. The sessions endpoint lists every session with its tier and modelled bytes, the tier endpoint runs a pass at a chosen moment, the replay endpoint replays a session on request, and `/api/v1/grid/decode` with a `session` replays before decoding when the session's pages have sunk. Left out: 3FS and its erasure coding, time-to-live retention, GPUDirect swap-in, and a prefix cache shared across sessions.

### 9.13 The Sweep Harness: Calibration as a Pareto Table

DeepInfer's `examples/sweep/run_sweep.sh` runs one benchmark point per decode batch size under one tag, writes every point as a JSON line with its whole configuration, and plots the points of a tag as a Pareto curve of output throughput against per-user speed; its plotter refuses a file whose points disagree on any fixed field, because such points are not one curve. `kadmium-sharding::sweep` keeps that discipline over the mesh. `POST /api/v1/grid/sweep` runs a decode loop per batch width and per node count (the first `n` nodes of the mesh, placed afresh), and every point carries its full configuration: model, shape, rows, sessions, steps, node count and node set, placement digest, the version each node runs, the kernels, the mode, the drafter when any, and the repository state. Points form a curve only when every field but the batch width matches; node counts are a second swept axis and each node set is its own curve, as each EP size is DeepInfer's. The Pareto frontier is marked within a curve and nowhere else: a point is on it when no other point of the same curve beats it on both throughput and per-user speed.

Each point measures the mean step wall, steps per second, tokens per second over the loop's wall and one session's tokens over its own wall, with the mean inter-step gap; a speculative point carries the full-acceptance throughput beside the accepted one. A plain single-session point is also a calibration record of its placement at its width, bound as `/api/v1/grid/calibrate` binds one, so a sweep over widths is how the verification scheduler's steps-per-second profile becomes measured (9.11). The table is appended to a ledger file per tag under `target/kadmium-state/sweeps/`, in the form of the benchmark ledger, and the loopback run of the harness over two TCP workers is recorded in `docs/BENCHMARKS_2026-10.md` as measured on the reference kernels. Left out: the prefill and vision-encoder sweeps and the plotting; the table is the plot.

### 9.14 DeepGEMM: Fine-Grained Block Scaling, Two-Level Accumulation and `M`-Grouped MoE Kernels

`deepseek-ai/DeepGEMM` achieves peak MoE efficiency without heavy template libraries by fixing four properties: fine-grained `1 x G` activation and `G x G` weight sub-block scaling, lossless `UE8M0` power-of-two exponent scales, column-major TMA-aligned scale buffers, two-level accumulation (inner tensor-core tile MMA promoted every `G` elements into outer CUDA-core registers), and `M`-axis grouping across MoE experts that share identical `(N, K)`. `kadmium-runtime::deep_gemm` implements those four pieces on top of `repops`:

| DeepGEMM | Here |
| :--- | :--- |
| `per_token_cast_to_fp8` (`1 x 128`) and `per_block_cast_to_fp8` (`128 x 128`), `ceil_to_ue8m0` | `PerTokenBlockQuantized` (`1 x G`) and `PerBlockQuantizedMatrix` (`G x G`), supporting both `ScaleFormat::ExactF32` (Hopper SM90 convention) and `ScaleFormat::Ue8m0PowerOfTwo` (`2^{ceil(log2(absmax / 127))}`, Blackwell / DeepGEMM lossless exponent byte) |
| `get_col_major_tma_aligned_tensor` | `ColMajorAlignedScales`: transposes `[rows, k_blocks]` activation scales into column-major storage with the leading row stride padded to a multiple of `TMA_ALIGN_ROWS = 4` (16-byte aligned) |
| Two-level accumulation (`WGMMA` tile -> `FFMA` promotion) | `two_level_scaled_dot`: exact `i32` inner product within each `G`-element tile along `K`, scaled by `lhs_scale[m, kb] * rhs_scale[nb, kb]` in `f32`, then reduced across `0..ceil_div(K, G)` through `repops::tree_sum_f32` |
| `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` and `m_grouped_gemm_fp8_fp8_bf16_nt_masked` | `DeepGemmExpertPool::forward_contiguous_grouped` (prefill layout, each expert's segment padded to `block_m` with `m_indices = -1` on padding rows) and `forward_masked_grouped` (decode layout, fixed `[num_groups, max_m, K]` buffer with device-side `masked_m: &[u32]` token counts) |

The unit tests verify that for both `ExactF32` and `Ue8M0PowerOfTwo` scales, `forward_contiguous_grouped` (across `block_m = 1, 4, 8`), `forward_masked_grouped` (across `max_m = 4, 12`) and single-row `forward_individual` produce bit-for-bit identical vectors, and that a row routed inside a multi-row batch produces the exact same bytes as when decoded alone.

### 9.15 DeepEP: Holder-Selective Dispatch, Asymmetric-Domain Forwarding and Zero-SM Hook Overlap

`deepseek-ai/DeepEP` separates expert-parallel all-to-all communication into layout inspection (`Buffer::get_dispatch_layout`), low-latency quantized `dispatch` and `combine`, asymmetric-domain bandwidth forwarding (coalescing inter-node RDMA traffic before fanning out over intra-node NVLink), and Zero-SM hook-based communication-computation overlap (`return_recv_hook=True`). `kadmium-sharding::deep_ep` and `kadmium-runtime::moe_layer` bring these mechanisms into the fan-out shape:

| DeepEP | Here |
| :--- | :--- |
| `Buffer::get_dispatch_layout(topk_idx, num_experts)` | `DispatchLayout::compute`: computes `num_tokens_per_rank` (deduplicating when a row hits several experts on the same holder and respecting EPLB's `ExpertRowShare`), `num_tokens_per_segment` over `LinkSegments` (DeepEP's `num_tokens_per_rdma_rank` for asymmetric-domain forwarding), `num_tokens_per_expert` (`masked_m` for `DeepGEMM`), and `is_token_in_rank` |
| `dispatch` and `low_latency_dispatch` | `HolderDispatchBatch` (`"DPDS"` magic) and `StepGraphBuilder::holder_dispatch_ref` (`<trunk>/dispatch/<holder>`): `TrunkForwardExecutor` emits a per-holder packet containing only the deduplicated rows `r` where `is_token_in_rank[holder][r]` is true, in `DispatchPrecision::ExactF32` (preserving bit-for-bit equivalence with chain execution) or `LowLatencyFp8Ue8m0` (`1 x G` block-scaled codes + 1-byte `UE8M0` scales, ~4x smaller on the wire) |
| `combine` and `low_latency_combine` | `LowLatencyCombineBatch` (`"DPCM"` magic): packs holder partials into `PerTokenBlockQuantized` on the return hop and dequantizes into `CombineExecutor`'s expert-id-ordered `repops::tree_sum_f32` reduction |
| `return_recv_hook=True` (0 SM occupancy overlap) | `RecvHook` and `run_with_recv_hook` in `kadmium-runtime::moe_layer`, plus `DeepEpStepAccount` on `StrategyChoice` (`/api/v1/grid/plan`), which reports side by side the modelled wire bytes (`broadcast_f32` vs `selective_f32` vs `low_latency_fp8` vs `segment_forwarded_fp8`) and the hop latency hidden behind local shared-expert / dual-microbatch compute |

### 9.16 Ollama (`ml/backend/ggml` & `ml/nn`): `vec_dot_q4_K_q8_K`, `MulMatID`, Quantized FlashAttention and the Single-Box `Q4_K_M` Roofline

`ollama/ollama` achieves high single-node and hybrid GPU/CPU throughput on quantized checkpoints (`Q4_K_M`) through four kernel and memory-layout mechanisms in `ml/backend/ggml` and `ml/nn`:

| Ollama (`ml/backend/ggml` & `ml/nn`) | Here |
| :--- | :--- |
| `block_q8_K` (`bsums`) + `vec_dot_q4_K_q8_K` | `Q8KBlockVector::quantize` and `Q4KBlockTensor::vec_dot_q4k_q8k` / `matvec_q8k` (`kadmium-runtime::quant_lora`): quantizes input activations once into 32-element `i8` blocks with precomputed block sums `bsums[b] = sum_j q_{x,b,j}`, evaluates the inner 32-element dot product in exact `i32` integer arithmetic (`q_w * q_x`), applies the `Q4_K` affine `min_val` offset in `O(1)` per block via `min_w * bsums[b]`, and reduces the `ceil(K / 32)` block terms via `repops::tree_sum_f32` (`32x` fewer tree leaves per row) |
| `ml.Tensor.MulMatID` (`ggml_mul_mat_id`) + `ggml_swiglu_split` | `MulMatIdExpertBank` (`kadmium-runtime::kernels`): packs each expert's `gate` and `up` weights into a single `[2 * intermediate_dim, hidden_dim]` `Q4KBlockTensor`, quantizes each batch row to `Q8KBlockVector` once, buckets `(row, slot)` pairs by `expert_id` so each active expert's `Q4_K` blocks are streamed once per batch, splits `gate` and `up` via `swiglu_split`, and reduces each token's `top_k` weighted outputs in canonical slot order via `repops::tree_sum_f32` (batch-invariant and bit-identical to individual `FusedSwiGluExpertKernel::forward_q8k` calls) |
| `OLLAMA_FLASH_ATTENTION` + `OLLAMA_KV_CACHE_TYPE` (`f16`, `q8_0`, `q4_0`), GQA, SWA, `GGML_ROPE_TYPE_NEOX` & YaRN | `OllamaKvCacheType` (`F16` 64 B/block, `Q8_0` 34 B/block, `Q4_0` 18 B/block), `QuantizedKvPageBlock`, `PagedGqaAttentionKernel::forward_flash_quantized_kv` and `forward_gqa_multihead_quantized_kv` (evaluating `Q * K^T` directly against quantized 32-element blocks with optional `swa_window` sliding-window attention), plus `RopeLayout::NeoXSplitHalf` and `YarnConfig` in `FusedRmsNormRopeKernel::forward_with_layout` |
| `EstimateGpuLayout` (`llm/memory.go`) hybrid VRAM + DDR5 split offload | `OllamaSingleBoxBaseline` (`modelled: true`) on `StrategyChoice` (`/api/v1/grid/plan`): evaluates every node in the mesh as a standalone `Q4_K_M` + `q8_0` KV runner that fills fast VRAM/UMA first and streams overflow weights in place from host DDR5 with zero network hops, reporting `single_box_tokens_per_sec`, `distributed_speedup_vs_single_box_q4km` and `beats_single_box_ollama_q4km` side by side with the distributed plan |


### 9.17 Native CUDA Kernels in Pure Rust (`kadmium-cuda`, NVIDIA cuda-oxide)

`kadmium-cuda` brings the RepOps contract onto NVIDIA GPUs with kernels written in Rust and compiled to PTX by NVIDIA's cuda-oxide backend (`NVIDIA/cuda-rust`, pinned at rev `18a0407cc2`, toolchain 1.99.1). The workspace crate holds no CUDA code and builds everywhere: its lane math is generic over four rounded operations (`fp::Fp`), so the host instantiates it with Rust's operators and `device/` instantiates it with explicit `add.rn.f32` / `mul.rn.f32` / `div.rn.f32` instructions, which `ptxas` may not contract into an `fma`. The RepOps tree is computed the way a warp can compute it: lane `j` holds leaf `32c + j`, a `shuffle_xor` butterfly at distances 1, 2, 4, 8, 16 joins exactly the canonical tree's sibling subtrees, and a `TreeStack` folds the aligned 32-leaf subtrees left to right. Absent leaves are skipped rather than added as zeros, as in `repops::tree_sum`.

| Kernel | Adapted from (InferSSD CUDA) | Guarantee |
| :--- | :--- | :--- |
| `q8k_quantize` | `activation-quant-reuse.cu`: quantize once per token, reuse for every routed expert | Bit-identical to `Q8KBlockVector::quantize` |
| `q4k_q8k_matvec` | `mmvq-q8k.cu`: fixed-order `Q4_K` x `Q8_K` (`dp4a` + `prmt` block dots) | Bit-identical to `Q4KBlockTensor::matvec_q8k` |
| `mxfp4_matvec` | `ssd/mxfp4-kernels.cuh`: GGUF `MXFP4` routed experts (`E2M1` codes, `E8M0` scale) | Bit-identical to the RepOps dot of the dequantized row |
| `swiglu` | (none) | Bit-identical to `MulMatIdExpertBank::swiglu_split` |
| `moe_top_k` | `top-k.cu`: NaN as `-f32::MAX`, ties to the lower expert id | Exact |
| `row_mask` | `row-mask.cu` | Exact |

The chain of evidence has two links. `device/` runs every kernel on the GPU and compares its output with `kadmium_cuda::reference` bit for bit (`cargo oxide run`). The runtime's `cuda_parity` tests compare that reference with the runtime's own `Q8KBlockVector`, `Q4KBlockTensor`, `repops::exp` and `repops::silu`. Every NaN counts as one value: IEEE 754 lets the hardware choose a NaN's payload, and x86 returns `0xffc00000` for `inf - inf` where an NVIDIA GPU returns `0x7fffffff`; whether a result is NaN is still exact. `HardwareKernelTarget::native_cuda_kernels` lists the kernels a CUDA node runs natively.

Reported for an RTX 5070 (sm_120, CUDA 13.3, driver 616.92, WSL2): all cases bit-identical; at 4096 x 4096, `q4k_q8k_matvec` takes 33-38 us per launch (330-380 GB/s of weights) and `mxfp4_matvec` 100-110 us (81-88 GB/s) across runs. Exact checkout, run date, repetition count, and raw logs are not recorded; see the provenance and Measured/Derived labels in the [benchmark ledger](BENCHMARKS_2026-10.md). The generated PTX holds no `fma`: every `f32` operation is an explicit `add`, `sub`, `mul` or `div` `.rn`. The `MXFP4` kernel makes every product a leaf of the tree, so it is bound by the reduction rather than by memory; a `MXFP4` x `Q8_K` integer block dot, as in InferSSD's own expert path, would move it onto the `Q4_K` kernel's shape and is the next step.

### 9.18 Portable NPU Kernels, Intel First (`kadmium-npu`)

`kadmium-npu` supplies a vendor-neutral `NpuBackend` / `MatMulKernel` contract and a real Intel AI Boost implementation, also exposed as `kadmium_runtime::npu`. Its first kernel computes row-major `C[M,N] = A[M,K] * B[K,N]`. Rust generates the static graph and weights, discovers an Intel VPU through Level Zero, compiles through the installed driver's graph extension, validates the compiled tensor ABI, and submits work with reusable host-visible buffers. All new application code is Rust; Intel's driver compiler and firmware supply the native device instructions. There is no OpenVINO runtime dependency or external model conversion step.

The arithmetic contract is explicitly approximate FP16 with FP32 I/O. It does not enter the RepOps digest-verification dispatcher or automatically replace existing CPU paths. Shape and tensor validation, bounded synchronization, poisoned execution state and resource lifetime management are part of the portable API boundary. A failed final synchronization quarantines native resources until process exit instead of freeing memory that an NPU could still access.

Windows Intel AI Boost hardware validation covers rectangular and non-tile-aligned matrices, repeated and random inputs, independent prepared weights, and resource ownership after dropping the backend. Ordinary cross-platform tests include driver fault injection and ABI checks. See the [NPU crate guide](../crates/kadmium-npu/README.md) for setup, exact tested hardware, numerical limits and commands. Further vendors implement the same traits; additional operators and automatic runtime scheduling remain separate work.

### 9.19 InferSSD: Expert Cache, Next-Layer Prefetch, Learned Cost Lines, Coalesced Reads, the Prompt Cache and the Host Envelope

`wici-ai/inferssd` runs large MoE checkpoints from SSD by keeping the model's native routing and managing everything around it: which expert weights stay in RAM, which are read ahead for the next layer, how reads are merged and submitted, what each placement costs on this machine (learned, not assumed), which cached prompt a new request starts from, and how much host memory the caches may take beside the process's other needs. The mechanisms below are its policies in Kadmium's shapes. None of them changes a layer's bytes: the cached and uncached paths are compared bit for bit in `kadmium-runtime::moe_layer` and `grid_executor`, and the prefetch hint fills slots only.

| InferSSD (`engine/ggml/src/ggml-ssd`, `tools/server/inferssd`, `framework`) | Here |
| :--- | :--- |
| RAM expert cache, `count_decay2` replacement (selection counts halving every 2 tokens), hits pinned before admission, admission only at or above the victim's score, uniform per-layer quota, `shrink_to` coldest-first, online LRU miss-ratio curve | `ExpertCache` (`kadmium-runtime::expert_cache`): entries keyed by model, weights revision, layer and expert; counts kept in log space against one token clock, so a layer that stops running decays like any other; a sorted victim index per domain, so a lookup is a hash probe and an admission a few ordered-set operations; a byte budget charged at what each fill measures, a bounded entry count, pins that last one step, `reclaim` / `grow` in bytes, `MissRatioCurve` at the quotas `1..512`. Every worker owns one (`LiveNodeWorkerServer::expert_cache`), locks it only around lookups, and skips it when its budget is zero; `StageForwardExecutor` and `ExpertSliceExecutor` (both the selective-dispatch and the full-routing inputs) serve their experts through it |
| Next-layer prefetch hint: layer `l + 1`'s norm and router run on layer `l`'s FFN input, `plan_prefetch` rank accounting (`novel` / `skipped` / `request`), reservations dropped when unused | `LayerReference::prefetch_hint`, `ExpertCache::plan_prefetch` / `prefetch_done` (a prediction takes the place of any resident whose count has decayed below one), used by chain stages (`run_layers_cached`) when `KADMIUM_EXPERT_PREFETCH_DEPTH` is set: the hint costs a second router run per layer, so it is off by default; `LayerCacheTrace` reports `hint_precision`. The fan-out shape does not prefetch yet: a holder of layer `l + 1` is another unit, and shipping it the hint is future work |
| Learned cost lines: EMA `0.9 / 0.1`, least squares pulled toward a prior pseudo-sample (`kimi_prefill_fit_line`), a 64-sample drive window, probe-then-decide thread selection (400 / 40 000 experts), the two-station tandem-queue time `kimi_prefill_coupled`, uniform-routing coverage `E (1 - (1 - k/E)^n)` | `kadmium-core::learning` (`Ewma`, `LineFit::fit_with_prior`, `WindowedLine`, `CandidateProbe`, `coupled_pipeline_time`, `batch_coverage`); `HopLine` on every `WorkerProfile` learns `fixed ms + ms per MB` over a 64-hop window from the coordinator's own dispatch timings (wall time less the worker's `exec_ms`, against the bytes carried) and persists in `TelemetryStore`; `StrategyCostModel` prices hops by that line (`measured_hop_lines`) once learned, and the round trip alone before, and prices expert bytes by `CoverageModel::Exact` when asked (`with_coverage`; the fitted power form stays the default) |
| Adaptive VRAM admission `admission_hits = upload_us / (miss_us - hit_us)` | `RepinPayback` on `RepinPolicy`: a live re-pin swap must earn, beyond the demoted expert, enough extra hits to repay its two transfers (across the mesh's slowest uplink, or two memory copies for a swap on one node) at the mesh's fast-tier and streamed bandwidths; opt-in at `POST /api/v1/grid/repin` with `payback: true` |
| Contiguous expert ranges merged into one read (`build_copy_plan`), 4096-byte alignment, adaptive submit chunk (`<= 4 -> n`, `<= 8 -> 4`, else `2`), a pool of at most 8 reader threads | `kadmium-data::batch_read` (`plan_reads` with a 64 MiB cap per merged read, `adaptive_submit_chunk`, `read_planned` with fallible buffers), `SafetensorsManifest::read_tensors` (one handle per shard, merged aligned reads in parallel instead of open + seek + read per tensor) and `HfByteRangeSlicer::coalesce_range_requests` (one HTTP range per run of adjacent tensors). Library paths so far: the importer and the pull planner still read tensor by tensor |
| Server prompt cache: LRU prompts with checkpoints, `f_keep >= 0.25`, a candidate must beat the best on both `f_keep` and similarity, whole-state or deepest-checkpoint reuse under full memory, one token left for logits, eviction by bytes then tokens, FNV-1a keys and a build/model fingerprint | `PromptCache` (`kadmium-sharding::prompt_cache`): the cross-session hit policy 9.12 left out; the gateway looks a decode step's prompt up before it runs and saves the session's modelled KV state after (`GridStepOutcome.prompt_reuse`) |
| Host memory envelope: a finite limit less a 512 MiB reserve, a system term against `MemAvailable` less a floor, tickets for every allocation, a reclaim chain the caches answer, a margin that only grows to untracked memory | `HostMemoryLedger` (`kadmium-core::host_ledger`) with the platform probe (`/proc/meminfo`, `GlobalMemoryStatusEx`, `vm_stat`); every worker builds one at start, books the bytes the expert loader holds (refreshed at heartbeats), takes its routed-expert cache budget under it and registers that cache as the reclaimer, books a delta link's two chunk stores before accepting the link (the cache yields if the ledger is short; a link the ledger cannot book is declined), grows the cache back toward its configured budget at heartbeats when room returns, and reports the ledger's room as `free_system_ram_gb` in its registration and heartbeats (recorded by the coordinator; the planners do not consume it yet) |

Settings: `KADMIUM_EXPERT_CACHE_MB` sizes a worker's expert cache (default a quarter of the ledger's room, at most 256 MiB; `0` turns it off), `KADMIUM_EXPERT_PREFETCH_DEPTH` turns the chain-stage prefetch on (default `0`, off), `KADMIUM_HOST_BUDGET_MB` fixes the envelope where the platform probe should not be trusted.

### 9.20 KIP/1 Interdevice Transfer Protocol (`kadmium-data::interdevice`)

`kadmium-data::interdevice` defines KIP/1, a node-local transfer protocol between `Endpoint { Ram, Vram, Storage }` with directional placement costs, bounded submission windows (`Limits`), and a safe `HostBackend`.

**Current status:** Implemented protocol and host backend; not wired into any execution path; GPU adapters are target only. See the [interdevice protocol guide](guides/interdevice-protocol.md).

### 9.21 Local Aggregate Compute (Target)

**Status: design proposal**, except opt-in queue scratch admission in
`kadmium-grid::WorkQueue`. No shipped workload supplies nonzero scratch estimates.
The [admission guide](guides/local-compute-fabric.md) documents implementation,
scientific-grid references and current limitations. This section owns the target
design; it does not introduce another execution substrate.

#### Resource authority and attempt identity

Preserve the section 9.5 authority order. Spend only an owner-approved
`ResourceOffer` through `PlanApproval`; use `PlacementFit` for resident allocations
and `CreditLedger` for economic accounting. Extend the existing `HostMemoryLedger`
with byte-denominated execution tickets shared by residency, scratch, expert caches,
KIP staging and transfer windows. Do not add an independent allocator. Issue
[#60](https://github.com/mESmaC/Kadmium/issues/60) tracks authority and memory-tier
mapping across those paths. Current offer headroom does not automatically subtract
resident placement, so admission alone is not safe physical memory enforcement.

Map proposed wire operations onto the existing `WorkInstance` lifecycle:

| Operation (target only) | Existing lifecycle and authority |
| --- | --- |
| Reserve/admit | `Unsent` to `Assigned`; use the instance ID, approved offer, coordinator epoch and resource ticket. A busy refusal must identify the exhausted owner-approved resource. |
| Execute | `Assigned` to `Running`; bind immutable input digests and execution version to that instance. Persist acceptance before acknowledging; replay identical receipts, reject conflicting identity reuse. |
| Renew | Extend the worker-enforced ticket for the same `Assigned`/`Running` attempt; not a new scheduler state. |
| Cancel/expire/fail | Keep existing terminal states, but retain physical tickets until worker-confirmed completion or teardown. A timeout does not prove nonexecution. Fence stale epochs and reconcile abandoned originals after duplicate validation. |
| Result/commit | `Returned` to `Validated` to `Assimilated` through existing quorum/settlement. Release scratch on actual execution completion; retain output handles until consumers acknowledge or their leases expire. |

Durable deduplication must precede non-idempotent retries on section 9.3 transports.
Use negotiated versions, bounded frames, monotonic sequences and explicit credit
denials; do not replace KDM2 framing or equate a transport request ID with an
execution receipt. Reserve separate bounded control capacity so bulk transfers
cannot starve heartbeats and cancellation. Before untrusted workers, require
authenticated identities, executor allowlists, per-peer quotas and cryptographic
artifact integrity; current FNV content labels are not a malicious-peer guarantee.

#### Data plane and adapters

Extend section 6.2 `ReplicaCatalog` / `StagingJob` placement and transfer scheduling
with immutable prerequisite output handles, consumer references and bounded direct
transfers; keep inline bytes for small messages. Share byte windows for demand
and prefetch rather than adding another transfer credit manager.

KIP/1 (section 9.20 and its [guide](guides/interdevice-protocol.md)) owns bounded
async I/O, Windows IoRing/Linux io_uring capability detection, pinned buffers and
DMA completion/cancellation lifetimes. Build user-space host/device adapters on
that contract and vendor-supported executor APIs first, then evaluate same-host
shared memory or verified RDMA using measured topology. A USB-C connector does not
establish RDMA or coherent remote VRAM. Consider a kernel component only if
profiling proves a missing user-space capability, with a narrow ABI, signed
installation, least privilege, bounded mappings and unplug/suspend recovery review.
No custom kernel driver is implemented.

#### Delivery gates

1. **Admission (implemented, opt-in):** push/pull reservations, restored state,
   invalid-value rejection, decimal exact fits, quorum-pending scratch release
   and queue-aware eligibility diagnostics. No starvation prevention yet.
2. **Shared worker tickets (target):** integrate residency/offer authority before
   enabling producers. Fault-test lost acceptance, restart, duplicate execution,
   expiry during kernels and cancellation before reclamation. Add explicit
   fairness/headroom policy for large units rather than silently lowering quality.
3. **Bounded direct data plane (target):** fault-test fan-out, slow consumers,
   resumable checksum verification and memory pressure against approved windows.
4. **Measured tuning/adapters (target):** compare serial, local-threaded and
   distributed runs on identical inputs. Record p50/p95 end-to-end latency,
   useful work/s, queue wait, hop bytes, peak RAM/VRAM, retries and owner
   responsiveness with hardware/topology provenance. Bucket execution costs by
   shape/backend/precision; benchmark dispatch scan/cloning overhead before
   introducing an indexed reservation cache. Enable optimizations only after
   measured benefits exceed dispatch, staging and validation costs. No speedup
   is claimed and no model precision/context/output default is reduced.
