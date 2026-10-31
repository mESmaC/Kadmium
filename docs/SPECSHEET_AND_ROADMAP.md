# Kadmium (`Cd-48`) — Technical Specsheet & Product Roadmap

Kadmium is an open-source Rust compute-fabric project for heterogeneous personal and lab hardware. The current implementation includes control-plane, TCP transport, planning, and reference-workload mechanisms. Real language-model execution, production distributed training, and managed cloud services remain integration work.

This document uses three labels:

- **Implemented reference:** code executes locally, but may use synthetic tensors, fixed scenarios, or small CPU workloads.
- **Modelled/simulated:** output is calculated from profiles or fixtures, not observed from the represented production system.
- **Planned/target:** design direction without an integrated current execution path.

Roadmap milestones describe code landing, not production readiness. The [API reference](api-reference.md) defines endpoint behavior and the [benchmark ledger](BENCHMARKS_2026-10.md) is the source for measured performance.

---

## Part I — Current technical specification

### 1. System identity and workspace

| Property | Current specification |
| :--- | :--- |
| Identity | Kadmium (`Cd-48`), `kadmium` CLI/crate namespace, `.ktb` tensor banks |
| Implementation | Rust 2024 edition; Python is used only for repository business/release scripts, not as a CLI runtime dependency |
| Host workspace | **12 crates**: core, sharding, data, grid, network, runtime, RL, gateway, CLI, desktop, Condenser, and CUDA |
| CUDA device harness | `crates/kadmium-cuda/device` is excluded from the host workspace and builds separately with `cargo oxide` |
| Toolchains | Host Rust 1.99.0; CUDA device harness Rust 1.99.1 |
| Design tokens | `--cadmium` and `--cherenkov` are visual design tokens, not CLI flags |

### 2. Hardware profiles and execution support

Hardware and OS entries in `kadmium-core::platform` are **classification and planning profiles**. Sharing a silicon or OS family does not automatically provide a working native backend, NPU path, mmap loader, or OS hook.

| Execution layer | Current support |
| :--- | :--- |
| Host reference math | Synthetic CPU/reference kernels and deterministic RepOps comparisons |
| Native CUDA | Six specific kernels in `kadmium-cuda`: `q8k_quantize`, `q4k_q8k_matvec`, `mxfp4_matvec`, `swiglu`, `moe_top_k`, and `row_mask`; see its README for tested hardware/results |
| Other GPU/CPU targets | Backend labels, reference implementations, and planning profiles; no general native ROCm, Metal, Vulkan, AVX-512, NEON, or NPU model backend |
| Model execution | Small reference tensors and fixed diagnostic responses; no integrated pretrained LLM serving |
| OS governance | Limited live sampling plus policy/state-machine models; native swapchain/foreground integration is not universal |
| File loading | Whole-file local import and zero-copy slices into loaded byte buffers; mmap/io_uring/IoRing file paths are planned |

The planner models accelerator memory, unified memory, host RAM, local storage, and peer/network placement. Some design documents split those concepts into five or six tiers for different purposes; neither count is a hardware compatibility guarantee. Bandwidth values in profiles are inputs to models, not measured Kadmium throughput.

### 3. Core mechanisms and exact boundaries

| Subsystem | Current mechanism | Qualification |
| :--- | :--- | :--- |
| $q^*$ placement heuristic | `HardwareNode::compute_q_star_ratio` | Unified memory returns `0.96`; no VRAM returns `0.0`; discrete GPUs use polite-adjusted FP16 compute, CPU, interconnect, and NVMe terms clamped to `0.10..=0.95`. It is not a RAM/VRAM bandwidth formula. |
| Asymmetric sharding | `ShardPlanner` and `MinMaxStageLatencyOptimizer` | Produces layer/expert placements and modelled latency estimates, not a loaded-model performance guarantee. |
| KV planning | `DistributedPagedKvPlanner` | Plans 16-token pages across memory profiles; capacities are modelled. |
| `.ktb` v1 | `KtbBankPacker`, `KtbBankReader`, `KtbMerkleTree` | `KDTB` magic, strict 65,536-byte alignment, FNV-1a 64-bit checks and Merkle proofs. FNV is not collision-resistant authentication. |
| Local import | SafeTensors/GGUF transcoders and `kadmium import` | One local file; GGUF support is restricted to compact zero-metadata streams. `compile` creates synthetic tensors. |
| Attention reference | `PagedGqaAttentionKernel` | Two-pass safe softmax over scores, not FlashAttention-2 online rescaling. |
| Training | 1F1B scheduler, stateless stage previews, `DiLoCoSession`, `DeMoDctCompressor` | Only the small session/CLI path maintains outer state for complete caller-supplied replicas. There is no integrated pipeline-sharded DiLoCo learner. DeMo ratio depends on tensor size and `top_k`; convergence is not guaranteed. |
| RL and replay | `GrpoEpochExecutor`, `WasiMicroSandbox`, `DreamRsiSimulator` | Fixed synthetic polynomial programs, a custom fuel-metered stack machine, and in-memory tree replay. Not general WASI or online model training. |
| Verification | RepOps deterministic reductions and Verde digest/bisection references | Validates defined reference computations; FNV digests do not provide cryptographic settlement or privacy. |
| Polite governance | `PoliteActivityMonitor` and OS-hook policy models | Sub-200 ms is a policy target/estimate, not an observed cross-platform SLA. |
| Trust placement | `NodeTrustTier` | Constrains ingress/egress and middle-layer placement; it is not authentication or confidential computing. |

Precise throughput, latency, memory-saving, convergence, failover, and cost figures emitted by demonstration commands are scenario outputs unless the benchmark ledger records them as measurements. They must not be presented as guarantees.

### 4. Protocols, APIs, and integrations

| Interface | Current behavior |
| :--- | :--- |
| Worker wire | Emits **KDM2** binary frames over TCP; legacy **KDMF** frames are accepted for decode compatibility. `KDMW` is not the current frame format. |
| Join metadata | `kdm1_...` endpoint descriptors with unkeyed FNV-1a integrity checks, not HMAC-SHA256 signatures or authentication. Registration currently ignores ticket authorization fields. |
| LAN discovery | UDP `KDMD` announcements and direct reachable-address worker registration. |
| NAT traversal | Route planning plus a loopback UDP self-test. STUN/DERP endpoints are descriptors; managed QUIC/relay traversal is planned. |
| Reference AI API | `/v1/chat/completions` and `/v1/messages` return fixed diagnostic text; `/v1/models` lists planning presets; `/v1/embeddings` returns reference vectors. |
| MCP | Seven tools, five resources, and two prompts over isolated stdio or shared `POST /mcp` coordinator state. |
| Ecosystem | HF selective-pull planning, in-memory range assembly, and simulated GitHub evaluation. No live HF/GitHub/OCI/S3/W&B connector. |
| Cloud and supply | Fixed spot/failover scenarios and local lease-policy quotes. No provider provisioning, paid settlement, managed failover, or enforced cloud spend cap. |
| Observability | Prometheus/OpenMetrics text and local W&B/OTel-shaped summary JSON; external export is not implied. |
| USB/interconnect | Interface discovery and topology/scheduling models. Throughput/latency values derived from rated bandwidth are estimates, not DMA benchmarks or unplug/failover guarantees. |

---

## Part II — Engineering roadmap

### Reference foundations landed (historical phases 1–4)

The repository contains the following implemented mechanisms. “Landed” does not mean production-ready or integrated with real models.

- **Core and data:** capability profiles, $q^*$ planning, asymmetric placement, 64 KiB `.ktb` banks, constrained SafeTensors/GGUF import, Merkle/checksum verification, and EXL3/QTIP reference models.
- **Runtime:** host reference kernels, TCP workers, stateless training previews, small stateful DiLoCo sessions, DeMo compression, RepOps/Verde references, grid work graphs, and synthetic decode/training/RL workloads.
- **Interfaces:** Control Room, Companion, Condenser, reference `/v1` diagnostics, MCP, CLI planners, and platform-specific probe/bridge commands.
- **Planning/simulation:** subcluster partitioning, responsibility shifts, spot/failover scenarios, supply/credit calculations, NAT route planning, USB topology, and sample governance catalogs.
- **Native device work:** six CUDA kernels with a separate parity harness and the hardware/results caveats documented in `crates/kadmium-cuda/README.md`.

### Phase 5 — Native backends and real model integration (next)

- [ ] Integrate pretrained model loading and incremental generation; replace fixed `/v1` diagnostics only through an explicit, reviewed backend design.
- [ ] Add general native CUDA/ROCm/Metal/Vulkan execution beyond reference host buffers and the six current CUDA kernels.
- [ ] Implement persistent model/optimizer state and integrated pipeline-sharded training; validate convergence separately from compression ratios.
- [ ] Implement actual mmap/io_uring/IoRing paths where supported and measure them on committed hardware configurations.
- [ ] Replace modelled interconnect and OS-yield outputs with measured execution and native platform hooks.

### Phase 6 — Managed services and production hardening (planned)

- [ ] Authenticated multi-tenancy, enforced RBAC/quotas, secure worker enrollment, and production network access controls.
- [ ] Managed WAN relay/NAT traversal and production remote worker lifecycle.
- [ ] Real cloud-provider order books, procurement, spend enforcement, billing, and measured CloudShield failover.
- [ ] Actual Supply Grid leasing, credit/USD settlement, abuse controls, and privacy/confidential-computing design.
- [ ] Real HF/GitHub/OCI/S3/W&B connectors and public registry services.
- [ ] Release-gate documentation, licensing artifacts, rollback verification, and reproducible performance evidence for every published claim.

### Research algorithm modules

The runtime also contains schedulers and reference implementations for RAID/parity striping, speculative decoding, chunked prefill, zero-bubble training, GaLore/Muon, FP8/Hadamard transforms, context parallelism, MoE routing, KV quantization, hierarchical pools, and decentralized storage. Their presence establishes an implementation or model to test—not the synthetic speedups, savings, or resilience figures printed by sample scenarios. Each requires its own integrated execution path and measured validation before promotion to a product guarantee.
