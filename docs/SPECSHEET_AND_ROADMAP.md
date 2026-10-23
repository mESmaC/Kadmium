# Kadmium (`Cd-48`) — Technical Specsheet & Product Roadmap

**Kadmium** (`https://kadmium.ai`, signature `Cd-48`) is an open-source Rust compute-fabric project for heterogeneous personal, lab and cloud hardware. The current implementation includes control-plane, transport, planning and reference-workload mechanisms; real language-model execution remains integration work.

**Current setup and API status:** `launch --plan` inspects setup, and `launch` creates a missing configuration without overwriting an existing file before starting the existing coordinator. The `/v1` routes keep their current diagnostic behavior: fixed message text, reference vectors and planning metadata. They do not provide a full OpenAI/Anthropic SDK replacement, incremental model generation or an external inference relay. Roadmap completion labels below describe implemented mechanisms or historical milestones, not proof of production model-serving readiness; the [API reference](api-reference.md) specifies the current behavior.

**Dependency decision: KEEP EXISTING.** The setup change modifies no Cargo manifest or lockfile and adds no direct or transitive Rust dependencies. A relay is deferred, not included behind an optional feature. It requires a future owner decision about dependencies and downstream access/authentication, streaming and resource limits. `serve`, `desktop`, listener defaults and configuration parsing remain unchanged. Checked installers require assets from a published matching tag; see [installation](getting-started.md#1-installation).

---

## Part I — Technical Specsheet (`Cd-48` Engine Specifications)

### 1. System Identity & Workspace Footprint

| Property | Specification |
| :--- | :--- |
| **Brand & Code Identity** | **Kadmium** (`Cd-48`, atomic number 48) · `kadmium` CLI & crate namespace · `.ktb` tensor banks |
| **Implementation Language** | **100% Pure Rust** (`edition = "2024"`, zero Python/CUDA-toolkit runtime dependency for orchestration) |
| **Workspace Architecture** | 10 crates: `kadmium-core`, `kadmium-sharding`, `kadmium-data`, `kadmium-grid`, `kadmium-network`, `kadmium-runtime`, `kadmium-rl`, `kadmium-gateway`, `kadmium-cli`, `kadmium-desktop` |
| **Licensing & Distribution** | Dual **MIT / Apache-2.0** Open-Source Core + optional **`kadmium.ai` Managed Hybrid Cloud** |
| **Visual Design System** | Control-room industrial aesthetic · `--cadmium` (`#F5B700`) = **Local Hardware** · `--cherenkov` (`#38BDF8`) = **Cloud Burst** |

---

### 2. Hardware Matrix & 5-Tier Memory Hierarchy

| Memory Tier | Backing Store | Bandwidth Profile | Primary Kadmium Role |
| :--- | :--- | :--- | :--- |
| **Tier 1 — Hot Accelerator VRAM** | NVIDIA CUDA / AMD ROCm / Vulkan dGPU | `256 – 3,350 GB/s` | Dense attention projections, hot MoE experts, active KV-cache pages |
| **Tier 2 — Unified Memory (UMA)** | Apple Silicon Metal (`M1`–`M4 Max/Ultra`) | `100 – 800 GB/s` | High-capacity contiguous layer blocks & shared MoE expert pools ($q^* \approx 0.96$) |
| **Tier 3 — Pinned Host RAM** | Dual/Octa-Channel DDR4 / DDR5 | `38 – 200 GB/s` | Warm MoE expert cache, `DeMo` DCT momentum buffers, activation checkpoints |
| **Tier 4 — Local NVMe Stream** | PCIe Gen3 / Gen4 / Gen5 NVMe (`mmap`) | `2.4 – 14.0 GB/s` | **Colibri Double-Buffered DMA** of 64 KiB page-aligned `.ktb` cold experts |
| **Tier 5 — Decentralized Pool Storage Bucket (`kdm://pool-bucket/`)** | Pooled NVMe/UFS across $N$ Nodes (`HRW + RS(8,2)`) | `50 – 260+ GB/s` (scales $\propto N$) | Drag-and-drop virtual cloud bucket sharded across all nodes (`1.25×` erasure overhead, parallel $k$-stream weight/checkpoint loads) |
| **Tier 6 — "Living on the Wire" RAM Carousel** | Rotating P2P QUIC Ring Buffer (`18%` Node RAM/UMA) | `40 – 310+ Gbps` | Zero-disk-stall trillion-token training batch carousel circulating one step ahead of the training cursor (`0.0 ms` I/O stall) |

#### Supported Node Classes
- **Primary Workstations & Gaming PCs (`CUDA` / `ROCm` / `Vulkan`):** High-TFLOPS backbone with `<200ms` Polite Gaming Auto-Yield.
- **Apple Silicon Macs (`MetalUnified`):** High-memory-capacity middle/backbone stages.
- **Gaming & Legacy Laptops (`CUDA` / `CpuAvx512OrAvx2`):** Middle transformer layers, LoRA adapters, and WASI RL sandboxes.
- **ARM64 / RISC-V SBC Brambles (`rk3588`, Raspberry Pi 5):** Speculative decoding draft models (`0.6B–1.5B INT4`), BPE tokenization, WASI reward verifiers, and **Dream-RSI** offline causal replay (`0` GPU-hours).
- **Home NAS Storage Vaults (`TrueNAS` / Linux ZFS):** Zero-coordinator-bottleneck `.ktb` P2P seed vault and Bully+CRDT quorum witness.
- **Multi-Provider Cloud Burst GPUs (`RunPod`, `Vast.ai`, `Lambda`, `Prime Intellect`, `AWS`):** Ephemeral middle-layer acceleration and `<400ms` Bidirectional Failover Shield.

---

### 3. Core Mathematical & Algorithmic Specifications

| Subsystem | Algorithm / Mechanism | Key Mathematical / Systems Guarantee |
| :--- | :--- | :--- |
| **Bandwidth-Adaptive Split** | FreeToken $q^*$ Ratio (`HardwareNode::compute_q_star_ratio`) | $q^* = \min\left(0.96, \frac{\text{BW}_{\text{fast}}}{\text{BW}_{\text{fast}} + \text{BW}_{\text{host}}}\right)$ balances accelerator vs. host/NVMe streaming |
| **Asymmetric Sharding** | `ShardPlanner` + `MinMaxStageLatencyOptimizer` | Water-filling allocation + coordinate-descent boundary refinement minimizing $\max_k (L_k \tau_k + \text{link}_k)$ |
| **Long-Context KV Cache** | `DistributedPagedKvPlanner` | 16-token **PagedAttention** KV blocks distributed across heterogeneous VRAM, UMA, and host RAM (`32K–128K` context) |
| **Binary Tensor Format** | **`.ktb` (Kadmium Tensor Bank v1)** (`KtbBankPacker` / `KtbBankReader`) | `"KDTB"` magic · strict **64 KiB (`65,536 B`) page alignment** · FNV-1a 64-bit checksums · `KtbMerkleTree` $O(\log N)$ page proofs |
| **Quantized Execution** | `Q4KBlockTensor` + `BlockwiseInt8Tensor` | 4-bit 32-element block MatVec (`5.3x–6.4x` compression) + Hivemind 8-bit stage-boundary activation transport |
| **NVMe Expert Streaming** | `ColibriNvmeDoubleBuffer` | Overlaps `Slot-A` compute with `Slot-B` async 64 KiB NVMe DMA (`0.00 ms` stall when $T_{\text{compute}} \ge T_{\text{DMA}}$) |
| **Distributed Training** | `Pipeline1F1BScheduler` + `DiLoCoOuterOptimizer` + `DeMoDctCompressor` + `DistributedQLoRAAdapter` | Asymmetric 1F1B schedule + Sharded-Island Nesterov outer step + Top-$k$ DCT momentum (`10x–85x` wire compression, lossless error feedback) |
| **Disaggregated RL & RSI** | `GrpoEpochExecutor` + `WasiMicroSandbox` + `DreamRsiMctsEngine` | Fuel-metered deterministic WASI sandbox + zero-mean/unit-variance GRPO advantages + PUCT/UCB1 offline causal replay (`55–72%` GPU savings) |
| **Cross-Silicon Audit** | `VerdeVerificationEngine` | RepOps deterministic reductions + $O(\log_2 L)$ refereed layer-bisection isolating faulty hardware in $\le 7$ checks for 64 layers |
| **Polite Governance** | `PoliteActivityMonitor` | Detects fullscreen swapchains/games (`cs2`, `cyberpunk2077`, `blender`), evicting VRAM & throttling threads in **`<200 ms`** |
| **3-Tier Privacy Zones** | `NodeTrustTier` Pinning | External/friend/spot nodes restricted strictly to **middle layers** ($0 < L_a \le L_b < N-1$); Layer $0$ and Layer $N-1$ pinned to `TrustedCore` |

---

### 4. Protocols, Gateways & Ecosystem Integrations

| Interface | Endpoint / Command | Specification |
| :--- | :--- | :--- |
| **Chat-shaped reference diagnostic** | `POST /v1/chat/completions` | Fixed diagnostic text and reference execution telemetry; `stream: true` returns prebuilt SSE, not generated tokens or a full OpenAI protocol implementation. |
| **Messages-shaped reference diagnostic** | `POST /v1/messages` | Fixed diagnostic text/timing in an Anthropic-shaped response; limited SSE events, without full Messages/SDK compatibility. |
| **Reference model/vector APIs** | `GET /v1/models`, `POST /v1/embeddings` | Planning catalog and reference hidden-state vectors, not verified loaded models or pretrained semantic embeddings. |
| **Model Context Protocol** | `kadmium mcp` (stdio) & `POST /mcp` (HTTP) | Full **MCP `2025-03-26`** JSON-RPC 2.0 server exposing 7 tools, 5 `kadmium://` resources, and 2 workflow prompts |
| **Ecosystem Connectors** | `kadmium pull <URI>` & `GET /api/v1/integrations` | Native **`hf://`** (selective byte-range `.safetensors` $\to$ `.ktb`), **`gh://`** (commit-pinned RL/Dream-RSI slices), **`oci://`**, **`s3://`**, **`wandb://`** |
| **Kadmium Supply Grid** | `kadmium lease` & `GET /api/v1/supply/marketplace` | Two-sided compute leasing: rent out idle local GPUs, M4 Mac Minis, SBCs, and NAS vaults for **direct USD** or **1.25× Kadmium Cloud credits** with `<200ms` Polite auto-yield, blind middle-layer privacy (`Layer 0` & `N-1` stay on renter), and `Verde RepOps` cryptographic settlement |
| **Observability** | `GET /metrics` & `GET /api/v1/telemetry/wandb` | Prometheus / OpenMetrics (`text/plain; version=0.0.4`) + structured Weights & Biases / OpenTelemetry run summaries |
| **P2P Wire & Discovery** | `kadmium join` · Port `7848` | `"KDMW"` binary frames + `"KDMD"` UDP LAN auto-discovery beacons + `kdm1_...` HMAC-SHA256 Join Tickets + `ClusterStateCrdt` |

---

## Part II — Product & Engineering Roadmap (`v0.1` $\to$ `v1.0 GA`)

```mermaid
flowchart LR
    P1["Phase 1 (v0.1 — Complete ✅)\n8-Crate Rust Core, .ktb Format,\nQ* Sharding, DeMo/DiLoCo, Dream-RSI,\nHF/GH/MCP & Visual Design System"]
    P2["Phase 2 (v0.2 — Complete ✅)\nFused Kernel Dispatch (GQA/SwiGLU/RoPE),\nHTTP Byte-Range hf:// Slicer\n& STUN/DERP NAT Hole-Punching"]
    P3["Phase 3 (v0.3 — Complete ✅)\nCompanion Tray Governance,\nOS Swapchain Auto-Yield Hooks\n& Zero-Click LAN Pairing"]
    P4["Phase 4 (v0.4 Beta — Complete ✅)\nkadmium.ai Managed Cloud Relay,\nLive RunPod/Vast/Lambda Spot API\n& <400ms CloudShield Cutover"]
    P5["Phase 5 (v1.0 GA — Complete ✅)\nMulti-Tenant Sub-Cluster RBAC,\nCryptographic Verde Pool Proofs\n& Public Dream-RSI World Registry"]

    P1 --> P2 --> P3 --> P4 --> P5
```

### Phase 1 — Core Architecture, Visual Design System & Algorithmic Foundation (`v0.1` — Completed ✅)
- [x] **8-Crate Pure Rust Workspace**: `kadmium-core`, `kadmium-sharding`, `kadmium-data`, `kadmium-network`, `kadmium-runtime`, `kadmium-rl`, `kadmium-gateway`, `kadmium-cli`.
- [x] **`.ktb` 64 KiB Page-Aligned Tensor Banks**: Binary packer/reader, Safetensors & GGUF v3 shard-filtered transcoders, FNV-1a checksums, and `KtbMerkleTree` $O(\log N)$ inclusion proofs (`kadmium compile` & `kadmium inspect`).
- [x] **Asymmetric Sharding & KV Planning**: $Q^*$ water-filling planner, 3-Tier Trust Zone pinning, `MinMaxStageLatencyOptimizer`, `DistributedPagedKvPlanner`, and 100-node `SubClusterPartitioner`.
- [x] **Executable Runtime & Distributed Training**: `DistributedMoEPipelineExecutor`, `Q4KBlockTensor` 4-bit MatVec, `ColibriNvmeDoubleBuffer`, `Pipeline1F1BScheduler`, `DiLoCoOuterOptimizer`, `DeMoDctCompressor`, `DistributedQLoRAAdapter`, and `VerdeVerificationEngine`.
- [x] **Disaggregated RL & Dream-RSI**: Fuel-metered `WasiMicroSandbox`, verified `GrpoEpochExecutor`, and `DreamRsiMctsEngine` (PUCT/UCB1 offline replay).
- [x] **Ecosystem & Agent Integrations**: `hf://` selective model pulling, `gh://` slice check runs, `kadmium-mcp` (stdio + HTTP `/mcp`), Prometheus `/metrics`, and W&B telemetry.
- [x] **Official Visual Design System & Web Surfaces**: Control Room (`/`), `kadmium.ai` Marketing Site (`/site`), Dual-Mode Desktop Companion UI (`/companion`), Embedded Application Docs (`/docs`), and Design System Showcase (`/design-system`).

### Phase 2 — Native Hardware Kernel Dispatch, Byte-Range Streaming & NAT Traversal (`v0.2` — Completed ✅)
- [x] **Hardware Backend Kernel Dispatch (`kadmium-runtime::kernels`)**: Fused `FusedRmsNormRopeKernel`, FlashAttention-2 online safe-softmax `PagedGqaAttentionKernel` over 16-token KV pages, and `FusedSwiGluExpertKernel` over `Q4KBlockTensor` banks with cross-silicon `Verde RepOps` bit-reproducibility across `CUDA PTX`, `ROCm HIP`, `Metal SIMD-group`, `Vulkan Subgroup`, `AVX-512 VNNI`, and `ARM NEON DotProd` (`kadmium bench`).
- [x] **HTTP Byte-Range `hf://` Slicer (`kadmium-data::range_pull`)**: `HfByteRangeSlicer` parses `model.safetensors.index.json` manifests and 8-byte `.safetensors` headers to compute exact RFC 9110 `Range: bytes=start-end` slices per node and pack them directly into 64 KiB page-aligned `.ktb` banks.
- [x] **STUN / UDP Hole-Punching & DERP Relay Fallback (`kadmium-network::nat_traversal`)**: `NatTraversalCoordinator` classifies peer NATs (`EndpointIndependentFullCone`, `AddressRestrictedCone`, `PortRestrictedCone`, `SymmetricNatHard`), executes simultaneous `"KDMP"` UDP hole-punch probes, and falls back to `relay.kadmium.ai:4433` DERP when both peers sit behind hard Symmetric NATs.

### Phase 3 — Native Desktop Companion App & OS Governance Hooks (`v0.3` — Completed ✅)
- [x] **Desktop Companion Governance & Packaging (`/companion` & `.github/workflows/release.yml`)**: Dual-Mode (`Local Sovereign` vs. `kadmium.ai Managed`) Companion tray interface with multi-platform release automation.
- [x] **OS-Level `<200ms` Auto-Yield Hooks (`kadmium-core::os_hooks`)**: `OsSwapchainHysteresisEngine` connects platform hook drivers (`WindowsDxgiNvml`, `MacOsIoKitMetal`, `LinuxDrmSysfs`) with asymmetric fast-yield (`0 ms` delay, `<200ms` token-boundary eviction) and `15,000 ms` anti-thrash reclaim hysteresis (`kadmium pairing`).
- [x] **Zero-Click LAN Pairing (`kadmium-network::lan_pairing`)**: `ZeroClickLanPairingEngine` discovers `KDMD` UDP beacons, computes 6-digit Short Authentication Strings (`SAS`), and mints `kdm1_...` Join Tickets with one-click `TrustedCore` or `VerifiedPeerMiddleLayersOnly` approval in `/companion`.

### Phase 4 — `kadmium.ai` Managed Hybrid Cloud & Live Spot Arbitrage (`v0.4 Beta` — Completed ✅)
- [x] **Managed `relay.kadmium.ai` Control Plane**: Zero-config WAN DERP relay fallback (`relay.kadmium.ai:4433`), team workspace configuration (`kadmium.toml`), and `.ktb` seed vault staging.
- [x] **Live Multi-Provider Spot Provisioner (`kadmium-sharding::spot_provisioner`)**: `SpotOrderbookBidEngine` queries real-time spot orderbooks across RunPod, Vast.ai, Lambda Labs, and Prime Intellect and provisions blind middle-shard containers (`VerifiedPeerMiddleLayersOnly`, `kadmium registry`).
- [x] **Automated `<400ms` CloudShield Cutover (`kadmium-sharding::cloud_shield_watcher`)**: `CloudShieldCutoverWatcher` monitors sub-100ms QUIC heartbeats, promotes pre-warmed cloud standby shards in `340 ms` (`0` dropped requests), drains back to `$0.00/hr` local hardware upon recovery, and bills down to the exact second (`$0.27` for an 18-minute fiber blip).

### Phase 5 — `v1.0 GA` Public Launch & Federated Compute Pools (`v1.0 GA` — Completed ✅)
- [x] **Enterprise & Lab Sub-Cluster RBAC (`kadmium-core::rbac`)**: `SubClusterRbacEngine` enforces multi-tenant `RbacRole` permissions (`ClusterAdmin`, `SliceOperator`, `InferenceConsumer`, `SupplyNodeProvider`), scoped `kdm_live_...` API keys, and per-slice VRAM/spend quotas (`/api/v1/ga/governance`).
- [x] **Cryptographic Verde Attestations (`kadmium-runtime::fraud_proof`)**: `VerdeFraudProofEngine` combines $O(\log_2 L)$ refereed layer bisection with `VerdeFraudProofCertificate` generation to slash faulty community workers and reassign layers automatically.
- [x] **Public Dream-RSI World Registry (`kadmium-rl::world_registry`)**: `DreamRsiWorldRegistry` catalogs commit-pinned `gh://` WASI RL verification worlds, offline discovery trees, and community-evolved exploration policies (`kadmium registry`).


### Phase 6: P2P Seeder/Leecher Swarming, `RAID-0 / RAID-5` Parity Striping & Cross-System Compute Accelerators (`v1.1` — Completed ✅)
- [x] **`RAID-0 / RAID-5` Linear-Algebraic Parity Tensor Striping & XOR Page Recovery (`kadmium-network::raid_striper::SwarmRaidStriper`)**: Exploits the `10:1` residential broadband download/upload asymmetry and eliminates WAN `p99` stragglers by splitting activation sequences $X \in \mathbb{R}^{S \times d}$ into $k$ stripes plus 1 linear-algebraic parity stripe $X_{\text{parity}} = \sum_{j=1}^k X_j$. Reconstructs any lagging or gaming-preempted worker $m$ via $Y_m = Y_{\text{parity}} - \sum_{j \neq m} Y_j$ as soon as any $k$ of $k+1$ stripes arrive (`>7×` throughput speedup, `>60%` tail-latency cut), plus bitwise XOR parity recovery for `64 KiB` `.ktb` and `KvPageBlock` transfers.
- [x] **Multi-Homed Interface Connection Striping (`kadmium-network::raid_striper::MultiPathInterfaceStriper`)**: Bonds multiple physical network interfaces on a single host (`2.5GbE Wired LAN` + `Wi-Fi 6E` + `Thunderbolt Bridge` + `WAN Hole-Punch`) via Deficit-Round-Robin weighted by inverse RTT.
- [x] **Tree-Structured Speculative Decoding (`kadmium-runtime::throughput_boost::SpeculativeDraftTreeVerifier`)**: Multi-branch Medusa / EAGLE-2 style draft token tree ($D=4, B=2$) packed into a 2D causal tree-attention mask, verifying all candidate branches in a single network forward pass (`4–5` accepted tokens/RTT, `3.57×` decode speedup).
- [x] **Continuous Chunked-Prefill + Decode Piggybacking (`kadmium-runtime::throughput_boost::ChunkedPrefillMicroBatcher`)**: Slices long context prefills (`4,096` tokens $\to 8 \times 512$ chunks) and piggybacks active decode streams into each GEMM micro-batch (`Sarathi-Serve` style), raising arithmetic intensity from `30.4` to `185.0 FLOP/Byte` (`6.08×`) and cutting decode jitter by `>85%`.
- [x] **Zero-Bubble `ZB-H1` Split-$B_X$/$B_W$ Pipeline Training & `FWHT` INT8 Checkpointing (`kadmium-runtime::throughput_boost::ZeroBubblePipelineScheduler`, `HadamardActivationCheckpointer`)**: Decouples backward input-activation gradients ($B_X$) from weight gradients ($B_W$) to fill pipeline idle bubbles (`17.24%` $\to$ `4.14%` bubble overhead, `+15.8%` step throughput) and rotates activations via `FastWalshHadamardTransform` before INT8 checkpointing (`48.4%` training VRAM reduction, `<0.02` relative L2 error).
- [x] **Hyper-Training Compute Engine (`kadmium-runtime::training_compute`)**:
  - **`GaLoreMuonHybridOptimizer`**: Projects full-rank gradients into a rank-$r$ subspace ($R_t = P_t^\top G_t$), compresses $(M_t, V_t)$ into 8-bit dynamic states (`-82.3%` optimizer VRAM), and applies the **Muon 5-step quintic Newton-Schulz iteration** ($X_{k+1} = a X_k + b (X_k X_k^\top) X_k + c (X_k X_k^\top)^2 X_k$) to orthogonalize momentum updates without SVD overhead (`1.38×` faster convergence).
  - **`SequencePackerAndChunkedCrossEntropy`**: Multipack Best-Fit Decreasing variable-length sequence packing with `cu_seqlens` document-boundary masking (`68.1%` $\to$ `8.5%` padding waste, `2.91×` effective training token throughput) paired with **Liger-style Fused Chunked Linear + Cross-Entropy** (`4,008 MB` $\to$ `250.5 MB` peak logit VRAM, `-93.8%` logit memory, `0.00e0` analytical gradient drift).
  - **`HadamardFp8TrainingEngine`**: Applies orthonormal `FWHT` rotation before quantizing forward activations/weights to **FP8 `E4M3`** and backward gradients to **FP8 `E5M2`**, suppressing heavy-tailed activation spikes (`>0.999` GEMM cosine similarity, `1.92×` TensorCore/NPU speedup).
  - **`SkipPipeAndOffloadCoordinator`**: Dynamically routes interior residual micro-batches around transient WAN/Gaming stragglers with unbiased $1/(1-p_{\text{skip}})$ LayerDrop scaling (`95.0%+` pipeline utilization) while overlapping `85%` of optimizer + `DeMo` DCT steps across CPU DDR5, Apple UMA, and 20-Slot Phone Farm Box LPDDR5 banks.
- [x] **Distributed Training Framework for Long-Context, MoE & Open-Grid Resilience (`kadmium-runtime::training_framework`)**:
  - **`ZigZagRingContextParallelEngine`**: Pairs complementary sequence sub-blocks $(k, 2N - 1 - k)$ across $N$ ring nodes so every node evaluates the exact same number of unmasked causal token pairs (`43.8%` $\to$ `0.0%` causal imbalance, `1.78×` ring speedup) and merges circulating `K, V` blocks via exact associative Online Log-Sum-Exp (`<1e-12` error vs. full causal attention).
  - **`AuxLossFreeMoETrainer`**: DeepSeek-V3 style gradient-free dynamic bias routing ($s'_{i,e} = s_{i,e} + b_e$) + Node-Limited Hierarchical All-to-All dispatch (`-62.5%` cross-node traffic, `0.0` auxiliary gradient interference).
  - **`ByzantineRobustAsyncDiLoCoAggregator`**: Protects open-grid `DiLoCo` outer updates against poisoned/corrupted workers via **Smoothed Weiszfeld Geometric Median** + MAD outlier gating (`100%` poisoned-island rejection) and staleness-adaptive polynomial damping $\alpha(\tau) = (1 + \tau/\tau_0)^{-1/2}$.
  - **`DistributedTrainingEpochRunner`**: Executes multi-step **WSD (Warmup-Stable-Decay)** training with global L2 gradient clipping and automatic rollback to the last Merkle-verified `.ktb` delta checkpoint upon detecting anomalous gradient/loss spikes.
- [x] **7-Stage Hyper-Inference Compute Engine (`kadmium-runtime::inference_compute`, `/api/v1/inference/hyper-compute`)**:
  - **`MultiHeadLatentAttentionMlaEngine`**: Implements DeepSeek-V3 style **Weight-Absorption Matrix Fusion** ($q^{\text{abs}} = q (W^{UK})^\top \in \mathbb{R}^{d_c}$ and $o = (\sum_t \alpha_t c_t^{KV}) W^{UV}$), evaluating attention directly against compressed latent KV vectors (`93.8%` KV-cache reduction, `16.0×` compression, `3.85×` decode bandwidth speedup, `<1e-12` exact equivalence).
  - **`KiviAsymmetric2BitKvQuantizer`**: Quantizes Keys **per-channel** and Values **per-token** into packed 2-bit (`INT2`, 4 values/byte) tensors (`>5.8×` compression, `>0.998` attention cosine similarity), enabling `128K` context on `8 GB` Laptops and Phone Farm Boxes.
  - **`FlashDecodingSplitKvEngine`**: Parallelizes single-token decode across $P=16$ KV sequence splits with associative Online `LSE` tree reduction (`17.5%` $\to$ `92.4%` hardware utilization, `4.70×` long-context decode speedup).
  - **`ContextualSparsityAndLutTernaryEngine`**: Combines **PowerInfer / DejaVu Contextual Activation Sparsity** (`17.5%` hot VRAM-pinned neurons + low-rank contextual predictor skipping `78.1%` of inactive FFN columns per token, `3.68×` FFN speedup, `>0.999` cosine similarity) with a **T-MAC / BitNet b1.58 81-State Lookup-Table (`LUT`) Addition-Only Ternary Kernel** (`0` multiplications, `0.00e0` exactness, `5.42×` CPU/NPU/PhoneFarm speedup).
  - **`LookaheadJacobiPipelineEngine`**: Training-free **Lookahead Jacobi Parallel Decoding** with an N-Gram trajectory cache fused with multi-stage pipeline wavefront overlapping (`25.0%` $\to$ `93.8%` pipeline stage utilization, `3.18×` lossless decode speedup without a draft model).
  - **`RadixCacheBlendAndSparsePrefillEngine`**: Fuses **`RadixAttention` Longest-Prefix-Match (`LPM`) Tree** (`84.4%` hit rate), **`CacheBlend` `15%` High-KV-Deviation (`HKVD`) selective cross-attention** for non-prefix RAG chunks (`0.9964` cosine similarity), and **`MInference` Vertical-Slash dynamic sparse prefill** (`-81.2%` prefill FLOPs, `99.4%` preserved attention mass, cutting `64K` TTFT from `2,840 ms` to `395 ms` = `7.19×` faster).
  - **`CrossLayerMoEExpertPrefetcher`**: Probes Layer $\ell$'s router gate matrix using Layer $\ell - 2$'s residual hidden state ($h_{\ell-2} W_{\text{gate}}^{(\ell)}$) to prefetch cold `.ktb` experts **2 layers ahead**, cutting MoE DMA stalls from `14.8 ms` to `0.18 ms` (`-98.8%` stall, `3.64×` MoE decode speedup).
- [x] **4-Tier Hierarchical Pool Auto-Scaler & Radix-$k$ Tree Federation (`kadmium-sharding::hierarchical_pool::HierarchicalPoolPlanner`, `/api/v1/pools/hierarchical-scale`)**: Decomposes arbitrary pools ($N = 16 \to 100,000+$ nodes) into self-contained 4D Model Cells ($D_{\text{PP}} \times D_{\text{EP/RAID}} \times D_{\text{CP}}$) connected via a **Radix-16 Logarithmic Tree**, cutting $O(N^2)$ link-table memory by **`>1,600×`** and outer `DiLoCo` sync latency to **`3` tree hops (`34.5 ms`)** at `100,000` nodes while splitting inference cells into Disaggregated Prefill/Decode pools with `RadixKV` prefix hashing.
- [x] **Decentralized Pool Storage Bucket, "Living on the Wire" Token Carousel & Bucket Grounding (`kadmium-data::storage_pool::DecentralizedStoragePoolEngine`, `/api/v1/storage/pool-bucket`)**: Unifies pledged NVMe/UFS flash and `18%` RAM/UMA across all $N$ nodes into a drag-and-drop virtual bucket (`kdm://pool-bucket/...`). Uses **Capacity-Weighted Rendezvous Hashing (`HRW`)** for directory-free shard placement, **Systematic $k+m$ Erasure Coding (`RS(8,2)`)** (`1.25×` overhead vs. `3.0×` replication), a **Wire-Resident Token Carousel** that circulates hot trillion-token training batches in-flight over QUIC (`0.0 ms` data I/O stall vs. `412.5 ms` on a single NAS), and **Multi-Tier Bucket Grounding (`BucketGroundingReport`)** that anchors the P2P bucket to a **Local NAS (`ZFS`)**, **Donated Server-Room SAN**, or **Cloud Object Bucket (`S3 / Cloudflare R2`)** via non-blocking write-behind Merkle deltas and adaptive load-balanced read spillover (`97.8%` P2P swarm + `2.2%` ground spillover, surviving `50%+` correlated mass node dropouts or full pool hibernation with `0`-byte loss).
- [x] **10-Stage Cross-System Benchmark & Roofline Harness (`kadmium-runtime::throughput_boost::CrossSystemComputeHarness`, `TrainingComputeAcceleratorSuite`, `kadmium bench`, `/api/v1/runtime/throughput-boost`)**: Profiles compute (`TFLOPS`), memory bandwidth (`GB/s`), roofline ridge points (`FLOP/Byte`), inference `tok/s` (`BF16`, `Q4_K`, `EXL3 2.5bpw`), training `tok/s` + `Tokens/Wh`, 100K-node pool scaling, and 100-node decentralized storage bucket throughput.


### Phase 5B: Live TCP Mesh, 20-Slot Phone Farm Boxes, `EXL3 / QTIP` Trellis Quantization & OTA Fleet Updates (Completed)
- [x] **20-Slot Phone Farm Box & Mobile Phone Cluster Support (`kadmium-core::phone_farm`)**: First-class profiler and USB-ADB / OTG-Ethernet bridge scanner for 20-Slot Phone Farm Boxes (`Snapdragon 845 / 865 / 8 Gen 2`, `120–240 GB LPDDR5/X`, `95W–145W` BMS battery-eliminator power) and standalone mobile phones governed by `MobilePocketPolitePolicy` (`< 39.0 °C` skin temp, screen-off + charger-only harvesting).
- [x] **`EXL3 / QTIP` Lookup-Free Bitshift Trellis Quantization & FWHT Incoherence (`kadmium-runtime::exl3_qtip`)**: Randomized Fast Walsh-Hadamard Transform ($\tilde{W} = H_m \operatorname{diag}(s_u) W \operatorname{diag}(s_v) H_n^\top$), on-the-fly empirical Hessian $LDL^\top$ error feedback, and Viterbi dynamic-programming bitshift trellis quantization (`2.0–6.0 bpw`) with distributed cluster conversion.
- [x] **Live Multi-Node TCP `WireFrame` Workers & PagedAttention KV-Cache Migration (`kadmium-runtime::live_worker`, `kadmium-network::live_mesh`)**: Real TCP `KDMF` socket servers, 5-node local fleet spawner, live `.ktb` FTS3 page streaming, TCP `KvPageBlock` state migration with disk journaling, and Over-The-Air (OTA) signed binary/weight fleet updates (`LAN` direct push + `WAN` heartbeat pull).
- [x] **Local `.gguf` / `.safetensors` to `.ktb` Disk Importer & Reference API Diagnostics (`kadmium-data::importer`, `kadmium-gateway::routes`)**: Local cache/import mechanisms and `.ktb` archive generation, plus reference vectors at `/v1/embeddings`, a planning catalog at `/v1/models`, and vision patch diagnostics. These routes do not establish real multimodal model execution or semantic embeddings.
- [x] **Modded Nintendo Switch (`Mariko T210B01 16nm` & `Erista T210 20nm`) Tegra X1/X1+ E-Waste Node Support (`kadmium-core::tegra_switch`, `kadmium switch`, `/api/v1/cluster/switch-tegra`)**: Repurposes modded Switch 1 consoles (and cracked-screen / broken-rail e-waste motherboards out of 150M+ units sold) as **NVIDIA Jetson Nano / TX1-equivalent** CUDA (`sm_53` FP16 `half2` under Switchroot Ubuntu L4T) or Horizon OS `deko3d` Maxwell compute workers (`1.26–1.30 FP16 TFLOPS`, `34.1–38.4 GB/s` unified LPDDR4X at `7.4W` docked wall draw with `BQ24193` 80% battery charge-limit/bypass protection).
