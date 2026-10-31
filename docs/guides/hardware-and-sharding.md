# Guide: Hardware Setup, Asymmetric Sharding & Polite Host Governance

Kadmium profiles heterogeneous machines and computes placement and governance plans. Native model-related execution uses small reference tensors; hardware classification does not imply that every native backend or model-serving path is implemented. Condenser's [local checkpoint workflow](local-checkpoints.md) provides generation through a separate local backend; [distributed compute](distributed-compute.md) provides supplied FP32 matrix jobs and learner integration guidance.

---

## 1. Hardware profiles and planning roles

Nodes are represented with fast accelerator memory, unified memory, system RAM, NVMe, and peer/network tiers. The planner can assign different layer and expert ranges according to those profiles.

| Device class | Example | Planning role |
| :--- | :--- | :--- |
| Gaming desktop | RTX 4090, DDR5, NVMe | Compute-heavy layers and trusted ingress |
| Apple Silicon Mac | M4 Pro unified memory | High-capacity contiguous stages; unified-memory profiles use $q^*=0.96$ |
| Gaming laptop | RTX 4070 Mobile | Middle stages with battery/thermal governance |
| NAS | TrueNAS/Linux ZFS | Tensor-bank storage and coordinator standby metadata |
| Legacy laptop | x86_64 CPU | Lightweight reference work or sandbox planning |
| ARM/RISC-V SBC | RK3588, CM4, K230 | Utility, replay, and verification planning |
| Remote peer | Reachable TCP worker | Middle-layer placement according to trust policy |

Native device execution currently covers a specific set of CUDA kernels and one Intel Level Zero NPU MatMul with explicitly approximate FP16 arithmetic. Other backends primarily use host reference math and platform classification. See the [support and execution boundaries](../SPECSHEET_AND_ROADMAP.md#2-hardware-profiles-and-execution-support) and [NPU crate guide](../../crates/kadmium-npu/README.md).

## 2. Local `.ktb` ingestion

For an actual local model, prefer `kadmium checkpoint normalize <PATH>` or Condenser's **Models** view. This lossless package path supports GGUF metadata and complete sharded checkpoints, retains originals and tokenizer/config files, and records SHA-256 hashes. See [local checkpoints](local-checkpoints.md). It does not make KTB banks a native inference runtime.

`kadmium import --path <FILE>` is a constrained **local single-file reference importer** for SafeTensors and GGUF v3. The GGUF parser currently accepts only compact streams with zero metadata key-value pairs, which excludes many standard GGUF models. Omitting `--path` synthesizes a reference input file.

```bash
kadmium import --model qwen2.5-coder-7b --path ./model.safetensors --start-layer 0 --end-layer 3
kadmium inspect <PATH_PRINTED_BY_IMPORT>
```

The importer reads the whole source file and writes 64 KiB-aligned `.ktb` pages with FNV-1a checksums and Merkle proofs. `kadmium compile` is different: it generates deterministic synthetic tensor payloads for a requested layer range; it does not transcode a named pretrained model.

Zero-copy slices into already loaded byte buffers are implemented. Actual memory-mapped file loading, `io_uring`, and DirectStorage paths are planned, not established by the current importer.

## 3. Transfer planning and live reference streaming

[`Fts3ChunkSlicer`](../../crates/kadmium-data/src/lib.rs) exposes `compute_chunk_manifest` and `diff_missing_chunks`. Callers select a chunk size, clamped to at least 64 KiB, and compare FNV-1a chunk hashes to identify missing data.

The current live worker path synthesizes a reference bank, after which the coordinator sends pages to workers over TCP and verifies patch/digest behavior. It does not order a NAS to stream directly to a worker and does not establish third-party QUIC/FTS3 network transfers. Direct peer staging, production retry behavior, and mmap-based caches remain target work.

## 4. Polite host governance

The policy model includes:

- `IdleDedicated`: full planning capacity.
- `LightDesktopUse`: **75%** compute multiplier; it does not reserve a fixed 25% of VRAM.
- `GamingOrHeavyApp`: reduced background budget and a **target** sub-200 ms yield policy.
- `BatteryOrThermalThrottle`: reduced compute and VRAM budgets for battery/thermal pressure.

Live sampling and hysteresis are limited; native foreground/swapchain drivers are not universally implemented. The displayed yield latency is a policy estimate, not a measured responsiveness or 120 Hz guarantee. Responsibility-shift reports are planner scenarios, not evidence that an active model migrated without interruption.

## 5. Trust tiers and placement boundaries

[`NodeTrustTier`](../../crates/kadmium-core/src/lib.rs) supplies placement constraints:

1. `TrustedCore` may hold layer 0 and layer $N-1$.
2. `VerifiedPeerMiddleLayersOnly` is restricted to middle layers.
3. `UntrustedAuditedMiddleLayersOnly` is restricted to middle layers and selected for audit-oriented verification.

These are topological planning rules, not cryptographic privacy or authentication. Intermediate activations can still reveal information. FNV checksums and RepOps digest comparisons provide deterministic integrity/reference checks, not confidential computing or a settlement proof. Protect the coordinator with external network access controls.
