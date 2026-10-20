# Guide: Ecosystem Integrations (`hf://`, `gh://`, `oci://`, `s3://`, `wandb://`)

Kadmium includes first-class URI connectors in [`kadmium-data`](../../crates/kadmium-data/src/lib.rs) so you can pull models, datasets, and RL/Dream-RSI environments directly from **Hugging Face Hub**, **GitHub**, **OCI Container Registries**, and **S3/R2 Object Storage** without manual downloads or glue scripts.

---

## 1. Hugging Face Hub Integration (`hf://`)

### A. Layer-Aware Selective Model Pulling (`hf://org/model@revision`)
When you run a frontier model like `hf://deepseek-ai/DeepSeek-V3@main` across a heterogeneous cluster, no single consumer machine needs the entire 351 GB Safetensors repository in VRAM/RAM:
1. [`HuggingFaceHubConnector::plan_selective_model_pull`](../../crates/kadmium-data/src/lib.rs) resolves `model.safetensors.index.json` on your Seed / Home NAS node (`truenas-core-vault`).
2. Each worker's assigned layer range (`start_layer..=end_layer`) is mapped to the exact subset of `.safetensors` shard files (e.g., `model-00006-of-00016.safetensors`..`model-00010-of-00016.safetensors`).
3. Those shards are transcoded on-the-fly into `64 KiB` page-aligned `.ktb` banks (`/var/kadmium/ktb/deepseek-ai__DeepSeek-V3.L23-L37.ktb`) and staged peer-to-peer via CERN FTS3.

```bash
kadmium pull hf://deepseek-ai/DeepSeek-V3@main
```

### B. Streaming Parquet & JSONL Datasets (`hf://datasets/org/name`)
For Sharded-Island DiLoCo training and GRPO reinforcement learning, point Kadmium at any Hugging Face dataset:
```bash
kadmium pull hf://datasets/open-r1/OpenR1-Math-220k@main
```
Dataset splits stream directly to `TrainingDataLoaderAndAugmenter` and `RlSwarmRewardVerifierSandbox` nodes (`rk3588-sbc-bramble`, `legacy-thinkpad-t480`, `truenas-core-vault`).

### C. Pushing Fine-Tuned DiLoCo / LoRA Checkpoints
After an outer-step Nesterov sync (`DiLoCoOuterOptimizer`), export merged or LoRA adapter weights (`adapter_model.safetensors`) back to Hugging Face Hub.

---

## 2. GitHub & GitHub Actions Integration (`gh://`)

### A. Commit-Pinned RL & Dream-RSI Environment Suites (`gh://owner/repo@ref#subpath`)
To guarantee 100% deterministic reward evaluation across heterogeneous CPUs, SBCs, and GPUs, [`GitHubConnector`](../../crates/kadmium-data/src/lib.rs) mounts RL verifier suites and [**Dream-RSI**](https://www.dream-rsi.com/) evolving-world tasks by their exact Git commit SHA:

```bash
kadmium pull gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt
```

### B. Sub-Cluster Slice-on-PR Webhooks (`GitHubSliceCiRun`)
Link a GitHub repository to an isolated Kadmium Sub-Cluster Slice (e.g., `slice-02-staging-test` or `slice-04-rl-dream-rsi`):
- Whenever a Pull Request is opened or updated, Kadmium runs the target prompt/kernel benchmark across the slice, verifies deterministic execution via `Verde` $O(\log_2 L)$ refereed bisection, and reports token throughput (`tok/s`) and Dream-RSI speedup (`2.31×`) back as a GitHub Check Run.

---

## 3. Additional Ecosystem Connectors

| Scheme | Provider | Primary Use in Kadmium |
| :--- | :--- | :--- |
| **`hf://`** | **Hugging Face Hub** | Selective Safetensors/GGUF pulling $\to$ `.ktb` transcoding, dataset streaming, and LoRA adapter export. |
| **`gh://`** | **GitHub & Actions** | Commit-pinned RL/Dream-RSI environment mounting and automated Slice-on-PR evaluation check runs. |
| **`oci://`** | **GHCR / Docker OCI** | Rootless container bridge (`oci://ghcr.io/kadmium-ai/gym-cuda-sandbox:latest`) alongside embedded WASI micro-sandboxes. |
| **`s3://`** | **S3 / Cloudflare R2 / MinIO** | Trickle-syncs encrypted `.ktb` delta chunks from your Home NAS to a warm cloud vault for `<400ms` **CloudShield** failover. |
| **`wandb://`** | **Weights & Biases / OTel** | Streams DiLoCo outer-step loss, `DeMo` DCT compression ratios (`142×`), and Dream-RSI discovery trees. |
