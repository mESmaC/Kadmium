# Guide: Ecosystem Integration Planners (`hf://` and `gh://`)

Kadmium's current ecosystem support is a **planning and simulation layer** in [`kadmium-data`](../../crates/kadmium-data/src/lib.rs). It parses Hugging Face and GitHub URIs and produces example transfer or evaluation plans. It does not currently contact Hugging Face, GitHub, container registries, object storage, or Weights & Biases.

---

## 1. Hugging Face selective-pull planning (`hf://`)

```bash
kadmium pull hf://deepseek-ai/DeepSeek-V3@main
```

For an `hf://org/model@revision` URI, [`HuggingFaceHubConnector::plan_selective_model_pull`](../../crates/kadmium-data/src/lib.rs) constructs an illustrative layer-to-shard mapping, output paths, and transfer sizes. The lower-level range-pull code can parse manifests and assemble **caller-supplied in-memory bytes** into `.ktb` banks.

The command does **not** resolve a repository, fetch an index, download shards, stream datasets, mount a Hub repository, or push checkpoints. The example filenames and sizes in its output are not proof that those remote objects exist.

For local files, use the constrained [`kadmium import`](../cli-reference.md#kadmium-import) reference importer instead. It is separate from `pull` and does not accept a Hugging Face repository URI.

## 2. GitHub slice evaluation simulation (`gh://`)

```bash
kadmium pull gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt
```

For a `gh://owner/repo@ref#subpath` URI, the connector returns a simulated slice evaluation. Its commit SHA, throughput, speedup, and check status are fixed demonstration values. Kadmium does not currently clone or mount the repository, resolve the requested ref, receive pull-request webhooks, or create GitHub Check Runs.

## 3. Integration overview and telemetry

`GET /api/v1/integrations` returns sample integration descriptors. `GET /api/v1/telemetry/wandb` returns a structured local run summary; it does not send data to Weights & Biases or OpenTelemetry.

The following schemes may appear in sample data or target designs, but do not have working network connectors in this repository:

| Scheme | Target use | Current status |
| :--- | :--- | :--- |
| `hf://` | Selective model/dataset retrieval and checkpoint export | URI parsing and pull planning only |
| `gh://` | Commit-pinned worlds and CI evaluations | Simulated evaluation only |
| `oci://` | OCI sandbox/container artifacts | Descriptor only; no CLI connector |
| `s3://` | S3/R2/MinIO checkpoint or vault synchronization | Descriptor only; no CLI connector |
| `wandb://` | External experiment tracking | Local summary JSON only |

Real downloads, authentication, retries, remote mounts, CI webhooks, checkpoint pushes, and managed cloud synchronization remain integration work. Do not use the current output as evidence that a remote operation succeeded.
