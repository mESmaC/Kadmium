# Guide: Ecosystem Integration Planners (`hf://` and `gh://`)

Kadmium's current ecosystem support is a **planning and simulation layer** in [`kadmium-data`](../../crates/kadmium-data/src/lib.rs). It parses Hugging Face and GitHub URIs and produces example transfer or evaluation plans. It does not currently contact Hugging Face, GitHub, container registries, object storage, or Weights & Biases.

---

## 1. Hugging Face selective-pull planning (`hf://`)

```bash
# Selective model pull plan (with --index <path/to/model.safetensors.index.json> or --index-manifest)
kadmium pull hf://deepseek-ai/DeepSeek-V3@main --index /path/to/model.safetensors.index.json

# Dataset metadata pull plan (plans dataset_infos.json without fabricating .safetensors, .ktb, or split filenames)
kadmium pull hf://datasets/open-r1/OpenR1-Math-220k@main
```

For an `hf://org/model@revision` URI, [`HuggingFaceHubConnector::plan_selective_model_pull`](../../crates/kadmium-data/src/lib.rs) constructs a selective pull plan labeled with `plan_only: true`, `files_downloaded: 0`, and `ktb_banks_compiled: 0`. When `--index` (alias `--index-manifest`) is supplied (or [`HuggingFaceHubConnector::plan_selective_model_pull_with_index`](../../crates/kadmium-data/src/lib.rs) is called with a parsed `SafetensorsIndexManifest`), `required_safetensors_files` and byte sizes are derived from the parsed `model.safetensors.index.json` (`index_source: "parsed_safetensors_index"`); otherwise `required_safetensors_files` is left empty and the plan is explicitly labeled `index_source: "unverified_missing_index_manifest"`. For `hf://datasets/...` URIs, [`HuggingFaceHubConnector::plan_dataset_pull`](../../crates/kadmium-data/src/lib.rs) resolves `dataset_infos.json` (and [`HuggingFaceHubConnector::plan_dataset_pull_with_splits`](../../crates/kadmium-data/src/lib.rs) resolves caller-supplied split filenames) without fabricating `.safetensors` shards, `.ktb` weight banks, or unverified split names.

For local files, use the constrained [`kadmium import`](../cli-reference.md#kadmium-import) reference importer instead. It is separate from `pull` and does not accept a Hugging Face repository URI.

## 2. GitHub slice staging and dry-run planning (`gh://`)

```bash
# Stage and verify a real local Git repository checkout:
kadmium pull gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt --local-path /path/to/Dream-RSI

# Inspect a dry-run slice plan without mounting a local repository:
kadmium pull gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt --plan
```

For a `gh://owner/repo@ref#subpath` URI, `kadmium pull` requires a valid `--local-path <dir>` (via [`GitHubConnector::stage_local_repo`](../../crates/kadmium-data/src/lib.rs)) and verifies that `<dir>/.git` exists, its `HEAD` matches `@ref` (and any configured remote URL matches `owner/repo`), and `#subpath` exists on disk before reporting `mounted: true` and recording the resolved commit SHA in `pinned_commit_sha`. Without `--local-path`, `kadmium pull gh://...` fails with a non-zero exit code unless `--plan` is explicitly passed. In dry-run/plan mode (`evaluate_repo_on_slice`), `pinned_commit_sha` is `"unverified_unmounted"` and the response reports `mounted: false`, `execution_mode: "plan_only_unmounted"`, and `verified_node_count: 0`.

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
