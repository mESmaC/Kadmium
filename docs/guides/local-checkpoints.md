# Load a local checkpoint in Condenser

Use **Models & compute** in Condenser to normalize a local checkpoint, load it in a local inference backend, and select it for a task. The same model can answer questions, write code, or support research, math and science; the workflow does not substitute a smaller model for a different task.

Normalization is lossless storage preparation. It does not quantize weights, discard layers or experts, change routing, rewrite tokenizer/chat-template files, or recover precision already lost in the source. The KTB banks are storage artifacts. **Generation uses the preserved originals through local Ollama**, not Kadmium's diagnostic `/v1` endpoints or a native KTB language-model runtime. Ollama is a separately installed dependency; this path does not claim RepOps parity for its arithmetic or KV cache.

## Use the workspace

1. Start your local Ollama server and run `cargo run -p kadmium-condenser -- serve` from your Kadmium workspace.
2. Open `http://127.0.0.1:7856` and choose **Models & compute**. **Provider settings → Load a local checkpoint** opens the same panel.
3. Enter the path to a GGUF file or a complete model directory on the Condenser host. Choose **Normalize checkpoint**. The source remains untouched.
4. Review the format, architecture, tensor count, source types, byte counts, and declared context length. Select **Load in Ollama**. The default backend address is `http://127.0.0.1:11434`; only loopback HTTP addresses are accepted for checkpoint uploads.
5. After the backend confirms loading, choose **Use in this task**, then enter a prompt. Importing and loading do not change a task's provider by themselves. Existing queued prompts retain the routes captured when they were queued.

Loading verifies SHA-256 hashes for every original and bank, uploads originals by content digest, creates a content-named Ollama model without requesting quantization, and compares the backend's reported tensor names, shapes and dtypes against the source. It then preloads the model with the checkpoint's declared context length and confirms the loaded context through /api/ps. Missing metadata, an unsupported architecture, changed precision, or insufficient memory produces an error; no smaller model, shorter context, or different backend is selected. A backend that converts SafeTensors to differently named or typed GGUF tensors will fail this check; use an Ollama version and architecture with native SafeTensors support, or supply a compatible GGUF checkpoint.

These checks cover artifact integrity and reported tensor metadata, not numerical equivalence of the backend implementation. Sampling controls and explicit run/output limits still apply. Stay within the model's context limit; this integration does not independently tokenize prompts to detect provider-side truncation.

## Supported checkpoint packages

| Source | Accepted layout |
| --- | --- |
| SafeTensors | `model.safetensors`, or all root-level shards named by `model.safetensors.index.json`, with `config.json` identifying `model_type` and tokenizer JSON/model files. The index must exactly cover the supplied shards and tensors. |
| GGUF | Little-endian GGUF v3 with architecture and tokenizer metadata. Select a directory for split GGUF; all declared splits and tensors must be present. Metadata may live in the first split. |

SafeTensors normalization accepts F32, F16, BF16, F8_E4M3, I8 and U8. GGUF accepts F32, F16, BF16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1 and Q2_K through Q8_K. Unknown types fail rather than being relabelled. GGUF block types remain distinct from Kadmium's reference quantization codecs. Names, source dimensions and offsets are recorded in `manifest.json`; tensors are copied without casting into version-1 banks with 64 KiB-aligned payloads and at most 64 entries per bank.

The package also preserves recognized config, generation config, tokenizer, vocabulary, special-token and chat-template files byte for byte. It never executes checkpoint Python code. A lone SafeTensors shard, a legacy KTB bank without architecture/tokenizer metadata, an adapter-only checkpoint, and a directory mixing GGUF and SafeTensors are not complete runnable inputs for this flow. Successful normalization is not architecture support or inference readiness: the backend performs that check during loading.

Packages are stored under `target/kadmium-model-library` relative to the working directory. Set `KADMIUM_MODEL_LIBRARY` to a durable path before launching the CLI or Condenser to share a library across workspaces. The CLI also accepts `--library`. Allow roughly two copies of the source size for preserved originals plus banks, with additional alignment overhead; Ollama's blob store can add another copy. Payload copying uses a bounded buffer, not a model-sized allocation. Metadata is bounded separately.

Each package has a content-derived SHA-256 ID. A completed import becomes visible by directory rename. Failed imports do not publish a partial package. Importing the same intact files returns the existing verified package. Source files are never overwritten. A damaged package is rejected; restore it from backup or normalize the original into another library. Do not edit manifests to bypass verification.

## Use the CLI

```sh
cargo run -p kadmium-cli -- checkpoint normalize /models/model.gguf
cargo run -p kadmium-cli -- checkpoint list
cargo run -p kadmium-cli -- checkpoint verify CHECKPOINT_ID
cargo run -p kadmium-condenser -- load-checkpoint CHECKPOINT_ID
cargo run -p kadmium-condenser -- chat "Explain this algorithm" --provider ollama --model kadmium-CHECKPOINT_ID:latest
```

Replace `CHECKPOINT_ID` with the complete ID printed by normalization. The load command prints the exact provider URL and model name to use. Quote filesystem paths containing spaces. This workflow does not download a replacement model.

## HTTP contract

Condenser's existing loopback Host/Origin guard protects these endpoints:

| Method and path | Request / response |
| --- | --- |
| `GET /api/v1/checkpoints` | `{directory, checkpoints}`. Listing reads manifests; it is not a new integrity check. |
| `POST /api/v1/checkpoints/import` | `{source_path}` → checkpoint summary. |
| `POST /api/v1/checkpoints/load` | `{checkpoint_id, base_url}` → `{checkpoint, provider, tensor_metadata_verified, execution}` after verification and preload. |

Only one import/load runs per Condenser server at a time. Long imports remain in progress if the browser closes; refresh the library after reconnecting. A backend load is not automatically retried. Failed loads can leave content-addressed blobs or a created model in Ollama, but do not switch a task's route. Manage those artifacts with Ollama's own tools. Loading keeps the model warm for five minutes; the backend may reload it on a later request.

## Validation provenance

On 2026-10-31, the Windows workflow normalized and loaded [ggml-org/models, tinyllamas/stories260K.gguf](https://huggingface.co/ggml-org/models/blob/main/tinyllamas/stories260K.gguf) through installed Ollama 0.32.9 and generated a response through Condenser. Source SHA-256: `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`; 1,185,376 source bytes, 48 F32 tensors, declared context 2,048 tokens. This small model validates the plumbing, not production-model quality or throughput. Automated fixtures cover byte retention, shard coverage, corruption, backend precision-change rejection, task routing and desktop/mobile layout.

The design follows InferSSD's explicit local-model entry point and source-preserving storage approach, and Ollama's content-addressed create/show APIs. No inference implementation was copied from either project.
