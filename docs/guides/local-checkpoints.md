# Load a local checkpoint in Condenser

Use **Models & compute** in Condenser to normalize a local checkpoint, load it in a local inference backend, and select it for a task. The same model can answer questions, write code, or support research, math and science; the workflow does not substitute a smaller model for a different task.

Normalization is lossless storage preparation. It does not quantize weights, discard layers or experts, change routing, rewrite tokenizer/chat-template files, or recover precision already lost in the source. The KTB banks are storage artifacts. **Generation uses the preserved originals through local Ollama**, not Kadmium's diagnostic `/v1` endpoints or a native KTB language-model runtime. Ollama is a separately installed dependency; this path does not claim RepOps parity for its arithmetic or KV cache.

## Use the workspace

1. Start Kadmium (`kadmium serve`, or the desktop app) and your local Ollama server.
2. Open **Models → Prepare a local model** in the Control Room or desktop Companion. Condenser is included at `http://127.0.0.1:7848/condenser/`; no second process is needed. In a browser, first open the **Condenser workspace** access link printed by `kadmium serve`; the desktop app does this for you. **Launch Condenser** opens chat directly. The desktop keeps Condenser in its native window; **← Kadmium** returns to Models. Standalone Condenser still works on port `7856`.
3. Choose **Browse model files** for a GGUF or SafeTensors file, or **Choose model folder** for a complete package. Kadmium's desktop Companion and standalone Condenser use the system file dialog (Windows, macOS, or the Linux desktop portal in a `linux-native` build). Browser sessions use the embedded chooser because browser file inputs cannot provide the absolute host paths needed for import. Selecting a SafeTensors file selects its folder, including config, tokenizer and other shards. Cancel leaves the current path unchanged. You can also paste a path on the coordinator computer, or choose **Use embedded browser** if a system dialog is unavailable. Choose **Normalize checkpoint**. The source remains untouched.
4. Review the format, architecture, tensor count, source types, byte counts, and declared context length. Select **Load in Ollama**. The default backend address is `http://127.0.0.1:11434`; only loopback HTTP addresses are accepted for checkpoint uploads.
5. After the backend confirms loading, choose **Use in this task**, then enter a prompt. Importing and loading do not change a task's provider by themselves. Existing queued prompts retain the routes captured when they were queued.

Loading verifies SHA-256 hashes for every original and bank, uploads originals by content digest, creates a content-named Ollama model without requesting quantization, and compares the backend's reported tensor names, shapes and dtypes against the source. It then preloads the model with the checkpoint's declared context length and confirms the loaded context through /api/ps. Missing metadata, an unsupported architecture, changed precision, or insufficient memory produces an error; no smaller model, shorter context, or different backend is selected. A backend that converts SafeTensors to differently named or typed GGUF tensors will fail this check; use an Ollama version and architecture with native SafeTensors support, or supply a compatible GGUF checkpoint.

The load step uses the full declared context unless you explicitly lower **Context window for loading**. Every Ollama chat request resolves native context through `/api/show`, sends `num_ctx` explicitly, requests unlimited-until-stop output (`num_predict: -1`) by default, and sets `truncate: false` and `shift: false` to avoid discarding history. Input and output share the model's context window. These checks cover artifact integrity and reported tensor metadata, not numerical equivalence of the backend implementation. Condenser does not independently tokenize prompts or certify backend KV-cache precision; use an Ollama version that honors the no-truncation option.

## Supported checkpoint packages

| Source | Accepted layout |
| --- | --- |
| SafeTensors | `model.safetensors`, or all root-level shards named by `model.safetensors.index.json`, with `config.json` identifying `model_type` and tokenizer JSON/model files. The index must exactly cover the supplied shards and tensors. |
| GGUF | Little-endian GGUF v3 with architecture and tokenizer metadata. Select a directory for split GGUF; all declared splits and tensors must be present. Metadata may live in the first split. |

SafeTensors normalization accepts F32, F16, BF16, F8_E4M3, I8 and U8. GGUF accepts F32, F16, BF16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1 and Q2_K through Q8_K. Unknown types fail rather than being relabelled. GGUF block types remain distinct from Kadmium's reference quantization codecs. Names, source dimensions and offsets are recorded in `manifest.json`; tensors are copied without casting into version-1 banks with 64 KiB-aligned payloads and at most 64 entries per bank.

The package also preserves recognized config, generation config, tokenizer, vocabulary, special-token and chat-template files byte for byte. It never executes checkpoint Python code. A lone SafeTensors shard, a legacy KTB bank without architecture/tokenizer metadata, an adapter-only checkpoint, and a directory mixing GGUF and SafeTensors are not complete runnable inputs for this flow. Successful normalization is not architecture support or inference readiness: the backend performs that check during loading.

The panel shows the library directory and checkpoint ID. Each package contains `manifest.json`, originals in `source/`, and normalized `bank-00000.ktb` (and subsequent) shards in `banks/`. The shared default library is `%LOCALAPPDATA%\Kadmium\models` on Windows, `~/Library/Application Support/Kadmium/models` on macOS, and `$XDG_DATA_HOME/kadmium/models` or `~/.local/share/kadmium/models` on Linux. Set `KADMIUM_MODEL_LIBRARY` before launching Kadmium or Condenser to use another disk. The CLI also accepts `--library`. Existing packages in the former `target/kadmium-model-library` directory are not moved or deleted: point the override to that absolute path to keep using them. Allow roughly two copies of source size plus alignment overhead; Ollama can add another copy. Payload copying uses a bounded buffer. Metadata is bounded separately.

Each package has a content-derived SHA-256 ID. A completed import becomes visible by directory rename. Failed imports do not publish a partial package. Importing the same intact files returns the existing verified package. Source files are never overwritten. A damaged package is rejected; restore it from backup or normalize the original into another library. Do not edit manifests to bypass verification.

## Use the CLI

```sh
cargo run -p kadmium-cli -- checkpoint normalize /models/model.gguf
cargo run -p kadmium-cli -- checkpoint list
cargo run -p kadmium-cli -- checkpoint verify CHECKPOINT_ID
cargo run -p kadmium-condenser-app -- load-checkpoint CHECKPOINT_ID
cargo run -p kadmium-condenser-app -- chat "Explain this algorithm" --provider ollama --model kadmium-CHECKPOINT_ID:latest
```

Replace `CHECKPOINT_ID` with the complete ID printed by normalization. The load command prints the exact provider URL and model name to use. Quote filesystem paths containing spaces. This workflow does not download a replacement model.

## HTTP contract

Condenser's loopback Host/Origin guard protects these endpoints. When embedded, prefix paths with `/condenser`; Kadmium also checks the transport peer and same-origin requests. Remote callers cannot browse or load files by forging a loopback Host header.

| Method and path | Request / response |
| --- | --- |
| `GET /api/v1/checkpoints` | `{directory, checkpoints}`. Listing reads manifests; it is not a new integrity check. |
| `POST /api/v1/checkpoints/import` | `{source_path}` → checkpoint summary. |
| `POST /api/v1/checkpoints/browse` | `{path?}` → host directory, parent and up to 500 model/folder entries. |
| `POST /api/v1/checkpoints/load` | `{checkpoint_id, base_url, context_length?}` → `{checkpoint, provider, context_length, tensor_metadata_verified, execution}` after verification and preload. |

Only one import/load runs per Condenser server at a time. Long imports remain in progress if the browser closes; refresh the library after reconnecting. A backend load is not automatically retried. Failed loads can leave content-addressed blobs or a created model in Ollama, but do not switch a task's route. Manage those artifacts with Ollama's own tools. Loading keeps the model warm for five minutes; the backend may reload it on a later request.

## Validation provenance

On 2026-10-31, the Windows workflow normalized and loaded [ggml-org/models, tinyllamas/stories260K.gguf](https://huggingface.co/ggml-org/models/blob/main/tinyllamas/stories260K.gguf) through installed Ollama 0.32.9 and generated a response through Condenser. Source SHA-256: `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`; 1,185,376 source bytes, 48 F32 tensors, declared context 2,048 tokens. This small model validates the plumbing, not production-model quality or throughput. Automated fixtures cover byte retention, shard coverage, corruption, backend precision-change rejection, task routing and desktop/mobile layout.

The design follows InferSSD's explicit local-model entry point and source-preserving storage approach, and Ollama's content-addressed create/show APIs. No inference implementation was copied from either project.
