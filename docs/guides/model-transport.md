# Model transport

Preparing a model in Condenser writes a transport manifest (`transport.json`) next to the package. It describes every byte a receiver needs, so a model can move to a Kadmium coordinator — Kadmium Cloud or your own — without changing a weight, and a dropped connection resumes instead of starting over.

## Layout

The layout follows InferSSD's streaming rules and adds the integrity checks InferSSD does not carry:

| Rule | Value | Why |
| --- | --- | --- |
| Extent alignment | 4,096 bytes for every tensor offset and reserved length | Direct I/O (`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`) rejects unaligned reads |
| Bank page | 64 KiB, zero-padded | KTB banks already satisfy the 4 KiB rule |
| Chunk | 16 MiB | InferSSD's copy chunk; the unit of verification and resume |
| Read order | Expert-major: layer, then expert, then up, gate, down, gate_up | Streams one expert's weights together, as InferSSD's packs do |
| Integrity | SHA-256 per chunk and per file; package ID is the SHA-256 of the source file list | Every chunk is verified on arrival, every file before publishing |
| Publication | Staged in a hidden directory, then renamed | A partial package is never visible |

The manifest lists `manifest.json` first, then `source/` originals, then `banks/` shards. Tensor bytes are never cast, quantized, or reordered inside a file; the read order is metadata only. Building the manifest re-reads the package and fails if any byte differs from what preparation recorded.

## Protocol

A receiver is any Kadmium coordinator that runs these endpoints. This repository implements the coordinator side; `https://api.kadmium.ai` is only the default address in Condenser, and a hosted Kadmium Cloud service is not part of this repository.

| Method and path | Purpose |
| --- | --- |
| `POST /api/v1/model-transport` | Body: the transport manifest (up to 64 MiB). Validates layout, paths and identity, stages the package, and returns `{checkpoint_id, complete, total_bytes, received_bytes, missing: [{file, chunk}]}`. Repeating it resumes. Files are created as verified chunks arrive; nothing is pre-allocated from declared sizes. |
| `PUT /api/v1/model-transport/{id}/files/{file}/chunks/{chunk}` | Raw chunk bytes. Rejected unless length and SHA-256 match. |
| `POST /api/v1/model-transport/{id}/complete` | Starts verifying the whole package (identity, every byte, metadata, bank layout) in the background. Returns `202 {state: "verifying"}` until it publishes (`200 {state: "published", checkpoint}`) or fails (`400` with the reason). Senders poll it. |
| `POST /api/v1/model-transport/{id}/load` | `{context_length?}` loads the preserved originals into the coordinator's Ollama at full context by default and returns `{model, context_length, checkpoint, execution, ollama_path}`. |
| `GET /api/v1/model-transport` | Lists received models and the ones currently served. |
| `POST /api/v1/model-transport/ollama/api/chat`, `…/api/show` | Native Ollama proxy, restricted to received models. |

## Where received models run

A received model runs in the **single Ollama the coordinator is configured with** (`KADMIUM_OLLAMA_URL`). It is not split across Kadmium's grid, and Kadmium's arithmetic guarantees do not apply to it.

Condenser routes tasks for a received model through the native Ollama proxy, using the Ollama provider. Every request therefore carries the same quality settings as a local model: an explicit `num_ctx` at the loaded context, `num_predict: -1` (unlimited until stop), `truncate: false` and `shift: false`, with tools and sampling options passed through unchanged.

`POST /v1/chat/completions` with `model: "kadmium-<id>:latest"` also reaches the loaded model through Ollama's OpenAI-compatible endpoint, but that endpoint cannot carry the no-truncation options, so prefer the native proxy. A received-model name that is not loaded returns `409` instead of a reply from any other handler. After a coordinator restart, a received model that is still present in Ollama is found again automatically. `POST /v1/messages` with a received-model name returns `400` and names the supported routes.

## Access

Set `KADMIUM_TRANSPORT_TOKEN` on a coordinator to accept uploads and chat for received models from other computers with `Authorization: Bearer <token>`; Condenser calls this the **access token** and keeps it in memory only. It is not a Kadmium Cloud account key. Without a token, only requests made from the coordinator itself are accepted: the peer must be loopback, the `Host` must be a loopback name, any `Origin` must match it, and forwarding headers (`Forwarded`, `X-Forwarded-*`, `X-Real-IP`) are refused, so a reverse proxy on the same machine cannot bypass the check. Use a token whenever a proxy sits in front of the coordinator.

`KADMIUM_OLLAMA_URL` (default `http://127.0.0.1:11434`) selects the backend that loads received models, and `KADMIUM_MODEL_LIBRARY` selects where they are stored. `KADMIUM_TRANSPORT_MAX_BYTES`, when set, refuses larger models with `413`. Abandoned transfers older than seven days are removed when a new transfer starts. Condenser only sends over HTTPS, except to loopback, private-network or `.local` addresses.
