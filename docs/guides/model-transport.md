# Model transport

Preparing a model in Condenser writes a transport manifest (`transport.json`) next to the package. It describes every byte a receiver needs, so a model can move to another Kadmium coordinator without changing a weight, and a dropped connection resumes instead of starting over.

## Layout

The layout follows InferSSD's streaming rules and adds the integrity checks InferSSD does not carry:

| Rule | Value | Why |
| --- | --- | --- |
| Extent alignment | 4,096 bytes for every tensor offset and reserved length | Direct I/O (`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`) rejects unaligned reads |
| Bank page | 64 KiB, zero-padded | KTB banks already satisfy the 4 KiB rule |
| Chunk | 16 MiB | InferSSD's copy chunk; the unit of verification and resume |
| Read order | Expert-major: layer, then expert, then up, gate, down, gate_up | Streams one expert's weights together, as InferSSD's packs do |
| Integrity | SHA-256 per chunk and per file; package ID is the SHA-256 of the source file list | Every chunk is verified on arrival, every file before publishing |
| Publication | Staged in a hidden directory, claimed into a private directory, verified there, then renamed | Only a package that verified as complete is published |

The manifest lists `manifest.json` first, then `source/` originals, then `banks/` shards. Tensor bytes are never cast, quantized, or reordered inside a file; the read order is metadata only. Building the manifest re-reads the package and fails if any byte differs from what preparation recorded.

The chunk map is **Proved (model)** in `formal/KadmiumFormal/Transport/ChunkMap.lean`, tied to the Rust code by a differential fixture (see [formal/README.md](../../formal/README.md)):

- `Kadmium.Transport.readLoop_eq_chunkPlan`: the map the sender reads from each file (`hash_stream`) is exactly `chunk_plan`, so every map it builds passes the receiver's check.
- `Kadmium.Transport.validChunks_iff`: the receiver's check in `TransportManifest::validate` accepts exactly one map per file, `chunk_plan`.
- `Kadmium.Transport.chunkPlan_tiles`: that map starts at offset 0, each chunk starts where the previous one ends, every chunk holds 1 byte to 16 MiB, and together they cover the file exactly once.
- `Kadmium.Transport.chunkPlan_aligned` and `layout_divides`: every chunk offset is a multiple of 16 MiB, so of the 64 KiB page and the 4 KiB direct-I/O boundary.

The rest of the layout (extent alignment inside banks, expert-major read order, identity) is checked by `validate` and the tests in `crates/kadmium-data/tests/checkpoint_transport.rs`, not by a theorem.

Publication: `receive_finish` first renames the staged transfer into a private `.verifying/<claim>/<id>` directory, verifies it there and renames that same directory into the library, so a concurrent `begin` for the same checkpoint cannot change what is published. `Kadmium.Transport.Publish.publish_safe` proves that every published package verified as complete, for every interleaving of begins, chunk arrivals and concurrent finishes; `Kadmium.Transport.Publish.legacy_unsafe` shows the previous verify-then-rename order could publish a directory without `manifest.json`. These are **Proved (model)** about a model only: renames are assumed atomic, and the unit test `begin_during_finish_cannot_publish_a_package_without_its_manifest` runs the counterexample's interleaving against the Rust code. A transfer with missing chunks goes back to staging for resume; one that fails verification is discarded, and the sender's next attempt starts clean.

## Protocol

A receiver is any Kadmium coordinator that runs these endpoints. This repository implements the coordinator side. Kadmium Cloud, the planned premium hosted service, will use the same protocol; it is not available yet, so Condenser has no default address.

| Method and path | Purpose |
| --- | --- |
| `POST /api/v1/model-transport` | Body: the transport manifest (up to 64 MiB). Validates layout, paths and identity, stages the package, and returns `{checkpoint_id, complete, total_bytes, received_bytes, missing: [{file, chunk}]}`. Repeating it resumes. Files are created as verified chunks arrive; nothing is pre-allocated from declared sizes. |
| `PUT /api/v1/model-transport/{id}/files/{file}/chunks/{chunk}` | Raw chunk bytes. Rejected unless length and SHA-256 match. |
| `POST /api/v1/model-transport/{id}/complete` | Starts verifying the whole package (identity, every byte, metadata, bank layout) in the background. Returns `202 {state: "verifying"}` until it publishes (`200 {state: "published", checkpoint}`) or fails (`400` with the reason). One package is verified at a time; another returns `409 {state: "busy"}`. Senders poll it. |
| `POST /api/v1/model-transport/{id}/load` | `{context_length?}` starts loading the preserved originals into the coordinator's Ollama in the background, at full context unless you lower it. Returns `202 {state: "loading"}` until it answers `200 {model, context_length, checkpoint, execution, ollama_path}` or fails (`502`). One model loads at a time; another returns `409 {state: "busy"}`. A checkpoint the coordinator has not received returns `404`. Senders poll it. |
| `GET /api/v1/model-transport` | Lists received models and the ones currently served. |
| `POST /api/v1/model-transport/ollama/api/chat`, `…/api/show` | Native Ollama proxy, restricted to received models. |

## Where received models run

A received model runs in the **single Ollama the coordinator is configured with** (`KADMIUM_OLLAMA_URL`). It is not split across Kadmium's grid, and Kadmium's arithmetic guarantees do not apply to it.

Condenser routes tasks for a received model through the native Ollama proxy, using the Ollama provider. The proxy makes every chat request full quality before it reaches Ollama, whichever client sent it: it sets `truncate: false` and `shift: false`, sets `num_ctx` to the context the model was loaded with, and sets `num_predict: -1` (unlimited until stop) unless the request chose an output limit. A request that asks for truncation, context shifting or a different `num_ctx` is refused with `400`, naming the numbers; it is never adjusted silently. To use a smaller context, load the model again with a smaller **Context window**. Tools and sampling options pass through unchanged.

Received models are served **only** through the native proxy. Ollama's OpenAI-compatible endpoint cannot turn off truncation or context shifting, so `POST /v1/chat/completions` and `POST /v1/messages` with a received-model name (`kadmium-<id>`, with any tag) return `400` and name the native route instead of answering with reduced quality. A received-model name that is not loaded, including any tag other than `:latest`, returns `409` instead of a reply from Kadmium's diagnostic handlers. After a coordinator restart, a received model that is still present in Ollama is found again automatically.

## Access

A coordinator receives and serves models only when `KADMIUM_TRANSPORT_TOKEN` is set to at least 16 characters; a shorter token disables transport and every transport route answers `503` with the reason. Every upload, load and chat request needs `Authorization: Bearer <token>` (the scheme name is case-insensitive); Condenser calls this the **access token** and keeps it in memory only. Without a token the transport routes answer `403`. A loopback peer is not accepted as proof of the owner, because a reverse proxy or another user on the same computer also connects from loopback. Only the read-only listing (`GET /api/v1/model-transport`) is also open to a same-origin browser on the coordinator itself, through the check the embedded Condenser workspace uses: a loopback peer, a loopback `Host` name and a matching `Origin`. The `/v1` interceptor runs before authentication, so it reads at most 2 MiB of a request body, the same limit as the `/v1` handlers.

`KADMIUM_OLLAMA_URL` (default `http://127.0.0.1:11434`) selects the backend that loads received models, and `KADMIUM_MODEL_LIBRARY` selects where they are stored. `KADMIUM_TRANSPORT_MAX_BYTES`, when set to a whole number of bytes, refuses any single model larger than that with `413`; it is a per-model cap, not a total for the library. A value that is not a byte count, such as `50GB`, disables transport with `503` instead of silently meaning no cap. Abandoned transfers older than seven days are removed when a new transfer starts. The coordinator does not check free disk space before accepting a transfer; a full disk fails the chunk that cannot be written, and resending resumes after you free space. Condenser only sends over HTTPS, except to loopback, private-network or `.local` addresses.
