# Launch and configure Kadmium

Kadmium's launch workflow separates hardware planning from model execution. It can recommend a preset that fits the profiled host, preserve a workspace configuration and start the local Control Room. A separately running inference backend supplies real model responses through `/v1`.

## Inspect before starting

```bash
kadmium launch --plan
kadmium launch --plan --config lab.toml --inference-url http://127.0.0.1:8080
```

The plan prints human-readable hardware-fit recommendations and resolved listener/backend settings. It does not create a configuration, start workers, fetch weights or benchmark the proposed model. Without `--plan`, `launch` creates the selected configuration file if missing and starts the coordinator. Existing files are preserved. Optional hardware-bridge scans are disabled unless `--discover-bridges` is passed.

## Configuration and listener selection

`launch` reads `kadmium.toml` by default. `serve` reads a file only when `--config` is supplied. An explicit `--bind IP:PORT` overrides the file; `--port PORT` changes only its port. These flags cannot be combined. New `launch` configurations use `127.0.0.1:7848`; `serve` retains `0.0.0.0:7848` when no file or override is supplied. An existing file's listener is respected by either command.

The parser rejects unknown sections/keys and malformed values. Supported sections are `[cluster]`, `[host]`, `[governance]`, `[integrations]` and `[cloud_shield]`. For example:

```toml
[cluster]
name = "local-lab"
coordinator_bind = "127.0.0.1:7848"
default_model = "qwen2.5-coder-7b-instruct"
```

`default_model` chooses a reference planning preset, not a backend-loaded model. Host capacity/backend fields in a generated file are historical observations: live profiling determines current capabilities; only the configured host ID and display name are applied. Generate a full file with `kadmium init --out lab.toml`; replacing an existing output requires `--force`. No `[inference]` TOML section is accepted. Backend URL selection is `--inference-url`, then `KADMIUM_INFERENCE_URL`; the backend key is environment-only.

## Use a running inference backend

```bash
kadmium launch --bind 127.0.0.1:7848 --inference-url http://127.0.0.1:8080
```

The backend URL may be an origin or its `/v1` endpoint. Kadmium appends the matching native route. Use `/v1/models` to see that backend's IDs; Kadmium's hardware-planning catalog is separate. Native Anthropic requests require native `/v1/messages` support at the backend. Embeddings, tool calling, multimodal input and structured output are supported only to the extent the backend implements them.

Configure credentials in the environment before launch:

```bash
export KADMIUM_INFERENCE_URL=http://127.0.0.1:8080
export KADMIUM_INFERENCE_API_KEY=YOUR_BACKEND_KEY
kadmium launch
```

```powershell
$env:KADMIUM_INFERENCE_URL = 'http://127.0.0.1:8080'
$env:KADMIUM_INFERENCE_API_KEY = 'YOUR_BACKEND_KEY'
kadmium launch
```

The key is not written to the configuration or shown in status. Keys supplied by downstream clients are ignored; Kadmium forwards only this configured backend credential. Local listener access is not authenticated by those client keys.

## Capacity and readiness

```bash
kadmium serve --inference-url http://127.0.0.1:8080 --inference-concurrency 4 --inference-timeout-secs 120
curl http://127.0.0.1:7848/api/v1/inference/status
curl http://127.0.0.1:7848/v1/models
```

Status describes configuration and available request slots, not backend health. `/v1/models` makes an actual backend request. Slots remain occupied while response bodies stream; a full relay returns `429` with `Retry-After`. Unconfigured inference returns `503`; the rest of the Control Room still runs. See the [API reference](../api-reference.md) for request limits and error handling.

`serve --demo` adds a simulated topology, and `/api/v1/reference/*` exposes reference diagnostics. Neither enables real language-model generation. GPU inference depends on the backend's model, drivers and configuration.

The setup approach is inspired by the reviewed [Strata revision](https://github.com/niko1221/strata/tree/7ddc17a38fe7f31637e895ecfba58c2cd43d65ec). This implementation uses Kadmium's existing coordinator and a native-protocol backend relay; it does not bundle Strata or claim all of its runtime capabilities.
