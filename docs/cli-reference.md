# Kadmium CLI Command Reference

The `kadmium` CLI is a single-binary hardware profiler, planner, reference-workload runner, and HTTP/TCP coordinator. Many commands intentionally print **simulated or modelled reports**; those outputs are not live cloud operations, remote downloads, production benchmarks, or language-model generation.

Run `kadmium --help` or `kadmium <COMMAND> --help` for the clap-generated synopsis. Examples below use an installed binary; prefix a command with `cargo run -p kadmium-cli --locked --` to run it from a checkout.

## Coordinator and desktop

### `kadmium serve`

Starts the Control Room, reference HTTP API, and built-in TCP worker.

```bash
kadmium serve --port 7848
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p`, `--port <PORT>` | `7848` | HTTP port on all interfaces. |
| `-c`, `--config <PATH>` | none | Optional tolerant `kadmium.toml` parser input. |
| `--live` | on unless `--demo` | Use only probed hardware nodes. |
| `--demo` | off | Add the simulated seven-node fleet. |
| `--spawn-fleet` | off | Start the five-node local/online TCP test fleet on ports `7851..=7855`. |
| `--worker-port <PORT>` | `7850` | Built-in worker TCP port; `0` requests an ephemeral port. |

The `/v1` routes return fixed diagnostic text and reference vectors, not model output. The coordinator has no built-in tenant authentication; use a trusted network or external access controls.

### `kadmium launch`

Probes the host, prints an informational setup recommendation, preserves or creates a workspace file, and starts the coordinator. `--plan` performs no writes and starts no listeners.

```bash
kadmium launch --plan
kadmium launch --config kadmium.toml --port 7848
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--plan` | off | Print the setup plan and exit. |
| `-p`, `--port <PORT>` | `7848` | HTTP port; configuration cannot override it. |
| `-c`, `--config <PATH>` | `kadmium.toml` | Preserve an existing file or create a missing one. |
| `--demo` | off | Add the simulated fleet. |
| `--worker-port <PORT>` | `7850` | Worker TCP port; `0` requests an ephemeral port. |
| `--spawn-fleet` | off | Start the local TCP test fleet. |

The recommendation does not download weights, select a loaded model, or prove model fit.

### `kadmium desktop`

Opens the Companion and starts or attaches to a local coordinator. The `app` alias is equivalent.

```bash
cargo build -p kadmium-desktop
kadmium desktop --control-room
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p`, `--port <PORT>` | `7848` | Coordinator port. |
| `--live` | on unless `--demo` | Use only probed hardware. |
| `--demo` | off | Add the simulated fleet. |
| `--control-room` | off | Also open the Control Room in the browser. |
| `--check-only` | off | Print hardware and desktop-app discovery information, then exit. |

You can also launch the installed app directly with `kadmium-desktop --port 7848` (or `cargo run -p kadmium-desktop --locked -- --port 7848` from a checkout).

The app is searched for next to `kadmium`, then on `PATH`. Linux uses the browser by default; a native WebView requires the `linux-native` feature and WebKitGTK 4.1/GTK 3 development packages.

## Hardware and placement

### `kadmium checkpoint`

Normalize, list or verify complete local checkpoints without changing tensor precision, layers, experts or tokenizer files:

```bash
kadmium checkpoint normalize /models/model.gguf --library /models/kadmium
kadmium checkpoint normalize /models/model-Q4_K_M.gguf --accept-pre-quantized
kadmium checkpoint list --library /models/kadmium
kadmium checkpoint verify CHECKPOINT_ID --library /models/kadmium
```

All three commands print JSON. `--library <DIR>` defaults to `KADMIUM_MODEL_LIBRARY`, then the shared per-user model directory (for example `%LOCALAPPDATA%\Kadmium\models` on Windows). Normalization preserves originals and writes 64 KiB-aligned KTB banks with SHA-256 identities; it does not load or execute the model. A checkpoint that was quantized before import is refused unless you pass `--accept-pre-quantized`. Its bytes are kept exactly, but precision lost before import cannot be recovered, and the summary records the method and formats as `pre_quantized`. See [local checkpoints](guides/local-checkpoints.md) for supported formats, library migration, shard completeness, storage requirements and backend limits.

### `kadmium-condenser load-checkpoint`

Verify a normalized package and load its preserved originals in a local Ollama server:

```bash
kadmium-condenser load-checkpoint CHECKPOINT_ID --library /models/kadmium
kadmium-condenser chat "Explain this algorithm" --provider ollama --model kadmium-CHECKPOINT_ID:latest
```

`--library` uses the same default as above. `--base-url` defaults to `http://127.0.0.1:11434` and must be loopback HTTP. The command prints the provider configuration only after source verification, backend tensor checks and preload at the declared context length. It does not quantize, pull a replacement model or select a smaller context. Tensor conversion or unsupported backend behavior fails explicitly. Run from source with `cargo run -p kadmium-condenser-app --locked -- load-checkpoint CHECKPOINT_ID`.

### `kadmium probe`

Profiles local hardware and runs reference packing/page probes. The displayed NVMe figure is a clamped estimate from a 16 MiB synthetic in-memory operation, not an NVMe-device benchmark.

The $q^*$ heuristic returns `0.96` for unified memory, `0.0` without VRAM, and otherwise uses compute/interconnect/NVMe terms clamped to `0.10..=0.95`. The printed join-ticket prefix is truncated and uses example endpoints; it is not a usable signed credential. A generic P2P roundtrip is not part of this command.

```bash
kadmium probe
```

### `kadmium plan`

Computes a model/workload placement over the sample personal cluster.

```bash
kadmium plan --model deepseek-v3-671b-moe --workload inference
kadmium plan --model llama-3-70b-lora --workload training
kadmium plan --workload batch
kadmium plan --workload rl
kadmium plan --workload dream-rsi
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m`, `--model <ID>` | `deepseek-v3-671b-moe` | Known preset ID or alias (listed below); unknown IDs fall back to `deepseek-v3-671b-moe`. |
| `-w`, `--workload <MODE>` | `inference` | `inference`, `training`, `batch`, `rl`, or `dream-rsi`. |

The eight presets and accepted aliases are:

| Preset | Aliases |
| :--- | :--- |
| `llama-3.2-3b-instruct` | `llama-3.2-3b` |
| `qwen2.5-coder-7b-instruct` | `qwen2.5-coder-7b` |
| `qwen2.5-coder-14b-instruct` | `qwen2.5-coder-14b` |
| `qwen2.5-coder-32b-instruct` | `qwen2.5-coder-32b` |
| `qwen3-235b-moe` | none |
| `llama-3-70b-lora` | `llama-3.3-70b-instruct` |
| `deepseek-v3-671b-exl3-2.5bpw` | `deepseek-v3-exl3`, `exl3` |
| `deepseek-v3-671b-moe` | none |

RL and Dream-RSI use the dedicated RL topology planner; the other modes use `ShardPlanner`. Estimates, staging streams, and fit reports are modelled. Usage/calibration records are loaded from `target/kadmium-state` when present.

### `kadmium calibrate`

Asks a running coordinator to measure up to eight synthetic decode steps of one placement shape within its 20-second budget.

```bash
kadmium calibrate --coordinator http://127.0.0.1:7848 --strategy expert-parallel --steps 8
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-c`, `--coordinator <URL>` | `http://127.0.0.1:7848` | Coordinator HTTP gateway. |
| `-s`, `--strategy <SHAPE>` | planner's choice | `chain` or `expert-parallel`. |
| `--steps <N>` | `8` | Decode steps, at most 8. |
| `--rows <N>` | `1` | Rows per step. |

### `kadmium simulate`

Prints five groups of sample output: hardware discovery, an inference plan, a training plan, a gaming responsibility-shift plan, and a cloud-burst shift plan, followed by RL and BOINC-style sample reports. It does not execute FTS3 network staging or live model migration.

```bash
kadmium simulate
```

### `kadmium diloco`

Runs a deterministic single-process CPU regression experiment. Independent islands learn `y = 2*x + 1` from different input ranges, perform local AdamW steps, and synchronize complete replicas through the shared Nesterov outer optimizer.

```bash
kadmium diloco --inner-steps 20 --rounds 25 --islands 3 --json
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--inner-steps <H>` | `20` | Local steps per round, `1..=10000`. |
| `--rounds <N>` | `25` | Outer rounds, `1..=1000`. |
| `--islands <N>` | `3` | Independent learners, `2..=64`. |
| `--json` | off | Emit the full JSON report. |

The inner AdamW optimizer uses learning rate `0.02`, betas `0.9`/`0.999`, epsilon `1e-8`, and weight decay `0.01`. The outer Nesterov optimizer uses learning rate `0.7` and momentum `0.9`.

Total local work is capped at two million steps. The default report compares 25 outer synchronizations with a 500-sync per-step baseline; payload estimates are not measured network savings. This is not LLM, GPU, WAN, or distributed pipeline training. Each CLI run starts fresh.

## Pool, RL, and cloud simulations

### `kadmium slice`

Builds a stratified pool-partition report. The default five-slice catalog uses purposes `ProductionDeployment`, `StagingAndTesting`, `DistributedTrainingLab`, `RlAndDreamRsiArena`, and `BatchAndEvalPool`. It does not create isolated network namespaces, per-slice coordinators, or tenant boundaries.

```bash
kadmium slice --pool-size 100 --slices 5
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p`, `--pool-size <N>` | `100` | Nodes in the sample pool. |
| `-s`, `--slices <K>` | `5` | Number of partitions. |

### `kadmium dataflow`

Evaluates `AdaptiveTieredHybrid`, `ComputeInMemoryTokenStreaming`, `TokenStationaryWeightStreaming`, and `DisaggregatedDualStream` planning profiles. Recommendations and throughput are modelled.

```bash
kadmium dataflow --model deepseek-v3-671b-moe --batch-tokens 512
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m`, `--model <ID>` | `deepseek-v3-671b-moe` | Model preset. |
| `-b`, `--batch-tokens <N>` | `512` | Active token batch used by the roofline model. |

### `kadmium dream`

Replays the bundled in-memory Dream-RSI discovery tree. The promoted sample policy reaches its normalized 2.31× peak after 4 of 9 unit-cost evaluations, a 55.6% evaluation-cost saving. No GPU benchmark, persistent NAS tree, online RL epoch, or meta-prompt injection is performed.

```bash
kadmium dream
```

### `kadmium cloud-shield`

Runs fixed spot-arbitrage and bidirectional-failover scenarios. The current sample prices total $1.82/hr versus $13.70/hr (86.7% modelled savings); simulated timings are 340 ms local-to-cloud, 185 ms cloud-to-local/multicloud, and 210 ms failback. No provider is contacted and no capacity is provisioned.

```bash
kadmium cloud-shield
```

### `kadmium lease`

Prints a local settlement **quote** for idle capacity. It does not create a lease, transfer money, or issue credits.

```bash
kadmium lease --hours 8 --payout credits
kadmium lease --payout usd
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--hours <HOURS>` | `12.0` | Daily idle hours, `0.0..=24.0`. |
| `-p`, `--payout <MODE>` | `credits` | `credits` (illustrative 1.25× rate) or `usd`. |

## Workers and integrations

### `kadmium join`

Registers a persistent TCP worker with a reachable coordinator. Use `--coordinator` for a direct address or omit it for UDP LAN discovery on port `7849`.

```bash
kadmium join --coordinator http://192.168.1.10:7848
kadmium join --offer-ram-gb 24 --offer-fast-gb 10 --slots 2 --require-approval
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-c`, `--coordinator <URL>` | UDP discovery | Coordinator HTTP gateway. |
| `-t`, `--ticket <TOKEN>` | none | Optional `kdm1_...` endpoint descriptor. |
| `--worker-port <PORT>` | `7850` | Worker TCP listener port. |
| `--hostname <NAME>` | detected | Custom node label. |
| `--live-only` | off | Switch coordinator to live-only mode on registration. |
| `--once` | off | Register and run one verification pass instead of the heartbeat loop. |
| `--offer-fast-gb <GIB>` | 90% VRAM + unified memory | Offered accelerator/unified memory. |
| `--offer-ram-gb <GIB>` | 70% outside reserve | Offered RAM; reserve is one quarter of RAM bounded to 1–4 GiB. |
| `--offer-nvme-gb <GIB>` | whole Kadmium cache | Offered cold-storage capacity. |
| `--slots <N>` | `4` | Concurrent work instances. |
| `--require-approval` | off | Require owner approval before staging. |

The worker measures memory streaming with a 64 MiB single-thread copy before registration. Tickets use an unkeyed FNV-1a integrity checksum, not HMAC or a signature; registration does not authenticate the ticket or enforce its slice/trust fields. Remote joining requires reachable HTTP/TCP addresses through a LAN, VPN, or port forwarding, not automatic QUIC/NAT traversal.

Set `KADMIUM_USBC_BRIDGE_BIND` (for example `0.0.0.0:7871`) on a joining worker to also accept [USB-C delta bridge](guides/usbc-delta-bridge.md) sessions. A coordinator can then use a `kdbr://` address with one lane per path; lane failover preserves the delta-link stores. This is separate from join-ticket authentication, which remains unimplemented.

### `kadmium pull`

Plans an `hf://` model or dataset pull against the local host or stages a verified local `gh://` checkout. For `gh://` URIs, pass `--local-path <DIR>` to verify a local `.git` checkout's `HEAD`, remote URL, and subpath, or pass `--plan` (`--dry-run`) to inspect a plan-only summary without a local checkout. For `hf://` model URIs, pass `--index <PATH>` (`--index-manifest <PATH>`) to map layer ranges onto actual `.safetensors` files from a `model.safetensors.index.json` manifest.

```bash
kadmium pull hf://deepseek-ai/DeepSeek-V3@main --index /path/to/model.safetensors.index.json
kadmium pull gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt --local-path ./Dream-RSI
kadmium pull gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt --plan
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--plan`, `--dry-run` | off | Allow `gh://` evaluation without a mounted local checkout (`execution_mode: "plan_only_unmounted"`). |
| `--local-path <DIR>` | none | Verify and stage a local `.git` repository checkout matching the `gh://` owner, repo, ref, and subpath. |
| `--index <PATH>`, `--index-manifest <PATH>` | none | Supply a local `model.safetensors.index.json` manifest for `hf://` model pulls. |

The optional URI defaults to `hf://deepseek-ai/DeepSeek-V3@main`. See the [integration guide](guides/integrations-hf-github.md).

### `kadmium mcp`

Starts an isolated stdio MCP server with a fresh in-process gateway state (`live_unconnected` by default with 0 nodes unless `--demo` is passed), or prints its seven-tool manifest. It does not attach to a running coordinator or read `KADMIUM_COORDINATOR_URL`; use `POST /mcp` for shared coordinator state.

```bash
kadmium mcp --inspect
kadmium mcp
kadmium mcp --demo
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--inspect` | off | Print the manifest and client snippet, then exit. |
| `--demo`, `--sample-cluster` | off | Populate the in-process MCP server with the 4-node synthetic demo fixture instead of starting in `live_unconnected` mode. |

## Tensor banks and quantization

### `kadmium import`

Imports one local SafeTensors or compact zero-metadata GGUF v3 file into `.ktb`. Typical GGUF files with metadata are outside the current parser's support. Without `--path`, the command synthesizes a reference input under `target/kadmium-models/`.

```bash
kadmium import --model qwen2.5-coder-7b --path ./model.safetensors --start-layer 0 --end-layer 3
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m`, `--model <ID>` | `qwen2.5-coder-32b-instruct` | Known model preset ID or alias; unknown IDs are rejected (custom architecture metadata is not supported). |
| `-p`, `--path <PATH>` | synthesize input | Local `.gguf` or `.safetensors` file. |
| `-f`, `--format <FORMAT>` | `gguf` | Synthesized-input hint: `gguf` or `safetensors`. |
| `--start-layer <N>` | `0` | Inclusive first layer. |
| `--end-layer <N>` | `3` | Inclusive last layer. |

### `kadmium compile`

Generates a synthetic `.ktb` archive for a layer range, parses its header, and prints its Merkle root. It does not read or compile pretrained model weights. Use `kadmium inspect` to verify every page proof.

```bash
kadmium compile --model example --start-layer 0 --end-layer 3 --out shard.ktb
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m`, `--model <ID>` | `deepseek-v3-671b-moe` | Archive model label. |
| `--start-layer <N>` | `0` | Inclusive first layer. |
| `--end-layer <N>` | `3` | Inclusive last layer. |
| `-o`, `--out <PATH>` | in-memory only | Optional output file. |

### `kadmium inspect`

Reads a `.ktb` file and verifies every 64 KiB page against its FNV-1a Merkle proof. Without a path, inspects an in-memory synthetic archive.

```bash
kadmium inspect shard.ktb
```

### `kadmium exl3`

Runs the EXL3/QTIP conversion and benchmark **model** over the sample cluster. Reported sizes, SNR, and throughput are synthetic/modelled, not conversion of a downloaded checkpoint.

```bash
kadmium exl3 --model deepseek-v3-671b-exl3-2.5bpw --bpw 2.55
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m`, `--model <ID>` | `deepseek-v3-671b-exl3-2.5bpw` | Model preset/label. |
| `-b`, `--bpw <BPW>` | `2.55` | Target bits per weight, `2.0..=6.0`. |

## Configuration, diagnostics, and platform profiles

### `kadmium init`

Probes the host and prints a workspace configuration. `--out` also writes it, replacing the specified file.

```bash
kadmium init --out kadmium.toml
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-o`, `--out <PATH>` | stdout only | Output configuration path. |

### `kadmium bench`

Runs synthetic host kernel, range-slicer, and NAT-planning reports over sample node profiles. Its cross-silicon labels and estimated GFLOPS do not establish native execution on each labelled backend. NAT output includes route planning and a loopback UDP self-test, not a live STUN/DERP service.

```bash
kadmium bench
```

### `kadmium pairing`

Prints sample LAN-pairing candidates, an OS-hysteresis timeline, and the fixed CloudShield watcher scenario (340 ms simulated RTO). These are reference state machines, not proof of foreground-hook integration, measured failover, or zero dropped requests.

```bash
kadmium pairing
```

### `kadmium registry`

Displays fixed spot quotes, a sample RBAC/key catalog, a deterministic reference fraud-proof example, and the built-in Dream-RSI world catalog. It does not query providers, enforce tenant access, slash workers, or contact a public registry.

```bash
kadmium registry
```

### `kadmium update`

Submits an update-manifest request to `POST /api/v1/update/push`. If no coordinator answers, it starts an in-process test fleet and pushes the synthetic manifest to those test workers. This is not a general release-binary deployment mechanism or proof of zero-downtime WAN delivery.

```bash
kadmium update --coordinator http://127.0.0.1:7848 --version v0.2.0-cd48 --kind binary --scope all
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-c`, `--coordinator <URL>` | `http://127.0.0.1:7848` | Coordinator HTTP gateway. |
| `-v`, `--version <VERSION>` | `v0.2.0-cd48` | Target version label. |
| `-k`, `--kind <KIND>` | `binary` | `binary`, `ktb-delta`, or `policy`. |
| `-s`, `--scope <SCOPE>` | `all` | `all`, `local`, or `online`. |

### `kadmium phone-farm`

Builds a 20-slot phone-farm hardware profile and simulated topology, with ADB discovery dependent on the local `adb` installation and attached devices.

```bash
kadmium phone-farm --preset sd865 --disaggregated
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p`, `--preset <PRESET>` | `sd865` | `sd845`, `sd865`, or `sd8gen2`. |
| `-d`, `--disaggregated` | off | Model 20 individual micro-workers instead of one coherent array. |

### `kadmium switch`

Builds a Tegra X1/X1+ platform profile; alias `tegra-x1`.

```bash
kadmium switch --revision mariko
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-r`, `--revision <REV>` | `mariko` | `mariko`, `erista`, or `mariko-8gb`. |

### `kadmium reterminal`

Probes the CM4 LAN bridge and runs its platform verification path; aliases `cm4` and `rpi-cm4`.

```bash
kadmium reterminal --ip 192.168.40.75
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--ip <ADDR>` | `192.168.40.75` | LAN address of the CM4 node. |

### `kadmium m4`

Probes the Apple M4 LAN bridge and runs its platform verification path; alias `apple-m4`.

```bash
kadmium m4 --ip 192.168.40.237
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--ip <ADDR>` | `192.168.40.237` | LAN address of the Apple M4 node. |
