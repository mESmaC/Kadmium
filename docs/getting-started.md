# Getting Started with Kadmium

This guide covers installing **Kadmium**, opening Condenser and the Control Room, and connecting local workers. The gateway's model routes return diagnostic text and vectors. For language-model answers, use Condenser with a configured provider or the local checkpoint workflow below.

---

## 1. Installation

### Build the current source

Install `rustup`; the repository's `rust-toolchain.toml` automatically selects the pinned Rust 1.99.0 toolchain with `rustfmt` and Clippy:

```bash
git clone https://github.com/mESmaC/Kadmium.git
cd Kadmium
cargo build --release -p kadmium-cli
```

The compiled binary will be located at `./target/release/kadmium` (or `.\target\release\kadmium.exe` on Windows).

### Load a local model and start a task

Install and start Ollama separately, then build and start Condenser:

```bash
cargo run -p kadmium-condenser-app --locked -- serve
```

Open `http://127.0.0.1:7856` and choose **Models**. Enter your GGUF file or complete SafeTensors folder and choose **Prepare model**. Under **Run it**, keep **This computer**, choose **Load on this computer**, then **Use in current task**. To run it on a Kadmium coordinator you run instead, pick **Remote coordinator**, enter its address and access token, and choose **Send to coordinator**. Normalization preserves source bytes; loading rejects backend tensor changes and never selects a smaller model or context to make it fit. Backend arithmetic is not RepOps-certified. See [local checkpoints](guides/local-checkpoints.md) for supported inputs, storage needs, CLI commands and compatibility limits.

### Checked release installers (next tagged release)

The installer scripts require published assets for an explicit `vX.Y.Z` tag. This documentation does not imply that those assets already exist. Once a matching tag is published, run the script from a checked-out repository and replace the placeholder with that tag:

```bash
bash scripts/install.sh --version vX.Y.Z
# Add --launch to start the installed CLI after verification.
```

```powershell
.\scripts\install.ps1 -Version vX.Y.Z
# Add -Launch to start the installed CLI after verification.
```

| Platform | Release archive |
| :--- | :--- |
| Windows x64 | `kadmium-windows-x86_64.zip` |
| macOS Apple Silicon | `kadmium-macos-arm64.tar.gz` |
| Linux x64 | `kadmium-linux-x86_64.tar.gz` |
| Linux ARM64 | `kadmium-linux-arm64.tar.gz` |

The scripts verify the archive against `SHA256SUMS`, check its executable names and CLI version, and install into a versioned per-user directory. They print the binary path and do not modify `PATH`. Windows and macOS archives include the desktop companion. Linux builds use GNU targets, with Ubuntu 22.04 for x64 and Ubuntu 24.04 for ARM64 as build baselines; compatibility is not universal across Linux distributions or musl systems. No release archive is promised for other architectures.

The checksums detect corrupted or mismatched downloads; they are not an independent signature.

### Roll back or remove an installed release

Installs are versioned under the per-user Kadmium directory printed by the installer. To use an earlier release, invoke its executable directly from that release's `releases/<tag>-<platform>` directory; there is no automated rollback command and the installer does not change `PATH`. To remove a version, close Kadmium and delete only that version's `releases/<tag>-<platform>` directory. Keep the directory if other shortcuts or scripts still reference it.

---

## 2. Probe Your Local Machine

Before joining or starting a cluster, inspect how Kadmium profiles your machine's CPU threads, memory, GPU VRAM, platform family, and **FreeToken $q^*$ placement heuristic**:

```bash
cargo run -p kadmium-cli -- probe
```

**What `kadmium probe` does:**
1. Queries available OS memory and hardware interfaces, including `nvidia-smi` when present. Reported platform/backend labels do not guarantee a native model backend.
2. Runs a 16 MiB synthetic in-memory page-chunk probe and clamps its result to produce the displayed NVMe estimate. This does not read an NVMe device.
3. Computes $q^*$ using `HardwareNode::compute_q_star_ratio`: unified-memory nodes return `0.96`, nodes without VRAM return `0.0`, and discrete-GPU nodes use polite-adjusted FP16 compute, CPU, interconnect, and NVMe terms clamped to `0.10..=0.95`. It is not a RAM/VRAM bandwidth ratio.
4. Prints a truncated example join-ticket prefix with hard-coded endpoints. It is not a usable ticket for your coordinator and is not cryptographically signed.

A generic local P2P roundtrip is not part of `probe`; some physical-platform branches run their own hardware checks.

---

## 3. Launch the Coordinator & Live Control Dashboard

Inspect the setup without creating files or starting listeners:

```bash
cargo run -p kadmium-cli -- launch --plan
```

Then start the existing coordinator on port `7848`:

```bash
cargo run -p kadmium-cli -- launch
```

`launch` preserves an existing `kadmium.toml` and creates it only when missing. Its hardware recommendation is informational: it does not download weights, fix the coordinator's model selection or establish model fit. The existing `serve --port 7848` command remains available and unchanged.

Open **`http://127.0.0.1:7848`** in your browser to inspect live hardware, reference workloads and planner telemetry. Add `--demo` for the simulated seven-node fleet or `--spawn-fleet` for the existing worker test fleet. The HTTP coordinator and built-in worker listen on all interfaces; use a trusted network or external access controls. See the [launch guide](guides/cli.md) for configuration semantics and the [API reference](api-reference.md) for diagnostic behavior.

### Open the desktop app

The Companion (this machine's sharing state, yield control, models, pairing and earnings) runs in its own window, apart from the browser Control Room:

```bash
cargo build -p kadmium-desktop        # once; add --release for a build without a console window
cargo run -p kadmium-cli -- desktop
```

`kadmium desktop` opens the window and starts a local coordinator inside it, or attaches to one already running on the port, so closing the window stops it. The window only ever shows the Companion: the Control Room, the docs and every other link open in your default browser.

- **Windows** draws the window with the Microsoft Edge WebView2 Runtime, which Windows 11 includes. If the app reports that WebView2 is missing, install the Evergreen runtime from Microsoft.
- **macOS** uses the system WKWebView.
- **Linux** needs the WebKitGTK 4.1 and GTK 3 development packages and `cargo build -p kadmium-desktop --features linux-native`. Without the feature, the app runs the coordinator and opens the Companion in your browser instead.

---

## 4. Connect Additional Devices

### Local LAN devices

On a peer, specify the coordinator's reachable LAN address:

```bash
kadmium join --coordinator http://192.168.1.10:7848
```

Replace the example IP with the coordinator's real address. Omitting `--coordinator` uses Kadmium's UDP discovery announcements on port `7849`, not mDNS. Discovery and joining require the relevant HTTP, worker and UDP ports to be reachable. Client API keys do not authenticate access to the coordinator.

### Reachable remote workers and join-ticket limitations

A remote worker needs a reachable coordinator HTTP address to register. Coordinator-to-worker TCP reachability is also required for execution and health checks; registration itself can report acceptance with a fallback RTT when that ping fails. Use a LAN, VPN, or explicitly configured port forwarding and external access controls. The current join path does not perform QUIC hole punching or managed relay traversal.

`kdm1_...` tickets contain endpoint metadata with an unkeyed FNV-1a integrity checksum. They are **not authentication credentials or signed authorization**, and registration currently does not enforce a ticket's target slice or trust tier. Do not treat a ticket as proof that a node is trusted.

`kadmium probe` prints only a truncated example ticket, while `kadmium serve` prints join commands rather than a full ticket. Prefer the direct `--coordinator` flow above. The supported trust-tier names used by placement policy are `TrustedCore`, `VerifiedPeerMiddleLayersOnly`, and `UntrustedAuditedMiddleLayersOnly`; these constrain layer placement, not cryptographic privacy.

---

## 5. Inspect the reference API

The existing `/v1` routes return OpenAI/Anthropic-shaped diagnostic responses. They are not drop-in SDK replacements or a working chat assistant. A request exercises reference compute and returns fixed diagnostic text with telemetry; `stream: true` returns a prebuilt diagnostic event stream, not incremental model generation.

### Chat-shaped diagnostic (`POST /v1/chat/completions`)

```bash
curl -X POST http://127.0.0.1:7848/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Run a reference compute step."}
    ]
  }'
```

### Messages-shaped diagnostic (`POST /v1/messages`)

```bash
curl -X POST http://127.0.0.1:7848/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Run a reference compute step."}
    ]
  }'
```

---

`GET /v1/models` lists planning presets, not verified loaded models. `/v1/embeddings` produces reference vectors, not pretrained semantic embeddings. Condenser connects directly to its configured provider; the gateway does not relay these requests to that provider.

## Next Steps

- Load your own weights with [**Local checkpoints**](./guides/local-checkpoints.md).
- Run user-supplied matrix jobs on TCP workers and integrate persistent learners with [**Distributed compute**](./guides/distributed-compute.md).
- Explore every CLI subcommand in the [**CLI Command Reference**](./cli-reference.md).
- Learn how to combine heterogeneous hardware and `.ktb` shard banks in [**Hardware Setup, Asymmetric Sharding & Polite Governance**](./guides/hardware-and-sharding.md).
- Run distributed training and **Dream-RSI** offline causal replay in [**Distributed Training, RL & Dream-RSI Evolving Worlds**](./guides/training-and-dream-rsi.md).
