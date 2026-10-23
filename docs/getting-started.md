# Getting Started with Kadmium

This guide covers installing **Kadmium**, inspecting a launch plan, opening the Control Room and connecting local workers. The current gateway runs reference compute and returns diagnostic text and vectors; it does not generate language-model answers.

---

## 1. Installation

### Build the current source

Use the current stable Rust toolchain (`rustup`):

```bash
git clone https://github.com/mESmaC/Kadmium.git
cd Kadmium
cargo build --release -p kadmium-cli
```

The compiled binary will be located at `./target/release/kadmium` (or `.\target\release\kadmium.exe` on Windows).

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

The checksums detect corrupted or mismatched downloads; they are not an independent signature. This setup change adds no direct or transitive Rust dependencies and preserves the existing Cargo manifests and lockfile.

---

## 2. Probe Your Local Machine

Before joining or starting a cluster, inspect how Kadmium sees your machine's CPU threads, GPU VRAM, unified memory bandwidth, NVMe streaming throughput, and **FreeToken $q^*$ bandwidth-adaptive ratio**:

```bash
cargo run -p kadmium-cli -- probe
```

**What `kadmium probe` does:**
1. Queries OS memory APIs and GPU drivers (`nvidia-smi` for CUDA, Metal unified memory on macOS, or CPU AVX2/AVX-512 + Vulkan fallback).
2. Runs a live 16 MiB page-aligned streaming benchmark (`64 KiB` pages) to measure real memory/NVMe streaming bandwidth.
3. Computes your node's **FreeToken $q^*$ ratio**:
   $$q^* = \frac{\text{BW}_{\text{RAM}}}{\text{BW}_{\text{VRAM}} + \text{BW}_{\text{RAM}}}$$
   - On an **Apple Silicon Mac Mini (`M4 Pro`)**, unified memory yields $q^* \approx 0.96$.
   - On a **Gaming Desktop (`RTX 4090` + DDR5)**, discrete PCIe bandwidth yields $q^* \approx 0.74$.
4. Generates a cryptographically signed **Kadmium Join Ticket (`kdm1_...`)** ready for peers to join.

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

### Remote Friends, Family, Lab Nodes, or Cloud GPUs (`kdm1_...` Join Tickets)
To add a machine across the internet (behind home NAT) or in a cloud datacenter, copy the `kdm1_...` ticket printed by `kadmium probe` or `kadmium serve`.

Tickets encode:
- The coordinator's QUIC hole-punching relay endpoints
- An optional target **Sub-Cluster Slice ID** (e.g., `slice-01-prod-deploy` or `slice-04-rl-dream-rsi`)
- The node's **Trust Tier**:
  - `TrustedCore` (owned hardware allowed to hold Layer $0$ embedding ingress and Layer $N-1$ output logits)
  - `VerifiedPeerMiddleLayersOnly` (friends/lab peers restricted to middle hidden-state layers $1 \dots N-2$ so raw prompts and decoded tokens never leave your owned devices)
  - `UntrustedPublicSwarm` (verified via Gensyn `Verde` $O(\log_2 L)$ refereed layer bisection)

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

`GET /v1/models` lists planning presets, not verified loaded models. `/v1/embeddings` produces reference vectors, not pretrained semantic embeddings. There is no external backend relay, backend URL flag or credential setting in this setup change. A relay requires a future owner decision about dependencies and downstream access design.

## Next Steps

- Explore every CLI subcommand in the [**CLI Command Reference**](./cli-reference.md).
- Learn how to combine heterogeneous hardware and `.ktb` shard banks in [**Hardware Setup, Asymmetric Sharding & Polite Governance**](./guides/hardware-and-sharding.md).
- Run distributed training and **Dream-RSI** offline causal replay in [**Distributed Training, RL & Dream-RSI Evolving Worlds**](./guides/training-and-dream-rsi.md).
