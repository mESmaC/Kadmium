# Getting Started with Kadmium

This guide covers installing **Kadmium**, inspecting a launch plan, opening the Control Room and connecting an existing inference backend. The backend supplies real generated text and embeddings; Kadmium's reference kernels and topology demonstrations remain separate.

---

## 1. Installation

### Build the current source

Use a current stable Rust toolchain:

```bash
git clone https://github.com/mESmaC/Kadmium.git
cd Kadmium
cargo build --release -p kadmium-cli
```

The compiled binary will be located at `./target/release/kadmium` (or `.\target\release\kadmium.exe` on Windows).

### Checked release installers (next tagged release)

The release workflow and installer scripts are prepared for the next published `vX.Y.Z` tag. These instructions do not imply that a matching release already exists. Once its assets are published, run the installer from a checked-out repository and replace `vX.Y.Z` with that actual tag:

```bash
bash scripts/install.sh --version vX.Y.Z
# Add --launch to start the installed binary after verification.
```

```powershell
.\scripts\install.ps1 -Version vX.Y.Z
# Add -Launch to start the installed binary after verification.
```

The installers verify the downloaded archive against `SHA256SUMS`, install into a per-user version directory and print the executable path. They do not change `PATH`; use the printed path or add its directory yourself. Checksums detect a damaged or mismatched archive, and are not independent release signatures.

| Platform | Release archive |
| :--- | :--- |
| Windows x64 | `kadmium-windows-x86_64.zip` |
| Apple Silicon macOS | `kadmium-macos-arm64.tar.gz` |
| GNU Linux x64 | `kadmium-linux-x86_64.tar.gz` |
| GNU Linux ARM64 | `kadmium-linux-arm64.tar.gz` |

Windows/macOS archives also include the desktop companion. GNU Linux builds use Ubuntu 22.04 for x64 and Ubuntu 24.04 for ARM64 and require a compatible system runtime; these archives do not establish support for every distribution, musl, Intel macOS, Windows ARM or RISC-V. Build from source for other targets when their dependencies are available.

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

Inspect the hardware-fit recommendation without writing configuration or starting workers:

```bash
cargo run -p kadmium-cli -- launch --plan
```

Then launch against an inference backend you have already started:

```bash
cargo run -p kadmium-cli -- launch --inference-url http://127.0.0.1:8080
```

`launch` creates `kadmium.toml` if absent, preserves an existing file and starts the coordinator. The recommendation is a capacity estimate, not a downloaded model or a measured performance result. `serve` also accepts backend configuration:

```bash
cargo run -p kadmium-cli -- serve --bind 127.0.0.1:7848 --inference-url http://127.0.0.1:8080
```

Open **`http://127.0.0.1:7848`** for the Control Room. Live hardware is the default; add `--demo` to `serve` for a simulated seven-node fleet. The coordinator can run without a backend for control-plane work, but real `/v1` inference then returns `503`. Read [`/api/v1/inference/status`](api-reference.md#configure-an-inference-backend) and call `/v1/models` to distinguish configured state from a reachable backend. See the [launch guide](guides/cli.md) for flags and configuration.

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

### Join devices on a trusted LAN

A new `launch` configuration listens on loopback, so other computers cannot join it. To accept peers, choose this coordinator's trusted-LAN address explicitly (replace the example IP with its real address):

```bash
kadmium launch --bind 192.168.1.10:7848 --inference-url http://127.0.0.1:8080
# On another machine:
kadmium join --coordinator http://192.168.1.10:7848
```

Automatic LAN discovery uses Kadmium UDP announcements on port `7849`, not mDNS. Peers can use `kadmium join` without `--coordinator` when these announcements reach them. The HTTP listener and worker/discovery ports must be reachable between the devices. Keep this on a trusted network or provide external access control; client inference API keys do not authenticate access to the coordinator.

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

## 5. Connect an inference client

Call `GET http://127.0.0.1:7848/v1/models` and choose a model ID reported by your backend. Client compatibility follows that backend's supported protocol and model features. Kadmium preserves complete request JSON and streams backend responses; it does not convert between OpenAI and Anthropic protocols.

### OpenAI-Compatible Endpoint (`POST /v1/chat/completions`)

```bash
curl -X POST http://127.0.0.1:7848/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_BACKEND_MODEL_ID",
    "messages": [
      {"role": "user", "content": "Say hello."}
    ],
    "max_tokens": 32
  }'
```

### Anthropic-Compatible Endpoint (`POST /v1/messages`)

```bash
curl -X POST http://127.0.0.1:7848/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "YOUR_BACKEND_MODEL_ID",
    "max_tokens": 32,
    "messages": [
      {"role": "user", "content": "Hello from an Anthropic-compatible client!"}
    ]
  }'
```

The backend must itself implement `/v1/messages` for the Anthropic example. OpenAI SDK base URLs end in `/v1`; Anthropic SDK base URLs use the origin `http://127.0.0.1:7848`. Some SDKs require a nonempty local `api_key` value; that client value is ignored by Kadmium. Configure the actual backend credential through `KADMIUM_INFERENCE_API_KEY`. Add `"stream": true` and use `curl -N` to see incremental events. See [API limits and error handling](api-reference.md#admission-credentials-and-errors).

---

## Next Steps

- Explore every CLI subcommand in the [**CLI Command Reference**](./cli-reference.md).
- Learn how to combine heterogeneous hardware and `.ktb` shard banks in [**Hardware Setup, Asymmetric Sharding & Polite Governance**](./guides/hardware-and-sharding.md).
- Run distributed training and **Dream-RSI** offline causal replay in [**Distributed Training, RL & Dream-RSI Evolving Worlds**](./guides/training-and-dream-rsi.md).
