# Getting Started with Kadmium

This guide walks you through installing **Kadmium**, profiling your local hardware, spinning up a heterogeneous cluster across your home or lab devices, and running your first distributed inference, training, or **Dream-RSI** workload.

---

## 1. Installation

Kadmium compiles into a single dependency-free native binary (`kadmium`) across Windows, macOS (Apple Silicon), Linux (`x86_64`), and ARM64/RISC-V single-board computers.

### Option A: Download Prebuilt Binary or Desktop Companion App
Grab the latest release for your OS from the GitHub Releases page or `https://kadmium.ai/download`:
- **Windows (`x86_64-pc-windows-msvc`):** `kadmium-x86_64-pc-windows-msvc.exe`, with `kadmium-desktop.exe` (the desktop app) next to it
- **macOS Apple Silicon (`aarch64-apple-darwin`):** `kadmium-aarch64-apple-darwin.tar.gz`, which includes `kadmium-desktop`
- **Linux Workstation / NAS / Cloud (`x86_64-unknown-linux-gnu`):** `kadmium-x86_64-unknown-linux-gnu.tar.gz`
- **ARM64 SBCs — Orange Pi 5 / RK3588 / Raspberry Pi 5 (`aarch64-unknown-linux-gnu`):** `kadmium-aarch64-unknown-linux-gnu.tar.gz`

### Option B: Build from Source (100% Pure Rust)
Requires Rust `1.85+` (`rustup`):

```bash
git clone https://github.com/kadmium-ai/kadmium.git
cd kadmium
cargo build --release -p kadmium-cli
```

The compiled binary will be located at `./target/release/kadmium` (or `.\target\release\kadmium.exe` on Windows).

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

Start the local Kadmium coordinator and Strata-compatible AI Gateway on port `7848`:

```bash
cargo run -p kadmium-cli -- serve --port 7848
```

Open **`http://127.0.0.1:7848`** in your browser. From the interactive dashboard you can:
- Inspect your **7-node heterogeneous cluster topology** (Gaming PC, Apple Silicon Mac Mini, Gaming Laptop, Home NAS Vault, Legacy Laptop, ARM SBC Bramble, and Friend's Remote PC).
- Trigger live **Responsibility Shifts** (`Host Gaming Mode` auto-yield, `Cloud Burst Node Joined`, or `Coordinator Quorum Failover`) and watch CERN FTS3 P2P transfers rebalance layers without dropping active requests.
- Inspect the **100-Node Sub-Cluster Pool Slicer**, **TPU/CIM Token vs. Weight Streaming** analyzer, **Dream-RSI Evolving Worlds** discovery tree, and **CloudShield** spot arbitrage planner.

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

### Local LAN Devices (Zero-Config mDNS)
Any device on the same local Wi-Fi or Ethernet subnet running the Kadmium daemon or Desktop Companion App automatically discovers the active coordinator via mDNS (`_kadmium._udp.local`).

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

## 5. Run Inference with Any OpenAI or Anthropic Client

Once `kadmium serve` is running, point any existing SDK, IDE plugin (Cursor, Continue, Claude Code), or chat UI (OpenWebUI) to `http://127.0.0.1:7848`:

### OpenAI-Compatible Endpoint (`POST /v1/chat/completions`)

```bash
curl -X POST http://127.0.0.1:7848/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3-671B-MoE-Q4_K_M",
    "messages": [
      {"role": "user", "content": "Explain how Kadmium shards MoE experts across NVMe and VRAM."}
    ]
  }'
```

### Anthropic-Compatible Endpoint (`POST /v1/messages`)

```bash
curl -X POST http://127.0.0.1:7848/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3-671B-MoE-Q4_K_M",
    "messages": [
      {"role": "user", "content": "Hello from an Anthropic-compatible client!"}
    ]
  }'
```

---

## Next Steps

- Explore every CLI subcommand in the [**CLI Command Reference**](./cli-reference.md).
- Learn how to combine heterogeneous hardware and `.ktb` shard banks in [**Hardware Setup, Asymmetric Sharding & Polite Governance**](./guides/hardware-and-sharding.md).
- Run distributed training and **Dream-RSI** offline causal replay in [**Distributed Training, RL & Dream-RSI Evolving Worlds**](./guides/training-and-dream-rsi.md).
