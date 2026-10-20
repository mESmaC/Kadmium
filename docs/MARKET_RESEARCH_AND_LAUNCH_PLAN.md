# Kadmium (`Cd-48`) — Market Research & Rapid-Revenue Launch Plan

> **Primary Objective:** Go from **$0 to first dollar in $\le 14$ days** and reach **$10,000+ MRR within 60 days** without getting trapped in a two-sided marketplace "cold-start" problem.

---

## 1. Executive Diagnosis: Why Pure P2P & Pure OSS Startups Stall at $0 (and How Kadmium Wins)

Most infrastructure startups in the local/decentralized AI space get stuck in one of two traps:
1. **The Open-Source Local Trap (`EXO`, `llama.cpp RPC`):** Users download the free binary, cluster two Mac Minis on their desk, star the GitHub repo, and pay **$0 forever** because there is no natural transactional monetization hook inside a purely local tool.
2. **The Two-Sided Marketplace Cold-Start Trap (Early P2P GPU networks):** If you wait until 1,000 community gamers list their GPUs before you can sell compute to researchers, you have zero supply on Day 1—and supply hosts churn within 48 hours if there isn't immediate utilization.

### Kadmium's Asymmetric Monetization Wedge
Kadmium (`Cd-48`) already has the architectural pieces to **bypass both traps on Day 1**:
- **Local-First Hook (`$0` CAC):** Users install `kadmium` for free to pool their PC + Mac + laptop + NAS (`--cadmium`) and connect Claude Code / Cursor to `http://127.0.0.1:7848/v1`.
- **Day-1 Guaranteed Supply via Wholesale Spot Arbitrage:** When a user's local cluster is 20 GB short of running a 70B–671B model—or when they launch a game and their PC auto-yields (`< 200 ms`)—they click **"Attach Burst GPU"** (`--cherenkov`). On Day 1, `SpotOrderbookBidEngine` (`crates/kadmium-sharding/src/spot_provisioner.rs`) fulfills that burst request using wholesale RunPod / Vast.ai / Lambda spot APIs (`$0.28–$0.79/hr`) and bills the user's prepaid Kadmium balance at a **25–32% gross margin**.
- **Margin Expansion via `1.25×` Credit Supply Flywheel:** As users enable **Supply Grid Hosting** (`kadmium lease --credits true`) to earn `1.25×` Kadmium Cloud credits overnight, Kadmium routes burst middle-layer shards onto community hosts instead of wholesale datacenter spot—raising gross margin from **~28% to >60%** while **$0 cash leaves the company** for credit-based hosts.

---

## 2. 2026 Market & Competitive Landscape

### 2.1 Live Competitor Pricing Benchmarks (Q3/Q4 2026)

| Hardware / Tier | Hyperscalers (AWS / GCP) | RunPod (Community / Spot) | Vast.ai (Spot Floor / Median) | Shadeform / Prime Intellect | **Kadmium Wholesale Cost** | **Kadmium Retail Burst Rate** | **Kadmium Gross Margin** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 4090 (24 GB)** | N/A | `$0.34 – $0.89/hr` | `$0.18 – $0.45/hr` | `$0.35 – $0.70/hr` | **`$0.29/hr`** | **`$0.45/hr`** | **35.5%** |
| **NVIDIA L40S (48 GB)** | `$2.70+/hr` | `$0.79 – $1.09/hr` | `$0.38 – $0.65/hr` | `$0.79 – $0.95/hr` | **`$0.52/hr`** | **`$0.75/hr`** | **30.7%** |
| **NVIDIA A100 (80 GB)** | `$3.67 – $4.10/hr` | `$1.19 – $1.79/hr` | `$0.48 – $1.15/hr` | `$1.10 – $1.85/hr` | **`$0.82/hr`** | **`$1.19/hr`** | **31.1%** |
| **NVIDIA H100 (80 GB)** | `$4.50 – $6.90/hr` | `$1.99 – $2.89/hr` | `$1.10 – $2.40/hr` | `$1.69 – $2.80/hr` | **`$1.65/hr`** | **`$2.29/hr`** | **27.9%** |

### 2.2 Competitive Positioning Matrix

| Platform | Primary Focus | Local Heterogeneous Sharding? | `<200ms` Gaming / Battery Yield? | Blind Middle-Layer Cloud Burst? | Zero-GPU `Dream-RSI` / WASI RL? | How They Monetize |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXO Labs (`exo`)** | Local Mac / Linux clustering | Yes (Uniform/Mac focus) | No | No (Local only) | No | Enterprise consulting ("EXO for Business") |
| **Ollama / vLLM** | Single-host or homogeneous server | Homogeneous tensor/pipeline | No | No | No | Cloud hosting / Enterprise SaaS |
| **RunPod / Vast.ai** | Raw cloud GPU container rental | No (100% Cloud) | N/A | No (Full model in cloud; sees raw prompts) | No | Per-second GPU markup |
| **Prime Intellect** | Decentralized cluster + RL post-training | Datacenter / homogeneous nodes | No | No | Online GPU RL (`$0.02–$1.00/1M tok`) | GPU marketplace + per-token RL |
| **Kadmium (`Cd-48`)** | **Local-First Heterogeneous Mesh + Hybrid Burst + Supply Grid** | **Yes (GPU + Mac UMA + RAM $q^*$ + NVMe `.ktb`)** | **Yes (`DXGI`/`IOKit`/`DRM` `<200ms`)** | **Yes (`L0` & `LN-1` stay local; burst only `L1..N-2`)** | **Yes (ARM SBC / CPU offline replay + WASI)** | **Prepaid Burst Credits + `CloudShield` Subscriptions + Supply Grid Spread** |

---

## 3. Why Users Will Pay Kadmium Instead of Renting a Raw Vast.ai / RunPod GPU

1. **You Only Rent the Missing Middle Layers (50–70% Lower Bill):**
   - To run a 120 GB model on RunPod, you must rent **2× A100 80GB (`~$2.60/hr`)** for the *entire* model, while your `$3,000` local RTX 4090 PC and `$2,000` MacBook Pro sit idle on your desk.
   - With Kadmium, your local PC + Mac host **75 GB of layers for `$0.00/hr`**, and you burst **only the remaining 45 GB of middle layers** onto a single L40S for **`$0.75/hr`**—saving **`$1.85/hr` (71%)**.
2. **Zero-Trust Prompt Privacy (Blind Middle-Layer Execution):**
   - Developers using cheap community GPUs on Vast.ai or RunPod Community Cloud cannot send proprietary company code or private customer prompts there because the host machine can inspect plaintext memory.
   - Kadmium's `NodeTrustTier::VerifiedPeerMiddleLayersOnly` pins token embeddings (`Layer 0`), the LM head (`Layer N-1`), and the tokenizer **strictly on the user's local machine**. The external burst GPU only ever receives quantized intermediate INT8 activation tensors (`BlockwiseInt8Tensor`) and is verified by `Verde` $O(\log_2 L)$ canary bisection.
3. **Zero Container Setup & `< 400 ms` `CloudShield` Failover:**
   - Spinning up a raw Vast.ai pod takes 5–15 minutes of Docker pulls and 100+ GB of Hugging Face weight downloads every time a spot instance is preempted.
   - Kadmium mounts pre-compiled `64 KiB` page-aligned `.ktb` layer slices via HTTP Byte-Range / FTS3 in seconds and automatically fails over in `< 400 ms` if a spot node is preempted.
4. **"No New Datacenters" ESG & Institutional Mandate:**
   - With municipal moratoria and power-grid opposition delaying hyperscale AI datacenters across the US and EU in 2025–2026, universities, public-sector agencies, and sustainability-conscious companies have an immediate mandate to **maximize existing on-premise hardware utilization** before buying cloud capacity.

---

## 4. Three-Track Monetization Architecture (Ranked by Speed to Cash)

### Track 1: Prepaid Burst Credit Packs + `CloudShield Pro` (Target: First Dollar in Days 1–14)
- **Why Prepaid Credits First:** Selling **$25 / $100 / $500 Prepaid Burst Packs** collects cash **upfront on Day 1** before a single GPU-hour of wholesale cost is incurred.
- **Pricing Tiers:**
  1. **Kadmium Local (`$0` forever):** Unlimited local nodes, LAN discovery, OpenAI/Anthropic/MCP gateway.
  2. **Prepaid Burst Credits (`$25`, `$100`, `$500`):** Pay-as-you-go middle-layer burst GPUs (`$0.45/hr` RTX 4090 class, `$0.75/hr` L40S 48GB, `$1.19/hr` A100 80GB). Includes `15%` bonus credits on the `$100` pack if auto-reload is enabled.
  3. **`CloudShield Pro` (`$29/mo`):** Includes **$15/mo of burst credits**, managed global WAN hole-punching + mTLS DERP relay across home/office networks, and `< 400 ms` automatic `CloudShield` failover when your home PC reboots, loses internet, or enters Gaming Mode.

### Track 2: University Labs & Bootstrapped AI Teams (`$199/mo` + `$499` Onboarding) (Target: Days 7–30)
- **The Buyer:** Every university CS/AI lab, robotics group, and seed-stage AI startup has **8 to 40 heterogeneous machines** (a mix of RTX 3090/4090 towers, A6000 workstations, Mac Studios, and Linux servers) where GPUs sit idle 80% of the week because researchers manually claim whole machines over SSH/Slurm.
- **The Offer (`Kadmium Lab & Team` — `$199/mo`):**
  - Pools all lab machines into one fabric with `SubClusterPartitioner` (`Prod`, `Staging`, `Train-DiLoCo`, `RL-Dream-RSI`, `Batch`).
  - Issues slice-scoped `kdm_live_...` API keys per student/engineer via `SubClusterRbacEngine` with hard hourly cloud burst caps so no student can accidentally run up a `$5,000` AWS bill.
  - Native Prometheus (`/metrics`) and Weights & Biases (`/api/v1/telemetry/wandb`) export.
- **Fast-Cash Accelerator (`$499` "Founding Lab Onboarding"):**
  - A 45-minute live setup call where we run `kadmium probe` across their lab machines, configure their `kadmium.toml` slices, compile their primary model into `.ktb` on their NAS, and include **$100 in Burst Credits**. Only **10 Founding Labs = $4,990 in immediate cash + $1,990/mo recurring MRR**.

### Track 3: Disaggregated RL & `Dream-RSI` Post-Training Cloud (Target: Days 21–60)
- **The Buyer:** Teams running GRPO / RLOO post-training for coding agents, math provers, or custom domain models.
- **The Pitch:** *"Stop paying `$2.50/hr` on H100s to wait on Python unit tests and compiler sandboxes. Kadmium runs deterministic WASI verifiers and `Dream-RSI` MCTS offline replay across cheap CPU/SBC nodes and bursts only the policy learner step."*

---

## 5. 30-Day Launch & Revenue Execution Playbook

### Phase 1: Days 1–5 — Monetization Wiring ("Ready to Take Cards")
1. **Self-Serve Checkout & Credit Ledger on `/site`, `/`, and `/companion`:**
   - Wire 3 instant checkout flows: **$25 Starter Burst Pack**, **$29/mo `CloudShield Pro`**, and **$199/mo `Kadmium Lab & Team`** (`+$499` Founding Lab Setup).
2. **Wholesale Spot API Fulfillment Bridge:**
   - Connect `SpotOrderbookBidEngine` in `kadmium-gateway` to a master RunPod / Vast.ai API account with a strict per-key spend limiter.
3. **One-Line Installer & Prebuilt Binaries:**
   - Publish precompiled GitHub Release binaries (`linux-x86_64`, `linux-aarch64`, `macos-arm64`, `windows-x86_64.exe`).

### Phase 2: Days 6–12 — Direct Outbound to 10 "Founding Labs" (Target: First $2,000–$5,000)
1. **Target 150 High-Probability Leads:**
   - 75 university labs / independent groups publishing on `DiLoCo`, `MoE offloading`, `GRPO`, or `local LLMs`.
   - 75 indie AI builders active in `r/LocalLLaMA` and `exo` / `vLLM` / `llama.cpp` GitHub issues who own multi-device setups.
2. **Goal:** Close **5 Founding Labs at $499 ($2,495)** + **20 Indie Builders on $25–$29 packs ($550)** = **~$3,045 revenue before public launch**.

### Phase 3: Days 13–20 — Public Viral Launch (`Show HN`, `r/LocalLLaMA`, X/Twitter, MCP Directories)
- **`r/LocalLLaMA` Launch Post:** `[Project] Kadmium (Cd-48): Single Rust binary that shards 671B MoE across a gaming PC + Mac Mini + NAS + RK3588 SBC — with <200ms auto-yield when you launch a game` (accompanied by a 90-second split-screen video showing `< 200 ms` gaming auto-yield and 1-click middle-layer spot burst).
- **`Show HN` Launch Post:** `Show HN: Kadmium – Heterogeneous local/cloud LLM sharding and offline RL in pure Rust`.
- **MCP Ecosystem Distribution:** List `kadmium-mcp` on the official MCP Registry, Smithery, and Cursor Directory so agent users can burst compute from inside Claude Code and Cursor.

### Phase 4: Days 21–30 — Igniting the Supply Grid Credit Flywheel
- Prompt free `Kadmium Local` users in the Companion App to enable **Supply Grid Hosting overnight (`kadmium lease --credits true`)** to earn **`1.25×` Kadmium Cloud Credits**—replacing wholesale RunPod/Vast spot spend with community silicon and expanding gross margin above **70%**.

---

## 6. 90-Day Financial & Unit Economics Projection

| Milestone | Timeline | Active Free Clusters | Paid `CloudShield Pro` (`$29/mo`) | Paid `Lab / Team` (`$199/mo`) | Monthly Burst Volume (GPU-hrs) | Gross Revenue (MRR + Burst) | Wholesale Spot & Relay COGS | **Net Gross Profit** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Design Partners** | **Day 14** | 40 | 15 (`$435`) | 5 (`$995` + `$2,495` setup) | 1,200 hrs (`$900`) | **`$4,825`** | `$650` | **`$4,175` (86%)** |
| **Phase 2: Public Launch** | **Day 30** | 650 | 85 (`$2,465`) | 15 (`$2,985`) | 6,500 hrs (`$4,875`) | **`$10,325`** | `$3,100` | **`$7,225` (70%)** |
| **Phase 3: Supply Flywheel** | **Day 60** | 2,400 | 240 (`$6,960`) | 40 (`$7,960`) | 22,000 hrs (`$16,500`) | **`$31,420`** | `$7,800` *(40% community credit supply)* | **`$23,620` (75%)** |
| **Phase 4: Scale & RL Teams** | **Day 90** | 6,000 | 520 (`$15,080`) | 95 (`$18,905`) | 55,000 hrs (`$41,250`) | **`$75,235`** | `$15,900` *(60% community credit supply)* | **`$59,335` (79%)** |
