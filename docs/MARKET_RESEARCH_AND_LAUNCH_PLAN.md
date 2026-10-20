# Kadmium (`Cd-48`) — Deep-Dive Market Research & Rapid-Revenue Launch Plan

> **Primary Objective:** Go from **$0 to first dollar in $\le 14$ days** and scale to **$10,000–$90,000+ MRR within 90 days** by combining a zero-CAC open-source local hook with immediate prepaid middle-layer burst credits, Tailscale/Ollama-style B2B team subscriptions, and a non-crypto `1.25×` compute-credit supply flywheel.

---

## 1. Executive Diagnosis: Why Pure P2P & Pure OSS Startups Stall at $0 (and How Kadmium Wins)

Most infrastructure startups in the local and decentralized AI space get stuck in one of three traps:
1. **The Open-Source Local Trap (`EXO Labs`, `llama.cpp RPC`):** Users download the free binary, cluster two Mac Minis on their desk, star the GitHub repo, and pay **$0 forever** because there is no built-in transactional monetization hook inside a purely local tool.
2. **The Two-Sided Marketplace Cold-Start Trap (Early P2P GPU networks):** Waiting until 1,000 community gamers list their GPUs before selling compute to researchers means zero supply on Day 1—and supply hosts churn within 48 hours if there isn't immediate utilization.
3. **The Crypto-Verification Latency Trap (`Gensyn`, `Akash`, `io.net`):** Token-driven DePIN networks suffer from speculative node spoofing ("utilization gap") or heavy cryptographic proof overhead that stalls real-time token streaming.

### Kadmium's Asymmetric Monetization Wedge
Kadmium (`Cd-48`) bypasses all three traps on Day 1:
- **Local-First Hook (`$0` CAC):** Users install `kadmium` for free to pool their PC + Mac + laptop + NAS (`--cadmium`) and point Claude Code, Cursor, or Cline to `http://127.0.0.1:7848/v1`.
- **Day-1 Guaranteed Supply via Wholesale Spot Arbitrage:** When a user's local cluster is 20 GB short of running a 70B–671B model—or when they launch a game and their PC auto-yields (`< 200 ms`)—they click **"Attach Burst GPU"** (`--cherenkov`). On Day 1, `SpotOrderbookBidEngine` (`crates/kadmium-sharding/src/spot_provisioner.rs`) fulfills that burst request using wholesale RunPod / Vast.ai / Lambda spot APIs (`$0.29–$0.82/hr`) and bills the user's prepaid Kadmium balance at a **28–35% gross margin**.
- **Margin Expansion via `1.25×` Credit Supply Flywheel:** As users enable **Supply Grid Hosting** (`kadmium lease --credits true`) to earn `1.25×` Kadmium Cloud credits overnight, Kadmium routes burst middle-layer shards onto community hosts instead of wholesale datacenter spot—raising gross margin from **~30% to >65%** while **$0 cash leaves the company** for credit-based hosts.

---

## 2. Deep-Dive Competitive Landscape & Teardowns (2025–2026)

### 2.1 Live GPU & Serverless Pricing Benchmarks (Q3/Q4 2026)

| Hardware / Tier | Hyperscalers (AWS / GCP) | Managed Serverless (Modal / Baseten) | RunPod (Community / Spot) | Vast.ai (Spot Floor / Median) | **Kadmium Wholesale Cost** | **Kadmium Retail Burst Rate** | **Kadmium Gross Margin** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 4090 (24 GB)** | N/A | N/A | `$0.34 – $0.89/hr` | `$0.18 – $0.45/hr` | **`$0.29/hr`** | **`$0.45/hr`** | **35.5%** |
| **NVIDIA L40S (48 GB)** | `$2.70+/hr` | `$1.95 – $2.40/hr` | `$0.79 – $1.09/hr` | `$0.38 – $0.65/hr` | **`$0.52/hr`** | **`$0.75/hr`** | **30.7%** |
| **NVIDIA A100 (80 GB)** | `$3.67 – $4.10/hr` | `$2.80 – $3.50/hr` | `$1.19 – $1.79/hr` | `$0.48 – $1.15/hr` | **`$0.82/hr`** | **`$1.19/hr`** | **31.1%** |
| **NVIDIA H100 (80 GB)** | `$4.50 – $6.90/hr` | `$3.95 – $6.50/hr` | `$1.99 – $2.89/hr` | `$1.10 – $2.40/hr` | **`$1.65/hr`** | **`$2.29/hr`** | **27.9%** |

### 2.2 Competitor-by-Competitor Teardown: What Works & Where They Fail

#### 1. `Ollama Cloud` & `LM Studio Bionic` (Local-to-Cloud Hybrids)
- **What they proved (2025–2026):** Both Ollama and LM Studio proved that local-first developers will pay recurring monthly subscriptions for seamless cloud overflow:
  - **Ollama Cloud Pricing:** `Free ($0)` $\to$ `Pro ($20/mo, includes $60 in cloud token credits, 3 concurrent requests)` $\to$ `Max ($100/mo, $300 in credits, 10 concurrent)` $\to$ `Team ($500/mo, $1,000 shared credits + RBAC)`.
  - **LM Studio Bionic:** Free local runtime + paid team/organization cloud credit pools with Zero Data Retention (ZDR).
- **Where both fail:** In Ollama Cloud and LM Studio Bionic, execution is **binary (all-local OR all-cloud)**. If a model is 15 GB too large for your desktop, 100% of the model runs in their cloud—meaning your `$3,000` local GPU sits completely idle and your **raw plaintext prompt is transmitted to a remote server**.
- **Kadmium's Edge:** Kadmium is the only platform that **splits a single model across your local hardware + cloud burst simultaneously** (`Layer 0` + `Layer N-1` + 65% of middle layers on your local PC+Mac for `$0/hr`, and only the remaining 35% of middle layers on cloud burst)—delivering **mathematical zero-plaintext privacy** and **3× lower cloud credit burn**.

#### 2. `Tailscale` (The $1.5B Open-Core Mesh Conversion Blueprint)
- **What they proved:** Tailscale built a `$1.5B+` business on top of open-source WireGuard (and alongside open-source `Headscale`) by making personal/homelab mesh networking 100% free and gating **multi-user team governance** (`SSO`, `RBAC ACLs`, audit logs, and managed DERP relays) behind per-seat/team tiers.
- **How Kadmium applies the Tailscale Playbook:**
  - Single-user `Kadmium Local` is 100% free and open source (`MIT/Apache-2.0`).
  - The moment a **second lab member, engineer, or employee** needs shared access to the cluster (`SubClusterRbacEngine`, scoped `kdm_live_...` keys, per-user cloud spend caps, managed zero-config WAN hole-punching/DERP), the organization converts to **`Kadmium Team & Lab` (`$199–$499/mo`)**.

#### 3. `Salad.com` (Consumer Gaming PC Cloud)
- **What they proved:** Tens of thousands of gamers ("Salad Chefs") across 190 countries willingly run a background agent on RTX 3090/4090/5080/5090 PCs without crypto tokens in exchange for store credits and PayPal payouts.
- **Where Salad fails for frontier LLMs:**
  - **Single-Node VRAM Wall:** Salad schedules monolithic Docker containers onto individual consumer PCs (capped at 16–24 GB VRAM). It cannot shard a 70B–671B model across a user's local machines + remote nodes.
  - **Zero Prompt Privacy:** Sending an inference container to a residential Salad host exposes plaintext user prompts and weights in the host's RAM/VRAM.
  - **Brutal Preemption Cold Starts:** When a gamer launches a game, the container is killed and another host must pull a 15–30 GB Docker image from scratch.
- **Kadmium's Edge:** Kadmium uses consumer GPUs **strictly for blind middle layers (`L1..N-2`)** via `NodeTrustTier::VerifiedPeerMiddleLayersOnly`, streams `64 KiB` page-aligned `.ktb` layer slices in seconds, and shifts layers in **`< 200 ms`** when a host starts gaming.

#### 4. `Modal`, `Baseten`, `Together AI` & `Fireworks AI` (Serverless vs. Dedicated Endpoints)
- **What they proved:** Engineering teams gladly pay **`$3.95/hr` to `$6.50/hr` per H100** on Dedicated Endpoints purely to host custom fine-tunes/LoRAs, avoid public rate limits, and eliminate cold-start weight downloads.
- **Kadmium's Edge:** Kadmium's `.ktb` HTTP Byte-Range slicer + local NAS seed vault keeps custom weights and `DistributedQLoRAAdapter` states warm 24/7 at **`$0/hr`**, bursting middle layers at **`$0.45–$0.75/hr`** only during active batches (**85–92% cheaper than Together/Fireworks/Baseten Dedicated Endpoints**).

---

## 3. Hard Technical & Economic Reality Checks (Stress-Testing Our Assumptions)

### 3.1 Reality Check #1: WAN Activation Physics & Why Multi-Hop Residential Rings Fail for Decode
In pipeline parallelism, each stage boundary transmits an intermediate activation tensor of size:
$$\text{Payload} = B \times S \times d_{\text{model}} \times \text{bytes\_per\_elem}$$
For `Llama-3-70B` / `DeepSeek-V3` ($d \approx 7168\dots 8192$):
- **Autoregressive Decode ($S = 1$):** `~16 KB` (`BF16`) or **`~8 KB` (`Hivemind BlockwiseInt8Tensor`)**. Bandwidth is tiny, **but WAN latency ($RTT$) is additive across hops**. Chaining 5 residential volunteer PCs over the internet (`40 ms` per hop, as in `Petals`) adds $200\text{ ms/token}$ of pure network delay, capping decode at **`< 5 tok/s`**.
- **Long-Context Prefill ($S = 4096$):** **`~33 MB` (`INT8`) to `~67 MB` (`BF16`)** per boundary hop. Over a typical residential cable upload link (`20–35 Mbps`), transferring `33 MB` takes **8–13 seconds per hop**!

#### Critical Product & Routing Rule for Kadmium:
Never chain multiple low-upload residential WAN nodes in series for interactive single-stream decode. Instead, Kadmium enforces a **Two-Tier Topology Rule**:
1. **Interactive Inference (`< 1 ms` LAN + Single-Hop Datacenter Burst + Local Speculative Drafts):**
   - Multi-node sharding happens across the user's **high-speed Local LAN** (`1–10 Gbps` Ethernet / Wi-Fi 6E / Thunderbolt, where `33 MB` prefill takes `< 30 ms`).
   - If cloud burst is attached, `ShardPlanner` pins a **single contiguous middle block (`L_a..L_b`)** onto **one datacenter-connected spot GPU** (`10 Gbps` symmetrical link)—requiring only **one WAN round-trip per forward pass**—while the local SBC/laptop runs **Leviathan Speculative Drafts ($K=4$ tokens verified in parallel per single WAN pass)**, boosting effective WAN throughput from `~12 tok/s` to **`32–42 tok/s`**.
2. **Residential Supply Grid Hosts $\to$ Routed to High-Compute / Low-Bandwidth Workloads:**
   - Residential gamer PCs (`20–50 Mbps` upload) are assigned to **Parallel GRPO Rollouts + WASI Verification**, **Sharded-Island `DiLoCo + DeMo` Training** (`85×` DCT-compressed sparse momentum synced once every 500 inner steps), and **Asynchronous Batch Document Processing**.

---

### 3.2 Reality Check #2: When Does Kadmium Beat Cheap Serverless APIs (`OpenRouter` / `DeepSeek API`)?

| Customer Scenario | Cheap Serverless API (`OpenRouter`) Cost | Dedicated Cloud Endpoint (`Together` / `Baseten`) | **Kadmium (`Local + Middle-Layer Burst`)** | **Winner & Why** |
| :--- | :--- | :--- | :--- | :--- |
| **Casual Solo Chat** (`30 prompts/day`, ~40k tokens/day, public weights, zero privacy need) | **`$1.50 / mo`** | `$2,160+ / mo` | `$0/mo` (if fits locally) or `$15/mo` burst | **OpenRouter** (unless model fits 100% on user's local PC+Mac for `$0`). **Do not pitch paid burst to casual chatters.** |
| **Custom Fine-Tuned / QLoRA 70B+ Model** (Custom weights not hosted on shared serverless APIs) | *Impossible* (Shared APIs only host stock public models) | **`$2,160 – $4,680 / mo`** (`$3.00–$6.50/hr` 24/7 dedicated GPU) | **`$29/mo` + `$0.45–$0.75/hr` active burst (`~$110/mo`)** | **Kadmium wins by 95%** (Base `.ktb` + LoRA adapter stays warm on local NAS/RAM at `$0/hr`; burst only on active requests). |
| **Heavy Multi-Agent Coding / Repo Indexing** (`Claude Code` / `Cursor` / `OpenCode` running 5–15M tokens/day across a 5-dev team) | **`$900 – $2,400 / mo`** (+ 429 rate limits) | `$2,500+ / mo` | **`$199/mo` Team Plan + `$120/mo` burst (`$319/mo`)** | **Kadmium wins by 70–85%** (Pools team's existing MacBooks + desktop GPUs on office LAN; bursts only peak overflow). |
| **Regulated Legal / Medical / Defense Workflows** (HIPAA PHI, Attorney-Client Privilege, Classified IP) | *Disqualified* (Plaintext leaves building to 3rd-party routers) | Requires `$25k+/yr` Enterprise BAA + VPC | **`$499/mo` Private Office Fabric** (`Layer 0` & `Layer N-1` never leave building) | **Kadmium wins on both compliance and 80% lower TCO.** |
| **GRPO RL Post-Training & Code Verification** (10,000+ compiler/test rollouts per epoch) | `$1.00 / 1M` training tokens + no custom WASI env | `$1,800+ / mo` (GPUs stall waiting on CPU tests) | **Free SBC/CPU WASI + `Dream-RSI` replay + `$0.75/hr` learner burst** | **Kadmium wins by 65–75%** by disaggregating CPU sandboxes from GPU learner steps. |

---

### 3.3 Reality Check #3: Supply-Side Host Electricity Math (`kWh`) & The "Apple Silicon + Studio Fleet" Advantage
Why do residential RTX 4090/5090 hosts on `Salad.com` churn? **Electricity math:**
- An **RTX 4090 rig** under 100% continuous load draws **`~500W` (`0.50 kW`)**; an **RTX 5090 rig** draws **`~650W` (`0.65 kW`)**.
- At the US average of `$0.15/kWh`, power costs **`$0.075–$0.098/hr`**. In Europe, California, or the US Northeast (`$0.32–$0.42/kWh`), power costs **`$0.16–$0.27/hr`**—wiping out Salad's `$0.18–$0.25/hr` cash payout!

#### How Kadmium Solves Supply-Side Economics:
1. **The Apple Silicon Unified Memory Arbitrage (`40W–85W` Total System Draw):**
   - An **M4 Pro Mac Mini (64 GB UMA)** or **Mac Studio M2/M3/M4 Max/Ultra (128–512 GB UMA)** draws only **`40W to 85W` under full LLM load**—costing just **`$0.006 to $0.013/hr` in electricity** (nearly **10× less power** than an RTX 5090) while holding **2× to 8× more model weights**!
2. **Rewarding Individual Hosts in `1.25×` Kadmium Cloud Credits (`$0` Cash Outlay):**
   - Giving developer hosts **`$0.90–$1.15/hr` in `1.25×` Kadmium Cloud Credits** (which they spend during the workday to burst 671B models or multi-node training jobs) delivers **4× higher perceived value** than raw cash while costing Kadmium **`$0` in upfront cash**.
3. **Commercial Overnight Fleets (VFX Studios, Architecture Firms, Esports LAN Centers & University Labs):**
   - Following the proven B2B studio model (`ParalonCloud` / `Hosted.ai`), a single VFX animation studio, architecture firm, or esports arena has **40 to 200 RTX 4080/4090/A6000 workstations on symmetrical 10GbE fiber and cheap commercial power (`$0.08/kWh`)** that sit 100% idle from 7:00 PM to 8:00 AM.

---

## 4. Synthesized Pricing & Monetization Architecture (`Tailscale` + `Ollama Cloud` Model)

| Tier | Price | Target Buyer | Included Compute & Features | Kadmium Gross Margin |
| :--- | :--- | :--- | :--- | :--- |
| **1. `Kadmium Local`** | **`$0` forever** | Solo devs & homelabbers | Unlimited local LAN nodes (`PC + Mac + NAS + SBC`), `<200ms` gaming auto-yield, OpenAI / Anthropic / MCP gateway. | **N/A ($0 CAC Funnel)** |
| **2. `Burst Credit Packs`** | **`$25` / `$100` / `$500`** *(one-time prepaid)* | Devs needing instant overflow without a subscription | Pay-per-second middle-layer burst GPUs (`$0.45/hr` 4090, `$0.75/hr` L40S 48GB, `$1.19/hr` A100 80GB). Non-expiring; `+15%` bonus on `$100+`. | **31% (Wholesale Spot) $\to$ 68% (Community Supply)** |
| **3. `CloudShield Pro`** | **`$29 / mo`** | Coding agent power users (`Claude Code` / `Cursor`) & custom LoRA builders | Includes **`$40/mo` in Burst Credits** (`~53 hrs` of L40S middle-layer burst), managed global WAN hole-punch + DERP relay, and `<400ms` `CloudShield` failover. | **62%–78%** *(blended utilization + credit breakage)* |
| **4. `Kadmium Lab & Team`** | **`$199 / mo`** *(+$499 White-Glove Setup)* | University AI labs & seed/Series-A startups (up to 25 nodes) | Includes **`$250/mo` in Shared Team Burst Credits**, `SubClusterPartitioner` slices (`Prod`/`Train`/`RL`), `SubClusterRbacEngine` (`kdm_live_...` keys & spend caps), Prometheus/W&B. | **75%–85%** |
| **5. `Private Office Fabric`** | **`$499 / mo`** *(+$990 Onboarding)* | Legal, Healthcare (HIPAA) & Defense SMBs (up to 50 office Macs/PCs) | Turns existing office Macs/PCs into a zero-CapEx private AI appliance; air-gapped or blind middle-layer burst (`L0` & `LN-1` pinned on-prem); audit logs & BAA. | **85%–92%** |

---

## 5. 90-Day Revenue, Margin & Cash-Flow Model

| Metric | Day 14 (Outbound + Alpha) | Day 30 (Post-HN / Reddit Launch) | Day 60 (Supply Flywheel Active) | Day 90 (B2B Labs + Legal/Med SMBs) |
| :--- | :--- | :--- | :--- | :--- |
| **Active Free Local Clusters** | 40 | 650 | 2,400 | 6,000 |
| **Prepaid Burst Credit Packs Sold (`$25` / `$100`)** | 35 packs (`$1,625`) | 190 packs (`$8,550`) | 520 packs (`$23,400`) | 1,150 packs (`$51,750`) |
| **`CloudShield Pro` Subscribers (`$29/mo`)** | 15 (`$435/mo`) | 85 (`$2,465/mo`) | 240 (`$6,960/mo`) | 520 (`$15,080/mo`) |
| **`Lab / Team` Subscribers (`$199/mo` + `$499` setup)** | 5 (`$3,490` cash) | 15 (`$4,980`) | 40 (`$10,455`) | 95 (`$23,895`) |
| **`Private Office Appliance` SMBs (`$499/mo` + `$990` setup)** | 1 (`$1,489` cash) | 4 (`$3,976`) | 12 (`$9,948`) | 28 (`$19,912`) |
| **Total Monthly Cash Collected** | **`$7,039`** | **`$19,971`** | **`$50,763`** | **`$110,637`** |
| **Wholesale Spot & DERP Relay COGS** | `$780` *(100% wholesale spot)* | `$3,900` *(80% wholesale spot)* | `$9,200` *(45% Mac/Studio supply)* | `$18,400` *(65% Mac/Studio supply)* |
| **Net Gross Profit (Cash Basis)** | **`$6,259` (88.9%)** | **`$16,071` (80.5%)** | **`$41,563` (81.9%)** | **`$92,237` (83.4%)** |
