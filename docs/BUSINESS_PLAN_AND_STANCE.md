# 🧪 Kadmium (`Cd-48`) — Master Business Plan, Pricing & Settlement Model, and Ethical Stance

> **"If the AI companies and other companies want compute, then they can pay the people who would otherwise suffer from scaling operations for it."**
> — *The Core Thesis of Kadmium (`Cd-48`) & Kadmium Reactor*

This document is the single canonical record of **Kadmium's moral, ethical, and civic stance**, **Pay-In pricing (what we charge users and institutional buyers)**, **Pay-Out settlement mechanics (how and what we pay hardware hosts)**, and **how Kadmium makes money while remaining 100% solvent and steward-owned**.

Every rule, formula, and split in this document is also enforced directly in the Rust codebase ([`crates/kadmium-core/src/supply.rs`](../crates/kadmium-core/src/supply.rs), [`crates/kadmium-core/src/reactor_clearinghouse.rs`](../crates/kadmium-core/src/reactor_clearinghouse.rs), and [`crates/kadmium-sharding/src/spot_provisioner.rs`](../crates/kadmium-sharding/src/spot_provisioner.rs)).

---

## Part I — Moral, Ethical & Civic Stance ("Why Kadmium Exists")

### 1.1 Reversing the Regressive Hyperscaler AI Tax ("Pay the People")
Today's hyperscale AI infrastructure model socializes massive costs onto everyday households while privatizing 100% of the revenue to a handful of trillion-dollar corporations:
1. **Exploding Household Utility Bills (`9.3×` PJM Capacity Price Surge = `$16.4B` Ratepayer Hit):**
   - According to the independent market monitor for `PJM Interconnection` (serving 65 million people across 13 states + D.C.), hyperscale datacenter load growth caused PJM capacity auction clearing prices to surge **`9.3×`** from **`$28.92 / MW-day` (`2024/2025`)** to **`$269.92 / MW-day` (`2025/2026`)** and **`$329.17–$333.44 / MW-day` (`2026/2027`)**.
   - Datacenter load alone accounted for **`$6.3B–$6.5B` (~40%)** of the single-auction capacity bill—translating directly into **15%–25%+ monthly electric bill hikes** for families and small businesses who never touch those AI clusters.
2. **Taxpayer Giveaways for Concrete Warehouses with Near-Zero Permanent Jobs:**
   - Municipalities routinely grant hundreds of millions in sales- and property-tax exemptions to hyperscale AI campuses that employ only a handful of permanent staff once construction ends, shifting local school and infrastructure tax burdens onto residents.
3. **Hardware & Memory Crowding-Out:**
   - Hyperscalers buy out regional dark fiber, utility transformer queues, and global DRAM/NAND/wafer capacity (`HBM3e`, `GDDR7`, `LPDDR5X`), doubling consumer memory and GPU prices for students, gamers, universities, and small businesses.

**Kadmium's Moral Position:** Instead of extracting ratepayer and taxpayer subsidies to build another concrete warehouse for silicon that already exists in people's homes, dorms, offices, and labs, **pay the everyday people who already bought the silicon and already pay the electric bill.**

---

### 1.2 Grid-Zero Environmental Stewardship ("We Don't Need Any Additional Datacenters")
1. **Zero New Cooling Water & Zero New Concrete:**
   - Hyperscale evaporative cooling towers consume `~1.8 liters` of water per kWh. By contrast, more than **30 million air-cooled GPUs, Apple Silicon Macs, handheld gaming PCs, and campus computers** already sit on desks plugged in at `<10%` average utilization—requiring **`0 L` of additional cooling water** and **`0 kg` of new concrete**.
   - Kadmium aggregates dormant local, campus, and community silicon *first*, and reaches for already-built datacenter spot capacity only when local silicon overflows.
2. **Fighting the `62 Million Tonnes/Yr` Global E-Waste Crisis ("Closet Silicon"):**
   - The world generates **`62 million metric tonnes`** of e-waste annually (UN Global E-waste Monitor, only `22.3%` formally recycled), while manufacturing a single new computer and monitor consumes `1,500 liters` of water and `240 kg` of fossil fuels.
   - Kadmium turns dormant **"Closet Silicon"**—older smartphones (`8–16 GB LPDDR4X/5`, `UFS 3.1/4.0` flash, `15–45 TOPS` NPU), retired laptops, older desktops, SBCs, and decommissioned enterprise servers (**Sentinel & Scholar Donations**)—into productive, paid AI compute nodes.

---

### 1.3 Anti-Executive-Greed & Open-Book Steward-Ownership (`75 / 20 / 5` Charter)
Venture-backed AI clouds (`CoreWeave`, `Modal`, `Together`, `AWS`) must extract **`60%–75%` gross margins** to service `$10B` datacenter debt, bloated executive compensation packages, and aggressive VC multiples.

Kadmium operates under an **Open-Book, Cost-Plus Steward-Ownership Charter** ([`CompanySustainabilitySummary`](../crates/kadmium-core/src/supply.rs) & [`ReactorClearinghouseLedgerReport`](../crates/kadmium-core/src/reactor_clearinghouse.rs)):
- **`75.0%` Directly to Hardware Hosts:** Paid strictly from verified buyer billings.
- **`20.0%` Capped Cost-Plus Operations (Zero Executive Greed):**
  - Operational draw is tied strictly to **modest, family-supporting living-wage salaries proportional to headcount** (so the founder and every team member can comfortably support their families—never bloated 6- or 7-figure executive greed) plus the exact cost of running `relay.kadmium.ai`, `Verde` referee nodes, and Stripe settlement.
  - **Automatic Cost-Plus Ceiling Overflow Sweep:** Whenever network volume grows so that the `20%` operational share exceeds actual monthly living-wage payroll + relay infrastructure (`monthly_cost_plus_operational_ceiling_usd`), **100% of the surplus is automatically swept into the Community STEM & Host Grant Pool**.
- **`5.0%` Base Community STEM, Academic & Small-Business Grant Pool:**
  - Permanently funds free compute grants for local high-school/community-college STEM programs, `.edu` students (`$50/mo` free off-peak Scholar Compute Credits), independent open-source researchers, and main-street small businesses.

---

### 1.4 Respecting User Hardware, Safety & Right-to-Use
1. **The User Owns Their Machine First (`< 200 ms` Polite Auto-Yield):**
   - When a host launches a game, 3D application, or heavy local job, [`OsSwapchainHysteresisEngine`](../crates/kadmium-core/src/os_hooks.rs) yields `90%` of GPU/VRAM resources in **`< 200 ms`** with zero financial penalty and enforces a `15-second` swapchain cooldown before reclaiming capacity.
2. **Mandatory Battery & Power Safety Advisory:**
   - Displayed prominently whenever a user configures compute leasing ([`SupplyMarketplaceSnapshot::battery_and_power_safety_advisory`](../crates/kadmium-core/src/supply.rs)):
     > *"BATTERY & POWER ADVISORY: If your device has a removable battery, we strongly recommend removing the battery and plugging directly into a stable AC/DC power source before leasing compute. Most battery-powered laptops and phones operate normally without a battery when connected to wall power—eliminating unnecessary battery cycle wear and preventing potential lithium-ion swelling risks."*
   - For non-removable batteries, [`ClosetSiliconLongevityGovernor`](../crates/kadmium-core/src/reactor_clearinghouse.rs) enforces a **`55%` BMS state-of-charge clamp** and thermal sweet-spot locking (`61.5°C`).
3. **Zero-Wear Read-Only Flash Protection (`99.86%` Wear Reduction):**
   - `.ktb` weight banks are pinned via read-only `mmap` while ephemeral activations and KV caches stay in `LPDDR / DDR / zram`—preventing SSD/UFS flash wear (`12.5+ yr` flash lifespan).
4. **Right-to-Use Open Hardware (Never a Locked Appliance):**
   - Unlike single-purpose crypto miners that become paperweights, Kadmium's `$249` `Cd-48` Reactor Box runs mainline Debian/Ubuntu Linux with full root access, Docker, M.2 NVMe, and 40-pin GPIO so the owner can use it for Home Assistant, a local NAS, Plex, or embedded engineering projects alongside Kadmium.

---

### 1.5 Privacy by Construction & Honest Solvency (No Crypto Token / No Ponzi Math)
1. **Zero Plaintext Egress by Construction ([`NodeTrustTier`](../crates/kadmium-core/src/lib.rs)):**
   - `Layer 0` (token embeddings / raw prompt ingress) and `Layer N-1` (LM head / token sampling) are architecturally pinned to the user's own `TrustedCore` hardware.
   - External community hosts and spot GPUs are restricted strictly to **blind middle layers (`1..N-2`)** wrapped in an **Orthogonal Fast Walsh-Hadamard Transform (`FWHT`) Blind-Activation Shield (`CosSim = 0.0489`)**. No external node ever sees raw prompts, token IDs, or decoded text.
2. **No Speculative Crypto Token & Strict `No-Demand = No-Compute-Payout` Solvency:**
   - Kadmium settles in real **USD** and **1.25× Compute Credits**—never a volatile token.
   - Hosts are paid strictly when a paying buyer's verified workload executes on their node, and every earnings quote transparently deducts estimated electricity cost (`$0.14/kWh`).

---

## Part II — Pay-In Model: How Much We Charge Users & Buyers

### 2.1 Software, CloudShield & Team Governance Tiers

| Tier | Track | Price | Target User / Buyer | What Is Included | Kadmium Gross Margin |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **1. `Kadmium Local`** | Self-Hosted OSS | **`$0` forever** *(MIT / Apache-2.0)* | Individual devs, gamers & homelabbers | Unlimited local nodes (PC, Mac, Laptop, USB4 Handheld, SBC, NAS), `<200 ms` gaming auto-yield, 6-digit SAS pairing, RAID-5 P2P striping, OpenAI / Anthropic / MCP gateway. **Zero token markup on local silicon, ever.** | **N/A** (`$0` CAC Acquisition Funnel) |
| **2. `CloudShield Pro`** | Individual / Prosumer | **`$29 / month`** | AI coding power users (`Claude Code`, `Cursor`) & LoRA builders | Includes **`$40 / month` in Burst Credits** (`~53 hrs` of L40S middle-layer burst), managed global WAN hole-punch + DERP relay, and `<340 ms` `CloudShield` failover on game launch or fiber drop. | **62% – 78%** |
| **3. `Kadmium Lab & Team`** | University Labs & AI Startups | **`$199 / month`** *(+`$499` White-Glove Setup)* | Academic labs & seed/Series-A teams (up to 25 nodes) | Includes **`$250 / month` in shared Team Burst Credits**, `SubClusterPartitioner` slices (`prod`, `train-DiLoCo`, `RL-Dream-RSI`), `SubClusterRbacEngine` scoped `kdm_live_...` keys with per-student/engineer spend caps, Prometheus & W&B telemetry. | **75% – 85%** |
| **4. `Private Office Fabric`** | Regulated SMBs | **`$499 / month`** *(+`$990` Onboarding)* | Legal (`ABA Rule 1.6`), Healthcare (`HIPAA`), Engineering & Defense (`CMMC`) SMBs (5–50 machines) | Turns existing office Macs/PCs into a zero-CapEx compliant AI cloud; air-gapped LAN mode or blind FWHT middle-layer burst (`Layer 0` & `Layer N-1` never leave the building); HIPAA BAA, CMMC boundary & audit log export. | **85% – 92%** |

---

### 2.2 Prepaid Burst Credit Packs & On-Demand Hourly Compute Rates

Users who do not want a monthly subscription—or subscribers topping up their pool—can purchase **non-expiring Prepaid Burst Credit Packs**:
- **`$25` Starter Pack** (`$25.00` credit value)
- **`$100` Builder Pack** (`+15%` bonus = **`$115.00` credit value**)
- **`$500` Lab & Studio Pack** (`+15%` bonus = **`$575.00` credit value**)

#### Hourly Pay-In Rates Charged Against Burst Credits:

| Burst Resource Tier | Use Case | Kadmium Wholesale / Fulfillment Cost | **Retail Rate Charged to Buyer** | **Kadmium Gross Margin** |
| :--- | :--- | :---: | :---: | :---: |
| **Fractional `"Just-Enough-Boost"` (`8 GB` Blind Middle Slice)** | Local 16–24 GB GPU is only `8 GB` short for a 70B model (`L56..L75` spill) | `$0.048 / hr` | **`$0.07 / hr`** | **31.4%** |
| **RTX 4090 (24 GB VRAM)** | Interactive middle-layer decode & QLoRA fine-tuning | `$0.29 / hr` | **`$0.45 / hr`** | **35.5%** |
| **NVIDIA L40S (48 GB VRAM)** | 70B–235B middle-layer burst & `<340 ms` `CloudShield` vault | `$0.52 / hr` | **`$0.75 / hr`** | **30.7%** |
| **NVIDIA A100 (80 GB HBM2e)** | Heavy long-context prefill & Sharded-Island DiLoCo outer steps | `$0.82 / hr` | **`$1.19 / hr`** | **31.1%** |
| **NVIDIA H100 (80 GB HBM3)** | Frontier 405B–671B middle-layer burst & hyper-training (`1F1B / ZBH1`) | `$1.65 / hr` | **`$2.29 / hr`** | **27.9%** |
| **20-Slot Phone Farm Box (`160 ARM64 Cores`)** | `2,400+` parallel `WasiMicroSandbox` compiler/unit-test RL rollouts | `$0.27 / hr` *(Host 75%)* | **`$0.36 / hr`** | **25.0%** |

---

### 2.3 Dynamic 5-Factor Market-Clearing Formula for Supply Grid Nodes

When a buyer's workload runs on a community or commercial host in the **Kadmium Supply Grid**, [`SupplySettlementEngine::evaluate_cluster_supply`](../crates/kadmium-core/src/supply.rs) computes the exact **Gross Buyer Clearing Rate ($\text{BuyerRate}_{\text{active}}$)** across five hardware dimensions:

$$\text{BuyerRate}_{\text{active}} = \eta_{\text{net}} \cdot \left( \$0.0065 \cdot M_{\text{VRAM}} + \$0.0048 \cdot M_{\text{UMA}} + \$0.00035 \cdot M_{\text{RAM}} + \$0.000085 \cdot \text{BW}_{\text{GB/s}} + \$0.00055 \cdot \text{TFLOPS} + S_{\text{bucket}} \right)$$

Where:
- $M_{\text{VRAM}}, M_{\text{UMA}}, M_{\text{RAM}}$: Leased fast GPU VRAM, Unified Memory, and System DDR capacity in GiB.
- $\text{BW}_{\text{GB/s}}$: Effective memory bandwidth in GB/s (the primary physical determinant of token decode speed).
- $\text{TFLOPS}$: Effective FP16/BF16 compute throughput (governs prefill, training, and RL rollouts).
- $S_{\text{bucket}} = \$0.0035 \cdot \text{NVMe}_{\text{TiB}} + \$0.0015 \cdot \min(\text{Uplink}_{\text{Gbps}}, 10)$: Decentralized `.ktb` storage & seeding contribution.
- $\eta_{\text{net}} \in [0.92, 1.02]$: Network latency and uptime reliability multiplier.

---

### 2.4 Open-Hardware & Pre-Certified Node Pricing

| Hardware Product | Price | Est. BOM Cost | Specs & Included Value | Target Buyer |
| :--- | :---: | :---: | :--- | :--- |
| **`Kadmium Cd-48 Reactor Box`** *(Cost-Plus Sidecar)* | **`$249`** *(one-time)* | `~$185` | `32 GB LPDDR5` + `M.2 NVMe` + `Dual 2.5GbE` + `USB-C` (`16W` battery-free wall power, open Debian Linux). Acts as a `$0/hr` local desktop memory sidecar, home server/NAS, and `+$19.40/mo` net Supply Grid earner (`~12.8 mo` payback). | Desktop GPU owners, homelabbers & students |
| **`Cd-48 Control Rod Node`** *(Pre-Certified Dev Kit)* | **`$699`** *(one-time, or `$1` refundable Founder reservation)* | `~$390` | `32 GB RK3588 ARM64` + `2 TB Gen4 NVMe` pre-loaded with `1.5 TB` of `.ktb` models, **`$200` of included Burst Credits**, and a Founder Pro badge. | AI builders, labs & small offices |

---

## Part III — Pay-Out Model: How & What We Pay Out to Supply Grid Hosts

### 3.1 Two Host Compensation Modes ([`SupplyPayoutMode`](../crates/kadmium-core/src/supply.rs))
Hosts leasing their idle hardware (`kadmium lease --hours 12 --payout credits|usd`) choose between two settlement modes at any time:
1. **Mode 1 — Direct USD Payout (`UsdDirectPayout`):**
   - Settled in **USD via Stripe Connect or USDC**.
   - The host receives **`75.0%` of the Gross Buyer Clearing Rate** for every cryptographically verified active compute slice, plus a standby storage retainer (`$0.001–$0.015/hr`) funded by `kdm://pool-bucket/` storage fees while seeding `RS(8,2)` erasure-coded `.ktb` chunks.
2. **Mode 2 — Closed-Loop Kadmium Cloud Credits (`KadmiumCloudCredits { bonus_multiplier: 1.25 }`):**
   - The host receives **`1.25×` their USD payout rate** in Kadmium Cloud burst and failover credits (`+25%` purchasing-power bonus).
   - Ideal for researchers, students, and developers who lease their rig overnight so they can burst 405B–671B models or DiLoCo training runs for free during the day.

---

### 3.2 Cryptographic Proof-of-Compute Settlement (`Verde RepOps` + Blind Canaries)
Every micro-slice executed on a Supply Grid node is audited before a single cent moves from buyer escrow ([`ReactorEscrowAndStewardLedgerEngine`](../crates/kadmium-core/src/reactor_clearinghouse.rs)):
- **Deterministic `Verde RepOps` FNV-1a Digests:** Every forward or backward stage returns a deterministic reduction digest. Disputed stages are pinpointed in $\lceil \log_2 L \rceil + 1$ referee evaluations via [`VerdeFraudProofEngine`](../crates/kadmium-runtime/src/lib.rs).
- **`2.0%` Blind FWHT Honey-Token Canaries:** `2%` of micro-batches carry pre-solved canary activations that are mathematically indistinguishable (`CosSim = 0.0489`) from normal traffic.
- **Zero-Tolerance Slashing Rule:** If a host returns a corrupted or lazy/spoofed activation digest, **the buyer is charged `$0.00` and the host is paid `$0.00`**.

---

### 3.3 Transparent Per-Node Host Payout Table (12 hrs/day Lease, Deducting `$0.14/kWh` Power)

Below is the exact per-node valuation produced by [`SupplySettlementEngine`](../crates/kadmium-core/src/supply.rs) across a 7-node personal/lab cluster (assuming a 12 hr/day lease window and realistic marketplace demand duty cycles of `32%–65%`):

| Hardware Node | Buyer Active Rate (`100%`) | **Host Active Rate (`75%`)** | Realistic Demand Duty Cycle | Demand-Weighted Effective USD/hr | Monthly Gross USD (`360 hrs`) | Monthly Power (`@ $0.14/kWh`) | **Host Net Monthly Cash Profit** | **Host Monthly `1.25×` Credits** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Desktop PC (`RTX 4090 24GB`, `290W`)** | `$0.324/hr` | **`$0.243/hr`** | `62%` | `$0.152/hr` | `$54.60/mo` | `-$9.73/mo` | **`+$44.87/mo`** | **`$68.25/mo`** |
| **M4 Mac Mini Pro (`48GB UMA`, `38W`)** | `$0.252/hr` | **`$0.189/hr`** | `65%` | `$0.124/hr` | `$44.57/mo` | `-$1.37/mo` | **`+$43.20/mo`** | **`$55.71/mo`** |
| **Gaming Laptop (`RTX 4070M 8GB`, `95W`)** | `$0.097/hr` | **`$0.073/hr`** | `48%` | `$0.036/hr` | `$12.89/mo` | `-$2.67/mo` | **`+$10.22/mo`** | **`$16.11/mo`** |
| **Gaming Laptop (`RTX 3070M 8GB`, `95W`)** | `$0.103/hr` | **`$0.077/hr`** | `48%` | `$0.038/hr` | `$13.62/mo` | `-$2.67/mo` | **`+$10.95/mo`** | **`$17.03/mo`** |
| **ARM64 SBC Cluster (`RK3588 16GB`, `16W`)** | `$0.073/hr` | **`$0.055/hr`** | `45%` | `$0.026/hr` | `$9.19/mo` | `-$0.47/mo` | **`+$8.72/mo`** | **`$11.49/mo`** |
| **Home / Lab NAS (`4TB NVMe`, `55W`)** | `$0.049/hr` | **`$0.037/hr`** | `55%` | `$0.025/hr` | `$8.85/mo` | `-$2.16/mo` | **`+$6.69/mo`** | **`$11.06/mo`** |
| **Legacy Laptop (`ThinkPad AVX2`, `28W`)** | `$0.016/hr` | **`$0.012/hr`** | `32%` | `$0.005/hr` | `$1.74/mo` | `-$0.66/mo` | **`+$1.08/mo`** | **`$2.18/mo`** |
| **7-Node Cluster Total** | — | — | — | **`$0.40/hr`** | **`$145.46/mo`** | **`-$19.72/mo`** | **`+$125.74/mo` Net** | **`$181.83/mo` Credits** |

*(Note: Under 100% continuous active duty—as shown in the interactive calculator on `/site`—an RTX 4090 + M4 Mac Mini Pro + RTX 4070M Laptop + SBC + NAS cluster clears up to `$0.92/active hr` in USD or `$1.15/active hr` in `1.25×` Cloud Credits.)*

---

## Part IV — How Kadmium Makes Money (Revenue Streams, Margins & Solvency Math)

```mermaid
flowchart LR
    subgraph PayIn["1. Pay-In (Buyers & Subscribers)"]
        Sub["SaaS & Governance\n$29 Pro | $199 Lab | $499 SMB"]
        Packs["Prepaid Burst Packs\n$25 | $100 | $500"]
        Inst["B2B / Academic / Gov\nEscrow Workloads"]
    end

    subgraph Clearinghouse["2. Kadmium Reactor Double-Entry Clearinghouse"]
        Verify["Verde RepOps + 2% Blind Canary\n(Failed Digest = $0 Billed / $0 Paid)"]
        Split["Verified Dollar Split:\n• 75% Host Payout\n• 20% Capped Cost-Plus Ops\n• 5% + Surplus -> STEM Grants"]
    end

    subgraph Fulfillment["3. Fulfillment & Margin Flywheel"]
        Spot["Day-1 Wholesale Spot Bridge\n(28% - 35.5% Gross Margin)"]
        GridUSD["Supply Grid (Direct USD)\n(25.0% Platform Take-Rate)"]
        GridCred["Supply Grid (1.25x Credits)\n(36.3% Net Margin, $0 Cash Out)"]
    end

    PayIn --> Verify --> Split --> Fulfillment
```

### 4.1 Four Diversified Revenue Engines
1. **Engine 1 — Multi-User Governance & Compliance Subscriptions (`62%–92%` Gross Margin):**
   - Following the proven **Tailscale + Ollama Cloud** open-core playbook, single-user local clustering is 100% free (`$0` CAC), while multi-user RBAC (`kdm_live_...` keys), sub-cluster slicing, `<340 ms` `CloudShield` failover, and HIPAA/ABA/CMMC compliance convert teams to `$29/mo`, `$199/mo`, and `$499/mo` recurring plans.
2. **Engine 2 — Day-1 Wholesale Spot Arbitrage (`28%–35.5%` Gross Margin):**
   - Before a single community host is required in a region, [`SpotOrderbookBidEngine`](../crates/kadmium-sharding/src/spot_provisioner.rs) fulfills middle-layer burst requests using wholesale datacenter spot APIs (`$0.29–$1.65/hr`) and bills the user's prepaid Burst Credit balance at `$0.45–$2.29/hr`.
3. **Engine 3 — Two-Sided Supply Grid Take-Rate & `1.25×` Credit Flywheel (`25.0%` to `36.3%` Margin):**
   - **On Direct USD Payouts:** Kadmium retains **`25.0%`** (`20%` Capped Cost-Plus Ops + `5%` STEM Grant Pool) and pays **`75.0%`** to the host.
   - **On `1.25×` Cloud Credit Payouts:** `$0` upfront cash leaves Kadmium. When the host later redeems those `1.25×` credits against wholesale/community capacity (`~68%` fulfillment cost), Kadmium nets a **`36.3%` gross margin** (**`31.8%` blended margin** across a 40% USD / 60% Credit mix).
4. **Engine 4 — MSP Channel Partner Program & Cost-Plus Hardware:**
   - Regional Legal/Medical IT Managed Service Providers (MSPs) deploy `Private Office Fabric` (`$499/mo` + `$990` setup) across client offices and keep `100%` of the `$990` setup + **`25%` recurring rev-share (`$125/mo`)**, netting Kadmium **`$374/mo` per SMB client at `$0` direct sales CAC**.

---

### 4.2 90-Day Revenue, COGS & Cash-Flow Model

| Metric | Day 14 (Outbound + Alpha) | Day 30 (Post-Launch) | Day 60 (Supply Flywheel) | Day 90 (Labs + SMB Channel) |
| :--- | :---: | :---: | :---: | :---: |
| **Active Free Local Clusters** | 40 | 650 | 2,400 | 6,000 |
| **Prepaid Burst Credit Packs Sold (`$25` / `$100`)** | 35 (`$1,625`) | 190 (`$8,550`) | 520 (`$23,400`) | 1,150 (`$51,750`) |
| **`CloudShield Pro` Subscribers (`$29/mo`)** | 15 (`$435/mo`) | 85 (`$2,465/mo`) | 240 (`$6,960/mo`) | 520 (`$15,080/mo`) |
| **`Kadmium Lab & Team` (`$199/mo` + `$499` setup)** | 5 (`$3,490`) | 15 (`$4,980`) | 40 (`$10,455`) | 95 (`$23,895`) |
| **`Private Office Fabric` SMBs (`$499/mo` + `$990` setup)** | 1 (`$1,489`) | 4 (`$3,976`) | 12 (`$9,948`) | 28 (`$19,912`) |
| **Total Monthly Cash Collected** | **`$7,039`** | **`$19,971`** | **`$50,763`** | **`$110,637`** |
| **Wholesale Spot, Host Payouts & Relay COGS** | `$780` | `$3,900` | `$9,200` | `$18,400` |
| **Net Gross Margin (Cash Basis)** | **`$6,259` (88.9%)** | **`$16,071` (80.5%)** | **`$41,563` (81.9%)** | **`$92,237` (83.4%)** |

---

## Part V — 4-Segment Institutional Buyer Positioning Summary

| Buyer Segment | Why Hyperscalers Fail Them | How Kadmium Wins the Contract | Verified Buyer ROI |
| :--- | :--- | :--- | :--- |
| **1. University & Academic STEM Labs** | `$4–$6.50/hr` H100s & 6-month NAIRR queues while 50–300 campus lab PCs/Macs sit idle every night. | Unifies existing campus PCs/Macs at `$0` CapEx (`<200 ms` student login yield) + earns `1.25×` Reactor Credits overnight to burst 70B–671B evals & DiLoCo runs. | **Stretches NSF / NIH / DoE grant budgets `3.2×–4.0×`**. |
| **2. Regulated SMBs (`Legal`, `Medical`, `Engineering`)** | Public APIs violate HIPAA / ABA privilege; dedicated server appliances cost `$25,000+`. | Pools office PCs/Macs + spills only missing `25%` middle layers (`$0.07/hr` `"Just-Enough-Boost"`) wrapped in blind FWHT shielding (`Layer 0` & `N-1` stay on-prem). | **`-94.1%` vs. full cloud GPU**, `9.34×` faster than CPU offload, **100% on-prem privacy**. |
| **3. AI Startups & Agentic SaaS** | Burning 65%–75% of runway idling `$4/hr` GPUs waiting on CPU compiler/test RL rewards. | Disaggregates `100,000` parallel `WasiMicroSandbox` compiler/test rollouts onto Phone Farm Boxes & SBCs + DeMo $1{,}000\times$ compressed training. | **Cuts RL post-training & batch bills by `55%–75%`**. |
| **4. Government & Municipal Agencies** | Strict sovereignty (`ITAR` / `CMMC` / `CJIS`), tight budgets, citizen backlash over utility rate hikes, and warehouses of retired IT hardware. | 100% Pure Rust zero-C-dependency sovereign fabric running air-gapped across existing municipal desktops and retired servers with `Verde RepOps` audit logs. | **`$0` new datacenter CapEx** and converts government e-waste into sovereign AI capacity. |

---

## Part VI — Codebase Verification Cross-Reference

Every economic, pricing, and ethical invariant in this document is implemented and unit-tested in the workspace:

- **5-Factor Valuation, `75/20/5` Steward Split & Battery Safety Advisory:** [`crates/kadmium-core/src/supply.rs`](../crates/kadmium-core/src/supply.rs) (`test_supply_5_factor_valuation_and_company_sustainability`)
- **Double-Entry Clearinghouse, Cost-Plus Ceiling Overflow Sweep, Closet-Silicon Longevity & `$249` `Cd-48` Reactor Box:** [`crates/kadmium-core/src/reactor_clearinghouse.rs`](../crates/kadmium-core/src/reactor_clearinghouse.rs) (`test_reactor_double_entry_clearinghouse_and_cost_plus_overflow_sweep`)
- **Wholesale Spot Orderbook & Fractional `"Just-Enough-Boost"` (`$0.07/hr`):** [`crates/kadmium-sharding/src/spot_provisioner.rs`](../crates/kadmium-sharding/src/spot_provisioner.rs) (`test_elastic_local_just_enough_boost_planner`)
- **Live REST Endpoints Serving These Models:**
  - `GET /api/v1/supply/marketplace` & `POST /api/v1/supply/configure`
  - `GET /api/v1/reactor/clearinghouse`
  - `GET /api/v1/cluster/elastic-boost`
  - `GET /api/v1/ga/governance`
