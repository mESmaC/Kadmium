# Kadmium (`Cd-48`): Master Business Plan, Financial Ledger & Ethical Charter

**Document Version:** `v2.0 — Post-R4 Audit Alignment`  
**Canonical Numbers Date Stamp:** `2026-10-20` ([`docs/NUMBERS_2026-10.md`](NUMBERS_2026-10.md) & [`crates/kadmium-core/src/canonical_numbers.rs`](../crates/kadmium-core/src/canonical_numbers.rs))  
**Audit Reference:** [`docs/MARKET_RESEARCH_REVIEW_2026-10.md`](MARKET_RESEARCH_REVIEW_2026-10.md) (*R4 Market Research & Fact-Check Review*)  
**Executable Ledger Engines:** [`crates/kadmium-core/src/supply.rs`](../crates/kadmium-core/src/supply.rs) & [`crates/kadmium-core/src/reactor_clearinghouse.rs`](../crates/kadmium-core/src/reactor_clearinghouse.rs)

> **Process Invariant (Single Dated Source of Truth):**
> Every wholesale GPU rate, serverless token price floor, electricity tariff (`$0.182/kWh`), Steam-derived GPU count, DeMo compression range (`10×–85×`), credit redemption liability parameter, and hardware BOM figure in this document is locked to the **`2026-10-20` Canonical Numbers Ledger** ([`docs/NUMBERS_2026-10.md`](NUMBERS_2026-10.md)) and [`crates/kadmium-core/src/canonical_numbers.rs`](../crates/kadmium-core/src/canonical_numbers.rs). All throughput and buyer ROI tables are explicitly marked **Modelled** until automated `kadmium bench` hardware logs are published (`R4 §10`).

---

## 1. Executive Thesis & Verified Macro Context

### 1.1 The Civic & Economic Wedge: Pay the People Who Already Bought the Silicon
According to the International Energy Agency (*Energy and AI*), global datacenter electricity demand reached **`~415 TWh` in 2024** (`~1.5%` of global electricity) and is projected to more than double to **`~945 TWh` by 2030** in the base case (`R4 §8.1`). In the US Mid-Atlantic grid (PJM Interconnection), the 2026/2027 Base Residual Auction cleared at the federal cap of **`$329.17 / MW-day`** (`$16.1B` total capacity cost—a **`~9.4×` surge** from `$28.92 / MW-day` in 2024/2025), driven primarily by greenfield AI datacenter load additions (`R4 §8.2`). Residential utility ratepayers absorb those transmission and capacity surcharges on their monthly electric bills (`$0.182 / kWh` US residential average in October 2026, projected to reach `$0.187 / kWh` in 2027 per the US EIA *Short-Term Energy Outlook*, `R4 §7.9`) while receiving `$0.00` of the AI economy's revenue.

At the same time, an enormous fleet of capable consumer, prosumer, and institutional silicon already sits Manufactured, Paid-For, and Plugged Into Existing Wall Outlets:
- **Steam-Derived Consumer GPU Fleet (`2026-10-20`, `R4 §7.4`):** Scaling the September/October 2026 Steam Hardware Survey VRAM distribution against Steam's **`43.5M` concurrent user peak** (October 11, 2026) yields approximately:
  - **`~0.56M` GPUs with `32 GB` VRAM** (`~1.3%` share, RTX 5090 class)
  - **`~2.40M` GPUs with `24 GB` VRAM** (`5.50%` share, RTX 3090 / 4090 / RX 7900 XTX class)
  - **`~11.84M` GPUs with `16 GB` VRAM** (`27.21%` share)
  - **`~21.58M` cumulative GPUs with `12 GB+` VRAM** (`49.61%` share)
- **Unified-Memory Workstations & Edge Boards:** Millions of Apple M-series Macs (`16 GB – 192 GB` UMA at `120 – 800 GB/s`), AMD Ryzen AI Max+ 395 (`Strix Halo` `128 GB` UMA), and home/lab Linux servers sit at low utilization outside active work or gaming hours.

**Kadmium (`Cd-48`, atomic number 48—the neutron-absorbing control rod element in a nuclear reactor)** is a local-first heterogeneous clustering runtime and output-metered clearinghouse built around three rules:
1. **Free Forever on Your Own LAN (`Tier 0`):** Combining your own desktop, laptop, Mac mini, and home NAS over Thunderbolt/USB4, Ethernet, or Wi-Fi is **100% free forever (`$0.00`)** with no account, telemetry, or cloud dependency.
2. **Buyer-Constraint-First Clearing (`Tier 1 & Tier 2`):** When a workload needs more memory or throughput than local hardware provides, Kadmium either leases fractional capacity from the **Kadmium Supply Grid**—priced strictly per **delivered token or verified work unit at a 20–35% discount to serverless APIs**—or bursts into verified wholesale cloud spot GPUs priced transparently at **`WholesaleVerifiedIndex(class) × 1.25`**.
3. **The `75 / 20 / 5` Steward-Ownership Charter:** When external buyers pay for verified Supply Grid work, **`75%` of gross compute billings go directly to the hardware hosts**, **`20%` funds a capped Cost-Plus living-wage operations reserve**, and **`5%` (plus 100% of net surplus above the Cost-Plus cap) sweeps quarterly into the Community STEM, Academic & Small-Business Grant Pool**.

---

## 2. How Money Flows In (Pay-In Architecture) & Wholesale GPU Index (`2026-10-20`)

### 2.1 SaaS, Enterprise & Channel Tiers

| Tier | Target User | Monthly Price | What Is Included & How COGS Is Bounded |
| :--- | :--- | :--- | :--- |
| **1. Community LAN Core** | Hobbyists, students, home-lab tinkerers, `r/LocalLLaMA` | **`$0 / mo`** *(Free Forever)* | Full open-source `kadmium` CLI, native desktop companion, `4D` heterogeneous LAN sharding, `.ktb` zero-copy weight loader, USB4/Thunderbolt ring DMA, Polite Gaming Mode, and local OpenAI/Anthropic gateway (`:8080`). Zero Kadmium cloud COGS. |
| **2. CloudShield Pro** | Prosumers, indie AI devs, researchers with 16–64 GB local rigs | **`$29 / mo`** | Automated 25ms local DDR5 shadow fallback + 245ms cloud/grid failover, dynamic `4K -> 64K` KV-cache layer slider, and **`$35.00 / mo` in Indexed Burst & Supply Grid Credits**. **COGS Protection (`R4 §6.2`):** Credits draw down at the live `WholesaleVerifiedIndex(class) × 1.25` rate (or `$0.22 / 1M` batch tokens on the credit-paid community/studio grid). Workloads fitting 24–32 GB slices route first to credit-paid community supply, ensuring an active `$29/mo` subscriber never creates an unhedged `$50–$100` wholesale cash loss. |
| **3. Team / Lab Cluster** | AI startups, university labs, ML engineering teams (up to 15 seats) | **`$199 / mo`** *(flat per lab)* | Multi-user RBAC queue priorities, centralized `.ktb` S3/R2/NAS team weight registry, automated multi-provider spot bidding across 85+ providers, WSD checkpoint recovery, and verified `WASI` RL sandbox orchestration. |
| **4. Private Office Fabric (100% On-Premise LAN Mode)** | Regulated SMBs: law firms, medical/dental practices, defense subcontractors | **`$499 / mo`** *(per office)* | **100% Air-Gapped / On-Premise Office LAN Mode** (`Zero Egress`). Pools existing office PCs/Macs (`16–64 GB` each) into a local document/coding/transcription fabric inside the office firewall. Includes tamper-evident SHA-256 local audit logs and Active Directory / Okta SSO. *(See §6 for strict HIPAA / CMMC / ABA compliance boundaries.)* |
| **5. MSP Partner Channel** | Regional Managed IT Service Providers managing 5–50 SMB offices | **`$299 / mo`** *(wholesale per office)* | Multi-tenant MSP fleet console; MSP bills the end client `$499–$799/mo` for managed on-premise installation + SLA support, retaining `$200–$500/mo` recurring service margin while remitting `$299/mo` wholesale to Kadmium. |

### 2.2 Dynamic Wholesale GPU Index & Index-Plus-Spread Burst Pricing (`R4 §5.3, §6.2, §7.1`)

> **Why Static Burst Tables Fail (`R4 §6.2`):**
> A static retail table that prices an L40S 48 GB at `$0.75/hr` sits **below** every realistic verified wholesale provider in October 2026 (Spheron `$0.96/hr`, Modal `$1.95/hr`, 85-provider median `$1.64/hr`). Under a fixed `$0.75/hr` rate, a user redeeming `$40` of credits for `53` L40S hours would cost Kadmium `$51–$104` in wholesale cash against `$29` of revenue.

Kadmium eliminates fixed-rate inversion by pricing all cloud burst dynamically as:

$$\text{RetailBurstRate}(\text{class}) = \text{WholesaleVerifiedIndex}(\text{class}) \times (1 + \text{Spread}_{\text{25\%}})$$

refreshed daily from the multi-provider spot orderbook ([`crates/kadmium-sharding/src/spot_provisioner.rs`](../crates/kadmium-sharding/src/spot_provisioner.rs)) and [`crates/kadmium-core/src/canonical_numbers.rs`](../crates/kadmium-core/src/canonical_numbers.rs):

| GPU Class | 85-Provider Floor (Unverified / Preemptible) | Realistic Verified Wholesale Index (`2026-10-20`) | 85-Provider Market Median | Kadmium Indexed On-Demand Burst (`Index × 1.25`) | Kadmium Gross Spread (`$/hr` / `%` of Retail) | Dedicated Endpoint Reference (Together / Baseten / Fireworks) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 4090 (`24 GB`)** | `$0.33 / hr` (Salad) | **`$0.38 / hr`** (RunPod comm. `$0.34` – Spheron `$0.48`) | `$0.48 / hr` | **`$0.48 / hr`** | **`+$0.10 / hr`** (`20.8%` margin) | N/A (Consumer/Prosumer Tier) |
| **NVIDIA L40S (`48 GB`)** | `$0.55 / hr` (Novita) | **`$0.96 / hr`** (Spheron `$0.96`; Modal `$1.95`) | `$1.64 / hr` | **`$1.20 / hr`** | **`+$0.24 / hr`** (`20.0%` margin) | `$1.95 – $2.80 / hr` |
| **NVIDIA A100 (`80 GB`)** | `$0.89 / hr` | **`$1.46 / hr`** (Spheron `$1.43` – RunPod `$1.49`) | `$1.80 / hr` | **`$1.83 / hr`** | **`+$0.37 / hr`** (`20.2%` margin) | `$3.29 – $4.10 / hr` |
| **NVIDIA H100 (`80 GB`)** | `$1.80 / hr` | **`$2.20 / hr`** (Vast verified `$1.87` – Spheron `$2.65`) | `$3.99 / hr` | **`$2.75 / hr`** | **`+$0.55 / hr`** (`20.0%` margin) | `$5.49 – $7.00 / hr` |

### 2.3 Fractional Grid Micro-Boost (`"Just-Enough-Boost"`)
When a user's local desktop PC (e.g., an RTX 4090 `24 GB`) wants to run a `70B` model with a `32K` KV cache (`30.5 GB` total VRAM required), they do not need to rent an entire `$1.83/hr` (`$1.46/hr` wholesale) 80 GB A100 and leave their `$2,000` local GPU idle.
- Kadmium pins `75%` of the model (`L00..L55` + `L76..L79`) in local RTX 4090 VRAM and spills only the missing **`8.0 GB` middle-layer slice (`L56..L75`)** to a secondary LAN device (`$0.00/hr`) or a fractional Supply Grid peer (`~$0.07/hr` metered by tokens/layers).

---

## 3. How Money Flows Out (Pay-Out Architecture): Output-Delivered Supply Grid Pricing

### 3.1 Why Capacity-Based Pricing Fails & How Output-Delivered Clearing Fixes It (`R4 §6.3`)

> **The Physics & Market Constraint (`R4 §6.3`):**
> In October 2026, open-weight serverless inference clears at **`$0.21–$0.28 / 1M input tokens` and `$0.31–$0.42 / 1M output tokens`** for `DeepSeek-V3.2` (OpenRouter / GMICloud), and **`$0.95–$2.10 / 1M input` and `$3.00–$4.40 / 1M output`** for frontier-open models (`DeepSeek V4-Pro`, `Kimi K2.6`, `GLM-5.1`). Small 8B models clear at **`$0.05–$0.20 / 1M tokens`**.
>
> If a grid formula prices hardware by *what it is* (raw GiB of RAM installed) rather than *what it produces* (tokens or verified rollouts delivered per hour):
> - A `32 GB` RK3588 SBC with `~25 GB/s` LPDDR5 bandwidth decodes an 8B Q4 model at `~4 tok/s` (`~14,400 tokens/hr`). Paying that board `$0.16/hr` forces the buyer to pay **`~$11.11 / 1M tokens`**—**`55×–220×` more expensive than serverless** (`$0.05–$0.20/1M`).
> - An Apple M4 Pro Mac mini (`273 GB/s`) decoding a 70B shard at `~4 tok/s` (`~14,400 tokens/hr`) at `$0.252/hr` forces the buyer to pay **`~$17.50 / 1M tokens`** against `$0.28–$0.90/1M` on the open market.
> - Because **the buyer side is the binding constraint**, no rational buyer clears at `$11–$17 / 1M tokens`, actual duty cycles collapse to `0%`, and host payouts drop to zero.

### 3.2 The Output-Delivered Work-Unit Clearing Formula (`crates/kadmium-core/src/supply.rs`)
Kadmium prices Supply Grid capacity strictly per **verified work unit delivered** at a **20–35% discount to serverless APIs**, and derives the host's **`75%` cash share** directly from that delivered output:

$$\text{BuyerClearing}_{\text{hr}} = \underbrace{\left(\$0.22 \times \frac{\text{Tokens}_{\text{batch,hr}}}{10^6}\right)}_{\text{Batch LLM Inference / Eval}} + \underbrace{\left(\$0.14 \times \frac{\text{Rollouts}_{\text{wasi,hr}}}{10^3}\right)}_{\text{Deterministic WASI RL Verification}} + \underbrace{\left(\$0.008 \times \text{GiB}_{\text{seeded,hr}}\right)}_{\text{Merkle .ktb Shard Seeding}}$$

1. **Batch Token Throughput (`$0.22 / 1M tokens`):**
   - Determined by effective memory bandwidth $\text{BW}_{\text{eff}}$ (`GB/s`), micro-batch amortization (`batch=8–16` for high-bandwidth GPUs/UMA), and WAN uplink penalty $\phi_{\text{net}} = \min(1.0, \text{Uplink}_{\text{Mbps}} / 100.0)$.
   - **Strict Memory-Bandwidth Routing Gate ($\text{BW}_{\text{eff}} < 60\text{ GB/s}$):** Any device with memory bandwidth below `60 GB/s` (such as a `25.6 GB/s` RK3588 SBC or smartphone array) is **automatically gated out of dense LLM token decode (`0` batch decode tokens assigned)** and routed strictly to **deterministic CPU/NPU `WASI` RL sandbox verification, speculative draft trees, and `.ktb` Merkle shard seeding**, where its output is genuinely cost-competitive.
2. **Verified `WASI` RL Sandbox Rollouts (`$0.14 / 1,000 rollouts`):**
   - Bounded by CPU/compute throughput and ` Verde RepOps` deterministic hash verification—a workload that is compute- and sandbox-bound rather than memory-bandwidth-bound.
3. **`.ktb` Merkle Weight Shard Seeding (`$0.008 / GiB`):**
   - Bounded by the host's residential/campus upload bandwidth, undercutting hyperscaler egress (`$0.05–$0.09/GiB`) by `80–90%`.

### 3.3 Modelled Output-Delivered Host Payout & Buyer Clearing Table (`2026-10-20` Constants)

> **Modelled Capability Envelope Notice (`R4 §10`):**
> All throughput (`tok/hr`, `WASI rollouts/hr`), duty-cycle (`18%–35%`), and monthly net payout figures below are **Modelled** from [`crates/kadmium-core/src/supply.rs`](../crates/kadmium-core/src/supply.rs) using the US EIA October 2026 residential electricity rate of **`$0.182 / kWh`** (`18.2¢/kWh`). Because every LLM-capable node clears at **`$0.22 / 1M batch tokens`** (plus auxiliary WASI/seeding work), buyers always pay `~$0.22–$0.27 / 1M effective tokens`—below OpenRouter/GMICloud's `$0.28–$0.42 / 1M` floor.

| Hardware Profile | Assigned Grid Workload | Modelled Delivered Output / Hr | Buyer Clearing Rate (`$/hr` / Effective `$/1M tok`) | Host Gross Share (`75%` USD) | Incremental Wall Power & Cost (`@ $0.182/kWh`) | Host Net Cash / Hr (After Power) | Modelled Monthly Net Cash (`Duty Cycle` / `Hrs/Day`) | Overnight Credit Value (`1.25×` Index) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. `Cd-48` RK3588 32 GB NAS / Coordinator Node** (`25.6 GB/s` LPDDR5) | **WASI RL Verifier + `.ktb` NAS Seed + Coordinator** *(Gated out of LLM decode)* | `0` LLM tok/hr; `227` WASI rollouts/hr + `1.6 GiB/hr` `.ktb` seed | **`$0.044 / hr`** *(N/A — WASI & Seed only)* | `$0.033 / hr` | `+9 W` (`$0.0016 / hr`) | **`+$0.031 / hr`** | **`~$6.80 / mo`** (`30%` duty, `7.2h/day` active demand) | **`$0.055 / hr`** (`~$11.90/mo` in Burst Credits) |
| **2. Apple Mac mini M4 Pro (`64 GB` UMA, `273 GB/s`)** | **64 GB High-Context MoE / 70B Batch Shard + WASI** | `583k` batch tok/hr + `380` WASI rollouts/hr | **`$0.189 / hr`** (`$0.22 / 1M` batch tok) | `$0.142 / hr` | `+30 W` (`$0.0055 / hr`) | **`+$0.137 / hr`** | **`~$14.75 / mo`** (`15%` duty, `3.6h/day` active demand) | **`$0.236 / hr`** (`~$25.50/mo` in Burst Credits) |
| **3. Gaming Desktop (`RTX 4090 24 GB`, `1,008 GB/s`)** | **High-Throughput Batch Inference / QLoRA / GRPO** | `1.97M` batch tok/hr + `633` WASI rollouts/hr | **`$0.525 / hr`** (`$0.22 / 1M` batch tok) | `$0.394 / hr` | `+265 W` (`$0.0482 / hr`) | **`+$0.346 / hr`** | **`~$20.75 / mo`** (`8.3%` duty, `2.0h/day` active demand) | **`$0.656 / hr`** (`~$39.35/mo` in Burst Credits) |
| **4. Pro Workstation (`Mac Studio M4 Max 128 GB` / `Strix Halo 128 GB`)** | **128 GB Frontier MoE Expert Shard (`400–546 GB/s`)** | `1.42M` batch tok/hr + `654` WASI rollouts/hr | **`$0.417 / hr`** (`$0.22 / 1M` batch tok) | `$0.313 / hr` | `+68 W` (`$0.0124 / hr`) | **`+$0.301 / hr`** | **`~$32.50 / mo`** (`15%` duty, `3.6h/day` active demand) | **`$0.521 / hr`** (`~$56.25/mo` in Burst Credits) |
| **5. 20-Slot USB-C Smartphone Array (`160 GB` Pool, `34 GB/s` agg.)** | **Parallel WASI RL Sandbox Verification Array** *(Gated out of LLM decode)* | `0` LLM tok/hr; `642` WASI rollouts/hr + `1.6 GiB/hr` seed | **`$0.103 / hr`** *(N/A — WASI & Seed only)* | `$0.077 / hr` | `+55 W` (`$0.0100 / hr`) | **`+$0.067 / hr`** | **`~$9.05 / mo`** (`18.8%` duty, `4.5h/day` active demand) | **`$0.129 / hr`** (`~$17.40/mo` in Burst Credits) |

*Note on Realistic Duty Cycles & Host Solvency:*
1. **No Demand, No Cash Payout (Hard Solvency Rule):** Cash payouts (`75%` of buyer clearing) are funded **100% out of verified external buyer compute billings**. Kadmium never pays cash for idle uptime without a paying buyer job.
2. **Conservative Modelled Duty Cycles (`2.0 – 7.2 hrs/day`):** Early-grid supply exceeds demand (`R4 §6.3`). The table above models realistic early duty cycles (`8%–30%` of the day with active buyer jobs, e.g., `2 hours/day` of matched batch work on an RTX 4090 yielding `~$20.75/mo` net of `$0.182/kWh` electricity, or `3.6 hours/day` on a 64 GB M4 Pro yielding `~$14.75/mo` net of electricity)—not heroic `50–65%` utilization assumptions.
3. **Automatic Power-Price Floor Guard:** If a host's local utility tariff (or Time-of-Use peak rate) causes their incremental electricity cost $P_{\text{wall}} \cdot C_{\text{kWh}}$ to exceed `80%` of the current job clearing payout, the Kadmium daemon automatically pauses cash-mode execution during that tariff window so a host **never loses money to their electric utility**.

---

## 4. Credit Redemption Liability, Deferred Revenue & 90-Day Financial Model (`R4 §6.3, §6.5`)

### 4.1 Accounting for Compute Credits & Prepaid Burst Packs (No "Free Margin" Illusion)
A rigorous clearinghouse must account for future fulfillment obligations at the moment credits or prepaid packs are issued (`R4 §6.3, §6.5`):
1. **Compute Credit Redemption Liability (`70%` Expected Redemption, 12-Month Expiry):**
   - When an overnight host elects to earn **Kadmium Compute Credits (`1.25×` wholesale-indexed credit value)** instead of USD cash, `$0` leaves Kadmium's bank account *on the day of issuance*, **but Kadmium books a Credit Redemption Liability** on its balance sheet:
     $$\text{CreditLiabilityReserve} = \text{CreditsIssued}_{\text{USD}} \times \underbrace{0.70}_{\text{Expected Redemption Rate}} \times \underbrace{0.75}_{\text{Fulfillment Cost Ratio}}$$
   - Credits expire **12 months** after issuance (`CREDIT_EXPIRY_MONTHS = 12`) and are non-transferable. When a user redeems credits during daytime burst, Kadmium routes `24–32 GB` slices first to other credit-earning community hosts (netting mutual obligations internally) and routes `80 GB+` slices to wholesale spot providers funded by the liability reserve.
2. **Prepaid Burst Packs Are Deferred Revenue (`~75%` Fulfillment Cost on Redemption):**
   - When a developer or lab buys a `$100`, `$500`, or `$2,500` **Prepaid Burst Pack**, the cash received is **Deferred Revenue (unearned compute liability)** until consumed.
   - As the customer consumes burst hours at `WholesaleVerifiedIndex × 1.25` (or Supply Grid work units where `75%` goes to the host), the **fulfillment COGS is `75%–80%` of the pack's face value** (`PREPAID_PACK_FULFILLMENT_COGS_RATIO = 0.75`).
   - Therefore, Prepaid Burst Packs earn a **`~20%–25%` realized gross margin upon consumption**—**never `83%`**.

### 4.2 Three-Scenario 90-Day Financial Projections (`Bear / Base / Stretch`, `R4 §6.5`)
Per `R4 §6.5`, Kadmium budgets operating expenses strictly against the **Base Case (`~$35,900/mo` Day-90 gross billings, `3.0%` Pro conversion)** and maintains solvency even in the **Bear Case (`~$16,200/mo` Day-90 gross billings, `1.5%` Pro conversion)**. The `$110,637/mo` (`8.7%` Pro conversion) trajectory is tracked explicitly as the **Stretch Case**:

| Metric at Day 90 (Month 3 Run-Rate) | **Bear Case** *(1.5% Pro Conv.)* | **Base Case — Budget Target** *(3.0% Pro Conv., `R4 §6.5`)* | **Stretch Case** *(8.7% Pro Conv.)* | Accounting & COGS Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Active Free LAN Clusters (`Tier 0`)** | `1,500` clusters | **`2,500` clusters** | `6,000` clusters | `$0` revenue, `$0` cloud COGS (100% local LAN) |
| **1. `CloudShield Pro` (`$29/mo`)** | `45` subs (`$1,305`) | **`150` subs (`$4,350`)** | `520` subs (`$15,080`) | `$35` credit cap routed to community grid + `1.25×` indexed burst (`~52%` avg utilization = `~$15.10/sub` COGS) |
| **2. `Team / Lab Cluster` (`$199/mo`)** | `12` labs (`$2,388`) | **`35` labs (`$6,965`)** | `95` labs (`$18,905`) | Pure SaaS control plane (`~90%` gross margin) |
| **3. `Private Office Fabric` (`$499/mo` LAN) & MSP (`$299/mo`)** | `4` offices (`$1,596`) | **`12` offices (`$4,788`)** | `32` offices (`$12,768`) | 100% On-Premise LAN software (`~92%` gross margin) |
| **4. Indexed Cloud Burst & Supply Grid Gross Billings** | `$6,400` | **`$11,800`** | `$38,884` | `75%–80%` wholesale/host payout COGS; **`20%–25%` net spread retained** |
| **5. Prepaid Burst Pack Cash Sales *(Deferred Revenue)*** | `$4,500` | **`$8,000`** | `$25,000` | Carries **`75%` fulfillment COGS reserve (`$6,000` in Base)** + `3.2%` Stripe fees |
| **Gross Monthly Billings (Cash Inflow)** | **`$16,189 / mo`** | **`$35,903 / mo`** | **`$110,637 / mo`** | Total top-line customer billings |
| **Less: Wholesale Spot COGS + `75%` Host Cash Payouts** | `-$5,610 / mo` | **`-$11,595 / mo`** | `-$37,020 / mo` | Direct burst wholesale + `75%` Supply Grid host share |
| **Less: Deferred Pack Fulfillment Reserve (`75%`) & Credit Liability** | `-$3,825 / mo` | **`-$6,850 / mo`** | `-$21,150 / mo` | Accrual reserve for future pack & overnight credit redemption |
| **Less: Stripe/Payment Fees (`3.2%`) + Discovery Relays** | `-$968 / mo` | **`-$1,899 / mo`** | `-$4,990 / mo` | Payment rails + 3 global `libp2p` coordinator relays (`$750/mo`) |
| **Accrual Net Operating Contribution (Real Gross Margin)** | **`+$5,786 / mo`** *(35.7%)* | **`+$15,559 / mo`** *(**43.3% Accrual Margin**)* | **`+$47,477 / mo`** *(42.9%)* | **True accrual margin after `75%` pack & credit fulfillment reserves** |
| **Less: Cost-Plus Living-Wage Operations Reserve** | `-$5,000 / mo` | **`-$12,000 / mo`** | `-$22,127 / mo` | Covers founder/steward living wage + tax/accounting |
| **Quarterly `5%` + Overflow STEM & Community Grant Sweep** | **`$786 / mo`** | **`$3,559 / mo`** (`$10,677/qtr`) | **`$25,350 / mo`** (`$76,050/qtr`) | Swept to open-source contributors, student labs & SMB grants |

---

## 5. Open-Hardware Economics: The `Cd-48` 32 GB RK3588 Coordinator & `.ktb` NAS Seed Node (`R4 §3.4, §6.6`)

### 5.1 Physical Interconnect Reality Check: Why It Is a Coordinator & `.ktb` NAS Seed Vault (Not a Desktop RAM Execution Sidecar)
Per `R4 §3.4`, any external ARM box attached to a desktop PC over **`2.5GbE` (`0.28 GB/s`)** or **USB-C 10–20 Gbps (`0.9–2.4 GB/s`)** is **`~100–200×` slower than the desktop's own GPU VRAM (`500–1,800 GB/s`)**, **`~25–100×` slower than internal dual-channel DDR5 (`60–90 GB/s`)**, and **`~6–25×` slower than a `$90` internal PCIe Gen4 NVMe SSD (`7.0 GB/s`)**, while its internal `32 GB` LPDDR5 bus runs at `~25.6 GB/s`.

Consequently, Kadmium **never** markets an external RK3588 appliance as a "desktop memory execution sidecar" for running spilled 70B layers faster than local DDR5. Instead, the **`Cd-48` 32 GB Node** is engineered and sold for the four roles where an always-on `6–15 W` ARM64 box with `32 GB` RAM, an M.2 NVMe slot, and dual `2.5GbE` excels:
1. **24/7 `libp2p` Cluster Coordinator & `CloudShield` Watchdog:** Keeps the home/lab cluster's CRDT state, `.ktb` routing tables, and instant failover watchdog alive 24/7 at `6 W` idle while heavy `350 W` gaming PCs sleep.
2. **Always-On `.ktb` Merkle Weight Seed Vault & FTS3 Cache (`NVMe + 32 GB RAM Cache`):** Pre-stages and seeds page-aligned `.ktb` model shards across the local LAN at `2.5GbE` wire speed (`~280 MB/s`) and to the Kadmium Supply Grid (`$0.008 / GiB`), eliminating cold-start Hugging Face downloads.
3. **Deterministic `WASI` RL Verifier & Speculative Draft/Tokenizer Node:** Executes fuel-metered `WASI` sandbox unit-test rollouts (`$0.14 / 1k rollouts`) and small `0.5B–1.5B` speculative draft trees (`6 TOPS` NPU + 8-core Cortex-A76/A55).
4. **100% Open-Root Linux Home Server:** Ships with mainline Debian/Armbian Linux, unlocked bootloader, full `root` SSH, and pre-configured Docker/Samba/WireGuard/Nextcloud alongside the `kadmium` daemon.

### 5.2 October 2026 Memory-Supercycle BOM & Sustainable `$379` Cost-Plus Pricing (`R4 §6.6`)
During the 2025–2026 memory supercycle (`+80–90%` LPDDR5 contract price increase; `16 GB` Orange Pi 5 Max retailing at `$189–$235`), a `32 GB` LPDDR5 RK3588 board + CNC aluminum heatsink enclosure + 30W USB-C PD PSU + packaging carries a realistic small-batch **BOM of `$240–$300` (`$275.00` midpoint)** (`R4 §6.6`). Selling such a unit at `$249` would lose `$50+` per box after `9%` crowdfunding and payment processing fees (`Crowd Supply / Kickstarter + Stripe`), shipping variance, and RMA reserve.

| Hardware Cost / Margin Component | Canonical Value (`2026-10-20`) | Notes & Gating Policy |
| :--- | :--- | :--- |
| **32 GB LPDDR5 RK3588 Board + CNC Case + PSU BOM** | **`$275.00`** (`$240–$300` range) | Reflects Q3/Q4 2026 32 GB LPDDR5 pricing (`R4 §6.6`) |
| **Sustainable Cost-Plus Retail Price** | **`$379.00`** (`$349–$399` tier) | Priced sustainably above BOM + fees + warranty reserve |
| **Crowdfunding + Payment Processing Fees (`9.0%`)** | **`-$34.11`** | `5–6%` platform fee + `3–4%` payment processing (`R4 §6.6`) |
| **Warranty / RMA Reserve (`4.0%`)** | **`-$15.16`** | Covers DOA replacement and shipping buffer |
| **Net Hardware Contribution Margin** | **`+$54.73 / unit` (`14.4%`)** | **Hardware Gate:** Campaign launches *only* after a signed ODM quote (Radxa / FriendlyElec / Pine64) confirms landed BOM $\le \$275$ at MOQ 500. Until then, Kadmium ships free one-command images for user-owned Raspberry Pi 5, Orange Pi 5, Radxa Rock 5B, and Zimaboard hardware. |

---

## 6. Security & Compliance Architecture: What Survives a Formal Audit (`R4 Critique`)

### 6.1 Cryptographic & Regulatory Reality Check on Orthogonal Rotations (`FWHT`)
In non-regulated developer/prosumer tiers, Kadmium supports **Zero Plaintext Egress** (`Layer 0` token embeddings and `Layer N-1` output logits remain pinned on the user's local machine) plus an optional session-keyed orthogonal Fast Walsh-Hadamard rotation ($\tilde{h} = h Q^\top$, $\tilde{W} = Q W Q^\top$) for wire obfuscation and quantization incoherence.

**However, per our R4 Security & Compliance Audit, Kadmium explicitly rejects using orthogonal rotations (`FWHT`) to claim regulatory compliance (HIPAA, CMMC, ABA) on external community hosts:**
1. **An Orthogonal Rotation Is Invertible and Not Cryptographic Encryption:** Near-zero cosine similarity between $h$ and $\tilde{h} = h Q^\top$ simply reflects a high-dimensional angle change, not information-theoretic secrecy.
2. **Known-Weight Recovery (`Public Models`):** For any public-weight model $W$, a host holding the rotated weights $\tilde{W} = Q W Q^\top$ and downloading the public weights $W$ can recover the orthogonal transformation $Q$ in a single least-squares/spectral solve and un-rotate every activation.
3. **Activation-Inversion Attacks (`Private Models`):** Even when $W$ is private, academic activation-inversion attacks reconstruct input text from intermediate transformer hidden states with high fidelity.
4. **HIPAA & CMMC Legal Boundaries:**
   - Under **HIPAA (45 CFR §164.514)**, intermediate neural activations derived from Protected Health Information (PHI) remain PHI unless formally de-identified by expert determination or safe harbor; every external host touching those activations would legally require a signed **Business Associate Agreement (BAA)**—which is impossible on an anonymous consumer grid.
   - Under **CMMC 2.0 Level 2 (NIST SP 800-171)**, Controlled Unclassified Information (CUI) in transit requires **FIPS 140-3 validated cryptographic modules** inside an assessed authorization boundary, which a linear-algebraic rotation is not and a consumer swarm does not possess.

### 6.2 Audited Compliance Matrix for `Private Office Fabric ($499/mo)`

| Regulated Vertical | Permitted Execution Topology | Cryptographic & Network Boundary | BAA / Assessment Policy |
| :--- | :--- | :--- | :--- |
| **1. Healthcare Clinics, Dental & Therapy Practices (HIPAA)** | **100% On-Premise / Air-Gapped Office LAN Mode ONLY** *(Zero community grid nodes; zero cloud burst unless customer connects their own direct BAA-signed AWS/Azure/GCP account over FIPS TLS)* | All weights, prompts, KV caches, and activations execute strictly inside the clinic's local LAN (`Zero Egress`). | **Self-Hosted On-Premise Software Model:** Kadmium software runs 100% on the clinic's own hardware and never receives, stores, or transmits PHI. **Kadmium does not sign standalone cloud BAAs** until a formal third-party SOC 2 Type II / HIPAA security program and cyber-liability insurance policy are in place. |
| **2. Defense Subcontractors & Engineering Shops (CMMC 2.0 / ITAR)** | **100% Air-Gapped Enclave LAN Mode ONLY** (`--airgap --fips-tls`) | Zero WAN sockets compiled/enabled in air-gap profile; inter-desk LAN links encrypted via OS-validated **FIPS 140-3 TLS 1.3** modules; local SHA-256 tamper-evident audit logs. | Customer retains 100% CUI custody inside their existing CMMC assessment boundary; zero external burst permitted. |
| **3. Law Firms, CPAs & M&A Advisory (ABA Model Rule 1.6)** | **100% On-Premise Office LAN Mode** *(Default)* | Zero privileged client documents or activations leave the firm's local firewall; per-matter RBAC isolation on the local office cluster. | Preserves attorney-client privilege and ABA Rule 1.6 confidentiality by eliminating third-party cloud sub-processors. |
| **4. Non-Regulated Prosumers, Indie Devs & Open-Source Labs** | **Local LAN + Optional Supply Grid / Spot Burst** | **Zero Plaintext Egress** (`Layer 0` & `Layer N-1` stay local on user's PC; only middle-layer tensors traverse mTLS 1.3 + optional `FWHT` wire obfuscation). | Suitable for non-PHI / non-CUI software development, open-dataset QLoRA/GRPO training, and synthetic benchmarks. |

---

## 7. Ethical & Moral Charter of Kadmium (`Cd-48`)

### 7.1 The Five Inviolable Moral Commitments
1. **Steward-Ownership & Anti-Enshittification (`75 / 20 / 5` Split):**
   - Kadmium is structured as a **Steward-Owned / Public-Benefit enterprise** with a Golden Share veto against predatory buyouts, ad-tech telemetry harvesting, or raising the platform take-rate above the `20%` Cost-Plus operations cap.
   - Executive and founder compensation is capped at **`4×` the US median full-time wage** (`~$250k/yr` ceiling). `100%` of net operating surplus above the Cost-Plus reserve and 6-month operating buffer is swept quarterly into the **Community STEM, Academic & Small-Business Grant Pool**.
2. **Strict No-Crypto / No-Token / No-Speculation Policy:**
   - Kadmium will **never** mint a cryptocurrency, utility token, NFT, or speculative point system. Every transaction is denominated in real **USD (`$`)** via Stripe Connect / ACH / Wire and wholesale-indexed **USD Compute Credits** with a transparent 12-month expiry.
3. **Honest Power-Cost Transparency & Hardware Lifespan Protection:**
   - Every host payout quote in the CLI, companion app, and web calculator deducts the host's incremental wall-socket electricity cost (`$0.182 / kWh` EIA default, customizable to local utility tariffs) **before** displaying net earnings.
   - **Polite Gaming & Work Yield (`<100ms`):** The moment a user launches a game, compiler, video editor, or local LLM prompt, Kadmium yields local GPU/CPU resources within `100ms`.
   - **Battery & Thermal Guard:** Mobile and laptop nodes **never** execute grid jobs on battery power or above `72°C` sustained junction temperature.
   - **NVMe Flash-Wear Cap:** Zero-copy `mmap` `.ktb` shard loading caps background cache writes so host SSD endurance (`TBW`) is never degraded by swap thrashing.
4. **Right-to-Repair & Open Root Access:**
   - Every piece of software and optional reference hardware Kadmium ships gives the owner **100% root access, unlocked bootloaders, and zero DRM**. A user can uninstall Kadmium at any time and use their hardware as a standard Linux server or workstation.
5. **Measurement-First Honesty (`No Unverified Benchmarks`):**
   - Kadmium separates **Spec-Verified Mathematical Constants** (e.g., `10×–85×` DeMo DCT-II compression, `0.00e0` online Softmax LSE reduction error), **Dated Empirical Market Constants** (`2026-10-20` [`docs/NUMBERS_2026-10.md`](NUMBERS_2026-10.md)), and **Modelled Capability Envelopes** (explicitly labelled **Modelled** until reproducible `kadmium bench` hardware logs are attached).
