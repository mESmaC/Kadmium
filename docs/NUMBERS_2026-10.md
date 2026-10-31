# Kadmium (`Cd-48`) Historical Engineering Numbers Ledger (`2026-10-20`)

> **Scope update — 2026-10-30:** This ledger and [canonical_numbers.rs](../crates/kadmium-core/src/canonical_numbers.rs) preserve historical engineering/model inputs. They are not a live wholesale feed, verified customer price book, compliance determination or approved financial policy. The commercial [evidence review](MARKETING_BUSINESS_REVIEW_2026-10-30.md) supersedes the earlier business interpretation; proposed operating assumptions now live in [business/assumptions.json](business/assumptions.json).
>
> The historical tables below remain unchanged for traceability. The 75% wholesale reserve, assumed 70% credit redemption, Steam-derived host counts, universal token discounts and legal/charter assertions must not be reused as validated commercial facts. A 1.25× wholesale markup implies 80% fulfillment cost. See the [recalculated economics](business/ECONOMICS.md). This review does not change runtime behavior.
>
> - **Dataset Date Stamp:** `2026-10-20`
> - **Primary Audit Reference:** [`docs/MARKET_RESEARCH_REVIEW_2026-10.md`](MARKET_RESEARCH_REVIEW_2026-10.md) (*R4 Market Research & Fact-Check Review*)
> - **Consuming Code Modules:** [`crates/kadmium-core/src/canonical_numbers.rs`](../crates/kadmium-core/src/canonical_numbers.rs), [`crates/kadmium-core/src/supply.rs`](../crates/kadmium-core/src/supply.rs), [`crates/kadmium-core/src/reactor_clearinghouse.rs`](../crates/kadmium-core/src/reactor_clearinghouse.rs)
> - **Historical Consumers:** earlier business and launch plans; some [README](../readme.md) and application claims still require the publication-remediation pass identified in the review.

---

## 1. Macro Grid, Electricity & Hardware Population Constants (R4 §7.4, §7.9, §8.1, §8.2)

| Constant / Metric | Canonical Value (`2026-10-20`) | Rust Symbol (`canonical_numbers.rs`) | Primary Source |
| :--- | :--- | :--- | :--- |
| **US Residential Electricity Rate (2026)** | **`$0.182 / kWh`** (`18.2¢/kWh`) | `EIA_US_RESIDENTIAL_KWH_USD_2026` | US EIA *Short-Term Energy Outlook* (Oct 6, 2026) |
| **US Residential Electricity Forecast (2027)** | **`$0.187 / kWh`** (`18.7¢/kWh`) | `EIA_US_RESIDENTIAL_KWH_USD_2027` | US EIA *Short-Term Energy Outlook* (Oct 6, 2026) |
| **Global Datacenter Electricity (2024)** | **`415 TWh`** (`~1.5%` of global electricity) | `IEA_DATACENTER_TWH_2024` | IEA *Energy and AI* Special Report |
| **Global Datacenter Electricity (2030 Base Case)** | **`945 TWh`** | `IEA_DATACENTER_TWH_2030_BASE` | IEA *Energy and AI* Special Report |
| **PJM 2026/2027 Capacity Auction Clearing** | **`$329.17 / MW-day`** (`$16.1B` total; `~9.4x` vs. `$28.92` in 24/25) | — | PJM Interconnection Base Residual Auction |
| **Steam Concurrent User Peak (Oct 11, 2026)** | **`43.5M` concurrent users** | `STEAM_PEAK_CONCURRENT_USERS_MILLIONS` | SteamDB / Valve Hardware & Software Survey |
| **Derived `32 GB` Consumer GPUs Online at Peak** | **`~0.56M` GPUs** (`~1.3%` share, RTX 5090 class) | `STEAM_DERIVED_32GB_GPUS_MILLIONS` | Sep/Oct 2026 Steam Survey × `43.5M` peak |
| **Derived `24 GB` Consumer GPUs Online at Peak** | **`~2.40M` GPUs** (`5.50%` share, RTX 3090/4090/7900 XTX) | `STEAM_DERIVED_24GB_GPUS_MILLIONS` | Sep/Oct 2026 Steam Survey × `43.5M` peak |
| **Derived `16 GB` Consumer GPUs Online at Peak** | **`~11.84M` GPUs** (`27.21%` share) | `STEAM_DERIVED_16GB_GPUS_MILLIONS` | Sep/Oct 2026 Steam Survey × `43.5M` peak |
| **Derived Cumulative `12 GB+` GPUs Online at Peak** | **`~21.58M` GPUs** (`49.61%` share) | `STEAM_DERIVED_12GB_PLUS_GPUS_MILLIONS` | Sep/Oct 2026 Steam Survey × `43.5M` peak |

---

## 2. Wholesale GPU Index & Index-Plus-Spread Burst Pricing (R4 §5.3, §6.2, §7.1)

Kadmium does **not** use static legacy wholesale tables (`$0.29 / $0.52 / $0.82 / $1.65`) or fixed retail burst rates that sit below verified wholesale costs. All burst compute is priced dynamically as:

$$\text{RetailBurstRate}(\text{class}) = \text{WholesaleVerifiedIndex}(\text{class}) \times (1 + \text{Spread})$$

where the default on-demand spread is **`+25%` (`1.25×` verified wholesale index)**. Furthermore, 24–32 GB workloads are routed first to **credit-paid community/studio supply**, while 48–80 GB+ burst routes to verified spot/wholesale providers.

| GPU Class | 85-Provider Floor (Unverified / Preemptible) | Realistic Verified Wholesale Index (Oct 20, 2026) | 85-Provider Market Median | Kadmium Indexed On-Demand Burst (`Index × 1.25`) | Dedicated Endpoint Comparison (Together / Baseten / Fireworks) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 4090 (`24 GB`)** | `$0.33/hr` (Salad) | **`$0.38/hr`** (`$0.34` RunPod comm. – `$0.48` Spheron) | `$0.48/hr` | **`$0.48/hr`** | N/A (Consumer/Prosumer Tier) |
| **NVIDIA L40S (`48 GB`)** | `$0.55/hr` (Novita) | **`$0.96/hr`** (Spheron `$0.96`; Modal `$1.95`) | `$1.64/hr` | **`$1.20/hr`** | `$1.95 – $2.80/hr` |
| **NVIDIA A100 (`80 GB`)** | `$0.89/hr` | **`$1.46/hr`** (`$1.43` Spheron – `$1.49` RunPod) | `$1.80/hr` | **`$1.83/hr`** | `$3.29 – $4.10/hr` |
| **NVIDIA H100 (`80 GB`)** | `$1.80/hr` | **`$2.20/hr`** (`$1.87` Vast verified – `$2.65` Spheron) | `$3.99/hr` | **`$2.75/hr`** | `$5.49 – $7.00/hr` |

---

## 3. Serverless Token Benchmarks & Output-Delivered Supply Grid Clearing Rates (R4 §5.5, §6.3)

### 3.1 October 2026 Open-Weight Serverless Token Benchmarks
- **Commodity Tier (`DeepSeek-V3.2` on OpenRouter / GMICloud / Parasail):**
  - Input: **`$0.21 – $0.28 / 1M tokens`** (`SERVERLESS_V3_2_INPUT_USD_PER_1M = 0.28`)
  - Output: **`$0.31 – $0.42 / 1M tokens`** (`SERVERLESS_V3_2_OUTPUT_USD_PER_1M = 0.42`)
- **Frontier-Open Tier (`DeepSeek V4-Pro`, `Kimi K2.6`, `GLM-5.1` on Together / Fireworks / Baseten):**
  - Input: **`$0.95 – $2.10 / 1M tokens`** (`SERVERLESS_FRONTIER_OPEN_INPUT_USD_PER_1M = 1.74`)
  - Output: **`$3.00 – $4.40 / 1M tokens`** (`SERVERLESS_FRONTIER_OPEN_OUTPUT_USD_PER_1M = 3.48`)

### 3.2 Output-Delivered Supply Grid Clearing Model
To prevent low-bandwidth edge/UMA boards from being priced above serverless token APIs, Kadmium prices Supply Grid nodes strictly by **verified output delivered per hour** (not raw GiB installed):

$$\text{BuyerGrossClearing}(\text{USD/hr}) = R_{\text{tok}} \cdot \frac{\text{Tokens}_{\text{batch,hr}}}{10^6} + R_{\text{wasi}} \cdot \frac{\text{Rollouts}_{\text{wasi,hr}}}{10^3} + R_{\text{seed}} \cdot \text{GiB}_{\text{seeded,hr}}$$

| Verified Work Unit | Kadmium Buyer Clearing Rate | Rust Symbol | Buyer Discount vs. Open Market |
| :--- | :--- | :--- | :--- |
| **1M Batch Inference / Eval Tokens** | **`$0.22 / 1M tokens`** | `SUPPLY_GRID_BATCH_CLEARING_USD_PER_1M_TOKENS` | **~25–35% below** V3.2 serverless (`$0.28/$0.42`) & **~85% below** frontier-open (`$1.74/$3.48`) |
| **1,000 Verified WASI RL Sandbox Rollouts** | **`$0.14 / 1k rollouts`** | `SUPPLY_GRID_WASI_CLEARING_USD_PER_1K_ROLLOUTS` | **~45% below** managed cloud CPU sandbox containers |
| **1 GiB `.ktb` Merkle Weight Shard Seeding** | **`$0.008 / GiB`** | `SUPPLY_GRID_KTB_SEED_CLEARING_USD_PER_GIB` | **~80–90% below** hyperscaler egress (`$0.05–$0.09/GiB`) |

- **Bandwidth Gate on LLM Decode:** Nodes with memory bandwidth $\text{BW}_{\text{eff}} < 60\text{ GB/s}$ (e.g., `~25 GB/s` RK3588 SBCs) are automatically routed to **WASI RL verification, speculative draft trees, and `.ktb` P2P seeding** rather than dense 70B/8B batch decode, ensuring buyers never pay `$11–$17/1M tokens` for slow memory buses.

---

## 4. Steward Split, Credit Liability & Deferred Revenue Accounting (R4 §6.3, §6.5)

| Parameter | Canonical Value | Rust Symbol | Accounting Treatment |
| :--- | :--- | :--- | :--- |
| **Host Share of Verified Buyer Billings** | **`75.0%`** | `STEWARD_HOST_SHARE_FRAC` | Paid in USD (`75%` of buyer clearing) or issued as Compute Credits |
| **Cost-Plus Operations & Relay Cap** | **`20.0%`** | `STEWARD_COST_PLUS_OPS_FRAC` | Living-wage salaries (`4x` US median cap), relay servers, tax/audit |
| **Community STEM & Grant Pool** | **`5.0%` + 100% of overflow** | `STEWARD_STEM_GRANT_FRAC` | Automated quarterly sweep of net operating surplus above Cost-Plus cap |
| **Overnight Host Credit Multiplier** | **`1.25x`** (`00:00–08:00` local) | `OVERNIGHT_CREDIT_MULTIPLIER` | 12-month expiry; booked as **Credit Redemption Liability** |
| **Expected Credit Redemption Rate** | **`70.0%`** within 12 months | `EXPECTED_CREDIT_REDEMPTION_RATE` | `Credit COGS = Credits Issued × 0.70 × Fulfillment Cost` |
| **Prepaid Burst Pack Fulfillment COGS** | **`~75.0%`** of pack face value | `PREPAID_PACK_FULFILLMENT_COGS_RATIO` | Prepaid packs are **Deferred Revenue** on purchase; recognized on redemption (`~25%` net spread, yielding **~45%–55% blended company accrual gross margin** at Day 90, never `83%`) |

---

## 5. Protocol & Hardware BOM Constants (R4 §3.4, §6.6, §7.6, §10)

| Constant / Hardware Parameter | Canonical Value (`2026-10-20`) | Rust Symbol | Source / Physical Rationale |
| :--- | :--- | :--- | :--- |
| **Nous DeMo DCT-II Gradient Compression** | **`10× – 85×`** | `DEMO_DCT_COMPRESSION_MIN_X` / `MAX_X` | Matches spec (`arXiv:2411.19870`); replaces stale `1,000×` prose |
| **32 GB LPDDR5 RK3588 Node BOM (Oct 2026)** | **`$275.00`** (`$240–$300` range) | `RK3588_32GB_NAS_NODE_BOM_USD_2026` | 2026 memory supercycle (`+80–90%` LPDDR5 contract rise; Orange Pi 5 Max 16 GB at `$189–$235`) |
| **32 GB RK3588 Coordinator / `.ktb` NAS Retail** | **`$379.00`** (`$349–$399` range) | `RK3588_32GB_NAS_NODE_RETAIL_USD_2026` | Sustainable Cost-Plus pricing after `9.0%` crowdfunding + payment fees (`$34.11`) and `$69.89` net hardware buffer |
| **RK3588 Node Role Classification** | **24/7 Coordinator, `.ktb` NAS Seed Vault, WASI Verifier & Speculative Draft Node** | — | Over `2.5GbE` (`0.28 GB/s`) or USB-C (`0.9–2.4 GB/s`), an external box is `~100–200×` slower than desktop VRAM/DDR5 and `~20×` slower than PCIe Gen4 NVMe (`R4 §3.4`). Never marketed as a desktop memory execution sidecar. |
| **Reference Prosumer Cluster Benchmarks** | **Framework Desktop 128 GB (`$2,459`)** / **Mac Studio M4 Max 128 GB (`$3,699`)** / **DGX Spark 128 GB (`$6,950`)** | — | Oct 2026 MSRP/street prices after memory-price revisions (`R4 §7.3`) |
| **Buyer ROI & Throughput Figures** | **Modelled** (until `kadmium bench` verified logs land) | — | All projected throughput and ROI tables are explicitly labelled **Modelled** (`R4 §10`) |

---

## 6. Resource Offer Defaults (engineering constants, not market data)

What a node offers the grid before its owner edits the offer. One set of figures for the planner, the scheduler and the capability vector; the reserve rule is lumabri's donor default, the RAM fraction is the resident-weight budget the planner has always used.

| Constant | Value | Rust Symbol | Rationale |
| :--- | :--- | :--- | :--- |
| **Fast memory offered by an idle host** | **`90%` of discrete VRAM, all unified memory** | `OFFER_FAST_MEMORY_FRAC` | The share of VRAM an idle host lends; unified memory has no game to keep clear for |
| **System RAM offered** | **`70%`**, capped by the reserve | `OFFER_SYSTEM_RAM_FRAC` | The resident-weight budget behind `resident_weight_capacity_gb` |
| **RAM reserve kept for the owner** | **`25%`, bounded to `1–4 GiB`** | `OFFER_RAM_RESERVE_FRAC`, `_MIN_GB`, `_MAX_GB` | lumabri's donor reserve: a 16 GiB host keeps 4 GiB, an 8 GiB host keeps 2 GiB; never offered even by an explicit offer |
| **Slots per node** | **`4`** | `OFFER_DEFAULT_SLOTS` | Work instances a node runs at once |
| **Model allocations per node** | **`4`** | `OFFER_MAX_ALLOCATIONS` | lumabri's cap on models sharing one donor |
