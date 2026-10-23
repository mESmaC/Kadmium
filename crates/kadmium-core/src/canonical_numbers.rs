//! # Canonical Dated Market, Energy & Hardware Constants (`2026-10-20` R4 Audit)
//!
//! Single source of truth consumed by `kadmium-core::supply`, `kadmium-core::reactor_clearinghouse`,
//! `kadmium-sharding::spot_provisioner`, and the documentation suite (`docs/NUMBERS_2026-10.md`,
//! `docs/BUSINESS_PLAN_AND_STANCE.md`, and `docs/MARKET_RESEARCH_AND_LAUNCH_PLAN.md`).
//!
//! Every figure in this module is date-stamped to the **20 October 2026 R4 Market Research Review**
//! (`docs/MARKET_RESEARCH_REVIEW_2026-10.md`) so code and docs never drift.

/// Date stamp of the canonical market dataset (`YYYY-MM-DD`).
pub const CANONICAL_NUMBERS_DATE_STAMP: &str = "2026-10-20";

// ================================================================================================
// 1. Macro Grid, Electricity & Hardware Population Constants (R4 §7.4, §7.9, §8.1, §8.2)
// ================================================================================================

/// US EIA Short-Term Energy Outlook (Oct 6, 2026) residential average electricity rate (`$0.182 / kWh`).
pub const EIA_US_RESIDENTIAL_KWH_USD_2026: f64 = 0.182;

/// US EIA Short-Term Energy Outlook (Oct 6, 2026) projected 2027 residential rate (`$0.187 / kWh`).
pub const EIA_US_RESIDENTIAL_KWH_USD_2027: f64 = 0.187;

/// IEA *Energy and AI* global datacenter electricity consumption in 2024 (`415 TWh`).
pub const IEA_DATACENTER_TWH_2024: f64 = 415.0;

/// IEA *Energy and AI* base-case global datacenter electricity projection for 2030 (`945 TWh`).
pub const IEA_DATACENTER_TWH_2030_BASE: f64 = 945.0;

/// Steam concurrent user peak (Oct 11, 2026) in millions (`43.5M`).
pub const STEAM_PEAK_CONCURRENT_USERS_MILLIONS: f64 = 43.5;

/// Derived 32 GB VRAM GPUs online at Steam peak (`~1.3%` share = `0.56M` systems).
pub const STEAM_DERIVED_32GB_GPUS_MILLIONS: f64 = 0.56;

/// Derived 24 GB VRAM GPUs online at Steam peak (`5.50%` share = `2.40M` systems).
pub const STEAM_DERIVED_24GB_GPUS_MILLIONS: f64 = 2.40;

/// Derived 16 GB VRAM GPUs online at Steam peak (`27.21%` share = `11.84M` systems).
pub const STEAM_DERIVED_16GB_GPUS_MILLIONS: f64 = 11.84;

/// Derived cumulative 12 GB+ VRAM GPUs online at Steam peak (`49.61%` share = `21.58M` systems).
pub const STEAM_DERIVED_12GB_PLUS_GPUS_MILLIONS: f64 = 21.58;

// ================================================================================================
// 2. Wholesale GPU Index & Index-Plus-Spread Burst Pricing (R4 §5.3, §6.2, §7.1 — 20 Oct 2026)
// ================================================================================================

/// Cheapest 85-provider listing for RTX 4090 24 GB (`$0.33/hr`, Salad).
pub const WHOLESALE_FLOOR_RTX4090_USD_HR: f64 = 0.33;
/// Realistic verified wholesale rate for RTX 4090 24 GB (`$0.38/hr`, RunPod community $0.34 – Spheron $0.48 midpoint).
pub const WHOLESALE_VERIFIED_RTX4090_USD_HR: f64 = 0.38;
/// 85-provider median for RTX 4090 24 GB (`$0.48/hr`).
pub const WHOLESALE_MEDIAN_RTX4090_USD_HR: f64 = 0.48;

/// Cheapest 85-provider listing for NVIDIA L40S 48 GB (`$0.55/hr`, Novita).
pub const WHOLESALE_FLOOR_L40S_USD_HR: f64 = 0.55;
/// Realistic verified wholesale rate for NVIDIA L40S 48 GB (`$0.96/hr`, Spheron; Modal is `$1.95/hr`).
pub const WHOLESALE_VERIFIED_L40S_USD_HR: f64 = 0.96;
/// 85-provider median for NVIDIA L40S 48 GB (`$1.64/hr`).
pub const WHOLESALE_MEDIAN_L40S_USD_HR: f64 = 1.64;

/// Cheapest 85-provider listing for NVIDIA A100 80 GB (`$0.89/hr`).
pub const WHOLESALE_FLOOR_A100_80GB_USD_HR: f64 = 0.89;
/// Realistic verified wholesale rate for NVIDIA A100 80 GB (`$1.43/hr` Spheron – `$1.49/hr` RunPod).
pub const WHOLESALE_VERIFIED_A100_80GB_USD_HR: f64 = 1.46;
/// 85-provider median for NVIDIA A100 80 GB (`$1.80/hr`).
pub const WHOLESALE_MEDIAN_A100_80GB_USD_HR: f64 = 1.80;

/// Cheapest 85-provider listing for NVIDIA H100 80 GB (`$1.80/hr`).
pub const WHOLESALE_FLOOR_H100_80GB_USD_HR: f64 = 1.80;
/// Realistic verified wholesale rate for NVIDIA H100 80 GB (`$1.87/hr` Vast verified – `$2.65/hr` Spheron).
pub const WHOLESALE_VERIFIED_H100_80GB_USD_HR: f64 = 2.20;
/// 85-provider median for NVIDIA H100 80 GB (`$3.99/hr`).
pub const WHOLESALE_MEDIAN_H100_80GB_USD_HR: f64 = 3.99;

/// Default retail spread over realistic verified wholesale for on-demand burst (`+25%` = `1.25x` wholesale index).
pub const BURST_INDEX_SPREAD_MULTIPLIER: f64 = 1.25;

/// Computes the indexed retail burst rate (`wholesale_verified * 1.25`, rounded to cents).
pub fn indexed_retail_burst_usd_hr(verified_wholesale_usd_hr: f64) -> f64 {
    ((verified_wholesale_usd_hr * BURST_INDEX_SPREAD_MULTIPLIER) * 100.0).round() / 100.0
}

// ================================================================================================
// 3. Serverless Token Price Floor & Output-Delivered Supply Grid Clearing Rates (R4 §5.5, §6.3)
// ================================================================================================

/// OpenRouter / GMICloud DeepSeek-V3.2 serverless input floor (`$0.21–$0.28 / 1M tokens`).
pub const SERVERLESS_V3_2_INPUT_USD_PER_1M: f64 = 0.28;
/// OpenRouter / GMICloud DeepSeek-V3.2 serverless output floor (`$0.31–$0.42 / 1M tokens`).
pub const SERVERLESS_V3_2_OUTPUT_USD_PER_1M: f64 = 0.42;

/// Frontier-open tier (DeepSeek V4-Pro / Kimi K2.6 / GLM-5.1) input rate (`$0.95–$2.10 / 1M tokens`).
pub const SERVERLESS_FRONTIER_OPEN_INPUT_USD_PER_1M: f64 = 1.74;
/// Frontier-open tier (DeepSeek V4-Pro / Kimi K2.6 / GLM-5.1) output rate (`$3.48–$4.40 / 1M tokens`).
pub const SERVERLESS_FRONTIER_OPEN_OUTPUT_USD_PER_1M: f64 = 3.48;

/// Kadmium Supply Grid batch token clearing price per 1M tokens processed (`$0.22 / 1M`, ~25% discount
/// to blended V3.2 serverless rates and ~85% discount to frontier-open batch rates).
pub const SUPPLY_GRID_BATCH_CLEARING_USD_PER_1M_TOKENS: f64 = 0.22;

/// Kadmium Supply Grid verified WASI RL sandbox clearing price per 1,000 fuel-metered rollouts (`$0.14 / 1k`).
pub const SUPPLY_GRID_WASI_CLEARING_USD_PER_1K_ROLLOUTS: f64 = 0.14;

/// Kadmium Supply Grid `.ktb` Merkle-verified P2P shard seeding clearing price per GiB served (`$0.008 / GiB`).
pub const SUPPLY_GRID_KTB_SEED_CLEARING_USD_PER_GIB: f64 = 0.008;

// ================================================================================================
// 4. Steward Split, Credit Liability & Deferred Revenue Parameters (R4 §6.3, §6.5)
// ================================================================================================

/// Host share of verified buyer compute billings (`75.0%`).
pub const STEWARD_HOST_SHARE_FRAC: f64 = 0.75;
/// Cost-Plus living-wage operations & relay infrastructure cap share (`20.0%`).
pub const STEWARD_COST_PLUS_OPS_FRAC: f64 = 0.20;
/// Base Community STEM, Academic & Small-Business Grant Pool share (`5.0%`).
pub const STEWARD_STEM_GRANT_FRAC: f64 = 0.05;

/// Overnight host credit reciprocity multiplier (`1.25x` wholesale-indexed credit rate, 12-month expiry).
pub const OVERNIGHT_CREDIT_MULTIPLIER: f64 = 1.25;
/// Standard daytime host credit multiplier (`1.00x` wholesale-indexed credit rate).
pub const DAYTIME_CREDIT_MULTIPLIER: f64 = 1.00;
/// Credit expiration window in months (`12` months).
pub const CREDIT_EXPIRY_MONTHS: u32 = 12;
/// Expected credit redemption rate before 12-month expiry (`70.0%`, R4 §6.3).
pub const EXPECTED_CREDIT_REDEMPTION_RATE: f64 = 0.70;
/// Deferred revenue fulfillment cost ratio on prepaid burst packs (`~75.0%` wholesale fulfillment cost).
pub const PREPAID_PACK_FULFILLMENT_COGS_RATIO: f64 = 0.75;

// ================================================================================================
// 5. Protocol & Hardware BOM Constants (R4 §6.6, §7.6, §10)
// ================================================================================================

/// Spec-verified Nous DeMo DCT-II gradient compression range (`10x` to `85x`).
pub const DEMO_DCT_COMPRESSION_MIN_X: f64 = 10.0;
pub const DEMO_DCT_COMPRESSION_MAX_X: f64 = 85.0;

/// October 2026 memory-supercycle BOM for a 32 GB LPDDR5 RK3588 board + enclosure + PSU (`$275.00`, range `$240–$300`).
pub const RK3588_32GB_NAS_NODE_BOM_USD_2026: f64 = 275.0;
/// Sustainable Cost-Plus retail price for the 32 GB RK3588 Coordinator / `.ktb` Seed Vault / Home Server (`$379.00`).
pub const RK3588_32GB_NAS_NODE_RETAIL_USD_2026: f64 = 379.0;
/// Crowdfunding + payment processing fee assumption (`9.0%`, R4 §6.6).
pub const CROWDFUNDING_AND_PAYMENT_FEE_PCT: f64 = 9.0;

// ================================================================================================
// 6. Resource Offer Defaults (engineering constants, not market data)
// ================================================================================================
//
// What a node offers the grid by default, before its owner edits the offer. One set of figures
// for the planner, the scheduler and the capability vector; the reserve rule is lumabri's donor
// default (one quarter of RAM, bounded to 1-4 GiB, never offered), the RAM fraction is the
// resident-weight budget the planner has always used, and the fast-memory fraction is the share
// of VRAM an idle host lends.

/// Share of VRAM or unified memory an idle host offers (`90%`).
pub const OFFER_FAST_MEMORY_FRAC: f64 = 0.90;
/// Share of system RAM offered for resident weights (`70%`), capped by the reserve rule below.
pub const OFFER_SYSTEM_RAM_FRAC: f64 = 0.70;
/// Share of system RAM kept for the owner (`25%`), bounded by the min and max below.
pub const OFFER_RAM_RESERVE_FRAC: f64 = 0.25;
/// Smallest RAM reserve in GiB (`1 GiB`).
pub const OFFER_RAM_RESERVE_MIN_GB: f64 = 1.0;
/// Largest RAM reserve in GiB (`4 GiB`; a 16 GiB host keeps 4, an 8 GiB host keeps 2).
pub const OFFER_RAM_RESERVE_MAX_GB: f64 = 4.0;
/// Work instances a node runs at once by default (`4`).
pub const OFFER_DEFAULT_SLOTS: u32 = 4;
/// Distinct model allocations a node accepts at once by default (`4`).
pub const OFFER_MAX_ALLOCATIONS: u32 = 4;
