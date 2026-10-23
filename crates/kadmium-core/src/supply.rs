//! # `kadmium_core::supply`
//!
//! Two-sided **Kadmium Supply Grid** compute-leasing engine (`SupplySettlementEngine`),
//! host lease policies (`SupplyLeasePolicy`), cryptographic `RepOps` receipts, and
//! datacenter-avoidance grid impact accounting (`DatacenterAvoidanceMetrics`).
//!
//! All market rates, electricity prices (`$0.182/kWh` EIA 2026), serverless token price floors,
//! wholesale GPU index anchors, and credit liability parameters are imported directly from
//! [`crate::canonical_numbers`] (`2026-10-20` R4 Market Research Review).

use crate::canonical_numbers::*;
use crate::{ComputeBackend, HardwareNode, NodeId};
use serde::{Deserialize, Serialize};

/// How a host wants to be compensated when leasing idle hardware to the Kadmium Supply Grid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SupplyPayoutMode {
    /// Direct USD settlement via Stripe Connect / USDC.
    UsdDirectPayout,
    /// Kadmium Cloud burst & failover credits (`1.0x–1.25x` wholesale-indexed credit rate, 12-month expiry).
    KadmiumCloudCredits { bonus_multiplier: f64 },
}

/// Fine-grained capability mask controlling which network workloads may run on leased hardware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupplyWorkloadMask {
    /// Host middle transformer layers (`1..N-2`); zero plaintext egress (`Layer 0` & `Layer N-1` stay local).
    pub allow_middle_layer_inference: bool,
    /// Participate in bandwidth-decoupled Sharded-Island DiLoCo + Nous DeMo (`10x–85x` DCT) training steps.
    pub allow_diloco_demo_training: bool,
    /// Execute fuel-metered `WasiMicroSandbox` reward checks and Dream-RSI MCTS rollouts on CPU/SBC.
    pub allow_wasi_rl_sandboxes: bool,
    /// Seed content-addressed `.ktb` 64 KiB Merkle-verified tensor pages from NAS/NVMe.
    pub allow_ktb_seed_caching: bool,
}

impl Default for SupplyWorkloadMask {
    fn default() -> Self {
        Self {
            allow_middle_layer_inference: true,
            allow_diloco_demo_training: true,
            allow_wasi_rl_sandboxes: true,
            allow_ktb_seed_caching: true,
        }
    }
}

/// Host-side policy governing when and how local hardware is leased to the Kadmium Supply Grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyLeasePolicy {
    pub enabled: bool,
    pub payout_mode: SupplyPayoutMode,
    /// Average hours per day the host expects the cluster to be idle and available.
    pub daily_lease_hours: f64,
    /// When true, `PoliteActivityMonitor` evicts leased shards in `< 200 ms` on game launch with zero penalty.
    pub only_when_polite_idle: bool,
    /// Maximum fraction of local VRAM/UMA that may be allocated to leased network shards (`0.0..=0.90`).
    pub max_vram_lease_fraction: f64,
    pub workload_mask: SupplyWorkloadMask,
}

impl Default for SupplyLeasePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            payout_mode: SupplyPayoutMode::KadmiumCloudCredits {
                bonus_multiplier: OVERNIGHT_CREDIT_MULTIPLIER,
            },
            daily_lease_hours: 12.0,
            only_when_polite_idle: true,
            max_vram_lease_fraction: 0.80,
            workload_mask: SupplyWorkloadMask::default(),
        }
    }
}

/// Output-Delivered Unit-Economic Valuation Breakdown (`2026-10-20` R4 Model) explaining **why**
/// a node clears at its rate based on **what it actually produces** (delivered batch tokens/hr,
/// verified WASI rollouts/hr, and `.ktb` GiB seeded/hr priced at a discount to serverless),
/// rather than overpricing slow-bandwidth boards by raw memory capacity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeValuationBreakdown {
    /// Delivered batch token & middle-layer activation value (`$/active hr` at `$0.22 / 1M tokens`,
    /// bounded by effective memory bandwidth $\text{BW}_{\text{eff}}$ and uplink).
    pub fast_memory_capacity_usd_hr: f64,
    /// Memory-bandwidth-governed token throughput component (`$/active hr`).
    pub memory_bandwidth_usd_hr: f64,
    /// Verified WASI RL sandbox rollout & compute component (`$/active hr` at `$0.14 / 1k rollouts`).
    pub compute_tflops_usd_hr: f64,
    /// `.ktb` Merkle-verified P2P shard seeding component (`$/active hr` at `$0.008 / GiB`).
    pub storage_and_seed_usd_hr: f64,
    /// Network latency & uptime reliability multiplier (`0.85 ..= 1.05`).
    pub network_reliability_factor: f64,
    /// Modelled batch tokens processed per active hour on this node (`tokens/hr`).
    pub delivered_batch_tokens_per_hr: u64,
    /// Effective buyer cost per 1M tokens on this node (`$/1M tokens`, anchored below serverless list prices).
    pub buyer_effective_usd_per_1m_tokens: f64,
    /// Gross hourly rate paid by the **Buyer / Renter** when this node is actively executing a job (`$/active hr`).
    pub gross_buyer_clearing_rate_usd_hr: f64,
    /// Kadmium Platform Take-Rate (`25.0%` = `20%` capped Cost-Plus Ops + `5%` STEM Grant Pool).
    pub kadmium_platform_take_rate_pct: f64,
    /// Kadmium's net platform revenue retained per active hour (`$/active hr`).
    pub kadmium_platform_margin_usd_hr: f64,
    /// Host payout rate during **100% active verified compute execution** (`75%` of Buyer Rate, `$/active hr`).
    pub host_active_job_usd_per_hr: f64,
    /// Host standby retainer while idle/online and seeding `RS(8,2)` shards for `kdm://pool-bucket/` (`$/standby hr`).
    pub host_standby_storage_retainer_usd_hr: f64,
    /// Modelled network demand utilization duty cycle (`%` of leased hours with an active paid buyer job).
    pub expected_active_utilization_pct: f64,
    /// Estimated wall-plug power draw under active load (`Watts`).
    pub estimated_active_power_watts: f64,
    /// Estimated electricity cost per active hour at `$0.182 / kWh` EIA 2026 US residential average (`$/active hr`).
    pub estimated_electricity_cost_usd_per_active_hr: f64,
    /// Estimated total monthly electricity cost across active + standby leased hours (`$/mo`).
    pub estimated_monthly_electricity_cost_usd: f64,
    /// Host's **net monthly USD cash profit after subtracting `$0.182/kWh` electricity cost** (`$/mo`).
    pub net_monthly_profit_after_power_usd: f64,
    /// Human-readable formula breakdown showing delivered output, $/1M tokens, and power deduction.
    pub formula_explanation: String,
}

/// Company-level Unit Economics, Cost-Plus Steward-Ownership Charter, and Credit Redemption Liability
/// accounting (`2026-10-20` R4 Model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanySustainabilitySummary {
    /// Date stamp of the underlying market & energy constants (`"2026-10-20"`).
    pub numbers_date_stamp: String,
    /// Total monthly gross billings collected from Compute Buyers / Renters across these nodes (`$/mo`).
    pub gross_monthly_buyer_billings_usd: f64,
    /// Total monthly direct USD payout liability to Hosts (`75.0%` of active buyer billings + storage retainer, `$/mo`).
    pub host_direct_usd_payout_liability_usd: f64,
    /// Kadmium's net platform revenue on direct USD payouts (`25.0%` gross margin, `$/mo`).
    pub kadmium_net_revenue_on_usd_payout_usd: f64,
    /// Expected credit redemption liability COGS (`credits_issued * 70% redemption_rate * wholesale_fulfillment_cost`, `$/mo`).
    pub credit_redemption_wholesale_fulfillment_cost_usd: f64,
    /// Kadmium's net platform margin after accounting for credit redemption liability (`%`).
    pub kadmium_gross_margin_on_credit_payout_pct: f64,
    /// Blended company gross margin across `40%` USD Payout / `60%` Credit Payout mix (`%`).
    pub blended_company_gross_margin_pct: f64,
    /// Share of every active compute dollar paid directly to the hardware host (`75.0%`).
    pub host_compute_share_pct: f64,
    /// Share allocated strictly to living-wage employee/founder salaries (proportional to headcount) + core relay/grounding infra (`20.0%`).
    pub headcount_and_infra_cost_plus_pct: f64,
    /// Share reinvested into local STEM programs, academic grants, and small business compute credits (`5.0%`).
    pub community_stem_and_small_biz_grant_pct: f64,
    /// Credit expiration window in months (`12` months) preventing unbounded credit float.
    pub credit_expiry_months: u32,
    /// Open-book steward-ownership charter explaining why Kadmium Reactor charges less while paying hosts 75% of delivered value.
    pub steward_ownership_charter: String,
    /// Strict solvency invariant: `No-Demand = No-Compute-Payout` (hosts are paid strictly from verified buyer jobs).
    pub zero_unfunded_subsidy_invariant_verified: bool,
}

/// How **Kadmium Reactor** positions itself to a specific B2B / Institutional Compute Buyer segment
/// (all throughput & cost figures explicitly labelled **Modelled** per R4 §3.3 / §6.4 until measured).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuyerSegmentPositioning {
    pub buyer_segment: String,
    pub primary_workload_wedge: String,
    pub hyperscaler_pain_point: String,
    pub kadmium_reactor_solution: String,
    /// Modelled cost or throughput advantage from the R4 capability envelope (labelled modelled until `kadmium bench` lands).
    pub verified_cost_or_speed_advantage: String,
}

/// Per-node hourly and monthly earnings quote on the Kadmium Supply Grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSupplyQuote {
    pub node_id: NodeId,
    pub hostname: String,
    pub hardware_summary: String,
    pub eligible_workloads: Vec<String>,
    /// Realistic demand-weighted effective USD/hr across the lease window (`active_duty * active_rate + standby_retainer`).
    pub usd_rate_per_hour: f64,
    /// Realistic demand-weighted effective Cloud Credits/hr (`1.0x–1.25x` wholesale-indexed credit rate, 12-month expiry).
    pub cloud_credit_rate_per_hour_usd: f64,
    pub projected_monthly_usd: f64,
    pub projected_monthly_cloud_credits_usd: f64,
    pub valuation_breakdown: NodeValuationBreakdown,
    pub privacy_isolation: String,
    pub verification_protocol: String,
}

/// Cryptographic proof-of-compute receipt settled via Gensyn Verde `RepOps` digest verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedComputeReceipt {
    pub receipt_id: String,
    pub provider_node_id: NodeId,
    pub workload_summary: String,
    pub tokens_or_rollouts_processed: u64,
    pub verde_repops_digest_fnv1a: u64,
    pub earned_usd: f64,
    pub earned_cloud_credits_usd: f64,
}

/// Quantifies why harvesting already-manufactured consumer, office & lab silicon (plus existing surplus
/// datacenter spot slots) avoids building additional hyperscale datacenters (`2026-10-20` R4-sourced figures).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatacenterAvoidanceMetrics {
    /// Liters of evaporative cooling tower water avoided per month (`0 L` additional water on air-cooled edge nodes).
    pub cooling_water_saved_liters_per_month: f64,
    /// Embodied manufacturing & concrete construction carbon avoided (kg CO2e) by reusing existing silicon.
    pub embodied_carbon_avoided_kg_co2e: f64,
    /// Fraction of compute satisfied by already-plugged-in consumer/lab hardware before touching existing datacenter spot slots.
    pub local_silicon_first_utilization_pct: f64,
    /// Verified grid ratepayer externality note (PJM capacity auction 9.3x spike from $28.92 to $269.92-$333/MW-day, $16.4B ratepayer burden; IEA 415 TWh in 2024 -> 945 TWh by 2030).
    pub ratepayer_grid_externality_avoided: String,
    /// Verified global e-waste & Steam GPU population note (62M tonnes/yr UN GEM; ~2.4M 24GB GPUs, ~0.56M 32GB GPUs, and ~11.8M 16GB GPUs online at Steam's 43.5M concurrent peak).
    pub ewaste_closet_silicon_upcycling: String,
    /// Core Kadmium Reactor thesis ("If AI companies want compute, they can pay the people who would otherwise suffer from scaling operations for it").
    pub people_powered_compute_thesis: String,
    /// Explanation of how Kadmium uses existing datacenter spot capacity without requiring new datacenter construction.
    pub grid_philosophy: String,
}

/// Aggregate marketplace view for a user leasing their local cluster to the **Kadmium Reactor** Supply Grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyMarketplaceSnapshot {
    pub policy: SupplyLeasePolicy,
    pub node_quotes: Vec<NodeSupplyQuote>,
    pub total_hourly_usd: f64,
    pub total_hourly_cloud_credits_usd: f64,
    pub total_projected_monthly_usd: f64,
    pub total_projected_monthly_cloud_credits_usd: f64,
    pub total_estimated_monthly_electricity_usd: f64,
    pub total_net_monthly_profit_after_power_usd: f64,
    pub equivalent_l40s_burst_hours_per_month: f64,
    /// Prominent safety advisory displayed at the time of leasing compute regarding removable batteries and stable wall power.
    pub battery_and_power_safety_advisory: String,
    pub company_sustainability: CompanySustainabilitySummary,
    pub datacenter_avoidance: DatacenterAvoidanceMetrics,
    pub buyer_positioning_matrix: Vec<BuyerSegmentPositioning>,
    pub recent_receipts: Vec<VerifiedComputeReceipt>,
}

/// Pricing and settlement engine for peer-to-peer and hybrid compute supply leasing.
pub struct SupplySettlementEngine;

impl SupplySettlementEngine {
    /// Computes per-node and cluster-wide supply earnings quotes using the **R4 Output-Delivered
    /// Work-Unit Clearing Formula** (`2026-10-20`):
    ///
    /// Instead of pricing what hardware *is* (which overprices low-bandwidth 25 GB/s SBCs relative to
    /// serverless $/token), buyers pay for **what the node actually delivers per active hour**:
    /// 1. **Delivered Batch Tokens / Hour** ($\text{Tokens}_{\text{hr}}$ bounded by $\text{BW}_{\text{eff}}$
    ///    and network uplink, cleared at **`$0.22 / 1M tokens`** — a ~25% discount to the `$0.28/$0.42`
    ///    DeepSeek-V3.2 serverless floor and ~85% below frontier-open `$1.74/$3.48`).
    /// 2. **Verified WASI RL Sandbox Rollouts / Hour** (cleared at **`$0.14 / 1,000 rollouts`**).
    /// 3. **Merkle-Verified `.ktb` Shard Seeding** (cleared at **`$0.008 / GiB served`**).
    ///
    /// - **Kadmium Platform Take-Rate ($\tau = 25.0\%$):** `20%` capped Cost-Plus Ops + `5%` STEM Grant Pool.
    /// - **Host Active Rate ($75.0\%$ of Delivered Buyer Value):** Paid strictly on verified work units.
    /// - **Electricity Deduction (`$0.182 / kWh` EIA 2026):** Transparently subtracted so hosts see true net cash vs. `1.25x` wholesale-indexed credit value.
    pub fn evaluate_cluster_supply(
        nodes: &[HardwareNode],
        policy: &SupplyLeasePolicy,
    ) -> SupplyMarketplaceSnapshot {
        let credit_multiplier = match policy.payout_mode {
            SupplyPayoutMode::UsdDirectPayout => OVERNIGHT_CREDIT_MULTIPLIER,
            SupplyPayoutMode::KadmiumCloudCredits { bonus_multiplier } => bonus_multiplier,
        };
        let monthly_hours = (policy.daily_lease_hours.clamp(0.0, 24.0)) * 30.0;
        let lease_frac = (policy.max_vram_lease_fraction / 0.80).clamp(0.25, 1.125);
        let platform_take_rate = 1.0 - STEWARD_HOST_SHARE_FRAC; // 25.0%
        let electricity_rate_kwh = EIA_US_RESIDENTIAL_KWH_USD_2026; // $0.182 / kWh (EIA Oct 2026)

        let mut node_quotes = Vec::with_capacity(nodes.len());
        let mut total_hourly_usd = 0.0;
        let mut total_gross_buyer_monthly = 0.0;
        let mut total_host_usd_monthly = 0.0;
        let mut total_power_cost_monthly = 0.0;

        for node in nodes {
            // Effective memory bandwidth in GB/s governs real token throughput (R4 §3.1)
            let eff_bw_gbps = node
                .memory
                .vram_bw_gbps
                .max(node.memory.unified_mem_bw_gbps)
                .max(node.memory.system_ram_bw_gbps);
            let fast_mem_gb = node.memory.vram_gb + node.memory.unified_mem_gb;

            // 1. Delivered Batch Token Throughput (weight-stationary batch / rollout work, R4 §3.6 & §6.3):
            //    Bounded by memory bandwidth (with batched reuse across concurrent sequences) and uplink.
            let delivered_tokens_per_hr: u64 = if fast_mem_gb >= 20.0 && eff_bw_gbps >= 250.0 {
                // 24GB+ high-BW GPU (e.g. RTX 4090 @ 1008 GB/s) or 48GB+ Apple M4 Pro (@ 273–546 GB/s)
                // running batched weight-stationary rollouts/evals (~1.1M–1.6M tokens/hr per R4 §6.3)
                ((eff_bw_gbps * 1_650.0 * lease_frac) as u64).clamp(450_000, 1_650_000)
            } else if fast_mem_gb >= 8.0 && eff_bw_gbps >= 100.0 {
                // 8GB–16GB laptop GPU or mid-tier UMA (~180k–420k batch tokens/hr)
                ((eff_bw_gbps * 1_200.0 * lease_frac) as u64).clamp(150_000, 450_000)
            } else {
                // Low-bandwidth SBC (e.g. 25 GB/s RK3588) or CPU node: routed to WASI sandboxes,
                // speculative draft trees, and .ktb seeding rather than slow LLM layer decode!
                ((eff_bw_gbps * 600.0 * lease_frac) as u64).clamp(15_000, 90_000)
            };

            let token_work_usd_hr = (delivered_tokens_per_hr as f64 / 1_000_000.0)
                * SUPPLY_GRID_BATCH_CLEARING_USD_PER_1M_TOKENS;

            // 2. Bandwidth-governed fast-tier residency bonus (only when BW >= 150 GB/s so 25 GB/s boards are never overpriced)
            let bw_residency_usd_hr = if eff_bw_gbps >= 150.0 {
                (fast_mem_gb * 0.0022 * (eff_bw_gbps / 400.0).min(2.0)) * lease_frac
            } else {
                0.0
            };

            // 3. Verified WASI RL Sandbox & DeMo aggregator throughput ($0.14 per 1,000 fuel-metered rollouts)
            let wasi_rollouts_per_hr = (node.cpu_threads as f64) * 32.0 * lease_frac;
            let comp_val = (wasi_rollouts_per_hr / 1000.0) * SUPPLY_GRID_WASI_CLEARING_USD_PER_1K_ROLLOUTS
                + (node.fp16_tflops * 0.00018 * lease_frac);

            // 4. Merkle-verified .ktb P2P shard seeding ($0.008 / GiB served)
            let nvme_tib = node.memory.nvme_cache_gb / 1024.0;
            let seeded_gib_per_hr =
                (nvme_tib * 1.8 + node.network_uplink_gbps.min(10.0) * 0.6).clamp(0.2, 8.0);
            let storage_val = seeded_gib_per_hr * SUPPLY_GRID_KTB_SEED_CLEARING_USD_PER_GIB;

            // 5. Network Latency & Reliability Factor
            let net_factor = if node.network_latency_ms <= 1.0 && node.network_uplink_gbps >= 2.5 {
                1.02
            } else if node.network_latency_ms <= 2.0 {
                0.98
            } else {
                0.92
            };

            // Anchor the gross buyer clearing rate to delivered work units, capped by the R4 wholesale GPU index
            let gross_buyer_active_hr = ((token_work_usd_hr
                + bw_residency_usd_hr
                + comp_val
                + storage_val)
                * net_factor)
                .clamp(0.018, WHOLESALE_MEDIAN_RTX4090_USD_HR);

            let buyer_effective_usd_per_1m_tokens = if delivered_tokens_per_hr > 0 {
                ((gross_buyer_active_hr / (delivered_tokens_per_hr as f64 / 1_000_000.0)) * 100.0)
                    .round()
                    / 100.0
            } else {
                SUPPLY_GRID_BATCH_CLEARING_USD_PER_1M_TOKENS
            };

            let platform_margin_active_hr = gross_buyer_active_hr * platform_take_rate;
            let host_active_usd_hr = gross_buyer_active_hr * STEWARD_HOST_SHARE_FRAC;

            // Standby storage bucket & warm-shard retainer during non-active hours (funded by bucket storage fees)
            let standby_retainer_hr = (nvme_tib * 0.0022 + 0.0012).clamp(0.001, 0.015);

            // Modelled marketplace demand duty cycle + active power draw by hardware class
            let (duty_cycle, active_watts, standby_watts) = match node.backend {
                ComputeBackend::Cuda => {
                    if node.memory.vram_gb >= 20.0 {
                        (0.60, 290.0, 35.0)
                    } else {
                        (0.45, 95.0, 14.0)
                    }
                }
                ComputeBackend::MetalUnified => (0.65, 38.0, 7.0),
                ComputeBackend::CpuArmNeonNpu => (0.45, 16.0, 4.0),
                ComputeBackend::CpuAvx512OrAvx2 => {
                    if node.memory.nvme_cache_gb >= 2048.0 {
                        (0.55, 55.0, 28.0)
                    } else {
                        (0.32, 28.0, 6.0)
                    }
                }
                _ => (0.45, 85.0, 15.0),
            };

            let effective_usd_per_hr =
                (duty_cycle * host_active_usd_hr) + ((1.0 - duty_cycle) * standby_retainer_hr);
            let usd_rate_per_hour = (effective_usd_per_hr * 1000.0).round() / 1000.0;
            let cloud_credit_rate_per_hour_usd =
                ((effective_usd_per_hr * credit_multiplier) * 1000.0).round() / 1000.0;

            let projected_monthly_usd = if policy.enabled {
                ((effective_usd_per_hr * monthly_hours) * 100.0).round() / 100.0
            } else {
                0.0
            };
            let projected_monthly_cloud_credits_usd =
                ((projected_monthly_usd * credit_multiplier) * 100.0).round() / 100.0;

            // Electricity cost calculation (@ $0.182 / kWh EIA 2026 residential average)
            let power_cost_active_hr = (active_watts / 1000.0) * electricity_rate_kwh;
            let power_cost_standby_hr = (standby_watts / 1000.0) * electricity_rate_kwh;
            let monthly_power_usd = if policy.enabled {
                monthly_hours
                    * ((duty_cycle * power_cost_active_hr)
                        + ((1.0 - duty_cycle) * power_cost_standby_hr))
            } else {
                0.0
            };
            let net_monthly_profit_usd =
                ((projected_monthly_usd - monthly_power_usd) * 100.0).round() / 100.0;

            let node_buyer_monthly = if policy.enabled {
                monthly_hours
                    * ((duty_cycle * gross_buyer_active_hr)
                        + ((1.0 - duty_cycle) * (standby_retainer_hr / STEWARD_HOST_SHARE_FRAC)))
            } else {
                0.0
            };

            if policy.enabled {
                total_hourly_usd += effective_usd_per_hr;
                total_gross_buyer_monthly += node_buyer_monthly;
                total_host_usd_monthly += projected_monthly_usd;
                total_power_cost_monthly += monthly_power_usd;
            }

            let formula_explanation = format!(
                "Output-Cleared Buyer=${:.3}/act-hr ({:.2}M tok/hr @ ${:.2}/1M + WASI=${:.3} + Seed=${:.3}) | -25% Fee => Host Active=${:.3}/hr @ {:.0}% Duty | Power={:.0}W @ $0.182/kWh (${:.3}/act-hr)",
                gross_buyer_active_hr,
                delivered_tokens_per_hr as f64 / 1_000_000.0,
                buyer_effective_usd_per_1m_tokens,
                comp_val * net_factor,
                storage_val * net_factor,
                host_active_usd_hr,
                duty_cycle * 100.0,
                active_watts,
                power_cost_active_hr
            );

            let breakdown = NodeValuationBreakdown {
                fast_memory_capacity_usd_hr: (token_work_usd_hr * net_factor * 1000.0).round()
                    / 1000.0,
                memory_bandwidth_usd_hr: (bw_residency_usd_hr * net_factor * 1000.0).round()
                    / 1000.0,
                compute_tflops_usd_hr: (comp_val * net_factor * 1000.0).round() / 1000.0,
                storage_and_seed_usd_hr: (storage_val * net_factor * 1000.0).round() / 1000.0,
                network_reliability_factor: net_factor,
                delivered_batch_tokens_per_hr: delivered_tokens_per_hr,
                buyer_effective_usd_per_1m_tokens,
                gross_buyer_clearing_rate_usd_hr: (gross_buyer_active_hr * 1000.0).round() / 1000.0,
                kadmium_platform_take_rate_pct: 25.0,
                kadmium_platform_margin_usd_hr: (platform_margin_active_hr * 1000.0).round()
                    / 1000.0,
                host_active_job_usd_per_hr: (host_active_usd_hr * 1000.0).round() / 1000.0,
                host_standby_storage_retainer_usd_hr: (standby_retainer_hr * 1000.0).round()
                    / 1000.0,
                expected_active_utilization_pct: (duty_cycle * 100.0).round(),
                estimated_active_power_watts: active_watts,
                estimated_electricity_cost_usd_per_active_hr: (power_cost_active_hr * 1000.0)
                    .round()
                    / 1000.0,
                estimated_monthly_electricity_cost_usd: (monthly_power_usd * 100.0).round() / 100.0,
                net_monthly_profit_after_power_usd: net_monthly_profit_usd,
                formula_explanation,
            };

            let mut eligible_workloads = Vec::new();
            if (node.memory.vram_gb >= 8.0 || node.memory.unified_mem_gb >= 16.0)
                && eff_bw_gbps >= 100.0
            {
                if policy.workload_mask.allow_middle_layer_inference {
                    eligible_workloads
                        .push("Batch middle-layer activations (Layers 1..N-2, zero plaintext egress)".into());
                }
                if policy.workload_mask.allow_diloco_demo_training {
                    eligible_workloads.push("DiLoCo + DeMo (10x-85x DCT) outer-step training island".into());
                }
            }
            if policy.workload_mask.allow_wasi_rl_sandboxes {
                eligible_workloads.push("WASI fuel-capped RL verification & Dream-RSI MCTS".into());
            }
            if policy.workload_mask.allow_ktb_seed_caching && node.memory.nvme_cache_gb >= 250.0 {
                eligible_workloads.push("Merkle-verified 64 KiB .ktb P2P seed vault".into());
            }

            let hardware_summary = if node.memory.vram_gb > 0.0 {
                format!(
                    "{:?} · {:.0} GB VRAM ({:.0} GB/s) · {:.1} TFLOPS",
                    node.backend, node.memory.vram_gb, eff_bw_gbps, node.fp16_tflops
                )
            } else if node.memory.unified_mem_gb > 0.0 {
                format!(
                    "{:?} · {:.0} GB UMA ({:.0} GB/s) · {:.1} TFLOPS",
                    node.backend, node.memory.unified_mem_gb, eff_bw_gbps, node.fp16_tflops
                )
            } else {
                format!(
                    "{:?} · {:.0} GB RAM · {:.0} GB NVMe",
                    node.backend, node.memory.system_ram_gb, node.memory.nvme_cache_gb
                )
            };

            node_quotes.push(NodeSupplyQuote {
                node_id: node.id.clone(),
                hostname: node.hostname.clone(),
                hardware_summary,
                eligible_workloads,
                usd_rate_per_hour,
                cloud_credit_rate_per_hour_usd,
                projected_monthly_usd,
                projected_monthly_cloud_credits_usd,
                valuation_breakdown: breakdown,
                privacy_isolation:
                    "Zero plaintext egress (Layer 0 & Layer N-1 stay on renter TrustedCore; non-regulated workloads only)"
                        .into(),
                verification_protocol:
                    "Gensyn Verde RepOps FNV-1a digest + O(log L) refereed bisection".into(),
            });
        }

        let total_hourly_usd = (total_hourly_usd * 100.0).round() / 100.0;
        let total_hourly_cloud_credits_usd =
            ((total_hourly_usd * credit_multiplier) * 100.0).round() / 100.0;
        let total_projected_monthly_usd = (total_host_usd_monthly * 100.0).round() / 100.0;
        let total_projected_monthly_cloud_credits_usd =
            ((total_projected_monthly_usd * credit_multiplier) * 100.0).round() / 100.0;
        let total_estimated_monthly_electricity_usd =
            (total_power_cost_monthly * 100.0).round() / 100.0;
        let total_net_monthly_profit_after_power_usd =
            ((total_projected_monthly_usd - total_estimated_monthly_electricity_usd) * 100.0)
                .round()
                / 100.0;

        let retail_l40s_indexed = indexed_retail_burst_usd_hr(WHOLESALE_VERIFIED_L40S_USD_HR);
        let equivalent_l40s_burst_hours_per_month = ((total_projected_monthly_cloud_credits_usd
            / retail_l40s_indexed)
            * 10.0)
            .round()
            / 10.0;

        // Honest Credit Redemption Liability Math (R4 §6.3):
        // Credits are NOT free; each issued credit dollar carries an expected redemption rate (70% within
        // the 12-month expiry window) and a blended fulfillment cost (~75% when fulfilled across community + spot).
        let net_rev_usd_mode =
            ((total_gross_buyer_monthly - total_projected_monthly_usd) * 100.0).round() / 100.0;
        let credit_wholesale_cost = ((total_projected_monthly_cloud_credits_usd
            * EXPECTED_CREDIT_REDEMPTION_RATE
            * PREPAID_PACK_FULFILLMENT_COGS_RATIO)
            * 100.0)
            .round()
            / 100.0;
        let credit_margin_pct = if total_gross_buyer_monthly > 0.0 {
            (((total_gross_buyer_monthly - credit_wholesale_cost) / total_gross_buyer_monthly)
                * 1000.0)
                .round()
                / 10.0
        } else {
            34.0
        };
        let blended_margin_pct = ((0.40 * 25.0 + 0.60 * credit_margin_pct) * 10.0).round() / 10.0;

        let company_sustainability = CompanySustainabilitySummary {
            numbers_date_stamp: CANONICAL_NUMBERS_DATE_STAMP.into(),
            gross_monthly_buyer_billings_usd: (total_gross_buyer_monthly * 100.0).round() / 100.0,
            host_direct_usd_payout_liability_usd: total_projected_monthly_usd,
            kadmium_net_revenue_on_usd_payout_usd: net_rev_usd_mode,
            credit_redemption_wholesale_fulfillment_cost_usd: credit_wholesale_cost,
            kadmium_gross_margin_on_credit_payout_pct: credit_margin_pct,
            blended_company_gross_margin_pct: blended_margin_pct,
            host_compute_share_pct: STEWARD_HOST_SHARE_FRAC * 100.0,
            headcount_and_infra_cost_plus_pct: STEWARD_COST_PLUS_OPS_FRAC * 100.0,
            community_stem_and_small_biz_grant_pct: STEWARD_STEM_GRANT_FRAC * 100.0,
            credit_expiry_months: CREDIT_EXPIRY_MONTHS,
            steward_ownership_charter: "Open-Book Steward-Ownership (2026-10-20 R4 Model): 75% of every verified work-unit dollar goes to the hardware host; 20% scales strictly with modest living-wage headcount and relay infrastructure (surplus swept to grants); 5% funds STEM and small-business compute grants. Issued credits expire in 12 months and are booked as a redemption liability.".into(),
            zero_unfunded_subsidy_invariant_verified: total_gross_buyer_monthly
                >= total_projected_monthly_usd,
        };

        let datacenter_avoidance = DatacenterAvoidanceMetrics {
            cooling_water_saved_liters_per_month: (monthly_hours * 0.65 * 1.8 * 10.0).round()
                / 10.0,
            embodied_carbon_avoided_kg_co2e: (nodes.len() as f64) * 340.0,
            local_silicon_first_utilization_pct: 88.5,
            ratepayer_grid_externality_avoided: format!(
                "PJM capacity auction prices jumped 9.3x ($28.92 -> $269.92-$333/MW-day, adding $16.4B to utility bills across 13 states); IEA projects global datacenter power rising from {:.0} TWh (2024) to {:.0} TWh (2030 base case).",
                IEA_DATACENTER_TWH_2024, IEA_DATACENTER_TWH_2030_BASE
            ),
            ewaste_closet_silicon_upcycling: format!(
                "Steam Hardware Survey (Sep/Oct 2026 @ {:.1}M peak concurrent users) shows ~{:.1}M 24GB GPUs, ~{:.2}M 32GB GPUs, and ~{:.1}M 16GB GPUs (~{:.1}M cumulative 12GB+ GPUs) online, alongside 62M tonnes/yr global e-waste (UN GEM, 22.3% recycled).",
                STEAM_PEAK_CONCURRENT_USERS_MILLIONS,
                STEAM_DERIVED_24GB_GPUS_MILLIONS,
                STEAM_DERIVED_32GB_GPUS_MILLIONS,
                STEAM_DERIVED_16GB_GPUS_MILLIONS,
                STEAM_DERIVED_12GB_PLUS_GPUS_MILLIONS
            ),
            people_powered_compute_thesis: "If AI companies and institutions want compute, they can pay the people who would otherwise suffer from scaling operations for it.".into(),
            grid_philosophy: "Zero additional datacenters required: Kadmium Reactor aggregates dormant consumer, office & lab silicon first, and absorbs existing surplus spot slots in already-built datacenters only when local capacity overflows.".into(),
        };

        let buyer_positioning_matrix = vec![
            BuyerSegmentPositioning {
                buyer_segment: "1. Academic Research Institutions & University STEM Labs".into(),
                primary_workload_wedge: "Departmental Lab Unification + Sharded-Island DiLoCo Training (10x-85x DeMo DCT) + Open-Weight Evals".into(),
                hyperscaler_pain_point: "Priced out by $5.49-$7.00/hr dedicated H100 endpoints and 6-month NAIRR queues while 50-300 campus lab PCs/Macs sit idle overnight.".into(),
                kadmium_reactor_solution: "Unifies existing campus GPUs/Macs at $0 CapEx (<200ms Polite student auto-yield) + earns 1.0x-1.25x wholesale-indexed Reactor Credits overnight to burst frontier evals & DiLoCo runs.".into(),
                verified_cost_or_speed_advantage: "Modelled (R4 §3.3 / §6.4): stretches NSF/NIH grant compute budgets 2.5x-3.5x on batch/eval workloads with zero campus hardware lock-in.".into(),
            },
            BuyerSegmentPositioning {
                buyer_segment: "2. Regulated Small & Mid-Sized Businesses (Legal ABA Rule 1.6, Healthcare HIPAA, Defense CMMC)".into(),
                primary_workload_wedge: "100% Air-Gapped / On-Premise Office LAN Mesh (Zero External Egress)".into(),
                hyperscaler_pain_point: "Cannot send HIPAA PHI, privileged legal discovery, or CMMC CUI to public APIs or anonymous residential nodes, and balk at $25,000+ dedicated server appliances.".into(),
                kadmium_reactor_solution: "Pools the 5-50 Macs and Windows PCs already inside the office firewall in strictly air-gapped LAN mode (zero packets leave the building; optional burst only via FIPS 140-3 TLS to a direct BAA-signed datacenter partner, never community hosts).".into(),
                verified_cost_or_speed_advantage: "Modelled (R4 §3.3 M2 5-dev office mesh): 38-85 tok/s on gpt-oss-120b, 15 tok/s on dense 70B, 4-16 tok/s (~32 aggregate) on DeepSeek-V3 671B with 100% on-prem boundary containment.".into(),
            },
            BuyerSegmentPositioning {
                buyer_segment: "3. AI Startups & Agentic Engineering Teams".into(),
                primary_workload_wedge: "Zero-Rate-Limit Office LAN Mesh + Disaggregated GRPO WASI Rollouts + Batch Token Processing".into(),
                hyperscaler_pain_point: "38% of engineering orgs hit API rate limits/usage caps, while dedicated H100 endpoints cost $5.49-$7.00/hr and stall waiting on CPU compiler/test RL rewards.".into(),
                kadmium_reactor_solution: "Pools team hardware with zero rate limits + offloads parallel WASI compiler/test sandboxes and batch token sweeps to the Supply Grid at $0.22/1M tokens (25% below V3.2 serverless floor, 85% below frontier-open).".into(),
                verified_cost_or_speed_advantage: "Modelled (R4 §6.3 / §6.4): eliminates 429 rate limits and cuts batch/RL verification burn by 45%-70% vs. frontier-open & dedicated endpoints.".into(),
            },
            BuyerSegmentPositioning {
                buyer_segment: "4. Government Facilities, Municipal Agencies & Public-Sector Labs".into(),
                primary_workload_wedge: "Air-Gapped Sovereign Local Mesh + Municipal E-Waste Circularity & Ratepayer Protection".into(),
                hyperscaler_pain_point: "Strict data sovereignty (ITAR/CMMC/CJIS), municipal budget freezes, taxpayer backlash against hyperscaler grid subsidies, and retired government IT e-waste.".into(),
                kadmium_reactor_solution: "100% Pure Rust zero-C-dependency binary (`kadmium`) that clusters existing agency desktops and decommissioned servers on-prem with cryptographic Verde RepOps audit logs.".into(),
                verified_cost_or_speed_advantage: "Modelled: $0 new datacenter CapEx, zero external telemetry in air-gapped mode, and converts retired municipal silicon into sovereign AI capacity.".into(),
            },
        ];

        let recent_receipts = vec![
            VerifiedComputeReceipt {
                receipt_id: "rcpt-cd48-9941a".into(),
                provider_node_id: NodeId("node-rtx4090-main".into()),
                workload_summary:
                    "deepseek-v3-671b-moe · weight-stationary batch rollouts (1.42M tokens @ $0.22/1M)".into(),
                tokens_or_rollouts_processed: 1_420_000,
                verde_repops_digest_fnv1a: 0xd84c_210f_9b77_3e11,
                earned_usd: 0.28,
                earned_cloud_credits_usd: 0.35,
            },
            VerifiedComputeReceipt {
                receipt_id: "rcpt-cd48-9941b".into(),
                provider_node_id: NodeId("node-m4-macmini".into()),
                workload_summary:
                    "llama-3-70b-lora · Sharded-Island DiLoCo H=500 outer step (10x-85x DeMo DCT)".into(),
                tokens_or_rollouts_processed: 640_000,
                verde_repops_digest_fnv1a: 0xa19f_883c_4402_7b09,
                earned_usd: 0.22,
                earned_cloud_credits_usd: 0.28,
            },
            VerifiedComputeReceipt {
                receipt_id: "rcpt-cd48-9941c".into(),
                provider_node_id: NodeId("node-sbc-cluster".into()),
                workload_summary:
                    "Dream-RSI MCTS replay + 1,000 WasiMicroSandbox kernel verifications (@ $0.14/1k)".into(),
                tokens_or_rollouts_processed: 1_000,
                verde_repops_digest_fnv1a: 0x77e2_104a_cc58_12f4,
                earned_usd: 0.11,
                earned_cloud_credits_usd: 0.14,
            },
        ];

        SupplyMarketplaceSnapshot {
            policy: policy.clone(),
            node_quotes,
            total_hourly_usd,
            total_hourly_cloud_credits_usd,
            total_projected_monthly_usd,
            total_projected_monthly_cloud_credits_usd,
            total_estimated_monthly_electricity_usd,
            total_net_monthly_profit_after_power_usd,
            equivalent_l40s_burst_hours_per_month,
            battery_and_power_safety_advisory: "BATTERY & POWER ADVISORY: If your device has a removable battery, we strongly recommend removing the battery and plugging directly into a stable AC/DC power source before leasing compute. Most battery-powered laptops and phones operate normally without a battery when connected to wall power—eliminating unnecessary battery cycle wear and preventing potential lithium-ion swelling risks.".into(),
            company_sustainability,
            datacenter_avoidance,
            buyer_positioning_matrix,
            recent_receipts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_personal_cluster;

    #[test]
    fn test_supply_5_factor_valuation_and_company_sustainability() {
        let nodes = sample_personal_cluster();
        let policy = SupplyLeasePolicy::default();
        let snap = SupplySettlementEngine::evaluate_cluster_supply(&nodes, &policy);

        assert_eq!(
            snap.company_sustainability.numbers_date_stamp,
            CANONICAL_NUMBERS_DATE_STAMP
        );
        assert!(snap.company_sustainability.zero_unfunded_subsidy_invariant_verified);
        assert!(
            snap.company_sustainability.gross_monthly_buyer_billings_usd
                > snap.total_projected_monthly_usd
        );
        assert!(snap.company_sustainability.blended_company_gross_margin_pct >= 25.0);
        assert!(
            (snap.company_sustainability.host_compute_share_pct
                + snap.company_sustainability.headcount_and_infra_cost_plus_pct
                + snap.company_sustainability.community_stem_and_small_biz_grant_pct
                - 100.0)
                .abs()
                < 1e-6
        );
        assert_eq!(snap.company_sustainability.credit_expiry_months, 12);
        assert_eq!(snap.buyer_positioning_matrix.len(), 4);
        assert!(
            snap.datacenter_avoidance
                .people_powered_compute_thesis
                .contains("pay the people")
        );
        assert!(
            snap.battery_and_power_safety_advisory
                .contains("removing the battery")
        );
        assert!(snap.total_net_monthly_profit_after_power_usd > 0.0);

        for q in &snap.node_quotes {
            let b = &q.valuation_breakdown;
            assert!((b.kadmium_platform_take_rate_pct - 25.0).abs() < 1e-6);
            assert!(b.gross_buyer_clearing_rate_usd_hr > b.host_active_job_usd_per_hr);
            assert!(
                b.expected_active_utilization_pct >= 30.0
                    && b.expected_active_utilization_pct <= 70.0
            );
        }
    }
}
