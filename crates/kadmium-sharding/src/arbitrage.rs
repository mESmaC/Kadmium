//! # `kadmium_sharding::arbitrage`
//!
//! Multi-datacenter spot & surplus GPU arbitrage (`HybridCloudArbitragePlanner`) and
//! bidirectional `< 400 ms` local $\leftrightarrow$ cloud failover shield simulations.

use kadmium_core::{ComputeProviderKind, FailoverDirection, ModelSpec};
use serde::{Deserialize, Serialize};

/// A fragmented or spot compute instance offer from an external datacenter provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderComputeOffer {
    pub offer_id: String,
    pub provider_name: String,
    pub provider_kind: ComputeProviderKind,
    pub instance_description: String,
    pub fp16_tflops: f64,
    pub vram_or_unified_gb: f64,
    pub system_ram_gb: f64,
    pub hourly_spot_usd: f64,
    /// Equivalent monolithic hyperscaler on-demand price in USD/hr for comparison.
    pub hyperscaler_equivalent_usd: f64,
}

/// Multi-provider arbitrage plan stitching fragmented cheap instances + local hardware together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiProviderArbitragePlan {
    pub target_model_id: String,
    pub quote_source: String,
    pub live_provider_quotes: bool,
    pub simulation_only: bool,
    pub selected_offers: Vec<ProviderComputeOffer>,
    pub total_combined_tflops: f64,
    pub total_combined_vram_gb: f64,
    pub kadmium_blended_hourly_usd: f64,
    pub monolithic_hyperscaler_hourly_usd: f64,
    pub cost_reduction_pct: f64,
}

/// A single simulated outage and automatic live shard failover event (`Local <-> Cloud`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverShieldSimulation {
    pub direction: FailoverDirection,
    pub simulation_only: bool,
    pub incident_summary: String,
    pub affected_layers: (u32, u32),
    pub source_environment: String,
    pub target_failover_environment: String,
    /// Recovery Time Objective (RTO) achieved via Warm Vault `.ktb` pre-staging + live delta handoff.
    pub switchover_rto_ms: f64,
    /// Zero dropped requests or preserved training step continuity.
    pub zero_downtime_preserved: bool,
    pub temporary_failover_hourly_usd: f64,
}

/// Planner for Kadmium's Hybrid Cloud Arbitrage & Downtime Protection Shield.
pub struct HybridCloudArbitragePlanner;

impl HybridCloudArbitragePlanner {
    /// Builds a reference multi-datacenter arbitrage plan that stitches fragmented, low-cost single-GPU
    /// spot instances and high-RAM bare-metal servers across 4 different providers using
    /// Kadmium's asymmetric sharding + DeMo/DisTrO low-bandwidth compression.
    pub fn compute_arbitrage_plan(model: &ModelSpec) -> MultiProviderArbitragePlan {
        let offers = vec![
            ProviderComputeOffer {
                offer_id: "arb-local-01".into(),
                provider_name: "User Local Cluster (Main PC + M4 Mac Mini + NAS)".into(),
                provider_kind: ComputeProviderKind::SelfHostedLocalOnPrem,
                instance_description: "RTX 4090 24GB + M4 Pro 48GB Unified + 40TB NAS Seed".into(),
                fp16_tflops: 210.8,
                vram_or_unified_gb: 72.0,
                system_ram_gb: 128.0,
                hourly_spot_usd: 0.00,
                hyperscaler_equivalent_usd: 4.80,
            },
            ProviderComputeOffer {
                offer_id: "arb-spot-02".into(),
                provider_name: "Specialist GPU Spot #1 (Stranded Single L40S)".into(),
                provider_kind: ComputeProviderKind::SpecialistGpuSpotMarket,
                instance_description: "1x NVIDIA L40S 48GB VRAM (No NVLink required)".into(),
                fp16_tflops: 362.0,
                vram_or_unified_gb: 48.0,
                system_ram_gb: 96.0,
                hourly_spot_usd: 0.69,
                hyperscaler_equivalent_usd: 2.85,
            },
            ProviderComputeOffer {
                offer_id: "arb-spot-03".into(),
                provider_name: "Green Stranded-Energy DC (Preemptible 1x A100-80GB)".into(),
                provider_kind: ComputeProviderKind::StrandedEnergyDatacenter,
                instance_description: "1x NVIDIA A100 80GB PCIe (Spot Preemptible)".into(),
                fp16_tflops: 312.0,
                vram_or_unified_gb: 80.0,
                system_ram_gb: 128.0,
                hourly_spot_usd: 0.89,
                hyperscaler_equivalent_usd: 4.10,
            },
            ProviderComputeOffer {
                offer_id: "arb-baremetal-04".into(),
                provider_name: "Regional Bare-Metal (Hetzner/OVH High-RAM NVMe Host)".into(),
                provider_kind: ComputeProviderKind::RegionalBareMetalProvider,
                instance_description:
                    "AMD EPYC 256GB DDR5 + 2x Gen5 NVMe (Warm/Cold MoE Expert Host)".into(),
                fp16_tflops: 18.5,
                vram_or_unified_gb: 0.0,
                system_ram_gb: 256.0,
                hourly_spot_usd: 0.24,
                hyperscaler_equivalent_usd: 1.95,
            },
        ];

        let total_tflops: f64 = offers.iter().map(|o| o.fp16_tflops).sum();
        let total_vram: f64 = offers.iter().map(|o| o.vram_or_unified_gb).sum();
        let blended_usd: f64 = offers.iter().map(|o| o.hourly_spot_usd).sum();
        let hyper_usd: f64 = offers.iter().map(|o| o.hyperscaler_equivalent_usd).sum();
        let savings_pct = (((hyper_usd - blended_usd) / hyper_usd.max(0.01)) * 100.0).max(0.0);

        MultiProviderArbitragePlan {
            target_model_id: model.id.clone(),
            quote_source: "modelled_reference_catalog".into(),
            live_provider_quotes: false,
            simulation_only: true,
            selected_offers: offers,
            total_combined_tflops: (total_tflops * 10.0).round() / 10.0,
            total_combined_vram_gb: (total_vram * 10.0).round() / 10.0,
            kadmium_blended_hourly_usd: (blended_usd * 100.0).round() / 100.0,
            monolithic_hyperscaler_hourly_usd: (hyper_usd * 100.0).round() / 100.0,
            cost_reduction_pct: (savings_pct * 10.0).round() / 100.0 * 10.0,
        }
    }

    /// Simulates the three stages of Kadmium's **Bidirectional Downtime Protection Shield**:
    /// 1. Local Cluster Outage -> Automatic Cloud Standby Failover
    /// 2. Cloud Provider Spot Preemption / Outage -> Cross-Provider + Local Fallback
    /// 3. Local Cluster Recovery -> Automatic Failback to Zero-Marginal-Cost Local Hardware
    pub fn simulate_bidirectional_failover_shield() -> Vec<FailoverShieldSimulation> {
        vec![
            FailoverShieldSimulation {
                direction: FailoverDirection::LocalOutageToCloudStandby,
                simulation_only: true,
                incident_summary:
                    "Home ISP fiber drop / power blip on `cameron-main-rig` (Layers L00..L25)"
                        .into(),
                affected_layers: (0, 25),
                source_environment: "Local Home Cluster (Main PC + NAS)".into(),
                target_failover_environment:
                    "Kadmium Cloud Standby (Pre-warmed .ktb Vault + Spot L40S)".into(),
                switchover_rto_ms: 340.0,
                zero_downtime_preserved: true,
                temporary_failover_hourly_usd: 0.69,
            },
            FailoverShieldSimulation {
                direction: FailoverDirection::CloudOutageOrPreemptionToLocalAndMultiCloud,
                simulation_only: true,
                incident_summary:
                    "30s Spot Eviction Notice on Cloud A100 instance (Layers L26..L55)".into(),
                affected_layers: (26, 55),
                source_environment: "Cloud Provider A (Preemptible A100 Spot)".into(),
                target_failover_environment: "Local M4 Mac Mini + Gaming Laptops + Provider B Spot"
                    .into(),
                switchover_rto_ms: 185.0,
                zero_downtime_preserved: true,
                temporary_failover_hourly_usd: 0.42,
            },
            FailoverShieldSimulation {
                direction: FailoverDirection::CloudFailbackToRecoveredLocal,
                simulation_only: true,
                incident_summary:
                    "Local ISP & Main PC back online -> draining Cloud Standby back to Local".into(),
                affected_layers: (0, 25),
                source_environment: "Kadmium Cloud Standby (Spot L40S)".into(),
                target_failover_environment: "Local Home Cluster ($0.00/hr marginal compute)"
                    .into(),
                switchover_rto_ms: 210.0,
                zero_downtime_preserved: true,
                temporary_failover_hourly_usd: 0.00,
            },
        ]
    }
}
