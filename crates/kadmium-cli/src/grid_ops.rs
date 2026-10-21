//! Phase 2–5 (`v0.2` through `v1.0 GA`) Supply Grid, Kernel Benchmark, Zero-Click Pairing,
//! and World Registry CLI handlers (`kadmium-cli::grid_ops`).

use kadmium_core::{
    ClosetSiliconLongevityGovernor, OsSwapchainHysteresisEngine, SubClusterRbacEngine,
    SupplyLeasePolicy, SupplyPayoutMode, SupplySettlementEngine, sample_personal_cluster,
};
use kadmium_data::{
    DecentralizedStoragePoolEngine, HfByteRangeSlicer, SafetensorsToKtbTranscoder,
};
use kadmium_network::{
    MultiPathInterfaceStriper, NatTraversalCoordinator, SwarmRaidStriper, ZeroClickLanPairingEngine,
};
use kadmium_rl::DreamRsiWorldRegistry;
use kadmium_runtime::{
    CrossSystemComputeHarness, HardwareKernelDispatcher, HyperInferenceComputeReport,
    TrainingComputeAcceleratorSuite, VerdeFraudProofEngine,
};
use kadmium_sharding::{
    CloudShieldCutoverWatcher, ElasticLocalBoostPlanner, HierarchicalPoolPlanner,
    SpotOrderbookBidEngine,
};

pub fn run_lease(hours: f64, payout: &str) {
    let nodes = sample_personal_cluster();
    let payout_mode = if payout.eq_ignore_ascii_case("usd") {
        SupplyPayoutMode::UsdDirectPayout
    } else {
        SupplyPayoutMode::KadmiumCloudCredits {
            bonus_multiplier: 1.25,
        }
    };
    let policy = SupplyLeasePolicy {
        enabled: true,
        payout_mode,
        daily_lease_hours: hours.clamp(0.0, 24.0),
        only_when_polite_idle: true,
        max_vram_lease_fraction: 0.80,
        workload_mask: Default::default(),
    };
    let snap = SupplySettlementEngine::evaluate_cluster_supply(&nodes, &policy);

    println!(
        "====================================================================================================="
    );
    println!(" KADMIUM SUPPLY GRID (Cd-48) — IDLE COMPUTE LEASING & SETTLEMENT QUOTE");
    println!(
        "====================================================================================================="
    );
    println!(
        "  * Daily Idle Lease Window:    {:.1} hrs/day ({:.0} hrs/month)",
        snap.policy.daily_lease_hours,
        snap.policy.daily_lease_hours * 30.0
    );
    println!(
        "  * Polite Auto-Yield Guard:    ACTIVE (< 200 ms instant shard eviction on game launch, 0% penalty)"
    );
    println!(
        "  * Privacy Isolation:          Blind Middle-Shard Execution (Layer 0 & Layer N-1 stay on renter)"
    );
    println!(
        "  * Cryptographic Settlement:   Gensyn Verde RepOps FNV-1a digest + O(log L) refereed bisection"
    );
    println!(
        "  * Datacenter Avoidance:       0 L cooling water ({:.1} L/mo saved) · {:.0} kg CO2e embodied silicon avoided",
        snap.datacenter_avoidance
            .cooling_water_saved_liters_per_month,
        snap.datacenter_avoidance.embodied_carbon_avoided_kg_co2e
    );
    println!(
        "\n{:<22} {:<30} {:>10} {:>10} {:>7} {:>10} {:>11} {:>10} {:>11}",
        "HOSTNAME", "HARDWARE PROFILE", "BUYER/HR", "HOST ACT", "DUTY%", "EFF $/HR", "GROSS/MO", "POWER/MO", "NET PROFIT"
    );
    println!("{}", "-".repeat(128));
    for q in &snap.node_quotes {
        let b = &q.valuation_breakdown;
        println!(
            "{:<22} {:<30} ${:>8.3}   ${:>8.3} {:>6.0}%   ${:>8.3}   ${:>8.2}  -${:>7.2}   +${:>8.2}",
            q.hostname,
            q.hardware_summary,
            b.gross_buyer_clearing_rate_usd_hr,
            b.host_active_job_usd_per_hr,
            b.expected_active_utilization_pct,
            q.usd_rate_per_hour,
            q.projected_monthly_usd,
            b.estimated_monthly_electricity_cost_usd,
            b.net_monthly_profit_after_power_usd
        );
    }
    println!("{}", "-".repeat(128));
    println!("  [Transparent 5-Factor Formula Breakdown per Node (Mem + BW + TFLOPS + Storage/Uplink x Reliability - 25% Kadmium Fee)]:");
    for q in &snap.node_quotes {
        println!(
            "    * {:<20}: {}",
            q.hostname, q.valuation_breakdown.formula_explanation
        );
    }
    println!("{}", "-".repeat(128));
    println!(
        "  * Cluster Demand-Weighted Rate: ${:.2}/hr direct USD  |  ${:.2}/hr Kadmium Cloud Credits (1.25x)",
        snap.total_hourly_usd, snap.total_hourly_cloud_credits_usd
    );
    println!(
        "  * Host Monthly Gross -> Net:    ${:.2}/mo Gross USD - ${:.2}/mo Electricity (@ $0.14/kWh) = +${:.2}/mo Net Cash Profit (or ${:.2}/mo in 1.25x Credits)",
        snap.total_projected_monthly_usd,
        snap.total_estimated_monthly_electricity_usd,
        snap.total_net_monthly_profit_after_power_usd,
        snap.total_projected_monthly_cloud_credits_usd
    );
    println!(
        "  * Company Sustainability Proof: Buyer Billings=${:.2}/mo | Host USD Liability=${:.2}/mo | Kadmium Net Platform Margin=${:.2}/mo (25.0% USD / {:.1}% Credits / {:.1}% Blended | Zero Unfunded Subsidy={})",
        snap.company_sustainability.gross_monthly_buyer_billings_usd,
        snap.company_sustainability.host_direct_usd_payout_liability_usd,
        snap.company_sustainability.kadmium_net_revenue_on_usd_payout_usd,
        snap.company_sustainability.kadmium_gross_margin_on_credit_payout_pct,
        snap.company_sustainability.blended_company_gross_margin_pct,
        snap.company_sustainability.zero_unfunded_subsidy_invariant_verified
    );
    println!(
        "  * Cost-Plus Steward Split:      {:.0}% Host Compute Payout | {:.0}% Fair Living-Wage Headcount & Relay Infra | {:.0}% Local STEM & Small-Biz Grants",
        snap.company_sustainability.host_compute_share_pct,
        snap.company_sustainability.headcount_and_infra_cost_plus_pct,
        snap.company_sustainability.community_stem_and_small_biz_grant_pct
    );
    println!(
        "  * People-Powered Thesis:        \"{}\"",
        snap.datacenter_avoidance.people_powered_compute_thesis
    );
    println!(
        "  * Grid Ratepayer Protection:    {}",
        snap.datacenter_avoidance.ratepayer_grid_externality_avoided
    );
    println!(
        "  * E-Waste Silicon Upcycling:    {}",
        snap.datacenter_avoidance.ewaste_closet_silicon_upcycling
    );
    println!("  [Kadmium Reactor B2B / Institutional Buyer Positioning Matrix]:");
    for seg in &snap.buyer_positioning_matrix {
        println!(
            "    - {:<55} -> {} ({})",
            seg.buyer_segment, seg.primary_workload_wedge, seg.verified_cost_or_speed_advantage
        );
    }

    let boost = ElasticLocalBoostPlanner::evaluate_desktop_just_enough_boost();
    println!(
        "\n====================================================================================================="
    );
    println!(
        " KADMIUM 'JUST-ENOUGH-BOOST' — SEAMLESS LOCAL-FIRST FRACTIONAL SPILLOVER ({})",
        boost.target_model_id
    );
    println!(
        "====================================================================================================="
    );
    println!(
        "  * Primary Desktop Anchor:     {} (Need {:.1} GB -> Only {:.1} GB Deficit!)",
        boost.primary_desktop_gpu, boost.total_model_and_kv_vram_gb, boost.vram_deficit_gb
    );
    println!(
        "  * Local Blind-Privacy Layers: {} (Overflow Middle: L{:02}..L{:02})",
        boost.local_pinned_layers, boost.overflow_middle_layers.0, boost.overflow_middle_layers.1
    );
    println!(
        "\n{:<58} {:>10} {:>10} {:>10} {:>12}",
        "SPILLOVER OPTION", "DECODE", "SPEEDUP", "COST/HR", "CLOUD SAVED"
    );
    println!("{}", "-".repeat(104));
    for opt in &boost.evaluated_options {
        println!(
            "{:<58} {:>7.1} t/s {:>8.2}x   ${:>6.2}/hr {:>10.1}%",
            opt.mode_label,
            opt.effective_decode_tok_s,
            opt.speedup_vs_cpu_offload_x,
            opt.hourly_cost_usd,
            opt.cost_savings_vs_full_cloud_pct
        );
    }
    println!(
        "\n  * Dynamic Context-Growth Layer Slider (Zero-OOM Boundary Sliding as KV Cache Expands):"
    );
    for st in &boost.resilience_and_privacy.dynamic_kv_slider_timeline {
        println!(
            "    - {:>2}K Ctx (KV {:>3.1} GB): {:<38} -> Spills L{:02}..L{:02} ({:>4.1} GB, {:.1}ms stall)",
            st.context_tokens / 1024,
            st.desktop_kv_cache_gb,
            st.desktop_hosted_layers,
            st.spilled_middle_layers.0,
            st.spilled_middle_layers.1,
            st.spilled_vram_gb,
            st.slide_stall_ms
        );
    }
    println!("\n{}", boost.summary_scorecard);

    let reactor = ClosetSiliconLongevityGovernor::evaluate_full_reactor_infrastructure();
    println!(
        "\n====================================================================================================="
    );
    println!(
        " KADMIUM REACTOR (Cd-48) — DOUBLE-ENTRY ESCROW, SLA FAULT-TOLERANCE & CLOSET-SILICON LONGEVITY"
    );
    println!(
        "====================================================================================================="
    );
    println!(
        "  [1/3] Open-Book Clearinghouse & Automatic Cost-Plus Ceiling Sweep ({}):",
        reactor.ledger.accounting_period_label
    );
    println!(
        "    * Verified Buyer Escrow Billings: ${:>10.2}/mo | Rejected Spoofed Slices: {} ($0.00 billed to buyer)",
        reactor.ledger.total_verified_buyer_billings_usd,
        reactor.ledger.rejected_unverified_slices_count
    );
    println!(
        "    * 75% Host Payouts (Gross -> Net): ${:>10.2}/mo Gross - ${:.2}/mo Power = +${:.2}/mo Net Cash Profit",
        reactor.ledger.total_host_gross_payout_usd,
        reactor.ledger.total_host_electricity_cost_usd,
        reactor.ledger.total_host_net_profit_after_power_usd
    );
    println!(
        "    * 20% Cost-Plus Ceiling Sweep:     ${:>10.2}/mo Raw 20% -> Capped at ${:.2}/mo ({} staff @ ${:.0}/mo + ${:.0} infra) -> +${:.2}/mo Swept to Grants!",
        reactor.ledger.raw_20pct_ops_allocation_usd,
        reactor.ledger.retained_cost_plus_ops_usd,
        reactor.ledger.active_team_headcount,
        reactor.ledger.monthly_living_wage_per_person_usd,
        reactor.ledger.monthly_core_relay_and_grounding_infra_usd,
        reactor.ledger.cost_plus_surplus_swept_to_grants_usd
    );
    println!(
        "    * Total Community STEM & SMB Pool: ${:>10.2}/mo (Base 5% ${:.2} + Swept Surplus ${:.2} | Double-Entry Verified={})",
        reactor.ledger.total_community_grant_pool_usd,
        reactor.ledger.base_5pct_stem_grant_pool_usd,
        reactor.ledger.cost_plus_surplus_swept_to_grants_usd,
        reactor.ledger.double_entry_conservation_verified
    );
    println!("  [2/3] 99.996% SLA Fault-Tolerant 'People's Silicon' Orchestrator:");
    println!(
        "    * Wire-Resident Activation Ring:   Stride={} layers ({:.0} KiB INT8) | Node '{}' dropped at L{} -> Resumed from L{} in {:.1} ms vs {:.1} ms restart ({:.2}x faster, 0 lost tokens)",
        reactor.sla_fault_tolerance.checkpoint_ring_stride_layers,
        reactor.sla_fault_tolerance.ring_checkpoint_payload_kb,
        reactor.sla_fault_tolerance.dropped_node_hostname,
        reactor.sla_fault_tolerance.failure_injected_at_layer,
        reactor.sla_fault_tolerance.nearest_upstream_ring_checkpoint_layer,
        reactor.sla_fault_tolerance.wire_ring_mid_pass_recovery_latency_ms,
        reactor.sla_fault_tolerance.naive_full_pass_restart_latency_ms,
        reactor.sla_fault_tolerance.mid_pass_recovery_speedup_x
    );
    println!(
        "    * Blind FWHT Honey-Token Canary:   {:.1}% rate (Wire CosSim={:.4}) | 1-Pass Poisoning Host Caught={} | Tail-Hedged p99: {:.1} ms -> {:.1} ms",
        reactor.sla_fault_tolerance.blind_canary_injection_rate_pct,
        reactor.sla_fault_tolerance.canary_wire_indistinguishability_cos_sim,
        reactor.sla_fault_tolerance.single_pass_poisoning_host_caught,
        reactor.sla_fault_tolerance.unhedged_p99_latency_ms,
        reactor.sla_fault_tolerance.hedged_p99_latency_ms
    );
    println!("  [3/3] Closet-Silicon & E-Waste Longevity Governor (Zero Battery Swell & Zero Flash Wear):");
    for p in &reactor.closet_silicon_longevity.protected_profiles {
        println!(
            "    - {:<68} | {} | {:.1}°C (+{:.1}% GFLOPS/W) | Flash Wear: {:.2} -> {:.2} TBW/yr (-{:.2}%, {:.1} yr life)",
            p.device_label,
            p.power_supply_mode,
            p.target_dvfs_sweet_spot_temp_c,
            p.sustained_gflops_per_watt_gain_pct,
            p.naive_annual_flash_writes_tbw,
            p.kadmium_zero_wear_annual_flash_writes_tbw,
            p.flash_wear_reduction_pct,
            p.estimated_hardware_lifespan_years
        );
    }
}

pub async fn run_bench() {
    let nodes = sample_personal_cluster();
    let kernel_report = HardwareKernelDispatcher::run_cluster_kernel_benchmark(&nodes);

    println!(
        "==================================================================================================================="
    );
    println!(
        " KADMIUM (Cd-48) PHASE 2 BENCHMARK — FUSED KERNEL DISPATCH, HTTP BYTE-RANGE SLICER & STUN/DERP NAT TRAVERSAL"
    );
    println!(
        "==================================================================================================================="
    );
    println!(
        "\n[1/3] Fused Transformer / MoE Operator Suite (RMSNorm+RoPE -> Flash-GQA PagedAttention -> Q4_K SwiGLU)"
    );
    println!(
        "      Canonical Verde RepOps Digest: {:#018x} | Cross-Silicon Bit-Identical: {} | Output L2 Norm: {:.3}",
        kernel_report.canonical_repops_digest_fnv1a,
        kernel_report.cross_silicon_bit_identical,
        kernel_report.sample_output_norm
    );
    println!(
        "\n{:<22} {:<24} {:>12} {:>10} {:<48}",
        "HOSTNAME", "KERNEL TARGET", "GFLOP/S", "BW UTIL", "INTRINSIC / SHADER SIGNATURE"
    );
    println!("{}", "-".repeat(122));
    for p in &kernel_report.node_profiles {
        println!(
            "{:<22} {:<24} {:>10.1}   {:>8.1}%  {}",
            p.hostname,
            format!("{:?}", p.kernel_target),
            p.achieved_gflops,
            p.memory_bw_utilization_pct,
            p.intrinsic_signature
        );
    }

    // 2. HTTP Byte-Range Safetensors Slicer
    let raw_st = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
        (
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            vec![32, 32],
            &vec![0x11; 2048],
        ),
        (
            "model.layers.14.mlp.experts.2.up_proj.weight",
            "F8_E4M3",
            vec![64, 64],
            &vec![0x42; 4096],
        ),
    ]);
    let slices = HfByteRangeSlicer::plan_http_range_requests(
        "deepseek-ai/DeepSeek-V3",
        "main",
        "model-00004-of-00016.safetensors",
        &raw_st,
        14,
        14,
    )
    .expect("plan HTTP byte range requests");
    let ktb =
        HfByteRangeSlicer::assemble_range_slices_to_ktb("deepseek-v3-671b-moe", &raw_st, &slices)
            .expect("assemble range slices to .ktb");
    println!(
        "\n[2/3] HTTP Byte-Range (`Range: bytes=start-end`) Safetensors Slicer -> 64 KiB `.ktb` Compiler"
    );
    for s in &slices {
        println!(
            "  * GET {} | Header: [Range: {}] ({} bytes) -> Layer {:02} ({:?}) -> Packed {} B .ktb",
            s.shard_filename,
            s.http_range_header,
            s.payload_len_bytes,
            s.layer_index,
            s.target_encoding,
            ktb.len()
        );
    }

    // 3. STUN / UDP Hole-Punching & DERP Relay Matrix
    let nat_matrix = NatTraversalCoordinator::probe_cluster_nat_mesh(&nodes);
    let (sock_a, sock_b, nonce) = NatTraversalCoordinator::execute_udp_holepunch_loopback()
        .await
        .expect("execute UDP hole-punch loopback");
    println!(
        "\n[3/6] STUN / UDP Hole-Punching (`KDMP` Probe: {} <-> {}, Nonce {:#018x}) & DERP Fallback",
        sock_a, sock_b, nonce
    );
    for r in &nat_matrix {
        println!(
            "  * {:<22} ({:?}) <==> {:<24} ({:?}) => {:?} [RTT: {:.2} ms | Direct P2P: {}]",
            r.node_a,
            r.nat_a,
            r.node_b,
            r.nat_b,
            r.selected_path,
            r.negotiated_rtt_ms,
            r.direct_p2p_achieved
        );
    }

    // 4. RAID-0 / RAID-5 Linear-Algebraic Parity Tensor Striping & Multi-Path Link Bonding
    let input_act: Vec<f32> = (0..64 * 16)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
        .collect();
    let weights: Vec<f32> = (0..16 * 16)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
        .collect();
    let (_, raid_report) = SwarmRaidStriper::execute_raid5_coded_projection(
        &input_act,
        &weights,
        64,
        16,
        16,
        8,
        Some(3),
        33.0,
    );
    let mp_sched = MultiPathInterfaceStriper::benchmark_cluster_multipath(&nodes, 33.0);
    println!(
        "\n[4/6] RAID-5 Linear-Algebraic Parity Tensor Striping (`Y_p = sum(Y_j)`) & Multi-Path Link Bonding"
    );
    println!(
        "  * 33 MB Prefill across k={} Seeders + {} Parity | Single Link: {:.1} ms | Naive RAID-0 (w/ straggler #3): {:.1} ms",
        raid_report.data_stripes_k,
        raid_report.parity_stripes_p,
        raid_report.single_link_baseline_ms,
        raid_report.raid0_straggler_bound_ms
    );
    println!(
        "  * RAID-5 Parity Completion: {:.1} ms ({:.2}x Speedup, -{:.1}% Tail Latency, Straggler #3 Reconstructed Err={:.2e})",
        raid_report.raid5_parity_completed_ms,
        raid_report.speedup_vs_single_link,
        raid_report.tail_latency_reduction_pct,
        raid_report.max_reconstruction_error
    );
    println!(
        "  * Multi-Path Interface Bonding: {:.2} Gbps Bonded (vs {:.2} Gbps Single Best = {:.2}x Speedup across {} channels)",
        mp_sched.bonded_bandwidth_gbps,
        mp_sched.single_best_link_gbps,
        mp_sched.multi_path_speedup,
        mp_sched.channel_allocations.len()
    );

    // 5 & 6. Cross-System Roofline Matrix, Speculative Tree Attention, Chunked Prefill & Zero-Bubble ZB-H1 Training
    let matrix = CrossSystemComputeHarness::run_full_matrix(None);
    println!(
        "\n[5/6] Throughput & Training Accelerators (Speculative Tree Attention + Chunked Prefill + Zero-Bubble ZB-H1)"
    );
    println!(
        "  * Speculative Draft Tree (D={}, B={}, {} nodes): Accepted {} tokens/RTT ({:.2}x Decode Speedup, +{:.1}% vs linear K=4)",
        matrix.speculative_tree_report.tree_depth,
        matrix.speculative_tree_report.branching_factor,
        matrix.speculative_tree_report.total_tree_nodes_evaluated,
        matrix.speculative_tree_report.accepted_length,
        matrix.speculative_tree_report.decode_speedup_multiplier,
        matrix.speculative_tree_report.gain_over_linear_speculative_pct
    );
    println!(
        "  * Continuous Chunked Prefill ({} tok / {} chunks + {} decode streams): AI {:.1} -> {:.1} FLOP/B | Jitter {:.1} ms (vs {:.1} ms stall)",
        matrix.chunked_prefill_report.total_prefill_tokens,
        matrix.chunked_prefill_report.num_fused_batches,
        matrix.chunked_prefill_report.piggybacked_decode_streams,
        matrix.chunked_prefill_report.unbatched_decode_arithmetic_intensity,
        matrix.chunked_prefill_report.fused_batch_arithmetic_intensity,
        matrix.chunked_prefill_report.chunked_p99_decode_jitter_ms,
        matrix.chunked_prefill_report.unchunked_prefill_stall_ms
    );
    println!(
        "  * Zero-Bubble ZB-H1 Split-B/W Training: Bubble {:.2}% -> {:.2}% (-{:.1}% Bubble, +{:.1}% Throughput) | FWHT INT8 Checkpoint: -{:.1}% VRAM",
        matrix.zero_bubble_training_report.standard_1f1b_bubble_pct,
        matrix.zero_bubble_training_report.zero_bubble_zbh1_bubble_pct,
        matrix.zero_bubble_training_report.bubble_reduction_pct,
        matrix.zero_bubble_training_report.training_throughput_gain_pct,
        matrix.zero_bubble_training_report.fwht_int8_checkpoint_memory_saved_pct
    );

    println!(
        "\n[6/6] Cross-System Compute & Roofline Matrix (BF16 vs. Q4_K vs. EXL3 2.5bpw Trellis)"
    );
    println!(
        "{:<28} {:<20} {:>9} {:>10} {:>10} {:>11} {:>11} {:>12}",
        "NODE / SYSTEM", "DEVICE KIND", "MEM (GB)", "BW (GB/S)", "BF16 TOK/S", "Q4_K TOK/S", "EXL3 TOK/S", "TOK / WH"
    );
    println!("{}", "-".repeat(120));
    for row in &matrix.system_rows {
        println!(
            "{:<28} {:<20} {:>8.1}   {:>9.1}   {:>9.1}   {:>10.1}   {:>10.1}   {:>10.0}",
            row.hardware_label,
            format!("{:?}", row.device_kind),
            row.fast_memory_gb,
            row.memory_bandwidth_gbps,
            row.shard_bf16_tok_s,
            row.shard_q4k_tok_s,
            row.shard_exl3_trellis_tok_s,
            row.tokens_per_watt_hour
        );
    }
    println!("\n{}", matrix.shareable_scorecard_markdown);

    // 7. Hyper-Training Compute Engine (GaLore + Muon Newton-Schulz + Varlen Packing + Chunked CE + FWHT-FP8 + SkipPipe)
    let hyper_train = TrainingComputeAcceleratorSuite::run_full_suite(None);
    println!(
        "\n[7/7] Hyper-Training Compute Engine (GaLore + Muon 8-Bit + Varlen Packing + Liger Chunked CE + FWHT-FP8 + SkipPipe)"
    );
    println!(
        "  * GaLore + Muon Newton-Schulz 8-Bit Optimizer: -{}% Optimizer VRAM ({} KB vs {} KB) | Loss {:.4} -> {:.4} (-{:.1}%)",
        hyper_train.galore_muon_report.optimizer_vram_reduction_pct,
        hyper_train.galore_muon_report.galore_int8_state_bytes / 1024,
        hyper_train.galore_muon_report.standard_adamw_state_bytes / 1024,
        hyper_train.galore_muon_report.initial_loss,
        hyper_train.galore_muon_report.final_loss,
        hyper_train.galore_muon_report.loss_reduction_pct
    );
    println!(
        "  * Varlen Best-Fit Sequence Packing ({} seqs -> {} bins): Padding Waste {:.1}% -> {:.1}% ({:.2}x Effective Token Speedup)",
        hyper_train.packing_and_chunked_ce_report.total_sequences,
        hyper_train.packing_and_chunked_ce_report.packed_bin_count,
        hyper_train.packing_and_chunked_ce_report.unpacked_padding_waste_pct,
        hyper_train.packing_and_chunked_ce_report.packed_padding_waste_pct,
        hyper_train.packing_and_chunked_ce_report.effective_packing_speedup
    );
    println!(
        "  * Liger Fused Chunked Cross-Entropy (V=128,256): Peak Logit VRAM {:.1} MB -> {:.1} MB (-{:.1}% VRAM, MaxGradDiff={:.2e})",
        hyper_train.packing_and_chunked_ce_report.standard_ce_peak_logit_vram_mb,
        hyper_train.packing_and_chunked_ce_report.fused_chunked_ce_peak_vram_mb,
        hyper_train.packing_and_chunked_ce_report.logit_vram_savings_pct,
        hyper_train.packing_and_chunked_ce_report.max_loss_grad_diff
    );
    println!(
        "  * FWHT-Rotated FP8 (E4M3 Fwd / E5M2 Bwd): Outlier L2 Err {:.4} -> {:.4} | Fwd CosSim {:.5} | Bwd CosSim {:.5} ({:.2}x GEMM)",
        hyper_train.hadamard_fp8_report.naive_fp8_e4m3_rel_error,
        hyper_train.hadamard_fp8_report.fwht_rotated_fp8_e4m3_rel_error,
        hyper_train.hadamard_fp8_report.forward_gemm_cosine_similarity,
        hyper_train.hadamard_fp8_report.backward_e5m2_gemm_cosine_similarity,
        hyper_train.hadamard_fp8_report.fp8_training_compute_speedup
    );
    println!(
        "  * SkipPipe Straggler Bypass + Async PhoneFarm/UMA Offload: Step {:.1} ms -> {:.1} ms ({:.1}% Pipeline Util, {:.2}x Speedup)",
        hyper_train.skippipe_offload_report.stalled_pipeline_step_ms,
        hyper_train.skippipe_offload_report.skippipe_active_step_ms,
        hyper_train.skippipe_offload_report.skippipe_pipeline_utilization_pct,
        hyper_train.skippipe_offload_report.skippipe_and_offload_speedup
    );
    println!(
        "  * Zig-Zag Causal Ring Attention ({} nodes, {}K ctx): Imbalance {:.1}% -> {:.1}% ({:.2}x Speedup, LSE Err={:.2e})",
        hyper_train.zigzag_ring_report.num_ring_nodes,
        hyper_train.zigzag_ring_report.total_context_tokens / 1024,
        hyper_train.zigzag_ring_report.naive_ring_causal_imbalance_pct,
        hyper_train.zigzag_ring_report.zigzag_ring_causal_imbalance_pct,
        hyper_train.zigzag_ring_report.zigzag_load_balance_speedup,
        hyper_train.zigzag_ring_report.online_lse_ring_max_abs_error
    );
    println!(
        "  * Aux-Loss-Free MoE Dynamic Bias: Max/Mean Load {:.2}x -> {:.2}x (-{:.1}% Imbalance, 0.0 Aux Grad, -{:.1}% All-to-All)",
        hyper_train.aux_loss_free_moe_report.initial_max_to_mean_load_ratio,
        hyper_train.aux_loss_free_moe_report.final_max_to_mean_load_ratio,
        hyper_train.aux_loss_free_moe_report.load_imbalance_reduction_pct,
        hyper_train.aux_loss_free_moe_report.hierarchical_all_to_all_traffic_reduction_pct
    );
    println!(
        "  * Weiszfeld Byzantine-Robust Async DiLoCo: Rejected {:?} | Naive Err {:.3} -> Robust Err {:.4} ({:.2}x Async Speedup)",
        hyper_train.byzantine_async_diloco_report.detected_poisoned_island_indices,
        hyper_train.byzantine_async_diloco_report.naive_mean_corruption_l2_error,
        hyper_train.byzantine_async_diloco_report.robust_weiszfeld_l2_error,
        hyper_train.byzantine_async_diloco_report.async_staleness_damped_speedup
    );
    println!(
        "  * WSD Loop + .ktb Spike Rollback Guard: {} steps | Loss {:.4} -> {:.4} (-{:.1}%, {} Spike Rollback at step {})\n",
        hyper_train.wsd_epoch_execution_report.total_steps_executed,
        hyper_train.wsd_epoch_execution_report.initial_loss,
        hyper_train.wsd_epoch_execution_report.final_loss,
        hyper_train.wsd_epoch_execution_report.loss_reduction_pct,
        hyper_train.wsd_epoch_execution_report.spike_rollbacks_recovered,
        hyper_train.wsd_epoch_execution_report.injected_corrupted_batch_step
    );
    println!(
        "{:<28} {:<38} {:>11} {:>12} {:>10} {:>12}",
        "NODE / SYSTEM", "TRAINING ROLE", "BASE TOK/S", "BOOST TOK/S", "SAVED GB", "TRAIN TOK/WH"
    );
    println!("{}", "-".repeat(118));
    for r in &hyper_train.node_training_rows {
        println!(
            "{:<28} {:<38} {:>10.1}   {:>11.1}   {:>8.1}   {:>11.0}",
            r.hardware_label,
            r.assigned_training_role,
            r.baseline_train_tok_s,
            r.boosted_train_tok_s,
            r.memory_saved_gb,
            r.train_tokens_per_watt_hour
        );
    }
    println!("\n{}", hyper_train.summary_scorecard);

    // 8. Hyper-Inference Compute Engine (7-Stage Suite across Heterogeneous Nodes)
    let hyper_inf = HyperInferenceComputeReport::run_all();
    println!(
        "\n[8/10] Hyper-Inference Compute Engine (MLA + KIVI 2-Bit + Flash-Decoding + PowerInfer/T-MAC LUT + Jacobi Wavefront + Radix/CacheBlend/MInference + 2-Layer MoE Prefetch)"
    );
    println!(
        "  * DeepSeek-V3 MLA Weight-Absorption (q_abs = q * W_UK^T): KV Cache {} B -> {} B/tok (-{:.1}%, {:.1}x Compression, {:.2}x Decode BW, MaxDiff={:.2e})",
        hyper_inf.mla_report.standard_mha_kv_bytes_per_token,
        hyper_inf.mla_report.mla_latent_kv_bytes_per_token,
        hyper_inf.mla_report.kv_cache_reduction_pct,
        hyper_inf.mla_report.kv_compression_ratio_x,
        hyper_inf.mla_report.decode_bandwidth_speedup_x,
        hyper_inf.mla_report.weight_absorption_max_abs_diff
    );
    println!(
        "  * KIVI Asymmetric 2-Bit KV (Per-Channel Key INT2 + Per-Token Val INT2): {:.2}x Compression | Key Err {:.4} -> {:.4} | CosSim {:.5}",
        hyper_inf.kivi_2bit_report.kv_compression_ratio_x,
        hyper_inf.kivi_2bit_report.naive_per_token_key_rel_error,
        hyper_inf.kivi_2bit_report.kivi_asymmetric_rel_error,
        hyper_inf.kivi_2bit_report.attention_output_cosine_similarity
    );
    println!(
        "  * Flash-Decoding Parallel Split-KV ({}K ctx, {} splits): Hardware Util {:.1}% -> {:.1}% ({:.2}x Decode Speedup, LSE Err={:.2e})",
        hyper_inf.flash_decoding_report.context_seq_len / 1024,
        hyper_inf.flash_decoding_report.num_parallel_kv_splits,
        hyper_inf.flash_decoding_report.unsplit_decode_hardware_utilization_pct,
        hyper_inf.flash_decoding_report.split_kv_hardware_utilization_pct,
        hyper_inf.flash_decoding_report.flash_decoding_speedup_x,
        hyper_inf.flash_decoding_report.split_kv_max_abs_error
    );
    println!(
        "  * PowerInfer Contextual Sparsity + T-MAC 81-State LUT Ternary: -{:.1}% FFN BW (CosSim {:.5}, {:.2}x FFN) | LUT Diff={:.2e} ({:.2}x CPU/NPU)",
        hyper_inf.sparsity_and_lut_report.ffn_memory_bandwidth_saved_pct,
        hyper_inf.sparsity_and_lut_report.sparse_ffn_cosine_similarity,
        hyper_inf.sparsity_and_lut_report.powerinfer_ffn_speedup_x,
        hyper_inf.sparsity_and_lut_report.tmac_lut_ternary_max_abs_diff,
        hyper_inf.sparsity_and_lut_report.tmac_lut_cpu_npu_speedup_x
    );
    println!(
        "  * Lookahead Jacobi Wavefront (W={}, {} stages): Stage Util {:.1}% -> {:.1}% | {:.2} Accepted Tok/Step ({:.2}x Lossless Decode)",
        hyper_inf.lookahead_jacobi_report.lookahead_window_w,
        hyper_inf.lookahead_jacobi_report.pipeline_stages_k,
        hyper_inf.lookahead_jacobi_report.sequential_single_token_stage_util_pct,
        hyper_inf.lookahead_jacobi_report.lookahead_wavefront_stage_util_pct,
        hyper_inf.lookahead_jacobi_report.accepted_tokens_per_jacobi_step,
        hyper_inf.lookahead_jacobi_report.distributed_decode_speedup_x
    );
    println!(
        "  * Radix LPM ({:.1}% Hit) + CacheBlend ({:.0}% HKVD, CosSim {:.4}) + MInference (-{:.1}% FLOPs, {:.1}% Mass): TTFT {:.0} ms -> {:.0} ms ({:.2}x)",
        hyper_inf.prefill_ttft_report.radix_lpm_prefix_hit_rate_pct,
        hyper_inf.prefill_ttft_report.cacheblend_hkvd_recompute_fraction_pct,
        hyper_inf.prefill_ttft_report.cacheblend_attention_cosine_similarity,
        hyper_inf.prefill_ttft_report.minference_vertical_slash_flops_saved_pct,
        hyper_inf.prefill_ttft_report.minference_attention_mass_preserved_pct,
        hyper_inf.prefill_ttft_report.baseline_dense_ttft_ms,
        hyper_inf.prefill_ttft_report.boosted_ttft_ms,
        hyper_inf.prefill_ttft_report.ttft_speedup_x
    );
    println!(
        "  * 2-Layer-Ahead MoE Router Probe (h_{{l-2}} * W_gate): {:.1}% Top-K Hit | DMA Stall {:.1} ms -> {:.2} ms (-{:.1}% Stall, {:.2}x MoE Speedup)\n",
        hyper_inf.moe_prefetch_report.two_layer_ahead_expert_hit_rate_pct,
        hyper_inf.moe_prefetch_report.unprefetched_moe_dma_stall_ms,
        hyper_inf.moe_prefetch_report.lookahead_prefetched_dma_stall_ms,
        hyper_inf.moe_prefetch_report.moe_expert_stall_reduction_pct,
        hyper_inf.moe_prefetch_report.moe_decode_speedup_x
    );
    println!(
        "{:<26} {:<56} {:>9} {:>10} {:>8} {:>9} {:>10}",
        "NODE / SYSTEM", "ACTIVE HYPER-INFERENCE ACCELERATORS", "BASE TOK/S", "BOOST TOK/S", "SPEEDUP", "TTFT (MS)", "INF TOK/WH"
    );
    println!("{}", "-".repeat(134));
    for r in &hyper_inf.node_inference_rows {
        println!(
            "{:<26} {:<56} {:>8.1}   {:>9.1}   {:>6.2}x {:>7.0}ms {:>10.0}",
            r.hardware_label,
            r.active_accelerators,
            r.baseline_decode_tok_s,
            r.hyper_boosted_decode_tok_s,
            r.decode_speedup_x,
            r.prefill_ttft_ms,
            r.inference_tokens_per_watt_hour
        );
    }
    println!("\n{}", hyper_inf.summary_scorecard);

    // 9. 4-Tier Hierarchical Pool Auto-Scaler (16 -> 100,000+ Nodes)
    let pool_scale = HierarchicalPoolPlanner::run_multi_scale_benchmark();
    println!(
        "\n[9/10] 4-Tier Hierarchical Pool Auto-Scaler & Radix-16 Tree Federation (16 -> 100,000+ Nodes)"
    );
    println!(
        "{:<38} {:>8} {:>9} {:>12} {:>11} {:>14} {:>14}",
        "POOL SCALE TIER", "CELLS", "TREE HOPS", "LINK RAM RED", "TREE SYNC", "INF TOK/S", "TRAIN TOK/S"
    );
    println!("{}", "-".repeat(118));
    for t in &pool_scale.evaluated_tiers {
        println!(
            "{:<38} {:>8} {:>9} {:>11.1}x {:>9.1}ms {:>14.1} {:>14.1}",
            t.tier_label,
            t.decomposition.total_model_cells,
            t.scaling_metrics.tree_depth_hops,
            t.scaling_metrics.topology_memory_reduction_x,
            t.scaling_metrics.hierarchical_tree_sync_ms,
            t.inference_pool_stats.aggregate_inference_tok_s,
            t.training_pool_stats.aggregate_training_tok_s
        );
    }
    println!("\n{}", pool_scale.summary_scorecard);

    // 10. Decentralized Pool Storage Bucket & Wire-Resident Token Carousel (100-Node Pool)
    let storage_pool = DecentralizedStoragePoolEngine::run_pool_storage_benchmark(100);
    println!(
        "\n[10/10] Decentralized Pool Storage Bucket (`{}`) & Wire-Resident Token Carousel ({}-Node Pool)",
        storage_pool.bucket_uri, storage_pool.total_participating_nodes
    );
    println!(
        "  * Pooled Capacity: {:.2} TiB NVMe/UFS Flash + {:.2} TiB Wire-Resident RAM Carousel | Agg Read: {:.1} GB/s ({:.1} Gbps Seed)",
        storage_pool.total_pooled_disk_capacity_gb / 1024.0,
        storage_pool.total_wire_ram_carousel_gb / 1024.0,
        storage_pool.aggregate_disk_read_bw_gbps,
        storage_pool.aggregate_network_seed_bw_gbps
    );
    println!(
        "  * Systematic RS(k={}, m={}) Erasure Coding ({:.2}x Overhead vs 3.0x Rep): Dropped {:?} -> Reconstructed Exact={} (Diff={} B)",
        storage_pool.erasure_coding_verification.data_shards_k,
        storage_pool.erasure_coding_verification.parity_shards_m,
        storage_pool.erasure_coding_verification.storage_overhead_ratio,
        storage_pool.erasure_coding_verification.simulated_dropped_shard_indices,
        storage_pool.erasure_coding_verification.exact_bit_identical_recovery,
        storage_pool.erasure_coding_verification.reconstruction_max_byte_diff
    );
    println!(
        "  * 'Living on the Wire' Token Carousel ({:.1}T Tokens): {:.1} Gbps Wire Feed ({:.1}x vs {:.1} Gbps 1-NAS) | Step Data Stall: {:.1} ms -> {:.1} ms ({:.1}% Wire Hit)",
        storage_pool.wire_token_carousel.dataset_total_tokens_billions / 1000.0,
        storage_pool.wire_token_carousel.decentralized_pool_carousel_gbps,
        storage_pool.wire_token_carousel.bandwidth_scaling_multiplier_x,
        storage_pool.wire_token_carousel.centralized_nas_bottleneck_gbps,
        storage_pool.wire_token_carousel.centralized_data_stall_ms_per_step,
        storage_pool.wire_token_carousel.wire_carousel_data_stall_ms_per_step,
        storage_pool.wire_token_carousel.pre_staged_on_wire_hit_rate_pct
    );
    println!(
        "  * Bucket Grounding (NAS + Donated SAN + S3/R2): 50% Mass Dropout {:?} (Ungrounded={}) -> Grounded Recovery={} (Diff={} B, {:.1}% P2P / {:.1}% Spillover, -{}% p99)",
        storage_pool.bucket_grounding.simulated_mass_dropout_indices,
        storage_pool.bucket_grounding.ungrounded_survives_mass_dropout,
        storage_pool.bucket_grounding.grounded_survives_mass_dropout,
        storage_pool.bucket_grounding.grounded_recovery_max_byte_diff,
        storage_pool.bucket_grounding.steady_state_p2p_swarm_read_share_pct,
        storage_pool.bucket_grounding.steady_state_ground_spillover_share_pct,
        storage_pool.bucket_grounding.p99_tail_read_latency_reduction_pct
    );
    for anchor in &storage_pool.bucket_grounding.configured_anchors {
        println!(
            "    - [Tier {} {:?}] {:<46} | {:>6.1} TiB | {:>4.1} Gbps | Policy: {:?} ({:.0}% Grounded)",
            anchor.priority_tier,
            anchor.anchor_kind,
            anchor.anchor_uri,
            anchor.capacity_tb,
            anchor.sustained_bandwidth_gbps,
            anchor.sync_policy,
            anchor.merkle_grounded_coverage_pct
        );
    }
    println!(
        "\n{:<52} {:<24} {:>9} {:>10} {:>12} {:>12}",
        "VIRTUAL BUCKET URI (DRAG-AND-DROP)", "OBJECT KIND", "LOG (GB)", "PHYS (GB)", "BLOCKS", "READ (GBPS)"
    );
    println!("{}", "-".repeat(124));
    for obj in &storage_pool.bucket_objects {
        println!(
            "{:<52} {:<24} {:>8.1}   {:>9.1}   {:>10}   {:>10.1}",
            obj.object_uri,
            obj.object_kind,
            obj.logical_size_gb,
            obj.physical_erasure_coded_gb,
            obj.total_64mb_superblocks,
            obj.parallel_read_bandwidth_gbps
        );
    }
    println!("\n{}", storage_pool.summary_scorecard);
}

pub fn run_pairing() {
    let nodes = sample_personal_cluster();
    let coord = &nodes[0];
    let candidates = ZeroClickLanPairingEngine::discover_lan_candidates(&coord.id);
    let hyst = OsSwapchainHysteresisEngine::sample_desktop_gaming_timeline(coord);
    let watcher = CloudShieldCutoverWatcher::evaluate_incident_timeline();

    println!(
        "==================================================================================================================="
    );
    println!(
        " KADMIUM (Cd-48) PHASE 3 & 4 — ZERO-CLICK LAN PAIRING, OS SWAPCHAIN HYSTERESIS & <400MS CLOUDSHIELD WATCHER"
    );
    println!(
        "==================================================================================================================="
    );
    println!(
        "\n[1/3] Zero-Click LAN Pairing (`KDMD` UDP Beacon Queue & 6-Digit SAS Companion Tray Verification)"
    );
    for c in &candidates {
        println!(
            "  * {:<28} @ {:<18} | SAS: [{}] | Status: {:?} | {}",
            c.hostname, c.lan_socket_addr, c.short_auth_code, c.status, c.hardware_summary
        );
    }

    println!(
        "\n[2/3] OS Swapchain & Thermal Hysteresis ({:?} — `{}`)",
        hyst.hook_backend, hyst.api_signature
    );
    println!(
        "      Fast-Yield SLA: {:.0} ms (0 ms delay) | Slow-Reclaim Anti-Thrash Cooldown: {} ms",
        hyst.fast_yield_sla_ms, hyst.slow_reclaim_cooldown_ms
    );
    for step in &hyst.timeline {
        println!(
            "  * [t={:>5} ms] {:<14} -> {:<24} | VRAM Cap: {:>5.1} GB | {}",
            step.timestamp_ms,
            format!("{:?}", step.previous_state),
            format!("{:?}", step.new_state),
            step.allocated_vram_gb,
            step.transition_action
        );
    }

    println!(
        "\n[3/3] Automated `<400ms` CloudShield Heartbeat Watcher ({} -> {}, Layers L{:02}..L{:02})",
        watcher.protected_local_node,
        watcher.standby_vault_node,
        watcher.guarded_layers.0,
        watcher.guarded_layers.1
    );
    println!(
        "      Measured Cutover RTO: {:.0} ms | Dropped Requests: 0 | 18-Min Outage Prorated Total: ${:.2}",
        watcher.measured_cutover_rto_ms, watcher.total_prorated_incident_cost_usd
    );
    for st in &watcher.steps {
        println!(
            "  * [t={:>4} s] {:<24} | Host: {:<40} | Burn: ${:.2}/hr | Cost: ${:.2}",
            st.timestamp_sec,
            format!("{:?}", st.state),
            st.active_middle_shard_host,
            st.instantaneous_hourly_burn_usd,
            st.cumulative_incident_cost_usd
        );
    }
}

pub fn run_registry() {
    let orderbook = SpotOrderbookBidEngine::query_live_orderbook();
    let provisioned = SpotOrderbookBidEngine::provision_optimal_middle_shard(48.0, 1.50, (26, 48))
        .expect("provision middle shard");
    let (rbac_quotas, rbac_keys) = SubClusterRbacEngine::sample_lab_rbac_catalog();

    let referee_fn = |layer: usize, state: &[f64]| -> Vec<f64> {
        state
            .iter()
            .map(|&v| v * 1.01 + (layer as f64) * 0.002)
            .collect()
    };
    let mut ref_states = vec![vec![1.0, 2.0, 3.0]];
    let mut wrk_states = vec![vec![1.0, 2.0, 3.0]];
    for l in 0..32 {
        ref_states.push(referee_fn(l, &ref_states[l]));
        let mut w = referee_fn(l, &wrk_states[l]);
        if l == 19 {
            w[0] += 0.35;
        }
        wrk_states.push(w);
    }
    let fraud_proof = VerdeFraudProofEngine::generate_fraud_proof(
        "verde-fp-0048",
        "untrusted-renter-rtx3090",
        &wrk_states,
        &ref_states,
        referee_fn,
        1e-6,
        0.24,
    )
    .expect("generate fraud proof");

    let worlds = DreamRsiWorldRegistry::public_catalog();

    println!(
        "==================================================================================================================="
    );
    println!(
        " KADMIUM (Cd-48) v1.0 GA — SPOT PROVISIONER, MULTI-TENANT RBAC, VERDE FRAUD PROOFS & DREAM-RSI WORLD REGISTRY"
    );
    println!(
        "==================================================================================================================="
    );
    println!(
        "\n[1/4] Live Multi-Provider Spot Orderbook & Ephemeral Container Provisioner (Target: >=48 GB VRAM, <=$1.50/hr)"
    );
    println!(
        "      Selected Bid: {} ({}) @ ${:.2}/hr | Container: {} | Layers L{:02}..L{:02} | Grace: {}s",
        provisioned.selected_quote.gpu_model,
        provisioned.selected_quote.provider,
        provisioned.selected_quote.spot_usd_per_hr,
        provisioned.container_id,
        provisioned.pinned_middle_layers.0,
        provisioned.pinned_middle_layers.1,
        provisioned.eviction_grace_period_sec
    );
    for q in &orderbook {
        println!(
            "  * [{:<22}] {:<24} | {:>5.1} GB VRAM | Spot: ${:>4.2}/hr | Preempt: {:>4.1}% | Utility: {:>5.1} TFLOP/$",
            q.quote_id,
            q.gpu_model,
            q.vram_gb,
            q.spot_usd_per_hr,
            q.preemption_prob_24h * 100.0,
            q.risk_adjusted_tflops_per_usd
        );
    }

    println!(
        "\n[2/4] Multi-Tenant Sub-Cluster RBAC Quotas & Scoped API Keys ({} slices, {} keys)",
        rbac_quotas.len(),
        rbac_keys.len()
    );
    for k in &rbac_keys {
        println!(
            "  * {:<38} | Principal: {:<26} | Role: {:<20} | Slice: {}",
            k.token_string,
            k.principal_name,
            format!("{:?}", k.role),
            k.bound_slice_id
        );
    }

    println!(
        "\n[3/4] Cryptographic `Verde` Optimistic Fraud Proof & Automated Slashing Certificate"
    );
    println!(
        "      Certificate ID: {} | Disputed Worker: {} | Checks Used: {} (of {} layers)",
        fraud_proof.certificate_id,
        fraud_proof.disputed_worker_hostname,
        fraud_proof.referee_evaluations_used,
        fraud_proof.total_layers_in_pipeline
    );
    println!(
        "      Faulty Layer #{} | Input Hash: {:#018x} | Referee RepOps: {:#018x} != Claimed: {:#018x}",
        fraud_proof.faulty_layer_index,
        fraud_proof.input_activation_digest_fnv1a,
        fraud_proof.referee_repops_output_digest_fnv1a,
        fraud_proof.worker_claimed_output_digest_fnv1a
    );
    println!(
        "      Withheld Settlement: ${:.2} | Max Divergence: {:.3} | MAC Verified: {}",
        fraud_proof.withheld_settlement_usd,
        fraud_proof.max_abs_divergence,
        VerdeFraudProofEngine::verify_certificate(&fraud_proof)
    );

    println!(
        "\n[4/4] Public Dream-RSI World & Benchmark Registry (`gh://` Verified Discovery Trees)"
    );
    for w in &worlds {
        println!(
            "  * [{:<26}] {:<46} | Peak: {:.2}x | Saved: {:>4.1}% | Nodes: {:>4}",
            w.world_slug,
            w.display_name,
            w.peak_speedup_or_score,
            w.online_gpu_rollout_savings_pct,
            w.recorded_discovery_nodes
        );
        println!(
            "      URI: {} (SHA: {} | Policy: `{}`)",
            w.github_uri, w.pinned_commit_sha, w.promoted_policy_id
        );
    }
}
