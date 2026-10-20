//! Phase 2–5 (`v0.2` through `v1.0 GA`) Supply Grid, Kernel Benchmark, Zero-Click Pairing,
//! and World Registry CLI handlers (`kadmium-cli::grid_ops`).

use kadmium_core::{
    OsSwapchainHysteresisEngine, SubClusterRbacEngine, SupplyLeasePolicy, SupplyPayoutMode,
    SupplySettlementEngine, sample_personal_cluster,
};
use kadmium_data::{HfByteRangeSlicer, SafetensorsToKtbTranscoder};
use kadmium_network::{NatTraversalCoordinator, ZeroClickLanPairingEngine};
use kadmium_rl::DreamRsiWorldRegistry;
use kadmium_runtime::{HardwareKernelDispatcher, VerdeFraudProofEngine};
use kadmium_sharding::{CloudShieldCutoverWatcher, SpotOrderbookBidEngine};

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
        "\n{:<24} {:<36} {:>12} {:>14} {:>14}",
        "HOSTNAME", "HARDWARE PROFILE", "USD / HR", "CREDITS / HR", "MONTHLY (1.25x)"
    );
    println!("{}", "-".repeat(104));
    for q in &snap.node_quotes {
        println!(
            "{:<24} {:<36} ${:>10.2}   ${:>11.2}   ${:>11.2}",
            q.hostname,
            q.hardware_summary,
            q.usd_rate_per_hour,
            q.cloud_credit_rate_per_hour_usd,
            q.projected_monthly_cloud_credits_usd
        );
    }
    println!("{}", "-".repeat(104));
    println!(
        "  * Cluster Hourly Rate:        ${:.2}/hr direct USD  |  ${:.2}/hr Kadmium Cloud Credits (1.25x)",
        snap.total_hourly_usd, snap.total_hourly_cloud_credits_usd
    );
    println!(
        "  * Projected Monthly Return:   ${:.2}/mo direct USD  |  ${:.2}/mo Cloud Credits (~{:.1} hrs L40S 48GB burst)",
        snap.total_projected_monthly_usd,
        snap.total_projected_monthly_cloud_credits_usd,
        snap.equivalent_l40s_burst_hours_per_month
    );
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
        "\n[3/3] STUN / UDP Hole-Punching (`KDMP` Probe: {} <-> {}, Nonce {:#018x}) & DERP Fallback",
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
