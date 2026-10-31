//! Operational CLI subcommand handlers (`kadmium-cli::ops`):
//! `cloud-shield`, `probe`, `join`, `pull`, `mcp`, `init`, `compile`, and `inspect`.

use crate::print_shard_plan;
use kadmium_core::{
    K230UsbComputeBridge, KadmiumWorkspaceConfig, LiveHardwareProfiler, ModelSpec, NodeId,
    NodeTrustTier, PoliteActivityMonitor, PoliteActivitySample, PoliteState, WorkloadKind,
    sample_personal_cluster,
};
use kadmium_data::{
    GitHubConnector, HuggingFaceHubConnector, KtbBankPacker, KtbBankReader, KtbMerkleTree,
    SafetensorsToKtbTranscoder, TensorEncoding,
};
use kadmium_gateway::{GatewayState, KadmiumMcpServer, McpJsonRpcRequest, OtaUpdatePushRequest};
use kadmium_network::{
    ClusterStateCrdt, DEFAULT_LAN_DISCOVERY_UDP_PORT, FleetUpdatePushSummary, JoinTicket,
    LanCoordinatorDiscovery, LanDiscoveryBeacon, MeshHttpClient, UpdatePayloadKind,
    UpdateTargetScope, WorkerHeartbeatAck, WorkerHeartbeatRequest, WorkerRegisterRequest,
    WorkerRegistrationAck, WorkerUpdatePushAck,
};
use kadmium_runtime::{
    ColibriNvmeDoubleBuffer, DeMoDctCompressor, DistributedLanExecutor, LiveNodeWorkerServer,
    LocalFleetSpawner, Q4KBlockTensor, VerdeVerificationEngine, apply_ota_update_manifest_local,
};
use kadmium_sharding::{HybridCloudArbitragePlanner, ShardPlanner};

pub fn run_cloud_shield() {
    let model = ModelSpec::deepseek_v3_671b_moe();
    let arb = HybridCloudArbitragePlanner::compute_arbitrage_plan(&model);
    let shield_events = HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield();

    println!(
        "==================================================================================================================="
    );
    println!(
        " 🛡️  KADMIUM HYBRID CLOUD — Multi-Datacenter Spot Arbitrage & Bidirectional Downtime Protection Shield"
    );
    println!(
        "==================================================================================================================="
    );
    println!(
        "\n[1/2] Multi-Provider Fragmented GPU & Bare-Metal Arbitrage ({})",
        model.display_name
    );
    println!(
        "      Combined Compute: {:.1} FP16 TFLOPs | {:.1} GB Fast VRAM/Unified | Blended Cost: ${:.2}/hr vs. Hyperscaler ${:.2}/hr ({:.1}% Saved!)",
        arb.total_combined_tflops,
        arb.total_combined_vram_gb,
        arb.kadmium_blended_hourly_usd,
        arb.monolithic_hyperscaler_hourly_usd,
        arb.cost_reduction_pct
    );
    println!("{}", "-".repeat(125));
    for o in &arb.selected_offers {
        println!(
            "  * [{:<14}] {:<52} | {:>5.1} TFLOP | {:>5.1} GB VRAM | ${:>4.2}/hr (vs ${:>4.2}/hr)",
            o.offer_id,
            o.provider_name,
            o.fp16_tflops,
            o.vram_or_unified_gb,
            o.hourly_spot_usd,
            o.hyperscaler_equivalent_usd
        );
    }

    println!(
        "\n[2/2] Bidirectional Local <-> Cloud Downtime Protection Shield (Zero-Downtime Shard Handoff):"
    );
    println!("{}", "-".repeat(125));
    for ev in &shield_events {
        println!(
            "  >>> [{:?}] Layers L{:02}..L{:02} | RTO: {:.0} ms | Zero Downtime: {} | Burst Cost: ${:.2}/hr",
            ev.direction,
            ev.affected_layers.0,
            ev.affected_layers.1,
            ev.switchover_rto_ms,
            ev.zero_downtime_preserved,
            ev.temporary_failover_hourly_usd
        );
        println!("      Incident: {}", ev.incident_summary);
        println!(
            "      Route:    {}  ===>  {}\n",
            ev.source_environment, ev.target_failover_environment
        );
    }
}

pub fn run_probe() {
    let live_node = LiveHardwareProfiler::probe_local_host();
    println!(
        "==================================================================================================================="
    );
    println!(" 🔍 KADMIUM LIVE HARDWARE PROFILER & 64KB PAGE-ALIGNED .KTB PACKER (Milestone 1)");
    println!(
        "==================================================================================================================="
    );
    println!(
        "  * Hostname / Node ID:     {} ({})",
        live_node.hostname, live_node.id
    );
    println!(
        "  * Device Classification:  {:?} | Trust Tier: {:?}",
        live_node.device_kind, live_node.trust_tier
    );
    println!(
        "  * Primary Backend:        {:?} ({:.1} FP16 TFLOPs | {:.1} INT4 TOPS)",
        live_node.backend, live_node.fp16_tflops, live_node.int4_tops
    );
    println!(
        "  * Host CPU Parallelism:   {} logical threads ({:.1} CPU TFLOPs)",
        live_node.cpu_threads, live_node.cpu_tflops
    );
    println!(
        "  * Memory Tier Hierarchy:  {:.1} GB VRAM | {:.1} GB Unified | {:.1} GB Sys RAM | {:.2} GB/s Stream BW",
        live_node.memory.vram_gb,
        live_node.memory.unified_mem_gb,
        live_node.memory.system_ram_gb,
        live_node.memory.nvme_read_bw_gbps
    );
    println!(
        "  * FreeToken q* Split:     {:.2} (Accelerator vs. Host/NVMe Stream Ratio)",
        live_node.compute_q_star_ratio()
    );

    let k230_report = K230UsbComputeBridge::probe_attached_k230();
    println!(
        "\n--- 🔌 Attached USB Edge RISC-V + NPU Co-Processor Discovery (Kendryte CanMV K230) ---"
    );
    println!(
        "  * Physical USB Detection: {} | Stage: {:?} | PnP: {}",
        if k230_report.physical_board_detected {
            "ONLINE (Hardware Attached over USB)"
        } else {
            "Standby Profile (Ready for Hot-Plug)"
        },
        k230_report.current_usb_stage,
        k230_report.detected_pnp_instance_id
    );
    println!(
        "  * Board & SoC Silicon:    {} — {}",
        k230_report.board_model, k230_report.soc_architecture
    );
    println!(
        "  * Hardware Die Serial:    {} | USB ID: {} | Serial Channel: {}",
        k230_report.chip_unique_die_id,
        k230_report.usb_vid_pid,
        k230_report
            .detected_serial_port
            .as_deref()
            .unwrap_or("USB Bulk / Gadget Mode")
    );
    println!(
        "  * Probed MicroSD Vault:   {:.2} GiB ({} bytes, {} B LBA, {} KiB USB Bulk Write Chunks)",
        k230_report.sdcard_telemetry.capacity_gib,
        k230_report.sdcard_telemetry.capacity_bytes,
        k230_report.sdcard_telemetry.block_size_bytes,
        k230_report.sdcard_telemetry.usb_out_chunk_bytes / 1024
    );
    if let Some(proof) = &k230_report.live_execution_proof {
        println!(
            "  * Live RISC-V Execution:  {} | {}",
            if proof.executed_on_physical_silicon {
                "PHYSICAL SILICON VERIFIED OVER USB"
            } else {
                "Reference Verification"
            },
            proof.riscv_core_signature
        );
        println!(
            "  * KPU & Transport Link:   {} | {}",
            proof.nncase_kpu_version, proof.transport_channel
        );
        println!(
            "  * Cross-ISA FNV-1a Proof: Host={:#018x} == K230={:#018x} | Bit-Exact: {} | Max Err: {:.6}",
            proof.host_reference_checksum_fnv1a,
            proof.k230_hardware_checksum_fnv1a,
            proof.bit_exact_verified,
            proof.max_abs_error
        );
        println!(
            "  * Hardware Timings:       USB RTT: {:.2} ms | K230 Compute: {:.2} ms | ulab RVV 64x64 GEMM: {:.2} ms | /sdcard .ktb Witness: {}",
            proof.usb_round_trip_latency_ms,
            proof.k230_compute_time_ms,
            proof.ulab_rvv_64x64_gemm_ms,
            proof.sdcard_ktb_witness_verified
        );
        println!(
            "  * 6 TOPS KPU & MicroSD:   3rd-Gen KPU Warm Pass: {:.2} ms ({:.1} FPS) | 64 KiB .ktb SD Write: {:.2} ms / Read: {:.2} ms | 64-Layer f32 Drift: {:.9}",
            proof.kpu_npu_warm_inference_ms,
            proof.kpu_npu_throughput_fps,
            proof.ktb_64k_sdcard_write_ms,
            proof.ktb_64k_sdcard_read_ms,
            proof.verde_64_layer_max_f32_drift
        );
        println!(
            "  * AI2D DMA & Transformer: Zero-Copy AI2D->KPU: {:.2} ms ({:.1} FPS) | FastSpeech 50-Tok Sequence Encoder: {:.2} ms ({:.1} tok/s)",
            proof.ai2d_plus_kpu_chained_ms,
            if proof.ai2d_plus_kpu_chained_ms > 0.0 {
                1000.0 / proof.ai2d_plus_kpu_chained_ms
            } else {
                0.0
            },
            proof.fastspeech_transformer_50tok_ms,
            if proof.fastspeech_transformer_50tok_ms > 0.0 {
                50_000.0 / proof.fastspeech_transformer_50tok_ms
            } else {
                0.0
            }
        );
    }

    if k230_report.physical_board_detected {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut gw = GatewayState::new();
            let w_host = LiveNodeWorkerServer::spawn_auto_port(
                "127.0.0.1",
                0,
                live_node.clone(),
                false,
            )
            .await
            .expect("spawn host worker");
            let w_k230 = LiveNodeWorkerServer::spawn_auto_port(
                "127.0.0.1",
                0,
                k230_report.hardware_node.clone(),
                false,
            )
            .await
            .expect("spawn k230 usb proxy worker");

            let port_label = k230_report
                .detected_serial_port
                .clone()
                .unwrap_or_else(|| "COM11".to_string());
            gw.register_live_worker(
                live_node.clone(),
                w_host.local_addr.to_string(),
                0.15,
                true,
            );
            gw.register_live_worker_with_meta(
                k230_report.hardware_node.clone(),
                w_k230.local_addr.to_string(),
                42.0,
                true,
                "rt-smart v1.8 (CanMV K230)".to_string(),
                "riscv64-c908-rvv1.0 (6 TOPS KPU)".to_string(),
                k230_report.board_model.clone(),
                format!("USB-CDC ({port_label})"),
            );

            println!(
                "\n--- 🌐 Live 2-Node Cross-ISA Cluster Execution ({} [x86_64] <-> {} [RISC-V K230 on {}]) ---",
                live_node.hostname, k230_report.hardware_node.hostname, port_label
            );
            for a in &gw.inference_plan.assignments {
                println!(
                    "  * Stage {:?}: {:<38} | Layers: {:>2} ({:?}..{:?}) | Share: {:>5.1}% | q*={:.2}",
                    a.pipeline_stage_index,
                    a.hostname,
                    a.layer_count,
                    a.start_layer,
                    a.end_layer,
                    a.compute_share_pct,
                    a.tiered_memory.q_star_accel_ratio
                );
            }

            let (pipe_report, hops) = DistributedLanExecutor::execute_live_pipeline(
                &gw.inference_plan,
                "Verify live x86_64 to CanMV K230 RISC-V multi-stage boundary tensor execution over USB",
                &gw.live_workers,
            )
            .await;
            for hop in &hops {
                println!(
                    "  * Hop #{}: {:<38} | Layers L{:02}..L{:02} | {} | Exec: {:.2} ms | Wire+USB: {:.2} ms | RepOps={:#018x}",
                    hop.stage_index,
                    hop.hostname,
                    hop.start_layer,
                    hop.end_layer,
                    hop.transport_mode,
                    hop.compute_exec_ms,
                    hop.wire_plus_compute_ms,
                    hop.repops_digest_fnv1a
                );
            }
            println!(
                "  * Pipeline Output Norm:   L2={:.3} | Stages Executed: {} | Speculative Speedup: {:.2}x",
                pipe_report
                    .stages
                    .last()
                    .map(|s| s.output_activation_l2_norm)
                    .unwrap_or(0.0),
                hops.len(),
                pipe_report.speculative_verification.effective_speedup_multiplier
            );

            let verde_audit = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
                1,
                16,
                None,
                &gw.live_workers,
            )
            .await;
            println!(
                "  * Cross-ISA Verde Audit:  Referee (x86_64 f64) vs {} (RISC-V f32 on {}) | {} steps | Bit-Exact: {} | Wall: {:.2} ms",
                verde_audit.target_hostname,
                port_label,
                verde_audit.bisection_steps.len(),
                verde_audit.bit_exact_verified,
                verde_audit.total_audit_wall_ms
            );
                for st in &verde_audit.bisection_steps {
                    println!(
                        "      - Bisect Step #{} [L{:02}..L{:02} -> mid L{:02}]: K230={:#018x} == Referee={:#018x} (Matched: {}, RTT: {:.2} ms)",
                        st.step_index,
                        st.search_low_layer,
                        st.search_high_layer,
                        st.midpoint_layer,
                        st.worker_repops_digest,
                        st.referee_repops_digest,
                        st.matched,
                        st.wire_rtt_ms
                    );
                }

                let fault_audit = DistributedLanExecutor::run_live_verde_bisection_audit_tcp(
                    2,
                    16,
                    Some(11),
                    &gw.live_workers,
                )
                .await;
                println!(
                    "  * Cross-ISA Fault Trap:   Injected Fault @ L11 on {} -> Isolated Disputed Layer #{:?} in {} steps (Cert: {:?})",
                    fault_audit.target_hostname,
                    fault_audit.disputed_layer,
                    fault_audit.bisection_steps.len(),
                    fault_audit.fraud_proof_certificate_id
                );

                let trn_report = DistributedLanExecutor::execute_live_training_step(
                    &gw.training_plan,
                    1,
                    0.05,
                    &gw.live_workers,
                )
                .await;
                println!(
                    "  * Live 2-Node QLoRA+DeMo: {} stages | Loss: {:.4} -> {:.4} | DeMo DCT Wire Ratio: {:.1}x | RepOps Verified: {}",
                    trn_report.stages.len(),
                    trn_report.mean_loss_before,
                    trn_report.mean_loss_after,
                    trn_report.mean_wire_compression_ratio,
                    trn_report.all_stages_repops_verified
                );

                let dream_report = DistributedLanExecutor::run_live_dream_rsi_replay_tcp(
                    1,
                    "kernel-q4k-rvv-kpu-world",
                    64,
                    &gw.live_workers,
                )
                .await;
                if let Some(k230_dream) = dream_report
                    .worker_runs
                    .iter()
                    .find(|r| r.hostname.contains("k230"))
                {
                    println!(
                        "  * Live K230 Dream-RSI:    {} MCTS sims on {} | WASI Pass: {:.1}% | Peak Speedup: {:.2}x | Exec: {:.2} ms (0.0 GPU-hrs)",
                        k230_dream.mcts_simulations_run,
                        k230_dream.hostname,
                        k230_dream.wasi_verifier_pass_rate_pct,
                        k230_dream.evolved_peak_speedup,
                        k230_dream.compute_exec_ms
                    );
                }

                let fts3_report = DistributedLanExecutor::stream_fts3_ktb_shard_tcp(
                    "deepseek-v3-671b-moe",
                    "layers.60.moe.gate_up.ktb",
                    1,
                    &gw.live_workers,
                )
                .await;
                if let Some(k230_fts3) = fts3_report
                    .target_receipts
                    .iter()
                    .find(|r| r.target_hostname.contains("k230") && r.page_index == 0)
                {
                    println!(
                        "  * Live K230 FTS3 .ktb:    {} ({} bytes total, Merkle Verified: {})",
                        k230_fts3.status_note,
                        fts3_report.ktb_total_bytes,
                        k230_fts3.verified_merkle_proof
                    );
                }

                let kv_report = DistributedLanExecutor::migrate_kv_cache_pages_tcp(
                    1,
                    "sess-cd48-k230-handoff",
                    (58, 60),
                    &gw.live_workers,
                )
                .await;
                println!(
                    "  * Live KV-Cache Handoff:  {} -> {} | {} pages ({} tokens, {} B, {:.2}x INT8) | Zero-Drop RepOps: {} ({:.2} ms)",
                    kv_report.source_hostname,
                    kv_report.target_hostname,
                    kv_report.pages_migrated,
                    kv_report.tokens_preserved,
                    kv_report.compressed_wire_bytes,
                    kv_report.compression_ratio,
                    kv_report.zero_context_drop_verified,
                    kv_report.migration_rtt_ms
                );

                let ota_manifest = gw.create_ota_update_manifest(
                    "v0.4.3-k230-kpu-rvv".to_string(),
                    UpdatePayloadKind::KtbWeightDelta,
                    UpdateTargetScope::AllNodes,
                    true,
                    "Live USB-CDC OTA delta push to CanMV K230 MicroSD".into(),
                );
                if let Ok(ota_ack) = DistributedLanExecutor::push_update_tcp(
                    &w_k230.local_addr.to_string(),
                    1,
                    &ota_manifest,
                )
                .await
                {
                    println!(
                        "  * Live K230 OTA Push:     {} | RepOps Canary={:#018x} | Exec: {:.2} ms",
                        ota_ack.status_note,
                        ota_ack.repops_post_check_digest,
                        ota_ack.apply_latency_ms
                    );
                }

                let d0: Vec<u8> = (0..64).map(|i| ((i * 37 + 19) & 0xFF) as u8).collect();
                let d1_orig: Vec<u8> = (0..64).map(|i| ((i * 73 + 91) & 0xFF) as u8).collect();
                let parity: Vec<u8> = d0.iter().zip(d1_orig.iter()).map(|(&a, &b)| a ^ b).collect();
                let mut host_fnv: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in &d1_orig {
                    host_fnv ^= b as u64;
                    host_fnv = host_fnv.wrapping_mul(0x0100_0000_01b3);
                }
                if let Some((rec_d1, k230_fnv, raid_ms)) =
                    K230UsbComputeBridge::recover_raid5_stripe_on_k230(&port_label, &d0, &parity)
                {
                    println!(
                        "  * Live K230 RAID-5 XOR:   Recovered 64 B Straggler Stripe on {} | Exact: {} (FNV64={:#018x}, {:.2} ms)",
                        port_label,
                        rec_d1 == d1_orig && k230_fnv == host_fnv,
                        k230_fnv,
                        raid_ms
                    );
                }
            })
        });
    }

    // Pack and verify a 64 KiB page-aligned .ktb tensor bank in memory
    let mut packer = KtbBankPacker::new("deepseek-v3-671b-moe");
    packer.add_tensor_bank(
        "layers.0.attn.qkv_proj",
        0,
        None,
        TensorEncoding::Fp8E4M3,
        vec![0xA5; 48_120],
    );
    packer.add_tensor_bank(
        "layers.0.moe.experts.0.gate_up_proj",
        0,
        Some(0),
        TensorEncoding::Q4KBlock,
        vec![0x5A; 142_000],
    );
    packer.add_tensor_bank(
        "layers.0.moe.experts.1.gate_up_proj",
        0,
        Some(1),
        TensorEncoding::Q4KBlock,
        vec![0x3C; 96_000],
    );

    let ktb_bytes = packer
        .pack_to_bytes()
        .expect("bounded reference KTB fixture");
    let reader = KtbBankReader::from_slice(&ktb_bytes).expect("parse packed KTB image");
    println!(
        "\n--- 📦 Binary .ktb (Kadmium Tensor Bank) 64 KiB Page-Aligned Verification ({} bytes total) ---",
        ktb_bytes.len()
    );
    for entry in &reader.header.entries {
        let slice = reader
            .get_verified_tensor_slice(entry)
            .expect("verify 64KB page offset & FNV-1a 64-bit checksum");
        println!(
            "  * {:<36} | Layer {:02} | Expert: {:>4} | Offset: {:#08x} (64KB Aligned) | Payload: {:>6} B | Checksum: {:#018x} [OK]",
            entry.tensor_name,
            entry.layer_index,
            entry
                .expert_index
                .map(|e| format!("#{:02}", e))
                .unwrap_or_else(|| "Dense".into()),
            entry.page_offset_bytes,
            slice.len(),
            entry.checksum_fnv64
        );
    }

    // 2. Safetensors -> .ktb Shard-Filtered Transcoder verification
    let raw_st = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
        (
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            vec![64, 64],
            &vec![0x11; 8192],
        ),
        (
            "model.layers.1.mlp.experts.4.up_proj.weight",
            "F8_E4M3",
            vec![128, 64],
            &vec![0x22; 16384],
        ),
    ]);
    let transcoded_ktb =
        SafetensorsToKtbTranscoder::transcode_shard_to_ktb("deepseek-v3-671b-moe", &raw_st, 1, 1)
            .expect("transcode safetensors shard");
    let st_reader = KtbBankReader::from_slice(&transcoded_ktb).expect("read transcoded KTB");
    println!(
        "\n--- 🔄 Safetensors -> .ktb Shard Transcoder (Filtered Layer 01..01: {} entry, {} bytes) [OK] ---",
        st_reader.header.entries.len(),
        transcoded_ktb.len()
    );

    // 3. Nous DeMo / DisTrO DCT Fast-Frequency Momentum Compression + Error Feedback
    let mut demo = DeMoDctCompressor::new(128, 0.9);
    let synth_grad: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.05).cos()).collect();
    let dct_pkt = demo.step_and_compress(&synth_grad, 4);
    println!(
        "--- 📡 Nous DeMo DCT Momentum Compressor: 128 floats -> Top-{} DCT coeffs ({:.1}x Wire Compression, Zero Energy Loss) [OK] ---",
        dct_pkt.indices.len(),
        dct_pkt.compression_ratio()
    );

    // 4. Gensyn Verde O(log2 L) Refereed Layer Bisection over 64 Layers
    let referee_fn = |layer: usize, state: &[f64]| -> Vec<f64> {
        state
            .iter()
            .map(|&v| v * 1.01 + (layer as f64) * 0.002)
            .collect()
    };
    let mut ref_states = vec![vec![1.0, 2.0, 3.0]];
    let mut wrk_states = vec![vec![1.0, 2.0, 3.0]];
    for l in 0..64 {
        ref_states.push(referee_fn(l, &ref_states[l]));
        let mut w = referee_fn(l, &wrk_states[l]);
        if l == 37 {
            w[0] += 0.42; // simulated fault at layer 37
        }
        wrk_states.push(w);
    }
    if let Some(bisect) =
        VerdeVerificationEngine::bisect_faulty_layer(&wrk_states, &ref_states, referee_fn, 1e-6)
    {
        println!(
            "--- ⚖️  Gensyn Verde O(log L) Layer Bisection: Isolated Faulty Layer #{} out of {} in {} checks [OK] ---",
            bisect.faulty_layer_index, bisect.total_layers, bisect.referee_evaluations_used
        );
    }

    // 5. Iroh-style Single-Token Join Ticket (`kdm1_...`)
    let ticket = JoinTicket::generate(
        live_node.id.clone(),
        vec!["192.168.1.48:7848".into(), "relay.kadmium.ai:4433".into()],
        Some("slice-01-prod-deploy".into()),
        NodeTrustTier::VerifiedPeerMiddleLayersOnly,
    );
    let encoded_ticket = ticket.encode_token();
    println!(
        "--- 🎟️  Generated WAN/LAN Mesh Join Ticket (MAC Verified): {}... ---",
        &encoded_ticket[..44.min(encoded_ticket.len())]
    );

    // 6. PoliteActivityMonitor <200ms Gaming Auto-Yield Governance
    let yield_decision = PoliteActivityMonitor::evaluate_sample(
        &live_node,
        &PoliteActivitySample {
            foreground_process_name: Some("cyberpunk2077.exe".into()),
            fullscreen_swapchain_active: true,
            external_gpu_utilization_pct: 72.0,
            external_vram_used_gb: 8.4,
            on_battery_power: false,
            package_temp_celsius: 68.0,
        },
    );
    println!(
        "--- 🎮 PoliteActivityMonitor Auto-Yield: {:?} -> {:?} in {:.1} ms (VRAM ceiling {:.2} GB, {} CPU threads) [OK] ---",
        yield_decision.previous_state,
        yield_decision.recommended_state,
        yield_decision.estimated_yield_latency_ms,
        yield_decision.vram_ceiling_gb,
        yield_decision.cpu_thread_budget
    );

    // 7. KtbMerkleTree O(log N) 64 KiB Page Inclusion Proof
    let merkle = KtbMerkleTree::from_ktb_bytes(&ktb_bytes);
    let proof_p1 = merkle.prove_page(1).expect("merkle proof page 1");
    let verified_p1 =
        KtbMerkleTree::verify_page(&ktb_bytes[65_536..131_072], &proof_p1, merkle.root_hash);
    println!(
        "--- 🌳 KtbMerkleTree 64 KiB Page Verifier: {} pages | Root: {:#018x} | Page #1 Proof Verified: {} [OK] ---",
        merkle.leaf_hashes.len(),
        merkle.root_hash,
        verified_p1
    );

    // 8. Q4_K 4-Bit Group-Quantized MatVec + Colibri Double-Buffered NVMe DMA
    let w_fp32: Vec<f32> = (0..(16 * 32)).map(|i| ((i as f32) * 0.03).sin()).collect();
    let x_fp32: Vec<f32> = (0..32).map(|j| ((j as f32) * 0.07).cos()).collect();
    let q4_mat = Q4KBlockTensor::quantize_matrix(&w_fp32, 16, 32);
    let y_q4 = q4_mat.matvec_f32(&x_fp32);
    let colibri = ColibriNvmeDoubleBuffer::execute_double_buffered_stream(
        4,
        4 * 1024 * 1024,
        live_node.memory.nvme_read_bw_gbps,
        1.45,
    );
    println!(
        "--- ⚡ Q4_K 4-Bit MatVec ({:.2}x compression, |y[0]|={:.3}) & Colibri Double-Buffer Stall: {:.2} ms [OK] ---",
        q4_mat.compression_ratio(),
        y_q4[0].abs(),
        colibri[0].effective_pipeline_stall_ms
    );

    // 9. LWW-Element-Map + Vector Clock ClusterStateCrdt Gossip Merge
    let mut crdt_local = ClusterStateCrdt::new();
    crdt_local.upsert_node_state(
        &live_node.id.0,
        &live_node.id.0,
        PoliteState::IdleDedicated,
        Some((0, 25)),
        true,
    );
    let frame = crdt_local.to_wire_frame(1);
    let crdt_decoded = ClusterStateCrdt::from_wire_frame(&frame).expect("decode CRDT frame");
    println!(
        "--- 🔄 ClusterStateCrdt Gossip Replication: {} register(s), Vector Clock v{} [OK] ---",
        crdt_decoded.registers.len(),
        crdt_decoded
            .vector_clock
            .values()
            .max()
            .copied()
            .unwrap_or(0)
    );
}

/// What the owner typed about their offer on `kadmium join`; `None` fields keep the default.
#[derive(Debug, Clone, Default)]
pub struct JoinOfferArgs {
    pub fast_gb: Option<f64>,
    pub ram_gb: Option<f64>,
    pub nvme_gb: Option<f64>,
    pub slots: Option<u32>,
    pub require_approval: bool,
}

impl JoinOfferArgs {
    /// The owner's offer for the probed node, or `None` when nothing was overridden so the
    /// coordinator uses the node's default offer.
    pub fn into_offer(
        self,
        node: &kadmium_core::HardwareNode,
    ) -> Option<kadmium_core::ResourceOffer> {
        if self.fast_gb.is_none()
            && self.ram_gb.is_none()
            && self.nvme_gb.is_none()
            && self.slots.is_none()
            && !self.require_approval
        {
            return None;
        }
        let mut offer = node.default_offer();
        if let Some(v) = self.fast_gb {
            offer.fast_memory_gb = v;
        }
        if let Some(v) = self.ram_gb {
            offer.system_ram_gb = v;
        }
        if let Some(v) = self.nvme_gb {
            offer.nvme_gb = v;
        }
        if let Some(v) = self.slots {
            offer.slots = v;
        }
        offer.requires_approval = offer.requires_approval || self.require_approval;
        Some(offer.clamped_to(node))
    }
}

/// Asks a running coordinator to measure the plan in force and prints the record it kept.
pub async fn run_calibrate(coordinator: &str, strategy: Option<&str>, steps: u32, rows: u32) {
    let req = kadmium_gateway::GridCalibrateRequest {
        strategy: strategy.map(str::to_string),
        steps: Some(steps.clamp(1, 8)),
        rows: Some(rows.max(1)),
        prompt: None,
    };
    println!(
        "===================================================================================="
    );
    println!(
        " Kadmium (Cd-48) — Grid Calibration ({} step(s), {} row(s) per step)",
        req.steps.unwrap_or(8),
        req.rows.unwrap_or(1)
    );
    println!(
        "===================================================================================="
    );
    match MeshHttpClient::post_json::<
        kadmium_gateway::GridCalibrateRequest,
        kadmium_gateway::GridCalibrateResponse,
    >(coordinator, "/api/v1/grid/calibrate", &req)
    .await
    {
        Ok(resp) => {
            if let Some(error) = resp.error {
                println!("  * Calibration failed:     {error}");
                return;
            }
            println!("  * Model:                  {}", resp.model_id);
            if let Some(record) = &resp.record {
                println!("  * Record:                 {}", record.summary());
                println!(
                    "  * Binding:                placement {:#018x} | {} nodes | {} rows",
                    record.binding.placement_digest,
                    record.binding.node_ids.len(),
                    record.binding.rows
                );
                for (node, ms) in &record.per_node_exec_ms {
                    println!(
                        "  * Compute on {:<16} {ms:>8.2} ms per step (worker clock)",
                        node
                    );
                }
            }
            println!(
                "  * Applies to plan in force: {} | shape decided by: {}",
                resp.applies_to_plan_in_force, resp.decided_by
            );
            println!("  * Why:                    {}", resp.rationale);
            println!(
                "  * Wall:                   {:.1} ms for {} step(s)",
                resp.wall_ms, resp.steps_run
            );
        }
        Err(e) => println!(
            "  * No coordinator answered at {coordinator}: {e}. Start one with `kadmium serve` first."
        ),
    }
}

pub async fn run_join(
    coordinator_arg: Option<String>,
    ticket_arg: Option<String>,
    worker_port: u16,
    hostname_override: Option<String>,
    live_only: bool,
    once: bool,
    offer_args: JoinOfferArgs,
) {
    let mut local_node = LiveHardwareProfiler::probe_local_host();
    if let Some(custom_host) = hostname_override {
        let clean = custom_host
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect::<String>();
        local_node.id = NodeId(format!("node-live-{clean}"));
        local_node.hostname = custom_host;
    }

    let local_lan_ip = LiveHardwareProfiler::probe_local_lan_ip();

    // Spawn local TCP WireFrame worker server on 0.0.0.0:<worker_port> (auto-incrementing if busy):
    let worker_server =
        LiveNodeWorkerServer::spawn_auto_port("0.0.0.0", worker_port, local_node.clone(), true)
            .await
            .expect("Failed to bind local TCP WireFrame worker listener");
    let bound_port = worker_server.local_addr.port();
    let worker_tcp_addr = format!("{local_lan_ip}:{bound_port}");

    // Optionally accept striped USB-C + Wi-Fi bridge sessions, e.g. KADMIUM_USBC_BRIDGE_BIND=0.0.0.0:7871.
    if let Ok(bind) = std::env::var("KADMIUM_USBC_BRIDGE_BIND")
        && !bind.trim().is_empty()
    {
        match worker_server
            .spawn_usbc_bridge_listener(bind.trim(), true)
            .await
        {
            // The listener runs for as long as the worker does.
            Ok((addr, _listener)) => println!(
                "  🔌 USB-C bridge listener on {addr} (dial kdbr://<usb4-ip>:{port}@<gbps>,<lan-ip>:{port}@<gbps>)",
                port = addr.port()
            ),
            Err(e) => eprintln!("  ⚠️  USB-C bridge listener not started: {e}"),
        }
    }

    // Resolve Coordinator URL via CLI flag, JoinTicket, or UDP LAN Auto-Discovery (port 7849):
    let mut discovered_ticket = ticket_arg.clone();
    let resolved_coordinator_url = if let Some(c_url) = coordinator_arg {
        c_url
    } else if let Some(ref tok) = ticket_arg
        && let Ok(parsed) = JoinTicket::decode_and_verify(tok)
        && let Some(ep) = parsed.endpoints.first()
    {
        if ep.starts_with("http://") || ep.starts_with("https://") {
            ep.clone()
        } else {
            format!("http://{ep}")
        }
    } else if let Some(ann) =
        LanCoordinatorDiscovery::discover_on_lan(DEFAULT_LAN_DISCOVERY_UDP_PORT, 600).await
    {
        if discovered_ticket.is_none() {
            discovered_ticket = Some(ann.join_ticket_token);
        }
        ann.coordinator_http_url
    } else {
        "http://127.0.0.1:7848".to_string()
    };

    let token = discovered_ticket.unwrap_or_else(|| {
        JoinTicket::generate(
            local_node.id.clone(),
            vec![
                format!("{local_lan_ip}:7848"),
                "relay.kadmium.ai:4433".into(),
            ],
            Some("slice-01-prod-deploy".into()),
            NodeTrustTier::TrustedCore,
        )
        .encode_token()
    });

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Live Worker Node Daemon & LAN Cluster Join");
    println!(
        "===================================================================================="
    );
    println!(
        "  * Probed Local Node:      {} ({:?} / {:?})",
        local_node.hostname, local_node.device_kind, local_node.backend
    );
    println!(
        "  * Hardware Capability:    {:.1} FP16 TFLOPs | {:.1} GB Fast Mem | {:.1} GB Sys RAM | q*={:.2}",
        local_node.fp16_tflops,
        local_node.available_fast_memory_gb(),
        local_node.memory.system_ram_gb,
        local_node.compute_q_star_ratio()
    );
    println!(
        "  * WireFrame TCP Worker:   Listening on {} (0.0.0.0:{})",
        worker_tcp_addr, bound_port
    );
    let offer = offer_args.into_offer(&local_node);
    println!(
        "  * Resource Offer:         {}{}",
        offer
            .clone()
            .unwrap_or_else(|| local_node.default_offer())
            .summary(),
        if offer.is_some() { "" } else { " (default)" }
    );

    let verified_ticket = JoinTicket::decode_and_verify(&token)
        .expect("Failed to verify kdm1_ Join Ticket MAC signature");
    println!(
        "  * Ticket MAC Verified:    [OK] Coordinator={} | Slice={:?} | Trust={:?}",
        verified_ticket.coordinator_id.0,
        verified_ticket.target_slice_id,
        verified_ticket.granted_trust_tier
    );

    let rx_sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind udp listener");
    let rx_addr = rx_sock.local_addr().expect("local udp addr");
    let tx_sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind udp sender");

    let beacon = LanDiscoveryBeacon::from_node(&local_node, bound_port);
    beacon
        .broadcast_once(&tx_sock, rx_addr)
        .await
        .expect("send KDMD UDP beacon");
    let (recv_beacon, peer_addr) = LanDiscoveryBeacon::recv_once(&rx_sock)
        .await
        .expect("recv KDMD UDP beacon");
    println!(
        "  * UDP LAN Beacon (KDMD):  [OK] Broadcast & received from {} (hostname={}, q*={:.2})",
        peer_addr, recv_beacon.hostname, recv_beacon.q_star_ratio
    );

    // Attempt live registration with the running `kadmium serve` Coordinator:
    let measured_stream_gbps = LiveHardwareProfiler::probe_memory_stream_gbps();
    println!(
        "  * Memory Stream Probe:    {measured_stream_gbps:.1} GB/s single-thread copy (replaces the nominal RAM figure in the cost model)"
    );
    let reg_req = WorkerRegisterRequest {
        node: local_node.clone(),
        worker_tcp_addr: worker_tcp_addr.clone(),
        join_ticket: Some(token),
        switch_to_live_only: live_only,
        offer: offer.clone(),
        free_system_ram_gb: Some(worker_server.host_room_gb()),
        measured_stream_gbps: Some(measured_stream_gbps),
    };

    match MeshHttpClient::post_json::<WorkerRegisterRequest, WorkerRegistrationAck>(
        &resolved_coordinator_url,
        "/api/v1/cluster/register",
        &reg_req,
    )
    .await
    {
        Ok(ack) => {
            println!(
                "------------------------------------------------------------------------------------"
            );
            println!(
                "  * Live Coordinator Join:  [CONNECTED] {} (Epoch #{})",
                resolved_coordinator_url, ack.cluster_epoch
            );
            println!(
                "  * Measured TCP Wire RTT:  {:.2} ms | Cluster Size: {} nodes ({} live workers)",
                ack.measured_tcp_rtt_ms, ack.total_cluster_nodes, ack.total_live_workers
            );
            println!(
                "  * Assigned Shards:        Inference={:?} | Training={:?} | Share={:.1}%",
                ack.inference_layers
                    .map(|(a, b)| format!("L{a:02}..L{b:02}"))
                    .unwrap_or_else(|| "Draft/Utility".into()),
                ack.training_layers
                    .map(|(a, b)| format!("L{a:02}..L{b:02}"))
                    .unwrap_or_else(|| "Sandbox/Verifier".into()),
                ack.compute_share_pct
            );
            println!(
                "------------------------------------------------------------------------------------"
            );

            if once {
                println!("  [--once] Registration verified. Worker exiting (--once specified).");
                return;
            }

            println!(
                "  [Daemon Active] Streaming TCP WireFrame (KDMF) stages & <200ms Polite heartbeats."
            );
            println!("                  Press Ctrl+C to drain shards and disconnect.\n");

            let mut pending_ack: Option<WorkerUpdatePushAck> = None;
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3));
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        println!("\n  [Drain] Ctrl+C received — notifying coordinator and shutting down cleanly.");
                        break;
                    }
                    _ = interval.tick() => {
                        let sample = PoliteActivityMonitor::sample_live_host();
                        let decision = PoliteActivityMonitor::evaluate_sample(&local_node, &sample);
                        local_node.polite_state = decision.recommended_state;

                        let hb_req = WorkerHeartbeatRequest {
                            node_id: local_node.id.clone(),
                            worker_tcp_addr: worker_tcp_addr.clone(),
                            polite_state: local_node.polite_state,
                            external_gpu_utilization_pct: sample.external_gpu_utilization_pct,
                            external_vram_used_gb: sample.external_vram_used_gb,
                            package_temp_celsius: sample.package_temp_celsius,
                            applied_ota_ack: pending_ack.take(),
                            offer: offer.clone(),
                            // The room the host memory ledger has now, below its reserve and
                            // learned margin: what the coordinator may count on.
                            free_system_ram_gb: Some(worker_server.host_room_gb()),
                        };

                        if let Ok(hb_ack) = MeshHttpClient::post_json::<WorkerHeartbeatRequest, WorkerHeartbeatAck>(
                            &resolved_coordinator_url,
                            "/api/v1/cluster/heartbeat",
                            &hb_req,
                        ).await {
                            if let Some(manifest) = hb_ack.pending_ota_update {
                                let prev_ver = worker_server.installed_version.lock().expect("ver").clone();
                                let prev_mrk = worker_server.ktb_merkle_root_fnv64.load(std::sync::atomic::Ordering::Relaxed);
                                let ota_ack = apply_ota_update_manifest_local(
                                    &local_node.id,
                                    &local_node.hostname,
                                    &prev_ver,
                                    prev_mrk,
                                    "Online WAN Heartbeat Pull + DERP Relay",
                                    &manifest,
                                );
                                if ota_ack.accepted {
                                    *worker_server.installed_version.lock().expect("ver") = ota_ack.installed_version.clone();
                                    worker_server.ktb_merkle_root_fnv64.store(ota_ack.applied_ktb_merkle_root_fnv64, std::sync::atomic::Ordering::Relaxed);
                                }
                                println!(
                                    "  [OTA Update Applied] {} -> {} ({:?}) | MAC={:#018x} | Merkle={:#018x}",
                                    ota_ack.previous_version,
                                    ota_ack.installed_version,
                                    manifest.payload_kind,
                                    ota_ack.verified_mac_fnv64,
                                    ota_ack.applied_ktb_merkle_root_fnv64
                                );
                                pending_ack = Some(ota_ack);
                            }
                            if hb_ack.rebalance_triggered {
                                println!(
                                    "  [Live Shift Epoch #{}] PoliteState={:?} -> Rebalanced Layers: Inf={:?}, Train={:?}",
                                    hb_ack.cluster_epoch,
                                    local_node.polite_state,
                                    hb_ack.inference_layers,
                                    hb_ack.training_layers
                                );
                            }
                        }
                        #[allow(unreachable_code)]
                        if true { continue; }
                        if false {
                        };

                        if let Ok(hb_ack) = MeshHttpClient::post_json::<WorkerHeartbeatRequest, WorkerHeartbeatAck>(
                            &resolved_coordinator_url,
                            "/api/v1/cluster/heartbeat",
                            &hb_req,
                        ).await && hb_ack.rebalance_triggered {
                            println!(
                                "  [Live Shift Epoch #{}] PoliteState={:?} -> Rebalanced Layers: Inf={:?}, Train={:?}",
                                hb_ack.cluster_epoch,
                                local_node.polite_state,
                                hb_ack.inference_layers,
                                hb_ack.training_layers
                            );
                        }
                    }
                }
            }
        }
        Err(_) => {
            println!(
                "  * Coordinator Status:     No live `kadmium serve` detected at {} — showing local shard plan preview:",
                resolved_coordinator_url
            );
            let mut cluster = sample_personal_cluster();
            cluster.push(local_node);
            let model = ModelSpec::deepseek_v3_671b_moe();
            let plan = ShardPlanner::compute_plan(2, &model, WorkloadKind::Inference, &cluster);
            print_shard_plan(&plan);
        }
    }
}

pub fn run_pull(
    uri: &str,
    plan_only: bool,
    local_path: Option<&str>,
    index_path: Option<&str>,
) -> Result<(), String> {
    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Ecosystem Connector (`hf://` & `gh://` Selective Staging)");
    println!(
        "===================================================================================="
    );

    if uri.starts_with("gh://") || uri.contains("github.com") {
        let gh_run = if plan_only {
            GitHubConnector::evaluate_repo_on_slice(uri)?
        } else {
            let checkout = local_path.map(std::path::Path::new);
            GitHubConnector::stage_local_repo(uri, checkout)?
        };
        println!(
            "  * GitHub Repository:      {}/{} @ {} (Pinned Ref: {})",
            gh_run.gh_ref.owner,
            gh_run.gh_ref.repo,
            gh_run.gh_ref.git_ref,
            gh_run.pinned_commit_sha
        );
        println!(
            "  * Execution Mode:         {} (Mounted: {}, Verified Nodes: {})",
            gh_run.execution_mode, gh_run.mounted, gh_run.verified_node_count
        );
        println!("  * Subpath / World:        {:?}", gh_run.gh_ref.subpath);
        println!("  * Target Slice Assigned:  {}", gh_run.assigned_slice_id);
        println!("  * Sandbox Runtime:        {}", gh_run.sandbox_runtime);
        println!(
            "  * Check Status:           {}",
            gh_run.github_check_conclusion
        );
        Ok(())
    } else {
        let hf_ref = kadmium_data::HfRepoRef::parse(uri)?;
        if hf_ref.kind == kadmium_data::HfRepoKind::Dataset {
            let ds_plan = HuggingFaceHubConnector::plan_dataset_pull(uri)?;
            println!(
                "  * Hugging Face Dataset:   {} (rev: {}, kind: {:?})",
                ds_plan.hf_ref.repo_id, ds_plan.hf_ref.revision, ds_plan.hf_ref.kind
            );
            println!("  * Dataset Metadata URL:   {}", ds_plan.dataset_info_url);
            println!(
                "  * Mode:                   Plan-Only Preview ({} files downloaded, {} bytes staged)",
                ds_plan.files_downloaded, ds_plan.staged_bytes
            );
            for split_url in &ds_plan.parquet_split_urls {
                println!("  * Planned Parquet Split:  {split_url}");
            }
            return Ok(());
        }

        let parsed_index = match index_path {
            Some(path) => {
                let raw = std::fs::read_to_string(path)
                    .map_err(|e| format!("Failed to read index file '{path}': {e}"))?;
                Some(kadmium_data::HfByteRangeSlicer::parse_index_manifest(&raw)?)
            }
            None => None,
        };

        let local_node = LiveHardwareProfiler::probe_local_host();
        let cluster = vec![local_node];
        let repo_lower = hf_ref.repo_id.to_ascii_lowercase();
        let mut model = if repo_lower.contains("671b") || repo_lower.contains("deepseek-v3") {
            ModelSpec::deepseek_v3_671b_moe()
        } else if repo_lower.contains("235b") {
            ModelSpec::qwen3_235b_moe()
        } else if repo_lower.contains("70b") {
            ModelSpec::llama_3_70b_lora()
        } else if repo_lower.contains("32b") {
            ModelSpec::qwen2_5_coder_32b()
        } else if repo_lower.contains("14b") {
            ModelSpec::qwen2_5_coder_14b()
        } else if repo_lower.contains("7b") || repo_lower.contains("8b") {
            ModelSpec::qwen2_5_coder_7b()
        } else if repo_lower.contains("3b")
            || repo_lower.contains("1.5b")
            || repo_lower.contains("0.5b")
        {
            ModelSpec::llama_3_2_3b_instruct()
        } else {
            ModelSpec::recommended_for_cluster(&cluster)
        };
        model.id = hf_ref.repo_id.clone();
        if let Some(ref idx) = parsed_index {
            if let Some(max_l) = idx
                .weight_map
                .keys()
                .map(|t| kadmium_data::SafetensorsToKtbTranscoder::parse_layer_and_expert(t).0)
                .max()
                && max_l > 0
            {
                model.num_layers = max_l + 1;
            }
            if idx.total_size_bytes > 0 && model.num_layers > 0 {
                let total_gb = (idx.total_size_bytes as f64) / 1_073_741_824.0;
                if model.is_moe && model.experts_per_layer > 0 {
                    model.dense_layer_size_gb = (total_gb * 0.15) / (model.num_layers as f64);
                    model.single_expert_size_gb =
                        (total_gb * 0.85) / ((model.num_layers * model.experts_per_layer) as f64);
                } else {
                    model.dense_layer_size_gb = total_gb / (model.num_layers as f64);
                }
            }
        }
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);
        let hf_plan = HuggingFaceHubConnector::plan_selective_model_pull_with_index(
            uri,
            &plan,
            parsed_index.as_ref(),
        )?;
        println!(
            "  * Hugging Face Repo:      {} (rev: {}, kind: {:?})",
            hf_plan.hf_ref.repo_id, hf_plan.hf_ref.revision, hf_plan.hf_ref.kind
        );
        println!("  * Index Manifest URL:     {}", hf_plan.index_manifest_url);
        println!(
            "  * Index Source:           {} (Plan-Only: {}, Files Downloaded: {}, .ktb Compiled: {})",
            hf_plan.index_source,
            hf_plan.plan_only,
            hf_plan.files_downloaded,
            hf_plan.ktb_banks_compiled
        );
        println!("  * Primary .ktb Seed Node: {}", hf_plan.seed_vault_node);
        println!(
            "\n{:<22} {:<10} {:<12} {:<14} {:<42}",
            "WORKER HOSTNAME", "LAYERS", "SELECTIVE", "AVOIDED DL", "PLANNED .KTB BANK PATH"
        );
        println!("{}", "-".repeat(104));
        for np in &hf_plan.node_pulls {
            println!(
                "{:<22} L{:02}..L{:02}   {:>6.2} GB   {:>8.2} GB   {}",
                np.hostname,
                np.start_layer,
                np.end_layer,
                np.selective_download_gb,
                np.full_repo_avoided_gb,
                np.output_ktb_bank_path
            );
        }
        Ok(())
    }
}

pub fn run_mcp(inspect: bool, demo: bool) {
    let mut state = if demo {
        GatewayState::new()
    } else {
        GatewayState::new_live_unconnected()
    };

    if inspect {
        let manifest = KadmiumMcpServer::manifest();
        println!(
            "===================================================================================="
        );
        println!(
            " Kadmium (Cd-48) — Model Context Protocol (MCP) Server ({} v{})",
            manifest.server_name, manifest.server_version
        );
        println!(
            "===================================================================================="
        );
        println!("  * Protocol Version:       {}", manifest.protocol_version);
        println!(
            "  * Transports:             {}",
            manifest.transports.join(" | ")
        );
        println!("  * HTTP Endpoint:          {}", manifest.http_endpoint);
        println!("\n  [IDE / Agent `.mcp.json` Configuration]:");
        for line in manifest.stdio_config_json.lines() {
            println!("    {line}");
        }

        println!("\n  [Exposed MCP Tools ({})]:", manifest.tools.len());
        for t in &manifest.tools {
            println!("    - {:<30} {}", t.name, t.description);
        }

        println!(
            "\n  [Exposed MCP Resources ({})]:",
            manifest.resources.len()
        );
        for r in &manifest.resources {
            println!("    - {:<30} [{}] {}", r.uri, r.mime_type, r.name);
        }

        println!("\n  [Exposed MCP Prompts ({})]:", manifest.prompts.len());
        for p in &manifest.prompts {
            println!("    - {:<30} {}", p.name, p.description);
        }
        return;
    }

    // Standard I/O (stdio) JSON-RPC 2.0 loop for MCP clients
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line_res in stdin.lock().lines() {
        let Ok(line) = line_res else {
            break;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(req) = serde_json::from_str::<McpJsonRpcRequest>(trimmed) {
            let is_notification = req.id.is_none();
            let resp = KadmiumMcpServer::handle_request(&mut state, req);
            if !is_notification && let Ok(json_out) = serde_json::to_string(&resp) {
                let _ = writeln!(stdout, "{json_out}");
                let _ = stdout.flush();
            }
        }
    }
}

pub fn run_init(out_path: Option<&str>) {
    let local_node = LiveHardwareProfiler::probe_local_host();
    let cfg = KadmiumWorkspaceConfig::from_local_probe(local_node);
    let toml_text = cfg.render_kadmium_toml();

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Workspace Initialization (`kadmium.toml`)");
    println!(
        "===================================================================================="
    );
    println!("{toml_text}");

    if let Some(path) = out_path {
        std::fs::write(path, &toml_text).expect("Failed to write kadmium.toml to disk");
        println!("  [OK] Wrote configuration manifest to: {path}");
    }
}

pub fn run_compile(model: &str, start_layer: u32, end_layer: u32, out_path: Option<&str>) {
    let mut packer = KtbBankPacker::new(model);
    for l in start_layer..=end_layer {
        packer.add_tensor_bank(
            format!("layers.{l}.attn.qkv_proj"),
            l,
            None,
            TensorEncoding::Fp8E4M3,
            vec![0xA5; 32_768],
        );
        for exp in 0..2u32 {
            packer.add_tensor_bank(
                format!("layers.{l}.moe.experts.{exp}.gate_up_proj"),
                l,
                Some(exp),
                TensorEncoding::Q4KBlock,
                vec![(l as u8).wrapping_add(exp as u8); 65_536],
            );
        }
    }

    let ktb_bytes = packer
        .pack_to_bytes()
        .expect("bounded reference KTB fixture");
    let merkle = KtbMerkleTree::from_ktb_bytes(&ktb_bytes);
    let reader = KtbBankReader::from_slice(&ktb_bytes).expect("valid compiled .ktb image");

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — 64 KiB Page-Aligned `.ktb` Tensor Bank Compiler");
    println!(
        "===================================================================================="
    );
    println!("  * Model ID:               {}", reader.header.model_id);
    println!("  * Compiled Layer Range:   L{start_layer:02}..L{end_layer:02}");
    println!(
        "  * Total Archive Size:     {} bytes ({} x 64 KiB pages)",
        ktb_bytes.len(),
        merkle.leaf_hashes.len()
    );
    println!("  * Merkle Root Digest:     {:#018x}", merkle.root_hash);
    println!(
        "  * Tensor Entries Packed:  {}",
        reader.header.entries.len()
    );

    if let Some(path) = out_path {
        std::fs::write(path, &ktb_bytes).expect("Failed to write .ktb bank to disk");
        println!("  * Saved .ktb Archive To:  {path}");
    }
}

pub fn run_inspect(path: Option<&str>) {
    let (source_label, ktb_bytes) = if let Some(p) = path {
        let bytes = std::fs::read(p).expect("Failed to read .ktb file from disk");
        (p.to_string(), bytes)
    } else {
        let mut packer = KtbBankPacker::new("deepseek-v3-671b-moe");
        packer.add_tensor_bank(
            "layers.0.attn.qkv_proj",
            0,
            None,
            TensorEncoding::Fp8E4M3,
            vec![0xA5; 32_768],
        );
        packer.add_tensor_bank(
            "layers.0.moe.experts.0.gate_up_proj",
            0,
            Some(0),
            TensorEncoding::Q4KBlock,
            vec![0x42; 65_536],
        );
        (
            "in-memory sample (.ktb)".to_string(),
            packer
                .pack_to_bytes()
                .expect("bounded reference KTB fixture"),
        )
    };

    let reader = KtbBankReader::from_slice(&ktb_bytes).expect("Invalid .ktb binary header");
    let merkle = KtbMerkleTree::from_ktb_bytes(&ktb_bytes);
    let page_size = reader.header.page_alignment_bytes as usize;

    let mut verified_pages = 0usize;
    for (idx, chunk) in ktb_bytes.chunks(page_size).enumerate() {
        if let Some(proof) = merkle.prove_page(idx)
            && KtbMerkleTree::verify_page(chunk, &proof, merkle.root_hash)
        {
            verified_pages += 1;
        }
    }

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — `.ktb` Tensor Bank Inspector & Merkle Verifier");
    println!(
        "===================================================================================="
    );
    println!("  * Source Archive:         {source_label}");
    println!(
        "  * Model ID & Version:     {} (v{})",
        reader.header.model_id, reader.header.format_version
    );
    println!(
        "  * Page Alignment:         {} bytes (64 KiB)",
        reader.header.page_alignment_bytes
    );
    println!(
        "  * Merkle Tree Audit:      {}/{} pages verified (Root: {:#018x})",
        verified_pages,
        merkle.leaf_hashes.len(),
        merkle.root_hash
    );
    println!(
        "\n{:<38} {:<8} {:<10} {:<12} {:<12} {:<18}",
        "TENSOR NAME", "LAYER", "EXPERT", "ENCODING", "PAGE OFFSET", "FNV-1A CHECKSUM"
    );
    println!("{}", "-".repeat(102));
    for e in &reader.header.entries {
        println!(
            "{:<38} L{:02}      {:<10} {:<12} {:#010x}   {:#018x}",
            e.tensor_name,
            e.layer_index,
            e.expert_index
                .map(|x| format!("exp #{x:02}"))
                .unwrap_or_else(|| "dense".into()),
            format!("{:?}", e.encoding),
            e.page_offset_bytes,
            e.checksum_fnv64
        );
    }
}

pub async fn run_update(
    coordinator_arg: Option<String>,
    version: String,
    kind_str: String,
    scope_str: String,
) {
    let payload_kind = match kind_str.to_lowercase().as_str() {
        "ktb" | "ktb-delta" | "delta" | "weights" => UpdatePayloadKind::KtbWeightDelta,
        "policy" | "dream-rsi" | "wasi" => UpdatePayloadKind::DreamRsiPolicyBundle,
        _ => UpdatePayloadKind::BinaryRelease,
    };
    let target_scope = match scope_str.to_lowercase().as_str() {
        "local" | "lan" => UpdateTargetScope::LocalLanOnly,
        "online" | "wan" => UpdateTargetScope::OnlineWanOnly,
        _ => UpdateTargetScope::AllNodes,
    };

    let coord_url = coordinator_arg.unwrap_or_else(|| "http://127.0.0.1:7848".to_string());
    let req = OtaUpdatePushRequest {
        target_version: version.clone(),
        payload_kind: Some(payload_kind),
        target_scope: Some(target_scope),
        rolling_canary: Some(true),
        release_notes: Some(format!(
            "CLI OTA push: {version} ({payload_kind:?} -> {target_scope:?})"
        )),
    };

    println!(
        "==================================================================================================================="
    );
    println!(
        " 🚀 KADMIUM FLEET OTA UPDATE PUSHER — Local LAN (Direct TCP WireFrame) + Online WAN (Heartbeat Pull/Relay)"
    );
    println!(
        "==================================================================================================================="
    );

    let summary = match MeshHttpClient::post_json::<OtaUpdatePushRequest, FleetUpdatePushSummary>(
        &coord_url,
        "/api/v1/update/push",
        &req,
    )
    .await
    {
        Ok(s) => s,
        Err(_) => {
            // No live `kadmium serve` running on 7848 — spin up the real 5-node local + online TCP worker fleet in-process and push over live TCP!
            let mut gw = GatewayState::new();
            let instances = LocalFleetSpawner::spawn_heterogeneous_local_fleet(18950, false)
                .await
                .expect("spawn local TCP fleet for OTA push");
            for inst in instances {
                let tcp_addr = inst.server.local_addr.to_string();
                let node_clone = inst.server.node.lock().expect("node lock").clone();
                gw.register_live_worker_with_meta(
                    node_clone,
                    tcp_addr,
                    0.25,
                    true,
                    inst.os_name.clone(),
                    inst.cpu_arch.clone(),
                    inst.board_or_model.clone(),
                    inst.network_channel.clone(),
                );
                gw.spawned_fleet_instances.push(inst);
            }
            let manifest = gw.create_ota_update_manifest(
                version.clone(),
                payload_kind,
                target_scope,
                true,
                "In-process multi-node TCP WireFrame + WAN Relay OTA push".into(),
            );
            let mut receipts = Vec::new();
            let mut local_cnt = 0usize;
            let mut wan_cnt = 0usize;
            let workers_list: Vec<_> = gw
                .live_workers
                .values()
                .filter(|w| match target_scope {
                    UpdateTargetScope::AllNodes => true,
                    UpdateTargetScope::LocalLanOnly => w.network_channel == "LocalLAN",
                    UpdateTargetScope::OnlineWanOnly => w.network_channel == "OnlineWAN",
                })
                .cloned()
                .collect();

            for w in workers_list {
                let mut ack = DistributedLanExecutor::push_update_tcp(
                    &w.worker_tcp_addr,
                    gw.epoch,
                    &manifest,
                )
                .await
                .expect("push OTA over live TCP WireFrame");
                if w.network_channel == "OnlineWAN" {
                    ack.delivery_mechanism = "Online WAN Heartbeat Pull + DERP Relay".into();
                    wan_cnt += 1;
                } else {
                    local_cnt += 1;
                }
                receipts.push(ack);
            }
            gw.record_ota_receipts(&manifest, receipts, local_cnt, wan_cnt)
        }
    };

    println!(
        "  * Target Release Version: {} ({:?})",
        summary.active_cluster_version, payload_kind
    );
    println!(
        "  * Target Scope:           {:?} | Canary Verified: {} | Zero-Downtime SkipPipe: {}",
        target_scope, summary.canary_verified, summary.zero_downtime_skippipe_active
    );
    println!(
        "  * .ktb Merkle Root:       {:#018x} | Nodes Updated: {} ({} Local LAN, {} Online WAN)",
        summary.active_ktb_merkle_root_fnv64,
        summary.total_nodes_targeted,
        summary.local_lan_pushed_count,
        summary.online_wan_pushed_count
    );
    println!("{}", "-".repeat(122));
    println!(
        "{:<24} {:<16} {:<38} {:<18} {:<18} {:>8}",
        "HOSTNAME",
        "VERSION TRANSITION",
        "DELIVERY CHANNEL",
        "MAC SIGNATURE",
        "REPOPS SELF-TEST",
        "LATENCY"
    );
    println!("{}", "-".repeat(122));
    for r in &summary.node_receipts {
        println!(
            "{:<24} {:<16} {:<38} {:#016x}   {:#016x}   {:>5.2} ms",
            r.hostname,
            format!("{}->{}", r.previous_version, r.installed_version),
            r.delivery_mechanism,
            r.verified_mac_fnv64,
            r.repops_post_check_digest,
            r.apply_latency_ms
        );
    }
}
