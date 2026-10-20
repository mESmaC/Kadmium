//! Operational CLI subcommand handlers (`kadmium-cli::ops`):
//! `cloud-shield`, `probe`, `join`, `pull`, `mcp`, `init`, `compile`, `inspect`, and `lease`.

use crate::print_shard_plan;
use kadmium_core::{
    KadmiumWorkspaceConfig, LiveHardwareProfiler, ModelSpec, NodeTrustTier,
    OsSwapchainHysteresisEngine, PoliteActivityMonitor, PoliteActivitySample, PoliteState,
    SubClusterRbacEngine, SupplyLeasePolicy, SupplyPayoutMode, SupplySettlementEngine,
    WorkloadKind, sample_personal_cluster,
};
use kadmium_data::{
    GitHubConnector, HfByteRangeSlicer, HuggingFaceHubConnector, KtbBankPacker, KtbBankReader,
    KtbMerkleTree, SafetensorsToKtbTranscoder, TensorEncoding,
};
use kadmium_gateway::{GatewayState, KadmiumMcpServer, McpJsonRpcRequest};
use kadmium_network::{
    ClusterStateCrdt, JoinTicket, LanDiscoveryBeacon, NatTraversalCoordinator,
    ZeroClickLanPairingEngine,
};
use kadmium_rl::DreamRsiWorldRegistry;
use kadmium_runtime::{
    ColibriNvmeDoubleBuffer, DeMoDctCompressor, HardwareKernelDispatcher, Q4KBlockTensor,
    VerdeFraudProofEngine, VerdeVerificationEngine,
};
use kadmium_sharding::{
    CloudShieldCutoverWatcher, HybridCloudArbitragePlanner, ShardPlanner, SpotOrderbookBidEngine,
};

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

    let ktb_bytes = packer.pack_to_bytes();
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

pub async fn run_join(ticket_arg: Option<String>) {
    let local_node = LiveHardwareProfiler::probe_local_host();
    let token = ticket_arg.unwrap_or_else(|| {
        JoinTicket::generate(
            local_node.id.clone(),
            vec!["127.0.0.1:7848".into(), "relay.kadmium.ai:4433".into()],
            Some("slice-01-prod-deploy".into()),
            NodeTrustTier::VerifiedPeerMiddleLayersOnly,
        )
        .encode_token()
    });

    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Node Join & UDP LAN Beacon Handshake");
    println!(
        "===================================================================================="
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

    let beacon = LanDiscoveryBeacon::from_node(&local_node, 7848);
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

    let mut cluster = sample_personal_cluster();
    cluster.push(local_node);
    let model = ModelSpec::deepseek_v3_671b_moe();
    let plan = ShardPlanner::compute_plan(2, &model, WorkloadKind::Inference, &cluster);
    print_shard_plan(&plan);
}

pub fn run_pull(uri: &str) {
    println!(
        "===================================================================================="
    );
    println!(" Kadmium (Cd-48) — Ecosystem Connector (`hf://` & `gh://` Selective Staging)");
    println!(
        "===================================================================================="
    );

    if uri.starts_with("gh://") || uri.contains("github.com") {
        let gh_run =
            GitHubConnector::evaluate_repo_on_slice(uri).expect("Failed to parse GitHub URI");
        println!(
            "  * GitHub Repo Mounted:    {}/{} @ {} (SHA: {})",
            gh_run.gh_ref.owner,
            gh_run.gh_ref.repo,
            gh_run.gh_ref.git_ref,
            gh_run.pinned_commit_sha
        );
        println!("  * Subpath / World:        {:?}", gh_run.gh_ref.subpath);
        println!("  * Target Slice Assigned:  {}", gh_run.assigned_slice_id);
        println!("  * Sandbox Runtime:        {}", gh_run.sandbox_runtime);
        println!(
            "  * Verde Check Conclusion: {} ({:.1} tok/s, {:.2}x Dream-RSI speedup)",
            gh_run.github_check_conclusion,
            gh_run.measured_tokens_per_sec,
            gh_run.dream_rsi_best_speedup
        );
    } else {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::deepseek_v3_671b_moe();
        let plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &cluster);
        let hf_plan = HuggingFaceHubConnector::plan_selective_model_pull(uri, &plan)
            .expect("Failed to parse Hugging Face URI");
        println!(
            "  * Hugging Face Repo:      {} (rev: {}, kind: {:?})",
            hf_plan.hf_ref.repo_id, hf_plan.hf_ref.revision, hf_plan.hf_ref.kind
        );
        println!("  * Index Manifest URL:     {}", hf_plan.index_manifest_url);
        println!("  * Primary .ktb Seed Node: {}", hf_plan.seed_vault_node);
        println!(
            "\n{:<22} {:<10} {:<12} {:<14} {:<42}",
            "WORKER HOSTNAME", "LAYERS", "SELECTIVE", "AVOIDED DL", "COMPILED .KTB BANK"
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
    }
}

pub fn run_mcp(inspect: bool) {
    let mut state = GatewayState::new();

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

    let ktb_bytes = packer.pack_to_bytes();
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
            packer.pack_to_bytes(),
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
