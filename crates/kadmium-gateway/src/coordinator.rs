//! Local coordinator bootstrap shared by `kadmium serve`, `kadmium desktop` and the
//! native `kadmium-desktop` window.
//!
//! It probes this machine, starts the coordinator's own `WireFrame` worker, attaches any
//! live LAN/USB silicon bridges it finds, answers LAN discovery beacons, keeps worker
//! health fresh, and serves the Control Room, Companion and AI gateway over HTTP.

use crate::{GatewayState, build_router};
use kadmium_core::{KadmiumWorkspaceConfig, LiveHardwareProfiler};
use kadmium_network::{
    DEFAULT_LAN_DISCOVERY_UDP_PORT, LanCoordinatorAnnouncement, LanCoordinatorDiscovery,
};
use kadmium_runtime::{DistributedLanExecutor, LiveNodeWorkerServer, LocalFleetSpawner};
use std::sync::{Arc, Mutex};

/// How to start a local coordinator.
#[derive(Debug, Clone)]
pub struct CoordinatorOptions {
    /// HTTP port for the Control Room, Companion and AI gateway (default 7848).
    pub port: u16,
    /// Optional `kadmium.toml` workspace configuration.
    pub config_path: Option<String>,
    /// Register only real probed hardware (no simulated demo fleet).
    pub live_only: bool,
    /// TCP port for the coordinator's built-in `WireFrame` stage worker (default 7850).
    pub worker_port: u16,
    /// Spawn a 5-node local + online TCP test fleet for single-machine testing.
    pub spawn_fleet: bool,
}

impl Default for CoordinatorOptions {
    fn default() -> Self {
        Self {
            port: 7848,
            config_path: None,
            live_only: true,
            worker_port: kadmium_network::DEFAULT_WORKER_TCP_PORT,
            spawn_fleet: false,
        }
    }
}

/// Starts the coordinator and serves until the HTTP server stops.
///
/// Returns an error instead of panicking when a port is already taken, so a caller such as
/// the desktop window can explain the problem to the person instead of crashing.
pub async fn run_coordinator(options: CoordinatorOptions) -> Result<(), String> {
    let CoordinatorOptions {
        port,
        config_path,
        live_only,
        worker_port,
        spawn_fleet,
    } = options;
    let config_path = config_path.as_deref();
    let mut gw_state = GatewayState::new();
    gw_state.coordinator_http_port = port;
    match gw_state.load_grid_telemetry() {
        Ok(n) if n > 0 => println!(
            "  [Grid] Restored measured telemetry for {n} workers from {}",
            GatewayState::grid_telemetry_path().display()
        ),
        Ok(_) => {}
        Err(e) => println!("  [Grid] Saved worker telemetry could not be read: {e}"),
    }
    match gw_state.load_grid_usage() {
        Ok(n) if n > 0 => println!(
            "  [Grid] Restored learned routing heat for {n} models from {}",
            GatewayState::grid_usage_dir().display()
        ),
        Ok(_) => {}
        Err(e) => println!("  [Grid] Saved usage ledgers could not be read: {e}"),
    }
    match gw_state.load_grid_calibrations() {
        Ok(n) if n > 0 => println!(
            "  [Grid] Restored {n} calibration records from {}",
            GatewayState::grid_calibrations_path().display()
        ),
        Ok(_) => {}
        Err(e) => println!("  [Grid] Saved calibration records could not be read: {e}"),
    }
    // GatewayState::new() assumes the default port; point the self-hosted URL at the real one.
    if let crate::CompanionConnectionMode::SelfHostedOpenSource {
        local_coordinator_url,
    } = &mut gw_state.companion_mode
    {
        *local_coordinator_url = format!("http://{}:{port}", gw_state.coordinator_lan_ip);
    }

    if let Some(cfg_file) = config_path
        && let Ok(raw_toml) = std::fs::read_to_string(cfg_file)
        && let Ok(parsed_cfg) = KadmiumWorkspaceConfig::parse_kadmium_toml(&raw_toml)
    {
        if parsed_cfg.cloud_shield_enabled {
            gw_state.companion_mode = crate::CompanionConnectionMode::KadmiumCloudManaged {
                workspace_slug: parsed_cfg.cluster_name.clone(),
                cloud_endpoint: "https://api.kadmium.ai".into(),
                downtime_shield_enabled: true,
                spot_arbitrage_burst_enabled: true,
                max_hourly_spend_cap_usd: parsed_cfg.cloud_hourly_cap_usd,
            };
        }
        println!(
            "  [Config] Loaded workspace '{}' from {}",
            parsed_cfg.cluster_name, cfg_file
        );
    }

    let local_node = LiveHardwareProfiler::probe_local_host();
    let lan_ip = gw_state.coordinator_lan_ip.clone();

    // Spawn built-in TCP WireFrame worker server for the coordinator's own hardware:
    let local_worker =
        LiveNodeWorkerServer::spawn_auto_port("0.0.0.0", worker_port, local_node.clone(), true)
            .await
            .map_err(|e| {
                format!("could not start the local worker listener on port {worker_port}: {e}")
            })?;
    let bound_worker_port = local_worker.local_addr.port();
    let worker_tcp_addr = format!("{lan_ip}:{bound_worker_port}");

    gw_state.register_live_worker(local_node.clone(), worker_tcp_addr.clone(), 0.15, live_only);
    let local_stream_gbps = LiveHardwareProfiler::probe_memory_stream_gbps();
    gw_state.set_worker_offer(&local_node.id.0, None, None, Some(local_stream_gbps));
    println!(
        "  [Grid] Measured memory stream on this machine: {local_stream_gbps:.1} GB/s (single thread, feeds the cost model)"
    );

    // Automatically detect and attach a physical CanMV K230 RISC-V + 6 TOPS KPU board over USB-CDC if connected:
    let k230_probe = kadmium_core::K230UsbComputeBridge::probe_attached_k230();
    let mut attached_k230_banner: Option<String> = None;
    let mut next_fleet_port = bound_worker_port.saturating_add(1);
    if k230_probe.physical_board_detected
        && let Ok(k230_server) = LiveNodeWorkerServer::spawn_auto_port(
            "127.0.0.1",
            next_fleet_port,
            k230_probe.hardware_node.clone(),
            true,
        )
        .await
    {
        next_fleet_port = k230_server.local_addr.port().saturating_add(1);
        let k230_tcp_addr = k230_server.local_addr.to_string();
        let port_label = k230_probe
            .detected_serial_port
            .clone()
            .unwrap_or_else(|| "USB-BootROM".to_string());
        let (usb_rtt, os_label, live_silicon, k230_digest) =
            if let Some(proof) = &k230_probe.live_execution_proof {
                (
                    proof.usb_round_trip_latency_ms.max(0.85),
                    proof.riscv_core_signature.clone(),
                    proof.executed_on_physical_silicon,
                    proof.k230_hardware_checksum_fnv1a,
                )
            } else {
                (
                    0.85,
                    "rt-smart v1.8 (CanMV K230)".to_string(),
                    false,
                    0x627e_7e5b_5419_aca7,
                )
            };
        gw_state.register_live_worker_with_meta(
            k230_probe.hardware_node.clone(),
            k230_tcp_addr.clone(),
            usb_rtt,
            live_only,
            os_label.clone(),
            "riscv64-c908-rvv1.0 (6 TOPS KPU)".to_string(),
            format!(
                "{} [{}]",
                k230_probe.board_model, k230_probe.chip_unique_die_id
            ),
            format!("USB-CDC ({port_label})"),
        );
        gw_state
            .spawned_fleet_instances
            .push(kadmium_runtime::SpawnedLocalWorkerInstance {
                server: k230_server,
                os_name: os_label,
                cpu_arch: "riscv64-c908-rvv1.0 (6 TOPS KPU)".to_string(),
                board_or_model: format!(
                    "{} [{}]",
                    k230_probe.board_model, k230_probe.chip_unique_die_id
                ),
                network_channel: format!("USB-CDC ({port_label})"),
            });
        attached_k230_banner = Some(format!(
            "{} -> {} ({}, {}, RepOps={:#018x})",
            port_label,
            k230_tcp_addr,
            k230_probe.board_model,
            if live_silicon {
                "LIVE RISC-V SILICON"
            } else {
                "USB-BRIDGE"
            },
            k230_digest
        ));
    }

    // Automatically detect and attach a live Apple M4 platform node (192.168.40.237:7860) and Raspberry Pi CM4 node (192.168.40.75:7859) on the LAN:
    let mut attached_m4_banner: Option<String> = None;
    let m4_probe = kadmium_core::M4MacMiniLanBridge::probe_and_verify(None);
    if m4_probe.physical_board_online
        && let Ok(m4_server) = LiveNodeWorkerServer::spawn_auto_port(
            "127.0.0.1",
            next_fleet_port,
            m4_probe.hardware_node.clone(),
            true,
        )
        .await
    {
        next_fleet_port = m4_server.local_addr.port().saturating_add(1);
        let m4_tcp_addr = m4_server.local_addr.to_string();
        let os_label = "macOS 26 (Darwin 25.0.0)".to_string();
        let arch_label = "arm64e (Apple M4 · Metal 4 + AMX)".to_string();
        let board_label = m4_probe.board_model.clone();
        gw_state.register_live_worker_with_meta(
            m4_probe.hardware_node.clone(),
            m4_tcp_addr.clone(),
            m4_probe.execution_proof.lan_round_trip_ms.max(0.25),
            live_only,
            os_label.clone(),
            arch_label.clone(),
            board_label.clone(),
            "LocalLAN".to_string(),
        );
        if let Some(w) = gw_state.live_workers.get_mut(&m4_probe.hardware_node.id.0) {
            w.worker_lan_ip = kadmium_core::DEFAULT_M4_MAC_MINI_IP.to_string();
            w.worker_tcp_port = kadmium_core::DEFAULT_M4_MAC_MINI_PORT;
        }
        gw_state
            .spawned_fleet_instances
            .push(kadmium_runtime::SpawnedLocalWorkerInstance {
                server: m4_server,
                os_name: os_label,
                cpu_arch: arch_label,
                board_or_model: board_label,
                network_channel: "LocalLAN".to_string(),
            });
        attached_m4_banner = Some(format!(
            "{} -> {} ({}, LIVE METAL 4 + AMX {:.1} GFLOP/s, RepOps={:#018x})",
            m4_probe.lan_endpoint,
            m4_tcp_addr,
            m4_probe.board_model,
            m4_probe.execution_proof.amx_sgemm_512_gflops,
            m4_probe.execution_proof.metal_gpu_gemm_fnv1a64
        ));
    }

    let mut attached_cm4_banner: Option<String> = None;
    let cm4_probe = kadmium_core::ReTerminalCm4LanBridge::probe_and_verify(None);
    if cm4_probe.physical_board_online
        && let Ok(cm4_server) = LiveNodeWorkerServer::spawn_auto_port(
            "127.0.0.1",
            next_fleet_port,
            cm4_probe.hardware_node.clone(),
            true,
        )
        .await
    {
        next_fleet_port = cm4_server.local_addr.port().saturating_add(1);
        let cm4_tcp_addr = cm4_server.local_addr.to_string();
        let os_label = "Linux 6.12 (Debian Bookworm)".to_string();
        let arch_label = "aarch64 (BCM2711 Cortex-A72 NEON)".to_string();
        let board_label = cm4_probe.board_model.clone();
        gw_state.register_live_worker_with_meta(
            cm4_probe.hardware_node.clone(),
            cm4_tcp_addr.clone(),
            cm4_probe.execution_proof.lan_round_trip_ms.max(0.25),
            live_only,
            os_label.clone(),
            arch_label.clone(),
            board_label.clone(),
            "LocalLAN".to_string(),
        );
        if let Some(w) = gw_state.live_workers.get_mut(&cm4_probe.hardware_node.id.0) {
            w.worker_lan_ip = kadmium_core::DEFAULT_RETERMINAL_CM4_IP.to_string();
            w.worker_tcp_port = kadmium_core::DEFAULT_RETERMINAL_CM4_PORT;
        }
        gw_state
            .spawned_fleet_instances
            .push(kadmium_runtime::SpawnedLocalWorkerInstance {
                server: cm4_server,
                os_name: os_label,
                cpu_arch: arch_label,
                board_or_model: board_label,
                network_channel: "LocalLAN".to_string(),
            });
        attached_cm4_banner = Some(format!(
            "{} -> {} ({}, LIVE ARMv8 NEON, RepOps={:#018x})",
            cm4_probe.lan_endpoint,
            cm4_tcp_addr,
            cm4_probe.board_model,
            cm4_probe.execution_proof.neon_gemm_fnv1a64
        ));
    }

    if spawn_fleet
        && let Ok(instances) =
            LocalFleetSpawner::spawn_heterogeneous_local_fleet(next_fleet_port, true).await
    {
        for inst in instances {
            let tcp_addr = inst.server.local_addr.to_string();
            let node_clone = inst.server.node.lock().expect("node lock").clone();
            let rtt = match DistributedLanExecutor::ping_worker_tcp(&tcp_addr, 1).await {
                Ok((_, ms)) => ms,
                Err(_) => 0.25,
            };
            gw_state.register_live_worker_with_meta(
                node_clone,
                tcp_addr,
                rtt,
                true,
                inst.os_name.clone(),
                inst.cpu_arch.clone(),
                inst.board_or_model.clone(),
                inst.network_channel.clone(),
            );
            gw_state.spawned_fleet_instances.push(inst);
        }
        gw_state.set_live_only_mode(true);
    }

    // Measure initial TCP WireFrame RTT for registered live workers without running synthetic warmup jobs
    if !gw_state.live_workers.is_empty() {
        let worker_addrs: Vec<(String, String)> = gw_state
            .live_workers
            .iter()
            .map(|(id, w)| (id.clone(), w.worker_tcp_addr.clone()))
            .collect();
        for (node_id, w_addr) in worker_addrs {
            if let Ok((_, rtt_ms)) = DistributedLanExecutor::ping_worker_tcp(&w_addr, 1).await
                && let Some(w) = gw_state.live_workers.get_mut(&node_id)
            {
                w.last_measured_rtt_ms = rtt_ms;
            }
        }
    }

    let overview = gw_state.workers_overview();
    let announcement = LanCoordinatorAnnouncement {
        coordinator_id: gw_state.coordinator_quorum.active_leader.clone(),
        coordinator_hostname: local_node.hostname.clone(),
        coordinator_http_url: overview.coordinator_http_url.clone(),
        join_ticket_token: overview.join_ticket_token.clone(),
        active_model_id: gw_state.inference_plan.model_id.clone(),
        cluster_epoch: gw_state.epoch,
    };
    let udp_disc_status = match LanCoordinatorDiscovery::spawn_responder(
        DEFAULT_LAN_DISCOVERY_UDP_PORT,
        announcement,
    )
    .await
    {
        Ok(addr) => format!("Active on UDP {addr} (KDMD auto-discovery)"),
        Err(_) => "Port 7849 busy (use --coordinator URL on workers)".to_string(),
    };

    let state = Arc::new(Mutex::new(gw_state));
    let health_state = Arc::clone(&state);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(12));
        loop {
            interval.tick().await;
            let (epoch, targets) = {
                let guard = health_state.lock().expect("gateway state lock");
                let list: Vec<(String, String)> = guard
                    .live_workers
                    .values()
                    .filter(|w| w.online)
                    .map(|w| (w.node.id.0.clone(), w.worker_tcp_addr.clone()))
                    .collect();
                (guard.epoch, list)
            };
            for (nid, tcp_addr) in targets {
                match DistributedLanExecutor::ping_worker_tcp(&tcp_addr, epoch).await {
                    Ok((pong, rtt)) => {
                        let mut guard = health_state.lock().expect("gateway state lock");
                        if let Some(w) = guard.live_workers.get_mut(&nid) {
                            w.last_measured_rtt_ms = rtt;
                            w.installed_version = pong.installed_version;
                            w.ktb_merkle_root_fnv64 = pong.ktb_merkle_root_fnv64;
                        }
                    }
                    Err(_) => {
                        let mut guard = health_state.lock().expect("gateway state lock");
                        guard.set_worker_fault_or_polite_state(
                            &nid,
                            kadmium_core::PoliteState::IdleDedicated,
                            false,
                        );
                    }
                }
            }
        }
    });

    let app = build_router(state);
    let addr = format!("0.0.0.0:{}", port);
    println!(
        "===================================================================================="
    );
    println!(
        " Kadmium (Cd-48) Control Room & AI Gateway listening on http://127.0.0.1:{}",
        port
    );
    println!("    - LAN Coordinator URL:    http://{}:{}/", lan_ip, port);
    println!("    - Cluster Control Room:   http://127.0.0.1:{}/", port);
    println!(
        "    - kadmium.ai Website:     http://127.0.0.1:{}/site",
        port
    );
    println!(
        "    - Desktop Companion UI:   http://127.0.0.1:{}/companion",
        port
    );
    println!(
        "    - Application Docs:       http://127.0.0.1:{}/docs",
        port
    );
    println!(
        "    - Design System Showcase: http://127.0.0.1:{}/design-system",
        port
    );
    println!(
        "    - OpenAI Chat Endpoint:   http://127.0.0.1:{}/v1/chat/completions",
        port
    );
    println!(
        "    - Anthropic Endpoint:     http://127.0.0.1:{}/v1/messages",
        port
    );
    println!(
        "    - MCP JSON-RPC Endpoint:  http://127.0.0.1:{}/mcp",
        port
    );
    println!(
        "    - Prometheus Metrics:     http://127.0.0.1:{}/metrics",
        port
    );
    println!(
        "------------------------------------------------------------------------------------"
    );
    println!(
        "    - Local WireFrame Worker: {} ({:?}, {:.1} GB fast mem)",
        worker_tcp_addr,
        local_node.device_kind,
        local_node.available_fast_memory_gb()
    );
    if let Some(m4_line) = &attached_m4_banner {
        println!("    - LAN Apple M4 Worker:    {}", m4_line);
    }
    if let Some(cm4_line) = &attached_cm4_banner {
        println!("    - LAN RPi CM4 Worker:     {}", cm4_line);
    }
    if let Some(k230_line) = &attached_k230_banner {
        println!("    - USB RISC-V K230 Worker: {}", k230_line);
    }
    println!("    - LAN UDP Auto-Discovery: {}", udp_disc_status);
    println!(
        "    - Connect any PC/Mac/SBC: {}",
        overview.join_command_cli
    );
    println!(
        "    - Linux/macOS/SBC 1-Line: curl -sSL http://{}:{}/join.sh | sh",
        lan_ip, port
    );
    println!(
        "    - Windows 1-Line Join:    irm http://{}:{}/join.ps1 | iex",
        lan_ip, port
    );
    println!(
        "===================================================================================="
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("could not bind the coordinator on {addr}: {e}"))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| format!("coordinator stopped: {e}"))
}
