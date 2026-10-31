//! Isolated loopback fixture for Control Room regression tests; no live workers or discovery.
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    // Ensure the UI test fixture never dials external LAN/USB/ADB hardware probes.
    unsafe {
        std::env::set_var("KADMIUM_DISABLE_LIVE_PROBES", "1");
    }

    let mut port: u16 = 0;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--port" && i + 1 < args.len() {
            port = args[i + 1].parse().unwrap_or(0);
            i += 2;
        } else {
            i += 1;
        }
    }

    let bind_addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    let bound_addr = listener.local_addr().unwrap();
    println!("http://{bound_addr}");

    let mut gateway_state = kadmium_gateway::GatewayState::new();
    gateway_state.coordinator_lan_ip = "127.0.0.1".to_string();
    gateway_state.coordinator_http_port = bound_addr.port();
    if let Ok(state_dir) = std::env::var("KADMIUM_STATE_DIR") {
        gateway_state.persistent_state_dir = std::path::PathBuf::from(state_dir);
    }

    let state = Arc::new(Mutex::new(gateway_state));
    axum::serve(listener, kadmium_gateway::build_router(state))
        .await
        .unwrap();
}
