//! Isolated loopback fixture for Control Room regression tests; no live workers or discovery.
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
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
    println!(
        "http://{}{}",
        bound_addr,
        kadmium_gateway::condenser_access_path("/")
    );

    let mut gateway_state = kadmium_gateway::GatewayState::new();
    gateway_state.coordinator_lan_ip = "127.0.0.1".to_string();
    gateway_state.coordinator_http_port = bound_addr.port();
    if let Ok(state_dir) = std::env::var("KADMIUM_STATE_DIR") {
        gateway_state.persistent_state_dir = std::path::PathBuf::from(state_dir);
    }

    let state = Arc::new(Mutex::new(gateway_state));
    axum::serve(
        listener,
        kadmium_gateway::build_router(state)
            .into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
    .unwrap();
}
