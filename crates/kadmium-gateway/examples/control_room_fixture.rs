//! Isolated loopback fixture for Control Room regression tests; no live workers or discovery.
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    println!(
        "http://{}{}",
        listener.local_addr().unwrap(),
        kadmium_gateway::condenser_access_path("/")
    );
    let state = Arc::new(Mutex::new(kadmium_gateway::GatewayState::new()));
    axum::serve(
        listener,
        kadmium_gateway::build_router(state)
            .into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
    .unwrap();
}
