use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
};
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tower::ServiceExt;

#[tokio::test]
async fn mounted_workspace_serves_assets_and_blocks_remote_or_cross_origin_access() {
    let app =
        kadmium_gateway::build_router(Arc::new(Mutex::new(kadmium_gateway::GatewayState::new())));
    for path in [
        "/condenser/",
        "/condenser/condenser.js",
        "/condenser/api/v1/health",
        "/condenser/api/v1/checkpoints",
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(path)
                    .header("host", "127.0.0.1:7848")
                    .extension(ConnectInfo("127.0.0.1:1234".parse::<SocketAddr>().unwrap()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{path}");
    }
    for (peer, origin) in [
        ("192.0.2.1:1234", "http://127.0.0.1:7848"),
        ("127.0.0.1:1234", "http://localhost:9999"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/condenser/api/v1/checkpoints")
                    .header("host", "127.0.0.1:7848")
                    .header("origin", origin)
                    .extension(ConnectInfo(peer.parse::<SocketAddr>().unwrap()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }
    let response = app
        .oneshot(
            Request::builder()
                .uri("/condenser/api/v1/health")
                .header("host", "127.0.0.1:7848")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}
