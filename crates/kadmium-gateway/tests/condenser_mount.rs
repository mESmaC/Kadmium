use axum::{
    Router,
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode, header},
};
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tower::ServiceExt;

fn request(path: &str, peer: Option<&str>, cookie: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .uri(path)
        .header("host", "127.0.0.1:7848");
    if let Some(cookie) = cookie {
        builder = builder.header(header::COOKIE, cookie);
    }
    if let Some(peer) = peer {
        builder = builder.extension(ConnectInfo(peer.parse::<SocketAddr>().unwrap()));
    }
    builder.body(Body::empty()).unwrap()
}

async fn status(app: &Router, request: Request<Body>) -> StatusCode {
    app.clone().oneshot(request).await.unwrap().status()
}

#[tokio::test]
async fn mounted_workspace_requires_the_launch_secret_and_a_local_same_origin_peer() {
    let app =
        kadmium_gateway::build_router(Arc::new(Mutex::new(kadmium_gateway::GatewayState::new())));
    let local = Some("127.0.0.1:1234");
    let paths = [
        "/condenser/",
        "/condenser/condenser.js",
        "/condenser/api/v1/health",
        "/condenser/api/v1/checkpoints",
    ];

    // A loopback peer alone (another local user, a Host-rewriting proxy) is not enough.
    for path in paths {
        assert_eq!(
            status(&app, request(path, local, None)).await,
            StatusCode::UNAUTHORIZED,
            "{path}"
        );
    }
    assert_eq!(
        status(
            &app,
            request(
                "/condenser/api/v1/health",
                local,
                Some("kadmium_condenser=guess")
            )
        )
        .await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        status(
            &app,
            request(
                &format!("/condenser/access?token={}", "0".repeat(64)),
                local,
                None
            )
        )
        .await,
        StatusCode::UNAUTHORIZED
    );

    let redeemed = app
        .clone()
        .oneshot(request(
            &kadmium_gateway::condenser_access_path("/companion"),
            local,
            None,
        ))
        .await
        .unwrap();
    assert_eq!(redeemed.status(), StatusCode::SEE_OTHER);
    assert_eq!(redeemed.headers()[header::LOCATION], "/companion");
    let set_cookie = redeemed.headers()[header::SET_COOKIE].to_str().unwrap();
    assert!(set_cookie.contains("HttpOnly") && set_cookie.contains("SameSite=Strict"));
    let cookie = set_cookie.split(';').next().unwrap().to_owned();

    // Only same-origin paths are followed after redemption.
    let open_redirect = app
        .clone()
        .oneshot(request(
            &kadmium_gateway::condenser_access_path("//evil.example/"),
            local,
            None,
        ))
        .await
        .unwrap();
    assert_eq!(open_redirect.headers()[header::LOCATION], "/condenser/");

    for path in paths {
        assert_eq!(
            status(&app, request(path, local, Some(&cookie))).await,
            StatusCode::OK,
            "{path}"
        );
    }

    // The embedded workspace has an explicit per-user root, never the process cwd.
    let workspace = app
        .clone()
        .oneshot(request("/condenser/api/v1/workspace", local, Some(&cookie)))
        .await
        .unwrap();
    let body = axum::body::to_bytes(workspace.into_body(), usize::MAX)
        .await
        .unwrap();
    let root = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["root"]
        .as_str()
        .unwrap()
        .to_owned();
    let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
    assert_ne!(std::path::Path::new(&root), cwd.as_path());
    if std::env::var_os("KADMIUM_CONDENSER_WORKSPACE").is_none() {
        assert!(root.ends_with("condenser-workspace"), "{root}");
    }

    for (peer, origin) in [
        ("192.0.2.1:1234", "http://127.0.0.1:7848"),
        ("127.0.0.1:1234", "http://localhost:9999"),
    ] {
        let mut cross = request("/condenser/api/v1/checkpoints", Some(peer), Some(&cookie));
        cross
            .headers_mut()
            .insert(header::ORIGIN, origin.parse().unwrap());
        assert_eq!(status(&app, cross).await, StatusCode::FORBIDDEN);
    }
    assert_eq!(
        status(
            &app,
            request("/condenser/api/v1/health", None, Some(&cookie))
        )
        .await,
        StatusCode::FORBIDDEN
    );
}
