//! The embedded Condenser workspace at `/condenser/`.
//!
//! The coordinator listens on the LAN and shares its origin with the Control Room, so the
//! mount needs more than a loopback check: a same-host process, another local user or a
//! Host-rewriting reverse proxy all reach it from a loopback peer. Every request must carry
//! this launch's secret, which the owner receives out of band (the console banner, a
//! user-only file, or the desktop window). Opening the access link stores the secret in an
//! HttpOnly cookie. The link is reusable, not single-use: it stays valid until the
//! coordinator restarts, so treat it like a password for this launch.
use axum::{
    Router,
    extract::{ConnectInfo, Query, Request},
    http::{StatusCode, header},
    middleware::Next,
    response::{Html, IntoResponse, Redirect, Response},
    routing::get,
};
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::OnceLock;

const COOKIE: &str = "kadmium_condenser";

/// Per-process secret for the embedded workspace. Regenerated on every launch.
pub fn condenser_access_token() -> &'static str {
    static TOKEN: OnceLock<String> = OnceLock::new();
    TOKEN.get_or_init(|| {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).expect("the OS random number generator is unavailable");
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    })
}

fn access_path(token: &str, next: &str) -> String {
    let next: String = next
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                char::from(b).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect();
    format!("/condenser/access?token={token}&next={next}")
}

/// Reusable access link for this launch: stores the access cookie, then continues to `next`.
pub fn condenser_access_path(next: &str) -> String {
    access_path(condenser_access_token(), next)
}

fn access_file(port: u16) -> Result<PathBuf, String> {
    Ok(crate::coordinator::coordinator_journal_directory()?
        .join(format!("condenser-access-{port}")))
}

/// Saves this launch's access link where only the owner's account can read it, so the
/// desktop app (or the owner) can open the workspace of an already-running coordinator.
pub fn publish_condenser_access(port: u16) -> Result<PathBuf, String> {
    use std::io::Write;
    let path = access_file(port)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let _ = std::fs::remove_file(&path);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(&path).map_err(|e| e.to_string())?;
    file.write_all(condenser_access_token().as_bytes())
        .map_err(|e| e.to_string())?;
    Ok(path)
}

/// The access link for the coordinator on `port`, if it published one for this user.
pub fn condenser_access_url(port: u16, next: &str) -> Option<String> {
    let token = std::fs::read_to_string(access_file(port).ok()?).ok()?;
    let token = token.trim();
    (token.len() == 64 && token.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| format!("http://127.0.0.1:{port}{}", access_path(token, next)))
}

/// Explicit per-user workspace root for the embedded agent's files, terminal and plugins.
/// Never the coordinator's working directory, which desktop launchers may set to `/`.
pub fn condenser_workspace_root() -> Result<PathBuf, String> {
    let root = match std::env::var_os("KADMIUM_CONDENSER_WORKSPACE") {
        Some(path) => {
            let path = PathBuf::from(path);
            if !path.is_absolute() {
                return Err(
                    "KADMIUM_CONDENSER_WORKSPACE must be an absolute directory path".into(),
                );
            }
            path
        }
        None => crate::coordinator::coordinator_journal_directory()?.join("condenser-workspace"),
    };
    std::fs::create_dir_all(&root).map_err(|e| {
        format!(
            "cannot create the Condenser workspace {}: {e}",
            root.display()
        )
    })?;
    Ok(root)
}

fn equal(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

fn cookie_token(request: &Request) -> Option<&str> {
    request
        .headers()
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .find_map(|pair| pair.trim().strip_prefix(COOKIE)?.strip_prefix('='))
}

#[derive(Deserialize)]
struct Access {
    token: String,
    next: Option<String>,
}

async fn redeem(Query(access): Query<Access>) -> Response {
    if !equal(&access.token, condenser_access_token()) {
        return denied();
    }
    // Only same-origin paths: never an open redirect.
    let next = access
        .next
        .filter(|next| next.starts_with('/') && !next.starts_with("//") && !next.contains('\\'))
        .unwrap_or_else(|| "/condenser/".into());
    (
        [(
            header::SET_COOKIE,
            format!(
                "{COOKIE}={}; Path=/condenser; HttpOnly; SameSite=Strict",
                condenser_access_token()
            ),
        )],
        Redirect::to(&next),
    )
        .into_response()
}

fn denied() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::CACHE_CONTROL, "no-store")],
        Html(
            "<!doctype html><meta charset=utf-8><title>Condenser access</title>\
             <h1>Open Condenser with this launch's access link</h1>\
             <p>Condenser can read and write files and run commands on this computer, so each \
             launch has its own access link. Use the <code>Condenser workspace</code> link printed \
             by <code>kadmium serve</code>, open Condenser from the Kadmium desktop app, or read the \
             link saved for your account in the Kadmium state directory \
             (<code>condenser-access-&lt;port&gt;</code>).</p>",
        ),
    )
        .into_response()
}

/// A same-origin request from a loopback peer. Necessary for the embedded workspace but not
/// sufficient: a local reverse proxy or another local user also passes it, so anything beyond
/// read-only data also needs a secret (this launch's access cookie, or a bearer token).
pub(crate) fn local_browser_request(request: &Request) -> bool {
    let local = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .is_some_and(|peer| peer.0.ip().is_loopback());
    // A rebound DNS name reaches a loopback listener with its own Host; accept loopback names
    // only.
    let loopback_host = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(|host| {
            host.rsplit_once(':')
                .filter(|(_, port)| port.chars().all(|c| c.is_ascii_digit()))
                .map_or(host, |(name, _)| name)
        })
        .is_some_and(|name| matches!(name, "127.0.0.1" | "localhost" | "[::1]"));
    let same_origin = request.headers().get(header::ORIGIN).is_none_or(|origin| {
        let expected = request
            .headers()
            .get(header::HOST)
            .and_then(|h| h.to_str().ok())
            .map(|host| format!("http://{host}"));
        origin
            .to_str()
            .ok()
            .zip(expected.as_deref())
            .is_some_and(|(origin, expected)| origin == expected)
    });
    local && loopback_host && same_origin
}

async fn guard(request: Request, next: Next) -> Response {
    if !local_browser_request(&request) {
        return (
            StatusCode::FORBIDDEN,
            "Open Condenser on the coordinator computer using http://127.0.0.1:<port>/condenser/",
        )
            .into_response();
    }
    if request.uri().path() == "/access" {
        return next.run(request).await;
    }
    if !cookie_token(&request).is_some_and(|token| equal(token, condenser_access_token())) {
        return denied();
    }
    next.run(request).await
}

/// Mounts Condenser with an explicit workspace root. A startup failure is served with
/// its reason (and logged) instead of a bare 503.
pub(crate) fn router() -> Router {
    let condenser = condenser_workspace_root()
        .map_err(kadmium_condenser::CondenserError::Config)
        .and_then(|root| {
            kadmium_condenser::server::build_router(kadmium_condenser::server::ServerOptions {
                plugin_roots: vec![
                    root.join("plugins"),
                    kadmium_condenser::plugin::condenser_home().join("plugins"),
                ],
                workspace_root: Some(root),
                ..Default::default()
            })
        });
    let condenser = condenser.unwrap_or_else(|error| {
        let message = format!("Condenser could not start: {error}");
        eprintln!("kadmium: {message}");
        Router::new().fallback(move || {
            let message = message.clone();
            async move { (StatusCode::SERVICE_UNAVAILABLE, message) }
        })
    });
    Router::new()
        .route(
            "/condenser",
            get(|| async { Redirect::permanent("/condenser/") }),
        )
        .nest(
            "/condenser/",
            Router::new()
                .route("/access", get(redeem))
                .merge(condenser)
                .layer(axum::middleware::from_fn(guard)),
        )
}
