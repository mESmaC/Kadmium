use kadmium_condenser::plugin::default_plugin_roots;
use kadmium_condenser::server::{self, ServerOptions};
use kadmium_condenser::{CondenserError, Result};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

#[cfg(any(not(target_os = "linux"), feature = "linux-native"))]
mod icon;
#[cfg(any(not(target_os = "linux"), feature = "linux-native"))]
mod native;

#[derive(Debug, Clone)]
pub struct UiOptions {
    /// An available port is selected when absent; zero has the same behavior.
    pub port: Option<u16>,
    pub browser: bool,
    pub plugin_roots: Vec<PathBuf>,
}

impl Default for UiOptions {
    fn default() -> Self {
        Self {
            port: None,
            browser: false,
            plugin_roots: default_plugin_roots(),
        }
    }
}

struct EmbeddedServer {
    url: String,
    stopped: Receiver<String>,
}

pub fn run(options: UiOptions) -> Result<()> {
    let server = start_embedded(&options)?;
    #[cfg(any(not(target_os = "linux"), feature = "linux-native"))]
    {
        if !options.browser {
            native::run(options, server);
            return Ok(());
        }
    }
    run_in_browser(server)
}

fn start_embedded(options: &UiOptions) -> Result<EmbeddedServer> {
    let server_options = ServerOptions {
        port: options.port.unwrap_or(0),
        plugin_roots: options.plugin_roots.clone(),
        ..ServerOptions::default()
    };
    let (ready_tx, ready_rx) = channel();
    let (stopped_tx, stopped_rx) = channel();
    std::thread::Builder::new()
        .name("condenser-server".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let message = format!("could not start the async runtime: {error}");
                    let _ = ready_tx.send(Err(message.clone()));
                    let _ = stopped_tx.send(message);
                    return;
                }
            };
            runtime.block_on(async move {
                let (address, future) = match server::bind(server_options).await {
                    Ok(bound) => bound,
                    Err(error) => {
                        let message = error.to_string();
                        let _ = ready_tx.send(Err(message.clone()));
                        let _ = stopped_tx.send(message);
                        return;
                    }
                };
                let url = format!("http://{address}/");
                let _ = ready_tx.send(Ok(url));
                let message = match future.await {
                    Ok(()) => "the Condenser service stopped".into(),
                    Err(error) => error.to_string(),
                };
                let _ = stopped_tx.send(message);
            });
        })
        .map_err(|error| {
            CondenserError::Config(format!("could not start server thread: {error}"))
        })?;
    let url = ready_rx
        .recv_timeout(Duration::from_secs(15))
        .map_err(|_| CondenserError::Config("Condenser service did not become ready".into()))?
        .map_err(CondenserError::Config)?;
    Ok(EmbeddedServer {
        url,
        stopped: stopped_rx,
    })
}

fn run_in_browser(server: EmbeddedServer) -> Result<()> {
    if cfg!(target_os = "linux") && !cfg!(feature = "linux-native") {
        eprintln!(
            "This Condenser build uses the browser shell. Enable `linux-native` with WebKitGTK 4.1 and GTK 3 for a native window."
        );
    }
    println!("Condenser is ready at {}", server.url);
    open_in_browser(&server.url);
    // Closing the terminal remains the operator-facing stop action. If the
    // service fails independently, surface its actual error.
    let message = server
        .stopped
        .recv()
        .unwrap_or_else(|_| "the Condenser service stopped".into());
    Err(CondenserError::Config(message))
}

/// Opens only ordinary web/mail links in the system browser. Page content
/// cannot use this bridge to launch arbitrary programs or local files.
pub fn open_in_browser(url: &str) {
    let lower = url.to_ascii_lowercase();
    if !["http://", "https://", "mailto:"]
        .iter()
        .any(|scheme| lower.starts_with(scheme))
    {
        return;
    }
    spawn_browser(url);
}

#[cfg(windows)]
fn spawn_browser(url: &str) {
    let _ = std::process::Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn();
}

#[cfg(target_os = "macos")]
fn spawn_browser(url: &str) {
    let _ = std::process::Command::new("open").arg(url).spawn();
}

#[cfg(all(unix, not(target_os = "macos")))]
fn spawn_browser(url: &str) {
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

#[cfg(any(not(target_os = "linux"), feature = "linux-native"))]
fn webview_data_dir() -> Option<PathBuf> {
    Some(kadmium_condenser::plugin::condenser_home().join("webview"))
}
