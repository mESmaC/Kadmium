//! Platform-independent parts of the desktop app: arguments, finding or starting the local
//! coordinator, readiness checks, and opening pages in the default browser.

use kadmium_gateway::CoordinatorOptions;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::{Duration, Instant};

pub const USAGE: &str = "\
Kadmium desktop app: the Companion in its own window.

Usage: kadmium-desktop [OPTIONS]

Options:
  -p, --port <PORT>   Coordinator HTTP port to attach to or start on [default: 7848]
      --demo          Include the 7-node simulated demo fleet when starting a coordinator
      --control-room  Also open the Cluster Control Room in your browser
  -h, --help          Print help";

/// Command-line options. Parsed by hand so the desktop app stays free of a CLI framework.
#[derive(Debug, Clone)]
pub struct Args {
    pub port: u16,
    pub demo: bool,
    pub control_room: bool,
    pub help: bool,
}

impl Args {
    pub fn parse(mut iter: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut args = Args {
            port: 7848,
            demo: false,
            control_room: false,
            help: false,
        };
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "-h" | "--help" => args.help = true,
                "--demo" => args.demo = true,
                "--live" => args.demo = false,
                "--control-room" => args.control_room = true,
                "-p" | "--port" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--port needs a value".to_string())?;
                    args.port = parse_port(&value)?;
                }
                other => match other.strip_prefix("--port=") {
                    Some(value) => args.port = parse_port(value)?,
                    None => return Err(format!("unexpected argument '{other}'")),
                },
            }
        }
        Ok(args)
    }
}

fn parse_port(value: &str) -> Result<u16, String> {
    value
        .parse::<u16>()
        .ok()
        .filter(|port| *port != 0)
        .ok_or_else(|| format!("'{value}' is not a valid port"))
}

/// Where the coordinator behind the window comes from.
pub enum CoordinatorSource {
    /// A coordinator was already answering on the port (for example `kadmium serve`).
    Attached,
    /// This process started one; the receiver yields a message once it stops.
    Embedded { stopped: Receiver<String> },
}

impl CoordinatorSource {
    #[cfg_attr(
        all(target_os = "linux", not(feature = "linux-native")),
        allow(dead_code)
    )]
    pub fn is_embedded(&self) -> bool {
        matches!(self, CoordinatorSource::Embedded { .. })
    }
}

/// Attaches to a running coordinator on `args.port`, or starts one on a background thread.
pub fn ensure_coordinator(args: &Args) -> CoordinatorSource {
    if probe_coordinator(args.port).is_some() {
        return CoordinatorSource::Attached;
    }

    let (tx, stopped) = channel();
    let options = CoordinatorOptions {
        port: args.port,
        bind: Some(SocketAddr::from(([127, 0, 0, 1], args.port))),
        live_only: !args.demo,
        ..CoordinatorOptions::default()
    };
    let spawned = std::thread::Builder::new()
        .name("kadmium-coordinator".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    let _ = tx.send(format!("could not start the async runtime: {err}"));
                    return;
                }
            };
            let message = match runtime.block_on(kadmium_gateway::run_coordinator(options)) {
                Ok(()) => "the local coordinator stopped".to_string(),
                Err(err) => err,
            };
            let _ = tx.send(message);
        });
    if let Err(err) = spawned {
        let (tx, stopped) = channel();
        let _ = tx.send(format!("could not start the coordinator thread: {err}"));
        return CoordinatorSource::Embedded { stopped };
    }
    CoordinatorSource::Embedded { stopped }
}

/// Blocks until the coordinator answers, the embedded one fails, or `timeout` passes.
/// Returns the coordinator's version on success.
pub fn wait_until_ready(
    port: u16,
    source: &CoordinatorSource,
    timeout: Duration,
) -> Result<String, String> {
    let started = Instant::now();
    loop {
        if let Some(version) = probe_coordinator(port) {
            return Ok(version);
        }
        if let CoordinatorSource::Embedded { stopped } = source {
            match stopped.try_recv() {
                Ok(message) => return Err(message),
                Err(TryRecvError::Disconnected) => {
                    return Err("the local coordinator exited unexpectedly".to_string());
                }
                Err(TryRecvError::Empty) => {}
            }
        }
        if started.elapsed() >= timeout {
            return Err(format!(
                "nothing answered on http://127.0.0.1:{port} within {} seconds",
                timeout.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

/// Asks `http://127.0.0.1:<port>/api/v1/companion/status` whether a Kadmium coordinator is
/// listening, using a plain HTTP/1.1 request so no HTTP client crate is needed.
pub fn probe_coordinator(port: u16) -> Option<String> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(300)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_millis(2_000)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_millis(500)))
        .ok()?;
    stream
        .write_all(
            b"GET /api/v1/companion/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\
              Accept: application/json\r\nConnection: close\r\n\r\n",
        )
        .ok()?;
    let mut raw = Vec::new();
    let _ = stream.take(512 * 1024).read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw);
    let (head, body) = text.split_once("\r\n\r\n")?;
    if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
        return None;
    }
    let json_start = body.find('{')?;
    let json_end = body.rfind('}')?;
    let status: serde_json::Value = serde_json::from_str(&body[json_start..=json_end]).ok()?;
    status
        .get("app_version")
        .and_then(|version| version.as_str())
        .map(str::to_string)
}

pub fn origin(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

pub fn companion_url(port: u16) -> String {
    format!("{}/companion", origin(port))
}

pub fn control_room_url(port: u16) -> String {
    format!("{}/", origin(port))
}

/// Opens `url` in the default browser. Only `http`, `https` and `mailto` links are allowed,
/// so page content can never make the app launch arbitrary programs or local files.
pub fn open_in_browser(url: &str) {
    let lowered = url.to_ascii_lowercase();
    let allowed = ["http://", "https://", "mailto:"]
        .iter()
        .any(|scheme| lowered.starts_with(scheme));
    if !allowed {
        return;
    }
    spawn_browser(url);
}

#[cfg(windows)]
fn spawn_browser(url: &str) {
    // rundll32 hands the URL to the shell's protocol handler without a console window and
    // without `cmd` re-parsing characters such as `&` in query strings.
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

/// Where the webview keeps its profile (cache, local storage). WebView2 cannot write next to
/// an installed executable, so this always points at a per-user folder.
#[cfg_attr(
    all(target_os = "linux", not(feature = "linux-native")),
    allow(dead_code)
)]
pub fn webview_data_dir() -> Option<PathBuf> {
    data_root().map(|root| root.join("webview"))
}

#[cfg(windows)]
fn data_root() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|dir| PathBuf::from(dir).join("Kadmium"))
}

#[cfg(target_os = "macos")]
fn data_root() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join("Library/Application Support/Kadmium"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn data_root() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .map(|dir| dir.join("kadmium"))
}

/// Used when no native window is available: waits for the coordinator, opens the Companion
/// in the default browser, and keeps an embedded coordinator alive until it stops.
#[cfg_attr(
    any(not(target_os = "linux"), feature = "linux-native"),
    allow(dead_code)
)]
pub fn run_in_browser(args: &Args, source: CoordinatorSource) {
    if cfg!(target_os = "linux") {
        eprintln!(
            "kadmium-desktop: this build has no native window. Rebuild with \
             `--features linux-native` (needs WebKitGTK 4.1 and GTK 3 development packages) \
             to get one. Opening the Companion in your browser instead."
        );
    }
    match wait_until_ready(args.port, &source, Duration::from_secs(60)) {
        Ok(version) => {
            println!(
                "Kadmium {version} is answering on {}",
                companion_url(args.port)
            );
            open_in_browser(&companion_url(args.port));
            if args.control_room {
                open_in_browser(&control_room_url(args.port));
            }
        }
        Err(err) => {
            eprintln!("kadmium-desktop: {err}");
            std::process::exit(1);
        }
    }
    if let CoordinatorSource::Embedded { stopped } = source {
        let message = stopped
            .recv()
            .unwrap_or_else(|_| "the local coordinator stopped".to_string());
        eprintln!("kadmium-desktop: {message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(list: &[&str]) -> Result<Args, String> {
        Args::parse(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn parses_ports_and_flags() {
        let args = parse(&["--port", "7900", "--demo", "--control-room"]).unwrap();
        assert_eq!(args.port, 7900);
        assert!(args.demo && args.control_room && !args.help);
        assert_eq!(parse(&["--port=7911"]).unwrap().port, 7911);
        assert_eq!(parse(&[]).unwrap().port, 7848);
        assert!(parse(&["--port", "0"]).is_err());
        assert!(parse(&["--port"]).is_err());
        assert!(parse(&["--nope"]).is_err());
    }

    #[test]
    fn nothing_listening_means_no_coordinator() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        assert!(probe_coordinator(port).is_none());
    }
}
