//! The native window: a tao window hosting a wry webview that shows the Companion.

use crate::icon;
use crate::launch::{self, Args, CoordinatorSource};
use std::time::Duration;
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy};
use tao::window::{Theme, Window, WindowBuilder};
use wry::{NewWindowResponse, WebContext, WebView, WebViewBuilder};

const SPLASH: &str = include_str!("splash.html");
const WINDOW_TITLE: &str = "Kadmium";

/// Messages from background threads to the window's event loop.
#[derive(Debug)]
enum AppEvent {
    /// The coordinator answers; show (or reload) the Companion.
    Ready,
    /// The coordinator cannot be reached; show the reason in the window.
    Unavailable { reason: String },
}

pub fn run(args: Args, source: CoordinatorSource) {
    let event_loop = EventLoopBuilder::<AppEvent>::with_user_event().build();
    let window = match build_window(&event_loop) {
        Ok(window) => window,
        Err(err) => {
            eprintln!("kadmium-desktop: could not open a window ({err}); using the browser.");
            launch::run_in_browser(&args, source);
            return;
        }
    };

    let port = args.port;
    let attaching = !source.is_embedded();
    let mut web_context = WebContext::new(launch::webview_data_dir());
    let webview = match build_webview(&mut web_context, &window, port, attaching) {
        Ok(webview) => webview,
        Err(err) => {
            eprintln!(
                "kadmium-desktop: the system webview is unavailable ({err}); using the browser. \
                 On Windows, install the Microsoft Edge WebView2 Runtime to get the app window."
            );
            drop(window);
            launch::run_in_browser(&args, source);
            return;
        }
    };

    watch_coordinator(port, source, event_loop.create_proxy());

    let mut opened_control_room = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        // Keep the profile context alive for as long as the webview.
        let _ = &web_context;
        match event {
            Event::UserEvent(AppEvent::Ready) => {
                let _ = webview.load_url(&launch::companion_url(port));
                if args.control_room && !opened_control_room {
                    opened_control_room = true;
                    launch::open_in_browser(&launch::control_room_url(port));
                }
            }
            Event::UserEvent(AppEvent::Unavailable { reason }) => {
                let _ = webview.load_html(&splash_page(port, attaching, Some(&reason)));
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn build_window(event_loop: &EventLoop<AppEvent>) -> Result<Window, String> {
    let builder = WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .with_inner_size(LogicalSize::new(1180.0, 780.0))
        .with_min_inner_size(LogicalSize::new(940.0, 620.0))
        .with_theme(Some(Theme::Dark))
        .with_window_icon(icon::window_icon(64));
    #[cfg(windows)]
    let builder = {
        use tao::platform::windows::WindowBuilderExtWindows;
        builder.with_taskbar_icon(icon::window_icon(256))
    };
    builder.build(event_loop).map_err(|err| err.to_string())
}

fn build_webview(
    web_context: &mut WebContext,
    window: &Window,
    port: u16,
    attaching: bool,
) -> wry::Result<WebView> {
    let builder = WebViewBuilder::new_with_web_context(web_context)
        .with_html(splash_page(port, attaching, None))
        .with_background_color((14, 16, 19, 255))
        .with_initialization_script(bridge_script())
        .with_devtools(cfg!(debug_assertions))
        .with_navigation_handler(move |url| {
            let stay = stays_in_window(port, &url);
            if !stay {
                launch::open_in_browser(&url);
            }
            stay
        })
        .with_new_window_req_handler(|url, _features| {
            launch::open_in_browser(&url);
            NewWindowResponse::Deny
        })
        .with_ipc_handler(move |request| handle_message(port, request.body()));

    // Release builds drop the browser context menu and shortcuts (print, find, reload) so the
    // window behaves like an app; debug builds keep them for inspecting the page.
    #[cfg(windows)]
    let builder = {
        use wry::WebViewBuilderExtWindows;
        builder
            .with_default_context_menus(cfg!(debug_assertions))
            .with_browser_accelerator_keys(cfg!(debug_assertions))
    };

    #[cfg(not(target_os = "linux"))]
    {
        builder.build(window)
    }
    #[cfg(target_os = "linux")]
    {
        use tao::platform::unix::WindowExtUnix;
        use wry::WebViewBuilderExtUnix;
        let vbox = window
            .default_vbox()
            .ok_or(wry::Error::UnsupportedWindowHandle)?;
        builder.build_gtk(vbox)
    }
}

/// Waits for the coordinator, then keeps checking it so the window can say when it goes
/// away (and reload the Companion if an attached coordinator comes back).
fn watch_coordinator(port: u16, source: CoordinatorSource, proxy: EventLoopProxy<AppEvent>) {
    let spawned = std::thread::Builder::new()
        .name("kadmium-coordinator-watch".into())
        .spawn(move || {
            if let Err(reason) = launch::wait_until_ready(port, &source, Duration::from_secs(90)) {
                let _ = proxy.send_event(AppEvent::Unavailable { reason });
                return;
            }
            if proxy.send_event(AppEvent::Ready).is_err() {
                return;
            }
            let mut reachable = true;
            let mut misses = 0u32;
            loop {
                std::thread::sleep(Duration::from_secs(3));
                if let CoordinatorSource::Embedded { stopped } = &source
                    && let Ok(reason) = stopped.try_recv()
                {
                    let _ = proxy.send_event(AppEvent::Unavailable { reason });
                    return;
                }
                let up = launch::probe_coordinator(port).is_some();
                misses = if up { 0 } else { misses + 1 };
                let event = if reachable && misses >= 3 {
                    reachable = false;
                    Some(AppEvent::Unavailable {
                        reason: format!(
                            "Lost the connection to the coordinator on port {port}. \
                             The window reconnects as soon as it answers again."
                        ),
                    })
                } else if !reachable && up {
                    reachable = true;
                    Some(AppEvent::Ready)
                } else {
                    None
                };
                if let Some(event) = event
                    && proxy.send_event(event).is_err()
                {
                    return;
                }
            }
        });
    if let Err(err) = spawned {
        eprintln!("kadmium-desktop: could not watch the coordinator: {err}");
    }
}

/// Only the Companion (and the built-in splash) render inside the window; everything else,
/// including the Control Room, opens in the default browser.
fn stays_in_window(port: u16, url: &str) -> bool {
    if url.starts_with("about:") || url.starts_with("data:") {
        return true;
    }
    let companion = launch::companion_url(port);
    url.strip_prefix(companion.as_str())
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(['?', '#', '/']))
}

/// Handles `window.ipc.postMessage` calls made through `window.__KADMIUM_DESKTOP__`.
fn handle_message(port: u16, body: &str) {
    let Ok(message) = serde_json::from_str::<serde_json::Value>(body) else {
        return;
    };
    match message.get("cmd").and_then(|cmd| cmd.as_str()) {
        Some("open") => {
            if let Some(url) = message.get("url").and_then(|url| url.as_str()) {
                launch::open_in_browser(url);
            }
        }
        Some("openControlRoom") => {
            let hash = message
                .get("hash")
                .and_then(|hash| hash.as_str())
                .filter(|hash| hash.starts_with("#/"))
                .unwrap_or("");
            launch::open_in_browser(&format!("{}{hash}", launch::control_room_url(port)));
        }
        _ => {}
    }
}

/// Runs before every page in the window. The Companion reads `window.__KADMIUM_DESKTOP__` to
/// switch into its desktop layout and to open links in the default browser.
fn bridge_script() -> String {
    let platform = if cfg!(windows) {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    };
    format!(
        "window.__KADMIUM_DESKTOP__ = Object.freeze({{\
            shell: 'native',\
            version: '{version}',\
            platform: '{platform}',\
            openExternal: function (url) {{\
                window.ipc.postMessage(JSON.stringify({{ cmd: 'open', url: String(url) }}));\
            }},\
            openControlRoom: function (hash) {{\
                window.ipc.postMessage(JSON.stringify({{ cmd: 'openControlRoom', hash: String(hash || '') }}));\
            }}\
        }});",
        version = env!("CARGO_PKG_VERSION"),
    )
}

/// The page shown while the coordinator starts, or when it cannot be reached.
fn splash_page(port: u16, attaching: bool, error: Option<&str>) -> String {
    let (state, message, hint) = match error {
        None if attaching => (
            "",
            format!("Connecting to the coordinator already running on port {port}…"),
            String::new(),
        ),
        None => (
            "",
            "Starting your local coordinator and probing this machine…".to_string(),
            String::new(),
        ),
        Some(reason) => (
            "error",
            escape_html(reason),
            if reason.contains("bind") || reason.contains("in use") {
                format!(
                    "Another program is using port <code>{port}</code>. Close it, or start the \
                     app on a free port: <code>kadmium-desktop --port {next}</code>",
                    next = port.saturating_add(1)
                )
            } else {
                "Check the console for details, or run <code>kadmium serve</code> in a \
                 terminal and reopen the app; it attaches to a running coordinator."
                    .to_string()
            },
        ),
    };
    SPLASH
        .replace("{{STATE}}", state)
        .replace("{{MESSAGE}}", &message)
        .replace("{{HINT}}", &hint)
        .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
        .replace("{{PORT}}", &port.to_string())
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_companion_stays_in_the_window() {
        assert!(stays_in_window(7848, "about:blank"));
        assert!(stays_in_window(7848, "http://127.0.0.1:7848/companion"));
        assert!(stays_in_window(
            7848,
            "http://127.0.0.1:7848/companion?tab=models"
        ));
        assert!(stays_in_window(
            7848,
            "http://127.0.0.1:7848/companion#pairing"
        ));
        assert!(!stays_in_window(
            7848,
            "http://127.0.0.1:7848/companionship"
        ));
        assert!(!stays_in_window(7848, "http://127.0.0.1:7848/"));
        assert!(!stays_in_window(7848, "http://127.0.0.1:7848/docs"));
        assert!(!stays_in_window(7848, "javascript:alert(1)"));
    }

    #[test]
    fn splash_explains_a_taken_port() {
        let page = splash_page(7848, false, Some("could not bind the coordinator: in use"));
        assert!(page.contains("class=\"error\""));
        assert!(page.contains("--port 7849"));
        assert!(!page.contains("{{"));
    }
}
