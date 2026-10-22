//! Kadmium desktop app.
//!
//! Opens the Companion in its own native window: WebView2 on Windows, WKWebView on macOS,
//! and WebKitGTK on Linux when built with `--features linux-native`. If a coordinator is
//! already running on the chosen port the window attaches to it; otherwise the app starts
//! one in-process and stops it when the window closes.
//!
//! The window only ever shows the Companion. The Control Room, the docs and every other
//! page open in the default browser, so the desktop app and the browser never look alike.

// Release builds on Windows are GUI apps: no console window behind the app window.
// Debug builds keep the console so `cargo run -p kadmium-desktop` shows coordinator logs.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(any(not(target_os = "linux"), feature = "linux-native"))]
mod icon;
mod launch;
#[cfg(any(not(target_os = "linux"), feature = "linux-native"))]
mod native;

fn main() {
    let args = match launch::Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("kadmium-desktop: {message}\n\n{}", launch::USAGE);
            std::process::exit(2);
        }
    };
    if args.help {
        println!("{}", launch::USAGE);
        return;
    }

    let coordinator = launch::ensure_coordinator(&args);

    #[cfg(any(not(target_os = "linux"), feature = "linux-native"))]
    native::run(args, coordinator);

    #[cfg(all(target_os = "linux", not(feature = "linux-native")))]
    launch::run_in_browser(&args, coordinator);
}
