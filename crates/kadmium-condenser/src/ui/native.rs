use super::{EmbeddedServer, UiOptions, open_in_browser, webview_data_dir};
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
use tao::window::{Theme, Window, WindowBuilder};
use wry::{NewWindowResponse, WebContext, WebView, WebViewBuilder};

pub fn run(options: UiOptions, server: EmbeddedServer) {
    let event_loop = EventLoopBuilder::<()>::with_user_event().build();
    let window = match build_window(&event_loop) {
        Ok(window) => window,
        Err(error) => {
            eprintln!("Condenser could not create a native window: {error}; using the browser");
            let _ = super::run_in_browser(server);
            return;
        }
    };
    let mut web_context = WebContext::new(webview_data_dir());
    let webview = match build_webview(&mut web_context, &window, &server.url) {
        Ok(webview) => webview,
        Err(error) => {
            eprintln!(
                "Condenser could not initialize the system webview: {error}; using the browser"
            );
            drop(window);
            let _ = super::run_in_browser(server);
            return;
        }
    };
    let _ = options;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let _ = &web_context;
        let _ = &webview;
        if matches!(
            event,
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            }
        ) {
            *control_flow = ControlFlow::Exit;
        }
    });
}

fn build_window(event_loop: &EventLoop<()>) -> Result<Window, String> {
    let builder = WindowBuilder::new()
        .with_title("Condenser — Agent Workspace")
        .with_inner_size(LogicalSize::new(1280.0, 820.0))
        .with_min_inner_size(LogicalSize::new(960.0, 640.0))
        .with_theme(Some(Theme::Dark))
        .with_window_icon(super::icon::window_icon(64));
    #[cfg(windows)]
    let builder = {
        use tao::platform::windows::WindowBuilderExtWindows;
        builder.with_taskbar_icon(super::icon::window_icon(256))
    };
    builder.build(event_loop).map_err(|error| error.to_string())
}

fn stays_in_window(origin: &str, next: &str) -> bool {
    if next.starts_with("about:") || next.starts_with("data:") {
        return true;
    }
    let (Ok(base), Ok(url)) = (reqwest::Url::parse(origin), reqwest::Url::parse(next)) else {
        return false;
    };
    base.scheme() == url.scheme()
        && base.host_str() == url.host_str()
        && base.port_or_known_default() == url.port_or_known_default()
}

fn build_webview(context: &mut WebContext, window: &Window, url: &str) -> wry::Result<WebView> {
    let origin = url.trim_end_matches('/').to_string();
    let builder = WebViewBuilder::new_with_web_context(context)
        .with_url(url)
        .with_background_color((8, 12, 20, 255))
        .with_devtools(cfg!(debug_assertions))
        .with_navigation_handler(move |next| {
            let stay = stays_in_window(&origin, &next);
            if !stay {
                open_in_browser(&next);
            }
            stay
        })
        .with_new_window_req_handler(|next, _| {
            open_in_browser(&next);
            NewWindowResponse::Deny
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigation_is_limited_to_the_actual_server_origin() {
        assert!(stays_in_window(
            "http://127.0.0.1:7856",
            "http://127.0.0.1:7856/plugins"
        ));
        assert!(stays_in_window("http://127.0.0.1:7856", "about:blank"));
        assert!(!stays_in_window(
            "http://127.0.0.1:7856",
            "http://127.0.0.1:7856.evil.example/"
        ));
        assert!(!stays_in_window(
            "http://127.0.0.1:7856",
            "https://example.com/"
        ));
        assert!(!stays_in_window(
            "http://127.0.0.1:7856",
            "javascript:alert(1)"
        ));
    }
}
