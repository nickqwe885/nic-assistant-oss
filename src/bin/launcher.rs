#![windows_subsystem = "windows"]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tao::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};
use tray_icon::{
    Icon as TrayIcon,
    TrayIconBuilder,
    TrayIconEvent,
    MouseButton,
    MouseButtonState,
    menu::{Menu, MenuEvent, MenuItem},
};
use wry::{WebViewBuilder, WebViewBuilderExtWindows};

const ICON_PNG: &[u8] = include_bytes!("icon.png");
const UI_HTML: &str = include_str!("../ui.html");

// Content-Security-Policy sent with the embedded UI (served from the nic://app
// custom protocol). The page is vanilla inline JS/CSS, so 'unsafe-inline' is
// required for script/style; everything else is locked down. `connect-src` is
// pinned to the loopback API ports the UI actually talks to (mirrors ui.html's
// API_PORTS), so even if injected script ran it could not exfiltrate to a remote
// host. object/frame/base/form are denied outright. Images allow data: (inline
// SVG backgrounds) and the same loopback ports.
const CSP: &str = "default-src 'self'; \
script-src 'self' 'unsafe-inline'; \
style-src 'self' 'unsafe-inline'; \
img-src 'self' data: http://127.0.0.1:7878 http://127.0.0.1:7879 http://127.0.0.1:7880; \
font-src 'self' data:; \
connect-src http://127.0.0.1:7878 http://127.0.0.1:7879 http://127.0.0.1:7880; \
object-src 'none'; base-uri 'none'; frame-src 'none'; form-action 'none'";

// Compile-time integrity guard for the embedded UI.
//
// History: ui.html was once silently truncated (432 of 747 lines — losing all
// JS and the chat input), and the resulting binary shipped with a blank window.
// The full UI is ~134 KB; a truncated one is a fraction of that. This assert
// fails the BUILD if the file is suspiciously small, so a corrupted/truncated
// ui.html can never reach a release again.
//
// (A size check is enough — substring scanning over 130 KB in const context
// blows the const-eval step limit, and truncation always shrinks the file.)
const _: () = assert!(
    UI_HTML.len() > 80_000,
    "ui.html is suspiciously small — likely truncated/corrupted; refusing to build a blank-window binary",
);

// Runtime sanity check (cheap, runs once at startup): log a loud error if the
// embedded UI is missing its essential markers, in case it was edited in a way
// that kept the byte size but broke the structure.
fn assert_ui_markers() {
    for marker in ["</html>", "<script", "/query"] {
        if !UI_HTML.contains(marker) {
            tracing::error!("Embedded UI is missing '{marker}' — the window may render blank.");
        }
    }
}

#[derive(Debug, Clone)]
enum UserEvent {
    Minimize,
    ShowWindow,
    Quit,
}

/// Opens an external URL in the user's default browser via `ShellExecuteW`.
///
/// The caller guarantees `url` is `http(s)://` (validated in the navigation
/// handler). We deliberately do NOT shell out to `cmd /c start "" <url>`: cmd
/// treats `&` as a command separator, so a model- or web-sourced link such as
/// `https://a&calc` would run `calc` — a command injection. `ShellExecuteW`
/// hands the URL straight to the shell-open handler with no command-line parsing.
fn open_external_url(url: &str) {
    use windows::core::{w, HSTRING, PCWSTR};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let file = HSTRING::from(url);
    unsafe {
        let _ = ShellExecuteW(
            HWND(std::ptr::null_mut()),
            w!("open"),
            PCWSTR(file.as_ptr()),
            PCWSTR::null(),
            PCWSTR::null(),
            SW_SHOWNORMAL,
        );
    }
}

fn load_tray_icon() -> TrayIcon {
    let img = image::load_from_memory(ICON_PNG)
        .expect("tray icon PNG decode failed")
        .into_rgba8();
    let (w, h) = (img.width(), img.height());
    TrayIcon::from_rgba(img.into_raw(), w, h).expect("tray icon creation failed")
}

fn main() {
    assert_ui_markers();

    // Security: generate a random per-launch token shared between this UI and
    // the backend via an env var. The backend then requires it on every request
    // (see api.rs auth_middleware), so a malicious website cannot reach the
    // localhost API and read your memory — it never sees this token.
    let local_token = std::env::var("NIC_LOCAL_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    std::env::set_var("NIC_LOCAL_TOKEN", &local_token);
    let ui_html = UI_HTML.replace("__NIC_TOKEN__", &local_token);

    let backend_done = Arc::new(AtomicBool::new(false));
    let done = backend_done.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime for backend");
        if let Err(e) = rt.block_on(nic_assistant_lib::run_backend()) {
            tracing::error!("Backend error: {}", e);
        }
        done.store(true, Ordering::SeqCst);
    });

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    {
        let p = proxy.clone();
        std::thread::spawn(move || {
            use windows::Win32::Foundation::HWND;
            use windows::Win32::UI::Input::KeyboardAndMouse::{
                RegisterHotKey, HOT_KEY_MODIFIERS, MOD_ALT, MOD_WIN,
            };
            use windows::Win32::UI::WindowsAndMessaging::{GetMessageW, MSG, WM_HOTKEY};

            const HOTKEY_ID: i32 = 1;
            const VK_N: u32 = 0x4E;

            unsafe {
                let mods = HOT_KEY_MODIFIERS(MOD_WIN.0 | MOD_ALT.0);
                if RegisterHotKey(HWND(std::ptr::null_mut()), HOTKEY_ID, mods, VK_N).is_ok() {
                    let mut msg = MSG::default();
                    loop {
                        if GetMessageW(&mut msg, HWND(std::ptr::null_mut()), 0, 0).as_bool() {
                            if msg.message == WM_HOTKEY && msg.wParam.0 as i32 == HOTKEY_ID {
                                let _ = p.send_event(UserEvent::ShowWindow);
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        });
    }

    let window = WindowBuilder::new()
        .with_title("NIC-Assistant")
        .with_inner_size(tao::dpi::LogicalSize::new(420_f64, 680_f64))
        .with_min_inner_size(tao::dpi::LogicalSize::new(360_f64, 480_f64))
        .with_decorations(false)
        // Opaque, solid-background window. (Transparency used to bleed the
        // desktop through the panel and break when maximized — fixed by a solid
        // body background, so resize/maximize are now allowed.)
        .build(&event_loop)
        .unwrap();

    if let Some(monitor) = window.current_monitor() {
        let screen = monitor.size();
        let win    = window.outer_size();
        let x = (screen.width.saturating_sub(win.width)) / 2;
        let y = (screen.height.saturating_sub(win.height)) / 2;
        window.set_outer_position(tao::dpi::PhysicalPosition::new(x as i32, y as i32));
    }

    let show_item = MenuItem::new("Show", true, None);
    let quit_item = MenuItem::new("Quit", true, None);

    let show_id = show_item.id().clone();
    let quit_id = quit_item.id().clone();

    let tray_menu = Menu::new();
    tray_menu.append_items(&[
        &show_item as &dyn tray_icon::menu::IsMenuItem,
        &quit_item as &dyn tray_icon::menu::IsMenuItem,
    ]).ok();

    let _tray = TrayIconBuilder::new()
        .with_icon(load_tray_icon())
        .with_tooltip("NIC-Assistant")
        .with_menu(Box::new(tray_menu))
        .build()
        .expect("system tray creation failed");

    {
        let p = proxy.clone();
        TrayIconEvent::set_event_handler(Some(move |ev: TrayIconEvent| {
            if matches!(
                ev,
                TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } | TrayIconEvent::DoubleClick {
                    button: MouseButton::Left,
                    ..
                }
            ) {
                let _ = p.send_event(UserEvent::ShowWindow);
            }
        }));
    }

    {
        let p = proxy.clone();
        MenuEvent::set_event_handler(Some(move |ev: MenuEvent| {
            if ev.id == show_id {
                let _ = p.send_event(UserEvent::ShowWindow);
            } else if ev.id == quit_id {
                let _ = p.send_event(UserEvent::Quit);
            }
        }));
    }

    let proxy_ipc = proxy.clone();
    let _webview = WebViewBuilder::new()
        // Serve the embedded UI from the `nic://app` custom protocol instead of
        // `with_html`. A custom protocol gives the page a real, stable origin
        // (`https://nic.app` on Windows, via with_https_scheme below) rather than
        // the opaque "null" origin of a data document. A real origin (a) unblocks
        // localStorage — history persistence and the autostart-consent flag —
        // and (b) lets us enforce a Content-Security-Policy. This is the first
        // half of the v8 move to nic://app; the API still rides HTTP+token for
        // now (MASTER_PLAN §8б — staged transport).
        //
        // with_https_scheme(true) makes the origin `https://` (a secure context),
        // which is what lets the page's `fetch('http://127.0.0.1:…')` pass
        // Private Network Access checks. (127.0.0.1 is exempt from mixed-content
        // blocking, so https→loopback http is allowed.)
        .with_https_scheme(true)
        .with_custom_protocol("nic".into(), move |_id, _req| {
            wry::http::Response::builder()
                .header(wry::http::header::CONTENT_TYPE, "text/html; charset=utf-8")
                .header(wry::http::header::CONTENT_SECURITY_POLICY, CSP)
                .body(std::borrow::Cow::<'static, [u8]>::Owned(ui_html.clone().into_bytes()))
                .unwrap()
        })
        .with_url("nic://app")
        // Navigation lockdown (MASTER_PLAN §2.1): only our own origin may load
        // inside the webview. Real external links open in the SYSTEM browser —
        // never let a click navigate the chat webview away (that would blank it).
        // Everything else (data:, file:, javascript:, redirects) is blocked.
        .with_navigation_handler(|url: String| {
            // Our own app shell — allow. The custom protocol intercepts every
            // `https://nic.*` host (with_https_scheme), so `https://nic.app` can
            // only ever be our local page, never the real internet domain. We do
            // NOT allow `http://nic.app`: that scheme is not intercepted and would
            // reach the real `nic.app` on the network.
            if url.starts_with("https://nic.app")
                || url.starts_with("nic://app")
                || url.starts_with("about:")
            {
                return true;
            }
            // Genuine external link → OS browser, then cancel in-webview nav.
            if url.starts_with("http://") || url.starts_with("https://") {
                open_external_url(&url);
            }
            false
        })
        .with_ipc_handler(move |req: wry::http::Request<String>| {
            match req.body().as_str() {
                // ✕ fully quits NIC (window + backend + llama-server). Use «–» to
                // minimize and keep memory running in the background.
                "close"    => { let _ = proxy_ipc.send_event(UserEvent::Quit); }
                "minimize" => { let _ = proxy_ipc.send_event(UserEvent::Minimize); }
                _ => {}
            }
        })
        .build(&window)
        .unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                // Alt+F4 / native close → full quit (consistent with the ✕ button).
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(UserEvent::Minimize) => {
                // Proper minimize to the taskbar (not hide-to-tray) so the
                // window doesn't appear to "disappear" when minimized.
                window.set_minimized(true);
            }
            Event::UserEvent(UserEvent::ShowWindow) => {
                window.set_visible(true);
                window.set_minimized(false);
                window.set_focus();
            }
            Event::UserEvent(UserEvent::Quit) => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
}