use arboard::Clipboard;
use screenshots::Screen;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::shakal::{active_window_info, changed_block_ratio, ShakalProcessor, GRID_CELLS};
use super::{SentinelEvent, PERFORMANCE_MODE};


/// Returns true for code editors and terminal emulators where even a single
/// changed line matters — these get a tighter pHash threshold (2 bits vs 5).
fn is_code_editor(app_name: &str) -> bool {
    const EDITORS: &[&str] = &["code", "cursor", "windowsterminal", "nvim", "vim", "idea", "rider"];
    let lower = app_name.to_lowercase();
    EDITORS.iter().any(|e| lower.contains(e))
}

// ── Anti-cheat-friendly foreground guard (MASTER_PLAN §4) ───────────────────
//
// Screen capture must never run while a kernel-level anti-cheat (Riot Vanguard,
// EAC, BattlEye) is active, or NIC looks like a cheat that reads the screen.
// Rather than scanning processes, we install ONE passive Windows hook on
// EVENT_SYSTEM_FOREGROUND in a dedicated native OS thread with a classic Win32
// message loop. The hook fires only when the foreground window changes; we then
// do a single GetWindowThreadProcessId + OpenProcess(PROCESS_QUERY_LIMITED_
// INFORMATION) — which does NOT read process memory, so it stays legitimate to
// the OS and to anti-cheats — and match the exe against a games/anti-cheat
// table. A hit flips an atomic flag and the capture loop suspends instantly.

/// True while a known game / anti-cheat process owns the foreground window.
/// Writer: the foreground-hook thread (AcqRel). Reader: the screen-capture loop
/// (Acquire). Explicit ordering per MASTER_PLAN §8d — never Relaxed.
static GAME_FOREGROUND: AtomicBool = AtomicBool::new(false);

/// Screen-reading consent. Defaults to `false` so a fresh install captures
/// NOTHING — not a single screenshot — until the user explicitly enables memory
/// in the UI. Set once at startup from the on-disk consent marker, and flipped
/// live by `POST /consent`. Reader: the capture loop (Acquire).
static CAPTURE_CONSENT: AtomicBool = AtomicBool::new(false);

/// Sets whether the screen-capture loop is allowed to run at all (see
/// [`CAPTURE_CONSENT`]). Called at startup and by the consent endpoint.
pub fn set_capture_consent(allowed: bool) {
    CAPTURE_CONSENT.store(allowed, Ordering::Release);
}

/// True once the user has consented to screen capture.
pub fn capture_consent() -> bool {
    CAPTURE_CONSENT.load(Ordering::Acquire)
}

/// Foreground exe stems (lowercase, no extension) treated as games.
const GAME_EXES: &[&str] = &[
    "valorant-win64-shipping", "valorant",
    "csgo", "cs2",
    "league of legends", "leagueoflegends", "riotclientservices",
    "fortniteclient-win64-shipping",
    "r5apex", "r5apex_dx12",
    "destiny2",
    "rainbowsix", "rainbowsix_vulkan",
    "tslgame",            // PUBG
    "dota2",
    "overwatch",
    "modernwarfare", "cod", "bo6",
    "eldenring",
];

/// Substrings that, anywhere in the full image path, mark an anti-cheat (its
/// presence in the foreground process means a protected game is live). Matched
/// with `contains`, so each is anchored enough not to hit ordinary apps.
const ANTICHEAT_HINTS: &[&str] = &[
    "vanguard", "vgtray", "vgc",
    "easyanticheat", "\\eac\\",
    "battleye", "beservice",
    "punkbuster",
    "faceit",
];

/// Pure, testable core of the game check: does this lowercased full image path
/// belong to a game / anti-cheat? Split out so it is unit-testable without
/// Win32. Matches the exe stem against [`GAME_EXES`] (equality, so "cod" only
/// hits cod.exe) or any [`ANTICHEAT_HINTS`] substring anywhere in the path.
fn path_is_game(full_path_lower: &str) -> bool {
    let stem = std::path::Path::new(full_path_lower)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| full_path_lower.to_string());
    GAME_EXES.iter().any(|g| stem == *g)
        || ANTICHEAT_HINTS.iter().any(|h| full_path_lower.contains(h))
}

/// Reads the anti-cheat guard flag (Acquire). Always `false` on non-Windows.
pub(crate) fn game_in_foreground() -> bool {
    GAME_FOREGROUND.load(Ordering::Acquire)
}

/// Public read for the API: lets the UI show "game detected — memory paused",
/// which proves the anti-cheat guard is working exactly when the user cares.
pub fn game_active() -> bool {
    GAME_FOREGROUND.load(Ordering::Acquire)
}

#[cfg(not(windows))]
pub(crate) fn spawn_foreground_watch() {}

#[cfg(windows)]
type WinEventProc = unsafe extern "system" fn(isize, u32, isize, i32, i32, u32, u32);

#[cfg(windows)]
#[repr(C)]
#[allow(dead_code)] // fields are written by the OS via GetMessageW, never read by us
struct Point { x: i32, y: i32 }

#[cfg(windows)]
#[repr(C)]
#[allow(dead_code)] // ditto — this is an FFI out-parameter buffer for GetMessageW
struct Msg {
    hwnd:    isize,
    message: u32,
    w_param: usize,
    l_param: isize,
    time:    u32,
    pt:      Point,
}

#[cfg(windows)]
#[link(name = "user32")]
extern "system" {
    fn GetForegroundWindow() -> isize;
    fn GetWindowThreadProcessId(hwnd: isize, pid: *mut u32) -> u32;
    fn SetWinEventHook(
        event_min: u32, event_max: u32, hmod: isize,
        callback: WinEventProc, id_process: u32, id_thread: u32, flags: u32,
    ) -> isize;
    fn UnhookWinEvent(hook: isize) -> i32;
    fn GetMessageW(msg: *mut Msg, hwnd: isize, min: u32, max: u32) -> i32;
    fn TranslateMessage(msg: *const Msg) -> i32;
    fn DispatchMessageW(msg: *const Msg) -> isize;
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn OpenProcess(access: u32, inherit: i32, pid: u32) -> isize;
    fn QueryFullProcessImageNameW(h: isize, flags: u32, name: *mut u16, size: *mut u32) -> i32;
    fn CloseHandle(h: isize) -> i32;
}

#[cfg(windows)] const EVENT_SYSTEM_FOREGROUND:           u32 = 0x0003;
#[cfg(windows)] const WINEVENT_OUTOFCONTEXT:             u32 = 0x0000;
#[cfg(windows)] const WINEVENT_SKIPOWNPROCESS:           u32 = 0x0002;
#[cfg(windows)] const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

/// Resolves a pid to its full image path and checks it against the games table.
/// Uses PROCESS_QUERY_LIMITED_INFORMATION only — never reads process memory.
#[cfg(windows)]
fn foreground_process_is_game(pid: u32) -> bool {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h == 0 { return false; }
        let mut buf  = [0u16; 512];
        let mut size = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(h, 0, buf.as_mut_ptr(), &mut size);
        CloseHandle(h);
        if ok == 0 || size == 0 { return false; }
        let full = OsString::from_wide(&buf[..size as usize])
            .to_string_lossy()
            .to_lowercase();
        path_is_game(&full)
    }
}

/// Win-event hook callback — fires on every foreground-window change. A plain
/// `extern "system"` fn (the OS passes no user pointer), so it shares state with
/// the capture loop only through the [`GAME_FOREGROUND`] static.
#[cfg(windows)]
unsafe extern "system" fn on_foreground_changed(
    _hook: isize, _event: u32, hwnd: isize,
    _id_object: i32, _id_child: i32, _id_event_thread: u32, _ms: u32,
) {
    if hwnd == 0 { return; }
    let mut pid: u32 = 0;
    GetWindowThreadProcessId(hwnd, &mut pid);
    if pid == 0 { return; }
    let is_game = foreground_process_is_game(pid);
    let was = GAME_FOREGROUND.swap(is_game, Ordering::AcqRel);
    if is_game && !was {
        info!("[Sentinel/AntiCheat] game/anti-cheat in foreground — screen capture SUSPENDED");
    } else if !is_game && was {
        info!("[Sentinel/AntiCheat] foreground left game — screen capture RESUMED");
    }
}

/// Spawns the dedicated OS thread that installs the foreground hook and pumps its
/// message loop. Isolated with `catch_unwind` so a panic here can never bring
/// down the backend — it only disables the guard (and is logged).
#[cfg(windows)]
pub(crate) fn spawn_foreground_watch() {
    let _ = std::thread::Builder::new()
        .name("nic-foreground-watch".into())
        .spawn(|| {
            let run = std::panic::AssertUnwindSafe(|| unsafe { run_foreground_hook() });
            if std::panic::catch_unwind(run).is_err() {
                warn!("[Sentinel/AntiCheat] foreground watch thread panicked — guard OFF");
            }
        });
}

#[cfg(windows)]
unsafe fn run_foreground_hook() {
    // The hook only fires on CHANGE, so probe the current foreground once at
    // startup in case a game is already focused when NIC launches.
    let hwnd = GetForegroundWindow();
    if hwnd != 0 {
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid != 0 {
            GAME_FOREGROUND.store(foreground_process_is_game(pid), Ordering::Release);
        }
    }

    let hook = SetWinEventHook(
        EVENT_SYSTEM_FOREGROUND,
        EVENT_SYSTEM_FOREGROUND,
        0,
        on_foreground_changed,
        0,
        0,
        WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS,
    );
    if hook == 0 {
        warn!("[Sentinel/AntiCheat] SetWinEventHook failed — anti-cheat guard is OFF");
        return;
    }
    info!("[Sentinel/AntiCheat] foreground hook installed (EVENT_SYSTEM_FOREGROUND)");

    // Classic Win32 message loop — required even for WINEVENT_OUTOFCONTEXT, whose
    // callbacks are delivered through this thread's message queue.
    let mut msg: Msg = std::mem::zeroed();
    while GetMessageW(&mut msg, 0, 0, 0) > 0 {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    UnhookWinEvent(hook);
}

pub struct Sentinel {
    tx: mpsc::Sender<SentinelEvent>,
}

impl Sentinel {
    pub fn new(tx: mpsc::Sender<SentinelEvent>) -> Self {
        Self { tx }
    }


    /// Spawns two independent blocking tasks connected to Librarian via mpsc.
    ///
    /// Task 1 — Screen capture pipeline:
    ///   - CPU gate: skip when PERFORMANCE_MODE = true (CPU > 70 %)
    ///   - active_window_info() for app_name + window_title
    ///   - ShakalProcessor::process() → Privacy Shield + JPEG compression
    ///   - Dirty rectangles: skip if < 5% of 64×64 blocks changed vs previous frame
    ///   - pHash dedup: skip if Hamming distance < 5 (belt-and-suspenders)
    ///   - Adaptive interval: 3 s active, 20 s idle
    ///
    /// Task 2 — Clipboard watcher (2 s poll, skip private content)
    pub fn start(self: std::sync::Arc<Self>) {
        let tx_screen = self.tx.clone();
        let tx_clip   = self.tx.clone();

        // Anti-cheat guard: passive foreground hook on its own OS thread (§4).
        spawn_foreground_watch();

        // ── Task 1: Screen capture ────────────────────────────────────────────
        tokio::task::spawn_blocking(move || {
            let mut last_phash:        u64              = 0;
            let mut last_block_hashes: [u64; GRID_CELLS] = [0u64; GRID_CELLS];
            let mut last_activity                       = Instant::now();
            let mut static_streak:     u32              = 0;

            loop {
                let idle     = last_activity.elapsed() >= Duration::from_secs(30);
                // Intelligent FPS: if screen unchanged for 5+ consecutive checks → deep sleep
                let interval = if static_streak >= 5 {
                    Duration::from_secs(60)
                } else if idle {
                    Duration::from_secs(20)
                } else {
                    Duration::from_secs(3)
                };
                std::thread::sleep(interval);

                if PERFORMANCE_MODE.load(Ordering::Relaxed) {
                    continue;
                }

                // Anti-cheat guard (§4): a known game / anti-cheat owns the
                // foreground — do not touch the screen, so kernel anti-cheats
                // (Vanguard/EAC) never observe a capture. Set by the foreground
                // hook thread; read here with Acquire (§8d).
                if game_in_foreground() {
                    continue;
                }

                // Consent gate: until the user enables memory, take NO screenshot
                // at all — capture never even runs. This is the honest first-run
                // promise ("nothing is read until you say so"), not a store-time
                // filter. Cheap Acquire load, checked every tick.
                if !capture_consent() {
                    continue;
                }

                let (app_name, window_title) = active_window_info();

                // Secret shield, layer 0: a wallet / password manager / banking
                // window is never captured at all — no screenshot, no OCR, no
                // record it was open. Covers secrets we have no pattern for.
                if crate::modules::scrubber::is_private_window(&window_title, &app_name) {
                    continue;
                }

                let screens = match Screen::all() {
                    Ok(s)  => s,
                    Err(e) => { warn!("[Sentinel/Screen] enum: {}", e); continue; }
                };
                let Some(screen) = screens.into_iter().next() else { continue };

                let t0      = Instant::now();
                let capture = match screen.capture() {
                    Ok(c)  => c,
                    Err(e) => { warn!("[Sentinel/Screen] capture: {}", e); continue; }
                };

                let frame = match ShakalProcessor::process(
                    capture.as_raw(),
                    capture.width(),
                    capture.height(),
                    &window_title,
                    &app_name,
                ) {
                    Some(f) => f,
                    None    => continue,
                };
                // Raw pixel buffer released here — only OCR text travels forward.
                drop(capture);

                // Dirty-rectangle gate: skip when < 5% of blocks changed.
                // Only active after the first frame (zero hashes = uninitialized).
                let dirty_ratio = changed_block_ratio(&last_block_hashes, &frame.block_hashes);
                if dirty_ratio < 0.05 && last_block_hashes != [0u64; GRID_CELLS] {
                    static_streak = static_streak.saturating_add(1);
                    continue;
                }

                // pHash dedup — tighter threshold for code editors to catch single-line changes.
                let phash_threshold = if is_code_editor(&frame.app_name) { 2 } else { 5 };
                if (frame.phash ^ last_phash).count_ones() < phash_threshold {
                    static_streak = static_streak.saturating_add(1);
                    continue;
                }

                last_phash        = frame.phash;
                last_block_hashes = frame.block_hashes;
                last_activity     = Instant::now();
                static_streak     = 0;

                let thresh_label = if phash_threshold == 2 { "2b (code)" } else { "5b" };
                info!(
                    "[SENTINEL]: Focus: {} | pHash: {} threshold | Status: CHANGE DETECTED | dirty={:.0}% | {}ms",
                    frame.app_name, thresh_label, dirty_ratio * 100.0, t0.elapsed().as_millis(),
                );
                if frame.ocr_char_count > 0 {
                    info!("[OCR]: Captured {} chars from active window.", frame.ocr_char_count);
                }

                // Secret shield (§privacy): seed phrases, card numbers and API
                // keys are redacted BEFORE anything is persisted — they never
                // reach the database, so they can never surface in an answer.
                let description = crate::modules::scrubber::scrub(&frame.text_summary);
                if description != frame.text_summary {
                    info!("[Sentinel/Privacy] secret redacted from screen capture");
                }

                let event = SentinelEvent::Screen {
                    app_name:     frame.app_name,
                    window_title: frame.window_title,
                    description,
                };

                if let Err(e) = tx_screen.blocking_send(event) {
                    warn!("[Sentinel/Screen] channel closed: {}", e);
                    break;
                }
            }
        });

        // ── Task 2: Clipboard watcher ─────────────────────────────────────────
        tokio::task::spawn_blocking(move || {
            let mut clipboard = match Clipboard::new() {
                Ok(c)  => c,
                Err(e) => { warn!("[Sentinel/Clip] init: {}", e); return; }
            };
            let mut last = String::new();

            loop {
                std::thread::sleep(Duration::from_secs(2));

                let text = match clipboard.get_text() {
                    Ok(t) => t.trim().to_string(),
                    Err(_) => continue,
                };

                if text.is_empty() || text == last || text.len() < 10 {
                    continue;
                }
                last = text.clone();

                // Secret shield: a copied seed phrase / card / API key is the
                // single most dangerous thing the clipboard ever holds. Drop the
                // whole item rather than storing a redacted husk of it.
                if crate::modules::scrubber::contains_secret(&text) {
                    info!("[Sentinel/Privacy] clipboard item contained a secret — NOT stored");
                    continue;
                }
                info!("[Sentinel/Clip] {} chars", text.len());

                if let Err(e) = tx_clip.blocking_send(SentinelEvent::Clipboard { text }) {
                    warn!("[Sentinel/Clip] channel closed: {}", e);
                    break;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::path_is_game;

    #[test]
    fn detects_game_by_exe_stem() {
        assert!(path_is_game(r"c:\riot games\valorant\live\valorant-win64-shipping.exe"));
        assert!(path_is_game(r"d:\steam\steamapps\common\cs2\game\bin\win64\cs2.exe"));
        assert!(path_is_game(r"e:\games\dota 2 beta\game\bin\win64\dota2.exe"));
    }

    #[test]
    fn detects_anticheat_by_path_substring() {
        assert!(path_is_game(r"c:\program files\riot vanguard\vgc.exe"));
        assert!(path_is_game(r"c:\program files (x86)\easyanticheat\easyanticheat.exe"));
        assert!(path_is_game(r"c:\games\somegame\battleye\beservice.exe"));
    }

    #[test]
    fn ignores_ordinary_apps() {
        assert!(!path_is_game(r"c:\windows\explorer.exe"));
        assert!(!path_is_game(r"c:\users\me\appdata\local\programs\microsoft vs code\code.exe"));
        assert!(!path_is_game(r"c:\program files\google\chrome\application\chrome.exe"));
        // 'cod' is matched by stem equality, so 'codec.exe' must NOT trip it.
        assert!(!path_is_game(r"c:\windows\system32\codec.exe"));
    }
}
