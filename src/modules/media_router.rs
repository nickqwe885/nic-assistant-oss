use std::process::Command;
use std::os::windows::process::CommandExt;
use tracing::info;

const NO_WINDOW: u32 = 0x08000000;

const KNOWN_PLAYERS: &[&str] = &[
    "spotify.exe", "vlc.exe", "mpv.exe", "aimp.exe", "foobar2000.exe",
    "wmplayer.exe", "musicbee.exe", "winamp.exe", "groove.exe",
];

/// Play a song/video by name.
/// Priority chain:
///   1. mpv with yt-dlp (if mpv.exe is available next to binary or in PATH)
///   2. YouTube search URL in default browser
pub fn play(query: &str) {
    if let Some(mpv) = find_mpv() {
        let ytdl_url = format!("ytdl://ytsearch1:{}", query);
        info!("[MediaRouter] mpv ytdl: {}", query);
        let _ = Command::new(&mpv)
            .args(["--no-video", &ytdl_url])
            .creation_flags(NO_WINDOW)
            .spawn();
        return;
    }

    // Fallback: open YouTube search in browser
    let url = format!(
        "https://www.youtube.com/results?search_query={}",
        urlencoding::encode(query),
    );
    info!("[MediaRouter] browser fallback: {}", query);
    let _ = Command::new("cmd")
        .args(["/c", "start", "", &url])
        .creation_flags(NO_WINDOW)
        .spawn();
}

/// Returns the name of a running media player process, if any.
pub fn detect_active_player() -> Option<String> {
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, false);
    for (_, proc) in sys.processes() {
        let name = proc.name().to_string_lossy().to_lowercase();
        if KNOWN_PLAYERS.iter().any(|p| name.as_str() == *p) {
            return Some(name);
        }
    }
    None
}

fn find_mpv() -> Option<String> {
    // Check next to own binary first
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("mpv.exe");
            if candidate.exists() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }
    // Check PATH
    if Command::new("mpv")
        .arg("--version")
        .creation_flags(NO_WINDOW)
        .output()
        .is_ok()
    {
        return Some("mpv".to_string());
    }
    None
}
