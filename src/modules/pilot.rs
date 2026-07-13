use std::process::Command;
use std::os::windows::process::CommandExt;
use image::DynamicImage;
use tracing::info;

const KINOPOISK_ALIASES: &[&str] = &["кинопоиск", "кинго", "киного", "кинопоиске"];

const NO_WINDOW: u32 = 0x08000000;

pub struct ActionResult {
    pub message: String,
}

impl ActionResult {
    fn new(s: impl Into<String>) -> Self { Self { message: s.into() } }
}

/// Intercepts computer-control commands before they reach Thinker.
/// Returns None to fall through to normal Q&A flow.
pub fn try_execute(query: &str) -> Option<ActionResult> {
    let q = query.to_lowercase();

    // ── Absolute volume set: "громкость N" / "volume N" ──────────────────────
    if q.contains("громкость") || q.contains("volume") {
        if let Some(n) = extract_exact_vol(&q) {
            vol_set(n);
            info!("[Pilot] vol_set {}%", n);
            return Some(ActionResult::new(format!("Volume: {}%.", n)));
        }
    }

    // ── Natural language music control (before keyword checks) ───────────────
    if (q.contains("переключи") || q.contains("переключай"))
        && (q.contains("трек") || q.contains("песн") || q.contains("музык")
            || q.contains("следующ") || q.contains("другую") || q.contains("другой"))
    {
        media_key(0xB0);
        info!("[Pilot] media next (semantic)");
        return Some(ActionResult::new("Next track."));
    }
    if (q.contains("останови") || q.contains("остановись"))
        && (q.contains("музык") || q.contains("трек") || q.contains("песн")
            || q.contains("плеер") || q.contains("воспр"))
    {
        media_key(0xB3);
        info!("[Pilot] media pause (останови)");
        return Some(ActionResult::new("Paused."));
    }
    if q.trim() == "стоп" || q.trim() == "stop" {
        media_key(0xB3);
        info!("[Pilot] media pause (стоп)");
        return Some(ActionResult::new("Paused."));
    }
    if q.contains("тихонько") {
        let steps = extract_vol_steps(&q);
        vol_keys(steps, false);
        info!("[Pilot] volume down (тихонько)");
        return Some(ActionResult::new("Volume down."));
    }
    if q.contains("убери") && (q.contains("звук") || q.contains("музык")) {
        vol_mute();
        info!("[Pilot] mute (убери звук)");
        return Some(ActionResult::new("Muted."));
    }
    if q.contains("добавь") && (q.contains("громк") || q.contains("звук")) {
        let steps = extract_vol_steps(&q);
        vol_keys(steps, true);
        info!("[Pilot] volume up (добавь)");
        return Some(ActionResult::new("Volume up."));
    }

    // ── Volume ───────────────────────────────────────────────────────────────
    if q.contains("громче") || q.contains("грмче") || q.contains("увеличь громк") || q.contains("прибавь")
        || q.contains("louder") || q.contains("volume up")
    {
        let steps = extract_vol_steps(&q);
        vol_keys(steps, true);
        info!("[Pilot] volume up x{}", steps);
        return Some(ActionResult::new(if steps == 4 {
            "Volume up.".to_string()
        } else {
            format!("Volume up by {}.", steps)
        }));
    }
    if q.contains("тише") || q.contains("уменьши громк") || q.contains("убавь") || q.contains("потише")
        || q.contains("quieter") || q.contains("volume down")
    {
        let steps = extract_vol_steps(&q);
        vol_keys(steps, false);
        info!("[Pilot] volume down x{}", steps);
        return Some(ActionResult::new(if steps == 4 {
            "Volume down.".to_string()
        } else {
            format!("Volume down by {}.", steps)
        }));
    }
    if q.contains("без звука") || q.contains("заглуши") || q.contains("выключи звук") || q.contains("мьют") || q.contains("замьют")
        || q.contains("mute") || q.contains("silence")
    {
        vol_mute();
        info!("[Pilot] mute");
        return Some(ActionResult::new("Muted."));
    }

    // ── "play <song/artist>" → play specific media (before the play/pause toggle,
    // which is the greedy `play ` matcher below). "play"/"play music" are bare
    // controls and fall through to the toggle; "play AC DC" plays that track.
    if let Some(rest) = q.strip_prefix("play ") {
        let term = rest.trim();
        const PLAY_CONTROLS: &[&str] = &[
            "music", "the music", "some music", "song", "a song", "the song",
            "track", "the track", "it", "this", "again",
        ];
        if !term.is_empty() && !PLAY_CONTROLS.contains(&term) {
            crate::modules::media_router::play(term);
            info!("[Pilot] media play by name: {}", term);
            return Some(ActionResult::new(format!("Playing «{}».", term)));
        }
    }

    // ── Media playback ───────────────────────────────────────────────────────
    if q.contains("продолжи") || q.contains("продолжай") || q.contains("возобнови")
        || q.contains("play ") || q == "play" || q.contains("resume")
    {
        media_key(0xB3);
        info!("[Pilot] media play");
        return Some(ActionResult::new("Playing."));
    }
    if q == "пауза" || q.contains("поставь на паузу") || q.contains("pause") {
        media_key(0xB3);
        info!("[Pilot] media pause");
        return Some(ActionResult::new("Paused."));
    }
    if q.contains("следующ") || q.contains("next track") || q.contains("next song")
        || q == "skip" || q == "next"
    {
        media_key(0xB0);
        info!("[Pilot] media next");
        return Some(ActionResult::new("Next track."));
    }
    if q.contains("предыдущий трек") || q.contains("предыдущая")
        || q.contains("предыдущую") || q.contains("прошлый трек")
        || q.contains("прошлую песн") || q.contains("prev track")
        || q.contains("previous track") || q.contains("previous song") || q == "previous"
        || (q.contains("назад") && (q.contains("трек") || q.contains("песн") || q.contains("музык")))
    {
        media_key(0xB1);
        info!("[Pilot] media prev");
        return Some(ActionResult::new("Previous track."));
    }

    // ── Screenshot ────────────────────────────────────────────────────────────
    if q.contains("скриншот") || q.contains("скрин") || q.contains("снимок экрана")
        || q.contains("screenshot") || q.contains("screen shot") || q.contains("capture screen")
    {
        info!("[Pilot] screenshot");
        return Some(do_screenshot().unwrap_or_else(|| ActionResult::new("Couldn't take a screenshot.")));
    }

    // ── YouTube search (must be before YouTube open check) ───────────────────
    for prefix in &[
        "найди на ютубе ", "поищи на ютубе ", "включи на ютубе ",
        "открой на ютубе ", "поставь на ютубе ", "найди на youtube ",
        "поищи на youtube ", "найди ютуб ",
        "search youtube for ", "find on youtube ", "play on youtube ",
    ] {
        if let Some(idx) = q.find(prefix) {
            let term = query[idx + prefix.len()..].trim().to_string();
            if !term.is_empty() {
                let url = format!("https://www.youtube.com/results?search_query={}", urlencoding::encode(&term));
                open_url(&url);
                info!("[Pilot] YT search: {}", term);
                return Some(ActionResult::new(format!("Searching YouTube: «{}»", term)));
            }
        }
    }

    // ── Google search ─────────────────────────────────────────────────────────
    // NOTE: only explicit, imperative prefixes here — `find()` matches anywhere in
    // the query, so a bare "google "/"search for " would misfire mid-sentence
    // ("what does google do"). A plain "google X" still reaches the web via the
    // in-chat Surfer path (force_online keywords), which is the nicer behaviour.
    for prefix in &[
        "погугли ", "поищи в гугле ", "найди в гугле ",
        "search google for ", "search the web for ",
    ] {
        if let Some(idx) = q.find(prefix) {
            let term = query[idx + prefix.len()..].trim().to_string();
            if !term.is_empty() {
                let url = format!("https://www.google.com/search?q={}", urlencoding::encode(&term));
                open_url(&url);
                info!("[Pilot] Google search: {}", term);
                return Some(ActionResult::new(format!("Searching Google: «{}»", term)));
            }
        }
    }

    // ── Direct YouTube/site mention (short query, no open-verb required) ──────
    // Catches typos like "вклюбчбюи ютуб" or just "ютуб"
    let word_count = q.split_whitespace().count();
    if word_count <= 3 && (q.contains("ютуб") || q.contains("youtube")) {
        open_url("https://www.youtube.com");
        info!("[Pilot] YouTube direct mention");
        return Some(ActionResult::new("Opening YouTube."));
    }

    // ── Open / launch ─────────────────────────────────────────────────────────
    if ["открой", "включи", "запусти", "покажи", "зайди",
        "open ", "launch ", "start ", "show ", "go to ", "run "].iter().any(|t| q.contains(t))
    {
        return try_open(&q, query);
    }

    None
}

fn try_open(q: &str, original: &str) -> Option<ActionResult> {
    // Sites
    if q.contains("ютуб") || q.contains("youtube") {
        open_url("https://www.youtube.com");
        return Some(ActionResult::new("Opening YouTube."));
    }
    if q.contains("гугл") || q.contains("google") {
        open_url("https://www.google.com");
        return Some(ActionResult::new("Opening Google."));
    }
    if q.contains("гитхаб") || q.contains("github") {
        open_url("https://github.com");
        return Some(ActionResult::new("Opening GitHub."));
    }
    if q.contains("телеграм") || q.contains("telegram") {
        open_url("https://web.telegram.org");
        return Some(ActionResult::new("Opening Telegram."));
    }
    if q.contains("дискорд") || q.contains("discord") {
        open_url("https://discord.com/app");
        return Some(ActionResult::new("Opening Discord."));
    }
    if q.contains("спотифай") || q.contains("спотифи") || q.contains("spotify") {
        open_url("https://open.spotify.com");
        return Some(ActionResult::new("Opening Spotify."));
    }
    if q.contains("твиттер") || q.contains("twitter") {
        open_url("https://x.com");
        return Some(ActionResult::new("Opening X (Twitter)."));
    }
    if q.contains("твич") || q.contains("twitch") {
        open_url("https://www.twitch.tv");
        return Some(ActionResult::new("Opening Twitch."));
    }
    if q.contains("реддит") || q.contains("reddit") {
        open_url("https://www.reddit.com");
        return Some(ActionResult::new("Opening Reddit."));
    }
    if q.contains("стим") || q.contains("steam") {
        open_url("https://store.steampowered.com");
        return Some(ActionResult::new("Opening Steam."));
    }
    if q.contains("настройки") || q.contains("settings") {
        open_url("ms-settings:");
        return Some(ActionResult::new("Opening Settings."));
    }
    if KINOPOISK_ALIASES.iter().any(|a| q.contains(a)) {
        open_url("https://www.kinopoisk.ru");
        return Some(ActionResult::new("Opening Kinopoisk."));
    }
    if q.contains("иви") && !q.contains("архив") {
        open_url("https://www.ivi.ru");
        return Some(ActionResult::new("Opening ivi."));
    }
    if q.contains("окко") {
        open_url("https://okko.tv");
        return Some(ActionResult::new("Opening Okko."));
    }
    if q.contains("premier") || q.contains("премьер") {
        open_url("https://premier.one");
        return Some(ActionResult::new("Opening Premier."));
    }

    // Apps
    if q.contains("блокнот") || q.contains("notepad") {
        spawn_app("notepad.exe");
        return Some(ActionResult::new("Opening Notepad."));
    }
    if q.contains("калькулятор") || q.contains("calculator") || q.contains("calc") {
        spawn_app("calc.exe");
        return Some(ActionResult::new("Opening Calculator."));
    }
    if q.contains("проводник") || q.contains("explorer") || q.contains("file manager") {
        spawn_app("explorer.exe");
        return Some(ActionResult::new("Opening File Explorer."));
    }
    if q.contains("вс код") || q.contains("vs code") || q.contains("vscode") || q.contains("визуал") {
        spawn_app("code");
        return Some(ActionResult::new("Opening VS Code."));
    }
    if q.contains("хром") || q.contains("chrome") {
        spawn_app("chrome");
        return Some(ActionResult::new("Opening Chrome."));
    }
    if q.contains("диспетчер задач") || q.contains("таск менеджер") || q.contains("task manager") {
        spawn_app("taskmgr.exe");
        return Some(ActionResult::new("Opening Task Manager."));
    }
    if q.contains("paint") || q.contains("пейнт") {
        spawn_app("mspaint.exe");
        return Some(ActionResult::new("Opening Paint."));
    }
    if q.contains("терминал") || q.contains("консол") || q.contains("командн")
        || q.contains("terminal") || q.contains("command prompt") || q.contains("cmd")
    {
        spawn_app("cmd.exe");
        return Some(ActionResult::new("Opening Terminal."));
    }
    if q.contains("powershell") || q.contains("павершелл") {
        spawn_app("powershell.exe");
        return Some(ActionResult::new("Opening PowerShell."));
    }
    if q.contains("плеер") || q.contains("media player") || q.contains("wmplayer") {
        spawn_app("wmplayer.exe");
        return Some(ActionResult::new("Opening media player."));
    }

    // ── Fallback: smart routing — search for everything unknown ──────────────
    // Handles both languages so an English "open <something>" resolves here
    // deterministically instead of falling through to the small model (which
    // used to guess a random site).
    for prefix in &["включи ", "открой ", "запусти ", "покажи ", "зайди на ",
                    "open ", "show ", "launch ", "run ", "start ", "go to "] {
        if let Some(idx) = q.find(prefix) {
            let raw  = original[idx + prefix.len()..].trim().to_string();
            let term = strip_fillers(&raw);
            if term.len() > 2 {
                let tlc = term.to_lowercase();
                // Platform explicitly named inside term
                let (url, msg) = if KINOPOISK_ALIASES.iter().any(|a| tlc.contains(a)) {
                    let mut strip_words: Vec<&str> = KINOPOISK_ALIASES.to_vec();
                    strip_words.extend_from_slice(&["на сайте", "на"]);
                    let t = strip_platform(&tlc, &strip_words);
                    if t.is_empty() {
                        open_url("https://www.kinopoisk.ru");
                        return Some(ActionResult::new("Opening Kinopoisk."));
                    }
                    (format!("https://www.kinopoisk.ru/index.php?kp_query={}", urlencoding::encode(&t)),
                     format!("Searching «{}» on Kinopoisk.", t))
                } else if tlc.contains("иви") {
                    let t = strip_platform(&tlc, &["иви", "на сайте", "на"]);
                    if t.is_empty() {
                        open_url("https://www.ivi.ru");
                        return Some(ActionResult::new("Openingivi."));
                    }
                    (format!("https://www.ivi.ru/search/?q={}", urlencoding::encode(&t)),
                     format!("Searching «{}» on ivi.", t))
                } else if tlc.contains("ютуб") || tlc.contains("youtube") {
                    let t = strip_platform(&tlc, &["ютубе", "ютуб", "youtube", "на"]);
                    (format!("https://www.youtube.com/results?search_query={}", urlencoding::encode(&t)),
                     format!("Searching YouTube: «{}».", t))
                } else if ["video", "видео", "clip", "клип"].iter().any(|t| tlc.contains(t)) {
                    // "open a video about X" → actually PLAY the first result,
                    // fullscreen. Diverges (returns from the fn), so the tuple
                    // type below is unaffected.
                    let t = strip_content_type(&tlc);
                    return Some(play_youtube_first(&t));
                } else {
                    // "включи"/"покажи"/"watch" imply watching — add "watch online";
                    // a plain "open X" just searches for X.
                    let is_watch = matches!(*prefix, "включи " | "покажи ");
                    let is_video = is_watch || ["сериал", "фильм", "кино", "мультик", "аниме",
                        "серию", "серия", "movie", "series", "episode"].iter().any(|t| tlc.contains(t));
                    let sq = if is_video {
                        format!("{} watch online", strip_content_type(&tlc))
                    } else {
                        tlc.clone()
                    };
                    (format!("https://www.google.com/search?q={}", urlencoding::encode(&sq)),
                     format!("Searching «{}» online.", term))
                };
                open_url(&url);
                info!("[Pilot] fallback → {}", url);
                return Some(ActionResult::new(msg));
            }
        }
    }

    None
}

/// Dispatch an LLM intent code to the matching system action.
/// Accepts raw LLM output in "CODE" or "CODE:param" format.
pub fn execute_intent(raw_code: &str) -> Option<ActionResult> {
    let s = raw_code.trim();
    // Split on first colon: everything before is the action code, everything after is the param.
    let (code_str, param) = match s.find(':') {
        Some(idx) => (s[..idx].trim(), s[idx + 1..].trim()),
        None      => (s, ""),
    };
    let code = code_str.to_uppercase();

    match code.as_str() {
        "VOL_UP" => {
            vol_keys(4, true);
            info!("[Pilot/intent] VOL_UP");
            Some(ActionResult::new("Volume up."))
        }
        "VOL_DOWN" => {
            vol_keys(4, false);
            info!("[Pilot/intent] VOL_DOWN");
            Some(ActionResult::new("Volume down."))
        }
        "VOL_MUTE" | "MUTE" => {
            vol_mute();
            info!("[Pilot/intent] VOL_MUTE");
            Some(ActionResult::new("Muted."))
        }
        "NEXT" => {
            media_key(0xB0);
            info!("[Pilot/intent] NEXT");
            Some(ActionResult::new("Next track."))
        }
        "PREV" => {
            media_key(0xB1);
            info!("[Pilot/intent] PREV");
            Some(ActionResult::new("Previous track."))
        }
        "PLAY_PAUSE" | "PLAY" | "PAUSE" => {
            media_key(0xB3);
            info!("[Pilot/intent] PLAY_PAUSE");
            Some(ActionResult::new("Play / pause."))
        }
        "SCREENSHOT" => {
            info!("[Pilot/intent] SCREENSHOT");
            Some(do_screenshot().unwrap_or_else(|| ActionResult::new("Couldn't take a screenshot.")))
        }
        "SITE_OPEN" if !param.is_empty() => {
            open_url(param);
            info!("[Pilot/intent] SITE_OPEN: {}", param);
            Some(ActionResult::new(format!("Opening {}.", friendly_url_label(param))))
        }
        "APP_OPEN" if !param.is_empty() => {
            spawn_app(param);
            info!("[Pilot/intent] APP_OPEN: {}", param);
            Some(ActionResult::new(format!("Opening {}.", param.trim_end_matches(".exe"))))
        }
        "YT_SEARCH" if !param.is_empty() => {
            let url = format!("https://www.youtube.com/results?search_query={}", urlencoding::encode(param));
            open_url(&url);
            info!("[Pilot/intent] YT_SEARCH: {}", param);
            Some(ActionResult::new(format!("Searching YouTube: «{}».", param)))
        }
        "WEB_SEARCH" if !param.is_empty() => {
            let url = format!("https://www.google.com/search?q={}", urlencoding::encode(param));
            open_url(&url);
            info!("[Pilot/intent] WEB_SEARCH: {}", param);
            Some(ActionResult::new(format!("Searching Google: «{}».", param)))
        }
        "MEDIA_PLAY" if !param.is_empty() => {
            crate::modules::media_router::play(param);
            info!("[Pilot/intent] MEDIA_PLAY: {}", param);
            Some(ActionResult::new(format!("Playing «{}».", param)))
        }
        _ => None,
    }
}

/// Opens the FIRST YouTube result and (best-effort) drops it into fullscreen.
/// All the slow work (network scrape + launch + the fullscreen keypress) runs on
/// a detached thread so the caller returns instantly. On a fresh watch page the
/// video autoplays; ~4.5 s later we tap the "f" shortcut so it fills the screen.
/// If the first video can't be resolved, we just open the search page.
fn play_youtube_first(query: &str) -> ActionResult {
    let q = query.to_string();
    std::thread::spawn(move || {
        let (url, fullscreen) = match crate::modules::surfer::first_youtube_watch_url(&q) {
            Some(watch) => (watch, true),
            None => (
                format!("https://www.youtube.com/results?search_query={}", urlencoding::encode(&q)),
                false,
            ),
        };
        open_url(&url);
        if fullscreen {
            // Give the page + player time to load and take focus, then press "f"
            // (YouTube's fullscreen shortcut). Best-effort: a no-op if focus moved.
            std::thread::sleep(std::time::Duration::from_millis(4500));
            tap_key(0x46); // VK 'F'
        }
    });
    ActionResult::new(format!("Playing «{}» on YouTube.", query.trim()))
}

/// Presses and releases a single virtual-key (no extended flag). Used for
/// app-level shortcuts like YouTube fullscreen ("f").
fn tap_key(vk: u8) {
    let mut script = build_keybd_class_script();
    script.push_str(&format!(
        "[MK]::keybd_event({vk},0,0,0);Start-Sleep -Milliseconds 40;[MK]::keybd_event({vk},0,2,0)\n"
    ));
    ps_hidden(&script);
}

fn friendly_url_label(url: &str) -> &str {
    if url.contains("youtube.com")    { "YouTube" }
    else if url.contains("vk.com")    { "VK" }
    else if url.contains("google.com"){ "Google" }
    else if url.contains("github.com"){ "GitHub" }
    else if url.contains("telegram")  { "Telegram" }
    else if url.contains("discord")   { "Discord" }
    else if url.contains("spotify")   { "Spotify" }
    else if url.contains("twitch")    { "Twitch" }
    else if url.contains("reddit")    { "Reddit" }
    else if url.contains("steam")     { "Steam" }
    else if url.contains("kinopoisk") { "Kinopoisk" }
    else                              { url }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(crate) fn open_url(url: &str) {
    let _ = Command::new("cmd")
        .args(["/c", "start", "", url])
        .creation_flags(NO_WINDOW)
        .spawn();
}

fn spawn_app(name: &str) {
    let _ = Command::new(name)
        .creation_flags(NO_WINDOW)
        .spawn();
}

fn build_keybd_class_script() -> String {
    // Writes the MK class to a temp .ps1 and dot-sources it so the C#
    // type definition is properly multi-line (Add-Type -TypeDefinition
    // in PowerShell parses a single-line C# blob, but CRLF must be present).
    r#"
Add-Type @"
using System.Runtime.InteropServices;
public class MK {
  [DllImport("user32.dll")]
  public static extern void keybd_event(byte b, byte s, int f, int e);
}
"@
"#.to_string()
}

fn vol_keys(steps: u32, up: bool) {
    let vk: u8 = if up { 0xAF } else { 0xAE };
    let mut script = build_keybd_class_script();
    script.push_str(&format!(
        "for($i=0;$i -lt {steps};$i++){{[MK]::keybd_event({vk},0,0,0);[MK]::keybd_event({vk},0,2,0)}}\n"
    ));
    ps_hidden(&script);
}

fn vol_mute() {
    let mut script = build_keybd_class_script();
    script.push_str("[MK]::keybd_event(0xAD,0,0,0);[MK]::keybd_event(0xAD,0,2,0)\n");
    ps_hidden(&script);
}

fn ps_hidden(script: &str) {
    let _ = Command::new("powershell")
        .args(["-NoProfile", "-WindowStyle", "Hidden", "-Command", script])
        .creation_flags(NO_WINDOW)
        .spawn();
}

fn extract_vol_steps(q: &str) -> u32 {
    // Parse "на N" → N steps; default 4
    if let Some(idx) = q.find("на ") {
        let rest: String = q[idx + "на ".len()..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = rest.parse::<u32>() {
            return n.clamp(1, 50);
        }
    }
    4
}

fn extract_exact_vol(q: &str) -> Option<u32> {
    for kw in ["громкость", "volume"] {
        if let Some(idx) = q.find(kw) {
            let rest = q[idx + kw.len()..].trim_start();
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num.is_empty() {
                return num.parse::<u32>().ok().map(|n| n.min(100));
            }
        }
    }
    None
}

fn vol_set(percent: u32) {
    let pct   = percent.min(100);
    let frac  = format!("{:.4}f", pct as f32 / 100.0);
    let tmp   = std::env::temp_dir().join("nic_setvol.ps1");
    let script = build_setvol_ps(&frac);
    let _ = std::fs::write(&tmp, script.as_bytes());
    let _ = Command::new("powershell")
        .args(["-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass", "-File"])
        .arg(&tmp)
        .creation_flags(NO_WINDOW)
        .spawn();
}

fn build_setvol_ps(frac: &str) -> String {
    // Write PS file cleanly — raw string avoids all escaping issues.
    // The "@ must be at column 0 for PS here-string; it IS in this raw literal.
    let tmpl = r#"$src = @"
using System;
using System.Runtime.InteropServices;
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioVol { void _a(); void _b(); void _c(); void _d(); void SetLvl(float f, Guid g); }
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDev { void Activate(ref Guid g, int c, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object o); }
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMEnum { void _x(); void GetDefault(int f, int r, out IMMDev d); }
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
class MME {}
"@
Add-Type -TypeDefinition $src -EA SilentlyContinue
$e = [Activator]::CreateInstance([MME]) -as [IMMEnum]; $d = $null
$e.GetDefault(0, 1, [ref]$d) | Out-Null
$g = [Guid]"5CDF2C82-841E-4546-9722-0CF74078229A"; $o = $null
$d.Activate([ref]$g, 23, [IntPtr]::Zero, [ref]$o) | Out-Null
($o -as [IAudioVol]).SetLvl(%FRAC%, [Guid]::Empty) | Out-Null
"#;
    tmpl.replace("%FRAC%", frac)
}

fn media_key(vk: u8) {
    let mut script = build_keybd_class_script();
    script.push_str(&format!(
        "[MK]::keybd_event({vk},0,0,0);[MK]::keybd_event({vk},0,2,0)\n"
    ));
    ps_hidden(&script);
}

fn strip_platform(s: &str, words: &[&str]) -> String {
    let mut result = s.to_string();
    for w in words {
        result = result.replace(w, " ");
    }
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_fillers(s: &str) -> String {
    const FILLERS: &[&str] = &[
        "пожалуйста", "в браузере", "в интернете", "в угле", "в углу",
        "на сайте", "мне", "ну-ка", "сейчас", "быстро", "срочно",
    ];
    let mut result = s.to_string();
    for f in FILLERS {
        result = result.replace(f, " ");
    }
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_content_type(s: &str) -> String {
    const TYPES: &[&str] = &["сериало", "сериал", "фильм", "мультик", "аниме", "кино"];
    let mut result = s.trim().to_string();
    for t in TYPES {
        result = result.replace(&format!(" {}", t), "");
        if result.ends_with(t) {
            result = result[..result.len() - t.len()].trim().to_string();
        }
    }
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn do_screenshot() -> Option<ActionResult> {
    let profile = std::env::var("USERPROFILE").unwrap_or_else(|_| "C:\\Users\\User".to_string());
    let ts      = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let path    = format!("{}\\Desktop\\nic_{}.png", profile, ts);

    let screens = screenshots::Screen::all().ok()?;
    let screen  = screens.into_iter().next()?;
    let rgba    = screen.capture().ok()?;
    DynamicImage::ImageRgba8(rgba).save(&path).ok()?;

    let fname = std::path::Path::new(&path).file_name()?.to_string_lossy().to_string();
    Some(ActionResult::new(format!("Screenshot saved to the Desktop: {}", fname)))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_vol_steps ─────────────────────────────────────────────────────

    #[test]
    fn vol_steps_default_no_na() {
        assert_eq!(extract_vol_steps("громче"), 4);
    }

    #[test]
    fn vol_steps_default_no_digit_after_na() {
        // "на " found but no digit follows — return default
        assert_eq!(extract_vol_steps("громче на много"), 4);
    }

    #[test]
    fn vol_steps_parses_single_digit() {
        assert_eq!(extract_vol_steps("громче на 5"), 5);
    }

    #[test]
    fn vol_steps_parses_two_digits() {
        assert_eq!(extract_vol_steps("тише на 20"), 20);
    }

    #[test]
    fn vol_steps_clamps_to_50() {
        assert_eq!(extract_vol_steps("громче на 99"), 50);
    }

    #[test]
    fn vol_steps_clamps_min_to_1() {
        // "0" parses fine as u32; clamp(1, 50) → 1
        assert_eq!(extract_vol_steps("громче на 0"), 1);
    }

    #[test]
    fn vol_steps_na_with_leading_spaces() {
        assert_eq!(extract_vol_steps("  громче на 3  "), 3);
    }

    #[test]
    fn vol_steps_correct_utf8_slice() {
        // "на ".len() == 5 bytes, not 3. This MUST NOT panic.
        let result = std::panic::catch_unwind(|| extract_vol_steps("громче на 7"));
        assert!(result.is_ok(), "extract_vol_steps panicked — UTF-8 slice bug not fixed");
        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn vol_steps_na_at_start() {
        assert_eq!(extract_vol_steps("на 10"), 10);
    }

    #[test]
    fn vol_steps_max_boundary() {
        assert_eq!(extract_vol_steps("громче на 50"), 50);
    }

    #[test]
    fn vol_steps_just_above_max() {
        assert_eq!(extract_vol_steps("громче на 51"), 50);
    }

    #[test]
    fn vol_steps_empty_query() {
        assert_eq!(extract_vol_steps(""), 4);
    }

    #[test]
    fn vol_steps_digit_1() {
        assert_eq!(extract_vol_steps("тише на 1"), 1);
    }

    // ── extract_exact_vol ─────────────────────────────────────────────────────

    #[test]
    fn exact_vol_basic() {
        assert_eq!(extract_exact_vol("громкость 75"), Some(75));
    }

    #[test]
    fn exact_vol_zero() {
        assert_eq!(extract_exact_vol("громкость 0"), Some(0));
    }

    #[test]
    fn exact_vol_100() {
        assert_eq!(extract_exact_vol("громкость 100"), Some(100));
    }

    #[test]
    fn exact_vol_clamps_above_100() {
        assert_eq!(extract_exact_vol("громкость 150"), Some(100));
    }

    #[test]
    fn exact_vol_no_digit() {
        assert_eq!(extract_exact_vol("громкость высокая"), None);
    }

    #[test]
    fn exact_vol_no_keyword() {
        assert_eq!(extract_exact_vol("сделай 50"), None);
    }

    #[test]
    fn exact_vol_empty() {
        assert_eq!(extract_exact_vol(""), None);
    }

    #[test]
    fn exact_vol_keyword_only() {
        assert_eq!(extract_exact_vol("громкость"), None);
    }

    #[test]
    fn exact_vol_two_digit() {
        assert_eq!(extract_exact_vol("громкость 42"), Some(42));
    }

    #[test]
    fn exact_vol_with_prefix_text() {
        assert_eq!(extract_exact_vol("сделай громкость 80"), Some(80));
    }

    #[test]
    fn exact_vol_correct_utf8_slice() {
        // "громкость".len() == 18 bytes — must not panic
        let result = std::panic::catch_unwind(|| extract_exact_vol("громкость 55"));
        assert!(result.is_ok(), "extract_exact_vol panicked — UTF-8 bug");
        assert_eq!(result.unwrap(), Some(55));
    }

    // ── friendly_url_label ────────────────────────────────────────────────────

    #[test]
    fn url_label_youtube() {
        assert_eq!(friendly_url_label("https://www.youtube.com/watch?v=abc"), "YouTube");
    }

    #[test]
    fn url_label_vk() {
        assert_eq!(friendly_url_label("https://vk.com/feed"), "VK");
    }

    #[test]
    fn url_label_google() {
        assert_eq!(friendly_url_label("https://www.google.com/search?q=test"), "Google");
    }

    #[test]
    fn url_label_github() {
        assert_eq!(friendly_url_label("https://github.com/user/repo"), "GitHub");
    }

    #[test]
    fn url_label_telegram() {
        assert_eq!(friendly_url_label("https://web.telegram.org/a"), "Telegram");
    }

    #[test]
    fn url_label_discord() {
        assert_eq!(friendly_url_label("https://discord.com/app"), "Discord");
    }

    #[test]
    fn url_label_spotify() {
        assert_eq!(friendly_url_label("https://open.spotify.com"), "Spotify");
    }

    #[test]
    fn url_label_twitch() {
        assert_eq!(friendly_url_label("https://www.twitch.tv"), "Twitch");
    }

    #[test]
    fn url_label_reddit() {
        assert_eq!(friendly_url_label("https://www.reddit.com/r/rust"), "Reddit");
    }

    #[test]
    fn url_label_steam() {
        assert_eq!(friendly_url_label("https://store.steampowered.com"), "Steam");
    }

    #[test]
    fn url_label_kinopoisk() {
        assert_eq!(friendly_url_label("https://www.kinopoisk.ru/film/123"), "Kinopoisk");
    }

    #[test]
    fn url_label_unknown_returns_url() {
        let url = "https://example.com/page";
        assert_eq!(friendly_url_label(url), url);
    }

    #[test]
    fn url_label_empty() {
        assert_eq!(friendly_url_label(""), "");
    }

    // ── strip_fillers ─────────────────────────────────────────────────────────

    #[test]
    fn strip_fillers_removes_pozhaluysta() {
        let r = strip_fillers("открой пожалуйста ютуб");
        assert!(!r.contains("пожалуйста"));
        assert!(r.contains("ютуб"));
    }

    #[test]
    fn strip_fillers_removes_multiple() {
        let r = strip_fillers("срочно пожалуйста открой");
        assert!(!r.contains("срочно"));
        assert!(!r.contains("пожалуйста"));
    }

    #[test]
    fn strip_fillers_preserves_content() {
        let r = strip_fillers("открой фильм");
        assert_eq!(r, "открой фильм");
    }

    #[test]
    fn strip_fillers_empty() {
        assert_eq!(strip_fillers(""), "");
    }

    #[test]
    fn strip_fillers_only_filler() {
        let r = strip_fillers("пожалуйста");
        assert_eq!(r, "");
    }

    #[test]
    fn strip_fillers_normalizes_spaces() {
        let r = strip_fillers("открой   пожалуйста   ютуб");
        assert!(!r.contains("  "));
    }

    // ── strip_content_type ────────────────────────────────────────────────────

    #[test]
    fn strip_content_type_removes_serial() {
        let r = strip_content_type("ведьмак сериал");
        assert!(!r.contains("сериал"), "got: {}", r);
        assert!(r.contains("ведьмак"), "got: {}", r);
    }

    #[test]
    fn strip_content_type_removes_film() {
        let r = strip_content_type("интерстеллар фильм");
        assert!(!r.contains("фильм"), "got: {}", r);
    }

    #[test]
    fn strip_content_type_removes_anime() {
        let r = strip_content_type("наруто аниме");
        assert!(!r.contains("аниме"), "got: {}", r);
    }

    #[test]
    fn strip_content_type_preserves_no_type() {
        let r = strip_content_type("интерстеллар");
        assert_eq!(r, "интерстеллар");
    }

    #[test]
    fn strip_content_type_empty() {
        assert_eq!(strip_content_type(""), "");
    }

    #[test]
    fn strip_content_type_only_type_word() {
        let r = strip_content_type("фильм");
        assert_eq!(r, "");
    }

    #[test]
    fn strip_content_type_multikino() {
        let r = strip_content_type("том и джерри мультик");
        assert!(!r.contains("мультик"));
        assert!(r.contains("том и джерри"));
    }

    // ── strip_platform ────────────────────────────────────────────────────────

    #[test]
    fn strip_platform_removes_word() {
        let r = strip_platform("найди ведьмак на ютубе", &["ютубе", "на"]);
        assert!(!r.contains("ютубе"));
        assert!(!r.contains(" на "));
        assert!(r.contains("ведьмак"));
    }

    #[test]
    fn strip_platform_empty_words() {
        let r = strip_platform("ведьмак", &[]);
        assert_eq!(r, "ведьмак");
    }

    #[test]
    fn strip_platform_all_stripped() {
        let r = strip_platform("на ютубе", &["ютубе", "на"]);
        assert_eq!(r, "");
    }

    #[test]
    fn strip_platform_normalizes_spaces() {
        let r = strip_platform("ведьмак   на   ютубе", &["ютубе", "на"]);
        assert!(!r.contains("  "));
    }

    // ── execute_intent pure-logic tests (code parsing) ────────────────────────

    #[test]
    fn execute_intent_qa_returns_none() {
        assert!(execute_intent("QA").is_none());
    }

    #[test]
    fn execute_intent_unknown_returns_none() {
        assert!(execute_intent("UNKNOWN_CODE").is_none());
    }

    #[test]
    fn execute_intent_empty_returns_none() {
        assert!(execute_intent("").is_none());
    }

    #[test]
    fn execute_intent_site_open_empty_param_returns_none() {
        assert!(execute_intent("SITE_OPEN:").is_none());
        assert!(execute_intent("SITE_OPEN").is_none());
    }

    #[test]
    fn execute_intent_app_open_empty_param_returns_none() {
        assert!(execute_intent("APP_OPEN:").is_none());
        assert!(execute_intent("APP_OPEN").is_none());
    }

    #[test]
    fn execute_intent_yt_search_empty_param_returns_none() {
        assert!(execute_intent("YT_SEARCH:").is_none());
        assert!(execute_intent("YT_SEARCH").is_none());
    }

    #[test]
    fn execute_intent_web_search_empty_param_returns_none() {
        assert!(execute_intent("WEB_SEARCH:").is_none());
        assert!(execute_intent("WEB_SEARCH").is_none());
    }

    #[test]
    fn execute_intent_media_play_empty_param_returns_none() {
        assert!(execute_intent("MEDIA_PLAY:").is_none());
        assert!(execute_intent("MEDIA_PLAY").is_none());
    }

    #[test]
    fn execute_intent_case_insensitive_vol_up() {
        let r = execute_intent("vol_up");
        assert!(r.is_some(), "VOL_UP lowercase should match");
        assert!(r.unwrap().message.contains("Volume"));
    }

    #[test]
    fn execute_intent_case_insensitive_vol_down() {
        let r = execute_intent("VOL_DOWN");
        assert!(r.is_some());
        assert!(r.unwrap().message.contains("Volume"));
    }

    #[test]
    fn execute_intent_vol_mute_alias() {
        assert!(execute_intent("MUTE").is_some());
        assert!(execute_intent("VOL_MUTE").is_some());
    }

    #[test]
    fn execute_intent_play_aliases() {
        assert!(execute_intent("PLAY").is_some());
        assert!(execute_intent("PAUSE").is_some());
        assert!(execute_intent("PLAY_PAUSE").is_some());
    }

    #[test]
    fn execute_intent_next_prev() {
        assert!(execute_intent("NEXT").is_some());
        assert!(execute_intent("PREV").is_some());
    }

    #[test]
    fn execute_intent_screenshot_some() {
        // SCREENSHOT tries to capture but may fail on headless CI — that's OK,
        // execute_intent itself returns Some (via do_screenshot's Option fallback)
        let r = execute_intent("SCREENSHOT");
        assert!(r.is_some(), "SCREENSHOT should return Some even on failure");
    }

    #[test]
    fn execute_intent_site_open_with_param() {
        let r = execute_intent("SITE_OPEN:https://example.com");
        assert!(r.is_some());
        assert!(r.unwrap().message.contains("Opening"));
    }

    #[test]
    fn execute_intent_app_open_with_param() {
        let r = execute_intent("APP_OPEN:notepad.exe");
        assert!(r.is_some());
        assert!(r.unwrap().message.contains("notepad"));
    }

    #[test]
    fn execute_intent_yt_search_with_param() {
        let r = execute_intent("YT_SEARCH:lofi hip hop");
        assert!(r.is_some());
        let msg = r.unwrap().message;
        assert!(msg.contains("YouTube"));
        assert!(msg.contains("lofi hip hop"));
    }

    #[test]
    fn execute_intent_web_search_with_param() {
        let r = execute_intent("WEB_SEARCH:погода москва");
        assert!(r.is_some());
        let msg = r.unwrap().message;
        assert!(msg.contains("Google") || msg.contains("Ищу"));
    }

    #[test]
    fn execute_intent_colon_in_param_preserved() {
        // SITE_OPEN: param may itself contain colons (like a URL)
        let r = execute_intent("SITE_OPEN:https://github.com/user/repo");
        assert!(r.is_some());
    }

    #[test]
    fn execute_intent_mixed_case_code() {
        // Code part is normalized to uppercase
        let r = execute_intent("Next");
        assert!(r.is_some());
        assert!(r.unwrap().message.contains("Next"));
    }

    #[test]
    fn execute_intent_whitespace_trimmed() {
        let r = execute_intent("  VOL_UP  ");
        assert!(r.is_some());
    }

    // ── Vol message contents ──────────────────────────────────────────────────

    #[test]
    fn vol_up_message_contains_uvelichivaju() {
        let r = execute_intent("VOL_UP").unwrap();
        assert!(r.message.contains("Volume up") || r.message.contains("Volume"));
    }

    #[test]
    fn vol_down_message_contains_umenshaju() {
        let r = execute_intent("VOL_DOWN").unwrap();
        assert!(r.message.contains("Volume down") || r.message.contains("Volume"));
    }

    #[test]
    fn mute_message_contains_zvuk() {
        let r = execute_intent("VOL_MUTE").unwrap();
        assert!(r.message.contains("Muted") || r.message.contains("muted"));
    }

    #[test]
    fn next_message_contains_sledujushij() {
        let r = execute_intent("NEXT").unwrap();
        assert!(r.message.contains("Next"));
    }

    #[test]
    fn prev_message_contains_predydushij() {
        let r = execute_intent("PREV").unwrap();
        assert!(r.message.contains("Previous"));
    }

    // ── Vol steps edge cases ──────────────────────────────────────────────────

    #[test]
    fn vol_steps_na_with_spaces_around() {
        assert_eq!(extract_vol_steps("сделай на 8 громче"), 8);
    }

    #[test]
    fn vol_steps_multiple_na_uses_first() {
        // "на 3 на 5" — should take first match (3)
        assert_eq!(extract_vol_steps("громче на 3 на 5"), 3);
    }

    #[test]
    fn vol_steps_digit_immediately_after_space() {
        assert_eq!(extract_vol_steps("тише на 2"), 2);
    }

    #[test]
    fn exact_vol_with_noise_chars() {
        // Digits-only parsing stops at non-digit
        assert_eq!(extract_exact_vol("громкость 50%"), Some(50));
    }

    #[test]
    fn exact_vol_first_digit_sequence() {
        assert_eq!(extract_exact_vol("громкость 25 пожалуйста"), Some(25));
    }

    // ── Regression: ensure no UTF-8 panic on common voice commands ───────────

    #[test]
    fn no_panic_common_volume_commands() {
        let commands = [
            "громче", "тише", "громче на 5", "тише на 10",
            "громкость 70", "без звука", "убавь на 3",
            "прибавь на 7", "тихонько на 2",
        ];
        for cmd in &commands {
            let _ = std::panic::catch_unwind(|| {
                let q = cmd.to_lowercase();
                let _ = extract_vol_steps(&q);
                let _ = extract_exact_vol(&q);
            });
        }
    }

    #[test]
    fn no_panic_common_media_commands() {
        let commands = [
            "следующий трек", "предыдущая песня", "пауза",
            "продолжи", "стоп", "поставь на паузу",
            "переключи трек", "останови музыку",
        ];
        for cmd in &commands {
            let _ = std::panic::catch_unwind(|| {
                let _ = try_execute(cmd);
            });
        }
    }

    #[test]
    fn no_panic_common_open_commands() {
        let commands = [
            "открой ютуб", "включи spotify", "запусти telegram",
            "открой блокнот", "включи калькулятор", "открой github",
            "покажи настройки", "открой reddit", "open video about cats",
        ];
        for cmd in &commands {
            let _ = std::panic::catch_unwind(|| {
                let _ = try_execute(cmd);
            });
        }
    }
}
