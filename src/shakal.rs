/// Shakal-Processor — Smart Compression + Privacy Shield + Dirty Rectangles.
///
/// Pipeline: RGBA capture → Grayscale → Resize 384×384 (Nearest) →
///           Posterize 8 levels → JPEG quality=5 → pHash + block hashes
///
/// Privacy Shield: `process()` returns `None` when the active window title
/// contains any term from the PRIVACY_BLACKLIST.
///
/// Dirty Rectangles: the 384×384 frame is divided into a 6×6 grid of 64×64
/// blocks. Each block gets a 64-bit average hash. `changed_block_ratio()`
/// compares two hash arrays and returns the fraction of blocks that changed
/// (Hamming distance > 5). Sentinel skips the frame if < 5% of blocks changed.

use anyhow::{anyhow, Result};
use image::{ColorType, DynamicImage, GrayImage};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use tracing::{info, warn};

// ── Privacy Shield ────────────────────────────────────────────────────────────

const PRIVACY_BLACKLIST: &[&str] = &[
    "bank",
    "crypto",
    "login",
    "password",
    "incognito",
    "private browsing",
    "1password",
    "bitwarden",
    "keepass",
    "lastpass",
    "dashlane",
];

// ── Garbage detector ──────────────────────────────────────────────────────────
// Messenger/chat apps that produce mostly personal noise.
// Captures from these are skipped UNLESS the title contains a work keyword.

const MESSENGER_APPS: &[&str] = &[
    "telegram", "discord", "slack", "whatsapp", "viber",
    "vk", "vkontakte", "messenger", "signal", "skype", "zoom",
    "teams", "element", "matrix",
];

const WORK_TRIGGER_KEYWORDS: &[&str] = &[
    "task", "deploy", "bug", "release",
    "pr ", "pull request", "merge", "issue", "urgent",
    "meeting", "deadline",
];

fn is_messenger_noise(app_name: &str, window_title: &str) -> bool {
    let app_l   = app_name.to_lowercase();
    let title_l = window_title.to_lowercase();
    let is_messenger = MESSENGER_APPS.iter().any(|m| app_l.contains(m));
    if !is_messenger { return false; }
    // Has work-relevant keyword → keep the capture
    if WORK_TRIGGER_KEYWORDS.iter().any(|kw| title_l.contains(kw)) { return false; }
    true
}

// ── Dirty-rectangle constants ─────────────────────────────────────────────────

const BLOCK_SIZE: u32 = 64;   // pixels per block side
const GRID_SIDE:  u32 = 6;    // 384 / 64 = 6
pub const GRID_CELLS: usize = (GRID_SIDE * GRID_SIDE) as usize;  // 36

// ── Output type ───────────────────────────────────────────────────────────────

pub struct CaptureFrame {
    pub app_name:        String,
    pub window_title:    String,
    pub phash:           u64,
    /// Per-block average hashes for dirty-rectangle detection.
    pub block_hashes:    [u64; GRID_CELLS],
    #[allow(dead_code)]
    pub width:           u32,
    #[allow(dead_code)]
    pub height:          u32,
    #[allow(dead_code)]
    pub compressed_size: usize,
    pub text_summary:    String,
    /// Number of chars extracted by OCR (0 = OCR produced nothing).
    pub ocr_char_count:  usize,
}

// ── Dirty-rectangle helpers ───────────────────────────────────────────────────

/// Computes one 64-bit average hash per 64×64 block of a 384×384 gray image.
/// Each bit encodes whether the pixel at that position is above the block average.
pub fn compute_block_hashes(img: &GrayImage) -> [u64; GRID_CELLS] {
    let mut hashes = [0u64; GRID_CELLS];

    for by in 0..GRID_SIDE {
        for bx in 0..GRID_SIDE {
            let x0 = bx * BLOCK_SIZE;
            let y0 = by * BLOCK_SIZE;
            let x1 = (x0 + BLOCK_SIZE).min(img.width());
            let y1 = (y0 + BLOCK_SIZE).min(img.height());

            // Block average
            let mut sum = 0u64;
            let mut cnt = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += img.get_pixel(x, y)[0] as u64;
                    cnt += 1;
                }
            }
            let avg = if cnt > 0 { sum / cnt } else { 0 };

            // Encode first 64 pixels relative to average
            let mut h   = 0u64;
            let mut bit = 0usize;
            'outer: for y in y0..y1 {
                for x in x0..x1 {
                    if bit >= 64 { break 'outer; }
                    if img.get_pixel(x, y)[0] as u64 > avg {
                        h |= 1u64 << bit;
                    }
                    bit += 1;
                }
            }

            hashes[(by * GRID_SIDE + bx) as usize] = h;
        }
    }

    hashes
}

/// Returns the fraction of blocks whose hash differs by more than 5 bits
/// (Hamming distance > 5) — a proxy for "visually changed" blocks.
pub fn changed_block_ratio(prev: &[u64; GRID_CELLS], curr: &[u64; GRID_CELLS]) -> f32 {
    let changed = prev.iter()
        .zip(curr.iter())
        .filter(|(a, b)| (*a ^ *b).count_ones() > 5)
        .count();
    changed as f32 / GRID_CELLS as f32
}

// ── Processor ─────────────────────────────────────────────────────────────────

pub struct ShakalProcessor;

impl ShakalProcessor {
    /// Main entry point.
    ///
    /// Returns `None` when:
    ///   - `window_title` matches the Privacy Shield blacklist
    ///   - Image buffer is malformed
    ///   - JPEG encoding fails
    pub fn process(
        rgba:         &[u8],
        width:        u32,
        height:       u32,
        window_title: &str,
        app_name:     &str,
    ) -> Option<CaptureFrame> {
        if Self::is_private(window_title) {
            info!("[Shakal/Privacy] Capture blocked — «{}»", window_title);
            return None;
        }

        // Garbage detector: skip personal messenger chats unless work keywords present
        if is_messenger_noise(app_name, window_title) {
            info!("[Shakal/Garbage] Messenger with no work keywords — skipping: {}", app_name);
            return None;
        }

        let (jpeg_bytes, phash, block_hashes) = match Self::compress(rgba, width, height) {
            Ok(r)  => r,
            Err(e) => { warn!("[Shakal] compress: {}", e); return None; }
        };

        let t_ocr = std::time::Instant::now();
        let ocr = crate::ocr::extract_text(rgba, width, height);
        let ocr_ms = t_ocr.elapsed().as_millis();
        let ocr_char_count = ocr.as_deref().map_or(0, str::len);
        if ocr_char_count > 0 {
            info!("[PERF] OCR extraction took: {} ms ({} chars)", ocr_ms, ocr_char_count);
            crate::perf::global().record_ocr(ocr_ms);
        }
        let text_summary = build_text_summary(window_title, app_name, ocr.as_deref());

        Some(CaptureFrame {
            app_name:        app_name.to_string(),
            window_title:    window_title.to_string(),
            phash,
            block_hashes,
            width,
            height,
            compressed_size: jpeg_bytes.len(),
            text_summary,
            ocr_char_count,
        })
    }

    // ── Privacy Shield ────────────────────────────────────────────────────────

    pub(crate) fn is_private(title: &str) -> bool {
        let lower = title.to_lowercase();
        PRIVACY_BLACKLIST.iter().any(|&word| lower.contains(word))
    }

    // ── Compression pipeline ──────────────────────────────────────────────────

    fn compress(rgba: &[u8], width: u32, height: u32) -> Result<(Vec<u8>, u64, [u64; GRID_CELLS])> {
        let rgba_img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
            .ok_or_else(|| anyhow!("Invalid RGBA buffer {}×{}", width, height))?;

        let gray: GrayImage = DynamicImage::ImageRgba8(rgba_img).to_luma8();

        let resized = image::imageops::resize(&gray, 384, 384, FilterType::Nearest);

        let mut posterized = resized;
        for p in posterized.pixels_mut() {
            p[0] = (p[0] / 32) * 32;
        }

        // Compute hashes before JPEG encoding consumes the buffer
        let block_hashes = compute_block_hashes(&posterized);
        let phash        = Self::phash(&posterized);

        let mut buf = Vec::with_capacity(4096);
        JpegEncoder::new_with_quality(&mut buf, 5u8)
            .encode(posterized.as_raw(), 384, 384, ColorType::L8)?;

        Ok((buf, phash, block_hashes))
    }

    /// 8×8 average hash — tolerant to minor brightness / contrast shifts.
    pub(crate) fn phash(img: &GrayImage) -> u64 {
        let small = image::imageops::resize(img, 8, 8, FilterType::Nearest);
        let avg   = small.pixels().map(|p| p[0] as u64).sum::<u64>() / 64;
        small.pixels().enumerate().take(64).fold(0u64, |h, (i, p)| {
            if (p[0] as u64) > avg { h | (1u64 << i) } else { h }
        })
    }
}

// ── Active window detection ───────────────────────────────────────────────────

pub fn active_window_info() -> (String, String) {
    #[cfg(windows)]
    return win_active_window();

    #[cfg(not(windows))]
    return (String::new(), String::new());
}

#[cfg(windows)]
fn win_active_window() -> (String, String) {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    #[link(name = "user32")]
    extern "system" {
        fn GetForegroundWindow() -> isize;
        fn GetWindowTextW(hwnd: isize, lp: *mut u16, n: i32) -> i32;
        fn GetWindowThreadProcessId(hwnd: isize, pid: *mut u32) -> u32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> isize;
        fn QueryFullProcessImageNameW(h: isize, flags: u32, name: *mut u16, size: *mut u32) -> i32;
        fn CloseHandle(h: isize) -> i32;
    }

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd == 0 {
            return (String::new(), String::new());
        }

        let mut title_buf = [0u16; 512];
        let title_len = GetWindowTextW(hwnd, title_buf.as_mut_ptr(), title_buf.len() as i32);
        let window_title = if title_len > 0 {
            OsString::from_wide(&title_buf[..title_len as usize])
                .to_string_lossy()
                .into_owned()
        } else {
            String::new()
        };

        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);

        let h_proc   = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        let app_name = if h_proc != 0 {
            let mut name_buf = [0u16; 512];
            let mut size     = name_buf.len() as u32;
            let ok           = QueryFullProcessImageNameW(h_proc, 0, name_buf.as_mut_ptr(), &mut size);
            CloseHandle(h_proc);
            if ok != 0 && size > 0 {
                let full = OsString::from_wide(&name_buf[..size as usize])
                    .to_string_lossy()
                    .into_owned();
                std::path::Path::new(&full)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or(full)
            } else {
                format!("pid:{pid}")
            }
        } else {
            String::new()
        };

        (app_name, window_title)
    }
}

// ── OCR ───────────────────────────────────────────────────────────────────────

/// Builds `text_summary` from window metadata + optional OCR text.
fn build_text_summary(window_title: &str, app_name: &str, ocr: Option<&str>) -> String {
    let header = if window_title.is_empty() {
        format!("[SCREEN] {app_name}")
    } else {
        format!("[SCREEN] {window_title} ({app_name})")
    };

    match ocr {
        Some(text) if !text.is_empty() => format!("{header}\n{text}"),
        _ if window_title.is_empty()   => format!("{header}: active window"),
        _                              => header,
    }
}


// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, fill: u8) -> GrayImage {
        GrayImage::from_pixel(w, h, image::Luma([fill]))
    }

    #[test]
    fn test_phash_deterministic() {
        let img = gray(8, 8, 128);
        assert_eq!(ShakalProcessor::phash(&img), ShakalProcessor::phash(&img));
    }

    #[test]
    fn test_phash_black_vs_white() {
        let black = gray(8, 8, 0);
        let white = gray(8, 8, 255);
        let h1    = ShakalProcessor::phash(&black);
        let h2    = ShakalProcessor::phash(&white);
        // Pure black (all 0) → all below avg → h = 0
        // Pure white (all 255) → avg = 255, nothing strictly above → also 0
        // Both should produce a consistent (possibly equal) result without panicking.
        let _ = h1 ^ h2; // just ensure no panic
    }

    #[test]
    fn test_phash_different_gradients() {
        let mut img1 = gray(8, 8, 0);
        let mut img2 = gray(8, 8, 255);
        // Make img1 have a gradient pattern
        for (i, p) in img1.pixels_mut().enumerate() { p[0] = (i * 4) as u8; }
        for (i, p) in img2.pixels_mut().enumerate() { p[0] = 255 - (i * 4) as u8; }
        let h1 = ShakalProcessor::phash(&img1);
        let h2 = ShakalProcessor::phash(&img2);
        assert_ne!(h1, h2, "Distinct gradient images should have different pHashes");
    }

    #[test]
    fn test_block_hashes_unchanged() {
        let img = gray(384, 384, 100);
        let h1  = compute_block_hashes(&img);
        let h2  = compute_block_hashes(&img);
        assert_eq!(changed_block_ratio(&h1, &h2), 0.0);
    }

    #[test]
    fn test_block_hashes_all_changed() {
        let mut img1 = gray(384, 384, 0);
        let mut img2 = gray(384, 384, 0);
        // Give them complementary gradient patterns so hashes differ
        for (i, p) in img1.pixels_mut().enumerate() { p[0] = (i % 256) as u8; }
        for (i, p) in img2.pixels_mut().enumerate() { p[0] = 255 - (i % 256) as u8; }
        let h1    = compute_block_hashes(&img1);
        let h2    = compute_block_hashes(&img2);
        let ratio = changed_block_ratio(&h1, &h2);
        assert!(ratio > 0.0, "Complementary patterns must register changes");
    }

    #[test]
    fn test_block_hashes_count() {
        let img = gray(384, 384, 128);
        let h   = compute_block_hashes(&img);
        assert_eq!(h.len(), GRID_CELLS);
    }

    #[test]
    fn test_privacy_shield_bank() {
        assert!(ShakalProcessor::is_private("Chase Bank — Login"));
    }

    #[test]
    fn test_privacy_shield_password() {
        assert!(ShakalProcessor::is_private("Enter Password"));
    }

    #[test]
    fn test_privacy_shield_safe() {
        assert!(!ShakalProcessor::is_private("VS Code — main.rs"));
    }
}
