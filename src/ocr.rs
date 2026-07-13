/// One-shot OCR processor.
///
/// Pixel data is consumed inside `extract_text` and never persisted beyond that
/// call — the caller can safely drop the RGBA buffer immediately after.
///
/// `ShotGuard`: RAII type for workflows that must write a temp image to disk
/// (e.g. Moondream, Tesseract CLI).  The file is deleted when the guard is
/// dropped, even on panic.  The current WinRT path never touches disk, so
/// `ShotGuard` exists purely for future vision-model integration.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

// ── Public API ────────────────────────────────────────────────────────────────

/// Extracts text from a raw RGBA frame.
/// Returns `None` when OCR finds nothing or on any failure.
/// The `rgba` buffer is read-only; the caller retains ownership and can drop it
/// immediately after this call.
pub fn extract_text(rgba: &[u8], width: u32, height: u32) -> Option<String> {
    run_ocr(rgba, width, height)
}

// ── RAII temp-file guard ──────────────────────────────────────────────────────

/// Holds the path to a temporary screenshot PNG and deletes it on drop.
///
/// Usage:
/// ```ignore
/// let guard = ShotGuard::write_temp(rgba, width, height)?;
/// some_vision_model(guard.path()); // model reads from disk
/// // guard dropped here → file deleted
/// ```
pub struct ShotGuard(PathBuf);

impl ShotGuard {
    /// Encodes `rgba` as PNG into `$TEMP/nic_shot_<uuid>.png` and returns a guard.
    #[allow(dead_code)]
    pub fn write_temp(rgba: &[u8], width: u32, height: u32) -> Result<Self> {
        use image::{ImageBuffer, RgbaImage};
        let img: RgbaImage = ImageBuffer::from_raw(width, height, rgba.to_vec())
            .ok_or_else(|| anyhow!("ShotGuard: invalid RGBA buffer {}×{}", width, height))?;
        let path = std::env::temp_dir()
            .join(format!("nic_shot_{}.png", uuid::Uuid::new_v4()));
        img.save_with_format(&path, image::ImageFormat::Png)
            .map_err(|e| anyhow!("ShotGuard write: {}", e))?;
        Ok(Self(path))
    }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path { &self.0 }
}

impl Drop for ShotGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ── OCR implementation ────────────────────────────────────────────────────────

fn clean_ocr_text(raw: &str) -> String {
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(windows)]
fn run_ocr(rgba: &[u8], width: u32, height: u32) -> Option<String> {
    use windows::{
        Graphics::Imaging::{
            BitmapAlphaMode, BitmapBufferAccessMode, BitmapPixelFormat, SoftwareBitmap,
        },
        Media::Ocr::OcrEngine,
        Win32::System::WinRT::{IMemoryBufferByteAccess, RO_INIT_MULTITHREADED, RoInitialize},
        core::Interface,
    };

    // RoInitialize is idempotent per-thread; S_FALSE ("already init") is fine.
    let _ = unsafe { RoInitialize(RO_INIT_MULTITHREADED) };

    // screenshots crate returns RGBA; WinRT SoftwareBitmap expects BGRA8.
    let bgra: Vec<u8> = rgba
        .chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0], px[3]])
        .collect();

    let engine = OcrEngine::TryCreateFromUserProfileLanguages().ok()?;
    let bitmap = SoftwareBitmap::CreateWithAlpha(
        BitmapPixelFormat::Bgra8,
        width as i32,
        height as i32,
        BitmapAlphaMode::Premultiplied,
    )
    .ok()?;

    // Write BGRA pixels directly into the locked bitmap buffer.
    // The block scope ensures LockBuffer is released before RecognizeAsync.
    {
        let bb        = bitmap.LockBuffer(BitmapBufferAccessMode::Write).ok()?;
        let reference = bb.CreateReference().ok()?;
        let access: IMemoryBufferByteAccess = reference.cast().ok()?;
        unsafe {
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut capacity: u32 = 0;
            access.GetBuffer(&mut ptr, &mut capacity).ok()?;
            if ptr.is_null() { return None; }
            let n = (capacity as usize).min(bgra.len());
            std::ptr::copy_nonoverlapping(bgra.as_ptr(), ptr, n);
        }
    }

    let result  = engine.RecognizeAsync(&bitmap).ok()?.get().ok()?;
    let raw     = result.Text().ok()?.to_string();
    let cleaned = clean_ocr_text(&raw);
    if cleaned.is_empty() { None } else { Some(cleaned) }
}

#[cfg(not(windows))]
fn run_ocr(_rgba: &[u8], _width: u32, _height: u32) -> Option<String> {
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── clean_ocr_text ────────────────────────────────────────────────────────

    #[test]
    fn clean_ocr_empty_input_empty_output() {
        assert_eq!(clean_ocr_text(""), "");
    }

    #[test]
    fn clean_ocr_single_line_no_whitespace() {
        assert_eq!(clean_ocr_text("hello"), "hello");
    }

    #[test]
    fn clean_ocr_trims_leading_spaces() {
        assert_eq!(clean_ocr_text("   hello"), "hello");
    }

    #[test]
    fn clean_ocr_trims_trailing_spaces() {
        assert_eq!(clean_ocr_text("hello   "), "hello");
    }

    #[test]
    fn clean_ocr_trims_both_sides() {
        assert_eq!(clean_ocr_text("  hello world  "), "hello world");
    }

    #[test]
    fn clean_ocr_removes_empty_lines() {
        let input = "line1\n\nline2\n\nline3";
        let result = clean_ocr_text(input);
        assert_eq!(result, "line1\nline2\nline3");
    }

    #[test]
    fn clean_ocr_removes_whitespace_only_lines() {
        let input = "line1\n   \nline2";
        let result = clean_ocr_text(input);
        assert_eq!(result, "line1\nline2");
    }

    #[test]
    fn clean_ocr_trims_and_removes_blank() {
        let input = "  hello  \n\n  world  \n";
        let result = clean_ocr_text(input);
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn clean_ocr_all_empty_lines_returns_empty() {
        let result = clean_ocr_text("\n\n\n   \n\n");
        assert_eq!(result, "");
    }

    #[test]
    fn clean_ocr_joins_with_newline_not_space() {
        let result = clean_ocr_text("a\nb");
        assert_eq!(result, "a\nb");
        assert!(!result.contains("a b"));
    }

    #[test]
    fn clean_ocr_single_line_with_inner_spaces_preserved() {
        // Trim only strips leading/trailing — inner spaces stay
        assert_eq!(clean_ocr_text("hello   world"), "hello   world");
    }

    #[test]
    fn clean_ocr_cyrillic_text_preserved() {
        let input = "  Привет мир  \n  как дела  ";
        let result = clean_ocr_text(input);
        assert_eq!(result, "Привет мир\nкак дела");
    }

    #[test]
    fn clean_ocr_tabs_treated_as_trimable() {
        // str::trim() removes ASCII whitespace including tabs
        let input = "\thello\t";
        let result = clean_ocr_text(input);
        assert_eq!(result, "hello");
    }

    #[test]
    fn clean_ocr_preserves_punctuation() {
        assert_eq!(clean_ocr_text("Hello, world!"), "Hello, world!");
    }

    #[test]
    fn clean_ocr_many_blank_lines_between() {
        let input = "a\n\n\n\n\n\nb";
        let result = clean_ocr_text(input);
        assert_eq!(result, "a\nb");
    }

    // ── ShotGuard ─────────────────────────────────────────────────────────────

    #[test]
    fn shot_guard_write_invalid_buffer_returns_err() {
        // Buffer size doesn't match width × height × 4 → Err
        let result = ShotGuard::write_temp(&[0u8; 4], 100, 100);
        assert!(result.is_err(), "invalid buffer dimensions should fail");
    }

    #[test]
    fn shot_guard_write_zero_size_returns_err() {
        let result = ShotGuard::write_temp(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn shot_guard_valid_1x1_pixel_writes_and_cleans() {
        // 1×1 RGBA = 4 bytes
        let rgba = vec![255u8, 0, 0, 255]; // red pixel
        let guard = ShotGuard::write_temp(&rgba, 1, 1).expect("1×1 should succeed");
        let path = guard.path().to_path_buf();
        assert!(path.exists(), "file should exist while guard is alive");
        drop(guard);
        assert!(!path.exists(), "file should be deleted on drop");
    }

    #[test]
    fn shot_guard_path_ends_with_png() {
        let rgba = vec![0u8; 4]; // 1×1 black pixel
        if let Ok(guard) = ShotGuard::write_temp(&rgba, 1, 1) {
            assert!(guard.path().extension().map(|e| e == "png").unwrap_or(false));
            // allow guard to clean up
        }
    }

    // ── extract_text passthrough ──────────────────────────────────────────────

    #[test]
    fn extract_text_empty_buffer_no_panic() {
        // On non-Windows, run_ocr returns None; should never panic
        let result = extract_text(&[], 0, 0);
        // Either None (non-windows) or Some(...) — just no panic
        let _ = result;
    }
}
