//! First-run boot progress, shared between the startup sequence and a tiny
//! bootstrap HTTP server so the UI can show "Downloading model… 340/1024 MB"
//! instead of a silent "connecting…" for the minutes a first launch takes.
//!
//! The state is a process-global set of atomics (lock-free, written from the
//! download path in `setup.rs` and read by the `/boot` endpoint). A global is
//! used deliberately: the download helpers are nested several calls deep and
//! threading a handle through all of them would add noise for no benefit.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Coarse startup phase. Numeric values are what the `/boot` endpoint returns;
/// the UI maps them to localized text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    /// Process started, nothing heavy began yet.
    Starting = 0,
    /// Downloading the llama-server runtime (first run only).
    DownloadingRuntime = 1,
    /// Downloading the GGUF model file (first run only).
    DownloadingModel = 2,
    /// Model is downloaded; llama-server is loading it into memory.
    StartingEngine = 3,
    /// Engine is up; building the memory index / embedder.
    LoadingMemory = 4,
    /// Fully ready — the real API server is now serving.
    Ready = 5,
    /// Startup failed; `error` holds a human-readable message.
    Error = 6,
}

/// Lock-free boot status, updated as startup progresses.
pub struct BootStatus {
    phase:    AtomicU8,
    done_mb:  AtomicU64,
    total_mb: AtomicU64,
    error:    Mutex<String>,
}

impl BootStatus {
    fn new() -> Self {
        Self {
            phase:    AtomicU8::new(Phase::Starting as u8),
            done_mb:  AtomicU64::new(0),
            total_mb: AtomicU64::new(0),
            error:    Mutex::new(String::new()),
        }
    }

    /// Sets the current phase and clears any stale download counters when the
    /// new phase is not itself a download phase.
    pub fn set_phase(&self, p: Phase) {
        self.phase.store(p as u8, Ordering::Relaxed);
        if !matches!(p, Phase::DownloadingRuntime | Phase::DownloadingModel) {
            self.done_mb.store(0, Ordering::Relaxed);
            self.total_mb.store(0, Ordering::Relaxed);
        }
    }

    /// Updates the download progress counters (megabytes).
    pub fn set_progress(&self, done_mb: u64, total_mb: u64) {
        self.done_mb.store(done_mb, Ordering::Relaxed);
        self.total_mb.store(total_mb, Ordering::Relaxed);
    }

    /// Records a fatal startup error and flips the phase to `Error`.
    pub fn set_error(&self, msg: impl Into<String>) {
        if let Ok(mut e) = self.error.lock() {
            *e = msg.into();
        }
        self.phase.store(Phase::Error as u8, Ordering::Relaxed);
    }

    /// Returns `(phase, done_mb, total_mb, error)` for the `/boot` endpoint.
    pub fn snapshot(&self) -> (u8, u64, u64, String) {
        (
            self.phase.load(Ordering::Relaxed),
            self.done_mb.load(Ordering::Relaxed),
            self.total_mb.load(Ordering::Relaxed),
            self.error.lock().map(|e| e.clone()).unwrap_or_default(),
        )
    }
}

static BOOT: OnceLock<Arc<BootStatus>> = OnceLock::new();

/// Returns the process-global boot status, creating it on first access.
pub fn boot() -> Arc<BootStatus> {
    BOOT.get_or_init(|| Arc::new(BootStatus::new())).clone()
}
