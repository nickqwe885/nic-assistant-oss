use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Manages a llama-server child process.
/// Wrapped in Arc so the watchdog can share ownership with main().
pub struct LlamaServer {
    child:        Mutex<Child>,
    pub base_url: String,
    // Stored for restart
    bin:          PathBuf,
    model:        PathBuf,
    port:         u16,
    n_gpu_layers: i32,
    ctx_size:     usize,
    parallel:     u8,
    log_path:     PathBuf,
    /// Windows Job Object handle (as isize so it's Send) with KILL_ON_JOB_CLOSE.
    /// llama-server is assigned to it, so when THIS process exits for any reason
    /// (clean Quit, crash, Task Manager) the OS terminates llama-server too —
    /// no orphaned background process. `None` on non-Windows / if creation failed.
    job_raw:      Option<isize>,
}

impl LlamaServer {
    /// Spawns llama-server and waits until the model is loaded and ready.
    pub async fn spawn(
        bin:          &Path,
        model:        &Path,
        port:         u16,
        n_gpu_layers: i32,
        ctx_size:     usize,
        parallel:     u8,
    ) -> Result<Arc<Self>> {
        info!("[LlamaServer] spawning {} model={} port={} parallel={}",
              bin.display(), model.display(), port, parallel);

        // Clean up any stray llama-server from a previous unclean exit (e.g. an
        // old build without the kill-on-close guard) so we can bind the port.
        kill_stale_llama();

        let log_path = bin.parent().unwrap_or(Path::new(".")).join("llama-server.log");
        let child = Self::launch(bin, model, port, n_gpu_layers, ctx_size, parallel, &log_path)?;

        // Tie llama-server's lifetime to ours via a kill-on-close Job Object, so a
        // full Quit (or a crash) can never leave it running in the background.
        let job_raw = create_kill_on_close_job();
        if let Some(j) = job_raw { assign_child_to_job(&child, j); }

        let base_url = format!("http://127.0.0.1:{}", port);
        let server   = Arc::new(Self {
            child: Mutex::new(child),
            base_url,
            bin:   bin.to_path_buf(),
            model: model.to_path_buf(),
            port, n_gpu_layers, ctx_size, parallel,
            log_path: log_path.clone(),
            job_raw,
        });

        server.wait_ready(Duration::from_secs(180)).await
            .with_context(|| format!(
                "llama-server failed to start. Log: {}",
                log_path.display()
            ))?;

        Ok(server)
    }

    fn launch(
        bin:          &Path,
        model:        &Path,
        port:         u16,
        n_gpu_layers: i32,
        ctx_size:     usize,
        parallel:     u8,
        log_path:     &Path,
    ) -> Result<Child> {
        let log_file = std::fs::OpenOptions::new()
            .write(true).create(true).append(true)
            .open(log_path)
            .unwrap_or_else(|_| {
                std::fs::File::create(std::env::temp_dir().join("llama-server.log"))
                    .expect("cannot open fallback log file")
            });

        // llama-server uses fopen() (ANSI), so paths with non-ASCII characters
        // (e.g. Cyrillic) fail on Windows. Convert to 8.3 short path (ASCII only).
        let model_short = win_short_path(model);
        let model_str   = model_short.to_str().context("model path: non-UTF-8")?;

        let mut cmd = Command::new(bin);
        cmd.args([
                "--model",        model_str,
                "--host",         "127.0.0.1",
                "--port",         &port.to_string(),
                "--n-gpu-layers", &n_gpu_layers.to_string(),
                "--ctx-size",     &ctx_size.to_string(),
                "--parallel",     &parallel.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::from(log_file));
        // Hide the console window llama-server would otherwise pop up.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        cmd.spawn()
            .with_context(|| format!("failed to launch {}", bin.display()))
    }

    /// Returns true if the child process is still running.
    pub fn is_alive(&self) -> bool {
        self.child.lock().unwrap()
            .try_wait()
            .map(|s| s.is_none())
            .unwrap_or(false)
    }

    /// Kill the current child and start a fresh one; wait for ready.
    async fn restart(&self) -> Result<()> {
        warn!("[LlamaServer] restarting…");
        {
            let mut child = self.child.lock().unwrap();
            let _ = child.kill();
            let _ = child.wait();
        }
        let new_child = Self::launch(
            &self.bin, &self.model,
            self.port, self.n_gpu_layers, self.ctx_size, self.parallel,
            &self.log_path,
        )?;
        if let Some(j) = self.job_raw { assign_child_to_job(&new_child, j); }
        *self.child.lock().unwrap() = new_child;
        self.wait_ready(Duration::from_secs(180)).await
    }

    /// Spawns a background task that restarts the server if it dies.
    pub fn start_watchdog(server: Arc<Self>) {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                if !server.is_alive() {
                    warn!("[LlamaServer] process exited unexpectedly");
                    match server.restart().await {
                        Ok(_)  => info!("[LlamaServer] restarted successfully"),
                        Err(e) => warn!("[LlamaServer] restart failed: {}", e),
                    }
                }
            }
        });
    }

    async fn wait_ready(&self, timeout: Duration) -> Result<()> {
        let client      = reqwest::Client::new();
        let health_url  = format!("{}/health", self.base_url);
        let deadline    = Instant::now() + timeout;
        let mut dots    = 0usize;

        while Instant::now() < deadline {
            match client.get(&health_url)
                .timeout(Duration::from_secs(2))
                .send().await
            {
                Ok(r) if r.status().is_success() => {
                    println!(" ready ✓");
                    info!("[LlamaServer] ready at {}", self.base_url);
                    return Ok(());
                }
                _ => {}
            }
            tokio::time::sleep(Duration::from_millis(800)).await;
            dots += 1;
            if dots % 5 == 0 {
                print!(".");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
        }
        anyhow::bail!("timeout {}s", timeout.as_secs())
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        info!("[LlamaServer] killing child process");
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
    }
}

/// Returns the Windows 8.3 short path (ASCII-only) for a given path.
/// llama-server uses fopen() which can't handle non-ASCII paths on Windows.
/// Falls back to the original path if GetShortPathNameW fails or 8.3 names are disabled.
#[cfg(windows)]
fn win_short_path(path: &Path) -> PathBuf {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    extern "system" {
        fn GetShortPathNameW(
            lpszLongPath:  *const u16,
            lpszShortPath: *mut u16,
            cchBuffer:     u32,
        ) -> u32;
    }

    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut buf = vec![0u16; 32768];
    let len = unsafe { GetShortPathNameW(wide.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
    if len > 0 && (len as usize) < buf.len() {
        PathBuf::from(OsString::from_wide(&buf[..len as usize]))
    } else {
        path.to_path_buf()
    }
}

#[cfg(not(windows))]
fn win_short_path(path: &Path) -> PathBuf { path.to_path_buf() }

// ── Kill-on-close Job Object ───────────────────────────────────────────────────
//
// On Windows the launcher runs the backend (and thus llama-server) but the event
// loop exits via process::exit, which skips Rust destructors — so the Drop that
// kills llama-server doesn't always run, orphaning it. A Job Object with
// JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE fixes this at the OS level: llama-server is
// assigned to the job, and the moment our process exits (clean, crash, or kill)
// the job's last handle closes and Windows terminates llama-server with it.

/// Creates a kill-on-close Job Object. Returns its handle as `isize` (Send-safe),
/// or `None` if creation/config failed.
#[cfg(windows)]
fn create_kill_on_close_job() -> Option<isize> {
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::JobObjects::{
        CreateJobObjectW, SetInformationJobObject, JobObjectExtendedLimitInformation,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    unsafe {
        let job = CreateJobObjectW(None, PCWSTR::null()).ok()?;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .is_ok();
        if !ok {
            let _ = CloseHandle(job);
            warn!("[LlamaServer] SetInformationJobObject failed — no kill-on-close guard");
            return None;
        }
        info!("[LlamaServer] kill-on-close Job Object created");
        Some(job.0 as isize)
    }
}

/// Assigns a child process to the given Job Object handle.
#[cfg(windows)]
fn assign_child_to_job(child: &Child, job_raw: isize) {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::AssignProcessToJobObject;

    unsafe {
        let job   = HANDLE(job_raw as *mut std::ffi::c_void);
        let proc_ = HANDLE(child.as_raw_handle() as *mut std::ffi::c_void);
        if AssignProcessToJobObject(job, proc_).is_err() {
            warn!("[LlamaServer] AssignProcessToJobObject failed — llama may outlive a crash");
        }
    }
}

#[cfg(not(windows))]
fn create_kill_on_close_job() -> Option<isize> { None }

#[cfg(not(windows))]
fn assign_child_to_job(_child: &Child, _job_raw: isize) {}

/// Best-effort: terminate any stray llama-server.exe left over from a previous
/// run (e.g. an old build without the kill-on-close guard) so our fresh,
/// job-managed instance can bind the port cleanly.
#[cfg(windows)]
fn kill_stale_llama() {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let killed = Command::new("taskkill")
        .args(["/F", "/IM", "llama-server.exe"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if killed {
        info!("[LlamaServer] terminated a stray llama-server from a previous run");
        std::thread::sleep(Duration::from_millis(300)); // let the port free up
    }
}

#[cfg(not(windows))]
fn kill_stale_llama() {}
