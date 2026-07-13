pub mod analyst;
pub mod api;
pub mod autostart;
pub mod boot;
pub mod config;
pub mod error;
pub mod hardware;
pub mod initiative;
pub mod llm_server;
pub mod llm_utils;
pub mod locale;
pub mod modules;
pub mod notes;
pub mod ocr;
pub mod perf;
pub mod services;
pub mod setup;
pub mod shakal;
pub mod librarian;
pub mod verify;
pub mod wry_bridge;

use anyhow::Result;
use llm_utils::LlmEngine;
use modules::{
    sentinel::Sentinel,
    surfer::Surfer,
    thinker::Thinker,
    ContextCollector,
    SentinelEvent,
    start_performance_monitor,
};
use crate::analyst::Analyst;
use crate::librarian::Librarian;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;


/// Run the backend server (called from the launcher in a background thread).
pub async fn run_backend() -> Result<()> {
    // --- [0] Logging to file, no console ---
    #[cfg(windows)]
    unsafe { extern "system" { fn FreeConsole() -> i32; } FreeConsole(); }

    let log_dir = config::app_bin_dir().join("logs");
    std::fs::create_dir_all(&log_dir).ok();
    let (non_blocking, _log_guard) = tracing_appender::non_blocking(
        tracing_appender::rolling::never(&log_dir, "nic.log")
    );
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_writer(non_blocking)
        .init();

    tracing::info!("NIC started");

    // --- [autostart] ---
    if !autostart::is_enabled() {
        let _ = autostart::enable();
    }

    // All fallible startup lives in init_backend. A failure used to bubble up to
    // the launcher thread, which dropped the tokio runtime — killing the
    // bootstrap /boot server with it, so the window showed a dead
    // "nic-assistant unavailable" with zero explanation (seen live when a
    // first-run engine download hit a broken llama.cpp release). Instead:
    // publish the error on /boot (Phase::Error) and keep serving it forever, so
    // the UI renders the real message and the user knows what to fix.
    if let Err(e) = init_backend().await {
        let msg = format!("{e:#}");
        tracing::error!("Startup failed: {msg}");
        crate::boot::boot().set_error(&msg);
        // If the first-run bootstrap server is still bound (failure happened
        // before the port handoff) it already serves the error and the bind
        // below lands on the next port — harmless, both read the same state.
        let ports = config::AppConfig::default().server.ports;
        if let Ok((listener, port)) = bind_first_port(&ports).await {
            tracing::info!("Serving startup error on http://127.0.0.1:{}/boot", port);
            let _ = api::run_bootstrap_server(listener, std::future::pending::<()>()).await;
        }
        // No port could be bound at all — park forever; the log has the error.
        std::future::pending::<()>().await
    }
    Ok(())
}

/// The actual startup sequence — everything that can fail. Split out of
/// `run_backend` so a failure can be rendered by the UI instead of silently
/// killing the runtime (see the caller).
async fn init_backend() -> Result<()> {
    // --- [1] Components ---
    let cfg = config::load_config()?;
    // English-only beta (MASTER_PLAN §9[1]): NO OS-locale auto-detection — it
    // used to flip the whole UI + model prompts to the machine's language
    // (seen live: Russian Windows → Russian UI on first run). English unless
    // the user EXPLICITLY sets `language` in config.toml; the i18n dictionaries
    // stay in ui.html as that dormant opt-in, nothing activates them by itself.
    let language = if cfg.adaptive.language.is_empty() {
        "English".to_string()
    } else {
        cfg.adaptive.language.clone()
    };
    tracing::info!("Language: {}", language);

    let city = if cfg.adaptive.city.is_empty() {
        locale::detect_city()
    } else {
        Some(cfg.adaptive.city.clone())
    };
    if let Some(ref c) = city { tracing::info!("City: {}", c); }

    // Reconcile the model URL with the currently selected preset.
    // If the user picked a different model in the UI on a previous run, the
    // config.toml will have the new `selected_model` id but the old `model_url`
    // — overwrite the URL so `setup::ensure_ready` downloads the right file.
    let mut cfg = cfg;

    // Auto-pick a model that fits this machine (first run / until the user picks
    // one manually in the UI). Keeps the out-of-the-box experience usable on
    // weak hardware instead of always defaulting to the medium model.
    if cfg.models.auto_select_model {
        let ids: Vec<String> = cfg.models.presets.iter().map(|p| p.id.clone()).collect();
        let hw = hardware::Hardware::detect();
        if let Some(rec) = hw.recommended_model(&ids) {
            if rec != cfg.models.selected_model {
                tracing::info!(
                    "[Hardware] Auto-selected model '{}' for this machine (was '{}')",
                    rec, cfg.models.selected_model
                );
                cfg.models.selected_model = rec.to_string();
            }
        }
    }

    if let Some(preset) = cfg.models.preset(&cfg.models.selected_model).cloned() {
        let preset_url = preset.download_url();
        if cfg.models.model_url != preset_url {
            tracing::info!(
                "Syncing model_url to selected preset '{}' → {}",
                preset.id, preset_url
            );
            cfg.models.model_url = preset_url;
        }
    } else if !cfg.models.presets.is_empty() {
        // selected_model is unknown (e.g. a typo in config.toml) → fall back
        // to the first preset so the app still boots.
        let fallback = cfg.models.presets[0].clone();
        tracing::warn!(
            "selected_model '{}' not found in presets — falling back to '{}'",
            cfg.models.selected_model, fallback.id
        );
        cfg.models.selected_model = fallback.id.clone();
        cfg.models.model_url = fallback.download_url();
    }

    // Bind the UI port up-front and run a tiny bootstrap server on it while the
    // model downloads and the engine starts (minutes on first run), so the UI
    // can show real progress via /boot instead of a silent "connecting…". It is
    // shut down and handed off to the real API server once everything is ready.
    let (boot_listener, ui_port) = bind_first_port(&cfg.server.ports).await?;
    tracing::info!("Bootstrap UI server on http://127.0.0.1:{}", ui_port);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let bootstrap_handle = tokio::spawn(async move {
        let _ = api::run_bootstrap_server(boot_listener, async move {
            let _ = shutdown_rx.await;
        }).await;
    });

    // Download + start llama-server for the selected model, gracefully falling
    // back to smaller presets if it can't be loaded (low RAM/VRAM, corrupt file)
    // instead of failing the whole backend.
    crate::boot::boot().set_phase(crate::boot::Phase::StartingEngine);
    let (setup_paths, llama_server) = ensure_and_spawn(&mut cfg).await?;
    llm_server::LlamaServer::start_watchdog(llama_server.clone());

    // Engine is up; building the memory index / embedder.
    crate::boot::boot().set_phase(crate::boot::Phase::LoadingMemory);
    let server_url = llama_server.base_url.clone();
    let make_engine = || Arc::new(std::sync::Mutex::new(LlmEngine::new(&server_url)));

    let librarian = Arc::new(Librarian::new_from_config(&cfg, make_engine()).await?);
    librarian.clone().start_background_tasks().await;

    let surfer    = Arc::new(Surfer::new()?);
    let thinker   = Arc::new(Mutex::new(Thinker::new(make_engine(), &cfg.profile, &language)));
    let collector = Arc::new(ContextCollector::new(librarian.clone(), surfer.clone()));

    // --- [2] Sentinel ---
    let (sentinel_tx, mut sentinel_rx) = tokio::sync::mpsc::channel::<SentinelEvent>(512);
    let lib_sink       = librarian.clone();
    let incognito      = Arc::new(std::sync::atomic::AtomicBool::new(cfg.adaptive.incognito_mode));
    let incognito_task = incognito.clone();

    tokio::spawn(async move {
        while let Some(event) = sentinel_rx.recv().await {
            if incognito_task.load(Ordering::Relaxed) { continue; }
            match event {
                SentinelEvent::Screen { app_name, window_title, description } =>
                    { let _ = lib_sink.store(&app_name, &window_title, &description).await; },
                SentinelEvent::Clipboard { text } =>
                    { let _ = lib_sink.store("clipboard", "Clipboard", &text).await; },
            }
        }
    });

    // Screen-capture consent (privacy: first run reads NOTHING until the user
    // enables memory). The marker lives next to the DB; its presence = consent.
    // The Sentinel loop reads the flag every tick, so setting it here — before
    // or after start — is equally safe.
    let consent_marker = cfg.librarian.db_path.parent()
        .map(|p| p.join(".consent"))
        .unwrap_or_else(|| PathBuf::from("data/.consent"));
    crate::modules::sentinel::set_capture_consent(consent_marker.exists());

    let sentinel = Arc::new(Sentinel::new(sentinel_tx));
    sentinel.start();
    start_performance_monitor(cfg.adaptive.cpu_limit_pct);

    // --- [3] Background tasks ---
    Analyst::new(librarian.clone(), thinker.clone(), &cfg.adaptive.summary_time, &language).start();
    initiative::start(
        librarian.clone(), thinker.clone(), language.clone(),
        cfg.adaptive.initiative_enabled, incognito.clone(),
    );

    // --- [4] API server ---
    //
    // Locate the live config.toml so /models/select can persist the user's
    // choice — the same lookup order as `config::load_config`.
    let config_path = config::app_bin_dir()
        .join("config.toml");
    let config_path = if config_path.exists() {
        config_path
    } else {
        PathBuf::from("config.toml")
    };

    // Security token shared with the UI by the launcher (per-launch random).
    // Falls back to a configured key, or empty (dev/no-launcher) if neither.
    let api_key = std::env::var("NIC_LOCAL_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| cfg.security.api_key.clone());

    let state = api::ApiState {
        collector,
        thinker:       thinker.clone(),
        librarian:     librarian.clone(),
        incognito,
        api_key,
        last_exchange: Arc::new(tokio::sync::Mutex::new(None)),
        llm_url:       server_url.clone(),
        answer_cache:  Arc::new(tokio::sync::Mutex::new(Vec::new())),
        profile:       Arc::new(tokio::sync::Mutex::new(cfg.profile.clone())),
        city,
        language:       Arc::new(tokio::sync::Mutex::new(language)),
        selected_model: Arc::new(tokio::sync::Mutex::new(cfg.models.selected_model.clone())),
        presets:        cfg.models.presets.clone(),
        model_path:     setup_paths.model_path.clone(),
        config_path,
        consent_marker,
    };


    // Everything is ready: stop the bootstrap server, take back its port, and
    // start the real API server on it. A brief connection gap during the swap is
    // harmless — the UI's 1.5 s poll simply reconnects to the real server.
    crate::boot::boot().set_phase(crate::boot::Phase::Ready);
    let _ = shutdown_tx.send(());
    let _ = bootstrap_handle.await;
    let listener = rebind_port(ui_port).await?;

    tracing::info!("Backend API ready");

    api::start_api_server_with_listener(state, listener).await?;

    Ok(())
}

/// Binds the first free port from `ports`, returning the listener and the port.
async fn bind_first_port(ports: &[u16]) -> Result<(tokio::net::TcpListener, u16)> {
    for &port in ports {
        if let Ok(l) = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await {
            return Ok((l, port));
        }
    }
    Err(anyhow::anyhow!("Could not bind to any of the ports: {:?}", ports))
}

/// Re-binds a specific port, retrying briefly — the bootstrap server has just
/// released it and the OS may need a moment to free the listening socket.
async fn rebind_port(port: u16) -> Result<tokio::net::TcpListener> {
    let mut last_err = None;
    for _ in 0..25 {
        match tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await {
            Ok(l) => return Ok(l),
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    Err(anyhow::anyhow!(
        "could not re-bind port {port} after bootstrap handoff: {:?}", last_err
    ))
}

/// File path for a fallback preset's own GGUF, kept separate from the primary
/// `main.gguf` so a smaller fallback downloads cleanly instead of colliding with
/// the larger model that failed to load.
fn preset_model_path(base: &std::path::Path, preset_id: &str) -> PathBuf {
    base.with_file_name(format!("{preset_id}.gguf"))
}

/// Brings up llama-server for the selected model, falling back to progressively
/// smaller presets if it cannot be loaded (not enough RAM/VRAM, or a corrupt
/// download) — degrading gracefully instead of failing the whole backend.
///
/// Returns the paths + running server of whichever model succeeded, and rewrites
/// `cfg.models` to reflect the model that was actually loaded so the UI and the
/// rest of startup agree on it.
async fn ensure_and_spawn(
    cfg: &mut config::AppConfig,
) -> Result<(setup::SetupPaths, Arc<llm_server::LlamaServer>)> {
    // Attempt chain: selected model first (using its configured path so an
    // existing download is reused), then every smaller preset largest-first,
    // each with its own file so a fallback actually re-downloads.
    let mut chain: Vec<(config::ModelPreset, PathBuf, String)> = Vec::new();
    if let Some(sel) = cfg.models.preset(&cfg.models.selected_model).cloned() {
        chain.push((sel, cfg.models.model_path.clone(), cfg.models.model_url.clone()));
    }
    let sel_size = chain.first().map(|(p, _, _)| p.size_mb).unwrap_or(u32::MAX);
    let mut smaller: Vec<config::ModelPreset> = cfg.models.presets.iter()
        .filter(|p| p.size_mb < sel_size)
        .cloned()
        .collect();
    smaller.sort_by(|a, b| b.size_mb.cmp(&a.size_mb)); // largest of the smaller first
    for p in smaller {
        let path = preset_model_path(&cfg.models.model_path, &p.id);
        let url  = p.download_url();
        chain.push((p, path, url));
    }

    // Fully custom config (selected id not among presets) → configured paths verbatim.
    if chain.is_empty() {
        let setup_paths = setup::ensure_ready(
            &cfg.models.server_bin, &cfg.models.model_path, &cfg.models.model_url,
            &cfg.models.compute,
        ).await?;
        let server = llm_server::LlamaServer::spawn(
            &setup_paths.server_bin, &setup_paths.model_path, cfg.models.server_port,
            setup_paths.n_gpu_layers, cfg.models.ctx_size, cfg.models.parallel,
        ).await?;
        return Ok((setup_paths, server));
    }

    let total = chain.len();
    let mut last_err = anyhow::anyhow!("no model could be loaded");
    for (i, (preset, model_path, model_url)) in chain.into_iter().enumerate() {
        if i > 0 {
            tracing::warn!(
                "[Fallback] previous model failed — trying smaller model '{}' (~{} MB)",
                preset.id, preset.size_mb
            );
        }
        let setup_paths = match setup::ensure_ready(
            &cfg.models.server_bin, &model_path, &model_url, &cfg.models.compute,
        ).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("[Fallback] setup for '{}' failed: {:#}", preset.id, e);
                last_err = e;
                continue;
            }
        };
        match llm_server::LlamaServer::spawn(
            &setup_paths.server_bin, &setup_paths.model_path, cfg.models.server_port,
            setup_paths.n_gpu_layers, cfg.models.ctx_size, cfg.models.parallel,
        ).await {
            Ok(server) => {
                if i > 0 {
                    tracing::info!(
                        "[Fallback] loaded smaller model '{}' after {} failed attempt(s)",
                        preset.id, i
                    );
                    // Reflect the model actually loaded so the UI + later code agree.
                    cfg.models.selected_model = preset.id.clone();
                    cfg.models.model_path     = model_path;
                    cfg.models.model_url      = model_url;
                }
                return Ok((setup_paths, server));
            }
            Err(e) => {
                tracing::warn!(
                    "[Fallback] '{}' failed to start ({} of {}): {:#}",
                    preset.id, i + 1, total, e
                );
                last_err = e;
                continue;
            }
        }
    }
    Err(last_err.context("all model presets failed to load — see llama-server.log"))
}