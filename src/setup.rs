use anyhow::{Context, Result};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use tracing::info;

pub struct SetupPaths {
    pub server_bin:   PathBuf,
    pub model_path:   PathBuf,
    /// GPU layers to pass to llama-server: -1 = auto (GPU), 0 = CPU-only.
    pub n_gpu_layers: i32,
}

/// Returns true if Vulkan runtime is available on this machine.
/// Checks for vulkan-1.dll in System32 (present when any GPU driver with Vulkan support is installed).
fn vulkan_available() -> bool {
    #[cfg(windows)]
    return std::path::Path::new(r"C:\Windows\System32\vulkan-1.dll").exists();
    #[cfg(not(windows))]
    return true;
}

/// Ensures llama-server binary and GGUF model are present.
/// Downloads them on first run with progress reporting to stdout.
/// Auto-detects Vulkan availability: uses GPU build + GPU layers when possible,
/// falls back to AVX2 CPU build on machines without Vulkan drivers.
pub async fn ensure_ready(
    server_bin: &Path,
    model_path: &Path,
    model_url:  &str,
    compute:    &str,
) -> Result<SetupPaths> {
    // Auto-detect GPU vs CPU, but let the user override via `compute`
    // ("gpu"/"cpu"). Forcing CPU is useful when the GPU is busy (e.g. gaming)
    // or an integrated GPU can't really run the model well.
    let auto_layers = if !server_bin.exists() {
        println!("[Setup] llama-server not found — downloading from GitHub…");
        crate::boot::boot().set_phase(crate::boot::Phase::DownloadingRuntime);
        let dir = server_bin.parent().context("server_bin has no parent directory")?;
        let gpu = download_llama_server(dir).await.context("failed to download llama-server")?;
        if gpu { -1 } else { 0 }
    } else {
        // Already downloaded — infer backend from presence of vulkan DLL next to binary.
        let dir = server_bin.parent().unwrap_or(Path::new("."));
        let has_vulkan_build = dir.join("ggml-vulkan.dll").exists();
        if has_vulkan_build && vulkan_available() { -1 } else { 0 }
    };
    let n_gpu_layers = match compute.to_lowercase().as_str() {
        "cpu" => { info!("[Setup] compute=cpu — forcing CPU inference"); 0 }
        "gpu" => { info!("[Setup] compute=gpu — forcing GPU inference"); -1 }
        _     => auto_layers,
    };

    anyhow::ensure!(
        server_bin.exists(),
        "llama-server.exe not found in {}.\n\
         Download it manually: https://github.com/ggerganov/llama.cpp/releases\n\
         → llama-*-bin-win-vulkan-x64.zip or llama-*-bin-win-avx2-x64.zip → unzip into {}",
        server_bin.display(),
        server_bin.parent().unwrap().display()
    );

    if !model_path.exists() {
        std::fs::create_dir_all(model_path.parent().context("model_path has no parent")?)?;
        // BYO-first. Three tiers, each falling back to the next:
        //   1. a .gguf the user dropped next to the app → adopt silently, fully offline.
        //   2. otherwise ask via the system file explorer → adopt the picked file.
        //   3. otherwise (cancelled / unavailable) → download the recommended model.
        let byo: Option<PathBuf> = match find_byo_gguf(model_path.parent()) {
            Some(found) => {
                println!("[Setup] Found a local model (BYO): {}", found.display());
                Some(found)
            }
            // Native "open file" dialog — best-effort: a cancel or any failure just
            // falls through to the download, so a fresh install never hard-stops here.
            None => tokio::task::spawn_blocking(prompt_gguf_via_dialog).await.ok().flatten(),
        };
        if let Some(src) = byo {
            // llama-server opens the file via fopen() (ANSI), which fails on non-ASCII
            // paths — so copy the chosen GGUF onto the guaranteed-ASCII model_path.
            println!("[Setup] Using your model → {}", model_path.display());
            std::fs::copy(&src, model_path)
                .with_context(|| format!("failed to copy your model from {}", src.display()))?;
        } else {
            println!("[Setup] No local model chosen — downloading ({})…", model_url);
            crate::boot::boot().set_phase(crate::boot::Phase::DownloadingModel);
            download_model(model_url, model_path).await.context("failed to download model")?;
        }
    }

    info!("[Setup] server_bin={} model={} n_gpu_layers={}",
          server_bin.display(), model_path.display(), n_gpu_layers);
    Ok(SetupPaths {
        server_bin:   server_bin.to_path_buf(),
        model_path:   model_path.to_path_buf(),
        n_gpu_layers,
    })
}

/// Returns the first `*.gguf` file directly inside `dir`, if any. Non-recursive.
fn first_gguf_in(dir: &Path) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    for entry in rd.flatten() {
        let p = entry.path();
        let is_gguf = p.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if is_gguf && p.is_file() {
            return Some(p);
        }
    }
    None
}

/// BYO-GGUF: ask the user to point at an already-downloaded model through the
/// system file explorer. Returns the chosen path, or None (cancelled / unavailable)
/// to fall back to the optional download. Best-effort — must never panic the boot.
fn prompt_gguf_via_dialog() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("GGUF model", &["gguf"])
        .set_title("Select your .gguf model — or Cancel to download the recommended one")
        .pick_file()
}

/// BYO-GGUF: looks for a user-supplied model the user dropped into the models
/// dir, next to the app, or in `<app>/models`, so a fresh install can run fully
/// offline without the optional download. Returns the first match.
fn find_byo_gguf(model_dir: Option<&Path>) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = model_dir { dirs.push(d.to_path_buf()); }
    let bin = crate::config::app_bin_dir();
    dirs.push(bin.join("models"));
    dirs.push(bin);
    dirs.into_iter().find_map(|d| first_gguf_in(&d))
}

/// One download attempt already failed hard this run — later `ensure_ready`
/// calls (the smaller-model fallback chain) must not re-download hundreds of MB
/// just to hit the same wall. Seen live: a broken release made the chain fetch
/// the same 373 MB zip once per preset (~15 min of nothing).
static SERVER_DL_FAILED: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Downloads llama-server. Returns true if a GPU (Vulkan) build was downloaded,
/// false if a CPU build was used instead.
///
/// Robust against llama.cpp release-layout drift (all seen live, 2026-07-07):
///   * scans SEVERAL recent releases, not just `latest` — a fresh tag can ship
///     with no Windows builds at all (b9894 had only `cudart-*.zip`);
///   * never matches `cudart-*` packs — 373 MB of CUDA DLLs, no llama-server.exe;
///   * knows the renamed CPU build (`win-avx2-x64` → `win-cpu-x64`);
///   * if an archive turns out to have no llama-server.exe, tries the next
///     candidate instead of failing the whole boot.
async fn download_llama_server(dest_dir: &Path) -> Result<bool> {
    if let Some(msg) = SERVER_DL_FAILED.get() {
        anyhow::bail!("engine download already failed this run: {msg}");
    }
    match try_download_llama_server(dest_dir).await {
        Ok(gpu) => Ok(gpu),
        Err(e) => {
            let _ = SERVER_DL_FAILED.set(format!("{e:#}"));
            Err(e)
        }
    }
}

async fn try_download_llama_server(dest_dir: &Path) -> Result<bool> {
    std::fs::create_dir_all(dest_dir)?;

    let client = reqwest::Client::builder()
        .user_agent("nic-assistant/0.2.0")
        .connect_timeout(std::time::Duration::from_secs(15))
        // A stalled connection must become a visible error, not an eternal hang.
        .read_timeout(std::time::Duration::from_secs(60))
        .build()?;

    print!("[Setup] Fetching llama.cpp releases… ");
    let _ = std::io::stdout().flush();

    let rels: serde_json::Value = client
        .get("https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=8")
        .send().await
        .context("GitHub API unreachable")?
        .json().await
        .context("invalid JSON from GitHub API")?;
    let releases = rels.as_array().context("unexpected GitHub API response")?;

    let use_gpu = vulkan_available();
    // Ranked build flavors. Vulkan covers every modern GPU vendor; the plain CPU
    // build always runs (SLOWER, never broken). CUDA builds are deliberately
    // absent: they need the separate cudart DLL pack and die without it.
    let prefs: &[&str] = if use_gpu {
        &["win-vulkan-x64", "win-cpu-x64", "win-avx2-x64"]
    } else {
        &["win-cpu-x64", "win-avx2-x64", "win-noavx-x64"]
    };

    // One candidate per flavor — from the newest release that carries it.
    let mut candidates: Vec<(String, String, u64)> = Vec::new();
    for pref in prefs {
        'rel: for rel in releases {
            for a in rel["assets"].as_array().into_iter().flatten() {
                let name = a["name"].as_str().unwrap_or("");
                if name.starts_with("llama-") && name.ends_with(".zip") && name.contains(pref) {
                    if let Some(url) = a["browser_download_url"].as_str() {
                        if !candidates.iter().any(|(n, _, _)| n == name) {
                            let mb = a["size"].as_u64().unwrap_or(0) / 1_048_576;
                            candidates.push((name.to_string(), url.to_string(), mb));
                        }
                        break 'rel;
                    }
                }
            }
        }
    }
    anyhow::ensure!(
        !candidates.is_empty(),
        "no Windows llama-server build found in recent llama.cpp releases.\n\
         Download it manually: https://github.com/ggerganov/llama.cpp/releases\n\
         → llama-*-bin-win-vulkan-x64.zip (GPU) or llama-*-bin-win-cpu-x64.zip → unzip into {}",
        dest_dir.display()
    );

    let mut last_err = anyhow::anyhow!("no build candidates tried");
    for (name, url, mb) in candidates.iter().take(3) {
        println!("found {} ({} MB)", name, mb);
        let bytes = match download_bytes_progress(&client, url, "llama-server").await {
            Ok(b) => b,
            Err(e) => {
                println!("\n[Setup] Download failed ({e}) — trying the next build…");
                last_err = e;
                continue;
            }
        };
        println!("\n[Setup] Extracting {}…", name);
        if let Err(e) = extract_zip_exedll(&bytes, dest_dir) {
            println!("[Setup] Extract failed ({e}) — trying the next build…");
            last_err = e;
            continue;
        }
        if dest_dir.join("llama-server.exe").exists() {
            let gpu = name.contains("vulkan");
            println!("[Setup] llama-server ready: {} [{}]",
                     dest_dir.display(), if gpu { "GPU/Vulkan" } else { "CPU" });
            return Ok(gpu);
        }
        last_err = anyhow::anyhow!("{name} contains no llama-server.exe");
        println!("[Setup] {last_err} — trying the next build…");
    }
    Err(last_err.context("could not obtain a working llama-server build"))
}

async fn download_model(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        // A stalled connection must become a visible error, not an eternal hang.
        .read_timeout(std::time::Duration::from_secs(60))
        .build()?;

    let mut sources: Vec<String> = vec![url.to_string()];
    if url.contains("huggingface.co") {
        sources.push(url.replace("huggingface.co", "hf-mirror.com"));
    }

    let tmp = dest.with_extension("gguf.tmp");
    let mut last_err = anyhow::anyhow!("empty source list");

    for (i, src) in sources.iter().enumerate() {
        let label = if i == 0 { "HuggingFace" } else { "HF mirror" };
        println!("[Setup] Downloading from {}…", label);
        match stream_to_file(&client, src, &tmp).await {
            Ok(_) => {
                std::fs::rename(&tmp, dest)?;
                println!("[Setup] Model: {} ({:.0} MB)",
                    dest.display(),
                    dest.metadata().map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0));
                return Ok(());
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                println!("[Setup] Error ({}): {}", label, e);
                last_err = e;
            }
        }
    }
    Err(last_err)
}

async fn download_bytes_progress(
    client: &reqwest::Client,
    url:    &str,
    label:  &str,
) -> Result<Vec<u8>> {
    use futures::StreamExt;
    let resp  = client.get(url).send().await?;
    let total = resp.content_length().unwrap_or(0);
    let mut buf      = Vec::with_capacity(total as usize);
    let mut done     = 0u64;
    let mut last_pct = 0u64;
    let mut stream   = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        done += chunk.len() as u64;
        buf.extend_from_slice(&chunk);
        crate::boot::boot().set_progress(done / 1_048_576, total / 1_048_576);
        if total > 0 {
            let pct = done * 100 / total;
            if pct >= last_pct + 10 {
                print!("\r[Setup] Downloading {}… {}%   ", label, pct);
                let _ = std::io::stdout().flush();
                last_pct = pct;
            }
        }
    }
    Ok(buf)
}

async fn stream_to_file(
    client: &reqwest::Client,
    url:    &str,
    dest:   &Path,
) -> Result<()> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt as _;

    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }
    let total    = resp.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(dest).await?;
    let mut done     = 0u64;
    let mut last_pct = 0u64;
    let mut stream   = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        crate::boot::boot().set_progress(done / 1_048_576, total / 1_048_576);
        if total > 0 {
            let pct = done * 100 / total;
            if pct >= last_pct + 5 {
                print!("\r[Setup] {}%  ({}/{} MB)    ",
                    pct, done / 1_048_576, total / 1_048_576);
                let _ = std::io::stdout().flush();
                last_pct = pct;
            }
        }
    }
    file.flush().await?;
    Ok(())
}

fn extract_zip_exedll(bytes: &[u8], dest_dir: &Path) -> Result<()> {
    let cursor  = std::io::Cursor::new(bytes);
    let mut arc = zip::ZipArchive::new(cursor)?;

    for i in 0..arc.len() {
        let mut entry = arc.by_index(i)?;
        if entry.is_dir() { continue; }

        let name  = entry.name().to_string();
        let lower = name.to_lowercase();
        if !lower.ends_with(".exe") && !lower.ends_with(".dll") { continue; }

        let fname = std::path::Path::new(&name)
            .file_name()
            .context("zip entry has no filename")?;
        let out = dest_dir.join(fname);
        let mut f = std::fs::File::create(&out)?;
        std::io::copy(&mut entry, &mut f)?;
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique temp dir for an isolated, deterministic fs test.
    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nic_byo_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn first_gguf_in_finds_a_gguf() {
        let d = tmp("find");
        std::fs::write(d.join("notes.txt"), b"x").unwrap();
        let model = d.join("mymodel.gguf");
        std::fs::write(&model, b"x").unwrap();
        assert_eq!(first_gguf_in(&d).as_deref(), Some(model.as_path()));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn first_gguf_in_is_case_insensitive() {
        let d = tmp("case");
        let model = d.join("Model.GGUF");
        std::fs::write(&model, b"x").unwrap();
        assert!(first_gguf_in(&d).is_some());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn first_gguf_in_none_when_absent() {
        let d = tmp("none");
        std::fs::write(d.join("readme.md"), b"x").unwrap();
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        assert!(first_gguf_in(&d).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn first_gguf_in_missing_dir_is_none() {
        assert!(first_gguf_in(Path::new("C:/nic_does_not_exist_zzz")).is_none());
    }
}
