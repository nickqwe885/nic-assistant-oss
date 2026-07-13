use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Returns the directory containing the running binary.
/// Falls back to CWD if the path cannot be resolved (e.g. in unit tests).
pub fn app_bin_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct AppConfig {
    pub models:    ModelsConfig,
    pub server:    ServerConfig,
    pub capture:   CaptureConfig,
    pub librarian: LibrarianConfig,
    pub security:  SecurityConfig,
    pub adaptive:  AdaptiveConfig,
    pub profile:   ProfileConfig,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct ModelsConfig {
    /// BERT embedder for vector search (unchanged).
    pub embedder_dir:  PathBuf,
    /// Path to llama-server.exe — auto-downloaded on first run.
    pub server_bin:    PathBuf,
    /// Path to the GGUF model file — auto-downloaded on first run.
    pub model_path:    PathBuf,
    /// Download URL used when model_path does not exist.
    pub model_url:     String,
    /// Port for the llama-server HTTP API.
    pub server_port:   u16,
    /// Number of model layers to offload to GPU (-1 = all).
    pub n_gpu_layers:  i32,
    /// Where to run inference: "auto" (detect GPU), "gpu" (force GPU), or
    /// "cpu" (force CPU). Useful when the GPU is busy with games, or when an
    /// integrated GPU reports VRAM it can't really use well.
    #[serde(default = "default_compute")]
    pub compute:       String,
    /// Context window size in tokens.
    pub ctx_size:      usize,
    /// Number of parallel inference slots (default 2: Surfer + Thinker).
    pub parallel:      u8,
    /// ID of the currently selected model preset (must match one of `presets[*].id`).
    /// Changing this triggers a model re-download on next start.
    pub selected_model: String,
    /// When true, NIC picks the model to fit the machine's RAM/VRAM on startup
    /// (out-of-the-box experience). Set to false automatically once the user
    /// picks a model in the UI, so their choice is never overridden.
    #[serde(default = "default_true")]
    pub auto_select_model: bool,
    /// Available model presets (small / medium / large). Surfaced in the UI
    /// so the user can pick a model that fits their hardware.
    ///
    /// User-supplied presets in config.toml are *merged* onto the built-in
    /// defaults (deduped by `id`, user entry wins) — adding one custom model
    /// never drops the four built-ins.
    #[serde(deserialize_with = "merge_presets")]
    pub presets:       Vec<ModelPreset>,
}

/// Merges any user-supplied presets onto the built-in defaults, deduping by
/// `id` (a user entry with a matching id overrides the built-in). Built-ins
/// that the user did not redefine are preserved.
fn default_true() -> bool { true }
fn default_compute() -> String { "auto".to_string() }

fn merge_presets<'de, D>(deserializer: D) -> Result<Vec<ModelPreset>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let user: Vec<ModelPreset> = Vec::deserialize(deserializer)?;
    let mut merged = default_presets();
    for up in user {
        if let Some(existing) = merged.iter_mut().find(|p| p.id == up.id) {
            *existing = up;
        } else {
            merged.push(up);
        }
    }
    Ok(merged)
}

/// One selectable model. The `repo` + `file` pair is downloaded from
/// HuggingFace on first use; `size_mb` is shown in the UI to help the
/// user pick an option that fits their disk + RAM.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelPreset {
    /// Stable identifier — used in `selected_model` and in `/models/select`.
    pub id:          String,
    /// Human-readable name shown in the UI dropdown.
    pub name:        String,
    /// HuggingFace repo id (e.g. "Qwen/Qwen2.5-1.5B-Instruct-GGUF").
    pub repo:        String,
    /// GGUF filename inside the repo (e.g. "qwen2.5-1.5b-instruct-q4_k_m.gguf").
    pub file:        String,
    /// Approximate download size in MB (used in UI; not enforced).
    pub size_mb:     u32,
    /// Short description shown next to the name in the UI.
    pub description: String,
}

impl ModelPreset {
    /// Constructs the canonical HuggingFace download URL for this preset.
    pub fn download_url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.repo, self.file
        )
    }
}

impl ModelsConfig {
    /// Look up a preset by id. Returns `None` if the id is unknown — the
    /// caller should fall back to the currently selected preset in that case.
    pub fn preset(&self, id: &str) -> Option<&ModelPreset> {
        self.presets.iter().find(|p| p.id == id)
    }
}



/// Returns %LOCALAPPDATA%\nic-assistant, falling back to a relative path.
/// llama-server uses fopen() (ANSI) which fails on non-ASCII paths, so we
/// store downloaded binaries and models in AppData where the path is ASCII.
pub fn nic_data_dir() -> PathBuf {
    std::env::var("LOCALAPPDATA")
        .map(|p| PathBuf::from(p).join("nic-assistant"))
        .unwrap_or_else(|_| PathBuf::from("nic-data"))
}

impl Default for ModelsConfig {
    fn default() -> Self {
        let data = nic_data_dir();
        let presets = default_presets();
        // Default URL points to the medium preset so out-of-the-box behaviour
        // matches what the user sees as "Medium" in the UI.
        let medium = presets.iter()
            .find(|p| p.id == "qwen-1.5b")
            .expect("default presets must include qwen-1.5b");
        Self {
            // Embedder is loaded via Rust (handles Unicode) → keep next to exe.
            embedder_dir: PathBuf::from("models/all-MiniLM-L6-v2"),
            // llama-server and model use fopen() → must be on an ASCII path.
            server_bin:   data.join("llama/llama-server.exe"),
            model_path:   data.join("models/main.gguf"),
            model_url:    medium.download_url(),
            server_port:  8090,
            n_gpu_layers: -1,
            compute:      "auto".to_string(),
            ctx_size:     4096,
            parallel:     2,
            selected_model: medium.id.clone(),
            auto_select_model: true,
            presets,
        }
    }
}

/// Built-in list of selectable models. Surfaced verbatim in the UI; the
/// `selected_model` field of `ModelsConfig` must match one of these `id`s.
pub fn default_presets() -> Vec<ModelPreset> {
    vec![
        ModelPreset {
            id:          "qwen-0.5b".into(),
            name:        "Qwen 2.5 0.5B Instruct".into(),
            repo:        "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
            file:        "qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
            size_mb:     500,
            description: "Ultra-light. Fast on any CPU, weak reasoning.".into(),
        },
        ModelPreset {
            id:          "qwen-1.5b".into(),
            name:        "Qwen 2.5 1.5B Instruct".into(),
            repo:        "Qwen/Qwen2.5-1.5B-Instruct-GGUF".into(),
            file:        "qwen2.5-1.5b-instruct-q4_k_m.gguf".into(),
            size_mb:     1100,
            description: "Balanced default. Good quality, runs on most laptops.".into(),
        },
        ModelPreset {
            id:          "qwen-3b".into(),
            name:        "Qwen 2.5 3B Instruct".into(),
            repo:        "Qwen/Qwen2.5-3B-Instruct-GGUF".into(),
            file:        "qwen2.5-3b-instruct-q4_k_m.gguf".into(),
            size_mb:     2000,
            description: "Best quality. Needs 6+ GB RAM or any modern GPU.".into(),
        },
        ModelPreset {
            id:          "llama-3.2-3b".into(),
            name:        "Llama 3.2 3B Instruct".into(),
            repo:        "bartowski/Llama-3.2-3B-Instruct-GGUF".into(),
            file:        "Llama-3.2-3B-Instruct-Q4_K_M.gguf".into(),
            size_mb:     2000,
            description: "Meta's 3B. Multilingual, strong at code & chat.".into(),
        },
    ]
}


#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct ServerConfig {
    pub ports: Vec<u16>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { ports: vec![7878, 7879, 7880] }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct CaptureConfig {
    pub active_interval_secs:   u64,
    pub idle_interval_secs:     u64,
    pub idle_after_secs:        u64,
    pub dirty_rect_threshold:   f32,
    pub phash_hamming_threshold: u32,
    pub privacy_blacklist:      Vec<String>,
    pub clip_blacklist:         Vec<String>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            active_interval_secs:    3,
            idle_interval_secs:      20,
            idle_after_secs:         30,
            dirty_rect_threshold:    0.05,
            phash_hamming_threshold: 5,
            privacy_blacklist: vec![],
            clip_blacklist:     vec![],
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct LibrarianConfig {
    pub db_path:            PathBuf,
    pub archive_path:       PathBuf,
    pub embed_cache_size:   usize,
    pub surfer_timeout_secs: u64,
    pub llm_timeout_secs:   u64,
    pub search_timeout_secs: u64,
    pub deep_miner_idle_secs: u64,
    /// Hours after which L0 raw events are pruned from hot DB (default 72 = 3 days).
    pub l0_ttl_hours: i64,
}

impl Default for LibrarianConfig {
    fn default() -> Self {
        Self {
            db_path:              PathBuf::from("data/lancedb"),
            archive_path:         PathBuf::from("data/archive"),
            embed_cache_size:     100,
            surfer_timeout_secs:  10,
            llm_timeout_secs:     30,
            search_timeout_secs:  5,
            deep_miner_idle_secs: 1800,
            l0_ttl_hours:         72,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct SecurityConfig {
    pub encrypt_archives: bool,
    pub key_file:         PathBuf,
    /// If non-empty, non-localhost requests must include `X-Api-Key: <value>` header.
    /// Leave empty (default) to disable — UI on localhost always works without a key.
    pub api_key:          String,
}

fn default_key_path() -> PathBuf {
    // Store key outside the data directory so the archive and key are never co-located.
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata).join("nic-assistant").join(".nic_key");
    }
    // Fallback for non-Windows / missing env
    PathBuf::from("data/.nic_key")
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            encrypt_archives: false,
            key_file: default_key_path(),
            api_key:  String::new(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct AdaptiveConfig {
    /// Time of day the daily summary is generated, "HH:MM".
    pub summary_time:    String,
    /// When true, Sentinel writes no OCR (incognito mode).
    pub incognito_mode:  bool,
    /// CPU threshold (%) above which Thinker hibernates.
    pub cpu_limit_pct:   f32,
    /// Language for LLM responses, e.g. "Russian", "English".
    /// Empty string = auto-detect from OS locale at startup.
    pub language:        String,
    /// City name for geo-aware queries (e.g. weather, news).
    /// Empty string = auto-detect from OS timezone at startup.
    pub city:            String,
    /// Proactive reminders (Initiative). When false, NIC never pops a
    /// notification on its own. Rate-limited to a couple per day regardless.
    #[serde(default = "default_true")]
    pub initiative_enabled: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            summary_time:   "18:00".to_string(),
            incognito_mode: false,
            cpu_limit_pct:  90.0,
            language:       String::new(),
            city:           String::new(),
            initiative_enabled: true,
        }
    }
}

/// Who the user is — injected into every Thinker prompt so answers are personalised.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct ProfileConfig {
    /// Display name, e.g. "Alex"
    pub name:        String,
    /// Professional role, e.g. "Senior Rust developer"
    pub role:        String,
    /// Active projects, e.g. ["NIC-Assistant", "billing-service"]
    pub projects:    Vec<String>,
    /// Free-form style preferences, e.g. "I prefer short, technical answers"
    pub preferences: String,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            name:        String::new(),
            role:        String::new(),
            projects:    vec![],
            preferences: String::new(),
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            models:    ModelsConfig::default(),
            server:    ServerConfig::default(),
            capture:   CaptureConfig::default(),
            librarian: LibrarianConfig::default(),
            security:  SecurityConfig::default(),
            adaptive:  AdaptiveConfig::default(),
            profile:   ProfileConfig::default(),
        }
    }
}

pub fn load_config() -> Result<AppConfig> {
    let bin_dir = app_bin_dir();

    // Look for config.toml next to the binary first, then fall back to CWD.
    let config_path = [bin_dir.join("config.toml"), PathBuf::from("config.toml")]
        .into_iter()
        .find(|p| p.exists());

    let mut cfg = if let Some(path) = config_path {
        let content = std::fs::read_to_string(&path)?;
        toml::from_str(&content).map_err(|e| anyhow::anyhow!("Config parse error: {}", e))?
    } else {
        tracing::info!("config.toml not found — using built-in defaults");
        AppConfig::default()
    };

    // Resolve all relative paths against the binary directory so the binary
    // is self-contained regardless of the working directory at launch time.
    cfg.resolve_paths(&bin_dir);
    Ok(cfg)
}

impl AppConfig {
    pub fn resolve_paths(&mut self, base: &Path) {
        resolve_rel(&mut self.librarian.db_path,      base);
        resolve_rel(&mut self.librarian.archive_path, base);
        resolve_rel(&mut self.security.key_file,      base);
        resolve_rel(&mut self.models.embedder_dir,    base);
        resolve_rel(&mut self.models.server_bin,      base);
        resolve_rel(&mut self.models.model_path,      base);
    }
}

fn resolve_rel(path: &mut PathBuf, base: &Path) {
    if path.is_relative() {
        *path = base.join(path.as_path());
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── resolve_rel ───────────────────────────────────────────────────────────

    #[test]
    fn resolve_rel_relative_gets_joined() {
        let base = PathBuf::from("C:\\app");
        let mut p = PathBuf::from("data/db");
        resolve_rel(&mut p, &base);
        assert_eq!(p, PathBuf::from("C:\\app\\data/db"));
    }

    #[test]
    fn resolve_rel_absolute_unchanged() {
        let base = PathBuf::from("C:\\app");
        let mut p = PathBuf::from("C:\\absolute\\path");
        let orig = p.clone();
        resolve_rel(&mut p, &base);
        assert_eq!(p, orig);
    }

    #[test]
    fn resolve_rel_empty_base() {
        let base = PathBuf::from("");
        let mut p = PathBuf::from("relative");
        resolve_rel(&mut p, &base);
        // empty base join still doesn't panic
        assert!(p.to_str().is_some());
    }

    #[test]
    fn resolve_rel_called_twice_no_double_join() {
        let base = PathBuf::from("C:\\app");
        let mut p = PathBuf::from("data");
        resolve_rel(&mut p, &base);
        let after_first = p.clone();
        resolve_rel(&mut p, &base); // second call — now p is absolute
        assert_eq!(p, after_first, "second resolve_rel should be no-op on already-absolute path");
    }

    // ── AppConfig defaults ────────────────────────────────────────────────────

    #[test]
    fn app_config_default_server_port() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.models.server_port, 8090);
    }

    #[test]
    fn app_config_default_server_ports() {
        let cfg = AppConfig::default();
        assert!(cfg.server.ports.contains(&7878));
        assert!(cfg.server.ports.contains(&7879));
        assert!(cfg.server.ports.contains(&7880));
    }

    #[test]
    fn app_config_default_ctx_size() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.models.ctx_size, 4096);
    }

    #[test]
    fn app_config_default_l0_ttl() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.librarian.l0_ttl_hours, 72);
    }

    #[test]
    fn app_config_default_embed_cache_size() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.librarian.embed_cache_size, 100);
    }

    #[test]
    fn app_config_default_cpu_limit() {
        let cfg = AppConfig::default();
        assert!((cfg.adaptive.cpu_limit_pct - 90.0).abs() < 0.001);
    }

    #[test]
    fn app_config_default_incognito_false() {
        let cfg = AppConfig::default();
        assert!(!cfg.adaptive.incognito_mode);
    }

    #[test]
    fn app_config_default_encrypt_archives_false() {
        let cfg = AppConfig::default();
        assert!(!cfg.security.encrypt_archives);
    }

    #[test]
    fn app_config_default_api_key_empty() {
        let cfg = AppConfig::default();
        assert!(cfg.security.api_key.is_empty());
    }

    #[test]
    fn app_config_default_summary_time() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.adaptive.summary_time, "18:00");
    }

    #[test]
    fn app_config_default_n_gpu_layers_all() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.models.n_gpu_layers, -1);
    }

    #[test]
    fn app_config_default_parallel_2() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.models.parallel, 2);
    }

    // ── ProfileConfig defaults ────────────────────────────────────────────────

    #[test]
    fn profile_default_fields_empty() {
        let p = ProfileConfig::default();
        assert!(p.name.is_empty());
        assert!(p.role.is_empty());
        assert!(p.projects.is_empty());
        assert!(p.preferences.is_empty());
    }

    #[test]
    fn profile_fields_set_and_clone() {
        let p = ProfileConfig {
            name:        "Alex".to_string(),
            role:        "Rust dev".to_string(),
            projects:    vec!["NIS".to_string()],
            preferences: "short answers".to_string(),
        };
        let q = p.clone();
        assert_eq!(q.name, "Alex");
        assert_eq!(q.projects, vec!["NIS"]);
    }

    // ── TOML deserialization ──────────────────────────────────────────────────

    #[test]
    fn toml_partial_override_server_port() {
        let toml = r#"
[models]
server_port = 9000
"#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.models.server_port, 9000);
        // Unset fields use Default
        assert_eq!(cfg.models.ctx_size, 4096);
    }

    #[test]
    fn toml_profile_roundtrip() {
        let toml = r#"
[profile]
name        = "Alex"
role        = "Backend engineer"
projects    = ["billing", "nic-assistant"]
preferences = "short answers"
"#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.profile.name, "Alex");
        assert_eq!(cfg.profile.projects.len(), 2);
        assert_eq!(cfg.profile.preferences, "short answers");
    }

    #[test]
    fn toml_security_api_key() {
        let toml = r#"
[security]
api_key = "secret123"
"#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.security.api_key, "secret123");
    }

    #[test]
    fn toml_capture_blacklist() {
        let toml = r#"
[capture]
privacy_blacklist = ["banking", "passwords"]
"#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.capture.privacy_blacklist, vec!["banking", "passwords"]);
    }

    #[test]
    fn toml_empty_string_uses_all_defaults() {
        let cfg: AppConfig = toml::from_str("").unwrap();
        // All fields come from Default
        assert_eq!(cfg.models.server_port, 8090);
        assert_eq!(cfg.adaptive.summary_time, "18:00");
    }

    #[test]
    fn toml_adaptive_cpu_limit() {
        let toml = r#"
[adaptive]
cpu_limit_pct = 70.0
"#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert!((cfg.adaptive.cpu_limit_pct - 70.0).abs() < 0.001);
    }

    // ── resolve_paths ─────────────────────────────────────────────────────────

    #[test]
    fn resolve_paths_makes_db_absolute() {
        let mut cfg = AppConfig::default();
        let base = PathBuf::from("C:\\nic");
        cfg.resolve_paths(&base);
        assert!(cfg.librarian.db_path.is_absolute());
    }

    #[test]
    fn resolve_paths_absolute_paths_unchanged() {
        let mut cfg = AppConfig::default();
        cfg.models.model_path = PathBuf::from("C:\\models\\main.gguf");
        let base = PathBuf::from("C:\\nic");
        cfg.resolve_paths(&base);
        assert_eq!(cfg.models.model_path, PathBuf::from("C:\\models\\main.gguf"));
    }

    // ── CaptureConfig defaults ────────────────────────────────────────────────

    #[test]
    fn capture_default_active_interval() {
        let c = CaptureConfig::default();
        assert_eq!(c.active_interval_secs, 3);
    }

    #[test]
    fn capture_default_dirty_rect_threshold() {
        let c = CaptureConfig::default();
        assert!((c.dirty_rect_threshold - 0.05).abs() < 0.0001);
    }

    #[test]
    fn capture_default_phash_threshold() {
        let c = CaptureConfig::default();
        assert_eq!(c.phash_hamming_threshold, 5);
    }

    #[test]
    fn capture_blacklists_empty_by_default() {
        let c = CaptureConfig::default();
        assert!(c.privacy_blacklist.is_empty());
        assert!(c.clip_blacklist.is_empty());
    }

    // ── ModelPreset ─────────────────────────────────────────────────────────

    #[test]
    fn default_presets_has_four_entries() {
        let presets = default_presets();
        assert_eq!(presets.len(), 4, "expected exactly 4 built-in presets");
    }

    #[test]
    fn default_presets_ids_are_unique() {
        let presets = default_presets();
        let mut ids: Vec<&str> = presets.iter().map(|p| p.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), presets.len(), "preset ids must be unique");
    }

    #[test]
    fn default_presets_all_have_https_url() {
        for p in default_presets() {
            let url = p.download_url();
            assert!(url.starts_with("https://huggingface.co/"), "bad url: {url}");
            assert!(url.ends_with(".gguf"), "expected .gguf endpoint, got: {url}");
        }
    }

    #[test]
    fn app_config_default_selected_is_medium() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.models.selected_model, "qwen-1.5b",
            "default selection should be the medium (1.5B) model");
    }

    #[test]
    fn app_config_default_url_matches_selected() {
        let cfg = AppConfig::default();
        let preset = cfg.models.presets.iter()
            .find(|p| p.id == cfg.models.selected_model)
            .expect("selected_model must be one of the presets");
        assert_eq!(cfg.models.model_url, preset.download_url());
    }

    #[test]
    fn toml_preset_roundtrip() {
        let toml = r#"
[models]
selected_model = "qwen-3b"
"#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.models.selected_model, "qwen-3b");
        // Presets still come from Default
        assert_eq!(cfg.models.presets.len(), 4);
    }

    #[test]
    fn toml_full_preset_block_roundtrip() {
        let toml = r#"
[[models.presets]]
id          = "custom-x"
name        = "Custom X"
repo        = "me/custom-x-GGUF"
file        = "custom-x-q4.gguf"
size_mb     = 999
description = "test"

[models]
selected_model = "custom-x"
"#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.models.presets.len(), 5); // 4 default + 1 custom
        assert_eq!(cfg.models.selected_model, "custom-x");
        assert_eq!(cfg.models.presets[4].size_mb, 999);
    }

    #[test]
    fn models_config_has_preset_helper() {
        let cfg = AppConfig::default();
        assert!(cfg.models.preset(&cfg.models.selected_model).is_some());
        assert!(cfg.models.preset("nonexistent").is_none());
    }
}


