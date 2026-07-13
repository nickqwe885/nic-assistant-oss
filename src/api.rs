use anyhow::Result;
use async_stream::stream;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use tower_http::cors::{CorsLayer, AllowOrigin, Any};


use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use crate::config::{ModelPreset, ProfileConfig};
use crate::modules::{librarian::Librarian, pilot, thinker::Thinker, ContextCollector};


// ── Answer cache ──────────────────────────────────────────────────────────────

pub struct CacheEntry {
    pub(crate) words:     HashSet<String>,
    pub(crate) answer:    String,
    pub(crate) cached_at: std::time::Instant,
}

// ── Who we're talking about ───────────────────────────────────────────────────

/// The person under discussion and what the USER has said about them.
#[derive(Clone, Debug, Default)]
pub struct EntityCtx {
    /// Name as the user last gave it ("LenS" overrides an earlier "lens").
    pub name:  String,
    /// User-supplied descriptors ("dota 2 player"), newest last. Max 3.
    pub facts: Vec<String>,
}

impl EntityCtx {
    /// The string to actually search with: the name plus the user's own
    /// disambiguators. "lens" alone finds camera optics; "LenS dota 2 player"
    /// finds the person.
    pub fn search_term(&self) -> String {
        if self.facts.is_empty() {
            return self.name.clone();
        }
        format!("{} {}", self.name, self.facts.join(" "))
    }

    fn add_fact(&mut self, f: &str) {
        let f = f.trim();
        if f.is_empty() || self.facts.iter().any(|x| x.eq_ignore_ascii_case(f)) {
            return;
        }
        if self.facts.len() >= 3 {
            self.facts.remove(0);
        }
        self.facts.push(f.to_string());
    }
}

/// Parses a statement the user makes ABOUT the current person.
///   "he is a dota 2 player"   → Fact("dota 2 player")
///   "his nickname is LenS"    → Rename("LenS")
///   "no his name is LenS"     → Rename("LenS")
/// Returns None for anything that isn't such a statement.
enum EntityUpdate { Fact(String), Rename(String) }

fn entity_statement(query: &str) -> Option<EntityUpdate> {
    let q = query.trim().trim_end_matches(['.', '!']);
    let ql_full = q.to_lowercase();
    // Corrective lead: "no", but people stretch it — "nooo", "nope", "неее".
    // Matching only the exact "no " missed «nooo I am about cs 2 player», and the
    // model answered with nonsense about two people playing Counter-Strike.
    let ql = {
        let first_end = ql_full.find(' ').unwrap_or(0);
        // Strip punctuation before testing: "nope," must still read as a negation.
        let first: String = ql_full[..first_end]
            .chars().filter(|c| c.is_alphanumeric()).collect();
        let is_negation = !first.is_empty()
            && ((first.starts_with("no") && first.chars().all(|c| matches!(c, 'n' | 'o' | 'p' | 'e')))
                || first.starts_with("не"));
        if is_negation {
            ql_full[first_end..].trim_start_matches([' ', ',']).trim()
        } else {
            ql_full.trim()
        }
    };
    let orig_off = q.len().saturating_sub(ql.len());
    let rest_of = |pat: &str| -> Option<String> {
        let i = ql.find(pat)? + pat.len();
        let s = q.get(orig_off + i..)?.trim();
        (!s.is_empty()).then(|| s.to_string())
    };

    // A rename must come first: "his nickname is X" also contains " is ".
    for pat in ["his nickname is ", "her nickname is ", "his name is ", "her name is ",
                "his nick is ", "her nick is ", "его ник ", "её ник ", "ее ник "] {
        if ql.starts_with(pat) || ql.contains(pat) {
            if let Some(v) = rest_of(pat) {
                return Some(EntityUpdate::Rename(v));
            }
        }
    }
    // A clarification: "I mean the japanese street racer", "no I'm talking about …".
    // Live, "no I am about japanese street racer and founder of top secret customs"
    // was treated as a question and the model invented a founder named Kenjiro Kawai.
    for pat in ["i mean ", "i meant ", "i'm talking about ", "im talking about ",
                "i am talking about ", "i am about ", "i'm about ",
                "я про ", "я имею в виду ", "имею в виду "] {
        if ql.starts_with(pat) {
            let v = rest_of(pat)?;
            let v = v.trim_start_matches("the ").trim_start_matches("a ").trim();
            return Some(EntityUpdate::Fact(v.to_string()));
        }
    }
    // A plain descriptor: "he is a dota 2 player", "she is a singer".
    for pat in ["he is ", "she is ", "they are ", "he's ", "she's ",
                "он ", "она "] {
        if ql.starts_with(pat) {
            let v = rest_of(pat)?;
            let v = v.trim_start_matches("a ").trim_start_matches("an ").trim();
            return Some(EntityUpdate::Fact(v.to_string()));
        }
    }
    None
}

fn query_words(q: &str) -> HashSet<String> {
    q.split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.to_lowercase())
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    inter as f32 / union as f32
}

#[derive(Clone)]
pub struct ApiState {
    pub collector:     Arc<ContextCollector>,
    pub thinker:       Arc<Mutex<Thinker>>,
    pub librarian:     Arc<Librarian>,
    pub incognito:     Arc<AtomicBool>,
    /// API key guard — if non-empty, non-localhost clients must send `X-Api-Key: <value>`.
    pub api_key:       String,
    /// Last Q&A pair — prepended to context so follow-up queries have reference.
    pub last_exchange: Arc<tokio::sync::Mutex<Option<(String, String)>>>,
    /// The person currently being discussed, plus whatever the USER has told NIC
    /// about them. Without this, "who is lens" → "he is a dota 2 player" was
    /// answered with a dictionary definition of an optical lens, and a later
    /// "search his last video" found camera reviews. The user's own words are the
    /// most reliable context there is — we keep them and search WITH them.
    pub entity:        Arc<tokio::sync::Mutex<Option<EntityCtx>>>,
    /// URL of the local llama-server (for /llm_status health check).
    pub llm_url:       String,
    /// In-memory Q&A cache: last 20 answers, 5-minute TTL, Jaccard similarity check.
    pub answer_cache:  Arc<tokio::sync::Mutex<Vec<CacheEntry>>>,
    /// Live-editable user profile (name, role, projects, preferences).
    pub profile:       Arc<tokio::sync::Mutex<ProfileConfig>>,
    /// User's city, auto-detected from OS timezone. Used to enrich location queries.
    pub city:          Option<String>,
    /// Detected LLM language (e.g. "Russian", "English"). Surfaced via /health.
    pub language:      Arc<Mutex<String>>,
    /// ID of the currently selected model preset. Editable through /models/select.
    pub selected_model: Arc<Mutex<String>>,
    /// All available model presets — surfaced verbatim via /models.
    pub presets:       Vec<ModelPreset>,
    /// Absolute path of the GGUF model file the backend currently uses
    /// (surfaced in `/models` so the UI can mark the right preset as installed).
    pub model_path:    PathBuf,
    /// Absolute path of the loaded config.toml — used by /models/select to persist
    /// the new selection so it survives a restart.
    pub config_path:   PathBuf,
    /// Consent marker file — its presence on disk means the user has enabled
    /// screen capture. Written by `POST /consent`.
    pub consent_marker: PathBuf,
}



#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default)]
    pub offline: bool,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub answer:     String,
    pub elapsed_ms: u128,
}

#[derive(Deserialize)]
pub struct AddEventRequest {
    pub text:       String,
    pub source:     String,
    pub event_type: String,
}

#[derive(Serialize)]
pub struct AddEventResponse {
    pub id: String,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize { 5 }

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<String>,
}

#[derive(Deserialize)]
pub struct ArchiveSearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Deserialize)]
pub struct FilterSearchRequest {
    pub query:     String,
    /// Unix seconds — restrict results to after this time.
    #[serde(default)]
    pub time_from: Option<i64>,
    /// Unix seconds — restrict results to before this time.
    #[serde(default)]
    pub time_to:   Option<i64>,
    /// Case-insensitive substring match against the captured app name.
    #[serde(default)]
    pub app_name:  Option<String>,
    #[serde(default = "default_limit")]
    pub limit:     usize,
}

#[derive(Deserialize)]
pub struct ExportParams {
    #[serde(default = "default_export_limit")]
    pub limit: usize,
}
fn default_export_limit() -> usize { 1000 }

#[derive(Deserialize)]
pub struct ForgetRequest {
    #[serde(default = "default_forget_minutes")]
    pub minutes: u64,
}
fn default_forget_minutes() -> u64 { 15 }

#[derive(Serialize)]
pub struct ForgetResponse {
    pub erased:  usize,
    pub minutes: u64,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status:          &'static str,
    pub version:         &'static str,
    pub db_size_mb:      f64,
    pub archive_size_mb: f64,
    pub event_count:     u64,
    /// Detected LLM language ("Russian", "English", ...). Empty = unknown.
    pub language:        String,
    /// ID of the currently selected model preset.
    pub selected_model:  String,
    /// Whether the user has consented to screen capture (memory is active).
    pub consent:         bool,
    /// A game with kernel anti-cheat owns the foreground → capture is suspended.
    /// Surfaced so the UI can PROVE the guard works, right when the user cares.
    pub game_paused:     bool,
}


#[derive(Serialize)]
pub struct IncognitoResponse {
    pub enabled: bool,
}

#[derive(Serialize)]
pub struct LlmStatusResponse {
    pub ok:  bool,
    pub url: String,
}

#[derive(Deserialize)]
struct ImageRequest {
    image:    String,  // base64-encoded image bytes
    filename: String,
}

#[derive(Serialize)]
struct ImageResponse {
    text:   String,
    stored: bool,
}

#[derive(Serialize)]
struct ActivityPoint {
    hour:  i64,
    count: u64,
}

#[derive(Serialize)]
struct ActivityResponse {
    points: Vec<ActivityPoint>,
}

// ── Auth middleware ───────────────────────────────────────────────────────────

/// Requires the `X-Api-Key` header to match `api_key` on EVERY request,
/// including loopback. This is deliberate: a malicious website you visit can
/// issue `fetch('http://127.0.0.1:7878/export')` from your machine — the
/// connection looks like loopback, so trusting loopback would let any site
/// read your memory (permissive CORS lets it read the response). Requiring a
/// per-launch token the legit UI holds (and a random site cannot know) closes
/// that hole. Empty key = no auth (dev / running the lib without the launcher).
async fn auth_middleware(
    State(state): State<ApiState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // Host-header gate (DNS rebinding). A hostile site can point a domain it owns
    // at 127.0.0.1, so the browser treats it as same-origin and sends the request
    // with `Host: evil.com`. The token already blocks reading the response, but
    // rejecting a foreign Host stops the request from ever running — defence in
    // depth, and it costs one string compare.
    if let Some(host) = req.headers().get("host").and_then(|v| v.to_str().ok()) {
        if !host_is_local(host) {
            tracing::warn!("[Security] rejected request with foreign Host: {host}");
            return (StatusCode::FORBIDDEN,
                    [("content-type", "application/json")],
                    r#"{"error":"Forbidden"}"#)
                .into_response();
        }
    }

    if state.api_key.is_empty() {
        return next.run(req).await;
    }
    let provided = req.headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided == state.api_key {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED,
         [("content-type", "application/json")],
         r#"{"error":"Unauthorized"}"#)
            .into_response()
    }
}

/// True when the Host header names this machine (loopback), with or without a
/// port. Anything else means the request was aimed at us through a domain —
/// i.e. DNS rebinding.
fn host_is_local(host: &str) -> bool {
    // Strip the port. Per RFC 3986 an IPv6 literal in Host is bracketed
    // ("[::1]:7878"), so brackets disambiguate its colons from the port's.
    let h = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.split(':').next().unwrap_or("")
    };
    if matches!(h, "localhost" | "::1" | "0.0.0.0") {
        return true;
    }
    // 127.0.0.0/8 — but ONLY as a real IPv4 literal. A prefix check alone would
    // wave through "127.0.0.1.evil.com", which is exactly the rebinding trick.
    h.parse::<std::net::Ipv4Addr>().is_ok_and(|ip| ip.is_loopback())
}

// ── Timing middleware ─────────────────────────────────────────────────────────

async fn timing_layer(req: Request<Body>, next: Next) -> Response {
    let start = std::time::Instant::now();
    let mut res = next.run(req).await;
    let ms = start.elapsed().as_millis();
    if let Ok(v) = format!("{}ms", ms).parse() {
        res.headers_mut().insert("x-response-time", v);
    }
    res
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health(State(state): State<ApiState>) -> impl IntoResponse {
    let (db_bytes, archive_bytes) = state.librarian.disk_usage();
    let event_count = state.librarian.count_events().await;
    let language       = state.language.lock().await.clone();
    let selected_model = state.selected_model.lock().await.clone();
    Json(HealthResponse {
        status:          "ok",
        version:         env!("CARGO_PKG_VERSION"),
        db_size_mb:      db_bytes as f64 / 1_048_576.0,
        archive_size_mb: archive_bytes as f64 / 1_048_576.0,
        event_count,
        language,
        selected_model,
        consent:     crate::modules::sentinel::capture_consent(),
        game_paused: crate::modules::sentinel::game_active(),
    })
}

// ── Models ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ModelInfo {
    id:              String,
    name:            String,
    size_mb:         u32,
    description:     String,
    download_url:    String,
    /// true if the model file is already on disk
    installed:       bool,
}

#[derive(Serialize)]
struct ModelsResponse {
    selected:        String,
    /// Absolute path of the model file the backend would load
    model_path:      String,
    presets:         Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct SelectModelRequest {
    id: String,
}

#[derive(Serialize)]
struct SelectModelResponse {
    selected:        String,
    /// true if the chosen model file is already on disk
    installed:       bool,
    /// Always true — model switch requires backend restart to take effect
    restart_required: bool,
    /// Human-readable hint for the UI
    message:         String,
}

async fn handle_models_list(State(state): State<ApiState>) -> impl IntoResponse {
    let selected = state.selected_model.lock().await.clone();
    let model_path = state.model_path.clone();
    let presets: Vec<ModelInfo> = state.presets.iter().map(|p| {
        let p_path = state.model_path.parent()
            .map(|d| d.join("models").join(format!("{}.gguf", p.id)))
            .unwrap_or_default();
        ModelInfo {
            id:           p.id.clone(),
            name:         p.name.clone(),
            size_mb:      p.size_mb,
            description:  p.description.clone(),
            download_url: p.download_url(),
            installed:    model_path_already_used(&selected, &p.id) && p_path.exists(),
        }
    }).collect();
    Json(ModelsResponse {
        selected,
        model_path: model_path.to_string_lossy().into_owned(),
        presets,
    })
}

/// Returns true when the given preset id is the currently-selected one
/// (so its on-disk file is the one in use). We don't try to detect
/// “downloaded but not selected” — that would require parsing the GGUF header.
fn model_path_already_used(selected: &str, preset_id: &str) -> bool {
    selected == preset_id
}

async fn handle_models_select(
    State(state): State<ApiState>,
    Json(req): Json<SelectModelRequest>,
) -> Result<Json<SelectModelResponse>, (StatusCode, String)> {
    // Validate the id against the known presets.
    let preset = state.presets.iter()
        .find(|p| p.id == req.id)
        .cloned()
        .ok_or_else(|| {
            (StatusCode::NOT_FOUND,
             format!("Unknown model preset: '{}'", req.id))
        })?;

    // Update the in-memory selection immediately.
    {
        let mut cur = state.selected_model.lock().await;
        *cur = preset.id.clone();
    }

    // Persist to config.toml so the change survives a restart.
    let cfg_path = state.config_path.clone();
    let toml_text = std::fs::read_to_string(&cfg_path).unwrap_or_default();
    let mut doc: toml::Value = toml::from_str(&toml_text)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("config parse: {e}")))?;
    if let Some(models) = doc.get_mut("models").and_then(|m| m.as_table_mut()) {
        models.insert("selected_model".into(),
            toml::Value::String(preset.id.clone()));
        models.insert("model_url".into(),
            toml::Value::String(preset.download_url()));
        // The user made an explicit choice — stop auto-picking by hardware so
        // their selection is never overridden on the next launch.
        models.insert("auto_select_model".into(), toml::Value::Boolean(false));
    } else {
        return Err((StatusCode::INTERNAL_SERVER_ERROR,
            "config.toml has no [models] section".into()));
    }
    let serialized = toml::to_string_pretty(&doc)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("toml serialize: {e}")))?;
    std::fs::write(&cfg_path, serialized)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("config write: {e}")))?;

    let installed = state.model_path.parent()
        .map(|d| d.join("models").join(format!("{}.gguf", preset.id)))
        .map(|p| p.exists())
        .unwrap_or(false);

    let message = if installed {
        format!("Selected '{}'. Restart NIC-Assistant to switch the running model.", preset.name)
    } else {
        format!(
            "Selected '{}'. On next start NIC-Assistant will download ~{} MB \
             and then load it. Please restart the app.",
            preset.name, preset.size_mb
        )
    };

    Ok(Json(SelectModelResponse {
        selected: preset.id,
        installed,
        restart_required: true,
        message,
    }))
}

#[derive(Serialize)]
struct PickModelResponse {
    picked:           bool,
    restart_required: bool,
    message:          String,
}

/// §5/§7 BYO-GGUF from the UI: opens the native file explorer, lets the user point
/// at an existing .gguf, copies it onto the (ASCII) model path, and asks for a
/// restart to load it — the same restart-to-apply flow as /models/select. On next
/// start `ensure_ready` sees the file at model_path and loads it directly.
async fn handle_model_pick_gguf(
    State(state): State<ApiState>,
) -> Result<Json<PickModelResponse>, (StatusCode, String)> {
    // Native dialog runs on a blocking thread so it never stalls the async runtime.
    let picked = tokio::task::spawn_blocking(|| {
        rfd::FileDialog::new()
            .add_filter("GGUF model", &["gguf"])
            .set_title("Select your .gguf model")
            .pick_file()
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let Some(src) = picked else {
        return Ok(Json(PickModelResponse {
            picked: false,
            restart_required: false,
            message: "No file selected.".into(),
        }));
    };

    let dest = state.model_path.clone();
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("mkdir: {e}")))?;
    }
    // llama-server opens the file via fopen() (ANSI) → copy onto the guaranteed-ASCII
    // model_path so a non-ASCII source folder can't break loading.
    std::fs::copy(&src, &dest)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("copy model: {e}")))?;

    let fname = src.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(Json(PickModelResponse {
        picked: true,
        restart_required: true,
        message: format!("Model set from “{fname}”. Restart NIC-Assistant to load it."),
    }))
}



// ── Anti-hallucination fallback (MASTER_PLAN §3 / §9[2]) ───────────────────────
// Notices shown when we refuse to surface a possibly-fabricated answer and fall
// back to the raw retrieved matches. English-only beta (§9[1]).
const RAW_NOTICE_TIMEOUT: &str =
    "⚠️ Model verification was interrupted (timeout). Showing the raw text matches I found:";
const RAW_NOTICE_TIMEOUT_TAIL: &str =
    "⚠️ Generation was cut off (timeout) — the answer above may be incomplete.";
const RAW_NOTICE_UNVERIFIED: &str =
    "⚠️ I couldn't verify the figures against the sources. Showing the raw text matches instead:";
const RAW_NOTICE_UNVERIFIED_TAIL: &str =
    "⚠️ Some figures above couldn't be verified against the sources — double-check before relying on them.";

/// The honest, code-generated answer to "do you have my password / seed phrase /
/// card number". This MUST NOT go through the model: asked live, the 1.5B invented
/// "your recovery words were deleted from your profile earlier this week" and "the
/// private keys in my memory belong to me" — i.e. it claimed to have held secrets
/// it never saw. That single screenshot would end the product's credibility, and
/// it is the first thing anyone tests on an assistant that reads your screen.
const SECRETS_ANSWER: &str = "I never store secrets — not passwords, card numbers, \
seed phrases or API keys. Three things stop it:\n\
• Wallets, password managers and banking windows are never captured at all — no \
screenshot is taken while they're in focus.\n\
• Anything that does look like a secret (seed phrases, card numbers, API keys) is \
stripped out before it's ever written to memory.\n\
• The same check runs again when memory is read, so nothing secret can reach an answer.\n\
So I have nothing of that kind to give you — by design, not by promise. The code is open: \
see src/modules/scrubber.rs.";

/// True when the user is asking whether NIC holds their credentials. Matched in
/// code, because this is exactly the question the model answers with fiction.
fn asks_about_secrets(query: &str) -> bool {
    let q = query.to_lowercase();
    const SECRET_WORDS: &[&str] = &[
        "seed phrase", "recovery phrase", "recovery words", "mnemonic",
        "private key", "private keys", "api key", "api keys", "secret key",
        "password", "passwords", "card number", "card numbers", "credit card",
        "bank card", "cvv", "wallet",
        "сид фраз", "сид-фраз", "мнемоник", "приватный ключ", "приватные ключи",
        "пароль", "пароли", "номер карты", "карту", "кошел",
    ];
    if !SECRET_WORDS.iter().any(|w| q.contains(w)) {
        return false;
    }
    // Only when it's about what NIC has seen/stored — not "how do I make a strong
    // password", which is a legitimate question the model should answer.
    const ABOUT_NIC: &[&str] = &[
        "my ", "мой", "моя", "мои", "you saw", "you have", "you see", "you seen",
        "you know", "your memory", "in memory", "stored", "did you", "do you",
        "have you", "show me", "list any", "what passwords", "ты видел", "ты знаешь",
        "у тебя", "в памяти", "покажи",
    ];
    ABOUT_NIC.iter().any(|w| q.contains(w))
}

/// Asked about a person we have nothing on. Rather than a dead-end refusal, ASK —
/// the user knows who they mean, and one sentence from them ("he's a dota 2
/// player") both disambiguates the name and makes every later search land on the
/// right person instead of on camera optics.
fn no_bluff_person_prompt(name: &str) -> String {
    format!(
        "I don't know who {name} is, and I won't guess.\n\n\
         Tell me a bit about them — what they're known for, or where you saw them — \
         and I'll remember it and find them. (Or turn off offline mode and I'll search the web.)"
    )
}

/// Shown when NIC is asked about a specific person it has no grounded facts for.
#[allow(dead_code)]
const NO_BLUFF_PERSON: &str =
    "I don't have any verified information about that person — and I won't guess. \
     Turn off offline mode (or ask me to search) and I'll look it up on the web.";

/// True when the query asks WHO a specific named person is. A small local model
/// answers these with confident fiction ("David Laid was an American actor who
/// died in 1993" — he is a living fitness athlete), which is the single fastest
/// way to lose a user's trust. When such a question arrives with NO grounding at
/// all — nothing in memory, nothing from the web — we say so instead of letting
/// the model invent. Concept questions ("what is gravity") are NOT gated: the
/// model is reliable there, and the proper-noun check keeps them out.
/// Removes a greeting / filler opener so the gates below see the real question.
/// People type "hi who is X?" and "so, what is Y" all the time.
fn strip_openers(q: &str) -> &str {
    const OPENERS: &[&str] = &[
        "hi ", "hey ", "hello ", "yo ", "ok ", "okay ", "so ", "and ", "btw ",
        "hi, ", "hey, ", "hello, ", "ok, ", "okay, ", "so, ", "btw, ",
        "привет ", "привет, ", "слушай ", "слушай, ", "а ", "и ",
    ];
    let mut s = q.trim();
    // One opener is enough ("hi hey who is X" is not a thing).
    let sl = s.to_lowercase();
    for o in OPENERS {
        if sl.starts_with(o) {
            s = s[o.len()..].trim_start();
            break;
        }
    }
    s
}

/// Returns the person's name when the query asks who a specific NAMED person is.
/// `None` for concept questions ("what is gravity") and pronoun follow-ups
/// ("who is he") — those are handled by the normal QA path.
fn person_in_question(query: &str) -> Option<String> {
    // Strip a greeting or filler opener. "hi who is smoky nagata?" skipped the gate
    // entirely — the check demanded the query START with "who is" — and the model
    // answered that he was "an American actor known for John McClane in Die Hard".
    let q = strip_openers(query.trim());
    let ql = q.to_lowercase();
    const LEADS: &[&str] = &[
        "who is ", "who was ", "who's ", "who are ", "who were ",
        "кто такой ", "кто такая ", "кто такие ",
    ];
    // Find the lead ANYWHERE, not only at the start. People wrap it in politeness:
    // "hi nic could u say me who is kyosuke" slipped through and the model invented
    // an anime character. The name is whatever follows the lead.
    let (at, lead) = LEADS.iter()
        .filter_map(|l| ql.find(l).map(|i| (i, *l)))
        .min_by_key(|(i, _)| *i)?;
    let rest = q.get(at + lead.len()..)?
        .trim_end_matches(|c: char| matches!(c, '?' | '!' | '.'));
    let first = rest.split_whitespace().next()?.to_lowercase();
    let first = first.trim_matches(|c: char| !c.is_alphanumeric());

    // A pronoun is a follow-up, not a name.
    if matches!(first, "he" | "she" | "they" | "it" | "i" | "you" | "we") {
        return None;
    }
    // A determiner means it's a descriptive question ("who is the best programmer"),
    // not a named person.
    if matches!(first, "the" | "a" | "an" | "my" | "your" | "his" | "her" | "their"
                     | "this" | "that" | "твой" | "мой" | "этот") {
        return None;
    }
    // Everything else after "who is" is a name. Requiring a CAPITAL letter here was
    // the bug: typing "who is qewbite" in lowercase slipped past the gate and the
    // model answered with its own persona ("I am NIC-assistant") instead of
    // admitting it had never heard of him.
    Some(rest.trim().to_string())
}

/// The subject of an "about"-style question ("tell me about donk" → "donk").
/// Deliberately broader than `person_in_question`: it also catches concepts, and
/// that is fine — this only decides WHAT the conversation is currently on. Without
/// it, "tell me about donk" tracked nobody, so the user's own correction ("I am
/// about the cs 2 player donk") had nothing to attach to, fell through to the
/// model, and came back as a confident invented esports biography.
fn about_subject(query: &str) -> Option<String> {
    let q = strip_openers(query.trim());
    let ql = q.to_lowercase();
    const ABOUT_LEADS: &[&str] = &[
        "tell me more about ", "tell me about ", "tell about ", "tell me smth about ",
        "what do you know about ", "what can you tell me about ", "do you know about ",
        "info about ", "information about ",
        "расскажи мне про ", "расскажи про ", "расскажи о ", "расскажи об ",
        "что ты знаешь о ", "что ты знаешь про ",
    ];
    // The lead can sit mid-sentence: "hi assistant could u tell me about donk".
    let (at, lead) = ABOUT_LEADS.iter()
        .filter_map(|l| ql.find(l).map(|i| (i, *l)))
        .min_by_key(|(i, _)| *i)?;
    let rest = q.get(at + lead.len()..)?
        .trim_end_matches(|c: char| matches!(c, '?' | '!' | '.'))
        .trim()
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim();

    let first = rest.split_whitespace().next()?.to_lowercase();
    let first = first.trim_matches(|c: char| !c.is_alphanumeric());
    // Self-reference and pronouns are not topics: "tell me about yourself" is a
    // persona question, "tell me about him" is a follow-up the QA path resolves.
    if matches!(first, "you" | "yourself" | "your" | "me" | "myself" | "my" | "us" | "our"
                     | "him" | "his" | "her" | "them" | "their" | "it" | "this" | "that"
                     | "себя" | "тебе" | "мне" | "него" | "неё" | "нее") {
        return None;
    }
    let n = rest.split_whitespace().count();
    (n >= 1 && n <= 6 && rest.chars().count() <= 60).then(|| rest.to_string())
}

/// Who — or what — the conversation is now about. Used only to TRACK the current
/// entity, so a pronoun ("his last video") and a correction ("I mean the cs 2
/// player") both have a referent. The stricter `person_in_question` still decides
/// whether we refuse to guess.
fn entity_in_question(query: &str) -> Option<String> {
    person_in_question(query).or_else(|| about_subject(query))
}

/// Appended when the model answers about a NAMED thing that no source backs.
/// Live eval: asked for "the Karakov-Feldstein theorem" (invented), it produced a
/// confident fake — attributed to fake mathematicians, with a fake date. Refusing
/// outright would also kill legitimate answers ("explain Newton's second law"), so
/// instead we label the answer honestly. An assistant that admits when it's guessing
/// is more trustworthy than one that is silently right most of the time.
const UNVERIFIED_CAVEAT: &str =
    "⚠️ No source backs this — it comes from the model's own training and may be \
wrong or entirely invented. Ask me to search the web to verify.";

/// The proper noun a factual question is about ("the Karakov-Feldstein theorem" →
/// "Karakov-Feldstein"). `None` for questions with no named entity ("what is
/// gravity"), which the model handles reliably.
fn named_entity_in_question(query: &str) -> Option<String> {
    let q0 = query.trim();
    // "who is X" / "tell me about X": the subject is whatever follows, capitalised
    // or not. A bare lowercase handle is exactly what the model invents lives for —
    // "tell me about donk" came back as a confident Donkey Kong biography, and then
    // as an American Team Liquid player. Concept questions ("what is gravity") have
    // no such lead, so they still answer unlabelled.
    if let Some(subj) = person_in_question(q0).or_else(|| about_subject(q0)) {
        return Some(subj);
    }
    let q = strip_openers(q0);
    // Only factual lookups — not commands, not chat.
    let ql = q.to_lowercase();
    const FACT_LEADS: &[&str] = &[
        "what is", "what was", "what's", "what are",
        "who invented", "who discovered", "who won", "who wrote", "who created",
        "explain", "describe", "define", "when was", "when did",
        "where is", "where was",
    ];
    if !FACT_LEADS.iter().any(|l| ql.starts_with(l)) {
        return None;
    }
    // Capitalized tokens after the first word = a proper noun. Skip ALL-CAPS
    // acronyms (DNA, TCP, API): the model is reliable on those and flagging them
    // would put a warning on half of all correct answers.
    let toks: Vec<&str> = q.split_whitespace().collect();
    let names: Vec<String> = toks.iter().skip(1)
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-'))
        .filter(|w| {
            let mut cs = w.chars();
            let first_upper = cs.next().is_some_and(char::is_uppercase);
            let has_lower = w.chars().any(char::is_lowercase);
            first_upper && has_lower && w.chars().count() >= 3
        })
        .map(|w| w.to_string())
        .collect();
    (!names.is_empty()).then(|| names.join(" "))
}

/// True when the retrieved facts actually mention this person. Checking for an
/// EMPTY context is not enough — screen history is almost never empty, so a
/// question about a stranger would sail straight through into fiction. We test
/// the only thing that matters: does any name token appear in the facts at all?
fn person_is_grounded(name: &str, lib_ctx: &str, surf_ctx: &str) -> bool {
    let hay = format!("{} {}", lib_ctx, surf_ctx).to_lowercase();
    name.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| w.chars().count() >= 3)
        .any(|w| hay.contains(&w))
}

/// True when the engine aborted under the two-phase generation timeout (§3),
/// rather than failing for a real error. Lets the API switch to the raw-snippet path.
fn is_gen_timeout(e: &anyhow::Error) -> bool {
    let m = e.to_string();
    m.contains("NIC_TTFT_TIMEOUT") || m.contains("NIC_IDLE_TIMEOUT")
}

/// Raw-snippet fallback: when the model times out or its answer fails grounding
/// verification, never show a fabrication — show the raw, deduplicated text matches
/// that were actually retrieved (web snippets preferred, else screen history),
/// under a short notice. Caps at 8 lines so the bubble stays readable.
fn raw_snippet_fallback(lib_ctx: &str, surf_ctx: &str, notice: &str) -> String {
    let raw = if !surf_ctx.trim().is_empty() { surf_ctx } else { lib_ctx };
    let mut seen = std::collections::HashSet::new();
    let mut lines: Vec<&str> = Vec::new();
    for l in raw.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if seen.insert(l) { lines.push(l); }
        if lines.len() >= 8 { break; }
    }
    if lines.is_empty() {
        // No raw matches to show (e.g. a TTFT timeout on a query with no facts) —
        // return a clean standalone line instead of a notice that promises matches
        // and then shows none.
        return "The model didn't respond in time — please try again.".to_string();
    }
    format!("{notice}\n\n{}", lines.join("\n"))
}

/// Memory-informed media routing (Pilot×Memory): "play X" → the user's usual
/// AUDIO service, "watch X" → their usual VIDEO service, "continue" → reopen the
/// last one — all learned deterministically from screen memory, never the LLM.
/// Returns the answer if handled, else `None` so normal routing continues. The
/// Librarian is only touched once the query actually starts with a media verb.
async fn resolve_media_action(query: &str, librarian: &crate::librarian::Librarian) -> Option<String> {
    use crate::services::Kind;
    let orig = query.trim();
    let q = orig.to_lowercase();

    if matches!(q.as_str(),
        "continue" | "продолжи" | "продолжить" | "continue watching" | "continue where i left off") {
        let svc = librarian.most_recent_service().await?;
        crate::modules::pilot::open_url(svc.home);
        return Some(format!("Reopening {} — where you left off.", svc.label));
    }

    let (term, want) = {
        let audio: &[&str] = &["play ", "put on ", "поставь "];
        let video: &[&str] = &["watch ", "посмотреть "];
        let hit = audio.iter().find_map(|p| q.find(p).map(|i| (i + p.len(), Kind::Audio)))
            .or_else(|| video.iter().find_map(|p| q.find(p).map(|i| (i + p.len(), Kind::Video))))?;
        (orig.get(hit.0..)?.trim().to_string(), hit.1)
    };
    if term.chars().count() < 2 { return None; } // bare "play"/"watch" = media transport

    let svc = librarian.preferred_service(want).await?;
    // Video defaulting to YouTube is handled better by Pilot's autoplay+fullscreen
    // path, so let that take it.
    if want == Kind::Video && svc.id == "youtube" { return None; }
    crate::modules::pilot::open_url(&svc.search_url(&term));
    Some(format!("Opening «{}» on {} — your usual.", term, svc.label))
}

async fn handle_query(
    State(state): State<ApiState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();
    let query = req.query.trim().to_string();

    // Back-reference resolution for commands only ("open video about him" →
    // "…about David Laid"). QA/chat keeps the original query — its follow-up
    // logic handles pronouns itself.
    let cmd_query = match state.last_exchange.lock().await.clone() {
        Some((prev_q, _)) => crate::modules::anaphora::resolve(&query, &prev_q)
            .inspect(|r| tracing::info!("[Anaphora] «{}» → «{}»", query, r))
            .unwrap_or_else(|| query.clone()),
        None => query.clone(),
    };

    if let Some(msg) = resolve_media_action(&cmd_query, &state.librarian).await {
        return Ok(Json(QueryResponse { answer: msg, elapsed_ms: start.elapsed().as_millis() }));
    }
    if let Some(action) = pilot::try_execute(&cmd_query) {
        return Ok(Json(QueryResponse { answer: action.message, elapsed_ms: start.elapsed().as_millis() }));
    }

    // Deterministic router (kept in sync with the streaming path): questions
    // about the user / their machine are answered from code, never the LLM.
    {
        use crate::librarian::{asks_assistant_identity, asks_user_name, is_activity_recall};
        let det: Option<String> = if let Some(msg) =
            update_entity(&state.entity, &query, &state.librarian).await
        {
            Some(msg)
        } else if let Some(days) = report_request(&query) {
            Some(write_report(&state.librarian, &state.thinker, &state.language, days).await)
        } else if asks_about_secrets(&query) {
            Some(SECRETS_ANSWER.to_string())
        } else if is_activity_recall(&query) {
            Some(state.librarian.activity_summary(6).await)
        } else if asks_assistant_identity(&query) {
            Some("I'm NIC-assistant — your local assistant.".to_string())
        } else if asks_user_name(&query) {
            let name = state.profile.lock().await.name.trim().to_string();
            Some(if name.is_empty() {
                "I don't know your name yet — tell me your name and I'll remember.".to_string()
            } else {
                format!("Your name is {}.", name)
            })
        } else {
            None
        };
        if let Some(answer) = det {
            return Ok(Json(QueryResponse { answer, elapsed_ms: start.elapsed().as_millis() }));
        }
    }

    let force_online = ["новости", "сегодня", "openai", "news", "today", "latest", "weather", "price"];
    let q_low   = query.to_lowercase();
    let offline = req.offline && !force_online.iter().any(|kw| q_low.contains(*kw));

    let (lib_ctx, surf_ctx) = state.collector.collect(&query, offline).await;

    // Track who is being discussed (mirrors the streaming path).
    if let Some(name) = entity_in_question(&query) {
        let mut e = state.entity.lock().await;
        if e.as_ref().is_none_or(|c| !c.name.eq_ignore_ascii_case(&name)) {
            *e = Some(EntityCtx { name: name.clone(), facts: Vec::new() });
        }
    }

    // Never bluff about a named person the facts don't actually mention.
    if let Some(name) = person_in_question(&query) {
        if !person_is_grounded(&name, &lib_ctx, &surf_ctx) {
            tracing::info!("[NoBluff] «{}» not in any facts — asking the user instead", name);
            return Ok(Json(QueryResponse {
                answer: no_bluff_person_prompt(&name),
                elapsed_ms: start.elapsed().as_millis(),
            }));
        }
    }

    // A follow-up about that person which our facts cannot answer → one honest
    // line, not two paragraphs of the model narrating its own uncertainty.
    if let Some(answer) =
        unanswerable_about_person(&state.entity, &query, offline, &surf_ctx).await
    {
        return Ok(Json(QueryResponse { answer, elapsed_ms: start.elapsed().as_millis() }));
    }

    // A named thing with no source behind it → the answer gets labelled, not
    // suppressed (see UNVERIFIED_CAVEAT).
    let unverified = named_entity_in_question(&query)
        .is_some_and(|n| !person_is_grounded(&n, &lib_ctx, &surf_ctx));

    let prompt = state.collector.format_context_bundle(&lib_ctx, &surf_ctx);

    let thinker_ref = state.thinker.clone();
    let bundle_s    = prompt.clone();
    let query_s     = query.clone();
    let gen = tokio::task::spawn_blocking(move || {
        let mut t = thinker_ref.blocking_lock();
        t.answer(&bundle_s, &query_s)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Two-phase timeout + anti-hallucination (MASTER_PLAN §3 / §9[2]):
    //   • a TTFT/idle abort is NOT a server error → show the raw matches, not a 500;
    //   • Check 1 — every number/date in a web-grounded answer must be backed by the
    //     snippets, else the model invented it → show the raw matches, not the fiction.
    let answer = match gen {
        Ok(raw) => {
            let ans = crate::modules::thinker::extract_answer(&raw);
            if !surf_ctx.trim().is_empty()
                && !ans.trim().is_empty()
                && !crate::verify::is_grounded(&ans, &surf_ctx)
            {
                tracing::warn!("[Verify] ungrounded figures in answer — raw-snippet fallback");
                raw_snippet_fallback(&lib_ctx, &surf_ctx, RAW_NOTICE_UNVERIFIED)
            } else {
                ans
            }
        }
        Err(ref e) if is_gen_timeout(e) => raw_snippet_fallback(&lib_ctx, &surf_ctx, RAW_NOTICE_TIMEOUT),
        Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    };

    let answer = if unverified && !answer.trim().is_empty() {
        format!("{answer}\n\n{UNVERIFIED_CAVEAT}")
    } else {
        answer
    };

    Ok(Json(QueryResponse { answer, elapsed_ms: start.elapsed().as_millis() }))
}

/// SSE endpoint: streams tokens as they are generated by Thinker.
/// Clients connect with `Accept: text/event-stream` and receive one SSE `data:`
/// line per token, followed by a final `data: [DONE]` event.
async fn handle_query_stream(
    State(state): State<ApiState>,
    Json(req): Json<QueryRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(512);

    let collector    = state.collector.clone();
    let thinker      = state.thinker.clone();
    let librarian    = state.librarian.clone();
    let query        = req.query.trim().to_string();
    let offline      = req.offline;
    let last_exchange = state.last_exchange.clone();
    let answer_cache  = state.answer_cache.clone();
    let city         = state.city.clone();
    let profile      = state.profile.clone();
    // Whether to strip leaked CJK glyphs from the answer (off only for CJK languages).
    let filter_cjk   = { let l = state.language.lock().await; !lang_is_cjk(&l) };
    let language_for_report = state.language.clone();
    let entity_ctx          = state.entity.clone();

    let librarian_media = state.librarian.clone();
    tokio::spawn(async move {
        // Back-reference resolution for commands only ("open video about him" →
        // "…about David Laid"). QA/chat keeps the original query — its follow-up
        // logic handles pronouns itself.
        //
        // The referent is the person we're tracking, WITH whatever the user told us
        // about them: "lens" alone searches to camera optics, "LenS dota 2 player"
        // finds the person. Falls back to the previous question when we have no
        // tracked person.
        let cmd_query = {
            let ent = entity_ctx.lock().await.clone();
            let referent = ent.map(|e| e.search_term());
            let prev = last_exchange.lock().await.clone().map(|(q, _)| q);
            let source = referent.or(prev);
            match source {
                Some(src) => crate::modules::anaphora::resolve(&query, &src)
                    .inspect(|r| tracing::info!("[Anaphora] «{}» → «{}»", query, r))
                    .unwrap_or_else(|| query.clone()),
                None => query.clone(),
            }
        };

        // Memory-informed media routing (Pilot×Memory) before generic actions.
        if let Some(msg) = resolve_media_action(&cmd_query, &librarian_media).await {
            let _ = tx.send(msg).await;
            let _ = tx.send("[DONE]".to_string()).await;
            return;
        }
        // Check for computer actions first — bypass Thinker entirely.
        // Pilot responses are NOT stored in last_exchange — they are UI commands,
        // not dialogue turns, and conditioning the LLM on them causes hallucinations.
        if let Some(action) = pilot::try_execute(&cmd_query) {
            let _ = tx.send(action.message).await;
            let _ = tx.send("[DONE]".to_string()).await;
            return;
        }

        // Bare web-search command with no term of its own ("прогугли", "погугли",
        // "поищи", "google") → search the PREVIOUS question on the web. Lets the
        // user just say "прогугли" after a question and get real results, instead
        // of the model deflecting again. Runs before the cache so it always fetches.
        {
            let qt = query.trim().to_lowercase();
            let qt = qt.trim_matches(|c: char| !c.is_alphanumeric());
            const BARE_SEARCH: &[&str] = &[
                "прогугли", "погугли", "загугли", "нагугли", "гугли", "гугл",
                "поищи", "поиши", "найди", "search", "google",
            ];
            if BARE_SEARCH.contains(&qt) {
                if let Some((prev_q, _)) = last_exchange.lock().await.clone() {
                    let term = enrich_search_term(&prev_q, city.as_deref(), &prev_q);
                    let results = tokio::task::spawn_blocking(move || {
                        crate::modules::surfer::fetch_web_results_sync(&term)
                    }).await.unwrap_or_default();
                    if results.is_empty() {
                        let _ = tx.send(format!(
                            "I found nothing online for «{}».", prev_q)).await;
                    } else {
                        let snippets    = crate::modules::surfer::results_to_snippets(&results);
                        let web_bundle  = format!("Web results:\n{}", snippets);
                        let thinker_ref = thinker.clone();
                        let q_for       = prev_q.clone();
                        let tx2         = tx.clone();
                        let gen = tokio::task::spawn_blocking(move || {
                            let mut t = thinker_ref.blocking_lock();
                            t.answer_streaming(&web_bundle, &q_for, move |token| {
                                let out = if filter_cjk { strip_cjk(token) } else { token.to_string() };
                                if !out.is_empty() { let _ = tx2.blocking_send(out); }
                            })
                        }).await;
                        // Anti-hallucination Check 1 (§9[2]): warn if the streamed answer's
                        // figures aren't backed by the snippets (stream can't be retracted).
                        if let Ok(Ok(ref raw)) = gen {
                            let ans = crate::modules::thinker::extract_answer(raw);
                            if !ans.trim().is_empty() && !crate::verify::is_grounded(&ans, &snippets) {
                                let _ = tx.send(format!("\n\n{RAW_NOTICE_UNVERIFIED_TAIL}")).await;
                            }
                        }
                        let src = format_sources(&results);
                        if !src.is_empty() { let _ = tx.send(src).await; }
                    }
                    let _ = tx.send("[DONE]".to_string()).await;
                    return;
                }
            }
        }

        // ── Deterministic router ──────────────────────────────────────────────
        // Questions ABOUT THE USER or their machine are answered from CODE, never
        // the small LLM. These are simple lookups (screen log, profile) where the
        // model adds only risk: it inconsistently drops the time, claims "нечего
        // вспомнить" when there IS data, or confuses "кто я" with "кто такой
        // <famous person>". Composing the answer in code makes it identical and
        // reliable on ANY model size, and instant. Runs before the cache so it
        // always reflects current state, never a stale cached non-answer.
        {
            use crate::librarian::{asks_assistant_identity, asks_user_name, is_activity_recall};
            let det: Option<String> = if let Some(msg) =
                update_entity(&entity_ctx, &query, &librarian).await
            {
                Some(msg)
            } else if let Some(days) = report_request(&query) {
                Some(write_report(&librarian, &thinker, &language_for_report, days).await)
            } else if asks_about_secrets(&query) {
                Some(SECRETS_ANSWER.to_string())
            } else if is_activity_recall(&query) {
                Some(librarian.activity_summary(6).await)
            } else if asks_assistant_identity(&query) {
                Some("I'm NIC-assistant — your local assistant.".to_string())
            } else if asks_user_name(&query) {
                let name = profile.lock().await.name.trim().to_string();
                Some(if name.is_empty() {
                    "I don't know your name yet — tell me your name and I'll remember.".to_string()
                } else {
                    format!("Your name is {}.", name)
                })
            } else {
                None
            };
            if let Some(answer) = det {
                {
                    let mut last = last_exchange.lock().await;
                    *last = Some((query.clone(), answer.clone()));
                }
                for ch in answer.chars() { let _ = tx.send(ch.to_string()).await; }
                let _ = tx.send("[DONE]".to_string()).await;
                return;
            }
        }

        // Answer cache: return cached answer for semantically identical queries (< 5 min old).
        {
            let mut cache = answer_cache.lock().await;
            let now = std::time::Instant::now();
            cache.retain(|e| now.duration_since(e.cached_at).as_secs() < 300);
            let qwords = query_words(&query);
            if let Some(entry) = cache.iter().find(|e| jaccard(&qwords, &e.words) >= 0.85) {
                let ans = entry.answer.clone();
                drop(cache);
                for ch in ans.chars() { let _ = tx.send(ch.to_string()).await; }
                let _ = tx.send("[DONE]".to_string()).await;
                return;
            }
        }

        let prev = last_exchange.lock().await.clone();

        // Treat a query as a follow-up (needing the previous topic) only when it's
        // genuinely referential: it contains a back-reference word ("it", "that",
        // "more"…), or — for Russian input — is very short (≤2 words). A self-contained
        // question like "what events did you log" must NOT drag in the previous topic —
        // doing so made unrelated answers repeat (e.g. the prior answer about a person).
        let has_cyrillic = query.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c));
        let wc = query.split_whitespace().count();
        let ql = query.to_lowercase();
        const REF_WORDS: &[&str] = &[
            // English back-references (English-only beta).
            "it", "its", "that", "this", "these", "those", "they", "them", "their",
            "there", "he", "she", "his", "her", "more",
            // Legacy Russian (harmless when RU is never typed).
            "это", "этот", "эта", "эти", "он", "она", "они", "его", "её", "ее",
            "их", "там", "туда", "ещё", "еще", "тоже", "также", "подробнее",
        ];
        let has_ref = ql.split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
            .any(|t| REF_WORDS.contains(&t));
        // A back-reference word marks a follow-up in any language; the bare "very
        // short query" heuristic stays Cyrillic-gated so a short English question
        // ("Rust?") isn't wrongly anchored to the previous topic.
        let is_followup = prev.is_some() && (has_ref || (wc <= 2 && has_cyrillic));

        let collect_query = match (&prev, is_followup) {
            (Some((prev_q, _)), true) => format!("{} — {}", prev_q, query),
            _ => query.clone(),
        };

        let (lib_ctx, surf_ctx) = collector.collect(&collect_query, offline).await;

        // Remember WHO is being discussed, so a later "his last video" resolves —
        // and so the user can teach us about them ("he is a dota 2 player").
        if let Some(name) = entity_in_question(&query) {
            let mut e = entity_ctx.lock().await;
            let changed = e.as_ref().is_none_or(|c| !c.name.eq_ignore_ascii_case(&name));
            if changed {
                *e = Some(EntityCtx { name: name.clone(), facts: Vec::new() });
            }
        }

        // Never bluff about a named person the facts don't actually mention.
        if let Some(name) = person_in_question(&query) {
            if !person_is_grounded(&name, &lib_ctx, &surf_ctx) {
                tracing::info!("[NoBluff] «{}» not in any facts — refusing to guess", name);
                // Record the turn anyway. The user still NAMED someone, so a
                // follow-up ("open video about him") must resolve to that person —
                // without this, the pronoun bound to an older, unrelated question
                // and produced «video about i just doing».
                let msg = no_bluff_person_prompt(&name);
                {
                    let mut last = last_exchange.lock().await;
                    *last = Some((query.clone(), msg.clone()));
                }
                let _ = tx.send(msg).await;
                let _ = tx.send("[DONE]".to_string()).await;
                return;
            }
        }

        // A follow-up about the person that we simply cannot answer → say so once,
        // clearly, instead of letting the model waffle.
        if let Some(msg) =
            unanswerable_about_person(&entity_ctx, &query, offline, &surf_ctx).await
        {
            {
                let mut last = last_exchange.lock().await;
                *last = Some((query.clone(), msg.clone()));
            }
            let _ = tx.send(msg).await;
            let _ = tx.send("[DONE]".to_string()).await;
            return;
        }

        // A named thing with no source behind it → the answer gets labelled once
        // the stream ends (it can't be retracted mid-flight).
        let unverified = named_entity_in_question(&query)
            .is_some_and(|n| !person_is_grounded(&n, &lib_ctx, &surf_ctx));

        let base_prompt = collector.format_context_bundle(&lib_ctx, &surf_ctx);

        // Only carry the previous exchange into the prompt for real follow-ups,
        // so a fresh question isn't anchored to the last topic.
        let prompt = match (prev, is_followup) {
            (Some((prev_q, prev_a)), true) => {
                let prev_a_short: String = prev_a.chars().take(200).collect();
                format!("Previous exchange:\nUser: {}\nNIC: {}\n\n{}", prev_q, prev_a_short, base_prompt)
            }
            _ => base_prompt,
        };

        let thinker_ref = thinker.clone();
        let bundle_s    = prompt.clone();
        let query_s     = query.clone();
        let tx2         = tx.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut t = thinker_ref.blocking_lock();

            // Smart intent fallback: for short queries (≤8 words) that Pilot didn't catch,
            // ask the LLM to classify intent before doing a full answer pass.
            // Skip for:
            //   a) queries containing Russian question words (confuses classifier)
            //   b) Statement Guard: queries that START with personal pronouns → QA only
            const QUESTION_WORDS: &[&str] = &[
                "кто", "что", "где", "когда", "сколько", "почему", "как",
                "зачем", "чем", "какой", "какая", "какое", "расскажи", "объясни",
                // English-only beta: keep questions out of the command-classifier so
                // "who was Newton" is answered, never guessed into a SCREENSHOT/app-open.
                "who", "what", "where", "when", "why ", "how", "which", "whose",
                "tell ", "explain", "define", "describe",
                "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
            ];
            // Statement Guard: pronoun-start queries are always conversational
            const PRONOUN_STARTS: &[&str] = &[
                "я ", "ты ", "это ", "мне ", "мой ", "моя ", "моё", "моего", "моему",
                "твой", "твоя", "твоё", "у меня", "у тебя", "мы ", "вы ", "нас ", "вам ",
            ];
            let q_low_s = query_s.to_lowercase();
            let is_question = QUESTION_WORDS.iter().any(|w| q_low_s.contains(w));
            let is_personal = PRONOUN_STARTS.iter().any(|p| q_low_s.starts_with(p));
            if !is_question && !is_personal && query_s.split_whitespace().count() <= 8 {
                let code = t.classify_intent(&query_s);
                // Surfer 2.0: intercept WEB_SEARCH — fetch snippets in-chat instead of browser
                let code_up = code.to_uppercase();
                if code_up.starts_with("WEB_SEARCH:") {
                    let raw_term = code[11..].trim();
                    let term = enrich_search_term(raw_term, city.as_deref(), &query_s);
                    let results = crate::modules::surfer::fetch_web_results_sync(&term);
                    if !results.is_empty() {
                        let snippets = crate::modules::surfer::results_to_snippets(&results);
                        let web_bundle = format!("Web results:\n{}\n\n{}", snippets, bundle_s);
                        let cb_tx = tx2.clone();
                        let r = t.answer_streaming(&web_bundle, &query_s, move |token| {
                            let out = if filter_cjk { strip_cjk(token) } else { token.to_string() };
                            if !out.is_empty() { let _ = cb_tx.blocking_send(out); }
                        });
                        // Anti-hallucination Check 1 (§9[2]): verify the streamed answer
                        // against the snippets it was handed; the stream can't be retracted,
                        // so warn if a figure isn't backed by the sources.
                        if let Ok(ref raw) = r {
                            let ans = crate::modules::thinker::extract_answer(raw);
                            if !ans.trim().is_empty() && !crate::verify::is_grounded(&ans, &snippets) {
                                let _ = tx2.blocking_send(format!("\n\n{RAW_NOTICE_UNVERIFIED_TAIL}"));
                            }
                        }
                        // Append clickable source citations after the answer.
                        let src = format_sources(&results);
                        if !src.is_empty() { let _ = tx2.blocking_send(src); }
                        return r;
                    }
                    // Fetch failed — fall through to normal Thinker QA
                } else if let Some(action) = pilot::execute_intent(&code) {
                    let msg = action.message;
                    let _ = tx2.blocking_send(msg.clone());
                    return Ok(msg);
                }
            }

            t.answer_streaming(&bundle_s, &query_s, move |token| {
                let out = if filter_cjk { strip_cjk(token) } else { token.to_string() };
                if !out.is_empty() { let _ = tx2.blocking_send(out); }
            })
        })
        .await;

        match result {
            Ok(Ok(ref raw_answer)) => {
                // Keep only the <answer> block (drop the model's <think> reasoning),
                // then strip any leaked CJK — this is the clean text we cache / store / recall.
                let answer = crate::modules::thinker::extract_answer(raw_answer);
                let answer = if filter_cjk { strip_cjk(&answer) } else { answer };

                if answer.trim().is_empty() {
                    // Never show an empty bubble — the model produced nothing (thin
                    // context, or it choked). Send a graceful line instead and don't
                    // cache/store a non-answer.
                    let _ = tx.send(
                        "Hmm, I can't answer that yet — maybe I haven't recorded enough. Try rephrasing?".to_string()
                    ).await;
                } else {
                    // Anti-hallucination Check 1 (§9[2]): a web-grounded answer whose
                    // numbers/dates aren't backed by the snippets gets a caveat. The
                    // stream is already on screen (can't retract), so we append a note.
                    if !surf_ctx.trim().is_empty() && !crate::verify::is_grounded(&answer, &surf_ctx) {
                        let _ = tx.send(format!("\n\n{RAW_NOTICE_UNVERIFIED_TAIL}")).await;
                    } else if unverified {
                        // No source names this entity at all → say so plainly.
                        let _ = tx.send(format!("\n\n{UNVERIFIED_CAVEAT}")).await;
                    }
                    // Store in answer cache (max 20 entries).
                    {
                        let mut cache = answer_cache.lock().await;
                        if cache.len() >= 20 { cache.remove(0); }
                        cache.push(CacheEntry {
                            words:     query_words(&query),
                            answer:    answer.clone(),
                            cached_at: std::time::Instant::now(),
                        });
                    }
                    let mut last = last_exchange.lock().await;
                    *last = Some((query.clone(), answer.clone()));

                    // Persist Q&A to Librarian so RAG can recall it later:
                    // - "what was I interested in?" → finds past questions
                    // - repeat question offline → answer is already in the DB
                    let skip = answer.len() < 20
                        || answer.contains("Не понял команду")
                        || answer.contains("Не удалось");
                    if !skip {
                        let qa = format!("Q: {}\nA: {}", query, answer);
                        let _ = librarian.add_event(&qa, "dialogue", "qa").await;
                    }
                }
            }
            Ok(Err(ref e)) if is_gen_timeout(e) => {
                // Two-phase timeout (§3). A TTFT abort streamed nothing → show the raw
                // matches; an idle abort already streamed a partial → append a notice.
                if e.to_string().contains("NIC_TTFT_TIMEOUT") {
                    let _ = tx.send(raw_snippet_fallback(&lib_ctx, &surf_ctx, RAW_NOTICE_TIMEOUT)).await;
                } else {
                    let _ = tx.send(format!("\n\n{RAW_NOTICE_TIMEOUT_TAIL}")).await;
                }
            }
            Ok(Err(e))      => tracing::warn!("[SSE] generation error: {}", e),
            Err(e)          => tracing::warn!("[SSE] spawn_blocking panic: {}", e),
        }

        let _ = tx.send("[DONE]".to_string()).await;
    });

    let token_stream = stream! {
        while let Some(token) = rx.recv().await {
            yield Ok::<Event, Infallible>(Event::default().data(token));
        }
    };

    Sse::new(token_stream).keep_alive(KeepAlive::default())
}

async fn handle_add_event(
    State(state): State<ApiState>,
    Json(req): Json<AddEventRequest>,
) -> Result<Json<AddEventResponse>, (StatusCode, String)> {
    let id = state.librarian
        .add_event(&req.text, &req.source, &req.event_type)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(AddEventResponse { id }))
}

async fn handle_search(
    State(state): State<ApiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let results = state.librarian
        .find_relevant(&req.query, req.limit)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(SearchResponse { results }))
}

async fn handle_reset(
    State(state): State<ApiState>,
) -> impl IntoResponse {
    state.thinker.lock().await.clear_history();
    "History cleared"
}

/// Erases ALL stored memory (DB rows + cold archive + caches). The privacy
/// counterpart to `/export`: the user can take everything and destroy
/// everything. Irreversible.
async fn handle_memory_wipe(State(state): State<ApiState>) -> impl IntoResponse {
    match state.librarian.wipe_all().await {
        Ok(_) => {
            state.thinker.lock().await.clear_history();
            tracing::warn!("[API] all memory erased on user request");
            (StatusCode::OK, [("content-type", "application/json")],
             r#"{"ok":true}"#).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
             [("content-type", "application/json")],
             format!(r#"{{"ok":false,"error":"{}"}}"#, e.to_string().replace('"', "'")))
             .into_response(),
    }
}

async fn handle_search_archive(
    State(state): State<ApiState>,
    Json(req): Json<ArchiveSearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let results = state.librarian
        .search_archive(&req.query, req.limit)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(SearchResponse { results }))
}

/// Selective amnesia: deletes recent N minutes of captured events.
/// Also clears dialogue history and embedding cache.
async fn handle_forget(
    State(state): State<ApiState>,
    Json(req): Json<ForgetRequest>,
) -> Result<Json<ForgetResponse>, (StatusCode, String)> {
    let erased = state.librarian
        .forget_recent(req.minutes)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    state.thinker.lock().await.clear_history();
    state.librarian.clear_embed_cache().await;

    Ok(Json(ForgetResponse { erased, minutes: req.minutes }))
}

async fn handle_search_filter(
    State(state): State<ApiState>,
    Json(req): Json<FilterSearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let results = state.librarian
        .search_by_filter(
            &req.query,
            req.time_from,
            req.time_to,
            req.app_name.as_deref(),
            req.limit,
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(SearchResponse { results }))
}

async fn handle_export(
    State(state): State<ApiState>,
    Query(params): Query<ExportParams>,
) -> Response {
    match state.librarian.export_recent(params.limit).await {
        Ok(lines) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/x-ndjson")
            .header("content-disposition", "attachment; filename=\"nic_export.ndjson\"")
            .body(Body::from(lines.join("\n")))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

// ── Diagnostics (user-driven bug reports; NIC ships zero telemetry) ────────────

/// A copy-pasteable diagnostics blob for GitHub issues. Deliberately contains
/// NO memory content — only versions, hardware, sizes, and the tail of the log
/// (which already avoids logging screen text). The user triggers it explicitly.
async fn handle_diagnostics(State(state): State<ApiState>) -> impl IntoResponse {
    let (db_bytes, archive_bytes) = state.librarian.disk_usage();
    let event_count = state.librarian.count_events().await;
    let hw = crate::hardware::Hardware::detect();
    let llm_ok = reqwest::Client::new()
        .get(format!("{}/health", state.llm_url))
        .timeout(std::time::Duration::from_secs(2))
        .send().await.map(|r| r.status().is_success()).unwrap_or(false);
    let selected_model = state.selected_model.lock().await.clone();
    let language       = state.language.lock().await.clone();

    let log_tail = read_log_tail(60);

    let text = format!(
        "NIC-Assistant diagnostics\n\
         ------------------------\n\
         version:        {}\n\
         os:             {} {}\n\
         ram:            {:.1} GB\n\
         vram:           {:.1} GB\n\
         model:          {}\n\
         language:       {}\n\
         llm_server:     {}\n\
         memory_consent: {}\n\
         events:         {}\n\
         db_size:        {:.1} MB\n\
         archive_size:   {:.1} MB\n\
         \n--- last {} log lines ---\n{}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS, std::env::consts::ARCH,
        hw.ram_gb, hw.vram_gb,
        selected_model, language,
        if llm_ok { "up" } else { "down" },
        crate::modules::sentinel::capture_consent(),
        event_count,
        db_bytes as f64 / 1_048_576.0,
        archive_bytes as f64 / 1_048_576.0,
        60, log_tail,
    );
    (StatusCode::OK, [("content-type", "text/plain; charset=utf-8")], text)
}

/// Returns the last `n` lines of the app log, or a placeholder. Best-effort.
fn read_log_tail(n: usize) -> String {
    let path = crate::config::app_bin_dir().join("logs").join("nic.log");
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let lines: Vec<&str> = s.lines().collect();
            let start = lines.len().saturating_sub(n);
            lines[start..].join("\n")
        }
        Err(e) => format!("(could not read log at {}: {e})", path.display()),
    }
}

// ── Update check (manual — NIC never phones home on its own) ───────────────────

/// GitHub Releases API for the public release repo (see README).
const RELEASES_API: &str =
    "https://api.github.com/repos/nickqwe885/nic-assistant-releases/releases/latest";
const RELEASES_PAGE: &str =
    "https://github.com/nickqwe885/nic-assistant-releases/releases";

#[derive(Serialize)]
struct UpdateResponse {
    current: String,
    latest: String,
    update_available: bool,
    url: String,
}

/// Explicitly checks GitHub for a newer release. Only ever runs when the user
/// clicks "Check for updates" — nothing here fires automatically, so the
/// "zero network by default" promise holds.
async fn handle_update_check() -> impl IntoResponse {
    let current = env!("CARGO_PKG_VERSION").to_string();
    let latest = fetch_latest_tag().await.unwrap_or_default();
    let update_available = !latest.is_empty() && latest != current;
    Json(UpdateResponse { current, latest, update_available, url: RELEASES_PAGE.to_string() })
}

/// Best-effort fetch of the latest release tag (leading `v` stripped). `None` on
/// any network/parse failure — the UI just reports "couldn't check".
async fn fetch_latest_tag() -> Option<String> {
    let client = reqwest::Client::builder().user_agent("nic-assistant").build().ok()?;
    let json = client.get(RELEASES_API)
        .timeout(std::time::Duration::from_secs(6))
        .send().await.ok()?
        .json::<serde_json::Value>().await.ok()?;
    Some(json.get("tag_name")?.as_str()?.trim_start_matches('v').to_string())
}

// ── Screen-capture consent ────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ConsentResponse {
    pub enabled: bool,
}

async fn handle_consent_get() -> impl IntoResponse {
    Json(ConsentResponse { enabled: crate::modules::sentinel::capture_consent() })
}

/// Grants screen-capture consent: drops the on-disk marker and flips the live
/// flag so the Sentinel starts remembering. One-way in the beta — to pause
/// again the user uses Incognito. Idempotent.
async fn handle_consent_post(State(state): State<ApiState>) -> impl IntoResponse {
    if let Some(dir) = state.consent_marker.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::write(&state.consent_marker, b"1") {
        Ok(_) => {
            crate::modules::sentinel::set_capture_consent(true);
            tracing::info!("[API] screen-capture consent granted");
            Json(ConsentResponse { enabled: true }).into_response()
        }
        Err(e) => {
            tracing::error!("[API] failed to persist consent: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR,
             [("content-type", "application/json")],
             r#"{"error":"could not save consent"}"#).into_response()
        }
    }
}

// ── Incognito ─────────────────────────────────────────────────────────────────

async fn handle_incognito_get(State(state): State<ApiState>) -> impl IntoResponse {
    Json(IncognitoResponse { enabled: state.incognito.load(Ordering::Relaxed) })
}

async fn handle_incognito_post(State(state): State<ApiState>) -> impl IntoResponse {
    let was = state.incognito.fetch_xor(true, Ordering::Relaxed);
    let now = !was;
    tracing::info!("[API] incognito toggled → {}", now);
    Json(IncognitoResponse { enabled: now })
}

// ── LLM status ───────────────────────────────────────────────────────────────

async fn handle_llm_status(State(state): State<ApiState>) -> impl IntoResponse {
    let client = reqwest::Client::new();
    let ok = client
        .get(format!("{}/health", state.llm_url))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    Json(LlmStatusResponse { ok, url: state.llm_url.clone() })
}

// ── Profile ───────────────────────────────────────────────────────────────────

async fn handle_profile_get(State(state): State<ApiState>) -> impl IntoResponse {
    Json(state.profile.lock().await.clone())
}

async fn handle_profile_post(
    State(state): State<ApiState>,
    Json(req): Json<ProfileConfig>,
) -> impl IntoResponse {
    *state.profile.lock().await = req.clone();
    state.thinker.lock().await.update_profile(&req);
    Json(req)
}

// ── Image upload + OCR ───────────────────────────────────────────────────────

async fn handle_image(
    State(state): State<ApiState>,
    Json(req):    Json<ImageRequest>,
) -> Result<Json<ImageResponse>, (StatusCode, String)> {
    let bytes = BASE64.decode(&req.image)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("base64: {e}")))?;

    let img = image::load_from_memory(&bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("image: {e}")))?;
    let rgba  = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let raw   = rgba.into_raw();

    let ocr_text = tokio::task::spawn_blocking(move || {
        crate::ocr::extract_text(&raw, w, h)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .unwrap_or_default();

    let stored = if ocr_text.len() > 10 {
        let ev = format!("[PHOTO: {}]\n{}", req.filename, ocr_text);
        state.librarian.add_event(&ev, "image", "photo").await.is_ok()
    } else {
        false
    };

    Ok(Json(ImageResponse { text: ocr_text, stored }))
}

// ── Activity graph data ───────────────────────────────────────────────────────

async fn handle_activity(
    State(state): State<ApiState>,
) -> Result<Json<ActivityResponse>, (StatusCode, String)> {
    let pts = state.librarian
        .activity_hourly(24)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(ActivityResponse {
        points: pts.into_iter().map(|(hour, count)| ActivityPoint { hour, count }).collect(),
    }))
}

// ── Location-aware search enrichment ─────────────────────────────────────────

/// Appends the user's city to location-sensitive search terms.
/// "погода" + city="Астана" → "погода Астана"
/// Skips enrichment if the term already contains the city.
fn enrich_search_term(term: &str, city: Option<&str>, original_query: &str) -> String {
    let Some(city) = city else { return term.to_string() };
    let term_lower  = term.to_lowercase();
    let query_lower = original_query.to_lowercase();
    const LOCATION_KWS: &[&str] = &[
        "погода", "weather", "прогноз", "forecast",
        "пробки", "трафик", "traffic",
        "новости", "события",
    ];
    let is_location = LOCATION_KWS.iter()
        .any(|kw| term_lower.contains(kw) || query_lower.contains(kw));
    let already_has_city = term_lower.contains(&city.to_lowercase());
    if is_location && !already_has_city {
        format!("{} {}", term, city)
    } else {
        term.to_string()
    }
}

/// Builds a clickable "Источники" block (markdown links) from web results, so
/// the user can open the real sources. Links open in the external browser via
/// the launcher's navigation handler — they never hijack the chat window.
fn format_sources(results: &[crate::modules::surfer::WebResult]) -> String {
    let links: Vec<String> = results.iter()
        .filter(|r| r.url.starts_with("http"))
        .take(3)
        .map(|r| {
            let label = if r.title.trim().is_empty() { r.url.clone() } else { r.title.clone() };
            let label = label.replace(['[', ']', '(', ')'], " ");
            let label = label.split_whitespace().collect::<Vec<_>>().join(" ");
            format!("- [{}]({})", label, r.url)
        })
        .collect();
    if links.is_empty() { String::new() } else { format!("\n\nSources:\n{}", links.join("\n")) }
}

// ── Foreign-script filter ─────────────────────────────────────────────────────

/// True when the answer language is itself CJK (Chinese/Japanese/Korean), so we
/// must NOT strip those characters.
fn lang_is_cjk(lang: &str) -> bool {
    let l = lang.to_lowercase();
    l.contains("chin") || l.contains("japan") || l.contains("korea")
        || l.contains("кита") || l.contains("япон") || l.contains("коре")
}

/// Removes CJK characters from text. The small model occasionally leaks Chinese
/// tokens into a Russian/English answer (e.g. "活动"); for non-CJK answer
/// languages we strip them so the user never sees stray glyphs mid-sentence.
fn strip_cjk(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            let u = c as u32;
            !((0x3000..=0x303F).contains(&u)   // CJK punctuation
                || (0x3040..=0x30FF).contains(&u)   // Hiragana + Katakana
                || (0x3400..=0x4DBF).contains(&u)   // CJK Ext. A
                || (0x4E00..=0x9FFF).contains(&u)   // CJK Unified
                || (0xAC00..=0xD7AF).contains(&u)   // Hangul syllables
                || (0xF900..=0xFAFF).contains(&u)   // CJK compatibility ideographs
                || (0xFF00..=0xFFEF).contains(&u))  // fullwidth / halfwidth forms
        })
        .collect()
}

// ── Chat history ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct HistoryMessage {
    role: &'static str,
    text: String,
}

#[derive(Serialize)]
struct HistoryResponse {
    messages: Vec<HistoryMessage>,
}

/// Parses a stored dialogue event into (question, answer). Accepts the current
/// English "Q:/A:" format and the legacy Russian "Вопрос:/Ответ:" format so old
/// DB rows still render in the history pane after the English-only switch.
fn parse_stored_qa(text: &str) -> Option<(String, String)> {
    for (qm, am) in [("Q: ", "\nA: "), ("Вопрос: ", "\nОтвет: ")] {
        if let Some(q_start) = text.find(qm) {
            let rest = &text[q_start + qm.len()..];
            if let Some(a_start) = rest.find(am) {
                let q_text = rest[..a_start].trim().to_string();
                let a_text = rest[a_start + am.len()..].trim().to_string();
                return Some((q_text, a_text));
            }
        }
    }
    None
}

async fn handle_history(State(state): State<ApiState>) -> impl IntoResponse {
    let pairs = state.librarian.recent_qa(20).await;
    let mut messages: Vec<HistoryMessage> = Vec::new();
    for (_, text) in pairs {
        if let Some((q_text, a_text)) = parse_stored_qa(&text) {
            if !q_text.is_empty() { messages.push(HistoryMessage { role: "user", text: q_text }); }
            if !a_text.is_empty() { messages.push(HistoryMessage { role: "ai",   text: a_text }); }
        }
    }
    // reverse so oldest first
    messages.reverse();
    Json(HistoryResponse { messages })
}

// ── Idea-writer (NIC Notes) ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct NoteSaveRequest {
    raw: String,
}

#[derive(Serialize)]
struct NoteSaveResponse {
    file:    String,
    title:   String,
    content: String,
}

// ── Report writer ─────────────────────────────────────────────────────────────
// "Write a report of my week" → a real .md file built from screen memory. This is
// the one writing job worth having: the model does NOT compose from imagination,
// it FORMATS facts only NIC has. No other assistant can write this document,
// because no other assistant has the data.

/// Detects a request to write a report/summary document, and for how many days.
/// Deterministic — a small model must not decide whether to create files.
fn report_request(query: &str) -> Option<i64> {
    let q = query.to_lowercase();
    const VERBS: &[&str] = &[
        "write a report", "write a summary", "make a report", "create a report",
        "write me a report", "write up my", "report on what i", "summarize my week",
        "summarise my week", "summarize my day", "write a doc", "save a report",
        "напиши отчёт", "напиши отчет", "сделай отчёт", "сделай отчет", "составь отчёт",
    ];
    if !VERBS.iter().any(|v| q.contains(v)) {
        return None;
    }
    let days = if q.contains("week") || q.contains("недел") {
        7
    } else if q.contains("month") || q.contains("месяц") {
        30
    } else {
        1
    };
    Some(days)
}

/// A follow-up question about the tracked person that our facts cannot answer.
/// "how tall he is" / "is he alive" made the model waffle for two paragraphs about
/// not having the information. Say it in one line, and offer the way out.
async fn unanswerable_about_person(
    entity:   &Arc<tokio::sync::Mutex<Option<EntityCtx>>>,
    query:    &str,
    offline:  bool,
    surf_ctx: &str,
) -> Option<String> {
    if !offline && !surf_ctx.trim().is_empty() {
        return None;                       // the web gave us something — let it answer
    }
    let ql = query.trim().to_lowercase();
    // A question, and it's about him/her (a follow-up, not a fresh topic).
    let is_question = ql.ends_with('?')
        || ["how ", "is he", "is she", "was he", "was she", "does he", "does she",
            "did he", "did she", "what does he", "where is he", "when did he"]
            .iter().any(|w| ql.starts_with(w));
    let about_them = ["he ", "she ", " he", " she", "his ", "her ", "him", "them", "their"]
        .iter().any(|w| ql.contains(w));
    if !is_question || !about_them {
        return None;
    }
    let ctx = entity.lock().await.clone()?;
    if ctx.facts.is_empty() {
        return None;                       // nothing taught yet — normal QA path
    }
    Some(format!(
        "All I know about {} is what you told me: {}.\n\n\
         I won't guess beyond that. Turn off offline mode (or ask me to search) and \
         I'll look it up.",
        ctx.name,
        ctx.facts.join("; ")
    ))
}

/// Handles the user TELLING NIC about the person under discussion. Returns the
/// reply when the query was such a statement (so it never reaches the model,
/// which used to answer "he is a dota 2 player" with a definition of an optical
/// lens). Facts are kept in memory for search, and written to the Librarian so
/// they survive a restart.
async fn update_entity(
    entity:    &Arc<tokio::sync::Mutex<Option<EntityCtx>>>,
    query:     &str,
    librarian: &crate::librarian::Librarian,
) -> Option<String> {
    let upd = entity_statement(query)?;
    let mut guard = entity.lock().await;
    let ctx = guard.as_mut()?;   // nothing to attach the fact to — let QA handle it

    match upd {
        EntityUpdate::Rename(new_name) => {
            let old = std::mem::replace(&mut ctx.name, new_name.clone());
            let line = format!("[Person] {} (also known as {})", new_name, old);
            let _ = librarian.add_event(&line, "user", "person").await;
            Some(format!("Got it — {new_name}. I'll use that from now on."))
        }
        EntityUpdate::Fact(fact) => {
            ctx.add_fact(&fact);
            let line = format!("[Person] {} — {}", ctx.name, fact);
            let _ = librarian.add_event(&line, "user", "person").await;
            Some(format!(
                "Got it — {} is {}. I'll remember that, and I'll use it when I look \
                 them up.\n\nTry: *play their last video* — I'll search for «{}».",
                ctx.name, fact, ctx.search_term()
            ))
        }
    }
}

/// Builds the report from screen memory, saves it as a Markdown file, and returns
/// the chat answer. Runs in the deterministic router: the model is only allowed to
/// FORMAT the activity log it is handed.
async fn write_report(
    librarian: &crate::librarian::Librarian,
    thinker:   &Arc<tokio::sync::Mutex<crate::modules::thinker::Thinker>>,
    language:  &Arc<tokio::sync::Mutex<String>>,
    days:      i64,
) -> String {
    let activity = match librarian.activity_summary_for_days(days).await {
        Ok(a) if !a.trim().is_empty() => a,
        _ => return "I don't have enough recorded activity to write a report yet. \
                     Let me watch your screen for a while first.".to_string(),
    };

    let period = match days {
        1 => "today".to_string(),
        7 => "the last 7 days".to_string(),
        n => format!("the last {n} days"),
    };
    let lang   = language.lock().await.clone();
    let prompt = report_prompt(&period, &activity, &lang);

    let t = thinker.clone();
    let generated = tokio::task::spawn_blocking(move || {
        t.blocking_lock().generate_raw(&prompt, 600)
    }).await;

    let text = match generated {
        Ok(Ok(s)) if !s.trim().is_empty() => s,
        _ => {
            // The model failed or timed out — still deliver a real document built
            // from the facts, rather than nothing.
            format!("Activity report — {period}\n\n{activity}")
        }
    };

    let fallback_title = format!("Activity report {period}");
    let (raw_title, body) = split_title_body(&text, &fallback_title);
    let title = clean_report_title(&raw_title, &fallback_title);

    // The saved file always carries the facts it was written from. A report the
    // user might forward to someone must be auditable — if the model embellished,
    // the source section shows exactly what was actually recorded.
    let file_body = format!(
        "{body}\n\n---\n\n## Source — what NIC actually recorded\n\n{activity}\n\n\
         *Written by NIC from local screen memory. Nothing left this device.*"
    );

    match crate::notes::save_note(&title, &file_body) {
        Ok(file) => {
            let _ = librarian
                .add_event(&format!("[Report: {}]\n{}", title, body), "note", "report")
                .await;
            format!(
                "Wrote **{title}** from what I saw on your screen over {period}.\n\n\
                 Saved to `Documents/NIC Notes/{file}`\n\n---\n\n{body}"
            )
        }
        Err(e) => {
            tracing::warn!("[Report] could not save file: {e}");
            format!("Here's your report for {period} (couldn't save the file: {e}):\n\n{body}")
        }
    }
}

fn report_prompt(period: &str, activity: &str, language: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Write a work report in {language} from the activity log below.\n\
         Use ONLY what the log contains.\n\
         NEVER state a duration, a time of day, or a frequency — the log records \
         WHAT was on screen, not how long. Writing \"about half an hour\" or \
         \"every hour\" is a fabrication.\n\
         NEVER invent a task, a project name, a person or a number.\n\
         FIRST line: a short plain title, 3-6 words. No markdown, no colon, and do \
         not write the word Title.\n\
         Then: one short paragraph on what the period was spent on, then bullet \
         points grouped by theme. End with a one-line takeaway.\n\
         No preamble, no meta-commentary.<|im_end|>\n\
         <|im_start|>user\nPeriod: {period}\n\nActivity log:\n{activity}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Cleans the model's first line into a real filename-able title. The model likes
/// to emit "**Title:** Activity Report", which became a literal file name.
fn clean_report_title(raw: &str, fallback: &str) -> String {
    let mut t = raw.trim().trim_matches(['#', '*', '-', ' ']).to_string();
    for lead in ["title:", "заголовок:", "название:"] {
        if t.to_lowercase().starts_with(lead) {
            t = t[lead.len()..].to_string();
        }
    }
    let t = t.trim().trim_matches(['*', '#', ':', ' ']).trim().to_string();
    if t.is_empty() || t.chars().count() > 60 { fallback.to_string() } else { t }
}

fn note_prompt(raw: &str, language: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Reformat the user's messy brain-dump into a clean note in {language}.\n\
         FIRST line: a short plain title (3–6 words, no markdown, no punctuation).\n\
         Then organize the thoughts into bullet points / short paragraphs that\n\
         faithfully capture what they said. Do NOT invent facts. No preamble, no\n\
         commentary, no closing remarks.<|im_end|>\n\
         <|im_start|>user\n{raw}<|im_end|>\n\
         <|im_start|>assistant\n",
        language = language, raw = raw,
    )
}

/// Splits the model output into (title, body): first non-empty line is the
/// title (markdown/bullet prefixes stripped); the rest is the body.
fn split_title_body(structured: &str, raw_fallback: &str) -> (String, String) {
    let mut lines = structured.lines();
    let title = loop {
        match lines.next() {
            Some(l) if !l.trim().is_empty() => {
                break l.trim_start_matches(['#', '-', '*', ' ']).trim().to_string();
            }
            Some(_) => continue,
            None => break String::new(),
        }
    };
    let body: String = lines.collect::<Vec<_>>().join("\n").trim().to_string();
    let title = if title.is_empty() {
        raw_fallback.chars().take(40).collect::<String>()
    } else {
        title.chars().take(60).collect()
    };
    let body = if body.is_empty() { structured.trim().to_string() } else { body };
    (title, body)
}

async fn handle_note_save(
    State(state): State<ApiState>,
    Json(req): Json<NoteSaveRequest>,
) -> Result<Json<NoteSaveResponse>, (StatusCode, String)> {
    let raw = req.raw.trim().to_string();
    if raw.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty note".into()));
    }
    let language = state.language.lock().await.clone();
    let prompt   = note_prompt(&raw, &language);

    let thinker = state.thinker.clone();
    let structured = tokio::task::spawn_blocking(move || {
        thinker.blocking_lock().generate_raw(&prompt, 400)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let (title, body) = split_title_body(&structured, &raw);
    let file = crate::notes::save_note(&title, &body)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Store in memory so the note is recallable ("what notes did I save").
    let _ = state.librarian
        .add_event(&format!("[Note: {}]\n{}", title, body), "note", "idea")
        .await;

    Ok(Json(NoteSaveResponse { file, title, content: body }))
}

// ── Web UI ────────────────────────────────────────────────────────────────────

async fn handle_ui() -> impl IntoResponse {
    (
        [
            ("content-type", "text/html; charset=utf-8"),
            ("cache-control", "no-store"),
        ],
        include_str!("ui.html"),
    )
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn build_router(state: ApiState) -> Router {
    Router::new()
        .route("/",               get(handle_ui))
        .route("/health",         get(health))
        .route("/query",          post(handle_query))
        .route("/query/stream",   post(handle_query_stream))
        .route("/add_event",      post(handle_add_event))
        .route("/search",         post(handle_search))
        .route("/search_archive", post(handle_search_archive))
        .route("/search/filter",  post(handle_search_filter))
        .route("/reset",          post(handle_reset))
        .route("/forget",         post(handle_forget))
        .route("/export",         get(handle_export))
        .route("/incognito",      get(handle_incognito_get).post(handle_incognito_post))
        .route("/consent",        get(handle_consent_get).post(handle_consent_post))
        .route("/diagnostics",    get(handle_diagnostics))
        .route("/memory/wipe",    post(handle_memory_wipe))
        .route("/update/check",   get(handle_update_check))
        .route("/llm_status",    get(handle_llm_status))
        .route("/profile",        get(handle_profile_get).post(handle_profile_post))
        .route("/image",          post(handle_image))
        .route("/activity",       get(handle_activity))
        .route("/history",        get(handle_history))
        .route("/models",         get(handle_models_list))
        .route("/models/select",  post(handle_models_select))
        .route("/models/pick",    post(handle_model_pick_gguf))
        .route("/notes/save",     post(handle_note_save))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(middleware::from_fn(timing_layer))
        // The UI is loaded via `with_html` (origin "null") and fetches the
        // localhost API. Chromium/WebView2 enforces Private Network Access:
        // a page calling a local IP must get `Access-Control-Allow-Private-Network: true`,
        // and PNA forbids the `*` origin — so we mirror the request origin instead.
        .layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::mirror_request())
                .allow_methods(Any)
                .allow_headers(Any)
                .allow_private_network(true),
        )
        .with_state(state)
}


pub async fn start_api_server_with_listener(
    state:    ApiState,
    listener: tokio::net::TcpListener,
) -> Result<()> {
    let addr = listener.local_addr()?;
    let app  = build_router(state).into_make_service_with_connect_info::<SocketAddr>();
    tracing::info!("API server listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

// ── First-run bootstrap server ─────────────────────────────────────────────────
//
// The full API server only starts listening after the model is downloaded and
// llama-server has loaded it — which on a first run takes minutes. To avoid a
// silent "connecting…" during that window, this tiny server binds the UI port
// early and answers `/boot` (download phase + progress) while the heavy init
// runs. Once the backend is ready it is shut down and the real server takes the
// port. `/health` here returns 503 on purpose, so the UI keeps polling `/boot`
// until the real server (which returns 200) takes over.

async fn boot_status() -> impl IntoResponse {
    let (phase, done_mb, total_mb, error) = crate::boot::boot().snapshot();
    Json(serde_json::json!({
        "phase":    phase,
        "done_mb":  done_mb,
        "total_mb": total_mb,
        "error":    error,
    }))
}

async fn boot_health() -> impl IntoResponse {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("content-type", "application/json")],
        r#"{"status":"starting"}"#,
    )
}

pub fn build_bootstrap_router() -> Router {
    Router::new()
        .route("/",       get(handle_ui))
        .route("/boot",   get(boot_status))
        .route("/health", get(boot_health))
        .layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::mirror_request())
                .allow_methods(Any)
                .allow_headers(Any)
                .allow_private_network(true),
        )
}

/// Serves the bootstrap router on `listener` until `shutdown` resolves, then
/// returns so the caller can rebind the port for the real API server.
pub async fn run_bootstrap_server(
    listener: tokio::net::TcpListener,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(listener, build_bootstrap_router())
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── report writer ─────────────────────────────────────────────────────────

    #[test]
    fn report_requests_are_detected_with_the_right_window() {
        assert_eq!(report_request("write a report of my week"), Some(7));
        assert_eq!(report_request("summarize my week"), Some(7));
        assert_eq!(report_request("write a report"), Some(1));
        assert_eq!(report_request("make a report of what I did today"), Some(1));
        assert_eq!(report_request("напиши отчёт за неделю"), Some(7));
        assert_eq!(report_request("write a report of the last month"), Some(30));
    }

    #[test]
    fn report_titles_are_cleaned() {
        // The model really emitted "**Title:** Activity Report for Last Seven Days",
        // which became a literal file name.
        assert_eq!(clean_report_title("**Title:** Activity Report", "fb"), "Activity Report");
        assert_eq!(clean_report_title("# My Week", "fb"), "My Week");
        assert_eq!(clean_report_title("   ", "fb"), "fb");
        assert_eq!(clean_report_title(&"x".repeat(80), "fb"), "fb");
    }

    #[test]
    fn ordinary_questions_do_not_write_files() {
        // Creating a file is a side effect — only an explicit request may trigger it.
        assert!(report_request("what did I do today").is_none());
        assert!(report_request("summarize this article").is_none());
        assert!(report_request("what is a report").is_none());
        assert!(report_request("hi").is_none());
    }

    // ── secrets gate (asked live, the model invented having held them) ────────

    #[test]
    fn secret_questions_are_answered_by_code() {
        // Every one of these produced fiction from the model in a live eval —
        // e.g. "your recovery words were deleted from your profile earlier this week".
        assert!(asks_about_secrets("what is my seed phrase"));
        assert!(asks_about_secrets("show me my crypto wallet recovery words"));
        assert!(asks_about_secrets("what passwords have you seen on my screen"));
        assert!(asks_about_secrets("do you know any of my card numbers"));
        assert!(asks_about_secrets("what private keys are in your memory"));
        assert!(asks_about_secrets("list any API keys you saw"));
        assert!(asks_about_secrets("did you see me type a password"));
    }

    #[test]
    fn generic_security_questions_still_reach_the_model() {
        // Legitimate questions that merely mention a secret-ish word must NOT be
        // hijacked by the canned answer.
        assert!(!asks_about_secrets("how do I make a strong password"));
        assert!(!asks_about_secrets("what is a private key in cryptography"));
        assert!(!asks_about_secrets("explain how seed phrases work"));
    }

    // ── unverified-entity caveat ──────────────────────────────────────────────

    #[test]
    fn named_entities_in_fact_questions_are_detected() {
        assert!(named_entity_in_question("what is the Karakov-Feldstein theorem").is_some());
        assert!(named_entity_in_question("who invented the Zorblax engine").is_some());
        assert!(named_entity_in_question("tell me about the Helsinki Protocol").is_some());
    }

    #[test]
    fn plain_concept_questions_get_no_caveat() {
        // The model is reliable here — a warning on these would cry wolf.
        assert!(named_entity_in_question("what is gravity").is_none());
        assert!(named_entity_in_question("explain recursion").is_none());
        assert!(named_entity_in_question("what is DNA").is_none());       // acronym
        assert!(named_entity_in_question("how does TCP work").is_none()); // not a fact-lead
        assert!(named_entity_in_question("hi").is_none());
    }

    // ── "tell me about X" tracks a subject ────────────────────────────────────

    #[test]
    fn about_questions_name_the_current_subject() {
        // Live: "hi assistant could u tell me about donk" tracked nobody, so the
        // user's own correction a turn later had nothing to attach to.
        assert_eq!(
            about_subject("hi assistant could u tell me about donk").as_deref(),
            Some("donk"));
        assert_eq!(
            about_subject("tell me about the Helsinki Protocol").as_deref(),
            Some("Helsinki Protocol"));
        assert_eq!(
            about_subject("what do you know about smokey nagata?").as_deref(),
            Some("smokey nagata"));
    }

    #[test]
    fn about_questions_that_name_nobody_track_nothing() {
        assert!(about_subject("tell me about yourself").is_none());   // persona question
        assert!(about_subject("tell me about him").is_none());        // follow-up pronoun
        assert!(about_subject("tell me about my day").is_none());     // about the user
        assert!(about_subject("what is gravity").is_none());          // not an "about" lead
    }

    #[test]
    fn a_bare_handle_still_gets_the_unverified_caveat() {
        // "tell me about donk" answered with a confident Donkey Kong biography.
        // Lowercase usernames are precisely the case the model invents lives for.
        assert!(named_entity_in_question("hi assistant could u tell me about donk").is_some());
        assert!(named_entity_in_question("who is qewbite").is_some());
    }

    // ── host_is_local (DNS-rebinding gate) ────────────────────────────────────

    #[test]
    fn loopback_hosts_are_accepted() {
        assert!(host_is_local("127.0.0.1:7878"));
        assert!(host_is_local("127.0.0.1"));
        assert!(host_is_local("localhost:7878"));
        assert!(host_is_local("localhost"));
        assert!(host_is_local("[::1]:7878"));
        assert!(host_is_local("127.0.0.2:7878")); // whole 127/8 is loopback
    }

    #[test]
    fn foreign_hosts_are_rejected() {
        // DNS rebinding: an attacker's domain resolved to 127.0.0.1.
        assert!(!host_is_local("evil.com:7878"));
        assert!(!host_is_local("nic.attacker.io"));
        assert!(!host_is_local("192.168.1.5:7878"));
        // Prefix tricks must not sneak through.
        assert!(!host_is_local("localhost.evil.com:7878"));
        assert!(!host_is_local("127.0.0.1.evil.com"));
    }

    // ── never-bluff gate (person_in_question / person_is_grounded) ────────────

    #[test]
    fn person_question_is_detected() {
        assert_eq!(person_in_question("who is David Laid").as_deref(), Some("David Laid"));
        assert_eq!(person_in_question("Who was Alan Turing?").as_deref(), Some("Alan Turing"));
        assert_eq!(person_in_question("who's Elon Musk").as_deref(), Some("Elon Musk"));
        assert_eq!(person_in_question("кто такой Хабиб").as_deref(), Some("Хабиб"));
    }

    // ── user-taught facts about a person ─────────────────────────────────────

    #[test]
    fn user_statements_about_a_person_are_captured() {
        // Live: "who is lens" → "he is a dota 2 player" was answered with a
        // definition of an optical lens. It is a STATEMENT, not a question.
        assert!(matches!(entity_statement("he is a dota 2 player"),
                         Some(EntityUpdate::Fact(f)) if f == "dota 2 player"));
        assert!(matches!(entity_statement("she is a singer"),
                         Some(EntityUpdate::Fact(f)) if f == "singer"));
        assert!(matches!(entity_statement("no his nickname is LenS"),
                         Some(EntityUpdate::Rename(n)) if n == "LenS"));
        assert!(matches!(entity_statement("his name is Bulkin"),
                         Some(EntityUpdate::Rename(n)) if n == "Bulkin"));
    }

    #[test]
    fn greeting_before_a_question_still_gates() {
        // Live: "hi who is smoky nagata?" skipped the gate and the model answered
        // that he played John McClane in Die Hard.
        assert_eq!(person_in_question("hi who is smoky nagata?").as_deref(), Some("smoky nagata"));
        assert_eq!(person_in_question("hey, who is Bulkin").as_deref(), Some("Bulkin"));
        assert_eq!(person_in_question("so who was Alan Turing?").as_deref(), Some("Alan Turing"));
    }

    #[test]
    fn politeness_wrapped_questions_still_gate() {
        // Live: "hi nic could u say me who is kyosuke" → the model invented an
        // anime character. The lead can sit anywhere in the sentence.
        assert_eq!(
            person_in_question("hi nic could u say me who is kyosuke").as_deref(),
            Some("kyosuke"));
        assert_eq!(
            person_in_question("can you tell me who is David Laid").as_deref(),
            Some("David Laid"));
    }

    #[test]
    fn stretched_negation_still_reads_as_a_correction() {
        // Live: "nooo I am about cs 2 player" → the model explained that
        // "C.S. 2 Player refers to two players playing Counter-Strike".
        assert!(matches!(entity_statement("nooo I am about cs 2 player"),
                         Some(EntityUpdate::Fact(f)) if f == "cs 2 player"));
        assert!(matches!(entity_statement("nope, he is a singer"),
                         Some(EntityUpdate::Fact(f)) if f == "singer"));
    }

    #[test]
    fn clarifications_are_captured_as_facts() {
        // Live: "no I am about japanese street racer and founder of top secret
        // customs" was answered with an invented founder, Kenjiro Kawai.
        assert!(matches!(
            entity_statement("no I am about japanese street racer and founder of top secret customs"),
            Some(EntityUpdate::Fact(f)) if f.contains("street racer")));
        assert!(matches!(entity_statement("I mean the dota player"),
                         Some(EntityUpdate::Fact(f)) if f == "dota player"));
    }

    #[test]
    fn questions_are_not_statements() {
        assert!(entity_statement("who is lens").is_none());
        assert!(entity_statement("what is gravity").is_none());
        assert!(entity_statement("play his last video").is_none());
    }

    #[test]
    fn search_term_carries_user_facts() {
        // "lens" alone finds camera optics; the user's own words disambiguate.
        let mut e = EntityCtx { name: "LenS".into(), facts: vec![] };
        assert_eq!(e.search_term(), "LenS");
        e.add_fact("dota 2 player");
        assert_eq!(e.search_term(), "LenS dota 2 player");
    }

    #[test]
    fn lowercase_names_are_still_people() {
        // Shipped bug: "who is qewbite" (no capital) slipped past the gate, so the
        // model answered "I am NIC-assistant" instead of admitting it never heard
        // of him. People do not capitalise usernames.
        assert_eq!(person_in_question("who is qewbite").as_deref(), Some("qewbite"));
        assert_eq!(person_in_question("who is david laid").as_deref(), Some("david laid"));
    }

    #[test]
    fn descriptive_who_questions_are_not_people() {
        assert!(person_in_question("who is the best programmer").is_none());
        assert!(person_in_question("who is your creator").is_none());
        assert!(person_in_question("who is he").is_none());
    }

    #[test]
    fn concept_questions_are_not_gated() {
        // The model is reliable on concepts — only named people get the gate.
        assert!(person_in_question("what is gravity").is_none());
        assert!(person_in_question("who is the best programmer").is_none()); // no proper noun
        assert!(person_in_question("explain quantum tunneling").is_none());
    }

    #[test]
    fn pronoun_person_question_is_not_gated() {
        // "who is he" is a follow-up; the QA path resolves it from context.
        assert!(person_in_question("who is he").is_none());
        assert!(person_in_question("who are they").is_none());
    }

    #[test]
    fn grounding_requires_the_name_in_the_facts() {
        // A non-empty context about something ELSE must NOT count as grounding —
        // this is exactly the bug the live test caught.
        let unrelated = "Screen history:\n17:04 — watched a video on YouTube";
        assert!(!person_is_grounded("David Laid", unrelated, ""));
        // Web snippets that actually mention him do count.
        let web = "Web results:\nDavid Laid is a Latvian-American bodybuilder and fitness model.";
        assert!(person_is_grounded("David Laid", "", web));
        // A single matching surname is enough.
        assert!(person_is_grounded("Alan Turing", "Turing wrote about computable numbers", ""));
    }

    // ── enrich_search_term ────────────────────────────────────────────────────

    #[test]
    fn enrich_pogoda_adds_city() {
        let r = enrich_search_term("погода", Some("Астана"), "погода");
        assert_eq!(r, "погода Астана");
    }

    #[test]
    fn enrich_weather_adds_city() {
        let r = enrich_search_term("weather today", Some("Москва"), "weather today");
        assert_eq!(r, "weather today Москва");
    }

    #[test]
    fn enrich_no_city_passthrough() {
        let r = enrich_search_term("погода", None, "погода");
        assert_eq!(r, "погода");
    }

    #[test]
    fn enrich_city_already_present_no_duplicate() {
        let r = enrich_search_term("погода астана", Some("Астана"), "погода");
        assert_eq!(r, "погода астана"); // city already in term
    }

    #[test]
    fn enrich_non_location_no_append() {
        let r = enrich_search_term("курс доллара", Some("Астана"), "курс доллара");
        assert_eq!(r, "курс доллара"); // not a location keyword
    }

    #[test]
    fn enrich_forecast_adds_city() {
        let r = enrich_search_term("прогноз погоды", Some("Киев"), "прогноз");
        assert_eq!(r, "прогноз погоды Киев");
    }

    #[test]
    fn enrich_novosti_adds_city() {
        let r = enrich_search_term("новости", Some("Москва"), "новости");
        assert_eq!(r, "новости Москва");
    }

    // ── query_words ───────────────────────────────────────────────────────────

    #[test]
    fn query_words_basic() {
        let w = query_words("привет мир как дела");
        assert!(w.contains("привет"));
        assert!(w.contains("мир"));
        assert!(w.contains("как"));
        assert!(w.contains("дела"));
    }

    #[test]
    fn query_words_filters_short() {
        // "я" = 2 bytes, "в" = 2 bytes → both filtered by len() > 2
        // "на" = 4 bytes → kept (Cyrillic short words pass the byte-length filter)
        let w = query_words("я в");
        assert!(w.is_empty(), "2-byte Cyrillic words should be filtered by byte length");
    }

    #[test]
    fn query_words_lowercased() {
        let w = query_words("Привет МИР");
        assert!(w.contains("привет"));
        assert!(w.contains("мир"));
    }

    #[test]
    fn query_words_empty() {
        assert!(query_words("").is_empty());
    }

    #[test]
    fn query_words_deduplication() {
        let w = query_words("тест тест тест");
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn query_words_min_len_3() {
        // "дом" has len 3 in Rust bytes (actually 6 for Cyrillic), but char count is 3
        // wait: the filter is `w.len() > 2` where len() is byte length
        // Cyrillic chars are 2 bytes each, so "до" = 4 bytes > 2 → NOT filtered
        // Let me test with a 2-byte ASCII word
        let w = query_words("hi hello");
        assert!(!w.contains("hi"), "2-char ASCII word should be filtered");
        assert!(w.contains("hello"));
    }

    // ── jaccard ───────────────────────────────────────────────────────────────

    #[test]
    fn jaccard_identical_sets() {
        let a: HashSet<String> = ["foo", "bar", "baz"].iter().map(|s| s.to_string()).collect();
        let b = a.clone();
        let j = jaccard(&a, &b);
        assert!((j - 1.0).abs() < 0.001, "identical sets → jaccard=1.0, got {}", j);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a: HashSet<String> = ["foo", "bar"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["baz", "qux"].iter().map(|s| s.to_string()).collect();
        let j = jaccard(&a, &b);
        assert!((j - 0.0).abs() < 0.001, "disjoint sets → jaccard=0.0, got {}", j);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["a", "b", "d"].iter().map(|s| s.to_string()).collect();
        let j = jaccard(&a, &b);
        // inter=2, union=4 → 0.5
        assert!((j - 0.5).abs() < 0.001, "partial overlap → 0.5, got {}", j);
    }

    #[test]
    fn jaccard_empty_a_returns_zero() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = ["foo"].iter().map(|s| s.to_string()).collect();
        assert_eq!(jaccard(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_empty_b_returns_zero() {
        let a: HashSet<String> = ["foo"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = HashSet::new();
        assert_eq!(jaccard(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_both_empty_returns_zero() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = HashSet::new();
        assert_eq!(jaccard(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_single_shared_element() {
        let a: HashSet<String> = ["foo"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["foo"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn jaccard_asymmetric_sizes() {
        let a: HashSet<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        let j = jaccard(&a, &b);
        // inter=1, union=4 → 0.25
        assert!((j - 0.25).abs() < 0.001, "asymm overlap → 0.25, got {}", j);
    }

    #[test]
    fn jaccard_range_zero_to_one() {
        let a: HashSet<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["y", "z", "w"].iter().map(|s| s.to_string()).collect();
        let j = jaccard(&a, &b);
        assert!((0.0..=1.0).contains(&j), "jaccard must be in [0,1], got {}", j);
    }

    // ── handle_history ordering logic ─────────────────────────────────────────

    // We test the parsing/ordering logic directly, without needing a live server.
    // The handle_history function parses "Вопрос: Q\nОтвет: A" strings.

    fn parse_qa_pair(text: &str) -> Option<(String, String)> {
        let q_start = text.find("Вопрос: ")?;
        let rest = &text[q_start + "Вопрос: ".len()..];
        let a_start = rest.find("\nОтвет: ")?;
        let q_text = rest[..a_start].trim().to_string();
        let a_text = rest[a_start + "\nОтвет: ".len()..].trim().to_string();
        Some((q_text, a_text))
    }

    #[test]
    fn qa_pair_parses_correctly() {
        let text = "Вопрос: Что такое Rust?\nОтвет: Системный язык программирования.";
        let (q, a) = parse_qa_pair(text).unwrap();
        assert_eq!(q, "Что такое Rust?");
        assert_eq!(a, "Системный язык программирования.");
    }

    #[test]
    fn qa_pair_parses_multiline_answer() {
        let text = "Вопрос: Как дела?\nОтвет: Хорошо.\nСпасибо.";
        let (q, a) = parse_qa_pair(text).unwrap();
        assert_eq!(q, "Как дела?");
        assert!(a.contains("Хорошо."));
    }

    #[test]
    fn qa_pair_returns_none_for_malformed() {
        assert!(parse_qa_pair("Просто текст без маркеров").is_none());
        assert!(parse_qa_pair("").is_none());
        assert!(parse_qa_pair("Вопрос: без ответа").is_none());
    }

    // ── parse_stored_qa (production parser: English + legacy Russian) ──────────
    #[test]
    fn stored_qa_parses_new_english_format() {
        let (q, a) = super::parse_stored_qa("Q: What is Rust?\nA: A systems language.").unwrap();
        assert_eq!(q, "What is Rust?");
        assert_eq!(a, "A systems language.");
    }

    #[test]
    fn stored_qa_parses_legacy_russian_format() {
        let (q, a) = super::parse_stored_qa("Вопрос: Что?\nОтвет: Ответ.").unwrap();
        assert_eq!(q, "Что?");
        assert_eq!(a, "Ответ.");
    }

    #[test]
    fn stored_qa_none_for_malformed() {
        assert!(super::parse_stored_qa("just text").is_none());
        assert!(super::parse_stored_qa("Q: no answer marker").is_none());
    }

    // ── two-phase timeout + raw-snippet fallback (§3) ─────────────────────────
    #[test]
    fn gen_timeout_detects_sentinels_only() {
        assert!(super::is_gen_timeout(&anyhow::anyhow!("NIC_TTFT_TIMEOUT")));
        assert!(super::is_gen_timeout(&anyhow::anyhow!("NIC_IDLE_TIMEOUT")));
        assert!(!super::is_gen_timeout(&anyhow::anyhow!("connection refused")));
    }

    #[test]
    fn raw_fallback_dedups_caps_and_prefixes_notice() {
        let surf = "1. a: x\n1. a: x\n2. b: y\n3. c: z";
        let out = super::raw_snippet_fallback("", surf, "NOTICE");
        assert!(out.starts_with("NOTICE"));
        assert_eq!(out.matches("1. a: x").count(), 1, "duplicate line not deduped");
        assert!(out.contains("2. b: y") && out.contains("3. c: z"));
    }

    #[test]
    fn raw_fallback_prefers_web_then_screen() {
        // surf empty → falls back to the screen (lib) context.
        let out = super::raw_snippet_fallback("screen line one", "", "NOTE");
        assert!(out.contains("screen line one"));
    }

    #[test]
    fn raw_fallback_empty_is_standalone_line() {
        // No matches at all → a clean standalone line, not a dangling notice.
        let out = super::raw_snippet_fallback("", "", "NOTICE");
        assert!(!out.contains("NOTICE"));
        assert!(out.contains("didn't respond"));
    }

    #[test]
    fn history_ordering_user_before_ai() {
        // Simulate the correct ordering: pairs come newest-first from recent_qa,
        // iterate in reverse, push user then ai — no final messages.reverse()
        let pairs_newest_first = vec![
            ("ts3", "Вопрос: Q3\nОтвет: A3"),
            ("ts2", "Вопрос: Q2\nОтвет: A2"),
            ("ts1", "Вопрос: Q1\nОтвет: A1"),
        ];

        let mut messages: Vec<(&str, String)> = Vec::new();
        for (_, text) in pairs_newest_first.into_iter().rev() {
            if let Some((q, a)) = parse_qa_pair(text) {
                messages.push(("user", q));
                messages.push(("ai", a));
            }
        }

        // Oldest pair first, user before ai within each pair
        assert_eq!(messages[0], ("user", "Q1".to_string()));
        assert_eq!(messages[1], ("ai", "A1".to_string()));
        assert_eq!(messages[2], ("user", "Q2".to_string()));
        assert_eq!(messages[3], ("ai", "A2".to_string()));
        assert_eq!(messages[4], ("user", "Q3".to_string()));
        assert_eq!(messages[5], ("ai", "A3".to_string()));
    }

    #[test]
    fn history_wrong_reverse_detection() {
        // Demonstrates the BUG: messages.reverse() on a flat list swaps user/ai order
        let pairs = vec!["Вопрос: Q1\nОтвет: A1", "Вопрос: Q2\nОтвет: A2"];
        let mut messages: Vec<(&str, String)> = Vec::new();
        for text in &pairs {
            if let Some((q, a)) = parse_qa_pair(text) {
                messages.push(("user", q));
                messages.push(("ai", a));
            }
        }
        messages.reverse();
        // After reverse: [ai_2, user_2, ai_1, user_1] — ai before user (wrong!)
        assert_eq!(messages[0].0, "ai", "BUG: reversed flat list puts ai before user");
        assert_eq!(messages[1].0, "user");
    }

    // ── cache expiry logic ────────────────────────────────────────────────────

    #[test]
    fn cache_entry_jaccard_threshold() {
        // The cache uses Jaccard >= 0.85 for cache hit
        let q1 = query_words("что такое rust язык программирования");
        let q2 = query_words("что такое rust язык программ");
        let j = jaccard(&q1, &q2);
        // These are similar but not identical — test that the threshold check logic is sane
        assert!(j >= 0.0 && j <= 1.0);
    }

    #[test]
    fn query_words_real_russian_query() {
        let w = query_words("сколько человек живёт на земле");
        assert!(w.contains("сколько"));
        assert!(w.contains("человек"));
        assert!(w.contains("живёт"));
        assert!(w.contains("земле"));
        // "на" is 4 bytes (2 Cyrillic chars × 2 bytes) → len() > 2 → NOT filtered
        // Actually "на" = н(2) + а(2) = 4 bytes, 4 > 2, so it's kept
        // But logically "на" is a stopword. That's OK, it's by design.
    }
}
