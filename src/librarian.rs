/// Librarian — Eternal Context Engine (production-hardened).
///
/// Tables in LanceDB:
///
///   "context"  (legacy, kept for backward compat):
///     id, timestamp, app_name, window_title, vector, text_summary
///
///   "unified_memory"  (primary eternal context):
///     id          Utf8
///     time_start  Timestamp(µs)
///     time_end    Timestamp(µs)
///     level       Int32   — 0: raw  1: 1-hour summary  2: 7-day topic node
///     archived    Boolean
///     data_json   Utf8    — JSON payload
///     embedding   FixedSizeList<f32, 384>
///
/// All embed() calls run on the blocking thread pool (spawn_blocking).
/// An LRU cache (configurable, default 100 entries) avoids redundant embedding.
/// Archive files are optionally AES-256-GCM encrypted.

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow_array::{BooleanArray, Float32Array, Int32Array, TimestampMicrosecondArray};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, OsRng};
use aes_gcm::aead::rand_core::RngCore as _;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use chrono::{DateTime, Duration, Utc};
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use lru::LruCache;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tokio::time;
use tracing::{info, warn};
use uuid::Uuid;

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use crate::llm_utils::LlmEngine;

// ── Chronos-Linking ───────────────────────────────────────────────────────────

/// Minimum cosine similarity for two facts to be considered the same ongoing session.
/// At 0.90 the content must be ≥90% semantically identical — safe for same-file editing.
const CHRONOS_SIMILARITY_THRESHOLD: f32 = 0.90;

/// In-memory pointer to the last written L0 record.
/// Used by Chronos-Linking to extend an existing fact's time span instead of inserting a new one.
struct LastFact {
    id:           String,
    emb:          Vec<f32>,
    app_name:     String,
    window_title: String,
    time_start:   i64,
    data_json:    String,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32  = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32  = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na * nb)
}

/// Updated on every query so the Deep Miner can detect user idleness.
static LAST_QUERY_TS: AtomicI64 = AtomicI64::new(0);

// ── "context" table column indices ────────────────────────────────────────────
const COL_TS:      usize = 1;
const COL_APP:     usize = 2;
const COL_WIN:     usize = 3;
const COL_SUMMARY: usize = 5;

// ── "unified_memory" column indices ───────────────────────────────────────────
const UM_ID:         usize = 0;
const UM_TIME_START: usize = 1;
const UM_TIME_END:   usize = 2;
const UM_LEVEL:      usize = 3;
// archived = 4 (accessed only during compression)
const UM_DATA:      usize = 5;
// embedding = 6 — referenced by name in vector_search(..).column("embedding")

// ── Device selection ──────────────────────────────────────────────────────────

fn best_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        tracing::info!("Embedder: CUDA GPU");
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        tracing::info!("Embedder: Metal GPU");
        return d;
    }
    Device::Cpu
}

/// Strips function bodies, keeping signatures / comments / declarations.
fn summarize_code(content: &str) -> String {
    let is_code = ["fn ", "pub ", "impl ", "struct ", "enum ", "def ", "class ", "function "]
        .iter()
        .any(|kw| content.contains(kw));

    if !is_code {
        return content.chars().take(500).collect();
    }

    let keep_prefixes = [
        "///", "//!", "pub ", "fn ", "async fn", "impl ", "struct ", "enum ",
        "type ", "trait ", "use ", "mod ", "const ", "static ", "#[", "where ",
    ];

    let mut out   = String::new();
    let mut depth: i32 = 0;

    for line in content.lines() {
        let t      = line.trim();
        let opens  = t.chars().filter(|&c| c == '{').count() as i32;
        let closes = t.chars().filter(|&c| c == '}').count() as i32;

        if depth == 0 || t.is_empty() || keep_prefixes.iter().any(|p| t.starts_with(p)) {
            out.push_str(line);
            out.push('\n');
        }

        depth = (depth + opens - closes).max(0);
    }

    out.chars().take(2000).collect()
}

// ── Embedder ──────────────────────────────────────────────────────────────────

struct Embedder {
    model:     BertModel,
    tokenizer: Tokenizer,
    device:    Device,
}

impl Embedder {
    fn new(model_dir: &PathBuf) -> Result<Self> {
        let device    = best_device();
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("Tokenizer: {}", e))?;
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        Ok(Self { model: BertModel::load(vb, &config)?, tokenizer, device })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens   = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("Tokenize: {}", e))?;
        let ids      = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let type_ids = ids.zeros_like()?;
        let mask     = ids.ones_like()?;
        let out      = self.model.forward(&ids, &type_ids, Some(&mask))?;
        Ok(out.mean(1)?.squeeze(0)?.to_vec1()?)
    }
}

// ── AES-256-GCM archive cipher ────────────────────────────────────────────────

struct ArchiveCipher {
    key: [u8; 32],
}

impl ArchiveCipher {
    fn new(key: [u8; 32]) -> Self { Self { key } }

    fn encrypt(&self, plaintext: &str) -> Result<String> {
        let cipher = Aes256Gcm::new(&self.key.into());
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce      = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher.encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| anyhow!("AES-GCM encrypt: {:?}", e))?;
        let mut combined = nonce_bytes.to_vec();
        combined.extend_from_slice(&ciphertext);
        Ok(BASE64.encode(&combined))
    }

    fn decrypt(&self, encoded: &str) -> Result<String> {
        let combined = BASE64.decode(encoded)
            .map_err(|e| anyhow!("Base64 decode: {}", e))?;
        if combined.len() < 13 {
            return Err(anyhow!("Ciphertext too short"));
        }
        let (nonce_bytes, ciphertext) = combined.split_at(12);
        let cipher    = Aes256Gcm::new(&self.key.into());
        let nonce     = Nonce::from_slice(nonce_bytes);
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("AES-GCM decrypt: {:?}", e))?;
        String::from_utf8(plaintext).map_err(|e| anyhow!("UTF-8: {}", e))
    }
}

fn load_or_create_key(path: &Path) -> Result<[u8; 32]> {
    if path.exists() {
        let s     = std::fs::read_to_string(path)?;
        let bytes = BASE64.decode(s.trim()).map_err(|e| anyhow!("Key decode: {}", e))?;
        if bytes.len() != 32 {
            return Err(anyhow!("Invalid key length in {}", path.display()));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes);
        Ok(key)
    } else {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, BASE64.encode(&key))?;
        tracing::info!("Generated new archive key at {}", path.display());
        Ok(key)
    }
}

// ── Schemas ───────────────────────────────────────────────────────────────────

fn make_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id",           DataType::Utf8, false),
        Field::new("timestamp",    DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("app_name",     DataType::Utf8, false),
        Field::new("window_title", DataType::Utf8, false),
        Field::new("vector",       DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)), 384), false),
        Field::new("text_summary", DataType::Utf8, false),
    ]))
}

fn make_unified_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id",         DataType::Utf8, false),
        Field::new("time_start", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("time_end",   DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("level",      DataType::Int32, false),
        Field::new("archived",   DataType::Boolean, false),
        Field::new("data_json",  DataType::Utf8, false),
        Field::new("embedding",  DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)), 384), false),
    ]))
}

// ── Internal row types ────────────────────────────────────────────────────────

struct EventRow {
    timestamp:    i64,
    app_name:     String,
    window_title: String,
    text_summary: String,
}

struct UnifiedRow {
    id:         String,
    time_start: i64,
    time_end:   i64,
    level:      i32,
    data_json:  String,
}

// ── Librarian ─────────────────────────────────────────────────────────────────

pub struct Librarian {
    embedder_read:  Arc<Mutex<Embedder>>,  // query / RAG path — LRU-cached
    embedder_write: Arc<Mutex<Embedder>>,  // write path (Sentinel, compression) — no lock contention with queries
    embed_cache:    Arc<Mutex<LruCache<String, Vec<f32>>>>,
    table:          Table,
    unified:        Table,
    summarizer:     Arc<std::sync::Mutex<LlmEngine>>,
    cipher:         Option<Arc<ArchiveCipher>>,
    archive_path:   PathBuf,
    _db_path:       PathBuf,
    /// Chronos-Linking: pointer to the last written L0 record for deduplication.
    last_fact:      Mutex<Option<LastFact>>,
    /// In-memory L0 event counter — incremented on every fresh insert, initialized from DB on startup.
    event_count:    Arc<AtomicU64>,
    /// Cached disk sizes (bytes) — refreshed every 60 s by a background task; avoids blocking 22k-file walks on /health.
    cached_db_bytes:      Arc<AtomicU64>,
    cached_archive_bytes: Arc<AtomicU64>,
}

impl Librarian {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Constructor driven by `AppConfig` + a shared LLM engine.
    pub async fn new_from_config(
        cfg: &crate::config::AppConfig,
        llm: Arc<std::sync::Mutex<LlmEngine>>,
    ) -> Result<Self> {
        let key_path = if cfg.security.encrypt_archives {
            Some(cfg.security.key_file.clone())
        } else {
            None
        };
        Self::new_configured(
            cfg.models.embedder_dir.clone(),
            llm,
            cfg.librarian.db_path.clone(),
            cfg.librarian.archive_path.clone(),
            cfg.librarian.embed_cache_size,
            cfg.security.encrypt_archives,
            key_path,
        ).await
    }

    async fn new_configured(
        embedder_dir:     PathBuf,
        llm:              Arc<std::sync::Mutex<LlmEngine>>,
        db_path:          PathBuf,
        archive_path:     PathBuf,
        embed_cache_size: usize,
        encrypt_archives: bool,
        key_path:         Option<PathBuf>,
    ) -> Result<Self> {
        let embedder_read  = Embedder::new(&embedder_dir)?;
        let embedder_write = Embedder::new(&embedder_dir)?;
        let summarizer = llm;

        // Open the memory DB, self-healing on corruption: a bad shutdown / full
        // disk can leave LanceDB unreadable. Rather than brick the whole app on
        // boot, quarantine the corrupt store (renamed aside, never deleted) and
        // start fresh — the user gets a working NIC + a recoverable backup, not
        // a dead window. One retry only, so a truly unwritable dir still errors
        // (surfaced on the boot screen, not a silent hang).
        let db_str = db_path.to_string_lossy().to_string();
        let mut recovered = false;
        // `_db` (the Connection) is kept alive for the whole init like before;
        // the Tables are what we actually store.
        let (_db, table, unified) = loop {
            let attempt = async {
                let db = connect(&db_str).execute().await?;
                let table = match db.open_table("context").execute().await {
                    Ok(t)  => t,
                    Err(_) => db.create_empty_table("context", make_schema()).execute().await?,
                };
                let unified = match db.open_table("unified_memory").execute().await {
                    Ok(t)  => t,
                    Err(_) => db.create_empty_table("unified_memory", make_unified_schema()).execute().await?,
                };
                Ok::<_, anyhow::Error>((db, table, unified))
            }.await;
            match attempt {
                Ok(v) => break v,
                Err(e) if !recovered => {
                    tracing::error!(
                        "[Librarian] memory DB failed to open ({e:#}); quarantining and starting fresh");
                    quarantine_broken_db(&db_path);
                    recovered = true;
                }
                Err(e) => return Err(e).context("memory DB unusable even after a fresh start"),
            }
        };

        std::fs::create_dir_all(&archive_path)?;

        let cipher = if encrypt_archives {
            let kp  = key_path.unwrap_or_else(|| PathBuf::from("data/.nic_key"));
            let key = load_or_create_key(&kp)?;
            Some(Arc::new(ArchiveCipher::new(key)))
        } else {
            None
        };

        let cache_size = NonZeroUsize::new(embed_cache_size.max(1))
            .unwrap_or(NonZeroUsize::MIN);

        let event_count = Arc::new(AtomicU64::new(0));

        let cached_db_bytes      = Arc::new(AtomicU64::new(0));
        let cached_archive_bytes = Arc::new(AtomicU64::new(0));
        // Background task: refresh disk-size cache every 60 s (avoids blocking walks on /health).
        {
            let db_bg  = db_path.clone();
            let ar_bg  = archive_path.clone();
            let db_cnt = cached_db_bytes.clone();
            let ar_cnt = cached_archive_bytes.clone();
            tokio::spawn(async move {
                loop {
                    let db_p = db_bg.clone();
                    let ar_p = ar_bg.clone();
                    let (db_sz, ar_sz) = tokio::task::spawn_blocking(move || {
                        (dir_size_bytes(&db_p), dir_size_bytes(&ar_p))
                    }).await.unwrap_or((0, 0));
                    db_cnt.store(db_sz, Ordering::Relaxed);
                    ar_cnt.store(ar_sz, Ordering::Relaxed);
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                }
            });
        }

        Ok(Self {
            embedder_read:  Arc::new(Mutex::new(embedder_read)),
            embedder_write: Arc::new(Mutex::new(embedder_write)),
            embed_cache:    Arc::new(Mutex::new(LruCache::new(cache_size))),
            table,
            unified,
            summarizer,
            cipher,
            archive_path,
            _db_path:    db_path,
            last_fact:   Mutex::new(None),
            event_count,
            cached_db_bytes,
            cached_archive_bytes,
        })
    }

    // ── Async embed with LRU cache ────────────────────────────────────────────

    /// Embeds `text` for the **read / RAG path**. Results are LRU-cached.
    /// Uses `embedder_read` which is never held during Sentinel writes → zero contention.
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let key = embed_cache_key(text);

        if let Some(cached) = self.embed_cache.lock().await.get(&key).cloned() {
            return Ok(cached);
        }

        let embedder = self.embedder_read.clone();
        let text_s   = text.to_string();
        let emb = tokio::task::spawn_blocking(move || {
            embedder.blocking_lock().embed(&text_s)
        })
        .await
        .map_err(|e| anyhow!("spawn_blocking join: {}", e))??;

        self.embed_cache.lock().await.put(key, emb.clone());
        Ok(emb)
    }

    /// Embeds `text` for the **write path** (Sentinel stores, daily compression).
    /// Uses a dedicated `embedder_write` instance — never contends with query RAG.
    async fn embed_write(&self, text: &str) -> Result<Vec<f32>> {
        let embedder = self.embedder_write.clone();
        let text_s   = text.to_string();
        tokio::task::spawn_blocking(move || {
            embedder.blocking_lock().embed(&text_s)
        })
        .await
        .map_err(|e| anyhow!("spawn_blocking join: {}", e))?
    }

    // ── Primary write API ─────────────────────────────────────────────────────

    /// Stores one OCR event in unified_memory (level 0) with Chronos-Linking.
    ///
    /// When Chronos-Linking fires (same app+window, similarity ≥ 0.90), the
    /// existing fact is extended in-place and the legacy "context" write is
    /// skipped — no duplicate record is created.  Returns the new record id on
    /// a fresh insert, or an empty string when the fact was extended.
    pub async fn store(
        &self,
        app_name:     &str,
        window_title: &str,
        text_summary: &str,
    ) -> Result<String> {
        let t_store = std::time::Instant::now();
        let ts  = Utc::now().timestamp_micros();
        let emb = self.embed_write(text_summary).await?;

        // unified_memory write with Chronos-Linking dedup.
        // Returns true = fresh L0 record; false = existing fact extended.
        let is_fresh = self.store_unified(app_name, window_title, text_summary, ts, emb.clone())
            .await
            .unwrap_or(true);

        let store_ms = t_store.elapsed().as_millis();
        info!("[PERF] Vector Search & Librarian write: {} ms (fresh={})", store_ms, is_fresh);
        crate::perf::global().record_store(store_ms, is_fresh);

        if !is_fresh {
            return Ok(String::new());
        }

        self.event_count.fetch_add(1, Ordering::Relaxed);

        // Fresh fact: also write to the legacy "context" table for fallback RAG.
        let id     = Uuid::new_v4().to_string();
        let schema = make_schema();
        let values = Arc::new(Float32Array::from(emb)) as ArrayRef;
        let vector = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            384, values, None,
        );
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec![id.as_str()]))         as ArrayRef,
            Arc::new(TimestampMicrosecondArray::from(vec![ts]))    as ArrayRef,
            Arc::new(StringArray::from(vec![app_name]))            as ArrayRef,
            Arc::new(StringArray::from(vec![window_title]))        as ArrayRef,
            Arc::new(vector)                                       as ArrayRef,
            Arc::new(StringArray::from(vec![text_summary]))        as ArrayRef,
        ])?;
        self.table.add(vec![batch]).execute().await?;

        info!("[LIBRARIAN]: New L0 fact. OCR: {} chars | {} / {}",
              text_summary.len(), app_name, window_title);
        Ok(id)
    }

    /// Backward-compat shim.
    pub async fn add_event(&self, text: &str, source: &str, event_type: &str) -> Result<String> {
        self.store(source, event_type, text).await
    }

    /// Evicts all cached embeddings. Call on "reset" to force fresh context retrieval.
    pub async fn clear_embed_cache(&self) {
        self.embed_cache.lock().await.clear();
        info!("[Librarian] Embedding cache cleared.");
    }

    // ── Diagnostic API ────────────────────────────────────────────────────────

    /// Returns (timestamp_str, ocr_text) of the most recent screen (non-clipboard) Level-0 record.
    pub async fn last_screen_snapshot(&self) -> Option<(String, String)> {
        let mut stream = self.unified.query().only_if("level = 0").execute().await.ok()?;
        let mut best: Option<(i64, String)> = None;

        while let Some(batch) = stream.next().await {
            let batch = match batch { Ok(b) => b, Err(_) => continue };
            let ts_arr  = match batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>() {
                Some(a) => a, None => continue,
            };
            let dat_arr = match batch.column(UM_DATA).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            for i in 0..batch.num_rows() {
                let data = dat_arr.value(i);
                if !is_screen_event(data) { continue; }
                let ts = ts_arr.value(i);
                if best.as_ref().map_or(true, |(t, _)| ts > *t) {
                    best = Some((ts, data.to_string()));
                }
            }
        }

        best.map(|(ts, data)| (fmt_ts(ts), extract_text_from_json(&data)))
    }

    /// Returns the last `n` screen (non-clipboard) L0 facts as (timestamp_str, ocr_text) pairs,
    /// sorted newest-first.
    pub async fn last_n_screen_facts(&self, n: usize) -> Vec<(i64, String)> {
        let mut stream = match self.unified.query().only_if("level = 0").execute().await {
            Ok(s)  => s,
            Err(_) => return vec![],
        };
        let mut all: Vec<(i64, String)> = Vec::new();

        while let Some(batch) = stream.next().await {
            let batch = match batch { Ok(b) => b, Err(_) => continue };
            let ts_arr  = match batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>() {
                Some(a) => a, None => continue,
            };
            let dat_arr = match batch.column(UM_DATA).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            for i in 0..batch.num_rows() {
                let data = dat_arr.value(i);
                if !is_screen_event(data) { continue; }
                all.push((ts_arr.value(i), data.to_string()));
            }
        }

        all.sort_by(|a, b| b.0.cmp(&a.0));
        all.into_iter()
            .take(n)
            .map(|(ts, data)| (ts, extract_text_from_json(&data)))
            .collect()
    }

    /// Scans all L0 screen events and returns them as `(ts, app, title)` tuples,
    /// newest-first, with dialogue / notes / NIC's own window filtered out.
    /// Shared by `iron_log` (formatted lines) and `activity_summary` (retelling).
    async fn collect_recent_activity(&self) -> Vec<(i64, String, String)> {
        let mut stream = match self.unified.query().only_if("level = 0").execute().await {
            Ok(s)  => s,
            Err(_) => return vec![],
        };
        let mut all: Vec<(i64, String, String)> = Vec::new(); // (ts, app, title)

        while let Some(batch) = stream.next().await {
            let batch = match batch { Ok(b) => b, Err(_) => continue };
            let ts_arr  = match batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>() {
                Some(a) => a, None => continue,
            };
            let dat_arr = match batch.column(UM_DATA).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            for i in 0..batch.num_rows() {
                let data = dat_arr.value(i);
                if !is_screen_event(data) { continue; }
                let app   = extract_app_name_from_json(data);
                let title = extract_window_title_from_json(data);
                let al = app.to_lowercase();
                // Skip dialogue/notes (not screen activity) and NIC's own window.
                if al == "dialogue" || al == "qa" || al == "note" { continue; }
                if al.contains("nic-assistant") || al.contains("nic assistant")
                    || title.to_lowercase().contains("nic-assistant") { continue; }
                all.push((ts_arr.value(i), app, title));
            }
        }

        all.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
        all
    }

    /// Learns the user's preferred service for `want` (Audio/Video) from recent
    /// screen memory, so "play X" opens X where the user actually listens/watches.
    /// A dedicated service (SoundCloud/Spotify/Netflix…) wins over the versatile
    /// YouTube when the user has one; otherwise YouTube is the fallback. Pure
    /// frequency×recency over app+title — NO LLM. `None` when memory shows none.
    pub async fn preferred_service(
        &self,
        want: crate::services::Kind,
    ) -> Option<&'static crate::services::Service> {
        let acts = self.collect_recent_activity().await;
        if acts.is_empty() { return None; }
        top_service(&acts, want, true).or_else(|| top_service(&acts, want, false))
    }

    /// The most recently seen music/video service (any kind), for "continue".
    pub async fn most_recent_service(&self) -> Option<&'static crate::services::Service> {
        let acts = self.collect_recent_activity().await;
        for (_, app, title) in acts.iter() {
            let hay = format!("{} {}", app, title).to_lowercase();
            if let Some(svc) = crate::services::SERVICES.iter()
                .find(|s| s.aliases.iter().any(|a| hay.contains(a)))
            {
                return Some(svc);
            }
        }
        None
    }

    /// The "iron log": last `n` screen events as clean, human, chronological log
    /// lines (newest first), built deterministically from app + window title —
    /// NO OCR, NO LLM. Consecutive identical activities are collapsed. This is
    /// what the model-path recall feeds in for screen questions it phrases.
    /// Each line: `14:22 (5 min ago) — watched «iOS 27 review» on YouTube`.
    pub async fn iron_log(&self, n: usize) -> Vec<String> {
        let all = self.collect_recent_activity().await;
        let mut out: Vec<String> = Vec::new();
        let mut last_phrase = String::new();
        for (ts, app, title) in all {
            let phrase = activity_line(&app, &title);
            if phrase == last_phrase { continue; } // collapse consecutive repeats
            last_phrase = phrase.clone();
            out.push(format!("{} ({}) — {}", fmt_hm(ts), rel_age(ts), phrase));
            if out.len() >= n { break; }
        }
        out
    }

    /// Deterministic "iron retelling" of recent activity: a natural summary
    /// sentence (e.g. "Вчера ты смотрел видео и работал в коде.") followed by a
    /// clean, time-stamped list — composed ENTIRELY in code, no LLM. Reads like
    /// a human summary yet is identical and reliable on any model size, and
    /// instant. This is the direct answer to "что я делал / смотрел".
    pub async fn activity_summary(&self, n: usize) -> String {
        let all = self.collect_recent_activity().await;
        if all.is_empty() {
            return "Nothing to recall yet — I haven't recorded anything from the screen.".to_string();
        }
        let freshest_ts = all[0].0;
        let mut lines:  Vec<String> = Vec::new();
        let mut cats:   Vec<&str>   = Vec::new();
        let mut last_phrase = String::new();
        for (ts, app, title) in &all {
            let phrase = activity_line(app, title);
            if phrase == last_phrase { continue; } // collapse consecutive repeats
            last_phrase = phrase.clone();
            let cat = activity_category(app, title);
            if !cats.contains(&cat) { cats.push(cat); }
            lines.push(format!("• {} ({}) — {}", fmt_hm(*ts), rel_age(*ts), phrase));
            if lines.len() >= n { break; }
        }
        let acts  = cats.iter().take(2).cloned().collect::<Vec<_>>().join(" and ");
        let intro = format!("{} you {}. Timeline:", when_word(freshest_ts), acts);
        format!("{}\n{}", intro, lines.join("\n"))
    }

    /// Returns last `n` Q&A dialogue pairs, newest-first, as (timestamp, text) tuples.
    pub async fn recent_qa(&self, n: usize) -> Vec<(String, String)> {
        let mut stream = match self.unified.query().only_if("level = 0").execute().await {
            Ok(s)  => s,
            Err(_) => return vec![],
        };
        let mut all: Vec<(i64, String)> = Vec::new();

        while let Some(batch) = stream.next().await {
            let batch = match batch { Ok(b) => b, Err(_) => continue };
            let ts_arr  = match batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>() {
                Some(a) => a, None => continue,
            };
            let dat_arr = match batch.column(UM_DATA).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            for i in 0..batch.num_rows() {
                let data = dat_arr.value(i);
                if is_qa_event(data) {
                    all.push((ts_arr.value(i), data.to_string()));
                }
            }
        }

        all.sort_by(|a, b| b.0.cmp(&a.0));
        all.into_iter()
            .take(n)
            .map(|(ts, data)| (fmt_ts(ts), extract_text_from_json(&data)))
            .collect()
    }

    /// Counts how many Level-0 and Level-1 records match the query semantically.
    /// Embedding is LRU-cached, so a follow-up `collect()` call is cheap.
    pub async fn count_relevant(&self, query: &str) -> (usize, usize) {
        let emb = match self.embed(query).await { Ok(e) => e, Err(_) => return (0, 0) };
        let l0 = self.search_unified(emb.clone(), "level = 0",  5).await
            .map(|r| r.len()).unwrap_or(0);
        let l1 = self.search_unified(emb,          "level >= 1", 5).await
            .map(|r| r.len()).unwrap_or(0);
        (l0, l1)
    }

    /// Aggregates today's activity from unified_memory for the Analyst module.
    /// Returns a formatted string grouped by app, sorted by event count descending.
    pub async fn daily_activity_summary(&self) -> Result<String> {
        let today_start = chrono::Local::now()
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .unwrap_or_default();
        let start_ts = chrono::TimeZone::from_local_datetime(&chrono::Local, &today_start)
            .single()
            .map(|dt| dt.with_timezone(&Utc).timestamp_micros())
            .unwrap_or_else(|| Utc::now().timestamp_micros() - 86_400_000_000i64);

        // LanceDB cannot compare Int64 literal against Timestamp column directly.
        // Filter by level in SQL; apply the timestamp cutoff in Rust.
        let mut stream = match self.unified.query().only_if("level = 0").execute().await {
            Ok(s) => s,
            Err(e) => return Err(anyhow!("daily_activity_summary query failed: {}", e)),
        };

        let mut app_entries: HashMap<String, (usize, Vec<String>)> = HashMap::new();

        while let Some(batch) = stream.next().await {
            let batch = match batch { Ok(b) => b, Err(_) => continue };
            let ts_arr = match batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>() {
                Some(a) => a, None => continue,
            };
            let dat_arr = match batch.column(UM_DATA).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            for i in 0..batch.num_rows() {
                if ts_arr.value(i) < start_ts { continue; }
                let data = dat_arr.value(i);
                let v: Value = match serde_json::from_str(data) { Ok(v) => v, Err(_) => continue };
                let app  = v["app_name"].as_str().unwrap_or("unknown").to_string();
                let text = v["text"].as_str().unwrap_or("").to_string();
                if matches!(app.as_str(), "clipboard" | "system" | "analyst" | "user" | "dialogue") { continue; }
                let entry = app_entries.entry(app).or_insert((0, Vec::new()));
                entry.0 += 1;
                if entry.1.len() < 2 && text.len() > 10 {
                    entry.1.push(text.chars().take(80).collect());
                }
            }
        }

        if app_entries.is_empty() { return Ok(String::new()); }

        let mut sorted: Vec<_> = app_entries.into_iter().collect();
        sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));

        let lines: Vec<String> = sorted.into_iter().map(|(app, (count, snippets))| {
            let snip = snippets.first().map(|s| format!(": \"{}\"", s)).unwrap_or_default();
            format!("- {} ({} events){}", app, count, snip)
        }).collect();

        Ok(lines.join("\n"))
    }

    /// Selective amnesia: deletes all L0 events from the last `minutes` minutes.
    /// Also clears the embedding cache so next query starts fresh.
    /// Returns the number of events erased.
    pub async fn forget_recent(&self, minutes: u64) -> Result<usize> {
        let cutoff = Utc::now().timestamp_micros() - (minutes as i64 * 60 * 1_000_000);
        let ids = self.collect_l0_ids_since(cutoff).await?;
        let count = ids.len();
        if !ids.is_empty() {
            self.delete_by_ids(&ids).await?;
            self.embed_cache.lock().await.clear();
            *self.last_fact.lock().await = None;
            info!("[Librarian/amnesia] Erased {} events (last {} min)", count, minutes);
        }
        Ok(count)
    }

    /// Erases ALL memory: every row in both tables, the embed cache, the last
    /// fact, the event counter, and the cold archive on disk. Used by the
    /// user-facing "Delete all my data" control. Irreversible by design.
    pub async fn wipe_all(&self) -> Result<()> {
        // `"true"` is an always-true predicate → delete every row.
        let _ = self.table.delete("true").await;
        self.unified.delete("true").await?;
        self.embed_cache.lock().await.clear();
        *self.last_fact.lock().await = None;
        self.event_count.store(0, Ordering::Relaxed);
        if let Ok(entries) = std::fs::read_dir(&self.archive_path) {
            for e in entries.flatten() {
                let _ = std::fs::remove_file(e.path());
            }
        }
        info!("[Librarian] ALL memory erased on user request");
        Ok(())
    }

    /// Deletes all records tagged as stress-test data.
    /// Matches by `app_name = "stress_long"` OR `is_stress = true` in data_json.
    /// Resets last_fact cache so normal operation resumes with a clean slate.
    pub async fn delete_stress_records(&self) -> Result<usize> {
        // Legacy context table: direct SQL filter
        let _ = self.table.delete("app_name = 'stress_long'").await;

        // unified_memory: scan all L0 rows and filter by JSON content
        let mut stream = self.unified.query().only_if("level = 0").execute().await?;
        let mut ids    = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch   = match batch { Ok(b) => b, Err(_) => continue };
            let id_arr  = match batch.column(UM_ID).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            let dat_arr = match batch.column(UM_DATA).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            for i in 0..batch.num_rows() {
                let data = dat_arr.value(i);
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    let stress_app = v.get("app_name").and_then(Value::as_str) == Some("stress_long");
                    let stress_tag = v.get("is_stress").and_then(Value::as_bool).unwrap_or(false);
                    if stress_app || stress_tag {
                        ids.push(id_arr.value(i).to_string());
                    }
                }
            }
        }

        let count = ids.len();
        if !ids.is_empty() {
            self.delete_by_ids(&ids).await?;
            *self.last_fact.lock().await = None;
            self.embed_cache.lock().await.clear();
            info!("[Librarian] Deleted {} stress test records — DB sterile", count);
        }
        Ok(count)
    }

    async fn collect_l0_ids_since(&self, cutoff_us: i64) -> Result<Vec<String>> {
        let mut stream = self.unified.query().only_if("level = 0").execute().await?;
        let mut ids = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch  = match batch { Ok(b) => b, Err(_) => continue };
            let ts_arr = match batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>() {
                Some(a) => a, None => continue,
            };
            let id_arr = match batch.column(UM_ID).as_any().downcast_ref::<StringArray>() {
                Some(a) => a, None => continue,
            };
            for i in 0..batch.num_rows() {
                if ts_arr.value(i) >= cutoff_us {
                    ids.push(id_arr.value(i).to_string());
                }
            }
        }
        Ok(ids)
    }

    async fn delete_by_ids(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() { return Ok(()); }
        let id_list = ids.iter().map(|id| format!("'{}'", id)).collect::<Vec<_>>().join(", ");
        self.unified.delete(&format!("id IN ({})", id_list)).await?;
        Ok(())
    }

    /// Returns (db_bytes, archive_bytes) from the in-memory cache — O(1), no filesystem walk.
    pub fn disk_usage(&self) -> (u64, u64) {
        (
            self.cached_db_bytes.load(Ordering::Relaxed),
            self.cached_archive_bytes.load(Ordering::Relaxed),
        )
    }

    /// Pure calculation: estimates storage for `n` level-0 OCR records and
    /// projects weekly disk usage after daily L0→L1 compression.
    pub fn stress_test_report(n: usize, avg_ocr_chars: usize, interval_secs: u64) -> String {
        const EMBEDDING_BYTES: usize = 384 * 4;
        const JSON_OVERHEAD:   usize = 200;
        const LANCEDB_META:    usize = 200;

        let per_record = avg_ocr_chars + EMBEDDING_BYTES + JSON_OVERHEAD + LANCEDB_META;

        // Theoretical captures in 8 active hours; dirty-rect + pHash filter passes ~20 %.
        let theoretical_per_day = (8 * 3600) / (interval_secs as usize).max(1);
        let effective_per_day   = (theoretical_per_day / 5).max(1);
        let effective_per_hour  = (effective_per_day / 8).max(1);

        let daily_raw_bytes   = effective_per_day * per_record;
        let l1_per_hour_bytes = effective_per_hour * (avg_ocr_chars + 100); // sub_events JSON
        let daily_l1_bytes    = 8 * l1_per_hour_bytes;

        let weekly_archive = 7 * daily_raw_bytes;
        let weekly_l1      = 7 * daily_l1_bytes;
        let weekly_total   = weekly_archive + weekly_l1;

        fn fmt_bytes(b: usize) -> String {
            if b >= 1_048_576 { format!("{:.1} MB", b as f64 / 1_048_576.0) }
            else if b >= 1024 { format!("{:.1} KB", b as f64 / 1024.0) }
            else              { format!("{} B", b) }
        }

        format!(
            "[STRESS_TEST] Оценка хранилища:\n\
             ─────────────────────────────────────────\n\
             avg OCR текст:            {} chars\n\
             embedding (BERT-384):     {} bytes\n\
             JSON + LanceDB overhead:  {} bytes\n\
             Итого на запись:          {} байт (~{})\n\
             \n\
             Для {} записей (сырые L0): {}\n\
             \n\
             Реальные условия (20%% кадров проходят фильтры):\n\
             Теоретически/день:        {} снимков\n\
             Фактически/день:          {} снимков\n\
             L0 в день (до сжатия):    {}\n\
             L1 sub_events в день:     {} (8 hourly records)\n\
             Archive .ndjson/день:     {}\n\
             \n\
             За 7 дней:\n\
             Archive .ndjson:          {}\n\
             Level-1 в DB:             {}\n\
             После L1→L2 сжатия DB:   < 1 MB (topic nodes)\n\
             Итого на диске:           ~{}\n\
             ─────────────────────────────────────────\n\
             Вывод: «Детальная память» не съест диск.",
            avg_ocr_chars, EMBEDDING_BYTES, JSON_OVERHEAD + LANCEDB_META,
            per_record, fmt_bytes(per_record),
            n, fmt_bytes(n * per_record),
            theoretical_per_day, effective_per_day,
            fmt_bytes(daily_raw_bytes),
            fmt_bytes(daily_l1_bytes),
            fmt_bytes(daily_raw_bytes),
            fmt_bytes(weekly_archive),
            fmt_bytes(weekly_l1),
            fmt_bytes(weekly_total),
        )
    }

    // ── unified_memory write ──────────────────────────────────────────────────

    /// Stores a new L0 fact with Chronos-Linking deduplication.
    ///
    /// Returns `true` on a fresh insert, `false` when an existing fact was
    /// extended (Chronos-Link fired).  Callers use this to decide whether to
    /// write a parallel legacy record.
    async fn store_unified(
        &self,
        app_name:     &str,
        window_title: &str,
        text:         &str,
        ts:           i64,
        emb:          Vec<f32>,
    ) -> Result<bool> {
        let tags      = auto_tag(text, app_name, window_title);
        let mut guard = self.last_fact.lock().await;

        if let Some(ref last) = *guard {
            if last.app_name     == app_name
                && last.window_title == window_title
                && cosine_similarity(&last.emb, &emb) >= CHRONOS_SIMILARITY_THRESHOLD
            {
                // Extend existing fact: append delta, widen time span.
                let mut data: serde_json::Value = serde_json::from_str(&last.data_json)
                    .unwrap_or_else(|_| json!({
                        "app_name": app_name, "window_title": window_title,
                        "text": text, "tags": tags, "deltas": [],
                    }));

                let mut deltas = data["deltas"].as_array().cloned().unwrap_or_default();
                deltas.push(json!({ "t": ts, "text": text }));
                data["deltas"] = serde_json::Value::Array(deltas);

                let new_id   = Uuid::new_v4().to_string();
                let new_data = data.to_string();

                let _ = self.unified.delete(&format!("id = '{}'", last.id)).await;
                self.insert_unified_record(
                    &new_id, last.time_start, ts, 0, false, &new_data, emb.clone(),
                ).await?;

                info!("[LIBRARIAN/Chronos] Extended fact «{}» (sim ≥ {:.0}%)",
                      app_name, CHRONOS_SIMILARITY_THRESHOLD * 100.0);

                *guard = Some(LastFact {
                    id:           new_id,
                    emb,
                    app_name:     app_name.to_string(),
                    window_title: window_title.to_string(),
                    time_start:   last.time_start,
                    data_json:    new_data,
                });
                return Ok(false);
            }
        }

        // Fresh fact: insert new record and prime the cache.
        let new_id    = Uuid::new_v4().to_string();
        let is_stress = crate::perf::is_stress_mode();
        let data_json = json!({
            "app_name":     app_name,
            "window_title": window_title,
            "text":         text,
            "tags":         tags,
            "deltas":       [],
            "is_stress":    is_stress,
        }).to_string();

        self.insert_unified_record(
            &new_id, ts, ts, 0, false, &data_json, emb.clone(),
        ).await?;

        *guard = Some(LastFact {
            id:           new_id,
            emb,
            app_name:     app_name.to_string(),
            window_title: window_title.to_string(),
            time_start:   ts,
            data_json,
        });
        Ok(true)
    }

    /// Inserts any level record into unified_memory with an explicit `id`.
    async fn insert_unified_record(
        &self,
        id:         &str,
        time_start: i64,
        time_end:   i64,
        level:      i32,
        archived:   bool,
        data_json:  &str,
        emb:        Vec<f32>,
    ) -> Result<()> {
        let id     = id.to_string();
        let schema = make_unified_schema();
        let values = Arc::new(Float32Array::from(emb)) as ArrayRef;
        let embedding = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            384, values, None,
        );
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec![id.as_str()]))              as ArrayRef,
            Arc::new(TimestampMicrosecondArray::from(vec![time_start])) as ArrayRef,
            Arc::new(TimestampMicrosecondArray::from(vec![time_end]))   as ArrayRef,
            Arc::new(Int32Array::from(vec![level]))                     as ArrayRef,
            Arc::new(BooleanArray::from(vec![archived]))                as ArrayRef,
            Arc::new(StringArray::from(vec![data_json]))                as ArrayRef,
            Arc::new(embedding)                                         as ArrayRef,
        ])?;
        self.unified.add(vec![batch]).execute().await?;
        Ok(())
    }

    // ── Read API ──────────────────────────────────────────────────────────────

    /// Basic vector search (legacy table) — returns up to `limit` summaries.
    pub async fn find_relevant(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let emb = self.embed(query).await?;
        let mut stream = self.table
            .vector_search(emb)?
            .column("vector")
            .limit(limit)
            .execute()
            .await?;

        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            if let Some(arr) = batch.column(COL_SUMMARY).as_any().downcast_ref::<StringArray>() {
                for i in 0..arr.len() { out.push(arr.value(i).to_string()); }
            }
        }
        Ok(out)
    }

    // ── Smart RAG ─────────────────────────────────────────────────────────────

    /// Searches unified_memory with level hierarchy:
    ///   1. Level ≥ 1 (summaries) — fast topic navigation.
    ///   2. Level = 0 (raw events) — rich recent detail.
    pub async fn smart_rag(&self, query: &str) -> Result<String> {
        let emb = self.embed(query).await?;

        let want_screen = is_screen_query(query);

        let summary_rows = self.search_unified(emb.clone(), "level >= 1", 5).await?;
        // Fetch 8 rows, filter noise, keep top 5 — precise context, fewer tokens.
        let raw_rows_all = self.search_unified(emb, "level = 0", 8).await?;

        // Drop user queries and system responses — they pollute RAG context.
        // The query itself is stored before RAG runs, so without this filter
        // the user's own question becomes its own #1 semantic result.
        //
        // For GENERAL (non-screen) questions also drop raw screen-capture events
        // (real app names), keeping only dialogue/notes — otherwise a question
        // like "какой сайт самый быстрый" pulls in screen OCR and the small model
        // parrots the latest screen line instead of answering.
        let raw_rows: Vec<UnifiedRow> = raw_rows_all.into_iter()
            .filter(|r| {
                let app = serde_json::from_str::<Value>(&r.data_json).ok()
                    .and_then(|v| v.get("app_name").and_then(|a| a.as_str()).map(String::from))
                    .unwrap_or_default();
                if app == "user" || app == "system" { return false; }
                if !want_screen && app != "dialogue" && app != "note" { return false; }
                true
            })
            .collect();

        let mut out = String::new();

        // Include screen activity only when the query is about on-screen content.
        // Always inserting VS Code / terminal OCR confuses the model for general questions.
        //
        // For activity questions ("what did I do an hour ago") a single snapshot
        // isn't enough — semantic vector search doesn't match a time-based question
        // against OCR text, which is why these used to return "no activity". So we
        // surface a short timeline of the most recent screen facts by time.
        if want_screen {
            // Deterministic "iron log" — clean human lines from app + window title,
            // not raw OCR. Reliable on any model and reads naturally to the user.
            let log = self.iron_log(8).await;
            if !log.is_empty() {
                out.push_str("Screen activity (newest first):\n");
                for line in &log {
                    out.push_str(&format!("- {line}\n"));
                }
                out.push('\n');
            } else if let Some((ts, text)) = self.last_screen_snapshot().await {
                out.push_str(&format!("[CURRENT_SCREEN {}]\n{}\n\n", ts, text));
            }
        }

        if summary_rows.is_empty() && raw_rows.is_empty() {
            if out.is_empty() {
                return Ok("[Librarian] Context is empty — no relevant events.".to_string());
            }
            return Ok(out.chars().take(2500).collect());
        }

        if !summary_rows.is_empty() {
            // Sort newest-first so the model never reads an older summary as "the
            // last thing" (vector search returns them in similarity order).
            let mut summary_rows = summary_rows;
            summary_rows.sort_by(|a, b| b.time_start.cmp(&a.time_start));
            out.push_str("Summaries (newest first):\n");
            for row in &summary_rows {
                out.push_str(&format!("[{} · {}] {}\n\n",
                    fmt_ts(row.time_start), rel_age(row.time_start),
                    extract_text_preview(&row.data_json)));
            }
        }

        if !raw_rows.is_empty() {
            out.push_str("\n## Similar events\n");
            let recent_n = 5.min(raw_rows.len());
            for row in &raw_rows[..recent_n] {
                let ts       = fmt_ts(row.time_start);
                let age_secs = (Utc::now().timestamp_micros() - row.time_start) / 1_000_000;
                let age_tag  = if age_secs < 3600 {
                    format!("{} min ago", age_secs / 60)
                } else {
                    format!("{} h ago", age_secs / 3600)
                };
                let app = extract_app_name_from_json(&row.data_json);
                let app_tag = if !app.is_empty() { format!(" [{}]", app) } else { String::new() };
                out.push_str(&format!("[{ts}]{app_tag} ({age_tag})\n{}\n\n",
                    summarize_code(&extract_text_from_json(&row.data_json))));
            }
        }

        Ok(out.chars().take(2500).collect())
    }

    /// Internal: vector search on unified_memory with a SQL-style level filter.
    async fn search_unified(
        &self,
        emb:    Vec<f32>,
        filter: &str,
        limit:  usize,
    ) -> Result<Vec<UnifiedRow>> {
        let mut stream = self.unified
            .vector_search(emb)?
            .column("embedding")
            .only_if(filter)
            .limit(limit)
            .execute()
            .await?;

        let mut rows = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            // Graceful downcast — skip malformed batches instead of panicking
            let id_arr  = batch.column(UM_ID).as_any().downcast_ref::<StringArray>();
            let ts_arr  = batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let te_arr  = batch.column(UM_TIME_END).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let lv_arr  = batch.column(UM_LEVEL).as_any().downcast_ref::<Int32Array>();
            let dat_arr = batch.column(UM_DATA).as_any().downcast_ref::<StringArray>();

            if let (Some(ids), Some(ts), Some(te), Some(lv), Some(dat)) =
                (id_arr, ts_arr, te_arr, lv_arr, dat_arr)
            {
                for i in 0..batch.num_rows() {
                    rows.push(UnifiedRow {
                        id:         ids.value(i).to_string(),
                        time_start: ts.value(i),
                        time_end:   te.value(i),
                        level:      lv.value(i),
                        data_json:  dat.value(i).to_string(),
                    });
                }
            }
        }

        rows.sort_by(|a, b| b.time_start.cmp(&a.time_start));
        Ok(rows)
    }

    // ── Archive search (cold storage) ─────────────────────────────────────────

    /// Keyword search in NDJSON cold-storage archive files.
    pub async fn search_archive(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let query_lc  = query.to_lowercase();
        let dir       = &self.archive_path;
        let cipher    = self.cipher.clone();

        if !dir.exists() {
            return Ok(vec!["[Archive] Cold storage is empty.".to_string()]);
        }

        let dir_clone = dir.clone();
        let results = tokio::task::spawn_blocking(move || -> Result<Vec<(i64, String)>> {
            let mut matches: Vec<(i64, String)> = Vec::new();

            for entry in std::fs::read_dir(&dir_clone)?.flatten() {
                let path     = entry.path();
                let path_str = path.to_string_lossy();
                let is_zst   = path_str.ends_with(".ndjson.zst");
                let is_plain = path_str.ends_with(".ndjson") && !is_zst;
                if !is_zst && !is_plain { continue; }

                let content = if is_zst {
                    match std::fs::read(&path).and_then(|b| {
                        zstd::decode_all(&b[..])
                    }) {
                        Ok(b)  => match String::from_utf8(b) {
                            Ok(s)  => s,
                            Err(e) => { warn!("[Archive] UTF-8 {:?}: {}", path, e); continue; }
                        },
                        Err(e) => { warn!("[Archive] zstd decode {:?}: {}", path, e); continue; }
                    }
                } else {
                    match std::fs::read_to_string(&path) {
                        Ok(c)  => c,
                        Err(e) => { warn!("[Archive] read {:?}: {}", path, e); continue; }
                    }
                };

                for raw_line in content.lines() {
                    // Decrypt if cipher is set
                    let line = if let Some(ref c) = cipher {
                        match c.decrypt(raw_line) {
                            Ok(p)  => p,
                            Err(e) => { warn!("[Archive] decrypt: {}", e); raw_line.to_string() }
                        }
                    } else {
                        raw_line.to_string()
                    };

                    if let Ok(v) = serde_json::from_str::<Value>(&line) {
                        let data    = v.get("data_json").and_then(|d| d.as_str()).unwrap_or(&line);
                        let preview = extract_text_preview(data);
                        if preview.to_lowercase().contains(&query_lc) {
                            let ts = v.get("time_start").and_then(|t| t.as_i64()).unwrap_or(0);
                            matches.push((ts, preview));
                        }
                    }
                }
            }

            matches.sort_by(|a, b| b.0.cmp(&a.0));
            Ok(matches)
        }).await??;

        let out: Vec<String> = results.into_iter()
            .take(limit)
            .map(|(ts, text)| format!("[{}] {}", fmt_ts(ts), text))
            .collect();

        if out.is_empty() {
            Ok(vec![format!("[Archive] Nothing found for query: {}", query)])
        } else {
            Ok(out)
        }
    }

    /// Filtered semantic search: narrow results by time window and/or app name.
    /// `time_from` / `time_to` are Unix seconds (inclusive).
    /// `app_name` matches as a case-insensitive substring against the stored app name.
    pub async fn search_by_filter(
        &self,
        query:      &str,
        time_from:  Option<i64>,
        time_to:    Option<i64>,
        app_filter: Option<&str>,
        limit:      usize,
    ) -> Result<Vec<String>> {
        let emb  = self.embed(query).await?;
        let rows = self.search_unified(emb, "level = 0", limit * 4).await?;

        let app_lc: Option<String> = app_filter.map(|s| s.to_lowercase());

        let filtered: Vec<String> = rows.into_iter()
            .filter(|r| {
                let ts_secs = r.time_start / 1_000_000;
                if let Some(from) = time_from { if ts_secs < from { return false; } }
                if let Some(to)   = time_to   { if ts_secs > to   { return false; } }
                if let Some(ref app) = app_lc {
                    let row_app = extract_app_name_from_json(&r.data_json).to_lowercase();
                    if !row_app.contains(app.as_str()) { return false; }
                }
                true
            })
            .take(limit)
            .map(|r| {
                let ts  = fmt_ts(r.time_start);
                let app = extract_app_name_from_json(&r.data_json);
                let txt = extract_text_from_json(&r.data_json);
                format!("[{}] [{}] {}", ts, app, txt.chars().take(200).collect::<String>())
            })
            .collect();

        Ok(filtered)
    }

    /// Returns the total number of level-0 (raw) events stored in unified_memory.
    /// Reads from an in-memory AtomicU64 — O(1), no LanceDB access.
    pub async fn count_events(&self) -> u64 {
        self.event_count.load(Ordering::Relaxed)
    }

    /// Returns hourly event counts for the last `hours` hours, oldest-first.
    /// Each element is (unix_seconds_of_hour_start, event_count).
    pub async fn activity_hourly(&self, hours: u32) -> Result<Vec<(i64, u64)>> {
        let now_us  = Utc::now().timestamp_micros();
        let from_us = now_us - (hours as i64) * 3_600_000_000;

        let mut stream = self.unified.query().only_if("level = 0").execute().await?;
        let mut hour_counts: HashMap<i64, u64> = HashMap::new();

        while let Some(batch) = stream.next().await {
            let batch = match batch { Ok(b) => b, Err(_) => continue };
            let ts_col = batch.column(UM_TIME_START)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>();
            if let Some(ts_arr) = ts_col {
                for i in 0..ts_arr.len() {
                    if !ts_arr.is_valid(i) { continue; }
                    let ts_us = ts_arr.value(i);
                    if ts_us < from_us { continue; }
                    let hour_sec = (ts_us / 3_600_000_000) * 3600;
                    *hour_counts.entry(hour_sec).or_insert(0) += 1;
                }
            }
        }

        let hour_now_sec = (Utc::now().timestamp() / 3600) * 3600;
        let mut result = Vec::with_capacity(hours as usize);
        for h in 0..(hours as i64) {
            let hour_ts = hour_now_sec - (hours as i64 - 1 - h) * 3600;
            result.push((hour_ts, *hour_counts.get(&hour_ts).unwrap_or(&0)));
        }
        Ok(result)
    }

    /// Returns up to `limit` most-recent level-0 events as NDJSON lines,
    /// sorted newest-first.  Each line is a self-contained JSON object.
    pub async fn export_recent(&self, limit: usize) -> Result<Vec<String>> {
        let mut stream = self.unified.query().only_if("level = 0").execute().await?;
        let mut rows: Vec<(i64, String)> = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = match batch { Ok(b) => b, Err(_) => continue };
            let id_arr  = batch.column(UM_ID).as_any().downcast_ref::<StringArray>();
            let ts_arr  = batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let te_arr  = batch.column(UM_TIME_END).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let dat_arr = batch.column(UM_DATA).as_any().downcast_ref::<StringArray>();
            if let (Some(ids), Some(tss), Some(tes), Some(dats)) = (id_arr, ts_arr, te_arr, dat_arr) {
                for i in 0..batch.num_rows() {
                    let ts   = tss.value(i);
                    let line = json!({
                        "id":           ids.value(i),
                        "time_start":   fmt_ts(ts),
                        "time_end":     fmt_ts(tes.value(i)),
                        "timestamp_us": ts,
                        "data":         serde_json::from_str::<Value>(dats.value(i)).unwrap_or_default(),
                    });
                    rows.push((ts, line.to_string()));
                }
            }
        }
        rows.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        rows.truncate(limit);
        Ok(rows.into_iter().map(|(_, s)| s).collect())
    }

    // ── Topic Timeline RAG ────────────────────────────────────────────────────

    /// ICT context builder — tries smart_rag first, falls back to legacy table.
    pub async fn topic_timeline_rag(&self, query: &str) -> Result<String> {
        LAST_QUERY_TS.store(Utc::now().timestamp(), Ordering::Relaxed);
        let smart = self.smart_rag(query).await?;
        if !smart.contains("Context is empty") {
            return Ok(smart);
        }
        // The legacy fallback surfaces recent SCREEN events by similarity — useful
        // only for screen/activity questions. For a general question, return the
        // empty context so the model answers from its own knowledge instead of
        // parroting the latest screen line.
        if !is_screen_query(query) {
            return Ok(smart);
        }
        self.legacy_topic_timeline_rag(query).await
    }

    async fn legacy_topic_timeline_rag(&self, query: &str) -> Result<String> {
        let emb = self.embed(query).await?;
        let mut stream = self.table
            .vector_search(emb)?
            .column("vector")
            .limit(10)
            .execute()
            .await?;

        let mut rows: Vec<EventRow> = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch   = batch?;
            let ts_arr  = batch.column(COL_TS).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let app_arr = batch.column(COL_APP).as_any().downcast_ref::<StringArray>();
            let win_arr = batch.column(COL_WIN).as_any().downcast_ref::<StringArray>();
            let txt_arr = batch.column(COL_SUMMARY).as_any().downcast_ref::<StringArray>();

            if let (Some(ts), Some(app), Some(win), Some(txt)) = (ts_arr, app_arr, win_arr, txt_arr) {
                for i in 0..batch.num_rows() {
                    rows.push(EventRow {
                        timestamp:    ts.value(i),
                        app_name:     app.value(i).to_string(),
                        window_title: win.value(i).to_string(),
                        text_summary: txt.value(i).to_string(),
                    });
                }
            }
        }

        if rows.is_empty() {
            return Ok("[Librarian] Context is empty — no relevant events.".to_string());
        }

        rows.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        let recent_n = 3.min(rows.len());
        let (recent, historical) = rows.split_at(recent_n);

        let recent_block = recent.iter().map(|r| {
            format!("[{}] {} / {}\n{}",
                fmt_ts(r.timestamp), r.app_name, r.window_title,
                summarize_code(&r.text_summary))
        }).collect::<Vec<_>>().join("\n\n");

        let hist_block = if historical.is_empty() {
            String::new()
        } else {
            let events_text = historical.iter()
                .map(|r| format!("{} ({}): {}",
                    r.app_name, r.window_title,
                    r.text_summary.chars().take(120).collect::<String>()))
                .collect::<Vec<_>>().join("\n");

            let summary = self.summarize_events(&events_text).await
                .unwrap_or_else(|_| events_text.chars().take(300).collect());

            format!("\n\n## Historical context (LLM-compressed)\n{}", summary)
        };

        let raw = format!("## Recent events\n{}{}", recent_block, hist_block);
        Ok(raw.chars().take(2500).collect())
    }

    // ── Daily compression ─────────────────────────────────────────────────────

    /// Groups all level-0 records into 1-hour buckets, writes a level-1 summary
    /// for each, archives originals to disk, then deletes them from the hot table.
    async fn compress_daily(&self) -> Result<()> {
        info!("[Librarian/compress] Starting daily compression…");

        let mut stream = self.unified.query().only_if("level = 0").execute().await?;
        let mut rows: Vec<UnifiedRow> = Vec::new();

        while let Some(batch) = stream.next().await {
            let batch   = batch?;
            let id_arr  = batch.column(UM_ID).as_any().downcast_ref::<StringArray>();
            let ts_arr  = batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let te_arr  = batch.column(UM_TIME_END).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let lv_arr  = batch.column(UM_LEVEL).as_any().downcast_ref::<Int32Array>();
            let dat_arr = batch.column(UM_DATA).as_any().downcast_ref::<StringArray>();

            if let (Some(ids), Some(ts), Some(te), Some(lv), Some(dat)) =
                (id_arr, ts_arr, te_arr, lv_arr, dat_arr)
            {
                for i in 0..batch.num_rows() {
                    rows.push(UnifiedRow {
                        id:         ids.value(i).to_string(),
                        time_start: ts.value(i),
                        time_end:   te.value(i),
                        level:      lv.value(i),
                        data_json:  dat.value(i).to_string(),
                    });
                }
            }
        }

        if rows.is_empty() {
            info!("[Librarian/compress] Nothing to compress.");
            return Ok(());
        }

        let groups = group_by_hour(rows);
        info!("[Librarian/compress] {} groups to process", groups.len());

        let today            = Utc::now().format("%Y-%m-%d").to_string();
        let mut archived_ids = Vec::new();

        for group in &groups {
            if group.is_empty() { continue; }

            // Build structured sub_events — no LLM summary, full fidelity.
            let sub_events: Vec<Value> = group.iter().map(|r| {
                let parsed: Value = serde_json::from_str(&r.data_json).unwrap_or(Value::Null);
                let app  = parsed.get("app_name").and_then(Value::as_str).unwrap_or("screen");
                let text = parsed.get("text").and_then(Value::as_str).unwrap_or("");
                if app == "clipboard" {
                    json!({ "t": r.time_start, "act": "clipboard", "ocr_full": "", "clipboard": text })
                } else {
                    json!({ "t": r.time_start, "act": app, "ocr_full": text, "clipboard": "" })
                }
            }).collect();

            let group_start = group.iter().map(|r| r.time_start).min().unwrap_or(0);
            let group_end   = group.iter().map(|r| r.time_end).max().unwrap_or(group_start);

            // Embed from OCR content (first 2000 chars covers BERT-384 token budget).
            let embed_text: String = sub_events.iter()
                .filter_map(|e| {
                    e.get("ocr_full").and_then(Value::as_str)
                        .filter(|t| !t.is_empty())
                        .or_else(|| e.get("clipboard").and_then(Value::as_str).filter(|t| !t.is_empty()))
                })
                .collect::<Vec<_>>()
                .join("\n")
                .chars()
                .take(2000)
                .collect();

            if let Ok(emb) = self.embed_write(&embed_text).await {
                let level1_json = json!({
                    "sub_events": sub_events,
                    "count":      group.len(),
                }).to_string();
                let _ = self.insert_unified_record(
                    &Uuid::new_v4().to_string(),
                    group_start, group_end, 1, false, &level1_json, emb,
                ).await;
            }

            if let Err(e) = self.archive_rows_to_disk(group, &today) {
                warn!("[Librarian/compress] archive_to_disk: {}", e);
            }

            archived_ids.extend(group.iter().map(|r| r.id.clone()));
        }

        if !archived_ids.is_empty() {
            let id_list = archived_ids.iter()
                .map(|id| format!("'{}'", id))
                .collect::<Vec<_>>()
                .join(", ");
            if let Err(e) = self.unified.delete(&format!("id IN ({})", id_list)).await {
                warn!("[Librarian/compress] delete: {}", e);
            }
        }

        info!("[Librarian/compress] Done. {} records archived.", archived_ids.len());
        Ok(())
    }

    // ── Level-1 → Level-2 topic node compression ─────────────────────────────

    async fn compress_level1_to_level2(&self) -> Result<()> {
        info!("[Librarian/L2] Starting level-1 → level-2 compression…");

        let mut stream = self.unified.query().only_if("level = 1").execute().await?;
        let mut rows: Vec<UnifiedRow> = Vec::new();

        while let Some(batch) = stream.next().await {
            let batch   = batch?;
            let id_arr  = batch.column(UM_ID).as_any().downcast_ref::<StringArray>();
            let ts_arr  = batch.column(UM_TIME_START).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let te_arr  = batch.column(UM_TIME_END).as_any().downcast_ref::<TimestampMicrosecondArray>();
            let lv_arr  = batch.column(UM_LEVEL).as_any().downcast_ref::<Int32Array>();
            let dat_arr = batch.column(UM_DATA).as_any().downcast_ref::<StringArray>();

            if let (Some(ids), Some(ts), Some(te), Some(lv), Some(dat)) =
                (id_arr, ts_arr, te_arr, lv_arr, dat_arr)
            {
                for i in 0..batch.num_rows() {
                    rows.push(UnifiedRow {
                        id:         ids.value(i).to_string(),
                        time_start: ts.value(i),
                        time_end:   te.value(i),
                        level:      lv.value(i),
                        data_json:  dat.value(i).to_string(),
                    });
                }
            }
        }

        if rows.len() < 5 {
            info!("[Librarian/L2] Not enough level-1 records ({}) for topic nodes.", rows.len());
            return Ok(());
        }

        let groups = group_by_week(rows);
        info!("[Librarian/L2] {} weekly groups", groups.len());

        let mut compressed_ids = Vec::new();
        let today = Utc::now().format("%Y-%m-%d").to_string();

        for group in &groups {
            if group.len() < 3 { continue; }

            let combined = group.iter().map(|r| extract_text_preview(&r.data_json))
                .collect::<Vec<_>>().join("\n");

            let node_summary = self.summarize_events(&combined).await
                .unwrap_or_else(|_| combined.chars().take(300).collect());

            let group_start = group.iter().map(|r| r.time_start).min().unwrap_or(0);
            let group_end   = group.iter().map(|r| r.time_end).max().unwrap_or(group_start);

            let embed_text = format!("Topic node: {}", node_summary);
            if let Ok(emb) = self.embed_write(&embed_text).await {
                let node_json = json!({
                    "summary":   node_summary,
                    "span_days": (group_end - group_start) / 86_400_000_000i64,
                    "count":     group.len(),
                }).to_string();
                let _ = self.insert_unified_record(
                    &Uuid::new_v4().to_string(),
                    group_start, group_end, 2, false, &node_json, emb,
                ).await;
            }

            // Archive L1 rows to permanent cold storage before removing from active DB.
            if let Err(e) = self.archive_rows_to_disk(group, &format!("L1_{}", today)) {
                warn!("[Librarian/L2] archive_to_disk L1: {}", e);
            }

            compressed_ids.extend(group.iter().map(|r| r.id.clone()));
        }

        if !compressed_ids.is_empty() {
            let id_list = compressed_ids.iter()
                .map(|id| format!("'{}'", id))
                .collect::<Vec<_>>()
                .join(", ");
            if let Err(e) = self.unified.delete(&format!("id IN ({})", id_list)).await {
                warn!("[Librarian/L2] delete: {}", e);
            }
        }

        info!("[Librarian/L2] Done. {} level-2 topic nodes created.", groups.len());
        Ok(())
    }

    // ── Mini-LLM summarizer ───────────────────────────────────────────────────

    async fn summarize_events(&self, text: &str) -> Result<String> {
        let prompt = format!(
            "Summarize these events in 2-3 sentences, highlighting the key topics and actions:\n\
             {}\n\nSummary:",
            text.chars().take(800).collect::<String>(),
        );
        let llm = self.summarizer.clone();
        tokio::task::spawn_blocking(move || {
            llm.lock()
                .map_err(|e| anyhow!("Summarizer lock poisoned: {}", e))?
                // Faithful compression of stored events → deterministic (temp 0).
                .generate(&prompt, 80, 0.0)
        })
        .await
        .map_err(|e| anyhow!("join: {}", e))?
    }

    // ── Archive write — zstd-compressed, optionally AES-encrypted ───────────
    //
    // Files are stored as `{date}.ndjson.zst` (zstd level-6 frame).
    // On each call the existing file is decompressed, new records appended, then
    // recompressed. This keeps the archive append-friendly without streaming zstd.
    // Legacy `.ndjson` files (uncompressed) are migrated transparently on first write.

    fn archive_rows_to_disk(&self, rows: &[UnifiedRow], date: &str) -> Result<()> {
        let zst_path    = self.archive_path.join(format!("{}.ndjson.zst", date));
        let legacy_path = self.archive_path.join(format!("{}.ndjson", date));

        // Read existing content: prefer .ndjson.zst, fall back to legacy .ndjson
        let mut content = if zst_path.exists() {
            let compressed = std::fs::read(&zst_path)?;
            let raw = zstd::decode_all(&compressed[..])
                .map_err(|e| anyhow!("zstd decode {}: {}", zst_path.display(), e))?;
            String::from_utf8(raw)
                .map_err(|e| anyhow!("archive UTF-8: {}", e))?
        } else if legacy_path.exists() {
            std::fs::read_to_string(&legacy_path)?
        } else {
            String::new()
        };

        // Append new records
        for row in rows {
            let line = json!({
                "id":         row.id,
                "time_start": row.time_start,
                "time_end":   row.time_end,
                "level":      row.level,
                "data_json":  row.data_json,
            }).to_string();

            let written = if let Some(ref c) = self.cipher {
                c.encrypt(&line).unwrap_or(line)
            } else {
                line
            };

            content.push_str(&written);
            content.push('\n');
        }

        // Compress at level 6 (good ratio, fast enough for batch writes)
        let compressed = zstd::encode_all(std::io::Cursor::new(content.as_bytes()), 6)
            .map_err(|e| anyhow!("zstd encode {}: {}", zst_path.display(), e))?;
        std::fs::write(&zst_path, compressed)?;

        // Remove legacy plain-text file after migration to avoid duplicate reads
        if legacy_path.exists() {
            let _ = std::fs::remove_file(&legacy_path);
        }

        Ok(())
    }

    // ── Background tasks ──────────────────────────────────────────────────────

    pub async fn start_background_tasks(self: Arc<Self>) {
        // Every 5 min: log app stats
        let s = self.clone();
        tokio::spawn(async move {
            let mut interval = time::interval(std::time::Duration::from_secs(300));
            loop {
                interval.tick().await;
                if let Err(e) = s.log_app_stats().await {
                    warn!("[Librarian/bg] stats: {}", e);
                }
                log_process_ram();
            }
        });

        // Daily at 03:00 UTC: compress level-0 → level-1
        let s = self.clone();
        tokio::spawn(async move {
            loop {
                let now      = Utc::now();
                let tomorrow = (now.date_naive() + Duration::days(1))
                    .and_hms_opt(3, 0, 0)
                    .map(|dt| dt.and_utc())
                    .unwrap_or_else(|| now + Duration::hours(24));
                let wait = (tomorrow - now).to_std()
                    .unwrap_or(std::time::Duration::from_secs(3600));
                tokio::time::sleep(wait).await;
                if let Err(e) = s.compress_daily().await {
                    warn!("[Librarian/compress] {}", e);
                }
                if let Ok(report) = s.full_analysis().await {
                    println!("{}", report);
                }
            }
        });

        // Deep Miner: triggers compress_daily after 30 min of user inactivity.
        let s = self.clone();
        tokio::spawn(async move {
            const IDLE_SECS: i64 = 1800;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(IDLE_SECS as u64)).await;
                let last = LAST_QUERY_TS.load(Ordering::Relaxed);
                if last == 0 { continue; }
                let idle = Utc::now().timestamp() - last;
                if idle >= IDLE_SECS {
                    info!("[Deep Miner] Idle {}s — starting background mining", idle);
                    if let Err(e) = s.compress_daily().await {
                        warn!("[Deep Miner] compress: {}", e);
                    }
                    LAST_QUERY_TS.store(0, Ordering::Relaxed);
                }
            }
        });

        // Weekly: compress level-1 → level-2 topic nodes
        let s = self.clone();
        tokio::spawn(async move {
            let week = std::time::Duration::from_secs(7 * 24 * 3600);
            loop {
                tokio::time::sleep(week).await;
                if let Err(e) = s.compress_level1_to_level2().await {
                    warn!("[Librarian/L2] {}", e);
                }
            }
        });

        // TTL pruning intentionally removed: L0 data is NEVER auto-deleted.
        // The only path out of LanceDB for L0 is compress_daily() which archives
        // to disk first. Manual forget_recent() exists for user-initiated amnesia.
    }

    async fn app_event_counts(&self) -> Result<HashMap<String, usize>> {
        let mut stream = self.table.query().execute().await?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            if let Some(arr) = batch.column(COL_APP).as_any().downcast_ref::<StringArray>() {
                for i in 0..arr.len() {
                    *counts.entry(arr.value(i).to_string()).or_default() += 1;
                }
            }
        }
        Ok(counts)
    }

    async fn log_app_stats(&self) -> Result<()> {
        let counts = self.app_event_counts().await?;
        info!("[Librarian] app stats: {:?}", counts);
        Ok(())
    }

    async fn full_analysis(&self) -> Result<String> {
        let counts = self.app_event_counts().await?;
        let mut report = "[Librarian] App event counts:\n".to_string();
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (app, n) in sorted {
            report.push_str(&format!("  {}: {}\n", app, n));
        }
        Ok(report)
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

fn group_by_hour(rows: Vec<UnifiedRow>) -> Vec<Vec<UnifiedRow>> {
    let mut buckets: HashMap<i64, Vec<UnifiedRow>> = HashMap::new();
    for row in rows {
        let hour_key = row.time_start / 3_600_000_000;
        buckets.entry(hour_key).or_default().push(row);
    }
    let mut groups: Vec<Vec<UnifiedRow>> = buckets.into_values().collect();
    for g in &mut groups { g.sort_by_key(|r| r.time_start); }
    groups.sort_by_key(|g| g.first().map(|r| r.time_start).unwrap_or(0));
    groups
}

fn group_by_week(rows: Vec<UnifiedRow>) -> Vec<Vec<UnifiedRow>> {
    let mut buckets: HashMap<i64, Vec<UnifiedRow>> = HashMap::new();
    for row in rows {
        let week_key = row.time_start / (7 * 24 * 3_600_000_000i64);
        buckets.entry(week_key).or_default().push(row);
    }
    let mut groups: Vec<Vec<UnifiedRow>> = buckets.into_values().collect();
    for g in &mut groups { g.sort_by_key(|r| r.time_start); }
    groups.sort_by_key(|g| g.first().map(|r| r.time_start).unwrap_or(0));
    groups
}

/// Moves a corrupt memory-DB directory aside (never deletes) so boot can start
/// clean. Falls back to removal only if the rename itself fails, so a fresh
/// start is still possible. Best-effort: logs and returns either way.
/// Scores each candidate service by how often (recency-weighted) its aliases
/// appear in recent activity; returns the top one. `exact_only` restricts to
/// services of exactly `want` kind (so a dedicated audio/video app beats the
/// versatile YouTube); pass `false` for the YouTube-inclusive fallback pass.
fn top_service(
    acts: &[(i64, String, String)],
    want: crate::services::Kind,
    exact_only: bool,
) -> Option<&'static crate::services::Service> {
    let mut best: Option<(&'static crate::services::Service, f64)> = None;
    for svc in crate::services::SERVICES {
        let ok = if exact_only { svc.kind == want } else { crate::services::kind_matches(svc.kind, want) };
        if !ok { continue; }
        let mut score = 0.0f64;
        for (rank, (_, app, title)) in acts.iter().take(300).enumerate() {
            let hay = format!("{} {}", app, title).to_lowercase();
            if svc.aliases.iter().any(|a| hay.contains(a)) {
                score += 1.0 / (1.0 + rank as f64 * 0.03); // newer events weigh more
            }
        }
        if score > 0.0 && best.map_or(true, |(_, b)| score > b) {
            best = Some((svc, score));
        }
    }
    best.map(|(s, _)| s)
}

fn quarantine_broken_db(db_path: &std::path::Path) {
    if !db_path.exists() { return; }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = db_path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "lancedb".to_string());
    let broken = db_path.with_file_name(format!("{name}.broken-{stamp}"));
    match std::fs::rename(db_path, &broken) {
        Ok(_) => tracing::warn!(
            "[Librarian] corrupt memory DB moved to {} (safe to delete)", broken.display()),
        Err(e) => {
            tracing::error!("[Librarian] could not quarantine corrupt DB ({e}); removing to recover");
            let _ = std::fs::remove_dir_all(db_path);
        }
    }
}

fn dir_size_bytes(path: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else { return 0 };
    entries.flatten().fold(0u64, |acc, e| {
        let p = e.path();
        if p.is_dir() { acc + dir_size_bytes(&p) }
        else { acc + std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0) }
    })
}

fn fmt_ts(ts: i64) -> String {
    DateTime::from_timestamp_micros(ts)
        .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| "?".to_string())
}

/// Local clock time "HH:MM" for a microsecond timestamp (for the iron log).
fn fmt_hm(ts_micros: i64) -> String {
    DateTime::from_timestamp_micros(ts_micros)
        .map(|d| d.with_timezone(&chrono::Local).format("%H:%M").to_string())
        .unwrap_or_default()
}

/// Human relative age of a microsecond timestamp ("just now", "5 h ago",
/// "yesterday", "3 d ago"). Lets the model — and the user — tell fresh activity
/// from stale, so old events aren't reported as if they were happening now.
fn rel_age(ts_micros: i64) -> String {
    let secs = (Utc::now().timestamp_micros() - ts_micros) / 1_000_000;
    if secs < 90              { "just now".to_string() }
    else if secs < 3600       { format!("{} min ago", secs / 60) }
    else if secs < 86_400     { format!("{} h ago", secs / 3600) }
    else if secs < 172_800    { "yesterday".to_string() }
    else                      { format!("{} d ago", secs / 86_400) }
}

fn extract_text_preview(data_json: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(data_json) {
        // Level-1 new format: aggregate OCR/clipboard text from sub_events
        if let Some(events) = v.get("sub_events").and_then(Value::as_array) {
            let text: String = events.iter()
                .filter_map(|e| {
                    e.get("ocr_full").and_then(Value::as_str).filter(|t| !t.is_empty())
                        .or_else(|| e.get("clipboard").and_then(Value::as_str).filter(|t| !t.is_empty()))
                })
                .collect::<Vec<_>>()
                .join(" ");
            return text.chars().take(200).collect();
        }
        // Legacy Level-1 format and Level-0: summary or text field
        let text = v.get("summary")
            .or_else(|| v.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or(data_json);
        return text.chars().take(200).collect();
    }
    data_json.chars().take(200).collect()
}

fn extract_text_from_json(data_json: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(data_json) {
        if let Some(text) = v.get("text").and_then(|v| v.as_str()) {
            return text.to_string();
        }
    }
    data_json.to_string()
}

fn extract_app_name_from_json(data_json: &str) -> String {
    serde_json::from_str::<Value>(data_json)
        .ok()
        .and_then(|v| v.get("app_name").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_default()
}

fn extract_window_title_from_json(data_json: &str) -> String {
    serde_json::from_str::<Value>(data_json)
        .ok()
        .and_then(|v| v.get("window_title").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_default()
}

/// Recognises a known website from an app/title fragment → a clean label.
fn known_site(s: &str) -> Option<&'static str> {
    let l = s.to_lowercase();
    const SITES: &[(&str, &str)] = &[
        ("youtube", "YouTube"), ("youtu.be", "YouTube"),
        ("wikipedia", "Wikipedia"), ("википеди", "Wikipedia"),
        ("github", "GitHub"), ("reddit", "Reddit"), ("twitch", "Twitch"),
        ("vk.com", "VK"), ("вконтакте", "VK"),
        ("telegram", "Telegram"), ("stack overflow", "Stack Overflow"),
        ("habr", "Habr"), ("хабр", "Habr"),
        ("kinopoisk", "Kinopoisk"), ("кинопоиск", "Kinopoisk"),
        ("soundcloud", "SoundCloud"), ("spotify", "Spotify"),
    ];
    SITES.iter().find(|(n, _)| l.contains(n)).map(|(_, label)| *label)
}

/// Deterministically turns a screen event's (app, window title) into ONE clean,
/// human log phrase — no OCR, no LLM — so recall reads well on ANY model. The
/// window title is structured by the app itself (page/file/project), so this is
/// reliable: "смотрел «Обзор iOS 27» на YouTube", "работал в VS Code: nic-assistant".
/// Strips a leading unread/tab counter that browsers prepend to the window
/// title, e.g. "(20) Real Page Title" → "Real Page Title".
fn strip_leading_count(s: &str) -> &str {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix('(') {
        if let Some(close) = rest.find(')') {
            let inner = &rest[..close];
            if !inner.is_empty() && inner.chars().all(|c| c.is_ascii_digit()) {
                return rest[close + 1..].trim_start();
            }
        }
    }
    s
}

fn activity_line(app: &str, title: &str) -> String {
    let app_l = app.to_lowercase();
    let title = strip_leading_count(title.trim());
    // Friendly display name for bare system apps.
    let app_pretty = match app_l.as_str() {
        "explorer" | "explorer.exe" => "File Explorer",
        _ => app,
    };

    // Split the title into clean segments on the usual separators.
    let norm = title
        .replace(" — ", "\u{1}").replace(" – ", "\u{1}")
        .replace(" - ", "\u{1}").replace(" | ", "\u{1}");
    let mut segs: Vec<&str> = norm.split('\u{1}').map(str::trim).filter(|s| !s.is_empty()).collect();

    const BROWSERS: &[&str] = &["firefox", "chrome", "edge", "opera", "yandex",
                                "brave", "chromium", "safari", "mozilla"];
    let is_browser = BROWSERS.iter().any(|b| app_l.contains(b));
    let is_editor = ["visual studio code", "vs code", "code", "intellij", "pycharm",
                     "clion", "webstorm", "rider", "sublime", "notepad++", "zed"]
        .iter().any(|e| app_l.contains(e));

    // Drop a trailing segment that's just the app/browser brand.
    if segs.len() > 1 {
        let last = segs[segs.len() - 1].to_lowercase();
        if BROWSERS.iter().any(|b| last.contains(b)) || last.contains("studio code") {
            segs.pop();
        }
    }

    if is_browser {
        let mut site: Option<&'static str> = None;
        for &s in &segs { if let Some(k) = known_site(s) { site = Some(k); break; } }
        if site.is_none() { site = known_site(title); }
        let mut page: &str = "";
        for &s in &segs { if known_site(s).is_none() { page = s; break; } }
        return match (page.is_empty(), site) {
            (false, Some(site)) => format!("watched «{page}» on {site}"),
            (false, None)       => format!("opened «{page}» in the browser"),
            (true,  Some(site)) => format!("was on {site}"),
            (true,  None)       => "browsing the web".to_string(),
        };
    }
    if is_editor {
        let editor = if app_l.contains("code") { "VS Code" } else { app_pretty };
        let what = segs.join(" — ");
        return if what.is_empty() {
            format!("worked in {editor}")
        } else {
            format!("worked in {editor}: {what}")
        };
    }
    let what = segs.join(" — ");
    if what.is_empty() { format!("in {app_pretty}") } else { format!("{app_pretty}: {what}") }
}

/// Coarse activity bucket as a 2nd-person past-tense verb phrase, used to build
/// the natural intro of `activity_summary` (e.g. "смотрел видео и работал в
/// коде"). Masculine past is the conventional Russian default. Pure & cheap.
fn activity_category(app: &str, title: &str) -> &'static str {
    let app_l = app.to_lowercase();
    let full  = format!("{} {}", app_l, title.to_lowercase());

    const BROWSERS: &[&str] = &["firefox", "chrome", "edge", "opera", "yandex",
                                "brave", "chromium", "safari", "mozilla"];
    if BROWSERS.iter().any(|b| app_l.contains(b)) {
        if ["youtube", "youtu.be", "twitch", "kinopoisk", "кинопоиск", "rutube"]
            .iter().any(|s| full.contains(s)) {
            return "watched videos";
        }
        if ["soundcloud", "spotify", "music.yandex", "яндекс.музык"]
            .iter().any(|s| full.contains(s)) {
            return "listened to music";
        }
        return "browsed the web";
    }
    if ["visual studio code", "vs code", "code", "intellij", "pycharm", "clion",
        "webstorm", "rider", "sublime", "notepad++", "zed"]
        .iter().any(|e| app_l.contains(e)) {
        return "worked on code";
    }
    if ["telegram", "discord", "whatsapp", "vk.com", "вконтакте", "slack"]
        .iter().any(|m| full.contains(m)) {
        return "chatted in messengers";
    }
    "worked in apps"
}

/// Capitalised relative day for the activity-summary intro: "Сегодня" (< 24 h),
/// "Вчера" (24–48 h), else "Недавно".
fn when_word(ts_micros: i64) -> &'static str {
    let secs = (Utc::now().timestamp_micros() - ts_micros) / 1_000_000;
    if secs < 86_400        { "Today" }
    else if secs < 172_800  { "Yesterday" }
    else                    { "Recently" }
}

/// Cache key for the LRU embed cache: stable hex hash of the full text.
fn embed_cache_key(text: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Narrow, unambiguous "my own past activity" check ("what did I do / what
/// happened / which apps"). For these queries we (a) always include the screen
/// timeline and (b) skip the web, so an empty «Из интернета» section can't
/// distract the model from the timeline. Kept separate from `is_screen_query`
/// because that one also matches bare time words ("сегодня", "недавно") that
/// legitimately need the web.
pub(crate) fn is_activity_recall(query: &str) -> bool {
    let q = query.to_lowercase();
    ["что я делал", "что делал", "чем занимался", "чем я занимался",
     "что я сделал", "что происходило", "что я запускал", "что запускал",
     "какие приложения", "какие программы", "что я смотрел", "что смотрел",
     "что я читал", "что читал", "что было на экране", "что открывал",
     "что я открывал", "моя активность", "мою активность", "чем я занят",
     "в течение сесси", "за сессию", "за эту сессию", "что я тут делал",
     "что я делаю", "чем занят",
     // English-only beta:
     "what did i do", "what did i", "what was i doing", "what was i just",
     "what have i been doing",
     "what did i open", "what did i watch", "what did i read", "what did i launch",
     "which apps", "which programs", "what apps", "what programs",
     "what happened", "what was on screen", "what was on the screen", "my activity",
    ].iter().any(|&t| q.contains(t))
}

/// "Кто я / как меня зовут" — a question about the USER's own name, answered
/// deterministically from the profile. Kept narrow so it never swallows "кто
/// такой <famous person>" (a general-knowledge question for the model).
pub(crate) fn asks_user_name(query: &str) -> bool {
    let q = query.to_lowercase();
    let bare: String = q.chars().filter(|c| c.is_alphanumeric() || *c == ' ').collect();
    if matches!(bare.trim(),
        "кто я" | "а кто я" | "кто я такой" | "кто я такая"
        | "ты знаешь кто я" | "ты знаешь как меня зовут"
        | "who am i" | "do you know who i am" | "do you know my name") { return true; }
    ["как меня зов", "как меня зват", "как моё имя", "как мое имя",
     "знаешь моё имя", "знаешь мое имя",
     "what is my name", "what's my name", "whats my name", "tell me my name"]
        .iter().any(|t| q.contains(t))
}

/// "Кто ты / как тебя зовут" — a question about NIC's own identity, answered
/// with a fixed line so the small model can't drift on it.
pub(crate) fn asks_assistant_identity(query: &str) -> bool {
    let q = query.to_lowercase();
    let bare: String = q.chars().filter(|c| c.is_alphanumeric() || *c == ' ').collect();
    if matches!(bare.trim(),
        "кто ты" | "ты кто" | "а кто ты" | "кто ты такой"
        | "who are you" | "what are you" | "who r u") { return true; }
    ["как тебя зов", "как тебя зват", "твоё имя", "твое имя", "как звать тебя",
     "what is your name", "what's your name", "whats your name", "tell me your name"]
        .iter().any(|t| q.contains(t))
}

fn is_screen_query(query: &str) -> bool {
    if is_activity_recall(query) { return true; }
    let q = query.to_lowercase();

    // Unambiguous screen / window phrasing → always look at the screen.
    let strong = [
        "экран", "на экране", "что открыто", "что вижу", "мой экран",
        "какое окно", "что за окно", "вкладка", "что было открыто",
        "открытый сайт", "что я слушаю", "что слушал", "какие приложения",
    ];
    if strong.iter().any(|&t| q.contains(t)) { return true; }

    // Ambiguous media / browser words ("сайт", "видео", "браузер", "смотрю")
    // appear in plenty of GENERAL questions ("какой сайт самый быстрый",
    // "что за видео"). Treat them as a screen query ONLY when the user refers to
    // themselves ("что Я смотрю") or asks about open-state ("какой браузер
    // открыт"). Without that, it's a general question — don't inject the screen
    // timeline, or the small model just parrots the latest screen line.
    let activity_word = ["ютуб", "youtube", "смотрел", "смотрю", "смотришь",
                         "видео", "браузер", "сайт"];
    if !activity_word.iter().any(|&t| q.contains(t)) { return false; }

    let self_ref = q.split(|c: char| !c.is_alphanumeric())
        .any(|w| matches!(w, "я" | "мне" | "мой" | "моя" | "моё" | "мои" | "мою" | "меня" | "мной"));
    let open_cue = ["открыт", "открыта", "открыто", "открыты"].iter().any(|&t| q.contains(t));
    self_ref || open_cue
}

/// Keyword-based semantic auto-tagger. Returns a list of tags like ["#code", "#work"].
/// Pure function, zero latency — no LLM needed.
fn auto_tag(text: &str, app_name: &str, window_title: &str) -> Vec<String> {
    let t = text.to_lowercase();
    let a = app_name.to_lowercase();
    let w = window_title.to_lowercase();

    let mut tags = Vec::new();

    let code_text = ["fn ", "pub ", "impl ", "struct ", "class ", "def ", "import ",
                     "return ", "let ", "const ", "async ", "await ", "cargo", "npm",
                     "git ", "docker", "kubectl", "pip ", "error[", "warning[",
                     "traceback", "exception", "panic", "stack trace"];
    let code_app  = ["code", "terminal", "cmd", "powershell", "vim", "neovim",
                     "rider", "intellij", "pycharm", "clion", "webstorm", "xcode",
                     "cursor", "zed", "notepad++", "sublime"];
    if code_text.iter().any(|k| t.contains(k))
        || code_app.iter().any(|k| a.contains(k) || w.contains(k))
    {
        tags.push("#code".to_string());
    }

    let work_text = ["excel", "word", "outlook", "teams", "meeting", "совещани",
                     "задача", "проект", "дедлайн", "отчёт", "отчет", "invoice",
                     "договор", "контракт", "клиент", "заказчик", "jira", "trello",
                     "confluence", "notion", "slack", "zoom", "google meet"];
    let work_app  = ["excel", "word", "outlook", "teams", "notion", "jira",
                     "confluence", "trello", "asana", "linear"];
    if work_text.iter().any(|k| t.contains(k))
        || work_app.iter().any(|k| a.contains(k) || w.contains(k))
    {
        tags.push("#work".to_string());
    }

    let finance_kw = ["банк", "платёж", "платеж", "счёт", "счет", "рубл", "долл",
                      "евро", "криптo", "биткоин", "бюджет", "расход", "доход",
                      "зарплат", "налог", "invoice", "оплат", "перевод", "card",
                      "wallet", "trading", "binance", "tinkoff", "sber"];
    if finance_kw.iter().any(|k| t.contains(k) || a.contains(k) || w.contains(k)) {
        tags.push("#finance".to_string());
    }

    let entertain_kw = ["youtube", "twitch", "netflix", "кинопоиск", "spotify",
                        "steam", "игр", "game", "movie", "фильм", "сериал",
                        "музык", "music", "podcast", "подкаст", "тикток", "tiktok",
                        "instagram", "vk.com", "vkvideo", "dzen"];
    if entertain_kw.iter().any(|k| t.contains(k) || a.contains(k) || w.contains(k)) {
        tags.push("#entertainment".to_string());
    }

    tags
}

pub(crate) fn log_process_ram() {
    let pid = sysinfo::Pid::from_u32(std::process::id());
    let sys = sysinfo::System::new_all();
    if let Some(proc) = sys.process(pid) {
        info!("[PERF/RAM] Process RSS: {} MB", proc.memory() / 1_048_576);
    }
}

fn is_screen_event(data_json: &str) -> bool {
    serde_json::from_str::<Value>(data_json)
        .ok()
        .and_then(|v| v.get("app_name").and_then(|a| a.as_str()).map(str::to_string))
        .map(|app| app != "clipboard" && app != "system" && app != "user")
        .unwrap_or(false)
}

fn is_qa_event(data_json: &str) -> bool {
    serde_json::from_str::<Value>(data_json)
        .ok()
        .and_then(|v| v.get("app_name").and_then(|a| a.as_str()).map(str::to_string))
        .map(|app| app == "dialogue")
        .unwrap_or(false)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(id: &str, time_start: i64) -> UnifiedRow {
        UnifiedRow { id: id.into(), time_start, time_end: time_start, level: 0, data_json: "{}".into() }
    }

    #[test]
    fn test_group_by_hour_same_bucket() {
        let rows = vec![
            make_row("a", 0),
            make_row("b", 1_800_000_000), // 30 min later — same hour
        ];
        let groups = group_by_hour(rows);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn test_group_by_hour_different_buckets() {
        let rows = vec![
            make_row("a", 0),
            make_row("b", 4_000_000_000), // >1 h later — different hour
        ];
        let groups = group_by_hour(rows);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_group_by_week_spanning_two_weeks() {
        let week = 7 * 24 * 3_600_000_000i64;
        let rows = vec![
            make_row("a", 0),
            make_row("b", week + 1),
        ];
        let groups = group_by_week(rows);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_extract_text_preview_summary_field() {
        let data = r#"{"summary": "Event summary"}"#;
        assert_eq!(extract_text_preview(data), "Event summary");
    }

    #[test]
    fn test_extract_text_preview_text_field() {
        let data = r#"{"text": "Raw text"}"#;
        assert_eq!(extract_text_preview(data), "Raw text");
    }

    #[test]
    fn test_extract_text_preview_fallback() {
        let data = "not json";
        assert_eq!(extract_text_preview(data), "not json");
    }

    #[test]
    fn test_is_screen_event_screen() {
        let d = r#"{"app_name":"chrome","window_title":"Google","text":"..."}"#;
        assert!(is_screen_event(d));
    }

    #[test]
    fn test_is_screen_event_clipboard() {
        let d = r#"{"app_name":"clipboard","window_title":"Clipboard Capture","text":"..."}"#;
        assert!(!is_screen_event(d));
    }

    #[test]
    fn test_is_screen_event_system() {
        let d = r#"{"app_name":"system","window_title":"response","text":"..."}"#;
        assert!(!is_screen_event(d));
    }

    #[test]
    fn test_fmt_ts_zero() {
        // Epoch 0 µs should produce a valid date string, not panic.
        let s = fmt_ts(0);
        assert!(!s.is_empty());
        assert_ne!(s, "?");
    }

    #[test]
    fn test_archive_cipher_roundtrip() {
        let mut key = [0u8; 32];
        key[0] = 42; key[31] = 7;
        let cipher = ArchiveCipher::new(key);
        let plain  = r#"{"id":"test","data":"hello world"}"#;
        let enc    = cipher.encrypt(plain).expect("encrypt");
        let dec    = cipher.decrypt(&enc).expect("decrypt");
        assert_eq!(dec, plain);
    }

    // ── Deep eternal-memory tests ─────────────────────────────────────────────

    // -- auto_tag: code triggers --

    #[test]
    fn autotag_code_fn_keyword() {
        assert!(auto_tag("fn main() {}", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_pub_keyword() {
        assert!(auto_tag("pub struct Foo {}", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_impl_keyword() {
        assert!(auto_tag("impl Foo {}", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_struct_keyword() {
        assert!(auto_tag("struct Bar;", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_class_keyword() {
        assert!(auto_tag("class Baz:", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_docker_keyword() {
        assert!(auto_tag("docker build .", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_cargo_keyword() {
        assert!(auto_tag("cargo test --bins", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_panic_keyword() {
        assert!(auto_tag("thread panicked at...", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_error_bracket() {
        assert!(auto_tag("error[E0502]: borrow...", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_traceback_python() {
        assert!(auto_tag("Traceback (most recent call last)", "notepad", "x").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_vscode_app() {
        assert!(auto_tag("some text", "code", "main.rs").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_terminal_app() {
        assert!(auto_tag("ls -la", "terminal", "bash").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_powershell_app() {
        assert!(auto_tag("Get-Process", "powershell", "Admin").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_intellij_window() {
        assert!(auto_tag("some text", "idea", "intellij idea").contains(&"#code".into()));
    }

    #[test]
    fn autotag_code_vim_app() {
        assert!(auto_tag("", "vim", "main.rs").contains(&"#code".into()));
    }

    // -- auto_tag: work triggers --

    #[test]
    fn autotag_work_meeting_keyword() {
        assert!(auto_tag("team meeting at 14:00", "chrome", "calendar").contains(&"#work".into()));
    }

    #[test]
    fn autotag_work_jira_text() {
        assert!(auto_tag("check jira ticket", "chrome", "x").contains(&"#work".into()));
    }

    #[test]
    fn autotag_work_deadline_cyrillic() {
        assert!(auto_tag("дедлайн завтра в 18:00", "chrome", "x").contains(&"#work".into()));
    }

    #[test]
    fn autotag_work_zadacha_cyrillic() {
        assert!(auto_tag("задача по проекту", "chrome", "x").contains(&"#work".into()));
    }

    #[test]
    fn autotag_work_invoice_text() {
        assert!(auto_tag("invoice #1234 due Monday", "word", "invoice.docx").contains(&"#work".into()));
    }

    #[test]
    fn autotag_work_slack_in_text() {
        // "slack" is a work keyword in work_text, not work_app
        assert!(auto_tag("check slack channel", "chrome", "x").contains(&"#work".into()));
    }

    #[test]
    fn autotag_work_zoom_text() {
        assert!(auto_tag("zoom call with client", "chrome", "x").contains(&"#work".into()));
    }

    #[test]
    fn autotag_work_notion_app() {
        assert!(auto_tag("notes", "notion", "Project").contains(&"#work".into()));
    }

    // -- auto_tag: finance triggers --

    #[test]
    fn autotag_finance_bank_keyword() {
        assert!(auto_tag("перевод в банк сбербанк", "chrome", "x").contains(&"#finance".into()));
    }

    #[test]
    fn autotag_finance_tinkoff_app() {
        assert!(auto_tag("баланс", "tinkoff", "счёт").contains(&"#finance".into()));
    }

    #[test]
    fn autotag_finance_binance() {
        assert!(auto_tag("торги на binance сегодня", "chrome", "x").contains(&"#finance".into()));
    }

    #[test]
    fn autotag_finance_rubl_keyword() {
        assert!(auto_tag("стоимость 500 рублей", "chrome", "x").contains(&"#finance".into()));
    }

    #[test]
    fn autotag_finance_nalog_keyword() {
        assert!(auto_tag("налоговая декларация", "chrome", "x").contains(&"#finance".into()));
    }

    #[test]
    fn autotag_finance_wallet_keyword() {
        assert!(auto_tag("check wallet balance", "chrome", "x").contains(&"#finance".into()));
    }

    #[test]
    fn autotag_finance_trading_keyword() {
        assert!(auto_tag("trading signals today", "chrome", "x").contains(&"#finance".into()));
    }

    // -- auto_tag: entertainment triggers --

    #[test]
    fn autotag_entertainment_youtube_text() {
        assert!(auto_tag("watching youtube", "chrome", "YouTube").contains(&"#entertainment".into()));
    }

    #[test]
    fn autotag_entertainment_netflix() {
        assert!(auto_tag("", "chrome", "Netflix — Watch Movies").contains(&"#entertainment".into()));
    }

    #[test]
    fn autotag_entertainment_steam_app() {
        assert!(auto_tag("", "steam", "Library").contains(&"#entertainment".into()));
    }

    #[test]
    fn autotag_entertainment_music_keyword() {
        assert!(auto_tag("listening to music playlist", "chrome", "x").contains(&"#entertainment".into()));
    }

    #[test]
    fn autotag_entertainment_spotify() {
        assert!(auto_tag("", "spotify", "Now Playing").contains(&"#entertainment".into()));
    }

    #[test]
    fn autotag_entertainment_game_keyword() {
        assert!(auto_tag("game over score 1000", "steam", "x").contains(&"#entertainment".into()));
    }

    #[test]
    fn autotag_entertainment_serial_cyrillic() {
        assert!(auto_tag("смотрю сериал", "chrome", "x").contains(&"#entertainment".into()));
    }

    #[test]
    fn autotag_entertainment_tiktok() {
        assert!(auto_tag("", "chrome", "TikTok").contains(&"#entertainment".into()));
    }

    // -- auto_tag: negative cases --

    #[test]
    fn autotag_plain_text_no_tags() {
        let tags = auto_tag("обычный текст ничего особенного", "chrome", "Google Search");
        assert!(tags.is_empty(), "plain search query should produce no tags");
    }

    #[test]
    fn autotag_empty_all_no_tags() {
        assert!(auto_tag("", "", "").is_empty());
    }

    #[test]
    fn autotag_no_duplicate_tags() {
        // Multiple code keywords but only one #code tag
        let tags = auto_tag("fn foo() {} fn bar() {} struct Baz {}", "code", "main.rs");
        let code_count = tags.iter().filter(|t| *t == "#code").count();
        assert_eq!(code_count, 1, "should have exactly one #code tag");
    }

    #[test]
    fn autotag_tag_order_stable() {
        // tags are pushed in order: code → work → finance → entertainment
        let tags = auto_tag("fn foo() {} задача оплатить youtube", "code", "x");
        if tags.len() >= 2 {
            let code_pos   = tags.iter().position(|t| t == "#code");
            let work_pos   = tags.iter().position(|t| t == "#work");
            assert!(code_pos < work_pos, "#code should come before #work");
        }
    }

    // -- embed_cache_key: collision resistance --

    #[test]
    fn embed_key_no_collision_similar_strings() {
        let keys: Vec<_> = (0..100).map(|i| embed_cache_key(&format!("query {}", i))).collect();
        let unique: std::collections::HashSet<_> = keys.iter().collect();
        assert_eq!(unique.len(), 100, "100 distinct inputs should produce 100 distinct keys");
    }

    #[test]
    fn embed_key_transposition_different() {
        // "ab" and "ba" are different strings — keys must differ
        assert_ne!(embed_cache_key("ab"), embed_cache_key("ba"));
    }

    #[test]
    fn embed_key_punctuation_matters() {
        assert_ne!(embed_cache_key("hello"), embed_cache_key("hello!"));
    }

    #[test]
    fn embed_key_whitespace_matters() {
        assert_ne!(embed_cache_key("hello world"), embed_cache_key("helloworld"));
    }

    #[test]
    fn embed_key_deterministic_across_calls_many() {
        for i in 0..50 {
            let s = format!("тестовый запрос номер {}", i);
            assert_eq!(embed_cache_key(&s), embed_cache_key(&s));
        }
    }

    // -- group_by_hour: comprehensive --

    #[test]
    fn group_by_hour_preserves_all_rows() {
        let rows: Vec<_> = (0..100).map(|i| make_row(&i.to_string(), i as i64 * 60_000_000)).collect();
        let total_in: usize = 100;
        let groups = group_by_hour(rows);
        let total_out: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total_in, total_out, "no rows should be lost or duplicated");
    }

    #[test]
    fn group_by_hour_buckets_24_hours_in_day() {
        // 24 events, one per hour
        let rows: Vec<_> = (0i64..24).map(|h| make_row(&h.to_string(), h * 3_600_000_000)).collect();
        let groups = group_by_hour(rows);
        assert_eq!(groups.len(), 24);
    }

    #[test]
    fn group_by_hour_all_same_bucket() {
        let rows: Vec<_> = (0i64..10).map(|i| make_row(&i.to_string(), i * 100_000_000)).collect();
        // all within same hour (hour 0)
        let groups = group_by_hour(rows);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 10);
    }

    #[test]
    fn group_by_hour_consistent_sort_order_among_groups() {
        let rows = vec![
            make_row("c", 7_300_000_000),  // hour 2
            make_row("a", 0),               // hour 0
            make_row("b", 3_700_000_000),   // hour 1
        ];
        let groups = group_by_hour(rows);
        let first_times: Vec<i64> = groups.iter().map(|g| g[0].time_start).collect();
        let sorted = {
            let mut s = first_times.clone();
            s.sort();
            s
        };
        assert_eq!(first_times, sorted, "groups should be sorted by time");
    }

    // -- group_by_week: comprehensive --

    #[test]
    fn group_by_week_preserves_all_rows() {
        let week = 7 * 24 * 3_600_000_000i64;
        let rows: Vec<_> = (0i64..10).map(|i| make_row(&i.to_string(), i * week / 3)).collect();
        let total_out: usize = group_by_week(rows).iter().map(|g| g.len()).sum();
        assert_eq!(total_out, 10);
    }

    #[test]
    fn group_by_week_exactly_at_boundary() {
        let week = 7 * 24 * 3_600_000_000i64;
        let rows = vec![
            make_row("a", week - 1),   // last µs of week 0
            make_row("b", week),        // first µs of week 1
            make_row("c", week + 1),    // still week 1
        ];
        let groups = group_by_week(rows);
        assert_eq!(groups.len(), 2);
        let g0_ids: Vec<_> = groups[0].iter().map(|r| r.id.as_str()).collect();
        assert!(g0_ids.contains(&"a"));
    }

    // -- fmt_ts: full coverage --

    #[test]
    fn fmt_ts_max_positive_no_panic() {
        // Large positive timestamp — should not panic
        let s = fmt_ts(i64::MAX / 10);
        assert!(!s.is_empty());
    }

    #[test]
    fn fmt_ts_one_second_in_micros() {
        // 1970-01-01 00:00:01 UTC = 1_000_000 µs
        let s = fmt_ts(1_000_000);
        assert!(s.contains("1970"));
    }

    #[test]
    fn fmt_ts_format_length_19() {
        // Valid timestamp → "YYYY-MM-DD HH:MM:SS" = 19 chars
        let s = fmt_ts(1_000_000_000_000_000);
        assert_eq!(s.len(), 19);
    }

    // -- is_qa_event: fuzzing --

    #[test]
    fn qa_event_null_json_false() {
        assert!(!is_qa_event("null"));
    }

    #[test]
    fn qa_event_array_json_false() {
        assert!(!is_qa_event(r#"["dialogue"]"#));
    }

    #[test]
    fn qa_event_nested_app_name_false() {
        // app_name is nested, not top-level
        assert!(!is_qa_event(r#"{"outer":{"app_name":"dialogue"}}"#));
    }

    #[test]
    fn qa_event_dialogue_unicode_exact() {
        // Only exactly "dialogue" matches, not similar strings
        for s in &[
            "Dialogue",   // wrong case
            "DIALOGUE",   // all caps
            "dialog",     // truncated
            "dialoguе",   // last char is Cyrillic 'е' (U+0435) not ASCII 'e' (U+0065)
        ] {
            let json = format!("{{\"app_name\":\"{}\"}}", s);
            assert!(!is_qa_event(&json), "should not match: {}", s);
        }
    }

    // -- extract_text_preview: sub_events --

    #[test]
    fn text_preview_sub_events_empty_ocr_skipped() {
        let json = r#"{"sub_events":[{"ocr_full":""},{"ocr_full":"real text"}]}"#;
        let result = extract_text_preview(json);
        // Empty ocr_full filtered, "real text" kept
        assert!(result.contains("real text"));
        assert!(!result.is_empty());
    }

    #[test]
    fn text_preview_sub_events_mixed_ocr_and_clipboard() {
        let json = r#"{"sub_events":[{"ocr_full":"ocr"},{"clipboard":"clip"}]}"#;
        let result = extract_text_preview(json);
        assert!(result.contains("ocr"));
    }

    #[test]
    fn text_preview_sub_events_all_empty_strings_is_empty() {
        let json = r#"{"sub_events":[{"ocr_full":""},{"clipboard":""}]}"#;
        let result = extract_text_preview(json);
        assert!(result.is_empty());
    }

    #[test]
    fn text_preview_sub_events_200_char_limit() {
        let long = "a".repeat(300);
        let json = format!(r#"{{"sub_events":[{{"ocr_full":"{}"}}]}}"#, long);
        let result = extract_text_preview(&json);
        assert!(result.len() <= 200);
    }

    // -- summarize_code: extended --

    #[test]
    fn summarize_non_code_exactly_500_boundary() {
        // 500 chars → returned as-is (no truncation)
        let text = "z".repeat(500);
        assert_eq!(summarize_code(&text).len(), 500);
    }

    #[test]
    fn summarize_non_code_501_chars_truncated() {
        let text = "z".repeat(501);
        assert_eq!(summarize_code(&text).len(), 500);
    }

    #[test]
    fn summarize_code_use_statements_kept() {
        let src = "use std::collections::HashMap;\nfn foo() {}\n";
        let r = summarize_code(src);
        assert!(r.contains("use std::collections::HashMap;"));
    }

    #[test]
    fn summarize_code_const_kept() {
        let src = "const MAX: usize = 100;\nfn foo() {}\n";
        let r = summarize_code(src);
        assert!(r.contains("const MAX: usize = 100;"));
    }

    #[test]
    fn summarize_code_trait_kept() {
        let src = "trait Animal { fn speak(&self); }\n";
        let r = summarize_code(src);
        assert!(r.contains("trait Animal"));
    }

    #[test]
    fn summarize_code_python_class() {
        let src = "class MyClass:\n    def __init__(self):\n        self.x = 1\n";
        let r = summarize_code(src);
        assert!(r.contains("class MyClass"));
    }

    #[test]
    fn summarize_code_depth_tracking_nested_braces() {
        // Deeply nested code should be summarised, not panic
        let nested = "fn outer() {\n".to_string()
            + &"fn inner() {\n".repeat(20)
            + "let x = 1;\n"
            + &"}\n".repeat(21);
        let _ = summarize_code(&nested); // no panic
    }

    // ── archive cipher: more roundtrip cases ─────────────────────────────────

    #[test]
    fn test_archive_cipher_different_nonces() {
        let key    = [1u8; 32];
        let cipher = ArchiveCipher::new(key);
        let enc1   = cipher.encrypt("same").expect("e1");
        let enc2   = cipher.encrypt("same").expect("e2");
        // Different nonces → different ciphertext each call
        assert_ne!(enc1, enc2);
    }

    // ── is_qa_event ───────────────────────────────────────────────────────────

    #[test]
    fn qa_event_dialogue_app_name_true() {
        let json = r#"{"app_name":"dialogue","text":"Q\nA"}"#;
        assert!(is_qa_event(json));
    }

    #[test]
    fn qa_event_chrome_app_name_false() {
        let json = r#"{"app_name":"chrome","text":"some page"}"#;
        assert!(!is_qa_event(json));
    }

    #[test]
    fn qa_event_no_app_name_false() {
        let json = r#"{"text":"hello"}"#;
        assert!(!is_qa_event(json));
    }

    #[test]
    fn qa_event_invalid_json_false() {
        assert!(!is_qa_event("not json at all"));
    }

    #[test]
    fn qa_event_empty_string_false() {
        assert!(!is_qa_event(""));
    }

    #[test]
    fn qa_event_dialogue_case_sensitive() {
        // "Dialogue" != "dialogue" — must be exact lowercase
        let json = r#"{"app_name":"Dialogue","text":"Q\nA"}"#;
        assert!(!is_qa_event(json));
    }

    #[test]
    fn qa_event_clipboard_is_not_qa() {
        let json = r#"{"app_name":"clipboard","text":"copied"}"#;
        assert!(!is_qa_event(json));
    }

    #[test]
    fn qa_event_system_is_not_qa() {
        let json = r#"{"app_name":"system","text":"sys"}"#;
        assert!(!is_qa_event(json));
    }

    // ── extract_text_from_json ────────────────────────────────────────────────

    #[test]
    fn extract_text_has_text_field() {
        let json = r#"{"app_name":"chrome","text":"hello world"}"#;
        assert_eq!(extract_text_from_json(json), "hello world");
    }

    #[test]
    fn extract_text_no_text_field_returns_raw() {
        let json = r#"{"app_name":"chrome"}"#;
        assert_eq!(extract_text_from_json(json), json);
    }

    #[test]
    fn extract_text_invalid_json_returns_raw() {
        let raw = "не JSON вообще";
        assert_eq!(extract_text_from_json(raw), raw);
    }

    #[test]
    fn extract_text_empty_text_field() {
        let json = r#"{"text":""}"#;
        assert_eq!(extract_text_from_json(json), "");
    }

    #[test]
    fn extract_text_numeric_text_field_not_returned() {
        // text field is a number, not a string — falls back to raw
        let json = r#"{"text":42}"#;
        assert_eq!(extract_text_from_json(json), json);
    }

    #[test]
    fn extract_text_unicode_content() {
        let json = r#"{"text":"Привет мир 🌍"}"#;
        assert_eq!(extract_text_from_json(json), "Привет мир 🌍");
    }

    #[test]
    fn extract_text_multiline_value() {
        let json = "{\"text\":\"line1\\nline2\\nline3\"}";
        let result = extract_text_from_json(json);
        assert!(result.contains("line1"));
        assert!(result.contains("line3"));
    }

    // ── extract_app_name_from_json ────────────────────────────────────────────

    #[test]
    fn app_name_chrome_extracted() {
        let json = r#"{"app_name":"chrome","text":"..."}"#;
        assert_eq!(extract_app_name_from_json(json), "chrome");
    }

    #[test]
    fn app_name_missing_returns_empty() {
        let json = r#"{"text":"hello"}"#;
        assert_eq!(extract_app_name_from_json(json), "");
    }

    #[test]
    fn app_name_invalid_json_returns_empty() {
        assert_eq!(extract_app_name_from_json("garbage"), "");
    }

    #[test]
    fn app_name_numeric_type_returns_empty() {
        let json = r#"{"app_name":123}"#;
        assert_eq!(extract_app_name_from_json(json), "");
    }

    #[test]
    fn app_name_empty_string_value() {
        let json = r#"{"app_name":""}"#;
        assert_eq!(extract_app_name_from_json(json), "");
    }

    #[test]
    fn app_name_dialogue_extracted() {
        let json = r#"{"app_name":"dialogue","text":"Вопрос: Q\nОтвет: A"}"#;
        assert_eq!(extract_app_name_from_json(json), "dialogue");
    }

    // ── embed_cache_key ───────────────────────────────────────────────────────

    #[test]
    fn embed_key_same_input_same_output() {
        let k1 = embed_cache_key("hello world");
        let k2 = embed_cache_key("hello world");
        assert_eq!(k1, k2);
    }

    #[test]
    fn embed_key_different_inputs_different_keys() {
        let k1 = embed_cache_key("hello");
        let k2 = embed_cache_key("world");
        assert_ne!(k1, k2);
    }

    #[test]
    fn embed_key_is_16_hex_chars() {
        let k = embed_cache_key("test");
        assert_eq!(k.len(), 16, "key should be 16 hex chars");
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()), "should be hex");
    }

    #[test]
    fn embed_key_empty_string_returns_key() {
        let k = embed_cache_key("");
        assert_eq!(k.len(), 16);
    }

    #[test]
    fn embed_key_cyrillic_stable() {
        let k1 = embed_cache_key("Привет мир");
        let k2 = embed_cache_key("Привет мир");
        assert_eq!(k1, k2);
    }

    #[test]
    fn embed_key_long_text_stable() {
        let long = "a".repeat(10_000);
        let k1 = embed_cache_key(&long);
        let k2 = embed_cache_key(&long);
        assert_eq!(k1, k2);
    }

    #[test]
    fn embed_key_case_sensitive() {
        let k1 = embed_cache_key("Hello");
        let k2 = embed_cache_key("hello");
        assert_ne!(k1, k2);
    }

    // ── is_screen_query ───────────────────────────────────────────────────────

    #[test]
    fn screen_query_ekran_true() {
        assert!(is_screen_query("что на экране?"));
    }

    #[test]
    fn screen_query_okno_true() {
        assert!(is_screen_query("какое окно открыто"));
    }

    #[test]
    fn screen_query_youtube_true() {
        assert!(is_screen_query("что я смотрел на youtube"));
    }

    #[test]
    fn screen_query_browser_true() {
        assert!(is_screen_query("какой браузер открыт"));
    }

    #[test]
    fn screen_query_ordinary_question_false() {
        assert!(!is_screen_query("какая погода в Москве?"));
    }

    #[test]
    fn screen_query_empty_false() {
        assert!(!is_screen_query(""));
    }

    #[test]
    fn screen_query_case_insensitive_upper() {
        // "ЭКРАН" should match because is_screen_query lowercases input
        assert!(is_screen_query("ЭКРАН"));
    }

    #[test]
    fn screen_query_video_needs_self_or_open() {
        // "что за видео" alone is ambiguous/general now (a former false-positive
        // source); it only counts as a screen query with a self-reference.
        assert!(!is_screen_query("что за видео"));
        assert!(is_screen_query("что за видео я смотрю"));
    }

    #[test]
    fn screen_query_general_questions_false() {
        // Regression for the "every answer is the same screen line" bug: these are
        // GENERAL questions and must NOT pull in the screen timeline.
        assert!(!is_screen_query("какой сайт самый быстрый"));
        assert!(!is_screen_query("какой браузер самый быстрый"));
        assert!(!is_screen_query("какое видео посоветуешь"));
    }

    #[test]
    fn screen_query_chto_otkryto_true() {
        assert!(is_screen_query("что открыто на экране"));
    }

    #[test]
    fn screen_query_smotrel_true() {
        assert!(is_screen_query("что я смотрел недавно"));
    }

    // ── summarize_code ────────────────────────────────────────────────────────

    #[test]
    fn summarize_code_non_code_truncates_to_500() {
        let text = "a".repeat(600);
        let result = summarize_code(&text);
        assert_eq!(result.len(), 500);
    }

    #[test]
    fn summarize_code_non_code_short_unchanged() {
        let text = "hello world";
        assert_eq!(summarize_code(text), "hello world");
    }

    #[test]
    fn summarize_code_fn_keyword_triggers_strip() {
        let src = "fn foo() {\n    let x = 1;\n    x + 1\n}\n";
        let result = summarize_code(src);
        assert!(result.contains("fn foo()"));
    }

    #[test]
    fn summarize_code_pub_struct_kept() {
        let src = "pub struct Foo {\n    x: i32,\n}\n";
        let result = summarize_code(src);
        assert!(result.contains("pub struct Foo"));
    }

    #[test]
    fn summarize_code_result_under_2000_chars() {
        let big_fn = format!("fn big() {{\n{}\n}}\n", "let x = 1;\n".repeat(500));
        let result = summarize_code(&big_fn);
        assert!(result.len() <= 2000);
    }

    #[test]
    fn summarize_code_empty_input() {
        assert_eq!(summarize_code(""), "");
    }

    #[test]
    fn summarize_code_comment_lines_kept() {
        let src = "/// This is a doc comment\nfn bar() {}\n";
        let result = summarize_code(src);
        assert!(result.contains("/// This is a doc comment"));
    }

    // ── auto_tag ──────────────────────────────────────────────────────────────

    #[test]
    fn auto_tag_code_by_text_keyword() {
        let tags = auto_tag("fn main() { println!(\"hello\"); }", "notepad", "main.rs");
        assert!(tags.contains(&"#code".to_string()));
    }

    #[test]
    fn auto_tag_code_by_app_name() {
        let tags = auto_tag("some text", "code", "untitled");
        assert!(tags.contains(&"#code".to_string()));
    }

    #[test]
    fn auto_tag_work_by_keyword() {
        let tags = auto_tag("дедлайн проекта через неделю", "chrome", "outlook");
        assert!(tags.contains(&"#work".to_string()));
    }

    #[test]
    fn auto_tag_finance_by_keyword() {
        let tags = auto_tag("перевод на счёт в банке", "chrome", "банк");
        assert!(tags.contains(&"#finance".to_string()));
    }

    #[test]
    fn auto_tag_entertainment_youtube() {
        let tags = auto_tag("watching youtube videos", "chrome", "YouTube");
        assert!(tags.contains(&"#entertainment".to_string()));
    }

    #[test]
    fn auto_tag_no_match_empty_tags() {
        let tags = auto_tag("обычный текст без ключевых слов", "notepad", "untitled");
        assert!(tags.is_empty());
    }

    #[test]
    fn auto_tag_multiple_categories() {
        let tags = auto_tag("invoice fn main() {} банк", "code", "работа");
        assert!(tags.contains(&"#code".to_string()));
        assert!(tags.contains(&"#work".to_string()) || tags.contains(&"#finance".to_string()));
    }

    // ── group_by_hour extended ────────────────────────────────────────────────

    #[test]
    fn group_by_hour_empty_input() {
        let groups = group_by_hour(vec![]);
        assert!(groups.is_empty());
    }

    #[test]
    fn group_by_hour_single_row() {
        let groups = group_by_hour(vec![make_row("x", 0)]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
    }

    #[test]
    fn group_by_hour_sorted_within_bucket() {
        let rows = vec![
            make_row("b", 3_000_000_000),
            make_row("a", 1_000_000_000),
        ];
        let groups = group_by_hour(rows);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0][0].id, "a");
        assert_eq!(groups[0][1].id, "b");
    }

    #[test]
    fn group_by_hour_boundary_at_one_hour() {
        // Exactly at 1-hour boundary = 3_600_000_000 µs
        let rows = vec![
            make_row("a", 3_599_999_999), // last µs of hour 0
            make_row("b", 3_600_000_000), // first µs of hour 1
        ];
        let groups = group_by_hour(rows);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn group_by_hour_buckets_sorted_by_time() {
        let rows = vec![
            make_row("c", 10_000_000_000),
            make_row("a", 0),
            make_row("b", 3_700_000_000),
        ];
        let groups = group_by_hour(rows);
        assert_eq!(groups.len(), 3);
        assert!(groups[0][0].time_start < groups[1][0].time_start);
        assert!(groups[1][0].time_start < groups[2][0].time_start);
    }

    // ── group_by_week extended ────────────────────────────────────────────────

    #[test]
    fn group_by_week_empty_input() {
        let groups = group_by_week(vec![]);
        assert!(groups.is_empty());
    }

    #[test]
    fn group_by_week_same_week() {
        let week = 7 * 24 * 3_600_000_000i64;
        let rows = vec![
            make_row("a", 0),
            make_row("b", week - 1),
        ];
        let groups = group_by_week(rows);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn group_by_week_sorted_within_bucket() {
        let rows = vec![
            make_row("b", 86_400_000_000),
            make_row("a", 0),
        ];
        let groups = group_by_week(rows);
        assert_eq!(groups[0][0].id, "a");
    }

    // ── fmt_ts ────────────────────────────────────────────────────────────────

    #[test]
    fn fmt_ts_known_timestamp() {
        // 1_000_000_000_000_000 µs = 2001-09-08 21:46:40 UTC
        let s = fmt_ts(1_000_000_000_000_000);
        assert!(s.contains("2001"));
    }

    #[test]
    fn fmt_ts_negative_invalid() {
        // Very negative timestamp — should return "?" gracefully
        let s = fmt_ts(i64::MIN);
        assert_eq!(s, "?");
    }

    #[test]
    fn fmt_ts_format_contains_dashes_and_colons() {
        let s = fmt_ts(1_000_000_000_000_000);
        // Format is YYYY-MM-DD HH:MM:SS
        assert!(s.contains('-'));
        assert!(s.contains(':'));
        assert_eq!(s.len(), 19);
    }

    // ── extract_text_preview ──────────────────────────────────────────────────

    #[test]
    fn text_preview_sub_events_ocr() {
        let json = r#"{"sub_events":[{"ocr_full":"ocr text"},{"ocr_full":"more"}]}"#;
        let result = extract_text_preview(json);
        assert!(result.contains("ocr text"));
        assert!(result.contains("more"));
    }

    #[test]
    fn text_preview_sub_events_clipboard_fallback() {
        let json = r#"{"sub_events":[{"clipboard":"pasted text"},{"ocr_full":""}]}"#;
        let result = extract_text_preview(json);
        assert!(result.contains("pasted text"));
    }

    #[test]
    fn text_preview_truncates_at_200() {
        let long = "a".repeat(300);
        let json = format!("{{\"text\":\"{}\"}}", long);
        let result = extract_text_preview(&json);
        assert_eq!(result.len(), 200);
    }

    #[test]
    fn text_preview_summary_preferred_over_text() {
        let json = r#"{"summary":"the summary","text":"the text"}"#;
        let result = extract_text_preview(json);
        assert_eq!(result, "the summary");
    }

    // ── is_screen_event ───────────────────────────────────────────────────────

    #[test]
    fn screen_event_user_app_is_not_screen() {
        let d = r#"{"app_name":"user","text":"..."}"#;
        assert!(!is_screen_event(d));
    }

    #[test]
    fn screen_event_empty_json_false() {
        assert!(!is_screen_event("{}"));
    }

    #[test]
    fn screen_event_dialogue_is_screen() {
        let d = r#"{"app_name":"dialogue","text":"..."}"#;
        assert!(is_screen_event(d));
    }

    // ── iron-log helpers ──────────────────────────────────────────────────────

    #[test]
    fn strip_leading_count_removes_browser_counter() {
        assert_eq!(strip_leading_count("(20) Реальный заголовок"), "Реальный заголовок");
        assert_eq!(strip_leading_count("(1) Видео"), "Видео");
    }

    #[test]
    fn strip_leading_count_keeps_real_parens() {
        // A non-numeric or content-bearing parenthesis must not be stripped.
        assert_eq!(strip_leading_count("(beta) NIC"), "(beta) NIC");
        assert_eq!(strip_leading_count("обычный заголовок"), "обычный заголовок");
    }

    #[test]
    fn activity_line_youtube_strips_count_and_brand() {
        let line = activity_line("firefox", "(20) Обзор iOS 27 - YouTube — Mozilla Firefox");
        assert_eq!(line, "watched «Обзор iOS 27» on YouTube");
    }

    #[test]
    fn activity_line_explorer_is_friendly() {
        assert_eq!(activity_line("explorer", ""), "in File Explorer");
    }

    #[test]
    fn activity_category_buckets() {
        assert_eq!(activity_category("firefox", "что-то - YouTube"), "watched videos");
        assert_eq!(activity_category("Code", "main.rs"), "worked on code");
        assert_eq!(activity_category("explorer", "Загрузки"), "worked in apps");
    }

    #[test]
    fn when_word_buckets() {
        let now = Utc::now().timestamp_micros();
        assert_eq!(when_word(now), "Today");
        assert_eq!(when_word(now - 30 * 3_600_000_000i64), "Yesterday");
        assert_eq!(when_word(now - 5  * 86_400_000_000i64), "Recently");
    }

    // ── deterministic-router predicates ───────────────────────────────────────

    #[test]
    fn asks_user_name_matches_self_identity() {
        assert!(asks_user_name("как меня зовут"));
        assert!(asks_user_name("кто я?"));
        assert!(asks_user_name("кто я такой"));
        // Must NOT swallow a general-knowledge question about a person.
        assert!(!asks_user_name("кто такой дарио амодеи"));
        assert!(!asks_user_name("кто такой эйнштейн"));
    }

    #[test]
    fn asks_assistant_identity_matches_nic() {
        assert!(asks_assistant_identity("кто ты"));
        assert!(asks_assistant_identity("как тебя зовут"));
        // A "what's your X" should not be hijacked unless it's the bare identity.
        assert!(!asks_assistant_identity("кто такой дарио"));
        assert!(!asks_assistant_identity("кто я"));
    }
}
