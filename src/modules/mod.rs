pub mod anaphora;
pub mod librarian;
pub mod media_router;
pub mod pilot;
pub mod scrubber;
pub mod sentinel;
pub mod shakal;
pub mod surfer;
pub mod thinker;

use crate::librarian::Librarian;
use surfer::Surfer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{info, warn};

// ── Sentinel → Librarian channel event type ───────────────────────────────────

pub enum SentinelEvent {
    Screen {
        app_name:     String,
        window_title: String,
        description:  String,
    },
    Clipboard {
        text: String,
    },
}

// ── Performance / Hibernation gate ────────────────────────────────────────────

/// `true`  → CPU > 70 % — Thinker hibernates, Sentinel skips capture.
/// `false` → normal operation.
pub static PERFORMANCE_MODE: AtomicBool = AtomicBool::new(false);

pub fn start_performance_monitor(threshold: f32) {
    tokio::spawn(async move {
        let mut sys = sysinfo::System::new();
        sys.refresh_cpu_all();
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            sys.refresh_cpu_all();
            let cpu  = sys.global_cpu_usage();
            let high = cpu > threshold;
            let was  = PERFORMANCE_MODE.swap(high, Ordering::Relaxed);

            if high && !was {
                warn!("[Performance] CPU {:.0}% > {:.0}% — Thinker hibernating", cpu, threshold);
            } else if !high && was {
                info!("[Performance] CPU {:.0}% — Thinker resuming", cpu);
            }
        }
    });
}

// ── Context Collector (ICT: Internal Chain of Thought) ────────────────────────

pub struct ContextCollector {
    librarian: Arc<Librarian>,
    surfer:    Arc<Surfer>,
}

impl ContextCollector {
    pub fn new(librarian: Arc<Librarian>, surfer: Arc<Surfer>) -> Self {
        Self { librarian, surfer }
    }

    /// Parallel context collection: Librarian (Topic Timeline RAG) + Surfer (web).
    /// Both run concurrently via tokio::join!. Surfer has a 20-second hard timeout.
    /// Returns `(lib_ctx, surf_ctx)`.
    pub async fn collect(&self, query: &str, offline: bool) -> (String, String) {
        self.collect_with(query, offline, false).await
    }

    /// As `collect`, but `force_web` bypasses the Surfer's keyword allowlist.
    /// A bare name ("donk") contains none of its trigger words, so a question about
    /// a person never reached the internet and the model filled the gap with
    /// fiction. When the caller already knows a person is being asked about, the
    /// decision is made — go and look.
    pub async fn collect_with(
        &self,
        query:     &str,
        offline:   bool,
        force_web: bool,
    ) -> (String, String) {
        let librarian = self.librarian.clone();
        let surfer    = self.surfer.clone();
        let q         = query.to_string();
        // Activity-recall questions ("what did I do / what happened") must answer
        // from the screen timeline only — skip the web so an empty "Web results"
        // section can't distract the model into "nothing found online".
        let skip_web  = offline || crate::librarian::is_activity_recall(&q);

        let lib_fut = async {
            // On error, return the empty-context marker so the bundle treats it as
            // "no screen history" rather than feeding a Russian error string to the
            // English-only model (which nudged it into code-switching).
            librarian.topic_timeline_rag(&q).await
                .unwrap_or_else(|_| "Context is empty.".to_string())
        };

        let surf_fut = async {
            if skip_web {
                return String::new();
            }
            let search = async {
                if force_web {
                    surfer.search_web(&q).await
                } else {
                    surfer.maybe_search_web(&q, false).await
                }
            };
            match tokio::time::timeout(Duration::from_secs(20), search).await {
                Ok(Some(s)) => s,
                Ok(None)    => String::new(),
                Err(_)      => {
                    println!("[Surfer] 20s timeout — continuing without web data.");
                    String::new()
                }
            }
        };

        tokio::join!(lib_fut, surf_fut)
    }

    /// Formats the ICT Context Bundle passed to Thinker.
    /// Detects potential conflicts between local and web data and appends a notice.
    pub fn format_context_bundle(
        &self,
        lib_ctx:  &str,
        surf_ctx: &str,
    ) -> String {
        let now = chrono::Local::now();
        let time_str = now.format("%d.%m.%Y %H:%M").to_string();

        let lib_empty = lib_ctx.is_empty()
            || lib_ctx.contains("Context is empty")
            || lib_ctx.contains("no relevant");

        // Secret shield, second layer: memory captured BEFORE the scrubber
        // shipped may still hold a seed phrase / card / key. Redact on the way
        // OUT too, so nothing secret can ever reach the model — and therefore
        // the answer — regardless of when it was recorded.
        let lib_clean = if lib_empty {
            String::new()
        } else {
            scrubber::scrub(&strip_internal_markers(&clean_ocr_noise(lib_ctx)))
        };

        // Label it unmistakably. Given a bare "Date and time: …", the model read it
        // as its own training cutoff and answered "this may have changed since my
        // last update on July 13th 2026 at 10:37 AM" — nonsense that reads like a
        // broken product.
        let mut parts: Vec<String> = vec![format!(
            "Right now it is {time_str} for the user. This is the current clock time — \
             it is NOT your knowledge cutoff, so never describe it as your last update."
        )];

        if !lib_clean.is_empty() {
            parts.push(format!("Screen history:\n{lib_clean}"));
        }

        if !surf_ctx.is_empty() {
            parts.push(format!("Web results:\n{surf_ctx}"));
        }

        if let Some(conflict) = detect_conflicts(&lib_clean, surf_ctx) {
            parts.push(conflict);
        }

        parts.join("\n\n")
    }
}

/// Removes internal RAG/section markers from the screen-history context so the
/// small model can't paste them back to the user (it sometimes echoes the raw
/// context verbatim — markdown headers, "SCREEN" tags, level labels). Keeps the
/// actual event text.
fn strip_internal_markers(s: &str) -> String {
    s.lines()
        .filter(|l| !l.trim_start().starts_with("##"))
        .map(|l| l.replace("SCREEN ", "")
                  .replace("(level 1+)", "")
                  .replace("level 1+", ""))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strips OCR garbage characters from screen-capture text.
/// Keeps Cyrillic, Latin, digits, and common punctuation; removes noise glyphs.
fn clean_ocr_noise(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            c.is_alphabetic()
                || c.is_ascii_digit()
                || " \n\t.,;:!?-—()\"'«»/\\@#%&+=<>_".contains(c)
        })
        .collect()
}

// ── Conflict detection ────────────────────────────────────────────────────────

/// Detects potential factual conflicts between local memory and web search results.
///
/// Strategy: extract (word, adjacent-token) pairs where the adjacent token contains
/// a digit (version numbers, dates, measurements). If the same anchor word appears
/// in both contexts with different adjacent values, flag it.
///
/// Returns a formatted notice or None if no contradictions are found.
fn detect_conflicts(lib_ctx: &str, web_ctx: &str) -> Option<String> {
    if web_ctx.is_empty() || lib_ctx.is_empty() {
        return None;
    }

    fn key_facts(text: &str) -> std::collections::HashMap<String, String> {
        let mut facts = std::collections::HashMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        for pair in words.windows(2) {
            // Second token must contain a digit (version, year, count, etc.)
            if !pair[1].chars().any(|c| c.is_ascii_digit()) { continue; }
            let key = pair[0]
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            // Skip short / noise words
            if key.len() < 4 { continue; }
            facts.entry(key).or_insert_with(|| pair[1].to_string());
        }
        facts
    }

    let lib_facts = key_facts(lib_ctx);
    let web_facts = key_facts(web_ctx);

    let conflicts: Vec<String> = lib_facts
        .iter()
        .filter_map(|(key, lib_val)| {
            let web_val = web_facts.get(key)?;
            if web_val == lib_val { return None; }
            Some(format!("«{}»: local={}, web={}", key, lib_val, web_val))
        })
        .take(5)
        .collect();

    if conflicts.is_empty() {
        None
    } else {
        Some(format!(
            "Data mismatch (history vs web): {}. Check the source before answering.",
            conflicts.join("; ")
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── clean_ocr_noise ───────────────────────────────────────────────────────

    #[test]
    fn ocr_keeps_cyrillic() {
        let r = clean_ocr_noise("Hello world");
        assert_eq!(r, "Hello world");
    }

    #[test]
    fn ocr_keeps_latin() {
        let r = clean_ocr_noise("Hello World");
        assert_eq!(r, "Hello World");
    }

    #[test]
    fn ocr_keeps_digits() {
        let r = clean_ocr_noise("42 answers");
        assert_eq!(r, "42 answers");
    }

    #[test]
    fn ocr_keeps_allowed_punctuation() {
        let r = clean_ocr_noise(".,;:!?-—()\"'«»/\\@#%&+=<>_");
        assert_eq!(r, ".,;:!?-—()\"'«»/\\@#%&+=<>_");
    }

    #[test]
    fn ocr_removes_noise_chars() {
        let r = clean_ocr_noise("Hello\u{2022}world\u{00B0}test"); // • and °
        assert!(!r.contains('\u{2022}'), "bullet should be removed");
        assert!(!r.contains('\u{00B0}'), "degree symbol should be removed");
        assert!(r.contains("Hello"));
        assert!(r.contains("world"));
        assert!(r.contains("test"));
    }

    #[test]
    fn ocr_removes_box_drawing_chars() {
        let r = clean_ocr_noise("key\u{2502}value"); // │
        assert!(!r.contains('\u{2502}'));
    }

    #[test]
    fn ocr_empty_input() {
        assert_eq!(clean_ocr_noise(""), "");
    }

    #[test]
    fn ocr_keeps_newline_tab() {
        let r = clean_ocr_noise("line1\nline2\ttab");
        assert!(r.contains('\n'));
        assert!(r.contains('\t'));
    }

    #[test]
    fn ocr_pure_noise_becomes_empty() {
        let r = clean_ocr_noise("\u{2022}\u{00B7}\u{25A0}\u{25CF}"); // ••□●
        assert_eq!(r, "");
    }

    #[test]
    fn ocr_mixed_code_text() {
        let r = clean_ocr_noise("fn main() { println!(\"hello\"); }");
        assert!(r.contains("fn"));
        assert!(r.contains("main"));
        assert!(r.contains("println"));
    }

    // ── detect_conflicts ──────────────────────────────────────────────────────

    #[test]
    fn conflicts_both_empty_returns_none() {
        assert!(detect_conflicts("", "").is_none());
    }

    #[test]
    fn conflicts_lib_empty_returns_none() {
        assert!(detect_conflicts("", "rust version 2.0").is_none());
    }

    #[test]
    fn conflicts_web_empty_returns_none() {
        assert!(detect_conflicts("rust version 1.0", "").is_none());
    }

    #[test]
    fn conflicts_agreement_returns_none() {
        let lib = "rust version 1.75";
        let web = "rust version 1.75";
        assert!(detect_conflicts(lib, web).is_none());
    }

    #[test]
    fn conflicts_version_mismatch_detected() {
        let lib = "rust version 1.70";
        let web = "rust version 1.80";
        let r = detect_conflicts(lib, web);
        assert!(r.is_some(), "version mismatch should be detected");
        let msg = r.unwrap();
        assert!(msg.contains("Data mismatch"));
        assert!(msg.contains("rust") || msg.contains("version"));
    }

    #[test]
    fn conflicts_no_digits_in_either_returns_none() {
        let lib = "machine learning is a technology";
        let web = "machine learning is a science";
        // No digit-containing tokens → no key_facts → no conflicts
        assert!(detect_conflicts(lib, web).is_none());
    }

    #[test]
    fn conflicts_message_format() {
        let lib = "population 8000000000";
        let web = "population 7900000000";
        let r = detect_conflicts(lib, web);
        if let Some(msg) = r {
            assert!(msg.contains("Data mismatch"));
            assert!(msg.contains("Check the source"));
        }
    }

    #[test]
    fn conflicts_capped_at_5() {
        // Create many conflicts
        let lib = "aaa 1 bbb 2 ccc 3 ddd 4 eee 5 fff 6 ggg 7";
        let web = "aaa 9 bbb 8 ccc 7 ddd 6 eee 5 fff 4 ggg 3";
        let r = detect_conflicts(lib, web);
        if let Some(msg) = r {
            // Count "local=" occurrences ≤ 5
            let count = msg.matches("local=").count();
            assert!(count <= 5, "should show at most 5 conflicts, got {}", count);
        }
    }

    #[test]
    fn conflicts_short_anchor_words_ignored() {
        // "the 2024" — "the" has len < 4 chars (3 ASCII bytes < 4) → ignored
        let lib = "the 2024 year";
        let web = "the 2025 year";
        // "the" = 3 bytes < 4 → filtered out → no conflict
        // "year" is 4 bytes but no digit follows "year"
        // So no conflicts detected
        assert!(detect_conflicts(lib, web).is_none());
    }

    #[test]
    fn conflicts_same_anchor_same_value_no_conflict() {
        let lib = "price 100 dollars";
        let web = "price 100 dollars";
        assert!(detect_conflicts(lib, web).is_none());
    }

}
