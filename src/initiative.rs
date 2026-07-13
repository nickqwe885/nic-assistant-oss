/// InitiativeLoop — autonomous activity analyser.
///
/// Every 10 minutes it pulls a recent-activity summary from the Librarian
/// and asks the Thinker whether there is an urgent task or reminder for the user.
///
/// If the reply carries an `[ACTION: REMIND("text")]` marker it raises a
/// desktop notification via notify-rust. Otherwise (IDLE) it stays silent.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};
use tracing::{info, warn};

use crate::librarian::Librarian;
use crate::modules::thinker::Thinker;

const INTERVAL_SECS: u64 = 600; // 10 minutes
/// Hard cap: at most this many proactive reminders per day. Initiative must be
/// rare and dismissible, never a nagging "Clippy".
const MAX_PER_DAY: usize = 2;

/// Local day number, for resetting the daily reminder counter at midnight.
fn current_day() -> i64 {
    chrono::Local::now().timestamp() / 86_400
}

pub fn start(
    librarian: Arc<Librarian>,
    thinker:   Arc<Mutex<Thinker>>,
    language:  String,
    enabled:   bool,
    incognito: Arc<AtomicBool>,
) {
    if !enabled {
        info!("[Initiative] disabled by config — no proactive reminders");
        return;
    }
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(INTERVAL_SECS));
        ticker.tick().await; // skip first tick (right at startup)

        // Daily rate-limit + de-dup so we never repeat or spam.
        let mut day:        i64         = current_day();
        let mut sent_today: Vec<String> = Vec::new();

        loop {
            ticker.tick().await;

            // Respect incognito: if the user paused screen memory, stay silent.
            if incognito.load(Ordering::Relaxed) { continue; }

            let today = current_day();
            if today != day { day = today; sent_today.clear(); }
            if sent_today.len() >= MAX_PER_DAY { continue; }

            if let Some(text) = run_once(&librarian, &thinker, &language).await {
                if !sent_today.iter().any(|t| t == &text) {
                    info!("[Initiative] Reminder ({}/{}): {}", sent_today.len() + 1, MAX_PER_DAY, text);
                    show_notification(&text);
                    sent_today.push(text);
                }
            }
        }
    });
}

/// Runs one analysis pass; returns the reminder text if there is a genuine,
/// specific one, else `None`. Does NOT show the notification itself — the caller
/// rate-limits and de-dups before deciding to surface it.
async fn run_once(
    librarian: &Arc<Librarian>,
    thinker:   &Arc<Mutex<Thinker>>,
    language:  &str,
) -> Option<String> {
    let activity = match librarian.daily_activity_summary().await {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => return None,
        Err(e) => { warn!("[Initiative] Librarian error: {}", e); return None; }
    };

    let prompt = build_prompt(&activity, language);
    let thinker_ref = thinker.clone();

    let response = tokio::task::spawn_blocking(move || {
        thinker_ref.blocking_lock().generate_raw(&prompt, 80)
    })
    .await;

    match response {
        Ok(Ok(text)) => parse_remind(text.trim()),
        Ok(Err(e)) => { warn!("[Initiative] LLM error: {}", e); None }
        Err(e)     => { warn!("[Initiative] spawn_blocking panic: {}", e); None }
    }
}

fn build_prompt(activity: &str, language: &str) -> String {
    let activity_trimmed: String = activity.chars().take(1200).collect();
    format!(
        "<|im_start|>system\n\
         You are a background monitor. Analyse the user's activity.\n\
         Reply with EXACTLY ONE of these two options and nothing else:\n\
         [ACTION: REMIND(\"text\")] — only if the activity contains a SPECIFIC missed task\n\
         or a deadline within the next 2 hours (meeting, submission, call, payment).\n\
         IDLE — in all other cases, when in doubt, or when data is insufficient.\n\
         The reminder text must be in {language}, maximum 10 words, specific and to the point.<|im_end|>\n\
         <|im_start|>user\n\
         [ACTIVITY]\n{activity}<|im_end|>\n\
         <|im_start|>assistant\n",
        language = language,
        activity = activity_trimmed,
    )
}

/// Parses `[ACTION: REMIND("...")]` out of the LLM reply.
fn parse_remind(text: &str) -> Option<String> {
    let prefix = "[ACTION: REMIND(\"";
    let suffix = "\")]";
    let start = text.find(prefix)?;
    let after = &text[start + prefix.len()..];
    let end   = after.find(suffix)?;
    let msg   = after[..end].trim().to_string();
    if msg.is_empty() { None } else { Some(msg) }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_remind ──────────────────────────────────────────────────────────

    #[test]
    fn remind_basic_extraction() {
        let r = parse_remind(r#"[ACTION: REMIND("meeting at 15:00")]"#);
        assert_eq!(r, Some("meeting at 15:00".to_string()));
    }

    #[test]
    fn remind_idle_returns_none() {
        assert!(parse_remind("IDLE").is_none());
    }

    #[test]
    fn remind_empty_string_none() {
        assert!(parse_remind("").is_none());
    }

    #[test]
    fn remind_empty_body_returns_none() {
        // Empty after trim → None
        let r = parse_remind(r#"[ACTION: REMIND("")]"#);
        assert!(r.is_none());
    }

    #[test]
    fn remind_whitespace_only_body_returns_none() {
        let r = parse_remind("[ACTION: REMIND(\"   \")]");
        assert!(r.is_none(), "whitespace-only reminder should be None");
    }

    #[test]
    fn remind_body_trimmed() {
        let r = parse_remind("[ACTION: REMIND(\"  check email  \")]");
        assert_eq!(r, Some("check email".to_string()));
    }

    #[test]
    fn remind_embedded_in_longer_text() {
        let text = "I see a task. [ACTION: REMIND(\"call the client\")] Thanks.";
        assert_eq!(parse_remind(text), Some("call the client".to_string()));
    }

    #[test]
    fn remind_missing_prefix_none() {
        assert!(parse_remind("REMIND(\"test\")").is_none());
    }

    #[test]
    fn remind_missing_suffix_none() {
        assert!(parse_remind("[ACTION: REMIND(\"test\"").is_none());
    }

    #[test]
    fn remind_uses_first_occurrence() {
        let text = r#"[ACTION: REMIND("first")] and [ACTION: REMIND("second")]"#;
        assert_eq!(parse_remind(text), Some("first".to_string()));
    }

    #[test]
    fn remind_cyrillic_message() {
        let r = parse_remind(r#"[ACTION: REMIND("Close the Jira ticket by Friday")]"#);
        assert_eq!(r, Some("Close the Jira ticket by Friday".to_string()));
    }

    #[test]
    fn remind_message_with_special_chars() {
        let r = parse_remind("[ACTION: REMIND(\"code review @ 14:30 — urgent!\")]");
        assert_eq!(r, Some("code review @ 14:30 — urgent!".to_string()));
    }

    #[test]
    fn remind_only_idle_pattern_returns_none() {
        for s in &["IDLE", "idle", "  IDLE  ", "Ok, IDLE.", "Nothing to do."] {
            assert!(parse_remind(s).is_none(), "expected None for: {}", s);
        }
    }

    #[test]
    fn remind_no_closing_bracket_none() {
        assert!(parse_remind(r#"[ACTION: REMIND("test")"#).is_none());
    }

    #[test]
    fn remind_unicode_emoji_in_message() {
        let r = parse_remind("[ACTION: REMIND(\"deadline 🔥 today\")]");
        assert_eq!(r, Some("deadline 🔥 today".to_string()));
    }

    #[test]
    fn remind_long_message_extracted() {
        let long = "a".repeat(500);
        let input = format!("[ACTION: REMIND(\"{}\")]", long);
        let r = parse_remind(&input);
        assert!(r.is_some());
        assert_eq!(r.unwrap().len(), 500);
    }

    // ── build_prompt ──────────────────────────────────────────────────────────

    #[test]
    fn build_prompt_contains_activity() {
        let p = build_prompt("Rust code is open", "English");
        assert!(p.contains("Rust code is open"));
    }

    #[test]
    fn build_prompt_contains_system_marker() {
        let p = build_prompt("test", "English");
        assert!(p.contains("<|im_start|>system"));
        assert!(p.contains("<|im_start|>user"));
        assert!(p.contains("<|im_start|>assistant"));
    }

    #[test]
    fn build_prompt_contains_remind_instruction() {
        let p = build_prompt("test", "English");
        assert!(p.contains("REMIND") || p.contains("ACTION"));
    }

    #[test]
    fn build_prompt_contains_idle_instruction() {
        let p = build_prompt("test", "English");
        assert!(p.contains("IDLE"));
    }

    #[test]
    fn build_prompt_contains_language() {
        let p = build_prompt("test", "German");
        assert!(p.contains("German"));
    }

    #[test]
    fn build_prompt_truncates_activity_at_1200() {
        // Use '§' — never appears in the prompt template, so the count is exact.
        let long_activity = "§".repeat(2000);
        let p = build_prompt(&long_activity, "English");
        let count = p.chars().filter(|&c| c == '§').count();
        assert!(count <= 1200, "activity should be truncated to 1200 chars, got {}", count);
    }

    #[test]
    fn build_prompt_exact_1200_not_truncated() {
        let exactly_1200 = "§".repeat(1200);
        let p = build_prompt(&exactly_1200, "English");
        assert!(p.contains(&exactly_1200), "1200-char activity should not be truncated");
    }

    #[test]
    fn build_prompt_empty_activity() {
        let p = build_prompt("", "English");
        assert!(p.contains("<|im_start|>"));
    }

    #[test]
    fn build_prompt_no_panic_unicode() {
        let _ = build_prompt("Today I worked with Rust 🦀 and LanceDB", "English");
    }
}

fn show_notification(text: &str) {
    let body = text.to_string();
    std::thread::spawn(move || {
        let _ = notify_rust::Notification::new()
            .summary("nic-assistant — Reminder")
            .body(&body)
            .appname("nic-assistant")
            .show();
    });
}
