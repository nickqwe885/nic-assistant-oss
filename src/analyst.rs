use crate::librarian::Librarian;
use crate::modules::thinker::Thinker;
use chrono::{Duration as ChronoDuration, Local, NaiveTime};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

const LAST_SUMMARY_FILE: &str = "data/last_summary_date.txt";

pub struct Analyst {
    librarian: Arc<Librarian>,
    thinker:   Arc<Mutex<Thinker>>,
    time:      NaiveTime,
    language:  String,
}

impl Analyst {
    pub fn new(
        librarian:    Arc<Librarian>,
        thinker:      Arc<Mutex<Thinker>>,
        summary_time: &str,
        language:     &str,
    ) -> Self {
        let time = NaiveTime::parse_from_str(summary_time, "%H:%M")
            .unwrap_or_else(|_| NaiveTime::from_hms_opt(18, 0, 0).unwrap());
        Self { librarian, thinker, time, language: language.to_string() }
    }

    pub fn start(self) {
        tokio::spawn(async move {
            if self.missed_today() {
                info!("[Analyst] Missed today's summary — generating now");
                self.generate_and_store().await;
            }
            loop {
                let secs = self.secs_until_next();
                info!("[Analyst] Next summary in {:.0} min", secs as f64 / 60.0);
                tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                self.generate_and_store().await;
            }
        });
    }

    fn missed_today(&self) -> bool {
        let today = Local::now().format("%Y-%m-%d").to_string();
        let last  = std::fs::read_to_string(LAST_SUMMARY_FILE).unwrap_or_default();
        check_missed_today(&today, last.trim(), Local::now().time(), self.time)
    }

    fn secs_until_next(&self) -> u64 {
        compute_secs_until(Local::now().naive_local(), self.time)
    }

    async fn generate_and_store(&self) {
        let today = Local::now().format("%Y-%m-%d").to_string();

        let raw = match self.librarian.daily_activity_summary().await {
            Ok(r) if !r.is_empty() => r,
            Ok(_) => {
                warn!("[Analyst] No activity today — summary skipped");
                self.save_date(&today);
                return;
            }
            Err(e) => {
                warn!("[Analyst] Librarian error: {}", e);
                return;
            }
        };

        let prompt = format!(
            "<|im_start|>system\n\
             You are a productivity analyst. Write strictly in {language}.\n\
             Use ONLY data from the activity below — do not invent anything.\n\
             Response format (strict, no extra words):\n\
             FOCUS: [2-3 main tasks of the day, comma-separated]\n\
             SUMMARY: [1-2 sentences — what was done, what remains unfinished]\n\
             TOMORROW: [1 concrete action worth doing tomorrow]<|im_end|>\n\
             <|im_start|>user\n[ACTIVITY FOR {today}]\n{activity}<|im_end|>\n\
             <|im_start|>assistant\n",
            language = self.language,
            today    = today,
            activity = raw.chars().take(1500).collect::<String>(),
        );

        let thinker = self.thinker.clone();
        let result = tokio::task::spawn_blocking(move || {
            thinker.blocking_lock().generate_raw(&prompt, 300)
        }).await;

        match result {
            Ok(Ok(summary)) => {
                info!("[Analyst] Summary ready ({} chars)", summary.len());
                let record = format!("[DAILY SUMMARY — {}]\n{}", today, summary);
                let _ = self.librarian.add_event(&record, "analyst", "daily_summary").await;
                self.save_date(&today);
                println!("\n╔═══ NIS Daily Summary ════════════════════╗");
                println!("║ {}", today);
                println!("╠══════════════════════════════════════════╣");
                for line in summary.lines() {
                    println!("║ {}", line);
                }
                println!("╚══════════════════════════════════════════╝\n");
                // Desktop notification — non-blocking, ignore failures
                let preview = summary.chars().take(120).collect::<String>();
                std::thread::spawn(move || {
                    let _ = notify_rust::Notification::new()
                        .summary("nic-assistant — Daily summary ready")
                        .body(&preview)
                        .appname("nic-assistant")
                        .show();
                });
            }
            Ok(Err(e)) => warn!("[Analyst] LLM error: {}", e),
            Err(e)     => warn!("[Analyst] spawn_blocking panicked: {}", e),
        }
    }

    fn save_date(&self, date: &str) {
        let _ = std::fs::create_dir_all("data");
        let _ = std::fs::write(LAST_SUMMARY_FILE, date);
    }
}

// ── Pure helpers (extracted for testability) ──────────────────────────────────

fn compute_secs_until(now: chrono::NaiveDateTime, target_time: NaiveTime) -> u64 {
    let today_at = now.date().and_time(target_time);
    let target   = if today_at > now { today_at } else { today_at + ChronoDuration::days(1) };
    (target - now).num_seconds().max(60) as u64
}

fn check_missed_today(today: &str, last: &str, now_time: NaiveTime, summary_time: NaiveTime) -> bool {
    last != today && now_time >= summary_time
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, Timelike};

    fn nd(h: u32, m: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, 0).unwrap()
    }

    fn ndt(y: i32, mo: u32, d: u32, h: u32, m: u32) -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(y, mo, d).unwrap().and_hms_opt(h, m, 0).unwrap()
    }

    // ── compute_secs_until ────────────────────────────────────────────────────

    #[test]
    fn secs_until_future_same_day() {
        // now = 10:00, target = 18:00 → 8 hours = 28800 s
        let now    = ndt(2025, 6, 1, 10, 0);
        let target = nd(18, 0);
        let secs   = compute_secs_until(now, target);
        assert_eq!(secs, 8 * 3600);
    }

    #[test]
    fn secs_until_past_same_day_rolls_over() {
        // now = 20:00, target = 18:00 → rolls to tomorrow 18:00 = 22 hours
        let now    = ndt(2025, 6, 1, 20, 0);
        let target = nd(18, 0);
        let secs   = compute_secs_until(now, target);
        assert_eq!(secs, 22 * 3600);
    }

    #[test]
    fn secs_until_exact_same_time_rolls_over() {
        // now == target → today_at == now, NOT > now → add 1 day → 24 h
        let now    = ndt(2025, 6, 1, 18, 0);
        let target = nd(18, 0);
        let secs   = compute_secs_until(now, target);
        assert_eq!(secs, 24 * 3600);
    }

    #[test]
    fn secs_until_minimum_60() {
        // now = 17:59:01, target = 18:00 → 59 s → clamped to 60
        let now = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap()
            .and_hms_opt(17, 59, 1).unwrap();
        let target = nd(18, 0);
        let secs = compute_secs_until(now, target);
        assert_eq!(secs, 60);
    }

    #[test]
    fn secs_until_1_second_before_clamped() {
        let now = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap()
            .and_hms_opt(17, 59, 59).unwrap();
        let target = nd(18, 0);
        let secs = compute_secs_until(now, target);
        assert_eq!(secs, 60); // 1s < 60 → clamped
    }

    #[test]
    fn secs_until_1_minute_before() {
        let now = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap()
            .and_hms_opt(17, 59, 0).unwrap();
        let target = nd(18, 0);
        let secs = compute_secs_until(now, target);
        assert_eq!(secs, 60);
    }

    #[test]
    fn secs_until_midnight_target() {
        // now = 23:00, target = 00:00 → 1 hour = 3600
        let now    = ndt(2025, 6, 1, 23, 0);
        let target = nd(0, 0);
        let secs   = compute_secs_until(now, target);
        assert_eq!(secs, 3600);
    }

    #[test]
    fn secs_until_morning_target_from_night() {
        // now = 02:00, target = 06:00 → 4 hours = 14400
        let now    = ndt(2025, 6, 1, 2, 0);
        let target = nd(6, 0);
        let secs   = compute_secs_until(now, target);
        assert_eq!(secs, 4 * 3600);
    }

    #[test]
    fn secs_until_return_type_u64_no_overflow() {
        // max future: 24h = 86400 — well within u64
        let now    = ndt(2025, 6, 1, 0, 1);
        let target = nd(0, 0);
        let secs   = compute_secs_until(now, target);
        assert!(secs <= 24 * 3600);
        assert!(secs >= 60);
    }

    // ── check_missed_today ────────────────────────────────────────────────────

    #[test]
    fn missed_same_date_not_missed() {
        assert!(!check_missed_today("2025-06-01", "2025-06-01", nd(20, 0), nd(18, 0)));
    }

    #[test]
    fn missed_different_date_after_time_is_missed() {
        assert!(check_missed_today("2025-06-01", "2025-05-31", nd(19, 0), nd(18, 0)));
    }

    #[test]
    fn missed_different_date_before_time_not_missed() {
        // It's 10:00 and summary is at 18:00 — not yet time, so not missed
        assert!(!check_missed_today("2025-06-01", "2025-05-31", nd(10, 0), nd(18, 0)));
    }

    #[test]
    fn missed_exact_summary_time_is_missed() {
        // now_time == summary_time → >= holds → missed
        assert!(check_missed_today("2025-06-01", "2025-05-31", nd(18, 0), nd(18, 0)));
    }

    #[test]
    fn missed_empty_last_date_after_time_is_missed() {
        // Fresh install: last = "" → never ran → missed if past time
        assert!(check_missed_today("2025-06-01", "", nd(20, 0), nd(18, 0)));
    }

    #[test]
    fn missed_empty_last_date_before_time_not_missed() {
        assert!(!check_missed_today("2025-06-01", "", nd(10, 0), nd(18, 0)));
    }

    #[test]
    fn missed_whitespace_trimmed_last_not_missed() {
        // caller trims before passing — "2025-06-01" == "2025-06-01"
        assert!(!check_missed_today("2025-06-01", "2025-06-01", nd(20, 0), nd(18, 0)));
    }

    // ── NaiveTime::parse_from_str fallback in Analyst::new ───────────────────

    #[test]
    fn analyst_time_parse_valid() {
        let t = NaiveTime::parse_from_str("18:30", "%H:%M").unwrap();
        assert_eq!(t.hour(), 18);
        assert_eq!(t.minute(), 30);
    }

    #[test]
    fn analyst_time_parse_invalid_falls_back_to_18_00() {
        let t = NaiveTime::parse_from_str("invalid", "%H:%M")
            .unwrap_or_else(|_| NaiveTime::from_hms_opt(18, 0, 0).unwrap());
        assert_eq!(t.hour(), 18);
        assert_eq!(t.minute(), 0);
    }

    #[test]
    fn analyst_time_parse_empty_falls_back() {
        let t = NaiveTime::parse_from_str("", "%H:%M")
            .unwrap_or_else(|_| NaiveTime::from_hms_opt(18, 0, 0).unwrap());
        assert_eq!(t.hour(), 18);
    }

    #[test]
    fn analyst_time_parse_midnight() {
        let t = NaiveTime::parse_from_str("00:00", "%H:%M").unwrap();
        assert_eq!(t.hour(), 0);
        assert_eq!(t.minute(), 0);
    }

    #[test]
    fn analyst_time_parse_23_59() {
        let t = NaiveTime::parse_from_str("23:59", "%H:%M").unwrap();
        assert_eq!(t.hour(), 23);
        assert_eq!(t.minute(), 59);
    }
}
