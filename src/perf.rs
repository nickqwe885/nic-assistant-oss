/// PerformanceAggregator — scoped rolling statistics for the NIS performance passport.
///
/// Two independent instances keep real and synthetic data completely separate:
///   `realtime()` — populated exclusively by live Sentinel captures
///   `stress()`   — populated during /stress_test_long, reset between runs
///
/// `global()` dispatches to the active instance via the `STRESS_MODE` flag.
/// Calling `set_stress_mode(true)` before a stress run ensures no synthetic
/// data contaminates the production histogram.
///
/// Per-instance overhead: ~2.4 KB RAM (3×VecDeque<100 × u128> + 2 atomics).

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::time::Instant;

const WINDOW:      usize = 100;
const HIST_WINDOW: usize = 50;  // last N real captures for the OCR histogram

static REALTIME:    OnceLock<PerfAggregator> = OnceLock::new();
static STRESS:      OnceLock<PerfAggregator> = OnceLock::new();
static STRESS_MODE: AtomicBool               = AtomicBool::new(false);

// ── Public routing API ────────────────────────────────────────────────────────

/// Routes to `stress()` when a stress test is active, `realtime()` otherwise.
pub fn global() -> &'static PerfAggregator {
    if STRESS_MODE.load(Ordering::Relaxed) { stress() } else { realtime() }
}

/// Live production statistics — only real Sentinel captures.
pub fn realtime() -> &'static PerfAggregator {
    REALTIME.get_or_init(PerfAggregator::new)
}

/// Synthetic stress-test statistics — populated only during /stress_test_long.
pub fn stress() -> &'static PerfAggregator {
    STRESS.get_or_init(PerfAggregator::new)
}

/// Activates stress mode — subsequent `global()` calls route to `stress()`.
pub fn set_stress_mode(active: bool) {
    STRESS_MODE.store(active, Ordering::Relaxed);
}

/// Returns true when a stress test is currently running.
pub fn is_stress_mode() -> bool {
    STRESS_MODE.load(Ordering::Relaxed)
}

// ── Aggregator ────────────────────────────────────────────────────────────────

pub struct PerfAggregator {
    ocr_ms:       Mutex<VecDeque<u128>>,
    store_ms:     Mutex<VecDeque<u128>>,
    ttft_ms:      Mutex<VecDeque<u128>>,
    total_frames: AtomicU64,
    chronos_hits: AtomicU64,
    start:        Instant,
}

impl PerfAggregator {
    fn new() -> Self {
        Self {
            ocr_ms:       Mutex::new(VecDeque::with_capacity(WINDOW)),
            store_ms:     Mutex::new(VecDeque::with_capacity(WINDOW)),
            ttft_ms:      Mutex::new(VecDeque::with_capacity(WINDOW)),
            total_frames: AtomicU64::new(0),
            chronos_hits: AtomicU64::new(0),
            start:        Instant::now(),
        }
    }

    pub fn record_ocr(&self, ms: u128) {
        push_window(&self.ocr_ms, ms);
    }

    /// Records one Store/Embed call. Pass `is_fresh = false` when Chronos-Link fired.
    pub fn record_store(&self, ms: u128, is_fresh: bool) {
        push_window(&self.store_ms, ms);
        self.total_frames.fetch_add(1, Ordering::Relaxed);
        if !is_fresh {
            self.chronos_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_ttft(&self, ms: u128) {
        push_window(&self.ttft_ms, ms);
    }

    /// OCR latency distribution computed live from the last `HIST_WINDOW` (50) samples.
    /// Returns (0-49ms count, 50-149ms count, >=150ms count, total samples used).
    fn ocr_histogram_50(&self) -> (u64, u64, u64, usize) {
        let q      = self.ocr_ms.lock().unwrap_or_else(|e| e.into_inner());
        let window: Vec<u128> = q.iter().rev().take(HIST_WINDOW).copied().collect();
        let n    = window.len();
        let fast = window.iter().filter(|&&ms| ms < 50).count() as u64;
        let mid  = window.iter().filter(|&&ms| ms >= 50 && ms < 150).count() as u64;
        let slow = window.iter().filter(|&&ms| ms >= 150).count() as u64;
        (fast, mid, slow, n)
    }

    #[allow(dead_code)]
    pub fn stats_report(&self) -> String {
        self.stats_report_titled("NIS Performance Passport")
    }

    /// Builds a 57-char-wide box table. `title` is centred in the header row (≤ 53 chars).
    pub fn stats_report_titled(&self, title: &str) -> String {
        let (oa, o95, o99, on) = compute_stats(
            &self.ocr_ms.lock().unwrap_or_else(|e| e.into_inner()));
        let (sa, s95, s99, sn) = compute_stats(
            &self.store_ms.lock().unwrap_or_else(|e| e.into_inner()));
        let (ta, t95, t99, tn) = compute_stats(
            &self.ttft_ms.lock().unwrap_or_else(|e| e.into_inner()));

        let total = self.total_frames.load(Ordering::Relaxed);
        let hits  = self.chronos_hits.load(Ordering::Relaxed);
        let stab  = if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 };

        let secs = self.start.elapsed().as_secs();
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;

        let ram = query_ram_mb();

        let (fast, mid, slow, hist_n) = self.ocr_histogram_50();
        let hn = hist_n.max(1) as f64;

        let mut out = String::with_capacity(1024);
        out.push_str("┌─────────────────────────────────────────────────────────┐\n");
        out.push_str(&format!("│ {:^53} │\n", title));
        out.push_str("├────────────────┬──────────┬──────────┬──────────┬─────┤\n");
        out.push_str(&format!("│ {:<14} │{:^10}│{:^10}│{:^10}│{:^5}│\n",
            "Metric", "Avg", "P95", "P99", "N"));
        out.push_str("├────────────────┼──────────┼──────────┼──────────┼─────┤\n");
        out.push_str(&metric_row("OCR",          oa, o95, o99, on));
        out.push_str(&metric_row("Store/Embed",  sa, s95, s99, sn));
        out.push_str(&metric_row("Thinker TTFT", ta, t95, t99, tn));
        out.push_str("├────────────────┴──────────┴──────────┴──────────┴─────┤\n");
        out.push_str(&info_row(&format!("Total frames  : {}", total)));
        out.push_str(&info_row(&format!(
            "Chronos hits  : {}  ({:.1}% stability / DB saved)", hits, stab)));
        out.push_str(&info_row(&format!("Process RAM   : {} MB", ram)));
        out.push_str(&info_row(&format!("Uptime        : {:02}h {:02}m {:02}s", h, m, s)));
        out.push_str("├─────────────────────────────────────────────────────────┤\n");
        out.push_str(&info_row(&format!(
            "OCR Histogram (last {} real captures):", hist_n)));
        out.push_str(&info_row(&format!(
            "  0-49 ms   : {:>4}  ({:.1}%)", fast, fast as f64/hn*100.0)));
        out.push_str(&info_row(&format!(
            "  50-149 ms : {:>4}  ({:.1}%)", mid,  mid  as f64/hn*100.0)));
        out.push_str(&info_row(&format!(
            "  >=150 ms  : {:>4}  ({:.1}%)", slow, slow as f64/hn*100.0)));
        out.push_str("└─────────────────────────────────────────────────────────┘");
        out
    }

    pub fn perf_json_value(&self) -> serde_json::Value {
        let (oa, o95, o99, on) = compute_stats(
            &self.ocr_ms.lock().unwrap_or_else(|e| e.into_inner()));
        let (sa, s95, s99, sn) = compute_stats(
            &self.store_ms.lock().unwrap_or_else(|e| e.into_inner()));
        let (ta, t95, t99, tn) = compute_stats(
            &self.ttft_ms.lock().unwrap_or_else(|e| e.into_inner()));

        let total = self.total_frames.load(Ordering::Relaxed);
        let hits  = self.chronos_hits.load(Ordering::Relaxed);
        let stab  = if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 };
        let (fast, mid, slow, _) = self.ocr_histogram_50();
        let ram  = query_ram_mb();
        let secs = self.start.elapsed().as_secs();

        serde_json::json!({
            "ocr":   { "avg_ms": oa,  "p95_ms": o95, "p99_ms": o99, "n": on  },
            "store": { "avg_ms": sa,  "p95_ms": s95, "p99_ms": s99, "n": sn  },
            "ttft":  { "avg_ms": ta,  "p95_ms": t95, "p99_ms": t99, "n": tn  },
            "stability_pct":  (stab * 10.0).round() / 10.0,
            "total_frames":   total,
            "chronos_hits":   hits,
            "ocr_histogram": {
                "0_50ms":     fast,
                "50_150ms":   mid,
                "150ms_plus": slow
            },
            "process_ram_mb": ram,
            "uptime_secs":    secs
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn push_window(w: &Mutex<VecDeque<u128>>, ms: u128) {
    let mut lock = w.lock().unwrap_or_else(|e| e.into_inner());
    if lock.len() >= WINDOW { lock.pop_front(); }
    lock.push_back(ms);
}

/// Returns (avg, p95, p99, n). Returns (0,0,0,0) for empty input.
fn compute_stats(data: &VecDeque<u128>) -> (u128, u128, u128, usize) {
    let n = data.len();
    if n == 0 { return (0, 0, 0, 0); }
    let avg = data.iter().sum::<u128>() / n as u128;
    let mut sorted: Vec<u128> = data.iter().copied().collect();
    sorted.sort_unstable();
    let p95 = sorted[n * 95 / 100];
    let p99 = sorted[(n * 99 / 100).min(n - 1)];
    (avg, p95, p99, n)
}

fn metric_row(name: &str, avg: u128, p95: u128, p99: u128, n: usize) -> String {
    format!("│ {:<14} │{:>6} ms │{:>6} ms │{:>6} ms │ {:>3} │\n",
            name, avg, p95, p99, n)
}

fn info_row(text: &str) -> String {
    format!("│ {:<53} │\n", text)
}

fn query_ram_mb() -> u64 {
    let pid = sysinfo::Pid::from_u32(std::process::id());
    let sys = sysinfo::System::new_all();
    sys.process(pid).map(|p| p.memory() / 1_048_576).unwrap_or(0)
}


// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agg() -> PerfAggregator { PerfAggregator::new() }

    #[test]
    fn test_rolling_window_capped() {
        let agg = make_agg();
        for i in 0..150u128 {
            push_window(&agg.ocr_ms, i);
        }
        assert_eq!(agg.ocr_ms.lock().unwrap().len(), WINDOW);
    }

    #[test]
    fn test_avg_simple() {
        let mut d = VecDeque::new();
        d.extend([10u128, 20, 30]);
        let (avg, _, _, n) = compute_stats(&d);
        assert_eq!(avg, 20);
        assert_eq!(n, 3);
    }

    #[test]
    fn test_percentiles_sorted() {
        let mut d = VecDeque::new();
        for i in 1u128..=100 { d.push_back(i); }
        let (avg, p95, p99, _) = compute_stats(&d);
        assert_eq!(avg, 50);
        // index = n*95/100 = 95 → sorted[95] = 96 (0-indexed, values start at 1)
        assert_eq!(p95, 96);
        // index = n*99/100 = 99 → sorted[99] = 100
        assert_eq!(p99, 100);
    }

    #[test]
    fn test_empty_returns_zeros() {
        let d: VecDeque<u128> = VecDeque::new();
        assert_eq!(compute_stats(&d), (0, 0, 0, 0));
    }

    #[test]
    fn test_stability_score() {
        let agg = make_agg();
        agg.record_store(10, true);   // fresh
        agg.record_store(10, false);  // chronos hit
        agg.record_store(10, false);  // chronos hit
        assert_eq!(agg.total_frames.load(Ordering::Relaxed), 3);
        assert_eq!(agg.chronos_hits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_ocr_histogram_buckets() {
        let agg = make_agg();
        // 3 fast, 2 mid, 1 slow
        for ms in [10u128, 30, 45, 55, 100, 200] { agg.record_ocr(ms); }
        let (fast, mid, slow, n) = agg.ocr_histogram_50();
        assert_eq!(fast, 3);
        assert_eq!(mid,  2);
        assert_eq!(slow, 1);
        assert_eq!(n,    6);
    }

    #[test]
    fn test_scoped_instances_independent() {
        // stress() and realtime() must be separate instances
        realtime().record_ocr(10);
        stress().record_ocr(999);
        let (ra, _, _, _) = compute_stats(&realtime().ocr_ms.lock().unwrap());
        let (sa, _, _, _) = compute_stats(&stress().ocr_ms.lock().unwrap());
        assert_ne!(ra, sa, "Realtime and stress stats must not share state");
    }

    #[test]
    #[ignore]
    fn print_demo_table() {
        let agg = PerfAggregator::new();
        for ms in [32,45,50,55,60,120,135,200,280,350u128] { agg.record_ocr(ms); }
        for ms in [18,20,22,25,28,30,55,70,85,100u128]     { agg.record_store(ms, ms < 50); }
        for ms in [750,820,890,950,1100,1250,1400u128]      { agg.record_ttft(ms); }
        agg.total_frames.store(1337, Ordering::Relaxed);
        agg.chronos_hits.store(872,  Ordering::Relaxed);
        println!("{}", agg.stats_report_titled("Realtime — Production Stats"));
        println!("{}", agg.stats_report_titled("Stress Test — Synthetic Load"));
    }

    #[test]
    fn test_stats_report_contains_metrics() {
        let agg = make_agg();
        agg.record_ocr(50);
        agg.record_store(30, true);
        agg.record_ttft(900);
        let report = agg.stats_report();
        assert!(report.contains("OCR"));
        assert!(report.contains("Store/Embed"));
        assert!(report.contains("Thinker TTFT"));
        assert!(report.contains("Uptime"));
        assert!(report.contains("OCR Histogram"));
    }

    // ── compute_stats extended ────────────────────────────────────────────────

    #[test]
    fn compute_stats_single_element() {
        let mut d = VecDeque::new();
        d.push_back(42u128);
        let (avg, p95, p99, n) = compute_stats(&d);
        assert_eq!(avg, 42);
        assert_eq!(p95, 42);
        assert_eq!(p99, 42);
        assert_eq!(n, 1);
    }

    #[test]
    fn compute_stats_two_elements() {
        let mut d = VecDeque::new();
        d.extend([10u128, 20]);
        let (avg, _, _, n) = compute_stats(&d);
        assert_eq!(avg, 15);
        assert_eq!(n, 2);
    }

    #[test]
    fn compute_stats_all_equal() {
        let mut d = VecDeque::new();
        for _ in 0..10 { d.push_back(100u128); }
        let (avg, p95, p99, _) = compute_stats(&d);
        assert_eq!(avg, 100);
        assert_eq!(p95, 100);
        assert_eq!(p99, 100);
    }

    #[test]
    fn compute_stats_p99_capped_at_last() {
        // n=5, n*99/100 = 4, min(4, 4) = 4 → sorted[4] = 5
        let mut d = VecDeque::new();
        d.extend([1u128, 2, 3, 4, 5]);
        let (_, _, p99, _) = compute_stats(&d);
        assert_eq!(p99, 5);
    }

    #[test]
    fn compute_stats_unsorted_input_correct_percentiles() {
        let mut d = VecDeque::new();
        d.extend([50u128, 10, 90, 30, 70]); // unsorted
        let (avg, _, p99, _) = compute_stats(&d);
        assert_eq!(avg, 50); // (50+10+90+30+70)/5 = 250/5 = 50
        // sorted = [10,30,50,70,90]; n*99/100 = 4 → sorted[4] = 90
        assert_eq!(p99, 90);
    }

    #[test]
    fn compute_stats_large_values_no_overflow() {
        let mut d = VecDeque::new();
        // 10 values of 1_000_000 ms each (1000 seconds)
        for _ in 0..10 { d.push_back(1_000_000u128); }
        let (avg, _, _, _) = compute_stats(&d);
        assert_eq!(avg, 1_000_000);
    }

    // ── push_window / rolling cap ─────────────────────────────────────────────

    #[test]
    fn push_window_respects_cap_exactly_at_boundary() {
        let agg = make_agg();
        for i in 0..100u128 { push_window(&agg.ocr_ms, i); }
        assert_eq!(agg.ocr_ms.lock().unwrap().len(), WINDOW);
        // Oldest pushed off: front should be 0 → after 100 pushes (cap=100), len=100
        // Push one more
        push_window(&agg.ocr_ms, 999);
        assert_eq!(agg.ocr_ms.lock().unwrap().len(), WINDOW);
        // Most recent value is 999
        assert_eq!(*agg.ocr_ms.lock().unwrap().back().unwrap(), 999);
    }

    #[test]
    fn push_window_oldest_evicted() {
        let agg = make_agg();
        push_window(&agg.ocr_ms, 1); // will be evicted
        for i in 2..=WINDOW as u128 + 1 {
            push_window(&agg.ocr_ms, i);
        }
        // After WINDOW+1 pushes, the first value (1) should be gone
        let lock = agg.ocr_ms.lock().unwrap();
        assert!(!lock.contains(&1u128), "oldest value should be evicted");
    }

    // ── OCR histogram edge cases ──────────────────────────────────────────────

    #[test]
    fn ocr_histogram_empty_aggregator_zeros() {
        let agg = make_agg();
        let (fast, mid, slow, n) = agg.ocr_histogram_50();
        assert_eq!(fast, 0);
        assert_eq!(mid, 0);
        assert_eq!(slow, 0);
        assert_eq!(n, 0);
    }

    #[test]
    fn ocr_histogram_boundary_49ms_is_fast() {
        let agg = make_agg();
        agg.record_ocr(49);
        let (fast, mid, slow, _) = agg.ocr_histogram_50();
        assert_eq!(fast, 1);
        assert_eq!(mid, 0);
        assert_eq!(slow, 0);
    }

    #[test]
    fn ocr_histogram_boundary_50ms_is_mid() {
        let agg = make_agg();
        agg.record_ocr(50);
        let (fast, mid, slow, _) = agg.ocr_histogram_50();
        assert_eq!(fast, 0);
        assert_eq!(mid, 1);
        assert_eq!(slow, 0);
    }

    #[test]
    fn ocr_histogram_boundary_149ms_is_mid() {
        let agg = make_agg();
        agg.record_ocr(149);
        let (fast, mid, slow, _) = agg.ocr_histogram_50();
        assert_eq!(fast, 0);
        assert_eq!(mid, 1);
        assert_eq!(slow, 0);
    }

    #[test]
    fn ocr_histogram_boundary_150ms_is_slow() {
        let agg = make_agg();
        agg.record_ocr(150);
        let (fast, mid, slow, _) = agg.ocr_histogram_50();
        assert_eq!(fast, 0);
        assert_eq!(mid, 0);
        assert_eq!(slow, 1);
    }

    #[test]
    fn ocr_histogram_zero_ms_is_fast() {
        let agg = make_agg();
        agg.record_ocr(0);
        let (fast, _, _, _) = agg.ocr_histogram_50();
        assert_eq!(fast, 1);
    }

    #[test]
    fn ocr_histogram_sum_equals_n() {
        let agg = make_agg();
        for ms in [5u128, 55, 155, 10, 60, 160, 0, 49, 50, 149, 150, 200] {
            agg.record_ocr(ms);
        }
        let (fast, mid, slow, n) = agg.ocr_histogram_50();
        assert_eq!(fast + mid + slow, n as u64);
    }

    #[test]
    fn ocr_histogram_uses_last_50_samples() {
        let agg = make_agg();
        // Push 60 slow samples, then 10 fast samples
        for _ in 0..60 { agg.record_ocr(500); }
        for _ in 0..10 { agg.record_ocr(1);   }
        // histogram window = 50, takes last 50 samples = 10 fast + 40 slow
        let (fast, _, slow, n) = agg.ocr_histogram_50();
        assert_eq!(n, 50, "histogram should use last HIST_WINDOW=50 samples");
        assert_eq!(fast, 10);
        assert_eq!(slow, 40);
    }

    // ── record_store / counters ───────────────────────────────────────────────

    #[test]
    fn record_store_increments_total_each_call() {
        let agg = make_agg();
        for i in 0..7 { agg.record_store(i, true); }
        assert_eq!(agg.total_frames.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn record_store_fresh_does_not_increment_chronos() {
        let agg = make_agg();
        agg.record_store(10, true); // is_fresh = true → NOT a chronos hit
        assert_eq!(agg.chronos_hits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn record_store_not_fresh_increments_chronos() {
        let agg = make_agg();
        agg.record_store(10, false); // is_fresh = false → chronos hit
        assert_eq!(agg.chronos_hits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stability_100_percent_all_chronos() {
        let agg = make_agg();
        for _ in 0..10 { agg.record_store(5, false); }
        let total = agg.total_frames.load(Ordering::Relaxed);
        let hits  = agg.chronos_hits.load(Ordering::Relaxed);
        assert_eq!(total, 10);
        assert_eq!(hits, 10);
    }

    #[test]
    fn stability_0_percent_all_fresh() {
        let agg = make_agg();
        for _ in 0..10 { agg.record_store(5, true); }
        assert_eq!(agg.chronos_hits.load(Ordering::Relaxed), 0);
    }

    // ── stress mode routing ───────────────────────────────────────────────────

    #[test]
    fn stress_mode_toggle_affects_global() {
        set_stress_mode(false);
        assert!(!is_stress_mode());
        set_stress_mode(true);
        assert!(is_stress_mode());
        set_stress_mode(false); // restore
    }

    #[test]
    fn global_routes_to_realtime_when_not_stress() {
        set_stress_mode(false);
        let ptr_global   = global() as *const PerfAggregator;
        let ptr_realtime = realtime() as *const PerfAggregator;
        assert_eq!(ptr_global, ptr_realtime);
    }

    #[test]
    fn global_routes_to_stress_when_active() {
        set_stress_mode(true);
        let ptr_global = global() as *const PerfAggregator;
        let ptr_stress = stress() as *const PerfAggregator;
        assert_eq!(ptr_global, ptr_stress);
        set_stress_mode(false);
    }

    // ── stats_report_titled ───────────────────────────────────────────────────

    #[test]
    fn stats_report_titled_custom_title() {
        let agg = make_agg();
        let r = agg.stats_report_titled("My Custom Title");
        assert!(r.contains("My Custom Title"));
    }

    #[test]
    fn stats_report_titled_zero_samples_no_panic() {
        let agg = make_agg();
        let r = agg.stats_report_titled("Empty");
        assert!(r.contains("OCR"));
        assert!(r.contains("0")); // n=0 shows somewhere
    }

    #[test]
    fn stats_report_has_table_borders() {
        let agg = make_agg();
        agg.record_ocr(10);
        let r = agg.stats_report();
        assert!(r.contains('┌') || r.contains('│') || r.contains('└'));
    }

    #[test]
    fn stats_report_total_frames_shown() {
        let agg = make_agg();
        agg.record_store(1, true);
        agg.record_store(2, true);
        let r = agg.stats_report();
        assert!(r.contains("Total frames"));
        assert!(r.contains('2'));
    }

    #[test]
    fn stats_report_ram_row_present() {
        let agg = make_agg();
        let r = agg.stats_report();
        assert!(r.contains("RAM") || r.contains("MB"));
    }

    // ── perf_json_value ───────────────────────────────────────────────────────

    #[test]
    fn perf_json_has_expected_keys() {
        let agg = make_agg();
        agg.record_ocr(30);
        agg.record_store(20, true);
        agg.record_ttft(800);
        let v = agg.perf_json_value();
        assert!(v.get("ocr").is_some());
        assert!(v.get("store").is_some());
        assert!(v.get("ttft").is_some());
        assert!(v.get("stability_pct").is_some());
        assert!(v.get("total_frames").is_some());
        assert!(v.get("uptime_secs").is_some());
        assert!(v.get("process_ram_mb").is_some());
    }

    #[test]
    fn perf_json_ocr_avg_correct() {
        let agg = make_agg();
        agg.record_ocr(100);
        agg.record_ocr(200);
        let v = agg.perf_json_value();
        let avg = v["ocr"]["avg_ms"].as_u64().unwrap();
        assert_eq!(avg, 150);
    }

    #[test]
    fn perf_json_stability_zero_when_no_frames() {
        let agg = make_agg();
        let v = agg.perf_json_value();
        let stab = v["stability_pct"].as_f64().unwrap();
        assert!((stab - 0.0).abs() < 0.01);
    }

    #[test]
    fn perf_json_stability_nonzero_with_hits() {
        let agg = make_agg();
        agg.record_store(10, false); // chronos hit
        agg.record_store(10, false); // chronos hit
        agg.record_store(10, true);  // fresh
        let v = agg.perf_json_value();
        let stab = v["stability_pct"].as_f64().unwrap();
        // 2/3 = 66.7%
        assert!(stab > 60.0 && stab < 70.0, "stability should be ~66.7%, got {}", stab);
    }

    #[test]
    fn perf_json_histogram_keys() {
        let agg = make_agg();
        let v = agg.perf_json_value();
        let hist = v.get("ocr_histogram").unwrap();
        assert!(hist.get("0_50ms").is_some());
        assert!(hist.get("50_150ms").is_some());
        assert!(hist.get("150ms_plus").is_some());
    }

    #[test]
    fn perf_json_total_frames_matches_record_count() {
        let agg = make_agg();
        for _ in 0..5 { agg.record_store(1, true); }
        let v = agg.perf_json_value();
        assert_eq!(v["total_frames"].as_u64().unwrap(), 5);
    }

    // ── metric_row / info_row formatting ─────────────────────────────────────

    #[test]
    fn metric_row_contains_name() {
        let r = metric_row("TestMetric", 10, 20, 30, 5);
        assert!(r.contains("TestMetric"));
    }

    #[test]
    fn metric_row_contains_values() {
        let r = metric_row("M", 100, 200, 300, 42);
        assert!(r.contains("100"));
        assert!(r.contains("200"));
        assert!(r.contains("300"));
        assert!(r.contains("42"));
    }

    #[test]
    fn info_row_contains_text() {
        let r = info_row("hello world");
        assert!(r.contains("hello world"));
    }

    #[test]
    fn info_row_has_pipe_borders() {
        let r = info_row("x");
        assert!(r.starts_with('│'));
        assert!(r.trim_end().ends_with('│'));
    }
}
