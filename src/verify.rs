//! Deterministic grounding verifier — anti-hallucination "Check 1".
//!
//! The small model reads grounded facts and produces an answer. This module is the
//! code-side guard: every NUMBER and DATE that appears in the answer MUST also appear
//! in the supplied facts — otherwise the model invented it. The QA branch is expected
//! to call [`verify_grounded`] on the model's `<answer>` against the fact bundle and,
//! on failure, fall back to the raw snippets instead of showing a fabricated answer.
//!
//! Design (high precision, few false abstains):
//!   * NUMBERS — strict. Compared as a numeric value when parseable (so `500`,
//!     `500.00` and `1,000` vs `1000` all match), otherwise by digit-core (e.g.
//!     times `9:30`, IPs `127.0.0.1`). Bare single digits `0-9` are skipped — they
//!     are list-marker noise, not fabricated facts.
//!   * DATES — strict, but only a month that sits NEXT TO a number ("June 5",
//!     "5 June") counts as a date, so the verb "may" never false-triggers. The
//!     month must be present (any form) in the facts.
//!   * NAMES — deliberately NOT gated. Substring matching breaks on synonyms and
//!     abbreviations (JS/JavaScript, Mike/Michael) and would cause false abstains.
//!
//! Pure: no I/O, no model calls. Numbers are language-agnostic; month names are
//! English (English-only beta).

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// What the verifier found to be ungrounded. Empty == the answer is fully grounded.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct VerifyReport {
    /// Numbers present in the answer but absent from the facts.
    pub ungrounded_numbers: Vec<String>,
    /// Month-dates present in the answer but absent from the facts.
    pub ungrounded_months: Vec<String>,
}

impl VerifyReport {
    /// True when nothing fabricated was found — the answer may be shown as-is.
    pub fn ok(&self) -> bool {
        self.ungrounded_numbers.is_empty() && self.ungrounded_months.is_empty()
    }
}

/// Convenience predicate: does every number/date in `answer` appear in `facts`?
pub fn is_grounded(answer: &str, facts: &str) -> bool {
    verify_grounded(answer, facts).ok()
}

/// Check that every number and month-date in `answer` is backed by `facts`.
pub fn verify_grounded(answer: &str, facts: &str) -> VerifyReport {
    let mut report = VerifyReport::default();

    // ── numbers ────────────────────────────────────────────────────────────────
    let (fact_vals, fact_cores) = collect_numbers(facts);
    for tok in number_tokens(answer) {
        match num_key(&tok) {
            // Skip bare single digits (0-9): list markers, trivial counts.
            NumKey::Val(v) if v.fract() == 0.0 && v.abs() < 10.0 => {}
            NumKey::Val(v) => {
                if !fact_vals.iter().any(|f| (f - v).abs() < 1e-6) {
                    report.ungrounded_numbers.push(tok);
                }
            }
            NumKey::Core(core) => {
                if core.len() >= 2 && !fact_cores.contains(&core_key(&core)) {
                    report.ungrounded_numbers.push(tok);
                }
            }
        }
    }

    // ── month-dates ──────────────────────────────────────────────────────────────
    let fact_months = months_in(facts);
    for code in answer_date_months(answer) {
        if !fact_months.contains(code) {
            report.ungrounded_months.push(code.to_string());
        }
    }

    report
}

// ── numbers ──────────────────────────────────────────────────────────────────────

enum NumKey {
    /// A value parseable as a single decimal (compared numerically).
    Val(f64),
    /// A digit-only fingerprint for non-decimal tokens (times, IPs, versions).
    Core(String),
}

/// Maximal digit runs, keeping internal `. , :` (e.g. `1,000`, `3.14`, `9:30`).
fn number_tokens(text: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\d[\d.,:]*\d|\d").unwrap());
    re.find_iter(text).map(|m| m.as_str().to_string()).collect()
}

/// Leading-zero-insensitive digit fingerprint, so `9:30` matches `09:30`.
fn core_key(core: &str) -> String {
    let t = core.trim_start_matches('0');
    if t.is_empty() { "0".to_string() } else { t.to_string() }
}

fn num_key(tok: &str) -> NumKey {
    // Drop thousands separators and spaces, then try a plain decimal parse.
    let cleaned: String = tok.chars().filter(|&c| c != ',' && c != ' ').collect();
    if let Ok(v) = cleaned.parse::<f64>() {
        NumKey::Val(v)
    } else {
        NumKey::Core(tok.chars().filter(|c| c.is_ascii_digit()).collect())
    }
}

/// Split the facts into a list of decimal values and a set of digit-cores.
fn collect_numbers(facts: &str) -> (Vec<f64>, HashSet<String>) {
    let mut vals = Vec::new();
    let mut cores = HashSet::new();
    for tok in number_tokens(facts) {
        match num_key(&tok) {
            NumKey::Val(v) => vals.push(v),
            NumKey::Core(c) => {
                if !c.is_empty() {
                    cores.insert(core_key(&c));
                }
            }
        }
    }
    (vals, cores)
}

// ── months / dates ─────────────────────────────────────────────────────────────

/// Canonical 3-letter code for an English month word, else `None`.
fn month_code(word: &str) -> Option<&'static str> {
    match word.to_ascii_lowercase().as_str() {
        "january" | "jan" => Some("jan"),
        "february" | "feb" => Some("feb"),
        "march" | "mar" => Some("mar"),
        "april" | "apr" => Some("apr"),
        "may" => Some("may"),
        "june" | "jun" => Some("jun"),
        "july" | "jul" => Some("jul"),
        "august" | "aug" => Some("aug"),
        "september" | "sept" | "sep" => Some("sep"),
        "october" | "oct" => Some("oct"),
        "november" | "nov" => Some("nov"),
        "december" | "dec" => Some("dec"),
        _ => None,
    }
}

const MONTH_ALT: &str = r"january|february|march|april|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec";

/// Every month mentioned in `text`, in any form (generous — used for the facts side).
fn months_in(text: &str) -> HashSet<&'static str> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(&format!(r"(?i)\b(?:{MONTH_ALT})\b")).unwrap());
    re.find_iter(text).filter_map(|m| month_code(m.as_str())).collect()
}

/// Months in `text` that sit next to a number ("June 5", "5 June", "Jun 2024").
/// Requiring an adjacent digit keeps the verb "may" from false-triggering.
fn answer_date_months(text: &str) -> Vec<&'static str> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(&format!(
            r"(?i)\b\d{{1,2}}\s+({MONTH_ALT})\b|\b({MONTH_ALT})\.?\s+\d{{1,4}}\b"
        ))
        .unwrap()
    });
    re.captures_iter(text)
        .filter_map(|c| c.get(1).or_else(|| c.get(2)))
        .filter_map(|m| month_code(m.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── numbers ──────────────────────────────────────────────────────────────
    #[test]
    fn fabricated_number_is_flagged() {
        let r = verify_grounded("It costs $5000.", "the invoice was $500");
        assert_eq!(r.ungrounded_numbers, vec!["5000"]);
        assert!(!r.ok());
    }

    #[test]
    fn grounded_number_passes() {
        assert!(is_grounded("500 dollars", "total was $500.00"));
    }

    #[test]
    fn decimal_and_thousands_separators_match() {
        // 500 vs 500.00 (value match) and 1,000 vs 1000 (separator stripped).
        assert!(is_grounded("you spent 1,000", "1000 spent"));
        assert!(is_grounded("3.14 here", "pi is 3.14"));
    }

    #[test]
    fn single_digits_are_ignored() {
        // List markers / trivial counts must not trip the gate.
        assert!(is_grounded("step 5 is done, point 2 too", ""));
    }

    #[test]
    fn fabricated_year_is_flagged() {
        let r = verify_grounded("this happened in 2019", "you logged in during 2024");
        assert_eq!(r.ungrounded_numbers, vec!["2019"]);
    }

    #[test]
    fn non_decimal_token_uses_digit_core() {
        // A time present in facts (formatting differs) still grounds.
        assert!(is_grounded("at 9:30 today", "logged 09:30 entry"));
        let r = verify_grounded("ip was 10.0.0.5", "no addresses here");
        assert!(!r.ok());
    }

    // ── month-dates ──────────────────────────────────────────────────────────
    #[test]
    fn fabricated_month_date_is_flagged() {
        let r = verify_grounded("you coded on June 5", "you worked on May 5");
        assert_eq!(r.ungrounded_months, vec!["jun"]);
    }

    #[test]
    fn grounded_month_date_passes() {
        assert!(is_grounded("June 5 you used VS Code", "June 5: VS Code session"));
        assert!(is_grounded("5 Jun was busy", "lots happened in June"));
    }

    #[test]
    fn bare_may_verb_does_not_trigger() {
        // "may" without an adjacent number is not treated as a date.
        assert!(is_grounded("you may want to rest", ""));
    }

    #[test]
    fn empty_answer_is_grounded() {
        assert!(is_grounded("", "anything at all 2024"));
        assert!(VerifyReport::default().ok());
    }
}
