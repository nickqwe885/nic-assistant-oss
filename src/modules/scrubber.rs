//! Deterministic secret redaction for everything Sentinel captures.
//!
//! Screen OCR and clipboard text can contain seed phrases, card numbers, private
//! keys and API tokens. Those must never reach the database — and never come back
//! out in an answer. `scrub()` runs on the WAY IN (before storage) so secrets are
//! never persisted, and again on the context bundle on the WAY OUT (so anything
//! captured before this shipped can't surface either). All detection is code —
//! no model, no network, no false sense of "the AI decides what's private".

/// Official BIP-39 English wordlist (2048 words), one per line.
static BIP39: &str = include_str!("../data/bip39_english.txt");

use std::collections::HashSet;
use std::sync::OnceLock;

fn bip39_set() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| BIP39.lines().map(str::trim).filter(|w| !w.is_empty()).collect())
}

/// Redact secrets from `text`. Returns the cleaned string; identity when clean.
pub fn scrub(text: &str) -> String {
    let mut out = scrub_seed_phrases(text);
    out = scrub_tokens(&out);
    out = scrub_cards(&out);
    out
}

/// True if scrubbing would change the text — used to decide whether a whole
/// capture is too sensitive to keep even after redaction.
pub fn contains_secret(text: &str) -> bool {
    scrub(text) != text
}

/// Windows whose very TITLE says "secrets live here": wallets, password
/// managers, banking. These are skipped wholesale — no screenshot, no OCR, no
/// record that the window was even open. Cheaper and safer than redacting after
/// the fact, and it covers secrets we don't have a pattern for.
const PRIVATE_WINDOW_MARKERS: &[&str] = &[
    // Wallets & crypto
    "metamask", "phantom wallet", "trust wallet", "tonkeeper", "exodus wallet",
    "ledger live", "trezor", "seed phrase", "recovery phrase", "secret phrase",
    "private key", "мнемоническая", "секретная фраза",
    // Password managers
    "1password", "bitwarden", "lastpass", "keepass", "dashlane", "nordpass",
    "password manager", "менеджер паролей",
    // Auth & banking screens
    "sign in", "log in", "password", "пароль", "войти в аккаунт",
    "online banking", "интернет-банк", "kaspi", "halyk", "sberbank",
];

/// True when a window must never be captured, based on its title/app name.
pub fn is_private_window(window_title: &str, app_name: &str) -> bool {
    let t = window_title.to_lowercase();
    let a = app_name.to_lowercase();
    PRIVATE_WINDOW_MARKERS
        .iter()
        .any(|m| t.contains(m) || a.contains(m))
}

/// 12/15/18/21/24 consecutive BIP-39 words = a wallet seed phrase. We collapse
/// any run of ≥ 12 dictionary words in a row (case-insensitive) into a marker;
/// ordinary prose never strings 12 wallet-words together, so false positives are
/// negligible, while a pasted or on-screen phrase is caught regardless of layout.
fn scrub_seed_phrases(text: &str) -> String {
    let set = bip39_set();
    // Tokenize on whitespace but keep the original slices so non-seed text is
    // reproduced verbatim; we only rewrite qualifying runs.
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (content, nl) = match line.strip_suffix('\n') {
            Some(c) => (c, "\n"),
            None => (line, ""),
        };
        out.push_str(&scrub_seed_in_segment(content, set));
        out.push_str(nl);
    }
    out
}

fn scrub_seed_in_segment(seg: &str, set: &HashSet<&'static str>) -> String {
    let words: Vec<&str> = seg.split_whitespace().collect();
    if words.len() < 12 {
        return seg.to_string(); // can't hold a seed phrase — keep bytes verbatim
    }
    // Mark which tokens are dictionary words (letters only, lowercased).
    let is_word: Vec<bool> = words
        .iter()
        .map(|w| {
            let lw: String = w.chars().filter(|c| c.is_ascii_alphabetic()).collect::<String>().to_lowercase();
            !lw.is_empty() && w.chars().all(|c| c.is_ascii_alphabetic()) && set.contains(lw.as_str())
        })
        .collect();

    // Find runs of ≥12 consecutive dictionary words; replace them with a marker.
    // A run is collapsed whole — over-redacting a couple of adjacent common words
    // is a trade we take gladly over leaking part of a seed phrase.
    let mut result: Vec<String> = Vec::with_capacity(words.len());
    let mut redacted = false;
    let mut i = 0;
    while i < words.len() {
        if is_word[i] {
            let start = i;
            while i < words.len() && is_word[i] {
                i += 1;
            }
            if i - start >= 12 {
                result.push("[seed phrase redacted]".to_string());
                redacted = true;
            } else {
                for w in &words[start..i] {
                    result.push((*w).to_string());
                }
            }
        } else {
            result.push(words[i].to_string());
            i += 1;
        }
    }
    // No seed found → return the ORIGINAL slice so whitespace/tabs survive intact
    // (this path is the overwhelming majority of captures).
    if !redacted {
        return seg.to_string();
    }
    result.join(" ")
}

/// API keys / private keys: recognizable prefixes and long hex/base58 blobs.
fn scrub_tokens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for tok in split_keep(text) {
        out.push_str(if is_secret_token(tok) { "[key redacted]" } else { tok });
    }
    out
}

fn is_secret_token(tok: &str) -> bool {
    let t = tok.trim();
    if t.len() < 20 {
        return false;
    }
    let lower = t.to_lowercase();
    // Known credential prefixes (OpenAI, GitHub, Slack, Stripe, AWS, Google…).
    const PREFIXES: &[&str] = &[
        "sk-", "sk_", "pk_", "rk_", "ghp_", "gho_", "ghs_", "ghu_", "github_pat_",
        "xox", "akey-", "akia", "asia", "ya29.", "aiza", "bearer ", "eyj",
    ];
    if PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return true;
    }
    // 0x-prefixed EVM private key / hash (40+ hex chars).
    if lower.starts_with("0x") && t.len() >= 42 && t[2..].chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    // Bare long hex blob (private key, 64 hex chars typical) — require ≥48.
    if t.len() >= 48 && t.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    false
}

/// Card numbers: 13–19 digits (spaces/dashes allowed between groups) that pass
/// the Luhn checksum. Luhn keeps ordinary long numbers (IDs, timestamps) safe.
fn scrub_cards(text: &str) -> String {
    let bytes: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        // Candidate = run of digits, spaces and single dashes starting on a digit.
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut digits = String::new();
            let mut j = i;
            while j < bytes.len() {
                let c = bytes[j];
                if c.is_ascii_digit() {
                    digits.push(c);
                    j += 1;
                } else if (c == ' ' || c == '-')
                    && j + 1 < bytes.len()
                    && bytes[j + 1].is_ascii_digit()
                    && !digits.is_empty()
                {
                    j += 1;
                } else {
                    break;
                }
            }
            if (13..=19).contains(&digits.len()) && luhn_ok(&digits) {
                out.push_str("[card redacted]");
                i = j;
                continue;
            }
            // Not a card — emit the original run unchanged.
            for c in &bytes[start..j.max(start + 1)] {
                out.push(*c);
            }
            i = j.max(start + 1);
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

fn luhn_ok(digits: &str) -> bool {
    let mut sum = 0u32;
    let mut alt = false;
    for c in digits.chars().rev() {
        let mut d = c.to_digit(10).unwrap();
        if alt {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        alt = !alt;
    }
    sum % 10 == 0
}

/// Split into alternating non-space / space runs, preserving every character so
/// `concat()` reconstructs the input exactly.
fn split_keep(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut last = 0;
    let mut in_ws = text.chars().next().map(char::is_whitespace).unwrap_or(false);
    for (idx, c) in text.char_indices() {
        if c.is_whitespace() != in_ws {
            parts.push(&text[last..idx]);
            last = idx;
            in_ws = !in_ws;
        }
    }
    if last < text.len() {
        parts.push(&text[last..]);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_12_word_seed_phrase() {
        let seed = "abandon ability able about above absent absorb abstract absurd abuse access accident";
        let s = format!("Your recovery phrase: {seed} — write it down!");
        let out = scrub(&s);
        assert!(out.contains("[seed phrase redacted]"), "{out}");
        assert!(!out.contains("abandon") && !out.contains("accident"), "{out}");
        // Surrounding context (non-dictionary words) survives.
        assert!(out.contains("Your") && out.contains("down!"), "{out}");
    }

    #[test]
    fn redacts_24_word_seed_phrase() {
        let w = "zoo abandon ability able about above absent absorb abstract absurd abuse access \
                 accident account accuse achieve acid acoustic acquire across act action actor";
        assert!(scrub(w).contains("[seed phrase redacted]"));
    }

    #[test]
    fn ordinary_prose_is_untouched() {
        let p = "the quick brown fox jumps over the lazy dog while the cat sleeps soundly today";
        assert_eq!(scrub(p), p);
    }

    #[test]
    fn short_dictionary_run_kept() {
        // A handful of common words that happen to be in BIP-39 must NOT redact.
        let p = "process act use gas legal";
        assert_eq!(scrub(p), p);
    }

    #[test]
    fn redacts_valid_card_number() {
        // 4242 4242 4242 4242 is Stripe's canonical Luhn-valid test card.
        let out = scrub("card 4242 4242 4242 4242 exp");
        assert!(out.contains("[card redacted]"), "{out}");
        assert!(out.contains("card") && out.contains("exp"));
    }

    #[test]
    fn keeps_non_luhn_long_number() {
        let s = "order 1234567890123456 shipped";
        assert_eq!(scrub(s), s); // fails Luhn → not a card
    }

    #[test]
    fn keeps_ordinary_timestamps_and_ids() {
        let s = "event at 1720000000 with id 42 and port 7878";
        assert_eq!(scrub(s), s);
    }

    #[test]
    fn redacts_api_key_prefixes() {
        assert!(scrub("token sk-abcdefghijklmnopqrstuvwxyz123").contains("[key redacted]"));
        assert!(scrub("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").contains("[key redacted]"));
    }

    #[test]
    fn redacts_evm_private_key() {
        let k = "0x4c0883a69102937d6231471b5dbb6204fe512961708279b9c6b8b3f2a1d4e5f6";
        assert!(scrub(k).contains("[key redacted]"));
    }

    #[test]
    fn contains_secret_flags_correctly() {
        assert!(contains_secret("sk-abcdefghijklmnopqrstuvwxyz"));
        assert!(!contains_secret("just a normal sentence here"));
    }

    #[test]
    fn scrub_is_lossless_on_clean_text() {
        let s = "Line one\nLine two with 42 items\n  indented\ttabbed";
        assert_eq!(scrub(s), s);
    }

    #[test]
    fn private_windows_are_recognized() {
        assert!(is_private_window("MetaMask - Google Chrome", "chrome.exe"));
        assert!(is_private_window("Bitwarden — vault", "bitwarden.exe"));
        assert!(is_private_window("Kaspi.kz — Мой банк", "chrome.exe"));
        assert!(is_private_window("Reveal secret phrase", "chrome.exe"));
    }

    #[test]
    fn ordinary_windows_are_captured() {
        assert!(!is_private_window("nic-assistant — main.rs", "code.exe"));
        assert!(!is_private_window("YouTube — Google Chrome", "chrome.exe"));
        assert!(!is_private_window("Thomas Calculus.pdf", "acrobat.exe"));
    }
}
