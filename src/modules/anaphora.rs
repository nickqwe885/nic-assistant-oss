//! Deterministic back-reference resolution for Pilot commands.
//!
//! "open video about him" right after "who is David Laid" should open a video
//! about David Laid. The 1.5B model can't be trusted with that link, so it's
//! resolved in code: a pronoun in the OBJECT of a command is replaced with the
//! referent extracted from the previous question. Chat/QA queries are never
//! rewritten — the existing follow-up prompt logic already handles those.

/// Command verbs that mark a query as a Pilot/media action. Start-anchored so a
/// conversational sentence that merely contains a verb is never rewritten.
const COMMAND_STARTS: &[&str] = &[
    "open ", "show ", "launch ", "run ", "start ", "go to ", "play ", "put on ", "watch ",
    "включи ", "открой ", "запусти ", "покажи ", "зайди на ", "поставь ", "посмотреть ",
];

/// Words that link a command verb to its object ("video about him").
const CONNECTORS: &[&str] = &["about", "of", "by", "from", "with", "про", "о", "об", "обо"];

/// Pronouns we resolve. Deliberately narrow: subject/quantity words like
/// "there", "they", "these", "more" stay with the QA follow-up logic.
const PRONOUNS: &[&str] = &[
    "him", "her", "them", "it", "his", "its", "their", "this", "that",
    // Legacy Russian (harmless when RU is never typed).
    "него", "неё", "нее", "ней", "нём", "нем", "них", "его", "её", "ее", "их",
];

/// Rewrite `query` with the pronoun replaced by the previous question's
/// referent. `None` = not a command / no pronoun / no safe referent — caller
/// keeps the original query and nothing changes.
pub fn resolve(query: &str, prev_q: &str) -> Option<String> {
    let q_trim = query.trim();
    let q_lower = q_trim.to_lowercase();
    if !COMMAND_STARTS.iter().any(|p| q_lower.starts_with(p)) {
        return None;
    }

    let words: Vec<&str> = q_trim.split_whitespace().collect();
    let lower: Vec<String> = words
        .iter()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .collect();

    // A pronoun counts as the command's object when it follows a connector
    // ("about him") or closes the query ("play her").
    let idx = lower.iter().enumerate().find_map(|(i, lw)| {
        if !PRONOUNS.contains(&lw.as_str()) {
            return None;
        }
        let after_connector = i > 0 && CONNECTORS.contains(&lower[i - 1].as_str());
        let is_last = i == lower.len() - 1;
        (after_connector || is_last).then_some(i)
    })?;

    // Bare trailing "it/this/that" ("play it", "watch that") usually means the
    // thing on screen, not the previous topic — only rewrite those after a
    // connector ("video about it").
    if matches!(lower[idx].as_str(), "it" | "this" | "that")
        && !(idx > 0 && CONNECTORS.contains(&lower[idx - 1].as_str()))
    {
        return None;
    }

    let referent = referent_of(prev_q)?;
    let mut out: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    out[idx] = referent;
    Some(out.join(" "))
}

/// Pull the topic out of the previous question: strip question scaffolding
/// ("who is …", "tell me about …"), else fall back to the longest run of
/// capitalized words (a proper noun). `None` = nothing safe to substitute.
fn referent_of(prev_q: &str) -> Option<String> {
    let p = prev_q
        .trim()
        .trim_end_matches(|c: char| matches!(c, '?' | '!' | '.' | ','));
    let p_lower = p.to_lowercase();

    const Q_PREFIXES: &[&str] = &[
        "who is ", "who was ", "who's ", "who are ", "who were ",
        "what is ", "what was ", "what's ", "what are ",
        "tell me more about ", "tell me about ", "tell about ",
        "explain ", "describe ", "define ",
        "кто такой ", "кто такая ", "кто такие ", "что такое ",
        "расскажи мне про ", "расскажи мне о ", "расскажи про ", "расскажи о ",
    ];

    let mut cand: Option<String> = None;
    for pre in Q_PREFIXES {
        if p_lower.starts_with(pre) {
            // Prefixes are lowercase-stable (ASCII + Cyrillic), so byte length
            // matches; .get() keeps it panic-free regardless.
            cand = p.get(pre.len()..).map(|r| r.trim().to_string());
            break;
        }
    }
    if cand.is_none() {
        if let Some(pos) = p_lower.rfind(" about ") {
            cand = p.get(pos + " about ".len()..).map(|r| r.trim().to_string());
        }
    }
    if cand.is_none() {
        cand = capitalized_run(p);
    }

    // Trim quoting and trailing connector debris ("Interstellar about" → "Interstellar").
    let c = cand?;
    let c = c.trim_matches(|c: char| matches!(c, '"' | '\'' | '«' | '»'));
    let mut toks: Vec<&str> = c.split_whitespace().collect();
    while let Some(last) = toks.last() {
        let ll = last.to_lowercase();
        if CONNECTORS.contains(&ll.as_str()) || matches!(ll.as_str(), "mean" | "means" | "like") {
            toks.pop();
        } else {
            break;
        }
    }
    if toks.is_empty() {
        return None;
    }
    let joined = toks.join(" ");
    let jl = joined.to_lowercase();
    let n_chars = joined.chars().count();
    if n_chars < 2 || n_chars > 60 || toks.len() > 6 {
        return None;
    }
    // The previous question must actually name something — a pronoun referring
    // to a pronoun resolves nothing.
    if PRONOUNS.contains(&jl.as_str())
        || matches!(jl.as_str(), "he" | "she" | "they" | "you" | "i" | "we" | "он" | "она" | "они")
    {
        return None;
    }
    Some(joined)
}

/// Longest run of capitalized words ("How tall is David Laid" → "David Laid").
/// A lone capitalized first word is just the sentence start — ignored.
fn capitalized_run(p: &str) -> Option<String> {
    let words: Vec<&str> = p
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();
    let (mut best_start, mut best_len) = (0usize, 0usize);
    let (mut cur_start, mut cur_len) = (0usize, 0usize);
    for (i, w) in words.iter().enumerate() {
        let cap = w.chars().next().map(char::is_uppercase).unwrap_or(false);
        if cap {
            if cur_len == 0 {
                cur_start = i;
            }
            cur_len += 1;
            // ">=" prefers the later run — the freshest topic in the question.
            if cur_len >= best_len {
                best_start = cur_start;
                best_len = cur_len;
            }
        } else {
            cur_len = 0;
        }
    }
    if best_len == 0 || (best_len == 1 && best_start == 0) {
        return None;
    }
    Some(words[best_start..best_start + best_len].join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_pronoun_after_connector() {
        assert_eq!(
            resolve("open video about him", "who is david laid"),
            Some("open video about david laid".to_string())
        );
    }

    #[test]
    fn resolves_trailing_personal_pronoun() {
        assert_eq!(
            resolve("play her", "tell me about Adele"),
            Some("play Adele".to_string())
        );
    }

    #[test]
    fn referent_from_capitalized_run() {
        assert_eq!(
            resolve("open video about him", "how tall is David Laid"),
            Some("open video about David Laid".to_string())
        );
    }

    #[test]
    fn strips_trailing_connector_from_referent() {
        assert_eq!(
            resolve("open video about it", "what is Interstellar about"),
            Some("open video about Interstellar".to_string())
        );
    }

    #[test]
    fn resolves_russian_command() {
        assert_eq!(
            resolve("открой видео про него", "кто такой Хабиб"),
            Some("открой видео про Хабиб".to_string())
        );
    }

    #[test]
    fn conversational_query_untouched() {
        assert_eq!(resolve("what do you think about him", "who is david laid"), None);
    }

    #[test]
    fn command_without_pronoun_untouched() {
        assert_eq!(resolve("open youtube", "who is david laid"), None);
    }

    #[test]
    fn bare_play_it_is_not_a_back_reference() {
        assert_eq!(resolve("play it", "who is David Laid"), None);
    }

    #[test]
    fn pronoun_previous_question_gives_nothing() {
        assert_eq!(resolve("open video about him", "who is he"), None);
        assert_eq!(resolve("open video about him", "hmm ok"), None);
    }
}
