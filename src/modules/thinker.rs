use anyhow::Result;
use std::sync::{Arc, Mutex};
use crate::llm_utils::LlmEngine;
use crate::config::ProfileConfig;

// English-only beta (MASTER_PLAN §9[1]). The model emits flat text in two
// pseudo-HTML blocks — <think> for reasoning, <answer> for the user-facing reply
// — which Qwen2.5-1.5B produces reliably (v8 §3). The frontend shows only the
// <answer>; the backend stores only the <answer> (see `extract_answer`).
//
// `language` is accepted for signature stability but the beta is English-only.
fn build_chat_system(_language: &str) -> String {
    String::from(
        "You are NIC-assistant, a private on-device assistant. Always reply in English.\n\
         Never write Chinese, Japanese or Korean characters.\n\
         \n\
         Output format — always exactly these two blocks, nothing before or after:\n\
         <think>name exactly WHO or WHAT the user is asking about, then which part of the context (if any) answers it</think>\n\
         <answer>the reply for the user — 1-3 sentences of clean plain prose</answer>\n\
         \n\
         Answering:\n\
         — A question about any person, place, thing, event, science, history, math or definition\n\
           (\"who is X\", \"what is X\", \"how does X work\") → answer about THAT subject from your own\n\
           knowledge. If you truly don't know → \"Not sure.\"\n\
         — Greeting / small talk / unclear or nonsense input → reply briefly and naturally, like a person.\n\
         — \"What did I do / open / watch\", \"which apps\", \"what happened\" → use the screen-history\n\
           section of <context>, naming the concrete sites/apps/titles. If there is none → \"Nothing to recall yet.\"\n\
         — News / prices / weather / latest versions → use the live web results in <context>; if absent, say it needs a web check.\n\
         \n\
         Only describe yourself when the user explicitly asks about NIC. A question that names any other\n\
         subject is about THAT subject — never answer about yourself. Never invent facts (say \"Not sure.\"),\n\
         never claim you \"have no access to the device\", never paste raw context or its headers, never add\n\
         a greeting before the answer, and never write anything outside the <think> and <answer> blocks.",
    )
}

const MAX_TOKENS: usize = 400;
const MAX_PROMPT: usize = 2500;

const INTENT_SYSTEM: &str =
    "Classify the user's intent. Output ONLY the code, nothing else.\n\
     \n\
     VOL_UP / VOL_DOWN / VOL_MUTE — volume\n\
     NEXT / PREV / PLAY_PAUSE — media transport\n\
     SCREENSHOT — capture screen\n\
     SITE_OPEN:<url>    — open website\n\
     APP_OPEN:<exe>     — launch application\n\
     YT_SEARCH:<query>  — search YouTube for music or video\n\
     WEB_SEARCH:<query> — search web for LIVE data: news, weather, prices, scores, exchange rates, current events\n\
     MEDIA_PLAY:<name>  — play specific song or artist\n\
     QA                 — everything else: facts, science, math, history, code, definitions, conversation\n\
     \n\
     RULES:\n\
     Keep <query>/<name> in the ORIGINAL language — do NOT translate.\n\
     \"find/search X on youtube\" / \"find a video of X\" → YT_SEARCH\n\
     \"weather\" / \"rate\" / \"news\" / \"google X\" / \"search X\" → WEB_SEARCH\n\
     Science, math, history, definitions, coding → always QA, never WEB_SEARCH.\n\
     \n\
     SITES: yt→https://www.youtube.com  tg→https://web.telegram.org\n\
     dc→https://discord.com/app  gh→https://github.com  steam→https://store.steampowered.com\n\
     reddit→https://www.reddit.com  twitch→https://www.twitch.tv\n\
     \n\
     APPS: calc.exe  notepad.exe  code  chrome  explorer.exe  taskmgr.exe  mspaint.exe\n\
     powershell.exe  spotify.exe  vlc.exe  steam.exe\n\
     \n\
     EXAMPLES:\n\
     youtube→SITE_OPEN:https://www.youtube.com\n\
     twitch→SITE_OPEN:https://www.twitch.tv\n\
     notepad→APP_OPEN:notepad.exe\n\
     calculator→APP_OPEN:calc.exe\n\
     play AC DC→MEDIA_PLAY:AC DC\n\
     put on some lofi→MEDIA_PLAY:lofi\n\
     find lofi on youtube→YT_SEARCH:lofi\n\
     weather today→WEB_SEARCH:weather today\n\
     dollar rate→WEB_SEARCH:dollar rate\n\
     news→WEB_SEARCH:news\n\
     google rust async→WEB_SEARCH:rust async\n\
     louder→VOL_UP\n\
     quieter→VOL_DOWN\n\
     screenshot→SCREENSHOT\n\
     skip→NEXT\n\
     pause→PLAY_PAUSE\n\
     what is gravity→QA\n\
     who wrote War and Peace→QA\n\
     how does TCP work→QA";

/// Fully stateless inference engine — each call is built from the current fact bundle only.
/// No dialogue history is stored; every request is a clean synthesis from Librarian + Surfer data.
pub struct Thinker {
    engine:        Arc<Mutex<LlmEngine>>,
    profile_block: String,
    chat_system:   String,
}

impl Thinker {
    pub fn new(engine: Arc<Mutex<LlmEngine>>, profile: &ProfileConfig, language: &str) -> Self {
        Self {
            engine,
            profile_block: build_profile_block(profile),
            chat_system:   build_chat_system(language),
        }
    }

    /// No-op: kept for backward compatibility with call sites that previously reset history.
    pub fn clear_history(&mut self) {}

    /// Hot-update the user profile block without restarting the binary.
    pub fn update_profile(&mut self, profile: &ProfileConfig) {
        self.profile_block = build_profile_block(profile);
    }

    pub fn answer(&mut self, ict_bundle: &str, query: &str) -> Result<String> {
        let temp = grounded_temp(ict_bundle, query);
        self.engine
            .lock().unwrap()
            .generate(&self.build_prompt(ict_bundle, query), MAX_TOKENS, temp)
    }

    pub fn answer_streaming<F: FnMut(&str)>(
        &mut self,
        ict_bundle: &str,
        query:      &str,
        mut on_token: F,
    ) -> Result<String> {
        let prompt = self.build_prompt(ict_bundle, query);
        let temp   = grounded_temp(ict_bundle, query);
        self.engine.lock().unwrap().generate_stream(&prompt, MAX_TOKENS, temp, &mut on_token)
    }

    /// Quick intent classification for short queries — returns a raw code string.
    /// The caller should fuzzy-match via `contains()` on the uppercased result.
    pub fn classify_intent(&mut self, query: &str) -> String {
        let prompt = format!(
            "<|im_start|>system\n{INTENT_SYSTEM}<|im_end|>\n\
             <|im_start|>user\n{query}<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        self.engine.lock().unwrap()
            .generate(&prompt, 30, 0.0) // classification is deterministic
            .unwrap_or_else(|_| "QA".to_string())
    }

    /// Raw LLM call with a pre-built prompt — used by Analyst and Initiative.
    pub fn generate_raw(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Analyst summaries / note formatting / reminders are faithful reformatting,
        // never invention → decode deterministically at temp 0.
        self.engine.lock().unwrap().generate(prompt, max_tokens, 0.0)
    }

    fn build_prompt(&self, ict_bundle: &str, query: &str) -> String {
        let ctx: String = ict_bundle.chars().take(MAX_PROMPT).collect();
        let profile_prefix = if self.profile_block.is_empty() {
            String::new()
        } else {
            format!("{}\n\n", self.profile_block)
        };

        // Context is wrapped in <context> XML tags so the model treats it as
        // reference material, not as text to continue generating.
        // The actual question comes after </context>, making the boundary clear.
        let has_ctx = !ctx.trim().is_empty() && !ctx.contains("Context is empty");

        let user_content = if has_ctx {
            format!("<context>\n{profile_prefix}{ctx}\n</context>\n\n{query}")
        } else if !profile_prefix.is_empty() {
            format!("{profile_prefix}{query}")
        } else {
            query.to_string()
        };

        format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{user_content}<|im_end|>\n\
             <|im_start|>assistant\n",
            self.chat_system,
        )
    }
}


/// Temperature for the answer branch (Phase 2, MASTER_PLAN v8 §3). When the bundle
/// carries real grounding facts ("Screen history:" / "Web results:") the model must
/// READ them, not improvise — so decode greedily at temp 0, the cheapest guard
/// against recombined-relation drift the numeric verifier can't catch. Ungrounded
/// small-talk keeps a little warmth at 0.3.
fn grounded_temp(ict_bundle: &str, query: &str) -> f32 {
    // Pure small-talk stays warm even if the retriever happened to attach screen
    // context — greetings/thanks aren't factual, so greedy decoding just makes
    // NIC sound robotic. Everything grounded stays greedy (temp 0).
    if is_smalltalk(query) {
        return 0.3;
    }
    if ict_bundle.contains("Screen history:") || ict_bundle.contains("Web results:") {
        0.0
    } else {
        0.3
    }
}

/// Narrow small-talk detector: only clear social openers/closers, so it never
/// warms a real factual question. EN + RU.
fn is_smalltalk(query: &str) -> bool {
    let q: String = query.trim().to_lowercase()
        .chars().filter(|c| c.is_alphanumeric() || c.is_whitespace()).collect();
    let q = q.trim();
    const EXACT: &[&str] = &[
        "hi", "hii", "hey", "hello", "yo", "sup", "hiya",
        "how are you", "how are you doing", "hows it going", "how is it going",
        "whats up", "what's up", "thanks", "thank you", "thx", "ty", "cheers",
        "good morning", "good evening", "good night", "gm", "gn",
        "lol", "haha", "hahaha", "ok", "okay", "cool", "nice", "bye", "goodbye",
    ];
    EXACT.contains(&q)
}

fn build_profile_block(p: &ProfileConfig) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !p.name.is_empty()        { parts.push(format!("Name: {}", p.name)); }
    if !p.role.is_empty()        { parts.push(format!("Role: {}", p.role)); }
    if !p.projects.is_empty()    { parts.push(format!("Projects: {}", p.projects.join(", "))); }
    if !p.preferences.is_empty() { parts.push(format!("Preferences: {}", p.preferences)); }
    if parts.is_empty() { return String::new(); }
    format!("[USER PROFILE]\n{}", parts.join("\n"))
}

/// Extracts the user-facing reply from the model's `<think>/<answer>` output.
///
/// Returns the text inside `<answer>…</answer>`. If those tags are absent (a model
/// that ignored the format), it drops any `<think>…</think>` block and returns the
/// remainder. Always trimmed. This is the backend twin of the frontend tag-strip,
/// so the answer cache / RAG / last-exchange store the clean reply, never the
/// reasoning or the tags. Tags are ASCII, so byte-index slicing stays on char
/// boundaries.
pub fn extract_answer(raw: &str) -> String {
    // A 1.5B model does not always close its tags in order. Live, "what is gravity"
    // came back as "<answer>…</think>\n<answer>…", and reading the FIRST <answer>
    // block shipped the reasoning — tags and all — straight to the user. So close
    // the reasoning first, whatever shape it arrived in, and only then read the reply.
    let mut s = match raw.rfind("</think>") {
        Some(i) => raw[i + "</think>".len()..].to_string(),
        // No closing tag. If reasoning started and never finished, there is no
        // answer here at all — everything from <think> on is thought, not reply.
        None => match raw.find("<think>") {
            Some(i) if !raw.contains("<answer>") => raw[..i].to_string(),
            _ => raw.to_string(),
        },
    };
    if let Some(start) = s.find("<answer>") {
        let after = &s[start + "<answer>".len()..];
        s = match after.find("</answer>") {
            Some(end) => after[..end].to_string(),
            None      => after.to_string(), // truncated mid-answer — keep what we have
        };
    }
    // Whatever the model left lying around, a tag must never reach the user.
    for tag in ["<think>", "</think>", "<answer>", "</answer>"] {
        s = s.replace(tag, "");
    }
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::extract_answer;

    #[test]
    fn pulls_answer_block() {
        assert_eq!(
            extract_answer("<think>he asks who Newton is</think>\n<answer>Isaac Newton was an English physicist.</answer>"),
            "Isaac Newton was an English physicist."
        );
    }

    #[test]
    fn truncated_answer_keeps_partial() {
        assert_eq!(
            extract_answer("<think>x</think><answer>partial reply with no close"),
            "partial reply with no close"
        );
    }

    #[test]
    fn no_tags_returns_text_as_is() {
        assert_eq!(extract_answer("just a plain reply"), "just a plain reply");
    }

    #[test]
    fn think_without_answer_is_stripped() {
        assert_eq!(extract_answer("<think>reasoning</think>the leftover reply"), "the leftover reply");
    }

    #[test]
    fn unterminated_think_dropped() {
        // Still streaming the think block — nothing user-facing yet.
        assert_eq!(extract_answer("<think>still reasoning and not done"), "");
    }

    #[test]
    fn tags_out_of_order_never_leak() {
        // Live, "what is gravity" came back with the tags scrambled and the user
        // was shown the reasoning plus a literal "</think>\n<answer>".
        assert_eq!(
            extract_answer(
                "<answer>The force attracting two masses.</think>\n<answer>Gravity attracts objects with mass."
            ),
            "Gravity attracts objects with mass."
        );
        // Whatever survives, no tag reaches the user.
        assert!(!extract_answer("<answer>a</think><answer>b</answer>").contains('<'));
    }

    // ── Phase 2: grounded → temp 0, chat → 0.3 ────────────────────────────────
    #[test]
    fn grounded_temp_is_zero_for_facts_else_warm() {
        use super::grounded_temp;
        let factual = "who is newton";
        assert_eq!(grounded_temp("Date and time: x\n\nWeb results:\n1. a: b", factual), 0.0);
        assert_eq!(grounded_temp("Date and time: x\n\nScreen history:\nfoo", factual), 0.0);
        // No grounding facts (date/profile only) → keep a little chat warmth.
        assert_eq!(grounded_temp("Date and time: 22.06.2026 14:00", factual), 0.3);
        assert_eq!(grounded_temp("", factual), 0.3);
    }

    #[test]
    fn smalltalk_stays_warm_even_with_screen_context() {
        use super::grounded_temp;
        // Greeting must stay warm despite a retriever-attached screen section.
        assert_eq!(grounded_temp("Screen history:\nfoo", "hey"), 0.3);
        assert_eq!(grounded_temp("Screen history:\nfoo", "how are you"), 0.3);
        // But a real factual query with the same context stays greedy.
        assert_eq!(grounded_temp("Screen history:\nfoo", "what did i do"), 0.0);
    }
}
