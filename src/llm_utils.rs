use anyhow::{anyhow, Result};
use std::io::BufRead;
use tracing::info;

/// Thin synchronous wrapper around the llama-server HTTP API.
///
/// Uses `ureq` (pure-sync, no tokio runtime) so it is safe to call from
/// `tokio::task::spawn_blocking` — same thread pool used by Thinker, Surfer, Librarian.
pub struct LlmEngine {
    url:   String,
    agent: ureq::Agent,
}

impl LlmEngine {
    pub fn new(base_url: impl Into<String>) -> Self {
        let url = format!("{}/completion", base_url.into());
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            // Per-read (per-token) socket timeout. Healthy generation streams tokens
            // every few ms, so this only bounds how long a *stalled* read lingers
            // before the reader thread exits and frees the llama slot. The real
            // user-facing budgets (TTFT / idle) are enforced by the watchdog below.
            .timeout_read(std::time::Duration::from_secs(12))
            .build();
        info!("[LLM] engine → {}", url);
        Self { url, agent }
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        self.generate_stream(prompt, max_tokens, temperature, &mut |_| {})
    }

    pub fn generate_stream<F: FnMut(&str)>(
    &mut self,
    prompt:      &str,
    max_tokens:  usize,
    temperature: f32,
    on_token:    &mut F,
        ) -> Result<String> {
            let body = serde_json::json!({
                "prompt":            prompt,
                "n_predict":         max_tokens,
                "stream":            true,
                // Phase 2 (MASTER_PLAN v8 §3): the grounded/factual "reader" branch
                // passes 0.0 so the model reads the supplied facts instead of drifting
                // — temperature is the cheapest guard against recombined-relation
                // errors the numeric verifier can't catch; chat keeps a little warmth
                // (0.3). Loops are handled by repeat_penalty, never by raising temp.
                "temperature":       temperature,
                "repeat_penalty":    1.15,
                "repeat_last_n":     256,
                "top_p":             0.9,
                "top_k":             40,
                "min_p":             0.05,
                "tfs_z":             1.0,
                "stop":              [
                    "<|im_end|>", "</s>", "<|endoftext|>",
                    "</context>", "User:", "User :", "Assistant:"
                ]
            });

        let resp = self.agent
            .post(&self.url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .map_err(|e| anyhow!("llama-server unreachable: {}", e))?;

        // ── Two-phase timeout (MASTER_PLAN §3) ───────────────────────────────────
        // A dedicated reader thread forwards parsed SSE tokens over a channel; the
        // caller thread applies the budgets:
        //   • TTFT — up to 7 s for the FIRST token (prompt eval / lazy weight load).
        //   • Idle — once streaming, ≤2 s between tokens; a longer gap = a stall, so
        //     Rust aborts. Dropping the receiver makes the reader thread's next send
        //     fail (or its read time out), so it exits and closes the connection —
        //     freeing the single llama-server slot instead of hanging it.
        // On abort we return a sentinel error (NIC_TTFT_TIMEOUT / NIC_IDLE_TIMEOUT)
        // so the API layer can show the raw-snippet fallback rather than a failure.
        let reader = std::io::BufReader::new(resp.into_reader());
        let (tx, rx) = std::sync::mpsc::channel::<SseMsg>();
        std::thread::spawn(move || {
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        if let Some((tok, stop)) = parse_sse_line(&l) {
                            if tx.send(SseMsg::Token { tok, stop }).is_err() { return; }
                            if stop { return; }
                        }
                    }
                    Err(_) => { let _ = tx.send(SseMsg::End); return; }
                }
            }
            let _ = tx.send(SseMsg::End);
        });

        const TTFT: std::time::Duration = std::time::Duration::from_secs(7);
        const IDLE: std::time::Duration = std::time::Duration::from_millis(2000);

        let mut result = String::new();
        let mut got_first = false;
        let mut timed_out = false;
        loop {
            let budget = if got_first { IDLE } else { TTFT };
            match rx.recv_timeout(budget) {
                Ok(SseMsg::Token { tok, stop }) => {
                    got_first = true;
                    if !tok.is_empty() {
                        on_token(&tok);
                        result.push_str(&tok);
                    }
                    if stop { break; }
                }
                Ok(SseMsg::End) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)      => { timed_out = true; break; }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        drop(rx); // reader thread exits on its next send/read — frees the llama slot.

        if timed_out {
            let kind = if result.is_empty() { "NIC_TTFT_TIMEOUT" } else { "NIC_IDLE_TIMEOUT" };
            info!("[LLM] {} — {} chars before abort", kind, result.len());
            return Err(anyhow!("{}", kind));
        }
        info!("[LLM] Done — {} chars", result.len());
        Ok(result)
    }
}

/// Message from the SSE reader thread to the timeout-watched caller (see §3).
enum SseMsg {
    Token { tok: String, stop: bool },
    /// Stream ended (explicit `stop`, EOF, or a read error — caller keeps the partial).
    End,
}

/// Parses one SSE line from llama-server.
/// Returns `Some((token, is_stop))` if the line is a valid JSON event, `None` to skip.
pub(crate) fn parse_sse_line(line: &str) -> Option<(String, bool)> {
    if line.is_empty() { return None; }
    let json_str = line.strip_prefix("data: ").unwrap_or(line);
    if json_str.is_empty() { return None; }
    let val = serde_json::from_str::<serde_json::Value>(json_str).ok()?;
    let tok  = val["content"].as_str().unwrap_or("").to_string();
    let stop = val["stop"].as_bool().unwrap_or(false);
    Some((tok, stop))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LlmEngine::new URL building ───────────────────────────────────────────

    #[test]
    fn engine_url_appends_completion() {
        let e = LlmEngine::new("http://127.0.0.1:8090");
        assert_eq!(e.url, "http://127.0.0.1:8090/completion");
    }

    #[test]
    fn engine_url_no_double_slash() {
        let e = LlmEngine::new("http://127.0.0.1:8090");
        assert!(!e.url.contains("//completion"));
    }

    #[test]
    fn engine_url_custom_port() {
        let e = LlmEngine::new("http://127.0.0.1:9999");
        assert!(e.url.contains("9999"));
    }

    // ── parse_sse_line ────────────────────────────────────────────────────────

    #[test]
    fn sse_empty_line_returns_none() {
        assert!(parse_sse_line("").is_none());
    }

    #[test]
    fn sse_data_prefix_stripped() {
        let line = r#"data: {"content":"hello","stop":false}"#;
        let (tok, stop) = parse_sse_line(line).unwrap();
        assert_eq!(tok, "hello");
        assert!(!stop);
    }

    #[test]
    fn sse_raw_json_no_prefix() {
        let line = r#"{"content":"world","stop":false}"#;
        let (tok, stop) = parse_sse_line(line).unwrap();
        assert_eq!(tok, "world");
        assert!(!stop);
    }

    #[test]
    fn sse_stop_true_detected() {
        let line = r#"data: {"content":"","stop":true}"#;
        let (_, stop) = parse_sse_line(line).unwrap();
        assert!(stop);
    }

    #[test]
    fn sse_stop_false_not_stop() {
        let line = r#"{"content":"token","stop":false}"#;
        let (_, stop) = parse_sse_line(line).unwrap();
        assert!(!stop);
    }

    #[test]
    fn sse_missing_stop_field_defaults_false() {
        let line = r#"{"content":"hi"}"#;
        let (tok, stop) = parse_sse_line(line).unwrap();
        assert_eq!(tok, "hi");
        assert!(!stop);
    }

    #[test]
    fn sse_missing_content_field_empty_string() {
        let line = r#"{"stop":false}"#;
        let (tok, _) = parse_sse_line(line).unwrap();
        assert_eq!(tok, "");
    }

    #[test]
    fn sse_invalid_json_returns_none() {
        assert!(parse_sse_line("not json").is_none());
    }

    #[test]
    fn sse_data_prefix_only_returns_none() {
        assert!(parse_sse_line("data: ").is_none());
    }

    #[test]
    fn sse_cyrillic_content_preserved() {
        let line = r#"{"content":"Привет","stop":false}"#;
        let (tok, _) = parse_sse_line(line).unwrap();
        assert_eq!(tok, "Привет");
    }

    #[test]
    fn sse_content_with_spaces() {
        let line = r#"{"content":" токен ","stop":false}"#;
        let (tok, _) = parse_sse_line(line).unwrap();
        assert_eq!(tok, " токен ");
    }

    #[test]
    fn sse_content_with_newline_escape() {
        let line = "{\"content\":\"line1\\nline2\",\"stop\":false}";
        let (tok, _) = parse_sse_line(line).unwrap();
        assert!(tok.contains('\n'));
    }

    #[test]
    fn sse_stop_true_with_empty_content() {
        let line = r#"{"content":"","stop":true}"#;
        let (tok, stop) = parse_sse_line(line).unwrap();
        assert!(tok.is_empty());
        assert!(stop);
    }

    #[test]
    fn sse_numeric_content_ignored_as_empty() {
        // content is not a string → defaults to ""
        let line = r#"{"content":42,"stop":false}"#;
        let (tok, _) = parse_sse_line(line).unwrap();
        assert_eq!(tok, "");
    }
}
