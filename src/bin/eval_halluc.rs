//! End-to-end hallucination harness (MASTER_PLAN §8c / §9[2] "eval-harness first").
//!
//! Fires factual prompts at a RUNNING NIC backend and checks each answer against
//! must-contain / must-not-contain references, reporting a crude pass rate. This is
//! the scaffold §8c calls for — extend `PROBES` with grounded cases over time.
//!
//! Needs the app running. Reads NIC_LOCAL_TOKEN (set by the launcher) and
//! NIC_API_PORT (default 7878). Run: `cargo run --bin eval_halluc`.
//! Sends HTTP only — it does NOT launch the app and performs no OS actions.

use std::time::Duration;

struct Probe {
    prompt:   &'static str,
    /// Answer should contain at least one of these (empty = no positive check).
    must:     &'static [&'static str],
    /// Answer must contain none of these (fabrication markers).
    must_not: &'static [&'static str],
}

const PROBES: &[Probe] = &[
    Probe { prompt: "Who wrote War and Peace?",          must: &["Tolstoy"], must_not: &[] },
    Probe { prompt: "What is the capital of France?",    must: &["Paris"],   must_not: &[] },
    Probe { prompt: "What is 17 times 4?",               must: &["68"],      must_not: &[] },
    Probe { prompt: "Who are you?",                      must: &["NIC"],     must_not: &[] },
    // No grounding + nothing to recall → the model must NOT invent screen activity.
    Probe { prompt: "What exact app did I use at 03:00 today?",
            must: &[], must_not: &["I watched", "you opened", "you were"] },
];

fn main() {
    let port  = std::env::var("NIC_API_PORT").unwrap_or_else(|_| "7878".into());
    let token = std::env::var("NIC_LOCAL_TOKEN").unwrap_or_default();
    let url   = format!("http://127.0.0.1:{port}/query");
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(90)).build();

    if token.is_empty() {
        eprintln!("NIC_LOCAL_TOKEN is empty — start the app first, or export the token it printed.");
    }
    println!("\n=== NIC hallucination harness (§8c) → {url} ===\n");

    let (mut pass, mut total) = (0usize, 0usize);
    for p in PROBES {
        total += 1;
        let body = serde_json::json!({ "query": p.prompt, "offline": true }).to_string();
        let answer = match agent.post(&url)
            .set("Content-Type", "application/json")
            .set("X-Api-Key", &token)
            .send_string(&body)
        {
            Ok(r) => r.into_string().ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v["answer"].as_str().map(str::to_string))
                .unwrap_or_default(),
            Err(e) => {
                eprintln!("[ERR] {} → {e}\n      (is the NIC app running on port {port}?)", p.prompt);
                continue;
            }
        };
        let has_must = p.must.is_empty() || p.must.iter().any(|m| answer.contains(m));
        let has_bad  = p.must_not.iter().any(|m| answer.contains(m));
        let ok = has_must && !has_bad;
        if ok { pass += 1; }
        println!("[{}] {}\n      → {}\n",
            if ok { "PASS" } else { "FAIL" }, p.prompt, answer.replace('\n', " "));
    }
    println!("{pass}/{total} probes passed.");
    if pass != total { std::process::exit(1); }
}
