//! Offline eval harness for the anti-hallucination verifier.
//!
//! Measures `verify::verify_grounded` on a labelled fixture set: each case carries
//! an answer, the grounding facts, and whether the answer SHOULD pass (every
//! number/date backed by the facts). Prints a confusion summary and exits non-zero
//! if any case is misclassified, so it can gate CI.
//!
//! Pure: no server, no model, no OS actions. Run: `cargo run --bin eval_verify`.

use nic_assistant_lib::verify::is_grounded;

struct Case {
    name:   &'static str,
    answer: &'static str,
    facts:  &'static str,
    /// true = the answer SHOULD be judged grounded (no fabricated numbers/dates).
    expect: bool,
}

const CASES: &[Case] = &[
    // — grounded (must PASS) —
    Case { name: "grounded number",       answer: "It cost $500.",             facts: "the invoice was $500.00", expect: true },
    Case { name: "thousands separator",   answer: "you spent 1,000",           facts: "1000 spent",              expect: true },
    Case { name: "decimal value",         answer: "pi is 3.14",                facts: "3.14 here",               expect: true },
    Case { name: "grounded month-date",   answer: "June 5 you used VS Code",   facts: "June 5: VS Code session", expect: true },
    Case { name: "single digits ignored", answer: "step 5, point 2 done",      facts: "",                        expect: true },
    Case { name: "bare 'may' verb",       answer: "you may want to rest",      facts: "",                        expect: true },
    Case { name: "time reformatted",      answer: "at 9:30 today",             facts: "logged 09:30 entry",      expect: true },
    Case { name: "no numbers at all",     answer: "Rust is a systems language", facts: "",                       expect: true },
    // — hallucinated (must FAIL) —
    Case { name: "fabricated number",     answer: "It costs $5000.",           facts: "the invoice was $500",    expect: false },
    Case { name: "fabricated year",       answer: "this happened in 2019",     facts: "you logged in 2024",      expect: false },
    Case { name: "fabricated month-date", answer: "you coded on June 5",       facts: "you worked on May 5",     expect: false },
    Case { name: "fabricated IP",         answer: "ip was 10.0.0.5",           facts: "no addresses here",       expect: false },
];

fn main() {
    let (mut pass, mut tp, mut tn, mut fp, mut miss) = (0usize, 0, 0, 0, 0);
    println!("\n=== NIC verifier eval — {} cases ===\n", CASES.len());
    for c in CASES {
        let got = is_grounded(c.answer, c.facts);
        let ok = got == c.expect;
        if ok { pass += 1; }
        match (c.expect, got) {
            (true, true)   => tn += 1,   // grounded, judged grounded
            (false, false) => tp += 1,   // hallucination caught
            (true, false)  => fp += 1,   // false alarm
            (false, true)  => miss += 1, // MISS — fabrication slipped through
        }
        println!("[{}] {:<22} expect={:<5} got={}",
            if ok { "PASS" } else { "FAIL" }, c.name, c.expect, got);
    }
    println!(
        "\n{pass}/{} correct | caught(TP)={tp} clean(TN)={tn} false-alarm(FP)={fp} MISS(FN)={miss}",
        CASES.len()
    );
    if pass != CASES.len() {
        eprintln!("\nEVAL FAILED — verifier regressed.");
        std::process::exit(1);
    }
    println!("\nEVAL OK — verifier matches every labelled case.");
}
