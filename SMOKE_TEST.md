# SMOKE TEST — run before every release build

Hand-test pass for NIC-Assistant. The automated suite covers pure logic; this covers
the things only a live run can prove (real LLM, GPU, OS automation, WebView). Takes
~5–10 min. Tick everything before freezing a build.

> Build:  `cargo build --release --bin NIC-Assistant`  (close the running app first)
> Perf:   `cargo run --bin bench`           — load time / VRAM / tok-s
> Halluc: `cargo run --bin eval_halluc`     — baseline (app must be running)

---

## 1. Boot & first run
- [ ] App launches; window appears; status goes **online**.
- [ ] (fresh install) no model → BYO flow: drop-in `.gguf` is found, OR the native
      file picker appears, OR it downloads. App reaches **online** afterwards.
- [ ] (existing install) loads the configured model without prompting.

## 2. Chat & inference (Шаг 3 / §3)
- [ ] A question streams a reply; the `<think>` block is **hidden**, only the answer shows.
- [ ] **Regression guard:** "who is Isaac Newton" → facts about Newton, **not** a
      self-description.
- [ ] Small talk ("hi, how are you") feels natural (this is the temp-0.3 branch).

## 3. Anti-hallucination (§9[2] verify + §3 Phase-2 temp-0)
- [ ] Ask something that triggers a **web search** (e.g. "bitcoin price today"): the
      answer cites figures that match the **Sources** shown. If a figure can't be
      verified, a caveat / raw snippets appear instead of an invented number.
- [ ] **Phase-2 check:** a grounded answer (after a web/screen lookup) reads
      deterministically — re-asking gives the same wording (temp 0 on the reader branch).

## 4. РУЛИТ — Pilot, English commands
- [ ] "open youtube" / "open calculator" / "open notepad" → opens it, English confirm.
- [ ] "volume up" / "volume down" / "mute" → changes volume.
- [ ] "next" / "pause" / "play AC DC" → media control / plays the named track.
- [ ] "screenshot" → saves to Desktop, English confirm.
- [ ] "search youtube for lofi" / "google rust async" → opens / in-chat results.

## 5. ПОМНИТ — memory recall
- [ ] Use the PC for a bit (browser, an app), then ask "what did I do recently?" →
      a concrete English retelling of the actual sites/apps.
- [ ] "find the site I looked at" → returns a real recent page.

## 6. Web (Surfer)
- [ ] "weather" / "latest news" → live results in chat + clickable **Sources**.
- [ ] A definition ("what is recursion") answers locally (no needless web hit).

## 7. BYO-GGUF in-UI (§5/§7) — the untested-blind parts
- [ ] 💾 toolbar button → native file dialog opens *(verifies rfd-from-API-thread)*.
- [ ] Pick a `.gguf` → toast: "Model set… Restart to load it."; after restart it loads.
- [ ] Cancel the dialog → graceful toast, nothing breaks.

## 8. Privacy & security
- [ ] Incognito toggle pauses screen memory; "forget 15 min" erases recent events.
- [ ] A normal browser tab on `localhost:7878` **cannot** read `/query` without the
      token (per-launch token enforced).
- [ ] No outbound network at idle (no telemetry by default).

## 9. Performance / live tuning (record the numbers)
- [ ] `cargo run --bin bench` → note **model-load s / VRAM MB / tok-s / TTFT ms**.
- [ ] Healthy answers are NOT cut off by the two-phase timeout. If they are under load,
      raise TTFT (`llm_utils.rs`, currently 7 s) and/or idle (2 s).
- [ ] `cargo run --bin eval_halluc` (app running) → record the pass rate as a baseline.

---

### Known, accepted limits (don't file as bugs)
- After an idle-abort, the freed llama slot can linger up to ~12 s (socket read timeout).
- TTFT 7 s can false-trigger under cold weight load / GPU contention → graceful
  raw-snippet fallback, not a crash. Tune live if it bites.
