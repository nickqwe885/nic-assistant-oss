# NIC Assistant

**A private AI assistant that lives on your device — and remembers.**

NIC runs fully on your own machine. It quietly builds a searchable, time-stamped memory of what you do (screen, clipboard, files), and answers your questions using that memory — without sending anything to the cloud.

> Your second brain should belong to you, not a corporation.

<!-- TODO: insert a 5-second demo GIF here — ask NIC "what was I doing an hour ago?" and it answers from memory -->

---

## Why it's different

- **Local-first.** The model and your data run on your hardware. Works offline after setup.
- **Memory with a timeline.** NIC understands *yesterday*, *an hour ago*, *right now* — not just keywords.
- **Private by design.** Nothing leaves your device. Incognito mode stores nothing. Cold archive is AES-256-GCM encrypted.
- **Never remembers your secrets.** Wallets, password managers and banking windows are never captured; seed phrases, card numbers and API keys are stripped before anything is stored. Enforced in code — see [PRIVACY.md](PRIVACY.md).
- **Adapts to your hardware.** Picks a model that fits your RAM/GPU automatically — runs on modest machines, scales up on strong ones.
- **The LLM is just an engine.** All the context-building (memory, web, intent) is done by local modules, so the model answers from prepared facts instead of guessing.

## Download & run

1. Download the latest `NIC-Assistant-*.zip` from [Releases](https://github.com/nickqwe885/nic-assistant-releases/releases).
2. Unzip the whole folder, then run `NIC-Assistant.exe`.
3. On first launch it downloads the local model + inference server (~1 GB, one time, needs internet). After that — offline.

**Notes**
- Keep all files from the archive in the same folder.
- First launch can take a minute while the model downloads.
- Nothing is captured until you press **Enable memory** — see [PRIVACY.md](PRIVACY.md).

### "Windows says unknown publisher" — yes, and here's why

NIC is **not code-signed yet** (a certificate costs a few hundred dollars a year;
it's the first thing donations will buy). So on first run Windows SmartScreen shows
*"Windows protected your PC"* → click **More info → Run anyway**.

Some antivirus software may also flag it. That's a **false positive with an honest
cause**: NIC reads the screen and can press keys — the same capabilities malware
uses. The difference is that everything NIC does is in the open:

- **The full source is public** (this repo) — read the screen-capture and
  key-press code yourself, and build it from source if you'd rather not trust the binary.
- **Zero outgoing traffic by default** — verify with a firewall, or just turn Wi-Fi off
  and watch it keep working. The only network calls are the one-time model download
  and web search when *you* explicitly ask a question that needs it.
- **Reproducible check:** every release lists SHA-256 hashes, and you can scan
  the exact zip on VirusTotal before running it.

If you don't want to trust a binary at all: `cargo build --release`.

### Gaming / anti-cheat

NIC **automatically pauses screen memory** whenever a game with kernel-level
anti-cheat (Valorant/Vanguard, CS2, Faceit, EAC, BattlEye…) is in the foreground —
no screenshot is taken while you play, and the app shows a badge confirming it.
If you'd rather be extra safe, close NIC before ranked games.

## How it works

```
Your question
   ↓
Local modules build the context:
   • Librarian  — semantic + time-filtered search over your memory
   • Surfer     — live web data when needed
   • Intent     — volume / apps / media / search commands
   ↓
A compact, structured prompt → local LLM (Qwen / Llama)
   ↓
Answer — grounded in your data, on your machine
```

## Privacy

All data stays on your device. NIC does not phone home. Memory, logs, caches and databases are local user data and are never uploaded.

## Status

Early beta, Windows x64 (Linux via Vulkan also supported). Built in Rust. Actively developed — feedback and bug reports welcome via Issues.

## Support

NIC is free and runs entirely on your hardware — there are no servers to pay for, only development time. If it's useful to you:

- ⭐ Star the repo — it genuinely helps others find it
- 🐛 [Report bugs](https://github.com/nickqwe885/nic-assistant-releases/issues) — every report makes the beta better
- 💛 Donate — USDT (TRC-20): `TLQMML6haVqjAt8jVUrwJEozhpbXHYA46n`

---

*Built by a 16-year-old, in Rust. Private, local, yours.*
