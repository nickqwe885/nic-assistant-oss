# NIC-Assistant — Privacy

NIC is built privacy-first. The short version: **your data never leaves your
computer, and NIC does not read your screen until you say yes.**

## What NIC does

- **Screen memory is opt-in.** On first run NIC captures *nothing*. Not one
  screenshot is taken until you press **Enable memory**. You can turn it back
  off any time (Incognito), and wipe everything with one click
  (Settings → Delete all my data).
- **Everything is local.** Screen text (via on-device OCR), embeddings, the
  vector database, and the language model all live and run on your machine.
- **No cloud AI.** Inference runs locally through `llama-server`. Your questions
  and your memory are never sent to OpenAI, Anthropic, Google, or anyone else.

## Secrets are never remembered

Screen memory is powerful, so NIC actively refuses to remember the things that
could hurt you most. This is enforced **in code**, not promised in a policy:

- **Sensitive windows are never captured at all.** Crypto wallets (MetaMask,
  Trust Wallet, Tonkeeper, Ledger…), password managers (1Password, Bitwarden,
  KeePass…), banking pages, and any window whose title mentions a seed/recovery
  phrase or a password: no screenshot is taken, no text is read, and no record is
  kept that the window was even open.
- **Secrets are stripped before anything is stored.** If a secret still appears
  somewhere unexpected, it is redacted on capture — wallet seed phrases (matched
  against the BIP-39 word list), card numbers (Luhn-checked), private keys, and
  API tokens never reach the database.
- **Copied secrets are dropped entirely.** A seed phrase or card number on your
  clipboard is discarded, not stored.
- **Older memories are filtered too.** The same check runs again on anything
  loaded from memory, so a secret recorded before this feature existed still
  cannot appear in an answer.

All of it is plain deterministic code — no model decides what counts as private.

## What NIC sends over the network — and only then

By default NIC makes **zero** network requests once installed. Network is used
only for these explicit, user-initiated actions:

| Action | What leaves your machine | When |
| --- | --- | --- |
| Web search in chat (Surfer) | Only the search terms of that one question — never your memory or screen | When a question needs live data (weather, news, prices, "who is…") |
| First-run setup | Downloads `llama-server` (from GitHub) and, if you choose, a model (from Hugging Face) | First launch only |
| Check for updates | A request to GitHub Releases | Only when you click "Check for updates" |

NIC ships **no telemetry, no analytics, no crash reporting, no ads.**

## Your controls

- **Enable / disable memory** — screen capture consent (Settings → Privacy).
- **Incognito** — pause memory instantly.
- **Forget 15 min** — erase the last 15 minutes.
- **Export my data** — take everything as NDJSON.
- **Delete all my data** — irreversibly erase the whole memory + archive.
- **Bring your own model** — point NIC at a local `.gguf`; no download at all.

## Where your data lives

- Memory database and logs: next to the app (`data/`, `logs/`).
- Downloaded model + engine: `%LOCALAPPDATA%\nic-assistant\`.

Nothing here is uploaded anywhere. Deleting these folders removes all traces.

## Reporting

NIC has no automatic error reporting. If something breaks, use
**Settings → Copy diagnostics** and paste it into a GitHub issue — it contains
only versions, hardware, sizes and recent log lines, never your memory content.
