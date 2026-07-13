# Third-party credits

NIC-Assistant stands on excellent open-source work. Thank you to:

## Local inference
- **llama.cpp** (`llama-server`) — MIT — the engine that runs the language model
  on your own hardware. https://github.com/ggerganov/llama.cpp
- **Qwen2.5** (Alibaba) — the default local language model.
- **all-MiniLM-L6-v2** (Sentence-Transformers) — Apache-2.0 — text embeddings.

## Core stack (Rust)
- **tokio**, **axum** — async runtime and HTTP.
- **tao** + **wry** — native window + WebView2 shell.
- **LanceDB** — on-device vector database for memory.
- **Candle** (Hugging Face) — running the embedder in Rust.
- **reqwest**, **ureq**, **serde**, **regex**, **rfd**, **tracing**.

## Model & data sources
- **Hugging Face** — optional model downloads.
- **GitHub Releases** (llama.cpp) — the inference engine binary.

Full dependency licenses are available via `cargo license`. NIC-Assistant's own
core is licensed AGPL-3.0 (see `LICENSE` and `LICENSING.md`).
