# Licensing — NIC-Assistant

NIC-Assistant uses a **dual-license** model (MASTER_PLAN §6.1). The goal: keep the
core genuinely open and copyleft forever, while letting NIC Corp offer a commercial
license to businesses that cannot comply with the AGPL.

> ⚠️ **Founder decision still required.** This file states the *intended* split. Before
> the public GitHub release, confirm exactly which paths ship as the open core vs. the
> proprietary shell, then move the proprietary parts out of the AGPL tree (or gate them
> behind a separate, unpublished crate). Until then, treat the published repo as
> **AGPL-3.0 in full**.

## 1. `nic-core` — open core (AGPL-3.0)

Strong copyleft (GNU **Affero** GPL v3). The Affero clause matters: anyone who runs a
modified version as a **network service** must also release their source. A competitor
cannot take the core into a closed product — they are legally obliged to open their
whole derivative.

Intended to be AGPL-3.0:
- the Win32 capture layer (Sentinel: `SetWinEventHook`, OCR plumbing)
- the memory engine (LanceDB handling, embeddings, retrieval, summarisation)
- text/OCR cleaning and the deterministic router
- the inference wrapper (`llm_utils`, llama-server management)
- the anti-hallucination verifier (`verify.rs`)

See `LICENSE` for the full AGPL-3.0 text.

## 2. `nic-app` — commercial layer (proprietary, © NIC Corp)

NOT under AGPL. All rights reserved by NIC Corp:
- the polished `ui.html` design / brand assets
- deep OS-automation scripts beyond the basic whitelist
- B2B / compliance / deployment modules

Businesses that want to embed or ship NIC without AGPL obligations buy a **commercial
license**. Contact: proinub09@gmail.com.

## 3. What AGPL-3.0 means for you (the short version)

- ✅ Use it, study it, modify it, run it locally — free, forever.
- ✅ Redistribute it — under AGPL-3.0, with source.
- ⚠️ Offer it (modified) **as a network/SaaS service** → you must publish your source too.
- ❌ Ship it inside a closed-source product → you need the commercial license.

This is not legal advice. The authoritative terms are in `LICENSE` (AGPL-3.0).

## 4. Contributions

By contributing you agree your contribution is licensed under AGPL-3.0. If the project
later needs to relicense contributions for the commercial layer, a separate
Contributor License Agreement (CLA) will be requested — none is required today.
