# Benchmark Results — full history vs managed context

Run on **pop** (RTX 5070 Ti), model **gemma4:26b** via Ollama, 8-turn scripted conversation,
each turn answered both ways and timed. Raw data in `benchmark_results.json`.

## Headline: managed context LOST here

| metric | value |
|---|---|
| total full-history time | 56.0 s |
| total managed-context time | 151.1 s |
| overall speedup (full/managed) | **0.37× (≈2.7× slower)** |
| second-half speedup | **0.19× (≈5× slower)** |
| avg full-context tokens | 109 |
| avg managed-context tokens | 84 |
| token reduction | 23% |

Per turn (`full_s` = full history answer; `mgd_s` = select + managed answer):

```
turn  full_s  mgd_s  sel_s  ans_s  full_tok  mgd_tok  speedup
  1   28.28  20.65   0.00  20.65      61       61     1.37x   (cold start / model load)
  2    3.97  23.88   2.25  21.63      74       74     0.17x
  3    4.14  13.16   2.25  10.91      87       87     0.31x
  4    2.75   5.72   2.26   3.46      95       82     0.48x
  5    4.20  23.98   2.30  21.67     117       92     0.18x
  6    4.19  21.84   2.33  19.51     132       94     0.19x
  7    4.24  17.55   2.30  15.25     147       92     0.24x
  8    4.26  24.32   2.33  21.99     160       91     0.18x
```

## Why managed lost (both causes are real, not artifacts)

1. **Prefix caching makes append-only full history almost free to prefill.** The full prompt
   grows by appending; each turn shares a long prefix with the last one, so Ollama reuses the
   cached KV and only prefills the new tokens. Result: `full_s` stays flat at ~4 s even as the
   context grows 74 → 160 tokens. The managed path assembles a *different* subset each turn, so
   its prefix does **not** match the cache — it re-prefills from scratch every time, making the
   managed answer ~20 s vs the full answer's ~4 s, even though it has fewer tokens. This is
   exactly the KV-cache caveat from PLAN.md: reordering/subsetting the context forfeits prefix
   caching.

2. **The select call is pure added latency.** Every managed turn pays an extra ~2.3 s for
   pass-1 selection on top of the answer.

So managed pays a cache penalty *and* an extra call, while the conversation is too short for the
token savings (23%) to matter. Append-only + prefix caching is hard to beat at this length.

## A real bug surfaced: the selector returned empty every turn

`selected_ids` was `[]` on all 8 turns — managed context was effectively **recency-anchor only**
(last 2 turns), never doing intelligent retrieval. Cause: **gemma4:26b is a reasoning model**;
the select call's 64-token budget was consumed by its thinking, leaving empty final content.

Confirmed in `diag2.py`:
- 16-token budget → empty output (even for "reply READY").
- 256-token budget, reason-then-answer prompt → `"...refers back to photosynthesis in [id 1]... SELECT: [id 1]"` → parses `[1]` correctly.

Fix applied to `managed_context.py`: select call now uses 256 tokens and a reason-then-`SELECT:`
prompt; `_parse_select` reads the last `SELECT:` line.

## What this means

- As a **latency** play, managed context does **not** win on short conversations against a
  prefix-caching server. It would lose *harder* with the selector fixed, because a 26B reasoning
  model is a slow, token-hungry selector (~10–20 s once it actually thinks).
- The regimes where it can pay off:
  - **Long conversations**, where full context is large enough that even cached prefill + raw
    size cost exceeds the managed subset (the crossover this 8-turn run never reaches).
  - **A small, fast, non-reasoning selector** (the plan's first optimization) so pass-1 costs
    tens of milliseconds, not seconds.
  - **Cache-aware assembly** (stable prefix ordering, or KV reuse à la PromptCache/CacheBlend)
    so the managed path stops forfeiting prefix caching.
- Honest summary: managed context is a **memory/scaling** mechanism (fit very long histories,
  control what the model sees), not a free **latency** win. On short chats with prefix caching,
  plain full history is faster.

## Part 2 — Long context, embedding selection (gemma4-ctx4k, 4096 window)

Switched selection to fast embedding retrieval (`nomic-embed-text`, ~38 ms/turn) — the realistic
production choice — and pinned the answer model to a 4096-token window so the ceiling is
reachable. Padded prompts grow context ~205 tokens/turn. Raw: `benchmark_long_ceiling.json`.

### Latency curve (30 turns, full_tok 252 → 6120)

| region | full_tok | full_s | managed_s |
|---|---|---|---|
| under window | ~450–3700 | flat ~2.5 s | ~3.5–3.9 s |
| at window (turn 20) | 4105 | ~2.5 s | ~4 s |
| well past window (turns 29–30) | ~6000 | **~5.5 s** | **~3.2 s** |

- Under and around the window, **full history wins**: prefix caching keeps its prefill cheap
  (full_s flat ~2.5 s as context grows), while managed re-prefills its bounded ~1060-token
  subset cold each turn (~3.5–4 s, with occasional ~12 s reload spikes under VRAM pressure).
- Only **well past the window (~1.5× ≈ 6000 tokens)** does full_s climb to ~5.5 s — Ollama is
  truncating heavily each turn and churning the cache — while managed stays flat ~3.2 s. That's
  the latency crossover, and it's far beyond "the context got big."
- Embedding selection is effectively free (~38 ms) and correctly retrieves relevant old turns.

### The window ceiling — full history FORGETS (needle test)

Planted a code in turn 1 (`ZEPHYR-7-MAGENTA`), padded the conversation to 5,675 tokens (past the
4096 window), then asked for the code:

| path | context | answer | correct |
|---|---|---|---|
| full history | 5,675 tok (turn 1 truncated out) | `ZEPHYR-` | ✗ |
| managed | retrieved turns [1, 22, 30] = 479 tok | `ZEPHYR-7-MAGENTA` | ✓ |

Full history's answer was literally chopped at the truncation boundary — it kept a corrupted
fragment (`ZEPHYR-`) of the early turn and lost the rest. Managed embedded the query, pulled
turn 1 back in, and answered correctly from 479 tokens.

## Bottom line

Managed context is a **memory / capability** mechanism, not a latency optimization:
- It does **not** beat full history on speed until the conversation is ~1.5× past the window —
  prefix caching makes append-only full history very hard to beat on latency.
- Its real payoff is **remembering past the window**: once the transcript exceeds the model's
  context, full history silently truncates and forgets (or corrupts) old turns, while managed
  selectively retrieves the relevant ones and stays small.
- Practical design: keep full history while it fits the window (the `--threshold`), switch to
  managed (fast embedding selector + recency anchor) as you approach the window, to keep
  remembering past it. Latency is a wash until then; capability is the win.
- Selector choice matters: a slow reasoning LLM as selector loses (Part 1); a fast embedding
  model is effectively free. Avoid reasoning models for selection.
