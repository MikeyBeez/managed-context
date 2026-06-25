# Managed Context — Plan

A conversational memory layer that sits *outside* the model. Instead of feeding the
whole history into Gemma on every turn, we keep the history as an ordered store of
turns, and for each new prompt we select only the turns that prompt actually needs.

This is the attention-vs-SSM tradeoff lifted to the orchestration layer:

- The full store of turns = keeping every token for exact attention (exact, grows forever).
- A running summary / the index = the SSM's compressed state (cheap, lossy).
- The selector that picks which turns to include = the router ("how much context does this need?").

The point of v1 is to test the **policy** (what to include) cheaply and externally,
with no training and no model surgery. If selective context holds quality while
shrinking the context, that is the evidence worth having before touching architecture.

## v1 scope (deliberately lean)

- **One big model for everything.** Gemma does selection (pass 1) AND generation (pass 2)
  AND turn summarization. No small selector model yet — that is the first optimization later.
- **Model-based selection, not embeddings.** The model reads an index of summaries and
  picks the entry IDs it needs. This is the two-pass "table of contents" design. No vector
  index in v1 (fewer dependencies, runs anywhere). Embedding retrieval is a later option.
- **Text-level context only.** Assemble selected turns as text and run one normal forward
  pass over the reduced context. Correct and simple. KV-cache reuse is explicitly out of
  v1 (it is not "free" — KV is position- and context-dependent; see Deferred below).
- **Recency anchor as a safety net.** Always include the last 2 turns on top of whatever
  the selector picks, so dangling references ("explain that again") do not break.

## Turn lifecycle

1. User sends a prompt.
2. **Pass 1 — select.** Feed Gemma: the new prompt + the index (id -> one-line summary for
   every stored turn) + instructions. It returns the IDs it needs, or `none`.
3. **Assemble.** Pull the full text of the selected turns, union with the recency anchor
   (last 2 turns), order chronologically. Pure string work, no compute.
4. **Pass 2 — answer.** Feed Gemma: assembled context + the new prompt. Get the reply.
5. **Write path.** Summarize the new (prompt, reply) into a one-line index entry. Append
   the turn to the store. Persist to disk.
6. **Metrics.** Log selected IDs, managed-context size, and full-history size so we can see
   the savings turn by turn.

## Components

- `MemoryStore` — ordered list of `Turn`s (the linked list / spine), JSON persistence,
  `build_index()`.
- `Turn` — `{id, user, assistant, summary, ts}`.
- `LLMClient` — thin wrapper over an OpenAI-compatible chat endpoint (works with Ollama,
  llama.cpp server, vLLM, SGLang). Has a `mock` mode for offline pipeline testing.
- `ManagedContext` — orchestrates the lifecycle above; exposes `chat(prompt)`.
- CLI — a small REPL to talk to it; `--mock` to exercise the pipeline with no model;
  `--compare` to also build the full-history prompt and report the size delta.

## Success metrics

- **Context reduction**: managed-context size vs full-history size per turn (want it small).
- **Selection recall**: did the selector grab every turn that was actually needed?
  (The dangerous failure is silent — a missed turn means a confident wrong answer.)
- **Quality hold**: do answers stay as good as full-history answers? (Spot-check first;
  later, judge head-to-head.)
- The interesting output is *where it breaks* — those cases are the lesson.

## Deferred optimizations (later, in rough order)

1. **Small fast model for the librarian.** Use a 1–4B instruct model for pass-1 selection
   and for turn summarization; keep the big model only for pass-2 answering.
2. **Embedding retrieval** as a pre-filter for cheap/obvious cases, escalating to the
   model selector only when ambiguous.
3. **Running summary as an explicit SSM-state anchor**, updated each turn.
4. **KV-cache reuse.** Store per-turn KV and splice it instead of recomputing attention.
   NOT free: RoPE bakes position into the keys (needs re-indexing) and a chunk's KV encodes
   what preceded it (stale when recombined). The real techniques — PromptCache (fixed
   position slots) and CacheBlend (reuse most, selectively recompute the seams) — show it is
   "mostly reuse + a little recompute," not zero. Only worth it once the policy is proven.

## File layout

- `PLAN.md` — this file.
- `managed_context.py` — the v1 implementation (store, index, two-pass, CLI, metrics).
- `README.md` — how to run it on pop against local Gemma.
- `memory.json` — created at runtime; the persisted store.
