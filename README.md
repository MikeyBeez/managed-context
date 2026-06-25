# Managed Context (v1)

A conversational memory layer that lives outside the model. Each turn is stored; for every
new prompt the model picks (from a compact index of summaries) which earlier turns it needs,
and we answer over only that reduced context. See `PLAN.md` for the design and roadmap.

v1 uses one big model for everything (select, answer, summarize), model-based selection
(no embeddings), and text-level assembly (no KV-cache reuse). Stdlib only — no pip installs.

## Requirements

- Python 3.9+
- A local LLM exposing an **OpenAI-compatible** chat endpoint. Any of these work:
  - Ollama  → `http://localhost:11434/v1`  (default)
  - llama.cpp server → `http://localhost:8080/v1`
  - vLLM / SGLang → `http://localhost:8000/v1`

## Run on pop

With Ollama and a Gemma model:

```
ollama serve                 # if not already running
ollama pull gemma3           # or whatever Gemma you have
python3 managed_context.py --model gemma3 --compare
```

`--model` must match what your endpoint calls the model. For a different server, add
`--base-url`, e.g. `--base-url http://localhost:8000/v1` for vLLM.

Type at the `you>` prompt. After each turn, a line on stderr shows what was selected and how
much smaller the managed context is than the full history:

```
  [ctx] selected=[1] anchor=[4] managed~107 tok | full~151 tok | reduction=29%
```

The store persists to `memory.json`, so the conversation (and its index) survive restarts.

## Try it with no model (offline)

```
python3 test_pipeline.py        # runs the mock pipeline and asserts behavior
python3 managed_context.py --mock --compare   # chat against the deterministic mock
```

The mock model is intentionally dumb (it selects on shared keywords), so it only verifies
the plumbing — store, index, selection parsing, assembly, anchor, metrics, persistence.
Real Gemma selects on meaning and resolves references.

## Flags

- `--model`     model name as the endpoint knows it (default `gemma3`)
- `--base-url`  OpenAI-compatible endpoint (default Ollama)
- `--store`     path to the persisted store (default `memory.json`)
- `--anchor`    how many recent turns to always include as a safety net (default 2)
- `--compare`   also build the full history to report the size reduction
- `--mock`      use the offline mock model (no endpoint needed)

## What to watch for

- **Selection recall** is the metric that matters. A missed turn is silent — the model
  answers confidently on incomplete context. Keep `--anchor` >= 1 and read the `[ctx]` logs.
- The interesting result is *where it breaks*: queries that secretly depend on a turn the
  selector dropped. Those cases are the lesson, and they tell you whether to grow the anchor,
  improve the index summaries, or move to embedding retrieval.
