#!/usr/bin/env python3
"""
Benchmark: full history vs managed context, timed, against a real LLM.

Replays a scripted multi-turn conversation. At each turn we answer the SAME prompt
given the SAME growing history two ways and time each:

  - FULL: send the entire history + prompt to the model (one call).
  - MANAGED: pass-1 select relevant turns from the index, assemble selected + recency
    anchor, pass-2 answer (two calls).

The canonical history (used by both modes going forward) is the FULL-mode answer, so
both modes always face an identical prior conversation. Per-turn wall-clock times and
context sizes are recorded; aggregate stats and a crossover read are printed and saved.

Run on pop against local Ollama:
    python3 benchmark.py --base-url http://localhost:11434/v1 --model gemma4:26b
"""

import argparse
import json
import time

from managed_context import (
    LLMClient, MemoryStore, ManagedContext, ANSWER_SYSTEM,
)

# Scripted conversation: interleaved topics, back-references, one self-contained query.
PROMPTS = [
    "Explain how photosynthesis works, in plain terms.",
    "Now explain cellular respiration and how it differs.",          # refers back to t1
    "Switch topics: name three classic dishes from Paris.",
    "What is the capital of France?",                                 # self-contained
    "Write a Python function to reverse a singly linked list.",
    "Rewrite that function to be iterative instead of recursive.",    # refers to t5
    "Go back to photosynthesis: what role does chlorophyll play?",    # refers to t1/t2
    "Summarize the three Paris dishes you listed earlier.",           # refers to t3
]


def full_messages(history, prompt):
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    for h in history:
        msgs.append({"role": "user", "content": h["user"]})
        msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--anchor", type=int, default=2)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    llm = LLMClient(args.base_url, args.model)
    store = MemoryStore("/tmp/bench_memory.json")
    store.turns = []  # fresh
    store._next_id = 1
    mc = ManagedContext(llm, store, anchor=args.anchor, compare=False, verbose=False)

    history = []   # canonical [{user, assistant}]
    records = []

    print(f"# benchmark: full vs managed | model={args.model} | {len(PROMPTS)} turns\n")
    header = (f"{'turn':>4} {'full_s':>8} {'mgd_s':>8} {'sel_s':>7} {'ans_s':>7} "
              f"{'full_tok':>9} {'mgd_tok':>8} {'selected':>12} {'speedup':>8}")
    print(header)

    for i, prompt in enumerate(PROMPTS, 1):
        # ---- FULL ----
        fmsgs = full_messages(history, prompt)
        full_tok = sum(len(m["content"]) for m in fmsgs) // 4
        t0 = time.perf_counter()
        full_ans = llm.chat(fmsgs, temperature=0.3, max_tokens=args.max_tokens)
        full_s = time.perf_counter() - t0

        # ---- MANAGED ----
        t0 = time.perf_counter()
        selected = mc.select(prompt)
        sel_s = time.perf_counter() - t0
        mmsgs, metrics = mc.assemble(prompt, selected)
        mgd_tok = metrics.managed_chars // 4
        t0 = time.perf_counter()
        _mgd_ans = mc.answer(mmsgs)
        ans_s = time.perf_counter() - t0
        mgd_s = sel_s + ans_s

        # ---- canonical history grows from the FULL answer ----
        summary = mc.summarize(prompt, full_ans)
        store.add(prompt, full_ans, summary)
        history.append({"user": prompt, "assistant": full_ans})

        speedup = full_s / mgd_s if mgd_s > 0 else 0.0
        rec = dict(turn=i, prompt=prompt, full_s=round(full_s, 2),
                   managed_s=round(mgd_s, 2), select_s=round(sel_s, 2),
                   answer_s=round(ans_s, 2), full_tok=full_tok, managed_tok=mgd_tok,
                   selected_ids=metrics.selected_ids, anchor_ids=metrics.anchor_ids,
                   speedup=round(speedup, 2))
        records.append(rec)
        print(f"{i:>4} {full_s:>8.2f} {mgd_s:>8.2f} {sel_s:>7.2f} {ans_s:>7.2f} "
              f"{full_tok:>9} {mgd_tok:>8} {str(metrics.selected_ids or '-'):>12} "
              f"{speedup:>7.2f}x")

    # ---- aggregate ----
    n = len(records)
    sum_full = sum(r["full_s"] for r in records)
    sum_mgd = sum(r["managed_s"] for r in records)
    avg_full_tok = sum(r["full_tok"] for r in records) / n
    avg_mgd_tok = sum(r["managed_tok"] for r in records) / n
    # second half = once history has grown
    half = records[n // 2:]
    h_full = sum(r["full_s"] for r in half)
    h_mgd = sum(r["managed_s"] for r in half)

    summary = dict(
        model=args.model, turns=n,
        total_full_s=round(sum_full, 2), total_managed_s=round(sum_mgd, 2),
        overall_speedup=round(sum_full / sum_mgd, 2) if sum_mgd else 0,
        second_half_speedup=round(h_full / h_mgd, 2) if h_mgd else 0,
        avg_full_tokens=round(avg_full_tok), avg_managed_tokens=round(avg_mgd_tok),
        token_reduction=round(1 - avg_mgd_tok / avg_full_tok, 3) if avg_full_tok else 0,
    )

    print("\n# summary")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
