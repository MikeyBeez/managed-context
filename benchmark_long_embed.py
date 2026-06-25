#!/usr/bin/env python3
"""
Long-context sweep with EMBEDDING selection (the fair, realistic setup).

- Answer model: gemma4-ctx4k (gemma4:26b pinned to a 4096 window so the ceiling is reachable).
- Selector: nomic-embed-text. Each turn is embedded once on write (~ms); for a new prompt we
  embed it and take cosine top-k over prior turns. No slow LLM selector, no per-turn summary
  call -> managed overhead is just one fast embedding.

Per turn we time the full-history answer and the managed answer (embed-select + answer over
anchor + top-k), track context size, flag the crossover (managed first faster) and the ceiling
(full-history tokens first exceed the window, where the server starts truncating old turns).
"""

import argparse
import json
import math
import time
import urllib.request

from managed_context import LLMClient, ANSWER_SYSTEM

TOPICS = ["photosynthesis", "the French Revolution", "neural networks", "espresso brewing",
          "the Roman aqueducts", "black holes", "sourdough bread", "the printing press"]


def gen_prompts(n, pad=0):
    out = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        if i >= 8 and i % 4 == 0:
            early = TOPICS[(i // 2) % len(TOPICS)]
            base = f"Go back to {early}: expand on the most important point you made about it."
        else:
            base = (f"Tell me one substantive, specific fact about {topic} I might not know "
                    f"(detail #{i // len(TOPICS) + 1}). Two or three sentences.")
        if pad:  # deterministic filler so context grows to the window regardless of answer length
            unit = (f"For context, we are continuing a long discussion that has already "
                    f"touched on {topic} among other subjects. ")
            base = base + " " + (unit * (pad // len(unit) + 1))[:pad]
        out.append(base)
    return out


def embed(base_url, model, text, timeout=60):
    url = base_url.rstrip("/") + "/embeddings"
    data = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["data"][0]["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def full_messages(history, prompt):
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    for h in history:
        msgs.append({"role": "user", "content": h["user"]})
        msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def managed_messages(turns, include_ids, prompt):
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    for t in turns:
        if t["id"] in include_ids:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--answer-model", default="gemma4-ctx4k")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--turns", type=int, default=45)
    ap.add_argument("--pad", type=int, default=0,
                    help="Chars of filler appended to each prompt to grow context to the window faster.")
    ap.add_argument("--answer-tokens", type=int, default=90)
    ap.add_argument("--anchor", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--out", default="benchmark_long_results.json")
    args = ap.parse_args()

    ans = LLMClient(args.base_url, args.answer_model)

    # self-check the embedding endpoint up front
    dim = len(embed(args.base_url, args.embed_model, "hello"))
    print(f"# embed ok ({args.embed_model}, dim={dim})")
    print(f"# long sweep | answer={args.answer_model} ctx={args.num_ctx} | {args.turns} turns\n")
    print(f"{'turn':>4} {'full_tok':>9} {'mgd_tok':>8} {'full_s':>7} {'mgd_s':>7} "
          f"{'emb_s':>6} {'ans_s':>6} {'picked':>10} {'note':>16}")

    turns = []          # [{id,user,assistant,emb}]
    history = []
    records = []
    crossover = None
    ceiling = None

    for i, prompt in enumerate(gen_prompts(args.turns, args.pad), 1):
        # FULL
        fmsgs = full_messages(history, prompt)
        full_tok = sum(len(m["content"]) for m in fmsgs) // 4
        t0 = time.perf_counter()
        full_ans = ans.chat(fmsgs, temperature=0.3, max_tokens=args.answer_tokens)
        full_s = time.perf_counter() - t0

        # MANAGED: embed-select top-k + anchor
        t0 = time.perf_counter()
        if turns:
            qemb = embed(args.base_url, args.embed_model, prompt)
            scored = sorted(((cosine(qemb, t["emb"]), t["id"]) for t in turns), reverse=True)
            picked = [tid for _, tid in scored[:args.topk]]
        else:
            picked = []
        emb_s = time.perf_counter() - t0
        anchor_ids = [t["id"] for t in turns[-args.anchor:]]
        include = set(picked) | set(anchor_ids)
        mmsgs = managed_messages(turns, include, prompt)
        mgd_tok = sum(len(m["content"]) for m in mmsgs) // 4
        t0 = time.perf_counter()
        _ = ans.chat(mmsgs, temperature=0.3, max_tokens=args.answer_tokens)
        ans_s = time.perf_counter() - t0
        mgd_s = emb_s + ans_s

        # write path: embed the new turn (cheap), append
        tid = i
        emb_new = embed(args.base_url, args.embed_model, f"{prompt}\n{full_ans}")
        turns.append({"id": tid, "user": prompt, "assistant": full_ans, "emb": emb_new})
        history.append({"user": prompt, "assistant": full_ans})

        note = ""
        if ceiling is None and full_tok > args.num_ctx:
            ceiling = i; note = "WINDOW EXCEEDED"
        if crossover is None and i > 1 and mgd_s < full_s:
            crossover = i; note = (note + " CROSSOVER").strip()

        records.append(dict(turn=i, full_tok=full_tok, managed_tok=mgd_tok,
                            full_s=round(full_s, 2), managed_s=round(mgd_s, 2),
                            embed_s=round(emb_s, 3), answer_s=round(ans_s, 2),
                            picked=sorted(picked), note=note))
        print(f"{i:>4} {full_tok:>9} {mgd_tok:>8} {full_s:>7.2f} {mgd_s:>7.2f} "
              f"{emb_s:>6.2f} {ans_s:>6.2f} {str(sorted(picked) or '-'):>10} {note:>16}")

    summary = dict(answer_model=args.answer_model, embed_model=args.embed_model,
                   num_ctx=args.num_ctx, turns=len(records),
                   crossover_turn=crossover, ceiling_turn=ceiling,
                   final_full_tok=records[-1]["full_tok"] if records else 0,
                   avg_embed_s=round(sum(r["embed_s"] for r in records) / len(records), 3) if records else 0)
    print("\n# summary")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
