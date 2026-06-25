#!/usr/bin/env python3
"""
Needle past the window: does full history FORGET an early fact once the conversation
exceeds the context window, while managed context still retrieves it?

Turn 1 plants a unique fact (the "needle"). We then pad with ~22 unrelated turns so the
full transcript exceeds the answer model's 4096-token window — pushing turn 1 out (Ollama
keeps only the most recent 4096 tokens). Then we ask for the needle two ways:
  FULL: whole transcript (turn 1 truncated away)  -> should fail / hallucinate.
  MANAGED: embed the query, retrieve top-k turns   -> should pull turn 1 and answer.
"""
import argparse, json, math, urllib.request
from managed_context import LLMClient, ANSWER_SYSTEM

NEEDLE_CODE = "ZEPHYR-7-MAGENTA"
NEEDLE = (f"Please remember this for later: the internal launch code for this project is "
          f"{NEEDLE_CODE}. It will matter at the end.")
FILLERS = ["photosynthesis", "the French Revolution", "espresso crema", "Roman aqueducts",
           "black hole accretion", "sourdough starters", "the printing press", "tide tables"]


def embed(base, model, text):
    data = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/embeddings", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["data"][0]["embedding"]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--answer-model", default="gemma4-ctx4k")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--fillers", type=int, default=30)
    ap.add_argument("--pad", type=int, default=700)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    ans = LLMClient(args.base_url, args.answer_model)

    # turn 1 = the needle
    turns = [{"id": 1, "user": NEEDLE, "assistant": "Understood — I'll remember that code.",
              "emb": embed(args.base_url, args.embed_model, NEEDLE)}]
    # padded filler turns
    for i in range(2, args.fillers + 2):
        topic = FILLERS[i % len(FILLERS)]
        u = (f"Tell me something about {topic}. " +
             (f"This is ongoing background discussion of {topic} and related matters. " * 20)[:args.pad])
        a = ans.chat([{"role": "system", "content": ANSWER_SYSTEM},
                      {"role": "user", "content": u}], temperature=0.3, max_tokens=50)
        turns.append({"id": i, "user": u, "assistant": a,
                      "emb": embed(args.base_url, args.embed_model, u + " " + a)})

    q = "What is the internal launch code for this project? Reply with only the code."

    # FULL: whole transcript (oldest turns, incl. the needle, truncated by the 4096 window)
    fmsgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    for t in turns:
        fmsgs += [{"role": "user", "content": t["user"]},
                  {"role": "assistant", "content": t["assistant"]}]
    fmsgs.append({"role": "user", "content": q})
    full_tok = sum(len(m["content"]) for m in fmsgs) // 4
    full_ans = ans.chat(fmsgs, temperature=0.0, max_tokens=256)  # reasoning headroom

    # MANAGED: retrieve top-k by embedding
    qe = embed(args.base_url, args.embed_model, q)
    picked = [tid for _, tid in sorted(((cosine(qe, t["emb"]), t["id"]) for t in turns),
                                       reverse=True)[:args.topk]]
    mmsgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    for t in turns:
        if t["id"] in picked:
            mmsgs += [{"role": "user", "content": t["user"]},
                      {"role": "assistant", "content": t["assistant"]}]
    mmsgs.append({"role": "user", "content": q})
    mgd_tok = sum(len(m["content"]) for m in mmsgs) // 4
    mgd_ans = ans.chat(mmsgs, temperature=0.0, max_tokens=256)  # reasoning headroom

    print(f"needle  : {NEEDLE_CODE} (planted in turn 1)")
    print(f"window  : 4096 tokens | full transcript = {full_tok} tokens "
          f"(turn 1 is {'INSIDE' if full_tok <= 4096 else 'TRUNCATED OUT'})")
    print(f"managed : picked turns {sorted(picked)} = {mgd_tok} tokens")
    print(f"\nFULL    answer: {full_ans.strip()!r}")
    print(f"MANAGED answer: {mgd_ans.strip()!r}")
    print(f"\nFULL    correct: {NEEDLE_CODE in full_ans}")
    print(f"MANAGED correct: {NEEDLE_CODE in mgd_ans}")


if __name__ == "__main__":
    main()
