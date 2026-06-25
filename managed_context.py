#!/usr/bin/env python3
"""
Managed Context — v1.

A conversational memory layer that lives outside the model. Instead of sending the
whole history to the LLM every turn, we keep an ordered store of turns and, for each
new prompt, let the model pick (from a compact index of summaries) which earlier turns
it actually needs. Then we answer over only that reduced context.

v1 deliberately uses ONE big model for everything (select, answer, summarize), does
model-based selection (no embeddings), and assembles context at the text level (no
KV-cache reuse). See PLAN.md for the roadmap.

Stdlib only — no pip installs. Talks to any OpenAI-compatible chat endpoint
(Ollama, llama.cpp server, vLLM, SGLang).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Optional


# --------------------------------------------------------------------------------------
# LLM client (OpenAI-compatible chat) with an offline mock mode for pipeline testing.
# --------------------------------------------------------------------------------------

class LLMClient:
    def __init__(self, base_url: str, model: str, mock: bool = False, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mock = mock
        self.timeout = timeout

    def chat(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 1024) -> str:
        if self.mock:
            return _mock_chat(messages)
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM endpoint call failed at {url}: {e}") from e
        return body["choices"][0]["message"]["content"]


def _mock_chat(messages: list[dict]) -> str:
    """Deterministic stand-in so the pipeline can be tested with no model running."""
    last = messages[-1]["content"]
    if "SELECT:" in last and "## INDEX" in last:
        # Selection task: pick index entries whose summary shares a word with the message.
        index_block = last.split("## INDEX", 1)[1].split("## NEW MESSAGE", 1)[0]
        message = last.split("## NEW MESSAGE", 1)[1]
        msg_words = set(re.findall(r"[a-z]{4,}", message.lower()))
        hits = []
        for line in index_block.splitlines():
            m = re.match(r"\s*\[id (\d+)\]\s*(.*)", line)
            if not m:
                continue
            tid, summary = int(m.group(1)), m.group(2).lower()
            if msg_words & set(re.findall(r"[a-z]{4,}", summary)):
                hits.append(tid)
        return "SELECT: " + (", ".join(map(str, hits)) if hits else "none")
    if "Summarize this exchange" in last:
        body = last.split("User:", 1)[-1]
        words = re.findall(r"\S+", body)[:10]
        return "topic: " + " ".join(words)
    # Answer task: report how much context we were given so assembly is verifiable.
    n_ctx = sum(1 for m in messages if m["role"] == "assistant")
    return f"[mock answer over {n_ctx} context turn(s)]"


# --------------------------------------------------------------------------------------
# Memory store: an ordered list of turns (the linked-list spine) with JSON persistence.
# --------------------------------------------------------------------------------------

@dataclass
class Turn:
    id: int
    user: str
    assistant: str
    summary: str
    ts: float


class MemoryStore:
    def __init__(self, path: str):
        self.path = path
        self.turns: list[Turn] = []
        self._next_id = 1
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.turns = [Turn(**t) for t in raw.get("turns", [])]
            self._next_id = raw.get("next_id", len(self.turns) + 1)
        except FileNotFoundError:
            self.turns = []
            self._next_id = 1

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"next_id": self._next_id, "turns": [asdict(t) for t in self.turns]},
                      f, indent=2, ensure_ascii=False)

    def add(self, user: str, assistant: str, summary: str) -> Turn:
        turn = Turn(id=self._next_id, user=user, assistant=assistant,
                    summary=summary, ts=time.time())
        self._next_id += 1
        self.turns.append(turn)
        return turn

    def build_index(self) -> str:
        return "\n".join(f"[id {t.id}] {t.summary}" for t in self.turns)

    def by_ids(self, ids: list[int]) -> list[Turn]:
        idset = set(ids)
        return [t for t in self.turns if t.id in idset]

    def recent(self, n: int) -> list[Turn]:
        return self.turns[-n:] if n > 0 else []


# --------------------------------------------------------------------------------------
# Orchestration: two-pass select -> assemble -> answer, plus the write path.
# --------------------------------------------------------------------------------------

SELECT_SYSTEM = (
    "You are a memory selector. You choose which earlier conversation turns are needed "
    "to answer a new message. Be precise: include a turn only if it is actually needed, "
    "but resolve references like 'that', 'it', or 'again' to the turns they point to."
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. You are given only the earlier turns that are relevant "
    "to the current message, plus the most recent turns for continuity. Answer the current "
    "message using that context."
)


@dataclass
class TurnMetrics:
    selected_ids: list[int]
    anchor_ids: list[int]
    managed_chars: int
    full_chars: int

    @property
    def reduction(self) -> float:
        if self.full_chars == 0:
            return 0.0
        return 1.0 - self.managed_chars / self.full_chars


class ManagedContext:
    def __init__(self, llm: LLMClient, store: MemoryStore, anchor: int = 2,
                 compare: bool = False, verbose: bool = True, threshold: int = 0):
        self.llm = llm
        self.store = store
        self.anchor = anchor          # always-include recency window (the safety net)
        self.compare = compare        # also build full history to report savings
        self.verbose = verbose
        self.threshold = threshold    # full-history token count below which we DON'T manage
                                      # (full history is cheap + prefix-cached; only manage once big)

    # ---- pass 1: selection -----------------------------------------------------------
    def select(self, prompt: str) -> list[int]:
        if not self.store.turns:
            return []
        index = self.store.build_index()
        user = (
            "A user is continuing a conversation. Below is an INDEX of earlier turns, "
            "each with an id and a one-line summary. Decide which earlier turns are needed "
            "to answer the NEW MESSAGE well. If the message is self-contained, say none.\n\n"
            f"## INDEX\n{index}\n\n## NEW MESSAGE\n{prompt}\n\n"
            "Think briefly about which turns are needed, then on the FINAL line write exactly:\n"
            "SELECT: <comma-separated ids, or the word none>"
        )
        reply = self.llm.chat(
            [{"role": "system", "content": SELECT_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=256,  # reasoning models burn tokens thinking; too few -> empty output
        )
        return self._parse_select(reply)

    @staticmethod
    def _parse_select(reply: str) -> list[int]:
        # Take the LAST "SELECT:" line so a reasoning trace before it can't confuse parsing.
        lines = re.findall(r"SELECT:\s*([^\n]*)", reply, re.IGNORECASE)
        payload = lines[-1] if lines else reply
        if re.search(r"\bnone\b", payload, re.IGNORECASE):
            return []
        return [int(x) for x in re.findall(r"\d+", payload)]

    # ---- assembly --------------------------------------------------------------------
    def assemble(self, prompt: str, selected_ids: list[int]) -> tuple[list[dict], TurnMetrics]:
        anchor_turns = self.store.recent(self.anchor)
        anchor_ids = [t.id for t in anchor_turns]
        include_ids = set(selected_ids) | set(anchor_ids)
        included = [t for t in self.store.turns if t.id in include_ids]  # chronological

        messages = [{"role": "system", "content": ANSWER_SYSTEM}]
        for t in included:
            messages.append({"role": "user", "content": t.user})
            messages.append({"role": "assistant", "content": t.assistant})
        messages.append({"role": "user", "content": prompt})

        managed_chars = sum(len(m["content"]) for m in messages)
        full_chars = managed_chars
        if self.compare:
            full = [{"role": "system", "content": ANSWER_SYSTEM}]
            for t in self.store.turns:
                full.append({"role": "user", "content": t.user})
                full.append({"role": "assistant", "content": t.assistant})
            full.append({"role": "user", "content": prompt})
            full_chars = sum(len(m["content"]) for m in full)

        metrics = TurnMetrics(selected_ids=sorted(selected_ids), anchor_ids=anchor_ids,
                              managed_chars=managed_chars, full_chars=full_chars)
        return messages, metrics

    # ---- pass 2: answer --------------------------------------------------------------
    def answer(self, messages: list[dict]) -> str:
        return self.llm.chat(messages, temperature=0.4, max_tokens=1024)

    # ---- write path ------------------------------------------------------------------
    def summarize(self, user: str, assistant: str) -> str:
        prompt = (
            "Summarize this exchange in one short line (<=15 words), capturing the topic and "
            f"any key facts.\n\nUser: {user}\nAssistant: {assistant}"
        )
        line = self.llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=48
        ).strip().splitlines()
        return line[0].strip() if line else (user[:60] + " ...")

    # ---- full turn -------------------------------------------------------------------
    def chat(self, prompt: str) -> str:
        full_tokens = sum(len(t.user) + len(t.assistant) for t in self.store.turns) // 4
        metrics = None
        if self.threshold and full_tokens < self.threshold:
            # cheap regime: full history is small + prefix-cached -> just use it all.
            msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
            for t in self.store.turns:
                msgs.append({"role": "user", "content": t.user})
                msgs.append({"role": "assistant", "content": t.assistant})
            msgs.append({"role": "user", "content": prompt})
            reply = self.answer(msgs)
            mode = "full"
        else:
            selected = self.select(prompt)
            messages, metrics = self.assemble(prompt, selected)
            reply = self.answer(messages)
            mode = "managed"
        summary = self.summarize(prompt, reply)
        self.store.add(prompt, reply, summary)
        self.store.save()
        if self.verbose:
            print(f"  [mode={mode} full_tok~{full_tokens}]", file=sys.stderr)
            if metrics:
                self._log(metrics)
        return reply

    def _log(self, m: TurnMetrics) -> None:
        approx_tok = lambda c: c // 4
        line = (f"  [ctx] selected={m.selected_ids or '-'} anchor={m.anchor_ids or '-'} "
                f"managed~{approx_tok(m.managed_chars)} tok")
        if self.compare:
            line += (f" | full~{approx_tok(m.full_chars)} tok | "
                     f"reduction={m.reduction*100:.0f}%")
        print(line, file=sys.stderr)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Managed-context chat over a local LLM.")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible endpoint (Ollama default shown).")
    ap.add_argument("--model", default="gemma3", help="Model name as the endpoint knows it.")
    ap.add_argument("--store", default="memory.json", help="Path to the persisted store.")
    ap.add_argument("--anchor", type=int, default=2, help="Recent turns always included.")
    ap.add_argument("--threshold", type=int, default=0,
                    help="Full-history token count below which to skip managing (0 = always manage).")
    ap.add_argument("--compare", action="store_true",
                    help="Also build full history to report size reduction.")
    ap.add_argument("--mock", action="store_true",
                    help="Use the offline mock model (no endpoint needed).")
    args = ap.parse_args()

    llm = LLMClient(args.base_url, args.model, mock=args.mock)
    store = MemoryStore(args.store)
    mc = ManagedContext(llm, store, anchor=args.anchor, compare=args.compare,
                        threshold=args.threshold)

    banner = "managed-context" + (" [mock]" if args.mock else f" [{args.model}]")
    print(f"{banner} — {len(store.turns)} turns loaded. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        reply = mc.chat(prompt)
        print(f"bot> {reply}")


if __name__ == "__main__":
    main()
