#!/usr/bin/env python3
"""
Offline pipeline test for managed_context.py using the mock model (no LLM needed).

It walks a small conversation that mixes topics, then checks that:
  - the store grows and persists,
  - selection pulls an OLD relevant turn while skipping unrelated middle turns,
  - the recency anchor is always included,
  - context assembly is smaller than full history (--compare style metric),
  - reload from disk round-trips.
"""

import os
import tempfile

from managed_context import LLMClient, MemoryStore, ManagedContext


def run():
    tmp = tempfile.mkdtemp()
    store_path = os.path.join(tmp, "memory.json")

    llm = LLMClient(base_url="", model="mock", mock=True)
    store = MemoryStore(store_path)
    # anchor=1 so the recency window does NOT cover everything; selection has to do work.
    mc = ManagedContext(llm, store, anchor=1, compare=True, verbose=False)

    convo = [
        "Let us discuss photosynthesis and how chloroplasts convert light energy.",   # id 1
        "What about cellular respiration inside mitochondria?",                         # id 2
        "Tell me about French cuisine and Paris restaurants.",                          # id 3
        "What is the capital of France?",                                               # id 4
        "Back to photosynthesis: which light wavelength works best?",                   # id 5
    ]

    last_metrics = None
    for prompt in convo:
        selected = mc.select(prompt)
        messages, metrics = mc.assemble(prompt, selected)
        reply = mc.answer(messages)
        summary = mc.summarize(prompt, reply)
        store.add(prompt, reply, summary)
        store.save()
        last_metrics = metrics
        inc_ids = sorted(set(metrics.selected_ids) | set(metrics.anchor_ids))
        print(f"T{store.turns[-1].id}: select={metrics.selected_ids or '-'} "
              f"anchor={metrics.anchor_ids} included={inc_ids} "
              f"managed={metrics.managed_chars}c full={metrics.full_chars}c "
              f"reduction={metrics.reduction*100:.0f}%")

    # ---- assertions ------------------------------------------------------------------
    assert len(store.turns) == 5, "store should hold 5 turns"

    # Turn 5 ("Back to photosynthesis ...") should select the photosynthesis turn (id 1)
    # via shared keywords, while the anchor pulls only the most recent turn (id 4).
    # The unrelated middle (ids 2, 3) should NOT be included -> real savings.
    sel5 = ManagedContext._parse_select("SELECT: 1")  # parsing sanity
    assert sel5 == [1]

    # Re-derive turn 5's selection deterministically from the store/index.
    store2 = MemoryStore(store_path)               # reload from disk (persistence test)
    assert len(store2.turns) == 5, "reload should round-trip 5 turns"
    mc2 = ManagedContext(LLMClient("", "mock", mock=True), store2, anchor=1,
                         compare=True, verbose=False)
    # remove the last turn so we reproduce the state just before turn 5 was asked
    store2.turns = store2.turns[:4]
    sel = mc2.select(convo[4])
    _, m = mc2.assemble(convo[4], sel)
    inc = sorted(set(m.selected_ids) | set(m.anchor_ids))
    print(f"\nReconstructed turn 5: select={m.selected_ids} anchor={m.anchor_ids} "
          f"included={inc} reduction={m.reduction*100:.0f}%")

    assert 1 in m.selected_ids, "turn 5 should select the photosynthesis turn (id 1)"
    assert 4 in m.anchor_ids, "anchor should include the most recent turn (id 4)"
    assert 2 not in inc and 3 not in inc, "unrelated middle turns should be skipped"
    assert m.managed_chars < m.full_chars, "managed context must be smaller than full history"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    run()
