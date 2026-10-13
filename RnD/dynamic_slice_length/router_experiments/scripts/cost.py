"""Compute cost of routing policies: summariser calls and prompt text (slice chars) relative to fixed k=2 / k=3."""
import glob
import json
import os
import sys

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import sweep  # noqa: E402  (loads routing_original.json, fixed texts, positions)

RND = sweep.ek.RND
docs = {}
for p in sorted(glob.glob(f"{RND}/split_documents/*.json")):
    d = json.load(open(p))
    docs[os.path.basename(p)] = d


def slice_chars(chunks, i, k):
    if k == 0:
        return 0
    lo, hi = i - k, i + k
    if lo < 0:
        hi, lo = min(len(chunks) - 1, hi - lo), 0
    if hi > len(chunks) - 1:
        lo, hi = max(0, lo - (hi - (len(chunks) - 1))), len(chunks) - 1
    return sum(len(c["text"]) for c in chunks[lo:hi + 1])


def cost(policy):
    calls = chars = 0
    for name, chunks in docs.items():
        for i, c in enumerate(chunks):
            k = policy.get(c["id"], 2)
            if k > 0:
                calls += 1
                chars += slice_chars(chunks, i, k) + len(c["text"])   # FULL DOCUMENT slice + CHUNK repeated in the prompt
    return calls, chars


POLICIES = {
    "fixed k=2": {i: 2 for i in sweep.ids},
    "fixed k=3": {i: 3 for i in sweep.ids},
    "fixed k=1": {i: 1 for i in sweep.ids},
    "genre router (original: A skip, B1, C2, D3, T3)": sweep.remap({}),
    "genre, tables->2": sweep.remap({"T": 2}),
    "genre, tables->1": sweep.remap({"T": 1}),
    "genre, A->1 (never skip)": sweep.remap({0: 1}),
    "genre, A->2 (never skip)": sweep.remap({0: 2}),
    "genre, C->3": sweep.remap({0: 2, 2: 3}),
    "length <700->3 else 2, T3": sweep.length_policy(k_long=2),
    "position edge->1 else 2, T3": sweep.position_policy(),
}
if __name__ == "__main__":
    base_calls, base_chars = cost(POLICIES["fixed k=2"])
    print(f"{'policy':50s} {'calls':>6s} {'prompt Mchars':>13s} {'vs k=2':>7s}")
    for name, pol in POLICIES.items():
        calls, chars = cost(pol)
        print(f"{name:50s} {calls:6d} {chars/1e6:13.2f} {chars/base_chars:7.0%}")
