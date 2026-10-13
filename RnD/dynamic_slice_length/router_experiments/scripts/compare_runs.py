"""Compare two contextualised-chunk JSONs (same chunks, same or different k) : how many summaries differ, and where.
Usage: python compare_runs.py <a.json> <b.json>
"""
import difflib
import json
import sys
from collections import Counter

a = {r["id"]: r for r in json.load(open(sys.argv[1]))}
b = {r["id"]: r for r in json.load(open(sys.argv[2]))}
common = [i for i in a if i in b]
same = sum(1 for i in common if a[i]["context"].strip() == b[i]["context"].strip())
ratios = [difflib.SequenceMatcher(None, a[i]["context"], b[i]["context"]).ratio() for i in common]
print(f"chunks compared: {len(common)} | identical contexts: {same} ({same/len(common):.1%}) | mean similarity of the rest: "
      f"{sum(r for r in ratios if r < 1)/max(1, sum(1 for r in ratios if r < 1)):.2f}")
buckets = Counter("identical" if r == 1 else ("minor edit (>0.8)" if r > 0.8 else ("rephrased (0.5-0.8)" if r > 0.5 else "different (<0.5)")) for r in ratios)
print("similarity buckets:", dict(buckets))
ka = Counter(r.get("doc_slice_radius", "?") for r in a.values()); kb = Counter(r.get("doc_slice_radius", "?") for r in b.values())
print("k distribution a:", dict(ka), "| b:", dict(kb))
ex = [i for i in common if 0 < difflib.SequenceMatcher(None, a[i]["context"], b[i]["context"]).ratio() < 0.5][:4]
for i in ex:
    print(f"\n[{a[i]['original_text'][:60].replace(chr(10),' ')!r}]\n  A: {a[i]['context'][:150]!r}\n  B: {b[i]['context'][:150]!r}")
