"""Headroom analysis: per question, the best fixed k (<=3) vs the baseline; per gold chunk, which k retrieves it best."""
import json
import os
import sys
from collections import Counter, defaultdict

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import evalkit as ek

qna = ek.qna()
names = {0: "ablation_doc_slice_radius_0", 1: "ablation_doc_slice_radius_1", 2: "ablation_doc_slice_radius_2", 3: "ablation_doc_slice_radius_3",
         "none": "hybrid_retrieval", "base": "anthropic_control_table", "dyn": "ablation_doc_slice_radius_dynamic"}
R = {k: json.load(open(f"{S}/res_{n}.json")) for k, n in names.items() if os.path.exists(f"{S}/res_{n}.json")}
print("tables available:", list(R))
N = "20"
for k, r in R.items():
    print(f"  {str(k):5s} R@20 {r['avg'][N]['recall']:.4f}  MRR {r['avg'][N]['mrr']:.5f}  MAR {r['avg'][N]['mar']:.5f}")

fixed_ks = [k for k in [1, 2, 3] if k in R]
# per-question oracle over fixed k in {1,2,3} (and with 'none' allowed)
def oracle(keys):
    return sum(max(R[k]["per_q"][i][N]["recall"] for k in keys) for i in range(len(qna))) / len(qna)
print(f"\nper-question oracle R@20 over k in {fixed_ks}: {oracle(fixed_ks):.4f}")
if "none" in R:
    print(f"per-question oracle R@20 over k in {fixed_ks} + no-context: {oracle(fixed_ks + ['none']):.4f}")
if "base" in R:
    print(f"per-question oracle incl. baseline: {oracle(fixed_ks + ['base']):.4f}   (baseline alone {R['base']['avg'][N]['recall']:.4f})")

# per gold chunk: rank under each k (None if outside top-20), aggregated over the questions that cite it
dyn = ek.dynamic_records()
text_of = {r["id"]: r["original_text"] for r in dyn}
k_dyn = {r["id"]: r["doc_slice_radius"] for r in dyn}
tier = {r["id"]: r["slice_tier"] for r in dyn}
best_k = Counter()
rows = []
for gid in {c for q in qna for c in q["supporting_chunks"]}:
    qs = [i for i, q in enumerate(qna) if gid in q["supporting_chunks"]]
    found = {k: sum(1 for i in qs if gid in R[k]["top20"][i]) for k in fixed_ks + (["none"] if "none" in R else []) + (["base"] if "base" in R else [])}
    if len(set(found[k] for k in fixed_ks)) > 1:      # k matters for this chunk
        bk = max(fixed_ks, key=lambda k: found[k])
        best_k[bk] += 1
        rows.append((gid, found, len(qs)))
print(f"\ngold chunks where the fixed k changes whether they are found: {len(rows)} / {len({c for q in qna for c in q['supporting_chunks']})}; best k among them: {dict(best_k)}")
for gid, found, nq in sorted(rows, key=lambda r: -r[2])[:40]:
    L = len(text_of[gid])
    print(f"  found/{nq}: " + " ".join(f"k{k}={found[k]}" for k in fixed_ks) + (f" none={found['none']}" if 'none' in found else "") + (f" base={found['base']}" if 'base' in found else "")
          + f" | routed {k_dyn[gid]}/{tier[gid]:5s} | {L:5d} chars | {text_of[gid][:70].replace(chr(10),' ')!r}")
