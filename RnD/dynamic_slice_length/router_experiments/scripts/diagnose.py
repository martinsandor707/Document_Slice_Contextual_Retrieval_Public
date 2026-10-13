"""Per-question diagnostics: where does the dynamic table lose recall against fixed k=2 (and the baseline)?"""
import json
import os
import sys
from collections import Counter, defaultdict

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import evalkit as ek

qna = ek.qna()
dyn = ek.dynamic_records()
k_of = {r["id"]: r["doc_slice_radius"] for r in dyn}
tier_of = {r["id"]: r["slice_tier"] for r in dyn}
text_of = {r["id"]: r["original_text"] for r in dyn}
ctx_dyn = {r["id"]: r["context"] for r in dyn}
doc_of = {r["id"]: r["document"] for r in dyn}

res = {n: json.load(open(f"{S}/res_{n}.json")) for n in ["ablation_doc_slice_radius_dynamic", "ablation_doc_slice_radius_2", "anthropic_control_table"]}
D, K2, B = (res[n] for n in ["ablation_doc_slice_radius_dynamic", "ablation_doc_slice_radius_2", "anthropic_control_table"])

N = 20
lost, gained = [], []
for qi, q in enumerate(qna):
    rd, rk = D["per_q"][qi][str(N)]["recall"], K2["per_q"][qi][str(N)]["recall"]
    if rd < rk:
        lost.append(qi)
    elif rd > rk:
        gained.append(qi)
print(f"vs fixed k=2 @20: questions worse {len(lost)}, better {len(gained)}, equal {len(qna) - len(lost) - len(gained)}")
tot_loss = sum(K2["per_q"][i][str(N)]["recall"] - D["per_q"][i][str(N)]["recall"] for i in lost)
tot_gain = sum(D["per_q"][i][str(N)]["recall"] - K2["per_q"][i][str(N)]["recall"] for i in gained)
print(f"summed recall lost {tot_loss:.3f} (= {tot_loss/len(qna):.4f} of R@20), gained {tot_gain:.3f} (= {tot_gain/len(qna):.4f})")

# which gold chunks are missed by dynamic but found by k=2, and how were they routed?
missed_k = Counter(); found_k = Counter()
rows = []
for qi in lost + gained:
    q = qna[qi]; gold = set(q["supporting_chunks"])
    topD, topK = set(D["top20"][qi]), set(K2["top20"][qi])
    for g in gold:
        if g in topK and g not in topD:
            missed_k[(k_of[g], tier_of[g])] += 1
            rows.append(("MISS", qi, k_of[g], tier_of[g], g))
        if g in topD and g not in topK:
            found_k[(k_of[g], tier_of[g])] += 1
            rows.append(("GAIN", qi, k_of[g], tier_of[g], g))
print("\ngold chunks found by k=2 but missed by dynamic, by (routed k, tier):", dict(sorted(missed_k.items())))
print("gold chunks found by dynamic but missed by k=2, by (routed k, tier):", dict(sorted(found_k.items())))

# expected miss rate per k if routing were irrelevant: share of gold refs per k
gold_refs = Counter(k_of[c] for q in qna for c in q["supporting_chunks"])
print("gold refs per routed k:", dict(sorted(gold_refs.items())))

print("\n--- missed gold chunks (dynamic) with their routed k and the context they got ---")
for kind, qi, k, tier, g in rows:
    if kind != "MISS":
        continue
    rankD = D["top20"][qi].index(g) + 1 if g in D["top20"][qi] else None
    rankK = K2["top20"][qi].index(g) + 1 if g in K2["top20"][qi] else None
    print(f"q{qi:3d} k={k} {tier:5s} rank k2={rankK} dyn={rankD} | {text_of[g][:80].replace(chr(10),' ')!r}")
    print(f"      ctx(dyn): {ctx_dyn[g][:110]!r}")

# baseline comparison
lostB = [i for i in range(len(qna)) if D["per_q"][i][str(N)]["recall"] < B["per_q"][i][str(N)]["recall"]]
gainB = [i for i in range(len(qna)) if D["per_q"][i][str(N)]["recall"] > B["per_q"][i][str(N)]["recall"]]
print(f"\nvs anthropic baseline @20: worse {len(lostB)}, better {len(gainB)}")
lostKB = [i for i in range(len(qna)) if K2["per_q"][i][str(N)]["recall"] < B["per_q"][i][str(N)]["recall"]]
gainKB = [i for i in range(len(qna)) if K2["per_q"][i][str(N)]["recall"] > B["per_q"][i][str(N)]["recall"]]
print(f"fixed k=2 vs anthropic baseline @20: worse {len(lostKB)}, better {len(gainKB)}")
