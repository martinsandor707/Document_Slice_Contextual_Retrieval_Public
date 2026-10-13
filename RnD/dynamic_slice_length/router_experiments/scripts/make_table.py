"""Compile every measurement (fixed tables, real runs, offline simulations) into one markdown table.
Usage: python make_table.py   (run from the directory that holds res_*.json / sweep_results.jsonl / real_runs/)
"""
import glob
import json
import os
import sys

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import sweep  # noqa: E402  (for cost + routing)
from cost import cost, POLICIES as _COST_POLICIES  # noqa: E402

base_calls, base_chars = cost({i: 2 for i in sweep.ids})


def avg(res_file):
    r = json.load(open(res_file))
    a = r["avg"]
    a = {int(k): v for k, v in a.items()}
    return a


REAL_TABLES = [
    ("Anthropic full-document baseline", "res_anthropic_control_table.json", "real (paper)", None),
    ("Hybrid retrieval, no context", "res_hybrid_retrieval.json", "real (paper)", {i: 0 for i in sweep.ids}),
    ("Fixed k=0 (chunk-only summary)", "res_ablation_doc_slice_radius_0.json", "real (paper)", None),
    ("Fixed k=1", "res_ablation_doc_slice_radius_1.json", "real (paper)", {i: 1 for i in sweep.ids}),
    ("Fixed k=2", "res_ablation_doc_slice_radius_2.json", "real (paper)", {i: 2 for i in sweep.ids}),
    ("Fixed k=3", "res_ablation_doc_slice_radius_3.json", "real (paper)", {i: 3 for i in sweep.ids}),
    ("Dynamic router, original config (Martin's run)", "res_ablation_doc_slice_radius_dynamic.json", "real", sweep.remap({})),
    ("Constant k=2 through the dynamic notebook (FIXED_K=2)", "res_real_fixed2.json", "real (re-generation)", {i: 2 for i in sweep.ids}),
]


def row(label, a, kind, policy, note=""):
    if policy is None:
        cost_s = "—"
    else:
        calls, chars = cost(policy)
        cost_s = f"{calls} / {chars / base_chars:.0%}"
    return f"| {label} | {kind} | {a[20]['recall']:.4f} | {a[20]['mrr']:.4f} | {a[20]['mar']:.3f} | {a[5]['recall']:.4f} | {a[10]['recall']:.4f} | {cost_s} | {note} |"


lines = ["| Configuration | Kind | R@20 | MRR@20 | MAR@20 | R@5 | R@10 | summariser calls / prompt text vs k=2 | Note |",
         "|---|---|---|---|---|---|---|---|---|"]
for label, f, kind, pol in REAL_TABLES:
    if os.path.exists(os.path.join(S, f)):
        lines.append(row(label, avg(os.path.join(S, f)), kind, pol))
# real runs recorded by run_pipeline.py
for f in sorted(glob.glob(os.path.join(S, "real_runs", "*.json"))):
    if f.endswith("_chunks.json"):
        continue
    rec = json.load(open(f))
    m = rec["metrics"]
    if rec["tag"] == "fixed2":
        continue
    chunks_f = os.path.join(S, "real_runs", f"{rec['tag']}_chunks.json")
    recs_ = json.load(open(chunks_f)) if os.path.exists(chunks_f) else []
    pol = {r["id"]: int(r["doc_slice_radius"]) for r in recs_} or None
    table_ks = sorted({int(r["doc_slice_radius"]) for r in recs_ if r.get("slice_tier") == "table"})
    kd = ""
    for s in rec["generation_summary"]:
        if s.startswith("Suggested radius distribution"):
            kd = s.split(":", 1)[1].split("(")[0].strip()
    label = {"tables1": "genre router, tables→1 (exemplars from benchmark papers)", "tables1_run2": "genre router, tables→1 (identical repeat)",
             "binary_t2": "2-class router (S skip, N→2), tables→2 (exemplars from benchmark papers)",
             "binary_t2_heldout": "2-class router (S skip, N→2), tables→2, exemplars from the HELD-OUT papers (final)"}.get(rec["tag"], rec["tag"])
    note = f"{label}; k {kd}; table k={table_ks}; generation {rec['gen_minutes']} min"
    res_f = os.path.join(S, f"res_real_{rec['tag']}.json")
    if os.path.exists(res_f):
        lines.append(row(f"Real run `{rec['tag']}`", avg(res_f), "real", pol, note))
    else:
        calls, chars = cost(pol) if pol else (0, 0)
        lines.append(f"| Real run `{rec['tag']}` | real | {m['20']['RECALL']['own']:.4f} | {m['20']['MRR']['own']:.4f} | "
                     f"{m['20']['MAR']['own']:.3f} | {m['5']['RECALL']['own']:.4f} | {m['10']['RECALL']['own']:.4f} | {calls} / {chars / base_chars:.0%} | {note} |")
# simulations
sims = [json.loads(l) for l in open(os.path.join(S, "sweep_results.jsonl"))]
seen = set()
for r in sorted(sims, key=lambda r: -r["R20"]):
    if r["name"] in seen:
        continue
    seen.add(r["name"])
    pol = None
    for grp in sweep.SWEEPS.values():
        for name, mk, k0m, note in grp:
            if name == r["name"]:
                try:
                    pol = mk()
                except Exception:
                    pol = None
    a = {20: {"recall": r["R20"], "mrr": r["MRR20"], "mar": r["MAR20"]}, 5: {"recall": r["R5"]}, 10: {"recall": r["R10"]}}
    lines.append(row(f"`{r['name']}`: {r['note']}", a, "offline sim", pol, f"k {r['k_dist']}"))
print("\n".join(lines))
