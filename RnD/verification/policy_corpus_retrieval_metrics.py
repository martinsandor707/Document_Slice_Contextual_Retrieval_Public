"""Retrieval metrics of the policy (tobacco) corpus, Table 8 of sn_green_rag_article.tex: Recall / MRR / MAR at
K = 5, 10, 15, 20 on the 80 questions of RnD/q_and_a/Gemini/smoking.json for
  * tobacco_traditional_table      full-document baseline (embeds summary + chunk),
  * tobacco_sliding_text_table     k=3 document slice, REBUILT with summary + chunk as the embedded text
                                   (RnD/build_tobacco_sliding_text_table.py),
  * tobacco_sliding_table          the original k=3 table, which embedded the raw chunk only (kept for comparison),
with the notebook-identical hybrid search + ColBERT re-ranking of evalkit.evaluate(), plus per-question comparisons
against the baseline (better/worse counts, paired 99 % BCa bootstrap of mean Recall@20, exact McNemar on
"all gold chunks in the top 20"). Per-question results are saved as
RnD/dynamic_slice_length/router_experiments/results/res_<table>.json (new files only).
GPU: loads the nomic embedder and ColBERT on CUDA; do not run while an Ollama model is resident.
Run from the repository root:  .venv/bin/python RnD/verification/policy_corpus_retrieval_metrics.py"""
import json, sys, os
import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from _common import RND, RESULTS, QNA_POLICY
sys.path.insert(0, str(RND / "dynamic_slice_length" / "router_experiments" / "scripts"))
import evalkit

questions = json.load(open(QNA_POLICY, encoding="utf-8"))
TABLES = ["tobacco_traditional_table", "tobacco_sliding_text_table", "tobacco_sliding_table"]
res = {}
for name in TABLES:
    t = evalkit.open_rnd_table(name)
    meta = json.loads((t.schema.metadata or {})[b"embedding_functions"])[0]
    r = evalkit.evaluate(t, questions)
    res[name] = r
    out = RESULTS / f"res_{name}.json"
    if not out.exists():
        json.dump(r, open(out, "w", encoding="utf-8"))
    nohit = [sum(1 for q in r["per_q"] if np.isnan(q[K]["reciprocal_rank"])) for K in (5, 10, 15, 20)]
    print(evalkit.fmt(r, f"{name} (embeds {meta['source_column']})"), f" | no-hit @5/10/15/20 = {nohit}")

def rec(name, K=20): return np.array([q[K]["recall"] for q in res[name]["per_q"]])
b20 = rec("tobacco_traditional_table")
print("\n== per-question comparison against the full-document baseline (Recall@20, 80 questions)")
for name in TABLES[1:]:
    s20 = rec(name)
    bs = bootstrap((b20, s20), statistic=lambda b, s: np.mean(b) - np.mean(s), confidence_level=0.99, paired=True, method="BCa", random_state=42)
    bf, sf = b20 == 1.0, s20 == 1.0
    tab = [[int((bf & sf).sum()), int((~bf & sf).sum())], [int((bf & ~sf).sum()), int((~bf & ~sf).sum())]]
    print(f"{name}: diff (baseline-own) {b20.mean()-s20.mean():+.5f}; 99% BCa CI [{bs.confidence_interval.low:+.4f}, {bs.confidence_interval.high:+.4f}]; "
          f"McNemar [[both, own only], [baseline only, neither]] = {tab}, exact p = {mcnemar(tab, exact=True).pvalue:.4f}; "
          f"own better/worse = {int((s20 > b20).sum())}/{int((s20 < b20).sum())}")
s_new, s_old = rec("tobacco_sliding_text_table"), rec("tobacco_sliding_table")
print(f"\nrebuilt vs original slice table: better/worse per question = {int((s_new > s_old).sum())}/{int((s_new < s_old).sum())}; "
      f"top-20 lists identical for {sum(a == b for a, b in zip(res['tobacco_sliding_text_table']['top20'], res['tobacco_sliding_table']['top20']))}/80 questions")
