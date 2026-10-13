"""Verifies the routing statistics of the Springer manuscript (Sect. 3.3, Sect. 4.3, Table 7, abstract):
  radius/tier distribution of the primary run ({0: 121, 2: 261}; 42 table-gated, 219 SLM->N, 121 SLM->S),
  mean radius k-bar = 1.37, summariser calls 261 (31.7 % skipped), summary length (mean 201 chars), the replicate
  run ({0: 120, 2: 262}); the 2-class router's agreement with the 119-item gold set (82/99 on the SLM tier, gate
  20/20 with no false positive); and the NEGATIVE check that short chunks (<= 250 tokens) do NOT explain the
  skipped chunks (only 51 of 121 skipped chunks are short), which is why no such sentence appears in the text.
Run from the repository root:  .venv/bin/python RnD/verification/router_routing_statistics.py   (tokenizer part is CPU only)"""
import json, statistics, collections
from _common import DYNAMIC_JSON, REAL_RUNS, GOLD, RESULTS, QNA_SCI, nomic_tokenizer

def summarise(name, recs):
    kd = collections.Counter(r["doc_slice_radius"] for r in recs)
    tiers = collections.Counter((r["doc_slice_radius"], r["slice_tier"]) for r in recs)
    ctx = [len(r["context"]) for r in recs if r["context"]]
    n = len(recs)
    print(f"== {name}: n={n} k distribution {dict(sorted(kd.items()))} tiers {dict(sorted(tiers.items()))} "
          f"k-bar={sum(k*v for k, v in kd.items())/n:.4f} calls={sum(1 for r in recs if r['doc_slice_radius'] > 0)} "
          f"skipped={sum(1 for r in recs if r['doc_slice_radius'] == 0)/n:.1%} summary chars mean/median/min/max={statistics.mean(ctx):.1f}/{statistics.median(ctx)}/{min(ctx)}/{max(ctx)}")

primary = json.load(open(DYNAMIC_JSON, encoding="utf-8"))
replicate = json.load(open(REAL_RUNS / "binary_t2_heldout_chunks.json", encoding="utf-8"))
summarise("primary run (preprocessed_chunks/ablation_doc_slice_radius_dynamic.json)", primary)
summarise("replicate run (real_runs/binary_t2_heldout_chunks.json)", replicate)

gold = json.load(open(GOLD, encoding="utf-8"))["items"]
routes = json.load(open(RESULTS / "routes_binary_heldout.json", encoding="utf-8"))
slm = [g for g in gold if g["tier"] == "slm"]; tables = [g for g in gold if g["tier"] != "slm"]
conf = collections.Counter(("S" if g["expected_k"] == 0 else "N", routes[g["id"]]["letter"]) for g in slm)
agree = conf[("S", "S")] + conf[("N", "N")]
print(f"== 2-class router vs gold set: {len(gold)} items over {len({g['document'] for g in gold})} papers; SLM tier {len(slm)}: agreement {agree}/{len(slm)} = {agree/len(slm):.1%}, confusion {dict(conf)}; "
      f"table tier {len(tables)}: gate fired on {sum(routes[g['id']]['table'] for g in tables)}, false positives among SLM-tier items {sum(routes[g['id']]['table'] for g in slm)}")

tok = nomic_tokenizer()
toks = [len(tok.tokenize(r["original_text"])) for r in primary]
k0 = [r["doc_slice_radius"] == 0 for r in primary]; short = [t <= 250 for t in toks]
both = sum(a and b for a, b in zip(k0, short))
gold_ids = {c for q in json.load(open(QNA_SCI, encoding="utf-8")) for c in q["supporting_chunks"]}
print(f"== short-chunk overlap (negative result): skipped {sum(k0)}, short (<=250 tokens) {sum(short)}, both {both} "
      f"= {both/sum(short):.1%} of short chunks skipped, {both/sum(k0):.1%} of skipped chunks short; skipped chunks tokens median {statistics.median([t for t, z in zip(toks, k0) if z]):.0f} vs summarised {statistics.median([t for t, z in zip(toks, k0) if not z]):.0f}; "
      f"skipped chunks that are gold chunks: {sum(1 for r, z in zip(primary, k0) if z and r['id'] in gold_ids)}")
