"""Verifies the retrieval-metric bookkeeping and the PROVISIONAL significance tests of the Springer manuscript:
  Table 7 (tab:primary) |Q_20^+| counts (questions with at least one gold chunk in the top 20) and the
  provisional footnote / Sect. 4.2 paragraph: paired 99 % BCa bootstrap interval of mean Recall@20 (baseline
  minus own; scipy, 9 999 resamples, random_state 42), exact McNemar on 'all gold chunks retrieved at 20'
  (discordant counts), per-question better/worse counts;  Sect. 3.6 / 4.4 no-hit counts at K = 5/10/15/20.
Inputs are the stored per-question results under RnD/dynamic_slice_length/router_experiments/results/
(res_real_binary_t2_heldout.json = replicate run; the primary notebook run has no stored per-question file).
Run from the repository root:  .venv/bin/python RnD/verification/retrieval_significance_tests.py"""
import json, numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from _common import RESULTS

def load(name): return json.load(open(RESULTS / f"res_{name}.json", encoding="utf-8"))
def rec(r, K="20"): return np.array([q[K]["recall"] for q in r["per_q"]])
def nohit(r, K): return sum(1 for q in r["per_q"] if q[K]["reciprocal_rank"] is None or np.isnan(q[K]["reciprocal_rank"]))
base = load("anthropic_control_table"); b20 = rec(base)
for name, own in [("routed replicate (real_binary_t2_heldout)", load("real_binary_t2_heldout")),
                  ("static k=3 (ablation_doc_slice_radius_3)", load("ablation_doc_slice_radius_3")),
                  ("static k=2 (ablation_doc_slice_radius_2)", load("ablation_doc_slice_radius_2"))]:
    s20 = rec(own)
    bs = bootstrap((b20, s20), statistic=lambda b, s: np.mean(b) - np.mean(s), confidence_level=0.99, paired=True, method="BCa", random_state=42)
    bf, sf = b20 == 1.0, s20 == 1.0
    tab = [[int((bf & sf).sum()), int((~bf & sf).sum())], [int((bf & ~sf).sum()), int((~bf & ~sf).sum())]]
    print(f"== {name}: R@20 own {s20.mean():.5f} vs baseline {b20.mean():.5f}; diff (baseline-own) {b20.mean()-s20.mean():+.5f}; "
          f"99% BCa CI [{bs.confidence_interval.low:+.4f}, {bs.confidence_interval.high:+.4f}] (n_resamples {bs.bootstrap_distribution.size}); "
          f"McNemar [[both, own only], [baseline only, neither]] = {tab}, exact p = {mcnemar(tab, exact=True).pvalue:.4f}; "
          f"per question own better/worse = {int((s20 > b20).sum())}/{int((s20 < b20).sum())}; "
          f"no-hit counts @5/10/15/20 = {[nohit(own, K) for K in ('5', '10', '15', '20')]}  -> |Q_20^+| = {250 - nohit(own, '20')}")
print(f"baseline no-hit counts @5/10/15/20 = {[nohit(base, K) for K in ('5','10','15','20')]} -> |Q_20^+| = {250 - nohit(base, '20')}; baseline all-gold-found@20 = {int((b20 == 1).sum())}")
