"""Figure 8 of sn_green_rag_article.tex (label fig:recall): Recall@K for K in {5, 10, 15, 20} of the full-document
baseline, the routed pipeline, the static k=3 slice and the no-context hybrid reference on the 250 scientific questions.
Baseline, k=3 and no-context values are the `avg` blocks of the stored per-question result files
RnD/dynamic_slice_length/router_experiments/results/res_anthropic_control_table.json, res_ablation_doc_slice_radius_3.json
and res_hybrid_retrieval.json (the same numbers the "METRICS AT N=" printouts of RnD/ablation_doc_slice_retrieval.ipynb show).
The routed pipeline's primary run has no stored per-question file; its values are the printout of
RnD/benchmarking_retrievals.ipynb (OWN = ablation_doc_slice_radius_dynamic, Martin's 2026-09-25 rerun) and are typed in
below; the replicate run (res_real_binary_t2_heldout.json: 0.75167 / 0.86567 / 0.92667 / 0.952) is asserted to agree with
it at K=5 and K=20. The shaded band at K=20 is the +/-0.7-point single-run resolution of Sect. 3.6. PNG only.
Run from the repository root:  .venv/bin/python RnD/nature_article/images/fig08_recall_trajectories.py"""
import json, os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "verification"))
import _style
from _common import RESULTS
_style.apply()

KS = [5, 10, 15, 20]
def stored(name):
    a = json.load(open(RESULTS / f"res_{name}.json"))["avg"]
    return [a[str(k)]["recall"] for k in KS]
series = {
    "full": ("full-document baseline", stored("anthropic_control_table")),
    "routed": ("routed pipeline", [0.75167, 0.86700, 0.92600, 0.95200]),      # benchmarking_retrievals.ipynb printout
    "k3": ("static $k=3$ slice", stored("ablation_doc_slice_radius_3")),
    "none": ("no context (hybrid only)", stored("hybrid_retrieval")),
}
rep = stored("real_binary_t2_heldout")
assert abs(rep[0] - series["routed"][1][0]) < 1e-5 and abs(rep[3] - series["routed"][1][3]) < 1e-5, "replicate disagrees with the primary run"

fig, ax = plt.subplots(figsize=(5.4, 2.9))
ax.grid(axis="x", visible=False)
b20 = series["full"][1][3]
ax.add_patch(Rectangle((19.25, b20 - 0.007), 1.5, 0.014, facecolor=_style.GRID, edgecolor="none", zorder=1))
ax.text(20.0, b20 + 0.0115, "±0.7 pt single-run band", fontsize=6.3, color=_style.INK2, va="bottom", ha="center")
for key, (label, vals) in series.items():
    col = _style.MUTED if key == "none" else _style.SERIES[key]
    ax.plot(KS, vals, color=col, lw=1.4 if key != "none" else 1.1, ls="--" if key == "none" else "-", marker="o", ms=3.6,
            markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.7, label=label, zorder=3)
# selective end labels at K=20: only the two series the caption compares (baseline, routed); the rest stay in the legend
ends = sorted(((series[k][1][3], k) for k in ("full", "routed")), reverse=True)
pos = {}; GAP = 0.0135
for v, key in ends:
    pos[key] = v if not pos else min(v, min(pos.values()) - GAP)
for key in ("full", "routed"):
    ax.annotate(f"{series[key][1][3]:.3f}", xy=(20, series[key][1][3]), xytext=(21.6, pos[key]), fontsize=6.8, color=_style.INK2, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=_style.SERIES[key], lw=0.6, shrinkA=0, shrinkB=2))
ax.set_xticks(KS); ax.set_xticklabels([f"$K={k}$" for k in KS]); ax.set_xlim(4, 23.6)
ax.set_ylim(0.70, 0.985); ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95]); ax.set_ylabel("Recall@K")
ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.03), handlelength=1.8, fontsize=6.8)
_style.save(fig, os.path.join(HERE, "fig08_recall_trajectories"))
for key, (label, vals) in series.items(): print(f"{label:28s}", [round(v, 5) for v in vals])
