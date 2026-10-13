"""Figure 6 of sn_green_rag_article.tex (label fig:heatmap): cost surface of the static radius k in {0,...,4}.
Rows: generation time in minutes (notebook wall-clock comments of the fixed-radius ablation, Table 6) and prompt volume
sent to the summariser in millions of characters (RnD/verification/prompt_volume_and_attention_proxy.py). The resident
GPU memory of the runner is not a row any more: it is the same 4.54 GiB for every k (RnD/verification/ollama_runner_vram.json),
and the caption / Table 2 state it. Each row uses its own sequential blue ramp.
Run from the repository root:  .venv/bin/python RnD/nature_article/images/fig06_radius_heatmap.py"""
import sys, os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "verification"))
import _style
import prompt_volume_and_attention_proxy as pv

_style.apply()
_, _, res = pv.compute()
ks = [0, 1, 2, 3, 4]
gen_min = [7 + 28 / 60, 12 + 8 / 60, 17 + 28 / 60, 22 + 41 / 60, 28 + 11 / 60]          # 7m28s ... 28m11s
vol_m = [res["fixed k=0 (self-slice)"][1] / 1e6] + [res[f"fixed k={k}"][1] / 1e6 for k in ks[1:]]
rows = [("generation time (min)", gen_min, "{:.1f}"), ("prompt volume (M chars)", vol_m, "{:.2f}")]

fig = plt.figure(figsize=(7.0, 1.75))
ax = fig.add_axes([0.26, 0.06, 0.72, 0.80])
ax.grid(False)
for r, (name, vals, fmt) in enumerate(rows):
    for c, k in enumerate(ks):
        t = 0.15 + 0.85 * (vals[c] - min(vals)) / (max(vals) - min(vals))
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=_style.SEQ_CMAP(t), edgecolor="white", linewidth=2))
        ax.text(c + 0.5, r + 0.5, fmt.format(vals[c]), ha="center", va="center", fontsize=8, color="white" if t > 0.55 else _style.INK)
ax.set_xlim(0, 5); ax.set_ylim(len(rows), 0)
ax.set_xticks([c + 0.5 for c in range(5)]); ax.set_xticklabels([f"$k={k}$" for k in ks])
ax.xaxis.tick_top(); ax.tick_params(axis="both", length=0)
ax.set_yticks([r + 0.5 for r in range(len(rows))]); ax.set_yticklabels([r[0] for r in rows])
for s in ax.spines.values(): s.set_visible(False)
_style.save(fig, os.path.join(HERE, "fig06_radius_heatmap"))
