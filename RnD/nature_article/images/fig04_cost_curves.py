"""Figure 4 of sn_green_rag_article.tex (label fig:complexity): theoretical index-time cost curves.
Panel (a): prompt tokens per document as a function of the number of chunks n, for the full-document design
(n+1)L with L = n*lbar, fixed k = 1, 2, 3 (n(2k+2) lbar, Eq. (volume)) and the routed pipeline (measured skip
rate of 32 %, k = 2 otherwise), with lbar = 664 nomic tokens. Panel (b): the quadratic attention proxy
sum_i P_i^2 for the same designs (log scale). Both panels overlay the MEASURED per-document values of the
25 corpus papers (actual chunk lengths, cost.py formula via RnD/verification/prompt_volume_and_attention_proxy.py,
characters / 4.17 per token) as markers; the corpus-level totals appear in the legend and the 4.6x / 10.8x ratios
are annotated in panel (b).
Run from the repository root:  .venv/bin/python RnD/nature_article/images/fig04_cost_curves.py"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "verification"))
import _style
import prompt_volume_and_attention_proxy as pv

_style.apply()
docs, pol, res = pv.compute()
cpt = pv.CHARS_PER_TOKEN
LBAR = 253488 / 382                      # mean chunk length in nomic tokens (663.6)
SKIP = 121 / 382                         # measured share of chunks routed to k = 0 (primary run)
n = np.arange(8, 34)                     # document lengths of the corpus: 8..33 chunks

theory = {                               # tokens per document, and sum of squared prompt lengths per document
    "full":   ((n + 1) * n * LBAR,        n * ((n + 1) * LBAR) ** 2),
    "k1":     (n * 4 * LBAR,              n * (4 * LBAR) ** 2),
    "k2":     (n * 6 * LBAR,              n * (6 * LBAR) ** 2),
    "k3":     (n * 8 * LBAR,              n * (8 * LBAR) ** 2),
    "routed": ((1 - SKIP) * n * 6 * LBAR, (1 - SKIP) * n * (6 * LBAR) ** 2),
}
short = {"full": "full document", "k1": "fixed $k=1$", "k2": "fixed $k=2$", "k3": "fixed $k=3$", "routed": "routed, $k\\in\\{0,2\\}$"}
measured_key = {"full": "full document", "k1": "fixed k=1", "k2": "fixed k=2", "k3": "fixed k=3", "routed": "routed (primary run)"}
totals = {s: res[measured_key[s]][1] / cpt / 1e6 for s in theory}          # corpus totals in M tokens
legend_label = {s: f"{short[s]} ({totals[s]:.2f} M)" for s in theory}        # corpus total in M tokens
per_doc = {s: pv.per_document(docs, pol[measured_key[s]]) for s in theory}
n_doc = {name: len(ch) for name, ch in docs.items()}
order = ["full", "k1", "k2", "k3", "routed"]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
for s in order:
    col = _style.SERIES[s]
    tok, sq = theory[s]
    ax_a.plot(n, tok / 1e3, color=col, zorder=3)
    ax_b.plot(n, sq, color=col, zorder=3)
    xs = [n_doc[d] for d in per_doc[s]]
    ax_a.scatter(xs, [v[1] / cpt / 1e3 for v in per_doc[s].values()], s=9, color=col, edgecolor="white", linewidth=0.5, zorder=4)
    ax_b.scatter(xs, [v[2] / cpt ** 2 for v in per_doc[s].values()], s=9, color=col, edgecolor="white", linewidth=0.5, zorder=4)

ax_a.set_xlabel("chunks per document $n$"); ax_a.set_ylabel("prompt tokens per document (thousands)")
ax_a.set_title("(a) index-time prompt volume", loc="left", color=_style.INK)
ax_a.set_xlim(8, 41); ax_a.set_ylim(0, 820)
ax_b.set_xlabel("chunks per document $n$"); ax_b.set_ylabel(r"$\sum_i P_i^2$ (tokens$^2$)")
ax_b.set_yscale("log"); ax_b.set_title("(b) quadratic attention proxy", loc="left", color=_style.INK)
ax_b.set_xlim(8, 41); ax_b.set_ylim(2.5e7, 3e10)
ax_a.text(0.03, 0.97, "lines: theory with $\\bar\\ell=664$ tokens; markers: the 25 corpus papers\nlegend: corpus totals in M tokens", transform=ax_a.transAxes, va="top", fontsize=6.5, color=_style.INK2)
full, k3, routed = res["full document"], res["fixed k=3"], res["routed (primary run)"]
ax_b.text(0.97, 0.04, f"corpus totals of $\\sum_i P_i^2$:\nfull / fixed $k{{=}}3$ = {full[2]/k3[2]:.1f}$\\times$\nfull / routed = {full[2]/routed[2]:.1f}$\\times$",
          transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.5, color=_style.INK2)

def end_labels(ax, ends, min_gap_pt=8.5):
    """Direct labels at the right end of each curve, pushed apart vertically so that none overlap."""
    fig.canvas.draw()
    to_disp = ax.transData.transform; to_data = ax.transData.inverted().transform
    items = sorted(((to_disp((n[-1], y))[1], s) for s, y in ends.items()))
    gap = min_gap_pt * fig.dpi / 72
    ys = [items[0][0]]
    for y, _ in items[1:]:
        ys.append(max(y, ys[-1] + gap))
    shift = (ys[-1] - items[-1][0]) / 2                        # centre the stack on its original span
    for (y0, s), y in zip(items, ys):
        yd = to_data((0, y - shift))[1]
        ax.annotate(short[s], (n[-1], yd), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.8, color=_style.INK2,
                    arrowprops=None)
end_labels(ax_a, {s: theory[s][0][-1] / 1e3 for s in order})
end_labels(ax_b, {s: theory[s][1][-1] for s in order})
handles = [plt.Line2D([], [], color=_style.SERIES[s], marker="o", markersize=3, markeredgecolor="white", label=legend_label[s]) for s in order]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), handlelength=1.8, columnspacing=1.2)
_style.save(fig, os.path.join(HERE, "fig04_cost_curves"))
