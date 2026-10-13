"""Figure 7 of sn_green_rag_article.tex (label fig:runtime): preprocessing runtime of the three pipelines in the two
timing scopes of Sect. 3.8. Panel (a), tracked generation scope: `duration` of the codecarbon rows 2026-03-05T23:59:43
(full-document baseline), 2026-03-05T22:53:27 (static k=3) and 2026-09-25T20:06:47 (routed, primary run) in
RnD/emissions_data/emissions.csv, read with RnD/verification/_common.emissions_rows. Panel (b), end-to-end scope including
PDF parsing: notebook wall-clock comments "Total runtime: 71m 34s for 25 documents" (RnD/anthropic_traditional_chunking.ipynb)
and "Total runtime for scientific documents: 29m 9s" (RnD/doc_slice_chunking.ipynb); for the routed pipeline, wall_clock_s of
RnD/benchmarking_results/preprocessing/end_to_end_routed_timings.json, written by RnD/end_to_end_dynamic_doc_slice_generation.ipynb
(Docling conversion + chunking + routing + summarisation of the 25 PDFs in one run, 2026-10-05). Speed-ups are ratios of the
plotted values. PNG only.
Run from the repository root:  .venv/bin/python RnD/nature_article/images/fig07_runtime_scopes.py"""
import json, os, sys
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "verification"))
import _style
from _common import emissions_rows
_style.apply()

rows = emissions_rows()
TS = {"full": "2026-03-05T23:59:43", "k3": "2026-03-05T22:53:27", "routed": "2026-09-25T20:06:47"}
tracked = {k: rows[ts]["duration"] / 60 for k, ts in TS.items()}                      # minutes
end2end = {"full": (71 * 60 + 34) / 60, "k3": (29 * 60 + 9) / 60, "routed": None}      # notebook comments
TIMINGS = os.path.join(HERE, "..", "..", "benchmarking_results", "preprocessing", "end_to_end_routed_timings.json")
if os.path.exists(TIMINGS):
    e2e = json.load(open(TIMINGS)); end2end["routed"] = e2e["wall_clock_s"] / 60
NAMES = {"full": "full document", "k3": "static $k=3$", "routed": "routed"}
ORDER = ["full", "k3", "routed"]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, data, title in ((axes[0], tracked, "(a) tracked generation scope"), (axes[1], end2end, "(b) end-to-end scope, incl. PDF parsing")):
    ax.grid(axis="x", visible=False)
    base = data["full"]
    for i, key in enumerate(ORDER):
        v = data[key]
        if v is None:
            ax.text(i, 2.2, "not\nmeasured", ha="center", va="bottom", fontsize=7, color=_style.MUTED, linespacing=1.3)
            continue
        ax.bar(i, v, width=0.58, color=_style.SERIES[key], linewidth=0)
        ax.text(i, v + 1.2, f"{v:.1f} min", ha="center", va="bottom", fontsize=7.2, color=_style.INK)
        if key != "full":
            ax.text(i, v + 9.5, f"{base / v:.1f}× faster", ha="center", va="bottom", fontsize=7.2, color=_style.INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[k] for k in ORDER]); ax.set_xlim(-0.6, 2.6)
    ax.set_title(title, loc="left", fontsize=8, color=_style.INK2, pad=6)
axes[0].set_ylabel("preprocessing time (min)"); axes[0].set_ylim(0, 88); axes[0].set_yticks([0, 20, 40, 60, 80])
_style.save(fig, os.path.join(HERE, "fig07_runtime_scopes"))
print({k: round(v, 2) for k, v in tracked.items()}, {k: (round(v, 2) if v else None) for k, v in end2end.items()},
      "speed-ups (a):", round(tracked["full"] / tracked["k3"], 2), round(tracked["full"] / tracked["routed"], 2),
      "(b):", round(end2end["full"] / end2end["k3"], 2), round(end2end["full"] / end2end["routed"], 2) if end2end["routed"] else None)
