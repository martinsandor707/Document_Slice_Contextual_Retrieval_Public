"""Figure 6 of sn_green_rag_article.tex (label fig:energy): tracked generation energy and emissions of the three pipelines
on the 25-paper corpus. Data: the codecarbon rows 2026-03-05T23:59:43 (full-document baseline), 2026-03-05T22:53:27
(static k=3) and 2026-09-25T20:06:47 (routed, primary run) of RnD/emissions_data/emissions.csv, read with
RnD/verification/_common.emissions_rows (gpu_energy, cpu_energy, ram_energy, energy_consumed in kWh; emissions in kg).
Panel (a) stacks the GPU, CPU and RAM components of the tracked energy in Wh; panel (b) shows emissions in g CO2eq.
Reductions are 1 - value/baseline of the plotted values. CPU energy of the March rows came from RAPL counters, of the
September row from a load estimate (Table 5), so the GPU component is the one comparable without qualification. PNG only.
Run from the repository root:  .venv/bin/python RnD/nature_article/images/fig06_energy_emissions.py"""
import os, sys
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "verification"))
import _style
from _common import emissions_rows
_style.apply()

rows = emissions_rows()
TS = {"full": "2026-03-05T23:59:43", "k3": "2026-03-05T22:53:27", "routed": "2026-09-25T20:06:47"}
NAMES = {"full": "full\ndocument", "k3": "static\n$k=3$", "routed": "routed"}
ORDER = ["full", "k3", "routed"]
wh = {k: {c: rows[ts][f"{c}_energy"] * 1e3 for c in ("gpu", "cpu", "ram")} for k, ts in TS.items()}
total = {k: rows[ts]["energy_consumed"] * 1e3 for k, ts in TS.items()}
g = {k: rows[ts]["emissions"] * 1e3 for k, ts in TS.items()}
for k in ORDER: assert abs(sum(wh[k].values()) - total[k]) < 0.02, k          # components add up to the tracked total
COMP = [("gpu", "GPU (NVML)", _style.SEQ_BLUE[11]), ("cpu", "CPU (RAPL / load estimate)", _style.SEQ_BLUE[6]), ("ram", "RAM (estimate)", _style.SEQ_BLUE[2])]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.38, width_ratios=[1.2, 1]))
ax = axes[0]; ax.grid(axis="x", visible=False)
for i, k in enumerate(ORDER):
    bottom = 0
    for c, label, col in COMP:
        v = wh[k][c]
        ax.bar(i, v, bottom=bottom, width=0.58, color=col, linewidth=0, label=label if i == 0 else None)
        if c == "gpu": ax.text(i, bottom + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=6.8, color="white")
        bottom += v + 0.6                                                     # small surface gap between segments
    ax.text(i, bottom + 1.5, f"{total[k]:.1f} Wh" + ("" if k == "full" else f"\n−{100 * (1 - total[k] / total['full']):.1f} %"),
            ha="center", va="bottom", fontsize=7, color=_style.INK, linespacing=1.25)
ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[k] for k in ORDER]); ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0, 140); ax.set_ylabel("tracked generation energy (Wh)")
ax.set_title("(a) energy by component", loc="left", fontsize=8, color=_style.INK2, pad=6)
ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), fontsize=6.3, handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)

ax = axes[1]; ax.grid(axis="x", visible=False)
for i, k in enumerate(ORDER):
    ax.bar(i, g[k], width=0.58, color=_style.SEQ_BLUE[8], linewidth=0)
    ax.text(i, g[k] + 0.4, f"{g[k]:.2f} g" + ("" if k == "full" else f"\n−{100 * (1 - g[k] / g['full']):.1f} %"),
            ha="center", va="bottom", fontsize=7, color=_style.INK, linespacing=1.25)
ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[k] for k in ORDER]); ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0, 27); ax.set_ylabel("emissions (g CO$_2$eq)")
ax.set_title("(b) emissions", loc="left", fontsize=8, color=_style.INK2, pad=6)
_style.save(fig, os.path.join(HERE, "fig06_energy_emissions"))
for k in ORDER:
    print(f"{NAMES[k]:16s} total {total[k]:7.2f} Wh (gpu {wh[k]['gpu']:.2f}, cpu {wh[k]['cpu']:.2f}, ram {wh[k]['ram']:.2f}); {g[k]:.2f} g; "
          f"vs baseline: total −{100*(1-total[k]/total['full']):.1f} %, gpu −{100*(1-wh[k]['gpu']/wh['full']['gpu']):.1f} %, emissions −{100*(1-g[k]/g['full']):.1f} %")
