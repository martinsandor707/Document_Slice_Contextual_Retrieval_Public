"""Figure 10 of sn_green_rag_article.tex (label fig:offload): what happens to the gemma3:4b-it-qat runner at the
30 000-token window when other processes already hold GPU memory. Data: every RnD/verification/ollama_offload_pressure_*.json
(script ollama_offload_under_vram_pressure.py), "30000" entry: x = GPU memory in use before the model loads (GiB, includes
the 15 MiB of the desktop), left panel = layers kept on the GPU (of 35), right panel = warm prompt-evaluation time divided
by the same prompt's time on an empty GPU for the same release (k=3 and full-document prompts normalised separately).
Shaded bands: measured footprints of a live Docling converter (1.8 GiB) and of a kernel after Docling + LanceDB embedding
(5.9-6.1 GiB), from coresident_gpu_footprint.py. Run from the repository root:
.venv/bin/python RnD/nature_article/images/fig10_shared_gpu_offload.py"""
import glob, json, os, sys
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _style
_style.apply()
VER = os.path.join(HERE, "..", "..", "verification")
pts = {}
for f in sorted(glob.glob(os.path.join(VER, "ollama_offload_pressure_*.json"))):
    d = json.load(open(f)); r = d["30000"]
    ver = "0.17.5" if "0.17.5" in d["version_label"] else "0.35.1"
    pts.setdefault(ver, []).append(dict(x=r["gpu_before_load_mib"] / 1024, layers=int(r["offloaded_layers"].split("/")[0]),
                                        t=r["warm_prompt_eval_s"], prompt=d.get("prompt_mode", "k3")))
for ver, rows in pts.items():
    for p in rows:
        base = min((q for q in rows if q["prompt"] == p["prompt"]), key=lambda q: q["x"])
        p["slow"] = p["t"] / base["t"]
    rows.sort(key=lambda q: q["x"])
COL = {"0.17.5": _style.SERIES["k1"], "0.35.1": _style.SERIES["full"]}
LAB = {"0.17.5": "Ollama 0.17.5 (release of the March 2026 runs)", "0.35.1": "Ollama 0.35.1 (current)"}
MARK = {"k3": "o", "full": "s"}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(wspace=0.32))
for ax, key, ylab in ((axes[0], "layers", "layers kept on the GPU (of 35)"), (axes[1], "slow", "prompt-evaluation slowdown vs empty GPU")):
    for (x0, x1, name) in ((1.65, 1.85, "Docling\nconverter"), (5.85, 6.15, "Docling +\nembedder kernel")):
        ax.axvspan(x0, x1, color=_style.GRID, alpha=0.8, lw=0)
        ax.text((x0 + x1) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.3, color=_style.INK2)
    for ver, rows in pts.items():
        ax.plot([p["x"] for p in rows], [p[key] for p in rows], color=COL[ver], lw=1.3, zorder=2)
        for prompt in ("k3", "full"):
            sel = [p for p in rows if p["prompt"] == prompt]
            ax.scatter([p["x"] for p in sel], [p[key] for p in sel], s=18, marker=MARK[prompt], color=COL[ver], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(-0.2, 7.0); ax.set_xlabel("GPU memory held by other processes (GiB)"); ax.set_ylabel(ylab)
    ax.set_xticks(range(0, 8))
axes[0].set_ylim(-1.5, 37); axes[0].set_yticks([0, 12, 24, 35])
axes[1].set_ylim(0.9, 2.65); axes[1].set_yticks([1.0, 1.5, 2.0, 2.5]); axes[1].set_yticklabels(["1.0×", "1.5×", "2.0×", "2.5×"])
from matplotlib.lines import Line2D
h = [Line2D([], [], color=COL[v], lw=1.3, label=LAB[v]) for v in ("0.17.5", "0.35.1")]
h += [Line2D([], [], color=_style.INK2, marker="o", ls="", ms=4, label="longest k=3 prompt (11.8k tokens)"),
      Line2D([], [], color=_style.INK2, marker="s", ls="", ms=4, label="full-document prompt, longest paper (19.9k tokens)")]
fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17), fontsize=6.6, handlelength=1.6, columnspacing=1.5)
_style.save(fig, os.path.join(HERE, "fig10_shared_gpu_offload"))
for ver, rows in pts.items(): print(ver, [(round(p["x"], 2), p["layers"], round(p["slow"], 2), p["prompt"]) for p in rows])
