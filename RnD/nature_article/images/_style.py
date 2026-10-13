"""Shared matplotlib styling for the manuscript figures (static, print-oriented; palette: dataviz reference instance)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical slots in fixed order (blue, orange, aqua, yellow, magenta); the fixed radii keep their slot in every figure
SERIES = {"full": "#2a78d6", "k1": "#eb6834", "k2": "#1baf7a", "k3": "#eda100", "routed": "#e87ba4"}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white", "text.color": INK,
    })


def save(fig, stem):
    """Writes the PNG only (300 dpi, tight bounding box); the manuscript includes PNGs, no PDF is needed."""
    fig.savefig(f"{stem}.png")
    print("saved", f"{stem}.png")
