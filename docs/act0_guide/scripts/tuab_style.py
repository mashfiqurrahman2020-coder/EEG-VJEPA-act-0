"""Shared look for the TUAB figures (import AFTER the hw_guard lines).
Readability budget: figures are at most W_IN inches wide and no text is below 12 pt, so at A4 text
width (~6.7 in) the smallest printed text is >= 12 * 6.7 / 8.5 = 9.5 pt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/mashfiq/eeg_vjepa"
FIGS = f"{ROOT}/docs/act0_guide/figs"
NORMAL, ABNORMAL, THIRD, FOURTH = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"   # dataviz reference slots 1,2,3,7
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W_IN, SMALL, DPI = 8.5, 12, 170

plt.rcParams.update({
    "font.size": 12.5, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


def save(fig, name):
    p = f"{FIGS}/{name}"
    fig.savefig(p)
    from PIL import Image
    w, h = Image.open(p).size
    assert w <= min(2400, (W_IN + 0.35) * DPI), (name, w)   # guide rule <= 2400 px + our font budget
    print(f"saved {p} ({w}x{h} px)")
