import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Shared look for the stroke_* figures (section C of the Act-0 guide). Import AFTER the hw_guard line."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/mashfiq/eeg_vjepa"
RAW = f"{ROOT}/data/zenodo_stroke"
FIGS = f"{ROOT}/docs/act0_guide/figs"
RUNS = f"{ROOT}/code/runs/act0"
STROKE, CONTROL, THIRD, VIOLET = "#eb6834", "#2a78d6", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
GATE = lambda: cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    """dpi 170; callers keep figsize width <= 9.5 in (= 1615 px <= 2400 px)."""
    w = fig.get_size_inches()[0] * 170
    assert w <= 2400, w
    fig.savefig(f"{FIGS}/{name}", dpi=170, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)
