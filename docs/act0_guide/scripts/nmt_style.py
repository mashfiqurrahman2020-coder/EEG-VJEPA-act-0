import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Shared paths + matplotlib style for the NMT (Section A) figures. Import this FIRST in every nmt_*.py."""
import os
os.environ["HW_THREADS"] = "1"          # code/*.py call cap_threads(4); HW_THREADS makes that 1 too
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/mashfiq/eeg_vjepa"
RAW = f"{ROOT}/data/NMT-Scalp-EEG"
PREP = f"{ROOT}/data/NMT_preprocessed"
GUIDE = f"{ROOT}/docs/act0_guide"
FIGS = f"{GUIDE}/figs"
FACTS = f"{GUIDE}/facts"
CH = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
      'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']        # code/preprocess_nmt.py CHANNELS
FS = 200

# dataviz reference palette (light), fixed categorical order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
NORMAL, ABNORMAL = BLUE, ORANGE
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
FOLD_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
DPI = 170   # figures are ~8 in wide -> 1360 px; fonts >= 11 pt stay >= 9 pt at A4 text width

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def cool():
    cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=True)


def save(fig, name):
    path = f"{FIGS}/{name}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(path).size
    assert w <= 2400, f"{name} too wide: {w}px"
    print(f"saved {path}  ({w}x{h}px)", flush=True)
