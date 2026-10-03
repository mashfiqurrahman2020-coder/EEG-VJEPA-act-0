import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Shared helpers for the Section-E (FEI + Branch C) figures.
Loads the REAL model classes from code/, the REAL NMT seed-0 fold-0 checkpoints and a REAL
NMT recording, records tensor shapes with forward hooks, and draws simple box diagrams."""
import os
os.environ["HW_THREADS"] = "1"          # code/ modules call cap_threads(4); HW_THREADS overrides that to 1
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU (a training job is running there)

import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import fei_pretrain as F          # Encoder, FEI, freq_mask, load_continuous, embed_recording, fit_probe
import fei_branchC as C           # spectrogram, DynEncoder, DynJEPA, embed_dyn, _configure
from head_to_head_cv import h20_args

CODE = "/home/mashfiq/eeg_vjepa/code"
FIGS = "/home/mashfiq/eeg_vjepa/docs/act0_guide/figs"
FEI_CKPT = f"{CODE}/fei_enc_cv_s0_f{{}}.pt"            # PLAIN FEI, NMT seed-0 fold-k
C_CKPT = f"{CODE}/branchC_h20_enc_cv_s0_f{{}}.pt"      # Branch C h20, NMT seed-0 fold-k
assert F.DEV == "cpu", F.DEV

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "savefig.dpi": 170, "font.family": "DejaVu Sans"})

BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#dbe8f6", "#fde3c8", "#dcefd9", "#eeeeee", "#f8d4d4", "#e7ddf3"


def configure_h20():
    """Set Branch-C globals to the Act-0 'h20' config (L=4000, nfft=200, hop=100, tc_frac=0.6)."""
    args = h20_args()
    C._configure(args)
    return args


def fold0_recording(label=0, min_frames=0):
    """First recording of the requested class in NMT seed-0 fold-0's TEST set
    (the fold-0 encoders never saw it, not even unlabeled)."""
    from sklearn.model_selection import StratifiedKFold
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))[1]
    for i in te:
        p, c = files[i]
        if c == label and np.load(p, mmap_mode="r").shape[0] >= min_frames:
            return p, len(files), len(te)
    raise RuntimeError("no recording found")


def load_fei(k=0, random_init=False, seed=0):
    torch.manual_seed(seed)
    enc = F.Encoder(256)
    if not random_init:
        enc.load_state_dict(torch.load(FEI_CKPT.format(k), map_location="cpu"))
    return enc.eval()


def load_c(k=0, random_init=False, seed=0):
    torch.manual_seed(seed)
    enc = C.DynEncoder(256)
    if not random_init:
        enc.load_state_dict(torch.load(C_CKPT.format(k), map_location="cpu"))
    return enc.eval()


def record_shapes(model, x, names):
    """Run model(x) with forward hooks on the named submodules; return [(name, module, in_shape, out_shape)]."""
    mods = dict(model.named_modules())
    rec, hooks = [], []

    def shp(t):
        if isinstance(t, (tuple, list)):
            return tuple(shp(u) for u in t)
        return tuple(t.shape)

    for n in names:
        hooks.append(mods[n].register_forward_hook(
            lambda m, i, o, n=n: rec.append((n, m, shp(i[0]), shp(o)))))
    with torch.no_grad():
        out = model(x)
    for h in hooks:
        h.remove()
    return rec, out


def nparams(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


# ------------------------------------------------------------- box-diagram helpers
def canvas(W, H):
    """Figure whose data units are inches (1 unit = 1 in), so box/text sizes are predictable."""
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=GREY, fs=9.5, ec="#444444", lw=1.2, ls="-"):
    """Rounded box centred at (x,y) (inch units, see canvas). First line bold, rest normal."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.08",
                                fc=color, ec=ec, lw=lw, ls=ls, zorder=2))
    lines = text.split("\n")
    line_h = fs / 72 * 1.28                       # inches per text line
    top = y + (len(lines) * line_h) / 2
    ax.text(x, top, lines[0], ha="center", va="top", fontsize=fs, weight="bold", zorder=3)
    if len(lines) > 1:
        ax.text(x, top - line_h, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 0.5,
                zorder=3, linespacing=1.28)


def arrow(ax, p, q, text="", color="#333333", ls="-", lw=1.4, fs=8.5, off=(0, 0), rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
        ax.text(mx, my, text, ha=ha, va="center", fontsize=fs, color=color, zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def shape_str(s):
    return "(" + ", ".join(str(v) for v in s) + ")"


def save(fig, name):
    path = f"{FIGS}/{name}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(path).size
    assert w <= 2400, f"{name} too wide: {w}px"
    print(f"saved {path}  ({w}x{h}px)")
