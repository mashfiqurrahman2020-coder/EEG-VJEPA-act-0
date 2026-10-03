import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Shared loaders for the section-D (EEG-VJEPA) figure scripts. CPU only, one thread.
Uses the project's own code: MODELS + clip() from code/tuab_cached_eval.py, init_model() from upstream eval.py."""
import os
os.environ.setdefault("HW_THREADS", "1")   # tuab_cached_eval calls cap_threads(4); this keeps it at 1
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import glob
import json

import numpy as np
import torch

torch.set_num_threads(1)
import tuab_cached_eval as T                                   # MODELS dict + clip() (the loader's indexing)
from evals.eeg_classification_frozen.eval import init_model    # upstream model builder

ROOT = "/home/mashfiq/eeg_vjepa"
GUIDE = f"{ROOT}/docs/act0_guide"
FIGS = f"{GUIDE}/figs"
EVAL = f"{ROOT}/data/TUAB_preprocessed_100hz/eval"
CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']   # = code/preprocess_nmt.py CHANNELS
GATE = lambda: cool_gate(pause=88.0, resume=78.0, abort=92.0)


def eval_files(label, n):
    return sorted(glob.glob(f"{EVAL}/{label}/*.npy"))[:n]


def load_clip(path, model="vitm", end=None):
    """Frames of one recording -> the clip ending at `end` (default = first position, end = clip_len),
    exactly as tuab_cached_eval.clip() / VideoDataset.loadvideo_decord index it. Returns (frames, idx, clip)."""
    M = T.MODELS[model]
    fr = np.load(path).astype("float32")                       # (119, 19, 500)
    L = M["frames_per_clip"] * M["frame_step"]
    end = L if end is None else end
    idx = T.clip(np.arange(len(fr)), end, M)                   # frame indices used by the clip
    return fr, idx, fr[idx]


def encoder(model="vitm", ckpt="released", seed=0):
    M = T.MODELS[model]
    torch.manual_seed(seed)
    return init_model("cpu", M["ckpt"] if ckpt == "released" else "none", M["model_name"], patch_size=(4, 30),
                      crop_size=224, frames_per_clip=M["frames_per_clip"], tubelet_size=M["tubelet_size"],
                      use_sdpa=True, use_SiLU=False, tight_SiLU=False, uniform_power=True).eval()


def save_json(name, obj):
    os.makedirs(f"{GUIDE}/facts", exist_ok=True)
    json.dump(obj, open(f"{GUIDE}/facts/{name}", "w"), indent=1)


def fmt_params(n):
    return f"{n/1e6:.2f} M" if n >= 1e6 else (f"{n/1e3:.1f} k" if n >= 1e3 else str(n))


# ---- tiny drawing helpers for the block diagrams (matplotlib patches only) ----
BLUE, ORANGE, AQUA, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#8c8b86"
FILL = {"data": "#f1f0ec", "frozen": "#e3edfa", "train": "#fde6dc", "ema": "#dcf3ea", "none": "white"}
EDGE = {"data": GREY, "frozen": BLUE, "train": ORANGE, "ema": AQUA, "none": GREY}


def box(ax, x, y, w, h, text, kind="data", fs=10, bold_first=True, ls="-"):
    """Rounded box centred at (x, y); first line of `text` in bold."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4, ls=ls, zorder=2))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:     # bold title line + body lines, stacked by point offsets
        lh = fs * 1.3
        n = len(lines) - 1
        ax.annotate(lines[0], (x, y), xytext=(0, n * lh / 2), textcoords="offset points", ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=INK, zorder=3)
        ax.annotate("\n".join(lines[1:]), (x, y), xytext=(0, -lh / 2), textcoords="offset points", ha="center",
                    va="center", fontsize=fs - 0.5, color=INK, zorder=3, linespacing=1.2)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3,
                fontweight="bold" if bold_first else "normal")


def arrow(ax, p, q, text=None, color=INK2, fs=9.5, ls="-", side="right", rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx + (0.08 if side == "right" else -0.08), my, text, ha="left" if side == "right" else "right",
                va="center", fontsize=fs, color=color)


def shape_str(s):
    return "(" + ", ".join(str(v) for v in s) + ")"
