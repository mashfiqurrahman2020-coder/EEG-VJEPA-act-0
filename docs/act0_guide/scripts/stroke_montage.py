import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure C: the real 63-channel layout of the Zenodo stroke files (PAC02.vhdr header + the same
standard_1005 montage call as stroke_phaseB.load), with (left) the 19 channels the encoders use and
(right) the 21 central channels + 6 left/right pairs used by the Brain Symmetry Index (BSI)."""
import os
os.environ.setdefault("HW_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stroke_style import plt, save, RAW, STROKE, CONTROL, THIRD, VIOLET, INK, INK2, GATE
import numpy as np
import mne
from mne.channels.layout import _find_topomap_coords
import stroke_phaseB as B
from preprocess_nmt import CHANNELS

GATE()
raw = mne.io.read_raw_brainvision(f"{RAW}/PAC02.vhdr", preload=False, verbose=False)
raw.set_channel_types({c: "eog" for c in raw.ch_names if "EOG" in c} | {"A1": "misc"})
raw.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose=False)
info = raw.info
eeg = mne.pick_types(info, eeg=True)
names = [info.ch_names[i] for i in eeg]
xy = _find_topomap_coords(info, eeg)
pos = dict(zip(names, xy))
non_eeg = [c for c in raw.ch_names if c not in names]
inv = {v: k for k, v in B.RENAME.items()}                      # T3 -> T7 etc. (file name of each NMT channel)
enc_file = [inv.get(c, c) for c in CHANNELS]
print(f"{len(raw.ch_names)} channels in file: {len(names)} EEG + {non_eeg}; fs {info['sfreq']}")
assert all(c in pos for c in enc_file + B.CENTRAL21 + B.AH + B.UH)

r = np.abs(xy).max() * 1.08
fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
for ax, title in zip(axes, ["(a) 19 channels fed to the encoders", "(b) BSI: 21 central channels (zoomed)"]):
    ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=INK2, lw=1.2))
    ax.plot([-0.12 * r, 0, 0.12 * r], [0.99 * r, 1.12 * r, 0.99 * r], color=INK2, lw=1.2)   # nose
    ax.set_title(title, fontsize=13.5)
    ax.set_aspect("equal"); ax.axis("off")
axes[0].set_xlim(-1.15 * r, 1.15 * r); axes[0].set_ylim(-1.1 * r, 1.2 * r)
axes[0].text(-1.14 * r, 0.05 * r, "L", fontsize=13, color=INK2, va="center", fontweight="bold")
axes[0].text(1.06 * r, 0.05 * r, "R", fontsize=13, color=INK2, va="center", fontweight="bold")

ax = axes[0]
for c, (x, y) in pos.items():
    on = c in enc_file
    ax.scatter(x, y, s=150 if on else 22, color=INK2 if on else "#c9c8c3", zorder=3,
               edgecolor="white", linewidth=1.0)
    if not on:
        continue
    if c in B.RENAME:   # renamed side channels: label outward so they do not collide with C3/C4/P3/P4
        left = x < 0
        ax.text(x + (-0.06 if left else 0.06) * r, y + 0.07 * r, f"{c}→{B.RENAME[c]}", ha="right" if left else "left",
                va="bottom", fontsize=10.5, color=INK, zorder=4)
    else:
        ax.text(x, y - 0.075 * r, c, ha="center", va="top", fontsize=10.5, color=INK, zorder=4)

ax = axes[1]
cxy = np.array([pos[c] for c in B.CENTRAL21])
(x0, y0), (x1, y1) = cxy.min(0), cxy.max(0)
mx, my = 0.18 * (x1 - x0), 0.55 * (y1 - y0)
ax.set_xlim(x0 - mx, x1 + mx); ax.set_ylim(y0 - my, y1 + my * 0.8)
ax.text(x0 - mx, y1 + my * 0.55, "L", fontsize=13, color=INK2, va="center", fontweight="bold")
ax.text(x1 + mx, y1 + my * 0.55, "R", fontsize=13, color=INK2, va="center", ha="right", fontweight="bold")
pair = {c: k + 1 for k, (a, b) in enumerate(zip(B.AH, B.UH)) for c in (a, b)}
for c, (x, y) in pos.items():
    if c in pair:
        col, s_ = (VIOLET if c in B.AH else THIRD), 420
    elif c in B.CENTRAL21:
        col, s_ = "#8f8e89", 200
    else:
        col, s_ = "#c9c8c3", 30
    ax.scatter(x, y, s=s_, color=col, zorder=3, edgecolor="white", linewidth=1.0)
    if c in pair:
        ax.text(x, y, str(pair[c]), ha="center", va="center", fontsize=11, color="white", fontweight="bold", zorder=5)
    if c in B.CENTRAL21:
        ax.text(x, y - 0.012 * r * 4, c, ha="center", va="top", fontsize=10.5, color=INK, zorder=4)
h = [plt.Line2D([], [], marker="o", ls="", ms=11, color=VIOLET), plt.Line2D([], [], marker="o", ls="", ms=11, color=THIRD),
     plt.Line2D([], [], marker="o", ls="", ms=9, color="#8f8e89")]
axes[1].legend(h, ["left member of pair n", "right member of pair n", "other central (band-power features only)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1, frameon=False, fontsize=11)
axes[0].legend([plt.Line2D([], [], marker="o", ls="", ms=10, color=INK2),
                plt.Line2D([], [], marker="o", ls="", ms=5, color="#c9c8c3")],
               ["used (file name → NMT name)", "recorded, not used"], loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=1, frameon=False, fontsize=11)
save(fig, "stroke_montage.png")
