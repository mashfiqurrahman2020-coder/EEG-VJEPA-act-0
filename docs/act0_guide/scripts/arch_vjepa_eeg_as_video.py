import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Fig D.1  EEG as a video: real TUAB eval recording -> 119 frames -> one frame with the 4x30 patch grid ->
the 32-frame ViT-M clip grouped into tubelets of 4. Clip indexing = tuab_cached_eval.clip() (first position)."""
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_vjepa_common import *          # noqa: E402,F403
import matplotlib                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
BLUE, ORANGE, AQUA, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"

GATE()
f = eval_files("normal", 1)[0]
fr, idx, clip = load_clip(f, "vitm")
M = T.MODELS["vitm"]
nF, nC, nS = fr.shape                      # 119, 19, 500
hop = 250
cont = np.concatenate([fr[i, :, :hop] for i in range(nF - 1)] + [fr[-1]], axis=1)   # undo the 50 % overlap
t = np.arange(cont.shape[1]) / 100.0
k = 0                                       # frame shown in panel b/c = first clip frame
fz = CHANNELS.index("C3")

fig = plt.figure(figsize=(8.2, 11.0), dpi=170)
gs = fig.add_gridspec(7, 2, height_ratios=[0.85, 0.52, 1.4, 0.3, 0.9, 0.16, 0.8], width_ratios=[1, 1], hspace=0.0, wspace=0.28,
                      left=0.1, right=0.97, top=0.955, bottom=0.02)

# (a) whole recording + which frames the clip uses
ax = fig.add_subplot(gs[0, :])
ax.plot(t, np.clip(cont[fz], -4.5, 4.5), lw=0.4, color=INK2)
for j in range(nF):
    y0 = -6.3 if j % 2 == 0 else -7.3
    used = j in set(idx.tolist())
    ax.add_patch(Rectangle((j * 2.5, y0), 5, 0.8, fc=ORANGE if used else "#d9d8d4", ec="white", lw=0.3))
ax.set_xlim(0, 300); ax.set_ylim(-7.8, 4.8)
ax.set_yticks([])
ax.set_xlabel("time in the recording (s)")
ax.set_title("(a) One 300-s recording (channel C3) cut into 119 overlapping 5-s frames\n"
             f"(bars on two rows; orange = the {len(idx)} frames that one ViT-M clip uses)", loc="left")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# (b) one frame as traces
ax = fig.add_subplot(gs[2, 0])
F0 = clip[k]
for c in range(nC):
    ax.plot(np.arange(nS) / 100, F0[c] * 0.28 - c, lw=0.5, color=BLUE if c < 16 else ORANGE)
ax.set_yticks(-np.arange(nC)); ax.set_yticklabels(CHANNELS, fontsize=9)
ax.set_ylim(-nC + 0.3, 0.9); ax.set_xlim(0, 5)
ax.set_xlabel("time within the frame (s)")
ax.set_title("(b) One frame = 19 channels x 500 samples\n(5 s at 100 Hz), drawn as traces", loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (c) same frame as an image with the 4x30 patch grid
ax = fig.add_subplot(gs[2, 1])
ax.imshow(F0, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest", extent=(0, nS, nC, 0))
ph, pw = 4, 30
for r in range(0, 17, ph):
    ax.plot([0, 480], [r, r], color=INK, lw=0.6)
for c in range(0, 481, pw):
    ax.plot([c, c], [0, 16], color=INK, lw=0.6)
ax.add_patch(Rectangle((0, 16), 500, 3, fc="#8c8b86", alpha=0.75, hatch="///", ec=INK, lw=0.6))
ax.add_patch(Rectangle((480, 0), 20, 16, fc="#8c8b86", alpha=0.75, hatch="///", ec=INK, lw=0.6))
ax.add_patch(Rectangle((0, 0), pw, ph, fill=False, ec=ORANGE, lw=2.2))
ax.set_yticks(np.arange(nC) + 0.5); ax.set_yticklabels(CHANNELS, fontsize=9)
ax.set_xticks([0, 120, 240, 360, 480]); ax.set_xlabel("sample index (0-499)")
ax.set_title("(c) The same frame as an image,\ncut into 4 x 30 patches", loc="left")
ax.text(250, 17.6, "FZ, CZ, PZ: not inside any patch", ha="center", va="center", fontsize=9.5,
        bbox=dict(fc="white", ec="none", pad=1.5))

# (d) patch grid bookkeeping
ax = fig.add_subplot(gs[4, :])
ax.set_axis_off()
ax.set_xlim(-3.3, 16.1); ax.set_ylim(-1.0, 5.7)
for r in range(4):
    for c in range(16):
        ax.add_patch(Rectangle((c, 3 - r), 0.94, 0.9, fc="#e8eef8", ec=BLUE, lw=0.8))
        ax.text(c + 0.47, 3 - r + 0.45, f"{r * 16 + c}", ha="center", va="center", fontsize=8.5, color=INK2)
    ax.text(-0.15, 3 - r + 0.45, " ".join(CHANNELS[4 * r:4 * r + 4]), ha="right", va="center", fontsize=9.5)
ax.add_patch(Rectangle((0, 3), 0.94, 0.9, fill=False, ec=ORANGE, lw=2.2))
for c in range(0, 16, 4):
    ax.text(c + 0.47, 4.15, f"{c * 30}-{c * 30 + 29}", ha="center", fontsize=8.5, color=INK2)
ax.text(8, 4.75, "samples covered by each column (30 samples = 0.3 s)", ha="center", fontsize=10, color=INK2)
ax.text(-3.3, 5.35, "(d) One frame -> 4 rows x 16 columns = 64 patch positions (number = position index)",
        ha="left", fontsize=12)
ax.text(6.4, -0.7, "Row = 4 neighbouring channels in the file's order; orange box = the patch outlined in (c).", ha="center", fontsize=10, color=INK2)
ax.text(-1.7, 4.15, "channels in the row", fontsize=9, color=INK2, ha="center")

# (e) the clip: 32 frames -> 8 tubelets of 4 frames
ax = fig.add_subplot(gs[6, :])
ax.set_axis_off()
tub = M["tubelet_size"]
nT = len(idx) // tub
W, H, dx, dy = 1.55, 0.9, 0.12, 0.16
for g in range(nT):
    x0 = g * 2.05
    for j in range(tub):
        fi = g * tub + j
        e = (x0 + j * dx, 0.35 + j * dy, W, H)
        ax.imshow(clip[fi], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
                  extent=(e[0], e[0] + W, e[1], e[1] + H), zorder=10 + j)
        ax.add_patch(Rectangle((e[0], e[1]), W, H, fill=False, ec=INK, lw=0.5, zorder=10 + j))
    ax.add_patch(Rectangle((x0 - 0.06, 0.28), W + tub * dx, H + tub * dy + 0.05, fill=False, ec=ORANGE,
                           lw=1.8, zorder=30))
    ax.text(x0 + 0.85, 0.08, "frames\n" + ",".join(str(v) for v in idx[g * tub:(g + 1) * tub]), ha="center",
            va="top", fontsize=8.5, color=INK2)
    ax.text(x0 + 0.85, 1.9, f"tubelet {g + 1}", ha="center", fontsize=10)
ax.set_xlim(-0.2, nT * 2.05); ax.set_ylim(-0.75, 2.75)
ax.text(-0.2, 2.45, f"(e) One ViT-M clip = {len(idx)} frames (about every 3rd frame, frames {idx[0]}-{idx[-1]}),\n"
        f"     grouped into {nT} tubelets of {tub} frames each", ha="left", fontsize=12)

out = f"{FIGS}/arch_vjepa_eeg_as_video.png"
fig.savefig(out, dpi=170)
print("saved", out, "recording", os.path.basename(f), "frames", fr.shape, "clip idx", idx.tolist())
