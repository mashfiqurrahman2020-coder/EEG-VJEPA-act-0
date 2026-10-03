import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure A: how ONE real stored NMT recording is cut into model windows.
(a) whole-recording timeline: random pre-training windows drawn by the REAL fei_pretrain.load_window
    (FEI: L=1000 = 5 s; Branch C h20: L=4000 = 20 s; wpe=4 draws per recording per epoch), and the
    non-overlapping 20 s embedding windows (embed_recording / embed_dyn rule) with the dropped tail.
(b) zoom on one 20 s window: the 2.5 s storage frames and which frames load_window actually reads.
Also asserts load_window returns exactly cont[:, s:s+L]. Writes figs/nmt_windows.png, facts/nmt_windows.json."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
from nmt_style import *   # noqa: F401,F403
import json, random
import numpy as np
from matplotlib.patches import Rectangle
import fei_pretrain as FP            # real loader code (CPU: CUDA_VISIBLE_DEVICES="")

cool()
RID, SPLIT, LAB = "0000024", "eval", "normal"
path = f"{PREP}/{SPLIT}/{LAB}/{RID}.npy"
cont = FP.load_continuous(path)                         # (19, T)
T = cont.shape[1]
WPE = 4                                                 # fei_pretrain/fei_branchC --wpe default


def draws(seed):
    """Replay load_window's random start (random.randint(0, T-L)): 4 FEI (L=1000) then 4 C (L=4000) draws
    from one seeded stream, then re-seed and check the REAL load_window returns exactly those slices."""
    plan = [1000] * WPE + [4000] * WPE
    random.seed(seed); starts = [random.randint(0, T - L) for L in plan]
    random.seed(seed); wins = [FP.load_window(path, L) for L in plan]
    for L, s, w in zip(plan, starts, wins):
        assert w.shape == (19, L) and np.array_equal(w, cont[:, s:s + L]), "load_window mismatch"
    return starts[:WPE], starts[WPE:]


fei_starts, c_starts = draws(0)
emb_starts = list(range(0, T - 4000 + 1, 4000))         # embed_recording / embed_dyn, L=4000
tail = T - (emb_starts[-1] + 4000)
emb1000 = len(range(0, T - 1000 + 1, 1000))
facts = dict(recording=f"{SPLIT}/{LAB}/{RID}.npy", stored_shape=list(np.load(path, mmap_mode="r").shape),
             T_samples=T, duration_s=T / FS, wpe=WPE,
             fei_pretrain_window_starts_s=[s / FS for s in fei_starts],
             c_pretrain_window_starts_s=[s / FS for s in c_starts],
             n_embedding_windows_L4000=len(emb_starts), tail_dropped_L4000_samples=tail,
             n_embedding_windows_L1000=emb1000, tail_dropped_L1000_samples=T - emb1000 * 1000,
             load_window_matches_slice=True)
json.dump(facts, open(f"{FACTS}/nmt_windows.json", "w"), indent=2)
print(facts)

# ---------------- figure ----------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.2), gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.55))
dur = T / FS
lane = 0.22                                             # one thin lane per random draw, so overlaps stay visible
a1.add_patch(Rectangle((0, 3.0 - 0.3), dur, 0.6, color=MUTED, alpha=0.35, lw=0))
for row, starts, L, col in ((2.0, fei_starts, 1000, BLUE), (1.0, c_starts, 4000, AQUA)):
    for j, s0 in enumerate(starts):
        a1.add_patch(Rectangle((s0 / FS, row + 0.36 - (j + 1) * lane), L / FS, lane * 0.8, color=col, lw=0))
for i, s0 in enumerate(emb_starts):
    a1.add_patch(Rectangle((s0 / FS, -0.3), 20, 0.6, facecolor=ORANGE if i % 2 == 0 else YELLOW,
                           edgecolor="white", lw=0.8))
a1.add_patch(Rectangle((emb_starts[-1] / FS + 20, -0.3), tail / FS, 0.6, facecolor="white",
                       edgecolor=INK2, hatch="////", lw=0.8))
a1.annotate(f"last {tail / FS:.0f} s dropped", xy=(dur - tail / FS / 2, -0.3), xytext=(dur - 5, -0.95),
            fontsize=10.5, ha="right", va="center", arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
a1.text(dur / 2, 3.0, f"{RID}.npy: {dur:.0f} s = {T // 500} frames of 2.5 s", ha="center", va="center",
        fontsize=10.5)
a1.text(8, -0.95, f"{len(emb_starts)} windows, averaged\ninto one vector", ha="left", va="center", fontsize=10.5)
a1.set_yticks([3.0, 2.0, 1.0, 0.0], ["stored\nrecording", "FEI training\n4 x 5 s", "Branch C training\n4 x 20 s",
                                     "embedding\n20 s, no overlap"], fontsize=10.5)
a1.set_ylim(-1.35, 3.4); a1.set_xlim(0, dur)
a1.set_xlabel("time in recording (s)")
a1.grid(axis="y", visible=False)
a1.set_title("(a) One recording, cut three ways", loc="left")

# (b) zoom on the first Branch-C training window
s = c_starts[0]; L = 4000
t = np.arange(s, s + L) / FS
k = CH.index("C3")
f0, f1 = s // 500, (s + L - 1) // 500
a2.axvspan(s / FS, (s + L) / FS, color=AQUA, alpha=0.15, lw=0, zorder=0)
for f in range(f0, f1 + 2):
    a2.axvline(f * 2.5, color=ORANGE, lw=1.0, ls="--", zorder=1)
a2.plot(t, cont[k, s:s + L], color=INK2, lw=0.6, zorder=2)
a2.grid(axis="x", visible=False)
a2.set_xlim(f0 * 2.5 - 0.6, (f1 + 1) * 2.5 + 0.6)
yl = np.abs(cont[k, s:s + L]).max() * 1.1
a2.set_ylim(-yl, yl * 1.7)
a2.text(f0 * 2.5 - 0.4, yl * 1.12, f"the window starts at {s / FS:.2f} s; load_window reads only frames {f0}-{f1}\n"
        f"({f1 - f0 + 1} frames x 500 samples) from disk and cuts the 4000 samples out", ha="left", fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=3)
a2.set_xlabel("time in recording (s)"); a2.set_ylabel("C3 (z-score)")
a2.set_title("(b) Zoom: one 20 s window (green) over 2.5 s frames (dashed)", loc="left")
save(fig, "nmt_windows.png")
