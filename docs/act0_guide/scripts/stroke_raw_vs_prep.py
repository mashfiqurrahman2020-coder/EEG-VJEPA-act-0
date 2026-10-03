import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure C: one real subject (PAC02), the same 10 s of signal at each stage:
raw 1000 Hz file (A2 reference) -> ViT-M input (CAR, 1-40 Hz, 100 Hz, z) -> FEI+C preregistered input
(CAR, 0.5-40 Hz, 200 Hz, z) -> FEI+C clean input (CAR, 200 Hz, NMT-spectrum-matched filter, z).
The three prepared signals are read from the SAVED arrays the experiments used; before plotting, the script
re-runs the real prep functions (stroke_phaseB.encoder_preps, stroke_phaseB_clean.base/shape) on the raw file
and asserts they reproduce the saved arrays."""
import os
os.environ.setdefault("HW_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stroke_style import plt, save, ROOT, RUNS, INK, INK2, GRID, STROKE, CONTROL, THIRD, VIOLET, GATE
import numpy as np
import stroke_phaseB as B
import stroke_phaseB_clean as BC
from preprocess_nmt import CHANNELS

SUBJ, T0, T1 = "PAC02", 100.0, 110.0
SHOW = ["FZ", "C3", "C4", "O1"]          # NMT names (all 4 have the same name in the file)

GATE()
raw, bads = B.load(SUBJ, B.COHORTS["primary"][1])          # 285 s, bad-channel interpolation (none EEG for PAC02)
print("raw:", len(raw.ch_names), "ch,", raw.info["sfreq"], "Hz,", raw.n_times, "samples; bads", bads)
vit = np.load(f"{ROOT}/data/zenodo_stroke_prep/primary/vitm/{SUBJ}.npy")
fei = np.load(f"{ROOT}/data/zenodo_stroke_prep/primary/feic/{SUBJ}.npy")
cln = np.load(f"{ROOT}/data/zenodo_stroke_prep_nmtmatch/primary/feic/{SUBJ}.npy")
print("saved shapes: vitm", vit.shape, " feic", fei.shape, " clean feic", cln.shape)

GATE()   # verify: the real prep functions reproduce the saved arrays
re = B.encoder_preps(raw)
assert re["vitm"].shape == vit.shape and np.abs(re["vitm"] - vit).max() < 1e-3, np.abs(re["vitm"] - vit).max()
assert re["feic"].shape == fei.shape and np.abs(re["feic"] - fei).max() < 1e-3, np.abs(re["feic"] - fei).max()
GATE()
filt = np.load(f"{RUNS}/phaseB_clean/filter_primary.npz")
d = BC.zs(BC.shape(BC.base(raw), filt["f"], filt["log10_amp_gain"]))
rc = BC.frames(d)
print("max |diff| vs saved: vitm %.2e feic %.2e clean %.2e" % (np.abs(re["vitm"] - vit).max(),
      np.abs(re["feic"] - fei).max(), np.abs(rc - cln).max()))
assert rc.shape == cln.shape and np.abs(rc - cln).max() < 1e-3


def cont(frames, fs, stride_s, t0, t1):
    """Rebuild continuous signal (19, n) for [t0, t1) from frames that start every stride_s seconds."""
    step = int(round(stride_s * fs)); W = frames.shape[2]
    k = int(round(W / step))                             # use every k-th frame -> non-overlapping pieces
    f0 = int(round(t0 / stride_s))
    pieces = [frames[i] for i in range(f0, f0 + int(round((t1 - t0) * fs / W)) * k, k)]
    return np.concatenate(pieces, 1)


fs_raw = raw.info["sfreq"]
rx = raw.get_data(picks=SHOW, start=int(T0 * fs_raw), stop=int(T1 * fs_raw)) * 1e6        # uV, A2 reference
rx = rx - rx.mean(1, keepdims=True)
ci = [CHANNELS.index(c) for c in SHOW]
stages = [("(1) raw file: 1000 Hz, reference A2, µV", rx, 1000, "µV", INK2),
          ("(2) ViT-M input: 1–40 Hz, 100 Hz, z-scored", cont(vit, 100, 2.5, T0, T1)[ci], 100, "SD", VIOLET),
          ("(3) FEI+C input (preregistered): 0.5–40 Hz, 200 Hz", cont(fei, 200, 2.5, T0, T1)[ci], 200, "SD", STROKE),
          ("(4) FEI+C input (clean): NMT-matched, 200 Hz", cont(cln, 200, 2.5, T0, T1)[ci], 200, "SD", THIRD)]

fig, axes = plt.subplots(4, 2, figsize=(9.4, 10.2), gridspec_kw=dict(width_ratios=[3.2, 1], hspace=0.6, wspace=0.16))
Z0, Z1 = 103.0, 103.5                                    # zoom window (s)
for row, (title, x, fs, unit, col) in enumerate(stages):
    t = T0 + np.arange(x.shape[1]) / fs
    gap = 6 * np.median(x.std(1))
    ax = axes[row, 0]
    for j, c in enumerate(SHOW):
        ax.plot(t, x[j] - j * gap, color=col, lw=0.7)
    ax.set_yticks([-j * gap for j in range(len(SHOW))], SHOW)
    ax.set_xlim(T0, T1)
    ax.set_title(title, fontsize=12.5)
    ax.axvspan(Z0, Z1, color=GRID, alpha=0.6, lw=0)
    if row == 1:          # 5 s frames, a new one every 2.5 s
        for k, s0 in enumerate(np.arange(T0, T1 - 4.99, 2.5)):
            yy = (0.75 if k % 2 == 0 else 1.25) * gap
            ax.annotate("", xy=(s0 + 5, yy), xytext=(s0, yy),
                        arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0))
        ax.text(T1 - 0.05, 1.45 * gap, "5 s frames, one every 2.5 s", ha="right", va="bottom", fontsize=11, color=INK2)
    if row >= 2:          # 2.5 s non-overlapping frames
        for s0 in np.arange(T0, T1 + 0.01, 2.5):
            ax.axvline(s0, color=INK2, lw=0.8, ls=":")
        ax.text(T1 - 0.05, 1.2 * gap, "2.5 s frames, no overlap", ha="right", va="bottom", fontsize=11, color=INK2)
    ax.set_ylim(-(len(SHOW) - 1) * gap - 1.6 * gap, 2.1 * gap)
    # scale bar
    sb = 50 if unit == "µV" else 2
    yb = -(len(SHOW) - 1) * gap - 1.5 * gap
    ax.plot([T0 + 0.1, T0 + 0.1], [yb, yb + sb], color=INK, lw=2)
    ax.text(T0 + 0.2, yb + sb / 2, f"{sb} {unit}", va="center", fontsize=11)
    if row == 3:
        ax.set_xlabel("time in the recording (s)")
    # zoom: C3 only, every sample is a dot
    az = axes[row, 1]
    m = (t >= Z0) & (t < Z1)
    az.plot(t[m], x[1, m], color=col, lw=0.8)
    az.plot(t[m], x[1, m], "o", color=col, ms=2.2 if fs == 1000 else 3.5)
    az.set_title(f"C3 zoom: {int(m.sum())} samples", fontsize=11.5)
    az.set_yticks([]); az.set_xticks([Z0, Z1], [f"{Z0:g}", f"{Z1:g} s"])
    az.spines["left"].set_visible(False)
from scipy.signal import welch   # caption check: is there mains hum (50 Hz) in the raw file?
f, p = welch(raw.get_data(picks=["C3"])[0], fs=fs_raw, nperseg=4000)
print("raw C3 PSD at 50 Hz / median 45-55 Hz (excl. 49-51):", p[np.argmin(abs(f - 50))] / np.median(p[((f > 45) & (f < 49)) | ((f > 51) & (f < 55))]))
save(fig, "stroke_raw_vs_prep.png")
