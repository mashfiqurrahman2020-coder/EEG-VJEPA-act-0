import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure A: one real NMT recording, raw EDF vs the stored 200 Hz .npy (same 10 s, 4 channels).
Also VERIFIES the 200 Hz recipe on 8 recordings (2 per class x split): stored == per-channel z-score of the
raw 19 channels over the WHOLE recording, cut into floor(T/500) non-overlapping 500-sample frames,
no filter, no resampling, no re-referencing. And probes the reference (A1 vs A2, 19-channel mean).
Writes figs/nmt_raw_vs_prep.png and facts/nmt_recipe_check.json."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
from nmt_style import *   # noqa: F401,F403
import glob, json, os
import numpy as np
import mne

z = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
checks = []
for lab in ("normal", "abnormal"):
    for split in ("train", "eval"):
        for idx in (0, 5):
            cool()
            edf = sorted(glob.glob(f"{RAW}/{lab}/{split}/*.edf"))[idx]
            rid = os.path.basename(edf)[:-4]
            x = np.load(f"{PREP}/{split}/{lab}/{rid}.npy")                  # (F,19,500) float32
            raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
            allv = raw.get_data()                                            # volts (MNE converts the EDF "uV")
            r19 = allv[[raw.ch_names.index(c) for c in CH]]
            F = x.shape[0]; T = r19.shape[1]
            cont = x.transpose(1, 0, 2).reshape(19, -1).astype(np.float64)
            zfull = z(r19)[:, :F * 500]                                      # z over ALL T samples, then drop tail
            ztrunc = z(r19[:, :F * 500])                                      # alternative: z after dropping tail
            corr = [np.corrcoef(cont[i], r19[i, :F * 500])[0, 1] for i in range(19)]
            a1, a2 = allv[raw.ch_names.index("A1")], allv[raw.ch_names.index("A2")]
            checks.append(dict(
                rid=rid, label=lab, split=split, T=T, F=F, tail_dropped=T - F * 500, stored_shape=list(x.shape),
                stored_dtype=str(x.dtype), max_abs_diff_vs_z_full=float(np.abs(cont - zfull).max()),
                max_abs_diff_vs_z_after_tail_drop=float(np.abs(cont - ztrunc).max()),
                min_corr_stored_vs_raw=float(min(corr)),
                stored_channel_mean_max=float(np.abs(cont.mean(1)).max()),
                corr_A1_A2=float(np.corrcoef(a1, a2)[0, 1]),
                rms_of_19ch_mean_over_mean_ch_rms=float((r19.mean(0)).std() / r19.std(1).mean()),
                median_ch_std_uV=float(np.median(r19.std(1)) * 1e6),
                sfreq=raw.info["sfreq"], mne_highpass=raw.info["highpass"], mne_lowpass=raw.info["lowpass"]))
            print(checks[-1], flush=True)
json.dump(checks, open(f"{FACTS}/nmt_recipe_check.json", "w"), indent=2)

# ---------------- figure: first normal/eval recording, 4 channels, 10 s ----------------
cool()
lab, split = "normal", "eval"
edf = sorted(glob.glob(f"{RAW}/{lab}/{split}/*.edf"))[0]
rid = os.path.basename(edf)[:-4]
x = np.load(f"{PREP}/{split}/{lab}/{rid}.npy")
cont = x.transpose(1, 0, 2).reshape(19, -1)
raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
r19 = raw.get_data(picks=CH) * 1e6                    # back to the EDF's own unit label ("uV")
show = ["FP1", "C3", "O1", "T4"]
t0, dur = 60.0, 10.0
s0, s1 = int(t0 * FS), int((t0 + dur) * FS)
t = np.arange(s0, s1) / FS
chk = next(c for c in checks if c["rid"] == rid)

fig, axes = plt.subplots(len(show), 2, figsize=(8.0, 5.6), sharex=True,
                         gridspec_kw=dict(hspace=0.12, wspace=0.28))
for i, ch in enumerate(show):
    k = CH.index(ch)
    axes[i, 0].plot(t, r19[k, s0:s1], color=INK2, lw=0.8)
    axes[i, 1].plot(t, cont[k, s0:s1], color=BLUE, lw=0.8)
    axes[i, 0].set_ylabel(ch, rotation=0, ha="right", va="center", fontsize=12, fontweight="bold")
    for a in axes[i]:
        a.grid(axis="x", visible=False)
    axes[i, 1].axhline(0, color=MUTED, lw=0.6)
# frame borders (every 2.5 s = 500 samples) on the stored side
for a in axes[:, 1]:
    for fb in np.arange(np.ceil(t0 / 2.5) * 2.5, t0 + dur + 1e-9, 2.5):
        a.axvline(fb, color=ORANGE, lw=1.0, ls="--", zorder=0)
axes[0, 0].set_title(f"(a) Raw EDF {rid}.edf\n(file's own unit: \"uV\")", loc="left", fontsize=12)
axes[0, 1].set_title("(b) Stored .npy (z-scored)\ndashed = 2.5 s frame edges", loc="left", fontsize=12)
axes[-1, 0].set_xlabel("time in recording (s)")
axes[-1, 1].set_xlabel("time in recording (s)")
save(fig, "nmt_raw_vs_prep.png")
