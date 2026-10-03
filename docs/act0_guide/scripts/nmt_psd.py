import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure A: NMT's natural spectrum. Mean log10 Welch PSD of 20 normal + 20 abnormal stored recordings
(randomly drawn, seed 0, first 300 s each, 0.25 Hz bins, channel-averaged). Data are z-scored, so the PSD
integrates to ~1 = "relative" PSD. Shows the roll-off near 35-40 Hz onto a flat floor and how little
power sits below 0.5 Hz. Writes figs/nmt_psd.png and facts/nmt_psd.json."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
from nmt_style import *   # noqa: F401,F403
import glob, json
import numpy as np
from scipy.signal import welch

N_PER_CLASS, NPERSEG, SECONDS = 20, 800, 300
rng = np.random.default_rng(0)
res, fq = {}, None
for lab in ("normal", "abnormal"):
    files = sorted(glob.glob(f"{PREP}/*/{lab}/*.npy"))
    long_enough = [p for p in files if np.load(p, mmap_mode="r").shape[0] * 500 >= SECONDS * FS]
    pick = [long_enough[i] for i in rng.choice(len(long_enough), N_PER_CLASS, replace=False)]
    L, frac = [], []
    for p in pick:
        cool()
        x = np.asarray(np.load(p, mmap_mode="r")[: SECONDS * FS // 500])            # first 300 s = 120 frames
        sig = x.transpose(1, 0, 2).reshape(19, -1).astype(np.float64)
        fq, P = welch(sig, fs=FS, nperseg=NPERSEG, axis=-1)                         # (19, 401)
        L.append(np.log10(P + 1e-20).mean(0))                                       # channel-averaged log10 PSD
        tot = P.sum(1)
        frac.append([(P[:, fq < 0.5].sum(1) / tot).mean(), (P[:, (fq >= 0.5) & (fq <= 40)].sum(1) / tot).mean(),
                     (P[:, fq > 40].sum(1) / tot).mean()])
    res[lab] = dict(files=[p.split("/")[-1] for p in pick], logpsd=np.array(L), frac=np.array(frac))

allL = np.vstack([res[l]["logpsd"] for l in res])
mean_all = allL.mean(0)
at = lambda hz: float(np.interp(hz, fq, mean_all))
floor = float(np.median(mean_all[(fq >= 60) & (fq <= 95)]))
facts = {
    "n_recordings": {l: len(res[l]["files"]) for l in res},
    "files": {l: res[l]["files"] for l in res},
    "welch": dict(fs=FS, nperseg=NPERSEG, bin_hz=FS / NPERSEG, seconds_used=SECONDS),
    "mean_log10_psd_at_hz": {str(h): at(h) for h in (0.25, 0.5, 1, 2, 5, 10, 20, 30, 35, 40, 45, 50, 60, 80, 99)},
    "floor_median_60_95Hz": floor,
    "variance_fraction_mean": {l: dict(zip(["below_0.5Hz", "0.5_to_40Hz", "above_40Hz"],
                                           res[l]["frac"].mean(0).round(5).tolist())) for l in res},
}
# cross-check against the stored 100-recording NMT-train reference used by stroke_phaseB_clean.py
try:
    ref = np.load(f"{ROOT}/code/runs/act0/phaseB_clean/nmt_ref_psd.npz")
    rm = ref["mean"].mean(0)
    facts["crosscheck_phaseB_clean_nmt_ref_psd"] = {
        "same_grid": bool(np.allclose(ref["f"], fq)),
        "max_abs_diff_of_channel_mean_logpsd_0.5_99Hz": float(np.abs(rm - mean_all)[(fq >= 0.5) & (fq <= 99)].max()),
        "ref_floor_median_60_95Hz": float(np.median(rm[(fq >= 60) & (fq <= 95)]))}
except Exception as e:   # noqa: BLE001  (the cross-check is optional)
    facts["crosscheck_phaseB_clean_nmt_ref_psd"] = f"unavailable: {e}"
json.dump(facts, open(f"{FACTS}/nmt_psd.json", "w"), indent=2)
print(json.dumps({k: v for k, v in facts.items() if k != "files"}, indent=1))

# ---------------- figure ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 4.1), gridspec_kw=dict(width_ratios=[2.1, 1], wspace=0.25))
m = fq > 0
for lab, col in (("normal", NORMAL), ("abnormal", ABNORMAL)):
    L = res[lab]["logpsd"]
    mu, lo, hi = L.mean(0), np.percentile(L, 25, 0), np.percentile(L, 75, 0)
    for a in (a1, a2):
        a.fill_between(fq[m], lo[m], hi[m], color=col, alpha=0.18, lw=0)
        a.plot(fq[m], mu[m], color=col, lw=2, label=f"{lab} (n={len(L)})")
bands = [("\u03b4", 0.5, 4), ("\u03b8", 4, 8), ("\u03b1", 8, 13), ("\u03b2", 13, 30)]   # delta theta alpha beta
for name, lo_, hi_ in bands:
    a1.text((lo_ + hi_) / 2, -0.3, name, ha="center", fontsize=12, color=INK2)
    a1.axvline(hi_, color=GRID, lw=1, zorder=0)
a1.axvspan(35, 40, color=YELLOW, alpha=0.25, lw=0, zorder=0)
a1.annotate("roll-off\n~35-40 Hz", xy=(38, at(38)), xytext=(42, at(38) + 2.4), fontsize=11,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
a1.axhline(floor, color=INK2, lw=1, ls="--")
a1.text(98, floor + 0.45, f"flat floor ~ {floor:.1f}", ha="right", fontsize=11, color=INK2)
a1.annotate("50 Hz\npower line", xy=(50, at(50) + 0.15), xytext=(57, at(50) + 1.0), fontsize=10.5, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
a1.set_xlim(0, 100); a1.set_ylim(floor - 1.0, 0.1)
a1.set_xlabel("frequency (Hz)"); a1.set_ylabel("log10 relative power (per Hz)")
a1.set_title("(a) Mean spectrum, 0-100 Hz", loc="left")
a1.legend(frameon=False, loc="upper right")
a2.axvspan(0, 0.5, color=YELLOW, alpha=0.25, lw=0, zorder=0)
fb = np.mean([facts["variance_fraction_mean"][l]["below_0.5Hz"] for l in ("normal", "abnormal")])
a2.text(0.6, -5.0, f"below 0.5 Hz:\nonly {100*fb:.1f}% of\nthe power", ha="left", fontsize=10.5, color=INK)
a2.set_xlim(0, 4); a2.set_ylim(floor - 1.0, 0.1)
a2.set_xlabel("frequency (Hz)")
a2.set_title("(b) Zoom: 0-4 Hz", loc="left")
save(fig, "nmt_psd.png")
