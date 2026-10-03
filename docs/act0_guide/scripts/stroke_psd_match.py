import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figures C: why the clean FEI+C re-prep exists and what its gates showed.
stroke_psd_match.png  (a) mean log10 PSD: NMT-train reference (nmt_ref_psd.npz) vs the 15 primary-cohort stroke
                      inputs: unfiltered (recomputed with stroke_phaseB_clean.base), preregistered 0.5-40 Hz prep and
                      clean matched prep (both from the saved arrays); (b) the fixed matching filter (filter_primary.npz).
stroke_clean_gates.png (c) Branch-C input BatchNorm z per 1 Hz bin and (d) Branch-C embedding spread, from
                      runs/act0/phaseB/collapse_primary.json and runs/act0/phaseB_clean/collapse_primary.json."""
import json
import os
os.environ.setdefault("HW_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stroke_style import plt, save, ROOT, RUNS, INK, INK2, GRID, STROKE, CONTROL, THIRD, VIOLET, GATE
import numpy as np
import stroke_phaseB as B
import stroke_phaseB_clean as BC

subs = B.COHORTS["primary"][0]
ref = np.load(f"{RUNS}/phaseB_clean/nmt_ref_psd.npz")
f, nmt = ref["f"], ref["mean"].mean(0)                      # (401,) channel-averaged log10 PSD, 0.25 Hz bins
flt = np.load(f"{RUNS}/phaseB_clean/filter_primary.npz")
summ = json.load(open(f"{RUNS}/phaseB_clean/prep_primary.json"))


def group_logpsd(get):
    L = []
    for s in subs:
        GATE()
        L.append(BC.logpsd(get(s))[1])                       # (19, F)
    return np.mean(L, 0).mean(0)


pre = group_logpsd(lambda s: np.load(f"{ROOT}/data/zenodo_stroke_prep/primary/feic/{s}.npy")
                   .transpose(1, 0, 2).reshape(19, -1))
cln = group_logpsd(lambda s: np.load(f"{ROOT}/data/zenodo_stroke_prep_nmtmatch/primary/feic/{s}.npy")
                   .transpose(1, 0, 2).reshape(19, -1))
unf = group_logpsd(lambda s: BC.base(B.load(s, B.COHORTS["primary"][1])[0]))
for name, c in [("prereg_prep", pre), ("matched", cln), ("unfiltered", unf), ("nmt", nmt)]:   # cross-check vs json
    for hz in (2, 10, 35, 60):
        v, j = float(np.interp(hz, f, c)), summ[name]["log10psd"][f"{hz}Hz"]
        print(f"  {name:12s} {hz:3d} Hz: recomputed {v:7.3f}  prep_primary.json {j:7.3f}")
        assert abs(v - j) < 0.05, (name, hz, v, j)

fig, (a, b) = plt.subplots(2, 1, figsize=(9.2, 8.6), gridspec_kw=dict(height_ratios=[1.7, 1], hspace=0.38))
a.axvspan(0.5, 40, color=GRID, alpha=0.45, lw=0)
a.text(1.0, -10.9, "band kept by the preregistered\n0.5–40 Hz filter (grey)", ha="left", fontsize=11, color=INK2)
a.plot(f, unf, color=INK2, lw=1.4, ls="--", label="stroke, CAR + 200 Hz, no filter (starting point)")
a.plot(f, pre, color=STROKE, lw=2, label="stroke, preregistered prep (0.5–40 Hz)")
a.plot(f, nmt, color=INK, lw=3.2, label="NMT reference (what the encoders were trained on)")
a.plot(f, cln, color=THIRD, lw=1.8, label="stroke, clean prep (matched to NMT)")
a.set_xlim(0, 100); a.set_ylim(-11.3, 0.3)
a.set_ylabel("log10 power (z-scored signal)")
a.set_xlabel("frequency (Hz)")
a.set_title("(a) Average power spectrum of the encoder input, 15 primary-cohort subjects vs NMT")
a.legend(loc="upper right", frameon=True, framealpha=1, edgecolor=GRID, fontsize=11)
a.annotate("preregistered prep keeps power\n40–45 Hz, then drops far below NMT", xy=(62, float(np.interp(62, f, pre))),
           xytext=(55, -4.9), va="center", fontsize=11, color=STROKE, arrowprops=dict(arrowstyle="->", color=STROKE))
a.annotate("NMT: steep roll-off near\n35–40 Hz onto a flat floor", xy=(38, float(np.interp(38, f, nmt))),
           xytext=(8, -8.3), fontsize=11, color=INK, arrowprops=dict(arrowstyle="->", color=INK))

g = flt["log10_amp_gain"]
b.fill_between(f, g.min(0), g.max(0), color=THIRD, alpha=0.25, lw=0, label="range over the 19 channels")
b.plot(f, g.mean(0), color=THIRD, lw=2, label="mean over channels")
b.axhline(0, color=INK2, lw=0.8)
b.set_xlim(0, 100)
b.set_xlabel("frequency (Hz)"); b.set_ylabel("log10 amplitude gain")
b.set_title("(b) The one fixed matching filter (same for every subject, labels never used)")
b.legend(loc="lower left", frameon=False, fontsize=11)
b.text(99, 0.25, "0 = unchanged;  −1 = amplitude ÷ 10", ha="right", fontsize=11, color=INK2)
print("filter log10 gain range:", float(g.min()), float(g.max()), " json:", summ["filter_log10_gain_range"])
save(fig, "stroke_psd_match.png")

# ------------------------------------------------------------------ gates of the clean re-prep
col_pre = json.load(open(f"{RUNS}/phaseB/collapse_primary.json"))
col_cln = json.load(open(f"{RUNS}/phaseB_clean/collapse_primary.json"))
gate = col_cln["gate"]
bz = lambda key: np.mean([col_cln[f"c_f{k}"][key] for k in range(5)], 0)
hz = np.arange(len(bz("bnz_stroke")))                       # 1 Hz STFT bins, 0..100 Hz

fig, (c, d) = plt.subplots(1, 2, figsize=(9.4, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.32))
c.axhspan(-gate["thresholds"]["bnz_max"], gate["thresholds"]["bnz_max"], color=GRID, alpha=0.5, lw=0)
c.plot(hz, bz("bnz_stroke_prereg"), color=STROKE, lw=2, label="stroke, preregistered prep")
c.plot(hz, bz("bnz_stroke"), color=THIRD, lw=2, label="stroke, clean prep")
c.plot(hz, bz("bnz_nmt"), color=INK, lw=1.2, ls="--", label="15 NMT eval recordings")
c.text(1, gate["thresholds"]["bnz_max"] + 0.15, "gate: |z| ≤ 1.5 (grey band)", ha="left", fontsize=11, color=INK2)
c.set_xlim(0, 100); c.set_xlabel("spectrogram frequency bin (Hz)")
c.set_ylabel("z of Branch-C input vs\nits NMT BatchNorm statistics")
c.set_title("(c) Is the input in range for Branch C?", fontsize=13)
fig.legend(*c.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
           fontsize=11)

groups = [("stroke\nprereg.", [col_pre[f"c_f{k}"]["rec_std"] for k in range(5)], STROKE),
          ("stroke\nclean", [col_cln[f"c_f{k}"]["rec_std"] for k in range(5)], THIRD),
          ("NMT\neval", [col_cln[f"c_f{k}"]["nmt"]["rec_std"] for k in range(5)], INK2)]
for i, (lab, v, cc) in enumerate(groups):
    d.bar(i, np.mean(v), width=0.62, color=cc, alpha=0.85)
    d.scatter(np.full(5, i) + np.linspace(-0.16, 0.16, 5), v, s=22, color=INK, zorder=3)
    d.text(i, np.mean(v) + 0.015, f"{np.mean(v):.2f}", ha="center", va="bottom", fontsize=11.5)
d.set_xticks(range(3), [g_[0] for g_ in groups])
d.set_ylabel("spread of Branch-C features\nacross recordings (mean std)")
d.set_title("(d) Did Branch C collapse?", fontsize=13)
d.set_ylim(0, 0.47)
print("gate:", gate)
print("C rec_std prereg / clean / NMT:", [round(np.mean(g_[1]), 3) for g_ in groups])
save(fig, "stroke_clean_gates.png")
