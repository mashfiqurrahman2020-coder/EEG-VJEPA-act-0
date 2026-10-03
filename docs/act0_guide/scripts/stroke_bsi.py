import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure C: the Brain Symmetry Index (BSI) pipeline on real data.
(a) one typical stroke subject (PAC09, BSI = cohort median) and one typical control (CONTROL07): power spectra of
    the left/right pair C3/C4, recomputed with the exact steps of stroke_phaseB.gate_feats (notch 50, 0.5-100 Hz,
    2 s epochs, +-100 uV reject, Welch 1 s) -- the script asserts its pdBSI equals the stored gate_primary.json value.
(b) broadband pdBSI (1-30 Hz) for every subject in both cohorts, from gate_{primary,sensitivity}.json.
(c) the 5 BSI numbers that form the BSI classifier's input (stroke_phaseB.BSI_KEYS), primary cohort."""
import json
import os
os.environ.setdefault("HW_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stroke_style import plt, save, RUNS, INK, INK2, GRID, STROKE, CONTROL, THIRD, VIOLET, GATE
import numpy as np
from scipy.signal import welch
import stroke_phaseB as B

G = {c: json.load(open(f"{RUNS}/phaseB/gate_{c}.json")) for c in ("primary", "sensitivity")}
EXAMPLES = [("PAC09", "stroke patient PAC09", STROKE), ("CONTROL07", "healthy control CONTROL07", CONTROL)]


def pair_spectra(subj):
    """gate_feats() steps, but keep the per-channel spectra (21, F)."""
    raw, _ = B.load(subj, B.COHORTS["primary"][1])
    r = raw.copy().notch_filter(50, verbose=False).filter(0.5, 100, verbose=False)
    fs = int(r.info["sfreq"])
    x = r.get_data(picks=B.CENTRAL21) * 1e6
    ep = x[:, :x.shape[1] // (2 * fs) * 2 * fs].reshape(21, -1, 2 * fs).transpose(1, 0, 2)
    keep = np.abs(ep).max((1, 2)) <= 100
    f, p = welch(ep[keep], fs=fs, nperseg=fs, axis=-1)
    p = p.mean(0)
    ch = {c: p[i] for i, c in enumerate(B.CENTRAL21)}
    sel = (f >= 1) & (f < 30)
    bsi = B.pdbsi(np.stack([ch[c][sel] for c in B.AH]), np.stack([ch[c][sel] for c in B.UH]))
    stored = G["primary"]["subjects"][subj]["bsi_1_30"]
    print(f"{subj}: kept {keep.sum()}/{len(keep)} epochs, pdBSI recomputed {bsi:.5f} stored {stored:.5f}")
    assert abs(bsi - stored) < 1e-6
    return f[sel], ch["C3"][sel], ch["C4"][sel], bsi


fig = plt.figure(figsize=(9.4, 9.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.55, wspace=0.28)
for k, (subj, title, col) in enumerate(EXAMPLES):
    GATE()
    f, L, R, bsi = pair_spectra(subj)
    a = fig.add_subplot(gs[0, k])
    a.fill_between(f, L, R, color=GRID, alpha=0.8, lw=0)
    a.semilogy(f, L, color=VIOLET, lw=2, label="C3 (left)")
    a.semilogy(f, R, color=THIRD, lw=2, label="C4 (right)")
    a.set_xlim(1, 30); a.set_xlabel("frequency (Hz)")
    if k == 0:
        a.set_ylabel("power (µV²/Hz, log scale)")
    a.set_title(f"({'ab'[k]}) {title}", fontsize=12.5, color=col)
    a.text(0.97, 0.95, f"pdBSI = {bsi:.3f}\n(all 6 pairs, 1–30 Hz)", transform=a.transAxes, ha="right", va="top",
           fontsize=11, color=INK)
    a.legend(loc="lower left", frameon=False, fontsize=11)

# (c) broadband pdBSI per subject, both cohorts
b = fig.add_subplot(gs[1, 0])
rng = np.random.default_rng(0)
xt, xl = [], []
for i, coh in enumerate(("primary", "sensitivity")):
    subs = G[coh]["subjects"]
    for j, (grp, col) in enumerate([(1, STROKE), (0, CONTROL)]):
        v = np.array([r["bsi_1_30"] for r in subs.values() if r["y"] == grp])
        x0 = i * 2.6 + j
        b.scatter(x0 + rng.uniform(-0.18, 0.18, len(v)), v, s=34, color=col, edgecolor="white", lw=0.6, zorder=3)
        b.plot([x0 - 0.3, x0 + 0.3], [np.median(v)] * 2, color=INK, lw=2)
        xt.append(x0); xl.append(f"{'stroke' if grp else 'control'}\n(n={len(v)})")
    p = G[coh]["tests"]["bsi_1_30"]["p"]
    b.text(i * 2.6 + 0.5, 0.215, f"{coh}\np = {p:.3f}" if p >= 0.001 else f"{coh}\np < 0.001", ha="center",
           va="bottom", fontsize=11)
b.set_xticks(xt, xl, fontsize=10.5)
b.set_ylim(0, 0.26); b.set_xlim(-0.6, 4.2)
b.set_ylabel("broadband pdBSI (1–30 Hz)")
b.set_title("(c) Asymmetry, every subject", fontsize=12.5)

# (d) the 5 classifier features, primary cohort
c = fig.add_subplot(gs[1, 1])
subs = G["primary"]["subjects"]
for i, key in enumerate(B.BSI_KEYS):
    for j, (grp, col) in enumerate([(1, STROKE), (0, CONTROL)]):
        v = np.array([r[key] for r in subs.values() if r["y"] == grp])
        x0 = i + (j - 0.5) * 0.38
        c.scatter(x0 + rng.uniform(-0.07, 0.07, len(v)), v, s=18, color=col, edgecolor="white", lw=0.4, zorder=3)
        c.plot([x0 - 0.13, x0 + 0.13], [np.median(v)] * 2, color=INK, lw=1.6)
    p = G["primary"]["tests"][key]["p"]
    c.text(i, 0.345, f"p={p:.3f}", ha="center", fontsize=9.5, color=INK2)
c.set_xticks(range(5), ["1–30 Hz", "delta", "theta", "alpha", "beta"], fontsize=10.5)
c.set_ylim(0, 0.37)
c.set_ylabel("pdBSI in that band")
c.set_title("(d) The 5 BSI classifier inputs (primary)", fontsize=12.5)
fig.legend([plt.Line2D([], [], marker="o", ls="", ms=8, color=STROKE),
            plt.Line2D([], [], marker="o", ls="", ms=8, color=CONTROL), plt.Line2D([], [], color=INK, lw=2)],
           ["stroke", "control", "group median"], loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3,
           frameon=False, fontsize=11)
save(fig, "stroke_bsi.png")
