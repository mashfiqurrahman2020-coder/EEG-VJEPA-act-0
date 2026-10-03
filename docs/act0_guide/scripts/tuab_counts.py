"""Figure tuab_counts.png: what TUAB contains, read from the real EDF headers (no signal is loaded).
Also writes scripts/tuab_headers.csv (one row per EDF) used by the facts sheet.
  CUDA_VISIBLE_DEVICES="" python tuab_counts.py
"""
import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
import csv
import glob
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuab_style import ROOT, NORMAL, ABNORMAL, INK2, W_IN, SMALL, plt, save
import numpy as np

SRC = f"{ROOT}/data/TUAB/edf"
CACHE = f"{ROOT}/docs/act0_guide/scripts/tuab_headers.csv"
KEEP = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'FZ', 'CZ', 'PZ']   # = preprocess_nmt.CHANNELS (asserted below)


def scan():
    if os.path.exists(CACHE):
        return list(csv.DictReader(open(CACHE)))
    import mne
    rows = []
    files = [(s, l, f) for s in ("train", "eval") for l in ("normal", "abnormal")
             for f in sorted(glob.glob(f"{SRC}/{s}/{l}/01_tcp_ar/*.edf"))]
    for i, (s, l, f) in enumerate(files):
        if i % 50 == 0:
            cool_gate(pause=88.0, resume=78.0, abort=92.0)
            print(f"  header {i}/{len(files)}", flush=True)
        r = mne.io.read_raw_edf(f, preload=False, verbose=False)   # header only
        rows.append(dict(split=s, label=l, file=os.path.basename(f), subject=os.path.basename(f).split("_")[0],
                         sfreq=r.info["sfreq"], n_times=r.n_times, duration_s=r.n_times / r.info["sfreq"],
                         n_ch=len(r.ch_names), has19=all(f"EEG {c}-REF" in r.ch_names for c in KEEP),
                         highpass=r.info["highpass"], lowpass=r.info["lowpass"]))
    with open(CACHE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return [{k: str(v) for k, v in r.items()} for r in rows]


def main():
    code = open(f"{ROOT}/code/preprocess_nmt.py").read()
    assert all(f"'{c}'" in code for c in KEEP)
    rows = scan()
    dur = np.array([float(r["duration_s"]) for r in rows]) / 60
    sf = Counter(float(r["sfreq"]) for r in rows)
    print("files", len(rows), "| all have the 19 channels:", all(r["has19"] == "True" for r in rows))
    print("sampling rates:", dict(sf))
    print(f"duration min: min {dur.min():.2f} median {np.median(dur):.1f} max {dur.max():.1f}; "
          f"<5 min: {(dur < 5).sum()}  <15 min: {(dur < 15).sum()}  <20 min: {(dur < 20).sum()}")
    print("n_ch:", dict(Counter(int(r["n_ch"]) for r in rows)))
    print("highpass/lowpass in header:", dict(Counter((r["highpass"], r["lowpass"]) for r in rows)))
    for s in ("train", "eval"):
        subj = {r["subject"] for r in rows if r["split"] == s}
        print(s, "subjects", len(subj), {l: sum(r["split"] == s and r["label"] == l for r in rows)
                                          for l in ("normal", "abnormal")})
    tr = {r["subject"] for r in rows if r["split"] == "train"}; ev = {r["subject"] for r in rows if r["split"] == "eval"}
    print("train/eval subject overlap:", len(tr & ev))
    for s in ("train", "eval"):
        for l in ("normal", "abnormal"):
            d = [float(r["duration_s"]) / 60 for r in rows if r["split"] == s and r["label"] == l]
            print(f"  {s}/{l}: hours {sum(d)/60:.2f}")

    fig = plt.figure(figsize=(W_IN, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.9, 0.95], hspace=0.62)
    ax = [fig.add_subplot(gs[i]) for i in range(3)]
    # (a) recordings per split x class, with number of distinct subjects
    groups = [(s, l) for s in ("train", "eval") for l in ("normal", "abnormal")]
    for y, (s, l) in enumerate(groups):
        n = sum(r["split"] == s and r["label"] == l for r in rows)
        u = len({r["subject"] for r in rows if r["split"] == s and r["label"] == l})
        ax[0].barh(y, n, 0.7, color=NORMAL if l == "normal" else ABNORMAL)
        ax[0].text(n + 30, y, f"{n} recordings, {u} subjects", va="center", fontsize=SMALL, color=INK2)
    ax[0].set_yticks(range(4), [f"{s} / {l}" for s, l in groups]); ax[0].invert_yaxis()
    ax[0].set_xlim(0, 2500); ax[0].set_xlabel("recordings (EDF files)"); ax[0].grid(axis="y", visible=False)
    ax[0].set_title("(a) Recordings per split and class")
    # (b) duration histogram
    CAP = 60   # long tail (max ~5.5 h) is folded into the last bin so the bulk stays readable
    bins = np.arange(0, CAP + 2, 1)
    for lab, col in (("normal", NORMAL), ("abnormal", ABNORMAL)):
        d = np.minimum([float(r["duration_s"]) / 60 for r in rows if r["label"] == lab], CAP + 0.5)
        ax[1].hist(d, bins=bins, color=col, label=lab, histtype="step", linewidth=2)
    ax[1].set_ylim(0, 760)
    ax[1].annotate(f"shortest file:\n{dur.min():.1f} min", xy=(dur.min(), 3), xytext=(dur.min() - 4.5, 230),
                   fontsize=SMALL, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax[1].text(CAP + 1, 150, f"{(dur > CAP).sum()} files > {CAP} min\n(max {dur.max():.0f} min)",
               fontsize=SMALL, color=INK2, ha="right")
    for x, t in ((5, "5 min kept by\nEEG-VJEPA prep"), (20, "20 min kept by\nFEI+C prep")):
        ax[1].axvline(x, color=INK2, ls="--", lw=1.2)
        ax[1].text(x + 0.7, 740, t, fontsize=SMALL, color=INK2, va="top")
    ax[1].set_xlabel("recording length (minutes)"); ax[1].set_ylabel("recordings")
    ax[1].set_title(f"(b) Recording length, all {len(rows)} files (median {np.median(dur):.1f} min)")
    ax[1].legend(loc="center right", frameon=False)
    # (c) native sampling rates
    keys = sorted(sf)
    ax[2].barh(range(len(keys)), [sf[k] for k in keys], 0.65, color=INK2)
    for y, k in enumerate(keys):
        ax[2].text(sf[k] * 1.12, y, f"{sf[k]} recordings", va="center", fontsize=SMALL, color=INK2)
    ax[2].set_yticks(range(len(keys)), [f"{k:g} Hz" for k in keys]); ax[2].invert_yaxis()
    ax[2].set_xscale("log"); ax[2].set_xlim(8, 3e4); ax[2].grid(axis="y", visible=False)
    ax[2].set_xlabel("recordings (log scale)"); ax[2].set_title("(c) Native sampling rate (samples per second)")
    save(fig, "tuab_counts.png")


if __name__ == "__main__":
    main()
