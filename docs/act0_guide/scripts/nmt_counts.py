import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure A: NMT counts per class x split + recording-duration histogram (from EDF headers).
Also a census of every EDF header (sampling rate, channels, prefilter) and a check that each stored
.npy has exactly floor(T/500) frames. Only headers are read (a few KB per file), never the signals.
Writes figs/nmt_counts.png and facts/nmt_census.json."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
from nmt_style import *   # noqa: F401,F403  (paths, palette, style, cool, save)
import glob, json
from collections import Counter
import numpy as np
import pandas as pd


def edf_header(path):
    """Minimal EDF header parse -> (labels, n_records, record_s, nsamp per record, prefilters)."""
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256])
        hh = f.read(256 * ns)
    n_rec, rec_s = int(h[236:244]), float(h[244:252])
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]   # label, transducer, dim, pmin, pmax, dmin, dmax, prefilt, nsamp, res
    off, fields = 0, []
    for w in widths:
        fields.append([hh[off + i * w: off + (i + 1) * w].decode("latin1").strip() for i in range(ns)])
        off += w * ns
    return fields[0], n_rec, rec_s, [int(v) for v in fields[8]], fields[7], fields[2]


df = pd.read_csv(f"{RAW}/Labels.csv")
rows = []
for i, r in enumerate(df.itertuples()):
    if i % 100 == 0:
        cool()
    p = f"{RAW}/{r.label}/{r.loc}/{r.recordname}"
    labels, n_rec, rec_s, nsamp, pref, dims = edf_header(p)
    fs = {n / rec_s for n in nsamp}
    T = n_rec * nsamp[0]                                   # samples per channel
    npy = f"{PREP}/{r.loc}/{r.label}/{r.recordname[:-4]}.npy"
    shp = np.load(npy, mmap_mode="r").shape               # reads the .npy header only
    rows.append(dict(rid=r.recordname, label=r.label, split=r.loc, n_ch=len(labels), labels="|".join(labels),
                     fs="|".join(str(v) for v in sorted(fs)), T=T, dur_s=T / FS, npy_F=shp[0],
                     npy_shape=str(shp), pref=Counter(pref).most_common(1)[0][0], dim="|".join(sorted(set(dims)))))
d = pd.DataFrame(rows)

census = {
    "n_recordings": len(d),
    "counts_label_split": {f"{l}/{s}": int(((d.label == l) & (d.split == s)).sum())
                           for l in ("normal", "abnormal") for s in ("train", "eval")},
    "sampling_rates_found": sorted(d.fs.unique().tolist()),
    "n_channels_found": sorted(d.n_ch.unique().tolist()),
    "channel_lists_found": d.labels.value_counts().to_dict(),
    "prefilter_values_found": d.pref.value_counts().to_dict(),
    "physical_dimension_found": d.dim.value_counts().to_dict(),
    "all_npy_frames_equal_floor_T_over_500": bool((d.npy_F == d["T"] // 500).all()),
    "all_npy_shapes_are_F_19_500": bool(d.npy_shape.str.endswith("19, 500)").all()),
    "duration_s": {k: float(v) for k, v in d.dur_s.describe(percentiles=[.1, .25, .5, .75, .9]).items()},
    "duration_s_median_by_label": d.groupby("label").dur_s.median().to_dict(),
    "n_shorter_than_20s": int((d.dur_s < 20).sum()),
    "n_shorter_than_5s": int((d.dur_s < 5).sum()),
    "n_shorter_than_300s": int((d.dur_s < 300).sum()),
    "shortest": d.nsmallest(3, "dur_s")[["rid", "label", "split", "dur_s"]].to_dict("records"),
    "longest": d.nlargest(3, "dur_s")[["rid", "label", "split", "dur_s"]].to_dict("records"),
    "total_hours": float(d.dur_s.sum() / 3600),
    "age_median_by_label": df.groupby("label").age.median().to_dict(),
    "gender_counts": df.gender.value_counts().to_dict(),
}
json.dump(census, open(f"{FACTS}/nmt_census.json", "w"), indent=2)
print(json.dumps({k: v for k, v in census.items() if k != "channel_lists_found"}, indent=1))

# ---------------- figure ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.32))
splits = ["train", "eval"]
x = np.arange(2); w = 0.36
for j, (lab, col) in enumerate((("normal", NORMAL), ("abnormal", ABNORMAL))):
    v = [census["counts_label_split"][f"{lab}/{s}"] for s in splits]
    b = a1.bar(x + (j - 0.5) * w, v, w * 0.92, color=col, label=lab, zorder=3)
    for xi, vi in zip(x + (j - 0.5) * w, v):
        a1.text(xi, vi + 25, str(vi), ha="center", va="bottom", fontsize=11, color=INK)
a1.set_xticks(x, [f"train\n({(d.split == 'train').sum()} recs)", f"eval\n({(d.split == 'eval').sum()} recs)"])
a1.set_ylabel("number of recordings")
a1.set_ylim(0, 2150)
a1.set_title("(a) Recordings per class and split", loc="left")
a1.legend(frameon=False, loc="upper right")
a1.grid(axis="x", visible=False)

bins = np.logspace(np.log10(3), np.log10(d.dur_s.max() * 1.01), 40)
a2.hist([d.dur_s[d.label == "normal"], d.dur_s[d.label == "abnormal"]], bins=bins, stacked=True,
        color=[NORMAL, ABNORMAL], label=["normal", "abnormal"], zorder=3, edgecolor="white", linewidth=0.6)
a2.set_xscale("log")
a2.minorticks_off()
a2.set_xticks([5, 20, 120, 600, 3600], ["5 s", "20 s", "2 min", "10 min", "1 h"])
med = census["duration_s"]["50%"]
a2.axvline(med, color=INK, lw=1.2, ls="--", zorder=4)
ymax = np.histogram(d.dur_s, bins=bins)[0].max()
a2.set_ylim(0, ymax * 1.12)
a2.annotate(f"median\n{med/60:.1f} min", xy=(med, ymax * 0.8), xytext=(med * 2.6, ymax * 0.8), fontsize=11,
            va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
a2.axvline(20, color=INK2, lw=1, ls=":", zorder=4)
a2.text(24, ymax * 0.55, f"20 s = one\nmodel window\n({census['n_shorter_than_20s']} recordings\nare shorter)",
        fontsize=10.5, color=INK2, va="top")
a2.set_xlabel("recording length (log scale)")
a2.set_ylabel("number of recordings")
a2.set_title(f"(b) Recording length (all {len(d)}, stacked)", loc="left")
a2.legend(frameon=False, loc="upper left")
save(fig, "nmt_counts.png")
