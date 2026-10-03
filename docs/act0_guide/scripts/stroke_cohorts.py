import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure C: who is in each Phase-B cohort, and how long each subject's pure-rest block is.
Rest block = from file start to the first segment break ('New Segment') or first task marker (any Stimulus
except the S255 start code), read from the real BrainVision .vmrk marker files. Cohorts = stroke_phaseB.COHORTS."""
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stroke_style import plt, save, RAW, STROKE, CONTROL, INK2, GATE

sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code")
os.environ.setdefault("HW_THREADS", "1")
import stroke_phaseB as B   # read-only: cohort lists

PRIMARY, SENS = B.COHORTS["primary"][0], B.COHORTS["sensitivity"][0]
subs = [s for s in SENS if s.startswith("PAC")] + [s for s in SENS if not s.startswith("PAC")]


def markers(s):
    mk = [l.split("=", 1)[1].split(",") for l in open(f"{RAW}/{s}.vmrk", encoding="utf-8", errors="ignore")
          if l.startswith("Mk")]
    return [(m[0], m[1].strip(), int(m[2])) for m in mk]


rows = []
for s in subs:
    GATE()
    fs = 1000.0                                            # SamplingInterval=1000 us in every .vhdr
    n = os.path.getsize(f"{RAW}/{s}.eeg") // (63 * 2)      # 63 ch x INT_16, multiplexed
    brk = [p for t, d, p in markers(s) if p > 1 and (t == "New Segment" or (t == "Stimulus" and d != "S255"))]
    rest = (brk[0] - 1) / fs if brk else n / fs
    rows.append((s, rest, n / fs))
    print(f"{s:10s} rest {rest:6.1f} s   total {n / fs:7.1f} s  primary={s in PRIMARY}")
n_post = len([f for f in os.listdir(RAW) if f.endswith("-POST.vhdr")])

fig, ax = plt.subplots(figsize=(9.4, 7.2))
for i, (s, rest, tot) in enumerate(rows):
    col = STROKE if s.startswith("PAC") else CONTROL
    prim = s in PRIMARY
    ax.barh(i, rest, height=0.72, color=col if prim else "white", edgecolor=col, linewidth=1.6,
            hatch=None if prim else "///")
    ax.text(max(rest, 300) + 12, i, f"{rest:.0f} s rest  (whole file {tot / 60:.0f} min)", va="center", fontsize=11.5,
            color=INK2)
ax.axvline(285, color="#0b0b0b", lw=1.6, ls="-")
ax.axvline(300, color="#0b0b0b", lw=1.6, ls="--")
ax.text(283, -1.25, "primary: first 285 s", ha="right", va="center", fontsize=12)
ax.text(302, -1.25, "sensitivity: first 300 s", ha="left", va="center", fontsize=12)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.set_ylim(len(rows) - 0.4, -1.8)
ax.set_xlim(0, 560)
ax.set_xlabel("length of the pure-rest block at the start of the file (seconds)")
ax.set_title("Phase B subjects: 10 stroke (PAC) + 8 controls, PRE files only")
h = [plt.Rectangle((0, 0), 1, 1, fc=STROKE, ec=STROKE), plt.Rectangle((0, 0), 1, 1, fc=CONTROL, ec=CONTROL),
     plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK2, hatch="///")]
ax.legend(h, ["stroke, in primary cohort", "control, in primary cohort", "sensitivity cohort only (short rest)"],
          loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, frameon=False)
print("POST files in folder (never used):", n_post)
save(fig, "stroke_cohorts.png")
