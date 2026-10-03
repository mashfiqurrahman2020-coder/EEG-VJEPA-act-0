import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure A: the REAL seed-0 5-fold split of NMT used by FEI, Branch C and the Act-0 A0 anchor.
Reads the test lists saved by code/feic_nmt_anchor.py (code/runs/act0/a0_feic_nmt/emb_f{k}.npz: rec_te)
and checks them against StratifiedKFold(5, shuffle=True, random_state=0) over the list_split order
(train then eval; normal then abnormal) used by fei_pretrain.run_cv / fei_branchC.cv.
Writes figs/nmt_cv_folds.png and facts/nmt_cv_folds.json."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
from nmt_style import *   # noqa: F401,F403
import glob, json
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.model_selection import StratifiedKFold

A0 = f"{ROOT}/code/runs/act0/a0_feic_nmt"


def list_split(split):                      # verbatim logic of code/fei_pretrain.py list_split
    return [(p, c) for lab, c in (("normal", 0), ("abnormal", 1))
            for p in sorted(glob.glob(f"{PREP}/{split}/{lab}/*.npy"))]


files = list_split("train") + list_split("eval")
y = np.array([c for _, c in files]); paths = [p for p, _ in files]
skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
fold_of = np.full(len(files), -1)
res = json.load(open(f"{A0}/results.json"))
rows = []
for k in range(5):
    cool()
    d = np.load(f"{A0}/emb_f{k}.npz")
    saved_te, saved_tr = set(d["rec_te"].tolist()), set(d["rec_tr"].tolist())
    te = skf[k][1]
    assert saved_te == {paths[i] for i in te}, f"fold {k}: saved test list != StratifiedKFold(seed 0)"
    assert not (saved_te & saved_tr) and len(saved_te | saved_tr) == len(files)
    fold_of[te] = k
    yt = y[te]
    rows.append(dict(fold=k, n_test=len(te), n_test_normal=int((yt == 0).sum()), n_test_abnormal=int(yt.sum()),
                     pct_abnormal=round(100 * yt.mean(), 2), n_train=len(skf[k][0]),
                     n_test_from_eval_split=int(sum("/eval/" in paths[i] for i in te)),
                     auroc_FEI4000=res[str(k)]["FEI@4000"]["auroc"], auroc_C=res[str(k)]["C"]["auroc"],
                     auroc_FEIplusC=res[str(k)]["FEI+C"]["auroc"],
                     encoder_ckpts=[f"code/fei_enc_cv_s0_f{k}.pt", f"code/branchC_h20_enc_cv_s0_f{k}.pt"]))
assert (fold_of >= 0).all()
facts = dict(n_recordings=len(files), pct_abnormal_overall=round(100 * y.mean(), 2),
             saved_test_lists_match_StratifiedKFold_seed0=True, folds_disjoint_and_cover_all=True,
             first40_fold_ids=fold_of[:40].tolist(), folds=rows, summary=res["summary"])
json.dump(facts, open(f"{FACTS}/nmt_cv_folds.json", "w"), indent=2)
for r in rows:
    print({k: v for k, v in r.items() if k != "encoder_ckpts"})

# ---------------- figure ----------------
fig = plt.figure(figsize=(7.6, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.16, 1], width_ratios=[1, 1.05], hspace=0.45, wspace=0.42)
a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])

# (a) first 40 recordings of the list, with the test fold each one landed in (digits, not colours)
n = 40
for i in range(n):
    a0.add_patch(Rectangle((i, 0), 0.92, 1, facecolor=GRID, edgecolor="none"))
    a0.text(i + 0.46, 0.5, str(fold_of[i]), ha="center", va="center", fontsize=10.5)
a0.set_xlim(-0.2, n); a0.set_ylim(0, 1); a0.axis("off")
a0.set_title("(a) First 40 recordings in the list and the part (0-4) each was put in, at random",
             loc="left", fontsize=12)

# (b) which part is held out for which encoder
for r in range(5):
    for c in range(5):
        test = r == c
        a1.add_patch(Rectangle((c, 4 - r), 0.94, 0.94, facecolor=INK2 if test else GRID, edgecolor="none"))
        a1.text(c + 0.47, 4 - r + 0.47, "test" if test else "train", ha="center", va="center",
                fontsize=10, color="white" if test else INK)
a1.set_xlim(0, 5); a1.set_ylim(0, 5)
a1.set_xticks(np.arange(5) + 0.47, [f"part {c}" for c in range(5)], fontsize=10, rotation=45, ha="right")
a1.set_yticks(np.arange(5) + 0.47, [f"f{4 - r}" for r in range(5)], fontsize=10.5)
a1.set_ylabel("fold encoder")
a1.tick_params(length=0); a1.grid(False)
for s in a1.spines.values():
    s.set_visible(False)
a1.set_title("(b) Encoder fk never sees part k", loc="left", fontsize=12)

# (c) test-part composition: stratified -> same abnormal share everywhere
yy = np.arange(5)[::-1]
nn = [r["n_test_normal"] for r in rows]; na = [r["n_test_abnormal"] for r in rows]
a2.barh(yy, nn, color=NORMAL, height=0.62, label="normal", zorder=3)
a2.barh(yy, na, left=nn, color=ABNORMAL, height=0.62, label="abnormal", zorder=3)
for yv, r in zip(yy, rows):
    a2.text(12, yv, f"{r['n_test_normal']} normal", va="center", fontsize=10, color="white", zorder=4)
    a2.text(r["n_test"] + 8, yv, f"+ {r['n_test_abnormal']} abnormal\n({r['pct_abnormal']:.1f}%)", va="center",
            fontsize=10)
a2.set_yticks(yy, [f"part {r['fold']}" for r in rows], fontsize=10.5)
a2.set_xlim(0, 680); a2.set_xlabel("recordings in the test part")
a2.grid(axis="y", visible=False)
a2.set_title("(c) What is inside each test part", loc="left", fontsize=12)
save(fig, "nmt_cv_folds.png")
