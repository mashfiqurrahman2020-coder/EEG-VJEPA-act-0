"""P1.3 diagnostic (the gate failed at AUROC 0.757): is our §4.1 TUAB prep losing information, or is
0.80 simply too high a bar for relative band power + LogReg? Same features/classifier on the
independent CBraMod prep (16 bipolar ch, 200 Hz, 0.3-75 Hz, uV, 10 s windows; passed its own gate):
  B = first 5 min, relative BP        (duration-matched to §4.1)
  C = up to 20 min, relative BP       (duration effect)
  D = up to 20 min, relative + log-absolute BP   (amplitude info, which §4.1's z-score removes)
Train = CBraMod train+val (= all TUAB train), test = official eval.
  python tuab_bandpower_diag.py -> code/runs/act0/p13_bandpower/diag.json
"""
import glob
import os
import pickle
import re
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads, cool_gate
cap_threads(4)
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from act0_metrics import fmt, metrics, save

PR = "/home/mashfiq/eeg_vjepa/data/TUAB/process_refine"
PREP = "/home/mashfiq/eeg_vjepa/data/TUAB_preprocessed_100hz"
OUT = "/home/mashfiq/eeg_vjepa/code/runs/act0/p13_bandpower"
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
MAXW = 120   # 20 min cap bounds the 99 GB read


def recs(splits):
    g = defaultdict(list)
    for s in splits:
        for f in os.listdir(f"{PR}/{s}"):
            r, i = re.match(r"(.+)_(\d+)\.pkl$", f).groups()
            g[r].append((int(i), f"{PR}/{s}/{f}"))
    return {r: [p for _, p in sorted(v)[:MAXW]] for r, v in sorted(g.items())}


def feats(groups, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); return d["P5"], d["P20"], d["y"], list(d["names"])
    P5, P20, y = [], [], []
    for k, (r, paths) in enumerate(groups.items()):
        if k % 50 == 0:
            cool_gate(pause=88.0, resume=78.0, abort=92.0)
            print(f"  {k}/{len(groups)}", flush=True)
        ps, lab = [], None
        for p in paths:
            d = pickle.load(open(p, "rb")); lab = d["y"]
            f, pxx = welch(d["X"], fs=200, nperseg=500, axis=-1); ps.append(pxx)
        ps = np.array(ps)
        bp = lambda P: np.stack([P[:, (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], -1)
        P5.append(bp(ps[:30].mean(0))); P20.append(bp(ps.mean(0))); y.append(lab)
    P5, P20, y = np.array(P5), np.array(P20), np.array(y)
    np.savez(cache, P5=P5, P20=P20, y=y, names=np.array(list(groups)))
    return P5, P20, y, list(groups)


rel = lambda P: (P / P.sum(-1, keepdims=True)).reshape(len(P), -1)
logabs = lambda P: np.log(P + 1e-12).reshape(len(P), -1)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr5, tr20, ytr, ntr = feats(recs(["train", "val"]), f"{OUT}/diag_train.npz")
    ev5, ev20, yev, nev = feats(recs(["test"]), f"{OUT}/diag_eval.npz")
    # label sanity vs our dirs (CBraMod y must be abnormal=1)
    ab = {os.path.basename(f)[:-4] for f in glob.glob(f"{PREP}/eval/abnormal/*.npy")}
    assert all((y == 1) == (n in ab) for n, y in zip(nev, yev)), "label convention mismatch"
    print(f"train recs {len(ytr)} ({ytr.sum()} abn), eval recs {len(yev)} ({yev.sum()} abn)")
    res = {}
    for name, Xtr, Xev in [("B_rel_5min", rel(tr5), rel(ev5)), ("C_rel_20min", rel(tr20), rel(ev20)),
                           ("D_rel+logabs_20min", np.hstack([rel(tr20), logabs(tr20)]),
                            np.hstack([rel(ev20), logabs(ev20)]))]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr, ytr)
        res[name] = metrics(yev, clf.predict_proba(Xev)[:, 1])
        print(f"{name:20s} {fmt(res[name])}", flush=True)
    save(f"{OUT}/diag.json", res)


if __name__ == "__main__":
    main()
