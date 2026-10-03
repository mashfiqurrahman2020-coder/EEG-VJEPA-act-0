# EEG-VJEPA Act 0: reproduction and FEI+GTJ

Code, checkpoints and result records for the EEE 402 (BUET, Section G2, Group 01) Act-0 report,
[docs/Act0_Final_Report.pdf](docs/Act0_Final_Report.pdf). The `.docx` beside it is the source of the report.
§7.2 of the report explains how to reproduce the results, and Appendix §11 lists the records kept for that.

## Layout

| Path | Contents |
|---|---|
| `code/*.py` | One script per experiment (report Table 5) and per input (Table 4), plus the modules they import |
| `code/app`, `code/src`, `code/evals`, `code/configs`, `code/setup.py` | Our copy of the EEG-VJEPA code at upstream commit `c739fad5ad57ec4439a094343901a245edf611ec`. Five files carry labelled additions, listed in Experiment 1 of the report. The CC BY-NC 4.0 licence is in `code/LICENSE` and the upstream README in `code/UPSTREAM_README.md` |
| `code/*.pt` | Our 40 FEI/GTJ encoder checkpoints: 5 folds × {original, rep1, rep2, joint NMT+TUAB} × {FEI, GTJ} |
| `code/runs/act0/` | Saved results (JSON), per-recording predictions (`preds_*.npz`, `oof_*.npz`, `kf5_lpo_*.npz`), run logs, the stroke `PREREGISTRATION.md` and `POSTHOC.md` |
| `docs/act0_repro/` | Fold, subset and cohort lists; SHA-256 of every checkpoint and result file; pinned packages and environment |
| `docs/act-0_report.md`, `docs/act0_results/`, `docs/act0_guide/` | Working notes, result analysis and the architecture guide behind the report |

Check the checkpoints and results against the manifests:

```bash
tr -d '\r' < docs/act0_repro/checkpoints_sha256.csv | awk -F, 'NR>1 && $1!~/^pretrained/{print $3"  "$1}' | sha256sum -c --quiet
tr -d '\r' < docs/act0_repro/results_sha256.csv | awk -F, 'NR>1{print $3"  "$1}' | sha256sum -c --quiet
```

Check your copy of the recordings against `docs/act0_repro/raw_recordings_sha256.csv` (run from the folder that holds
`data/`; folders as in `code/act0_raw_hashes.py`):

```bash
tr -d '\r' < docs/act0_repro/raw_recordings_sha256.csv | awk -F, 'NR>1{d=($1=="TUAB")?"data/TUAB/edf":($1=="NMT")?"data/NMT-Scalp-EEG":"data/zenodo_stroke"; print $4"  "d"/"$2}' | sha256sum -c --quiet
```

## Not included

- **Recordings.** NMT and the stroke cohort are public. TUAB must be requested from its maintainers under their
  data use agreement (report §5.5.3). Our TUAB copy came from an unofficial mirror, so the TUAB numbers here are for
  this course submission only.
- **The released EEG-VJEPA weights.** Download them from Hugging Face `amir-hlp/EEG-VJEPA` at commit
  `ec98405f5f3423fd59d3296157ee1570efd8d947` into `pretrained/`. Their SHA-256 values are in
  `docs/act0_repro/checkpoints_sha256.csv`.

- **Saved embeddings** (`emb_*.npz`, about 446 MB). Each is a forward pass of a checkpoint here or of a seeded
  random-init twin, so the scripts regenerate them.

## Running

- **Version used for the report:** the code, checkpoints and results are the commit the report's §7.2 names by its
  full SHA; the report itself is under tag `act0-report-round15b`. The earlier tags `act0-report-round14` and `act0-report-round15` are kept.
- **Reproducibility:** the saved checkpoints and predictions reproduce every reported number exactly. Re-training
  reproduces them only statistically: the original FEI/GTJ runs set no seed, and cuDNN was left non-deterministic.

- **Environment:** Python 3.9 with `docs/act0_repro/requirements-act0.txt`. `docs/act0_repro/environment.txt`
  records the original machine.
- **Paths:** the scripts expect this checkout at `/home/mashfiq/eeg_vjepa` (the `ROOT` constant at the top of each
  script), with the data under `data/` and the weights under `pretrained/`. Change `ROOT` to run from anywhere else.
- **Hardware guard:** every long run goes through `code/hw_guard.py`, which pauses the run when the CPU gets hot.
- **Resuming:** long runs checkpoint after each fold, so an interrupted run resumes where it stopped.
