# NMT Data Pipeline

The **NMT** dataset (NMT Scalp EEG, Pakistan) is a large collection of clinical EEG recordings, each labelled *normal* or *abnormal*. In Act-0 it is used to **pre-train our own FEI and Branch C encoders** (see "FEI+C Architecture"). EEG-VJEPA was not pre-trained on NMT by us.

## Dataset Overview

- **Total Recordings:** 2,417 recordings
- **Split:** train has 1,907 normal + 325 abnormal recordings; eval has 95 normal + 90 abnormal.
- **Length:** about 12 minutes per recording (median 11.7 min); ~489 hours in total.
- **Sampling Rate:** **200 Hz**
- **Channels:** The raw files contain 21 channels (e.g., `FP1|FP2|F3|...|A1|A2`). We drop the two ear references (A1, A2) and keep the **19 standard 10-20 scalp channels**.

![NMT Counts](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_counts.png)

## Preprocessing

NMT preprocessing is deliberately light. We use the dataset's own `NMT_preprocessed` files, where each channel is only **z-scored** (mean 0, spread 1). **No band-pass filter** is applied, so noise and artifacts are still present. The z-score only puts all channels and recordings on the same amplitude scale.

In code, `DATA` points at these files. `load_continuous` joins a recording's stored (frames, 19, 500) array back into one continuous (19, samples) signal:

{{code:fei_pretrain.py:36-37,50-54}}


![NMT Raw vs Prep](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_raw_vs_prep.png)

The **PSD** (power spectral density) plot shows how much power the signal has at each frequency. Because no filter is used, the NMT spectrum keeps its full shape. This spectrum matters later: FEI and Branch C learned their statistics from it, so any new dataset fed to them must have a similar spectrum (see "Stroke", PSD matching).

![NMT PSD](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_psd.png)

## Windowing and Folds

Instead of feeding a whole 12-minute recording into the model at once, we cut it into **20 s windows** (19 channels × 4,000 samples):

- **Pre-training:** random 20 s windows are drawn from the training recordings.
- **Embedding a recording:** it is cut into non-overlapping 20 s windows; each window is encoded and the embeddings are averaged into one vector.

Pre-training reads one random window per item. `load_window` memory-maps the file and reads only the 2–3 stored frames that the window spans:

{{code:fei_pretrain.py::load_window}}

Embedding a recording cuts it into non-overlapping windows and averages the encoder outputs:

{{code:fei_pretrain.py::embed_recording}}


![NMT Windows](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_windows.png)

To test on unseen recordings, we use **5-fold cross-validation (CV)**. The train and eval recordings are pooled (2,417) and split into 5 folds with `StratifiedKFold` (seed 0), so each fold keeps the same normal/abnormal ratio. In fold *k*, the fold-*k* recordings (about 484) are the test set and the other ~1,933 are for training. For fold *k* we pre-trained a separate encoder (f0–f4) that **never saw the fold-*k* recordings**. These five fold encoders are the ones later ensembled on TUAB and stroke.

The fold loop in `run_cv`. `pretrain` receives only the training folds' files, and each fold's encoder is saved as `..._cv_s0_f{k}.pt`:

{{code:fei_pretrain.py:270-279}}


![NMT CV Folds](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_cv_folds.png)
