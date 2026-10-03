# Stroke Data Pipeline

The stroke evaluation (**Part 2 / Phase B**) asks one question: can frozen EEG features tell **stroke patients** from **healthy controls**?

- **Data:** Zenodo record 19599466, resting-state EEG in BrainVision format.
- **Recording:** 63 channels, 1,000 Hz, referenced to the right ear (A2).
- **Sessions:** only the **PRE** (before therapy) files are used, so every subject contributes one recording.

## Cohorts

- **Primary cohort:** 9 patients (PAC02–PAC10) vs 6 controls (C03, CONTROL04–08). Every subject gives the same fixed **285 s** of rest, so no class has more data than the other.
- **Sensitivity cohort:** 10 patients vs 8 controls, using the first 300 s. It checks that the result does not depend on which subjects were included.

With so few subjects, every result is a small-sample result.

![Stroke Cohorts](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_cohorts.png)

## Preprocessing and PSD Matching

Channels that the dataset authors marked as bad are first rebuilt from their neighbours (**spherical-spline interpolation**). After that, the same recording is prepared in **three different ways**, one for each model:

`load` (in `stroke_phaseB.py`) reads the BrainVision file, crops the rest period, marks the authors' bad channels and interpolates them:

{{code:stroke_phaseB.py::load}}


| Prep | Used by | Steps |
|---|---|---|
| ViT-M prep | EEG-VJEPA | average reference, 19 channels (T7/T8/P7/P8 renamed T3/T4/T5/T6), 1–40 Hz, 100 Hz, z-score, 5 s frames every 2.5 s (same as TUAB) |
| FEI+C prep (pre-registered) | FEI+C | 0.5–40 Hz, 200 Hz, 19 channels, z-score |
| Clean FEI+C prep (post-hoc) | FEI+C only | average reference, 19 channels, 200 Hz, **no** band-pass, PSD matching to NMT (below), z-score, 2.5 s frames |

The BSI features (below) use their own prep on the original 63 channels.

The two pre-registered encoder preps come from one function. The loop runs over the `vitm` and `feic` settings (low cut-off, sampling rate, frame stride):

{{code:stroke_phaseB.py::encoder_preps}}


![Stroke Raw vs Prep](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_raw_vs_prep.png)

**PSD matching** is used **only in the clean FEI+C prep**. It was added after the pre-registered run, for this reason: after the pre-registered prep, the stroke spectrum was still far from NMT's. Branch C's BatchNorm then received inputs far outside the range it had learned on NMT, and Branch C collapsed (almost the same output for every subject). The clean prep builds **one fixed zero-phase filter per cohort** (per channel). It maps the cohort's average log-spectrum (all subjects pooled, labels never used) onto NMT's average log-spectrum. Zero-phase means the filter changes power at each frequency but does not shift the waveform in time. Because this step was decided after seeing results, clean-prep numbers are reported as post-hoc.

The clean prep is in `stroke_phaseB_clean.py`. `logpsd` is the Welch log-spectrum; `shape` multiplies each channel's FFT by a real, positive gain, which changes power but not timing (zero-phase):

{{code:stroke_phaseB_clean.py::logpsd}}

{{code:stroke_phaseB_clean.py::shape}}

The gain `logh` is found by iteration (`N_ITER` = 4). Each round moves the cohort's mean log-spectrum half-way towards NMT's (`ref`). All subjects are pooled and the labels are never read:

{{code:stroke_phaseB_clean.py:95-99}}


![Stroke PSD Match](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_psd_match.png)

The **gates** are checks run before any classifier is trained. They confirm that the clean prep worked:

- the largest gap between the stroke and NMT log-spectra fell from 5.81 to 0.18 decades (gate: ≤ 0.25);
- Branch C's BatchNorm inputs fell from |z| ≈ 4.3 to ≈ 0.37;
- no embedding dimension is constant across subjects (no collapse).

PSD matching does **not** remove artifacts; it only changes the overall spectral shape.

![Stroke Clean Gates](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_clean_gates.png)

## Montage and BSI

The recordings have **63 electrodes**. The encoders use a **19-channel** 10-20 subset, the same channels as NMT and TUAB. The BSI uses **21 central channels** (FC, C and CP rows) and forms **6 left/right pairs**: FC1/FC2, FC3/FC4, C1/C2, C3/C4, C5/C6 and CP1/CP2.

![Stroke Montage](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_montage.png)

The **Brain Symmetry Index (BSI)** is a hand-crafted clinical feature. A stroke usually damages one side of the brain, so the two hemispheres' power becomes unequal. The BSI prep is: A2 reference, 50 Hz notch, 0.5–100 Hz band-pass, 2 s epochs, rejection of epochs beyond ±100 µV, then Welch power spectra. For each pair and frequency, with R and L the right and left power:

**pdBSI = mean of |(R − L) / (R + L)|**

It is 0 for perfect symmetry and grows towards 1 with asymmetry. We compute it over 1–30 Hz and in the delta, theta, alpha and beta bands.

In code, `pdbsi` is the formula. `gate_feats` does the BSI prep and calls it for 1–30 Hz and each band; `AH` holds the left and `UH` the right channel of each pair:

{{code:stroke_phaseB.py::pdbsi}}

{{code:stroke_phaseB.py:74-79,87-89}}


![Stroke BSI](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_bsi.png)

## How the stroke numbers are produced

1. **One vector per subject:** encoder embeddings (averaged over the subject's windows, 5 fold encoders ensembled), BSI features, or both joined.
2. **Classifier:** StandardScaler + logistic regression, the same simple head as everywhere else.
3. **Leave-one-subject-out (LOSO):** train on all subjects but one, predict the left-out one, repeat for every subject, then compute one AUROC.
4. **Permutation test (1,000 shuffles):** the labels are shuffled and the whole LOSO is rerun. The p-value is how often shuffled labels score as well as the real ones.
5. **Bootstrap CI:** subjects are resampled to give a confidence interval for the AUROC.
6. **Random-init control:** the same pipeline with untrained encoders.

`loso` is step 3. `arm` runs it for every ensemble member, averages their probabilities, and does steps 4–5. The `perm_p` line counts the shuffles that score at least as well as the real labels:

{{code:stroke_phaseB.py::loso}}

{{code:stroke_phaseB.py:201-211}}

