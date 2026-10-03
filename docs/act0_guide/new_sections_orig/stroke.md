# Stroke Data Pipeline

The stroke dataset evaluation (Phase B) tests how well the foundation models can detect brain damage (like strokes) by comparing stroke patients to healthy controls.

## Cohorts

The evaluation is split into primary and sensitivity cohorts, each containing stroke patients and healthy controls.

![Stroke Cohorts](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_cohorts.png)

## Preprocessing and PSD Matching

Just like with other datasets, raw EEG requires preprocessing.

![Stroke Raw vs Prep](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_raw_vs_prep.png)

A critical part of the stroke pipeline is **PSD matching**. Since the stroke recordings might come from a different device than the pretraining data (NMT), the raw power spectral density (PSD) can look different. We apply a zero-phase filter to match the PSD of the stroke data to the NMT dataset, ensuring the neural network sees a familiar signal.

![Stroke PSD Match](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_psd_match.png)

This clean prep eliminates artifacts that could cause model collapse, which is verified through our clean gating checks.

![Stroke Clean Gates](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_clean_gates.png)

## Montage and BSI

We use a standard 19-channel montage to process the spatial layout of the electrodes on the scalp.

![Stroke Montage](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_montage.png)

A key clinical feature used in stroke detection is the **Brain Symmetry Index (BSI)**. BSI measures the power asymmetry between the left and right hemispheres. A higher BSI indicates greater asymmetry, which is a strong signal for unilateral stroke. The foundation model features are often evaluated alongside this hand-crafted BSI metric.

![Stroke BSI](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/stroke_bsi.png)
