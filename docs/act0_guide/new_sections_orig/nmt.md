# NMT Data Pipeline

The **NMT** (Normal/Abnormal) dataset is a large collection of EEG recordings used to train and evaluate our foundation models.

## Dataset Overview

The NMT dataset provides a robust foundation for learning EEG patterns.
- **Total Recordings:** 2,417 recordings
- **Total Duration:** ~489 hours of EEG data
- **Sampling Rate:** Originally recorded and standardized to **200 Hz**
- **Channels:** The raw files contain 21 channels (e.g., `FP1|FP2|F3|...|A1|A2`). After our preprocessing, we use **19 core channels**, matching standard clinical montages.

![NMT Counts](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_counts.png)

## Preprocessing

Raw EEG data contains noise, artifacts, and varying baselines. Our preprocessing pipeline ensures that the neural networks receive clean, standardized signals. 
As seen in the figure below, the raw signal is transformed into a clean, normalized signal that is much easier for the model to process.

![NMT Raw vs Prep](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_raw_vs_prep.png)

This standardization is critical for avoiding model collapse and ensuring stable learning. The Power Spectral Density (PSD) plot below shows the frequency distribution of the dataset, helping us confirm the physical properties of the signal are preserved.

![NMT PSD](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_psd.png)

## Windowing and Folds

Instead of feeding the entire 10-minute recording into the model at once, we slice the continuous EEG into smaller, manageable **windows**.

![NMT Windows](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_windows.png)

To rigorously evaluate how well our models generalize to unseen patients, we use **Cross-Validation (CV)**. The dataset is split into different folds, ensuring that the model is tested on data it has never seen during training.

![NMT CV Folds](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/nmt_cv_folds.png)
