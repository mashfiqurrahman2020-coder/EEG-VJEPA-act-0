# TUAB Data Pipeline

The **TUAB** (Temple University Hospital Abnormal) dataset is a major resource for abnormal EEG classification.

## Dataset and Preprocessing

The TUAB recordings often contain extra channels and varying sampling rates (e.g., 250 Hz). To use this data, we apply a strict preprocessing pipeline that aligns it with our models' expected input format.

![TUAB Counts](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_counts.png)

Our pipeline performs the following steps:
1. **Channel Selection:** We extract 19 core channels, matching the 10-20 system.
2. **Cropping:** We crop a standard 5-minute (300 seconds) segment from the recording.
3. **Filtering and Resampling:** The signal is filtered (0.5 to 40 Hz) and resampled to 100 Hz.
4. **Standardization:** We apply z-scoring to normalize the amplitude.

![TUAB Raw vs Prep](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_raw_vs_prep.png)

This standardization ensures the signal's frequency characteristics are consistent, which is verified by analyzing the Power Spectral Density (PSD).

![TUAB PSD](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_psd.png)

## Framing and Clipping

After preprocessing, the 5-minute EEG segment is divided into smaller overlapping **frames**.

![TUAB Framing](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_framing.png)

From these frames, specific **clips** are sampled to feed into the different network architectures (like ViT-M or ViT-B). This process transforms a continuous recording into a sequence of discrete inputs.

![TUAB Clip](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_clip.png)
