# TUAB Data Pipeline

The **TUAB** dataset (TUH Abnormal EEG Corpus v3.0.1, Temple University Hospital, USA) is the standard benchmark for normal vs abnormal EEG. It is the dataset of **Part 1**: we test whether the released EEG-VJEPA checkpoints reach the paper's TUAB results.

- **Labels:** each recording is labelled *normal* or *abnormal* from the neurologist's clinical report.
- **Official split:** train has 1,371 normal + 1,346 abnormal recordings; eval has 150 normal + 126 abnormal. There are 2,329 patients, and no patient appears in both train and eval, so the test is on unseen people.
- **Sampling rates:** most files are 250 Hz (2,786). Some are 256 Hz (189) or 512 Hz (18).
- **Channels:** files have about 30 channels named like `EEG FP1-REF`. We keep the 19 standard 10-20 scalp channels.

<div class="warn">Our TUAB copy comes from Kaggle and is not officially licensed. It is for internal team use only; do not share TUAB numbers outside the group.</div>

## Dataset and Preprocessing

Different sampling rates, extra channels and different amplitudes would confuse a network. So every recording is converted to exactly the format the released EEG-VJEPA checkpoints were pre-trained on (paper §4.1).

![TUAB Counts](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_counts.png)

Our pipeline performs the following steps:

1. **Channel selection:** keep the 19 scalp channels, in a fixed order. Every recording then has the same channels in the same positions.
2. **Cropping:** keep the **first 300 s (5 min)**, so every recording has the same length.
3. **Band-pass filter 1–40 Hz:** remove slow drift below 1 Hz, and muscle noise and the 60 Hz power-line hum above 40 Hz. The brain rhythms (delta to beta) lie inside this band.
4. **Resample to 100 Hz:** 100 samples per second are enough for signals up to 40 Hz, and fewer samples mean less computation.
5. **Z-score each channel:** subtract the channel's mean and divide by its standard deviation. Every channel then has mean 0 and spread 1, which makes channels and recordings comparable.
6. **Framing:** cut the recording into **5 s frames** (500 samples) that start every 2.5 s (50 % overlap). This gives an array of shape (frames, 19, 500).

All six steps are one function, `preprocess_edf` in `preprocess_nmt.py`; TUAB calls the same function. The settings:

{{code:preprocess_nmt.py:29-38}}

The function body, steps 1–6 in order:

{{code:preprocess_nmt.py:43-79}}

TUAB channel names look like `EEG FP1-REF`. `preprocess_tuab.py` passes this map so that they become `FP1`:

{{code:preprocess_tuab.py:22}}


(FEI+C uses a *separate* TUAB prep matched to NMT: first 20 min, 0.5–40 Hz, 200 Hz, z-score, 2.5 s frames. It is not used by EEG-VJEPA.)

That separate prep, from `tuab_feic.py` (non-overlapping 500-sample frames):

{{code:tuab_feic.py:57-63}}


![TUAB Raw vs Prep](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_raw_vs_prep.png)

The **PSD** (power spectral density) shows how much power the signal has at each frequency. After preprocessing, the power outside 1–40 Hz should be strongly reduced, while the brain rhythms inside the band stay unchanged. The PSD figure lets us check that the filter did exactly this.

![TUAB PSD](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_psd.png)

## Framing and Clipping

After preprocessing, the 5-minute EEG segment is divided into smaller overlapping **frames**.

![TUAB Framing](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_framing.png)

The network does not see single frames. It sees **clips**, short "videos" made of several frames:

- **ViT-M:** a clip is **32 frames**, taken every 3rd frame (frame_step = 3).
- **ViT-B:** a clip is 16 frames, taken every 2nd frame.

The input tensor is (batch, 1, 32, 19, 500) for ViT-M. During training and evaluation, a random clip position is drawn from each recording, and the head's output for that clip is the recording's prediction.

The clip settings per model, the loader (the authors' `VideoDataset`, one random clip per recording) and the index rule (`clip`): 32 indices spread evenly over the 32 × 3 = 96 frames before `end`:

{{code:tuab_cached_eval.py:39-42}}

{{code:tuab_cached_eval.py::dataset}}

{{code:tuab_cached_eval.py::clip}}


**Training subsets:** following the paper, we do not train the head on all 2,717 training recordings. Instead we draw **5 subsets (s0–s4)** of 546 recordings each (276 normal + 270 abnormal). Each subset is random but class-balanced, and has no patient overlap. Every result is then tested on the same 276 official eval recordings, and reported as mean ± sd over the 5 subsets.

How the subsets are built (`build_subsets` in `tuab_released_eval.py`): shuffle with seed *s*, take at most one recording per patient until the quotas (`N_TRAIN`: 276 normal, 270 abnormal) are met, and link the shared eval set:

{{code:tuab_released_eval.py::build_subsets}}


![TUAB Clip](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/tuab_clip.png)
