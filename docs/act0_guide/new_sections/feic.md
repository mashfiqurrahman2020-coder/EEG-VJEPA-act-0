# FEI+C Architecture

**FEI+C** is our own two-branch model. It is made of two small encoders, each pre-trained **separately** on NMT with its own self-supervised task:

- **FEI** (Frequency-masked Embedding Inference, "Branch A") reads the raw waveform.
- **Branch C** reads a spectrogram and learns how the spectrum changes over time.

The two branches only meet at the very end, where their outputs are joined for the classifier (see "From encoders to a prediction" below). Act-0 always uses **plain FEI** (never Ada-FEI).

An **encoder** is a network that turns a piece of EEG into a short list of numbers, called an **embedding**. Two similar EEG pieces should give similar embeddings.

## Branch C: the spectrogram branch

While the main FEI branch learns from raw EEG waveforms, **Branch-C** processes the **spectrogram**, which explicitly captures the frequency content (like alpha and beta bands) over time.

![FEI+C Spectrogram](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_c_spectrogram.png)

A **spectrogram** shows how much power each frequency has, moment by moment. Branch C builds it from a 20 s window (4,000 samples at 200 Hz):

- **Short-time FFT:** a 1 s window (nfft = 200) moves in 0.5 s steps (hop = 100). This gives **39 time steps**.
- **Frequencies:** each step has **101 frequency bins** of 1 Hz each (0–100 Hz).
- **Log-power:** the power is converted to log-power.
- **All channels:** the 19 channels are stacked, so each time step is described by 19 × 101 = **1,919 numbers**.

The Branch-C settings used in Act-0 (`h20_args`: L = 4000, nfft = 200, hop = 100; the context is 60 % of the 39 steps = 23) and the spectrogram function:

{{code:head_to_head_cv.py::h20_args}}

{{code:fei_branchC.py::spectrogram}}


Branch C's encoder (`DynEncoder`) then processes the 39 steps in order:

1. **BatchNorm:** each of the 1,919 inputs is shifted and scaled, using a mean and variance remembered from NMT. This keeps every input near 0 with spread about 1. If new data has a very different spectrum, these remembered values no longer fit, and the inputs become huge. This is exactly what happened on the stroke data (Section "Stroke", clean re-prep).
2. **Linear 1,919 → 256, then ReLU:** this compresses each step into 256 numbers.
3. **GRU (256):** a recurrent layer reads the 39 steps one after another and keeps a running "memory". Its final memory state (256 numbers) is the window's embedding.

{{code:fei_branchC.py::DynEncoder}}


![FEI Branch-C Blocks](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_c_blocks.png)

### Branch-C JEPA (DynJEPA)

Branch C is trained on its own, independently of FEI. Its self-supervised task is "predict the future from the past":

- **Context encoder:** reads the first **23 time steps** (the past) and produces an embedding.
- **Target encoder:** a slowly updated copy of the encoder, called the EMA copy. It reads the last **16 steps** (the future) and produces the target embedding.
- **Predictor:** a small network that tries to turn the past embedding into the future one.

The loss compares *embeddings* (L2-normalised MSE), never raw spectrogram values. **EMA** (exponential moving average) means that after every training step the target's weights move 0.4 % of the way towards the context encoder's (factor 0.996). The target therefore changes slowly and gives stable targets.

`DynJEPA.forward` splits the steps at `TC`; the target runs under `no_grad`, and `ema` is the slow update:

{{code:fei_branchC.py:142-152}}


**Self-supervised** means no labels are used. The data itself provides the task.

![Branch-C JEPA](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_c_jepa.png)

## FEI (Branch A): encoder and frequency masking

FEI's encoder is a small **1-D convolutional network**, not a transformer. Its input is a 20 s window of 19 channels × 4,000 samples (200 Hz). It has four blocks, each one Conv1d (kernel 7, stride 2) → BatchNorm → GELU:

- The number of channels grows 19 → 64 → 128 → 128 → 256.
- Stride 2 halves the time length in every block.
- An average over time then gives 256 numbers, and a final Linear layer outputs the **256-number embedding**.

A convolution slides a small learned filter along time, so each block detects local waveform patterns.

{{code:fei_pretrain.py::Encoder}}


![FEI Blocks](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_fei_blocks.png)

FEI's masking hides **frequencies, not time**. For each training window:

1. Take the FFT of every channel. The FFT turns the signal into its list of frequency components.
2. Pick a random fraction between 0 % and 70 % of the frequency bins. Set them to zero, using the same bins on all 19 channels.
3. Transform back to a waveform (inverse FFT). The result looks like the same EEG with some rhythms removed.

`freq_mask`. Act-0 uses plain FEI, which is `bias = 0`: the `else` branch, `p = ratio`, so every bin has the same chance of being masked. The `if bias` branch is Ada-FEI and is not used in Act-0:

{{code:fei_pretrain.py::freq_mask}}


The figure shows this on a real NMT window.

![FEI Masking](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_fei_masking.png)

## FEI pre-training (the JEPA task)

FEI is pre-trained with a **JEPA** task (Joint-Embedding Predictive Architecture: predict an *embedding*, not the raw signal). **Branch C plays no part in it.** One training step works like this:

1. **Online encoder:** reads the **original** window and gives embedding *u*.
2. **Target encoder** (the EMA copy): reads the **frequency-masked** window and gives the target *u′*.
3. **Mask prompt:** a small layer turns "which bins were removed" into a vector *m*.
4. **Predictor 1:** from *u* + *m*, predict *u′*. The question it answers: "what will the EEG look like to the target encoder once these frequencies are removed?"
5. **Predictor 2:** from *u* − *u′*, predict *m*. The question: "which frequencies were removed?"
6. **Loss:** the sum of both errors, using L2-normalised MSE. Then the target encoder takes one EMA step (0.996).

To succeed, the embedding has to "know" the spectral content of the EEG.

`FEI.forward` is steps 1–6 (`x` = original window, `xp` = masked window, `M` = mask), and `ema` is the target update:

{{code:fei_pretrain.py:135-152}}

One training step inside `pretrain` (for plain FEI `args.mask_bias` is 0):

{{code:fei_pretrain.py:238-241}}


![FEI JEPA](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_fei_jepa.png)

## From encoders to a prediction (FEI+C readout)

After pre-training, both encoders are **frozen**: their weights are never changed again. A whole recording becomes one prediction like this:

1. **Windows:** cut the recording into non-overlapping 20 s windows.
2. **Average:** run each window through FEI (256 numbers) and through Branch C (256 numbers), then average over all windows of the recording.
3. **Join:** concatenate the two vectors into **512 numbers** per recording. This is the only place where the two branches meet (late fusion).
4. **Classify:** apply StandardScaler (each feature is shifted to mean 0 and scaled to spread 1), then **logistic regression** (C = 1, balanced class weights). This gives a probability of "abnormal" (TUAB) or "stroke" (Phase B).
5. **Ensemble:** we have 5 pre-trained versions of each encoder, one per NMT cross-validation fold (f0–f4). Their 5 probabilities are averaged.

In `tuab_feic.py`, `feat` builds each arm's features (FEI+C = the two 256-number blocks side by side). The probe fits one classifier per fold encoder and averages the 5 probabilities (`P_.mean(0)`):

{{code:tuab_feic.py:111,120-123}}


**Random-init control:** the same pipeline is also run with *untrained* (randomly initialised) encoders. If the pre-trained encoders do not beat these, pre-training did not help.

How `embed` in `tuab_feic.py` builds the encoders: fold encoders `f0`–`f4` load their NMT weights, while `rand0`–`rand2` keep the random weights drawn after `torch.manual_seed`:

{{code:tuab_feic.py:87-92}}

