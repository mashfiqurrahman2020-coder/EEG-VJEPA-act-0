# FEI+C Architecture

**FEI+C** is our custom dual-branch architecture tailored for EEG analysis. It combines a foundation EEG encoder (FEI) with a secondary branch (Branch-C) that processes frequency information.

## Spectrogram Branch (Branch-C)

While the main FEI branch learns from raw EEG waveforms, **Branch-C** processes the **spectrogram**, which explicitly captures the frequency content (like alpha and beta bands) over time.

![FEI+C Spectrogram](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_c_spectrogram.png)

This explicit frequency representation gives the model a strong inductive bias that matches how neurologists manually interpret EEGs. 

Branch-C operates using its own dynamic encoder block that processes the sequence of spectrogram frames:

![FEI Branch-C Blocks](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_c_blocks.png)

### Branch-C JEPA (DynJEPA)

Branch-C operates independently of the main FEI branch. It is a self-contained JEPA that learns by predicting the spectrogram features of future frames using the context of past frames. This spectral-dynamics prediction forces the network to learn how the frequency content of the EEG evolves over time.

![Branch-C JEPA](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_c_jepa.png)

## FEI Blocks and Masking

The main FEI branch uses an architecture similar to V-JEPA but optimized for 1D signals. It consists of multiple transformer blocks that process the embedded signal.

![FEI Blocks](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_fei_blocks.png)

Like V-JEPA, FEI uses extensive **masking** during self-supervised pretraining. Sections of the raw EEG are zeroed out or removed.

![FEI Masking](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_fei_masking.png)

## Joint Embedding (FEI-JEPA)

The final architecture connects the masked inputs and the targets through the **JEPA** predictive process. 

### The Actual JEPA Block Diagram

The JEPA mechanism in FEI+C utilizes both branches. The network attempts to predict the latent representation of the masked portions (processed by the target encoder), guided by the context encoder. In FEI+C, this prediction is heavily augmented by the supplementary spectrogram features from Branch-C.

![FEI JEPA](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_feic_fei_jepa.png)

The predictor must fuse the raw waveform context and the Branch-C spectrogram data to accurately predict the latent targets of the masked signal.
