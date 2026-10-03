# V-JEPA Architecture

**V-JEPA** (Video Joint-Embedding Predictive Architecture) is a self-supervised model first designed for video. **EEG-VJEPA** is the same model applied to EEG. In Act-0 we use the checkpoints **released by the paper's authors** (mainly ViT-M); we did not pre-train it ourselves.

## EEG as Video

To use a video model, the EEG is treated as a video:

- **One frame** = one 5 s piece of EEG, a 19 × 500 "image" (19 channels × 500 samples at 100 Hz).
- **One clip** = 32 frames in a row (ViT-M), like 32 video frames.

![V-JEPA EEG as Video](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_vjepa_eeg_as_video.png)

The model can then look at patterns across channels (space) and across frames (time) at the same time.

## Network Blocks

The clip is first cut into small pieces called **tokens**. One Conv3d layer does this:

- a patch covers **4 channels × 30 samples** of a frame, and a **tubelet** joins 4 consecutive frames;
- this gives 8 (time) × 4 (channel groups) × 16 (sample groups) = **512 tokens**, each described by **384 numbers**;
- the patches do not overlap and there is no padding, so the last 3 channels and the last 20 samples of each frame are not covered.

The tokeniser in the authors' code is one `Conv3d` whose kernel equals its stride, so the patches cannot overlap. Our loader builds ViT-M with `patch_size=(4, 30)` and `tubelet_size=4`:

{{code:src/models/utils/patch_embed.py:56-66}}

{{code:tuab_cached_eval.py::encoder}}


The 512 tokens then pass through **12 transformer blocks** (ViT-M). Each block has **self-attention** with 6 heads, where every token can take information from every other token, followed by a small MLP (384 → 1,536 → 384). The output is again 512 tokens × 384: this is what the heads in "Downstream Evaluation" read. (ViT-B is larger: 768 numbers per token, 12 heads, 16-frame clips.)

![V-JEPA Blocks](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_vjepa_blocks.png)

## Masking & The JEPA Paradigm

During pre-training, V-JEPA learns by predicting missing parts of the clip. **Masking** hides large blocks of tokens from the encoder.

The authors' mask generator (`src/masks/multiblock3d.py`). A block is a box in (time, channel group, sample group) set to 0. Several boxes are multiplied together; the zeros become the positions to predict (`mask_p`) and the ones the visible context (`mask_e`):

{{code:src/masks/multiblock3d.py::_MaskGenerator._sample_block_mask}}

{{code:src/masks/multiblock3d.py:190-196}}


![V-JEPA Masking](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_vjepa_masking.png)

### The Actual JEPA Block Diagram

The Joint Embedding Predictive Architecture (JEPA) fundamentally relies on an interaction between an **Encoder** and a **Predictor**:

1. **Context encoder:** sees only the **visible** tokens and encodes them.
2. **Target encoder (EMA):** sees the **full, unmasked** clip. Its outputs at the **masked positions** are the targets. It is a slowly updated (EMA) copy of the context encoder.
3. **Predictor:** a small transformer that takes the context tokens plus "mask tokens" marking where the hidden positions are, and predicts the target encoder's outputs there.
4. **Loss:** the difference between predicted and target *embeddings*, never raw EEG values.

The training step in the authors' `app/vjepa/train.py`. The target encoder runs without gradients on the full clip; the context encoder sees only `masks_enc`; the predictor fills in `masks_pred`. The loss is mean |z − h|^`loss_exp` (1.0 in the repository's EEG pre-training config, i.e. an L1 distance). After the optimiser step the target takes one EMA step:

{{code:app/vjepa/train.py:428-455,492-496}}


![V-JEPA JEPA Architecture](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_vjepa_jepa.png)

The idea is that predicting embeddings, rather than the exact waveform, pushes the network to learn the important structure and ignore noise.

**Collapse check.** A known failure of JEPA training is **collapse**: the encoder gives nearly the same output for every input, which makes the task trivially easy. The released ViT-M shows this: its token standard deviation is very low and different recordings give almost identical embeddings (pairwise cosine near 1). This is why its frozen results are weak.

The check (`tuab_collapse_check.py`, input `T` = tokens of shape (recordings, 512, 384)): token spread within a recording, spread of the pooled embedding across recordings, and the mean pairwise cosine:

{{code:tuab_collapse_check.py:44-49}}

