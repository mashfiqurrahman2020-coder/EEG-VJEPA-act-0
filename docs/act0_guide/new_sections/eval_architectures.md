# Downstream Evaluation Architectures

A pre-trained encoder only produces embeddings. To get a decision (normal vs abnormal), a small classifier called a **head** is put on top. We use three ways of training it, and each answers a different question:

| Setting | What is trained | Question it answers |
|---|---|---|
| **Frozen + head** (a "probe") | only the head; encoder weights fixed | Are the pre-trained features themselves useful? (the paper's main TUAB result) |
| **Fine-tune** | encoder **and** head, end to end | How good can the pre-trained network become on this task? |
| **From scratch** | same network and head, starting from **random** weights | What do we get with **no** pre-training? (fine-tune minus scratch = the value of pre-training) |

Results are reported as **AUROC**, the area under the ROC curve. It is the probability that a random abnormal recording gets a higher "abnormal" score than a random normal one: 0.5 is chance and 1.0 is perfect.

**What the released EEG-VJEPA checkpoints are.** The paper's authors pre-trained them; in Act-0 we did **not** pre-train EEG-VJEPA ourselves. We also found that the released ViT-M gives almost the same tokens for every recording. This is called "collapse", and it is measured by a very low token standard deviation and a pairwise cosine similarity near 1. So any gain from its features is expected to be small.

## 1. Linear Probe (Logistic Regression)

The **Linear Probe** is the simplest head network. It is used to strictly evaluate the quality of the frozen features produced by the foundation models.

![Linear Probe Architecture](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_head_linear.png)

- **How it works (ViT-M):**
  - The encoder is frozen.
  - For every clip position, we take the **mean and standard deviation over the 512 tokens** (384 + 384 = 768 numbers).
  - We then average these over all clip positions of the recording, which gives one 768-number vector per recording.
- **How it works (FEI+C):** the features are the 512 numbers from the FEI+C readout (FEI 256 + C 256), not tokens.
- **The head:** `StandardScaler` followed by `LogisticRegression` (C = 1, `class_weight="balanced"`), trained with scikit-learn.
- **Where it is used:**
  - `tuab_vitm_linear.py` and `tuab_feic.py` on TUAB;
  - `stroke_phaseB.py` on the stroke data.
- **Why:** the same simple head for every encoder gives a fair comparison, because a powerful head cannot hide weak features.

The ViT-M pooling (`tuab_vitm_linear.py`) and the head (`tuab_feic.py`; `tuab_vitm_linear.py` and `stroke_phaseB.py` build the same scaler + logistic-regression pipeline):

{{code:tuab_vitm_linear.py::pool}}

{{code:tuab_feic.py::proba}}


## 2. Attention Classifier (the paper's §4.2 head: frozen probe and fine-tuning)

This is the head from the authors' own evaluation code (`AttentionClassifier`, `src/models/attentive_pooler.py`). We use it in two places:

- **Frozen:** on the frozen released encoder (the main Part-1 table, `tuab_cached_eval.py`);
- **Fine-tune and scratch:** as the head of the fine-tuned and from-scratch networks (`tuab_finetune_fast.py`).

The authors' recipe is 500 epochs, batch size 2 and learning rate 1e-3.

![Attention Classifier Architecture](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_head_attention.png)

- **The head is very small.** It has only two layers:
  1. **Score:** a Linear(384 → 1) layer gives each of the 512 tokens one score.
  2. **Weights:** softmax turns the 512 scores into weights that add up to 1.
  3. **Weighted average:** the tokens are averaged with these weights, giving 384 numbers. The head thus learns *which tokens to listen to*.
  4. **Classify:** a Linear(384 → 2) layer gives the normal / abnormal scores.
- **Fine-tune:** all ViT-M weights are updated together with the head, starting from the released checkpoint.
- **From scratch:** exactly the same, but the ViT-M starts from random weights.

The authors' code:

{{code:src/models/attentive_pooler.py::AttentionClassifier}}


## 3. Attentive probe (the paper's Fig-3 head, frozen only)

The second frozen head (`AttentiveClassifier`) is larger. It is a cross-attention block (one layer, 6 attention heads for ViT-M: 384 / 64), where a learnable **query token** "asks" all 512 tokens for information, followed by a linear layer. It is only used on the frozen encoders, as the second column of the Part-1 table.

![Attentive Probe Architecture](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_head_attentive.png)

The difference from the §4.2 head: there, each token gets a score from a single Linear layer. Here, the query token is compared with every token through full multi-head attention (queries, keys and values), and the result then passes through an MLP. This gives the head many more trainable weights.

The authors' code (`AttentivePooler`, in the same file, holds the query token and the cross-attention block):

{{code:src/models/attentive_pooler.py::AttentiveClassifier}}


## Summary of Integration

- **Linear probe:** `[Preprocessed clips] -> [Frozen encoder] -> [Mean/Std pooling, averaged per recording] -> [StandardScaler + Logistic Regression] -> [Prediction]`
- **Frozen §4.2 / Fig-3 probe:** `[Preprocessed clip] -> [Frozen ViT-M] -> [512 tokens x 384] -> [AttentionClassifier or AttentiveClassifier] -> [Prediction]`
- **Fine-tune / scratch:** `[Preprocessed clip] -> [Trainable ViT-M] -> [512 tokens x 384] -> [AttentionClassifier] -> [Prediction]`
