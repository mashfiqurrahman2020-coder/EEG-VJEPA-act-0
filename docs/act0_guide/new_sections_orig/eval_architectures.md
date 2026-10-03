# Downstream Evaluation Architectures

To evaluate the foundation models (V-JEPA and FEI+C) on clinical tasks like the TUAB (Temple University Abnormal EEG Corpus) dataset, Act-0 uses specific downstream classification heads. These architectures connect to the pretrained encoders to produce the final predictions.

## 1. Linear Probe (Logistic Regression)

The **Linear Probe** is the simplest head network. It is used to strictly evaluate the quality of the frozen features produced by the foundation models.

![Linear Probe Architecture](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_head_linear.png)

- **How it works:** The foundation model (V-JEPA or FEI+C) is completely frozen (its weights are not updated). The token representations from the encoder are extracted and pooled across the sequence. Specifically, we calculate the **mean and standard deviation** of all tokens to form a single feature vector for the EEG window.
- **The Head:** A simple `LogisticRegression` classifier (with `class_weight="balanced"`) is trained on these pooled features to predict normal vs. abnormal EEG.
- **Where it is used:** It is used extensively in the head-to-head comparisons (`tuab_vitm_linear.py` and `tuab_feic.py`) because it doesn't allow a complex head to compensate for a weak foundation model.

## 2. Attention Classifier (Fine-Tuning Head)

For end-to-end training where we want maximum performance, we use a more complex **Attention Classifier** head network.

![Attention Classifier Architecture](/home/mashfiq/eeg_vjepa/docs/act0_guide/figs/arch_head_attention.png)

- **How it works:** Instead of freezing the foundation model, all weights in the ViT-M backbone are updated (fine-tuned) alongside the head. 
- **The Head:** This head uses an `AttentivePooler`. Instead of taking a simple average of the tokens, it uses a Multi-Head Attention mechanism. A specialized, learnable "class token" acts as the query, attending to all the patch tokens (the keys and values) to dynamically pool the most relevant information. This is followed by a linear classification layer.
- **Where it is used:** This head is used in the fast fine-tuning pipeline (`tuab_finetune_fast.py`) and for supervised-from-scratch baselines, where the network needs to aggressively adapt its representations for the specific TUAB task.

## Summary of Integration

- **Frozen Feature Evaluation:** `[Raw EEG] -> [Frozen Foundation Model] -> [Mean/Std Pooling] -> [Logistic Regression] -> [Prediction]`
- **End-to-End Fine-Tuning:** `[Raw EEG] -> [Trainable Foundation Model] -> [Attentive Pooler] -> [Linear Layer] -> [Prediction]`
