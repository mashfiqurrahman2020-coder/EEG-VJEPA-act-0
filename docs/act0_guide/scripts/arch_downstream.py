import sys
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import arch_feic_common as K

# 1. Linear Probe
W, H = 8.5, 8.5
fig, ax = K.canvas(W, H)
ax.text(W / 2, H - 0.4, "Downstream Linear Probe Architecture", ha="center", fontsize=14, weight="bold")
ax.text(W / 2, H - 0.7, "Used for frozen evaluation on TUAB", ha="center", fontsize=10, color="#555555")

# Top down approach
# Input
K.box(ax, 4.25, 6.8, 4.6, 1.0, "Preprocessed clip (input)\n(batch, 1, 32 frames, 19 channels, 500 samples)", color=K.GREY, fs=10)

# Foundation Model
K.box(ax, 4.25, 5.2, 4.6, 1.2, "Frozen Foundation Model\n(V-JEPA or FEI+C)\nNo gradients updated. Used as feature extractor.", color=K.BLUE, fs=11)
K.arrow(ax, (4.25, 6.3), (4.25, 5.8))

# Token Embeddings
K.box(ax, 4.25, 3.7, 4.6, 1.0, "Token Embeddings\n(batch, tokens, dim)\ne.g. (batch, 512, 384)", color=K.GREY, fs=10)
K.arrow(ax, (4.25, 4.6), (4.25, 4.2))

# Pooling
K.box(ax, 4.25, 2.2, 4.6, 1.2, "Feature Pooling\nConcatenated Mean & Std across all tokens\n(batch, 384) + (batch, 384) -> (batch, 768)", color=K.GREY, fs=10)
K.arrow(ax, (4.25, 3.2), (4.25, 2.8))

# Linear Head
K.box(ax, 4.25, 0.8, 4.6, 1.0, "StandardScaler + Logistic Regression\n768 features -> P(abnormal)\n(averaged over all clip positions first)", color=K.ORANGE, fs=11)
K.arrow(ax, (4.25, 1.6), (4.25, 1.3))

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(fc=K.BLUE, ec="#444444", label="Frozen Module"),
    Patch(fc=K.ORANGE, ec="#444444", label="Trainable Head"),
    Patch(fc=K.GREY, ec="#444444", label="Data / Tensor / Math Op")
], loc="lower right", bbox_to_anchor=(1.0, -0.05), fontsize=9.5, frameon=False, ncol=3)

K.save(fig, "arch_head_linear.png")
print("Saved arch_head_linear.png")


# 2. Attention Classifier
W, H = 8.5, 8.5
fig, ax = K.canvas(W, H)
ax.text(W / 2, H - 0.4, "Downstream Attention Classifier (Fine-Tuning)", ha="center", fontsize=14, weight="bold")
ax.text(W / 2, H - 0.7, "Used for end-to-end fine-tuning on TUAB", ha="center", fontsize=10, color="#555555")

# Top down approach
# Input
K.box(ax, 4.25, 6.8, 4.6, 1.0, "Preprocessed clip (input)\n(batch, 1, 32 frames, 19 channels, 500 samples)", color=K.GREY, fs=10)

# Foundation Model
K.box(ax, 4.25, 5.2, 4.6, 1.2, "Trainable Foundation Model\n(V-JEPA)\nWeights updated end-to-end.", color=K.ORANGE, fs=11)
K.arrow(ax, (4.25, 6.3), (4.25, 5.8))

# Token Embeddings
K.box(ax, 4.25, 3.7, 4.6, 1.0, "Token Embeddings\n(batch, tokens, dim)\ne.g. (batch, 512, 384)", color=K.GREY, fs=10)
K.arrow(ax, (4.25, 4.6), (4.25, 4.2))

# Pooling
K.box(ax, 4.25, 2.2, 5.2, 1.2, "Attention pooling (AttentionClassifier)\nLinear(384 -> 1) score per token -> softmax weights\nweighted sum of 512 tokens -> (batch, 384)", color=K.ORANGE, fs=10)
K.arrow(ax, (4.25, 3.2), (4.25, 2.8))

# Linear Head
K.box(ax, 4.25, 0.8, 4.6, 1.0, "Linear Classification Head\nLinear(384 -> 2 classes)\nPredicts Normal vs. Abnormal", color=K.ORANGE, fs=11)
K.arrow(ax, (4.25, 1.6), (4.25, 1.3))

ax.legend(handles=[
    Patch(fc=K.ORANGE, ec="#444444", label="Trainable Module"),
    Patch(fc=K.GREY, ec="#444444", label="Data / Tensor")
], loc="lower right", bbox_to_anchor=(1.0, -0.05), fontsize=9.5, frameon=False, ncol=2)

K.save(fig, "arch_head_attention.png")
print("Saved arch_head_attention.png")


# 3. Attentive probe (AttentiveClassifier, Fig-3 head; structure from src/models/attentive_pooler.py)
W, H = 8.5, 10.5
fig, ax = K.canvas(W, H)
ax.text(W / 2, H - 0.4, "Attentive Probe (AttentiveClassifier, Fig-3 head)", ha="center", fontsize=14, weight="bold")
ax.text(W / 2, H - 0.7, "Frozen evaluation on TUAB only; ViT-M sizes shown", ha="center", fontsize=10, color="#555555")

K.box(ax, 3.4, 9.0, 4.6, 0.9, "Preprocessed clip (input)\n(batch, 1, 32 frames, 19 channels, 500 samples)", color=K.GREY, fs=10)
K.box(ax, 3.4, 7.7, 4.6, 0.9, "Frozen Foundation Model (V-JEPA ViT-M)\nNo gradients updated.", color=K.BLUE, fs=11)
K.arrow(ax, (3.4, 8.55), (3.4, 8.15))
K.box(ax, 3.4, 6.4, 4.6, 0.9, "Token Embeddings\n(batch, 512, 384) -> LayerNorm\n= keys and values", color=K.GREY, fs=10)
K.arrow(ax, (3.4, 7.25), (3.4, 6.85))

K.box(ax, 7.1, 6.4, 2.4, 0.9, "Learnable query token\n(1, 1, 384)\n= the query", color=K.ORANGE, fs=10)
K.box(ax, 4.25, 4.9, 6.2, 1.1, "Cross-attention, 6 heads (384 / 64)\nthe query attends to all 512 tokens -> (batch, 1, 384)\nresidual: query + attention output", color=K.ORANGE, fs=10)
K.arrow(ax, (3.4, 5.95), (3.4, 5.45))
K.arrow(ax, (7.1, 5.95), (7.1, 5.45))

K.box(ax, 4.25, 3.4, 6.2, 1.0, "LayerNorm -> MLP 384 -> 1536 -> 384 (GELU)\nresidual add, then squeeze -> (batch, 384)", color=K.ORANGE, fs=10)
K.arrow(ax, (4.25, 4.35), (4.25, 3.9))

K.box(ax, 4.25, 1.9, 4.6, 1.0, "Linear Classification Head\nLinear(384 -> 2 classes)\nPredicts Normal vs. Abnormal", color=K.ORANGE, fs=11)
K.arrow(ax, (4.25, 2.9), (4.25, 2.4))

ax.legend(handles=[
    Patch(fc=K.BLUE, ec="#444444", label="Frozen Module"),
    Patch(fc=K.ORANGE, ec="#444444", label="Trainable Head"),
    Patch(fc=K.GREY, ec="#444444", label="Data / Tensor")
], loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=9.5, frameon=False, ncol=3)

K.save(fig, "arch_head_attentive.png")
print("Saved arch_head_attentive.png")
