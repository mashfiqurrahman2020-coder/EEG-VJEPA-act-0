import sys
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import arch_feic_common as K

W, H = 8.5, 9.0
fig, ax = K.canvas(W, H)
ax.text(W / 2, H - 0.4, "V-JEPA: Joint Embedding Predictive Architecture", ha="center", fontsize=14, weight="bold")
ax.text(W / 2, H - 0.7, "Shapes and blocks representing the pre-training process", ha="center", fontsize=10, color="#555555")

# Bottom up approach
# Inputs
K.box(ax, 2.25, 1.5, 3.6, 0.9, "Unmasked Patches (x)\ne.g. (1, 1, 32, 19, 500) -> visible subset\n(Raw EEG chunks)", color=K.GREY, fs=10)
K.box(ax, 6.25, 1.5, 3.6, 0.9, "Masked Patches (y)\nhidden subset\n(Raw EEG chunks)", color=K.GREY, fs=10)

# Encoders
K.box(ax, 2.25, 3.0, 3.6, 1.1, "Context Encoder (ViT)\n(Predicts from visible)\nTrainable Weights", color=K.ORANGE, fs=11)
K.box(ax, 6.25, 3.0, 3.6, 1.1, "Target Encoder (EMA)\n(Moving Average of Context)\nNo Gradients", color=K.PURPLE, fs=11)

K.arrow(ax, (2.25, 1.95), (2.25, 2.45), text="encode")
K.arrow(ax, (6.25, 1.95), (6.25, 2.45), text="encode")

# EMA update arrow
K.arrow(ax, (4.05, 3.0), (4.45, 3.0), color="#8a4b00", ls="--")
ax.text(4.25, 2.7, "EMA Update", ha="center", va="center", fontsize=9, color="#8a4b00")

# Features
K.box(ax, 2.25, 4.5, 3.6, 0.9, "Context Features (c)\ne.g. (batch, N_visible, 384)", color=K.GREY, fs=10)
K.box(ax, 6.25, 4.5, 3.6, 0.9, "Target Features (sy)\ne.g. (batch, N_masked, 384)", color=K.GREY, fs=10)

K.arrow(ax, (2.25, 3.55), (2.25, 4.05))
K.arrow(ax, (6.25, 3.55), (6.25, 4.05))

# Predictor
K.box(ax, 2.25, 6.0, 3.6, 1.1, "Predictor\n(Trainable Transformer)\nInput: Context + Mask Tokens", color=K.ORANGE, fs=11)
K.arrow(ax, (2.25, 4.95), (2.25, 5.45))

# Predicted Features
K.box(ax, 2.25, 7.5, 3.6, 0.9, "Predicted Features (s'y)\ne.g. (batch, N_masked, 384)", color=K.GREY, fs=10)
K.arrow(ax, (2.25, 6.55), (2.25, 7.05))

# Loss Route
ax.plot([6.25, 6.25], [4.95, 7.5], color="#333333", lw=1.4, zorder=1)
K.arrow(ax, (4.05, 7.5), (6.25, 7.5), color="red", ls="--")
ax.text(5.15, 7.7, "L2 Loss (MSE)", ha="center", va="center", fontsize=11, weight="bold", color="red")

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(fc=K.ORANGE, ec="#444444", label="Trainable Module"),
    Patch(fc=K.PURPLE, ec="#444444", label="EMA Module (Frozen)"),
    Patch(fc=K.GREY, ec="#444444", label="Data / Tensor")
], loc="lower right", bbox_to_anchor=(1.0, -0.05), fontsize=9.5, frameon=False, ncol=3)

K.save(fig, "arch_vjepa_jepa.png")
print("Saved arch_vjepa_jepa.png")
