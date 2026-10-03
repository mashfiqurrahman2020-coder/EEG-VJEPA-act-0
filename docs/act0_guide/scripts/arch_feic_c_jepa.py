import sys
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import arch_feic_common as K

W, H = 8.5, 9.0
fig, ax = K.canvas(W, H)
ax.text(W / 2, H - 0.4, "Branch-C JEPA: Spectral-Dynamics (DynJEPA)", ha="center", fontsize=14, weight="bold")
ax.text(W / 2, H - 0.7, "Predicts the future-span embedding from the past-span context", ha="center", fontsize=10, color="#555555")

# Bottom up approach
# Inputs
K.box(ax, 4.25, 0.7, 5.0, 0.7, "Full Spectrogram Sequence (z)\ne.g. 39 time steps x 1919 bins", color=K.GREY, fs=10)

K.arrow(ax, (4.25, 1.05), (2.25, 1.5), ls="--")
K.arrow(ax, (4.25, 1.05), (6.25, 1.5), ls="--")

K.box(ax, 2.25, 1.7, 3.6, 0.9, "Past / Context Frames\ne.g. first 23 steps (z[:, :TC])", color=K.GREY, fs=10)
K.box(ax, 6.25, 1.7, 3.6, 0.9, "Future / Target Frames\ne.g. last 16 steps (z[:, TC:])", color=K.GREY, fs=10)

# Encoders
K.box(ax, 2.25, 3.2, 3.6, 1.1, "Context Encoder + Proj\n(DynEncoder)\nTrainable Weights", color=K.ORANGE, fs=11)
K.box(ax, 6.25, 3.2, 3.6, 1.1, "Target Encoder + Proj\n(EMA of Context)\nNo Gradients", color=K.PURPLE, fs=11)

K.arrow(ax, (2.25, 2.15), (2.25, 2.65), text="encode")
K.arrow(ax, (6.25, 2.15), (6.25, 2.65), text="encode")

# EMA update arrow
K.arrow(ax, (4.05, 3.2), (4.45, 3.2), color="#8a4b00", ls="--")
ax.text(4.25, 2.9, "EMA Update", ha="center", va="center", fontsize=9, color="#8a4b00")

# Features
K.box(ax, 2.25, 4.7, 3.6, 0.9, "Context Embedding (u)\n(batch, 128)", color=K.GREY, fs=10)
K.box(ax, 6.25, 4.7, 3.6, 0.9, "Target Embedding (v)\n(batch, 128)", color=K.GREY, fs=10)

K.arrow(ax, (2.25, 3.75), (2.25, 4.25))
K.arrow(ax, (6.25, 3.75), (6.25, 4.25))

# Predictor
K.box(ax, 2.25, 6.2, 3.6, 1.1, "Predictor\n(MLP)\nInput: Context Embedding", color=K.ORANGE, fs=11)
K.arrow(ax, (2.25, 5.15), (2.25, 5.65))

# Predicted Features
K.box(ax, 2.25, 7.7, 3.6, 0.9, "Predicted Future (u_hat)\n(batch, 128)", color=K.GREY, fs=10)
K.arrow(ax, (2.25, 6.75), (2.25, 7.25))

# Loss Route
ax.plot([6.25, 6.25], [5.15, 7.7], color="#333333", lw=1.4, zorder=1)
K.arrow(ax, (4.05, 7.7), (6.25, 7.7), color="red", ls="--")
ax.text(5.15, 7.9, "L2 Loss (MSE)\non normalized outputs", ha="center", va="center", fontsize=11, weight="bold", color="red")

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(fc=K.ORANGE, ec="#444444", label="Trainable Module"),
    Patch(fc=K.PURPLE, ec="#444444", label="EMA Module (Frozen)"),
    Patch(fc=K.GREY, ec="#444444", label="Data / Tensor")
], loc="lower right", bbox_to_anchor=(1.0, -0.05), fontsize=9.5, frameon=False, ncol=3)

K.save(fig, "arch_feic_c_jepa.png")
print("Saved arch_feic_c_jepa.png")
