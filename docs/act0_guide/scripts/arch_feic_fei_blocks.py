import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure E: the REAL FEI encoder (fei_pretrain.Encoder, checkpoint fei_enc_cv_s0_f0.pt) as a block
diagram. Every shape is recorded by forward hooks during a real forward pass on a real NMT window:
a 20 s window (how Act-0 embeds recordings) and a 5 s window (the pre-training window length)."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import random
import arch_feic_common as K
import numpy as np, torch

cool_gate(pause=88.0, resume=78.0, abort=92.0)
F = K.F
path, _, _ = K.fold0_recording(label=0)
enc = K.load_fei(0)
sig = F.load_continuous(path)
x20 = torch.from_numpy(sig[:, :4000])[None]                   # window 0 of embed_recording at L=4000
random.seed(0)
x5 = torch.from_numpy(F.load_window(path, 1000))[None]         # a pre-training window (L=1000)
names = [f"conv.{i}" for i in range(12)] + ["head"]
rec20, out20 = K.record_shapes(enc, x20, names)
rec5, out5 = K.record_shapes(enc, x5, names)
mods = dict(enc.named_modules())

# receptive field of one output step of the last conv block: formula from the real layer attributes ...
rf, jump = 1, 1
for i in (0, 3, 6, 9):
    m = mods[f"conv.{i}"]
    rf += (m.kernel_size[0] - 1) * jump
    jump *= m.stride[0]
# ... checked empirically: which input samples change one output step of block 4?
xg = x20.clone().requires_grad_(True)
h = enc.conv(xg)
h[0, :, h.shape[-1] // 2].sum().backward()
rf_emp = int((xg.grad[0].abs().sum(0) > 0).sum())
assert rf_emp == rf, (rf_emp, rf)
print(f"receptive field of one block-4 output step = {rf} samples = {rf / F.FS:.3f} s; step spacing {jump} samples")

blocks = []
for b in range(4):
    conv, bn = mods[f"conv.{3*b}"], mods[f"conv.{3*b+1}"]
    i20 = rec20[3 * b][2]; o20 = rec20[3 * b + 2][3]; o5 = rec5[3 * b + 2][3]
    p = K.nparams(conv) + K.nparams(bn)
    blocks.append((f"Block {b+1}: Conv1d {conv.in_channels}->{conv.out_channels}, kernel {conv.kernel_size[0]}, "
                   f"stride {conv.stride[0]}\n+ BatchNorm1d({bn.num_features}) + GELU", o20, o5, p))
    print(f"block{b+1}: in {i20} out20 {o20} out5 {o5} params {p}")
head = mods["head"]
pool20, pool5 = rec20[-1][2], rec5[-1][2]
assert tuple(out20.shape) == (1, 256) and tuple(out5.shape) == (1, 256)
p_enc = K.nparams(enc)
full = F.FEI(1000 // 2 + 1, d=256, h=128)                      # the whole pre-training model (FEI) for the count
p_full = K.nparams(full); p_train = K.nparams(full, trainable_only=True)
print(f"encoder params {p_enc}; FEI pre-training model {p_full} ({p_train} trainable)")

# ------------------------------------------------------------------ draw
rows = [("Input window (19 channels x time)", "19 electrodes, z-scored, 200 samples per second",
         tuple(x20.shape), tuple(x5.shape), 0, "#ffffff")]
rows += [(t.split("\n")[0], t.split("\n")[1], o20, o5, p, K.BLUE) for t, o20, o5, p in blocks]
rows += [("Global average pool over time", "mean of each of the 256 feature rows", pool20, pool5, 0, K.GREEN),
         (f"Linear head {head.in_features}->{head.out_features}", "one fully-connected layer",
          tuple(out20.shape), tuple(out5.shape), K.nparams(head), K.ORANGE)]

W, H = 7.6, 8.05
fig, ax = K.canvas(W, H)
bx, bw, bh = 2.35, 4.3, 0.70
c20, c5 = 5.35, 6.75
top = H - 1.05
ax.text(bx, H - 0.35, "FEI encoder (Branch A): layer by layer", ha="center", fontsize=12, weight="bold")
ax.text(c20, H - 0.62, "20 s window\n(embedding)", ha="center", va="center", fontsize=9.5, weight="bold")
ax.text(c5, H - 0.62, "5 s window\n(pre-training)", ha="center", va="center", fontsize=9.5, weight="bold")
gap = 0.92
for r, (t1, t2, s20, s5, p, col) in enumerate(rows):
    y = top - r * gap
    ptxt = f"   [{p:,} parameters]" if p else ""
    K.box(ax, bx, y, bw, bh, f"{t1}\n{t2}{ptxt}", color=col, fs=9.5)
    ax.text(c20, y, K.shape_str(s20), ha="center", va="center", fontsize=9.5, family="DejaVu Sans Mono")
    ax.text(c5, y, K.shape_str(s5), ha="center", va="center", fontsize=9.5, family="DejaVu Sans Mono")
    if r:
        K.arrow(ax, (bx, y + gap - bh / 2), (bx, y + bh / 2))
yb = top - len(rows) * gap + 0.2
ax.text(0.15, yb,
        f"Output: one embedding of 256 numbers per window, whatever the window length.\n"
        f"Shapes are (batch, channels or features, time steps), recorded by forward hooks on\n"
        f"fei_pretrain.Encoder with checkpoint fei_enc_cv_s0_f0.pt. Each block halves the time axis;\n"
        f"one step after block 4 sees {rf} input samples = {rf / F.FS:.2f} s. "
        f"Encoder: {p_enc:,} parameters.",
        ha="left", va="top", fontsize=9.5, linespacing=1.35)
K.save(fig, "arch_feic_fei_blocks.png")
print(f"FACT fei_blocks: in20 {tuple(x20.shape)} in5 {tuple(x5.shape)} pool20 {pool20} pool5 {pool5} out {tuple(out20.shape)} "
      f"rf={rf} params_enc={p_enc} params_FEI={p_full} trainable={p_train}")
