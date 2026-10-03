import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure E: the REAL Branch-C encoder (fei_branchC.DynEncoder, checkpoint branchC_h20_enc_cv_s0_f0.pt)
layer by layer. Shapes come from forward hooks during a real forward pass on the real spectrogram of a
real 20 s NMT window (fold-0 test recording, never seen by this encoder). The lower panel shows the
real GRU outputs at all 39 time steps; the last one is the embedding."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import arch_feic_common as K
import numpy as np, torch

cool_gate(pause=88.0, resume=78.0, abort=92.0)
F, C = K.F, K.C
args = K.configure_h20()                       # must run BEFORE DynEncoder is built (it reads IN_DIM)
path, _, _ = K.fold0_recording(label=0)
sig = F.load_continuous(path)
z = C.spectrogram(torch.from_numpy(sig[:, :C.L])[None])       # (1,39,1919) = window 0 in embed_dyn
enc = K.load_c(0)
rec, out = K.record_shapes(enc, z, ["bn", "embed", "gru"])
sh = {n: (i, o) for n, _, i, o in rec}
for n in ("bn", "embed", "gru"):
    print(f"{n:6s} in {sh[n][0]} -> out {sh[n][1]}")
assert tuple(out.shape) == (1, 256)

# replicate forward() to get the ReLU output and the full GRU output sequence, and check it is exact
with torch.no_grad():
    B_, T, D = z.shape
    h_bn = enc.bn(z.reshape(B_ * T, D)).reshape(B_, T, D)
    h_lin = enc.embed(h_bn)
    h_relu = torch.relu(h_lin)
    seq, h_last = enc.gru(h_relu)
assert torch.allclose(seq[:, -1], out) and torch.allclose(h_last[0], out)
frac_zero = float((h_relu == 0).float().mean())
bn = enc.bn
p_bn = K.nparams(bn); p_buf = bn.running_mean.numel() + bn.running_var.numel()
p_lin, p_gru, p_enc = K.nparams(enc.embed), K.nparams(enc.gru), K.nparams(enc)
full = C.DynJEPA(256, 128)
p_full, p_train = K.nparams(full), K.nparams(full, trainable_only=True)
print(f"params bn {p_bn} (+{p_buf} running stats) linear {p_lin} gru {p_gru} encoder {p_enc}; "
      f"DynJEPA {p_full} ({p_train} trainable); ReLU zeros {frac_zero:.1%}")

# ------------------------------------------------------------------ draw
S = K.shape_str
rows = [
    ("Input: spectrogram sequence (Figure E.4)", "39 time steps; each step = 19 channels x 101 bins = 1919 numbers",
     S(z.shape), 0, "#ffffff"),
    (f"BatchNorm1d({bn.num_features})", "flatten to (39 x 1919); each of the 1919 inputs is standardised\n"
     "with the mean and variance it had on NMT during pre-training", f"{S(sh['bn'][0])}\n-> {S(sh['bn'][1])}",
     p_bn, K.RED),
    (f"Linear {enc.embed.in_features} -> {enc.embed.out_features}", "the same layer is applied to every time step:\n"
     "1919 numbers are compressed to 256", S(sh["embed"][1]), p_lin, K.BLUE),
    ("ReLU", f"negative values become 0 (here {frac_zero:.0%} of them)", S(h_relu.shape), 0, K.BLUE),
    (f"GRU (hidden size {enc.gru.hidden_size})", "reads the 39 steps in order and keeps a memory of 256 numbers;\n"
     "outputs its memory after every step", f"{S(sh['gru'][1][0])}\n+ final {S(sh['gru'][1][1])}", p_gru, K.PURPLE),
    ("Keep only the last step:  out[:, -1]", "the memory after reading all 39 steps = the embedding", S(out.shape), 0,
     K.GREEN),
]
W, H = 7.6, 11.0
fig, ax = K.canvas(W, H)
ax.text(W / 2, H - 0.32, "Branch C encoder (DynEncoder): layer by layer", ha="center", fontsize=12, weight="bold")
bx, bw, c_sh = 2.75, 5.1, 6.55
ax.text(c_sh, H - 0.7, "shape\n(batch, time, features)", ha="center", va="center", fontsize=9.5, weight="bold")
top, gap = H - 1.35, 0.98
hts = [0.62, 0.86, 0.72, 0.52, 0.72, 0.62]
for r, (t1, t2, s, p, col) in enumerate(rows):
    y = top - r * gap
    ptxt = f"   [{p:,} parameters]" if p else ""
    K.box(ax, bx, y, bw, hts[r], f"{t1}{ptxt}\n{t2}", color=col, fs=9.5)
    ax.text(c_sh, y, s, ha="center", va="center", fontsize=9.5, family="DejaVu Sans Mono", linespacing=1.3)
    if r:
        K.arrow(ax, (bx, top - (r - 1) * gap - hts[r - 1] / 2), (bx, y + hts[r] / 2))
yb = top - len(rows) * gap + 0.42
ax.text(0.15, yb,
        f"BatchNorm also stores {p_buf:,} running statistics (mean and variance of each input). They are not trained by\n"
        f"gradients; in eval mode they are frozen at their NMT values. Encoder: {p_enc:,} parameters. Shapes recorded\n"
        f"by forward hooks on fei_branchC.DynEncoder, checkpoint branchC_h20_enc_cv_s0_f0.pt.",
        ha="left", va="top", fontsize=9, linespacing=1.35)

# lower panel: the real GRU outputs over time
axh = fig.add_axes([0.1, 0.09, 0.77, 0.22])
seqn = seq[0].numpy().T                                   # (256, 39)
im = axh.imshow(seqn, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
axh.add_patch(__import__("matplotlib").patches.Rectangle((T - 1.5, -0.5), 1, seqn.shape[0], fill=False, ec="black", lw=2))
axh.annotate("last step =\nthe embedding\n(256 numbers)", xy=(T - 1, 40), xytext=(T + 3.2, 40), fontsize=9,
             va="center", annotation_clip=False, arrowprops=dict(arrowstyle="-|>", lw=1.2))
axh.set_xlabel("time step (spectrogram frame, 0.5 s apart)")
axh.set_ylabel("GRU memory unit")
axh.set_title("GRU output after each of the 39 steps (real values, this window)", loc="left", fontsize=10.5)
cax = fig.add_axes([0.1, 0.02, 0.3, 0.012])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8.5)
cax.text(1.04, 0.5, "value (GRU outputs lie between -1 and +1)", transform=cax.transAxes, va="center", fontsize=9)
K.save(fig, "arch_feic_c_blocks.png")
print(f"FACT c_blocks: z {tuple(z.shape)} shapes {sh} out {tuple(out.shape)} params bn={p_bn} buf={p_buf} lin={p_lin} "
      f"gru={p_gru} enc={p_enc} DynJEPA={p_full} trainable={p_train} relu_zero={frac_zero:.4f} "
      f"gru_out_range {seqn.min():.3f}..{seqn.max():.3f}")
