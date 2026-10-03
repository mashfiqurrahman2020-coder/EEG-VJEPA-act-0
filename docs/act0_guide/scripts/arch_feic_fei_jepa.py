import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure E: one FEI pre-training step, drawn from a REAL forward pass of fei_pretrain.FEI on a batch of
4 REAL NMT windows (fei_pretrain.load_window + freq_mask, fold-0 training recordings). Shapes come from
forward hooks. The script also checks, with autograd, which parts each loss term trains."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import random
import arch_feic_common as K
import numpy as np, torch
import torch.nn.functional as NF
from sklearn.model_selection import StratifiedKFold

cool_gate(pause=88.0, resume=78.0, abort=92.0)
F = K.F
L, B = 1000, 4
files = F.list_split("train") + F.list_split("eval")
y = np.array([c for _, c in files])
tr = next(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))[0]
random.seed(0); torch.manual_seed(0)
picks = [files[i][0] for i in random.sample(list(tr), B)]
x = torch.stack([torch.from_numpy(F.load_window(p, L)) for p in picks])       # (4,19,1000)
xp, M = F.freq_mask(x)                                                        # real call, default U(0,0.7)
print("windows from", [p.split("/")[-1] for p in picks], "mask fractions", M.mean(1).numpy().round(3))

model = F.FEI(L // 2 + 1, d=256, h=128)
sd = torch.load(K.FEI_CKPT.format(0), map_location="cpu")
model.enc.load_state_dict(sd); model.enc_t.load_state_dict(sd)     # only the online encoder was saved
model.train()
names = ["enc", "proj", "enc_t", "proj_t", "mask_enc", "z1", "z2"]
shapes, hooks = {}, []
mods = dict(model.named_modules())
for n in names:
    hooks.append(mods[n].register_forward_hook(lambda m, i, o, n=n: shapes.setdefault(n, (tuple(i[0].shape), tuple(o.shape))) and None))
loss, e, l_embed = model(x, xp, M)
for h in hooks: h.remove()
for n in names: print(f"{n:9s} in {shapes[n][0]} -> out {shapes[n][1]}")

# replicate forward() line by line to split the loss into its two terms, and check the replica is exact
nrm = lambda a: NF.normalize(a, dim=-1)
u = model.proj(model.enc(x))
with torch.no_grad():
    up = model.proj_t(model.enc_t(xp))
m = model.mask_enc(M)
uhat = model.z1(u + m.detach()); mhat = model.z2(u.detach() - up)
l1 = NF.mse_loss(nrm(uhat), nrm(up)); l2 = NF.mse_loss(nrm(mhat), nrm(m))
assert torch.allclose(l1 + l2, loss, atol=1e-6), (l1 + l2, loss)
cos = (nrm(uhat) * nrm(up)).sum(-1)
assert torch.allclose(l1, ((2 - 2 * cos) / 128).mean(), atol=1e-6)          # MSE of unit vectors = (2 - 2 cos)/h
trains = {}
for tag, lt in (("loss1", l1), ("loss2", l2)):
    model.zero_grad(set_to_none=True)
    lt.backward(retain_graph=True)
    trains[tag] = [n for n in ("enc", "proj", "mask_enc", "z1", "z2", "enc_t", "proj_t")
                   if any(p.grad is not None and p.grad.abs().sum() > 0 for p in mods[n].parameters())]
    print(tag, "sends gradient to:", trains[tag])
assert "enc" in trains["loss1"] and "enc" not in trains["loss2"] and "enc_t" not in trains["loss1"] + trains["loss2"]

# ------------------------------------------------------------------ draw
S = lambda n, k=1: K.shape_str(shapes[n][k])
W, H = 7.6, 7.3
fig, ax = K.canvas(W, H)
ax.text(W / 2, H - 0.3, "One FEI pre-training step (real shapes, batch of 4 windows)", ha="center",
        fontsize=12, weight="bold")
yA, yB, yC, yD = 6.3, 4.6, 2.3, 0.72
K.box(ax, 1.35, yA, 2.45, 0.9, f"Input batch x\n4 windows, 19 channels,\n1000 samples (5 s) each\n{K.shape_str(x.shape)}",
      color="#ffffff")
K.box(ax, 5.05, yA, 4.3, 0.9, "freq_mask  (Figure E.1)\nFFT each channel -> remove a random fraction\n"
      f"(0 to 70 %) of the {M.shape[1]} frequency bins -> inverse FFT\n"
      f"x' {K.shape_str(xp.shape)}     mask M {K.shape_str(M.shape)}", color=K.RED)
K.arrow(ax, (2.6, yA), (2.88, yA))
K.box(ax, 1.35, yB, 2.45, 1.25, f"Online encoder + proj.\ntrainable\nEncoder (Fig. E.2):\n{S('enc')}\n"
      f"Linear 256->128:\nu {S('proj')}", color=K.BLUE)
K.box(ax, 3.8, yB, 2.1, 1.25, f"Mask prompt enc.\nLinear {M.shape[1]}->128\n(no bias)\n\n"
      f"m {S('mask_enc')}", color=K.PURPLE)
K.box(ax, 6.25, yB, 2.45, 1.25, f"Target encoder + proj.\nEMA copy, no gradient\nEncoder:\n{S('enc_t')}\n"
      f"Linear 256->128:\nu' {S('proj_t')}", color=K.ORANGE)
K.arrow(ax, (1.35, yA - 0.45), (1.35, yB + 0.63), "x", off=(0.15, 0), ha="left")
K.arrow(ax, (3.8, yA - 0.45), (3.8, yB + 0.63), "M", off=(0.15, 0), ha="left")
K.arrow(ax, (6.25, yA - 0.45), (6.25, yB + 0.63), "x'", off=(0.15, 0), ha="left")
K.arrow(ax, (2.1, yB - 0.63), (5.5, yB - 0.63), color="#8a4b00", ls="--", rad=0.2)
ax.text(3.8, yB - 1.0, "EMA, after every step: target = 0.996 x target + 0.004 x online", ha="center", va="center",
        fontsize=9, color="#8a4b00", bbox=dict(fc="white", ec="none", pad=0.4))
# bus: u, m, u' feed both predictors
ax.plot([1.35, 1.35, 6.25, 6.25], [yB - 0.63, yC + 0.75, yC + 0.75, yB - 0.63], color="#333333", lw=1.2, zorder=1)
ax.plot([3.8, 3.8], [yB - 0.63, yC + 0.75], color="#333333", lw=1.2, zorder=1)
for xx, lab in ((1.35, "u"), (3.8, "m"), (6.25, "u'")):
    ax.text(xx - 0.08, yC + 0.87, lab, ha="right", va="center", fontsize=9.5, weight="bold")
ax.text(3.8, yC + 0.87, "u, m and u' go to both predictors", ha="center", va="center", fontsize=9,
        bbox=dict(fc="white", ec="none", pad=0.3))
K.box(ax, 2.0, yC - 0.05, 3.65, 1.4, "Prediction 1 (this trains the encoder)\n"
      f"u_hat = z1(u + m)   z1 = MLP 128->128->128\nu_hat {S('z1')}  should match u'\n"
      "'from the full window and the list of removed\nbins, guess the embedding of the masked window'\n"
      "loss 1 = mean( (u_hat/|u_hat| - u'/|u'|)^2 )", color=K.GREEN, fs=9.5)
K.box(ax, 5.75, yC - 0.05, 3.5, 1.4, "Prediction 2 (helper task)\n"
      f"m_hat = z2(u - u')   z2 = MLP 128->128->128\nm_hat {S('z2')}  should match m\n"
      "'from the difference between the two\nembeddings, guess which bins were removed'\n"
      "loss 2 = mean( (m_hat/|m_hat| - m/|m|)^2 )", color=K.GREEN, fs=9.5)
K.box(ax, W / 2, yD, 7.2, 0.92, "Update\ntotal loss = loss 1 + loss 2  ->  backpropagation  ->  AdamW step (lr 2e-4, weight decay 1e-4)\n"
      "loss 1 trains: online encoder, projection, z1.   loss 2 trains: z2 and the mask prompt encoder only.\n"
      "Then the EMA step updates the target.  Batch 128 windows, 60 epochs, 4 random windows per recording.",
      color=K.GREY, fs=9.5)
K.arrow(ax, (2.0, yC - 0.75), (2.0, yD + 0.46)); K.arrow(ax, (5.75, yC - 0.75), (5.75, yD + 0.46))
K.save(fig, "arch_feic_fei_jepa.png")
print(f"FACT fei_jepa: shapes {shapes}; loss={float(loss):.5f} l1={float(l1):.5f} l2={float(l2):.5f} trains={trains}")
