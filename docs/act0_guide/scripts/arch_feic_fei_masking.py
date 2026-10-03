import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure E: FEI's frequency masking on a REAL NMT pre-training window, using the REAL
fei_pretrain.load_window (the training data loader) and fei_pretrain.freq_mask (the masking function)."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import random
import arch_feic_common as K
import numpy as np, torch
import matplotlib.pyplot as plt

cool_gate(pause=88.0, resume=78.0, abort=92.0)
F = K.F
L = 1000                                   # FEI pre-training window = fei_pretrain --L default (5 s @ 200 Hz)
path, _, _ = K.fold0_recording(label=0)
random.seed(0)
x = torch.from_numpy(F.load_window(path, L))[None]          # (1,19,1000), exactly what the DataLoader yields
torch.manual_seed(0)
FRAC = 0.4                                  # fixed for the picture only (training draws it from U(0, 0.7))
xp, M = F.freq_mask(x, b1=FRAC, b2=FRAC)    # REAL function; plain FEI (default bias=0 -> uniform)
M = M[0].numpy().astype(bool)               # (501,) True = bin removed
n = M.size
freqs = np.fft.rfftfreq(L, d=1 / F.FS)      # 0, 0.2, ..., 100 Hz
ch = 8                                      # channel index 8 = O1 (preprocess_nmt.CHANNELS order)
from preprocess_nmt import CHANNELS
chn = CHANNELS[ch]

Xo = np.abs(np.fft.rfft(x[0, ch].numpy())) ** 2
Xm = np.abs(np.fft.rfft(xp[0, ch].numpy())) ** 2
# sanity: masked bins really carry no power, kept bins are unchanged
assert Xm[M].max() < 1e-6 * Xo.max() and np.allclose(Xm[~M], Xo[~M], rtol=1e-3, atol=1e-6 * Xo.max())
kept_energy = Xo[~M].sum() / Xo.sum()
print(f"rec={path}  window (1,19,{L})  n_bins={n}  bin width={freqs[1]:.2f} Hz  masked={M.sum()} ({M.mean():.1%})"
      f"  energy kept on {chn}={kept_energy:.1%}")

fig = plt.figure(figsize=(7.6, 8.4))
gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 0.45, 1.8, 2.0], hspace=0.62)

# (a) full spectrum 0-100 Hz
ax = fig.add_subplot(gs[0])
ax.semilogy(freqs, Xo, color="#1f5fa8", lw=0.9, label="original spectrum")
ax.semilogy(freqs[M], Xo[M], "o", color="#c62828", ms=2.6, label="removed bins (set to 0 in x')")
ax.set_xlim(0, 100); ax.set_ylim(Xo[1:].min() * 0.5, Xo.max() * 3)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (log scale)")
ax.set_title(f"(a) Spectrum of one channel ({chn}) of a real 5 s NMT window", loc="left")
ax.legend(loc="upper right", frameon=True)
ax.grid(alpha=0.25)

# (b) the mask itself, as a barcode over all 501 bins
ax = fig.add_subplot(gs[1])
ax.bar(freqs, M.astype(float), width=freqs[1], color="black", align="center")
ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_yticks([])
ax.set_xlabel("frequency (Hz)")
ax.set_title(f"(b) Mask M: {n} bins of {freqs[1]:.1f} Hz, black = removed ({M.sum()} bins = {M.mean():.0%})", loc="left")

# (c) zoom 6-14 Hz: individual bins
ax = fig.add_subplot(gs[2])
sel = (freqs >= 6) & (freqs <= 14)
for f0, m in zip(freqs[sel], M[sel]):
    if m:
        ax.axvspan(f0 - 0.1, f0 + 0.1, color="#f3b6b6", lw=0)
ax.plot(freqs[sel], Xo[sel], "o-", color="#1f5fa8", ms=3, lw=0.9, label="original")
ax.plot(freqs[sel][~M[sel]], Xm[sel][~M[sel]], "s", color="#e07b00", ms=3.5, label="kept bins (unchanged)")
ax.set_xlim(6, 14); ax.set_ylim(0, Xo[sel].max() * 1.45); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power")
ax.set_title("(c) Zoom on 6-14 Hz (alpha rhythm region): red stripes = removed bins", loc="left")
ax.legend(loc="upper left", ncol=2)
ax.grid(alpha=0.25)

# (d) waveform before/after
ax = fig.add_subplot(gs[3])
t = np.arange(L) / F.FS
ax.plot(t, x[0, ch].numpy(), color="#1f5fa8", lw=0.8, label="original x")
ax.plot(t, xp[0, ch].numpy(), color="#e07b00", lw=0.8, alpha=0.9, label="masked x' (inverse FFT)")
ax.set_xlim(0, L / F.FS); ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (z-score)")
ax.set_title(f"(d) The same channel in time: x (encoder input) and x' (target-encoder input)", loc="left")
ax.legend(loc="upper right", ncol=2)
ax.grid(alpha=0.25)

fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.06)
K.save(fig, "arch_feic_fei_masking.png")
print(f"FACT masking: L={L} n_bins={n} df={freqs[1]} masked={int(M.sum())} frac={M.mean():.4f} kept_energy_{chn}={kept_energy:.4f}")
