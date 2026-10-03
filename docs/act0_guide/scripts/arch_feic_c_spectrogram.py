import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Figure E: Branch C's input. A REAL 20 s NMT window (window 0 of embed_dyn) turned into the REAL
fei_branchC.spectrogram() output under the Act-0 'h20' config, with the past/future (context/target) split."""
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/docs/act0_guide/scripts")
import arch_feic_common as K
import numpy as np, torch
import matplotlib.pyplot as plt
from preprocess_nmt import CHANNELS

cool_gate(pause=88.0, resume=78.0, abort=92.0)
F, C = K.F, K.C
args = K.configure_h20()
L, NFFT, HOP, NT, NF, TC = C.L, C.NFFT, C.HOP, C.NT, C.NF, C.TC
path, _, _ = K.fold0_recording(label=0)
sig = F.load_continuous(path)
x = torch.from_numpy(sig[:, :L])[None]                        # (1,19,4000) = window 0 in embed_dyn
z = C.spectrogram(x)                                          # (1, NT, 19*NF)
assert z.shape == (1, NT, 19 * NF)
S = z[0].numpy().reshape(NT, 19, NF)                         # (frames, channels, freq bins); channel-major, as in code
ch = CHANNELS.index("O1")
freqs = np.arange(NF) * F.FS / NFFT
centers = (np.arange(NT) * HOP + NFFT / 2) / F.FS             # frame t covers samples [t*HOP, t*HOP+NFFT)
split_t = (centers[TC - 1] + centers[TC]) / 2
ctx_end = ((TC - 1) * HOP + NFFT) / F.FS                      # last raw sample used by the context frames
fut_start = TC * HOP / F.FS                                   # first raw sample used by the future frames
print(f"config L={L} nfft={NFFT} hop={HOP}: {NT} frames x {NF} bins ({freqs[1]:.1f} Hz), TC={TC}, future={NT-TC}; "
      f"per-frame vector {z.shape[-1]}; context raw 0-{ctx_end:.1f}s, future raw {fut_start:.1f}-{L/F.FS:.0f}s")

fig = plt.figure(figsize=(7.6, 9.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 2.2, 2.6], width_ratios=[1, 0.025], hspace=0.55, wspace=0.03)

# (a) raw trace with context/future regions and the first STFT frames
ax = fig.add_subplot(gs[0, 0])
t = np.arange(L) / F.FS
ax.axvspan(0, ctx_end, color="#cfe0f5", lw=0); ax.axvspan(fut_start, L / F.FS, color="#fde0c2", alpha=0.8, lw=0)
ax.plot(t, sig[ch, :L], color="#333333", lw=0.5)
for k in range(3):
    a, b = k * HOP / F.FS, (k * HOP + NFFT) / F.FS
    yk = 5.2 + 1.3 * k
    ax.annotate("", xy=(a, yk), xytext=(b, yk), arrowprops=dict(arrowstyle="|-|", lw=1.0, mutation_scale=3))
ax.text(2.3, 6.5, f"STFT frames 0, 1, 2 ...: each {NFFT/F.FS:.0f} s long, a new one every {HOP/F.FS:.1f} s",
        fontsize=9, va="center")
ax.text(ctx_end / 2, -5.5, f"context (past): frames 0-{TC-1}, 0-{ctx_end:.0f} s", ha="center", fontsize=9.5, color="#1f4f8a")
ax.text((fut_start + L / F.FS) / 2, -5.5, f"future: frames {TC}-{NT-1}, {fut_start:.1f}-{L/F.FS:.0f} s", ha="center",
        fontsize=9.5, color="#8a4b00")
ax.set_ylim(-7, 9); ax.set_xlim(0, L / F.FS)
ax.set_xlabel("time (s)"); ax.set_ylabel("O1 (z-score)")
ax.set_title("(a) One real 20 s window (channel O1), cut into overlapping 1 s STFT frames", loc="left")

# (b) spectrogram of O1
ax = fig.add_subplot(gs[1, 0])
edges_t = np.r_[centers - HOP / F.FS / 2, centers[-1] + HOP / F.FS / 2]
edges_f = np.r_[freqs - 0.5, freqs[-1] + 0.5]
im = ax.pcolormesh(edges_t, edges_f, S[:, ch, :].T, cmap="viridis", shading="flat")
ax.axvline(split_t, color="white", lw=2, ls="--")
ax.text(split_t / 2, 92, f"context: {TC} frames", color="white", ha="center", fontsize=10, weight="bold")
ax.text((split_t + L / F.FS) / 2, 92, f"future: {NT-TC} frames", color="white", ha="center", fontsize=10, weight="bold")
ax.set_ylim(-0.5, 100.5); ax.set_xlim(0, L / F.FS)
ax.set_xlabel("time of frame centre (s)"); ax.set_ylabel("frequency (Hz)")
ax.set_title(f"(b) spectrogram(): O1 as {NT} time steps x {NF} frequency bins of {freqs[1]:.0f} Hz", loc="left")
cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 1])); cb.set_label("log power")

# (c) the whole matrix that the GRU reads: one row per time step, 19 x 101 = 1919 numbers per row
ax = fig.add_subplot(gs[2, 0])
im2 = ax.imshow(z[0].numpy(), aspect="auto", cmap="viridis", interpolation="nearest",
                extent=(-0.5, 19 * NF - 0.5, NT - 0.5, -0.5))
for c in range(1, 19):
    ax.axvline(c * NF - 0.5, color="white", lw=0.6)
ax.axhline(TC - 0.5, color="white", lw=2, ls="--")
ax.set_xticks([c * NF + NF / 2 for c in range(19)]); ax.set_xticklabels(CHANNELS, rotation=90, fontsize=9)
ax.set_ylabel("time step (frame index)")
ax.set_xlabel(f"channel (each block = its {NF} frequency bins, 0 -> 100 Hz)")
ax.set_title(f"(c) The input to Branch C: {NT} rows x {19*NF} numbers (19 channels x {NF} bins)", loc="left")
ax.text(19 * NF * 0.99, TC - 1.2, f"rows 0-{TC-1}: context", color="white", ha="right", fontsize=9.5, weight="bold")
ax.text(19 * NF * 0.99, TC + 1.4, f"rows {TC}-{NT-1}: future", color="white", ha="right", fontsize=9.5, weight="bold")
cb2 = fig.colorbar(im2, cax=fig.add_subplot(gs[2, 1])); cb2.set_label("log power")

fig.subplots_adjust(left=0.1, right=0.9, top=0.965, bottom=0.075)
K.save(fig, "arch_feic_c_spectrogram.png")
print(f"FACT c_spec: z {tuple(z.shape)} NT={NT} NF={NF} df={freqs[1]} TC={TC} fut={NT-TC} split_t={split_t} "
      f"ctx_raw=0-{ctx_end}s fut_raw={fut_start}-{L/F.FS}s logpow range {z.min():.2f}..{z.max():.2f}")
