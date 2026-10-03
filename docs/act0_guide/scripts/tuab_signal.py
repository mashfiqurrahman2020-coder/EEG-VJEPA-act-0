"""Figures from ONE real TUAB recording, pushed through the EEG-VJEPA §4.1 prep step by step:
  tuab_raw_vs_prep.png  raw EDF (native rate, microvolts) vs preprocessed (100 Hz, z-scored), same 10 s
  tuab_psd.png          power spectrum before / after the 1-40 Hz filter + the filter's own response
  tuab_framing.png      300 s -> 119 overlapping 5 s frames -> one ViT-M / ViT-B clip
  tuab_clip.png         a clip seen as a 'video' of 19 x 500 frames, with the ViT patch grid
The steps below copy code/preprocess_nmt.py:preprocess_edf line by line; the result is asserted equal
to the stored data/TUAB_preprocessed_100hz/.../<rec>.npy, so the figures show the real pipeline.
  CUDA_VISIBLE_DEVICES="" HW_THREADS=1 python tuab_signal.py
"""
import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuab_style import ROOT, NORMAL, ABNORMAL, THIRD, FOURTH, INK, INK2, W_IN, SMALL, plt, save
import numpy as np
import mne
from scipy.signal import welch, freqz
from matplotlib.patches import Rectangle

import preprocess_nmt as P                      # constants of the real prep (run with HW_THREADS=1)
from preprocess_tuab import CH_MAP
from tuab_bandpower import BANDS

REC = "aaaaaacq_s010_t000"                      # train/abnormal, 250 Hz, member of subset s0
EDF = f"{ROOT}/data/TUAB/edf/train/abnormal/01_tcp_ar/{REC}.edf"
NPY = f"{ROOT}/data/TUAB_preprocessed_100hz/train/abnormal/{REC}.npy"
SHOW = ["FP1", "C3", "O1", "T3"]
FACTS = {}


def run_prep():
    """= preprocess_edf(EDF, ..., ch_map=CH_MAP, crop_start=0), keeping every intermediate."""
    cool_gate(pause=88.0, resume=78.0, abort=92.0)
    raw = mne.io.read_raw_edf(EDF, preload=False, verbose=False)
    FACTS["edf_n_channels"], FACTS["edf_sfreq"], FACTS["edf_seconds"] = len(raw.ch_names), raw.info["sfreq"], raw.n_times / raw.info["sfreq"]
    FACTS["edf_channel_names"] = raw.ch_names
    raw.rename_channels({k: v for k, v in CH_MAP.items() if k in raw.ch_names})
    raw.pick_channels(P.CHANNELS); raw.reorder_channels(P.CHANNELS)
    t0 = min(0.0, max(0.0, raw.times[-1] - P.CROP_SECONDS))
    raw.crop(tmin=t0, tmax=min(t0 + P.CROP_SECONDS, raw.times[-1]))
    raw.load_data(verbose=False)
    x_raw, fs0 = raw.get_data().copy(), raw.info["sfreq"]
    raw.filter(P.BANDPASS[0], P.BANDPASS[1], verbose=False)
    x_filt = raw.get_data().copy()
    raw.resample(P.TARGET_FS)
    x100 = raw.get_data()
    sd = np.std(x100, axis=1, keepdims=True); sd[sd == 0] = 1.0
    z = (x100 - x100.mean(1, keepdims=True)) / sd
    nf = (z.shape[1] - P.WINDOW_SIZE) // P.WINDOW_STRIDE + 1
    frames = np.stack([z[:, i * P.WINDOW_STRIDE: i * P.WINDOW_STRIDE + P.WINDOW_SIZE] for i in range(nf)]).astype(np.float32)
    stored = np.load(NPY)
    FACTS["max_abs_diff_vs_stored_npy"] = float(np.abs(frames - stored).max())
    assert frames.shape == stored.shape and FACTS["max_abs_diff_vs_stored_npy"] < 1e-5, FACTS
    FACTS.update(shape_after_pick_crop=list(x_raw.shape), shape_after_resample=list(x100.shape),
                 frames_shape=list(frames.shape), raw_uV_std_per_ch=float((x_raw * 1e6).std(1).mean()),
                 z_mean_absmax=float(np.abs(z.mean(1)).max()), z_std=float(z.std(1).mean()))
    return x_raw, x_filt, fs0, x100, z, frames


def fig_raw_vs_prep(x_raw, x_filt, fs0, x100, z):
    t_a, t_b = 100.0, 110.0
    ci = [P.CHANNELS.index(c) for c in SHOW]
    fig = plt.figure(figsize=(W_IN, 11.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.25, 1], hspace=0.5)
    a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))
    t0 = np.arange(x_raw.shape[1]) / fs0; m0 = (t0 >= t_a) & (t0 < t_b)
    t1 = np.arange(z.shape[1]) / P.TARGET_FS; m1 = (t1 >= t_a) & (t1 < t_b)
    for k, c in enumerate(ci):
        a0.plot(t0[m0], x_raw[c, m0] * 1e6 - k * 150, lw=0.7, color=INK)
        a1.plot(t1[m1], z[c, m1] - k * 6, lw=0.8, color=NORMAL)
    a0.set_yticks([-k * 150 for k in range(len(ci))], SHOW); a1.set_yticks([-k * 6 for k in range(len(ci))], SHOW)
    a0.set_title(f"(a) Raw EDF: {fs0:g} samples/s, microvolts (rows 150 uV apart)")
    a1.set_title(f"(b) After the prep: {P.TARGET_FS} samples/s, z-scored (rows 6 units apart)")
    for a in (a0, a1):
        a.set_xlabel("time in the recording (s)"); a.grid(axis="y", visible=False); a.set_xlim(t_a, t_b)
    # (c) zoom: what filtering + resampling do to individual samples
    c = P.CHANNELS.index("O1"); za, zb = 104.0, 104.6
    mz0 = (t0 >= za) & (t0 < zb); mz1 = (t1 >= za) & (t1 < zb)
    a2.plot(t0[mz0], x_raw[c, mz0] * 1e6, "o-", ms=3.5, lw=0.8, color=INK2, label=f"raw, {fs0:g} Hz (one dot = one sample)")
    a2.plot(t0[mz0], x_filt[c, mz0] * 1e6, "-", lw=2, color=ABNORMAL, label="after the 1-40 Hz band-pass (still 250 Hz)")
    a2.plot(t1[mz1], x100[c, mz1] * 1e6, "s", ms=7, color=NORMAL, label=f"after resampling to {P.TARGET_FS} Hz (before z-score)")
    a2.set_xlabel("time (s)"); a2.set_ylabel("O1 (microvolts)")
    a2.set_title("(c) Zoom: 0.6 s of channel O1, sample by sample")
    a2.legend(loc="upper left", bbox_to_anchor=(0, -0.3), frameon=False)
    save(fig, "tuab_raw_vs_prep.png")


def fig_psd(x_raw, x_filt, fs0, x100):
    f0, p0 = welch(x_raw * 1e6, fs=fs0, nperseg=int(4 * fs0), axis=-1)
    f1, p1 = welch(x100 * 1e6, fs=P.TARGET_FS, nperseg=4 * P.TARGET_FS, axis=-1)
    p0, p1 = p0.mean(0), p1.mean(0)
    h = mne.filter.create_filter(None, fs0, P.BANDPASS[0], P.BANDPASS[1], verbose=False)   # = raw.filter defaults
    w, H = freqz(h, worN=8192, fs=fs0)
    gdb = 20 * np.log10(np.maximum(np.abs(H), 1e-8))
    FACTS["filter_taps_at_250Hz"] = int(len(h))
    FACTS["filter_minus6dB_Hz"] = [float(w[np.argmax(gdb > -6)]), float(w[len(w) - 1 - np.argmax(gdb[::-1] > -6)])]
    FACTS["psd_60Hz_over_55Hz_raw"] = float(p0[np.argmin(abs(f0 - 60))] / p0[np.argmin(abs(f0 - 55))])
    fig, ax = plt.subplots(2, 1, figsize=(W_IN, 9.4), gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.42))
    ticks = [0.5, 1, 2, 4, 8, 13, 20, 40, 125]
    for k, ((lo, hi), name) in enumerate(zip(BANDS, ["delta", "theta", "alpha", "beta", "gamma"])):
        ax[0].axvspan(lo, hi, color=THIRD, alpha=0.16 if k % 2 == 0 else 0.07, lw=0)
        ax[0].text(np.sqrt(lo * hi), 1.5e4, name, ha="center", fontsize=SMALL, color=INK2)
    m0, m1 = f0 >= 0.25, f1 >= 0.25
    ax[0].loglog(f0[m0], p0[m0], color=INK2, lw=1.4, label=f"raw ({fs0:g} Hz, the kept 300 s)")
    ax[0].loglog(f1[m1], p1[m1], color=NORMAL, lw=2, label=f"after band-pass + resample to {P.TARGET_FS} Hz")
    for x in (P.BANDPASS[0], P.BANDPASS[1], P.TARGET_FS / 2):
        ax[0].axvline(x, color=INK2, ls="--", lw=1)
    ax[0].annotate("50 Hz (dashed) = highest frequency\na 100 Hz signal can hold", xy=(49, 3e-5), xytext=(36, 3e-5),
                   fontsize=SMALL, color=INK2, ha="right", va="center", arrowprops=dict(arrowstyle="->", color=INK2))
    ax[0].text(0.27, 1.5e-4, "below 1 Hz:\nslow drift,\nremoved", fontsize=SMALL, color=INK2)
    ax[0].annotate("above 40 Hz: flat,\nlittle brain rhythm,\nremoved", xy=(75, 0.8), xytext=(36, 0.05),
                   fontsize=SMALL, color=INK2, ha="right", va="top", arrowprops=dict(arrowstyle="->", color=INK2))
    ax[0].set_xlim(0.25, fs0 / 2); ax[0].set_ylim(1e-5, 1e5)
    ax[0].set_xticks(ticks, [f"{t:g}" for t in ticks]); ax[0].minorticks_off()
    ax[0].set_xlabel("frequency (Hz, log scale)"); ax[0].set_ylabel("power (uV^2/Hz, log)")
    ax[0].set_title("(a) Power spectrum, mean of the 19 channels")
    ax[0].legend(loc="upper left", bbox_to_anchor=(0.235, 0.88), frameon=False)
    ax[1].semilogx(w[1:], gdb[1:], color=ABNORMAL, lw=2)
    lo6, hi6 = FACTS["filter_minus6dB_Hz"]
    ax[1].axhline(-6, color=INK2, ls=":", lw=1)
    ax[1].axvspan(lo6, hi6, color=THIRD, alpha=0.12, lw=0)
    ax[1].text(4.5, -11, f"pass band: kept unchanged\n(-6 dB = half amplitude at\n{lo6:.1f} Hz and {hi6:.0f} Hz)",
               fontsize=SMALL, color=INK2, ha="center", va="top")
    ax[1].text(88, -45, "stop band:\nremoved", ha="center", fontsize=SMALL, color=INK2)
    ax[1].set_xlim(0.25, fs0 / 2); ax[1].set_ylim(-80, 8)
    ax[1].set_xticks(ticks, [f"{t:g}" for t in ticks]); ax[1].minorticks_off()
    ax[1].set_xlabel("frequency (Hz, log scale)"); ax[1].set_ylabel("gain (dB)")
    ax[1].set_title(f"(b) The 1-40 Hz filter itself ({len(h)}-tap FIR at {fs0:g} Hz)")
    save(fig, "tuab_psd.png")


def clip_idx(end, fpc, step):   # = tuab_cached_eval.clip = VideoDataset.loadvideo_decord (num_clips=1)
    start = end - fpc * step
    return np.clip(np.linspace(start, end, num=fpc), start, end - 1).astype(np.int64)


def fig_framing(frames):
    nf = frames.shape[0]; c = P.CHANNELS.index("O1")
    fs, S, W = P.TARGET_FS, P.WINDOW_STRIDE, P.WINDOW_SIZE
    cont = np.concatenate([frames[i, c, :S] for i in range(nf - 1)] + [frames[-1, c]])   # undo the 50 % overlap
    assert len(cont) == (nf - 1) * S + W
    t = np.arange(len(cont)) / fs
    vm = dict(name="ViT-M", fpc=32, step=3); vb = dict(name="ViT-B", fpc=16, step=2)   # = tuab_cached_eval.MODELS
    for m in (vm, vb):
        m["L"] = m["fpc"] * m["step"]; m["n_pos"] = len(range(m["L"], nf))
    FACTS["clip_positions"] = {m["name"]: m["n_pos"] for m in (vm, vb)}
    end_m = 110                                     # one example clip end (any of 96..118 is possible)
    idx_m = clip_idx(end_m, vm["fpc"], vm["step"]); idx_b = clip_idx(70, vb["fpc"], vb["step"])
    FACTS["example_vitm_clip_frames"] = idx_m.tolist(); FACTS["example_vitb_clip_frames"] = idx_b.tolist()
    fig = plt.figure(figsize=(W_IN, 10.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.15, 1.1], hspace=0.62)
    a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))
    a0.plot(t, cont, lw=0.3, color=INK)
    a0.axvspan(idx_m[0] * S / fs, idx_m[-1] * S / fs + W / fs, color=ABNORMAL, alpha=0.18, lw=0)
    a0.text((idx_m[0] * S + W / 2) / fs, 6.2, f"one ViT-M clip (frames {idx_m[0]}-{idx_m[-1]})", color=INK, fontsize=SMALL)
    a0.set_xlim(0, t[-1]); a0.set_ylim(-8, 9); a0.set_xlabel("time (s)"); a0.set_ylabel("O1 (z-score)")
    a0.set_title(f"(a) The kept 300 s of channel O1 ({len(cont)} samples at {fs} Hz)")
    # (b) first 20 s with the overlapping frames drawn as bars
    m = t < 20
    a1.plot(t[m], cont[m], lw=0.7, color=INK)
    for i in range(7):
        y = 5.0 + (i % 2) * 1.9; col = NORMAL if i % 2 == 0 else FOURTH
        a1.add_patch(Rectangle((i * S / fs, y), W / fs, 1.5, color=col, alpha=0.9, lw=0))
        a1.text(i * S / fs + W / fs / 2, y + 0.75, f"frame {i}", ha="center", va="center", color="white", fontsize=SMALL)
    a1.set_xlim(0, 20); a1.set_ylim(-5, 8.7); a1.set_xlabel("time (s)"); a1.set_ylabel("O1 (z-score)")
    a1.set_xticks(np.arange(0, 20.1, 2.5))
    a1.set_title(f"(b) First 20 s: frames are {W/fs:g} s long and start every {S/fs:g} s (50 % overlap)")
    # (c) which frames a clip picks
    a2.scatter(np.arange(nf), np.full(nf, 2), s=14, color="#c9c8c2", zorder=2)
    a2.scatter(idx_m, np.full(len(idx_m), 2), s=34, color=ABNORMAL, zorder=3)
    a2.scatter(np.arange(nf), np.full(nf, 1), s=14, color="#c9c8c2", zorder=2)
    a2.scatter(idx_b, np.full(len(idx_b), 1), s=34, color=NORMAL, zorder=3)
    a2.set_yticks([2, 1], [f"ViT-M clip\n{vm['fpc']} frames,\nstep {vm['step']}", f"ViT-B clip\n{vb['fpc']} frames,\nstep {vb['step']}"])
    a2.set_xlim(-1, nf); a2.set_ylim(0.4, 2.85); a2.set_xlabel(f"frame index (0 ... {nf-1})")
    a2.text(-0.5, 2.3, f"example ends at frame {end_m}; {vm['n_pos']} possible end frames ({vm['L']}...{nf-1})",
            fontsize=SMALL, color=INK2)
    a2.text(-0.5, 1.3, f"example ends at frame 70; {vb['n_pos']} possible end frames ({vb['L']}...{nf-1})",
            fontsize=SMALL, color=INK2)
    a2.set_title(f"(c) All {nf} frames (grey) and the frames one clip uses (colour)")
    a2.grid(axis="y", visible=False)
    save(fig, "tuab_framing.png")
    return idx_m


def fig_clip(frames, idx_m):
    """Top: clip frame 0 with the ViT patch grid. Bottom: clip frames 0-3 = ONE tubelet (ViT-M tubelet_size 4):
    the highlighted 4-channel x 30-sample box in all four frames becomes ONE token."""
    fig = plt.figure(figsize=(W_IN, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.7, 1], hspace=0.42, wspace=0.12)
    top = fig.add_subplot(gs[0, :])
    im = top.imshow(frames[idx_m[0]], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    for r in range(0, 17, 4):                        # patch grid: 4 channels x 30 samples
        top.axhline(r - 0.5, color=INK, lw=0.7)
    for x in range(0, 481, 30):
        top.axvline(x - 0.5, color=INK, lw=0.7)
    top.add_patch(Rectangle((-0.5, 15.5), 500, 3, color="white", alpha=0.8, lw=0))    # rows 16-18 (FZ, CZ, PZ)
    top.add_patch(Rectangle((479.5, -0.5), 20, 19, color="white", alpha=0.8, lw=0))   # samples 480-499
    top.add_patch(Rectangle((59.5, 3.5), 30, 4, fill=False, ec=FOURTH, lw=3))
    top.text(250, 17.1, "rows 16-18 (FZ, CZ, PZ): not seen by the ViT", ha="center", va="center", fontsize=SMALL)
    top.set_yticks(range(19), P.CHANNELS, fontsize=11); top.grid(False)
    top.set_xticks([0, 120, 240, 360, 480], ["0", "120", "240", "360", "480"])
    top.set_xlabel("sample within the frame (100 samples = 1 s)")
    top.set_title(f"(a) Clip frame 0 (= recording frame {idx_m[0]}) with the 4 x 30 patch grid")
    cb = fig.colorbar(im, ax=top, fraction=0.03, pad=0.015); cb.set_label("z-score")
    for k in range(4):
        a = fig.add_subplot(gs[1, k])
        a.imshow(frames[idx_m[k], :16, :120], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
        a.add_patch(Rectangle((59.5, 3.5), 30, 4, fill=False, ec=FOURTH, lw=2.5))
        a.set_title(f"clip frame {k}", fontsize=SMALL); a.grid(False)
        a.set_xticks([0, 60, 90], ["0", "60", "90"]); a.set_yticks([])
        if k == 0:
            a.set_yticks([0, 4, 8, 12], [P.CHANNELS[i] for i in (0, 4, 8, 12)], fontsize=11)
    fig.text(0.13, 0.075, "(b) Clip frames 0-3, first 16 channels, samples 0-119. The purple box, taken in all 4 frames\n"
             "(4 frames x 4 channels x 30 samples = 480 numbers), becomes ONE token (tubelet size 4).",
             fontsize=SMALL, va="top")
    save(fig, "tuab_clip.png")


def main():
    x_raw, x_filt, fs0, x100, z, frames = run_prep()
    cool_gate(pause=88.0, resume=78.0, abort=92.0)
    fig_raw_vs_prep(x_raw, x_filt, fs0, x100, z)
    cool_gate(pause=88.0, resume=78.0, abort=92.0)
    fig_psd(x_raw, x_filt, fs0, x100)
    idx_m = fig_framing(frames)
    fig_clip(frames, idx_m)
    FACTS["rec"] = REC
    json.dump(FACTS, open(f"{ROOT}/docs/act0_guide/scripts/tuab_signal_facts.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in FACTS.items() if k != "edf_channel_names"}, indent=1))


if __name__ == "__main__":
    main()
