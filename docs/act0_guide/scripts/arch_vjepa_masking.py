import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Fig D.3  Real multiblock-3D masks from the authors' mask generator (src/masks/multiblock3d.MaskCollator) with the
mask settings of the authors' pre-training config (configs/pretrain/cluster_vitl16_EEG.yaml), on the ViT-M token
grid (8 tubelet times x 4 channel rows x 16 time columns = 512 tokens)."""
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_vjepa_common import *          # noqa: E402,F403
import json
import yaml                                # noqa: E402
import matplotlib                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from src.masks.multiblock3d import MaskCollator  # noqa: E402

GATE()
cfg = yaml.safe_load(open(f"{ROOT}/code/configs/pretrain/cluster_vitl16_EEG.yaml"))
D = cfg["data"]
coll = MaskCollator(cfgs_mask=cfg["mask"], crop_size=D["crop_size"], num_frames=D["num_frames"],
                    patch_size=tuple(D["patch_size"]), tubelet_size=D["tubelet_size"])
g0 = coll.mask_generators[0]
Tg, Hg, Wg = g0.duration, g0.height, g0.width
N = Tg * Hg * Wg

torch.manual_seed(0)                       # block LOCATIONS use the global RNG; fixed here so the figure is reproducible
draws = []
for i in range(4):                         # 4 successive calls = 4 training steps (seeds 0..3)
    _, m_enc, m_pred = coll([torch.zeros(1)])
    one = []
    for e, p in zip(m_enc, m_pred):
        grid = np.zeros(N, int); grid[p[0].numpy()] = 1          # 1 = hidden (predict), 0 = context (visible)
        one.append(dict(grid=grid.reshape(Tg, Hg, Wg), n_ctx=int(e.shape[1]), n_hid=int(p.shape[1])))
    draws.append(one)

same_over_time = all((d["grid"] == d["grid"][0]).all() for one in draws for d in one)

# block-size range and average hidden share over many steps (sizes use the step counter as seed, like training)
stats = []
for g in coll.mask_generators:
    sizes = {tuple(g._sample_block_size(torch.Generator().manual_seed(s), g.temporal_pred_mask_scale,
                                        g.spatial_pred_mask_scale, g.aspect_ratio)) for s in range(2000)}
    stats.append(dict(block_sizes_t_h_w=sorted(sizes)))
torch.manual_seed(1)
hid = [[], []]
for _ in range(500):
    _, m_enc, m_pred = coll([torch.zeros(1)])
    for k, p in enumerate(m_pred):
        hid[k].append(p.shape[1] / N)
for k in range(2):
    stats[k].update(mean_hidden_share_500_steps=float(np.mean(hid[k])), min=float(np.min(hid[k])),
                    max=float(np.max(hid[k])))
save_json("vjepa_masks_recorded.json", dict(
    config="code/configs/pretrain/cluster_vitl16_EEG.yaml", mask_cfg=cfg["mask"], token_grid=[Tg, Hg, Wg],
    n_tokens=N, draws=[[dict(n_context=d["n_ctx"], n_hidden=d["n_hid"]) for d in one] for one in draws],
    hidden_pattern_identical_for_all_8_tubelet_times=bool(same_over_time), per_mask_stats=stats))
print("grid", (Tg, Hg, Wg), "same over time:", same_over_time)
for i, one in enumerate(draws):
    print(i, [(d["n_ctx"], d["n_hid"]) for d in one])
print(json.dumps(stats))

cmap = ListedColormap(["#cfe0f6", ORANGE])
ROWS = [" ".join(CHANNELS[4 * r:4 * r + 4]) for r in range(4)]


def grid_ax(ax, g2, stack=0, labels=True):
    """one 4 x 16 token slice; `stack` > 0 draws that many shadow copies behind it (= the other tubelet times)"""
    for j in range(stack, 0, -1):
        ox, oy = 0.28 * j, -0.16 * j
        ax.imshow(g2, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", alpha=0.35,
                  extent=(ox - 0.5, ox + Wg - 0.5, oy + Hg - 0.5, oy - 0.5), zorder=1)
        ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), Wg, Hg, fill=False, ec=GREY, lw=0.6, zorder=1))
    ax.imshow(g2, cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=(-0.5, Wg - 0.5, Hg - 0.5, -0.5), zorder=2)
    for x in np.arange(-0.5, Wg, 1):
        ax.plot([x, x], [-0.5, Hg - 0.5], color="white", lw=0.8, zorder=3)
    for y in np.arange(-0.5, Hg, 1):
        ax.plot([-0.5, Wg - 0.5], [y, y], color="white", lw=0.8, zorder=3)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), Wg, Hg, fill=False, ec=INK, lw=0.8, zorder=4))
    ax.set_xlim(-0.6, Wg - 0.4 + 0.28 * stack); ax.set_ylim(Hg - 0.4, -0.6 - 0.16 * stack)
    ax.set_aspect("equal"); ax.set_axis_off()
    if labels:
        for r in range(Hg):
            ax.text(-0.8, r, ROWS[r], ha="right", va="center", fontsize=9, color=INK2)
        for c in (0, 4, 8, 12):
            ax.text(c, Hg - 0.25, f"{c * 0.3:.1f} s", ha="center", va="top", fontsize=9, color=INK2)


fig = plt.figure(figsize=(8.2, 8.5), dpi=170)
fig.text(0.02, 0.985, "Real masks from the authors' mask generator (ViT-M, 8 x 4 x 16 = 512 tokens)",
         fontsize=12, fontweight="bold", va="top")
fig.text(0.02, 0.957, "Blue = context: the encoder sees these tokens.\nOrange = hidden: the predictor must guess the "
         "target encoder's vectors for these tokens.", fontsize=10, color=INK2, va="top")
kinds = ["short-range", "long-range"]
for r in range(2):
    d = draws[0][r]; m = cfg["mask"][r]; bs = stats[r]["block_sizes_t_h_w"]
    ax = fig.add_axes([0.2, 0.645 - r * 0.29, 0.72, 0.17])
    grid_ax(ax, d["grid"][0], stack=Tg - 1)
    hs = sorted({b[1] for b in bs}); ws = sorted({b[2] for b in bs})
    rows = f"all {Hg} rows" if hs == [Hg] else f"{hs[0]}-{hs[-1]} rows"
    fig.text(0.02, 0.9 - r * 0.29,
             f"({'ab'[r]}) {kinds[r].capitalize()} mask: {m['num_blocks']} blocks, each asked to cover "
             f"{int(m['spatial_scale'][0] * 100)} % of the 4 x 16 slice\n"
             f"     -> each block is {rows} x {ws[0]}-{ws[-1]} columns and spans all {Tg} tubelet times\n"
             f"     (the stack behind the front slice).\n"
             f"     This step: context = {d['n_ctx']} tokens, hidden = {d['n_hid']} of {N} "
             f"({100 * d['n_hid'] / N:.0f} %).", fontsize=10, va="top", linespacing=1.35)
fig.text(0.02, 0.325, "(c) Three more training steps: a new random mask every step (front slice only).", fontsize=10, va="top")
for r in range(2):
    for j in range(3):
        d = draws[j + 1][r]
        ax = fig.add_axes([0.03 + j * 0.325, 0.215 - r * 0.13, 0.3, 0.075])
        grid_ax(ax, d["grid"][0], labels=False)
        ax.text(Wg / 2 - 0.5, Hg + 0.1, f"{kinds[r]}, step {j + 2}: {100 * d['n_hid'] / N:.0f} % hidden",
                ha="center", va="top", fontsize=9.5)
fig.text(0.02, 0.012, f"Average over 500 steps: {100 * stats[0]['mean_hidden_share_500_steps']:.0f} % hidden (short-range), "
         f"{100 * stats[1]['mean_hidden_share_500_steps']:.0f} % (long-range). Blocks can overlap, so the share varies.", fontsize=9.5, color=INK2)
out = f"{FIGS}/arch_vjepa_masking.png"
fig.savefig(out, dpi=170)
print("saved", out)
