import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads, cool_gate; cap_threads(1)
"""Fig D.2  Block diagram of the REAL released ViT-M encoder, built from forward hooks on one REAL TUAB eval clip
(CPU). Also records the same shapes for the released ViT-B -> facts/vjepa_recorded_shapes.json."""
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_vjepa_common import *          # noqa: E402,F403
import matplotlib                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402


def n_params(m, trainable_only=False):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only))


def record(model):
    GATE()
    f = eval_files("normal", 1)[0]
    _, idx, c = load_clip(f, model)
    enc = encoder(model)
    rec, hooks = {}, []

    def hook(name):
        def h(_m, _i, o):
            o = o[0] if isinstance(o, tuple) else o
            rec[name] = list(o.shape)
        return h
    b0 = enc.blocks[0]
    named = {"patch_embed.proj (Conv3d)": enc.patch_embed.proj, "patch_embed (tokens)": enc.patch_embed,
             "block0.norm1": b0.norm1, "block0.attn.qkv": b0.attn.qkv, "block0.attn": b0.attn,
             "block0.norm2": b0.norm2, "block0.mlp.fc1": b0.mlp.fc1, "block0.mlp.act": b0.mlp.act,
             "block0.mlp.fc2": b0.mlp.fc2, "norm": enc.norm}
    named.update({f"block{i}": b for i, b in enumerate(enc.blocks)})
    for k, m in named.items():
        hooks.append(m.register_forward_hook(hook(k)))
    x = torch.from_numpy(c)[None, None]
    with torch.no_grad():
        out = enc(x)
    for h in hooks:
        h.remove()
    # which input values can influence the output? perturb the channels / samples outside the patch grid
    x2 = x.clone(); x2[..., 16:, :] = 5.0; x2[..., 480:] = -5.0
    with torch.no_grad():
        d = float((enc(x2) - out).abs().max())
    conv = enc.patch_embed.proj
    info = dict(
        recording=os.path.basename(f), clip_frame_indices=idx.tolist(), input=list(x.shape), shapes=rec,
        output=list(out.shape), conv_kernel=list(conv.kernel_size), conv_stride=list(conv.stride),
        conv_in=conv.in_channels, conv_out=conv.out_channels, depth=len(enc.blocks), num_heads=enc.num_heads,
        embed_dim=enc.embed_dim, head_dim=enc.embed_dim // enc.num_heads, mlp_hidden=b0.mlp.fc1.out_features,
        mlp_act=type(b0.mlp.act).__name__, ln_eps=b0.norm1.eps, pos_embed=list(enc.pos_embed.shape),
        pos_embed_trainable=bool(enc.pos_embed.requires_grad),
        params=dict(total=n_params(enc), trainable=n_params(enc, True), patch_embed=n_params(enc.patch_embed),
                    block=n_params(b0), attn=n_params(b0.attn), mlp=n_params(b0.mlp),
                    norms_in_block=n_params(b0.norm1) + n_params(b0.norm2), final_norm=n_params(enc.norm),
                    pos_embed=int(enc.pos_embed.numel())),
        max_abs_change_when_FZ_CZ_PZ_and_samples_480_499_perturbed=d,
        token_std_this_clip=float(out[0].std(0).mean()))
    del enc
    return info


R = {m: record(m) for m in ("vitm", "vitb")}
save_json("vjepa_recorded_shapes.json", R)
for m, r in R.items():
    print(m, r["input"], "->", r["shapes"]["patch_embed.proj (Conv3d)"], "->", r["output"], r["params"],
          "perturb diff", r["max_abs_change_when_FZ_CZ_PZ_and_samples_480_499_perturbed"])

# ---------------- draw the ViT-M diagram from the recorded numbers ----------------
r = R["vitm"]
S, P = r["shapes"], r["params"]
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(8.2, 9.6), dpi=170)
ax.set_xlim(0, 8.2); ax.set_ylim(0, 9.6); ax.set_axis_off()
fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
ax.text(0.1, 9.45, "Released ViT-M (4x30x4) encoder: shapes recorded on one real TUAB clip",
        fontsize=12, fontweight="bold", va="top")
ax.text(0.1, 9.12, "Shapes are (batch, ...). Batch = 1 clip here.", fontsize=10, color=INK2, va="top")

L, W = 2.25, 4.1
ys = [8.45, 7.2, 5.95, 4.85, 3.55, 2.3, 1.2]
k, s0 = r["conv_kernel"], S["patch_embed.proj (Conv3d)"]
spec = [
    ("Input clip", f"{shape_str(r['input'])}\n(batch, 1, frames, channels, samples)", "data", 0.9),
    ("Patch embedding (Conv3d)",
     f"kernel = stride = ({k[0]} frames, {k[1]} channels, {k[2]} samples)\n"
     f"1 -> {r['conv_out']} filters, {fmt_params(P['patch_embed'])} weights\nout {shape_str(s0)}", "frozen", 1.05),
    ("Flatten into tokens",
     f"{s0[2]} x {s0[3]} x {s0[4]} = {S['patch_embed (tokens)'][1]} tokens, {s0[1]} numbers each\n"
     f"out {shape_str(S['patch_embed (tokens)'])}", "data", 0.85),
    ("+ positional embedding",
     f"fixed sine-cosine table {shape_str(r['pos_embed'])}\n"
     + ("not learned (requires_grad = False)" if not r['pos_embed_trainable'] else "learned"), "data", 0.85),
    (f"Transformer block x {r['depth']}",
     f"each block: {shape_str(S['block0'])} -> {shape_str(S['block11'])}\n"
     f"{fmt_params(P['block'])} weights per block, {fmt_params(P['block'] * r['depth'])} in all", "frozen", 1.0),
    ("Final LayerNorm", f"out {shape_str(S['norm'])}, {P['final_norm']} weights", "frozen", 0.72),
    ("Output: one 384-number vector per token", f"{shape_str(r['output'])}  (= 'token embeddings')", "data", 0.8),
]
for (t, b, kind, h), y in zip(spec, ys):
    if t.startswith("Transformer"):
        for j in (2, 1):
            box(ax, L + 0.07 * j, y - 0.07 * j, W, h, "", kind)
    box(ax, L, y, W, h, t + "\n" + b, kind, fs=10)
for (y1, h1), (y2, h2) in zip([(y, s[3]) for y, s in zip(ys, spec)][:-1], [(y, s[3]) for y, s in zip(ys, spec)][1:]):
    arrow(ax, (L, y1 - h1 / 2), (L, y2 + h2 / 2))

# right: inside one transformer block (hooks on block 0)
X, Wr = 6.2, 2.95
ax.plot([L + W / 2 + 0.15, X - Wr / 2 - 0.05], [ys[4] + 0.5, 8.35], color=GREY, lw=1, ls="--")
ax.plot([L + W / 2 + 0.15, X - Wr / 2 - 0.05], [ys[4] - 0.5, 0.75], color=GREY, lw=1, ls="--")
ax.text(X, 8.72, "Inside one block (block 1 shown)", ha="center", fontsize=10.5, fontweight="bold")
hd, nh = r["head_dim"], r["num_heads"]
qkv = S["block0.attn.qkv"]
rs = [
    (8.2, "in", f"x {shape_str(S['block0'])}", "none", 0.42),
    (7.55, "LayerNorm 1", f"{shape_str(S['block0.norm1'])}", "frozen", 0.58),
    (6.35, "Multi-head self-attention",
     f"Linear {r['embed_dim']} -> {qkv[2]}: {shape_str(qkv)}\n= query, key, value for\n"
     f"{nh} heads x {hd} numbers\nmix tokens, Linear {r['embed_dim']} -> {r['embed_dim']}\n"
     f"out {shape_str(S['block0.attn'])}, {fmt_params(P['attn'])} weights", "frozen", 1.55),
    (5.1, "add (residual)", "x = x + attention(x)", "none", 0.55),
    (4.35, "LayerNorm 2", f"{shape_str(S['block0.norm2'])}", "frozen", 0.58),
    (3.05, "MLP (per token)",
     f"Linear {r['embed_dim']} -> {r['mlp_hidden']}: {shape_str(S['block0.mlp.fc1'])}\n"
     f"{r['mlp_act']}\nLinear {r['mlp_hidden']} -> {r['embed_dim']}: {shape_str(S['block0.mlp.fc2'])}\n"
     f"{fmt_params(P['mlp'])} weights", "frozen", 1.4),
    (1.8, "add (residual)", "x = x + MLP(x)", "none", 0.55),
    (1.05, "out", f"{shape_str(S['block0'])} -> next block", "none", 0.42),
]
for y, t, b, kind, h in rs:
    box(ax, X, y, Wr, h, (t + "\n" + b) if t not in ("in", "out") else b, kind, fs=9.5,
        bold_first=t not in ("in", "out"))
for (y1, *_a, h1), (y2, *_b, h2) in zip(rs[:-1], rs[1:]):
    arrow(ax, (X, y1 - h1 / 2), (X, y2 + h2 / 2))
# residual skip arrows: from the block input around the sub-layer into the "add" box
xr = X + Wr / 2
for (ya, yb) in [(rs[0][0], rs[3][0]), (rs[3][0], rs[6][0])]:
    ax.plot([xr, xr + 0.28, xr + 0.28], [ya, ya, yb], color=AQUA, lw=1.4)
    ax.annotate("", (xr + 0.02, yb), (xr + 0.28, yb), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.4))
ax.text(xr + 0.36, (rs[3][0] + rs[6][0]) / 2, "skip path (residual)", color=AQUA, fontsize=9, rotation=90,
        ha="center", va="center")

# legend + totals
from matplotlib.patches import Patch     # noqa: E402
ax.legend(handles=[Patch(fc=FILL["frozen"], ec=EDGE["frozen"], label="has learned weights (from the checkpoint)"),
                   Patch(fc=FILL["data"], ec=EDGE["data"], label="tensor / fixed operation, no learned weights")],
          loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=9.5, frameon=False, ncol=1)
ax.text(4.3, 0.2, f"Encoder total: {P['total']:,} numbers\n({P['trainable']:,} learned + "
        f"{P['pos_embed']:,} fixed position table)", fontsize=9.5, color=INK, ha="left", va="bottom")
out = f"{FIGS}/arch_vjepa_blocks.png"
fig.savefig(out, dpi=170)
print("saved", out)
