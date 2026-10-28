"""ViT pixel decoder: frozen-encoder last-layer patch latents (B, N, D) -> image (B, 3, H, W) in [0, 1].

12 pre-LN transformer blocks (SigLIP2 block shape, learned position embeddings) at the encoder's width, a linear
patch head and unpatchify. `load_decoder` rebuilds a trained decoder (EMA weights by default) from a checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, -1))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTPixelDecoder(nn.Module):
    def __init__(self, feat_dim, dim, depth, num_heads, grid, patch_px, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.grid = grid
        self.patch_px = patch_px
        self.n_patches = grid * grid
        self.in_norm = nn.LayerNorm(feat_dim)
        self.proj = nn.Linear(feat_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.blocks = nn.ModuleList(Block(dim, num_heads, mlp_ratio, qk_norm) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_px * patch_px * 3)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        x = self.proj(self.in_norm(feats)) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.unpatchify(self.head(self.norm(x)))

    def unpatchify(self, x):
        B, N, _ = x.shape
        g, p = self.grid, self.patch_px
        x = x.reshape(B, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, 3, g * p, g * p)


def _to_current_layout(sd):
    """Checkpoints trained with nn.MultiheadAttention use in_proj / out_proj names; same math, renamed."""
    ren = {".attn.in_proj_": ".attn.qkv.", ".attn.out_proj.": ".attn.proj."}
    out = {}
    for k, v in sd.items():
        for a, b in ren.items():
            k = k.replace(a, b)
        out[k] = v
    return out


def load_decoder(ckpt_path, device, weights="ema"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = ck["cfg"]["decoder"]
    heads = d.get("num_heads") or d["dim"] // 64
    dec = ViTPixelDecoder(
        ck["decoder"]["proj.weight"].shape[1],
        d["dim"],
        d["depth"],
        heads,
        ck["grid"],
        ck["patch_px"],
        d.get("mlp_ratio", 4.0),
        d.get("qk_norm", False),
    )
    dec.load_state_dict(_to_current_layout(ck[weights]))
    print(
        f"pixel decoder: {ckpt_path} ({weights}, step {ck.get('step')}, "
        f"{ck['grid']}x{ck['grid']} patches -> {ck['grid'] * ck['patch_px']}px)"
    )
    return dec.to(device).eval()
