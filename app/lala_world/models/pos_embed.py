"""3D (t, h, w) position ids and their additive sin-cos embedding.

Context frame j: t = step after text j-1 ends, (h, w) = patch grid. Text token: (t, t, t) climbing by
one per token. Queries: the target frame's step, (h, w) = target grid. Pad slots keep 0 (masked).
"""

import torch
import torch.nn as nn

AXIS_RATIO = (0.25, 0.375, 0.375)  # share of the embedding width given to t, h, w


def _grid_ids(gh, gw, n=None):
    n = gh * gw if n is None else n
    hh = torch.arange(gh).repeat_interleave(gw)[:n]
    ww = torch.arange(gw).repeat(gh)[:n]
    return torch.stack([torch.zeros(n, dtype=torch.long), hh, ww])


def position_ids_3d_queries(grid_hw, text_lens, max_text_len, n_queries):
    """(3, B, S) for the left-padded layout [pad | vis | text | queries], S = max_text_len + N + n_queries."""
    gh, gw = grid_hw
    vis = _grid_ids(gh, gw)
    n_vis = vis.shape[1]
    text_lens = [int(x) for x in text_lens]
    B, S = len(text_lens), max_text_len + n_vis + n_queries
    pos = torch.zeros(3, B, S, dtype=torch.long)
    qgrid = _grid_ids(gh, gw, n_queries)
    for i, L in enumerate(text_lens):
        off = max_text_len - L
        pos[:, i, off : off + n_vis] = vis
        pos[:, i, off + n_vis : off + n_vis + L] = torch.arange(1, L + 1).expand(3, L)
        pos[:, i, S - n_queries :] = qgrid
        pos[0, i, S - n_queries :] = L + 1
    return pos


def position_ids_3d_seq(grid_hw, text_lens, ctx_len, n_queries, total_len):
    """(3, B, S) for the interleaved layout [pad | f_0 | x_0 | ... | f_{d-1} | x_{d-1} | queries]."""
    gh, gw = grid_hw
    vis = _grid_ids(gh, gw)
    n_vis = vis.shape[1]
    qgrid = _grid_ids(gh, gw, n_queries)
    pos = torch.zeros(3, len(ctx_len), total_len, dtype=torch.long)
    for i in range(len(ctx_len)):
        d = int(ctx_len[i])
        lens = [int(x) for x in text_lens[i][:d]]
        off = total_len - (d * n_vis + sum(lens) + n_queries)
        t = 0
        for j in range(d):
            pos[:, i, off : off + n_vis] = vis
            pos[0, i, off : off + n_vis] = t
            off += n_vis
            pos[:, i, off : off + lens[j]] = torch.arange(t + 1, t + 1 + lens[j]).expand(3, lens[j])
            off += lens[j]
            t += lens[j] + 1
        pos[:, i, off : off + n_queries] = qgrid
        pos[0, i, off : off + n_queries] = t
    return pos


def _split_dims(dim, ratio):
    parts = [2 * int(dim * r / 2) for r in ratio]
    parts[0] += dim - sum(parts)
    return parts


def _sincos_1d(pos, dim, base):
    half = dim // 2
    inv = base ** (-torch.arange(half, device=pos.device, dtype=torch.float32) / half)
    ang = pos.float()[..., None] * inv
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class Additive3DSinCos(nn.Module):
    """Fixed per-axis sin-cos features with one learnable gain (initialised to the token-embedding RMS)."""

    def __init__(self, dim, ratio=AXIS_RATIO, base=10000.0, ref_scale=1.0):
        super().__init__()
        self.dims = _split_dims(dim, ratio)
        self.base = base
        self.gain = nn.Parameter(torch.tensor(float(ref_scale)))

    def forward(self, position_ids):
        parts = [_sincos_1d(position_ids[a], d, self.base) for a, d in enumerate(self.dims)]
        return self.gain * torch.cat(parts, dim=-1)
