"""LaLa-World predictor: (context patch latents, action texts) -> next-frame patch latents.

connector -> left-padded sequence [pad | f_0 | x_0 | ... | f_{d-1} | x_{d-1} | queries] -> Qwen3.5 -> out_head.
Queries are copies of one learned token; their target location comes from the 3D position embedding.
"""

import torch
import torch.nn as nn

from app.lala_world.models.connector import MLP2x
from app.lala_world.models.pos_embed import Additive3DSinCos, position_ids_3d_queries, position_ids_3d_seq


def pack_left(vis_emb, txt_emb, text_lens, q_emb):
    """[pad | vis | text | queries] with real tokens contiguous. txt_emb is right-padded by the tokenizer."""
    B, n_vis, D = vis_emb.shape
    L, n_q = txt_emb.shape[1], q_emb.shape[1]
    device = vis_emb.device
    source = torch.cat([vis_emb, txt_emb], dim=1)
    offsets = (L - text_lens).long().to(device)
    slots = torch.arange(n_vis + L, device=device)
    idx = (slots[None, :] - offsets[:, None]).clamp(min=0)
    head = source.gather(1, idx[..., None].expand(-1, -1, D))
    x = torch.cat([head, q_emb], dim=1)
    attn_mask = (torch.arange(n_vis + L + n_q, device=device)[None, :] >= offsets[:, None]).long()
    return x, attn_mask


def pack_left_seq(vis_emb, txt_emb, text_lens, ctx_len, q_emb):
    """Interleaved [pad | f_0 | x_0 | ... | f_{d-1} | x_{d-1} | queries]; blocks beyond ctx_len are dropped."""
    B, C, n_vis, D = vis_emb.shape
    L, n_q = txt_emb.shape[2], q_emb.shape[1]
    device = vis_emb.device
    source = torch.cat([vis_emb, txt_emb], dim=2).reshape(B, C * (n_vis + L), D)
    source = torch.cat([source, q_emb], dim=1)
    q_base = C * (n_vis + L)
    lens, ctx = text_lens.long().cpu(), ctx_len.long().cpu()
    slot = torch.arange(C)[None, :] < ctx[:, None]
    real = (ctx * n_vis + (lens * slot).sum(dim=1) + n_q).tolist()
    S = max(real)
    idx = torch.zeros(B, S, dtype=torch.long)
    for i in range(B):
        pos = S - real[i]
        for j in range(int(ctx[i])):
            base = j * (n_vis + L)
            idx[i, pos : pos + n_vis] = torch.arange(base, base + n_vis)
            pos += n_vis
            lij = int(lens[i, j])
            idx[i, pos : pos + lij] = torch.arange(base + n_vis, base + n_vis + lij)
            pos += lij
        idx[i, pos:] = torch.arange(q_base, q_base + n_q)
    attn_mask = (torch.arange(S)[None, :] >= (S - torch.tensor(real))[:, None]).long().to(device)
    x = source.gather(1, idx.to(device)[..., None].expand(-1, -1, D))
    return x, attn_mask


class LaLaWorld(nn.Module):
    def __init__(self, backbone, vis_dim, target_dim, grid_hw, n_queries=None, query_init_std=0.02):
        super().__init__()
        self.backbone = backbone
        self.grid_hw = tuple(grid_hw)
        self.n_vis = self.grid_hw[0] * self.grid_hw[1]
        self.n_queries = n_queries if n_queries is not None else self.n_vis
        dim = backbone.hidden_size
        self.connector = MLP2x(vis_dim, dim)
        self.out_head = MLP2x(dim, target_dim)
        self.query_token = nn.Parameter(torch.randn(dim) * query_init_std)
        self.additive_pe = Additive3DSinCos(dim, ref_scale=backbone.embed_rms)

    def forward(self, vis_patches, input_ids, text_mask, ctx_len=None):
        """Single frame: vis (B,N,Dv), input_ids/text_mask (B,L). Multi-frame: vis (B,C,N,Dv),
        input_ids/text_mask (B,C,L), ctx_len (B). Returns (B, Nq, target_dim)."""
        if input_ids.ndim == 3:
            if input_ids.shape[1] == 1:
                vis_patches, input_ids, text_mask = vis_patches[:, 0], input_ids[:, 0], text_mask[:, 0]
            else:
                return self._forward_seq(vis_patches, input_ids, text_mask, ctx_len)
        B, L = input_ids.shape
        scale = self.backbone.embed_scale
        vis_emb = self.connector(vis_patches) * scale
        txt_emb = self.backbone.embed_text(input_ids).to(vis_emb.dtype)
        text_lens = text_mask.sum(dim=1)
        q_emb = (self.query_token * scale).to(vis_emb.dtype).expand(B, self.n_queries, -1)
        x, attn_mask = pack_left(vis_emb, txt_emb, text_lens, q_emb)
        pos3d = position_ids_3d_queries(self.grid_hw, text_lens.tolist(), L, self.n_queries)
        h = self._run_backbone(x, attn_mask, pos3d.to(x.device))
        return self.out_head(h[:, -self.n_queries :])

    def _forward_seq(self, vis_patches, input_ids, text_mask, ctx_len):
        if ctx_len is None:
            raise ValueError("multi-frame input needs ctx_len")
        B, C, L = input_ids.shape
        scale = self.backbone.embed_scale
        vis_emb = self.connector(vis_patches) * scale
        txt_flat = self.backbone.embed_text(input_ids.reshape(B * C, L))
        txt_emb = txt_flat.reshape(B, C, L, txt_flat.shape[-1]).to(vis_emb.dtype)
        slot = torch.arange(C, device=text_mask.device)[None, :] < ctx_len[:, None]
        text_lens = text_mask.sum(dim=2) * slot.long()
        q_emb = (self.query_token * scale).to(vis_emb.dtype).expand(B, self.n_queries, -1)
        x, attn_mask = pack_left_seq(vis_emb, txt_emb, text_lens, ctx_len, q_emb)
        pos3d = position_ids_3d_seq(self.grid_hw, text_lens.cpu(), ctx_len.cpu(), self.n_queries, x.shape[1])
        h = self._run_backbone(x, attn_mask, pos3d.to(x.device))
        return self.out_head(h[:, -self.n_queries :])

    def _run_backbone(self, x, attn_mask, pos3d):
        x = x + self.additive_pe(pos3d).to(x.dtype)
        return self.backbone(x, attn_mask)
