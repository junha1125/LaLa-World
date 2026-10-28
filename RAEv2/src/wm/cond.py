"""In-context conditioning of the DiT: [noised latent][timestep][context-frame latent patches][text tokens]."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from stage2.models.model_utils import RMSNorm, RoPE


class I2ITextCondEmbedder(nn.Module):
    def __init__(self, hidden_size: int, latent_channels: int, text_dim: int, n_text_tokens: int, latent_hw: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_ctx_tokens = latent_hw * latent_hw
        self.n_text_tokens = int(n_text_tokens)
        self.ctx_embed = nn.Conv2d(latent_channels, hidden_size, kernel_size=1, stride=1)
        self.ctx_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.text_norm = RMSNorm(text_dim)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)

    @property
    def n_tokens(self):
        return self.n_ctx_tokens + self.n_text_tokens

    def init_ctx_from_patch_embed(self, patch_embed: nn.Module):
        w = patch_embed.proj.weight.data
        if tuple(w.shape) != tuple(self.ctx_embed.weight.shape):
            raise ValueError(f"s_embedder {tuple(w.shape)} vs ctx_embed {tuple(self.ctx_embed.weight.shape)}")
        self.ctx_embed.weight.data.copy_(w)
        self.ctx_embed.bias.data.copy_(patch_embed.proj.bias.data)

    def forward(self, y: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx = self.ctx_embed(y["ctx"]).flatten(2).transpose(1, 2) + self.ctx_type
        txt = self.text_proj(self.text_norm(y["text"]))
        if txt.shape[1] != self.n_text_tokens:
            raise ValueError(f"expected {self.n_text_tokens} text tokens, got {txt.shape[1]}")
        return torch.cat([ctx, txt], dim=1)


def i2i_rope(head_dim: int, num_patches: int, num_t_tokens: int, n_text_tokens: int) -> RoPE:
    """Context patches reuse the 2-D grid angles of the noised latent; text tokens get no position."""
    rope = RoPE(head_dim, num_patches, num_t_tokens + num_patches + n_text_tokens)
    grid_cos, grid_sin = rope.freqs_cos[:num_patches].clone(), rope.freqs_sin[:num_patches].clone()
    s = num_patches + num_t_tokens
    rope.freqs_cos[s : s + num_patches] = grid_cos
    rope.freqs_sin[s : s + num_patches] = grid_sin
    return rope


def make_context(ctx_latents: torch.Tensor, text_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"ctx": ctx_latents, "text": text_tokens}


def apply_text_dropout(context, attn_mask, null_text, null_mask, p: float):
    """Per-sample replace the text tokens by the encoding of "" (the context frame is always kept)."""
    if p <= 0:
        return context, attn_mask
    B = context["text"].shape[0]
    drop = torch.rand(B, device=context["text"].device) < p
    text = torch.where(drop[:, None, None], null_text.expand(B, -1, -1), context["text"])
    mask = torch.where(drop[:, None], null_mask.expand(B, -1), attn_mask)
    return {"ctx": context["ctx"], "text": text}, mask
