"""DiT (RAEv2 DiTwDDTHeadIG) with image-to-image + text conditioning, initialised from the public stage-2 checkpoint."""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from stage2.models.DDT import DiTwDDTHeadIG
from wm.cond import I2ITextCondEmbedder, i2i_rope

STAGE2_KEEP_DROP = ("ctx_embedder.", "enc_rope.")


@dataclass
class ConditioningArch:
    num_t_tokens: int = 4
    num_c_tokens: int = 8


def build_dit(model_cfg: dict, text_dim: int, n_text_tokens: int) -> DiTwDDTHeadIG:
    input_size = int(model_cfg.get("input_size", 16))
    in_channels = int(model_cfg.get("in_channels", 1024))
    hidden = list(model_cfg.get("hidden_size", [1440, 2048]))
    depth = list(model_cfg.get("depth", [28, 2]))
    heads = list(model_cfg.get("num_heads", [20, 16]))
    num_t = int(model_cfg.get("num_t_tokens", 4))
    num_patches = input_size * input_size
    model = DiTwDDTHeadIG(
        base_model_depth=int(model_cfg.get("base_model_depth", 8)),
        input_size=input_size,
        in_channels=in_channels,
        patch_size=[1, 1],
        hidden_size=hidden,
        depth=depth,
        num_heads=heads,
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        condition_type="label",  # placeholder embedder, replaced below
        num_classes=1,
        context_dim=text_dim,
        cond_arch=ConditioningArch(num_t_tokens=num_t, num_c_tokens=num_patches + n_text_tokens),
    )
    if model.base_model_depth > model.num_enc_blocks:
        raise ValueError(f"base_model_depth {model.base_model_depth} > encoder depth {model.num_enc_blocks}")
    model.ctx_embedder = I2ITextCondEmbedder(hidden[0], in_channels, text_dim, n_text_tokens, input_size)
    model.enc_rope = i2i_rope(hidden[0] // heads[0], num_patches, num_t, n_text_tokens)
    model.ctx_embedder.init_ctx_from_patch_embed(model.s_embedder)
    return model


def _remap_truncated(sd: Dict[str, torch.Tensor], n_enc_ckpt: int, n_enc: int, n_dec: int):
    """Keep encoder blocks 0..n_enc-1 and move the decoder blocks right after them."""
    out = {}
    for k, v in sd.items():
        if k.startswith("blocks."):
            i = int(k.split(".")[1])
            rest = k.split(".", 2)[2]
            if i < n_enc_ckpt:
                if i >= n_enc:
                    continue
                out[f"blocks.{i}.{rest}"] = v
            else:
                j = i - n_enc_ckpt
                if j >= n_dec:
                    continue
                out[f"blocks.{n_enc + j}.{rest}"] = v
        else:
            out[k] = v
    return out


def load_stage2_init(model: DiTwDDTHeadIG, ckpt_path: str, which: str = "ema", n_enc_ckpt: int = 28, logger=None):
    """Everything transfers except the conditioning embedder and the RoPE tables."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = {k: v for k, v in ck[which].items() if not k.startswith(STAGE2_KEEP_DROP)}
    if model.num_enc_blocks != n_enc_ckpt:
        sd = _remap_truncated(sd, n_enc_ckpt, model.num_enc_blocks, model.num_dec_blocks)
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith(STAGE2_KEEP_DROP)]
    if res.unexpected_keys or missing:
        raise RuntimeError(f"stage2 init mismatch: missing={missing[:8]} unexpected={res.unexpected_keys[:8]}")
    msg = (f"stage2 init from {ckpt_path} [{which}]: {sum(v.numel() for v in sd.values())/1e6:.1f}M params loaded; "
           f"fresh: {', '.join(sorted(set(k.split('.')[0] for k in res.missing_keys)))}")
    (logger.info if logger else print)(msg)
    del ck
    return msg


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_x_t1(model, context, attn_mask, latent_size, generator: Optional[torch.Generator] = None, noise=None):
    """One-shot x-prediction from pure noise (t = 1): the model's estimate of E[x | context, text]."""
    B = context["ctx"].shape[0]
    if noise is None:
        noise = torch.randn(B, *latent_size, device=context["ctx"].device, generator=generator)
    out = model(noise, torch.ones(B, device=noise.device), context=context, attn_mask=attn_mask)
    return out[0] if isinstance(out, tuple) else out


def time_shift(misc_cfg: dict, latent_size) -> float:
    dim = misc_cfg.get("time_dist_shift_dim") or math.prod(latent_size)
    return math.sqrt(dim / misc_cfg.get("time_dist_shift_base", 4096))
