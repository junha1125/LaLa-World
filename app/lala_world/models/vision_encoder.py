"""Frozen vision encoders. Every wrapper exposes
forward(pixels[B,3,H,W]) -> {"patches": (B, N, D), "pooled": (B, D)} and
embed_dim / num_patches / grid_size / image_size / image_mean / image_std."""

import json
import os
import warnings

import torch
import torch.nn as nn

from app.lala_world.models.hub import resolve_model_id, torch_hub_checkout

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _preproc_stats(model_id, default=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    path = os.path.join(model_id, "preprocessor_config.json")
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        return tuple(p.get("image_mean", default[0])), tuple(p.get("image_std", default[1]))
    try:
        from transformers import AutoImageProcessor

        p = AutoImageProcessor.from_pretrained(model_id)
        return tuple(p.image_mean), tuple(p.image_std)
    except Exception:
        return default


class _Frozen(nn.Module):
    frozen = True

    def _freeze(self):
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        return super().train(False)


class SigLIP2Encoder(_Frozen):
    def __init__(self, model_id):
        super().__init__()
        from transformers import SiglipVisionModel

        model_id = resolve_model_id(model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id)
        cfg = self.model.config
        self.embed_dim, self.image_size, self.patch_size = cfg.hidden_size, cfg.image_size, cfg.patch_size
        g = cfg.image_size // cfg.patch_size
        self.grid_size, self.num_patches = (g, g), g * g
        self.image_mean, self.image_std = _preproc_stats(model_id)
        self._freeze()

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return {"patches": out.last_hidden_state, "pooled": out.pooler_output}


class DINOv3Encoder(_Frozen):
    def __init__(self, model_id):
        super().__init__()
        from transformers import DINOv3ViTModel

        model_id = resolve_model_id(model_id)
        self.model = DINOv3ViTModel.from_pretrained(model_id)
        cfg = self.model.config
        self.embed_dim, self.image_size, self.patch_size = cfg.hidden_size, cfg.image_size, cfg.patch_size
        self.num_prefix = 1 + cfg.num_register_tokens
        g = cfg.image_size // cfg.patch_size
        self.grid_size, self.num_patches = (g, g), g * g
        self.image_mean, self.image_std = _preproc_stats(model_id, (IMAGENET_MEAN, IMAGENET_STD))
        self._freeze()

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return {"patches": out.last_hidden_state[:, self.num_prefix :], "pooled": out.pooler_output}


class DINOv2Encoder(_Frozen):
    """facebookresearch/dinov2 torch.hub backbone (the DINO-WM / JEPA-WMs encoder).

    model_id: local hub checkout directory or "facebookresearch/dinov2"; weights: optional local
    state-dict file (torch.hub downloads the weights otherwise)."""

    def __init__(
        self, model_id, hub_model="dinov2_vits14", weights=None, image_size=224, image_mean=None, image_std=None
    ):
        super().__init__()
        warnings.filterwarnings("ignore", message="xFormers is not available")
        model_id = resolve_model_id(model_id)
        if weights is not None and not os.path.exists(weights):
            weights = None
        if not os.path.isdir(model_id):
            repo, hub_weights = torch_hub_checkout("facebookresearch_dinov2_main", f"{hub_model}_pretrain.pth")
            if repo is not None:
                model_id, weights = repo, weights or hub_weights
        if os.path.isdir(model_id):
            self.model = torch.hub.load(model_id, hub_model, source="local", pretrained=weights is None)
        else:
            self.model = torch.hub.load(model_id, hub_model, source="github", pretrained=True, skip_validation=True)
        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        self.embed_dim, self.patch_size = self.model.embed_dim, self.model.patch_size
        if image_size % self.patch_size:
            raise ValueError(f"image_size {image_size} must be a multiple of the patch size {self.patch_size}")
        self.image_size = image_size
        g = image_size // self.patch_size
        self.grid_size, self.num_patches = (g, g), g * g
        self.image_mean = tuple(image_mean) if image_mean is not None else IMAGENET_MEAN
        self.image_std = tuple(image_std) if image_std is not None else IMAGENET_STD
        self._freeze()

    def forward(self, pixel_values):
        out = self.model.forward_features(pixel_values)
        return {"patches": out["x_norm_patchtokens"], "pooled": out["x_norm_clstoken"]}


ENCODER_REGISTRY = {"siglip2": SigLIP2Encoder, "dinov3": DINOv3Encoder, "dinov2": DINOv2Encoder}


def build_vision_encoder(cfg):
    """cfg: {kind: siglip2|dinov3|dinov2, model_id, ...kind-specific keys (dinov2: hub_model, weights, image_size)}."""
    cfg = dict(cfg)
    kind = cfg.pop("kind", "siglip2")
    if kind not in ENCODER_REGISTRY:
        raise ValueError(f"unknown vision encoder kind {kind!r}; known: {sorted(ENCODER_REGISTRY)}")
    if not cfg.pop("frozen", True) or not cfg.pop("pretrained", True):
        raise ValueError("the vision encoder is always frozen and pretrained in this release")
    return ENCODER_REGISTRY[kind](**cfg)
