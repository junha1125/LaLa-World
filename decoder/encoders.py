"""Frozen vision encoders built through app.lala_world, so the features are exactly what LaLa-World predicts."""

from app.lala_world.models import build_vision_encoder


def build_frozen_encoder(vision_cfg, device):
    enc = build_vision_encoder(dict(vision_cfg)).to(device).eval()
    enc.requires_grad_(False)
    gh, gw = enc.grid_size
    if gh != gw:
        raise ValueError(f"non-square patch grid {enc.grid_size}")
    return enc


def normalize_pixels(x01, enc):
    """[0, 1] pixels -> the encoder's own normalization (as app.lala_world.data.default_transform)."""
    mean = x01.new_tensor(enc.image_mean).view(1, 3, 1, 1)
    std = x01.new_tensor(enc.image_std).view(1, 3, 1, 1)
    return (x01 - mean) / std
