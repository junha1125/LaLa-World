"""Decode imagined latents to pixels with a decoder/ checkpoint trained for the checkpoint's vision encoder."""

import numpy as np
import torch
from PIL import Image

from decoder.model import load_decoder  # noqa: F401  (re-exported for the planning runners)


@torch.no_grad()
def save_pred_decode_strip(dec, cur_lat, pred_encs, goal_lat, cur_img, goal_img, path, banner=None, label=None):
    """[current px | current decoded | imagined +1.. decoded | goal decoded | goal px]."""
    if pred_encs.dim() != 4:
        return
    device = next(dec.parameters()).device
    lats = torch.cat([cur_lat, *pred_encs, goal_lat]).to(device).float()
    if lats.shape[1] != dec.n_patches:
        raise ValueError(
            f"decoder expects {dec.n_patches} patches, latents have {lats.shape[1]} (wrong decoder for this encoder)"
        )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        rec = dec(lats).float().clamp(0, 1)
    panels = (rec.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    side = panels.shape[1]
    cur_px, goal_px = (
        np.asarray(Image.fromarray(im).resize((side, side), Image.BILINEAR)) for im in (cur_img, goal_img)
    )
    strip = np.concatenate([cur_px, *panels, goal_px], axis=1)
    (banner(strip, label) if banner else Image.fromarray(strip)).save(path)
