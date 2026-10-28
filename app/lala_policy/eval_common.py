"""Rebuild a trained LaLa-Policy from its checkpoint."""

import torch

from app.lala_policy.train import build_predictor
from app.lala_world.models import build_vision_encoder


def load_policy_model(ckpt_path, device):
    """-> (encoder, policy, cfg, stats, epoch); `stats` is the action / proprio min-max dict used for scaling."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if "pv_pred" not in cfg["model"]["policy"]:
        raise ValueError(f"{ckpt_path}: checkpoint config has no model.policy.pv_pred (unsupported layout)")
    encoder = build_vision_encoder(cfg["model"]["vision"]).to(device)
    predictor = build_predictor(cfg["model"], encoder).to(device)
    state = {k[len("predictor.") :]: v for k, v in ckpt["model"].items() if k.startswith("predictor.")}
    predictor.load_state_dict(state, strict=True)
    encoder.eval()
    predictor.eval()
    return encoder, predictor, cfg, ckpt.get("stats"), int(ckpt["epoch"])
