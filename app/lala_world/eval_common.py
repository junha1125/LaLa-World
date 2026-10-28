"""Rebuild a trained LaLa-World (encoder + predictor) from a checkpoint."""

import torch

from app.lala_world.models import LaLaWorld, build_llm_backbone, build_vision_encoder


def load_model(ckpt_path, device):
    """-> (encoder, predictor, cfg, epoch). The encoder is the frozen one named in the checkpoint config;
    the predictor weights are the checkpoint's predictor.* entries."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    encoder = build_vision_encoder(cfg["model"]["vision"]).to(device)
    backbone = build_llm_backbone(cfg["model"]["llm"])
    predictor = LaLaWorld(
        backbone, vis_dim=encoder.embed_dim, target_dim=encoder.embed_dim, grid_hw=encoder.grid_size
    ).to(device)
    state = {k[len("predictor.") :]: v for k, v in ckpt["model"].items() if k.startswith("predictor.")}
    predictor.load_state_dict(state, strict=True)
    encoder.eval()
    predictor.eval()
    return encoder, predictor, cfg, int(ckpt["epoch"])
