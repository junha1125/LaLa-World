"""Model ids may be local directories or Hugging Face hub ids.

A hub id, or a local path that does not exist on this machine (e.g. inside a checkpoint trained elsewhere), is
resolved to $LALA_MODELS/<name> (default ./hf-models, where scripts/download_models.sh puts the weights) when that
directory exists, otherwise to the public hub id.
"""

import os

HUB_IDS = {
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "siglip2-base-patch16-224": "google/siglip2-base-patch16-224",
    "siglip2-large-patch16-256": "google/siglip2-large-patch16-256",
    "siglip2-so400m-patch14-224": "google/siglip2-so400m-patch14-224",
    "dinov3-vits16-pretrain-lvd1689m": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3-vitb16-pretrain-lvd1689m": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vitl16-pretrain-lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebookresearch_dinov2_main": "facebookresearch/dinov2",
}


def resolve_model_id(model_id):
    model_id = str(model_id)
    if os.path.isdir(model_id):
        return model_id
    name = os.path.basename(model_id.rstrip("/"))
    root = os.environ.get("LALA_MODELS", "hf-models")
    for cand in (os.path.join(root, name), os.path.join(root, model_id)):
        if os.path.isdir(cand):
            return cand
    return HUB_IDS.get(name, model_id)


def torch_hub_checkout(repo_dir_name, weights_name):
    """(local torch.hub checkout dir, weights file) if present under $LALA_MODELS/torch/hub or torch.hub's cache."""
    import torch

    hubs = [os.path.join(os.environ.get("LALA_MODELS", "hf-models"), "torch", "hub"), torch.hub.get_dir()]
    for hub in hubs:
        repo = os.path.join(hub, repo_dir_name)
        if os.path.isdir(repo):
            weights = os.path.join(hub, "checkpoints", weights_name)
            return repo, (weights if os.path.exists(weights) else None)
    return None, None


def model_name(model_id):
    """Comparable short name of a model id (local path or hub id)."""
    return os.path.basename(str(model_id).rstrip("/"))
