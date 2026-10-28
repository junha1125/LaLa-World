"""Test models: hub ids, resolved through $LALA_MODELS/<name> when a local copy exists (app.lala_world.models.hub)."""

import os

import pytest
import torch

from app.lala_world.models.hub import resolve_model_id

SIGLIP2 = "google/siglip2-base-patch16-224"
QWEN35 = "Qwen/Qwen3.5-0.8B"
DINOV2 = {"kind": "dinov2", "model_id": "facebookresearch/dinov2", "hub_model": "dinov2_vits14", "image_size": 224}


def _available(model_id):
    if os.path.isdir(resolve_model_id(model_id)):
        return True
    return not (os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))


needs_siglip2 = pytest.mark.skipif(not _available(SIGLIP2), reason="siglip2 weights not available offline")
needs_qwen = pytest.mark.skipif(
    not (_available(QWEN35) and torch.cuda.is_available()), reason="Qwen3.5 needs the weights and a GPU (fla kernels)"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
