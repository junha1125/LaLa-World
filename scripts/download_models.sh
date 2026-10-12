#!/usr/bin/env bash
# Pretrained encoders / LLM used by the configs, downloaded to ./hf-models (DINOv3 is gated: `hf auth login` first).
#   bash scripts/download_models.sh            # Qwen3.5-0.8B, DINOv2 ViT-S/14, DINOv3 ViT-L/16, SigLIP2 So400M/14
#   bash scripts/download_models.sh all        # + DINOv3 ViT-S/16, SigLIP2 B/16
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=0
M=${LALA_MODELS:-$PWD/hf-models}
mkdir -p "$M"
hf download Qwen/Qwen3.5-0.8B --local-dir "$M/Qwen3.5-0.8B"
hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir "$M/dinov3-vitl16-pretrain-lvd1689m"
hf download google/siglip2-so400m-patch14-224 --local-dir "$M/siglip2-so400m-patch14-224"
if [ "${1:-}" = all ]; then
  hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir "$M/dinov3-vits16-pretrain-lvd1689m"
  hf download google/siglip2-base-patch16-224 --local-dir "$M/siglip2-base-patch16-224"
fi
# DINOv2 ViT-S/14 (torch.hub): code checkout + weights
export TORCH_HOME=${TORCH_HOME:-$M/torch}
python - <<'PY'
import os, torch
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
hub = torch.hub.get_dir()
print("dinov2 repo:", os.path.join(hub, "facebookresearch_dinov2_main"))
print("dinov2 weights:", os.path.join(hub, "checkpoints", "dinov2_vits14_pretrain.pth"))
PY
ls "$M"
