#!/bin/bash
# Public weights the baseline starts from (~12 GB): RAE encoder/decoder, the ImageNet stage-2 DiT, Qwen3-0.6B.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
hf download nyu-visionx/RAEv2-models --local-dir pretrained_models \
  --include "stage1/imagenet/dinov3l-k7/*" "stage2/imagenet/dinov3l-k7/*" "encoders/dinov3/dinov3_vitl16_*"
hf download Qwen/Qwen3-0.6B --local-dir pretrained_models/Qwen3-0.6B
ls -la pretrained_models/stage1/imagenet/dinov3l-k7 pretrained_models/stage2/imagenet/dinov3l-k7 pretrained_models/encoders/dinov3
