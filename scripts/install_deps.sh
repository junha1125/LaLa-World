#!/usr/bin/env bash
# Ubuntu + NVIDIA PyTorch container (nvcr.io/nvidia/pytorch:25.06-py3) or any Python >= 3.10 env with torch installed.
#   bash scripts/install_deps.sh
set -euo pipefail
cd "$(dirname "$0")/.."
unset PIP_CONSTRAINT || true

if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git curl ca-certificates ffmpeg zip unzip libwebp-dev libjpeg-dev zlib1g-dev libosmesa6 libegl1 libgl1 libglvnd0
  rm -rf /var/lib/apt/lists/*
fi

python -m pip install --upgrade pip setuptools wheel
python -c "import torch" 2>/dev/null || python -m pip install "torch>=2.8" torchvision
# keep the installed numpy major version (the NGC image ships numpy 1.26)
NUMPY_PIN=$(python -c "import numpy; print('numpy==' + numpy.__version__)")
python -m pip install -c <(echo "$NUMPY_PIN") -e ".[sim,data,decoder,dev]"
python -c "import torch, transformers, fla, mujoco, metaworld; print('torch', torch.__version__, '| transformers', transformers.__version__, '| fla', fla.__version__)"
