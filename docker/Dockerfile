FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive PIP_CONSTRAINT= HF_HUB_ENABLE_HF_TRANSFER=0
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates ffmpeg zip unzip tmux htop \
        libwebp-dev libjpeg-dev zlib1g-dev libosmesa6 libegl1 libgl1 libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/LaLa-World
COPY . .
RUN python -m pip install --upgrade pip setuptools wheel \
    && echo "numpy==1.26.4" > /tmp/constraints.txt \
    && python -m pip install -c /tmp/constraints.txt -e ".[sim,data,decoder,dev]"
