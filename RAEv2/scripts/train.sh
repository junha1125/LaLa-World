#!/bin/bash
# bash scripts/train.sh configs/rae_xl.yaml [extra train.py args]   (re-running resumes from <run_dir>/checkpoints/latest.pt)
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?config yaml}; shift || true
export PYTHONPATH=src
export TMPDIR=${TMPDIR:-/tmp}
torchrun --nproc_per_node=${NGPU:-8} --master_port=${MASTER_PORT:-29582} src/wm/train.py --config "$CONFIG" --compile "$@"
