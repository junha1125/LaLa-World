#!/bin/bash
# 8-GPU decoder training:  bash decoder/launch.sh decoder/configs/dinov2s.yaml   (same config again = resume)
set -u
CFG=${1:?usage: launch.sh <config.yaml>}
NGPU=${NGPU:-8}
RUN_DIR=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_dir'])")
mkdir -p "$RUN_DIR"
echo "=== $(date -u +%F' '%T) launch $CFG -> $RUN_DIR" >> "$RUN_DIR/train.log"
exec python -m torch.distributed.run --standalone --nproc_per_node "$NGPU" -m decoder.train --config "$CFG" >> "$RUN_DIR/train.log" 2>&1
