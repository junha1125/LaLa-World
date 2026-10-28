#!/bin/bash
# Planning evaluation of one snapshot on 8 GPUs (episode-sharded), merged into <out_dir>/summary.md.
#   bash scripts/eval_planning.sh <snapshot.pt> <out_dir>
# Protocol: wall 96 / pusht 96 / point_maze 24 (open-loop), metaworld reach + reach-wall 24 (MPC), DROID 64.
# Needs LALA_DATA_ROOT (the LaLa-World data root) and the planning simulators of the main README.
# NFE (denoising steps), CFG (text guidance scale) and COST (latent | pixel) can be overridden from the environment.
set -u
CK=$(realpath "${1:?snapshot.pt}"); OUT=$(realpath -m "${2:?out_dir}")
RAE=$(cd "$(dirname "$0")/.." && pwd)
LALA=${LALA_ROOT:-$(dirname "$RAE")}
NFE=${NFE:-1}; CFG=${CFG:-2.0}; COST=${COST:-latent}
mkdir -p "$OUT"
exec >> "$OUT/eval_planning.log" 2>&1
echo "=== $(date) eval_planning $CK -> $OUT (nfe $NFE cfg $CFG cost $COST)"
cd "$LALA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAE_ROOT="$RAE"
PLAN=${PLAN_MODULE:-evals.lala_world}
COMMON="--wm rae --ckpt $CK --rae-nfe $NFE --rae-cfg $CFG --rae-cost $COST --video-episodes 0 --out $OUT"
shard() {
  local g=$1
  export CUDA_VISIBLE_DEVICES=$g
  python -m $PLAN.run_planning $COMMON --env wall       --episodes 12 --ep-offset $((g*12))
  python -m $PLAN.run_planning $COMMON --env pusht      --episodes 12 --ep-offset $((g*12))
  python -m $PLAN.run_planning $COMMON --env point_maze --episodes 3  --ep-offset $((g*3))
  python -m $PLAN.run_planning $COMMON --env metaworld --mw-task reach      --goal-H 20 --horizon 6 --iterations 15 --num-act-stepped 3 --episodes 3 --ep-offset $((g*3))
  python -m $PLAN.run_planning $COMMON --env metaworld --mw-task reach-wall --goal-H 20 --horizon 6 --iterations 15 --num-act-stepped 3 --episodes 3 --ep-offset $((g*3))
  python -m $PLAN.run_planning_droid_franka --wm rae --ckpt $CK --horizon 3 --ctxt-window 1 --episodes 8 --ep-offset $((g*8)) \
      --rae-nfe $NFE --rae-cfg $CFG --rae-cost $COST --probe $([ $g -eq 0 ] && echo 256 || echo 0) --out $OUT
}
for g in 0 1 2 3 4 5 6 7; do shard $g > "$OUT/shard_gpu$g.log" 2>&1 & done
wait
echo "=== $(date) shards done; merging"
cd "$RAE" && PYTHONPATH=src python src/wm/merge_planning.py --out "$OUT"
echo "=== $(date) eval_planning end"
