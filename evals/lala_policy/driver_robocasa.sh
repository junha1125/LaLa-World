#!/usr/bin/env bash
# RoboCasa eval, one process per task list, inside the RoboCasa venv (third_party/venv-robocasa or $ROBOCASA_PYTHON).
#   bash evals/lala_policy/driver_robocasa.sh <ckpt> <gpu> [trials=50] [tag] [extra args e.g. --task PnPCounterToCab OpenDrawer]
set -euo pipefail
CK=$1; GPU=$2; N=${3:-50}; TAG=${4:-}; shift 4 2>/dev/null || shift $#
PY=${ROBOCASA_PYTHON:-third_party/venv-robocasa/bin/python}; [ -x "$PY" ] || PY=python
OUT=${LALA_POLICY_EVAL_OUT:-logs/policy_eval}
EGL_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}; export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${EGL_ID:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/run_logs"
LOG="$OUT/run_logs/robocasa-${TAG:-run}-gpu${GPU}-$$.log"
$PY -u -m evals.lala_policy.run_robocasa_eval --ckpt "$CK" --trials "$N" --gpu "$GPU" --out "$OUT" \
  ${TAG:+--tag "$TAG"} "$@" 2>&1 | tee "$LOG"
