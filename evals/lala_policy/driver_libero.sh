#!/usr/bin/env bash
# LIBERO eval, one process per (suite, task shard), inside the LIBERO venv (third_party/venv-libero or $LIBERO_PYTHON).
#   bash evals/lala_policy/driver_libero.sh <ckpt> <suite> <gpu> [trials=50] [tag] [extra run_libero_eval args...]
set -euo pipefail
CK=$1; SUITE=$2; GPU=$3; N=${4:-50}; TAG=${5:-}; shift 5 2>/dev/null || shift $#
PY=${LIBERO_PYTHON:-third_party/venv-libero/bin/python}; [ -x "$PY" ] || PY=python
OUT=${LALA_POLICY_EVAL_OUT:-logs/policy_eval}
EGL_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}; export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${EGL_ID:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/run_logs"
LOG="$OUT/run_logs/libero-${TAG:-run}-${SUITE}-gpu${GPU}-$$.log"
$PY -u -m evals.lala_policy.run_libero_eval --ckpt "$CK" --suite "$SUITE" --trials "$N" --gpu "$GPU" --out "$OUT" \
  ${TAG:+--tag "$TAG"} "$@" 2>&1 | tee "$LOG"
