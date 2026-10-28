#!/usr/bin/env bash
# Zero-shot RoboCasa planning (JEPA-WMs Rc-R / Rc-Pl protocol) inside the RoboCasa venv (third_party/venv-robocasa or $ROBOCASA_PYTHON).
#   bash evals/lala_world/driver_robocasa.sh <ckpt> <reach|place> <gpu> [plan|hold|zero|random] [episodes=32]
set -euo pipefail
CK=$1; SUB=$2; GPU=$3; POL=${4:-plan}; N=${5:-32}
PY=${ROBOCASA_PYTHON:-third_party/venv-robocasa/bin/python}; [ -x "$PY" ] || PY=python
OUT=${LALA_PLANNING_OUT:-logs/planning}
RUN=$(basename "$(dirname "$CK")"); EP=$(basename "$CK" | sed -E 's/.*-(e[0-9]+)\.pth\.tar/\1/')
TAG=${RUN}-${EP}-${SUB}$([ "$POL" = plan ] || echo "-$POL")-ep${N}
EGL_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}; export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${EGL_ID:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/run_logs"
$PY -u -m evals.lala_world.run_planning_robocasa --ckpt "$CK" --subtask "$SUB" --episodes "$N" --policy "$POL" \
  --text-source semantic --ctxt-window 2 --horizon 3 --iterations 15 --num-samples 300 --num-elites 10 \
  --num-act-stepped 1 --xyz-max-norm 0.05 --rot-max-norm 0.01 --seed 1 --gpu "$GPU" --out "$OUT" --tag "$TAG" \
  2>&1 | tee "$OUT/run_logs/robocasa-${TAG}.log"
