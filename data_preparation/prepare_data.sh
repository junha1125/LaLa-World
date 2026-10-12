#!/usr/bin/env bash
# One-shot download + preprocessing. Everything lands under --root (default $LALA_DATA_ROOT or ./data).
#   bash data_preparation/prepare_data.sh world   [--root data] [--workers 24]   # sim + droid + openvid + edit
#   bash data_preparation/prepare_data.sh policy  [--root data] [--workers 24]   # libero + robocasa + robocasa_mg
#   bash data_preparation/prepare_data.sh sim|droid|openvid|edit|eval|libero|robocasa|robocasa_mg|all
set -euo pipefail
cd "$(dirname "$0")/.."
D=data_preparation
TARGET=${1:?target: world | policy | all | sim | droid | openvid | edit | eval | libero | robocasa | robocasa_mg}; shift
ROOT=${LALA_DATA_ROOT:-data}; WORKERS=24
while [ $# -gt 0 ]; do case "$1" in --root) ROOT=$2; shift 2;; --workers) WORKERS=$2; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
mkdir -p "$ROOT"; ROOT=$(cd "$ROOT" && pwd); export LALA_DATA_ROOT=$ROOT
echo "== data root: $ROOT"

sim() {
  python $D/download_sim.py
  for env in wall point_maze pusht metaworld rope granular; do
    [ -f "$ROOT/jsonl/traj/$env.jsonl" ] && { echo "skip $env (exists)"; continue; }
    python $D/build_sim_traj.py --env $env --workers "$WORKERS"
  done
}
droid() {
  python $D/droid/download_droid.py
  [ -f "$ROOT/jsonl/traj/droid.jsonl" ] || python $D/droid/build_droid_traj.py --workers "$WORKERS"
  python $D/droid/validate_droid_traj.py
}
openvid() {
  python $D/openvid/download_openvid.py
  if [ ! -f "$ROOT/openvid_raw/outputs/captions.jsonl" ]; then
    echo "captions.jsonl not found: run data_preparation/openvid/caption_sequences.py (needs a VLM endpoint), then re-run"; return 1
  fi
  [ -f "$ROOT/jsonl/traj/openvid.jsonl" ] || python $D/openvid/build_openvid_traj.py --check-frames all
  python $D/openvid/validate_openvid_traj.py
}
edit() {
  python $D/edit/download_omniedit.py
  python $D/edit/build_omniedit_pair.py --workers "$WORKERS"
  python $D/edit/download_sharegpt.py
  python $D/edit/build_sharegpt_pair.py
}
eval() {
  python $D/download_eval_only.py
}
libero() {
  python $D/policy/download_cosmos.py --what libero
  python $D/policy/build_cosmos_libero_jsonl.py
}
robocasa() {
  python $D/policy/download_cosmos.py --what robocasa
  python $D/policy/build_cosmos_robocasa_jsonl.py
}
robocasa_mg() {
  bash $D/policy/download_robocasa_mg.sh --root "$ROOT" 8
  python $D/policy/verify_robocasa_mg.py
  python $D/policy/build_robocasa_mg_jsonl.py --mask 300_demos --workers "$WORKERS"
}

case "$TARGET" in
  world)  sim; droid; openvid; edit; eval ;;
  policy) libero; robocasa; robocasa_mg ;;
  all)    sim; droid; openvid; edit; eval; libero; robocasa; robocasa_mg ;;
  sim|droid|openvid|edit|eval|libero|robocasa|robocasa_mg) "$TARGET" ;;
  *) echo "unknown target $TARGET"; exit 2 ;;
esac
echo "== done: $TARGET"
