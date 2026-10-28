#!/usr/bin/env bash
# Full RoboCasa protocol (24 tasks x 50 episodes), one task per process, round-robin over the GPUs.
#   bash evals/lala_policy/run_robocasa_all.sh <ckpt> [tag] [trials=50] [parallel=12] [gpus="0 1 2 3 4 5 6 7"]
# LALA_POLICY_EVAL_EXTRA = extra driver args for every task (e.g. "--flow-steps 1"); LALA_POLICY_EVAL_DRYRUN=1 prints the lines.
set -euo pipefail
CK=$1; TAG=${2:-$(basename "$(dirname "$CK")")-$(basename "$CK" | sed -E 's/.*-(e[0-9]+)\.pth\.tar/\1/')}
N=${3:-50}; PAR=${4:-12}; GPUS=(${5:-0 1 2 3 4 5 6 7})
EXTRA=${LALA_POLICY_EVAL_EXTRA:-}
DRYRUN=${LALA_POLICY_EVAL_DRYRUN:-}
TASKS=$(python -c "from evals.lala_policy.run_robocasa_eval import TASKS; print(' '.join(TASKS))")
i=0
for T in $TASKS; do
  G=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  echo "$CK $G $N $TAG --task $T${EXTRA:+ $EXTRA}"
done | sed 's/[[:blank:]]*$//' | if [ -n "$DRYRUN" ]; then cat; else
  xargs -P "$PAR" -L 1 bash -c 'bash evals/lala_policy/driver_robocasa.sh $0 "$@" > /dev/null'; fi
if [ -n "$DRYRUN" ]; then exit 0; fi
python -m evals.lala_policy.collect --out "${LALA_POLICY_EVAL_OUT:-logs/policy_eval}" --tag "$TAG"
