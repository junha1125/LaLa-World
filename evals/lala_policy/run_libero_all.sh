#!/usr/bin/env bash
# Full LIBERO protocol (4 suites x 10 tasks x 50 episodes), one task per process, round-robin over the GPUs.
#   bash evals/lala_policy/run_libero_all.sh <ckpt> [tag] [trials=50] [parallel=16] [gpus="0 1 2 3 4 5 6 7"]
# LALA_POLICY_EVAL_EXTRA = extra driver args for every task (e.g. "--flow-steps 1"); LALA_POLICY_EVAL_DRYRUN=1 prints the lines.
set -euo pipefail
CK=$1; TAG=${2:-$(basename "$(dirname "$CK")")-$(basename "$CK" | sed -E 's/.*-(e[0-9]+)\.pth\.tar/\1/')}
N=${3:-50}; PAR=${4:-16}; GPUS=(${5:-0 1 2 3 4 5 6 7})
EXTRA=${LALA_POLICY_EVAL_EXTRA:-}
DRYRUN=${LALA_POLICY_EVAL_DRYRUN:-}
i=0
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  for T in 0 1 2 3 4 5 6 7 8 9; do
    G=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
    echo "$CK $SUITE $G $N $TAG --tasks $T${EXTRA:+ $EXTRA}"
  done
done | sed 's/[[:blank:]]*$//' | if [ -n "$DRYRUN" ]; then cat; else
  xargs -P "$PAR" -L 1 bash -c 'bash evals/lala_policy/driver_libero.sh $0 "$@" > /dev/null'; fi
if [ -n "$DRYRUN" ]; then exit 0; fi
python -m evals.lala_policy.collect --out "${LALA_POLICY_EVAL_OUT:-logs/policy_eval}" --tag "$TAG"
