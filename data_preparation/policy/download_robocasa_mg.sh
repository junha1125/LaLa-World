#!/usr/bin/env bash
# RoboCasa MimicGen data (mg_im, 3000 demos per task, 128 px, 816 GB for 24 tasks) from the official Box links.
# Resumable: re-running skips complete files and continues partial ones.
#   bash data_preparation/policy/download_robocasa_mg.sh [--root DIR] [PARALLEL=3] [TASK ...]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${LALA_DATA_ROOT:-data}
if [ "${1:-}" = "--root" ]; then ROOT=$2; shift 2; fi
ROOT=$(cd "$(dirname "$ROOT")" 2>/dev/null && pwd)/$(basename "$ROOT")/robocasa_mg_raw
PAR=${1:-3}; shift || true
ONLY="$*"
mkdir -p "$ROOT"
LOG="$ROOT/download.log"
echo "[$(date '+%F %T')] start parallel=$PAR only='${ONLY}' -> $ROOT" | tee -a "$LOG"

dl_one () {  # task relpath url size
  local task=$1 rel=$2 url=$3 size=$4 dest="$ROOT/$2"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then echo "[$(date '+%F %T')] SKIP $task (complete)"; return 0; fi
  local part="$dest.part"; local t0=$(date +%s)
  exec 9>"$dest.lock"; flock -n 9 || { echo "[$(date '+%F %T')] SKIP $task (another worker has it)"; return 0; }
  echo "[$(date '+%F %T')] start $task ($((size/1000000000)) GB)"
  for attempt in 1 2 3 4 5 6; do
    curl -sSL -C - --retry 10 --retry-delay 15 --retry-all-errors --max-time 36000 -o "$part" "$url" && break
    echo "[$(date '+%F %T')] retry $task attempt=$attempt"; sleep 30
  done
  local got=$(stat -c %s "$part" 2>/dev/null || echo 0)
  if [ "$got" = "$size" ]; then
    mv "$part" "$dest"; rm -f "$dest.lock"
    echo "[$(date '+%F %T')] done $task $((size/1000000)) MB in $(( $(date +%s) - t0 )) s"
  else
    echo "[$(date '+%F %T')] FAIL $task got=$got expected=$size (re-run to resume)"; return 1
  fi
}
export -f dl_one; export ROOT
tail -n +2 "$HERE/robocasa_mg_urls.tsv" | awk -v only="$ONLY" -F'\t' 'only=="" || index(" " only " ", " " $1 " ") {print}' \
  | xargs -P "$PAR" -L 1 bash -c 'dl_one "$@"' _ 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] finished; $(find "$ROOT" -name '*.hdf5' | wc -l)/24 files" | tee -a "$LOG"
