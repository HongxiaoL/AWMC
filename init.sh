#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

export PYTHONUNBUFFERED=1

if [[ -f "$BASE_DIR/.env" ]]; then
  set -a
  source "$BASE_DIR/.env"
  set +a
fi

mkdir -p "$BASE_DIR/log" "$BASE_DIR/result"

LOG_FILE="$BASE_DIR/log/shell.log"
touch "$LOG_FILE"
export LOG_FILE

log_echo() {
  local msg="$1"
  local ts
  ts="$(date '+%F %T')"
  echo "$msg"
  printf "[%s] %s\n" "$ts" "$msg" >> "$LOG_FILE"
}

STAGE_JOBS=10
DML_JOBS=1
PAIR_DML_PARALLEL=0

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

run_stage_parallel() {
  local stage_name="$1"
  local jobs="$2"
  shift 2
  local files=("$@")

  log_echo "===================="
  log_echo "Stage: ${stage_name}"
  log_echo "Parallel jobs: ${jobs}"

  if [[ ${#files[@]} -eq 0 ]]; then
    log_echo "No scripts found for stage: ${stage_name}"
    return 0
  fi

  printf "%s\0" "${files[@]}" | xargs -0 -n1 -P "${jobs}" -I{} bash -lc 'f="$1"; ts="$(date "+%F %T")"; msg="[RUN] $f"; echo "$msg"; printf "[%s] %s\n" "$ts" "$msg" >> "$LOG_FILE"; python "$f"' _ {}
}

log_echo "===================="
log_echo "Stage: coord.py"
log_echo "===================="
python coord.py

log_echo "===================="
log_echo "Stage: extract.py"
log_echo "===================="
python extract.py

shopt -s nullglob

run_stage_parallel "category_agent" "${STAGE_JOBS}" "$BASE_DIR"/category_agent/*_extract.py

run_stage_parallel "category_pair (paired-difference)" "${STAGE_JOBS}" "$BASE_DIR"/category_pair/*_pair.py
run_stage_parallel "category_DML (DML)" "${DML_JOBS}" "$BASE_DIR"/category_DML/*_dml.py

run_stage_parallel "category_report" "${STAGE_JOBS}" "$BASE_DIR"/category_report/*_report.py

log_echo "===================="
log_echo "Stage: final_report.py"
log_echo "===================="
python final_report.py

log_echo "ALL DONE. Output: $BASE_DIR/result/final_report.md"
