#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE=$(cd -- "$HERE/../../.." && pwd)
ARM=${1:?arm required}
case "$ARM" in SHAM_FIX|NO_ALIGN) ;; *) exit 2;; esac
ROOT=/home/sachin.chaudhary/ces_fl_runs/lh_sham_noalign_s17_20261103T165600Z/$ARM
PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
CODE="$SOURCE/CES_FL/src/ces_fl_head_ablation.py"
cd "$SOURCE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
case "${2:?stage required}" in
 cpu)
  "$PY" -u "$HERE/checks.py" verify "$ARM" > "$ROOT/request_check.log" 2>&1
  "$PY" -u "$CODE" preflight --config "$HERE/$ARM.json" --out "$ROOT/preflight" > "$ROOT/preflight.log" 2>&1
  "$PY" -u "$CODE" preview --config "$HERE/$ARM.json" --preflight "$ROOT/preflight" --out "$ROOT/preview" > "$ROOT/preview.log" 2>&1
  "$PY" -u "$HERE/checks.py" review "$ARM" > "$ROOT/review.log" 2>&1
  ;;
 gpu)
  CES_TEST_DEVICE=cuda "$PY" -u CES_FL/tests/test_lh_loss_ablations.py > "$ROOT/loss_tests_cuda.log" 2>&1
  "$PY" -u "$CODE" smoke --device cuda --head-type linear --out "$ROOT/smoke_cuda" > "$ROOT/smoke_cuda.log" 2>&1
  "$PY" -u "$CODE" dry-run --config "$HERE/$ARM.json" --preflight "$ROOT/preflight" --arm D --device cuda --out "$ROOT/dry" > "$ROOT/dry.log" 2>&1
  "$PY" -u "$CODE" train --config "$HERE/$ARM.json" --preflight "$ROOT/preflight" --arm D --device cuda --out "$ROOT/training" --software-receipt "$ROOT/smoke_cuda/SMOKE_RESULT.json" --dry-run-receipt "$ROOT/dry/DRY_RUN.json" --qualification "$ROOT/review/qualification.json" > "$ROOT/train.log" 2>&1
  test -f "$ROOT/training/DONE"
  "$PY" -u "$HERE/report.py" "$ARM" > "$ROOT/comparison.log" 2>&1
  ;;
 *) exit 2;;
esac
