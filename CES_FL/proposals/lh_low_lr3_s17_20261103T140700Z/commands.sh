#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE=$(cd -- "$HERE/../../.." && pwd)
ROOT=/home/sachin.chaudhary/ces_fl_runs/lh_low_lr3_s17_20261103T140700Z
PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
CODE="$SOURCE/CES_FL/src/ces_fl_head_ablation.py"
cd "$SOURCE"
case "${1:?Explicit stage required}" in
 gpu)
  "$PY" -u "$CODE" smoke --device cuda --head-type linear --out "$ROOT/smoke_cuda" > "$ROOT/smoke_cuda.log" 2>&1
  "$PY" -u "$CODE" dry-run --config "$HERE/config.json" --preflight "$ROOT/preflight" --arm D --device cuda --out "$ROOT/dry" > "$ROOT/dry.log" 2>&1
  "$PY" -u "$CODE" train --config "$HERE/config.json" --preflight "$ROOT/preflight" --arm D --device cuda --out "$ROOT/training" --software-receipt "$ROOT/smoke_cuda/SMOKE_RESULT.json" --dry-run-receipt "$ROOT/dry/DRY_RUN.json" --qualification "$ROOT/review/qualification.json" > "$ROOT/train.log" 2>&1
  test -f "$ROOT/training/DONE"
  ;;
 *) echo 'Unknown stage' >&2; exit 2;;
esac
