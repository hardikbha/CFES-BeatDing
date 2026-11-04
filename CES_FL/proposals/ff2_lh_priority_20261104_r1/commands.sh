#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CES_PY=${CES_PY:-/home/sachin.chaudhary/.conda/envs/aiface/bin/python}
: "${LH_FF2_ROOT:?Set a fresh absolute private output root}"
[[ "$LH_FF2_ROOT" = /* ]] || exit 2
CODE="$REPO/CES_FL/src/ces_fl_head_ablation.py"
cd -- "$REPO"
case "${1:-}" in
 prepare)
  : "${LH_NATIVE_BCE_RUN:?Point to completed verified native FF2 BCE50 training directory}"
  "$CES_PY" "$HERE/prepare.py" --native-run "$LH_NATIVE_BCE_RUN" --out "$LH_FF2_ROOT"
  "$CES_PY" "$CODE" preflight --config "$LH_FF2_ROOT/L_H.json" --out "$LH_FF2_ROOT/preflight"
  "$CES_PY" "$CODE" preview --config "$LH_FF2_ROOT/L_H.json" --preflight "$LH_FF2_ROOT/preflight" --out "$LH_FF2_ROOT/preview"
  ;;
 qualify)
  : "${LH_REVIEW:?Provide actual completed named review for the new preview; previous rejects cannot be automatically approved}"
  "$CES_PY" "$CODE" qualify --preview "$LH_FF2_ROOT/preview" --review "$LH_REVIEW" --out "$LH_FF2_ROOT/qualification.json"
  ;;
 cuda-checks)
  "$CES_PY" "$CODE" smoke --device cuda --head-type linear --out "$LH_FF2_ROOT/smoke_cuda"
  for ARM in D A; do
   "$CES_PY" "$CODE" dry-run --config "$LH_FF2_ROOT/L_H.json" --preflight "$LH_FF2_ROOT/preflight" --arm "$ARM" --device cuda --out "$LH_FF2_ROOT/dry_$ARM"
  done
  ;;
 train-h|train-a)
  : "${LH_CUMULATIVE_GPU_HOURS:?Set only a ledger-verified allowance after profiling}"
  ARM=D;NAME=L_H
  [[ "$1" != train-a ]] || { ARM=A;NAME=L_A; }
  EXTRA=();[[ "$ARM" != D ]] || EXTRA+=(--qualification "$LH_FF2_ROOT/qualification.json")
  "$CES_PY" "$CODE" train --config "$LH_FF2_ROOT/$NAME.json" --preflight "$LH_FF2_ROOT/preflight" --arm "$ARM" --device cuda --out "$LH_FF2_ROOT/$NAME" --software-receipt "$LH_FF2_ROOT/smoke_cuda/SMOKE_RESULT.json" --dry-run-receipt "$LH_FF2_ROOT/dry_$ARM/DRY_RUN.json" --budget-hours "$LH_CUMULATIVE_GPU_HOURS" "${EXTRA[@]}"
  ;;
 report-h|report-a)
  NAME=L_H;[[ "$1" != report-a ]] || NAME=L_A
  "$CES_PY" "$HERE/report.py" --run "$LH_FF2_ROOT/$NAME" --scores "$LH_FF2_ROOT/$NAME/validation_scores_PRIVATE.json" --split val --out "$LH_FF2_ROOT/${NAME}_validation_extended.json"
  ;;
 *) echo 'Stages: prepare, qualify, cuda-checks, train-h, train-a, report-h, report-a' >&2;exit 2;;
esac
