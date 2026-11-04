#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CES_PY=${CES_PY:-/home/sachin.chaudhary/.conda/envs/aiface/bin/python}
: "${CES_RUN_ROOT:?Fresh absolute private run root required}"
: "${CES_CONTROLS_ROOT:?Actual completed nuisance-pilot root containing A0 and H0 required}"
[[ "$CES_RUN_ROOT" = /* && "$CES_CONTROLS_ROOT" = /* ]] || exit 2
CODE="$REPO/CES_FL/src/ces_fl_nuisance_resnet50.py"
STAGE=${1:-};NAME=${2:-B0};NATIVE=B
case "$NAME" in B0) NATIVE=B;; C0) NATIVE=C;; *) exit 2;; esac
case "$STAGE" in
 prepare)
  [[ ! -e "$CES_RUN_ROOT" ]] || { echo 'Preserve earlier outputs' >&2;exit 1; }
  mkdir -p "$CES_RUN_ROOT"
  "$CES_PY" "$HERE/verify_controls.py" --root "$CES_CONTROLS_ROOT" --out "$CES_RUN_ROOT/CONTROLS_VERIFIED.json"
  "$CES_PY" "$CODE" preflight --config "$HERE/configs/B0.json" --out "$CES_RUN_ROOT/preflight"
  ;;
 checks)
  : "${CUDA_VISIBLE_DEVICES:?Actual bound GPU required}"
  "$CES_PY" "$CODE" smoke --device cuda --nuisance-mode none --out "$CES_RUN_ROOT/smoke_$NAME"
  "$CES_PY" "$CODE" dry-run --config "$HERE/configs/$NAME.json" --preflight "$CES_RUN_ROOT/preflight" --arm "$NATIVE" --device cuda --out "$CES_RUN_ROOT/dry_$NAME"
  ;;
 train)
  : "${CES_ACCOUNTED_STAGE_RECEIPT:?Actual resource/budget profiling receipt required}"
  [[ -f "$CES_ACCOUNTED_STAGE_RECEIPT" && -f "$CES_RUN_ROOT/CONTROLS_VERIFIED.json" ]] || exit 1
  EXTRA=()
  if [[ "$NAME" = C0 ]]; then
   : "${CES_C0_QUALIFICATION:?Verified identical-config/source H0 qualification or a genuine new review required}"
   EXTRA+=(--qualification "$CES_C0_QUALIFICATION")
  fi
  "$CES_PY" "$CODE" train --config "$HERE/configs/$NAME.json" --preflight "$CES_RUN_ROOT/preflight" --arm "$NATIVE" --device cuda --out "$CES_RUN_ROOT/$NAME" --software-receipt "$CES_RUN_ROOT/smoke_$NAME/SMOKE_RESULT.json" --dry-run-receipt "$CES_RUN_ROOT/dry_$NAME/DRY_RUN.json" "${EXTRA[@]}"
  ;;
 compare)
  for PAIR in B0_A0 C0_B0 H0_C0 B0_H0; do
   LEFT=${PAIR%_*};RIGHT=${PAIR#*_};LD="$CES_RUN_ROOT/$LEFT";RD="$CES_RUN_ROOT/$RIGHT"
   [[ "$LEFT" != A0 && "$LEFT" != H0 ]] || LD="$CES_CONTROLS_ROOT/$LEFT"
   [[ "$RIGHT" != A0 && "$RIGHT" != H0 ]] || RD="$CES_CONTROLS_ROOT/$RIGHT"
   "$CES_PY" "$CODE" compare --first "$LD/validation_scores_PRIVATE.json" --second "$RD/validation_scores_PRIVATE.json" --out "$CES_RUN_ROOT/${PAIR}_bootstrap.json" --bootstrap 1000 --seed 17
  done
  ;;
 *) echo 'Stages: prepare; checks B0/C0; train B0/C0; compare' >&2;exit 2;;
esac
