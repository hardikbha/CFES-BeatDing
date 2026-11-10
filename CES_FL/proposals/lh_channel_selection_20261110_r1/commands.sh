#!/usr/bin/env bash
set -euo pipefail
stage=${1:?stage required}
run=${2:?new absolute run directory required}
[[ "$run" = /* ]] || { echo 'Absolute run directory required' >&2; exit 2; }
py=${CES_PYTHON:-python}
code=CES_FL/src/ces_fl_lh_channel_selection.py
proposal=CES_FL/proposals/lh_channel_selection_20261110_r1
case "$stage" in
  prepare)
    [[ ! -e "$run" ]] || { echo 'Preserve existing run directory' >&2; exit 2; }
    mkdir -p "$run"
    git rev-parse HEAD > "$run/SOURCE_COMMIT.txt"
    "$py" "$code" bind --template "$proposal/audit_config.json" --out "$run/audit.json"
    "$py" "$code" preflight --config "$run/audit.json" --out "$run/preflight"
    "$py" "$code" preview --config "$run/audit.json" --preflight "$run/preflight" --out "$run/preview"
    ;;
  qualify)
    review=${3:?named human review file required}
    "$py" "$code" qualify --preview "$run/preview" --review "$review" --out "$run/QUALIFIED.json"
    ;;
  cuda-smoke)
    "$py" "$code" smoke --device cuda --out "$run/cuda_smoke"
    ;;
  audit)
    "$py" "$code" select-channels --config "$run/audit.json" --preflight "$run/preflight" --qualification "$run/QUALIFIED.json" --device cuda --out "$run/channel_audit"
    "$py" "$code" bind-masks --config "$run/audit.json" --masks "$run/channel_audit/MASKS.json" --out "$run/configs"
    ;;
  dry|train)
    arm=${3:?mask arm required}
    [[ "$arm" = none || "$arm" = random || "$arm" = informed ]] || exit 2
    if [[ "$stage" = dry ]]; then
      "$py" "$code" dry-run --config "$run/configs/$arm.json" --preflight "$run/preflight" --arm D --device cuda --out "$run/dry_$arm"
    else
      "$py" "$code" train --config "$run/configs/$arm.json" --preflight "$run/preflight" --arm D --device cuda --qualification "$run/QUALIFIED.json" --dry-run-receipt "$run/dry_$arm/DRY_RUN.json" --software-receipt "$run/cuda_smoke/SMOKE_RESULT.json" --out "$run/$arm"
    fi
    ;;
  compare)
    "$py" "$code" compare --first "$run/informed/validation_scores_PRIVATE.json" --second "$run/none/validation_scores_PRIVATE.json" --out "$run/informed_minus_none.json" --bootstrap 2000 --seed 17
    "$py" "$code" compare --first "$run/informed/validation_scores_PRIVATE.json" --second "$run/random/validation_scores_PRIVATE.json" --out "$run/informed_minus_random.json" --bootstrap 2000 --seed 17
    ;;
  *) echo 'Allowed stages: prepare qualify cuda-smoke audit dry train compare' >&2; exit 2 ;;
esac
