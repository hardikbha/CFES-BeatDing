#!/usr/bin/env bash
# Explicit stages; never allocates GPUs or submits jobs.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CES_PY=${CES_PY:-/home/sachin.chaudhary/.conda/envs/aiface/bin/python}
: "${CES_RUN_ROOT:?Set a new absolute PRIVATE run directory}"
[[ "$CES_RUN_ROOT" = /* ]] || { echo 'Absolute run root required' >&2; exit 1; }
CODE="$REPO/CES_FL/src/ces_fl_nuisance_resnet50.py"
STAGE=${1:-}; ARM=${2:-}
case "$STAGE" in
  preflight)
    [[ ! -e "$CES_RUN_ROOT" ]] || { echo 'Preserve existing output root' >&2; exit 1; }
    mkdir -p -- "$CES_RUN_ROOT"
    git -C "$REPO" rev-parse HEAD > "$CES_RUN_ROOT/LAUNCH_COMMIT.txt"
    for ITEM in A0 H0 A1 H1; do
      "$CES_PY" "$CODE" preflight --config "$HERE/configs/$ITEM.json" --out "$CES_RUN_ROOT/preflight_$ITEM"
    done
    ;;
  preview|checks|train|qualify)
    case "$ARM" in A0|A1) NATIVE=A;; H0|H1) NATIVE=D;; *) echo 'Specify A0/H0/A1/H1' >&2; exit 1;; esac
    case "$ARM" in A0|H0) MODE=none;; *) MODE=matched;; esac
    CONFIG="$HERE/configs/$ARM.json"
    case "$STAGE" in
      preview)
        [[ "$NATIVE" = D ]] || { echo 'Procedural preview is for H0/H1' >&2; exit 1; }
        "$CES_PY" "$CODE" preview --config "$CONFIG" --preflight "$CES_RUN_ROOT/preflight_$ARM" --out "$CES_RUN_ROOT/preview_$ARM"
        ;;
      qualify)
        : "${3:?Supply actual completed named review JSON; never fill approval automatically}"
        [[ "$NATIVE" = D ]] || exit 1
        "$CES_PY" "$CODE" qualify --preview "$CES_RUN_ROOT/preview_$ARM" --review "$3" --out "$CES_RUN_ROOT/qualification_$ARM.json"
        ;;
      checks|train)
        : "${CUDA_VISIBLE_DEVICES:?Set actual allocated/authorized GPU binding before starting Python}"
        : "${CES_GPU_BINDING_RECEIPT:?Path to operator-verified allocation/UUID receipt}"
        [[ -f "$CES_GPU_BINDING_RECEIPT" ]] || { echo 'Missing actual binding receipt' >&2; exit 1; }
        if [[ "$STAGE" = checks ]]; then
          "$CES_PY" "$CODE" smoke --device cuda --nuisance-mode "$MODE" --out "$CES_RUN_ROOT/smoke_$ARM"
          "$CES_PY" "$CODE" dry-run --config "$CONFIG" --preflight "$CES_RUN_ROOT/preflight_$ARM" --arm "$NATIVE" --device cuda --out "$CES_RUN_ROOT/dry_$ARM"
        else
          EXTRA=(); [[ "$NATIVE" != D ]] || EXTRA+=(--qualification "$CES_RUN_ROOT/qualification_$ARM.json")
          "$CES_PY" "$CODE" train --config "$CONFIG" --preflight "$CES_RUN_ROOT/preflight_$ARM" --arm "$NATIVE" --device cuda \
            --out "$CES_RUN_ROOT/$ARM" --software-receipt "$CES_RUN_ROOT/smoke_$ARM/SMOKE_RESULT.json" \
            --dry-run-receipt "$CES_RUN_ROOT/dry_$ARM/DRY_RUN.json" "${EXTRA[@]}"
        fi
        ;;
    esac
    ;;
  compare)
    "$CES_PY" - "$CES_RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);p=r/'runs.json'
if p.exists(): raise ValueError('Preserve existing comparison binding')
p.write_text(json.dumps({k:str(r/k) for k in ['A0','H0','A1','H1']},indent=2)+'\n')
PY
    "$CES_PY" "$REPO/CES_FL/scripts/report_lh_nuisance.py" --runs-json "$CES_RUN_ROOT/runs.json" --out "$CES_RUN_ROOT/comparison"
    for PAIR in H1_A1 H1_H0; do
      LEFT=${PAIR%_*}; RIGHT=${PAIR#*_}
      "$CES_PY" "$CODE" compare --first "$CES_RUN_ROOT/$LEFT/validation_scores_PRIVATE.json" \
        --second "$CES_RUN_ROOT/$RIGHT/validation_scores_PRIVATE.json" --out "$CES_RUN_ROOT/comparison/$PAIR.json" --bootstrap 1000 --seed 17
    done
    ;;
  *) echo 'Stages: preflight; preview H0/H1; qualify H0/H1 REVIEW; checks ARM; train ARM; compare' >&2; exit 2;;
esac
