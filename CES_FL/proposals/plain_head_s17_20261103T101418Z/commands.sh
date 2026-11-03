#!/usr/bin/env bash
# Explicit stages only. Run inside a legitimately allocated GPU job for CUDA stages.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CES_PY=${CES_PY:-/home/sachin.chaudhary/.conda/envs/aiface/bin/python}
: "${CES_RUN_ROOT:?Set a new absolute PRIVATE cluster output directory}"
[[ "$CES_RUN_ROOT" = /* ]] || { echo 'CES_RUN_ROOT must be absolute' >&2; exit 1; }
CODE="$REPO/CES_FL/src/ces_fl_head_ablation.py"
A="$HERE/L_A_plain_bce.json"
H="$HERE/L_H_plain_hierarchical.json"
cd -- "$REPO"
case "${1:-}" in
  prepare)
    [[ ! -e "$CES_RUN_ROOT" ]] || { echo 'Preserve existing run root; inspect progress before resuming individual stages' >&2; exit 1; }
    mkdir -p -- "$CES_RUN_ROOT"
    git rev-parse HEAD > "$CES_RUN_ROOT/LAUNCH_COMMIT.txt"
    "$CES_PY" "$CODE" preflight --config "$A" --out "$CES_RUN_ROOT/preflight_A"
    "$CES_PY" "$CODE" preflight --config "$H" --out "$CES_RUN_ROOT/preflight_H"
    "$CES_PY" "$CODE" smoke --device cuda --head-type linear --out "$CES_RUN_ROOT/smoke_cuda"
    "$CES_PY" "$CODE" dry-run --config "$A" --preflight "$CES_RUN_ROOT/preflight_A" --arm A --device cuda --out "$CES_RUN_ROOT/dry_A"
    "$CES_PY" "$CODE" dry-run --config "$H" --preflight "$CES_RUN_ROOT/preflight_H" --arm D --device cuda --out "$CES_RUN_ROOT/dry_H"
    ;;
  preview-h)
    "$CES_PY" "$CODE" preview --config "$H" --preflight "$CES_RUN_ROOT/preflight_H" --out "$CES_RUN_ROOT/preview_H"
    ;;
  reuse-review-h)
    : "${2:?Supply actual prior H qualification.json path}"
    "$CES_PY" "$HERE/reuse_review.py" --old-qualification "$2" --preview "$CES_RUN_ROOT/preview_H" --out "$CES_RUN_ROOT/review_H"
    ;;
  train-a|resume-a)
    CES_EXTRA=(); [[ "$1" != resume-a ]] || CES_EXTRA+=(--resume)
    "$CES_PY" "$CODE" train --config "$A" --preflight "$CES_RUN_ROOT/preflight_A" --arm A --device cuda \
      --out "$CES_RUN_ROOT/L_A" --software-receipt "$CES_RUN_ROOT/smoke_cuda/SMOKE_RESULT.json" \
      --dry-run-receipt "$CES_RUN_ROOT/dry_A/DRY_RUN.json" "${CES_EXTRA[@]}"
    ;;
  train-h|resume-h)
    CES_EXTRA=(); [[ "$1" != resume-h ]] || CES_EXTRA+=(--resume)
    "$CES_PY" "$CODE" train --config "$H" --preflight "$CES_RUN_ROOT/preflight_H" --arm D --device cuda \
      --out "$CES_RUN_ROOT/L_H" --software-receipt "$CES_RUN_ROOT/smoke_cuda/SMOKE_RESULT.json" \
      --dry-run-receipt "$CES_RUN_ROOT/dry_H/DRY_RUN.json" --qualification "$CES_RUN_ROOT/review_H/qualification.json" "${CES_EXTRA[@]}"
    ;;
  replay-a|replay-h)
    : "${2:?Supply actual completed reference run directory}"
    CES_REF=A; [[ "$1" != replay-h ]] || CES_REF=H
    "$CES_PY" "$HERE/replay_reference.py" --reference "$CES_REF" --run "$2" --out "$CES_RUN_ROOT/reference_$CES_REF" --device cuda
    ;;
  compare)
    for CES_REF in A H; do
      "$CES_PY" "$CODE" compare --first "$CES_RUN_ROOT/L_$CES_REF/validation_scores_PRIVATE.json" \
        --second "$CES_RUN_ROOT/reference_$CES_REF/validation_scores_PRIVATE.json" \
        --out "$CES_RUN_ROOT/plain_minus_gated_$CES_REF.json" --bootstrap 1000 --seed 17
    done
    ;;
  *) echo 'Stages: prepare, preview-h, reuse-review-h OLD_QUALIFICATION, train-a, train-h, resume-a, resume-h, replay-a OLD_RUN, replay-h OLD_RUN, compare' >&2; exit 2;;
esac
