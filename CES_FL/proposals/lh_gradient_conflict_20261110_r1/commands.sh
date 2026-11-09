#!/usr/bin/env bash
# Explicit stages only. Does not allocate, submit, invent approvals or read TEST.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CES_PY=${CES_PY:-/home/sachin.chaudhary/.conda/envs/aiface/bin/python}
: "${CES_EXPECTED_COMMIT:?Set the exact pushed handoff commit}"
: "${CES_RUN_ROOT:?Set a new absolute private cohort directory}"
[[ "$CES_RUN_ROOT" = /* ]] || { echo 'Absolute private output required' >&2; exit 1; }
CODE="$REPO/CES_FL/scripts/lh_gradient_conflict.py"
case "${1:-}" in
  metadata)
    "$CES_PY" "$REPO/CES_FL/scripts/ff2_provenance_inventory.py" --out "$CES_RUN_ROOT/provenance"
    ;;
  bind)
    "$CES_PY" "$CODE" bind --references "$CES_RUN_ROOT/references.json" --out "$CES_RUN_ROOT/binding.json"
    ;;
  diagnose)
    "$CES_PY" "$CODE" diagnose --binding "$CES_RUN_ROOT/binding.json" --out "$CES_RUN_ROOT/diagnostic" --device cuda
    ;;
  software-cuda)
    "$CES_PY" "$CODE" software-check --device cuda --out "$CES_RUN_ROOT/software_cuda.json"
    ;;
  profile)
    "$CES_PY" "$CODE" profile --binding "$CES_RUN_ROOT/binding.json" --diagnostic "$CES_RUN_ROOT/diagnostic/DIAGNOSTIC.json" --software "$CES_RUN_ROOT/software_cuda.json" --out "$CES_RUN_ROOT/profile" --device cuda
    ;;
  train)
    : "${CES_TRAIN_BUDGET_HOURS:?Use the measured allowance, at most1.5h and within the remaining cohort window}"
    "$CES_PY" "$CODE" train --binding "$CES_RUN_ROOT/binding.json" --diagnostic "$CES_RUN_ROOT/diagnostic/DIAGNOSTIC.json" --software "$CES_RUN_ROOT/software_cuda.json" --profile "$CES_RUN_ROOT/profile/PROFILE.json" --out "$CES_RUN_ROOT/H_GP" --budget-hours "$CES_TRAIN_BUDGET_HOURS"
    ;;
  report)
    "$CES_PY" "$CODE" report --binding "$CES_RUN_ROOT/binding.json" --candidate "$CES_RUN_ROOT/H_GP" --out "$CES_RUN_ROOT/report"
    ;;
  *) echo 'Stages: metadata | bind | diagnose | software-cuda | profile | train | report' >&2; exit 2;;
esac
