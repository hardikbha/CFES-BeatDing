#!/usr/bin/env bash
set -euo pipefail
SOURCE=${SOURCE:?set SOURCE to detached worktree path}
COMMIT=${COMMIT:?set COMMIT to the exact proposal commit}
RUNROOT=/home/sachin.chaudhary/ces_fl_runs/ff2_sjgentle_confirmation_20261108_r1
PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
ID=CES_FL/proposals/ff2_sjgentle_confirmation_20261108_r1
case ${1:?preflight|launch-seed7|launch-seed27} in
  preflight)
    for seed in 7 27; do
      name=SJ_gentle_s${seed}
      mkdir -p "$RUNROOT/$name"
      "$PY" "$SOURCE/$ID/operator/rebind_preflight.py" \
        --source "$SOURCE" \
        --config "$SOURCE/$ID/$name.json" \
        --reference-config /home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107/SJ_gentle/config.json \
        --reference-preflight /home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107/SJ_gentle/preflight \
        --out "$RUNROOT/$name/preflight_rebound"
    done
    ;;
  launch-seed7|launch-seed27)
    seed=${1#launch-seed}; name=SJ_gentle_s${seed}
    # Requires an idle, lockable direct-SSH shared-GPU lease. No PBS job is killed or preempted.
    ssh -o BatchMode=yes gpu01 "bash $SOURCE/$ID/operator/run_one_shared_lease.sh $name $SOURCE $COMMIT"
    ;;
  *) echo 'Unknown command' >&2; exit 2 ;;
esac
