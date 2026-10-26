#!/bin/bash
[ -e "${MA_STOP_FILE:-/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/STOP_MA}" ] && { echo "CESEL_MA stopped by user (${MA_STOP_FILE:-work/STOP_MA})"; exit 0; }
# One task of the CESEL_MA priority runner (scripts/priority_runner.py; task list run/tasks.yaml). Usable by hand:
#   TASK_NAME=<name> CUDA_VISIBLE_DEVICES=<i> bash run/ma_run.sh <kind> <args...>
# Every task takes the RUNLOCK <MA_OUT>/work/tasklocks/<TASK_NAME>.RUNLOCK first (the CESel protocol of
# agents/CESEL/run/cesel_run.sh, Codex review F1/F11 fix; functions copied verbatim below): one runner per task across
# hosts; exit 4 = held by another live runner (never counted as a failed attempt).
# Kinds (roots overridable for the dry run: MA_OUT, CESEL_ROOT, CFES_ROOT, CLIPFAIR_ROOT, FAIRREL_ROOT, DING_ROOT):
#   ma <ma|ma_floor> <seed>          CESel-MA FF++ (configs/<cfg>_ffpp.yaml) from CFES runs/clip_s<seed>_p1/last.pt
#                                    (backbone only), validated cues = CESEL runs/ffpp_B0_s7/validated_cues.json (all
#                                    seeds, Amendment 6), 10 x 1976 steps, val_oppoint_ma selection -> runs/ffpp_<cfg>_s<seed>
#                                    + evaluate_ces --leak_free + calibrate + ces_posteval + post-hoc cue validation
#                                    (validated_cues_post.json: the frequency-branch drop rule)
#   ma_compare <seed>                CPU: MA vs the single-axis winner on validation (scripts/ma_compare.py)
#   clipfair_sfd <seed>              CLIPFAIR run/clipfair_run.sh sfd <seed> (unchanged, its own RUNLOCK too)
#   clipfair <daw|dag|pgfdd|ding> <seed>   CLIPFAIR run/clipfair_run.sh train <m> <seed>
#   ding_gdagrad <seed>              Ding SFD+GDA with the GDA term CONNECTED (clipfair.ding.gda_grad=true, Amendment 6)
#                                    with CLIPFAIR's scripts -> CLIPFAIR/runs/ffpp_ding_gdagrad_s<seed> + eval/calibrate/A1
#   fairrel <seed>                   FairReL official code: FAIRREL_REPRO/code/run_task.sh FAIRREL <seed>
#   dingrepro <seed>                 Ding official Xception SFD+GDA: DINGREPRO_LIN/code/run_task_full.sh <seed>
#   armE <seed> <w>                  CFES run/cfer_run.sh fullE <seed> <w> (rev4 Arm E; CFER_WORKERS=1 as s7)
#   ff2p1 <seed>                     FF2-public phase 1 (configs/ff2_public_p1.yaml) -> runs_ff2public/clip_s<seed>_p1
#   ff2_cfesd <seed>                 CFES-D on FF2-public (configs/cfesd_ff2_public.yaml) -> runs_ff2public/cfesd_s<seed>
#   ff2_ces <config> <name> <seed> [validated json]   CES/CESel-MA run on FF2-public from the public phase 1
#                                    -> runs_ff2public/<name> (+ evaluation)
#   ff2_validate <seed>              validated-cue JSON of runs_ff2public/B0_s<seed>/best.pt (FF2-public SELECT only)
#   cmd                              coordinator-added task: bash -c "$TASK_CMD" in $TASK_CWD (default A)
# Dry run (MA_DRY=1, MA_OUT = scratch): every kind except ma / ma_compare is a stub that writes the task's done markers
# ($TASK_DONE, ':'-separated); ma runs the REAL trainer on CPU with the stub backbone ($MA_DRY_SET overrides).
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA
AG=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents
OUT=${MA_OUT:-$A}
CESEL_ROOT=${CESEL_ROOT:-$AG/CESEL}; CFES_ROOT=${CFES_ROOT:-$AG/CFES}; CLIPFAIR_ROOT=${CLIPFAIR_ROOT:-$AG/CLIPFAIR}
FAIRREL_ROOT=${FAIRREL_ROOT:-$AG/FAIRREL_REPRO}; DING_ROOT=${DING_ROOT:-$AG/DINGREPRO_LIN}
PY=/home/sachin.chaudhary/.conda/envs/pulid/bin/python
PYD=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
LOCK_LEASE=${CESEL_LOCK_LEASE:-1800}
LOCK_OBSERVE=${CESEL_LOCK_OBSERVE:-240}
LOCK_BEAT=${CESEL_LOCK_BEAT:-60}
cd "$A" || exit 2
kind=$1; shift
TASK=${TASK_NAME:-${kind}_$(echo "$*" | tr ' /' '__')}
DRYM=0; DEV=(); EV=(); DSET=()
CPU_KINDS=" ma_compare cmd_cpu "
if [ "${MA_DRY:-0}" = 1 ]; then
  [ "$OUT" = "$A" ] && { echo "MA_DRY=1 needs MA_OUT set to a scratch root (never the real run directories)"; exit 2; }
  DRYM=1; DEV=(--device cpu); EV=(--max_items 64); read -r -a DSET <<< "${MA_DRY_SET:-}"
elif [ "${CESEL_LOCK_TEST:-0}" != 1 ] && [[ "$CPU_KINDS" != *" $kind "* ]] && [ "${TASK_GPU:-1}" = 1 ]; then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "$(date -Is) REFUSE: CUDA_VISIBLE_DEVICES empty"; exit 3; }
  "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "$(date -Is) REFUSE: torch sees no CUDA device"; exit 3; }
fi
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${MA_WORKERS:-2}

# ------------------------------------------------------------------------------------------------ RUNLOCK
lock_owner_line(){   # $1 = lock dir -> first line of its owner file (empty if unreadable)
  head -n1 "$1/owner" 2>/dev/null
}
lock_token_of(){     # owner line -> generation token (pre-fix locks: "legacy_" + md5 of the whole line)
  local l=$1
  case $l in
    "") return 1;;
    token=*) l=${l#token=}; printf '%s\n' "${l%% *}";;
    *) printf 'legacy_%s\n' "$(printf '%s' "$l" | md5sum | cut -c1-16)";;
  esac
}
lock_field(){        # owner line, field name (host|pid) -> value (legacy format: "host job pid")
  local l=$1 f=$2 w
  case $l in
    token=*) for w in $l; do case $w in "$f"=*) printf '%s\n' "${w#*=}"; return 0;; esac; done; return 1;;
    *) set -- $l; case $f in host) printf '%s\n' "${1:-}";; pid) printf '%s\n' "${3:-}";; esac;;
  esac
}
lock_beat(){         # heartbeat content (counter) + mtime of the lock's heartbeat file
  printf '%s|%s\n' "$(cat "$1/heartbeat" 2>/dev/null)" "$(stat -c %Y "$1/heartbeat" 2>/dev/null || stat -c %Y "$1" 2>/dev/null)"
}
lock_sweep(){        # remove tombstones / abandoned staging dirs older than 7 days (never recent ones)
  find "$(dirname "$1")" -maxdepth 1 \( -name "$(basename "$1").gone.*" -o -name "$(basename "$1").new.*" \) \
       -mmin +10080 -exec rm -rf {} + 2>/dev/null
}

take_lock(){   # take_lock <out>: exclusive RUNLOCK on <out>.RUNLOCK; exits 4 when held / lost race
  local L="$1.RUNLOCK" tok st line cur host pid b1 b2 now mt age
  tok="$(hostname -s)_${PBS_JOBID:-nojob}_$$_$(date +%s%N)_${RANDOM}${RANDOM}"
  tok=${tok//[^A-Za-z0-9._-]/_}
  st="$L.new.$tok"
  mkdir -p "$(dirname "$L")" || exit 1
  lock_sweep "$L"
  mkdir "$st" || { echo "cannot create $st"; exit 1; }
  printf 'token=%s host=%s job=%s pid=%s since=%s\n' "$tok" "$(hostname -s)" "${PBS_JOBID:-nojob}" "$$" "$(date -Is)" \
    > "$st/owner" && echo 0 > "$st/heartbeat" || { rm -rf "$st"; echo "cannot write $st"; exit 1; }
  if ! mv -T "$st" "$L" 2>/dev/null; then          # <out>.RUNLOCK exists: held, or stale
    line=$(lock_owner_line "$L")
    cur=$(lock_token_of "$line") || { rm -rf "$st"; echo "skip $1: lock without a readable owner (being created?)"; exit 4; }
    host=$(lock_field "$line" host); pid=$(lock_field "$line" pid)
    if [ "$host" = "$(hostname -s)" ] && [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "stale lock on $1 ($line): owner pid $pid on this host is dead"
    else
      mt=$(stat -c %Y "$L/heartbeat" 2>/dev/null || stat -c %Y "$L" 2>/dev/null || echo 0)
      now=$(date +%s); age=$(( now - mt ))
      if [ "$age" -le "$LOCK_LEASE" ]; then
        rm -rf "$st"; echo "skip $1: held by $line (heartbeat clock age ${age}s <= lease ${LOCK_LEASE}s)"; exit 4
      fi
      b1=$(lock_beat "$L")
      echo "lock on $1 looks stale (clock age ${age}s); observing its heartbeat for ${LOCK_OBSERVE}s"
      sleep "$LOCK_OBSERVE"
      b2=$(lock_beat "$L")
      [ "$(lock_token_of "$(lock_owner_line "$L")")" = "$cur" ] || { rm -rf "$st"; echo "skip $1: lock changed hands while observing"; exit 4; }
      if [ "$b1" != "$b2" ]; then
        rm -rf "$st"; echo "skip $1: heartbeat advanced while observing ($b1 -> $b2): owner alive"; exit 4
      fi
      echo "stale lock on $1 ($line): heartbeat unchanged for ${LOCK_OBSERVE}s, clock age ${age}s"
    fi
    # @lock-test-barrier (tests/test_runlock_f1.sh inserts a scheduling barrier here)
    if ! mv -T "$L" "$L.gone.$cur" 2>/dev/null; then
      rm -rf "$st"; echo "skip $1: another runner retired generation $cur first"; exit 4
    fi
    if [ "$(lock_token_of "$(lock_owner_line "$L.gone.$cur")")" != "$cur" ]; then
      # cannot happen under this protocol (a generation is only ever retired to its own tombstone); never steal
      mv -T "$L.gone.$cur" "$L" 2>/dev/null
      rm -rf "$st"; echo "skip $1: retired lock was not generation $cur; restored, backing off"; exit 4
    fi
    echo "took over $1 from generation $cur"
    if ! mv -T "$st" "$L" 2>/dev/null; then
      rm -rf "$st"; echo "skip $1: another runner published a fresh lock first"; exit 4
    fi
  fi
  LOCK_DIR=$L; LOCK_TOKEN=$tok
  echo "RUNLOCK $L acquired: token $tok"
  (                                   # heartbeat + fencing
    n=0
    while kill -0 $$ 2>/dev/null; do
      if [ -e "$L.gone.$tok/owner" ]; then
        echo "$(date -Is) RUNLOCK LOST: generation $tok of $L was retired by another runner; terminating this run" >&2
        for c in $(pgrep -P $$); do [ "$c" != "$BASHPID" ] && kill -TERM "$c" 2>/dev/null; done
        kill -TERM $$ 2>/dev/null
        exit 0
      fi
      if [ "$(lock_token_of "$(lock_owner_line "$L")")" = "$tok" ]; then
        n=$((n + 1)); echo "$n" > "$L/heartbeat.tmp.$tok" 2>/dev/null && mv -f "$L/heartbeat.tmp.$tok" "$L/heartbeat" 2>/dev/null
      fi
      sleep "$LOCK_BEAT"
    done
  ) &
  LOCK_HB=$!
  trap release_lock EXIT
}

release_lock(){   # EXIT trap: retire ONLY our own generation (atomic rename to our tombstone); never touch a successor
  [ -n "${LOCK_HB:-}" ] && kill "$LOCK_HB" 2>/dev/null
  [ -n "${LOCK_DIR:-}" ] || return 0
  if [ "$(lock_token_of "$(lock_owner_line "$LOCK_DIR")")" = "$LOCK_TOKEN" ]; then
    mv -T "$LOCK_DIR" "$LOCK_DIR.gone.$LOCK_TOKEN" 2>/dev/null
  fi
}

stub(){   # dry run: write the task's done markers
  local f IFS=:
  sleep "${MA_DRY_SLEEP:-1}"
  for f in ${TASK_DONE:-}; do mkdir -p "$(dirname "$f")"; echo "dry stub $TASK $(date -Is)" > "$f"; done
  echo "DRY STUB $TASK ($kind $*): wrote ${TASK_DONE:-nothing}"
}

ces_train(){   # ces_train <config> <out> <seed> [--set overrides...]
  local cfg=$1 out=$2 s=$3; shift 3
  local R=()
  [ -f "$out/TRAINED" ] && return 0
  [ -f "$out/last.pt" ] && R=(--resume)
  "$PY" -u scripts/train_ces.py --config "$cfg" --out "$out" "${R[@]}" "${DEV[@]}" --set "seed=$s" "data.num_workers=$W" "$@" "${DSET[@]}"
}

ces_eval(){   # ces_eval <out>: full test sets + calibration + A1 / per-cue analysis (as CESel)
  local out=$1
  if [ ! -f "$out/eval/summary_calibrated.csv" ]; then
    "$PY" -u scripts/evaluate_ces.py --run "$out" --ckpt best.pt --leak_free "${EV[@]}" "${DEV[@]}" && "$PY" -u scripts/calibrate.py --eval "$out/eval" || return 1
  fi
  [ -f "$out/eval/interp.json" ] || "$PY" -u scripts/ces_posteval.py --run "$out" || return 1
}

ces_validate(){   # ces_validate <run> <json>: validated-cue JSON of <run>/best.pt on the run's validation split
  local run=$1 vj=$2
  [ -f "$vj" ] && return 0
  "$PY" -u scripts/ces_validate_cues.py --config "$run/config.yaml" --ckpt "$run/best.pt" --out "$vj.tmp" "${EV[@]}" \
      --set "data.num_workers=$W" "${DSET[@]}" && mv "$vj.tmp" "$vj"
}

timing(){ echo "$(date -Is) $TASK host=$(hostname -s) job=${PBS_JOBID:-none} gpu=${CUDA_VISIBLE_DEVICES:-cpu} min=$(( ($(date +%s)-T0)/60 ))" >> "$OUT/runs/TIMING_ma.log"; }

if [ "${CESEL_LOCK_TEST:-0}" = 1 ]; then      # lock protocol only: CESEL_LOCK_TEST=1 bash run/ma_run.sh locktest <path>
  [ "$kind" = locktest ] || { echo "CESEL_LOCK_TEST=1 only allows: locktest <path>"; exit 2; }
  take_lock "$1"; echo "ACQUIRED_${TAG:-x} $LOCK_TOKEN"; sleep "${CESEL_LOCK_HOLD:-1}"; exit 0
fi

mkdir -p "$OUT/runs" "$OUT/results" "$OUT/work/tasklocks" "$OUT/runs_ff2public"
T0=$(date +%s)
take_lock "$OUT/work/tasklocks/$TASK"
echo "$(date -Is) task $TASK kind $kind args $* host $(hostname -s) gpu ${CUDA_VISIBLE_DEVICES:-none} dry $DRYM"
case $kind in
  ma)
    cfg=$1; s=$2; out=$OUT/runs/ffpp_${cfg}_s$s
    case $cfg in ma|ma_floor) ;; *) echo "unknown MA config $cfg"; exit 2;; esac
    vj=$CESEL_ROOT/runs/ffpp_B0_s7/validated_cues.json
    [ -f "$vj" ] || { echo "missing $vj"; exit 1; }
    init=$CFES_ROOT/runs/clip_s${s}_p1/last.pt; [ $DRYM = 1 ] && init=null
    ces_train "configs/${cfg}_ffpp.yaml" "$out" "$s" "model.init_backbone_ckpt=$init" "ces.validated_json=$vj" || exit 1
    ces_eval "$out" || exit 1
    ces_validate "$out" "$out/validated_cues_post.json" || exit 1;;
  ma_compare)
    s=$1
    "$PY" -u scripts/ma_compare.py --out_root "$OUT" --seed "$s" || exit 1;;
  clipfair_sfd|clipfair)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    if [ $kind = clipfair_sfd ]; then a=(sfd "$1"); else a=(train "$1" "$2"); fi
    (cd "$CLIPFAIR_ROOT" && CLIPFAIR_OUT=$CLIPFAIR_ROOT CLIPFAIR_WORKERS=$W bash run/clipfair_run.sh "${a[@]}"); rc=$?
    exit $rc;;
  ding_gdagrad)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    s=$1; out=$CLIPFAIR_ROOT/runs/ffpp_ding_gdagrad_s$s; cj=$CLIPFAIR_ROOT/runs/ffpp_ding_sfd_s$s/sfd_channels.json
    [ -f "$cj" ] || { echo "missing $cj (task clipfair_sfd $s)"; exit 1; }
    cd "$CLIPFAIR_ROOT" || exit 2
    if [ ! -f "$out/TRAINED" ]; then
      R=(); [ -f "$out/last.pt" ] && R=(--resume)
      CLIPFAIR_ALLOW_GDA_GRAD=1 "$PY" -u scripts/train_clipfair.py --config configs/ffpp/ding.yaml --out "$out" "${R[@]}" \
          --set "seed=$s" "data.num_workers=$W" "model.init_backbone_ckpt=$CFES_ROOT/runs/clip_s${s}_p1/last.pt" \
          "clipfair.ding.channels_json=$cj" clipfair.ding.gda_grad=true || exit 1
    fi
    if [ ! -f "$out/eval/summary_calibrated.csv" ]; then
      "$PY" -u scripts/evaluate_clipfair.py --run "$out" --ckpt best.pt --leak_free && "$PY" -u scripts/calibrate.py --eval "$out/eval" || exit 1
    fi
    [ -f "$out/eval/a1.json" ] || "$PY" -u scripts/clipfair_posteval.py --run "$out" || exit 1;;
  fairrel)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    bash "$FAIRREL_ROOT/code/run_task.sh" FAIRREL "$1"; rc=$?; [ $rc = 4 ] && rc=1; exit $rc;;
  dingrepro)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    bash "$DING_ROOT/code/run_task_full.sh" "$1"; rc=$?; [ $rc = 4 ] && rc=1; exit $rc;;
  armE)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    (cd "$CFES_ROOT" && CFER_WORKERS=1 bash run/cfer_run.sh fullE "$1" "$2"); exit $?;;
  ff2p1)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    s=$1; out=$OUT/runs_ff2public/clip_s${s}_p1
    [ -f "$out/PUBLIC_P1" ] && exit 0
    R=(); [ -f "$out/last.pt" ] && R=(--resume)
    "$PY" -u scripts/train.py --config configs/ff2_public_p1.yaml --out "$out" "${R[@]}" --set "seed=$s" "data.num_workers=$W" || exit 1
    [ -f "$out/PUBLIC_P1" ] || { echo "phase 1 finished without PUBLIC_P1"; exit 1; };;
  ff2_cfesd)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    s=$1; out=$OUT/runs_ff2public/cfesd_s$s
    if [ ! -f "$out/TRAINED" ]; then
      R=(); [ -f "$out/last.pt" ] && R=(--resume)
      "$PY" -u scripts/train.py --config configs/cfesd_ff2_public.yaml --out "$out" "${R[@]}" --set "seed=$s" \
          "model.init_ckpt=$OUT/runs_ff2public/clip_s${s}_p1/last.pt" "data.num_workers=$W" || exit 1
    fi
    if [ ! -f "$out/eval/summary_calibrated.csv" ]; then
      "$PY" -u scripts/evaluate_cfes.py --run "$out" --ckpt best.pt --leak_free && "$PY" -u scripts/calibrate.py --eval "$out/eval" || exit 1
    fi;;
  ff2_ces)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    cfg=$1; name=$2; s=$3; vj=${4:-}; out=$OUT/runs_ff2public/$name
    VS=(); [ -n "$vj" ] && { [ -f "$vj" ] || { echo "missing $vj"; exit 1; }; VS=("ces.validated_json=$vj"); }
    ces_train "$cfg" "$out" "$s" "model.init_backbone_ckpt=$OUT/runs_ff2public/clip_s${s}_p1/last.pt" "${VS[@]}" || exit 1
    ces_eval "$out" || exit 1;;
  ff2_validate)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    s=$1; ces_validate "$OUT/runs_ff2public/B0_s$s" "$OUT/runs_ff2public/B0_s$s/validated_cues.json" || exit 1;;
  cmd|cmd_cpu)
    [ $DRYM = 1 ] && { stub "$@"; exit 0; }
    [ -n "${TASK_CMD:-}" ] || { echo "cmd task without TASK_CMD"; exit 2; }
    (cd "${TASK_CWD:-$A}" && bash -c "$TASK_CMD"); exit $?;;
  *) echo "unknown kind $kind"; exit 2;;
esac
timing
echo "$(date -Is) task $TASK done"
