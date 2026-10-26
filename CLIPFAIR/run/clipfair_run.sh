#!/bin/bash
[ -e /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR/work/STOP_CLIPFAIR ] && { echo "CLIPFAIR stopped by user (work/STOP_CLIPFAIR)"; exit 0; }
# One CLIPFAIR step (decisions/2026-10-25-cesel-plan.md, Amendment 3); called by scripts/clipfair_pipeline.py, usable by
# hand. Output root CLIPFAIR_OUT (default A); runs in <root>/runs/.
#   bash run/clipfair_run.sh sfd   <seed>            Ding SFD on CFES runs/clip_s<seed>_p1 (whole FF++ train set)
#                                                    -> runs/ffpp_ding_sfd_s<seed>/sfd_channels.json
#   bash run/clipfair_run.sh train <method> <seed>   method in daw dag pgfdd ding; from CFES runs/clip_s<seed>_p1/last.pt,
#                                                    B0 budget (10 x 1976 steps), Amendment-9 selection -> runs/ffpp_<method>_s<seed>
#   bash run/clipfair_run.sh smoke <scratch dir>     GPU smoke: 20 bf16 steps per baseline (Ding: SFD on 20 batches first),
#                                                    then eval / calibrate / A1 on 256 items per set, and the report
# Every training run: resumable (--resume when last.pt exists; skipped when TRAINED exists), then evaluate_clipfair.py
# --leak_free on the FULL test sets, calibrate.py (intercept on FF++ val), clipfair_posteval.py (A1).
# One runner per output directory: <out>.RUNLOCK = the CESel RUNLOCK protocol (Codex review F1 fix, 25 Oct; copied from
# agents/CESEL/run/cesel_run.sh, which documents it): ownership token published by one atomic rename, takeover = atomic
# rename of the observed generation to its tombstone (compare-and-swap), stale only when the heartbeat counter did not
# advance during an observation window AND its clock age > lease (or same host + dead pid), fencing, release of the own
# generation only. Exit 4 = held elsewhere. Lock-only test: CESEL_LOCK_TEST=1 bash run/clipfair_run.sh locktest <path>.
# Environment: CUDA_VISIBLE_DEVICES set by the caller (one physical index); CLIPFAIR_WORKERS = loader workers per loader
# (3 loaders per run; default 2); CLIPFAIR_DRY=1 = tiny CPU dry run with the stub backbone (needs CLIPFAIR_OUT = scratch).
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR
CF=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES
OUT=${CLIPFAIR_OUT:-$A}
PY=/home/sachin.chaudhary/.conda/envs/pulid/bin/python
LOCK_LEASE=${CESEL_LOCK_LEASE:-1800}
LOCK_OBSERVE=${CESEL_LOCK_OBSERVE:-240}
LOCK_BEAT=${CESEL_LOCK_BEAT:-60}
cd "$A" || exit 2
kind=$1; shift
DRY=""; EV=""; DEV=""
if [ "${CLIPFAIR_DRY:-0}" = 1 ]; then
  [ "$OUT" = "$A" ] && { echo "CLIPFAIR_DRY=1 needs CLIPFAIR_OUT set to a scratch root (never the real run directories)"; exit 2; }
  DRY="model.backend=stub model.stub={dim:32,grid:8} data.input_size=64 debug.max_steps_per_epoch=2 debug.max_eval_items=64 data.batch_size=8 pairs.per_step=2 strat.per_cell=1 eval.batch_size=32 debug.max_pairs=64 log_every=1 optim.epochs=1 clipfair.ding.sfd_max_batches=3 clipfair.ding.sfd_batch_size=16"
  EV="--max_items 64"; DEV="--device cpu"
elif [ "${CESEL_LOCK_TEST:-0}" != 1 ]; then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "$(date -Is) REFUSE: CUDA_VISIBLE_DEVICES empty"; exit 3; }
  $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "$(date -Is) REFUSE: torch sees no CUDA device"; exit 3; }
fi
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${CLIPFAIR_WORKERS:-2}

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
    # @lock-test-barrier (CESEL tests/test_runlock_f1.sh inserts a scheduling barrier here)
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

sfd(){   # sfd <out> <seed> [--set overrides...]
  local out=$1 s=$2; shift 2
  [ -f $out/sfd_channels.json ] && return 0
  local init=$CF/runs/clip_s${s}_p1/last.pt; [ -n "$DRY" ] && init=null
  $PY -u scripts/ding_sfd.py --config configs/ffpp/ding.yaml --out $out $DEV --set seed=$s data.num_workers=$((W*3)) \
      model.init_backbone_ckpt=$init "$@" $DRY
}

train(){   # train <method> <out> <seed> [--set overrides...]
  local m=$1 out=$2 s=$3; shift 3
  local R="" init=$CF/runs/clip_s${s}_p1/last.pt; [ -n "$DRY" ] && init=null
  [ -f $out/TRAINED ] && return 0
  [ -f $out/last.pt ] && R=--resume
  $PY -u scripts/train_clipfair.py --config configs/ffpp/$m.yaml --out $out $R $DEV --set seed=$s data.num_workers=$W \
      model.init_backbone_ckpt=$init "$@" $DRY
}

evaluate(){   # evaluate <out> [--max_items N]
  local out=$1; shift
  if [ ! -f $out/eval/summary_calibrated.csv ]; then
    $PY -u scripts/evaluate_clipfair.py --run $out --ckpt best.pt --leak_free $EV $DEV "$@" && \
      $PY -u scripts/calibrate.py --eval $out/eval || return 1
  fi
  [ -f $out/eval/a1.json ] || $PY -u scripts/clipfair_posteval.py --run $out || return 1
}

timing(){ echo "$(date -Is) $1 host=$(hostname -s) job=${PBS_JOBID:-none} gpu=${CUDA_VISIBLE_DEVICES:-cpu} min=$(( ($(date +%s)-T0)/60 ))" >> $OUT/runs/TIMING_clipfair.log; }

if [ "${CESEL_LOCK_TEST:-0}" = 1 ]; then      # lock protocol only: CESEL_LOCK_TEST=1 bash run/clipfair_run.sh locktest <path>
  [ "$kind" = locktest ] || { echo "CESEL_LOCK_TEST=1 only allows: locktest <path>"; exit 2; }
  take_lock "$1"; echo "ACQUIRED_${TAG:-x} $LOCK_TOKEN"; sleep "${CESEL_LOCK_HOLD:-1}"; exit 0
fi

mkdir -p "$OUT/runs" "$OUT/results"
T0=$(date +%s)
case $kind in
  sfd)
    s=$1; out=$OUT/runs/ffpp_ding_sfd_s$s
    take_lock "$out"
    sfd $out $s || exit 1
    timing ffpp_ding_sfd_s$s;;
  train)
    m=$1; s=$2; out=$OUT/runs/ffpp_${m}_s$s
    case $m in daw|dag|pgfdd|ding) ;; *) echo "unknown method $m"; exit 2;; esac
    EXTRA=""
    if [ $m = ding ]; then
      cj=$OUT/runs/ffpp_ding_sfd_s$s/sfd_channels.json
      [ -f $cj ] || { echo "missing $cj (run: clipfair_run.sh sfd $s)"; exit 1; }
      EXTRA="clipfair.ding.channels_json=$cj"
    fi
    take_lock "$out"
    train $m $out $s $EXTRA || exit 1
    evaluate $out || exit 1
    timing ffpp_${m}_s$s;;
  smoke)
    sd=$1; mkdir -p $sd/runs || exit 1
    SM="debug.max_steps_per_epoch=20 optim.epochs=1 debug.max_eval_items=256 log_every=1"
    [ -n "$DRY" ] && SM=""
    sfd $sd/runs/ffpp_ding_sfd_s7 7 clipfair.ding.sfd_max_batches=20 || exit 1
    [ -z "$DRY" ] && EV="--max_items 256"
    for m in daw dag pgfdd ding; do
      EXTRA=""; [ $m = ding ] && EXTRA="clipfair.ding.channels_json=$sd/runs/ffpp_ding_sfd_s7/sfd_channels.json"
      train $m $sd/runs/ffpp_${m}_s7 7 $SM $EXTRA || exit 1
      evaluate $sd/runs/ffpp_${m}_s7 || exit 1
    done
    $PY - $sd <<'PYEOF'
import json, sys, numpy as np
for m in ("daw", "dag", "pgfdd", "ding"):
    L = [json.loads(l) for l in open(f"{sys.argv[1]}/runs/ffpp_{m}_s7/train_log.jsonl")]
    st = [l for l in L if l.get("event") == "step"]
    ep = [l for l in L if l.get("event") == "epoch" and l.get("epoch", -1) >= 0][-1]
    t = np.array([l["sec_step"] for l in st])
    terms = sorted(k for k in st[-1] if k in ("daw", "dag", "sha", "spe", "rec", "con", "fair", "fuse", "ce", "gda_fair", "orth"))
    print(f"SMOKE {m}: steps {len(st)}, s/step median(6-20) {np.median(t[5:]):.3f} first {t[0]:.1f}; peak allocated "
          f"{ep['peak_mem_gb']:.1f} GB, reserved {ep['peak_reserved_gb']:.1f} GB; terms {terms}; last total {st[-1]['total']:.4f} "
          f"val_auc {ep['val_auc']:.2f}" + (f"; gda_fair {st[-1].get('gda_fair'):.4f} requires_grad {st[-1].get('gda_fair_requires_grad')}" if m == "ding" else ""))
print("SFD", json.load(open(f"{sys.argv[1]}/runs/ffpp_ding_sfd_s7/sfd_channels.json"))["channels"])
PYEOF
    /home/sachin.chaudhary/.conda/envs/aiface/bin/python scripts/clipfair_report.py --out_root $sd --seeds 7 --partial || exit 1
    echo "SMOKE OK";;
  *) echo "unknown kind $kind"; exit 2;;
esac
