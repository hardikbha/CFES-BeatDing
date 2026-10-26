#!/bin/bash
[ -e "${CESEL_STOP_FILE:-/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/work/STOP_CESEL}" ] && { echo "CESel stopped by user (${CESEL_STOP_FILE:-work/STOP_CESEL})"; exit 0; }
# One CESel FF++ screen step (decisions/2026-10-25-cesel-plan.md + Amendment 1); called by scripts/cesel_pipeline.py,
# usable by hand. Output root CESEL_OUT (default A); runs in <root>/runs/.
#   bash run/cesel_run.sh B0       <seed>          B0 from CFES runs/clip_s<seed>_p1/last.pt (backbone only), CFES-D
#                                                  phase-2 budget (10 x 1976 steps), Amendment-9 selection -> runs/ffpp_B0_s<seed>
#   bash run/cesel_run.sh validate <seed>          ces_validate_cues.py on B0 best.pt (FF++ VALIDATION only)
#                                                  -> runs/ffpp_B0_s<seed>/validated_cues.json
#   bash run/cesel_run.sh arm      <arm> <seed>    arm from B0 best.pt + validated JSON, extra budget X5 = 5 x 1976 steps
#                                                  -> runs/ffpp_x5_<arm>_s<seed>  (arm B0 = "B0-continued")
#   bash run/cesel_run.sh arm_p1   <arm> <seed>    Amendment 5 ("arms like B0"): arm trained EXACTLY like B0, from CFES
#                                                  runs/clip_s<seed>_p1/last.pt (backbone only; fresh heads, branches,
#                                                  CES state), 10 x 1976 steps (config default, as B0), Amendment-9
#                                                  selection with the start checkpoint as reference, B0's validated-cue
#                                                  JSON -> runs/ffpp_p1_<arm>_s<seed>   (arms: all but B0)
#   bash run/cesel_run.sh starve   <run dir> <tag> [<results subdir, default b0_cue_starvation>]
#                                                  per-group cue-starvation table (CPU) -> results/<subdir>/
#   bash run/cesel_run.sh smoke    <scratch dir>   GPU smoke: 20 bf16 steps of Full and of B0+CFES-D (batch 176 / 240)
#                                                  from clip_s7_p1, then eval / calibrate / posteval / validate on 256 items
# Every training run: resumable (--resume when last.pt exists; skipped when TRAINED exists), then evaluate_ces.py
# --leak_free on the FULL test sets, calibrate.py (intercept on FF++ val), ces_posteval.py (A1 + per-cue analysis).
#
# RUNLOCK (Codex review F1/F11 fix, 25 Oct): one runner per output directory, across hosts (shared Lustre).
#   * A lock is the directory <out>.RUNLOCK whose owner file carries a unique ownership TOKEN, written once, BEFORE the
#     lock becomes visible: the runner builds <out>.RUNLOCK.new.<token>/ (owner + heartbeat) and publishes it with one
#     atomic rename (mv -T); the rename fails when <out>.RUNLOCK already exists, so a fresh lock has exactly one owner.
#   * Retiring a lock (owner release or stale takeover) is the atomic rename <out>.RUNLOCK -> <out>.RUNLOCK.gone.<T>,
#     T = the token READ from the lock. The tombstone .gone.<T> is never removed (only tombstones older than 7 days are
#     swept), so generation T can be retired exactly once: a second contender that saw the same stale T, or an old
#     owner whose lock was taken over, fails its rename (target exists and is non-empty) and backs off. A rename that
#     succeeds therefore retired exactly the generation that was observed (compare-and-swap on the token).
#   * Stale = (a) heartbeat not advanced during an observation window of CESEL_LOCK_OBSERVE s (default 240 s = 4
#     heartbeat periods) AND its clock age > CESEL_LOCK_LEASE s (default 1800 s; gpu01/gpu02 run ~11 min behind the
#     login node, so clock age alone is not trusted), or (b) same host and the owner pid is dead. A live run touches
#     its heartbeat (a counter) every 60 s and is never fenced out by a stale clock.
#   * Fencing: the heartbeat loop stops refreshing and terminates the run's children if its own generation was
#     retired (tombstone .gone.<own token> appears), so a fenced-out runner cannot keep writing next to its successor.
#   * The EXIT trap releases only its own generation (rename to .gone.<own token>); it never deletes a successor's lock.
#   * Exit 4 = the run is held by another live runner (or another contender won the takeover race).
# Environment: CUDA_VISIBLE_DEVICES set by the caller (one physical index); CESEL_WORKERS = loader workers per loader
# (3 loaders per run; default 2); CESEL_DRY=1 = tiny CPU dry run with the stub backbone (needs CESEL_OUT = scratch).
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL
CF=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES
OUT=${CESEL_OUT:-$A}
PY=/home/sachin.chaudhary/.conda/envs/pulid/bin/python
X5_EPOCHS=5                 # Amendment 1: extra budget of every arm = 5 epochs x 1976 steps
LOCK_LEASE=${CESEL_LOCK_LEASE:-1800}
LOCK_OBSERVE=${CESEL_LOCK_OBSERVE:-240}
LOCK_BEAT=${CESEL_LOCK_BEAT:-60}
cd "$A" || exit 2
kind=$1; shift
DRY=(); EV=(); DEV=()
if [ "${CESEL_DRY:-0}" = 1 ]; then
  [ "$OUT" = "$A" ] && { echo "CESEL_DRY=1 needs CESEL_OUT set to a scratch root (never the real run directories)"; exit 2; }
  DRY=("model.backend=stub" "model.stub={dim:32,grid:8}" "data.input_size=64" "ces.dim=16" "ces.hidden=32"
       "ces.freq_hidden=16" "ces.freq_widths=[8,8,16,16]" "debug.max_steps_per_epoch=2" "debug.max_eval_items=64"
       "data.batch_size=8" "pairs.per_step=2" "strat.per_cell=1" "eval.batch_size=32" "debug.max_pairs=64"
       "loss.inv.warmup_steps=1" "log_every=1")
  EV=(--max_items 64); DEV=(--device cpu)
elif [ "${CESEL_LOCK_TEST:-0}" != 1 ]; then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || [ "$kind" = starve ] || { echo "$(date -Is) REFUSE: CUDA_VISIBLE_DEVICES empty"; exit 3; }
  if [ "$kind" != starve ]; then
    "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "$(date -Is) REFUSE: torch sees no CUDA device"; exit 3; }
  fi
fi
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${CESEL_WORKERS:-2}

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

train(){   # train <config> <out> <seed> [--set overrides...]
  local cfg=$1 out=$2 s=$3; shift 3
  local R=()
  [ -f "$out/TRAINED" ] && return 0
  [ -f "$out/last.pt" ] && R=(--resume)
  "$PY" -u scripts/train_ces.py --config "$cfg" --out "$out" "${R[@]}" "${DEV[@]}" --set "seed=$s" "data.num_workers=$W" "$@" "${DRY[@]}"
}

evaluate(){   # evaluate <out>: full test sets + calibration + A1 / per-cue analysis
  local out=$1
  if [ ! -f "$out/eval/summary_calibrated.csv" ]; then
    "$PY" -u scripts/evaluate_ces.py --run "$out" --ckpt best.pt --leak_free "${EV[@]}" "${DEV[@]}" && "$PY" -u scripts/calibrate.py --eval "$out/eval" || return 1
  fi
  [ -f "$out/eval/interp.json" ] || "$PY" -u scripts/ces_posteval.py --run "$out" || return 1
}

timing(){ echo "$(date -Is) $1 host=$(hostname -s) job=${PBS_JOBID:-none} gpu=${CUDA_VISIBLE_DEVICES:-cpu} min=$(( ($(date +%s)-T0)/60 ))" >> "$OUT/runs/TIMING_cesel.log"; }

if [ "${CESEL_LOCK_TEST:-0}" = 1 ]; then      # tests/test_runlock_f1.sh: CESEL_LOCK_TEST=1 bash cesel_run.sh locktest <path>
  [ "$kind" = locktest ] || { echo "CESEL_LOCK_TEST=1 only allows: locktest <path>"; exit 2; }
  take_lock "$1"; echo "ACQUIRED_${TAG:-x} $LOCK_TOKEN"; sleep "${CESEL_LOCK_HOLD:-1}"; exit 0
fi

mkdir -p "$OUT/runs" "$OUT/results"
T0=$(date +%s)
case $kind in
  B0)
    s=$1; out=$OUT/runs/ffpp_B0_s$s
    init=$CF/runs/clip_s${s}_p1/last.pt; [ ${#DRY[@]} -gt 0 ] && init=null
    take_lock "$out"
    train configs/ffpp/B0.yaml "$out" "$s" "model.init_backbone_ckpt=$init" || exit 1
    evaluate "$out" || exit 1
    timing "ffpp_B0_s$s";;
  validate)
    s=$1; b0=$OUT/runs/ffpp_B0_s$s; vj=$b0/validated_cues.json
    [ -f "$vj" ] && exit 0
    [ -f "$b0/TRAINED" ] || { echo "missing $b0/TRAINED"; exit 1; }
    take_lock "$b0.validate"
    "$PY" -u scripts/ces_validate_cues.py --config "$b0/config.yaml" --ckpt "$b0/best.pt" --out "$vj.tmp" "${EV[@]}" \
        --set "data.num_workers=$W" "${DRY[@]}" && mv "$vj.tmp" "$vj" || exit 1
    timing "validate_B0_s$s";;
  arm)
    arm=$1; s=$2; b0=$OUT/runs/ffpp_B0_s$s; out=$OUT/runs/ffpp_x5_${arm}_s$s
    case $arm in B0|sens|inv|floor|full|C_randcue|C_shufgroup|C_novalid|B0_cfesD) ;; *) echo "unknown arm $arm"; exit 2;; esac
    [ -f "$b0/validated_cues.json" ] || { echo "missing $b0/validated_cues.json"; exit 1; }
    take_lock "$out"
    ep=$X5_EPOCHS; [ ${#DRY[@]} -gt 0 ] && ep=1
    train "configs/ffpp/$arm.yaml" "$out" "$s" "model.init_ckpt=$b0/best.pt" model.init_backbone_ckpt=null \
          "ces.validated_json=$b0/validated_cues.json" "optim.epochs=$ep" || exit 1
    evaluate "$out" || exit 1
    timing "ffpp_x5_${arm}_s$s";;
  arm_p1)
    arm=$1; s=$2; b0=$OUT/runs/ffpp_B0_s$s; out=$OUT/runs/ffpp_p1_${arm}_s$s
    case $arm in sens|inv|floor|full|C_randcue|C_shufgroup|C_novalid|B0_cfesD) ;; *) echo "unknown p1 arm $arm"; exit 2;; esac
    [ -f "$b0/validated_cues.json" ] || { echo "missing $b0/validated_cues.json"; exit 1; }
    init=$CF/runs/clip_s${s}_p1/last.pt; [ ${#DRY[@]} -gt 0 ] && init=null
    take_lock "$out"
    train "configs/ffpp/$arm.yaml" "$out" "$s" "model.init_backbone_ckpt=$init" \
          "ces.validated_json=$b0/validated_cues.json" || exit 1
    evaluate "$out" || exit 1
    timing "ffpp_p1_${arm}_s$s";;
  starve)
    run=$1; tag=$2; sub=${3:-b0_cue_starvation}; mkdir -p "$OUT/results/$sub"
    [ -f "$OUT/results/$sub/${tag}_cue_starvation.done.json" ] && exit 0
    take_lock "$OUT/results/$sub/$tag"
    SB=(); [ -n "${CESEL_STARVE_B:-}" ] && SB=(--B "$CESEL_STARVE_B")
    "$PY" -u scripts/ces_cue_starvation.py --run "$run" --out_dir "$OUT/results/$sub" --tag "$tag" "${SB[@]}" || exit 1;;
  smoke)
    sd=$1; mkdir -p "$sd" || exit 1
    SM=("debug.max_steps_per_epoch=20" "optim.epochs=1" "debug.max_eval_items=256" "log_every=1" "loss.inv.warmup_steps=5")
    [ ${#DRY[@]} -gt 0 ] && SM=()
    init=$CF/runs/clip_s7_p1/last.pt; [ ${#DRY[@]} -gt 0 ] && init=null
    for arm in full B0_cfesD; do
      train "configs/ffpp/$arm.yaml" "$sd/$arm" 7 "model.init_backbone_ckpt=$init" "${SM[@]}" || exit 1
    done
    "$PY" -u scripts/evaluate_ces.py --run "$sd/full" --ckpt best.pt --leak_free --max_items 256 "${DEV[@]}" && \
      "$PY" -u scripts/calibrate.py --eval "$sd/full/eval" && "$PY" -u scripts/ces_posteval.py --run "$sd/full" && \
      "$PY" -u scripts/ces_validate_cues.py --config "$sd/full/config.yaml" --ckpt "$sd/full/best.pt" --out "$sd/full/vc.json" \
          --max_items 256 --set "data.num_workers=$W" "${DRY[@]}" && \
      "$PY" -u scripts/ces_cue_starvation.py --run "$sd/full" --out_dir "$sd/starve" --tag smoke --B 50 || exit 1
    "$PY" - "$sd" <<'EOF'
import json, sys, numpy as np
for arm in ("full", "B0_cfesD"):
    L = [json.loads(l) for l in open(f"{sys.argv[1]}/{arm}/train_log.jsonl")]
    st = [l for l in L if l.get("event") == "step"]
    ep = [l for l in L if l.get("event") == "epoch" and l.get("epoch", -1) >= 0][-1]
    t = np.array([l["sec_step"] for l in st])
    print(f"SMOKE {arm}: steps {len(st)}, s/step median(steps 6-20) {np.median(t[5:]):.3f} mean(6-20) {t[5:].mean():.3f} "
          f"first {t[0]:.1f}; peak allocated {ep['peak_mem_gb']:.1f} GB, reserved {ep['peak_reserved_gb']:.1f} GB; "
          f"terms {sorted(k for k in st[-1] if k in ('det','pair_det','strat_det','sens_ces','inv','floor','stab','sens','risk','orth'))}; "
          f"inv_ready {st[-1].get('inv_ready')} inv_valid_rate {st[-1].get('inv_valid_rate')}")
EOF
    echo "SMOKE OK";;
  *) echo "unknown kind $kind"; exit 2;;
esac
