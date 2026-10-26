#!/bin/bash
[ -e "${BD_STOP_FILE:-/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/work/STOP_BD}" ] && { echo "Beat-Ding stopped by user (${BD_STOP_FILE:-work/STOP_BD})"; exit 0; }
# Beat-Ding runner (decisions/2026-10-26-beat-ding-plan.md + Amendments 1-2). One task per call; called by the CESEL_MA
# priority runner through run/bd_tasks.json (kind cmd), usable by hand. Output root BD_OUT (default A): runs/, results/.
#   bash run/bd_run.sh smoke [<scratch>]   GPU smoke, bf16: 20 steps of CLIP BD4 (all components, batch 208) from CFES
#                                          runs/clip_s7_p1/last.pt + 20 steps of ResNet-50 BD4 (fp32) from
#                                          runs/resnet_s7_p1/last.pt, then evaluate / calibrate / bd_eval on 256 items per
#                                          set; checks -> runs/SMOKE_OK.json (the bd_smoke done marker; scratch removed)
#   bash run/bd_run.sh arm <BD0..BD4> <s>  CLIP arm from CFES runs/clip_s<s>_p1/last.pt, 10 x 1976 steps (Arm E optimiser)
#                                          -> runs/<arm>_s<s> + evaluate --leak_free + calibrate + bd_eval
#   bash run/bd_run.sh refs                CPU: reference rows (CFES-D s7, Arm E w1/w3/w10 s7, Ding repro) -> results/refs/
#   bash run/bd_run.sh screen              CPU: seed-7 screen report + winner / promotion (plan section 4) -> results/bd_screen.*
#   bash run/bd_run.sh winner <s>          the screen's recipe for seed s: promoted BD arm -> runs/<arm>_s<s>; else the best
#                                          reference (Arm E w: reuse CFES runs_rev4/clip_E_w<w>_s<s> if trained, else train
#                                          configs/rev4/clip_E.yaml w here -> runs/E_w<w>_s<s>; CFES-D: reuse) -> runs/WINNER_s<s>.TRAINED
#   bash run/bd_run.sh winner_eval <s>     evaluate that run (BD tree: evaluate + calibrate + bd_eval; CFES run: bd_eval from its
#                                          saved scores into results/refs/) -> runs/WINNER_s<s>.EVALED
#   bash run/bd_run.sh final               CPU: 3-seed report vs Ding published Tables 1-2 -> results/bd_final.*
#   bash run/bd_run.sh resnet <s>          ResNet-50 (Table 3, Amendment 2) with the CLIP screen's winning component set
#                                          (promoted BD arm, else 'base'): configs/bd_resnet/<set>.yaml from CFES
#                                          runs/resnet_s<s>_p1/last.pt -> runs/R50_<set>_s<s> + eval -> runs/RESNET_s<s>.DONE
#   bash run/bd_run.sh resnet_final        CPU: 3-seed ResNet report vs Ding published Table 3 -> results/bd_final_resnet.*
# Every training run is resumable (--resume when last.pt exists; skipped when TRAINED exists); evaluation steps are
# skipped when their outputs exist.
#
# RUNLOCK: functions copied VERBATIM from agents/CESEL/run/cesel_run.sh (Codex review F1/F11 fix, 25 Oct): token +
# atomic-rename compare-and-swap, heartbeat, fencing, own-generation release; exit 4 = held by another live runner.
# Lock timing env names are the CESel ones (CESEL_LOCK_LEASE / _OBSERVE / _BEAT). All paths are quoted (F11).
# Environment: CUDA_VISIBLE_DEVICES set by the caller (one physical index); BD_WORKERS = loader workers per loader (3
# loaders per run; default 3); BD_DRY=1 = CPU dry run with the stub backbone (needs BD_OUT = scratch and BD_DRY_SETFILE =
# a file of --set overrides, one per line; tests/bd_checks.py); BD_LOCK_TEST=1 ... locktest <path> = lock code only.
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING
CF=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES
OUT=${BD_OUT:-$A}
PY=/home/sachin.chaudhary/.conda/envs/pulid/bin/python
PYD=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
LOCK_LEASE=${CESEL_LOCK_LEASE:-1800}
LOCK_OBSERVE=${CESEL_LOCK_OBSERVE:-240}
LOCK_BEAT=${CESEL_LOCK_BEAT:-60}
cd "$A" || exit 2
kind=$1; shift
DRY=(); EV=(); DRYM=0
CPU_KINDS=" refs screen final resnet_final locktest "
if [ "${BD_DRY:-0}" = 1 ]; then
  [ "$OUT" = "$A" ] && { echo "BD_DRY=1 needs BD_OUT set to a scratch root (never the real run directories)"; exit 2; }
  [ -f "${BD_DRY_SETFILE:-}" ] || { echo "BD_DRY=1 needs BD_DRY_SETFILE (one --set override per line)"; exit 2; }
  mapfile -t DRY < "$BD_DRY_SETFILE"; DRYM=1; EV=(--max_items 64)
elif [ "${BD_LOCK_TEST:-0}" != 1 ] && [[ "$CPU_KINDS" != *" $kind "* ]]; then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "$(date -Is) REFUSE: CUDA_VISIBLE_DEVICES empty"; exit 3; }
  "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "$(date -Is) REFUSE: torch sees no CUDA device"; exit 3; }
fi
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${BD_WORKERS:-3}

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

init_of(){   # init_of <clip|resnet> <seed>: CFES phase-1 checkpoint (null in the dry run)
  [ $DRYM = 1 ] && { echo null; return; }
  if [ "$1" = resnet ]; then echo "$CF/runs/resnet_s${2}_p1/last.pt"; else echo "$CF/runs/clip_s${2}_p1/last.pt"; fi
}

train(){   # train <config> <out> <seed> [--set overrides...]   resumable
  local cfg=$1 out=$2 s=$3; shift 3
  local R=()
  [ -f "$out/TRAINED" ] && return 0
  [ -f "$out/last.pt" ] && R=(--resume)
  "$PY" -u scripts/train.py --config "$cfg" --out "$out" "${R[@]}" --set "seed=$s" "data.num_workers=$W" "$@" "${DRY[@]}"
}

evaluate(){   # evaluate <out>: full test sets (+ leak-free variants) + calibration + Beat-Ding evaluation
  local out=$1
  if [ ! -f "$out/eval/summary_calibrated.csv" ]; then
    "$PY" -u scripts/evaluate.py --run "$out" --ckpt best.pt --leak_free "${EV[@]}" && \
      "$PY" -u scripts/calibrate.py --eval "$out/eval" || return 1
  fi
  [ -f "$out/eval/bd_eval.json" ] || "$PYD" -u scripts/bd_eval.py --run "$out" || return 1
}

screen_field(){   # screen_field <python expr on d = decision>: read results/bd_screen.json
  "$PYD" -c "import json,sys; d=json.load(open(sys.argv[1]))['decision']; print($1)" "$OUT/results/bd_screen.json"
}

atomic_write(){   # atomic_write <file> <content>
  printf '%s\n' "$2" > "$1.tmp.$$" && mv -f "$1.tmp.$$" "$1"
}

timing(){ echo "$(date -Is) $1 host=$(hostname -s) job=${PBS_JOBID:-none} gpu=${CUDA_VISIBLE_DEVICES:-cpu} min=$(( ($(date +%s)-T0)/60 ))" >> "$OUT/runs/TIMING_bd.log"; }

if [ "${BD_LOCK_TEST:-0}" = 1 ]; then      # tests: BD_LOCK_TEST=1 bash run/bd_run.sh locktest <path>
  [ "$kind" = locktest ] || { echo "BD_LOCK_TEST=1 only allows: locktest <path>"; exit 2; }
  take_lock "$1"; echo "ACQUIRED_${TAG:-x} $LOCK_TOKEN"; sleep "${CESEL_LOCK_HOLD:-1}"; exit 0
fi

mkdir -p "$OUT/runs" "$OUT/results"
T0=$(date +%s)
case $kind in
  smoke)
    sd=${1:-$OUT/work/gpu_smoke_${PBS_JOBID:-$$}}; mkdir -p "$sd" || exit 1
    take_lock "$OUT/runs/SMOKE"
    [ -f "$OUT/runs/SMOKE_OK.json" ] && { echo "smoke already passed: $OUT/runs/SMOKE_OK.json"; exit 0; }
    SM=("debug.max_steps_per_epoch=20" "optim.epochs=1" "debug.max_eval_items=256" "log_every=1")
    [ $DRYM = 1 ] && SM=("debug.max_steps_per_epoch=3" "optim.epochs=1" "log_every=1")
    for fam in clip resnet; do
      cfgd=configs/bd; [ "$fam" = resnet ] && cfgd=configs/bd_resnet
      [ $DRYM = 1 ] && [ "$fam" = resnet ] && [ -z "${BD_DRY_RESNET_SETFILE:-}" ] && continue
      XS=(); [ "$fam" = resnet ] && [ $DRYM = 1 ] && mapfile -t XS < "$BD_DRY_RESNET_SETFILE"
      rm -rf "$sd/$fam"      # the smoke always starts fresh
      DRY0=("${DRY[@]}"); [ "$fam" = resnet ] && [ $DRYM = 1 ] && DRY=("${XS[@]}")
      train "$cfgd/BD4.yaml" "$sd/$fam" 7 "model.init_ckpt=$(init_of "$fam" 7)" "${SM[@]}" || exit 1
      DRY=("${DRY0[@]}")
      "$PY" -u scripts/evaluate.py --run "$sd/$fam" --ckpt best.pt --max_items 256 && \
        "$PY" -u scripts/calibrate.py --eval "$sd/$fam/eval" && \
        "$PYD" -u scripts/bd_eval.py --run "$sd/$fam" --n_perm 200 || exit 1
    done
    "$PYD" - "$sd" "$OUT/runs/SMOKE_OK.json" "$DRYM" <<'PYEOF' || exit 1
import json, math, os, sys
sd, dst, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
want = {"det", "pair_det", "strat_det", "stab", "sens", "fprgap", "esa", "gauc"}
rep, ok = {}, True
for fam in ("clip", "resnet"):
    if not os.path.exists(f"{sd}/{fam}/train_log.jsonl"):
        if dry and fam == "resnet":
            continue
        ok = False; rep[fam] = "missing"; continue
    L = [json.loads(l) for l in open(f"{sd}/{fam}/train_log.jsonl")]
    st = [l for l in L if l.get("event") == "step"]
    ep = [l for l in L if l.get("event") == "epoch" and l.get("epoch", -1) >= 0][-1]
    ev = json.load(open(f"{sd}/{fam}/eval/bd_eval.json"))
    t = sorted(l["sec_step"] for l in st[5:]) or [st[-1]["sec_step"]]
    med = t[len(t) // 2]
    terms = sorted(k for k in st[-1] if k in want | {"orth", "risk"})
    finite = all(math.isfinite(l[k]) for l in st for k in terms + ["total"])
    r = {"steps": len(st), "sec_step_median_6_on": med, "sec_step_first": st[0]["sec_step"],
         "projected_hours_10x1976": med * 19760 / 3600, "peak_alloc_gb": ep.get("peak_mem_gb"),
         "peak_reserved_gb": ep.get("peak_reserved_gb"), "terms": terms, "finite": finite,
         "ml_gate_last": st[-1].get("ml_gate"), "aug_frac_mean": sum(l.get("aug_frac", 0) for l in st) / len(st),
         "bd_pool_n": st[-1].get("bd_pool_n"), "esa_cells": st[-1].get("esa_cells"), "gauc_groups": st[-1].get("gauc_groups"),
         "bd_eval_complete": ev.get("complete"), "val_ding_score": ev["val_ding_score"]["score"]}
    good = finite and want <= set(terms) and r["bd_eval_complete"] and r["ml_gate_last"] is not None and \
        r["aug_frac_mean"] > 0 and r["steps"] == (3 if dry else 20)
    ok &= bool(good); r["ok"] = bool(good); rep[fam] = r
    print(f"SMOKE {fam}: {json.dumps(r)}")
if not ok:
    print("SMOKE FAILED"); sys.exit(1)
tmp = dst + ".tmp"; json.dump(rep, open(tmp, "w"), indent=1); os.replace(tmp, dst)
print(f"SMOKE OK -> {dst}")
PYEOF
    [ $DRYM = 1 ] || rm -rf "$sd"
    timing smoke;;
  arm)
    arm=$1; s=$2
    case $arm in BD0|BD1|BD2|BD3|BD4) ;; *) echo "unknown arm $arm"; exit 2;; esac
    out=$OUT/runs/${arm}_s$s
    take_lock "$out"
    train "configs/bd/$arm.yaml" "$out" "$s" "model.init_ckpt=$(init_of clip "$s")" || exit 1
    evaluate "$out" || exit 1
    timing "${arm}_s$s";;
  refs)
    take_lock "$OUT/results/refs"
    "$PYD" -u scripts/bd_report.py refs --runs_root "$OUT/runs" --results "$OUT/results" || exit 1;;
  screen)
    take_lock "$OUT/results/bd_screen"
    "$PYD" -u scripts/bd_report.py screen --runs_root "$OUT/runs" --results "$OUT/results" || exit 1;;
  winner)
    s=$1; [ -f "$OUT/results/bd_screen.json" ] || { echo "missing $OUT/results/bd_screen.json"; exit 1; }
    recipe=$(screen_field "d['recipe']") || exit 1
    prom=$(screen_field "int(d['promoted'])") || exit 1
    m=$OUT/runs/WINNER_s$s.TRAINED
    [ -f "$m" ] && { echo "done: $m"; exit 0; }
    take_lock "$OUT/runs/WINNER_s$s"
    case $recipe in
      BD[0-4])
        out=$OUT/runs/${recipe}_s$s
        train "configs/bd/$recipe.yaml" "$out" "$s" "model.init_ckpt=$(init_of clip "$s")" || exit 1;;
      armE_w*_s7)
        w=${recipe#armE_w}; w=${w%_s7}; out=$CF/runs_rev4/clip_E_w${w}_s$s
        if [ ! -f "$out/TRAINED" ] || [ ! -f "$out/eval/val_scores.npz" ]; then     # not available in CFES: train it here
          out=$OUT/runs/E_w${w}_s$s
          train configs/rev4/clip_E.yaml "$out" "$s" "model.init_ckpt=$(init_of clip "$s")" "loss.fprgap.weight=$w" || exit 1
        fi;;
      CFES-D_s7)
        out=$CF/runs_rev4/clip_D_w0.3_s$s
        [ -f "$out/TRAINED" ] || { echo "CFES-D seed $s run missing: $out"; exit 1; };;
      *) echo "unknown recipe $recipe in bd_screen.json"; exit 2;;
    esac
    atomic_write "$m" "{\"recipe\": \"$recipe\", \"promoted\": $prom, \"seed\": $s, \"run\": \"$out\"}"
    timing "winner_${recipe}_s$s";;
  winner_eval)
    s=$1; m=$OUT/runs/WINNER_s$s.TRAINED; e=$OUT/runs/WINNER_s$s.EVALED
    [ -f "$e" ] && { echo "done: $e"; exit 0; }
    [ -f "$m" ] || { echo "missing $m"; exit 1; }
    run=$("$PYD" -c "import json,sys; print(json.load(open(sys.argv[1]))['run'])" "$m") || exit 1
    recipe=$("$PYD" -c "import json,sys; print(json.load(open(sys.argv[1]))['recipe'])" "$m") || exit 1
    take_lock "$OUT/runs/WINNER_s$s.eval"
    case $run in
      "$OUT"/runs/*) evaluate "$run" || exit 1; res=$run/eval/bd_eval.json
        case $recipe in BD[0-4]) ;; *) mkdir -p "$OUT/results/refs"; cp -f "$res" "$OUT/results/refs/${recipe%_s7}_s$s.json";; esac;;
      *) [ -f "$run/eval/val_scores.npz" ] || { echo "no saved scores in $run/eval"; exit 1; }
         res=$OUT/results/refs/${recipe%_s7}_s$s.json
         [ -f "$res" ] || "$PYD" -u scripts/bd_eval.py --run "$run" --out "$res" --name "${recipe%_s7}_s$s" || exit 1;;
    esac
    atomic_write "$e" "{\"recipe\": \"$recipe\", \"seed\": $s, \"run\": \"$run\", \"bd_eval\": \"$res\"}"
    timing "winner_eval_${recipe}_s$s";;
  final)
    take_lock "$OUT/results/bd_final"
    "$PYD" -u scripts/bd_report.py final --runs_root "$OUT/runs" --results "$OUT/results" || exit 1;;
  resnet)
    s=$1; [ -f "$OUT/results/bd_screen.json" ] || { echo "missing $OUT/results/bd_screen.json"; exit 1; }
    m=$OUT/runs/RESNET_s$s.DONE
    [ -f "$m" ] && { echo "done: $m"; exit 0; }
    comp=$(screen_field "d['winner'].rsplit('_s', 1)[0] if d['promoted'] else 'base'") || exit 1
    case $comp in BD[0-4]) cfg=configs/bd_resnet/$comp.yaml;; base) cfg=configs/bd_resnet/base_E.yaml;; *) echo "bad set $comp"; exit 2;; esac
    out=$OUT/runs/R50_${comp}_s$s
    take_lock "$out"
    XS=(); [ $DRYM = 1 ] && [ -n "${BD_DRY_RESNET_SETFILE:-}" ] && { mapfile -t XS < "$BD_DRY_RESNET_SETFILE"; DRY=("${XS[@]}"); }
    train "$cfg" "$out" "$s" "model.init_ckpt=$(init_of resnet "$s")" || exit 1
    evaluate "$out" || exit 1
    atomic_write "$m" "{\"components\": \"$comp\", \"seed\": $s, \"run\": \"$out\"}"
    timing "R50_${comp}_s$s";;
  resnet_final)
    take_lock "$OUT/results/bd_final_resnet"
    "$PYD" -u scripts/bd_report.py final --family resnet --runs_root "$OUT/runs" --results "$OUT/results" || exit 1;;
  *) echo "unknown kind $kind"; exit 2;;
esac
echo "$(date -Is) bd_run $kind $* done"
