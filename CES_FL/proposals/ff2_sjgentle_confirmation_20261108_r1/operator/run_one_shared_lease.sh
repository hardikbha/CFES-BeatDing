#!/usr/bin/env bash
set -euo pipefail
RUN=${1:?RUN required}
SRC_TREE=${2:?source tree required}
EXPECTED_COMMIT=${3:?commit required}
EXPECTED_CONFIG_SHA=${4:?expected config SHA256 required}
case "$RUN" in SJ_gentle_s7|SJ_gentle_s27) ;; *) echo "bad run name $RUN" >&2; exit 4 ;; esac
ROOT=/home/sachin.chaudhary/ces_fl_runs/ff2_sjgentle_confirmation_20261108_r1
CFG="$SRC_TREE/CES_FL/proposals/ff2_sjgentle_confirmation_20261108_r1/$RUN.json"
PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
MODEL="$SRC_TREE/CES_FL/src/ces_fl_realfpr.py"
LOCKROOT=/home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107/locks
HOST=$(hostname -s)
case "$HOST" in
  gpu01) ALLOW="0,1,2,3,4,6,7" ;;
  gpu02) ALLOW="6,7"; export CUDA_MPS_PIPE_DIRECTORY=/tmp/sc_nomps CUDA_MPS_LOG_DIRECTORY=/tmp/sc_nomps ;;
  *) echo "unapproved host $HOST" >&2; exit 4 ;;
esac
[ -f "$CFG" ] || { echo "missing config $CFG" >&2; exit 5; }
[ -f "$ROOT/$RUN/preflight_rebound/PREFLIGHT.json" ] || { echo "missing fresh preflight" >&2; exit 5; }
[ ! -e "$ROOT/$RUN/out" ] || { echo "refusing existing output path" >&2; exit 5; }
ACTUAL=$EXPECTED_COMMIT
printf '%s  %s\n' 2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1 "$MODEL" | sha256sum -c - >/dev/null || { echo "realfpr source hash mismatch" >&2; exit 5; }
printf '%s  %s\n' 8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733 "$SRC_TREE/CES_FL/src/ces_fl_head_ablation.py" | sha256sum -c - >/dev/null || { echo "head-ablation source hash mismatch" >&2; exit 5; }
printf '%s  %s\n' "$EXPECTED_CONFIG_SHA" "$CFG" | sha256sum -c - >/dev/null || { echo "config hash mismatch" >&2; exit 5; }
mkdir -p "$LOCKROOT" "$ROOT/$RUN" "$ROOT/logs"
alive() { timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=5 "$1" "kill -0 $2" >/dev/null 2>&1; }
take() {
  if mkdir "$1" 2>/dev/null; then printf '%s %s\n' "$HOST" "$$" > "$1/owner"; return 0; fi
  read -r OH OP < "$1/owner" 2>/dev/null || true
  [ -n "${OH:-}" ] && [ -n "${OP:-}" ] && alive "$OH" "$OP" && return 1
  rm -rf "$1"; mkdir "$1" 2>/dev/null || return 1
  printf '%s %s\n' "$HOST" "$$" > "$1/owner"
}
mine() { read -r OH OP < "$1/owner" 2>/dev/null || true; [ "${OH:-} ${OP:-}" = "$HOST $$" ]; }
take "$LOCKROOT/run_$RUN" || { echo "run lock busy: $RUN"; exit 7; }
G=""
cleanup() { [ -n "$G" ] && mine "$LOCKROOT/gpu_${HOST}_${G}" && rm -rf "$LOCKROOT/gpu_${HOST}_${G}"; mine "$LOCKROOT/run_$RUN" && rm -rf "$LOCKROOT/run_$RUN"; }
trap cleanup EXIT
N=0
for d in "$LOCKROOT"/gpu_*; do [ -d "$d" ] || continue; read -r OH OP < "$d/owner" 2>/dev/null || true; [ -n "${OH:-}" ] && [ -n "${OP:-}" ] && alive "$OH" "$OP" && N=$((N+1)); done
[ "$N" -lt 3 ] || { echo "three shared leases already active"; exit 3; }
for row in $(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v a=",$ALLOW," 'index(a,","$1",") && $3<10 && $2>=20480 {print $2 ":" $1}' | sort -rn); do
  g=${row#*:}; if take "$LOCKROOT/gpu_${HOST}_${g}"; then G=$g; break; fi
done
[ -n "$G" ] || { echo "no idle allowed shared GPU on $HOST"; exit 3; }
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
UUID=$(nvidia-smi --query-gpu=uuid,memory.free,utilization.gpu --format=csv,noheader -i "$G")
$PY -c 'import json,os,torch,sys; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps({"visible":os.environ["CUDA_VISIBLE_DEVICES"],"physical_uuid_memory_util":sys.argv[1],"device_name":torch.cuda.get_device_name(0),"device_count":torch.cuda.device_count(),"commit_label":sys.argv[2],"config_sha256":sys.argv[3]}))' "$UUID" "$EXPECTED_COMMIT" "$EXPECTED_CONFIG_SHA" > "$ROOT/$RUN/CUDA_BINDING.json"
T0=$(date +%s)
echo "$(date -Is) start $RUN host=$HOST visible=$CUDA_VISIBLE_DEVICES $UUID pid=$$ commit=$ACTUAL"
cd "$ROOT/$RUN"
set +e
$PY -u "$MODEL" --config "$CFG" --preflight "$ROOT/$RUN/preflight_rebound" --out "$ROOT/$RUN/out" --budget-hours 4 >> "$ROOT/logs/$RUN.log" 2>&1
RC=$?
set -e
T1=$(date +%s)
$PY -c 'import json,sys; json.dump({"host":sys.argv[1],"gpu_uuid_memory_util":sys.argv[2],"exit_code":int(sys.argv[3]),"start_unix":int(sys.argv[4]),"end_unix":int(sys.argv[5]),"elapsed_gpu_hours":(int(sys.argv[5])-int(sys.argv[4]))/3600,"commit":sys.argv[6]},open(sys.argv[7],"w"),indent=2)' "$HOST" "$UUID" "$RC" "$T0" "$T1" "$ACTUAL" "$ROOT/$RUN/ALLOCATION_END.json"
echo "$(date -Is) end $RUN rc=$RC elapsed_hours=$(python -c "print(($T1-$T0)/3600)")"
exit "$RC"
