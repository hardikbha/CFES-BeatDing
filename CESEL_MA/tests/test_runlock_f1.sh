#!/bin/bash
# Codex review F1 / F11 check: RUNLOCK protocol of run/cesel_run.sh (lock code only: CESEL_LOCK_TEST=1 ... locktest,
# no training, no GPU, no PBS). Runs on the shared Lustre filesystem (tests/_runlock_scratch, removed at the end).
#   bash tests/test_runlock_f1.sh [script]     (default run/cesel_run.sh)  -> prints PASS lines, ALL_PASS at the end
# Cases: (1) Codex interleaving: N contenders all decide "stale" (barrier inserted right before the compare-and-swap)
# -> exactly one ACQUIRED; (2) same with a dead-pid owner on this host; (3) fresh live lock -> contender exits 4;
# (4) live owner whose heartbeat clock looks old (clock skew) is NOT fenced out (heartbeat advances while observed);
# (5) an owner whose generation was taken over never removes the successor's lock on exit, and (6) is fenced
# (its run is terminated) when its generation is retired; (7) paths with spaces (F11).
T=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/tests
SRC=${1:-$T/../run/cesel_run.sh}
S="$T/_runlock_scratch"
rm -rf "$S"; mkdir -p "$S"
fail(){ echo "FAIL: $*"; exit 1; }
H=$(hostname -s)
# harness = the real script with a scheduling barrier at the marker (all contenders pass the staleness decision first)
sed 's|^    # @lock-test-barrier.*|    touch "$BARRIER/ready_$TAG"; for _i in $(seq 1 600); do [ "$(ls "$BARRIER" \| wc -l)" -ge "$NCONT" ] \&\& break; sleep 0.05; done|' \
  "$SRC" > "$S/harness.sh"
grep -q 'ready_\$TAG' "$S/harness.sh" || fail "barrier marker not found in $SRC"
export CESEL_LOCK_TEST=1 CESEL_OUT="$S/out"
mkdir -p "$S/out"

race(){   # race <name> <ncont> <owner line> <heartbeat age spec>
  local name=$1 n=$2 owner=$3 age=$4 d="$S/$1" i
  mkdir -p "$d/bar"
  mkdir "$d/run x.RUNLOCK"; printf '%s\n' "$owner" > "$d/run x.RUNLOCK/owner"; echo 7 > "$d/run x.RUNLOCK/heartbeat"
  touch -d "$age" "$d/run x.RUNLOCK/heartbeat" "$d/run x.RUNLOCK"
  for i in $(seq 1 "$n"); do
    BARRIER="$d/bar" NCONT=$n TAG=$i CESEL_LOCK_LEASE=60 CESEL_LOCK_OBSERVE=2 CESEL_LOCK_BEAT=1 CESEL_LOCK_HOLD=3 \
      bash "$S/harness.sh" locktest "$d/run x" > "$d/c$i.log" 2>&1 &
  done
  wait
  local acq; acq=$(cat "$d"/c*.log | grep -c '^ACQUIRED_')
  [ "$acq" = 1 ] || { cat "$d"/c*.log; fail "$name: $acq contenders ACQUIRED (expected exactly 1)"; }
  local won; won=$(grep -l '^ACQUIRED_' "$d"/c*.log); local tok; tok=$(awk '/^ACQUIRED_/{print $2}' "$won")
  [ -d "$d/run x.RUNLOCK.gone.$tok" ] || fail "$name: winner did not release its own generation"
  [ ! -e "$d/run x.RUNLOCK" ] || fail "$name: lock left behind after the winner exited"
  local skipped; skipped=$(grep -l 'skip ' "$d"/c*.log | wc -l)
  [ "$skipped" = $((n - 1)) ] || { cat "$d"/c*.log; fail "$name: $skipped losers exited with skip (expected $((n - 1)))"; }
  echo "PASS $name: $n contenders all judged the lock stale, exactly 1 ACQUIRED, $((n - 1)) backed off (exit 4)"
}
race stale_heartbeat_2 2 "otherhost nojob 99999999" "2 hours ago"
race stale_heartbeat_8 8 "token=oldgen host=otherhost job=nojob pid=1 since=x" "2 hours ago"
race dead_pid_same_host_6 6 "token=deadgen host=$H job=nojob pid=99999999 since=x" "now"

# (3) fresh live lock
d="$S/fresh"; mkdir -p "$d"
CESEL_LOCK_BEAT=1 CESEL_LOCK_HOLD=6 TAG=owner bash "$SRC" locktest "$d/run" > "$d/owner.log" 2>&1 &
OWN=$!; sleep 1.5
CESEL_LOCK_LEASE=60 CESEL_LOCK_OBSERVE=2 TAG=c bash "$SRC" locktest "$d/run" > "$d/c.log" 2>&1; rc=$?
[ $rc = 4 ] || { cat "$d/c.log"; fail "fresh: contender rc=$rc (expected 4)"; }
echo "PASS fresh live lock: contender exits 4"

# (4) clock-skewed live owner: lease -1 s makes ANY clock age look expired (= arbitrary skew); the owner is still not
# fenced out because its heartbeat counter advances while observed
CESEL_LOCK_LEASE=-1 CESEL_LOCK_OBSERVE=2.5 TAG=c2 bash "$SRC" locktest "$d/run" > "$d/c2.log" 2>&1; rc=$?
[ $rc = 4 ] && grep -q 'heartbeat advanced' "$d/c2.log" || { cat "$d/c2.log"; fail "skew: live owner fenced out (rc=$rc)"; }
wait $OWN; grep -q '^ACQUIRED_owner' "$d/owner.log" || fail "skew: owner log"
echo "PASS live owner with a 2-h-old heartbeat clock is not taken over (heartbeat advanced while observed)"

# (5) owner A taken over by B: A's exit must not remove B's lock
d="$S/takeover"; mkdir -p "$d"
CESEL_LOCK_BEAT=1000 CESEL_LOCK_HOLD=8 TAG=A bash "$SRC" locktest "$d/run" > "$d/A.log" 2>&1 &
PA=$!; sleep 1.5
touch -d "2 hours ago" "$d/run.RUNLOCK/heartbeat"
CESEL_LOCK_LEASE=-1 CESEL_LOCK_OBSERVE=1 CESEL_LOCK_BEAT=1000 CESEL_LOCK_HOLD=12 TAG=B bash "$SRC" locktest "$d/run" > "$d/B.log" 2>&1 &
PB=$!
wait $PA
sleep 0.5
grep -q '^ACQUIRED_B' "$d/B.log" || { cat "$d/B.log"; fail "takeover: B did not take over"; }
tokB=$(awk '/^ACQUIRED_B/{print $2}' "$d/B.log")
[ "$(head -n1 "$d/run.RUNLOCK/owner" | sed 's/^token=\([^ ]*\).*/\1/')" = "$tokB" ] || fail "takeover: A's exit removed or replaced B's lock"
wait $PB
[ ! -e "$d/run.RUNLOCK" ] && [ -d "$d/run.RUNLOCK.gone.$tokB" ] || fail "takeover: B did not release its own generation"
echo "PASS taken-over owner exits without touching the successor's lock; successor releases its own generation"

# (6) fencing: retire A's generation behind its back -> A's heartbeat loop terminates A's run
d="$S/fence"; mkdir -p "$d"
CESEL_LOCK_BEAT=1 CESEL_LOCK_HOLD=30 TAG=A bash "$SRC" locktest "$d/run" > "$d/A.log" 2>&1 &
PA=$!; sleep 1.5
tokA=$(head -n1 "$d/run.RUNLOCK/owner" | sed 's/^token=\([^ ]*\).*/\1/')
mv -T "$d/run.RUNLOCK" "$d/run.RUNLOCK.gone.$tokA"
t0=$(date +%s); wait $PA; dt=$(( $(date +%s) - t0 ))
[ $dt -lt 10 ] && grep -q 'RUNLOCK LOST' "$d/A.log" || { cat "$d/A.log"; fail "fence: run not terminated ($dt s)"; }
echo "PASS fencing: a runner whose generation was retired terminates its run within ${dt}s"

# (7) spaces in the output root
d="$S/root with spaces/out dir"; mkdir -p "$d"
CESEL_LOCK_HOLD=0 TAG=sp bash "$SRC" locktest "$d/run a b" > "$S/sp.log" 2>&1 || { cat "$S/sp.log"; fail "spaces: rc"; }
ls "$d" | grep -q '^run a b.RUNLOCK.gone.' && [ ! -e "$d/run a b.RUNLOCK" ] || fail "spaces: lock path mangled"
[ "$(ls "$S/root with spaces" | wc -l)" = 1 ] || fail "spaces: stray paths created"
echo "PASS paths with spaces: one lock path, acquired and released"
rm -rf "$S"
echo "ALL_PASS test_runlock_f1"
