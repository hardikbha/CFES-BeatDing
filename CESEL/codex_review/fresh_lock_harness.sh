take_lock(){   # one runner per output directory, across hosts (shared filesystem)
  local L=$1.RUNLOCK me; me="$(hostname -s) ${PBS_JOBID:-nojob} $$"
  if ! mkdir $L 2>/dev/null; then
    local h p hb age; h=$(awk '{print $1}' $L/owner 2>/dev/null); p=$(awk '{print $3}' $L/owner 2>/dev/null); hb=$L/heartbeat
    age=$(( $(date +%s) - $(stat -c %Y $hb 2>/dev/null || stat -c %Y $L) ))
    if [ $age -gt 900 ]; then
      echo "stale lock on $1 (owner $(cat $L/owner 2>/dev/null), heartbeat ${age}s old): taking over"
    elif [ "$h" = "$(hostname -s)" ] && [ -n "$p" ] && ! kill -0 $p 2>/dev/null; then
      echo "stale lock on $1 (host $h, owner pid $p dead): taking over"
    else echo "skip $1: held by $(cat $L/owner 2>/dev/null) (heartbeat ${age}s)"; exit 4; fi
  fi
  echo "$me" > $L/owner; touch $L/heartbeat
  ( while kill -0 $$ 2>/dev/null; do touch $L/heartbeat 2>/dev/null; sleep 60; done ) &
  HB=$!
  trap "kill $HB 2>/dev/null; rm -rf $L" EXIT
}

take_lock "$1"
