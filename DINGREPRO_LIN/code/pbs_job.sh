#!/bin/bash
# PBS job: ONE DINGREPRO_LIN DING_base task on a self-detected free, ALLOWED GPU.
# Allowed only: gpu01:{0,1,3,4} or gpu02:{6,7}. Never gpu02:4/5. Never touch other users' / other projects'
# processes (factmux_/e11_grpo/FULL_ZX/F2_BX50/xgendet/D1_CTRLX/faceq/scf).
# Submit: qsub -q gpu -N ding_base_s<seed> -v SEED=<7|17|27> pbs_job.sh
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/logs/pbs/

A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN
mkdir -p $A/logs/pbs $A/work/gpulocks
host=$(hostname -s)
echo "$(date) PBS job ${PBS_JOBID:-} SEED=$SEED host=$host CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

allowed_idx(){
  local h=$1 idx=$2
  if [ "$h" = gpu01 ]; then
    case $idx in 0|1|3|4) return 0;; *) return 1;; esac
  elif [ "$h" = gpu02 ]; then
    case $idx in 6|7) return 0;; *) return 1;; esac
  fi
  return 1
}

pick_and_lock_gpu(){
  local candidates
  for attempt in $(seq 1 180); do   # wait up to ~2h for an allowed GPU with >=45GB free
    candidates=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
                 | awk -F', *' '$2>=45000{print $3, -$2, $1}' | sort -n -k1,1 -k2,2 | awk '{print $3}')
    for idx in $candidates; do
      allowed_idx "$host" "$idx" || continue
      lockdir="$A/work/gpulocks/${host}_${idx}"
      if mkdir "$lockdir" 2>/dev/null; then
        echo "ding_base_s${SEED} ${PBS_JOBID:-}" > "$lockdir/by" 2>/dev/null
        echo "$idx"; return 0
      fi
    done
    sleep "$(( (RANDOM % 30) + 30 ))"
  done
  return 1
}
gpu=$(pick_and_lock_gpu) || gpu=""
trap '[ -n "$gpu" ] && rm -rf "$A/work/gpulocks/${host}_${gpu}" 2>/dev/null' EXIT
if [ -z "$gpu" ]; then
  echo "GIVE_UP: could not lock any allowed (>=45GB free) GPU on $host"; exit 3
fi
if ! allowed_idx "$host" "$gpu"; then
  echo "REFUSE: landed on non-allowed $host:$gpu, exiting without running"; exit 0
fi
export CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "running SEED=$SEED on self-detected+locked gpu $host:$gpu"
bash $A/code/run_task.sh "$SEED"
rc=$?
echo "run_task.sh exited rc=$rc"
exit $rc
