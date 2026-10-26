#!/bin/bash
# PBS job: ONE FAIRREL_REPRO task on a self-detected free GPU (this cluster's PBS does not reliably set
# CUDA_VISIBLE_DEVICES; pattern from TRAIN_ON_LIN/code/pbs_train.sh and EXACTMATCH_plan/work/pbs_score.sh).
# Submit: qsub -q gpu -N fr_<arm>_s<seed> -v ARM=<FAIRREL|CTRL|GRADLOG>,SEED=<7|17|27> pbs_job.sh
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO/logs/pbs/

A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO
mkdir -p $A/logs/pbs $A/work/gpulocks
host=$(hostname -s)
echo "$(date) PBS job ${PBS_JOBID:-} ARM=$ARM SEED=$SEED host=$host CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

# FairReL peaks at ~37 GB -> require >= 45 GB free; prefer the least-utilised, then the most-free GPU.
pick_and_lock_gpu(){
  local candidates
  for attempt in $(seq 1 120); do   # wait up to ~1.5 h for a GPU with >= 45 GB free on this node
    candidates=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
                 | awk -F', *' '$2>=45000{print $3, -$2, $1}' | sort -n -k1,1 -k2,2 | awk '{print $3}')
    for idx in $candidates; do
      [ "$host" = gpu02 ] && { [ "$idx" = 4 ] || [ "$idx" = 5 ]; } && continue
      lockdir="$A/work/gpulocks/${host}_${idx}"
      if mkdir "$lockdir" 2>/dev/null; then
        echo "${ARM}_s${SEED} ${PBS_JOBID:-}" > "$lockdir/by" 2>/dev/null
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
  echo "GIVE_UP: could not lock any free (>=45GB), non-forbidden GPU on $host"; exit 3
fi
if [ "$host" = gpu02 ] && { [ "$gpu" = 4 ] || [ "$gpu" = 5 ]; }; then
  echo "REFUSE: landed on forbidden gpu02:$gpu, exiting without running"; exit 0
fi
export CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "running ARM=$ARM SEED=$SEED on self-detected+locked gpu $host:$gpu"
bash $A/code/run_task.sh "$ARM" "$SEED"
rc=$?
echo "run_task.sh exited rc=$rc"
exit $rc
