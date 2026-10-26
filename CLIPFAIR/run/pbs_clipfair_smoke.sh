#!/bin/bash
# CLIPFAIR GPU smoke (Amendment 3): Ding SFD on 20 batches, then 20 bf16 steps of each baseline (DAW-FDD, DAG-FDD, PG-FDD,
# Ding SFD+GDA) at 176 images/step from CFES runs/clip_s7_p1/last.pt (read only) on ONE free allowed GPU of gpu01, then
# evaluate / calibrate / A1 on 256 items per set and the report. Checkpoints deleted on success, logs kept.
# Submit: qsub run/pbs_clipfair_smoke.sh
#PBS -N clipfair_smoke
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=96gb:host=gpu01
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR
cd $A
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 CLIPFAIR_WORKERS=2 OMP_NUM_THREADS=4
for t in $(seq 1 90); do g=$(bash run/pick_gpu.sh 0,1,2,3,4,6,7); [ -n "$g" ] && break; sleep 60; done
[ -n "$g" ] || { echo "no free allowed GPU in 90 min"; exit 9; }
echo "$(date -Is) using GPU $g"
mkdir -p $A/work/gpulocks && mkdir $A/work/gpulocks/$(hostname -s)_$g && echo "clipfair_smoke $(hostname -s) $$ ${PBS_JOBID:-x}" > $A/work/gpulocks/$(hostname -s)_$g/by
trap "rm -rf $A/work/gpulocks/$(hostname -s)_$g" EXIT
sd=$A/work/gpu_smoke_${PBS_JOBID:-x}
CUDA_VISIBLE_DEVICES=$g bash run/clipfair_run.sh smoke $sd 2>&1 | grep -v "^ep 0 it"
rc=${PIPESTATUS[0]}
[ $rc = 0 ] && find $sd -name "*.pt" -delete
echo "$(date -Is) smoke exit $rc"
exit $rc
