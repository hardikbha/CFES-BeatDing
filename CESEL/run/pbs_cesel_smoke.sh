#!/bin/bash
# CESel GPU smoke (plan step 1, Amendment 1): 20 bf16 steps of Full (batch 176) and B0+CFES-D (176 + 64 views) from
# CFES runs/clip_s7_p1/last.pt (read only) on ONE free allowed GPU of gpu01, then eval/calibrate/posteval/validate on
# 256 items. Scratch output deleted on success. Submit: qsub run/pbs_cesel_smoke.sh
#PBS -N cesel_smoke
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=1:mem=200gb:host=gpu01
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL
cd $A
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 CESEL_WORKERS=3 OMP_NUM_THREADS=4
for t in $(seq 1 30); do g=$(bash run/pick_gpu.sh 0,1,2,3,4,6,7); [ -n "$g" ] && break; sleep 60; done
[ -n "$g" ] || { echo "no free allowed GPU in 30 min"; exit 9; }
echo "$(date -Is) using GPU $g"
sd=$A/work/gpu_smoke_${PBS_JOBID:-x}
CUDA_VISIBLE_DEVICES=$g bash run/cesel_run.sh smoke $sd 2>&1 | grep -v "^ep 0 it"
rc=${PIPESTATUS[0]}
[ $rc = 0 ] && rm -rf $sd
echo "$(date -Is) smoke exit $rc"
exit $rc
