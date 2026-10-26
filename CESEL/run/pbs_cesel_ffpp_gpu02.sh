#!/bin/bash
# CESel seed-7 FF++ screen, runner 2 of 2 (gpu02, GPUs 6 and 7 ONLY; never gpu02:4/5). Shares the task list with
# run/pbs_cesel_ffpp.sh through the RUNLOCK dirs; mainly useful in the arm phase. gpu02 runs MPS: every CUDA process
# here gets CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY = /tmp/sc_nomps (inherited by the driver's children).
#PBS -N cesel_ffpp2
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=2:mem=200gb:host=gpu02
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL
cd $A
export CUDA_MPS_PIPE_DIRECTORY=/tmp/sc_nomps CUDA_MPS_LOG_DIRECTORY=/tmp/sc_nomps
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 ALLOWED_GPUS=6,7 MAX_CONC=2 CESEL_WORKERS=2
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u $A/scripts/cesel_pipeline.py
rc=$?
echo "$(date -Is) cesel_pipeline exit $rc"
exit $rc
