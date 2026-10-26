#!/bin/bash
# CLIPFAIR FF++ baselines, runner 2 of 2 (gpu02, GPUs 6 and 7 ONLY). Shares the task list with run/pbs_clipfair_ffpp.sh
# through the RUNLOCK dirs. Queued after the CESel gpu02 runner: qsub -W depend=afterany:44068 run/pbs_clipfair_ffpp_gpu02.sh
# gpu02 runs MPS: every CUDA process here gets CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY = /tmp/sc_nomps.
#PBS -N clipfair_ffpp2
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=2:mem=200gb:host=gpu02
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR
cd $A
[ -e $A/work/STOP_CLIPFAIR ] && { echo "STOP_CLIPFAIR present: exit"; exit 0; }
export CUDA_MPS_PIPE_DIRECTORY=/tmp/sc_nomps CUDA_MPS_LOG_DIRECTORY=/tmp/sc_nomps
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 ALLOWED_GPUS=6,7 MAX_CONC=2 CLIPFAIR_WORKERS=2
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u $A/scripts/clipfair_pipeline.py
rc=$?
echo "$(date -Is) clipfair_pipeline exit $rc"
exit $rc
