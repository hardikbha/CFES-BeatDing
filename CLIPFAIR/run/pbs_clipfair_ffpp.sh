#!/bin/bash
# CLIPFAIR FF++ baselines, runner 1 of 2 (gpu01; decisions/2026-10-25-cesel-plan.md, Amendment 3): 4 baselines x seeds
# 7/17/27. Queued AFTER the CESel FF++ gpu01 runner: qsub -W depend=afterany:44067 run/pbs_clipfair_ffpp.sh
# (runner 2 on gpu02: run/pbs_clipfair_ffpp_gpu02.sh, depend=afterany:44068; both share one task list).
# Driver: scripts/clipfair_pipeline.py. Logs: logs/clipfair_pipeline_gpu01.log, logs/clipfair_runs/<task>.log;
# report: results/clipfair_ffpp.md. Resumable: resubmit. Stop new launches: touch work/STOP_CLIPFAIR.
#PBS -N clipfair_ffpp
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=4:mem=200gb:host=gpu01
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR
cd $A
[ -e $A/work/STOP_CLIPFAIR ] && { echo "STOP_CLIPFAIR present: exit"; exit 0; }
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 ALLOWED_GPUS=0,1,2,3,4,6,7 MAX_CONC=4 CLIPFAIR_WORKERS=2
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u $A/scripts/clipfair_pipeline.py
rc=$?
echo "$(date -Is) clipfair_pipeline exit $rc"
exit $rc
