#!/bin/bash
# CESEL_MA priority runner 1 of 2 (gpu01, GPUs {0,1,2,3,4,6,7}, <= 4 concurrent). Shares run/tasks.yaml (+ includes, e.g.
# BEAT_DING bd_tasks.json) with run/pbs_priority_gpu02.sh through RUNLOCKs. Submitted with -W depend=afterany:44105
# (the CESel screen runner on gpu01). HOLD: no launch until BOTH screen jobs have ended (their PBS output files exist).
# Logs: logs/priority_gpu01.log, logs/ma_runs/<task>.log; status results/priority_status_gpu01.json.
# Stop new launches: touch work/STOP_MA (2 h start grace: STOP_MA found in the first 2 h = wait for its removal).
#PBS -N ma_prio1
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=4:mem=200gb:host=gpu01
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA
cd $A
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 ALLOWED_GPUS=0,1,2,3,4,6,7 MAX_CONC=4 MA_WORKERS=2 MAX_TRIES=4 RETRY_WAIT_MIN=600 \
       RETRY_WAIT_MAX=1200 MA_STOP_GRACE_S=7200 HOLD_UNTIL_FILES=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/logs/44105.mgmt01.OU:/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/logs/44104.mgmt01.OU
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u $A/scripts/priority_runner.py
rc=$?
echo "$(date -Is) priority_runner exit $rc"
exit $rc
