#!/bin/bash
# CESel seed-7 FF++ screen, Amendment 5 design ("arms trained like B0"), runner 1 of 2 (gpu01).
#   qsub run/pbs_cesel_p1_gpu01.sh        (runner 2: run/pbs_cesel_p1_gpu02.sh; both share one task list via RUNLOCKs)
# PBS asks for 1 GPU only (fast start); the driver still picks free ALLOWED GPUs by the free rule (>= 60 GB free,
# < 10 % util), <= 4 concurrent runs. Retry: <= 4 attempts per run, randomised 10-20 min wait, resume from last.pt.
# STOP grace 2 h: STOP_CESEL found within 2 h of start = wait for its removal (later: no new launches, then exit).
# Logs: logs/cesel_pipeline_gpu01.log, logs/cesel_runs/p1_<arm>.log; report results/cesel_ffpp_screen_p1.md.
#PBS -N cesel_p1
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=1:mem=200gb:host=gpu01
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL
cd $A
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 ALLOWED_GPUS=0,1,2,3,4,6,7 MAX_CONC=4 CESEL_WORKERS=2 CESEL_DESIGN=p1 MAX_TRIES=4 \
       RETRY_WAIT_MIN=600 RETRY_WAIT_MAX=1200 CESEL_STOP_GRACE_S=7200
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u $A/scripts/cesel_pipeline.py
rc=$?
echo "$(date -Is) cesel_pipeline exit $rc"
exit $rc
