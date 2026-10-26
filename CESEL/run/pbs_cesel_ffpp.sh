#!/bin/bash
# CESel seed-7 FF++ screen, runner 1 of 2 (gpu01; decisions/2026-10-25-cesel-plan.md + Amendments 1-2).
#   qsub run/pbs_cesel_ffpp.sh            (runner 2 on gpu02: run/pbs_cesel_ffpp_gpu02.sh; both share one task list)
# Driver: scripts/cesel_pipeline.py (docstring). Logs: logs/cesel_pipeline_gpu01.log, logs/cesel_runs/<task>.log;
# report: results/cesel_ffpp_screen.md. Resumable: resubmit. Stop new launches: touch work/STOP_CESEL.
#PBS -N cesel_ffpp
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=4:mem=200gb:host=gpu01
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/logs/
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL
cd $A
echo "$(date -Is) PBS ${PBS_JOBID:-none} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
export HF_HUB_OFFLINE=1 ALLOWED_GPUS=0,1,2,3,4,6,7 MAX_CONC=4 CESEL_WORKERS=2
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u $A/scripts/cesel_pipeline.py
rc=$?
echo "$(date -Is) cesel_pipeline exit $rc"
exit $rc
