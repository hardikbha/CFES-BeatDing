#!/bin/bash
#PBS -q workq
#PBS -l select=1:ncpus=32:mem=96gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/logs/pbs/
# CPU-only scoring of Ding checkpoints with the paper's Eq. 8-10 metrics.
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN
PY=/home/sachin.chaudhary/.conda/envs/qwen38/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 HF_HUB_OFFLINE=1
cd $A/code/Ding_mod
OUT=$A/results/eval_ding_protocol
mkdir -p $OUT $A/logs/runs
exec >> $A/logs/runs/eval_cpu.log 2>&1
echo "start $(date -Is) host=$(hostname) jobs=$JOBS"
IFS=';' read -ra ITEMS <<< "$JOBS"
for it in "${ITEMS[@]}"; do
  tag=${it%%=*}; ckpt=${it#*=}
  echo "+ $tag $ckpt $(date -Is)"
  $PY -u eval_ding_protocol.py --ckpt "$ckpt" --tag "$tag" --outdir $OUT --workers 6 --batch_size 128 || echo "FAILED $tag"
done
echo "end $(date -Is)"
