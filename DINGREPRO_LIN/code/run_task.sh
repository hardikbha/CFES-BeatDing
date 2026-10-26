#!/bin/bash
# DINGREPRO_LIN: DING_base = train_ori_model.py (official, 50 epochs) on Lin et al.'s real FF++ train/val.
# usage: bash run_task.sh SEED
set -uo pipefail
SEED=${1:?SEED}
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN
PY=/home/sachin.chaudhary/.conda/envs/qwen38/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUT=$A/results/DING_base_s${SEED}; mkdir -p $OUT $A/logs/runs
[ -f $OUT/DONE ] && { echo "already done: $OUT"; exit 0; }
exec > >(tee -a $A/logs/runs/DING_base_s${SEED}.log) 2>&1
echo "start $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
[ -f $A/PREREG.sha256 ] && (cd $A && sha256sum -c --quiet PREREG.sha256) || { echo "PREREG missing/changed - refusing to run"; exit 5; }
cd $A/code/Ding_mod
run() { echo "+ $*"; "$@"; local r=$?; [ $r -ne 0 ] && { echo "FAILED rc=$r: $*"; exit $r; }; return $r; }
run $PY -u train_ori_model.py --seed $SEED --num_epochs 50 --lr 0.001 \
    --result_dir $OUT --dataset_dir $A/data/ff++_ding --batch_size 64 --workers 8 --model_name XceptionDetector
[ -f $OUT/49.pth ] && date -Is > $OUT/DONE
echo "end $(date -Is) rc=$?"
