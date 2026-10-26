#!/bin/bash
# DINGREPRO_LIN: DING_full = SFD.py (compute Fk on DING_base's trained checkpoint, prc=2%, their reported best)
# + GDA.py (Sinkhorn-OT alignment fine-tune) on top of the SAME seed's DING_base checkpoint.
# Requires DING_base_s<seed>/49.pth to already exist (run_task.sh must have completed first).
# usage: bash run_task_full.sh SEED
set -uo pipefail
SEED=${1:?SEED}
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN
PY=/home/sachin.chaudhary/.conda/envs/qwen38/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
BASE=$A/results/DING_base_s${SEED}
OUT=$A/results/DING_full_s${SEED}; mkdir -p $OUT $A/logs/runs
[ -f $OUT/DONE ] && { echo "already done: $OUT"; exit 0; }
exec > >(tee -a $A/logs/runs/DING_full_s${SEED}.log) 2>&1
echo "start $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED"
[ -f $BASE/DONE ] || { echo "DING_base_s${SEED} not DONE yet - refusing (dependency)"; exit 4; }
[ -f $A/PREREG.sha256 ] && (cd $A && sha256sum -c --quiet PREREG.sha256) || { echo "PREREG missing/changed - refusing to run"; exit 5; }
cd $A/code/Ding_mod
run() { echo "+ $*"; "$@"; local r=$?; [ $r -ne 0 ] && { echo "FAILED rc=$r: $*"; exit $r; }; return $r; }

# Stage 1: SFD -- compute per-group SNNL Fk on the base checkpoint, write channels to decouple (prc=2%, best per ablation Fig 5).
run $PY -u SFD.py --model_name XceptionDetector --checkpoint $BASE/49.pth --dataset_dir $A/data/ff++_ding \
    --prc 2.0 --out_channels_file $OUT/SFD_channels.txt --batch_size 64 --workers 8
[ -f $OUT/SFD_channels.txt ] && date -Is > $OUT/SFD_DONE

# Stage 2: GDA -- Sinkhorn-OT distribution-alignment fine-tune, decoupling the SFD channels, starting from base ckpt.
run $PY -u GDA.py --model_name XceptionDetector --checkpoints $BASE/49.pth --dataset_dir $A/data/ff++_ding \
    --channels_file $OUT/SFD_channels.txt --num_epochs 50 --lr 0.001 --batch_size 32 --workers 8 \
    --sinkhorn_blur 5e-4 --lambda_fairloss 0.005 --result_dir $OUT --training_title ding_full_s${SEED}
FINAL=$OUT/XceptionDetector_prune2%_only_head/lr_1.0e-03_lambda_5.0e-03/49.pth
[ -f "$FINAL" ] && { echo "$FINAL" > $OUT/FINAL_CKPT; date -Is > $OUT/DONE; }
echo "end $(date -Is)"
