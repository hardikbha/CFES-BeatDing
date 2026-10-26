#!/bin/bash
[ -e /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES/work/STOP_FF2ID ] && [ "$1" != "7" ] && { echo "Ding FF2 seed $1 stopped by user (STOP_FF2ID)"; exit 0; }
# Pre-registration decisions/2026-10-22-1920-fairness-and-ff2-plan.md, section D: Ding et al. OFFICIAL pipeline
# (code/Ding_mod, same patched copy as DING_base/DING_full, see VERIFICATION.md) trained on FairForge-2 TRAIN.
#   train_ori_model.py (E_BASE epochs, seed SEED) -> SFD.py (prc 2%, on the base's final epoch)
#   -> GDA.py (E_GDA epochs; GDA itself always uses Ding's hard-coded seed 121, as released).
# Budget: one epoch of either stage = 2 x 469,043 fakes = 938,086 trained images (each fake paired with a random
# real). E_BASE + E_GDA = 5 + 5 = 9,380,860 images <= the CFES FF2 budget 10,151,040 (Ding's own 50+50 default =
# 93.8M is larger, so the budget binds; Ding's 1:1 stage ratio kept, integer epochs). SFD's single forward pass over
# train.csv (502,035 images) has no gradient and is not counted.
# Selection: Ding's code has no validation-based selection; its chain uses the final epoch of each stage
# (GDA default --checkpoints .../49.pth), so the model is the final GDA epoch. The FF2 select split is scored for the
# validation-chosen threshold (rule A1); per-epoch GDA select scores are kept for a declared sensitivity check only.
# usage: bash run_task_ff2.sh SEED     (smoke overrides via env: FF2_DATA OUT E_BASE E_GDA SCORE_MAX WORKERS)
set -uo pipefail
SEED=${1:?SEED}
D=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN
PY=/home/sachin.chaudhary/.conda/envs/qwen38/bin/python
FF2_DATA=${FF2_DATA:-/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2_full}
OUT=${OUT:-$D/results/DING_ff2_s${SEED}}
E_BASE=${E_BASE:-5}; E_GDA=${E_GDA:-5}; WORKERS=${WORKERS:-8}
SMAX=${SCORE_MAX:+--max_items $SCORE_MAX}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p $OUT
[ -f $OUT/DONE ] && { echo "already done: $OUT"; exit 0; }
exec > >(tee -a $OUT/run.log) 2>&1
echo "start $(date -Is) host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED E_BASE=$E_BASE E_GDA=$E_GDA data=$FF2_DATA"
echo "script_sha256 $(sha256sum $0 | cut -c1-16) prereg_sha256 $(sha256sum /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/decisions/2026-10-22-1920-fairness-and-ff2-plan.md | cut -c1-16)"
# GPU guard: a runner task once got an empty CUDA_VISIBLE_DEVICES and trained 17 h on CPU.
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "REFUSE: CUDA_VISIBLE_DEVICES empty"; exit 3; }
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "REFUSE: no CUDA device visible (MPS env on gpu02?)"; exit 3; }
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
cd $D/code/Ding_mod
run() { echo "+ $*"; "$@"; local r=$?; [ $r -ne 0 ] && { echo "FAILED rc=$r: $*"; exit $r; }; return $r; }
LB=$((E_BASE-1)); LG=$((E_GDA-1))

# Stage 1: base CE training (official script is not resumable: an incomplete stage restarts from epoch 0).
if [ ! -f $OUT/base/BASE_DONE ]; then
  run $PY -u train_ori_model.py --seed $SEED --num_epochs $E_BASE --lr 0.001 --result_dir $OUT/base \
      --dataset_dir $FF2_DATA --batch_size 64 --workers $WORKERS --model_name XceptionDetector
  [ -f $OUT/base/$LB.pth ] && date -Is > $OUT/base/BASE_DONE
fi
[ -f $OUT/base/BASE_DONE ] || { echo "base stage incomplete"; exit 6; }

# Stage 2: SFD channels (prc 2% = 10 of 512) from the base's final epoch.
[ -s $OUT/SFD_channels.txt ] || run $PY -u SFD.py --model_name XceptionDetector --checkpoint $OUT/base/$LB.pth \
    --dataset_dir $FF2_DATA --prc 2.0 --out_channels_file $OUT/SFD_channels.txt --batch_size 64 --workers $WORKERS

# Stage 3: GDA (not resumable: restarts if its final epoch is missing). --gpu equals the runner's GPU, because
# GDA.py sets CUDA_VISIBLE_DEVICES=str(--gpu) (default 0) before touching CUDA.
GD=$OUT/XceptionDetector_prune2%_only_head/lr_1.0e-03_lambda_5.0e-03
FINAL=$GD/$LG.pth
[ -f $FINAL ] || run $PY -u GDA.py --model_name XceptionDetector --checkpoints $OUT/base/$LB.pth --dataset_dir $FF2_DATA \
    --channels_file $OUT/SFD_channels.txt --num_epochs $E_GDA --lr 0.001 --batch_size 32 --workers $WORKERS \
    --sinkhorn_blur 5e-4 --lambda_fairloss 0.005 --result_dir $OUT --training_title ding_ff2_s${SEED} \
    --gpu ${CUDA_VISIBLE_DEVICES%%,*}
[ -f $FINAL ] || { echo "GDA final epoch missing"; exit 7; }
echo "$FINAL" > $OUT/FINAL_CKPT; date -Is > $OUT/TRAINED

# Scoring: final model (FF2 TEST + select), base final epoch (= Ding's "Ori" stage), every GDA epoch on select only.
run $PY -u ding_score_ff2.py --ckpt $FINAL --out_dir $OUT/scores --sets ff2_test,ff2_select --data_dir $FF2_DATA $SMAX
run $PY -u ding_score_ff2.py --ckpt $OUT/base/$LB.pth --out_dir $OUT/scores --prefix base_ --sets ff2_test,ff2_select --data_dir $FF2_DATA $SMAX
for e in $(seq 0 $((LG-1))); do
  run $PY -u ding_score_ff2.py --ckpt $GD/$e.pth --out_dir $OUT/scores --prefix gda_ep${e}_ --sets ff2_select --data_dir $FF2_DATA $SMAX
done
date -Is > $OUT/DONE
echo "end $(date -Is)"
