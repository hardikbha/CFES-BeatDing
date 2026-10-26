#!/bin/bash
# FAIRREL_REPRO: one task = train 10 epochs (official train.py from git HEAD, unmodified) + ONE pre-registered scoring
# of best.pt (official test.py AND score_eval.py, asserted equal) + FF++-val features for the probe.
#   bash run_task.sh ARM SEED     ARM in {FAIRREL, CTRL, GRADLOG}
#   FAIRREL = code/train_official_HEAD.py (= git show HEAD:train.py) + official configs/fairrel_clip_vitl14.yaml
#   CTRL    = same script + code/config_ctrl.yaml (only change: lambda_gcwd = lambda_slma = 0)
#   GRADLOG = code/train_gradlog.py (instrumented copy) + official config; diagnostic only: NO test scoring, then
#             frozen-CLIP (--init) FF++-val features.
# GPU = CUDA_VISIBLE_DEVICES exported by pbs_job.sh (this script never picks a GPU).
set -uo pipefail
ARM=${1:?ARM}; SEED=${2:?SEED}
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO
FR=/home/sachin.chaudhary/ICLR_2027/related_work/FairReL
PY=/home/sachin.chaudhary/.conda/envs/pulid/bin/python
D=$A/data; NW=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$FR
OUT=$A/runs/${ARM}_s${SEED}; mkdir -p $OUT $A/logs/runs
[ -f $OUT/DONE ] && { echo "already done: $OUT"; exit 0; }
exec > >(tee -a $A/logs/runs/${ARM}_s${SEED}.log) 2>&1
S=$(date +%s); SD=$(date -Is)
echo "start $SD host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} arm=$ARM seed=$SEED"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
(cd $D && sha256sum -c --quiet DATA_SHA256.txt) || { echo "DATA HASH MISMATCH - refusing to run"; exit 3; }
(cd $A/code && sha256sum -c --quiet CODE_SHA256.txt) || { echo "CODE HASH MISMATCH - refusing to run"; exit 3; }
OFFCFG=$FR/configs/fairrel_clip_vitl14.yaml
case $ARM in
  FAIRREL) TRAIN=$A/code/train_official_HEAD.py; CFG=$OFFCFG ;;
  CTRL)    TRAIN=$A/code/train_official_HEAD.py; CFG=$A/code/config_ctrl.yaml ;;
  GRADLOG) TRAIN=$A/code/train_gradlog.py;       CFG=$OFFCFG ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
RC=0
run() { echo "+ $*"; "$@"; local r=$?; [ $r -ne 0 ] && { echo "FAILED rc=$r: $*"; RC=$r; }; return $r; }
if [ ! -f $OUT/TRAINED ]; then
  run bash -c "cd $FR && $PY -u $TRAIN --config $CFG --train-csv $D/ffpp_train.csv --val-csv $D/ffpp_val.csv \
      --data-root '' --output-dir $OUT --seed $SEED" && [ -f $OUT/best.pt ] && date -Is > $OUT/TRAINED
fi
[ -f $OUT/TRAINED ] || { [ $RC -eq 0 ] && RC=4; }
TESTS="ffpp=$D/ffpp_test.csv celebdf=$D/celebdf_test.csv dfd=$D/dfd_test.csv dfdc=$D/dfdc_test.csv"
if [ $RC -eq 0 ] && [ $ARM != GRADLOG ]; then
  # PRE-REGISTERED single TEST read of best.pt (official selection rule). Results are not inspected until all
  # 6 runs are DONE (PREREG.txt).
  [ -f $A/PREREG.sha256 ] && (cd $A && sha256sum -c --quiet PREREG.sha256) || { echo "PREREG missing/changed - refusing TEST"; exit 5; }
  run bash -c "cd $FR && $PY -u test.py --config $CFG --checkpoint $OUT/best.pt --val-csv $D/ffpp_val.csv \
      --data-root '' --test $TESTS --num-workers $NW --output $OUT/test_best_official.json"
  [ $RC -eq 0 ] && run $PY -u $A/code/score_eval.py --config $CFG --checkpoint $OUT/best.pt --val-csv $D/ffpp_val.csv \
      --test $TESTS --num-workers $NW --output $OUT/test_best.json --save-scores $OUT/scores_best.npz \
      --save-feats $OUT/valfeats_best.npz
  [ $RC -eq 0 ] && run $PY -c "
import json, math
a = json.load(open('$OUT/test_best_official.json')); b = json.load(open('$OUT/test_best.json'))
assert abs(a['threshold'] - b['threshold']) < 1e-6, (a['threshold'], b['threshold'])
for n in ('ffpp', 'celebdf', 'dfd', 'dfdc'):
    for k in ('auc', 'auc_avg', 'auc_gap', 'f_fpr', 'f_meo'):
        x, y = a[n][k], b[n][k]
        assert (math.isnan(x) and math.isnan(y)) or abs(x - y) < 1e-3, (n, k, x, y)
print('official test.py == score_eval.py on best.pt: OK')"
fi
if [ $RC -eq 0 ] && [ $ARM = GRADLOG ]; then
  run $PY -u $A/code/score_eval.py --config $CFG --init --val-csv $D/ffpp_val.csv --num-workers $NW \
      --save-feats $A/runs/valfeats_frozen_clip.npz
  rm -f $OUT/best.pt $OUT/last.pt   # diagnostic run: checkpoints not needed
fi
E=$(date +%s); ED=$(date -Is)
GH=$(awk -v s=$((E-S)) 'BEGIN{printf "%.3f", s/3600}')
[ -f $A/logs/GPU_USE.csv ] || echo "job,host,cuda_visible_devices,start,end,seconds,gpu_hours,rc" > $A/logs/GPU_USE.csv
echo "${ARM}_s${SEED},$(hostname),${CUDA_VISIBLE_DEVICES:-unset},$SD,$ED,$((E-S)),$GH,$RC" >> $A/logs/GPU_USE.csv
[ -f $OUT/best.pt ] && sha256sum $OUT/best.pt | cut -c1-16 > $OUT/best_pt.sha256_16
[ $RC -eq 0 ] && [ -f $OUT/last.pt ] && { sha256sum $OUT/last.pt | cut -c1-16 > $OUT/last_pt.sha256_16; rm -f $OUT/last.pt; }
[ $RC -eq 0 ] && echo "rc=0 $ED" > $OUT/DONE
echo "TASK_DONE ${ARM}_s${SEED} rc=$RC $ED"
exit $RC
