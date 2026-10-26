#!/bin/bash
# DINGREPRO_LIN: DING_full "fast-loader" (decisions/2026-10-26-beat-ding-plan.md Amendment 4a, D2 option b).
# Copy of run_task_full.sh: SFD.py (released, unchanged) on DING_base_s<seed>/49.pth, then GDA_fastloader.py = GDA.py
# with ONE disclosed change (persistent group-DataLoader iterators; code/Ding_mod/GDA_fastloader.diff,
# FASTLOADER_NOTE.md), same CLI/hyper-parameters as the s7 run, then the audit scores (FF++ val + 4 test sets) exactly
# like DING_full_s7 (run_audit_scores.sh: ding_score_logits.py on the final-epoch 49.pth).
# Output: results/DING_full_s<seed>_fast/ (SFD_channels.txt, SFD_DONE, checkpoints, FINAL_CKPT, DONE, PROVENANCE.txt)
#         results/audit_scores/full_s<seed>_fast_ep49_{ffval,ff++,dfdc,celebdf,dfd}.npz
# Resumable by stage: SFD skipped once SFD_DONE exists; GDA skipped once DONE exists (the released GDA cannot resume
# mid-run, so an interrupted GDA restarts from epoch 0 and the partial checkpoint dir is moved aside); every npz is
# scored into $OUT/audit_staging, validated against the s7 rows (check_audit_npz.py) and only then moved into place.
# Never exits 4 (ma_run.sh / priority runner read 4 as "held by another runner").
# usage: bash run_task_full_fast.sh SEED
# dry run (no GPU, nothing real touched): DINGFAST_DRY=1 DINGFAST_RESULTS=<scratch> DINGFAST_LOGS=<scratch> bash run_task_full_fast.sh SEED
set -uo pipefail
SEED=${1:?SEED}
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN
PY=/home/sachin.chaudhary/.conda/envs/qwen38/bin/python
CODE=$A/code/Ding_mod
DRY=${DINGFAST_DRY:-0}
RES=${DINGFAST_RESULTS:-$A/results}
LOGD=${DINGFAST_LOGS:-$A/logs/runs}
if [ "$DRY" = 1 ] && { [ "$RES" = "$A/results" ] || [ "$LOGD" = "$A/logs/runs" ]; }; then
  echo "DINGFAST_DRY=1 needs DINGFAST_RESULTS and DINGFAST_LOGS set to scratch dirs"; exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ "$(hostname -s)" = gpu02 ]; then            # MPS off on gpu02 (as run/pbs_priority_gpu02.sh); keep an inherited value
  export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/sc_nomps} CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/sc_nomps}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" 2>/dev/null
fi
BASE=$A/results/DING_base_s${SEED}
TAG=full_s${SEED}_fast_ep49
OUT=$RES/DING_full_s${SEED}_fast
AUD=$RES/audit_scores
STG=$OUT/audit_staging
SETS="ffval ff++ dfdc celebdf dfd"
CKD="$OUT/XceptionDetector_prune2%_only_head/lr_1.0e-03_lambda_5.0e-03"
FINAL="$CKD/49.pth"
mkdir -p "$OUT" "$AUD" "$LOGD" "$OUT/work"
exec > >(tee -a "$LOGD/DING_full_s${SEED}_fast.log") 2>&1
echo "start $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED dry=$DRY MPS_PIPE=${CUDA_MPS_PIPE_DIRECTORY:-unset}"
all_npz(){ local s; for s in $SETS; do [ -f "$AUD/${TAG}_$s.npz" ] || return 1; done; return 0; }
if [ -f "$OUT/DONE" ] && all_npz; then echo "already done: $OUT + $AUD/${TAG}_*.npz"; exit 0; fi
[ -f "$BASE/DONE" ] || { echo "DING_base_s${SEED} not DONE - refusing (dependency)"; exit 6; }
(cd "$A" && sha256sum -c --quiet PREREG.sha256) || { echo "PREREG missing/changed - refusing to run"; exit 5; }
if [ "$DRY" != 1 ]; then
  "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "REFUSE: no CUDA device visible (would train on CPU)"; exit 3; }
fi
run(){ echo "+ $*"; if [ "$DRY" = 1 ]; then return 0; fi; "$@"; local r=$?; [ $r -ne 0 ] && { echo "FAILED rc=$r: $*"; [ $r = 4 ] && r=1; exit $r; }; return 0; }

if [ ! -f "$OUT/PROVENANCE.txt" ]; then
  { echo "DING_full_s${SEED}_fast  created $(date -Is) host=$(hostname -s)"
    echo "Amendment 4a D2 option (b): released SFD.py + GDA_fastloader.py (persistent group iterators only)"
    (cd "$CODE" && sha256sum SFD.py GDA.py GDA_fastloader.py GDA_fastloader.diff)
    sha256sum "$BASE/49.pth" "$A/code/run_task_full_fast.sh"; } > "$OUT/PROVENANCE.txt"
fi

# Stage 1: SFD (released code, unchanged) -- per-group SNNL Fk on the base checkpoint, 2% = 10 channels.
if [ ! -f "$OUT/SFD_DONE" ] || [ ! -s "$OUT/SFD_channels.txt" ]; then
  (cd "$OUT/work" && run "$PY" -u "$CODE/SFD.py" --model_name XceptionDetector --checkpoint "$BASE/49.pth" \
      --dataset_dir "$A/data/ff++_ding" --prc 2.0 --out_channels_file "$OUT/SFD_channels.txt" --batch_size 64 --workers 8) || exit $?
  [ "$DRY" = 1 ] && echo "1 2 3 4 5 6 7 8 9 10" > "$OUT/SFD_channels.txt"     # dry stub
  [ -s "$OUT/SFD_channels.txt" ] && date -Is > "$OUT/SFD_DONE"
fi
[ -f "$OUT/SFD_DONE" ] || { echo "SFD produced no channels"; exit 1; }
echo "SFD channels: $(cat "$OUT/SFD_channels.txt")"

# Stage 2: GDA fast-loader -- identical CLI to the s7 run (run_task_full.sh), from epoch 0.
if [ ! -f "$OUT/DONE" ]; then
  if [ -d "$OUT/XceptionDetector_prune2%_only_head" ]; then
    mv "$OUT/XceptionDetector_prune2%_only_head" "$OUT/partial_$(date +%Y%m%dT%H%M%S)_XceptionDetector_prune2%_only_head"
    echo "moved an interrupted GDA attempt aside (restart from epoch 0)"
  fi
  (cd "$OUT/work" && run "$PY" -u "$CODE/GDA_fastloader.py" --model_name XceptionDetector --checkpoints "$BASE/49.pth" \
      --dataset_dir "$A/data/ff++_ding" --channels_file "$OUT/SFD_channels.txt" --num_epochs 50 --lr 0.001 \
      --batch_size 32 --workers 8 --sinkhorn_blur 5e-4 --lambda_fairloss 0.005 --result_dir "$OUT" \
      --training_title ding_full_s${SEED}_fast) || exit $?
  [ "$DRY" = 1 ] && { mkdir -p "$CKD"; echo dry > "$FINAL"; }          # dry stub
  [ -f "$FINAL" ] || { echo "GDA finished without $FINAL"; exit 1; }
  echo "$FINAL" > "$OUT/FINAL_CKPT"; date -Is > "$OUT/DONE"
fi

# Stage 3: audit scores (as run_audit_scores.sh for DING_full_s7): FF++ val + 4 test sets, final epoch 49.
mkdir -p "$STG"
for s in $SETS; do                                 # drop any half-written staged file of an interrupted attempt
  f="$STG/${TAG}_$s.npz"; [ -f "$f" ] && ! "$PY" "$A/code/check_audit_npz.py" "$f" "$s" && rm -f "$f"
done
todo=""; for s in $SETS; do [ -f "$AUD/${TAG}_$s.npz" ] || [ -f "$STG/${TAG}_$s.npz" ] || todo="$todo,$s"; done
if [ -n "$todo" ]; then
  (cd "$CODE" && run "$PY" -u ding_score_logits.py --ckpt "$FINAL" --tag "$TAG" --outdir "$STG" --sets "${todo#,}") || exit $?
  if [ "$DRY" = 1 ]; then for s in ${todo//,/ }; do cp "$A/results/audit_scores/full_s7_ep49_$s.npz" "$STG/${TAG}_$s.npz"; done; fi
fi
for s in $SETS; do
  f="$STG/${TAG}_$s.npz"; [ -f "$AUD/${TAG}_$s.npz" ] && continue
  "$PY" "$A/code/check_audit_npz.py" "$f" "$s" || { rm -f "$f"; echo "audit npz $s invalid"; exit 1; }
  mv "$f" "$AUD/${TAG}_$s.npz"
done
all_npz || { echo "audit npz missing"; exit 1; }
echo "audit scores: $(ls "$AUD"/${TAG}_*.npz | wc -l) files $AUD/${TAG}_*.npz"
echo "end $(date -Is)"
