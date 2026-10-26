#!/bin/bash
# Audit 21 Oct: logits of Ding full (s7, final epoch) and Ding Ori (s7/17/27, epoch 49) on FF++ val + 4 test sets.
cd /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/code/Ding_mod
PY=/home/sachin.chaudhary/.conda/envs/qwen38/bin/python
$PY -u ding_score_logits.py --ckpt "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/DING_full_s7/XceptionDetector_prune2%_only_head/lr_1.0e-03_lambda_5.0e-03/49.pth" --tag full_s7_ep49 --outdir /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/audit_scores
for s in 7 17 27; do $PY -u ding_score_logits.py --ckpt /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/DING_base_s$s/49.pth --tag base_s${s}_ep49 --outdir /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/audit_scores; done
echo AUDIT_SCORES_DONE
