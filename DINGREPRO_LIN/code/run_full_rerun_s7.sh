#!/bin/bash
# Review 4 item 1: faithful from-scratch 50-epoch rerun of Ding's full method (released GDA.py cannot resume).
# Unmodified released training step; our only change vs the repo is reading SFD's own channel output (documented).
CUDA_VISIBLE_DEVICES=6 bash /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/code/run_task_full.sh 7
rm -rf /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES/work/gpulocks/gpu02_6
