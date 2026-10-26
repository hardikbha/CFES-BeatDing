#!/bin/bash
#PBS -q workq
#PBS -l select=1:ncpus=24:mem=64gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/logs/
D=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN
cd $D/code/Ding_mod; export OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=""
for s in 7 17 27; do /home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u eval_ding_protocol.py --ckpt $D/results/DING_base_s$s/49.pth --tag base_s${s}_ep49 --datasets ff2 --outdir $D/results/eval_ding_protocol --workers 8; done
echo ALLDONE
