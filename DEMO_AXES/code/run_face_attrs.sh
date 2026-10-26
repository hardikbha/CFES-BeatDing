#!/bin/bash
# runs face_attrs.py over every list (16 procs, onnxruntime 1 thread each); outputs labels/raw/face_attrs_<set>.csv.gz
cd /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DEMO_AXES
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for s in ffpp_test dfdc celebdf dfd ffpp_val ffpp_train ff2_trainreal ff2_test; do
  [ -f labels/raw/face_attrs_$s.csv.gz ] && continue
  ~/.conda/envs/aiface/bin/python code/face_attrs.py --list labels/raw/lists/$s.txt --out labels/raw/face_attrs_$s.tmp.csv.gz --procs 16 \
    && mv labels/raw/face_attrs_$s.tmp.csv.gz labels/raw/face_attrs_$s.csv.gz
  date
done
