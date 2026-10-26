#!/bin/bash
# print the first allowed GPU index that is free (>= 60 GB free AND < 10% util) and not locked by a CLIPFAIR or a CESel
# runner (work/gpulocks/<host>_<i> of either; CESel's are only READ); empty if none.   bash run/pick_gpu.sh 0,1,2,3,4,6,7
A=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR
C=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | while IFS=, read i mf ut; do
  i=$(echo $i); mf=$(echo $mf); ut=$(echo $ut)
  case ",$1," in *",$i,"*) ;; *) continue;; esac
  [ "$mf" -ge 61440 ] && [ "$ut" -lt 10 ] && [ ! -e $A/work/gpulocks/$(hostname -s)_$i ] && \
    [ ! -e $C/work/gpulocks/$(hostname -s)_$i ] && { echo $i; break; }
done
