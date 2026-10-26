"""Summarise runs/GRADLOG_s7/gradlog.jsonl (PREREG section 7) and compare GRADLOG_s7 history with FAIRREL_s7.
INACTIVE rule: median over logged steps of ||grad(lambda_k L_k)||_bb / ||grad(L_CE)||_bb < 0.01."""
import json
import sys

import numpy as np
import pandas as pd

A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO"
run = sys.argv[1] if len(sys.argv) > 1 else "GRADLOG_s7"
rows = [json.loads(l) for l in open(f"{A}/runs/{run}/gradlog.jsonl")]
df = pd.DataFrame(rows)
L = [f"gradient-norm diagnostic: {run}, {len(df)} logged steps (every 25), steps {df.step.min()}-{df.step.max()}", ""]
for k in ["orth", "gcwd", "slma"]:
    df[f"ratio_{k}"] = df[f"{k}_gnorm_backbone"] / df["ce_gnorm_backbone"]
L.append("term   value(median)   ||grad(lam*L)||_bb median [p10,p90]   ratio to CE median [p10,p90]   cos(.,CE) median   verdict")
for k in ["ce", "orth", "gcwd", "slma"]:
    v = df[f"{k}_value"]; gn = df[f"{k}_gnorm_backbone"]
    s = f"{k:5s}  {v.median():.3e}   {gn.median():.3e} [{gn.quantile(.1):.3e},{gn.quantile(.9):.3e}]"
    if k != "ce":
        r = df[f"ratio_{k}"]; c = df.get(f"cos_{k}_ce_backbone")
        s += (f"   {r.median():.4f} [{r.quantile(.1):.4f},{r.quantile(.9):.4f}]   {c.median():+.3f}   "
              + ("INACTIVE" if r.median() < 0.01 else "active"))
    L.append(s)
L.append("")
L.append("per epoch medians: ce_gn_bb | ce_gn_cls | ratio orth | ratio gcwd | ratio slma | gcwd value | slma value")
for e, s in df.groupby("epoch"):
    L.append(f"  ep{e:2d} {s.ce_gnorm_backbone.median():.3e} {s.ce_gnorm_cls.median():.3e} "
             f"{s.ratio_orth.median():.4f} {s.ratio_gcwd.median():.5f} {s.ratio_slma.median():.4f} "
             f"{s.gcwd_value.median():.3e} {s.slma_value.median():.3e}")
if "gcwd_D" in df:
    D = np.array([d for d in df.gcwd_D if isinstance(d, list)])
    W = np.array([d for d in df.gcwd_w if isinstance(d, list)])
    L.append("")
    L.append("GCWD per-band D_l median: " + " ".join(f"{x:.2e}" for x in np.median(D, 0)))
    L.append("GCWD per-band softmax w_l median: " + " ".join(f"{x:.3f}" for x in np.median(W, 0)))
try:
    h1 = json.load(open(f"{A}/runs/{run}/history.json")); h2 = json.load(open(f"{A}/runs/FAIRREL_s7/history.json"))
    L.append("")
    L.append("determinism check GRADLOG_s7 vs FAIRREL_s7 (val AUC / val F_FPR / train ce per epoch):")
    for a, b in zip(h1, h2):
        L.append(f"  ep{a['epoch']:2d} {a['val']['auc']:.3f}/{b['val']['auc']:.3f}  {a['val']['f_fpr']:.2f}/{b['val']['f_fpr']:.2f}"
                 f"  {a['ce']:.4f}/{b['ce']:.4f}")
except FileNotFoundError as e:
    L.append(f"(history comparison skipped: {e})")
open(f"{A}/results/GRADLOG_SUMMARY.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
