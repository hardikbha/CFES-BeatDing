import json, sys, numpy as np
sys.path.insert(0, "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/code")
from gap_analysis_lib import *
R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
def val_of(name):
    if name.startswith("Ding"): return ld(f"{DS}/full_s7_ep49_ffval.npz")
    tag = {"CFES-D": "clip_D_w0.3_s", "armC": "clip_C_w0.3_s"}
    if name.startswith("armE"):
        w, s = name.split()[1][1:], name.split()[2][1:]; return ld(f"{RUNS}/clip_E_w{w}_s{s}/eval/val_scores.npz")
    a, s = name.split(); return ld(f"{RUNS}/{tag[a]}{s[1:]}/eval/val_scores.npz")
M = models()
zv, yv, gv = val_of("Ding repro s7")
ding_val_tpr = (zv[yv == 1] >= 0).mean(); ding_val_fpr = (zv[yv == 0] >= 0).mean()
print(f"Ding repro at 0.5 on FF++ val: TPR {100*ding_val_tpr:.2f} FPR {100*ding_val_fpr:.2f}")
out = {}
for rule in ("matchTPR", "matchFPR"):
    print(f"\n######## rule {rule}")
    for name, (get, _) in M.items():
        zv, yv, gv = val_of(name)
        t = np.quantile(zv[yv == 1], 1 - ding_val_tpr) if rule == "matchTPR" else np.quantile(zv[yv == 0], 1 - ding_val_fpr)
        tot = 0; line = []
        for k, (_, sch) in SETS.items():
            z, y, g = get(k); d = cols(y, z, g, sch, t); d.pop("grp")
            w = [int((d[c] > DING[k][i]) if UP[i] else (d[c] < DING[k][i])) for i, c in enumerate(K)]
            tot += sum(w); out.setdefault(rule, {}).setdefault(name, {})[k] = {"cols": {c: round(float(d[c]), 2) for c in K + ["TPR", "FPR"]}, "wins": w}
            line.append(f"{k[:5]} {sum(w)}/10 (TPR {d['TPR']:.0f} FPR {d['FPR']:.0f} iF {d['iF_FPR']:.1f} iDP {d['iF_DP']:.1f})")
        print(f"{name:14s} TOTAL {tot:2d}/40 | " + " | ".join(line))
json.dump(out, open(R + "/agents/BEAT_DING/results/matched_op.json", "w"), indent=1)
