import json, sys, numpy as np
sys.path.insert(0, "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/code")
from gap_analysis_lib import *
C = R + "/agents/CLIPFAIR/runs"
SRC = {"DAW-FDD (CLIP) s7": f"{C}/ffpp_daw_s7/eval", "DAG-FDD (CLIP) s7": f"{C}/ffpp_dag_s7/eval",
       "Ding SFD+GDA (CLIP port) s7": f"{C}/ffpp_ding_s7/eval",
       "CFES-D (ours) s7": f"{RUNS}/clip_D_w0.3_s7/eval", "Arm E w1 (ours) s7": f"{RUNS}/clip_E_w1_s7/eval"}
def get(name, k):
    if name.startswith("Ding repro"): return ld(f"{DS}/full_s7_ep49_{SETS[k][0]}.npz") if k != "val" else ld(f"{DS}/full_s7_ep49_ffval.npz")
    return ld(f"{SRC[name]}/{'val' if k=='val' else k}_scores.npz")
names = list(SRC) + ["Ding repro (Xception) s7"]
zv, yv, _ = get("Ding repro (Xception) s7", "val"); TPRV = (zv[yv == 1] >= 0).mean()
out = {}
for rule in ("matched", "0.5"):
    print(f"\n######## threshold rule: {rule}" + (f" (FF++ val TPR = {100*TPRV:.2f}%, Ding repro at 0.5)" if rule == "matched" else " (logit >= 0, raw model output)"))
    for n in names:
        if rule == "matched":
            zv, yv, _ = get(n, "val"); t = np.quantile(zv[yv == 1], 1 - TPRV)
        else:
            t = 0.0
        tot = 0; line = []
        for k, (_, sch) in SETS.items():
            z, y, g = get(n, k); d = cols(y, z, g, sch, t); d.pop("grp")
            w = [int((d[c] > DING[k][i]) if UP[i] else (d[c] < DING[k][i])) for i, c in enumerate(K)]
            tot += sum(w); out.setdefault(rule, {}).setdefault(n, {})[k] = {"cols": {c: round(float(d[c]), 2) for c in K + ["TPR", "FPR"]}, "wins": w}
            line.append(f"{k[:5]} {sum(w):2d}/10 AUC {d['AUC']:.1f} TPR {d['TPR']:.0f} FPR {d['FPR']:.0f} iF {d['iF_FPR']:.1f} iES {d['iesAUC']:.1f}")
        print(f"{n:28s} {tot:2d}/40 | " + " | ".join(line))
json.dump(out, open(R + "/agents/BEAT_DING/results/baselines_vs_ding.json", "w"), indent=1)
