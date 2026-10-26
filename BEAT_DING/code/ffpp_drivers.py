"""Which FF++ test groups/videos drive i-F_FPR at Ding's operating point (pooled FPR matched to Ding repro's 12.5%)."""
import json, sys, numpy as np, re, collections
R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
sys.path.insert(0, R + "/agents/CFES")
from cfes.metrics import SCHEMES
def ld(f):
    z = np.load(f, allow_pickle=True); return z["logit"].astype(float), z["label"].astype(int), z["group"].astype(int), z["img_path"]
DS = R + "/agents/DINGREPRO_LIN/results/audit_scores"; RU = R + "/agents/CFES/runs_rev4"
M = {"Ding repro": f"{DS}/full_s7_ep49_ff++.npz", "armE w3 s7": f"{RU}/clip_E_w3_s7/eval/ffpp_test_scores.npz",
     "CFES-D s27": f"{RU}/clip_D_w0.3_s27/eval/ffpp_test_scores.npz", "CFES-D s7": f"{RU}/clip_D_w0.3_s7/eval/ffpp_test_scores.npz"}
names = SCHEMES["eight"]
def vid(p):
    b = str(p).split("/")[-2] if "/" in str(p) else str(p)
    return b
for target in (0.125, 0.05):
    print(f"\n### pooled real FPR fixed at {100*target:.1f}% (threshold = real-score quantile; diagnostic)")
    for n, f in M.items():
        z, y, g, p = ld(f)
        t = np.quantile(z[y == 0], 1 - target)
        pr = z >= t
        FPR = pr[y == 0].mean(); TPR = pr[y == 1].mean()
        rows = []; F = 0
        for k, nm in enumerate(names):
            m = (g == k)
            if len(np.unique(y[m])) < 2: continue
            r = m & (y == 0); fk = m & (y == 1)
            fg = pr[r].mean(); F += abs(fg - FPR)
            vids = collections.Counter(vid(x) for x in p[r]); fpv = collections.Counter(vid(x) for x in p[r & pr])
            top = sorted(((fpv[v] / vids[v], v, vids[v]) for v in vids), reverse=True)[:2]
            rows.append(f"{nm:15s} reals {r.sum():5d} ({len(vids):2d} vid) FPR {100*fg:5.1f} TPR {100*pr[fk].mean():5.1f} | worst vids " + ", ".join(f"{100*a:.0f}% of {c}" for a, v, c in top))
        print(f"-- {n}: TPR {100*TPR:.1f}  i-F_FPR {100*F:.1f}")
        print("\n".join("   " + x for x in rows))
