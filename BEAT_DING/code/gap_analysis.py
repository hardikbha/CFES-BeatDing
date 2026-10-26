"""Beat-Ding gap analysis (26 Oct 2026). CPU, read-only on saved per-image logits.
For every model and Ding-protocol test set: Ding's 10 columns at the model's deployed threshold, plus TPR/FPR;
threshold sweeps (i-F_FPR, i-F_DP vs TPR); per-group TPR/FPR drivers. Labels = the Ding/Lin protocol labels
(Xu et al.), i.e. exactly the labels behind Ding's published tables."""
import json, os, sys, numpy as np
R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
sys.path.insert(0, R + "/agents/CFES")
from cfes.metrics import SCHEMES, partitions
from sklearn.metrics import roc_auc_score
SETS = {"ffpp_test": ("ff++", "eight"), "dfdc": ("dfdc", "eight"), "celebdf": ("celebdf", "six"), "dfd": ("dfd", "six")}
DS = R + "/agents/DINGREPRO_LIN/results/audit_scores"
RUNS = R + "/agents/CFES/runs_rev4"

def ld(f):
    z = np.load(f, allow_pickle=True)
    return z["logit"].astype(np.float64), z["label"].astype(int), z["group"].astype(int)

def models():
    M = {}
    for s in (7, 17, 27):
        r = f"{RUNS}/clip_D_w0.3_s{s}"; b = json.load(open(f"{r}/eval/calibration.json"))["intercept_b"]
        M[f"CFES-D s{s}"] = (lambda k, r=r: ld(f"{r}/eval/{k}_scores.npz"), -b)
        r = f"{RUNS}/clip_C_w0.3_s{s}"
        M[f"armC s{s}"] = (lambda k, r=r: ld(f"{r}/eval/{k}_scores.npz"), 0.0)
    for w in (1, 3, 10):
        r = f"{RUNS}/clip_E_w{w}_s7"; b = json.load(open(f"{r}/eval/calibration.json"))["intercept_b"]
        M[f"armE w{w} s7"] = (lambda k, r=r: ld(f"{r}/eval/{k}_scores.npz"), -b)
    M["Ding repro s7"] = (lambda k: ld(f"{DS}/full_s7_ep49_{SETS[k][0]}.npz"), 0.0)
    return M

def cols(y, z, g, scheme, t):
    """Ding's 9 fairness columns at decision z>=t (+ AUC, TPR, FPR), groups with both classes only."""
    p = z >= t; out = {}
    FPR = p[y == 0].mean(); TPR = p[y == 1].mean(); auc = roc_auc_score(y, z)
    for lv, pre in (("gender", "g"), ("race", "r"), ("intersection", "i")):
        st = []
        for ids in partitions(scheme)[lv].values():
            m = np.isin(g, ids)
            if m.sum() == 0 or len(np.unique(y[m])) < 2: continue
            st.append((p[m & (y == 0)].mean(), p[m & (y == 1)].mean(), p[m].mean(), roc_auc_score(y[m], z[m]), (m & (y == 0)).sum(), (m & (y == 1)).sum()))
        st = np.array(st)
        out[pre + "F_FPR"] = 100 * np.abs(st[:, 0] - FPR).sum()
        out[pre + "F_DP"] = 100 * max(st[:, 2].max() - st[:, 2].min(), (1 - st[:, 2]).max() - (1 - st[:, 2]).min())
        out[pre + "esAUC"] = 100 * auc / (1 + np.abs(st[:, 3] - auc).sum())
        if pre == "i":
            out["grp"] = st
    out.update(AUC=100 * auc, TPR=100 * TPR, FPR=100 * FPR)
    return out

DING = {  # published (Tables 1-2): gF gDP gES rF rDP rES iF iDP iES AUC
 "ffpp_test": [0.53, 3.61, 96.45, 9.29, 4.35, 94.86, 20.18, 9.47, 86.91, 97.71],
 "dfdc": [1.76, 3.67, 57.94, 17.93, 7.58, 52.33, 37.37, 11.92, 39.03, 61.86],
 "celebdf": [6.41, 10.81, 69.68, 16.60, 7.29, 69.96, 23.32, 12.01, 63.27, 76.75],
 "dfd": [4.72, 5.49, 78.98, 2.09, 18.66, 77.44, 27.32, 22.81, 69.73, 81.46]}
K = ["gF_FPR", "gF_DP", "gesAUC", "rF_FPR", "rF_DP", "resAUC", "iF_FPR", "iF_DP", "iesAUC", "AUC"]
UP = [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
res = {}
M = models()
for name, (get, t_dep) in M.items():
    res[name] = {}
    for k, (_, sch) in SETS.items():
        z, y, g = get(k)
        d = cols(y, z, g, sch, t_dep)
        wins = [int((d[c] > DING[k][i]) if UP[i] else (d[c] < DING[k][i])) for i, c in enumerate(K)]
        # sweep: thresholds at TPR targets on THIS test set (diagnostic only; not a selection rule)
        zf = np.sort(z[y == 1]); sweep = {}
        for tt in (50, 60, 70, 80, 90, 95):
            thr = zf[int(len(zf) * (1 - tt / 100))]
            e = cols(y, z, g, sch, thr)
            sweep[tt] = {c: round(e[c], 2) for c in ("FPR", "iF_FPR", "iF_DP", "gF_FPR", "rF_FPR", "gF_DP", "rF_DP")}
        grp = d.pop("grp")
        res[name][k] = {"deployed": {c: round(float(v), 2) for c, v in d.items()}, "wins": wins, "nwin": sum(wins), "sweep": sweep,
                        "grp_FPR": [round(100 * x, 1) for x in grp[:, 0]], "grp_TPR": [round(100 * x, 1) for x in grp[:, 1]],
                        "grp_nreal": grp[:, 4].astype(int).tolist(), "grp_nfake": grp[:, 5].astype(int).tolist()}
json.dump(res, open(R + "/agents/BEAT_DING/results/gap_analysis.json", "w"), indent=1)
for k in SETS:
    print(f"\n== {k}  Ding: iF_FPR {DING[k][6]} iF_DP {DING[k][7]} iesAUC {DING[k][8]} AUC {DING[k][9]}")
    for n in res:
        d = res[n][k]["deployed"]
        print(f"{n:14s} wins {res[n][k]['nwin']:2d}/10 | AUC {d['AUC']:.2f} TPR {d['TPR']:.1f} FPR {d['FPR']:.1f} | iF_FPR {d['iF_FPR']:.1f} iF_DP {d['iF_DP']:.1f} iES {d['iesAUC']:.1f} | gF {d['gF_FPR']:.1f}/{d['gF_DP']:.1f} rF {d['rF_FPR']:.1f}/{d['rF_DP']:.1f}")
