"""Beat-Ding operating points and Ding et al.'s 10 columns (decisions/2026-10-26-beat-ding-plan.md, sections 1, 4, 5).
CPU only, numpy / sklearn on saved per-image logits (eval/<set>_scores.npz, eval/val_scores.npz).

Columns, in Ding et al. CVPR'26 Tables 1-2 order (K): g/r/i F_FPR, F_DP, es-AUC, then AUC; UP marks higher-is-better.
cols() is the column code of code/gap_analysis_lib.py (the 26 Oct diagnosis): decision fake iff logit >= t; for each
level (gender, race, intersection) only groups holding both classes enter; F_FPR = 100 sum_g |FPR_g - FPR|,
F_DP = 100 max(range_g PR_g, range_g (1 - PR_g)), es-AUC = 100 AUC / (1 + sum_g |AUC_g - AUC|); AUCs on float64 logits.

Operating points (one global logit threshold per model, chosen on FF++ VALIDATION only):
  matched : val TPR = TPR of the Ding reproduction (DINGREPRO_LIN full_s7_ep49) on FF++ val at its own 0.5
            (= 0.94177, recomputed from the file), t = np.quantile(val fake logits, 1 - TPR) (code/matched_op.py rule);
  native  : the class-balanced common intercept of scripts/calibrate.py, t = -b;
  A1      : scripts/fair_a1.py (min val intersectional F_FPR s.t. val TPR >= 80 %, FPR <= 20 %).
val Ding-score (plan section 4): on FF++ val at the matched threshold, the number of Ding's 10 columns on which a model
beats the Ding reproduction's FF++ val columns at its 0.5 (strictly: lower F_FPR / F_DP, higher es-AUC / AUC);
ties broken by val AUC.
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import partitions

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
DING_DIR = R + "/agents/DINGREPRO_LIN/results/audit_scores"
DING_VAL_NPZ = DING_DIR + "/full_s7_ep49_ffval.npz"
DING_TEST_NPZ = {"ffpp_test": "ff++", "dfdc": "dfdc", "celebdf": "celebdf", "dfd": "dfd"}   # full_s7_ep49_<x>.npz
SETS = {"ffpp_test": "eight", "dfdc": "eight", "celebdf": "six", "dfd": "six"}
NAMES = {"ffpp_test": "FF++", "dfdc": "DFDC", "celebdf": "Celeb-DF", "dfd": "DFD"}
K = ["gF_FPR", "gF_DP", "gesAUC", "rF_FPR", "rF_DP", "resAUC", "iF_FPR", "iF_DP", "iesAUC", "AUC"]
UP = [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
# Ding et al. published "Ours" rows, Tables 1-2 (Xception, threshold 0.5), in K order (= code/gap_analysis_lib.DING)
DING_PUB = {
    "ffpp_test": [0.53, 3.61, 96.45, 9.29, 4.35, 94.86, 20.18, 9.47, 86.91, 97.71],
    "dfdc": [1.76, 3.67, 57.94, 17.93, 7.58, 52.33, 37.37, 11.92, 39.03, 61.86],
    "celebdf": [6.41, 10.81, 69.68, 16.60, 7.29, 69.96, 23.32, 12.01, 63.27, 76.75],
    "dfd": [4.72, 5.49, 78.98, 2.09, 18.66, 77.44, 27.32, 22.81, 69.73, 81.46]}
# Ding et al. published Table 3 (ResNet-50, "Ours" rows), K order; checked against refs/Ding_CVPR2026.txt (tests)
DING_PUB_R50 = {
    "ffpp_test": [3.94, 5.14, 94.46, 18.67, 2.43, 92.69, 34.30, 8.61, 82.87, 95.53],
    "dfdc": [3.97, 3.61, 56.96, 11.94, 5.15, 50.99, 32.14, 10.27, 37.55, 62.54],
    "celebdf": [4.53, 12.00, 68.98, 6.76, 8.16, 71.03, 5.56, 9.11, 68.25, 77.35],
    "dfd": [3.12, 1.76, 77.68, 6.51, 3.03, 70.93, 20.99, 4.85, 66.19, 78.63]}
TABLES = {"xception": ("Tables 1-2 (Xception)", DING_PUB), "resnet": ("Table 3 (ResNet-50)", DING_PUB_R50)}


def published_for(backend):
    """The published Ding table our model is compared with: Table 3 for the ResNet-50 backend, Tables 1-2 otherwise
    (CLIP ViT-L/14 rows are compared with Ding's Xception rows, plan section 0)."""
    return TABLES["resnet" if backend == "resnet" else "xception"]


def load(f):
    """(logit float64, label, group, img_path or None) from a saved score file."""
    z = np.load(f, allow_pickle=True)
    p = z["img_path"].astype(str) if "img_path" in z.files else None
    return z["logit"].astype(np.float64), z["label"].astype(int), z["group"].astype(int), p


def cols(y, z, g, scheme, t, with_groups=False):
    """Ding's 9 fairness columns at decision z >= t, plus AUC, TPR, FPR (all in %)."""
    y, z, g = np.asarray(y), np.asarray(z, dtype=np.float64), np.asarray(g)
    p = z >= t
    out = {}
    FPR, TPR, auc = p[y == 0].mean(), p[y == 1].mean(), roc_auc_score(y, z)
    for lv, pre in (("gender", "g"), ("race", "r"), ("intersection", "i")):
        st, names = [], []
        for name, ids in partitions(scheme)[lv].items():
            m = np.isin(g, ids)
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                continue
            st.append((p[m & (y == 0)].mean(), p[m & (y == 1)].mean(), p[m].mean(), roc_auc_score(y[m], z[m]),
                       (m & (y == 0)).sum(), (m & (y == 1)).sum()))
            names.append(name)
        st = np.array(st)
        out[pre + "F_FPR"] = 100 * float(np.abs(st[:, 0] - FPR).sum())
        out[pre + "F_DP"] = 100 * float(max(st[:, 2].max() - st[:, 2].min(), (1 - st[:, 2]).max() - (1 - st[:, 2]).min()))
        out[pre + "esAUC"] = 100 * float(auc / (1 + np.abs(st[:, 3] - auc).sum()))
        if with_groups:
            out[pre + "_groups"] = {n: {"FPR": 100 * float(s[0]), "TPR": 100 * float(s[1]), "PR": 100 * float(s[2]),
                                        "AUC": 100 * float(s[3]), "n_real": int(s[4]), "n_fake": int(s[5])}
                                    for n, s in zip(names, st)}
    out.update(AUC=100 * float(auc), TPR=100 * float(TPR), FPR=100 * float(FPR))
    return out


def wins(d, ref):
    """1 where column d beats ref (strict), K order. ref: dict of columns or a K-ordered list."""
    r = [ref[c] for c in K] if isinstance(ref, dict) else list(ref)
    return [int((d[c] > r[i]) if UP[i] else (d[c] < r[i])) for i, c in enumerate(K)]


def ding_val_reference():
    """The Ding reproduction on FF++ val at its own 0.5 (logit 0): TPR / FPR and its 10 columns."""
    z, y, g, _ = load(DING_VAL_NPZ)
    d = cols(y, z, g, "eight", 0.0)
    return {"file": DING_VAL_NPZ, "tpr": float((z[y == 1] >= 0).mean()), "fpr": float((z[y == 0] >= 0).mean()),
            "n_fake": int((y == 1).sum()), "n_real": int((y == 0).sum()), "cols": d}


def matched_threshold(zv, yv, tpr):
    """code/matched_op.py rule: t = quantile of the FF++ val fake logits at 1 - TPR_target."""
    return float(np.quantile(np.asarray(zv, dtype=np.float64)[np.asarray(yv) == 1], 1 - tpr))


def val_ding_score(zv, yv, gv, ref=None):
    """Plan section 4: wins on FF++ val at the matched threshold vs the Ding reproduction's val columns at 0.5."""
    ref = ref or ding_val_reference()
    t = matched_threshold(zv, yv, ref["tpr"])
    d = cols(yv, zv, gv, "eight", t)
    w = wins(d, ref["cols"])
    return {"score": int(sum(w)), "wins": w, "t": t, "cols": d, "val_auc": d["AUC"], "val_TPR": d["TPR"],
            "val_FPR": d["FPR"], "target_tpr": ref["tpr"]}


def rank_key(rec):
    """Arm comparison order (plan section 4): val Ding-score, then val AUC."""
    return (rec["score"], rec["val_auc"])


def a1_threshold(zv, yv, gv):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from fair_a1 import pick_a1, val_curves
    u, fpr, tpr, f = val_curves(np.asarray(zv, dtype=np.float64), np.asarray(yv), np.asarray(gv), "eight")
    j = pick_a1(u, fpr, tpr, f)
    return None if j is None else float(u[j])


def fdp_oracle(y, g, scheme):
    """F_DP of a perfect detector (decision = label) per level: the structural floor of F_DP on that set."""
    y = np.asarray(y)
    d = cols(y, y.astype(np.float64), g, scheme, 0.5)
    return {"gF_DP": d["gF_DP"], "rF_DP": d["rF_DP"], "iF_DP": d["iF_DP"]}


def video_of(path):
    """Lin/Ding frame name <label>_<video>_<frame>.png -> <label>_<video> (code/noise_floor.py)."""
    b = os.path.basename(str(path))
    return "_".join(b.rsplit(".", 1)[0].split("_")[:-1])


def noise_floor(z, y, g, paths, t, n_perm=2000, seed=0, n_groups=8):
    """code/noise_floor.py null, at decision z >= t: reals-only intersectional F_FPR (100 sum_k |FPR_k - FPR| over the
    groups present among the reals) with group labels permuted across REAL VIDEOS (video integrity kept). Returns the
    observed value, the null median and 5-95 % range, and P(null >= observed)."""
    z, y, g = np.asarray(z, dtype=np.float64), np.asarray(y), np.asarray(g)
    r = y == 0
    fl, gr = z[r] >= t, g[r]
    vr = np.array([video_of(p) for p in np.asarray(paths)[r]])
    vids = np.unique(vr)
    vg = {v: gr[vr == v][0] for v in vids}
    vidx = {v: np.where(vr == v)[0] for v in vids}
    F0 = fl.mean()

    def iF(glab):
        return 100 * sum(abs(fl[glab == k].mean() - F0) for k in range(n_groups) if (glab == k).any())
    obs = iF(gr)
    rng = np.random.default_rng(seed)
    labs = np.array([vg[v] for v in vids])
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(labs)
        gl = np.empty(len(gr), int)
        for v, lab in zip(vids, perm):
            gl[vidx[v]] = lab
        null[b] = iF(gl)
    return {"t": float(t), "real_FPR": 100 * float(F0), "observed_reals_only_iF_FPR": float(obs),
            "null_median": float(np.median(null)), "null_q05": float(np.quantile(null, 0.05)),
            "null_q95": float(np.quantile(null, 0.95)), "p_null_ge_obs": float((null >= obs).mean()),
            "n_perm": int(n_perm), "n_real_videos": int(len(vids))}
