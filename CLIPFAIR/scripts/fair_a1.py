"""A1 operating point (pre-registration 2026-10-22-1920 section A; CESel plan step 4), the rule of
CFES/scripts/fair_oppoint.py + fair_oppoint_tables.py, copied VERBATIM (val_curves, pick_a1, full_metrics, constants)
so CESel does not import the CFES package (same module name `cfes`); CFES is read only.

  A1: one global threshold t per model on the logit scale, decision fake iff logit >= t. Grid = every distinct FF++
      validation logit. Minimise validation intersectional F_FPR (eight-group scheme) subject to validation pooled
      TPR >= 80% and pooled FPR <= 20%. Ties -> higher validation TPR, then higher t.
  Test metrics: cfes.metrics.fairness_block on rank_prob(logit - t) at 0.5 (identical decisions to logit >= t).
"""
import numpy as np

from cfes.metrics import SCHEMES, fairness_block, rank_prob

SETS = {"ffpp_test": ("ff++", "eight"), "dfdc": ("dfdc", "eight"), "celebdf": ("celebdf", "six"), "dfd": ("dfd", "six")}
NAMES = {"ffpp_test": "FF++", "dfdc": "DFDC", "celebdf": "Celeb-DF", "dfd": "DFD"}
TPR_MIN, FPR_MAX = 0.80, 0.20
# Ding et al. published "Ours" rows, Tables 1-2 (threshold 0.5): [AUC, gF_FPR, gF_DP, gesAUC, rF_FPR, rF_DP, resAUC,
# iF_FPR, iF_DP, iesAUC] (as in CFES/scripts/compare_ding.py)
DING_PUB = {"ffpp_test": [97.71, 0.53, 3.61, 96.45, 9.29, 4.35, 94.86, 20.18, 9.47, 86.91],
            "dfdc": [61.86, 1.76, 3.67, 57.94, 17.93, 7.58, 52.33, 37.37, 11.92, 39.03],
            "celebdf": [76.75, 6.41, 10.81, 69.68, 16.60, 7.29, 69.96, 23.32, 12.01, 63.27],
            "dfd": [81.46, 4.72, 5.49, 78.98, 2.09, 18.66, 77.44, 27.32, 22.81, 69.73]}
PUB_COLS = ["AUC", "g_F_FPR", "g_F_DP", "g_es_AUC", "r_F_FPR", "r_F_DP", "r_es_AUC", "i_F_FPR", "i_F_DP", "i_es_AUC"]


def val_curves(z, y, g, scheme):
    """For every distinct validation logit u_j (ascending), decision z >= u_j: pooled FPR/TPR and the intersectional
    F_FPR (fairness_block definition: groups holding both classes)."""
    u = np.unique(z)
    zr, zf = np.sort(z[y == 0]), np.sort(z[y == 1])
    fpr = 1 - np.searchsorted(zr, u, "left") / len(zr)
    tpr = 1 - np.searchsorted(zf, u, "left") / len(zf)
    f = np.zeros_like(u)
    elig = [k for k in range(len(SCHEMES[scheme])) if ((g == k) & (y == 0)).any() and ((g == k) & (y == 1)).any()]
    assert len(elig) >= 2
    for k in elig:
        rk = np.sort(z[(g == k) & (y == 0)])
        f += np.abs(1 - np.searchsorted(rk, u, "left") / len(rk) - fpr)
    return u, fpr, tpr, 100 * f


def pick_a1(u, fpr, tpr, f):
    ok = (tpr >= TPR_MIN) & (fpr <= FPR_MAX)
    if not ok.any():
        return None
    fm = f[ok].min()
    c = ok & (f <= fm + 1e-9)
    c &= tpr >= tpr[c].max() - 1e-12
    j = np.where(c)[0].max()                     # remaining tie -> higher threshold
    return int(j)


def full_metrics(z, y, g, scheme, t):
    f = fairness_block(y, rank_prob(z - t), g, scheme, 0.5)
    o = {"AUC": 100 * f["overall"]["auc"], "FPR": 100 * f["overall"]["FPR"], "TPR": 100 * f["overall"]["TPR"]}
    for lv, p in (("gender", "g"), ("race", "r"), ("intersection", "i")):
        for k in ("F_FPR", "F_DP", "es_AUC"):
            o[f"{p}_{k}"] = f[lv].get(k, float("nan"))
    o["groups_skipped_single_class"] = f["intersection"]["groups_skipped_single_class"]
    return o


def a1_for(val, test, schemes=None):
    """val = (z, y, g); test = {set: (z, y, g)} -> record as in fair_oppoint_tables.run (None t if no box point).
    Codex review F7 (25 Oct): EVERY supplied test set is scored (incl. FF2's ff2_test), with its scheme from `schemes`
    ({set: scheme}, e.g. from cfg.eval.test_sets), else the FF++ defaults in SETS; a set with no known scheme raises."""
    z, y, g = val
    u, fpr, tpr, f = val_curves(z, y, g, "eight")
    j = pick_a1(u, fpr, tpr, f)
    if j is None:
        return {"t": None}
    t = float(u[j])
    rec = {"t": t, "val_FPR": 100 * fpr[j], "val_TPR": 100 * tpr[j], "val_i_F_FPR": float(f[j]),
           "n_eligible": int(((tpr >= TPR_MIN) & (fpr <= FPR_MAX)).sum()), "test": {}}
    for k, (zt, yt, gt) in test.items():
        sch = (schemes or {}).get(k) or (SETS[k][1] if k in SETS else None)
        if sch is None:
            raise KeyError(f"a1_for: no group scheme for test set {k!r}; pass schemes={{set: scheme}}")
        rec["test"][k] = full_metrics(zt, yt, gt, sch, t)
    return rec
