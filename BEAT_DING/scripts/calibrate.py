"""Calibration as a SEPARATE analysis factor (CFES revision study).

One common intercept b, fitted on FF++ VALIDATION logits only, by a predefined objective:
    b* = argmin_b  0.5 * mean_{fake} softplus(-(z + b)) + 0.5 * mean_{real} softplus(z + b)
(class-balanced validation log loss). Never fitted per group or per external dataset, never chosen by test metrics.
Applied unchanged to every test set: calibrated p = sigmoid(z + b). Reports Ding metrics at threshold 0.5 before and
after, and checks that AUC is numerically unchanged (a common shift preserves the ranking).

  python scripts/calibrate.py --eval runs_study/<run>/eval      # needs val_scores.npz and <set>_scores.npz with 'logit'
Writes <eval>/calibration.json and <eval>/summary_calibrated.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes.config import load_config  # noqa: E402
from cfes.metrics import fairness_block, rank_prob  # noqa: E402

COLS = ["gender_F_FPR", "gender_F_DP", "gender_es_AUC", "race_F_FPR", "race_F_DP", "race_es_AUC",
        "intersection_F_FPR", "intersection_F_DP", "intersection_es_AUC"]


def softplus(x):
    return np.logaddexp(0.0, x)


def fit_intercept(z, y):
    zf, zr = z[y == 1].astype(np.float64), z[y == 0].astype(np.float64)
    obj = lambda b: 0.5 * softplus(-(zf + b)).mean() + 0.5 * softplus(zr + b).mean()
    # The objective is convex in b. Search +-60 first (unchanged for every interior solution); if the minimum sits on
    # the bound, widen the interval (review-4 ResNet arm C pushed logits down by more than 60; Amendment 10).
    for lim in (60.0, 600.0, 6000.0):
        r = minimize_scalar(obj, bounds=(-lim, lim), method="bounded", options={"xatol": 1e-6})
        if abs(r.x) < lim - 1e-3:
            break
    return float(r.x), float(r.fun), float(obj(0.0))


def row(y, z, g, scheme, b):
    p = rank_prob(z.astype(np.float64) + b)      # same 0.5 decisions as sigmoid(z + b); exact (tie-free) AUC
    f = fairness_block(y, p, g, scheme, 0.5)
    out = {"AUC": 100 * f["overall"]["auc"], "FPR@0.5": 100 * float((p[y == 0] >= 0.5).mean()),  # p >= 0.5  <=>  z + b >= 0
           "TPR@0.5": 100 * float((p[y == 1] >= 0.5).mean())}
    for c in COLS:
        lvl, k = c.split("_", 1)
        out[c] = f[lvl].get(k, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    a = ap.parse_args()
    run = os.path.dirname(os.path.normpath(a.eval))
    cfg = load_config(os.path.join(run, "config.yaml"))
    v = np.load(os.path.join(a.eval, "val_scores.npz"), allow_pickle=True)
    if "logit" not in v:
        raise SystemExit("val_scores.npz has no logits (evaluated with older code); re-run evaluate.py")
    b, loss_b, loss_0 = fit_intercept(v["logit"], v["label"])
    rep = {"intercept_b": b, "val_balanced_logloss_at_b": loss_b, "val_balanced_logloss_at_0": loss_0,
           "objective": "class-balanced validation log loss, common intercept", "sets": {}}
    rows = []
    for name, spec in cfg["eval"]["test_sets"].items():
        f = os.path.join(a.eval, f"{name}_scores.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        y, lg, g = z["label"], z["logit"], z["group"]
        raw, cal = row(y, lg, g, spec.get("scheme", "eight"), 0.0), row(y, lg, g, spec.get("scheme", "eight"), b)
        from sklearn.metrics import roc_auc_score
        a0, a1 = roc_auc_score(y, lg.astype(np.float64)), roc_auc_score(y, lg.astype(np.float64) + b)
        if a0 != a1:      # exact on logits: a common shift preserves every rank
            raise AssertionError(f"{name}: logit AUC changed under a common shift ({a0} vs {a1})")
        if abs(raw["AUC"] - cal["AUC"]) > 1e-9:
            raise AssertionError(f"{name}: AUC moved under a common shift ({raw['AUC']} vs {cal['AUC']})")
        rep["sets"][name] = {"raw": raw, "calibrated": cal}
        rows += [{"test_set": name, "version": "raw", **raw}, {"test_set": name, "version": "calibrated", **cal}]
    json.dump(rep, open(os.path.join(a.eval, "calibration.json"), "w"), indent=1, default=float)
    pd.DataFrame(rows).to_csv(os.path.join(a.eval, "summary_calibrated.csv"), index=False, float_format="%.2f")
    print(f"b = {b:+.4f} (val balanced log loss {loss_0:.4f} -> {loss_b:.4f})")
    print(pd.DataFrame(rows)[["test_set", "version", "AUC", "FPR@0.5", "TPR@0.5", "intersection_F_FPR",
                               "intersection_F_DP"]].to_string(index=False))


if __name__ == "__main__":
    main()
