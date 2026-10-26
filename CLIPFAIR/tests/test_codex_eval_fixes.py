"""Codex review 25 Oct: fixed-behaviour checks for F3 (validation AUC from logits / rank_prob), F4 (validation-TPR
threshold stored and applied in logit units), F7 (A1 scores every supplied test set incl. ff2_test) and F9 (exact
deletion reconstruction). Same inputs as codex_review/test_metrics_protocol.py, asserting the CORRECT result.
  CUDA_VISIBLE_DEVICES="" ~/.conda/envs/pulid/bin/python tests/test_codex_eval_fixes.py"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from cfes.metrics import evaluate_scores, fairness_block, rank_prob, threshold_at_tpr  # noqa: E402
import ces_posteval as P  # noqa: E402
import fair_a1 as A  # noqa: E402
import train as T  # noqa: E402

# ---- F3: validate() with saturated logits
y = np.tile([0, 0, 1, 1], 2); g = np.repeat([0, 1], 4); z = np.tile([-50., 40., 41., 42.], 2)
p = 1 / (1 + np.exp(-z))
cfg = {"optim": {"amp": False}, "eval": {"prob_float64": True, "rank_metrics": True, "val_tpr": .9},
       "data": {"val": {"scheme": "eight"}}}
with patch.object(T.engine, "predict", return_value=(p, z)):
    res = T.validate(None, cfg, None, pd.DataFrame({"label": y, "group": g}), None)
exact = 100 * roc_auc_score(y, z)
assert res["auc"] == exact == 100 * fairness_block(y, rank_prob(z), g, "eight", .5)["overall"]["auc"], res["auc"]
for nm, st in res["ding@0.5"]["intersection"]["groups"].items():
    m = np.isin(g, [{"male,asian": 0, "male,white": 1}[nm]])
    assert st["auc"] == roc_auc_score(y[m], z[m]), nm
assert res["threshold_tpr_units"] == "logit" and res["threshold_tpr"] == threshold_at_tpr(y, z, .9)
assert res["oppoint"]["pooled_TPR"] >= 0          # operating point still computed from logits
cfg["eval"]["rank_metrics"] = False               # rank_metrics off: old (probability) behaviour kept
with patch.object(T.engine, "predict", return_value=(p, z)):
    res0 = T.validate(None, cfg, None, pd.DataFrame({"label": y, "group": g}), None)
assert res0["auc"] == 75.0
print(f"PASS F3: validate() AUC from logits = {res['auc']} (saturated probabilities gave 75.0); group AUCs exact; "
      f"threshold in logit units")

# ---- F4: one fixed logit threshold on every test set
yv = np.array([0, 0, 1, 1]); zv = np.array([-2., -1., 1., 2.]); zt = np.array([-100., -1., 1., 2.])
thr = threshold_at_tpr(yv, zv, .9)
r = evaluate_scores(yv, rank_prob(zt), np.array([0, 0, 0, 0]), "eight", val_threshold=thr, thr_scores=zt)
assert r["threshold_val_units"] == "logit" and r["acc@val"] == 100.0 and r["tpr@val"] == 100.0 and r["fpr@val"] == 0.0
assert r["auc"] == 100 * roc_auc_score(yv, zt)
# the same threshold gives the same decision for the same logit whatever the other test images are
r2 = evaluate_scores(yv, rank_prob(zv), np.array([0, 0, 0, 0]), "eight", val_threshold=thr, thr_scores=zv)
assert r2["acc@val"] == r["acc@val"]
src = open(os.path.join(ROOT, "scripts", "evaluate_ces.py")).read()
assert "threshold_at_tpr(val_df[\"label\"].values, zv.astype(np.float64)" in src and "thr_scores=z.astype(np.float64)[keep]" in src
print(f"PASS F4: logit threshold {thr} -> test decisions [real, real, fake, fake] (acc@val 100) despite the -100 outlier")

# ---- F7: A1 keeps every supplied test set with its scheme
rng = np.random.default_rng(5)
y = np.tile([0, 1], 80); g = np.repeat(np.arange(8), 20); z = rng.normal(size=160) + 3 * y
rec = A.a1_for((z, y, g), {"ffpp_test": (z, y, g), "ff2_test": (z, y, g)}, {"ff2_test": "eight"})
assert rec["t"] is not None and set(rec["test"]) == {"ffpp_test", "ff2_test"}
assert rec["test"]["ff2_test"] == A.full_metrics(z, y, g, "eight", rec["t"])
try:
    A.a1_for((z, y, g), {"mystery": (z, y, g)})
    raise AssertionError("unknown scheme must raise")
except KeyError:
    pass
ps = open(os.path.join(ROOT, "scripts", "ces_posteval.py")).read()
assert "for k, d in T.items()}, schemes)" in ps
print("PASS F7: a1_for scores ff2_test (and raises on a set without a scheme); ces_posteval passes the config schemes")

# ---- F9: exact deletion reconstruction
sk = np.array([[.1, .2, .3, .4], [.1, .2, .3, .4], [1., 2., 3., 4.], [1., 2., 3., 4.]], dtype=np.float32)
s = sk.sum(1).astype(np.float64)
b, beta0 = 0.25, np.array([0.5, -0.1, 0.2, 0.3])
full = P.intervene(s, sk, b, beta0, np.ones(4))
assert np.ptp(full) == 0.0 and full[0] == b + beta0.sum()
assert roc_auc_score([0, 0, 1, 1], full) == 0.5
assert np.array_equal(P.intervene(s, sk, b, beta0, np.zeros(4)), s)
one = P.intervene(s, sk, b, beta0, [1, 0, 0, 0])
assert np.allclose(one, b + beta0[0] + sk[:, 1:].astype(np.float64).sum(1), rtol=0, atol=0)
half = P.intervene(s, sk, b, beta0, [0.5, 0, 0, 0])
assert np.allclose(half, b + 0.5 * beta0[0] + 0.5 * sk[:, 0].astype(np.float64) + sk[:, 1:].astype(np.float64).sum(1))
print(f"PASS F9: full deletion = constant {full[0]} (AUC 50.0; the old residual form gave 100.0)")
print("ALL_PASS test_codex_eval_fixes")
