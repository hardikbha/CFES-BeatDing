"""Evaluation-lock tests: our metric code against the reference evaluators, on deterministic synthetic data.

  /home/sachin.chaudhary/.conda/envs/qwen38/bin/python tests/test_metrics.py

References (imported, never edited):
  Ding et al. Eq. 8-10 : agents/DINGREPRO_LIN/code/Ding_mod/eval_ding_protocol.py  (ding_metrics)
  FairReL              : <paths.fairrel_repo>/fairrel/metrics.py  (fairness_report, threshold_at_tpr); its sha256 must
                         equal third_party/fairrel/PROVENANCE.txt
Independent formulas (written here from the definitions in EVAL_PROTOCOL.md, not from cfes/metrics.py) cover the
metrics no reference evaluator implements (macro / worst-group AUC, AUC/FPR/TPR gaps, F_OAE, per-method AUC).
"""
import hashlib
import os
import sys
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cfes import metrics as CM  # noqa: E402
from cfes.config import load_config  # noqa: E402
from cfes.data import SCHEMES  # noqa: E402

CFG = load_config(os.path.join(ROOT, "configs", "ffpp_ding.yaml"))
DING_DIR = os.path.join(os.path.dirname(ROOT), "DINGREPRO_LIN", "code", "Ding_mod")
sys.path.insert(0, DING_DIR)
try:                       # Beat-Ding copy: the pulid / aiface envs lack efficientnet_pytorch, which Ding's evaluator
    import efficientnet_pytorch  # noqa: F401,E402   imports only to register a backbone (its metric code never uses it)
except ImportError:
    import types  # noqa: E402
    sys.modules["efficientnet_pytorch"] = types.SimpleNamespace(EfficientNet=object)
import eval_ding_protocol as DREF  # noqa: E402
sys.path.insert(0, CFG["paths"]["fairrel_repo"])
import fairrel.metrics as FREF  # noqa: E402
import protocol_eval as PE  # noqa: E402

RESULTS = []
TOL = 1e-9


def case(fn):
    def run():
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", ""))
        except Exception as e:
            RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
    run.__name__ = fn.__name__
    return run


def synth(scheme, n=6000, seed=0, single_class=None):
    """Scores with group-dependent shift and spread; optional groups forced to one class {group: 0 or 1}."""
    rng = np.random.default_rng(seed)
    k = len(SCHEMES[scheme])
    g = rng.integers(0, k, n)
    y = rng.integers(0, 2, n)
    for grp, cls in (single_class or {}).items():
        y[g == grp] = cls
    shift = np.linspace(-0.8, 0.8, k)[g]
    logit = (2 * y - 1) * (1.0 + 0.3 * np.cos(g)) + shift + rng.normal(0, 1.2, n)
    p = 1 / (1 + np.exp(-logit))
    p = np.round(p, 3)                                    # creates ties, as real bf16 scores do
    return y, p, g


DING_SCHEME = {"eight": DREF.EIGHT, "six": DREF.SIX}


@case
def test_ding_eq8_10_match_reference():
    for scheme in ("eight", "six"):
        for single in (None, {0: 0, 2: 1}):               # second case: single-class groups as in Celeb-DF / DFD
            y, p, g = synth(scheme, seed=1, single_class=single)
            ref = DREF.ding_metrics(y, p, g, DING_SCHEME[scheme])
            ours = CM.fairness_block(y, p, g, scheme, 0.5)
            assert abs(100 * ours["overall"]["auc"] - ref["overall_auc"]) < TOL
            for lvl in ("gender", "race", "intersection"):
                assert set(ours[lvl]["groups"]) == set(ref["groups"][lvl]), (scheme, lvl)
                for k in ("F_FPR", "F_DP", "es_AUC"):
                    assert abs(ours[lvl][k] - ref[lvl][k]) < TOL, (scheme, single, lvl, k, ours[lvl][k], ref[lvl][k])


@case
def test_pooled_auc_and_direction():
    y, p, g = synth("eight", seed=2)
    assert abs(CM.evaluate_scores(y, p, g, "eight")["auc"] - 100 * roc_auc_score(y, p)) < TOL
    # identical score distribution in every group -> zero FPR gaps, es-AUC = AUC; a group shift raises F_FPR
    rng = np.random.default_rng(3)
    y2 = np.tile([0, 1], 4000)
    base = np.where(y2 == 1, 0.7, 0.3) + rng.normal(0, 0.15, y2.size)
    g2 = np.repeat(np.arange(8), 1000)
    fair = CM.fairness_block(y2, np.tile(base[:1000], 8), g2, "eight", 0.5)["intersection"]
    assert fair["F_FPR"] < 1e-9 and abs(fair["es_AUC"] - 100 * roc_auc_score(y2[:1000], base[:1000])) < 1e-9
    biased = np.tile(base[:1000], 8) + 0.1 * (g2 == 3)
    assert CM.fairness_block(y2, biased, g2, "eight", 0.5)["intersection"]["F_FPR"] > 1.0


@case
def test_added_group_metrics_independent():
    y, p, g = synth("eight", seed=4, single_class={7: 0})
    blk = CM.fairness_block(y, p, g, "eight", 0.5)["intersection"]
    auc, fpr, tpr, acc = [], [], [], []
    for k in range(8):
        m = g == k
        if len(np.unique(y[m])) < 2:
            continue
        auc.append(roc_auc_score(y[m], p[m]))
        fpr.append(np.mean(p[m & (y == 0)] >= 0.5))
        tpr.append(np.mean(p[m & (y == 1)] >= 0.5))
        acc.append(np.mean((p[m] >= 0.5) == (y[m] == 1)))
    exp = dict(macro_group_auc=100 * np.mean(auc), group_auc_gap=100 * (max(auc) - min(auc)),
               worst_group_auc=100 * min(auc), fpr_gap=100 * (max(fpr) - min(fpr)), tpr_gap=100 * (max(tpr) - min(tpr)),
               worst_group_tpr=100 * min(tpr), max_group_fpr=100 * max(fpr), F_OAE=100 * (max(acc) - min(acc)),
               F_MEO_gap=100 * max(max(fpr) - min(fpr), max(tpr) - min(tpr)))
    for k, v in exp.items():
        assert abs(blk[k] - v) < 1e-9, (k, blk[k], v)
    assert blk["groups_skipped_single_class"] == ["nonmale,others"], blk["groups_skipped_single_class"]
    r = CM.evaluate_scores(y, p, g, "eight", val_threshold=0.62)
    assert abs(r["fpr@val"] - 100 * np.mean(p[y == 0] >= 0.62)) < 1e-9
    assert abs(r["tpr@val"] - 100 * np.mean(p[y == 1] >= 0.62)) < 1e-9


@case
def test_per_method_auc():
    y, p, g = synth("eight", seed=5)
    meth = np.where(y == 1, np.array(["A", "B", "C"])[np.arange(y.size) % 3], "real")
    per = CM.evaluate_scores(y, p, g, "eight", methods=meth)["per_method"]
    for mth in ("A", "B", "C"):
        sel = (y == 0) | (meth == mth)
        assert abs(per[mth]["auc"] - 100 * roc_auc_score(y[sel], p[sel])) < 1e-9


@case
def test_fairrel_reference_hash():
    ref = os.path.join(CFG["paths"]["fairrel_repo"], "fairrel", "metrics.py")
    cp = os.path.join(ROOT, "third_party", "fairrel", "metrics.py")
    h = [hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (ref, cp)]
    prov = open(os.path.join(ROOT, "third_party", "fairrel", "PROVENANCE.txt")).read()
    assert h[0] == h[1] and h[0] in prov, h


@case
def test_fairrel_threshold_and_report():
    for seed in range(5):
        y, p, g = synth("eight", seed=10 + seed)
        assert CM.threshold_at_tpr(y, p, 0.90) == FREF.threshold_at_tpr(p, y, 0.90)
    y, p, g = synth("six", seed=20, single_class={0: 0, 2: 1})       # Celeb-DF-like single-class groups
    fy, fp, fg, _ = PE.to_fairrel(y, p, g, "six")
    thr = FREF.threshold_at_tpr(fp, fy, 0.90)
    ours = PE.fairrel_point(FREF, fy, fp, fg, thr)
    ref = FREF.fairness_report(fp, fy, fg, thr)
    for k in ("auc", "auc_avg", "auc_gap", "f_fpr", "f_meo", "fpr_overall", "tpr_overall"):
        assert ours[k] == ref[k], k
    assert abs(ref["f_meo"] - max(ours["fpr_gap"], ours["tpr_gap"])) < 1e-9     # f_meo = max(FPR gap, TPR gap)
    # mapping to FairReL's SUBGROUPS order (M-W, M-A, M-B, F-W, F-A, F-B); Others dropped
    names8 = SCHEMES["eight"]
    for ours_id, fr_id in PE.FAIRREL_MAP["eight"].items():
        sex, race = names8[ours_id].split(",")
        assert FREF.SUBGROUPS[fr_id] == ("M" if sex == "male" else "F") + "-" + race[0].upper(), (ours_id, fr_id)
    names6 = SCHEMES["six"]
    for ours_id, fr_id in PE.FAIRREL_MAP["six"].items():
        sex, race = names6[ours_id].split(",")
        assert FREF.SUBGROUPS[fr_id] == ("M" if sex == "male" else "F") + "-" + race[0].upper(), (ours_id, fr_id)
    assert all("others" not in names8[i] for i in PE.FAIRREL_MAP["eight"])


@case
def test_known_difference_meo_gap_vs_fairrel():
    """Pins the documented discrepancy: our F_MEO_gap uses only groups with both classes; FairReL's f_meo also
    counts a group's FPR when it has only reals. They agree when every group has both classes."""
    y, p, g = synth("six", seed=30)
    fy, fp, fg, keep = PE.to_fairrel(y, p, g, "six")
    ref = FREF.fairness_report(fp, fy, fg, 0.5)
    ours = CM.fairness_block(y[keep], p[keep], g[keep], "six", 0.5)["intersection"]
    assert abs(ours["F_MEO_gap"] - ref["f_meo"]) < 1e-9
    y, p, g = synth("six", seed=31, single_class={0: 0})
    p[(g == 0)] = 0.99                                    # all-real group with FPR 1.0: FairReL counts it, we skip it
    fy, fp, fg, keep = PE.to_fairrel(y, p, g, "six")
    ref = FREF.fairness_report(fp, fy, fg, 0.5)
    ours = CM.fairness_block(y[keep], p[keep], g[keep], "six", 0.5)["intersection"]
    assert ref["f_meo"] > ours["F_MEO_gap"] + 10, (ref["f_meo"], ours["F_MEO_gap"])


@case
def test_protocol_eval_points_run():
    y, p, g = synth("eight", seed=40)
    d = PE.ding_point(y, p, g, "eight")
    ref = DREF.ding_metrics(y, p, g, DREF.EIGHT)
    assert abs(d["intersection_F_FPR"] - ref["intersection"]["F_FPR"]) < TOL
    for k in ("macro_group_auc", "group_auc_gap", "fpr_gap", "tpr_gap", "max_group_fpr", "F_OAE"):
        assert np.isfinite(d[f"intersection_{k}"]), k


@case
def test_prereg_endpoints():
    rng = np.random.default_rng(50)
    n = 4000
    g = rng.integers(0, 4, n)
    y = rng.integers(0, 2, n)
    cl = np.array([f"v{g_}_{y_}_{i % 7}" for i, (g_, y_) in enumerate(zip(g, y))])
    cl[(g == 3) & (y == 1)] = "v3_fake_only_two"           # group 3: fakes come from 1 video -> not eligible
    p = np.clip(0.5 + (2 * y - 1) * 0.2 + 0.05 * g + rng.normal(0, 0.2, n), 0, 1)
    el = PE.eligible_groups(y, g, cl)
    assert el == [0, 1, 2], el
    thr = 0.6
    ep = PE.endpoint_point(y, p, g, el, thr)
    auc = [100 * roc_auc_score(y[g == k], p[g == k]) for k in el]
    tpr = [100 * np.mean(p[(g == k) & (y == 1)] >= thr) for k in el]
    fpr = [100 * np.mean(p[(g == k) & (y == 0)] >= thr) for k in el]
    exp = dict(worst_group_auc=min(auc), best_group_auc=max(auc), macro_group_auc=np.mean(auc),
               group_auc_gap=max(auc) - min(auc),
               worst_group_tpr=min(tpr), tpr_gap=max(tpr) - min(tpr), fpr_gap=max(fpr) - min(fpr),
               max_group_fpr=max(fpr), pooled_auc=100 * roc_auc_score(y, p), pooled_fpr=100 * np.mean(p[y == 0] >= thr))
    for k, v in exp.items():
        assert abs(ep[k] - v) < 1e-9, (k, ep[k], v)


@case
def test_paired_p_values():
    pts = {"ep:pooled_auc": 0.5, "x": 0.0}
    samples = [{"ep:pooled_auc": v, "x": w} for v, w in zip(np.linspace(-2, 3, 101), np.linspace(-1, 1, 101))]
    r = PE.summarize(pts, samples, with_p=True)
    arr = np.linspace(-2, 3, 101)
    assert abs(r["ep:pooled_auc"]["p_two_sided"] - 2 * min((arr <= 0).mean(), (arr >= 0).mean())) < 1e-12
    assert abs(r["ep:pooled_auc"]["p_noninferiority"] - (arr <= -1.0).mean()) < 1e-12
    assert "p_noninferiority" not in r["x"] and abs(r["x"]["p_two_sided"] - 1.0) < 1e-12
    assert "p_two_sided" not in PE.summarize(pts, samples)["x"]


def main():
    tests = [test_ding_eq8_10_match_reference, test_pooled_auc_and_direction, test_added_group_metrics_independent,
             test_per_method_auc, test_fairrel_reference_hash, test_fairrel_threshold_and_report,
             test_known_difference_meo_gap_vs_fairrel, test_protocol_eval_points_run, test_prereg_endpoints,
             test_paired_p_values]
    for t in tests:
        t()
    for name, status, msg in RESULTS:
        print(f"{status}  {name}" + (f"\n      {msg}" if msg else ""))
    n_fail = sum(s == "FAIL" for _, s, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
