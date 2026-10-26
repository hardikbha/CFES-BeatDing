"""Paper-protocol metrics with video/person-clustered bootstrap CIs, from the score files evaluate.py saves.

  python scripts/protocol_eval.py --run runs/clip_s7_p2 --eval eval                    # one run
  python scripts/protocol_eval.py --run runs/clip_s7_p2 --eval eval \
         --compare runs/clip_s7_p1 --compare_eval eval                                  # paired difference (run - compare)

Protocols (each uses that paper's own definition and threshold rule):
  ding     : Ding et al. Eq. 8-10 at the fixed 0.5 threshold, 8 (FF++/DFDC) or 6 (Celeb-DF/DFD) groups incl. Others
             (cfes.metrics.fairness_block, verified against their released formulas).
  fairrel  : FairReL's own fairrel.metrics.fairness_report, imported unchanged: 6 groups (M/F x W/A/B), Others
             excluded, ONE threshold from FF++ validation at TPR = 0.90 (fairrel.metrics.threshold_at_tpr, Others
             excluded) reused on every test set. f_meo = max(FPR max-min, TPR max-min).
  FairForensics has not released code; its F_DP/F_OAE definitions and threshold rule are unverified, so it is not
  offered as a protocol here (cfes.metrics reports F_DP/F_OAE with stated definitions instead).
CIs: 95% percentile bootstrap over clusters (FF++/DFDC/Celeb-DF/DFD: video key from the frame name; FairForge-2:
person_id), never over frames. The threshold is fixed on validation and not resampled.

Seed-averaged paired comparison (added 16 Oct, PREREG_MFER.md):
  python scripts/protocol_eval.py --pairs runs/D_s7:runs/A_s7 runs/D_s17:runs/A_s17 runs/D_s27:runs/A_s27 \
         --eval eval --compare_eval eval --out D_vs_A.json
  Every run must have scored the same test rows. Each bootstrap replicate draws ONE set of clusters and applies it
  to every seed, so the CI covers the mean over seeds of (run - compare) with test-set sampling uncertainty; the
  per-seed point differences are also written (seed variation is reported separately, not folded into the CI).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes.config import load_config  # noqa: E402
from cfes.metrics import fairness_block, threshold_at_tpr  # noqa: E402
from cfes.data import SCHEMES  # noqa: E402

FAIRREL_MAP = {"eight": {0: 1, 1: 0, 2: 2, 4: 4, 5: 3, 6: 5},     # male,asian->M-A ... others dropped
               "six": {0: 0, 1: 2, 3: 3, 4: 5}}



LOGIT_SCALE = 1.0e4   # fixed for every set: 0.5 + z / (2 * LOGIT_SCALE) is strictly monotone, keeps z >= 0 <=> p >= 0.5,
                      # gives exact (tie-free) AUCs and lets a validation threshold transfer to test (same map everywhere)


def load_scores(path):
    """Score file as a dict; 'prob' is replaced by a tie-free monotone map of the logit when the logit was saved
    (CFES logits saturate float sigmoids: STUDY_PLAN Amendment 2)."""
    z = dict(np.load(path, allow_pickle=True))
    if "logit" in z:
        lg = z["logit"].astype(np.float64)
        assert np.abs(lg).max() < LOGIT_SCALE, np.abs(lg).max()
        z["prob"] = 0.5 + lg / (2 * LOGIT_SCALE)
    return z

def fairrel_module(cfg):
    """FairReL's own metrics.py, imported unchanged from the official repository."""
    repo = cfg["paths"]["fairrel_repo"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import fairrel.metrics as fm
    return fm


def to_fairrel(y, p, g, scheme):
    mp = FAIRREL_MAP[scheme]
    keep = np.isin(g, list(mp))
    return y[keep], p[keep], np.array([mp[x] for x in g[keep]]), keep


def ding_point(y, p, g, scheme):
    d = fairness_block(y, p, g, scheme, 0.5)
    out = {"auc": 100 * d["overall"]["auc"]}
    for lvl in ("gender", "race", "intersection"):
        for k in ("F_FPR", "F_DP", "es_AUC", "worst_group_auc", "worst_group_tpr",
                  # added 16 Oct (EVAL_PROTOCOL.md); same groups and 0.5 threshold as above
                  "macro_group_auc", "group_auc_gap", "fpr_gap", "tpr_gap", "max_group_fpr", "F_OAE"):
            out[f"{lvl}_{k}"] = d[lvl].get(k, np.nan)
    for name, s in d["intersection"]["groups"].items():
        out[f"FPR[{name}]"], out[f"TPR[{name}]"] = 100 * s["FPR"], 100 * s["TPR"]
    return out


def fairrel_point(fm, y, p, g, thr):
    r = fm.fairness_report(p, y, g, thr)
    out = {k: r[k] for k in ("auc", "auc_avg", "auc_gap", "f_fpr", "f_meo", "fpr_overall", "tpr_overall")}
    pa = r["per_subgroup_auc"]
    out["worst_group_auc"] = min(pa.values()) if pa else np.nan
    tprs = [100 * float((p[(g == k) & (y == 1)] >= thr).mean()) for k in np.unique(g) if ((g == k) & (y == 1)).any()]
    out["worst_group_tpr"] = min(tprs) if tprs else np.nan
    for name, v in r["per_subgroup_fpr"].items():
        out[f"FPR[{name}]"] = v
    # added 16 Oct (EVAL_PROTOCOL.md): gaps at FairReL's common threshold, FairReL's group inclusion rule (a group
    # enters the FPR side if it has reals and the TPR side if it has fakes)
    fprs = list(r["per_subgroup_fpr"].values())
    out["fpr_gap"] = max(fprs) - min(fprs) if len(fprs) > 1 else np.nan
    out["tpr_gap"] = max(tprs) - min(tprs) if len(tprs) > 1 else np.nan
    out["max_group_fpr"] = max(fprs) if fprs else np.nan
    return out


# PREREG_MFER.md endpoints ("ep:"), added 16 Oct. Intersectional groups of the Ding scheme (Others included) that
# have >= EP_MIN_FRAMES frames AND >= EP_MIN_VIDEOS distinct videos of EACH class in the FULL test set; the eligible
# list is fixed once per dataset from labels/clusters only (never from scores) and is not re-chosen per replicate.
EP_MIN_FRAMES, EP_MIN_VIDEOS, EP_VAL_TPR = 50, 5, 0.90
# PREREG_MFER.md non-inferiority margins on paired differences (run - compare): key -> (margin, higher_is_better)
NI_MARGINS = {"ep:pooled_auc": (-1.0, True), "ep:worst_group_auc": (-1.0, True), "ep:best_group_auc": (-1.0, True),
              "ding:intersection_F_FPR": (5.0, False)}


def eligible_groups(y, g, cl):
    out = []
    for k in np.unique(g):
        ok = True
        for c in (0, 1):
            m = (g == k) & (y == c)
            ok &= int(m.sum()) >= EP_MIN_FRAMES and len(set(cl[m])) >= EP_MIN_VIDEOS
        if ok:
            out.append(int(k))
    return out


def val_threshold(run, ev):
    """Common operating point: pooled TPR = 0.90 on the FF++ validation split, all groups (Others included), the same
    rule as scripts/evaluate.py's @val threshold. One threshold per run, reused on every test set and every group."""
    v = load_scores(os.path.join(run, ev, "val_scores.npz"))
    return threshold_at_tpr(v["label"], v["prob"], EP_VAL_TPR)


def endpoint_point(y, p, g, eligible, thr):
    from sklearn.metrics import roc_auc_score
    out = {"pooled_auc": 100 * roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
           "pooled_fpr": 100 * float((p[y == 0] >= thr).mean()) if (y == 0).any() else np.nan,
           "pooled_tpr": 100 * float((p[y == 1] >= thr).mean()) if (y == 1).any() else np.nan}
    auc, tpr, fpr = [], [], []
    for k in eligible:
        m = g == k
        if len(np.unique(y[m])) < 2:           # can happen inside a bootstrap replicate -> replicate dropped
            return {**out, **{k2: np.nan for k2 in ("worst_group_auc", "best_group_auc", "macro_group_auc",
                                                     "group_auc_gap", "worst_group_tpr", "tpr_gap", "fpr_gap",
                                                     "max_group_fpr")}}
        auc.append(100 * roc_auc_score(y[m], p[m]))
        tpr.append(100 * float((p[m & (y == 1)] >= thr).mean()))
        fpr.append(100 * float((p[m & (y == 0)] >= thr).mean()))
    if len(auc) < 2:
        return out
    out.update(worst_group_auc=min(auc), best_group_auc=max(auc), macro_group_auc=float(np.mean(auc)),
               group_auc_gap=max(auc) - min(auc),
               worst_group_tpr=min(tpr), tpr_gap=max(tpr) - min(tpr), fpr_gap=max(fpr) - min(fpr),
               max_group_fpr=max(fpr))
    return out


def cluster_index(cl):
    codes, uniq = pd.factorize(cl)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def bootstrap(fn, idx_lists, reps, seed):
    rng = np.random.default_rng(seed)
    n = len(idx_lists)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, n, n)
        out.append(fn(np.concatenate([idx_lists[i] for i in pick])))
    return out


def summarize(point, samples, with_p=False):
    """95% percentile CI. with_p (paired differences only): two-sided bootstrap p for difference = 0,
    p = min(1, 2 * min(P*(d <= 0), P*(d >= 0))) over valid replicates (PREREG_MFER.md, Holm-adjusted there)."""
    res = {}
    for k, v in point.items():
        arr = np.array([s.get(k, np.nan) for s in samples], dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) > 10 else (np.nan, np.nan))
        res[k] = {"point": float(v), "lo": float(lo), "hi": float(hi), "valid_reps": int(len(arr))}
        if with_p and len(arr) > 10:
            res[k]["p_two_sided"] = float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean())))
            if k in NI_MARGINS:      # one-sided p for H0: the difference is beyond the margin (worse)
                m, hib = NI_MARGINS[k]
                res[k]["ni_margin"] = m
                res[k]["p_noninferiority"] = float((arr <= m).mean() if hib else (arr >= m).mean())
    return res


def _thr(fm, run, ev, scheme):
    v = load_scores(os.path.join(run, ev, "val_scores.npz"))
    vy, vp, _, _ = to_fairrel(v["label"], v["prob"], v["group"], scheme)
    return fm.threshold_at_tpr(vp, vy, 0.90)


def main_pairs(a):
    pairs = [tuple(x.split(":")) for x in a.pairs]
    cfg = load_config(os.path.join(pairs[0][0], "config.yaml"))
    fm = fairrel_module(cfg)
    vs = cfg["data"]["val"].get("scheme", "eight")
    thr = {r: _thr(fm, r, a.eval, vs) for r, _ in pairs}
    thr.update({c: _thr(fm, c, a.compare_eval, vs) for _, c in pairs})
    tval = {r: val_threshold(r, a.eval) for r, _ in pairs}
    tval.update({c: val_threshold(c, a.compare_eval) for _, c in pairs})
    report = {"pairs": pairs, "eval": a.eval, "compare_eval": a.compare_eval, "reps": a.reps,
              "fairrel_thresholds": thr, "datasets": {}}
    for name, spec in cfg["eval"]["test_sets"].items():
        files = {r: os.path.join(r, a.eval, f"{name}_scores.npz") for r, _ in pairs}
        files.update({c: os.path.join(c, a.compare_eval, f"{name}_scores.npz") for _, c in pairs})
        if not all(os.path.exists(f) for f in files.values()):
            continue
        z = {k: load_scores(f) for k, f in files.items()}
        z0 = z[pairs[0][0]]
        y, g, cl = z0["label"], z0["group"], z0["cluster"].astype(str)
        for k, zz in z.items():
            if not (np.array_equal(zz["label"], y) and np.array_equal(zz["cluster"].astype(str), cl)):
                raise SystemExit(f"{name}: {k} was scored on different rows; cannot pair")
        scheme = spec.get("scheme", "eight")
        elig = eligible_groups(y, g, cl)

        def metrics(idx, run):
            yy, pp, gg = y[idx], z[run]["prob"][idx], g[idx]
            out = {f"ding:{k}": v for k, v in ding_point(yy, pp, gg, scheme).items()}
            fy, fp, fg, _ = to_fairrel(yy, pp, gg, scheme)
            out.update({f"fairrel:{k}": v for k, v in fairrel_point(fm, fy, fp, fg, thr[run]).items()})
            out.update({f"ep:{k}": v for k, v in endpoint_point(yy, pp, gg, elig, tval[run]).items()})
            return out

        def per_seed(idx):
            out = []
            for r, c in pairs:
                m, mc = metrics(idx, r), metrics(idx, c)
                out.append({k: m[k] - mc[k] for k in m if k in mc})
            return out

        def fn(idx):
            ds = per_seed(idx)
            keys = set.intersection(*[set(d) for d in ds])
            return {k: float(np.mean([d[k] for d in ds])) for k in keys}

        allidx = np.arange(len(y))
        idx_lists = cluster_index(cl)
        samples = bootstrap(fn, idx_lists, a.reps, a.seed)
        report["datasets"][name] = {"n_frames": int(len(y)), "n_clusters": len(idx_lists),
                                    "ep_eligible_groups": [SCHEMES[scheme][k] for k in elig],
                                    "seed_mean_difference": summarize(fn(allidx), samples, with_p=True),
                                    "per_seed_difference_point": per_seed(allidx)}
        print(f"{name}: {len(pairs)} paired seeds, {len(idx_lists)} clusters", flush=True)
    out = a.out or "protocol_eval_pairs.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--pairs", nargs="*", default=None, help="run:compare ... (seed-averaged paired mode)")
    ap.add_argument("--eval", default="eval")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--compare_eval", default="eval")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.pairs:
        return main_pairs(a)
    if not a.run:
        ap.error("--run (or --pairs) is required")
    cfg = load_config(os.path.join(a.run, "config.yaml"))
    fm = fairrel_module(cfg)
    ed = os.path.join(a.run, a.eval)
    val = load_scores(os.path.join(ed, "val_scores.npz"))
    vs = cfg["data"]["val"].get("scheme", "eight")
    vy, vp, _, _ = to_fairrel(val["label"], val["prob"], val["group"], vs)
    thr = fm.threshold_at_tpr(vp, vy, 0.90)
    thr_cmp = None
    if a.compare:
        cv = load_scores(os.path.join(a.compare, a.compare_eval, "val_scores.npz"))
        cy, cp, _, _ = to_fairrel(cv["label"], cv["prob"], cv["group"], vs)
        thr_cmp = fm.threshold_at_tpr(cp, cy, 0.90)
    report = {"run": a.run, "eval": a.eval, "compare": a.compare, "reps": a.reps,
              "fairrel_threshold": thr, "fairrel_threshold_compare": thr_cmp, "datasets": {}}
    tval = val_threshold(a.run, a.eval)
    tval_cmp = val_threshold(a.compare, a.compare_eval) if a.compare else None
    report.update(ep_val_threshold=tval, ep_val_threshold_compare=tval_cmp)
    for name, spec in cfg["eval"]["test_sets"].items():
        f = os.path.join(ed, f"{name}_scores.npz")
        if not os.path.exists(f):
            continue
        z = load_scores(f)
        y, p, g, cl = z["label"], z["prob"], z["group"], z["cluster"].astype(str)
        scheme = spec.get("scheme", "eight")
        elig = eligible_groups(y, g, cl)
        if a.compare:
            zc = load_scores(os.path.join(a.compare, a.compare_eval, f"{name}_scores.npz"))
            if not (np.array_equal(zc["label"], y) and np.array_equal(zc["cluster"].astype(str), cl)):
                raise SystemExit(f"{name}: compared runs were scored on different rows; cannot pair")
            pc = zc["prob"]

        def metrics(idx, prob):
            yy, pp, gg = y[idx], prob[idx], g[idx]
            out = {f"ding:{k}": v for k, v in ding_point(yy, pp, gg, scheme).items()}
            fy, fp, fg, _ = to_fairrel(yy, pp, gg, scheme)
            t = thr if prob is p else thr_cmp
            out.update({f"fairrel:{k}": v for k, v in fairrel_point(fm, fy, fp, fg, t).items()})
            tv = tval if prob is p else tval_cmp
            out.update({f"ep:{k}": v for k, v in endpoint_point(yy, prob[idx], gg, elig, tv).items()})
            return out

        def fn(idx):
            m = metrics(idx, p)
            if a.compare:
                mc = metrics(idx, pc)
                m = {k: m[k] - mc[k] for k in m if k in mc}
            return m

        allidx = np.arange(len(y))
        point = fn(allidx)
        idx_lists = cluster_index(cl)
        samples = bootstrap(fn, idx_lists, a.reps, a.seed)
        report["datasets"][name] = {"n_frames": int(len(y)), "n_clusters": len(idx_lists),
                                    "ep_eligible_groups": [SCHEMES[scheme][k] for k in elig],
                                    "metrics": summarize(point, samples, with_p=bool(a.compare))}
        hl = report["datasets"][name]["metrics"]
        keys = ["ding:auc", "ding:intersection_es_AUC", "ding:intersection_F_FPR", "fairrel:auc_avg", "fairrel:f_fpr",
                "fairrel:f_meo", "fairrel:worst_group_tpr"]
        print(f"{name:10s} clusters={len(idx_lists):5d} " + "  ".join(
            f"{k.split(':')[1]}={hl[k]['point']:.2f}[{hl[k]['lo']:.2f},{hl[k]['hi']:.2f}]" for k in keys if k in hl), flush=True)
    out = a.out or os.path.join(ed, "protocol_eval" + ("_vs_" + os.path.basename(a.compare) if a.compare else "") + ".json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
