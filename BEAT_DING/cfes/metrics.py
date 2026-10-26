"""Evaluation metrics.

Ding et al. (CVPR'26) Eq. 8-10, computed at their fixed 0.5 threshold on P(fake), separately for the gender,
race and intersection partitions (same code path as DINGREPRO_LIN/code/Ding_mod/eval_ding_protocol.py, which
reproduces their released formulas):
  F_FPR  = sum_g |FPR_g - FPR_all|
  F_DP   = max_k (max_g rate_k,g - min_g rate_k,g), k in {pred fake, pred real}
  es-AUC = AUC_all / (1 + sum_g |AUC_g - AUC_all|)
Also reported (definitions are ours, stated here, not claimed equal to any paper's):
  F_MEO_maxdev = max_g max(|FPR_g - FPR_all|, |TPR_g - TPR_all|)   (deviation from the pooled rates; ours.
                 Named F_MEO in result files written before 16 Oct 18:30)
  F_MEO_gap    = max(max_g FPR_g - min_g FPR_g, max_g TPR_g - min_g TPR_g)   (largest between-group gap; the
                 form FairReL's released fairrel/metrics.py uses; FairForensics gives no definition)
  F_OAE  = max_g ACC_g - min_g ACC_g   (overall accuracy equality, as listed in FairForensics' tables)
  per-group AUC / FPR / TPR at 0.5 and at a validation-calibrated threshold (TPR_val = target)
  per-method AUC (each fake method vs all reals) for cross-generator evaluation.
Added 16 Oct (evaluation lock, EVAL_PROTOCOL.md), all over the groups that have both classes:
  macro_group_auc = mean_g AUC_g        group_auc_gap = max_g AUC_g - min_g AUC_g
  fpr_gap = max_g FPR_g - min_g FPR_g   tpr_gap = max_g TPR_g - min_g TPR_g      (at the block's threshold)
  groups_skipped_single_class = groups present in the data but holding only reals or only fakes
  evaluate_scores(..., val_threshold=t) also returns pooled fpr@val / tpr@val (external-real FPR when the test
  set's reals come from a dataset not used in training).
Groups with a single class are skipped for AUC/FPR/TPR (Ding's code does the same). FairReL's released code does
NOT skip them for FPR/TPR (an all-real group still enters its f_fpr and the FPR side of f_meo), so F_MEO_gap here
equals FairReL's f_meo only when every group has both classes (not on Celeb-DF or DFD); FairReL-protocol numbers
must come from fairrel.metrics via scripts/protocol_eval.py, never from this block.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data import SCHEMES


def _rates(y, p, thr):
    pred = p >= thr
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    fpr = float((pred & (y == 0)).sum() / n0) if n0 else float("nan")
    tpr = float((pred & (y == 1)).sum() / n1) if n1 else float("nan")
    return fpr, tpr, float(pred.mean())


def group_stats(y, p, thr):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    fpr, tpr, pr = _rates(y, p, thr)
    return dict(n=int(len(y)), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
                auc=float(roc_auc_score(y, p)), FPR=fpr, TPR=tpr, PR=pr, NR=1 - pr,
                ACC=float(((p >= thr) == (y == 1)).mean()))


def partitions(scheme):
    names = SCHEMES[scheme]
    gender = {s: [i for i, n in enumerate(names) if n.split(",")[0] == s] for s in ("male", "nonmale")}
    race = {r: [i for i, n in enumerate(names) if n.split(",")[1] == r] for r in ("asian", "white", "black", "others")}
    inter = {n: [i] for i, n in enumerate(names)}
    return {"gender": gender, "race": {k: v for k, v in race.items() if v}, "intersection": inter}


def fairness_block(y, p, g, scheme, thr):
    overall = group_stats(y, p, thr)
    out = {"overall": overall}
    for level, part in partitions(scheme).items():
        stats, skipped = {}, []
        for name, ids in part.items():
            m = np.isin(g, ids)
            s = group_stats(y[m], p[m], thr)
            if s:
                stats[name] = s
            elif m.any():
                skipped.append(name)
        v = list(stats.values())
        res = {"groups": stats, "groups_skipped_single_class": skipped}
        if len(v) >= 2:
            res.update(
                F_FPR=100 * sum(abs(s["FPR"] - overall["FPR"]) for s in v),
                F_DP=100 * max(max(s["PR"] for s in v) - min(s["PR"] for s in v),
                               max(s["NR"] for s in v) - min(s["NR"] for s in v)),
                es_AUC=100 * overall["auc"] / (1 + sum(abs(s["auc"] - overall["auc"]) for s in v)),
                F_MEO_maxdev=100 * max(max(abs(s["FPR"] - overall["FPR"]), abs(s["TPR"] - overall["TPR"])) for s in v),
                F_MEO_gap=100 * max(max(s["FPR"] for s in v) - min(s["FPR"] for s in v),
                                    max(s["TPR"] for s in v) - min(s["TPR"] for s in v)),
                F_OAE=100 * (max(s["ACC"] for s in v) - min(s["ACC"] for s in v)),
                worst_group_auc=100 * min(s["auc"] for s in v),
                worst_group_tpr=100 * min(s["TPR"] for s in v),
                max_group_fpr=100 * max(s["FPR"] for s in v),
                macro_group_auc=100 * float(np.mean([s["auc"] for s in v])),
                group_auc_gap=100 * (max(s["auc"] for s in v) - min(s["auc"] for s in v)),
                fpr_gap=100 * (max(s["FPR"] for s in v) - min(s["FPR"] for s in v)),
                tpr_gap=100 * (max(s["TPR"] for s in v) - min(s["TPR"] for s in v)),
            )
        out[level] = res
    return out


def threshold_at_tpr(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.searchsorted(tpr, target, side="left"))
    return float(thr[min(i, len(thr) - 1)])


def evaluate_scores(y, p, g, scheme, methods=None, val_threshold=None, thr_scores=None):
    """thr_scores (Codex review F4, 25 Oct; copied from agents/CESEL/cfes/metrics.py): the scores the validation threshold
    is applied to, in the SAME units as val_threshold (raw logits, so one fixed logit threshold means the same thing on
    every test set). Default p (old behaviour). AUC / ding@0.5 always use p (rank_prob(logit) under eval.rank_metrics);
    group AUCs inside fair@val are rank-invariant, so they equal the p-based ones."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(g)
    res = {"n": int(len(y)), "auc": 100 * float(roc_auc_score(y, p)),
           "acc@0.5": 100 * float(((p >= 0.5) == (y == 1)).mean()),
           "ding@0.5": fairness_block(y, p, g, scheme, 0.5)}
    if val_threshold is not None:
        q = p if thr_scores is None else np.asarray(thr_scores, dtype=np.float64)
        res["threshold_val"] = val_threshold
        res["threshold_val_units"] = "prob" if thr_scores is None else "logit"
        res["acc@val"] = 100 * float(((q >= val_threshold) == (y == 1)).mean())
        fv, tv, _ = _rates(y, q, val_threshold)
        res["fpr@val"], res["tpr@val"] = 100 * fv, 100 * tv
        res["fair@val"] = fairness_block(y, q, g, scheme, val_threshold)
    if methods is not None:
        methods = np.asarray(methods)
        real = y == 0
        per = {}
        for mth in sorted(set(methods[y == 1])):
            sel = real | ((y == 1) & (methods == mth))
            per[mth] = {"n_fake": int(((y == 1) & (methods == mth)).sum()),
                        "auc": 100 * float(roc_auc_score(y[sel], p[sel]))}
        res["per_method"] = per
    return res


def headline(res):
    """Flat row matching Ding's Table 2 column order (0.5 threshold)."""
    d = res["ding@0.5"]
    row = {"AUC": res["auc"]}
    for level in ("gender", "race", "intersection"):
        for k in ("F_FPR", "F_DP", "es_AUC"):
            row[f"{level}_{k}"] = d[level].get(k, float("nan"))
    return row


def rank_prob(z):
    """Strictly monotone, non-saturating map of logits to (0, 1) with rank_prob(0) = 0.5:
    0.5 + z / (2 (max|z| + 1)). Every threshold-0.5 decision equals sigmoid(z) >= 0.5 (i.e. z >= 0), and AUCs are
    exact (no ties from sigmoid saturation, which float32 hits at |z| ~ 17 and float64 at |z| ~ 37). It is NOT a
    probability; use it only for metric computation (CFES revision study, eval.rank_metrics)."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 + z / (2.0 * (np.abs(z).max() + 1.0))
