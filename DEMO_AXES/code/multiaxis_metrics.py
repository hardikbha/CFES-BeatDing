"""Multi-axis fairness metrics with FPR/TPR-specific eligibility and a minimum-support rule.

Definitions are those of agents/CFES/cfes/metrics.py (Ding et al. CVPR'26 Eq. 8-10 + the 16-Oct additions), computed
on an arbitrary partition (gender, race, age bin, ITA bin, gender x race, ...). Decision: fake iff logit >= t.

  per group g: n_real/n_fake frames, real/fake clusters (videos; persons on FF2), real/fake persons (FF++: YouTube
  id / target id; CDF: celebrity id; DFD: first actor; DFDC: video), FPR_g, TPR_g, PR_g, AUC_g
  eligibility (declared in decisions/2026-10-26-final-model-spec.md; Codex cross-check gap 1):
      FPR-eligible   >= MIN_CLUSTERS real clusters in g      (FPR needs only reals)
      TPR-eligible   >= MIN_CLUSTERS fake clusters in g
      both-eligible  FPR- and TPR-eligible (needed for AUC_g, PR_g-based F_DP and es-AUC)
      MIN_CLUSTERS = 3 on test sets (video datasets: real videos; FF2: persons); 2 for validation selection
  F_FPR      = 100 * sum_{g FPR-elig} |FPR_g - FPR|                  (Ding Eq. 8 form)
  fpr_gap    = 100 * (max - min) FPR_g over FPR-eligible groups       (H2/H3 endpoint form: "max-gap FPR")
  tpr_gap    = 100 * (max - min) TPR_g over TPR-eligible groups
  EO_gap     = max(fpr_gap, tpr_gap)                                  (= F_MEO_gap of metrics.py)
  EO_maxdev  = 100 * max_g max(|FPR_g - FPR| [FPR-elig], |TPR_g - TPR| [TPR-elig])  (= F_MEO_maxdev)
  F_DP       = 100 * (max - min) PR_g over both-eligible groups (pred-fake rate; the pred-real term is identical)
  es_AUC     = 100 * AUC / (1 + sum_{g both-elig} |AUC_g - AUC|)
  worst_fpr  = max FPR_g (FPR-elig);  worst_tpr = min TPR_g (TPR-elig)
Metrics need >= 2 eligible groups, else NaN. `ding_compat` = the historical metrics.py rule (groups holding both
classes, no minimum support) for the gender/race/intersection reproduction check.

Cluster bootstrap (count-based; eligibility FIXED from the full sample so the estimand does not change across
replicates): clusters resampled with replacement; a replicate group with 0 reals (fakes) is dropped from that
replicate's FPR (TPR) terms.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

MIN_CLUSTERS_TEST = 3
MIN_CLUSTERS_VAL = 2


def _auc(y, z):
    return float(roc_auc_score(y, z)) if len(np.unique(y)) == 2 else float("nan")


def group_table(y, z, t, groups, cluster, person=None, min_clusters=MIN_CLUSTERS_TEST):
    """groups: object array of group names, None/NaN = unlabeled (excluded from every group, kept in pooled rates)."""
    y, z = np.asarray(y), np.asarray(z, dtype=np.float64)
    pred = z >= t
    g = np.asarray(groups, dtype=object)
    lab = np.array([isinstance(v, str) for v in g])
    rows = {}
    for name in sorted(set(g[lab])):
        m = lab & (g == name)
        r, f = m & (y == 0), m & (y == 1)
        d = dict(n_real=int(r.sum()), n_fake=int(f.sum()),
                 real_clusters=int(len(set(cluster[r]))), fake_clusters=int(len(set(cluster[f]))),
                 real_persons=int(len(set(person[r]))) if person is not None else None,
                 fake_persons=int(len(set(person[f]))) if person is not None else None,
                 FPR=float(pred[r].mean()) if r.any() else float("nan"),
                 TPR=float(pred[f].mean()) if f.any() else float("nan"),
                 PR=float(pred[m].mean()), AUC=_auc(y[m], z[m]))
        d["fpr_elig"] = d["real_clusters"] >= min_clusters
        d["tpr_elig"] = d["fake_clusters"] >= min_clusters
        d["both_elig"] = d["fpr_elig"] and d["tpr_elig"]
        d["both_classes"] = d["n_real"] > 0 and d["n_fake"] > 0
        rows[name] = d
    return rows


def summarize(rows, FPR, TPR, AUC, rule="min_support"):
    if rule == "min_support":
        ef = [d for d in rows.values() if d["fpr_elig"]]
        et = [d for d in rows.values() if d["tpr_elig"]]
        eb = [d for d in rows.values() if d["both_elig"]]
    else:                                                        # ding_compat
        ef = et = eb = [d for d in rows.values() if d["both_classes"]]
    o = {"n_groups": len(rows), "n_fpr_elig": len(ef), "n_tpr_elig": len(et), "n_both_elig": len(eb)}
    nan = float("nan")
    o["F_FPR"] = 100 * sum(abs(d["FPR"] - FPR) for d in ef) if len(ef) >= 2 else nan
    o["fpr_gap"] = 100 * (max(d["FPR"] for d in ef) - min(d["FPR"] for d in ef)) if len(ef) >= 2 else nan
    o["worst_fpr"] = 100 * max(d["FPR"] for d in ef) if len(ef) >= 2 else nan
    o["tpr_gap"] = 100 * (max(d["TPR"] for d in et) - min(d["TPR"] for d in et)) if len(et) >= 2 else nan
    o["worst_tpr"] = 100 * min(d["TPR"] for d in et) if len(et) >= 2 else nan
    o["EO_gap"] = np.nanmax([o["fpr_gap"], o["tpr_gap"]]) if np.isfinite([o["fpr_gap"], o["tpr_gap"]]).any() else nan
    devs = [abs(d["FPR"] - FPR) for d in ef] + [abs(d["TPR"] - TPR) for d in et]
    o["EO_maxdev"] = 100 * max(devs) if (len(ef) >= 2 or len(et) >= 2) else nan
    o["F_DP"] = 100 * (max(d["PR"] for d in eb) - min(d["PR"] for d in eb)) if len(eb) >= 2 else nan
    o["es_AUC"] = 100 * AUC / (1 + sum(abs(d["AUC"] - AUC) for d in eb)) if len(eb) >= 2 else nan
    return o


def axis_metrics(y, z, t, groups, cluster, person=None, min_clusters=MIN_CLUSTERS_TEST):
    y, z = np.asarray(y), np.asarray(z, dtype=np.float64)
    pred = z >= t
    FPR, TPR, AUC = float(pred[y == 0].mean()), float(pred[y == 1].mean()), _auc(y, z)
    rows = group_table(y, z, t, groups, cluster, person, min_clusters)
    out = {"pooled": {"AUC": 100 * AUC, "FPR": 100 * FPR, "TPR": 100 * TPR, "n": int(len(y)),
                      "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())},
           "groups": rows, "min_support": summarize(rows, FPR, TPR, AUC, "min_support"),
           "ding_compat": summarize(rows, FPR, TPR, AUC, "ding_compat")}
    return out


# ------------------------------------------------------------------------------------------------ bootstrap
def cluster_counts(y, pred, groups, cl_idx, K, names):
    """(K, G, 4): per cluster and group [reals, fakes, FP, TP]."""
    g = np.asarray(groups, dtype=object)
    gi = np.full(len(g), -1)
    for j, n in enumerate(names):
        gi[g == n] = j
    C = np.zeros((K, len(names), 4))
    ok = gi >= 0
    for col, m in enumerate(((y == 0), (y == 1), (y == 0) & pred, (y == 1) & pred)):
        mm = m & ok
        np.add.at(C[:, :, col], (cl_idx[mm], gi[mm]), 1)
    P = np.zeros((K, 4))                                           # pooled (incl. unlabeled rows)
    for col, m in enumerate(((y == 0), (y == 1), (y == 0) & pred, (y == 1) & pred)):
        np.add.at(P[:, col], cl_idx[m], 1)
    return C, P


def boot_rate_metrics(C, P, W, fpr_elig, tpr_elig):
    """W: (B, K) cluster multiplicities. Returns dict of (B,) arrays: FPR, TPR, F_FPR, fpr_gap, tpr_gap, EO_gap,
    plus per-group FPR/TPR (B, G)."""
    G = W @ C.reshape(C.shape[0], -1)
    G = G.reshape(W.shape[0], C.shape[1], 4)
    Q = W @ P
    with np.errstate(invalid="ignore", divide="ignore"):
        FPR, TPR = Q[:, 2] / Q[:, 0], Q[:, 3] / Q[:, 1]
        fg, tg = G[:, :, 2] / G[:, :, 0], G[:, :, 3] / G[:, :, 1]
    fg = np.where(fpr_elig[None], fg, np.nan)
    tg = np.where(tpr_elig[None], tg, np.nan)
    nf, nt = np.isfinite(fg).sum(1), np.isfinite(tg).sum(1)
    o = {"FPR": 100 * FPR, "TPR": 100 * TPR, "group_FPR": 100 * fg, "group_TPR": 100 * tg}
    if C.shape[1] == 0:                                            # axis without any labelled group
        nanv = np.full(W.shape[0], np.nan)
        o.update(F_FPR=nanv, fpr_gap=nanv, tpr_gap=nanv, EO_gap=nanv)
        return o
    with np.errstate(invalid="ignore", all="ignore"):
        o["F_FPR"] = np.where(nf >= 2, 100 * np.nansum(np.abs(fg - FPR[:, None]), 1), np.nan)
        o["fpr_gap"] = np.where(nf >= 2, 100 * (np.nanmax(np.where(np.isfinite(fg), fg, -np.inf), 1)
                                               - np.nanmin(np.where(np.isfinite(fg), fg, np.inf), 1)), np.nan)
        o["tpr_gap"] = np.where(nt >= 2, 100 * (np.nanmax(np.where(np.isfinite(tg), tg, -np.inf), 1)
                                               - np.nanmin(np.where(np.isfinite(tg), tg, np.inf), 1)), np.nan)
    o["EO_gap"] = np.fmax(o["fpr_gap"], o["tpr_gap"])
    return o


def multinomial_weights(K, B, rng):
    return rng.multinomial(K, np.full(K, 1.0 / K), size=B).astype(np.float64)
