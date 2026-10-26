"""Mechanism analysis: do CESel B0's per-group cue deficits track its per-group error rates? (CPU, saved scores.)

  OMP_NUM_THREADS=8 ~/.conda/envs/aiface/bin/python mechanism.py [--B 1000]

Per-image cue utilities u_ik are recomputed exactly as in CESEL/scripts/ces_cue_starvation.py (cue_stats.utilities:
training margin, training replacement rule, EMA reference from runs/ffpp_B0_s7/best.pt; the replacement baseline for
fakes stays the Xu-label group of training, only the AGGREGATION groups change). For every test set (FF++ test, DFDC,
Celeb-DF, DFD) and grouping (gender x race with Xu labels = the starvation table's groups; gender x race with AI-Face
cleaner labels; AI-Face age; ITA bin), per eligible group (>= 3 real videos for the real side, >= 3 fake videos for
the fake side):
  real side   deficit_gk = mean_{reals in set} u_k - mean_{reals in g} u_k      vs  dFPR_g = FPR_g - FPR_set
  fake side   deficit_gk = mean_{fakes in set} u_k - mean_{fakes in g} u_k      vs  dFNR_g = FNR_g - FNR_set
  (both within-set centred, so set-level differences cannot drive the correlation). Cues: tex, freq, bound, sem and
  "total" (sum over cues). Statistic: Spearman rho over all (set, group) points pooled across the 4 test sets.
  Mechanical reference: the same correlation with the group's mean training margin deficit (pool - group mean of
  m(s, y)); u sums to a margin difference, so part of any deficit-error link is arithmetic, not mechanism. The
  partial Spearman of (cue deficit, error) given the margin deficit asks whether a cue adds anything beyond that.
  CI: video-cluster bootstrap within each set (B replicates, same resample for deficits and error rates), percentile.
Threshold: B0's stored A1 (eval/a1.json; primary) and t05 (logit >= 0).
Writes results/mechanism_b0.csv (correlations), results/mechanism_b0_points.csv (per group points).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
V = f"{R}/agents/DEMO_AXES"
C = f"{R}/agents/CESEL"
RUN = f"{C}/runs/ffpp_B0_s7"
sys.path.insert(0, C)
sys.path.insert(0, f"{C}/scripts")
sys.path.insert(0, f"{V}/code")
import cue_stats as CS  # noqa: E402
from cfes.config import load_config  # noqa: E402
AGE_AIF = {0: "<25", 1: "25-44", 2: "45-64", 3: "65+"}          # = rescore.py (not imported: its cfes package differs)
ITA4 = {0: "ITA-Q1 (darkest)", 1: "ITA-Q2", 2: "ITA-Q3", 3: "ITA-Q4 (lightest)"}


def _map(x, d):
    return np.array([d.get(int(v)) if np.isfinite(v) else None for v in np.asarray(x, dtype=float)], dtype=object)

CUES = ["tex", "freq", "bound", "sem"]
TEST = {"ffpp_test": "eight", "dfdc": "eight", "celebdf": "six", "dfd": "six"}


def spearman_rows(X, Y):
    """row-wise Spearman between X (B, n) and Y (B, n), NaN-free columns assumed."""
    rx = np.apply_along_axis(rankdata, 1, X)
    ry = np.apply_along_axis(rankdata, 1, Y)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def partial_spearman_rows(X, Y, Z):
    """row-wise partial Spearman of X and Y given Z (ranks residualised linearly on the rank of Z)."""
    r = [np.apply_along_axis(rankdata, 1, A) for A in (X, Y, Z)]
    r = [q - q.mean(1, keepdims=True) for q in r]
    rz = r[2]
    den = (rz ** 2).sum(1, keepdims=True)
    ex = r[0] - (r[0] * rz).sum(1, keepdims=True) / den * rz
    ey = r[1] - (r[1] * rz).sum(1, keepdims=True) / den * rz
    return (ex * ey).sum(1) / np.sqrt((ex ** 2).sum(1) * (ey ** 2).sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(f"{RUN}/config.yaml")
    ces = cfg.get("ces", {})
    ref = CS.load_ref(f"{RUN}/best.pt")
    thr = {"A1": float(json.load(open(f"{RUN}/eval/a1.json"))["t"]), "t05": 0.0}
    rng = np.random.default_rng(7)
    data = {}
    for s, sch in TEST.items():
        d = np.load(f"{RUN}/eval/{s}_scores.npz", allow_pickle=True)
        L = pd.read_csv(f"{V}/labels/labels_{s}.csv", low_memory=False)
        assert all(os.path.basename(p) == os.path.basename(q) for p, q in zip(d["img_path"].astype(str), L.img_path))
        z, sk, y, g = d["logit"].astype(np.float64), d["s_k"].astype(np.float64), d["label"].astype(int), d["group"].astype(int)
        u = CS.utilities(z, sk, y, CS.to_train_groups(g, sch, "eight"), ref, ces)
        u = np.c_[u, u.sum(1)]
        mg = CS.margin(z, y, ces)
        six = sch == "six"
        gm = np.where(L.cl_male.values == 1, "male", "nonmale")
        rc = L.cl_race.values.astype(object)
        if six:
            rc = np.where(rc == "asian", "others", rc)
        gx = np.where(L.xu_male.values == 1, "male", "nonmale")
        groupings = {"gender_x_race (Xu)": np.array([f"{p},{q}" for p, q in zip(gx, L.xu_race.values)], dtype=object),
                     "gender_x_race (AI-Face clean)": np.array([f"{p},{q}" for p, q in zip(gm, rc)], dtype=object),
                     "age (AI-Face)": _map(L.age4_aif, AGE_AIF), "skin (ITA4)": _map(L.ita4, ITA4)}
        rv = L.cl_race_vid.values.astype(object)
        if six:
            rv = np.where(rv == "asian", "others", rv)
        gv = np.where(L.cl_male_vid.values == 1, "male", "nonmale")
        groupings["gender_x_race (AI-Face clean, video majority)"] = np.array([f"{p},{q}" for p, q in zip(gv, rv)], dtype=object)
        uv, ci = np.unique(L.video.astype(str).values, return_inverse=True)
        data[s] = dict(z=z, y=y, u=u, mg=mg, groupings=groupings, ci=ci, K=len(uv))
    out, pts = [], []
    cols = CUES + ["total"]
    for gname in ["gender_x_race (Xu)", "gender_x_race (AI-Face clean)", "gender_x_race (AI-Face clean, video majority)",
                  "age (AI-Face)", "skin (ITA4)"]:
        for tn, t in thr.items():
            for side, cls in (("real", 0), ("fake", 1)):
                Xpt, Ypt, Mpt, Xb, Yb, Mb = [], [], [], [], [], []
                for s, D in data.items():
                    y, u, mg, ci, K = D["y"], D["u"], D["mg"], D["ci"], D["K"]
                    grp = D["groupings"][gname]
                    err = (D["z"] >= t) if cls == 0 else (D["z"] < t)          # FP for reals, FN for fakes
                    m_c = y == cls
                    names = sorted({x for x in grp[m_c] if isinstance(x, str)})
                    W = rng.multinomial(K, np.full(K, 1.0 / K), size=a.B).astype(np.float64)
                    # per-cluster sums for the class pool
                    def csum(mask, vals):
                        S = np.zeros((K,) + vals.shape[1:])
                        np.add.at(S, ci[mask], vals[mask])
                        return S
                    one = np.ones(len(y))
                    Pn, Pu, Pe, Pm = csum(m_c, one), csum(m_c, u), csum(m_c, err.astype(float)), csum(m_c, mg)
                    for n in names:
                        mm = m_c & (grp == n)
                        if len(set(ci[mm])) < 3:
                            continue
                        Gn, Gu, Ge, Gm = csum(mm, one), csum(mm, u), csum(mm, err.astype(float)), csum(mm, mg)
                        dpt = Pu.sum(0) / Pn.sum() - Gu.sum(0) / Gn.sum()
                        ept = Ge.sum() / Gn.sum() - Pe.sum() / Pn.sum()
                        mpt = Pm.sum() / Pn.sum() - Gm.sum() / Gn.sum()
                        with np.errstate(invalid="ignore", divide="ignore"):
                            db = (W @ Pu) / (W @ Pn)[:, None] - (W @ Gu) / (W @ Gn)[:, None]
                            eb = (W @ Ge) / (W @ Gn) - (W @ Pe) / (W @ Pn)
                            mb = (W @ Pm) / (W @ Pn) - (W @ Gm) / (W @ Gn)
                        Xpt.append(dpt); Ypt.append(ept); Mpt.append(mpt); Xb.append(db); Yb.append(eb); Mb.append(mb)
                        pts.append({"grouping": gname, "threshold": tn, "side": side, "set": s, "group": n,
                                    "n": int(mm.sum()), "videos": int(len(set(ci[mm]))), "d_err_pp": 100 * ept,
                                    "margin_deficit": mpt, **{f"deficit_{c}": dpt[k] for k, c in enumerate(cols)}})
                if len(Xpt) < 4:
                    continue
                Xpt, Ypt, Mpt = np.array(Xpt), np.array(Ypt), np.array(Mpt)
                Xb, Yb, Mb = np.stack(Xb, 1), np.stack(Yb, 1), np.stack(Mb, 1)   # (B, n, cues) / (B, n)
                okb = np.isfinite(Yb).all(1) & np.isfinite(Xb).all((1, 2)) & np.isfinite(Mb).all(1)
                for k, c in enumerate(cols + ["margin"]):
                    xp = Mpt if c == "margin" else Xpt[:, k]
                    xb = Mb if c == "margin" else Xb[:, :, k]
                    rho = float(spearman_rows(xp[None], Ypt[None])[0])
                    rb = spearman_rows(xb[okb], Yb[okb])
                    row = {"grouping": gname, "threshold": tn, "side": side, "cue": c, "n_points": len(Ypt),
                           "spearman": rho, "ci_lo": float(np.nanpercentile(rb, 2.5)), "ci_hi": float(np.nanpercentile(rb, 97.5)),
                           "B_valid": int(okb.sum())}
                    if c != "margin":                    # does the cue deficit carry information beyond the margin?
                        pr = float(partial_spearman_rows(xp[None], Ypt[None], Mpt[None])[0])
                        pb = partial_spearman_rows(xb[okb], Yb[okb], Mb[okb])
                        row.update(partial_given_margin=pr, pci_lo=float(np.nanpercentile(pb, 2.5)),
                                   pci_hi=float(np.nanpercentile(pb, 97.5)))
                    out.append(row)
    pd.DataFrame(out).to_csv(f"{V}/results/mechanism_b0.csv", index=False, float_format="%.4f")
    pd.DataFrame(pts).to_csv(f"{V}/results/mechanism_b0_points.csv", index=False, float_format="%.5g")
    print(pd.DataFrame(out).query("threshold == 'A1'").to_string())


if __name__ == "__main__":
    main()
