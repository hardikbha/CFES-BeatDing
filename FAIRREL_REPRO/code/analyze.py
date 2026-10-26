"""FAIRREL_REPRO single TEST analysis (PREREG sections 4-6). Reads runs/{FAIRREL,CTRL}_s{7,17,27}/scores_best.npz,
checks that point estimates reproduce each run's test_best.json, then: per-run metrics, per-arm means, paired
two-level (cluster x seed) bootstrap FAIRREL - CTRL, estimability, Gate G1 vs paper.
Metrics: official fairrel/metrics.py (imported read-only).
usage: pulid python analyze.py [reps] [nproc]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FR = "/home/sachin.chaudhary/ICLR_2027/related_work/FairReL"
sys.path.insert(0, FR)
from fairrel.metrics import fairness_report, threshold_at_tpr  # noqa: E402
from fairrel.data.subgroups import SUBGROUPS                   # noqa: E402

A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO"
EM = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan"
ARMS, SEEDS = ["FAIRREL", "CTRL"], [7, 17, 27]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 24
MIN_CL = 5
PAPER = {"ffpp": dict(auc=96.14, auc_avg=96.54, f_fpr=10.97, f_meo=7.41),
         "celebdf": dict(auc=80.64, auc_avg=81.26, f_fpr=10.12, f_meo=7.03),
         "dfd": dict(auc=86.13, auc_avg=86.78, f_fpr=13.55, f_meo=8.98),
         "dfdc": dict(auc=66.34, auc_avg=67.03, f_fpr=38.36, f_meo=30.02)}
PAPER_EFFORT = {"ffpp": 94.83, "celebdf": 78.52, "dfd": 84.22, "dfdc": 65.08}
G1_TOL = {"ffpp": 2.0, "celebdf": 3.0, "dfd": 3.0, "dfdc": 3.0}
KEYS = ["auc", "auc_avg", "auc_gap", "f_fpr", "f_meo", "fpr_overall", "tpr_overall",
        "auc_avg_est", "auc_gap_est", "f_fpr_est", "f_meo_est", "auc_worst", "auc_worst_est"]
# dataset name -> (score key in npz, csv for clusters, optional row-weights)
DSETS = ["ffpp", "ffpp_clean", "celebdf", "dfd", "dfdc", "dfdc_asshipped"]


def flat(full, est):
    o = {k: full[k] for k in ["auc", "auc_avg", "auc_gap", "f_fpr", "f_meo", "fpr_overall", "tpr_overall"]}
    pa = full["per_subgroup_auc"]
    o["auc_worst"] = min(pa.values()) if pa else np.nan
    for sg in SUBGROUPS:
        o[f"auc_{sg}"] = pa.get(sg, np.nan)
        o[f"fpr_{sg}"] = full["per_subgroup_fpr"].get(sg, np.nan)
    if est is None:
        o.update(auc_avg_est=np.nan, auc_gap_est=np.nan, f_fpr_est=np.nan, f_meo_est=np.nan, auc_worst_est=np.nan)
    else:
        pe = est["per_subgroup_auc"]
        o.update(auc_avg_est=est["auc_avg"], auc_gap_est=est["auc_gap"], f_fpr_est=est["f_fpr"],
                 f_meo_est=est["f_meo"], auc_worst_est=min(pe.values()) if pe else np.nan)
    return o


def evaluate(s, y, g, thr, em):
    if len(np.unique(y)) < 2:
        return None
    full = fairness_report(s, y, g, thr)
    est = fairness_report(s[em], y[em], g[em], thr) if em.any() and len(np.unique(y[em])) == 2 else None
    return flat(full, est)


G = {}


def load():
    runs = {}
    repro = []
    for a in ARMS:
        for sd in SEEDS:
            d = f"{A}/runs/{a}_s{sd}"
            z = np.load(f"{d}/scores_best.npz", allow_pickle=True)
            thr = threshold_at_tpr(z["val"], z["val_y"], 0.90)
            tj = json.load(open(f"{d}/test_best.json"))
            off = json.load(open(f"{d}/test_best_official.json"))
            for ds in ["ffpp", "celebdf", "dfd", "dfdc"]:
                mine = fairness_report(z[ds], z[f"{ds}_y"], z[f"{ds}_g"], thr)
                dd = max(abs(mine[k] - tj[ds][k]) for k in ["auc", "auc_avg", "f_fpr", "f_meo"])
                do = max(abs(mine[k] - off[ds][k]) for k in ["auc", "auc_avg", "f_fpr", "f_meo"])
                repro.append(dict(arm=a, seed=sd, ds=ds, thr=thr, thr_json=tj["threshold"], thr_official=off["threshold"],
                                  max_diff_vs_score_eval=dd, max_diff_vs_official=do))
            runs[(a, sd)] = dict(z=z, thr=thr, epoch=tj.get("checkpoint_epoch"))
    return runs, pd.DataFrame(repro)


def build(runs):
    for ds in DSETS:
        base = {"ffpp_clean": "ffpp", "dfdc_asshipped": "dfdc"}.get(ds, ds)
        csvname = {"ffpp": "ffpp_test", "ffpp_clean": "ffpp_test_clean"}.get(ds, f"{base}_test")
        df = pd.read_csv(f"{A}/data/{csvname}.csv")
        z0 = runs[(ARMS[0], SEEDS[0])]["z"]
        paths = z0[f"{base}_path"].astype(str)
        pos = {p: i for i, p in enumerate(paths)}
        if ds == "dfdc_asshipped":   # duplicates counted twice, as shipped in Lin's dfdc/test.csv (EM parse)
            em = pd.read_csv(f"{EM}/data/dfdc_test.csv")
            nm2i = {os.path.basename(p): i for i, p in enumerate(paths)}
            idx = np.array([nm2i[os.path.basename(p)] for p in em["path"]])
            clusters = em["cluster"].astype(str).values
        else:
            idx = np.array([pos[p] for p in df["path"]])
            clusters = df["cluster"].astype(str).values
        assert len(idx) == len(df) or ds == "dfdc_asshipped"
        y = z0[f"{base}_y"].astype(int)[idx]
        g = z0[f"{base}_g"].astype(int)[idx]
        if ds != "dfdc_asshipped":
            assert (y == df["label"].values).all()
            assert all(SUBGROUPS[k] == s for k, s in zip(g, df["subgroup"].values))
        sc = {}
        for k, r in runs.items():
            zz = r["z"]
            assert (zz[f"{base}_path"].astype(str) == paths).all()
            sc[k] = zz[base].astype(np.float64)[idx]
        cnt, est_ids = [], []
        for gid, sg in enumerate(SUBGROUPS):
            m = g == gid
            if not m.any():
                continue
            rc, fc = len(set(clusters[m & (y == 0)])), len(set(clusters[m & (y == 1)]))
            e = rc >= MIN_CL and fc >= MIN_CL
            cnt.append(dict(dataset=ds, subgroup=sg, real_frames=int((m & (y == 0)).sum()),
                            fake_frames=int((m & (y == 1)).sum()), real_clusters=rc, fake_clusters=fc, estimable=e))
            if e:
                est_ids.append(gid)
        u, inv = np.unique(clusters, return_inverse=True)
        G[ds] = dict(y=y, g=g, sc=sc, em=np.isin(g, est_ids), members=[np.where(inv == c)[0] for c in range(len(u))],
                     counts=cnt, thr={k: r["thr"] for k, r in runs.items()})


def one_rep(args):
    ds, r = args
    D = G[ds]
    rng = np.random.default_rng([2026, DSETS.index(ds), r])
    sidx = rng.integers(0, 3, 3)
    c = rng.integers(0, len(D["members"]), len(D["members"]))
    idx = np.concatenate([D["members"][i] for i in c])
    y, g, em = D["y"][idx], D["g"][idx], D["em"][idx]
    out, cache = {}, {}
    for a in ARMS:
        vals = []
        for i in sidx:
            k = (a, SEEDS[i])
            if k not in cache:
                cache[k] = evaluate(D["sc"][k][idx], y, g, D["thr"][k], em)
            vals.append(cache[k])
        if any(v is None for v in vals):
            out[a] = None
        else:
            out[a] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    return out


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return np.nan, np.nan, len(x)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi), len(x)


def main():
    t0 = time.time()
    runs, repro = load()
    repro.to_csv(f"{A}/results/repro_check.csv", index=False)
    print(repro.to_string())
    assert (repro.max_diff_vs_score_eval < 1e-6).all() and (repro.max_diff_vs_official < 1e-3).all(), "ABORT repro"
    build(runs)
    pd.DataFrame([c for ds in DSETS for c in G[ds]["counts"]]).to_csv(f"{A}/results/estimability.csv", index=False)

    per_run = []
    for ds in DSETS:
        D = G[ds]
        for (a, sd) in runs:
            m = evaluate(D["sc"][(a, sd)], D["y"], D["g"], D["thr"][(a, sd)], D["em"])
            per_run.append(dict(dataset=ds, arm=a, seed=sd, threshold=D["thr"][(a, sd)],
                                best_epoch=runs[(a, sd)]["epoch"], **m))
    pr = pd.DataFrame(per_run)
    pr.to_csv(f"{A}/results/per_run_metrics.csv", index=False)

    rows = []
    with Pool(NPROC) as pool:
        for ds in DSETS:
            reps = pool.map(one_rep, [(ds, r) for r in range(REPS)], chunksize=20)
            sub = pr[pr.dataset == ds]
            for k in KEYS + [f"auc_{s}" for s in SUBGROUPS] + [f"fpr_{s}" for s in SUBGROUPS]:
                pt = {a: float(sub[sub.arm == a][k].mean()) for a in ARMS}
                sd_ = {a: float(sub[sub.arm == a][k].std(ddof=1)) for a in ARMS}
                arm_ci = {a: ci([rp[a][k] if rp[a] else np.nan for rp in reps]) for a in ARMS}
                dif = [rp["FAIRREL"][k] - rp["CTRL"][k] if rp["FAIRREL"] and rp["CTRL"] else np.nan for rp in reps]
                lo, hi, nf = ci(dif)
                rows.append(dict(dataset=ds, metric=k,
                                 FAIRREL=pt["FAIRREL"], FAIRREL_sd=sd_["FAIRREL"], FAIRREL_lo=arm_ci["FAIRREL"][0],
                                 FAIRREL_hi=arm_ci["FAIRREL"][1],
                                 CTRL=pt["CTRL"], CTRL_sd=sd_["CTRL"], CTRL_lo=arm_ci["CTRL"][0], CTRL_hi=arm_ci["CTRL"][1],
                                 diff=pt["FAIRREL"] - pt["CTRL"], diff_lo=lo, diff_hi=hi, n_finite=nf,
                                 n_clusters=len(G[ds]["members"])))
            print(f"{ds} done {time.time() - t0:.0f}s", flush=True)
    bt = pd.DataFrame(rows)
    bt.to_csv(f"{A}/results/bootstrap_FAIRREL_vs_CTRL.csv", index=False)

    # ---- Gate G1 ----
    g1 = []
    for ds in ["ffpp", "celebdf", "dfd", "dfdc"]:
        r = bt[(bt.dataset == ds) & (bt.metric == "auc")].iloc[0]
        g1.append(dict(dataset=ds, paper_fairrel_auc=PAPER[ds]["auc"], ours_fairrel_auc=r.FAIRREL,
                       ours_ci=f"[{r.FAIRREL_lo:.2f}, {r.FAIRREL_hi:.2f}]", gap=r.FAIRREL - PAPER[ds]["auc"],
                       tol=G1_TOL[ds], within=abs(r.FAIRREL - PAPER[ds]["auc"]) <= G1_TOL[ds],
                       paper_effort_auc=PAPER_EFFORT[ds], ours_ctrl_auc=r.CTRL))
    g1 = pd.DataFrame(g1)
    g1.to_csv(f"{A}/results/gate_G1.csv", index=False)

    # ---- text summary ----
    L = [f"FAIRREL_REPRO analysis ({REPS} reps, cluster x seed paired bootstrap; point = 3-seed mean)", ""]
    L.append("GATE G1 (FAIRREL AUC vs paper): " + ("PASS" if g1.within.all() else "FAIL"))
    L.append(g1.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    L.append("")
    for ds in DSETS:
        L.append(f"== {ds} (clusters {len(G[ds]['members'])}; estimable "
                 f"{sum(c['estimable'] for c in G[ds]['counts'])}/{len(G[ds]['counts'])}) ==")
        for k in ["auc", "auc_avg", "auc_gap", "f_fpr", "f_meo", "f_fpr_est", "f_meo_est", "auc_avg_est",
                  "auc_worst_est", "fpr_overall", "tpr_overall"]:
            r = bt[(bt.dataset == ds) & (bt.metric == k)].iloc[0]
            L.append(f"  {k:14s} FAIRREL {r.FAIRREL:7.2f} [{r.FAIRREL_lo:7.2f},{r.FAIRREL_hi:7.2f}]  "
                     f"CTRL {r.CTRL:7.2f} [{r.CTRL_lo:7.2f},{r.CTRL_hi:7.2f}]  "
                     f"F-C {r['diff']:+7.2f} [{r.diff_lo:+7.2f},{r.diff_hi:+7.2f}]"
                     + ("  *" if np.isfinite(r.diff_lo) and (r.diff_lo > 0 or r.diff_hi < 0) else ""))
    L.append("")
    L.append("per-run (dataset, arm, seed, epoch, thr, AUC, AUC_avg, F_FPR, F_MEO):")
    for _, r in pr.iterrows():
        L.append(f"  {r.dataset:15s} {r.arm:8s} s{r.seed:<3d} ep{r.best_epoch} thr {r.threshold:.4f} "
                 f"AUC {r.auc:6.2f} AUCavg {r.auc_avg:6.2f} F_FPR {r.f_fpr:6.2f} F_MEO {r.f_meo:6.2f}")
    open(f"{A}/results/SUMMARY.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
