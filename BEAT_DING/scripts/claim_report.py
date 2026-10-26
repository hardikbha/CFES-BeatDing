"""Claim report: "better detection everywhere, comparable or better fairness, and the metric is fragile".

For one or more trained BD runs (seeds), on FF++ test, DFDC, Celeb-DF and DFD:
  1. Detection: AUC per seed (mean +- sd) vs our Ding reproduction and Ding published; paired video-bootstrap 95% CI of
     delta-AUC (seed-mean ours - Ding repro), clusters = videos.
  2. Fairness at the matched operating point (each run's own val-matched threshold, Ding repro at its 0.5 rule): Ding's
     10 columns, wins vs Ding published, mean over seeds.
  3. Fragility of intersectional F_FPR (Xu labels, as the metric uses), for ours and for Ding repro:
     video-permutation null (group labels shuffled across real videos) p-value, video-bootstrap 95% CI,
     leave-one-video-out range, number of greedy video removals that halve it, and the value with the cleaner AI-Face
     labels (agents/DEMO_AXES/labels).
  ~/.conda/envs/aiface/bin/python scripts/claim_report.py --runs runs/BD0_s7 [runs/BD0_s17 runs/BD0_s27] --out results/claim_BD0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A)
from cfes import bd_metrics as BM  # noqa: E402

LAB = BM.R + "/agents/DEMO_AXES/labels"
LABF = {"ffpp_test": "labels_ffpp_test.csv", "dfdc": "labels_dfdc.csv", "celebdf": "labels_celebdf.csv", "dfd": "labels_dfd.csv"}
GROUP_ID = {"eight": {(1, "asian"): 0, (1, "white"): 1, (1, "black"): 2, (1, "others"): 3,
                      (0, "asian"): 4, (0, "white"): 5, (0, "black"): 6, (0, "others"): 7},
            "six": {(1, "white"): 0, (1, "black"): 1, (1, "others"): 2, (0, "white"): 3, (0, "black"): 4, (0, "others"): 5}}


def matched_t(run):
    th = json.load(open(f"{run}/eval/bd_eval.json"))["thresholds"]["matched"]
    return float(th["t"] if isinstance(th, dict) else th)


def ifpr(y, pred, g):
    """Intersectional F_FPR exactly as cfes.metrics: groups holding both classes; FPR_all over all reals."""
    r = y == 0
    if r.sum() == 0:
        return float("nan")
    fpr_all = pred[r].mean()
    tot, n_el = 0.0, 0
    for k in np.unique(g):
        m = g == k
        if (y[m] == 0).any() and (y[m] == 1).any():
            tot += abs(pred[m & r].mean() - fpr_all)
            n_el += 1
    return 100 * tot if n_el >= 2 else float("nan")


def video_ids(paths):
    return np.array([BM.video_of(p) for p in paths])


def cleaner_groups(paths, scheme, setkey):
    f = f"{LAB}/{LABF[setkey]}"
    if not os.path.exists(f):
        return None
    lab = pd.read_csv(f, usecols=["img_path", "cl_male", "cl_race"])
    lab["b"] = lab.img_path.map(os.path.basename)
    m = dict(zip(lab.b, zip(lab.cl_male, lab.cl_race)))
    gid = GROUP_ID[scheme]
    out = []
    for p in paths:
        v = m.get(os.path.basename(p))
        if v is None or pd.isna(v[0]) or pd.isna(v[1]):
            out.append(-1)
            continue
        race = str(v[1]).lower()
        if scheme == "six" and race == "asian":
            race = "others"
        out.append(gid.get((int(v[0]), race), -1))
    return np.array(out)


def fragility(y, pred, g, vid, rng, n_perm, n_boot):
    obs = ifpr(y, pred, g)
    r = y == 0
    rv = np.unique(vid[r])
    # video-permutation null: shuffle group labels across real videos (a video's frames keep one label)
    vg = pd.Series(g[r]).groupby(vid[r]).agg(lambda s: s.iloc[0])
    null = []
    for _ in range(n_perm):
        perm = dict(zip(vg.index, rng.permutation(vg.values)))
        g2 = g.copy()
        g2[r] = np.array([perm[v] for v in vid[r]])
        null.append(ifpr(y, pred, g2))
    null = np.array(null)
    # video bootstrap (all videos, both classes)
    uv = np.unique(vid)
    idx_by_v = {v: np.flatnonzero(vid == v) for v in uv}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uv, len(uv), replace=True)
        ii = np.concatenate([idx_by_v[v] for v in pick])
        boots.append(ifpr(y[ii], pred[ii], g[ii]))
    boots = np.array(boots)
    # leave-one-real-video-out and greedy halving
    loo = []
    for v in rv:
        keep = vid != v
        loo.append((v, ifpr(y[keep], pred[keep], g[keep])))
    loo_vals = np.array([x[1] for x in loo])
    keep = np.ones(len(y), bool)
    cur, steps, removed = obs, 0, []
    while cur > obs / 2 and steps < 25:
        best = None
        for v in np.unique(vid[keep & r]):
            k2 = keep & (vid != v)
            val = ifpr(y[k2], pred[k2], g[k2])
            if np.isfinite(val) and (best is None or val < best[1]):
                best = (v, val)
        if best is None:
            break
        keep &= vid != best[0]
        cur, steps = best[1], steps + 1
        removed.append(best[0])
    return {"iF_FPR": round(obs, 2), "perm_null_median": round(float(np.nanmedian(null)), 2),
            "perm_null_q05_q95": [round(float(np.nanpercentile(null, 5)), 2), round(float(np.nanpercentile(null, 95)), 2)],
            "p_null_ge_obs": round(float(np.nanmean(null >= obs)), 3),
            "boot95": [round(float(np.nanpercentile(boots, 2.5)), 2), round(float(np.nanpercentile(boots, 97.5)), 2)],
            "loo_min_max": [round(float(np.nanmin(loo_vals)), 2), round(float(np.nanmax(loo_vals)), 2)],
            "videos_to_halve": steps if cur <= obs / 2 else f">{steps}", "halving_videos": removed[:10],
            "n_real_videos": int(len(rv))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--sets", nargs="+", default=list(BM.SETS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20261026)
    pub = {k: dict(zip(BM.K, v)) for k, v in BM.DING_PUB.items()}
    report = {}
    for s in a.sets:
        scheme = BM.SETS[s]
        seeds = []
        for run in a.runs:
            z_, y_, g_, p_ = BM.load(f"{run}/eval/{s}_scores.npz")
            seeds.append((run, {"logit": z_, "label": y_, "group": g_, "img_path": p_}, matched_t(run)))
        z_, y_, g_, p_ = BM.load(f"{BM.DING_DIR}/full_s7_ep49_{BM.DING_TEST_NPZ[s]}.npz")
        dd = {"logit": z_, "label": y_, "group": g_, "img_path": p_}
        y, g, paths = seeds[0][1]["label"], seeds[0][1]["group"], seeds[0][1]["img_path"]
        for _, d, _ in seeds[1:]:
            assert (d["label"] == y).all() and (d["group"] == g).all(), "row order differs between seeds"
        assert (dd["label"] == y).all() and all(os.path.basename(p) == os.path.basename(q) for p, q in zip(dd["img_path"], paths))
        vid = video_ids(paths)
        zs = [d["logit"].astype(np.float64) for _, d, _ in seeds]
        zd = dd["logit"].astype(np.float64)
        # 1. detection
        auc_s = [100 * roc_auc_score(y, z) for z in zs]
        auc_d = 100 * roc_auc_score(y, zd)
        uv = np.unique(vid)
        idx_by_v = {v: np.flatnonzero(vid == v) for v in uv}
        deltas = []
        for _ in range(a.n_boot):
            ii = np.concatenate([idx_by_v[v] for v in rng.choice(uv, len(uv), replace=True)])
            if len(np.unique(y[ii])) < 2:
                continue
            deltas.append(np.mean([roc_auc_score(y[ii], z[ii]) for z in zs]) - roc_auc_score(y[ii], zd[ii]))
        deltas = 100 * np.array(deltas)
        # 2. columns at matched op
        colsets = [BM.cols(y, z, g, scheme, t) for (_, _, t), z in zip(seeds, zs)]
        mean_cols = {k: float(np.mean([c[k] for c in colsets])) for k in BM.K}
        ding_cols = BM.cols(y, zd, g, scheme, 0.0)
        w_mean = BM.wins(mean_cols, pub[s])
        # 3. fragility (seed 1 of ours; Ding repro), Xu labels; plus cleaner labels
        pred0 = zs[0] >= seeds[0][2]
        predd = zd >= 0.0
        assert abs(ifpr(y, pred0, g) - colsets[0]["iF_FPR"]) < 0.05, (ifpr(y, pred0, g), colsets[0]["iF_FPR"])
        frag_o = fragility(y, pred0, g, vid, rng, a.n_perm, a.n_boot)
        frag_d = fragility(y, predd, g, vid, rng, a.n_perm, a.n_boot)
        gc = cleaner_groups(paths, scheme, s)
        clean = None
        if gc is not None:
            ok = gc >= 0
            clean = {"coverage": round(float(ok.mean()), 3),
                     "ours_iF_FPR_cleaner": round(ifpr(y[ok], pred0[ok], gc[ok]), 2),
                     "ding_iF_FPR_cleaner": round(ifpr(y[ok], predd[ok], gc[ok]), 2)}
        report[s] = {
            "detection": {"auc_ours_seeds": [round(x, 2) for x in auc_s], "auc_ours_mean": round(float(np.mean(auc_s)), 2),
                          "auc_ours_sd": round(float(np.std(auc_s, ddof=1)), 2) if len(auc_s) > 1 else None,
                          "auc_ding_repro": round(auc_d, 2), "auc_ding_published": pub[s]["AUC"],
                          "delta_vs_repro_boot95": [round(float(np.percentile(deltas, 2.5)), 2), round(float(np.percentile(deltas, 97.5)), 2)]},
            "matched_cols_mean": {k: round(v, 2) for k, v in mean_cols.items()},
            "ding_repro_cols": {k: round(float(ding_cols[k]), 2) for k in BM.K},
            "ding_published": pub[s], "wins_vs_published": int(sum(w_mean)), "win_flags": [int(x) for x in w_mean],
            "fragility_ours_seed1": frag_o, "fragility_ding_repro": frag_d, "cleaner_labels": clean,
            "tpr_fpr_ours_seed1": [round(float(100 * pred0[y == 1].mean()), 1), round(float(100 * pred0[y == 0].mean()), 1)],
            "tpr_fpr_ding_repro": [round(float(100 * predd[y == 1].mean()), 1), round(float(100 * predd[y == 0].mean()), 1)],
        }
        print(f"[{s}] AUC ours {np.mean(auc_s):.2f} vs Ding repro {auc_d:.2f} (pub {pub[s]['AUC']}), dAUC 95% "
              f"{report[s]['detection']['delta_vs_repro_boot95']}; wins {int(sum(w_mean))}/10; iF_FPR ours {frag_o['iF_FPR']} "
              f"(null p {frag_o['p_null_ge_obs']}, boot {frag_o['boot95']}, halve after {frag_o['videos_to_halve']} videos) | "
              f"Ding {frag_d['iF_FPR']} (null p {frag_d['p_null_ge_obs']}, boot {frag_d['boot95']}) | cleaner {clean}", flush=True)
    json.dump(report, open(f"{a.out}/claim_report.json", "w"), indent=1, default=str)
    L = ["# Claim report: detection, fairness at the matched operating point, metric fragility", "",
         f"Runs: {', '.join(a.runs)}. Ding repro = official code, Xception, seed 7 (0.5 rule). Published = Ding CVPR'26 Tables 1-2.", "",
         "| Set | AUC ours | AUC Ding repro | AUC Ding pub | dAUC 95% CI (vs repro) | wins /10 | iF_FPR ours [boot95] (null p) | iF_FPR Ding repro [boot95] (null p) | videos to halve ours | iF_FPR cleaner labels ours / Ding |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    tot = 0
    for s, r in report.items():
        dt, fo, fd, cl = r["detection"], r["fragility_ours_seed1"], r["fragility_ding_repro"], r["cleaner_labels"] or {}
        tot += r["wins_vs_published"]
        L.append(f"| {BM.NAMES[s]} | {dt['auc_ours_mean']}{' +- ' + str(dt['auc_ours_sd']) if dt['auc_ours_sd'] is not None else ''} | "
                 f"{dt['auc_ding_repro']} | {dt['auc_ding_published']} | {dt['delta_vs_repro_boot95']} | {r['wins_vs_published']} | "
                 f"{fo['iF_FPR']} {fo['boot95']} ({fo['p_null_ge_obs']}) | {fd['iF_FPR']} {fd['boot95']} ({fd['p_null_ge_obs']}) | "
                 f"{fo['videos_to_halve']} | {cl.get('ours_iF_FPR_cleaner')} / {cl.get('ding_iF_FPR_cleaner')} |")
    L += ["", f"Total wins vs Ding published: {tot}/40.", ""]
    open(f"{a.out}/CLAIM_REPORT.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
