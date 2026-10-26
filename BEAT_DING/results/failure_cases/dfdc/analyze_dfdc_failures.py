"""Failure-case analysis of BD0 seed 7 on DFDC (cross-domain), CPU only.

Run: ~/.conda/envs/aiface/bin/python analyze_dfdc_failures.py
Inputs (read-only): BD0_s7 eval scores, Ding repro (official code, Xception, s7, ep49) scores, DEMO_AXES labels.
Outputs: this folder (json/csv tables, fp_sheet.png, fn_sheet.png, tables.md).
All counts are over protocol ROWS (the DFDC test CSV has 1,476 image paths listed twice; the fairness metric
counts every row, so we do too). Unique-image counts are reported alongside where it matters.
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
OUT = f"{R}/BEAT_DING/results/failure_cases/dfdc"
KEY = "dfdc"
GROUPS = ["M-Asian", "M-White", "M-Black", "M-Others", "F-Asian", "F-White", "F-Black", "F-Others"]
AGE_IF = {0: "<30", 1: "30-44", 2: "45-59", 3: "60+"}
AGE_AIF = {0: "<25", 1: "25-44", 2: "45-64", 3: "65+"}
ITA4 = {0: "Q1 darkest", 1: "Q2", 2: "Q3", 3: "Q4 lightest"}
MST3 = {0: "MST 1-3", 1: "MST 4-6", 2: "MST 7-10"}
os.makedirs(OUT, exist_ok=True)


def load():
    d = np.load(f"{R}/BEAT_DING/runs/BD0_s7/eval/{KEY}_scores.npz", allow_pickle=True)
    df = pd.DataFrame({k: d[k] for k in d.files})
    e = np.load(f"{R}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_{KEY}.npz", allow_pickle=True)
    assert (e["img_path"].astype(str) == df.img_path.values).all() and (e["label"] == df.label.values).all()
    df["ding_logit"] = e["logit"].astype(np.float64)
    df["ding_pred"] = df.ding_logit >= 0
    L = pd.read_csv(f"{R}/DEMO_AXES/labels/labels_{KEY}.csv")
    assert (L.img_path.values == df.img_path.values).all() and (L.xu_group.values == df.group.values).all()
    for c in L.columns:
        if c not in ("img_path", "label"):
            df[c] = L[c].values
    F = pd.read_csv(f"{R}/DEMO_AXES/labels_fairface/{KEY}.csv")
    F = F[["img_path", "race7", "sex", "age4", "vid_race2", "vid_sex", "vid_age4"]].drop_duplicates("img_path")
    F.columns = ["img_path"] + ["ff_" + c for c in F.columns[1:]]
    df = df.merge(F, on="img_path", how="left")
    df["grp"] = df.group.map(dict(enumerate(GROUPS)))
    df["xu_gender"] = np.where(df.xu_male == 1, "male", "female")
    df["cl_gender"] = np.where(df.cl_male == 1, "male", "female")
    df["cl_gender_vid"] = np.where(df.cl_male_vid == 1, "male", "female")
    df["age_if"] = df.age4_if.map(AGE_IF)
    df["age_aif"] = df.age4_aif.map(AGE_AIF)
    df["ita4_ffpp"] = df.ita4.map(ITA4)
    df["ita4_loc"] = df.ita4_local.map(ITA4)
    df["mst3"] = df.mst3_vid.map(MST3)
    df["det_q"] = pd.qcut(df.det_score, 4, labels=["det Q1 (lowest)", "det Q2", "det Q3", "det Q4"])
    df["gray_s"] = np.where(df.gray == 1, "grayscale", "colour")
    return df


def img_stats(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = y[1:-1, 1:-1] * 4 - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    return p, float(y.mean()), float(y.std()), float(lap.var()), float(sat.mean())


def add_quality(df):
    cache = f"{OUT}/image_quality.csv"
    if os.path.exists(cache):
        q = pd.read_csv(cache)
    else:
        paths = sorted(df.img_path.unique())
        with Pool(8) as pool:
            rows = pool.map(img_stats, paths, chunksize=64)
        q = pd.DataFrame(rows, columns=["img_path", "lum_mean", "lum_std", "sharp_lapvar", "sat_mean"])
        q.to_csv(cache, index=False)
    df = df.merge(q, on="img_path", how="left")
    for c, lab in [("lum_mean", "brightness"), ("lum_std", "contrast"), ("sharp_lapvar", "sharpness"), ("sat_mean", "saturation")]:
        df[c + "_q"] = pd.qcut(df[c], 4, labels=[f"{lab} Q1 (low)", f"{lab} Q2", f"{lab} Q3", f"{lab} Q4 (high)"])
    return df


def ffpr(fp, n):
    """intersectional F_FPR (cfes/metrics.py fairness_block) from per-group FP counts and real counts (x100)."""
    ok = n > 0
    fpr_all = fp[ok].sum() / n[ok].sum()
    return 100 * np.abs(fp[ok] / n[ok] - fpr_all).sum()


def breakdown(df, col, pred_col, kind):
    """kind 'fp': rows = reals, rate = FPR; kind 'fn': rows = fakes, rate = FNR (miss rate)."""
    sub = df[df.label == (0 if kind == "fp" else 1)]
    err = sub[pred_col] if kind == "fp" else ~sub[pred_col]
    g = pd.DataFrame({"k": sub[col].astype(object).fillna("n/a").values, "err": err.values, "vid": sub.video.values})
    t = g.groupby("k").agg(n=("err", "size"), errors=("err", "sum"), videos=("vid", "nunique"))
    t["rate%"] = (100 * t.errors / t.n).round(2)
    t["share_of_errors%"] = (100 * t.errors / max(err.sum(), 1)).round(2)
    return t.reset_index().rename(columns={"k": col})


def md(t, floatfmt=2):
    cols = list(t.columns)
    s = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in t.iterrows():
        s += "| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |\n"
    return s


def main():
    df = load()
    df = add_quality(df)
    thr = json.load(open(f"{R}/BEAT_DING/runs/BD0_s7/eval/bd_eval.json"))["thresholds"]
    res = {"n_rows": len(df), "n_unique_images": int(df.img_path.nunique()),
           "n_duplicated_paths": int(df.img_path.duplicated().sum()),
           "n_real_rows": int((df.label == 0).sum()), "n_fake_rows": int((df.label == 1).sum()),
           "n_real_videos": int(df[df.label == 0].video.nunique()), "n_fake_videos": int(df[df.label == 1].video.nunique()),
           "thresholds": thr, "ops": {}}
    real, fake = df.label == 0, df.label == 1
    res["logit_percentiles_real_10_50_90"] = np.percentile(df.logit[real], [10, 50, 90]).round(2).tolist()
    res["logit_percentiles_fake_10_50_90"] = np.percentile(df.logit[fake], [10, 50, 90]).round(2).tolist()
    v = np.load(f"{R}/BEAT_DING/runs/BD0_s7/eval/val_scores.npz", allow_pickle=True)
    res["val_logit_percentiles_real_10_50_90"] = np.percentile(v["logit"][v["label"] == 0], [10, 50, 90]).round(2).tolist()
    res["ding"] = {"FPR%": round(100 * df.ding_pred[real].mean(), 2), "TPR%": round(100 * df.ding_pred[fake].mean(), 2),
                   "FP": int(df.ding_pred[real].sum()), "FN": int((~df.ding_pred[fake]).sum())}
    T = "# DFDC failure tables (generated by analyze_dfdc_failures.py)\n\n"
    T += (f"Rows {len(df)} ({res['n_unique_images']} unique images; {res['n_duplicated_paths']} rows are repeated paths, "
          f"counted as the metric counts them). Reals {res['n_real_rows']} rows / {res['n_real_videos']} videos; "
          f"fakes {res['n_fake_rows']} rows / {res['n_fake_videos']} videos. DFDC fakes carry NO method label in the "
          f"protocol CSV (all 'unknown').\n\n")
    T += "## Operating points\n\n| op | logit t | FP | FPR% | FN | FNR% | TPR% | FP videos (>=1 FP frame) | FN videos (>=1 miss) |\n|---|---|---|---|---|---|---|---|---|\n"
    for op, t in list(thr.items()) + [("Ding repro 0.5", None)]:
        pred = (df.logit >= t) if t is not None else df.ding_pred
        df["pred_" + op] = pred
        fp, fn = int((pred & real).sum()), int((~pred & fake).sum())
        o = {"t": t, "FP": fp, "FPR%": round(100 * fp / real.sum(), 2), "FN": fn, "FNR%": round(100 * fn / fake.sum(), 2),
             "TPR%": round(100 - 100 * fn / fake.sum(), 2),
             "FP_videos": int(df[pred & real].video.nunique()), "FN_videos": int(df[~pred & fake].video.nunique())}
        fpg = df[real].groupby("group")["pred_" + op].agg(["sum", "size"]).reindex(range(8))
        o["iF_FPR_recomputed"] = round(float(ffpr(fpg["sum"].values.astype(float), fpg["size"].values.astype(float))), 4)
        res["ops"][op] = o
        T += f"| {op} | {t if t is None else round(t, 2)} | {fp} | {o['FPR%']} | {fn} | {o['FNR%']} | {o['TPR%']} | {o['FP_videos']}/{res['n_real_videos']} | {o['FN_videos']}/{res['n_fake_videos']} |\n"
    T += "\n"

    # ---------- 1. false positives ----------
    axes_fp = ["grp", "xu_gender", "xu_race", "cl_gender", "cl_race", "cl_race_vid", "ff_race7", "ff_vid_sex", "age_if",
               "age_aif", "ff_vid_age4", "ita4_ffpp", "ita4_loc", "mst3", "gray_s", "det_q", "lum_mean_q", "lum_std_q",
               "sharp_lapvar_q", "sat_mean_q"]
    res["fp_breakdowns"], res["fn_breakdowns"] = {}, {}
    for op in thr:
        res["fp_breakdowns"][op] = {}
        for a in axes_fp:
            res["fp_breakdowns"][op][a] = breakdown(df, a, "pred_" + op, "fp").to_dict("records")
    T += "## 1. False positives (reals flagged fake)\n\n"
    for a in axes_fp:
        tabs = [breakdown(df, a, "pred_" + op, "fp").set_index(a)[["n", "videos", "errors", "rate%"]].rename(
            columns={"errors": f"FP@{op}", "rate%": f"FPR%@{op}"}) for op in thr]
        t = pd.concat([tabs[0]] + [x.drop(columns=["n", "videos"]) for x in tabs[1:]], axis=1).reset_index()
        dg = breakdown(df, a, "pred_Ding repro 0.5", "fp").set_index(a)["rate%"].rename("FPR%@Ding")
        t = t.merge(dg.reset_index(), on=a, how="left")
        T += f"**by {a}**\n\n" + md(t) + "\n"

    # per-video concentration (matched)
    vids = {}
    for op in list(thr) + ["Ding repro 0.5"]:
        rv = df[real].groupby("video").agg(n=("label", "size"), fp=("pred_" + op, "sum"), group=("group", "first"))
        vids[op] = rv
    rv = vids["matched"].copy()
    rv["fpr%"] = 100 * rv.fp / rv.n
    rv["ding_fpr%"] = 100 * vids["Ding repro 0.5"].fp / rv.n
    attrs = df[real].groupby("video").agg(grp=("grp", "first"), cl_race_vid=("cl_race_vid", "first"),
                                          cl_gender_vid=("cl_gender_vid", "first"), ff_race7=("ff_race7", lambda s: s.mode().iat[0] if s.notna().any() else "n/a"),
                                          age_vid_if=("age_vid_if", "first"), ita_vid=("ita_vid", "first"), ita4=("ita4_ffpp", "first"),
                                          lum=("lum_mean", "median"), sharp=("sharp_lapvar", "median"),
                                          max_logit=("logit", "max"), med_logit=("logit", "median"))
    rv = rv.join(attrs).sort_values(["fp", "fpr%"], ascending=False)
    rv.to_csv(f"{OUT}/fp_videos_matched.csv")
    tot = rv.fp.sum()
    cum = rv.fp.cumsum().values / tot
    conc = {f"top{k}_share%": round(100 * cum[k - 1], 2) for k in (5, 10, 15, 25, 50, 100)}
    conc["videos_for_50pct_of_FP"] = int(np.searchsorted(cum, 0.5) + 1)
    conc["videos_for_80pct_of_FP"] = int(np.searchsorted(cum, 0.8) + 1)
    conc["videos_with_any_FP"] = int((rv.fp > 0).sum())
    conc["videos_with_FPR>=50%"] = int((rv["fpr%"] >= 50).sum())
    conc["videos_with_FPR>=90%"] = int((rv["fpr%"] >= 90).sum())
    conc["video_fpr_median%"] = round(float(rv["fpr%"].median()), 2)
    res["fp_video_concentration_matched"] = conc
    T += "### FP concentration by video (matched)\n\n" + "\n".join(f"- {k}: {v}" for k, v in conc.items()) + "\n\n"
    top = rv.head(15).reset_index()[["video", "grp", "n", "fp", "fpr%", "ding_fpr%", "cl_gender_vid", "cl_race_vid", "ff_race7",
                                      "age_vid_if", "ita_vid", "ita4", "lum", "sharp", "max_logit", "med_logit"]]
    T += "**Top-15 real videos by #FP frames (matched)**\n\n" + md(top) + "\n"
    res["fp_top15_videos_matched"] = top.to_dict("records")

    # how much of i-F_FPR is driven by few videos
    def gstats(rvv, drop=()):
        k = rvv.drop(index=list(drop))
        a = k.groupby("group").agg(fp=("fp", "sum"), n=("n", "sum")).reindex(range(8)).fillna(0)
        return a.fp.values.astype(float), a.n.values.astype(float)
    disp = {}
    for op in thr:
        rvo = vids[op]
        base = ffpr(*gstats(rvo))
        # per-group terms
        fp_, n_ = gstats(rvo)
        all_ = fp_.sum() / n_.sum()
        terms = {GROUPS[i]: round(100 * (fp_[i] / n_[i] - all_), 2) for i in range(8)}
        # greedy: drop the real video whose removal lowers i-F_FPR the most
        vg, vfp, vn = rvo.group.values, rvo.fp.values.astype(float), rvo.n.values.astype(float)
        alive = np.ones(len(rvo), bool)
        gfp, gn = fp_.copy(), n_.copy()

        def drop_scores():
            out = np.full(len(rvo), np.inf)
            for i in np.where(alive)[0]:
                a, b = gfp.copy(), gn.copy()
                a[vg[i]] -= vfp[i]
                b[vg[i]] -= vn[i]
                out[i] = ffpr(a, b)
            return out
        loo_arr = drop_scores()
        dropped, path = [], [round(base, 3)]
        for _ in range(40):
            sc = drop_scores()
            i = int(np.argmin(sc))
            dropped.append(rvo.index[i])
            path.append(round(float(sc[i]), 3))
            alive[i] = False
            gfp[vg[i]] -= vfp[i]
            gn[vg[i]] -= vn[i]
        half = next((i for i, x in enumerate(path) if x <= base / 2), None)
        # leave-one-video-out maximum change
        loo = dict(zip(rvo.index, loo_arr))
        lv = min(loo, key=loo.get)
        # permutation null: shuffle video->group labels among real videos (same group sizes in videos)
        rng = np.random.default_rng(0)
        grp = rvo.group.values.copy()
        nulls = []
        for _ in range(2000):
            pg = rng.permutation(grp)
            a = pd.DataFrame({"g": pg, "fp": rvo.fp.values, "n": rvo.n.values}).groupby("g").sum().reindex(range(8)).fillna(0)
            nulls.append(ffpr(a.fp.values.astype(float), a.n.values.astype(float)))
        nulls = np.array(nulls)
        disp[op] = {"iF_FPR": round(base, 4), "group_FPR_minus_pooled_pp": terms,
                    "greedy_drop_path_first10": path[:11], "greedy_dropped_videos_first10": dropped[:10],
                    "greedy_videos_to_halve_iF_FPR": half,
                    "greedy_iF_FPR_after_5_10_20_drops": [path[5], path[10], path[20]],
                    "loo_best_single_video": lv, "loo_best_iF_FPR": round(loo[lv], 3),
                    "perm_null_median": round(float(np.median(nulls)), 2),
                    "perm_null_p5_p95": np.percentile(nulls, [5, 95]).round(2).tolist(),
                    "perm_p_value_one_sided": round(float((nulls >= base).mean()), 4)}
    res["fp_disparity_vs_videos"] = disp
    T += "### How few videos drive intersectional F_FPR\n\n```\n" + json.dumps(disp, indent=1, default=str) + "\n```\n\n"

    # ---------- 2. false negatives ----------
    axes_fn = ["grp", "xu_gender", "xu_race", "cl_gender", "cl_race_vid", "ff_race7", "age_if", "age_aif", "ita4_ffpp",
               "ita4_loc", "mst3", "gray_s", "det_q", "lum_mean_q", "lum_std_q", "sharp_lapvar_q", "sat_mean_q", "method"]
    T += "## 2. False negatives (fakes missed)\n\nDFDC fakes have no method label in the protocol release (method = 'unknown' for all 8,158 fake rows), so no per-method breakdown is possible.\n\n"
    for op in thr:
        res["fn_breakdowns"][op] = {a: breakdown(df, a, "pred_" + op, "fn").to_dict("records") for a in axes_fn}
    for a in axes_fn:
        tabs = [breakdown(df, a, "pred_" + op, "fn").set_index(a)[["n", "videos", "errors", "rate%"]].rename(
            columns={"errors": f"FN@{op}", "rate%": f"FNR%@{op}"}) for op in thr]
        t = pd.concat([tabs[0]] + [x.drop(columns=["n", "videos"]) for x in tabs[1:]], axis=1).reset_index()
        dg = breakdown(df, a, "pred_Ding repro 0.5", "fn").set_index(a)["rate%"].rename("FNR%@Ding")
        t = t.merge(dg.reset_index(), on=a, how="left")
        T += f"**by {a}**\n\n" + md(t) + "\n"
    fv = df[fake].groupby("video").agg(n=("label", "size"), fn=("pred_matched", lambda s: int((~s).sum())),
                                       ding_fn=("pred_Ding repro 0.5", lambda s: int((~s).sum())),
                                       grp=("grp", "first"), cl_gender_vid=("cl_gender_vid", "first"),
                                       cl_race_vid=("cl_race_vid", "first"), age_vid_if=("age_vid_if", "first"),
                                       ita_vid=("ita_vid", "first"), lum=("lum_mean", "median"), sharp=("sharp_lapvar", "median"),
                                       min_logit=("logit", "min"), med_logit=("logit", "median"))
    fv["fnr%"] = 100 * fv.fn / fv.n
    fv["ding_fnr%"] = 100 * fv.ding_fn / fv.n
    fv = fv.sort_values(["fn", "fnr%"], ascending=False)
    fv.to_csv(f"{OUT}/fn_videos_matched.csv")
    cumf = fv.fn.cumsum().values / fv.fn.sum()
    fconc = {f"top{k}_share%": round(100 * cumf[k - 1], 2) for k in (5, 10, 15, 25, 50)}
    fconc.update({"videos_for_50pct_of_FN": int(np.searchsorted(cumf, 0.5) + 1),
                  "videos_with_any_FN": int((fv.fn > 0).sum()), "videos_with_FNR>=50%": int((fv["fnr%"] >= 50).sum()),
                  "videos_with_FNR=100%": int((fv["fnr%"] >= 100).sum()),
                  "ding_videos_with_FNR>=50%": int((fv["ding_fnr%"] >= 50).sum())})
    res["fn_video_concentration_matched"] = fconc
    T += "### FN concentration by video (matched)\n\n" + "\n".join(f"- {k}: {v}" for k, v in fconc.items()) + "\n\n"
    topf = fv.head(15).reset_index()[["video", "grp", "n", "fn", "fnr%", "ding_fnr%", "cl_gender_vid", "cl_race_vid", "age_vid_if",
                                       "ita_vid", "lum", "sharp", "min_logit", "med_logit"]]
    T += "**Top-15 fake videos by #missed frames (matched)**\n\n" + md(topf) + "\n"
    res["fn_top15_videos_matched"] = topf.to_dict("records")
    # Asian fakes are tiny
    res["asian_fake_videos"] = df[fake & (df.xu_race == "asian")].video.unique().tolist()

    # ---------- 3. shared vs unique with Ding ----------
    sh = {}
    for op in list(thr):
        p = df["pred_" + op]
        ours_fp, ding_fp = p & real, df.ding_pred & real
        ours_fn, ding_fn = ~p & fake, ~df.ding_pred & fake
        sh[op] = {"ours_FP": int(ours_fp.sum()), "ours_FP_also_Ding%": round(100 * (ours_fp & ding_fp).sum() / max(ours_fp.sum(), 1), 2),
                  "Ding_FP": int(ding_fp.sum()), "Ding_FP_also_ours%": round(100 * (ours_fp & ding_fp).sum() / ding_fp.sum(), 2),
                  "FP_only_ours": int((ours_fp & ~ding_fp).sum()), "FP_only_Ding": int((~ours_fp & ding_fp).sum()),
                  "ours_FN": int(ours_fn.sum()), "ours_FN_also_Ding%": round(100 * (ours_fn & ding_fn).sum() / max(ours_fn.sum(), 1), 2),
                  "Ding_FN": int(ding_fn.sum()), "Ding_FN_also_ours%": round(100 * (ours_fn & ding_fn).sum() / ding_fn.sum(), 2),
                  "FN_only_ours": int((ours_fn & ~ding_fn).sum()), "FN_only_Ding": int((~ours_fn & ding_fn).sum()),
                  "chance_overlap_FP%(=Ding FPR)": round(100 * ding_fp.sum() / real.sum(), 2),
                  "chance_overlap_FN%(=Ding FNR)": round(100 * ding_fn.sum() / fake.sum(), 2)}
    # rank correlation of the two models' logits within each class (frame and video median)
    from scipy.stats import spearmanr
    sh["spearman_logit_real_frames"] = round(float(spearmanr(df.logit[real], df.ding_logit[real])[0]), 3)
    sh["spearman_logit_fake_frames"] = round(float(spearmanr(df.logit[fake], df.ding_logit[fake])[0]), 3)
    vm = df.groupby(["video", "label"]).agg(o=("logit", "median"), d=("ding_logit", "median")).reset_index()
    sh["spearman_video_median_real"] = round(float(spearmanr(vm.o[vm.label == 0], vm.d[vm.label == 0])[0]), 3)
    sh["spearman_video_median_fake"] = round(float(spearmanr(vm.o[vm.label == 1], vm.d[vm.label == 1])[0]), 3)
    res["shared_vs_ding"] = sh
    T += "## 3. Shared vs unique with the Ding reproduction (0.5 rule)\n\n```\n" + json.dumps(sh, indent=1) + "\n```\n\n"
    cols = ["video", "grp", "logit", "ding_logit", "cl_race_vid", "age_vid_if", "ita_vid", "img_path"]
    u = df.drop_duplicates("img_path")
    uf = u[(u.label == 0) & u.pred_matched & ~u.ding_pred].sort_values("logit", ascending=False)
    uf_vid = uf.drop_duplicates("video").head(10)[cols]
    df_ = u[(u.label == 0) & ~u.pred_matched & u.ding_pred].sort_values("ding_logit", ascending=False).drop_duplicates("video").head(10)[cols]
    un = u[(u.label == 1) & ~u.pred_matched & u.ding_pred].sort_values("logit").drop_duplicates("video").head(10)[cols]
    dn = u[(u.label == 1) & u.pred_matched & ~u.ding_pred].sort_values("ding_logit").drop_duplicates("video").head(10)[cols]
    # video level: FP-rate difference
    rvd = rv.assign(diff=rv["fpr%"] - rv["ding_fpr%"]).reset_index()
    fvd = fv.assign(diff=fv["fnr%"] - fv["ding_fnr%"]).reset_index()
    T += "**FP: we flag, Ding does not (top-10 frames, one per video, by our logit)**\n\n" + md(uf_vid) + "\n"
    T += "**FP: Ding flags, we do not (top-10, by Ding logit)**\n\n" + md(df_) + "\n"
    T += "**FN: we miss, Ding catches (top-10, lowest our logit)**\n\n" + md(un) + "\n"
    T += "**FN: Ding misses, we catch (top-10, lowest Ding logit)**\n\n" + md(dn) + "\n"
    vcols = ["video", "grp", "n", "fpr%", "ding_fpr%", "diff", "cl_race_vid", "age_vid_if", "ita_vid"]
    T += "**Real videos where our FPR exceeds Ding's most (matched)**\n\n" + md(rvd.sort_values("diff", ascending=False).head(10)[vcols]) + "\n"
    T += "**Real videos where Ding's FPR exceeds ours most**\n\n" + md(rvd.sort_values("diff").head(10)[vcols]) + "\n"
    vcols = ["video", "grp", "n", "fnr%", "ding_fnr%", "diff", "cl_race_vid", "age_vid_if", "ita_vid"]
    T += "**Fake videos where our miss rate exceeds Ding's most (matched)**\n\n" + md(fvd.sort_values("diff", ascending=False).head(10)[vcols]) + "\n"
    T += "**Fake videos where Ding's miss rate exceeds ours most**\n\n" + md(fvd.sort_values("diff").head(10)[vcols]) + "\n"
    res["video_level_vs_ding"] = {
        "real_videos_ours_fpr_gt_ding_by_50pp": int((rvd["diff"] >= 50).sum()),
        "real_videos_ding_fpr_gt_ours_by_50pp": int((rvd["diff"] <= -50).sum()),
        "fake_videos_ours_fnr_gt_ding_by_50pp": int((fvd["diff"] >= 50).sum()),
        "fake_videos_ding_fnr_gt_ours_by_50pp": int((fvd["diff"] <= -50).sum())}
    T += "```\n" + json.dumps(res["video_level_vs_ding"], indent=1) + "\n```\n\n"

    # ---------- 4. most confident errors ----------
    cc = ["video", "grp", "logit", "ding_logit", "ding_pred", "cl_gender_vid", "cl_race_vid", "age_vid_if", "ita_vid", "img_path"]
    tfp = u[u.label == 0].sort_values("logit", ascending=False).head(12)[cc]
    tfn = u[u.label == 1].sort_values("logit").head(12)[cc]
    T += "## 4. Most confident errors\n\n**Top-12 FP (highest logit among reals; unique images)**\n\n" + md(tfp) + "\n"
    T += "**Top-12 FN (lowest logit among fakes)**\n\n" + md(tfn) + "\n"
    res["top12_fp"] = tfp.to_dict("records")
    res["top12_fn"] = tfn.to_dict("records")
    res["top12_fp_videos_distinct"] = int(tfp.video.nunique())
    res["top12_fn_videos_distinct"] = int(tfn.video.nunique())

    # ---------- 5. contact sheets ----------
    sfp = u[(u.label == 0) & u.pred_matched].sort_values("logit", ascending=False).drop_duplicates("video").head(24)
    sfn = u[(u.label == 1) & ~u.pred_matched].sort_values("logit").drop_duplicates("video").head(24)
    sheet(sfp, f"{OUT}/fp_sheet.png", "BD0 s7 on DFDC: top-24 false positives (real frames flagged fake), one per video, matched t=%.2f" % thr["matched"], "fp")
    sheet(sfn, f"{OUT}/fn_sheet.png", "BD0 s7 on DFDC: top-24 false negatives (missed fakes), one per video, matched t=%.2f" % thr["matched"], "fn")
    sfp[cc].to_csv(f"{OUT}/fp_sheet_items.csv", index=False)
    sfn[cc].to_csv(f"{OUT}/fn_sheet_items.csv", index=False)
    res["fp_sheet_ding_also_wrong"] = int(sfp.ding_pred.sum())
    res["fn_sheet_ding_also_wrong"] = int((~sfn.ding_pred).sum())

    # ---------- extras ----------
    ex = {}
    # (a) Xu-vs-cleaner label disagreement and i-F_FPR recomputed on cleaner video-level labels
    race_rank = {"asian": 0, "white": 1, "black": 2, "others": 3}
    df["cl_group"] = 4 * (df.cl_male_vid != 1).astype(int) + df.cl_race_vid.map(race_rank)
    rvid = df[real].groupby("video").agg(xg=("group", "first"), cg=("cl_group", "first"))
    ex["real_videos_xu_vs_cleaner_group_disagree"] = int((rvid.xg != rvid.cg).sum())
    ex["real_videos_xu_vs_cleaner_gender_disagree"] = int((df[real].groupby("video").apply(
        lambda s: (s.xu_male.iat[0] != s.cl_male_vid.iat[0]))).sum())
    t15 = rv.head(15).index
    ex["top15_fp_videos_xu_vs_cleaner_group_disagree"] = int((rvid.loc[t15].xg != rvid.loc[t15].cg).sum())
    fvid = df[fake].groupby("video").agg(xg=("group", "first"), cg=("cl_group", "first"))
    ex["fake_videos_xu_vs_cleaner_group_disagree"] = int((fvid.xg != fvid.cg).sum())
    ex["n_real_videos"], ex["n_fake_videos"] = len(rvid), len(fvid)
    ex["iF_FPR_cleaner_vid_labels"] = {}
    for op in list(thr) + ["Ding repro 0.5"]:
        a = df[real].groupby("cl_group")["pred_" + op].agg(["sum", "size"]).reindex(range(8)).fillna(0)
        ex["iF_FPR_cleaner_vid_labels"][op] = round(float(ffpr(a["sum"].values.astype(float), a["size"].values.astype(float))), 2)
    cv = df[real].groupby("video").agg(g=("cl_group", "first"), fp=("pred_matched", "sum"), n=("label", "size"))
    obs = ffpr(np.bincount(cv.g, cv.fp, 8), np.bincount(cv.g, cv.n, 8))
    rng0 = np.random.default_rng(2)
    nl = np.array([ffpr(np.bincount(pg, cv.fp, 8), np.bincount(pg, cv.n, 8))
                   for pg in (rng0.permutation(cv.g.values) for _ in range(2000))])
    ex["cleaner_perm_null_matched"] = {"obs": round(float(obs), 2), "median": round(float(np.median(nl)), 2),
                                       "p5_p95": np.percentile(nl, [5, 95]).round(2).tolist(),
                                       "p_one_sided": round(float((nl >= obs).mean()), 4)}
    fb = rv[rv.grp == "F-Black"].sort_values("fp", ascending=False)
    ex["F_Black_FPR_after_dropping_top_k_FBlack_videos"] = {
        "n_videos": len(fb), **{f"drop{k}": round(100 * fb.iloc[k:].fp.sum() / fb.iloc[k:].n.sum(), 2) for k in (0, 3, 5, 10)},
        "pooled_after_drop10": round(100 * (rv.fp.sum() - fb.iloc[:10].fp.sum()) / (rv.n.sum() - fb.iloc[:10].n.sum()), 2),
        "top10_FBlack_videos": fb.index[:10].tolist()}
    ex["cleaner_group_FPR_matched%"] = {GROUPS[g]: round(100 * float(x), 2) for g, x in
                                        df[real].groupby("cl_group").pred_matched.mean().items()}
    # (b) brightness quartile x cleaner race, FPR and FNR at matched (does darkness explain the Black-group excess?)
    for kind, lab in (("fp", 0), ("fn", 1)):
        s = df[df.label == lab]
        e = s.pred_matched if kind == "fp" else ~s.pred_matched
        tab = (100 * e.groupby([s.cl_race_vid, s.lum_mean_q], observed=True).mean()).round(1).unstack()
        cnt = e.groupby([s.cl_race_vid, s.lum_mean_q], observed=True).size().unstack()
        ex[f"{kind}_rate_by_race_x_brightness_matched"] = {r: {str(k): f"{tab.loc[r, k]} (n={cnt.loc[r, k]})" for k in tab.columns if pd.notna(tab.loc[r, k])} for r in tab.index}
    # brightness gap within reals: median frame brightness of real frames by cleaner race
    ex["real_median_brightness_by_race"] = df[real].groupby("cl_race_vid").lum_mean.median().round(1).to_dict()
    # Spearman of logit with brightness / sharpness among reals and fakes
    from scipy.stats import spearmanr
    for c in ("lum_mean", "lum_std", "sharp_lapvar"):
        ex[f"spearman_logit_{c}_real"] = round(float(spearmanr(df.logit[real], df[c][real])[0]), 3)
        ex[f"spearman_logit_{c}_fake"] = round(float(spearmanr(df.logit[fake], df[c][fake])[0]), 3)
    # (c) greedy-halving calibration: same greedy on 20 random video->group permutations (matched)
    rvo = vids["matched"]
    vfp, vn = rvo.fp.values.astype(float), rvo.n.values.astype(float)
    rng = np.random.default_rng(1)
    halves = []
    for _ in range(20):
        vg = rng.permutation(rvo.group.values)
        gfp = np.bincount(vg, vfp, 8)
        gn = np.bincount(vg, vn, 8)
        base = ffpr(gfp, gn)
        alive = np.ones(len(vg), bool)
        k = 0
        while ffpr(gfp, gn) > base / 2 and k < 60:
            best, bi = np.inf, -1
            for i in np.where(alive)[0]:
                a, b = gfp.copy(), gn.copy()
                a[vg[i]] -= vfp[i]
                b[vg[i]] -= vn[i]
                v_ = ffpr(a, b)
                if v_ < best:
                    best, bi = v_, i
            alive[bi] = False
            gfp[vg[bi]] -= vfp[bi]
            gn[vg[bi]] -= vn[bi]
            k += 1
        halves.append(k)
    ex["perm_greedy_videos_to_halve_matched"] = {"values": halves, "median": float(np.median(halves))}
    # (d) contact-sheet composition
    ex["fn_sheet_group_counts"] = sfn.grp.value_counts().to_dict()
    ex["fp_sheet_group_counts"] = sfp.grp.value_counts().to_dict()
    ex["top15_fn_videos_group_counts"] = topf.grp.value_counts().to_dict()
    ex["fn_sheet_cleaner_white_female"] = int(((sfn.cl_race_vid == "white") & (sfn.cl_male_vid != 1)).sum())
    # (e) fraction of fakes whose logit sits within 10 logits of the matched threshold
    ex["fake_rows_within_10_logits_of_t_matched%"] = round(100 * float((np.abs(df.logit[fake] - thr["matched"]) < 10).mean()), 2)
    ex["real_rows_within_10_logits_of_t_matched%"] = round(100 * float((np.abs(df.logit[real] - thr["matched"]) < 10).mean()), 2)
    res["extras"] = ex
    T += "## Extras\n\n```\n" + json.dumps(ex, indent=1, default=str) + "\n```\n"
    print(json.dumps(ex, indent=1, default=str))

    open(f"{OUT}/tables.md", "w").write(T)
    json.dump(res, open(f"{OUT}/failures_dfdc.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("ops", "fp_video_concentration_matched", "fn_video_concentration_matched", "ding",
                                         "logit_percentiles_real_10_50_90", "logit_percentiles_fake_10_50_90",
                                         "val_logit_percentiles_real_10_50_90", "shared_vs_ding", "video_level_vs_ding",
                                         "fp_sheet_ding_also_wrong", "fn_sheet_ding_also_wrong", "asian_fake_videos")}, indent=1, default=str))
    print(json.dumps(disp, indent=1, default=str))


def sheet(s, path, title, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(s)
    cols, rows = 6, int(np.ceil(n / 6))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 4.0))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axs.ravel(), s.iterrows()):
        ax.imshow(Image.open(r.img_path).convert("RGB"))
        if kind == "fp":
            dd = "Ding: FP too" if r.ding_pred else "Ding: correct"
        else:
            dd = "Ding: missed too" if not r.ding_pred else "Ding: caught"
        ax.set_title(f"vid {r.video} | {r.grp}\nDFDC {'real' if kind == 'fp' else 'fake (method n/a)'}\n"
                     f"logit {r.logit:.1f} | {dd} ({r.ding_logit:.2f})", fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
