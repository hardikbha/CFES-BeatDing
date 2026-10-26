"""Failure-case analysis of BD0 seed 7 on Celeb-DF (cross-domain), CPU only.

Run: ~/.conda/envs/aiface/bin/python analyze_celebdf_failures.py
Reads (read-only): BD0_s7 eval scores + bd_eval.json thresholds, Ding reproduction (full_s7_ep49) scores,
DEMO_AXES labels (Xu protocol labels, cleaner AI-Face labels, InsightFace age, ITA) and FairFace-protocol labels.
Writes only into this folder: failures.json (every number), tables/*.csv, fp_sheet.png, fn_sheet.png,
FAILURES_tables.md (auto-generated tables that FAILURES.md embeds).

Conventions: label 1 = fake. BD decision = logit >= t (t from bd_eval.json 'thresholds', logit units).
Ding reproduction decision = logit >= 0 (its 0.5 rule). Group = Xu intersectional id, six-group scheme
(0 M-White, 1 M-Black, 2 M-Others, 3 F-White, 4 F-Black, 5 F-Others; F = 'nonmale' in the metric code).
F_FPR = 100 * sum_g |FPR_g - FPR_all| over groups holding both classes (same as cfes/metrics.py fairness_block);
verified against bd_eval.json test.celebdf.*.cols below (assert).
"""
import json
import os
import re

import numpy as np
import pandas as pd

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
EV = f"{R}/BEAT_DING/runs/BD0_s7/eval"
OUT = f"{R}/BEAT_DING/results/failure_cases/celebdf"
os.makedirs(f"{OUT}/tables", exist_ok=True)
SIX = ["male,white", "male,black", "male,others", "nonmale,white", "nonmale,black", "nonmale,others"]
SHORT = {"male": "M", "nonmale": "F"}
GNAME = {i: SHORT[n.split(",")[0]] + "-" + n.split(",")[1].capitalize() for i, n in enumerate(SIX)}
RES = {}

# ---------------------------------------------------------------- load + align
bd = np.load(f"{EV}/celebdf_scores.npz", allow_pickle=True)
dg = np.load(f"{R}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_celebdf.npz", allow_pickle=True)
assert (bd["img_path"] == dg["img_path"]).all() and (bd["label"] == dg["label"]).all() and (bd["group"] == dg["group"]).all()
df = pd.DataFrame({k: bd[k] for k in ("img_path", "label", "group", "method", "cluster")})
df["logit"] = bd["logit"].astype(np.float64)
df["ding_logit"] = dg["logit"].astype(np.float64)
assert df.img_path.is_unique
lab = pd.read_csv(f"{R}/DEMO_AXES/labels/labels_celebdf.csv")
ff = pd.read_csv(f"{R}/DEMO_AXES/labels_fairface/celebdf.csv")[
    ["img_path", "race7", "sex", "age4", "vid_sex", "vid_race2", "vid_age4"]].rename(
    columns={"race7": "ff_race7", "sex": "ff_sex", "age4": "ff_age4", "vid_sex": "ff_vid_sex",
             "vid_race2": "ff_vid_race2", "vid_age4": "ff_vid_age4"})
n0 = len(df)
df = df.merge(lab.drop(columns=["label"]), on="img_path", how="left", validate="1:1").merge(ff, on="img_path", how="left", validate="1:1")
assert len(df) == n0 and df.video.notna().all()
assert (df.xu_group == df.group).all(), "npz group != Xu group"
df["gname"] = df.group.map(GNAME)
df["xu_gender"] = np.where(df.xu_male == 1, "male", "nonmale")
df["cl_gender"] = np.where(df.cl_male == 1, "male", "nonmale")
df["cl_gender_vid"] = np.where(df.cl_male_vid == 1, "male", "nonmale")
df["src"] = df.video.str.split("_").str[1]
m = df.video.str.extract(r"Celeb-synthesis_(id\d+)_(id\d+)_\d+")
df["tgt_id"], df["src_id"] = m[0], m[1]
df.loc[df.src == "Celeb-real", "tgt_id"] = df.video.str.extract(r"Celeb-real_(id\d+)_")[0]
df["pair"] = df.tgt_id + "->" + df.src_id
AGE4 = {0: "<30", 1: "30-44", 2: "45-59", 3: "60+"}
df["age4_if_name"] = df.age4_if.map(AGE4)
df["ita4_name"] = df.ita4.map({0: "Q1 darkest", 1: "Q2", 2: "Q3", 3: "Q4 lightest"}).fillna("no ITA")
df["ita4_local_name"] = df.ita4_local.map({0: "Q1 darkest", 1: "Q2", 2: "Q3", 3: "Q4 lightest"}).fillna("no ITA")
df["vid_short"] = df.video.str.replace(r"^\d_", "", regex=True)
df["frame_idx"] = df.img_path.str.extract(r"_(\d+)\.png$")[0].astype(int)
df["frame_pos"] = np.where(df.frame_idx == 1, "first frame (idx 1)", "later frames")
real, fake = df.label == 0, df.label == 1
RES["data"] = dict(n=int(len(df)), n_real=int(real.sum()), n_fake=int(fake.sum()),
                   real_videos=int(df[real].video.nunique()), fake_videos=int(df[fake].video.nunique()),
                   real_videos_by_src=df[real].groupby("src").video.nunique().to_dict(),
                   real_frames_by_src=df[real].src.value_counts().to_dict(),
                   methods=df[fake].method.value_counts().to_dict(),
                   fake_target_ids=sorted(df[fake].tgt_id.dropna().unique().tolist()),
                   fake_source_ids=sorted(df[fake].src_id.dropna().unique().tolist()),
                   groups_real=df[real].gname.value_counts().to_dict(), groups_fake=df[fake].gname.value_counts().to_dict(),
                   groups_real_videos=df[real].groupby("gname").video.nunique().to_dict(),
                   groups_fake_videos=df[fake].groupby("gname").video.nunique().to_dict())

_rv = df[df.label == 0].drop_duplicates("video")
_dis = _rv[(_rv.xu_male != _rv.cl_male_vid) | (_rv.xu_race != _rv.cl_race_vid)]
RES["xu_vs_cleaner_real_video_disagreements"] = dict(
    n=int(len(_dis)), of=int(len(_rv)),
    videos=_dis[["vid_short", "gname", "cl_male_vid", "cl_race_vid"]].to_dict("records"))
ev = json.load(open(f"{EV}/bd_eval.json"))
T = {k: float(v) for k, v in ev["thresholds"].items()}
RES["thresholds"] = T
df["ding_pred"] = df.ding_logit >= 0
df["ding_err"] = np.where(real, df.ding_pred, ~df.ding_pred)


# ---------------------------------------------------------------- metric (verified)
def parts():
    gender = {s: [i for i, n in enumerate(SIX) if n.split(",")[0] == s] for s in ("male", "nonmale")}
    race = {r: [i for i, n in enumerate(SIX) if n.split(",")[1] == r] for r in ("white", "black", "others")}
    inter = {n: [i] for i, n in enumerate(SIX)}
    return {"g": gender, "r": race, "i": inter}


PARTS = parts()


def f_fpr(y, pred, g, level, keep=None):
    """100 * sum_g |FPR_g - FPR_all| over groups with both classes; keep = boolean row mask."""
    if keep is not None:
        y, pred, g = y[keep], pred[keep], g[keep]
    r = y == 0
    fall = pred[r].mean()
    tot, per = 0.0, {}
    for name, ids in PARTS[level].items():
        mm = np.isin(g, ids)
        if (y[mm] == 0).sum() == 0 or (y[mm] == 1).sum() == 0:
            continue
        fg = pred[mm & r].mean()
        per[name] = (float(fg), int((mm & r).sum()))
        tot += abs(fg - fall)
    return 100 * tot, float(fall), per


Y, G = df.label.values, df.group.values
for name, t in T.items():
    ref = ev["test"]["celebdf"][name]["cols"]
    pred = df.logit.values >= t
    for lev in ("g", "r", "i"):
        v = f_fpr(Y, pred, G, lev)[0]
        assert abs(v - ref[f"{lev}F_FPR"]) < 1e-3, (name, lev, v, ref)
    assert abs(100 * pred[Y == 0].mean() - ref["FPR"]) < 1e-3 and abs(100 * pred[Y == 1].mean() - ref["TPR"]) < 1e-3
RES["metric_check"] = "F_FPR (g/r/i), FPR, TPR at matched/native/A1 reproduce bd_eval.json test.celebdf to 1e-3"


# ---------------------------------------------------------------- helpers
def rate_table(d, col, err, base_mask, n_label):
    sub = d[base_mask]
    g = sub.groupby(col, dropna=False)
    t = pd.DataFrame({n_label: g.size(), "videos": g.video.nunique(),
                      "errors": g[err].sum().astype(int), "err_videos": g.apply(lambda x: x.loc[x[err], "video"].nunique())})
    t["rate%"] = 100 * t.errors / t[n_label]
    t["ding_rate%"] = 100 * g.ding_err.mean()
    return t.reset_index().rename(columns={col: "value"}).sort_values("rate%", ascending=False)


def md(t, floatfmt=1):
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in t.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


FP_AXES = [("gname", "Xu intersectional group (metric)"), ("xu_gender", "Xu gender (metric)"),
           ("xu_race", "Xu race (metric)"), ("cl_gender", "AI-Face cleaner gender (frame)"),
           ("cl_race", "AI-Face cleaner race (frame)"), ("cl_race_vid", "AI-Face cleaner race (video majority)"),
           ("age4_if_name", "InsightFace age, video median"), ("ita4_name", "ITA skin tone, FF++-train quartile edges"),
           ("ita4_local_name", "ITA skin tone, Celeb-DF real-video quartiles"), ("ff_race7", "FairFace race7 (frame)"),
           ("ff_sex", "FairFace sex (frame)"), ("src", "real source"), ("frame_pos", "frame position in video")]
FN_AXES = [("gname", "Xu intersectional group (metric)"), ("xu_gender", "Xu gender"), ("xu_race", "Xu race"),
           ("cl_gender", "AI-Face cleaner gender (frame)"), ("cl_race", "AI-Face cleaner race (frame)"),
           ("age4_if_name", "InsightFace age, video median"), ("ita4_name", "ITA skin tone (FF++-train edges)"),
           ("ita4_local_name", "ITA skin tone (Celeb-DF quartiles)"), ("ff_race7", "FairFace race7 (frame)"),
           ("tgt_id", "target identity (video whose face is replaced)"), ("src_id", "source identity (face swapped in)"), ("frame_pos", "frame position in video")]

TAB = []  # markdown chunks
VCOLS = ["vid_short", "gname", "cl_gender_vid", "cl_race_vid", "age_vid_if", "ita_vid", "ita4_name"]

for name, t in T.items():
    R_ = {}
    d = df.copy()
    d["pred"] = d.logit >= t
    d["fp"] = real & d.pred
    d["fn"] = fake & ~d.pred
    d["err"] = d.fp | d.fn
    nfp, nfn = int(d.fp.sum()), int(d.fn.sum())
    R_["overall"] = dict(t=t, FP=nfp, FPR=100 * nfp / real.sum(), FN=nfn, FNR=100 * nfn / fake.sum(),
                         TPR=100 * (1 - nfn / fake.sum()),
                         ding_FP=int((real & d.ding_pred).sum()), ding_FPR=100 * (real & d.ding_pred).sum() / real.sum(),
                         ding_FN=int((fake & ~d.ding_pred).sum()), ding_FNR=100 * (fake & ~d.ding_pred).sum() / fake.sum(),
                         fp_videos=int(d[d.fp].video.nunique()), fn_videos=int(d[d.fn].video.nunique()))
    for lev in ("g", "r", "i"):
        v, fall, per = f_fpr(Y, d.pred.values, G, lev)
        R_[f"{lev}F_FPR"] = v
        R_[f"{lev}_groups"] = {k: dict(FPR=100 * a, n_real=b, term=100 * abs(a - fall)) for k, (a, b) in per.items()}
        dv, _, _ = f_fpr(Y, d.ding_pred.values, G, lev)
        R_[f"ding_{lev}F_FPR"] = dv

    # ---- 1. FP tables by axis
    fp_tabs = {}
    for col, title in FP_AXES:
        tt = rate_table(d.assign(err=d.fp), col, "err", real, "n_real")
        tt.to_csv(f"{OUT}/tables/{name}_fp_by_{col}.csv", index=False)
        fp_tabs[col] = tt
    fn_tabs = {}
    for col, title in FN_AXES:
        tt = rate_table(d.assign(err=d.fn), col, "err", fake, "n_fake")
        tt.to_csv(f"{OUT}/tables/{name}_fn_by_{col}.csv", index=False)
        fn_tabs[col] = tt
    R_["fp_by"] = {c: tt.set_index("value")[["n_real", "videos", "errors", "err_videos", "rate%", "ding_rate%"]].round(2).to_dict("index") for c, tt in fp_tabs.items()}
    R_["fn_by"] = {c: tt.set_index("value")[["n_fake", "videos", "errors", "err_videos", "rate%", "ding_rate%"]].round(2).to_dict("index") for c, tt in fn_tabs.items()}

    # ---- video concentration (FP)
    rv = d[real].groupby("video").agg(n=("fp", "size"), fp=("fp", "sum"), ding_fp=("ding_pred", "sum"),
                                      max_logit=("logit", "max"), med_logit=("logit", "median"),
                                      **{c: (c, "first") for c in VCOLS}).reset_index()
    rv["fp_rate%"] = 100 * rv.fp / rv.n
    rv["ding_fp_rate%"] = 100 * rv.ding_fp / rv.n
    rv = rv.sort_values(["fp", "max_logit"], ascending=False)
    rv.to_csv(f"{OUT}/tables/{name}_fp_by_video.csv", index=False)
    cs = rv.fp.cumsum().values / max(nfp, 1)
    R_["fp_video_concentration"] = dict(
        real_videos=int(len(rv)), videos_with_fp=int((rv.fp > 0).sum()),
        videos_fp_rate_ge50=int((rv["fp_rate%"] >= 50).sum()), videos_fp_rate_100=int((rv["fp_rate%"] >= 100).sum()),
        top_share={k: float(100 * cs[k - 1]) for k in (1, 3, 5, 10, 15, 20) if k <= len(cs)},
        n_videos_for_50pct=int(np.searchsorted(cs, 0.5) + 1) if nfp else 0,
        n_videos_for_80pct=int(np.searchsorted(cs, 0.8) + 1) if nfp else 0)
    R_["top15_fp_videos"] = rv.head(15).round(2).to_dict("records")
    # per-group: share of a group's FPs from its top-1/top-3 videos
    pg = {}
    for gn, sub in rv.groupby("gname"):
        s = sub.sort_values("fp", ascending=False)
        tot = s.fp.sum()
        pg[gn] = dict(real_videos=int(len(s)), fp=int(tot), videos_with_fp=int((s.fp > 0).sum()),
                      top1_share=float(100 * s.fp.iloc[0] / tot) if tot else 0.0,
                      top3_share=float(100 * s.fp.iloc[:3].sum() / tot) if tot else 0.0,
                      video_fp_rates=sorted(s["fp_rate%"].round(1).tolist(), reverse=True))
    R_["fp_by_group_videos"] = pg

    # ---- F_FPR (intersection, Xu) driven by how few videos: greedy removal of real videos
    vids_real = rv.video.values
    vid_arr = d.video.values
    pred_arr = d.pred.values
    keep = np.ones(len(d), bool)
    base_i = R_["iF_FPR"]
    steps = []
    removed = []
    for k in range(10):
        best = None
        for v in vids_real:
            if v in removed:
                continue
            kk = keep & (vid_arr != v)
            val = f_fpr(Y, pred_arr, G, "i", kk)[0]
            if best is None or val < best[1]:
                best = (v, val)
        removed.append(best[0])
        keep &= vid_arr != best[0]
        row = rv.set_index("video").loc[best[0]]
        steps.append(dict(k=k + 1, removed=best[0].replace("0_", "", 1), group=row.gname, fp=int(row.fp), n=int(row.n),
                          iF_FPR_after=float(best[1])))
    R_["iF_FPR_greedy_video_removal"] = dict(base=base_i, steps=steps,
                                            k_to_halve=next((s["k"] for s in steps if s["iF_FPR_after"] <= base_i / 2), None))
    # leave-one-video-out on the top FP videos (single deletions, not greedy)
    R_["iF_FPR_drop_single_top_fp_video"] = [
        dict(video=v.replace("0_", "", 1), iF_FPR=float(f_fpr(Y, pred_arr, G, "i", vid_arr != v)[0])) for v in rv.video.values[:5]]
    # null: permute Xu group labels across REAL videos (fakes fixed), 2000 perms
    rng = np.random.default_rng(0)
    vg = d[real].groupby("video").group.first()
    vlist, glist = vg.index.values, vg.values
    real_idx = np.where(real.values)[0]
    pos = pd.Index(vlist).get_indexer(vid_arr[real_idx])
    null = []
    for _ in range(2000):
        gp = rng.permutation(glist)
        G2 = G.copy()
        G2[real_idx] = gp[pos]
        null.append(f_fpr(Y, pred_arr, G2, "i")[0])
    null = np.array(null)
    R_["iF_FPR_video_perm_null"] = dict(observed=base_i, mean=float(null.mean()), p05=float(np.percentile(null, 5)),
                                       p50=float(np.median(null)), p95=float(np.percentile(null, 95)),
                                       frac_null_ge_obs=float((null >= base_i).mean()), n_perm=2000)

    # ---- FN by video / pair
    fv = d[fake].groupby("video").agg(n=("fn", "size"), fn=("fn", "sum"), ding_fn=("ding_pred", lambda x: int((~x).sum())),
                                      min_logit=("logit", "min"), med_logit=("logit", "median"), tgt_id=("tgt_id", "first"),
                                      src_id=("src_id", "first"), **{c: (c, "first") for c in VCOLS}).reset_index()
    fv["fn_rate%"] = 100 * fv.fn / fv.n
    fv["ding_fn_rate%"] = 100 * fv.ding_fn / fv.n
    fv = fv.sort_values(["fn", "min_logit"], ascending=[False, True])
    fv.to_csv(f"{OUT}/tables/{name}_fn_by_video.csv", index=False)
    cs2 = fv.fn.cumsum().values / max(nfn, 1)
    R_["fn_video_concentration"] = dict(fake_videos=int(len(fv)), videos_with_fn=int((fv.fn > 0).sum()),
                                        videos_fn_rate_ge50=int((fv["fn_rate%"] >= 50).sum()),
                                        top_share={k: float(100 * cs2[k - 1]) for k in (1, 5, 15, 50)},
                                        n_videos_for_50pct=int(np.searchsorted(cs2, 0.5) + 1) if nfn else 0)
    R_["top15_fn_videos"] = fv.head(15).round(2).to_dict("records")
    pt = rate_table(d.assign(err=d.fn), "pair", "err", fake, "n_fake")
    pt.to_csv(f"{OUT}/tables/{name}_fn_by_pair.csv", index=False)
    R_["fn_by_pair_top10"] = pt.head(10).round(2).to_dict("records")
    R_["fn_pairs_rate_ge50"] = int((pt["rate%"] >= 50).sum())
    R_["fn_pairs_total"] = int(len(pt))

    # ---- 3. shared vs unique with Ding
    fpm, fnm = d.fp, d.fn
    R_["shared"] = dict(
        fp_also_ding_fp=100 * float(d.loc[fpm, "ding_pred"].mean()) if nfp else None,
        ding_fpr_all_reals=100 * float(d.loc[real, "ding_pred"].mean()),
        ding_fp_also_ours=100 * float(d.loc[real & d.ding_pred, "fp"].mean()),
        fp_ours_only=int((fpm & ~d.ding_pred).sum()), fp_ding_only=int((real & d.ding_pred & ~d.pred).sum()),
        fp_both=int((fpm & d.ding_pred).sum()),
        fn_also_ding_fn=100 * float((~d.loc[fnm, "ding_pred"]).mean()) if nfn else None,
        ding_fnr_all_fakes=100 * float((~d.loc[fake, "ding_pred"]).mean()),
        ding_fn_also_ours=100 * float(d.loc[fake & ~d.ding_pred, "fn"].mean()),
        fn_ours_only=int((fnm & d.ding_pred).sum()), fn_ding_only=int((fake & ~d.ding_pred & d.pred).sum()),
        fn_both=int((fnm & ~d.ding_pred).sum()),
        # video-level: real videos where >=50% frames are FP
        fp_videos50_ours=int((rv["fp_rate%"] >= 50).sum()),
        fp_videos50_both=int(((rv["fp_rate%"] >= 50) & (rv["ding_fp_rate%"] >= 50)).sum()),
        fp_videos50_ding=int((rv["ding_fp_rate%"] >= 50).sum()),
        fn_videos50_ours=int((fv["fn_rate%"] >= 50).sum()),
        fn_videos50_both=int(((fv["fn_rate%"] >= 50) & (fv["ding_fn_rate%"] >= 50)).sum()),
        fn_videos50_ding=int((fv["ding_fn_rate%"] >= 50).sum()))
    # Ding-shared FP fraction by group (is the overlap uniform?)
    R_["shared_fp_by_group"] = {gn: dict(fp=int(s.fp.sum()), also_ding=100 * float(s.loc[s.fp, "ding_pred"].mean()) if s.fp.any() else None)
                                for gn, s in d[real].groupby("gname")}
    cols_case = ["vid_short", "gname", "cl_gender", "cl_race", "if_age", "ita_gw", "logit", "ding_logit", "img_path"]
    ours_only_fp = d[fpm & ~d.ding_pred].sort_values("logit", ascending=False)
    ding_only_fp = d[real & d.ding_pred & ~d.pred].sort_values("ding_logit", ascending=False)
    ours_only_fn = d[fnm & d.ding_pred].sort_values("logit")
    ding_only_fn = d[fake & ~d.ding_pred & d.pred].sort_values("ding_logit")
    for nm, s in (("ours_only_fp", ours_only_fp), ("ding_only_fp", ding_only_fp), ("ours_only_fn", ours_only_fn), ("ding_only_fn", ding_only_fn)):
        s[cols_case + (["tgt_id", "src_id"] if "fn" in nm else [])].to_csv(f"{OUT}/tables/{name}_{nm}.csv", index=False)
        R_[f"{nm}_n"] = int(len(s))
        R_[f"{nm}_videos_top"] = s.groupby("vid_short").size().sort_values(ascending=False).head(8).to_dict()
    R_["ours_only_fp_top12"] = ours_only_fp[cols_case].head(12).round(2).to_dict("records")
    R_["ding_only_fp_top12"] = ding_only_fp[cols_case].head(12).round(2).to_dict("records")
    R_["ours_only_fn_top12"] = ours_only_fn[cols_case + ["tgt_id", "src_id"]].head(12).round(2).to_dict("records")
    R_["ding_only_fn_top12"] = ding_only_fn[cols_case + ["tgt_id", "src_id"]].head(12).round(2).to_dict("records")
    # video-level unique: real videos we flag >=50% and Ding < 50%, and vice versa
    R_["fp_videos_ours_not_ding"] = rv[(rv["fp_rate%"] >= 50) & (rv["ding_fp_rate%"] < 50)][["vid_short", "gname", "fp_rate%", "ding_fp_rate%"]].round(1).to_dict("records")
    R_["fp_videos_ding_not_ours"] = int(((rv["fp_rate%"] < 50) & (rv["ding_fp_rate%"] >= 50)).sum())
    R_["fn_videos_ours_not_ding"] = fv[(fv["fn_rate%"] >= 50) & (fv["ding_fn_rate%"] < 50)][["vid_short", "gname", "fn_rate%", "ding_fn_rate%"]].round(1).to_dict("records")
    R_["fn_videos_ding_not_ours"] = fv[(fv["fn_rate%"] < 50) & (fv["ding_fn_rate%"] >= 50)][["vid_short", "gname", "fn_rate%", "ding_fn_rate%"]].round(1).to_dict("records")

    # ---- 4. confidence
    R_["top12_fp"] = d[fpm].sort_values("logit", ascending=False)[cols_case].head(12).round(2).to_dict("records")
    R_["top12_fn"] = d[fnm].sort_values("logit")[cols_case + ["tgt_id", "src_id"]].head(12).round(2).to_dict("records")
    RES[name] = R_
    d["fp"], d["fn"] = fpm, fnm
    if name == "matched":
        DM = d

# ---------------------------------------------------------------- logit scale context
RES["logit_quantiles"] = {
    "real": {q: float(np.percentile(df.loc[real, "logit"], q)) for q in (5, 25, 50, 75, 95)},
    "fake": {q: float(np.percentile(df.loc[fake, "logit"], q)) for q in (5, 25, 50, 75, 95)},
    "real_by_src_median": df[real].groupby("src").logit.median().to_dict()}

# ---------------------------------------------------------------- score shift by group / identity
from sklearn.metrics import roc_auc_score  # noqa: E402
sg = {}
for gn, s in df.groupby("gname"):
    e = dict(n_real=int((s.label == 0).sum()), n_fake=int((s.label == 1).sum()),
             real_median_logit=float(s.loc[s.label == 0, "logit"].median()) if (s.label == 0).any() else None,
             fake_median_logit=float(s.loc[s.label == 1, "logit"].median()) if (s.label == 1).any() else None,
             ding_real_median=float(s.loc[s.label == 0, "ding_logit"].median()) if (s.label == 0).any() else None,
             ding_fake_median=float(s.loc[s.label == 1, "ding_logit"].median()) if (s.label == 1).any() else None)
    if s.label.nunique() == 2:
        e["auc"] = 100 * float(roc_auc_score(s.label, s.logit))
        e["ding_auc"] = 100 * float(roc_auc_score(s.label, s.ding_logit))
    sg[gn] = e
RES["score_by_group"] = sg
dm = DM
idt = dm[real].assign(ident=np.where(dm[real].src == "Celeb-real", dm[real].tgt_id, "YouTube (51 people)")).groupby("ident").agg(
    n_real=("fp", "size"), videos=("video", "nunique"), fp=("fp", "sum"), ding_fp=("ding_pred", "sum"),
    median_logit=("logit", "median"), gname=("gname", lambda x: x.iloc[0] if x.nunique() == 1 else "mixed"))
idt["fp_rate%"] = 100 * idt.fp / idt.n_real
idt["ding_fp_rate%"] = 100 * idt.ding_fp / idt.n_real
idt = idt.sort_values("fp_rate%", ascending=False)
idt.to_csv(f"{OUT}/tables/matched_fp_by_real_identity.csv")
RES["matched_fp_by_real_identity"] = idt.round(2).to_dict("index")
fnb = dm[dm.fn]
RES["matched_fn_mblack"] = dict(fn=int((fnb.gname == "M-Black").sum()), share_of_fn=100 * float((fnb.gname == "M-Black").mean()),
                                share_of_fakes=100 * float((dm[fake].gname == "M-Black").mean()),
                                also_ding=100 * float((~fnb.loc[fnb.gname == "M-Black", "ding_pred"]).mean()),
                                fw_also_ding=100 * float((~fnb.loc[fnb.gname == "F-White", "ding_pred"]).mean()))
fpb = dm[dm.fp]
RES["matched_fp_ding_shared_by_src"] = {k: 100 * float(v.ding_pred.mean()) for k, v in fpb.groupby("src")}

# ---------------------------------------------------------------- 5. contact sheets
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def sheet(rows, title, path, kind, t):
    n = len(rows)
    nc = 6
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.0, nr * 4.1), gridspec_kw=dict(hspace=0.55, wspace=0.08))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        ax.imshow(Image.open(r.img_path).convert("RGB"))
        dec = "FAKE" if r.ding_logit >= 0 else "real"
        ok = (dec == "real") if kind == "fp" else (dec == "FAKE")
        rate = r["vrate"]
        cap = (f"{r.vid_short}\n{r.gname} | {r.method}\n"
               f"BD logit {r.logit:.1f} (t {t:.1f})\n"
               f"Ding: {dec} (z {r.ding_logit:.1f}){' OK' if ok else ''}\n"
               f"video {kind.upper()} rate {rate:.0f}%")
        ax.set_title(cap, fontsize=8.5, loc="left", family="monospace")
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.01)
    fig.savefig(path, dpi=110)
    plt.close(fig)


d = DM
t = T["matched"]
vr = d[real].groupby("video").fp.mean() * 100
fpf = d[d.fp].sort_values("logit", ascending=False).drop_duplicates("video").head(24).copy()
fpf["vrate"] = fpf.video.map(vr)
sheet(fpf, f"BD0 s7, Celeb-DF, matched threshold: top-24 false positives (real flagged fake), one frame per video, "
           f"highest-logit frame", f"{OUT}/fp_sheet.png", "fp", t)
vr2 = d[fake].groupby("video").fn.mean() * 100
fnf = d[d.fn].sort_values("logit").drop_duplicates("video").head(24).copy()
fnf["vrate"] = fnf.video.map(vr2)
sheet(fnf, f"BD0 s7, Celeb-DF, matched threshold: top-24 false negatives (fake missed), one frame per video, "
           f"lowest-logit frame", f"{OUT}/fn_sheet.png", "fn", t)
RES["sheets"] = dict(fp=fpf[["vid_short", "gname", "logit", "ding_logit", "vrate", "img_path"]].round(2).to_dict("records"),
                     fn=fnf[["vid_short", "gname", "tgt_id", "src_id", "logit", "ding_logit", "vrate", "img_path"]].round(2).to_dict("records"))

# ---------------------------------------------------------------- derived numbers quoted in FAILURES.md
_m = RES["matched"]
RES["derived"] = dict(
    first_frame_share_of_matched_FN=100 * _m["fn_by"]["frame_pos"]["first frame (idx 1)"]["errors"] / _m["overall"]["FN"],
    fn_overlap_lift_matched=_m["shared"]["fn_also_ding_fn"] / _m["shared"]["ding_fnr_all_fakes"],
    fp_overlap_lift_matched=_m["shared"]["fp_also_ding_fp"] / _m["shared"]["ding_fpr_all_reals"],
    fp_videos_ours_not_ding_by_group=pd.Series([r["gname"] for r in _m["fp_videos_ours_not_ding"]]).value_counts().to_dict(),
    fn_videos_ours_not_ding_tgt=pd.Series([re.search(r"synthesis_(id\d+)_", r["vid_short"]).group(1) for r in _m["fn_videos_ours_not_ding"]]).value_counts().to_dict(),
    top12_fp_from_YouTube_real_00247=int(sum(r["vid_short"] == "YouTube-real_00247" for r in _m["top12_fp"])),
    top12_fn_MBlack=int(sum(r["gname"] == "M-Black" for r in _m["top12_fn"])),
    sheet_fp_groups=pd.Series([r["gname"] for r in RES["sheets"]["fp"]]).value_counts().to_dict(),
    sheet_fn_groups=pd.Series([r["gname"] for r in RES["sheets"]["fn"]]).value_counts().to_dict(),
    sheet_fp_ding_flags=int(sum(r["ding_logit"] >= 0 for r in RES["sheets"]["fp"])),
    sheet_fn_ding_detects=int(sum(r["ding_logit"] >= 0 for r in RES["sheets"]["fn"])),
    iF_FPR_change_drop_top1_video=max(abs(x["iF_FPR"] - _m["iF_FPR"]) for x in _m["iF_FPR_drop_single_top_fp_video"]))
print(RES["derived"])
json.dump(RES, open(f"{OUT}/failures.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------------------------------------------------------- auto tables for FAILURES.md
L = []
L.append("## A. Operating points\n")
rows = []
for name in T:
    o = RES[name]["overall"]
    rows.append(dict(threshold=name, t=round(o["t"], 2), FP=o["FP"], FPR=o["FPR"], FN=o["FN"], FNR=o["FNR"],
                     FP_videos=o["fp_videos"], FN_videos=o["fn_videos"], gF_FPR=RES[name]["gF_FPR"],
                     rF_FPR=RES[name]["rF_FPR"], iF_FPR=RES[name]["iF_FPR"]))
o = RES["matched"]["overall"]
rows.append(dict(threshold="Ding repro (z>=0)", t=0.0, FP=o["ding_FP"], FPR=o["ding_FPR"], FN=o["ding_FN"], FNR=o["ding_FNR"],
                 FP_videos="", FN_videos="", gF_FPR=RES["matched"]["ding_gF_FPR"], rF_FPR=RES["matched"]["ding_rF_FPR"],
                 iF_FPR=RES["matched"]["ding_iF_FPR"]))
L.append(md(pd.DataFrame(rows), 2))
L.append(f"\nReal frames {RES['data']['n_real']} in {RES['data']['real_videos']} videos "
         f"({RES['data']['real_videos_by_src']}); fake frames {RES['data']['n_fake']} in {RES['data']['fake_videos']} videos, "
         f"one method ({RES['data']['methods']}).\n")

for name in T:
    L.append(f"\n## B-{name}. False positives by axis ({name}, t = {T[name]:.2f})\n")
    for col, title in FP_AXES:
        tt = pd.read_csv(f"{OUT}/tables/{name}_fp_by_{col}.csv")
        L.append(f"\n**{title}**\n\n" + md(tt))
    if name != "matched":
        continue
for name in T:
    R_ = RES[name]
    L.append(f"\n## C-{name}. FP concentration by video ({name})\n")
    c = R_["fp_video_concentration"]
    L.append(f"{c['videos_with_fp']}/{c['real_videos']} real videos have >= 1 FP frame; {c['videos_fp_rate_ge50']} have FP rate >= 50%; "
             f"{c['videos_fp_rate_100']} are flagged on every frame. Top-k videos' share of all FP frames: "
             + ", ".join(f"top-{k} {v:.1f}%" for k, v in c["top_share"].items())
             + f". Videos needed for 50% / 80% of FPs: {c['n_videos_for_50pct']} / {c['n_videos_for_80pct']}.\n")
    tv = pd.DataFrame(R_["top15_fp_videos"])[["vid_short", "gname", "cl_gender_vid", "cl_race_vid", "age_vid_if", "ita_vid",
                                             "n", "fp", "fp_rate%", "ding_fp_rate%", "max_logit"]]
    L.append("\nTop-15 real videos by FP frames\n\n" + md(tv))
    gr = R_["iF_FPR_greedy_video_removal"]
    L.append(f"\nIntersectional F_FPR = {gr['base']:.2f}. Greedy removal of the real video that most reduces it:\n\n"
             + md(pd.DataFrame(gr["steps"]), 2)
             + f"\n\nVideos to remove to halve iF_FPR: {gr['k_to_halve']}.\n")
    nl = R_["iF_FPR_video_perm_null"]
    L.append(f"\nPermutation null (Xu group labels shuffled across the {c['real_videos']} real videos, fakes fixed, "
             f"{nl['n_perm']} perms): mean {nl['mean']:.2f}, 5-95% [{nl['p05']:.2f}, {nl['p95']:.2f}]; "
             f"observed {nl['observed']:.2f}; P(null >= observed) = {nl['frac_null_ge_obs']:.3f}.\n")
    pg = pd.DataFrame(R_["fp_by_group_videos"]).T.reset_index().rename(columns={"index": "group"})
    pg["video_fp_rates"] = pg.video_fp_rates.map(lambda v: ", ".join(f"{x:.0f}" for x in v[:12]) + (" ..." if len(v) > 12 else ""))
    L.append("\nPer-group FP video structure (video FP rates, %, descending)\n\n" + md(pg))
    ig = pd.DataFrame(R_["i_groups"]).T.reset_index().rename(columns={"index": "group (both classes)"})
    L.append("\nGroup terms of intersectional F_FPR (|FPR_g - FPR_all|, %)\n\n" + md(ig, 2))

for name in T:
    R_ = RES[name]
    L.append(f"\n## D-{name}. False negatives ({name})\n")
    for col, title in FN_AXES:
        tt = pd.read_csv(f"{OUT}/tables/{name}_fn_by_{col}.csv")
        L.append(f"\n**{title}**\n\n" + md(tt))
    c = R_["fn_video_concentration"]
    L.append(f"\n{c['videos_with_fn']}/{c['fake_videos']} fake videos have >= 1 FN frame; {c['videos_fn_rate_ge50']} have FN rate >= 50%. "
             + ", ".join(f"top-{k} {v:.1f}%" for k, v in c["top_share"].items()) + " of FN frames.\n")
    tv = pd.DataFrame(R_["top15_fn_videos"])[["vid_short", "gname", "tgt_id", "src_id", "cl_race_vid", "n", "fn", "fn_rate%", "ding_fn_rate%", "min_logit"]]
    L.append("\nTop-15 fake videos by FN frames\n\n" + md(tv))
    L.append(f"\nSwap pairs with FN rate >= 50%: {R_['fn_pairs_rate_ge50']}/{R_['fn_pairs_total']}. Top-10 pairs:\n\n"
             + md(pd.DataFrame(R_["fn_by_pair_top10"])))

L.append("\n## G. Score shift by group (all thresholds; logit medians)\n\n" + md(pd.DataFrame(RES["score_by_group"]).T.reset_index().rename(columns={"index": "group"}), 1))
L.append("\n## H. Matched FP by real identity (Celeb-real ids; YouTube pooled)\n\n" + md(pd.read_csv(f"{OUT}/tables/matched_fp_by_real_identity.csv"), 1))
L.append(f"\nM-Black fakes at matched: {RES['matched_fn_mblack']}\n")
L.append("\n## E. Shared vs unique with the Ding reproduction (Ding rule z >= 0)\n")
rows = []
for name in T:
    s = RES[name]["shared"]
    rows.append(dict(threshold=name, our_FP=RES[name]["overall"]["FP"], FP_also_Ding=s["fp_also_ding_fp"],
                     Ding_FPR_base=s["ding_fpr_all_reals"], FP_ours_only=s["fp_ours_only"], FP_Ding_only=s["fp_ding_only"],
                     our_FN=RES[name]["overall"]["FN"], FN_also_Ding=s["fn_also_ding_fn"], Ding_FNR_base=s["ding_fnr_all_fakes"],
                     FN_ours_only=s["fn_ours_only"], FN_Ding_only=s["fn_ding_only"],
                     FPvid50_ours_both_ding=f"{s['fp_videos50_ours']}/{s['fp_videos50_both']}/{s['fp_videos50_ding']}",
                     FNvid50_ours_both_ding=f"{s['fn_videos50_ours']}/{s['fn_videos50_both']}/{s['fn_videos50_ding']}"))
L.append(md(pd.DataFrame(rows), 1))
L.append("\n(*_also_Ding: % of our errors on which Ding is also wrong; *_base: Ding's error rate on all reals/fakes, the chance "
         "overlap; vid50 = videos with >= 50% erroneous frames, ours / both / Ding.)\n")
for name in ("matched", "A1"):
    R_ = RES[name]
    L.append(f"\n**{name}: real videos we flag on >= 50% of frames while Ding flags < 50%** ({len(R_['fp_videos_ours_not_ding'])}): "
             + "; ".join(f"{r['vid_short']} {r['gname']} {r['fp_rate%']:.0f}% vs {r['ding_fp_rate%']:.0f}%" for r in R_["fp_videos_ours_not_ding"][:15]))
    L.append(f"\n\n**{name}: fake videos we miss on >= 50% of frames while Ding misses < 50%** ({len(R_['fn_videos_ours_not_ding'])}): "
             + "; ".join(f"{r['vid_short']} {r['fn_rate%']:.0f}% vs {r['ding_fn_rate%']:.0f}%" for r in R_["fn_videos_ours_not_ding"][:15]))
    L.append(f"\n\n**{name}: fake videos Ding misses on >= 50% while we miss < 50%** ({len(R_['fn_videos_ding_not_ours'])}): "
             + "; ".join(f"{r['vid_short']} {r['ding_fn_rate%']:.0f}% vs ours {r['fn_rate%']:.0f}%" for r in R_["fn_videos_ding_not_ours"][:15]) + "\n")
    for nm, title in (("ours_only_fp_top12", "FP where we fail and Ding does not (highest BD logit)"),
                      ("ding_only_fp_top12", "FP where Ding fails and we do not (highest Ding logit)"),
                      ("ours_only_fn_top12", "FN where we fail and Ding does not (lowest BD logit)"),
                      ("ding_only_fn_top12", "FN where Ding fails and we do not (lowest Ding logit)")):
        tt = pd.DataFrame(R_[nm])
        if len(tt):
            tt["frame"] = tt.img_path.map(os.path.basename)
            tt = tt.drop(columns=["img_path"])
        L.append(f"\n**{name}: {title}** (n = {R_[nm.replace('_top12', '_n')]})\n\n" + (md(tt, 1) if len(tt) else "none"))

L.append("\n## F. Most confident errors (matched)\n")
for nm, title in (("top12_fp", "Top-12 FP (highest BD logit)"), ("top12_fn", "Top-12 FN (lowest BD logit)")):
    tt = pd.DataFrame(RES["matched"][nm])
    L.append(f"\n**{title}** (full paths in failures.json and tables/)\n\n" + md(tt, 1))
open(f"{OUT}/FAILURES_tables.md", "w").write("\n".join(L) + "\n")
print("done")
