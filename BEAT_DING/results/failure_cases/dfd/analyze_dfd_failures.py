"""Failure-case analysis of BD0 seed 7 on DFD (cross-domain).

  ~/.conda/envs/aiface/bin/python analyze_dfd_failures.py

Read-only on every input; writes only into this folder. CPU only.
Decisions: ours = logit >= t (t from runs/BD0_s7/eval/bd_eval.json 'thresholds': matched [primary], native, A1).
Ding reproduction (official code, Xception, seed 7, ep49) = logit >= 0 (Ding's 0.5 rule on P(fake)).
DFD 'six' group scheme (cfes/data.py SCHEMES): 0 M-White, 1 M-Black, 2 M-Others, 3 F-White, 4 F-Black, 5 F-Others.
Note: the npz 'method' column on DFD is NOT a manipulation method. DFD has one manipulation family (face swap);
cfes.data.ffpp_method() parses the first actor id of '<a>_<b>__<scene>__<hash>' into 'method'. We therefore break
fakes down by actor ids (id_a = 'method', id_b) and scene instead.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BD = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING"
AG = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
OUT = f"{BD}/results/failure_cases/dfd"
sys.path.insert(0, BD)
from cfes.metrics import fairness_block  # noqa: E402

FONT = "/home/sachin.chaudhary/.conda/envs/aiface/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
GNAME = {0: "M-White", 1: "M-Black", 2: "M-Others", 3: "F-White", 4: "F-Black", 5: "F-Others"}

# ---------------------------------------------------------------- load
d = np.load(f"{BD}/runs/BD0_s7/eval/dfd_scores.npz", allow_pickle=True)
e = np.load(f"{AG}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_dfd.npz", allow_pickle=True)
L = pd.read_csv(f"{AG}/DEMO_AXES/labels/labels_dfd.csv")
FF = pd.read_csv(f"{AG}/DEMO_AXES/labels_fairface/dfd.csv")
TH = json.load(open(f"{BD}/runs/BD0_s7/eval/bd_eval.json"))["thresholds"]

assert (e["img_path"] == d["img_path"]).all() and (e["label"] == d["label"]).all() and (e["group"] == d["group"]).all()
assert (L.img_path.values == d["img_path"]).all() and (FF.img_path.values == d["img_path"]).all()
assert (L.xu_group.values == d["group"]).all()

df = pd.DataFrame({k: d[k] for k in ("logit", "label", "group", "method", "cluster", "img_path")})
df["logit"] = df.logit.astype(np.float64)
df["ding_logit"] = e["logit"].astype(np.float64)
df["video"] = df.cluster
df["gname"] = df.group.map(GNAME)
for c in ("xu_male", "xu_race", "cl_male", "cl_race", "cl_male_vid", "cl_race_vid", "cl_age_v2", "if_age",
          "age_vid_if", "age4_if", "age4_aif", "ita_vid", "ita4", "ita4_local", "aif_mst3", "person"):
    df[c] = L[c].values
for c in ("sex", "race2", "race7", "age4", "vid_sex", "vid_race2", "vid_age4"):
    df["ff_" + c] = FF[c].values
df["xu_gender"] = np.where(df.xu_male == 1, "male", "nonmale")
df["cl_gender"] = np.where(df.cl_male == 1, "male", "nonmale")
df["cl_gender_vid"] = np.where(df.cl_male_vid == 1, "male", "nonmale")
df["age4_if"] = df.age4_if.map({0: "<30", 1: "30-44", 2: "45-59", 3: "60+"})
df["age4_aif"] = df.age4_aif.map({0: "<25", 1: "25-44", 2: "45-64", 3: "65+"})
df["cl_age_v2"] = df.cl_age_v2.map({0: "young", 1: "middle", 2: "senior", 3: "others"})
df["ita4"] = df.ita4.map({0: "Q1 darkest", 1: "Q2", 2: "Q3", 3: "Q4 lightest"})
df["ita4_local"] = df.ita4_local.map({0: "Q1 darkest", 1: "Q2", 2: "Q3", 3: "Q4 lightest"})
df["aif_mst3"] = df.aif_mst3.map({0: "MST1-3", 1: "MST4-6", 2: "MST7-10"})

# DFD name parsing: real '0_<a>__<scene>', fake '1_<a>_<b>__<scene>__<hash>'
def parse(v):
    body = v[2:]
    parts = body.split("__")
    ids = parts[0].split("_")
    return pd.Series({"id_a": ids[0], "id_b": ids[1] if len(ids) > 1 else "", "scene": parts[1] if len(parts) > 1 else ""})
df = pd.concat([df, df.video.apply(parse)], axis=1)
assert (df.loc[df.label == 1, "method"] == df.loc[df.label == 1, "id_a"]).all()

y = df.label.values
real, fake = y == 0, y == 1
DING = df.ding_logit.values >= 0.0
df["ding_pred"] = DING
OPS = {k: float(TH[k]) for k in ("matched", "native", "A1")}
for k, t in OPS.items():
    df["pred_" + k] = df.logit.values >= t
df["fp"] = real & df.pred_matched.values
df["fn"] = fake & ~df.pred_matched.values
df["ding_fp"] = real & DING
df["ding_fn"] = fake & ~DING

S = {"thresholds": OPS, "n": int(len(df)), "n_real": int(real.sum()), "n_fake": int(fake.sum()),
     "n_real_videos": int(df[real].video.nunique()), "n_fake_videos": int(df[fake].video.nunique())}

# ---------------------------------------------------------------- overall rates + reproduction check
def rates(pred):
    return dict(FP=int((pred & real).sum()), FPR=100 * float((pred & real).sum() / real.sum()),
                FN=int((~pred & fake).sum()), FNR=100 * float((~pred & fake).sum() / fake.sum()),
                TPR=100 * float((pred & fake).sum() / fake.sum()))
S["overall"] = {k: rates(df["pred_" + k].values) for k in OPS}
S["overall"]["ding_0.5"] = rates(DING)
fb = {k: fairness_block(y, df.logit.values, df.group.values, "six", t) for k, t in OPS.items()}
fb["ding_0.5"] = fairness_block(y, df.ding_logit.values, df.group.values, "six", 0.0)
S["fairness_FPR"] = {k: {lv: {"F_FPR": v[lv].get("F_FPR"), "fpr_gap": v[lv].get("fpr_gap"),
                              "group_FPR": {g: 100 * s["FPR"] for g, s in v[lv]["groups"].items()}}
                         for lv in ("gender", "race", "intersection")} for k, v in fb.items()}

# ---------------------------------------------------------------- breakdown tables
AXES = [("xu_group(metric)", "gname"), ("xu_gender(metric)", "xu_gender"), ("xu_race(metric)", "xu_race"),
        ("cl_gender(AI-Face frame)", "cl_gender"), ("cl_race(AI-Face frame)", "cl_race"),
        ("cl_gender_vid(AI-Face video-maj)", "cl_gender_vid"), ("cl_race_vid(AI-Face video-maj)", "cl_race_vid"),
        ("fairface_sex(video)", "ff_vid_sex"), ("fairface_race2(video)", "ff_vid_race2"),
        ("age4_if(InsightFace video median)", "age4_if"), ("age4_aif(AI-Face video maj)", "age4_aif"),
        ("cl_age_v2(AI-Face)", "cl_age_v2"), ("ita4(declared FF++-train edges)", "ita4"),
        ("ita4_local(DFD own real quartiles)", "ita4_local"), ("aif_mst3(AI-Face MST)", "aif_mst3")]


def breakdown(col):
    rows = []
    for b, g in df.groupby(col, dropna=False):
        r, f = g[g.label == 0], g[g.label == 1]
        row = {"bin": b, "n_real": len(r), "real_videos": r.video.nunique(), "n_fake": len(f), "fake_videos": f.video.nunique()}
        if len(r):
            row["FP"] = int(r.fp.sum())
            row["FP_videos"] = int(r[r.fp].video.nunique())
            for k in OPS:
                row[f"FPR_{k}"] = 100 * r["pred_" + k].mean()
            row["FPR_ding"] = 100 * r.ding_pred.mean()
        if len(f):
            row["FN"] = int(f.fn.sum())
            row["FN_videos"] = int(f[f.fn].video.nunique())
            for k in OPS:
                row[f"FNR_{k}"] = 100 * (~f["pred_" + k]).mean()
            row["FNR_ding"] = 100 * (~f.ding_pred).mean()
        rows.append(row)
    return pd.DataFrame(rows)


tabs = []
for name, col in AXES:
    t = breakdown(col)
    t.insert(0, "axis", name)
    tabs.append(t)
axes_tab = pd.concat(tabs, ignore_index=True)
axes_tab.round(2).to_csv(f"{OUT}/breakdown_by_attribute.csv", index=False)

# ---------------------------------------------------------------- video-level tables
vid = df.groupby("video").agg(label=("label", "first"), n=("label", "size"), group=("gname", "first"),
                              group_mixed=("gname", "nunique"), cl_gender=("cl_gender_vid", "first"),
                              cl_race=("cl_race_vid", "first"), age_vid_if=("age_vid_if", "first"),
                              ita_vid=("ita_vid", "first"), ita4=("ita4", "first"), id_a=("id_a", "first"),
                              id_b=("id_b", "first"), scene=("scene", "first"),
                              fp=("fp", "sum"), fn=("fn", "sum"), ding_fp=("ding_fp", "sum"), ding_fn=("ding_fn", "sum"),
                              fp_native=("pred_native", "sum"), fp_A1=("pred_A1", "sum"),
                              max_logit=("logit", "max"), min_logit=("logit", "min"),
                              med_logit=("logit", "median"), ding_med_logit=("ding_logit", "median")).reset_index()
vr = vid[vid.label == 0].copy()
vr["fp_rate"] = 100 * vr.fp / vr.n
vr["ding_fp_rate"] = 100 * vr.ding_fp / vr.n
vr["fp_rate_native"] = 100 * vr.fp_native / vr.n
vr["fp_rate_A1"] = 100 * vr.fp_A1 / vr.n
vr = vr.sort_values(["fp", "fp_rate"], ascending=False)
vf = vid[vid.label == 1].copy()
vf["fn_rate"] = 100 * vf.fn / vf.n
vf["ding_fn_rate"] = 100 * vf.ding_fn / vf.n
vf = vf.sort_values(["fn", "fn_rate"], ascending=False)
keep_r = ["video", "n", "fp", "fp_rate", "fp_rate_native", "fp_rate_A1", "ding_fp", "ding_fp_rate", "group",
          "cl_gender", "cl_race", "age_vid_if", "ita_vid", "ita4", "scene", "med_logit", "ding_med_logit"]
keep_f = ["video", "n", "fn", "fn_rate", "ding_fn", "ding_fn_rate", "group", "cl_gender", "cl_race", "age_vid_if",
          "ita4", "id_a", "id_b", "scene", "med_logit", "ding_med_logit"]
vr[keep_r].round(2).to_csv(f"{OUT}/fp_by_video.csv", index=False)
vf[keep_f].round(2).to_csv(f"{OUT}/fn_by_video.csv", index=False)

fp_tot, fn_tot = int(df.fp.sum()), int(df.fn.sum())
cum = np.cumsum(vr.fp.values) / fp_tot
cumn = np.cumsum(vf.fn.values) / fn_tot
S["fp_concentration"] = {
    "videos_with_any_fp": int((vr.fp > 0).sum()), "real_videos": len(vr),
    "share_top5": 100 * float(cum[4]), "share_top10": 100 * float(cum[9]), "share_top15": 100 * float(cum[14]),
    "n_videos_for_50pct": int(np.searchsorted(cum, 0.5) + 1), "n_videos_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
    "videos_fp_rate_ge_90": int((vr.fp_rate >= 90).sum()), "videos_fp_rate_ge_50": int((vr.fp_rate >= 50).sum()),
    "videos_fp_rate_0": int((vr.fp_rate == 0).sum()),
    "frames_per_real_video_median": float(vr.n.median())}
S["fn_concentration"] = {
    "videos_with_any_fn": int((vf.fn > 0).sum()), "fake_videos": len(vf),
    "share_top5": 100 * float(cumn[4]), "share_top10": 100 * float(cumn[9]), "share_top15": 100 * float(cumn[14]),
    "n_videos_for_50pct": int(np.searchsorted(cumn, 0.5) + 1), "n_videos_for_80pct": int(np.searchsorted(cumn, 0.8) + 1),
    "videos_fn_rate_ge_50": int((vf.fn_rate >= 50).sum())}

# real videos per group (support of the FPR terms)
S["real_videos_per_group"] = {GNAME[g]: int(df[(df.label == 0) & (df.group == g)].video.nunique()) for g in range(6)}
S["real_frames_per_group"] = {GNAME[g]: int(((df.label == 0) & (df.group == g)).sum()) for g in range(6)}
S["fake_videos_per_group"] = {GNAME[g]: int(df[(df.label == 1) & (df.group == g)].video.nunique()) for g in range(6)}
S["videos_with_mixed_group"] = int((vid.group_mixed > 1).sum())

# ---------------------------------------------------------------- F_FPR driven by how few videos (matched)
def iffpr(mask_keep, t):
    m = mask_keep
    return fairness_block(y[m], df.logit.values[m], df.group.values[m], "six", t)["intersection"]["F_FPR"]


t_m = OPS["matched"]
base = iffpr(np.ones(len(df), bool), t_m)
real_videos = vr.video.tolist()
removed, path, keep = [], [{"k": 0, "iF_FPR": base}], np.ones(len(df), bool)
for k in range(1, 11):  # greedy: drop the real video whose removal lowers intersection F_FPR most
    best = None
    for v in real_videos:
        if v in removed:
            continue
        m = keep & (df.video.values != v)
        val = iffpr(m, t_m)
        if best is None or val < best[1]:
            best = (v, val)
    removed.append(best[0])
    keep &= df.video.values != best[0]
    vrow = vr.set_index("video").loc[best[0]]
    path.append({"k": k, "removed": best[0], "group": vrow.group, "n": int(vrow.n), "fp_rate": float(vrow.fp_rate),
                 "iF_FPR": best[1]})
S["iF_FPR_greedy_video_removal"] = path
# naive: drop the top-k real videos by FP count
naive = []
for k in (1, 3, 5, 10, 15):
    drop = set(vr.video.iloc[:k])
    naive.append({"k": k, "iF_FPR": iffpr(~df.video.isin(drop).values, t_m)})
S["iF_FPR_drop_topk_fp_videos"] = naive
# per group: contribution decomposition |FPR_g - FPR_all| and videos behind each group's FPR
gd = []
ov = fb["matched"]["overall"]["FPR"]
for gname, s in fb["matched"]["intersection"]["groups"].items():
    gi = [k for k, v in GNAME.items() if v.lower().replace("m-", "male,").replace("f-", "nonmale,") == gname][0]
    rv = vr[vr.group == GNAME[gi]]
    gd.append({"group": GNAME[gi], "FPR": 100 * s["FPR"], "abs_dev_from_pooled": 100 * abs(s["FPR"] - ov),
               "real_videos": len(rv), "real_frames": int(rv.n.sum()), "FP": int(rv.fp.sum()),
               "videos_fp_rate_ge_50": int((rv.fp_rate >= 50).sum()),
               "video_fp_rates": [round(x, 1) for x in sorted(rv.fp_rate.tolist(), reverse=True)]})
S["iF_FPR_decomposition_matched"] = {"pooled_FPR": 100 * ov, "groups": gd}

# video-cluster bootstrap of intersection F_FPR at matched (resample real AND fake videos within label)
rng = np.random.default_rng(0)
vids_r = df[real].video.unique()
vids_f = df[fake].video.unique()
idx_by_vid = {v: np.flatnonzero(df.video.values == v) for v in df.video.unique()}
boots = []
for _ in range(1000):
    sel = np.concatenate([idx_by_vid[v] for v in rng.choice(vids_r, len(vids_r))] +
                         [idx_by_vid[v] for v in rng.choice(vids_f, len(vids_f))])
    r = fairness_block(y[sel], df.logit.values[sel], df.group.values[sel], "six", t_m)["intersection"]
    boots.append(r.get("F_FPR", np.nan))
boots = np.array(boots, float)
S["iF_FPR_video_bootstrap_matched"] = {"point": base, "B": 1000, "seed": 0, "mean": float(np.nanmean(boots)),
                                       "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]}

# ---------------------------------------------------------------- FN by actor / scene
fk = df[fake]
def fn_tab(col):
    t = fk.groupby(col).agg(n=("fn", "size"), videos=("video", "nunique"), FN=("fn", "sum"), ding_FN=("ding_fn", "sum"))
    t["FNR"] = 100 * t.FN / t.n
    t["FNR_ding"] = 100 * t.ding_FN / t.n
    for k in ("native", "A1"):
        t["FNR_" + k] = 100 * fk.groupby(col)["pred_" + k].apply(lambda s: float((~s.values).mean()))
    return t.reset_index().sort_values("FNR", ascending=False)
fn_ida = fn_tab("id_a"); fn_ida.round(2).to_csv(f"{OUT}/fn_by_id_a_method_col.csv", index=False)
fn_idb = fn_tab("id_b"); fn_idb.round(2).to_csv(f"{OUT}/fn_by_id_b.csv", index=False)
fn_sc = fn_tab("scene"); fn_sc.round(2).to_csv(f"{OUT}/fn_by_scene.csv", index=False)
rl = df[real]
fp_sc = rl.groupby("scene").agg(n=("fp", "size"), videos=("video", "nunique"), FP=("fp", "sum"), ding_FP=("ding_fp", "sum"))
fp_sc["FPR"] = 100 * fp_sc.FP / fp_sc.n
fp_sc["FPR_ding"] = 100 * fp_sc.ding_FP / fp_sc.n
fp_sc = fp_sc.reset_index().sort_values("FPR", ascending=False)
fp_sc.round(2).to_csv(f"{OUT}/fp_by_scene.csv", index=False)
fp_act = rl.groupby("id_a").agg(n=("fp", "size"), videos=("video", "nunique"), FP=("fp", "sum"), ding_FP=("ding_fp", "sum"),
                                group=("gname", "first"))
fp_act["FPR"] = 100 * fp_act.FP / fp_act.n
fp_act["FPR_ding"] = 100 * fp_act.ding_FP / fp_act.n
fp_act = fp_act.reset_index().sort_values("FPR", ascending=False)
fp_act.round(2).to_csv(f"{OUT}/fp_by_actor.csv", index=False)

# ---------------------------------------------------------------- shared vs unique with Ding
def shared(ours, ding, label):
    n = int(ours.sum())
    both = int((ours & ding).sum())
    return {"ours": n, "ding": int(ding.sum()), "both": both, "frac_ours_also_ding": 100 * both / n if n else float("nan"),
            "frac_ding_also_ours": 100 * both / int(ding.sum()) if ding.sum() else float("nan"),
            "ours_only": n - both, "ding_only": int(ding.sum()) - both, "label": label}
SH = {}
for k in OPS:
    p = df["pred_" + k].values
    SH[k] = {"FP": shared(real & p, df.ding_fp.values, "real"), "FN": shared(fake & ~p, df.ding_fn.values, "fake")}
# operating-point-free check (TEST-ORACLE, analysis only): Ding thresholded to the SAME number of FPs / FNs as ours
def ding_same_count(ours_mask, cls_mask, high):
    n = int(ours_mask.sum())
    s = df.ding_logit.values[cls_mask]
    idx = np.flatnonzero(cls_mask)
    order = np.argsort(-s if high else s, kind="stable")[:n]
    m = np.zeros(len(df), bool); m[idx[order]] = True
    return 100 * float((m & ours_mask).sum() / n)
SH["matched"]["FP_overlap_if_ding_had_same_FP_count_oracle"] = ding_same_count(df.fp.values, real, True)
SH["matched"]["FN_overlap_if_ding_had_same_FN_count_oracle"] = ding_same_count(df.fn.values, fake, False)
# video level: a video "fails" if >= 50 % of its frames are errors
vr["ours_fail"] = vr.fp_rate >= 50; vr["ding_fail"] = vr.ding_fp_rate >= 50
vf["ours_fail"] = vf.fn_rate >= 50; vf["ding_fail"] = vf.ding_fn_rate >= 50
SH["video_level_matched_ge50pct"] = {
    "FP_videos_ours": int(vr.ours_fail.sum()), "FP_videos_ding": int(vr.ding_fail.sum()),
    "FP_videos_both": int((vr.ours_fail & vr.ding_fail).sum()),
    "FN_videos_ours": int(vf.ours_fail.sum()), "FN_videos_ding": int(vf.ding_fail.sum()),
    "FN_videos_both": int((vf.ours_fail & vf.ding_fail).sum())}
S["shared_with_ding"] = SH

cols_case = ["video", "gname", "cl_gender", "cl_race", "if_age", "ita_vid", "scene", "id_a", "id_b", "logit",
             "ding_logit", "img_path"]
fp_ours_only = df[df.fp & ~df.ding_fp].sort_values("logit", ascending=False)
fp_ding_only = df[df.ding_fp & ~df.fp].sort_values("ding_logit", ascending=False)
fn_ours_only = df[df.fn & ~df.ding_fn].sort_values("logit")
fn_ding_only = df[df.ding_fn & ~df.fn].sort_values("ding_logit")
for nm, t in (("fp_ours_only", fp_ours_only), ("fp_ding_only", fp_ding_only), ("fn_ours_only", fn_ours_only),
              ("fn_ding_only", fn_ding_only)):
    t[cols_case].round(3).to_csv(f"{OUT}/{nm}_frames.csv", index=False)
# video-level unique lists
vr["ours_minus_ding_fp"] = vr.fp - vr.ding_fp
vf["ours_minus_ding_fn"] = vf.fn - vf.ding_fn
vr.sort_values("ours_minus_ding_fp", ascending=False)[keep_r + ["ours_minus_ding_fp"]].round(2).to_csv(
    f"{OUT}/fp_video_ours_vs_ding.csv", index=False)
vf.sort_values("ours_minus_ding_fn", ascending=False)[keep_f + ["ours_minus_ding_fn"]].round(2).to_csv(
    f"{OUT}/fn_video_ours_vs_ding.csv", index=False)

# ---------------------------------------------------------------- confident errors
top_fp = df[df.fp].sort_values("logit", ascending=False).head(12)
top_fn = df[df.fn].sort_values("logit").head(12)
top_fp[cols_case].round(3).to_csv(f"{OUT}/top12_confident_fp.csv", index=False)
top_fn[cols_case].round(3).to_csv(f"{OUT}/top12_confident_fn.csv", index=False)
S["confident"] = {"fp_logit_quantiles": df[df.fp].logit.quantile([0, .5, .9, 1]).round(2).tolist(),
                  "fn_logit_quantiles": df[df.fn].logit.quantile([0, .1, .5, 1]).round(2).tolist(),
                  "fp_above_A1": int((df.fp & (df.logit >= OPS["A1"])).sum()),
                  "fp_above_0": int((df.fp & (df.logit >= 0)).sum()),
                  "real_logit_median": float(df[real].logit.median()), "fake_logit_median": float(df[fake].logit.median())}

# ---------------------------------------------------------------- contact sheets
font = ImageFont.truetype(FONT, 15)
fontb = ImageFont.truetype(FONT, 22)


def sheet(rows, path, title):
    tile, cap, ncol = 256, 84, 6
    nrow = int(np.ceil(len(rows) / ncol))
    W, H = ncol * tile, 44 + nrow * (tile + cap)
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    dr.text((10, 10), title, fill="black", font=fontb)
    for i, (_, r) in enumerate(rows.iterrows()):
        x, y0 = (i % ncol) * tile, 44 + (i // ncol) * (tile + cap)
        try:
            fr = Image.open(r.img_path).convert("RGB").resize((tile - 6, tile - 6), Image.BICUBIC)
            im.paste(fr, (x + 3, y0 + 3))
        except Exception as ex:  # noqa: BLE001
            dr.text((x + 5, y0 + 100), f"missing: {ex}"[:30], fill="red", font=font)
        v = r.video[2:]
        v = v if len(v) <= 30 else v[:30] + ".."
        dpred = "fake" if r.ding_logit >= 0 else "real"
        dcol = (0, 120, 0) if (dpred == "fake") == (r.label == 1) else (200, 0, 0)
        meth = f"swap {r.id_a}<-{r.id_b}" if r.label == 1 else "real"
        dr.text((x + 4, y0 + tile + 2), v, fill="black", font=font)
        dr.text((x + 4, y0 + tile + 20), f"{r.gname} | {meth}", fill="black", font=font)
        dr.text((x + 4, y0 + tile + 38), f"ours logit {r.logit:.1f}", fill="black", font=font)
        dr.text((x + 4, y0 + tile + 56), f"Ding: {dpred} ({r.ding_logit:.1f})", fill=dcol, font=font)
    im.save(path)


fp_sheet_rows = df[df.fp].sort_values("logit", ascending=False).drop_duplicates("video").head(24)
fn_sheet_rows = df[df.fn].sort_values("logit").drop_duplicates("video").head(24)
sheet(fp_sheet_rows, f"{OUT}/fp_sheet.png",
      f"BD0 s7 on DFD: top-{len(fp_sheet_rows)} false positives (real flagged fake), one frame per video, matched t={t_m:.2f}")
sheet(fn_sheet_rows, f"{OUT}/fn_sheet.png",
      f"BD0 s7 on DFD: top-{len(fn_sheet_rows)} false negatives (fake missed), one frame per video, matched t={t_m:.2f}")
fp_sheet_rows[cols_case].round(3).to_csv(f"{OUT}/fp_sheet_tiles.csv", index=False)
fn_sheet_rows[cols_case].round(3).to_csv(f"{OUT}/fn_sheet_tiles.csv", index=False)
S["sheet_ding_correct"] = {"fp_tiles_ding_says_real": int((fp_sheet_rows.ding_logit < 0).sum()),
                           "fn_tiles_ding_says_fake": int((fn_sheet_rows.ding_logit >= 0).sum())}

# ---------------------------------------------------------------- save
def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))
json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=conv)
print(json.dumps(S, indent=1, default=conv))
