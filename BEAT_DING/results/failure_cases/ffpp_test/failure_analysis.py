"""Failure-case analysis of BD0_s7 on FF++ test (in-domain). CPU only, read-only on inputs.

Run: ~/.conda/envs/aiface/bin/python failure_analysis.py
Writes (next to this script): results.json, tables/*.csv, fp_sheet.png, fn_sheet.png, FAILURES_tables.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
OUT = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
sys.path.insert(0, f"{R}/BEAT_DING")
from cfes.metrics import fairness_block  # noqa: E402
from cfes.data import SCHEMES  # noqa: E402

EV = f"{R}/BEAT_DING/runs/BD0_s7/eval"
d = np.load(f"{EV}/ffpp_test_scores.npz", allow_pickle=True)
e = np.load(f"{R}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_ff++.npz", allow_pickle=True)
L = pd.read_csv(f"{R}/DEMO_AXES/labels/labels_ffpp_test.csv")
FF = pd.read_csv(f"{R}/DEMO_AXES/labels_fairface/ffpp_test.csv")
TH = json.load(open(f"{EV}/bd_eval.json"))["thresholds"]

paths = d["img_path"].astype(str)
assert (L.img_path.values == paths).all() and (e["img_path"].astype(str) == paths).all()
assert (FF.img_path.values == paths).all()
assert (e["label"] == d["label"]).all() and (e["group"] == d["group"]).all()

GN = ["M-Asian", "M-White", "M-Black", "M-Others", "F-Asian", "F-White", "F-Black", "F-Others"]
AGE = {0: "<30", 1: "30-44", 2: "45-59", 3: "60+"}
ITA = {0: "Q1 darkest", 1: "Q2", 2: "Q3", 3: "Q4 lightest"}
FAMILY = {"Deepfakes": "identity-swap", "FaceSwap": "identity-swap", "FaceShifter": "identity-swap",
          "Face2Face": "reenactment", "NeuralTextures": "reenactment", "real": "real"}

df = pd.DataFrame({
    "img_path": paths, "label": d["label"].astype(int), "group": d["group"].astype(int),
    "method": d["method"].astype(str), "logit": d["logit"].astype(np.float64),
    "ding_logit": e["logit"].astype(np.float64), "video": L.video.values,
    "xu_gender": np.where(L.xu_male.values == 1, "male", "female"), "xu_race": L.xu_race.values,
    "cl_gender": np.where(L.cl_male.values == 1, "male", "female"), "cl_race": L.cl_race.values,
    "age4": L.age4_if.map(AGE).values, "age_vid": L.age_vid_if.values,
    "ita4": L.ita4.map(ITA).fillna("missing").values, "ita_vid": L.ita_vid.values,
    "ff_race7": FF.race7.values, "ff_sex": FF.sex.values,
})
df["gname"] = df.group.map(dict(enumerate(GN)))
df["family"] = df.method.map(FAMILY)
df["frame"] = df.img_path.map(os.path.basename)
df["ding_fake"] = df.ding_logit >= 0  # Ding's 0.5 rule
df["ding_wrong"] = df.ding_fake != (df.label == 1)
y = df.label.values
real, fake = y == 0, y == 1
res = {"n": len(df), "n_real": int(real.sum()), "n_fake": int(fake.sum()),
       "n_real_videos": int(df[real].video.nunique()), "n_fake_videos": int(df[fake].video.nunique()),
       "thresholds": TH}


def iffpr(mask_keep, thr):
    m = mask_keep
    fb = fairness_block(y[m], df.logit.values[m], df.group.values[m], "eight", thr)
    return fb["intersection"]["F_FPR"], fb


# ---- sanity: reproduce bd_eval matched numbers
_, fb = iffpr(np.ones(len(df), bool), TH["matched"])
res["check_matched"] = {"TPR": 100 * fb["overall"]["TPR"], "FPR": 100 * fb["overall"]["FPR"],
                        "iF_FPR": fb["intersection"]["F_FPR"], "gF_FPR": fb["gender"]["F_FPR"],
                        "rF_FPR": fb["race"]["F_FPR"]}
dr = df.ding_fake.values
res["ding_rates"] = {"FPR": 100 * float(dr[real].mean()), "TPR": 100 * float(dr[fake].mean()),
                     "n_FP": int(dr[real].sum()), "n_FN": int((~dr[fake]).sum())}
fbd = fairness_block(y, df.ding_logit.values, df.group.values, "eight", 0.0)
res["ding_rates"]["iF_FPR"] = fbd["intersection"]["F_FPR"]


def rate_table(sub, col, err_col):
    g = sub.groupby(col)
    t = pd.DataFrame({"n": g.size(), "n_err": g[err_col].sum(), "n_videos": g.video.nunique(),
                      "n_videos_with_err": g.apply(lambda x: x.loc[x[err_col], "video"].nunique(), include_groups=False)})
    t["rate_%"] = 100 * t.n_err / t.n
    t["share_of_errors_%"] = 100 * t.n_err / max(1, t.n_err.sum())
    return t.reset_index()


md = []
ops = ["matched", "native", "A1"]
res["ops"] = {}
for op in ops:
    t = TH[op]
    df["pred"] = df.logit >= t
    df["fp"] = real & df.pred.values
    df["fn"] = fake & ~df.pred.values
    R_ = df[real]
    Fk = df[fake]
    o = {"t": t, "n_FP": int(df.fp.sum()), "FPR_%": 100 * float(df.fp.sum() / real.sum()),
         "n_FN": int(df.fn.sum()), "FNR_%": 100 * float(df.fn.sum() / fake.sum())}
    # shared with Ding
    o["FP_also_ding_%"] = 100 * float(df.loc[df.fp, "ding_fake"].mean())
    o["FN_also_ding_%"] = 100 * float((~df.loc[df.fn, "ding_fake"]).mean())
    dfp = real & dr
    dfn = fake & ~dr
    o["dingFP_also_ours_%"] = 100 * float(df.fp.values[dfp].mean())
    o["dingFN_also_ours_%"] = 100 * float(df.fn.values[dfn].mean())
    o["FP_ours_only"] = int((df.fp & ~df.ding_fake).sum())
    o["FP_ding_only"] = int((dfp & ~df.fp.values).sum())
    o["FN_ours_only"] = int((df.fn & df.ding_fake).sum())
    o["FN_ding_only"] = int((dfn & ~df.fn.values).sum())
    o["FP_jaccard"] = float((df.fp.values & dfp).sum() / (df.fp.values | dfp).sum())
    o["FN_jaccard"] = float((df.fn.values & dfn).sum() / (df.fn.values | dfn).sum())
    # FP tables
    tabs = {}
    for col in ["gname", "xu_gender", "cl_gender", "xu_race", "cl_race", "age4", "ita4", "ff_race7"]:
        tabs[f"fp_{col}"] = rate_table(R_, col, "fp")
    for col in ["method", "family", "gname", "xu_gender", "cl_gender", "xu_race", "cl_race", "age4", "ita4"]:
        tabs[f"fn_{col}"] = rate_table(Fk, col, "fn")
    for k, v in tabs.items():
        v.to_csv(f"{TAB}/{op}_{k}.csv", index=False)
    o["tables"] = {k: v.round(3).to_dict(orient="records") for k, v in tabs.items()}
    # video concentration of FPs
    vfp = R_.groupby("video").agg(n=("fp", "size"), n_fp=("fp", "sum"), group=("gname", "first"),
                                  cl_gender=("cl_gender", "first"), cl_race=("cl_race", "first"),
                                  ff_race7=("ff_race7", lambda s: s.mode().iat[0]),
                                  age_vid=("age_vid", "first"), ita_vid=("ita_vid", "first"),
                                  ita4=("ita4", "first"), max_logit=("logit", "max"),
                                  ding_fp=("ding_fake", "sum"))
    vfp["fp_rate_%"] = 100 * vfp.n_fp / vfp.n
    vfp["ding_fp_rate_%"] = 100 * vfp.ding_fp / vfp.n
    vfp = vfp.sort_values(["n_fp", "max_logit"], ascending=False)
    vfp.to_csv(f"{TAB}/{op}_fp_by_video.csv")
    cs = vfp.n_fp.cumsum() / max(1, vfp.n_fp.sum())
    o["fp_videos_any"] = int((vfp.n_fp > 0).sum())
    o["fp_videos_ge50pct"] = int((vfp["fp_rate_%"] >= 50).sum())
    o["fp_share_top5_videos_%"] = 100 * float(cs.iloc[4])
    o["fp_share_top15_videos_%"] = 100 * float(cs.iloc[14])
    o["n_videos_for_50pct_fp"] = int((cs < 0.5).sum() + 1)
    # i-F_FPR when the top-k FP videos are dropped (greedy by FP count), same threshold
    base, fb = iffpr(np.ones(len(df), bool), t)
    o["iF_FPR"] = base
    N2G = dict(zip(SCHEMES["eight"], GN))
    o["iF_FPR_group_dev"] = {N2G[n]: 100 * abs(s["FPR"] - fb["overall"]["FPR"]) for n, s in fb["intersection"]["groups"].items()}
    o["group_FPR_%"] = {N2G[n]: 100 * s["FPR"] for n, s in fb["intersection"]["groups"].items()}
    o["ding_group_FPR_%"] = {N2G[n]: 100 * s["FPR"] for n, s in fbd["intersection"]["groups"].items()}
    o["ding_iF_FPR_group_dev"] = {N2G[n]: 100 * abs(s["FPR"] - fbd["overall"]["FPR"]) for n, s in fbd["intersection"]["groups"].items()}
    drop = {}
    order = vfp.index.tolist()
    for k in [1, 2, 3, 5, 10, 15, 20]:
        keep = ~(real & df.video.isin(order[:k]).values)
        v, fbk = iffpr(keep, t)
        # re-threshold so the pooled FPR on the remaining reals equals the full-set FPR (removes the FPR-level effect)
        rl = df.logit.values[keep & real]
        t2 = float(np.quantile(rl, 1 - fb["overall"]["FPR"], method="higher"))
        v2, fbk2 = iffpr(keep, t2)
        drop[k] = {"iF_FPR": v, "FPR_%": 100 * fbk["overall"]["FPR"], "videos": order[:k],
                   "iF_FPR_same_FPR": v2, "FPR_same_%": 100 * fbk2["overall"]["FPR"]}
    o["iF_FPR_drop_topk_videos"] = drop
    # random-drop control: drop k random real videos (not chosen by FP), 200 draws
    rng = np.random.default_rng(0)
    allv = np.array(sorted(R_.video.unique()))
    ctrl = {}
    for k in [5, 10]:
        vals = []
        for _ in range(200):
            sel = rng.choice(allv, k, replace=False)
            kp = ~(real & df.video.isin(sel).values)
            t2 = float(np.quantile(df.logit.values[kp & real], 1 - fb["overall"]["FPR"], method="higher"))
            vals.append(iffpr(kp, t2)[0])
        ctrl[k] = {"median": float(np.median(vals)), "p5": float(np.percentile(vals, 5)),
                   "p95": float(np.percentile(vals, 95))}
    o["iF_FPR_drop_random_videos"] = ctrl
    # video-level view: mean over videos of the per-video FP rate, by group (each video weighs 1)
    # within-group concentration: share of each group's FP frames held by its top-1 / top-3 videos
    wg = {}
    for gname_, sub in vfp.groupby("group"):
        tot = sub.n_fp.sum()
        srt = sub.n_fp.sort_values(ascending=False)
        wg[gname_] = {"n_fp": int(tot), "n_videos": int(len(sub)), "videos_with_fp": int((sub.n_fp > 0).sum()),
                      "top1_share_%": 100 * float(srt.iloc[:1].sum() / tot) if tot else 0.0,
                      "top3_share_%": 100 * float(srt.iloc[:3].sum() / tot) if tot else 0.0,
                      "top3_videos": srt.index[:3].tolist()}
    o["within_group_fp_concentration"] = wg
    o["video_weighted_group_FPR_%"] = vfp.groupby("group")["fp_rate_%"].mean().round(2).to_dict()
    # FN by video
    vfn = Fk.groupby("video").agg(n=("fn", "size"), n_fn=("fn", "sum"), method=("method", "first"),
                                  group=("gname", "first"), cl_gender=("cl_gender", "first"),
                                  cl_race=("cl_race", "first"), age_vid=("age_vid", "first"),
                                  ita4=("ita4", "first"), min_logit=("logit", "min"),
                                  ding_fn=("ding_fake", lambda s: int((~s).sum())))
    vfn["fn_rate_%"] = 100 * vfn.n_fn / vfn.n
    vfn["ding_fn_rate_%"] = 100 * vfn.ding_fn / vfn.n
    vfn = vfn.sort_values(["n_fn", "min_logit"], ascending=[False, True])
    vfn.to_csv(f"{TAB}/{op}_fn_by_video.csv")
    csn = vfn.n_fn.cumsum() / max(1, vfn.n_fn.sum())
    o["fn_videos_any"] = int((vfn.n_fn > 0).sum())
    o["fn_videos_ge50pct"] = int((vfn["fn_rate_%"] >= 50).sum())
    o["fn_share_top15_videos_%"] = 100 * float(csn.iloc[14])
    o["n_videos_for_50pct_fn"] = int((csn < 0.5).sum() + 1)
    o["fn_videos_ge50pct_by_method"] = vfn[vfn["fn_rate_%"] >= 50].method.value_counts().to_dict()
    # target identity of FN videos (fake video id = Method_TTT_SSS)
    o["top_fp_videos"] = vfp.head(15).reset_index().round(2).to_dict(orient="records")
    o["top_fn_videos"] = vfn.head(15).reset_index().round(2).to_dict(orient="records")
    # unique-vs-Ding per video
    df["fp_only_ours"] = df.fp & ~df.ding_fake
    df["fp_only_ding"] = real & df.ding_fake & ~df.fp
    df["fn_only_ours"] = df.fn & df.ding_fake
    df["fn_only_ding"] = fake & ~df.ding_fake & ~df.fn
    uv = {}
    for c in ["fp_only_ours", "fp_only_ding", "fn_only_ours", "fn_only_ding"]:
        sub = df[real] if c.startswith("fp") else df[fake]
        g = sub.groupby("video").agg(n=(c, "size"), n_err=(c, "sum"), group=("gname", "first"),
                                     method=("method", "first"), ours_mean_logit=("logit", "mean"),
                                     ding_mean_logit=("ding_logit", "mean"))
        g["rate_%"] = 100 * g.n_err / g.n
        g = g.sort_values("n_err", ascending=False)
        g.to_csv(f"{TAB}/{op}_{c}_by_video.csv")
        uv[c] = g.head(10).reset_index().round(2).to_dict(orient="records")
        uv[c + "_by_group"] = sub.groupby("gname")[c].sum().to_dict()
        if c.startswith("fn"):
            uv[c + "_by_method"] = sub.groupby("method")[c].sum().to_dict()
    o["unique_vs_ding"] = uv
    # most confident errors
    cfp = df[df.fp].sort_values("logit", ascending=False).head(12)
    cfn = df[df.fn].sort_values("logit").head(12)
    keep_cols = ["frame", "video", "gname", "method", "logit", "ding_logit", "ding_fake", "cl_gender", "cl_race",
                 "age4", "ita4", "img_path"]
    cfp[keep_cols].to_csv(f"{TAB}/{op}_top12_fp.csv", index=False)
    cfn[keep_cols].to_csv(f"{TAB}/{op}_top12_fn.csv", index=False)
    o["top12_fp"] = cfp[keep_cols].round(3).to_dict(orient="records")
    o["top12_fn"] = cfn[keep_cols].round(3).to_dict(orient="records")
    # ours-only frames with the largest margin (we confidently wrong, Ding right)
    oo = df[df.fp_only_ours].sort_values("logit", ascending=False).head(12)
    on = df[df.fn_only_ours].sort_values("logit").head(12)
    oo[keep_cols].to_csv(f"{TAB}/{op}_top12_fp_ours_only.csv", index=False)
    on[keep_cols].to_csv(f"{TAB}/{op}_top12_fn_ours_only.csv", index=False)
    do = df[df.fp_only_ding].sort_values("ding_logit", ascending=False).head(12)
    dn = df[df.fn_only_ding].sort_values("ding_logit").head(12)
    do[keep_cols].to_csv(f"{TAB}/{op}_top12_fp_ding_only.csv", index=False)
    dn[keep_cols].to_csv(f"{TAB}/{op}_top12_fn_ding_only.csv", index=False)
    res["ops"][op] = o
    if op == "matched":
        MATCHED_DF = df.copy()

# ---- target-identity view (matched): FF++ fake video "0_Method_TTT_SSS" alters target video TTT
df = MATCHED_DF
fk = df[fake].copy()
fk["target"] = fk.video.str.split("_").str[2]
rl = df[real].copy()
rl["target"] = rl.video.str.split("_").str[2]
tg = fk.groupby("target").agg(n_fake=("fn", "size"), n_fn=("fn", "sum"), n_methods=("method", "nunique"),
                              fake_mean_logit=("logit", "mean"), group=("gname", "first"))
tg["fn_rate_%"] = 100 * tg.n_fn / tg.n_fake
tg["methods_ge50pct_fn"] = fk.groupby(["target", "method"]).fn.mean().ge(0.5).groupby("target").sum()
rv = rl.groupby("target").agg(real_mean_logit=("logit", "mean"), real_fp_rate=("fp", "mean"),
                              ding_real_mean_logit=("ding_logit", "mean"))
tg = tg.join(rv, how="left")
fkd = fk.groupby("target").agg(ding_fn_rate=("ding_fake", lambda s: 100 * float((~s).mean())))
tg = tg.join(fkd).sort_values("n_fn", ascending=False)
tg.to_csv(f"{TAB}/matched_fn_by_target_identity.csv")
both = tg.dropna(subset=["real_mean_logit"])
cfn = tg.n_fn.cumsum() / tg.n_fn.sum()
res["fn_target_identity"] = {
    "n_targets": int(len(tg)), "targets_with_fn": int((tg.n_fn > 0).sum()),
    "n_targets_for_50pct_fn": int((cfn < 0.5).sum() + 1),
    "share_fn_top10_targets_%": 100 * float(cfn.iloc[9]),
    "targets_with_ge2_methods_ge50pct_fn": int((tg.methods_ge50pct_fn >= 2).sum()),
    "n_targets_with_real_video_in_test": int(len(both)),
    "spearman_real_logit_vs_fake_fnrate": float(both.real_mean_logit.rank().corr(both["fn_rate_%"].rank())),
    "spearman_real_logit_vs_fake_mean_logit": float(both.real_mean_logit.rank().corr(both.fake_mean_logit.rank())),
    "ding_spearman_real_logit_vs_fake_fnrate": float(both.ding_real_mean_logit.rank().corr(both.ding_fn_rate.rank())),
    "top10": tg.head(10).reset_index().round(2).to_dict(orient="records")}
# same idea for FPs: real videos flagged often -- are their fakes caught? (FN rate of fakes of the same target)
fpv = rl.groupby("target").fp.mean().rename("fp_rate")
j = tg.join(fpv, how="inner")
res["fp_target_identity"] = {"spearman_real_fprate_vs_fake_fnrate": float(j.fp_rate.rank().corr(j["fn_rate_%"].rank()))}

# ---- score-level view: our logit vs Ding's on the matched FP set (rank correlation over reals)
real_df = df[real]
res["spearman_reals_ours_vs_ding"] = float(real_df.logit.rank().corr(real_df.ding_logit.rank()))
res["spearman_fakes_ours_vs_ding"] = float(df[fake].logit.rank().corr(df[fake].ding_logit.rank()))


# ---- contact sheets (matched op)
def font(sz, bold=False):
    base = os.path.dirname(__import__("matplotlib").__file__) + "/mpl-data/fonts/ttf/"
    return ImageFont.truetype(base + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)


def sheet(rows, title, fname, kind):
    cols, tile, cap = 6, 300, 92
    nr = int(np.ceil(len(rows) / cols))
    W, H = cols * tile + (cols + 1) * 8, 60 + nr * (tile + cap + 8) + 8
    S = Image.new("RGB", (W, H), "white")
    dr_ = ImageDraw.Draw(S)
    dr_.text((10, 14), title, fill="black", font=font(24, True))
    f1, f2 = font(15, True), font(14)
    for i, r in enumerate(rows):
        x = 8 + (i % cols) * (tile + 8)
        yy = 60 + (i // cols) * (tile + cap + 8)
        im = Image.open(r.img_path).convert("RGB").resize((tile, tile), Image.BICUBIC)
        S.paste(im, (x, yy))
        ding_dec = "fake" if r.ding_fake else "real"
        ding_ok = (ding_dec == "fake") == (kind == "FN")  # Ding right?
        col = (0, 120, 0) if ding_ok else (190, 0, 0)
        vid = "real " + r.video[len("0_youtube_"):] if r.video.startswith("0_youtube_") else r.video[2:]
        dr_.text((x + 2, yy + tile + 3), f"#{i+1} {vid}", fill="black", font=f1)
        dr_.text((x + 2, yy + tile + 22), f"{r.gname} | {r.method}", fill="black", font=f2)
        dr_.text((x + 2, yy + tile + 40), f"logit {r.logit:.1f} (t={TH['matched']:.1f}) | {r.vid_err}/{r.vid_n} fr",
                 fill="black", font=f2)
        dr_.text((x + 2, yy + tile + 58), f"Ding: {ding_dec} ({r.ding_logit:+.2f}) "
                 + ("Ding right" if ding_ok else "Ding also wrong"), fill=col, font=f2)
    S.save(os.path.join(OUT, fname))


df = MATCHED_DF
t = TH["matched"]
fpf = df[df.fp].copy()
fpf["vid_err"] = fpf.groupby("video").fp.transform("size")
fpf["vid_n"] = fpf.video.map(df[real].groupby("video").size())
fp_rows = fpf.sort_values("logit", ascending=False).drop_duplicates("video").head(24)
fnf = df[df.fn].copy()
fnf["vid_err"] = fnf.groupby("video").fn.transform("size")
fnf["vid_n"] = fnf.video.map(df[fake].groupby("video").size())
fn_rows = fnf.sort_values("logit").drop_duplicates("video").head(24)
sheet(list(fp_rows.itertuples()), "BD0_s7 FF++ test: top-24 false positives (real frames flagged fake), "
      "matched threshold, highest-logit FP frame per video", "fp_sheet.png", "FP")
sheet(list(fn_rows.itertuples()), "BD0_s7 FF++ test: top-24 false negatives (fakes missed), "
      "matched threshold, lowest-logit FN frame per video", "fn_sheet.png", "FN")
cs_cols = ["video", "gname", "method", "logit", "ding_logit", "ding_fake", "vid_err", "vid_n", "img_path"]
fp_rows[cs_cols].to_csv(f"{TAB}/fp_sheet_tiles.csv", index=False)
fn_rows[cs_cols].to_csv(f"{TAB}/fn_sheet_tiles.csv", index=False)
res["fp_sheet_ding_also_wrong"] = int(fp_rows.ding_fake.sum())
res["fn_sheet_ding_also_wrong"] = int((~fn_rows.ding_fake).sum())


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)


json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, cls=NpEnc)


# ---- markdown tables
def mdt(frame, cols=None, fmt=2):
    f = frame if cols is None else frame[cols]
    f = f.copy()
    for c in f.columns:
        if f[c].dtype.kind == "f":
            f[c] = f[c].round(fmt)
    head = "| " + " | ".join(map(str, f.columns)) + " |\n|" + "---|" * len(f.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in f.values)


md = ["## Operating points (FF++ test, n_real=%d frames / %d videos, n_fake=%d / %d)\n" %
      (res["n_real"], res["n_real_videos"], res["n_fake"], res["n_fake_videos"])]
rows = []
for op in ops:
    o = res["ops"][op]
    rows.append([op, round(o["t"], 2), o["n_FP"], round(o["FPR_%"], 2), o["n_FN"], round(o["FNR_%"], 2),
                 round(o["iF_FPR"], 2), round(o["FP_also_ding_%"], 1), round(o["FN_also_ding_%"], 1),
                 o["FP_ours_only"], o["FP_ding_only"], o["FN_ours_only"], o["FN_ding_only"]])
rows.append(["Ding repro @0.5", 0.0, res["ding_rates"]["n_FP"], round(res["ding_rates"]["FPR"], 2),
             res["ding_rates"]["n_FN"], round(100 - res["ding_rates"]["TPR"], 2), round(res["ding_rates"]["iF_FPR"], 2),
             "", "", "", "", "", ""])
md.append(mdt(pd.DataFrame(rows, columns=["op", "logit t", "#FP", "FPR %", "#FN", "FNR %", "i-F_FPR",
                                          "FP also Ding %", "FN also Ding %", "FP ours-only", "FP Ding-only",
                                          "FN ours-only", "FN Ding-only"])))
for op in ops:
    o = res["ops"][op]
    md.append(f"\n## {op}: false positives by attribute\n")
    for k in ["fp_gname", "fp_xu_gender", "fp_cl_gender", "fp_xu_race", "fp_cl_race", "fp_age4", "fp_ita4"]:
        md.append(f"\n**{k[3:]}**\n\n" + mdt(pd.DataFrame(o["tables"][k])))
    if op != "A1":
        md.append(f"\n## {op}: false negatives by attribute\n")
        for k in ["fn_method", "fn_family", "fn_gname", "fn_cl_gender", "fn_cl_race", "fn_age4", "fn_ita4"]:
            md.append(f"\n**{k[3:]}**\n\n" + mdt(pd.DataFrame(o["tables"][k])))
    md.append(f"\n## {op}: FP video concentration\n")
    md.append(f"- real videos with >=1 FP frame: {o['fp_videos_any']} / {res['n_real_videos']}; with >=50% FP frames: "
              f"{o['fp_videos_ge50pct']}\n- top-5 videos hold {o['fp_share_top5_videos_%']:.1f}% of FP frames, top-15 "
              f"hold {o['fp_share_top15_videos_%']:.1f}%; {o['n_videos_for_50pct_fp']} videos hold half\n")
    dd = [[0, round(o["iF_FPR"], 2), round(o["FPR_%"], 2), round(o["iF_FPR"], 2), ""]] + \
         [[k, round(v["iF_FPR"], 2), round(v["FPR_%"], 2), round(v["iF_FPR_same_FPR"], 2),
           ", ".join(x.replace("0_youtube_", "") for x in v["videos"][-5:])]
          for k, v in o["iF_FPR_drop_topk_videos"].items()]
    md.append("\n" + mdt(pd.DataFrame(dd, columns=["top-k FP videos dropped", "i-F_FPR (same t)", "FPR % (same t)",
                                                    "i-F_FPR (re-thresholded to same FPR)", "last dropped (<=5)"])))
    md.append("\nrandom-drop control, re-thresholded to same FPR (200 draws): " + "; ".join(
        f"k={k}: median {v['median']:.2f} [p5 {v['p5']:.2f}, p95 {v['p95']:.2f}]" for k, v in o["iF_FPR_drop_random_videos"].items()) + "\n")
    wgc = o["within_group_fp_concentration"]
    gd = pd.DataFrame({"group_FPR_%": o["group_FPR_%"], "|FPR_g-FPR_all|": o["iF_FPR_group_dev"],
                       "video-weighted FPR %": pd.Series(o["video_weighted_group_FPR_%"]),
                       "videos with FP": pd.Series({k: f"{v['videos_with_fp']}/{v['n_videos']}" for k, v in wgc.items()}),
                       "top-3 video share of group FPs %": pd.Series({k: v["top3_share_%"] for k, v in wgc.items()}),
                       "Ding FPR_g %": o["ding_group_FPR_%"], "Ding |dev|": o["ding_iF_FPR_group_dev"]})
    md.append("\n" + mdt(gd.reset_index().rename(columns={"index": "group"})))
    md.append(f"\n**top-15 FP videos ({op})**\n\n" + mdt(pd.DataFrame(o["top_fp_videos"]),
              ["video", "group", "cl_gender", "cl_race", "ff_race7", "age_vid", "ita4", "n", "n_fp", "fp_rate_%",
               "ding_fp_rate_%", "max_logit"]))
    if op != "A1":
        md.append(f"\n**top-15 FN videos ({op})**\n\n" + mdt(pd.DataFrame(o["top_fn_videos"]),
                  ["video", "method", "group", "cl_gender", "cl_race", "age_vid", "ita4", "n", "n_fn", "fn_rate_%",
                   "ding_fn_rate_%", "min_logit"]))
    uv = o["unique_vs_ding"]
    for c, lab in [("fp_only_ours", "FP ours-only (we flag, Ding says real)"), ("fp_only_ding", "FP Ding-only"),
                   ("fn_only_ours", "FN ours-only (we miss, Ding catches)"), ("fn_only_ding", "FN Ding-only")]:
        if op == "A1" and c.startswith("fn"):
            continue
        md.append(f"\n**{lab}, top videos ({op})** — by group: {uv[c + '_by_group']}"
                  + (f"; by method: {uv[c + '_by_method']}" if c + "_by_method" in uv else "") + "\n\n"
                  + mdt(pd.DataFrame(uv[c]).head(8), ["video", "group", "method", "n", "n_err", "rate_%",
                                                      "ours_mean_logit", "ding_mean_logit"]))
    if op == "matched":
        md.append("\n**top-12 most confident FP (matched)**\n\n" + mdt(pd.DataFrame(o["top12_fp"]),
                  ["frame", "gname", "logit", "ding_logit", "cl_race", "age4", "ita4", "img_path"]))
        md.append("\n**top-12 most confident FN (matched)**\n\n" + mdt(pd.DataFrame(o["top12_fn"]),
                  ["frame", "gname", "method", "logit", "ding_logit", "cl_race", "age4", "ita4", "img_path"]))
ti = res["fn_target_identity"]
md.append("\n## FN by target identity (matched; fake video Method_TTT_SSS alters target TTT)\n\n"
          f"- {ti['targets_with_fn']}/{ti['n_targets']} targets have >=1 FN; {ti['n_targets_for_50pct_fn']} targets hold half of "
          f"all FNs; top-10 targets hold {ti['share_fn_top10_targets_%']:.1f}%; {ti['targets_with_ge2_methods_ge50pct_fn']} targets "
          f"are missed (>=50% frames) under >=2 methods\n"
          f"- over {ti['n_targets_with_real_video_in_test']} targets whose real video is in test: Spearman(real-video mean logit, "
          f"FN rate of its fakes) = {ti['spearman_real_logit_vs_fake_fnrate']:.3f} (ours), "
          f"{ti['ding_spearman_real_logit_vs_fake_fnrate']:.3f} (Ding); Spearman(real mean logit, fake mean logit) = "
          f"{ti['spearman_real_logit_vs_fake_mean_logit']:.3f}\n\n"
          + mdt(pd.DataFrame(ti["top10"]), ["target", "group", "n_fake", "n_fn", "fn_rate_%", "methods_ge50pct_fn",
                                            "fake_mean_logit", "real_mean_logit", "real_fp_rate", "ding_fn_rate"]))
open(os.path.join(OUT, "FAILURES_tables.md"), "w").write("\n".join(md))
print(json.dumps({k: res[k] for k in ["check_matched", "ding_rates", "spearman_reals_ours_vs_ding",
                                     "spearman_fakes_ours_vs_ding", "fp_sheet_ding_also_wrong",
                                     "fn_sheet_ding_also_wrong", "fp_target_identity"]}, indent=1, cls=NpEnc))
