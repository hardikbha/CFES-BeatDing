"""Independent verifier for FF++ test failure-case claims (BD0_s7 vs Ding repro). Fresh code; read-only inputs."""
import json, re, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
sys.path.insert(0, f"{A}/BEAT_DING")
from cfes.metrics import fairness_block
o = np.load(f"{A}/BEAT_DING/runs/BD0_s7/eval/ffpp_test_scores.npz", allow_pickle=True)
d = np.load(f"{A}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_ff++.npz", allow_pickle=True)
TH = json.load(open(f"{A}/BEAT_DING/runs/BD0_s7/eval/bd_eval.json"))["thresholds"]
assert (o["img_path"] == d["img_path"].astype(str)).all() and (o["label"] == d["label"]).all() and (o["group"] == d["group"]).all()
df = pd.DataFrame({k: o[k] for k in ["logit", "label", "group", "method", "cluster", "img_path"]})
df["logit"] = df.logit.astype(np.float64); df["dl"] = d["logit"].astype(np.float64)
lab = pd.read_csv(f"{A}/DEMO_AXES/labels/labels_ffpp_test.csv")
df = df.merge(lab[["img_path", "ita4", "age4_if", "xu_male", "cl_male", "xu_group"]], on="img_path", how="left")
assert len(df) == 25401
print("xu_group==group:", (df.xu_group == df.group).mean())
G = ["M-Asian", "M-White", "M-Black", "M-Others", "F-Asian", "F-White", "F-Black", "F-Others"]
y = df.label.values; g = df.group.values; real = y == 0; fake = y == 1
df["vid"] = df.cluster
dfake = df.dl.values >= 0
R = {}
def block(score, t):
    fb = fairness_block(y, score, g, "eight", t)
    return fb
for op in ["matched", "native", "A1"]:
    t = TH[op]; pred = df.logit.values >= t
    fb = block(df.logit.values, t)
    fp = pred & real; fn = ~pred & fake
    r = dict(t=t, TPR=100*fb["overall"]["TPR"], FPR=100*fb["overall"]["FPR"], iF_FPR=fb["intersection"]["F_FPR"],
             nFP=int(fp.sum()), nFN=int(fn.sum()), nreal=int(real.sum()), nfake=int(fake.sum()),
             FNR=100*fn.sum()/fake.sum(),
             FPvid=int(df[fp].vid.nunique()), realvid=int(df[real].vid.nunique()),
             FNvid=int(df[fn].vid.nunique()), fakevid=int(df[fake].vid.nunique()))
    dfp = dfake & real; dfn = ~dfake & fake
    r["FP_and_ding"] = int((fp & dfp).sum()); r["FP_ours_only"] = int((fp & ~dfp).sum()); r["FP_ding_only"] = int((dfp & ~fp).sum())
    r["FN_and_ding"] = int((fn & dfn).sum()); r["FN_ours_only"] = int((fn & ~dfn).sum()); r["FN_ding_only"] = int((dfn & ~fn).sum())
    r["jac_FP"] = r["FP_and_ding"] / (fp | dfp).sum(); r["jac_FN"] = r["FN_and_ding"] / (fn | dfn).sum()
    r["pct_dingFN_also_ours"] = 100 * r["FN_and_ding"] / dfn.sum(); r["pct_dingFP_also_ours"] = 100 * r["FP_and_ding"] / dfp.sum()
    r["groups"] = {G[i]: dict(FPR=100*s["FPR"], dev=100*abs(s["FPR"]-fb["overall"]["FPR"]), n_real=s["n_real"],
                   nFP=int((fp & (g == i)).sum()), FPvid=int(df[fp & (g == i)].vid.nunique()), vid=int(df[real & (g == i)].vid.nunique()),
                   FNR=100*(1-s["TPR"])) for i, s in ((G.index(k) if k in G else None, v) for k, v in [(G[int(n.split(',')[0]=='nonmale')*4 + ['asian','white','black','others'].index(n.split(',')[1])], v) for n, v in fb["intersection"]["groups"].items()])}
    vstat = df[real].assign(fp=fp[real]).groupby("vid").fp.agg(["sum", "mean", "size"]).sort_values("sum", ascending=False)
    cs = vstat["sum"].cumsum() / vstat["sum"].sum()
    r["vids_for_half_FP"] = int((cs < 0.5).sum() + 1); r["top15_FP_share"] = 100 * cs.iloc[14]
    r["vids_FP_ge50"] = int((vstat["mean"] >= 0.5).sum()); r["top5_FP_vids"] = list(vstat.index[:6]); r["top_vid_counts"] = vstat.head(8)["sum"].to_dict()
    r["FP_vids_for_half"] = int((cs < 0.5).sum() + 1)
    fstat = df[fake].assign(fn=fn[fake]).groupby("vid").agg(fn=("fn", "sum"), rate=("fn", "mean"), m=("method", "first"))
    r["fakevids_FN_ge50"] = int((fstat.rate >= 0.5).sum()); r["fakevids_FN_ge50_NT"] = int(((fstat.rate >= 0.5) & (fstat.m == "NeuralTextures")).sum())
    R[op] = (r, fp, fn, vstat, fstat)
# Ding at 0.5
fbd = block(df.dl.values, 0.0)
dfp = dfake & real; dfn = ~dfake & fake
ding = dict(FPR=100*fbd["overall"]["FPR"], nFP=int(dfp.sum()), FNR=100*dfn.sum()/fake.sum(), nFN=int(dfn.sum()), iF_FPR=fbd["intersection"]["F_FPR"],
            groups={n: dict(FPR=100*s["FPR"], dev=100*abs(s["FPR"]-fbd["overall"]["FPR"])) for n, s in fbd["intersection"]["groups"].items()})
out = {"ding": ding}
for op in R: out[op] = R[op][0]
r, fp, fn, vstat, fstat = R["matched"]
pooled = r["FPR"]
devs = {k: v["dev"] for k, v in r["groups"].items()}
out["matched_dev_share_3"] = 100 * (devs["M-Black"] + devs["M-Others"] + devs["F-Asian"]) / r["iF_FPR"]
out["matched_dev_share_2male"] = 100 * (devs["M-Black"] + devs["M-Others"]) / r["iF_FPR"]
out["matched_dev_sum3"] = devs["M-Black"] + devs["M-Others"] + devs["F-Asian"]
# spearman
out["spearman_real"] = spearmanr(df.logit[real], df.dl[real]).correlation; out["spearman_fake"] = spearmanr(df.logit[fake], df.dl[fake]).correlation
# F-Asian
fa = fp & (g == 4)
out["FAsian_FP"] = int(fa.sum()); out["FAsian_FP_ding_real"] = int((fa & ~dfake).sum())
fav = df[real & (g == 4)].assign(fp=fp[real & (g == 4)], dfp=dfake[real & (g == 4)]).groupby("vid").agg(fp=("fp", "sum"), rate=("fp", "mean"), drate=("dfp", "mean"), n=("fp", "size")).sort_values("fp", ascending=False)
out["FAsian_top3"] = fav.head(5).to_dict("index"); out["FAsian_top3_share"] = 100 * fav.fp.head(3).sum() / fa.sum()
# video removal
rv = vstat.index.tolist(); rng = np.random.default_rng(0)
def ifpr_drop(drop, fixed):
    keep = ~(real & df.vid.isin(drop).values)
    yy, ll, gg = y[keep], df.logit.values[keep], g[keep]
    if fixed: t = TH["matched"]
    else:
        rl = ll[yy == 0]; t = float(np.quantile(rl, 1 - pooled/100, method="higher"))
    fb = fairness_block(yy, ll, gg, "eight", t)
    return fb["intersection"]["F_FPR"], 100*fb["overall"]["FPR"]
rem = {}
for k in [5, 15, 20]:
    rem[k] = dict(requant=ifpr_drop(rv[:k], False), fixed=ifpr_drop(rv[:k], True))
    rv_all = df[real].vid.unique()
    vals = [ifpr_drop(list(rng.choice(rv_all, k, replace=False)), False)[0] for _ in range(200)]
    rem[k]["random_requant"] = [float(np.median(vals)), float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]
out["removal"] = rem
# demographics
def rate(mask_err, base, col):
    s = df[base].assign(e=mask_err[base]).groupby(col).e.agg(["mean", "sum", "size"]); s["mean"] *= 100; return s.to_dict("index")
out["FPR_ita4"] = rate(fp, real, "ita4"); out["FPR_age"] = rate(fp, real, "age4_if")
out["FPR_xu_male"] = rate(fp, real, "xu_male"); out["FPR_cl_male"] = rate(fp, real, "cl_male")
out["FNR_xu_male"] = rate(fn, fake, "xu_male"); out["FNR_cl_male"] = rate(fn, fake, "cl_male")
out["FN_male_share_xu"] = 100 * (fn & (df.xu_male.values == 1)).sum() / fn.sum()
out["FN_male_share_cl"] = 100 * (fn & (df.cl_male.values == 1)).sum() / fn.sum()
out["FNR_group"] = {G[k]: v for k, v in rate(fn, fake, "group").items()}
# methods
out["FNR_method"] = rate(fn, fake, "method")
out["FN_share_NT"] = 100 * (fn & (df.method.values == "NeuralTextures")).sum() / fn.sum()
reen = fake & np.isin(df.method.values, ["Face2Face", "NeuralTextures"])
out["FNR_reen"] = 100 * (fn & reen).sum() / reen.sum(); out["FN_share_reen"] = 100 * (fn & reen).sum() / fn.sum()
out["fakevids_FN_ge50"] = int((fstat.rate >= 0.5).sum()); out["fakevids_FN_ge50_NT"] = int(((fstat.rate >= 0.5) & (fstat.m == "NeuralTextures")).sum())
oo = fn & ~dfn
out["ours_only_FN_NT"] = int((oo & (df.method.values == "NeuralTextures")).sum())
v = df.vid.str.endswith("_NeuralTextures_306_278")
out["NT_306_278"] = dict(n=int(v.sum()), ours_fn=int((fn & v).sum()), ding_caught=int((dfake & v).sum()))
# targets
df["target"] = df.vid.str.extract(r"^\d+_(?!youtube)[^_]+_(\d+)_\d+$")[0]
df["rid"] = df.vid.str.extract(r"^\d+_youtube_(\d+)$")[0]
ff = df[fake].assign(fn=fn[fake], dfn=dfn[fake])
tg = ff.groupby("target").agg(fn=("fn", "sum"), fnr=("fn", "mean"), dfnr=("dfn", "mean"), fml=("logit", "mean"))
out["n_targets"] = len(tg)
c = tg.fn.sort_values(ascending=False).cumsum() / tg.fn.sum()
out["targets_half_FN"] = int((c < 0.5).sum() + 1); out["top10_target_share"] = 100 * c.iloc[9]
vm = ff.groupby(["target", "method"]).fn.mean().reset_index()
multi = vm[vm.fn >= 0.5].groupby("target").size(); out["targets_ge50_2methods"] = sorted(multi[multi >= 2].index.tolist())
rr = df[real].assign(fp=fp[real]).groupby("rid").agg(rml=("logit", "mean"), fpr=("fp", "mean"))
j = tg.join(rr, how="inner"); out["n_targets_with_real"] = len(j)
out["sp_realml_fnr_ours"] = spearmanr(j.rml, j.fnr).correlation; out["sp_realml_fnr_ding"] = spearmanr(j.rml, j.dfnr).correlation
# Ding's version may use Ding's own real logit:
jd = j.join(df[real].groupby("rid").dl.mean().rename("drml")); out["sp_dingrealml_dingfnr"] = spearmanr(jd.drml, jd.dfnr).correlation
out["sp_realml_fakeml"] = spearmanr(j.rml, j.fml).correlation; out["sp_realfpr_fnr"] = spearmanr(j.fpr, j.fnr).correlation
t800 = ff[ff.target == "800"]
out["t800"] = dict(real_ml=float(rr.loc["800", "rml"]) if "800" in rr.index else None, fn=int(t800.fn.sum()), by_method=t800.groupby("method").fn.sum().to_dict())
# A1 FP concentration
r1, fp1, fn1, vs1, fs1 = R["A1"]
c1 = vs1["sum"].cumsum() / vs1["sum"].sum()
out["A1_FP"] = dict(n=int(fp1.sum()), vids=int((vs1["sum"] > 0).sum()), vids_half=int((c1 < 0.5).sum() + 1), top=vs1.head(7)["sum"].to_dict())
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_out.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
# --- tie-break check for video removal (top-5 tie at 20 FP frames: 237, 438, 062)
print("ranks 4-8:", vstat.iloc[3:8]["sum"].to_dict(), "ranks 13-17:", vstat.iloc[12:17]["sum"].to_dict())
for drop in [["0_youtube_305", "0_youtube_845", "0_youtube_307", "0_youtube_383", "0_youtube_438"],
             ["0_youtube_305", "0_youtube_845", "0_youtube_307", "0_youtube_383", "0_youtube_237"],
             ["0_youtube_305", "0_youtube_845", "0_youtube_307", "0_youtube_383", "0_youtube_062"]]:
    print(drop[-1], "requant", ifpr_drop(drop, False), "fixed", ifpr_drop(drop, True))
# stable sort by (sum desc, vid asc) and by (sum desc, rate desc)
for name, vs in [("vid asc", vstat.reset_index().sort_values(["sum", "vid"], ascending=[False, True])),
                 ("rate desc", vstat.reset_index().sort_values(["sum", "mean"], ascending=[False, False]))]:
    lst = vs.vid.tolist(); print(name, lst[:6], {k: ifpr_drop(lst[:k], False)[0] for k in (5, 15, 20)}, "fixed15", ifpr_drop(lst[:15], True))
# --- contact-sheet check: independent tile selection (FP: highest-logit FP frame per video, top-24 videos; FN: lowest-logit FN frame per video)
from PIL import Image
T = f"{A}/BEAT_DING/results/failure_cases/ffpp_test"
gname = {i: n for i, n in enumerate(G)}
for kind, err, asc in [("fp", fp, False), ("fn", fn, True)]:
    e = df[err].sort_values("logit", ascending=asc).drop_duplicates("vid").head(24)
    tiles = pd.read_csv(f"{T}/tables/{kind}_sheet_tiles.csv")
    same = set(e.img_path) == set(tiles.img_path)
    dw = (e.dl >= 0) if kind == "fp" else (e.dl < 0)
    m = tiles.merge(df[["img_path", "logit", "dl", "group", "method", "vid"]], on="img_path", how="left")
    ok = (np.isclose(m.logit_x, m.logit_y, atol=1e-3) & np.isclose(m.ding_logit, m.dl, atol=1e-3) & (m.gname == m.group.map(gname)) & (m.video == m.vid)).all()
    ex = all(os.path.exists(p) for p in tiles.img_path)
    vr = df[real if kind == "fp" else fake].assign(e=err[real if kind == "fp" else fake]).groupby("vid").e.agg(["sum", "size"])
    okv = all((vr.loc[r.video, "sum"] == r.vid_err) and (vr.loc[r.video, "size"] == r.vid_n) for r in tiles.itertuples())
    print(kind, "same set:", same, "ding wrong:", int(dw.sum()), "csv fields match:", ok, "vid_err/n match:", okv, "files exist:", ex)
    for r in tiles.head(3).itertuples():
        im = Image.open(r.img_path); print("  ", os.path.basename(r.img_path), im.size, im.mode, r.gname, r.method, round(r.logit, 1), round(r.ding_logit, 2), f"{r.vid_err}/{r.vid_n}")
