"""Independent verifier for the Celeb-DF failure-case claims (BD0 seed 7 vs Ding repro). Written from scratch;
reads only the raw score npz files, bd_eval.json and the DEMO_AXES label csv. Prints every number."""
import json, re, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
ev = json.load(open(f"{A}/BEAT_DING/runs/BD0_s7/eval/bd_eval.json"))
T = ev["thresholds"]
s = np.load(f"{A}/BEAT_DING/runs/BD0_s7/eval/celebdf_scores.npz", allow_pickle=True)
dg = np.load(f"{A}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_celebdf.npz", allow_pickle=True)
lab = pd.read_csv(f"{A}/DEMO_AXES/labels/labels_celebdf.csv")

df = pd.DataFrame({k: s[k] for k in ["logit", "label", "group", "img_path"]})
df["logit"] = df.logit.astype(np.float64)
assert (dg["img_path"].astype(str) == df.img_path.values).all()
assert (dg["label"] == df.label.values).all() and (dg["group"] == df.group.values).all()
df["dlogit"] = dg["logit"].astype(np.float64)
lab = lab.set_index("img_path").loc[df.img_path].reset_index()
for c in ["video", "xu_male", "xu_race", "cl_male_vid", "cl_race_vid", "ita4", "ita4_local", "ita_vid"]:
    df[c] = lab[c].values
GN = ["M-White", "M-Black", "M-Others", "F-White", "F-Black", "F-Others"]
df["gname"] = df.group.map(dict(enumerate(GN)))
b = df.img_path.map(os.path.basename).str.replace(".png", "", regex=False)
df["frame"] = b.str.extract(r"_(\d+)$")[0].astype(int)
df["vid"] = b.str.replace(r"_\d+$", "", regex=True)
assert (df.vid == df.video).all()
df["src"] = df.vid.str.extract(r"^\d_(Celeb-real|YouTube-real|Celeb-synthesis)")[0]
m = df.vid.str.extract(r"Celeb-synthesis_id(\d+)_id(\d+)_")
df["tgt"], df["srcid"] = m[0], m[1]
df["rid"] = df.vid.str.extract(r"Celeb-real_id(\d+)_")[0]
y = df.label.values
R, F = y == 0, y == 1
df["dpred"] = df.dlogit >= 0
out = {}

def pr(*a):
    print(*a, flush=True)

def ifpr(pred, groups, yy):
    """sum_g |FPR_g - FPR_all| over groups having both classes (intersection partition)."""
    fall = pred[yy == 0].mean()
    tot, terms = 0.0, {}
    for g in np.unique(groups):
        mg = groups == g
        if (yy[mg] == 0).any() and (yy[mg] == 1).any():
            terms[g] = 100 * abs(pred[mg & (yy == 0)].mean() - fall)
            tot += terms[g]
    return tot, terms

for op in ["matched", "native", "A1"]:
    p = (df.logit >= T[op]).values
    c = ev["test"]["celebdf"][op]["cols"]
    fpr, tpr = 100 * p[R].mean(), 100 * p[F].mean()
    ii, _ = ifpr(p, df.group.values, y)
    pr(op, "t", T[op], "FPR", fpr, "TPR", tpr, "FNR", 100 - tpr, "iF_FPR", ii, "| bd_eval", c["FPR"], c["TPR"], c["iF_FPR"])
    assert abs(fpr - c["FPR"]) < 1e-3 and abs(tpr - c["TPR"]) < 1e-3 and abs(ii - c["iF_FPR"]) < 1e-3

pr("\n==== CLAIM 1 ====")
t = T["matched"]; df["p"] = df.logit >= t
pr("median real logit", df.logit[R].median(), "real logit pct 10/25/50/75/90", np.percentile(df.logit[R], [10, 25, 50, 75, 90]))
pr("FP", int(df.p[R].sum()), "/", int(R.sum()), 100 * df.p[R].mean())
rv = df[R].groupby("vid").agg(n=("p", "size"), fp=("p", "sum"), src=("src", "first"), g=("gname", "first"))
rv["rate"] = rv.fp / rv.n
pr("real videos", len(rv), ">=1 FP", (rv.fp >= 1).sum(), ">=50%", (rv.rate >= 0.5).sum(), "100%", (rv.rate == 1).sum())
srt = rv.fp.sort_values(ascending=False).values; cs = np.cumsum(srt) / srt.sum()
pr("top15 share", 100 * cs[14], "videos to reach 50%", int(np.searchsorted(cs, 0.5) + 1))
for sname in ["Celeb-real", "YouTube-real"]:
    mm = R & (df.src == sname).values
    pr(sname, "FPR", 100 * df.p[mm].mean(), "n", mm.sum())
pr("Ding FPR", 100 * df.dpred[R].mean(), "Ding FNR", 100 * (~df.dpred[F]).mean())
pA = df.logit >= T["A1"]
pr("A1 FPR", 100 * pA[R].mean(), "A1 FNR", 100 * (~pA[F]).mean(), "native FPR", 100 * (df.logit >= T["native"])[R].mean())

pr("\n==== CLAIM 2 ====")
def claim_iffpr(pred, tag, nperm=2000, seed=0):
    g = df.group.values
    tot, terms = ifpr(pred, g, y)
    pr(tag, "iF_FPR", tot, {GN[k]: round(v, 2) for k, v in terms.items()})
    for k in terms:
        mm = (g == k) & R
        pr("  ", GN[k], "FPR", 100 * pred[mm].mean(), "n_real", mm.sum(), "n_real_vids", df.vid[mm].nunique(),
           "n_fake", ((g == k) & F).sum(), "n_fake_vids", df.vid[(g == k) & F].nunique())
    # per-video arrays for reals
    vids = df.vid[R].unique()
    vi = {v: i for i, v in enumerate(vids)}
    rvi = df.vid[R].map(vi).values
    rp = pred[R].astype(float); rg = g[R]
    nv = np.bincount(rvi, minlength=len(vids)); fv = np.bincount(rvi, weights=rp, minlength=len(vids))
    vg = np.array([rg[rvi == i][0] for i in range(len(vids))])
    fake_groups = set(np.unique(g[F]))
    def iff_from(keep, vgrp):
        n = nv[keep]; f = fv[keep]; gg = vgrp[keep]
        fall = f.sum() / n.sum(); tot = 0
        for k in fake_groups:
            mk = gg == k
            if mk.any():
                tot += 100 * abs(f[mk].sum() / n[mk].sum() - fall)
        return tot
    full = iff_from(np.ones(len(vids), bool), vg); assert abs(full - tot) < 1e-6
    # drop-one
    d1 = np.array([iff_from(np.arange(len(vids)) != i, vg) for i in range(len(vids))])
    order = np.argsort(-fv)
    pr("  drop top-FP video:", vids[order[0]], "fp", fv[order[0]], "iF->", d1[order[0]], "delta", d1[order[0]] - tot)
    pr("  drop-one over top-10 FP videos max |delta|", np.abs(d1[order[:10]] - tot).max(),
       " over top-20", np.abs(d1[order[:20]] - tot).max(), " over all videos max |delta|", np.abs(d1 - tot).max(),
       "argmax", vids[np.argmax(np.abs(d1 - tot))])
    # greedy
    keep = np.ones(len(vids), bool); cur = tot; steps = []
    while cur > tot / 2 and keep.sum() > 1:
        best, bi = None, None
        for i in np.where(keep)[0]:
            k2 = keep.copy(); k2[i] = False
            v = iff_from(k2, vg)
            if best is None or v < best:
                best, bi = v, i
        keep[bi] = False; cur = best
        steps.append((vids[bi], GN[vg[bi]], int(fv[bi]), int(nv[bi]), round(cur, 2)))
    pr("  greedy steps to halve:", len(steps))
    for st in steps:
        pr("    ", st)
    # permutation null: shuffle group labels across real videos
    rng = np.random.default_rng(seed)
    null = np.array([iff_from(np.ones(len(vids), bool), rng.permutation(vg)) for _ in range(nperm)])
    pval = (null >= tot).mean()
    pr("  null mean", null.mean(), "p5,p95", np.percentile(null, [5, 95]), "p", pval, "p(+1)", ((null >= tot).sum() + 1) / (nperm + 1))
    return vids, fv, nv, vg, iff_from

claim_iffpr(df.p.values, "matched")
for i in ["59", "60", "61"]:
    mm = R & (df.rid == i).values
    pr("real id", i, "FPR", 100 * df.p[mm].mean(), "n", mm.sum(), "nvid", df.vid[mm].nunique())

pr("\n==== CLAIM 3 ====")
pA = (df.logit >= T["A1"]).values
vids, fv, nv, vg, iff_from = claim_iffpr(pA, "A1")
for rm in [["0_YouTube-real_00262"], ["0_YouTube-real_00262", "0_YouTube-real_00255"]]:
    keep = ~np.isin(vids, rm)
    pr("  remove", rm, "iF_FPR", iff_from(keep, vg))
fo = df[R & (df.gname == "F-Others").values].groupby("vid").agg(n=("p", "size"), cl_race=("cl_race_vid", "first"),
                                                                    cl_male=("cl_male_vid", "first"), src=("src", "first"))
fo["A1_fp"] = df[R & (df.gname == "F-Others").values].assign(a=pA[R & (df.gname == "F-Others").values]).groupby("vid").a.sum()
pr(fo)
pr("F-Others fake videos", df.vid[F & (df.gname == "F-Others").values].unique(),
   df[F & (df.gname == "F-Others").values].cl_race_vid.unique())
srt = np.sort(fv)[::-1]; cs = np.cumsum(srt) / srt.sum()
pr("A1 FP videos touched", (fv > 0).sum(), "/", len(fv), "top20 share", 100 * cs[19])

pr("\n==== CLAIM 4 ====")
df["fn"] = F & ~df.p.values
df["dfn"] = F & ~df.dpred.values
df["fnA"] = F & ~pA
fk = df[F]
gt = fk.groupby("gname").agg(n=("fn", "size"), fn=("fn", "sum"), dfn=("dfn", "sum"), fnA=("fnA", "sum"))
gt["FNR"] = 100 * gt.fn / gt.n; gt["dFNR"] = 100 * gt.dfn / gt.n; gt["A1FNR"] = 100 * gt.fnA / gt.n
gt["share_fakes"] = 100 * gt.n / gt.n.sum(); gt["share_FN"] = 100 * gt.fn / gt.fn.sum()
pr(gt)
tg = fk.groupby("tgt").agg(n=("fn", "size"), fn=("fn", "mean"), dfn=("dfn", "mean"), fnA=("fnA", "mean"))
pr(tg.loc[["59", "60", "61"]] * [1, 100, 100, 100])
pp = fk.assign(pair=fk.tgt + "->" + fk.srcid).groupby("pair").agg(n=("fn", "size"), fn=("fn", "mean"), dfn=("dfn", "mean"))
pr(pp[pp.index.str.match(r"^(59|60|61)->")] * [1, 100, 100])
pr("M-Black fake target/source ids", fk[fk.gname == "M-Black"].tgt.unique(), fk[fk.gname == "M-Black"].srcid.unique())
for gname in ["M-Black", "F-White"]:
    mg = (df.gname == gname).values
    pr(gname, "median real logit", df.logit[mg & R].median(), "median fake logit", df.logit[mg & F].median(),
       "AUC", 100 * roc_auc_score(y[mg], df.logit[mg]), "Ding AUC", 100 * roc_auc_score(y[mg], df.dlogit[mg]))

pr("\n==== CLAIM 5 ====")
f1 = (df.frame == 1).values
for tag, pred in [("BD matched", df.p.values), ("Ding", df.dpred.values), ("BD A1", pA)]:
    pr(tag, "FNR f1", 100 * (~pred[F & f1]).mean(), int((~pred[F & f1]).sum()), "/", int((F & f1).sum()),
       "FNR other", 100 * (~pred[F & ~f1]).mean(), "| FPR f1", 100 * pred[R & f1].mean(), int(pred[R & f1].sum()), "/", int((R & f1).sum()),
       "FPR other", 100 * pred[R & ~f1].mean(), "| f1 share of FN", 100 * (~pred[F & f1]).sum() / (~pred[F]).sum())
pr("frames per video min/max", df.groupby("vid").size().min(), df.groupby("vid").size().max(), "videos with frame1", df.vid[f1].nunique())

pr("\n==== CLAIM 6 ====")
for tag, pred in [("matched", df.p.values), ("A1", pA)]:
    fp = R & pred; fn = F & ~pred; dfp = R & df.dpred.values; dfn = F & ~df.dpred.values
    pr(tag, "our FN", fn.sum(), "Ding also misses", 100 * (fn & dfn).sum() / fn.sum(), "Ding base miss", 100 * dfn.sum() / F.sum(),
       "ratio", ((fn & dfn).sum() / fn.sum()) / (dfn.sum() / F.sum()))
    pr(tag, "our FP", fp.sum(), "Ding also flags", 100 * (fp & dfp).sum() / fp.sum(), "Ding base FPR", 100 * dfp.sum() / R.sum(),
       "ratio", ((fp & dfp).sum() / fp.sum()) / (dfp.sum() / R.sum()))
    pr(tag, "only-ours FP", (fp & ~dfp).sum(), "FN", (fn & ~dfn).sum(), "| only-Ding FP", (dfp & ~fp).sum(), "FN", (dfn & ~fn).sum())
    fv_ = df[F].assign(o=fn[F], d=dfn[F]).groupby("vid").agg(o=("o", "mean"), d=("d", "mean"))
    pr(tag, "fake videos ours>=50% & Ding<50%", ((fv_.o >= 0.5) & (fv_.d < 0.5)).sum())

pr("\n==== CLAIM 7 ====")
rr = df[R].assign(o=df.p[R], d=df.dpred[R]).groupby("vid").agg(o=("o", "mean"), d=("d", "mean"), g=("gname", "first"))
u = rr[(rr.o >= 0.5) & (rr.d < 0.5)]
pr("real ours>=50 & Ding<50:", len(u), u.g.value_counts().to_dict())
pr(rr.loc[["0_Celeb-real_id57_0001", "0_Celeb-real_id55_0006"]] if all(v in rr.index for v in ["0_Celeb-real_id57_0001", "0_Celeb-real_id55_0006"]) else "missing ids")
pr("real other way (Ding>=50 & ours<50):", ((rr.d >= 0.5) & (rr.o < 0.5)).sum())
ff = df[F].assign(o=df.fn[F], d=df.dfn[F]).groupby("vid").agg(o=("o", "mean"), d=("d", "mean"), tgt=("tgt", "first"))
u2 = ff[(ff.o >= 0.5) & (ff.d < 0.5)]
pr("fake ours-missed>=50 & Ding<50:", len(u2)); pr(u2)
pr("fake other way:", ((ff.d >= 0.5) & (ff.o < 0.5)).sum())
pr("total other way (real+fake):", ((rr.d >= 0.5) & (rr.o < 0.5)).sum() + ((ff.d >= 0.5) & (ff.o < 0.5)).sum())

pr("\n==== CLAIM 8 ====")
fpd = df[R & df.p.values].sort_values("logit", ascending=False)
pr(fpd.head(12)[["vid", "frame", "logit", "dpred", "gname"]])
fnd = df[F & ~df.p.values].sort_values("logit")
pr(fnd.head(12)[["vid", "frame", "logit", "dpred", "gname"]])
s_fp = fpd.drop_duplicates("vid").head(24); s_fn = fnd.drop_duplicates("vid").head(24)
pr("FP sheet groups", s_fp.gname.value_counts().to_dict(), "Ding flags", int(s_fp.dpred.sum()))
pr("FN sheet groups", s_fn.gname.value_counts().to_dict(), "Ding detects", int(s_fn.dpred.sum()))
s_fp[["vid", "frame", "logit", "dlogit", "gname", "img_path"]].to_csv(os.path.dirname(__file__) + "/my_fp_sheet_rows.csv", index=False)
s_fn[["vid", "frame", "logit", "dlogit", "gname", "img_path"]].to_csv(os.path.dirname(__file__) + "/my_fn_sheet_rows.csv", index=False)

pr("\n==== CLAIM 9 ====")
vv = df[R].groupby("vid").agg(xm=("xu_male", "first"), xr=("xu_race", "first"), cm=("cl_male_vid", "first"),
                               cr=("cl_race_vid", "first"), o=("p", "mean"), nun=("cl_race_vid", "nunique"))
xr_map = vv.xr.replace({"others": "others"}); cr_map = vv.cr.replace({"asian": "others"})
dis = vv[(vv.xm != vv.cm) | (xr_map != cr_map)]
pr("disagree (race asian->others mapped):", len(dis)); pr(dis)
dis2 = vv[(vv.xm != vv.cm) | (vv.xr != vv.cr)]
pr("disagree (raw strings):", len(dis2))
for col in ["xu_male", "cl_male_vid"]:
    pr(col, {k: round(100 * df.p[R & (df[col] == k).values].mean(), 2) for k in [1, 0]})
pr("M-Black identities (real)", df.rid[R & (df.gname == "M-Black").values].unique(), "fake tgt", df.tgt[F & (df.gname == "M-Black").values].unique())
for col in ["ita4", "ita4_local"]:
    pr(col, {k: round(100 * df.p[R & (df[col] == k).values].mean(), 2) for k in [0, 1, 2, 3]},
       "n_real_vids", {k: df.vid[R & (df[col] == k).values].nunique() for k in [0, 1, 2, 3]})
