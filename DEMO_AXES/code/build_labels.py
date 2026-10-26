"""Multi-axis label tables (one CSV per split / test set), keyed EXACTLY like the score npz `img_path`.

  ~/.conda/envs/aiface/bin/python build_labels.py          (needs labels/raw/face_attrs_<set>.csv.gz)

Outputs (labels/):
  labels_<set>.csv          set in ffpp_train, ffpp_val, ffpp_test, dfdc, celebdf, dfd, ff2_test
  ita_bin_edges.json        4-bin ITA edges = quartiles of the TRAIN distribution, declared here (no score is read
                            anywhere in this script): FF++ protocol -> FF++ TRAIN real videos (per-video median);
                            FF2 -> FF2-public TRAIN reals (per-person median)
  ita_stability.csv/.json   per-video ITA stability
  coverage.csv              label coverage per set and column
  human_check_sample.csv    stratified sample for the human skin-tone check (no study is run)

Columns (frame = one row of the protocol CSV):
  img_path, label, video, person
  xu_male, xu_race, xu_group               original Xu labels (= the Ding/Lin protocol labels; "noisy" in Lin et al.)
  cl_male, cl_race, cl_src                 AI-Face "cleaner" gender/race (Purdue robust-fairness release, v2);
                                           cl_src = 'aiface_v2' or 'xu_fallback' (row missing there, e.g. FF++ val)
  cl_male_vid, cl_race_vid                 video-majority of cl_male / cl_race (ties -> Xu label); FF2: = cl_*
  cl_age_v2                                v2 Ground Truth Age (0 young / 1 middle / 2 senior / 3 others)
  aif_mst, aif_mst3, aif_age5, aif_male    AI-Face v1 annotations (MST 1-10; MST group 0 = 1-3, 1 = 4-6, 2 = 7-10;
                                           Predicted Age 0 child 0-14 / 1 youth 15-24 / 2 adult 25-44 /
                                           3 middle 45-64 / 4 senior 65+; Predicted Gender 1 = male)
  if_age, if_male, det_score               InsightFace buffalo_l genderage, per frame (ours, face_attrs.py)
  ita_raw, ita_gw, ita_gwexp               per-frame ITA (face_attrs.py; ita_gw = PRIMARY)
  ita_vid, ita_vid_n, ita_vid_iqr          per-video median / number of valid frames / IQR of ita_gw
  ita4                                     ITA bin of ita_vid (0 = darkest ... 3 = lightest), declared edges
  ita4_raw, ita4_gwexp                     same rule on the sensitivity variants (their own TRAIN quartiles)
  ita4_local                               sensitivity: ita_vid binned at the test set's own real-video quartiles
  age4_aif                                 AI-Face age, video majority, collapsed to 4 bins:
                                           0 = child+youth (<25), 1 = adult (25-44), 2 = middle (45-64), 3 = senior (65+)
  age4_if                                  InsightFace numeric age, per-video median, bins <30 / 30-44 / 45-59 / 60+
  mst3_vid                                 AI-Face MST group, video majority
FF2 (one image per row; video = person = person_id; no video aggregation): gender/race = FF2 manifest labels
(cl_* = xu_* = FF2 labels, cl_src = 'ff2'), age4_if from the manifest `if_age` (per image, InsightFace buffalo_l;
our re-estimate is stored as if_age_ours), ita4 from the per-image ita_gw with the FF2 TRAIN edges.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
V = f"{R}/agents/DEMO_AXES"
D = f"{R}/agents/DINGREPRO_LIN/data"
F = f"{R}/release_ff2_public"
I = "/home/sachin.chaudhary/ICLR_2027/datasets_external/AI-Face/annotations"
N = "/home/sachin.chaudhary/models_external/robust_fairness_withNoisy_labels/dataset"
OUT = f"{V}/labels"
RAW = f"{OUT}/raw"
sys.path.insert(0, f"{R}/agents/CFES/scripts")
sys.path.insert(0, f"{V}/code")
from verify_metrics import identity, video  # noqa: E402
from make_lists import rel  # noqa: E402

SETS = {"ffpp_train": (["ff++_ding/train.csv"], "ff++", ["ff++/train_real.csv", "ff++/train_fake_spe.csv"]),
        "ffpp_val": (["ff++_ding/realval.csv", "ff++_ding/fakeval.csv"], "ff++", []),
        "ffpp_test": (["ff++/test.csv"], "ff++", ["ff++/test_noise.csv"]),
        "dfdc": (["dfdc/test.csv"], "dfdc", ["dfdc/test_noise.csv"]),
        "celebdf": (["celebdf/test.csv"], "celebdf", ["celebdf/test_noise.csv"]),
        "dfd": (["dfd/test.csv"], "dfd", ["dfd/test_noise.csv"])}
RACE = {0: "asian", 1: "white", 2: "black", 3: "others"}
ITA_COLS = ("ita_gw", "ita_raw", "ita_gwexp")


def xu_race(df):
    r = np.full(len(df), "others", dtype=object)
    for c, n in (("isblack", "black"), ("iswhite", "white"), ("isasian", "asian")):
        if c in df:
            r[df[c].values == 1] = n
    return r


def age_bin_num(a):
    a = np.asarray(a, dtype=float)
    out = np.full(len(a), np.nan)
    ok = np.isfinite(a)
    out[ok] = np.digitize(a[ok], [30, 45, 60])
    return out


def vid_majority(df, col):
    m = df.dropna(subset=[col]).groupby("video")[col].agg(lambda s: s.value_counts().sort_index().idxmax())
    return df.video.map(m)


def load_aiface():
    af = pd.concat([pd.read_csv(f"{I}/train_data.csv"), pd.read_csv(f"{I}/test_data.csv")])
    af["ds"] = af["Image Path"].str.extract(r"/deepfakes/([^/]+)/")[0]
    af = af[af.ds.notna()].copy()
    af["b"] = af["Image Path"].map(os.path.basename)
    return af.drop_duplicates(["ds", "b"])


def protocol_tables(af):
    tabs = {}
    for s, (fs, ds, pf) in SETS.items():
        o = pd.concat([pd.read_csv(f"{D}/{f}") for f in fs], ignore_index=True)
        o["b"] = o.img_path.map(os.path.basename)
        o["video"] = o.img_path.map(video)
        o["person"] = o.img_path.map(identity)
        o["xu_male"] = (o.ismale == 1).astype(int)
        o["xu_race"] = xu_race(o)
        o["xu_group"] = o.intersec_label
        # cleaner (v2) labels
        if pf:
            p = pd.concat([pd.read_csv(f"{N}/{f}") for f in pf], ignore_index=True)
            p["b"] = p["Image Path"].map(os.path.basename)
            p = p.drop_duplicates("b").set_index("b")
            q = p.reindex(o.b)
            hit = q["Ground Truth Gender"].notna().values
        else:
            q, hit = None, np.zeros(len(o), bool)
        o["cl_src"] = np.where(hit, "aiface_v2", "xu_fallback")
        o["cl_male"] = np.where(hit, q["Ground Truth Gender"].values if q is not None else 0, o.xu_male).astype(int)
        o["cl_race"] = np.where(hit, pd.Series(q["Ground Truth Race"].values if q is not None else np.zeros(len(o))).map(RACE).values,
                                o.xu_race)
        o["cl_age_v2"] = q["Ground Truth Age"].values if q is not None else np.nan
        # AI-Face v1
        a = af[af.ds == ds].set_index("b").reindex(o.b)
        o["aif_mst"], o["aif_mst3"] = a["Skin Tone"].values, a["Skin Tone Group"].values
        o["aif_age5"], o["aif_male"] = a["Predicted Age"].values, a["Predicted Gender"].values
        # our per-frame attributes
        fa = pd.read_csv(f"{RAW}/face_attrs_{s}.csv.gz").drop_duplicates("path").set_index("path").reindex(o.img_path)
        for c in ("if_age", "if_male", "det_score", "roi_mode", "gray") + ITA_COLS:
            o[c] = fa[c].values
        # video aggregation
        for c in ITA_COLS:
            g = o.dropna(subset=[c]).drop_duplicates("img_path").groupby("video")[c]
            o[c.replace("ita", "itavid")] = o.video.map(g.median())
        g = o.dropna(subset=["ita_gw"]).drop_duplicates("img_path").groupby("video").ita_gw
        o["ita_vid"] = o.itavid_gw
        o["ita_vid_n"] = o.video.map(g.size()).fillna(0).astype(int)
        o["ita_vid_iqr"] = o.video.map(g.quantile(0.75) - g.quantile(0.25))
        o["age_vid_if"] = o.video.map(o.dropna(subset=["if_age"]).drop_duplicates("img_path").groupby("video").if_age.median())
        o["age4_if"] = age_bin_num(o.age_vid_if)
        av = vid_majority(o, "aif_age5")
        o["age4_aif"] = av.map({0: 0, 1: 0, 2: 1, 3: 2, 4: 3})
        o["mst3_vid"] = vid_majority(o, "aif_mst3")
        # video-majority version of the frame-level cleaner labels (AI-Face labels flip within 2-40 % of videos);
        # ties -> the video's Xu label when it is among the tied values, else the alphabetically first value
        for c, x in (("cl_male", "xu_male"), ("cl_race", "xu_race")):
            def maj(df):
                vc = df[c].value_counts()
                top = sorted(vc[vc == vc.max()].index.tolist(), key=str)
                xv = df[x].iloc[0]
                return xv if xv in top else top[0]
            o[c + "_vid"] = o.video.map(o.drop_duplicates("img_path").groupby("video")[[c, x]].apply(maj))
        tabs[s] = o
    return tabs


def ff2_tables():
    man = pd.read_csv(f"{F}/manifests/public_manifest.csv", low_memory=False)
    T = pd.read_csv(f"{R}/data_ding_schema/fairforge2_full/test.csv")
    T["rel"] = T.img_path.map(rel)
    mt = man[man.split == "TEST"].set_index("rel_path")
    T = T[T.rel.isin(mt.index)].copy()
    m = mt.reindex(T.rel)
    T["cell_valid"] = m.use_in_cell_metrics.fillna(False).astype(bool).values
    # cluster = person; de-novo T2I/GAN fakes have no person (18.6 % of TEST fakes) -> their own image is the cluster
    T["person"] = np.where(m.person_id.notna().values, m.person_id.values, "img:" + T.rel.values)
    T["video"] = T.person
    T["method"], T["family"], T["src_dataset"] = m.method.values, m.family.values, m.src_dataset.values
    T["xu_male"] = (T.ismale == 1).astype(int)
    T["xu_race"] = xu_race(T)
    T["xu_group"] = T.intersec_label
    T["cl_male"], T["cl_race"], T["cl_src"] = T.xu_male, T.xu_race, "ff2"
    T["cl_male_vid"], T["cl_race_vid"] = T.cl_male, T.cl_race
    T["if_age"] = m.if_age.values
    T["age4_if"] = age_bin_num(T.if_age)
    fa = pd.read_csv(f"{RAW}/face_attrs_ff2_test.csv.gz").drop_duplicates("path").set_index("path").reindex(T.img_path)
    T["if_age_ours"], T["if_male_ours"], T["det_score"] = fa.if_age.values, fa.if_male.values, fa.det_score.values
    for c in ("roi_mode", "gray") + ITA_COLS:
        T[c] = fa[c].values
    T["ita_vid"] = T.ita_gw
    T["itavid_raw"], T["itavid_gwexp"] = T.ita_raw, T.ita_gwexp
    T["ita_vid_n"] = T.ita_gw.notna().astype(int)
    return T


def ff2_train_ita():
    tr = pd.read_csv(f"{F}/manifests/ding/realtrain.csv")
    man = pd.read_csv(f"{F}/manifests/public_manifest.csv", low_memory=False, usecols=["rel_path", "person_id"]).set_index("rel_path")
    from make_lists import unrel
    tr["abs"] = tr.img_path.map(unrel)
    tr["person"] = man.reindex(tr.img_path).person_id.values
    fa = pd.read_csv(f"{RAW}/face_attrs_ff2_trainreal.csv.gz").drop_duplicates("path").set_index("path").reindex(tr["abs"])
    for c in ITA_COLS:
        tr[c] = fa[c].values
    return tr


def edges_from(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [25, 50, 75])], int(len(x))


def stability(o, s):
    f = o.dropna(subset=["ita_gw"]).drop_duplicates("img_path")
    g = f.groupby("video").ita_gw
    sd, iqr, n = g.std(), g.quantile(0.75) - g.quantile(0.25), g.size()
    rows = {"set": s, "videos": int(o.video.nunique()), "videos_with_valid_ita": int(n.size),
            "frames_valid_frac": float(o.drop_duplicates("img_path").ita_gw.notna().mean()),
            "median_frames_per_video": float(n.median()), "within_video_sd_median": float(sd.median()),
            "within_video_iqr_median": float(iqr.median()),
            "between_video_sd": float(g.median().std())}
    # ICC(1) of frame ITA within video (one-way random effects), videos with >= 3 valid frames
    k = n[n >= 3].index
    ff = f[f.video.isin(k)]
    gm = ff.groupby("video").ita_gw
    nbar = gm.size().mean()
    msb = (gm.size() * (gm.mean() - ff.ita_gw.mean()) ** 2).sum() / (len(k) - 1)
    msw = ((ff.ita_gw - ff.video.map(gm.mean())) ** 2).sum() / (len(ff) - len(k))
    rows["icc1"] = float((msb - msw) / (msb + (nbar - 1) * msw))
    return rows


def main():
    af = load_aiface()
    tabs = protocol_tables(af)
    ff2 = ff2_tables()
    tr2 = ff2_train_ita()
    # ---- declared ITA bin edges (TRAIN quartiles; no score read anywhere in this script)
    ftr = tabs["ffpp_train"]
    edges = {"declared": "2026-10-26, before any score was read; quartiles of the TRAIN distribution",
             "ffpp_protocol": {}, "ff2": {}}
    for c in ITA_COLS:
        vcol = c.replace("ita", "itavid")
        v = ftr[ftr.label == 0].drop_duplicates("video")[vcol]
        edges["ffpp_protocol"][c] = dict(zip(("edges", "n_real_train_videos"), edges_from(v)))
        p = tr2.dropna(subset=[c]).groupby("person")[c].median()
        edges["ff2"][c] = dict(zip(("edges", "n_real_train_persons"), edges_from(p)))
    edges["unit"] = {"ffpp_protocol": "per-video median of frame ITA over FF++ TRAIN real videos (Ding/Lin split)",
                     "ff2": "per-person median of image ITA over FF2-public TRAIN reals (manifests/ding/realtrain.csv)"}
    edges["bins"] = "ita4 = 0 if ITA < q25; 1 if < q50; 2 if < q75; 3 otherwise (0 = darkest skin, 3 = lightest)"
    edges["local_sensitivity"] = {"rule": "ita4_local (SENSITIVITY, added 26 Oct before any score was read, after "
                                  "seeing that the FF++-TRAIN edges put most DFDC videos in Q1 because DFDC footage is "
                                  "darker overall): quartiles of the test set's OWN real-video (FF2: real-person) "
                                  "median ita_gw; label- and score-free", "edges": {}}

    def binit(x, e):
        x = np.asarray(x, dtype=float)
        out = np.full(len(x), np.nan)
        ok = np.isfinite(x)
        out[ok] = np.digitize(x[ok], e)
        return out
    stab, cov = [], []
    for s, o in list(tabs.items()) + [("ff2_test", ff2)]:
        key = "ff2" if s == "ff2_test" else "ffpp_protocol"
        o["ita4"] = binit(o.ita_vid, edges[key]["ita_gw"]["edges"])
        o["ita4_raw"] = binit(o.itavid_raw, edges[key]["ita_raw"]["edges"])
        o["ita4_gwexp"] = binit(o.itavid_gwexp, edges[key]["ita_gwexp"]["edges"])
        le, nle = edges_from(o[o.label == 0].drop_duplicates("video").ita_vid)
        edges["local_sensitivity"]["edges"][s] = {"edges": le, "n_real_units": nle}
        o["ita4_local"] = binit(o.ita_vid, le)
        if s != "ff2_test":
            stab.append(stability(o, s))
            # share of videos whose valid frames span > 1 bin
            f = o.dropna(subset=["ita_gw"]).drop_duplicates("img_path")
            fb = binit(f.ita_gw, edges[key]["ita_gw"]["edges"])
            stab[-1]["videos_frames_span_gt1_bin"] = float(pd.Series(fb).groupby(f.video.values).nunique().gt(1).mean())
            stab[-1]["videos_frame_majority_bin_eq_median_bin"] = float(
                (pd.Series(fb).groupby(f.video.values).agg(lambda q: q.value_counts().idxmax())
                 == f.groupby("video").ita_gw.median().pipe(lambda m: pd.Series(binit(m.values, edges[key]["ita_gw"]["edges"]), index=m.index))).mean())
        cols = ["img_path", "label", "video", "person", "xu_male", "xu_race", "xu_group", "cl_male", "cl_race", "cl_src",
                "cl_male_vid", "cl_race_vid",
                "cl_age_v2", "aif_mst", "aif_mst3", "aif_age5", "aif_male", "mst3_vid", "if_age", "if_male", "det_score",
                "roi_mode", "gray", "ita_raw", "ita_gw", "ita_gwexp", "ita_vid", "ita_vid_n", "ita_vid_iqr",
                "itavid_raw", "itavid_gwexp", "ita4", "ita4_raw", "ita4_gwexp", "ita4_local", "age_vid_if", "age4_if", "age4_aif"]
        if s == "ff2_test":
            cols = [c for c in cols if c in o] + ["if_age_ours", "if_male_ours", "cell_valid", "method", "family", "src_dataset"]
        cols = [c for c in cols if c in o]
        o[cols].to_csv(f"{OUT}/labels_{s}.csv", index=False, float_format="%.4g")
        for c in cols[4:]:
            if c in ("roi_mode",):
                continue
            cov.append({"set": s, "column": c, "rows": len(o), "non_null": int(o[c].notna().sum()),
                        "frac": float(o[c].notna().mean())})
        cov.append({"set": s, "column": "cl_src==aiface_v2|ff2", "rows": len(o),
                    "non_null": int(o.cl_src.isin(["aiface_v2", "ff2"]).sum()), "frac": float(o.cl_src.isin(["aiface_v2", "ff2"]).mean())})
        print(s, len(o), "ita4 cov", round(float(o.ita4.notna().mean()), 3), "bins", o.drop_duplicates("video").ita4.value_counts().sort_index().to_dict())
    json.dump(edges, open(f"{OUT}/ita_bin_edges.json", "w"), indent=1)
    pd.DataFrame(stab).to_csv(f"{OUT}/ita_stability.csv", index=False, float_format="%.4g")
    pd.DataFrame(cov).to_csv(f"{OUT}/coverage.csv", index=False, float_format="%.4g")
    # ---- construct checks (label-only): ITA vs AI-Face MST and vs race labels, per set (video level)
    chk = []
    from scipy.stats import spearmanr
    for s, o in list(tabs.items()) + [("ff2_test", ff2)]:
        v = o.drop_duplicates("video")
        for c in ("ita_vid", "itavid_raw", "itavid_gwexp"):
            row = {"set": s, "ita": c}
            if "aif_mst" in v and v.aif_mst.notna().any():
                mv = o.groupby("video").aif_mst.median().reindex(v.video)
                ok = np.isfinite(v[c].values) & np.isfinite(mv.values)
                row["spearman_vs_aif_mst(video median)"] = float(spearmanr(v[c].values[ok], -mv.values[ok])[0])
            for r in ("white", "black", "asian", "others"):
                row[f"median_{r}(cl_race)"] = float(v[v.cl_race == r][c].median())
            row["auc_white_vs_black"] = _auc(v[v.cl_race == "white"][c].dropna().values, v[v.cl_race == "black"][c].dropna().values)
            chk.append(row)
    pd.DataFrame(chk).to_csv(f"{OUT}/ita_construct_checks.csv", index=False, float_format="%.4g")
    # ---- stratified sample for the human skin-tone check
    sample_for_raters(tabs, ff2)


def _auc(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


def sample_for_raters(tabs, ff2, n_total=300, seed=20261026):
    """300 items, stratified over dataset x ITA bin x gender (cleaner labels), one item per video/person. Reals only
    (the rated construct is the person's apparent skin tone; fakes are a separate question). Per dataset 60 items
    (FF++ test, DFDC, Celeb-DF, DFD, FF2-public TEST); within a dataset, 4 ITA bins x 2 genders = 8 cells x ~7-8
    items, topped up from the largest cells when a cell is short. For video data the frame shown is the one whose
    ITA is closest to the video median."""
    rng = np.random.default_rng(seed)
    out = []
    srcs = [("ffpp_test", tabs["ffpp_test"]), ("dfdc", tabs["dfdc"]), ("celebdf", tabs["celebdf"]), ("dfd", tabs["dfd"]),
            ("ff2_test", ff2[ff2.cell_valid])]
    per = n_total // len(srcs)
    for s, o in srcs:
        r = o[(o.label == 0) & o.ita4.notna() & o.ita_gw.notna()].copy()
        r["d"] = (r.ita_gw - r.ita_vid).abs()
        rep = r.sort_values("d").drop_duplicates("video")
        cells = {k: g for k, g in rep.groupby(["ita4", "cl_male"])}
        take = {k: min(len(g), per // 8) for k, g in cells.items()}
        short = per - sum(take.values())
        while short > 0:
            k = max(cells, key=lambda q: len(cells[q]) - take[q])
            if len(cells[k]) - take[k] <= 0:
                break
            take[k] += 1
            short -= 1
        for k, g in cells.items():
            if take[k]:
                pick = g.iloc[rng.choice(len(g), take[k], replace=False)]
                for _, x in pick.iterrows():
                    out.append({"dataset": s, "img_path": x.img_path, "video_or_person": x.video, "ita_bin": int(x.ita4),
                                "ita_frame": x.ita_gw, "ita_video_median": x.ita_vid, "cl_male": int(x.cl_male),
                                "cl_race": x.cl_race, "aif_mst": x.get("aif_mst", np.nan)})
    df = pd.DataFrame(out).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "item_id", [f"ST{i:04d}" for i in range(len(df))])
    df.to_csv(f"{OUT}/human_check_sample.csv", index=False, float_format="%.4g")
    print("human check sample", len(df), df.groupby(["dataset", "ita_bin"]).size().unstack(fill_value=0).to_dict())


if __name__ == "__main__":
    main()
