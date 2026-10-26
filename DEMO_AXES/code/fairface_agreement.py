"""Agreement of FairFace labels (labels_fairface/<set>.csv) with the other label sources; per-video consistency.

  ~/.conda/envs/aiface/bin/python fairface_agreement.py
Sources (all joined by image basename within the dataset, as relabel.py / aiface_join.py do):
  Xu        protocol CSVs (DINGREPRO_LIN/data; ismale, iswhite)                         sex, race2
  AIF-v2    Purdue robust-fairness "cleaner" labels (test_noise.csv; train_real / train_fake_spe for FF++ train)
            Ground Truth Gender (1 male), Race (1 white), Age (0 young / 1 middle / 2 senior / 3 others)
  AIF-v1    AI-Face annotations (outputs/aiface_<set>.csv): Predicted Gender (1 male), Predicted Age
            (0 child 0-14 / 1 youth 15-24 / 2 adult 25-44 / 3 middle 45-64 / 4 senior 65+)
  IF        InsightFace buffalo_l numeric age (if_age, labels/labels_<set>.csv of the multi-axis layer, when present)
  FF2       public manifest: gender_native, race_native, if_age (InsightFace), p_adult
Agreement = % equal and Cohen's kappa; age: age2 vs IF age < 30 (same boundary), vs AIF-v1 age < 25 (boundary
differs: 25 vs 30, so agreement is bounded), crosstab vs AIF-v2 age; Spearman of FairFace expected age vs IF age.
Writes results/fairforensics_ids/{agreement.csv, agreement_crosstabs.txt, video_consistency.csv}.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
V = f"{R}/agents/DEMO_AXES"
LF = f"{V}/labels_fairface"
OUT = f"{V}/results/fairforensics_ids"
D = f"{R}/agents/DINGREPRO_LIN/data"
N = "/home/sachin.chaudhary/models_external/robust_fairness_withNoisy_labels/dataset"
PROTO = {"ffpp_test": ["ff++/test.csv"], "dfdc": ["dfdc/test.csv"], "celebdf": ["celebdf/test.csv"], "dfd": ["dfd/test.csv"],
         "ffpp_val": ["ff++_ding/realval.csv", "ff++_ding/fakeval.csv"], "ffpp_train": ["ff++_ding/train.csv"]}
V2 = {"ffpp_test": ["ff++/test_noise.csv"], "dfdc": ["dfdc/test_noise.csv"], "celebdf": ["celebdf/test_noise.csv"],
      "dfd": ["dfd/test_noise.csv"], "ffpp_train": ["ff++/train_real.csv", "ff++/train_fake_spe.csv"], "ffpp_val": []}
MID = np.array([1, 6, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75])
AGE9 = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def agree(a, b, name, s, level, src, attr, extra=None):
    m = pd.notna(a) & pd.notna(b)
    a, b = np.asarray(a)[m], np.asarray(b)[m]
    r = {"set": s, "level": level, "source": src, "attribute": attr, "n": int(m.sum()),
         "agree_pct": 100 * float((a == b).mean()) if len(a) else np.nan,
         "kappa": float(cohen_kappa_score(a.astype(str), b.astype(str))) if len(a) and len(set(a) | set(b)) > 1 else np.nan,
         "fairface_pos_pct": 100 * float((a == name).mean()) if len(a) else np.nan,
         "other_pos_pct": 100 * float((b == name).mean()) if len(b) else np.nan, "pos_class": name}
    if extra:
        r.update(extra)
    return r


def vmaj(df, col):
    return df.groupby("vid")[col].agg(lambda x: x.dropna().value_counts().index[0] if x.notna().any() else np.nan)


def main():
    rows, cons, txt = [], [], []
    for s in ["ffpp_test", "dfdc", "celebdf", "dfd", "ffpp_val", "ffpp_train"]:
        f = f"{LF}/{s}.csv"
        if not os.path.exists(f):
            continue
        L = pd.read_csv(f, low_memory=False)
        P = pd.concat([pd.read_csv(f"{D}/{x}") for x in PROTO[s]]).drop_duplicates("img_path")
        L = L.merge(P[["img_path", "label", "ismale", "iswhite"]], on="img_path", how="left")
        P = P[[c for c in ("img_path", "isblack", "iswhite", "isasian") if c in P]]
        L["b"] = L.img_path.map(os.path.basename)
        L["xu_sex"] = np.where(L.ismale == 1, "Male", "Female")
        L["xu_race2"] = np.where(L.iswhite == 1, "White", "NonWhite")
        if V2[s]:
            q = pd.concat([pd.read_csv(f"{N}/{x}") for x in V2[s]])
            q["b"] = q["Image Path"].map(os.path.basename)
            q = q.drop_duplicates("b").set_index("b")
            L["v2_sex"] = L.b.map(q["Ground Truth Gender"]).map({1: "Male", 0: "Female"})
            L["v2_race2"] = L.b.map(q["Ground Truth Race"]).map(lambda r: np.nan if pd.isna(r) else ("White" if r == 1 else "NonWhite"))
            L["v2_age"] = L.b.map(q["Ground Truth Age"]).map({0: "young", 1: "middle", 2: "senior", 3: "others"})
            L["v2_age2"] = L.v2_age.map({"young": "0-29", "middle": "30+", "senior": "30+"})   # 'others' -> NaN
        a1 = f"{V}/outputs/aiface_{s}.csv"
        if os.path.exists(a1):
            q = pd.read_csv(a1).drop_duplicates("b").set_index("b")
            L["v1_sex"] = L.b.map(q["Predicted Gender"]).map({1: "Male", 0: "Female"})
            L["v1_age5"] = L.b.map(q["Predicted Age"])
            L["v1_age_lt25"] = L.v1_age5.map(lambda a: np.nan if pd.isna(a) else ("<25" if a <= 1 else "25+"))
        lab = f"{V}/labels/labels_{s}.csv"
        if os.path.exists(lab):
            q = pd.read_csv(lab, usecols=["img_path", "if_age"], low_memory=False).drop_duplicates("img_path").set_index("img_path")
            L["if_age"] = L.img_path.map(q.if_age)
        L["ff_age_exp"] = L[[f"p_age_{a}" for a in AGE9]].values @ MID
        L["ff_age2_cmp25"] = np.where(L.age2 == "0-29", "<25", "25+")   # FairFace <30 vs AIF-v1 <25 (cut differs)
        for level in ("frame", "video"):
            if level == "frame":
                X = L
                ffs, ffr, ffa = X.sex, X.race2, X.age2
            else:
                X = pd.DataFrame({"vid": sorted(L.vid.unique())}).set_index("vid")
                for c in ["xu_sex", "xu_race2", "v2_sex", "v2_race2", "v2_age", "v2_age2", "v1_sex", "v1_age_lt25"]:
                    if c in L:
                        X[c] = vmaj(L, c)
                if "if_age" in L:
                    X["if_age"] = L.groupby("vid").if_age.median()
                X["label"] = L.groupby("vid").label.first()
                X["sex"], X["race2"], X["age2"] = vmaj(L, "vid_sex"), vmaj(L, "vid_race2"), vmaj(L, "vid_age2")
                X["ff_age_exp"] = L.groupby("vid").ff_age_exp.median()
                ffs, ffr, ffa = X.sex, X.race2, X.age2
            for src, cs, cr in (("Xu", "xu_sex", "xu_race2"), ("AIF-v2", "v2_sex", "v2_race2"), ("AIF-v1", "v1_sex", None)):
                if cs in X:
                    rows.append(agree(ffs.values, X[cs].values, "Male", s, level, src, "sex"))
                if cr and cr in X:
                    rows.append(agree(ffr.values, X[cr].values, "White", s, level, src, "race2 (White vs Non-White)"))
            if "if_age" in X and X.if_age.notna().any():
                ia = X.if_age.map(lambda a: np.nan if pd.isna(a) else ("0-29" if a < 30 else "30+"))
                ok = X.if_age.notna()
                rho = spearmanr(X.ff_age_exp[ok], X.if_age[ok]).correlation
                rows.append(agree(ffa.values, ia.values, "0-29", s, level, "InsightFace", "age2 (<30 vs 30+)",
                                  {"spearman_expected_age": float(rho)}))
            if "v1_age_lt25" in X:
                ff25 = (L.ff_age2_cmp25 if level == "frame" else vmaj(L.assign(_a=L.ff_age2_cmp25), "_a").reindex(X.index))
                rows.append(agree(np.asarray(ff25), X.v1_age_lt25.values, "<25", s, level, "AIF-v1",
                                  "age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25)"))
            if "v2_age2" in X:
                rows.append(agree(ffa.values, X.v2_age2.values, "0-29", s, level, "AIF-v2",
                                  "age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented)"))
            if level == "frame":
                xr = np.full(len(L), "others", dtype=object)
                Pq = P.set_index("img_path")
                for c, nm in (("isblack", "black"), ("iswhite", "white"), ("isasian", "asian")):
                    if c in Pq:
                        xr[L.img_path.map(Pq[c]).values == 1] = nm
                txt.append(f"== {s} frame: FairFace race7 x Xu race\n" + pd.crosstab(L.race7, xr).to_string())
            if "v2_age" in X:
                txt.append(f"== {s} {level}: FairFace age2 x AIF-v2 age (rows)\n" +
                           pd.crosstab(ffa.values, X.v2_age.values, dropna=False).to_string())
            if "v1_age5" in L and level == "frame":
                txt.append(f"== {s} frame: FairFace age9 x AIF-v1 age5 (0 child 0-14,1 youth 15-24,2 adult 25-44,3 middle 45-64,4 senior 65+)\n" +
                           pd.crosstab(L.age9, L.v1_age5).reindex(AGE9).dropna(how="all").to_string())
        # per-video consistency of FairFace frame labels
        for lab_, sub in (("all", L), ("real", L[L.label == 0]), ("fake", L[L.label == 1])):
            g = sub.groupby("vid")
            r = {"set": s, "videos": lab_, "n_videos": int(g.ngroups), "frames_per_video_median": float(g.size().median())}
            for c in ("sex", "age2", "race2", "ids8_name", "age4", "race7"):
                fr = g[c].agg(lambda x: x.value_counts().iloc[0] / len(x))
                r[f"{c}_mean_majority_frac"] = 100 * float(fr.mean())
                r[f"{c}_pct_videos_unanimous"] = 100 * float((fr == 1).mean())
                r[f"{c}_pct_videos_ge80"] = 100 * float((fr >= 0.8).mean())
            r["frames_ids_changed_by_video_majority_pct"] = 100 * float((sub.ids8_name != sub.vid_ids8_name).mean())
            cons.append(r)
        print(s, "done", flush=True)
    # ---- FF2-public (per image; manifest labels)
    man = pd.read_csv(f"{R}/release_ff2_public/manifests/public_manifest.csv", low_memory=False,
                      usecols=["rel_path", "label", "person_id", "split", "gender_native", "race_native", "if_age", "p_adult",
                               "use_in_cell_metrics", "src_tier", "family"])
    for s in ["ff2pub_test", "ff2pub_select", "ff2pub_train", "ff2pub_cal"]:
        f = f"{LF}/{s}.csv"
        if not os.path.exists(f):
            continue
        L = pd.read_csv(f, low_memory=False).merge(man, on="rel_path", how="left", validate="one_to_one")
        L["m_sex"] = L.gender_native
        L["m_race2"] = np.where(L.race_native == "White", "White", "NonWhite")
        L["ff_age_exp"] = L[[f"p_age_{a}" for a in AGE9]].values @ MID
        for sub_name, X in (("all", L), ("real", L[L.label == 0]), ("fake", L[L.label == 1]),
                            ("cell-valid", L[L.use_in_cell_metrics.astype(bool)])):
            rows.append(agree(X.sex.values, X.m_sex.values, "Male", s, f"image:{sub_name}", "FF2 manifest", "sex"))
            rows.append(agree(X.race2.values, X.m_race2.values, "White", s, f"image:{sub_name}", "FF2 manifest", "race2 (White vs Non-White)"))
            ok = X.if_age.notna()
            ia = np.where(X.if_age < 30, "0-29", "30+")
            rows.append(agree(X.age2.values[ok], ia[ok], "0-29", s, f"image:{sub_name}", "FF2 manifest if_age (InsightFace)",
                              "age2 (<30 vs 30+)", {"spearman_expected_age": float(spearmanr(X.ff_age_exp[ok], X.if_age[ok]).correlation)}))
            ok = X.p_adult.notna()
            if ok.any():
                rows.append({"set": s, "level": f"image:{sub_name}", "source": "FF2 manifest p_adult", "attribute": "P(adult) vs FairFace age9",
                             "n": int(ok.sum()), "spearman_expected_age": float(spearmanr(X.ff_age_exp[ok], X.p_adult[ok]).correlation),
                             "fairface_pos_pct": 100 * float((X.age9[ok].isin(AGE9[:3])).mean()), "pos_class": "FairFace age <20 (bins<=10-19)"})
        txt.append(f"== {s}: FairFace race7 x manifest race_native (images)\n" + pd.crosstab(L.race7, L.race_native).to_string())
        # per-person consistency (FF2 reals; fakes inherit identity but are generated)
        for lab_, sub in (("real", L[L.label == 0]), ("fake", L[L.label == 1])):
            g = sub.groupby("person_id")
            g = sub[sub.person_id.map(sub.person_id.value_counts()) >= 2].groupby("person_id")
            r = {"set": s, "videos": f"persons ({lab_}, >=2 images)", "n_videos": int(g.ngroups),
                 "frames_per_video_median": float(g.size().median()) if g.ngroups else np.nan}
            for c in ("sex", "age2", "race2", "ids8_name", "age4", "race7"):
                fr = g[c].agg(lambda x: x.value_counts().iloc[0] / len(x))
                r[f"{c}_mean_majority_frac"] = 100 * float(fr.mean())
                r[f"{c}_pct_videos_unanimous"] = 100 * float((fr == 1).mean())
                r[f"{c}_pct_videos_ge80"] = 100 * float((fr >= 0.8).mean())
            cons.append(r)
        print(s, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/agreement.csv", index=False, float_format="%.4f")
    pd.DataFrame(cons).to_csv(f"{OUT}/video_consistency.csv", index=False, float_format="%.3f")
    open(f"{OUT}/agreement_crosstabs.txt", "w").write("\n\n".join(txt) + "\n")


if __name__ == "__main__":
    main()
