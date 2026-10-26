"""FairForensics / GenFace 8-IDS evaluation (secondary, descriptive; decisions/2026-10-26-final-model-spec.md section 4).

IDS = sex (Male/Female) x age (0-29 Youth / 30+ Adult) x race (White / Non-White), FairFace-classifier labels
(labels_fairface/<set>.csv, written by fairface_annotate.py). Order as FairForensics Table I:
  0 MYW, 1 FYW, 2 MAW, 3 FAW, 4 MAN, 5 FAN, 6 MYN, 7 FYN.

Library function (the part other code should import):
  ids8_metrics(y, z, t, ids, cluster, person=None, min_clusters=3) -> dict
    y: 0 real / 1 fake; z: logit (any monotone score); decision fake iff z >= t; ids: IDS name per row (None = unlabeled)
    returns {"pooled": AUC/FPR/TPR, "groups": per-IDS table, "min_support": metrics, "ding_exact": metrics}
  Metric definitions (in %, lower = fairer except es_AUC / AUC / TPR):
    F_FPR  = sum_g |FPR_g - FPR|                       Ding et al. CVPR'26 Eq. 8 (agents/CFES/cfes/metrics.py)
    F_DP   = max_g PR_g - min_g PR_g                   Ding Eq. 9 (pred-fake rate; pred-real term identical)
    es_AUC = AUC / (1 + sum_g |AUC_g - AUC|)            Ding Eq. 10
    fpr_gap = max_g FPR_g - min_g FPR_g, tpr_gap = max_g TPR_g - min_g TPR_g, worst_fpr / worst_tpr
    F_MEO  = max(fpr_gap, tpr_gap)  (FairReL gap form; FairForensics gives no formula)
    F_OAE  = max_g ACC_g - min_g ACC_g   (FairForensics' 4th table metric)
  Eligibility "min_support" (primary, = multiaxis_metrics.py / spec H4): FPR terms over groups with >= 3 real clusters
  (videos; FF2 persons), TPR terms over >= 3 fake clusters, AUC / PR / ACC terms over groups meeting both.
  "ding_exact": Ding's released rule (every group holding both classes; no minimum support).
  A metric needs >= 2 eligible groups, else NaN.

Script (re-scores saved per-image logits; CPU only; no inference):
  OMP_NUM_THREADS=4 ~/.conda/envs/aiface/bin/python fairforensics_ids.py [--B 1000]
Models and thresholds = DEMO_AXES/code/rescore.py registry() (imported, not modified): t05 = deployed rule (CFES-D
with folded FF++-val intercept), A1 = stored 22-Oct rule threshold or the rule re-run on FF++ val logits (Xu labels).
Writes results/fairforensics_ids/{ids8_metrics.csv, ids8_groups.csv, ids8_group_sizes.csv, ids8_seedmean.csv,
thresholds_used.csv, run_meta.json}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
V = f"{R}/agents/DEMO_AXES"
LF = f"{V}/labels_fairface"
OUT = f"{V}/results/fairforensics_ids"
IDS8 = ["MYW", "FYW", "MAW", "FAW", "MAN", "FAN", "MYN", "FYN"]
MIN_CLUSTERS = 3
SPEC_MODELS = ["CFES-D", "arm C", "arm E w1", "arm E w3", "arm E w10", "control (det-only)", "Ding repro (SFD+GDA)",
               "Ding Ori repro (SFD)", "CESel B0"]
EXTRA_MODELS = ["CFES A (rev4)", "CFES-ResNet", "FF2-trained phase 1", "FF2-trained control", "FF2-trained CFES"]


# ------------------------------------------------------------------------------------------------ metrics
def _auc(y, z):
    return float(roc_auc_score(y, z)) if len(np.unique(y)) == 2 else float("nan")


def ids8_groups(y, z, t, ids, cluster, person=None, min_clusters=MIN_CLUSTERS):
    y, z = np.asarray(y), np.asarray(z, dtype=np.float64)
    pred = z >= t
    g = np.asarray(ids, dtype=object)
    cluster = np.asarray(cluster)
    rows = {}
    for name in IDS8:
        m = g == name
        if not m.any():
            continue
        r, f = m & (y == 0), m & (y == 1)
        d = dict(n_real=int(r.sum()), n_fake=int(f.sum()),
                 real_clusters=int(len(set(cluster[r]))), fake_clusters=int(len(set(cluster[f]))),
                 real_persons=int(len(set(person[r]))) if person is not None else None,
                 fake_persons=int(len(set(person[f]))) if person is not None else None,
                 FPR=float(pred[r].mean()) if r.any() else float("nan"),
                 TPR=float(pred[f].mean()) if f.any() else float("nan"),
                 PR=float(pred[m].mean()), ACC=float((pred[m] == (y[m] == 1)).mean()), AUC=_auc(y[m], z[m]))
        d["fpr_elig"] = d["real_clusters"] >= min_clusters
        d["tpr_elig"] = d["fake_clusters"] >= min_clusters
        d["both_elig"] = d["fpr_elig"] and d["tpr_elig"]
        d["both_classes"] = d["n_real"] > 0 and d["n_fake"] > 0
        rows[name] = d
    return rows


def _summ(rows, FPR, TPR, AUC, rule):
    if rule == "min_support":
        ef = [d for d in rows.values() if d["fpr_elig"]]
        et = [d for d in rows.values() if d["tpr_elig"]]
        eb = [d for d in rows.values() if d["both_elig"]]
    else:
        ef = et = eb = [d for d in rows.values() if d["both_classes"]]
    nan = float("nan")
    o = {"n_groups": len(rows), "n_fpr_elig": len(ef), "n_tpr_elig": len(et), "n_both_elig": len(eb)}
    o["F_FPR"] = 100 * sum(abs(d["FPR"] - FPR) for d in ef) if len(ef) >= 2 else nan
    o["fpr_gap"] = 100 * (max(d["FPR"] for d in ef) - min(d["FPR"] for d in ef)) if len(ef) >= 2 else nan
    o["worst_fpr"] = 100 * max(d["FPR"] for d in ef) if len(ef) >= 2 else nan
    o["tpr_gap"] = 100 * (max(d["TPR"] for d in et) - min(d["TPR"] for d in et)) if len(et) >= 2 else nan
    o["worst_tpr"] = 100 * min(d["TPR"] for d in et) if len(et) >= 2 else nan
    o["F_MEO"] = np.nanmax([o["fpr_gap"], o["tpr_gap"]]) if np.isfinite([o["fpr_gap"], o["tpr_gap"]]).any() else nan
    o["F_DP"] = 100 * (max(d["PR"] for d in eb) - min(d["PR"] for d in eb)) if len(eb) >= 2 else nan
    o["F_OAE"] = 100 * (max(d["ACC"] for d in eb) - min(d["ACC"] for d in eb)) if len(eb) >= 2 else nan
    o["es_AUC"] = 100 * AUC / (1 + sum(abs(d["AUC"] - AUC) for d in eb)) if len(eb) >= 2 else nan
    return o


def ids8_metrics(y, z, t, ids, cluster, person=None, min_clusters=MIN_CLUSTERS):
    y, z = np.asarray(y), np.asarray(z, dtype=np.float64)
    pred = z >= t
    FPR, TPR, AUC = float(pred[y == 0].mean()), float(pred[y == 1].mean()), _auc(y, z)
    rows = ids8_groups(y, z, t, ids, cluster, person, min_clusters)
    return {"pooled": {"AUC": 100 * AUC, "FPR": 100 * FPR, "TPR": 100 * TPR, "n": int(len(y)),
                       "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                       "n_unlabeled": int(sum(1 for v in ids if v not in IDS8))},
            "groups": rows, "min_support": _summ(rows, FPR, TPR, AUC, "min_support"),
            "ding_exact": _summ(rows, FPR, TPR, AUC, "ding_exact")}


# ------------------------------------------------------------------------------------------------ labels
SETMAP = {"ffpp_test": "ffpp_test", "dfdc": "dfdc", "celebdf": "celebdf", "dfd": "dfd", "ff2_test": "ff2pub_test"}


def load_labels(s):
    """Rows keyed like the score npz. Video sets: the protocol CSV row order (incl. DFDC duplicates)."""
    L = pd.read_csv(f"{LF}/{SETMAP[s]}.csv", low_memory=False)
    if s == "ff2_test":
        pub = pd.read_csv(f"{R}/data_ding_schema/fairforge2_public/test.csv")       # 87,564 cell-valid public TEST rows
        man = pd.read_csv(f"{R}/release_ff2_public/manifests/public_manifest.csv", low_memory=False,
                          usecols=["rel_path", "person_id", "split"])
        man = man[man.split == "TEST"]
        L = pub[["img_path", "label"]].merge(L, on="img_path", how="left", validate="one_to_one")
        L = L.merge(man[["rel_path", "person_id"]], on="rel_path", how="left", validate="one_to_one")
        assert L.ids8_name.notna().all() and len(L) == len(pub)
        # cluster = person; de novo fakes (GAN / T2I, no person) are their own cluster, as in labels/labels_ff2_test.csv
        L["video"] = np.where(L.person_id.notna(), L.person_id.astype(str), "img:" + L.rel_path.astype(str))
        L["person"] = L.video
        L["ids_main"] = L.ids8_name
        L["ids_frame"] = L.ids8_name
        return L
    D = f"{R}/agents/DINGREPRO_LIN/data"
    src = {"ffpp_test": "ff++/test.csv", "dfdc": "dfdc/test.csv", "celebdf": "celebdf/test.csv", "dfd": "dfd/test.csv"}[s]
    P = pd.read_csv(f"{D}/{src}")[["img_path", "label"]]
    L = P.merge(L, on="img_path", how="left", validate="many_to_one")
    assert L.ids8_name.notna().all() and len(L) == len(P)
    L["video"] = L.vid.astype(str)
    person = None
    lab = f"{V}/labels/labels_{s}.csv"                     # other agent's tables: person ids (same img_path key)
    if os.path.exists(lab):
        o = pd.read_csv(lab, usecols=["img_path", "person"], low_memory=False).drop_duplicates("img_path")
        person = L[["img_path"]].merge(o, on="img_path", how="left").person
    L["person"] = person.astype(str).values if person is not None and person.notna().all() else L.video
    L["person_src"] = "labels/labels_<set>.csv" if person is not None and person.notna().all() else "video id"
    L["ids_main"] = L.vid_ids8_name
    L["ids_frame"] = L.ids8_name
    return L


def group_sizes(L, s):
    rows = []
    for var, col in (("video-majority" if s != "ff2_test" else "image", "ids_main"), ("frame", "ids_frame")):
        if s == "ff2_test" and var == "frame":
            continue
        for g in IDS8:
            m = (L[col] == g).values
            r, f = m & (L.label.values == 0), m & (L.label.values == 1)
            rows.append({"set": s, "labels": var, "ids": IDS8.index(g), "ids_name": g,
                         "frames_real": int(r.sum()), "frames_fake": int(f.sum()),
                         "videos_real": int(L.video[r].nunique()), "videos_fake": int(L.video[f].nunique()),
                         "persons_real": int(L.person[r].nunique()), "persons_fake": int(L.person[f].nunique()),
                         "fpr_elig": int(L.video[r].nunique()) >= MIN_CLUSTERS,
                         "tpr_elig": int(L.video[f].nunique()) >= MIN_CLUSTERS})
    return rows


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--sets", default="ffpp_test,dfdc,celebdf,dfd,ff2_test")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, f"{V}/code")
    import rescore as RS                       # registry / loaders / A1 rule of the multi-axis layer (read-only use)
    import multiaxis_metrics as MM
    t0 = time.time()
    # Spec section 4 models + finished historical rows of the registry. CESel p1 screen arms and CLIPFAIR baselines are
    # excluded on purpose: they are under the running Step-1 gate / baseline selection (validation-only decisions).
    M = [m for m in RS.registry() if m["model"] in SPEC_MODELS + EXTRA_MODELS]
    fo = json.load(open(f"{RS.A}/results/fair_oppoint/fair_oppoint.json"))["thresholds"]
    stored = {("CFES-D", s): fo[f"CFES-D s{s}"]["t"]["A1"] for s in (7, 17, 27)}
    stored[("Ding repro (SFD+GDA)", 7)] = fo["Ding repro"]["t"]["A1"]
    b0 = f"{RS.C}/runs/ffpp_B0_s7/eval/a1.json"
    if os.path.exists(b0):
        stored[("CESel B0", 7)] = json.load(open(b0))["t"]
    thr = []
    for m in M:
        m["t05"], m["A1"], m["a1_src"] = -m["b05"], None, "none (no FF++ val scores)"
        if m["val"]:
            key = (m["model"], m["seed"])
            if key in stored:
                m["A1"], m["a1_src"] = float(stored[key]), "stored"
            else:
                z, y, _ = RS.load_scores(m["val"])
                g = np.load(m["val"], allow_pickle=True)["group"].astype(int)
                j = RS.pick_a1(*RS.val_curves(z, y, g, "eight"))
                if j is not None:
                    m["A1"], m["a1_src"] = float(np.unique(z)[j]), "rule re-run on saved FF++ val logits (Xu labels)"
        thr.append({"model": m["model"], "seed": m["seed"], "t05_logit": m["t05"], "A1_logit": m["A1"], "a1_src": m["a1_src"],
                    "sets": ",".join(sorted(m["sets"]))})
    pd.DataFrame(thr).to_csv(f"{OUT}/thresholds_used.csv", index=False, float_format="%.6g")
    print("thresholds", round(time.time() - t0, 1), "s", flush=True)
    rng = np.random.default_rng(2026)
    rows, grows, sizes, meta = [], [], [], {"sets": {}}
    for s in a.sets.split(","):
        if not os.path.exists(f"{LF}/{SETMAP[s]}.csv"):
            print("labels missing:", s, flush=True)
            continue
        L = load_labels(s)
        sizes += group_sizes(L, s)
        cl = L.video.astype(str).values
        person = L.person.astype(str).values
        u, cl_idx = np.unique(cl, return_inverse=True)
        W = MM.multinomial_weights(len(u), a.B, rng)
        meta["sets"][s] = {"rows": int(len(L)), "clusters": int(len(u)),
                           "person_src": str(L.person_src.iloc[0]) if "person_src" in L else "FF2 person_id"}
        variants = [("main", "ids_main")] + ([("frame (sens)", "ids_frame")] if s != "ff2_test" else [])
        for m in M:
            if s not in m["sets"]:
                continue
            z, y, paths = RS.load_scores(m["sets"][s])
            if s == "ff2_test":
                full = pd.read_csv(f"{R}/data_ding_schema/fairforge2_full/test.csv", usecols=["img_path"]).img_path.values
                if paths is None:
                    assert len(full) == len(z), (m["model"], "ff2 npz without paths has unexpected length")
                    paths = full
                idx = pd.Series(np.arange(len(paths)), index=paths)
                sel = idx.reindex(L.img_path.values).values
                if not np.isfinite(sel).all():
                    print("  skip (FF2-public rows missing in score file)", m["model"], m["seed"], flush=True)
                    continue
                sel = sel.astype(int)
                z, y = z[sel], y[sel]
            else:
                assert paths is not None and len(paths) == len(L) and \
                    all(os.path.basename(p) == os.path.basename(q) for p, q in zip(paths, L.img_path.values)), (m["model"], s)
            assert (y == L.label.values).all(), (m["model"], s, "labels")
            for tn in ("t05", "A1"):
                t = m[tn]
                if t is None:
                    continue
                pred = z >= t
                for var, col in variants:
                    ids = L[col].values.astype(object)
                    res = ids8_metrics(y, z, t, ids, cl, person)
                    names = [n for n in IDS8 if n in res["groups"]]
                    fe = np.array([res["groups"][n]["fpr_elig"] for n in names])
                    te = np.array([res["groups"][n]["tpr_elig"] for n in names])
                    Cc, P = MM.cluster_counts(y, pred, ids, cl_idx, len(u), names)
                    bt = MM.boot_rate_metrics(Cc, P, W, fe, te)
                    base = {"set": s, "model": m["model"], "family": m["family"], "seed": m["seed"], "threshold": tn,
                            "t": t, "a1_src": m["a1_src"], "labels": var, **res["pooled"]}
                    for rule in ("min_support", "ding_exact"):
                        row = dict(base, rule=rule, **res[rule])
                        if rule == "min_support":
                            for k in ("F_FPR", "fpr_gap", "tpr_gap"):
                                v = bt[k]
                                row[f"{k}_lo"], row[f"{k}_hi"] = (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) \
                                    if np.isfinite(v).any() else (np.nan, np.nan)
                        rows.append(row)
                    for n in names:
                        d = res["groups"][n]
                        grows.append({"set": s, "model": m["model"], "seed": m["seed"], "threshold": tn, "labels": var,
                                      "ids": IDS8.index(n), "ids_name": n, **{k: (100 * v if k in ("FPR", "TPR", "PR", "ACC", "AUC") else v)
                                                                             for k, v in d.items()}})
        print(s, "done", round(time.time() - t0, 1), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ids8_metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(grows).to_csv(f"{OUT}/ids8_groups.csv", index=False, float_format="%.4f")
    pd.DataFrame(sizes).to_csv(f"{OUT}/ids8_group_sizes.csv", index=False)
    keys = ["set", "model", "threshold", "labels", "rule"]
    vals = ["AUC", "FPR", "TPR", "F_FPR", "F_DP", "es_AUC", "fpr_gap", "tpr_gap", "worst_fpr", "worst_tpr", "F_MEO",
            "F_OAE", "n_fpr_elig", "n_tpr_elig", "n_both_elig"]
    g = df.groupby(keys, sort=False)
    sm = g[vals].mean()
    sm.columns = [f"{c}" for c in sm.columns]
    sd = g[["AUC", "TPR", "F_FPR", "fpr_gap", "F_DP", "es_AUC"]].std(ddof=1)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = pd.concat([g.size().rename("n_seeds"), sm, sd], axis=1).reset_index()
    out.to_csv(f"{OUT}/ids8_seedmean.csv", index=False, float_format="%.3f")
    meta.update(B=a.B, rng=2026, min_clusters=MIN_CLUSTERS, runtime_s=round(time.time() - t0, 1),
                models=[{k: m[k] for k in ("model", "family", "seed", "run", "a1_src", "A1", "t05")} for m in M])
    json.dump(meta, open(f"{OUT}/run_meta.json", "w"), indent=1, default=str)
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
