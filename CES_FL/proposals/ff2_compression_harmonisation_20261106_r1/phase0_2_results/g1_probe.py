# Phase 2, gate G1 (pre-declared): logistic regression on the diagnosis' compression/sharpness statistics
# (feat_stats.feats, feature list exactly as analyse_stats.py), real vs fake WITHIN each source, fit on FF2-H TRAIN
# (all reals + equal-size method-stratified fake sample), AUC on FF2-H VAL (ces_fl val manifest = SELECT).
# PASS iff VAL AUC <= 0.60 in BOTH sources. Context: same probe on FF2 v1.0 crops (same rows), and the
# FFHQ-vs-FairFace real-source probe on both versions. TEST is never read. No model scoring of images.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff2h_common as H
H.single_thread_env()
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106")
from feat_stats import feats  # the diagnosis extractor, unchanged
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

THR = 0.60
SC = ['bytes', 'log_lap', 'noise', 'hf_y', 'hf_c', 'block8', 'sat', 'val', 'y_std', 'dark', 'flat8']  # analyse_stats.py
EXTRA = ['chroma_ratio', 'cr_mean', 'cb_mean', 'edge_flat']  # computed by feats() but not in the diagnosis FEAT list


def fsets(cols):
    primary = [c for c in cols if c in SC or c.startswith(('ry', 'rc'))]
    return {"diag_FEAT": primary, "extended": primary + [c for c in EXTRA if c in cols]}


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def boot_auc(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)]


if __name__ == "__main__":
    NW = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    inv = pd.read_pickle(H.WORK + "/_inventory_PRIVATE.pkl")
    src_of = dict(zip(inv.abs_path, inv.src_dataset))
    tr = pd.read_csv("/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/train.csv", usecols=["image", "label", "method", "source_ids"])
    va = pd.read_csv("/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/val.csv", usecols=["image", "label", "method", "source_ids"])
    for d in (tr, va):
        d["src"] = d.image.map(src_of)
        assert d.src.notna().all()
    S = {"source_ids_prefix_vs_src": {f"{a}|{b}": int(n) for (a, b), n in pd.concat([tr, va]).assign(
        p=lambda d: d.source_ids.astype(str).str.split(":").str[0]).groupby(["p", "src"]).size().items()}}
    # TRAIN sample: within each source, all reals + an equal number of fakes stratified by method (proportional)
    parts = []
    for s, g in tr.groupby("src"):
        r = g[g.label == 0]; f = g[g.label == 1]; n = len(r)
        frac = n / len(f)
        fs = f.groupby("method", group_keys=False).apply(lambda x: x.sample(int(round(len(x) * frac)), random_state=7))
        parts += [r, fs]
    trs = pd.concat(parts).assign(split="train")
    vas = va.assign(split="val")
    s = pd.concat([trs, vas]).reset_index(drop=True)
    s["new_image"] = s.image.map(H.new_abs)
    S["n_train_by_src_label"] = {f"{a}|{b}": int(n) for (a, b), n in trs.groupby(["src", "label"]).size().items()}
    S["n_val_by_src_label"] = {f"{a}|{b}": int(n) for (a, b), n in vas.groupby(["src", "label"]).size().items()}
    paths = s.new_image.tolist() + s.image.tolist()
    with Pool(NW) as p:
        F = p.map(feats, paths, chunksize=64)
    F = pd.DataFrame(F)
    S["feature_errors"] = int(F.get("err", pd.Series(dtype=object)).notna().sum())
    fh = pd.concat([s, F.iloc[:len(s)].reset_index(drop=True)], axis=1)
    fo = pd.concat([s, F.iloc[len(s):].reset_index(drop=True)], axis=1)
    pd.to_pickle({"ff2h": fh, "ff2_v10": fo}, H.WORK + "/_g1_features_PRIVATE.pkl")
    FS = fsets(list(F.columns))
    S["n_features"] = {k: len(v) for k, v in FS.items()}
    S["jpeg_qtable_by_version"] = {"ff2h": fh.q_hash.value_counts().to_dict(), "ff2_v10": fo.q_hash.value_counts().to_dict()}
    res = {}
    for ver, f in (("ff2h", fh), ("ff2_v10", fo)):
        f = f[f.get("err", pd.Series(index=f.index, dtype=object)).isna()]
        T, Vd = f[f.split == "train"], f[f.split == "val"]
        for fsn, FEAT in FS.items():
            key = f"{ver}/{fsn}"; res[key] = {}
            for src in ["FFHQ", "FairFace"]:
                t, v = T[T.src == src], Vd[Vd.src == src]
                m = lr().fit(t[FEAT], t.label)
                sc = m.predict_proba(v[FEAT])[:, 1]
                auc = float(roc_auc_score(v.label, sc))
                res[key][f"real_vs_fake_within_{src}"] = dict(val_auc=round(auc, 4), ci95=boot_auc(v.label, sc),
                                                              n_train=int(len(t)), n_val_real=int((v.label == 0).sum()),
                                                              n_val_fake=int((v.label == 1).sum()))
                # per-method AUC (each method's VAL fakes vs all VAL reals of this source), for context
                res[key][f"per_method_val_auc_{src}"] = {mm: round(float(roc_auc_score(
                    np.r_[np.zeros((v.label == 0).sum()), np.ones(len(g))],
                    np.r_[sc[(v.label == 0).values], sc[v.index.get_indexer(g.index)]])), 3)
                    for mm, g in v[v.label == 1].groupby("method") if len(g) >= 20}
            r_t, r_v = T[T.label == 0], Vd[Vd.label == 0]
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(
                r_t[FEAT], (r_t.src == "FFHQ").astype(int))
            sc = m.predict_proba(r_v[FEAT])[:, 1]
            res[key]["real_source_FFHQ_vs_FairFace"] = dict(val_auc=round(float(roc_auc_score((r_v.src == "FFHQ").astype(int), sc)), 4),
                                                            ci95=boot_auc((r_v.src == "FFHQ").astype(int), sc))
            # pooled real-vs-fake (both sources), context only
            m = lr().fit(T[FEAT], T.label); sc = m.predict_proba(Vd[FEAT])[:, 1]
            res[key]["real_vs_fake_pooled"] = dict(val_auc=round(float(roc_auc_score(Vd.label, sc)), 4))
    S["results"] = res
    P = res["ff2h/diag_FEAT"]
    a1, a2 = P["real_vs_fake_within_FFHQ"]["val_auc"], P["real_vs_fake_within_FairFace"]["val_auc"]
    S["G1"] = dict(threshold=THR, auc_FFHQ=a1, auc_FairFace=a2, verdict="PASS" if (a1 <= THR and a2 <= THR) else "FAIL",
                   extended_featureset_verdict="PASS" if max(res["ff2h/extended"]["real_vs_fake_within_FFHQ"]["val_auc"],
                                                            res["ff2h/extended"]["real_vs_fake_within_FairFace"]["val_auc"]) <= THR else "FAIL")
    # medians of key statistics on VAL by version x source x label (aggregates)
    KS = ["bytes", "log_lap", "noise", "hf_y", "hf_c", "block8"]
    S["val_medians"] = {ver: {f"{a}|{b}": g[KS].median().round(3).to_dict() for (a, b), g in f[f.split == "val"].groupby(["src", "label"])}
                        for ver, f in (("ff2h", fh), ("ff2_v10", fo))}
    json.dump(S, open(H.WORK + "/g1_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))
