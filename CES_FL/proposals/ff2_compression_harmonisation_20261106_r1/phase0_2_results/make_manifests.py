# Phase 1b: FF2-H manifests with the SAME rows/labels/columns/order as the FF2 originals; only the image path column
# (and sha256 where present) changes. Verifies identity of every non-path column. Writes BUILD_REPORT.txt (+ .json).
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff2h_common as H
import numpy as np, pandas as pd
from multiprocessing.pool import ThreadPool

R = H.R
DING_O = f"{R}/data_ding_schema/fairforge2_public"
DING_N = f"{R}/data_ding_schema/fairforge2_h"
CES_O = "/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2"
CES_N = "/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2h"
for d in (DING_N, CES_N, H.OUT_ROOT + "manifests"):
    os.makedirs(d, exist_ok=True)

inv = pd.read_pickle(H.WORK + "/_inventory_PRIVATE.pkl")
rec = pd.read_csv(H.WORK + "/_build_records_PRIVATE.csv", low_memory=False)
rec = rec.sort_values("out_sha256", na_position="first").drop_duplicates("idx", keep="last").set_index("idx").sort_index()
S = {"n_manifest_rows": int(len(inv)), "n_records": int(len(rec))}
b = inv.join(rec[["out_sha256", "err", "master_sha256", "iod_master", "scale", "interp", "lm_src_master", "lm_src_align",
                  "align_scale", "out_bytes"]])
b["built"] = b.out_sha256.notna()
b["new_abs"] = H.OUT_ROOT + b.rel_path
with ThreadPool(32) as tp:
    b["new_exists"] = tp.map(os.path.isfile, b.new_abs.tolist(), chunksize=256)
b["ok"] = b.built & b.new_exists
fail = b[~b.ok]
S["n_built_ok"] = int(b.ok.sum()); S["n_failed"] = int(len(fail))
S["failures_by_split_method_reason"] = {f"{s}|{m}|{e}": int(n) for (s, m, e), n in
                                        fail.assign(e=fail.err.fillna("not_built_or_missing_file")).groupby(["split", "method", "e"]).size().items()}
S["built_by_split"] = b.groupby("split").ok.agg(["size", "sum"]).astype(int).rename(columns={"size": "rows", "sum": "built"}).to_dict("index")
S["built_by_split_src_label"] = {"|".join(map(str, k)): v for k, v in b.groupby(["split", "src_dataset", "label"]).ok.agg(
    ["size", "sum"]).astype(int).rename(columns={"size": "rows", "sum": "built"}).to_dict("index").items()}
S["built_by_method"] = b.groupby("method").ok.agg(["size", "sum"]).astype(int).rename(columns={"size": "rows", "sum": "built"}).to_dict("index")
sm = b.stored_master_sha256.notna() & b.master_sha256.notna()
S["master_sha256_vs_stored"] = dict(n_compared=int(sm.sum()), n_match=int((b.master_sha256[sm] == b.stored_master_sha256[sm]).sum()),
                                    mismatch_by_kind=b[sm & (b.master_sha256 != b.stored_master_sha256)].master_kind.value_counts().to_dict())
S["landmark_source_master"] = b.lm_src_master.value_counts().to_dict()
S["landmark_source_align"] = b.lm_src_align.value_counts().to_dict()
S["interp"] = b.interp.value_counts().to_dict()
S["dup_out_sha256"] = int(b.out_sha256.dropna().duplicated().sum())
S["dup_out_sha256_v10"] = int(b.sha256.duplicated().sum())
q = lambda s: {k: round(float(s.quantile(v)), 4) for k, v in [("p01", .01), ("p50", .5), ("p99", .99)]}
grp = b.src_dataset + "|" + b.label.map({0: "real", 1: "fake"})
S["scale_by_src_label"] = {g: q(x.scale.dropna()) for g, x in b.groupby(grp)}
S["align_scale_by_src_label"] = {g: q(x.align_scale.dropna()) for g, x in b.groupby(grp)}
S["upsampled_frac_by_src_label"] = {g: round(float((x.scale.dropna() > 1).mean()), 4) for g, x in b.groupby(grp)}
S["out_bytes_median_by_src_label_ff2h"] = {g: float(x.out_bytes.median()) for g, x in b.groupby(grp)}
S["total_out_GB"] = round(float(b.out_bytes.sum()) / 1e9, 3)

# per-item build record (aggregate-safe internal file; contains relative paths, kept in work dir, private)
b[["rel_path", "split", "src_dataset", "label", "method", "master_kind", "master_sha256", "iod_master", "scale", "interp",
   "lm_src_master", "lm_src_align", "align_scale", "out_sha256", "err"]].to_csv(H.WORK + "/_ff2h_item_records_PRIVATE.csv", index=False)

# ---------------------------------------------------------------- (a) public manifest: rel_path identical (same layout), sha256 -> FF2-H
pub = pd.read_csv(H.PUB, low_memory=False, dtype=str, keep_default_na=False)
assert len(pub) == len(b) and (pub.rel_path.values == b.rel_path.values).all()
newpub = pub.copy()
newpub["sha256"] = b.out_sha256.fillna("").values
newpub.to_csv(f"{DING_N}/public_manifest.csv", index=False)
shutil.copyfile(f"{DING_N}/public_manifest.csv", H.OUT_ROOT + "manifests/public_manifest.csv")
CHK = {}


def verify(name, o, n, pathcol, shacol=None):
    cols = [c for c in o.columns if c not in (pathcol, shacol)]
    same_cols = list(o.columns) == list(n.columns)
    same_rows = len(o) == len(n)
    same_vals = bool((o[cols].values == n[cols].values).all()) if same_rows and same_cols else False
    CHK[name] = dict(rows_orig=int(len(o)), rows_new=int(len(n)), same_columns_and_order=same_cols,
                     non_path_columns_identical=same_vals, n_non_path_columns=len(cols))
    assert same_cols and same_rows and same_vals, name


verify("public_manifest", pub, newpub, "rel_path", "sha256")
CHK["public_manifest"]["rel_path_identical"] = bool((pub.rel_path == newpub.rel_path).all())

old2new = dict(zip(b.abs_path, b.new_abs))
okset = set(b.new_abs[b.ok])


def remap(o, col, name):
    n = o.copy()
    miss = ~o[col].isin(old2new.keys())
    assert not miss.any(), (name, int(miss.sum()))
    n[col] = o[col].map(old2new)
    CHK.setdefault(name, {})
    return n


# ---------------------------------------------------------------- (b) Ding-schema CSVs
for k in ["train", "realval", "fakeval", "cal", "test", "provenance"]:
    o = pd.read_csv(f"{DING_O}/{k}.csv", dtype=str, keep_default_na=False)
    n = remap(o, "img_path", f"ding/{k}")
    n.to_csv(f"{DING_N}/{k}.csv", index=False)
    verify(f"ding/{k}", o, n, "img_path")
    CHK[f"ding/{k}"]["n_rows_pointing_to_failed_items"] = int((~n.img_path.isin(okset)).sum())
# ---------------------------------------------------------------- (c) CES-FL pipeline manifests
for k in ["train", "val"]:
    o = pd.read_csv(f"{CES_O}/{k}.csv", dtype=str, keep_default_na=False)
    n = remap(o, "image", f"ces_fl/{k}")
    n.to_csv(f"{CES_N}/{k}.csv", index=False)
    verify(f"ces_fl/{k}", o, n, "image")
    CHK[f"ces_fl/{k}"]["n_rows_pointing_to_failed_items"] = int((~n.image.isin(okset)).sum())
# cross-checks: split / person / group columns of the public manifest agree with the Ding provenance rows
pv = pd.read_csv(f"{DING_N}/provenance.csv", dtype=str, keep_default_na=False)
pb = b.set_index("new_abs")
CHK["provenance_vs_public"] = dict(
    split_agree=bool((pv.split.values == pb.loc[pv.img_path, "split"].values).all()),
    person_agree=bool((pv.person_id.values == pb.loc[pv.img_path, "person_id"].astype(str).values).all()))
S["manifest_checks"] = CHK
S["outputs"] = dict(public_manifest=[f"{DING_N}/public_manifest.csv", H.OUT_ROOT + "manifests/public_manifest.csv"],
                    ding=DING_N, ces_fl=CES_N, images=H.OUT_ROOT)
json.dump(S, open(H.WORK + "/build_summary.json", "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
