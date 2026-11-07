# Phase 0: inventory (path existence only; no TEST pixels opened) + FairFace TRAIN-real interocular median.
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff2h_common as H
H.single_thread_env()
import numpy as np, pandas as pd
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

APP = None


def _init():
    global APP, V, C, cv2
    APP = H.make_app()
    import canon_v2 as V, canonicalize_ff2 as C, cv2 as _cv2
    cv2 = _cv2


def iod_one(p):
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        return dict(err="decode")
    f, _, _ = V.detect_v2(APP, bgr)
    if not f:
        return dict(err="noface", h=bgr.shape[0], w=bgr.shape[1])
    return dict(iod=C.face_scale(f[0]["kps"]), n_faces=len(f), det_score=f[0]["det_score"], h=bgr.shape[0], w=bgr.shape[1])


if __name__ == "__main__":
    NW = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    t0 = time.time()
    m = pd.read_pickle(H.WORK + "/_inventory_stage0_PRIVATE.pkl")
    with ThreadPool(32) as tp:
        m["master_exists"] = tp.map(lambda p: isinstance(p, str) and os.path.isfile(p), m.master_path.tolist(), chunksize=256)
        m["orig_exists"] = tp.map(os.path.isfile, m.abs_path.tolist(), chunksize=256)
    S = {"n_rows": int(len(m)), "t_exist_sec": round(time.time() - t0, 1)}
    m["src"] = m.src_dataset; m["lab"] = m.label.map({0: "real", 1: "fake"})
    cov = m.groupby(["split", "src", "lab", "method"]).agg(n=("master_exists", "size"), master_found=("master_exists", "sum"),
                                                           orig_found=("orig_exists", "sum"))
    cov.to_csv(H.WORK + "/phase0_coverage_by_split_source_label_method.csv")
    S["coverage_by_split"] = m.groupby("split").agg(n=("master_exists", "size"), master_found=("master_exists", "sum")).astype(int).to_dict("index")
    S["coverage_by_split_src_label"] = {"|".join(map(str, k)): v for k, v in m.groupby(["split", "src", "lab"]).agg(
        n=("master_exists", "size"), master_found=("master_exists", "sum")).astype(int).to_dict("index").items()}
    S["master_kind_counts"] = m.master_kind.value_counts().to_dict()
    S["missing_master_by_method_split"] = {"|".join(k): int(v) for k, v in m[~m.master_exists].groupby(["method", "split"]).size().items()}
    S["ref_sha_ok_mean_by_origin"] = m.groupby("origin").ref_sha_ok.mean().round(5).to_dict()
    S["ff2v1_fallback_generation_manifest_by_split"] = m[m.master_kind == "ff2v1_generation_manifest"].split.value_counts().to_dict()
    # FairFace TRAIN reals: interocular distance on the original 448 px files (SCRFD, v2 canvas detection, largest face)
    ff = m[(m.label == 0) & (m.src == "FairFace") & (m.split == "TRAIN")]
    with Pool(NW, initializer=_init) as p:
        R = p.map(iod_one, ff.master_path.tolist(), chunksize=32)
    r = pd.DataFrame(R)
    iod = r.iod.dropna()
    rm = pd.read_csv(H.RM, usecols=["path", "face_iod_src_px"], low_memory=False).set_index("path").face_iod_src_px
    stored = ff.abs_path.map(rm).values
    S["fairface_train_real_iod"] = dict(
        n=int(len(ff)), n_detected=int(iod.size), n_err=r.get("err", pd.Series(dtype=object)).value_counts().to_dict(),
        median=float(iod.median()), mean=float(iod.mean()), p05=float(iod.quantile(.05)), p25=float(iod.quantile(.25)),
        p75=float(iod.quantile(.75)), p95=float(iod.quantile(.95)),
        src_hw=r.groupby(["h", "w"]).size().sort_values(ascending=False).head(3).to_dict().__repr__(),
        stored_face_iod_src_px_median=float(np.nanmedian(stored)),
        median_abs_diff_vs_stored=float(np.nanmedian(np.abs(r.iod.values - stored))))
    S["IOD_TARGET"] = round(float(iod.median()), 4)
    m.to_pickle(H.WORK + "/_inventory_PRIVATE.pkl")
    S["t_total_sec"] = round(time.time() - t0, 1)
    json.dump(S, open(H.WORK + "/phase0_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))
